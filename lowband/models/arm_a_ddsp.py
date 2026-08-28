"""Arm A — DDSP / Harmonic+Noise synthesizer (task spec §6).

The network predicts control parameters (spectral envelope in mel, sub-band
periodicity).  F0 comes from the input signal (YIN) or an oracle path, NEVER
from the target (information leak, §6.5).

Output mode (default): direct magnitude-spectrum synthesis (no waveform).
Waveform synthesis path is implemented but OFF by default (config switch).

§6.1 anti-aliasing, §6.2 phase precision, §6.3 sub-band periodicity, §6.4
control upsampling, §6.5 F0 paths, §6.6 noise fusion — all handled here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import LowBandReconstructor
from ..dsp.stft import StftConfig as _StftConfig
from ..dsp.stft import causal_stft as _causal_stft
from ..dsp.stft import stft as _stft_fn, frame_step as _frame_step
from ..dsp import ddsp as ddsp_mod
from ..dsp import f0 as f0_mod


class ControlNet(nn.Module):
    """Small conv-recurrent net: input magnitude -> mel envelope + periodicity.

    ~18K params.  Input is (B, F_bins, N_frames); output two heads.
    """

    def __init__(self, n_bins: int = 65, n_mel: int = 16, n_bands: int = 12,
                 hidden: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(16, 32, (3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(32, 32, (3, 1), padding=(1, 0))
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.head_env = nn.Linear(hidden, n_mel)
        self.head_period = nn.Linear(hidden, n_bands)
        self.n_bins = n_bins

    def forward(self, mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # mag: (B, F, N) -> (B, 1, F, N) conv over freq only (kernel (3,1))
        x = mag.unsqueeze(1)         # (B, 1, F, N)
        x2 = F.relu(self.conv1(x))     # (B, 16, F, N)
        x2 = F.relu(self.conv2(x2))    # (B, 32, F, N)
        x2 = F.relu(self.conv3(x2))    # (B, 32, F, N)
        # Pool over frequency to get a per-frame feature
        feat = x2.mean(dim=2)          # (B, 32, N)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, N, 32)
        out, _ = self.gru(feat)        # (B, N, hidden)
        env_mel = self.head_env(out)   # (B, N, n_mel)
        period = torch.sigmoid(self.head_period(out))  # (B, N, n_bands) in [0,1]
        return env_mel, period


    def forward_step(self, mag_frame: torch.Tensor, gru_h: torch.Tensor | None):
        """Streaming: process one STFT frame, carry GRU state (§5.3)."""
        # mag_frame: (B, F) -> (B, 1, F, 1)
        x = mag_frame.unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        x2 = F.relu(self.conv1(x))     # (B, 16, F, 1)
        x2 = F.relu(self.conv2(x2))    # (B, 32, F, 1)
        x2 = F.relu(self.conv3(x2))    # (B, 32, F, 1)
        feat = x2.mean(dim=2)          # (B, 32, 1)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, 1, 32)
        out, h = self.gru(feat, gru_h)  # (B, 1, hidden), (1, B, hidden)
        env_mel = self.head_env(out)    # (B, 1, n_mel)
        period = torch.sigmoid(self.head_period(out))  # (B, 1, n_bands)
        return env_mel, period, h.detach() if h is not None else None


class ArmA_DDSP(LowBandReconstructor):
    """DDSP harmonic+noise reconstructor."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.sample_rate = cfg["sample_rate"]            # 4000
        self.stft_cfg = _StftConfig(
            n_fft=cfg.get("stft_n_fft", 128),
            hop=cfg.get("stft_hop", 32),
            win=cfg.get("stft_win", 128),
            window=cfg.get("stft_window", "hann"),
        )
        self.n_bins = self.stft_cfg.num_bins             # 65
        self.nyquist = self.sample_rate / 2               # 2000
        self.bin_width = self.sample_rate / self.stft_cfg.n_fft  # 31.25 Hz/bin

        self.n_mel = cfg.get("n_mel", 16)
        self.n_bands = cfg.get("n_bands", 12)
        self.max_harm = cfg.get("max_harm", 32)  # F0 min ~ 2000/32 ≈ 62.5 Hz
        self.f0_mode = cfg.get("f0_mode", "estimated")  # "estimated" | "oracle"
        self.f0_min = cfg.get("f0_min", 50.0)
        self.f0_max = cfg.get("f0_max", 400.0)
        self.waveform_synth = cfg.get("waveform_synth", False)  # off by default

        self.control_net = ControlNet(self.n_bins, self.n_mel, self.n_bands,
                                      hidden=cfg.get("ctrl_hidden", 48))

        # Precompute mel-to-linear conversion matrix (mel pseudo-inverse).
        fb = ddsp_mod.mel_filterbank(self.n_mel, self.stft_cfg.n_fft,
                                      self.sample_rate, f_min=0.0,
                                      f_max=self.nyquist)
        # Register mel filterbank and its pseudo-inverse
        self.register_buffer("mel_fb", fb)  # (n_mel, F)
        mel_fb_inv = torch.linalg.pinv(fb)   # (F, n_mel)
        self.register_buffer("mel_fb_inv", mel_fb_inv)

        # Bin frequencies
        bin_freqs = torch.arange(self.n_bins) * self.bin_width
        self.register_buffer("bin_freqs", bin_freqs)

        # Harmonic smearing width (bins) — Hann main lobe ~ 4 bins at n_fft=128
        self.smear_sigma = cfg.get("smear_sigma", 2.0)

    def _get_f0(self, x: torch.Tensor, cond: dict | None) -> torch.Tensor:
        """Return per-frame F0 (B, N_frames).

        §6.5: estimated from INPUT, never target.
        """
        if self.f0_mode == "oracle" and cond is not None and "f0" in cond:
            return cond["f0"]  # (B, N_frames) — provided by dataset/oracle
        # Estimated path: YIN from input
        f0, _ = f0_mod.yin_f0(x, self.sample_rate, frame_len=self.stft_cfg.win,
                              f0_min=self.f0_min, f0_max=self.f0_max)
        return f0  # (B, N_frames)

    def _mel_to_linear(self, env_mel: torch.Tensor) -> torch.Tensor:
        """env_mel: (B, N, n_mel) -> env_lin: (B, N, F)."""
        # Normalize mel envelope to non-negative
        env_mel = F.softplus(env_mel)
        env_lin = env_mel @ self.mel_fb_inv.T  # (B, N, n_mel) @ (n_mel, F)
        return env_lin.clamp_min(0.0)

    def _harmonic_amps(self, env_lin: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample spectral envelope at harmonic frequencies.

        Args:
            env_lin: (B, N, F) linear spectral envelope.
            f0: (B, N) fundamental frequency per frame.

        Returns:
            amps: (B, N, K) harmonic amplitudes.
            mask: (B, N, K) anti-alias mask.
        """
        B, N, Fb = env_lin.shape
        K = self.max_harm
        device = env_lin.device
        # Harmonic frequencies: (B, N, K)
        k = torch.arange(1, K + 1, device=device, dtype=env_lin.dtype)
        harm_freq = f0.unsqueeze(-1) * k  # (B, N, K)
        # Convert to fractional bin index
        bin_idx = harm_freq / self.bin_width  # (B, N, K)
        # Linear interpolation of env_lin at fractional bin_idx
        bin_idx_clamped = bin_idx.clamp(0.0, float(Fb - 1))
        idx_lo = bin_idx_clamped.floor().long()
        idx_hi = (idx_lo + 1).clamp(max=Fb - 1)
        frac = bin_idx_clamped - idx_lo.float()
        # Gather: env_lin is (B, N, F); need (B, N, K) by gathering along F
        amps_lo = torch.gather(env_lin, -1, idx_lo)
        amps_hi = torch.gather(env_lin, -1, idx_hi)
        amps = amps_lo * (1 - frac) + amps_hi * frac
        # Anti-alias mask (§6.1)
        mask = (harm_freq < (self.nyquist - 1.0)).float()
        return amps, mask

    def _harmonic_mag(self, amps: torch.Tensor, f0: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """Smeared harmonic comb -> magnitude spectrum (B, F, N).

        Each harmonic contributes a Gaussian centered at k*f0 with width sigma.
        """
        B, N, K = amps.shape
        device = amps.device
        freqs = self.bin_freqs.view(1, 1, -1, 1)  # (1, 1, F, 1)
        harm_freq = f0.unsqueeze(-1) * torch.arange(1, K + 1, device=device,
                                                    dtype=amps.dtype)  # (B, N, K)
        harm_freq = harm_freq.unsqueeze(2)  # (B, N, 1, K)
        dist = freqs - harm_freq  # (1, 1, F, K) broadcast
        sigma = self.smear_sigma * self.bin_width
        gauss = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))  # (B, N, F, K)
        amps_masked = (amps * mask).unsqueeze(2)  # (B, N, 1, K)
        harm_mag = (gauss * amps_masked).sum(dim=-1)  # (B, N, F)
        return harm_mag.permute(0, 2, 1).contiguous()  # (B, F, N)

    def _periodicity_spec(self, period: torch.Tensor) -> torch.Tensor:
        """Interpolate per-band periodicity to per-bin (B, F, N)."""
        # period: (B, N, n_bands) -> interpolate n_bands -> F along last dim
        period_f = F.interpolate(period, size=self.n_bins, mode="linear",
                                 align_corners=False)  # (B, N, F)
        return period_f.permute(0, 2, 1).contiguous()  # (B, F, N)

    def _noise_mag(self, env_lin: torch.Tensor) -> torch.Tensor:
        """Noise spectral envelope = same as linear envelope (B, F, N)."""
        return env_lin.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor, cond: dict | None = None) -> dict:
        B, T = x.shape
        spec, in_mag = _causal_stft(x, self.stft_cfg)  # (B, F, N)
        N = in_mag.shape[-1]

        # F0 (B, N)
        f0 = self._get_f0(x, cond)  # (B, N_frames)
        # Align F0 frame count to STFT frame count
        if f0.shape[-1] != N:
            f0 = F.interpolate(f0.unsqueeze(1), size=N, mode="linear",
                               align_corners=False).squeeze(1)
        f0 = f0.clamp(self.f0_min, self.f0_max)

        # Control network
        env_mel, period = self.control_net(in_mag)  # (B, N, n_mel), (B, N, n_bands)
        env_lin = self._mel_to_linear(env_mel)       # (B, N, F)

        amps, harm_mask = self._harmonic_amps(env_lin, f0)  # (B, N, K), (B, N, K)
        harm_mag = self._harmonic_mag(amps, f0, harm_mask)   # (B, F, N)
        noise_mag = self._noise_mag(env_lin)                  # (B, F, N)
        period_spec = self._periodicity_spec(period)         # (B, F, N)

        mag = period_spec * harm_mag + (1.0 - period_spec) * noise_mag

        result = {
            "mag": mag,
            "aux": {
                "f0": f0.detach(),
                "periodicity": period.detach(),
                "env_mel": env_mel.detach(),
                "harmonic_amps": amps.detach(),
                "harm_mask": harm_mask.detach(),
            },
        }

        if self.waveform_synth:
            wav = self._synth_waveform(f0, amps, harm_mask, env_lin, period, T)
            result["wav"] = wav
        return result

    def _synth_waveform(self, f0, amps, harm_mask, env_lin, period, T):
        """Full waveform synthesis (off by default)."""
        # Upsample control to sample rate
        f0_per_sample = ddsp_mod.upsample_control(f0, T)  # (B, T)
        amps_per_sample = ddsp_mod.upsample_control(
            amps.permute(0, 2, 1), T).permute(0, 2, 1)  # (B, K, T)
        mask_per_sample = harm_mask.permute(0, 2, 1)  # (B, K, T) constant per frame expanded
        mask_per_sample = ddsp_mod.upsample_control(mask_per_sample, T).permute(0, 2, 1)

        phase = ddsp_mod.accumulate_phase(f0_per_sample, T, self.sample_rate)
        harm_wav = ddsp_mod.harmonic_synth(phase, amps_per_sample, mask_per_sample)

        noise_mags = env_lin.permute(0, 2, 1)  # (B, F, N)
        noise_wav = ddsp_mod.noise_synth(noise_mags, self.stft_cfg.n_fft,
                                          self.stft_cfg.hop, length=T)
        period_spec = self._periodicity_spec(period)
        period_per_sample = ddsp_mod.upsample_control(
            period_spec.permute(0, 2, 1), T).permute(0, 2, 1)  # (B, F, T)
        # Mix in frequency domain (simplified: weight in time domain)
        gain_h = period_per_sample.mean(dim=1)  # (B, T)
        wav = gain_h.unsqueeze(-1) * harm_wav + (1 - gain_h).unsqueeze(-1) * noise_wav
        return wav

    # --- streaming interface (§5.3 must be numerically equivalent) ---------
    def stream_init(self, batch_size: int) -> dict:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        return {
            "stft_tail": torch.zeros(batch_size, win - hop),
            "gru_h": None,
            "f0_buf": torch.zeros(batch_size, win),  # rolling window for F0
            "f0_override": None,  # set externally for equivalence test
        }

    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        B = x_frame.shape[0]
        device = x_frame.device

        # Update F0 rolling buffer
        f0_buf = torch.cat([state["f0_buf"][:, hop:], x_frame], dim=1)
        # STFT frame
        mag_frame, new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])

        # F0: oracle from state (set by caller) or estimated per-frame
        if state.get("f0_override") is not None:
            f0_frame = state["f0_override"].view(B)  # (B,) — for equivalence test
        elif self.f0_mode == "oracle":
            f0_frame = torch.full((B,), 150.0, device=device)  # fallback
        else:
            f0_frame, _ = f0_mod._yin_frame(
                f0_buf, self.sample_rate,
                tau_min=max(1, int(self.sample_rate / self.f0_max) - 1),
                tau_max=int(self.sample_rate / self.f0_min) + 2,
                threshold=0.1,
            )
        f0_frame = f0_frame.clamp(self.f0_min, self.f0_max).unsqueeze(-1)  # (B, 1)

        # Control network with GRU state carried (§5.3 equivalence)
        env_mel, period, gru_h = self.control_net.forward_step(
            mag_frame, state["gru_h"])
        env_lin = self._mel_to_linear(env_mel)       # (B, 1, F)

        amps, harm_mask = self._harmonic_amps(env_lin, f0_frame)  # f0: (B,1)
        harm_mag = self._harmonic_mag(amps, f0_frame, harm_mask)  # (B, F, 1)
        noise_mag = self._noise_mag(env_lin)  # (B, F, 1)
        period_spec = self._periodicity_spec(period)  # (B, F, 1)

        mag = period_spec * harm_mag + (1 - period_spec) * noise_mag
        out_frame = mag.squeeze(-1)  # (B, F)

        new_state = {
            "stft_tail": new_tail,
            "gru_h": gru_h,
            "f0_buf": f0_buf,
        }
        return out_frame, new_state
