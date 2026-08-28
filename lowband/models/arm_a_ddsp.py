"""Arm A — DDSP / Harmonic+Noise synthesizer (task spec §6).

Spec change (口径迁移): output is the TRUNCATED COMPLEX spectrum
(B, keep_bins=65, N).  An amplitude-domain harmonic Gaussian smear CANNOT yield a
complex spectrum (no phase), so the waveform-synthesis path
(oscillator → waveform → STFT → truncate) is now the MAIN path, on by default.

Anti-aliasing (§6.1) semantics changed: Nyquist is now 8 kHz (sr=16 k) and no
longer coincides with the band top 2 kHz.  The harmonic mask cuts at the BAND
TOP (``band_top_hz`` = 2000 Hz), NOT at Nyquist — harmonics above 2 kHz would
only be synthesized-then-truncated-away = wasted compute.

F0 comes from the input (YIN) or an oracle path, NEVER the target (§6.5).
Sub-band periodicity (§6.3) remains a hard requirement.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import LowBandReconstructor
from ..dsp.stft import StftConfig as _StftConfig
from ..dsp.stft import complex_stft_truncated as _complex_stft
from ..dsp.stft import frame_step as _frame_step
from ..dsp.stft import bin_to_hz, hz_to_bin
from ..dsp import ddsp as ddsp_mod
from ..dsp import f0 as f0_mod

TWO_PI = 2.0 * math.pi


class ControlNet(nn.Module):
    """Small conv-recurrent net: input magnitude -> mel envelope + periodicity."""

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

    def forward(self, mag):
        x = mag.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = x.mean(dim=2).permute(0, 2, 1).contiguous()
        out, _ = self.gru(feat)
        return self.head_env(out), torch.sigmoid(self.head_period(out))

    def forward_step(self, mag_frame, gru_h):
        x = mag_frame.unsqueeze(1).unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = x.mean(dim=2).permute(0, 2, 1).contiguous()
        out, h = self.gru(feat, gru_h)
        return self.head_env(out), torch.sigmoid(self.head_period(out)), h.detach() if h is not None else None


class ArmA_DDSP(LowBandReconstructor):
    """DDSP harmonic+noise reconstructor → truncated COMPLEX spectrum."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.sample_rate = cfg["sample_rate"]
        self.stft_cfg = _StftConfig(
            n_fft=cfg.get("stft_n_fft", 512),
            hop=cfg.get("stft_hop", 160),
            win=cfg.get("stft_win", 480),
            window=cfg.get("stft_window", "hann"),
            keep_bins=cfg.get("keep_bins", 64),
        )
        self.n_bins = self.stft_cfg.keep_bins
        self.nyquist = self.sample_rate / 2          # 8 kHz (for ref only)
        self.bin_width = self.sample_rate / self.stft_cfg.n_fft  # 31.25 Hz/bin
        # anti-alias mask cuts at the BAND TOP (kept 0–2 kHz), not Nyquist
        self.band_top_hz = cfg.get("band_top_hz", 2000.0)

        self.n_mel = cfg.get("n_mel", 16)
        self.n_bands = cfg.get("n_bands", 12)
        self.max_harm = cfg.get("max_harm", 32)
        self.f0_mode = cfg.get("f0_mode", "estimated")
        self.f0_min = cfg.get("f0_min", 50.0)
        self.f0_max = cfg.get("f0_max", 400.0)
        self.waveform_synth = cfg.get("waveform_synth", True)  # ON by default now

        self.control_net = ControlNet(self.n_bins, self.n_mel, self.n_bands,
                                      hidden=cfg.get("ctrl_hidden", 48))
        # mel filterbank over the DC-dropped model bins (freqs via bin_to_hz,
        # the single source of the +1 offset) — NOT the full FFT bins
        bin_freqs = bin_to_hz(torch.arange(self.n_bins), self.sample_rate,
                              self.stft_cfg.n_fft)
        fb = ddsp_mod.mel_filterbank(self.n_mel, self.stft_cfg.n_fft,
                                      self.sample_rate,
                                      f_min=float(bin_freqs[0]),
                                      f_max=float(bin_freqs[-1]),
                                      bin_freqs=bin_freqs)
        self.register_buffer("mel_fb", fb)                # (n_mel, n_bins=64)
        self.register_buffer("mel_fb_inv", torch.linalg.pinv(fb))  # (64, n_mel)
        self.register_buffer("bin_freqs", bin_freqs)       # (n_bins,) Hz of each model bin
        self.smear_sigma = cfg.get("smear_sigma", 2.0)
        # fixed seed for EVAL noise (makes stream≡batch reproducible; train mode
        # uses fresh noise — equiv is an eval-only check)
        self._noise_seed = 0

    # --- helpers (mostly unchanged; mask now uses band_top_hz) ----------------
    def _get_f0(self, x, cond):
        if self.f0_mode == "oracle" and cond is not None and "f0" in cond:
            return cond["f0"]
        f0, _ = f0_mod.yin_f0(x, self.sample_rate, frame_len=self.stft_cfg.win,
                              f0_min=self.f0_min, f0_max=self.f0_max)
        return f0

    def _mel_to_linear(self, env_mel):
        env_mel = F.softplus(env_mel)
        return (env_mel @ self.mel_fb_inv.T).clamp_min(0.0)

    def _harmonic_amps(self, env_lin, f0):
        B, N, Fb = env_lin.shape
        K = self.max_harm
        k = torch.arange(1, K + 1, device=env_lin.device, dtype=env_lin.dtype)
        harm_freq = f0.unsqueeze(-1) * k                       # (B, N, K)
        bin_idx = hz_to_bin(harm_freq, self.sample_rate, self.stft_cfg.n_fft).clamp(0.0, float(Fb - 1))
        idx_lo = bin_idx.floor().long()
        idx_hi = (idx_lo + 1).clamp(max=Fb - 1)
        frac = bin_idx - idx_lo.float()
        amps = (torch.gather(env_lin, -1, idx_lo) * (1 - frac)
                + torch.gather(env_lin, -1, idx_hi) * frac)
        # §6.1 anti-alias mask: cut at BAND TOP (kept 0–2k), not Nyquist 8k
        mask = (harm_freq < (self.band_top_hz - 1.0)).float()
        return amps, mask

    def _periodicity_spec(self, period):
        return F.interpolate(period, size=self.n_bins, mode="linear",
                              align_corners=False).permute(0, 2, 1).contiguous()

    def _noise_spec(self, env_lin):
        return env_lin.permute(0, 2, 1).contiguous()           # (B, F, N)

    # --- waveform synthesis (now the MAIN path) -------------------------------
    def _synth_waveform(self, f0, amps, harm_mask, env_lin, period, T, train=True):
        hop = self.stft_cfg.hop
        # zero-order hold upsample (each frame's value repeated for `hop` samples)
        # — MUST match stream_step's per-frame constant control, else stream≢batch.
        # (phase is integrated per-sample from this piecewise-constant f0, so it
        # stays smooth; only the envelope/mask are piecewise-constant, which is
        # fine for a vocoder-style control.)
        def _zoh3(v):   # (B,N,K) -> (B,K,T)
            r = v.permute(0, 2, 1).repeat_interleave(hop, dim=-1)  # (B,K,N*hop)
            if r.shape[-1] < T:
                r = F.pad(r, (0, T - r.shape[-1]), mode="replicate")
            return r[..., :T]
        def _zoh2(v):   # (B,N) -> (B,T)
            r = v.repeat_interleave(hop, dim=-1)
            if r.shape[-1] < T:
                r = F.pad(r, (0, T - r.shape[-1]), mode="replicate")
            return r[..., :T]
        f0_ps = _zoh2(f0)                                  # (B, T)
        amps_ps = _zoh3(amps)                             # (B, K, T)
        mask_ps = _zoh3(harm_mask)                         # (B, K, T)
        phase = ddsp_mod.accumulate_phase(f0_ps, T, self.sample_rate)
        harm_wav = ddsp_mod.harmonic_synth(phase, amps_ps, mask_ps)        # (B, T)
        noise_wav = ddsp_mod.noise_synth(
            self._noise_spec(env_lin), self.stft_cfg.n_fft, self.stft_cfg.hop,
            seed=(None if train else self._noise_seed), train=train, length=T)  # (B, T)
        period_T = self._periodicity_spec(period).repeat_interleave(hop, dim=-1)  # (B,F,N*hop)
        if period_T.shape[-1] < T:
            period_T = F.pad(period_T, (0, T - period_T.shape[-1]), mode="replicate")
        period_T = period_T[..., :T]
        gain_h = period_T.mean(dim=1)                      # (B, T)
        return gain_h * harm_wav + (1 - gain_h) * noise_wav            # (B, T)

    def forward(self, x, cond=None):
        B, T = x.shape
        in_spec = _complex_stft(x, self.stft_cfg)               # (B, 65, N) complex
        in_mag = in_spec.abs()
        N = in_mag.shape[-1]
        f0 = self._get_f0(x, cond)
        if f0.shape[-1] != N:
            f0 = F.interpolate(f0.unsqueeze(1), size=N, mode="linear",
                               align_corners=False).squeeze(1)
        f0 = f0.clamp(self.f0_min, self.f0_max)
        env_mel, period = self.control_net(in_mag)
        env_lin = self._mel_to_linear(env_mel)
        amps, harm_mask = self._harmonic_amps(env_lin, f0)
        wav = self._synth_waveform(f0, amps, harm_mask, env_lin, period, T,
                                   train=self.training)
        spec = _complex_stft(wav, self.stft_cfg)               # (B, 65, N) complex
        return {"spec": spec, "wav": wav,
                "aux": {"f0": f0.detach(), "periodicity": period.detach(),
                        "harmonic_amps": amps.detach(), "harm_mask": harm_mask.detach()}}

    # --- streaming (phase carry + noise OLA carry, eval seeded for equiv) ------
    def stream_init(self, batch_size):
        hop, win = self.stft_cfg.hop, self.stft_cfg.win
        n_fft = self.stft_cfg.n_fft
        return {
            "stft_tail": torch.zeros(batch_size, win - hop),   # INPUT spec STFT (x -> control)
            "out_tail": torch.zeros(batch_size, win - hop),     # OUTPUT spec STFT (wav -> spec)
            "gru_h": None,
            "f0_buf": torch.zeros(batch_size, win),
            "f0_override": None,
            "phase_offset": torch.zeros(batch_size),
            "noise_out": torch.zeros(batch_size, n_fft),
            "noise_norm": torch.zeros(batch_size, n_fft),
            "noise_gen": torch.Generator().manual_seed(self._noise_seed),
        }

    def stream_step(self, x_frame, state):
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        n_fft = self.stft_cfg.n_fft
        B = x_frame.shape[0]
        device = x_frame.device

        # --- input spectrum frame (for control net) ---
        f0_buf = torch.cat([state["f0_buf"][:, hop:], x_frame], dim=1)
        in_spec_frame, in_new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])
        mag_frame = in_spec_frame.abs()     # control net operates on magnitude
        if state.get("f0_override") is not None:
            f0_frame = state["f0_override"].view(B)
        elif self.f0_mode == "oracle":
            f0_frame = torch.full((B,), 150.0, device=device)
        else:
            f0_frame, _ = f0_mod._yin_frame(
                f0_buf, self.sample_rate,
                tau_min=max(1, int(self.sample_rate / self.f0_max) - 1),
                tau_max=int(self.sample_rate / self.f0_min) + 2, threshold=0.1)
        f0_frame = f0_frame.clamp(self.f0_min, self.f0_max).unsqueeze(-1)  # (B,1)

        env_mel, period, gru_h = self.control_net.forward_step(mag_frame, state["gru_h"])
        env_lin = self._mel_to_linear(env_mel)               # (B,1,F)
        amps, harm_mask = self._harmonic_amps(env_lin, f0_frame)  # (B,1,K)

        # --- synthesize hop samples (matching _synth_waveform's segment) ---
        amps_hop = ddsp_mod.upsample_control(amps.permute(0, 2, 1), hop)   # (B, K, hop)
        mask_hop = ddsp_mod.upsample_control(harm_mask.permute(0, 2, 1), hop)  # (B, K, hop)
        f0_hop = ddsp_mod.upsample_control(f0_frame, hop)               # (B, hop)
        phase_inc = ddsp_mod.accumulate_phase(f0_hop, hop, self.sample_rate)  # (B,hop) from 0
        phase = torch.remainder(phase_inc + state["phase_offset"].unsqueeze(1), TWO_PI)
        new_offset = phase[:, -1:].squeeze(1) if phase.numel() else state["phase_offset"]
        harm_wav = ddsp_mod.harmonic_synth(phase, amps_hop, mask_hop)    # (B,hop)

        # noise OLA per frame (mirrors noise_synth; seeded in eval for equiv)
        noise_mags = self._noise_spec(env_lin)  # (B,F,1)
        gen = state["noise_gen"] if not self.training else None
        nf = torch.randn(B, noise_mags.shape[1], 1, device=device,
                         generator=gen, dtype=noise_mags.dtype)
        fw = torch.fft.irfft((nf * noise_mags), n=n_fft, dim=1).squeeze(-1)   # (B, n_fft)
        from ..dsp.stft import get_window
        w = get_window(self.stft_cfg.window, n_fft, device=device, dtype=fw.dtype)
        fw = fw * w.unsqueeze(0)
        out_buf = state["noise_out"] + fw
        norm_buf = state["noise_norm"] + w.unsqueeze(0)
        noise_wav = out_buf[:, :hop] / norm_buf[:, :hop].clamp_min(1e-8)  # (B, hop)
        new_noise_out = torch.cat([out_buf[:, hop:],
                                    torch.zeros(B, hop, device=device, dtype=out_buf.dtype)], dim=1)
        new_noise_norm = torch.cat([norm_buf[:, hop:],
                                      torch.zeros(B, hop, device=device, dtype=norm_buf.dtype)], dim=1)

        period_spec = self._periodicity_spec(period)              # (B, F, 1)
        gain_h = ddsp_mod.upsample_control(period_spec, hop).mean(dim=1)  # (B, hop)
        wav_hop = gain_h * harm_wav + (1 - gain_h) * noise_wav    # (B, hop)

        # output spec STFT — INDEPENDENT tail from the input spec STFT
        spec_frame, out_new_tail = _frame_step(wav_hop, self.stft_cfg, state["out_tail"])

        new_state = {"stft_tail": in_new_tail, "out_tail": out_new_tail, "gru_h": gru_h,
                     "f0_buf": f0_buf,
                     "f0_override": state.get("f0_override"),
                     "phase_offset": new_offset,
                     "noise_out": new_noise_out, "noise_norm": new_noise_norm,
                     "noise_gen": state["noise_gen"]}
        return spec_frame, new_state
