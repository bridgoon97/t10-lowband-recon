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
from ..dsp.stft import causal_istft as _causal_istft
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
        """Return ``(f0, f0_confidence)`` — task ② soft path.

        f0_confidence ∈ [0,1], 1 = trustworthy.  SEMANTICS UNIFIED (req 2):
        ``prob = 1 − CMND`` clamped to [0,1].
        - oracle mode: f0 from ``cond['f0']`` (the §6.5 oracle ablation);
          confidence defaults to 1, or ``cond['f0_confidence']`` if injected
          (deterministic tests, req 5).  NEVER reads f0/confidence from the
          TARGET (confidence is not mined from ref).
        - estimated mode: SOFT YIN (``soft=True``) — always a best candidate in
          [f0_min, f0_max] (NEVER 0) + continuous prob.  No hard threshold,
          no voiced/unvoiced branch (req 1, req 4).
        """
        if self.f0_mode == "oracle" and cond is not None and "f0" in cond:
            f0 = cond["f0"]
            conf = cond.get("f0_confidence")
            if conf is None:
                conf = torch.ones_like(f0)
            return f0, conf.clamp(0.0, 1.0)
        f0, prob = f0_mod.yin_f0(x, self.sample_rate, frame_len=self.stft_cfg.win,
                                  f0_min=self.f0_min, f0_max=self.f0_max, soft=True)
        return f0, prob.clamp(0.0, 1.0)

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
    def _synth_components(self, f0, amps, harm_mask, env_lin, T, train=True):
        """Time-domain harmonic + noise components (B, T) each — NOT blended here.

        Task ②: the harmonic/noise blend is PER-BIN in the SPECTRAL domain
        (see ``forward`` / ``stream_step``), gated by ``effective_periodicity``.
        The old code averaged the interpolated band periodicity into a SCALAR
        ``gain_h`` and blended in time — that silently demoted per-subband
        gating to a full-band scalar; this split returns the two components so
        the caller blends per-bin with a (B, n_bins, N) weight.
        """
        hop = self.stft_cfg.hop
        # zero-order hold upsample (each frame's value repeated for `hop` samples)
        # — MUST match stream_step's per-frame constant control, else stream≢batch.
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
        return harm_wav, noise_wav

    def forward(self, x, cond=None):
        B, T = x.shape
        in_spec = _complex_stft(x, self.stft_cfg)               # (B, n_bins, N) complex
        in_mag = in_spec.abs()
        N = in_mag.shape[-1]
        f0, conf = self._get_f0(x, cond)                       # (B,Ny),(B,Ny)
        # align f0/conf to the STFT frame count N (YIN hop != STFT hop)
        if f0.shape[-1] != N:
            f0 = F.interpolate(f0.unsqueeze(1), size=N, mode="linear",
                               align_corners=False).squeeze(1)
            conf = F.interpolate(conf.unsqueeze(1), size=N, mode="linear",
                                 align_corners=False).squeeze(1)
        # candidate F0 clamped to legal range.  NO 0->50Hz bug: the soft path
        # never returns 0, so clamp cannot fabricate a 50 Hz comb (req 4).
        f0 = f0.clamp(self.f0_min, self.f0_max)
        conf = conf.clamp(0.0, 1.0)
        env_mel, period = self.control_net(in_mag)             # period (B,N,n_bands)
        env_lin = self._mel_to_linear(env_mel)
        amps, harm_mask = self._harmonic_amps(env_lin, f0)
        harm_wav, noise_wav = self._synth_components(f0, amps, harm_mask, env_lin, T,
                                                      train=self.training)
        # task ② req 3: per-subband effective periodicity = learned * confidence
        eff_period = period * conf.unsqueeze(-1)               # (B, N, n_bands)
        # per-bin blend weight (interpolate n_bands -> n_bins); (B, n_bins, N)
        eff_period_spec = self._periodicity_spec(eff_period)   # real in [0,1]
        # PER-BIN SPECTRAL blend (the ACTUAL synthesis path — NOT a scalar
        # average; effective_periodicity participates in synthesis, req 3/6).
        harm_spec = _complex_stft(harm_wav, self.stft_cfg)      # (B, n_bins, N) complex
        noise_spec = _complex_stft(noise_wav, self.stft_cfg)    # (B, n_bins, N) complex
        out_spec = (eff_period_spec * harm_spec
                    + (1.0 - eff_period_spec) * noise_spec)     # complex, per-bin
        wav = _causal_istft(out_spec, self.stft_cfg, length=T)   # (B, T) consistent w/ spec
        return {"spec": out_spec, "wav": wav,
                "aux": {"f0": f0.detach(), "f0_confidence": conf.detach(),
                        "periodicity": period.detach(),
                        "effective_periodicity": eff_period.detach(),
                        "blend_weight_spec": eff_period_spec.detach(),
                        "harmonic_amps": amps.detach(), "harm_mask": harm_mask.detach()}}

    # --- streaming (phase carry + noise OLA carry, eval seeded for equiv) ------
    def stream_init(self, batch_size):
        hop, win = self.stft_cfg.hop, self.stft_cfg.win
        n_fft = self.stft_cfg.n_fft
        return {
            "stft_tail": torch.zeros(batch_size, win - hop),   # INPUT spec STFT (x -> control)
            # task ②: per-frame harm/noise STFT tails for the PER-BIN spectral
            # blend (was a single blended out_tail; per-bin gating needs the two
            # components' spectra separately → two tails).
            "harm_tail": torch.zeros(batch_size, win - hop),
            "noise_tail": torch.zeros(batch_size, win - hop),
            "gru_h": None,
            "f0_buf": torch.zeros(batch_size, win),
            "f0_override": None,                 # inject F0 (Hz), (B,) — symmetric to oracle cond['f0']
            "f0_confidence_override": None,      # inject confidence in [0,1], (B,) — default 1 (req 5)
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

        # --- F0 + confidence (SOFT path, task ②) — no hard voiced/unvoiced branch ---
        if state.get("f0_override") is not None:
            f0_frame = state["f0_override"].view(B).to(device=device, dtype=torch.float32)
            cov = state.get("f0_confidence_override")
            conf_frame = (cov.view(B).to(device=device, dtype=torch.float32)
                          if cov is not None else torch.ones(B, device=device))
        elif self.f0_mode == "oracle":
            f0_frame = torch.full((B,), 150.0, device=device)
            conf_frame = torch.ones(B, device=device)
        else:
            f0_frame, conf_frame = f0_mod._yin_frame(
                f0_buf, self.sample_rate,
                tau_min=max(1, int(self.sample_rate / self.f0_max) - 1),
                tau_max=int(self.sample_rate / self.f0_min) + 2,
                threshold=0.1, soft=True)               # SOFT: candidate + prob, no f0=0
        # candidate clamped to legal range (soft path never 0 ⇒ no 0->50Hz bug)
        f0_frame = f0_frame.clamp(self.f0_min, self.f0_max).unsqueeze(-1)  # (B,1)
        conf_frame = conf_frame.clamp(0.0, 1.0)                            # (B,)

        env_mel, period, gru_h = self.control_net.forward_step(mag_frame, state["gru_h"])
        env_lin = self._mel_to_linear(env_mel)               # (B,1,F)
        amps, harm_mask = self._harmonic_amps(env_lin, f0_frame)  # (B,1,K)

        # --- synthesize hop samples (harm + noise components, time domain) ---
        amps_hop = ddsp_mod.upsample_control(amps.permute(0, 2, 1), hop)   # (B, K, hop)
        mask_hop = ddsp_mod.upsample_control(harm_mask.permute(0, 2, 1), hop)  # (B, K, hop)
        f0_hop = ddsp_mod.upsample_control(f0_frame, hop)               # (B, hop)
        phase_inc = ddsp_mod.accumulate_phase(f0_hop, hop, self.sample_rate)  # (B,hop) from 0
        phase = torch.remainder(phase_inc + state["phase_offset"].unsqueeze(1), TWO_PI)
        new_offset = phase[:, -1:].squeeze(1) if phase.numel() else state["phase_offset"]
        harm_wav_hop = ddsp_mod.harmonic_synth(phase, amps_hop, mask_hop)    # (B, hop)

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
        noise_wav_hop = out_buf[:, :hop] / norm_buf[:, :hop].clamp_min(1e-8)  # (B, hop)
        new_noise_out = torch.cat([out_buf[:, hop:],
                                    torch.zeros(B, hop, device=device, dtype=out_buf.dtype)], dim=1)
        new_noise_norm = torch.cat([norm_buf[:, hop:],
                                      torch.zeros(B, hop, device=device, dtype=norm_buf.dtype)], dim=1)

        # --- PER-BIN spectral blend (task ②) — same semantics as forward ---
        # eff_period = learned_periodicity * f0_confidence (per-subband), then
        # interpolated to n_bins; this weight BLENDS harm/noise SPECTRA per-bin
        # (was a scalar average → now per-bin, req 3).
        eff_period_frame = period * conf_frame.view(B, 1, 1)          # (B,1,n_bands)
        blend = self._periodicity_spec(eff_period_frame).squeeze(-1)  # (B, n_bins) real in [0,1]
        harm_spec_frame, harm_new_tail = _frame_step(harm_wav_hop, self.stft_cfg,
                                                       state["harm_tail"])     # (B, n_bins) complex
        noise_spec_frame, noise_new_tail = _frame_step(noise_wav_hop, self.stft_cfg,
                                                         state["noise_tail"])  # (B, n_bins) complex
        out_spec_frame = (blend * harm_spec_frame
                          + (1.0 - blend) * noise_spec_frame)         # (B, n_bins) complex

        new_state = {"stft_tail": in_new_tail,
                     "harm_tail": harm_new_tail, "noise_tail": noise_new_tail,
                     "gru_h": gru_h, "f0_buf": f0_buf,
                     "f0_override": state.get("f0_override"),
                     "f0_confidence_override": state.get("f0_confidence_override"),
                     "phase_offset": new_offset,
                     "noise_out": new_noise_out, "noise_norm": new_noise_norm,
                     "noise_gen": state["noise_gen"]}
        return out_spec_frame, new_state
