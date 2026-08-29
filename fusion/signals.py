"""Synthetic signal builders for the M1–M7 mechanism tests (T13-A).

All M-criteria are measurable on SYNTHETIC signals (spec mandates this — real
sensor domain not needed for mechanism correctness).  Builders return torch
tensors shaped (B, T) at sr=16 kHz unless noted.
"""
from __future__ import annotations

import numpy as np
import torch


def _t(dur_s: float, sr: int) -> torch.Tensor:
    return torch.arange(int(dur_s * sr), dtype=torch.float32) / sr


def harmonic_train(F0: float, dur_s: float, amps, sr: int = 16000,
                   noise_db: float = -60.0) -> torch.Tensor:
    """Sum_k amps[k]·sin(2π(k+1)F0 t) + low noise.  ``amps``: list/arr, len=K.
    Returns (1, T)."""
    t = _t(dur_s, sr)
    x = torch.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        x = x + a * torch.sin(2 * np.pi * k * F0 * t)
    # small noise floor (so killed harmonics are "buried", not silence)
    n = torch.randn_like(t) * (10 ** (noise_db / 20)) * amps[0] if len(amps) else 0
    x = x + n
    return x.unsqueeze(0)


def tilted_pair(F0: float, dur_s: float, K: int, tilt_db: float,
                sr: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
    """S = harmonic train; V = same train + known +tilt_db spectral tilt
    (per-harmonic gain ramp).  For M2 EQ convergence: C should recover the tilt."""
    t = _t(dur_s, sr)
    s = torch.zeros_like(t)
    v = torch.zeros_like(t)
    bz = sr / 512  # bin hz for tilt→per-harmonic mapping (approx)
    for k in range(1, K + 1):
        a = 1.0 / k
        # tilt: +tilt_db * (harmonic freq / F0_norm) ramp — known per-harmonic offset
        fk = k * F0
        g_db = tilt_db * (fk / (K * F0))   # 0..tilt_db across harmonics
        s = s + a * torch.sin(2 * np.pi * fk * t)
        v = v + a * (10 ** (g_db / 20)) * torch.sin(2 * np.pi * fk * t)
    s = s + 1e-4 * torch.randn_like(t)
    v = v + 1e-4 * torch.randn_like(t)
    return s.unsqueeze(0), v.unsqueeze(0)


def attenuated(v_clean: torch.Tensor, db: float) -> torch.Tensor:
    """V attenuated by ``db`` dB (broadband).  For M3 c_V monotonicity."""
    return v_clean * (10 ** (db / 20))


def step_v(off_s: float, on_s: float, dur_s: float, F0: float, sr: int = 16000
           ) -> torch.Tensor:
    """V = zeros(off) → harmonic(on) → zeros(rest).  For M4 (feed the raw-w
    smoother a step).  Returns (1, T)."""
    t = _t(dur_s, sr)
    x = torch.zeros_like(t)
    lo = int(off_s * sr)
    hi = int((off_s + on_s) * sr)
    seg = t[lo:hi] - t[lo]
    for k in range(1, 5):
        x[lo:hi] = x[lo:hi] + (1.0 / k) * torch.sin(2 * np.pi * k * F0 * seg)
    return x.unsqueeze(0)


def voiced_unvoiced(F0: float, dur_s: float, sr: int = 16000) -> torch.Tensor:
    """Alternating voiced (harmonic) / unvoiced (white noise) segments, equal
    level.  For M5: voiced frames have high f0_confidence, noise low.  (1, T)."""
    t = _t(dur_s, sr)
    x = torch.zeros_like(t)
    nseg = 8
    seg = dur_s / nseg
    for i in range(nseg):
        lo = int(i * seg * sr)
        hi = int((i + 1) * seg * sr)
        if i % 2 == 0:  # voiced
            seg_t = t[lo:hi] - t[lo]
            for k in range(1, 6):
                x[lo:hi] = x[lo:hi] + (1.0 / k) * torch.sin(2 * np.pi * k * F0 * seg_t)
        else:           # unvoiced
            x[lo:hi] = torch.randn(hi - lo) * 0.3
    return x.unsqueeze(0)


def tilted_noise_pair(dur_s: float, tilt_db: float, sr: int = 16000,
                      n_fft: int = 512, lo_hz: float = 100.0,
                      hi_hz: float = 2000.0
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous-frequency tilt pair for M2: X = bandpassed noise (signal at
    every bin in [lo,hi]), V = X with a known +tilt_db linear ramp across the
    band (so d=log|S|-log|V| = -tilt is well-defined at EVERY bin, no noise-bin
    contamination of the EQ).  Returns (S=X, V=tilted) each (1, T)."""
    from lowband.dsp.stft import causal_stft, causal_istft, StftConfig
    import torch.nn.functional as F
    T = int(dur_s * sr)
    x = torch.randn(1, T)
    # bandpass lo..hi via STFT masking (offline construction is fine — this is
    # data prep, not the algorithm path)
    cfg = StftConfig(n_fft=n_fft, hop=n_fft // 4, win=n_fft, window="hann")
    spec, _ = causal_stft(x, cfg)
    bz = sr / n_fft
    lo = max(1, int(lo_hz / bz)); hi = min(spec.shape[1] - 1, int(hi_hz / bz))
    mask = torch.zeros_like(spec, dtype=torch.float32)
    mask[:, lo:hi + 1] = 1.0
    xb = causal_istft(spec * mask, cfg, length=T)
    # continuous tilt ramp 0..tilt_db across [lo,hi] bins
    ramp = torch.zeros(spec.shape[1])
    idx = torch.arange(lo, hi + 1)
    ramp[lo:hi + 1] = tilt_db * (idx - lo).float() / max(1, (hi - lo))
    v_spec = spec * (10.0 ** (ramp.view(1, -1, 1) / 20.0))
    v = causal_istft(v_spec * mask, cfg, length=T)
    return xb, v
