"""Degradation simulation (§4.3).

Not a simple low-pass.  Each effect is independently switchable and unit-tested
(§5.8).  Used by LowpassSimAdapter (L0) and as augmentation for L1.

Effects:
  1. random cutoff frequency 300–1200 Hz (not fixed 500)
  2. random roll-off slope 6–36 dB/oct (gradual, never rectangular bin mask)
  3. mask to noise floor −40 to −70 dB, never hard zero
  4. intra-utterance slow time-varying cutoff (0.2–2 Hz modulation)
  5. spectral tilt + 1–2 random formants
  6. low-frequency body-conduction noise (random impacts/friction)
  7. occasional clipping (small-probability nonlinearity)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class DegradationConfig:
    """All knobs exposed — every effect independently switchable."""

    def __init__(self,
                 cutoff_min: float = 300.0, cutoff_max: float = 1200.0,
                 rolloff_min: float = 6.0, rolloff_max: float = 36.0,  # dB/oct
                 noise_floor_min_db: float = -70.0, noise_floor_max_db: float = -40.0,
                 time_vary: bool = True, time_vary_rate: tuple = (0.2, 2.0),  # Hz
                 time_vary_depth: float = 0.3,  # fractional cutoff drift
                 spectral_tilt: bool = True,
                 formants: bool = True, n_formants: int = 2,
                 body_noise: bool = True, body_noise_prob: float = 0.3,
                 clipping: bool = True, clip_prob: float = 0.1,
                 sample_rate: float = 4000.0):
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max
        self.rolloff_min = rolloff_min
        self.rolloff_max = rolloff_max
        self.noise_floor_min_db = noise_floor_min_db
        self.noise_floor_max_db = noise_floor_max_db
        self.time_vary = time_vary
        self.time_vary_rate = time_vary_rate
        self.time_vary_depth = time_vary_depth
        self.spectral_tilt = spectral_tilt
        self.formants = formants
        self.n_formants = n_formants
        self.body_noise = body_noise
        self.body_noise_prob = body_noise_prob
        self.clipping = clipping
        self.clip_prob = clip_prob
        self.sample_rate = sample_rate

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _hz_to_bin(hz: float, n_fft: int, sr: float) -> float:
    return hz * n_fft / sr


def _design_lowpass_response(n_fft: int, sr: float, cutoff_hz: float,
                              rolloff_dboct: float, noise_floor_db: float,
                              device=None, dtype=torch.float32) -> torch.Tensor:
    """Build a smooth low-pass magnitude response (n_fft//2+1 bins).

    Gradual roll-off, floor at noise_floor_db (NOT zero).
    """
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_bins, device=device, dtype=dtype)
    # Transition: passband (flat 1.0) -> stopband (noise floor)
    # roll-off in dB/oct: convert to dB/bin
    bins_per_oct = n_fft / (sr / 2) * (sr / 2 / cutoff_hz) if cutoff_hz > 0 else 1
    # Each octave = doubling of frequency.  dB per bin:
    # at bin b (freq f), distance from cutoff in octaves = log2(f / cutoff)
    # attenuation = rolloff_dboct * log2(f / cutoff) dB, clamp at noise_floor
    ratio = freqs / cutoff_hz
    # Avoid log(0)
    ratio = ratio.clamp_min(1e-8)
    octaves = torch.log2(ratio)
    atten_db = rolloff_dboct * octaves  # positive above cutoff
    # Passband: atten_db <= 0 → gain = 1 (0 dB). Stopband: roll down to floor.
    gain_db = torch.where(octaves > 0, -atten_db, torch.zeros_like(octaves))
    gain_db = gain_db.clamp(min=noise_floor_db, max=0.0)
    gain = 10.0 ** (gain_db / 20.0)
    return gain


def apply_degradation(x: torch.Tensor, cfg: DegradationConfig,
                      rng: np.random.Generator | None = None,
                      n_fft: int = 128) -> torch.Tensor:
    """Apply the full degradation chain to a single waveform.

    Args:
        x: (T,) or (B, T) waveform, float in [-1, 1].
        cfg: DegradationConfig.
        rng: numpy RNG for reproducibility.
        n_fft: FFT size for spectral processing.

    Returns:
        degraded: same shape as x.
    """
    if rng is None:
        rng = np.random.default_rng()
    sr = cfg.sample_rate
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    B, T = x.shape
    device = x.device

    # Sample per-utterance parameters
    base_cutoff = float(rng.uniform(cfg.cutoff_min, cfg.cutoff_max))
    rolloff = float(rng.uniform(cfg.rolloff_min, cfg.rolloff_max))
    noise_floor_db = float(rng.uniform(cfg.noise_floor_min_db,
                                       cfg.noise_floor_max_db))

    # --- time-varying cutoff (effect 4) ---
    if cfg.time_vary:
        mod_rate = float(rng.uniform(*cfg.time_vary_rate))
        depth = cfg.time_vary_depth
        t = np.arange(T) / sr
        mod = np.sin(2 * np.pi * mod_rate * t + rng.uniform(0, 2 * np.pi))
        cutoff_per_frame = base_cutoff * (1 + depth * mod)
        cutoff_per_frame = np.clip(cutoff_per_frame, cfg.cutoff_min, cfg.cutoff_max)
    else:
        cutoff_per_frame = np.full(T, base_cutoff)

    # Build per-sample spectral response (approximate: use STFT frames)
    hop = n_fft // 4
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                    window=torch.hann_window(n_fft, device=device),
                    return_complex=True, center=True)
    # X: (B, F, N_frames) — actual frame count depends on center padding
    n_frames = X.shape[-1]
    n_bins = X.shape[1]

    # Frame-wise cutoff (resample to STFT frame rate)
    frame_times = np.linspace(0, T / sr, n_frames)
    cutoff_frames = np.interp(frame_times, np.arange(T) / sr, cutoff_per_frame)

    # Build response per frame
    response = torch.zeros(B, n_bins, n_frames, device=device, dtype=x.dtype)
    freqs = torch.linspace(0, sr / 2, n_bins, device=device, dtype=x.dtype)
    for fi, cf in enumerate(cutoff_frames):
        cf_t = torch.tensor(cf, device=device, dtype=x.dtype)
        ratio = freqs / cf_t.clamp_min(1.0)
        ratio = ratio.clamp_min(1e-8)
        octaves = torch.log2(ratio)
        atten_db = torch.where(octaves > 0, -rolloff * octaves,
                              torch.zeros_like(octaves))
        atten_db = atten_db.clamp(min=noise_floor_db, max=0.0)
        response[:, :, fi] = 10.0 ** (atten_db / 20.0)

    # Apply response + noise floor (effect 3)
    noise_floor_lin = 10.0 ** (noise_floor_db / 20.0)
    noise_spec = (torch.randn_like(X) * noise_floor_lin)
    X_deg = X * response + noise_spec * (1 - response)

    # --- spectral tilt + formants (effect 5) ---
    tilt_db = float(rng.uniform(-3, 6))  # positive = tilt toward lows
    tilt = 10.0 ** (torch.linspace(0, tilt_db, n_bins, device=device, dtype=x.dtype) / 20.0)
    X_deg = X_deg * tilt.unsqueeze(0).unsqueeze(-1)
    if cfg.formants:
        for _ in range(cfg.n_formants):
            f_form = float(rng.uniform(200, sr / 2))
            bw = float(rng.uniform(50, 300))
            gain = float(rng.uniform(0.5, 3.0))
            formant = _formant_response(freqs, f_form, bw, gain)
            X_deg = X_deg * formant.unsqueeze(0).unsqueeze(-1)

    # iSTFT back to time
    x_deg = torch.istft(X_deg, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                         window=torch.hann_window(n_fft, device=device),
                         length=T, center=True)

    # --- body-conduction noise (effect 6) ---
    if cfg.body_noise and rng.random() < cfg.body_noise_prob:
        n_impacts = int(rng.integers(1, 5))
        for _ in range(n_impacts):
            pos = int(rng.integers(0, T))
            amp = float(rng.uniform(0.05, 0.3))
            decay = float(rng.uniform(50, 300))
            length = min(int(sr * 0.05), T - pos)
            if length > 0:
                env = torch.exp(-torch.arange(length, device=device).float() / decay * sr / 1000)
                x_deg[:, pos:pos + length] += amp * env * torch.randn(B, length, device=device)

    # --- clipping (effect 7) ---
    if cfg.clipping and rng.random() < cfg.clip_prob:
        clip_level = float(rng.uniform(0.3, 0.8))
        x_deg = torch.clamp(x_deg, -clip_level, clip_level)

    if squeeze:
        x_deg = x_deg.squeeze(0)
    return x_deg


def _formant_response(freqs: torch.Tensor, f0: float, bw: float,
                      gain: float) -> torch.Tensor:
    """Single-pole resonator magnitude response."""
    w = 2 * math.pi * freqs
    w0 = 2 * math.pi * f0
    alpha = 2 * math.pi * bw / 2
    h = w0 ** 2 / torch.sqrt((w0 ** 2 - w ** 2) ** 2 + (2 * alpha * w) ** 2 + 1e-12)
    h = h / h.max()
    return 1.0 + (gain - 1.0) * h


# --- measurement for §5.8 verification ----------------------------------------
def measure_cutoff(x: torch.Tensor, sr: float, n_fft: int = 512) -> tuple[float, float, float]:
    """Measure actual achieved cutoff, roll-off slope, and noise floor.

    Cutoff = first frequency where magnitude drops 3 dB below the PASSBAND
    LEVEL (median of the lowest 10% of bins), not below the global peak.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    X = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft, device=x.device),
                    return_complex=True, center=True)
    mag = X.abs().mean(dim=(0, 2))  # (F,)
    freqs = torch.linspace(0, sr / 2, len(mag))
    mag_db = 20 * torch.log10(mag.clamp_min(1e-10))
    # Passband level = median of lowest 10% of bins
    n_low = max(1, len(mag) // 10)
    passband_db = float(mag_db[:n_low].median())
    target = passband_db - 3.0
    above = (mag_db < target).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        cutoff_hz = float(sr / 2)
    else:
        idx = above[0].item()
        if idx > 0:
            lo, hi = mag_db[idx - 1], mag_db[idx]
            frac = (lo - target) / (lo - hi + 1e-12)
            cutoff_hz = float(freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1]))
        else:
            cutoff_hz = float(freqs[idx])
    # Roll-off: slope between cutoff and 2*cutoff
    idx_c = int(cutoff_hz / (sr / 2) * len(mag))
    idx_2c = min(int(2 * cutoff_hz / (sr / 2) * len(mag)), len(mag) - 1)
    if idx_2c > idx_c and mag_db[idx_c] > mag_db[idx_2c]:
        delta_db = float(mag_db[idx_c] - mag_db[idx_2c])
        rolloff = delta_db / 1.0  # one octave
    else:
        rolloff = 0.0
    # Noise floor: median of highest 1/3 of bins
    n_third = len(mag) // 3
    noise_floor_db = float(mag_db[-n_third:].median())
    return cutoff_hz, rolloff, noise_floor_db
