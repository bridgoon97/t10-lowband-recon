"""DDSP building blocks for Arm A: harmonic oscillator, noise source, anti-aliasing.

Correctness points (task spec §6):

§6.1 Anti-aliasing
    At 4 kHz Nyquist = 2 kHz == target band top.  Harmonic index
    ``K = floor(Nyquist / F0)`` changes with F0.  Any harmonic with
    ``k * F0 >= Nyquist`` MUST be zeroed, otherwise it folds back as a
    spurious in-band tone.  We use a FIXED K_max and a boolean mask so the
    tensor shape is static (batchable, torch.compile-able).

§6.2 Phase accumulator precision
    Phase is accumulated in float64 and reduced mod 2π periodically to prevent
    slow drift that manifests as detuning on long signals.

§6.3 Sub-band periodicity
    Periodicity is per mel-band (default 12 bands), never a single scalar.

§7.1 Device-agnostic
    All ops are pure torch — no numpy in forward.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


def harmonic_index_mask(f0_hz: torch.Tensor, max_harm: int, nyquist_hz: float) -> torch.Tensor:
    """Boolean mask (B, max_harm) — True where harmonic k*f0 < Nyquist (§6.1)."""
    k = torch.arange(1, max_harm + 1, device=f0_hz.device, dtype=f0_hz.dtype)
    harm_freq = k.unsqueeze(0) * f0_hz.unsqueeze(1)  # (B, K)
    return harm_freq < (nyquist_hz - 1.0)


def accumulate_phase(f0_hz: torch.Tensor, n_samples: int, sample_rate: float,
                     dtype64: bool = True) -> torch.Tensor:
    """Accumulate instantaneous phase = cumsum(2π f0 / fs) mod 2π (§6.2).

    Args:
        f0_hz: (B,) or (B, n_samples) fundamental frequency in Hz.
        n_samples: output length.
        sample_rate: Hz.
        dtype64: accumulate in float64 (recommended).

    Returns:
        phase: (B, n_samples) phase in [0, 2π), float32.
    """
    if f0_hz.dim() == 1:
        f0_per_sample = f0_hz.unsqueeze(0).expand(-1, n_samples)
    elif f0_hz.shape[-1] == n_samples:
        f0_per_sample = f0_hz
    elif f0_hz.shape[-1] == 1:
        f0_per_sample = f0_hz.expand(f0_hz.shape[0], n_samples)
    else:
        raise ValueError(f"f0_hz shape {tuple(f0_hz.shape)} incompatible with n_samples={n_samples}")

    acc_dtype = torch.float64 if dtype64 else torch.float32
    f0_64 = f0_per_sample.to(acc_dtype)
    inc = (TWO_PI * f0_64 / float(sample_rate))  # (B, n_samples)

    # Chunked cumsum with periodic mod to bound accumulator magnitude.
    chunk = 4096
    phase = torch.zeros(f0_per_sample.shape[0], n_samples, dtype=acc_dtype, device=f0_hz.device)
    carry = torch.zeros(f0_per_sample.shape[0], 1, dtype=acc_dtype, device=f0_hz.device)
    pos = 0
    while pos < n_samples:
        end = min(pos + chunk, n_samples)
        c = torch.cumsum(inc[:, pos:end], dim=1) + carry
        c = torch.remainder(c, TWO_PI)
        phase[:, pos:end] = c
        carry = c[:, -1:].clone()
        pos = end
    return phase.to(torch.float32)


def harmonic_synth(phase: torch.Tensor, amps: torch.Tensor,
                   harm_mask: torch.Tensor) -> torch.Tensor:
    """Synthesize summed harmonic waveform.

    Args:
        phase: (B, T) base phase (mod 2π), float32.
        amps: (B, K, T) amplitude per harmonic per sample.
        harm_mask: (B, K) bool — True = keep (anti-alias gate), OR (B, K, T)
            for a per-sample mask (when F0 varies sample-to-sample).

    Returns:
        wav: (B, T).
    """
    K = amps.shape[1]
    k = torch.arange(1, K + 1, device=phase.device, dtype=phase.dtype)
    harm_phase = k.view(1, -1, 1) * phase.unsqueeze(1)  # (B, K, T)
    harm_phase = torch.remainder(harm_phase, TWO_PI)
    harm = amps * torch.sin(harm_phase)
    if harm_mask.dim() == 2:                       # (B, K) static gate
        harm = harm * harm_mask.unsqueeze(-1).to(harm.dtype)
    else:                                         # (B, K, T) per-sample
        harm = harm * harm_mask.to(harm.dtype)
    return harm.sum(dim=1)


def noise_synth(noise_mags: torch.Tensor, n_fft: int, hop: int,
                seed: int | None = None, train: bool = True,
                length: int | None = None) -> torch.Tensor:
    """Generate filtered white noise via overlap-add (§6.6).

    Args:
        noise_mags: (B, F, N_frames) filter magnitude per STFT frame.
        n_fft: synthesis FFT size (must match noise_mags F = n_fft//2+1).
        hop: synthesis hop.
        seed: deterministic seed (reproducible tests).
        train: if False, use seed; if True, re-sample.
        length: output length; if None, inferred.

    Returns:
        wav: (B, T) filtered noise.
    """
    from .stft import get_window
    B, F_bins, N = noise_mags.shape
    device, dtype = noise_mags.device, noise_mags.dtype
    # Draw noise FRAME-BY-FRAME so an eval (seeded) call reproduces the EXACT
    # per-frame draws a streaming path makes with a carried generator (same seed,
    # same draw order) — this is what makes Arm-A stream≡batch reproducible.
    gen = None
    if not train and seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    noise_frames = torch.empty(B, F_bins, N, device=device, dtype=dtype)
    for i in range(N):
        noise_frames[:, :, i:i + 1] = torch.randn(
            B, F_bins, 1, generator=gen, device=device, dtype=dtype)

    filt_spec = noise_frames * noise_mags
    frame_wav = torch.fft.irfft(filt_spec, n=n_fft, dim=1)  # (B, n_fft, N)
    w = get_window("hann", n_fft, device=device, dtype=dtype)
    frame_wav = frame_wav * w.view(1, -1, 1)

    out_len = (N - 1) * hop + n_fft
    if length is not None:
        out_len = max(out_len, length)
    out = torch.zeros(B, out_len, device=device, dtype=dtype)
    norm = torch.zeros(B, out_len, device=device, dtype=dtype)
    for i in range(N):
        s = i * hop
        out[:, s:s + n_fft] += frame_wav[:, :, i]
        norm[:, s:s + n_fft] += w
    out = out / norm.clamp_min(1e-8)
    if length is not None:
        out = out[:, :length]
    return out


# --- mel helpers for §6.3 sub-band periodicity ----------------------------
def mel_filterbank(n_mels: int, n_fft: int, sample_rate: float,
                   f_min: float = 0.0, f_max: float | None = None,
                   bin_freqs: torch.Tensor | None = None,
                   device=None, dtype=torch.float32) -> torch.Tensor:
    """Slaney-style mel filterbank.

    By default shape (n_mels, n_fft//2+1) over the full FFT bin freqs.
    If ``bin_freqs`` is given (e.g. ``bin_to_hz(arange(keep), sr, n_fft)`` for
    the DC-dropped model bins), build over THOSE bin frequencies instead —
    shape (n_mels, len(bin_freqs)).
    """
    if f_max is None:
        f_max = sample_rate / 2
    if bin_freqs is not None:
        fft_freqs = bin_freqs.to(device=device, dtype=dtype)
        n_bins = fft_freqs.shape[0]
    else:
        n_bins = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_bins, device=device, dtype=dtype)
    # Mel scale
    def hz_to_mel(f):
        return 1127.0 * torch.log1p(f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (torch.exp(m / 1127.0) - 1.0)
    mel_min, mel_max = hz_to_mel(torch.tensor(f_min)), hz_to_mel(torch.tensor(f_max))
    mel_pts = torch.linspace(float(mel_min), float(mel_max), n_mels + 2, device=device, dtype=dtype)
    hz_pts = mel_to_hz(mel_pts)
    fb = torch.zeros(n_mels, n_bins, device=device, dtype=dtype)
    for m in range(n_mels):
        lo = hz_pts[m]
        ctr = hz_pts[m + 1]
        hi = hz_pts[m + 2]
        left = (fft_freqs - lo) / (ctr - lo).clamp_min(1e-8)
        right = (hi - fft_freqs) / (hi - ctr).clamp_min(1e-8)
        fb[m] = torch.clamp(torch.min(left, right), min=0.0)
    return fb


def upsample_control(frame_params: torch.Tensor, n_samples: int,
                     mode: str = "linear") -> torch.Tensor:
    """Upsample per-frame control params to per-sample (§6.4).

    Smooth interpolation is mandatory — ``repeat`` produces clicks.
    """
    if frame_params.dim() == 2:
        # (B, N_frames) -> (B, n_samples)
        return F.interpolate(frame_params.unsqueeze(1), size=n_samples,
                             mode="linear", align_corners=False).squeeze(1)
    elif frame_params.dim() == 3:
        # (B, C, N_frames) -> (B, C, n_samples)
        return F.interpolate(frame_params, size=n_samples, mode="linear",
                             align_corners=False)
    else:
        raise ValueError(f"expected 2D or 3D, got {frame_params.dim()}D")
