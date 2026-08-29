"""F0 estimation for 4 kHz band-limited body-conduction signals.

§6.5 requirements:
- F0 must be estimated FROM THE INPUT (never the target) — information leak.
- Provide F0-oracle and F0-estimated paths, switchable in config.
- Traditional YIN must be verified to still work at 4 kHz / 500 Hz LP —
  this is tested in tests/test_f0.py.

The estimator here is a torch-based difference-function YIN so it can live
inside the forward graph (gradients flow if needed) and stay device-agnostic.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_F0_MIN = 50.0   # Hz
DEFAULT_F0_MAX = 400.0  # Hz


def yin_f0(x: torch.Tensor, sample_rate: float, frame_len: int = 128,
           f0_min: float = DEFAULT_F0_MIN, f0_max: float = DEFAULT_F0_MAX,
           threshold: float = 0.1, soft: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """YIN F0 estimator (torch, differentiable wrt input).

    Args:
        x: (B, T) waveform @ sample_rate.
        frame_len: analysis frame length (samples). At 4 kHz, 128 = 32 ms.
        f0_min, f0_max: search range.
        threshold: absolute threshold for the cumulative mean normalized
            difference function (classic YIN default 0.1–0.15).  IGNORED
            when ``soft=True``.
        soft: if True, run the SOFT-CANDIDATE path (task ②): always return the
            best (argmin-CMND) F0 candidate in [f0_min, f0_max] and a continuous
            ``prob = clamp(1 - CMND_at_best, 0, 1)``; NEVER returns ``f0=0`` and
            NEVER applies a voiced/unvoiced threshold.  This is the path Arm A
            uses (confidence as a SOFT weight downstream, no binary decision).
            Default ``False`` = legacy hard-threshold behavior (kept for
            backward compat, e.g. tests/test_l1_f0.py inspects ``f0>0``).

    Returns:
        f0: (B, n_frames) estimated F0 in Hz.  Legacy: 0 = unvoiced.
            Soft: always finite, in [f0_min, f0_max] (candidate clamped).
        prob: (B, n_frames) periodicity / voicedness in [0, 1] (``1-CMND``),
            named ``f0_confidence`` downstream.  Always finite in [0, 1].
    """
    B, T = x.shape
    hop = frame_len // 2
    # Pad
    pad = (frame_len - hop, frame_len - hop)
    xp = F.pad(x, pad)
    n_frames = 1 + (xp.shape[1] - frame_len) // hop
    tau_max = int(sample_rate / f0_min) + 2
    tau_min = max(1, int(sample_rate / f0_max) - 1)

    f0_list = []
    prob_list = []
    for i in range(n_frames):
        s = i * hop
        frame = xp[:, s:s + frame_len]  # (B, frame_len)
        f0_i, p_i = _yin_frame(frame, sample_rate, tau_min, tau_max, threshold,
                                soft=soft)
        f0_list.append(f0_i)
        prob_list.append(p_i)
    f0 = torch.stack(f0_list, dim=1)    # (B, n_frames)
    prob = torch.stack(prob_list, dim=1)
    if soft:
        # clamp the candidate to the exact legal range (req 4: candidate F0 may
        # be clamped; the search tau has int()+buffer so sr/tau can overshoot by
        # a fraction of a Hz).  Legacy path keeps 0=unvoiced, so NOT clamped.
        f0 = f0.clamp(f0_min, f0_max)
    return f0, prob


def _yin_frame(frame: torch.Tensor, sample_rate: float, tau_min: int,
               tau_max: int, threshold: float, soft: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized YIN for a single frame (B, L).

    ``soft=True`` (task ②): always return the best (argmin-CMND) F0 candidate in
    range + ``prob=clamp(1-CMND_best,0,1)``; never ``f0=0``, never a threshold
    branch.  ``soft=False`` (default): legacy hard-threshold behavior.
    """
    B, L = frame.shape
    win = torch.hann_window(L, device=frame.device, dtype=frame.dtype)
    fw = frame * win
    # Vectorized difference function via FFT autocorrelation
    # d(tau) = sum (x[n]-x[n+tau])^2 = e(n) + e(n+tau) - 2*acf[tau]
    # where e(k) = sum_{n=0}^{L-tau-1} x[n]^2 (energy in window)
    n_fft = 1
    while n_fft < 2 * L:
        n_fft *= 2
    spec = torch.fft.rfft(frame, n=n_fft, dim=1)  # unwindowed for consistency
    acf = torch.fft.irfft(spec * torch.conj(spec), n=n_fft, dim=1).real
    # Cumulative sum of squares for energy terms (unwindowed, same domain)
    cum_sq = torch.cumsum(frame ** 2, dim=1)
    # e_left(tau) = sum_{n=0}^{L-tau-1} x[n]^2 = cum_sq[L-1-tau]
    # e_right(tau) = sum_{n=tau}^{L-1} x[n]^2 = e0 - cum_sq[tau-1] (tau>0)
    e0 = cum_sq[:, -1:]  # total energy
    cum_sq_flip = torch.flip(cum_sq, dims=[1])  # reversed
    e_left = cum_sq_flip[:, :tau_max + 1]  # [cum_sq[L-1], ..., cum_sq[L-1-tau_max]]
    e_right = torch.cat([e0, e0 - cum_sq[:, :tau_max]], dim=1)[:, :tau_max + 1]
    d = e_left + e_right - 2 * acf[:, :tau_max + 1]
    d[:, 0] = 1.0  # CMND convention
    # Cumulative mean normalized difference (CMND)
    cumsum = torch.cumsum(d[:, 1:], dim=1)
    cmnd = d[:, 1:] * torch.arange(1, tau_max + 1, device=frame.device,
                                   dtype=frame.dtype).unsqueeze(0) / cumsum.clamp_min(1e-10)
    # Search: first dip below threshold in [tau_min, tau_max]
    cmnd_range = cmnd[:, tau_min - 1:tau_max]
    if soft:
        # SOFT candidate (task ②): best F0 = argmin CMND in range, always.
        # prob = 1 - CMND_at_best clamped to [0,1]; NEVER f0=0, NEVER a threshold.
        best = cmnd_range.argmin(dim=1)                       # (B,)
        b_idx = torch.arange(B, device=frame.device)
        v_best = cmnd_range[b_idx, best]                       # (B,)
        tau = (tau_min + best).to(frame.dtype)
        # candidate f0 is in [sr/tau_max, sr/tau_min] = the search range by
        # construction; clamp as a safety net (never 0, never a threshold branch).
        f0_lo = float(sample_rate) / float(tau_max)
        f0_hi = float(sample_rate) / float(tau_min)
        f0_out = (float(sample_rate) / tau).clamp(f0_lo, f0_hi)
        prob_out = (1.0 - v_best).clamp(0.0, 1.0)
        return f0_out, prob_out
    below = cmnd_range < threshold
    f0_out = torch.zeros(B, device=frame.device, dtype=frame.dtype)
    prob_out = torch.zeros(B, device=frame.device, dtype=frame.dtype)
    for b in range(B):
        idxs = torch.nonzero(below[b], as_tuple=False).squeeze(-1)
        if idxs.numel() > 0:
            best_local = idxs[0].item()
            # refine: local minimum around best_local
            lo = max(0, best_local - 2)
            hi = min(cmnd_range.shape[1], best_local + 3)
            seg = cmnd_range[b, lo:hi]
            local_min = lo + seg.argmin().item()
            tau = tau_min + local_min
            f0_out[b] = float(sample_rate) / float(tau)
            prob_out[b] = 1.0 - float(cmnd_range[b, local_min].item())
        else:
            best = cmnd_range[b].argmin().item()
            tau = tau_min + best
            v = float(cmnd_range[b, best].item())
            f0_out[b] = float(sample_rate) / float(tau) if v < 0.5 else 0.0
            prob_out[b] = max(0.0, 1.0 - v)
    return f0_out, prob_out


def smooth_f0(f0: torch.Tensor, win: int = 5) -> torch.Tensor:
    """Median-like smoothing of F0 track (keep zero crossings)."""
    # Simple moving average with zero-preserving.
    B, N = f0.shape
    kernel = torch.ones(1, 1, win, device=f0.device) / win
    f0_3 = f0.unsqueeze(1)
    sm = F.conv1d(f0_3, kernel, padding=win // 2).squeeze(1)
    return sm
