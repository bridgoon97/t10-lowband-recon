"""Causal F0 estimation for the fusion module (16 kHz 口径).

Hard constraint (T13 §4): F0 must be CAUSAL — analysis window LEFT-aligned, only
history samples, NO symmetric/look-ahead padding.  The spec further requires
``F0 分析窗 ≤ STFT 窗长 (480 = 30 ms)`` to avoid algorithm delay beyond the
STFT.  We satisfy this by estimating F0 from the SAME 480-sample time-domain
buffer the STFT frame uses (the ``buf`` built in ``StftStreamer.step``).  ⇒
**0 extra delay** beyond the STFT.

Trade-off (reported, per spec "若做不到必须报告实际增加的时延"): with
frame_len = 480, ``tau_max = sr/f0_min`` is bounded by ≈ frame_len/2 ≈ 240,
giving ``f0_min ≈ 67 Hz``.  We set ``f0_min = 70 Hz``.  Adult voiced F0 ≥ ~85
Hz (male) to ~300 Hz (female/child) is covered; very-low-pitch speakers (<70 Hz)
are clipped — acceptable for the placeholder default; B-stage may widen the
window at the cost of added delay (documented there).

Direction (project has a前科 of reversing this): ``f0_confidence = 1 − CMND``
where CMND is the cumulative-mean-normalized difference function (0 = perfectly
periodic).  So HIGH confidence ⟹ voiced ⟹ we want MORE use of V.  This is the
direction M5 verifies, with a mutation sanity that flips it to ``CMND`` and
must fail.
"""
from __future__ import annotations

import torch

from .config import FusionConfig


def _yin_frame(frame: torch.Tensor, sr: float, tau_min: int, tau_max: int,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft YIN on one causal frame (B, L).  Returns (f0_hz (B,), conf (B,)).

    conf = clamp(1 - CMND_best, 0, 1):  HIGH ⟺ periodic/voiced.
    Never returns f0=0 and never applies a voiced/unvoiced threshold (soft path).
    """
    B, L = frame.shape
    n_fft = 1
    while n_fft < 2 * L:
        n_fft *= 2
    spec = torch.fft.rfft(frame, n=n_fft, dim=1)
    acf = torch.fft.irfft(spec * torch.conj(spec), n=n_fft, dim=1).real
    cum_sq = torch.cumsum(frame ** 2, dim=1)
    e0 = cum_sq[:, -1:]
    cum_sq_flip = torch.flip(cum_sq, dims=[1])
    e_left = cum_sq_flip[:, :tau_max + 1]
    e_right = torch.cat([e0, e0 - cum_sq[:, :tau_max]], dim=1)[:, :tau_max + 1]
    d = e_left + e_right - 2 * acf[:, :tau_max + 1]
    d[:, 0] = 1.0                                  # CMND convention
    cumsum = torch.cumsum(d[:, 1:], dim=1)
    cmnd = d[:, 1:] * torch.arange(1, tau_max + 1, device=frame.device,
                                   dtype=frame.dtype).unsqueeze(0) \
        / cumsum.clamp_min(1e-10)
    cmnd_range = cmnd[:, tau_min - 1:tau_max]      # (B, T)
    best = cmnd_range.argmin(dim=1)                 # (B,)
    b_idx = torch.arange(B, device=frame.device)
    v_best = cmnd_range[b_idx, best]
    tau = (tau_min + best).to(frame.dtype)
    f0 = (float(sr) / tau).clamp(float(sr) / tau_max, float(sr) / tau_min)
    conf = (1.0 - v_best).clamp(0.0, 1.0)
    return f0, conf


class F0Estimator:
    """Per-frame causal F0 from the STFT time-domain buffer."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.tau_min = max(1, int(cfg.sr / cfg.f0_max) - 1)
        self.tau_max = int(cfg.sr / cfg.f0_min) + 2

    def estimate(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``frame``: (B, win) time-domain (the STFT buf).  → (f0 (B,), conf (B,))."""
        return _yin_frame(frame.float(), self.cfg.sr, self.tau_min, self.tau_max)


def f0_batch(x: torch.Tensor, cfg: FusionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-STFT-frame causal F0 for a whole signal.  Mirrors causal_stft's
    left-pad unfold so each F0 frame uses EXACTLY the same samples the STFT
    frame uses (⇒ batch F0 == streaming F0 per frame).  Returns
    (f0 (B, N), conf (B, N)) with N = #STFT frames.
    """
    import torch.nn.functional as F
    B, T = x.shape
    left_pad = cfg.win - cfg.hop
    xp = F.pad(x.float(), (left_pad, 0), mode="constant")
    n_frames = 1 + (xp.shape[1] - cfg.win) // cfg.hop
    if n_frames < 1:
        xp = F.pad(xp, (0, cfg.win - xp.shape[1]))
        n_frames = 1
    frames = xp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)  # (B, N, win)
    est = F0Estimator(cfg)
    f0_list, conf_list = [], []
    for i in range(n_frames):
        f, c = est.estimate(frames[:, i, :])
        f0_list.append(f)
        conf_list.append(c)
    return torch.stack(f0_list, dim=1), torch.stack(conf_list, dim=1)
