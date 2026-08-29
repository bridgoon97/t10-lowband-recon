"""Causal STFT / iSTFT for the fusion module (full-spectrum, 16 kHz 口径).

Reuses the audited pair in ``lowband.dsp.stft`` (``causal_stft`` /
``causal_istft``) for the BATCH path — they are the exact inverse of each other
(``tests/test_stft_roundtrip.py`` regression for review finding C: the OLD
``torch.istft(center=True)`` inverse gave rel≈1.3; ``causal_istft`` WOLA gives
rel≈1e-7).  We do NOT reimplement the pair — we build on it.

Why full spectrum (not the truncated ``keep_bins`` feature): the fusion
algorithm modifies bins 1..64 (31.25–2000 Hz, the fusion band) and PASSES
THROUGH bins 0 & 65..256 from ``S`` (the spec: "2 kHz 以上 S 直通").  So the
synthesis spectrum is the FULL ``S`` spectrum with bins 1..64 replaced by ``Y``.

Streaming (the deployable, bounded-state path):
  * ``StftStreamer.step(x_hop) -> spec_full`` — one hop in, one full complex
    frame out, carrying a ``win-hop`` sample tail.  Byte-identical to
    ``causal_stft`` frame t (same left-pad framing).
  * ``IstftStreamer.step(spec_full) -> (y_hop | None)`` + ``.flush()`` — a
    rolling ``win``-buffer WOLA that adds each frame at offset 0, emits ``hop``
    samples, and shifts left by ``hop``.  This reproduces ``causal_istft``'s
    per-frame accumulation in the SAME order ⇒ bit-identical per sample.  A
    ``win-hop`` sample prefix (the causal left-pad delay) is discarded during
    warmup, matching ``causal_istft``'s ``strip the left-pad prefix``.

Causality is structural: no look-ahead, no ``center=True``, no ``filtfilt``,
no whole-segment statistics.  ``lowband`` is imported read-only (we never
mutate it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from lowband.dsp.stft import StftConfig, causal_stft, causal_istft, get_window

from .config import FusionConfig


def _stft_cfg(cfg: FusionConfig) -> StftConfig:
    """Build the lowband StftConfig matching the fusion 口径 (full spectrum)."""
    return StftConfig(n_fft=cfg.n_fft, hop=cfg.hop, win=cfg.win,
                      window=cfg.window, center=False, keep_bins=cfg.keep_bins)


def get_win(cfg: FusionConfig, device=None, dtype=torch.float32) -> torch.Tensor:
    return get_window(cfg.window, cfg.win, device=device, dtype=dtype)


# ============================ BATCH (whole signal) ========================

def stft_batch(x: torch.Tensor, cfg: FusionConfig) -> torch.Tensor:
    """Full complex causal STFT, (B, num_bins, N).  ``x``: (B, T) float."""
    spec, _ = causal_stft(x.float(), _stft_cfg(cfg))
    return spec


def istft_batch(spec: torch.Tensor, cfg: FusionConfig,
               length: Optional[int] = None) -> torch.Tensor:
    """Inverse of ``stft_batch`` (exact).  ``spec``: (B, num_bins, N) complex."""
    return causal_istft(spec, _stft_cfg(cfg), length=length)


# ============================ STREAMING ===================================

@dataclass
class StftStreamer:
    """Per-hop causal STFT producing the FULL complex spectrum.

    State = the ``win-hop`` sample carry (left-pad buffer).  ``step(x_hop)``
    returns (spec_full (B, num_bins) complex, ) and mutates internal state.
    """
    cfg: FusionConfig
    tail: torch.Tensor | None = None   # (B, win-hop)

    def step(self, x_hop: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = x_hop.shape[0]
        if x_hop.shape[-1] != cfg.hop:
            # allow last short frame by zero-pad to hop (caller usually drops)
            if x_hop.shape[-1] < cfg.hop:
                x_hop = F.pad(x_hop, (0, cfg.hop - x_hop.shape[-1]))
            else:
                raise ValueError(f"x_hop len {x_hop.shape[-1]} != hop {cfg.hop}")
        if self.tail is None:
            self.tail = x_hop.new_zeros(B, cfg.win - cfg.hop)
        buf = torch.cat([self.tail, x_hop], dim=1)        # (B, win)
        self.last_buf = buf.clone()                         # for F0 reuse
        self.tail = buf[:, cfg.hop:]                       # carry win-hop
        w = get_win(cfg, device=x_hop.device, dtype=x_hop.dtype)
        spec = torch.fft.rfft(buf * w.unsqueeze(0), n=cfg.n_fft)  # (B, num_bins)
        return spec

    @staticmethod
    def n_frames_for(T: int, cfg: FusionConfig) -> int:
        """How many hop-frames a length-T signal yields (matching batch)."""
        left_pad = cfg.win - cfg.hop
        xp = T + left_pad
        return 1 + (xp - cfg.win) // cfg.hop


@dataclass
class IstftStreamer:
    """Per-hop causal iSTFT (rolling ``win``-buffer WOLA), exact match to
    ``causal_istft``.  Prefix ``win-hop`` samples (causal delay) are discarded
    during warmup; ``flush()`` emits the trailing tail.

    Each frame: ``irfft(spec, n_fft)[:, :win] * w`` added at acc offset 0, then
    ``acc[:hop]/norm[:hop]`` emitted (if past warmup), then shift-left by hop.
    """
    cfg: FusionConfig
    acc: torch.Tensor | None = None       # (B, win) OLA accumulator
    nrm: torch.Tensor | None = None       # (B, win) window-sq OLA norm
    warmup: int = 0                       # remaining prefix samples to discard

    def __post_init__(self):
        if self.warmup == 0 and self.cfg is not None:
            self.warmup = self.cfg.win - self.cfg.hop

    def step(self, spec_full: torch.Tensor) -> Optional[torch.Tensor]:
        cfg = self.cfg
        B = spec_full.shape[0]
        w = get_win(cfg, device=spec_full.device, dtype=torch.float32)
        if self.acc is None:
            self.acc = torch.zeros(B, cfg.win, device=spec_full.device,
                                  dtype=torch.float32)
            self.nrm = torch.zeros(B, cfg.win, device=spec_full.device,
                                   dtype=torch.float32)
        frame_full = torch.fft.irfft(spec_full, n=cfg.n_fft)   # (B, n_fft)
        frame_win = frame_full[:, :cfg.win] * w               # (B, win)
        wsq = (w * w)                                         # (win,)
        self.acc = self.acc + frame_win
        self.nrm = self.nrm + wsq
        # emit hop samples (finalized: no future frame touches [pos, pos+hop))
        y = self.acc[:, :cfg.hop] / self.nrm[:, :cfg.hop].clamp_min(cfg.eps)
        if self.warmup > 0:
            # prefix region — discard; consume warmup
            self.warmup = max(0, self.warmup - cfg.hop)
            out = None
        else:
            out = y
        # shift left by hop, zero-pad the freed tail
        self.acc = torch.cat([self.acc[:, cfg.hop:],
                              torch.zeros(B, cfg.hop, device=self.acc.device,
                                          dtype=self.acc.dtype)], dim=1)
        self.nrm = torch.cat([self.nrm[:, cfg.hop:],
                              torch.zeros(B, cfg.hop, device=self.nrm.device,
                                          dtype=self.nrm.dtype)], dim=1)
        return out

    def flush(self) -> torch.Tensor:
        """Emit remaining tail samples (norm-guarded).  Returns (B, tail)."""
        cfg = self.cfg
        if self.acc is None:
            return torch.zeros(0)
        # remaining valid samples = win-hop (the overlap tail), norm-guarded
        n = cfg.win - cfg.hop
        y = self.acc[:, :n] / self.nrm[:, :n].clamp_min(cfg.eps)
        return y
