"""Fusion pipeline (layer 1 + 2 + 3) — batch and streaming, both CAUSAL.

  ``Fusion.process_batch(S, V) -> Y``  — vectorized STFT (causal_stft), per-frame
       decision loop (sequential EMAs), vectorized iSTFT (causal_istft).
  ``FusionStreamer.stream_step(s_hop, v_hop) -> y_hop | None`` — per-hop STFT +
       per-frame decision + per-hop iSTFT, bounded state.

Both paths share ``FusionCore.process_frame`` (the per-frame decision) and use
SEPARATE layer-object instances, so the only batch↔streaming difference is the
STFT/iSTFT engine — and those are bit-identical per frame (verified in
``tests/test_t13_streaming.py``: streaming-vs-batch diff == 0.0 interior).

S = stage-2 proxy; V = (delay-compensated, EQ-aligned) VPU; X (clean FF) NEVER
enters the algorithm path (static-checked).  Above 2 kHz (bins 65+) S passes
through verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import FusionConfig
from .stft import stft_batch, istft_batch, StftStreamer, IstftStreamer, get_win
from .f0 import F0Estimator
from .align import EQAlign
from .decision import CV, GF0, WBand, WLocal, AsymSmoother
from .synthesis import Synthesis
from .utils import alpha_from_tau, causal_ema


@dataclass
class NoiseFloor:
    """Causal min-trace-ish noise floor (slow EMA of |S|) for local-SNR."""
    cfg: FusionConfig
    floor: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.a = alpha_from_tau(2.0, self.cfg.hop, self.cfg.sr)

    def step(self, s_mag: torch.Tensor) -> torch.Tensor:
        if self.floor is None:
            self.floor = s_mag.clone()
        # min-trace flavour: EMA biased toward the smaller of (prev, new)
        new = torch.minimum(self.floor, s_mag) if self.floor.shape == s_mag.shape else s_mag
        self.floor = (1 - self.a) * self.floor + self.a * new
        return self.floor


class FusionCore:
    """Per-frame decision shared by batch and streaming (separate instances)."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.eq = EQAlign(cfg, enabled=cfg.enable_eq,
                          changepoint_enabled=cfg.enable_eq_changepoint)
        self.cv = CV(cfg, enabled=cfg.enable_c_V)
        self.gf0 = GF0(cfg, enabled=cfg.enable_g_f0)
        self.wband = WBand(cfg, enabled=cfg.enable_w_band,
                           fixed_curve=cfg.use_w_band_fixed_curve)
        self.wlocal = WLocal(cfg, enabled=cfg.enable_w_local,
                             pure_band=cfg.use_w_local_pure_band,
                             v_perturb=cfg.wl_v_perturb)
        self.smooth = AsymSmoother(cfg, enabled=cfg.enable_asym_smooth,
                                   symmetric=cfg.use_symmetric_smooth)
        self.synth = Synthesis(cfg)
        self.nf = NoiseFloor(cfg)
        self.f0est = F0Estimator(cfg)
        self.w_history = []   # per-frame w (B, Fb) — for M5 propagation diagnostics

    def process_frame(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
                      s_buf: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """``s_spec``/``v_spec``: (B, Fb) complex (full spectrum); ``s_buf``:
        (B, win) time-domain frame (for F0).  Returns (y_spec (B,Fb), w (B,Fb))."""
        cfg = self.cfg
        # F0 from the SAME buf the STFT used (0 extra delay).  No external F0
        # injection — tests needing injected F0 use a test-only subclass.
        f0, conf = self.f0est.estimate(s_buf)                  # (B,),(B,)
        # local SNR
        s_mag = s_spec.abs()
        floor = self.nf.step(s_mag)
        snr = (20.0 * torch.log10(s_mag.clamp_min(1e-8) /
                                   floor.clamp_min(1e-8))).mean(-1)   # (B,)
        # Layer 1
        v_prime, startup_floor, reset_flag = self.eq.step(s_spec, v_spec, snr, conf)
        eq_resid = (20 * torch.log10(s_spec.abs().clamp_min(1e-8)) -
                    20 * torch.log10(v_spec.abs().clamp_min(1e-8))
                    - self.eq.C).abs().mean(-1) if self.eq.C is not None else torch.zeros_like(snr)
        # Layer 2
        c_v = self.cv.step(v_prime, s_spec, eq_resid, bool(reset_flag.any()))
        g = self.gf0.step(conf)
        w_band = self.wband.step(v_prime, s_spec)
        w_local = self.wlocal.step(s_spec, v_prime, f0)
        w_raw = (c_v.unsqueeze(-1) * g.unsqueeze(-1) * w_band * w_local)
        floor_w = torch.maximum(startup_floor, reset_flag.float())
        w = w_raw * (1.0 - floor_w).unsqueeze(-1)
        w = self.smooth.step(w)
        self.w_history.append(w.detach().clone())
        # Layer 3
        y_spec = self.synth.step(s_spec, v_prime, w)
        # 2 kHz boundary: taper w to 0 by hi_bin already in w_band taper; pass-thru
        # of bins > hi handled in synth (S copied).
        return y_spec, w


# ============================ BATCH =======================================

class Fusion:
    """Batch (whole-signal) fusion."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.core = FusionCore(cfg)

    def process_batch(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """S, V: (B, T) → Y (B, T).  S=stage-2 proxy, V=VPU.  F0 always estimated.
        AC1: no delay comp (phase taken from S)."""
        s = s.float(); v = v.float()
        cfg = self.cfg
        spec_s = stft_batch(s, cfg)          # (B, Fb, N)
        spec_v = stft_batch(v, cfg)
        N = spec_s.shape[-1]
        # left-pad unfold of S frames (same as causal_stft) for s_buf per frame
        left_pad = cfg.win - cfg.hop
        sp = F.pad(s, (left_pad, 0), mode="constant")
        frames_s = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)  # (B, N, win)
        y_frames = []
        for t in range(N):
            y_t, _ = self.core.process_frame(spec_s[:, :, t], spec_v[:, :, t],
                                              frames_s[:, t, :])
            y_frames.append(y_t)
        y_spec = torch.stack(y_frames, dim=-1)            # (B, Fb, N)
        return istft_batch(y_spec, cfg, length=s.shape[-1])


# ============================ STREAMING ===================================

class FusionStreamer:
    """Per-hop streaming fusion (bounded state)."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.core = FusionCore(cfg)
        self.sfr_s = StftStreamer(cfg)
        self.sfr_v = StftStreamer(cfg)
        self.isr = IstftStreamer(cfg)

    def stream_step(self, s_hop: torch.Tensor, v_hop: torch.Tensor
                    ) -> Optional[torch.Tensor]:
        s_hop = s_hop.float(); v_hop = v_hop.float()
        s_spec = self.sfr_s.step(s_hop)
        v_spec = self.sfr_v.step(v_hop)
        s_buf = self.sfr_s.last_buf
        y_spec, _ = self.core.process_frame(s_spec, v_spec, s_buf)
        return self.isr.step(y_spec)

    def flush(self) -> torch.Tensor:
        return self.isr.flush()
