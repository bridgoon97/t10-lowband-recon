"""T13-N1 · voiced routing ``g_v[t]`` — estimated from the RAW VPU (never from
S, never from the EQ-aligned V′), reusing the shared causal F0 contract
(``f0_confidence = 1 − CMND``, higher = more voiced-confident).

``g_v_raw = conf^gamma_v`` — soft mapping, NO hard threshold; ``gamma_v`` is the
single shape parameter.  Then ASYMMETRIC smoothing with two SEPARATE time
constants:

  rise SLOW (gv_rise_tau_s = 100 ms) — a false-voiced decision opens Δ↓ and
    crushes a consonant (disaster), so trust must accumulate slowly;
  fall FAST (gv_fall_tau_s = 20 ms) — a false-unvoiced decision only forgoes
    some valley suppression (mild), so release must be immediate.

Direction is unit-tested separately (this project has a history of flipping
it); ``gv_flip`` is the mutation that inverts to CMND and must fail that test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import FusionConfig
from .f0 import F0Estimator
from .utils import alpha_from_tau, asym_ema


@dataclass
class VoicingGate:
    cfg: FusionConfig
    prev: float = 0.0
    _est: Optional[F0Estimator] = None

    def __post_init__(self):
        self.a_rise = alpha_from_tau(self.cfg.gv_rise_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_fall = alpha_from_tau(self.cfg.gv_fall_tau_s, self.cfg.hop, self.cfg.sr)

    def step(self, v_buf: torch.Tensor) -> float:
        """``v_buf``: (B, win) RAW VPU time frame (the STFT buffer).  Returns the
        smoothed scalar g_v (batch-mean — p/g_v are frame scalars this batch)."""
        if self._est is None:
            self._est = F0Estimator(self.cfg)
        _, conf = self._est.estimate(v_buf)              # conf = 1 − CMND ∈ [0,1]
        c = (1.0 - conf) if self.cfg.gv_flip else conf   # gv_flip = direction MUTATION
        g_raw = float(c.clamp(0.0, 1.0).pow(self.cfg.gv_gamma).mean())
        g = float(asym_ema(torch.tensor([self.prev]),
                           torch.tensor([g_raw]),
                           self.a_rise, self.a_fall)[0])
        self.prev = g
        if self.cfg.gv_override is not None:             # test-side force (I2)
            return float(self.cfg.gv_override)
        return g
