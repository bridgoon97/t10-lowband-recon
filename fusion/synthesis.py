"""Layer 3 · synthesis — how V' and S are combined under w.

B1 (AC1) — MAGNITUDE-ONLY fusion, phase taken from the MIC (S):
    log|Y| = log|V'| + (1−w)·clip( log|S| − log|V'|, −Δ, +Δ )     Δ ≈ 10 dB
    ∠Y     = ∠S                              ← AC1 (was: weighted vector sum)

The CLIP is retained: when a harmonic is killed, log|S| is very negative;
without clip, w=0.9 would still let S drag |Y| down (M6 verifies the boundary).
🔴 Cost (priced by G7): the complex spectrum is NOT self-consistent
(|Y| from the log-clip rule, ∠Y from S) ⇒ ISTFT smearing.  This is the
accepted trade for dropping the complex-vector-cancellation energy dip (the
old complex-convex arm, ~−3 dB at 90° phase mismatch — AC1 removes it).

Comfort noise (AC unchanged): adaptive level (FR2), injected AFTER fusion,
NOT scaled by w.  Switchable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import FusionConfig
from .utils import alpha_from_tau, causal_ema


def logclip_mix(s_spec: torch.Tensor, v_spec: torch.Tensor, w: torch.Tensor,
                delta_db: float) -> torch.Tensor:
    """AC1: log-clip magnitude + MIC PHASE (∠S).  ``w``: (B, Fb).
    Phase from S (not the weighted vector sum) ⇒ no complex-vector-cancellation
    dip, but |Y|/∠Y not self-consistent (G7 prices the smearing cost)."""
    eps = 1e-8
    s_mag = s_spec.abs().clamp_min(eps)
    v_mag = v_spec.abs().clamp_min(eps)
    d = 20.0 * torch.log10(s_mag) - 20.0 * torch.log10(v_mag)   # log|S|-log|V'|
    d_clip = d.clamp(-delta_db, delta_db)
    logY = 20.0 * torch.log10(v_mag) + (1.0 - w) * d_clip
    magY = 10.0 ** (logY / 20.0)
    return magY * torch.exp(1j * torch.angle(s_spec))   # ∠Y = ∠S (AC1)


# AC1: complex-convex contrast arm REMOVED (was the M7 ~−3 dB dip candidate).
# M7 test retained as a HISTORICAL record (marked "candidate removed").
def complex_convex(*args, **kwargs):   # pragma: no cover - removed, kept for import compat
    raise NotImplementedError("complex_convex removed in AC1 (B1); see M7 history.")


@dataclass
class ComfortNoise:
    """Streaming min-trace / VAD-gated causal EMA noise谱形, ADAPTIVE level
    (relative to in-band speech RMS — a constant dB gap; FR2), injected AFTER
    fusion, NOT scaled by w.  FR2 mutation: cn_fixed_level_db=True reverts to
    the old fixed absolute level (covers quiet speech ⇒ FR2-a fails)."""
    cfg: FusionConfig
    enabled: bool = True
    floor_ema: Optional[torch.Tensor] = None
    speech_db_ema: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.a = alpha_from_tau(self.cfg.cn_ema_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_sp = alpha_from_tau(self.cfg.cn_speech_tau_s, self.cfg.hop, self.cfg.sr)

    def step(self, s_spec: torch.Tensor, y_spec: torch.Tensor,
             v_spec: torch.Tensor) -> torch.Tensor:
        """Add comfort noise to ``y_spec`` (fusion band bins 1..hi)."""
        if not self.enabled:
            return y_spec
        B, Fb = s_spec.shape
        lo, hi = 1, self.cfg.fusion_hi_bin
        # noise shape: causal EMA of |S| (min-trace flavour)
        s_mag = s_spec.abs()
        if self.floor_ema is None:
            self.floor_ema = s_mag.clone()
        self.floor_ema = causal_ema(self.floor_ema, s_mag, self.a)
        noise_shape = self.floor_ema / (self.floor_ema.max(-1, keepdim=True).values.clamp_min(1e-8) + 1e-8)
        # FR2: ADAPTIVE level = in-band speech RMS (EMA, dB) − constant gap.
        #   speech scaled −g dB ⇒ speech_db and level BOTH drop g ⇒ gap
        #   constant (FR2-a); ≥40 dB below speech ⇒ inaudible (FR2-b); level
        #   independent of w (FR2-c).  Mutation cn_fixed_level_db ⇒ fixed level.
        band = s_mag[:, lo:hi + 1]
        speech_db = 10.0 * torch.log10(band.pow(2).mean(-1).clamp_min(1e-10))  # (B,)
        if self.speech_db_ema is None:
            self.speech_db_ema = speech_db.clone()
        else:
            self.speech_db_ema = causal_ema(self.speech_db_ema, speech_db, self.a_sp)
        if self.cfg.cn_fixed_level_db:
            level_lin = torch.full_like(self.speech_db_ema,
                                          10.0 ** (self.cfg.cn_floor_db / 20.0))
        else:
            level_db = self.speech_db_ema - self.cfg.cn_below_speech_db
            level_lin = 10.0 ** (level_db / 20.0)                  # (B,)
        noise_mag = noise_shape * level_lin.unsqueeze(-1)
        # independent of w: add (not scaled by w), inject after fusion
        noise = noise_mag * torch.exp(1j * torch.angle(s_spec))
        out = y_spec.clone()
        out[:, lo:hi + 1] = out[:, lo:hi + 1] + noise[:, lo:hi + 1]
        return out


@dataclass
class Synthesis:
    cfg: FusionConfig
    comfort: Optional[ComfortNoise] = None

    def __post_init__(self):
        self.comfort = ComfortNoise(self.cfg, self.cfg.enable_comfort_noise)

    def step(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
             w: torch.Tensor) -> torch.Tensor:
        """Combine S and V' into Y over the FULL spectrum (bins 1..hi modified,
        bins 0 & hi+1.. copied from S).  AC1: magnitude log-clip + ∠S."""
        B, Fb = s_spec.shape
        lo, hi = 1, self.cfg.fusion_hi_bin
        y_band = logclip_mix(s_spec[:, lo:hi + 1], v_spec[:, lo:hi + 1],
                              w[:, lo:hi + 1], self.cfg.delta_db)
        y = s_spec.clone()
        y[:, lo:hi + 1] = y_band
        y = self.comfort.step(s_spec, y, v_spec)
        return y
