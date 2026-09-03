"""T13-N1 · shaping gain ``G[f,t] = a[t] + s[t]·f̃`` — TWO frequency DOF (level +
spectral tilt), fitted ONLY on the S-trusted band (fit_lo..fit_hi = 100–800 Hz;
S-hole bands — especially 1–2 kHz — are structurally NOT read: they are covered
by extrapolation of the 2-DOF line.  Stability comes from not reading that
region, not from longer smoothing).

Time dimension: two SEPARATE causal states
  a[t] fast (shape_a_tau_s ≈ 80 ms) — follows syllable on/offsets;
  s[t] slow (shape_s_tau_s ≈ 2 s)  — wearing / transfer-function drift.
A unit test proves the time constants differ; ``n1_mutation_noncausal_a`` is a
MUTATION that makes a[t] read one future frame (must fail the causality test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import FusionConfig
from .utils import alpha_from_tau, causal_ema


@dataclass
class ShapeGain:
    cfg: FusionConfig
    a_state: Optional[torch.Tensor] = None    # (B,) dB — fast level
    s_state: Optional[torch.Tensor] = None    # (B,) dB — slow tilt
    f_tilde: Optional[torch.Tensor] = None    # (Fb,) fit-band-normalized freq

    def __post_init__(self):
        self.a_alpha = alpha_from_tau(self.cfg.shape_a_tau_s, self.cfg.hop, self.cfg.sr)
        self.s_alpha = alpha_from_tau(self.cfg.shape_s_tau_s, self.cfg.hop, self.cfg.sr)

    def _fit_axis(self, Fb: int, device):
        if self.f_tilde is None:
            bz = self.cfg.sr / self.cfg.n_fft
            f = torch.arange(Fb, device=device) * bz
            lo = self.cfg.shape_fit_lo_hz
            hi = self.cfg.shape_fit_hi_hz
            self.f_tilde = ((f - lo) / (hi - lo)).clamp(-0.5, 2.0)
        return self.f_tilde

    def step(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
             s_spec_next: Optional[torch.Tensor] = None):
        """Returns (G (B,Fb), a (B,), s (B,)).  ``s_spec_next`` is used ONLY by
        the noncausal MUTATION (a[t] then reads one future frame)."""
        B, Fb = s_spec.shape
        device = s_spec.device
        f_t = self._fit_axis(Fb, device)
        bz = self.cfg.sr / self.cfg.n_fft
        flo = max(1, int(self.cfg.shape_fit_lo_hz / bz))
        fhi = max(flo + 1, min(Fb - 1, int(self.cfg.shape_fit_hi_hz / bz)))
        tgt_all = (20.0 * torch.log10(s_spec.abs().clamp_min(1e-8))
                   - 20.0 * torch.log10(v_spec.abs().clamp_min(1e-8)))   # (B,Fb)
        tgt = (tgt_all if s_spec_next is None else
               (20.0 * torch.log10(s_spec_next.abs().clamp_min(1e-8))
                - 20.0 * torch.log10(v_spec.abs().clamp_min(1e-8))))
        # least-squares line over the FIT band only (S holes excluded structurally)
        t_fit = tgt[:, flo:fhi + 1]                       # (B, K)
        f_fit = f_t[flo:fhi + 1]
        a_tgt = t_fit.mean(dim=-1)                        # (B,) level
        fm = f_fit.mean()
        denom = float(((f_fit - fm) ** 2).sum()) + 1e-9
        s_tgt = ((t_fit - a_tgt.unsqueeze(-1)) * (f_fit - fm).unsqueeze(0)).sum(-1) / denom
        if self.a_state is None:
            self.a_state = a_tgt.clone()
            self.s_state = s_tgt.clone()
        else:
            self.a_state = causal_ema(self.a_state, a_tgt, self.a_alpha)  # fast
            self.s_state = causal_ema(self.s_state, s_tgt, self.s_alpha)  # slow
        G = self.a_state.unsqueeze(-1) + self.s_state.unsqueeze(-1) * f_t.unsqueeze(0)
        return G, self.a_state.clone(), self.s_state.clone()
