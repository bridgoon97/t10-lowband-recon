"""Layer 2 · decision — ``w[f,t] = c_V · g_f0 · w_band · w_local``.

Four multiplicative factors, each independent and individually switchable
(ablation interface).  Plus the w time/freq smoothing.

All per-frame CAUSAL state (EMA / RANSAC-across-k, never across time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import FusionConfig
from .utils import (alpha_from_tau, asym_ema, causal_ema, soft_gate,
                     smooth1d, CohTracker, sigmoid)


# ---------------------------------------------------------------- c_V -------
@dataclass
class CV:
    """V credibility scalar (NOT a wear-state machine).  Components:
    ① V in-band energy (relative to baseline)  ② MSC(V,S)  ③ EQ residual.
    Non-symmetric hysteresis (升慢降快).  Change-point ⇒ force压低."""
    cfg: FusionConfig
    enabled: bool = True
    e_v_ema: Optional[torch.Tensor] = None     # running EMA of in-band |V|^2 (speech level)
    nf_ema: Optional[torch.Tensor] = None     # slow EMA of per-frame noise-floor estimate
    e_max_db: Optional[torch.Tensor] = None   # FR1-c mutation: legacy running-MAX (ratchet)
    c_v: float = 0.0
    coh: Optional[CohTracker] = None

    def __post_init__(self):
        self.a_e = alpha_from_tau(self.cfg.cv_energy_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_nf = alpha_from_tau(self.cfg.cv_nf_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_m = alpha_from_tau(self.cfg.cv_msc_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_rise = alpha_from_tau(self.cfg.cv_rise_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_fall = alpha_from_tau(self.cfg.cv_fall_tau_s, self.cfg.hop, self.cfg.sr)

    def step(self, v_spec: torch.Tensor, s_spec: torch.Tensor,
             eq_resid_db: torch.Tensor, reset_flag: bool = False) -> torch.Tensor:
        """Returns c_V (B,).  ``eq_resid_db``: per-bin |d−C| dB (B, Fb) or (B,)."""
        B, Fb = v_spec.shape
        lo, hi = self._band_bins()
        v_band = v_spec[:, lo:hi + 1]
        s_band = s_spec[:, lo:hi + 1]
        if not self.enabled:
            self.c_v = 1.0
            return torch.full((B,), 1.0, device=v_spec.device)
        if self.e_v_ema is None:
            self.e_v_ema = (v_band.abs() ** 2).mean(-1, keepdim=True)  # (B,1)
            self.coh = CohTracker(Fb, self.a_m, v_spec.device)
        # ① FR1: in-band SNR = V speech level (EMA) − V's OWN device noise floor.
        # Noise floor = per-frame low-quantile of per-bin |V|^2 (the between-
        # harmonic bins carry VPU device noise), slow time-EMA for stability.
        # Scales with recording gain (⇒ SNR invariant to gain — FR1-a); holds
        # during a loud event (⇒ no ratchet, c_V recovers — FR1-c); drops when
        # V's signal weakens relative to its device noise (⇒ c_V drops — M3/FR1-b).
        e_v = (v_band.abs() ** 2).mean(-1, keepdim=True)              # (B,1)
        self.e_v_ema = causal_ema(self.e_v_ema, e_v, self.a_e)
        e_db = 10.0 * torch.log10(self.e_v_ema.clamp_min(1e-10))
        bin_db = 10.0 * torch.log10((v_band.abs() ** 2).clamp_min(1e-12))  # (B, Fbb)
        nf_frame = torch.quantile(bin_db, self.cfg.cv_nf_quantile, dim=-1,
                                    keepdim=True)                       # (B,1)
        if self.nf_ema is None:
            self.nf_ema = nf_frame.clone()
        else:
            self.nf_ema = causal_ema(self.nf_ema, nf_frame, self.a_nf)
        snr_db = (e_db - self.nf_ema).clamp_min(0.0)
        if self.cfg.cv_legacy_abslevel:
            # FR1-a MUTATION: pure absolute level ("how loud is V") — no SNR,
            # no noise floor, no max.  Level-dependent ⇒ c_V changes with
            # recording gain ⇒ FR1-a (invariance) FAILS.  Keeps FR1-b (level
            # drops ⇒ c_V drops) and FR1-c (no ratchet) — breaks ONLY FR1-a.
            e_term = torch.sigmoid((e_db - self.cfg.cv_e_full_db)
                                    / self.cfg.cv_snr_scale_db).clamp(0, 1)
        elif self.cfg.cv_legacy_ratchet:
            # FR1-c MUTATION: the OLD running-MAX + fixed-floor e_term.  One
            # loud event raises e_max permanently ⇒ c_V depressed forever
            # (the ratchet this task removes).  Must FAIL FR1-c.
            if self.e_max_db is None:
                self.e_max_db = e_db.clone()
            else:
                self.e_max_db = torch.maximum(self.e_max_db, e_db)
            e_term = ((e_db - self.cfg.cv_e_floor_db) /
                      (self.e_max_db - self.cfg.cv_e_floor_db).clamp_min(1e-3)).clamp(0, 1)
        else:
            e_term = torch.sigmoid((snr_db - self.cfg.cv_snr_ref_db)
                                    / self.cfg.cv_snr_scale_db).clamp(0, 1)
        # ② MSC term
        msc = torch.stack([self.coh.update(v_spec[b], s_spec[b]) for b in range(B)])
        msc_band = msc[:, lo:hi + 1].mean(-1, keepdim=True)          # (B,1)
        m_term = msc_band.clamp(0, 1)
        # ③ EQ residual term (low residual ⟹ high)
        if eq_resid_db.dim() == 1:
            r = eq_resid_db.unsqueeze(-1)
        else:
            r = eq_resid_db[:, lo:hi + 1].mean(-1, keepdim=True)
        q_term = torch.exp(-r / 6.0).clamp(0, 1)                     # ~1 at 0dB
        # geometric mean ⇒ credibility = weakest-link flavour
        c_raw = (e_term * m_term * q_term).clamp(0, 1).sqrt()
        # non-symmetric hysteresis (升慢降快) — per-batch scalar
        prev = torch.full_like(c_raw, self.c_v)
        c_new = asym_ema(prev, c_raw, self.a_rise, self.a_fall)
        self.c_v = float(c_new.mean().item())
        if reset_flag:
            c_new = torch.full_like(c_new, self.cfg.cv_changepoint_floor)
            self.c_v = self.cfg.cv_changepoint_floor
        return c_new.squeeze(-1)                                    # (B,)

    def _band_bins(self):
        bz = self.cfg.sr / self.cfg.n_fft
        lo = max(1, int(self.cfg.eq_band_lo_hz / bz))
        hi = min(self.cfg.fusion_hi_bin, int(self.cfg.eq_band_hi_hz / bz))
        return lo, hi


# ---------------------------------------------------------------- g_f0 -----
@dataclass
class GF0:
    """F0-confidence soft gate.  g = soft_gate(f0_confidence=1−CMND).  No threshold."""
    cfg: FusionConfig
    enabled: bool = True
    flip: bool = False   # mutation: use CMND (1−conf) instead ⇒ M5 must fail

    def step(self, f0_conf: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.ones_like(f0_conf)
        c = f0_conf if not self.flip else (1.0 - f0_conf)
        return soft_gate(c, self.cfg.g_f0_gamma, self.cfg.g_f0_floor)


# ---------------------------------------------------------------- w_band ---
@dataclass
class WBand:
    """Band weight from online causal MSC(f) EMA, tapered to 0 toward 2 kHz."""
    cfg: FusionConfig
    enabled: bool = True
    fixed_curve: bool = False     # ablation
    coh: Optional[CohTracker] = None

    def step(self, v_spec: torch.Tensor, s_spec: torch.Tensor) -> torch.Tensor:
        B, Fb = v_spec.shape
        if not self.enabled:
            return torch.ones(B, Fb, device=v_spec.device)
        if self.fixed_curve or self.coh is None:
            self.coh = CohTracker(Fb, alpha_from_tau(self.cfg.wb_msc_tau_s,
                                                       self.cfg.hop, self.cfg.sr),
                                  v_spec.device)
        msc = torch.stack([self.coh.update(v_spec[b], s_spec[b]) for b in range(B)])
        if self.fixed_curve:
            msc = torch.ones_like(msc)
        taper = self._taper(Fb, v_spec.device)
        return (msc * taper).clamp(0, 1)

    def _taper(self, Fb: int, device) -> torch.Tensor:
        f = torch.arange(Fb, device=device)
        # monotone 1→0 over ~1 octave toward hi_bin; 1 below taper_start
        ts = self.cfg.boundary_taper_start_bin
        te = self.cfg.fusion_hi_bin
        t = torch.where(f <= ts, torch.ones_like(f),
                        ((te - f) / max(1, te - ts)).clamp(0, 1))
        return t


# ---------------------------------------------------------------- w_local --
@dataclass
class WLocal:
    """AC3 (B1): BAND-LEVEL const-⑤ gate.  ``w_local_band[b] =
    sigmoid((Pv_overall − P_band[b] − thr)/slope)`` where Pv_overall = V's
    overall level (100–800 Hz, VPU usable) and P_band = S's per-band level.
    A suppressed band (S low) ⇒ evi high ⇒ w→1 (use V); a surviving band
    ⇒ evi≈0 ⇒ w→0 (use S).  Bands >800 Hz ⇒ w=0 (CR3: V has no info there).

    🔴 Per-harmonic ①②③④⑤ DELETED (B0.5: per-harm info can't transfer
    VPU→mic; ① maxes 0.863 even at iso=100%).  Conclusions retained in README.
    ER1 (shuffle/const) still applies — at BAND granularity (perturb the band
    V-level reference)."""
    cfg: FusionConfig
    enabled: bool = True
    pure_band: bool = False       # ablation: w_local ≡ 1 (no detection)
    v_perturb: str = "none"       # ER1 band-level: "none"|"shuffle"|"const"

    def step(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
             f0_hz: torch.Tensor) -> torch.Tensor:
        """Returns w_local (B, Fb) per-bin (band-broadcast).  f0 unused (band-level)."""
        B, Fb = s_spec.shape
        if not self.enabled or self.pure_band:
            return torch.ones(B, Fb, device=s_spec.device)
        w = torch.zeros(B, Fb, device=s_spec.device)
        bz = self.cfg.sr / self.cfg.n_fft
        # V's overall level over the VPU-usable band (100–800 Hz)
        vlo = max(1, int(self.cfg.eq_band_lo_hz / bz))
        vhi = min(self.cfg.fusion_hi_bin, int(self.cfg.wl_v_usable_hi_hz / bz))
        Pv = 10.0 * torch.log10(v_spec[:, vlo:vhi + 1].abs().pow(2)
                                .mean(-1).clamp_min(1e-10))              # (B,) dB
        if self.v_perturb == "const":
            Pv = torch.full_like(Pv, Pv.median())
        # per band
        for blo, bhi, hi_hz in self._bands(Fb, bz):
            if hi_hz > self.cfg.wl_v_usable_hi_hz:
                continue   # CR3: no V info above 800 Hz ⇒ w_local = 0 there
            P_band = 10.0 * torch.log10(s_spec[:, blo:bhi + 1].abs().pow(2)
                                         .mean(-1).clamp_min(1e-10))    # (B,)
            if self.v_perturb == "shuffle":
                # ER1: destroy band↔V correspondence by shuffling Pv across batch
                Pv_b = Pv[torch.randperm(B)]
            else:
                Pv_b = Pv
            evi = Pv_b - P_band                                      # (B,)
            wb = torch.sigmoid((evi - self.cfg.wl_band_thr_db)
                                / self.cfg.wl_band_slope)             # (B,)
            w[:, blo:bhi + 1] = wb.unsqueeze(-1)
        return w.clamp(0, 1)

    def _bands(self, Fb: int, bz: float):
        """6 sub-bands 100–200/200–315/315–500/500–800/800–1250/1250–2000 Hz
        (aligned with the G-metric banding; bin ranges + each band's hi_hz)."""
        edges = [100, 200, 315, 500, 800, 1250, 2000]
        out = []
        for i in range(len(edges) - 1):
            lo = max(1, int(edges[i] / bz)); hi = min(Fb - 1, int(edges[i + 1] / bz))
            if lo <= hi:
                out.append((lo, hi, edges[i + 1]))
        return out


# ---------------------------------------------------------- asym smoother --
@dataclass
class AsymSmoother:
    """True non-symmetric one-sided EMA on w (升慢降快).  M4 isolates this."""
    cfg: FusionConfig
    enabled: bool = True
    symmetric: bool = False      # ablation
    prev: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.a_rise = alpha_from_tau(self.cfg.w_rise_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_fall = alpha_from_tau(self.cfg.w_fall_tau_s, self.cfg.hop, self.cfg.sr)
        self.a_sym = alpha_from_tau((self.cfg.w_rise_tau_s + self.cfg.w_fall_tau_s) / 2,
                                      self.cfg.hop, self.cfg.sr)

    def step(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w
        if self.prev is None:
            self.prev = w.clone()
            return w
        if self.symmetric:
            out = causal_ema(self.prev, w, self.a_sym)
        else:
            out = asym_ema(self.prev, w, self.a_rise, self.a_fall)
        self.prev = out.clone()
        return out
