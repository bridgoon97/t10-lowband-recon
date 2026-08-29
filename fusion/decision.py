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
    e_v_ema: Optional[torch.Tensor] = None     # running baseline (max-ish)
    e_v_sq: Optional[torch.Tensor] = None       # current EMA of in-band |V|^2
    c_v: float = 0.0
    coh: Optional[CohTracker] = None

    def __post_init__(self):
        self.a_e = alpha_from_tau(self.cfg.cv_energy_tau_s, self.cfg.hop, self.cfg.sr)
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
        if self.e_v_sq is None:
            self.e_v_sq = (v_band.abs() ** 2).mean(-1, keepdim=True)  # (B,1)
            self.coh = CohTracker(Fb, self.a_m, v_spec.device)
        # ① energy term (monotone in V level — drives M3).  Baseline = running
        # MAX of E_db (causal) ⇒ e_term∈[0,1], 1 at the strongest V seen, lower
        # as V weakens; avoids fixed-reference saturation.
        e_v = (v_band.abs() ** 2).mean(-1, keepdim=True)             # (B,1)
        self.e_v_sq = causal_ema(self.e_v_sq, e_v, self.a_e)
        e_db = 10.0 * torch.log10(self.e_v_sq.clamp_min(1e-10))
        if not hasattr(self, "e_max_db") or self.e_max_db is None:
            self.e_max_db = e_db.clone()
        else:
            self.e_max_db = torch.maximum(self.e_max_db, e_db)
        e_term = ((e_db - self.cfg.cv_e_floor_db) /
                  (self.e_max_db - self.cfg.cv_e_floor_db).clamp_min(1e-3)).clamp(0, 1)
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
    """Per-harmonic survival detection (CORE).  RANSAC-style log-envelope fit
    ACROSS harmonic index k (causal — no time look-ahead).  Killed harmonics
    (P_S below envelope ⇒ r≪0) ⇒ w_local→1."""
    cfg: FusionConfig
    enabled: bool = True
    pure_band: bool = False       # ablation: w_local ≡ 1 (no detection)
    v_fallback: bool = True
    valley: bool = True

    def step(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
             f0_hz: torch.Tensor) -> torch.Tensor:
        """Returns w_local (B, Fb) per-bin (harmonic bins set, between→interp,
        noise bins → 0)."""
        B, Fb = s_spec.shape
        if not self.enabled or self.pure_band:
            return torch.ones(B, Fb, device=s_spec.device)
        w = torch.zeros(B, Fb, device=s_spec.device)
        bz = self.cfg.sr / self.cfg.n_fft
        for b in range(B):
            f0 = float(f0_hz[b]) if f0_hz.dim() else float(f0_hz)
            if f0 <= 0:
                continue
            kb = self._harm_bins(f0, Fb, bz)      # list of (k, bin), fusion band only
            if len(kb) < 3:
                continue
            P = torch.tensor([20 * torch.log10(s_spec[b, binidx].abs().clamp_min(1e-8))
                              for k, binidx in kb])
            Pv = torch.tensor([20 * torch.log10(v_spec[b, binidx].abs().clamp_min(1e-8))
                               for k, binidx in kb])
            # keep only REAL harmonics (above noise floor: P > max(P) − 80 dB).
            keep_real = P > (P.max() - 80.0)
            if keep_real.sum() < 3:
                continue
            Pr = P[keep_real]
            Pvr = Pv[keep_real]
            wl_h = self._detect(Pr, Pvr)      # per-harmonic w_local ∈[0,1] (killed→1)
            # freq smoothing in the HARMONIC DOMAIN (across k), NOT bin-domain
            # (bin-domain would抹掉 the harmonic/valley distinction just made).
            if self.cfg.enable_harm_freq_smooth and not self.cfg.use_bin_freq_smooth:
                wl_h = smooth1d(wl_h, 2 * self.cfg.w_k_smooth + 1)
            elif self.cfg.use_bin_freq_smooth:
                pass  # ablation: do bin-domain later (identity here)
            real_bins = [kb[i][1] for i in range(len(kb)) if keep_real[i]]
            for i, binidx in enumerate(real_bins):
                w[b, binidx] = wl_h[i]
            # interpolate between consecutive real harmonics
            for i in range(len(real_bins) - 1):
                a, c = real_bins[i], real_bins[i + 1]
                if c > a + 1:
                    w[b, a + 1:c] = torch.linspace(float(wl_h[i]), float(wl_h[i + 1]),
                                                     c - a - 1, device=s_spec.device)
            # ablation: bin-domain freq smoothing (the WRONG way)
            if self.cfg.enable_harm_freq_smooth and self.cfg.use_bin_freq_smooth:
                w[b] = smooth1d(w[b], 2 * self.cfg.w_k_smooth + 1)
        return w.clamp(0, 1)

    def _harm_bins(self, f0: float, Fb: int, bz: float):
        out = []
        for k in range(1, 200):
            f = k * f0
            if f >= self.cfg.sr / 2:
                break
            b = int(round(f / bz))
            # fusion band only: bins 1..fusion_hi_bin (above 2 kHz S passes through)
            if 1 <= b <= self.cfg.fusion_hi_bin:
                out.append((k, b))
        return out

    def _ransac(self, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exhaustive-pair RANSAC of the log-envelope E[k] vs position k.

        For every pair (i,j) fit a line, count inliers within a FIXED band
        ``wl_inlier_db`` (|resid|<band — NOT σ·std, which the −60 dB killed
        points inflate and degenerate to 'pick the highest line').  Keep the
        pair with the most inliers (the survivor pair ⇒ survivor line ⇒ ~all
        survivors inlier, killed far outside), then refit LSQ on the inliers.
        Deterministic (no RNG); n≤~13 in-band harmonics ⇒ ≤78 pairs, cheap.
        This is the spec's '全体拟合→剔除低于拟合→重拟合 (RANSAC 式 2–3 轮)'
        with a fixed band; the V-fallback covers ≥60 % kill.
        """
        n = len(P)
        if n < 3:
            return P.clone(), torch.ones(n, dtype=torch.bool, device=P.device)
        k = torch.arange(n, dtype=P.dtype, device=P.device)
        best_in = None; best_count = -1
        for i in range(n):
            for j in range(i + 1, n):
                slope = (P[j] - P[i]) / (j - i)
                offset = P[i] - slope * i
                resid = P - (slope * k + offset)
                inl = resid.abs() < self.cfg.wl_inlier_db
                c = int(inl.sum())
                if c > best_count:
                    best_count = c; best_in = inl
        survivors = best_in if best_in is not None else torch.ones(n, dtype=torch.bool, device=P.device)
        if survivors.sum() >= 2:
            ks, Ps = k[survivors], P[survivors]
            A = torch.stack([ks, torch.ones_like(ks)], -1)
            sol = torch.linalg.lstsq(A, Ps.unsqueeze(-1)).solution.squeeze(-1)
            E = sol[0] * k + sol[1]
        else:
            E = P.clone(); survivors = torch.ones(n, dtype=torch.bool, device=P.device)
        return E, survivors

    # ---- B0 envelope-model methods (①②③④), each independently switchable ----
    def _detect(self, P: torch.Tensor, Pv: torch.Tensor) -> torch.Tensor:
        """Per-harmonic w_local ∈[0,1] (killed→1) as the PRODUCT of the active
        methods (product ⇒ all must agree ⇒ low FAR; 'wrong-use-V-is-fabrication'
        ⇒ prioritize FAR over recall).  Methods:
          ① local-median baseline  ② abrupt-drop signature (max-neighbor)
          ③ absolute-floor gate   ④ V-envelope always-on weak evidence."""
        import numpy as _np
        w = torch.ones_like(P)
        if self.cfg.wl_use_local_median:                       # ①
            E = self._local_median(P)
            r = P - E
            w = w * torch.sigmoid(-(r + self.cfg.wl_r_kill_db) / self.cfg.wl_slope)
        if self.cfg.wl_use_abrupt_drop:                        # ② (most essential)
            drop = P - self._max_neighbor(P)                   # neg & large ⇒ killed
            w = w * torch.sigmoid(-(drop + self.cfg.wl_drop_thr_db) / self.cfg.wl_drop_slope)
        if self.cfg.wl_use_abs_gate:                           # ③ (key FAR suppressor, RELATIVE to frame peak)
            gate = torch.sigmoid((P.max() - self.cfg.wl_abs_headroom_db - P) / self.cfg.wl_abs_slope)
            w = w * gate
        if self.cfg.wl_use_v_envelope:                         # ④ weak, always-on
            w = w * torch.sigmoid((Pv - P - 3.0) / self.cfg.wl_v_env_slope)
        if not (self.cfg.wl_use_local_median or self.cfg.wl_use_abrupt_drop
                or self.cfg.wl_use_abs_gate or self.cfg.wl_use_v_envelope):
            w = torch.ones_like(P)   # no method ⇒ pure-band (ablation)
        return w.clamp(0, 1)

    def _local_median(self, P: torch.Tensor) -> torch.Tensor:
        """① local median baseline over k±window (reflect-padded).  Follows the
        slowly-varying formant envelope; a killed harmonic (sharp drop) sits
        far below its local median ⇒ r≪0."""
        import numpy as _np
        win = self.cfg.wl_local_window
        if len(P) < 2 * win + 1 or win < 1:
            return P.clone()
        a = P.numpy()
        pad = _np.pad(a, win, mode="reflect")
        E = _np.array([_np.median(pad[i:i + 2 * win + 1]) for i in range(len(a))])
        return torch.from_numpy(E).to(P.dtype)

    def _max_neighbor(self, P: torch.Tensor) -> torch.Tensor:
        """② max of k±window neighbors (excluding self).  Killed (pressed to a
        fixed floor) drops steeply below its HIGHEST neighbor; a smooth formant
        valley drops only mildly relative to neighbors ⇒ separable."""
        import numpy as _np
        win = self.cfg.wl_local_window
        n = len(P)
        if n < 2:
            return P.clone()
        a = P.numpy()
        out = _np.full(n, -_np.inf)
        for i in range(n):
            lo = max(0, i - win); hi = min(n, i + win + 1)
            neigh = _np.concatenate([a[lo:i], a[i + 1:hi]]) if hi - lo > 1 else a[lo:i]
            out[i] = neigh.max() if len(neigh) else a[i]
        return torch.from_numpy(out).to(P.dtype)


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
