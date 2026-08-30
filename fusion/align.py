"""Layer 1 · alignment (slow).

B1 (AC1) removed ``DelayComp`` (time-delay comp) entirely — the measured
0–10 sample (≤0.6 ms) inter-channel delay only mattered for PHASE coherence,
which AC1 no longer uses (phase taken from S; magnitude envelope needs only
frame-level alignment, 0.6 ms ≪ 10 ms hop).  GCC-PHAT constants deleted too.

B1 (AC2) changed ``EQAlign`` from continuous-adaptive to FROZEN: converge
once on donning → FREEZE (stop EMA updates) → event-triggered re-estimate
(MSC drop or EQ-residual sustained over-threshold; changepoint role is now
"watchdog for freeze failure", not "accelerate adaptive").  Rationale: deploy-
time C[f] is estimated from stage-2's DEGRADED output ⇒ continuous adaptive
pours degradation into the EQ; a frozen estimate at a good moment is cleaner.
Reviewer's 0624 variance: within-wear 1.34 dB ≪ between-wear 5.39 ≈
between-speaker 5.13 ⇒ continuous adaptive buys ~1.3 dB (1/4 of the 5.4 dB
wear problem) ⇒ over-engineering.  Ablation: eq_mode 'frozen' vs 'adaptive'.

⚠️ This C[f] is NOT the pre-reconstruction domain-alignment EQ (that one
fixes VPU→mic transfer BEFORE the recon network).  This one fixes the
residual timbre between V and S, BEFORE fusion.  Independent — do not merge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import FusionConfig
from .utils import alpha_from_tau, causal_ema, smooth1d, CohTracker


@dataclass
class EQAlign:
    """Causal robust EQ C[f] (residual V↔S timbre alignment).  AC2: FROZEN mode
    (converge once → freeze → event-triggered re-estimate).  eq_mode=
    'adaptive' reverts to the B0 continuous-EMA behavior (ablation arm)."""
    cfg: FusionConfig
    enabled: bool = True
    changepoint_enabled: bool = True
    # state
    C: Optional[torch.Tensor] = None       # (B, Fb) dB
    prev_C: Optional[torch.Tensor] = None
    converged_count: int = 0
    converged: bool = False
    frozen: bool = False                  # AC2: once converged, stop updating
    coh: Optional[CohTracker] = None
    msc_prev: Optional[torch.Tensor] = None
    alpha_mode: str = "normal"            # "normal" | "fast"
    hold: int = 0

    def __post_init__(self):
        self.alpha = alpha_from_tau(self.cfg.eq_ema_tau_s, self.cfg.hop, self.cfg.sr)
        self.alpha_fast = alpha_from_tau(self.cfg.cp_fast_tau_s, self.cfg.hop, self.cfg.sr)

    def _init_state(self, B: int, Fb: int, device):
        if self.C is None:
            self.C = torch.zeros(B, Fb, device=device)
            self.prev_C = torch.zeros(B, Fb, device=device)
            self.coh = CohTracker(Fb, alpha_from_tau(self.cfg.cv_msc_tau_s,
                                                     self.cfg.hop, self.cfg.sr),
                                  device)

    def step(self, s_spec: torch.Tensor, v_spec: torch.Tensor,
             s_local_snr: torch.Tensor, f0_conf: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (v'_spec (B,Fb) complex, w_floor (B,), reset_flag (B,)).
        w_floor is the startup/重置压低 to multiply into w."""
        B, Fb = s_spec.shape
        self._init_state(B, Fb, s_spec.device)
        if not self.enabled:
            return v_spec, torch.zeros(B, device=s_spec.device), torch.zeros(B, device=s_spec.device)
        s_mag = s_spec.abs().clamp_min(1e-8)
        v_mag = v_spec.abs().clamp_min(1e-8)
        d = 20.0 * torch.log10(s_mag) - 20.0 * torch.log10(v_mag)   # (B, Fb) dB
        # dual-credible gate (broadcast scalars to per-bin)
        snr_cred = s_local_snr  # (B,) or (B,Fb)
        if snr_cred.dim() == 1:
            snr_cred = snr_cred.unsqueeze(-1).expand(B, Fb)
        conf_cred = f0_conf if f0_conf.dim() == 1 else f0_conf.mean(-1)
        conf_cred = conf_cred.unsqueeze(-1).expand(B, Fb) if conf_cred.dim() == 1 else conf_cred
        credible = (snr_cred > self.cfg.eq_update_s_snr_db) & \
                   (conf_cred > self.cfg.eq_update_f0_conf)
        # outlier rejection — only AFTER cold-start convergence (see below).
        resid = d - self.C
        if self.converged:
            reject = resid.abs() > self.cfg.eq_outlier_reject_db
            upd = credible & (~reject)
        else:
            # cold-start: accept all credible so C fast-tracks the true d
            # (FF↔VPU gain mismatch can be ~26 dB; with C=0 the outlier gate
            # would reject everything ⇒ bootstrap deadlock ⇒ G1 fails)
            upd = credible
        a = self.alpha_fast if self.alpha_mode == "fast" else self.alpha
        # AC2: in 'frozen' mode, STOP updating C once converged (freeze).  The
        # changepoint watchdog UNfreezes (converged=False) to re-estimate.
        if self.cfg.eq_mode == "frozen" and self.converged:
            upd = torch.zeros_like(upd)
        self.C = torch.where(upd, (1.0 - a) * self.C + a * d, self.C)
        # freq-axis smoothing of C (within frame, no time look-ahead)
        self.C = smooth1d(self.C, self.cfg.eq_freq_smooth_bins)
        # apply: V' = V * 10^(C/20) (mag), phase unchanged
        v_prime = v_spec * (10.0 ** (self.C / 20.0))
        # AC2 frozen: count credible updates; after eq_coldstart_frames → FREEZE.
        # (Robust to per-frame d noise — the |ΔC|.max gate never tripped because
        # noise bins keep d−C large.  Fixed-duration cold-start then freeze is
        # simpler and matches AC2 "converge once → freeze".)
        if self.cfg.eq_mode == "frozen":
            self.converged_count += int(upd.any().item())
            self.converged = self.converged_count >= self.cfg.eq_coldstart_frames
            if self.converged:
                self.frozen = True
        else:   # adaptive (B0 ablation): |ΔC| gate
            delta = (self.C - self.prev_C).abs().max(-1).values.max().item()
            if delta < self.cfg.eq_converge_db:
                self.converged_count += 1
            else:
                self.converged_count = 0
            self.converged = self.converged_count >= self.cfg.eq_converge_n_frames
        startup_floor = torch.full((B,), 0.0 if self.converged else self.cfg.eq_startup_w_floor,
                                    device=s_spec.device)
        # change-point detection
        reset_flag = torch.zeros(B, dtype=torch.bool, device=s_spec.device)
        if self.changepoint_enabled:
            msc = self.coh.update(v_spec[0] if B == 1 else v_spec.mean(0),
                                  s_spec[0] if B == 1 else s_spec.mean(0))
            if self.msc_prev is not None:
                msc_jump = (msc - self.msc_prev).abs().max().item()
                eqres_jump = resid.abs().max().item()
                if msc_jump > self.cfg.cp_msc_jump or eqres_jump > self.cfg.cp_eqres_jump_db:
                    self.alpha_mode = "fast"
                    self.hold = self.cfg.cp_hold_frames
                    self.converged_count = 0
                    self.converged = False
                    reset_flag[:] = True
            self.msc_prev = msc.clone()
            if self.hold > 0:
                self.hold -= 1
                startup_floor = torch.full((B,), self.cfg.cp_reset_w_floor,
                                            device=s_spec.device)
                self.alpha_mode = "fast"
            else:
                self.alpha_mode = "normal"
        self.prev_C = self.C.clone()
        return v_prime, startup_floor, reset_flag
