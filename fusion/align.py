"""Layer 1 · alignment (slow).

Two pieces, BOTH causal:
  * ``DelayComp`` — a FIXED integer-sample delay applied to V (measured once
    offline by GCC-PHAT in 100–600 Hz; online only monitors).  NO per-frame
    re-estimation (spec: re-estimation ⇒ jitter).  Implemented as a causal
    FIFO shift (delay V later; sign convention documented).  ``delay=0`` ⇒
    passthrough (placeholder).
  * ``EQAlign`` — the residual V↔S timbre EQ ``C[f]``: a ROBUST CAUSAL EMA of
    ``d = log|S| − log|V|`` (dB), updated ONLY on dual-credible blocks (S
    local-SNR high AND f0_conf high), applied全时段 ``V' = V·10^(C/20)``.
    Outlier rejection (|d−C| > reject_db ⇒ discard).  No whole-segment median.
    Startup压低 w until C converges.  EQ change-point (MSC/|d−C| jump) ⇒ reset
    to fast re-estimate +压低 w.

⚠️ This C[f] is NOT the pre-reconstruction domain-alignment EQ (that one fixes
VPU→mic transfer BEFORE the recon network).  This one fixes the residual
timbre between V and S, BEFORE fusion.  Independent — do not merge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import FusionConfig
from .utils import alpha_from_tau, causal_ema, smooth1d, CohTracker


def measure_gcc_phat(s: torch.Tensor, v: torch.Tensor, cfg: FusionConfig) -> int:
    """OFFLINE: GCC-PHAT delay (samples) between S and V in [gcc_lo, gcc_hi] Hz.
    Used to SET the DelayComp constant once; NOT called online."""
    spec_s = torch.fft.rfft(s.float(), n=cfg.n_fft)
    spec_v = torch.fft.rfft(v.float(), n=cfg.n_fft)
    bz = cfg.sr / cfg.n_fft
    lo = max(1, int(cfg.gcc_lo_hz / bz))
    hi = min(cfg.n_fft // 2, int(cfg.gcc_hi_hz / bz))
    num = spec_s * torch.conj(spec_v)
    num[lo:hi + 1] *= 1.0
    num[:lo] = 0
    num[hi + 1:] = 0
    denom = (num.abs() + 1e-8).clamp_min(1e-8)
    xcorr = torch.fft.irfft(num / denom, n=cfg.n_fft)
    return int(torch.argmax(xcorr).item())


@dataclass
class DelayComp:
    """Causal fixed-delay FIFO shift on V (time-domain hop chunks)."""
    cfg: FusionConfig
    delay: int = 0
    buf: Optional[torch.Tensor] = None   # FIFO of `delay` samples

    def step(self, v_hop: torch.Tensor) -> torch.Tensor:
        if self.delay <= 0:
            return v_hop
        B = v_hop.shape[0]
        if self.buf is None:
            self.buf = v_hop.new_zeros(B, self.delay)
        # emit `hop` from FIFO, push new `hop` in
        out = torch.cat([self.buf, v_hop], dim=1)[:, :v_hop.shape[-1]]
        self.buf = torch.cat([self.buf, v_hop], dim=1)[:, -self.delay:]
        return out


@dataclass
class EQAlign:
    """Causal robust EQ C[f] (residual V↔S timbre alignment) + change-point."""
    cfg: FusionConfig
    enabled: bool = True
    changepoint_enabled: bool = True
    # state
    C: Optional[torch.Tensor] = None       # (B, Fb) dB
    prev_C: Optional[torch.Tensor] = None
    converged_count: int = 0
    converged: bool = False
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
        # outlier rejection
        resid = d - self.C
        reject = resid.abs() > self.cfg.eq_outlier_reject_db
        upd = credible & (~reject)
        a = self.alpha_fast if self.alpha_mode == "fast" else self.alpha
        self.C = torch.where(upd, (1.0 - a) * self.C + a * d, self.C)
        # freq-axis smoothing of C (within frame, no time look-ahead)
        self.C = smooth1d(self.C, self.cfg.eq_freq_smooth_bins)
        # apply: V' = V * 10^(C/20) (mag), phase unchanged
        v_prime = v_spec * (10.0 ** (self.C / 20.0))
        # convergence (global counter: frames where max|ΔC|<gate in a row)
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
