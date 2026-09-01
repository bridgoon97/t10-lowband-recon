"""Fusion pipeline (layer 1 + 2 + 3) — batch and streaming, both CAUSAL.

  ``Fusion.process_batch(S, V) -> Y``  — vectorized STFT (causal_stft), per-frame
       decision loop (sequential EMAs), vectorized iSTFT (causal_istft).
  ``FusionStreamer.stream_step(s_hop, v_hop) -> y_hop | None`` — per-hop STFT +
       per-frame decision + per-hop iSTFT, bounded state.

Both paths share ``FusionCore.process_frame`` (the per-frame decision) and use
SEPARATE layer-object instances, so the only batch↔streaming difference is the
STFT/iSTFT engine — and those are bit-identical per frame (verified in
``tests/test_t13_streaming.py``: streaming-vs-batch diff == 0.0 interior).

T13-MVP: ``cfg.decision_mode`` selects the layer-2 combination —
  "mvp"            (default) ONE main correction signal (w_local band evidence)
                    + BINARY safety vetoes (f0 conf / c_V / MSC; thresholds
                    pre-fixed in FusionConfig before any effect observation).
  "legacy_multiply" comparison mode: the historical c_V·g_f0·w_band·w_local
                    product, bit-identical to pre-MVP behavior at strength=1.
Both modes share layer 1 (EQ/F0), the smoother, and synthesis unchanged.
``cfg.strength`` scales the FINAL clipped correction (0 ⇒ Y≡S exactly).
``Fusion.process_batch`` exposes ``self.last_diagnostics`` (MVP mode) for the
CLI: per-band correction stats, intervention coverage, veto fractions.

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
from .decision import CV, GF0, WBand, WLocal, AsymSmoother, mvp_combine
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
        self.mvp = (cfg.decision_mode == "mvp")
        if self.mvp:
            # MVP veto uses c_V as a V-HEALTH signal: the KR1 EQ-residual bias
            # term measures S↔V relationship DRIFT (= the damage to correct,
            # NOT a V fault) — switch it off for the MVP decision only.
            cv_cfg = cfg.with_switches(cv_eqresid_mode="off")
            # comfort noise (−40 dB guard) off in MVP v1 ⇒ exact safety identity
            synth_cfg = (cfg if cfg.mvp_comfort_noise
                         else cfg.with_switches(enable_comfort_noise=False))
        else:
            cv_cfg = cfg
            synth_cfg = cfg
        self.eq = EQAlign(cfg, enabled=cfg.enable_eq,
                          changepoint_enabled=cfg.enable_eq_changepoint)
        self.cv = CV(cv_cfg, enabled=cfg.enable_c_V)
        self.gf0 = GF0(cfg, enabled=cfg.enable_g_f0)
        self.wband = WBand(cfg, enabled=cfg.enable_w_band,
                           fixed_curve=cfg.use_w_band_fixed_curve)
        self.wlocal = WLocal(cfg, enabled=cfg.enable_w_local,
                             pure_band=cfg.use_w_local_pure_band,
                             v_perturb=cfg.wl_v_perturb)
        self.smooth = AsymSmoother(cfg, enabled=cfg.enable_asym_smooth,
                                   symmetric=cfg.use_symmetric_smooth)
        self.synth = Synthesis(synth_cfg)
        self.nf = NoiseFloor(cfg)
        self.f0est = F0Estimator(cfg)
        self.w_history = []   # per-frame w (B, Fb) — for M5 propagation diagnostics
        self.veto_history = []  # MVP: per-frame veto mask (B, Fb) bool
        self.veto_frame_history = []  # MVP: per-frame frame-level veto (B,) bool
        self.corr_history = []  # MVP: per-frame applied correction dB (B, Fb)

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
                    - self.eq.C).mean(-1) if self.eq.C is not None else torch.zeros_like(snr)   # KR1: SIGNED (d−C), not abs — CV tracks long-term bias
        # Layer 2
        c_v = self.cv.step(v_prime, s_spec, eq_resid, bool(reset_flag.any()))
        g = self.gf0.step(conf)
        w_band = self.wband.step(v_prime, s_spec)
        w_local = self.wlocal.step(s_spec, v_prime, f0)
        veto = None
        if self.mvp:
            w_raw, veto, veto_frame = mvp_combine(w_local, g, c_v, w_band, cfg)
            self.veto_frame_history.append(veto_frame.detach().clone())
        else:
            w_raw = (c_v.unsqueeze(-1) * g.unsqueeze(-1) * w_band * w_local)
        floor_w = torch.maximum(startup_floor, reset_flag.float())
        w = w_raw * (1.0 - floor_w).unsqueeze(-1)
        w = self.smooth.step(w)
        if self.mvp:
            # T13-MVP rework: the binary safety mask is RE-APLIED AFTER the
            # smoother — a frame with any veto (frame-level f0/c_V, per-bin MSC)
            # or an active startup/reset floor gets w EXACTLY 0 into synthesis
            # (the smoother's fall tau would otherwise leave a residual, e.g.
            # ~0.51 on the first veto frame after established w).  Normal
            # main-signal smoothing is untouched: the smoother state keeps
            # evolving (w_raw already carries veto/floor zeros), so recovery
            # still rises via the existing rise tau once safety is restored.
            floor_bin = (floor_w > 0).to(w.dtype)
            w = w * (~veto).to(w.dtype) * (1.0 - floor_bin).unsqueeze(-1)
            final_veto = veto | (floor_w > 0).unsqueeze(-1)   # (B, Fb) bool
            self.veto_history.append(final_veto.detach().clone())
        self.w_history.append(w.detach().clone())
        # Layer 3
        y_spec = self.synth.step(s_spec, v_prime, w)
        if self.mvp:
            corr_db = (20.0 * torch.log10(y_spec.abs().clamp_min(1e-8))
                       - 20.0 * torch.log10(s_spec.abs().clamp_min(1e-8)))
            self.corr_history.append(corr_db.detach().clone())
        # 2 kHz boundary: taper w to 0 by hi_bin already in w_band taper; pass-thru
        # of bins > hi handled in synth (S copied).
        return y_spec, w


# ============================ BATCH =======================================

class Fusion:
    """Batch (whole-signal) fusion."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.core = FusionCore(cfg)
        self.last_diagnostics: Optional[dict] = None

    def process_batch(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """S, V: (B, T) → Y (B, T).  S=stage-2 proxy, V=VPU.  F0 always estimated.
        AC1: no delay comp (phase taken from S).  MVP mode fills
        ``self.last_diagnostics`` (per-band correction stats / coverage / vetoes)."""
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
        y = istft_batch(y_spec, cfg, length=s.shape[-1])
        if self.core.mvp:
            self.last_diagnostics = _mvp_diagnostics(self.core, cfg)
        return y


def _mvp_diagnostics(core: "FusionCore", cfg: FusionConfig) -> dict:
    """Aggregate the MVP per-frame histories into the CLI diagnostics dict.
    Correction stats are over the four report sub-bands (100–200 / 200–315 /
    315–500 / 500–800 Hz) — the region MVP can act in; coverage = fraction of
    (bin, frame) in 100–800 Hz with |applied correction| ≥ 1 dB."""
    import numpy as np
    corr = torch.stack(core.corr_history, dim=-1)[0].float()   # (Fb, N)
    veto = torch.stack(core.veto_history, dim=-1)[0]           # (Fb, N) bool
    bz = cfg.sr / cfg.n_fft
    edges = [100, 200, 315, 500, 800]
    band_stats = {}
    lo_all, hi_all = None, None
    for i in range(len(edges) - 1):
        blo = max(1, int(edges[i] / bz)); bhi = min(corr.shape[0] - 1, int(edges[i + 1] / bz))
        c = corr[blo:bhi + 1].flatten()
        band_stats[f"{edges[i]}-{edges[i + 1]}"] = {
            "p50_db": float(c.median()),
            "p90_abs_db": float(c.abs().quantile(0.9)),
            "max_abs_db": float(c.abs().max()),
        }
        lo_all = blo if lo_all is None else lo_all
        hi_all = bhi
    c_all = corr[lo_all:hi_all + 1].flatten()
    v_all = veto[lo_all:hi_all + 1].flatten()
    diag = {
        "decision_mode": cfg.decision_mode,
        "strength": cfg.strength,
        "coverage_100_800": float((c_all.abs() >= 1.0).float().mean()),
        "correction_100_800": {
            "p50_db": float(c_all.median()),
            "p90_abs_db": float(c_all.abs().quantile(0.9)),
            "max_abs_db": float(c_all.abs().max()),
            "min_db": float(c_all.min()),   # most negative (reverse) correction
        },
        "band_stats": band_stats,
        "veto_fraction_100_800": float(v_all.float().mean()),
        "veto_f0_frame_fraction": float(torch.stack(
            core.veto_frame_history).float().mean()),
    }
    return diag


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
