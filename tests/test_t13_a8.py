"""T13 A8 — ideal V* + real decision layer (the missing 2×2 cell).

BOUNDARY: 0624 only, four male speakers (F0 87-124 Hz), normal volume.
No 0625 speech is read.  V* and every use of X remain outside production.

All prior ceilings used oracle-w.  This fills the one unmeasured cell:
  ideal V* (alpha=1, beta=1, sigma_e=sigma_b=0, xband) + REAL decision layer
  (real c_V·g_f0·w_band·w_local, NOT oracle).  If the real layer delivers ~0
  even with perfect input, the decision layer is independently broken and Arm A
  alone cannot save it.  Per the ceiling-vs-detectability rule (README), this
  pushes the detectability variable to its extreme (ideal V* => d exactly equals
  the deficit => detection is trivial), so any residual failure is the decision
  layer's own.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from fusion import FusionConfig, FusionCore, realdata
from fusion.degrade import DegradationConfig
from fusion.f0 import f0_batch
from fusion.stft import istft_batch, stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_a3 import _g3_stats, _j_stats
from tests.test_t13_a4 import FACTORS, _run
from tests.test_t13_a5 import VSTAR_BANDS, _prepared, _within_state_permutation
from tests.test_t13_b1 import BAND_EDGES_HZ, _band_bins, _need

DEPTHS = [15, 20, 30]
B = 9
SIGMA_E = [0.5, 1.0, 2.0, 4.0, 6.0]
SIGMA_B = [0.0, 0.1, 0.3, 1.0, 3.0]


def _run_real_on_spec(ff, spec_vp, cfg, deg):
    """Real decision layer (real w = c_V·g·wband·wlocal, smoothed) on a GIVEN
    V* spec (no istft->stft roundtrip; the V* IS the input to the layer).
    Captures the four factors + w.  Mirrors tests.test_t13_a4._run but takes
    spec_vp directly instead of stft(vpu)."""
    spec_x, spec_s, s = eval_specs(ff, cfg, deg)
    spec_v = spec_vp
    left = cfg.win - cfg.hop
    frames = (F.pad(s.float(), (left, 0)).unsqueeze(1)
              .unfold(-1, cfg.win, cfg.hop).squeeze(1))
    core = FusionCore(cfg); y_frames = []
    captured = {key: [] for key in FACTORS}
    for t in range(spec_s.shape[-1]):
        ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
        f0, conf = core.f0est.estimate(buf)
        smag = ss.abs(); floor = core.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) /
                                 floor.clamp_min(1e-8))).mean(-1)
        vp, startup, reset = core.eq.step(ss, vs, snr, conf)
        eqr = ((20 * torch.log10(ss.abs().clamp_min(1e-8))
                - 20 * torch.log10(vs.abs().clamp_min(1e-8))
                - core.eq.C).mean(-1) if core.eq.C is not None
               else torch.zeros_like(snr))
        cv = core.cv.step(vp, ss, eqr, bool(reset.any()))
        gf = core.gf0.step(conf)
        wb = core.wband.step(vp, ss)
        wl = core.wlocal.step(ss, vp, f0)
        product = cv.unsqueeze(-1) * gf.unsqueeze(-1) * wb * wl
        fw = torch.maximum(startup, reset.float())
        w = core.smooth.step(product * (1 - fw).unsqueeze(-1))
        fb = spec_s.shape[1]
        captured["c_V"].append(cv.unsqueeze(-1).expand(-1, fb).detach())
        captured["g_f0"].append(gf.unsqueeze(-1).expand(-1, fb).detach())
        captured["w_band"].append(wb.detach())
        captured["w_local"].append(wl.detach())
        captured["w_product"].append(product.detach())
        captured["w"].append(w.detach())
        y_frames.append(core.synth.step(ss, vp, w))
    spec_y_direct = torch.stack(y_frames, -1)
    y = istft_batch(spec_y_direct, cfg, length=s.shape[-1])
    factors = {key: torch.stack(value, -1) for key, value in captured.items()}
    return dict(spec_x=spec_x, spec_s=spec_s, s=s, ff=ff,
                spec_y=stft_batch(y, cfg), factors=factors)


def _build_vstar_spec(prep, alpha=1.0, beta=1.0, noninfo_fill="xband",
                      permutation_seed=None, permute_true=False,
                      sigma_e=0.0, sigma_b=0.0, noise_seed=0, level_db=0.0):
    """Construct the V* spec (alpha/beta/xband or sigma_e/sigma_b perturb),
    with optional time permutation for the null.  level_db shifts the in-band
    magnitude (A9-3: w_local absolute-level sensitivity probe)."""
    from tests.test_t13_a5 import build_vstar
    if sigma_e == 0.0 and sigma_b == 0.0:
        spec_vp, _ = build_vstar(prep, float(alpha), beta=beta,
                                 noninfo_fill=noninfo_fill,
                                 permutation_seed=permutation_seed,
                                 permute_true=permute_true)
        if level_db != 0.0:
            cfg = FusionConfig()
            lo0, hi0 = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
            spec_vp = spec_vp.clone()
            spec_vp[0, lo0:hi0 + 1] = spec_vp[0, lo0:hi0 + 1] * (10.0 ** (level_db / 20.0))
        return spec_vp
    # perturb path: A_true + noise (per-bin sigma_e, per-(band,frame) sigma_b)
    cfg = FusionConfig(); sx = prep["spec_x"]; sv = prep["spec_v"]
    out = sv.clone(); n_frames = sx.shape[-1]
    mapping = (torch.arange(n_frames) if permutation_seed is None
               else _within_state_permutation(prep["conf_v"][0], permutation_seed))
    lo0, hi0 = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
    bins = slice(lo0, hi0 + 1)
    a_true_all = 20 * torch.log10(sx[0, bins].abs().clamp_min(1e-8))
    a_true = a_true_all[:, mapping] if permute_true else a_true_all
    rng = np.random.default_rng(int(noise_seed) * 1000003 + 17)
    mag_log = a_true.clone()
    if sigma_e > 0:
        mag_log = mag_log + torch.tensor(
            rng.normal(0.0, sigma_e, a_true.shape), dtype=a_true.dtype)
    if sigma_b > 0:
        for lo_hz, hi_hz in VSTAR_BANDS:
            blo, bhi = _band_bins(cfg, lo_hz, hi_hz)
            rlo, rhi = blo - lo0, bhi - lo0
            nb = torch.tensor(rng.normal(0.0, sigma_b, (1, n_frames)),
                              dtype=a_true.dtype)
            mag_log[rlo:rhi + 1, :] = mag_log[rlo:rhi + 1, :] + nb
    out[0, bins] = 10 ** (mag_log / 20) * torch.exp(1j * torch.angle(sv[0, bins]))
    return out


def _a8_run_one(prep, depth, cfg=None, **vstar_kw):
    if cfg is None:
        cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
    spec_vp = _build_vstar_spec(prep, **vstar_kw)
    out = _run_real_on_spec(prep["ff"], spec_vp, cfg, deg)
    _, conf = f0_batch(prep["ff"], cfg)
    g = _g3_stats(out["spec_x"], out["spec_s"], out["spec_y"], cfg)
    j = _j_stats(out["spec_x"], out["spec_s"], out["spec_y"], conf, cfg)
    return dict(g=g, j=j, factors=out["factors"],
                spec_x=out["spec_x"], spec_s=out["spec_s"], spec_y=out["spec_y"])


# ---------------------------------------------------------------------------
# A8-1: ideal V* + real decision layer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _a8_1_results():
    _, records = _prepared()
    out = {}
    for depth in DEPTHS:
        rows = []
        for prep in records:
            o = _a8_run_one(prep, depth, alpha=1.0, beta=1.0, noninfo_fill="xband")
            null = []
            for b in range(B):
                n = _a8_run_one(prep, depth, alpha=1.0, beta=1.0,
                    noninfo_fill="xband",
                    permutation_seed=91000 + 1000 * b + prep["index"],
                    permute_true=True)
                null.append(n)
            gmed = float(np.median([r["g"]["ratio"] for r in null]))
            jmed = float(np.median([r["j"]["j3"] for r in null]))
            rows.append(dict(
                rec=prep["name"], ratio=o["g"]["ratio"], null_ratio=gmed,
                recovery=1 - o["g"]["ratio"], null_recovery=1 - gmed,
                j1=o["j"]["j1"], j2=o["j"]["j2"], j3=o["j"]["j3"],
                null_j3=jmed, factors=o["factors"],
                spec_x=o["spec_x"], spec_s=o["spec_s"], spec_y=o["spec_y"]))
        out[depth] = rows
    return out


def test_A8_1_ideal_vstar_real_decision():
    """A8-1: ideal V* (alpha=1,beta=1,xband) + REAL decision layer (real w)."""
    res = _a8_1_results()
    print("  A8-1 ideal V* + real decision layer (10 rec, B=9 null):")
    print("  depth  dG3rec  dJ3    J1    J2   (oracle ceiling dG3=+0.391)")
    for depth in DEPTHS:
        rows = res[depth]
        dg = float(np.median([r["recovery"] - r["null_recovery"] for r in rows]))
        dj = float(np.median([r["j3"] - r["null_j3"] for r in rows]))
        j1 = float(np.median([r["j1"] for r in rows]))
        j2 = float(np.median([r["j2"] for r in rows]))
        print(f"  {depth:>5}  {dg:+.4f}  {dj:+.4f}  {j1:.3f}  {j2:.3f}")
    # w distribution + 4 factors at d20
    rows = res[20]
    w_all = torch.cat([r["factors"]["w"].flatten() for r in rows]).float()
    # suppressed vs unsuppressed w (band-frame sup = 10log10(px/ps))
    sup_w = []; unsup_w = []
    for r in rows:
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(FusionConfig(), BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            px = r["spec_x"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            ps = r["spec_s"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            sup = 10 * torch.log10(px / ps)
            wbf = r["factors"]["w"][0, lo:hi + 1, :].float()
            sup_w.append(wbf[:, sup > 6.0].flatten())
            unsup_w.append(wbf[:, sup <= 6.0].flatten())
    sup_w = torch.cat(sup_w) if sup_w and sup_w[0].numel() else torch.tensor([])
    unsup_w = torch.cat(unsup_w) if unsup_w and unsup_w[0].numel() else torch.tensor([])
    print("  w distribution (d20, real decision layer, ideal V*):")
    print(f"    all:   p50={torch.median(w_all).item():.4f} "
          f"p90={torch.quantile(w_all, 0.9).item():.4f} max={w_all.max().item():.4f}")
    if sup_w.numel():
        print(f"    sup:   p50={torch.median(sup_w).item():.4f} "
              f"p90={torch.quantile(sup_w, 0.9).item():.4f} max={sup_w.max().item():.4f} "
              f"(n={sup_w.numel()})")
    if unsup_w.numel():
        print(f"    unsup: p50={torch.median(unsup_w).item():.4f} "
              f"p90={torch.quantile(unsup_w, 0.9).item():.4f} max={unsup_w.max().item():.4f} "
              f"(n={unsup_w.numel()})")
    # 4 factors (median over all bins/frames), compare A4-1 V_real
    print("  4 factors under ideal V* (d20, median) vs A4-1 V_real:")
    for key in ("c_V", "g_f0", "w_band", "w_local"):
        vals = torch.cat([r["factors"][key].flatten() for r in rows]).float()
        print(f"    {key}: med={torch.median(vals).item():.3f} "
              f"p90={torch.quantile(vals, 0.9).item():.3f}")
    print("  (A4-1 V_real refs: c_V .33-.55, g_f0 .44-.52, w_band .62-.83, "
          "w_local .92-1)")


# ---------------------------------------------------------------------------
# A8-2: real decision layer sigma_e / sigma_b axes (deployable Arm A spec)
# ---------------------------------------------------------------------------

def _a8_2_axis(axis, sweep):
    _, records = _prepared()
    out = {}
    for s in sweep:
        rows = []
        for prep in records:
            kw = (dict(sigma_e=s, sigma_b=0.0) if axis == "e"
                  else dict(sigma_e=0.0, sigma_b=s))
            o = _a8_run_one(prep, 20, **kw)
            null = []
            for b in range(B):
                nk = dict(kw)
                nk["permutation_seed"] = 91000 + 1000 * b + prep["index"]
                nk["permute_true"] = True
                nk["noise_seed"] = 91000 + 1000 * b + prep["index"] + 1
                null.append(_a8_run_one(prep, 20, **nk))
            gmed = float(np.median([r["g"]["ratio"] for r in null]))
            rows.append(1 - o["g"]["ratio"] - (1 - gmed))
        out[s] = float(np.median(rows))
    return out


@lru_cache(maxsize=4)
def _a8_2_axis_cached(axis):
    return _a8_2_axis(axis, tuple(SIGMA_E if axis == "e" else SIGMA_B))


def test_A8_2_real_decision_sigma_axes():
    """A8-2: real decision layer sigma_e / sigma_b axes (deployable spec)."""
    se = _a8_2_axis_cached("e")
    sb = _a8_2_axis_cached("b")
    # A8-1 d20 ceiling (real decision, ideal V*) for the "half of own ceiling" target
    a8_1 = _a8_1_results()
    ceiling = float(np.median(
        [r["recovery"] - r["null_recovery"] for r in a8_1[20]]))
    half = ceiling / 2.0
    print("  A8-2 real decision layer sigma axes (deployable Arm A spec):")
    print(f"  A8-1 d20 ceiling (real decision, ideal V*) = {ceiling:+.4f}; "
          f"half-target = {half:+.4f}")
    print("  σ_e axis (real decision, β=1.0, σ_b=0):")
    print("  σ_e(dB)  dG3rec")
    for s in SIGMA_E:
        print(f"  {s:>6}  {se[s]:+.4f}")
    print("  σ_b axis (real decision, β=1.0, σ_e=0):")
    print("  σ_b(dB)  dG3rec")
    for s in SIGMA_B:
        print(f"  {s:>6}  {sb[s]:+.4f}")
    se_half = max([s for s in SIGMA_E if se[s] >= half], default=None)
    sb_half = max([s for s in SIGMA_B if sb[s] >= half], default=None)
    print(f"  reach half-ceiling ({half:+.4f}): σ_e<={se_half}  σ_b<={sb_half}")


def test_A8_decision():
    """Predeclared interpretation of A8-1."""
    res = _a8_1_results()
    dg20 = float(np.median(
        [r["recovery"] - r["null_recovery"] for r in res[20]]))
    print("  A8 decision (predeclared, d20 ideal V* + real decision):")
    print(f"  dG3rec = {dg20:+.4f}  (oracle ceiling +0.391)")
    if dg20 >= 0.30:
        print("  => ~oracle ceiling: decision layer FINE, bottleneck is "
              "reconstruction; Arm A is the only thing to do.")
    elif dg20 > 0.05:
        print(f"  => {dg20:+.4f} < 0.391 but >0: decision layer has loss but "
              f"is usable => Arm A + decision-layer tuning.")
    else:
        print(f"  *** dG3rec ≈ 0 ({dg20:+.4f}): even with perfect Arm A, the "
              f"current fusion delivers NOTHING => decision layer is "
              f"independently broken, must be redone; Arm A alone is WASTE. "
              f"(Most important outcome.) ***")
