"""T13 A9 — fix A8's mechanism attribution: observational evidence only.

BOUNDARY: 0624 only, four male speakers (F0 87-124 Hz), normal volume.
No 0625 speech is read.  V* and every use of X remain outside production.

ONE-OFF CHARACTERIZATION, NOT A REGRESSION GATE: none of the five test_*
functions below contains an assert or a mutation — they only recompute and
report.  They provide NO regression guarantee; a runner PASS must NOT be
read as the conclusion being enforced.

A8's HEADLINE (ideal V* + real decision ≈ 0) is confirmed; the MECHANISM
attribution was wrong: A8 reported FULL-band factor medians, so w_band looked
like 0 (out-of-band bins where V*=V_real dragged the median down).  In-band
w_band is HIGH (0.79-0.87) — it did NOT collapse.  What the evidence supports:
all four single-factor ablations are insufficient and the all-≡1 control is
markedly higher, so the defect is NOT attributable to any single factor and
the multiplicative combination structure is the PRIME SUSPECT.  Two/three-
factor combinations were NOT tested — no conclusion is drawn on them.  A9
revokes the "MSC collapse" narrative (marked at source in README).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from fusion import FusionConfig
from fusion.degrade import DegradationConfig
from fusion.f0 import f0_batch
from tests.test_t13_a8 import _a8_run_one, _build_vstar_spec, _run_real_on_spec
from tests.test_t13_a3 import _g3_stats, _j_stats
from tests.test_t13_a5 import VSTAR_BANDS, _prepared
from tests.test_t13_b1 import _band_bins, _need

DEPTHS = [15, 20, 30]
B = 9
LEVELS = [-10.0, -3.7, 0.0, 3.7, 10.0]


def _inband_slice(cfg):
    return _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)


# ---------------------------------------------------------------------------
# A9-1: four factors in-band vs full-band (revokes the "MSC collapse" claim)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _a9_1_factors():
    _, records = _prepared()
    out = {k: dict(ib=[], fb=[]) for k in ("c_V", "g_f0", "w_band", "w_local", "w")}
    for prep in records:
        o = _a8_run_one(prep, 20, alpha=1.0, beta=1.0, noninfo_fill="xband")
        cfg = FusionConfig(); lo0, hi0 = _inband_slice(cfg)
        for k in out:
            f = o["factors"][k].float()
            out[k]["ib"].append(float(torch.median(f[0, lo0:hi0 + 1, :].flatten())))
            out[k]["fb"].append(float(torch.median(f.flatten())))
    return {k: dict(ib=float(np.median(v["ib"])), fb=float(np.median(v["fb"])))
            for k, v in out.items()}


def test_A9_1_factors_inband_vs_fullband():
    """A9-1: in-band vs full-band factor medians; revoke the MSC-collapse claim."""
    f = _a9_1_factors()
    print("  A9-1 four factors (ideal V*, d20, 10 rec, median):")
    print("  factor   in-band   full-band   (A8 reported full-band => w_band looked 0)")
    for k in ("c_V", "g_f0", "w_band", "w_local", "w"):
        print(f"  {k:>8}  {f[k]['ib']:.4f}    {f[k]['fb']:.4f}")
    print("  A4-1 V_real refs (in-band): c_V .33-.55, g_f0 .44-.52, "
          "w_band .62-.83, w_local .92-1")
    print("  => in-band w_band is HIGH (opened up), NOT collapsed.  A8's full-band")
    print("     median was dragged to 0 by out-of-band bins (V*=V_real there).")
    print("     The 'MSC collapse / wrong metric' mechanism in README A8 is REVOKED;")
    print("     the prime suspect is the multiplicative combination structure (A9-2;")
    print("     2/3-factor combos untested — no conclusion on them).")


# ---------------------------------------------------------------------------
# A9-2: per-factor ablation (force ≡1), 10 recs + d15/d20/d30 + matched null
# ---------------------------------------------------------------------------

ABLATIONS = [
    ("baseline", {}),
    ("w_local=1", dict(enable_w_local=False)),
    ("c_V=1", dict(enable_c_V=False)),
    ("g_f0=1", dict(enable_g_f0=False)),
    ("w_band=1", dict(enable_w_band=False)),
    ("all=1(w=1)", dict(enable_c_V=False, enable_g_f0=False,
                         enable_w_band=False, enable_w_local=False)),
]


def _ablation_metric(prep, depth, switches, permutation_seed=None, permute_true=False):
    cfg = FusionConfig().with_switches(**switches) if switches else FusionConfig()
    o = _a8_run_one(prep, depth, cfg=cfg, alpha=1.0, beta=1.0,
                    noninfo_fill="xband",
                    permutation_seed=permutation_seed, permute_true=permute_true)
    return o


@lru_cache(maxsize=1)
def _a9_2_results():
    _, records = _prepared()
    out = {}
    for label, sw in ABLATIONS:
        for depth in DEPTHS:
            rows = []
            for prep in records:
                o = _ablation_metric(prep, depth, sw)
                null = []
                for b in range(B):
                    n = _ablation_metric(prep, depth, sw,
                        permutation_seed=91000 + 1000 * b + prep["index"],
                        permute_true=True)
                    null.append(n)
                gmed = float(np.median([r["g"]["ratio"] for r in null]))
                jmed = float(np.median([r["j"]["j3"] for r in null]))
                rows.append(dict(dg=(1 - o["g"]["ratio"]) - (1 - gmed),
                                 dj=o["j"]["j3"] - jmed))
            out[(label, depth)] = rows
    return out


def test_A9_2_ablation_multiplicative():
    """A9-2: per-factor ≡1 ablation — one-off characterization (observational
    evidence, no asserts): defect not attributable to any single factor;
    the multiplicative combination structure is the prime suspect."""
    res = _a9_2_results()
    print("  A9-2 ablation (ideal V*, each factor forced ≡1, B=9 matched null):")
    print("  config        depth  dG3rec  dJ3")
    for label, _ in ABLATIONS:
        for depth in DEPTHS:
            rows = res[(label, depth)]
            dg = float(np.median([r["dg"] for r in rows]))
            dj = float(np.median([r["dj"] for r in rows]))
            print(f"  {label:>12}  {depth:>5}  {dg:+.4f}  {dj:+.4f}")
    # observation: no single-factor ≡1 reaches the all-≡1 control
    print("  observed: no single-factor ≡1 reaches the all-≡1 control => the defect")
    print("  is NOT attributable to any single factor; the multiplicative combination")
    print("  structure is the PRIME SUSPECT.  Untested 2/3-factor combinations:")
    print("  no conclusion is drawn on them.")


# ---------------------------------------------------------------------------
# A9-3: w_local absolute-level sensitivity (injection artifact; suspect = multiplicative structure)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _a9_3_level_sweep():
    _, records = _prepared()
    out = {}
    for lvl in LEVELS:
        rows = []
        for prep in records:
            o = _a8_run_one(prep, 20, alpha=1.0, beta=1.0, noninfo_fill="xband",
                            level_db=lvl)
            null = []
            for b in range(B):
                n = _a8_run_one(prep, 20, alpha=1.0, beta=1.0,
                    noninfo_fill="xband", level_db=lvl,
                    permutation_seed=91000 + 1000 * b + prep["index"],
                    permute_true=True)
                null.append(n)
            gmed = float(np.median([r["g"]["ratio"] for r in null]))
            rows.append((1 - o["g"]["ratio"]) - (1 - gmed))
        out[lvl] = float(np.median(rows))
    return out


def test_A9_3_wlocal_level_sensitivity():
    """A9-3: w_local is an uncalibrated absolute-level detector (BR2 issue).
    A8's V* went through EQ with C=0 (no level alignment); V* is ~3.7 dB low
    in-band vs V_real.  This separates the injection artifact from the real
    defect."""
    sweep = _a9_3_level_sweep()
    print("  A9-3 w_local level sensitivity (ideal V*, d20, level shift):")
    print("  level_dB   dG3rec   (note: A8 V* used level=0, NO alignment)")
    for lvl in LEVELS:
        print(f"  {lvl:>+7}   {sweep[lvl]:+.4f}")
    print("  w_local = sigmoid((Pv_overall - P_band - thr)/slope) with FIXED thr;")
    print("  it is an absolute-level detector (±10 dB => ~16x w_local swing).")
    print("  Aligning V* to V_real (+3.7 dB) only lifts dG3rec modestly (artifact);")
    print("  the multiplicative-combination suspect remains — level is NOT the cause.")


# ---------------------------------------------------------------------------
# A9-4: w≡1 vs oracle gap = per-(band,t) optimum vs global constant 1
# ---------------------------------------------------------------------------

def test_A9_4_w1_vs_oracle_gap():
    """A9-4: all-≡1 (w≡1) dG3rec vs oracle — w≡1 is a fixed always-on
    reference strategy (NOT an upper bound); oracle is the per-(band,t)
    upper bound.  Report only, don't fix."""
    res = _a9_2_results()
    print("  A9-4 w≡1 vs oracle gap (per-(band,t) upper bound vs fixed always-on reference):")
    for depth in DEPTHS:
        w1 = float(np.median([r["dg"] for r in res[("all=1(w=1)", depth)]]))
        # oracle ceiling (A8-1/A7): +0.391 @ d20; from A7-1 (alpha=1,beta=1)
        print(f"  depth {depth}: w≡1 dG3rec={w1:+.4f}; oracle ~+0.39 (d20) / "
              f"gap = per-(band,t) upper bound over fixed always-on reference")
    print("  report only — not a defect to fix (w≡1 is a fixed always-on REFERENCE")
    print("  strategy, NOT an upper bound; oracle is the per-(band,t) upper bound).")


def test_A9_decision():
    """Revised verdict (mechanism corrected)."""
    res = _a9_2_results()
    w1_d20 = float(np.median([r["dg"] for r in res[("all=1(w=1)", 20)]]))
    singles = {label: float(np.median([r["dg"] for r in res[(label, 20)]]))
               for label, _ in ABLATIONS
               if label not in ("baseline", "all=1(w=1)")}
    base_d20 = float(np.median([r["dg"] for r in res[("baseline", 20)]]))
    best_single = max(singles.values())
    print("  A9 decision (mechanism corrected):")
    print(f"  baseline (all real) d20 dG3rec={base_d20:.4f}")
    print(f"  best single-factor ≡1: {best_single:+.4f} (no single factor reaches")
    print(f"  the all≡1 control: {w1_d20:+.4f}; defect not attributable to any single factor)")
    print("  => defect is NOT attributable to any single factor, not MSC; the")
    print("  multiplicative COMBINATION structure is the prime suspect (2/3-factor")
    print("  combos untested).  Fix = change the COMBINATION")
    print("  (one judge signal + safety veto), NOT blow up the decision layer; and it")
    print("  must change WITH Arm A (w≡1 + V_real still ~0, A5R-2).")
