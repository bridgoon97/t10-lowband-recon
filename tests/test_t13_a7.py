"""T13 A7 — translate the (beta, alpha) ceiling into an executable Arm A spec.

BOUNDARY: 0624 only, four male speakers (F0 87-124 Hz), normal volume.
No 0625 speech is read.  V* and every use of X remain outside production.

Methodological rule (per reviewer, written into reports/T13/README.md A6-1c):
oracles with DIFFERENT degrees of freedom have incomparable RAW values; each
grid point here carries its OWN matched null (recordwise B=9 permutation), so
the reported deltaG3rec / deltaJ3 are null-corrected and comparable across
(alpha) — alpha changes V*'s per-bin info content => different freedom => the
matched null is mandatory at every point, not just the corners.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from fusion import FusionConfig, realdata
from fusion.f0 import f0_batch
from fusion.stft import stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_a5 import (VSTAR_BANDS, _prepared, build_vstar,
    _oracle_metric_for_spec, _within_state_permutation)
from tests.test_t13_b1 import _band_bins, _need

DEPTHS = [20]
BETAS = [0.25, 0.5, 0.75, 1.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
B = 9
REPORT_DIR = os.path.join("reports", "T13A7")


# ---------------------------------------------------------------------------
# A7-1: 2D (beta, alpha) scan, xband fill, band-scalar oracle, d20, matched null
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _scan_grid():
    cfg, records = _prepared()
    out = {}
    for beta in BETAS:
        for alpha in ALPHAS:
            rows = []
            for prep in records:
                obs, _ = build_vstar(prep, float(alpha), beta=beta,
                                     noninfo_fill="xband")
                o = _oracle_metric_for_spec(
                    prep, obs, 20, eq_mode="zero", direct=True,
                    band_indices=range(4), oracle_mode="band")
                null = []
                for b in range(B):
                    ns, _ = build_vstar(
                        prep, float(alpha),
                        permutation_seed=91000 + 1000 * b + prep["index"],
                        beta=beta, permute_true=True, noninfo_fill="xband")
                    null.append(_oracle_metric_for_spec(
                        prep, ns, 20, eq_mode="zero", direct=True,
                        band_indices=range(4), oracle_mode="band"))
                gmed = float(np.median([r["recovery"] for r in null]))
                jmed = float(np.median([r["j3"] for r in null]))
                rows.append(dict(dg=o["recovery"] - gmed, dj=o["j3"] - jmed))
            out[(beta, alpha)] = rows
    return out


def test_A7_1_beta_alpha_scan():
    """A7-1: 2D (beta, alpha) ceiling scan.

    beta = fraction of in-band bins carrying true per-bin X info; alpha =
    per-bin fidelity of those bins (A* = alpha*A_true + (1-alpha)*A_band).
    Fixed xband fill (non-info bins = X band mean), band-scalar oracle w
    (per-bin proven useless in A6-1c), d20, matched B=9 null at EVERY grid
    point (different alpha => different freedom => raw values incomparable).
    Reports deltaG3rec and deltaJ3; plots contour.
    """
    out = _scan_grid()
    print("  A7-1 (beta, alpha) scan (xband, band-scalar oracle, d20, B=9 null):")
    print("  beta  alpha  dG3rec  dJ3")
    for beta in BETAS:
        for alpha in ALPHAS:
            rows = out[(beta, alpha)]
            dg = float(np.median([r["dg"] for r in rows]))
            dj = float(np.median([r["dj"] for r in rows]))
            print(f"  {beta:>4}  {alpha:>5}  {dg:+.4f}  {dj:+.4f}")
    _plot_contour(out)
    # sanity: ceiling (1,1) should reproduce the ~0.39 anchor
    ceil = float(np.median([r["dg"] for r in out[(1.0, 1.0)]]))
    print(f"  ceiling (beta=1, alpha=1) dG3rec={ceil:+.4f} (anchor ~+0.39)")


def _plot_contour(out):
    os.makedirs(REPORT_DIR, exist_ok=True)
    # SVG contour: x=alpha, y=beta, color=dG3rec
    w, h = 720, 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="360" y="24" text-anchor="middle" font-size="18">'
           'A7-1 deltaG3rec contour (xband, band-scalar oracle, d20)</text>']
    x0, y0, pw, ph = 70, 50, 560, 320
    svg.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" '
               f'fill="none" stroke="#666"/>')
    dgs = { (b, a): float(np.median([r["dg"] for r in out[(b, a)]]))
            for b in BETAS for a in ALPHAS }
    vmin, vmax = min(dgs.values()), max(dgs.values())
    span = max(1e-9, vmax - vmin)

    def cmap(v):
        t = (v - vmin) / span
        # blue (low) -> red (high)
        r = int(255 * t); bl = int(255 * (1 - t))
        return f"rgb({r},0,{bl})"

    for (b, a) in dgs:
        xi = x0 + (ALPHAS.index(a) + 0.5) * pw / len(ALPHAS)
        yi = y0 + ph - (BETAS.index(b) + 0.5) * ph / len(BETAS)
        svg.append(f'<rect x="{xi-24}" y="{yi-16}" width="48" height="32" '
                   f'fill="{cmap(dgs[(b,a)])}" stroke="#333"/>')
        svg.append(f'<text x="{xi}" y="{yi+4}" text-anchor="middle" '
                   f'font-size="11" fill="white">{dgs[(b,a)]:+.2f}</text>')
    for i, a in enumerate(ALPHAS):
        xi = x0 + (i + 0.5) * pw / len(ALPHAS)
        svg.append(f'<text x="{xi}" y="{y0+ph+18}" text-anchor="middle" '
                    f'font-size="12">alpha={a}</text>')
    for i, b in enumerate(BETAS):
        yi = y0 + ph - (i + 0.5) * ph / len(BETAS)
        svg.append(f'<text x="{x0-8}" y="{yi+4}" text-anchor="end" '
                    f'font-size="12">beta={b}</text>')
    svg.append(f'<text x="360" y="{h-10}" text-anchor="middle" font-size="12">'
               f'dG3rec range [{vmin:+.3f}, {vmax:+.3f}]</text></svg>')
    path = os.path.join(REPORT_DIR, "a7_beta_alpha_contour.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"  plot -> {path}")


# ---------------------------------------------------------------------------
# A7-2: translate alpha -> "Arm A per-bin log-magnitude error <= X dB"
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _sigma_fine():
    """sigma_fine = std of (A_true - A_band) per band, voiced frames, 10 recs.

    A* = alpha*A_true + (1-alpha)*A_band => per-bin error = (1-alpha)*(A_true
    - A_band), std = (1-alpha)*sigma_fine.  So alpha -> dB via
    (1-alpha)*sigma_fine.  sigma_fine = the in-band spectral fine structure
    (how much each bin deviates from its band mean), directly measurable.
    """
    cfg, records = _prepared()
    per_band = {b: [] for b in VSTAR_BANDS}
    for prep in records:
        sx = prep["spec_x"]; conf = prep["conf_x"]
        voiced = conf[0] >= 0.55
        for lo_hz, hi_hz in VSTAR_BANDS:
            lo, hi = _band_bins(cfg, lo_hz, hi_hz)
            a_true = 20 * torch.log10(sx[0, lo:hi + 1].abs().clamp_min(1e-8))
            a_band = a_true.mean(0, keepdim=True)  # band mean per frame
            dev = (a_true - a_band)[:, voiced].detach().cpu().numpy()
            per_band[(lo_hz, hi_hz)].append(float(dev.std()))
    out = {b: float(np.median(v)) for b, v in per_band.items()}
    out["__pooled__"] = float(np.median([v for b, v in out.items()
                                         if b != "__pooled__"]))
    return out


def test_A7_2_alpha_to_db():
    """A7-2: translate the alpha axis into 'Arm A per-bin log-mag error <= X dB'."""
    sf = _sigma_fine()
    pooled = sf["__pooled__"]
    print("  A7-2 sigma_fine (std of A_true - A_band, voiced, 10 recs):")
    for lo_hz, hi_hz in VSTAR_BANDS:
        print(f"    {lo_hz}-{hi_hz} Hz: {sf[(lo_hz,hi_hz)]:.2f} dB")
    print(f"    pooled median: {pooled:.2f} dB")
    print("  alpha -> Arm A per-bin log-mag error <= (1-alpha)*sigma_fine:")
    for alpha in ALPHAS:
        print(f"    alpha={alpha:.2f}  =>  <= {(1-alpha)*pooled:.2f} dB "
              f"(per-bin, vs band mean)")
    print(f"  Predeclared decision threshold: if the (beta,alpha) needed for "
          f"dG3rec>=0.30 requires per-bin error < 1.0 dB, that approaches "
          f"'directly reconstruct clean speech' => VPU-domain reconstruction "
          f"NOT engineering-viable.")


# ---------------------------------------------------------------------------
# A7-3: mark V_real's current (beta, alpha) position
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _vreal_split():
    """Split V_real's per-bin error into ①band-level tracking + ②fine-structure.

    ① = std(v_band - x_band) over voiced frames (band-mean energy tracking).
    ② = std((v_fine - x_fine)) over voiced (bin,frame) = std of V's fine
        structure vs X's, where v_fine = v - v_band, x_fine = x - x_band.
    The A* model's per-bin error is (1-alpha)*(A_true-A_band) = shrinkage of
    X's fine structure; it carries NO band-level error (A_band is from X, so
    V*'s band level is exact).  So only ② maps onto the alpha axis:
    (1-alpha_v)*sigma_fine = ②  =>  alpha_v = 1 - ②/sigma_fine.
    ① (band-level tracking, ~5 dB) is a SEPARATE account NOT on this chart.
    """
    cfg, records = _prepared()
    b1 = {b: [] for b in VSTAR_BANDS}; b2 = {b: [] for b in VSTAR_BANDS}
    for prep in records:
        sx = prep["spec_x"]; sv = prep["spec_v"]; conf = prep["conf_x"]
        voiced = conf[0] >= 0.55
        for lo_hz, hi_hz in VSTAR_BANDS:
            lo, hi = _band_bins(cfg, lo_hz, hi_hz)
            xdb = 20 * torch.log10(sx[0, lo:hi + 1].abs().clamp_min(1e-8))
            vdb = 20 * torch.log10(sv[0, lo:hi + 1].abs().clamp_min(1e-8))
            x_band = xdb.mean(0); v_band = vdb.mean(0)
            x_fine = xdb - x_band.unsqueeze(0); v_fine = vdb - v_band.unsqueeze(0)
            b1[(lo_hz, hi_hz)].append(float(((v_band - x_band)[voiced]).std()))
            b2[(lo_hz, hi_hz)].append(float(((v_fine - x_fine)[:, voiced]).std()))
    o1 = {b: float(np.median(v)) for b, v in b1.items()}
    o2 = {b: float(np.median(v)) for b, v in b2.items()}
    o1["__pooled__"] = float(np.median(list(o1.values())))
    o2["__pooled__"] = float(np.median(list(o2.values())))
    return o1, o2


def test_A7_3_vreal_position():
    """A7-3 (rework 1): V_real on the (beta, alpha) grid, split correctly.

    The alpha axis measures ONLY fine-structure precision (A_band is from X,
    so V*'s band level is exact at any alpha).  V_real's per-bin error mixes
    ①band-level tracking (~5 dB) + ②fine-structure (~4.7 dB).  Using the TOTAL
    error to position alpha conflates ① onto an axis that doesn't measure it.
    Correct: alpha_vreal = 1 - ②/sigma_fine; ① is a SEPARATE, lethal spec
    (band-level precision ~5 dB vs A6-1b deficit std ~0.3 dB => ~17x needed).
    """
    sf = _sigma_fine(); o1, o2 = _vreal_split()
    pooled_f = sf["__pooled__"]
    b1p = o1["__pooled__"]; b2p = o2["__pooled__"]
    alpha_v = 1.0 - b2p / pooled_f
    print("  A7-3 (rework 1) V_real split:")
    print(f"    sigma_fine(pooled)={pooled_f:.2f} dB")
    print(f"    ① band-level tracking (pooled)={b1p:.2f} dB  [NOT on the alpha axis]")
    print(f"    ② fine-structure tracking (pooled)={b2p:.2f} dB")
    print(f"    alpha_vreal = 1 - ②/sigma_fine = 1 - {b2p:.2f}/{pooled_f:.2f} = {alpha_v:+.3f}")
    print(f"    beta_vreal ~ 1.0 (every in-band bin carries some V content)")
    print(f"    => V_real ~ (beta=1.0, alpha={alpha_v:+.3f})  [② only; ① is separate]")
    print("  per-band:")
    for lo_hz, hi_hz in VSTAR_BANDS:
        a = 1 - o2[(lo_hz, hi_hz)] / sf[(lo_hz, hi_hz)]
        print(f"    {lo_hz}-{hi_hz} Hz: ①band={o1[(lo_hz,hi_hz)]:.2f} "
              f"②fine={o2[(lo_hz,hi_hz)]:.2f} sigma_f={sf[(lo_hz,hi_hz)]:.2f} "
              f"=> alpha_v={a:+.3f}")
    print(f"  --- TWO independent Arm A specs (rework 1) ---")
    print(f"  Spec 1 (band-level, LETHAL, NOT on (beta,alpha) chart): "
          f"V_real ①={b1p:.2f} dB; A6-1b band-level deficit std ~0.3 dB "
          f"=> need ~{b1p/0.3:.0f}x improvement to make the deficit detectable.")
    print(f"  Spec 2 (fine-structure, on the chart): V_real ②={b2p:.2f} dB "
          f"(alpha={alpha_v:+.2f}); need alpha>=0.75-0.84 => <=1.0-1.56 dB "
          f"=> ~{b2p/1.0:.0f}-{b2p/1.56:.0f}x improvement.")


# ---------------------------------------------------------------------------
# A7 decision: minimum (beta, alpha) for dG3rec >= 0.30, in dB
# ---------------------------------------------------------------------------

def test_A7_decision():
    """Predeclared: minimum (beta, alpha) for deltaG3rec >= 0.30 (3/4 of the
    0.39 ideal), converted to 'Arm A per-bin error <= X dB'.  If X < 1.0 dB,
    state VPU-domain reconstruction is not engineering-viable.
    """
    out = _scan_grid(); sf = _sigma_fine(); pooled_f = sf["__pooled__"]
    target = 0.30
    print(f"  A7 decision: minimum (beta, alpha) for dG3rec >= {target}")
    print(f"  (sigma_fine pooled = {pooled_f:.2f} dB => alpha-> (1-alpha)*{pooled_f:.2f} dB)")
    # find the min-effort (lowest beta then lowest alpha) reaching target
    reached = []
    for beta in BETAS:
        for alpha in ALPHAS:
            dg = float(np.median([r["dg"] for r in out[(beta, alpha)]]))
            if dg >= target:
                db = (1 - alpha) * pooled_f
                reached.append((beta, alpha, dg, db))
    if not reached:
        print(f"  NO grid point reaches dG3rec >= {target}; "
              f"max = {max(float(np.median([r['dg'] for r in out[(b,a)]])) for b in BETAS for a in ALPHAS):.4f}")
        return
    reached.sort(key=lambda x: (x[0], x[1]))
    print(f"  points reaching {target} (sorted by beta then alpha):")
    for beta, alpha, dg, db in reached:
        print(f"    beta={beta} alpha={alpha}  dG3={dg:.4f}  "
              f"=> Arm A per-bin error <= {db:.2f} dB")
    # the frontier: for each beta, the min alpha reaching target
    print("  frontier (min alpha per beta):")
    for beta in BETAS:
        alphas_ok = [a for a in ALPHAS
                     if float(np.median([r["dg"] for r in out[(beta, a)]])) >= target]
        if alphas_ok:
            a = alphas_ok[0]
            db = (1 - a) * pooled_f
            print(f"    beta={beta}: min alpha={a} => per-bin <= {db:.2f} dB")
        else:
            mx = max(ALPHAS, key=lambda a: float(np.median([r["dg"] for r in out[(beta, a)]])))
            print(f"    beta={beta}: none reach {target} (max dG3 at alpha={mx}="
                  f"{float(np.median([r['dg'] for r in out[(beta,mx)]])):.4f})")
    # the absolute minimum-effort point
    b0, a0, dg0, db0 = reached[0]
    print(f"  => minimum-effort: beta={b0}, alpha={a0} (dG3={dg0:.4f}); "
          f"Arm A per-bin log-mag error <= {db0:.2f} dB; "
          f"coverage beta={b0} (fraction of in-band bins with true info)")
    if db0 < 1.0:
        print(f"  *** per-bin precision < 1.0 dB required => approaches "
              f"'directly reconstruct clean speech' difficulty => VPU-domain "
              f"reconstruction NOT engineering-viable. ***")
    else:
        print(f"  per-bin precision {db0:.2f} dB >= 1.0 dB => within reach of "
              f"domain reconstruction (not clean-speech-hard).")


# ---------------------------------------------------------------------------
# A7 rework 2: σ_e / σ_b RANDOM-residual axes (model specs, not shrinkage)
# ---------------------------------------------------------------------------
#
# The alpha axis models SHRINKAGE: A* = alpha*A_true + (1-alpha)*A_band, whose
# error (1-alpha)*(A_band-A_true) is perfectly ANTI-correlated with the true
# fine structure (right direction, compressed amplitude) => oracle w can partly
# compensate by amplifying.  A real model's residual is RANDOM (wrong direction)
# => w cannot compensate => same std, random is strictly worse than shrinkage.
# So the alpha->dB spec is OPTIMISTIC.  These two axes add RANDOM perturbation:
#   σ_e: per-bin zero-mean Gaussian noise on A_true (fine-structure residual)
#   σ_b: per-(band,frame) zero-mean Gaussian noise on the band mean (band-level
#        residual), σ_e=0 (perfect fine structure) => isolates band-level.
# Both: xband fill (β=1.0 => all bins info, fill irrelevant), band-scalar oracle,
# d20, matched B=9 null (time-permuted truth + redrawn noise) at every grid.

SIGMA_E = [0.5, 1.0, 2.0, 4.0, 6.0]
SIGMA_B = [0.0, 0.1, 0.3, 1.0, 3.0]


def _build_vstar_perturb(prep, sigma_e=0.0, sigma_b=0.0,
                         permutation_seed=None, permute_true=False, noise_seed=0):
    """V* = A_true + N(0,σ_e) per bin + N(0,σ_b) per (band,frame), V_real phase.

    β=1.0 (all in-band bins are 'info').  σ_e perturbs fine structure
    (independent per bin); σ_b perturbs the band level (one draw per
    (band,frame), applied to all bins in that band) with fine structure exact
    (σ_e=0) => isolates band-level.  permute_true (null) time-permutes A_true;
    noise_seed redraws the noise so each null draw is independent.
    """
    cfg = FusionConfig(); sx = prep["spec_x"]; sv = prep["spec_v"]
    out = sv.clone(); n_frames = sx.shape[-1]
    mapping = (torch.arange(n_frames) if permutation_seed is None
               else _within_state_permutation(prep["conf_v"][0], permutation_seed))
    lo0, hi0 = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
    bins = slice(lo0, hi0 + 1)
    a_true_all = 20 * torch.log10(sx[0, bins].abs().clamp_min(1e-8))  # [nb, nf]
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
    mag = 10 ** (mag_log / 20)
    out[0, bins] = mag * torch.exp(1j * torch.angle(sv[0, bins]))
    return out


def _perturb_metric(prep, sigma_e, sigma_b, depth=20, noise_seed=0):
    obs = _build_vstar_perturb(prep, sigma_e, sigma_b, noise_seed=noise_seed)
    return _oracle_metric_for_spec(prep, obs, depth, eq_mode="zero", direct=True,
                                   band_indices=range(4), oracle_mode="band")


def _perturb_axis(sigma_e_sweep, sigma_b_sweep, label):
    """Sweep one axis (fixing the other at 0), matched B=9 null, 10 recs."""
    _, records = _prepared()
    out = {}
    for s in (sigma_e_sweep if sigma_e_sweep else sigma_b_sweep):
        rows = []
        for prep in records:
            se = s if sigma_e_sweep else 0.0
            sb = s if sigma_b_sweep else 0.0
            o = _perturb_metric(prep, se, sb, noise_seed=prep["index"] + 31)
            null = []
            for b in range(B):
                # null: time-permuted truth + redrawn noise (no real alignment)
                ns = _build_vstar_perturb(
                    prep, se, sb,
                    permutation_seed=91000 + 1000 * b + prep["index"],
                    permute_true=True, noise_seed=91000 + 1000 * b + prep["index"] + 1)
                null.append(_oracle_metric_for_spec(
                    prep, ns, 20, eq_mode="zero", direct=True,
                    band_indices=range(4), oracle_mode="band"))
            gmed = float(np.median([r["recovery"] for r in null]))
            jmed = float(np.median([r["j3"] for r in null]))
            rows.append(dict(dg=o["recovery"] - gmed, dj=o["j3"] - jmed))
        out[s] = (float(np.median([r["dg"] for r in rows])),
                  float(np.median([r["dj"] for r in rows])))
    return out


@lru_cache(maxsize=4)
def _perturb_axis_cached(axis):
    """axis in {'e','b'}; cached so the decision test doesn't recompute."""
    if axis == "e":
        return _perturb_axis(tuple(SIGMA_E), None, "sigma_e")
    return _perturb_axis(None, tuple(SIGMA_B), "sigma_b")


def test_A7_rework2_sigma_e_axis():
    """σ_e axis: per-bin random fine-structure residual (β=1.0, σ_b=0)."""
    out = _perturb_axis_cached("e")
    print("  A7 rework 2 — σ_e axis (per-bin random residual, β=1.0, σ_b=0, B=9 null):")
    print("  σ_e(dB)  dG3rec  dJ3")
    for s in SIGMA_E:
        dg, dj = out[s]
        print(f"  {s:>6}  {dg:+.4f}  {dj:+.4f}")
    target = 0.30
    ok = [s for s in SIGMA_E if out[s][0] >= target]
    print(f"  reach dG3rec>={target}: σ_e in {ok or 'none'}")
    if ok:
        s_max = max(ok)
        print(f"  => Arm A fine-structure random residual <= {s_max} dB "
              f"(largest σ_e still reaching {target})")
        if s_max < 1.0:
            print(f"  *** σ_e < 1.0 dB => fine-structure requirement approaches "
                  f"'directly reconstruct clean speech' ***")


def test_A7_rework2_sigma_b_axis():
    """σ_b axis: per-(band,frame) random band-level residual (β=1.0, σ_e=0)."""
    out = _perturb_axis_cached("b")
    print("  A7 rework 2 — σ_b axis (band-level random residual, β=1.0, σ_e=0, B=9 null):")
    print("  σ_b(dB)  dG3rec  dJ3")
    for s in SIGMA_B:
        dg, dj = out[s]
        print(f"  {s:>6}  {dg:+.4f}  {dj:+.4f}")
    target = 0.30
    ok = [s for s in SIGMA_B if out[s][0] >= target]
    print(f"  reach dG3rec>={target}: σ_b in {ok or 'none'}")
    if ok:
        s_max = max(ok)
        print(f"  => Arm A band-level random residual <= {s_max} dB")
        if s_max < 0.5:
            print(f"  *** σ_b < 0.5 dB => band-level requirement approaches "
                  f"'directly know clean-speech band energy'; corroborates "
                  f"A6-1b's ~0.3 dB deficit std => VPU-domain reconstruction "
                  f"NOT engineering-viable ***")


def test_A7_rework2_decision():
    """Which spec binds first?  Cross-reference σ_e and σ_b axes."""
    se_out = _perturb_axis_cached("e")
    sb_out = _perturb_axis_cached("b")
    target = 0.30
    se_ok = [s for s in SIGMA_E if se_out[s][0] >= target]
    sb_ok = [s for s in SIGMA_B if sb_out[s][0] >= target]
    print("  A7 rework 2 decision (which binds first for dG3rec>=0.30):")
    print(f"  σ_e fine-structure: reach at σ_e<={max(se_ok) if se_ok else 'NONE'} dB "
          f"(need <1.0 = clean-speech-hard: {'YES' if se_ok and max(se_ok)<1.0 else 'no'})")
    print(f"  σ_b band-level:      reach at σ_b<={max(sb_ok) if sb_ok else 'NONE'} dB "
          f"(need <0.5 = clean-speech-hard: {'YES' if sb_ok and max(sb_ok)<0.5 else 'no'})")
    if sb_ok and max(sb_ok) < 0.5 and (not se_ok or max(se_ok) >= 1.0):
        print(f"  => σ_b (band-level) BINDS FIRST; it is the lethal spec. "
              f"Arm A acceptance: band-level residual <= {max(sb_ok)} dB "
              f"AND per-bin random residual <= {max(se_ok) if se_ok else 'N/A'} dB.")
    elif se_ok and max(se_ok) < 1.0 and (not sb_ok or max(sb_ok) >= 0.5):
        print(f"  => σ_e (fine-structure) BINDS FIRST.  Arm A acceptance: "
              f"per-bin random <= {max(se_ok)} dB AND band-level <= "
              f"{max(sb_ok) if sb_ok else 'N/A'} dB.")
    else:
        print(f"  => both near their clean-speech-hard thresholds; report both.")
