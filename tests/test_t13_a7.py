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
from tests.test_t13_a5 import VSTAR_BANDS, _prepared, build_vstar, _oracle_metric_for_spec
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
def _vreal_sigma():
    """sigma_vreal = std of (V_real - X) per bin (the tracking error), voiced."""
    cfg, records = _prepared()
    per_band = {b: [] for b in VSTAR_BANDS}
    for prep in records:
        sx = prep["spec_x"]; sv = prep["spec_v"]; conf = prep["conf_x"]
        voiced = conf[0] >= 0.55
        for lo_hz, hi_hz in VSTAR_BANDS:
            lo, hi = _band_bins(cfg, lo_hz, hi_hz)
            xdb = 20 * torch.log10(sx[0, lo:hi + 1].abs().clamp_min(1e-8))
            vdb = 20 * torch.log10(sv[0, lo:hi + 1].abs().clamp_min(1e-8))
            dev = (vdb - xdb)[:, voiced].detach().cpu().numpy()
            per_band[(lo_hz, hi_hz)].append(float(dev.std()))
    out = {b: float(np.median(v)) for b, v in per_band.items()}
    out["__pooled__"] = float(np.median([v for b, v in out.items()
                                         if b != "__pooled__"]))
    return out


def test_A7_3_vreal_position():
    """A7-3: where does V_real sit on the (beta, alpha) grid?

    V_real has beta~1 (every in-band bin carries *some* content) but poor
    per-bin precision.  Map its tracking error sigma_vreal onto the alpha axis:
    (1-alpha_v)*sigma_fine = sigma_vreal  =>  alpha_v = 1 - sigma_vreal/sigma_fine.
    """
    sf = _sigma_fine(); sv = _vreal_sigma()
    pooled_f = sf["__pooled__"]; pooled_v = sv["__pooled__"]
    alpha_v = 1.0 - pooled_v / pooled_f
    print("  A7-3 V_real position on the (beta, alpha) grid:")
    print(f"    sigma_fine(pooled)={pooled_f:.2f} dB  "
          f"sigma_vreal(pooled)={pooled_v:.2f} dB")
    print(f"    alpha_vreal = 1 - {pooled_v:.2f}/{pooled_f:.2f} = {alpha_v:.3f}")
    print(f"    beta_vreal ~ 1.0 (every in-band bin carries some V content)")
    print(f"    => V_real ~ (beta=1.0, alpha={alpha_v:.3f})")
    if alpha_v < 0:
        print(f"    alpha_v<0: V_real is WORSE than alpha=0 (band mean) -- its "
              f"per-bin noise exceeds the spectral fine structure; Arm A must "
              f"first reach alpha=0 (band-mean fidelity) before improving.")
    print("  per-band V_real alpha:")
    for lo_hz, hi_hz in VSTAR_BANDS:
        a = 1 - sv[(lo_hz, hi_hz)] / sf[(lo_hz, hi_hz)]
        print(f"    {lo_hz}-{hi_hz} Hz: sigma_v={sv[(lo_hz,hi_hz)]:.2f} "
              f"sigma_f={sf[(lo_hz,hi_hz)]:.2f} => alpha_v={a:.3f}")


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
