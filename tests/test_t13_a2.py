"""T13 A2 — G3a' sample adequacy, band-gain oracle, and G4' attribution.

All X-dependent logic is evaluation-only.  Algorithm modules remain unchanged
and are still protected by the static no-X check.

BOUNDARY: conclusions apply only to the four male speakers in 0624/0625
(median F0 87–124 Hz), all at normal speaking volume.  No 0625 speech is read.
"""
from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch

from fusion import Fusion, FusionConfig, realdata
from fusion.degrade import DegradationConfig
from fusion.stft import stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_b1 import (
    BAND_EDGES_HZ,
    G3A_MIN_SAMPLES,
    _band_bins,
    _measure_G3aprime_recovery_curve,
    _need,
)


REPORT_DIR = "reports/T13A2"
DEPTHS = [0, 3, 6, 10, 15, 20, 30]


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def test_A20_g3a_sample_adequacy():
    """Only P sets with n_sup>=30 are eligible for the existential gate."""
    rows = _measure_G3aprime_recovery_curve()
    statuses = {r[0]: ("ELIGIBLE" if r[4] >= G3A_MIN_SAMPLES else "INSUFFICIENT")
                for r in rows}
    adequacy_ok = (G3A_MIN_SAMPLES == 30 and
                   all((r[4] >= 30) == (statuses[r[0]] == "ELIGIBLE") for r in rows))
    print(f"  A2-0 G3a' sample adequacy (minimum={G3A_MIN_SAMPLES}): "
          + ", ".join(f"d{r[0]} n={r[4]} {statuses[r[0]]}" for r in rows))
    assert adequacy_ok, "A2-0: G3a' eligibility does not enforce n_sup>=30"


def test_A20_min_sample_mutation():
    """Mutation: lower the eligibility floor 30→1; an under-30 set must be rejected."""
    rows = _measure_G3aprime_recovery_curve()
    mutant_min = 1  # MUTATION of G3A_MIN_SAMPLES=30
    mutant_eligible = [r for r in rows if r[4] >= mutant_min]
    broken = False; failure = ""
    try:
        for r in mutant_eligible:
            assert r[4] >= G3A_MIN_SAMPLES, (
                f"A2-0 adequacy: depth {r[0]} entered with n_sup={r[4]}<30")
    except AssertionError as exc:
        broken = True; failure = str(exc)
    print("  A2-0 mutation: changed minimum n_sup 30→1; under-30 depths must still fail; "
          f"eligible depths={[r[0] for r in mutant_eligible]}; failure={failure!r}")
    assert broken, "A2-0 mutation: lowering n_sup floor to 1 was not detected"


@lru_cache(maxsize=1)
def _measure_sup_distributions():
    _need(); cfg = FusionConfig()
    ff, _, _ = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    rows = []
    for depth in DEPTHS:
        spec_x, spec_s, _ = eval_specs(
            ff, cfg, DegradationConfig(
                d1_kill_rate=0.4, d1_kill_depth_db=float(depth)))
        values = []
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            values.extend((10.0 * torch.log10(px / ps)).cpu().tolist())
        a = np.asarray(values, dtype=np.float64)
        rows.append(dict(
            depth=depth, values=a, n=len(a),
            p50=float(np.percentile(a, 50)), p75=float(np.percentile(a, 75)),
            p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
            max=float(a.max()),
            gt1=float(np.mean(a > 1)), gt3=float(np.mean(a > 3)),
            gt6=float(np.mean(a > 6)), gt10=float(np.mean(a > 10)),
        ))
    return rows


def _plot_sup_histograms(rows):
    _ensure_report_dir()
    from pathlib import Path
    width, height = 1000, 980; panel_w, panel_h = 460, 210
    bins = np.linspace(-5, 35, 81); histograms = [np.histogram(r["values"], bins)[0] for r in rows]
    ymax = max(1, max(int(h.max()) for h in histograms))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="500" y="28" text-anchor="middle" font-size="20">A2 sup_dB distribution — 0624 male / normal volume</text>']
    for idx, (row, hist) in enumerate(zip(rows, histograms)):
        col, line = idx % 2, idx // 2; x0, y0 = 45 + col * 500, 55 + line * 225
        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#888"/>')
        svg.append(f'<text x="{x0+8}" y="{y0+18}" font-size="14">depth {row["depth"]} dB (n={row["n"]})</text>')
        for bi, count in enumerate(hist):
            bh = (float(count) / ymax) * (panel_h - 35); bx = x0 + bi * panel_w / len(hist)
            svg.append(f'<rect x="{bx:.2f}" y="{y0+panel_h-bh:.2f}" width="{panel_w/len(hist):.2f}" height="{bh:.2f}" fill="#4472C4"/>')
        for threshold, color in [(1, "#70AD47"), (3, "#FFC000"), (6, "#ED7D31"), (10, "#C00000")]:
            tx = x0 + (threshold + 5) / 40 * panel_w
            svg.append(f'<line x1="{tx:.2f}" y1="{y0}" x2="{tx:.2f}" y2="{y0+panel_h}" stroke="{color}" stroke-width="2"/>')
    svg.append('<text x="500" y="970" text-anchor="middle" font-size="15">sup_dB = 10log10(mean|X|² / mean|S|²); lines: 1 / 3 / 6 / 10 dB</text>')
    svg.append('</svg>')
    Path(f"{REPORT_DIR}/sup_db_histograms.svg").write_text("\n".join(svg), encoding="utf-8")


def test_A21_sup_distribution_report():
    """A2-1 independent evidence for later P/U threshold review; no retuning."""
    rows = _measure_sup_distributions(); _plot_sup_histograms(rows)
    print("  A2-1 sup_dB distribution (same segment and P definition as G3a'):")
    print("  depth   p50   p75   p90   p95    max    >1     >3     >6    >10")
    for r in rows:
        print(f"  {r['depth']:>5} {r['p50']:>5.2f} {r['p75']:>5.2f} "
              f"{r['p90']:>5.2f} {r['p95']:>5.2f} {r['max']:>6.2f} "
              f"{r['gt1']:>6.2%} {r['gt3']:>6.2%} {r['gt6']:>6.2%} {r['gt10']:>6.2%}")
    print(f"  plot → {REPORT_DIR}/sup_db_histograms.svg")
    finite_ok = all(np.isfinite(r["values"]).all() and r["n"] > 0 for r in rows)
    assert finite_ok, "A2-1: sup_dB distribution contains non-finite or empty data"


def _plot_oracle(rows):
    _ensure_report_dir()
    from pathlib import Path
    d = [r[0] for r in rows if r[4] > 0]
    actual = [r[3] for r in rows if r[4] > 0]
    oracle = [r[8] for r in rows if r[4] > 0]
    gap = [r[9] for r in rows if r[4] > 0]
    width, height = 900, 520; x0, y0, pw, ph = 80, 55, 740, 380
    xmax = max(d); ymax = max(1.0, max(actual + oracle + gap) * 1.1)
    x = lambda value: x0 + value / xmax * pw
    y = lambda value: y0 + ph - value / ymax * ph
    def poly(values):
        return " ".join(f"{x(di):.2f},{y(vi):.2f}" for di, vi in zip(d, values))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="450" y="28" text-anchor="middle" font-size="20">A2 band-gain oracle ceiling — 0624 male / normal volume</text>',
           f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#666"/>',
           f'<line x1="{x0}" y1="{y(0.5):.2f}" x2="{x0+pw}" y2="{y(0.5):.2f}" stroke="#111" stroke-dasharray="6,4"/>',
           f'<polyline points="{poly(actual)}" fill="none" stroke="#4472C4" stroke-width="3"/>',
           f'<polyline points="{poly(oracle)}" fill="none" stroke="#70AD47" stroke-width="3"/>',
           f'<polyline points="{poly(gap)}" fill="none" stroke="#C00000" stroke-width="3"/>']
    for label, values, color in [("actual", actual, "#4472C4"), ("oracle", oracle, "#70AD47"), ("normalized gap", gap, "#C00000")]:
        for di, vi in zip(d, values):
            svg.append(f'<circle cx="{x(di):.2f}" cy="{y(vi):.2f}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{x0+15}" y="{455 + 18 * [("actual", actual, "#4472C4"), ("oracle", oracle, "#70AD47"), ("normalized gap", gap, "#C00000")].index((label, values, color))}" font-size="14" fill="{color}">{label}</text>')
    for di in d:
        svg.append(f'<text x="{x(di):.2f}" y="{y0+ph+20}" text-anchor="middle" font-size="12">{di}</text>')
    svg.extend(['<text x="450" y="510" text-anchor="middle" font-size="14">D1 depth (dB)</text>',
                '<text x="15" y="260" transform="rotate(-90 15 260)" text-anchor="middle" font-size="14">ratio / normalized gap</text>',
                '</svg>'])
    Path(f"{REPORT_DIR}/g3a_oracle_ceiling.svg").write_text("\n".join(svg), encoding="utf-8")


def test_A22_oracle_band_ceiling_report():
    """A2-2 evaluation-only optimum uniform dB offset for every P point."""
    rows = _measure_G3aprime_recovery_curve(); _plot_oracle(rows)
    print("  A2-2 band-level oracle (same equal-weight P aggregation as G3a'):")
    print("  depth n_sup status        actual oracle norm_gap")
    for r in rows:
        status = "ELIGIBLE" if r[4] >= G3A_MIN_SAMPLES else "INSUFFICIENT"
        print(f"  {r[0]:>5} {r[4]:>5} {status:12s} {r[3]:>6.5f} "
              f"{r[8]:>6.5f} {r[9]:>8.5f}")
    print(f"  plot → {REPORT_DIR}/g3a_oracle_ceiling.svg")
    # g*=0 is always available, so the analytic optimum cannot be worse than S.
    oracle_ok = all(r[4] == 0 or r[8] <= 1.0 + 1e-6 for r in rows)
    assert oracle_ok, "A2-2: computed oracle is worse than the no-correction S arm"


def _record_labels(name):
    stem = name.removeprefix("FB_FF_TT_VPU_").removesuffix(".wav")
    speaker = stem.split("_", 1)[0]
    position = stem.split("left_ear_", 1)[-1]
    return speaker, position


def _counter():
    return dict(u=0, violations=0, j2=0)


@lru_cache(maxsize=1)
def _measure_g4_j2_attribution():
    """Run all ten 0624 recordings and expose only test-side diagnostics."""
    _need(); cfg = FusionConfig()
    by_depth = defaultdict(_counter); by_band = defaultdict(_counter)
    by_record = defaultdict(_counter); by_time = defaultdict(_counter); triples = []
    g4_keys = set(); j2_keys = set()
    for path in realdata.list_0624():
        name = os.path.basename(path); speaker, position = _record_labels(name)
        ff, vpu, _ = realdata.load_0624(name=name, seg_s=6.0, offset_s=1.0)
        for depth in DEPTHS:
            spec_x, spec_s, S = eval_specs(
                ff, cfg, DegradationConfig(
                    d1_kill_rate=0.4, d1_kill_depth_db=float(depth)))
            fusion = Fusion(cfg); Y = fusion.process_batch(S, vpu)
            spec_y = stft_batch(Y, cfg)
            w_spec = torch.stack(fusion.core.w_history, dim=-1)
            for bi in range(len(BAND_EDGES_HZ) - 1):
                lo, hi = _band_bins(cfg, BAND_EDGES_HZ[bi], BAND_EDGES_HZ[bi + 1])
                band = f"{BAND_EDGES_HZ[bi]}-{BAND_EDGES_HZ[bi + 1]}"
                xs = 20 * torch.log10(spec_x[0, lo:hi + 1].abs().clamp_min(1e-8))
                ss = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
                ys = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
                px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
                ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
                sup = 10.0 * torch.log10(px / ps)
                lsd_s = torch.sqrt(((ss - xs) ** 2).mean(0))
                lsd_y = torch.sqrt(((ys - xs) ** 2).mean(0))
                corr_bins = ys - ss
                for t in range(spec_s.shape[-1]):
                    key = (name, depth, bi, t)
                    sup_t = float(sup[t]); excess = float(lsd_y[t] - lsd_s[t] - 0.3)
                    corr_signed = float(corr_bins[:, t].mean())
                    corr_abs = float(corr_bins[:, t].abs().mean())
                    err_s = float((ss[:, t] - xs[:, t]).mean())
                    w_mean = float(w_spec[0, lo:hi + 1, t].mean())
                    is_u = sup_t <= 1.0
                    is_g4 = is_u and excess > 0.0
                    # Preserve the existing J2 implementation exactly:
                    # not mean-log suppressed by >6 dB, and |corr| mean >3 dB.
                    is_j2 = err_s >= -6.0 and corr_abs > 3.0
                    time_bin = f"{t // 100}-{t // 100 + 1}s"
                    groups = (by_depth[depth], by_band[band],
                              by_record[f"{speaker}/{position}"], by_time[time_bin])
                    for group in groups:
                        group["u"] += int(is_u); group["violations"] += int(is_g4)
                        group["j2"] += int(is_j2)
                    if is_g4:
                        g4_keys.add(key)
                        target = -err_s
                        wrong_direction = (abs(target) > 1e-12 and abs(corr_signed) > 1e-12
                                           and target * corr_signed < 0.0)
                        triples.append(dict(
                            key=key, depth=depth, band=band, speaker=speaker,
                            position=position, sup=sup_t, excess=excess,
                            w=w_mean, corr=corr_signed, corr_abs=corr_abs,
                            err_s=err_s, wrong_direction=wrong_direction,
                        ))
                    if is_j2:
                        j2_keys.add(key)
    overlap = g4_keys & j2_keys
    return dict(
        by_depth=dict(by_depth), by_band=dict(by_band), by_record=dict(by_record),
        by_time=dict(by_time),
        triples=triples, n_g4=len(g4_keys), n_j2=len(j2_keys),
        n_overlap=len(overlap),
        g4_covered=len(overlap) / max(1, len(g4_keys)),
        j2_covered=len(overlap) / max(1, len(j2_keys)),
        jaccard=len(overlap) / max(1, len(g4_keys | j2_keys)),
    )


def _pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if a else float("nan")


def test_A23_g4_j2_attribution_report():
    """A2-3 G4' tail attribution; report only, no parameter or registry change."""
    out = _measure_g4_j2_attribution(); triples = out["triples"]
    print("  A2-3 G4' violations by depth (all ten 0624 recordings):")
    print("  depth       U violations    rate      J2")
    for depth in DEPTHS:
        r = out["by_depth"][depth]
        print(f"  {depth:>5} {r['u']:>7} {r['violations']:>10} "
              f"{r['violations']/max(1,r['u']):>7.2%} {r['j2']:>7}")
    print("  A2-3 by band:")
    print("  band             U violations    rate      J2")
    for band in [f"{a}-{b}" for a, b in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]:
        r = out["by_band"][band]
        print(f"  {band:13s} {r['u']:>7} {r['violations']:>10} "
              f"{r['violations']/max(1,r['u']):>7.2%} {r['j2']:>7}")
    print("  A2-3 by speaker/position:")
    print("  recording             U violations    rate      J2")
    for label, r in sorted(out["by_record"].items()):
        print(f"  {label:20s} {r['u']:>7} {r['violations']:>10} "
              f"{r['violations']/max(1,r['u']):>7.2%} {r['j2']:>7}")
    print("  A2-3 by time (100 frames = 1 s):")
    print("  time             U violations    rate      J2")
    for label, r in sorted(out["by_time"].items(), key=lambda item: int(item[0].split("-")[0])):
        print(f"  {label:8s} {r['u']:>9} {r['violations']:>10} "
              f"{r['violations']/max(1,r['u']):>7.2%} {r['j2']:>7}")

    w = [r["w"] for r in triples]; corr = [r["corr"] for r in triples]
    corr_abs = [r["corr_abs"] for r in triples]; sup = [r["sup"] for r in triples]
    excess = [r["excess"] for r in triples]
    nonzero_w = np.mean(np.asarray(w) > 0.0) if w else float("nan")
    wrong = np.mean([r["wrong_direction"] for r in triples]) if triples else float("nan")
    print("  A2-3 violation triples (median / p90 / max):")
    for label, values in [("w", w), ("corr_signed", corr), ("|corr|", corr_abs),
                          ("sup_dB", sup), ("excess", excess)]:
        print(f"    {label:12s}: {_pct(values,50):.4f} / {_pct(values,90):.4f} / "
              f"{max(values) if values else float('nan'):.4f}")
    print(f"    w>0 exact share={nonzero_w:.2%}; corr wrong-direction share={wrong:.2%}")
    print("  A2-3 violation diagnostics by band:")
    print("  band          n   w_med |corr|med |corr|p90 |corr|max excess_max wrong_dir")
    for band in [f"{a}-{b}" for a, b in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]:
        br = [r for r in triples if r["band"] == band]
        babs = [r["corr_abs"] for r in br]
        print(f"  {band:11s} {len(br):>6} {_pct([r['w'] for r in br],50):>7.4f} "
              f"{_pct(babs,50):>9.4f} {_pct(babs,90):>9.4f} "
              f"{max(babs) if babs else float('nan'):>9.4f} "
              f"{max([r['excess'] for r in br]) if br else float('nan'):>10.4f} "
              f"{np.mean([r['wrong_direction'] for r in br]) if br else float('nan'):>9.2%}")
    print("  A2-3 G4'/J2 overlap: "
          f"G4={out['n_g4']} J2={out['n_j2']} intersection={out['n_overlap']} "
          f"G4-covered={out['g4_covered']:.2%} J2-covered={out['j2_covered']:.2%} "
          f"Jaccard={out['jaccard']:.2%}")
    ok = bool(triples) and 0.0 <= out["jaccard"] <= 1.0
    assert ok, "A2-3: attribution produced no G4 violations or invalid overlap"
