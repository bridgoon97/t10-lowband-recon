"""T13-N2 — public GTCRN ONNX denoising of the raw VPU, fed into the accepted
N1 structure.  OFFLINE data prep (this module never touches fusion/ internals;
static-checked).  All tensor dims explicit; LSD finite/non-empty; every
mutation actually invoked.

Fixed-gain protocol (pre-locked): raw | rms→-30 | rms→-24 | peak→-6, whole-clip
scalars, V_dn = GTCRN(g·V)/g.  Pre-fixed criteria for "worth real listening"
(all three, else the conclusion is that public GTCRN does not repair the VPU
branch premise):
  1. at L=10 or 15: N1 optimal p>0 AND best |valley error| improves >= 0.30 dB
     vs raw-V (median of the 10 per-recording differences);
  2. same-cell peak |error| worsens <= 0.50 dB vs raw-V;
  3. 100-800 Hz LSD worsens <= 0.50 dB vs raw-V.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from fusion import FusionConfig, Fusion, degrade, DegradationConfig, stft_batch
from fusion.trust import TrustSource
from fusion.realdata import list_0624
from scripts.gtcrn_denoise import (GAINS, GtcrnDenoiser, check_model, run_gain,
                                   bypass_diff)

SR = 16000
OUT = Path("reports/T13N2")
SEG_S, OFFSET_S = 8.0, 2.0          # same deterministic segments as the N1 scan
LS = [40, 25, 15, 10]
PS = [0.0, 0.25, 0.5, 0.75, 1.0]
_CFG = FusionConfig().with_switches(decision_mode="n1")


def _db(spec):
    return 20 * torch.log10(spec.abs().clamp_min(1e-8))


def _load_pair(name):
    path = next(p for p in list_0624() if Path(p).name == name)
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    i0 = int(OFFSET_S * SR)
    x = torch.from_numpy(y[i0:i0 + int(SEG_S * SR), 1].copy()).unsqueeze(0)
    v = torch.from_numpy(y[i0:i0 + int(SEG_S * SR), 3].copy()).unsqueeze(0)
    return x, v


def _regions(x, cfg=_CFG, w=1):
    from fusion.f0 import f0_batch
    f0, conf = f0_batch(x, cfg)
    Fb, N = cfg.n_fft // 2 + 1, f0.shape[1]
    bz = cfg.sr / cfg.n_fft
    pk = torch.zeros(Fb, N, dtype=torch.bool)
    for t in range(N):
        f0t = float(f0[0, t])
        if f0t <= 0 or float(conf[0, t]) < 0.5:
            continue
        k = 1
        while k * f0t < cfg.n1_wband_zero_hi_hz:
            b = int(round(k * f0t / bz))
            if 1 <= b < Fb:
                pk[max(1, b - w):min(Fb, b + w + 1), t] = True
            k += 1
    inband = torch.zeros(Fb, dtype=torch.bool)
    inband[1:int(cfg.n1_wband_zero_hi_hz / bz) + 1] = True
    vl = inband.unsqueeze(-1) & ~pk
    return vl, pk, (f0[0] > 0) & (conf[0] >= 0.5)


def _lsd(y, x, lo_hz, hi_hz):
    """Band LSD over the FREQUENCY axis (axis 1 of (B,F,N)); guarded
    finite/non-empty (the N1-rework blocker must not recur here)."""
    bz = _CFG.sr / _CFG.n_fft
    ly = _db(stft_batch(y, _CFG)); lx = _db(stft_batch(x, _CFG))
    lo = max(1, int(lo_hz / bz)); hi = min(_CFG.fusion_hi_bin, int(hi_hz / bz))
    d = ly[:, lo:hi + 1, :] - lx[:, lo:hi + 1, :]
    n = int(d.numel())
    val = float(d.pow(2).mean().sqrt()) if n else float("nan")
    if not np.isfinite(val) or n == 0:
        raise ValueError(f"LSD guard: empty ({n}) or non-finite ({val})")
    return val


def _cell(y, x, vl, pk):
    """(peak_err, valley_err, hc_peak, hc_valley) — peak and valley SEPARATE."""
    dly = _db(stft_batch(y, _CFG))[0]; dlx = _db(stft_batch(x, _CFG))[0]
    return (float((dly - dlx)[pk].mean()), float((dly - dlx)[vl].mean()),
            float(dly[pk].median() - dlx[pk].median()),
            float(dly[vl].median() - dlx[vl].median()))


def _run_n1(s, v, p):
    f = Fusion(_CFG)
    f.set_trust(TrustSource(source="manual", const=p))
    with torch.no_grad():
        return f.process_batch(s, v)


def _f0_stats(v):
    from fusion.f0 import f0_batch
    f0, conf = f0_batch(v, _CFG)
    return (float(conf.median()),
            float(((f0[0] > 0) & (conf[0] >= 0.5)).float().mean()))


# ------------------------------------------------------- provenance/bypass ---
def test_N2_model_provenance_and_deps():
    """Model provenance (URL/SHA256/bytes) and runtime deps recorded."""
    import sherpa_onnx
    import onnxruntime
    OUT.mkdir(parents=True, exist_ok=True)
    prov = check_model()
    assert prov["sha256_match"], f"model SHA256 mismatch: {prov}"
    print(f"  model: {prov['url']}")
    print(f"  sha256={prov['sha256']} bytes={prov['bytes']}")
    print(f"  deps: sherpa_onnx {sherpa_onnx.__version__}, "
          f"onnxruntime {onnxruntime.__version__}")
    (OUT / "model_provenance.json").write_text(json.dumps(
        {**prov, "sherpa_onnx": sherpa_onnx.__version__,
         "onnxruntime": onnxruntime.__version__}, indent=1))


def test_N2_fixed_gain_bypass_and_mutation():
    """Four fixed gains on a real 0624 VPU segment: g, in/out peak/RMS/DC,
    NaN/Inf, clip count; bypass control <=1e-6 asserted per gain; a mutation
    that skips the /g MUST fail the bypass check."""
    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(list_0624()[0]).name
    x, v = _load_pair(name)
    v1 = v[0].numpy()
    d = GtcrnDenoiser()
    rows = []
    for mode in GAINS:
        r = run_gain(v1, mode, d)
        assert r.bypass_max_diff <= 1e-6, (f"bypass control FAILED for {mode}: "
                                           f"{r.bypass_max_diff:.2e} > 1e-6")
        dm = bypass_diff(v1, r.g, skip_divide=True)   # mutation: forgot the /g
        if r.g != 1.0:   # raw: g=1, divide is identity — mutation vacuous there
            assert dm > 1e-6, (f"mutation sanity FAILED for {mode}: "
                               f"skip-divide diff {dm:.2e}")
        rows.append(dict(mode=r.mode, g=r.g, valid=r.valid, in_peak=r.in_peak,
                         in_rms=r.in_rms, out_peak=r.out_peak, out_rms=r.out_rms,
                         in_dc=r.in_dc, out_dc=r.out_dc, nan_inf=r.nan_inf,
                         clipped=r.clipped, bypass_max_diff=r.bypass_max_diff,
                         mutation_skip_divide_diff=dm))
        print(f"  {mode:>8}: g={r.g:8.3f} valid={r.valid} in_peak={r.in_peak:.3f} "
              f"out_peak={r.out_peak:.3f} rms {r.in_rms:.4f}->{r.out_rms:.4f} "
              f"dc {r.in_dc:+.1e}->{r.out_dc:+.1e} nan={r.nan_inf} clip={r.clipped} "
              f"bypass={r.bypass_max_diff:.1e} mut={dm:.1e}")
    (OUT / "fixed_gain_report.json").write_text(json.dumps(rows, indent=1))


# --------------------------------------------------- A: 0624 denoise metrics --
def test_N2_A_denoise_metrics_0624():
    """A: for raw V and each valid V_dn, per recording: LSD (100-800,
    800-2k), peak/valley error, HC split (peak and valley separately),
    F0 confidence / voiced coverage."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = GtcrnDenoiser()
    files = [Path(p).name for p in list_0624()]
    table = {}
    for name in files:
        x, v = _load_pair(name)
        v1 = v[0].numpy()
        vl, pk, voiced = _regions(x)
        vlm, pkm = vl & voiced, pk & voiced
        row = {}
        for ver, sig in [("raw", v)] + [(m, None) for m in GAINS if m != "raw"]:
            if ver != "raw":
                r = run_gain(v1, ver, d)
                if not r.valid:
                    row[ver] = {"invalid": True, "g": r.g}
                    continue
                sig = torch.from_numpy(r.v_dn).unsqueeze(0)
                row[ver] = {"g": r.g}
            dly = _db(stft_batch(sig, _CFG))[0]
            dlx = _db(stft_batch(x, _CFG))[0]
            pe, ve, hcp, hcv = _cell(sig, x, vlm, pkm)
            row.setdefault(ver, {}).update(peak_err=pe, valley_err=ve, hc_peak=hcp, hc_valley=hcv,
                            lsd_lo=_lsd(sig, x, 100, 800),
                            lsd_hi=_lsd(sig, x, 800, 2000),
                            **dict(zip(("f0_conf", "voiced_cov"), _f0_stats(sig))))
        table[name] = row
        print(f"  {name[:32]:<32} " + " | ".join(
            f"{m}: valley {table[name][m].get('valley_err', float('nan')):+.2f} "
            f"lsd_lo {table[name][m].get('lsd_lo', float('nan')):.2f}"
            for m in ["raw"] + [g for g in GAINS if g != "raw"
                                and not table[name][g].get("invalid", False)]))
    (OUT / "A_denoise_metrics.json").write_text(json.dumps(table, indent=1))
    print(f"  A metrics written: {OUT}/A_denoise_metrics.json ({len(files)} recordings)")


# ------------------------------------------------------- B: N1 scan per V ----
def test_N2_B_n1_scan():
    """B: L×p scan per V version (raw + valid V_dn), N1 untouched; I1 (p=0)
    asserted per V version; zero-info control at L=10, p=.5 (caller-side time
    permutation, non-identity, output measurably different)."""
    OUT.mkdir(parents=True, exist_ok=True)
    d = GtcrnDenoiser()
    files = [Path(p).name for p in list_0624()]
    S_cache, V_cache, regions = {}, {}, {}
    for name in files:
        x, v = _load_pair(name)
        V_cache[name] = {"raw": v}
        for mode in GAINS:
            if mode == "raw":
                continue
            r = run_gain(v[0].numpy(), mode, d)
            if r.valid:
                V_cache[name][mode] = torch.from_numpy(r.v_dn).unsqueeze(0)
        regions[name] = _regions(x)
        for L in LS:
            S_cache[(name, L)] = degrade(
                x, _CFG, DegradationConfig(d5_enable=True, d5_level_db=L, seed=0))
    versions = ["raw"] + [m for m in GAINS if m != "raw"]
    results = {}
    for version in versions:
        for L in LS:
            for p in PS:
                per_file = {}
                for name in files:
                    if version not in V_cache[name]:
                        continue
                    x = _load_pair(name)[0]
                    v_ver = V_cache[name][version]
                    s_d = S_cache[(name, L)]
                    y = _run_n1(s_d, v_ver, p)
                    vl, pk, voiced = regions[name]
                    vlm, pkm = vl & voiced, pk & voiced
                    dly = _db(stft_batch(y, _CFG))[0]
                    dlx = _db(stft_batch(x, _CFG))[0]
                    pe, ve, hcp, hcv = _cell(y, x, vlm, pkm)
                    per_file[name] = dict(
                        valley_err=ve, peak_err=pe, hc_peak=hcp, hc_valley=hcv,
                        lsd_lo=_lsd(y, x, 100, 800), lsd_hi=_lsd(y, x, 800, 2000),
                        n=int(voiced.sum()))
                    if p == 0.0:   # I1 must hold for EVERY V version
                        dmax = float((y - s_d).abs().max())
                        assert dmax <= 1e-4 + 1e-6, (f"I1 broken for V={version}: "
                                                     f"max|Y-S|={dmax:.2e}")
                results[f"{version}_L{L}_p{p}"] = per_file
        print(f"  scan done for V={version} ({len(LS)}x{len(PS)} cells, I1 p=0 ok)")
    # zero-info control at L=10, p=.5: caller-side TIME permutation of V_dn
    ctrl = {}
    for version in versions:
        non_id, diffs = [], []
        for name in files:
            if version not in V_cache[name]:
                continue
            v_ver = V_cache[name][version]
            g = torch.Generator().manual_seed(1234)
            v_sh = v_ver[:, torch.randperm(v_ver.shape[-1], generator=g)]
            non_id.append(float((v_sh - v_ver).abs().max()))
            x = _load_pair(name)[0]
            s_d = S_cache[(name, 10)]
            y_real = _run_n1(s_d, v_ver, 0.5)
            y_sh = _run_n1(s_d, v_sh, 0.5)
            diffs.append(float((y_real - y_sh).abs().max()))
        ctrl[version] = {"permute_max_diff": float(np.median(non_id)),
                         "output_diff_real_vs_shuffled": float(np.median(diffs))}
        print(f"  zero-info control V={version}: permute non-identity "
              f"(median {ctrl[version]['permute_max_diff']:.2e}), output differs "
              f"from real path (median {ctrl[version]['output_diff_real_vs_shuffled']:.2e})")
    (OUT / "B_scan_results.json").write_text(json.dumps(results, indent=1))
    (OUT / "B_zero_info_control.json").write_text(json.dumps(ctrl, indent=1))
    with open(OUT / "B_scan_summary.csv", "w") as f:
        f.write("version,L,p,valley_err_med,peak_err_med,lsd_lo_med,lsd_hi_med\n")
        for key, per in results.items():
            vname, rest = key.split("_L", 1)
            L, p = rest.split("_p")
            med = lambda k: float(np.median([r[k] for r in per.values()]))
            f.write(f"{vname},{L},{p},{med('valley_err'):.4f},"
                    f"{med('peak_err'):.4f},{med('lsd_lo'):.4f},{med('lsd_hi'):.4f}\n")
    print(f"  B written: {OUT}/B_scan_results.json, B_scan_summary.csv, "
          f"B_zero_info_control.json")
    _criteria(results)
    _heatmaps(results)


def _criteria(results):
    """Pre-fixed three-criterion verdict per candidate gain (NO post-hoc edits)."""
    print("  pre-fixed criteria (all three required for 'worth real listening'):")
    verdicts = {}
    for version in [m for m in GAINS if m != "raw"]:
        rows = []
        for L in [10, 15]:
            for p in PS[1:]:
                per = results.get(f"{version}_L{L}_p{p}")
                per_r = results[f"raw_L{L}_p{p}"]
                if not per:
                    continue
                ve = float(np.median([r["valley_err"] for r in per.values()]))
                ve_r = float(np.median([r["valley_err"] for r in per_r.values()]))
                pe = float(np.median([r["peak_err"] for r in per.values()]))
                pe_r = float(np.median([r["peak_err"] for r in per_r.values()]))
                ls = float(np.median([r["lsd_lo"] for r in per.values()]))
                ls_r = float(np.median([r["lsd_lo"] for r in per_r.values()]))
                rows.append(dict(L=L, p=p, valley_gain=ve_r - ve,
                                 peak_worse=pe - pe_r, lsd_worse=ls - ls_r,
                                 best_valley_abs=abs(ve)))
        c1 = any(r["valley_gain"] >= 0.30 for r in rows)      # p>0 by construction
        best = min(rows, key=lambda r: r["best_valley_abs"]) if rows else None
        c2 = best is not None and best["peak_worse"] <= 0.50
        c3 = best is not None and best["lsd_worse"] <= 0.50
        verdict = ("worth real listening" if (c1 and c2 and c3) else
                   "not repairing the premise / domain mismatch")
        verdicts[version] = dict(c1_best_cell=best, c1=c1, c2=c2, c3=c3,
                                 verdict=verdict)
        if best:
            print(f"  {version}: best cell L={best['L']} p={best['p']} "
                  f"valley_gain={best['valley_gain']:+.2f} peak_worse={best['peak_worse']:+.2f} "
                  f"lsd_worse={best['lsd_worse']:+.2f} => c1={c1} c2={c2} c3={c3} => {verdict}")
        else:
            print(f"  {version}: no valid cells => {verdict}")
    (OUT / "criteria_verdicts.json").write_text(json.dumps(verdicts, indent=1))


def _heatmaps(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    versions = ["raw"] + [m for m in GAINS if m != "raw"]
    for metric, title in (("valley_err", "valley-floor error |Y|-|X| (dB)"),
                          ("peak_err", "peak error |Y|-|X| (dB)")):
        fig, axes = plt.subplots(1, len(versions), figsize=(3.2 * len(versions), 3.4),
                                 sharey=True)
        for ax, ver in zip(axes, versions):
            grid = np.zeros((len(LS), len(PS)))
            for i, L in enumerate(LS):
                for j, p in enumerate(PS):
                    per = results[f"{ver}_L{L}_p{p}"]
                    grid[i, j] = float(np.median([r[metric] for r in per.values()]))
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(PS))); ax.set_xticklabels(PS, fontsize=7)
            ax.set_yticks(range(len(LS))); ax.set_yticklabels(LS, fontsize=7)
            ax.set_title(ver, fontsize=9)
            for i in range(len(LS)):
                for j in range(len(PS)):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="w", fontsize=6)
        fig.suptitle(title + " (median of recordings)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"heat_{metric}_by_version.png", dpi=110)
        plt.close(fig)
    print(f"  heatmaps: {OUT}/heat_valley_err_by_version.png, "
          f"heat_peak_err_by_version.png (peak/valley separate)")


def test_N2_C_real_pair():
    """C: the user's first real stage-2/VPU pair — SKIP unless a path pair is
    unambiguously recorded from the earlier MVP real-data task.  No guessing."""
    record = Path("reports/T13N2/C_real_pair_pointer.json")
    if not record.exists():
        print("  C SKIP: no unambiguous record of the user's real stage-2/VPU "
              "pair exists from the MVP task (its real-data run used synthetic "
              "smoke only); not guessing paths — awaiting the user.")
        (OUT / "C_SKIP.json").write_text(json.dumps(
            {"status": "SKIP", "reason": "no unambiguous path record"}, indent=1))
        return
    spec = json.loads(record.read_text())
    print(f"  C: would process {spec}")


if __name__ == "__main__":
    test_N2_model_provenance_and_deps()
    test_N2_fixed_gain_bypass_and_mutation()
    test_N2_A_denoise_metrics_0624()
    test_N2_B_n1_scan()
    test_N2_C_real_pair()
