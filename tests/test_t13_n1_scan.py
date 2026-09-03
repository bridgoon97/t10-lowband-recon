"""T13-N1 main experiment — L × p scan, three control arms, figures, samples.

Pre-fixed (BEFORE any observation):
  * metrics: valley-floor error (main), peak fidelity, HC split into peak/valley
    components, band LSD (100–800 / 800–2k);
  * bucketing: per recording (speaker × wearing position); NO pooled means;
  * interpretation: small L (dirty) ⇒ optimal p HIGH; large L (clean) ⇒ optimal
    p LOW/0.  If optimal p is uncorrelated with L, that is a RED FLAG reported
    as-is (no tuning to force the shape).
Controls: p≡1 upper arm; zero-info V (time-shuffled / constant-level noise,
BOTH applied at the CALLER on the time axis); p≡0 (= I1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from fusion import (FusionConfig, Fusion, degrade, DegradationConfig,
                    stft_batch)
from fusion.trust import TrustSource
from fusion.realdata import list_0624

OUT = Path("reports/T13N1")
SR = 16000
LS = [40, 30, 25, 20, 15, 10]
PS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEG_S, OFFSET_S = 8.0, 2.0


def _load_pair(name):
    y, sr = sf.read(f"{Path(list_0624()[0]).parent}/{name}", dtype="float32",
                    always_2d=True)
    i0 = int(OFFSET_S * SR)
    x = torch.from_numpy(y[i0:i0 + int(SEG_S * SR), 1].copy()).unsqueeze(0)
    v = torch.from_numpy(y[i0:i0 + int(SEG_S * SR), 3].copy()).unsqueeze(0)
    return x, v


def _regions(x, cfg, w=1):
    """Valley/peak (bin,frame) regions from X's oracle F0 grid (offline)."""
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


def _db(spec):
    return 20 * torch.log10(spec.abs().clamp_min(1e-8))


_CFG = FusionConfig().with_switches(decision_mode="n1")


def _metrics_cell(y, x, vl, pk):
    dly = _db(stft_batch(y, _CFG))[0]          # (Fb, N)
    dlx = _db(stft_batch(x, _CFG))[0]
    d_v = float((dly - dlx)[vl].mean())
    d_p = float((dly - dlx)[pk].mean())
    hc_y = float(dly[pk].median() - dly[vl].median())
    return d_v, d_p, hc_y


def _lsd(y, x, lo_hz, hi_hz, _mutation_batch_axis=False):
    """Band LSD.  The FREQUENCY axis is axis 1 of (B, F, N) — the pre-rework
    code sliced axis 0 (batch), which with B=1 and lo>=3 gave an EMPTY tensor
    and NaN (rework blocker 3).  ``_mutation_batch_axis=True`` reproduces the
    old wrong slice so the finite/non-empty guard provably catches it."""
    cfg = _CFG
    bz = cfg.sr / cfg.n_fft
    ly = _db(stft_batch(y, cfg)); lx = _db(stft_batch(x, cfg))
    lo = max(1, int(lo_hz / bz)); hi = min(cfg.fusion_hi_bin, int(hi_hz / bz))
    if _mutation_batch_axis:
        d = ly[lo:hi + 1] - lx[lo:hi + 1]              # WRONG: batch axis
    else:
        d = ly[:, lo:hi + 1, :] - lx[:, lo:hi + 1, :]  # frequency axis
    n = int(d.numel())
    val = float(d.pow(2).mean().sqrt()) if n else float("nan")
    if not np.isfinite(val) or n == 0:
        raise ValueError(f"LSD guard: empty ({n} samples) or non-finite ({val}) "
                         f"for band {lo_hz}-{hi_hz} Hz")
    return val


def _run_n1(s, v, p):
    f = Fusion(_CFG)
    f.set_trust(TrustSource(source="manual", const=p))
    with torch.no_grad():
        return f.process_batch(s, v)


def test_N1_scan_main():
    """L × p main scan (pre-fixed metrics/bucketing/interpretation)."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(p).name for p in list_0624()]
    results = {}
    region_cache = {}
    s_cache = {}
    for name in files:
        x, v = _load_pair(name)
        vl, pk, voiced = _regions(x, _CFG)
        region_cache[name] = (vl, pk, voiced)
        for L in LS:
            s_d = degrade(x, _CFG, DegradationConfig(d5_enable=True, d5_level_db=L, seed=0))
            s_cache[(name, L)] = s_d
    for L in LS:
        for p in PS:
            per_file = {}
            for name in files:
                x, v = _load_pair(name)
                vl, pk, voiced = region_cache[name]
                s_d = s_cache[(name, L)]
                y = _run_n1(s_d, v, p)
                d_v, d_p, hc_y = _metrics_cell(y, x, vl & voiced, pk & voiced)
                d_vs, d_ps, hc_s = _metrics_cell(s_d, x, vl & voiced, pk & voiced)
                per_file[name] = dict(
                    valley_err=d_v, peak_err=d_p, hc_y=hc_y, hc_s=hc_s,
                    hc_x=float(_db(stft_batch(x, _CFG))[0][pk & voiced].median()
                               - _db(stft_batch(x, _CFG))[0][vl & voiced].median()),
                    valley_err_s=d_vs, peak_err_s=d_ps,
                    lsd_lo=_lsd(y, x, 100, 800), lsd_lo_s=_lsd(s_d, x, 100, 800),
                    lsd_hi=_lsd(y, x, 800, 2000), lsd_hi_s=_lsd(s_d, x, 800, 2000),
                    n=int(voiced.sum()))
            results[f"L{L}_p{p}"] = per_file
            med_v = float(np.median([r["valley_err"] for r in per_file.values()]))
            med_s = float(np.median([r["valley_err_s"] for r in per_file.values()]))
            print(f"  L={L:>2} p={p:<4}: valley_err median {med_v:+.2f} dB "
                  f"(S: {med_s:+.2f})  n={len(per_file)} recordings")
    (OUT / "scan_results.json").write_text(json.dumps(results, indent=1))
    # pre-fixed interpretation: optimal p per L (valley err closest to 0)
    print("  optimal p per L (|valley_err| min; pre-fixed: small L ⇒ high p):")
    opts = {}
    for L in LS:
        errs = {p: float(np.median([abs(r["valley_err"])
                                    for r in results[f"L{L}_p{p}"].values()])) for p in PS}
        best = min(errs, key=errs.get)
        opts[L] = best
        print(f"    L={L:>2}: best p={best}  (|err| per p: "
              + " ".join(f"{p}:{errs[p]:.2f}" for p in PS) + ")")
    (OUT / "optimal_p.json").write_text(json.dumps({str(k): v for k, v in opts.items()}))
    _heatmaps(results)
    return results, opts, files, region_cache, s_cache


def _heatmaps(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("valley_err", "valley-floor error |Y|-|X| (dB)"),
            ("peak_err", "peak error |Y|-|X| (dB)"),
            ("hc_y", "HC of Y (dB)"),
            ("hc_s", "HC of S (dB)"),
            ("lsd_lo", "LSD 100-800 (dB)"),
            ("lsd_hi", "LSD 800-2k (dB)")]
    for key, title in keys:
        grid = np.zeros((len(LS), len(PS)))
        for i, L in enumerate(LS):
            for j, p in enumerate(PS):
                per = results[f"L{L}_p{p}"]
                grid[i, j] = float(np.median([r[key] for r in per.values()]))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(PS))); ax.set_xticklabels(PS)
        ax.set_yticks(range(len(LS))); ax.set_yticklabels(LS)
        ax.set_xlabel("trust p"); ax.set_ylabel("D5 valley depth L (dB)")
        ax.set_title(title + " (median across recordings)")
        for i in range(len(LS)):
            for j in range(len(PS)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=7)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(OUT / f"heat_lp_{key}.png", dpi=110)
        plt.close(fig)
    print(f"  heatmaps written to {OUT}/heat_lp_*.png")


def test_N1_controls():
    """Three control arms: p≡1 upper arm; zero-info V (caller-side time
    permutation / constant-level noise); p≡0 (= I1)."""
    files = [Path(p).name for p in list_0624()[:4]]
    print("  control: p≡0 (identity) — covered by I1 on real 0624 (PASS above)")
    for L in [10, 25]:
        for name in files:
            x, v = _load_pair(name)
            s_d = degrade(x, _CFG, DegradationConfig(d5_enable=True, d5_level_db=L, seed=0))
            vl, pk, voiced = _regions(x, _CFG)
            # (1) caller-side TIME permutation of V (not a switch — actual data op)
            g = torch.Generator().manual_seed(1234)
            v_sh = v[:, torch.randperm(v.shape[-1], generator=g)]
            y_sh = _run_n1(s_d, v_sh, 0.75)
            # (2) constant-level V-unrelated noise (fixed −30 dBFS)
            rng = np.random.default_rng(7)
            v_const = torch.from_numpy(
                rng.normal(0, 10 ** (-30 / 20), (1, v.shape[-1])).astype(np.float32))
            y_co = _run_n1(s_d, v_const, 0.75)
            y_real = _run_n1(s_d, v, 0.75)
            d_real = _metrics_cell(y_real, x, vl & voiced, pk & voiced)[0]
            d_sh = _metrics_cell(y_sh, x, vl & voiced, pk & voiced)[0]
            d_co = _metrics_cell(y_co, x, vl & voiced, pk & voiced)[0]
            print(f"  L={L} {name[:28]:<28}: valley_err real={d_real:+.2f} "
                  f"shuffled={d_sh:+.2f} const={d_co:+.2f} dB")
    print("  control: p≡1 upper arm — equals the p=1.0 scan column (samples below)")


def test_N1_figures_and_samples():
    """Typical-frame spectra, G a[t]/s[t] curves, w/Δ↓ maps, listening clips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    name = Path(list_0624()[0]).name
    x, v = _load_pair(name)
    L = 20
    s_d = degrade(x, _CFG, DegradationConfig(d5_enable=True, d5_level_db=L, seed=0))
    f = Fusion(_CFG)
    f.set_trust(TrustSource(source="manual", const=0.75))
    with torch.no_grad():
        y = f.process_batch(s_d, v)
    # typical frame: a loud voiced frame
    spec_s = stft_batch(s_d, _CFG)[0]; spec_x = stft_batch(x, _CFG)[0]
    spec_v = stft_batch(v, _CFG)[0]; spec_y = stft_batch(y, _CFG)[0]
    e = spec_s[:, :].abs().pow(2).sum(0)
    t0 = int(torch.nonzero((torch.arange(len(e)) > 100) & (e == e.max()))[0])
    bz = _CFG.sr / _CFG.n_fft
    freq = torch.arange(spec_s.shape[0]) * bz
    m = freq <= 2000
    fig, ax = plt.subplots(figsize=(7, 4))
    for sp_, lab in ((spec_x, "X (clean FF)"), (spec_s, "S (X+D5)"),
                     (spec_v, "V (raw VPU)"), (spec_y, "Y (fused)")):
        ax.plot(freq[m], _db(sp_)[:len(freq)][m].numpy(), label=lab, lw=1)
    ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.legend(); ax.set_title(
        f"typical voiced frame (t={t0}*10ms, {name[:24]}, L={L}, p=0.75)")
    fig.tight_layout(); fig.savefig(OUT / "spec_frame.png", dpi=110); plt.close(fig)
    # G curves: rerun frame loop capturing a/s
    from fusion.fusion import FusionCore
    core = FusionCore(_CFG)
    lp = _CFG.win - _CFG.hop
    spx = torch.nn.functional.pad(s_d, (lp, 0), mode="constant")
    vpx = torch.nn.functional.pad(v, (lp, 0), mode="constant")
    fr_s = spx.unsqueeze(1).unfold(-1, _CFG.win, _CFG.hop).squeeze(1)
    fr_v = vpx.unsqueeze(1).unfold(-1, _CFG.win, _CFG.hop).squeeze(1)
    spec_s3 = stft_batch(s_d, _CFG); spec_v3 = stft_batch(v, _CFG)
    a_h, s_h = [], []
    with torch.no_grad():
        for t in range(spec_s3.shape[-1]):
            _, a, s = core.shape.step(spec_s3[:, :, t], spec_v3[:, :, t])
            a_h.append(float(a)); s_h.append(float(s))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(a_h, label="a[t] (fast, τ=80ms)")
    ax.plot(s_h, label="s[t] (slow, τ=2s)")
    ax.set_xlabel("frame (10 ms)"); ax.legend()
    ax.set_title("shaping-gain states G = a + s·f̃")
    fig.tight_layout(); fig.savefig(OUT / "shape_curves.png", dpi=110); plt.close(fig)
    # w(f) and Δ↓(f)
    wb = core.wband_curve
    dd = (core.cfg.n1_delta_down_min_db + wb * (core.cfg.n1_delta_down_max_db
                                                - core.cfg.n1_delta_down_min_db))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(freq[:len(wb)], wb, label="w_band(f) at p=1")
    ax.plot(freq[:len(dd)], dd, label="Δ↓(f) at g_v=1")
    ax.set_xlabel("Hz"); ax.legend(); ax.set_title("fixed weight curve and down-clip")
    fig.tight_layout(); fig.savefig(OUT / "w_dd_curves.png", dpi=110); plt.close(fig)
    # listening clips: 3 conditions × 3 clips × 4 tracks
    samp = OUT / "samples"; samp.mkdir(exist_ok=True)
    conds = [("L25_p075", 25, 0.75), ("L10_p075", 10, 0.75), ("L25_p100", 25, 1.0)]
    for cname, Lv, pv in conds:
        Sv = degrade(x, _CFG, DegradationConfig(d5_enable=True, d5_level_db=Lv, seed=0))
        yv = _run_n1(Sv, v, pv)
        for ci, t0 in enumerate([int(1.0 * SR), int(3.0 * SR), int(5.0 * SR)]):
            sl = slice(t0, t0 + int(2.0 * SR))
            for lab, sig in (("S", Sv), ("V", v), ("Y", yv), ("X", x)):
                sf.write(samp / f"{cname}_c{ci}_{lab}.wav",
                         sig[0][sl].numpy(), SR, subtype="PCM_16")
    print(f"  figures + {len(list(samp.glob('*.wav')))} listening clips in {OUT}/")


if __name__ == "__main__":
    test_N1_scan_main()
    test_N1_controls()
    test_N1_figures_and_samples()


def test_N1_lsd_mutation_guard():
    """Formal mutation for the LSD guard: the correct frequency-axis slice is
    finite on non-empty deterministic input; the old BATCH-axis slice (the
    rework blocker) must be CAUGHT by the finite/non-empty guard."""
    g = np.random.default_rng(3)
    x = torch.from_numpy(g.normal(0, 0.05, (1, SR * 2)).astype(np.float32))
    v = torch.from_numpy(g.normal(0, 0.05, (1, SR * 2)).astype(np.float32))
    val = _lsd(x, v, 100, 800)
    assert np.isfinite(val), "correct _lsd returned non-finite"
    try:
        _lsd(x, v, 100, 800, _mutation_batch_axis=True)
        caught = False
    except ValueError:
        caught = True
    assert caught, "mutation sanity FAILED: batch-axis slice was not caught by the guard"
    print(f"  LSD mutation guard PASS: correct slice finite ({val:.3f} dB); "
          f"batch-axis mutation caught (ValueError from finite/non-empty guard)")
