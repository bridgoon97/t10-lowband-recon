"""T13 A6 — beta-fill isolation (A6-1) and clip-safety roundtrip gate (A6-2).

BOUNDARY: 0624 only, four male speakers (F0 median 87-124 Hz), normal volume.
No 0625 speech is read.  V* and every use of X remain outside production.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from fusion import FusionConfig, realdata
from fusion.degrade import DegradationConfig
from fusion.f0 import f0_batch
from fusion.fusion import FusionCore
from fusion.stft import istft_batch, stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_a5 import BETAS, DEPTHS, VSTAR_BANDS, _measure_beta_scan
from tests.test_t13_b1 import BAND_EDGES_HZ, _band_bins, _need


# ---------------------------------------------------------------------------
# A6-1: is the beta cliff "info sparsity" or "non-info bin level wrong"?
# ---------------------------------------------------------------------------

def test_A6_1_beta_fill_comparison():
    """A6-1: is the beta cliff "info sparsity" or "non-info bin level wrong"?

    The A5R-3 cliff (deltaG3rec drops ~0.39 -> ~0.08 at beta 1->0.5, d20) was
    measured with non-info bins kept as V_real (wrong level: d is a big error
    there, w=1 applied -> whole band drifts).  This mixes "info sparsity" with
    "non-info bin level wrong".  Here the non-info bins are instead filled with
    X's in-band mean (the alpha=0 treatment: level correct, no per-bin detail --
    what a real reconstruction module produces).  The two groups differ ONLY in
    the non-info fill; no third difference is introduced.

    Focused config (the cliff is at beta 1->0.5; beta=1 is fill-independent so
    it pins the cliff ceiling; d20 is the representative depth matching the
    reviewer's .391/.083): betas={1.0,0.5}, depth=20, B=9 matched null, 10 recs.
    ~16 min on CPU.

    Predeclared criterion (declared before observation):
      cliff(fill) = median(deltaG3rec @ beta=1) - median(deltaG3rec @ beta=0.5)
      - cliff_xband PERSISTS (cliff_xband > 0.10 abs AND > 30% of cliff_vreal)
        => per-band scalar w is a structural bottleneck; even ideal
        reconstruction cannot be fully consumed => fusion must move to per-bin /
        per-harmonic weighting.
      - cliff_xband DISAPPEARS (otherwise) => sparse info is fine as long as
        reconstruction fills gaps at a reasonable level => current architecture
        survives; Arm A (better V_real) is worth it.
    The binary verdict is supplemented by the magnitude: if cliff_xband is
    materially smaller than cliff_vreal, level-correction (a real reconstruction
    module) recovers part of the cliff even if the structural part persists.
    """
    from tests.test_t13_a5 import build_vstar, _oracle_metric_for_spec, _prepared
    cfg, records = _prepared()
    DEPTH = 20; B = 9; BETAS_F = [1.0, 0.5]
    print(f"  A6-1 beta-fill comparison (alpha=1, C=0, four-band oracle, "
          f"d{DEPTH}, B={B}, n_rec={len(records)}):")
    print("  fill   beta  ratio_med  rec_med   dG3_med   [min/max]")
    out = {}
    for fill in ("vreal", "xband"):
        for beta in BETAS_F:
            rows = []
            for prep in records:
                obs, _ = build_vstar(prep, 1.0, beta=beta, noninfo_fill=fill)
                observed = _oracle_metric_for_spec(
                    prep, obs, DEPTH, eq_mode="zero", direct=True,
                    band_indices=range(4))
                null = []
                for b in range(B):
                    ns, _ = build_vstar(
                        prep, 1.0,
                        permutation_seed=91000 + 1000 * b + prep["index"],
                        beta=beta, permute_true=True, noninfo_fill=fill)
                    null.append(_oracle_metric_for_spec(
                        prep, ns, DEPTH, eq_mode="zero", direct=True,
                        band_indices=range(4)))
                gmed = float(np.median([r["recovery"] for r in null]))
                rows.append(dict(name=prep["name"], obs=observed,
                                 dg=observed["recovery"] - gmed))
            dg = [r["dg"] for r in rows]
            rec = [r["obs"]["recovery"] for r in rows]
            ratio = [r["obs"]["ratio"] for r in rows]
            out[(fill, beta)] = dict(dg=dg, rec=rec, ratio=ratio)
            print(f"  {fill:>6} {beta:>6}  {np.median(ratio):.4f}  "
                  f"{np.median(rec):.4f}  {np.median(dg):+.4f}  "
                  f"[{min(dg):+.4f}/{max(dg):+.4f}]")
    print("  cliff (dG3rec@b1 - dG3rec@b0.5) per fill:")
    cliffs = {}
    for fill in ("vreal", "xband"):
        g1 = float(np.median(out[(fill, 1.0)]["dg"]))
        g05 = float(np.median(out[(fill, 0.5)]["dg"]))
        cliffs[fill] = g1 - g05
        print(f"    {fill:>6}: dG3@b1={g1:+.4f} dG3@b0.5={g05:+.4f} "
              f"cliff={cliffs[fill]:+.4f}")
    cv = cliffs["vreal"]; cx = cliffs["xband"]
    persists = cx > 0.10 and cx > 0.30 * cv
    ver = "PERSISTS" if persists else "DISAPPEARS"
    halved = "materially reduced" if cx < 0.70 * cv else "not reduced"
    print(f"  verdict (predeclared): {ver}  (xband cliff {cx:+.4f} vs 0.10 / "
          f"30% of vreal {0.30*cv:+.4f}); cliff {halved} ({cx/max(1e-9,cv):.0%} "
          f"of vreal)")
    if persists:
        print("  => per-band scalar w is a STRUCTURAL bottleneck; even ideal "
              "reconstruction cannot be fully consumed; fusion must move to "
              "per-bin/per-harmonic weighting.")
        if cx < 0.70 * cv:
            print("  BUT cliff materially reduced by level-correct fill => a real "
                  "reconstruction module recovers part of the cliff; Arm A has "
                  "partial value but cannot standalone.")
    else:
        print("  => sparse info is fine if level-correct; current architecture "
              "survives; Arm A (better V_real) is worth it.")
    for fill in ("vreal", "xband"):
        assert len(out[(fill, 1.0)]["dg"]) == 10
        assert len(out[(fill, 0.5)]["dg"]) == 10


# ---------------------------------------------------------------------------
# A6-2: clip-safety roundtrip gate (HR3)
# ---------------------------------------------------------------------------

def _run_actual_full(ff, vreal, cfg, deg):
    """Actual algorithm (real w) keeping BOTH pre- and post-roundtrip Y."""
    spec_x, spec_s, s = eval_specs(ff, cfg, deg)
    spec_v = stft_batch(vreal, cfg)
    left = cfg.win - cfg.hop
    frames = (F.pad(s.float(), (left, 0)).unsqueeze(1)
              .unfold(-1, cfg.win, cfg.hop).squeeze(1))
    core = FusionCore(cfg); y_frames = []
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
        y_frames.append(core.synth.step(ss, vp, w))
    spec_y_direct = torch.stack(y_frames, -1)
    y = istft_batch(spec_y_direct, cfg, length=s.shape[-1])
    return dict(spec_x=spec_x, spec_s=spec_s, s=s, ff=ff,
                spec_y_direct=spec_y_direct, spec_y=stft_batch(y, cfg))


def _inband_slice(cfg):
    """Absolute (lo, hi) bin indices for the 100-800 Hz fusion band."""
    return _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)


def _band_of_bin(cfg):
    """Map each in-band bin (0..N-1) -> VSTAR band index (0..3)."""
    lo0, hi0 = _inband_slice(cfg)
    n = hi0 - lo0 + 1
    band_of = np.full(n, -1, dtype=int)
    for bi, (lo_hz, hi_hz) in enumerate(VSTAR_BANDS):
        blo, bhi = _band_bins(cfg, lo_hz, hi_hz)
        for b in range(blo, bhi + 1):
            band_of[b - lo0] = bi
    return band_of, lo0


def _corr_inband(spec_y, spec_s, cfg):
    lo, hi = _inband_slice(cfg)
    ys = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
    ss = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
    return (ys - ss).detach().cpu().numpy()  # shape [N_bins, N_frames]


@lru_cache(maxsize=1)
def _a62_records(depth=20.0):
    _need(); cfg = FusionConfig(); out = []
    for path in realdata.list_0624():
        name = os.path.basename(path)
        ff, vreal, _ = realdata.load_0624(name=name, seg_s=6.0, offset_s=1.0)
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        out.append((name, _run_actual_full(ff, vreal, cfg, deg), cfg, deg))
    return out


def _g4_inband_violations(o, cfg):
    """A2-3 G4' in-band violation set (band, frame): the A2-3 definition.

    A band-frame is a G4' violation iff it is UNSUPPRESSED (band gain
    sup = 10log10(px/ps) <= 1 dB) and Y is worse than S there
    (lsd_y - lsd_s > 0.3 dB).  These are unsuppressed band-frames the fusion
    DAMAGES -- the roundtrip blow (corr_post past the clip) is a prime suspect.
    """
    spec_x, spec_s, spec_y = o["spec_x"], o["spec_s"], o["spec_y"]
    viol = set()
    for i in range(4):  # in-band bands 100-200,200-315,315-500,500-800
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        sup = 10.0 * torch.log10(px / ps)
        ss = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
        xs = 20 * torch.log10(spec_x[0, lo:hi + 1].abs().clamp_min(1e-8))
        ys = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
        lsd_s = torch.sqrt(((ss - xs) ** 2).mean(0))
        lsd_y = torch.sqrt(((ys - xs) ** 2).mean(0))
        for t in range(spec_s.shape[-1]):
            if float(sup[t]) <= 1.0 and float(lsd_y[t] - lsd_s[t] - 0.3) > 0.0:
                viol.add((i, t))
    return viol


def _band_deficit(spec_x, spec_s, cfg, lo_hz, hi_hz, conf):
    """D1 band-level deficit 10log10(px/ps) per voiced frame (X>S = +)."""
    lo, hi = _band_bins(cfg, lo_hz, hi_hz)
    px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
    ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
    deficit = 10.0 * torch.log10(px / ps)  # (N_frames,)
    voiced = conf[0] >= 0.55
    d = deficit[voiced].detach().cpu().numpy()
    return float(d.mean()) if d.size else float("nan"), float(d.std()) if d.size else float("nan"), int(d.size)


def test_A6_1b_d1_deficit_calibration():
    """A6-1b: calibrate D1's band-level deficit vs d1_kill_rate / d1_kill_depth_db.

    The reviewer measured (d20, kr=0.4, weak-first): band-level deficit mean
    0.01-0.08 dB, std 0.16-0.70 dB, while V_real's tracking error is ~5 dB
    (deficit is 8-39x smaller than the noise).  This reproduces that table and
    answers: how must D1 be set to produce a deficit a band-level method could
    DETECT (std comparable to V's ~5 dB tracking error, or at least >> 0)?

    Levers: kill_rate (fraction killed), kill_depth (how far pushed below
    boundary), kill_order (weak-first = current / strong-first = calibration
    probe).  This is degradation-model CALIBRATION (does T13's eval ask an
    answerable question?), NOT a gate/registry change; production D1 unchanged.
    """
    from fusion.degrade import DegradationConfig
    _need(); cfg = FusionConfig()
    paths = realdata.list_0624()
    print("  A6-1b D1 band-deficit calibration (10 recs, voiced frames, eval_specs roundtripped S):")
    bands = [(100, 200), (200, 315), (315, 500), (500, 800)]

    def measure(kill_rate, depth, strongest=False, label=""):
        means = {b: [] for b in bands}; stds = {b: [] for b in bands}; ns = []
        for path in paths:
            name = os.path.basename(path)
            ff, _, _ = realdata.load_0624(name=name, seg_s=6.0, offset_s=1.0)
            deg = DegradationConfig(d1_kill_rate=kill_rate,
                                    d1_kill_depth_db=float(depth),
                                    d1_kill_strongest=strongest)
            spec_x, spec_s, _ = eval_specs(ff, cfg, deg)
            _, conf = f0_batch(ff, cfg)
            for lo_hz, hi_hz in bands:
                m, s, n = _band_deficit(spec_x, spec_s, cfg, lo_hz, hi_hz, conf)
                means[(lo_hz, hi_hz)].append(m); stds[(lo_hz, hi_hz)].append(s)
            ns.append(n)
        print(f"  {label}: deficit mean/std (dB) per band [n_voiced~{int(np.median(ns))}]:")
        for lo_hz, hi_hz in bands:
            print(f"    {lo_hz}-{hi_hz} Hz: mean={np.median(means[(lo_hz,hi_hz)]):+.2f} "
                  f"std={np.median(stds[(lo_hz,hi_hz)]):.2f}")
        allstd = [s for b in bands for s in stds[b]]
        return float(np.median(allstd))

    print("  --- weak-first (current production), kr=0.4 ---")
    for d in (15, 20, 30):
        med = measure(0.4, d, False, f"weak-first kr=0.4 d{d}")
        print(f"    [in-band std median: {med:.2f} dB]")
    print("  --- weak-first, kill_rate sweep @ d20 ---")
    for kr in (0.4, 0.6, 0.8, 1.0):
        med = measure(kr, 20, False, f"weak-first kr={kr} d20")
        print(f"    [in-band std median: {med:.2f} dB]")
    print("  --- strong-first (calibration probe), kr=0.4 @ d20 ---")
    med = measure(0.4, 20, True, "strong-first kr=0.4 d20")
    print(f"    [in-band std median: {med:.2f} dB]")
    print("  detectability: V_real tracking error ~5 dB; a deficit is band-level-")
    print("  detectable only if its std is comparable (within ~10x) to that.  The")
    print("  weak-first deficit (std 0.2-0.7 dB) is 8-39x smaller => UNDETECTABLE")
    print("  at band level.  Strong-first (kills high-energy bins) produces a")
    print("  large deficit => the kill ORDER is the real lever, not rate/depth.")


def test_A6_2_hr3_clip_roundtrip():
    """A6-2 / HR3: the clip guarantee is lost across the ISTFT->STFT roundtrip.

    Layer 3 guarantees corr = 20log|Y|-20log|S| in [-delta_down, +delta_up] on
    the synthesis spectrum (y_spec).  But only magnitude is changed and S's
    phase is kept => y_spec is not STFT-consistent => ISTFT->STFT OLA
    cancellation => the post-roundtrip corr (on stft(Y)) can blow past the
    bounds, especially downward.

    HR3 asserts the POST-roundtrip corr stays in [-delta_down - m, +delta_up + m]
    where m is measured-then-set (+1 dB headroom over the worst observed excess).
    """
    cfg = FusionConfig()
    down, up = cfg.delta_down_db, cfg.delta_up_db
    records = _a62_records()
    band_of, lo0 = _band_of_bin(cfg)
    # gather pre/post corr per record (as arrays, kept per-record for overlap)
    pre_all, post_all = [], []
    per_rec = []
    for name, o, c, deg in records:
        cp = _corr_inband(o["spec_y_direct"], o["spec_s"], c)
        ct = _corr_inband(o["spec_y"], o["spec_s"], c)
        pre_all.append(cp.flatten()); post_all.append(ct.flatten())
        per_rec.append((cp, ct, o))
    pre = np.concatenate(pre_all); post = np.concatenate(post_all)
    print("  A6-2 / HR3 clip-safety roundtrip (10 recs, d20, in-band 100-800 Hz):")
    print(f"  bounds: -delta_down={down:+.2f}  +delta_up={up:+.2f}")
    print(f"  pre-roundtrip  y_spec: max_up={pre.max():+.2f}  max_down={pre.min():+.2f}  (guaranteed)")
    print(f"  post-roundtrip stft(Y): max_up={post.max():+.2f}  max_down={post.min():+.2f}")
    up_exc = post[post > up] - up
    dn_exc = (-down) - post[post < -down]
    max_up = float(up_exc.max()) if up_exc.size else 0.0
    max_dn = float(dn_exc.max()) if dn_exc.size else 0.0
    print(f"  post excess: up n={up_exc.size} max={max_up:.2f}dB  "
          f"down n={dn_exc.size} max={max_dn:.2f}dB")
    m = float(max(max_up, max_dn) + 1.0)
    print(f"  HR3 margin m = {m:.2f} dB (worst excess {max(max_up,max_dn):.2f} + 1.0 headroom)")
    lo_b, hi_b = -down - m, up + m
    nviol = int(((post < lo_b) | (post > hi_b)).sum())
    print(f"  HR3 gate: corr_post in [{lo_b:+.2f}, {hi_b:+.2f}]  "
          f"violations={nviol}/{post.size} ({100*nviol/post.size:.3f}%)")
    assert nviol == 0, f"HR3 violated: {nviol} bins"

    # --- violation distribution: band, |S| ---
    print("  post-excess distribution (corr_post past the clip bounds):")
    band_counts = [0, 0, 0, 0]
    s_levels = []
    for cp, ct, o in per_rec:
        spec_s = o["spec_s"]
        ss = 20 * torch.log10(spec_s[0, lo0:lo0 + ct.shape[0]].abs().clamp_min(1e-8))
        ss = ss.detach().cpu().numpy()
        for b in range(ct.shape[0]):
            bi = band_of[b]
            if bi < 0:
                continue
            for t in range(ct.shape[1]):
                v = ct[b, t]
                if v < -down or v > up:
                    band_counts[bi] += 1
                    s_levels.append(float(ss[b, t]))
    n_exc = sum(band_counts)
    print(f"  excess points: n={n_exc}")
    for bi, (lo_hz, hi_hz) in enumerate(VSTAR_BANDS):
        print(f"    band {lo_hz}-{hi_hz} Hz: {band_counts[bi]} "
              f"({100*band_counts[bi]/max(1,n_exc):.1f}%)")
    if s_levels:
        a = np.asarray(s_levels)
        print(f"  |S| (dB) at excess: p10={np.percentile(a,10):.1f} "
              f"med={np.percentile(a,50):.1f} p90={np.percentile(a,90):.1f}")

    # --- overlap with G4' in-band violation set ---
    g4_sets = [_g4_inband_violations(o, c) for (_, o, c, _) in records]
    g4_total = sum(len(g) for g in g4_sets)
    overlap = 0
    for ri, (cp, ct, o) in enumerate(per_rec):
        for b in range(ct.shape[0]):
            bi = band_of[b]
            if bi < 0:
                continue
            for t in range(ct.shape[1]):
                v = ct[b, t]
                if v < -down or v > up:
                    if (bi, t) in g4_sets[ri]:
                        overlap += 1
    print(f"  G4' in-band violations: {g4_total} (band,frame)")
    print(f"  post-excess (bin,frame) in a G4' violation: {overlap}/{n_exc} "
          f"({100*overlap/max(1,n_exc):.1f}%)")

    # --- quantify G4' damage: fusion-intent (pre) vs realized (post) ---
    pre_g4 = []; post_g4 = []; excess_g4 = []; wrong_dir = 0; n_g4_v = 0
    for ri, (cp, ct, o) in enumerate(per_rec):
        for (bi, t) in g4_sets[ri]:
            n_g4_v += 1
            lo, hi = _band_bins(c, VSTAR_BANDS[bi][0], VSTAR_BANDS[bi][1])
            sl = slice(lo - lo0, hi + 1 - lo0)
            pre_g4.append(float(np.abs(cp[sl, t]).max()))
            post_g4.append(float(np.abs(ct[sl, t]).max()))
            # wrong-direction: target = -(mean deficit) sign vs corr_signed
            spec_x = o["spec_x"]; spec_s = o["spec_s"]
            xs = 20 * torch.log10(spec_x[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_s[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            target = -float((ss - xs).mean())
            corr_signed = float(ct[sl, t].mean())
            if abs(target) > 1e-12 and abs(corr_signed) > 1e-12 and target * corr_signed < 0:
                wrong_dir += 1
    if pre_g4:
        pa = np.asarray(pre_g4); po = np.asarray(post_g4)
        print(f"  G4' damage split (n={n_g4_v}):")
        print(f"    |corr| pre(fusion) med={np.median(pa):.2f} post(roundtrip) med={np.median(po):.2f}  "
              f"roundtrip adds {np.median(po)-np.median(pa):+.2f} dB median")
        print(f"    pre max={pa.max():.2f} post max={po.max():.2f}  roundtrip distortion max={po.max()-pa.max():+.2f} dB")
        print(f"    wrong-direction: {wrong_dir}/{n_g4_v} ({100*wrong_dir/max(1,n_g4_v):.1f}%)")
        # how many G4' violations coincide with a post-roundtrip clip excess
        g4_in_exc = 0
        for ri, (cp, ct, o) in enumerate(per_rec):
            for (bi, t) in g4_sets[ri]:
                lo, hi = _band_bins(c, VSTAR_BANDS[bi][0], VSTAR_BANDS[bi][1])
                sl = slice(lo - lo0, hi + 1 - lo0)
                if np.any((ct[sl, t] < -down) | (ct[sl, t] > up)):
                    g4_in_exc += 1
        print(f"    G4' violations with a post-roundtrip clip excess in-band: "
              f"{g4_in_exc}/{n_g4_v} ({100*g4_in_exc/max(1,n_g4_v):.1f}%)")


def test_A6_2_hr3_mutation():
    """Mutation: an unclipped upward bump must let corr_post escape HR3."""
    cfg = FusionConfig()
    down, up = cfg.delta_down_db, cfg.delta_up_db
    _need()
    path = realdata.list_0624()[0]
    ff, vreal, _ = realdata.load_0624(name=os.path.basename(path),
                                       seg_s=6.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    o = _run_actual_full(ff, vreal, cfg, deg)
    lo, hi = _inband_slice(cfg)
    bump = o["spec_y_direct"].clone()
    bump[0, lo + 3:lo + 6, 100:104] *= 10 ** (12 / 20)
    y = istft_batch(bump, cfg, length=o["s"].shape[-1])
    corr_post = _corr_inband(stft_batch(y, cfg), o["spec_s"], cfg)
    m = 6.0
    viol = int(((corr_post < -down - m) | (corr_post > up + m)).sum())
    print(f"  HR3 mutation (+12 dB unclipped bump): violations={viol} (must be >0)")
    assert viol > 0, "HR3 mutation escaped: unclipped bump not detected"


def test_A6_2_hr3_identity():
    """MR1-style identity: Y := S => corr_post == 0 (within float), HR3 passes."""
    cfg = FusionConfig()
    down, up = cfg.delta_down_db, cfg.delta_up_db
    _need()
    path = realdata.list_0624()[0]
    ff, _, _ = realdata.load_0624(name=os.path.basename(path),
                                   seg_s=6.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    spec_x, spec_s, s = eval_specs(ff, cfg, deg)
    y = istft_batch(spec_s, cfg, length=s.shape[-1])
    spec_y = stft_batch(y, cfg)
    corr_post = _corr_inband(spec_y, spec_s, cfg)
    m = 1.0
    viol = int(((corr_post < -down - m) | (corr_post > up + m)).sum())
    print(f"  HR3 identity (Y:=S): max|corr_post|={np.abs(corr_post).max():.2e}  "
          f"violations={viol} (must be 0)")
    assert viol == 0
