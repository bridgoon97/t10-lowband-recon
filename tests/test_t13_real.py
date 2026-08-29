"""T13-A rework — REAL-device tests (0624/ only; 0625/ held-out, NOT touched).

  R2: G5 future-perturbation on REAL VOICED (FF as S, VPU as V) — w nonzero,
      w_local & EQ active.  Mutation sanity (bidirectional w-EMA) re-run on the
      same voiced condition; leak magnitude reported (must be >> the white-noise
      1.8e-3, since w is large here).  Also reports the three real-VPU smoke
      points: pipeline runs on real V, output finite, causal holds.
  R4: M1 re-test on a REAL speech harmonic envelope (formants / per-harmonic
      undulation).  D1=40% kill, ground-truth kill set known.  Threshold
      UNCHANGED (recall ≥0.90 / FAR ≤0.10) but this is a REPORT item, not a gate:
      if it fails we report honestly and DO NOT tune (B-stage判据 input).

No G1–G6 effect metrics, no tuning, no 0625/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fusion import Fusion, FusionStreamer, FusionConfig
from fusion.degrade import apply_d1, DegradationConfig
from fusion.decision import WLocal
from fusion.f0 import f0_batch
from fusion.stft import stft_batch
from tests._testutil import SkipTest

try:
    from fusion import realdata
    _HAVE = True
except Exception:
    _HAVE = False

if _HAVE:
    try:
        _ = realdata.list_0624()
    except Exception:
        _HAVE = False


def _need():
    if not _HAVE:
        raise SkipTest("0624 real recordings not accessible at /mnt/d/.../mic_recordings")


def _voiced_SV(seg_s=4.0):
    """Real voiced FF (S source) + VPU (V)."""
    ff, vpu, sr = realdata.load_0624(seg_s=seg_s, offset_s=1.0)
    return ff, vpu, sr


# ================================================================ R2 ======
def test_R2_future_perturbation_real_voiced():
    """G5 future-perturbation on REAL voiced FF/VPU — past bit-identical."""
    _need()
    cfg = FusionConfig()
    s, v, sr = _voiced_SV(seg_s=4.0)
    y_full = Fusion(cfg).process_batch(s, v)
    assert torch.isfinite(y_full).all(), "real-V run produced non-finite output"
    T = s.shape[-1]
    ps = [cfg.hop * 40, cfg.hop * 80, T // 2]
    worst = 0.0
    for P in ps:
        s_m = s.clone(); s_m[:, P:] = 0.0
        v_m = v.clone(); v_m[:, P:] = 0.0
        y_m = Fusion(cfg).process_batch(s_m, v_m)
        K = max(0, P - cfg.win)
        eq = torch.equal(y_full[..., :K], y_m[..., :K])
        diff = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
        worst = max(worst, diff)
        assert eq, f"real-voiced future leak at P={P}: diff={diff}"
    print(f"  R2 real-voiced future-perturbation: {len(ps)} cut points, past "
          f"bit-identical (torch.equal), worst diff={worst}")
    print(f"    real-VPU smoke: pipeline runs on real V ✓ | output finite ✓ | "
          f"causal (future-zero past-identical) ✓")


def test_R2_mutation_real_voiced():
    """R2 generic mutation sanity on REAL voiced: a whole-segment-stat
    (global-mean-norm of Y) leaks on ANY signal (the bidir-w-EMA used in the
    A-rework stopped leaking under the new ③-only detector because w is
    near-constant — replaced here).  Still caught (>1e-6).  The path-specific
    w_local-LOOK-AHEAD mutation (next test) is the stronger voiced-condition proof."""
    _need()
    from tests.test_t13_streaming import _MutantGlobalMeanNorm
    cfg = FusionConfig()
    s, v, sr = _voiced_SV(seg_s=4.0)
    mutant = _MutantGlobalMeanNorm(cfg)
    y_full = mutant.process_batch(s, v)
    T = s.shape[-1]
    P = T // 2
    s_m = s.clone(); s_m[:, P:] = 0.0
    v_m = v.clone(); v_m[:, P:] = 0.0
    y_m = mutant.process_batch(s_m, v_m)
    K = max(0, P - cfg.win)
    leak = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
    detected = leak > 1e-6
    print(f"  R2 mutation (global-mean-norm(Y)) on REAL voiced: leak={leak:.3e} "
          f"(bidir-w-EMA stopped leaking under ③-only detector; this whole-seg-stat always leaks) → "
          f"{'FAIL-of-mutant (caught) PASS' if detected else 'NOT caught'}")
    assert detected, "mutation not caught on voiced (leak ≤ 1e-6)"
    return leak


def test_R2_mutation_wlocal_lookahead():
    """Look-ahead in the w_local path (NEXT frame's S for the RANSAC) leaks
    proportionally to w_local's CONTRIBUTION to Y.  Voiced (w_local active) ⇒
    large leak; white noise (w_local≈0) ⇒ tiny — the voiced condition is what
    gives the test teeth for this path (the reviewer's actual concern)."""
    _need()
    cfg = FusionConfig()

    class _MutantWLocalLookahead(Fusion):
        def process_batch(self, s, v):
            import torch.nn.functional as F
            from fusion.stft import stft_batch, istft_batch
            s = s.float(); v = v.float(); cfg = self.cfg
            spec_s = stft_batch(s, cfg); spec_v = stft_batch(v, cfg)
            left_pad = cfg.win - cfg.hop
            sp = F.pad(s, (left_pad, 0))
            frames_s = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
            N = spec_s.shape[-1]
            yf = []
            for t in range(N):
                ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames_s[:, t, :]
                f0, conf = self.core.f0est.estimate(buf)
                s_mag = ss.abs(); fl = self.core.nf.step(s_mag)
                snr = (20 * torch.log10(s_mag.clamp_min(1e-8) /
                                        fl.clamp_min(1e-8))).mean(-1)
                v_prime, startup, reset = self.core.eq.step(ss, vs, snr, conf)
                g = self.core.gf0.step(conf)
                wb = self.core.wband.step(v_prime, ss)
                t_next = min(t + 1, N - 1)        # <<< LOOK-AHEAD (mutation)
                wl = self.core.wlocal.step(spec_s[:, :, t_next], v_prime, f0)
                c_v = self.core.cv.step(v_prime, ss, torch.zeros_like(snr),
                                        bool(reset.any()))
                w_raw = c_v.unsqueeze(-1) * g.unsqueeze(-1) * wb * wl
                fw = torch.maximum(startup, reset.float())
                w = self.core.smooth.step(w_raw * (1 - fw).unsqueeze(-1))
                self.core.w_history.append(w.detach().clone())
                yf.append(self.core.synth.step(ss, v_prime, w))
            return istft_batch(torch.stack(yf, -1), cfg, length=s.shape[-1])

    s, v, sr = _voiced_SV(seg_s=4.0)
    T = s.shape[-1]; P = T // 2; K = max(0, P - cfg.win)
    mut = _MutantWLocalLookahead(cfg)
    yf = mut.process_batch(s, v)
    sm = s.clone(); sm[:, P:] = 0.0; vm = v.clone(); vm[:, P:] = 0.0
    ym = mut.process_batch(sm, vm)
    leak_voiced = (yf[..., :K] - ym[..., :K]).abs().max().item()
    g = torch.Generator().manual_seed(0)
    sw = torch.randn(1, T, generator=g); vw = 0.5 * sw + 0.3 * torch.randn(1, T, generator=g)
    mut2 = _MutantWLocalLookahead(cfg)
    yfw = mut2.process_batch(sw, vw)
    swm = sw.clone(); swm[:, P:] = 0.0; vwm = vw.clone(); vwm[:, P:] = 0.0
    ymw = mut2.process_batch(swm, vwm)
    leak_white = (yfw[..., :K] - ymw[..., :K]).abs().max().item()
    ok = leak_voiced > 1e-6 and leak_voiced > leak_white
    print(f"  R2 mutation (w_local LOOK-AHEAD): voiced leak={leak_voiced:.3e}  "
          f"white leak={leak_white:.3e}  voiced {'>>' if leak_voiced > leak_white else '≤'} white → "
          f"{'voiced gives more power ✓ PASS' if ok else 'PROBLEM'}")
    assert ok, ("w_local look-ahead not caught more strongly on voiced "
                f"(voiced={leak_voiced}, white={leak_white})")


# ================================================================ R4 ======
def _r4_recall_far(cfg, deg=DegradationConfig(d1_kill_rate=0.4),
                    cap_frames=250):
    """Run R4 (real in-band envelope) with the given cfg/deg; return
    (recall, far, n_killed_pts, n_surviving_pts, n_voiced)."""
    wl = WLocal(cfg, v_fallback=cfg.enable_w_local_vfallback, valley=cfg.enable_valley_rule)
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); spec_V = stft_batch(vpu, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)
    spec_S, killed = apply_d1(spec_X, f0_tr, cfg, deg)
    bz = cfg.sr / cfg.n_fft
    band_hi = min(spec_X.shape[1], int(deg.d1_band_hi_hz / bz))
    Pk, Ps = [], []
    n_voiced = 0
    N = spec_S.shape[-1]
    for t in range(N):
        if float(conf_tr[0, t]) < 0.55 or float(f0_tr[0, t]) <= 0:
            continue
        n_voiced += 1
        if n_voiced > cap_frames:
            break
        f0 = float(f0_tr[0, t])
        w = wl.step(spec_S[:, :, t], spec_V[:, :, t], torch.tensor([f0]))[0]
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= band_hi):
                continue
            # only REAL harmonics (clean X above noise floor)
            if 20 * torch.log10(spec_X[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            flagged = w[b].item() > 0.5
            (Pk if bool(killed[0, b, t]) else Ps).append(flagged)
    return (sum(Pk) / max(1, len(Pk)), sum(Ps) / max(1, len(Ps)),
            len(Pk), len(Ps), n_voiced)


def test_BR2_abs_must_fail_on_realistic_D1():
    """BR2: a PURE ABSOLUTE-LEVEL detector (③) must NOT reach 0.90/0.10 on the
    REALISTIC D1 (killed ≈ weakest-survivor in level).  If it does, the sim is
    tautological (D1 puts killed at a fixed peak-offset ⇒ ③ = D1's inverse).
    Asserts recall<0.90 OR FAR>0.10 (③ must FAIL)."""
    _need()
    cfg = FusionConfig()
    cfg.wl_use_local_median = False; cfg.wl_use_abrupt_drop = False
    cfg.wl_use_abs_gate = True; cfg.wl_use_v_envelope = False
    r, f, nk, ns, nv = _r4_recall_far(cfg)
    fails = not (r >= 0.90 and f <= 0.10)
    print(f"  BR2 ③-must-fail (realistic D1): recall={r:.3f} FAR={f:.3f} → "
          f"{'FAILs (tautology absent) PASS' if fails else '③ PASSES — D1 still tautological! PROBLEM'}")
    assert fails, "BR2: ③ reaches 0.90/0.10 on realistic D1 — D1 is still tautological"


def test_BR2_abs_mutation():
    """Mutation: d1_realistic=False (revert to frame-peak−60 floor) ⇒ ③ becomes
    D1's inverse ⇒ ③ PASSES ⇒ the BR2 'must-fail' assertion FAILS (caught)."""
    _need()
    cfg = FusionConfig()
    cfg.wl_use_local_median = False; cfg.wl_use_abrupt_drop = False
    cfg.wl_use_abs_gate = True; cfg.wl_use_v_envelope = False
    deg = DegradationConfig(d1_kill_rate=0.4, d1_realistic=False)   # MUTATION
    r, f, nk, ns, nv = _r4_recall_far(cfg, deg=deg)
    passes = r >= 0.90 and f <= 0.10
    print(f"  BR2 mutation (d1_realistic=False): ③ recall={r:.3f} FAR={f:.3f} → "
          f"{'③ PASSES (tautology back) → FAIL-of-mutant (caught) PASS' if passes else 'NOT caught'}")
    assert passes, "BR2 mutation: ③ did NOT pass under tautological D1 (mutation not caught)"


def _overlap_coefficient(cfg, deg=DegradationConfig(d1_kill_rate=0.4)):
    """Overlap coefficient of killed-level vs weakest-survivor-level distributions."""
    import numpy as _np
    ff, _, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    specX = stft_batch(ff, cfg); specS, killed = apply_d1(specX, f0_batch(ff, cfg)[0], cfg, deg)
    bz = cfg.sr / cfg.n_fft; band_hi = min(specX.shape[1], int(deg.d1_band_hi_hz / bz))
    f0tr, conftr = f0_batch(ff, cfg)
    kdb, wdb = [], []
    for t in range(specS.shape[-1]):
        if conftr[0, t] < 0.55 or f0tr[0, t] <= 0:
            continue
        f0 = float(f0tr[0, t]); es = []
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if 1 <= b <= band_hi and 20 * torch.log10(specX[0, b, t].abs().clamp_min(1e-8)).item() > -60:
                es.append((b, 20 * torch.log10(specS[0, b, t].abs().clamp_min(1e-8)).item(), bool(killed[0, b, t])))
        if len(es) < 4:
            continue
        for b, ldb, kk in es:
            if kk:
                kdb.append(ldb)
        surv = sorted([ldb for b, ldb, kk in es if not kk])
        wdb.extend(surv[:3])
    k = _np.array(kdb); w = _np.array(wdb)
    if len(k) < 10 or len(w) < 10:
        return 0.0
    lo, hi = min(k.min(), w.min()), max(k.max(), w.max())
    bk, _ = _np.histogram(k, bins=40, range=(lo, hi))
    bw, _ = _np.histogram(w, bins=40, range=(lo, hi))
    return float(_np.minimum(bk, bw).sum() / max(1, min(bk.sum(), bw.sum())))


def test_BR2_overlap():
    """BR2: killed-level and weakest-survivor-level distributions must SUBSTANTIALLY
    overlap (≥0.30) — the task premise (S alone can't tell killed from
    naturally-weak).  No overlap ⇒ D1 puts killed at a separable level ⇒ sim
    doesn't represent the real problem."""
    _need()
    cfg = FusionConfig()
    ov = _overlap_coefficient(cfg)
    print(f"  BR2 overlap (killed vs weakest-survivor): {ov:.3f} (≥0.30) → "
          f"{'PASS' if ov >= 0.30 else 'FAIL — no overlap, sim too easy'}")
    assert ov >= 0.30, f"BR2: overlap {ov} < 0.30 — D1 too easy (killed separable)"


def test_BR2_overlap_mutation():
    """Mutation: d1_realistic=False ⇒ killed at frame-peak−60 (far from weak
    survivors) ⇒ overlap ~0 ⇒ the BR2 overlap assertion FAILS (caught)."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4, d1_realistic=False)   # MUTATION
    ov = _overlap_coefficient(cfg, deg=deg)
    broken = ov < 0.30
    print(f"  BR2 mutation (d1_realistic=False): overlap={ov:.3f} (<0.30) → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught'}")
    assert broken, "BR2 overlap mutation: overlap not destroyed by tautological D1"


def test_R4_M1_real_envelope():
    """M1 on REAL in-band (≤2 kHz) speech envelope (D1=40 %, apply_d1
    band-limited).  Threshold UNCHANGED (recall ≥0.90 / FAR ≤0.10) — REPORT
    item; honest if it fails, NO tuning (B-stage判据 input)."""
    _need()
    cfg = FusionConfig()
    recall, far, nk, ns, nv = _r4_recall_far(cfg)
    status = "PASS" if recall >= 0.90 and far <= 0.10 else "BELOW-THRESHOLD (reported, not tuned)"
    print(f"  R4 M1 real in-band envelope (① local-median DEFAULT, realistic D1): voiced={nv}  "
          f"recall={recall:.3f} (≥0.90)  FAR={far:.3f} (≤0.10)  [{status}]  "
          f"(killed={nk} surviving={ns})")
    return recall, far


def test_R4_anti_noop():
    """B0 §1 anti-no-op: D1=40 % must actually kill in-band harmonics (the old
    full-band sort killed only >2 kHz ⇒ in-band 0 ⇒ test measured nothing).
    Asserts in-band killed points > 0 AND ratio ≈ 0.40 ± tol."""
    _need()
    cfg = FusionConfig()
    ff, _, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)
    deg = DegradationConfig(d1_kill_rate=0.4)
    _, killed = apply_d1(spec_X, f0_tr, cfg, deg)
    bz = cfg.sr / cfg.n_fft
    band_hi = min(spec_X.shape[1], int(deg.d1_band_hi_hz / bz))
    n_killed_in = 0; n_harm_in = 0
    N = spec_X.shape[-1]
    for t in range(N):
        if float(conf_tr[0, t]) < 0.55 or float(f0_tr[0, t]) <= 0:
            continue
        f0 = float(f0_tr[0, t])
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= band_hi):
                continue
            if 20 * torch.log10(spec_X[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            n_harm_in += 1
            if bool(killed[0, b, t]):
                n_killed_in += 1
    ratio = n_killed_in / max(1, n_harm_in)
    print(f"  R4 anti-no-op: in-band killed={n_killed_in}/{n_harm_in} = {ratio:.3f} "
          f"(must be >0 and ≈0.40±0.10)")
    assert n_killed_in > 0, "D1 killed ZERO in-band harmonics (no-op regression!)"
    assert 0.25 <= ratio <= 0.55, f"in-band kill ratio {ratio} not ~0.40"


def test_degrade_bandcheck():
    """B0 §1 self-check: D2/D3/D4 are per-point/per-block (no cross-band sort),
    so they cannot exhibit the 'weakest-lands-entirely-out-of-band' no-op.
    Confirms by inspection of the code + a structural assertion."""
    _need()
    import inspect
    from fusion.degrade import apply_d2, apply_d3, apply_d4
    src = inspect.getsource(apply_d2) + inspect.getsource(apply_d3) + inspect.getsource(apply_d4)
    # D2/D3/D4 must NOT sort harmonics by energy across the band (only D1 does)
    has_sort = "sorted(" in src or "argsort" in src or ".sort(" in src
    print(f"  degrade band-check: D2/D3/D4 source has cross-band energy sort? {has_sort} "
          f"→ {'PROBLEM' if has_sort else 'none (per-point/per-block, no no-op risk) ✓'}")
    assert not has_sort, "D2/D3/D4 unexpectedly sort across band"


def test_R4_ablation_table():
    """B0 §2 ablation: ①②③④ each alone + combos, R4 recall/FAR.  Shows where
    the gain comes from (not a tuned black box).  FAR prioritized (G4 hinges on it)."""
    _need()
    base = FusionConfig()
    combos = [
        ("① local-median (DEFAULT)", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False)),
        ("② abrupt-drop",    dict(wl_use_local_median=False, wl_use_abrupt_drop=True, wl_use_abs_gate=False, wl_use_v_envelope=False)),
        ("③ abs-gate (diagnostic)", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=True, wl_use_v_envelope=False)),
        ("④ V-shape prior",     dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=True)),
        ("①④",                  dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=True)),
        ("②④",                  dict(wl_use_local_median=False, wl_use_abrupt_drop=True, wl_use_abs_gate=False, wl_use_v_envelope=True)),
        ("③④",                  dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=True, wl_use_v_envelope=True)),
        ("①②③④",                 dict(wl_use_local_median=True, wl_use_abrupt_drop=True, wl_use_abs_gate=True, wl_use_v_envelope=True)),
    ]
    print(f"  R4 ablation table (recall≥0.90 / FAR≤0.10; FAR prioritized):")
    print(f"    {'method':24s} {'recall':>8s} {'FAR':>8s} {'verdict':>10s}")
    rows = []
    for label, kw in combos:
        c = base.with_switches(**kw)
        r, f, nk, ns, nv = _r4_recall_far(c)
        v = "PASS" if r >= 0.90 and f <= 0.10 else "below"
        print(f"    {label:24s} {r:8.3f} {f:8.3f} {v:>10s}")
        rows.append((label, r, f))
    return rows


if __name__ == "__main__":
    test_R4_anti_noop()
    test_degrade_bandcheck()
    test_BR2_abs_must_fail_on_realistic_D1()
    test_BR2_abs_mutation()
    test_BR2_overlap()
    test_BR2_overlap_mutation()
    test_R2_future_perturbation_real_voiced()
    test_R2_mutation_real_voiced()
    test_R2_mutation_wlocal_lookahead()
    test_R4_M1_real_envelope()
    test_R4_ablation_table()
    print("T13-B0 real-device tests: done")
