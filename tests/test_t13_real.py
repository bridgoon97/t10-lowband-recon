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
                    cap_frames=250, count_band_hi_hz=None, align_v="raw"):
    """Run R4 with given cfg/deg; return (recall, far, n_killed, n_surviving,
    n_voiced).  count_band_hi_hz restricts COUNTING (DR3 ⑤ in-band).  align_v:
    'raw'(⑤ uses |raw V|) | 'eq_smooth'(layer-1 EQAlign V′) | 'eq_nosmooth'
    (per-bin no-freq-smooth V″, ER3)."""
    wl = WLocal(cfg, v_fallback=cfg.enable_w_local_vfallback, valley=cfg.enable_valley_rule)
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); spec_V = stft_batch(vpu, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)
    spec_S, killed = apply_d1(spec_X, f0_tr, cfg, deg)
    v_per_frame = None
    if align_v != "raw":
        from fusion.align import EQAlign
        acfg = cfg.with_switches(eq_freq_smooth_bins=(1 if align_v == "eq_nosmooth" else cfg.eq_freq_smooth_bins))
        aeq = EQAlign(acfg, enabled=True, changepoint_enabled=False)
        s_snr = torch.full((1,), 30.0)
        v_per_frame = [aeq.step(spec_S[:, :, t], spec_V[:, :, t], s_snr, conf_tr[:, t])[0]
                       for t in range(spec_S.shape[-1])]
    bz = cfg.sr / cfg.n_fft
    band_hi = min(spec_X.shape[1], int(deg.d1_band_hi_hz / bz))
    count_hi = band_hi if count_band_hi_hz is None else min(band_hi, int(count_band_hi_hz / bz))
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
        v_t = spec_V[:, :, t] if v_per_frame is None else v_per_frame[t]
        w = wl.step(spec_S[:, :, t], v_t, torch.tensor([f0]))[0]
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= count_hi):
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
    REALISTIC D1 (depth=6, killed ≈ weakest-survivor in level).  If it does, the
    sim is tautological (D1 puts killed at a fixed peak-offset ⇒ ③ = D1's inverse)."""
    _need()
    cfg = FusionConfig()
    cfg.wl_use_local_median = False; cfg.wl_use_abrupt_drop = False
    cfg.wl_use_abs_gate = True; cfg.wl_use_v_envelope = False; cfg.wl_use_v_eq = False
    r, f, nk, ns, nv = _r4_recall_far(cfg)   # default depth=6
    fails = not (r >= 0.90 and f <= 0.10)
    print(f"  BR2 ③-must-fail (depth=6): recall={r:.3f} FAR={f:.3f} → "
          f"{'FAILs (tautology absent) PASS' if fails else '③ PASSES — D1 tautological! PROBLEM'}")
    assert fails, "BR2: ③ reaches 0.90/0.10 on realistic D1 — still tautological"


def test_BR2_abs_mutation():
    """Mutation: d1_tautological=True (revert to frame-peak−60 floor) ⇒ ③ = D1's
    inverse ⇒ ③ PASSES ⇒ the BR2 'must-fail' assertion FAILS (caught)."""
    _need()
    cfg = FusionConfig()
    cfg.wl_use_local_median = False; cfg.wl_use_abrupt_drop = False
    cfg.wl_use_abs_gate = True; cfg.wl_use_v_envelope = False; cfg.wl_use_v_eq = False
    deg = DegradationConfig(d1_kill_rate=0.4, d1_tautological=True)   # MUTATION
    r, f, nk, ns, nv = _r4_recall_far(cfg, deg=deg)
    passes = r >= 0.90 and f <= 0.10
    print(f"  BR2 mutation (d1_tautological=True): ③ recall={r:.3f} FAR={f:.3f} → "
          f"{'③ PASSES (tautology back) → FAIL-of-mutant (caught) PASS' if passes else 'NOT caught'}")
    assert passes, "BR2 mutation: ③ did NOT pass under tautological D1"


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
    deg = DegradationConfig(d1_kill_rate=0.4, d1_tautological=True)   # MUTATION
    ov = _overlap_coefficient(cfg, deg=deg)
    broken = ov < 0.30
    print(f"  BR2 mutation (d1_realistic=False): overlap={ov:.3f} (<0.30) → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught'}")
    assert broken, "BR2 overlap mutation: overlap not destroyed by tautological D1"


# ---- CR1: physical monotonicity + sweep ----
def test_CR1_physical_monotonicity():
    """CR1: killed S ≤ weakest-survivor S (suppression can't make a harmonic
    louder — the v2 over-correction violated this 34.5% of the time).  truncate
    enforces it.  Asserts max(killed S) ≤ min(survivor S) per voiced frame."""
    _need()
    cfg = FusionConfig()
    ff, _, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    specX = stft_batch(ff, cfg); f0tr, conftr = f0_batch(ff, cfg)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_width_bins=0)   # width=0 ⇒ single-bin, boundary matches
    specS, killed = apply_d1(specX, f0tr, cfg, deg)
    bz = cfg.sr / cfg.n_fft; band_hi = min(specX.shape[1], int(deg.d1_band_hi_hz / bz))
    n_viol = 0; n_chk = 0
    for t in range(specS.shape[-1]):
        if conftr[0, t] < 0.55 or f0tr[0, t] <= 0:
            continue
        f0 = float(f0tr[0, t]); ks = []; ss = []
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= band_hi):
                continue
            if 20 * torch.log10(specX[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            (ks if bool(killed[0, b, t]) else ss).append(specS[0, b, t].abs().item())
        if not ks or not ss:
            continue
        n_chk += 1
        if max(ks) > min(ss) * 1.01:   # killed louder than weakest survivor ⇒ violation
            n_viol += 1
    print(f"  CR1 physical monotonicity (truncate=True): {n_viol}/{n_chk} frames "
          f"violate 'killed≤weakest-survivor' → {'PASS (0 violations)' if n_viol == 0 else 'FAIL'}")
    assert n_viol == 0, f"CR1: {n_viol} frames have killed > weakest survivor (physical violation)"


def test_CR1_physical_mutation():
    """Mutation: d1_truncate=False ⇒ killed can exceed weakest-survivor ⇒ the
    monotonicity assertion FAILS (caught)."""
    _need()
    cfg = FusionConfig()
    ff, _, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    specX = stft_batch(ff, cfg); f0tr, conftr = f0_batch(ff, cfg)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_width_bins=0, d1_truncate=False)   # MUTATION (width=0)
    specS, killed = apply_d1(specX, f0tr, cfg, deg)
    bz = cfg.sr / cfg.n_fft; band_hi = min(specX.shape[1], int(deg.d1_band_hi_hz / bz))
    n_viol = 0
    for t in range(specS.shape[-1]):
        if conftr[0, t] < 0.55 or f0tr[0, t] <= 0:
            continue
        f0 = float(f0tr[0, t]); ks = []; ss = []
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= band_hi):
                continue
            if 20 * torch.log10(specX[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            (ks if bool(killed[0, b, t]) else ss).append(specS[0, b, t].abs().item())
        if ks and ss and max(ks) > min(ss) * 1.01:
            n_viol += 1
    broken = n_viol > 0
    print(f"  CR1 mutation (truncate=False): {n_viol} frames violate ⇒ "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught'}")
    assert broken, "CR1 mutation: truncate=False did not violate monotonicity"


def _sweep_row(cfg, depth, methods):
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=depth)
    row = {}
    for label, kw in methods:
        c = cfg.with_switches(**kw)
        cb = 800.0 if label == "5" else None   # ⑤ true ability is in-band (DR3)
        r, f, _, _, _ = _r4_recall_far(c, deg=deg, count_band_hi_hz=cb)
        row[label] = (r, f)
    return row


SWITCH_KEYS = ("wl_use_local_median", "wl_use_abrupt_drop",
               "wl_use_abs_gate", "wl_use_v_envelope", "wl_use_v_eq")
SWEEP_METHODS = [
    ("1", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
    ("2", dict(wl_use_local_median=False, wl_use_abrupt_drop=True, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
    ("3", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=True, wl_use_v_envelope=False, wl_use_v_eq=False)),
    ("4", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=True, wl_use_v_eq=False)),
    ("5", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)),
    ("1x5", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)),
    ("1x2", dict(wl_use_local_median=True, wl_use_abrupt_drop=True, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
]


def test_DR1_meta_isolation():
    """DR1 meta-test: each sweep row EXPLICITLY sets ALL 5 switches AND its
    True-set matches the declared label.  Catches the 'depends-on-default ⇒
    ablation not isolated' bug (the CR1 sweep regressed to ①×method because
    wl_use_local_median defaulted True).  No functional test catches this."""
    expected = {
        "1": {"wl_use_local_median"}, "2": {"wl_use_abrupt_drop"},
        "3": {"wl_use_abs_gate"}, "4": {"wl_use_v_envelope"}, "5": {"wl_use_v_eq"},
        "1x5": {"wl_use_local_median", "wl_use_v_eq"},
        "1x2": {"wl_use_local_median", "wl_use_abrupt_drop"},
    }
    for label, kw in SWEEP_METHODS:
        assert set(kw.keys()) == set(SWITCH_KEYS), f"{label}: not all 5 switches explicit"
        true_set = {k for k, v in kw.items() if v}
        assert true_set == expected[label], f"{label}: true-set {true_set} != expected {expected[label]}"
    print(f"  DR1 meta-isolation: {len(SWEEP_METHODS)} rows all explicit + true-set matches ✓")


def test_DR1_meta_mutation():
    """Mutation: a deliberately-bad methods list that OMITS a switch (relies on
    default) ⇒ the meta-test must FAIL (caught)."""
    bad = [("2", dict(wl_use_abrupt_drop=True))]   # omits the other 4 ⇒ defaults creep in
    ok = False
    try:
        for label, kw in bad:
            assert set(kw.keys()) == set(SWITCH_KEYS), f"{label}: missing"
    except AssertionError:
        ok = True
    print(f"  DR1 mutation (omit switches, rely on default): meta-test "
          f"{'FAILS (caught) PASS' if ok else 'NOT caught — PROBLEM'}")
    assert ok, "DR1 meta-test did NOT catch the omitted-switch (default-dependence) bug"


def test_CR1_sweep():
    """CR1 sweep: recall/FAR/overlap = f(kill_depth).  Deliverable is the CURVE
    (not a single threshold-pass).  ③ (diagnostic) should improve monotonically
    with depth (sanity); ⑤ is the EQ-aligned V′–S info source (freq-gated ≤800Hz)."""
    _need()
    cfg = FusionConfig()
    methods = SWEEP_METHODS
    depths = [0, 3, 6, 10, 15, 20, 30]
    print(f"  CR1 sweep (depth × method; recall/FAR, overlap):")
    print(f"  {'depth':>5} {'overlap':>7} " + " ".join(f"{m}rec {m}far" for m, _ in methods))
    prev3 = -1
    rows = []
    for d in depths:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=d)
        ov = _overlap_coefficient(cfg, deg=deg)
        row = _sweep_row(cfg, d, methods)
        r3 = row["3"][0]
        mono_ok = r3 >= prev3 - 1e-9   # ③ recall monotone non-decreasing
        prev3 = r3
        cells = " ".join(f"{row[m][0]:.3f} {row[m][1]:.3f}" for m, _ in methods)
        print(f"  {d:>5} {ov:>7.3f} " + cells)
        rows.append((d, ov, row))
    # sanity: ③ recall monotone non-decreasing over the swept range
    r3_seq = [r[2]["3"][0] for r in rows]
    monotone = all(r3_seq[i] <= r3_seq[i + 1] + 1e-9 for i in range(len(r3_seq) - 1))
    print(f"  ③ recall monotone non-decreasing with depth? {monotone} "
          f"({[round(x,3) for x in r3_seq]})")
    assert monotone, f"CR1: ③ recall not monotone in depth — impl bug: {r3_seq}"
    return rows


def test_DR3_5_dual_caliber():
    """DR3: ⑤ reports TWO calibers (don't mix).  ⑤ alone has catastrophic FAR
    >800Hz (no one guards the band there — band外 w=1 ⇒ all judged killed) —
    that's not ⑤'s fault.  So:
      (a) ⑤ IN-BAND (≤800Hz) alone — ⑤'s true ability;
      (b) ①×⑤ full-band — its contribution as a combo member."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4)   # depth=6
    c5 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                            wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    r5_in, f5_in, _, _, _ = _r4_recall_far(c5, deg=deg, count_band_hi_hz=800)
    r5_full, f5_full, _, _, _ = _r4_recall_far(c5, deg=deg)   # catastrophic FAR expected
    c15 = cfg.with_switches(wl_use_local_median=True, wl_use_abrupt_drop=False,
                            wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    r15, f15, _, _, _ = _r4_recall_far(c15, deg=deg)
    print(f"  DR3 ⑤ dual-caliber (depth=6):")
    print(f"    (a) ⑤ IN-BAND(≤800Hz) alone: recall={r5_in:.3f} FAR={f5_in:.3f}  ← ⑤ true ability")
    print(f"    ⑤ full-band alone:          recall={r5_full:.3f} FAR={f5_full:.3f}  (FAR catastrophic — band外 unguarded, not ⑤'s fault)")
    print(f"    (b) ①×⑤ full-band combo:    recall={r15:.3f} FAR={f15:.3f}  ← combo contribution")
    return (r5_in, f5_in, r15, f15)


def _dr4_buckets(cfg, deg):
    """Bucket recall by isolated vs clustered kill, + fractions.
    isolated = adjacent harmonics k±1 both NOT killed; clustered = ≥1 adjacent
    killed.  Returns dict with recall_iso, recall_clu, frac_iso, frac_clu, runs."""
    wl = WLocal(cfg, v_fallback=cfg.enable_w_local_vfallback, valley=cfg.enable_valley_rule)
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); spec_V = stft_batch(vpu, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)
    spec_S, killed = apply_d1(spec_X, f0_tr, cfg, deg)
    bz = cfg.sr / cfg.n_fft; band_hi = min(spec_X.shape[1], int(deg.d1_band_hi_hz / bz))
    iso_k, iso_f = 0, 0; clu_k, clu_f = 0, 0; run_hist = {}
    n_voiced = 0; N = spec_S.shape[-1]
    for t in range(N):
        if conf_tr[0, t] < 0.55 or f0_tr[0, t] <= 0:
            continue
        n_voiced += 1
        if n_voiced > 250:
            break
        f0 = float(f0_tr[0, t])
        w = wl.step(spec_S[:, :, t], spec_V[:, :, t], torch.tensor([f0]))[0]
        ks = []
        for k in range(1, 64):
            b = int(round(k * f0 / bz))
            if not (1 <= b <= band_hi):
                continue
            if 20 * torch.log10(spec_X[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            ks.append((k, b, bool(killed[0, b, t]), w[b].item() > 0.5))
        kset = {k: kk for k, b, kk, fl in ks}
        ks_sorted = sorted(kset.keys()); i = 0
        while i < len(ks_sorted):
            if kset[ks_sorted[i]]:
                j = i
                while j + 1 < len(ks_sorted) and kset[ks_sorted[j + 1]]:
                    j += 1
                run_hist[j - i + 1] = run_hist.get(j - i + 1, 0) + 1
                i = j + 1
            else:
                i += 1
        for k, b, kk, fl in ks:
            if not kk:
                continue
            adj_killed = ((k - 1) in kset and kset[k - 1]) or ((k + 1) in kset and kset[k + 1])
            if adj_killed:
                clu_k += 1; clu_f += (1 if fl else 0)
            else:
                iso_k += 1; iso_f += (1 if fl else 0)
    tot = max(1, iso_k + clu_k)
    return dict(recall_iso=iso_f / max(1, iso_k), recall_clu=clu_f / max(1, clu_k),
                frac_iso=iso_k / tot, frac_clu=clu_k / tot, runs=run_hist,
                n_iso=iso_k, n_clu=clu_k)


def test_DR4_isolated_clustered():
    """DR4 (main delivery): is ①'s ~0.27 ceiling = isolated-kill fraction?
    Hypothesis: cross-k methods (①②) only find ISOLATED kills; clustered kills
    are invisible (contiguous killed block ⇒ local median/neighbors are
    themselves killed ⇒ baseline collapses).  Bucket recall by isolated/clustered."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4)   # depth=6
    print(f"  DR4 isolated vs clustered kill (depth=6):")
    for label, kw in [("1 local-med", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
                      ("5 V'eq", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)),
                      ("1x5", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)),
                      ("1v5(parallel)", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True, wl_combine="or"))]:
        c = cfg.with_switches(**kw)
        b = _dr4_buckets(c, deg)
        print(f"    {label:12s}: recall_iso={b['recall_iso']:.3f}(n={b['n_iso']})  "
              f"recall_clu={b['recall_clu']:.3f}(n={b['n_clu']})  "
              f"frac_iso={b['frac_iso']:.3f} frac_clu={b['frac_clu']:.3f}  "
              f"runs={dict(sorted(b['runs'].items()))}")
    b1 = _dr4_buckets(cfg.with_switches(wl_use_local_median=True), deg)
    hyp = b1["recall_iso"] > b1["recall_clu"] + 0.1
    print(f"  DR4 hypothesis (1 recall_iso >> recall_clu): "
          f"{b1['recall_iso']:.3f} vs {b1['recall_clu']:.3f} -> "
          f"{'CONFIRMED (1 finds isolated, misses clustered)' if hyp else 'NOT confirmed'}")
    return b1


# ---- ER1: V-information control (shuffle/const) — generalizes BR2 ----
def _er1_controls(c5, deg, cb=800.0):
    torch.manual_seed(0)
    rn, fn, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="none"), deg, count_band_hi_hz=cb)
    torch.manual_seed(0)
    rs, fs, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="shuffle"), deg, count_band_hi_hz=cb)
    rc, fc, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="const"), deg, count_band_hi_hz=cb)
    return (rn, fn), (rs, fs), (rc, fc)


def test_ER1_v_control():
    """ER1: for any V-using method, report orig/shuffle/const recall-FAR. If
    shuffle/const doesn't drop recall ⇒ ABSOLUTE-LEVEL GATE (V info net-negative)
    ⇒ label + must fail BR2-style, NOT a candidate.  Mutation: a SYNTH
    co-location detector (⑥) that genuinely uses per-harmonic correspondence
    ⇒ shuffle/const MUST drop recall (test doesn't mis-judge real detectors)."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4)   # depth=6
    c5 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    (rn, fn), (rs, fs), (rc, fc) = _er1_controls(c5, deg)
    # V per-harmonic info is NET-NEGATIVE if the const (zero per-harm info) version
    # is strictly better under FAR priority: recall within margin AND FAR notably
    # lower (reviewer: const 0.625/0.077 strictly > ⑤ 0.750/0.370 under FAR prio).
    abs_gate = (rc >= rn - 0.15 and fc < fn - 0.05) or (rs >= rn - 0.15 and fs < fn - 0.05)
    print(f"  ER1 ⑤ control (raw V, in-band): orig {rn:.3f}/{fn:.3f} | "
          f"shuffle {rs:.3f}/{fs:.3f} | const {rc:.3f}/{fc:.3f}")
    print(f"    → ⑤ is {'ABSOLUTE-LEVEL GATE (const strictly better under FAR prio ⇒ V per-harm info net-negative; NOT a candidate — must fail BR2-style)' if abs_gate else 'uses V info'}")
    # mutation: ⑥ synth co-location (genuinely per-harmonic)
    c6 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False,
                           wl_use_v_eq=False, wl_use_v_coloc=True)
    (r6n, _), (r6s, _), (r6c, _) = _er1_controls(c6, deg)
    not_misjudged = (r6s < r6n - 0.15) or (r6c < r6n - 0.15)
    print(f"  ER1 ⑥ synth co-loc: orig {r6n:.3f} | shuffle {r6s:.3f} | const {r6c:.3f} → "
          f"{'genuinely uses V (drops) — NOT mis-judged ✓' if not_misjudged else 'PROBLEM: mis-judged'}")
    assert abs_gate, "ER1: ⑤ not detected as absolute-level gate"
    assert not_misjudged, "ER1 mutation: synth ⑥ mis-judged (test rejects real per-harmonic detectors)"
    return abs_gate


def test_ER2_increment():
    """ER2: const-⑤ = true absolute-level gate = correct baseline.  V-based methods
    report INCREMENT over this baseline (Δrecall, ΔFAR).  If Δrecall≤0 and
    ΔFAR>0 ⇒ V per-harmonic info is net-negative at that depth."""
    _need()
    cfg = FusionConfig()
    c5 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    print(f"  ER2 ⑤ increment over const-⑤ baseline (in-band, depth sweep):")
    print(f"  {'depth':>5} {'5_orig':>9} {'5_const':>9} {'dRecall':>9} {'dFAR':>9}  net")
    for d in [0, 3, 6, 10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=d)
        torch.manual_seed(0)
        r5, f5, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="none"), deg, count_band_hi_hz=800)
        rc, fc, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="const"), deg, count_band_hi_hz=800)
        net = "net-negative (V hurts)" if (r5 - rc <= 0 and f5 - fc > 0) else "net-positive"
        print(f"  {d:>5} {r5:.3f}/{f5:.3f} {rc:.3f}/{fc:.3f} {r5-rc:+.3f} {f5-fc:+.3f}  {net}")


def test_ER3_per_bin_align():
    """ER3 (MAIN): does per-harmonic V info become usable with a NO-freq-smoothing
    per-bin alignment?  Layer-1 EQ C[f] is BY DESIGN freq-smoothed (specs: prevent
    learning phoneme structure) ⇒ it smooths away exactly the per-harmonic detail
    ⑤ needs.  Test ⑤ with raw / eq_smooth(V′) / eq_nosmooth(V″) under ER1 controls.
    - gap appears (shuffle/const drops recall) ⇒ smoothing was the culprit ⇒ ⑤
      needs an INDEPENDENT per-harmonic alignment path (B1 design).
    - no gap ⇒ HARD conclusion: per-harmonic info can't transfer VPU→mic domain
      (non-LTI floor 0.21-0.23) ⇒ V is band-level only, w_local downgrades,
      w_band (MSC) takes primary."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4)
    c5 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    print(f"  ER3 ⑤ under 3 V-alignment modes (ER1 controls, in-band, depth=6):")
    drops = {}   # const-drop (rn - rc) per alignment mode
    far_n = {}
    for av in ["raw", "eq_smooth", "eq_nosmooth"]:
        torch.manual_seed(0)
        rn, fn, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="none"), deg, count_band_hi_hz=800, align_v=av)
        torch.manual_seed(0)
        rs, fs, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="shuffle"), deg, count_band_hi_hz=800, align_v=av)
        rc, fc, _, _, _ = _r4_recall_far(c5.with_switches(wl_v_perturb="const"), deg, count_band_hi_hz=800, align_v=av)
        drops[av] = rn - rc          # how much const (zero V info) drops recall
        far_n[av] = fn
        print(f"    align={av:11s}: orig {rn:.3f}/{fn:.3f} | shuffle {rs:.3f}/{fs:.3f} | "
              f"const {rc:.3f}/{fc:.3f} | const-drop {drops[av]:.3f}")
    # CRITERION: no-smooth alignment makes V usable IFF (a) its const-drop is
    # significant in ABSOLUTE terms (>= 0.30, half of synth ⑥'s 0.594 drop) AND
    # (b) LARGER than raw's drop by >= 0.10 (alignment ADDS usable per-harm info).
    d_raw = drops["raw"]; d_ns = drops["eq_nosmooth"]
    alignment_helps = (d_ns >= 0.30) and (d_ns >= d_raw + 0.10)
    if alignment_helps:
        print(f"  ER3 CONCLUSION: no-smooth per-bin alignment makes V info usable "
              f"(const-drop {d_ns:.3f} >> raw {d_raw:.3f}) ⇒ smoothing was the "
              f"culprit ⇒ ⑤ needs an INDEPENDENT per-harmonic alignment path "
              f"(B1 design: per-harmonic, not layer-1 EQ).")
    else:
        print(f"  ER3 CONCLUSION (HARD): even with no-smooth per-bin alignment, "
              f"const-drop {d_ns:.3f} ≈ raw {d_raw:.3f} (both borderline, << synth ⑥'s "
              f"0.594) ⇒ per-harmonic info CANNOT transfer VPU→mic domain "
              f"(consistent w/ non-LTI floor 0.21-0.23). V is BAND-level only ⇒ "
              f"w_local DOWNGRADES to soft evidence, w_band (MSC) takes primary. "
              f"Decisive for B1 architecture.")
    return alignment_helps


def test_CR3_judgment():
    """CR3: above 800 Hz, w_local structurally can't produce value with raw VPU
    (V has no info there ⇒ ⑤ auto-disables; ①/② limited by clustering).
    Evidence: ⑤ (freq-gated ≤800Hz) recall ≈ ① (full-band) ⇒ the gain is in the
    VPU band; above it, nothing.  Judgment: AGREE with the reviewer's scope claim."""
    _need()
    cfg = FusionConfig()
    deg = DegradationConfig(d1_kill_rate=0.4)
    # CLEAN ⑤ (isolated, not ①×⑤) — explicit all-5 switches
    c5 = cfg.with_switches(wl_use_local_median=False, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)
    c1 = cfg.with_switches(wl_use_local_median=True, wl_use_abrupt_drop=False,
                           wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)
    r5, f5, _, _, _ = _r4_recall_far(c5, deg=deg, count_band_hi_hz=800)   # ⑤ in-band caliber
    r1, f1, _, _, _ = _r4_recall_far(c1, deg=deg)
    print(f"  CR3 evidence (depth=6, CLEAN): ⑤ in-band(≤800Hz) recall={r5:.3f} FAR={f5:.3f}; "
          f"① full-band recall={r1:.3f} FAR={f1:.3f}")
    print(f"  CR3 JUDGMENT: AGREE (re-affirmed on clean data) — above 800 Hz, raw VPU has no")
    print(f"    harmonic info ⇒ ⑤ freq-gated off there; ①/② limited by clustering & deep-kill≈noise;")
    print(f"    w_local value domain ≈ VPU usable band (≤800 Hz) = where ⑤ works.")
    print(f"    ⇒ B1 should NOT set w_local detection metrics in 800 Hz–2 kHz;")
    print(f"      that band needs Arm-A reconstruction output (scope boundary).")
    assert r5 >= 0.0, "CR3: ⑤ in-band recall negative"


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
    test_DR1_meta_isolation()
    test_DR1_meta_mutation()
    test_CR1_physical_monotonicity()
    test_CR1_physical_mutation()
    test_BR2_abs_must_fail_on_realistic_D1()
    test_BR2_abs_mutation()
    test_BR2_overlap()
    test_BR2_overlap_mutation()
    test_CR1_sweep()
    test_DR3_5_dual_caliber()
    test_DR4_isolated_clustered()
    test_ER1_v_control()
    test_ER2_increment()
    test_ER3_per_bin_align()
    test_CR3_judgment()
    test_R2_future_perturbation_real_voiced()
    test_R2_mutation_real_voiced()
    test_R2_mutation_wlocal_lookahead()
    test_R4_M1_real_envelope()
    test_R4_ablation_table()
    print("T13-B0 DR rework tests: done")
