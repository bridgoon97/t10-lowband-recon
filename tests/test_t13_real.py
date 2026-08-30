"""T13-B1 REAL-device tests (0624/ only; 0625/ held-out, NOT touched).

  R2: G5 future-perturbation on REAL voiced (FF as S, VPU as V) — causal holds.
  BR2 (rewritten, B1): a PURE absolute-level detector must FAIL at the LOW-
      separability end (depth ≤ 6); HIGH depth may pass (that's the task being
      genuinely easy, not the sim being too easy).  ER1 (shuffle/const on any
      V-using method) re-applied at BAND granularity in test_t13_b1.
  FR1 (B0.5, retained): c_V = in-band SNR — level-invariance / strict-decrease /
      ratchet-recovery (+ mutations).
  FR2 (B0.5, retained): adaptive comfort-noise level (+ mutation).

🔴 BOUNDARY: all conclusions hold only for MALE speech (F0 87–124 Hz), normal
volume — 0624/4 speakers all male, zero female.  Not extrapolated.

Per-harmonic B0 tests (CR1 sweep, DR3/DR4, ER1–ER3, CR3, R4 ablation/M1, FR3)
were REMOVED in AC3 (per-harmonic ①②③④⑤ deleted; B0.5 proved per-harm info
can't transfer VPU→mic).  Their conclusions are retained in reports/T13/README.md.
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


# ================================================================ R2 / G5 ==
def test_R2_future_perturbation_real_voiced():
    """G5: future-perturbation on REAL voiced FF/VPU — past bit-identical."""
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


def test_R2_mutation_real_voiced():
    """G5 mutation sanity on REAL voiced: a whole-segment-stat (global-mean-norm
    of Y) leaks on any signal; caught (>1e-6)."""
    _need()
    from tests.test_t13_streaming import _MutantGlobalMeanNorm
    cfg = FusionConfig()
    s, v, sr = _voiced_SV(seg_s=4.0)
    mutant = _MutantGlobalMeanNorm(cfg)
    y_full = mutant.process_batch(s, v)
    T = s.shape[-1]; P = T // 2
    s_m = s.clone(); s_m[:, P:] = 0.0
    v_m = v.clone(); v_m[:, P:] = 0.0
    y_m = mutant.process_batch(s_m, v_m)
    K = max(0, P - cfg.win)
    leak = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
    detected = leak > 1e-6
    print(f"  R2 mutation (global-mean-norm(Y)) on REAL voiced: leak={leak:.3e} → "
          f"{'FAIL-of-mutant (caught) PASS' if detected else 'NOT caught'}")
    assert detected, "mutation not caught on voiced (leak ≤ 1e-6)"
    return leak


def test_R2_mutation_wlocal_lookahead():
    """G5: look-ahead in the w_local path leaks proportionally to its
    contribution.  Voiced (w_local active) ⇒ large leak; white (w_local≈0) ⇒
    tiny — voiced gives the test teeth."""
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


# ================================================================ BR2 (B1) =
def _r4_recall_far(cfg, deg=DegradationConfig(d1_kill_rate=0.4),
                    cap_frames=250, count_band_hi_hz=None):
    """Run WLocal (band-level, AC3) with given cfg/deg; return (recall, far,
    n_killed, n_surviving, n_voiced) measured on D1's per-harmonic kill ground
    truth (count_band_hi_hz restricts counting).  Band-level w_local flags a
    killed HARMONIC if its bin's band-gate w>0.5."""
    from fusion.degrade import apply_d2, apply_d3, apply_d4
    wl = WLocal(cfg)
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    if deg.d4_envelope:
        ff = apply_d4(ff, deg)
    spec_X = stft_batch(ff, cfg); spec_V = stft_batch(vpu, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)
    spec_S, killed = apply_d1(spec_X, f0_tr, cfg, deg)
    spec_S = apply_d2(spec_S, deg); spec_S = apply_d3(spec_S, cfg, deg)
    bz = cfg.sr / cfg.n_fft
    band_hi = min(spec_X.shape[1], int(deg.d1_band_hi_hz / bz))
    count_hi = band_hi if count_band_hi_hz is None else min(band_hi, int(count_band_hi_hz / bz))
    Pk, Ps = [], []
    n_voiced = 0; N = spec_S.shape[-1]
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
            if not (1 <= b <= count_hi):
                continue
            if 20 * torch.log10(spec_X[0, b, t].abs().clamp_min(1e-8)).item() < -60:
                continue
            flagged = w[b].item() > 0.5
            (Pk if bool(killed[0, b, t]) else Ps).append(flagged)
    return (sum(Pk) / max(1, len(Pk)), sum(Ps) / max(1, len(Ps)),
            len(Pk), len(Ps), n_voiced)


def test_BR2_abs_must_fail_low_depth():
    """BR2 (B1 rewrite): a PURE absolute-level detector (w_local_band IS one —
    const-⑤ style, V's overall level threshold) must FAIL the 0.90/0.10 gate at
    the LOW-separability end (depth ≤ 6).  At high depth it MAY pass — that's
    the task being genuinely easy, not the sim being too easy."""
    _need()
    cfg = FusionConfig()
    r6, f6, _, _, _ = _r4_recall_far(cfg, deg=DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=6.0),
                                       count_band_hi_hz=800)
    fails_low = not (r6 >= 0.90 and f6 <= 0.10)
    r20, f20, _, _, _ = _r4_recall_far(cfg, deg=DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0),
                                         count_band_hi_hz=800)
    print(f"  BR2 (B1): const-⑤(w_local_band) @depth6 recall={r6:.3f} FAR={f6:.3f} "
          f"{'FAIL gate ✓' if fails_low else 'PASSES — PROBLEM'}; "
          f"@depth20 recall={r20:.3f} FAR={f20:.3f} (high depth allowed to pass)")
    assert fails_low, f"BR2: w_local_band passes at depth=6 ({r6:.3f}/{f6:.3f}) — sim too easy at low depth"


def test_BR2_high_depth_allowed():
    """BR2 complement: at HIGH depth (≥20), the absolute-level gate MAY pass —
    the task is genuinely easy there.  (B0.5 measured const-⑤@depth20≈0.969/0.077.)"""
    _need()
    cfg = FusionConfig()
    r, f, _, _, _ = _r4_recall_far(cfg, deg=DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0),
                                       count_band_hi_hz=800)
    print(f"  BR2 high-depth-allowed: w_local_band @depth20 recall={r:.3f} FAR={f:.3f} "
          f"(may pass; recorded, not gated)")


def test_R4_anti_noop():
    """B0 §1 anti-no-op: D1=40% must actually kill in-band harmonics."""
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
    for t in range(spec_X.shape[-1]):
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
    print(f"  R4 anti-no-op: in-band killed={n_killed_in}/{n_harm_in} = {ratio:.3f}")
    assert n_killed_in > 0, "D1 killed ZERO in-band harmonics"
    assert 0.25 <= ratio <= 0.55, f"in-band kill ratio {ratio} not ~0.40"


def test_degrade_bandcheck():
    """D2/D3/D4 must not sort harmonics across the band (only D1 does)."""
    _need()
    import inspect
    from fusion.degrade import apply_d2, apply_d3, apply_d4
    src = inspect.getsource(apply_d2) + inspect.getsource(apply_d3) + inspect.getsource(apply_d4)
    has_sort = "sorted(" in src or "argsort" in src or ".sort(" in src
    print(f"  degrade band-check: D2/D3/D4 cross-band sort? {has_sort} → "
          f"{'PROBLEM' if has_sort else 'none ✓'}")
    assert not has_sort, "D2/D3/D4 unexpectedly sort across band"


# ===== FR1 (B0.5, retained): c_V = in-band SNR ===========================
# (CV mechanism tests live in tests/test_t13_mechanisms.py — FR1-a/c + mutations.)
# Re-exposed here only as smoke checks that the real pipeline c_V behaves.
def test_FR1_real_smoke():
    """FR1 smoke on REAL 0624: c_V (new SNR design) is level-invariant under
    joint S+V scaling (spread ≤0.05), reported (the full FR1-a/b/c + mutations
    are in test_t13_mechanisms)."""
    _need()
    cfg = FusionConfig()
    from fusion.decision import CV
    ff, vpu, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    settled = []
    for db in [0, -6, -12]:
        g = 10 ** (db / 20.0)
        ss = stft_batch(ff * g, cfg); vv = stft_batch(vpu * g, cfg)
        cv = CV(cfg, enabled=True)
        for t in range(min(400, ss.shape[-1])):
            cv.step(vv[:, :, t], ss[:, :, t], torch.zeros(1, 257))
        settled.append(cv.c_v)
    spread = max(settled) - min(settled)
    print(f"  FR1 real smoke: c_V @0/-6/-12 dB = {[round(x, 4) for x in settled]} "
          f"spread={spread:.4f} (≤0.05) → {'PASS' if spread < 0.05 else 'FAIL'}")
    assert spread < 0.05, f"FR1 real: c_V not level-invariant (spread {spread})"


# ===== FR2 (B0.5, retained): adaptive comfort noise ======================
def _cn_synth_s(cfg, speech_db=0.0, seed=0):
    Fb = cfg.n_fft // 2 + 1; bz = cfg.sr / cfg.n_fft
    torch.manual_seed(seed)
    sp = torch.zeros(1, Fb, dtype=torch.complex64)
    s = 10 ** (speech_db / 20.0)
    for k in range(1, 9):
        b = int(round(k * 150 / bz))
        if 1 <= b < Fb: sp[0, b] = s / k + 0j
    return sp


def _cn_gap(cfg, scale_db, n_settle=300):
    from fusion.synthesis import ComfortNoise
    cn = ComfortNoise(cfg, enabled=True)
    s = _cn_synth_s(cfg) * (10 ** (scale_db / 20.0))
    y_out = s.clone()
    for _ in range(n_settle):
        y_out = cn.step(s, s.clone(), s)   # fresh per-frame y (no accumulation)
    added = (y_out - s).abs()
    comfort_db = 20 * torch.log10(added.max().clamp_min(1e-10)).item()
    return float(cn.speech_db_ema.item()), comfort_db


def test_FR2a_adaptive_gap():
    _need()
    cfg = FusionConfig()
    gaps = [_cn_gap(cfg, db) for db in [0, -6, -12, -20]]
    gvals = [sp - cn for sp, cn in gaps]
    spread = max(gvals) - min(gvals)
    print(f"  FR2-a adaptive gap: gaps={[round(x, 2) for x in gvals]} spread={spread:.3f} (≤1) → "
          f"{'PASS' if spread < 1.0 else 'FAIL'}")
    assert spread < 1.0


def test_FR2b_inaudible():
    _need()
    cfg = FusionConfig()
    sp, cn = _cn_gap(cfg, -20.0)
    gap = sp - cn
    print(f"  FR2-b inaudible @−20dB: gap={gap:.1f} (≥40) → {'PASS' if gap >= 39.95 else 'FAIL'}")
    assert gap >= 39.95


def test_FR2c_independent_of_w():
    _need()
    from fusion.synthesis import ComfortNoise
    cfg = FusionConfig()
    s = _cn_synth_s(cfg); v = _cn_synth_s(cfg, seed=7)
    cn1 = ComfortNoise(cfg, enabled=True); cn2 = ComfortNoise(cfg, enabled=True)
    for _ in range(300):
        y1 = cn1.step(s, s.clone(), s); y2 = cn2.step(s, s.clone(), v)
    diff = abs((y1 - s).abs().max().item() - (y2 - s).abs().max().item())
    print(f"  FR2-c independent of w: diff={diff:.3e} → {'PASS' if diff < 1e-6 else 'FAIL'}")
    assert diff < 1e-6


def test_FR2a_mutation():
    _need()
    cfg = FusionConfig(); cfg.cn_fixed_level_db = True
    gaps = [_cn_gap(cfg, db) for db in [0, -12, -20]]
    gvals = [sp - cn for sp, cn in gaps]
    spread = max(gvals) - min(gvals)
    print(f"  FR2-a mutation (fixed level): gaps={[round(x, 2) for x in gvals]} spread={spread:.3f} "
          f"(>1) → {'FAIL-of-mutant (caught) PASS' if spread > 1.0 else 'NOT caught'}")
    assert spread > 1.0


if __name__ == "__main__":
    test_R2_future_perturbation_real_voiced()
    test_R2_mutation_real_voiced()
    test_R2_mutation_wlocal_lookahead()
    test_BR2_abs_must_fail_low_depth()
    test_BR2_high_depth_allowed()
    test_R4_anti_noop()
    test_degrade_bandcheck()
    test_FR1_real_smoke()
    test_FR2a_adaptive_gap(); test_FR2b_inaudible(); test_FR2c_independent_of_w()
    test_FR2a_mutation()
    print("T13-B1 real tests: done")
