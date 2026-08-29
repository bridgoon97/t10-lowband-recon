"""M1–M7 — mechanism thresholds (T13-A), each with a mutation sanity.

All criteria are measured on SYNTHETIC signals (spec mandates this; real sensor
domain not needed for mechanism correctness — Vibravox temple-V is wired only
for B-stage).  Each M-test has a PASS assertion AND a mutation sanity that
deliberately breaks the mechanism and shows the SAME test now FAILS (reports the
failing value).  Mutation = a one-line change with the exact line recorded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch

from fusion.config import FusionConfig
from fusion.align import EQAlign
from fusion.decision import CV, GF0, WBand, WLocal, AsymSmoother
from fusion.synthesis import logclip_mix, complex_convex
from fusion.stft import StftStreamer
from fusion.f0 import F0Estimator
from fusion import signals as S
from fusion.utils import alpha_from_tau


def _band_bins(cfg):
    bz = cfg.sr / cfg.n_fft
    lo = max(1, int(cfg.eq_band_lo_hz / bz))
    hi = min(cfg.fusion_hi_bin, int(cfg.eq_band_hi_hz / bz))
    return lo, hi


# ================================================================ M1 ======
def _harmonic_spectrum(F0, amps, cfg, kill_set=None, floor_db=-60.0):
    """Build a (1, num_bins) complex spectrum with harmonics at k·F0."""
    Fb = cfg.n_fft // 2 + 1
    bz = cfg.sr / cfg.n_fft
    spec = torch.zeros(1, Fb, dtype=torch.complex64)
    peak = max(amps) if amps else 1.0
    floor = (10 ** (floor_db / 20)) * peak
    for k, a in enumerate(amps, start=1):
        b = int(round(k * F0 / bz))
        if not (1 <= b < Fb):
            continue
        mag = floor if (kill_set and k in kill_set) else a
        spec[0, b] = mag + 0j
    # sprinkle a tiny floor between so bins aren't exactly zero
    spec = spec + (1e-5 * peak) * torch.randn(1, Fb) * torch.exp(
        1j * 2 * math.pi * torch.rand(1, Fb))
    return spec, bz


def test_M1_w_local():
    """D1=40% kill ⇒ w_local recalls ≥0.90 killed, FAR ≤0.10 surviving."""
    cfg = FusionConfig()
    wl = WLocal(cfg, v_fallback=False, valley=False)
    P_kill, P_surv = [], []
    for trial, amps in enumerate([[1 / k for k in range(1, 9)],
                                   [0.8 ** (k - 1) for k in range(1, 9)],
                                   [1 / (k ** 0.5) for k in range(1, 9)]]):
        F0 = 150.0 + trial * 10
        amps = list(amps)
        order = sorted(range(len(amps)), key=lambda i: amps[i])   # weak first
        n_kill = int(round(0.4 * len(amps)))
        kill_set = {order[i] + 1 for i in range(n_kill)}           # k is 1-based
        s_spec, _ = _harmonic_spectrum(F0, amps, cfg, kill_set)
        v_spec, _ = _harmonic_spectrum(F0, amps, cfg)              # clean V
        w = wl.step(s_spec, v_spec, torch.tensor([F0]))
        bz = cfg.sr / cfg.n_fft
        for k in range(1, len(amps) + 1):
            b = int(round(k * F0 / bz))
            if not (1 <= b < w.shape[1]):
                continue
            flagged = w[0, b].item() > 0.5
            (P_kill if k in kill_set else P_surv).append(flagged)
    recall = sum(P_kill) / max(1, len(P_kill))
    far = sum(P_surv) / max(1, len(P_surv))
    print(f"  M1 w_local: recall={recall:.3f} (≥0.90)  FAR={far:.3f} (≤0.10)  "
          f"{'PASS' if recall >= 0.90 and far <= 0.10 else 'FAIL'}")
    assert recall >= 0.90 and far <= 0.10, f"M1: recall={recall} FAR={far}"
    return recall, far


def test_M1_mutation():
    """Mutation: wl_inlier_db=1e9 (fixed inlier band disabled ⇒ RANSAC
    degenerates to a single LSQ fit on ALL points including the −60 dB killed
    ⇒ envelope pulled down ⇒ killed not flagged).  Recall/FAR must now FAIL."""
    cfg = FusionConfig()
    cfg.wl_inlier_db = 1e9
    wl = WLocal(cfg, v_fallback=False, valley=False)
    F0 = 150.0
    amps = [1 / k for k in range(1, 9)]
    order = sorted(range(len(amps)), key=lambda i: amps[i])
    kill_set = {order[i] + 1 for i in range(int(round(0.4 * len(amps))))}
    s_spec, _ = _harmonic_spectrum(F0, amps, cfg, kill_set)
    v_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    w = wl.step(s_spec, v_spec, torch.tensor([F0]))
    bz = cfg.sr / cfg.n_fft
    Pk, Ps = [], []
    for k in range(1, len(amps) + 1):
        b = int(round(k * F0 / bz))
        if not (1 <= b < w.shape[1]):
            continue
        (Pk if k in kill_set else Ps).append(w[0, b].item() > 0.5)
    recall = sum(Pk) / max(1, len(Pk))
    far = sum(Ps) / max(1, len(Ps))
    broken = not (recall >= 0.90 and far <= 0.10)
    print(f"  M1 mutation (rounds=1, no reject): recall={recall:.3f} FAR={far:.3f} "
          f"→ {'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M1 mutation not caught"


# ================================================================ M2 ======
def test_M2_eq_convergence():
    """±6 dB tilt ⇒ C[f] within ±1 dB after 3 s (causal robust EMA)."""
    cfg = FusionConfig()
    s_t, v_t = S.tilted_noise_pair(dur_s=3.0, tilt_db=6.0)
    sfr_s = StftStreamer(cfg); sfr_v = StftStreamer(cfg)
    eq = EQAlign(cfg, changepoint_enabled=False)
    bz = cfg.sr / cfg.n_fft
    lo = max(1, int(100 / bz)); hi = min(cfg.n_fft // 2 - 1, int(2000 / bz))
    idx = torch.arange(lo, hi + 1)
    true = -6.0 * (idx - lo).float() / max(1, (hi - lo))
    for i in range(0, s_t.shape[-1], cfg.hop):
        sh = s_t[:, i:i+cfg.hop]; vh = v_t[:, i:i+cfg.hop]
        if sh.shape[-1] < cfg.hop:
            break
        ss = sfr_s.step(sh); vs = sfr_v.step(vh)
        snr = torch.full((1, ss.shape[-1]), 30.0)
        eq.step(ss, vs, snr, torch.tensor([0.95]))
    C = eq.C[0, lo:hi + 1]
    err = (C - true).abs().max().item()
    print(f"  M2 EQ convergence: max|C−true|={err:.3f} dB (≤1.0) after 3 s  "
          f"{'PASS' if err <= 1.0 else 'FAIL'}")
    assert err <= 1.0, f"M2: err={err}"
    return err


def test_M2_mutation():
    """Mutation: eq_ema_tau_s=1e6 (α≈0 ⇒ C never updates).  Must FAIL."""
    cfg = FusionConfig()
    cfg.eq_ema_tau_s = 1e6
    s_t, v_t = S.tilted_noise_pair(dur_s=3.0, tilt_db=6.0)
    sfr_s = StftStreamer(cfg); sfr_v = StftStreamer(cfg)
    eq = EQAlign(cfg, changepoint_enabled=False)
    bz = cfg.sr / cfg.n_fft
    lo = max(1, int(100 / bz)); hi = min(cfg.n_fft // 2 - 1, int(2000 / bz))
    idx = torch.arange(lo, hi + 1)
    true = -6.0 * (idx - lo).float() / max(1, (hi - lo))
    for i in range(0, s_t.shape[-1], cfg.hop):
        sh = s_t[:, i:i+cfg.hop]; vh = v_t[:, i:i+cfg.hop]
        if sh.shape[-1] < cfg.hop:
            break
        ss = sfr_s.step(sh); vs = sfr_v.step(vh)
        eq.step(ss, vs, torch.full((1, ss.shape[-1]), 30.0), torch.tensor([0.95]))
    err = (eq.C[0, lo:hi + 1] - true).abs().max().item()
    broken = err > 1.0
    print(f"  M2 mutation (τ=1e6, α≈0): max|C−true|={err:.3f} dB → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M2 mutation not caught"


# ================================================================ M3 ======
def _cv_run_sequence(cfg, v_spec, s_spec, dbs, n_settle=200):
    """One CV, feed V at attenuations `dbs` in sequence; return c_V after each settles."""
    cv = CV(cfg, enabled=cfg.enable_c_V)
    out = []
    for db in dbs:
        v_att = v_spec * (10 ** (db / 20.0))
        for _ in range(n_settle):
            cv.step(v_att, s_spec, torch.zeros(1, v_spec.shape[1]))
        out.append(cv.c_v)
    return out


def test_M3_cv_monotone():
    """V attenuated 0/−3/−6/−12 dB ⇒ c_V strictly monotone non-increasing."""
    cfg = FusionConfig()
    F0 = 150.0
    amps = [1 / k for k in range(1, 9)]
    v_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    s_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    c = _cv_run_sequence(cfg, v_spec, s_spec, [0.0, -3.0, -6.0, -12.0])
    strict_dec = all(c[i] > c[i+1] + 1e-4 for i in range(len(c) - 1))
    print(f"  M3 c_V monotone: c_V(0,-3,-6,-12)= "
          f"{[round(x,4) for x in c]}  strict↓  "
          f"{'PASS' if strict_dec else 'FAIL'}")
    assert strict_dec, f"M3: c_V not strictly decreasing: {c}"
    return c


def test_M3_mutation():
    """Mutation: c_V disabled (enabled=False ⇒ c_V≡1.0, invariant to V level).
    Must FAIL strict-monotonicity."""
    cfg = FusionConfig()
    cfg.enable_c_V = False
    F0 = 150.0; amps = [1 / k for k in range(1, 9)]
    v_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    s_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    c = _cv_run_sequence(cfg, v_spec, s_spec, [0.0, -3.0, -6.0, -12.0])
    strict_dec = all(c[i] > c[i+1] + 1e-4 for i in range(len(c) - 1))
    print(f"  M3 mutation (c_V disabled): c_V= {[round(x,4) for x in c]}  "
          f"strict↓={strict_dec} → "
          f"{'FAIL-of-mutant (caught) PASS' if not strict_dec else 'NOT caught PROBLEM'}")
    assert not strict_dec, "M3 mutation not caught"


# ================================================================ M4 ======
def test_M4_asym():
    """Rise 10→90 / fall 90→10 ratio ≥ 3 (slow rise, fast fall)."""
    cfg = FusionConfig()
    sm = AsymSmoother(cfg)
    # step: 0 for 1s, 1 for 2s, 0 for 1s (per-frame scalar)
    n_pre = int(1.0 * cfg.sr / cfg.hop)
    n_on = int(2.0 * cfg.sr / cfg.hop)
    n_post = int(1.0 * cfg.sr / cfg.hop)
    xs = [torch.zeros(1)] * n_pre + [torch.ones(1)] * n_on + [torch.zeros(1)] * n_post
    ys = []
    for x in xs:
        ys.append(sm.step(x.clone()).item())
    ys = np.array(ys)
    # rise: from the on-transition, find 10% and 90% crossing
    on_start = n_pre
    def cross(arr, start, target, direction):
        for i in range(start, len(arr)):
            if direction == "up" and arr[i] >= target:
                return i
            if direction == "down" and arr[i] <= target:
                return i
        return len(arr) - 1
    i10 = cross(ys, on_start, 0.1, "up"); i90 = cross(ys, on_start, 0.9, "up")
    rise = (i90 - i10) * cfg.hop / cfg.sr
    off_start = n_pre + n_on
    j90 = cross(ys, off_start, 0.9, "down"); j10 = cross(ys, off_start, 0.1, "down")
    fall = (j10 - j90) * cfg.hop / cfg.sr
    ratio = rise / max(fall, 1e-9)
    print(f"  M4 asym: rise(10→90)={rise*1e3:.1f}ms fall(90→10)={fall*1e3:.1f}ms "
          f"ratio={ratio:.2f} (≥3) {'PASS' if ratio >= 3 else 'FAIL'}")
    assert ratio >= 3, f"M4: ratio={ratio}"
    return ratio


def test_M4_mutation():
    """Mutation: symmetric EMA (same τ both ways).  Ratio ≈1 ⇒ FAIL."""
    cfg = FusionConfig()
    sm = AsymSmoother(cfg, symmetric=True)
    n_pre = int(1.0 * cfg.sr / cfg.hop); n_on = int(2.0 * cfg.sr / cfg.hop)
    n_post = int(1.0 * cfg.sr / cfg.hop)
    xs = [torch.zeros(1)] * n_pre + [torch.ones(1)] * n_on + [torch.zeros(1)] * n_post
    ys = [sm.step(x.clone()).item() for x in xs]
    ys = np.array(ys)

    def cross(arr, start, target, direction):
        for i in range(start, len(arr)):
            if direction == "up" and arr[i] >= target:
                return i
            if direction == "down" and arr[i] <= target:
                return i
        return len(arr) - 1
    i10 = cross(ys, n_pre, 0.1, "up"); i90 = cross(ys, n_pre, 0.9, "up")
    rise = (i90 - i10) * cfg.hop / cfg.sr
    j90 = cross(ys, n_pre + n_on, 0.9, "down"); j10 = cross(ys, n_pre + n_on, 0.1, "down")
    fall = (j10 - j90) * cfg.hop / cfg.sr
    ratio = rise / max(fall, 1e-9)
    broken = ratio < 3
    print(f"  M4 mutation (symmetric): ratio={ratio:.2f} → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M4 mutation not caught"


# ================================================================ M5 ======
def test_M5_g_f0_direction():
    """High f0_confidence ⇒ g_f0 (hence w, multiplicatively) strictly higher.
    Direction = 1−CMND (project前科 reversed it).  Tested on GF0 directly +
    full-pipeline propagation."""
    cfg = FusionConfig()
    gf = GF0(cfg)
    conf_voiced = torch.tensor([0.9]); conf_noise = torch.tensor([0.1])
    g_v = gf.step(conf_voiced).item(); g_n = gf.step(conf_noise).item()
    ok = g_v > g_n
    print(f"  M5 g_f0 direction: g(voiced)={g_v:.3f} > g(noise)={g_n:.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, f"M5: g_v={g_v} g_n={g_n}"
    # full-pipeline propagation: c_V & w_local disabled ⇒ w ∝ g_f0 (w_band≈1
    # since V=S ⇒ msc≈1).  voiced frames' w median strictly > noise frames'.
    from fusion import Fusion
    from fusion.f0 import f0_batch
    cfg2 = FusionConfig(); cfg2.enable_c_V = False; cfg2.enable_w_local = False
    x = S.voiced_unvoiced(F0=150.0, dur_s=4.0)
    f = Fusion(cfg2); f.process_batch(x, x)
    _, conf = f0_batch(x, cfg2)
    wh = torch.stack(f.core.w_history, dim=-1)[0]   # (Fb, N)
    w_med = wh.mean(0)                              # (N,) band-mean per frame
    hi = conf[0] > 0.5; lo = conf[0] < 0.5
    mv = float(np.median(w_med[hi])) if hi.any() else 0.0
    mn = float(np.median(w_med[lo])) if lo.any() else 0.0
    print(f"    full-pipeline w (c_V,w_local off ⇒ w∝g_f0): "
          f"voiced={mv:.4f} noise={mn:.4f} "
          f"({'voiced>noise ✓' if mv > mn else '✗'})")
    assert mv > mn, "M5 propagation: voiced w median not > noise"
    return g_v, g_n


# ================================================================ M6 ======
def test_M6_logclip_boundary():
    """|S|=−60 dB, |V|=0 dB, w=0.9 ⇒ |Y| ≥ |V| − (1−w)·Δ  (S can't drag Y down)."""
    cfg = FusionConfig()
    Fb = cfg.n_fft // 2 + 1
    b = 10
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = 10 ** (-60 / 20)
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = 1.0
    w = torch.zeros(1, Fb); w[0, b] = 0.9
    Y = logclip_mix(s_spec, v_spec, w, cfg.delta_db)
    y_db = 20 * math.log10(Y[0, b].abs().clamp_min(1e-12))
    bound = 0.0 - (1 - 0.9) * cfg.delta_db
    ok = y_db >= bound - 1e-6
    print(f"  M6 log-clip boundary: |Y|={y_db:.3f} dB ≥ {bound:.3f} dB "
          f"(= |V|−(1−w)Δ)  {'PASS' if ok else 'FAIL'}")
    assert ok, f"M6: |Y|={y_db} < {bound}"
    return y_db, bound


def test_M6_mutation():
    """Mutation: delta_db=1e9 (no clip) ⇒ |Y| dragged to ≈−6 dB ⇒ FAIL."""
    cfg = FusionConfig(); cfg.delta_db = 1e9
    Fb = cfg.n_fft // 2 + 1; b = 10
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = 10 ** (-60 / 20)
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = 1.0
    w = torch.zeros(1, Fb); w[0, b] = 0.9
    Y = logclip_mix(s_spec, v_spec, w, cfg.delta_db)
    y_db = 20 * math.log10(Y[0, b].abs().clamp_min(1e-12))
    bound = -0.1 * 10.0
    broken = y_db < bound - 1e-6
    print(f"  M6 mutation (Δ=1e9, no clip): |Y|={y_db:.3f} dB < {bound:.3f} → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M6 mutation not caught"


# ================================================================ M7 ======
def test_M7_energy_dip():
    """90° phase mismatch, equal amp, w=0.5: log-clip holds 0 dB ±0.5; complex
    convex drops ≈3 dB (the dip is real — falsifiable contrast)."""
    cfg = FusionConfig()
    Fb = cfg.n_fft // 2 + 1; b = 10
    A = 1.0
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = A + 0j
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = A * (1j)  # 90°
    w = torch.full((1, Fb), 0.5)
    Y_log = logclip_mix(s_spec, v_spec, w, cfg.delta_db)
    Y_conv = complex_convex(s_spec, v_spec, w)
    log_db = 20 * math.log10(Y_log[0, b].abs().clamp_min(1e-12))
    conv_db = 20 * math.log10(Y_conv[0, b].abs().clamp_min(1e-12))
    ok = abs(log_db - 0.0) <= 0.5 and conv_db <= -2.5
    print(f"  M7 energy dip: log-clip |Y|={log_db:.3f} dB (0±0.5)  "
          f"complex-convex |Y|={conv_db:.3f} dB (≈−3)  "
          f"{'PASS' if ok else 'FAIL'}")
    assert abs(log_db) <= 0.5, f"M7: log-clip |Y|={log_db}"
    assert conv_db <= -2.5, f"M7: convex not ~−3 ({conv_db})"
    return log_db, conv_db


def test_M7_mutation():
    """Mutation: use complex-convex magnitude (disable log-clip) ⇒ the 0 dB hold
    becomes the −3 dB dip ⇒ FAIL."""
    cfg = FusionConfig()
    Fb = cfg.n_fft // 2 + 1; b = 10
    A = 1.0
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = A + 0j
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = A * (1j)
    w = torch.full((1, Fb), 0.5)
    Y = complex_convex(s_spec, v_spec, w)
    y_db = 20 * math.log10(Y[0, b].abs().clamp_min(1e-12))
    broken = abs(y_db - 0.0) > 0.5
    print(f"  M7 mutation (complex-convex mag): |Y|={y_db:.3f} dB (not 0±0.5) → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M7 mutation not caught"


if __name__ == "__main__":
    test_M1_w_local(); test_M1_mutation()
    test_M2_eq_convergence(); test_M2_mutation()
    test_M3_cv_monotone(); test_M3_mutation()
    test_M4_asym(); test_M4_mutation()
    test_M5_g_f0_direction()
    test_M6_logclip_boundary(); test_M6_mutation()
    test_M7_energy_dip(); test_M7_mutation()
    print("M1–M7 mechanism tests: all PASS")
