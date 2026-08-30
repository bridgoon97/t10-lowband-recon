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
    """HISTORICAL (B0): per-harmonic w_local (RANSAC) — AC3 DELETED the
    per-harmonic family (B0.5: per-harm info can't transfer VPU→mic; ① maxes
    0.863 even at iso=100%).  w_local is now BAND-LEVEL (const-⑤ gate); its
    mechanism+effect are tested in test_t13_b1.  Conclusions retained in README."""
    from tests._testutil import SkipTest
    raise SkipTest("M1 per-harmonic w_local removed in AC3 (B1); see test_t13_b1 + README")
    cfg = FusionConfig()   # pragma: no cover (unreachable)
    wl = WLocal(cfg, v_fallback=False, valley=False)
    P_kill, P_surv = [], []
    # 3 formant-like envelopes whose WEAKEST 40% (3 of 8) land at k=2,5,8
    # (spacing 3 ⇒ isolated ⇒ ① local-median detects; tests the mechanism on
    # DETECTABLE kills — the realistic clustered+overlap case is R4/BR2).
    for trial, amps in enumerate([[0.5, 0.20, 0.8, 0.6, 0.15, 0.4, 0.3, 0.10],
                                   [0.4, 0.15, 0.7, 0.5, 0.10, 0.3, 0.6, 0.20],
                                   [0.6, 0.12, 0.5, 0.7, 0.18, 0.4, 0.3, 0.15]]):
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
    """HISTORICAL (B0): removed with M1 (AC3)."""
    from tests._testutil import SkipTest
    raise SkipTest("M1 mutation removed in AC3 (B1)")


# ================================================================ M2 ======
def test_M2_eq_convergence():
    """±6 dB tilt ⇒ C[f] within ±1 dB after 3 s (causal robust EMA; ADAPTIVE arm —
    frozen mode freezes after cold-start and wouldn't track a changing tilt)."""
    cfg = FusionConfig(); cfg.eq_mode = "adaptive"
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
    """Mutation: eq_ema_tau_s=1e6 (α≈0 ⇒ C never updates).  Must FAIL (adaptive)."""
    cfg = FusionConfig(); cfg.eq_mode = "adaptive"; cfg.eq_ema_tau_s = 1e6
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
    """One CV, feed V at attenuations `dbs` in sequence; return c_V after each settles.

    FR1/B0.5: V is attenuated in its SIGNAL (harmonic) content only — the device
    noise floor is held FIXED across attenuation levels (models the 'loose-fit /
    coupling-loss' scenario that is FR1's motivation, and is the only way an
    SNR-based c_V is sensitive to V-quality while invariant to recording gain).
    Criterion (strict monotone non-increasing) is UNCHANGED from the original M3."""
    cv = CV(cfg, enabled=cfg.enable_c_V)
    Fb = v_spec.shape[1]
    torch.manual_seed(12345)
    noise = (10 ** (cfg.cv_m3_noise_db / 20.0)) * torch.randn(1, Fb) * torch.exp(
        1j * 2 * math.pi * torch.rand(1, Fb))
    # harmonic-only signal (the bins well above the device-noise floor)
    mag = v_spec.abs()
    hb = mag[0] > 0.5 * mag[0].max()
    harm = torch.zeros(1, Fb, dtype=torch.complex64)
    harm[0, hb] = v_spec[0, hb]
    out = []
    for db in dbs:
        v_att = harm * (10 ** (db / 20.0)) + noise   # SIGNAL atten, device noise FIXED
        for _ in range(n_settle):
            cv.step(v_att, s_spec, torch.zeros(1, Fb))
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


# ===== FR1 (B0.5): c_V energy term = in-band SNR (directional + ratchet fix) ===
# Boundary: ALL T13 conclusions hold only for MALE speech (F0 87–124 Hz),
# normal volume — no female coverage in 0624/0625.  Not extrapolated beyond.

def _cv_synth_spec(cfg, sig_db=-20.0, noise_db=-45.0, F0=150.0, seed=0):
    """Build a (1, Fb) complex V spectrum: harmonics at sig_db, device noise at
    noise_db.  Levels chosen so both the SNR design and the abslevel mutation
    operate in the sigmoid's sensitive range (not saturated)."""
    Fb = cfg.n_fft // 2 + 1
    bz = cfg.sr / cfg.n_fft
    torch.manual_seed(seed)
    spec = torch.zeros(1, Fb, dtype=torch.complex64)
    sig = 10 ** (sig_db / 20.0); nf = 10 ** (noise_db / 20.0)
    for k in range(1, 9):
        b = int(round(k * F0 / bz))
        if 1 <= b < Fb:
            spec[0, b] = sig / k + 0j
    spec = spec + nf * torch.randn(1, Fb) * torch.exp(1j * 2 * math.pi * torch.rand(1, Fb))
    return spec


def _cv_settled(cfg, v_spec, s_spec, n=600):
    cv = CV(cfg, enabled=cfg.enable_c_V)
    for _ in range(n):
        cv.step(v_spec, s_spec, torch.zeros(1, v_spec.shape[1]))
    return cv.c_v


def test_FR1a_level_invariance():
    """FR1-a: S & V scaled TOGETHER (recording-gain) ⇒ c_V invariant (≤0.05).
    New SNR design: snr = e_db − nf, both shift by g ⇒ invariant.  (The
    CohTracker clamp was 1e-10 — too coarse for quiet bins, broke invariance;
    fixed to 1e-20.)"""
    cfg = FusionConfig()
    v0 = _cv_synth_spec(cfg, seed=1)
    s0 = _cv_synth_spec(cfg, seed=2)
    cs = [_cv_settled(cfg, v0 * (10 ** (db / 20.0)), s0 * (10 ** (db / 20.0)))
          for db in [0, -6, -12, -20]]
    spread = max(cs) - min(cs)
    print(f"  FR1-a c_V(0,-6,-12,-20)={[round(x,4) for x in cs]} spread={spread:.4f} "
          f"(≤0.05) → {'PASS' if spread < 0.05 else 'FAIL'}")
    assert spread < 0.05, f"FR1-a: c_V not invariant to joint scaling (spread {spread})"


def test_FR1a_mutation():
    """Mutation: cv_legacy_abslevel=True (pure absolute level, no SNR) ⇒ c_V
    level-dependent ⇒ FR1-a FAILS (spread > 0.05).  Breaks ONLY FR1-a
    (keeps FR1-b: level drops⇒c_V drops; FR1-c: no ratchet)."""
    cfg = FusionConfig(); cfg.cv_legacy_abslevel = True
    v0 = _cv_synth_spec(cfg, seed=1)
    s0 = _cv_synth_spec(cfg, seed=2)
    cs = [_cv_settled(cfg, v0 * (10 ** (db / 20.0)), s0 * (10 ** (db / 20.0)))
          for db in [0, -6, -12]]
    spread = max(cs) - min(cs)
    print(f"  FR1-a mutation (abslevel): c_V={[round(x,4) for x in cs]} spread={spread:.4f} "
          f"(>0.05) → {'FAIL-of-mutant (caught) PASS' if spread > 0.05 else 'NOT caught'}")
    assert spread > 0.05, f"FR1-a mutation: abslevel not level-dependent (spread {spread})"


def test_FR1c_ratchet_recovery():
    """FR1-c: +12 dB loud segment then back to normal ⇒ c_V recovers to within
    0.05 of a no-loud control within ≤2 s.  New SNR design has no running-MAX
    ⇒ no permanent depression."""
    cfg = FusionConfig()
    v0 = _cv_synth_spec(cfg, seed=1)
    s0 = _cv_synth_spec(cfg, seed=2)
    n_set = 600; n_loud = 100; hop_s = cfg.hop / cfg.sr
    # control: never sees the loud segment
    cv_c = CV(cfg, enabled=True)
    for _ in range(n_set): cv_c.step(v0, s0, torch.zeros(1, v0.shape[1]))
    # test: settle, +12 dB loud, then recover
    cv = CV(cfg, enabled=True)
    for _ in range(n_set): cv.step(v0, s0, torch.zeros(1, v0.shape[1]))
    for _ in range(n_loud): cv.step(v0 * 4.0, s0 * 4.0, torch.zeros(1, v0.shape[1]))
    diffs = []
    for n in [0, 100, 200]:   # 0, ~1 s, ~2 s post-loud (hop=10 ms)
        for _ in range(n):
            cv.step(v0, s0, torch.zeros(1, v0.shape[1]))
            cv_c.step(v0, s0, torch.zeros(1, v0.shape[1]))
        diffs.append((n * hop_s, abs(cv.c_v - cv_c.c_v)))
    rec2 = diffs[-1][1]
    print(f"  FR1-c ratchet recovery: post-loud |c_V − control| @ "
          f"{[f'{t:.1f}s={d:.4f}' for t, d in diffs]}  ≤0.05@2s? {rec2 < 0.05}")
    assert rec2 < 0.05, f"FR1-c: c_V did not recover within 2 s (diff {rec2})"


def test_FR1c_mutation():
    """Mutation: cv_legacy_ratchet=True (running-MAX e_term) ⇒ one loud event
    raises e_max permanently ⇒ c_V depressed ⇒ FR1-c FAILS (no recovery)."""
    cfg = FusionConfig(); cfg.cv_legacy_ratchet = True
    v0 = _cv_synth_spec(cfg, seed=1)
    s0 = _cv_synth_spec(cfg, seed=2)
    n_set = 600; n_loud = 100
    cv_c = CV(cfg, enabled=True)
    for _ in range(n_set): cv_c.step(v0, s0, torch.zeros(1, v0.shape[1]))
    cv = CV(cfg, enabled=True)
    for _ in range(n_set): cv.step(v0, s0, torch.zeros(1, v0.shape[1]))
    for _ in range(n_loud): cv.step(v0 * 31.6, s0 * 31.6, torch.zeros(1, v0.shape[1]))   # +30 dB loud (ratchet e_max)
    for _ in range(200):
        cv.step(v0, s0, torch.zeros(1, v0.shape[1]))
        cv_c.step(v0, s0, torch.zeros(1, v0.shape[1]))
    d = abs(cv.c_v - cv_c.c_v)
    print(f"  FR1-c mutation (ratchet): post-recover |c_V−control|={d:.4f} (>0.05) "
          f"→ {'FAIL-of-mutant (caught) PASS' if d > 0.05 else 'NOT caught'}")
    assert d > 0.05, f"FR1-c mutation: ratchet did not depress c_V (diff {d})"



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


def test_M5_mutation():
    """R1 mutation: flip GF0 direction to use CMND (=1−conf) instead of 1−CMND
    (the project前科 direction).  BOTH M5 assertions must now FAIL:
      • direct  g(voiced) > g(noise)   → reversed
      • full-pipeline  mv > mn          → reversed
    Line changed: GF0.step — ``c = f0_conf if not self.flip else (1.0 - f0_conf)``
    with ``flip=True`` ⇒ uses CMND (low ⟹ voiced)."""
    cfg = FusionConfig()
    gf = GF0(cfg, flip=True)                       # the one-line mutation
    g_v = gf.step(torch.tensor([0.9])).item()      # voiced conf
    g_n = gf.step(torch.tensor([0.1])).item()      # noise conf
    direct_fail = not (g_v > g_n)                  # original assertion fails
    print(f"  M5 mutation (GF0 flip→CMND): g(voiced)={g_v:.3f} g(noise)={g_n:.3f} "
          f"direct 'g_v>g_n' → {'FAIL' if direct_fail else 'pass'}")
    # full-pipeline: inject the flipped GF0 into the core
    from fusion import Fusion
    from fusion.f0 import f0_batch
    cfg2 = FusionConfig(); cfg2.enable_c_V = False; cfg2.enable_w_local = False
    x = S.voiced_unvoiced(F0=150.0, dur_s=4.0)
    f = Fusion(cfg2)
    f.core.gf0.flip = True                          # mutation applied to prod GF0
    f.process_batch(x, x)
    _, conf = f0_batch(x, cfg2)
    wh = torch.stack(f.core.w_history, dim=-1)[0].mean(0)
    hi = conf[0] > 0.5; lo = conf[0] < 0.5
    mv = float(np.median(wh[hi])) if hi.any() else 0.0
    mn = float(np.median(wh[lo])) if lo.any() else 0.0
    prop_fail = not (mv > mn)
    print(f"    full-pipeline w: voiced={mv:.4f} noise={mn:.4f} "
          f"'mv>mn' → {'FAIL' if prop_fail else 'pass'}")
    assert direct_fail and prop_fail, ("M5 mutation not caught: flipping g_f0 "
        "direction did NOT fail both assertions")


# ================================================================ M6 ======
def test_M6_logclip_boundary():
    """HR1 new formula: w=1, |S|=0, |V|=+40 ⇒ corr=clip(40, −Δ_down, +Δ_up)=+Δ_up
    ⇒ |Y|=Δ_up (bounded UP by Δ_up — V can boost S but only by Δ_up)."""
    cfg = FusionConfig()
    Fb = cfg.n_fft // 2 + 1; b = 10
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = 1.0           # |S|=0 dB
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = 10 ** (40 / 20)  # |V|=+40 dB
    w = torch.ones(1, Fb)
    Y = logclip_mix(s_spec, v_spec, w, cfg.delta_up_db, cfg.delta_down_db)
    y_db = 20 * math.log10(Y[0, b].abs().clamp_min(1e-12))
    ok = abs(y_db - cfg.delta_up_db) < 0.5
    print(f"  M6 asymmetric-clip up-bound: |Y|={y_db:.3f} dB ≈ Δ_up={cfg.delta_up_db} "
          f"(V boosts S by ≤Δ_up)  {'PASS' if ok else 'FAIL'}")
    assert ok, f"M6: |Y|={y_db} ≠ Δ_up={cfg.delta_up_db}"
    return y_db


def test_M6_mutation():
    """Mutation: delta_up_db=1e9 (no up-clip) ⇒ |Y| unbounded ⇒ 40 dB (not Δ_up) ⇒ FAIL."""
    cfg = FusionConfig(); cfg.delta_up_db = 1e9
    Fb = cfg.n_fft // 2 + 1; b = 10
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = 1.0
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = 10 ** (40 / 20)
    w = torch.ones(1, Fb)
    Y = logclip_mix(s_spec, v_spec, w, cfg.delta_up_db, cfg.delta_down_db)
    y_db = 20 * math.log10(Y[0, b].abs().clamp_min(1e-12))
    broken = y_db > 30.0   # unbounded ⇒ |Y|=40 ≫ default Δ_up=25 (bound removed)
    print(f"  M6 mutation (Δ_up=1e9, no up-clip): |Y|={y_db:.3f} dB (≈40, unbounded) → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught PROBLEM'}")
    assert broken, "M6 mutation not caught"


# ================================================================ M7 ======
def test_M7_energy_dip():
    """HISTORICAL (B0): the complex-convex contrast arm was REMOVED in AC1
    (B1) — its ~−3 dB energy dip at 90° phase mismatch was a complex-vector-
    cancellation artifact, IMPOSSIBLE in magnitude-only fusion (phase from S).
    Test retained as a historical record of the removed candidate; SKIPs.
    AC1's smearing cost is priced by G7."""
    from tests._testutil import SkipTest
    raise SkipTest("M7 complex-convex arm removed in AC1 (B1); see G7 phase pricing")


def test_M7_mutation():
    """HISTORICAL (B0): removed with M7 (AC1)."""
    from tests._testutil import SkipTest
    raise SkipTest("M7 mutation removed in AC1 (B1)")


if __name__ == "__main__":
    test_M1_w_local(); test_M1_mutation()
    test_M2_eq_convergence(); test_M2_mutation()
    test_M3_cv_monotone(); test_M3_mutation()
    test_M4_asym(); test_M4_mutation()
    test_M5_g_f0_direction()
    test_M5_mutation()
    test_M6_logclip_boundary(); test_M6_mutation()
    test_M7_energy_dip(); test_M7_mutation()
    print("M1–M7 mechanism tests: all PASS")
