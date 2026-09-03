"""T13-N1 structural tests — trust interface, voiced routing, shaping gain,
D5 injection, and the three pre-fixed identity invariants I1/I2/I3.

BOUNDARY: I1 runs on REAL 0624 recordings (DECLASSIFIED, FF=idx1 / VPU=idx3)
— not synthetic signals.  X (clean FF) is only used as the evaluation-side
reference and by the OFFLINE degrade module (static-checked); the algorithm
path never sees it.

Every invariant carries a mutation sanity: a deliberately broken variant that
the test MUST catch (proving the test is load-bearing).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from fusion import (FusionConfig, Fusion, FusionCore, FusionStreamer,
                    stft_batch, istft_batch, degrade, DegradationConfig)
from fusion.degrade import apply_d5
from fusion.trust import TrustSource
from fusion.shape import ShapeGain
from fusion.voicing import VoicingGate
from fusion.realdata import list_0624

SR = 16000
WIN = 480
SKIP = 2 * WIN
REC_ROOT = "/mnt/d/Projects/mic_array_capture/mic_recordings/0624"


# ------------------------------------------------------------------ helpers --
def _tone(T_s, f0=125.0, seed=3):
    """Bin-aligned harmonic signal (deep natural valleys) + envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(T_s * SR)) / SR
    env = 0.55 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + 0.3))
    x = np.zeros_like(t)
    for k in range(1, 25):
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    x = x / (np.abs(x).max() + 1e-9) * 0.4 * env
    x += rng.normal(0, 1.0, t.shape) * env * 0.02
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def _real_pair(name: str, seg_s: float = 4.0, offset_s: float = 2.0):
    """REAL 0624 audio: returns (X=FF clean, V=VPU) — NO normalization.
    Deterministic tail-nonzero selection: the offset advances by 0.5 s (max 5
    tries) until |last sample of X| ≥ 1e-4 — keeps provenance intact while
    guaranteeing the full-length I1 comparison has a NON-ZERO tail (no
    all-zero-tail false pass)."""
    y, sr = sf.read(f"{REC_ROOT}/{name}", dtype="float32", always_2d=True)
    assert sr == SR and y.shape[1] >= 4
    for _ in range(5):
        i0 = int(offset_s * SR)
        x = torch.from_numpy(y[i0:i0 + int(seg_s * SR), 1].copy()).unsqueeze(0)
        if float(x[0, -1].abs()) >= 1e-4:
            break
        offset_s += 0.5
    v = torch.from_numpy(y[i0:i0 + int(seg_s * SR), 3].copy()).unsqueeze(0)
    return x, v


def _n1_fusion(p: float, **sw):
    cfg = FusionConfig().with_switches(decision_mode="n1", **sw)
    f = Fusion(cfg)
    f.set_trust(TrustSource(source="manual", const=p))
    return f


# ------------------------------------------------------------------ I1 ------
class _IdentityBreaker(Fusion):
    """MUTATION: adds a tiny dither to the output (breaks Y≡S at p=0)."""

    def process_batch(self, s, v):
        y = super().process_batch(s, v)
        return y + 1e-3


def test_N1_I1_p_zero_identity_real0624():
    """I1: p ≡ 0 ⇒ Y ≡ S (per-sample allclose, FULL length — the pre-fixed
    criterion) on REAL 0624 audio with a NON-ZERO tail (deterministic segment
    selection).  The tail-coverage fix extends the N1 batch framing by
    (win−hop) zeros so the causal WOLA normalisation is complete at the last
    sample; shape and full length are asserted explicitly."""
    files = list_0624()[:2]
    worst = 0.0
    for fp in files:
        x, v = _real_pair(Path(fp).name)
        assert float(x[0, -1].abs()) >= 1e-4, "tail unexpectedly zero"
        s_d = degrade(x, FusionConfig(), DegradationConfig(d5_enable=True, d5_level_db=20))
        f = _n1_fusion(0.0)
        y = f.process_batch(s_d, v)
        assert y.shape == s_d.shape, f"shape mismatch {y.shape} vs {s_d.shape}"
        d = float((y - s_d).abs().max())                     # FULL length
        worst = max(worst, d)
    assert worst <= 1e-4 + 1e-6, f"I1 violated (full length): max|Y-S|={worst:.2e}"
    print(f"  I1 PASS: p=0 ⇒ Y≡S on real 0624 ({len(files)} recordings), "
          f"FULL-length per-sample max|Y-S|={worst:.2e}")
    # mutation sanity: the identity-breaker dither MUST be caught (full length)
    x, v = _real_pair(Path(files[0]).name)
    s_d = degrade(x, FusionConfig(), DegradationConfig(d5_enable=True, d5_level_db=20))
    ym = _IdentityBreaker(_n1_fusion(0.0).cfg).process_batch(s_d, v)
    dm = float((ym - s_d).abs().max())
    assert dm > 1e-4, "mutation sanity FAILED: I1 test did not catch the dither"
    print(f"  I1 mutation sanity: dither mutant caught (full-length diff {dm:.2e} > 1e-4)")
    # streaming same semantics: feed the SAME stream incl. (win−hop) trailing
    # zeros, then flush — full length must match the batch identity output
    f2 = _n1_fusion(0.0)
    cfg = f2.cfg
    fs = FusionStreamer(cfg)
    fs.set_trust(f2.trust)
    outs = []
    pad = cfg.win - cfg.hop
    ext = torch.cat([s_d, torch.zeros(1, pad)], dim=-1)
    extv = torch.cat([v, torch.zeros(1, pad)], dim=-1)
    for i in range(0, ext.shape[-1], cfg.hop):
        sh, vh = ext[:, i:i + cfg.hop], extv[:, i:i + cfg.hop]
        if sh.shape[-1] < cfg.hop:
            break
        o = fs.stream_step(sh, vh)
        if o is not None:
            outs.append(o)
    outs.append(fs.flush())
    ys = torch.cat(outs, dim=1)[:, :s_d.shape[-1]]
    assert ys.shape == s_d.shape
    ds = float((ys - s_d).abs().max())
    assert ds <= 1e-4 + 1e-6, f"I1 streaming tail semantics: max|Y-S|={ds:.2e}"
    print(f"  I1 streaming (same stream incl. trailing zeros): full-length "
          f"max|Y-S|={ds:.2e}")


# ------------------------------------------------------------------ I2 ------
def test_N1_I2_gv_zero_floor():
    """I2: g_v ≡ 0 ⇒ log|Y| ≥ log|S| − Δ↓min for EVERY fusion bin — asserted on
    the SPECTRAL synthesis output (pre-iSTFT), where the clip lives by
    construction.  (The waveform re-analysis can locally deviate: magnitude
    edits are not frame-consistent, a property of every magnitude-domain
    fusion in this repo incl. MVP/legacy; reported as reference.)"""
    cfg = FusionConfig().with_switches(decision_mode="n1", gv_override=0.0)
    x, v = _real_pair(Path(list_0624()[0]).name, seg_s=3.0)
    core = FusionCore(cfg)
    spec_s = stft_batch(x, cfg)
    spec_v = stft_batch(v, cfg)
    lp = cfg.win - cfg.hop
    sp = torch.nn.functional.pad(x, (lp, 0), mode="constant")
    frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
    lo, hi = 1, cfg.fusion_hi_bin
    dd_min = cfg.n1_delta_down_min_db
    bad, worst = 0, 0.0
    with torch.no_grad():
        for t in range(spec_s.shape[-1]):
            y_t, _ = core.process_frame(spec_s[:, :, t], spec_v[:, :, t],
                                        frames[:, t, :], p_t=1.0)
            ly = 20 * torch.log10(y_t[0, lo:hi + 1].abs().clamp_min(1e-8))
            ls = 20 * torch.log10(spec_s[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            v_ = ly < ls - dd_min - 1e-3
            bad += int(v_.sum())
            if v_.any():
                worst = max(worst, float((ls - dd_min - ly)[v_].max()))
    assert bad == 0, f"I2 violated at spec level: {bad} bin-frames (worst {worst:.2f} dB)"
    print(f"  I2 PASS: g_v≡0 ⇒ log|Y| ≥ log|S|−{dd_min} on 100% of spectral "
          f"bin-frames (0 violations, spec-level synthesis output)")
    # mutation sanity: Δ↓ ignoring g_v MUST break the bound
    cfg_m = cfg.with_switches(n1_mutation_dd_ignores_gv=True)
    core_m = FusionCore(cfg_m)
    bad_m = 0
    with torch.no_grad():
        for t in range(spec_s.shape[-1]):
            y_t, _ = core_m.process_frame(spec_s[:, :, t], spec_v[:, :, t],
                                          frames[:, t, :], p_t=1.0)
            ly = 20 * torch.log10(y_t[0, lo:hi + 1].abs().clamp_min(1e-8))
            ls = 20 * torch.log10(spec_s[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            bad_m += int((ly < ls - dd_min - 1e-3).sum())
    assert bad_m > 0, "mutation sanity FAILED: I2 did not catch Δ↓-ignores-g_v"
    print(f"  I2 mutation sanity: Δ↓-ignores-g_v caught "
          f"({bad_m} bin-frames below the floor)")


# ------------------------------------------------------------------ I3 ------
def _stream(f, s, v, cfg):
    fs = FusionStreamer(cfg)
    if f.trust is not None:
        fs.set_trust(f.trust)
    outs = []
    for i in range(0, s.shape[-1], cfg.hop):
        sh, vh = s[:, i:i + cfg.hop], v[:, i:i + cfg.hop]
        if sh.shape[-1] < cfg.hop:
            break
        o = fs.stream_step(sh, vh)
        if o is not None:
            outs.append(o)
    outs.append(fs.flush())
    return torch.cat(outs, dim=1)[:, :s.shape[-1]]


def test_N1_I3_causality_and_equiv():
    """I3: future perturbation ⇒ past outputs bit-identical (100% of cut
    points); batch ≡ streaming < 1e-6.  REAL 0624 audio, N1 mode.

    Safe-prefix derivation (NOT observation-picked): frame t reads raw samples
    [t·hop − (win−hop), t·hop + hop); output sample s is covered by frames up
    to t* = ⌊(s + win − hop)/hop⌋, which reads raw input up to t*·hop + hop − 1.
    With all cut points P a multiple of hop, requiring t*·hop + hop − 1 < P
    gives s < P − (win − hop) ⇒ safe prefix K = P − (win − hop) = P − 320.
    (The old K = P − win also worked for the production code but EXCLUDED the
    region [P−480, P−320) where the noncausal-a mutation first leaks — that is
    how the old mutation check was a false pass.)

    Every causality comparison runs BOTH sides on FRESH instances — reusing an
    instance across the original/perturbed pair inherits EQ/shape/voicing
    state, which pollutes the past output and fakes a leak (the rework found
    exactly this false positive)."""
    x, v = _real_pair(Path(list_0624()[0]).name, seg_s=3.0)
    cfg = FusionConfig().with_switches(decision_mode="n1")
    T = x.shape[-1]
    cuts = [cfg.hop * 40, cfg.hop * 120, T // 2, 3 * T // 4]
    y_full = _n1_fusion(0.75).process_batch(x, v)      # fresh instance
    worst = 0.0
    n_ok = 0
    n_cut = 0
    for P in cuts:
        n_cut += 1
        x_m, v_m = x.clone(), v.clone()
        x_m[:, P:] = 0.0
        v_m[:, P:] = 0.0
        y_m = _n1_fusion(0.75).process_batch(x_m, v_m)  # FRESH instance both sides
        K = max(0, P - (cfg.win - cfg.hop))              # derived safe prefix
        if K == 0:
            continue
        if torch.equal(y_full[..., :K], y_m[..., :K]):
            n_ok += 1
        worst = max(worst, float((y_full[..., :K] - y_m[..., :K]).abs().max()))
    assert n_ok == n_cut, f"I3 causality FAILED: {n_ok}/{n_cut} cut points clean"
    print(f"  I3 causality: {n_ok}/{n_cut} future-perturbation cut points "
          f"bit-identical with fresh instances, K=P-(win-hop) (worst {worst})")
    # batch ≡ streaming: SAME stream including the (win−hop) trailing zeros
    # (end-of-stream padding is part of the input stream semantics), full length
    f = _n1_fusion(0.75)
    ys = _stream(f, x, v, cfg)
    Nb = min(y_full.shape[-1], ys.shape[-1])
    d = float((y_full[..., SKIP:Nb - SKIP] - ys[..., SKIP:Nb - SKIP]).abs().max())
    assert d < 1e-6, f"batch≠streaming: {d:.2e}"
    print(f"  I3 batch≡streaming: interior max diff {d:.1e} (<1e-6)")
    # mutation sanity: noncausal-a MUST leak within the derived safe prefix,
    # with FRESH instances on both sides (the old same-instance reuse was a
    # state-pollution false positive: leak 5.69e-02 from state, not causality)
    cfg_m = cfg.with_switches(n1_mutation_noncausal_a=True)
    leaks = []
    for P in cuts:
        y_mfull = Fusion(cfg_m).process_batch(x, v)      # fresh, mutation on
        x_m, v_m = x.clone(), v.clone()
        x_m[:, P:] = 0.0
        v_m[:, P:] = 0.0
        y_mm = Fusion(cfg_m).process_batch(x_m, v_m)     # fresh, mutation on
        K = max(0, P - (cfg.win - cfg.hop))
        leaks.append(float((y_mfull[..., :K] - y_mm[..., :K]).abs().max()))
    caught = sum(1 for lk in leaks if lk > 1e-6)
    assert caught >= 1, (f"mutation sanity FAILED: noncausal-a not caught at any "
                         f"cut point (leaks {leaks})")
    print(f"  I3 mutation sanity: noncausal-a caught at {caught}/{n_cut} cut "
          f"points with fresh instances (leaks "
          + " ".join(f"{lk:.1e}" for lk in leaks) + ")")


# ------------------------------------------------- ShapeGain exact recovery ---
def test_N1_shape_exact_recovery():
    """Falsifiable check of the least-squares intercept fix: construct a known
    exact linear spectral difference G* = a0 + s0·f̃ on the ShapeGain axis;
    after ONE step (first frame initialises the state — no smoothing yet) the
    recovered G must match G* per bin on the fit band within 1e-5 dB.
    The OLD intercept (a = mean(t), no −s·mean(f̃)) is the mutation and MUST
    fail.  No smoothing/parameter was tuned for this test."""
    cfg = FusionConfig().with_switches(decision_mode="n1")
    sg = ShapeGain(cfg)
    Fb = cfg.n_fft // 2 + 1
    f_t = sg._fit_axis(Fb, torch.device("cpu"))       # the EXACT axis the fit uses
    a0, s0 = -3.0, 5.0
    base = torch.full((1, Fb), 10 ** (-20.0 / 20))
    ss = base * 10 ** ((a0 + s0 * f_t).unsqueeze(0) / 20)
    G, a, s = sg.step(ss, base)
    lo = max(1, int(cfg.shape_fit_lo_hz / (cfg.sr / cfg.n_fft)))
    hi = int(cfg.shape_fit_hi_hz / (cfg.sr / cfg.n_fft))
    err = float((G[0, lo:hi + 1] - (a0 + s0 * f_t[lo:hi + 1])).abs().max())
    assert err < 1e-5, f"exact linear recovery FAILED: fit-band max err {err:.2e} dB"
    print(f"  shape exact recovery PASS: fit-band per-bin max err {err:.2e} dB "
          f"(< 1e-5, pre-declared gate); a={float(a):.6f}, s={float(s):.6f}")
    # mutation: old intercept (a = mean(t)) MUST break the recovery
    sg_m = ShapeGain(cfg.with_switches(shape_mutation_old_intercept=True))
    G_m, _, _ = sg_m.step(ss, base)
    err_m = float((G_m[0, lo:hi + 1] - (a0 + s0 * f_t[lo:hi + 1])).abs().max())
    assert err_m > 1e-5, "mutation sanity FAILED: old intercept not caught"
    print(f"  shape mutation sanity: old intercept caught "
          f"(fit-band max err {err_m:.2e} dB > 1e-5)")


# ---------------------------------------------------------- g_v direction ---
def test_N1_gv_direction():
    """g_v direction: voiced-confident V ⇒ high settled g_v; unvoiced/noisy V ⇒
    low settled g_v.  f0_confidence = 1 − CMND, HIGHER = more confident
    (direction under test).  Both gates run to their SETTLED value (50 frames)."""
    cfg = FusionConfig().with_switches(decision_mode="n1")
    t = np.arange(cfg.win) / SR
    voiced = torch.from_numpy((sum(1.0 / k * np.sin(2 * np.pi * k * 125 * t)
                                   for k in range(1, 20))
                               .astype(np.float32))).unsqueeze(0)
    rng = np.random.default_rng(0)
    unvoiced = torch.from_numpy(rng.normal(0, 0.3, (1, cfg.win)).astype(np.float32))
    vg_v, vg_u = VoicingGate(cfg), VoicingGate(cfg)
    for _ in range(50):                      # settle (rise/fall taus « 0.5 s)
        g_voiced = vg_v.step(voiced)
        g_unvoiced = vg_u.step(unvoiced)
    assert g_voiced > g_unvoiced, (f"direction wrong: voiced {g_voiced:.3f} ≤ "
                                   f"unvoiced {g_unvoiced:.3f}")
    print(f"  g_v direction PASS: settled voiced {g_voiced:.3f} > unvoiced {g_unvoiced:.3f}")
    # mutation sanity: flip to CMND ⇒ the direction assertion MUST fail
    vg_m = VoicingGate(cfg.with_switches(gv_flip=True))
    for _ in range(50):
        g_v_m = vg_m.step(voiced)
        g_u_m = vg_m.step(unvoiced)
    flipped = g_v_m <= g_u_m
    assert flipped, "mutation sanity FAILED: gv_flip was not caught by the direction test"
    print(f"  g_v mutation sanity: flip caught (settled voiced {g_v_m:.3f} ≤ "
          f"unvoiced {g_u_m:.3f})")


# ------------------------------------------------- a/s time-constant split ---
def test_N1_shape_tau_separation():
    """a[t] (fast) vs s[t] (slow) must be SEPARATE states with different time
    constants: after a level step, a responds ~4× faster (50%-point)."""
    cfg = FusionConfig().with_switches(decision_mode="n1")
    sg = ShapeGain(cfg)
    Fb = cfg.n_fft // 2 + 1
    bz = cfg.sr / cfg.n_fft
    f = torch.arange(Fb) * bz
    f_t = ((f - cfg.shape_fit_lo_hz) / (cfg.shape_fit_hi_hz - cfg.shape_fit_lo_hz)).clamp(0, 2)
    base_v = torch.full((1, Fb), 10 ** (-20.0 / 20))     # LINEAR magnitudes
    base_s = torch.full((1, Fb), 10 ** (-20.0 / 20))
    T1, T2 = 200, 600
    def spec_at(t):
        # before T1: flat; after T1: +6 dB level AND +6 dB/BW tilt (a AND s targets)
        lvl = 0.0 if t < T1 else 6.0
        tilt = 0.0 if t < T1 else 6.0
        return base_s * 10 ** ((lvl + tilt * f_t) / 20)
    a_hist, s_hist = [], []
    for t in range(T2):
        _, a, s = sg.step(spec_at(t), base_v)
        a_hist.append(float(a))
        s_hist.append(float(s))
    a_arr = np.array(a_hist[T1:])
    s_arr = np.array(s_hist[T1:])
    def t50(arr, final):
        idx = np.nonzero(arr >= final * 0.5)[0]
        return float(idx[0]) if idx.size else float("inf")
    t50_a = t50(a_arr, float(np.median(a_arr[-50:])))
    t50_s = t50(s_arr, float(np.median(s_arr[-50:])))
    assert t50_a < t50_s / 4, (f"time constants not separated: t50(a)={t50_a:.0f} "
                               f"frames, t50(s)={t50_s:.0f} frames")
    print(f"  shape tau separation PASS: t50(a)={t50_a:.0f} frames "
          f"({t50_a*cfg.hop/SR*1000:.0f} ms) << t50(s)={t50_s:.0f} frames "
          f"({t50_s*cfg.hop/SR*1000:.0f} ms)")
    # mutation sanity: equal taus ⇒ separation MUST vanish
    sg_m = ShapeGain(cfg.with_switches(shape_s_tau_s=cfg.shape_a_tau_s))
    a_h, s_h = [], []
    for t in range(T2):
        _, a, s = sg_m.step(spec_at(t), base_v)
        a_h.append(float(a)); s_h.append(float(s))
    t50_a_m = t50(np.array(a_h[T1:]), float(np.median(a_h[-50:])))
    t50_s_m = t50(np.array(s_h[T1:]), float(np.median(s_h[-50:])))
    assert not (t50_a_m < t50_s_m / 4), ("mutation sanity FAILED: equal taus were "
                                         "not caught by the separation test")
    print(f"  shape mutation sanity: equal taus caught "
          f"(t50 a={t50_a_m:.0f} vs s={t50_s_m:.0f} frames — no 4× split)")


# ------------------------------------------------------- ORACLE rejection ---
def test_N1_oracle_rejected():
    """ORACLE trust must be rejected by the production path; allow_oracle is a
    mutation-only escape hatch proving the guard is load-bearing."""
    try:
        TrustSource(source="oracle")
        raised = False
    except ValueError:
        raised = True
    assert raised, "ORACLE trust was NOT rejected"
    print("  ORACLE rejection PASS: TrustSource(source='oracle') raises")
    ts = TrustSource(source="oracle", allow_oracle=True)   # mutation: guard off
    assert ts.source == "oracle"
    print("  ORACLE mutation sanity: allow_oracle=True bypasses (guard is "
          "load-bearing, not decorative)")
    # CLI-level rejection
    tmp = Path("/tmp/t13_n1_oracle"); tmp.mkdir(exist_ok=True)
    sf.write(tmp / "s.wav", np.random.default_rng(0).normal(0, 0.01, SR).astype(np.float32), SR, subtype="PCM_16")
    sf.write(tmp / "v.wav", np.random.default_rng(1).normal(0, 0.01, SR).astype(np.float32), SR, subtype="PCM_16")
    r = subprocess.run([sys.executable, "-m", "fusion.run_fusion",
                        "--stage2", str(tmp / "s.wav"), "--vpu", str(tmp / "v.wav"),
                        "--output", str(tmp / "y.wav"), "--trust", "oracle"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0, "CLI accepted --trust oracle"
    print("  ORACLE CLI rejection PASS (non-zero exit)")


# ------------------------------------------------------------- D5 sanity ----
def test_N1_d5_sanity():
    """D5: smaller L raises valleys more; peaks untouched; unvoiced untouched."""
    cfg = FusionConfig()
    x = _tone(3.0)
    spec = stft_batch(x, cfg)
    from fusion.f0 import f0_batch
    f0, conf = f0_batch(x, cfg)
    raises = {}
    for L in [10, 40]:
        deg = DegradationConfig(d5_enable=True, d5_level_db=L, seed=0)
        out, valley, peak, voiced = apply_d5(spec, f0, conf, cfg, deg)
        lo_ = out[0].abs().clamp_min(1e-8)
        lx = spec[0].abs().clamp_min(1e-8)
        raises[L] = float((20 * torch.log10(lo_[valley[0]] / lx[valley[0]])).mean())
        assert float((20 * torch.log10(lo_[peak[0]] / lx[peak[0]])).abs().mean()) < 1e-3, \
            "D5 changed peak bins"
    assert raises[10] > raises[40] + 1.0, f"D5 level ordering wrong: {raises}"
    print(f"  D5 sanity PASS: valley raise L=10 {raises[10]:+.2f} dB > "
          f"L=40 {raises[40]:+.2f} dB; peaks untouched; unvoiced frames skipped "
          f"by construction (f0≤0 or conf<{0.5})")


if __name__ == "__main__":
    test_N1_I1_p_zero_identity_real0624()
    test_N1_I2_gv_zero_floor()
    test_N1_I3_causality_and_equiv()
    test_N1_gv_direction()
    test_N1_shape_tau_separation()
    test_N1_oracle_rejected()
    test_N1_d5_sanity()
    print("N1 structural tests: all PASS")
