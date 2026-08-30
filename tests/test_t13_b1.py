"""T13-B1 — fusion effect verification + tuning (AC1/AC2/AC3 in place).

Architecture (AC1/AC2/AC3): magnitude-only fusion (∠Y=∠S), frozen EQ,
band-level w_local (const-⑤ gate; per-harmonic ①②③④⑤ deleted).

🔴 BOUNDARY: ALL conclusions hold only for MALE speech (F0 87–124 Hz), normal
volume — 0624 4 speakers all male, zero female.  Not extrapolated.

Tests use 0624 speech; 0625 speech remains held out and untouched.  The
explicitly authorised 0625/FB_FF_TT_VPU_noise_floor.wav is used only where a
real device-noise floor is required.
  G1/G2/G4'/G5/G6  hard thresholds (depth-independent, per recording)
  G3a'/G3b'        band-level recovery curves vs depth (main effect)
  G7               phase-non-self-consistency pricing (AC1's cost)
  scenarios        D1 depth sweep; D1+D2/D3/D4; dropout; progressive weakening
  ablation         eq frozen vs adaptive; c_V components; etc. (DR1: all-5 explicit)
  listening pack   S/V/Y/X 4-channel WAVs (≥3 clips/condition) → reports/T13B1/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import soundfile as sf
from functools import lru_cache

from fusion import Fusion, FusionConfig
from fusion.fusion import FusionCore
from fusion.degrade import DegradationConfig, apply_d2, apply_d3, apply_d4, degrade
from fusion.stft import stft_batch, istft_batch
from fusion.f0 import f0_batch
from fusion import realdata
from tests._testutil import SkipTest
from tests._t13_eval import eval_specs

try:
    realdata.list_0624(); _HAVE = True
except Exception:
    _HAVE = False

BAND_EDGES_HZ = [100, 200, 315, 500, 800, 1250, 2000]   # 6 sub-bands (spec §III)
REPORT_DIR = "reports/T13B1"


def _need():
    if not _HAVE:
        raise SkipTest("0624 real recordings not accessible")


def _band_bins(cfg, lo_hz, hi_hz):
    bz = cfg.sr / cfg.n_fft
    return max(1, int(lo_hz / bz)), min(cfg.fusion_hi_bin, int(hi_hz / bz))


def _lsd_db(a_spec, b_spec, lo, hi):
    """Log-spectral distortion (RMS of 20log|a|−20log|b|) over bins [lo,hi] and
    all frames.  dB."""
    la = 20.0 * torch.log10(a_spec[:, lo:hi + 1].abs().clamp_min(1e-8))
    lb = 20.0 * torch.log10(b_spec[:, lo:hi + 1].abs().clamp_min(1e-8))
    return float(torch.sqrt(((la - lb) ** 2).mean()).item())


def _cos_td(a, b):
    """Time-domain cosine similarity ∈ [−1,1]."""
    a = a.double().reshape(-1); b = b.double().reshape(-1)
    return float((a @ b / (a.norm() * b.norm().clamp_min(1e-12))).item())


def _make_SV(cfg, deg, seg_s=6.0, off=1.0):
    """Load 0624 FF (X), VPU (V); build S = degrade(X) under deg; return X,S,V."""
    ff, vpu, sr = realdata.load_0624(seg_s=seg_s, offset_s=off)
    X = ff
    S = degrade(ff, cfg, deg) if (deg.d2_contrast > 0 or deg.d3_musical or deg.d4_envelope) \
        else _d1_only(ff, cfg, deg)
    return X, S, vpu


def _d1_only(ff, cfg, deg):
    """S with D1 only (default path used by most scenarios)."""
    _, _, S = eval_specs(ff, cfg, deg)
    return S


def _Y(cfg, X, S, V):
    return Fusion(cfg).process_batch(S, V)


# ================================================================ HR2 ======
def test_HR2_zero_w_identity():
    """HR2: w≡0 ⇒ Y≡S is an IDENTITY (not a statistical threshold).  Force
    w=0 across the full pipeline (real 0624) and assert Y torch.allclose S
    at numerical precision.  Structural guarantee of the HR1 S-anchored formula."""
    _need()
    cfg = FusionConfig().with_switches(enable_comfort_noise=False, enable_eq=False)  # EQ off ⇒ V'=V raw ≠ S (stronger identity test)

    class _WZero(Fusion):
        def process_batch(self, s, v):
            import torch.nn.functional as F
            from fusion.stft import stft_batch, istft_batch
            s = s.float(); v = v.float(); cfg = self.cfg
            spec_s = stft_batch(s, cfg); spec_v = stft_batch(v, cfg)
            left = cfg.win - cfg.hop; sp = F.pad(s, (left, 0))
            frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
            N = spec_s.shape[-1]; yf = []
            for t in range(N):
                ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
                f0, conf = self.core.f0est.estimate(buf)
                smag = ss.abs(); fl = self.core.nf.step(smag)
                snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
                v_prime, startup, reset = self.core.eq.step(ss, vs, snr, conf)
                w = torch.zeros(ss.shape, dtype=torch.float32, device=ss.device)   # <<< force w≡0 (real)
                yf.append(self.core.synth.step(ss, v_prime, w))
            self.last_s_spec = spec_s
            self.last_y_spec = torch.stack(yf, -1)
            return istft_batch(self.last_y_spec, cfg, length=s.shape[-1])

    ff, vpu, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    S_rt = istft_batch(stft_batch(ff, cfg), cfg, length=ff.shape[-1])  # roundtrip S (the S the pipeline sees)
    wz = _WZero(cfg)
    Y = wz.process_batch(ff, vpu)                  # S = ff (D1=0)
    mag_ok = torch.allclose(wz.last_y_spec.abs(), wz.last_s_spec.abs(),
                            rtol=1e-5, atol=1e-7)
    phase_ok = torch.allclose(torch.angle(wz.last_y_spec), torch.angle(wz.last_s_spec),
                              rtol=1e-5, atol=1e-7)
    md = (Y - S_rt).abs().max().item()
    # float32 error budget (fixed before this observation): log magnitude spans
    # ~160 dB, so eps*|logS| ~= 1.9e-5 dB, or ~2.2e-6 relative magnitude.
    # ISTFT overlap-add accumulates 3 frames x 257 bins; sqrt(771) ~= 28 gives
    # a waveform-domain floor around 1.1e-5.  atol=1e-4 is one decade of
    # platform/FFT-backend headroom, not a threshold fitted to the arm64 result.
    wave_ok = torch.allclose(Y, S_rt, rtol=1e-5, atol=1e-4)
    print(f"  HR2 w=0 identity: spectrum magnitude={mag_ok} phase={phase_ok} "
          f"(rtol=1e-5 atol=1e-7); waveform={wave_ok} maxdiff={md:.3e} (atol=1e-4)")
    print(f"    (STFT roundtrip itself is {((S_rt-ff).abs().max().item()):.3e}; the identity isolates the FORMULA — "
          f"old V'-anchor gave ~8 dB, new gives ~1e-5)")
    assert mag_ok, "HR2: w=0 spectrum magnitude is not identical to S"
    assert phase_ok, "HR2: w=0 spectrum phase is not identical to S"
    assert wave_ok, f"HR2: w=0 waveform differs beyond float32 ISTFT budget (maxdiff {md})"


def test_HR2_mutation():
    """Mutation: revert to the OLD V'-anchored formula (synth_legacy_vprime=True)
    ⇒ w=0 gives Y=V'+clip(S−V') ≠ S when V'≠S ⇒ the identity MUST FAIL."""
    _need()
    cfg = FusionConfig().with_switches(enable_comfort_noise=False, enable_eq=False, synth_legacy_vprime=True)  # EQ off ⇒ V'≠S ⇒ legacy w=0 ≠ S

    class _WZero(Fusion):
        def process_batch(self, s, v):
            import torch.nn.functional as F
            from fusion.stft import stft_batch, istft_batch
            s = s.float(); v = v.float(); cfg = self.cfg
            spec_s = stft_batch(s, cfg); spec_v = stft_batch(v, cfg)
            left = cfg.win - cfg.hop; sp = F.pad(s, (left, 0))
            frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
            N = spec_s.shape[-1]; yf = []
            for t in range(N):
                ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
                f0, conf = self.core.f0est.estimate(buf)
                smag = ss.abs(); fl = self.core.nf.step(smag)
                snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
                v_prime, _, _ = self.core.eq.step(ss, vs, snr, conf)
                w = torch.zeros(ss.shape, device=ss.device)
                yf.append(self.core.synth.step(ss, v_prime, w))
            self.last_s_spec = spec_s
            self.last_y_spec = torch.stack(yf, -1)
            return istft_batch(self.last_y_spec, cfg, length=s.shape[-1])

    ff, vpu, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    S_rt = istft_batch(stft_batch(ff, cfg), cfg, length=ff.shape[-1])
    wz = _WZero(cfg)
    Y = wz.process_batch(ff, vpu)
    md = (Y - S_rt).abs().max().item()
    spectrum_broken = not torch.allclose(wz.last_y_spec.abs(), wz.last_s_spec.abs(),
                                         rtol=1e-5, atol=1e-7)
    waveform_broken = not torch.allclose(Y, S_rt, rtol=1e-5, atol=1e-4)
    print(f"  HR2 mutation (legacy V'-anchor, w=0): spectrum_broken={spectrum_broken} "
          f"waveform_broken={waveform_broken} maxdiff={md:.3e} → "
          f"{'FAIL-of-mutant (caught) PASS' if spectrum_broken and waveform_broken else 'NOT caught — PROBLEM'}")
    assert spectrum_broken, "HR2 spectrum mutation lost teeth against legacy V'-anchor"
    assert waveform_broken, "HR2 waveform mutation lost teeth against legacy V'-anchor"


# ================================================================ G1 ======
def test_G1_no_damage_clean():
    """G1: D1=0 (S=X clean) ⇒ fusion must not damage healthy signal.
    0–2 kHz LSD(Y,S) < 1.0 dB; cos(Y,X) ≥ cos(S,X) − 0.01."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    X = ff; S = ff            # D1=0 ⇒ S = X
    Y = _Y(cfg, X, S, vpu)
    lo, hi = _band_bins(cfg, 100, 2000)
    lsd = _lsd_db(stft_batch(Y, cfg), stft_batch(S, cfg), lo, hi)
    cy = _cos_td(Y, X); cs = _cos_td(S, X)
    ok = lsd < 1.0 and cy >= cs - 0.01
    print(f"  G1 (D1=0): LSD(Y,S)={lsd:.3f} dB (<1.0 {'PASS' if lsd<1.0 else 'FAIL'})  "
          f"cos(Y,X)={cy:.4f} ≥ cos(S,X)−0.01={cs - 0.01:.4f} {'PASS' if cy>=cs-0.01 else 'FAIL'}")
    assert ok, f"G1: LSD={lsd:.3f} dB, cos(Y,X)={cy:.6f}, floor={cs - 0.01:.6f}"


# ================================================================ G4'/G6 ==
def _per_band_lsd(Y, ref, cfg):
    """Return list of (lo_hz, hi_hz, lsd_db) per sub-band."""
    Ys = stft_batch(Y, cfg); Rs = stft_batch(ref, cfg)
    out = []
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        out.append((BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1], _lsd_db(Ys, Rs, lo, hi)))
    return out


@lru_cache(maxsize=1)
def _measure_G4prime_G6_depth_sweep():
    """Measure exact pointwise G4' and depth-wise G6 once.

    G4' uses evaluation-only X/S truth.  Every (band, frame) with
    ``sup_dB <= 1`` must independently satisfy the +0.3 dB bound; values are
    never averaged to decide the gate.  The 1..6 dB grey zone is reported but
    judged by neither G4' nor G3a'.
    """
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    print("  G4' exact pointwise depth x band table:")
    print("  depth band(Hz)       U  violations  max_excess  grey%")
    rows = []
    violations = []
    for d in [0, 3, 6, 10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        X = ff; S = _d1_only(ff, cfg, deg); Y = _Y(cfg, X, S, vpu)
        spec_x = stft_batch(X, cfg); spec_s = stft_batch(S, cfg); spec_y = stft_batch(Y, cfg)
        # G4' exact evaluation set: band-frames with suppression <=1 dB.
        # 1..6 dB is a grey zone (reported, judged by neither G4' nor G3a').
        n_grey = 0; n_sup = 0; n_total = 0; depth_violations = []; band_rows = []
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
            sup_db = 10.0 * torch.log10(px / ps)
            xs = 20 * torch.log10(spec_x[0, lo:hi + 1].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
            lsd_s = torch.sqrt(((ss - xs) ** 2).mean(0))
            lsd_y = torch.sqrt(((ys - xs) ** 2).mean(0))
            u_mask = sup_db <= 1.0
            grey_mask = (sup_db > 1.0) & (sup_db <= 6.0)
            p_mask = sup_db > 6.0
            excess = lsd_y - lsd_s - 0.3
            bad_idx = torch.nonzero(u_mask & (excess > 0.0), as_tuple=False).flatten()
            band_violations = []
            for t_tensor in bad_idx:
                t = int(t_tensor)
                item = dict(depth=d, band=f"{BAND_EDGES_HZ[i]}-{BAND_EDGES_HZ[i + 1]}",
                            t=t, sup_db=float(sup_db[t]), lsd_y=float(lsd_y[t]),
                            lsd_s=float(lsd_s[t]), excess=float(excess[t]))
                band_violations.append(item); depth_violations.append(item); violations.append(item)
            n_u = int(u_mask.sum())
            n_band_grey = int(grey_mask.sum())
            n_total += int(sup_db.numel()); n_grey += n_band_grey; n_sup += int(p_mask.sum())
            max_excess = max((v["excess"] for v in band_violations), default=0.0)
            band_rows.append(dict(band=f"{BAND_EDGES_HZ[i]}-{BAND_EDGES_HZ[i + 1]}",
                                  n_u=n_u, n_viol=len(band_violations),
                                  max_excess=max_excess,
                                  grey=n_band_grey / max(1, int(sup_db.numel()))))
            print(f"  {d:>5} {BAND_EDGES_HZ[i]:>4}-{BAND_EDGES_HZ[i + 1]:<4} "
                  f"{n_u:>6} {len(band_violations):>11} {max_excess:>11.3f} "
                  f"{n_band_grey / max(1, int(sup_db.numel())):>6.1%}")
        g4 = not depth_violations
        cy = _cos_td(Y, X); cs = _cos_td(S, X)
        g6 = cy >= cs - 0.01
        grey_ratio = n_grey / max(1, n_total)
        rows.append(dict(depth=d, g4=g4, n_viol=len(depth_violations), bands=band_rows,
                         cy=cy, cs=cs, g6=g6, grey=grey_ratio,
                         suppressed=n_sup / max(1, n_total)))
        print(f"        depth {d}: G4'={'PASS' if g4 else 'FAIL'} "
              f"violations={len(depth_violations)} grey={grey_ratio:.1%} "
              f"G6={'PASS' if g6 else 'FAIL'} cos={cy:.4f}/{cs:.4f}")
    print("  G4' first violations (depth, band, t, sup_dB, LSD_YX, LSD_SX, excess):")
    for v in violations[:12]:
        print(f"    ({v['depth']:>2}, {v['band']:>9}, {v['t']:>3}) "
              f"sup={v['sup_db']:+.3f} Y={v['lsd_y']:.3f} S={v['lsd_s']:.3f} "
              f"excess={v['excess']:.3f}")
    if not violations:
        print("    none")
    if any(r["grey"] > 0.40 for r in rows):
        print("  WARNING: grey-zone share exceeds 40% at one or more depths; partition needs review")
    return rows


def test_G4prime_depth_sweep():
    """G4': every U point passes independently; no depth or point averaging."""
    rows = _measure_G4prime_G6_depth_sweep()
    g4_all = all(r["g4"] for r in rows)
    failures = {r["depth"]: r["n_viol"] for r in rows if not r["g4"]}
    assert g4_all, f"G4': pointwise +0.3 dB bound violated; violations by depth={failures}"


def test_G6_depth_sweep():
    """G6 corrected form: cos(Y,X)>=cos(S,X)-0.01 at every depth."""
    rows = _measure_G4prime_G6_depth_sweep()
    g6_all = all(r["g6"] for r in rows)
    assert g6_all, "G6: corrected -0.01 cosine floor fails at one or more depths"


# ================================================================ G3a'/b' =
G3A_MIN_SAMPLES = 30


@lru_cache(maxsize=1)
def _measure_G3aprime_recovery_curve():
    """G3a' (MAIN effect): on band-frames suppressed >6 dB, LSD_band(Y,X) vs
    LSD_band(S,X).  🔴 criterion: EXISTS depth≤20 with LSD_band(Y,X) ≤ 0.5×
    LSD_band(S,X).  Reports the depth curve + plot."""
    _need()
    import os; os.makedirs(REPORT_DIR, exist_ok=True)
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg)
    _, conf_tr = f0_batch(ff, cfg)
    bz = cfg.sr / cfg.n_fft
    depths = [0, 3, 6, 10, 15, 20, 30]
    rows = []
    print(f"  G3a' band-recovery (suppressed>6dB band-frames) vs depth:")
    assert spec_X.dim() == 3, f"KR4: spec must be 3D (B,F,T), got {spec_X.dim()}D"   # KR4 dim assert
    print(f"  {'depth':>5} {'LSD_SX':>7} {'LSD_YX':>7} {'ratio':>8}  (ratio≤0.5 ⇒ recovery)")
    for d in depths:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        spec_X, spec_S, S = eval_specs(ff, cfg, deg)
        Y = _Y(cfg, ff, S, vpu)
        spec_Y = stft_batch(Y, cfg)
        # New exact set P uses only evaluation-side band-power suppression;
        # no F0-confidence filter is part of the reviewer's definition.
        lsd_sx_bt = []; lsd_yx_bt = []; lsd_oracle_bt = []; n_grey = 0; n_eval = 0
        legacy_sse_s = 0.0; legacy_sse_y = 0.0; legacy_n = 0
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            xs = 20 * torch.log10(spec_X[0, lo:hi + 1].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_S[0, lo:hi + 1].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_Y[0, lo:hi + 1].abs().clamp_min(1e-8))
            for t in range(spec_S.shape[-1]):
                px = spec_X[0, lo:hi + 1, t].abs().pow(2).mean().clamp_min(1e-16)
                ps = spec_S[0, lo:hi + 1, t].abs().pow(2).mean().clamp_min(1e-16)
                sup_db = float(10.0 * torch.log10(px / ps))
                n_eval += 1
                if sup_db > 6.0:
                    # Equal weight per (band, frame).  Energy weighting would
                    # let strong bands dominate the weak-band repair target.
                    lsd_sx_bt.append(float(torch.sqrt(((ss[:, t] - xs[:, t]) ** 2).mean())))
                    lsd_yx_bt.append(float(torch.sqrt(((ys[:, t] - xs[:, t]) ** 2).mean())))
                    g_star = (xs[:, t] - ss[:, t]).mean()
                    lsd_oracle_bt.append(float(torch.sqrt(
                        ((ss[:, t] + g_star - xs[:, t]) ** 2).mean())))
                elif sup_db > 1.0:
                    n_grey += 1
                # One-shot legacy comparison: voiced-only, mean-log-drop set,
                # then RMS over all selected band-bin errors.  It is reported
                # only to make the aggregation change explicit, never gated.
                if (float(conf_tr[0, t]) >= 0.55 and
                        float((ss[:, t] - xs[:, t]).mean()) < -6.0):
                    legacy_sse_s += float(((ss[:, t] - xs[:, t]) ** 2).mean())
                    legacy_sse_y += float(((ys[:, t] - xs[:, t]) ** 2).mean())
                    legacy_n += 1
        lsd_sx = float(np.mean(lsd_sx_bt)) if lsd_sx_bt else 0.0
        lsd_yx = float(np.mean(lsd_yx_bt)) if lsd_yx_bt else 0.0
        ratio = lsd_yx / max(1e-3, lsd_sx)
        lsd_oracle = float(np.mean(lsd_oracle_bt)) if lsd_oracle_bt else 0.0
        ratio_oracle = lsd_oracle / max(1e-3, lsd_sx)
        norm_gap = ((ratio - ratio_oracle) / max(1e-8, 1.0 - ratio_oracle)
                    if lsd_sx_bt else float("nan"))
        legacy_s = np.sqrt(legacy_sse_s / max(1, legacy_n))
        legacy_y = np.sqrt(legacy_sse_y / max(1, legacy_n))
        legacy_ratio = legacy_y / max(1e-3, legacy_s)
        rows.append((d, lsd_sx, lsd_yx, ratio, len(lsd_sx_bt),
                     n_grey / max(1, n_eval), legacy_ratio,
                     lsd_oracle, ratio_oracle, norm_gap))
        print(f"  {d:>5} {lsd_sx:>7.2f} {lsd_yx:>7.2f} {ratio:>8.5f} "
              f"n_sup={len(lsd_sx_bt)} grey={n_grey/max(1,n_eval):.1%} "
              f"legacy={legacy_ratio:.3f}")
    _plot_g3a(rows)
    return rows


def test_G3aprime_recovery_curve():
    """G3a' gate with the A2 adequacy floor fixed before observation.

    Depths with fewer than 30 points in P are INSUFFICIENT: they participate
    in neither the existential PASS nor a per-depth FAIL.  Sufficient depths
    retain the original ratio<=0.5 criterion.
    """
    rows = _measure_G3aprime_recovery_curve()
    insufficient = [(r[0], r[4]) for r in rows if r[4] < G3A_MIN_SAMPLES]
    sufficient = [r for r in rows if r[4] >= G3A_MIN_SAMPLES]
    print(f"  G3a' adequacy: INSUFFICIENT={insufficient}; "
          f"eligible depths={[r[0] for r in sufficient]}")
    exists = any(r[0] <= 20 and r[3] <= 0.5 for r in sufficient)
    print(f"  G3a' criterion (∃ sufficient depth≤20 with ratio≤0.5): "
          f"{'PASS' if exists else 'FAIL — not met'}")
    assert exists, "G3a': no depth<=20 with n_sup>=30 halves equal-weight suppressed-band-frame LSD"
    return rows


def _plot_g3a(rows):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    d = [r[0] for r in rows]
    ax.plot(d, [r[1] for r in rows], "o-", label="LSD(S,X) (suppressed bands)")
    ax.plot(d, [r[2] for r in rows], "s-", label="LSD(Y,X) (after fusion)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("D1 suppression depth (dB)")
    ax.set_ylabel("LSD (dB, suppressed band-frames)")
    ax.set_title("G3a' band-recovery vs depth — ♂ 0624 normal-volume")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{REPORT_DIR}/g3a_recovery.png", dpi=110); plt.close(fig)
    print(f"  plot → {REPORT_DIR}/g3a_recovery.png")


def test_G3bprime_out_of_band():
    """G3b': outside the VPU-usable band (online MSC判定 ≤800 Hz here), bands
    800–2000 Hz: LSD_band(Y,X) ≤ LSD_band(S,X)+0.5 dB, all depth."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    print(f"  G3b' out-of-VPU-band (800–2000 Hz) LSD_band(Y,X) vs LSD_band(S,X):")
    ok_all = True
    for d in [6, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        S = _d1_only(ff, cfg, deg); Y = _Y(cfg, ff, S, vpu)
        lo, hi = _band_bins(cfg, 800, 2000)
        yx = _lsd_db(stft_batch(Y, cfg), stft_batch(ff, cfg), lo, hi)
        sx = _lsd_db(stft_batch(S, cfg), stft_batch(ff, cfg), lo, hi)
        ok = yx <= sx + 0.5
        ok_all = ok_all and ok
        print(f"    depth={d}: LSD(Y,X)={yx:.2f}  LSD(S,X)={sx:.2f}  Δ={yx - sx:+.2f} (≤0.5) {'✓' if ok else '✗'}")
    print(f"  G3b' out-of-band: {'PASS' if ok_all else 'FAIL (reported — AC1 base-V on misaligned out-of-band bins)'}")
    assert ok_all, "G3b': out-of-VPU-band LSD worsens by >0.5 dB"


# ================================================================ G5 ======
def test_G5_causal_phase_change():
    """G5 (MUST pass or task not accepted): future-perturbation bit-identical
    on REAL voiced, AFTER the AC1 phase change (∠Y=∠S).  Plus 2 mutations
    (global-mean-norm; w_local look-ahead) still have teeth."""
    _need()
    cfg = FusionConfig()
    s, v, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    y_full = Fusion(cfg).process_batch(s, v)
    assert torch.isfinite(y_full).all()
    T = s.shape[-1]; worst = 0.0
    for P in [cfg.hop * 40, cfg.hop * 80, T // 2]:
        sm = s.clone(); sm[:, P:] = 0.0; vm = v.clone(); vm[:, P:] = 0.0
        ym = Fusion(cfg).process_batch(sm, vm)
        K = max(0, P - cfg.win)
        assert torch.equal(y_full[..., :K], ym[..., :K]), f"G5: future leak at P={P}"
        worst = max(worst, (y_full[..., :K] - ym[..., :K]).abs().max().item())
    print(f"  G5 causal (post-AC1): future-perturb bit-identical, worst diff={worst}")
    # mutation 1: global-mean-norm(Y)
    from tests.test_t13_streaming import _MutantGlobalMeanNorm
    mut = _MutantGlobalMeanNorm(cfg); yf = mut.process_batch(s, v)
    P = T // 2; sm = s.clone(); sm[:, P:] = 0.0; vm = v.clone(); vm[:, P:] = 0.0
    ym = mut.process_batch(sm, vm); K = max(0, P - cfg.win)
    leak1 = (yf[..., :K] - ym[..., :K]).abs().max().item()
    # mutation 2: w_local look-ahead
    from tests.test_t13_real import test_R2_mutation_wlocal_lookahead
    test_R2_mutation_wlocal_lookahead()   # asserts voiced>white, >1e-6
    print(f"  G5 mutations: global-mean-norm leak={leak1:.3e} (>1e-6 {'✓' if leak1 > 1e-6 else '✗'}); "
          f"w_local-lookahead voiced>white ✓ (teeth retained post-AC1)")
    assert leak1 > 1e-6, "G5 mutation 1 (global-mean-norm) lost teeth"


# ================================================================ G2 ======
def _g2_floor(T, sr, vpu):
    """Only the authorised 0625 noise-floor file; VPU is channel index 3."""
    rec = os.environ.get("MIC_REC_ROOT", "/mnt/d/Projects/mic_array_capture/mic_recordings")
    path = os.path.join(rec, "0625", "FB_FF_TT_VPU_noise_floor.wav")
    if not os.path.exists(path):
        raise SkipTest("G2 requires the authorised 0625 noise_floor.wav")
    wav, nf_sr = sf.read(path, dtype="float32")
    assert nf_sr == sr, f"G2 noise-floor sample rate {nf_sr} != {sr}"
    if wav.ndim > 1:
        wav = wav[:, 3]  # FB/FF/TT/VPU: dropout must use VPU, not FB
    floor = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    if floor.numel() < T:
        floor = floor.repeat(int(np.ceil(T / floor.numel())))
    floor = floor[:T].unsqueeze(0)
    # realdata currently peak-normalises V; retain the real noise-floor shape
    # and place it at V's stationary (10th-percentile) floor, as in K-c.
    floor *= float(torch.quantile(vpu.abs().flatten(), 0.10)) / (
        float(torch.quantile(floor.abs().flatten(), 0.10)) + 1e-9)
    return floor


def _g2_frame_lsd(spec_y, spec_s, lo, hi):
    ly = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
    ls = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
    return torch.sqrt(((ly - ls) ** 2).mean(0)).cpu().numpy()


def _g2_window_lsd(frame_lsd, a, b, hop):
    i0 = max(0, int(np.ceil(a / hop))); i1 = min(len(frame_lsd), int(np.floor(b / hop)))
    return float(np.sqrt(np.mean(np.square(frame_lsd[i0:i1])))) if i1 > i0 else float("nan")


@lru_cache(maxsize=1)
def _measure_G2_all_recordings():
    """Correct 3 s in / 2 s hold / 2 s out protocol on all ten 0624 files."""
    _need()
    cfg = FusionConfig(); rows = []
    for path in realdata.list_0624():
        name = os.path.basename(path)
        ff, vpu, sr = realdata.load_0624(name=name, seg_s=8.0, offset_s=1.0)
        T = ff.shape[-1]; floor = _g2_floor(T, sr, vpu)
        fi = int(0.5 * sr); fo = fi + int(3.0 * sr)
        ho = fo + int(2.0 * sr); to = ho + int(2.0 * sr)
        assert to <= T - cfg.win, "G2 protocol does not fit the selected segment"
        env = torch.zeros(T)
        env[fi:fo] = torch.linspace(0.0, 1.0, fo - fi)
        env[fo:ho] = 1.0                       # the legacy test accidentally omitted this hold
        env[ho:to] = torch.linspace(1.0, 0.0, to - ho)
        v = vpu * (1 - env) + floor * env
        Y = _Y(cfg, ff, ff, v)
        spec_y = stft_batch(Y, cfg); spec_s = stft_batch(ff, cfg)
        lo, hi = _band_bins(cfg, 100, 2000)
        flsd = _g2_frame_lsd(spec_y, spec_s, lo, hi)
        steady = _g2_window_lsd(flsd, fo + cfg.win, ho, cfg.hop)
        in_lsd = _g2_window_lsd(flsd, fi, fo, cfg.hop)
        out_lsd = _g2_window_lsd(flsd, ho, to, cfg.hop)
        tin = flsd[fi // cfg.hop:fo // cfg.hop]
        tout = flsd[ho // cfg.hop:to // cfg.hop]
        step_in = float(np.max(np.abs(np.diff(tin)))) if len(tin) > 1 else 0.0
        step_out = float(np.max(np.abs(np.diff(tout)))) if len(tout) > 1 else 0.0
        stem = name.removeprefix("FB_FF_TT_VPU_").removesuffix(".wav")
        speaker = stem.split("_", 1)[0]
        position = stem.split("left_ear_", 1)[-1]
        rows.append(dict(name=name, speaker=speaker, position=position,
                         steady_lsd=steady, in_lsd=in_lsd, out_lsd=out_lsd,
                         step_in=step_in, step_out=step_out,
                         whole_lsd=_lsd_db(spec_y, spec_s, lo, hi)))
    return rows


def test_G2_dropout_fallback():
    """G2 per 0624 recording: steady dropout LSD<0.5 dB and transition steps<3 dB."""
    rows = _measure_G2_all_recordings()
    print("  G2 corrected protocol (0625 VPU noise-floor idx3; 3s in/2s hold/2s out):")
    print("  speaker position    steady  fade-in fade-out step-in step-out whole")
    for r in rows:
        print(f"  {r['speaker']:7s} {r['position']:10s} {r['steady_lsd']:7.3f} "
              f"{r['in_lsd']:7.3f} {r['out_lsd']:8.3f} {r['step_in']:7.3f} "
              f"{r['step_out']:8.3f} {r['whole_lsd']:6.3f}")
    steady_ok = all(r["steady_lsd"] < 0.5 for r in rows)
    steps_ok = all(max(r["step_in"], r["step_out"]) < 3.0 for r in rows)
    print("  legacy 0.538 dB cannot be decomposed into steady vs transient: the old test "
          "used FB idx0 as the floor, measured the whole clip, and omitted env[fo:ho]=1, "
          "so it contained no dropout hold.  The table above supersedes that invalid aggregate.")
    assert steady_ok, "G2: at least one recording has steady dropout LSD(Y,S)>=0.5 dB"
    assert steps_ok, "G2: at least one transition has a >3 dB frame-to-frame LSD step"


# ================================================================ G7 ======
def test_G7_phase_pricing():
    """G7 (report, no threshold): AC1's cost = phase non-self-consistency.
    Same |Y| (∠S) vs ∠X (clean-FF oracle phase): report objective LSD gap +
    emit a listening sample pair.  This gap IS the AC1 tradeoff's full price."""
    _need()
    import os; os.makedirs(REPORT_DIR, exist_ok=True)
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    X = ff; S = _d1_only(ff, cfg, deg)
    YS = _Y(cfg, X, S, vpu)                       # ∠Y = ∠S (deployed, AC1)
    # oracle-phase variant: same |Y| but ∠X
    from fusion.synthesis import logclip_mix
    spec_S = stft_batch(S, cfg); spec_V = stft_batch(vpu, cfg); spec_X = stft_batch(X, cfg)
    YS_spec = stft_batch(YS, cfg)
    magY = YS_spec.abs()
    YX_spec = magY * torch.exp(1j * torch.angle(spec_X))
    YX = istft_batch(YX_spec, cfg, length=X.shape[-1])
    lo, hi = _band_bins(cfg, 100, 2000)
    gap = _lsd_db(stft_batch(YS, cfg), stft_batch(YX, cfg), lo, hi)
    cos_diff = _cos_td(YS, YX)
    ref = _lsd_db(stft_batch(S, cfg), stft_batch(X, cfg), lo, hi)   # HR3: LSD(S,X) = stage-2's OWN cost (degraded S vs clean X)
    print(f"  G7 phase pricing: LSD(∠S-variant, ∠X-variant)={gap:.3f} dB  cos(∠S,∠X)={cos_diff:.4f}")
    print(f"    HR3 reference LSD(S,X)={ref:.3f} dB (stage-2's own cost) — phase gap is {gap/ref:.2f}× of it")
    print(f"    (this gap = AC1's full cost; ∠X is oracle, unavailable at deploy)")
    sf.write(f"{REPORT_DIR}/g7_phase_Sphase.wav", YS.squeeze().numpy(), sr)
    sf.write(f"{REPORT_DIR}/g7_phase_Xphase.wav", YX.squeeze().numpy(), sr)
    sf.write(f"{REPORT_DIR}/g7_ref_X.wav", X.squeeze().numpy(), sr)
    print(f"    samples → {REPORT_DIR}/g7_phase_{{S,X}}phase.wav + g7_ref_X.wav")
    return gap


# ================================================================ scenarios
def test_scenario_D2D3D4_all():
    """Scenario 2: D1+D2 / D1+D3 / D1+D4 each once, + all-on.  Report G6 (cos)
    + finiteness.  Coverage, not a quality gate."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    cases = [
        ("D1+D2", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0, d2_contrast=0.5)),
        ("D1+D3", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0, d3_musical=True)),
        ("D1+D4", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0, d4_envelope=True)),
        ("all-on", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0, d2_contrast=0.5,
                                       d3_musical=True, d4_envelope=True)),
    ]
    print(f"  scenario D2/D3/D4/all:")
    for lab, deg in cases:
        X = ff; S = degrade(ff, cfg, deg); Y = _Y(cfg, X, S, vpu)
        fin = bool(torch.isfinite(Y).all())
        cy = _cos_td(Y, X); cs = _cos_td(S, X)
        print(f"    {lab:8s}: finite={fin}  cos(Y,X)={cy:.4f} (cos(S,X)={cs:.4f}, "
              f"G6ε {'✓' if cy >= cs - 0.01 else '✗ — reported'})")
        assert fin, f"{lab}: non-finite output"


def test_scenario_progressive_weakening():
    """Scenario 4 (🔑): VPU −3/−6/−12 dB + slow EQ shift (weak & transfer-fn
    changed).  Harder than dropout — the real test of c_V + EQ freeze/re-
    estimate.  Focus G6 (cos(Y,X) ≥ cos(S,X))."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    S = _d1_only(ff, cfg, deg)
    print(f"  scenario progressive weakening (VPU atten + EQ shift):")
    for db in [-3, -6, -12]:
        v = vpu * (10 ** (db / 20.0))
        # slow EQ shift: tilt V by +3 dB/oct above 200 Hz (transfer-fn changed)
        spec_v = stft_batch(v, cfg).clone()
        bz = cfg.sr / cfg.n_fft
        for b in range(1, spec_v.shape[1]):
            tilt = 10 ** (3.0 * (b * bz - 200) / (200 * 12) / 20.0)   # +3 dB/oct
            spec_v[0, b] *= tilt
        import torch.nn.functional as F
        v_shift = istft_batch(spec_v, cfg, length=v.shape[-1])
        Y = _Y(cfg, ff, S, v_shift)
        cy = _cos_td(Y, ff); cs = _cos_td(S, ff)
        g6 = cy >= cs - 0.01
        print(f"    VPU {db:+d} dB + EQ shift: cos(Y,X)={cy:.4f} (cos(S,X)={cs:.4f}) G6 {'✓' if g6 else '✗ — reported'}")


# ================================================================ ablation
ABLATION_BASELINE = {
    "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
    "enable_g_f0": True, "enable_w_band": True,
    "use_w_band_fixed_curve": False, "enable_w_local": True,
    "use_w_local_pure_band": False, "enable_comfort_noise": True,
    "delta_db": 10.0,
}
ABLATION_OVERRIDES = {
    "baseline (AC1/2/3)": {},
    "eq_mode=adaptive": {"eq_mode": "adaptive"},
    "enable_eq=False": {"enable_eq": False},
    "enable_c_V=False": {"enable_c_V": False},
    "enable_g_f0=False": {"enable_g_f0": False},
    "enable_w_band=False": {"enable_w_band": False},
    "w_band=fixed_curve": {"use_w_band_fixed_curve": True},
    "enable_w_local=False": {"enable_w_local": False},
    "w_local=pure_band": {"use_w_local_pure_band": True},
    "enable_comfort_noise=False": {"enable_comfort_noise": False},
    "delta_db=0 (no log-clip)": {"delta_db": 0.0},
}
ABLATION_SWITCHES = [
    ("baseline (AC1/2/3)", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("eq_mode=adaptive", {
        "eq_mode": "adaptive", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_eq=False", {
        "eq_mode": "frozen", "enable_eq": False, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_c_V=False", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": False,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_g_f0=False", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": False, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_w_band=False", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": False,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("w_band=fixed_curve", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": True, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_w_local=False", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": False,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("w_local=pure_band", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": True, "enable_comfort_noise": True,
        "delta_db": 10.0,
    }),
    ("enable_comfort_noise=False", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": False,
        "delta_db": 10.0,
    }),
    ("delta_db=0 (no log-clip)", {
        "eq_mode": "frozen", "enable_eq": True, "enable_c_V": True,
        "enable_g_f0": True, "enable_w_band": True,
        "use_w_band_fixed_curve": False, "enable_w_local": True,
        "use_w_local_pure_band": False, "enable_comfort_noise": True,
        "delta_db": 0.0,
    }),
]


def _validate_ablation_rows(rows):
    relkeys = set(ABLATION_BASELINE)
    labels = {label for label, _ in rows}
    assert labels == set(ABLATION_OVERRIDES), "DR1: missing or unknown ablation row label"
    for label, switches in rows:
        assert set(switches) == relkeys, f"DR1: {label} does not explicitly set every switch"
        expected = {**ABLATION_BASELINE, **ABLATION_OVERRIDES[label]}
        assert switches == expected, f"DR1: {label} values do not match its declared leave-one-out semantics"


def test_ablation_DR1_meta():
    """DR1 future guard: all switches explicit and labels match values.

    This repairs future protection only.  Existing B1 rows were independent
    leave-one-out switches with defaults equal to the baseline, so their
    measured ablation meaning is not withdrawn by this meta-test repair.
    """
    _validate_ablation_rows(ABLATION_SWITCHES)
    print(f"  DR1 meta (B1): {len(ABLATION_SWITCHES)} ablation rows, all switches explicit ✓")


def test_ablation_DR1_meta_mutation():
    """Mutation: omit one switch; the completeness guard must fail."""
    bad = [(label, dict(switches)) for label, switches in ABLATION_SWITCHES]
    changed_label = bad[1][0]; del bad[1][1]["enable_c_V"]
    broken = False; failure = ""
    try:
        _validate_ablation_rows(bad)
    except AssertionError as exc:
        broken = True; failure = str(exc)
    print(f"  DR1 mutation: changed row {changed_label!r} by removing 'enable_c_V'; "
          f"failure={failure!r} → {'FAIL-of-mutant caught' if broken else 'NOT caught'}")
    assert broken, "DR1 mutation: omitted switch did not fail completeness meta-test"


def test_ablation_frozen_vs_adaptive():
    """AC2 direct test: frozen EQ vs continuous-adaptive, on the depth-sweep +
    progressive-weakening scenarios.  If adaptive is significantly better, the
    reviewer's inference (continuous buys ~nothing) is WRONG — report honestly."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    S = _d1_only(ff, cfg, deg)
    print(f"  ablation frozen vs adaptive (cos(Y,X); higher=closer to clean):")
    fr = cfg.with_switches(eq_mode="frozen")
    ad = cfg.with_switches(eq_mode="adaptive")
    for lab, c in [("frozen (B1)", fr), ("adaptive (B0)", ad)]:
        Y = _Y(c, ff, S, vpu)
        print(f"    {lab:16s}: cos(Y,X)={_cos_td(Y, ff):.4f}")
    # progressive weakening
    print(f"    @ progressive weakening (VPU −12 dB + EQ shift):")
    v = vpu * (10 ** (-12 / 20.0))
    for lab, c in [("frozen", fr), ("adaptive", ad)]:
        Y = _Y(c, ff, S, v)
        print(f"      {lab:16s}: cos(Y,X)={_cos_td(Y, ff):.4f}")
    print(f"  (reported; no assertion — AC2 ablation is a REPORT item, reviewer's "
          f"inference checked honestly)")


# ================================================================ listening pack
def test_listening_pack():
    """Emit S/V/Y/X 4-channel WAVs, ≥3 conditions, → reports/T13B1/.
    Relative paths reported for the reviewer."""
    _need()
    import os; os.makedirs(REPORT_DIR, exist_ok=True)
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=3.0, offset_s=1.0)
    conds = [
        ("d0_clean", DegradationConfig(d1_kill_rate=0.0)),
        ("d6", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=6.0)),
        ("d20", DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)),
    ]
    paths = []
    for lab, deg in conds:
        X = ff; S = _d1_only(ff, cfg, deg) if deg.d1_kill_rate > 0 else ff
        Y = _Y(cfg, X, S, vpu)
        for ch, sig in [("S", S), ("V", vpu), ("Y", Y), ("X", X)]:
            p = f"{REPORT_DIR}/lp_{lab}_{ch}.wav"
            sf.write(p, sig.squeeze().numpy(), sr)
            paths.append(p)
    print(f"  listening pack: {len(paths)} WAVs → {REPORT_DIR}/lp_<cond>_<S|V|Y|X>.wav")
    print(f"    conditions: {[c[0] for c in conds]} (each 4-channel)")
    assert len(paths) >= 12


# ================================================================ HR4 ======
def test_HR4_w_local_band_uses_V():
    """HR4 (ER1 band-level): w_local_band uses V's overall level (Pv_overall).
    Replace Pv with a FIXED CONSTANT (v_perturb="const") and rerun G3a'.
    🔴 If performance doesn't significantly drop ⇒ w_local_band isn't using V ⇒
    DELETE it, keep only w_band (MSC)."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); f0_tr, conf_tr = f0_batch(ff, cfg)
    bz = cfg.sr / cfg.n_fft
    print(f"  HR4: w_local_band real-V vs const-Pv (G3a' suppressed-band recovery):")
    for lab, vp in [("real V", "none"), ("const Pv", "const")]:
        c = cfg.with_switches(wl_v_perturb=vp)
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=10.0)  # G3a' best depth
        spec_X, spec_S, S = eval_specs(ff, c, deg)
        Y = _Y(c, ff, S, vpu)
        spec_Y = stft_batch(Y, c)
        lo, hi = _band_bins(cfg, 100, 2000); lsd_sx = lsd_yx = 0.0; n = 0
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo2, hi2 = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            xs = 20 * torch.log10(spec_X[0, lo2:hi2 + 1].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_S[0, lo2:hi2 + 1].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_Y[0, lo2:hi2 + 1].abs().clamp_min(1e-8))
            for t in range(spec_S.shape[-1]):
                if float(conf_tr[0, t]) < 0.55: continue
                if (ss[:, t] - xs[:, t]).mean().item() < -6.0:
                    lsd_sx += ((ss[:, t] - xs[:, t]) ** 2).mean().item()
                    lsd_yx += ((ys[:, t] - xs[:, t]) ** 2).mean().item(); n += 1
        r_sx = np.sqrt(lsd_sx / max(1, n)); r_yx = np.sqrt(lsd_yx / max(1, n))   # KR4: no *10
        print(f"    {lab:9s}: LSD(S,X)={r_sx:.2f} LSD(Y,X)={r_yx:.2f} ratio={r_yx/max(1e-3,r_sx):.3f}")
    print(f"  (if const ≈ real ⇒ w_local_band not using V ⇒ candidate for deletion; "
          f"see README HR4 conclusion)")


# ================================================================ JR1 ======
def _pv_seq(cfg, vpu, mode):
    """Compute per-frame Pv override sequence (V level 100–800 Hz) under a
    TIME-axis ER1 perturbation.  Returns (N,) tensor or None (real V)."""
    spec_v = stft_batch(vpu, cfg)
    bz = cfg.sr / cfg.n_fft
    vlo = max(1, int(cfg.eq_band_lo_hz / bz)); vhi = min(cfg.fusion_hi_bin, int(cfg.wl_v_usable_hi_hz / bz))
    Pv = 10.0 * torch.log10(spec_v[0, vlo:vhi + 1].abs().pow(2).mean(0).clamp_min(1e-10))  # (N,) dB
    if mode == "real":
        return None
    if mode == "const-longterm":
        a = 1 - float(torch.exp(torch.tensor(-cfg.hop / (cfg.sr * 2.0))))  # 2-s EMA
        ema = float(Pv[0].item()); out = []
        for i in range(len(Pv)):
            ema = (1 - a) * ema + a * float(Pv[i].item())
            out.append(ema)
        return torch.tensor(out, dtype=Pv.dtype)
    if mode == "shuffle-time":
        g = torch.Generator().manual_seed(0)
        return Pv[torch.randperm(len(Pv), generator=g)]
    if mode == "fixed-arbitrary":
        return torch.zeros_like(Pv)   # 0 dB — a constant UNRELATED to V
    return None


def _g3a_ratio(cfg, ff, vpu, deg, pv_mode):
    """Run the pipeline with a Pv perturbation (via mutant) and return the
    G3a' ratio (LSD(Y,X)/LSD(S,X) on suppressed>6dB band-frames)."""
    spec_X, spec_S, S = eval_specs(ff, cfg, deg)
    _, conf_tr = f0_batch(ff, cfg)
    pv_seq = _pv_seq(cfg, vpu, pv_mode)

    class _PvMut(Fusion):
        def process_batch(self, s, v):
            import torch.nn.functional as F
            s = s.float(); v = v.float(); c = self.cfg
            spec_s = stft_batch(s, c); spec_v = stft_batch(v, c)
            left = c.win - c.hop; sp = F.pad(s, (left, 0))
            frames = sp.unsqueeze(1).unfold(-1, c.win, c.hop).squeeze(1)
            N = spec_s.shape[-1]; yf = []
            for t in range(N):
                ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
                f0, conf = self.core.f0est.estimate(buf)
                smag = ss.abs(); fl = self.core.nf.step(smag)
                snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
                v_prime, startup, reset = self.core.eq.step(ss, vs, snr, conf)
                eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8)) - 20 * torch.log10(vs.abs().clamp_min(1e-8)) - self.core.eq.C).mean(-1) if self.core.eq.C is not None else torch.zeros_like(snr)
                g = self.core.gf0.step(conf)
                wb = self.core.wband.step(v_prime, ss)
                pv = None if pv_seq is None else pv_seq[t]
                wl = self.core.wlocal.step(ss, v_prime, f0, pv_override=pv)
                c_v = self.core.cv.step(v_prime, ss, eqr, bool(reset.any()))
                w_raw = c_v.unsqueeze(-1) * g.unsqueeze(-1) * wb * wl
                fw = torch.maximum(startup, reset.float())
                w = self.core.smooth.step(w_raw * (1 - fw).unsqueeze(-1))
                yf.append(self.core.synth.step(ss, v_prime, w))
            return istft_batch(torch.stack(yf, -1), c, length=s.shape[-1])

    Y = _PvMut(cfg).process_batch(S, vpu)
    spec_Y = stft_batch(Y, cfg)
    bz = cfg.sr / cfg.n_fft
    lsd_sx = lsd_yx = 0.0; n = 0
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        xs = 20 * torch.log10(spec_X[0, lo:hi + 1].abs().clamp_min(1e-8))
        ss = 20 * torch.log10(spec_S[0, lo:hi + 1].abs().clamp_min(1e-8))
        ys = 20 * torch.log10(spec_Y[0, lo:hi + 1].abs().clamp_min(1e-8))
        for t in range(spec_S.shape[-1]):
            if float(conf_tr[0, t]) < 0.55: continue
            if (ss[:, t] - xs[:, t]).mean().item() < -6.0:
                lsd_sx += ((ss[:, t] - xs[:, t]) ** 2).mean().item()
                lsd_yx += ((ys[:, t] - xs[:, t]) ** 2).mean().item(); n += 1
    r_sx = np.sqrt(lsd_sx / max(1, n)); r_yx = np.sqrt(lsd_yx / max(1, n))   # KR4: no *10
    return r_sx, r_yx, r_yx / max(1e-3, r_sx)


def test_JR1_w_local_band_uses_V_time_axis():
    """JR1: redo the band-level ER1 control on the TIME axis (the B-axis
    const/shuffle were no-ops at B=1).  Three controls via a mutant that
    perturbs Pv[t] across frames: const-longterm (smooth), shuffle-time
    (destroy V↔S-suppression time correspondence), fixed-arbitrary (no V).
    🔴 if ③ vs real ratio diff <0.05 ⇒ w_local_band doesn't need V ⇒ DELETE."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=10.0)  # G3a' best depth
    print(f"  JR1 band-level ER1 (TIME-axis, depth10 suppressed bands):")
    print(f"  {'mode':16s} {'LSD(S,X)':>9} {'LSD(Y,X)':>9} {'ratio':>7}")
    ratios = {}
    for mode in ["real", "const-longterm", "shuffle-time", "fixed-arbitrary"]:
        rs, ry, r = _g3a_ratio(cfg, ff, vpu, deg, mode)
        ratios[mode] = r
        print(f"  {mode:16s} {rs:>9.2f} {ry:>9.2f} {r:>7.3f}")
    diff3 = abs(ratios["fixed-arbitrary"] - ratios["real"])
    if diff3 < 0.05:
        print(f"  JR1 ③: fixed-arbitrary ≈ real (Δ={diff3:.3f}<0.05) ⇒ w_local_band "
              f"does NOT need V ⇒ CANDIDATE FOR DELETION (w_band MSC sole).")
    else:
        print(f"  JR1 ③: fixed-arbitrary ≠ real (Δ={diff3:.3f}≥0.05) ⇒ w_local_band "
              f"USES V's level — KEEP (do not delete). ①② distinguish level vs time-variation.")
    # ①②: does time-variation matter?
    if abs(ratios["shuffle-time"] - ratios["real"]) > 0.05:
        print(f"    ② shuffle-time ≠ real ⇒ V's TIME-variation (level↔suppression "
              f"correspondence) matters; not just the level.")
    else:
        print(f"    ② shuffle-time ≈ real ⇒ only V's level matters, not its time-variation.")


# ================================================================ JR2 ======
@lru_cache(maxsize=1)
def _measure_JR2_intervention_metrics():
    """JR2: the MIRROR of 'can't get worse' — 'must actually intervene'.
    🔴 corr computed FROM THE ACTUAL Y (20log|Y|−20log|S|), NOT a manual-loop
    w (the previous manual loop's w ≠ Fusion.process_batch's w ⇒ corr假低 ⇒
    J1假0 — KR0 caught this).  corr-from-Y is config-consistent with G3a'.
    J1 coverage (suppressed, |corr|>3dB, depth≥10 ≥0.50); J2 false (unsup,
    ≤0.10); J3 recovery (≥0.30).  Full dist, depth axis."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); _, conf_tr = f0_batch(ff, cfg)
    print(f"  JR2 intervention metrics (corr=20log|Y|−20log|S|, from actual Y):")
    print(f"  {'depth':>5} {'J1cov':>6} {'J2false':>7} {'J3rec':>6}  (J1≥.50@d≥10 / J2≤.10 / J3≥.30)")
    rows = []
    for d in [0, 3, 6, 10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        spec_X, spec_S, S = eval_specs(ff, cfg, deg)
        Y = _Y(cfg, ff, S, vpu); spec_Y = stft_batch(Y, cfg)
        sup_c = []; unsup_c = []; sup_def = []; sup_rec = []
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            for t in range(spec_S.shape[-1]):
                if float(conf_tr[0, t]) < 0.55: continue
                xs = 20 * torch.log10(spec_X[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ss = 20 * torch.log10(spec_S[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ys = 20 * torch.log10(spec_Y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                corr = (ys - ss).abs().mean().item()            # |corr| per band-frame
                deficit = (xs - ss).abs().mean().item()           # |S−X| (deficit)
                if (ss - xs).mean().item() < -6.0:               # suppressed
                    sup_c.append(corr); sup_def.append(deficit)
                    sup_rec.append(min(corr, deficit))
                else:
                    unsup_c.append(corr)
        j1 = np.mean([c > 3.0 for c in sup_c]) if sup_c else 0.0
        j2 = np.mean([c > 3.0 for c in unsup_c]) if unsup_c else 0.0
        j3 = (np.sum(sup_rec) / max(1, np.sum(sup_def))) if sup_def else 0.0
        rows.append(dict(depth=d, j1=float(j1), j2=float(j2), j3=float(j3)))
        print(f"  {d:>5} {j1:>6.2f} {j2:>7.2f} {j3:>6.2f}")
    print(f"  (corr from actual Y — config-consistent with G3a'; KR0 cross-check in test_KR0)")
    return rows


def test_J1_intervention_coverage():
    """J1: intervention coverage >=0.50 at every evaluated depth>=10."""
    rows = _measure_JR2_intervention_metrics()
    j1_ok = all(r["j1"] >= 0.50 for r in rows if r["depth"] >= 10)
    assert j1_ok, "J1: intervention coverage falls below 0.50 at depth>=10"


def test_J2_false_intervention():
    """J2: false-intervention rate <=0.10 at every depth."""
    rows = _measure_JR2_intervention_metrics()
    j2_ok = all(r["j2"] <= 0.10 for r in rows)
    assert j2_ok, "J2: false-intervention rate exceeds 0.10 at one or more depths"


def test_J3_energy_recovery():
    """J3: energy-recovery ratio >=0.30 at every evaluated depth>=10."""
    rows = _measure_JR2_intervention_metrics()
    j3_ok = all(r["j3"] >= 0.30 for r in rows if r["depth"] >= 10)
    assert j3_ok, "J3: energy-recovery ratio falls below 0.30 at depth>=10"


def test_KR0_cross_check():
    """KR0: Y=S·10^(corr/20), ∠Y=∠S ⇒ per-bin log-change=corr EXACTLY ⇒
    |LSD(Y,X)−LSD(S,X)| ≤ max|corr| (per-bin max, same band-frame).
    A cross-check that auto-catches the J2-manual-loop bug class (corr假低 while
    G3a' shows recovery — mutually impossible).  Mutation: compute corr from a
    manual loop whose w ≠ the real Fusion w ⇒ corr假低 ⇒ inequality FAILS."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); _, conf_tr = f0_batch(ff, cfg)
    print(f"  KR0 cross-check |LSD(Y,X)−LSD(S,X)| ≤ max|corr| (per-bin, suppressed band-frames):")
    print(f"  {'depth':>5} {'imp':>6} {'max|corr|':>9} {'ok':>4}")
    all_ok = True
    for d in [6, 10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        spec_X, spec_S, S = eval_specs(ff, cfg, deg)
        Y = _Y(cfg, ff, S, vpu); spec_Y = stft_batch(Y, cfg)
        worst = 0.0; n = 0; imp_sum = 0
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            for t in range(spec_S.shape[-1]):
                if float(conf_tr[0, t]) < 0.55: continue
                xs = 20 * torch.log10(spec_X[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ss = 20 * torch.log10(spec_S[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ys = 20 * torch.log10(spec_Y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                if (ss - xs).mean().item() < -6.0:
                    lsx = torch.sqrt(((ss - xs) ** 2).mean()).item()
                    lyx = torch.sqrt(((ys - xs) ** 2).mean()).item()
                    maxc = (ys - ss).abs().max().item()    # per-bin MAX |corr|
                    imp = lsx - lyx
                    worst = max(worst, imp - maxc); n += 1; imp_sum += imp
        ok = worst <= 0.1  # improvement ≤ max|corr| (tol 0.1)
        all_ok = all_ok and ok
        print(f"  {d:>5} {imp_sum/max(1,n):>6.2f} {'—':>9} {'✓' if ok else '✗'} (worst over-bounds={worst:.2f})")
    assert all_ok, f"KR0: |improvement| > max|corr| somewhere (the J2-bug signature)"


def test_KR0_mutation():
    """Mutation: completely disconnect the reported corr path (corr≡0) while
    Y still changes.  The unchanged 0.1 dB KR0 bound must fail.  The former
    0.1× mutation lost teeth after A3 removed the much larger roundtrip credit."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=4.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); _, conf_tr = f0_batch(ff, cfg)
    deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
    spec_X, spec_S, S = eval_specs(ff, cfg, deg)
    Y = _Y(cfg, ff, S, vpu); spec_Y = stft_batch(Y, cfg)
    # MUTATION: reported corr path is disconnected even though the real Y moves.
    broken = False
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        for t in range(spec_S.shape[-1]):
            if float(conf_tr[0, t]) < 0.55: continue
            xs = 20 * torch.log10(spec_X[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_S[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_Y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            if (ss - xs).mean().item() < -6.0:
                lsx = torch.sqrt(((ss - xs) ** 2).mean()).item()
                lyx = torch.sqrt(((ys - xs) ** 2).mean()).item()
                fake_maxc = 0.0 * (ys - ss).abs().max().item()   # disconnected corr
                if abs(lsx - lyx) > fake_maxc + 0.1:
                    broken = True; break
        if broken:
            break
    print(f"  KR0 mutation (corr path disconnected, factor 0.1→0): inequality violated? "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught — PROBLEM'}")
    assert broken, "KR0 mutation: disconnected corr did not break the inequality"


# ================================================================ HR3-per-depth
def test_HR3_g7_per_depth():
    """HR3 small correction: G7 ratio (LSD(∠S,∠X) / LSD(S,X)) per depth —
    the ratio diverges at depth=0 (LSD(S,X)→0); report the curve, not one point."""
    _need()
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    print(f"  HR3 G7 phase-pricing ratio per depth:")
    print(f"  {'depth':>5} {'LSD(S,X)':>9} {'phase_gap':>10} {'ratio':>7}")
    for d in [0, 3, 6, 10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        S = _d1_only(ff, cfg, deg) if d > 0 else ff
        YS = _Y(cfg, ff, S, vpu)
        spec_S = stft_batch(S, cfg); spec_X = stft_batch(ff, cfg); YS_spec = stft_batch(YS, cfg)
        YX_spec = YS_spec.abs() * torch.exp(1j * torch.angle(spec_X))
        YX = istft_batch(YX_spec, cfg, length=ff.shape[-1])
        lo, hi = _band_bins(cfg, 100, 2000)
        gap = _lsd_db(stft_batch(YS, cfg), stft_batch(YX, cfg), lo, hi)
        ref = _lsd_db(stft_batch(S, cfg), stft_batch(ff, cfg), lo, hi)
        ratio = gap / ref if ref > 1e-6 else float("inf")
        print(f"  {d:>5} {ref:>9.3f} {gap:>10.3f} {ratio:>7.2f}")


# ================================================================ DR1 (JR1 ext) ==
def test_DR1_wl_v_perturb_wiring():
    """DR1 extension (JR1): cfg.wl_v_perturb set ⇒ WLocal.v_perturb actually
    wired (the HR4 bug was this flag NOT passed to WLocal ⇒ algorithm read the
    default 'none').  Guards the next no-op-flag bug."""
    _need()
    cfg = FusionConfig().with_switches(wl_v_perturb="fixed-arbitrary")
    core = FusionCore(cfg)
    wired = core.wlocal.v_perturb == "fixed-arbitrary"
    print(f"  DR1 wl_v_perturb wiring: cfg=fixed-arbitrary ⇒ WLocal.v_perturb="
          f"{core.wlocal.v_perturb!r} → {'PASS (wired)' if wired else 'FAIL (no-op!)'}")
    assert wired, "DR1: cfg.wl_v_perturb not wired into WLocal (HR4 bug class)"


def test_DR1_wl_v_perturb_mutation():
    """Mutation: construct WLocal WITHOUT passing cfg.wl_v_perturb (the HR4 bug)
    ⇒ v_perturb stays default 'none' ⇒ the wiring meta-test must FAIL (caught)."""
    _need()
    from fusion.decision import WLocal
    cfg = FusionConfig().with_switches(wl_v_perturb="fixed-arbitrary")
    wl_bad = WLocal(cfg, enabled=cfg.enable_w_local, pure_band=cfg.use_w_local_pure_band)  # ← no v_perturb=
    broken = wl_bad.v_perturb != "fixed-arbitrary"
    print(f"  DR1 mutation (omit v_perturb=): WLocal.v_perturb={wl_bad.v_perturb!r} → "
          f"{'FAIL-of-mutant (caught) PASS' if broken else 'NOT caught — PROBLEM'}")
    assert broken, "DR1 mutation: omitting v_perturb= did not break wiring"


# ================================================================ KR1/KR2 ==
def test_KR1_cv_three_components():
    """KR1: c_V median (healthy) + ablation cv_eqresid_mode='off' (remove EQ-residual
    term) — does the safety still hold?  Reports c_V; the 3 components (e/m/q)
    are computed inside CV.step (SNR/MSC/EQ-bias)."""
    _need()
    import torch.nn.functional as F, numpy as np
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    left = cfg.win - cfg.hop if False else FusionConfig().win - FusionConfig().hop
    for mode in ["bias", "abs", "off"]:
        cfg = FusionConfig().with_switches(cv_eqresid_mode=mode)
        spec_s = stft_batch(ff, cfg); spec_v = stft_batch(vpu, cfg); f0, conf = f0_batch(ff, cfg)
        sp = F.pad(ff, (left, 0)); frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
        core = FusionCore(cfg); cvs = []
        for t in range(min(spec_s.shape[-1], 400)):
            ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
            f0c, confc = core.f0est.estimate(buf); smag = ss.abs(); fl = core.nf.step(smag)
            snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
            vp, _, _ = core.eq.step(ss, vs, snr, confc)
            eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8)) - 20 * torch.log10(vs.abs().clamp_min(1e-8)) - core.eq.C).mean(-1) if core.eq.C is not None else torch.zeros_like(snr)
            cv = core.cv.step(vp, ss, eqr, False)
            if float(confc.mean()) > 0.55: cvs.append(cv)
        print(f"  KR1 cv_eqresid_mode={mode:5s}: c_V median={np.median(cvs):.3f} p10={np.percentile(cvs,10):.3f} p90={np.percentile(cvs,90):.3f} (n={len(cvs)})")
    print(f"  (if 'off' ≈ 'bias' on safety props ⇒ the EQ-residual term is removable — B0.5's MSC-only hypothesis)")


def _load_vpu_noisefloor(T, sr):
    """LR1: load the 0625/FB_FF_TT_VPU_noise_floor.wav VPU channel (idx3) as a
    device-noise proxy, unit-peak normalized.  Returns (1,T) float32 or None."""
    import soundfile as sf, os
    REC = os.environ.get("MIC_REC_ROOT", "/mnt/d/Projects/mic_array_capture/mic_recordings")
    nf_path = f"{REC}/0625/FB_FF_TT_VPU_noise_floor.wav"
    if not os.path.exists(nf_path):
        return None
    nf, nfsr = sf.read(nf_path)
    if nf.ndim > 1:
        nf = nf[:, 3]      # VPU = idx3 (FB/FF/TT/VPU)
    nf = torch.tensor(nf, dtype=torch.float32)
    if nf.shape[-1] < T:
        nf = torch.cat([nf, nf.flip(-1)], -1)
    nf = nf[:T].unsqueeze(0)
    return nf / (nf.abs().max() + 1e-9)


def _v_m3_atten(vpu, nf, scale):
    """LR1 M3 protocol: attenuate V's SPEECH only, keep device noise fixed.
    v_atten = (vpu - nf) * scale + nf  (speech×scale, device-noise floor 1×)."""
    return (vpu - nf) * scale + nf


@lru_cache(maxsize=1)
def _measure_KR2_cv_paired():
    """LR1: c_V four paired criteria under CORRECT protocols.
    K-b: M3 (attenuate V SPEECH only, device noise fixed) — cold-start on FULL
         V (freeze C), THEN attenuate speech; does c_V drop? does C stay frozen?
    K-c: real dropout — replace V speech with 0625 noise_floor (signal gone,
         device noise remains), NOT ×0.001 (which kills the floor → NF tracker
         drops → SNR looks normal).  Freeze-first (cold-start on full V).
    K-a healthy≥0.5 (post-freeze) · K-d joint-scale Δ≤0.05."""
    _need()
    import torch.nn.functional as F, numpy as np
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=8.0, offset_s=1.0)
    T = ff.shape[-1]
    nf = _load_vpu_noisefloor(T, sr)
    src = "real 0625 noise_floor (VPU ch)" if nf is not None else "synthetic (0625 absent)"
    if nf is None:
        g = torch.Generator().manual_seed(7); nf = 0.003 * torch.randn(1, T, generator=g)
    # scale nf to V's stationary floor (10th-pct |V| as floor estimate)
    v_floor = float(torch.quantile(vpu.abs().flatten(), 0.10))
    nf_floor = float(torch.quantile(nf.abs().flatten(), 0.10))
    nf = nf * (v_floor / (nf_floor + 1e-9))
    spec_s = stft_batch(ff, cfg); spec_v_full = stft_batch(vpu, cfg)
    f0, conf = f0_batch(ff, cfg)
    left = cfg.win - cfg.hop; sp = F.pad(ff, (left, 0)); frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
    N = min(spec_s.shape[-1], 700); cold = cfg.eq_coldstart_frames

    def _step(core, spec_s, spec_v, t0, t1, collect=True):
        cvs = []
        for t in range(t0, t1):
            ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
            f0c, confc = core.f0est.estimate(buf); smag = ss.abs(); fl = core.nf.step(smag)
            snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
            vp, _, _ = core.eq.step(ss, vs, snr, confc)
            eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8)) - 20 * torch.log10(vs.abs().clamp_min(1e-8)) - core.eq.C).mean(-1) if core.eq.C is not None else torch.zeros_like(snr)
            cv = core.cv.step(vp, ss, eqr, False)
            if collect and float(confc.mean()) > 0.55: cvs.append(float(cv))
        return cvs

    # K-a: full V, post-freeze c_V (the honest healthy value)
    core = FusionCore(cfg)
    cvs_a = _step(core, spec_s, spec_v_full, 0, N)
    ka_cvs = cvs_a[cold:] if len(cvs_a) > cold else cvs_a
    ka = np.median(ka_cvs) if ka_cvs else 0.0
    print(f"  LR1/K-a (healthy c_V, post-freeze full-V): {ka:.3f} (≥0.5 {'PASS' if ka>=0.5 else 'FAIL'})  [{src}]")

    # K-b: M3 freeze-first — cold-start full V (freeze C), then speech-attenuate
    kb = []; C_drifts = []
    for s in [1.0, 0.707, 0.5, 0.25]:   # 0/-3/-6/-12 dB speech atten
        v_atten = _v_m3_atten(vpu, nf, s)
        spec_v = stft_batch(v_atten, cfg)
        core = FusionCore(cfg)
        _step(core, spec_s, spec_v_full, 0, cold + 250, collect=False)   # freeze on FULL V
        C_before = core.eq.C.clone(); frozen_now = core.eq.frozen
        cvs_b = _step(core, spec_s, spec_v, cold + 250, N)               # attenuated regime
        C_drifts.append((core.eq.C - C_before).abs().max().item())
        kb.append(np.median(cvs_b) if cvs_b else 0.0)
    mono = all(kb[i] >= kb[i+1] - 1e-3 for i in range(len(kb)-1))
    print(f"  LR1/K-b (M3 speech-atten 0/-3/-6/-12, freeze-first): {[round(x,3) for x in kb]} strict↓ {'PASS' if mono else 'FAIL'}")
    print(f"         C frozen during K-b? frozen={frozen_now}  C-drift (max bin) per scale={[round(x,3) for x in C_drifts]} (≈0 ⇒ freeze holds)")

    # K-c: real dropout — replace V speech with device noise floor (freeze-first)
    core = FusionCore(cfg)
    _step(core, spec_s, spec_v_full, 0, cold + 250, collect=False)
    C_before = core.eq.C.clone()
    spec_v_drop = stft_batch(nf, cfg)               # signal gone, device noise remains
    cvs_c = _step(core, spec_s, spec_v_drop, cold + 250, N)
    kc = np.median(cvs_c) if cvs_c else 0.0
    C_drift_c = (core.eq.C - C_before).abs().max().item()
    print(f"  LR1/K-c (real dropout, V→noise_floor, freeze-first): {kc:.3f} (≤0.05 {'PASS' if kc<=0.05 else 'FAIL'})  C-drift={C_drift_c:.3f}")
    if kc > 0.05:
        print(f"         ⚠ K-c fails under correct protocol ⇒ prescription: NF tracker long-hold (τ≫dropout), not an abs level gate.")

    # K-d: JOINT scale (S & V both ×s) invariant Δ≤0.05
    def run_joint(s):
        core = FusionCore(cfg); cvs = []
        spec_s = stft_batch(ff * s, cfg); spec_v = stft_batch(vpu * s, cfg)
        sp = F.pad(ff * s, (left, 0)); frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
        for t in range(min(spec_s.shape[-1], 400)):
            ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
            f0c, confc = core.f0est.estimate(buf); smag = ss.abs(); fl = core.nf.step(smag)
            snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
            vp, _, _ = core.eq.step(ss, vs, snr, confc)
            eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8)) - 20 * torch.log10(vs.abs().clamp_min(1e-8)) - core.eq.C).mean(-1) if core.eq.C is not None else torch.zeros_like(snr)
            cv = core.cv.step(vp, ss, eqr, False)
            if float(confc.mean()) > 0.55: cvs.append(cv)
        return np.median(cvs) if cvs else 0.0
    kd = [run_joint(s) for s in [1.0, 0.5, 0.25, 0.1]]
    spread = max(kd) - min(kd)
    print(f"  LR1/K-d (joint-scale 0/-6/-12/-20 c_V): {[round(x,3) for x in kd]} spread={spread:.3f} (≤0.05 {'PASS' if spread<0.05 else 'FAIL'})")
    return dict(ka=float(ka), kb=[float(x) for x in kb], kb_mono=bool(mono),
                kc=float(kc), kd=[float(x) for x in kd], kd_spread=float(spread),
                c_drifts=[float(x) for x in C_drifts], kc_c_drift=float(C_drift_c))


def test_Ka_cv_healthy():
    """K-a: healthy post-freeze median c_V>=0.5."""
    metrics = _measure_KR2_cv_paired()
    ka_ok = metrics["ka"] >= 0.5
    assert ka_ok, f"K-a: healthy median c_V={metrics['ka']:.3f}<0.5"


def test_Kb_cv_monotone():
    """K-b: M3 speech-only attenuation gives monotone non-increasing c_V."""
    metrics = _measure_KR2_cv_paired()
    kb_ok = metrics["kb_mono"]
    assert kb_ok, f"K-b: c_V sequence is not monotone: {metrics['kb']}"


def test_Kc_cv_dropout():
    """K-c: real VPU-noise-floor dropout median c_V<=0.05."""
    metrics = _measure_KR2_cv_paired()
    kc_ok = metrics["kc"] <= 0.05
    assert kc_ok, f"K-c: dropout median c_V={metrics['kc']:.3f}>0.05"


def test_Kd_cv_joint_scale():
    """K-d: joint S/V scaling changes median c_V by at most 0.05."""
    metrics = _measure_KR2_cv_paired()
    kd_ok = metrics["kd_spread"] <= 0.05
    assert kd_ok, f"K-d: joint-scale c_V spread={metrics['kd_spread']:.3f}>0.05"


def test_LR2_eq_freeze_check():
    """LR2: is EQ C actually FROZEN after cold-start?  Does the watchdog mis-fire
    on V-atten (M3)?  Track C[t], converged/frozen flags, reset_count over a
    full-V cold-start then an M3 speech-attenuation phase.  Then re-measure the
    c_V 3-component distribution under the (confirmed) frozen regime."""
    _need()
    import torch.nn.functional as F, numpy as np
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=8.0, offset_s=1.0)
    T = ff.shape[-1]
    nf = _load_vpu_noisefloor(T, sr)
    if nf is None:
        g = torch.Generator().manual_seed(7); nf = 0.003 * torch.randn(1, T, generator=g)
    v_floor = float(torch.quantile(vpu.abs().flatten(), 0.10)); nf_floor = float(torch.quantile(nf.abs().flatten(), 0.10))
    nf = nf * (v_floor / (nf_floor + 1e-9))
    spec_s = stft_batch(ff, cfg); spec_v_full = stft_batch(vpu, cfg)
    f0, conf = f0_batch(ff, cfg)
    left = cfg.win - cfg.hop; sp = F.pad(ff, (left, 0)); frames = sp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
    N = min(spec_s.shape[-1], 700); cold = cfg.eq_coldstart_frames
    core = FusionCore(cfg)
    C_track = []; frozen_track = []; reset_count = 0
    # phase 1: full-V cold-start → freeze
    for t in range(cold + 250):
        ss = spec_s[:, :, t]; vs = spec_v_full[:, :, t]; buf = frames[:, t, :]
        f0c, confc = core.f0est.estimate(buf); smag = ss.abs(); fl = core.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
        _, _, reset = core.eq.step(ss, vs, snr, confc)
        if bool(reset.any()): reset_count += 1
        C_track.append(core.eq.C.clone()); frozen_track.append(core.eq.frozen)
    freeze_idx = next((i for i, f in enumerate(frozen_track) if f), None)
    C_at_freeze = C_track[freeze_idx] if freeze_idx is not None else None
    # phase 2: M3 speech-atten −6 dB — does C stay frozen? does watchdog fire?
    v_atten = _v_m3_atten(vpu, nf, 0.5); spec_v = stft_batch(v_atten, cfg)
    C_post = []; reset_post = 0
    for t in range(cold + 250, N):
        ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
        f0c, confc = core.f0est.estimate(buf); smag = ss.abs(); fl = core.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
        _, _, reset = core.eq.step(ss, vs, snr, confc)
        if bool(reset.any()): reset_post += 1
        C_post.append(core.eq.C.clone())
    drift = max((c - C_at_freeze).abs().max().item() for c in C_post) if C_post and C_at_freeze is not None else float('nan')
    print(f"  LR2 EQ-freeze check (full-V cold-start, then M3 −6 dB speech-atten):")
    print(f"         freeze at frame {freeze_idx} (cold={cold})  frozen-flag post-freeze: {all(frozen_track[freeze_idx:])}")
    print(f"         watchdog resets: cold-start={reset_count}  post-freeze(M3 atten)={reset_post}  (post-freeze>0 ⇒ watchdog MIS-fires on V-atten)")
    print(f"         C drift from freeze (max bin, post-freeze): {drift:.4f} dB (≈0 ⇒ C frozen; the bias term is meaningful)")
    if reset_post > 0:
        print(f"         ⚠ LR2 BUG: watchdog unfroze C on V-atten ⇒ bias term structurally→0 ⇒ KR1 dead.  Investigate cp_eqres_jump / cp_msc_jump thresholds.")
    else:
        print(f"         ✓ C stays frozen on V-atten ⇒ KR1 long-term bias is meaningful (measures relationship drift).")
    # 3-component c_V under confirmed frozen regime (re-run KR1 with the freeze-first core)
    # — re-measure by collecting (e,m,q) proxies post-freeze on full V
    e_l = []; m_l = []; q_l = []; cv_l = []
    core2 = FusionCore(cfg)
    _step2 = lambda core, ss, vs, buf: None
    for t in range(N):
        ss = spec_s[:, :, t]; vs = spec_v_full[:, :, t]; buf = frames[:, t, :]
        f0c, confc = core2.f0est.estimate(buf); smag = ss.abs(); fl = core2.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) / fl.clamp_min(1e-8))).mean(-1)
        vp, _, _ = core2.eq.step(ss, vs, snr, confc)
        eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8)) - 20 * torch.log10(vs.abs().clamp_min(1e-8)) - core2.eq.C).mean(-1) if core2.eq.C is not None else torch.zeros_like(snr)
        cv = core2.cv.step(vp, ss, eqr, False)
        if t >= cold and float(confc.mean()) > 0.55:
            # component proxies: e_term≈SNR-sigmoid, m_term≈MSC, q_term≈exp(-|bias|/6)
            lo, hi = core2.cv._band_bins(); vb = vp[:, lo:hi+1]
            e_v = (vb.abs()**2).mean(-1, keepdim=True); from fusion.utils import causal_ema
            core2.cv.e_v_ema = causal_ema(core2.cv.e_v_ema, e_v, core2.cv.a_e) if core2.cv.e_v_ema is not None else e_v
            e_db = 10.0*torch.log10(core2.cv.e_v_ema.clamp_min(1e-10))
            bin_db = 10.0*torch.log10((vb.abs()**2).clamp_min(1e-12)); nff = torch.quantile(bin_db, cfg.cv_nf_quantile, dim=-1, keepdim=True)
            core2.cv.nf_ema = causal_ema(core2.cv.nf_ema, nff, core2.cv.a_nf) if core2.cv.nf_ema is not None else nff
            snr_db = (e_db - core2.cv.nf_ema).clamp_min(0.0); e_t = torch.sigmoid((snr_db - cfg.cv_snr_ref_db)/cfg.cv_snr_scale_db).clamp(0,1)
            msc = torch.stack([core2.cv.coh.update(vp[0], ss[0])]); m_t = msc[:, lo:hi+1].mean(-1, keepdim=True).clamp(0,1)
            r = eqr.unsqueeze(-1) if eqr.dim()==1 else eqr[:, lo:hi+1].mean(-1, keepdim=True)
            from fusion.utils import alpha_from_tau; ab = alpha_from_tau(cfg.cv_bias_tau_s, cfg.hop, cfg.sr)
            core2.cv.bias_ema = (1-ab)*core2.cv.bias_ema + ab*r if hasattr(core2.cv,'bias_ema') and core2.cv.bias_ema is not None else r
            q_t = torch.exp(-core2.cv.bias_ema.abs()/6.0).clamp(0,1)
            e_l.append(float(e_t.mean())); m_l.append(float(m_t.mean())); q_l.append(float(q_t.mean())); cv_l.append(float(cv))
    print(f"  LR2 c_V 3-component (frozen regime, post-freeze voiced, n={len(cv_l)}):")
    for lab, arr in [("e_term(SNR)", e_l), ("m_term(MSC)", m_l), ("q_term(EQ-bias)", q_l), ("c_V", cv_l)]:
        a = np.array(arr); print(f"         {lab:16s}: med={np.median(a):.3f} p10={np.percentile(a,10):.3f} p90={np.percentile(a,90):.3f}")


def test_LR4_j2_corr_distribution():
    """LR4: on the J2 false-intervention band-frames (UNSUPPRESSED where
    |corr|>3 dB), report the |corr| distribution — analysis, not a knob.
    3–5 dB ⇒ harmless marginal; >10 dB ⇒ real problem worth investigating."""
    _need()
    import numpy as np
    cfg = FusionConfig()
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg); _, conf_tr = f0_batch(ff, cfg)
    print(f"  LR4 J2 false-intervention |corr| distribution (unsup band-frames, |corr|>3dB):")
    print(f"  {'depth':>5} {'n_false':>8} {'3-5dB':>7} {'5-10dB':>7} {'>10dB':>7} {'max':>6}")
    for d in [10, 15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(d))
        spec_X, spec_S, S = eval_specs(ff, cfg, deg)
        Y = _Y(cfg, ff, S, vpu); spec_Y = stft_batch(Y, cfg)
        false_corr = []
        for i in range(len(BAND_EDGES_HZ) - 1):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
            for t in range(spec_S.shape[-1]):
                if float(conf_tr[0, t]) < 0.55: continue
                xs = 20 * torch.log10(spec_X[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ss = 20 * torch.log10(spec_S[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                ys = 20 * torch.log10(spec_Y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
                if (ss - xs).mean().item() >= -6.0:   # UNSUPPRESSED (no deficit to repair)
                    corr = (ys - ss).abs().mean().item()
                    if corr > 3.0: false_corr.append(corr)
        a = np.array(false_corr) if false_corr else np.array([0.0])
        b35 = ((a >= 3) & (a < 5)).sum(); b510 = ((a >= 5) & (a < 10)).sum(); b10 = (a >= 10).sum()
        print(f"  {d:>5} {len(false_corr):>8} {int(b35):>7} {int(b510):>7} {int(b10):>7} {a.max():>6.2f}")
    print(f"  (if 3-5 dB dominates ⇒ harmless marginal; if >10 dB bucket non-empty ⇒ investigate conditions)")


# ================================================================ A1-0 meta
GATE_ASSERT_SPECS = {
    "G1": (test_G1_no_damage_clean, ("ok",)),
    "G2": (test_G2_dropout_fallback, ("steady_ok", "steps_ok")),
    "G3a'": (test_G3aprime_recovery_curve, ("exists",)),
    "G3b'": (test_G3bprime_out_of_band, ("ok_all",)),
    "G4'": (test_G4prime_depth_sweep, ("g4_all",)),
    "G5": (test_G5_causal_phase_change, ("y_full", "leak1")),
    "G6": (test_G6_depth_sweep, ("g6_all",)),
    "J1": (test_J1_intervention_coverage, ("j1_ok",)),
    "J2": (test_J2_false_intervention, ("j2_ok",)),
    "J3": (test_J3_energy_recovery, ("j3_ok",)),
    "K-a": (test_Ka_cv_healthy, ("ka_ok",)),
    "K-b": (test_Kb_cv_monotone, ("kb_ok",)),
    "K-c": (test_Kc_cv_dropout, ("kc_ok",)),
    "K-d": (test_Kd_cv_joint_scale, ("kd_ok",)),
}


def _assert_predicate_names(source):
    """Return names used by real ``assert`` predicates; comments never enter AST."""
    import ast, textwrap
    tree = ast.parse(textwrap.dedent(source))
    predicates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            predicates.append({n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)})
    return predicates


def _validate_gate_source(gate, source, required_names):
    predicates = _assert_predicate_names(source)
    assert predicates, f"A1-0 meta: {gate} has no executable assert (comments do not count)"
    for name in required_names:
        assert any(name in names for names in predicates), (
            f"A1-0 meta: {gate} has no assert referencing criterion variable {name!r}")


def test_A10_assert_presence_meta():
    """A/B/C/D gates contain executable asserts tied to their criterion variables."""
    import inspect
    for gate, (fn, required_names) in GATE_ASSERT_SPECS.items():
        _validate_gate_source(gate, inspect.getsource(fn), required_names)
    print(f"  A1-0 assert meta: {len(GATE_ASSERT_SPECS)} gates have criterion-bound executable asserts")


def test_A10_assert_presence_mutation():
    """Mutation: comment out G1's criterion assert; meta-test must fail."""
    import inspect
    source = inspect.getsource(test_G1_no_damage_clean)
    changed_line = next(line for line in source.splitlines() if line.lstrip().startswith("assert ok"))
    mutant = source.replace(changed_line, "    # MUTATION removed: " + changed_line.strip())
    broken = False; failure = ""
    try:
        _validate_gate_source("G1-mutant", mutant, ("ok",))
    except AssertionError as exc:
        broken = True; failure = str(exc)
    print(f"  A1-0 assert mutation: changed {changed_line.strip()!r}; failure={failure!r}")
    assert broken, "A1-0 mutation: removing G1 criterion assert was not detected"


def test_A10_known_fail_registry():
    """The audit registry is limited to the four reviewer-authorised gates."""
    from fusion.run_t13_tests import KNOWN_FAIL_AUDIT, ACTIVE_KNOWN_FAIL
    gates = {entry["gate"] for entry in KNOWN_FAIL_AUDIT.values()}
    active = {entry["gate"] for entry in ACTIVE_KNOWN_FAIL.values()}
    assert gates == {"G2", "J2", "K-a", "K-c"}, f"unauthorised audit registry gates: {gates}"
    assert active == {"J2", "K-a", "K-c"}, f"unexpected active XFAIL gates: {active}"
    required_fields = {"gate", "measured", "threshold", "reason_source"}
    assert all(set(entry) == required_fields for entry in KNOWN_FAIL_AUDIT.values()), (
        "KNOWN_FAIL audit entries must have exactly four fields")
    print("  A1-0 registry: 4 audited entries; G2 resolved, 3 active XFAILs")


def test_A10_G3a_empty_set_mutation():
    """Mutation sanity: the legacy existential accepted an empty suppression set."""
    rows = [(0, 0.0, 0.0, 0.0, 0, 0.0), (15, 1.0, 0.7, 0.7, 30, 0.0)]
    legacy_pass = any(r[0] <= 20 and r[3] <= 0.5 for r in rows)
    corrected_pass = any(r[0] <= 20 and r[4] >= G3A_MIN_SAMPLES and r[3] <= 0.5
                         for r in rows)
    broken = legacy_pass and not corrected_pass
    print("  G3a' mutation: removed n_sup adequacy guard ⇒ empty set falsely passes; "
          "corrected n_sup>=30 gate rejects")
    assert broken, "G3a' empty-set mutation did not expose the legacy no-op"


if __name__ == "__main__":
    test_G1_no_damage_clean()
    test_G4prime_depth_sweep()
    test_G6_depth_sweep()
    test_G3aprime_recovery_curve()
    test_G3bprime_out_of_band()
    test_G5_causal_phase_change()
    test_G2_dropout_fallback()
    test_G7_phase_pricing()
    test_HR2_zero_w_identity(); test_HR2_mutation()
    test_KR0_cross_check(); test_KR0_mutation()
    test_KR1_cv_three_components()
    test_Ka_cv_healthy(); test_Kb_cv_monotone(); test_Kc_cv_dropout(); test_Kd_cv_joint_scale()
    test_LR2_eq_freeze_check()
    test_LR4_j2_corr_distribution()
    test_JR1_w_local_band_uses_V_time_axis()
    test_J1_intervention_coverage(); test_J2_false_intervention(); test_J3_energy_recovery()
    test_HR3_g7_per_depth()
    test_DR1_wl_v_perturb_wiring(); test_DR1_wl_v_perturb_mutation()
    test_scenario_D2D3D4_all()
    test_scenario_progressive_weakening()
    test_ablation_DR1_meta()
    test_ablation_frozen_vs_adaptive()
    test_listening_pack()
    print("T13-B1 tests: done")
