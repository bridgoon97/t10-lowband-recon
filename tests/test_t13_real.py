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
    """R2 mutation sanity on REAL voiced: bidirectional w-EMA leak must be
    detectable.  NOTE: reviewer predicted voiced leak >> white-noise 1.8e-3;
    measured it is SMALLER (voiced w smoother ⇒ bidir backward pass propagates
    less) — still caught (>1e-6).  The w_local-LOOKAHEAD mutation below shows
    voiced gives MORE power for that path specifically."""
    _need()
    from tests.test_t13_streaming import _MutantBidirSmoothFusion
    cfg = FusionConfig()
    s, v, sr = _voiced_SV(seg_s=4.0)
    mutant = _MutantBidirSmoothFusion(cfg)
    y_full = mutant.process_batch(s, v)
    T = s.shape[-1]
    P = T // 2
    s_m = s.clone(); s_m[:, P:] = 0.0
    v_m = v.clone(); v_m[:, P:] = 0.0
    y_m = mutant.process_batch(s_m, v_m)
    K = max(0, P - cfg.win)
    leak = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
    detected = leak > 1e-6
    print(f"  R2 mutation (bidir w-EMA) on REAL voiced: leak={leak:.3e} "
          f"(white-noise ~1.8e-3; SMALLER here — voiced w smoother) → "
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
def test_R4_M1_real_envelope():
    """M1 on a REAL in-band (≤2 kHz) speech harmonic envelope (formants /
    per-harmonic undulation).  D1=40 % kill of the WEAKEST in-band harmonics
    (ground-truth known).  Threshold UNCHANGED (recall ≥0.90 / FAR ≤0.10) —
    REPORT item; honest if it fails, NO tuning.

    (apply_d1 kills across the FULL band 0–8 kHz, whose weakest 40 % land
    entirely above 2 kHz ⇒ in-band nothing is killed and the test measures
    nothing — so the in-band kill is done inline here for a correct exercise.)"""
    _need()
    cfg = FusionConfig()
    cfg.enable_harm_freq_smooth = False
    wl = WLocal(cfg, v_fallback=False, valley=False)
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    spec_X = stft_batch(ff, cfg)                  # (1, Fb, N)
    spec_V = stft_batch(vpu, cfg)
    f0_tr, conf_tr = f0_batch(ff, cfg)            # (1, N)
    bz = cfg.sr / cfg.n_fft
    floor_db = -60.0
    Pk, Ps = [], []
    n_voiced = 0
    N = spec_X.shape[-1]
    for t in range(N):
        if float(conf_tr[0, t]) < 0.55 or float(f0_tr[0, t]) <= 0:
            continue
        n_voiced += 1
        if n_voiced > 250:
            break
        f0 = float(f0_tr[0, t])
        # in-band harmonics (≤ fusion_hi_bin = 2 kHz) with REAL energy
        kb = [(k, b) for k in range(1, 64) for b in [int(round(k * f0 / bz))]
              if 1 <= b <= cfg.fusion_hi_bin]
        if len(kb) < 4:
            continue
        P = [20 * torch.log10(spec_X[0, b, t].abs().clamp_min(1e-8)).item() for k, b in kb]
        real = [i for i, p in enumerate(P) if p > (max(P) - 80.0)]
        if len(real) < 4:
            continue
        order = sorted(real, key=lambda i: P[i])          # weak first
        n_kill = int(round(0.4 * len(real)))
        kill_idx = set(order[:n_kill])
        # build degraded S: kill the weakest in-band harmonics → floor (real, ~0 phase)
        s_spec = spec_X[:, :, t].clone()
        peak_amp = 10 ** (max(P) / 20.0)
        floor_amp = (10 ** (floor_db / 20.0)) * peak_amp
        for i in kill_idx:
            b = kb[i][1]
            s_spec[0, b] = complex(floor_amp, 0.0)
        w = wl.step(s_spec, spec_V[:, :, t], torch.tensor([f0]))[0]
        for i in real:
            b = kb[i][1]
            flagged = w[b].item() > 0.5
            (Pk if i in kill_idx else Ps).append(flagged)
    recall = sum(Pk) / max(1, len(Pk))
    far = sum(Ps) / max(1, len(Ps))
    status = "PASS" if recall >= 0.90 and far <= 0.10 else "BELOW-THRESHOLD (reported, not tuned)"
    print(f"  R4 M1 real in-band envelope: voiced_frames={n_voiced}  "
          f"recall={recall:.3f} (≥0.90)  FAR={far:.3f} (≤0.10)  [{status}]  "
          f"(killed_pts={len(Pk)} surviving_pts={len(Ps)})")
    # NOT a gate — report only (honest if below; no tuning per reviewer).
    return recall, far


if __name__ == "__main__":
    test_R2_future_perturbation_real_voiced()
    test_R2_mutation_real_voiced()
    test_R2_mutation_wlocal_lookahead()
    test_R4_M1_real_envelope()
    print("T13-A real-device rework tests: done")
