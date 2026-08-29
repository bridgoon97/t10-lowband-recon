"""§6.5 / task ② — F0 confidence SOFT gating (no hard voiced/unvoiced decision).

Arm A's YIN path now runs in SOFT-candidate mode: it ALWAYS returns a best F0
candidate in [f0_min, f0_max] (never 0) + a continuous confidence in [0,1]; the
confidence is a SOFT per-subband weight (effective_periodicity = learned *
confidence) that blends the harmonic / noise SPECTRA per-bin.  There is NO
binary voiced/unvoiced branch on Arm A's path.  (Legacy YIN hard-threshold
behavior is kept as the default for backward compat, e.g. test_l1_f0.)

Criteria FIXED before observation (do not relax after running):
  A. soft path: finite in-range candidate + confidence in [0,1], no f0=0;
     effective_periodicity == periodicity * confidence (continuous, no threshold).
  B. confidence bucketing {0,.25,.5,.75,1}: effective_periodicity == periodicity
     * conf (atol 1e-6, rtol 1e-5); harmonic gain (blend_weight_spec) monotonic
     non-decreasing in conf, noise gain (1-blend) non-increasing; conf=0 ⇒ pure
     noise weight (blend=0), conf=1 ⇒ original learned periodicity weight.
  C. low-confidence WRONG F0 ⇒ no structural harmonics.  CHOSEN criterion
     (documented): at conf=0 the ACTUAL output is F0-INVARIANT — two clearly
     different injected F0s (150/300 Hz) yield identical out['spec']
     (max_abs <= 1e-7), because the noise component depends only on input
     envelope + seed, not F0; if the synthesis path leaked harmonic/F0 into
     the conf=0 output this fails.  Non-vacuous contrast: at conf=1 the same
     two F0s differ (max_abs > 1e-3, FIXED before observation).  (The 20 dB
     line-spectrum-drop alternative is NOT used — it depends on untrained
     amplitudes; F0-invariance is amplitude-independent and exact.  The
     previous isolable-component form was a tautology — replaced.)
  D. stream≡batch holds AND with non-1.0 confidence (oracle F0 + injected conf).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lowband import build_model
from lowband.dsp.f0 import yin_f0

BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64, "arm": "arm_a_ddsp",
            "f0_mode": "oracle"}
HOP = 160
N_BINS = 64


def _model():
    return build_model(BASE_CFG).eval()


def _harmonic_sig(f0_hz=150.0, T=16000, sr=16000, n_harm=20):
    t = torch.arange(T) / sr
    sig = torch.zeros(T)
    for k in range(1, n_harm + 1):
        sig += (1.0 / k) * torch.sin(2 * torch.pi * f0_hz * k * t)
    return sig / sig.abs().max()


# ---------------------------------------------------------------------------
def test_A_soft_candidate_no_hard_branch():
    """A. soft YIN returns finite in-range candidate + [0,1] confidence, no f0=0."""
    torch.manual_seed(0)
    # 1) soft path on NOISE: candidate finite in range, confidence in [0,1], no 0
    noise = torch.randn(2, 4800)
    f0, prob = yin_f0(noise, 16000, frame_len=480, f0_min=50.0, f0_max=400.0, soft=True)
    assert torch.isfinite(f0).all(), "soft f0 must be finite"
    assert (f0 >= 50.0).all() and (f0 <= 400.0).all(), "soft f0 in [f0_min,f0_max]"
    assert (f0 != 0).all(), "soft path must NEVER return f0=0 (no unvoiced branch)"
    assert (prob >= 0).all() and (prob <= 1).all(), "confidence in [0,1]"
    # 2) soft path on a HARMONIC signal: confidence should be HIGHER than noise
    sig = _harmonic_sig(150.0, 4800).unsqueeze(0).expand(2, -1)
    f0h, probh = yin_f0(sig, 16000, frame_len=480, f0_min=50.0, f0_max=400.0, soft=True)
    assert torch.isfinite(f0h).all() and (f0h != 0).all()
    assert (probh >= 0).all() and (probh <= 1).all()
    assert probh.mean() > prob.mean(), "harmonic sig ⇒ higher mean confidence than noise"
    # 3) Arm A ESTIMATED path on noise: aux f0 finite in range (no 0), conf in [0,1]
    cfg = dict(BASE_CFG, f0_mode="estimated")
    m = build_model(cfg).eval()
    with torch.no_grad():
        out = m(noise)
    af = out["aux"]["f0"]; ac = out["aux"]["f0_confidence"]
    assert torch.isfinite(af).all() and (af != 0).all() and (af >= 50).all() and (af <= 400).all()
    assert (ac >= 0).all() and (ac <= 1).all()
    # 4) NO thresholding: effective_periodicity == periodicity * confidence (continuous)
    per = out["aux"]["periodicity"]; eff = out["aux"]["effective_periodicity"]
    exp = per * ac.unsqueeze(-1)
    assert torch.allclose(eff, exp, atol=1e-6, rtol=1e-5), \
        "effective_periodicity must equal periodicity*confidence (no threshold branch)"
    print("  A: soft candidate finite + in-range, no f0=0, conf in [0,1], "
          "eff==periodicity*conf (continuous, no binary branch) ✓")


def test_B_confidence_bucketing():
    """B. eff==periodicity*conf across conf∈{0,.25,.5,.75,1}; monotonic; endpoints."""
    torch.manual_seed(1)
    m = _model()
    B, T = 2, 16000
    x = torch.randn(B, T)
    f0 = torch.full((B, 100), 150.0)            # fixed oracle F0
    confs = [0.0, 0.25, 0.5, 0.75, 1.0]
    per0 = None
    blends = {}
    effs = {}
    with torch.no_grad():
        for c in confs:
            cond = {"f0": f0, "f0_confidence": torch.full((B, 100), float(c))}
            out = m(x, cond)
            per = out["aux"]["periodicity"]
            eff = out["aux"]["effective_periodicity"]
            bl = out["aux"]["blend_weight_spec"]
            if per0 is None:
                per0 = per
                # ControlNet output is FIXED across conf (same x) — sanity
            else:
                assert torch.allclose(per, per0, atol=1e-7), "ControlNet output fixed across conf"
            # per-subband equality: eff == per * c  (atol 1e-6, rtol 1e-5)
            assert torch.allclose(eff, per * c, atol=1e-6, rtol=1e-5), \
                f"eff != per*{c} at conf={c}"
            effs[c] = eff
            blends[c] = bl
    # monotonic: harmonic gain (blend) non-decreasing in conf; noise gain (1-blend) non-increasing
    for i in range(len(confs) - 1):
        c0, c1 = confs[i], confs[i + 1]
        db = blends[c1] - blends[c0]
        assert (db >= -1e-6).all(), f"harmonic gain not non-decreasing {c0}->{c1}"
        dn = (1 - blends[c1]) - (1 - blends[c0])
        assert (dn <= 1e-6).all(), f"noise gain not non-increasing {c0}->{c1}"
    # endpoints: conf=0 ⇒ pure noise weight (blend=0); conf=1 ⇒ original learned periodicity
    assert torch.allclose(blends[0.0], torch.zeros_like(blends[0.0]), atol=1e-7), \
        "conf=0 ⇒ blend=0 (pure noise weight)"
    # conf=1 ⇒ eff == periodicity (original learned, unscaled)
    assert torch.allclose(effs[1.0], per0, atol=1e-6, rtol=1e-5), \
        "conf=1 ⇒ effective_periodicity == learned periodicity"
    print("  B: eff==periodicity*conf ∀c; harmonic↑ noise↓ monotonic; "
          "conf=0→pure-noise, conf=1→learned-periodicity ✓")


def test_C_low_conf_wrong_f0_no_structure():
    """C. conf=0 ⇒ ACTUAL output is F0-INVARIANT (harmonic gated); conf=1 ⇒ F0-sensitive.

    INDEPENDENT, falsifiable verification of the ACTUAL output (NOT a tautology):
    the noise component depends only on the input envelope + eval seed, NOT on F0;
    at conf=0 the blend=0 so out_spec = noise_spec and must be IDENTICAL for two
    different injected F0s.  If the synthesis path leaked ANY harmonic/F0 into
    the conf=0 output, the two would differ ⇒ this fails.  (The previous form
    `noise_spec = outs[0]; outs[0] - (1-blend0)*noise_spec` was a tautology — it
    reduced to outs[0]-outs[0]=0 regardless of the real output — and is replaced.)

    Non-vacuous contrast: at conf=1 the same two F0s produce SIGNIFICANTLY
    different outputs (threshold FIXED before observation: max_abs > 1e-3),
    proving the conf=0 invariance is the GATING, not F0-insensitivity.
    """
    torch.manual_seed(2)
    m = _model()
    B, T = 2, 16000
    x = torch.randn(B, T)
    f0s = [150.0, 300.0]                 # two clearly different candidates
    specs = {0.0: {}, 1.0: {}}
    blends0 = {}
    with torch.no_grad():
        for c in [0.0, 1.0]:
            for f in f0s:
                cond = {"f0": torch.full((B, 100), f),
                        "f0_confidence": torch.full((B, 100), float(c))}
                out = m(x, cond)
                specs[c][f] = out["spec"]
                if c == 0.0:
                    blends0[f] = out["aux"]["blend_weight_spec"]
    # --- control layer (NOT a substitute for the actual-output assertions) ---
    for f in f0s:
        assert torch.allclose(blends0[f], torch.zeros_like(blends0[f]), atol=1e-7), \
            f"conf=0 ⇒ blend_weight_spec=0 (F0={f})"
    # --- ACTUAL output: F0-INVARIANCE at conf=0 (independent reference = the
    #     OTHER F0's output, not outs[0] itself) ---
    diff0 = (specs[0.0][150.0] - specs[0.0][300.0]).abs().max()
    assert diff0 <= 1e-7, \
        f"conf=0 output must be F0-INVARIANT (harmonic gated); got max_abs={diff0:.2e} " \
        f"between F0=150 and F0=300 — synthesis path leaked F0 into conf=0 output"
    # --- non-vacuous contrast: conf=1 the two F0s differ (FIXED threshold > 1e-3) ---
    diff1 = (specs[1.0][150.0] - specs[1.0][300.0]).abs().max()
    assert diff1 > 1e-3, \
        f"conf=1 outputs must differ with F0 (else model F0-insensitive, test vacuous); " \
        f"got max_abs={diff1:.2e}"
    print(f"  C: conf=0 F0-invariance max_abs={diff0:.2e} (<=1e-7, harmonic gated); "
          f"conf=1 F0-contrast max_abs={diff1:.2e} (>1e-3, non-vacuous) ✓")


def test_D_stream_batch_equiv_with_confidence():
    """D. stream≡batch holds, incl. non-1.0 confidence (oracle F0 + injected conf)."""
    for conf in [1.0, 0.5, 0.0]:
        m = _model()
        B, T = 2, 16000
        x = torch.randn(B, T)
        cond = {"f0": torch.full((B, 100), 150.0),
                "f0_confidence": torch.full((B, 100), conf)}
        with torch.no_grad():
            out = m(x, cond)
            state = m.stream_init(B)
            state["f0_override"] = torch.full((B,), 150.0)
            state["f0_confidence_override"] = torch.full((B,), conf)
            frames = []
            for i in range(0, T, HOP):
                fr = x[:, i:i + HOP]
                if fr.shape[-1] < HOP:
                    break
                spec_f, state = m.stream_step(fr, state)
                frames.append(spec_f)
            stream_spec = torch.stack(frames, dim=-1)
        N = min(out["spec"].shape[-1], stream_spec.shape[-1])
        diff = (out["spec"][..., :N] - stream_spec[..., :N]).abs()
        rel = diff.max().item() / (out["spec"][..., :N].abs().max().item() + 1e-8)
        assert rel < 1e-4, f"stream≠batch at conf={conf}: rel_err={rel}"
        print(f"  D: stream≡batch conf={conf}: rel_err={rel:.2e} ✓")


if __name__ == "__main__":
    test_A_soft_candidate_no_hard_branch()
    test_B_confidence_bucketing()
    test_C_low_conf_wrong_f0_no_structure()
    test_D_stream_batch_equiv_with_confidence()
    print("F0 soft gating: PASS")
