"""§5.4 — Single-sample overfit test (spec change: complex spec).

The target + input are chosen so MEMORIZATION is both information-theoretically
possible AND representable by every arm's architecture — otherwise the test
cannot distinguish 'correct implementation' from 'capacity/representation
limit' (review finding F).

Three design fixes vs the old B=8/T=16000 random-amp setup (which NONE of the
arms could drive below 0.1, masking any real bug):

  1. SMALL target — B=1, T=4000 (→ 25 frames).  Complex target = 1×64×25×2 =
     3,200 values.  The smallest arm (C, ~13 K params) has 4.1× headroom, so
     a correct implementation MUST reach ~0 (memory outruns target).

  2. REPRESENTABLE target — a SMOOTH-FORMANT harmonic (Gaussian formant, center
     varying smoothly across frames), not random per-harmonic amps.  Arm A is a
     DDSP VOCODER: its harmonic amps are the spectral ENVELOPE sampled at the
     harmonic bins (``_harmonic_amps`` samples ``env_lin``, which comes from a
     mel filterbank + pseudoinverse — a SMOOTHING path), so it CANNOT represent
     arbitrary per-harmonic amp patterns.  A random-amp target floors A at
     ~0.15 (evidenced: magnitude-only floor, n_mel-invariant, self-overfit
     floor) — a structural limit, not a bug.  A smooth-formant envelope IS
     representable, so the test is fair to A.  (B/C are direct spec regressors
     and can represent any target, so this constraint costs them nothing.)

  3. SELF-overfit — input = the target waveform.  This tests the
     analysis→synthesis PATH (the actual §5.4 bug-detection goal: 'can the
     model reproduce a structured target it is given').  Random-input
     memorization is unfair to vocoder arms (A's envelope-derived amps can't
     map arbitrary input→arbitrary target); self-overfit is the fair, diagnostic
     framing for all three.

Result (uniform threshold 0.1, no per-arm relaxation):
  * A (DDSP):   ~0.003   — overfits (envelope path reproduces the smooth formant).
  * B (CRN):    ~0.02    — overfits (direct regression).
  * C (F-T LSTM): ~0.007  — overfits (LSTM needs more steps @ higher lr; the old
    'C cannot overfit, relax to 0.9' was a TARGET-TOO-BIG + too-few-steps
    artifact, NOT a phase bottleneck — RETRACTED.  See known_issues.md C2.)

This supersedes the (a)/(b) probe in scripts/armc_overfit_probe.py (that probe
ran under the same capacity-confounded big target, so its 'phase bottleneck'
conclusion was misread — the bottleneck was target size, not phase).
"""
import torch

from lowband import build_model
from lowband.dsp.stft import StftConfig, complex_stft_truncated
from lowband.dsp import ddsp as ddsp_mod
from lowband.losses.spectral import SpectralLoss

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64, "band_top_hz": 2000}
SR = 16000
T = 4000          # → 25 frames @hop160; target = 1×64×25×2 = 3,200 complex values
THRESHOLD = 0.1   # loss must drop below 10% of initial (uniform, all arms)
# per-architecture train settings (LSTM is slower + likes higher lr; CRN is
# unstable at high lr).  Threshold/steps are uniform in spirit: each arm gets
# ENOUGH optimization to reach <0.1 if correct; lr differs only by stability.
STEPS = {"arm_a_ddsp": 500, "arm_b_crn": 500, "arm_c_ftlstm": 1000}
LR = {"arm_a_ddsp": 1.0e-2, "arm_b_crn": 2.0e-3, "arm_c_ftlstm": 1.0e-2}


def _smooth_formant_ref(n_harm=32, f0=150.0, n_frames=25):
    """A structured, ALL-arms-representable target: a harmonic whose amps form a
    SMOOTH Gaussian formant whose center drifts across frames (representable by
    Arm A's mel-envelope path AND by B/C direct regression).  Per-frame amps are
    ZOH-upsampled to T (matching A's frame-rate control)."""
    k = torch.arange(1, n_harm + 1, dtype=torch.float32)
    k0 = 4.0 + 3.0 * torch.sin(torch.arange(n_frames, dtype=torch.float32) / 5.0)
    sigma = 3.0
    amps_fr = torch.exp(-((k.unsqueeze(0) - k0.unsqueeze(1)) ** 2)
                        / (2.0 * sigma ** 2)) * 0.5      # (n_frames, n_harm)
    hop = 160
    amps = amps_fr.t().unsqueeze(0).repeat_interleave(T // n_frames, dim=-1)
    # (1, n_harm, T) — ZOH per frame; trim/pad to T
    if amps.shape[-1] < T:
        amps = torch.nn.functional.pad(amps, (0, T - amps.shape[-1]), mode="replicate")
    amps = amps[..., :T]
    phase = ddsp_mod.accumulate_phase(torch.full((1, T), f0), T, SR)
    mask = ddsp_mod.harmonic_index_mask(torch.tensor([f0]), n_harm, 2000.0)
    return ddsp_mod.harmonic_synth(phase, amps, mask)    # (1, T)


def test_overfit_single_batch():
    stft_cfg = StftConfig(n_fft=512, hop=160, win=480, keep_bins=64)
    loss_fn = SpectralLoss()
    torch.manual_seed(0)
    ref = _smooth_formant_ref()                  # (1, T) structured, representable
    ref_spec = complex_stft_truncated(ref, stft_cfg)   # (1, 64, 25)
    x = ref.clone()                              # SELF-overfit (input = target)
    n_target_vals = ref_spec.numel() * 2
    print(f"  target: B=1 T={T} -> {tuple(ref_spec.shape)} = {n_target_vals} "
          f"complex values; input=target (self-overfit); threshold<{THRESHOLD}")
    for arm in ARMS:
        torch.manual_seed(42)
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LR[arm])
        initial_loss = final_loss = None
        for step in range(STEPS[arm]):
            optimizer.zero_grad()
            cond = {"f0": torch.full((1, 100), 150.0)} if arm == "arm_a_ddsp" else None
            out = model(x, cond)
            N = min(out["spec"].shape[-1], ref_spec.shape[-1])
            loss = loss_fn(out["spec"][..., :N], ref_spec[..., :N])["loss"]
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
        ratio = final_loss / (initial_loss + 1e-8)
        status = "PASS" if ratio < THRESHOLD else "FAIL"
        print(f"  {arm}: {n_params:>6d} params ({n_params/n_target_vals:.1f}x), "
              f"steps={STEPS[arm]} lr={LR[arm]:.0e}, loss {initial_loss:.3f} -> "
              f"{final_loss:.3f} (ratio {ratio:.3f}) {status}")
        assert ratio < THRESHOLD, (
            f"{arm} cannot overfit a {n_target_vals}-value representable target "
            f"with {n_params} params ({n_params/n_target_vals:.1f}x headroom) — "
            f"ratio {ratio:.3f} >= {THRESHOLD}: likely a real bug, not capacity")


if __name__ == "__main__":
    test_overfit_single_batch()
