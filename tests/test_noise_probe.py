"""T11 §4 — noise-only-band input probe (zero training cost).

If the target device has only noise above ~600 Hz, the network's high input
bins are fed PURE noise.  Probe: fix the speech (<600 Hz), replace ONLY the
>600 Hz noise realization (different seeds), forward, measure output change.

  * output ≈ unchanged ⇒ the network correctly ignores the noise band ✓
  * output changes ⇒ the network uses the noise band as signal ⇒ a robustness
    defect (output breaks when the noise realization changes)

⚠️ Untrained-model caveat: a randomly-initialized net uses ALL bins, so the
output WILL change — the "small diff ⇒ robust" criterion is a TRAINED-model
property.  This test fixes the PROBE MECHANISM (forward works on a noisy input,
two seeds, diff measured) and reports the UNTRAINED baseline; the robustness
verdict is deferred to post-training (gpu_todo).  For a trained model, re-run
and assert rel_diff < a threshold.
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from lowband import build_model

BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64, "band_top_hz": 2000}
SR = 16000
T = 16000
SPLIT_HZ = 600.0
ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]


def _speech_like(rng):
    """A deterministic-ish harmonic 'speech' proxy (f0=120, a few harmonics)."""
    t = np.arange(T) / SR
    sig = np.zeros(T, dtype=np.float32)
    for k in range(1, 6):   # 120, 240, 360, 480, 600 Hz (all < split)
        sig += (0.2 / k) * np.sin(2 * np.pi * 120 * k * t)
    return sig * 0.5


def _bandpass_noise(rng, lo_hz, hi_hz):
    """White noise bandpassed to [lo, hi] Hz."""
    w = rng.standard_normal(T).astype(np.float32)
    b, a = butter(4, [lo_hz / (SR / 2), hi_hz / (SR / 2)], btype="band")
    return filtfilt(b, a, w).astype(np.float32) * 0.1


def _input(seed):
    """Speech (<600 Hz) + noise (>600 Hz); same speech, different noise per seed."""
    rng = np.random.default_rng(seed)
    speech = _speech_like(rng)
    b, a = butter(6, SPLIT_HZ / (SR / 2), btype="low")
    sp_lo = filtfilt(b, a, speech)                          # <600 Hz speech
    noise_hi = _bandpass_noise(rng, SPLIT_HZ, SR / 2 - 100)  # >600 Hz noise
    x = (sp_lo + noise_hi).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def test_noise_only_band_probe():
    """Forward two inputs differing ONLY in >600 Hz noise; report output diff.

    Untrained baseline (the small-diff criterion is trained-model; deferred).
    """
    x1 = _input(seed=1)
    x2 = _input(seed=2)
    print(f"\n  split={SPLIT_HZ} Hz: <600 = same speech, >600 = noise (seed 1 vs 2)")
    for arm in ARMS:
        torch.manual_seed(0)
        model = build_model(dict(BASE_CFG, arm=arm, f0_mode="oracle"))
        cond = {"f0": torch.full((1, 100), 120.0)} if arm == "arm_a_ddsp" else None
        with torch.no_grad():
            out1 = model(x1, cond)["spec"]
            out2 = model(x2, cond)["spec"]
        N = min(out1.shape[-1], out2.shape[-1])
        diff = (out1[..., :N] - out2[..., :N]).abs().max().item()
        base = out1[..., :N].abs().max().item() + 1e-8
        rel = diff / base
        print(f"  {arm}: output rel_diff (seed1 vs seed2) = {rel:.3f}")
        # probe mechanism: forward runs, diff is finite
        assert torch.isfinite(torch.tensor(rel)), f"{arm}: non-finite output"
    print("  → probe MECHANISM works (forward on noisy input, two seeds, diff "
          "measured). 'small diff ⇒ robust' is a TRAINED-model criterion "
          "(untrained uses all bins); deferred to post-training (gpu_todo).")


if __name__ == "__main__":
    test_noise_only_band_probe()
    print("noise-only-band probe: PASS (mechanism)")
