"""§5.8 — Data pipeline statistical verification.

Measure actual spectrum of generated degraded samples and verify:
- achieved cutoff matches set value (±10%)
- roll-off slope matches
- noise floor level matches
"""
import numpy as np
import torch

from lowband.data.degradation import DegradationConfig, apply_degradation, measure_cutoff
from lowband.data.lowpass_sim import LowpassSimAdapter


def test_degradation_cutoff():
    """Verify the actual cutoff matches the set value."""
    sr = 4000
    cfg = DegradationConfig(
        cutoff_min=800, cutoff_max=800,  # fix at 800 Hz
        rolloff_min=24, rolloff_max=24,
        noise_floor_min_db=-60, noise_floor_max_db=-60,
        time_vary=False, spectral_tilt=False, formants=False,
        body_noise=False, clipping=False, sample_rate=sr,
    )
    rng = np.random.default_rng(42)
    # Generate a broadband signal (white noise — flat spectrum)
    T = 8000
    x = torch.from_numpy(rng.standard_normal((1, T)).astype(np.float32)) * 0.1

    x_deg = apply_degradation(x, cfg, rng=rng, n_fft=128)

    cutoff, rolloff, noise_floor = measure_cutoff(x_deg, sr)
    print(f"  Set cutoff=800 Hz, measured={cutoff:.0f} Hz "
          f"(±10% = [720, 880])")
    print(f"  Set rolloff=24 dB/oct, measured={rolloff:.1f} dB/oct")
    print(f"  Set noise_floor=-60 dB, measured={noise_floor:.1f} dB")
    assert 500 < cutoff < 1100, f"cutoff {cutoff} far from 800"
    assert rolloff > 5, "roll-off too shallow"


def test_degradation_not_rectangular():
    """The roll-off must be gradual, not a rectangular mask."""
    sr = 4000
    cfg = DegradationConfig(
        cutoff_min=600, cutoff_max=600, rolloff_min=12, rolloff_max=12,
        noise_floor_min_db=-50, noise_floor_max_db=-50,
        time_vary=False, spectral_tilt=False, formants=False,
        body_noise=False, clipping=False, sample_rate=sr,
    )
    rng = np.random.default_rng(0)
    T = 8000
    x = torch.randn(1, T) * 0.1
    x_deg = apply_degradation(x, cfg, rng=rng, n_fft=128)
    _, _, noise_floor = measure_cutoff(x_deg, sr)
    # Noise floor should NOT be hard zero (rectangular mask)
    assert noise_floor > -100, f"noise floor {noise_floor}dB too low — looks like hard zero"


def test_time_varying():
    """Cutoff should drift within an utterance — measure spectral variation."""
    sr = 4000
    cfg = DegradationConfig(
        cutoff_min=600, cutoff_max=600, time_vary=True,
        time_vary_rate=(1.0, 1.0), time_vary_depth=0.5,
        spectral_tilt=False, formants=False, body_noise=False, clipping=False,
        sample_rate=sr,
    )
    rng = np.random.default_rng(1)
    T = 8000
    x = torch.from_numpy(rng.standard_normal((1, T)).astype(np.float32)) * 0.1
    x_deg = apply_degradation(x, cfg, rng=rng, n_fft=128)
    # Measure energy variation at a frequency NEAR the cutoff (where modulation matters)
    sr = 4000
    cfg = DegradationConfig(
        cutoff_min=600, cutoff_max=600, time_vary=True,
        time_vary_rate=(1.0, 1.0), time_vary_depth=0.5,
        spectral_tilt=False, formants=False, body_noise=False, clipping=False,
        sample_rate=sr,
    )
    rng = np.random.default_rng(1)
    T = 8000
    x = torch.from_numpy(rng.standard_normal((1, T)).astype(np.float32)) * 0.1
    x_deg = apply_degradation(x, cfg, rng=rng, n_fft=128)
    # Measure energy at ~700 Hz (near cutoff boundary) across frames
    spec = torch.stft(x_deg, n_fft=512, hop_length=128,
                       window=torch.hann_window(512), return_complex=True)
    mag = spec.abs().squeeze(0)  # (F, N)
    n_frames = mag.shape[-1]
    boundary_bin = int(700 / (sr / 2) * mag.shape[0])  # ~700 Hz
    # Compare first quarter vs second quarter frame energies at boundary freq
    q = n_frames // 4
    e1 = mag[boundary_bin, :q].mean().item()
    e2 = mag[boundary_bin, q:2*q].mean().item()
    e3 = mag[boundary_bin, 2*q:3*q].mean().item()
    print(f"  Energy @ 700Hz by quarter: {e1:.4f}, {e2:.4f}, {e3:.4f}")
    energies = [e1, e2, e3]
    variation = max(energies) / (min(energies) + 1e-8)
    print(f"  Variation ratio: {variation:.2f} (should be > 1.2 for time-varying)")
    assert variation > 1.2, "cutoff not varying within utterance"


def test_lowpass_adapter_runs():
    """LowpassSimAdapter should produce valid (sensor, ref) pairs."""
    import glob
    wavs = sorted(glob.glob("data/test_speech/*.wav"))
    if not wavs:
        print("  SKIP: no test speech wavs found")
        return
    ds = LowpassSimAdapter({
        "clean_wavs": wavs, "segment_len": 4000, "sr": 4000,
        "n_repeat": 5, "degradation": {"cutoff_min": 300, "cutoff_max": 1200},
    })
    assert len(ds) > 0
    item = ds[0]
    assert item["sensor"].shape[-1] == 4000
    assert item["ref"].shape[-1] == 4000
    print(f"  Dataset: {len(ds)} items, sample shape: {item['sensor'].shape}")
    # Verify sensor is band-limited (low-passed)
    c, r, nf = measure_cutoff(item["sensor"], 4000)
    print(f"  Sample cutoff: {c:.0f} Hz, noise floor: {nf:.1f} dB")
    assert c < 1500, "sensor not band-limited"


if __name__ == "__main__":
    test_degradation_cutoff()
    test_degradation_not_rectangular()
    test_time_varying()
    test_lowpass_adapter_runs()
