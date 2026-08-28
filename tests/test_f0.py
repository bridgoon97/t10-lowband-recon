"""§6.5 — F0 estimation test.

Verify YIN works at 4 kHz on band-limited (500 Hz LP) signals.
This is a prerequisite that must be verified, not assumed.
"""
import math

import numpy as np
import torch

from lowband.dsp.f0 import yin_f0


def test_f0_synthetic():
    """YIN must recover known F0 from a band-limited harmonic signal."""
    sr = 4000
    f0_true = 150.0
    T = 4000
    t = np.arange(T) / sr
    # Harmonic signal
    sig = np.zeros(T)
    for k in range(1, 10):
        sig += (1.0 / k) * np.sin(2 * math.pi * k * f0_true * t)
    # Low-pass to ~500 Hz (only harmonics 1, 2, 3 survive)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 500 / (sr / 2), btype="low")
    sig_lp = filtfilt(b, a, sig).astype(np.float32)

    x = torch.from_numpy(sig_lp).unsqueeze(0)
    f0, prob = yin_f0(x, sr, frame_len=128, f0_min=50, f0_max=400)
    # Take median of voiced frames
    voiced = f0[f0 > 0]
    if len(voiced) > 0:
        f0_est = voiced.median().item()
    else:
        f0_est = 0
    err = abs(f0_est - f0_true) / f0_true
    print(f"  True F0={f0_true}, estimated={f0_est:.1f}, rel_err={err:.2e}")
    assert err < 0.05, f"F0 estimation error {err} too large (>5%)"


def test_f0_half_double_freq():
    """Check for octave errors (half/double frequency)."""
    sr = 4000
    for f0_true in [100, 200, 300]:
        T = 4000
        t = np.arange(T) / sr
        sig = np.zeros(T)
        for k in range(1, min(10, int(sr / 2 / f0_true))):
            sig += (1.0 / k) * np.sin(2 * math.pi * k * f0_true * t)
        x = torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)
        f0, _ = yin_f0(x, sr, frame_len=128, f0_min=50, f0_max=400)
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            f0_est = voiced.median().item()
        else:
            f0_est = 0
        is_half = abs(f0_est - f0_true / 2) < 5
        is_double = abs(f0_est - f0_true * 2) < 10
        is_correct = abs(f0_est - f0_true) < 10
        print(f"  F0={f0_true}: est={f0_est:.1f} "
              f"{'correct' if is_correct else 'OCTAVE-ERR' if (is_half or is_double) else 'WRONG'}")
        assert is_correct, f"F0 {f0_true}: octave/double error (est={f0_est})"


if __name__ == "__main__":
    test_f0_synthetic()
    test_f0_half_double_freq()
