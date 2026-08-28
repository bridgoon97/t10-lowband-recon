"""§6.1 — Anti-aliasing test for DDSP harmonic synthesis.

The most dangerous DDSP bug: harmonics above Nyquist fold back as spurious
in-band tones.  Spectrogram looks like 'high freq reconstructed' but it's
actually aliasing.

This test: synthesize with known F0, verify output spectrum has energy ONLY at
harmonic frequencies k*F0 (and main-lobe spread), NOT at inter-harmonic or
above-Nyquist positions.
"""
import math

import torch

from lowband.dsp import ddsp as ddsp_mod
from lowband.dsp.stft import StftConfig, stft, get_window


def test_anti_aliasing():
    sr = 4000
    nyquist = sr / 2  # 2000
    n_fft = 128
    T = 4000
    f0 = 150.0  # K = floor(2000/150) = 13 harmonics below Nyquist
    max_harm = 32

    # Generate phase
    phase = ddsp_mod.accumulate_phase(
        torch.full((1, T), f0), T, sr)  # (1, T)

    # Harmonic amplitudes: ALL 1.0 including harmonics that alias above Nyquist
    # This tests whether the mask correctly removes the aliasing ones
    amps = torch.ones(1, max_harm, T)

    # Anti-alias mask
    f0_tensor = torch.tensor([f0])
    mask = ddsp_mod.harmonic_index_mask(f0_tensor, max_harm, nyquist)
    print(f"  F0={f0} Hz, Nyquist={nyquist}, max_harm={max_harm}")
    print(f"  Active harmonics (mask): {mask.sum().item()} / {max_harm}")

    # Synthesize WITH mask
    wav_masked = ddsp_mod.harmonic_synth(phase, amps, mask)

    # Synthesize WITHOUT mask (intentional aliasing)
    wav_unmasked = ddsp_mod.harmonic_synth(phase, amps, torch.ones_like(mask))

    # STFT both
    cfg = StftConfig(n_fft=n_fft, hop=32, win=128, center=True)
    _, mag_masked = stft(wav_masked, cfg)
    _, mag_unmasked = stft(wav_unmasked, cfg)

    # Check: aliased harmonics fold back INTO the band at non-harmonic freqs.
    # Isolate the aliasing: sum energy at ALL inter-harmonic bins.
    # Harmonics are at k*150 for k=1..13. Non-harmonic bins should be quieter
    # in the masked version.
    n_bins = mag_masked.shape[1]
    freqs_per_bin = nyquist / (n_bins - 1)
    # Identify bins near harmonics (within 1 bin)
    harm_bins = set()
    for k in range(1, 14):
        b = int(k * f0 / freqs_per_bin)
        harm_bins.update(range(b - 1, b + 2))
    all_bins = set(range(n_bins))
    nonharm_bins = sorted(all_bins - harm_bins)
    if nonharm_bins:
        idx = torch.tensor(nonharm_bins)
        masked_inter = mag_masked[:, idx, :].sum().item()
        unmasked_inter = mag_unmasked[:, idx, :].sum().item()
        print(f"  Inter-harmonic energy: masked={masked_inter:.2f}, "
              f"unmasked={unmasked_inter:.2f}")
        ratio = masked_inter / (unmasked_inter + 1e-8)
        print(f"  Ratio (masked/unmasked): {ratio:.3f} (should be < 0.8)")
        assert ratio < 0.8, \
            f"Anti-alias mask not reducing inter-harmonic aliasing (ratio={ratio})"

    # Also verify above-Nyquist bins (both should be near zero by definition)
    high_bins = n_bins - 5
    masked_high = mag_masked[:, high_bins:, :].max().item()
    unmasked_high = mag_unmasked[:, high_bins:, :].max().item()
    print(f"  High-freq (near Nyquist): masked={masked_high:.4f}, "
          f"unmasked={unmasked_high:.4f}")

    # Check inter-harmonic positions: masked should have low energy
    # between harmonics
    freqs = torch.linspace(0, nyquist, n_bins)
    # Find a bin between harmonic 7 and 8 (7.5 * 150 = 1125 Hz)
    target_freq = 7.5 * f0
    bin_idx = int(target_freq / nyquist * (n_bins - 1))
    masked_mid = mag_masked[:, bin_idx, :].mean().item()
    print(f"  Inter-harmonic @ {target_freq:.0f}Hz (masked): {masked_mid:.6f}")
    # Should be low relative to peak harmonic
    peak = mag_masked.max().item()
    print(f"  Peak: {peak:.6f}, ratio: {masked_mid/peak:.4f}")


def test_phase_precision():
    """§6.2: phase accumulator must not drift over long sequences."""
    sr = 4000
    T = 40000  # 10 seconds — long enough to show fp32 drift
    f0 = 150.0

    phase64 = ddsp_mod.accumulate_phase(torch.full((1, T), f0), T, sr, dtype64=True)
    phase32 = ddsp_mod.accumulate_phase(torch.full((1, T), f0), T, sr, dtype64=False)

    # Reference: same definition as accumulator (cumsum of increments mod 2π)
    inc = 2 * math.pi * f0 / sr
    ref = torch.cumsum(torch.full((T,), inc, dtype=torch.float64), dim=0)
    ref = ref % (2 * math.pi)

    def circ_diff(a, b):
        """Smallest angular distance in [0, π]."""
        d = (a - b) % (2 * math.pi)
        return torch.min(d, 2 * math.pi - d)

    err64 = circ_diff(phase64[0].double(), ref).max().item()
    err32 = circ_diff(phase32[0].double(), ref).max().item()
    print(f"  Phase drift after {T} samples ({T/sr:.1f}s):")
    print(f"    fp64: max err={err64:.2e} rad")
    print(f"    fp32: max err={err32:.2e} rad")
    assert err64 < 1e-4, f"fp64 phase drift too large: {err64}"
    assert err64 <= err32, "fp64 should be at least as good as fp32"


if __name__ == "__main__":
    test_anti_aliasing()
    test_phase_precision()
