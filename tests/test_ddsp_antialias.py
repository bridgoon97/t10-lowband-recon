"""§6.1 — Anti-aliasing test for DDSP harmonic synthesis (NEW口径: 16 kHz).

Spec change (rework ② under new口径): Nyquist is now 8 kHz and NO LONGER
coincides with the band top 2 kHz.  The dangerous old case — above-Nyquist
harmonics folding BACK into 0–2 kHz, looking like 'reconstructed high freq' on
a spectrogram — is GONE (there's nothing above 8 kHz to fold).  Instead the
anti-alias concern is: harmonics above the BAND TOP (2 kHz) get synthesized into
the 2–8 kHz region, which is then TRUNCATED away by keep_bins=64 — pure wasted
compute, plus it can leak via the window sidelobes into the kept 0–2 kHz band.

So the harmonic mask cuts at the BAND TOP (``band_top_hz``=2000), not Nyquist,
and the test discriminates on the 2–8 kHz band (the old 4 kHz test discriminated
on the 0–2 kHz inter-harmonic bins via folding; that no longer happens at 16 kHz).

Why these gates (f0=150, sr=16k, n_fft=512, Hann, 13 active harmonics; spectrum
= time-AVERAGED magnitude; reference peak = max over 0–2 kHz harmonic bins):

  * MAX bin in 2–8 kHz / peak  <  -20 dB
    Masked: the >2 kHz harmonics are zeroed, so the 2–8 kHz band holds only the
    Hann sidelobe leakage from the <2 kHz harmonics (measured -28.5 dB).  Mask
    bypassed: the >2 kHz harmonics (k=14..32 → 2100–4800 Hz) appear at FULL
    harmonic amplitude in 2–8 kHz (measured +0.0 dB).  -20 dB leaves ≥8.5 dB
    margin on the correct side and catches the bypass by 20 dB.

  * @1125 Hz (midpoint of h7&h8, 0–2 kHz inter-harmonic) / peak  <  -20 dB
    A leakage-floor SANITY (not the discriminator at 16 kHz — there's no folding,
    so this bin stays ~-30 dB with OR without the mask).  Confirms the in-band
    harmonic comb is clean.

Negative test: deliberately bypass the mask → the 2–8 kHz gate MUST fail (the
bypassed >2 kHz harmonics flood the band).  A test that passes when the bug is
present is worthless.
"""
import math

import torch

from lowband.dsp import ddsp as ddsp_mod
from lowband.dsp.stft import StftConfig, stft

SR = 16000
NYQUIST = SR / 2          # 8000 Hz
BAND_TOP = 2000.0         # anti-alias mask cuts here (NOT Nyquist)
N_FFT = 512
T = SR                    # 1 s
F0 = 150.0                # K = floor(2000/150) = 13 harmonics below band top
MAX_HARM = 32
BIN_HZ = SR / N_FFT       # 31.25 Hz/bin

GATE_HI_BAND = -20.0      # MAX bin in 2–8 kHz / peak
GATE_AT_1125 = -20.0      # leakage-floor sanity


def _bin_sets():
    n_bins = N_FFT // 2 + 1  # 257
    freqs = torch.linspace(0, NYQUIST, n_bins)
    harm = set()
    for k in range(1, int(BAND_TOP // F0) + 1):   # k=1..13
        b = round(k * F0 / BIN_HZ)
        harm.update(range(max(0, b - 1), min(n_bins, b + 2)))
    hi_band = list(range(65, n_bins))            # 2–8 kHz (bins 65..256)
    b1125 = round(7.5 * F0 / BIN_HZ)              # 1125 Hz: midpoint h7&h8
    return (torch.tensor(sorted(harm)), torch.tensor(hi_band), b1125, freqs)


def _avg_mag(mask):
    phase = ddsp_mod.accumulate_phase(torch.full((1, T), F0), T, SR)
    amps = torch.ones(1, MAX_HARM, T)
    wav = ddsp_mod.harmonic_synth(phase, amps, mask)
    cfg = StftConfig(n_fft=N_FFT, hop=160, win=480, center=True)
    _, mag = stft(wav, cfg)            # (B, 257, N)
    return mag.mean(dim=-1)[0]        # (257,)


def _metrics(mask, sets):
    harm_idx, hi_idx, b1125, _ = sets
    m = _avg_mag(mask)
    peak = m[harm_idx].max().clamp_min(1e-9)
    db = lambda r: 20.0 * math.log10(r + 1e-12)
    return {
        "hi_band_db": db(m[hi_idx].max().item() / peak.item()),
        "at_1125_db": db(m[b1125].item() / peak.item()),
    }


def _assert_no_alias(mask, sets):
    """Raise if any gate violated (used by positive + negative tests)."""
    m = _metrics(mask, sets)
    gates = {"hi_band_db": GATE_HI_BAND, "at_1125_db": GATE_AT_1125}
    for k, gate in gates.items():
        assert m[k] < gate, (
            f"anti-alias gate violated: {k}={m[k]:.1f} dB >= {gate} dB "
            f"(>band-top harmonic leaked into 2–8 kHz)")
    return m


def _mask():
    return ddsp_mod.harmonic_index_mask(torch.tensor([F0]), MAX_HARM, BAND_TOP)


def test_anti_aliasing():
    """Masked synthesis: the 2–8 kHz band is at the leakage floor."""
    sets = _bin_sets()
    print(f"  F0={F0}, Nyquist={NYQUIST}, band_top={BAND_TOP}, max_harm={MAX_HARM}")
    print(f"  Active harmonics (mask at band_top): {_mask().sum().item()} / {MAX_HARM}")
    print(f"  gates: 2-8k band<{GATE_HI_BAND}dB  @1125<{GATE_AT_1125}dB (leakage sanity)")
    m = _assert_no_alias(_mask(), sets)
    print(f"  measured: 2-8k={m['hi_band_db']:.1f}dB  @1125={m['at_1125_db']:.1f}dB  "
          f"— all below gates ✓")


def test_anti_aliasing_negative():
    """Bypass the mask → 2–8 kHz must flood (gates MUST fail).

    A gate that passes with the bug present is worthless.  With the mask off,
    the >2 kHz harmonics fill 2–8 kHz, so _assert_no_alias raises.
    """
    sets = _bin_sets()
    mask_off = torch.ones_like(_mask())
    m = _metrics(mask_off, sets)
    print(f"  mask BYPASSED: 2-8k={m['hi_band_db']:.1f}dB  @1125={m['at_1125_db']:.1f}dB "
          f"(2-8k must exceed gate {GATE_HI_BAND}dB)")
    raised = False
    try:
        _assert_no_alias(mask_off, sets)
    except AssertionError:
        raised = True
    assert raised, (
        "anti-alias gates did NOT fire with the mask bypassed — the test is "
        "ineffective (>2 kHz harmonics should flood the 2–8 kHz band). "
        f"metrics={m}")


def test_phase_precision():
    """§6.2: phase accumulator must not drift over long sequences."""
    Tp = 160000  # 10 s at 16 kHz
    f0 = 150.0
    phase64 = ddsp_mod.accumulate_phase(torch.full((1, Tp), f0), Tp, SR, dtype64=True)
    phase32 = ddsp_mod.accumulate_phase(torch.full((1, Tp), f0), Tp, SR, dtype64=False)
    inc = 2 * math.pi * f0 / SR
    ref = torch.cumsum(torch.full((Tp,), inc, dtype=torch.float64), dim=0) % (2 * math.pi)

    def circ_diff(a, b):
        d = (a - b) % (2 * math.pi)
        return torch.min(d, 2 * math.pi - d)

    err64 = circ_diff(phase64[0].double(), ref).max().item()
    err32 = circ_diff(phase32[0].double(), ref).max().item()
    print(f"  Phase drift after {Tp} samples ({Tp/SR:.1f}s): "
          f"fp64={err64:.2e}  fp32={err32:.2e}")
    assert err64 < 1e-4, f"fp64 phase drift too large: {err64}"
    assert err64 <= err32, "fp64 should be at least as good as fp32"


if __name__ == "__main__":
    test_anti_aliasing()
    test_anti_aliasing_negative()
    test_phase_precision()
    print("anti-alias tests (16 kHz new口径): all PASS")
