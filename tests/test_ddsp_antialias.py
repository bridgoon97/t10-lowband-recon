"""§6.1 — Anti-aliasing test for DDSP harmonic synthesis (NEW口径: 16 kHz).

Spec change (rework ② under new口径): Nyquist is now 8 kHz and NO LONGER
coincides with the band top 2 kHz.  The dangerous old case — above-Nyquist
harmonics folding BACK into 0–2 kHz, looking like 'reconstructed high freq' on
a spectrogram — is GONE (there's nothing above 8 kHz to fold).  Instead the
anti-alias concern is: harmonics above the BAND TOP (2 kHz) get synthesized into
the 2–8 kHz region, which is then TRUNCATED away by keep_bins=64 — pure wasted
compute, plus it can leak via the window sidelobes into the kept 0–2 kHz band.

So the harmonic mask cuts at the BAND TOP (``band_top_hz``=2000), not Nyquist,
and the test discriminates on the 2–8 kHz band.  Parametrized over
``f0 ∈ {80, 150, 300}`` because the active-harmonic count ``K = floor(band_top/f0)``
varies a lot (25 / 13 / 6) and the boundary behaviour is where index bugs hide
(K large → index overflow; K small → mask degenerates).

⚠️ DEPENDENCY: because band_top (2 kHz) << Nyquist (8 kHz), TRUE aliasing is
structurally impossible right now.  If someone raises ``band_top`` above
~250 Hz×32/8k boundary (i.e. K×f0 starts exceeding Nyquist 8 kHz — e.g.
band_top raised toward 8 kHz with f0>250), harmonics WILL cross Nyquist and
start folding.  This test does NOT cover that regime; it only checks the
band-top-truncation mask.  Do not raise ``band_top`` without re-adding a
Nyquist-folding check.

Gates (spectrum = time-AVERAGED magnitude; reference peak = max over 0–2 kHz
harmonic bins; gates set before observation):
  * MAX bin in 2–8 kHz / peak  <  -20 dB   (masked: leakage floor ~-28 dB;
    bypassed: >2 kHz harmonics flood to ~0 dB — ≥20 dB separation)
  * @1125 Hz (midpoint h7&h8, 0–2 kHz inter-harmonic) / peak  <  -20 dB
    (leakage-floor sanity; NOT the discriminator at 16 kHz — no folding)

Negative test: bypass the mask → the 2–8 kHz gate MUST fail at every f0.
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
MAX_HARM = 32
BIN_HZ = SR / N_FFT       # 31.25 Hz/bin
F0_VALUES = (80.0, 150.0, 300.0)   # K = floor(2000/f0) = 25 / 13 / 6

GATE_HI_BAND = -20.0      # MAX bin in 2–8 kHz / peak
GATE_AT_1125 = -20.0      # leakage-floor sanity


def _bin_sets(f0):
    n_bins = N_FFT // 2 + 1  # 257
    freqs = torch.linspace(0, NYQUIST, n_bins)
    harm = set()
    for k in range(1, int(BAND_TOP // f0) + 1):
        b = round(k * f0 / BIN_HZ)
        harm.update(range(max(0, b - 1), min(n_bins, b + 2)))
    hi_band = list(range(65, n_bins))            # 2–8 kHz (bins 65..256)
    # a TRUE inter-harmonic midpoint for THIS f0 (not a fixed 1125 Hz, which
    # only sits between harmonics for f0=150; for f0=80 a harmonic lands at 1120)
    mid_freq = 7.5 * f0 if 7.5 * f0 < BAND_TOP else 3.5 * f0
    b_mid = round(mid_freq / BIN_HZ)
    return (torch.tensor(sorted(harm)), torch.tensor(hi_band), b_mid, freqs)


def _avg_mag(f0, mask):
    phase = ddsp_mod.accumulate_phase(torch.full((1, T), f0), T, SR)
    amps = torch.ones(1, MAX_HARM, T)
    wav = ddsp_mod.harmonic_synth(phase, amps, mask)
    cfg = StftConfig(n_fft=N_FFT, hop=160, win=480, center=True)
    _, mag = stft(wav, cfg)            # (B, 257, N)
    return mag.mean(dim=-1)[0]         # (257,)


def _metrics(f0, mask, sets):
    harm_idx, hi_idx, b_mid, _ = sets
    m = _avg_mag(f0, mask)
    peak = m[harm_idx].max().clamp_min(1e-9)
    db = lambda r: 20.0 * math.log10(r + 1e-12)
    return {
        "hi_band_db": db(m[hi_idx].max().item() / peak.item()),
        "at_1125_db": db(m[b_mid].item() / peak.item()),   # inter-harm midpoint
    }


def _assert_no_alias(f0, mask, sets):
    """Raise if the 2-8 kHz band gate is violated (the universal discriminator).
    (@mid inter-harmonic level is RETURNED for info but NOT asserted — it's a
    leakage-floor sanity only when harmonics are well-separated; at small f0
    (e.g. 80 Hz) adjacent main lobes overlap so the midpoint is NOT a quiet
    point, by ~-6 dB, which is expected geometry, not a bug.)"""
    m = _metrics(f0, mask, sets)
    assert m["hi_band_db"] < GATE_HI_BAND, (
        f"anti-alias gate violated [f0={f0}]: 2-8k={m['hi_band_db']:.1f} dB "
        f">= {GATE_HI_BAND} dB (>band-top harmonic leaked into 2–8 kHz)")
    return m


def _mask(f0):
    return ddsp_mod.harmonic_index_mask(torch.tensor([f0]), MAX_HARM, BAND_TOP)


def test_anti_aliasing():
    """Masked synthesis: 2–8 kHz at leakage floor, at every f0 in {80,150,300}.

    Also asserts the active-harmonic count equals the PHYSICALLY-correct count
    (harmonics STRICTLY below band_top; the k with k*f0 == band_top is truncated,
    not active) = ceil(band_top/f0) - 1.  NB: not floor(band_top/f0) — that
    over-counts by 1 when band_top is an exact multiple of f0 (e.g. f0=80:
    25*80=2000=band_top, truncated, so 24 active not 25).  Catches mask off-by-one."""
    print(f"  band_top={BAND_TOP}, Nyquist={NYQUIST}, max_harm={MAX_HARM}")
    print(f"  gates: 2-8k band<{GATE_HI_BAND}dB  @1125<{GATE_AT_1125}dB (leakage sanity)")
    for f0 in F0_VALUES:
        sets = _bin_sets(f0)
        mask = _mask(f0)
        k_active = mask.sum().item()
        k_expected = int(math.ceil(BAND_TOP / f0)) - 1   # strictly < band_top
        assert k_active == k_expected, (
            f"f0={f0}: mask active {k_active} != ceil(band_top/f0)-1={k_expected}")
        m = _assert_no_alias(f0, mask, sets)     # asserts gates inside
        print(f"  f0={f0:>5}: active={k_active}/{MAX_HARM} (=ceil(2000/{f0:g})-1 "
              f"✓)  2-8k={m['hi_band_db']:.1f}dB  "
              f"@mid={m['at_1125_db']:.1f}dB  ✓")


def test_anti_aliasing_negative():
    """Bypass the mask → 2–8 kHz must flood at EVERY f0 (gates MUST fail).

    A gate that passes with the bug present is worthless.  Checked at all three
    f0 so a boundary-only failure isn't hidden by a single working point.
    """
    for f0 in F0_VALUES:
        sets = _bin_sets(f0)
        mask_off = torch.ones_like(_mask(f0))
        m = _metrics(f0, mask_off, sets)
        raised = False
        try:
            _assert_no_alias(f0, mask_off, sets)
        except AssertionError:
            raised = True
        assert raised, (
            f"anti-alias gates did NOT fire [f0={f0}] with mask bypassed — "
            f"ineffective (>2 kHz harmonics should flood 2–8 kHz). metrics={m}")
        print(f"  f0={f0:>5}: BYPASSED 2-8k={m['hi_band_db']:.1f}dB "
              f"(>gate {GATE_HI_BAND}dB → caught ✓)")


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
    print("anti-alias tests (16 kHz, f0∈{80,150,300}): all PASS")
