"""L1 — effective-bandwidth characterization (rework ③-redo, per review).

The old metric (high/low energy ratio) was NOISE-FLOOR-CONTAMINATED: a body
sensor's high band holds its own noise floor, not rolled-off speech, so it
looked far less band-limited than it is (reported ~3 dB, should be tens of dB).

This test uses the two correct metrics:
  * bandwise magnitude-squared coherence MSC(temple, headset) on speech_clean —
    coherence drops where the sensor stops carrying speech; noise is incoherent;
  * noise-floor SNR per band = speech_clean power / speechless_clean power
    (speechless_clean = the sensor's noise-only recordings) — the band where
    SNR<0 dB is where speech is buried in the sensor's own noise.

Also a future wear-quality reliability metric (per the review).  SKIPs if the
local parquet shards are absent.
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from tests._testutil import SkipTest

SPK_SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
              "data/vibravox_parquet/speech_clean_test_2.parquet"]
NOISE_SHARD = "data/vibravox_parquet/speechless_clean_test_1.parquet"
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
N_FFT = 2048
HOP = 512
WIN = np.hanning(N_FFT)
BANDS = [(250, 750), (750, 1500), (1500, 2500), (2500, 8000)]


def _need():
    for p in [SPK_SHARDS[0], NOISE_SHARD]:
        if not os.path.exists(p):
            raise SkipTest(f"bandwidth characterization needs {p} (L1 shards)")


def _stft_pow(w):
    if len(w) < N_FFT:
        w = np.pad(w, (0, N_FFT - len(w)))
    n = 1 + (len(w) - N_FFT) // HOP
    acc = np.zeros(N_FFT // 2 + 1)
    for i in range(n):
        sp = np.fft.rfft(w[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        acc += np.abs(sp) ** 2
    return acc / max(n, 1)


def _cross_auto(a, b):
    n = 1 + (min(len(a), len(b)) - N_FFT) // HOP
    aa = np.zeros(N_FFT // 2 + 1); bb = np.zeros_like(aa); cc = np.zeros_like(aa, complex)
    for i in range(n):
        sa = np.fft.rfft(a[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        sb = np.fft.rfft(b[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        aa += np.abs(sa) ** 2; bb += np.abs(sb) ** 2; cc += sa * np.conj(sb)
    return aa / n, bb / n, np.abs(cc / n)


def _pairs(paths, sensor, ref, max_rows):
    seen = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        pf = pq.ParquetFile(path)
        tbl = pf.read(columns=[sensor, ref])
        for i in range(tbl.num_rows):
            if seen >= max_rows:
                return
            s = tbl.column(sensor)[i].as_py()["bytes"]
            r = tbl.column(ref)[i].as_py()["bytes"]
            sw, _ = sf.read(io.BytesIO(s), dtype="float32")
            rw, _ = sf.read(io.BytesIO(r), dtype="float32")
            if sw.ndim > 1: sw = sw.mean(1)
            if rw.ndim > 1: rw = rw.mean(1)
            yield sw, rw; seen += 1


def _band(freqs, vals, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.mean(vals[m])) if m.any() else float("nan")


def _characterize():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    a_s = np.zeros_like(freqs); a_r = np.zeros_like(freqs); cx = np.zeros_like(freqs)
    n = 0
    for sw, rw in _pairs(SPK_SHARDS, SENSOR, REF, 15):
        aa, bb, cc = _cross_auto(sw, rw)
        a_s += aa; a_r += bb; cx += cc; n += 1
    a_s /= n; a_r /= n; cx /= n
    coh = cx ** 2 / (a_s * a_r + 1e-20)
    ns = np.zeros_like(freqs); nr = np.zeros_like(freqs); k = 0
    for sw, rw in _pairs([NOISE_SHARD], SENSOR, REF, 7):
        ns += _stft_pow(sw); nr += _stft_pow(rw); k += 1
    ns /= k; nr /= k
    snr_s = 10 * np.log10(a_s / (ns + 1e-20))
    snr_r = 10 * np.log10(a_r / (nr + 1e-20))
    return freqs, coh, snr_s, snr_r


def test_l1_sensor_effective_bandwidth():
    """Sensor is band-limited vs ref; high band is its own noise floor (not signal).

    Proves the old high/low energy ratio was noise-contaminated: the proper
    metrics (coherence + noise-floor SNR) show the sensor's useful band is
    ~250-1000 Hz, dying by ~1.5 kHz, while the air ref stays high-SNR everywhere.
    """
    _need()
    freqs, coh, snr_s, snr_r = _characterize()
    print(f"\n  {'band Hz':>10} {'MSC':>7} {'SNR_sen':>9} {'SNR_ref':>9}")
    for lo, hi in BANDS:
        print(f"  {f'{lo}-{hi}':>10} {_band(freqs,coh,lo,hi):>7.3f} "
              f"{_band(freqs,snr_s,lo,hi):>8.1f}dB {_band(freqs,snr_r,lo,hi):>8.1f}dB")

    # 1. air reference stays high-SNR across the whole band (it's the clean target)
    assert _band(freqs, snr_r, 2500, 8000) > 20.0, "ref should be high-SNR up high"
    # 2. sensor high band (2-8k) is its NOISE FLOOR: SNR collapses (<< ref)
    assert _band(freqs, snr_s, 2500, 8000) < 1.0, \
        "sensor 2-8kHz should be ~noise floor (SNR<1dB), not signal"
    assert _band(freqs, snr_s, 2500, 8000) < _band(freqs, snr_r, 2500, 8000) - 30, \
        "sensor high band must be far below ref (tens of dB, not ~3)"
    # 3. coherence drops above ~1 kHz (sensor stops coherently carrying speech)
    assert _band(freqs, coh, 1500, 2500) < 0.5, \
        "sensor coherence should drop below 0.5 above ~1.5 kHz"
    assert _band(freqs, coh, 1500, 2500) < _band(freqs, coh, 250, 750), \
        "coherence in high band must be below the 250-750 Hz peak"
    # 4. sensor's BEST band is the low one (250-750): highest coherence
    assert _band(freqs, coh, 250, 750) > _band(freqs, coh, 1500, 2500)
    print("  → sensor useful band ~250-1000 Hz, dies by ~1.5 kHz; "
          "2-8 kHz is noise floor. Old high/low ratio was noise-contaminated ✓")


if __name__ == "__main__":
    test_l1_sensor_effective_bandwidth()
    print("L1 bandwidth characterization: PASS")
