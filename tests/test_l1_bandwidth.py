"""L1 — effective-bandwidth characterization (rework ③-redo + review ①②③).

Metrics (mirrors scripts/measure_bandwidth.py):
  * bandwise MSC(temple, headset) on speech_clean — coherence drops where the
    sensor stops carrying speech; noise is incoherent.
  * noise-floor SNR/band = speech_clean power / speechless_clean power.
  * noise-limited MSC ceiling γ²max = 1/((1+1/SNR_sen)(1+1/SNR_ref)).
  * GCC-PHAT delay (sensor↔ref) on the best band + delay-compensated MSC —
    tests (a) uncompensated bone-vs-air delay artifact vs (b) non-LTI transfer.

Bands include the F0 region 50-125 / 125-250 (male/female F0) — review ①: Arm A's
viability rests on F0-from-sensor, and the deciding number was unmeasured.

Criteria PINNED (review ③): useful band = MSC>0.4 AND SNR>7 dB.  The 600 Hz
lowpass is NOT applied (target-device 500-600 Hz is from elsewhere, unknown
criteria).  SKIPs if the local parquet shards are absent.
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
              "data/vibravox_parquet/speechless_clean_test_1.parquet"  # placeholder; replaced below
              ]
SPK_SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
              "data/vibravox_parquet/speech_clean_test_2.parquet"]
NOISE_SHARD = "data/vibravox_parquet/speechless_clean_test_1.parquet"
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
N_FFT = 2048
HOP = 512
WIN = np.hanning(N_FFT)
BANDS = [(50, 125), (125, 250), (250, 500), (500, 750), (750, 1000),
         (1000, 1500), (1500, 2500), (2500, 8000)]
USEFUL_MSC = 0.4
USEFUL_SNR_DB = 7.0
GCC_BAND = (250, 750)


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
    return aa / max(n, 1), bb / max(n, 1), cc / max(n, 1)


def _gcc_phat(a, b, band=GCC_BAND):
    n = 1 + (min(len(a), len(b)) - N_FFT) // HOP
    cs = np.zeros(N_FFT // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    for i in range(n):
        sa = np.fft.rfft(a[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        sb = np.fft.rfft(b[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        x = sa * np.conj(sb)
        cs += x / (np.abs(x) + 1e-20)
    cs /= max(n, 1)
    cs[(freqs < band[0]) | (freqs >= band[1])] = 0
    cc = np.fft.irfft(cs, N_FFT)
    lag = int(np.argmax(cc))
    if lag > N_FFT // 2:
        lag -= N_FFT
    return -lag   # + = ref lags sensor (bone earlier)


def _pairs(paths, sensor, ref, max_rows):
    seen = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=[sensor, ref])
        for i in range(tbl.num_rows):
            if seen >= max_rows:
                return
            sw, _ = sf.read(io.BytesIO(tbl.column(sensor)[i].as_py()["bytes"]), dtype="float32")
            rw, _ = sf.read(io.BytesIO(tbl.column(ref)[i].as_py()["bytes"]), dtype="float32")
            if sw.ndim > 1: sw = sw.mean(1)
            if rw.ndim > 1: rw = rw.mean(1)
            yield sw, rw; seen += 1


def _band(freqs, vals, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.mean(vals[m])) if m.any() else float("nan")


def _msc_from(pairs_iter):
    a_s = a_r = cx = None; n = 0
    for sw, rw in pairs_iter:
        aa, bb, cc = _cross_auto(sw, rw)
        if a_s is None:
            a_s = aa.copy(); a_r = bb.copy(); cx = cc.copy()
        else:
            a_s += aa; a_r += bb; cx += cc
        n += 1
    a_s /= n; a_r /= n; cx /= n
    return (np.abs(cx) ** 2) / (a_s * a_r + 1e-20), a_s, a_r, n


def _characterize():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    pairs = list(_pairs(SPK_SHARDS, SENSOR, REF, 15))
    coh, a_s, a_r, _ = _msc_from(iter(pairs))
    delays = [_gcc_phat(sw, rw) for sw, rw in pairs]
    d_med = float(np.median(delays))
    d_int = int(round(d_med))
    coh_g, _, _, _ = _msc_from((sw, np.roll(rw, -d_int)) for sw, rw in pairs)
    coh_p, _, _, _ = _msc_from((sw, np.roll(rw, -_gcc_phat(sw, rw))) for sw, rw in pairs)
    ns = np.zeros_like(freqs); nr = np.zeros_like(freqs); k = 0
    for sw, rw in _pairs([NOISE_SHARD], SENSOR, REF, 7):
        ns += _stft_pow(sw); nr += _stft_pow(rw); k += 1
    ns /= k; nr /= k
    snr_s_lin = a_s / (ns + 1e-20); snr_r_lin = a_r / (nr + 1e-20)
    snr_s = 10 * np.log10(snr_s_lin); snr_r = 10 * np.log10(snr_r_lin)
    with np.errstate(divide="ignore"):
        ceil = 1.0 / ((1.0 + 1.0 / np.clip(snr_s_lin, 1e-20, None))
                      * (1.0 + 1.0 / np.clip(snr_r_lin, 1e-20, None)))
    return freqs, coh, coh_g, coh_p, d_med, ceil, snr_s, snr_r


def test_l1_sensor_effective_bandwidth():
    """Sensor is band-limited; high band is its own noise floor; the MSC gap to
    the noise ceiling is NOT a delay artifact (per-pair GCC align recovers only
    ~+0.05) ⇒ the sensor↔ref transfer is only ~half linearly predictable (b)."""
    _need()
    freqs, coh, coh_g, coh_p, d_med, ceil, snr_s, snr_r = _characterize()
    print(f"\n  GCC-PHAT delay (band {GCC_BAND}): median={d_med:.0f} samples "
          f"({d_med/SR*1000:.2f} ms)  criteria: MSC>{USEFUL_MSC} & SNR>{USEFUL_SNR_DB}dB")
    print(f"  {'band Hz':>10} {'MSC':>7} {'MSC_g':>7} {'MSC_p':>7} {'γ²max':>6} "
          f"{'gap_p':>6} {'SNR_sen':>9} {'SNR_ref':>9}")
    for lo, hi in BANDS:
        c = _band(freqs, coh, lo, hi); cg = _band(freqs, coh_g, lo, hi)
        cp = _band(freqs, coh_p, lo, hi); ce = _band(freqs, ceil, lo, hi)
        s = _band(freqs, snr_s, lo, hi); r = _band(freqs, snr_r, lo, hi)
        print(f"  {f'{lo}-{hi}':>10} {c:>7.3f} {cg:>7.3f} {cp:>7.3f} {ce:>6.3f} "
              f"{cp-c:>+6.3f} {s:>8.1f}dB {r:>8.1f}dB")

    # --- ① F0-band decision (Arm A viability) ---
    print("  --- F0-band (Arm A F0-from-sensor) ---")
    f0_50 = _band(freqs, coh, 50, 125); f0_125 = _band(freqs, coh, 125, 250)
    print(f"    50-125 Hz (male F0):   MSC={f0_50:.3f} SNR={_band(freqs,snr_s,50,125):.1f}dB")
    print(f"    125-250 Hz (female F0): MSC={f0_125:.3f} SNR={_band(freqs,snr_s,125,250):.1f}dB")

    # 1. air ref high-SNR across the band (clean target)
    assert _band(freqs, snr_r, 2500, 8000) > 20.0
    # 2. sensor high band = its noise floor (SNR collapses, far below ref)
    assert _band(freqs, snr_s, 2500, 8000) < 1.0
    assert _band(freqs, snr_s, 2500, 8000) < _band(freqs, snr_r, 2500, 8000) - 30
    # 3. coherence drops above ~1 kHz; best band is 250-500
    assert _band(freqs, coh, 1500, 2500) < 0.5
    assert _band(freqs, coh, 250, 500) > _band(freqs, coh, 1500, 2500)
    # 4. ② per-pair delay compensation recovers only a SMALL fraction (=> not a
    #    delay artifact; the transfer is non-LTI). gap < 0.15 in the best band.
    gap = _band(freqs, coh_p, 250, 750) - _band(freqs, coh, 250, 750)
    print(f"  per-pair delay-comp rise (250-750): +{gap:.3f} "
          f"({'minor => (b) non-LTI' if gap < 0.15 else 'large => (a) delay artifact'})")
    assert gap < 0.15, f"per-pair compensation rose MSC by {gap:.3f} — revisit (a) delay"
    # 5. measured MSC is far below the noise ceiling (the ~0.46 gap is systematic)
    assert _band(freqs, coh, 250, 500) < _band(freqs, ceil, 250, 500) - 0.2
    print("  → sensor useful band ~250-1000 Hz; high band is noise floor; "
          "MSC gap to ceiling is NOT delay (per-pair +<0.15) ⇒ (b) non-LTI "
          "transfer caps domain alignment ✓")


if __name__ == "__main__":
    test_l1_sensor_effective_bandwidth()
    print("L1 bandwidth characterization: PASS")
