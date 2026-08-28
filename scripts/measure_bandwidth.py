#!/usr/bin/env python3
"""Proper effective-bandwidth characterization (rework ③ redo, per review).

The old metric (high/low energy ratio) was noise-floor-contaminated: a body
sensor's high band holds its OWN noise floor, not rolled-off speech, so it
looks far less band-limited than it is.  Replace with:

  1. Bandwise magnitude-squared coherence MSC(temple, headset) on speech_clean.
     Coherence drops where the sensor stops carrying speech; noise is
     incoherent → auto-excluded.  (Also a future wear-quality reliability metric.)
  2. Noise-floor SNR per band: speech_clean power / speechless_clean power
     (speechless_clean = the sensor's noise-only recordings).  The band where
     SNR<0 dB (speech buried in noise) is the effective-bandwidth edge.

Answers the 3-way: noise-floor contamination / wrong band split / temple wider
than the ~500-600 Hz target device.
"""
import io
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SPK_SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
              "data/vibravox_parquet/speech_clean_test_2.parquet"]
NOISE_SHARD = "data/vibravox_parquet/speechless_clean_test_1.parquet"
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
N_FFT = 2048
HOP = 512
WIN = np.hanning(N_FFT)
BANDS = [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500),
         (1500, 2000), (2000, 3000), (3000, 5000), (5000, 8000)]


def _stft_pows(wav):
    """Return (auto_power (F,), ) — averaged |STFT|^2 over frames."""
    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    n = 1 + (len(wav) - N_FFT) // HOP
    acc = np.zeros(N_FFT // 2 + 1)
    for i in range(n):
        s = i * HOP
        fr = wav[s:s + N_FFT] * WIN
        sp = np.fft.rfft(fr, N_FFT)
        acc += np.abs(sp) ** 2
    return acc / max(n, 1)


def _cross_auto(a, b):
    """Per-freq auto_a, auto_b, |cross_ab| averaged over frames. a,b (T,)."""
    if len(a) < N_FFT:
        a = np.pad(a, (0, N_FFT - len(a))); b = np.pad(b, (0, N_FFT - len(b)))
    n = 1 + (len(a) - N_FFT) // HOP
    aa = np.zeros(N_FFT // 2 + 1); bb = np.zeros_like(aa); cc = np.zeros_like(aa, dtype=complex)
    for i in range(n):
        s = i * HOP
        fa = a[s:s + N_FFT] * WIN; fb = b[s:s + N_FFT] * WIN
        sa = np.fft.rfft(fa, N_FFT); sb = np.fft.rfft(fb, N_FFT)
        aa += np.abs(sa) ** 2; bb += np.abs(sb) ** 2; cc += sa * np.conj(sb)
    return aa / n, bb / n, np.abs(cc / n)


def _band_reduce(freqs, vals):
    out = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(np.mean(vals[m])) if m.any() else float("nan"))
    return out


def _load_pairs(parquet, sensor, ref, max_rows):
    """Yield (sensor_wav, ref_wav) at SR from a list of parquet files."""
    seen = 0
    for path in parquet:
        if not __import__("os").path.exists(path):
            continue
        pf = pq.ParquetFile(path)
        try:
            tbl = pf.read(columns=[sensor, ref])
        except Exception as e:
            print(f"skip {path}: {e}"); continue
        for i in range(tbl.num_rows):
            if seen >= max_rows:
                return
            try:
                s = tbl.column(sensor)[i].as_py()["bytes"]
                r = tbl.column(ref)[i].as_py()["bytes"]
                sw, _ = sf.read(io.BytesIO(s), dtype="float32")
                rw, _ = sf.read(io.BytesIO(r), dtype="float32")
            except Exception as e:
                print(f"row {i} decode fail: {e}"); continue
            if sw.ndim > 1: sw = sw.mean(1)
            if rw.ndim > 1: rw = rw.mean(1)
            yield sw, rw
            seen += 1


def main():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)

    # speech: accumulate auto(sensor), auto(ref), cross; per-channel speech power
    a_s = np.zeros_like(freqs); a_r = np.zeros_like(freqs); cx = np.zeros_like(freqs)
    n_spk = 0
    for sw, rw in _load_pairs(SPK_SHARDS, SENSOR, REF, max_rows=15):
        aa, bb, cc = _cross_auto(sw, rw)
        a_s += aa; a_r += bb; cx += cc; n_spk += 1
    a_s /= n_spk; a_r /= n_spk; cx /= n_spk
    coh = cx ** 2 / (a_s * a_r + 1e-20)            # MSC per freq

    # noise floor (speechless_clean): per-channel noise power
    noise_s = np.zeros_like(freqs); noise_r = np.zeros_like(freqs); n_ns = 0
    for sw, rw in _load_pairs([NOISE_SHARD], SENSOR, REF, max_rows=7):
        noise_s += _stft_pows(sw); noise_r += _stft_pows(rw); n_ns += 1
    noise_s /= n_ns; noise_r /= n_ns

    snr_s = 10 * np.log10(a_s / (noise_s + 1e-20))   # sensor speech vs its noise floor
    snr_r = 10 * np.log10(a_r / (noise_r + 1e-20))

    print(f"rows: speech={n_spk}, noise(speechless)={n_ns}")
    print(f"\n{'band Hz':>12} {'MSC':>7} {'SNR_sen':>9} {'SNR_ref':>9}")
    coh_b = _band_reduce(freqs, coh)
    snrs_b = _band_reduce(freqs, snr_s)
    snrr_b = _band_reduce(freqs, snr_r)
    for (lo, hi), c, s, r in zip(BANDS, coh_b, snrs_b, snrr_b):
        print(f"{f'{lo}-{hi}':>12} {c:>7.3f} {s:>8.1f}dB {r:>8.1f}dB")

    # effective-bandwidth edge for the SENSOR: lowest freq where MSC<0.5 or SNR<0
    print("\n--- effective-bandwidth edge (sensor) ---")
    for i, f in enumerate(freqs):
        if f > 200 and (coh[i] < 0.5 or snr_s[i] < 0):
            print(f"  first break: ~{f:.0f} Hz  (MSC={coh[i]:.2f}, SNR_sen={snr_s[i]:.1f}dB)")
            break
    # where sensor SNR drops below 0 (speech buried in its own noise)
    snr0 = next((f for f, s in zip(freqs, snr_s) if s < 0 and f > 200), None)
    coh0 = next((f for f, c in zip(freqs, coh) if c < 0.5 and f > 200), None)
    print(f"  SNR_sen<0dB first at: {snr0}")
    print(f"  MSC<0.5 first at: {coh0}")


if __name__ == "__main__":
    main()
