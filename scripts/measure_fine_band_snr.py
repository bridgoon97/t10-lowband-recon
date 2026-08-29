#!/usr/bin/env python3
"""Fine-band SNR + usable-band crossing (T11 §1).

NEW criterion (T11): effective bandwidth = pure SNR, threshold 5 dB (NOT T10's
MSC>0.4 AND SNR>7 dB).  The target device has speech only below 400–600 Hz with
SNR just >5 dB, noise everywhere else, plus wind.  To compare temple to the
target device under ONE criterion, the crossing point (where SNR falls through
5 dB) must be found in FINE bands — T10's 750–1500 Hz band averaged 4.3 dB,
hiding the crossing.

Reports:
  * per-freq SNR (speech_clean / speechless_clean) at 100 Hz steps 50–2000 Hz
    + the SNR=5 dB crossing (upper edge of the usable band).
  * the bandpass-not-lowpass structure (T11 §1): the low edge 50–125 Hz is
    weak (7.7 dB) while 125–750 is 21–24 dB — the sensor is NOT a clean lowpass;
    male F0 (85–155 Hz) sits on the weak low edge.  Worth confirming in fine
    bands and writing into the report.
  * temple's crossing vs the target device's 400–600 Hz (same criterion now).
"""
import io
import os
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
SNR_THRESH_DB = 5.0          # T11 unified criterion
FINE_BANDS = [(lo, lo + 100) for lo in range(50, 2000, 100)]   # 50..2000 @100 Hz


def _stft_pow(wav):
    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    n = 1 + (len(wav) - N_FFT) // HOP
    acc = np.zeros(N_FFT // 2 + 1)
    for i in range(n):
        sp = np.fft.rfft(wav[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        acc += np.abs(sp) ** 2
    return acc / max(n, 1)


def _load(paths, channels, max_rows):
    seen = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=channels)
        for i in range(tbl.num_rows):
            if seen >= max_rows:
                return
            yield [tbl.column(c)[i].as_py()["bytes"] for c in channels]
            seen += 1


def _band(freqs, vals, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.mean(vals[m])) if m.any() else float("nan")


def main():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    # speech + noise power, per channel
    sp_s = np.zeros_like(freqs); sp_r = np.zeros_like(freqs); n_spk = 0
    for sb, rb in _load(SPK_SHARDS, [SENSOR, REF], 15):
        sw, _ = sf.read(io.BytesIO(sb), dtype="float32")
        rw, _ = sf.read(io.BytesIO(rb), dtype="float32")
        if sw.ndim > 1:
            sw = sw.mean(1)
        if rw.ndim > 1:
            rw = rw.mean(1)
        sp_s += _stft_pow(sw); sp_r += _stft_pow(rw); n_spk += 1
    sp_s /= n_spk; sp_r /= n_spk
    ns_s = np.zeros_like(freqs); ns_r = np.zeros_like(freqs); n_ns = 0
    for sb, rb in _load([NOISE_SHARD], [SENSOR, REF], 7):
        sw, _ = sf.read(io.BytesIO(sb), dtype="float32")
        rw, _ = sf.read(io.BytesIO(rb), dtype="float32")
        if sw.ndim > 1:
            sw = sw.mean(1)
        if rw.ndim > 1:
            rw = rw.mean(1)
        ns_s += _stft_pow(sw); ns_r += _stft_pow(rw); n_ns += 1
    ns_s /= n_ns; ns_r /= n_ns
    snr_s = 10 * np.log10(sp_s / (ns_s + 1e-20))   # per-freq sensor SNR
    snr_r = 10 * np.log10(sp_r / (ns_r + 1e-20))

    print(f"rows: speech={n_spk}, noise={n_ns}  criterion: SNR>{SNR_THRESH_DB} dB (T11 unified)")
    print(f"\n{'band Hz':>10} {'SNR_sen':>9} {'SNR_ref':>9}  {'sen>5dB':>7}")
    for lo, hi in FINE_BANDS:
        s = _band(freqs, snr_s, lo, hi); r = _band(freqs, snr_r, lo, hi)
        flag = "✓" if s > SNR_THRESH_DB else "✗"
        print(f"{f'{lo}-{hi}':>10} {s:>8.1f}dB {r:>8.1f}dB  {flag:>7}")

    # SNR=5 dB crossing (upper edge of usable band) — per-freq, interpolated
    usable = snr_s > SNR_THRESH_DB
    # find the last contiguous usable freq above 100 Hz (skip the low edge dip)
    crossing = None
    for i in range(len(freqs) - 1):
        if freqs[i] > 100 and usable[i] and not usable[i + 1]:
            # interpolate the crossing between i and i+1
            f0, f1 = freqs[i], freqs[i + 1]
            s0, s1 = snr_s[i], snr_s[i + 1]
            crossing = float(f0 + (SNR_THRESH_DB - s0) / (s1 - s0 + 1e-12) * (f1 - f0))
            break
    print(f"\n--- SNR={SNR_THRESH_DB} dB crossing (sensor, upper edge) ---")
    print(f"  temple crossing ≈ {crossing:.0f} Hz" if crossing else "  no crossing found")
    print(f"  target device speech band: 400–600 Hz (criterion now unified)")
    if crossing:
        if crossing < 400:
            print(f"  ⇒ temple usable band ({crossing:.0f} Hz) is NARROWER than target "
                  "(400-600) — temple is the weaker sensor; a lowpass won't help (already narrower).")
        elif crossing > 600:
            print(f"  ⇒ temple usable band ({crossing:.0f} Hz) is WIDER than target "
                  "(400-600) — add a lowpass on the sensor channel to align (§5 action).")
        else:
            print(f"  ⇒ temple usable band ({crossing:.0f} Hz) ≈ target (400-600) — aligned.")

    # bandpass-not-lowpass structure (low edge dip)
    low = _band(freqs, snr_s, 50, 125); mid = _band(freqs, snr_s, 125, 750)
    print(f"\n--- bandpass-not-lowpass check (T11 §1) ---")
    print(f"  50-125 Hz: {low:.1f} dB   125-750 Hz: {mid:.1f} dB")
    if low < mid - 5:
        print("  low edge WEAKER than mid band ⇒ BANDPASS shape (sensor highpass /")
        print("  drift-removal); male F0 (85-155 Hz) sits on the weak low edge.")
    else:
        print("  low edge not weaker than mid band ⇒ lowpass-like.")


if __name__ == "__main__":
    main()
