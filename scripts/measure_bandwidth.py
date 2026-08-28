#!/usr/bin/env python3
"""Effective-bandwidth characterization (rework ③-redo + review ①②③).

Metrics:
  1. Bandwise MSC(temple, headset) on speech_clean — coherence drops where the
     sensor stops carrying speech; noise is incoherent → auto-excluded.
  2. Noise-floor SNR/band = speech_clean power / speechless_clean power (the
     sensor's noise-only recordings) — SNR<0 dB = speech buried in sensor noise.
  3. Noise-limited MSC ceiling γ²max = 1/((1+1/SNR_sen)(1+1/SNR_ref)) — the
     MSC that NOISE alone can explain.  Measured MSC below this ⇒ a NON-noise
     factor also kills coherence (review ②: the ~0.46 gap is systematic).
  4. GCC-PHAT delay (sensor↔ref) on the best band + DELAY-COMPENSATED MSC —
     tests hypothesis (a) 'uncompensated bone-vs-air delay artifact' vs
     (b) 'the transfer is not LTI'.  Bone conduction arrives EARLIER than air
     (air mouth→headset ~0.3-0.6 ms); uncompensated, this delay makes MSC fall
     with frequency, confounded with the SNR roll-off.

Bands now include the F0 region 50-125 / 125-250 (male/female F0) — review ①:
Arm A's viability rests on estimating F0 from the sensor, and the number that
decides that (F0-band SNR + MSC) was unmeasured.

Criteria PINNED here (review ③): 'useful band' = MSC>0.4 AND SNR>7 dB.  The
target-device 500-600 Hz figure is from elsewhere (unknown criteria) — do NOT
add the 600 Hz lowpass until it is recomputed under THIS criterion.
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
# F0 region split (review ①) + the existing bands. 0-50 Hz dropped (DC/ultra-low).
BANDS = [(50, 125), (125, 250), (250, 500), (500, 750), (750, 1000),
         (1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000), (5000, 8000)]
USEFUL_MSC = 0.4   # criteria PINNED (review ③)
USEFUL_SNR_DB = 7.0
GCC_BAND = (250, 750)   # estimate delay on the best band


def _stft_pows(wav):
    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    n = 1 + (len(wav) - N_FFT) // HOP
    acc = np.zeros(N_FFT // 2 + 1)
    for i in range(n):
        sp = np.fft.rfft(wav[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        acc += np.abs(sp) ** 2
    return acc / max(n, 1)


def _cross_auto(a, b):
    """Per-freq auto_a, auto_b, cross_ab (complex, averaged). a,b (T,)."""
    n = 1 + (min(len(a), len(b)) - N_FFT) // HOP
    aa = np.zeros(N_FFT // 2 + 1); bb = np.zeros_like(aa); cc = np.zeros_like(aa, dtype=complex)
    for i in range(n):
        sa = np.fft.rfft(a[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        sb = np.fft.rfft(b[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        aa += np.abs(sa) ** 2; bb += np.abs(sb) ** 2; cc += sa * np.conj(sb)
    return aa / max(n, 1), bb / max(n, 1), cc / max(n, 1)


def _gcc_phat(a, b, band=GCC_BAND):
    """GCC-PHAT delay (samples; + = ref LAGS sensor, i.e. bone arrives earlier),
    band-limited to `band` (best band → cleanest estimate)."""
    n = 1 + (min(len(a), len(b)) - N_FFT) // HOP
    cs = np.zeros(N_FFT // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    for i in range(n):
        sa = np.fft.rfft(a[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        sb = np.fft.rfft(b[i * HOP:i * HOP + N_FFT] * WIN, N_FFT)
        x = sa * np.conj(sb)
        cs += x / (np.abs(x) + 1e-20)
    cs /= max(n, 1)
    cs[(freqs < band[0]) | (freqs >= band[1])] = 0   # band-limit
    cc = np.fft.irfft(cs, N_FFT)
    lag = int(np.argmax(cc))
    if lag > N_FFT // 2:
        lag -= N_FFT
    return -lag   # + = ref LAGS sensor (bone conduction arrives earlier)


def _band_reduce(freqs, vals):
    out = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(np.mean(vals[m])) if m.any() else float("nan"))
    return out


def _load_pairs(parquet, sensor, ref, max_rows):
    seen = 0
    for path in parquet:
        if not os.path.exists(path):
            continue
        try:
            tbl = pq.ParquetFile(path).read(columns=[sensor, ref])
        except Exception as e:
            print(f"skip {path}: {e}"); continue
        for i in range(tbl.num_rows):
            if seen >= max_rows:
                return
            try:
                sw, _ = sf.read(io.BytesIO(tbl.column(sensor)[i].as_py()["bytes"]), dtype="float32")
                rw, _ = sf.read(io.BytesIO(tbl.column(ref)[i].as_py()["bytes"]), dtype="float32")
            except Exception as e:
                print(f"row {i} decode fail: {e}"); continue
            if sw.ndim > 1: sw = sw.mean(1)
            if rw.ndim > 1: rw = rw.mean(1)
            yield sw, rw; seen += 1


def _msc_from(pairs_iter):
    """Accumulate MSC per freq from (sensor, ref) pairs."""
    a_s = None; a_r = None; cx = None; n = 0
    for sw, rw in pairs_iter:
        aa, bb, cc = _cross_auto(sw, rw)
        if a_s is None:
            a_s = aa.copy(); a_r = bb.copy(); cx = cc.copy()
        else:
            a_s += aa; a_r += bb; cx += cc
        n += 1
    a_s /= n; a_r /= n; cx /= n
    coh = (np.abs(cx) ** 2) / (a_s * a_r + 1e-20)
    return coh, a_s, a_r, n


def main():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    pairs = list(_load_pairs(SPK_SHARDS, SENSOR, REF, max_rows=15))
    n_spk = len(pairs)

    # --- ① original (uncompensated) MSC + speech power ---
    coh, a_s, a_r, _ = _msc_from(iter(pairs))

    # --- ② GCC-PHAT delay per pair (best band), median = systematic delay ---
    delays = [_gcc_phat(sw, rw) for sw, rw in pairs]
    d_med = float(np.median(delays))
    print(f"GCC-PHAT delay (band {GCC_BAND}): per-pair median = {d_med:.0f} samples "
          f"({d_med/SR*1000:.2f} ms), IQR [{np.percentile(delays,25):.0f},"
          f"{np.percentile(delays,75):.0f}]  (+ = ref lags sensor / bone earlier)")

    # --- ② delay-compensated MSC (align ref by -d_med) ---
    d_int = int(round(d_med))
    pairs_aligned = [(sw, np.roll(rw, -d_int)) for sw, rw in pairs]
    coh_c, _, _, _ = _msc_from(iter(pairs_aligned))
    # ②b PER-PAIR compensation (each pair aligned by its OWN gcc delay) — tests
    # whether a global median was too coarse (IQR spread) and per-pair align raises MSC
    pairs_pp = [(sw, np.roll(rw, -_gcc_phat(sw, rw))) for sw, rw in pairs]
    coh_pp, _, _, _ = _msc_from(iter(pairs_pp))

    # --- noise floor + SNR ---
    noise_s = np.zeros_like(freqs); noise_r = np.zeros_like(freqs); n_ns = 0
    for sw, rw in _load_pairs([NOISE_SHARD], SENSOR, REF, max_rows=7):
        noise_s += _stft_pows(sw); noise_r += _stft_pows(rw); n_ns += 1
    noise_s /= n_ns; noise_r /= n_ns
    snr_s_lin = a_s / (noise_s + 1e-20)
    snr_r_lin = a_r / (noise_r + 1e-20)
    snr_s = 10 * np.log10(snr_s_lin)
    snr_r = 10 * np.log10(snr_r_lin)
    # noise-limited MSC ceiling (review ②): γ²max = 1/((1+1/SNRs)(1+1/SNRr))
    with np.errstate(divide="ignore"):
        ceil = 1.0 / ((1.0 + 1.0 / np.clip(snr_s_lin, 1e-20, None))
                      * (1.0 + 1.0 / np.clip(snr_r_lin, 1e-20, None)))

    print(f"\nrows: speech={n_spk}, noise(speechless)={n_ns}")
    print(f"criteria: useful band = MSC>{USEFUL_MSC} AND SNR>{USEFUL_SNR_DB}dB "
          f"(PINNED; target-device 500-600Hz is from elsewhere, unknown criteria — "
          f"lowpass NOT applied until recomputed under this criterion)")
    print(f"\n{'band Hz':>10} {'MSC':>7} {'MSC_g':>7} {'MSC_p':>7} {'γ²max':>6} "
          f"{'gap_p':>6} {'SNR_sen':>9} {'SNR_ref':>9}")
    coh_b = _band_reduce(freqs, coh); cohc_b = _band_reduce(freqs, coh_c)
    cohpp_b = _band_reduce(freqs, coh_pp)
    ceil_b = _band_reduce(freqs, ceil)
    snrs_b = _band_reduce(freqs, snr_s); snrr_b = _band_reduce(freqs, snr_r)
    for (lo, hi), c, cg, cp, ce, s, r in zip(BANDS, coh_b, cohc_b, cohpp_b, ceil_b, snrs_b, snrr_b):
        gap_p = cp - c   # per-pair-compensated rise
        print(f"{f'{lo}-{hi}':>10} {c:>7.3f} {cg:>7.3f} {cp:>7.3f} {ce:>6.3f} "
              f"{gap_p:>+6.3f} {s:>8.1f}dB {r:>8.1f}dB")
    print("  (MSC_g = global-median align; MSC_p = PER-PAIR gcc align; "
          "γ²max = noise ceiling; gap_p = per-pair rise)")

    # --- ① F0-band decision (Arm A viability) ---
    print("\n--- F0-band decision (Arm A F0-from-sensor viability) ---")
    for lo, hi in [(50, 125), (125, 250)]:
        c = _band_reduce(freqs, coh)[BANDS.index((lo, hi))]
        s = snrs_b[BANDS.index((lo, hi))]
        viable = c > USEFUL_MSC and s > USEFUL_SNR_DB
        print(f"  {lo}-{hi} Hz: MSC={c:.3f} SNR_sen={s:.1f}dB  "
              f"{'VIABLE for F0 est ✓' if viable else 'marginal/low ✗'}")

    # --- effective-bandwidth edge (criteria PINNED) ---
    print(f"\n--- effective-bandwidth edge (sensor; MSC>{USEFUL_MSC} & SNR>{USEFUL_SNR_DB}dB) ---")
    useful = (coh_c > USEFUL_MSC) & (snr_s > USEFUL_SNR_DB)   # use compensated
    edge = next((f for f, u in zip(freqs, useful) if not u and f > 200), None)
    print(f"  compensated useful band edge (top): ~{edge:.0f} Hz")
    snr0 = next((f for f, s in zip(freqs, snr_s) if s < 0 and f > 50), None)
    print(f"  SNR_sen<0dB first at: {snr0}")

    # --- ② verdict (over the human's 250-750 best band, combined) ---
    def _band_val(freqs, v, lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.mean(v[m])) if m.any() else float("nan")
    msc_250_750 = _band_val(freqs, coh, 250, 750)
    mscg_250_750 = _band_val(freqs, coh_c, 250, 750)
    mscp_250_750 = _band_val(freqs, coh_pp, 250, 750)
    print(f"\n--- delay-compensation verdict (best band 250-750) ---")
    print(f"  MSC {msc_250_750:.3f} -> global {mscg_250_750:.3f} (+{mscg_250_750-msc_250_750:.3f}) "
          f"-> per-pair {mscp_250_750:.3f} (+{mscp_250_750-msc_250_750:.3f})")
    if mscp_250_750 - msc_250_750 > 0.1:
        print("  ⇒ (a) TIME-DELAY artifact: per-pair align raised MSC; compensate each "
              "pair's bone-vs-air delay in the data pipeline.")
    else:
        print("  ⇒ (b) not delay: even PER-PAIR aligned, MSC stays ~0.5 in the best band → "
              "the sensor↔ref transfer is only ~half linearly predictable (non-LTI / wear). "
              "This caps ANY transfer-function-based domain alignment.")


if __name__ == "__main__":
    main()
