#!/usr/bin/env python3
"""Variable-window MSC: distinguish (b1) time-varying LTI vs (b2) true non-linear.

Review ②: a long-time-average MSC (~0.5 best band) can't tell:
  (b1) time-varying LTI — the transfer drifts over ~seconds; each instant is a
       good linear system, but averaging different systems → low MSC.
  (b2) true non-linear — ~half the variance is non-linearly predictable at
       every instant; MSC ~0.5 regardless of window.

Distinguish by the averaging-window length.  ⚠️ BIAS CONTROL (review): short
windows → fewer averaged segments → MSC biased HIGH (single-segment MSC ≡ 1).
Two controls applied here:
  * NON-OVERLAPPING segments (hop=N_FFT) so the n_seg in the Welch bias formula
    E[γ²_hat]=γ²_true+(1-γ²_true)/n is the TRUE independent-segment count
    (75%-overlap frames would make n_eff << n and break the correction).
  * bias-correct γ²_corr=(n·γ²_meas−1)/(n−1).
MSC is computed per-RECORDING then averaged across recordings (concatenating
frames across recordings lets windows span recording boundaries — a confound).

  * corrected MSC FALLS as n_seg grows (short >> long) ⇒ (b1) → TF+compensate
    is viable but MUST be short-time adaptive; long-time averaging is wrong.
  * corrected MSC is FLAT ~0.5 ⇒ (b2) → linear compensation caps at ~50%.
"""
import io
import os
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
          "data/vibravox_parquet/speech_clean_test_2.parquet"]
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
N_FFT = 2048
HOP = N_FFT              # NON-overlapping (so n_seg = independent segments)
WIN = np.hanning(N_FFT)
N_SEGS = [4, 16, 64, None]   # ~170 ms / 680 ms / 2.7 s / whole recording
BAND = (250, 750)


def _seg_cross(a, b):
    """Non-overlapping per-segment auto_a, auto_b, cross_ab (n_seg, F)."""
    n = len(a) // N_FFT
    F = N_FFT // 2 + 1
    aa = np.zeros((n, F)); bb = np.zeros((n, F)); cc = np.zeros((n, F), dtype=complex)
    for i in range(n):
        sa = np.fft.rfft(a[i * N_FFT:i * N_FFT + N_FFT] * WIN, N_FFT)
        sb = np.fft.rfft(b[i * N_FFT:i * N_FFT + N_FFT] * WIN, N_FFT)
        aa[i] = np.abs(sa) ** 2; bb[i] = np.abs(sb) ** 2; cc[i] = sa * np.conj(sb)
    return aa, bb, cc


def _msc_for_nseg(aa, bb, cc, n_seg):
    """MSC (per freq) averaged over sliding n_seg-segment windows; n_seg=None=all."""
    n = aa.shape[0]
    if n_seg is None or n_seg >= n:
        n_seg = n
        starts = [0]
    else:
        starts = list(range(0, n - n_seg + 1, max(1, n_seg // 2)))
    acc = np.zeros(aa.shape[1]); cnt = 0
    for s in starts:
        sl = slice(s, s + n_seg)
        msc = (np.abs(cc[sl].mean(0)) ** 2) / (aa[sl].mean(0) * bb[sl].mean(0) + 1e-20)
        acc += msc; cnt += 1
    return acc / cnt, n_seg, cnt


def _correct(msc, n_seg):
    if n_seg <= 1:
        return msc * 0 + 1.0   # single segment → MSC≡1, undefined
    return (n_seg * msc - 1.0) / (n_seg - 1.0)


def _band_val(freqs, v, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(np.mean(v[m])) if m.any() else float("nan")


def _load():
    rows = []
    for path in SHARDS:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=[SENSOR, REF])
        for i in range(min(tbl.num_rows, 10)):
            rows.append((tbl.column(SENSOR)[i].as_py()["bytes"],
                         tbl.column(REF)[i].as_py()["bytes"]))
    return rows


def main():
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    rows = _load()
    rec_lens = []
    # per-RECORDING MSC at each n_seg, then average across recordings
    per_rec = {n: [] for n in N_SEGS}
    for sb, rb in rows:
        sw, _ = sf.read(io.BytesIO(sb), dtype="float32")
        rw, _ = sf.read(io.BytesIO(rb), dtype="float32")
        if sw.ndim > 1:
            sw = sw.mean(1)
        if rw.ndim > 1:
            rw = rw.mean(1)
        sw = sw / (np.abs(sw).max() + 1e-9)
        rw = rw / (np.abs(rw).max() + 1e-9)
        L = min(len(sw), len(rw))
        rec_lens.append(L)
        aa, bb, cc = _seg_cross(sw[:L], rw[:L])
        for n_seg in N_SEGS:
            msc, n_used, _ = _msc_for_nseg(aa, bb, cc, n_seg)
            msc_c = _correct(msc, n_used)
            per_rec[n_seg].append((_band_val(freqs, msc, *BAND),
                                   _band_val(freqs, msc_c, *BAND)))
    print(f"rows={len(rows)}  N_FFT={N_FFT} hop={HOP} (non-overlap)  band={BAND}")
    print(f"\n{'n_seg':>6} {'span':>8} {'MSC_raw':>8} {'MSC_corr':>9}")
    results = []
    for n_seg in N_SEGS:
        raw = np.mean([r[0] for r in per_rec[n_seg]])
        cor = np.mean([r[1] for r in per_rec[n_seg]])
        # use the median n_used (recordings have ~equal length)
        span_ms = (n_seg or 0) * N_FFT / SR * 1000
        label = "ALL" if n_seg is None else str(n_seg)
        if n_seg is None:
            span_ms = float(np.median(rec_lens)) * 1000 / SR  # ~recording length
        print(f"{label:>6} {span_ms:>7.0f}ms {raw:>8.3f} {cor:>9.3f}")
        results.append((label, span_ms, raw, cor))
    print("  (per-recording MSC averaged across recordings; non-overlap segments; "
          "MSC_corr=(n·γ²−1)/(n−1))")

    short = results[0]; mid = results[1]; long_r = results[-1]
    drop = short[3] - long_r[3]
    print(f"\n--- verdict (best band {BAND[0]}-{BAND[1]}) ---")
    print(f"  corrected MSC: n=4(171ms)={short[3]:.3f} -> n=16(680ms)={mid[3]:.3f} "
          f"-> n=64(2.7s)={results[2][3]:.3f} -> ALL={long_r[3]:.3f}")
    print(f"  (n=4 corrected is UNRELIABLE — the asymptotic bias correction breaks "
          f"at ~4 segments; trust n>=16.)")
    # (b1) predicts short-window MSC >> long. Here short (n=4, unreliable) is
    # NOT elevated; n>=16 plateaus ~0.67. So (b1) is REJECTED at the window scale.
    # But within-recording plateau ~0.67 vs cohort (across-recording) ~0.56 => a
    # MILD time-varying component; and the ~0.67 cap (not 1.0) => a non-LTI floor.
    if mid[3] - long_r[3] > 0.10 or short[3] - mid[3] > 0.10:
        print("  ⇒ (b1) TIME-VARYING LTI: short-window MSC markedly higher; transfer "
              "drifts over ~seconds. TF+compensate viable but MUST be short-time "
              "adaptive; long-time averaging is the wrong method.")
    else:
        print("  ⇒ NOT (b1): no short-window elevation (n>=16 plateaus ~0.67). But MIXED:")
        print(f"    • within-recording plateau ~{long_r[3]:.2f} vs across-recording cohort "
              f"~0.56 (see measure_bandwidth.py) => a MILD time-varying component: a "
              "per-recording (short-time-adaptive) LINEAR estimate captures up to "
              f"~{long_r[3]:.0%}, better than a fixed/cohort estimate (~56%).")
        print(f"    • but the within-recording cap ~{long_r[3]:.2f} (not 1.0) => a non-LTI "
              f"floor ~{1-long_r[3]:.0%} remains even per-recording. So: short-time-"
              "adaptive linear for the bulk + a non-linear residual net for the "
              "floor (hybrid). Pure linear compensation still caps; pure non-linear "
              "is overkill for the ~two-thirds that IS linear.")


if __name__ == "__main__":
    main()
