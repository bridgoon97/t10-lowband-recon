#!/usr/bin/env python3
"""Direct F0-estimation error measurement (review ①-correction).

The band-SNR proxy for F0 estimability was WRONG (review): YIN/autocorrelation
estimates PERIODICITY from the strongest harmonics, not from h1, so male F0
(dense harmonics h2-h7 in the good 250-750 Hz band) may estimate BETTER than
the h1-band SNR suggested.  The only honest measure is to estimate F0 on the
sensor with ref F0 as ground truth, bucketed by gender (Vibravox has it free).

Reports per gender:
  * voiced/unvoiced agreement (sensor voiced when ref voiced)
  * relative F0 error distribution (median |f0_s-f0_r|/f0_r)
  * error CATEGORY: correct (<10%), octave (~2x/0.5x), half-octave (1.5x/0.67x),
    gross — the octave/half-octave errors are the classic failure modes that
    decide whether Arm A's F0-from-sensor is a usable foundation.

Uses the project's own yin_f0 (lowband.dsp.f0) on BOTH channels at 48 kHz, so
the number reflects what Arm A would actually do.  Same estimator + params on
both → the error is sensor-vs-ref, not estimator-vs-truth.
"""
import io
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowband.dsp.f0 import yin_f0

SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
          "data/vibravox_parquet/speech_clean_test_2.parquet"]
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
FRAME = 2048          # ~43 ms; >= 2 periods of f0_min=50 Hz (960 samples)
MAX_ROWS = 16
OCT_TOL = 0.06        # relative tolerance for 2x/0.5x/1.5x/0.67x classification
CORRECT_TOL = 0.10    # <10% relative error = "correct"


def _estimate(wav):
    x = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
    f0, prob = yin_f0(x, SR, frame_len=FRAME, f0_min=50.0, f0_max=400.0)
    return f0[0].numpy(), prob[0].numpy()


def _category(fs, fr):
    """Classify sensor F0 (fs) vs ref F0 (fr) for co-voiced frames."""
    r = fs / fr
    if abs(r - 1.0) < CORRECT_TOL:
        return "correct"
    if abs(r - 2.0) < OCT_TOL or abs(r - 0.5) < OCT_TOL:
        return "octave"
    if abs(r - 1.5) < OCT_TOL or abs(r - 2.0 / 3.0) < OCT_TOL:
        return "half_octave"
    return "gross"


def _load():
    rows = []
    for path in SHARDS:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=[SENSOR, REF, "gender"])
        for i in range(tbl.num_rows):
            rows.append((tbl.column(SENSOR)[i].as_py()["bytes"],
                         tbl.column(REF)[i].as_py()["bytes"],
                         tbl.column("gender")[i].as_py()))
    return rows


def main():
    rows = _load()[:MAX_ROWS]
    by_g = {"male": [], "female": []}
    for sb, rb, g in rows:
        if g not in by_g:
            continue
        sw, _ = sf.read(io.BytesIO(sb), dtype="float32")
        rw, _ = sf.read(io.BytesIO(rb), dtype="float32")
        if sw.ndim > 1:
            sw = sw.mean(1)
        if rw.ndim > 1:
            rw = rw.mean(1)
        sw = sw / (np.abs(sw).max() + 1e-9)
        rw = rw / (np.abs(rw).max() + 1e-9)
        fs, _ = _estimate(sw)
        fr, _ = _estimate(rw)
        n = min(len(fs), len(fr))
        by_g[g].append((fs[:n], fr[:n]))

    print(f"rows: {len(rows)}  (frame={FRAME}={FRAME/SR*1000:.0f}ms @48k, "
          f"f0 search 50-400 Hz)")
    print("  口径: error on CO-VOICED frames (ref voiced AND sensor voiced); "
          "agree% = voiced-DECISION consistency (sensor voiced when ref voiced), "
          "NOT F0-value consistency")
    print(f"\n{'gender':>7} {'n_rows':>6} {'ref_voiced%':>11} "
          f"{'sen_voiced%':>11} {'agree%':>7} {'median_rel':>10} "
          f"{'<10%':>6} {'oct%':>6} {'half%':>6} {'gross%':>6}")
    for g in ["male", "female"]:
        pairs = by_g[g]
        if not pairs:
            continue
        # ref-voiced frames, sensor-voiced agreement, co-voiced errors
        n_ref_v = n_sen_v = n_agree = 0
        rel_errs = []
        cats = Counter()
        for fs, fr in pairs:
            rv = fr > 0
            sv = fs > 0
            n_ref_v += int(rv.sum())
            n_sen_v += int(sv.sum())
            co = rv & sv
            n_agree += int(co.sum())
            for f_s, f_r in zip(fs[co], fr[co]):
                rel_errs.append(abs(f_s - f_r) / f_r)
                cats[_category(f_s, f_r)] += 1
        n_co = n_agree
        med = float(np.median(rel_errs)) if rel_errs else float("nan")
        pct10 = 100.0 * sum(1 for e in rel_errs if e < CORRECT_TOL) / max(n_co, 1)
        tot = max(n_co, 1)
        oct_pct = 100 * cats['octave'] / tot
        avail = 100.0 * (n_agree / max(n_ref_v, 1)) * (1.0 - oct_pct / 100.0)  # agr×(1−oct)
        print(f"{g:>7} {len(pairs):>6} {100*n_ref_v/max(len(fr)*len(pairs),1):>10.1f}% "
              f"{100*n_sen_v/max(len(fr)*len(pairs),1):>10.1f}% "
              f"{100*n_agree/max(n_ref_v,1):>6.1f}% {med:>10.3f} "
              f"{pct10:>5.1f}% {oct_pct:>5.1f}% "
              f"{100*cats['half_octave']/tot:>5.1f}% {100*cats['gross']/tot:>5.1f}% "
              f"av={avail:>5.1f}%")
    print("\n  (av = available-F0 frame rate = agr×(1−oct), the composite PRIMARY "
          "metric; agree = sensor voiced when ref voiced; <10%/oct/half/gross = "
          "error categories on co-voiced)")


if __name__ == "__main__":
    main()
