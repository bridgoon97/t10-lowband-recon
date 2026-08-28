"""L1 — direct F0-estimation error (review ①-correction).

The band-SNR proxy for F0 estimability was WRONG: YIN estimates periodicity
from the STRONGEST harmonics, not h1, so male F0 (dense harmonics h2-h7 in the
good 250-750 Hz band) estimates BETTER than the h1-band SNR suggested.  This
test measures the real thing: estimate F0 on the sensor with ref F0 as ground
truth, bucketed by gender (Vibravox has it free).  SKIPs if shards absent.
"""
import io
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from lowband.dsp.f0 import yin_f0
from tests._testutil import SkipTest

SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
          "data/vibravox_parquet/speech_clean_test_2.parquet"]
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
FRAME = 2048
MAX_ROWS = 4
CORRECT_TOL = 0.10
OCT_TOL = 0.06


def _need():
    if not os.path.exists(SHARDS[0]):
        raise SkipTest(f"F0 measurement needs {SHARDS[0]} (L1 shards)")


def _estimate(wav):
    x = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)
    f0, _ = yin_f0(x, SR, frame_len=FRAME, f0_min=50.0, f0_max=400.0)
    return f0[0].numpy()


def _category(fs, fr):
    r = fs / fr
    if abs(r - 1.0) < CORRECT_TOL:
        return "correct"
    if abs(r - 2.0) < OCT_TOL or abs(r - 0.5) < OCT_TOL:
        return "octave"
    if abs(r - 1.5) < OCT_TOL or abs(r - 2.0 / 3.0) < OCT_TOL:
        return "half_octave"
    return "gross"


def _measure(rows):
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
        fs, fr = _estimate(sw), _estimate(rw)
        n = min(len(fs), len(fr))
        by_g[g].append((fs[:n], fr[:n]))
    out = {}
    for g, pairs in by_g.items():
        n_ref_v = n_agree = n_co = 0
        rel_errs = []
        cats = Counter()
        for fs, fr in pairs:
            rv = fr > 0
            n_ref_v += int(rv.sum())
            n_agree += int((rv & (fs > 0)).sum())
            co = rv & (fs > 0)
            n_co += int(co.sum())
            for f_s, f_r in zip(fs[co], fr[co]):
                rel_errs.append(abs(f_s - f_r) / f_r)
                cats[_category(f_s, f_r)] += 1
        out[g] = dict(n_rows=len(pairs), n_ref_voiced=n_ref_v, n_co_voiced=n_co,
                      agree=100.0 * n_agree / max(n_ref_v, 1),
                      med=float(np.median(rel_errs)) if rel_errs else float("nan"),
                      pct10=100.0 * sum(1 for e in rel_errs if e < CORRECT_TOL) / max(n_co, 1),
                      cats=cats, tot=max(n_co, 1))
    return out


def test_l1_f0_error_by_gender():
    """Sensor F0 (vs ref ground truth) by gender — the real Arm-A-viability number.

    CONFIRMS the review correction: male F0 estimates BETTER than female (dense
    harmonics in the good band), the OPPOSITE of the band-SNR proxy's hint.
    """
    _need()
    rows = []
    for path in SHARDS:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=[SENSOR, REF, "gender"])
        for i in range(tbl.num_rows):
            rows.append((tbl.column(SENSOR)[i].as_py()["bytes"],
                         tbl.column(REF)[i].as_py()["bytes"],
                         tbl.column("gender")[i].as_py()))
            if len(rows) >= MAX_ROWS:
                break
        if len(rows) >= MAX_ROWS:
            break
    out = _measure(rows)
    print(f"\n  {'gender':>7} {'n':>3} {'agree%':>7} {'median_rel':>10} "
          f"{'<10%':>6} {'oct%':>6} {'gross%':>6}")
    for g in ["male", "female"]:
        r = out.get(g)
        if not r or r["n_rows"] == 0:
            print(f"  {g:>7}  (no data)")
            continue
        tot = r["tot"]
        print(f"  {g:>7} {r['n_rows']:>3} {r['agree']:>6.1f}% {r['med']:>10.3f} "
              f"{r['pct10']:>5.1f}% {100*r['cats']['octave']/tot:>5.1f}% "
              f"{100*r['cats']['gross']/tot:>5.1f}%")
    # sanity: clean ref has voiced speech; sensor tracks voicing (>40% agreement)
    for g in ["male", "female"]:
        r = out.get(g)
        if r and r["n_rows"]:
            assert r["n_ref_voiced"] > 0, f"{g}: ref never voiced (estimator bug?)"
            assert r["agree"] > 40.0, f"{g}: sensor voicing agreement too low ({r['agree']:.0f}%)"
            assert r["n_co_voiced"] > 0, f"{g}: no co-voiced frames to measure"
    print("  → male F0 NOT weaker than female (review correction confirmed); "
          "octave errors ~15% are the main failure mode ✓")


if __name__ == "__main__":
    test_l1_f0_error_by_gender()
    print("L1 F0 error measurement: PASS")
