#!/usr/bin/env python3
"""F0 degradation — the 5 dB stress-test table (T11, review §3 + 追加; viability verdict OVERTURNED).

Sweeps lowpass × noise type × SNR × gender → available-F0 frame rate
(the composite `agr×(1−oct)`, survivorship-safe).  This is the table that
is a PUBLIC 5 dB stress test + historical selection evidence; it does NOT decide
viability at the real device operating point (verdict OVERTURNED — see known_issues.md C4).

Why lowpass matters (review 追加 ①): the §3 numbers were on RAW temple (977 Hz
speech).  The §5 alignment lowpass (400 or 600 Hz) cuts usable harmonics:
  F0=100 male  : 9 harmonics @977 → 6 @600 → 4 @400
  F0=200 female: 4 @977 → 3 @600 → 2 @400
  F0=250 female: 3 @977 → 2 @600 → 1 @400  ← 1 harmonic can't separate F0/F0/2
⇒ predicted: 600 Hz lowpass collapses FEMALE F0; 400 Hz → female has ~1
harmonic (no F0 to estimate).  Reverses the T10 'male not worse than female'.

口径 (review ②): SNR is IN-BAND (50–600 Hz device speech band), measured via
speech_band_power — NOT full-band.  Primary criterion: available-F0 frame rate
= agr×(1−oct) (survivorship-safe); agr/oct are the decomposition.
"""
import io
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lowband.dsp.f0 import yin_f0
from lowband.data import noise as N

SHARDS = ["data/vibravox_parquet/speech_clean_test_0.parquet",
          "data/vibravox_parquet/speech_clean_test_2.parquet"]
SENSOR = "audio.temple_vibration_pickup"
REF = "audio.headset_microphone"
SR = 48000
FRAME = 2048
SNR_TIERS = [20.0, 10.0, 5.0]
NOISE_TYPES = ["clean", "white", "wind", "body"]
LOWPASSES = [None, 400, 600]   # None = raw temple (~977 Hz speech band)
SPEECH_BAND = (50.0, 600.0)    # device口径, in-band
CORRECT_TOL = 0.10
OCT_TOL = 0.06
MAX_ROWS = 12                  # 6 male + 6 female


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


def _gen_noise(kind, T, rng):
    if kind == "white":
        return N.white_noise(T, SR, rng)
    if kind == "wind":
        return N.wind_noise(T, SR, rng, slope_dboct=15.0, corner_hz=30.0,
                            gust_rate_hz=1.0, gust_depth=0.5)
    if kind == "body":
        return N.body_noise(T, SR, rng, n_impacts=3)
    return np.zeros(T, dtype=np.float32)


def _lowpass(wav, hz):
    if hz is None or hz >= SR / 2:
        return wav
    b, a = butter(6, hz / (SR / 2), btype="low")
    return filtfilt(b, a, wav).astype(np.float32)


def _load():
    rows = []
    for path in SHARDS:
        if not os.path.exists(path):
            continue
        tbl = pq.ParquetFile(path).read(columns=[SENSOR, REF, "gender"])
        for i in range(tbl.num_rows):
            g = tbl.column("gender")[i].as_py()
            if g not in ("male", "female"):
                continue
            rows.append((tbl.column(SENSOR)[i].as_py()["bytes"],
                         tbl.column(REF)[i].as_py()["bytes"], g))
            if len(rows) >= MAX_ROWS:
                return rows
    return rows


def _metric(fs_noisy, fr):
    """av = agr×(1−oct) on co-voiced frames; returns (av, oct, agr, n_co)."""
    n = min(len(fs_noisy), len(fr))
    fs, fr = fs_noisy[:n], fr[:n]
    rv = fr > 0
    n_ref_v = int(rv.sum())
    co = rv & (fs > 0)
    n_agree = int(co.sum())
    cats = Counter()
    for f_s, f_r in zip(fs[co], fr[co]):
        cats[_category(f_s, f_r)] += 1
    n_co = n_agree
    tot = max(n_co, 1)
    oct_pct = 100 * cats["octave"] / tot
    agree = 100 * n_agree / max(n_ref_v, 1)
    avail = agree * (1.0 - oct_pct / 100.0)
    return avail, oct_pct, agree, n_co


def main():
    rows = _load()
    rng = np.random.default_rng(0)
    # cache ref F0 + pre-decode sensor per row
    decoded = []
    for sb, rb, g in rows:
        sw, _ = sf.read(io.BytesIO(sb), dtype="float32")
        rw, _ = sf.read(io.BytesIO(rb), dtype="float32")
        if sw.ndim > 1:
            sw = sw.mean(1)
        if rw.ndim > 1:
            rw = rw.mean(1)
        sw = sw / (np.abs(sw).max() + 1e-9)
        rw = rw / (np.abs(rw).max() + 1e-9)
        decoded.append((sw, rw, g, _estimate(rw)))   # ref F0 cached

    print(f"rows={len(decoded)} ({sum(1 for _,_,g,_ in decoded if g=='male')}m/"
          f"{sum(1 for _,_,g,_ in decoded if g=='female')}f)  "
          f"frame={FRAME}={FRAME/SR*1000:.0f}ms  SNR in-band {SPEECH_BAND}")
    print(f"lowpass ∈ {{raw, 400, 600}} Hz  noise ∈ {NOISE_TYPES}  SNR ∈ {SNR_TIERS} dB")
    print(f"PRIMARY = available-F0 frame rate (av = agr×(1−oct))\n")

    # for each lowpass, print a noise×SNR table PER GENDER (av%), + oct/agr for 5dB
    for lp in LOWPASSES:
        label = "raw" if lp is None else f"{lp}Hz"
        # pre-lowpass the sensors once per lowpass tier
        lp_sensors = [(_lowpass(sw, lp), rw, g, fref) for sw, rw, g, fref in decoded]
        for gender in ("male", "female"):
            sel = [(s, f) for s, rw, g, f in lp_sensors if g == gender]
            if not sel:
                continue
            print(f"=== lowpass={label}  gender={gender} (n={len(sel)}) ===")
            hdr = f"{'type':>6} | " + " | ".join(f"{'snr'+str(int(s)):>16}" for s in SNR_TIERS)
            print(hdr); print("-" * len(hdr))
            for kind in NOISE_TYPES:
                cells = []
                for snr in SNR_TIERS:
                    avs = []; octs = []; agrs = []
                    for sw_lp, fref in sel:
                        if kind == "clean":
                            noisy = sw_lp
                        else:
                            noise = _gen_noise(kind, len(sw_lp), rng)
                            noisy = N.add_noise(sw_lp, noise, SR, snr, SPEECH_BAND)
                            noisy = noisy / (np.abs(noisy).max() + 1e-9)
                        av, oct_p, agr, _ = _metric(_estimate(noisy), fref)
                        avs.append(av); octs.append(oct_p); agrs.append(agr)
                    av = float(np.mean(avs)); oct_p = float(np.mean(octs)); agr = float(np.mean(agrs))
                    cells.append(f"av={av:4.0f}%(o{oct_p:2.0f}/a{agr:2.0f})")
                print(f"{kind:>6} | " + " | ".join(f"{c:>16}" for c in cells))
        print()

    # --- 5 dB stress-test table across lowpass × gender × noise (NOT the operating point)
    print("--- 5 dB stress-test point (in-band) — NOT the device operating point; verdict OVERTURNED ---")
    print(f"{'lowpass':>8} {'gender':>7} | {'white av':>9} {'wind av':>9} {'body av':>9}")
    for lp in LOWPASSES:
        label = "raw" if lp is None else f"{lp}Hz"
        lp_sensors = [(_lowpass(sw, lp), rw, g, fref) for sw, rw, g, fref in decoded]
        for gender in ("male", "female"):
            sel = [(s, f) for s, rw, g, f in lp_sensors if g == gender]
            if not sel:
                continue
            row = {}
            for kind in ("white", "wind", "body"):
                avs = []
                for sw_lp, fref in sel:
                    noise = _gen_noise(kind, len(sw_lp), rng)
                    noisy = N.add_noise(sw_lp, noise, SR, 5.0, SPEECH_BAND)
                    noisy = noisy / (np.abs(noisy).max() + 1e-9)
                    av, _, _, _ = _metric(_estimate(noisy), fref)
                    avs.append(av)
                row[kind] = float(np.mean(avs))
            print(f"{label:>8} {gender:>7} | {row['white']:>8.0f}% {row['wind']:>8.0f}% {row['body']:>8.0f}%")

    print("\n--- verdict (T11 stress test — conclusion OVERTURNED, see below) ---")
    print("  This script measures a PUBLIC stress test: Vibravox + simulated noise +")
    print("  yin_f0 at the HARD threshold conf<0.15, swept at 5 dB (a stress point).")
    print("  Lowpass finding (STANDS): 600/400 Hz lowpass does NOT materially change")
    print("  the 5 dB av; female NOT worse than male (female wind 20-21% > male 11-13%).")
    print("  The low av is driven by the hard conf<0.15 threshold (over-conservative")
    print("  under noise) — agr collapses, not real F0 failure.")
    print("  ⚠️ VIABILITY CONCLUSION OVERTURNED (real-device metro review, PRIVATE —")
    print("  not in this repo): 5 dB was a stress-test point, NOT the metro operating")
    print("  point (real: ~10-14 dB at 100-400 Hz, usable band SNR>5 dB ~100-500 Hz);")
    print("  within retained frames F0 correctness was 98.4-99.6%.  Arm A is RETAINED.")
    print("  Required design: SOFT CONFIDENCE GATING (F0 conf as soft weight modulating")
    print("  per-sub-band periodicity; high⇒harmonic, low⇒noise, no threshold) — task ②,")
    print("  not done here.  pYIN = LOW-PRIORITY comparison, not the recovery path.")
    print("  Worst-speaker risk REMAINS: at conf<0.4, available-F0 median 80.7% (worst")
    print("  61.1%) — that is WHY soft gating is required.  See known_issues.md C4 +")
    print("  l1_characterization.md T11-B for the full overturn + provenance split.")


if __name__ == "__main__":
    main()
