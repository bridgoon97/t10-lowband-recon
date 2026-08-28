"""Measure per-segment high/low band ratio for temple vs forehead vs headset,
over the loaded shard, using the SAME metric as test_l1_adapter's
test_l1_sensor_is_bandlimited_vs_ref (median over N segments).

This tells us whether switching the default sensor to temple_vibration_pickup
(per the human's note) still satisfies the 'sensor more band-limited than ref'
assertion, or whether temple has comparable bandwidth to the headset (an honest
finding to report).
"""
import io, numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

F = "data/vibravox_parquet/speech_clean_test_0.parquet"
pf = pq.ParquetFile(F)
N_ROWS = 24

SENSORS = ["temple_vibration_pickup", "forehead_accelerometer"]
REF = "headset_microphone"
SEG = 4000   # 1s @4k after resample


def resample(w, sri, sro):
    if sri == sro:
        return w
    g = gcd(sri, sro)
    return resample_poly(w, sro // g, sri // g).astype(np.float32)


def hi_lo(w, sr=4000, split=1000):
    sp = np.abs(np.fft.rfft(w))
    f = np.fft.rfftfreq(len(w), 1 / sr)
    lo = np.sqrt(np.mean(sp[f < split] ** 2))
    hi = np.sqrt(np.mean(sp[f >= split] ** 2))
    return hi / (lo + 1e-9)


tbl = pf.read(columns=[f"audio.{c}" for c in SENSORS + [REF]])
rows = list(range(min(N_ROWS, tbl.num_rows)))
ratios = {c: [] for c in SENSORS + [REF]}
for i in rows:
    segs = {}
    for ch in SENSORS + [REF]:
        wav, sr = sf.read(io.BytesIO(tbl.column(f"audio.{ch}")[i].as_py()["bytes"]),
                          dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        w4 = resample(wav, sr, 4000)
        # center 1s segment (normalize like the adapter does)
        if len(w4) >= SEG:
            s = (len(w4) - SEG) // 2
            seg = w4[s:s + SEG]
        else:
            seg = np.pad(w4, (0, SEG - len(w4)))
        seg = seg / (np.abs(seg).max() + 1e-9)
        segs[ch] = seg
        ratios[ch].append(hi_lo(seg))
print(f"rows={len(rows)}  median high/low band ratio (lower = more band-limited):")
for ch in SENSORS + [REF]:
    med = float(np.median(ratios[ch]))
    print(f"  {ch:28s}: median={med:.4f}")
print()
for s in SENSORS:
    sm = float(np.median(ratios[s]))
    rm = float(np.median(ratios[REF]))
    print(f"  {s}: sensor={sm:.4f} vs ref={rm:.4f} -> "
          f"{'sensor MORE band-limited ✓' if sm < rm else 'sensor NOT more band-limited ✗'}")
