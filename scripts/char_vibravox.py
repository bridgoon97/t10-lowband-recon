#!/usr/bin/env python3
"""Quick characterization of the local vibravox shard:
- speaker variety
- effective bandwidth of body sensor vs air ref (justifies sensor choice +
  the L1 band-limitedness test). NOT an architecture ranking.
"""
import io, numpy as np, pyarrow.parquet as pq, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

F = "data/vibravox_parquet/speech_clean_test_0.parquet"
pf = pq.ParquetFile(F)
meta = pf.read(columns=["speaker_id", "sentence_id", "duration"]).to_pylist()
spk = sorted(set(m["speaker_id"] for m in meta))
durs = [m["duration"] for m in meta]
print(f"rows={len(meta)}  speakers={len(spk)}  spk_ids={spk}")
print(f"duration: min={min(durs):.2f}s max={max(durs):.2f}s mean={np.mean(durs):.2f}s")

def resample(w, sr_in, sr_out):
    if sr_in == sr_out: return w
    g = gcd(sr_in, sr_out)
    return resample_poly(w, sr_out//g, sr_in//g).astype(np.float32)

# decode 3 rows' sensor+ref, compute band energy split at 1kHz (Nyquist/2 of 4kHz target)
print("\n--- bandwidth check: body sensor vs air ref (3 rows) ---")
print("target sr=4kHz, Nyquist=2kHz. Split low[0,1kHz] vs high[1kHz,2kHz] after resample.")
rg0 = pf.read_row_group(0, columns=["audio.forehead_accelerometer","audio.headset_microphone"])
rows = list(range(min(3, len(rg0))))
print(f"{'row':>3} {'sensor':>14} {'low_rms':>9} {'high_rms':>9} {'hi/lo(dB)':>10} | {'ref_hi/lo(dB)':>14}")
for i in rows:
    s_raw = rg0.column("audio.forehead_accelerometer")[i].as_py()["bytes"]
    r_raw = rg0.column("audio.headset_microphone")[i].as_py()["bytes"]
    s_w, s_sr = sf.read(io.BytesIO(s_raw), dtype="float32")
    r_w, r_sr = sf.read(io.BytesIO(r_raw), dtype="float32")
    if s_w.ndim>1: s_w=s_w.mean(1)
    if r_w.ndim>1: r_w=r_w.mean(1)
    s4 = resample(s_w, s_sr, 4000); r4 = resample(r_w, r_sr, 4000)
    # rfft, split at 1kHz (bin = 1kHz / (4000/len) )
    def band(w):
        sp = np.abs(np.fft.rfft(w)); f = np.fft.rfftfreq(len(w), 1/4000)
        lo = np.sqrt(np.mean(sp[f<1000]**2)); hi = np.sqrt(np.mean(sp[f>=1000]**2))
        return lo, hi
    slo, shi = band(s4); rlo, rhi = band(r4)
    sdb = 20*np.log10(shi/(slo+1e-9)); rdb = 20*np.log10(rhi/(rlo+1e-9))
    print(f"{i:>3} {'forehead_acc':>14} {slo:9.4f} {shi:9.4f} {sdb:10.2f} | {rdb:14.2f}")
