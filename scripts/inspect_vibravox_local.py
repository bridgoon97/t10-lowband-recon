#!/usr/bin/env python3
"""Read the local vibravox parquet shard with pyarrow.

Confirms: row count, schema, sampling rate (decode 1 row's audio channels),
duration, and that sensor/ref are paired in the same row.
"""
import io
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

F = "data/vibravox_parquet/speech_clean_test_0.parquet"
pf = pq.ParquetFile(F)
print("=== schema ===")
print(pf.schema_arrow)

print("\n=== metadata ===")
print("num_rows:", pf.metadata.num_rows)
print("num_row_groups:", pf.metadata.num_row_groups)
print("num_columns:", pf.metadata.num_columns)

print("\n=== column names ===")
for n in pf.schema_arrow.names:
    print(f"  {n}")

# Read ONLY the metadata + a couple audio columns for row 0 (column projection)
meta_cols = ["gender", "speaker_id", "sentence_id", "duration", "raw_text"]
audio_cols = ["audio.headset_microphone", "audio.forehead_accelerometer",
              "audio.throat_microphone", "audio.temple_vibration_pickup"]
cols = meta_cols + audio_cols

print("\n=== row 0 (projected columns) ===")
t = pf.read_row_group(0, columns=cols)  # read first row group
# take row 0
row0 = {c: t.column(c)[0].as_py() for c in cols}
for c in meta_cols:
    print(f"  {c}: {row0[c]!r}")

print("\n=== decode audio of row 0 ===")
for c in audio_cols:
    cell = row0[c]  # dict {path, bytes} or {path, array?}
    if cell is None:
        print(f"  {c}: None"); continue
    raw = cell.get("bytes")
    path = cell.get("path")
    if raw:
        b = io.BytesIO(raw)
        try:
            wav, sr = sf.read(b, dtype="float32")
            mono = wav.mean(axis=1) if wav.ndim > 1 else wav
            dur = len(mono) / sr
            rms = float(np.sqrt(np.mean(mono ** 2)))
            print(f"  {c}: path={path}")
            print(f"      sr={sr} len={len(mono)} dur={dur:.3f}s "
                  f"range=[{mono.min():.3f},{mono.max():.3f}] rms={rms:.4f}")
        except Exception as e:
            print(f"  {c}: decode FAILED {type(e).__name__}: {e}; path={path}; nbytes={len(raw) if raw else 0}")
    else:
        # maybe already an array (struct with 'array')
        print(f"  {c}: no bytes; cell keys={list(cell.keys())}; path={path}")
