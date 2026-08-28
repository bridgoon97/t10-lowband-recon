#!/usr/bin/env python3
"""Decode ONE streamed example of speech_clean/test to get real sampling rates,
channel sizes, duration, and confirm speaker/sentence pairing structure.
Downloads only the first row-group (~few MB), not whole 14 GB.
"""
import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.getcwd(), "data", "vibravox_cache"))
import numpy as np
from datasets import load_dataset

ds = load_dataset("Cnam-LMSSC/vibravox", name="speech_clean",
                  split="test", streaming=True)

print("streaming first example of speech_clean/test ...", flush=True)
ex = next(iter(ds))

print("\n--- metadata ---")
for k in ["gender", "speaker_id", "sentence_id", "duration",
          "raw_text", "normalized_text"]:
    if k in ex:
        print(f"  {k}: {ex[k]}")

print("\n--- audio channels (decode on access) ---")
aud = ex["audio"]
print(f"  audio top-level keys: {list(aud.keys())}")
for ch in ["headset_microphone", "forehead_accelerometer",
           "temple_vibration_pickup", "throat_microphone",
           "soft_in_ear_microphone", "rigid_in_ear_microphone"]:
    a = aud.get(ch)
    if a is None:
        print(f"  {ch}: MISSING")
        continue
    arr = np.asarray(a["array"], dtype=float)
    sr = a.get("sampling_rate")
    dur = len(arr) / sr if sr else float("nan")
    print(f"  {ch}: sr={sr} len={len(arr)} dur={dur:.3f}s "
          f"dtype={arr.dtype} range=[{arr.min():.3f},{arr.max():.3f}] "
          f"rms={np.sqrt(np.mean(arr**2)):.4f}")

# second example to confirm pairing keys differ
print("\n--- second example metadata (confirm distinct speaker/utter) ---")
ex2 = next(iter(ds))
print(f"  speaker_id={ex2['speaker_id']} sentence_id={ex2['sentence_id']} "
      f"duration={ex2['duration']:.3f} text={ex2['raw_text'][:40]}")
