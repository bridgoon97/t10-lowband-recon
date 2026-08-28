#!/usr/bin/env python3
"""Inspect the REAL schema of Cnam-LMSSC/vibravox.

Goal: find out the actual config names, column names, sampling rates, and
speaker/utterance pairing — WITHOUT guessing.  Downloads NO audio (streaming
one example only).  This is the "test the assumptions" step of rework item (1).
"""
import os
import sys
import json

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.getcwd(), "data", "vibravox_cache"))

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder


def sep(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


def main():
    repo = "Cnam-LMSSC/vibravox"

    # 1. Config names (Vibravox is multi-config: one config per sensor type)
    sep("CONFIG NAMES")
    try:
        configs = get_dataset_config_names(repo)
        print(f"{len(configs)} configs:")
        for c in configs:
            print(f"  - {c}")
    except Exception as e:
        print(f"get_dataset_config_names FAILED: {type(e).__name__}: {e}")

    # 2. For each config, builder info (features, splits, sampling rate)
    sep("BUILDER / FEATURES PER CONFIG")
    for c in configs:
        try:
            b = load_dataset_builder(repo, name=c)
            info = b.info
            print(f"\n--- config: {c} ---")
            print(f"  description: {(info.description or '')[:200]}")
            print(f"  splits: {dict(info.splits) if info.splits else 'none'}")
            feats = info.features
            print(f"  features:")
            for k, v in (feats.items() if feats else []):
                print(f"    {k}: {v}")
            # Audio sampling rate if discoverable
            try:
                if feats and 'audio' in feats:
                    print(f"  audio.sampling_rate: {feats['audio'].sampling_rate}")
            except Exception as e2:
                print(f"  audio sr lookup: {e2}")
        except Exception as e:
            print(f"--- config {c}: builder FAILED: {type(e).__name__}: {e}")

    # 3. Stream ONE example from each config to see the actual dict
    sep("ONE STREAMED EXAMPLE PER CONFIG")
    for c in configs:
        try:
            ds = load_dataset(repo, name=c, split="test", streaming=True)
            ex = next(iter(ds))
            print(f"\n--- config: {c} (test split, 1st example) ---")
            for k, v in ex.items():
                if isinstance(v, dict) and "array" in v:
                    import numpy as np
                    arr = np.asarray(v["array"], dtype=float)
                    print(f"  {k}: Audio(path={v.get('path')}, sr={v.get('sampling_rate')}, "
                          f"len={len(arr)} dur={len(arr)/v.get('sampling_rate',1):.2f}s, "
                          f"dtype={arr.dtype}, min={arr.min():.3f} max={arr.max():.3f})")
                else:
                    s = str(v)
                    print(f"  {k}: {s[:200]}")
        except Exception as e:
            print(f"--- config {c}: stream FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
