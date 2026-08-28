"""Vibravox adapter (L1 — real body-conduction recordings).

§4.1:
- Only download a small subset (hundreds of clips), NOT full 45 h.
- Use the BONE-CONDUCTION pickup channel, NOT the in-ear mic.
- HuggingFace dataset: ``Cnam-LMSSC/vibravox``

The adapter streams from HuggingFace ``datasets`` if available; otherwise it
falls back to a local cache directory of .wav files laid out as::

    <root>/<speaker>/<sensor_type>/<utt_id>.wav
    <root>/<speaker>/<reference_mic>/<utt_id>.wav

where ``sensor_type`` is the bone-conduction channel name (e.g. ``bone_chin``)
and ``reference_mic`` is the air-conduction reference (e.g. ``air_oss``).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, apply_degradation
from .lowpass_sim import _resample


# Vibravox channel names (sensor_type → body-conduction pickup)
VIBRAVOX_BODY_SENSORS = [
    "bone_chin",      # chin-mounted accelerometer
    "bone_forehead",  # forehead
    "bone_throat",    # throat contact mic
    "bone_jaw",       # jaw
]
VIBRAVOX_AIR_REF = "air_oss"  # air-conduction reference mic


class VibravoxAdapter(Dataset):
    """L1: real body-conduction ↔ air-conduction paired recordings.

    Two modes:
      1. ``hf_dataset``: stream from HuggingFace (downloads a subset on the fly).
      2. ``local_root``: read paired .wav files from a local directory.

    cfg keys:
        mode: "hf" | "local"
        hf_split: "test" | "train" (test is smaller — use it for smoke runs)
        hf_cache_dir: local HF cache (keeps the download bounded)
        local_root: path to local paired .wav tree
        sensor_type: bone-conduction channel (default "bone_chin")
        ref_type: air-conduction reference (default "air_oss")
        segment_len: samples per item (default 4000)
        sr: 4000
        max_items: cap (§4.1 says hundreds, not full)
        augment: bool — apply degradation augmentation on top (default False)
        degradation: DegradationConfig kwargs (if augment)
    """

    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "local")
        self.sensor_type = cfg.get("sensor_type", "bone_chin")
        self.ref_type = cfg.get("ref_type", VIBRAVOX_AIR_REF)
        self.segment_len = cfg.get("segment_len", 4000)
        self.sr = cfg.get("sr", 4000)
        self.max_items = cfg.get("max_items", 200)
        self.augment = cfg.get("augment", False)
        if self.augment:
            d = cfg.get("degradation", {})
            d.setdefault("sample_rate", self.sr)
            self.deg_cfg = DegradationConfig(**d)
        else:
            self.deg_cfg = None
        self._pairs = []  # list of (sensor_path, ref_path) or HF indices

        if self.mode == "hf":
            self._init_hf(cfg)
        else:
            self._init_local(cfg)

    def _init_hf(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "HF mode needs `datasets` package; pip install datasets, "
                "or use mode='local' with pre-downloaded wavs."
            ) from e
        split = cfg.get("hf_split", "test")
        cache = cfg.get("hf_cache_dir", os.path.expanduser("~/.cache/vibravox"))
        ds = load_dataset("Cnam-LMSSC/vibravox", split=split, cache_dir=cache)
        # Filter to entries that have both sensor and ref
        n = 0
        for i in range(len(ds)):
            item = ds[i]
            if self.sensor_type in item and self.ref_type in item:
                self._pairs.append(("hf", i))
                n += 1
                if n >= self.max_items:
                    break
        self._ds = ds

    def _init_local(self, cfg):
        root = Path(cfg["local_root"])
        sensor_dir = root / self.sensor_type
        ref_dir = root / self.ref_type
        if not sensor_dir.exists() or not ref_dir.exists():
            print(f"[VibravoxAdapter] WARNING: {sensor_dir} or {ref_dir} "
                  f"missing; adapter will be empty.  See docs/data_adapter_guide.md")
            self._pairs = []
            return
        for spk_dir in sorted(sensor_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for wav_path in sorted(spk_dir.glob("*.wav")):
                ref_path = ref_dir / spk_dir.name / wav_path.name
                if ref_path.exists():
                    self._pairs.append(("local", str(wav_path), str(ref_path)))
                    if len(self._pairs) >= self.max_items:
                        return

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, i):
        entry = self._pairs[i]
        if entry[0] == "hf":
            item = self._ds[entry[1]]
            sensor_wav = np.asarray(item[self.sensor_type]["array"], dtype=np.float32)
            sr_in = item[self.sensor_type]["sampling_rate"]
            ref_wav = np.asarray(item[self.ref_type]["array"], dtype=np.float32)
            sr_ref = item[self.ref_type]["sampling_rate"]
        else:
            import soundfile as sf
            sensor_wav, sr_in = sf.read(entry[1], dtype="float32")
            if sensor_wav.ndim > 1:
                sensor_wav = sensor_wav.mean(axis=1)
            ref_wav, sr_ref = sf.read(entry[2], dtype="float32")
            if ref_wav.ndim > 1:
                ref_wav = ref_wav.mean(axis=1)

        sensor_wav = _resample(sensor_wav, sr_in, self.sr)
        ref_wav = _resample(ref_wav, sr_ref, self.sr)

        T = self.segment_len
        if len(sensor_wav) >= T:
            start = (len(sensor_wav) - T) // 2
            sensor = sensor_wav[start:start + T]
            ref = ref_wav[start:start + T]
        else:
            sensor = np.pad(sensor_wav, (0, T - len(sensor_wav)))
            ref = np.pad(ref_wav, (0, T - len(ref_wav)))

        sensor_t = torch.from_numpy(sensor.copy()).float()
        ref_t = torch.from_numpy(ref.copy()).float()
        if self.augment and self.deg_cfg is not None:
            rng = np.random.default_rng(i)
            sensor_t = apply_degradation(sensor_t, self.deg_cfg, rng=rng, n_fft=128)

        return {
            "sensor": sensor_t,
            "ref": ref_t,
            "meta": {"sr": self.sr, "sensor_type": self.sensor_type,
                      "utterance_id": str(i)},
        }
