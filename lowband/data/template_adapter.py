"""Template adapter — copy this and fill in your own data loading.

§4.2: "换数据集 = 新写一个 adapter + 改 config 的一行,不许碰模型和训练代码。"
This template has exhaustive comments showing every knob you may need.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, apply_degradation
from .lowpass_sim import _resample


class TemplateAdapter(Dataset):
    """Template — replace every ``TODO`` with your own logic.

    cfg keys (all in your YAML config under ``data:``):
        data_root:        str  — your dataset root
        sensor_glob:       str  — glob for body-conduction files
        ref_glob:          str  — glob for reference mic files
        pairing:           str  — "filename" | "index" | "custom"
                                    how to match sensor↔ref
        sensor_sr:         int  — native sample rate of your sensor
        ref_sr:            int  — native sample rate of your ref mic
        target_sr:         int  — 4000 (resample both to this)
        segment_len:       int  — samples per training segment
        normalize:         bool — peak-normalize each segment
        augment:           bool — apply degradation augmentation
        degradation:       dict — DegradationConfig kwargs
        max_items:         int  — cap for smoke runs
    """

    def __init__(self, cfg: dict):
        self.data_root = cfg["data_root"]
        self.sensor_glob = cfg.get("sensor_glob", "*.wav")
        self.ref_glob = cfg.get("ref_glob", "*.wav")
        self.sensor_sr = cfg.get("sensor_sr", 16000)
        self.ref_sr = cfg.get("ref_sr", 16000)
        self.target_sr = cfg.get("target_sr", 4000)
        self.segment_len = cfg.get("segment_len", 4000)
        self.normalize = cfg.get("normalize", True)
        self.augment = cfg.get("augment", False)
        self.max_items = cfg.get("max_items", 200)
        if self.augment:
            d = cfg.get("degradation", {})
            d.setdefault("sample_rate", self.target_sr)
            self.deg_cfg = DegradationConfig(**d)
        else:
            self.deg_cfg = None

        # TODO: build self._pairs: list of (sensor_path, ref_path) tuples.
        # Use glob/pathlib to find files, then pair them per ``cfg["pairing"]``.
        self._pairs: list[tuple[str, str]] = []
        # --- example pairing by filename stem ---
        from pathlib import Path
        root = Path(self.data_root)
        sensor_files = sorted(root.glob(self.sensor_glob))
        ref_files = {p.stem: p for p in root.glob(self.ref_glob)}
        for sf in sensor_files:
            if sf.stem in ref_files:
                self._pairs.append((str(sf), str(ref_files[sf.stem])))
            if len(self._pairs) >= self.max_items:
                break

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, i):
        # TODO: replace with your native loader (librosa, soundfile, custom, ...)
        import soundfile as sf
        sensor_wav, sr_in = sf.read(self._pairs[i][0], dtype="float32")
        ref_wav, sr_ref = sf.read(self._pairs[i][1], dtype="float32")
        if sensor_wav.ndim > 1:
            sensor_wav = sensor_wav.mean(axis=1)
        if ref_wav.ndim > 1:
            ref_wav = ref_wav.mean(axis=1)

        sensor_wav = _resample(sensor_wav, sr_in, self.target_sr)
        ref_wav = _resample(ref_wav, sr_ref, self.target_sr)

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
        if self.normalize:
            p = sensor_t.abs().max().clamp_min(1e-6)
            sensor_t = sensor_t / p
            ref_t = ref_t / ref_t.abs().max().clamp_min(1e-6)

        if self.augment and self.deg_cfg is not None:
            rng = np.random.default_rng(i)
            sensor_t = apply_degradation(sensor_t, self.deg_cfg, rng=rng, n_fft=128)

        return {
            "sensor": sensor_t,
            "ref": ref_t,
            "meta": {"sr": self.target_sr, "sensor_type": "custom",
                      "utterance_id": str(i)},
        }
