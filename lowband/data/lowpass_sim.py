"""L0 simulator: clean broadband speech → degraded body-conduction signal.

Uses the degradation module (§4.3) to produce realistic sensor-like inputs
from any clean speech waveform.  Output (``ref``) is the clean original.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, apply_degradation


class LowpassSimAdapter(Dataset):
    """L0: synthetically degraded clean speech.

    Args (via cfg dict):
        clean_wavs: list of file paths to clean speech (any language).
        segment_len: segment length in samples (default 16000 = 1s @16kHz).
        sr: 16000 (target rate; data resampled to this).
        degradation: DegradationConfig kwargs.
        seed: base seed for reproducibility.
        n_repeat: how many random segments to draw per file (augmentation).
    """

    def __init__(self, cfg: dict):
        self.clean_wavs = cfg["clean_wavs"]
        self.segment_len = cfg.get("segment_len", 16000)
        self.sr = cfg.get("sr", 16000)
        self.n_repeat = cfg.get("n_repeat", 20)
        self.seed = cfg.get("seed", 42)
        deg_cfg = cfg.get("degradation", {})
        deg_cfg.setdefault("sample_rate", self.sr)
        self.deg_cfg = DegradationConfig(**deg_cfg)
        # Pre-load and cache waveforms
        self._cache = []
        self._load_audio()
        # Build index
        self._index = [(i, seg) for i in range(len(self._cache))
                       for seg in range(self.n_repeat)]

    def _load_audio(self):
        import soundfile as sf
        for path in self.clean_wavs:
            try:
                wav, sr_orig = sf.read(path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                # Resample to self.sr
                wav = _resample(wav, sr_orig, self.sr)
                if len(wav) >= self.segment_len:
                    self._cache.append(wav)
            except Exception as e:
                print(f"[LowpassSimAdapter] skip {path}: {e}")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        file_idx, seg_idx = self._index[i]
        wav = self._cache[file_idx]
        # Deterministic-but-varied segment
        rng = np.random.default_rng(self.seed + i * 7919)
        start = int(rng.integers(0, len(wav) - self.segment_len))
        seg = wav[start:start + self.segment_len].copy()
        ref = torch.from_numpy(seg).float()

        x = torch.from_numpy(seg).float()
        x_deg = apply_degradation(x, self.deg_cfg, rng=rng, n_fft=128)
        return {
            "sensor": x_deg,
            "ref": ref,
            "meta": {"sr": self.sr, "sensor_type": "simulated_lowpass",
                      "utterance_id": f"{file_idx}_{seg_idx}"},
        }


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple polyphase resample via scipy."""
    if sr_in == sr_out:
        return wav
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(wav, up, down).astype(np.float32)
