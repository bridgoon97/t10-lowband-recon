"""Real-device recording loader for T13-A rework R2/R4 (0624/ only; 0625/ is the
held-out set — DO NOT TOUCH).

Path confirmed by the reviewer: read directly from
``/mnt/d/Projects/mic_array_capture/mic_recordings/`` (NOT copied into repo).
Files: 4ch @ 16 kHz Int16, channel order FB / FF / TT / VPU  ⇒  FF=idx1, VPU=idx3.
DECLASSIFIED real-device — may load/process/visualize locally; this loader
returns ONLY the FF and VPU channels as float32 tensors (no audio bytes leak).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import soundfile as sf
import torch

ROOT = "/mnt/d/Projects/mic_array_capture/mic_recordings"
DIR_0624 = os.path.join(ROOT, "0624")


def list_0624() -> list[str]:
    return sorted(glob.glob(os.path.join(DIR_0624, "*.wav")))


def load_0624(name: Optional[str] = None, seg_s: float = 8.0,
              offset_s: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load one 0624 recording → (FF (1,T), VPU (1,T), sr).  Takes a `seg_s`
    segment from `offset_s` (skip the very start).  `name` = basename or None
    (first file).  Returns float32 mono tensors."""
    files = list_0624()
    if not files:
        raise FileNotFoundError(f"no 0624 wavs under {DIR_0624}")
    if name is None:
        path = files[0]
    else:
        path = next((f for f in files if os.path.basename(f) == name), None)
        if path is None:
            raise FileNotFoundError(f"{name} not in 0624; have {[os.path.basename(f) for f in files]}")
    y, sr = sf.read(path, dtype="float32", start=int(offset_s * 16000),
                    frames=int(seg_s * 16000))
    if y.ndim == 1:
        y = np.stack([y, y], 1)
    ff = y[:, 1].astype(np.float32)      # FF = idx1
    vpu = y[:, 3].astype(np.float32)     # VPU = idx3
    # peak-normalize each (so the fusion's level references are sane)
    ff = ff / (np.max(np.abs(ff)) + 1e-9)
    vpu = vpu / (np.max(np.abs(vpu)) + 1e-9)
    return (torch.from_numpy(ff).unsqueeze(0),
            torch.from_numpy(vpu).unsqueeze(0), sr)
