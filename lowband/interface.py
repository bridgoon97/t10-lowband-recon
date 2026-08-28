"""Unified arm interface (task spec §3.1 — the most important section).

Every arm implements ``LowBandReconstructor``.  Training loop, data pipeline,
loss, eval and export are all shared; the arm is switched by config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class LowBandReconstructor(ABC, nn.Module):
    """Base contract for all three arms.

    forward(x, cond=None) -> dict
        x:    (B, T) waveform @4kHz, band-limited, normalized
        cond: (B, C, F) optional conditioning (None this stage)
        returns: {
            "mag": (B, F, N)   predicted 0–2k magnitude (REQUIRED)
            "wav": (B, T)      optional synthesized waveform (Arm A lock-phase only)
            "aux": dict        optional diagnostics (f0, periodicity, envelope)
        }

    stream_init(batch_size) -> state
    stream_step(x_frame, state) -> (out_frame, state)
        Frame-by-frame streaming; must be numerically equivalent to forward (§5.3).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        ...

    @abstractmethod
    def stream_init(self, batch_size: int) -> dict:
        ...

    @abstractmethod
    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        ...

    # --- shared helpers ----------------------------------------------------
    @property
    def arm_name(self) -> str:
        return self.__class__.__name__

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
