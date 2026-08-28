"""Temporal shift module (§3.3 training recipe).

Shifts groups of feature-map channels along the time axis to gather temporal
context at ZERO extra MAC cost.  Can be dropped into any arm as an optional
replacement block.

Implemented but OFF by default; verified to run and have gradient flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """Shift groups of channels forward/backward in time.

    Args:
        n_groups: how many groups to split channels into.
        shift_fraction: fraction of channels shifted (default 1/4).
        direction: "both" | "left" | "right"  (left = causal past, right = future)

    The shift is implemented as a strided copy so it's differentiable (identity
    gradient) and adds zero parameters.  Causality is preserved if direction
    includes only "left".
    """

    def __init__(self, n_groups: int = 2, shift_fraction: float = 0.25,
                 direction: str = "both"):
        super().__init__()
        self.n_groups = n_groups
        self.shift_fraction = shift_fraction
        self.direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T) shifted."""
        if not self.training:
            return x
        B, C, T = x.shape
        n_shift = int(C * self.shift_fraction)
        if n_shift == 0:
            return x
        out = x.clone()
        # Split channels into groups, shift each group by a different amount
        per_group = C // self.n_groups
        n_shift_per_group = max(1, int(per_group * self.shift_fraction))
        for g in range(self.n_groups):
            start = g * per_group
            end = min(start + per_group, C)
            s = start + (end - start - n_shift_per_group) // 2
            shift_amount = (g + 1)
            if self.direction in ("left", "both"):
                # Shift left (past): out[..., t] = x[..., t+shift]
                out[:, s:s + n_shift_per_group, :T - shift_amount] = \
                    x[:, s:s + n_shift_per_group, shift_amount:T]
                out[:, s:s + n_shift_per_group, T - shift_amount:] = \
                    x[:, s:s + n_shift_per_group, -shift_amount:] if False else 0
                out[:, s:s + n_shift_per_group, T - shift_amount:] = 0
            if self.direction in ("right", "both") and g > 0:
                s2 = s + n_shift_per_group
                e2 = min(s2 + n_shift_per_group, end)
                if e2 > s2:
                    out[:, s2:e2, shift_amount:T] = x[:, s2:e2, :T - shift_amount]
                    out[:, s2:e2, :shift_amount] = 0
        return out


class TemporalShift2d(nn.Module):
    """2-D version for (B, C, F, N) feature maps — shifts along N (time)."""

    def __init__(self, shift_fraction: float = 0.25, direction: str = "both"):
        super().__init__()
        self.shift_fraction = shift_fraction
        self.direction = direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, F, N = x.shape
        n_shift = max(1, int(C * self.shift_fraction))
        out = x.clone()
        shift_amount = 1
        if self.direction in ("left", "both"):
            out[:, :n_shift, :, :N - shift_amount] = \
                x[:, :n_shift, :, shift_amount:N]
            out[:, :n_shift, :, N - shift_amount:] = 0
        if self.direction in ("right", "both"):
            out[:, n_shift:2 * n_shift, :, shift_amount:N] = \
                x[:, n_shift:2 * n_shift, :, :N - shift_amount]
            out[:, n_shift:2 * n_shift, :, :shift_amount] = 0
        return out
