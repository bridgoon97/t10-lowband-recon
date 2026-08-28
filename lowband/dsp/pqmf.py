"""PQMF (Pseudo Quadrature Mirror Filter) bank for the multi-subband
discriminator (§3.3 training recipe).

Each branch of the bank is a linear-phase prototype filter modulated by
cosine carriers, giving near-perfect reconstruction with a synthesis bank.

This is implemented but OFF by default (training recipe only).  The
discriminator itself is in lowband/losses/discriminator.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _mq_proto(n_taps: int, device, dtype) -> torch.Tensor:
    """Soft prototype filter weights that get optimized during training.

    We initialize with a cosine window; the bank is allowed to adapt.
    """
    n = torch.arange(n_taps, device=device, dtype=dtype)
    proto = torch.sin(math.pi * (n + 0.5) / n_taps) ** 2
    return proto


class PQMFAnalysis(torch.nn.Module):
    """Analysis PQMF bank: 1 signal -> ``n_bands`` subband signals."""

    def __init__(self, n_bands: int = 4, n_taps: int = 64):
        super().__init__()
        self.n_bands = n_bands
        self.n_taps = n_taps
        # Prototype filter (learnable)
        self.proto = torch.nn.Parameter(_mq_proto(n_taps, torch.device("cpu"), torch.float32))
        self._build_modulated()

    def _build_modulated(self):
        # Modulation matrix: (n_bands, n_taps)
        k = torch.arange(self.n_bands).float().unsqueeze(1)
        n = torch.arange(self.n_taps).float().unsqueeze(0)
        cos_bank = torch.cos((2 * k + 1) * math.pi / (2 * self.n_bands) *
                             (n - (self.n_taps - 1) / 2.0))
        # Register as buffer (not learnable — derived from proto)
        self.register_buffer("cos_bank", cos_bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, n_bands, T') downsampled subbands."""
        B, T = x.shape
        hop = self.n_bands
        # Build analysis filters by modulating prototype
        proto = self.proto.view(1, 1, -1)  # (1,1,n_taps)
        filters = self.proto.unsqueeze(0) * self.cos_bank  # (n_bands, n_taps)
        filters = filters.unsqueeze(1)  # (n_bands, 1, n_taps)
        x_pad = F.pad(x.unsqueeze(1), (self.n_taps // 2, self.n_taps // 2))
        sub = F.conv1d(x_pad, filters, stride=hop)  # (B, n_bands, T//hop)
        return sub


class PQMFSynthesis(torch.nn.Module):
    """Synthesis PQMF bank: ``n_bands`` subbands -> 1 signal."""

    def __init__(self, analysis: PQMFAnalysis):
        super().__init__()
        self.n_bands = analysis.n_bands
        self.n_taps = analysis.n_taps
        self.proto = analysis.proto
        self.register_buffer("cos_bank", analysis.cos_bank)

    def forward(self, sub: torch.Tensor) -> torch.Tensor:
        """sub: (B, n_bands, T') -> (B, T)."""
        B, n_bands, Tp = sub.shape
        hop = self.n_bands
        filters = self.proto.unsqueeze(0) * self.cos_bank  # (n_bands, n_taps)
        # Transposed conv for upsampling
        out = F.conv_transpose1d(sub, filters, stride=hop)  # (B, n_bands, T'')
        out = out.sum(dim=1)  # mix bands
        return out
