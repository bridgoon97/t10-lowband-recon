"""Multi-resolution STFT loss (§3.3 training recipe).

Implemented but OFF by default — only verified to run and have gradient flow.
The window set is configurable; default [64, 128, 256] @4kHz.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 for a single STFT config."""

    def __init__(self, n_fft: int, hop: int, win: int, window: str = "hann",
                 eps: float = 1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.eps = eps
        if window == "hann":
            self.register_buffer("window", torch.hann_window(win, periodic=True))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(win, periodic=True))
        else:
            self.register_buffer("window", torch.ones(win))

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        w = self.window.to(pred_wav.device)
        pred_spec = torch.stft(pred_wav, n_fft=self.n_fft, hop_length=self.hop,
                                win_length=self.win, window=w, return_complex=True)
        tgt_spec = torch.stft(target_wav, n_fft=self.n_fft, hop_length=self.hop,
                               win_length=self.win, window=w, return_complex=True)
        pred_mag = pred_spec.abs().clamp_min(self.eps)
        tgt_mag = tgt_spec.abs().clamp_min(self.eps)
        # Spectral convergence
        sc = torch.norm(tgt_mag - pred_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + self.eps)
        # Log-magnitude L1
        log_mag = F.l1_loss(torch.log(pred_mag), torch.log(tgt_mag))
        return sc + log_mag


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of STFTLoss over a set of window/hop sizes.

    spec change: windows [240, 480, 960] @16 kHz = 15/30/60 ms (was
    [64,128,256] @4 kHz).  Multi-res with longer windows for 16 kHz speech.
    """

    def __init__(self, window_sizes=(240, 480, 960), hop_ratio: float = 0.25,
                 window: str = "hann"):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(n_fft=ws, hop=max(1, int(ws * hop_ratio)), win=ws, window=window)
            for ws in window_sizes
        ])

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for loss_fn in self.losses:
            total = total + loss_fn(pred_wav, target_wav)
        return total / len(self.losses)
