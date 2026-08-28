"""Magnitude-spectrum losses — the primary training objective (§3.1).

All arms output magnitude; phase is taken from the target (oracle) during
training.  This keeps phase as an isolated variable for this stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """L1 + L2 on magnitude in dB and linear domain.

    §5.7: log terms use eps calibrated to ≈ −80 dBFS, not 1e-8.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.5,
                 db_weight: float = 1.0, eps_db: float = -80.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.db_weight = db_weight
        self.eps_db = eps_db
        # eps in linear: 10^(-80/20) ≈ 1e-4
        self.eps_lin = 10.0 ** (eps_db / 20.0)

    def forward(self, pred_mag: torch.Tensor, target_mag: torch.Tensor) -> dict:
        # pred_mag, target_mag: (B, F, N)
        lin_eps = self.eps_lin
        p = pred_mag.clamp_min(lin_eps)
        t = target_mag.clamp_min(lin_eps)
        l1 = F.l1_loss(p, t)
        l2 = F.mse_loss(p, t)
        db_p = 20.0 * torch.log10(p)
        db_t = 20.0 * torch.log10(t)
        db_l1 = F.l1_loss(db_p, db_t)
        total = self.l1_weight * l1 + self.l2_weight * l2 + self.db_weight * db_l1
        return {"loss": total, "l1": l1.detach(), "l2": l2.detach(), "db_l1": db_l1.detach()}


def reconstruct_waveform_with_oracle_phase(mag: torch.Tensor,
                                            target_wav: torch.Tensor,
                                            stft_cfg) -> torch.Tensor:
    """Reconstruct a waveform from predicted magnitude using target's phase.

    §3.1: phase is taken from target (oracle) during training so we can apply
    waveform-based losses (multi-res STFT) if enabled.
    """
    from ..dsp.stft import stft, istft
    spec, _ = stft(target_wav, stft_cfg)  # target complex spectrogram
    phase = torch.angle(spec)
    # Align predicted magnitude frames to target spec frames
    N = min(mag.shape[-1], spec.shape[-1])
    mag = mag[..., :N]
    phase = phase[..., :N]
    # Combine predicted magnitude with oracle phase
    recon_spec = mag * torch.exp(1j * phase)
    wav = istft(recon_spec, stft_cfg, length=target_wav.shape[-1])
    return wav
