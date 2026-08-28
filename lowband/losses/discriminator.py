"""Multi-subband discriminator + feature matching (§3.3).

The discriminator is NOT constrained by §2 budget (only exists during training).
We build a PQMF-analysis → per-band discriminator → feature matching pipeline.

Implemented but OFF by default; verified to run and have gradient flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dsp.pqmf import PQMFAnalysis


class SubbandDiscriminator(nn.Module):
    """A small 1-D conv discriminator operating on one subband.

    Scaled up freely — discriminator size is unconstrained by §2.
    """

    def __init__(self, n_bands: int = 4, channels: tuple = (16, 32, 64),
                 kernel: int = 5):
        super().__init__()
        layers = []
        prev = n_bands  # operates on full band-mixed subbands
        for ch in channels:
            layers.append(nn.Conv1d(prev, ch, kernel, stride=2, padding=kernel // 2))
            layers.append(nn.LeakyReLU(0.1))
            prev = ch
        layers.append(nn.Conv1d(prev, 1, kernel, padding=kernel // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: (B, C, T) -> (logits, [features per layer])."""
        feats = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                feats.append(h)
        return h, feats


class MultiSubbandDiscriminator(nn.Module):
    """PQMF → multi-discriminator ensemble.

    Operates on waveforms (``pred_wav`` and ``target_wav``).
    """

    def __init__(self, n_bands: int = 4, n_discriminators: int = 3,
                 channels: tuple = (16, 32, 64), n_taps: int = 64):
        super().__init__()
        self.pqmf = PQMFAnalysis(n_bands=n_bands, n_taps=n_taps)
        self.discriminators = nn.ModuleList([
            SubbandDiscriminator(n_bands=n_bands, channels=channels)
            for _ in range(n_discriminators)
        ])

    def forward(self, pred_wav: torch.Tensor,
                target_wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (adv_loss, feat_match_loss) for the generator update.

        The discriminator training (hinge loss) is computed in the training loop.
        """
        pred_sub = self.pqmf(pred_wav)   # (B, n_bands, T')
        tgt_sub = self.pqmf(target_wav)

        # Treat each band as a channel: (B, n_bands, T')
        feat_loss = 0.0
        adv_loss = 0.0
        for disc in self.discriminators:
            pred_logits, pred_feats = disc(pred_sub)
            tgt_logits, tgt_feats = disc(tgt_sub)
            # Feature matching: L1 between pred and target features
            for pf, tf in zip(pred_feats, tgt_feats):
                feat_loss = feat_loss + F.l1_loss(pf, tf.detach())
            # Adversarial: generator wants pred to be classified as real
            adv_loss = adv_loss + F.softplus(-pred_logits).mean()
        n = len(self.discriminators)
        return adv_loss / n, feat_loss / n

    def disc_loss(self, pred_wav: torch.Tensor,
                  target_wav: torch.Tensor) -> torch.Tensor:
        """Discriminator hinge loss (called separately from generator loss)."""
        pred_sub = self.pqmf(pred_wav.detach())
        tgt_sub = self.pqmf(target_wav)
        loss = 0.0
        for disc in self.discriminators:
            pred_logits, _ = disc(pred_sub)
            tgt_logits, _ = disc(tgt_sub)
            loss = loss + (F.relu(1.0 - tgt_logits).mean() +
                           F.relu(1.0 + pred_logits).mean())
        return loss / len(self.discriminators)
