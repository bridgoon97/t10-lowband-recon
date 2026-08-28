"""Arm B — Spectral regression CRN (Convolution-Recurrent Network).

Directly regresses the target magnitude spectrum from the input magnitude
spectrum.  Inspired by GTCRN (grouped convolutions + sub-band features +
temporal recurrence), scaled to 4 kHz per §2 budget.

Target: ~25–40 K params, well under 60 MMACs/s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import LowBandReconstructor
from ..dsp.stft import StftConfig as _StftConfig
from ..dsp.stft import causal_stft as _causal_stft
from ..dsp.stft import stft as _stft_fn, frame_step as _frame_step


class CRNEncoder(nn.Module):
    """Frequency-downsampling conv stack (treats F as spatial, N as time)."""

    def __init__(self, n_bins: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(4, 8, (3, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(8, 16, (3, 1), stride=(2, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(16, 16, (3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        # x: (B, 1, F, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x  # (B, 16, F//8, N)


class CRNDecoder(nn.Module):
    """Frequency-upsampling conv-transpose stack with skip connections."""

    def __init__(self, n_bins: int):
        super().__init__()
        self.t1 = nn.ConvTranspose2d(16, 16, (3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.t2 = nn.ConvTranspose2d(16, 8, (3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.t3 = nn.ConvTranspose2d(8, 4, (3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.out = nn.Conv2d(4, 1, (3, 1), padding=(1, 0))

    def forward(self, x, skips):
        # skips: list of encoder outputs at matching resolutions
        x = F.relu(self.t1(x))
        if skips[0] is not None:
            x = x + skips[0]
        x = F.relu(self.t2(x))
        if skips[1] is not None:
            x = x + skips[1]
        x = F.relu(self.t3(x))
        if skips[2] is not None:
            x = x + skips[2]
        x = self.out(x)  # (B, 1, F', N)
        x = x.squeeze(1)  # (B, F', N)
        return x  # caller crops to n_bins if needed


class ArmB_CRN(LowBandReconstructor):
    """CRN spectral regressor."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.sample_rate = cfg["sample_rate"]
        self.stft_cfg = _StftConfig(
            n_fft=cfg.get("stft_n_fft", 128),
            hop=cfg.get("stft_hop", 32),
            win=cfg.get("stft_win", 128),
            window=cfg.get("stft_window", "hann"),
        )
        self.n_bins = self.stft_cfg.num_bins
        self.encoder = CRNEncoder(self.n_bins)
        # Compute actual encoder output frequency dim via a dry run
        with torch.no_grad():
            _dummy = torch.zeros(1, 1, self.n_bins, 16)
            _enc_out = self.encoder(_dummy)
            self._enc_channels = _enc_out.shape[1]
            self._enc_freq = _enc_out.shape[2]
        feat_dim = self._enc_channels * self._enc_freq
        self.gru = nn.GRU(feat_dim, 48, batch_first=True)
        self.gru_proj = nn.Linear(48, feat_dim)
        self.decoder = CRNDecoder(self.n_bins)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        _, in_mag = _causal_stft(x, self.stft_cfg)  # (B, F, N)
        B, Fb, N = in_mag.shape
        enc = self.encoder(in_mag.unsqueeze(1))  # (B, 16, F//8, N)
        Fd = enc.shape[1]
        feat = enc.permute(0, 3, 1, 2).contiguous()  # (B, N, 16, F//8)
        feat = feat.reshape(B, N, -1)               # (B, N, 16*F//8)
        out, _ = self.gru(feat)                     # (B, N, 64)
        out = self.gru_proj(out)                    # (B, N, feat_dim)
        out = out.reshape(B, N, Fd, -1).permute(0, 2, 3, 1).contiguous()  # (B, 16, F//8, N)
        mag = self.decoder(out, skips=[None, None, None])  # (B, F', N)
        # Crop to n_bins (decoder may overshoot due to non-power-of-2 n_bins)
        mag = mag[:, :self.n_bins, :]
        mag = F.softplus(mag)  # non-negative output
        return {"mag": mag, "wav": None, "aux": {"input_mag": in_mag.detach()}}

    def stream_init(self, batch_size: int) -> dict:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        return {
            "stft_tail": torch.zeros(batch_size, win - hop),
            "gru_h": None,
        }

    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        mag_frame, new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])
        # mag_frame: (B, F) -> (B, 1, F, 1)
        enc = self.encoder(mag_frame.unsqueeze(1).unsqueeze(-1))  # (B, 16, F//8, 1)
        B = enc.shape[0]
        Fd = enc.shape[1]  # channels (16)
        Fe = enc.shape[2]  # freq bins
        feat = enc.permute(0, 3, 1, 2).contiguous().reshape(B, 1, -1)  # (B, 1, feat_dim)
        out, h = self.gru(feat, state["gru_h"])  # (B, 1, 48)
        out = self.gru_proj(out)                 # (B, 1, feat_dim=ch*Fe)
        out = out.reshape(B, 1, self._enc_channels, Fe).permute(0, 2, 3, 1).contiguous()  # (B, ch, Fe, 1)
        mag = self.decoder(out, skips=[None, None, None]).squeeze(-1)  # (B, F')
        mag = mag[:, :self.n_bins]  # crop to n_bins
        mag = F.softplus(mag)
        new_state = {"stft_tail": new_tail, "gru_h": h.detach()}
        return mag, new_state
