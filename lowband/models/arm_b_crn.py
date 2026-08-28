"""Arm B — Spectral regression CRN (Convolution-Recurrent Network).

Directly regresses the target spectrum from the input spectrum.  Inspired by
GTCRN (grouped convolutions + sub-band features + temporal recurrence), scaled
per §2 budget.

Spec change (口径迁移): the model's input feature AND output are now the
TRUNCATED COMPLEX spectrum (B, keep_bins=65, N) of the 0–2000 Hz band — not a
magnitude template.  Phase is therefore the model's job (learned), no longer
oracle.  Internally the complex spectrum is carried as a 2-channel real/imag
tensor through the conv/GRU stack (complex64 in, complex64 out; real arithmetic
in between).

Target: ~25–40 K params, well under 60 MMACs/s.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import LowBandReconstructor
from ..dsp.stft import StftConfig as _StftConfig
from ..dsp.stft import complex_stft_truncated as _complex_stft
from ..dsp.stft import frame_step as _frame_step


def _cplx_to_ri(spec: torch.Tensor) -> torch.Tensor:
    """(B, F, N) complex -> (B, 2, F, N) real/imag."""
    return torch.stack([spec.real, spec.imag], dim=1)


def _ri_to_cplx(ri: torch.Tensor) -> torch.Tensor:
    """(B, 2, F, N) real/imag -> (B, F, N) complex."""
    return torch.complex(ri[:, 0], ri[:, 1])


class CRNEncoder(nn.Module):
    """Frequency-downsampling conv stack (treats F as spatial, N as time)."""

    def __init__(self, n_bins: int):
        super().__init__()
        # 2 input channels = real/imag of the complex input spectrum
        self.conv1 = nn.Conv2d(2, 4, (3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(4, 8, (3, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(8, 16, (3, 1), stride=(2, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(16, 16, (3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        # x: (B, 2, F, N)
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
        # 2 output channels = real/imag of the predicted COMPLEX spectrum
        self.out = nn.Conv2d(4, 2, (3, 1), padding=(1, 0))

    def forward(self, x, skips):
        x = F.relu(self.t1(x))
        if skips[0] is not None:
            x = x + skips[0]
        x = F.relu(self.t2(x))
        if skips[1] is not None:
            x = x + skips[1]
        x = F.relu(self.t3(x))
        if skips[2] is not None:
            x = x + skips[2]
        x = self.out(x)  # (B, 2, F', N)
        return x  # caller crops to n_bins and recombines to complex


class ArmB_CRN(LowBandReconstructor):
    """CRN spectral regressor (complex in / complex out)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.sample_rate = cfg["sample_rate"]
        self.stft_cfg = _StftConfig(
            n_fft=cfg.get("stft_n_fft", 512),
            hop=cfg.get("stft_hop", 160),
            win=cfg.get("stft_win", 480),
            window=cfg.get("stft_window", "hann"),
            keep_bins=cfg.get("keep_bins", 64),
        )
        self.n_bins = self.stft_cfg.keep_bins
        self.encoder = CRNEncoder(self.n_bins)
        with torch.no_grad():
            _dummy = torch.zeros(1, 2, self.n_bins, 16)
            _enc_out = self.encoder(_dummy)
            self._enc_channels = _enc_out.shape[1]
            self._enc_freq = _enc_out.shape[2]
        feat_dim = self._enc_channels * self._enc_freq
        self.gru = nn.GRU(feat_dim, 48, batch_first=True)
        self.gru_proj = nn.Linear(48, feat_dim)
        self.decoder = CRNDecoder(self.n_bins)

    def _decode(self, enc, N):
        """enc: (B, 16, Fe, N) -> complex spec (B, n_bins, N)."""
        B = enc.shape[0]
        Fd = enc.shape[1]
        feat = enc.permute(0, 3, 1, 2).contiguous().reshape(B, N, -1)
        out, _ = self.gru(feat)
        out = self.gru_proj(out)
        out = out.reshape(B, N, Fd, -1).permute(0, 2, 3, 1).contiguous()  # (B,16,Fe,N)
        ri = self.decoder(out, skips=[None, None, None])  # (B, 2, F', N)
        ri = ri[:, :, :self.n_bins]      # crop to n_bins freq
        return _ri_to_cplx(ri)           # (B, n_bins, N) complex

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        in_spec = _complex_stft(x, self.stft_cfg)   # (B, 65, N) complex
        B, Fb, N = in_spec.shape
        enc = self.encoder(_cplx_to_ri(in_spec))     # (B, 16, Fe, N)
        spec = self._decode(enc, N)
        return {"spec": spec, "wav": None,
                "aux": {"input_mag": in_spec.abs().detach()}}

    def stream_init(self, batch_size: int) -> dict:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        return {"stft_tail": torch.zeros(batch_size, win - hop),
                "gru_h": None}

    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        mag_frame, new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])
        # mag_frame: (B, keep_bins) complex -> (B, 2, keep_bins, 1)
        enc = self.encoder(_cplx_to_ri(mag_frame).unsqueeze(-1))  # (B, 16, Fe, 1)
        B = enc.shape[0]
        Fd = enc.shape[1]
        Fe = enc.shape[2]
        feat = enc.permute(0, 3, 1, 2).contiguous().reshape(B, 1, -1)
        out, h = self.gru(feat, state["gru_h"])
        out = self.gru_proj(out)
        out = out.reshape(B, 1, self._enc_channels, Fe).permute(0, 2, 3, 1).contiguous()
        ri = self.decoder(out, skips=[None, None, None]).squeeze(-1)  # (B, 2, F')
        ri = ri[:, :, :self.n_bins]
        spec = _ri_to_cplx(ri)  # (B, n_bins) complex
        new_state = {"stft_tail": new_tail, "gru_h": h.detach()}
        return spec, new_state
