"""Arm C — Frequency-then-Time LSTM (FT-JNF style).

Processes the spectrum with a frequency-direction LSTM (runs along the keep_bins
freq bins for each time frame) followed by a time-direction LSTM (runs along
time for each freq bin).  XS-scale ~13 K params.

§3.2: "参数省但 MAC 贵" — params small but MACs expensive; the complexity
table reports the actual MAC cost honestly.

Spec change (口径迁移): input/output are the TRUNCATED COMPLEX spectrum
(B, keep_bins=65, N).  Complex is carried as 2 real/imag features at the LSTM
input and 2 at the projection output (complex64 in/out, real arithmetic inside).
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
    """(..., F, N) complex -> (..., F, N, 2) real/imag (last dim 2)."""
    return torch.stack([spec.real, spec.imag], dim=-1)


def _ri_to_cplx(ri: torch.Tensor) -> torch.Tensor:
    """(..., F, N, 2) -> (..., F, N) complex."""
    return torch.complex(ri[..., 0], ri[..., 1])


class FTLSTMBlock(nn.Module):
    """F-LSTM -> T-LSTM -> linear projection (complex in / complex out)."""

    def __init__(self, n_bins: int, hidden: int = 32):
        super().__init__()
        self.n_bins = n_bins
        self.hidden = hidden
        # F-LSTM: processes freq bins as a sequence; input dim=2 (real/imag)
        self.f_lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True,
                              bidirectional=False)
        # T-LSTM: processes time frames as a sequence; input dim=hidden
        self.t_lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True,
                              bidirectional=False)
        # proj -> 2 (real/imag) of the predicted COMPLEX spectrum
        self.proj = nn.Linear(hidden, 2)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: (B, F, N) complex -> predicted complex (B, F, N)."""
        B, Fb, N = spec.shape
        ri = _cplx_to_ri(spec)                 # (B, F, N, 2)
        # F-LSTM: for each time frame, run LSTM along frequency
        f_in = ri.permute(0, 2, 1, 3).contiguous().reshape(B * N, Fb, 2)  # (B*N, F, 2)
        f_out, _ = self.f_lstm(f_in)           # (B*N, F, hidden)
        f_out = f_out.reshape(B, N, Fb, self.hidden)

        # T-LSTM: for each freq bin, run LSTM along time
        t_in = f_out.permute(0, 2, 1, 3).contiguous().reshape(B * Fb, N, self.hidden)
        t_out, _ = self.t_lstm(t_in)           # (B*F, N, hidden)
        t_out = t_out.reshape(B, Fb, N, self.hidden)

        ri_out = self.proj(t_out)              # (B, F, N, 2)
        return _ri_to_cplx(ri_out)             # (B, F, N) complex

    def forward_frame(self, spec_frame: torch.Tensor, f_state, t_states) -> tuple[torch.Tensor, tuple]:
        """Streaming: process one time frame, spec_frame (B, F) complex.

        F-LSTM has NO temporal state (frequency-sequence per frame), so it
        always starts fresh.  Only the T-LSTM carries state across frames.
        """
        B, Fb = spec_frame.shape
        f_in = _cplx_to_ri(spec_frame)        # (B, F, 2)
        f_out, _ = self.f_lstm(f_in, None)    # (B, F, hidden)

        if t_states is None:
            t_states = [None] * Fb
        new_t_states = []
        pred_parts = []
        for fb in range(Fb):
            t_in = f_out[:, fb:fb + 1, :]     # (B, 1, hidden)
            t_out, t_s = self.t_lstm(t_in, t_states[fb])
            new_t_states.append(t_s)
            pred_parts.append(self.proj(t_out))  # (B, 1, 2)
        ri_out = torch.cat(pred_parts, dim=1)   # (B, F, 2)
        return _ri_to_cplx(ri_out), (None, new_t_states)  # (B, F) complex


class ArmC_FTLSTM(LowBandReconstructor):
    """F-T LSTM reconstructor (complex in / complex out)."""

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
        self.hidden = cfg.get("ftlstm_hidden", 32)
        self.block = FTLSTMBlock(self.n_bins, self.hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        in_spec = _complex_stft(x, self.stft_cfg)   # (B, 65, N) complex
        spec = self.block(in_spec)
        return {"spec": spec, "wav": None,
                "aux": {"input_mag": in_spec.abs().detach()}}

    def stream_init(self, batch_size: int) -> dict:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        return {"stft_tail": torch.zeros(batch_size, win - hop),
                "f_state": None, "t_states": None}

    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        spec_frame, new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])
        pred, (f_new, t_new) = self.block.forward_frame(
            spec_frame, state["f_state"], state["t_states"])
        new_state = {
            "stft_tail": new_tail,
            "f_state": (tuple(s.detach() for s in f_new) if isinstance(f_new, tuple) else f_new),
            "t_states": [(tuple(s2.detach() for s2 in s) if isinstance(s, tuple) else s) for s in t_new],
        }
        return pred, new_state
