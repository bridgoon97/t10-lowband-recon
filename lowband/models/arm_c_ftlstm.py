"""Arm C — Frequency-then-Time LSTM (FT-JNF style).

Processes the magnitude spectrum with a frequency-direction LSTM (runs along
the 65 freq bins for each time frame) followed by a time-direction LSTM (runs
along time for each freq bin).  XS-scale ~13 K params.

§3.2: "参数省但 MAC 贵" — params are small but MACs are expensive; the
complexity table reports the actual MAC cost honestly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import LowBandReconstructor
from ..dsp.stft import StftConfig as _StftConfig
from ..dsp.stft import causal_stft as _causal_stft
from ..dsp.stft import stft as _stft_fn, frame_step as _frame_step


class FTLSTMBlock(nn.Module):
    """F-LSTM -> T-LSTM -> linear projection."""

    def __init__(self, n_bins: int, hidden: int = 32):
        super().__init__()
        self.n_bins = n_bins
        self.hidden = hidden
        # F-LSTM: processes freq bins as a sequence (input dim=1)
        self.f_lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True,
                              bidirectional=False)
        # T-LSTM: processes time frames as a sequence (input dim=hidden)
        self.t_lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True,
                              bidirectional=False)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        """mag: (B, F, N) -> predicted magnitude (B, F, N)."""
        B, Fb, N = mag.shape
        # F-LSTM: for each time frame, run LSTM along frequency
        # Reshape: (B*N, F, 1)
        f_in = mag.permute(0, 2, 1).contiguous().reshape(B * N, Fb, 1)
        f_out, _ = self.f_lstm(f_in)  # (B*N, F, hidden)
        f_out = f_out.reshape(B, N, Fb, self.hidden)

        # T-LSTM: for each freq bin, run LSTM along time
        # Reshape: (B*F, N, hidden)
        t_in = f_out.permute(0, 2, 1, 3).contiguous().reshape(B * Fb, N, self.hidden)
        t_out, _ = self.t_lstm(t_in)  # (B*F, N, hidden)
        t_out = t_out.reshape(B, Fb, N, self.hidden)

        pred = self.proj(t_out).squeeze(-1)  # (B, F, N)
        return pred

    def forward_frame(self, mag_frame: torch.Tensor, f_state, t_states) -> tuple[torch.Tensor, tuple]:
        """Streaming: process one time frame (B, F).

        F-LSTM has NO temporal state (it processes freq bins per frame), so it
        always starts from zero.  Only the T-LSTM carries state across frames.
        """
        B, Fb = mag_frame.shape
        f_in = mag_frame.unsqueeze(-1)  # (B, F, 1)
        # F-LSTM: always fresh (frequency-sequence, no inter-frame state)
        f_out, _ = self.f_lstm(f_in, None)  # (B, F, hidden)

        # T-LSTM: for each freq bin, step the temporal LSTM one frame
        # f_out: (B, F, hidden) -> per-freq: (B, 1, hidden) for each freq
        # t_state: (F, B, hidden*?) -> we need per-freq states
        pred_parts = []
        if t_states is None:
            t_states = [None] * Fb
        new_t_states = []
        for fb in range(Fb):
            t_in = f_out[:, fb:fb + 1, :]  # (B, 1, hidden)
            t_out, t_s = self.t_lstm(t_in, t_states[fb])  # (B, 1, hidden)
            new_t_states.append(t_s)
            pred_parts.append(self.proj(t_out).squeeze(-1))  # (B, 1)
        pred = torch.cat(pred_parts, dim=-1)  # (B, F)
        return pred, (None, new_t_states)


class ArmC_FTLSTM(LowBandReconstructor):
    """F-T LSTM reconstructor."""

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
        self.hidden = cfg.get("ftlstm_hidden", 32)
        self.block = FTLSTMBlock(self.n_bins, self.hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict:
        _, in_mag = _causal_stft(x, self.stft_cfg)  # (B, F, N)
        pred = self.block(in_mag)
        mag = F.softplus(pred)
        return {"mag": mag, "wav": None, "aux": {"input_mag": in_mag.detach()}}

    def stream_init(self, batch_size: int) -> dict:
        hop = self.stft_cfg.hop
        win = self.stft_cfg.win
        return {
            "stft_tail": torch.zeros(batch_size, win - hop),
            "f_state": None,
            "t_states": None,
        }

    def stream_step(self, x_frame: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        mag_frame, new_tail = _frame_step(x_frame, self.stft_cfg, state["stft_tail"])
        pred, (f_new, t_new) = self.block.forward_frame(
            mag_frame, state["f_state"], state["t_states"])
        mag = F.softplus(pred)
        new_state = {
            "stft_tail": new_tail,
            "f_state": (tuple(s.detach() for s in f_new) if isinstance(f_new, tuple) else f_new),
            "t_states": [(tuple(s2.detach() for s2 in s) if isinstance(s, tuple) else s) for s in t_new],
        }
        return mag, new_state
