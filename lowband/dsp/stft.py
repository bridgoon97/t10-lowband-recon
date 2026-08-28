"""STFT / iSTFT for 4 kHz speech.

Single shared analysis/synthesis engine used by all arms and losses.
Window/hop/n_fft come from config so nothing is hardcoded.

COLA (Constant Overlap-Add) is verified in tests/test_stft_roundtrip.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def get_window(name: str, n_fft: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a window of length ``n_fft`` built on CPU then moved to device."""
    if name == "hann":
        w = torch.hann_window(n_fft, periodic=True)
    elif name == "hamming":
        w = torch.hamming_window(n_fft, periodic=True)
    elif name in ("rect", "rectangular"):
        w = torch.ones(n_fft)
    elif name == "blackman":
        w = torch.blackman_window(n_fft, periodic=True)
    else:
        raise ValueError(f"unknown window '{name}'")
    return w.to(device=device, dtype=dtype)


class StftConfig:
    """Immutable-ish config for the shared STFT."""

    __slots__ = ("n_fft", "hop", "win", "window", "center", "pad_mode")

    def __init__(self, n_fft: int = 128, hop: int = 32, win: int = 128,
                 window: str = "hann", center: bool = True, pad_mode: str = "reflect"):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    @property
    def num_bins(self) -> int:
        return self.n_fft // 2 + 1

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def causal_stft(x: torch.Tensor, cfg: StftConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal STFT matching ``frame_step`` exactly (left-pad only, center=False).

    This guarantees batch/streaming numerical equivalence (§5.3).
    """
    w = get_window(cfg.window, cfg.win, device=x.device, dtype=x.dtype)
    left_pad = cfg.win - cfg.hop
    xp = F.pad(x, (left_pad, 0), mode="constant")  # left-pad with zeros
    spec = torch.stft(
        xp, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        window=w, center=False, pad_mode="constant",
        return_complex=True,
    )  # (B, F, N)
    mag = spec.abs()
    return spec, mag


def stft(x: torch.Tensor, cfg: StftConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute STFT.

    Args:
        x: (B, T) waveforms.
        cfg: StftConfig.

    Returns:
        spec: (B, F, N) complex
        mag: (B, F, N) magnitude
    """
    w = get_window(cfg.window, cfg.win, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        window=w, center=cfg.center, pad_mode=cfg.pad_mode,
        return_complex=True,
    )  # (B, F, N)
    mag = spec.abs()
    return spec, mag


def istft(spec: torch.Tensor, cfg: StftConfig, length: int | None = None) -> torch.Tensor:
    """Inverse STFT."""
    w = get_window(cfg.window, cfg.win, device=spec.device, dtype=torch.float32)
    return torch.istft(
        spec, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        window=w, center=cfg.center, length=length,
    )


def mag_to_db(mag: torch.Tensor, ref: float = 1.0, min_db: float = -80.0) -> torch.Tensor:
    """Convert magnitude to dB with a floor at ``min_db`` dB (≈ -80 dBFS)."""
    floor = ref * (10.0 ** (min_db / 20.0))
    return 20.0 * torch.log10(mag.clamp_min(floor) / ref)


def db_to_mag(db: torch.Tensor, ref: float = 1.0) -> torch.Tensor:
    return ref * (10.0 ** (db / 20.0))


# --- streaming (causal) STFT frame step -------------------------------------
def frame_step(x_frame: torch.Tensor, cfg: StftConfig, prev_tail: torch.Tensor | None):
    """Process one hop of samples through a causal STFT step.

    Args:
        x_frame: (B, hop) new samples.
        cfg: StftConfig.
        prev_tail: (B, win-hop) overlap buffer from the previous frame, or None.

    Returns:
        mag: (B, F) magnitude for this frame.
        new_tail: (B, win-hop) overlap buffer to carry to the next frame.
    """
    B = x_frame.shape[0]
    hop, win = cfg.hop, cfg.win
    if prev_tail is None:
        prev_tail = x_frame.new_zeros(B, win - hop)
    buf = torch.cat([prev_tail, x_frame], dim=1)  # (B, win)
    w = get_window(cfg.window, cfg.win, device=x_frame.device, dtype=x_frame.dtype)
    windowed = buf * w.unsqueeze(0)
    spec = torch.fft.rfft(windowed, n=cfg.n_fft)  # (B, F)
    mag = spec.abs()
    new_tail = buf[:, hop:]
    return mag, new_tail


def cola_check(cfg: StftConfig) -> float:
    """Verify the Constant-Overlap-Add (COLA) condition for the configured window.

    Returns the max relative deviation of the window-OLA sum from its mean.
    A value < 1e-6 indicates a numerically perfect COLA window/hop pair.
    """
    w = get_window(cfg.window, cfg.win)
    # Build an overlap-add of length 3*win to reach steady state.
    L = 3 * cfg.win
    acc = torch.zeros(L)
    n_starts = torch.arange(0, L - cfg.win + 1, cfg.hop)
    for s in n_starts.tolist():
        acc[s:s + cfg.win] += w
    # Steady-state region: samples well inside the buffer.
    interior = acc[cfg.win:2 * cfg.win]
    mean = interior.mean()
    if mean.item() == 0:
        return float("inf")
    return float((interior - mean).abs().max().item() / mean.item())
