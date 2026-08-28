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
    """Immutable-ish config for the shared STFT.

    New 口径 (spec change): the model operates on a TRUNCATED complex
    spectrum of ``keep_bins`` bins 1..keep_bins (DC bin 0 DROPPED — it has a
    dead imaginary channel for a real signal).  At sr=16 kHz, n_fft=512 →
    31.25 Hz/bin, keep_bins=64 → bins 1..64 = 31.25–2000 Hz (the reconstruction
    band).  The full STFT has n_fft//2+1 bins (up to Nyquist 8 kHz); only bins
    1..keep_bins are kept as the model's input feature / output.  bin↔Hz uses
    ``bin_to_hz``/``hz_to_bin`` (the +1 offset, single source of truth).
    """

    __slots__ = ("n_fft", "hop", "win", "window", "center", "pad_mode", "keep_bins")

    def __init__(self, n_fft: int = 512, hop: int = 160, win: int = 480,
                 window: str = "hann", center: bool = True, pad_mode: str = "reflect",
                 keep_bins: int | None = None):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        # default: keep the low spectrum with DC dropped (bins 1..keep_bins).
        # 64 → bins 1..64 = 31.25..2000 Hz at sr=16k/n_fft=512. 64 is conv-
        # friendly (64→32→16→8) and dropping DC removes a dead imaginary channel.
        self.keep_bins = keep_bins if keep_bins is not None else 64

    @property
    def num_bins(self) -> int:
        """Full STFT bin count (n_fft//2+1, up to Nyquist)."""
        return self.n_fft // 2 + 1

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


def causal_stft(x: torch.Tensor, cfg: StftConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal STFT matching ``frame_step`` EXACTLY (left-pad only, manual framing).

    Why manual framing (not torch.stft): when ``win != n_fft`` (new 口径:
    win=480, n_fft=512), torch.stft's center=False framing/window placement
    does not match a streaming ``frame_step`` that buffers ``win`` samples per
    hop — so batch/streaming equivalence (§5.3, a hard requirement) breaks.
    Framing manually (unfold win-length frames at stride hop, window, rfft n_fft)
    makes the batch op byte-for-byte identical to the streaming per-frame op, so
    equivalence holds by construction.  Differentiable (unfold + rfft).

    Returns the FULL complex spectrogram (B, num_bins, N) and its magnitude;
    callers wanting the model's truncated input/target use
    ``complex_stft_truncated``.
    """
    B, T = x.shape
    left_pad = cfg.win - cfg.hop
    xp = F.pad(x, (left_pad, 0), mode="constant")  # left-pad with zeros
    n_frames = 1 + (xp.shape[1] - cfg.win) // cfg.hop
    if n_frames < 1:
        # signal shorter than one window: pad up to one frame
        xp = F.pad(xp, (0, cfg.win - xp.shape[1]))
        n_frames = 1
    # (B, 1, n_frames, win) via unfold over the time axis
    frames = xp.unsqueeze(1).unfold(-1, cfg.win, cfg.hop).squeeze(1)
    w = get_window(cfg.window, cfg.win, device=x.device, dtype=x.dtype)
    windowed = frames * w.unsqueeze(0)            # (B, n_frames, win)
    spec = torch.fft.rfft(windowed, n=cfg.n_fft, dim=-1)  # (B, n_frames, num_bins)
    spec = spec.transpose(1, 2)                   # (B, num_bins, n_frames)
    return spec, spec.abs()


def causal_istft(spec: torch.Tensor, cfg: StftConfig,
                  length: int | None = None) -> torch.Tensor:
    """Causal iSTFT — the EXACT inverse of ``causal_stft`` (same left-pad /
    left-aligned window / normalized WOLA).

    Use this — NOT ``istft`` (torch, center=True) — to invert spectra produced
    by ``causal_stft`` / ``complex_stft_truncated``.  torch.istft assumes a
    CENTERED window placement (window centered in the n_fft frame, i.e. 16 zeros
    each side for win=480/n_fft=512) and CENTER framing (reflect-pad both sides).
    ``causal_stft`` instead LEFT-pads (win-hop zeros) and LEFT-aligns the window
    (rfft n=512 on 480 windowed samples zero-pads 32 at the END).  Feeding a
    causal_stft spectrum to torch.istft therefore yields a waveform shifted by
    ~16 samples and polluted by a cross-bin linear phase ramp exp(-j2πk·16/512)
    — a live training-path bug (review finding C; the bad ``istft`` call sat in
    ``reconstruct_waveform_with_oracle_phase``, 3 call sites in train.py feeding
    MR-STFT loss + discriminator).

    Weighted overlap-add (WOLA): synthesis window = analysis window (Hann), and
    the OLA is NORMALIZED by the window-squared OLA, so reconstruction is exact
    (full-bin) wherever the window-squared sum > 0, independent of COLA.  The
    Hann-zero endpoints (samples where only one frame lands and its window is 0)
    are ill-defined; callers skip the boundary (see test_causal_roundtrip).
    """
    B, Fb, N = spec.shape
    w = get_window(cfg.window, cfg.win, device=spec.device, dtype=torch.float32)
    # irfft each frame's spectrum to n_fft time samples.  For a FULL spec this
    # recovers [win windowed samples][n_fft-win zeros] exactly; for a
    # truncated/zero-padded spec it gives the band-limited reconstruction.
    frames_full = torch.fft.irfft(spec, n=cfg.n_fft, dim=1)   # (B, n_fft, N)
    frames_full = frames_full.transpose(1, 2)                # (B, N, n_fft)
    frames_win = frames_full[..., :cfg.win] * w               # (B, N, win) synth win
    # OLA at hop into an xp-length buffer (xp = x left-padded by win-hop),
    # frame t at xp-offset t*hop — matches causal_stft's unfold framing.
    xp_len = (N - 1) * cfg.hop + cfg.win
    out = torch.zeros(B, xp_len, device=spec.device, dtype=torch.float32)
    norm = torch.zeros(B, xp_len, device=spec.device, dtype=torch.float32)
    wsq = (w * w)                                           # (win,) window-squared
    for t in range(N):
        off = t * cfg.hop
        out[:, off:off + cfg.win] += frames_win[:, t]
        norm[:, off:off + cfg.win] += wsq
    out = out / norm.clamp_min(1e-8)
    x = out[:, cfg.win - cfg.hop:]                          # strip the left-pad prefix
    if length is not None:
        x = x[..., :length]
    return x


def complex_stft_truncated(x: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """Truncated complex STFT — the model's input feature / target (spec change).

    Full causal STFT then keep bins 1..keep_bins (DC bin 0 DROPPED — see
    ``bin_to_hz`` for the +1 index convention).  Returns (B, keep_bins, N)
    complex64.  At sr=16k/n_fft=512/keep=64 this is bins 1..64 = 31.25–2000 Hz,
    the reconstruction band with no dead-imaginary DC channel.
    """
    spec, _ = causal_stft(x, cfg)            # (B, num_bins, N) complex
    return spec[:, 1:1 + cfg.keep_bins, :]  # drop DC, keep bins 1..keep_bins


def bin_to_hz(model_bin_index, sample_rate: float, n_fft: int) -> torch.Tensor:
    """Frequency of the i-th KEPT bin (DC dropped).  Single source of truth for
    the +1 offset: truncated bin i (0-based) = FFT bin i+1 = (i+1)*sr/n_fft Hz.

    Use this EVERYWHERE a kept-bin index meets a frequency — do NOT recompute
    ``i * bin_hz`` elsewhere (that silently checks the wrong bin and still
    passes, the worst failure mode).
    """
    bin_hz = sample_rate / n_fft
    if isinstance(model_bin_index, torch.Tensor):
        return (model_bin_index + 1).to(model_bin_index.dtype) * bin_hz
    return (model_bin_index + 1) * bin_hz


def hz_to_bin(freq, sample_rate: float, n_fft: int) -> torch.Tensor:
    """Inverse of ``bin_to_hz``: frequency (Hz) -> kept-bin index (float;
    caller rounds).  FFT bin = freq/bin_hz; kept index = FFT bin - 1 (DC drop)."""
    bin_hz = sample_rate / n_fft
    if isinstance(freq, torch.Tensor):
        return freq.to(freq.dtype) / bin_hz - 1.0
    return freq / bin_hz - 1.0


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
    """Process one hop of samples through a causal STFT step (streaming).

    Args:
        x_frame: (B, hop) new samples.
        cfg: StftConfig.
        prev_tail: (B, win-hop) overlap buffer from the previous frame, or None.

    Returns:
        spec: (B, keep_bins) TRUNCATED complex spectrum for this frame
            (spec change: was magnitude, now complex truncated — the streaming
            input feature).
        new_tail: (B, win-hop) overlap buffer to carry to the next frame.
    """
    B = x_frame.shape[0]
    hop, win = cfg.hop, cfg.win
    if prev_tail is None:
        prev_tail = x_frame.new_zeros(B, win - hop)
    buf = torch.cat([prev_tail, x_frame], dim=1)  # (B, win)
    w = get_window(cfg.window, cfg.win, device=x_frame.device, dtype=x_frame.dtype)
    windowed = buf * w.unsqueeze(0)
    spec = torch.fft.rfft(windowed, n=cfg.n_fft)  # (B, num_bins) complex
    return spec[:, 1:1 + cfg.keep_bins], buf[:, hop:]   # drop DC, keep 1..keep_bins


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
