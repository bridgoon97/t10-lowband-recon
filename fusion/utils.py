"""Causal DSP helpers shared by the fusion layers.  All one-sided / EMA —
no look-ahead, no whole-segment statistics (T13 §4 hard constraint)."""
from __future__ import annotations

import torch


def alpha_from_tau(tau_s: float, hop: int, sr: int) -> float:
    """EMA weight for a time-constant ``tau_s`` at hop/sr (causal, one-sided)."""
    return 1.0 - float(torch.exp(torch.tensor(-hop / (sr * max(tau_s, 1e-9)))))


def causal_ema(prev: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """One-sided EMA: (1-α)·prev + α·x.  Causal by construction."""
    return (1.0 - alpha) * prev + alpha * x


def asym_ema(prev: torch.Tensor, x: torch.Tensor, alpha_rise: float,
             alpha_fall: float) -> torch.Tensor:
    """Non-symmetric one-sided EMA — slow rise, fast fall.

    If the new value x > prev (rising ⇒ "use more V"), use ``alpha_rise``
    (small ⇒ slow).  If x < prev (falling ⇒ "back to S"), use ``alpha_fall``
    (large ⇒ fast).  Two SEPARATE time constants in code (T5 lesson: a
    supposed "asym" mask turned out symmetric — this is the real one, and
    test_mechanisms M4 proves the asymmetry with a mutation sanity)."""
    rising = x > prev
    a = torch.where(rising, torch.full_like(x, alpha_rise),
                    torch.full_like(x, alpha_fall))
    return (1.0 - a) * prev + a * x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def soft_gate(conf: torch.Tensor, gamma: float = 1.0,
              floor: float = 0.0) -> torch.Tensor:
    """Soft (no-threshold) gate: g = floor + (1-floor)·conf^gamma, conf∈[0,1].
    High conf ⟹ high g.  Direction: ``f0_confidence = 1 − CMND`` (high ⟹ voiced)."""
    c = conf.clamp(0.0, 1.0) ** gamma
    return floor + (1.0 - floor) * c


def smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric 1-D moving average along the LAST dim (reflect-padded).

    Used for freq-axis smoothing of C[f] WITHIN one frame — this touches
    neighbouring FREQUENCY bins of the same frame, never future FRAMES, so it
    introduces no time-domain look-ahead (the §4 hard constraint is about TIME).
    """
    if k <= 1:
        return x
    pad = k // 2
    xr = x.reshape(-1, x.shape[-1])
    xp = torch.nn.functional.pad(xr, (pad, pad), mode="reflect")
    w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
    y = torch.nn.functional.conv1d(xp.unsqueeze(1), w, padding="valid").squeeze(1)
    return y.reshape(x.shape)


class CohTracker:
    """Per-bin online MSC (magnitude-square coherence) between two complex
    spectra, tracked causally via EMA.  MSC(f) = |EMA[v·s*]|² / (EMA[|v|²]·EMA[|s|²]).
    """

    def __init__(self, num_bins: int, alpha: float, device, dtype=torch.float32):
        self.alpha = alpha
        self.vs = torch.zeros(num_bins, device=device, dtype=torch.complex64)
        self.vv = torch.zeros(num_bins, device=device, dtype=dtype)
        self.ss = torch.zeros(num_bins, device=device, dtype=dtype)

    def update(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """v, s: (num_bins,) complex.  Returns MSC (num_bins,) real [0,1]."""
        vs = v * torch.conj(s)
        self.vs = (1 - self.alpha) * self.vs + self.alpha * vs
        self.vv = (1 - self.alpha) * self.vv + self.alpha * (v.abs() ** 2)
        self.ss = (1 - self.alpha) * self.ss + self.alpha * (s.abs() ** 2)
        # clamp must be TINY (1e-20): a 1e-10 floor inflates the denominator for
        # quiet bins (vv·ss < floor) ⇒ MSC DEFLATES ⇒ breaks scale-invariance
        # (FR1-a: joint S+V scaling must leave c_V invariant).  1e-20 only
        # guards the literal 0/0 of true silence; normal signals never hit it.
        return (self.vs.abs() ** 2) / (self.vv * self.ss).clamp_min(1e-20)


def local_snr_db(s_mag: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Per-bin local SNR (dB) vs a tracked noise floor (magnitude)."""
    return 20.0 * torch.log10(s_mag.clamp_min(1e-8) / floor.clamp_min(1e-8))
