"""T13-N1 · trust interface — frame-level VPU credibility ``p[t] ∈ [0,1]``.

This is an EXTERNAL INTERFACE, not an estimator: this batch never computes
trust (no c_V, no online MSC, no changepoint watchdog — all left for later
units).  Allowed sources, tagged (F0-contract discipline):

  MANUAL             constant p (this batch's scan variable; CLI default 1.0)
  EXTERNAL           per-frame values from a json {\"p\": [...]} or a 16 kHz wav
                     (sampled at each causal frame anchor t·hop — left-aligned,
                     zero extra delay, same frame indexing as the F0 buffer)
  INTERNAL_FALLBACK  reserved for a future internal estimator (not implemented)
  ORACLE             FORBIDDEN in the production path — constructing it raises
                     unless the mutation-only ``allow_oracle`` flag is set.

Semantics are monotone: 1 = worn well, V′ usable; 0 = off/too loose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import FusionConfig

VALID_SOURCES = ("manual", "external", "internal_fallback", "oracle")


@dataclass
class TrustSource:
    """Per-frame trust sequence ``p[t]`` with a mandatory source tag."""
    source: str = "manual"
    const: float = 1.0
    values: Optional[torch.Tensor] = None     # (N,) float, EXTERNAL only
    allow_oracle: bool = False                # MUTATION ONLY

    def __post_init__(self):
        s = self.source.lower()
        if s not in VALID_SOURCES:
            raise ValueError(f"unknown trust source: {self.source!r}")
        if s == "oracle" and not self.allow_oracle:
            raise ValueError(
                "ORACLE trust is forbidden in the production path "
                "(it would read ground-truth wear state); use MANUAL/EXTERNAL.")
        self.source = s
        if self.values is not None:
            self.values = self.values.float().clamp(0.0, 1.0)

    @classmethod
    def from_config(cls, cfg: FusionConfig, n_frames: int,
                    sr: int, hop: int) -> "TrustSource":
        """Build from config (CLI path).  EXTERNAL wav is read at each frame
        anchor t·hop (left-aligned — frame t's causal buffer ENDS at t·hop, so
        the anchor uses only past samples: zero extra delay)."""
        if cfg.trust_source == "manual":
            return cls(source="manual", const=float(cfg.trust_const))
        if cfg.trust_source == "internal_fallback":
            # reserved for the future internal estimator — this batch: constant
            return cls(source="internal_fallback", const=float(cfg.trust_const))
        if cfg.trust_source == "external":
            import soundfile as sf
            if cfg.trust_path is None:
                raise ValueError("trust_source=external needs trust_path")
            if str(cfg.trust_path).endswith(".json"):
                import json
                obj = json.loads(open(cfg.trust_path).read())
                vals = torch.tensor([float(v) for v in obj["p"]])
            elif str(cfg.trust_path).endswith(".wav"):
                y, wav_sr = sf.read(cfg.trust_path, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if wav_sr != sr:
                    raise ValueError(f"trust wav sr {wav_sr} != {sr}")
                idx = (torch.arange(n_frames) * hop).clamp_max(len(y) - 1)
                vals = torch.from_numpy(y)[idx]
            else:
                raise ValueError(f"unsupported trust path: {cfg.trust_path}")
            if vals.numel() < n_frames:
                raise ValueError(
                    f"trust sequence too short: {vals.numel()} < {n_frames} frames "
                    f"(short sequences are rejected, not tail-padded)")
            return cls(source="external", values=vals[:n_frames])
        raise ValueError(f"cannot build trust source {cfg.trust_source!r}")

    def frame(self, t: int) -> float:
        """p at causal frame index t (same indexing as the F0 buffer).

        Tail semantics (N1 rework): the batch path extends the tail by
        (win−hop) zeros, producing two frames beyond the original framing; t
        beyond the provided sequence HOLDS THE LAST PROVIDED VALUE — a causal
        hold that reads no future and no oracle.  Sequences shorter than the
        ORIGINAL frame count are still rejected at build time (from_config),
        so the hold only ever covers the tail-extension frames."""
        if self.source == "external":
            i = min(t, int(self.values.numel()) - 1)
            return float(self.values[i])
        return float(self.const)
