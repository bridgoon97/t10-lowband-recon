"""LowBandReconstructor — unified interface for three arms (§3.1).

Quick start:
    from lowband import build_model
    model = build_model(cfg)
    out = model(x)  # {"mag": (B,F,N), "wav": ..., "aux": ...}
"""
from .interface import LowBandReconstructor
from .models import build_model, ARMS
from .data import build_dataset, ADAPTERS
from .dsp import StftConfig
from . import losses, dsp, data, models, utils

__all__ = [
    "LowBandReconstructor", "build_model", "ARMS",
    "build_dataset", "ADAPTERS", "StftConfig",
    "losses", "dsp", "data", "models", "utils",
]
