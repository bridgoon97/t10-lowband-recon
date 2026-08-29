"""T13 fusion module — training-free, streaming-causal, pure DSP.

Public surface:
  FusionConfig        — single source of all placeholder constants + switches
  Fusion              — batch (whole-signal) pipeline
  FusionStreamer      — per-hop streaming pipeline
  degrade / DegradationConfig — stage-2 damage simulation (D1–D4)
  signals             — synthetic builders for the M1–M7 mechanism tests
  stft / f0 / align / decision / synthesis / utils — layer modules
"""
from .config import FusionConfig
from .stft import StftStreamer, IstftStreamer, stft_batch, istft_batch
from .f0 import F0Estimator, f0_batch
from .align import DelayComp, EQAlign, measure_gcc_phat
from .decision import CV, GF0, WBand, WLocal, AsymSmoother
from .synthesis import Synthesis, logclip_mix, complex_convex, ComfortNoise
from .fusion import Fusion, FusionStreamer, FusionCore
from .degrade import degrade, DegradationConfig, apply_d1
from . import signals

__all__ = [
    "FusionConfig", "Fusion", "FusionStreamer", "FusionCore",
    "StftStreamer", "IstftStreamer", "stft_batch", "istft_batch",
    "F0Estimator", "f0_batch", "DelayComp", "EQAlign", "measure_gcc_phat",
    "CV", "GF0", "WBand", "WLocal", "AsymSmoother",
    "Synthesis", "logclip_mix", "complex_convex", "ComfortNoise",
    "degrade", "DegradationConfig", "apply_d1", "signals",
]
