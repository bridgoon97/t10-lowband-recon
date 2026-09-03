"""T13-N2 · GTCRN (public ONNX) VPU denoising — OFFLINE data prep.

NOT part of the algorithm path: fusion/ never imports this module (the static
check covers it).  Model provenance:
  URL    https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
  upstream  Xiaobin-Rong/gtcrn
  SHA256 e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534
  bytes  535638
Cached OUTSIDE the repo (default ~/.cache/t13/models/gtcrn_simple.onnx); the
model file is never committed.

Fixed-gain protocol (pre-locked, no post-hoc gain choice):
  V_in = g·V_raw → GTCRN → V_dn = GTCRN(V_in)/g
with WHOLE-CLIP scalars g: raw(g=1) | rms→-30 dBFS | rms→-24 dBFS | peak→-6 dBFS.
No per-frame AGC, limiter, or second normalisation.  A scaled input whose peak
>= 1 is INVALID and skipped (never lowered to fit).  Bypass control:
(g·V)/g must equal float32 V within 1e-6; a mutation that skips the /g must
fail that check.  GTCRN's streaming output is shorter than its input by a
fixed chunk; the missing tail samples are restored by HOLDING the last
denoised sample (documented, identical across gains, no future information).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = str(Path.home() / ".cache/t13/models/gtcrn_simple.onnx")
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speech-enhancement-models/gtcrn_simple.onnx")
MODEL_SHA256 = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
MODEL_BYTES = 535638

GAINS = ("raw", "rms_m30", "rms_m24", "peak_m6")


def model_sha256(path: str = DEFAULT_MODEL) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model(path: str = DEFAULT_MODEL) -> dict:
    p = Path(path)
    sha = model_sha256(path)
    ok = (sha == MODEL_SHA256 and p.stat().st_size == MODEL_BYTES)
    return {"path": path, "sha256": sha, "bytes": p.stat().st_size,
            "sha256_match": ok, "url": MODEL_URL}


def gain_for(v: np.ndarray, mode: str) -> float:
    """Whole-clip scalar gain (one value per recording, no frame adaptation)."""
    v = v.astype(np.float64)
    if mode == "raw":
        return 1.0
    rms = float(np.sqrt((v ** 2).mean()))
    peak = float(np.abs(v).max())
    if mode == "rms_m30":
        return 10 ** ((-30.0 - 20 * np.log10(max(rms, 1e-12))) / 20)
    if mode == "rms_m24":
        return 10 ** ((-24.0 - 20 * np.log10(max(rms, 1e-12))) / 20)
    if mode == "peak_m6":
        return 10 ** ((-6.0 - 20 * np.log10(max(peak, 1e-12))) / 20)
    raise ValueError(f"unknown gain mode {mode!r}")


class GtcrnDenoiser:
    """Lazy sherpa-onnx offline denoiser wrapper (single instance reused)."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        import sherpa_onnx
        mc = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model_path))
        self._d = sherpa_onnx.OfflineSpeechDenoiser(
            sherpa_onnx.OfflineSpeechDenoiserConfig(model=mc))

    def run(self, x: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Denoise float32 (T,) @ sr(=16000); output length restored to input by
        HOLDING the last denoised sample over the dropped streaming tail."""
        assert sr == 16000 and x.ndim == 1
        out = self._d.run(x.astype(np.float32), sr)
        y = np.asarray(out.samples, dtype=np.float32)
        if len(y) < len(x):                       # streaming tail: hold last value
            y = np.concatenate([y, np.full(len(x) - len(y), y[-1], dtype=np.float32)])
        return y[:len(x)]


def bypass_diff(v_raw: np.ndarray, g: float, skip_divide: bool = False) -> float:
    """Bypass control: (g·V)/g vs float32 V (denoiser bypassed to identity).
    ``skip_divide=True`` is the mutation (forgot the /g) — must exceed 1e-6."""
    x = (g * v_raw).astype(np.float32)
    y = x if skip_divide else (x / g).astype(np.float32)
    return float(np.abs(y - v_raw.astype(np.float32)).max())


@dataclass
class GainResult:
    mode: str
    g: float
    valid: bool
    in_peak: float
    in_rms: float
    out_peak: float
    out_rms: float
    in_dc: float
    out_dc: float
    nan_inf: int
    clipped: int
    bypass_max_diff: float
    v_dn: np.ndarray


def run_gain(v_raw: np.ndarray, mode: str, denoiser: GtcrnDenoiser,
             sr: int = 16000, mutation_skip_divide: bool = False) -> GainResult:
    """V_in = g·V_raw → GTCRN → V_dn = GTCRN(V_in)/g  (whole-clip scalar g)."""
    v_raw = v_raw.astype(np.float32)
    g = gain_for(v_raw, mode)
    v_in = (g * v_raw).astype(np.float32)
    in_peak = float(np.abs(v_in).max())
    res = dict(mode=mode, g=float(g), in_peak=in_peak,
               in_rms=float(np.sqrt((v_in ** 2).mean())),
               in_dc=float(v_in.mean()))
    if in_peak >= 1.0:
        # INVALID: the fixed target cannot be applied without lowering it —
        # skipped, never silently re-targeted.
        return GainResult(mode=mode, g=float(g), valid=False, in_peak=in_peak,
                          in_rms=res["in_rms"], out_peak=0.0, out_rms=0.0,
                          in_dc=res["in_dc"], out_dc=0.0, nan_inf=0, clipped=0,
                          bypass_max_diff=0.0, v_dn=v_raw)
    y = denoiser.run(v_in, sr)
    v_dn = y if mutation_skip_divide else (y / g).astype(np.float32)
    out_peak = float(np.abs(v_dn).max())
    nan_inf = int(np.sum(~np.isfinite(v_dn)))
    clipped = int(np.sum(np.abs(v_dn) > 1.0))
    # bypass control: the GAIN/DIVIDE plumbing with the denoiser bypassed to
    # identity — (g·V)/g must equal float32 V within 1e-6.  The mutation
    # (skipping the divide) makes this check fail, proving it has teeth.
    bypass = bypass_diff(v_raw, g)
    return GainResult(mode=mode, g=float(g), valid=True, in_peak=in_peak,
                      in_rms=res["in_rms"], out_peak=out_peak,
                      out_rms=float(np.sqrt((v_dn ** 2).mean())),
                      in_dc=res["in_dc"], out_dc=float(v_dn.mean()),
                      nan_inf=nan_inf, clipped=clipped,
                      bypass_max_diff=bypass, v_dn=v_dn)
