"""T13-MVP CLI — run the safe fusion on real stage-2/VPU WAV files.

    python -m fusion.run_fusion --stage2 S.wav --vpu V.wav \
        --output Y.wav --diagnostics out.json [--strength 1.0] [--mode mvp]

Input contract (enforced, non-zero exit + clear message on violation):
  * both WAVs must be 16 kHz (any bit depth soundfile reads; mono preferred —
    multi-channel input is averaged to mono, reported in the diagnostics);
  * S and V must have the SAME length in samples (explicit error — no silent
    trimming/padding);
  * NO peak normalization is applied — the S/V relative level is preserved
    (peak/RMS are reported in the diagnostics so the caller can verify);
  * --strength scales the FINAL applied correction (0..1); strength=0 makes the
    output equal the stage-2 input at FULL length (no tail fade; only PCM_16
    quantization ~3e-5).
  * --mode mvp (default) | legacy_multiply (the pre-MVP four-factor product).

Exit codes: 0 success, 2 contract/usage error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import FusionConfig, Fusion

EXPECTED_SR = 16000
TAIL_FADE = 320   # win−hop samples (20 ms): causal-iSTFT edge region (ill-defined)


def _fail(msg: str) -> "None":
    print(f"run_fusion: ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _load(path: str, name: str) -> tuple[torch.Tensor, int, dict]:
    p = Path(path)
    if not p.is_file():
        _fail(f"{name} file not found: {path}")
    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if sr != EXPECTED_SR:
        _fail(f"{name} sample rate is {sr} Hz; only {EXPECTED_SR} Hz is supported "
              f"(resample upstream). File: {path}")
    channels = y.shape[1]
    if channels > 1:
        y = y.mean(axis=1)          # documented fixed rule: average to mono
    else:
        y = y[:, 0]
    meta = {"peak": float(np.abs(y).max()) if y.size else 0.0,
            "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)),
            "channels_in": int(channels),
            "channels_averaged": bool(channels > 1),
            "samples": int(y.shape[0])}
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0), sr, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fusion.run_fusion",
                                 description="T13-MVP safe fusion (stage-2 + VPU -> fused WAV)")
    ap.add_argument("--stage2", required=True, help="stage-2 (degraded mic) WAV, 16 kHz")
    ap.add_argument("--vpu", required=True, help="VPU WAV, 16 kHz, SAME length as stage2")
    ap.add_argument("--output", required=True, help="output WAV path (16 kHz)")
    ap.add_argument("--diagnostics", default=None, help="diagnostics JSON path")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="final-correction scale 0..1 (0 => output == stage2 exactly)")
    ap.add_argument("--mode", choices=["mvp", "legacy_multiply"], default="mvp",
                    help="decision combination (default mvp)")
    args = ap.parse_args(argv)

    if not (0.0 <= args.strength <= 1.0):
        _fail(f"--strength must be in [0, 1], got {args.strength}")

    s, sr_s, meta_s = _load(args.stage2, "stage2")
    v, sr_v, meta_v = _load(args.vpu, "vpu")
    if s.shape[-1] != v.shape[-1]:
        _fail(f"length mismatch: stage2 has {s.shape[-1]} samples, vpu has "
              f"{v.shape[-1]} samples — trim/align upstream (no silent trimming here)")

    cfg = FusionConfig().with_switches(decision_mode=args.mode,
                                       strength=float(args.strength))
    fusion = Fusion(cfg)
    with torch.no_grad():
        y = fusion.process_batch(s, v)

    y_np = y[0].numpy().astype(np.float64)
    # Output hygiene (strength > 0 only): the shared causal-iSTFT is ill-defined
    # over the last win−hop samples (window-squared OLA sum → 0 at the Hann
    # edge; PRE-EXISTING for any spec-editing path incl. legacy_multiply —
    # historical tests skip this boundary).  Fade those 20 ms to zero so no end
    # click/clipping is written.  strength=0 writes S verbatim INCLUDING the
    # tail (no fade) so the A/B contract "output == stage2" holds at full
    # length up to PCM_16 quantization (~3e-5).
    if args.strength > 0.0 and y_np.shape[0] > TAIL_FADE:
        y_np[-TAIL_FADE:] *= np.linspace(1.0, 0.0, TAIL_FADE)
    clipped = int(np.sum(np.abs(y_np) > 1.0))
    out_peak = float(np.abs(y_np).max()) if y_np.size else 0.0
    sf.write(args.output, np.clip(y_np, -1.0, 1.0), EXPECTED_SR, subtype="PCM_16")

    diag = {
        "commit": _git_commit(),
        "mode": args.mode,
        "strength": float(args.strength),
        "sr": EXPECTED_SR,
        "inputs": {"stage2": meta_s, "vpu": meta_v},
        "output": {"path": args.output, "peak": out_peak,
                   "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(y_np ** 2)) + 1e-12)),
                   "samples": int(y_np.shape[0]), "clipped_samples": clipped,
                   "tail_fade_samples": TAIL_FADE},
    }
    if fusion.last_diagnostics is not None:
        diag.update(fusion.last_diagnostics)
    if args.diagnostics:
        Path(args.diagnostics).write_text(json.dumps(diag, indent=2))

    print(f"run_fusion: wrote {args.output}  mode={args.mode} strength={args.strength}")
    print(f"  output peak={out_peak:.4f}  coverage_100_800="
          f"{diag.get('coverage_100_800', float('nan')):.3f}  "
          f"veto_fraction={diag.get('veto_fraction_100_800', float('nan')):.3f}")
    if clipped:
        print(f"  WARNING: {clipped} output samples exceeded ±1.0 and were clipped")
    if args.diagnostics:
        print(f"  diagnostics: {args.diagnostics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
