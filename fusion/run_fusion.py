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
  * --strength scales the FINAL applied correction (0..1); with strength=0 no
    extra 20 ms tail fade is applied, and every sample except the FINAL one
    (structurally unencodable by the causal Hann framing; reported separately)
    matches the stage-2 input within 1e-4.
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
from .trust import TrustSource

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
    ap.add_argument("--mode", choices=["n1", "mvp", "legacy_multiply"], default="n1",
                    help="decision structure (default n1 = trust-routed add/subtract)")
    ap.add_argument("--trust", default="1.0",
                    help="VPU trust p[t]: a number (MANUAL const, default 1.0), "
                         "a .json {\"p\": [...]} or a 16 kHz .wav (sampled at each "
                         "causal frame anchor). The literal 'oracle' is rejected.")
    args = ap.parse_args(argv)

    if not (0.0 <= args.strength <= 1.0):
        _fail(f"--strength must be in [0, 1], got {args.strength}")
    if args.trust.strip().lower() == "oracle":
        _fail("--trust 'oracle' is forbidden in the production path "
              "(ORACLE trust reads ground-truth wear state)")

    s, sr_s, meta_s = _load(args.stage2, "stage2")
    v, sr_v, meta_v = _load(args.vpu, "vpu")
    if s.shape[-1] != v.shape[-1]:
        _fail(f"length mismatch: stage2 has {s.shape[-1]} samples, vpu has "
              f"{v.shape[-1]} samples — trim/align upstream (no silent trimming here)")

    cfg = FusionConfig().with_switches(decision_mode=args.mode,
                                       strength=float(args.strength))
    fusion = Fusion(cfg)
    trust_src = "manual"
    trust_seq = None
    if args.mode == "n1":
        from .stft import StftStreamer
        n_frames = StftStreamer.n_frames_for(s.shape[-1], cfg)
        t = args.trust.strip()
        try:
            const = float(t)
            ts = TrustSource(source="manual", const=const)
        except ValueError:
            if not t.endswith((".json", ".wav")):
                _fail(f"--trust must be a number, a .json or a .wav, got {t!r}")
            cfg = cfg.with_switches(trust_source="external", trust_path=t)
            try:
                ts = TrustSource.from_config(cfg, n_frames, cfg.sr, cfg.hop)
            except ValueError as e:
                _fail(f"--trust {t!r}: {e}")
            trust_src = ts.source
            trust_seq = [float(x) for x in ts.values]
        fusion.set_trust(ts)
    with torch.no_grad():
        y = fusion.process_batch(s, v)

    y_np = y[0].numpy().astype(np.float64)
    # Output hygiene (strength > 0 only): the shared causal-iSTFT is ill-defined
    # over the last win−hop samples (window-squared OLA sum → 0 at the Hann
    # edge; PRE-EXISTING for any spec-editing path incl. legacy_multiply —
    # historical tests skip this boundary).  Fade those 20 ms to zero so no end
    # click/clipping is written.  strength=0 applies NO fade; every sample
    # except the structurally unencodable final one (causal Hann endpoint,
    # never encoded — pre-existing, identical in legacy mode) matches S within
    # 1e-4, and the final sample is reported separately by the M1 test.
    fade_n = TAIL_FADE if (args.strength > 0.0 and y_np.size > TAIL_FADE) else 0
    if fade_n:
        y_np[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    clipped = int(np.sum(np.abs(y_np) > 1.0))
    out_peak = float(np.abs(y_np).max()) if y_np.size else 0.0
    sf.write(args.output, np.clip(y_np, -1.0, 1.0), EXPECTED_SR, subtype="PCM_16")

    diag = {
        "commit": _git_commit(),
        "mode": args.mode,
        "strength": float(args.strength),
        "trust": {"source": trust_src,
                  "sequence": trust_seq,
                  "n": (len(trust_seq) if trust_seq else None)},
        "sr": EXPECTED_SR,
        "inputs": {"stage2": meta_s, "vpu": meta_v},
        "output": {"path": args.output, "peak": out_peak,
                   "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(y_np ** 2)) + 1e-12)),
                   "samples": int(y_np.shape[0]), "clipped_samples": clipped,
                   "tail_fade_samples": fade_n},
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
