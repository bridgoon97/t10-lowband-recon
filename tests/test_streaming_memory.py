"""§5.1 (streaming) — MEASURED streaming peak memory via stream_step.

rework item ③: the batch-mode figures (1085 / 287 / 1066 KB) were measured, but
the streaming peak was only ARGUED in prose ("well under 300 KB") — i.e. the
judgement was changed after measuring only the favourable (batch) number.  This
test actually runs ``stream_step`` frame-by-frame (125 hops = 1 s @4 kHz) and
measures the streaming peak the SAME way batch is measured, then reports BOTH.

Methodology note: hooks see module OUTPUTS (conv/GRU/LSTM activations — the
dominant cost).  They do NOT see transient tensor ops (e.g. the DDSP harmonic
Gaussian), but that is true of BOTH the batch and streaming measurements here,
so the two numbers are directly comparable.  F0 source does not affect the
conv/GRU activation memory (F0 is consumed after the control net), so streaming
(oracle F0) is comparable to batch (default estimated F0).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lowband import build_model
from lowband.utils.complexity import (measure_complexity,
                                       measure_streaming_complexity)

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64}
STREAM_BUDGET_KB = 300.0


def test_streaming_peak_memory_measured():
    """Measure both batch & streaming peak; streaming must be ≤300 KB (measured)."""
    print(f"\n  budget: streaming peak ≤ {STREAM_BUDGET_KB:.0f} KB")
    print(f"  {'arm':<14}{'batch KB':>10}{'stream KB':>11}"
          f"{'weight KB':>11}{'stream/batch':>13}{'≤300?':>7}")
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm)
        torch.manual_seed(42)
        model = build_model(cfg)
        b = measure_complexity(model, cfg["sample_rate"], hop=cfg["stft_hop"],
                               n_bins=64)
        # fresh model for streaming (hooks/eval shouldn't leak batch state)
        torch.manual_seed(42)
        model2 = build_model(cfg)
        oracle = 150.0 if arm == "arm_a_ddsp" else None
        s = measure_streaming_complexity(model2, cfg["sample_rate"],
                                         hop=cfg["stft_hop"],
                                         oracle_f0_hz=oracle)
        batch_kb = b["peak_total_kb"]
        stream_kb = s["peak_total_kb"]
        weight_kb = s["weight_kb"]
        ratio = stream_kb / (batch_kb + 1e-9)
        ok = "✓" if stream_kb <= STREAM_BUDGET_KB else "✗"
        print(f"  {arm:<14}{batch_kb:>10.1f}{stream_kb:>11.1f}"
              f"{weight_kb:>11.1f}{ratio:>13.2f}{ok:>7}")
        # streaming must be MEASURED under the deployment budget
        assert stream_kb <= STREAM_BUDGET_KB, (
            f"{arm} streaming peak {stream_kb:.1f} KB > {STREAM_BUDGET_KB:.0f} KB "
            f"budget (batch={batch_kb:.1f} KB)")
        # sanity: streaming MUST be (much) smaller than batch — else streaming
        # path isn't actually frame-by-frame
        assert stream_kb < batch_kb, (
            f"{arm} streaming peak {stream_kb:.1f} KB >= batch {batch_kb:.1f} KB — "
            f"streaming is not reducing peak; check stream_step")
    print("\n  all arms: streaming peak MEASURED ≤ 300 KB ✓")


if __name__ == "__main__":
    test_streaming_peak_memory_measured()
    print("streaming memory test: PASS")
