# Complexity Table (§5.1 — 16 kHz / complex / n_fft=512 / win=480 / hop=160 / keep 64, DC dropped)

> 64 kept bins = FFT bins 1..64 = 31.25–2000 Hz (DC bin 0 DROPPED — it has a
> dead imaginary channel for a real signal; 64 is conv-friendly 64→32→16→8).
> bin↔Hz uses `bin_to_hz(i)=(i+1)*sr/n_fft` (single source of the +1 offset).
> Arms output a TRUNCATED COMPLEX spectrum (real/imag through conv/GRU/LSTM).

## Summary

| Arm | Params | MACs/s | Peak mem (batch) | Peak mem (stream) | Weight KB | Budget check |
|-----|--------|--------|-----------------|-------------------|-----------|---------------|
| **A. DDSP** | 17,916 | 2.69 M | 870.0 KB | **78.0 KB** | 70.0 | params ✓ / MACs ✓ / mem ✓(stream) |
| **B. CRN** | 34,522 | 3.08 M | 234.9 KB | **135.9 KB** | 134.9 | params ✓ / MACs ✓ / mem ✓(stream) |
| **C. F-T LSTM** | 13,122 | 80.3 M | 851.3 KB | **59.3 KB** | 51.3 | params ✓ / MACs ✗ / mem ✓(stream) |

Budgets: params ≤100 K (target 15–60 K) · MACs ≤60 MMACs/s · mem ≤300 KB

\* Batch peak is during a full 1-second forward.  The 300 KB budget is a
**streaming/deployment** constraint → the relevant column is **Peak mem (stream)**,
MEASURED via `stream_step`.

## Arm A — DDSP (17,916 params, 2.69 MMACs/s)

waveform_synth ON (oscillator → waveform → STFT → truncate). DDSP oscillator/noise
ops are pure tensor ops (not hooked); MACs dominated by the ControlNet GRU.

| Module | Params | MACs/s |
|--------|--------|--------|
| ControlNet conv1 (2→16, k=(3,1)) | 64 | 307,200 |
| ControlNet conv2 (16→32, k=(3,1)) | 1,568 | 614,400 |
| ControlNet conv3 (32→32, k=(3,1)) | 3,104 | 614,400 |
| ControlNet GRU (32→48) | 11,808 | 1,152,000 |
| head_env (48→16) | 784 | 768 |
| head_period (48→12) | 588 | 576 |
| **Total** | **17,916** | **2,689,344** |

## Arm B — CRN (34,522 params, 3.08 MMACs/s)

64 freq bins → clean 64→32→16→8 downsampling; feat_dim = 16×8 = 128 (was 144 at
65 bins — dropping DC shrinks the GRU, the "64 conv-friendly" benefit).

| Module | Params | MACs/s |
|--------|--------|--------|
| encoder conv1 (2→4) | 40 | 76,800 |
| encoder conv2 (4→8) | 296 | 76,800 |
| encoder conv3 (8→16) | 1,168 | 76,800 |
| encoder conv4 (16→16) | 2,320 | 38,400 |
| GRU (128→48) | 24,576 | 2,534,400 |
| gru_proj (48→128) | 6,192 | 6,144 |
| decoder t1 (16→16) | 6,928 | 76,800 |
| decoder t2 (16→8) | 3,464 | 76,800 |
| decoder t3 (8→4) | 1,156 | 76,800 |
| decoder out (4→2) | 10 | 38,400 |
| **Total** | **34,522** | **3,078,144** |

## Arm C — F-T LSTM (13,122 params, 80.3 MMACs/s)

| Module | Params | MACs/s |
|--------|--------|--------|
| F-LSTM (input=2 real/imag, hidden=32) | 4,480 | 27,852,800 |
| T-LSTM (input=32, hidden=32) | 8,320 | 52,428,800 |
| proj (32→2 real/imag) | 66 | 64 |
| **Total** | **13,122** | **80,281,664** |

**⚠️ Arm C exceeds the 60 MMACs/s budget** — inherent to F-T LSTM at hidden=32
(§3.2 "参数省但 MAC 贵"); hidden=24 → ~7.2 K params / ~47 MMACs/s (within budget)
but below the 15 K target. Reported honestly; GPU ablation decides.

## Peak memory note — MEASURED (rework ③)

Both batch and streaming peaks measured with the SAME hook methodology
(module-output activations + weights); streaming runs `stream_step` over 100 hops
(1 s @16 kHz) one frame at a time.

| Arm | Batch peak | Streaming peak | Weight | Stream activation | Stream/batch |
|-----|-----------|---------------|--------|-------------------|--------------|
| A. DDSP | 870.0 KB | **78.0 KB** | 70.0 KB | ~8.0 KB | 0.09 |
| B. CRN | 234.9 KB | **135.9 KB** | 134.9 KB | ~1.0 KB | 0.58 |
| C. F-T LSTM | 851.3 KB | **59.3 KB** | 51.3 KB | ~8.0 KB | 0.07 |

All three streaming peaks ≤ 300 KB **by measurement**. Arm B's streaming peak is
dominated by its own weight footprint (134.9 KB); per-frame activation ~1 KB.
Reproduce: `python3 tests/test_streaming_memory.py`.

## Change log

- **DC drop (this round):** 65→64 kept bins (bins 1..64, 31.25–2000 Hz). Arm B
  shrunk 37,610→34,522 params (GRU feat_dim 144→128). `bin_to_hz`/`hz_to_bin`
  added as the single source of the +1 index offset.
- **Spec change (prev round):** 4 kHz/magnitude → 16 kHz/complex. MACs/s dropped
  ~20 % (frame rate 125→100 fps); params barely changed; streaming peak ~unchanged.
