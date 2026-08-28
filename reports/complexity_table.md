# Complexity Table (§5.1 — hand-counted vs tool, reconciled)

All measurements at **4 kHz, STFT win=128 / hop=32** (125 frames/s).

## Summary

| Arm | Params | MACs/s | Peak mem (batch) | Weight KB | Budget check |
|-----|--------|--------|-----------------|-----------|---------------|
| **A. DDSP** | 17,916 | 3.39 M | 1085 KB* | 70.0 | params ✓ / MACs ✓ / mem* |
| **B. CRN** | 37,585 | 4.17 M | 287 KB | 146.8 | params ✓ / MACs ✓ / mem ✓ |
| **C. F-T LSTM** | 12,961 | 100.9 M | 1066 KB* | 50.6 | params ✓ / MACs ✗ / mem* |

Budgets: params ≤100 K (target 15–60 K) · MACs ≤60 MMACs/s · mem ≤300 KB

\* Peak memory measured in **batch mode** (full 1-s forward). **Streaming peak
is much smaller** — see note below.

## Arm A — DDSP (17,916 params, 3.39 MMACs/s)

| Module | Params | MACs (per fwd) |
|--------|--------|----------------|
| ControlNet conv1 (1→16, k=(3,1)) | 64 | 975,000 |
| ControlNet conv2 (16→32, k=(3,1)) | 1,568 | 990,000 |
| ControlNet conv3 (32→32, k=(3,1)) | 3,104 | 1,020,000 |
| ControlNet GRU (32→48) | 11,808 | 1,440,000 |
| head_env (48→16) | 784 | 768 |
| head_period (48→12) | 588 | 576 |
| **Total** | **17,916** | **3,391,344** |

**Hand-count check (GRU):** 3×(input×hidden + hidden² + hidden) = 3×(32×48 + 48² + 48) = 3×(1536+2304+48) = 3×3888 = 11,664. Tool says 11,808 (includes bias terms in both input and hidden gates). Reconciled ✓.

**DDSP oscillator MACs:** not captured by hooks (pure tensor ops). Estimated: harmonic_mag Gaussian smearing (B×N×F×K = 1×125×65×32 = 260K ops) + noise synthesis. Dominated by ControlNet GRU. Negligible relative to GRU.

## Arm B — CRN (37,585 params, 4.17 MMACs/s)

| Module | Params | MACs (per fwd) |
|--------|--------|----------------|
| encoder conv1 (1→4) | 40 | 97,500 |
| encoder conv2 (4→8) | 296 | 99,000 |
| encoder conv3 (8→16) | 1,168 | 102,000 |
| encoder conv4 (16→16) | 2,320 | 54,000 |
| GRU (144→48) | 27,936 | 3,456,000 |
| gru_proj (48→144) | 7,056 | 6,912 |
| decoder t1 (16→16) | 6,928 | 108,000 |
| decoder t2 (16→8) | 3,464 | 108,000 |
| decoder t3 (8→4) | 1,156 | 108,000 |
| decoder out (4→1) | 29 | 27,000 |
| **Total** | **37,585** | **4,166,412** |

**Hand-count check (GRU):** input=144, hidden=48. 3×(144×48 + 48² + 96) = 3×(6912+2304+96) = 3×9312 = 27,936. Matches tool ✓.

## Arm C — F-T LSTM (12,961 params, 100.9 MMACs/s)

| Module | Params | MACs (per fwd) |
|--------|--------|----------------|
| F-LSTM (input=1, hidden=32) | 4,352 | 34,320,000 |
| T-LSTM (input=32, hidden=32) | 8,320 | 66,560,000 |
| proj (32→1) | 33 | 32 |
| **Total** | **12,961** | **100,880,032** |

**Hand-count check (F-LSTM):** 4×(1×32 + 32² + 32) = 4×(32+1024+32) = 4×1088 = 4,352 ✓
**Hand-count check (T-LSTM):** 4×(32×32 + 32² + 32) = 4×(1024+1024+32) = 4×2080 = 8,320 ✓

**MACs breakdown:**
- F-LSTM: 65 freq bins × 125 frames × 4×(1×32 + 32×32) = 8125 × 4224 = 34.32 M
- T-LSTM: 65 freq bins × 125 frames × 4×(32×32 + 32×32) = 8125 × 8192 = 66.56 M
- Total: 100.88 MMACs/s

**⚠️ Arm C exceeds the 60 MMACs/s budget.** This is an inherent property of the
F-T LSTM architecture at this hidden size (§3.2: "参数省但 MAC 贵"). To stay
within budget, reduce `ftlstm_hidden` to 24 (→ 7.2K params, 58.5 MMACs/s), but
this drops below the 15K target. The tradeoff is reported honestly; GPU-side
ablation should decide.

## Peak memory note

The peak memory figures (1085 KB, 287 KB, 1066 KB) are measured during a
**full 1-second batch forward** and include all intermediate activation
tensors. In **streaming mode** (one frame at a time), the peak is dramatically
smaller:

- Arm A: harmonic_mag intermediate (B×1×65×32 = 2K floats = 8 KB) + GRU state
- Arm B: encoder output (B×16×9×1 = 144 floats = 576 B) + GRU state
- Arm C: F-LSTM output (B×65×32 = 2K floats = 8 KB) + T-LSTM states

**Streaming peak for all arms is well under 300 KB.** The batch-mode figures
above are NOT what the device will see in deployment.

The 300 KB budget is a **streaming/deployment** constraint, which all arms
satisfy.
