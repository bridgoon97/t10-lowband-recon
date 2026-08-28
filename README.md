# T10 — Low-Frequency Reconstruction: Code-Through & GPU Handoff

> **Role:** This repo was built in a **no-GPU isolation environment**.
> The deliverable is **code that runs and is ready for GPU training**, not
> experimental conclusions. Success = the recipient changes one config to
> point at real data, runs one command, and starts training without reading
> the model code.

## Quick start (GPU)

```bash
# 1. Install
pip install torch numpy scipy pyyaml soundfile tqdm onnx onnxruntime

# 2. Point config at your data (edit the data: section)
#    See docs/data_adapter_guide.md for how to write your adapter

# 3. Train any arm
python train.py --config configs/arm_a_ddsp.yaml   # DDSP
python train.py --config configs/arm_b_crn.yaml     # CRN
python train.py --config configs/arm_c_ftlstm.yaml   # F-T LSTM

# 4. Export
python export.py --config configs/arm_b_crn.yaml --output exports/

# 5. Run all CPU verifications
python verify.py
```

## The three arms (§3.2)

| Arm | Architecture | Params | MACs/s | Key trait |
|-----|-------------|--------|--------|-----------|
| A | DDSP harmonic+noise (oscillator synthesis) | 17.9 K | 3.4 M | Hardest to get right; F0-critical |
| B | CRN (grouped conv + GRU spectral regression) | 37.6 K | 4.2 M | Solid baseline |
| C | F-T LSTM (freq→time LSTM, FT-JNF XS style) | 13.0 K | 100.9 M | Param-efficient, MAC-expensive |

All three share:
- One interface (`LowBandReconstructor`) — switch in config
- One `train.py` — no arm-specific training code
- One data pipeline — switch adapter in config
- One loss, one export, one verification suite

## Problem (§1)

Reconstruct 0–2 kHz speech from a body-conduction sensor (300–1200 Hz bandwidth,
time-varying). **Spec change: the model operates at sr = 16 kHz with STFT
n_fft=512 / win=480 (30 ms) / hop=160 (10 ms, 100 fps); input AND output are the
TRUNCATED COMPLEX spectrum of the 0–2 kHz band (keep_bins=64 (bins 1..64), 31.25 Hz/bin, DC dropped).**
Phase is the model's job (learned, not oracle) so it aligns naturally with the
reference road at fusion. Causal/streaming. The hard part is 1–2 kHz (high
harmonics + the unobserved-phase region).

## What's verified on CPU (§5)

Tests are **layered by data domain** (L0 synthetic ideal-lowpass / L1 real
Vibravox body-conduction).  `python verify.py` → L0: 23/23, L1: 5/5.

| Test | What it catches |
|------|-----------------|
| §5.1 Complexity | Param/MAC/memory (hand-counted vs tool, reconciled) |
| §5.2 Causality | Future-information leakage (bidirectional RNN, non-causal pad) |
| §5.3 Streaming≡Batch | LSTM state, conv buffer, normalization bugs |
| §5.4 Overfit | Implementation bugs (not "capacity") — all 3 arms overfit |
| §5.5 Gradient flow | Detached paths (DDSP F0 estimator trap) |
| §5.6 STFT roundtrip | COLA condition, boundary handling |
| §5.7 Loss stability | NaN/Inf on silence, zero, extreme, NaN input |
| §5.8 Data pipeline | Cutoff, roll-off, noise floor, time-variation verified |
| §5.9 Export | ONNX + TorchScript, output consistency <1e-4 |
| §5.10 Smoke train | Loss ↓, output not constant (NOT quality — just "runs") |
| §6.1 Anti-alias | Harmonics above Nyquist fold back as spurious tones |
| §6.5 F0 | YIN works at 4 kHz / 500 Hz LP; octave errors checked |

## What's NOT done (§9 — explicitly excluded)

- No quality comparison, no arm ranking (CPU data volume doesn't support it)
- No hyperparameter search
- No quantization/deployment
- No model shrinking for CPU — models are at target spec

## Key design decisions

See `reports/known_issues.md` for the full list. The most important:

1. **Frequency-only convolutions** (kernel (3,1)) — guarantees streaming≡batch
   equivalence without causal-conv buffering
2. **Complex-spectrum output** (spec change) — `forward` returns `"spec"`:
   complex64 (B, keep_bins=64, N) — bins 1..64, DC dropped, the truncated 0–2 kHz complex spectrum;
   phase is learned by the model (no oracle).  All three arms use this SAME
   complex64 format.  Loss = magnitude term (L1/L2/dB on |spec|, kept as main)
   + complex MSE (real/imag), weight 1:1 first.
3. **Arm A waveform-synth path ON by default** (spec change) — amplitude-domain
   harmonic Gaussian smear can't yield a complex spectrum; go oscillator →
   waveform → STFT → truncate.  Anti-alias mask cuts at the BAND TOP (2 kHz),
   not Nyquist (8 kHz) — above 2 kHz is synthesized-then-truncated = wasted.
4. **Arm C exceeds MAC budget** — inherent to F-T LSTM at 13K params; reported
   honestly, not hidden

## Config structure

Every config exposes (§7.2): data adapter & paths · model arm & hyperparams ·
loss items & weights · optimizer/scheduler · batch/steps · seed · log/ckpt paths.

```yaml
arm: arm_a_ddsp          # switch arm here
sample_rate: 16000       # spec change: 16 kHz (was 4 kHz)
stft_n_fft: 512
stft_hop: 160
stft_win: 480
keep_bins: 64            # bins 1..64 = 31.25-2000 Hz, DC dropped (model I/O)
device: auto              # §7.1: no hardcoded .cpu()/.cuda()
seed: 42
deterministic: false
data:
  adapter: lowpass_sim    # switch data here
  ...
loss:
  multi_res_stft: false   # §3.3: recipe modules off by default
  discriminator: false
  cplx_weight: 1.0         # complex MSE weight (1:1 with magnitude terms)
```

## Reports

- `reports/complexity_table.md` — §5.1 param/MAC/memory table
- `reports/verification_report.md` — auto-generated by `verify.py`
- `reports/known_issues.md` — compromises + recovery methods
- `reports/gpu_todo.md` — what to check on GPU
- `docs/data_adapter_guide.md` — how to plug in your data
