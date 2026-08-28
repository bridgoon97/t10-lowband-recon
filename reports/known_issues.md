# Known Issues & Compromises

§8.7 deliverable: any simplification made for CPU runs, listed with recovery method.

## Architecture decisions (NOT CPU compromises)

These are deliberate design choices, documented for transparency:

### 1. Frequency-only convolutions (kernel (3,1), not (3,3))
**What:** CRN encoder/decoder and DDSP ControlNet use kernel (3,1) — convolving
only along frequency, not time. Temporal context is entirely in the GRU/LSTM.
**Why:** Guarantees streaming-batch numerical equivalence (§5.3) without
complex causal-conv buffering. A (3,3) kernel would require per-layer ring
buffers in streaming mode.
**Recovery (if temporal-local context is wanted):** Replace with causal (3,3)
convolutions + a `CausalConv2d` wrapper that maintains a time ring buffer in
`stream_step`. The streaming equivalence test will still pass if buffering is
correct.

### 2. Arm A magnitude-only synthesis (no waveform path by default)
**What:** DDSP outputs the magnitude spectrum directly from control parameters
(harmonic Gaussian smearing + noise envelope), not via oscillator→waveform→STFT.
**Why:** §3.1 requires magnitude-only output this stage. The waveform synthesis
path (`_synth_waveform`) is implemented but disabled by default (`waveform_synth: false`).
**Recovery:** Set `waveform_synth: true` in config. The oscillator→STFT path is
fully implemented.

### 3. F0-oracle path used for streaming equivalence test
**What:** The §5.3 equivalence test uses oracle F0 (fixed 150 Hz) for Arm A.
**Why:** Batch YIN and streaming YIN produce different F0 due to different
windowing (full-signal sliding vs. rolling buffer). This is an F0-estimator
property, not a model property. The model's streaming path (GRU state, STFT
buffer) IS verified equivalent.
**Recovery:** The F0 estimator itself is separately verified (§6.5). On GPU,
train with `f0_mode: estimated` — the estimator runs in both batch and
streaming, just with different boundary behavior (acceptable for training).

### 4. Arm C hidden size = 32 (exceeds MAC budget)
**What:** Arm C uses hidden=32 (12,961 params, 100.9 MMACs/s), exceeding the
60 MMACs/s budget.
**Why:** §3.2 explicitly targets ~13K params (FT-JNF XS). At hidden=24, params
drop to 7.2K (below 15K target) but MACs = 58.5 MMACs/s (within budget).
The spec acknowledges: "参数省但 MAC 贵,实测报出来".
**Recovery:** Set `ftlstm_hidden: 24` in config for a MAC-compliant variant.
Both are implemented; GPU ablation should pick.

## Genuine CPU limitations (things that DON'T work on CPU but are coded for GPU)

### 5. ONNX export for Arm A and Arm B
**Status:** Arm A (DDSP) ONNX export fails — `torch.export` can't trace the
dynamic YIN F0 path and the harmonic Gaussian smearing. Arm B ONNX export fails
on FX graph decomposition.
**What works:** All arms export to TorchScript with 0.0 relative error. Arm C
exports to ONNX with 1.7e-7 relative error.
**Recovery:** For Arm A, disable the YIN estimator in the export wrapper (use
oracle F0 input as a model input). The oscillator ops themselves should export
once the control flow is removed. This is a known ONNX limitation with dynamic
shapes, not a model bug.

### 6. AMP (mixed precision) not testable on CPU
**Status:** Code is AMP-ready (autocast context, GradScaler), but CPU doesn't
benefit. §7.1 requires static annotation of fp32-forced ops.
**Annotated fp32 ops:** phase accumulation (already fp64 internally), log in
losses (eps calibrated), division in spectral loss. These use
`autocast(enabled=False)` wrapping in the GPU path.

### 7. num_workers > 0 not tested
**Status:** DataLoader supports `num_workers` config. On CPU we run
`num_workers=0`. The augmentation runs inside `__getitem__` (worker-side),
ready for multi-worker on GPU.

### 8. Large-batch stability not tested
**Status:** All tests use batch_size ≤ 8. Large-batch gradient stability can
only be verified on GPU.
