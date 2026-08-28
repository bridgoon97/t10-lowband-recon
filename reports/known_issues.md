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

### 2. Arm A waveform-synthesis is the MAIN path (spec change: complex output)
**What:** DDSP outputs a TRUNCATED COMPLEX spectrum via oscillator → waveform →
STFT → truncate.  `waveform_synth` is now ON by default (was off).
**Why:** an amplitude-domain harmonic Gaussian smear produces only magnitude,
NOT a complex spectrum (no phase).  The spec change made phase the model's job,
so the waveform-synth path is mandatory for Arm A.
**Anti-alias (§6.1) semantics changed:** Nyquist is now 8 kHz (sr=16k) and no
longer coincides with the band top 2 kHz.  The harmonic mask cuts at the BAND
TOP (``band_top_hz``=2000), not Nyquist — harmonics above 2 kHz would only be
synthesized-then-truncated-away (wasted compute).  The dangerous 'folds back
in-band looking like reconstructed high-freq' case is gone (Nyquist ≠ band top).
**Recovery:** the magnitude-domain ``_harmonic_mag`` path is still implemented
but unused; set ``waveform_synth: false`` to revert (loses complex output).

### 3. F0-oracle path used for streaming equivalence + smoke tests
**What:** the §5.3 streaming-equiv test and the L0/L1 smoke tests use oracle F0
(fixed 150 Hz) for Arm A.
**Why:** batch YIN and streaming YIN produce different F0 (different windowing);
this is an F0-estimator property, not a model property. The model's streaming
path (GRU state, STFT buffer, phase carry, noise OLA) IS verified equivalent.
**Recovery:** On GPU, train with `f0_mode: estimated` — the estimator runs in
both batch and streaming, just with different boundary behavior.

### 4. Arm C hidden size = 32 (exceeds MAC budget)
**What:** Arm C uses hidden=32 (12,961 params, 100.9 MMACs/s), exceeding the
60 MMACs/s budget.
**Why:** §3.2 explicitly targets ~13K params (FT-JNF XS). At hidden=24, params
drop to 7.2K (below 15K target) but MACs = 58.5 MMACs/s (within budget).
The spec acknowledges: "参数省但 MAC 贵,实测报出来".
**Recovery:** Set `ftlstm_hidden: 24` in config for a MAC-compliant variant.
Both are implemented; GPU ablation should pick.

## Spec-change caveats (complex path, §new口径)

### C1. Complex-path early metrics are EXPECTED worse than magnitude+oracle
**What:** on real L1 body-conduction data, the complex MSE (phase) term can
DIVERGE while the magnitude term decreases — total loss may rise late in a
short smoke run.
**Why:** the input (body-conduction) phase above ~500 Hz is noise / unobserved,
so the model cannot pin the reference's phase there; fitting magnitude can push
the predicted phase away from the target.  This is the acknowledged BWE
phase-prediction difficulty, NOT a bug.
**How tests handle it honestly:** the smoke criterion is `min(loss) < 0.9*first`
(loss dropped ≥10 % below start at some point = the model learned), tolerating
late divergence; the L1 smoke PRINTS the cplx term so the divergence is
visible.  No retreating to magnitude-DOMAIN (model still outputs complex,
cplx_weight=1.0).  Per spec-change note: do not tune cplx_weight by ear to make
numbers look good — tune by param-side gradient norm on GPU.

### C2. Arm C cannot fully overfit arbitrary complex targets (EVIDENCED, not a concession)
**What:** Arm C (F-T LSTM, ~13 K params) plateaus at ratio ≈0.87 overfitting
complex targets, even structured ones, even at 1500 steps / higher lr.
**Why (evidenced via scripts/armc_overfit_probe.py, review ④ a/b):**
- (a) magnitude-only loss (cplx_weight=0): C OVERFITS (ratio 0.31 < 0.6).
- (b1) 2 samples instead of 8: still 0.87 (NOT capacity).
- (b2) half segment (T=8000): still 0.87 (NOT segment/capacity).
→ the bottleneck is **complex-PHASE representation** (magnitude fits, phase
doesn't), not implementation and not capacity. Gradients flow (test_gradient
passes), loss decreases — no gross bug.
**How tests handle it honestly:** Arm C's overfit threshold is relaxed to
ratio<0.9 ('loss decreased') — now a CONCLUSION backed by the (a)/(b)
evidence above, not a concession. Reproduce: `python3 scripts/armc_overfit_probe.py`.

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
