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

### C3. Arm A cannot express arbitrary per-harmonic amplitudes (envelope path)
**What:** Arm A's harmonic amps are NOT free parameters — `_harmonic_amps`
samples `env_lin` (the spectral envelope) at the harmonic bins, and `env_lin`
comes from the control_net → mel filterbank → pseudoinverse (a SMOOTHING
path).  So Arm A can only produce harmonic-amp patterns that match a SMOOTH
spectral envelope.
**Evidence (review finding F probe):** overfitting a target with RANDOM
per-harmonic amps floors Arm A at ratio ~0.15 (magnitude-only, self-overfit,
n_mel-invariant 16→32) — a STRUCTURAL limit, not a bug.  A SMOOTH-formant
target (envelope-representable) overfits to 0.003.
**Why it matters for SELECTION:** this caps Arm A's expressivity at
harmonic-amp patterns that a mel-envelope can represent.  For real speech
(the formant structure IS roughly smooth) this is fine; for targets requiring
sharp per-harmonic control it is a ceiling.  B/C are direct spec regressors
and have no such limit.
**Recovery:** widen the mel filterbank / drop the pinv bottleneck, or add a
per-harmonic residual head on top of the envelope-derived amps.

### C4. Arm A F0 under noise — stress test vs real-device operating point (conclusion OVERTURNED; see below)

**CURRENT conclusion (corrected after a real-device metro SNR review):**
- **Arm A is RETAINED** — it is NOT eliminated by the noise-robustness work.
  The earlier "VPU-single-path F0 not viable at the device's 5 dB" reading was
  OVERTURNED (see the historical block below).
- **5 dB was a STRESS-TEST point, NOT the actual metro operating point.** The
  real metro 1/3-octave in-band SNR is ~10–14 dB at 100–400 Hz, 8.2 dB at
  500 Hz, 3.3 dB at 630 Hz; the usable band (SNR > 5 dB) is ~100–500 Hz.
- **The old low `available-F0` numbers were a HARD-VOICING-THRESHOLD artifact.**
  YIN's default `conf < 0.15` is over-conservative under noise, flagging many
  correct-F0 frames as unvoiced; WITHIN the retained (voiced) frames the F0
  correctness was 98.4–99.6 %.  The composite metric `available-F0 =
  agr×(1−oct)` is still the right (survivorship-safe) criterion — but `agr`
  there was driven by the hard threshold, so the low `agr` was a detector
  artifact, not real F0 failure.
- **REQUIRED design (not optional): SOFT CONFIDENCE GATING.** Do NOT make a
  hard voicing decision; use F0 confidence as a SOFT weight modulating
  per-sub-band periodicity (high confidence ⇒ harmonic branch, low ⇒ noise
  branch, no threshold anywhere).  This is required because even at the real
  operating point the worst-speaker profile still drops: available-F0 median
  80.7 % (worst 61.1 %) at conf < 0.4.  Soft gating is a required item, not an
  enhancement.  (Implementation is task ② — NOT done here; this issue is
  doc-only; no model code touched.)
- **pYIN is a LOW-PRIORITY comparison/enhancement**, NOT the key recovery path
  or a blocker.  It may trim the ~15 % intrinsic octave errors (clean) but is
  not what restores Arm A.
- **Two REAL risks REMAIN** (do not read this correction as "Arm A has no risk"):
  (i) weak-voice speakers need the soft gate — without it the worst-speaker
  available-F0 is 61.1 %; (ii) the per-harmonic-amplitude expression limit (C3,
  mel+pinv envelope, random-amp floor ~0.15) is a SEPARATE structural ceiling,
  unchanged by this correction.

**Provenance split (do not conflate):**
- PUBLIC / reproducible in this repo (Vibravox parquet + simulated white/wind
  noise + `yin_f0` at `conf < 0.15`): the T11-B stress-test numbers (e.g.
  white@5 dB `av` 1 %, wind@5 dB `av` 14 %; clean ~73 %; lowpass-sweep
  1–3 %/11–21 %).  These are honest stress-test measurements — their VALUES
  stand; the CONCLUSION drawn from them ("Arm A not viable") does not.
- PRIVATE / real-device metro review (NOT committed, NOT reproducible from this
  repo; cited only as summary statistics supplied for accurate documentation):
  the 100–400 Hz 10–14 dB operating-point SNR, the 98.4–99.6 % within-retained-
  frames F0 correctness, and the conf-threshold available-F0 table (metro-avg
  median 71.8/90.9/97.5/99.0, metro-avg worst 50.0/72.9/81.6/83.8, worst-speaker
  median 22.3/55.3/80.7/92.4 at conf<0.15/0.25/0.4/0.6).

**OVERTURNED earlier conclusion (historical, kept per review — do NOT read as
  current):** the prior C4 titled "Arm A's VPU-single-path F0 is NOT viable at
  the device's 5 dB noise" read the T11-B stress-test numbers as a current
  verdict and listed pYIN + a robust voicing detector as recovery paths for a
  5 dB operating point, with "if none recovers 5 dB, favor a regression arm
  (B/C)".  That is WRONG: 5 dB is not the operating point, and the failure was
  the hard threshold, not F0.  Arm A is retained; the required path is soft
  confidence gating (task ②), and pYIN is demoted to low-priority comparison.
  See `l1_characterization.md` T11-B for the stress-test data + the overturn
  note.

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

### C2. Arm C overfits fine — old "cannot overfit" was a target-size artifact (RETRACTED)
**What (RETRACTED):** an earlier note claimed Arm C plateaus at ratio ≈0.87
overfitting complex targets and relaxed C's threshold to 0.9, attributing it to
a "complex-phase representation bottleneck" (backed by the ④ a/b probe).
**Why that was WRONG (review finding F):** the old overfit target was
B=8×T=16000 → 102,400 complex values vs C's 13,122 params = target OUTRUNS
params 7.8×, so NO implementation could overfit it — the (a)/(b) probe ran
under the same capacity-confounded target, so its 'phase bottleneck'
conclusion was misread (the bottleneck was target SIZE, not phase).
**Corrected evidence (tests/test_overfit.py):** shrink the target to B=1,
T=4000 → 25 frames → 3,200 complex values (C has 4.1× param headroom), use a
REPRESENTABLE smooth-formant target + self-overfit (input=target), 1000 steps
@ lr=1e-2.  Result: A=0.003, B=0.017, **C=0.006** — all <0.1 with a UNIFORM
threshold (no per-arm relaxation).  C's LSTM simply needed more steps + higher
lr than A/B; given that, it overfits as well as the others.
**How tests handle it honestly now:** uniform threshold ratio<0.1 for ALL arms;
failure is a real bug (4–10× param headroom + representable target remove the
capacity/representation alibis).  The ④ a/b probe (scripts/armc_overfit_probe.py)
is SUPERSEDED — kept for the capacity-confound evidence, with a retraction note.

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
