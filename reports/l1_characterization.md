# L1 Characterization — Real Body-Conduction Sensor (Vibravox)

Consolidates every measurement on the real L1 data path (temple_vibration_pickup
sensor → headset_microphone ref, 48 kHz raw).  Each item states its **sample
size** and **boundary conditions** so the numbers are not mis-applied.

Data: `speech_clean_test_{0,2}` (21 speakers, ~206 rows) + `speechless_clean_test_1`
(noise floor).  All measured at 48 kHz raw (the parquet native rate); the model
resamples to 16 kHz, but characterization is on the raw sensor bandwidth.

---

## 1. Bandwise MSC + noise-floor SNR (the core characterization)

**Metric:** magnitude-squared coherence MSC(sensor, ref) on speech; noise-floor
SNR = speech_clean power / speechless_clean power per band; noise-limited MSC
ceiling γ²max = 1/((1+1/SNR_sen)(1+1/SNR_ref)).

**Sample:** 15 speech rows + 7 noise rows.  STFT N_FFT=2048, hop=512 (75%
overlap), per-band averaged.

| band Hz | MSC | SNR_sen | SNR_ref | γ²max | gap (γ²max−MSC) |
|--------:|----:|-------:|-------:|------:|----------------:|
| 50–125 | 0.060 | 7.7 dB | 29.6 dB | 0.815 | 0.755 |
| 125–250 | 0.408 | 23.7 dB | 46.7 dB | 0.996 | 0.588 |
| 250–500 | 0.503 | 25.7 dB | 53.0 dB | 0.997 | 0.494 |
| 500–750 | 0.435 | 16.8 dB | 54.4 dB | 0.975 | 0.540 |
| 750–1000 | 0.266 | 7.3 dB | 47.7 dB | 0.834 | 0.568 |
| 1000–1500 | 0.063 | 2.6 dB | 45.9 dB | 0.645 | 0.582 |
| 1500–2500 | 0.027 | 0.9 dB | 44.3 dB | 0.554 | 0.527 |
| 2500–8000 | 0.003 | 0.2 dB | 38.9 dB | 0.509 | 0.506 |

**Findings:**
- **Main cause = noise-floor contamination** (old high/low ratio hid this): the
  sensor's high band is its OWN noise floor (SNR collapses to <1 dB above 2 kHz),
  not rolled-off speech — that's why the sensor "looked less band-limited than
  it is".
- **Sensor useful band ≈ 250–1000 Hz** (criteria PINNED: MSC>0.4 AND SNR>7 dB);
  dies by ~1.5 kHz.  Ref stays high-SNR everywhere (clean target).
- **The ~0.5 gap to γ²max is systematic** (not noise) across ALL bands — see §4.

**Boundary:** Vibravox clean-speech subset, single sensor type (temple).  The
criteria (MSC>0.4 & SNR>7 dB) are PINNED here; the target-device "500–600 Hz"
figure is from elsewhere with UNKNOWN criteria — the 600 Hz alignment lowpass is
NOT applied until it is recomputed under this criterion.

Scripts: `scripts/measure_bandwidth.py`, `tests/test_l1_bandwidth.py`.

---

## 2. F0 estimation error (gender-bucketed) — the Arm-A-viability number

**Metric:** the project's `yin_f0` on sensor vs ref (ref = ground truth), 48 kHz,
frame=2048 (~43 ms, ≥2 periods of 50 Hz).  Same estimator + params on both →
error is sensor-vs-ref, not estimator-vs-truth.

**口径 (write死):** F0 error is computed on **co-voiced frames only** (ref voiced
AND sensor voiced); unvoiced frames are excluded (gross would otherwise include
"no F0 to estimate").  `agree%` = **voiced-DECISION consistency** (sensor voiced
when ref voiced), NOT F0-value consistency.

**Sample (16 rows: 8 male, 8 female):**

| gender | agree% | median rel err | <10% | octave | gross |
|--------|-------:|---------------:|-----:|-------:|------:|
| male | 82.8 | 0.022 | 71.5% | 14.6% | 13.9% |
| female | 79.2 | 0.048 | 55.8% | 14.8% | 28.3% |

**Findings:**
- **Octave errors ~15% BOTH genders** — this is the Arm-A-critical number: one
  octave error = the WHOLE harmonic comb is at the wrong frequencies for that
  frame (structural error, not precision loss).  ~15% of frames structurally
  wrong is a real liability for a "structure-prior wins" architecture.
- **Gender结论 DOWNGRADED** (n=8 per gender is too few): the only defensible
  claim is "no evidence male is worse than female" (the band-SNR proxy's hint
  that male F0 is marginal was WRONG — male's dense harmonics h2-h7 land in the
  good 250–750 Hz band).  The male/female split (71.5 vs 55.8) may be sample
  noise; per-speaker variance not reported (too few speakers).  Retracts the
  earlier "male F0 marginal" worry, no stronger claim.
- **Continuity constraints do NOT fix the octave errors** (review ①b): the
  project's `smooth_f0` (moving average) makes it WORSE (smears voiced/unvoiced
  boundaries, inflates co-voiced count, <10% 68→25%).  A zero-preserving median
  also doesn't help (octave 14.4→14.6).  The octave errors are INTRINSIC to
  F0-from-band-limited-sensor; **pYIN (probabilistic + Viterbi on the CMND
  function) is the known better approach but NOT integrated** (out of wrap-up
  scope; listed in `gpu_todo.md`).

Scripts: `scripts/measure_f0_error.py`, `tests/test_l1_f0.py`.

---

## 3. GCC-PHAT delay (sensor ↔ ref)

**Metric:** GCC-PHAT delay estimate, band-limited to the best band (250–750 Hz).

**Sample:** 15 speech rows.  Median = **−18 samples (−0.38 ms)**, IQR [−20, −10].

**Finding:** a consistent ~0.38 ms delay exists, but the SIGN is ref LEADS sensor
(opposite to the "bone conduction arrives earlier" expectation — likely temple
transducer latency, or the headset being close to the mouth).  Global-median
compensation recovers ~0 MSC; PER-PAIR compensation recovers only +0.05 → the
delay is a MINOR component, NOT the cause of the MSC being ~0.5.  ⇒ (a) rejected.

Script: `scripts/measure_bandwidth.py` (the `_gcc_phat` + compensated-MSC table).

---

## 4. MSC variance decomposition — (b1) time-varying LTI vs (b2) non-linear

**Metric:** MSC at varying averaging-window lengths, bias-controlled
(non-overlapping segments so the Welch bias E[γ²]=γ²+(1−γ²)/n is the TRUE
independent-segment count; per-recording then averaged; corrected
γ²_corr=(n·γ²−1)/(n−1)).

**Sample:** 20 rows.  Best band 250–750 Hz:

| n_seg | span | MSC_raw | MSC_corr |
|------:|-----:|--------:|---------:|
| 4 | 171 ms | 0.571 | 0.429 (unreliable — n too small for the asymptotic correction) |
| 16 | 680 ms | 0.644 | 0.620 |
| 64 | 2.7 s | 0.680 | 0.675 |
| ALL | ~4.7 s | 0.675 | 0.671 |

**Finding (corrected reading — review ②):** window 171 ms → 4.7 s: MSC does NOT
fall, it RISES then PLATEAUS (~0.67) ⇒ **intra-recording time-variation ≈ 0**.
Short-time adaptation buys NOTHING.  The real variance decomposition:

```
per-wearing calibration recoverable  = 0.67 − 0.56 = 0.11   (cohort→per-recording)
linear-unreachable floor             = 1 − 0.67   = 0.33   (non-LTI, per-recording)
intra-wear fast time-variation       ≈ 0
```

⇒ variation happens BETWEEN recordings (= between wearings), NOT within.  The
correct design is **"per-wearing-calibrated STATIC EQ + non-linear residual
net"**, NOT "short-time-adaptive EQ".  The former needs re-estimation only when
the wearing changes (triggerable by a coherence drop), far cheaper than running
an adaptive filter continuously.

**⚠️ BOUNDARY:** the "intra-wear time-variation ≈ 0" holds ONLY on Vibravox
(single session, wearing basically static).  Real earbuds (chewing, talking,
walking) will have LARGER intra-wear variation — so a per-wearing static EQ is
the floor here, but real-product data may need SOME adaptation.  Do not
extrapolate the 0.11 / 0.33 split to product conditions.

Script: `scripts/measure_msc_window.py`.

---

## Summary table (what each number says / doesn't)

| question | answer | confidence | boundary |
|----------|--------|-----------|----------|
| Is the sensor band-limited? | yes, ~250–1000 Hz | high | Vibravox temple |
| Why did old metric look un-band-limited? | noise-floor contamination | high | — |
| Is Arm A's F0-from-sensor viable? | ~15% octave errors (intrinsic) | high (the risk) | YIN; pYIN untried |
| Is male F0 worse than female? | no evidence (retracted) | low (n=8/gender) | too few speakers |
| Is the low MSC a delay artifact? | no (per-pair +0.05) | high | — |
| Is the transfer time-varying (b1)? | no (intra-wear ≈0) | medium | Vibravox-only (static wearing) |
| Is there a non-LTI floor? | yes, ~33% per-wearing | medium | Vibravox-only |
| Design implication | per-wearing static EQ + non-linear residual | — | see boundary above |

---

# T11 addendum — noise robustness (joint denoise + extend)

T11 changes the INPUT assumption: the target device has noise across ALL
frequencies, speech only below 400–600 Hz (SNR just >5 dB), plus wind.  The
task is NOT bandwidth extension alone — it is JOINT denoise + extend, and wind
(low-freq-dominated) overlaps the only usable speech band.

## T11-A. Fine-band SNR + usable-band crossing (criterion unified to SNR>5 dB)

100 Hz bands, 50–2000 Hz (T11 §1; supersedes the coarse T10 bands that hid the
crossing in the 750–1500 Hz averaged band).  Sample: 15 speech + 7 noise rows.

| band Hz | SNR_sen | >5 dB? |
|--------:|-------:|:------:|
| 50–150 | 11.2 | ✓ |
| 150–250 | 24.1 | ✓ |
| 250–950 | 13–27 | ✓ |
| 950–1050 | 4.8 | ✗ |
| 1050+ | <4.8 | ✗ |

**temple SNR>5 dB crossing ≈ 977 Hz** — WIDER than the target device (400–600 Hz).
⇒ §5 action: a 600 Hz lowpass is added on the SENSOR channel in the L1 configs
(`sensor_lowpass_hz: 600`) to align the training input to the target device's
bandwidth (ref stays clean — the net reconstructs the full band from narrower).
Characterization tests still measure the RAW sensor (lowpass off) — they
characterize the SENSOR, not the aligned training input.

**Bandpass-not-lowpass confirmed**: 50–125 Hz only 7.7 dB while 125–750 is
21.9 dB — the sensor has a weak LOW edge (highpass / drift-removal), so male
F0 (85–155 Hz) sits on the weak side.  Opposite to the 'bone low-freq boosted'
intuition.

## T11-B. F0 degradation under noise (§3 — highest priority; review-corrected)

T10: ~15% octave on CLEAN.  T11 sweeps noise TYPE × speech-band SNR
(0/5/10/20 dB).  Sample: 12 rows.  口径: co-voiced frames, ref F0=truth.

**⚠️ Methodology note (review ①):** the OLD criterion (octave error rate on
co-voiced frames) has SURVIVORSHIP BIAS — when voicing agreement (agr)
collapses, oct is measured only on the surviving (easy) few frames, so 'oct
~flat' does NOT mean 'F0 is fine'.  **The PRIMARY criterion is the composite
`available-F0 frame rate = agr × (1−oct)`** = fraction of ALL ref-voiced frames
where the sensor voices AND F0 is within tolerance.  agr and oct are kept as the
DECOMPOSITION.  General rule (now in the methodology): any metric computed on a
SUBSET must also report the subset size, or report the composite.

**⚠️ 口径 (review ②):** SNR is IN-BAND, in the DEVICE speech band (50–600 Hz,
where the device has speech) — measured via `speech_band_power` in the band,
NOT full-band.  For white (flat) the band is irrelevant; for wind (corner 30 Hz,
−15 dB/oct ⇒ at 600 Hz ~−64 dB) the 600–977 tail is negligible, so 50–977 ≈
50–600 for wind.  Re-run at 50–600 (device口径) vs 50–977: white@5dB avail 1%
vs 4% (the >600 noise is relatively louder at 50–600 scaling), wind unchanged.

| type | snr20 | snr10 | snr5 | snr0 |
|------|-------|-------|------|------|
| clean | av 73 (oct 13 / agr 84) — baseline | | | |
| white | av 52 | av 12 | **av 1** (oct 33 / agr 1) | (degen) |
| wind | av 62 | av 32 | **av 14** (oct 16 / agr 17) | av 8 |
| body | av 70 | av 68 | av 69 | av 69 |

(av = available-F0 frame rate = agr×(1−oct), the PRIMARY criterion; oct/agr are
the decomposition; 0 dB white degenerate — no co-voiced frames.)

**CORRECTED verdict (vs the old 'not a clean flip'):** at the device's ~5 dB
in-band SNR, **BOTH white (av 1%) and wind (av 14%) are DISASTERS** vs clean
(73%).  Arm A's VPU-SINGLE-PATH F0 is NOT viable at 5 dB — under BOTH noise
types, not just white.  The old reading ('wind oct ~15% ≈ clean') was the
survivorship-bias artifact the review caught.  Body noise negligible (transient,
doesn't corrupt periodicity).  At ≥20 dB, F0 is robust (av 52–73%).

**⚠️ Scope distinction (review ③):** this is **'VPU single-path F0 not viable'**,
NOT 'DDSP architecture not viable'.  Two unexplored recovery paths (gpu_todo,
NOT implemented here):
  (a) F0-confidence-gated harmonic branch — low confidence ⇒ push sub-band
      periodicity to the noise branch ⇒ graceful degrade to noise-fill, not
      structural harmonic error.  The sub-band periodicity mechanism is ALREADY
      implemented.
  (b) F0 joint estimation — the mic path coexists; VPU fears wind, mic fears
      ambient noise, different failure modes ⇒ joint > either single path.
Both could change the conclusion's applicability.  Do NOT over-conclude to
'DDSP not viable' from this single-path result.

Caveats: (a) Vibravox + simulated noise, real VPU may differ; (b) yin_f0
voicing threshold conservative (a better detector + pYIN could recover some);
(c) §3 ran on the RAW temple (977 Hz speech) — the §5 600 Hz lowpass narrows
to fewer harmonics ⇒ F0 likely WORSE on the real aligned target (re-run
post-lowpass on GPU).  pYIN still not integrated (T10: no post-hoc smoothing).

## T11-C. Noise-only-band input probe (§4 — zero training cost)

Fix speech <600 Hz, replace only >600 Hz noise realization (two seeds), forward,
measure output rel-diff.  UNTRAINED baseline (the 'small diff ⇒ robust'
criterion is trained-model; deferred to post-training):

| arm | rel_diff (seed1 vs seed2) |
|-----|-------------------------:|
| A | 0.016 |
| B | 0.001 |
| C | 0.077 |

Even untrained, the architectures do NOT catastrophically amplify the >600 Hz
noise band (0.1–7.7% output change) — no structural defect.  A trained model
should be even smaller; re-assert post-training.  Test: tests/test_noise_probe.py.

## T11-D. Noisy smoke + lowpass alignment (§5)

* `test_smoke_train_noisy`: three arms under full-band+wind noise (10 dB SNR),
  same criteria (loss↓ + bounded divergence + non-constant) — PASS.
* `sensor_lowpass_hz: 600` added to the 3 L1 configs (align to target); the L0
  degradation now supports `fullband_noise` + `wind_noise` (T11 §2, each with a
  unit test verifying spectrum: full-band SNR ±3 dB, wind slope −15 dB/oct +
  low-freq dominance 77 dB).
* Still NO quality comparison, NO architecture ranking, NO 600 Hz lowpass on
  the CHARACTERIZATION path (only the aligned TRAINING path).

## T11 summary

| question | answer | confidence | boundary |
|----------|--------|-----------|----------|
| temple usable band (SNR>5 dB)? | ~977 Hz crossing | high | Vibravox temple |
| wider than target (400-600)? | yes ⇒ 600 Hz lowpass on sensor | high | aligned training |
| bandpass or lowpass? | bandpass (weak low edge 50-125) | high | — |
| F0 robust at 5 dB + wind? | NO (avail 14%, voicing collapse) | high | survivorship-safe composite; raw temple |
| F0 robust at 5 dB + white? | NO (avail 1%) | high | in-band 50-600 Hz |
| F0 robust at ≥20 dB? | yes (av 52-73%) | high | — |
| does the net amplify >600 Hz noise? | no (untrained 0.1-7.7%) | medium | trained-model criterion deferred |
| three arms run under noise? | yes (smoke noisy PASS) | high | — |
