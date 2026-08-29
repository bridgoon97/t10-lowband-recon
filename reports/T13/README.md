# T13-A — fusion mechanism & streaming (report index)

> **T13-A only.** No effect metrics (G1–G6) are reported here — those are T13-B,
> gated on the declassified `0624/`/`0625/` real-device recordings arriving at
> the remote.  This stage implements the full 3-layer fusion algorithm + the
> streaming-causal test suite + the M1–M7 mechanism gates, all on SYNTHETIC
> signals (spec mandate).

## What is here

| path | role |
|---|---|
| `fusion/config.py` | **single source** of every placeholder constant + every factor's `enable_*` switch (B-stage tunes here only) |
| `fusion/stft.py` | causal full-spectrum STFT/iSTFT — batch reuses the audited `lowband.dsp.stft` pair; streaming `StftStreamer`/`IstftStreamer` reproduce it bit-identically |
| `fusion/f0.py` | causal YIN F0 from the SAME 480-sample STFT buf ⇒ **0 extra delay** beyond STFT; `f0_confidence = 1−CMND` (direction verified by M5) |
| `fusion/degrade.py` | D1–D4 stage-2 damage sim; D1 kills harmonics **weak→strong** (SI-SNR direction) |
| `fusion/align.py` | Layer 1: fixed DelayComp + robust causal EQ `C[f]` + change-point reset |
| `fusion/decision.py` | Layer 2: `c_V · g_f0 · w_band · w_local` + non-symmetric w smoother |
| `fusion/synthesis.py` | Layer 3: log-clip mix + weighted-vector-sum phase + complex-convex contrast arm + comfort noise |
| `fusion/fusion.py` | `Fusion` (batch) + `FusionStreamer` (per-hop) sharing `FusionCore.process_frame` |
| `fusion/utils.py` | causal EMA / non-sym EMA / online MSC / soft gate / 1-D smooth |
| `fusion/signals.py` | synthetic builders for the M1–M7 tests |
| `tests/test_t13_streaming.py` | **G5** (equiv < 1e-6, future-perturbation bit-identical, mutation sanity) |
| `tests/test_t13_mechanisms.py` | **M1–M7** + mutation sanity each |
| `tests/test_t13_ablation.py` | ablation interface existence (19 runs, all usable) |
| `tests/test_t13_static.py` | grep proof: algorithm path never references X / degrade internals |
| `fusion/run_t13_tests.py` | runner — **19/19 PASS** |
| `fusion/make_m_evidence.py` | regenerates the PNGs below |

## Mechanism-evidence PNGs (NOT effect conclusions)

- `m1_wlocal.png` — w_local at harmonics; killed (red) flagged, surviving (green) not.
- `m2_eq_convergence.png` — C[f] converging to the known ±6 dB tilt; gate ±1 dB at 3 s.
- `m3_cv_monotone.png` — c_V strictly decreasing as V weakens 0/−3/−6/−12 dB.
- `m4_asym.png` — non-symmetric w step response (slow rise ~130 ms / fast fall ~30 ms).
- `m5_g_f0_direction.png` — w tracks f0_confidence (voiced → higher w; direction NOT reversed).
- `m7_energy_dip.png` — log-clip holds 0 dB vs complex-convex −3 dB dip under 90° mismatch.

## T13-A results (gate summary)

| gate | criterion | measured | pass |
|---|---|---|---|
| **G5 equiv** | batch vs streaming interior diff < 1e-6 | **0.0** (bit-identical) | ✅ |
| **G5 future-perturbation** | zero future ⇒ past bit-identical | 5/5 cut points, **torch.equal** | ✅ |
| **G5 mutation sanity** | bidirectional-EMA ⇒ leak > 1e-6 | diff **2.29e-3** → caught | ✅ |
| **M1** | recall ≥0.90, FAR ≤0.10 @40% kill | recall **1.00**, FAR **0.00** | ✅ |
| **M2** | C[f] ±1 dB in 3 s | **0.83 dB** | ✅ |
| **M3** | c_V strictly monotone ↓ | 0.458>0.451>0.434>0.389 | ✅ |
| **M4** | rise/fall ratio ≥3 | 130/30 ms = **4.33** | ✅ |
| **M5** | voiced w > noise w | g 0.90>0.10; w 0.235>0.091 | ✅ |
| **M6** | \|Y\| ≥ \|V\|−(1−w)Δ | −1.000 ≥ −1.000 (boundary) | ✅ |
| **M7** | log-clip 0±0.5 dB; convex ≈−3 | 0.000 / −3.010 dB | ✅ |
| static | 0 forbidden refs in algorithm path | **0** | ✅ |
| ablation | each switch independently usable | **19/19** runs | ✅ |

Every M-gate has a mutation sanity that deliberately breaks the mechanism and
shows the SAME test now FAILS (the failing value is printed in the test output).

## Rework (R1–R4) — appended on top of d27e956

After reviewer acceptance (main body PASS), three reworks + one real-envelope
re-test were appended.  Test count: **19 → 25** (`fusion/run_t13_tests.py` →
25/25 PASS, 0 FAIL, 0 SKIP).

| item | what | result |
|---|---|---|
| **R1** | `test_M5_mutation` — flip GF0 to CMND (the project前科 direction); both M5 assertions fail (direct `g_v>g_n` → 0.10>0.90 false; full-pipeline `mv>mn` → reversed) | mutation caught ✓ |
| **R2** | G5 future-perturbation on REAL voiced FF/VPU (0624) — past bit-identical (worst 0.0); real-VPU smoke: runs/finite/causal ✓. Two mutations on voiced: (a) bidir w-EMA leak **1.46e-4** (smaller than white 1.8e-3 — voiced w smoother; still caught); (b) w_local LOOK-AHEAD leak **4.29e-2** ≫ white **6.6e-4** — voiced gives the test MORE power for the w_local path (the reviewer's actual concern) | ✓ |
| **R3** | oracle-F0 backdoor DELETED (config `f0_use_oracle` + fusion.py override + `oracle_f0` param + dead `f0_tr` line). Static check now INCLUDES `config.py` and forbids `oracle|f0_use_oracle|_oracle_f0`; mutation sanity re-introduces it → grep finds 2 hits → static FAILs | ✓ |
| **R4** | M1 re-test on REAL in-band (≤2 kHz) speech harmonic envelope (D1=40% weakest). Threshold UNCHANGED; REPORT item not gate. **recall=0.773 (<0.90), FAR=0.178 (>0.10) — BELOW threshold, reported honestly, NO tuning.** Cause: the linear-across-k envelope fit is too rigid for real formant undulation (formant-valley SURVIVING harmonics mis-flagged). B-stage input — needs a more flexible envelope model (local/higher-order), NOT a parameter tweak. | below (honest) |

R4 evidence: `reports/T13/r4_real_envelope.png`.
Records (R5/R6/R7) noted by the reviewer for B-stage; R7 (assert-message newline,
dead `f0_tr`) fixed in this rework.
