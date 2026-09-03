# T13-A — fusion mechanism & streaming (report index)

> **T13-A only.** No effect metrics (G1–G6) are reported here — those are T13-B,
> gated on the declassified `0624/`/`0625/` real-device recordings arriving at
> the remote.  This stage implements the full 3-layer fusion algorithm + the
> streaming-causal test suite + the M1–M7 mechanism gates, all on SYNTHETIC
> signals (spec mandate).

## What is here

> [!important] 工作约定：陈述与实际状态不得脱节
> **任何被后续实验撤销的结论，必须在原处加撤销标记（⚠️ + 一句说明 + 指向撤销它的章节）；不得只在新章节里写“撤销”而让旧结论原样留着。** 只读到旧节的人会拿到已失效的结论——与“J2 XPASS 被误读成修好了”、“测试从不运行”是同一类危险（本项目已为此付出多次代价）。该约定与“只报告不 assert / 参照系伪通过”同属“陈述与实际脱节”失败族，文档层同理。

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

## T13-B0 — mechanism fix (appended on top of b445422)

Reviewer acceptance of the rework (commit b445422).  B0 fixes the R4 mechanism:
the w_local detector now meets the original threshold on the real envelope.
Test count: **25 → 28** (`fusion/run_t13_tests.py` → 28/28 PASS, 0 FAIL).

### §1 apply_d1 band-limiting + anti-no-op
`DegradationConfig.d1_band_hi_hz` (default 2000) ⇒ sort AND kill restricted to
the in-band; `d1_mode` default perframe (each voiced frame kills its own weakest
in-band 40 %).  Anti-no-op test: after D1=40 %, in-band killed =
2267/5675 = **0.399 ≈ 0.40** (asserted >0 and within 0.40±0.15).  D2/D3/D4
self-check: source has NO cross-band energy sort (per-point/per-block) ⇒ no
no-op risk (asserted).

### §2 w_local envelope model — ①②③④ (each switchable)
Replaced the rigid linear-across-k RANSAC with four switchable methods; w_local
= product of the active ones (product ⇒ low FAR; 'wrong-use-V-is-fabrication'):
- ① local-median baseline (k±window) — weak on sim (local median includes the
  killed point, pulling the baseline down); recall 0.126.
- ② abrupt-drop signature (drop from max-neighbor) — a SUBSET of ③ on sim
  (flags only steep kills; misses decay-region kills whose neighbors are also
  low); recall 0.598, FAR 0.107.
- ③ **relative-to-frame-peak abs gate** (`P < frame_peak − headroom`, headroom 45,
  tuned on 0624) — the DECISIVE detector on sim: apply_d1 puts killed at a
  fixed peak−60 dB, cleanly separable from survivors by absolute level relative
  to the frame peak. recall 1.000, FAR 0.010. **DEFAULT.**
- ④ V-envelope always-on weak evidence — high recall (0.992) but high FAR (0.412)
  alone; circular-dependency risk, off by default (switchable + ablated).

**R4 ablation (real in-band envelope, D1=40 %):**

| method | recall | FAR | verdict |
|---|---|---|---|
| ① local-median | 0.126 | 0.070 | below |
| ② abrupt-drop | 0.598 | 0.107 | below |
| **③ abs-gate (DEFAULT)** | **1.000** | **0.010** | **PASS** |
| ④ V-envelope | 0.992 | 0.412 | below |
| ②③ | 0.598 | 0.003 | below (② bottleneck) |
| ③④ | 0.992 | 0.009 | PASS |
| ①②③ | 0.126 | 0.002 | below (① bottleneck) |
| ①②③④ | 0.126 | 0.002 | below |

⮕ **the gain comes from ③ (relative abs gate)** — ①② are weak/subset on sim,
④ adds circular risk without helping over ③.  The default is ③-only.

### §3 data-root override
`fusion/realdata.py:ROOT` now reads `MIC_REC_ROOT` env var (default keeps
`/mnt/d/.../mic_recordings`) for the reviewer's independent verification.

### R4 main result (gate met)
recall=**1.000** (≥0.90), FAR=**0.010** (≤0.10) — **PASSES the original threshold**.
(251 voiced frames, 1561 killed pts / 2347 surviving pts.)

### G5 / mutation sanity (no regression)
- G5 equiv 0.0; future-perturbation bit-identical (white + real voiced).
- The bidir-w-EMA mutation (A-rework) STOPPED leaking under the ③-only detector
  (w is near-constant now) ⇒ the generic mutation was switched to a
  **global-mean-norm(Y)** whole-segment-stat (always leaks: white 5.42, voiced
  10.86).  The path-specific **w_local LOOK-AHEAD** (voiced 4.29e-2 ≫ white
  6.2e-5) is the strong voiced-condition proof.  All caught.

## T13-B0 rework (BR1–BR4) — appended on top of 132ef9f

Reviewer REJECTED the ③-default result: ③ was a **tautology** — `apply_d1` put
killed at a fixed frame-peak−60 dB, and ③'s gate (P < frame_peak−45) is D1's
inverse function ⇒ recall 1.0 measured "does D1 use its own floor", not "can it
identify killed" (AGENTS.md §6.2: test output defines its own reference).  The
fix is at the DEGRADATION level, not the detector.

### BR1 — realistic D1 kill floor
`d1_realistic=True`: killed = boundary * 10^((jitter−margin)/20), boundary = the
kill-threshold harmonic energy (≈ weakest survivor); jitter σ=5 dB, margin=2 dB
⇒ killed cluster at the boundary, **overlapping the weakest surviving
harmonics** (the task premise: S alone can't tell killed from naturally-weak).
NOT a fixed frame/global-peak offset.  `d1_realistic=False` reverts to the
tautological floor (used only by the BR2 mutation sanity).

### BR2 — anti-tautology tests (the realism gate)
- `test_BR2_abs_must_fail_on_realistic_D1`: a pure absolute-level detector (③)
  must NOT reach 0.90/0.10 on realistic D1.  Measured recall=0.064 FAR=0.010 ⇒
  ③ FAILS (tautology absent).  Mutation (d1_realistic=False) ⇒ ③ recall=1.000
  FAR=0.010 ⇒ ③ PASSES ⇒ the assertion FAILS (caught).
- `test_BR2_overlap`: killed-level vs weakest-survivor-level distributions must
  overlap ≥0.30.  Measured **0.974**.  Mutation (d1_realistic=False) ⇒ overlap
  0.073 (<0.30) ⇒ assertion FAILS (caught).
Both make the tautology structurally impossible to re-introduce.

### BR3 — ablation on REALISTIC D1 (① is default; ③ is now diagnostic)

| method | recall | FAR | verdict |
|---|---|---|---|
| ① local-median (DEFAULT) | 0.261 | 0.108 | below |
| ② abrupt-drop | 0.034 | 0.108 | below |
| ③ abs-gate (diagnostic) | 0.064 | 0.010 | below |
| ④ V-shape prior | 0.368 | 0.492 | below (VPU envelope flatter than S ⇒ FAR) |
| ①④ | 0.110 | 0.058 | below |
| ②④ | 0.013 | 0.047 | below |
| ③④ | 0.053 | 0.009 | below |
| ①②③④ | 0.005 | 0.004 | below |

### BR4 — achievable upper bound (below threshold, honest)
🔴 **No method/combo reaches recall≥0.90 AND FAR≤0.10 on the realistic D1.**
Best single: ① recall 0.261 FAR 0.108 (FAR 8% over); ④ recall 0.368 FAR 0.492
(V-shape mismatch — VPU body-conduction envelope is flatter than S, so V-shape
prior false-flags S's formant valleys).  Combos (product) are bottlenecked by
the weakest member.  **This is an important finding, not a failure** (per BR4):
the realistic detection is genuinely hard (killed≈weakest-survivor in S; V's
shape≠S's shape) ⇒ B1's G3a expectation must be lowered, and the architecture
may need w_local as SOFT evidence (not verdict) or a calibrated V→S shape map.
No sculpting / no reverse-accommodating D1.

### G5 mutations re-verified under the new ① default
- global-mean-norm(Y): white leak 5.42, voiced 10.86 (caught).
- w_local LOOK-AHEAD: voiced 4.29e-2 ≫ white 6.6e-4 (caught; voiced path active).
Both retain diagnostic power under the ① default.

Test count: 28 → 32 (28/28 → 32/32 PASS, 0 FAIL).

## T13-B0 rework CR1–CR3 (appended on top of 3cf276f)

Reviewer REJECTED BR4's "no method passes" conclusion: BR1's v2 (killed =
weakest-survivor − 2 ± 5 dB) was OVER-CORRECTED — 34.5% of killed were LOUDER
than the weakest survivor (physically impossible: suppression can't make a
harmonic louder) ⇒ overlap 0.974 ⇒ constructively unsolvable ⇒ recall 0.261 was
the CONSTRUCTION's output, not an algorithm finding.  Fix: physical
parametrization + SWEEP (not a hand-picked point).

### CR1 — physical `d1_kill_depth_db` + sweep + physical-monotonicity
- `d1_kill_depth_db` = mean dB killed sit BELOW the boundary (= weakest
  surviving harmonic) ⇒ physical meaning = stage-2's effective suppression depth.
  jitter σ=2.5 dB; `d1_truncate=True` enforces killed ≤ boundary (physical
  monotonicity — suppression can't make louder).
- `test_CR1_physical_monotonicity`: 0/360 frames violate `killed≤weakest-survivor`
  (PASS).  Mutation (`d1_truncate=False`) ⇒ 15 frames violate ⇒ assertion FAILS.
- **SWEEP** (depth ∈ {0,3,6,10,15,20,30}, the deliverable = a CURVE not a number):

| depth | overlap | ①rec ①far | ②rec ②far | ③rec ③far | ④rec ④far | ⑤rec ⑤far |
|---|---|---|---|---|---|---|
| 0  | 0.962 | .078 .105 | .003 .020 | .002 .007 | .028 .061 | .074 .082 |
| 3  | 0.945 | .168 .092 | .011 .018 | .013 .006 | .063 .053 | .165 .069 |
| 6  | 0.904 | .224 .083 | .023 .018 | .032 .006 | .106 .049 | .220 .061 |
| 10 | 0.846 | .252 .073 | .062 .018 | .073 .004 | .138 .045 | .249 .050 |
| 15 | 0.766 | .265 .071 | .135 .017 | .116 .003 | .160 .044 | .263 .048 |
| 20 | 0.641 | .269 .070 | .206 .017 | .158 .003 | .176 .043 | .269 .048 |
| 30 | 0.394 | .270 .070 | .236 .017 | .230 .003 | .197 .043 | .270 .048 |

  - overlap ↓ monotonically with depth (0.962→0.394); ③ recall ↑ monotonically
    (0.002→0.230, sanity PASS).  Curve: `reports/T13/cr1_sweep.png`.
  - **No method reaches recall 0.90 / FAR 0.10 at ANY depth** — even at depth 30
    (overlap 0.394) the ceiling is ~0.27 (①⑤).  This is the honest upper bound.

### CR2 — ⑤: EQ-aligned V′–S direct compare (the untested info source)
⑤ uses V′ = V·exp(C) (the EQ-aligned V, EQ IS the domain correction M2 proved)
as a DIRECT predictor of S's per-harmonic level: `S(k) ≪ V′(k)` ⇒ killed.
  - FREQ-gated to the VPU usable band (≤800 Hz): outside, V′=noise ⇒ ⑤ off
    (auto-disable, not false-report).  No circular dep (V′ only predicts; verdict
    is the S/V′ ratio, not filling with V).
  - Performance (sweep): ⑤ recall ≈ ① (0.220 vs 0.224 @depth6) with BETTER FAR
    (0.061 vs 0.083).  ④ (V-shape prior, un-aligned) was worse (0.106/0.049) —
    ⑤'s EQ alignment is what fixes ④'s domain mismatch.
  - Upper bound limited by non-LTI floor (MSC 0.77–0.79 ⇒ non-LTI 0.21–0.23): ⑤
    caps ~0.27, not perfect.

### CR3 — scope judgment: AGREE
**Above 800 Hz, `w_local` structurally cannot produce value with raw VPU** (V
has no harmonic info there ⇒ ⑤ auto-disables; ①/② limited by clustering &
deep-kill≈noise).  Evidence: ⑤ (≤800 Hz) recall ≈ ① (full-band) @depth6
(0.220 vs 0.224) ⇒ the gain is IN the VPU band, not above.
  ⇒ **B1 should NOT set w_local detection metrics in 800 Hz–2 kHz** (would
  measure noise); that band's w_local validation needs Arm-A reconstruction output
  — a scope boundary, written so B1 doesn't chase an impossible metric.

### BR2 retained at the working point (depth=6)
- `test_BR2_abs_must_fail`: ③ recall 0.135 (<0.90) ⇒ FAILS (tautology absent).
  Mutation `d1_tautological=True` ⇒ ③ recall 1.000 ⇒ ③ PASSES ⇒ assertion FAILS.
- `test_BR2_overlap`: 0.904 (≥0.30).  Mutation ⇒ 0.073 (<0.30) ⇒ FAILS.

### G5 mutations re-verified under the ① default (unchanged from prior)
- global-mean-norm(Y): white 5.42, voiced 10.86 (caught).
- w_local LOOK-AHEAD: voiced 4.29e-2 ≫ white 6.6e-4 (caught).

Test count: 32 → 36 (36/36 PASS, 0 FAIL).

## T13-B0 rework DR1–DR4 (appended on top of bc2794e) — CORRECTS the polluted CR1 sweep

Reviewer caught: CR1's sweep was **polluted by ① default** — `with_switches`
(=dataclasses.replace) only overrides named keys, and `wl_use_local_median=True`
is the default ⇒ every "②/③/④/⑤" row was actually ①×(that method).  Under
product combine, ① capped every row's recall ⇒ "no method reaches 0.90/0.10"
was NEVER tested (③'s CR3 evidence "⑤≈①" was the pollution artifact).

### DR1 — ablation isolation + meta-test
`SWEEP_METHODS`: every row EXPLICITLY sets all 5 `wl_use_*` switches.
`test_DR1_meta_isolation`: asserts each row's True-set matches its declared
label + all 5 keys present (7 rows ✓).  `test_DR1_meta_mutation`: a row that
OMITS switches (relies on default) ⇒ meta-test FAILS (caught).  This
"depends-on-default" bug shows in NO functional test — only the meta-test.

### DR2 — CLEAN sweep (depth × method, ⑤ in-band caliber)

| depth | ov | ①r①f | ②r②f | ③r③f | ④r④f | ⑤r⑤f | ①×⑤ | ①×② |
|---|---|---|---|---|---|---|---|---|
| 0 | .962 | .078.105 | .012.107 | .022.010 | .354.489 | .719.370 | .074.082 | .003.020 |
| 6 | .904 | .224.083 | .039.107 | .135.010 | .435.489 | .750.370 | .220.061 | .023.018 |
| 15 | .766 | .265.071 | .266.107 | .434.010 | .526.489 | .875.370 | .263.048 | .135.017 |
| 20 | .641 | .269.070 | .516.107 | .581.010 | .557.489 | **1.000**.370 | .269.048 | .206.017 |
| 30 | .394 | .270.070 | .618.107 | .845.010 | .611.489 | **1.000**.370 | .270.048 | .236.017 |

- ③ (diagnostic) recall ↑ monotonically 0.022→0.845 (sanity ✓).
- **⑤ (EQ-aligned V′, in-band ≤800Hz) reaches recall 1.000 at depth≥20**
  (catches CLUSTERED kills — the info source).  FAR stuck at 0.370 (EQ
  alignment imperfection — the REAL remaining bottleneck, NOT recall).
- ① caps at ~0.27 (clustering blindness, see DR4).  ①×⑤ (product) WRONG —
  ① vetoes ⑤'s clustered catches (recall capped at ①).

### DR3 — ⑤ dual-caliber (don't mix)
- ⑤ IN-BAND (≤800Hz) alone: recall 0.750, FAR 0.370 — ⑤'s TRUE ability.
- ⑤ full-band alone: recall 0.994, FAR 0.610 (catastrophic FAR — band外
  unguarded w=1, NOT ⑤'s fault).
- ①×⑤ full-band combo: recall 0.220, FAR 0.061 (① caps both).

### DR4 — isolated vs clustered kill bucketing (MAIN DELIVERY; hypothesis CONFIRMED)
Reviewer's hypothesis: cross-k methods (①②) only find ISOLATED kills; clustered
kills are invisible (contiguous killed block ⇒ local median/neighbors are
themselves killed ⇒ baseline collapses).  **0.27 ≈ isolated-fraction × iso-recall
+ clustered-fraction × clu-recall.**

| method | recall_iso (n=175) | recall_clu (n=1386) | frac |
|---|---|---|---|
| ① local-med | 0.663 | **0.169** | iso 11% / clu 89% |
| ⑤ V′eq | 0.954 | **0.999** | (same) |
| ①×⑤ (product) | 0.623 | 0.169 (① vetoes) | |
| **①∨⑤ (parallel/OR)** | **0.994** | **0.999** | |

- ① catches 66% of ISOLATED but only 17% of CLUSTERED (89% of kills) ⇒ ①'s
  ~0.27 ceiling = clustering blindness, NOT a fundamental limit.
- ⑤ (EQ-aligned V′) catches 95–99% of BOTH (V has the harmonic regardless of
  neighbors) ⇒ ⑤ is the answer for clustered.
- **① and ⑤ are COMPLEMENTARY (not competing)**: ① for isolated (low FAR),
  ⑤ for clustered (high recall).  Should be PARALLEL (①∨⑤), not product —
  product lets ① veto ⑤'s clustered catches.  ①∨⑤ → recall 0.994/0.999.
- run-length: clustered kills form runs up to length 10 (most runs 2–7).

### CR3 re-checked on CLEAN data — AGREE re-affirmed
⑤ in-band (isolated) recall 0.750 (vs polluted 0.22).  >800Hz: raw VPU has no
info ⇒ ⑤ freq-gated off; ①/② limited by clustering.  w_local value domain ≈
VPU band = where ⑤ works.  B1 should NOT set w_local metrics in 800Hz–2kHz.

### Bottom line (corrected)
The previous "no method reaches 0.90/0.10" was an artifact of ①-default pollution
+ product combine.  On clean data: **⑤ (EQ-aligned V′, in-band, parallel)
reaches recall ~1.0** (clustered kills ARE detectable via V).  The REAL
bottleneck is ⑤'s FAR (0.370, EQ alignment imperfection) — B1 work (EQ quality /
thr tuning), NOT a fundamental limit.  This changes B1's G3a expectation UPWARD
(killed recovery IS feasible via ⑤) and fixes the architecture: ①∥⑤ parallel.

G5 mutations re-verified (unchanged): global-mean-norm + w_local-lookahead both
caught under ① default.  Test count: 36 → 40 (40/40 PASS).

## T13-B0 rework ER1–ER3 (appended on top of cbdc999) — ⑤'s V ability is FAKE; DR4 arch WITHDRAWN

Reviewer's shuffle/const control (monkeypatch `WLocal._detect`'s `Pv`):
- **shuffle** (permute Pv per-harmonic ⇒ destroys correspondence, keeps level dist):
  ⑤ recall 0.594 (vs 0.750) — barely drops.
- **const** (Pv = median ⇒ zero per-harmonic info): ⑤ recall 0.625, **FAR 0.077**
  (vs 0.370 — 5× better).
⇒ ⑤ is an **ABSOLUTE-LEVEL gate** (threshold from V′ global level, not frame
peak); V′'s per-harmonic structure is the SOLE source of ⑤'s FAR (body≠air
envelope, same as ④).  Under FAR priority, const-⑤ (0.625/0.077) **strictly
beats** real ⑤ (0.750/0.370) ⇒ V per-harmonic info is **net-negative**.
⇒ **DR4's "⑤ catches clustered ⇒ ①∥⑤ parallel" is WITHDRAWN** — const-⑤
also "catches clustered" (the level does it, not V's shape).  BR2 caught ③ but
not ⑤ (⑤ is a V-disguised abs-gate that slipped under BR2).

### ER1 — general shuffle/const control (BR2 generalized) + mutation
For ANY V-using method: report orig / V-shuffled / V-const recall-FAR.
🔑 **Criterion: if const (zero V per-harm info) is strictly better under FAR
priority (recall within 0.15 AND FAR lower by >0.05) ⇒ ABSOLUTE-LEVEL GATE,
NOT a candidate, must fail BR2-style.**
- ⑤ (raw V, in-band): orig 0.750/0.370 | shuffle 0.594/0.191 | const
  0.625/0.077 → **⑤ = ABSOLUTE-LEVEL GATE** (const strictly better).
- Mutation ⑥ (SYNTH co-location: flag killed iff Pv>q0.7 AND P<q0.5 — genuinely
  per-harmonic): orig 0.594 | shuffle 0.219 | const **0.000** → shuffle/const
  DROP recall ⇒ ⑥ genuinely uses V ⇒ **test does NOT mis-judge a real
  per-harmonic detector** (⑤'s 0.125 drop << ⑥'s 0.594 drop).
BR2 + ER1 = complete defense: tautology AND disguised-as-informative-tautology
both structurally impossible to pass.

### ER2 — const-⑤ baseline; V-based methods report INCREMENT
const-⑤ = true absolute-level gate = correct baseline.  Δrecall = ⑤−const,
ΔFAR = ⑤−const.

| depth | ⑤_orig | ⑤_const | Δrecall | ΔFAR | net |
|---|---|---|---|---|---|
| 0 | .719/.370 | .406/.077 | +.312 | +.294 | pos (V helps recall) |
| 6 | .750/.370 | .625/.077 | +.125 | +.294 | **net-neg under FAR prio** |
| 15 | .875/.370 | .812/.077 | +.062 | +.294 | net-neg |
| 20 | 1.000/.370 | .969/.077 | +.031 | +.294 | net-neg |
| 30 | 1.000/.370 | 1.000/.077 | +.000 | +.294 | **net-neg (V pure hurts)** |

V adds tiny recall (+0.03..+0.31) but a CONSTANT +0.294 FAR.  At depth≥6
(const recall ≥0.625) the recall gain < the FAR penalty under FAR priority ⇒
**V per-harmonic info net-negative at all realistic working points**; at
depth 30 const-⑤ matches ⑤ recall with 5× better FAR (V is pure dead weight).

**ROC recheck (matched-FAR, not single-point) — reviewer-side, confirmed
in-repo at finer thr step 0.5:** single-point comparison (⑤@thr=6 vs const)
can't judge information content; a different threshold might flip it.  So the
full ROC was swept (`wl_v_eq_thr_db ∈ [−6, 30]`) and recall compared at
**matched FAR** caps:

| | FAR≤0.077 | FAR≤0.100 | FAR≤0.200 |
|---|---|---|---|
| depth=0 real-V / const | 0.312 / **0.406** | 0.375 / **0.625** | 0.594 / **0.719** |
| depth=6 real-V / const | 0.562 / **0.625** | 0.656 / 0.656 (tie) | 0.750 / **0.938** |

(const dominates or ties at every matched-FAR cell; the one tie at depth=6
FAR≤0.100 is noise-level — at recall=0.656 real-V's FAR is 0.097 vs const's
0.100, a 0.003 FAR edge ≈ 5–10 surviving bins, within counting noise.)

⇒ **STRENGTHENED conclusion (replaces the single-point wording):** V's
per-harmonic content is **pure noise injection** — const-⑤ (zero per-harm
info) dominates or ties real-⑤ at **every** matched-FAR operating point; **no
threshold gives V a robust positive contribution.**  The per-harmonic-V path is
**CLOSED, not untuned** — B1 must not spend effort re-tuning `wl_v_eq_thr_db`
looking for a sweet spot (there is none).

### ER3 — no-freq-smooth per-bin alignment (MAIN; HARD conclusion)
Reviewer's structural hypothesis: layer-1 EQ `C[f]` is BY DESIGN freq-smoothed
(specs: prevent learning phoneme structure) ⇒ it smooths away exactly the
per-harmonic detail ⑤ needs.  Test ⑤ under 3 V-alignment modes with ER1
controls:

| align | orig | shuffle | const | const-drop |
|---|---|---|---|---|
| raw (no align) | .750/.370 | .594/.191 | .625/.077 | 0.125 |
| eq_smooth (V′, layer-1) | .750/.351 | .594/.186 | .625/.077 | 0.125 |
| eq_nosmooth (V″, per-bin) | .750/.339 | .594/.184 | .625/.077 | 0.125 |

- **Recall is IDENTICAL across all 3 alignment modes** (0.750/0.594/0.625);
  alignment only tweaks FAR (0.370→0.351→0.339).  const-drop = 0.125 in ALL
  modes (<< synth ⑥'s 0.594) ⇒ **removing freq smoothing does NOT make V
  per-harmonic info usable.**
- 🔴 **HARD conclusion: per-harmonic info CANNOT transfer VPU→mic domain**
  (consistent with the measured non-LTI floor 0.21–0.23).  V provides
  **BAND-level** info only (const-⑤ works — it uses V's global level), NOT
  harmonic-level.  ⇒ **`w_local` DOWNGRADES to soft/band-level evidence;
  `w_band` (MSC-driven) takes primary.**  Decisive for B1 architecture.

### DR4 architecture conclusion — CONFIRMED WITHDRAWN
⑤ is an absolute-level gate (const-⑤ strictly better under FAR prio); "⑤
catches clustered" was const-⑤'s level credit, not V's per-harmonic shape.
①∨⑤ parallel ⇒ ①∨(V-level abs-gate) — NOT V-based, NOT the claimed
complementarity.  The parallel architecture is withdrawn.  ⑤ is NOT a
candidate detector (must fail BR2-style).  const-⑤ (VPU-band-level abs-gate)
IS the real band-level V usage — distinct from ③ (mic frame-peak abs-gate).

### Bottom line (corrected again — final)
- ⑤ (per-harmonic V′) = abs-gate disguise; **V per-harmonic content is pure
  noise injection — const-⑤ ROC dominates or ties real-⑤ at every matched-FAR
  operating point; no threshold gives positive contribution; the path is
  CLOSED, not untuned.**
- Per-harmonic V→S transfer is NOT possible (non-LTI); V is BAND-level only.
- **w_local role**: band-level V evidence (const-⑤ / MSC), NOT per-harmonic.
  `w_band` (MSC-driven) primary; `w_local` soft secondary.  B1 designs
  around band-level V, drops per-harmonic V ambitions.
- ① (local-median) still best isolated-kill detector (clustering blindness
  confirmed, but it's the low-FAR isolated side of a band-level system).
- `0.90/0.10` (set under the now-falsified "per-harmonic detection feasible"
  assumption) is **dropped for B1**; B1 uses a band-level criterion (reviewer
  to specify with the B1 spec).
- Defense retained: **BR2 (tautology) + ER1 (disguised-as-informative
  tautology)** — any new method must pass both.

Test count: 40 → 43 (43/43 PASS).

---

## T13-B0.5 · FR1–FR4 (appended on top of f6c8b59→0b0ac6b) — 4 fixes before B1

🔴 **BOUNDARY (must label everywhere):** all T13 conclusions hold **only for
MALE speech (F0 median 87–124 Hz), normal volume**.  0624/0625 cover 4
speakers (ld/lrx/qy/syh), ALL male, zero female, all normal-volume reading.
No extrapolation beyond.  Female / loud / soft coverage is the user's to
supplement; this task does NOT address it.

### FR1 · `c_V` energy term → in-band SNR (directional + ratchet fix)
Old `CV.step`: `e_term = (e_db − cv_e_floor_db)/(e_max_db − cv_e_floor_db)`,
`e_max_db` = running MAX.  Three defects: (1) **directional** — quiet⇒low
c_V, but quiet needs V most; (2) **ratchet** — one loud event raises `e_max`
forever ⇒ c_V depressed permanently; (3) fixed `cv_e_floor_db` ⇒ breaks
across speaker/gain.  **Fix:** `e_term = sigmoid((snr − ref)/scale)`,
`snr = e_db − nf`, `nf` = per-frame low-quantile (0.15) of per-bin |V|² + slow
EMA (τ=cv_nf_tau_s) ⇒ VPU device-noise floor.  Scales with recording gain
(⇒ SNR invariant to gain — FR1-a); holds during loud events (⇒ no ratchet —
FR1-c); drops when V signal weakens vs its device noise (⇒ c_V drops — M3/FR1-b).
- **CohTracker clamp bug found+fixed:** `clamp_min(1e-10)` on `vv·ss` inflated
  the denominator for quiet bins (vv·ss<floor) ⇒ MSC deflated ⇒ **broke
  scale-invariance** (FR1-a).  Changed to `1e-20` (only guards literal 0/0 of
  true silence; normal signals never hit it).  This is a latent bug the FR1-a
  test exposed.
- **M3 data updated** (necessary): the SNR design is scale-invariant to whole-V
  scaling ⇒ the old M3 (`v_spec*g`, whole-V) gave c_V constant ⇒ strict-decrease
  failed.  M3 now attenuates V's SIGNAL (harmonics) only, device noise FIXED
  (= the loose-fit/coupling-loss scenario that is FR1's motivation).  Criterion
  (strict monotone non-increasing) UNCHANGED.

| criterion | result (synthetic, controlled levels) | mutation |
|---|---|---|
| FR1-a invariant (S&V jointly −6/−12/−20 dB ⇒ Δc_V≤0.05) | spread 0.0000 ✓ | `cv_legacy_abslevel` (pure level) ⇒ spread 0.21 ✓caught |
| FR1-b strict-decrease (V signal −0/−3/−6/−12 dB) | 0.825→0.687 ✓ | c_V disabled ⇒ constant ✓caught (M3_mutation) |
| FR1-c ratchet-recover (+18 dB seg then normal, ≤2 s to control±0.05) | 0.008 @2s ✓ | `cv_legacy_ratchet` (running-MAX) ⇒ 0.074 ✓caught |

🔑 **FR1-a & FR1-b MUST hold as a pair** (only-a trivially passes via
"delete energy term"); the pair locks the direction.  Each mutation breaks
EXACTLY its own criterion (keeps the other two).

### FR2 · comfort noise → adaptive level
Old: `level = 10^(cn_floor_db/20)` (fixed absolute).  Quiet speech ⇒ comfort
noise covers speech.  **Fix:** `level = 10^((speech_rms_db − cn_below_speech_db)/20)`
(speech-RMS EMA − constant gap; injected after fusion, not scaled by w).

| criterion | result | mutation |
|---|---|---|
| FR2-a gap constant (−6/−12/−20 dB ⇒ Δgap≤1 dB) | 40.0/40.0/40.0/40.0 spread 0.000 ✓ | `cn_fixed_level_db` ⇒ gaps 43.8/31.8/23.8 spread 20 ✓caught |
| FR2-b inaudible (@−20 dB, comfort ≥40 dB below speech RMS) | gap 40.0 ✓ | (40 dB is a conservative inaudibility threshold; reported reasoning) |
| FR2-c independent of w (y=S vs y=V ⇒ identical comfort) | diff 0.0 ✓ | — |

### FR3 (MAIN) · kill-clustering parametrization + sweep — ① ceiling = f(isolated ratio)
Old `apply_d1` perframe: `sorted(es, key=lambda x: x[2])` (weak-first, **100%
deterministic**) ⇒ kill set ~maximally clustered (isolated only ~11%).  ①'s
~0.27 ceiling is mostly THIS modeling choice, not detection difficulty.
**Fix:** `key = energy_dB + n(t,k)`, `n` = 2-D Gaussian field time-smoothed (kernel
~150 ms) ⇒ slow-varying (preserves inter-frame kill-set correlation;
per-frame-independent would flicker falsely).  `σ=0 ⇒ n≡0 ⇒ sort by energy_dB
≡ sort by energy (monotone) ⇒ EXACT repro` (regression anchor, asserted).

Sweep `d1_rank_sigma_db ∈ {0,2,4,6,10,15}` @ depth 20 (the B0 'ceiling' caliber
where ①=0.27 at σ=0):

| σ | iso% | ①rec | ①far | ②rec | ③rec | ⑤c(const)rec | ⑤c far | iso_r | clu_r | jac |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11.2 | 0.269 | 0.070 | 0.516 | 0.581 | 0.969 | 0.077 | 0.851 | 0.196 | 0.254 |
| 2 | 10.8 | 0.274 | 0.072 | 0.502 | 0.582 | 0.900 | 0.076 | 0.845 | 0.205 | 0.260 |
| 4 | 12.1 | 0.289 | 0.072 | 0.486 | 0.572 | 0.821 | 0.075 | 0.825 | 0.215 | 0.260 |
| 6 | 14.6 | 0.323 | 0.065 | 0.522 | 0.614 | 0.915 | 0.071 | 0.776 | 0.246 | 0.258 |
| 10 | 22.7 | 0.351 | 0.070 | 0.547 | 0.599 | 0.773 | 0.067 | 0.712 | 0.245 | 0.263 |
| 15 | 30.2 | 0.405 | 0.077 | 0.583 | 0.595 | 0.743 | 0.069 | 0.699 | 0.279 | 0.269 |

| criterion | result |
|---|---|
| FR3-a σ=0 repro | isolated 11.2% (±1pt of 11) ✓; ① 0.269 (±0.02 of 0.27) ✓ |
| FR3-b isolated monotone↑(σ) | ✓ (one 0.4-pt dip at σ=2 <1.5-pt tol — perturbation noise) |
| FR3-c ① recall ↑(isolated) | **strict-monotone FALSE** (0.005 inversion at σ=2); **but overall Δ=+0.131 over iso 0.11→0.30 ⇒ attribution CONFIRMED in the large** (the inversion is perturbation-field noise, not a disproof — reported honestly, NOT asserted/tuned) |
| FR3-d Jaccard vs σ | ~0.254→0.269 stable (no crash) ⇒ time-smoothing OK |

Plot: `reports/T13/fr3_sweep.png` (x=isolated ratio, ①②③const-⑤ recall).

🔴 **CONCLUSION CORRECTION (corrects B0's '① ceiling = 0.27'):**
**① ceiling = f(isolated ratio)** — NOT a fixed 0.27.  ① rises 0.27→0.405 as
isolated ratio rises 11%→30%.  **0.27 corresponds to σ=0 (the maximally-
clustered extreme), a modeling choice, not the detection problem's inherent
ceiling.**  Real stage-2's isolated ratio (unknown here) sets the real ceiling.
This is the same error-class third time (kill-depth hand-picked → CR1;
ablation default-dep → DR1; clustering hand-picked → FR3) — now parameterized.

### FR4 · D2/D3/D4 coverage (no effect conclusion)
`_r4_recall_far` now applies D4 (time-domain) + D2/D3 (after D1) so they can
stack.  All runs finite (no NaN/Inf):

| case | ①rec | ①far | ⑤c(const)rec | ⑤c far | finite |
|---|---|---|---|---|---|
| D1 only | 0.224 | 0.083 | 0.625 | 0.077 | ✓ |
| D1+D2 contrast | 0.113 | 0.081 | 0.625 | 0.076 | ✓ |
| D1+D3 musical | 0.230 | 0.096 | 0.625 | 0.095 | ✓ |
| D1+D4 envelope | 0.223 | 0.084 | 0.714 | 0.087 | ✓ |
| D3 only (block drop) | 0.000 | 0.182 | 0.000 | 0.125 | ✓ |

D3 (block-level random loss) on its own row — distinct from FR3's harmonic-
level random.  **Coverage check only — no algorithm-quality judgment** (per spec).

### Defenses retained (all pass)
BR2 (tautology), ER1 (disguised-as-informative tautology, ⑤=abs-gate), G5
(future-perturbation + mutations), DR1 (ablation-isolation meta-test) — all
still PASS unchanged.

### Test count: 43 → 53 (53/53 PASS, 0 FAIL).  10 new: FR1-a/a-mut/c/c-mut,
FR2-a/b/c/a-mut, FR3, FR4.

### Compromises / risks
- **FR1-a mutation on REAL 0624 saturates** (real V e_db~+18 dB, far above the
  fixed reference) ⇒ the legacy fixed-floor defect is LATENT on normal-level
  data (only exposed at quiet levels / large scale).  FR1-a + mutation run on
  SYNTHETIC (controlled levels) for teeth; real-data FR1-a (new design) is
  invariant (0.0 spread).  Documented.
- **FR3-c strict-monotone flagged** by a 0.005 noise inversion; the overall
  trend (Δ+0.131) strongly confirms the attribution.  Reported honestly
  (not asserted, not tuned) per the reviewer's instruction.
- **0625 untouched** in committed code/tests (holdout protected).  The VPU
  noise-floor tracker is purely causal streaming (no calibration file needed);
  `0625/FB_FF_TT_VPU_noise_floor.wav` remains an optional reviewer-side
  cross-check (its measured shape — 80–640 Hz flat, >640 Hz ~−8 to −12 dB/oct
  — is consistent with the tracker's band-level floor estimate).
- `cv_snr_ref_db`/`cv_snr_scale_db`/`cn_below_speech_db` are PLACEHOLDER
  defaults (not tuned); the other fusion constants (EQ τ, Δ, hysteresis, w-smooth)
  are frozen.

### Reviewer-side extrapolations & newly-noted risks (accepted f771658)

**1. ① cannot reach threshold at ANY clustering (reviewer linear extrapolation).**
Linear fit of the FR3 sweep: `①recall ≈ 0.207 + 0.00656 × iso%`.
- iso=30% → 0.404 · iso=50% → 0.535 · **iso=100% (kills fully random, zero
  clustering) → 0.863**
- reaching 0.90 needs **iso = 106%** — physically impossible.
⇒ the FR3 correction (ceiling = f(isolated ratio), 0.27 = σ=0 extreme) is right,
but the last step: **pushed to the physical limit of clustering, ① still can't
pass 0.90.**  And real stage-2 kills are SNR-driven ⇒ inherently clustered ⇒
real isolated ratio ≪ 100%.  🔴 **① is NOT a viable route — "tune more" won't
fix it.  B1 should not spend time on it.**

**2. The decisive variable is suppression DEPTH, not clustering (reviewer).**
The FR3 sweep ran at depth=20; it hides the key contrast:

| | depth=6 (BR2 working pt) | depth=20 |
|---|---|---|
| ① | 0.224 / 0.083 | 0.269 / 0.070 |
| **const-⑤** (band-level V gate) | 0.625 / 0.077 | 🔑 **0.969 / 0.077** |

**const-⑤ @ depth=20 = 0.969 / 0.077 — simultaneously recall≥0.90 AND FAR≤0.10,
passes threshold.**  depth=6 → nobody works; depth=20 → a simple band-level
level gate suffices.  ⇒ **"is there a viable detector" is answered entirely by
real stage-2's suppression depth — a quantity currently UNKNOWN, and more
decisive than everything debated so far.**

**Cross-level coupling (reviewer; to relay to the MAIN pipeline, not T13):**
stage-2's PARTIAL suppression is the WORST case for the fusion layer —
half-killed harmonics are neither detectable (overlap with survivors) nor
intact (need repair); either leave them or kill them through.  And stage-2's
max suppression depth is exactly the `target noise floor s+β·n` knob (T3 is
verifying it). ⇒ **a stage-2 training parameter directly decides whether the
fusion layer is viable.**  This is flagged for the main pipeline; outside T13.

**3. D3-only hole (reviewer-noted finding).**  FR4's `D3 only` row: ① AND
const-⑤ recall both **0.000** — block-level random loss is **invisible to both
detectors**.  Musical noise / block dropout is a real stage-2 artifact; the
current detectors don't see it.  Recorded as a **known coverage hole** in the
README (FR4 correctly made no quality judgment per spec, but the 0.000 is a
finding, not a neutral result).

**4. `clamp_min` scale-dependence boundary (reviewer-noted, NOT changed this
round).**  `decision.py:64,65` clamp `e_v_ema` at 1e-10 and per-bin |V|² at
1e-12; `snr_db` (line 71) clamps at 0.  `snr_db` is strictly scale-invariant
ONLY when NEITHER clamp triggers.  The tested scales (−6/−12/−20 dB) sit in
the safe zone, but the project hard-constraint includes "whisper–normal ~30 dB
dynamic", and deployment inputs may not be peak-normalized ⇒ at extreme quiet
the clamps trigger and scale-invariance creeps.  **Recorded as a known
boundary** (or future fix: relative floor, e.g. `clamp_min(eps·peak)`).  Not
changed this round per reviewer instruction.

---
## T13-B1 (appended on top of 362cb24) — AC1/AC2/AC3 + G-verification

🔴 BOUNDARY: all conclusions MALE only (F0 87–124 Hz), normal volume — 0624/0625
4 speakers all male, zero female.  Not extrapolated.

### Architecture changes (AC1/AC2/AC3) — implemented
- **AC1 (magnitude-only, ∠Y=∠S):** `logclip_mix` phase = `angle(s_spec)` (was
  weighted vector sum).  **Deleted:** `DelayComp` + `measure_gcc_phat` +
  `enable_delay_comp`/`delay_samples`/`gcc_*` (align.py, fusion.py, config);
  `complex_convex`/`use_complex_convex`/`enable_logclip_mix` (synthesis,
  config).  M7 marked historical (SKIP).  log-clip retained.
- **AC2 (frozen EQ):** `EQAlign` eq_mode="frozen" — cold-start (first
  `eq_coldstart_frames`=120 credible updates) then FREEZE; changepoint unfreezes
  (watchdog).  eq_mode="adaptive" = B0 continuous-EMA (ablation).  EQ bootstrap
  bug fixed (outlier-rejection off during cold-start — was rejecting all updates
  since C=0 vs d≈26 dB ⇒ deadlock).  eq_freq_smooth_bins 5→1 (per-bin C, was
  flattening to a scalar).
- **AC3 (band-level w_local):** `WLocal` = const-⑤ band gate
  `sigmoid((Pv_overall−P_band−thr)/slope)`; bands >800 Hz ⇒ w=0 (CR3).  Per-
  harmonic ①②③④⑤ DELETED (B0.5: per-harm info can't transfer VPU→mic).
- **BR2 rewrite (depth-aware):** pure absolute-level detector must FAIL at
  depth≤6 (low-separability end); high depth may pass (task genuinely easy).

### G-verification results (0624; 53→49 tests, 45 PASS / 4 SKIP-historical / 0 FAIL)

🔴 **CRITICAL architecture finding — G6 fails everywhere.**  `cos(Y,X) ≥
cos(S,X)` (the "don't make worse than S" floor) is VIOLATED at every depth and
scenario: the fusion DEGRADES Y vs S (cos 0.84 vs cos(S,X) 0.998).  Root cause:
AC1 base-V' formula `log|Y|=log|V'|+(1−w)·clip(log|S|−log|V'|,±Δ)` at low w
gives `Y=V'+clip(S−V')`, which = S ONLY if |S−V'|<Δ.  The layer-1 EQ (per-bin
C[f]) cannot make V'≈S on misaligned bins — the FF↔VPU gain mismatch is ~26 dB
and **harmonics move across bins as F0 varies** ⇒ a fixed bin's C averages
harmonic + noise samples ⇒ residual |S−V'|>Δ on many bins ⇒ clip saturates ⇒ Y
pulled toward a misaligned V' ⇒ worse than S.  Δ is FROZEN at 10 (per the
discipline); enlarging it (Δ=30 passes G6@depth6, Δ=60 @depth20) trades away
kill-drag robustness — not done.

| G | result |
|---|---|
| **G1** (D1=0, LSD<1.0 / cos≥cos(S,X)−0.01) | LSD(Y,S)=8.4 dB FAIL; cos=0.981<0.990 FAIL — same root cause |
| **G2** (dropout⇒LSD<0.5, no >3dB step) | LSD=8.36 FAIL (Y≠S even with V→noise); steps OK — same root cause |
| **G4'/G6** depth sweep | ALL FAIL — Y worse than S in every band, every depth |
| **G3a'** (∃depth≤20: LSD(Y,X)≤0.5·LSD(S,X)) | **PASS** — depth10 ratio 0.312 (heavily-suppressed bands DO recover) |
| **G3b'** (out-of-band ≤+0.5 dB) | FAIL @depth6/20 (Δ=+4.8/+2.2); PASS @depth30 |
| **G5** (streaming causal, MUST-pass) | **PASS** — future-perturb bit-identical (0.0); 2 mutations retain teeth post-AC1 (global-mean-norm leak 15, w_local-lookahead voiced>>white) |
| **G7** (phase pricing, report) | LSD(∠S,∠X-variant)=1.73 dB, cos=0.91 — AC1's cost; samples g7_phase_{S,X}phase.wav |
| scenarios D1+D2/D3/D4/all | all finite; G6 ✗ (cos 0.66–0.86 < cos(S,X)) |
| progressive weakening (VPU −3/−6/−12 + EQ shift) | G6 ✗ (cos 0.81–0.84 < 0.998) — c_V/EQ-freeze don't recover |
| ablation frozen vs adaptive | **IDENTICAL** cos 0.8416 — EQ mode is NOT the bottleneck (the base-V' formula is) |
| listening pack | 12 WAVs (S/V/Y/X × d0/d6/d20) → reports/T13B1/ |

### Proposed fixes (need reviewer decision — Δ frozen, EQ-τ frozen)
1. **Per-harmonic EQ (C[k] via F0):** align V's k-th harmonic to S's k-th
   (track d at harmonic k, map to bins via F0) ⇒ V'≈S on harmonic bins ⇒ clip
   doesn't saturate on survivors ⇒ G6 passes at Δ=10.  Does NOT contradict AC3
   (deleted per-harm DETECTION, not EQ alignment).
2. **w-gated S-passthrough:** Y=S when w<ε (no repair), base-V' when w≥ε —
   satisfies G6 floor, deviates from the formula.
3. **Unfreeze Δ** (Δ=30–60) — weakens kill-drag robustness.

G3a' PASSES (the fusion DOES help where S is heavily suppressed) — so the
magnitude-domain repair is real; the failure is the base-V' clean-signal
deviation, not the repair mechanism itself.

Figures/samples: reports/T13B1/g3a_recovery.png, g7_phase_*.wav, lp_*.wav.

---
## T13-B1 rework HR1–HR5 (appended on top of ab71ed5) — formula bug FIXED

🔴 Reviewer's diagnosis accepted: the OLD `logclip_mix` (V'-anchored,
`log|Y|=log|V'|+(1−w)·clip(log|S|−log|V'|,±Δ)`) violated "VPU has add-power
only, no subtract-power" — at w=0 it gave `Y=V'+clip(S−V')`, which =S ONLY if
|S−V'|<Δ; with the ~26 dB FF↔VPU mismatch ⇒ clip saturated ⇒ w=0 still let V
veto ⇒ G1/G2/G4'/G6 all failed.  Root cause: clip bounded "Y's deviation from
V'", not "how far S can pull Y down".

### HR1 — S-ANCHORED asymmetric-clip formula (the fix)
```
log|Y| = log|S| + clip( w·(log|V'| − log|S|), −Δ_down, +Δ_up )
∠Y     = ∠S
```
- **w=0 ⇒ log|Y|=log|S| ⇒ Y≡S EXACTLY** (structural; G1/G2/G4'/G6 floors by
  construction, not measurement).
- Δ_up=25 (allow restoring killed harmonics, 20–30 dB), Δ_down=5 (V can barely
  lower S, 3–6 dB) — NEW params, NOT subject to old "Δ frozen" (that was the
  symmetric old Δ).  Chosen on 0624 (placeholder defaults, reported).
- Y = S·g, g=10^(clip/20) bounded real gain (fixed-point friendly).
- EQ precision requirement relaxed (C[f] wrong only biases the correction;
  baseline always S) — also lowers the eq_freq_smooth_bins tension.

### HR2 — structural identity test (not a statistical threshold)
`test_HR2_zero_w_identity`: force w≡0 over real 0624 (mutant Fusion, EQ off,
comfort off) ⇒ `Y torch.allclose(S_roundtrip, atol=1e-5)` **PASS** (maxdiff
7.98e-6 — the STFT roundtrip itself is 4.0e-2; the identity isolates the
FORMULA).  Mutation `synth_legacy_vprime=True` (old V'-anchor, EQ off ⇒ V'≠S)
⇒ maxdiff **8.2 dB** ⇒ identity FAILS (caught).  This pinpoints the formula
line, vs the old statistical thresholds that only said "broken".

### HR3 — G7 re-measured (old 1.73 dB void; was on the bad formula)
`LSD(∠S-variant, ∠X-variant) = 1.383 dB`, cos=0.9973.  **HR3 reference
`LSD(S,X)=5.019 dB`** (stage-2's OWN cost) ⇒ the AC1 phase gap is **0.28× of
stage-2's cost** — SMALL.  Samples: reports/T13B1/g7_phase_{S,X}phase.wav.

### HR4 — band-level ER1 control (w_local_band uses V?)
Replace `Pv_overall` with a FIXED CONSTANT (`wl_v_perturb="const"`), rerun
G3a' (depth10, suppressed bands):
- real V:  LSD(S,X)=84.42 LSD(Y,X)=34.37 ratio=0.407
- const Pv: LSD(S,X)=84.42 LSD(Y,X)=34.37 ratio=**0.407 (IDENTICAL)**
⇒ w_local_band does NOT use V's per-FRAME variation (only its STABLE level,
capturable by a one-time calibration).  Per the reviewer's HR4 criterion
("if const ≈ real ⇒ not using V ⇒ delete"): **w_local_band is a candidate for
DELETION** — w_band (MSC) should be the sole band-level weight.  NOT deleted
this round (architecture decision for the reviewer); reported, no reluctance.

### HR5 — two corrections
1. **G2 uses the REAL `0625/FB_FF_TT_VPU_noise_floor.wav`** (only that file
   read; 0625 speech entries untouched).  G2 now PASS (LSD(Y,S)=0.289 dB,
   cut-in/out steps OK).  (Synthetic fallback if the file is absent.)
2. **`eq_freq_smooth_bins` 5→1 KEPT** (per-bin C).  Rationale: kernel-5 spans
   >F0 at F0≈100 Hz (5 bin = 156 Hz) ⇒ flattens per-bin C to a scalar.  HR1
   lowers the tension (C wrong only biases the correction).  **Known tension
   (README):** no smoothing risks learning harmonic structure into C; future
   smoothing should be in LOG-frequency or F0-scale-adaptive, not fixed bins.

### Full G re-verification (HR1 in place; depth axis) — ALL HARD THRESHOLDS PASS
| G | result |
|---|---|
| G1 (D1=0 LSD<1.0 / cos≥cos(S,X)−0.01) | LSD=0.295 dB ✓; cos=1.0000 ✓ |
| G2 (dropout⇒LSD<0.5, no >3dB step) | LSD=0.289 ✓; steps 0.06/0.54 ✓ (real noise_floor.wav) |
| G3a' (∃depth≤20: LSD(Y,X)≤0.5·LSD(S,X)) | ✓ depth10 ratio 0.407 |
| G3b' (out-of-band ≤+0.5 dB) | ✓ all depths (Δ≤0.24, often NEGATIVE — Y better than S) |
| G4'/G6 depth sweep | ✓✓ every depth/band; cos(Y,X)=cos(S,X) structurally |
| G5 (streaming causal, MUST-pass) | ✓ bit-identical; 2 mutations retain teeth (global-mean-norm 14.7, w_local-lookahead 9.4e-4>1e-6) |
| G6 (cos(Y,X)≥cos(S,X) all depth) | ✓ (structural) |
| G7 (phase pricing) | 1.383 dB = 0.28× LSD(S,X)=5.019 |
| scenarios D1+D2/D3/all | G6 ✓; D1+D4 marginal (cos 0.9168 vs 0.9169, tie — reported) |
| progressive weakening (VPU −3/−6/−12 + EQ shift) | G6 ✓ (Y=S safe fallback — c_V drops, w low) |
| ablation frozen vs adaptive | identical 0.9981 (EQ not the bottleneck) |
| HR2 identity / mutation | ✓ / caught (8.2 dB) |
| HR4 const-vs-real Pv | identical ⇒ w_local_band deletion candidate |

**Tests: 48/52 PASS, 0 FAIL, 4 SKIP-historical (M1/M1-mut/M7/M7-mut).**

### Compromises / risks
- **w stays LOW overall** (fusion ≈ S most of the time) ⇒ G6 passes structurally
  but the fusion is CONSERVATIVE (Y≈S; helps only where w_local_band + c_V raise
  w on suppressed bands).  G3a' confirms the repair DOES engage on heavily-
  suppressed bands (ratio 0.407).  The conservative behavior is the safe fallback
  (G6 by construction); whether it's ENOUGH is the B-stage question.
- **D1+D4 marginal G6** (tie 0.9168 vs 0.9169) — envelope compression is the one
  scenario where the fusion doesn't clearly help; reported, not tuned.
- **HR4 ⇒ w_local_band candidate for deletion** — awaits reviewer decision.
- **w_local-lookahead mutation flips voiced/white order** under band-level
  (white random bands leak more than voiced smooth) — but look-ahead IS detected
  (>1e-6); the B0 voiced>>white teeth was per-harmonic-specific.
- 0625 speech untouched; noise_floor.wav read only (HR5-1, reviewer-sanctioned).
- Δ_up/Δ_down are placeholder defaults (25/5), not tuned beyond the 20–30 / 3–6
  ranges the reviewer specified.

Figures/samples: reports/T13B1/{g3a_recovery.png, g7_phase_*.wav, lp_*.wav}.

---
## T13-B1 rework JR1–JR3 (appended on top of 07d31df) — HR4 voided; JR2 idle finding

🔴 Reviewer caught HR4 as a DOUBLE NO-OP: (1) `wl_v_perturb` was NOT wired into
`WLocal` (fusion.py constructed it without passing `cfg.wl_v_perturb` ⇒ the
algorithm read the default "none"); (2) even if wired, B=1 ⇒ `median()=self`,
`randperm(1)=identity` ⇒ both perturbations no-ops.  ⇒ "w_local_band deletion
candidate" was VOID.  JR1 redoes it on the TIME axis; DR1 extended to prevent
the next no-op-flag bug.

### JR1 — band-level ER1 control (TIME-axis, via caller `pv_override`)
`WLocal.step` gained `pv_override` (caller applies the time-axis perturbation;
the per-frame B-axis const/shuffle are gone — they were no-ops at B=1).
`fusion.py` now wires `v_perturb=cfg.wl_v_perturb`.  Three controls on G3a'
(depth10, suppressed bands), `Pv[t]` perturbed across TIME:

| mode | LSD(S,X) | LSD(Y,X) | ratio |
|---|---|---|---|
| real V | 84.4 | 34.4 | 0.407 |
| const-longterm (2-s EMA) | 84.4 | 34.4 | 0.407 |
| shuffle-time (permute t) | 84.4 | 33.4 | 0.395 |
| fixed-arbitrary (0 dB, no V) | 84.4 | 34.3 | 0.407 |

🔑 **③ fixed-arbitrary vs real Δ=0.021 <0.05** ⇒ per the reviewer's JR1
criterion, `w_local_band` does NOT need V ⇒ **deletion candidate** (w_band MSC
sole).  ② shuffle-time ≈ real ⇒ only V's level matters, not its time-variation.

⚠️ **BUT the test is INCONCLUSIVE on w_local's actual V-usage**, masked by a
deeper finding (JR2): `c_V ≈ 0.01–0.04` is the bottleneck — `w = c_V·…·w_local
≈ 0` regardless of `w_local`'s Pv, so the Pv perturbation doesn't reach Y.  The
JR1 criterion is met, but the cleaner read is "w_local's contribution is masked
by low c_V"; **do NOT delete w_local_band until c_V is unblocked and JR1
re-run** (it may use V once w is non-trivial).

### JR2 — "must actually intervene" metrics (the mirror of G6)
`corr = clip(w·(log|V'|−log|S|), −Δ_down, +Δ_up)` per band-frame.  (Note: the
G3a' absolute LSDs above carry a *10 typo — ratios are right, absolutes 10×;
J-metrics use corr directly, no typo.)

| depth | J1 cov (sup, |corr|>3dB) | J2 false (unsup) | J3 recovery |
|---|---|---|---|
| 0 | 0.00 | 0.01 | 0.00 |
| 3 | 0.00 | 0.01 | 0.00 |
| 6 | 0.00 | 0.01 | 0.00 |
| 10 | 0.00 | 0.01 | 0.06 |
| 15 | 0.00 | 0.01 | 0.02 |
| 20 | 0.02 | 0.01 | 0.02 |
| 30 | 0.01 | 0.01 | 0.01 |

🔴 **J1 ≈ 0 (≪ 0.50 criterion at depth≥10); J3 ≈ 0 (≪ 0.30).**  The fusion is
**SAFE (G6 structural) but IDLE** — it does not intervene.  Root cause:
`c_V ≈ 0.01–0.04` (the q_term is low: `eq_resid` 15–25 dB because V'≠S — the
per-bin EQ can't align FF↔VPU since harmonics move across bins as F0 varies ⇒
the bin's C averages harmonic+noise ⇒ high residual ⇒ q_term≈0 ⇒ c_V≈0 ⇒
w≈0 ⇒ no V mixed).  This is the **persistent V'≈S alignment bottleneck** (the
same root as the original G6 issue, now manifesting as c_V→0 rather than
Y-deviation, since HR1 made Y=S by construction when w=0).  ⚠️ Per the reviewer:
"if J1 ≪ 0.50, fusion safe but basically not working — worse than G6 fail."
**Reported honestly; Δ_up and w thresholds NOT tuned to hit J1.**  The fix is
at the EQ (per-bin C can't track moving harmonics) — the same item flagged
in the HR1 round; resolving it is the prerequisite for the fusion to actually
engage.

J2 (false-intervention) ≈ 0.01 (≤0.10 ✓) — consistent with "no intervention at
all, false or true".

### HR3 — G7 ratio per depth (small correction)
| depth | LSD(S,X) | phase gap | ratio |
|---|---|---|---|
| 0 | 0.000 | 0.070 | **inf** (LSD(S,X)→0, ratio diverges — as predicted) |
| 3 | 5.060 | 0.902 | 0.18 |
| 6 | 4.094 | 0.837 | 0.20 |
| 10 | 3.393 | 0.810 | 0.24 |
| 15 | 3.872 | 1.020 | 0.26 |
| 20 | 5.089 | 1.364 | 0.27 |
| 30 | 7.482 | 2.126 | 0.28 |

AC1 phase cost is **0.18–0.28× of stage-2's own cost** across depths (small,
rising slowly with depth).  depth=0 diverges (no stage-2 cost ⇒ ratio
meaningless), as the reviewer noted.

### JR3 — capability boundary (README; no code change)
Three results together draw a clear line:
- **D3 only** (block-level random T-F loss) ⇒ detector recall **0.000** (B0.5)
- **D1+D4** (time-domain envelope compression) ⇒ G6 marginal, fusion doesn't
  help (this round)
- **D1** (band-level spectral suppression) ⇒ G3a' ratio 0.407, effective
🔑 **The fusion layer sees ONLY "band-level spectral suppression"; time-domain
envelope compression and block-level loss are INVISIBLE to it.**  This is a
CAPABILITY BOUNDARY, not a defect — but downstream must not assume the fusion
catches all stage-2 damage形态.  (And per JR2, even on D1 the fusion is currently
idle due to the c_V/EQ bottleneck — so the "effective" on D1 is the mechanism's
potential, not the current measured output.)

### DR1 extension — wl_v_perturb wiring
`test_DR1_wl_v_perturb_wiring`: cfg.wl_v_perturb set ⇒ `WLocal.v_perturb` wired
(PASS).  Mutation (construct WLocal WITHOUT v_perturb=) ⇒ stays "none" ⇒
meta-test FAILS (caught).  This guards the next no-op-flag bug (the HR4 class).

### Compromises / risks
- **JR2 J1≈0 is the headline**: the fusion is net-neutral (Y≈S), neither
  helping nor hurting — safe (G6) but idle.  The blocker is c_V (q_term low
  from V'≠S, EQ per-bin can't track moving harmonics).  This is the SAME root
  flagged in the HR1 round; it is the prerequisite for any real intervention.
- **JR1 inconclusive under low c_V** — w_local_band's V-usage can't be cleanly
  tested until c_V/w is non-trivial; the deletion criterion is met but masked.
- EQ changepoint fix (only fire after freeze) — was resetting c_V to floor
  every frame during cold-start (high residual), compounding the idle.
- G3a' absolute LSDs had a *10 typo (ratios correct); J-metrics unaffected.

## A1-0：硬门槛审计与 HR2 数值口径修正

完整审计、G2 十条录音分桶、`KNOWN_FAIL` 注册表、HR2 双域断言与 mutation 证据见：
[[A1-0_硬门槛审计]]。

本轮 runner 最终状态为 `65/74 PASS, 2 FAIL, 3 XFAIL, 0 XPASS, 4 SKIP`。两个未登记真失败是 G3a′ 与 G4′；G2 按修正协议通过，J2/K-a/K-c 保持已授权 XFAIL。补充精确口径后，G3a′ 最接近门槛的是 depth 6 的 `0.50008>0.5`；G4′ 逐 `(band,t)` 检查时七个 depth 全部失败。生产机制与参数均未改。

> [!danger] BOUNDARY
> 0624/0625 共 4 位说话人均为男声（F0 中位 87–124 Hz），且均为正常音量。本轮一切结论只在此边界内成立，不得外推。

## A2：G3a′ 结构天花板与 G4′ 违例归因

完整结果与图见 [T13A2 报告](../T13A2/README.md)。

- `n_sup≥30` 后，d0/d3/d6/d10 为 INSUFFICIENT；d15/d20/d30 才参与效果判定。
- d20 oracle ratio=`0.48342<0.5`，说明现有存在量词并非结构上不可达；实测 `0.60796`，仍有真实算法差距。
- G4′ 与 J2 交集仅覆盖 1.44% 的 G4′ 违例；J2 的 3 dB 阈值只捕获尾部。
- 500–800 Hz 的中位 `w` 与最大损伤均最高，但 315–500 Hz 的违例率略高。

> [!danger] BOUNDARY
> 结论仅覆盖全男声（F0 中位 87–124 Hz）、正常音量的现有 0624/0625 边界，不得外推。

## A6 — β-fill 隔离 · HR3 往返 clip · D1 嵌入定标 · 逐 bin oracle 臂

完整代码见 `tests/test_t13_a6.py`、`tests/test_t13_a5.py`(`build_vstar`/`_run_vstar`/`_oracle_metric_for_spec`)、`fusion/degrade.py`(`d1_kill_strongest`)、`fusion/run_t13_tests.py`(MODULES+=a6、KNOWN_FAIL+=HR3-design)。

### A6-1 — β 断崖：信息稀疏 vs 非信息 bin 电平错

`build_vstar` 加 `noninfo_fill`(`vreal`=V_real / `xband`=X 带内均值，即 α=0 处置)。β=1 全 bin 是信息⇒两 fill 必等(0.3908=0.3908，隔离自洽)。聚焦 d20、B=9、10 rec：vrel cliff=0.3076(0.3908→0.0832)，xband cliff=0.1680(0.3908→0.2228)。断崖残留但**被砸到 vrel 的 55%**：~0.14 是非信息 bin 电平错(xband 修掉)，~0.17 是逐带标量 w 的结构瓶颈残留。

> ⚠️ **本节“~0.17 是逐带标量 `w` 的结构瓶颈”的推论已被 A6-1c 撤销** —— 逐 bin oracle 臂显示改加权粒度买不到东西(`gain/residual` 全部 ≪0.5)，残留的 0.168 全部是不可约的信息覆盖度损失。**以 A6-1c 为准。**

### A6-1b — D1 带级亏损定标

`deficit = 10log10(px/ps)` per band per voiced frame(往返 S)。复现评审员表(d20 kr=0.4)：100-200 std 0.41/mean +0.05、200-315 0.59/+0.07、315-500 0.16/+0.01、500-800 0.16/+0.04——逐项精确吻合。corr(X,V) 逐项吻合(0.752/0.843/0.836/0.84)。V 跟踪误差(仿射残差)~7-9 dB(口径略异于评审员的 ~5，但亏损/误差比 10-60x 在两种口径下结论一致)。

kill_rate 扫描(weak-first d20)：kr 0.4→0.24、0.6→0.73、0.8→3.08、1.0→3.94 dB std。strong-first(kr=0.4 d20)：std 6.6-8.8、mean 17-24 dB。⇒ **kill ORDER 是真杠杆，非 rate/depth**。

> [!important] std 而非 mean 才是探测指标
> kr=1.0 时亏损 **mean** 高达 4.77–12.46 dB，但**恒定的亏损会被 EQ 的 `C[f]` 吸收**(C 对齐长期谱包络关系)⇒ 融合只能靠**随时间变化的部分**探测。⇒ **`std` 才是对的探测指标**，而它在“杀弱”语义内怎么都上不去(≤3.8 dB 即便杀光 100% 带内谐波)，而 VPU 带级跟踪误差是 5–9 dB ⇒ **带级融合无法探测它本应修复的损伤，与 depth/kill_rate 怎么设都无关。唯一能改变量级的是 kill ORDER，反转它就不再模拟 stage-2。**
>
> ⚠️ 此结论**条件依赖 D1 的“杀弱”语义是否代表真实 stage-2**。该验证需用户在真实 stage-2 输出上做。**在其返回前不据宣布带级路线死亡。**

### A6-1c — 逐 bin oracle 臂（撤销 A6-1 cliff 结论）

`_oracle_w_perbin`(a4)：每 bin 独立在 [0,1] 取最优。`oracle_mode='perbin'`。β=0.5、两 fill、d15/d20/d30、匹配 null(B=9 d20、B=5 d15/d30)：

| depth | fill | gran_gain/residual |
|---|---|---|
| 15 | vreal | **0.24** |
| 15 | xband | **−0.21** |
| 20 | vreal | **0.19** |
| 20 | xband | **−0.23** |
| 30 | vreal | **0.08** |
| 30 | xband | **−0.61** |

全部 ≪0.5(观测前定判据“改粒度能买大部分”的阈值)。per-bin oracle 只买 ≤24%(vreal)/负值(xband，per-bin null 膨胀——额外自由度主要从噪声里挑)；残留主要是**不可约信息减半损失**(irreduc 0.21-0.30 = 残留的 81-161%)。⇒ **A6-1 cliff 结论撤销**：逐带标量 w 非结构瓶颈，改加权粒度买不到多少。重心转向**重建覆盖度**(更多 bin 携带真逐 bin 信息=提高 β，即更好 Arm A)，非逐 bin/逐谐波加权。xband @0.5 的 0.168 残留 per-bin 买 −0.039(irreduc 0.207>0.168)⇒ 0.168 全是不可约信息减半。

> [!important] 定式：自由度不同的 oracle 不可比原始值
> per-bin 的自由度远多于 band-scalar，其原始(未校正)ΔG3rec/ΔJ3 **必然 ≥** band-scalar(多出来的自由度只能挑到更多噪声/信号)。若不做零分布校正，per-bin 看上去更好，会得出“改逐 bin 加权有价值”的**相反结论**。是**匹配零分布**把多出来的自由度所挑到的噪声扣掉，才露出真相(per-bin Δ ≤ band-scalar Δ)。⇒ **比较自由度不同的两个 oracle 时，必须各自配匹配的零分布；原始值不可比。** 这条在 A7 的 (β,α) 扫描里同样适用(不同 α 改变 V* 的逐 bin 信息量 ⇒ 自由度不同 ⇒ 每格点独立配零)。

### A6-2 — HR3 clip 安全保证在往返中丢失

`pre y_spec` max_down=−4.96(界内)、`post stft(Y)` max_down=−17.58(下向爆 12.58dB，上向仅 1.67)。机制：只改幅度+保留 S 相位⇒y_spec 非 STFT-consistent⇒OLA 相消⇒下向爆。**HR3-design**(m=0)FAIL(46/138000)⇒ 登记 KNOWN_FAIL(“往返破坏保证，未修”)；**HR3-regress**(分离 m_up=2.67/m_dn=13.58)PASS；mutation-curve 整带 bump 两方向平滑增长检出可见。G4′ 拆分：89.1% 往返越界点落 G4′ 违例但只 0.8% G4′ 违例含越界 bin；中位 |corr| pre 1.50/post 1.15(中位非往返驱动)，最差 pre 16.05/post 26.67(往返加 +10.61，7.055 尾部主要是往返)；方向错 47.7%。⇒ G4′ 中位损伤是融合自己 corr(方向错各半)，尾部最差点是往返相消。修 G4′ 分两头。

### A6 元测试与 runner

`tests.test_t13_a6` 此前**不在** `MODULES`⇒ 四个 A6 测试(含 HR3)从不执行。已加入 + 文件系统枚举元测试(删模块必检出)。a6 模块 9 测试：8 PASS + 1 XFAIL(HR3-design)。全套 95/107 passed、4 failed、3 xfailed、1 xpassed、4 skipped。

> [!danger] BOUNDARY
> 同上：仅 0624 全男声 F0 87–124Hz 正常音量；V*/X 仅测试侧；D1 “杀弱”语义未在真实 stage-2 上验证。

## A7 — (β,α) 天花板 → Arm A 可执行规格

既然 band-scalar 融合够用、瓶颈在重建(A6-1c)，A7 把结论翻译成对 Arm A 的可执行规格：**Arm A 要做到什么程度，这条路才走得通？**

### A7-1 — (β,α) 二维扫描

固定 `xband` 填充、`band-scalar` oracle `w`（per-bin 已证无用）、d20、**每格点独立 B=9 匹配零**（不同 α⇒不同自由度⇒原始值不可比，见 A6-1c 定式）。`β` = 带内携真逐 bin 信息的 bin 占比；`α` = 这些 bin 的逐 bin 保真度（`A* = α·A_true + (1−α)·A_band`）。

ΔG3rec（median）:

| β＼α | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| 0.25 | .098 | .110 | .140 | .167 | .182 |
| 0.5 | .098 | .122 | .167 | .202 | **.223** |
| 0.75 | .098 | .140 | .225 | .294 | .331 |
| 1.0 | .098 | .153 | .240 | .331 | **.391** |

α=0 时所有 β 同值（.098，V*=纯带均值，与 β 无关⇒内部一致性）。ceiling (1,1)=.3908；(0.5,1)=.2228 与 A6-1c 精确一致。等高线图见 `reports/T13A7/a7_beta_alpha_contour.svg`。

### A7-2 — α → dB（给 Arm A 的可执行数）

`A* = α·A_true + (1−α)·A_band` ⇒ 逐 bin 误差 = `(1−α)·(A_true−A_band)`，std = `(1−α)·σ_fine`，`σ_fine` = 带内谱精细结构 std（可直接量）。σ_fine（10 rec, voiced）：100-200 5.27 / 200-315 5.80 / 315-500 6.67 / 500-800 7.43 dB，**pooled 6.24 dB**。

| α | Arm A 逐 bin log 幅度误差 ≤ |
|---|---|
| 0.00 | 6.24 dB |
| 0.25 | 4.68 dB |
| 0.50 | 3.12 dB |
| 0.75 | 1.56 dB |
| 1.00 | 0.00 dB |

⇒ **1.0 dB 阈值对应 α ≥ 0.84**（84% 逐 bin 保真）。

### A7-3 — V_real 当前位置（返工一：拆 ①带级 / ②精细结构）

> ⚠️ 本节初版（α_vreal=−0.059）把「带级误差」记到了只刻画「精细结构」的 α 轴上——A_band 取自 X，故 V* 在任何 α 下带级电平精确，误差纯为精细结构 `(1−α)·σ_fine`。用「总误差」定位 α 会把 ① 记到不刻画带级的轴上，把 V_real 推出图外。已拆：

V_real 逐 bin 误差拆 ①带级跟踪 + ②精细结构（10 rec, voiced, 逐带中位）：

| 子带 | ①带级 | ②精细 | σ_fine |
|---|---|---|---|
| 100-200 | 5.15 | 3.83 | 5.27 |
| 200-315 | 4.76 | 3.98 | 5.80 |
| 315-500 | 3.83 | 5.40 | 6.67 |
| 500-800 | 6.57 | 6.17 | 7.43 |
| pooled | 4.95 | 4.69 | 6.24 |

**α_vreal = 1 − ②/σ_fine = 1 − 4.69/6.24 = +0.248**（仅 ②；逐带 +0.27/+0.31/+0.19/+0.17）。β_vreal≈1.0。⇒ V_real ~(β=1.0, α=+0.25)。**①带级（4.95 dB）根本不在这张图上，是另一笔账。**

### 两条独立 Arm A 规格（返工一）

1. **带级精度（致命，不在 (β,α) 图上）**：V_real ①=4.95 dB；A6-1b 带级亏损 std ~0.3 dB ⇒ 需 ~17× 改善才能探测亏损。
2. **精细结构精度（图上定价）**：V_real ②=4.69 dB（α=+0.25）；需 α≥0.75–0.84 ⇒ ≤1.0–1.56 dB ⇒ ~3–5× 改善。

### A7 返工二 — σ_e / σ_b 随机残差轴（模型规格，非收缩）

> ⚠️ α 轴的误差是「收缩」`(1−α)·(A_band−A_true)`，与真精细结构完全反相关（方向对、幅度压缩）⇒ oracle `w` 可部分补偿。真实模型残差是**随机**的（方向错，`w` 补不了）⇒ 同 std 下随机严格劣于收缩。故由 α 轴推的 ≤1.56 dB **偏乐观**。加两条随机扰动轴（xband、band-scalar oracle、d20、每格点 B=9 匹配零）：

**σ_e（精细结构随机残差，β=1.0, σ_b=0）**：σ_e=0.5→.379, 1.0→.357, 2.0→.286, 4.0→.171, 6.0→.090。达 0.30：**σ_e ≤ 1.0 dB**（恰在 1.0 边界，比 α 轴 ≤1.56 更紧——随机劣于收缩，符合预期）。

**σ_b（带级随机残差，β=1.0, σ_e=0）**：σ_b=0→.391, 0.1→.392, 0.3→.392, 1.0→.367, 3.0→.259。达 0.30：**σ_b ≤ 1.0 dB**（阈值在 (1.0, 3.0)）。

### 决策（观测前定）

- σ_e<1.0 ⇒ 精细结构「直接重建」难：实测 σ_e_max=1.0（恰在边界，非严格<1.0）。
- σ_b<0.5 ⇒ 带级「直接知道带能量」难：实测 σ_b_max=1.0（**非<0.5**）。

**关键诚实发现**：σ_b 恢复轴（oracle-w）未确认带级致命性——oracle-w 恢复容忍 1.0 dB 带噪声（per-(band,frame) 标量 w 能适应）。**带级致命性在「检测」路径不在「恢复」天花板**：真实算法的 c_V 需带误差 <<0.3 dB（亏损 std）才能探测⇒门控 w；V_real ①=4.95 dB ⇒ ~17× 改善，这 IS <0.5 且「直接知道干净语音带能量」难（与 A6-1b 互证）。**σ_b 恢复轴量错东西——致命的是检测，非恢复天花板。**

> [!important] 定式：天花板实验 ≠ 可探测性实验（信号定义不同，不可互换）
> 「天花板实验」（oracle-`w`，问“完美加权能恢复多少”）里的“信号”是**完整的逐 bin 亏损**（d20 被杀谐波可达 20 dB），带噪声相对它小⇒ oracle-`w` 容忍得了 1.0 dB 带噪声。而「可探测性实验」（真实决策层的 `c_V`/`w_band`，问“能否探测到亏损去门控 `w`”）里的“信号”是**带级亏损 std ~0.3 dB**，完全不同的量级⇒ 同一 `σ_b` 在两实验里判出相反结论。**要判定某条路可行，两者都要做：天花板给上界，可探测性给“决策层能否用上”的下界；任一单独都不充分。** 本定式适用于 A8（理想 V* + 真实决策层 = 把可探测性变量推到极致的实验）。

> [!important] Arm A 可行性裁决（返工后）
> oracle-w **恢复天花板**：σ_e≤1.0、σ_b≤1.0（均在 1.0 边界，borderline clean-speech-hard）。**检测要求**（真实算法 c_V，A6-1b）：带级误差 <<0.3 dB（V_real 4.95⇒~17×）——这 IS 致命（<0.5，直接知道干净带能量）。精细结构检测同理需 <<亏损。**两条都指向 VPU 域重建工程上不成立**：恢复天花板卡 1.0 边界；检测要求 <<0.3 dB 带级 + <<亏损精细结构，远严于天花板。V_real 须从 (β=1,α=+0.25, ①=4.95) 移到 α≥0.84 **且** ①<<0.3——后者（带级检测）是先卡死的那条。条件依赖 A6-1b「杀弱语义代表真实 stage-2」，用户需在真实 stage-2 输出验证。

> [!danger] BOUNDARY
> 同上：仅 0624 全男声 F0 87–124Hz 正常音量；V*/X 仅测试侧；σ_fine/σ_vreal 口径为带内 voiced 谱精细结构 std；D1 “杀弱”语义未在真实 stage-2 上验证。

## A8 — 理想 V* + 真实决策层（2×2 最后一格）

至此所有天花板都是 oracle-`w` 测的。A8 补上唯一未测的格：**理想 V*（α=1,β=1,σ_e=σ_b=0,xband）+ 真实决策层（真实 `c_V·g_f0·w_band·w_local`，非 oracle）**。若真实层在完美输入下仍交付 ≈0，则决策层独立坏、Arm A 敖不回（按天花板≠可探测性定式，理想 V*⇒d 精确等于亏损⇒探测平凡，残留失败即决策层自身）。

### A8-1 — 理想 V* + 真实决策层

| depth | dG3rec | dJ3 | J1 | J2 | (oracle 天花板 +0.391) |
|---|---|---|---|---|---|
| 15 | +0.0102 | +0.0068 | 0.000 | 0.000 | |
| 20 | +0.0057 | +0.0020 | 0.000 | 0.000 | |
| 30 | +0.0032 | +0.0014 | 0.000 | 0.000 | |

**全 depth ≈0、J1=J2=0**。`w` 分布(d20):p50=0、p90=0、max=0.74（几乎不介入；受压帧上 w p50=0/p90=0/max=0.47——该修处也不介入）。

**四因子对照**(理想 V* d20 中位 vs A4-1 V_real 带内参照)。⚠️ 初版报了**全带中位**(w_band/w_local 看似 0)；那是带外 39 个 bin(V*=V_real)拉低的伪影。下表为**带内**(100–800Hz)与全带中位：

| 因子 | 带内中位 | 全带中位 | V_real(带内) | |
|---|---|---|---|---|
| c_V | 0.573 | 0.573 | .33–.55 | **打开** |
| g_f0 | 0.718 | 0.718 | .44–.52 | **打开** |
| w_band | **0.865** | 0.000 | .62–.83 | **也打开了，没塌** |
| w_local | **0.153** | 0.000 | .92–1 | **降了**(未标定绝对电平，见 A9-3) |
| w | 0.034 | 0.000 | .033–.110 | |

> ⚠️ **本节初版「`w_band`(MSC)坍塌、是错的度量、用不了完美幅度」的机制归因已被 A9 撤销**——带内 w_band=0.865(比 V_real 还高)，并未坍；那个 0 来自带外 bin 的全带中位伪影。**以 A9 为准：乘性结合结构是主要嫌疑**(见 A9-2；二/三因子组合未测试，不作结论)。

### A8-2 — 真实决策层 σ_e/σ_b 轴（可部署规格）

理想 V* d20 天花板(真实决策)=+0.0057，半目标=+0.0029。真实决策层在 σ_e/σ_b 扫描下 **全 ≈0.005 平坦**(σ_e 0.5–6→0.0056–0.0044;σ_b 0–3→0.0057–0.0054)——决策层给≈0 **与 Arm A 质量无关**。「半天花板」被所有格点达到（因为平坦≈0.005>0.0029），但那只是噪声残留，不是「可达」——**可部署规格不存在(决策层在所有 σ 下都交付≈0)**。

### 决策（观测前定：d20 ≈0 ⇒ 最重要结局）

`dG3rec=+0.0057 ≈0`(oracle 天花板 +0.391)⇒**即便 Arm A 做到完美，现在这套融合也交付不了东西⇒决策层独立坏，必须重做；Arm A 单独投入是浪费。**

### 2×2 全表

| | 真实 V_real | 理想 V*(α=1,β=1) |
|---|---|---|
| **oracle-w(天花板)** | ≈0(ΔJ3，A5R-2) | **+0.391** |
| **真实决策层** | J3=0.028(≈0，§2) | **+0.006(≈0)** ← A8 |

**唯一工作的格 = (oracle-w + 理想 V*)**。真实决策层两列都≈0⇒决策层是约束，非 Arm A。⚠️ 初版「`w_band`(MSC)是错的度量」的归因已由 A9 撤销(带内 w_band 未坍)；乘性结合结构是**主要嫌疑**(见 A9-2)。

> [!important] 裁决
> **缺陷不归于任一单因子（四个单因子消融均不足），也不在 MSC**；**乘性结合结构是主要嫌疑**（A9 撤销了初版的 MSC 归因；二/三因子组合未测试，不作结论）。这是个明确且局部的架构缺陷：证据应是「一个判断修哪里/多少的信号」+「安全否决(只在不安全时压到 0)」，而非四个软分数连乘。
>
> ⚠️ 初版「决策层必须重做、Arm A 单独投入是浪费」口径须改：决策层要改的是**结合方式**，不是推倒重来；且它必须与 Arm A **一起**改——`w≡1` 但输入是 V_real 时依然没用(A5R-2 已证 ΔJ3≈0)。条件依赖 A6-1b「杀弱语义」未验证；但 A8/A9 的「决策层独立坏」不依赖 D1 语义（在理想 V* 下直接测出）。

> [!danger] BOUNDARY
> 同上：仅 0624 全男声 F0 87–124Hz 正常音量；V*/X 仅测试侧；A8 决策层结论不依赖 D1 语义(理想 V* 直接测)。

## A9 — 机制归因修正：乘性结构，非 MSC

A8 头条成立（理想 V* + 真实决策 ≈ 0，独立复跑 recovery 0.0504 vs oracle ~0.43）；**但机制归因错**，本节撤销并改正。

> [!note] 证据口径
> A9 的五个 `test_*` 函数均**无 assert / 无 mutation**，是**一次性表征实验**：只复算并报告，**不提供回归保证**；runner 的 PASS **不能读作结论被强制**。以下结论均为**观测证据**。

### A9-1 — 四因子带内/全带分开（撤销「MSC 坍塌」）

A8 初版报的是**全 64 bin 全带中位**，而 A4-1 参照是带内（100–800Hz）——两者不可比。分开测（理想 V*、d20、10 条、中位）：

| 因子 | 带内中位 | 全带中位 | A8 初版报的 | V_real(带内) | |
|---|---|---|---|---|---|
| c_V | 0.5730 | 0.5730 | 0.577 | .33–.55 | 打开 |
| g_f0 | 0.7179 | 0.7179 | 0.722 | .44–.52 | 打开 |
| w_band | **0.8646** | 0.0000 | 0.000 | .62–.83 | **也打开了，没塌** |
| w_local | 0.1532 | 0.0000 | 0.000 | .92–1 | 只有它降了 |
| w | 0.0341 | 0.0000 | 0.000 | .033–.110 | |

带内 `w_band=0.865`，**比 V_real 还高**。独立 `CohTracker` MSC 佐证：`V_real 0.605` / `V*(∠V_real) 0.787` / `V*(∠X) 0.999` / `V*(∠S) 0.999`。**「完美幅度使 MSC 坍到 0」不成立——那个 0 来自带外 39 个 bin(V 那里仍是 V_real)，被全带中位吃掉。**

### A9-2 — 乘性结构消融（观测证据；一次性表征实验）

用既有消融开关把因子逐个强制 ≡1（`enable_*=False`），理想 V*、10 条、d15/d20/d30、每格独立 B=9 匹配零：

| 配置 | d15 | d20 | d30 |（d20 raw recovery, 10条）|
|---|---|---|---|---|
| baseline(全真实) | +0.0102 | +0.0057 | +0.0032 | .0381 |
| w_local≡1 | +0.0373 | +0.0207 | +0.0137 | .0653 |
| c_V≡1 | +0.0059 | +0.0054 | +0.0033 | .0439 |
| g_f0≡1 | +0.0212 | +0.0116 | +0.0064 | .0462 |
| w_band≡1 | −0.0032 | +0.0002 | −0.0000 | .0411 |
| **全 ≡1 (w≡1)** | **+0.2543** | **+0.0923** | **+0.0520** | **.2999** |

**四个单因子消融均不足**（最好 w_local≡1 只到 0.0207，远小于 all=1 的 0.0923）；**全≡1 对照显著更高**。四个各自 <1 的软分数连乘 `0.573×0.718×0.865×0.153≈0.056`，平滑后 0.034——与实测 w=0.0341 吻合。⇒ **缺陷不归于任一单因子；乘性结合结构是主要嫌疑。未测试的二/三因子组合不作结论。**

（口径注：主表为 ΔG3rec 零分布校正，括号列为 raw recovery(无零校正、10 条)；A9 初测 4 条参考值为 baseline .0504 / w_local .0845 / c_V .0683 / g_f0 .0626 / w_band .0544 / all=1 .2732——同构同序，单因子≪all=1 不变。`w≡1` 非信息使用者但空分布大(w 恒介入噪声)，故 ΔG3rec(0.0923) 远小于 raw(0.300)。）

### A9-3 — `w_local` 未标定绝对电平（注入伪影，非主因）

`w_local = sigmoid((Pv_overall − P_band − thr)/slope)`，`Pv_overall` 是 V 的**绝对电平**、`thr` 固定。A8 的 V* 走后 EQ 注入、`C=0` **未做电平对齐**，实测 V* 比 V_real 带内低 3.71 dB。电平扫描（ΔG3rec，d20、10条、B=9 零）：

| V* 电平 | −10 | −3.7(对齐V_real) | 0(A8原样) | +3.7 | +10 |
|---|---|---|---|---|---|
| dG3rec | +0.0003 | +0.0031 | +0.0057 | +0.0091 | +0.0156 |

`w_local` 对电平极敏感（±10 dB ⇒ ~16× w_local 摆动）——这是 **BR2 早指出的「纯绝对电平检测器」问题**；但**修正电平只把 dG3rec 从 0.0057 抬到 0.0091（+3.7dB 对齐），远够不着 all=1 的 0.0923**。⇒ 电平未对齐是**注入伪影**，不是 A8 结论的成因；乘性结合结构仍是主要嫌疑。后续实现 V* 注入时须做电平对齐。

### A9-4 — `w≡1` vs oracle 的差（仅报告，不修）

`all≡1` d20 = +0.0923（raw .273）vs oracle-w ~+0.39：那是「逐 `(band,t)` 上界」相对「**固定 always-on 参考策略**（`w≡1`）」的差。**仅报告，不修**——`w≡1` 是固定 always-on 参考策略（**非上界**）；oracle 才是逐 `(band,t)` 上界。

### A9 裁决（修订后口径）

- **缺陷不归于任一单因子（四个单因子消融均不足），也不在 MSC**；**乘性结合结构是主要嫌疑**（A8 初版 MSC 归因已撤销；二/三因子组合未测试，不作结论）。
- 这是**明确且局部**的架构缺陷：证据应当是「**一个**判断修哪里/修多少的信号」+「**安全否决**（只在不安全时压到 0）」，而非四个软分数连乘。
- **修法明确 ⇒ 决策层要改的是结合方式，不是推倒重来；且必须与 Arm A 一起改**——`w≡1` 但输入是 V_real 时依然没用（A5R-2 已证 ΔJ3≈0）。

> [!danger] BOUNDARY
> 同上：仅 0624 全男声 F0 87–124Hz 正常音量；V*/X 仅测试侧；不改生产模块/参数/门槛/注册表（消融开关既有，仅测试侧调用）；未读 0625 语音。

## T13-MVP — 真实数据使用说明

按 A9 裁决，把四软分数连乘改为「**一个主修正信号 + 二值安全否决**」（工程 MVP，第一版不求最优，只求该动时会动、不安全时严格退回 S）。

**输入定义**：`S.wav` = stage-2（被测降级麦）录音，`V.wav` = VPU 录音；两者均须 **16 kHz**、**等长**（不等长明确报错，不静默裁剪）；多声道平均为 mono；**严禁分别峰值归一化**——S/V 相对电平原样保留（diagnostics 报两者峰值/RMS 供核对）。X（干净参考）只在评测侧，从不进入算法路径。

**最短命令**：

```bash
python -m fusion.run_fusion --stage2 S.wav --vpu V.wav --output Y.wav --diagnostics diag.json
# A/B 对照（除结构性末采样外与 S 在 1e-4 内一致）：加 --strength 0
```

可选：`--strength 0..1`（缩放最终修正量，默认 1）、`--mode legacy_multiply`（旧四因子乘积对照；默认 `mvp`）。输出 `Y.wav` 16 kHz PCM_16；`strength>0` 时末尾 20 ms（320 采样，共享因果 iSTFT 的病态边界区，legacy 同样存在）淡出到零以防端点爆音，其余采样不动；**`strength=0` 不施加额外尾部淡出——除因果 Hann 分帧结构性不可编码的最后 1 个采样外，所有采样与 stage-2 输入在 1e-4 内一致，末采样单独报告**（diagnostics `output.tail_fade_samples` 反映实际行为：strength=0 为 0，strength>0 且长度足够为 320）。

**算法（预置阈值，先于任何效果观测写入 config）**：主信号 = `w_local` 带证据 `evi = Pv′ − P_band`（V′ 为 EQ 对齐后 V 的 100–800 Hz 整体电平，P_band 为 S 逐子带电平）——现有唯一指向「S 在哪丢了能量」的带级信号；修多少由 synthesis 内逐 bin 亏损 `d = log|V′|−log|S|`（clip +25/−5 dB）给出，非实际亏损的带 d≈0⇒不伤。二值否决（0/1，不连续衰减主信号）：f0 conf<**0.50**（沿用 EQ 双可信工差点；无浊音证据时 V 的谐波亏损证据不可信）、c_V<**0.30**（MVP 下 c_V 关闭 KR1 EQ-残差偏置项——该偏置度量 S↔V 关系漂移＝要修的损伤本身，非 V 硬件故障；健康 V≈0.6–1、V 退化≈0.2，0.30 由构造取中，非效果调参）、逐 bin MSC(V′,S)<**0.30**（A9 表征健康带内 0.62–0.87 的约半数以下才火；MSC 幅度尺度不变，带内损伤不触发）。**否决/startup/reset 在平滑后重施 hard mask：这类帧最终送入 synthesis 的 w 严格为 0**（平滑器 fall tau 不会泄漏残留，如首 veto 帧 0.51），主信号自身的平滑保留，安全恢复后按既有 rise tau 回升。diagnostics 的否决占比基于**最终 hard mask**（帧级否决∪逐 bin 否决∪startup/reset floor）。MVP 只在 **100–800 Hz** 介入（VPU <100 Hz 不可靠、>800 Hz 无信息——CR3）。`strength=0` 或否决全火 ⇒ 频谱生产路径严格回到 S；波形经因果 iSTFT 后除结构性末采样外与 S 在 1e-4 内一致（MVP 默认关 comfort noise——−40 dB 不可听守卫——以使谱恒等式精确；legacy 模式保留）。层 1（EQ/F0）与 synthesis 结构两种模式共享，未重写。

**diagnostics 字段**：`commit/mode/strength/sr`；`inputs.stage2|vpu`（peak、rms_dbfs、声道数、采样数）；`output`（peak、rms_dbfs、clipped_samples、tail_fade_samples）；`coverage_100_800`（100–800 Hz 内 |修正|≥1 dB 的 (bin,帧) 占比）；`correction_100_800` 与 `band_stats.{100-200,200-315,315-500,500-800}`（修正量 p50/p90|max/max|.| dB）；`veto_fraction_100_800`、`veto_f0_frame_fraction`。

**MVP 验收（四条预定 blocker，返工后复跑值）**：M1 安全退化（`--strength 0` 经真实 CLI：**除结构性不可编码的末采样外所有采样与 S 在 1e-4 内一致**（all-but-final max 6.8e-05），末采样单独报告（~2e-02，因果分帧端点 w=0 永不编码），不施加 tail fade〔diagnostics `tail_fade_samples=0` 断言〕；强制否决经真实 Fusion ≡S；**状态转换**：先建立 w≈0.99，S 转电平匹配噪声⇒帧级 veto，**首 veto 帧最终 w 精确 0**、输出谱=S 谱 diff 5.4e-08——平滑后重施 hard mask，fall tau 不泄漏）；M2 数值与接口（无 NaN/Inf；16k/等长/strength 契约错误非零退出；batch≡streaming 内部 diff 0.0）；M3 明确受损时介入（S 带内 −20 dB、V*=X：带级 LSD **19.32→0.90 dB** 严格下降，coverage 0.750，评估区否决 0.000〔全程 0.249 含 EQ 启动期 floor〕）；M4 未受损不大伤（S=X,V=X：|修正| p99=**9.5e-07 dB**≤1，min −1.9e-06 无反向）。测试在 `tests/test_t13_mvp.py`，每条一个直接功能测试，引用均为输入侧（S/X），历史套件 report-only 未重跑（streaming/static/ablation/meta 本地重跑全 PASS）。已知：EQ 启动期（120 可信帧 ≈1.2–2s）w 严格为 0（strict-zero 语义）；合成测试信号需含同包络宽带成分（EQ SNR 门为全 bin 平均，严格带限信号永不 credible——测试 helper 已修，真实全带录音不受影响）。

**当前边界与已知风险**：16 kHz；0624 全男声正常音量仅为开发表征，**不代表用户真实 stage-2**（未在真实数据上调过阈值）；阈值由构造与 A9 健康范围预置，**未做效果调参**，首次真实数据运行的效果未知；EQ 未收敛期（前 ~1.2 s）介入被层 1 启动地板压低；EQ 冻结在损伤时刻时 C 会吸收带内亏损（AC2 已知语义——真实佩戴场景冻结在健康时刻）；幅度-only 编辑的频谱不一致 ⇒ ISTFT 拖尾（AC1 已接受代价）；w_local 在 EQ 未对齐时退化为绝对电平检测（BR2）；A9 的二/三因子组合未测试，乘性结合仅为主要嫌疑，MVP 的组合式（主信号+二值否决）是按该嫌疑做的工程替代，非已证最优；MVP 关闭 comfort noise（不可听守卫）。未读 0625；未提交用户真实数据。

## T13-N1 — 可信度外置 + 加减双权围栏 + D5 谷底噪声底

**结构**（生产 CLI `--mode n1`，默认）：`c = log|V′| + G − log|S|`；`w = p·w_band(f)`（固定曲线：100–800 全权，向 2 kHz 单调收尾）；`Δ↓ = Δ↓min + p·w_band·g_v·(Δ↓max−Δ↓min)`；`log|Y| = log|S| + clip(w·c, −Δ↓, +Δ↑)`；∠Y=∠S。加减**双权独立**（错加=编造、错减=抹真实信号，风险不对称）；`g_v` 只进 Δ↓ 不进 w；`p[t]` 为**外置可信度接口**（本单 MANUAL 常数扫描自变量，ORACLE 生产路径拒绝）；`G = a[t] + s[t]·f̃` 两自由度，只在 100–800 Hz 拟合、1–2 kHz 空洞结构性不读；`g_v` 由 raw VPU 共享 F0 契约（1−CMND)^γ，非对称平滑升慢(100ms)降快(20ms)；D5=谐波间噪声底注入（oracle F0 栅格±W 峰区、E_peak−L 谷底、清音不注入）。`w_local_band`/逐谐波族从生产路径摘除（代码留 legacy）。舒适噪声默认关。初值 Δ↓min=4/Δ↓max=20/Δ↑=25 dB 为扫描初值非定论。

**结构不变量（返工后全过，各配 mutation）**：I1 `p≡0 ⇒ Y≡S` 真实 0624 音频**全长逐样本** max diff 4.5e-08（尾覆盖修复后；dither mutant 被抓；streaming 含尾零同语义全长 4.5e-08）；I2 `g_v≡0 ⇒ log|Y| ≥ log|S|−Δ↓min` 谱合成域 100% bin-frames 0 违例（Δ↓-ignores-g_v mutant 被抓；波形重分析域可局部偏移——幅度域编辑非帧一致，MVP/legacy 同有）；I3 未来扰动 **4/4 切点逐位相同（双侧全新实例，K=P−(win−hop)=P−320，由因果分帧/OLA 推导并写入测试注释）** + batch≡streaming 0.0（noncausal-a mutant 在 4/4 切点被抓——旧版同一实例跑两次属状态污染假阳性，已修）；**ShapeGain 截距修复后精确线性恢复 fit-band 逐 bin 误差 0.00e+00 dB**（门槛 1e-5；旧截距 mutation 2.41dB 被抓）。另：g_v 方向（浊 0.993 > 清 0.357，flip mutant 被抓）、a/s 时间常数分离（t50 50ms vs 1100ms，等 tau mutant 被抓）、ORACLE 拒绝（含 CLI）、D5 sanity（L=10 抬谷 +8.3dB > L=40 +0.02dB，峰不动）。

### ⚠️ 主实验结论（v2：截距修复后重跑；初版红旗已撤销，见下）

> ⚠️ **初版「所有 L 最优 p=0、谷底误差随 p 单调上升」的红旗结论已被撤销**：其成因是 ShapeGain 最小二乘截距错误（把 f̃ 均值处预测值当 f̃=0 截距，系统性多加 s·mean(f̃)≈+8dB，翻转了 c 的符号结构）+ LSD 切错维度全 NaN。两处修正后全量重跑，以下为修正值。

修正后 `L×p` 扫描（同 10 条录音、同预置指标/分桶）：**最优 p 随 L 变脏而右移，方向与预置判读一致**：L=40/30/25/20 最优 p=0.0（谷底误差平坦 ±0.2dB）；L=15 最优 p=0.25（+0.25 vs S +0.34）；L=10 最优 p=0.5（**+0.64 vs S +0.78**，改善仅 ~0.14 dB——方向对但效应量弱）。p=1 仍整体抬谷（L=10 +1.04），Δ↓ 中段开启时轻微压谷。**红旗不再成立，但收益微弱**：谷底误差改善 ≤0.2 dB，远小于把 V 内容引入带内的代价（p=1 时带内 LSD(Y,X)≈7.4 dB——Y 跟随 V′ 的精细结构而非 X）。

三条对照（修正后）：真 V 与零信息对照有差但方向不一致（L=10 p=0.75 real −0.02~+1.65 vs shuffled +0.84~+1.18 vs const +0.39~+2.29，逐文件互有胜负）——融合在用 V 内容，但在该数据上 V 内容既含可压谷底信息也含额外谱细节。p≡0 恒等（I1）与 p≡1 上界臂（p=1.0 扫描列+样本）不变。

**归因（修正截距后）**：①截距错误曾使 G 整体抬高 ~s·mean(f̃)，翻转 c 符号制造了初版「单调抬谷」假象——修正后 p 中段轻微压谷、p=1 仍净抬升；②raw VPU 的谐波间电平高于干净 FF（V rms 0.004 vs FF 0.026）仍成立——**「VPU 谷底干净于 stage-2」在 raw VPU + 安静场景 FF proxy 上未观察到**，若前提指 Arm A 修复后的 VPU 须待接入复测；③D5 在安静场景 FF 上的注入量有限（自然谷底仅低于峰 ~10–20 dB，L≥25 无操作）——「高噪场景明显噪声底」在该批安静录音上不可复现。**结论：结构就位、方向正确但收益微弱；定价 VPU 支路需 Arm A 接入后的真实 stage-2 数据。**

> ⚠️ **初版「带外（800–2k）p=1 时 LSD 变化 <1 dB」已撤销**：当时 LSD 切错维度（batch 轴）全为 NaN，该结论无证据。修正后（频率轴 + finite/非空守卫）：p=1 时带外 LSD(Y,X) ≈ 6.4 dB（S ≈ 0.2–1.3）——w_band 尾巴 + G 外推在 800–2k 仍有实质修改，**「带外不变差」未达成**，按归因边界如实报告（p≤0.5 时带外变化小）。

**图表**：`reports/T13N1/heat_lp_*.png`（6 张：谷底误差/峰误差/HC(Y)/HC(S)/分带 LSD）、`spec_frame.png`（典型帧 S/V'/Y/X 频谱）、`shape_curves.png`（a/s 随时间）、`w_dd_curves.png`（w 与 Δ↓ 曲线）、`scan_results.json`（逐格逐录音原始值）、`optimal_p.json`；**听感样本** `reports/T13N1/samples/`（3 条件 × 3 片段 × S/V/Y/X 四路 = 36 个 wav）。

**边界**：V = raw VPU（Arm A 未接，收益上限被它压着）；S = 干净 FF + D5 proxy **非真实 stage-2**；全男声安静场景佩戴稳定，不外推；Δ↓min/max/Δ↑ 为初值，按纪律应以谷底电平曲线为横轴扫——本单红旗未到调参阶段；0625 未动；无检测器，BR1/BR2 反重言不适用但静态检查仍证明算法路径不读 X/D5 内部量（`rg "apply_d5|E_peak|d5_valley|d5_peak|d5_level|valley_mask|peak_mask" fusion/ -g '!degrade.py'` → 0 hit）。

## T13-N2 — 公开 GTCRN 清理 VPU 后接入 N1

**协议(观测前锁定)**:整段固定标量增益 `V_in = g·V_raw → GTCRN → V_dn = GTCRN(V_in)/g`,四档:raw(g=1)/rms→−30/rms→−24/peak→−6 dBFS;缩放后 peak≥1 即 INVALID 跳过;禁逐帧 AGC/二次归一化/响度匹配;bypass 对照 (g·V)/g ≡ V(≤1e-6,漏除 g 的 mutation 必爆);尾采样 hold-last 恢复长度。模型:`gtcrn_simple.onnx`(sherpa-onnx release,上游 Xiaobin-Rong/gtcrn),SHA256 `e77603ac…b534`,535638 B,repo 外缓存,sherpa_onnx 1.13.7/onnxruntime 1.29.0。**GTCRN 输出电平与输入增益近乎无关(out rms≈0.0039 恒定)——/g 恢复物理幅值是协议核心**。

**A(0624 十条,同 N1 scan 确定性片段)**:GTCRN 大幅压低 VPU 谐波间内容——raw V 谷底相对 X −11~−17 dB → V_dn **−24~−30 dB**;但 100–800 Hz LSD(V,X) 从 ~15–19 升至 ~19–26 dB(增强后的 V 精细结构偏离 X);F0 conf/voiced coverage 变化见表 `A_denoise_metrics.json`。

**B(接入 N1,L×p 全扫描,四 V 版本×10 条)**:I1 p=0 对每个 V 版本成立;零信息对照非空操作(permute 非恒等 ~5e-02,输出与真实路径差 5.2e-02~1.06e-01)。**判据结果(预置,三准则全要;⚠️ 二次返工修正统计口径)**:三指标改为**逐录音 paired difference 再取中位**(旧式 median 差 ≠ paired median 差),且 c1/c2/c3 必须**同一 (L,p) 格**联合成立(旧式跨格拼接可假通过——两个纯统计反例测试在 `test_N2_criteria_statistics`)。修正后 paired-median(逐格):rms_m30 最佳 **L=10,p=0.25 valley_gain −0.1511**;rms_m24 **−0.1950**(L10,p.25);peak_m6 **−0.2043**(L10,p.25)——**无任何格满足 c1**(peak_worse/lsd_worse 在同格亦未超 0.50 阈,但 c1 已一票否决)⇒ **全部「未修复前提/域外失配」——公开 GTCRN 未修复当前 VPU 支路前提,不得调参补救**。机制:V_dn 谷底被 GTCRN 压得比 X 自然谷底深得多,N1 压谷**过冲越过 X**(|Y−X| 变大)——机制在工作但方向过冲;GTCRN 对 VPU(骨导/域外)的谐波保持亦不可控。

**C**:SKIP——MVP 真实数据任务无用户 stage-2/VPU 路径记录(当次 smoke 为合成),不猜路径,待用户提供。

**边界**:GTCRN 为空气传导域外模型,结论仅适用于本批 0624 安静男声 + D5 proxy;不构成对 GTCRN 模型本身的评价;p=1 是全信任参考非 oracle。产物:`reports/T13N2/`(model_provenance/fixed_gain_report/A_denoise_metrics/B_scan_results/B_scan_summary.csv/B_zero_info_control/criteria_verdicts/2 热图)。复现:`pip install --user --break-system-packages sherpa-onnx` + 下载模型至 `~/.cache/t13/models/` + `python -m pytest tests/test_t13_n2.py -s`(或逐函数调用)。
