"""FusionConfig — the SINGLE source of every constant in the fusion algorithm.

T13-A scope: every constant is a PLACEHOLDER DEFAULT.  No tuning is done in
this stage (spec: "不调参数 —— 所有常数留占位默认值,集中在单一 config
文件,便于 B 阶段一处调完").  B-stage tuning edits THIS file only.

The four decision factors (c_V, g_f0, w_band, w_local) and every other
ablatable component each carry an ``enable_*`` switch — the ablation interface
(``tests/test_t13_ablation.py``) flips these one at a time and checks the
pipeline still RUNS (existence, not effect — T13-A reports no effect numbers).

口径 (AGENTS.md §9 / T13 spec): 16 kHz, n_fft=512, win=480, hop=160,
causal framing, bins 1..64 (31.25–2000 Hz) = fusion band, bins 65+ = S
pass-through.  Above 2 kHz ``S`` is copied verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# --- STFT 口径 (must match lowband/dsp/stft.py StftConfig defaults) --------
SR = 16000
N_FFT = 512
WIN = 480
HOP = 160
WINDOW = "hann"
KEEP_BINS = 64            # bins 1..64 (DC dropped) = 31.25–2000 Hz fusion band
FUSION_LO_BIN = 1         # inclusive, 1-based FFT-bin index (bin 0 = DC, dropped)
FUSION_HI_BIN = 64        # inclusive → 2000 Hz; bins 65..256 pass through from S


@dataclass
class FusionConfig:
    """All knobs.  Defaults are placeholders; do NOT tune in T13-A."""

    # --- STFT / framing (口径) ---
    sr: int = SR
    n_fft: int = N_FFT
    win: int = WIN
    hop: int = HOP
    window: str = WINDOW
    keep_bins: int = KEEP_BINS
    fusion_lo_bin: int = FUSION_LO_BIN
    fusion_hi_bin: int = FUSION_HI_BIN   # 2 kHz

    # --- 2 kHz boundary ---
    boundary_taper_start_bin: int = 58    # ~1812 Hz; let w taper to 0 by hi_bin
    boundary_slope_check_lo_hz: float = 1800.0
    boundary_slope_check_hi_hz: float = 2200.0

    # ===== Layer 1 · alignment (AC2: EQ frozen; AC1: delay comp DELETED) =====
    # AC1 removed DelayComp + GCC-PHAT (phase taken from S ⇒ no phase coherence
    # needed; 0–10 sample delay irrelevant to magnitude envelope).
    enable_eq: bool = True
    eq_mode: str = "frozen"               # AC2: "frozen" (B1) | "adaptive" (B0, ablation)
    eq_ema_tau_s: float = 1.0             # 1–3 s order (placeholder; chosen
                                          # so M2's 3-s ±1 dB convergence gate holds)
    eq_outlier_reject_db: float = 6.0     # |d − C| > this → discard (robust)
    eq_freq_smooth_bins: int = 1           # AC2: per-bin C (median-style), NO freq smoothing
                                          # (B0 used 5; flattened per-bin C ⇒ V'
                                          # misaligned on harmonic bins ⇒ G1 fail)
    eq_startup_w_floor: float = 0.10     #压低 w until C converges
    eq_converge_db: float = 1.0          # legacy |ΔC| gate (adaptive ablation)
    eq_converge_n_frames: int = 20      # legacy
    eq_coldstart_frames: int = 120      # AC2: credible updates before FREEZE
    eq_update_s_snr_db: float = 6.0       # S local-SNR gate (dual-credible)
    eq_update_f0_conf: float = 0.50       # f0-confidence gate (dual-credible)
    eq_band_lo_hz: float = 100.0          # V usable band (安静场景实测)
    eq_band_hi_hz: float = 800.0

    # EQ change-point (reset trigger)
    enable_eq_changepoint: bool = True
    cp_msc_jump: float = 0.30             # legacy single-frame jump (DISABLED — LR2: mis-fires on V-atten max-bin)
    cp_eqres_jump_db: float = 8.0         # |d − C| jump (DISABLED by default — LR2: mis-fires on V-atten)
    cp_eqres_trigger: bool = False       # LR2: eqres_jump watchdog (default OFF; ablation arm)
    cp_msc_collapse: float = 0.05        # LR2: sustained band-mean MSC < this ⇒ donning/signal-loss
    cp_sustain_frames: int = 30          # LR2: consecutive low-MSC frames before watchdog fires
    cp_fast_tau_s: float = 0.30           # fast re-estimate EMA after reset
    cp_reset_w_floor: float = 0.05        #压低 w during re-estimation
    cp_hold_frames: int = 10

    # ===== Layer 2 · decision (w = c_V · g_f0 · w_band · w_local) =====
    enable_c_V: bool = True
    # c_V components: ① in-band SNR (V speech vs V's OWN device noise floor —
    #    FR1: replaces the old running-MAX + fixed-floor which was (a) directional
    #    (quiet⇒low c_V, but quiet needs V most) and (b) a ratchet (one loud event
    #    permanently depressed c_V).  Noise floor = causal per-frame low-percentile
    #    of per-bin |V|^2 with a slow time EMA (tracks VPU device noise; scales
    #    with recording gain ⇒ SNR invariant to gain; holds during loud events ⇒
    #    no ratchet).  ② MSC  ③ EQ residual.
    cv_energy_tau_s: float = 1.0          # EMA tau for in-band V speech level
    cv_nf_tau_s: float = 2.0             # slow EMA tau for V noise-floor estimate
    cv_nf_quantile: float = 0.15         # per-frame per-bin low-percentile (noise)
    cv_snr_ref_db: float = 15.0          # SNR operating point (placeholder)
    cv_snr_scale_db: float = 6.0        # SNR sigmoid scale (placeholder)
    cv_legacy_ratchet: bool = False      # FR1-c mutation: revert to running-MAX e_term
    cv_legacy_abslevel: bool = False     # FR1-a mutation: pure absolute level (no SNR)
    cv_msc_tau_s: float = 1.0
    cv_m3_noise_db: float = -20.0            # M3/FR1-b synthetic device-noise floor
    cv_eqres_tau_s: float = 1.0
    cv_bias_tau_s: float = 3.0             # KR1: slow EMA for the EQ-residual long-term bias
    cv_eqresid_mode: str = "bias"         # KR1: "bias"(long-term) | "abs"(per-frame, B0) | "off"
    # legacy (unused after FR1; kept for reference / static-test compat)
    cv_e_floor_db: float = -60.0
    cv_e_full_db: float = -20.0
    # non-symmetric hysteresis on c_V itself (升慢降快)
    cv_rise_tau_s: float = 0.50
    cv_fall_tau_s: float = 0.15
    cv_changepoint_floor: float = 0.05     # forced压低 on change-point

    enable_g_f0: bool = True
    g_f0_gamma: float = 1.0               # g = f0_confidence^gamma (soft, no threshold)
    g_f0_floor: float = 0.0

    enable_w_band: bool = True            # ablation → fixed curve (use_w_band_fixed_curve)
    use_w_band_fixed_curve: bool = False  # ablation switch
    wb_msc_tau_s: float = 1.0             # online causal MSC(f) EMA
    wb_lo_bin: int = 1
    wb_hi_bin: int = 64
    wb_transition_octaves: float = 1.0    # taper to 0 over ~1 oct toward hi_bin

    enable_w_local: bool = True           # ablation → pure band weight (no detection)
    use_w_local_pure_band: bool = False   # ablation: w_local ≡ 1 in band (no detection)
    # AC3 (B1): w_local = BAND-LEVEL const-⑤ gate (per-harmonic ①②③④⑤ DELETED;
    #   B0.5 proved per-harm info can't transfer VPU→mic; ① maxes 0.863 even at
    #   iso=100%).  w_local_band[b] = sigmoid((Pv_overall − P_band[b] − thr)/slope);
    #   Pv_overall = V level over 100–800 Hz; bands >wl_v_usable_hi_hz ⇒ w=0 (CR3).
    wl_band_thr_db: float = 6.0         # evi (Pv−P_band) above this ⇒ use V (killed band)
    wl_band_slope: float = 3.0          # sigmoid slope (per dB)
    wl_v_usable_hi_hz: float = 800.0    # VPU usable band upper edge (CR3 scope)
    wl_v_perturb: str = "none"          # ER1 band-level: "none"|"shuffle"|"const"
    enable_w_local_vfallback: bool = True  # legacy (unused after AC3; kept for refs)
    enable_valley_rule: bool = True         # legacy (unused after AC3)

    # w time smoothing — TRUE non-symmetric one-sided EMA (升慢降快)
    enable_asym_smooth: bool = True       # ablation → symmetric (use_symmetric_smooth)
    use_symmetric_smooth: bool = False
    w_rise_tau_s: float = 0.060           # 60 ms (more-V rises slowly)
    w_fall_tau_s: float = 0.015           # 15 ms (back-to-S falls fast)

    # harmonic-domain freq smoothing (AC3: removed — band-level has no harmonics)
    enable_harm_freq_smooth: bool = False
    use_bin_freq_smooth: bool = False
    w_k_smooth: int = 0

    # ===== Layer 3 · synthesis (AC1: magnitude-only, phase=∠S) =====
    # AC1 removed complex-convex arm (use_complex_convex) — the ~−3 dB dip at
    # 90° phase mismatch was a complex-vector-cancellation artifact, impossible
    # in magnitude-only fusion.  log-clip retained (kills' log|S| guard).
    # HR1 (B1 rework): S-ANCHORED asymmetric clip.  delta_db (old symmetric,
    # V'-anchored) replaced by delta_up_db (allow restore) / delta_down_db (V
    # can barely lower S).  These are NEW params — NOT subject to the old
    # "Δ frozen" rule (that constrained the old symmetric Δ).
    delta_up_db: float = 25.0             # allow restoring killed harmonics (20–30 dB)
    delta_down_db: float = 5.0            # V can barely lower S (3–6 dB)
    delta_db: float = 10.0               # legacy (old V'-anchored; HR1 mutation uses)
    synth_legacy_vprime: bool = False   # HR2 mutation: revert to old V'-anchor formula

    enable_comfort_noise: bool = True
    # FR2: comfort-noise level is ADAPTIVE (relative to current in-band speech
    #    RMS, a constant gap in dB) — NOT a fixed absolute floor (which covers
    #    quiet speech).  Gap is invariant to speech scaling (FR2-a); ≥40 dB below
    #    speech RMS = inaudible (FR2-b); injected after fusion, not scaled by w.
    cn_below_speech_db: float = 40.0     # gap below in-band speech RMS (placeholder)
    cn_speech_tau_s: float = 0.3         # EMA tau for speech-RMS tracking
    cn_fixed_level_db: bool = False      # FR2 mutation: revert to fixed level
    cn_floor_db: float = -60.0           # legacy fixed level (FR2 mutation uses)
    cn_ema_tau_s: float = 2.0             # noise-shape EMA (still min-trace flavour)
    cn_independent_of_w: bool = True      # inject after fusion, not scaled by w

    # ===== F0 estimation (causal) =====
    # analysis win = STFT win (480 = 30 ms) ⇒ 0 extra delay beyond STFT.
    # f0_min floor rises to ~70 Hz (tau_max ≈ win/2); adult F0 ≥ ~85 Hz covered.
    f0_frame_len: int = WIN               # = 480, same buffer as STFT frame
    f0_min: float = 70.0                  # Hz (placeholder; limited by win)
    f0_max: float = 400.0

    # ===== MVP (T13-MVP): ONE main correction signal + BINARY safety vetoes =====
    # Replaces the legacy four-soft-score product c_V·g_f0·w_band·w_local (A9:
    # the multiplicative COMBINATION structure is the prime suspect).  All
    # thresholds below were FIXED here, with their physical semantics, BEFORE
    # any MVP effect observation (no tune-after-look).
    #   Main signal  = w_local band evidence  evi = Pv′ − P_band  (V′ = EQ-
    #       aligned V, overall 100–800 Hz level; P_band = S per-sub-band level):
    #       the ONLY existing per-band signal that points WHERE S lost energy
    #       relative to V.  The correction AMOUNT is the per-bin deficit
    #       d = log|V′|−log|S| inside synthesis (logclip_mix), so a band that
    #       fires but is not actually deficient gets d≈0 ⇒ no harm.
    #   Vetoes (BINARY 0/1, never continuous attenuation):
    #     f0 conf < mvp_veto_f0_conf — unvoiced/uncertain-F0 frames: V's
    #       harmonic-deficit evidence is untrustworthy there.  0.50 reuses the
    #       EXISTING dual-credible EQ operating point (eq_update_f0_conf).
    #     c_V < mvp_veto_cv — V's OWN health collapsed (device noise / absent V).
    #       In MVP mode c_V is computed with the KR1 EQ-residual bias term OFF
    #       (cv_eqresid_mode='off'): that bias measures S↔V relationship DRIFT,
    #       i.e. exactly the damage the MVP must correct — keeping it would veto
    #       the damaged case.  0.30 sits between healthy (≈0.6–1: both sqrt
    #       components ≥~0.55 needed) and V-degraded (≈0.2) — from construction,
    #       not from effect runs.
    #     per-bin MSC(V′,S) < mvp_veto_msc — V and S not describing the same
    #       source.  0.30 is ~2× below the healthy in-band range measured in the
    #       A9 characterization (0.62–0.87; MSC is magnitude-scale-invariant, so
    #       it survives in-band damage) ⇒ fires only on genuine collapse.
    #   MVP intervenes ONLY in 100–800 Hz (w_local band scope; eq_band_lo_hz/
    #       wl_v_usable_hi_hz: VPU unreliable <100 Hz, no info >800 Hz — CR3).
    decision_mode: str = "mvp"            # "mvp" | "legacy_multiply"
    strength: float = 1.0                 # scales the FINAL clipped correction; 0 ⇒ Y≡S exactly
    mvp_veto_f0_conf: float = 0.50
    mvp_veto_cv: float = 0.30
    mvp_veto_msc: float = 0.30
    mvp_comfort_noise: bool = False      # MVP v1: comfort noise OFF ⇒ exact safety
                                         # identity (veto/strength-0 ⇒ Y≡S bit-exact);
                                         # legacy_multiply keeps it (inaudible −40 dB guard)

    # ===== T13-N1: trust-routed add/subtract fusion (production structure) =====
    # Structure (pre-fixed): c = log|V′| + G − log|S|;  w = p·w_band(f);
    #   Δ↓ = Δ↓min + p·w_band·g_v·(Δ↓max − Δ↓min);  log|Y| = log|S| + clip(w·c, −Δ↓, +Δ↑);
    #   ∠Y = ∠S.  "add" and "subtract" carry INDEPENDENT weights (risk asymmetry:
    #   wrong add = fabrication, wrong subtract = erasing real signal); g_v only
    #   enters Δ↓, never w (unvoiced V′ has no content — a wide Δ↓ would crush
    #   consonants).  w = 0 ⇒ Y ≡ S identity preserved (HR2).
    #   Initial clip values are FIRST GUESSES to be swept on the valley-level
    #   curve, not hand-picked conclusions.
    n1_delta_down_min_db: float = 4.0    # Δ↓ floor at g_v = 0 (initial 3–5)
    n1_delta_down_max_db: float = 20.0   # Δ↓ ceiling at p·w_band·g_v = 1 (initial 20)
    # delta_up_db (25) reused for Δ↑.
    # --- trust interface (p[t]; NOT an estimator this batch — MANUAL consts) ---
    trust_source: str = "manual"         # manual | external | internal_fallback
    trust_const: float = 1.0
    trust_path: Optional[str] = None     # json {"p": [...]} | wav (16 kHz, frame anchors)
    trust_allow_oracle: bool = False     # MUTATION ONLY: production must reject ORACLE
    # --- voiced routing g_v[t] (from RAW VPU; shared F0 contract; soft, no threshold)
    enable_g_v: bool = True
    gv_gamma: float = 1.0                # shape param: g_v = (1−CMND)^gamma
    gv_rise_tau_s: float = 0.10          # SLOW rise (false-voiced ⇒ crushed consonants)
    gv_fall_tau_s: float = 0.02          # FAST fall (false-unvoiced ⇒ less gain only)
    gv_override: Optional[float] = None  # test-side: force g_v constant (I2)
    gv_flip: bool = False                # MUTATION: use CMND instead of 1−CMND (direction)
    # --- shaping gain G[f,t] = a[t] + s[t]·f̃  (fit ONLY on S-trusted band =
    #   fit_lo..fit_hi; S-hole bands are structurally not read, only extrapolated)
    enable_shape: bool = True
    shape_a_tau_s: float = 0.08          # fast: syllable on/off (50–100 ms band)
    shape_s_tau_s: float = 2.0           # slow: wearing / transfer (seconds)
    shape_fit_lo_hz: float = 100.0
    shape_fit_hi_hz: float = 800.0
    n1_mutation_noncausal_a: bool = False  # MUTATION: a[t] reads one future frame
    shape_mutation_old_intercept: bool = False  # MUTATION: pre-rework intercept (=mean, no -s*mean(f))
    n1_mutation_dd_ignores_gv: bool = False  # MUTATION: Δ↓ no longer routes g_v (I2 must fail)
    # --- fixed w_band curve (MSC-driven weight belongs to the future trust module)
    n1_wband_lo_hz: float = 100.0        # full weight [lo, full_hi]
    n1_wband_full_hi_hz: float = 800.0
    n1_wband_zero_hi_hz: float = 2000.0  # zero above (monotone taper in between)
    # --- comfort noise in N1 (default OFF: the main metric is the valley floor,
    #   injected noise would pollute it)
    n1_comfort_noise: bool = False

    # ===== misc =====
    eps: float = 1e-8

    def with_switches(self, **changes: Any) -> "FusionConfig":
        """Return a copy with the given switches flipped (ablation helper)."""
        return replace(self, **changes)


def default_config() -> FusionConfig:
    return FusionConfig()
