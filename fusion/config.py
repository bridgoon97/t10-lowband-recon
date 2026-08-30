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
    cp_msc_jump: float = 0.30             # |ΔMSC| → reset
    cp_eqres_jump_db: float = 8.0         # |d − C| jump → reset
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

    # ===== misc =====
    eps: float = 1e-8

    def with_switches(self, **changes: Any) -> "FusionConfig":
        """Return a copy with the given switches flipped (ablation helper)."""
        return replace(self, **changes)


def default_config() -> FusionConfig:
    return FusionConfig()
