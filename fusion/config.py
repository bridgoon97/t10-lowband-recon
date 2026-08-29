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

    # ===== Layer 1 · alignment =====
    # GCC-PHAT fixed delay (samples, 100–600 Hz).  Online only applies + monitors.
    enable_delay_comp: bool = True
    delay_samples: int = 0                # placeholder; measured offline, 0–10
    gcc_lo_hz: float = 100.0
    gcc_hi_hz: float = 600.0

    # EQ C[f] robust causal EMA (residual V↔S timbre alignment, NOT the
    # pre-reconstruction domain-alignment EQ — see module docstring).
    enable_eq: bool = True
    eq_ema_tau_s: float = 1.0             # 1–3 s order (placeholder; chosen
                                          # so M2's 3-s ±1 dB convergence gate holds)
    eq_outlier_reject_db: float = 6.0     # |d − C| > this → discard (robust)
    eq_freq_smooth_bins: int = 5          # static 1-D smooth along f (not time)
    eq_startup_w_floor: float = 0.10     #压低 w until C converges
    eq_converge_db: float = 1.0          # |ΔC| < this for N frames → converged
    eq_converge_n_frames: int = 20
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
    # c_V components: energy (relative to baseline), MSC, EQ residual
    cv_energy_tau_s: float = 1.0          # baseline / running max EMA
    cv_msc_tau_s: float = 1.0
    cv_eqres_tau_s: float = 1.0
    cv_e_floor_db: float = -60.0          # noise floor reference
    cv_e_full_db: float = -20.0           # nominal full V level
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

    enable_w_local: bool = True           # ablation → pure band weight (no harmonic det)
    use_w_local_pure_band: bool = False   # ablation: w_local ≡ 1 in band (no detection)
    wl_ransac_rounds: int = 3
    wl_outlier_sigma: float = 2.0
    wl_inlier_db: float = 6.0           # fixed RANSAC inlier band (dB)
    wl_r_kill_db: float = 3.0             # ① r[k] < −this ⇒ killed (rkill=3⇒recall0.26/FAR0.11; rkill=6⇒0.13/0.05 on realistic D1)
    wl_slope: float = 1.5                  # sigmoid slope (per dB)
    # --- B0: envelope-model methods (① local-median baseline, ② abrupt-drop
    # signature, ③ absolute-floor gate, ④ V-envelope always-on weak evidence) ---
    wl_method: str = "local_median"        # ransac|local_median|abrupt_drop|combined
    wl_use_local_median: bool = True    # ① (DEFAULT on realistic D1 — best single: recall 0.26 FAR 0.11, below thresh)
    wl_use_abrupt_drop: bool = False    # ②  (misses clustered kills — decay region; recall 0.03 on realistic D1)
    wl_use_abs_gate: bool = False         # ③  (DIAGNOSTIC only now — BR2: must FAIL on realistic D1, was tautological)
    wl_use_v_envelope: bool = False       # ④  (V-shape prior + S-survivor anchor; high FAR on real VPU — envelope flatter than S)
    wl_use_v_eq: bool = False             # ⑤  (CR2: EQ-aligned V′–S direct compare, freq-gated) — the untested info source
    wl_v_eq_thr_db: float = 6.0          # ⑤ S ≪ V′ by this ⇒ killed
    wl_v_eq_slope: float = 3.0
    wl_v_eq_band_hi_hz: float = 800.0    # ⑤ only in VPU usable band (quiet scene); outside V′=noise ⇒ ⑤ off
    wl_local_window: int = 2              # ① k±window
    wl_drop_thr_db: float = 18.0          # ② drop below max-neighbor by this ⇒ killed
    wl_drop_slope: float = 3.0
    wl_abs_floor_db: float = -45.0         # ③ legacy absolute (unused by relative gate)
    wl_abs_headroom_db: float = 45.0      # ③ P < frame_peak−this ⇒ gate ~1 (relative; tuned on 0624)
    wl_abs_slope: float = 3.0
    wl_v_env_slope: float = 4.0           # ④ S≪V ⇒ killed (weak)
    enable_w_local_vfallback: bool = True  # V-envelope fallback (circular-arg risk)
    enable_valley_rule: bool = True         # |Y|_valley = min(|S|,|V'|) between harmonics

    # w time smoothing — TRUE non-symmetric one-sided EMA (升慢降快)
    enable_asym_smooth: bool = True       # ablation → symmetric (use_symmetric_smooth)
    use_symmetric_smooth: bool = False
    w_rise_tau_s: float = 0.060           # 60 ms (more-V rises slowly)
    w_fall_tau_s: float = 0.015           # 15 ms (back-to-S falls fast)

    # harmonic-domain freq smoothing (across k, NOT across bins)
    enable_harm_freq_smooth: bool = True  # ablation → bin-domain smooth
    use_bin_freq_smooth: bool = False
    w_k_smooth: int = 0                    # neighbors in harmonic index k (0=off; the new relative methods need no smoothing — it pulled isolated killed gates down)

    # ===== Layer 3 · synthesis =====
    enable_logclip_mix: bool = True       # ablation → complex convex combination
    use_complex_convex: bool = False
    delta_db: float = 10.0                # clip ±Δ (9–12 dB; placeholder 10)

    enable_comfort_noise: bool = True
    cn_floor_db: float = -60.0
    cn_ema_tau_s: float = 2.0             # streaming min-trace / VAD-gated EMA
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
