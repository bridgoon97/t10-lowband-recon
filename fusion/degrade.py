"""Degradation model D — simulates stage-2's damage on the clean FF ref ``X``
to produce the proxy ``S``.  OFFLINE data-prep, NOT part of the algorithm path:
``degrade`` may use the ORACLE F0 / known harmonic grid (stage-2 "knows" where
harmonics are).  The fusion ALGORITHM estimates its own F0 — the static test
(``tests/test_t13_static.py``) proves ``fusion/`` never imports this module's
internals or the kill mask.

Four INDEPENDENT, individually-switchable factors (spec §2).  Default D1 only.

  D1 谐波杀伤 (主变量): locate harmonics by X's F0 grid, kill the
     WEAKEST-energy fraction (weak→strong order), set to noise floor.  This
     reproduces SI-SNR's "kill weak harmonics" DIRECTION (not uniform random).
     kill_rate ∈ {0, 0.2, 0.4, 0.6}.
  D2 谱对比度压缩: log-domain shrink toward local spectral mean.
  D3 musical noise: sparse random T-F blocks → noise floor.
  D4 时域包络压缩: dynamic-range compression (T1 symptom).

D is applied full-band; fusion only acts on 0–2 kHz (bins 1..64); the 2 kHz
boundary is evaluated separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import FusionConfig
from .stft import stft_batch, istft_batch, _stft_cfg
from lowband.dsp.stft import causal_stft


@dataclass
class DegradationConfig:
    d1_kill_rate: float = 0.0      # 0 / 0.2 / 0.4 / 0.6
    d1_floor_db: float = -60.0    # killed-harmonic level (dB rel peak)
    d1_kill_width_bins: int = 1   # kill ±width around the harmonic bin
    d1_mode: str = "perframe"     # "global" (time-avg energies→fixed set) | "perframe"
    d1_band_hi_hz: float = 2000.0   # B0: sort+kill restricted to this band (in-band)
    # BR1→CR1: PHYSICAL kill-depth parametrization (no hand-picked point).
    # d1_kill_depth_db = mean dB the killed harmonics sit BELOW the boundary
    # (= weakest surviving harmonic) ⇒ physical meaning = stage-2's effective
    # suppression depth beyond the decision boundary.  jitter σ≈2.5 dB.  TRUNCATE
    # enforces killed ≤ boundary (suppression can't make a harmonic louder —
    # the v2 over-correction violated this 34.5% of the time ⇒ overlap 0.974
    # ⇒ constructively unsolvable).  d1_truncate=False is the CR1 mutation.
    d1_kill_depth_db: float = 6.0     # mean depth below boundary (swept 0..30)
    d1_jitter_db: float = 2.5       # per-harmonic jitter σ (dB), smaller than v2's 5
    d1_truncate: bool = True        # killed ≤ boundary (physical monotonicity)
    d1_tautological: bool = False  # CR1/BR2 mutation: revert to frame-peak−60 (v1, ③-inverse)
    # FR3: kill-clustering parametrization.  The deterministic weak-first sort
    # (`sorted(es, key=lambda x: x[2])`) makes the kill set ~maximally clustered
    # (isolated only ~11%) ⇒ ①'s 0.27 ceiling is mostly THIS modeling choice,
    # not the detection problem's difficulty.  Add a slow time-varying random
    # perturbation n(t,k) to the sort key = energy_dB + n(t,k).  n is a 2-D
    # Gaussian field SMOOTHED over time (kernel ~ d1_rank_smooth_s/hop) so it is
    # slow-varying (preserves the natural inter-frame kill-set correlation that
    # deterministic energy ordering gives — per-frame-independent perturbation
    # would create unrealistic flicker).  σ=0 ⇒ n≡0 ⇒ sort by energy_dB ≡ sort by
    # energy (monotone) ⇒ EXACT repro of the old behavior (regression anchor).
    # OFFLINE data-prep (this module is NOT the algorithm path; static-checked).
    d1_rank_sigma_db: float = 0.0   # perturbation σ (dB); 0 = deterministic (current)
    d1_rank_smooth_s: float = 0.15  # time-smoothing kernel (100–200 ms)
    # A6-1b D1 calibration probe (OFFLINE data-prep; default False = current
    # weak-first behavior, no regression).  When True, kill the STRONGEST-energy
    # fraction instead of the weakest — used ONLY to characterize how the kill
    # order maps to band-level deficit std (is the deficit detectable at all?).
    # Not a gate/registry change; production D1 stays weak-first.
    d1_kill_strongest: bool = False
    d2_contrast: float = 0.0      # 0 = off; 1 = full shrink to local mean
    d2_smooth_bins: int = 8
    d3_musical: bool = False
    d3_block_prob: float = 0.02
    d3_block_tbins: int = 2
    d3_block_fbins: int = 3
    d4_envelope: bool = False
    d4_ratio: float = 4.0         # compression ratio
    d4_threshold_db: float = -20.0
    # --- D5 谐波间噪声底注入 (T13-N1): raise INTER-harmonic valleys toward a
    # noise floor E_peak − L dB.  Peaks preserved (orthogonal to D1; stackable).
    # Unvoiced frames (f0 ≤ 0 or conf < d5_conf_thr) are NOT injected and are
    # reported separately by the metrics.  BR1/BR2 anti-tautology discipline
    # does NOT apply — the new fusion has NO detector reading these quantities.
    d5_enable: bool = False
    d5_level_db: float = 40.0       # L: valley depth below harmonic peaks (dB);
                                    # scan 40/30/25/20/15/10 (smaller = dirtier)
    d5_width_bins: int = 1          # ±W around each harmonic = peak region
    d5_shape: str = "white"         # white | pink | vpu (vpu needs d5_vpu_shape)
    d5_vpu_shape: Optional[str] = None  # path to a txt/npy of a magnitude curve
    d5_time: str = "const"          # const | wobble (±1.5 dB slow random)
    d5_conf_thr: float = 0.5        # voiced gate (same op-point as EQ gate)
    d5_band_hi_hz: float = 2000.0   # valleys injected inside the fusion band only
    d5_seed: int = 0
    seed: int = 0


def _bin_hz(cfg: FusionConfig) -> float:
    return cfg.sr / cfg.n_fft


def _harmonic_bins(f0_hz: float, cfg: FusionConfig, k_max: int = 64):
    """FFT-bin indices (full-spectrum, 0-based) of harmonics k·F0 within band."""
    bz = _bin_hz(cfg)
    bins = []
    for k in range(1, k_max + 1):
        f = k * f0_hz
        if f >= cfg.sr / 2:
            break
        b = int(round(f / bz))
        if b >= 1 and b < cfg.n_fft // 2:
            bins.append((k, b))
    return bins


def apply_d1(spec: torch.Tensor, f0_track: torch.Tensor, cfg: FusionConfig,
             deg: DegradationConfig) -> torch.Tensor:
    """D1 harmonic kill (weak→strong), BAND-LIMITED to ``d1_band_hi_hz`` (default
    2 kHz) — sort AND kill happen only in-band, so a 40 % kill actually removes
    40 % of the in-band harmonics (the old full-band sort killed only high-freq
    harmonics >2 kHz, leaving the 0–2 kHz fusion band untouched ⇒ the test
    measured nothing).  ``d1_mode``: perframe (each voiced frame kills its own
    weakest in-band fraction) | global (time-avg).  Returns (out, killed_mask)."""
    B, Fb, N = spec.shape
    out = spec.clone()
    mag = out.abs()
    floor = (10.0 ** (deg.d1_floor_db / 20.0)) * mag.amax(dim=1, keepdim=True).clamp_min(1e-8)  # (B,1,N) per-frame peak-relative
    bz = _bin_hz(cfg)
    band_hi_bin = min(Fb, int(deg.d1_band_hi_hz / bz))
    killed = torch.zeros(B, Fb, N, dtype=torch.bool, device=spec.device)
    # FR3: pre-generate the slow time-varying perturbation field n(t,k) (OFFLINE).
    # σ=0 ⇒ zero field ⇒ exact repro of deterministic weak-first sort.
    if deg.d1_rank_sigma_db > 0:
        rf = np.random.default_rng(int(deg.seed) * 7 + 7919)
        raw = rf.normal(0.0, deg.d1_rank_sigma_db, size=(N, 200))  # k up to 200
        w = max(1, int(round(deg.d1_rank_smooth_s * cfg.sr / cfg.hop)))
        if w > 1:  # moving-average smooth along TIME (reflect-pad) — slow-varying
            raw = np.pad(raw, ((w // 2, w // 2), (0, 0)), mode="reflect")
            ker = np.ones(w) / w
            nf = np.empty_like(raw[:N])
            for kk in range(200):
                nf[:, kk] = np.convolve(raw[:, kk], ker, mode="valid")
        else:
            nf = raw
        s = nf.std()
        if s > 0:
            nf = nf * (deg.d1_rank_sigma_db / s)   # rescale ⇒ effective σ = param
        n_field = nf
    else:
        n_field = None

    def inband_hb(f0):
        return [(k, b) for k, b in _harmonic_bins(f0, cfg) if b <= band_hi_bin]

    for b in range(B):
        if deg.d1_mode == "global":
            harm_energy = {}
            for t in range(N):
                f0 = float(f0_track[b, t])
                if f0 <= 0:
                    continue
                for k, binidx in inband_hb(f0):
                    lo = max(0, binidx - deg.d1_kill_width_bins)
                    hi = min(Fb, binidx + deg.d1_kill_width_bins + 1)
                    e = mag[b, binidx, t].item() if lo == hi - 1 else mag[b, lo:hi, t].max().item()
                    harm_energy.setdefault(k, []).append(e)
            if not harm_energy:
                continue
            mean_e = {k: float(np.mean(v)) for k, v in harm_energy.items()}
            order = sorted(mean_e, key=lambda k: mean_e[k])
            n_kill = int(round(deg.d1_kill_rate * len(order)))
            kill_set = set(order[:n_kill])
            for t in range(N):
                f0 = float(f0_track[b, t])
                if f0 <= 0:
                    continue
                for k, binidx in inband_hb(f0):
                    if k in kill_set:
                        lo = max(0, binidx - deg.d1_kill_width_bins)
                        hi = min(Fb, binidx + deg.d1_kill_width_bins + 1)
                        out[b, lo:hi, t] = out[b, lo:hi, t] / \
                            out[b, lo:hi, t].abs().clamp_min(1e-8) * floor[b, 0, t]
                        killed[b, lo:hi, t] = True
        else:  # perframe
            for t in range(N):
                f0 = float(f0_track[b, t])
                if f0 <= 0:
                    continue
                hb = inband_hb(f0)
                if len(hb) < 3:
                    continue
                es = []
                for k, binidx in hb:
                    lo = max(0, binidx - deg.d1_kill_width_bins)
                    hi = min(Fb, binidx + deg.d1_kill_width_bins + 1)
                    e = mag[b, binidx, t].item() if lo == hi - 1 else mag[b, lo:hi, t].max().item()
                    if n_field is not None:
                        e_db = 20.0 * math.log10(max(e, 1e-12))
                        key = e_db + float(n_field[t, k - 1])   # k is 1-based
                    else:
                        key = e   # σ=0: sort by energy (monotone w/ e_db) ⇒ EXACT repro
                    es.append((k, binidx, e, key))
                order = sorted(es, key=lambda x: x[3])      # weak-first (perturbed)
                if deg.d1_kill_strongest:
                    order = order[::-1]    # calibration probe: strong-first
                n_kill = int(round(deg.d1_kill_rate * len(order)))
                if n_kill == 0:
                    continue
                # CR1: boundary = kill-threshold harmonic energy (≈ weakest
                # survivor).  killed = boundary*10^((-depth+jit)/20); TRUNCATE
                # enforces killed ≤ boundary (suppression can't make a harmonic
                # louder — the v2 over-correction violated this 34.5% of the
                # time).  d1_truncate=False is the CR1 mutation.
                boundary = order[n_kill][2] if n_kill < len(order) else order[-1][2]
                rng = np.random.default_rng(int(deg.seed) * 1000003 + b * 131 + t)
                for k, binidx, _, _ in order[:n_kill]:
                    if deg.d1_tautological:
                        lev = floor[b, 0, t]   # CR1/BR2 mutation: frame-peak−60 (③-inverse)
                    else:
                        jit = float(rng.normal(0, deg.d1_jitter_db))
                        lev = boundary * (10.0 ** ((-deg.d1_kill_depth_db + jit) / 20.0))
                        if deg.d1_truncate:
                            lev = min(lev, boundary)   # physical: killed ≤ weakest survivor
                    lo = max(0, binidx - deg.d1_kill_width_bins)
                    hi = min(Fb, binidx + deg.d1_kill_width_bins + 1)
                    out[b, lo:hi, t] = out[b, lo:hi, t] / \
                        out[b, lo:hi, t].abs().clamp_min(1e-8) * lev
                    killed[b, lo:hi, t] = True
    return out, killed


def apply_d2(spec: torch.Tensor, deg: DegradationConfig) -> torch.Tensor:
    """D2 spectral contrast compression: log-mag shrink toward local spectral mean."""
    if deg.d2_contrast <= 0:
        return spec
    mag = spec.abs().clamp_min(1e-8)
    logm = 20.0 * torch.log10(mag)
    k = deg.d2_smooth_bins
    # local mean along freq (reflect-padded moving average) — OFFLINE sim, OK
    pad = k // 2
    logm_p = F.pad(logm, (0, 0, pad, pad), mode="reflect")
    w = torch.ones(1, 1, 2 * pad + 1, 1, device=spec.device) / (2 * pad + 1)
    local = F.conv2d(logm_p.unsqueeze(1), w, padding=(0, 0)).squeeze(1)
    logm_new = logm + deg.d2_contrast * (local - logm)
    mag_new = 10.0 ** (logm_new / 20.0)
    return spec * (mag_new / mag)


def apply_d3(spec: torch.Tensor, cfg: FusionConfig, deg: DegradationConfig) -> torch.Tensor:
    """D3 musical noise: sparse random T-F blocks → floor."""
    if not deg.d3_musical:
        return spec
    rng = np.random.default_rng(deg.seed)
    B, Fb, N = spec.shape
    out = spec.clone()
    mag = out.abs()
    floor = (10.0 ** (deg.d1_floor_db / 20.0)) * mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    mask = torch.zeros(B, Fb, N, dtype=torch.bool, device=spec.device)
    for b in range(B):
        n_blocks = int(deg.d3_block_prob * Fb * N / (deg.d3_block_fbins * deg.d3_block_tbins))
        for _ in range(n_blocks):
            f0 = rng.integers(1, Fb - deg.d3_block_fbins)
            t0 = rng.integers(0, max(1, N - deg.d3_block_tbins))
            mask[b, f0:f0 + deg.d3_block_fbins, t0:t0 + deg.d3_block_tbins] = True
    out[mask] = (out[mask] / out[mask].abs().clamp_min(1e-8)) * floor.expand_as(out)[mask]
    return out


def apply_d4(x: torch.Tensor, deg: DegradationConfig) -> torch.Tensor:
    """D4 time-domain envelope compression (feedforward compander)."""
    if not deg.d4_envelope:
        return x
    env = x.abs().unfold(-1, 256, 1).amax(-1)
    env = F.pad(env, (255, 0), mode="replicate")
    env = F.avg_pool1d(env, kernel_size=128, stride=1, padding=64)
    env = env[:, :x.shape[-1]]
    db = 20 * torch.log10(env.clamp_min(1e-8))
    over = (db - deg.d4_threshold_db).clamp_min(0)
    gain_db = -over * (1 - 1 / deg.d4_ratio)
    gain = 10 ** (gain_db / 20)
    return x * gain


def apply_d5(spec: torch.Tensor, f0_track: torch.Tensor,
             conf_track: torch.Tensor, cfg: FusionConfig,
             deg: "DegradationConfig"):
    """D5 · inter-harmonic noise-floor injection (T13-N1, OFFLINE data prep).

    Per VOICED frame: mark ±W bins around every harmonic k·F0(t) as the PEAK
    region; all other in-band (≤ d5_band_hi_hz) bins are VALLEY regions.
    E_peak[t] = robust median of |X|² over the peak region; valleys are raised
    to max(|X|, N) with N = sqrt(E_peak)·10^(−L/20) (spectral shape white/pink/
    vpu; time behaviour const/wobble).  Peaks are untouched.  Unvoiced frames
    are left untouched (reported separately by the metrics).

    Returns (out_spec, valley_mask (B,Fb,N) bool, peak_mask, voiced_mask (B,N)).
    OFFLINE ONLY — the algorithm path is static-checked against this module.
    """
    B, Fb, N = spec.shape
    bz = _bin_hz(cfg)
    band_hi_bin = min(Fb, int(deg.d5_band_hi_hz / bz))
    out = spec.clone()
    valley = torch.zeros(B, Fb, N, dtype=torch.bool)
    peak = torch.zeros(B, Fb, N, dtype=torch.bool)
    voiced = torch.zeros(B, N, dtype=torch.bool)
    shape_name = deg.d5_shape
    vpu_curve = None
    if shape_name == "vpu":
        if deg.d5_vpu_shape is None:
            raise ValueError("d5_shape='vpu' needs d5_vpu_shape (npy/txt curve)")
        cur = torch.tensor(np.load(deg.d5_vpu_shape) if str(deg.d5_vpu_shape).endswith(
            ".npy") else np.loadtxt(deg.d5_vpu_shape), dtype=torch.float32)
        vpu_curve = cur.clamp_min(1e-6)
    for b in range(B):
        for t in range(N):
            f0 = float(f0_track[b, t])
            conf = float(conf_track[b, t])
            if f0 <= 0 or conf < deg.d5_conf_thr:
                continue                      # unvoiced: no injection (bucketed)
            voiced[b, t] = True
            pk = torch.zeros(Fb, dtype=torch.bool)
            k = 1
            while k * f0 < deg.d5_band_hi_hz:
                c_bin = int(round(k * f0 / bz))
                lo = max(1, c_bin - deg.d5_width_bins)
                hi = min(Fb - 1, c_bin + deg.d5_width_bins)
                pk[lo:hi + 1] = True
                k += 1
            pk[band_hi_bin:] = False
            pk[0] = False
            vl = torch.zeros(Fb, dtype=torch.bool)
            vl[1:band_hi_bin] = True
            vl &= ~pk
            if pk.sum() == 0 or vl.sum() == 0:
                continue
            peak[b, :, t] = pk
            valley[b, :, t] = vl
            e_peak = float(spec[b, pk, t].abs().pow(2).median())
            n_level = (e_peak ** 0.5) * (10.0 ** (-deg.d5_level_db / 20.0))
            if shape_name == "pink":
                f_ax = torch.arange(Fb, dtype=torch.float32).clamp_min(1) * bz
                curve = (1.0 / f_ax.sqrt())
                curve = curve / curve[vl].median().clamp_min(1e-12)
            elif shape_name == "vpu":
                curve = vpu_curve / vpu_curve[vl].median().clamp_min(1e-12)
            else:
                curve = torch.ones(Fb)
            if deg.d5_time == "wobble":
                rng = np.random.default_rng(int(deg.d5_seed) * 31 + b * 7 + t)
                n_level *= 10.0 ** (float(rng.normal(0.0, 1.5)) / 20.0)
            n_mag = n_level * curve
            sel = vl
            old = out[b, sel, t].abs().clamp_min(1e-8)
            new = torch.maximum(old, n_mag[sel])
            out[b, sel, t] = out[b, sel, t] / old * new
    return out, valley, peak, voiced


def degrade(x: torch.Tensor, cfg: FusionConfig, deg: DegradationConfig,
           f0_track: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply D1–D4 to clean ``X`` (B, T) → proxy ``S`` (B, T)."""
    x = x.float()
    if deg.d4_envelope:
        x = apply_d4(x, deg)
    spec = stft_batch(x, cfg)                              # (B, F, N) complex
    if deg.d1_kill_rate > 0:
        if f0_track is None:
            from .f0 import f0_batch
            f0_track, _ = f0_batch(x, cfg)                 # causal F0 (no oracle given)
        spec, _ = apply_d1(spec, f0_track, cfg, deg)
    if deg.d5_enable:
        from .f0 import f0_batch
        f0_tr, conf_tr = f0_batch(x, cfg)                  # D5 needs the voiced gate
        spec, _, _, _ = apply_d5(spec, f0_tr, conf_tr, cfg, deg)
    spec = apply_d2(spec, deg)
    spec = apply_d3(spec, cfg, deg)
    return istft_batch(spec, cfg, length=x.shape[-1])
