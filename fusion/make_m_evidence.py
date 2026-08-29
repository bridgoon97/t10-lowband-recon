#!/usr/bin/env python3
"""Generate MECHANISM-evidence PNGs for T13-A (NOT effect conclusions — these
visualize the M1–M7 mechanism-correctness results, which are the T13-A gates).

Outputs to reports/T13/.  Reuses the fusion module + the same synthetic
constructs as tests/test_t13_mechanisms.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusion.config import FusionConfig
from fusion.align import EQAlign
from fusion.decision import CV, GF0, WLocal, AsymSmoother
from fusion.synthesis import logclip_mix, complex_convex
from fusion.stft import StftStreamer
from fusion.f0 import f0_batch
from fusion import signals as S
from fusion.utils import alpha_from_tau
from fusion import Fusion

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reports", "T13")
os.makedirs(OUT, exist_ok=True)


def _harmonic_spectrum(F0, amps, cfg, kill_set=None, floor_db=-60.0):
    Fb = cfg.n_fft // 2 + 1
    bz = cfg.sr / cfg.n_fft
    spec = torch.zeros(1, Fb, dtype=torch.complex64)
    peak = max(amps) if amps else 1.0
    floor = (10 ** (floor_db / 20)) * peak
    for k, a in enumerate(amps, start=1):
        b = int(round(k * F0 / bz))
        if 1 <= b < Fb:
            mag = floor if (kill_set and k in kill_set) else a
            spec[0, b] = mag + 0j
    return spec, bz


def m1_plot():
    cfg = FusionConfig()
    cfg.enable_harm_freq_smooth = False
    wl = WLocal(cfg, v_fallback=False, valley=False)
    F0 = 150.0
    amps = [1 / k for k in range(1, 9)]
    order = sorted(range(len(amps)), key=lambda i: amps[i])
    kill = {order[i] + 1 for i in range(int(round(0.4 * len(amps))))}
    s_spec, bz = _harmonic_spectrum(F0, amps, cfg, kill)
    v_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    w = wl.step(s_spec, v_spec, torch.tensor([F0]))
    ks = list(range(1, 9))
    bins = [int(round(k * F0 / bz)) for k in ks]
    wv = [w[0, b].item() for b in bins]
    colors = ["red" if k in kill else "green" for k in ks]
    plt.figure(figsize=(7, 4))
    plt.bar(ks, wv, color=colors)
    plt.axhline(0.5, ls="--", color="k", lw=0.8, label="flag threshold 0.5")
    plt.xlabel("harmonic index k")
    plt.ylabel("w_local")
    plt.title("M1: w_local at harmonics (red=killed, green=surviving), D1=40%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m1_wlocal.png"), dpi=110)
    plt.close()
    print("  m1_wlocal.png")


def m2_plot():
    cfg = FusionConfig()
    s_t, v_t = S.tilted_noise_pair(dur_s=3.0, tilt_db=6.0)
    sfr_s = StftStreamer(cfg); sfr_v = StftStreamer(cfg)
    eq = EQAlign(cfg, changepoint_enabled=False)
    bz = cfg.sr / cfg.n_fft
    lo = max(1, int(100 / bz)); hi = min(cfg.n_fft // 2 - 1, int(2000 / bz))
    errs = []
    for i in range(0, s_t.shape[-1], cfg.hop):
        sh = s_t[:, i:i + cfg.hop]; vh = v_t[:, i:i + cfg.hop]
        if sh.shape[-1] < cfg.hop:
            break
        ss = sfr_s.step(sh); vs = sfr_v.step(vh)
        eq.step(ss, vs, torch.full((1, ss.shape[-1]), 30.0), torch.tensor([0.95]))
        idx = torch.arange(lo, hi + 1)
        true = -6.0 * (idx - lo).float() / max(1, (hi - lo))
        errs.append((eq.C[0, lo:hi + 1] - true).abs().max().item())
    plt.figure(figsize=(7, 4))
    t = np.arange(len(errs)) * cfg.hop / cfg.sr
    plt.plot(t, errs, lw=1.5)
    plt.axhline(1.0, ls="--", color="r", label="gate ±1 dB")
    plt.axvline(3.0, ls=":", color="k", label="3 s")
    plt.xlabel("time (s)")
    plt.ylabel("max |C[f] − true tilt| (dB)")
    plt.title("M2: EQ C[f] causal convergence to known ±6 dB tilt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m2_eq_convergence.png"), dpi=110)
    plt.close()
    print("  m2_eq_convergence.png")


def m3_plot():
    cfg = FusionConfig()
    F0 = 150.0; amps = [1 / k for k in range(1, 9)]
    v_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    s_spec, _ = _harmonic_spectrum(F0, amps, cfg)
    cv = CV(cfg)
    c = []
    for db in [0.0, -3.0, -6.0, -12.0]:
        v_att = v_spec * (10 ** (db / 20.0))
        for _ in range(200):
            cv.step(v_att, s_spec, torch.zeros(1, v_spec.shape[1]))
        c.append(cv.c_v)
    plt.figure(figsize=(6, 4))
    plt.bar(["0", "-3", "-6", "-12"], c, color="steelblue")
    plt.xlabel("V attenuation (dB)")
    plt.ylabel("c_V (steady-state)")
    plt.title("M3: c_V strictly monotone non-increasing as V weakens")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m3_cv_monotone.png"), dpi=110)
    plt.close()
    print("  m3_cv_monotone.png")


def m4_plot():
    cfg = FusionConfig()
    sm = AsymSmoother(cfg)
    n_pre = int(1.0 * cfg.sr / cfg.hop); n_on = int(2.0 * cfg.sr / cfg.hop)
    n_post = int(1.0 * cfg.sr / cfg.hop)
    xs = [torch.zeros(1)] * n_pre + [torch.ones(1)] * n_on + [torch.zeros(1)] * n_post
    ys = np.array([sm.step(x.clone()).item() for x in xs])
    t = np.arange(len(ys)) * cfg.hop / cfg.sr * 1e3
    plt.figure(figsize=(8, 4))
    plt.plot(t, ys, lw=1.5)
    plt.axvline(n_pre * cfg.hop / cfg.sr * 1e3, ls=":", color="k")
    plt.axvline((n_pre + n_on) * cfg.hop / cfg.sr * 1e3, ls=":", color="k")
    plt.xlabel("time (ms)")
    plt.ylabel("w (step response)")
    plt.title("M4: non-symmetric w smoother (slow rise ~130ms, fast fall ~30ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m4_asym.png"), dpi=110)
    plt.close()
    print("  m4_asym.png")


def m7_plot():
    cfg = FusionConfig()
    Fb = cfg.n_fft // 2 + 1; b = 10; A = 1.0
    v_spec = torch.zeros(1, Fb, dtype=torch.complex64); v_spec[0, b] = A + 0j
    s_spec = torch.zeros(1, Fb, dtype=torch.complex64); s_spec[0, b] = A * 1j
    w = torch.full((1, Fb), 0.5)
    yl = logclip_mix(s_spec, v_spec, w, cfg.delta_db)[0, b].abs().item()
    yc = complex_convex(s_spec, v_spec, w)[0, b].abs().item()
    plt.figure(figsize=(6, 4))
    plt.bar(["log-clip mix", "complex convex"], [yl, yc], color=["green", "red"])
    plt.axhline(1.0, ls="--", color="k", lw=0.8, label="|V| (0 dB ref)")
    plt.ylabel("|Y| (relative)")
    plt.title("M7: 90° phase mismatch, w=0.5 — log-clip holds 0 dB, convex dips ~3 dB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m7_energy_dip.png"), dpi=110)
    plt.close()
    print("  m7_energy_dip.png")


def m5_plot():
    cfg = FusionConfig()
    cfg2 = FusionConfig(); cfg2.enable_c_V = False; cfg2.enable_w_local = False
    x = S.voiced_unvoiced(F0=150.0, dur_s=4.0)
    f = Fusion(cfg2); f.process_batch(x, x)
    _, conf = f0_batch(x, cfg2)
    wh = torch.stack(f.core.w_history, dim=-1)[0].mean(0).numpy()
    c = conf[0].numpy()
    t = np.arange(len(wh)) * cfg.hop / cfg.sr
    plt.figure(figsize=(9, 4))
    plt.plot(t, wh / wh.max(), label="w (band-mean, norm)", lw=1.2)
    plt.plot(t, c, label="f0_confidence (1−CMND)", lw=1.0, alpha=0.7)
    plt.xlabel("time (s)")
    plt.title("M5: w tracks f0_confidence (voiced frames → higher w; direction correct)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "m5_g_f0_direction.png"), dpi=110)
    plt.close()
    print("  m5_g_f0_direction.png")


def r4_plot():
    """R4: real in-band harmonic envelope (one voiced frame) — killed vs
    surviving, and which w_local flags.  Visualizes the BELOW-THRESHOLD result:
    formant-valley SURVIVING harmonics get mis-flagged (FAR up), killed ones
    partly missed (recall down) because the linear-across-k envelope fit is too
    rigid for real formant undulation.  Honest B-stage input, not a gate."""
    cfg = FusionConfig(); cfg.enable_harm_freq_smooth = False
    wl = WLocal(cfg, v_fallback=False, valley=False)
    from fusion import realdata
    from fusion.stft import stft_batch
    from fusion.f0 import f0_batch
    ff, vpu, sr = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    specX = stft_batch(ff, cfg); specV = stft_batch(vpu, cfg)
    f0tr, conftr = f0_batch(ff, cfg)
    bz = cfg.sr / cfg.n_fft
    # apply_d1 realistic (killed ≈ weakest-survivor level; ① limited)
    from fusion.degrade import apply_d1, DegradationConfig
    deg = DegradationConfig(d1_kill_rate=0.4)
    f0tr, conftr = f0_batch(ff, cfg)
    specS, killed = apply_d1(specX, f0tr, cfg, deg)
    # pick a strongly voiced frame
    t = None
    for i in range(specS.shape[-1]):
        if conftr[0, i] > 0.6 and f0tr[0, i] > 0:
            t = i; break
    f0 = float(f0tr[0, t])
    kb = [(k, int(round(k * f0 / bz))) for k in range(1, 64)
          if 1 <= int(round(k * f0 / bz)) <= cfg.fusion_hi_bin]
    P = [20 * torch.log10(specX[0, b, t].abs().clamp_min(1e-8)).item() for k, b in kb]
    Pd = [20 * torch.log10(specS[0, b, t].abs().clamp_min(1e-8)).item() for k, b in kb]
    kill = [i for i, (k, b) in enumerate(kb) if bool(killed[0, b, t])]
    w = wl.step(specS[:, :, t], specV[:, :, t], torch.tensor([f0]))[0]
    ks = [kb[i][0] for i in range(len(kb))]
    wv = [w[kb[i][1]].item() for i in range(len(kb))]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    cols = ["red" if i in kill else "green" for i in range(len(kb))]
    ax[0].bar(ks, P, color=["orange" if i in kill else "steelblue" for i in range(len(kb))], alpha=0.6)
    ax[0].bar(ks, Pd, color=cols, alpha=0.9)
    ax[0].set_ylabel("harmonic level (dB)")
    ax[0].set_title(f"R4 realistic D1 (f0={f0:.0f}Hz): killed≈weakest-survivor level (overlap) — ① limited")
    ax[1].bar(ks, wv, color=cols)
    ax[1].axhline(0.5, ls="--", color="k", lw=0.8)
    ax[1].set_ylabel("w_local")
    ax[1].set_xlabel("harmonic index k")
    ax[1].set_title("w_local flags (red=killed, green=surviving); green>0.5 = FAR, red<0.5 = recall miss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "r4_real_envelope.png"), dpi=110)
    plt.close()
    print("  r4_real_envelope.png")


def cr1_sweep_plot():
    """CR1 sweep: recall/FAR = f(kill_depth) for ①②③④⑤."""
    import tests.test_t13_real as R
    cfg = FusionConfig()
    methods = [("1 local-med", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
              ("2 abrupt", dict(wl_use_local_median=False, wl_use_abrupt_drop=True, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=False)),
              ("3 abs-gate", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=True, wl_use_v_envelope=False, wl_use_v_eq=False)),
              ("4 V-shape", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=True, wl_use_v_eq=False)),
              ("5 V'eq(in-band)", dict(wl_use_local_median=False, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True)),
              ("1v5 parallel", dict(wl_use_local_median=True, wl_use_abrupt_drop=False, wl_use_abs_gate=False, wl_use_v_envelope=False, wl_use_v_eq=True, wl_combine="or"))]
    depths = [0, 3, 6, 10, 15, 20, 30]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, kw in methods:
        rec, far = [], []
        for d in depths:
            deg = __import__("fusion.degrade", fromlist=["DegradationConfig"]).DegradationConfig(
                d1_kill_rate=0.4, d1_kill_depth_db=d)
            cb = 800.0 if "in-band" in label else None
            r, f, _, _, _ = R._r4_recall_far(cfg.with_switches(**kw), deg=deg, count_band_hi_hz=cb)
            rec.append(r); far.append(f)
        ax[0].plot(depths, rec, "-o", label=label, lw=1.3)
        ax[1].plot(depths, far, "-o", label=label, lw=1.3)
    ax[0].axhline(0.90, ls="--", color="k", lw=0.7); ax[0].set_title("recall vs kill_depth")
    ax[1].axhline(0.10, ls="--", color="k", lw=0.7); ax[1].set_title("FAR vs kill_depth")
    for a in ax:
        a.set_xlabel("d1_kill_depth_db"); a.legend(fontsize=8); a.set_ylim(-0.02, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cr1_sweep.png"), dpi=110)
    plt.close()
    print("  cr1_sweep.png")


if __name__ == "__main__":
    print("generating T13-A mechanism-evidence PNGs ->", OUT)
    m1_plot(); m2_plot(); m3_plot(); m4_plot(); m5_plot(); m7_plot(); r4_plot(); cr1_sweep_plot()
    print("done")
