"""T13 A4 — 纯测试侧诊断：w 的四因子信息量与现有公式上界。

BOUNDARY：只使用 0624 男声（F0 中位 87–124 Hz）、正常音量；不读取
0625 的任何语音条目。X 只用于评测集合与 oracle，绝不进入生产路径。
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from fusion import FusionConfig, realdata
from fusion.degrade import DegradationConfig
from fusion.fusion import FusionCore
from fusion.stft import istft_batch, stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_a3 import _g3_stats, _j_stats
from tests.test_t13_b1 import BAND_EDGES_HZ, _band_bins, _need


DEPTHS = [15, 20, 30]
FOCUS_BANDS = [(315, 500), (500, 800)]
FACTORS = ("c_V", "g_f0", "w_band", "w_local", "w_product", "w")


def _oracle_w_scalar(s_log, v_log, x_log, down, up):
    """Exact 1-D piecewise-quadratic optimum w in [0,1] for one band-frame."""
    d = (v_log - s_log).detach().double().cpu().numpy()
    base = (s_log - x_log).detach().double().cpu().numpy()
    breaks = [0.0, 1.0]
    for value in d:
        bp = up / value if value > 0 else (-down / value if value < 0 else -1)
        if 0.0 < bp < 1.0:
            breaks.append(float(bp))
    breaks = sorted(set(breaks)); candidates = set(breaks)
    for a, b in zip(breaks[:-1], breaks[1:]):
        mid = (a + b) / 2
        free = (mid * d >= -down) & (mid * d <= up)
        denom = float(np.sum(d[free] ** 2))
        if denom > 0:
            numerator = float(np.sum(d[free] * base[free]))
            optimum = min(b, max(a, -numerator / denom))
            candidates.add(float(optimum))
        # clipped bins are constant within the interval; no derivative term.
    def loss(w):
        return float(np.mean((base + np.clip(w * d, -down, up)) ** 2))
    return min(candidates, key=loss)


def _run(ff, vpu, cfg, deg, oracle=False):
    """Test-only exact replica of FusionCore, with factor taps and optional oracle w."""
    spec_x, spec_s, s = eval_specs(ff, cfg, deg)
    spec_v = stft_batch(vpu, cfg)
    left = cfg.win - cfg.hop
    frames = F.pad(s.float(), (left, 0)).unsqueeze(1).unfold(
        -1, cfg.win, cfg.hop).squeeze(1)
    core = FusionCore(cfg); y_frames = []
    captured = {key: [] for key in FACTORS}
    for t in range(spec_s.shape[-1]):
        ss = spec_s[:, :, t]; vs = spec_v[:, :, t]; buf = frames[:, t, :]
        f0, conf = core.f0est.estimate(buf)
        smag = ss.abs(); floor = core.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) /
                                 floor.clamp_min(1e-8))).mean(-1)
        vp, startup, reset = core.eq.step(ss, vs, snr, conf)
        eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8))
               - 20 * torch.log10(vs.abs().clamp_min(1e-8))
               - core.eq.C).mean(-1) if core.eq.C is not None else torch.zeros_like(snr)
        cv = core.cv.step(vp, ss, eqr, bool(reset.any()))
        gf = core.gf0.step(conf)
        wb = core.wband.step(vp, ss)
        wl = core.wlocal.step(ss, vp, f0)
        product = cv.unsqueeze(-1) * gf.unsqueeze(-1) * wb * wl
        fw = torch.maximum(startup, reset.float())
        w = core.smooth.step(product * (1 - fw).unsqueeze(-1))
        w_use = w
        if oracle:
            w_use = torch.zeros_like(w)
            sx = 20 * torch.log10(spec_x[:, :, t].abs().clamp_min(1e-8))
            sl = 20 * torch.log10(ss.abs().clamp_min(1e-8))
            vl = 20 * torch.log10(vp.abs().clamp_min(1e-8))
            for bi in range(len(BAND_EDGES_HZ) - 1):
                lo, hi = _band_bins(cfg, BAND_EDGES_HZ[bi], BAND_EDGES_HZ[bi + 1])
                value = _oracle_w_scalar(sl[0, lo:hi + 1], vl[0, lo:hi + 1],
                                         sx[0, lo:hi + 1], cfg.delta_down_db,
                                         cfg.delta_up_db)
                w_use[:, lo:hi + 1] = value
        fb = spec_s.shape[1]
        captured["c_V"].append(cv.unsqueeze(-1).expand(-1, fb).detach())
        captured["g_f0"].append(gf.unsqueeze(-1).expand(-1, fb).detach())
        captured["w_band"].append(wb.detach())
        captured["w_local"].append(wl.detach())
        captured["w_product"].append(product.detach())
        captured["w"].append(w_use.detach())
        y_frames.append(core.synth.step(ss, vp, w_use))
    y = istft_batch(torch.stack(y_frames, -1), cfg, length=s.shape[-1])
    factors = {key: torch.stack(value, -1) for key, value in captured.items()}
    return dict(spec_x=spec_x, spec_s=spec_s, spec_y=stft_batch(y, cfg),
                factors=factors, s=s)


def _auc(values, labels):
    """Tie-aware ROC AUC; labels True=P, False=U."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), float)
    sorted_values = values[order]; start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _separation(values, labels):
    auc = _auc(values, labels)
    return 2 * abs(auc - 0.5) if np.isfinite(auc) else float("nan")


def _collect_stratum(out, cfg, band):
    lo, hi = _band_bins(cfg, *band)
    px = out["spec_x"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
    ps = out["spec_s"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
    sup = (10 * torch.log10(px / ps)).cpu().numpy()
    keep = (sup > 6) | (sup <= 1); labels = sup[keep] > 6
    values = {}
    for key in FACTORS:
        values[key] = out["factors"][key][0, lo:hi + 1].mean(0).cpu().numpy()[keep]
    return values, labels, int(np.sum(sup > 6)), int(np.sum(sup <= 1))


@lru_cache(maxsize=1)
def _measure_factor_bundle():
    _need(); cfg = FusionConfig()
    rows = []; raw = {}
    for depth in DEPTHS:
        collected = {band: {key: [] for key in FACTORS} for band in FOCUS_BANDS}
        collected_labels = {band: [] for band in FOCUS_BANDS}
        counts = {band: [0, 0] for band in FOCUS_BANDS}
        for path in realdata.list_0624():
            ff, vpu, _ = realdata.load_0624(
                name=os.path.basename(path), seg_s=6.0, offset_s=1.0)
            out = _run(ff, vpu, cfg, DegradationConfig(
                d1_kill_rate=0.4, d1_kill_depth_db=float(depth)))
            for band in FOCUS_BANDS:
                values, labels, np_, nu = _collect_stratum(out, cfg, band)
                for key in FACTORS:
                    collected[band][key].append(values[key])
                collected_labels[band].append(labels)
                counts[band][0] += np_; counts[band][1] += nu
        for band in FOCUS_BANDS:
            values = {key: np.concatenate(collected[band][key]) for key in FACTORS}
            labels = np.concatenate(collected_labels[band])
            np_, nu = counts[band]
            for key in FACTORS:
                raw[(depth, band, key)] = (values[key], labels)
                p, u = values[key][labels], values[key][~labels]
                const = np.full_like(values[key], np.median(values[key]))
                rows.append(dict(depth=depth, band=f"{band[0]}-{band[1]}",
                                 factor=key, n_p=np_, n_u=nu,
                                 p50=float(np.median(p)) if len(p) else float("nan"),
                                 p90=float(np.percentile(p, 90)) if len(p) else float("nan"),
                                 pmax=float(np.max(p)) if len(p) else float("nan"),
                                 u50=float(np.median(u)), u90=float(np.percentile(u, 90)),
                                 umax=float(np.max(u)),
                                 auc=_auc(values[key], labels),
                                 sep=_separation(values[key], labels),
                                 const_sep=_separation(const, labels)))
    return rows, raw


def _measure_factor_rows():
    return _measure_factor_bundle()[0]


@lru_cache(maxsize=1)
def _measure_separation_rows():
    _, raw = _measure_factor_bundle(); rows = []
    groups = [("all", DEPTHS, FOCUS_BANDS)]
    groups += [(f"band {lo}-{hi}", DEPTHS, [(lo, hi)]) for lo, hi in FOCUS_BANDS]
    groups += [(f"depth {depth}", [depth], FOCUS_BANDS) for depth in DEPTHS]
    for label, depths, bands in groups:
        for key in FACTORS:
            values = np.concatenate([raw[(d, b, key)][0] for d in depths for b in bands])
            truth = np.concatenate([raw[(d, b, key)][1] for d in depths for b in bands])
            const = np.full_like(values, np.median(values))
            rows.append(dict(group=label, factor=key, n_p=int(truth.sum()),
                             n_u=int((~truth).sum()), auc=_auc(values, truth),
                             sep=_separation(values, truth),
                             const_sep=_separation(const, truth)))
    return rows


def test_A41_factor_decomposition():
    """A4-1：P/U 内四因子、原始乘积与最终 w 的分位数。"""
    rows = _measure_factor_rows()
    print("  A4-1 factor P/U distributions (P50/P90/max | U50/U90/max):")
    for r in rows:
        print(f"  d{r['depth']} {r['band']:>7} {r['factor']:>9} nP/U={r['n_p']}/{r['n_u']} "
              f"P={r['p50']:.4f}/{r['p90']:.4f}/{r['pmax']:.4f} | "
              f"U={r['u50']:.4f}/{r['u90']:.4f}/{r['umax']:.4f}")
    assert all(np.isfinite(r[k]) for r in rows for k in ("u50", "u90", "umax")), (
        "A4-1: non-finite factor distribution")


def test_A42_factor_separation():
    """A4-2：方向无关 separation=2|AUC-0.5|；常数中位替换必须为 0。"""
    rows = _measure_separation_rows()
    print("  A4-2 pooled factor separation (nP<30 is INSUFFICIENT):")
    for r in rows:
        status = "ELIGIBLE" if r["n_p"] >= 30 else "INSUFFICIENT"
        print(f"  {r['group']:>12} {r['factor']:>9} nP/U={r['n_p']}/{r['n_u']} "
              f"{status:12s} AUC={r['auc']:.4f} sep={r['sep']:.4f} "
              f"const={r['const_sep']:.4f}")
    assert all(np.isclose(r["const_sep"], 0.0, atol=1e-12) for r in rows), (
        "A4-2 MR1: a constant factor received non-zero separation credit")


@lru_cache(maxsize=1)
def _measure_oracle_rows():
    _need(); cfg = FusionConfig()
    ff, vpu, _ = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    from fusion.f0 import f0_batch
    _, conf = f0_batch(ff, cfg); rows = []
    for depth in DEPTHS:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        actual = _run(ff, vpu, cfg, deg, oracle=False)
        oracle = _run(ff, vpu, cfg, deg, oracle=True)
        null_g = _g3_stats(actual["spec_x"], actual["spec_s"], actual["spec_s"], cfg)
        null_j = _j_stats(actual["spec_x"], actual["spec_s"], actual["spec_s"], conf, cfg)
        actual_g = _g3_stats(actual["spec_x"], actual["spec_s"], actual["spec_y"], cfg)
        oracle_g = _g3_stats(oracle["spec_x"], oracle["spec_s"], oracle["spec_y"], cfg)
        actual_j = _j_stats(actual["spec_x"], actual["spec_s"], actual["spec_y"], conf, cfg)
        oracle_j = _j_stats(oracle["spec_x"], oracle["spec_s"], oracle["spec_y"], conf, cfg)
        rows.append(dict(depth=depth, actual_g=actual_g, oracle_g=oracle_g,
                         actual_j=actual_j, oracle_j=oracle_j,
                         null_recovery=1.0-null_g["ratio"], null_j3=null_j["j3"]))
    return rows


def test_A43_oracle_w_upper_bound():
    """A4-3：现有 V′、S 锚与 clip 内，每个 band-frame 的最优统一 w。"""
    rows = _measure_oracle_rows()
    print("  A4-3 actual -> oracle-w (post ISTFT/STFT evaluation):")
    print("  depth n_sup G3_actual G3_oracle J3_actual J3_oracle")
    for r in rows:
        print(f"  {r['depth']:>5} {r['oracle_g']['n']:>5} "
              f"{r['actual_g']['ratio']:>9.5f} {r['oracle_g']['ratio']:>9.5f} "
              f"{r['actual_j']['j3']:>9.5f} {r['oracle_j']['j3']:>9.5f}")
    assert all(r["oracle_g"]["ratio"] <= r["actual_g"]["ratio"] + 1e-6
               for r in rows), "A4-3: oracle-w did not improve the G3 ratio"


def test_A4_MR1_null_metrics():
    """MR1：常数因子、Y:=S 的 separation/recovery/J3 得分必须全为 0。"""
    factor_rows = _measure_factor_rows(); oracle_rows = _measure_oracle_rows()
    max_const_sep = max(r["const_sep"] for r in factor_rows)
    max_null_recovery = max(abs(r["null_recovery"]) for r in oracle_rows)
    max_null_j3 = max(abs(r["null_j3"]) for r in oracle_rows)
    print(f"  A4 MR1 null scores: factor-separation={max_const_sep:.3e}, "
          f"G3-recovery={max_null_recovery:.3e}, J3={max_null_j3:.3e}")
    assert max_const_sep == 0.0
    assert max_null_recovery <= 1e-6
    assert max_null_j3 == 0.0
