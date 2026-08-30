"""T13 A3 — 统一时域 S 的评测参照，并验证度量不奖励“什么也不做”。

所有 X 依赖仅存在于评测侧；生产算法不读取 X。

BOUNDARY：结论只适用于 0624 的四名男声说话人（F0 中位 87–124 Hz）、
正常音量；未读取 0625 的任何语音条目。
"""
from __future__ import annotations

import ast
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from fusion import Fusion, FusionConfig, realdata
from fusion.degrade import DegradationConfig, apply_d1
from fusion.f0 import f0_batch
from fusion.stft import istft_batch, stft_batch
from tests._t13_eval import eval_specs
from tests.test_t13_b1 import BAND_EDGES_HZ, G3A_MIN_SAMPLES, _band_bins, _need


DEPTHS = [0, 3, 6, 10, 15, 20, 30]
REPORT_DIR = Path("reports/T13A3")


def _validate_apply_d1_flow(source: str, filename: str = "<source>"):
    """禁止 D1 直接返回谱进入 log/abs 评测；mutation 函数是刻意破坏的例外。"""
    tree = ast.parse(source, filename=filename)
    errors = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if "mutation" in fn.name.lower():
            continue
        parents = {}
        for parent in ast.walk(fn):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "apply_d1"):
                continue
            assign = parents.get(node)
            if not isinstance(assign, ast.Assign) or not assign.targets:
                errors.append(f"{filename}:{node.lineno}: apply_d1 返回值未显式绑定")
                continue
            target = assign.targets[0]
            if not isinstance(target, (ast.Tuple, ast.List)) or not target.elts \
                    or not isinstance(target.elts[0], ast.Name):
                errors.append(f"{filename}:{node.lineno}: 无法审计 apply_d1 第一返回值")
                continue
            name = target.elts[0].id
            if name == "_":
                continue
            for use in [n for n in ast.walk(fn)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                        and n.id == name]:
                parent = parents.get(use)
                in_istft = (isinstance(parent, ast.Call)
                            and isinstance(parent.func, ast.Name)
                            and parent.func.id == "istft_batch"
                            and parent.args and parent.args[0] is use)
                direct_abs = (isinstance(parent, ast.Attribute) and parent.attr == "abs"
                              and isinstance(parents.get(parent), ast.Call))
                ancestor = parent; in_log10 = False
                while ancestor is not None and not isinstance(ancestor, ast.stmt):
                    if (isinstance(ancestor, ast.Call)
                            and ((isinstance(ancestor.func, ast.Name) and ancestor.func.id == "log10")
                                 or (isinstance(ancestor.func, ast.Attribute)
                                     and ancestor.func.attr == "log10"))):
                        in_log10 = True
                    ancestor = parents.get(ancestor)
                helper_requires_istft = Path(filename).name == "_t13_eval.py"
                if direct_abs or in_log10 or (helper_requires_istft and not in_istft):
                    errors.append(
                        f"{filename}:{use.lineno}: D1 往返前谱 {name} 流入 log/abs 评测或绕过 ISTFT")
    assert not errors, "A3-0 评测参照静态检查失败：" + "; ".join(errors)


def test_A30_single_reference_static_guard():
    """A3-0：测试侧评测 helper 中 D1 谱只能直接流向 ISTFT。"""
    root = Path(__file__).resolve().parent
    targets = [root / "_t13_eval.py", *sorted(root.glob("test_*.py"))]
    for path in targets:
        _validate_apply_d1_flow(path.read_text(encoding="utf-8"), str(path))
    assert "spec_s = stft_batch(s, cfg)" in (root / "_t13_eval.py").read_text(encoding="utf-8"), (
        "A3-0: shared helper no longer re-analyses the time-domain S")


def test_A30_reference_guard_mutation():
    """Mutation：把 D1 往返前谱直接送入 log/abs 评测，静态防线必须失败。"""
    mutant = '''
def bad_metric(ff, cfg, deg):
    x = stft_batch(ff, cfg)
    f0, _ = f0_batch(ff, cfg)
    pre, _ = apply_d1(x, f0, cfg, deg)
    s = istft_batch(pre, cfg, length=ff.shape[-1])
    wrong = 20 * torch.log10(pre.abs())
    return wrong, s
'''
    caught = False; failure = ""
    try:
        _validate_apply_d1_flow(mutant, "A3-0-mutant.py")
    except AssertionError as exc:
        caught = True; failure = str(exc)
    print(f"  A3-0 mutation: 新增 `wrong=20log10(pre.abs())`；失败={failure}")
    assert caught, "A3-0 mutation: pre-roundtrip evaluation reference escaped the guard"


def _g3_stats(spec_x, spec_s, spec_y, cfg):
    lsd_s, lsd_y, lsd_o = [], [], []
    delivered = []
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        xs = 20 * torch.log10(spec_x[0, lo:hi + 1].abs().clamp_min(1e-8))
        ss = 20 * torch.log10(spec_s[0, lo:hi + 1].abs().clamp_min(1e-8))
        ys = 20 * torch.log10(spec_y[0, lo:hi + 1].abs().clamp_min(1e-8))
        px = spec_x[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        ps = spec_s[0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        sup = 10 * torch.log10(px / ps)
        delivered.extend(sup.cpu().tolist())
        for t in torch.nonzero(sup > 6.0, as_tuple=False).flatten().tolist():
            lsd_s.append(float(torch.sqrt(((ss[:, t] - xs[:, t]) ** 2).mean())))
            lsd_y.append(float(torch.sqrt(((ys[:, t] - xs[:, t]) ** 2).mean())))
            g_star = (xs[:, t] - ss[:, t]).mean()
            lsd_o.append(float(torch.sqrt(((ss[:, t] + g_star - xs[:, t]) ** 2).mean())))
    mean_s = float(np.mean(lsd_s)) if lsd_s else 0.0
    ratio = float(np.mean(lsd_y)) / max(1e-3, mean_s) if lsd_s else float("nan")
    oracle = float(np.mean(lsd_o)) / max(1e-3, mean_s) if lsd_s else float("nan")
    gap = ((ratio - oracle) / max(1e-8, 1 - oracle)) if lsd_s else float("nan")
    a = np.asarray(delivered)
    return dict(n=len(lsd_s), ratio=ratio, oracle=oracle, norm_gap=gap,
                p50=float(np.percentile(a, 50)), p75=float(np.percentile(a, 75)),
                p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                max=float(a.max()), gt1=float(np.mean(a > 1)),
                gt3=float(np.mean(a > 3)), gt6=float(np.mean(a > 6)),
                gt10=float(np.mean(a > 10)))


def _j_stats(spec_x, spec_s, spec_y, conf, cfg):
    sup_c, unsup_c, sup_def, sup_rec = [], [], [], []
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        for t in range(spec_s.shape[-1]):
            if float(conf[0, t]) < 0.55:
                continue
            xs = 20 * torch.log10(spec_x[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_s[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            corr = float((ys - ss).abs().mean())
            deficit = float((xs - ss).abs().mean())
            if float((ss - xs).mean()) < -6.0:
                sup_c.append(corr); sup_def.append(deficit); sup_rec.append(min(corr, deficit))
            else:
                unsup_c.append(corr)
    return dict(
        j1=float(np.mean(np.asarray(sup_c) > 3.0)) if sup_c else 0.0,
        j2=float(np.mean(np.asarray(unsup_c) > 3.0)) if unsup_c else 0.0,
        j3=float(np.sum(sup_rec) / max(1.0, np.sum(sup_def))) if sup_def else 0.0,
        n_sup=len(sup_c), n_unsup=len(unsup_c))


def _kr0_stats(spec_x, spec_s, spec_y, conf, cfg):
    """复刻 KR0 的选择集与 0.1 dB 容差，报告而不改变门槛。"""
    violations = 0; n = 0; improvements = []; worst = -float("inf")
    for i in range(len(BAND_EDGES_HZ) - 1):
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1])
        for t in range(spec_s.shape[-1]):
            if float(conf[0, t]) < 0.55:
                continue
            xs = 20 * torch.log10(spec_x[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ss = 20 * torch.log10(spec_s[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            ys = 20 * torch.log10(spec_y[0, lo:hi + 1, t].abs().clamp_min(1e-8))
            if float((ss - xs).mean()) < -6.0:
                lsd_s = float(torch.sqrt(((ss - xs) ** 2).mean()))
                lsd_y = float(torch.sqrt(((ys - xs) ** 2).mean()))
                bound = float((ys - ss).abs().max())
                excess = abs(lsd_y - lsd_s) - bound
                violations += int(excess > 0.1); n += 1
                worst = max(worst, excess); improvements.append(lsd_s - lsd_y)
    return dict(n=n, violations=violations,
                mean_improvement=float(np.mean(improvements)) if improvements else 0.0,
                worst=worst if n else float("nan"))


def _pre_roundtrip_mutation_specs(ff, cfg, deg):
    """仅供 mutation：故意复刻 A3 修复前的错误参照。"""
    spec_x = stft_batch(ff, cfg)
    f0, _ = f0_batch(ff, cfg)
    pre, _ = apply_d1(spec_x, f0, cfg, deg)
    s = istft_batch(pre, cfg, length=ff.shape[-1])
    return spec_x, pre, s, stft_batch(s, cfg)


@lru_cache(maxsize=1)
def _measure_before_after():
    _need(); cfg = FusionConfig()
    ff, vpu, _ = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    _, conf = f0_batch(ff, cfg)
    rows = []
    for depth in DEPTHS:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        spec_x, pre, s, post = _pre_roundtrip_mutation_specs(ff, cfg, deg)
        y = Fusion(cfg).process_batch(s, vpu); spec_y = stft_batch(y, cfg)
        rows.append(dict(depth=depth,
                         old_g3=_g3_stats(spec_x, pre, spec_y, cfg),
                         new_g3=_g3_stats(spec_x, post, spec_y, cfg),
                         old_j=_j_stats(spec_x, pre, spec_y, conf, cfg),
                         new_j=_j_stats(spec_x, post, spec_y, conf, cfg),
                         old_kr0=_kr0_stats(spec_x, pre, spec_y, conf, cfg),
                         new_kr0=_kr0_stats(spec_x, post, spec_y, conf, cfg)))
    return rows


def _plot_reference_correction(rows):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    eligible = [r for r in rows if r["new_g3"]["n"] >= G3A_MIN_SAMPLES]
    depths = [r["depth"] for r in eligible]
    series = [
        ("actual before", [r["old_g3"]["ratio"] for r in eligible], "#9DC3E6"),
        ("actual after", [r["new_g3"]["ratio"] for r in eligible], "#2F5597"),
        ("oracle before", [r["old_g3"]["oracle"] for r in eligible], "#A9D18E"),
        ("oracle after", [r["new_g3"]["oracle"] for r in eligible], "#548235"),
    ]
    width, height = 900, 540; x0, y0, pw, ph = 80, 55, 740, 390
    xmax = max(depths); ymax = 1.05
    x = lambda d: x0 + d / xmax * pw
    y = lambda v: y0 + ph - v / ymax * ph
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="450" y="28" text-anchor="middle" font-size="20">A3 evaluation-reference correction</text>',
           f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#666"/>',
           f'<line x1="{x0}" y1="{y(.5):.2f}" x2="{x0+pw}" y2="{y(.5):.2f}" stroke="#C00000" stroke-dasharray="6,4"/>']
    for idx, (label, values, color) in enumerate(series):
        points = " ".join(f"{x(d):.2f},{y(v):.2f}" for d, v in zip(depths, values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for d, v in zip(depths, values):
            svg.append(f'<circle cx="{x(d):.2f}" cy="{y(v):.2f}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{x0+15+180*(idx%2)}" y="{475+22*(idx//2)}" fill="{color}" font-size="14">{label}</text>')
    for d in depths:
        svg.append(f'<text x="{x(d):.2f}" y="{y0+ph+20}" text-anchor="middle" font-size="12">{d}</text>')
    svg.extend(['<text x="450" y="535" text-anchor="middle" font-size="14">commanded D1 depth (delivered suppression reported separately)</text>', '</svg>'])
    (REPORT_DIR / "reference_correction.svg").write_text("\n".join(svg), encoding="utf-8")


def test_A31_metric_null_reference():
    """MR1：Y:=S 不得被任何恢复/介入度量记成功劳。"""
    _need(); cfg = FusionConfig()
    ff, _, _ = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    _, conf = f0_batch(ff, cfg)
    for depth in [15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        spec_x, spec_s, _ = eval_specs(ff, cfg, deg)
        g3 = _g3_stats(spec_x, spec_s, spec_s, cfg)
        jm = _j_stats(spec_x, spec_s, spec_s, conf, cfg)
        print(f"  MR1 d{depth}: n_sup={g3['n']} G3={g3['ratio']:.8f} "
              f"J1={jm['j1']:.8f} J3={jm['j3']:.8f}")
        assert np.isclose(g3["ratio"], 1.0, rtol=1e-6), (
            f"MR1: Y=S was credited G3 ratio {g3['ratio']} at depth {depth}")
        assert jm["j1"] == 0.0, f"MR1: Y=S was credited J1={jm['j1']}"
        assert jm["j3"] == 0.0, f"MR1: Y=S was credited J3={jm['j3']}"


def test_A31_metric_null_mutation():
    """Mutation：参照换回 D1 原谱，MR1 必须以被误记的数值失败。"""
    _need(); cfg = FusionConfig()
    ff, _, _ = realdata.load_0624(seg_s=6.0, offset_s=1.0)
    _, conf = f0_batch(ff, cfg); failures = []
    for depth in [15, 20, 30]:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        spec_x, pre, _, post = _pre_roundtrip_mutation_specs(ff, cfg, deg)
        g3 = _g3_stats(spec_x, pre, post, cfg)
        jm = _j_stats(spec_x, pre, post, conf, cfg)
        if not np.isclose(g3["ratio"], 1.0, rtol=1e-6) or jm["j1"] != 0 or jm["j3"] != 0:
            failures.append((depth, g3["ratio"], jm["j1"], jm["j3"]))
    print("  MR1 mutation：把 spec_S 改回 apply_d1 原谱；被误记值=" +
          ", ".join(f"d{d}:G3={g:.5f}/J1={j1:.3f}/J3={j3:.3f}"
                    for d, g, j1, j3 in failures))
    assert failures, "MR1 mutation: wrong pre-roundtrip reference was not detected"


def test_A32_corrected_before_after_report():
    """A3-2：逐 depth 报 G3/oracle、送达压制及 J1/J2/J3 修正前后。"""
    rows = _measure_before_after(); _plot_reference_correction(rows)
    print("  A3-2 G3a'/oracle（commanded depth；送达量用 p50/p95/max 与 n_sup 表示）:")
    print("  cmd old_n->new_n actual old->new oracle old->new gap old->new delivered p50/p95/max")
    for r in rows:
        o, n = r["old_g3"], r["new_g3"]
        print(f"  {r['depth']:>3} {o['n']:>4}->{n['n']:<4} {o['ratio']:>7.5f}->{n['ratio']:<7.5f} "
              f"{o['oracle']:>7.5f}->{n['oracle']:<7.5f} {o['norm_gap']:>7.5f}->{n['norm_gap']:<7.5f} "
              f"{n['p50']:.2f}/{n['p95']:.2f}/{n['max']:.2f}")
    print("  A3-2 J metrics old(pre-roundtrip)->new(post-roundtrip):")
    for r in rows:
        o, n = r["old_j"], r["new_j"]
        print(f"  d{r['depth']}: J1 {o['j1']:.3f}->{n['j1']:.3f}; "
              f"J2 {o['j2']:.3f}->{n['j2']:.3f}; J3 {o['j3']:.3f}->{n['j3']:.3f}; "
              f"n_sup {o['n_sup']}->{n['n_sup']}")
    print("  A3-2 KR0 old(pre-roundtrip)->new(post-roundtrip), unchanged tol=0.1 dB:")
    for r in rows:
        o, n = r["old_kr0"], r["new_kr0"]
        print(f"  d{r['depth']}: n {o['n']}->{n['n']}; mean improvement "
              f"{o['mean_improvement']:.3f}->{n['mean_improvement']:.3f}; "
              f"violations {o['violations']}->{n['violations']}")
    print(f"  correction plot → {REPORT_DIR / 'reference_correction.svg'}")
    finite = all(np.isfinite(v) for r in rows for side in (r["old_j"], r["new_j"])
                 for v in (side["j1"], side["j2"], side["j3"]))
    assert finite, "A3-2: corrected intervention table contains non-finite values"


def _oracle_for_record(name, depth, cfg):
    ff, _, _ = realdata.load_0624(name=name, seg_s=6.0, offset_s=1.0)
    spec_x, spec_s, _ = eval_specs(
        ff, cfg, DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=float(depth)))
    return _g3_stats(spec_x, spec_s, spec_s, cfg)


def test_A33_oracle_cross_record_robustness():
    """A3-3：0624 全十条；只聚合满足 n_sup>=30 的非退化集合。"""
    _need(); cfg = FusionConfig(); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in [15, 20, 30]:
        vals = []
        for path in realdata.list_0624():
            stats = _oracle_for_record(os.path.basename(path), depth, cfg)
            if stats["n"] >= G3A_MIN_SAMPLES:
                vals.append(stats["oracle"])
        a = np.asarray(vals, dtype=np.float64)
        row = dict(depth=depth, n=len(a), minimum=float(a.min()), median=float(np.median(a)),
                   maximum=float(a.max()), std=float(a.std()), below=float(np.mean(a < 0.5)))
        rows.append(row)
        print(f"  A3-3 d{depth}: eligible={len(a)} oracle min={row['minimum']:.5f} "
              f"med={row['median']:.5f} max={row['maximum']:.5f} std={row['std']:.5f} "
              f"<0.5={row['below']:.0%}")
    assert all(r["n"] > 0 for r in rows), "A3-3: no recording has an adequate P set"
