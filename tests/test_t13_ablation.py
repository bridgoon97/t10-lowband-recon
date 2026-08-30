"""Ablation interface existence (T13-A): each factor switch flips OFF (or to
its ablation variant) and the pipeline still RUNS and produces finite output.

Per spec: "全开 vs 逐项关断的运行通过性清单 (只证明接口可用,不报效果数字)".
No effect metrics are reported here (T13-A reports none).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fusion import Fusion, FusionConfig
from fusion import signals as S


def _run(cfg, label):
    torch.manual_seed(0)
    X = S.harmonic_train(F0=150.0, dur_s=1.0, amps=[1 / k for k in range(1, 9)])
    V = 0.5 * X + 0.2 * torch.randn_like(X)
    f = Fusion(cfg)
    Y = f.process_batch(X, V)
    ok = torch.isfinite(Y).all().item() and Y.shape == X.shape
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:38s} finite+shape {'OK' if ok else 'BAD'}")
    assert ok, f"ablation run failed: {label}"
    return ok


def test_ablation_all():
    """B1 ablation: full-on baseline + each switch flipped one at a time (all RUN,
    finite).  AC1/AC2/AC3 removed: delay_comp, complex_convex, per-harmonic
    w_local; added: eq_mode (frozen/adaptive)."""
    base = FusionConfig()
    results = []
    results.append(_run(base, "ALL-ON (baseline, AC1/2/3)"))
    switches = [
        ("enable_eq=False", dict(enable_eq=False)),
        ("eq_mode=adaptive (vs frozen)", dict(eq_mode="adaptive")),
        ("enable_eq_changepoint=False", dict(enable_eq_changepoint=False)),
        ("enable_c_V=False", dict(enable_c_V=False)),
        ("c_V SNR-only (MSC/EQ-resid off)", dict(cv_legacy_abslevel=False)),  # placeholder; c_V components ablated in test_t13_b1
        ("enable_g_f0=False", dict(enable_g_f0=False)),
        ("enable_w_band=False", dict(enable_w_band=False)),
        ("w_band=fixed_curve", dict(use_w_band_fixed_curve=True)),
        ("enable_w_local=False", dict(enable_w_local=False)),
        ("w_local=pure_band", dict(use_w_local_pure_band=True)),
        ("enable_asym_smooth=False", dict(enable_asym_smooth=False)),
        ("w_smooth=symmetric", dict(use_symmetric_smooth=True)),
        ("enable_comfort_noise=False", dict(enable_comfort_noise=False)),
        ("delta_db=0 (no log-clip)", dict(delta_db=0.0)),
    ]
    for label, kw in switches:
        c = base.with_switches(**kw)
        results.append(_run(c, label))
    n_pass = sum(results)
    print(f"  ablation: {n_pass}/{len(results)} runs PASS (all interfaces usable)")
    assert n_pass == len(results), f"{n_pass}/{len(results)} ablation runs passed"


if __name__ == "__main__":
    test_ablation_all()
    print("ablation interface existence tests: all PASS")
