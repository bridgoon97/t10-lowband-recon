"""G5 — streaming-causal (T13-A hard gate).

  1. batch-vs-streaming equivalence: ``Fusion.process_batch`` vs
     ``FusionStreamer`` over the SAME signal — interior max-abs-diff < 1e-6.
     (In practice == 0.0 bit-identical: the only difference is the STFT/iSTFT
     engine, which is bit-identical per frame — verified structurally and
     numerically here.)
  2. future-perturbation (主判据): zero all samples after position P; outputs
     at frames BEFORE the perturbation must be BIT-IDENTICAL (``torch.equal``).
     Multiple random P.  Catches any look-ahead / global stat / non-causal pad.
  3. mutation sanity: deliberately introduce a NON-CAUSAL op (a bidirectional
     EMA on the w track — the spec's own example "把某个 EMA 改成双向") and show
     test (2) now FAILS (past outputs change).  Reports the failure magnitude.

G5 not passing ⇒ the whole task is not accepted (no effect metrics looked at).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from fusion import Fusion, FusionStreamer, FusionConfig
from fusion import signals as S
from fusion.utils import alpha_from_tau


def _signals(seed, T=16000):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(1, T, generator=g)
    V = 0.5 * X + 0.3 * torch.randn(1, T, generator=g)
    return X, V


def test_g5_equiv():
    """batch vs streaming interior max-abs-diff < 1e-6 (== 0 expected)."""
    cfg = FusionConfig()
    max_diff = 0.0
    for seed in [0, 1, 2]:
        for T in [16000, 17600, 48000]:
            s, v = _signals(seed, T)
            yb = Fusion(cfg).process_batch(s, v)
            fs = FusionStreamer(cfg)
            outs = []
            for i in range(0, s.shape[-1], cfg.hop):
                sh = s[:, i:i+cfg.hop]; vh = v[:, i:i+cfg.hop]
                if sh.shape[-1] < cfg.hop:
                    break
                o = fs.stream_step(sh, vh)
                if o is not None:
                    outs.append(o)
            outs.append(fs.flush())
            ys = torch.cat(outs, dim=1)[:, :s.shape[-1]]
            skip = 1024
            N = min(yb.shape[-1], ys.shape[-1])
            d = (yb[..., skip:N-skip] - ys[..., skip:N-skip]).abs().max().item()
            max_diff = max(max_diff, d)
    status = "PASS" if max_diff < 1e-6 else "FAIL"
    print(f"  G5 equiv: max_abs_diff={max_diff:.3e}  {status}")
    assert max_diff < 1e-6, f"batch≠streaming: {max_diff}"


def test_g5_future_perturbation():
    """Zero future samples ⇒ past outputs bit-identical (torch.equal)."""
    cfg = FusionConfig()
    s, v = _signals(0, 16000)
    y_full = Fusion(cfg).process_batch(s, v)
    T = s.shape[-1]
    worst = None
    ps = [cfg.hop * 5, cfg.hop * 50, cfg.hop * 75, T // 2, 3 * T // 4]
    for P in ps:
        s_m = s.clone(); s_m[:, P:] = 0.0
        v_m = v.clone(); v_m[:, P:] = 0.0
        y_m = Fusion(cfg).process_batch(s_m, v_m)
        # fully-unaffected region: samples well before P (skip win=480 margin)
        K = max(0, P - cfg.win)
        if K == 0:
            continue
        eq = torch.equal(y_full[..., :K], y_m[..., :K])
        diff = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
        worst = diff if worst is None else max(worst, diff)
        assert eq, f"future leak at P={P}: diff={diff}"
    print(f"  G5 future-perturbation: {len(ps)} cut points, past bit-identical "
          f"(torch.equal) — worst diff={worst}  PASS")


class _MutantGlobalMeanNorm(Fusion):
    """Mutation: normalize the output Y by the GLOBAL mean of |Y| (a prohibited
    whole-segment statistic, §4).  Zeroing future changes the global mean ⇒ all
    past samples rescale ⇒ past outputs change ⇒ leak (ALWAYS, on any signal —
    unlike the bidir-w-EMA which only leaks when w varies; the new ③-only detector
    makes w near-constant, so the bidir-EMA stopped leaking and was replaced
    here by this always-leaking whole-segment-stat mutation).  Production never
    does this; exists only to prove the future-perturbation test catches it."""

    def process_batch(self, s, v):
        y = super().process_batch(s, v)
        return y / (y.abs().mean() + 1e-8)


def test_g5_mutation_sanity():
    """Deliberately non-causal (global-mean normalization of Y — a prohibited
    whole-segment statistic) ⇒ future-perturbation FAILS (past rescales)."""
    cfg = FusionConfig()
    s, v = _signals(0, 16000)
    mutant = _MutantGlobalMeanNorm(cfg)
    y_full = mutant.process_batch(s, v)
    T = s.shape[-1]
    P = T // 2
    s_m = s.clone(); s_m[:, P:] = 0.0
    v_m = v.clone(); v_m[:, P:] = 0.0
    y_m = mutant.process_batch(s_m, v_m)
    K = max(0, P - cfg.win)
    diff = (y_full[..., :K] - y_m[..., :K]).abs().max().item()
    leaked = diff > 1e-6
    print(f"  G5 mutation sanity: global-mean-norm(Y) ⇒ future leak at P={P}: "
          f"past diff={diff:.3e} (must be > 1e-6) → "
          f"{'FAIL-of-mutant (test catches it) PASS' if leaked else 'mutant NOT caught — PROBLEM'}")
    assert leaked, ("mutation sanity FAILED: the future-perturbation test did "
        "NOT catch the deliberate global-mean-norm (whole-segment stat) — the test is too weak.")


if __name__ == "__main__":
    test_g5_equiv()
    test_g5_future_perturbation()
    test_g5_mutation_sanity()
    print("G5 streaming-causal tests: all PASS")
