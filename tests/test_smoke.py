"""§5.10 — Small-scale training smoke test (spec change: complex spec, 16 kHz).

Few dozen samples, few hundred steps.  Criteria (set BEFORE observation, not
result-tuned):
1. loss drops ≥10% below start at some point (min < 0.9*first) — learning, no
   gross bug.  On real L1 the complex phase can diverge later; min tolerates it.
2. divergence BOUNDED: last-10-avg ≤ 1.5× first-10-avg ('dropped then blew up'
   vs 'dropped and stable' — without it they look identical).
3. output is not constant/noise.

Two variants:
  * test_smoke_train       — clean-ish lowpass_sim degradation (L0 baseline)
  * test_smoke_train_noisy — T11 §2: full-band + wind noise ON (the target-
    device input condition); same criteria (loss↓ + bounded + non-constant).
NOT a quality test — no arm-vs-arm comparison (data volume doesn't support it).
"""
import glob

import torch

from lowband import build_model
from lowband.data.lowpass_sim import LowpassSimAdapter
from lowband.dsp.stft import StftConfig, complex_stft_truncated
from lowband.losses.spectral import SpectralLoss
from lowband.utils.config import set_seed

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64}


def _smoke_arm(arm, degradation_cfg, label):
    wavs = sorted(glob.glob("data/test_speech/*.wav"))
    if not wavs:
        print(f"  [{label}] SKIP: no test speech wavs")
        return True
    set_seed(42)
    cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
    model = build_model(cfg)
    ds = LowpassSimAdapter({
        "clean_wavs": wavs, "segment_len": 16000, "sr": 16000,
        "n_repeat": 1, "degradation": degradation_cfg,
    })
    stft_cfg = StftConfig(n_fft=512, hop=160, win=480, keep_bins=64)
    loss_fn = SpectralLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    B = 4
    for step in range(100):
        idx = torch.randint(0, len(ds), (B,))
        batch = [ds[i.item()] for i in idx]
        x = torch.stack([b["sensor"] for b in batch])
        ref = torch.stack([b["ref"] for b in batch])
        ref_spec = complex_stft_truncated(ref, stft_cfg)
        optimizer.zero_grad()
        cond = {"f0": torch.full((B, 100), 150.0)} if arm == "arm_a_ddsp" else None
        out = model(x, cond)
        N = min(out["spec"].shape[-1], ref_spec.shape[-1])
        ld = loss_fn(out["spec"][..., :N], ref_spec[..., :N])
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(ld["loss"].item())
    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    min_loss = min(losses)
    decreasing = min_loss < 0.9 * first_avg
    bounded = last_avg <= 1.5 * first_avg
    with torch.no_grad():
        out_test = model(x[:1], {"f0": torch.full((1, 100), 150.0)}
                         if arm == "arm_a_ddsp" else None)
        std = out_test["spec"].abs().std().item()
    not_const = std > 1e-4
    status = "PASS" if (decreasing and bounded and not_const) else "FAIL"
    print(f"  [{label}] {arm}: first={first_avg:.2f} min={min_loss:.2f}@{losses.index(min_loss)} "
          f"last={last_avg:.2f} ({'✓' if decreasing else '✗'}learn "
          f"{'✓' if bounded else '✗'}bounded) std={std:.4f} {status}")
    assert decreasing, f"[{label}] {arm} loss not decreasing"
    assert bounded, f"[{label}] {arm} unbounded: {last_avg:.2f} > 1.5*{first_avg:.2f}"
    assert not_const, f"[{label}] {arm} output constant"
    return True


def test_smoke_train():
    """L0 baseline: clean-ish lowpass_sim degradation."""
    for arm in ARMS:
        _smoke_arm(arm, {"cutoff_min": 300, "cutoff_max": 1200}, "clean")


def test_smoke_train_noisy():
    """T11 §5: full-band + wind noise ON (target-device input condition).

    Same criteria (loss↓ + bounded divergence + non-constant) — the model must
    learn under noisy input, not just clean.  No quality comparison / ranking.
    """
    for arm in ARMS:
        _smoke_arm(arm, {
            "cutoff_min": 300, "cutoff_max": 1200,
            "fullband_noise": True, "fullband_snr_db": 10.0,
            "wind_noise": True, "wind_snr_db": 10.0,
        }, "noisy")


if __name__ == "__main__":
    test_smoke_train()
    test_smoke_train_noisy()
    print("smoke tests (clean + noisy): all PASS")
