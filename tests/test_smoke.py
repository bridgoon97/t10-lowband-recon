"""§5.10 — Small-scale training smoke test.

Few dozen samples, few hundred steps.  Only two criteria:
1. loss monotonically decreasing
2. output is not constant/noise

NOT a quality test — no arm-vs-arm comparison (data volume doesn't support it).
"""
import glob
import os

import torch

from lowband import build_model
from lowband.data.lowpass_sim import LowpassSimAdapter
from lowband.dsp.stft import StftConfig, causal_stft
from lowband.losses.spectral import SpectralLoss
from lowband.utils.config import set_seed

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 4000, "stft_n_fft": 128, "stft_hop": 32, "stft_win": 128}


def test_smoke_train():
    wavs = sorted(glob.glob("data/test_speech/*.wav"))
    if not wavs:
        print("  SKIP: no test speech wavs")
        return

    for arm in ARMS:
        set_seed(42)
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        ds = LowpassSimAdapter({
            "clean_wavs": wavs, "segment_len": 4000, "sr": 4000,
            "n_repeat": 3,
            "degradation": {"cutoff_min": 300, "cutoff_max": 1200},
        })
        stft_cfg = StftConfig(n_fft=128, hop=32, win=128)
        loss_fn = SpectralLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        losses = []
        B = 4
        for step in range(100):
            idx = torch.randint(0, len(ds), (B,))
            batch = [ds[i.item()] for i in idx]
            x = torch.stack([b["sensor"] for b in batch])
            ref = torch.stack([b["ref"] for b in batch])
            _, ref_mag = causal_stft(ref, stft_cfg)

            optimizer.zero_grad()
            cond = {"f0": torch.full((B, 125), 150.0)} if arm == "arm_a_ddsp" else None
            out = model(x, cond)
            N = min(out["mag"].shape[-1], ref_mag.shape[-1])
            loss = loss_fn(out["mag"][..., :N], ref_mag[..., :N])["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check 1: loss decreasing
        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        decreasing = last_avg < first_avg
        # Check 2: output not constant
        with torch.no_grad():
            out_test = model(x[:1], {"f0": torch.full((1, 125), 150.0)}
                              if arm == "arm_a_ddsp" else None)
            std = out_test["mag"].std().item()
        not_const = std > 1e-4
        status = "PASS" if (decreasing and not_const) else "FAIL"
        print(f"  {arm}: loss {first_avg:.4f} -> {last_avg:.4f} "
              f"(↓{'✓' if decreasing else '✗'}) output_std={std:.4f} {status}")
        assert decreasing, f"{arm} loss not decreasing"
        assert not_const, f"{arm} output is constant"


if __name__ == "__main__":
    test_smoke_train()
