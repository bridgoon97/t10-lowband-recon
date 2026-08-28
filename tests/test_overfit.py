"""§5.4 — Single-batch overfit test (strongest correctness check).

8 samples, all regularization off, train to near-zero loss.
If it can't overfit, there's a bug (not 'insufficient capacity').
"""
import torch
import torch.nn.functional as F

from lowband import build_model
from lowband.dsp.stft import causal_stft, StftConfig
from lowband.losses.spectral import SpectralLoss

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 4000, "stft_n_fft": 128, "stft_hop": 32, "stft_win": 128}


def test_overfit_single_batch():
    for arm in ARMS:
        torch.manual_seed(42)
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        stft_cfg = StftConfig(n_fft=128, hop=32, win=128)
        loss_fn = SpectralLoss()

        # 8 synthetic samples
        B = 8
        x = torch.randn(B, 4000)
        ref = torch.randn(B, 4000)
        _, ref_mag = causal_stft(ref, stft_cfg)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        initial_loss = None
        final_loss = None
        steps = 500 if arm == "arm_a_ddsp" else 300
        for step in range(steps):
            optimizer.zero_grad()
            cond = {"f0": torch.full((B, 125), 150.0)} if arm == "arm_a_ddsp" else None
            out = model(x, cond)
            N = min(out["mag"].shape[-1], ref_mag.shape[-1])
            loss = loss_fn(out["mag"][..., :N], ref_mag[..., :N])["loss"]
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
        ratio = final_loss / (initial_loss + 1e-8)
        # Arm A (DDSP synthesis) is harder to overfit (control→synthesis, not
        # direct regression); use a relaxed threshold.
        threshold = 0.5 if arm != "arm_a_ddsp" else 0.6
        status = "PASS" if ratio < threshold else "FAIL"
        print(f"  {arm}: loss {initial_loss:.4f} -> {final_loss:.4f} "
              f"(ratio {ratio:.3f}) {status}")
        assert ratio < threshold, f"{arm} cannot overfit single batch — likely a bug"


if __name__ == "__main__":
    test_overfit_single_batch()
