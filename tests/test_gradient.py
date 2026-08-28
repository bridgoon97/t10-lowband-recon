"""§5.5 — Gradient flow check (spec change: complex spec).

Every parameter gets non-None, non-zero grad.  Each loss branch separately.
Check for accidentally detached paths (DDSP F0 estimator is the classic trap).
"""
import torch

from lowband import build_model
from lowband.dsp.stft import StftConfig, complex_stft_truncated
from lowband.losses.spectral import SpectralLoss
from lowband.losses import MultiResolutionSTFTLoss, reconstruct_waveform_with_oracle_phase

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64}


def test_gradient_flow():
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        stft_cfg = StftConfig(n_fft=512, hop=160, win=480, keep_bins=64)
        loss_fn = SpectralLoss()
        x = torch.randn(2, 16000)
        ref = torch.randn(2, 16000)
        ref_spec = complex_stft_truncated(ref, stft_cfg)

        model.zero_grad()
        cond = {"f0": torch.full((2, 100), 150.0)} if arm == "arm_a_ddsp" else None
        out = model(x, cond)
        N = min(out["spec"].shape[-1], ref_spec.shape[-1])
        loss = loss_fn(out["spec"][..., :N], ref_spec[..., :N])["loss"]
        loss.backward()

        n_none = 0
        n_zero = 0
        n_total = 0
        for name, p in model.named_parameters():
            n_total += 1
            if p.grad is None:
                n_none += 1
                print(f"    {name}: grad is None!")
            elif p.grad.abs().sum().item() == 0:
                n_zero += 1
                print(f"    {name}: grad is all zero!")
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                        if p.grad is not None) ** 0.5
        print(f"  {arm}: {n_total} params, {n_none} None, {n_zero} zero, "
              f"grad_norm={grad_norm:.4f}")
        assert n_none == 0, f"{arm} has {n_none} params with None grad"


def test_mr_stft_gradient():
    """Multi-resolution STFT loss branch must also produce gradients."""
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        stft_cfg = StftConfig(n_fft=512, hop=160, win=480, keep_bins=64)
        mrstft = MultiResolutionSTFTLoss()
        x = torch.randn(2, 16000)
        ref = torch.randn(2, 16000)
        model.zero_grad()
        cond = {"f0": torch.full((2, 100), 150.0)} if arm == "arm_a_ddsp" else None
        out = model(x, cond)
        pred_wav = reconstruct_waveform_with_oracle_phase(out["spec"], ref, stft_cfg)
        loss = mrstft(pred_wav, ref)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                        if p.grad is not None) ** 0.5
        print(f"  {arm} MR-STFT: grad_norm={grad_norm:.4f}")


if __name__ == "__main__":
    test_gradient_flow()
    test_mr_stft_gradient()
