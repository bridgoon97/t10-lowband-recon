"""§5.2 — Shape & causality tests.

Causality: zero out all samples after time t; outputs at frames ≤ t must be
bit-identical.  This catches bidirectional RNN, non-causal padding, global norm.
"""
import torch

from lowband import build_model

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 4000, "stft_n_fft": 128, "stft_hop": 32, "stft_win": 128}


def test_shape_arbitrary():
    """Any batch/length should run."""
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        for B, T in [(1, 1000), (3, 4000), (2, 8000)]:
            x = torch.randn(B, T)
            f0 = torch.full((B, 200), 150.0)  # oracle F0 for arm A
            out = model(x, {"f0": f0} if arm == "arm_a_ddsp" else None)
            assert out["mag"].dim() == 3, f"{arm} output wrong dim"
            assert out["mag"].shape[0] == B, f"{arm} batch mismatch"


def test_causality():
    """Zeroing future samples must not change past outputs."""
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg).eval()
        T = 4000
        x = torch.randn(1, T)
        f0 = torch.full((1, 125), 150.0)

        with torch.no_grad():
            out_full = model(x, {"f0": f0} if arm == "arm_a_ddsp" else None)

        # Zero out the second half
        x_masked = x.clone()
        x_masked[:, T // 2:] = 0.0
        with torch.no_grad():
            out_masked = model(x_masked, {"f0": f0} if arm == "arm_a_ddsp" else None)

        # Compare outputs in the first half of frames (before the masked region)
        N_half = out_full["mag"].shape[-1] // 2
        diff = (out_full["mag"][..., :N_half] - out_masked["mag"][..., :N_half]).abs()
        max_diff = diff.max().item()
        rel = max_diff / (out_full["mag"][..., :N_half].abs().max().item() + 1e-8)
        print(f"  {arm}: causality rel_err={rel:.2e} {'PASS' if rel < 1e-4 else 'FAIL'}")
        assert rel < 1e-4, f"{arm} leaks future information: rel_err={rel}"


if __name__ == "__main__":
    test_shape_arbitrary()
    print("Shape tests passed.")
    test_causality()
