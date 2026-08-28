"""§5.2 — Shape & causality tests (spec change: output is complex "spec").

Causality: zero out all samples after time t; outputs at frames ≤ t must be
bit-identical.  This catches bidirectional RNN, non-causal padding, global norm.
"""
import torch

from lowband import build_model

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64}


def test_shape_arbitrary():
    """Any batch/length should run."""
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg)
        for B, T in [(1, 1600), (3, 16000), (2, 32000)]:
            x = torch.randn(B, T)
            f0 = torch.full((B, 100), 150.0)  # oracle F0 for arm A
            out = model(x, {"f0": f0} if arm == "arm_a_ddsp" else None)
            assert out["spec"].dim() == 3, f"{arm} output wrong dim"
            assert out["spec"].shape[0] == B, f"{arm} batch mismatch"
            assert torch.is_complex(out["spec"]), f"{arm} spec must be complex"


def test_causality():
    """Zeroing future samples must not change past outputs."""
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg).eval()
        T = 16000
        x = torch.randn(1, T)
        f0 = torch.full((1, 100), 150.0)

        with torch.no_grad():
            out_full = model(x, {"f0": f0} if arm == "arm_a_ddsp" else None)

        # Zero out the second half
        x_masked = x.clone()
        x_masked[:, T // 2:] = 0.0
        with torch.no_grad():
            out_masked = model(x_masked, {"f0": f0} if arm == "arm_a_ddsp" else None)

        # Compare outputs in the first half of frames (before the masked region)
        N_half = out_full["spec"].shape[-1] // 2
        diff = (out_full["spec"][..., :N_half] - out_masked["spec"][..., :N_half]).abs()
        max_diff = diff.max().item()
        rel = max_diff / (out_full["spec"][..., :N_half].abs().max().item() + 1e-8)
        print(f"  {arm}: causality rel_err={rel:.2e} {'PASS' if rel < 1e-4 else 'FAIL'}")
        assert rel < 1e-4, f"{arm} leaks future information: rel_err={rel}"


if __name__ == "__main__":
    test_shape_arbitrary()
    print("Shape tests passed.")
    test_causality()
