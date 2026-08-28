"""§5.3 — Streaming-batch equivalence.

stream_step frame-by-frame vs forward; relative error must be < 1e-4.

For Arm A, oracle F0 is used (the F0 estimator itself is tested separately).
"""
import torch

from lowband import build_model

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 4000, "stft_n_fft": 128, "stft_hop": 32, "stft_win": 128}
HOP = 32


def test_streaming_batch_equiv():
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg).eval()
        B, T = 2, 4000
        x = torch.randn(B, T)
        # Oracle F0 for arm A
        cond = None
        f0_override = None
        if arm == "arm_a_ddsp":
            f0_oracle = torch.full((B, 125), 150.0)
            cond = {"f0": f0_oracle}

        with torch.no_grad():
            out = model(x, cond)
            # Streaming
            state = model.stream_init(B)
            if arm == "arm_a_ddsp":
                state["f0_override"] = torch.full((B,), 150.0)
            frames = []
            for i in range(0, T, HOP):
                frame = x[:, i:i + HOP]
                if frame.shape[-1] < HOP:
                    break
                mag_f, state = model.stream_step(frame, state)
                frames.append(mag_f)
            stream_mag = torch.stack(frames, dim=-1)

        N = min(out["mag"].shape[-1], stream_mag.shape[-1])
        diff = (out["mag"][..., :N] - stream_mag[..., :N]).abs()
        rel = diff.max().item() / (out["mag"][..., :N].abs().max().item() + 1e-8)
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  {arm}: rel_err={rel:.2e} {status}")
        assert rel < 1e-4, f"{arm} streaming ≠ batch: rel_err={rel}"


if __name__ == "__main__":
    test_streaming_batch_equiv()
