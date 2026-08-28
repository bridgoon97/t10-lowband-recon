"""§5.3 — Streaming-batch equivalence (spec change: complex "spec").

stream_step frame-by-frame vs forward; relative error must be < 1e-4.

For Arm A, oracle F0 is used (the F0 estimator itself is tested separately).
"""
import torch

from lowband import build_model

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64}
HOP = 160


def test_streaming_batch_equiv():
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm, f0_mode="oracle")
        model = build_model(cfg).eval()
        B, T = 2, 16000
        x = torch.randn(B, T)
        cond = {"f0": torch.full((B, 100), 150.0)} if arm == "arm_a_ddsp" else None

        with torch.no_grad():
            out = model(x, cond)
            state = model.stream_init(B)
            if arm == "arm_a_ddsp":
                state["f0_override"] = torch.full((B,), 150.0)
            frames = []
            for i in range(0, T, HOP):
                frame = x[:, i:i + HOP]
                if frame.shape[-1] < HOP:
                    break
                spec_f, state = model.stream_step(frame, state)
                frames.append(spec_f)
            stream_spec = torch.stack(frames, dim=-1)

        N = min(out["spec"].shape[-1], stream_spec.shape[-1])
        diff = (out["spec"][..., :N] - stream_spec[..., :N]).abs()
        rel = diff.max().item() / (out["spec"][..., :N].abs().max().item() + 1e-8)
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  {arm}: rel_err={rel:.2e} {status}")
        assert rel < 1e-4, f"{arm} streaming ≠ batch: rel_err={rel}"


if __name__ == "__main__":
    test_streaming_batch_equiv()
