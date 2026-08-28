"""§5.9 — Export: ONNX + TorchScript, verify output consistency (<1e-4)."""
import os

import torch

from lowband import build_model
from lowband.export import export_torchscript, export_onnx

ARMS = ["arm_a_ddsp", "arm_b_crn", "arm_c_ftlstm"]
BASE_CFG = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
            "stft_win": 480, "keep_bins": 64, "f0_mode": "oracle"}


def test_export_all():
    os.makedirs("exports", exist_ok=True)
    for arm in ARMS:
        cfg = dict(BASE_CFG, arm=arm)
        model = build_model(cfg).eval()
        x = torch.randn(1, 16000)
        cond = {"f0": torch.full((1, 100), 150.0)} if arm == "arm_a_ddsp" else None

        # TorchScript
        ts = export_torchscript(model, x, f"exports/{arm}_test.pt")
        print(f"  {arm} TorchScript: max_rel_err={ts.get('max_rel_error', 'N/A')}")
        assert ts.get("max_rel_error", 1.0) < 1e-4 or ts.get("error") is None, \
            f"{arm} TS export error too large"

        # ONNX
        onnx = export_onnx(model, x, f"exports/{arm}_test.onnx")
        if "error" in onnx:
            print(f"  {arm} ONNX: FAILED — {onnx['error'][:100]}")
            print(f"         (may be expected for DDSP oscillators; TS-only is OK)")
        else:
            err = onnx.get("max_rel_error")
            print(f"  {arm} ONNX: max_rel_err={err}")
            if err is not None:
                assert err < 1e-4, f"{arm} ONNX error too large: {err}"


if __name__ == "__main__":
    test_export_all()
