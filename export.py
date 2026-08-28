#!/usr/bin/env python3
"""Export an arm to ONNX + TorchScript (§5.9).

    python export.py --config configs/arm_a_ddsp.yaml --output exports/
"""
from __future__ import annotations

import argparse

import torch

from lowband import build_model
from lowband.utils.config import load_config, get_device
from lowband.export import export_torchscript, export_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="exports")
    ap.add_argument("--length", type=int, default=4000)
    args = ap.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)
    device = get_device(cfg)
    model = build_model(cfg).to(device).eval()

    x = torch.randn(1, args.length).to(device)
    ts_result = export_torchscript(model, x,
                                    os.path.join(args.output, f"{cfg['arm']}.pt"))
    print(f"[export] TorchScript: {ts_result}")

    onnx_result = export_onnx(model, x,
                               os.path.join(args.output, f"{cfg['arm']}.onnx"))
    print(f"[export] ONNX: {onnx_result}")


if __name__ == "__main__":
    main()
