"""Export each arm to ONNX + TorchScript (§5.9).

DDSP oscillators may have ops unsupported by ONNX — we detect this and fall
back to TorchScript-only with an explicit annotation.
"""
from __future__ import annotations

import io
import warnings

import torch

from ..interface import LowBandReconstructor
from ..dsp import StftConfig


class ArmWrapper(torch.nn.Module):
    """Wraps an arm so that forward(x) returns only mag (for export simplicity).

    The full forward returns a dict, which ONNX/TS can't export directly.  This
    wrapper extracts just the magnitude output.
    """

    def __init__(self, arm: LowBandReconstructor):
        super().__init__()
        self.arm = arm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.arm(x)
        return out["mag"]


def export_torchscript(arm: LowBandReconstructor, x: torch.Tensor,
                        path: str) -> dict:
    """Export to TorchScript via tracing.

    Returns dict with "path", "max_rel_error".
    """
    wrapper = ArmWrapper(arm).eval()
    with torch.no_grad():
        ts_model = torch.jit.trace(wrapper, x, strict=False)
        ts_model.save(path)
        # Verify
        ts_out = ts_model(x)
        py_out = wrapper(x)
        max_err = (ts_out - py_out).abs().max().item()
        rel_err = max_err / (py_out.abs().max().item() + 1e-8)
    return {"path": path, "max_rel_error": float(rel_err), "format": "torchscript"}


def export_onnx(arm: LowBandReconstructor, x: torch.Tensor,
                 path: str) -> dict:
    """Export to ONNX.

    Returns dict with "path", "max_rel_error", or "error" if unsupported.
    """
    wrapper = ArmWrapper(arm).eval()
    buf = io.BytesIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper, (x,), buf, input_names=["x"],
                output_names=["mag"],
                opset_version=17,
                dynamic_axes={"x": {0: "batch", 1: "length"},
                              "mag": {0: "batch", 2: "frames"}},
            )
        with open(path, "wb") as f:
            f.write(buf.getvalue())
    except Exception as e:
        return {"path": None, "error": str(e), "format": "onnx"}

    # Verify with onnxruntime if available, else onnx itself
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        in_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {in_name: x.numpy()})[0]
        with torch.no_grad():
            py_out = wrapper(x).numpy()
        max_err = abs(ort_out - py_out).max()
        rel_err = max_err / (abs(py_out).max() + 1e-8)
        return {"path": path, "max_rel_error": float(rel_err), "format": "onnx"}
    except ImportError:
        # onnxruntime not installed; use onnx symbolic check only
        try:
            import onnx
            onnx.load(path)  # will raise if invalid
            return {"path": path, "max_rel_error": None,
                    "note": "onnxruntime not available; symbolic check only",
                    "format": "onnx"}
        except Exception as e:
            return {"path": path, "error": str(e), "format": "onnx"}
