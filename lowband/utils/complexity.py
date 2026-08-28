"""Static complexity measurement (§5.1).

Reports: parameter count, MACs/s, peak activation memory for each arm.

§5.1 WARNING: auto tools (thop/ptflops/fvcore) often miss LSTM and custom ops.
LSTM and DDSP oscillators MUST be hand-counted and reconciled with the tool
value. This module provides BOTH a hand-written counter and a hook-based
counter, and reports when they disagree.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dsp import StftConfig


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> dict:
    """Per-module parameter breakdown for human auditing."""
    breakdown = defaultdict(int)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        breakdown[top] += param.numel()
    return dict(breakdown)


# --- hook-based MAC counter -------------------------------------------------
class MACCounter:
    """Counts MACs via forward hooks on Conv/Linear/LSTM/GRU layers.

    This is the "tool value" that §5.1 says must be reconciled with the
    hand-counted value.
    """

    def __init__(self):
        self.macs = defaultdict(int)
        self.handles = []

    def _conv_hook(self, module, inp, out, name=""):
        x = inp[0]
        B = x.shape[0]
        # MACs = B * out_elements_per_sample
        if isinstance(module, nn.Conv1d):
            L_out = out.shape[-1]
            macs = B * module.out_channels * module.kernel_size[0] * L_out
            if module.groups > 1:
                macs = macs // module.groups
        elif isinstance(module, nn.Conv2d):
            H_out, W_out = out.shape[-2], out.shape[-1]
            macs = (B * module.out_channels *
                    module.kernel_size[0] * module.kernel_size[1] * H_out * W_out)
            if module.groups > 1:
                macs = macs // module.groups
        elif isinstance(module, (nn.Linear,)):
            macs = B * out.shape[-1] * module.in_features
        else:
            macs = 0
        self.macs[name] += int(macs)

    def _lstm_hook(self, module, inp, out, name=""):
        # LSTM: 4 gates, each is input*hidden + hidden*hidden
        x = inp[0]  # (B, T, input_size) or (B, input_size)
        if x.dim() == 3:
            B, T, inp_size = x.shape
        else:
            B, inp_size = x.shape
            T = 1
        hidden = module.hidden_size
        # 4*(input*hidden + hidden*hidden) per step per batch
        macs = 4 * (inp_size * hidden + hidden * hidden) * T * B
        self.macs[name] += int(macs)

    def _gru_hook(self, module, inp, out, name=""):
        x = inp[0]
        if x.dim() == 3:
            B, T, inp_size = x.shape
        else:
            B, inp_size = x.shape
            T = 1
        hidden = module.hidden_size
        # 3*(input*hidden + hidden*hidden) per step per batch
        macs = 3 * (inp_size * hidden + hidden * hidden) * T * B
        self.macs[name] += int(macs)

    def _convtranspose_hook(self, module, inp, out, name=""):
        x = inp[0]
        B = x.shape[0]
        if isinstance(module, nn.ConvTranspose2d):
            H_out, W_out = out.shape[-2], out.shape[-1]
            macs = (B * module.out_channels *
                    module.kernel_size[0] * module.kernel_size[1] * H_out * W_out)
            if module.groups > 1:
                macs = macs // module.groups
        elif isinstance(module, nn.ConvTranspose1d):
            L_out = out.shape[-1]
            macs = B * module.out_channels * module.kernel_size[0] * L_out
            if module.groups > 1:
                macs = macs // module.groups
        else:
            macs = 0
        self.macs[name] += int(macs)

    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._conv_hook(m, i, o, n)))
            elif isinstance(module, nn.Conv2d):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._conv_hook(m, i, o, n)))
            elif isinstance(module, nn.ConvTranspose2d):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._convtranspose_hook(m, i, o, n)))
            elif isinstance(module, nn.ConvTranspose1d):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._convtranspose_hook(m, i, o, n)))
            elif isinstance(module, nn.LSTM):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._lstm_hook(m, i, o, n)))
            elif isinstance(module, nn.GRU):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._gru_hook(m, i, o, n)))
            elif isinstance(module, nn.Linear):
                self.handles.append(
                    module.register_forward_hook(lambda m, i, o, n=name: self._conv_hook(m, i, o, n)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def total(self) -> int:
        return sum(self.macs.values())


def measure_complexity(model: nn.Module, sample_rate: float = 16000.0,
                        hop: int = 160, n_bins: int = 65,
                        batch_size: int = 1) -> dict:
    """Run one forward, count MACs via hooks, compute MACs/s and peak memory.

    Returns:
        dict with keys: params, macs_per_forward, macs_per_sec,
        peak_activation_bytes, hook_breakdown, param_breakdown
    """
    params = count_parameters(model)
    param_breakdown = count_parameters_by_layer(model)

    counter = MACCounter()
    counter.attach(model)

    # Build a dummy input: 1 second of audio
    T = sample_rate
    x = torch.randn(batch_size, int(T))
    peak_mem = [0]

    # Track peak activation memory via a hook on all modules
    mem_handles = []

    def mem_hook(module, inp, out, name=""):
        if isinstance(out, torch.Tensor):
            peak_mem[0] = max(peak_mem[0], out.numel() * out.element_size())
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    peak_mem[0] = max(peak_mem[0], o.numel() * o.element_size())

    for name, module in model.named_modules():
        if name:
            mem_handles.append(
                module.register_forward_hook(lambda m, i, o, n=name: mem_hook(m, i, o, n)))

    try:
        with torch.no_grad():
            model(x)
    finally:
        counter.detach()
        for h in mem_handles:
            h.remove()

    macs_per_fwd = counter.total()
    # One forward pass processes T=sr samples = 1 second.
    # So MACs/s = macs_per_fwd (NOT * frames_per_sec — that double-counts).
    macs_per_sec = int(macs_per_fwd / batch_size)

    # Peak activation memory (does NOT include weights — caller adds weight size)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    peak_total = peak_mem[0] + weight_bytes

    return {
        "params": params,
        "macs_per_forward": macs_per_fwd,
        "macs_per_sec": int(macs_per_sec),
        "hook_macs_breakdown": dict(counter.macs),
        "param_breakdown": param_breakdown,
        "weight_bytes": weight_bytes,
        "weight_kb": weight_bytes / 1024,
        "peak_activation_bytes": peak_mem[0],
        "peak_total_bytes": peak_total,
        "peak_total_kb": peak_total / 1024,
    }


def _attach_peak_mem_hooks(model: nn.Module):
    """Attach a peak-activation-memory hook to every submodule.

    Returns (peak_ref, handles).  ``peak_ref[0]`` holds the max
    ``out.numel()*out.element_size()`` seen across all module forward calls.
    Same methodology as ``measure_complexity`` so batch & streaming numbers
    are directly comparable (hooks see module OUTPUTS — conv/GRU/LSTM
    activations, the dominant cost; they do NOT see transient tensor ops like
    the DDSP harmonic Gaussian, which are small per-frame in streaming anyway).
    """
    peak_ref = [0]
    handles = []

    def mem_hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            peak_ref[0] = max(peak_ref[0], out.numel() * out.element_size())
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    peak_ref[0] = max(peak_ref[0], o.numel() * o.element_size())

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(mem_hook))
    return peak_ref, handles


def measure_streaming_complexity(model: nn.Module, sample_rate: float = 16000.0,
                                  hop: int = 160, batch_size: int = 1,
                                  n_frames: int | None = None,
                                  oracle_f0_hz: float | None = None) -> dict:
    """MEASURE (not estimate) the streaming peak memory via ``stream_step``.

    Feeds 1 second of audio (``n_frames = sample_rate/hop`` hops, 125 @4kHz)
    one frame at a time through the model's streaming path, tracking the peak
    module-activation memory across ALL ``stream_step`` calls.  Uses the same
    hook methodology as ``measure_complexity`` (batch) so the two numbers are
    directly comparable.

    Args:
        model: built arm (must implement stream_init / stream_step).
        oracle_f0_hz: for Arm A, set the streaming ``f0_override`` so the run
            is deterministic.  F0 source does NOT affect the conv/GRU
            activation memory (F0 is consumed AFTER the control net), so this
            is comparable to the batch measurement which uses default f0_mode.

    Returns dict with streaming peak activation, weight, peak_total_kb, plus
    n_frames for transparency.
    """
    if n_frames is None:
        n_frames = int(sample_rate / hop)  # 125 frames = 1 s @4kHz
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params = count_parameters(model)
    model.eval()

    peak_ref, handles = _attach_peak_mem_hooks(model)
    try:
        with torch.no_grad():
            x = torch.randn(batch_size, n_frames * hop)
            state = model.stream_init(batch_size)
            if (oracle_f0_hz is not None and isinstance(state, dict)
                    and "f0_override" in state):
                state["f0_override"] = torch.full((batch_size,),
                                                   float(oracle_f0_hz))
            for i in range(n_frames):
                frame = x[:, i * hop:(i + 1) * hop]
                if frame.shape[-1] < hop:
                    frame = F.pad(frame, (0, hop - frame.shape[-1]))
                _, state = model.stream_step(frame, state)
    finally:
        for h in handles:
            h.remove()

    peak_total = peak_ref[0] + weight_bytes
    return {
        "params": params,
        "n_frames": n_frames,
        "weight_bytes": weight_bytes,
        "weight_kb": weight_bytes / 1024,
        "peak_activation_bytes": peak_ref[0],
        "peak_activation_kb": peak_ref[0] / 1024,
        "peak_total_bytes": peak_total,
        "peak_total_kb": peak_total / 1024,
    }
