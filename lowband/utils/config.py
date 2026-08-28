"""Config loading: YAML → dict, with device handling (§7.1).

No hardcoded device anywhere; ``device`` is a config field.
"""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(cfg: dict) -> torch.device:
    dev = cfg.get("device", "cpu")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def resolve_device(obj, device: torch.device):
    """Move any tensor/module to device (§7.1: no hardcoded .cpu()/.cuda())."""
    if isinstance(obj, (torch.Tensor, torch.nn.Module)):
        return obj.to(device)
    return obj
