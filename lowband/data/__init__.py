"""Data subpackage: adapters + degradation."""
from .adapter import PairedSpeechDataset
from .degradation import DegradationConfig, apply_degradation, measure_cutoff
from .lowpass_sim import LowpassSimAdapter
from .vibravox import VibravoxAdapter
from .template_adapter import TemplateAdapter

ADAPTERS = {
    "lowpass_sim": LowpassSimAdapter,
    "vibravox": VibravoxAdapter,
    "template": TemplateAdapter,
}


def build_dataset(cfg: dict):
    name = cfg["adapter"]
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter '{name}'; choose from {list(ADAPTERS)}")
    return ADAPTERS[name](cfg)

__all__ = [
    "PairedSpeechDataset", "DegradationConfig", "apply_degradation",
    "measure_cutoff", "LowpassSimAdapter", "VibravoxAdapter",
    "TemplateAdapter", "ADAPTERS", "build_dataset",
]
