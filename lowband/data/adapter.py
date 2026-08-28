"""Dataset adapter interface (§4.2).

``PairedSpeechDataset`` is the Protocol every dataset implements.  Switching
data = writing a new adapter + changing one config line; model and training
code are untouched.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PairedSpeechDataset(Protocol):
    """Every dataset adapter must satisfy this Protocol."""

    def __getitem__(self, i: int) -> dict:
        """Return a dict with keys:

        - ``sensor``: (T,) body-conduction / band-limited signal @4kHz
        - ``ref``:    (T,) reference air-conduction clean speech @4kHz
        - ``meta``:   dict with at least ``sr``, ``sensor_type``, ``utterance_id``
        """
        ...

    def __len__(self) -> int:
        ...
