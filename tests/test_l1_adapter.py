"""L1 — Vibravox adapter correctness tests.

These exercise the REAL body-conduction data path (rework item ①).  The earlier
``VibravoxAdapter`` guessed channel names that do not exist in the dataset;
these tests pin the adapter to the MEASURED schema and prove the L1 data really
has the band-limitation the whole project is trying to reconstruct.

Skips (not fails) if the local parquet shard is absent — L1 data is a
~500 MB download that not every checkout will have.  Fetch it via the
scripts/inspect_vibravox*.py steps documented in reports/vibravox_schema_diff.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from lowband.data import build_dataset
from lowband.data.adapter import PairedSpeechDataset
from lowband.data.vibravox import (VIBRAVOX_BODY_SENSORS,
                                    VIBRAVOX_AIR_REF)
from tests._testutil import skip_if_no_l1

SHARD = "data/vibravox_parquet/speech_clean_test_0.parquet"
# Two shards (~100 rows each, ~206 total, 21 speakers, 0 overlap) = the
# "small subset" (§4.1: 几百条, not full 45 h).  Adapter skips any missing
# shard gracefully, so a single-shard checkout still runs.
SHARDS = [
    SHARD,
    "data/vibravox_parquet/speech_clean_test_2.parquet",
]

_L1_CFG = dict(
    adapter="vibravox", mode="parquet",
    parquet_files=SHARDS,
    sensor="temple_vibration_pickup", ref="headset_microphone",
    segment_len=16000, sr=16000, max_items=40, n_repeat=1, crop="random",
    normalize=True, seed=42,
)


def test_l1_adapter_loads_and_protocol():
    """Adapter loads from the real parquet shard and is Protocol-conformant."""
    skip_if_no_l1(SHARD)
    ds = build_dataset(_L1_CFG)
    assert len(ds) > 0, "adapter produced 0 items"
    assert isinstance(ds, PairedSpeechDataset), "must satisfy the dataset Protocol"
    item = ds[0]
    assert set(item.keys()) >= {"sensor", "ref", "meta"}, item.keys()
    assert item["sensor"].shape == (16000,), item["sensor"].shape
    assert item["ref"].shape == (16000,), item["ref"].shape
    assert item["sensor"].dtype == torch.float32
    assert item["ref"].dtype == torch.float32
    meta = item["meta"]
    assert meta["sr"] == 16000
    assert meta["sensor_type"] == "temple_vibration_pickup"
    assert meta.get("utterance_id"), "missing utterance_id"
    assert meta.get("speaker_id"), "missing speaker_id"
    print(f"  items={len(ds)} sensor_shape={tuple(item['sensor'].shape)} "
          f"sr={meta['sr']} sensor={meta['sensor_type']} "
          f"spk={meta['speaker_id']} utt={meta['utterance_id']}")


def test_l1_pairs_are_intrarow_aligned():
    """sensor ↔ ref come from the SAME row (simultaneous recording).

    We verify structurally: the full (pre-crop) sensor and ref arrays stored
    internally have EQUAL length (same utterance duration, because they are two
    channels of one recording).  A cross-file mismatch would give unequal
    lengths.  White-box on _items.
    """
    skip_if_no_l1(SHARD)
    ds = build_dataset(_L1_CFG)
    mis = [i for i, (s, r, _) in enumerate(ds._items) if len(s) != len(r)]
    assert not mis, f"rows with unequal sensor/ref length: {mis[:5]}"
    # meta carries (speaker_id, sentence_id); utterance_id must equal their join
    for s, r, m in ds._items[:10]:
        assert m["utterance_id"] == f"{m['speaker_id']}_{m['sentence_id']}", m
    print(f"  checked {len(ds._items)} rows: all sensor/ref same length (intrarow)")


def test_l1_sensor_is_bandlimited_vs_ref():
    """The body-conduction sensor is MORE band-limited than the air reference.

    This is the whole point of L1: a real domain gap to reconstruct.  Measured
    (reports/vibravox_schema_diff.md) the forehead accelerometer is ~3–5 dB more
    attenuated in the 1–2 kHz band than the headset air mic.  We assert the
    MEDIAN per-segment high/low-band energy ratio of the sensor is strictly
    below that of the reference over N segments (median is robust to the odd
    segment where a formant lands near the band boundary).

    NOTE: this is a SENSOR characterization, NOT an architecture quality verdict.
    """
    skip_if_no_l1(SHARD)
    ds = build_dataset(_L1_CFG)
    N = 24
    s_ratio, r_ratio = [], []
    for i in range(min(N, len(ds))):
        b = ds[i]
        s_ratio.append(_hi_lo_ratio(b["sensor"].numpy()))
        r_ratio.append(_hi_lo_ratio(b["ref"].numpy()))
    s_med = float(np.median(s_ratio))
    r_med = float(np.median(r_ratio))
    print(f"  median high/low ratio: sensor={s_med:.4f} ref={r_med:.4f} "
          f"(sensor must be lower)")
    assert s_med < r_med, \
        f"sensor not more band-limited than ref: {s_med:.4f} vs {r_med:.4f}"


def test_l1_default_sensor_is_body_conduction():
    """In-ear acoustic mics are the WRONG sensor type (wide-bandwidth).

    The DEFAULT sensor (no ``sensor`` key) must be a body-conduction pickup,
    and the air reference must be the headset mic (not an in-ear mic).
    """
    skip_if_no_l1(SHARD)
    assert "forehead_accelerometer" in VIBRAVOX_BODY_SENSORS
    assert "throat_microphone" in VIBRAVOX_BODY_SENSORS
    assert "temple_vibration_pickup" in VIBRAVOX_BODY_SENSORS
    # in-ear acoustic mics are NOT body sensors:
    assert "soft_in_ear_microphone" not in VIBRAVOX_BODY_SENSORS
    assert "rigid_in_ear_microphone" not in VIBRAVOX_BODY_SENSORS
    # air reference is the headset mic, not an in-ear mic:
    assert VIBRAVOX_AIR_REF == "headset_microphone"
    # constructing with an in-ear sensor still works (warns) but is discouraged:
    cfg_bad = dict(_L1_CFG, sensor="soft_in_ear_microphone", max_items=4)
    ds = build_dataset(cfg_bad)  # should print a WARNING, not raise
    assert len(ds) > 0
    # DEFAULT sensor (no sensor key) must be a body-conduction pickup:
    cfg_def = dict(_L1_CFG)
    cfg_def.pop("sensor")
    ds_def = build_dataset(cfg_def)
    assert ds_def[0]["meta"]["sensor_type"] in VIBRAVOX_BODY_SENSORS
    print(f"  default sensor = {ds_def[0]['meta']['sensor_type']} (body-conduction) ✓")


def _hi_lo_ratio(wav: np.ndarray, sr: int = 4000, split_hz: int = 1000) -> float:
    """High-band / low-band RMS energy ratio (scale-invariant under normalization)."""
    sp = np.abs(np.fft.rfft(wav))
    f = np.fft.rfftfreq(len(wav), 1 / sr)
    lo = np.sqrt(np.mean(sp[f < split_hz] ** 2))
    hi = np.sqrt(np.mean(sp[f >= split_hz] ** 2))
    return hi / (lo + 1e-9)


if __name__ == "__main__":
    test_l1_adapter_loads_and_protocol()
    test_l1_pairs_are_intrarow_aligned()
    test_l1_sensor_is_bandlimited_vs_ref()
    test_l1_default_sensor_is_body_conduction()
    print("L1 adapter tests: all PASS")
