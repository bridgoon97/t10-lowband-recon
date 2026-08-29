"""Vibravox adapter (L1 — real body-conduction recordings).

Schema verified against the ACTUAL HuggingFace dataset
``Cnam-LMSSC/vibravox`` (see reports/vibravox_schema_diff.md for the measured
diff vs the earlier guess).  Earlier code guessed channel names
(``bone_chin`` / ``air_oss``) that DO NOT EXIST in the real data and called
``load_dataset`` without a config name — both are wrong; this file is the fix.

REAL schema (measured 2026-08, speech_clean/test shard):
  * multi-CONFIG dataset: configs = speech_clean | speech_noisy |
    speechless_clean | speechless_noisy.  Must pass ``name=``.
  * audio channels are NESTED under a top-level ``audio`` dict, each a
    ``{bytes, path}`` struct (encoded WAV/FLAC bytes):
        audio.headset_microphone       # air-conduction reference (the clean ref)
        audio.temple_vibration_pickup  # body-conduction vibration  (BONE) — PRIMARY
        audio.forehead_accelerometer   # body-conduction accelerometer (BONE) — secondary
        audio.throat_microphone        # throat contact mic            (BONE)
        audio.soft_in_ear_microphone   # in-ear acoustic mic (NOT target)
        audio.rigid_in_ear_microphone  # in-ear acoustic mic (NOT target)
  * metadata: gender, speaker_id, sentence_id, duration, raw_text, ...
  * sampling rate = 48000 Hz for ALL channels (uniform).  Must resample to
    4 kHz (the model's operating rate).  Pairing is INTRA-row: the 6 channels
    of one utterance are recorded simultaneously in one row keyed by
    (speaker_id, sentence_id) — NO cross-file matching needed.

§4.1 compliance:
  * only a small subset is used — point ``parquet_files`` at one or two local
    parquet shards (~100 rows each); do NOT pull the full 45 h.
  * uses a BONE-CONDUCTION pickup (default temple_vibration_pickup, the
  * primary — it sits closest to the earphone wear point AND is measured
  * the MOST band-limited of the body sensors; forehead_accelerometer is the
  * secondary — run both for a natural bandwidth comparison), NOT an
    in-ear acoustic mic (those have much wider bandwidth — wrong sensor type).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import DegradationConfig, apply_degradation
from .lowpass_sim import _resample


# --- channel names validated against the real dataset ----------------------
VIBRAVOX_CHANNELS = {
    "headset_microphone",       # air-conduction reference
    "temple_vibration_pickup",  # body-conduction (vibration) — BONE, PRIMARY
    "forehead_accelerometer",   # body-conduction (accelerometer) — BONE, secondary
    "throat_microphone",        # body-conduction (contact mic)   — BONE
    "soft_in_ear_microphone",   # in-ear acoustic mic (NOT target sensor)
    "rigid_in_ear_microphone",  # in-ear acoustic mic (NOT target sensor)
}
# Sensors that pick up body-conduction vibration (the target sensor type).
# Do NOT use the in-ear mics as the sensor — they are acoustic, wide-bandwidth.
VIBRAVOX_BODY_SENSORS = {
    "forehead_accelerometer", "temple_vibration_pickup", "throat_microphone",
}
# Air-conduction reference (clean speech to reconstruct toward).
VIBRAVOX_AIR_REF = "headset_microphone"


def _decode_audio_bytes(cell: dict) -> tuple[np.ndarray, int]:
    """Decode a `{bytes, path}` audio cell → (mono float32 wav, sr)."""
    raw = cell.get("bytes")
    if not raw:
        raise ValueError(f"audio cell has no bytes (path={cell.get('path')!r})")
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


class VibravoxAdapter(Dataset):
    """L1: real body-conduction ↔ air-conduction paired recordings.

    cfg keys:
        mode: "parquet" (default, recommended) | "hf"
            parquet — read local parquet shard files (bounded subset).
            hf      — stream from HuggingFace (slow; downloads on the fly).
        parquet_files: list[str]   # mode=parquet: local .parquet paths
        sensor:  body-conduction channel (default "temple_vibration_pickup")
        ref:     air-conduction reference  (default "headset_microphone")
        # hf-mode only:
        hf_config: "speech_clean" | "speech_noisy" | ... (default "speech_clean")
        hf_split:  "test" | "train" | "validation" (default "test")
        hf_cache_dir: local HF cache
        # common:
        segment_len: samples per item @4kHz (default 4000 = 1s)
        sr: 4000  (target rate; data is resampled 48k → 4k)
        max_items: cap on rows loaded (§4.1: hundreds, not full)
        n_repeat:  random crops drawn per row (augmentation, default 1)
        crop: "center" | "random" (default "random")
        normalize: peak-normalize each segment (default True)
        augment: apply degradation on top (default False — real data is
                 already body-conduction; do NOT double-degrade by default)
        degradation: DegradationConfig kwargs (if augment=True)
        seed: base seed for reproducible random crops
    """

    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "parquet")
        self.sensor = cfg.get("sensor", "temple_vibration_pickup")
        self.ref = cfg.get("ref", VIBRAVOX_AIR_REF)
        self._validate_channels()
        self.segment_len = cfg.get("segment_len", 4000)
        self.sr = cfg.get("sr", 4000)
        self.max_items = cfg.get("max_items", 200)
        self.n_repeat = cfg.get("n_repeat", 1)
        self.crop = cfg.get("crop", "random")
        self.normalize = cfg.get("normalize", True)
        self.sensor_lowpass_hz = cfg.get("sensor_lowpass_hz", None)  # T11 §5: align sensor to target device bandwidth
        self.augment = cfg.get("augment", False)
        self.seed = cfg.get("seed", 42)
        if self.augment:
            d = dict(cfg.get("degradation", {}))
            d.setdefault("sample_rate", self.sr)
            self.deg_cfg = DegradationConfig(**d)
        else:
            self.deg_cfg = None

        # Preloaded, decoded, resampled pairs: list of (sensor_4k, ref_4k, meta)
        self._items: list[tuple[np.ndarray, np.ndarray, dict]] = []
        if self.mode == "parquet":
            self._init_parquet(cfg)
        elif self.mode == "hf":
            self._init_hf(cfg)
        else:
            raise ValueError(f"VibravoxAdapter mode must be 'parquet' or 'hf', "
                             f"got {self.mode!r}")

    def _apply_sensor_lowpass(self, wav: np.ndarray) -> np.ndarray:
        """T11 §5: lowpass the SENSOR (not ref) to align the L1 training input to
        the target device's bandwidth.  temple's SNR>5 dB band (~977 Hz, §1) is
        WIDER than the target's 400–600 Hz; cutting to ~600 Hz makes training
        match what the target device actually feeds the model.  REF stays clean
        (the network must reconstruct the full band from a narrower input)."""
        if not self.sensor_lowpass_hz or self.sensor_lowpass_hz >= self.sr / 2:
            return wav
        from scipy.signal import butter, filtfilt
        b, a = butter(6, self.sensor_lowpass_hz / (self.sr / 2), btype="low")
        return filtfilt(b, a, wav).astype(np.float32)


        if len(self._items) == 0:
            print("[VibravoxAdapter] WARNING: 0 items loaded — check "
                  "parquet_files / hf_config. See reports/vibravox_schema_diff.md")

    # --- channel validation -------------------------------------------------
    def _validate_channels(self):
        if self.sensor not in VIBRAVOX_CHANNELS:
            raise ValueError(
                f"sensor {self.sensor!r} is not a real Vibravox channel. "
                f"Valid: {sorted(VIBRAVOX_CHANNELS)}")
        if self.ref not in VIBRAVOX_CHANNELS:
            raise ValueError(
                f"ref {self.ref!r} is not a real Vibravox channel. "
                f"Valid: {sorted(VIBRAVOX_CHANNELS)}")
        if self.sensor not in VIBRAVOX_BODY_SENSORS:
            print(f"[VibravoxAdapter] WARNING: sensor {self.sensor!r} is an "
                  f"in-ear ACOUSTIC mic, not a body-conduction pickup — its "
                  f"bandwidth is much wider than the target sensor type. "
                  f"Body sensors: {sorted(VIBRAVOX_BODY_SENSORS)}")
        if self.sensor == self.ref:
            raise ValueError("sensor and ref must be different channels")

    # --- parquet mode (the bounded, practical path) -------------------------
    def _init_parquet(self, cfg: dict):
        import pyarrow.parquet as pq
        files = cfg.get("parquet_files") or []
        if isinstance(files, str):
            files = [files]
        # support glob via pathlib
        expanded: list[str] = []
        for f in files:
            if any(c in f for c in "*?["):
                expanded.extend(sorted(str(p) for p in Path(".").glob(f)))
            else:
                expanded.append(f)
        if not expanded:
            raise ValueError("mode='parquet' requires cfg['parquet_files'] "
                             "(list of local .parquet shard paths)")
        cols = [f"audio.{self.sensor}", f"audio.{self.ref}",
                "speaker_id", "sentence_id", "duration", "raw_text"]
        loaded = 0
        for path in expanded:
            if not Path(path).exists():
                print(f"[VibravoxAdapter] skip missing parquet: {path}")
                continue
            pf = pq.ParquetFile(path)
            try:
                tbl = pf.read(columns=cols)
            except Exception as e:
                print(f"[VibravoxAdapter] {path}: read failed: {e}")
                continue
            n = tbl.num_rows
            sen_col = tbl.column(f"audio.{self.sensor}")
            ref_col = tbl.column(f"audio.{self.ref}")
            spk_col = tbl.column("speaker_id")
            utt_col = tbl.column("sentence_id")
            dur_col = tbl.column("duration")
            txt_col = tbl.column("raw_text")
            for i in range(n):
                if loaded >= self.max_items:
                    break
                try:
                    s_wav, s_sr = _decode_audio_bytes(sen_col[i].as_py())
                    r_wav, r_sr = _decode_audio_bytes(ref_col[i].as_py())
                except Exception as e:
                    print(f"[VibravoxAdapter] row {i} decode failed: {e}; skip")
                    continue
                s4 = _resample(s_wav, s_sr, self.sr)
                s4 = self._apply_sensor_lowpass(s4)   # T11 §5: align to target device
                r4 = _resample(r_wav, r_sr, self.sr)
                meta = {
                    "sr": self.sr,
                    "sensor_type": self.sensor,
                    "speaker_id": str(spk_col[i].as_py()),
                    "sentence_id": str(utt_col[i].as_py()),
                    "utterance_id": f"{spk_col[i].as_py()}_{utt_col[i].as_py()}",
                    "duration_s": float(dur_col[i].as_py()) if dur_col[i].as_py() is not None else None,
                    "text": str(txt_col[i].as_py() or "")[:80],
                    "source": str(path),
                }
                self._items.append((s4, r4, meta))
                loaded += 1
            if loaded >= self.max_items:
                break
        print(f"[VibravoxAdapter] parquet mode: loaded {len(self._items)} "
              f"pairs from {len(expanded)} shard(s) "
              f"(sensor={self.sensor}, ref={self.ref}, sr_in=48000→{self.sr})")

    # --- hf streaming mode (corrected: name= + nested audio keys) -----------
    def _init_hf(self, cfg: dict):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "hf mode needs `datasets`; or use mode='parquet' with local shards."
            ) from e
        config = cfg.get("hf_config", "speech_clean")
        split = cfg.get("hf_split", "test")
        cache = cfg.get("hf_cache_dir",
                        os.path.expanduser("~/.cache/vibravox"))
        ds = load_dataset("Cnam-LMSSC/vibravox", name=config, split=split,
                          cache_dir=cache, streaming=True)
        loaded = 0
        for item in ds:
            if loaded >= self.max_items:
                break
            aud = item.get("audio")
            if not isinstance(aud, dict):
                continue
            sen_cell = aud.get(self.sensor)
            ref_cell = aud.get(self.ref)
            if not sen_cell or not ref_cell:
                continue
            try:
                s_wav, s_sr = _decode_audio_bytes(sen_cell)
                r_wav, r_sr = _decode_audio_bytes(ref_cell)
            except Exception as e:
                print(f"[VibravoxAdapter] hf row decode failed: {e}; skip")
                continue
            s4 = _resample(s_wav, s_sr, self.sr)
            s4 = self._apply_sensor_lowpass(s4)   # T11 §5
            r4 = _resample(r_wav, r_sr, self.sr)
            meta = {
                "sr": self.sr,
                "sensor_type": self.sensor,
                "speaker_id": str(item.get("speaker_id", "")),
                "sentence_id": str(item.get("sentence_id", "")),
                "utterance_id": f"{item.get('speaker_id','?')}_{item.get('sentence_id','?')}",
                "duration_s": float(item.get("duration", 0.0) or 0.0),
                "text": str(item.get("raw_text", "") or "")[:80],
                "source": f"hf:{config}/{split}",
            }
            self._items.append((s4, r4, meta))
            loaded += 1
        print(f"[VibravoxAdapter] hf mode: loaded {len(self._items)} "
              f"streamed pairs (config={config}, split={split})")

    # --- Dataset protocol ---------------------------------------------------
    def __len__(self):
        # n_repeat random crops per loaded row
        return len(self._items) * self.n_repeat

    def _crop(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        T = self.segment_len
        if len(wav) >= T:
            if self.crop == "center":
                start = (len(wav) - T) // 2
            else:
                start = int(rng.integers(0, len(wav) - T + 1))
            return wav[start:start + T].copy()
        return np.pad(wav, (0, T - len(wav)))

    def __getitem__(self, i):
        row_idx = i % len(self._items)
        s4, r4, meta = self._items[row_idx]
        rng = np.random.default_rng(self.seed + i * 7919)
        sensor = self._crop(s4, rng)
        # Use the SAME start offset for ref so sensor↔ref stay time-aligned.
        T = self.segment_len
        if len(r4) >= T:
            start = (len(r4) - T) // 2 if self.crop == "center" else \
                    max(0, min(int(rng.integers(0, len(r4) - T + 1)), len(r4) - T))
            ref = r4[start:start + T].copy()
        else:
            ref = np.pad(r4, (0, T - len(r4)))

        sensor_t = torch.from_numpy(sensor).float()
        ref_t = torch.from_numpy(ref).float()
        if self.normalize:
            sp = sensor_t.abs().max().clamp_min(1e-6)
            rp = ref_t.abs().max().clamp_min(1e-6)
            sensor_t = sensor_t / sp
            ref_t = ref_t / rp
        if self.augment and self.deg_cfg is not None:
            sensor_t = apply_degradation(sensor_t, self.deg_cfg, rng=rng, n_fft=128)

        return {
            "sensor": sensor_t,
            "ref": ref_t,
            "meta": dict(meta),
        }
