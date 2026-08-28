# How to Plug In Your Own Data

§4.2 deliverable: switching datasets = writing one adapter + changing one
config line. No model or training code changes.

## Quick start

1. Copy `lowband/data/template_adapter.py` → `my_adapter.py`
2. Fill in the `TODO` sections (file loading, pairing, resampling)
3. Add to config:
   ```yaml
   data:
     adapter: template   # or register a new name
     data_root: /path/to/your/data
     sensor_glob: "**/sensor/*.wav"
     ref_glob: "**/ref/*.wav"
     target_sr: 4000
   ```

## The interface contract

Every adapter must return a dict per item:

```python
{
    "sensor": torch.Tensor (T,),   # body-conduction signal @4kHz, normalized
    "ref":    torch.Tensor (T,),   # reference air-conduction @4kHz, normalized
    "meta":   {"sr": 4000, "sensor_type": "...", "utterance_id": "..."}
}
```

- Both tensors must be the **same length** (set `segment_len` in config)
- Both must be **resampled to 4 kHz** (the model expects this)
- Both should be **peak-normalized** to [-1, 1] (set `normalize: true`)

## The three built-in adapters

| Adapter | Config name | Use case |
|---------|-------------|----------|
| `LowpassSimAdapter` | `lowpass_sim` | L0: synthetic degradation of clean speech |
| `VibravoxAdapter` | `vibravox` | L1: real body-conduction recordings |
| `TemplateAdapter` | `template` | Copy this for your own data |

## Pairing sensor ↔ reference

The template uses filename-stem matching: `sensor/001.wav` ↔ `ref/001.wav`.

For other pairing strategies:
- **Index-based:** pair by position in sorted file lists
- **Custom:** override `__init__` to build `self._pairs` however you need

## Augmentation

If your data is clean (L0-style), enable degradation augmentation:

```yaml
data:
  adapter: template
  augment: true
  degradation:
    cutoff_min: 300
    cutoff_max: 1200
    time_vary: true
    spectral_tilt: true
    body_noise: true
```

Every degradation effect is independently switchable (§4.3) and unit-tested (§5.8).

## Native format support

The template uses `soundfile` (libsndfile) for WAV. For other formats:
- **MAT:** `scipy.io.loadmat`
- **HDF5:** `h5py`
- **Custom binary:** read with `np.fromfile` + your header parser
- **Database:** query in `__init__`, cache paths, load in `__getitem__`

Replace the `soundfile.read` calls in `__getitem__` with your loader. Everything
else (resampling, segmenting, normalization, augmentation) is handled.
