import numpy as np, torch
from lowband.data import build_dataset
from lowband.data.adapter import PairedSpeechDataset

cfg = dict(adapter='vibravox', mode='parquet',
           parquet_files=['data/vibravox_parquet/speech_clean_test_0.parquet'],
           sensor='forehead_accelerometer', ref='headset_microphone',
           segment_len=4000, sr=4000, max_items=20, n_repeat=2, crop='random',
           normalize=True, seed=42)
ds = build_dataset(cfg)
print('len:', len(ds), ' (expect 20*2=40)')
print('Protocol check:', isinstance(ds, PairedSpeechDataset))
b = ds[0]
print('keys:', list(b.keys()))
print('sensor:', tuple(b['sensor'].shape), b['sensor'].dtype,
      'min/max', float(b['sensor'].min()), float(b['sensor'].max()))
print('ref   :', tuple(b['ref'].shape), b['ref'].dtype,
      'min/max', float(b['ref'].min()), float(b['ref'].max()))
print('meta  :', b['meta'])
assert b['meta']['sr'] == 4000
assert b['sensor'].shape[0] == 4000 and b['ref'].shape[0] == 4000

def band_ratio(t):
    w = t.numpy()
    sp = np.abs(np.fft.rfft(w)); f = np.fft.rfftfreq(len(w), 1/4000)
    return np.sqrt(np.mean(sp[f >= 1000]**2)) / (np.sqrt(np.mean(sp[f < 1000]**2)) + 1e-9)

rows = []
for i in range(5):
    bb = ds[i]
    s = band_ratio(bb['sensor']); r = band_ratio(bb['ref'])
    rows.append((s, r, s <= r + 0.05))
print('band(sensor,ref):', [f'{s:.3f}/{r:.3f}' for s, r, _ in rows])
print('sensor more band-limited in all 5:', all(c for _, _, c in rows))
spk = set(ds[i]['meta']['speaker_id'] for i in range(min(20, len(ds))))
print('distinct speakers in first 20 items:', len(spk))
print('ALL GOOD')
