"""Measure masked vs unmasked anti-alias metrics to set absolute dB thresholds.

We want thresholds that the CORRECT (masked) version passes with margin and the
BROKEN (unmasked, aliasing) version fails clearly.
"""
import math
import torch
from lowband.dsp import ddsp as ddsp_mod
from lowband.dsp.stft import StftConfig, stft

sr = 4000
nyquist = sr / 2
n_fft = 128
T = 4000
f0 = 150.0
max_harm = 32
cfg = StftConfig(n_fft=n_fft, hop=32, win=128, center=True)

phase = ddsp_mod.accumulate_phase(torch.full((1, T), f0), T, sr)
amps = torch.ones(1, max_harm, T)
mask = ddsp_mod.harmonic_index_mask(torch.tensor([f0]), max_harm, nyquist)
wav_m = ddsp_mod.harmonic_synth(phase, amps, mask)
wav_u = ddsp_mod.harmonic_synth(phase, amps, torch.ones_like(mask))
_, mag_m = stft(wav_m, cfg)
_, mag_u = stft(wav_u, cfg)

n_bins = mag_m.shape[1]
freqs = torch.linspace(0, nyquist, n_bins)   # 0..2000, 65 bins
bin_hz = nyquist / (n_bins - 1)              # 31.25 Hz/bin

# harmonic bins (within 1 bin of k*150 for k=1..13)
harm_bins = set()
for k in range(1, 14):
    b = int(round(k * f0 / bin_hz))
    harm_bins.update(range(max(0, b - 1), min(n_bins, b + 2)))
inter_idx = sorted(set(range(n_bins)) - harm_bins)
hi_inter_idx = [b for b in inter_idx if freqs[b] >= 1000]   # 1-2kHz inter-harmonic
idx_1125 = int(round(7.5 * f0 / bin_hz))                     # exactly inter-harmonic

it = torch.tensor(inter_idx)
hit = torch.tensor(hi_inter_idx)


def stats(mag, label):
    peak = mag.max().item()
    inter_sum = mag[:, it, :].sum().item()
    hi_inter_sum = mag[:, hit, :].sum().item() if len(hi_inter_idx) else 0.0
    v1125 = mag[:, idx_1125, :].mean().item()
    # per-bin (bin-count-INDEPENDENT) ratios — the physical floor to gate on
    inter_max = mag[:, it, :].max().item()                          # worst inter-harm bin
    hi_inter_max = mag[:, hit, :].max().item() if len(hi_inter_idx) else 0.0  # worst 1-2kHz inter bin
    db = lambda r: 20 * math.log10(r + 1e-12)
    print(f"--- {label} ---")
    print(f"  peak                     = {peak:.3f}")
    print(f"  inter-harm sum (26 bins)  = {inter_sum:.3f}   /peak={inter_sum/peak:.4f}  ({db(inter_sum/peak):+.1f} dB)  [bin-count dep, ref only]")
    print(f"  MAX inter-harm bin /peak  = {inter_max/peak:.4f}  ({db(inter_max/peak):+.1f} dB)")
    print(f"  MAX hi-inter(1-2k) /peak  = {hi_inter_max/peak:.4f}  ({db(hi_inter_max/peak):+.1f} dB)")
    print(f"  @1125Hz bin /peak        = {v1125/peak:.4f}  ({db(v1125/peak):+.1f} dB)")
    print(f"  (bin_hz={bin_hz:.2f}, 1125Hz->bin {idx_1125}={freqs[idx_1125]:.1f}Hz)")


stats(mag_m, "MASKED (correct)")
stats(mag_u, "UNMASKED (aliasing/broken)")
print(f"\nbin 1125 freq={freqs[idx_1125]:.1f}Hz ; inter bins count={len(inter_idx)} hi_inter count={len(hi_inter_idx)}")
