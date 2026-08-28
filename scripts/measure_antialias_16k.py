"""Measure masked vs unmasked anti-alias metrics at the NEW口径 (16 kHz).
mask cuts at band_top=2000 (not Nyquist 8000).  Aliasing (un-masked >2 kHz
harmonics) lands in the 2–8 kHz band, NOT in 0–2 kHz (which is kept).
"""
import math
import torch
from lowband.dsp import ddsp as ddsp_mod
from lowband.dsp.stft import StftConfig, stft

SR = 16000
NYQ = SR / 2          # 8000
BAND_TOP = 2000.0
N_FFT = 512
T = SR                # 1 s
F0 = 150.0
MAX_HARM = 32

phase = ddsp_mod.accumulate_phase(torch.full((1, T), F0), T, SR)
amps = torch.ones(1, MAX_HARM, T)
cfg = StftConfig(n_fft=N_FFT, hop=160, win=480, center=True)


def spec_of(mask):
    wav = ddsp_mod.harmonic_synth(phase, amps, mask)
    _, mag = stft(wav, cfg)         # (B, 257, N) full
    return mag.mean(dim=-1)[0]      # (257,) time-averaged


# harmonic bins (k*150 in 0..2000 → k=1..13) and the 2–8 kHz band (bins 64..256)
bin_hz = SR / N_FFT                  # 31.25
harm_bins = set()
for k in range(1, int(BAND_TOP // F0) + 1):
    b = round(k * F0 / bin_hz)
    harm_bins.update(range(max(0, b - 1), min(257, b + 2)))
low_bins = [b for b in range(65) if b not in harm_bins]        # inter-harm 0–2k
hi_bins = list(range(65, 257))                                # the 2–8 kHz band
b1125 = round(7.5 * F0 / bin_hz)
db = lambda r: 20 * math.log10(r + 1e-12)

for label, mask in [
    ("MASKED (band_top=2k)", ddsp_mod.harmonic_index_mask(torch.tensor([F0]), MAX_HARM, BAND_TOP)),
    ("UNMASKED (bypass)", torch.ones_like(ddsp_mod.harmonic_index_mask(torch.tensor([F0]), MAX_HARM, BAND_TOP))),
]:
    m = spec_of(mask)
    hidx = torch.tensor(sorted(harm_bins))
    hi_idx = torch.tensor(hi_bins)
    peak = m[hidx].max().item() if harm_bins else 1.0
    hi_max = m[hi_idx].max().item()
    hi_sum = m[hi_idx].sum().item()
    v1125 = m[b1125].item()
    print(f"--- {label} ---")
    print(f"  peak(0-2k harm)   = {peak:.3f}")
    print(f"  MAX bin 2-8k /peak = {hi_max/peak:.4f}  ({db(hi_max/peak):+.1f} dB)")
    print(f"  SUM 2-8k /peak     = {hi_sum/peak:.4f}  ({db(hi_sum/peak):+.1f} dB)")
    print(f"  @1125Hz/peak       = {v1125/peak:.4f}  ({db(v1125/peak):+.1f} dB)")
print(f"\nbin_hz={bin_hz:.2f}, 1125Hz->bin {b1125}={b1125*bin_hz:.1f}Hz, 2-8k=bins 65..256")
