"""Arm C overfit localization (review ④): is the failure phase-representation
or capacity or a bug?  Run 4 conditions and compare ratios.

⚠️ SUPERSEDED (review finding F): this probe ran under the OLD big target
(B=8×T=16000 = 102,400 complex values vs C's 13,122 params = target OUTRUNS
params 7.8×), so NO condition could overfit — including (a) magnitude-only,
whose 0.31 was misread as 'phase bottleneck'.  The real bottleneck was target
SIZE.  tests/test_overfit.py now uses B=1×T=4000 (3,200 values, 4.1× headroom)
+ a REPRESENTABLE smooth-formant target + self-overfit: C reaches ratio 0.006
(<0.1) at 1000 steps/lr=1e-2 — C overfits FINE; the old 'phase bottleneck'
conclusion is RETRACTED (see known_issues.md C2).  This script is kept only as
the capacity-confound evidence that motivated the fix.

(a) magnitude-only loss (cplx_weight=0): if C overfits -> bottleneck is PHASE
    representation (my explanation holds, WITH evidence).
(b1) 2 samples instead of 8: if overfits -> was CAPACITY (8 too many to memorize).
(b2) half segment (T=8000): if overfits -> capacity.
baseline: 8 samples, full complex (the current relaxed-threshold case).
"""
import torch
from lowband import build_model
from lowband.dsp.stft import StftConfig, complex_stft_truncated
from lowband.dsp import ddsp as d
from lowband.losses.spectral import SpectralLoss

BASE = {"sample_rate": 16000, "stft_n_fft": 512, "stft_hop": 160,
        "stft_win": 480, "keep_bins": 64, "band_top_hz": 2000}
SR, N_HARM, F0 = 16000, 32, 150.0
SC = StftConfig(n_fft=512, hop=160, win=480, keep_bins=64)


def harm_ref(B, T):
    amps = torch.rand(B, N_HARM, T) * 0.5
    phase = d.accumulate_phase(torch.full((B, T), F0), T, SR)
    mask = d.harmonic_index_mask(torch.tensor([F0]), N_HARM, 2000.0)
    return d.harmonic_synth(phase, amps, mask)


def run(B, T, steps, cplx_w):
    torch.manual_seed(0)
    ref = harm_ref(B, T)
    rs = complex_stft_truncated(ref, SC)
    torch.manual_seed(42)
    m = build_model(dict(BASE, arm="arm_c_ftlstm", f0_mode="oracle"))
    lf = SpectralLoss(cplx_weight=cplx_w)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    x = torch.randn(B, T)
    li = lf_ = None
    for s in range(steps):
        opt.zero_grad()
        out = m(x, None)
        N = min(out["spec"].shape[-1], rs.shape[-1])
        ld = lf(out["spec"][..., :N], rs[..., :N])
        ld["loss"].backward()
        opt.step()
        if s == 0: li = ld["loss"].item()
        lf_ = ld["loss"].item()
    return li, lf_


cases = [
    ("baseline 8samp full-complex", 8, 16000, 300, 1.0),
    ("(a) 8samp MAG-only cplx=0",   8, 16000, 300, 0.0),
    ("(b1) 2samp full-complex",     2, 16000, 300, 1.0),
    ("(b2) 8samp T=8000 full-cplx", 8,  8000, 300, 1.0),
]
print(f"{'case':<32}{'first':>9}{'last':>9}{'ratio':>8}")
for name, B, T, st, cw in cases:
    li, la = run(B, T, st, cw)
    print(f"{name:<32}{li:>9.2f}{la:>9.2f}{la/(li+1e-8):>8.3f}")
