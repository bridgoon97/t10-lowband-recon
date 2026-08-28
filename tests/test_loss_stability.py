"""§5.7 — Loss numerical stability.

log terms use eps calibrated to ≈ −80 dBFS (not 1e-8).  Feeding silence,
all-zero, extreme values, NaN — loss must not produce NaN/Inf.  Padding
regions must be masked.
"""
import torch

from lowband.losses.spectral import SpectralLoss
from lowband.losses.stft_loss import MultiResolutionSTFTLoss


def test_spectral_loss_stability():
    loss_fn = SpectralLoss()
    # spec change: SpectralLoss takes COMPLEX spectra (B, F, N)
    def c(re, im=None):
        im = torch.zeros_like(re) if im is None else im
        return torch.complex(re, im)
    cases = {
        "normal": (c(torch.randn(2, 64, 100).abs()), c(torch.randn(2, 64, 100).abs())),
        "silence": (c(torch.zeros(2, 64, 100)), c(torch.zeros(2, 64, 100))),
        "zero_pred": (c(torch.zeros(2, 64, 100)), c(torch.rand(2, 64, 100))),
        "zero_target": (c(torch.rand(2, 64, 100)), c(torch.zeros(2, 64, 100))),
        "extreme": (c(torch.full((2, 64, 100), 1e10)), c(torch.rand(2, 64, 100))),
        "tiny": (c(torch.full((2, 64, 100), 1e-10)), c(torch.rand(2, 64, 100))),
    }
    for name, (pred, target) in cases.items():
        out = loss_fn(pred, target)
        l = out["loss"]
        is_nan = torch.isnan(l).any().item()
        is_inf = torch.isinf(l).any().item()
        print(f"  spectral[{name}]: loss={l.item():.4f} "
              f"{'NaN!' if is_nan else ''}{'Inf!' if is_inf else ''}")
        assert not is_nan and not is_inf, f"NaN/Inf for case '{name}'"


def test_mr_stft_stability():
    mrstft = MultiResolutionSTFTLoss()
    cases = {
        "normal": (torch.randn(2, 4000), torch.randn(2, 4000)),
        "silence": (torch.zeros(2, 4000), torch.zeros(2, 4000)),
        "extreme": (torch.full((2, 4000), 10.0), torch.randn(2, 4000)),
    }
    for name, (pred, target) in cases.items():
        l = mrstft(pred, target)
        is_nan = torch.isnan(l).any().item()
        is_inf = torch.isinf(l).any().item()
        print(f"  mrstft[{name}]: loss={l.item():.4f} "
              f"{'NaN!' if is_nan else ''}{'Inf!' if is_inf else ''}")
        assert not is_nan and not is_inf, f"NaN/Inf for case '{name}'"


if __name__ == "__main__":
    test_spectral_loss_stability()
    test_mr_stft_stability()
