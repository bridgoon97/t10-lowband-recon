"""Complex-spectrum losses — the primary training objective (spec change).

Spec change (口径迁移): arms output a TRUNCATED COMPLEX spectrum (B, keep_bins,
N); the target is the reference mic's truncated complex spectrum.  Phase is now
the MODEL's job (learned), not oracle.

Loss = magnitude term (L1/L2/dB on |spec| — kept as the main term) + complex
MSE on real/imag (the phase-sensitive term).  Weight 1:1 first, tune by
param-side gradient norm (NOT by ear).

⚠️ Honest expectation: 500–2000 Hz phase is unobserved by the model (input above
~500 Hz is noise floor), so the complex term's early metrics will likely be
WORSE than a magnitude+oracle-phase baseline — this is expected, not a bug.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Magnitude (L1/L2/dB) + complex MSE on the truncated complex spectrum.

    §5.7: log terms use eps calibrated to ≈ −80 dBFS, not 1e-8.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.5,
                 db_weight: float = 1.0, cplx_weight: float = 1.0,
                 eps_db: float = -80.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.db_weight = db_weight
        self.cplx_weight = cplx_weight
        self.eps_db = eps_db
        self.eps_lin = 10.0 ** (eps_db / 20.0)

    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> dict:
        # pred_spec, target_spec: (B, F, N) complex64
        pred_mag = pred_spec.abs().clamp_min(self.eps_lin)
        tgt_mag = target_spec.abs().clamp_min(self.eps_lin)
        l1 = F.l1_loss(pred_mag, tgt_mag)
        l2 = F.mse_loss(pred_mag, tgt_mag)
        db_l1 = F.l1_loss(20.0 * torch.log10(pred_mag),
                          20.0 * torch.log10(tgt_mag))
        # complex MSE (phase-sensitive): real + imag, equal weight
        cplx = (F.mse_loss(pred_spec.real, target_spec.real)
                + F.mse_loss(pred_spec.imag, target_spec.imag))
        total = (self.l1_weight * l1 + self.l2_weight * l2
                 + self.db_weight * db_l1 + self.cplx_weight * cplx)
        return {"loss": total, "l1": l1.detach(), "l2": l2.detach(),
                "db_l1": db_l1.detach(), "cplx": cplx.detach()}


def reconstruct_waveform_with_oracle_phase(spec: torch.Tensor,
                                            target_wav: torch.Tensor,
                                            stft_cfg) -> torch.Tensor:
    """Reconstruct a band-limited waveform from a (truncated) complex spectrum.

    spec change: the model already outputs a COMPLEX spectrum (phase learned),
    so there is no oracle phase to graft — we just invert the (truncated) spec.
    The truncated keep_bins complex spectrum is zero-padded to the full n_fft
    bin count (with conjugate symmetry for the upper half) and istft'd, giving
    a 0–2 kHz band-limited waveform for optional waveform-based losses.
    """
    from ..dsp.stft import causal_istft
    n_full = stft_cfg.num_bins
    B, Fb, N = spec.shape
    # spec is the KEPT spectrum (bins 1..Fb, DC dropped).  Reconstruct the full
    # rfft bin set: bin 0 (DC) = 0, bins 1..Fb = spec, bins Fb+1.. = 0 →
    # a 0–2 kHz band-limited waveform.  No manual hermitian (istft takes the
    # one-sided rfft directly).
    full = torch.zeros(B, n_full, N, device=spec.device, dtype=spec.dtype)
    full[:, 1:1 + Fb] = spec
    wav = causal_istft(full, stft_cfg, length=target_wav.shape[-1])
    return wav
