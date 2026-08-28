"""§5.6 — STFT/iSTFT roundtrip + COLA (new口径: 16 kHz, n_fft=512/hop=160/win=480).

Three layers:
  * COLA for the deployed window/hop (Hann 480 @hop 160 = win/3, 3× overlap).
  * torch center=True roundtrip (stft→istft) — a sanity that the torch pair is
    self-consistent at the new口径.
  * CAUSAL roundtrip (causal_stft→causal_istft) — the pair actually used in
    training (reconstruct_waveform_with_oracle_phase).  This is the regression
    test for review finding C: torch.istft(center=True) is NOT the inverse of
    causal_stft (left-pad framing + left-aligned window vs torch's centered
    window + center framing) and produced a shifted/phase-ramped waveform
    (rel err ~1.3) feeding MR-STFT loss + discriminator.  causal_istft (WOLA,
    synth=analysis window, normalized OLA) is the exact inverse (rel ~1e-7).
"""
import torch

from lowband.dsp.stft import (StftConfig, stft, istft, causal_stft,
                                    causal_istft, cola_check)


def test_cola():
    """COLA condition for the deployed Hann window at win/2 and win/3 hops."""
    for (n_fft, hop, win) in [(512, 160, 480), (512, 240, 480), (128, 64, 128)]:
        cfg = StftConfig(n_fft=n_fft, hop=hop, win=win, window="hann")
        err = cola_check(cfg)
        print(f"  COLA (hann, win={win}, hop={hop}, n_fft={n_fft}): "
              f"rel_err={err:.2e} {'PASS' if err < 1e-6 else 'CHECK'}")
        assert err < 1e-3, f"COLA violated for win={win}/hop={hop}"


def test_roundtrip():
    """torch center=True STFT → iSTFT reconstructs the signal (new口径)."""
    cfg = StftConfig(n_fft=512, hop=160, win=480, window="hann", center=True)
    T = 16000
    x = torch.randn(2, T)
    spec, _ = stft(x, cfg)
    wav = istft(spec, cfg, length=T)
    N = min(x.shape[-1], wav.shape[-1])
    skip = 512
    diff = (x[..., skip:N - skip] - wav[..., skip:N - skip]).abs()
    rel = diff.max().item() / (x.abs().max().item() + 1e-8)
    print(f"  torch roundtrip (center=True, win=480/hop=160): "
          f"rel_err={rel:.2e} {'PASS' if rel < 1e-5 else 'FAIL'}")
    assert rel < 1e-4, f"torch roundtrip failed: {rel}"


def test_causal_roundtrip():
    """causal_stft → causal_istft reconstructs x (the TRAINING pair, full-bin).

    This is the regression test for review finding C.  The OLD code used
    torch.istft(center=True) to invert causal_stft spectra — which does NOT
    match (left-pad framing + left-aligned window vs torch's centered) and
    produced rel ~1.3 (a shifted + cross-bin-phase-ramped waveform).  We assert
    BOTH: causal_istft reconstructs (<1e-5) AND the old istft does NOT (>1e-2),
    so the test documents why causal_istft exists and will fail if someone
    reverts to istft.
    """
    cfg = StftConfig(n_fft=512, hop=160, win=480, window="hann", keep_bins=64)
    T = 16000
    torch.manual_seed(0)
    x = torch.randn(1, T)
    spec, _ = causal_stft(x, cfg)                 # full 257-bin, causal convention

    y_causal = causal_istft(spec, cfg, length=T)  # the FIXED path
    skip = 512
    N = min(T, y_causal.shape[-1])
    rel_causal = (x[..., skip:N - skip] - y_causal[..., skip:N - skip]).abs().max().item() \
        / (x.abs().max().item() + 1e-8)

    y_torch = istft(spec, cfg, length=T)          # the OLD buggy path (center=True)
    Nt = min(T, y_torch.shape[-1])
    rel_torch = (x[..., skip:Nt - skip] - y_torch[..., skip:Nt - skip]).abs().max().item() \
        / (x.abs().max().item() + 1e-8)

    print(f"  causal_istft  mid rel_err = {rel_causal:.2e} "
          f"{'PASS' if rel_causal < 1e-5 else 'FAIL'}")
    print(f"  istft(torch) mid rel_err = {rel_torch:.2e} "
          f"({'bug confirmed' if rel_torch > 1e-2 else '??'})")
    assert rel_causal < 1e-5, f"causal roundtrip failed: {rel_causal}"
    assert rel_torch > 1e-2, (
        "istft(torch,center=True) now matches causal_stft — if so the mismatch "
        "is gone and causal_istft may be unnecessary; revisit. (finding C)")


if __name__ == "__main__":
    test_cola()
    test_roundtrip()
    test_causal_roundtrip()
    print("STFT roundtrip tests (new口径): all PASS")
