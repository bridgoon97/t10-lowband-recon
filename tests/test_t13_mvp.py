"""T13-MVP functional tests — one direct test per pre-fixed acceptance criterion.

M1 safety fallback      — strength=0 (CLI) and forced veto ⇒ Y ≡ S (allclose).
M2 numeric & interface  — no NaN/Inf; CLI contract errors; batch≡streaming.
M3 intervene when hurt  — S 100–800 Hz −20 dB (post-freeze), V*=X ⇒ band LSD
                          strictly improves, coverage > 0.
M4 no harm when clean   — S=X, V=X ⇒ 100–800 Hz |correction| p99 ≤ 1 dB, no
                          large reverse correction.

BOUNDARY: synthetic test-side signals only (no 0625, no user data).  X (the
clean reference) appears ONLY on the test/evaluation side — it is never read by
the production path (which only sees S and V).  References are the INPUTS
(S, X), never the output under test.  These are the four pre-declared MVP
blockers; the historical suites stay report-only and are NOT re-run here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from fusion import FusionConfig, Fusion, FusionStreamer, stft_batch
from fusion.f0 import f0_batch

SR = 16000
WIN = 480


# ----------------------------------------------------------------- helpers --
def _speechish(T_s: float, f0: float = 110.0, seed: int = 3) -> torch.Tensor:
    """Harmonic-rich voiced signal (male-F0 range) with slow envelope, 1/k tilt,
    tiny noise.  Test-side only."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(T_s * SR)) / SR
    env = 0.55 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + 0.3))
    x = np.zeros_like(t)
    for k in range(1, 25):
        ph = rng.uniform(0, 2 * np.pi)
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + ph)
    x = x / (np.abs(x).max() + 1e-9) * 0.4 * env
    x += rng.normal(0, 1e-3, t.shape)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)   # (1, T)


def _band_bins(cfg, lo_hz, hi_hz):
    bz = cfg.sr / cfg.n_fft
    return max(1, int(lo_hz / bz)), min(cfg.fusion_hi_bin, int(hi_hz / bz))


def _damage_inband(x: torch.Tensor, from_s: float, depth_db: float = 20.0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale S's 100–800 Hz bins by −depth_db for frame times ≥ from_s.
    Returns (S_damaged, frame_time_mask).  Frame n ≈ sample n·hop.
    NOTE: the modified spec is resynthesized on a LONGER tail and trimmed back
    — the causal iSTFT's window-overlap sum decays over the last ~win samples,
    and a modified tail frame otherwise leaves a boundary artifact there."""
    cfg = FusionConfig()
    T = x.shape[-1]
    pad = torch.zeros(1, cfg.win + cfg.hop)
    x_long = torch.cat([x, pad], dim=-1)
    spec = stft_batch(x_long, cfg)
    lo, hi = _band_bins(cfg, 100.0, 800.0)
    N = spec.shape[-1]
    ft = torch.arange(N).float() * cfg.hop / SR
    mask = (ft >= from_s) & (ft < T / SR)                 # (N,)
    scale = (10.0 ** (-depth_db / 20.0))
    spec_d = spec.clone()
    spec_d[0, lo:hi + 1, mask] = spec_d[0, lo:hi + 1, mask] * scale
    from fusion import istft_batch
    s_d = istft_batch(spec_d, cfg, length=x_long.shape[-1])[:, :T]
    return s_d, mask[:int(T / SR / cfg.hop)]


def _band_lsd(a: torch.Tensor, b: torch.Tensor, cfg, mask=None) -> float:
    """Mean band LSD (100–800 Hz) between two waveforms via the same STFT."""
    sa = stft_batch(a, cfg)[0].abs().clamp_min(1e-8)   # (Fb, N)
    sb = stft_batch(b, cfg)[0].abs().clamp_min(1e-8)
    lo, hi = _band_bins(cfg, 100.0, 800.0)
    d = 20 * (torch.log10(sa[lo:hi + 1]) - torch.log10(sb[lo:hi + 1]))
    if mask is not None:
        d = d[:, mask]
    return float(d.pow(2).mean().sqrt())


# ------------------------------------------------------------------- M1 -----
def test_M1_safety_fallback_strength0_and_forced_veto():
    """M1: strength=0 (real CLI end-to-end) and a forced veto both give Y ≡ S
    (torch.allclose rtol=1e-5, atol=1e-4) on the valid interior — through the
    REAL production path (Fusion + CLI), not via aux/w."""
    cfg = FusionConfig()
    T = 4.0
    x = _speechish(T)
    s_d, _ = _damage_inband(x, from_s=2.0)
    skip = 2 * WIN
    N = x.shape[-1]
    # roundtrip control: the valid region where stft→istft is exact for S itself
    from fusion import istft_batch
    s_rt = istft_batch(stft_batch(s_d, cfg), cfg, length=N)
    assert torch.allclose(s_rt[..., skip:N - skip], s_d[..., skip:N - skip],
                          rtol=1e-5, atol=1e-4), "STFT roundtrip not exact in region"
    # (a) CLI --strength 0 end-to-end (WAV in/out through the real entry point)
    tmp = Path("/tmp/t13_mvp_m1"); tmp.mkdir(exist_ok=True)
    sf.write(tmp / "s.wav", s_d[0].numpy(), SR, subtype="PCM_16")
    sf.write(tmp / "v.wav", x[0].numpy(), SR, subtype="PCM_16")
    r = subprocess.run([sys.executable, "-m", "fusion.run_fusion",
                        "--stage2", str(tmp / "s.wav"), "--vpu", str(tmp / "v.wav"),
                        "--output", str(tmp / "y0.wav"), "--strength", "0"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"CLI strength=0 failed: {r.stderr}"
    y0, sr = sf.read(tmp / "y0.wav", dtype="float32")
    assert sr == SR
    y0 = torch.from_numpy(y0).unsqueeze(0)
    # PCM_16 quantization (~3e-5) dominates: compare against the quantized S
    s_q = torch.clamp(s_d[0], -1, 1)
    s_q = torch.round(s_q * 32767) / 32767
    assert torch.allclose(y0[..., skip:N - skip],
                          s_q.unsqueeze(0)[..., skip:N - skip],
                          rtol=1e-5, atol=1e-4), "strength=0 output != S"
    # (b) forced safety veto through Fusion: MSC veto threshold unreachable
    cfg_v = FusionConfig().with_switches(mvp_veto_msc=1.1)   # MSC ≤ 1 < 1.1 ⇒ always veto
    fv = Fusion(cfg_v)
    with torch.no_grad():
        y_v = fv.process_batch(s_d, x)
    assert torch.allclose(y_v[..., skip:N - skip], s_d[..., skip:N - skip],
                          rtol=1e-5, atol=1e-4), "forced veto output != S"
    assert float(fv.last_diagnostics["coverage_100_800"]) == 0.0
    print("  M1 PASS: strength=0 (CLI) and forced veto both ≡ S "
          "(allclose 1e-5/1e-4, real production path)")


# ------------------------------------------------------------------- M2 -----
def test_M2_numeric_and_interface():
    """M2: no NaN/Inf; 16 kHz/length/strength CLI contract errors (non-zero
    exit + clear message); batch ≡ streaming on the MVP path."""
    cfg = FusionConfig()
    x = _speechish(3.0)
    s_d, _ = _damage_inband(x, from_s=1.5)
    f = Fusion(cfg)
    with torch.no_grad():
        y = f.process_batch(s_d, x)
    assert torch.isfinite(y).all(), "NaN/Inf in output"
    # streaming equivalence (same process_frame, separate instances)
    st = FusionStreamer(cfg)
    outs = []
    hop = cfg.hop
    for i in range(0, s_d.shape[-1] - hop + 1, hop):
        o = st.stream_step(s_d[:, i:i + hop], x[:, i:i + hop])
        if o is not None:
            outs.append(o)
    outs.append(st.flush())
    y_stream = torch.cat(outs, dim=-1)
    n = min(y.shape[-1], y_stream.shape[-1])
    interior = y[..., WIN:n - WIN] - y_stream[..., WIN:n - WIN]
    assert float(interior.abs().max()) < 1e-4, \
        f"batch vs streaming diff {float(interior.abs().max()):.2e}"
    # CLI contract errors → non-zero exit + clear message
    tmp = Path("/tmp/t13_mvp_m2"); tmp.mkdir(exist_ok=True)
    sf.write(tmp / "s.wav", s_d[0].numpy(), SR, subtype="PCM_16")
    sf.write(tmp / "v.wav", x[0].numpy(), SR, subtype="PCM_16")
    sf.write(tmp / "v_8k.wav", x[0].numpy(), 8000, subtype="PCM_16")   # wrong sr
    import soundfile as sf2
    v_long, _ = sf2.read(tmp / "v.wav", dtype="float32")
    sf2.write(tmp / "v_long.wav", np.concatenate([v_long, v_long]), SR, subtype="PCM_16")
    def run_cli(*extra):
        return subprocess.run([sys.executable, "-m", "fusion.run_fusion",
                               "--stage2", str(tmp / "s.wav"), "--vpu", str(tmp / "v.wav"),
                               "--output", str(tmp / "y.wav"), *extra],
                              capture_output=True, text=True, timeout=300)
    r = run_cli("--vpu", str(tmp / "v_8k.wav"))
    assert r.returncode != 0 and "sample rate" in (r.stderr + r.stdout).lower()
    r = run_cli("--vpu", str(tmp / "v_long.wav"))
    assert r.returncode != 0 and "length mismatch" in (r.stderr + r.stdout).lower()
    r = run_cli("--strength", "1.5")
    assert r.returncode != 0 and "strength" in (r.stderr + r.stdout).lower()
    print("  M2 PASS: finite output; sr/length/strength contract errors non-zero; "
          f"batch≡streaming (max interior diff {float(interior.abs().max()):.1e})")


# ------------------------------------------------------------------- M3 -----
def test_M3_intervene_when_clearly_damaged():
    """M3: S's 100–800 Hz lowered 20 dB AFTER the EQ freeze (2 s clean prefix —
    the deploy scenario AC2 was built for: C frozen at a healthy moment), V*=X,
    all safety conditions healthy ⇒ MVP output strictly closer to X than S in
    100–800 Hz (band LSD strictly down), intervention coverage > 0."""
    cfg = FusionConfig()
    T = 8.0
    x = _speechish(T)                       # X = clean reference (eval side only)
    s_d, mask = _damage_inband(x, from_s=2.0, depth_db=20.0)
    f = Fusion(cfg)                          # production path sees only (S, V=X)
    with torch.no_grad():
        y = f.process_batch(s_d, x)
    d = f.last_diagnostics
    # evaluate on the damaged region, ≥1 s after onset (smoother settled)
    N = stft_batch(x, cfg).shape[-1]
    ft = torch.arange(N).float() * cfg.hop / SR
    ev = ft >= 3.0
    lsd_s = _band_lsd(s_d, x, cfg, ev)
    lsd_y = _band_lsd(y, x, cfg, ev)
    assert lsd_y < lsd_s, f"no strict improvement: LSD(Y)={lsd_y:.2f} vs LSD(S)={lsd_s:.2f}"
    assert d["coverage_100_800"] > 0, "no intervention"
    # safety conditions actually healthy (vetoes did NOT do the work)
    assert d["veto_fraction_100_800"] < 0.2, f"vetoes fired: {d['veto_fraction_100_800']}"
    print(f"  M3 PASS: band LSD {lsd_s:.2f} -> {lsd_y:.2f} dB (strict improvement); "
          f"coverage={d['coverage_100_800']:.3f}; veto_frac={d['veto_fraction_100_800']:.3f}")
    print(f"  correction 100-800: p50={d['correction_100_800']['p50_db']:+.1f} dB, "
          f"max|.|={d['correction_100_800']['max_abs_db']:.1f} dB")
    for b, st in d["band_stats"].items():
        print(f"    {b:>8} Hz: p50={st['p50_db']:+6.1f}  p90|.|={st['p90_abs_db']:5.1f}  "
              f"max|.|={st['max_abs_db']:5.1f} dB")


# ------------------------------------------------------------------- M4 -----
def test_M4_no_harm_when_clean():
    """M4: S=X, V=X clean case ⇒ |correction| p99 ≤ 1 dB in 100–800 Hz and no
    large reverse correction.  X used only as the test-side input, never inside
    the production path."""
    cfg = FusionConfig()
    x = _speechish(4.0, seed=7)
    f = Fusion(cfg)
    with torch.no_grad():
        y = f.process_batch(x, x)            # S = X, V = X (aligned clean case)
    d = f.last_diagnostics
    p99 = d["correction_100_800"]["p90_abs_db"]  # p90 reported; compute p99 here:
    corr = torch.cat([h[0] for h in f.core.corr_history], dim=0)  # (Fb*N,)
    lo, hi = _band_bins(cfg, 100.0, 800.0)
    c = corr[lo:hi + 1].flatten()
    p99 = float(c.abs().quantile(0.99))
    min_c = float(c.min())
    assert p99 <= 1.0, f"clean-case correction too large: p99={p99:.2f} dB"
    assert min_c >= -1.0, f"large reverse correction: min={min_c:.2f} dB"
    assert torch.isfinite(y).all()
    print(f"  M4 PASS: clean case correction p99={p99:.3f} dB (≤1), "
          f"min={min_c:+.3f} dB (no large reverse)")


if __name__ == "__main__":
    test_M1_safety_fallback_strength0_and_forced_veto()
    test_M2_numeric_and_interface()
    test_M3_intervene_when_clearly_damaged()
    test_M4_no_harm_when_clean()
