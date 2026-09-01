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

from fusion import FusionConfig, Fusion, FusionCore, FusionStreamer, stft_batch
from fusion.f0 import f0_batch

SR = 16000
WIN = 480


# ----------------------------------------------------------------- helpers --
def _speechish(T_s: float, f0: float = 110.0, seed: int = 3) -> torch.Tensor:
    """Harmonic-rich voiced signal (male-F0 range) with slow envelope, 1/k tilt,
    plus the SAME-envelope broadband component (fricative/breath-like).  The
    broadband part is needed by the EQ dual-credible SNR gate: it averages
    20log10(|S|/floor) over ALL bins, so a strictly band-limited signal never
    becomes "credible" and the EQ never freezes (startup floor forever).  Real
    recordings are full-band; the envelope makes the min-trace floor sit at the
    dip level ⇒ active frames get SNR ≈ 14 dB > 6 dB gate.  Test-side only."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(T_s * SR)) / SR
    env = 0.55 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t + 0.3))
    x = np.zeros_like(t)
    for k in range(1, 25):
        ph = rng.uniform(0, 2 * np.pi)
        x += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + ph)
    x = x / (np.abs(x).max() + 1e-9) * 0.4 * env
    x += rng.normal(0, 1.0, t.shape) * env * 0.02      # enveloped broadband
    x += rng.normal(0, 1e-3, t.shape)                  # device noise floor
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
    """M1: strength=0 (real CLI end-to-end, FULL length) and a forced veto both
    give Y ≡ S (torch.allclose rtol=1e-5, atol=1e-4) — through the REAL
    production path (Fusion/CLI), not via aux/w.  Plus a STATE-TRANSITION
    scenario: w established > 0, then a safety veto fires mid-run ⇒ the FIRST
    veto frame must have final w exactly 0 and output spec == S spec (the
    post-smoothing hard mask; the smoother's fall tau must not leak)."""
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
    # FULL length (no tail fade at strength=0): vs quantized S.  The FINAL
    # sample is structurally unencodable by the causal framing (Hann window
    # endpoint w=0, no subsequent frame) — a PRE-EXISTING property of the
    # shared causal STFT/iSTFT pair, identical in legacy mode; it is reported
    # separately.  All other samples must match to PCM_16 quantization (~3e-5).
    s_q = torch.clamp(s_d[0], -1, 1)
    s_q = torch.round(s_q * 32767) / 32767
    diff = (y0 - s_q.unsqueeze(0)).abs()
    body_max = float(diff[..., :-1].max())
    last = float(diff[..., -1])
    assert body_max <= 1e-4, f"strength=0 output != S (body max {body_max:.2e})"
    print(f"  M1(a) CLI strength=0: full-length body max diff = {body_max:.2e} "
          f"(PCM_16 quantization level); final-sample diff {last:.2e} "
          f"(structural: causal frame endpoint w=0, never encoded)")
    # (b) forced safety veto through Fusion: MSC veto threshold unreachable
    cfg_v = FusionConfig().with_switches(mvp_veto_msc=1.1)   # MSC ≤ 1 < 1.1 ⇒ always veto
    fv = Fusion(cfg_v)
    with torch.no_grad():
        y_v = fv.process_batch(s_d, x)
    assert torch.allclose(y_v[..., skip:N - skip], s_d[..., skip:N - skip],
                          rtol=1e-5, atol=1e-4), "forced veto output != S"
    assert float(fv.last_diagnostics["coverage_100_800"]) == 0.0
    # (c) STATE TRANSITION via the real FusionCore path: establish w > 0 on a
    # damaged-S + healthy-V run, then S becomes LEVEL-MATCHED noise (mic
    # loses speech evidence) ⇒ f0 conf drops < 0.5 ⇒ FRAME-level veto fires
    # mid-run while the smoother still holds a large residual (fall tau 15 ms
    # ≈ 0.46/frame) ⇒ the FIRST veto frame must have final w == 0 exactly and
    # output spec == S spec (post-smoothing hard mask, not just aux).
    cfg_t = FusionConfig()
    x2 = _speechish(5.0, seed=5)
    s2, _ = _damage_inband(x2, from_s=2.0, depth_db=20.0)
    tail = slice(int(3.0 * SR), None)
    rms = float(s2[0, tail].pow(2).mean()).__pow__(0.5)
    rng = np.random.default_rng(9)
    s2[0, tail] = torch.from_numpy(
        rng.normal(0, 1.0, (1, int(5.0 * SR) - int(3.0 * SR))).astype(np.float32)) * rms / (2 ** 0.5)
    v2 = x2                                            # V healthy throughout
    core = FusionCore(cfg_t)
    spec_s2 = stft_batch(s2, cfg_t); spec_v2 = stft_batch(v2, cfg_t)
    left_pad = cfg_t.win - cfg_t.hop
    sp = torch.nn.functional.pad(s2, (left_pad, 0), mode="constant")
    frames_s = sp.unsqueeze(1).unfold(-1, cfg_t.win, cfg_t.hop).squeeze(1)
    lo, hi = _band_bins(cfg_t, 100.0, 800.0)
    w_frames, y_frames, veto_frames = [], [], []
    with torch.no_grad():
        for t in range(spec_s2.shape[-1]):
            y_t, w_t = core.process_frame(spec_s2[:, :, t], spec_v2[:, :, t],
                                          frames_s[:, t, :])
            w_frames.append(w_t[0, lo:hi + 1].clone())
            y_frames.append(y_t.clone())
            veto_frames.append(core.veto_history[-1][0, lo:hi + 1].clone())
    w_mat = torch.stack(w_frames, dim=-1)          # (in-band bins, N)
    y_spec = torch.stack(y_frames, dim=-1)         # full spec per frame
    veto_mat = torch.stack(veto_frames, dim=-1)
    any_veto = veto_mat.any(dim=0)                 # (N,) in-band (final hard mask)
    w_peak = w_mat.max(dim=0).values               # (N,)
    # locate the transition: w established (>0.3) AFTER the EQ startup floor
    # clears, then the FIRST veto frame after establishment
    est = int(torch.nonzero(w_peak > 0.3)[0])
    assert est > 0, "w never established"
    later = torch.nonzero(any_veto[est:])
    assert later.numel() > 0, "no veto fired after w was established"
    first = est + int(later[0])
    assert not bool(any_veto[first - 1]), "frame before first veto was already vetoed"
    pre_w = float(w_peak[first - 1])
    assert pre_w > 0.3, f"w not established right before veto (pre_w={pre_w:.3f})"
    assert float(w_mat[:, first].abs().max()) == 0.0, \
        f"first veto frame w not exactly 0 (max {float(w_mat[:, first].abs().max()):.4f})"
    # production output spec == S spec on the first veto frame (real Synthesis)
    s_spec2 = spec_s2[0, :, first]
    dmax = float((y_spec[0, :, first] - s_spec2).abs().max())
    assert torch.allclose(y_spec[0, :, first], s_spec2, rtol=1e-5, atol=1e-4), \
        f"first-veto-frame output spec != S spec (max diff {dmax:.2e})"
    print(f"  M1(b) forced veto ≡ S (interior); M1(c) state transition: "
          f"pre-veto w_max={pre_w:.2f} → first veto frame w=0 exactly, "
          f"output spec == S spec (diff {dmax:.1e})")
    print("  M1 PASS: strength=0 (CLI, full length) / forced veto / "
          "state transition all safe")


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
    # safety conditions actually healthy in the EVALUATED region (the whole-run
    # veto fraction also contains the EQ startup floor, by design — the rework
    # includes startup/reset frames in the final hard mask)
    lo, hi = _band_bins(cfg, 100.0, 800.0)
    lo_e = int(3.0 * SR // cfg.hop)
    v_evt = torch.stack([h[0, lo:hi + 1] for h in f.core.veto_history], dim=-1)
    region_veto = float(v_evt[:, lo_e:].float().mean())
    assert region_veto < 0.2, f"vetoes fired in evaluated region: {region_veto:.3f}"
    print(f"  M3 PASS: band LSD {lsd_s:.2f} -> {lsd_y:.2f} dB (strict improvement); "
          f"coverage={d['coverage_100_800']:.3f}; region veto_frac={region_veto:.3f} "
          f"(whole-run incl. EQ startup: {d['veto_fraction_100_800']:.3f})")
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
    # full-time indexing: (Fb, N) then in-band slice (the old cat-then-slice
    # only checked a few bins of frame 0)
    corr = torch.stack(f.core.corr_history, dim=-1)[0]        # (Fb, N)
    lo, hi = _band_bins(cfg, 100.0, 800.0)
    c = corr[lo:hi + 1].flatten()
    p99 = float(c.abs().quantile(0.99))
    min_c = float(c.min())
    max_c = float(c.abs().max())
    assert p99 <= 1.0, f"clean-case correction too large: p99={p99:.2f} dB"
    assert min_c >= -1.0, f"large reverse correction: min={min_c:.2f} dB"
    assert torch.isfinite(y).all()
    print(f"  M4 PASS: clean case correction p99={p99:.2e} dB (≤1), "
          f"min={min_c:+.2e} max={max_c:.2e} dB (no large reverse)")


if __name__ == "__main__":
    test_M1_safety_fallback_strength0_and_forced_veto()
    test_M2_numeric_and_interface()
    test_M3_intervene_when_clearly_damaged()
    test_M4_no_harm_when_clean()
