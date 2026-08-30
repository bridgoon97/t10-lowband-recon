"""T13 A5 — test-only idealised VPU V* and permutation-corrected ceiling.

BOUNDARY: 0624 only, four male speakers (F0 median 87–124 Hz), normal volume.
No 0625 speech is read.  V* and every use of X remain outside production.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fusion import FusionConfig, realdata
from fusion.degrade import DegradationConfig
from fusion.f0 import f0_batch
from fusion.fusion import FusionCore
from fusion.stft import istft_batch, stft_batch
from tests.test_t13_a4 import _oracle_w_scalar, _run
from tests.test_t13_b1 import BAND_EDGES_HZ, _band_bins, _need
from tests.test_t13_static import ALGO_FILES


DEPTHS = [15, 20, 30]
VSTAR_BANDS = [(100, 200), (200, 315), (315, 500), (500, 800)]
BETAS = [1.0, 0.5, 0.25, "harmonic"]
REPORT_DIR = Path("reports/T13A5")


@lru_cache(maxsize=1)
def _prepared():
    _need(); cfg = FusionConfig(); out = []
    for record_index, path in enumerate(realdata.list_0624()):
        name = os.path.basename(path)
        ff, vreal, _ = realdata.load_0624(name=name, seg_s=6.0, offset_s=1.0)
        spec_x = stft_batch(ff, cfg); spec_v = stft_batch(vreal, cfg)
        f0_v, conf_v = f0_batch(vreal, cfg); f0_x, conf_x = f0_batch(ff, cfg)
        out.append(dict(index=record_index, name=name, ff=ff, vreal=vreal,
                        spec_x=spec_x, spec_v=spec_v, f0_v=f0_v, conf_v=conf_v,
                        f0_x=f0_x, conf_x=conf_x))
    return cfg, out


def _within_state_permutation(conf, seed):
    """One shared time permutation, separately within voiced/unvoiced groups."""
    n = conf.numel(); mapping = torch.arange(n)
    generator = torch.Generator().manual_seed(int(seed))
    voiced = conf >= 0.55
    for state in (False, True):
        idx = torch.nonzero(voiced == state, as_tuple=False).flatten()
        if len(idx):
            mapping[idx] = idx[torch.randperm(len(idx), generator=generator)]
    return mapping


def _information_mask(prep, beta):
    """Deterministic closest-to-harmonic mask for beta information support."""
    cfg = FusionConfig(); sx = prep["spec_x"]
    mask = torch.zeros_like(sx.real, dtype=torch.bool)
    if beta == 1.0:
        lo, hi = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
        mask[:, lo:hi + 1] = True
        return mask
    bin_hz = cfg.sr / cfg.n_fft
    for t in range(sx.shape[-1]):
        f0 = float(prep["f0_v"][0, t])
        harmonic_hz = np.arange(max(1, int(np.ceil(cfg.eq_band_lo_hz / max(f0, 1e-6)))),
                                int(np.floor(cfg.eq_band_hi_hz / max(f0, 1e-6))) + 1) * f0
        for lo_hz, hi_hz in VSTAR_BANDS:
            lo, hi = _band_bins(cfg, lo_hz, hi_hz)
            bins = np.arange(lo, hi + 1)
            if len(harmonic_hz):
                distance = np.min(np.abs(bins[:, None] * bin_hz - harmonic_hz[None, :]), axis=1)
            else:
                distance = np.full(len(bins), np.inf)
            if beta == "harmonic":
                chosen = []
                for hz in harmonic_hz[(harmonic_hz >= lo_hz) & (harmonic_hz <= hi_hz)]:
                    chosen.append(int(round(hz / bin_hz)))
                chosen = np.unique(np.clip(chosen, lo, hi)).astype(int)
            else:
                count = max(1, int(np.ceil(float(beta) * len(bins))))
                chosen = bins[np.argsort(distance, kind="stable")[:count]]
            if len(chosen):
                mask[0, torch.as_tensor(chosen, dtype=torch.long), t] = True
    return mask


def build_vstar(prep, alpha, permutation_seed=None, beta=1.0,
                permute_true=False):
    """Construct V*: replace every 100–800 Hz bin, preserving V_real phase.

    ``alpha=0`` carries only the four-band mean log envelope; ``alpha=1``
    carries X's complete in-band log magnitude.  Only out-of-band bins remain
    V_real.  A single voiced/unvoiced-stratified time permutation is shared by
    all bands and changes only the alpha=0 envelope's time alignment.
    """
    cfg = FusionConfig(); sx = prep["spec_x"]; sv = prep["spec_v"]
    out = sv.clone(); n_frames = sx.shape[-1]
    mapping = (torch.arange(n_frames) if permutation_seed is None else
               _within_state_permutation(prep["conf_v"][0], permutation_seed))
    band_env = []
    for lo_hz, hi_hz in VSTAR_BANDS:
        lo, hi = _band_bins(cfg, lo_hz, hi_hz)
        band_env.append(20 * torch.log10(
            sx[0, lo:hi + 1].abs().clamp_min(1e-8)).mean(0))
    modified = torch.zeros_like(sx.real, dtype=torch.bool)
    target_log = torch.full_like(sx.real, float("nan"))
    source_band = torch.full_like(sx.real, -1, dtype=torch.int16)
    source_time = torch.full_like(sx.real, -1, dtype=torch.int32)
    info_mask = _information_mask(prep, beta)
    for band_index, (lo_hz, hi_hz) in enumerate(VSTAR_BANDS):
        lo, hi = _band_bins(cfg, lo_hz, hi_hz)
        bins = slice(lo, hi + 1)
        a_true_all = 20 * torch.log10(sx[0, bins].abs().clamp_min(1e-8))
        a_true = a_true_all[:, mapping] if permute_true else a_true_all
        a_band = band_env[band_index][mapping].unsqueeze(0).expand_as(a_true)
        a_star = float(alpha) * a_true + (1.0 - float(alpha)) * a_band
        selected = info_mask[0, bins]
        replacement = 10 ** (a_star / 20) * torch.exp(1j * torch.angle(sv[0, bins]))
        out[0, bins][selected] = replacement[selected]
        modified[0, bins] = selected
        target_log[0, bins][selected] = a_star[selected]
        source_band[0, bins][selected] = band_index
        mapped = mapping.to(source_time.dtype).unsqueeze(0).expand_as(source_time[0, bins])
        source_time[0, bins][selected] = mapped[selected]
    # V* is the Arm-A magnitude-spectrum interface.  Do not synthesize a
    # waveform: an arbitrary magnitude + V_real phase spectrum need not lie in
    # the STFT operator's range, so ISTFT->STFT would silently change A*.
    return out, dict(spec=out, modified=modified, mapping=mapping,
                     target_log=target_log, source_band=source_band,
                     source_time=source_time)


def _f0_agreement():
    _, records = _prepared(); both = agree = 0; voiced_same = total = 0
    errors = []
    for p in records:
        vv = p["conf_v"][0] >= 0.55; vx = p["conf_x"][0] >= 0.55
        voiced_same += int((vv == vx).sum()); total += vv.numel()
        mask = vv & vx; both += int(mask.sum())
        semitones = (12 * torch.log2(
            p["f0_v"][0, mask] / p["f0_x"][0, mask])).abs()
        agree += int((semitones <= 1.0).sum()); errors.extend(semitones.tolist())
    return dict(voiced_agreement=voiced_same / total,
                pitch_agreement=agree / max(1, both), both=both,
                median_semitones=float(np.median(errors)) if errors else float("nan"))


def _vstar_prod_hits(files):
    rx = re.compile(r"build_vstar|\bVStar\b|\bvstar\b|alpha_fidelity", re.I)
    hits = []
    for path in files:
        with open(path, errors="ignore") as handle:
            for lineno, line in enumerate(handle, 1):
                if rx.search(line):
                    hits.append(f"{path}:{lineno}:{line.strip()}")
    return hits


def test_A50_vstar_construction_and_static_guard():
    """A5R-0: full-band endpoint/phase/out-of-band and static checks."""
    cfg, records = _prepared(); p = records[0]
    v0, m0 = build_vstar(p, 0.0); v1, m1 = build_vstar(p, 1.0)
    mask = m0["modified"]
    unchanged_ok = torch.equal(m0["spec"][~mask], p["spec_v"][~mask])
    phase_ok = torch.allclose(torch.angle(m0["spec"][mask]),
                              torch.angle(p["spec_v"][mask]), atol=1e-7, rtol=0)
    true_mag_ok = torch.allclose(m1["spec"][mask].abs(),
                                 p["spec_x"][mask].abs(), atol=1e-7, rtol=1e-5)
    got0 = 20 * torch.log10(m0["spec"][mask].abs().clamp_min(1e-8))
    expected0 = []
    for _, b, t in torch.nonzero(mask, as_tuple=False).tolist():
        band_index = int(m0["source_band"][0, b, t])
        source_t = int(m0["source_time"][0, b, t])
        lo, hi = _band_bins(cfg, *VSTAR_BANDS[band_index])
        expected0.append(20 * torch.log10(
            p["spec_x"][0, lo:hi + 1, source_t].abs().clamp_min(1e-8)).mean())
    expected0 = torch.stack(expected0)
    band0_ok = torch.allclose(got0, expected0, atol=1e-5, rtol=1e-5)
    hits = _vstar_prod_hits(ALGO_FILES)
    f0 = _f0_agreement()
    print(f"  A5R-0 V*: modified={int(mask.sum())} phase={phase_ok} "
          f"outband unchanged={unchanged_ok} alpha0-band={band0_ok} "
          f"alpha1-true={true_mag_ok}")
    print(f"  A5-0 F0(V_real) vs F0(X): voiced-decision agreement="
          f"{f0['voiced_agreement']:.2%}; both-voiced n={f0['both']}; "
          f"within 1 semitone={f0['pitch_agreement']:.2%}; "
          f"median error={f0['median_semitones']:.3f} semitone")
    print(f"  A5-0 production V* references={len(hits)}")
    assert unchanged_ok and phase_ok and true_mag_ok and band0_ok
    assert torch.isfinite(v0).all() and torch.isfinite(v1).all()
    assert not hits, f"A5-0: V* leaked into production: {hits[:3]}"


def test_A50_vstar_static_mutation():
    """Mutation: production imports V*; static guard must catch it."""
    import tempfile
    mutant = "from tests.test_t13_a5 import build_vstar\ndef process(s,v): return build_vstar(s,v)\n"
    with tempfile.NamedTemporaryFile("w", suffix="fusion.py", delete=False) as handle:
        handle.write(mutant); path = handle.name
    try:
        hits = _vstar_prod_hits([path])
    finally:
        os.unlink(path)
    print(f"  A5-0 mutation: added production build_vstar import; hits={len(hits)}")
    assert hits, "A5-0 mutation: production V* leakage escaped static guard"


def _metric_parts(out, conf, cfg, band_indices=None, direct=False):
    if band_indices is None:
        band_indices = range(len(BAND_EDGES_HZ) - 1)
    y_key = "spec_y_direct" if direct else "spec_y"
    g3_s, g3_y = [], []; j_def = j_rec = 0.0
    for bi in band_indices:
        lo, hi = _band_bins(cfg, BAND_EDGES_HZ[bi], BAND_EDGES_HZ[bi + 1])
        xs = 20 * torch.log10(out["spec_x"][0, lo:hi + 1].abs().clamp_min(1e-8))
        ss = 20 * torch.log10(out["spec_s"][0, lo:hi + 1].abs().clamp_min(1e-8))
        ys = 20 * torch.log10(out[y_key][0, lo:hi + 1].abs().clamp_min(1e-8))
        px = out["spec_x"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        ps = out["spec_s"][0, lo:hi + 1].abs().pow(2).mean(0).clamp_min(1e-16)
        sup = 10 * torch.log10(px / ps)
        for t in range(out["spec_s"].shape[-1]):
            if float(sup[t]) > 6.0:
                g3_s.append(float(torch.sqrt(((ss[:, t] - xs[:, t]) ** 2).mean())))
                g3_y.append(float(torch.sqrt(((ys[:, t] - xs[:, t]) ** 2).mean())))
            if float(conf[0, t]) >= 0.55 and float((ss[:, t] - xs[:, t]).mean()) < -6:
                corr = float((ys[:, t] - ss[:, t]).abs().mean())
                deficit = float((xs[:, t] - ss[:, t]).abs().mean())
                j_def += deficit; j_rec += min(corr, deficit)
    return g3_s, g3_y, j_def, j_rec


def _run_vstar(ff, vstar_spec, cfg, deg, oracle=False, eq_mode="fit"):
    """Run V* through fitted EQ or the explicit C=0 comparison arm."""
    from tests._t13_eval import eval_specs
    spec_x, spec_s, s = eval_specs(ff, cfg, deg)
    spec_vp = vstar_spec
    frames = F.pad(s.float(), (cfg.win-cfg.hop, 0)).unsqueeze(1).unfold(
        -1, cfg.win, cfg.hop).squeeze(1)
    core = FusionCore(cfg); y_frames = []; oracle_ws = []; clipped = []
    d_values = []; loss_values = []
    for t in range(spec_s.shape[-1]):
        ss = spec_s[:, :, t]; vp = spec_vp[:, :, t]; buf = frames[:, t, :]
        f0, conf = core.f0est.estimate(buf)
        smag = ss.abs(); floor = core.nf.step(smag)
        snr = (20 * torch.log10(smag.clamp_min(1e-8) /
                                 floor.clamp_min(1e-8))).mean(-1)
        if eq_mode == "fit":
            raw_v = vp
            vp, startup, reset = core.eq.step(ss, raw_v, snr, conf)
            eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8))
                   - 20 * torch.log10(raw_v.abs().clamp_min(1e-8))
                   - core.eq.C).mean(-1) if core.eq.C is not None else torch.zeros_like(snr)
        elif eq_mode == "zero":
            startup = torch.zeros_like(snr); reset = torch.zeros_like(snr, dtype=torch.bool)
            eqr = (20 * torch.log10(ss.abs().clamp_min(1e-8))
                   - 20 * torch.log10(vp.abs().clamp_min(1e-8))).mean(-1)
        else:
            raise ValueError(f"unknown eq_mode={eq_mode}")
        cv = core.cv.step(vp, ss, eqr, bool(reset.any()))
        gf = core.gf0.step(conf); wb = core.wband.step(vp, ss)
        wl = core.wlocal.step(ss, vp, f0)
        product = cv.unsqueeze(-1) * gf.unsqueeze(-1) * wb * wl
        w = core.smooth.step(product)
        w_use = w
        if oracle:
            w_use = torch.zeros_like(w)
            sx = 20 * torch.log10(spec_x[:, :, t].abs().clamp_min(1e-8))
            sl = 20 * torch.log10(ss.abs().clamp_min(1e-8))
            vl = 20 * torch.log10(vp.abs().clamp_min(1e-8))
            for bi in range(len(BAND_EDGES_HZ)-1):
                lo, hi = _band_bins(cfg, BAND_EDGES_HZ[bi], BAND_EDGES_HZ[bi+1])
                value = _oracle_w_scalar(sl[0, lo:hi+1], vl[0, lo:hi+1],
                                         sx[0, lo:hi+1], cfg.delta_down_db,
                                         cfg.delta_up_db)
                w_use[:, lo:hi+1] = value
            # A5R-1's alpha=1 construction promise is only 100--800 Hz;
            # diagnostics must not be diluted by untouched out-of-band bins.
            diag_lo, diag_hi = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
            correction = w_use[:, diag_lo:diag_hi + 1] * (
                vl[:, diag_lo:diag_hi + 1] - sl[:, diag_lo:diag_hi + 1])
            oracle_ws.append(w_use[:, diag_lo:diag_hi + 1].detach().cpu())
            clipped.append(((correction <= -cfg.delta_down_db) |
                            (correction >= cfg.delta_up_db)).detach().cpu())
            d_values.append((vl[:, diag_lo:diag_hi + 1] -
                             sl[:, diag_lo:diag_hi + 1]).detach().cpu().flatten())
            loss_values.append((sx[:, diag_lo:diag_hi + 1] -
                                sl[:, diag_lo:diag_hi + 1]).detach().cpu().flatten())
        y_frames.append(core.synth.step(ss, vp, w_use))
    spec_y_direct = torch.stack(y_frames, -1)
    y = istft_batch(spec_y_direct, cfg, length=s.shape[-1])
    result = dict(spec_x=spec_x, spec_s=spec_s,
                  spec_y_direct=spec_y_direct, spec_y=stft_batch(y, cfg), s=s)
    if oracle:
        result["oracle_w"] = torch.stack(oracle_ws, -1)
        result["clip_fraction"] = float(torch.stack(clipped, -1).float().mean())
        result["d"] = torch.cat(d_values)
        result["loss"] = torch.cat(loss_values)
    return result


@lru_cache(maxsize=None)
def _alpha1_gate_row(depth, eq_mode):
    """Pooled A5R-1 metric plus the three predeclared path diagnostics."""
    _, records = _prepared(); gs = []; gy = []; jd = jr = 0.0
    ws = []; clip_weighted = 0.0; n_values = 0; ds = []; losses = []
    for prep, vstar in zip(records, _vstars(1.0)):
        cfg = FusionConfig(); deg = DegradationConfig(
            d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        out = _run_vstar(prep["ff"], vstar, cfg, deg, oracle=True, eq_mode=eq_mode)
        _, conf = f0_batch(prep["ff"], cfg)
        # The hard gate is the Layer-3 identity on V*'s four written bands.
        s, y, d0, r0 = _metric_parts(out, conf, cfg, band_indices=range(4), direct=True)
        gs.extend(s); gy.extend(y); jd += d0; jr += r0
        ws.append(out["oracle_w"].flatten())
        count = out["oracle_w"].numel(); n_values += count
        clip_weighted += out["clip_fraction"] * count
        ds.append(out["d"]); losses.append(out["loss"])
    w = torch.cat(ws).numpy(); d = torch.cat(ds).numpy(); loss = torch.cat(losses).numpy()
    ratio = float(np.mean(gy) / np.mean(gs)); corr = float(np.corrcoef(d, loss)[0, 1])
    # Six-band figures are report-only: V* does not write 800--2000 Hz.
    full_gs = []; full_gy = []; full_jd = full_jr = 0.0; outband_corr = []
    for prep, vstar in zip(records, _vstars(1.0)):
        cfg = FusionConfig(); deg = DegradationConfig(
            d1_kill_rate=0.4, d1_kill_depth_db=float(depth))
        out = _run_vstar(prep["ff"], vstar, cfg, deg, oracle=True, eq_mode=eq_mode)
        _, conf = f0_batch(prep["ff"], cfg)
        s, y, d0, r0 = _metric_parts(out, conf, cfg, direct=True)
        full_gs.extend(s); full_gy.extend(y); full_jd += d0; full_jr += r0
        for bi in (4, 5):
            lo, hi = _band_bins(cfg, BAND_EDGES_HZ[bi], BAND_EDGES_HZ[bi + 1])
            delta = (20 * torch.log10(out["spec_y_direct"][0, lo:hi + 1].abs().clamp_min(1e-8)) -
                     20 * torch.log10(out["spec_s"][0, lo:hi + 1].abs().clamp_min(1e-8))).abs()
            outband_corr.extend(delta.flatten().tolist())
    full_ratio = float(np.mean(full_gy) / np.mean(full_gs))
    return dict(n_sup=len(gs), ratio=ratio, j3=jr/max(1.0, jd),
                w_p50=float(np.median(w)), w_p90=float(np.percentile(w, 90)),
                w_max=float(np.max(w)), clip_fraction=clip_weighted/n_values,
                d_loss_corr=corr, full_ratio=full_ratio,
                full_j3=full_jr/max(1.0, full_jd),
                outband_corr_p99=float(np.percentile(outband_corr, 99)),
                outband_corr_max=float(np.max(outband_corr)))


def _alpha1_pipeline_stats():
    """A5R-1a: direct C=0 d must equal the true loss in 100--800 Hz."""
    cfg, records = _prepared(); ds = []; losses = []
    lo, hi = _band_bins(cfg, cfg.eq_band_lo_hz, cfg.eq_band_hi_hz)
    for prep, vstar in zip(records, _vstars(1.0)):
        for depth in DEPTHS:
            from tests._t13_eval import eval_specs
            sx, ss, _ = eval_specs(prep["ff"], cfg, DegradationConfig(
                d1_kill_rate=0.4, d1_kill_depth_db=float(depth)))
            ds.append((20 * torch.log10(vstar[:, lo:hi + 1].abs().clamp_min(1e-8)) -
                       20 * torch.log10(ss[:, lo:hi + 1].abs().clamp_min(1e-8))).flatten())
            losses.append((20 * torch.log10(sx[:, lo:hi + 1].abs().clamp_min(1e-8)) -
                           20 * torch.log10(ss[:, lo:hi + 1].abs().clamp_min(1e-8))).flatten())
    d = torch.cat(ds).double(); loss = torch.cat(losses).double()
    return dict(corr=float(np.corrcoef(d.numpy(), loss.numpy())[0, 1]),
                max_abs=float((d-loss).abs().max()), d=d, loss=loss)


def test_A5R1a_alpha1_pipeline_identity():
    """A5R-1a: direct spectral injection must preserve alpha=1 exactly."""
    r = _alpha1_pipeline_stats()
    print(f"  A5R-1a alpha=1 C=0 100-800Hz: corr(d,loss)={r['corr']:.12f}; "
          f"max|d-loss|={r['max_abs']:.3e} dB")
    assert abs(r["corr"] - 1.0) <= 1e-12, r
    assert r["max_abs"] <= 1e-5, r


def test_A5R1a_pipeline_identity_mutation():
    """Mutation: one in-band +1 dB injection error must break A5R-1a."""
    cfg, records = _prepared(); prep = records[0]
    vstar, _ = build_vstar(prep, 1.0); lo, _ = _band_bins(cfg, 100, 200)
    mutant = vstar.clone(); mutant[0, lo, 10] *= 10 ** (1.0 / 20)
    from tests._t13_eval import eval_specs
    sx, ss, _ = eval_specs(prep["ff"], cfg, DegradationConfig(
        d1_kill_rate=0.4, d1_kill_depth_db=20.0))
    d = 20 * torch.log10(mutant.abs().clamp_min(1e-8)) - 20 * torch.log10(ss.abs().clamp_min(1e-8))
    loss = 20 * torch.log10(sx.abs().clamp_min(1e-8)) - 20 * torch.log10(ss.abs().clamp_min(1e-8))
    error = float((d[0, lo, 10] - loss[0, lo, 10]).abs())
    print(f"  A5R-1a mutation: +1 dB at bin={lo},frame=10; max error={error:.6f} dB")
    assert error > 0.9, "A5R-1a mutation escaped the pipeline identity criterion"


def test_A5R1b_alpha1_hard_gate():
    """A5R-1: alpha=1 must expose X through the C=0 layer-3 input."""
    rows = {}
    print("  A5R-1 alpha=1 pooled hard-gate and EQ comparison:")
    print("  EQ depth n_sup ratio4 J3_4 ratio6 J3_6 w(p50/p90/max) "
          "clip_fraction corr(d,loss) outband_corr(p99/max)")
    for mode in ("zero", "fit"):
        for depth in DEPTHS:
            r = _alpha1_gate_row(depth, mode); rows[(mode, depth)] = r
            print(f"  {mode:>4} {depth:>5} {r['n_sup']:>5} {r['ratio']:.5f} "
                  f"{r['j3']:.5f} {r['full_ratio']:.5f} {r['full_j3']:.5f} "
                  f"{r['w_p50']:.5f}/{r['w_p90']:.5f}/{r['w_max']:.5f} "
                  f"{r['clip_fraction']:.5%} {r['d_loss_corr']:.6f} "
                  f"{r['outband_corr_p99']:.5f}/{r['outband_corr_max']:.5f}")
    failed = [(depth, rows[("zero", depth)]) for depth in DEPTHS
              if not (rows[("zero", depth)]["j3"] >= 0.90 and
                      rows[("zero", depth)]["ratio"] <= 0.10)]
    assert not failed, f"A5R-1 C=0 alpha=1 hard gate failed: {failed}"


def test_A5R1b_hard_gate_mutation():
    """Mutation: replacing alpha=1 V* by V_real must fail the d20 gate."""
    _, records = _prepared(); gs = []; gy = []; jd = jr = 0.0
    for prep in records:
        cfg = FusionConfig(); out = _run_vstar(
            prep["ff"], prep["spec_v"], cfg, DegradationConfig(
                d1_kill_rate=0.4, d1_kill_depth_db=20.0),
            oracle=True, eq_mode="zero")
        _, conf = f0_batch(prep["ff"], cfg)
        s, y, d0, r0 = _metric_parts(
            out, conf, cfg, band_indices=range(4), direct=True)
        gs.extend(s); gy.extend(y); jd += d0; jr += r0
    ratio = float(np.mean(gy)/np.mean(gs)); j3 = jr/max(1.0, jd)
    gate_passed = j3 >= 0.90 and ratio <= 0.10
    print(f"  A5R-1 mutation: alpha=1 V* -> V_real; ratio={ratio:.5f}, "
          f"J3={j3:.5f}, gate_passed={gate_passed}")
    assert not gate_passed, "A5R-1 mutation escaped the alpha=1 hard gate"


def _permute_vreal_envelope(prep, seed):
    """Permute V_real's four-band envelope within one recording/state.

    One mapping is shared across bands.  Within-band spectral residuals, phase,
    and all out-of-band bins stay at their original time, so only envelope/X
    temporal alignment is destroyed.
    """
    cfg = FusionConfig(); sv = prep["spec_v"]; out = sv.clone()
    mapping = _within_state_permutation(prep["conf_v"][0], seed)
    for band in VSTAR_BANDS:
        lo, hi = _band_bins(cfg, *band)
        logmag = 20 * torch.log10(sv[0, lo:hi + 1].abs().clamp_min(1e-8))
        env = logmag.mean(0)
        permuted = logmag - env.unsqueeze(0) + env[mapping].unsqueeze(0)
        out[0, lo:hi + 1] = 10 ** (permuted / 20) * torch.exp(
            1j * torch.angle(sv[0, lo:hi + 1]))
    return out


def _oracle_metric_for_spec(prep, spec_vp, depth, eq_mode="fit", direct=False,
                            band_indices=None):
    cfg = FusionConfig(); out = _run_vstar(
        prep["ff"], spec_vp, cfg, DegradationConfig(
            d1_kill_rate=0.4, d1_kill_depth_db=float(depth)),
        oracle=True, eq_mode=eq_mode)
    _, conf = f0_batch(prep["ff"], cfg)
    s, y, jd, jr = _metric_parts(out, conf, cfg, band_indices=band_indices,
                                  direct=direct)
    ratio = float(np.mean(y) / np.mean(s))
    return dict(n_sup=len(s), ratio=ratio, recovery=1-ratio,
                j3=jr/max(1.0, jd))


@lru_cache(maxsize=1)
def _measure_vreal_recordwise_null():
    """A5R-2: recordwise statistic and recordwise permutation, then summarize."""
    _, records = _prepared(); result = {}
    for depth, b_count in ((15, 9), (20, 39), (30, 9)):
        per_record = []
        start = time.perf_counter()
        for prep in records:
            observed = _oracle_metric_for_spec(prep, prep["spec_v"], depth)
            null = [_oracle_metric_for_spec(
                prep, _permute_vreal_envelope(
                    prep, 73000 + 1000*b + prep["index"]), depth)
                    for b in range(b_count)]
            jmed = float(np.median([r["j3"] for r in null]))
            gmed = float(np.median([r["recovery"] for r in null]))
            per_record.append(dict(name=prep["name"], observed=observed,
                                   null_j3_median=jmed, null_recovery_median=gmed,
                                   delta_j3=observed["j3"]-jmed,
                                   delta_recovery=observed["recovery"]-gmed))
        result[depth] = dict(b=b_count, rows=per_record,
                             elapsed=time.perf_counter()-start)
    return result


def test_A5R2_vreal_recordwise_permutation():
    """A5R-2: matched-granularity V_real null resolves pooled/single mismatch."""
    result = _measure_vreal_recordwise_null()
    print("  A5R-2 V_real recordwise statistic + within-record permutation:")
    print("  depth B metric observed[min/med/max] nullmed[min/med/max] "
          "delta[min/med/max/std] seconds")
    for depth in DEPTHS:
        block = result[depth]
        for metric, null_key, delta_key in (
                ("J3", "null_j3_median", "delta_j3"),
                ("G3rec", "null_recovery_median", "delta_recovery")):
            obs = np.array([r["observed"]["j3" if metric == "J3" else "recovery"]
                            for r in block["rows"]])
            null = np.array([r[null_key] for r in block["rows"]])
            delta = np.array([r[delta_key] for r in block["rows"]])
            print(f"  d{depth} B{block['b']} {metric:>5} "
                  f"{obs.min():.5f}/{np.median(obs):.5f}/{obs.max():.5f} "
                  f"{null.min():.5f}/{np.median(null):.5f}/{null.max():.5f} "
                  f"{delta.min():+.5f}/{np.median(delta):+.5f}/{delta.max():+.5f}/"
                  f"{delta.std():.5f} {block['elapsed']:.1f}")
        first = block["rows"][0]
        print(f"    default {first['name']}: deltaJ3={first['delta_j3']:+.5f}; "
              f"deltaG3rec={first['delta_recovery']:+.5f}")
    assert result[20]["b"] == 39 and result[15]["b"] == result[30]["b"] == 9
    assert all(len(result[d]["rows"]) == 10 for d in DEPTHS)


@lru_cache(maxsize=1)
def _measure_beta_scan():
    """A5R-3: alpha=1 information support, with matched B=9 null floors."""
    _, records = _prepared(); result = {}
    for beta in BETAS:
        for depth in DEPTHS:
            rows = []; start = time.perf_counter()
            for prep in records:
                observed_spec, meta = build_vstar(prep, 1.0, beta=beta)
                observed = _oracle_metric_for_spec(
                    prep, observed_spec, depth, eq_mode="zero", direct=True,
                    band_indices=range(4))
                null = []
                for b in range(9):
                    null_spec, _ = build_vstar(
                        prep, 1.0,
                        permutation_seed=91000 + 1000*b + prep["index"],
                        beta=beta, permute_true=True)
                    null.append(_oracle_metric_for_spec(
                        prep, null_spec, depth, eq_mode="zero", direct=True,
                        band_indices=range(4)))
                jmed = float(np.median([r["j3"] for r in null]))
                gmed = float(np.median([r["recovery"] for r in null]))
                rows.append(dict(name=prep["name"], n_info=int(meta["modified"].sum()),
                                 observed=observed, null_j3_median=jmed,
                                 null_recovery_median=gmed,
                                 delta_j3=observed["j3"]-jmed,
                                 delta_recovery=observed["recovery"]-gmed))
            result[(str(beta), depth)] = dict(rows=rows,
                                                  elapsed=time.perf_counter()-start)
    return result


def _plot_beta_scan(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520; x0, y0, pw, ph = 90, 50, 730, 370
    labels = ["1.0", "0.5", "0.25", "harmonic"]
    colors = {15: "#4472C4", 20: "#70AD47", 30: "#ED7D31"}
    x = lambda i: x0 + i * pw / (len(labels)-1)
    y = lambda value: y0 + ph - max(-0.05, min(1.0, value)) / 1.05 * ph
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="450" y="25" text-anchor="middle" font-size="20">A5R-3 oracle J3 after matched null correction</text>',
           f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#666"/>']
    for depth in DEPTHS:
        vals = []
        for label in labels:
            rows = result[(label, depth)]["rows"]
            vals.append(float(np.median([r["delta_j3"] for r in rows])))
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[depth]}" stroke-width="3"/>')
        for i, value in enumerate(vals):
            svg.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" fill="{colors[depth]}"/>')
        svg.append(f'<text x="{x0+15}" y="{450+20*DEPTHS.index(depth)}" fill="{colors[depth]}" font-size="14">depth {depth}</text>')
    for i, label in enumerate(labels):
        svg.append(f'<text x="{x(i):.1f}" y="{y0+ph+22}" text-anchor="middle" font-size="13">{label}</text>')
    svg.extend(['<text x="450" y="510" text-anchor="middle" font-size="14">beta (X-informed in-band support)</text>', '</svg>'])
    (REPORT_DIR / "beta_oracle_j3.svg").write_text("\n".join(svg), encoding="utf-8")


def test_A5R3_beta_information_support():
    """A5R-3: quantify the band-scalar penalty as X information gets sparse."""
    result = _measure_beta_scan(); _plot_beta_scan(result)
    print("  A5R-3 alpha=1 C=0 four-band oracle; per-record B=9 null correction:")
    print("  beta depth n_info_med ratio_med J3_med nullJ3_med deltaJ3[min/med/max] "
          "deltaG3rec[min/med/max] seconds")
    for beta in BETAS:
        label = str(beta)
        for depth in DEPTHS:
            block = result[(label, depth)]; rows = block["rows"]
            ratio = np.array([r["observed"]["ratio"] for r in rows])
            j3 = np.array([r["observed"]["j3"] for r in rows])
            nj = np.array([r["null_j3_median"] for r in rows])
            dj = np.array([r["delta_j3"] for r in rows])
            dg = np.array([r["delta_recovery"] for r in rows])
            ni = np.array([r["n_info"] for r in rows])
            print(f"  {label:>8} {depth:>5} {np.median(ni):>10.0f} "
                  f"{np.median(ratio):.5f} {np.median(j3):.5f} {np.median(nj):.5f} "
                  f"{dj.min():+.5f}/{np.median(dj):+.5f}/{dj.max():+.5f} "
                  f"{dg.min():+.5f}/{np.median(dg):+.5f}/{dg.max():+.5f} "
                  f"{block['elapsed']:.1f}")
    print(f"  plot -> {REPORT_DIR / 'beta_oracle_j3.svg'}")
    assert all(len(result[(str(beta), depth)]["rows"]) == 10
               for beta in BETAS for depth in DEPTHS)


@lru_cache(maxsize=None)
def _vstars(alpha, permutation_index=-1):
    _, records = _prepared(); signals = []
    for prep in records:
        seed = None if permutation_index < 0 else 51000 + 1000 * permutation_index + prep["index"]
        signals.append(build_vstar(prep, float(alpha), seed)[0])
    return signals


def test_A5_MR1_noop_metrics():
    """MR1: Y:=S receives exactly zero G3 recovery and J3."""
    cfg, records = _prepared(); g_recovery = []; j3 = []
    for p in records:
        deg = DegradationConfig(d1_kill_rate=0.4, d1_kill_depth_db=20.0)
        out = _run(p["ff"], p["vreal"], cfg, deg, oracle=False)
        out["spec_y"] = out["spec_s"]
        _, conf = f0_batch(p["ff"], cfg)
        gs, gy, jd, jr = _metric_parts(out, conf, cfg)
        g_recovery.append(1 - np.mean(gy) / np.mean(gs)); j3.append(jr/max(1, jd))
    print(f"  A5 MR1 no-op: max|G3 recovery|={max(map(abs,g_recovery)):.3e}; "
          f"max|J3|={max(map(abs,j3)):.3e}")
    assert max(map(abs, g_recovery)) <= 1e-12
    assert max(map(abs, j3)) == 0.0
