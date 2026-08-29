# GPU-Side TODO

Things that CANNOT be verified on CPU and MUST be checked when moving to GPU
(§8.8 deliverable).  Updated post-wrap-up; see `reports/l1_characterization.md`
for the CPU-side L1 findings that inform these.

## 1. AMP (mixed precision) actual behavior
- [ ] Verify `autocast` doesn't cause NaN in phase accumulation (fp64 internally
      per §6.2, but verify under real AMP)
- [ ] Confirm log/division ops in `SpectralLoss`/`stft_loss` are wrapped in
      `autocast(enabled=False)` (annotated, untested under real AMP)
- [ ] Verify GradScaler works with the multi-loss (spectral + MR-STFT + adv)

## 2. Large-batch stability
- [ ] Test batch_size = 16, 32, 64
- [ ] Gradient explosion at large batch (grad clipping max_norm=1.0 may need
      tuning)
- [ ] Verify multi-worker DataLoader doesn't deadlock

## 3. Throughput
- [ ] Report actual steps/s on the target GPU
- [ ] The CPU steps/s (train.py logs) is NOT extrapolable — only relative
      magnitude between arms

## 4. Complex-phase weight calibration (§spec-change)
- [ ] **Calibrate `cplx_weight` by param-side GRADIENT NORM, not by ear.** The
      complex MSE term (phase) is expected to behave differently from the
      magnitude terms; on real L1 data the phase above ~500 Hz is unobserved
      (see `l1_characterization.md` §1), so the complex term's early metrics
      will look worse than magnitude+oracle-phase — this is EXPECTED, do not
      retune to make numbers look good.  Use the ratio of gradient norms
      (magnitude-term grads vs complex-term grads) to set the weight.
- [ ] Do NOT retreat to magnitude-DOMAIN output to make the loss look better;
      the model stays complex-output (cplx_weight=1.0 baseline).

## 5. F0 estimator on real data (Arm A)
- [ ] `yin_f0` is verified on synthetic (§6.5) and measured on real L1
      (`l1_characterization.md` §2): **~15% octave errors**, intrinsic to
      F0-from-band-limited-sensor.  Simple continuity constraints (the
      existing `smooth_f0` MA, or a zero-preserving median) do NOT reduce them.
- [ ] **T11 F0-under-noise degradation** (`l1_characterization.md` T11-B): at
      5 dB SNR, WHITE blows octave >30%, WIND collapses voicing (17% agr).
      ⇒ re-run AFTER the §5 600 Hz lowpass (fewer harmonics ⇒ likely worse);
      the real aligned target is the harder case.
- [ ] **Try pYIN (probabilistic + Viterbi on the CMND function)** — the known
      better approach for octave jumps; not integrated (out of CPU-scope).
      Measure octave-error reduction vs the 15% clean / 31% white@5dB baselines.
- [ ] **Robust voicing detector** for the wind-collapse mode (yin's threshold
      is too conservative under low-freq wind).
- [ ] Run with `f0_mode: estimated` on full data; real sensor noise may differ.

## 5b. Noise-only-band robustness (T11 §4, trained model)
- [ ] `tests/test_noise_probe.py` is the SCAFFOLDING (forward with two >600 Hz
      noise seeds, measures output rel-diff).  UNTRAINED baseline is small
      (A 0.016 / B 0.001 / C 0.077) — no structural defect.  After training,
      re-assert rel_diff is SMALL (the network learned to ignore the noise
      band); a large trained-model diff ⇒ robustness defect (regression test).

## 6. ONNX export for production
- [ ] **Arm A ONNX export FAILS** — `torch.export` can't trace the dynamic YIN
      path.  Fix: remove YIN from the traced graph, feed F0 as a model input
      (the oscillator ops should export once control flow is gone).
- [ ] **Arm B ONNX export FAILS** — FX graph decomposition.  Needs a fix at
      the FX level.
- [ ] Arm C ONNX exports fine (1.7e-7 rel error); A/B export to TorchScript OK.
- [ ] Verify exported models match PyTorch output on GPU tensors.

## 7. Quality evaluation (NOT done on CPU — §5.10 explicitly forbids)
- [ ] Run full training on real data
- [ ] Evaluate PESQ / STOI / SI-SDR on held-out set
- [ ] **No quality comparison / arm ranking** until sufficient data volume (§9)

## 8. Streaming latency
- [ ] Measure end-to-end streaming latency (algorithmic + compute)
- [ ] Budget: hop=160 (10 ms algorithmic @16k) + compute ≤ ~40 ms total
- [ ] The `measure_streaming_complexity` peak memory (A 78 / B 136 / C 59 KB)
      is verified ≤300 KB on CPU; re-verify on GPU.

## 9. Training-recipe ablation
- [ ] Enable `multi_res_stft: true` and `discriminator: true` in config
- [ ] Run with/without each; these are verified to run + gradient flows (§5.5)
      but NOT trained.
