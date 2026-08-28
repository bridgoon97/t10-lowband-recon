# GPU-Side TODO

Things that CANNOT be verified on CPU and MUST be checked when moving to GPU
(§8.8 deliverable).

## 1. AMP (mixed precision) actual behavior
- [ ] Verify `autocast` doesn't cause NaN in phase accumulation (should be
  fp64 internally, but verify)
- [ ] Check if log/division ops need explicit `autocast(enabled=False)` —
  they're annotated but not tested under actual AMP
- [ ] Verify GradScaler works with the multi-loss (spectral + MR-STFT + adv)

## 2. Large-batch stability
- [ ] Test batch_size = 16, 32, 64
- [ ] Check gradient explosion at large batch (grad clipping at max_norm=1.0
  may need tuning)
- [ ] Verify multi-worker DataLoader doesn't deadlock

## 3. Throughput
- [ ] Report actual steps/s on target GPU
- [ ] The CPU steps/s (reported in train.py logs) is NOT extrapolable — only
  use it to perceive relative magnitude between arms

## 4. Full-dataset training
- [ ] Download Vibravox subset (§4.1: hundreds of clips, NOT full 45h)
- [ ] Use bone-conduction channel (`bone_chin` / `bone_throat`), NOT in-ear mic
- [ ] Add Chinese speech corpus for L0 (F0 dynamic range pressure on Arm A)
- [ ] Train with `f0_mode: estimated` — verify YIN works on real body-conduction
  signals (it's verified on synthetic, §6.5, but real sensor noise may differ)

## 5. ONNX export for production
- [ ] Fix Arm A ONNX export (remove YIN from traced graph, feed F0 as input)
- [ ] Fix Arm B ONNX export (FX graph decomposition issue)
- [ ] Verify exported models match PyTorch output on GPU tensors

## 6. Quality evaluation (NOT done on CPU — §5.10 explicitly forbids)
- [ ] Run full training on real data
- [ ] Evaluate PESQ / STOI / SI-SDR on held-out set
- [ ] Compare arms — ONLY after sufficient data volume (§9: no quality
  comparison on CPU)
- [ ] F0 error analysis (octave/half-frequency errors on real data)

## 7. Streaming latency
- [ ] Measure end-to-end streaming latency (algorithmic + compute)
- [ ] Verify ≤40 ms budget with real hop=32 (8 ms algorithmic + compute)

## 8. Temporal shift / discriminator ablation
- [ ] Enable `multi_res_stft: true` and `discriminator: true` in config
- [ ] Run ablation: with/without each training recipe module
- [ ] These are verified to run + gradient flows (§5.5) but NOT trained
