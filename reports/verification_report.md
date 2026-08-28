# Verification Report (§5 + §6 tests)

| Test | Status | Time (s) | Error |
|------|--------|----------|-------|
| 5.1_complexity::test_complexity_all_arms | PASS | 0.1 |  |
| 5.2_causality::test_causality | PASS | 0.0 |  |
| 5.2_causality::test_shape_arbitrary | PASS | 0.2 |  |
| 5.3_streaming::test_streaming_batch_equiv | PASS | 2.1 |  |
| 5.4_overfit::test_overfit_single_batch | PASS | 76.5 |  |
| 5.5_gradient::test_gradient_flow | PASS | 0.1 |  |
| 5.5_gradient::test_mr_stft_gradient | PASS | 0.1 |  |
| 5.6_stft_roundtrip::test_cola | PASS | 0.0 |  |
| 5.6_stft_roundtrip::test_roundtrip | PASS | 0.0 |  |
| 5.7_loss_stability::test_mr_stft_stability | PASS | 0.0 |  |
| 5.7_loss_stability::test_spectral_loss_stability | PASS | 0.0 |  |
| 5.8_degradation::test_degradation_cutoff | PASS | 0.0 |  |
| 5.8_degradation::test_degradation_not_rectangular | PASS | 0.0 |  |
| 5.8_degradation::test_lowpass_adapter_runs | PASS | 0.8 |  |
| 5.8_degradation::test_time_varying | PASS | 0.0 |  |
| 5.9_export::test_export_all | PASS | 47.3 |  |
| 5.10_smoke::test_smoke_train | PASS | 24.4 |  |
| 6.1_ddsp_antialias::test_anti_aliasing | PASS | 0.0 |  |
| 6.1_ddsp_antialias::test_phase_precision | PASS | 0.0 |  |
| 6.5_f0::test_f0_half_double_freq | PASS | 0.0 |  |
| 6.5_f0::test_f0_synthetic | PASS | 0.0 |  |

**Summary: 21 passed, 0 failed**
