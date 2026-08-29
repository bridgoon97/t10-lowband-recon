# Verification Report (layered: L0 ideal-lowpass / L1 real body-conduction)

Tests are split by DATA DOMAIN, not by feature.  L0 = synthetic
clean speech → ideal lowpass (``lowpass_sim``).  L1 = real Vibravox
body-conduction (forehead accelerometer) ↔ headset air reference.
L1 tests SKIP (not fail) if the local parquet shard is absent —
see ``reports/vibravox_schema_diff.md`` for how to fetch it.

## L0 — ideal lowpass (lowpass_sim)

| Test | Status | Time (s) | Error |
|------|--------|----------|-------|
| L0::5.1_complexity::test_complexity_all_arms | PASS | 0.1 |  |
| L0::5.2_causality::test_causality | PASS | 0.0 |  |
| L0::5.2_causality::test_shape_arbitrary | PASS | 0.2 |  |
| L0::5.3_streaming::test_streaming_batch_equiv | PASS | 1.5 |  |
| L0::5.4_overfit::test_overfit_single_batch | PASS | 15.2 |  |
| L0::5.5_gradient::test_gradient_flow | PASS | 0.1 |  |
| L0::5.5_gradient::test_mr_stft_gradient | PASS | 0.1 |  |
| L0::5.6_stft_roundtrip::test_causal_roundtrip | PASS | 0.0 |  |
| L0::5.6_stft_roundtrip::test_cola | PASS | 0.0 |  |
| L0::5.6_stft_roundtrip::test_roundtrip | PASS | 0.0 |  |
| L0::5.7_loss_stability::test_mr_stft_stability | PASS | 0.0 |  |
| L0::5.7_loss_stability::test_spectral_loss_stability | PASS | 0.0 |  |
| L0::5.8_degradation::test_degradation_cutoff | PASS | 0.0 |  |
| L0::5.8_degradation::test_degradation_not_rectangular | PASS | 0.0 |  |
| L0::5.8_degradation::test_fullband_noise_snr | PASS | 2.4 |  |
| L0::5.8_degradation::test_lowpass_adapter_runs | PASS | 0.0 |  |
| L0::5.8_degradation::test_time_varying | PASS | 0.0 |  |
| L0::5.8_degradation::test_wind_noise_slope | PASS | 0.0 |  |
| L0::5.9_export::test_export_all | PASS | 39.6 |  |
| L0::5.10_smoke::test_smoke_train | PASS | 53.5 |  |
| L0::5.10_smoke::test_smoke_train_noisy | PASS | 59.5 |  |
| L0::5.12_streaming_memory::test_streaming_peak_memory_measured | PASS | 1.3 |  |
| L0::5.13_noise_probe::test_noise_only_band_probe | PASS | 0.0 |  |
| L0::6.1_ddsp_antialias::test_anti_aliasing | PASS | 0.0 |  |
| L0::6.1_ddsp_antialias::test_anti_aliasing_negative | PASS | 0.0 |  |
| L0::6.1_ddsp_antialias::test_phase_precision | PASS | 0.0 |  |
| L0::6.5_f0::test_f0_half_double_freq | PASS | 0.0 |  |
| L0::6.5_f0::test_f0_synthetic | PASS | 0.0 |  |

**L0 summary: 28/28 passed, 0 failed, 0 skipped**

## L1 — real body-conduction (Vibravox)

| Test | Status | Time (s) | Error |
|------|--------|----------|-------|
| L1::4_l1_adapter::test_l1_adapter_loads_and_protocol | PASS | 0.6 |  |
| L1::4_l1_adapter::test_l1_default_sensor_is_body_conduction | PASS | 0.6 |  |
| L1::4_l1_adapter::test_l1_pairs_are_intrarow_aligned | PASS | 0.5 |  |
| L1::4_l1_adapter::test_l1_sensor_is_bandlimited_vs_ref | PASS | 0.4 |  |
| L1::5.11_smoke_l1::test_smoke_train_l1 | PASS | 15.5 |  |
| L1::4b_l1_bandwidth::test_l1_sensor_effective_bandwidth | PASS | 3.7 |  |
| L1::4c_l1_f0::test_l1_f0_error_by_gender | PASS | 0.7 |  |

**L1 summary: 7/7 passed, 0 failed, 0 skipped**

