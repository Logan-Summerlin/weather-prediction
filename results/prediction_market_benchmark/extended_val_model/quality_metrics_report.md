# Extended Validation Model Benchmark Report

Generated with retrained model from results/retrain_extended_validation/
Calibration windows: cal2023 (2023 only), cal2023_2024 (2023-2024)

## Reference Benchmarks
| Source | Overall Brier | OOS Brier |
|--------|--------------|-----------|
| Kalshi PreSettlement | 0.127061 | 0.098839 |
| NWS | 0.141775 | 0.139298 |

## Top 15 Variants by OOS Brier
| Rank | Variant | Overall Brier | OOS Brier | ECE | OOS ECE |
|------|---------|--------------|-----------|-----|---------|
| 1 | E19_platt_beta_calibration_cal2023 | 0.115187 | 0.102683 | 0.016148 | 0.022512 |
| 2 | E13_neural_synthesis_mlp_cal2023 | 0.115301 | 0.103024 | 0.017790 | 0.020112 |
| 3 | E11_synthesis_stacker_market_aware_cal2023 | 0.115488 | 0.103320 | 0.033033 | 0.060294 |
| 4 | E22_expanded_platt_e13_cal2023 | 0.116314 | 0.104366 | 0.022637 | 0.026945 |
| 5 | E17_contract_brier_synthesis_cal2023 | 0.113616 | 0.105331 | 0.010308 | 0.021897 |
| 6 | E31_quantile_crossing_synthesis_cal2023 | 0.115320 | 0.105630 | 0.022737 | 0.033368 |
| 7 | E27_conformal_prediction_cal2023 | 0.115578 | 0.106028 | 0.015295 | 0.021487 |
| 8 | E33_regime_resolution_boost_cal2023 | 0.115763 | 0.106320 | 0.015596 | 0.016888 |
| 9 | E21_platt_recalibrated_e17_cal2023 | 0.114088 | 0.106509 | 0.017734 | 0.028351 |
| 10 | E32_platt_conformal_e17_cal2023 | 0.114351 | 0.106784 | 0.028584 | 0.040919 |
| 11 | E26_tail_weighted_brier_synthesis_cal2023 | 0.118459 | 0.108090 | 0.032895 | 0.033256 |
| 12 | E28_ensemble_disagreement_cal2023 | 0.117858 | 0.109410 | 0.017268 | 0.018530 |
| 13 | E18_regime_adaptive_ensemble_cal2023 | 0.123168 | 0.111785 | 0.048619 | 0.040763 |
| 14 | E30_conformal_neural_sharpener_cal2023 | 0.121915 | 0.112168 | 0.025316 | 0.032588 |
| 15 | E25_regime_sigma_platt_cal2023_2024 | 0.131309 | 0.125433 | 0.011798 | 0.018936 |

## Brier Decomposition (Top 5)
| Variant | Brier | Reliability | Resolution | Uncertainty |
|---------|-------|------------|------------|-------------|
| E19_platt_beta_calibration_cal2023 | 0.115187 | 0.000469 | 0.026316 | 0.141453 |
| E13_neural_synthesis_mlp_cal2023 | 0.115301 | 0.000509 | 0.025825 | 0.141453 |
| E11_synthesis_stacker_market_aware_cal2023 | 0.115488 | 0.001363 | 0.026579 | 0.141453 |
| E22_expanded_platt_e13_cal2023 | 0.116314 | 0.000845 | 0.025844 | 0.141453 |
| E17_contract_brier_synthesis_cal2023 | 0.113616 | 0.000530 | 0.027585 | 0.141453 |

## Seasonal Brier (Top 3)
| Variant | DJF | MAM | JJA | SON |
|---------|-----|-----|-----|-----|
| E19_platt_beta_calibration_cal2023 | 0.107598 | 0.120698 | 0.118990 | 0.112847 |
| E13_neural_synthesis_mlp_cal2023 | 0.107919 | 0.120354 | 0.119527 | 0.112788 |
| E11_synthesis_stacker_market_aware_cal2023 | 0.106525 | 0.119761 | 0.120346 | 0.114745 |

## New Variants (E23-E33)
| Variant | Overall Brier | OOS Brier | ECE |
|---------|--------------|-----------|-----|
| E31_quantile_crossing_synthesis_cal2023 | 0.115320 | 0.105630 | 0.022737 |
| E27_conformal_prediction_cal2023 | 0.115578 | 0.106028 | 0.015295 |
| E33_regime_resolution_boost_cal2023 | 0.115763 | 0.106320 | 0.015596 |
| E32_platt_conformal_e17_cal2023 | 0.114351 | 0.106784 | 0.028584 |
| E26_tail_weighted_brier_synthesis_cal2023 | 0.118459 | 0.108090 | 0.032895 |
| E28_ensemble_disagreement_cal2023 | 0.117858 | 0.109410 | 0.017268 |
| E30_conformal_neural_sharpener_cal2023 | 0.121915 | 0.112168 | 0.025316 |
| E25_regime_sigma_platt_cal2023_2024 | 0.131309 | 0.125433 | 0.011798 |
| E25_regime_sigma_platt_cal2023 | 0.132043 | 0.125841 | 0.021936 |
| E29_learned_sigma_cal2023 | 0.132900 | 0.129307 | 0.027240 |
| E23_regime_sigma_cal2023 | 0.134284 | 0.130190 | 0.022333 |
| E24_combined_sigma_cal2023 | 0.133903 | 0.130190 | 0.021458 |
| E23_regime_sigma_cal2023_2024 | 0.134284 | 0.130190 | 0.022333 |
| E24_combined_sigma_cal2023_2024 | 0.133903 | 0.130190 | 0.021458 |

## Model vs Benchmarks Summary
- **Best variant**: E19_platt_beta_calibration_cal2023
- **Overall Brier**: 0.115187
- **OOS Brier**: 0.102683
- **vs Kalshi PreSettlement OOS** (0.098839): +0.003844 (worse)
- **vs Kalshi PreSettlement Overall** (0.127061): -0.011874 (better)
- **vs NWS OOS** (0.139298): -0.036615 (better)
- **vs NWS Overall** (0.141775): -0.026588 (better)
