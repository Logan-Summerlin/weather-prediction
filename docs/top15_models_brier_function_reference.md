# Top 15 Models by Brier Score (Current E0–E22 Benchmark) and Their Implemented Functions

## Source of ranking
Ranking below is taken from `results/prediction_market_benchmark/e0_e8_best_model_base/e0_e22_benchmark_summary.csv`, sorted by `overall_model_brier` ascending.

## Ranked Top 15

| Rank | Model Variant | Overall Brier |
|---:|---|---:|
| 1 | E17_contract_brier_synthesis | 0.114090 |
| 2 | E21_platt_recalibrated_e17 | 0.114406 |
| 3 | E13_neural_synthesis_mlp | 0.116196 |
| 4 | E22_expanded_platt_e13 | 0.116332 |
| 5 | E19_platt_beta_calibration | 0.116378 |
| 6 | E11_synthesis_stacker_market_aware | 0.116579 |
| 7 | E18_regime_adaptive_ensemble | 0.123860 |
| 8 | E3_weighted_ensemble_E4_uncertainty | 0.133306 |
| 9 | E1_global_isotonic | 0.133372 |
| 10 | E4_uncertainty_decomposition | 0.133388 |
| 11 | E5_mdn2 | 0.133444 |
| 12 | E0_baseline_ensemble | 0.133517 |
| 13 | E10_wga_mdn_regime_mixture | 0.133584 |
| 14 | E7_regularization_sweep | 0.133623 |
| 15 | E12_capacity_sweep_residual_synthesis | 0.133770 |

---

## Function and method map (how each top model is implemented)

> Core dispatcher: `_apply_variant(df, variant, cfg)` in `scripts/run_e0_e8_best_model_benchmark.py`.

### 1) E17_contract_brier_synthesis
- **Fit path:** `_fit_e17_contract_brier_synthesis(...)`
- **Apply path:** E17 branch inside `_apply_variant(...)`
- **Mechanics:** contract-level MLP synthesis trained directly on bucket-level outcomes; outputs calibrated `model_prob`.

### 2) E21_platt_recalibrated_e17
- **Fit path:** `_fit_e21_platt_e17(...)`
- **Apply path:** E21 branch in `_apply_variant(...)`
- **Mechanics:** two-stage recalibration over E17 output:
  1. Platt scaling on logits
  2. isotonic remapping to final probability.

### 3) E13_neural_synthesis_mlp
- **Fit path:** `_fit_neural_synthesis_stacker(...)`
- **Apply path:** E13 branch in `_apply_variant(...)`
- **Mechanics:** chronology-safe MLP stacker over model/NWS/market + state features, with post-calibration.

### 4) E22_expanded_platt_e13
- **Fit path:** `_fit_e22_expanded_platt_e13(...)`
- **Apply path:** E22 branch in `_apply_variant(...)`
- **Mechanics:** expanded-feature Platt layer on E13 probabilities (sigma, seasonality, bucket geometry, interactions) followed by isotonic.

### 5) E19_platt_beta_calibration
- **Fit path:** `_fit_e19_platt_beta_cal(...)`
- **Apply path:** E19 branch in `_apply_variant(...)`
- **Mechanics:** Platt+isotonic calibration over E13 outputs (compact recalibration pipeline).

### 6) E11_synthesis_stacker_market_aware
- **Fit path:** `_fit_synthesis_stacker(...)`
- **Apply path:** E11 branch in `_apply_variant(...)`
- **Mechanics:** logistic stacker combining model/NWS/market with spread, depth, staleness, and sigma-normalized confidence features.

### 7) E18_regime_adaptive_ensemble
- **Fit path:** `_fit_e18_regime_ensemble(...)`
- **Apply path:** E18 branch in `_apply_variant(...)`
- **Mechanics:** regime-conditioned blending of top variant probabilities with seasonal and volatility context.

### 8) E3_weighted_ensemble_E4_uncertainty
- **Fit dependency:** `_fit_experiment_transforms(...)` generates `sigma_mult_global`
- **Apply path:** E3 branch in `_apply_variant(...)`
- **Mechanics:** uncertainty scaling by global sigma multiplier.

### 9) E1_global_isotonic
- **Fit dependency:** `exp.calibrate_global(...)`
- **Apply path:** E1 branch in `_apply_variant(...)`
- **Mechanics:** isotonic calibration of CDF edges then bucket mass by difference.

### 10) E4_uncertainty_decomposition
- **Fit dependency:** residual scale from `_fit_experiment_transforms(...)`
- **Apply path:** E4 branch in `_apply_variant(...)`
- **Mechanics:** sigma decomposition with additive residual variance term.

### 11) E5_mdn2
- **Fit dependency:** global residual offset from `_fit_experiment_transforms(...)`
- **Apply path:** E5 branch in `_apply_variant(...)`
- **Mechanics:** mean-shift correction using global residual bias.

### 12) E0_baseline_ensemble
- **Apply path:** E0 branch in `_apply_variant(...)`
- **Mechanics:** canonical baseline using base `model_mu/model_sigma` to compute bucket probabilities.

### 13) E10_wga_mdn_regime_mixture
- **Apply path:** E10 branch in `_apply_variant(...)`
- **Mechanics:** regime signal derived from day-to-day `model_mu` change magnitude; used in variant-specific probability adjustment.

### 14) E7_regularization_sweep
- **Fit dependency:** `_fit_seasonal_sigma_multiplier(...)` through `_fit_experiment_transforms(...)`
- **Apply path:** E7 branch in `_apply_variant(...)`
- **Mechanics:** season-conditioned sigma multiplier regularization.

### 15) E12_capacity_sweep_residual_synthesis
- **Fit path:** `_fit_capacity_sweep(...)`
- **Apply path:** E12 branch in `_apply_variant(...)`
- **Mechanics:** small-capacity residual gain + sigma gain + global scale optimization from calibration-year sweep.

---

## Shared helper functions used by many top models
- `_build_market_state_features(...)`: liquidity/staleness/sigma normalization features.
- `bench.compute_bucket_probs(...)`: Gaussian bucket mass conversion from (`mu`, `sigma`).
- `e012._cdf(...)`: CDF boundary evaluation for calibrated bucket-probability construction.
- `_mlp_forward(...)`: deterministic forward pass helper for stored MLP weights in advanced variants.
