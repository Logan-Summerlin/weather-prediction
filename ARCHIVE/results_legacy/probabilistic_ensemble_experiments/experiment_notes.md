# Probabilistic Ensemble Experiments (E0-E8) — Best-Model-Based

All experiments in this file are derived from canonical best-model predictions (data/best_model_predictions_*).

                         experiment     crps      nll      mae  bucket_brier  bucket_log
E3_weighted_ensemble_E4_uncertainty 1.428507 2.342912 1.991580      0.032601    0.486044
       E4_uncertainty_decomposition 1.428621 2.342114 1.991580      0.032600    0.485398
                 E1_global_isotonic 1.428917 2.342318 1.991580      0.032655    0.486796
               E0_baseline_ensemble 1.428917 2.342318 1.991580      0.032617    0.485052
            E2_seasonal_calibration 1.428917 2.342318 1.991580      0.033421    0.559228
                            E5_mdn2 1.430663 2.343573 1.999526      0.032408    0.483187
            E7_regularization_sweep 1.434352 2.354375 1.991580      0.032685    0.491174
                        E6_quantile 1.450947 2.355189 2.039437      0.033126    0.489073
           E8_feature_pruning_sweep 1.456030 2.366666 2.039437      0.034331    0.571002

## Benchmark
```json
{
  "lineage": "all_experiments_best_model_based",
  "calibration_period": "2023",
  "test_period": "2024",
  "best_experiment": "E3_weighted_ensemble_E4_uncertainty",
  "baseline_crps": 1.428916605678299,
  "best_crps": 1.428506947916711,
  "improvement_pct": 0.028669116165355557,
  "baseline_cov90": 0.8961748633879781,
  "baseline_cov95": 0.953551912568306,
  "best_cov90": 0.8879781420765027,
  "best_cov95": 0.9453551912568307
}
```
