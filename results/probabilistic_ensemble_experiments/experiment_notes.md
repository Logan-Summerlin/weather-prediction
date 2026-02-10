# Probabilistic Ensemble Experiments (E0-E8)

                         experiment     crps      nll       mae  bucket_brier  bucket_log  sigma_ale_mean  sigma_epi_mean
           E8_feature_pruning_sweep 2.578682 3.186354  2.203869      0.059523    0.994150             NaN             NaN
E3_weighted_ensemble_E4_uncertainty 2.938679 3.368091  2.291748      0.066705    1.136169        11.18174         0.67332
            E7_regularization_sweep 2.958688 3.358185  2.465213      0.066329    1.133276             NaN             NaN
               E0_baseline_ensemble 3.037672 3.408631  2.315317      0.068318    1.170637             NaN             NaN
            E2_seasonal_calibration 3.037672 3.408631  2.315317      0.034710    0.589513             NaN             NaN
                 E1_global_isotonic 3.037672 3.408631  2.315317      0.033120    0.541749             NaN             NaN
                        E6_quantile 3.604543 3.341316  4.970491      0.067606    1.151526             NaN             NaN
                            E5_mdn2 8.445590 4.184897 11.831335      0.096151    1.937303             NaN             NaN

## Benchmark
```json
{
  "best_experiment": "E8_feature_pruning_sweep",
  "baseline_crps": 3.0376720428466797,
  "best_crps": 2.5786824226379395,
  "improvement_pct": 15.109913569820705,
  "baseline_cov90": 1.0,
  "baseline_cov95": 1.0,
  "weighted_cov90": 1.0,
  "weighted_cov95": 1.0
}
```
