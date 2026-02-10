# Top-3 Adjusted Models vs Existing Production Model

Baseline from `results/prediction_market_benchmark/presettlement_brier_scores.csv`:

- Baseline Overall Model Brier: **0.133517**
- Baseline OOS Model Brier: **0.130190**

## Top-3 adjusted results

                    model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl  overall_vs_baseline_delta  oos_vs_baseline_delta
       E8_feature_pruning             0.174274           0.141775                     0.127061         0.172878       0.139298                     -289.47                      -46.25                   0.040757               0.042688
           E7_regularized             0.181409           0.141775                     0.127061         0.180068       0.139298                     -301.02                      -49.93                   0.047893               0.049878
E3E4_weighted_uncertainty             0.181520           0.141775                     0.127061         0.179786       0.139298                     -310.09                      -51.54                   0.048004               0.049596