# Best-Model Lineage Audit + Top-2 Benchmark

## Lineage confirmation

- Canonical best-model-derived variants: **E0, E1, E2**.
- In `probabilistic_ensemble_experiments_v2`, **E3-E8 are retrained from raw features and are not built on canonical `data/best_model_predictions_*` artifacts**.
- E0 parity check vs `best_model_run`: max absolute Brier-score diff = **0.000000**.

## Top-2 benchmark (NWS + Kalshi PreSettlement) among best-model-derived variants

               model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
  E1_global_isotonic             0.133412           0.141775                     0.127061         0.130606       0.139298                     -121.42                      -16.80
E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
