# Rerun Notes: Probabilistic Ensemble + Top-3 Benchmark

## 1) Probabilistic ensemble experiment rerun (E0-E8)

Sorted by CRPS (lower is better):

| experiment                          | crps               | bucket_brier       | bucket_log         |
|-------------------------------------|--------------------|--------------------|--------------------|
| E8_feature_pruning_sweep            | 2.5786824226379395 | 0.0595228729157481 | 0.9941504783697726 |
| E3_weighted_ensemble_E4_uncertainty | 2.938679107289772  | 0.0667046401185634 | 1.1361690606019743 |
| E7_regularization_sweep             | 2.958688497543335  | 0.0663289727692986 | 1.1332756028750353 |
| E0_baseline_ensemble                | 3.03767204284668   | 0.0683180964631683 | 1.1706366773631138 |
| E2_seasonal_calibration             | 3.03767204284668   | 0.0347095068266104 | 0.5895130922873132 |
| E1_global_isotonic                  | 3.03767204284668   | 0.033119520263478  | 0.5417491401593058 |
| E6_quantile                         | 3.604543447494507  | 0.0676059453593787 | 1.151525833732945  |
| E5_mdn2                             | 8.445590019226074  | 0.0961505790287018 | 1.9373030046157595 |

## 2) E0 parity check vs canonical best-model baseline

- Compared `best_model_run/presettlement_brier_scores.csv` vs `e0_e1_e2/E0_baseline_ensemble/presettlement_brier_scores.csv`.
- Maximum absolute Brier-score difference across all slices/sources: **0.000000** (expected 0.0 if identical).

## 3) Top-3 model benchmark vs NWS and Kalshi PreSettlement

Top 3 selected from the rerun summary: E8_feature_pruning, E7_regularized, E3E4_weighted_uncertainty.

| model                     | overall_model_brier | overall_nws_brier  | overall_presettlement_brier | oos_model_brier    | oos_nws_brier      | best_model_all_trading_pnl | best_model_oos_trading_pnl |
|---------------------------|---------------------|--------------------|-----------------------------|--------------------|--------------------|----------------------------|----------------------------|
| E8_feature_pruning        | 0.1742735742464218  | 0.1417753307718879 | 0.1270611137975499          | 0.1728777742998014 | 0.1392981064926715 | -289.47                    | -46.25                     |
| E7_regularized            | 0.1814093457293967  | 0.1417753307718879 | 0.1270611137975499          | 0.1800680743442761 | 0.1392981064926715 | -301.02                    | -49.93                     |
| E3E4_weighted_uncertainty | 0.1815202320976746  | 0.1417753307718879 | 0.1270611137975499          | 0.1797862780813826 | 0.1392981064926715 | -310.09                    | -51.54                     |

## 4) Reference benchmark for best-model-based E0/E1/E2

| model                   | overall_model_brier | overall_nws_brier  | overall_presettlement_brier | oos_model_brier    | oos_nws_brier      | best_model_all_trading_pnl | best_model_oos_trading_pnl |
|-------------------------|---------------------|--------------------|-----------------------------|--------------------|--------------------|----------------------------|----------------------------|
| E1_global_isotonic      | 0.1334121005263738  | 0.1417753307718879 | 0.1270611137975499          | 0.1306061664610531 | 0.1392981064926715 | -121.42                    | -16.8                      |
| E0_baseline_ensemble    | 0.1335165484415357  | 0.1417753307718879 | 0.1270611137975499          | 0.1301897912589733 | 0.1392981064926715 | -121.53                    | -19.52                     |
| E2_seasonal_calibration | 0.134256089078011   | 0.1417753307718879 | 0.1270611137975499          | 0.1315652281929709 | 0.1392981064926715 | -126.52                    | -16.64                     |