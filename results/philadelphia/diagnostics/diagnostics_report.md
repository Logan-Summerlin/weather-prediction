# Model Diagnostics — Philadelphia (PHL)

- Model evaluated: **ridge_base**
- Forecast days: 364
- Overall residual bias (actual - mu): **-0.59 F**, MAE 6.63 F, RMSE 8.41 F
- Sigma calibration ratio (realized_rmse / mean_sigma): **1.19** (mean sigma 7.08 F; heteroscedastic)
- PIT uniform: **True** (KS 0.065, p 0.087)
- Mean CRPS: 4.693 F

## Residual bias by season

| Season | n | bias (F) | MAE | RMSE |
|---|---|---|---|---|
| Winter | 102 | -4.11 | 8.07 | 9.94 |
| Spring | 92 | +1.90 | 8.30 | 9.97 |
| Summer | 92 | -0.44 | 5.18 | 6.78 |
| Fall | 78 | +0.91 | 4.50 | 5.45 |

## Residual bias by temperature regime (mu terciles)

| Regime | n | bias (F) | MAE |
|---|---|---|---|
| cold | 121 | -2.61 | 8.16 |
| normal | 121 | +1.73 | 6.56 |
| hot | 122 | -0.88 | 5.19 |

## Model vs market (real Kalshi presettlement)

- Contracts: 2171 over 364 days (2024-11-20 -> 2026-02-14)
- **Model Brier 0.1964** vs **market Brier 0.1132** (edge -0.0832)
- Mean abs model-vs-market disagreement: 0.217
- Verdict: **NO_EDGE_MONITOR**
