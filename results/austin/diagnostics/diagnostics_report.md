# Model Diagnostics — Austin (AUS)

- Model evaluated: **HeteroscedasticNN**
- Forecast days: 1096
- Overall residual bias (actual - mu): **-0.24 F**, MAE 4.30 F, RMSE 5.82 F
- Sigma calibration ratio (realized_rmse / mean_sigma): **0.11** (mean sigma 54.60 F; CONSTANT)
- PIT uniform: **False** (KS 0.396, p 0.000)
- Mean CRPS: 13.006 F

> ⚠️ **Sigma pathology detected:** the model emits a single constant sigma of 54.6 F. This is the convergence failure where sigma absorbs the mu residual and the distribution is effectively uninformative. Retrain with the Phase 0 fix (mu head initialized at target mean; log_sigma clamped) before trusting any probability from this model.

## Residual bias by season

| Season | n | bias (F) | MAE | RMSE |
|---|---|---|---|---|
| Winter | 271 | -0.73 | 5.90 | 7.54 |
| Spring | 276 | -0.10 | 4.36 | 5.63 |
| Summer | 276 | -1.24 | 2.71 | 3.81 |
| Fall | 273 | +1.10 | 4.26 | 5.73 |

## Residual bias by temperature regime (mu terciles)

| Regime | n | bias (F) | MAE |
|---|---|---|---|
| cold | 365 | -0.10 | 5.96 |
| normal | 365 | +0.50 | 4.23 |
| hot | 366 | -1.12 | 2.72 |

## Model vs market (real Kalshi presettlement)

- Contracts: 3214 over 592 days (2023-05-11 -> 2024-12-31)
- **Model Brier 0.2166** vs **market Brier 0.1446** (edge -0.0720)
- Mean abs model-vs-market disagreement: 0.279
- Verdict: **NO_EDGE_MONITOR**
