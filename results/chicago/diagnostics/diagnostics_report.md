# Model Diagnostics — Chicago (CHI)

- Model evaluated: **ridge_base**
- Forecast days: 1055
- Overall residual bias (actual - mu): **+1.31 F**, MAE 5.55 F, RMSE 7.37 F
- Sigma calibration ratio (realized_rmse / mean_sigma): **0.98** (mean sigma 7.52 F; heteroscedastic)
- PIT uniform: **False** (KS 0.103, p 0.000)
- Mean CRPS: 4.017 F

## Residual bias by season

| Season | n | bias (F) | MAE | RMSE |
|---|---|---|---|---|
| Winter | 260 | -0.25 | 6.12 | 8.16 |
| Spring | 276 | +2.23 | 6.76 | 8.80 |
| Summer | 276 | +1.53 | 4.68 | 5.85 |
| Fall | 243 | +1.70 | 4.55 | 6.12 |

## Residual bias by temperature regime (mu terciles)

| Regime | n | bias (F) | MAE |
|---|---|---|---|
| cold | 352 | +0.74 | 6.09 |
| normal | 351 | +2.34 | 5.89 |
| hot | 352 | +0.86 | 4.68 |

## Model vs market (real Kalshi presettlement)

- Contracts: 6012 over 1055 days (2022-12-22 -> 2026-02-14)
- **Model Brier 0.1843** vs **market Brier 0.1283** (edge -0.0560)
- Mean abs model-vs-market disagreement: 0.218
- Verdict: **NO_EDGE_MONITOR**
