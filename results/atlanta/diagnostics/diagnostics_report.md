# Model Diagnostics — Atlanta (ATL)

- Model evaluated: **HeteroscedasticNN**
- Forecast days: 4506
- Overall residual bias (actual - mu): **-0.14 F**, MAE 3.58 F, RMSE 4.94 F
- Sigma calibration ratio (realized_rmse / mean_sigma): **0.88** (mean sigma 5.63 F; heteroscedastic)
- PIT uniform: **False** (KS 0.084, p 0.000)
- Mean CRPS: 2.676 F

## Residual bias by season

| Season | n | bias (F) | MAE | RMSE |
|---|---|---|---|---|
| Winter | 1160 | +0.13 | 4.79 | 6.60 |
| Spring | 1104 | -0.03 | 3.75 | 4.94 |
| Summer | 1104 | -0.49 | 2.49 | 3.28 |
| Fall | 1138 | -0.18 | 3.25 | 4.25 |

## Residual bias by temperature regime (mu terciles)

| Regime | n | bias (F) | MAE |
|---|---|---|---|
| cold | 1502 | +0.50 | 4.70 |
| normal | 1502 | -0.25 | 3.48 |
| hot | 1502 | -0.66 | 2.58 |

## Model vs market (real Kalshi presettlement)

- Contracts: 72 over 12 days (2026-02-04 -> 2026-02-15)
- **Model Brier 0.1695** vs **market Brier 0.1249** (edge -0.0446)
- Mean abs model-vs-market disagreement: 0.178
- Verdict: **NO_EDGE_MONITOR**
