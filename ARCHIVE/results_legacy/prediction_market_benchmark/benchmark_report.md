# NWS/MOS Probability Benchmark Report

## Overview
- **Total dates evaluated:** 1095 (IS: 730, OOS: 365)
- **Total bucket-rows:** 6542
- **Training cutoff:** 2023-01-01 (error dist fit on pre-2023 data only)
- **NWS approach:** MOS ensemble + monthly bias correction + monthly sigma -> N(mu, sigma)

## Monthly Error Distribution (Training Data)

| Month | Bias (F) | Sigma (F) | MAE (F) | N |
|-------|----------|-----------|---------|---|
|  1 | +0.38 | 3.72 | 2.70 | 589 |
|  2 | +1.10 | 4.08 | 3.10 | 537 |
|  3 | +1.56 | 3.92 | 3.28 | 589 |
|  4 | +1.02 | 3.95 | 3.15 | 570 |
|  5 | +0.82 | 3.49 | 2.79 | 589 |
|  6 | +0.16 | 2.90 | 2.17 | 570 |
|  7 | +0.24 | 2.57 | 2.01 | 589 |
|  8 | +0.37 | 2.74 | 2.14 | 589 |
|  9 | +0.18 | 2.52 | 1.94 | 570 |
| 10 | -0.19 | 2.89 | 2.18 | 589 |
| 11 | +0.79 | 3.13 | 2.41 | 570 |
| 12 | +0.08 | 3.52 | 2.65 | 589 |

## Point Forecast Accuracy (MAE in F)

| Slice | Source | MAE | Bias | RMSE | N Days |
|-------|--------|-----|------|------|--------|
| Overall | NWS (bias-corrected) | 2.23 | +0.55 | 2.94 | 1095 |
| Overall | NWS (raw MOS) | 2.33 | +1.09 | 3.10 | 1095 |
| Overall | NN Model | 4.48 | -0.08 | 5.81 | 1095 |
| IS | NWS (bias-corrected) | 2.23 | +0.70 | 2.92 | 730 |
| IS | NWS (raw MOS) | 2.36 | +1.24 | 3.13 | 730 |
| IS | NN Model | 4.45 | -0.18 | 5.73 | 730 |
| OOS | NWS (bias-corrected) | 2.25 | +0.24 | 2.97 | 365 |
| OOS | NWS (raw MOS) | 2.28 | +0.78 | 3.03 | 365 |
| OOS | NN Model | 4.53 | +0.12 | 5.98 | 365 |

## Probability Scores (Brier / Log Score)

Lower is better for both metrics.

| Slice | Source | Brier Score | Log Score | N Buckets |
|-------|--------|-------------|-----------|-----------|
| Overall | NWS | 0.1368 | 0.4366 | 6542 |
| Overall | Model | 0.1780 | 0.5718 | 6542 |
| Overall | Market | 0.0174 | 0.0688 | 6542 |
| Period: IS | NWS | 0.1356 | 0.4346 | 4377 |
| Period: IS | Model | 0.1772 | 0.5665 | 4377 |
| Period: IS | Market | 0.0250 | 0.0944 | 4377 |
| Period: OOS | NWS | 0.1392 | 0.4407 | 2165 |
| Period: OOS | Model | 0.1796 | 0.5826 | 2165 |
| Period: OOS | Market | 0.0021 | 0.0171 | 2165 |
| Season: Winter | NWS | 0.1357 | 0.4341 | 1592 |
| Season: Winter | Model | 0.1741 | 0.5584 | 1592 |
| Season: Winter | Market | 0.0176 | 0.0658 | 1592 |
| Season: Spring | NWS | 0.1389 | 0.4456 | 1656 |
| Season: Spring | Model | 0.1896 | 0.6201 | 1656 |
| Season: Spring | Market | 0.0104 | 0.0501 | 1656 |
| Season: Summer | NWS | 0.1360 | 0.4317 | 1656 |
| Season: Summer | Model | 0.1700 | 0.5409 | 1656 |
| Season: Summer | Market | 0.0168 | 0.0707 | 1656 |
| Season: Fall | NWS | 0.1365 | 0.4349 | 1638 |
| Season: Fall | Model | 0.1783 | 0.5673 | 1638 |
| Season: Fall | Market | 0.0249 | 0.0887 | 1638 |

## Key Findings

- **NWS outperforms Model** on Brier score by 23.2% (0.1368 vs 0.1780)
- **Market outperforms Model** on Brier score by 90.2% (0.0174 vs 0.1780)

## Methodology

1. **NWS Distribution:** For each test date, the NWS probability is modeled as N(MOS_forecast + bias_monthly, sigma_monthly), where bias and sigma are computed from training data (2004-2022) only.
2. **Model Distribution:** N(model_mu, model_sigma) from the neural network's probabilistic output.
3. **Market Distribution:** Settled Kalshi market probabilities (clipped to [0.001, 0.999]).
4. **Brier Score:** mean((predicted_prob - outcome)^2). Lower is better.
5. **Log Score:** -mean(y*log(p) + (1-y)*log(1-p)). Lower is better.
