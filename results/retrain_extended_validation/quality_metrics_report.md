# Quality Metrics Report: Extended Validation Retrain

## Overview

| Parameter | Value |
|-----------|-------|
| Train Period | 2000-06-01 to 2019-12-31 |
| Validation Period | 2020-01-01 to 2022-12-31 |
| Test Period | 2023-01-01 to 2024-12-31 |
| Architecture | A_NN_64_32 (FlexibleNN [64,32], BN, dropout=0.15) |
| Ensemble Seeds | [42, 123, 456, 789, 2024] |
| Train Size | 7131 days |
| Val Size | 1096 days |
| Test Size | 731 days |
| Features | 122 |

## Point Prediction Metrics (Test Set)

- **MAE**: 1.9905 F
- **RMSE**: 2.6349 F
- **R2**: 0.9737

### Per-Seed Performance (Test)

| Seed | MAE | RMSE | R2 |
|------|-----|------|----|
| 42 | 2.0179 | 2.666 | 0.973 |
| 123 | 2.0073 | 2.6367 | 0.9736 |
| 456 | 2.012 | 2.6638 | 0.9731 |
| 789 | 2.0255 | 2.6648 | 0.9731 |
| 2024 | 2.0127 | 2.6593 | 0.9732 |
| **Ensemble** | **1.9905** | **2.6349** | **0.9737** |

### Seasonal MAE (Test)

| Season | MAE |
|--------|-----|
| DJF | 2.1002 |
| MAM | 2.4338 |
| JJA | 1.8076 |
| SON | 1.6181 |

## Sigma Recalibration Results

### Monthly Sigma (from validation set)

| Month | Base Sigma | Scale Factor |
|-------|-----------|--------------|
| 1 | 2.5904 | 1.0 |
| 2 | 3.3602 | 1.0 |
| 3 | 3.5113 | 1.0 |
| 4 | 3.0227 | 1.0 |
| 5 | 2.8547 | 1.0 |
| 6 | 2.229 | 1.0 |
| 7 | 2.3733 | 1.0 |
| 8 | 2.0674 | 1.0 |
| 9 | 2.174 | 1.0 |
| 10 | 2.3533 | 1.0 |
| 11 | 2.6879 | 1.0 |
| 12 | 3.1674 | 1.0 |

### Regime Calibration

| Regime | Count | Scale Factor | Actual Std |
|--------|-------|--------------|------------|
| stable | 362 | 0.962 | 2.5064 |
| transition | 361 | 1.0608 | 2.7876 |
| volatile | 373 | 1.0258 | 2.9245 |

## Distributional Quality Metrics (Test Set)

| Metric | Base | Monthly Cal | Regime Cal | Combined Cal | Regime-Cond |
|--------|------|-------------|------------|--------------|-------------|
| CRPS (mean) | 1.4499 | 1.4499 | 1.4520 | 1.4534 | 1.4530 |
| Log Score (NLL) | 2.3755 | 2.3755 | 2.3788 | 2.3900 | 2.3739 |
| PIT KS stat | 0.0602 | 0.0602 | 0.0611 | 0.0575 | 0.0641 |
| PIT KS p-value | 0.0097 | 0.0097 | 0.0081 | 0.0152 | 0.0047 |
| ECE | 0.2188 | 0.2188 | 0.2192 | 0.2240 | 0.2188 |
| Mean Sigma | 2.6966 | 2.6966 | 2.7447 | 2.6626 | 2.7353 |
| 95% PI Width | 10.5700 | 10.5700 | 10.7600 | 10.4400 | 10.7200 |

## Prediction Interval Coverage (Test Set)

| Level | Target | Base | Monthly Cal | Combined Cal | Regime-Cond |
|-------|--------|------|-------------|--------------|-------------|
| 50% | 0.50 | 0.551 | 0.551 | 0.554 | 0.554 |
| 80% | 0.80 | 0.828 | 0.828 | 0.811 | 0.828 |
| 90% | 0.90 | 0.911 | 0.911 | 0.903 | 0.915 |
| 95% | 0.95 | 0.944 | 0.944 | 0.943 | 0.948 |

## Brier Score Decomposition (Combined Calibration)

| Threshold | Brier | Reliability | Resolution | Uncertainty | BSS |
|-----------|-------|-------------|------------|-------------|-----|
| 52.0 F | 0.034406 | 0.001962 | 0.158175 | 0.191039 | 0.8199 |
| 64.9 F | 0.027111 | 0.002981 | 0.225315 | 0.249962 | 0.8915 |
| 79.0 F | 0.034766 | 0.001555 | 0.148733 | 0.182798 | 0.8098 |

## Regime-Conditional Variance Models

| Regime | Count | Sigma | Bias | MAE |
|--------|-------|-------|------|-----|
| low_var | 242 | 2.3405 | 0.0502 | 1.7634 |
| medium_var | 298 | 2.5683 | 0.1133 | 1.8859 |
| high_var | 223 | 3.2722 | 0.4143 | 2.515 |
| seasonal_transition | 333 | 2.7904 | 0.14 | 2.1669 |

## PIT Histogram (Combined Calibration)

| Bin | Count | Expected |
|-----|-------|----------|
| 0.0-0.1 | 59 | 73 |
| 0.1-0.2 | 58 | 73 |
| 0.2-0.3 | 71 | 73 |
| 0.3-0.4 | 82 | 73 |
| 0.4-0.5 | 87 | 73 |
| 0.5-0.6 | 92 | 73 |
| 0.6-0.7 | 71 | 73 |
| 0.7-0.8 | 59 | 73 |
| 0.8-0.9 | 73 | 73 |
| 0.9-1.0 | 79 | 73 |

## Seasonal CRPS (Combined Calibration)

| Season | CRPS |
|--------|------|
| DJF | 1.5851 |
| MAM | 1.7253 |
| JJA | 1.3117 |
| SON | 1.1909 |
