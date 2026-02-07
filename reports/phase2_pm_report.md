# Phase 2 — Baseline Models: Project Manager Report

## Phase Summary

Phase 2 implemented four baseline models and a comprehensive evaluation framework for the NYC daily maximum temperature prediction project. All baselines were trained on 2018–2021 data and evaluated on the 2022 test set (274 days). This phase establishes performance benchmarks that the neural network (Phase 3) must meaningfully exceed to justify its added complexity.

## Deliverables

| File | Lines | Description |
|------|-------|-------------|
| `src/baselines.py` | 738 | Four baseline model classes + run_all_baselines convenience runner |
| `src/evaluate.py` | 877 | Full evaluation framework: metrics, seasonal breakdown, 5 plot types, report generation |
| `tests/test_baselines.py` | 922 | 61 unit tests for baselines |
| `tests/test_evaluate.py` | 728 | 55 unit tests for evaluation |
| `run_baselines.py` | 143 | End-to-end script: load data, fit models, evaluate, save results |
| `results/baselines/` | — | 17 PNG plots + evaluation report |

**Test suite**: 177 tests total (61 baselines + 55 evaluate + 61 Phase 1), all passing.

## Baseline Models Implemented

1. **Persistence** — Predicts TMAX(t) = TMAX(t-1). Oracle baseline using yesterday's actual observation.
2. **Climatology** — Predicts TMAX(t) as the training-set mean for that calendar day-of-year.
3. **Linear Regression (OLS)** — sklearn LinearRegression on 30 scaled features (28 lagged station TMAX/TMIN + sin/cos day).
4. **Ridge Regression (alpha=1.0)** — L2-regularized linear regression on the same feature set.

## Test Set Results (2022-04-02 to 2022-12-31, n=274)

| Model | MAE (°F) | RMSE (°F) | R² | Bias (°F) | Within ±3°F |
|-------|---------|----------|------|---------|------------|
| Persistence | 5.06 | 7.03 | 0.799 | +0.02 | 39.8% |
| Climatology | 6.15 | 7.89 | 0.747 | -0.56 | 32.5% |
| **Ridge (alpha=1.0)** | **4.33** | **5.53** | **0.876** | +0.12 | 40.9% |
| Linear Regression | 4.35 | 5.55 | 0.875 | +0.07 | 40.5% |

**Best baseline**: Ridge regression at 4.33°F MAE and R²=0.876.

## Key Observations

1. **Ridge ≈ Linear Regression**: Virtually identical performance, suggesting multicollinearity among surrounding stations is not causing significant OLS instability. Ridge's slight edge (0.02°F) is negligible.

2. **Regression baselines beat persistence by ~0.7°F MAE**: Surrounding-station information provides meaningful lift over naive persistence, validating the project's core hypothesis.

3. **Climatology is the weakest baseline**: As expected, ignoring recent weather state and relying only on seasonal averages performs worst.

4. **Seasonal variation**: All models struggle most in winter (DJF) and spring (MAM), where MAE is 1–2°F higher than summer/fall. This is consistent with higher weather volatility in those seasons.

5. **Low bias across all models**: All baselines have near-zero overall bias (< 0.6°F), indicating no systematic directional error.

6. **Stretch goal gap**: The project's stretch goal is ≤2°F MAE. The best baseline sits at 4.33°F — the neural network needs to roughly halve the error, which is ambitious but provides a clear target.

## Evaluation Framework

The evaluation module provides:
- **8 scalar metrics**: MAE, RMSE, R², bias, within ±1/2/3°F, max absolute error
- **Seasonal breakdown**: Per-season MAE/RMSE/bias for meteorological seasons
- **5 visualization types**: Actual vs. predicted scatter, time series overlay, residual histogram, residuals by month (box plot), multi-model bar chart comparison
- **Report generation**: Automated text + plot output to any directory
- **Reusable for future phases**: Same framework will evaluate NN models in Phase 3+

## Risks and Recommendations for Phase 3

1. **Ridge/OLS near-parity** suggests the relationship may be approximately linear. The NN must capture non-linear interactions to justify its complexity — monitor carefully.
2. **Winter performance is weakest** across all baselines. Phase 4 sensitivity experiments should test whether additional upstream (W/NW) stations or longer lag windows improve winter predictions.
3. **~40% of predictions within ±3°F** for the best baseline is a reasonable starting point but leaves significant room for improvement.

## Phase Status

Phase 2 is **COMPLETE**. Ready to proceed to Phase 3 (Neural Network V1).
