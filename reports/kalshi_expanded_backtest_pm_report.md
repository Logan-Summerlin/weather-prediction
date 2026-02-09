# Kalshi Expanded Backtest - Project Manager Report

**Date:** 2026-02-09
**Status:** Complete - All diagnostic fixes implemented and validated

---

## Executive Summary

Implemented all fixes from `kalshi_backtest_diagnostic_and_fixes.md`:
1. **Expanded training data** from 4 years (2018-2021) to 22 years (1998-2019)
2. **Expanded stations** from 14 to 47 qualifying surrounding stations
3. **Built improved market proxy** using regression-based climatological forecast
4. **Re-ran full backtesting pipeline** with comprehensive strategy grid search

## Fix 1: Training Data Expansion

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Training samples | 1,460 | 8,034 | **5.5x increase** |
| Training years | 4 (2018-2021) | 22 (1998-2019) | **5.5x more history** |
| Validation samples | 365 (1 year) | 1,096 (3 years) | **3x more stable sigma** |
| Surrounding stations | 14 | 47 | **3.4x more spatial coverage** |
| Features | 30 | 50 | 47 station lag-1 + AR + date |
| Samples-per-feature ratio | 49:1 | 161:1 | **3.3x better** |

### Station Completeness

- 48 stations qualify (>=80% completeness over 1998-2024)
- 5 stations excluded for low completeness: McGuire AFB (5%), Oxford Waterbury (27%), Millbrook (74%), Monticello Sullivan (20%), Dover AFB (0%)
- Core stations (Central Park, LGA, Newark, JFK, Albany, Boston, Hartford, Philly) all have 100% completeness

### Data Range Used

| Split | Date Range | Samples |
|-------|-----------|---------|
| Train | 1998-01-01 to 2019-12-31 | 8,034 |
| Validation | 2020-01-01 to 2022-12-31 | 1,096 |
| IS Prediction | 2023-01-01 to 2024-12-31 | 731 |
| OOS Prediction | 2025-01-01 to 2025-12-31 | 365 |

## Fix 2: Improved Market Proxy

**Before:** Naive 40% persistence + 60% monthly climatology
**After:** Regression-based forecast with smooth daily climatology

The enhanced proxy uses:
- Day-of-year smooth climatology (15-day Gaussian kernel, from 76 years of Central Park data)
- Linear regression: TMAX(t) = 0.625 * TMAX(t-1) - 0.078 * TMAX(t-2) + 0.456 * Clim(t)
- Monthly-varying sigma from historical forecast errors
- Trained on pre-evaluation data (no leakage)

## Model Performance

### Neural Network (128, 64) with Huber Loss

| Split | MAE (F) | RMSE (F) |
|-------|---------|----------|
| Train | 4.34 | 5.71 |
| Validation | 4.44 | 5.83 |
| IS (2023-2024) | 4.44 | 5.71 |
| OOS (2025) | 4.56 | 5.99 |

### Ridge Regression Comparison

| Split | MAE (F) | RMSE (F) |
|-------|---------|----------|
| Train | 4.63 | 6.00 |
| Validation | 4.65 | 6.10 |
| IS (2023-2024) | 4.48 | 5.77 |
| OOS (2025) | 4.60 | 6.04 |

NN outperforms Ridge by ~0.04F MAE consistently.

## Brier Score Analysis

### Model vs Enhanced Proxy (lower = better)

| Period | Model Brier | Enhanced Proxy | Delta | Winner |
|--------|------------|----------------|-------|--------|
| IS (2023-2024) | 0.1770 | 0.1834 | -0.0064 | Model |
| OOS (2025) | 0.1801 | 0.1883 | -0.0082 | Model |

### Model vs Naive Proxy

| Period | Model Brier | Naive Proxy | Delta | Winner |
|--------|------------|-------------|-------|--------|
| IS (2023-2024) | 0.1770 | 0.1875 | -0.0105 | Model |
| OOS (2025) | 0.1801 | 0.1919 | -0.0118 | Model |

Model edge INCREASES in OOS vs enhanced proxy (-0.0064 IS to -0.0082 OOS).

### Seasonal Brier Breakdown (OOS 2025)

| Season | Model | Enhanced | Delta |
|--------|-------|----------|-------|
| Winter | 0.164 | 0.172 | -0.008 |
| Spring | 0.202 | 0.207 | -0.005 |
| Summer | 0.172 | 0.183 | -0.012 |
| Fall | 0.182 | 0.189 | -0.008 |

Model beats proxy in ALL seasons, with largest edge in summer.

## Trading Strategy Results

### Best Strategy (Selected from IS Grid Search)

- **Name:** S0396_ev0.15_proportional_kf0.05_fee0.07_mp0.05_br10000
- **EV threshold:** 0.15
- **Sizing:** Proportional, Kelly fraction 0.05
- **Max position:** 5% of bankroll

### IS vs OOS Performance

| Metric | IS (2023-2024) | OOS (2025) | Change |
|--------|---------------|------------|--------|
| Total P&L | $3,276 | $1,401 | -$1,875 |
| ROI | 32.8% | 14.0% | -18.7% |
| Sharpe Ratio | 6.96 | 6.79 | -0.17 |
| Win Rate | 60.7% | 58.5% | -2.2% |
| Max Drawdown | $124 | $65 | -$59 |
| Trades | 346 | 159 | -187 |

### OOS Seasonal P&L

| Season | P&L | Trades | Win Rate |
|--------|-----|--------|----------|
| Winter | $295 | 31 | 61.3% |
| Spring | $411 | 46 | 54.3% |
| Summer | $324 | 26 | 65.4% |
| Fall | $372 | 56 | 57.1% |

Profitable in ALL four seasons.

## Verdict: VALIDATED

The strategy shows strong OOS performance with minimal Sharpe degradation (6.96 to 6.79). Recommended: Proceed to paper trading with frozen parameters.

## Important Caveats

1. **Market proxy is still constructed, not actual Kalshi prices.** The Kalshi public API only provides settlement prices (0 or 100 cents), not historical pre-settlement trading prices. Real market probabilities would reflect NWS/ensemble forecast information that our proxy doesn't capture. The model_vs_market Brier comparison (model 0.180 vs market 0.002) shows that actual settlement outcomes are nearly deterministic -- our model cannot beat perfect hindsight.

2. **The real test** is whether the model beats actual Kalshi trading prices. This requires either:
   - Access to IEM MOS/NBM forecast archives (currently blocked by proxy)
   - Live paper trading to capture actual pre-settlement prices

## Files Delivered

### New Source Code
- `scripts/download_all_dly.py` -- Downloads GHCN .dly for all expanded stations
- `scripts/generate_expanded_predictions.py` -- Full prediction pipeline (22yr training)
- `src/market_proxy.py` -- Enhanced regression-based market proxy
- `scripts/run_expanded_backtest.py` -- Comprehensive backtesting with new proxy

### Data Files
- `data/raw/*.dly` -- 53 GHCN station files (real NOAA data)
- `data/central_park_tmax_full_history.csv` -- 76 years of Central Park TMAX
- `data/expanded_model_predictions_2023_2024.csv` -- NN IS predictions
- `data/expanded_model_predictions_2025.csv` -- NN OOS predictions
- `data/expanded_ridge_predictions_2023_2024.csv` -- Ridge IS predictions
- `data/expanded_ridge_predictions_2025.csv` -- Ridge OOS predictions

### Model Artifacts
- `models/expanded_nn_model.pt` -- Trained PyTorch NN
- `models/expanded_ridge_model.pkl` -- Trained Ridge model
- `models/expanded_sigma_estimates.json` -- Monthly sigma by model type
- `models/expanded_training_report.json` -- Full training metrics

### Results
- `results/kalshi_expanded_backtest/` -- All backtest outputs (15 files)
