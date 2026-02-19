# MOS Integration Report

**Generated:** 2026-02-09 23:22

## Executive Summary

**Verdict: C: MOS SUPERIOR**

MOS proxy has lower Brier score than our model. Our enhanced proxy was too weak a benchmark, inflating apparent edge. MOS MAE=2.51F vs NN ~4.3F.

**Recommendation:** Rethink strategy. Consider using MOS forecasts as input features. The model's apparent edge against the enhanced proxy was a mirage.

## 1. MOS Forecast Accuracy vs Our Model

| Source | MAE (F) | RMSE (F) | R2 | Bias (F) | n |
|--------|---------|----------|-----|----------|---|
| GFS MOS | 2.69 | 3.59 | 0.960 | -0.32 | 8072 |
| NAM MOS | 2.80 | 3.75 | 0.956 | -0.90 | 8015 |
| MOS ENSEMBLE | 2.51 | 3.36 | 0.965 | -0.61 | 8072 |
| NN Model (benchmark) | ~4.3 | ~5.7 | ~0.87 | - | - |
| Ridge Model (benchmark) | ~4.3 | ~5.7 | ~0.88 | - | - |

### Seasonal MAE Breakdown

| Season | n | GFS_MOS | NAM_MOS | MOS_ENSEMBLE |
| --- | --- | --- | --- | --- |
| Winter | 2022 | 2.88 | 3.05 | 2.75 |
| Spring | 2024 | 3.34 | 3.25 | 3.04 |
| Summer | 2024 | 2.28 | 2.40 | 2.10 |
| Fall | 2002 | 2.24 | 2.51 | 2.15 |

## 2. Brier Score Comparison (THE Critical Test)

### IS (2023-2024)

| Comparison | Model Brier | Comp Brier | Delta | Winner |
|-----------|-------------|------------|-------|--------|
| vs Mos Proxy | 0.1772 | 0.1381 | +0.0391 | Comparison |
| vs Enhanced Proxy | 0.1772 | 0.1834 | -0.0061 | Model |
| vs Naive Proxy | 0.1772 | 0.1875 | -0.0103 | Model |
| vs Kalshi Market | 0.1772 | 0.0250 | +0.1522 | Comparison |

**Seasonal Brier (IS (2023-2024)):**

| Season | n | model_brier | mos_proxy_brier | enhanced_proxy_brier | naive_proxy_brier | kalshi_brier |
| --- | --- | --- | --- | --- | --- | --- |
| Winter | 1077 | 0.1799 | 0.1411 | 0.1911 | 0.1952 | 0.0236 |
| Spring | 1104 | 0.1835 | 0.1395 | 0.1866 | 0.1875 | 0.0153 |
| Summer | 1104 | 0.1686 | 0.1384 | 0.1733 | 0.1767 | 0.0251 |
| Fall | 1092 | 0.1770 | 0.1336 | 0.1826 | 0.1909 | 0.0360 |

### OOS (2025)

| Comparison | Model Brier | Comp Brier | Delta | Winner |
|-----------|-------------|------------|-------|--------|
| vs Mos Proxy | 0.1796 | 0.1399 | +0.0397 | Comparison |
| vs Enhanced Proxy | 0.1796 | 0.1881 | -0.0085 | Model |
| vs Naive Proxy | 0.1796 | 0.1919 | -0.0123 | Model |
| vs Kalshi Market | 0.1796 | 0.0021 | +0.1776 | Comparison |

**Seasonal Brier (OOS (2025)):**

| Season | n | model_brier | mos_proxy_brier | enhanced_proxy_brier | naive_proxy_brier | kalshi_brier |
| --- | --- | --- | --- | --- | --- | --- |
| Winter | 515 | 0.1620 | 0.1217 | 0.1727 | 0.1733 | 0.0051 |
| Spring | 552 | 0.2018 | 0.1506 | 0.2059 | 0.2066 | 0.0006 |
| Summer | 552 | 0.1728 | 0.1399 | 0.1836 | 0.1888 | 0.0003 |
| Fall | 546 | 0.1808 | 0.1463 | 0.1893 | 0.1979 | 0.0026 |

## 3. Strategy Profitability Against MOS Proxy

| Metric | vs MOS Proxy | vs Enhanced Proxy |
|--------|-------------|-------------------|
| Total strategies | 448 | 448 |
| With trades | 448 | 448 |
| Profitable | 0 | 448 |
| % Profitable | 0.0% | 100.0% |

**Best MOS Strategy (IS):**
- PnL: $-683
- Sharpe: -0.93
- Win Rate: 25.3%
- Trades: 768

### IS vs OOS Performance

| Proxy | Period | PnL | Sharpe | Win Rate | Trades |
|-------|--------|-----|--------|----------|--------|
| MOS Proxy | IS | $-683 | -0.93 | 25.3% | 768 |
| MOS Proxy | OOS | $-655 | -1.84 | 22.3% | 364 |
| Enhanced Proxy | IS | $3119 | 6.53 | 58.1% | 351 |
| Enhanced Proxy | OOS | $1573 | 7.48 | 60.5% | 162 |

## 4. Verdict and Recommendation

### Scenario: C: MOS SUPERIOR

MOS proxy has lower Brier score than our model. Our enhanced proxy was too weak a benchmark, inflating apparent edge. MOS MAE=2.51F vs NN ~4.3F.

**Recommendation:** Rethink strategy. Consider using MOS forecasts as input features. The model's apparent edge against the enhanced proxy was a mirage.

### Next Steps

1. Add MOS forecasts as input features to the NN
2. Retrain with MOS as additional signal
3. Re-evaluate edge after MOS integration
