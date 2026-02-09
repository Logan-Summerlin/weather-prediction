# Kalshi KXHIGHNY Real-Data Comprehensive Backtest Report

**Generated:** 2026-02-09 19:53:57

---

## 1. Executive Summary

**Strategy:** S0396_ev0.15_proportional_kf0.05_fee0.07_mp0.05_br10000
**In-Sample Period:** 2023-2024 (simulated market probabilities)
**Out-of-Sample Period:** 2025 (real GHCN data, real Kalshi structure)

**Overall Verdict:** VALIDATED

The strategy demonstrates strong out-of-sample performance (Sharpe=9.72, ROI=26.3%). The model's edge persists on unseen data.

---

## 2. Data Sources and Quality Assessment

| Data Element | Source | Type |
|-------------|--------|------|
| Temperature observations | NOAA GHCN-Daily (USW00094728) | Real |
| Market structure/buckets | Kalshi API (KXHIGHNY settled) | Real |
| Settlement outcomes | Kalshi API (result field) | Real |
| Settlement temperatures | Kalshi API (expiration_value) | Real |
| Model predictions | Ridge regression trained on 2018-2024 GHCN | Real (OOS) |
| Market ex-ante probabilities | Climatological Gaussian model | Constructed |

**Note on market probabilities:** The Kalshi public API for settled markets
provides only settlement prices (0 or 100 cents), not the historical trading
prices available to participants before the temperature was observed. Market
probabilities are constructed from a climatological + persistence forecast
model, which approximates the market's ex-ante pricing.

**OOS data quality:** 365 trading days, 2165 market records

---

## 3. Model Calibration

### Brier Score Comparison

| Period | Model Brier | Market Brier | Delta | Interpretation |
|--------|-------------|--------------|-------|----------------|
| In-Sample | 0.0243 | 0.0331 | -0.0088 | Model better |
| OOS | 0.2165 | 0.2315 | -0.0150 | Model better |

### Seasonal Brier Breakdown (OOS)

| Season | Model Brier | Market Brier | Delta | N |
|--------|-------------|--------------|-------|---|
| Winter (DJF) | 0.1944 | 0.2007 | -0.0063 | 515 |
| Spring (MAM) | 0.2434 | 0.2533 | -0.0099 | 552 |
| Summer (JJA) | 0.2120 | 0.2256 | -0.0136 | 552 |
| Fall (SON) | 0.2146 | 0.2445 | -0.0299 | 546 |

---

## 4. In-Sample Results (2023-2024)

| Metric | Value |
|--------|-------|
| Total P&L | $6452.63 |
| ROI | 64.5% |
| Sharpe Ratio | 9.34 |
| Win Rate | 60.8% |
| Max Drawdown | $155.73 |
| Trades | 610 |
| Avg EV | 0.2834 |

---

## 5. Out-of-Sample Results (2025)

| Metric | Value |
|--------|-------|
| Total P&L | $2625.30 |
| ROI | 26.3% |
| Sharpe Ratio | 9.72 |
| Win Rate | 64.8% |
| Max Drawdown | $138.93 |
| Trades | 193 |
| Avg EV | 0.2492 |

---

## 6. IS vs OOS Stability Analysis

| Metric | In-Sample | OOS | Change | Verdict |
|--------|-----------|-----|--------|---------|
| sharpe_ratio | 9.34 | 9.72 | +0.38 | STRONG: OOS Sharpe >= 1.5, strategy validated |
| roi | 64.5% | 26.3% | -38.3% | POSITIVE: OOS profitable |
| win_rate | 60.8% | 64.8% | +3.9% | STABLE: Win rate within +/-5% |
| max_drawdown | $155.73 | $138.93 | $-16.81 | STABLE: OOS DD < 2x in-sample |
| total_pnl | $6452.63 | $2625.30 | $-3827.33 | PROFITABLE |
| n_trades | 610.0 | 193.0 | -417.0 | See details |
| brier_delta | -0.0088 | -0.0150 | -0.0062 | EDGE PERSISTS: Model still beats market |

---

## 7. Risk Assessment

- **Max Drawdown (IS):** $155.73 (1.6% of bankroll)
- **Max Drawdown (OOS):** $138.93 (1.4% of bankroll)
- **Value at Risk (5%):** $-26.11
- **Expected Shortfall (5%):** $-30.51

---

## 8. Seasonal Performance (OOS)

| Season | P&L | Trades | Win Rate |
|--------|-----|--------|----------|
| Winter (DJF) | $242.15 | 15 | 66.7% |
| Spring (MAM) | $806.59 | 81 | 51.9% |
| Summer (JJA) | $597.47 | 45 | 68.9% |
| Fall (SON) | $979.09 | 52 | 80.8% |

---

## 9. Trading Recommendation

**Recommendation:** Proceed to live paper trading with frozen parameters

The OOS Sharpe ratio exceeds 1.5 with positive ROI. Per the backtesting plan, this validates the strategy for paper trading. Recommended next steps:
1. Run paper trading for 30-60 days
2. Monitor for regime changes in market efficiency
3. If paper trading confirms, deploy with half-Kelly sizing

### Strategy Configuration

```json
{
  "strategy_name": "S0396_ev0.15_proportional_kf0.05_fee0.07_mp0.05_br10000",
  "n_trades": 610,
  "n_days": 7310,
  "total_pnl": 6452.62861100753,
  "roi": 0.6452628611007529,
  "sharpe_ratio": 9.343655062200886,
  "max_drawdown": 155.73362845495785,
  "win_rate": 0.6081967213114754,
  "avg_ev": 0.28338276467990564,
  "selection_reason": "Both Sharpe > 2.0 and within 0.2; selected higher P&L"
}
```

---

## 10. Methodology Notes

### Model
- Ridge regression (alpha=1.0) trained on GHCN 2018-2024 data
- 14 surrounding station TMAX lag-1 features + NYC autoregressive + cyclical date
- Seasonally-varying prediction sigma from training residuals

### Market Probability Construction
- Climatological + persistence Gaussian: 40% yesterday's TMAX + 60% climatology
- Monthly-varying sigma from 30-year NYC climate normals
- This is a limitation: real market prices would reflect more information

### Backtest Mechanics
- Each Kalshi market is treated as a binary contract
- Model probability vs market probability determines trade direction and EV
- Kelly criterion sizing with frozen parameters from IS optimization
- Fee rate: 7% on winnings
