# Kalshi KXHIGHNY Comprehensive Backtest Report

**Generated:** 2026-02-09 17:49:34

---

## 1. Executive Summary

**Strategy:** Best Strategy
**In-Sample Period:** 2023-2024
**Out-of-Sample Period:** 2025

**Overall Verdict:** VALIDATED

The strategy demonstrates strong out-of-sample performance (Sharpe=6.93, ROI=25.6%). The model's edge persists on unseen data.

---

## 2. Model Calibration

### Brier Score Comparison

| Period | Model Brier | Market Brier | Delta | Interpretation |
|--------|-------------|--------------|-------|----------------|
| In-Sample | 0.0243 | 0.0331 | -0.0088 | Model better |
| OOS | 0.0275 | 0.0330 | -0.0055 | Model better |

### Seasonal Brier Breakdown (OOS)

| Season | Model Brier | Market Brier | Delta | N |
|--------|-------------|--------------|-------|---|
| Winter (DJF) | 0.0277 | 0.0393 | -0.0115 | 900 |
| Spring (MAM) | 0.0297 | 0.0281 | 0.0016 | 920 |
| Summer (JJA) | 0.0276 | 0.0315 | -0.0040 | 920 |
| Fall (SON) | 0.0249 | 0.0332 | -0.0083 | 910 |

---

## 3. In-Sample Results (2023-2024)

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

## 4. Out-of-Sample Results (2025)

| Metric | Value |
|--------|-------|
| Total P&L | $2555.74 |
| ROI | 25.6% |
| Sharpe Ratio | 6.93 |
| Win Rate | 55.6% |
| Max Drawdown | $205.85 |
| Trades | 311 |
| Avg EV | 0.2826 |

---

## 5. Stability Analysis

| Metric | In-Sample | OOS | Change | Verdict |
|--------|-----------|-----|--------|---------|
| sharpe_ratio | 9.34 | 6.93 | -2.42 | STRONG: OOS Sharpe >= 1.5, strategy validated |
| roi | 64.5% | 25.6% | -39.0% | POSITIVE: OOS profitable |
| win_rate | 60.8% | 55.6% | -5.2% | DEGRADED: Win rate dropped >5% |
| max_drawdown | $155.73 | $205.85 | $+50.12 | STABLE: OOS DD < 2x in-sample |
| total_pnl | $6452.63 | $2555.74 | $-3896.89 | PROFITABLE |
| n_trades | 610.0 | 311.0 | -299.0 | See details |
| brier_delta | -0.0088 | -0.0055 | +0.0033 | EDGE PERSISTS: Model still beats market |

---

## 6. Risk Assessment

- **Max Drawdown (IS):** $155.73 (1.6% of bankroll)
- **Max Drawdown (OOS):** $205.85 (2.1% of bankroll)
- **Value at Risk (5%):** $-20.61
- **Expected Shortfall (5%):** $-25.14

---

## 7. Seasonal Edge Analysis

| Season | P&L | Trades | Win Rate |
|--------|-----|--------|----------|
| Winter (DJF) | $756.14 | 67 | 67.2% |
| Spring (MAM) | $349.48 | 80 | 43.8% |
| Summer (JJA) | $657.60 | 88 | 52.3% |
| Fall (SON) | $792.51 | 76 | 61.8% |

---

## 8. Trading Recommendation

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
