# Real Kalshi Data Backtesting Plan

## Objective

Use **real** historical KXHIGHNY market data from Kalshi to:

1. **In-sample (2023-2024):** Discover the best trading strategy by backtesting our synthesis model's predictions against actual resolved Kalshi markets.
2. **Out-of-sample (2025):** Validate that best strategy on unseen 2025 market data to measure true forward performance.

This replaces the synthetic data pipeline (which used fabricated market prices and had a built-in model edge) with real market prices where the model must earn its edge honestly.

---

## Part 1: In-Sample Strategy Discovery (2023-2024)

### Step 1 — Pull Historical KXHIGHNY Markets

Use `KalshiClient.get_historical_markets()` to retrieve all settled KXHIGHNY contracts for 2023-2024. The Kalshi public API (`https://api.elections.kalshi.com/trade-api/v2`) serves this data without authentication.

```python
from src.kalshi_client import KalshiClient, parse_market_buckets

client = KalshiClient()

# Fetch all settled markets for 2023 and 2024
markets_2023 = client.get_historical_markets(
    series_ticker="KXHIGHNY",
    min_date="2023-01-01T00:00:00Z",
    max_date="2023-12-31T23:59:59Z",
)
markets_2024 = client.get_historical_markets(
    series_ticker="KXHIGHNY",
    min_date="2024-01-01T00:00:00Z",
    max_date="2024-12-31T23:59:59Z",
)

all_markets = markets_2023 + markets_2024
print(f"Total settled markets: {len(all_markets)}")
```

**Expected output:** Each market dict contains `ticker`, `title`, `yes_ask`, `no_ask`, `last_price`, `result` (outcome), `close_time`, `event_ticker`, and other fields. A typical KXHIGHNY event has 5-8 bucket contracts per day (e.g., "Below 40°F", "40-49°F", "50-59°F", ..., "Above 89°F").

**Expected volume:** Roughly 365 × 2 years × ~6 buckets/day = ~4,380 market rows.

### Step 2 — Parse Markets into Structured DataFrame

Convert raw market JSON into a clean DataFrame aligned by date and bucket:

```python
import pandas as pd
from src.kalshi_client import parse_market_threshold

records = []
for market in all_markets:
    parsed = parse_market_threshold(market)

    # Extract the event date from close_time or event_ticker
    # KXHIGHNY tickers follow: KXHIGHNY-{YYMMMDD}-T{threshold}
    # e.g., KXHIGHNY-24JAN15-T40
    close_time = market.get("close_time", "")
    event_date = pd.to_datetime(close_time).date() if close_time else None

    # Market-implied probability from last traded price
    last_price = market.get("last_price")
    yes_ask = market.get("yes_ask")
    implied_prob = (last_price / 100.0) if last_price else (
        (yes_ask / 100.0) if yes_ask else float("nan")
    )

    # Actual outcome: "yes" or "no" from the result field
    result_str = market.get("result", "").lower()
    actual_outcome = 1 if result_str == "yes" else (0 if result_str == "no" else None)

    records.append({
        "date": event_date,
        "ticker": market.get("ticker", ""),
        "bucket": parsed["direction"],
        "threshold_low": parsed.get("threshold_low") or parsed.get("threshold"),
        "threshold_high": parsed.get("threshold_high"),
        "direction": parsed["direction"],
        "market_prob": implied_prob,
        "actual_outcome": actual_outcome,
    })

historical_markets_df = pd.DataFrame(records)
historical_markets_df = historical_markets_df.dropna(subset=["date", "market_prob"])
historical_markets_df.to_csv("data/kalshi_historical_2023_2024.csv", index=False)
print(f"Parsed {len(historical_markets_df)} market rows across "
      f"{historical_markets_df['date'].nunique()} trading days")
```

**Critical checks before proceeding:**
- Verify `date` coverage: should have ~700 unique dates (weekdays + some weekends)
- Verify `market_prob` values are in (0, 1) — prices of 0 or 100 are degenerate
- Verify `actual_outcome` is populated (settled markets should have results)
- Verify `direction` is parsed correctly — spot-check 10 markets manually
- Verify bucket thresholds make sense (typical NYC TMAX range: 10°F to 105°F)

### Step 3 — Generate Model Predictions for 2023-2024

Run the synthesis model to produce Gaussian (mu, sigma) predictions for every day in the 2023-2024 window. The model must use only data available **before** each prediction date (no lookahead).

```python
import numpy as np
from src.synthesis_model import SynthesisTrainer, prepare_synthesis_data

# Load the trained synthesis model checkpoint
trainer = SynthesisTrainer.load("models/synthesis_best.pt")

# Prepare features for the 2023-2024 period
# This uses station observations from day t-1 and NWP data
X_eval, dates_eval, actuals_eval = prepare_synthesis_data(
    start_date="2023-01-01",
    end_date="2024-12-31",
)

# Generate predictions
predictions = trainer.predict(X_eval)
model_mu = predictions["mu"]        # shape (N,)
model_sigma = predictions["sigma"]  # shape (N,)

model_predictions_df = pd.DataFrame({
    "date": dates_eval,
    "model_mu": model_mu,
    "model_sigma": model_sigma,
    "actual_tmax": actuals_eval,
})
model_predictions_df.to_csv("data/model_predictions_2023_2024.csv", index=False)
```

**Important:** If the synthesis model was trained on data that includes 2023-2024, those predictions are in-sample for the model but the *market prices* are unseen. The model's edge (if any) comes from the gap between model-implied and market-implied probabilities. Even if the model "knows" the true temperature distribution, the market may price it differently, creating tradeable opportunities.

If the synthesis model was only trained through 2022, then 2023-2024 predictions are truly out-of-sample for the model as well — this is the ideal scenario.

### Step 4 — Align Model Predictions with Market Data

Use `build_historical_comparison()` to merge model predictions with market data by date:

```python
from src.kalshi_client import build_historical_comparison

comparison_df = build_historical_comparison(
    model_predictions_df=model_predictions_df,
    historical_markets_df=historical_markets_df,
)

print(f"Aligned rows: {len(comparison_df)}")
print(f"Unique dates: {comparison_df['date'].nunique()}")
print(f"Mean model prob: {comparison_df['model_prob'].mean():.3f}")
print(f"Mean market prob: {comparison_df['market_prob'].mean():.3f}")
print(f"Mean prob delta: {comparison_df['prob_delta'].mean():.4f}")
```

**Sanity checks:**
- `prob_delta` should be small on average (model and market roughly agree)
- If `prob_delta` is systematically large (>0.05), investigate calibration
- The `outcome` column should be populated (1 or 0) for every row

### Step 5 — Run Comprehensive Strategy Grid Search

Run the full strategy grid across the 2023-2024 data to find the best configuration:

```python
from src.trading import (
    generate_strategy_grid,
    run_comprehensive_backtest,
    BacktestEngine,
    TradingStrategy,
)

# Prepare the backtest data format
backtest_data = comparison_df.rename(columns={
    "market_prob": "market_price",
    "outcome": "actual_outcome",
}).copy()

# Generate strategy grid — comprehensive permutations
strategies = generate_strategy_grid(
    ev_thresholds=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    sizing_methods=["fixed", "proportional", "fractional_kelly", "capped_kelly"],
    kelly_fractions=[0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
    fee_rates=[0.07],        # Kalshi's actual fee rate
    max_positions=[0.05, 0.10, 0.15, 0.20],
    bankrolls=[10000],        # Normalize to $10K for comparison
)

# Run comprehensive backtest
results = run_comprehensive_backtest(
    backtest_data,
    output_dir="results/kalshi_real_2023_2024",
    strategies=strategies,
    max_strategies=1000,
)

comparison_csv = results["comparison_df"]
comparison_csv.to_csv("results/kalshi_real_2023_2024/all_strategies.csv", index=False)
```

### Step 6 — Analyze In-Sample Results

The key questions to answer from the 2023-2024 backtest:

**A) Does the model have real edge over the market?**
```python
from src.kalshi_client import compute_brier_scores

brier = compute_brier_scores(
    model_probs=comparison_df["model_prob"],
    market_probs=comparison_df["market_prob"],
    outcomes=comparison_df["outcome"],
)
print(f"Model Brier: {brier['model_brier']:.4f}")
print(f"Market Brier: {brier['market_brier']:.4f}")
print(f"Delta: {brier['brier_delta']:.4f}")
# Negative delta = model is better calibrated than the market
```

If the model Brier score is **not** lower than the market Brier score, the model has no demonstrated edge and trading is not advisable. Stop here and investigate calibration before proceeding.

**B) Which strategy maximizes risk-adjusted returns?**

Rank strategies by Sharpe ratio (not raw P&L) since Sharpe accounts for volatility:

```python
top = comparison_csv.sort_values("sharpe_ratio", ascending=False).head(10)
print(top[["strategy_name", "sharpe_ratio", "total_pnl", "roi",
           "win_rate", "max_drawdown", "n_trades"]])
```

**C) Is the edge seasonal?**

Check if the model's edge is concentrated in specific months/seasons. If it only works in summer, a year-round strategy will dilute returns:

```python
# From results/kalshi_real_2023_2024/seasonal_performance.csv
seasonal = pd.read_csv("results/kalshi_real_2023_2024/seasonal_performance.csv")
print(seasonal.pivot(index="strategy", columns="season", values="total_pnl"))
```

**D) Select the best strategy**

Choose the strategy that balances:
- Sharpe ratio >= 1.5 (strong risk-adjusted returns)
- Max drawdown < 15% of bankroll (survivable worst case)
- n_trades >= 100 (sufficient sample size for statistical significance)
- Win rate > 45% (not relying on rare large wins)

Record the exact configuration:
```python
best_strategy_config = {
    "ev_threshold": 0.02,        # example
    "sizing_method": "fractional_kelly",
    "kelly_fraction": 0.10,
    "fee_rate": 0.07,
    "max_position_frac": 0.10,
    "bankroll": 10000,
}
```

---

## Part 2: Out-of-Sample Validation (2025)

### Step 7 — Pull 2025 KXHIGHNY Markets

```python
markets_2025 = client.get_historical_markets(
    series_ticker="KXHIGHNY",
    min_date="2025-01-01T00:00:00Z",
    max_date="2025-12-31T23:59:59Z",
)
```

**Note:** As of February 2025, only ~40 days of data exist. The backtest will have limited statistical power. Options:
- Run on available 2025 data now and re-run quarterly as more data accumulates
- Supplement with paper trading (running the strategy daily without real money) to build a longer track record

### Step 8 — Generate Model Predictions for 2025

```python
X_2025, dates_2025, actuals_2025 = prepare_synthesis_data(
    start_date="2025-01-01",
    end_date="2025-12-31",
)

preds_2025 = trainer.predict(X_2025)

model_predictions_2025 = pd.DataFrame({
    "date": dates_2025,
    "model_mu": preds_2025["mu"],
    "model_sigma": preds_2025["sigma"],
    "actual_tmax": actuals_2025,
})
```

### Step 9 — Run the Best Strategy on 2025 Data

Apply **only** the single best strategy identified in Step 6 — no re-optimization:

```python
# Parse 2025 markets into DataFrame (same as Step 2)
historical_2025_df = parse_2025_markets(markets_2025)  # same logic as Step 2

# Align
comparison_2025 = build_historical_comparison(
    model_predictions_2025, historical_2025_df
)

# Run backtest with the FROZEN strategy from Step 6
best_strategy = TradingStrategy(
    name="Best_from_2023_2024",
    **best_strategy_config,
)

engine = BacktestEngine(best_strategy)

backtest_2025 = comparison_2025.rename(columns={
    "market_prob": "market_price",
    "outcome": "actual_outcome",
}).copy()

result_2025 = engine.run_backtest(backtest_2025)

print(f"2025 OOS Results:")
print(f"  Trades: {result_2025.n_trades}")
print(f"  Total P&L: ${result_2025.total_pnl:.2f}")
print(f"  ROI: {result_2025.roi:.1%}")
print(f"  Sharpe: {result_2025.sharpe_ratio:.2f}")
print(f"  Win Rate: {result_2025.win_rate:.1%}")
print(f"  Max Drawdown: ${result_2025.max_drawdown:.2f}")
```

### Step 10 — Evaluate Out-of-Sample Performance

The critical question: **does the 2023-2024 best strategy hold up on unseen 2025 data?**

| Metric | 2023-2024 (in-sample) | 2025 (out-of-sample) | Verdict |
|--------|----------------------|---------------------|---------|
| Sharpe | from Step 6 | from Step 9 | OOS Sharpe > 1.0 = promising |
| ROI | from Step 6 | from Step 9 | OOS ROI > 0 = profitable |
| Win rate | from Step 6 | from Step 9 | Stable ±5% = robust |
| Max DD | from Step 6 | from Step 9 | OOS DD < 2× in-sample = stable |
| Brier delta | from Step 6 | from Step 9 | Still negative = edge persists |

**Interpretation guidelines:**

- **OOS Sharpe >= 1.5 and positive ROI**: Strategy is validated. Proceed to live paper trading with the frozen parameters.
- **OOS Sharpe 0.5-1.5 and positive ROI**: Edge exists but is weaker than in-sample. Consider more conservative sizing (reduce Kelly fraction by 50%).
- **OOS Sharpe < 0.5 or negative ROI**: The in-sample results were likely overfit. Do not trade live. Diagnose whether the model's calibration degraded, the market became more efficient, or the strategy parameters were overfit to 2023-2024 patterns.
- **OOS Brier delta flips positive (market beats model)**: The model has lost its fundamental edge. No trading strategy can compensate for worse predictions than the market's consensus.

---

## Practical Considerations

### Kalshi API Rate Limits
- The public API allows unauthenticated access but enforces rate limits
- `KalshiClient` already implements 10 req/sec throttling and 3x retry with backoff
- For ~4,000 settled markets, expect ~20 paginated requests (200 per page)
- Total fetch time: ~5-10 seconds per year of data

### Data Availability Gaps
- KXHIGHNY may not trade every calendar day (weekends, holidays, low-volume days)
- Some days may have very few active buckets (winter months with less temperature variance)
- Model predictions require station observations from day t-1, which may have their own gaps
- The alignment step (`build_historical_comparison`) handles this via inner join — only days with both model predictions and market data are included

### Fee Structure
- Kalshi charges fees on winning trades only (currently 7% of profit, not of position)
- Use `fee_rate=0.07` in all strategy configurations
- Fee structure can change — verify current rates at kalshi.com before live trading

### Transaction Cost Assumptions
- The backtest uses `last_price` as the execution price (no slippage)
- Real execution will face bid-ask spreads, especially on low-volume buckets
- Conservative adjustment: reduce in-sample Sharpe by 20-30% for realistic OOS expectations
- The `bid_price` and `ask_price` fields from the API can be used for spread-aware backtesting

### Statistical Significance
- With ~700 trading days (2023-2024), a strategy needs >= 100 trades for the Sharpe ratio to be statistically meaningful at the 95% confidence level
- For 2025 OOS (currently ~40 days), results are directional but not statistically significant
- Consider bootstrapping: resample the trade P&L series 10,000 times to compute confidence intervals on Sharpe and total P&L

### Known Risks
- **Regime change:** Market efficiency may improve over time as more quantitative participants enter Kalshi weather markets
- **Model staleness:** The synthesis model was trained on historical station/NWP data. If the underlying data sources change (station relocations, NWP model upgrades), prediction quality may degrade
- **Liquidity risk:** Some bucket contracts may have very thin orderbooks, making execution at the backtested price unrealistic
- **Calendar effects:** The model may have systematic biases on certain day types (holidays, weekends, season transitions) that don't show up in aggregate metrics
