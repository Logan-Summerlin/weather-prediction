# Prediction Market Benchmark Report

**Date:** 2026-02-10
**Phase:** Model Validation — Prediction Market & NWS Benchmark Testing

## Objective

Test our neural network temperature prediction model against two independent benchmarks:
1. **Kalshi pre-settlement market prices** — real prediction market consensus captured before settlement (last candle before midnight ET close)
2. **NWS/MOS forecast probability distribution** — operational weather forecast converted to probability distribution using historically-calibrated error statistics

## Data Collection

### Kalshi Pre-Settlement Data
- **Source:** Kalshi public API candlestick endpoint (no authentication required)
- **Method:** 1-hour candlesticks, last candle before the overnight dead zone
- **Pre-settlement probability:** Bid/ask midpoint from last available candle
- **Coverage:** 4,611 market rows across 769 dates (2023-01-01 to 2025-02-09)
- **Markets per date:** ~6 binary contracts (above/below/between temperature brackets)
- **Script:** `scripts/fetch_kalshi_presettlement.py`

#### Kalshi Market Schedule & Snapshot Timing

The KXHIGHNY market for day t follows this lifecycle:

| Event | Time | Description |
|-------|------|-------------|
| Market opens | 10:00 AM ET, day t-1 | Trading begins |
| **Our snapshot window** | **5:00 PM – 12:30 AM ET, day t-1 evening** | **Last activity before overnight dead zone** |
| Dead zone | ~3:00 AM – 8:00 AM ET, day t | Zero trading activity; API returns no candles |
| Morning trading resumes | ~8:00-10:00 AM ET, day t | Thin liquidity; day-of weather starts to influence prices |
| Heavy repricing | 12:00-3:00 PM ET, day t | Market incorporates real-time temperature observations |
| Market closes | 11:59 PM ET, day t | Final trading |
| Settlement | ~10:00 AM ET, day t+1 | Settled against official NWS Daily Climate Report |

**Snapshot verification (4,286 rows with timestamps):**

| UTC Time | ET Time (evening of day t-1) | Rows | % |
|----------|------------------------------|------|---|
| 22:00 UTC (day t-1) | 5:00 PM ET | 64 | 1.5% |
| 23:00 UTC (day t-1) | 6:00 PM ET | 68 | 1.6% |
| 00:00 UTC (day t) | 7:00 PM ET | 97 | 2.3% |
| 01:00 UTC (day t) | 8:00 PM ET | 118 | 2.8% |
| 02:00 UTC (day t) | 9:00 PM ET | 271 | 6.3% |
| 03:00 UTC (day t) | 10:00 PM ET | 448 | 10.5% |
| 04:00 UTC (day t) | 11:00 PM ET | 799 | 18.6% |
| 05:00 UTC (day t) | 12:00 AM ET (midnight) | 2,421 | 56.5% |

**All snapshots are from the evening/night of day t-1** — 6 to 18 hours before peak daytime temperature is observed on day t. Zero rows come from daytime on day t. This means the pre-settlement prices reflect a genuine overnight forecast consensus, with no contamination from day-of temperature observations.

Note: 325 rows (7%) had no candlestick data at all (empty presettlement_prob), typically on dates with very low market liquidity.

#### Why Not 6:00 AM ET?

The user's original target was a 6am ET snapshot (18 hours before midnight close). Investigation confirmed that 6am ET falls in a **dead zone** (~3am-8am ET) with zero trading activity. The API returns no candles during this window. The midnight ET snapshot is the latest available genuine pre-observation price.

### NWS/MOS Probability Distribution
- **Source:** MOS (Model Output Statistics) ensemble forecasts from IEM archive
- **Method:** MOS point forecast + monthly bias correction + monthly error sigma → N(mu, sigma)
- **Error distribution:** Fit on training data only (2004-2022) — no data leakage
- **Coverage:** 1,095 dates (2023-2024 IS + 2025 OOS)
- **Script:** `scripts/build_nws_benchmark.py`

## Results Summary

### Probability Calibration (Brier Score — lower is better)

| Source | Overall | IS (2023-2024) | OOS (2025) |
|--------|---------|----------------|------------|
| Kalshi Pre-Settlement | **0.1384** | **0.1421** | **0.0679** |
| NWS/MOS | 0.1407 | 0.1431 | 0.0962 |
| NN Model | 0.1793 | 0.1825 | 0.1187 |
| Kalshi Settled | 0.0253 | 0.0266 | 0.0003 |

**Finding:** Both benchmarks outperform our model on probabilistic calibration. The pre-settlement Kalshi market is the strongest genuine forecaster (Brier 0.1384), followed closely by NWS/MOS (0.1407). Our model lags at 0.1793.

### Point Forecast Accuracy (MAE in °F)

| Source | Overall | IS | OOS |
|--------|---------|-----|-----|
| NWS (bias-corrected) | **2.23** | 2.23 | 2.25 |
| NWS (raw MOS) | 2.33 | 2.36 | 2.28 |
| NN Model | 4.48 | 4.45 | 4.53 |

**Finding:** The NWS/MOS point forecast (MAE 2.23°F) is roughly twice as accurate as our NN model (MAE 4.48°F) for the in-sample and out-of-sample periods tested.

### Calibration Quality (ECE — lower is better)

| Source | ECE |
|--------|-----|
| NWS | **0.0245** |
| Kalshi Pre-Settlement | 0.0768 |
| NN Model | 0.1397 |

**Finding:** NWS forecasts have the best calibration (ECE 0.025), meaning when NWS says 15% probability, the event occurs about 15.7% of the time. Our model's ECE of 0.14 indicates overconfidence — it spreads too much probability mass across buckets.

### Trading Simulation (Model & NWS vs Pre-Settlement Market)

Both our model and NWS can profitably trade against the pre-settlement Kalshi market:

**Best configurations (7% fee on winnings):**

| Signal Source | Threshold | Trades | Win Rate | Net P&L | ROI | Sharpe |
|--------------|-----------|--------|----------|---------|-----|--------|
| NWS | 0.20 | 1,255 | 60.4% | $174 | 32.8% | 0.354 |
| NWS | 0.15 | 1,702 | 57.2% | $182 | 25.1% | 0.279 |
| Model | 0.20 | 1,927 | 52.4% | $115 | 14.0% | 0.152 |
| Model | 0.15 | 2,327 | 53.1% | $107 | 10.3% | 0.118 |

**NWS outperforms the model as a trading signal** — higher ROI (33% vs 14%), higher Sharpe (0.35 vs 0.15), and higher win rate (60% vs 52%). Both remain profitable in OOS (2025).

### Seasonal Analysis

The pre-settlement market performs best in winter (Brier 0.121), where temperature variability is highest and market participants benefit from more information. In spring, NWS slightly outperforms the market (0.139 vs 0.152).

## Key Takeaways

1. **Our NN model underperforms both benchmarks** on probability calibration and point accuracy. The model's sigma (6.3°F constant) is too wide, leading to over-dispersed probability distributions.

2. **The Kalshi pre-settlement market is remarkably well-calibrated** — slightly better than NWS, suggesting market participants effectively aggregate weather forecast information plus local knowledge.

3. **NWS/MOS forecasts are a strong, freely available baseline** — 2.23°F MAE with excellent calibration. Any model intended to beat prediction markets should first beat NWS.

4. **Both our model and NWS can profitably trade against the pre-settlement market**, but the edge is modest. NWS is the better trading signal, suggesting our model's value-add beyond public forecasts is currently negative.

5. **OOS results confirm IS findings** — no evidence of overfitting in the trading simulation. Both signals remain profitable out-of-sample.

## Recommendations

1. **Improve model sigma estimation** — the constant sigma is a major calibration weakness. Per-sample or seasonal sigma would help.
2. **Incorporate MOS as a feature** — the model should at minimum match NWS accuracy. Adding MOS forecasts as an input feature could close the gap.
3. **Focus trading on high-confidence divergences** — the 0.15-0.20 EV threshold range maximizes risk-adjusted returns.
4. **Re-evaluate after Phase 1 improvements** — the best model (combined features, 5-seed ensemble) with sub-2°F MAE may perform significantly better against these benchmarks.

## Files Produced

| File | Description |
|------|-------------|
| `scripts/fetch_kalshi_presettlement.py` | Kalshi pre-settlement data fetcher |
| `scripts/build_nws_benchmark.py` | NWS probability distribution benchmark |
| `scripts/test_model_vs_benchmarks.py` | Full comparison script |
| `data/kalshi_presettlement.csv` | Pre-settlement market data (4,611 rows, 769 dates) |
| `results/prediction_market_benchmark/` | All benchmark results and reports |
