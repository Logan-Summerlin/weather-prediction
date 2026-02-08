# Critical Review: Prediction Market Project Plan

**Reviewer role:** Computer scientist and prediction market trader
**Date:** 2026-02-08
**Verdict:** The plan has sound high-level instincts but contains critical strategic errors that, if followed as-is, will almost certainly result in **negative expected value** in live trading. The plan confuses building a good weather model with building a profitable trading system. These are related but distinct problems, and the plan fails on the second one.

---

## 1. The Fatal Flaw: You Are Trying to Reinvent Weather Forecasting from Surface Obs Alone

The single biggest problem with this plan is that it treats the existing model — a neural network predicting NYC TMAX from surrounding stations' t-1 observations — as the foundation for a trading system. **This is backwards.**

Your current best model achieves **MAE = 3.95°F**. The National Weather Service day-1 TMAX forecast for NYC typically achieves **MAE ~2.0-2.5°F**. The ECMWF deterministic forecast is even better. The GFS ensemble mean for NYC TMAX is roughly ~2.5°F MAE. Your model is roughly **60-100% worse than freely available forecasts**.

Prediction market participants — including professional meteorological firms, quant desks, and even informed hobbyists — have access to these forecasts. The market price already reflects NWP consensus. You are proposing to enter this market with a model that is dramatically worse than what every other participant is using as their starting point.

**The plan acknowledges this in Phase B ("Add Forecast Inputs") but treats it as an enhancement rather than the core strategy.** This is like building a stock trading system that ignores current prices and tries to predict them from historical fundamentals alone, then later adding "oh, and also look at the current price." The NWP forecast IS the starting point. Everything else is a correction to it.

### Recommendation
Invert the entire architecture. The system should be:
1. **Primary input:** NWP ensemble forecasts (GFS, ECMWF, HRRR, NAM)
2. **Secondary input:** NWS MOS/official forecast for NYC
3. **Tertiary inputs:** Station observations (for detecting when NWP is wrong)
4. **Model task:** Bias-correct and calibrate the NWP consensus, not predict temperature from scratch

---

## 2. No Competitive Baseline Against What the Market Actually Prices

The plan benchmarks against persistence (MAE=5.06°F), climatology (MAE=6.15°F), and ridge regression (MAE=4.33°F). These are useful for understanding the learning problem but **irrelevant for trading profitability**. No prediction market participant uses persistence as their forecast.

The plan never establishes:
- What the NWS official forecast achieves for NYC TMAX
- What the GFS/ECMWF ensemble mean achieves
- What a simple "follow the NWS forecast" strategy would yield in the market
- What the market-implied forecast (consensus midpoint) achieves versus the outcome

If you cannot beat the **NWS forecast as a probability distribution**, you have zero edge. If you cannot beat the **market-implied distribution**, you have negative EV after fees. The plan proposes no test of either condition.

### Recommendation
Before writing a single line of trading code, answer:
1. Download historical NWS/GFS forecasts for NYC TMAX for your test period (2022)
2. Compute their MAE, CRPS, and calibration
3. If your model cannot beat them, stop. Do not proceed to trading.
4. If historical Kalshi data is available, compute the market-implied forecast accuracy
5. Your edge is the gap between your calibration and the market's calibration, minus transaction costs

---

## 3. Calibration Is Mentioned but Not Operationalized

The plan lists "calibration (reliability diagram, PIT histogram)" as a metric. But for a prediction market, calibration IS the product. A poorly calibrated model that outputs probabilities cannot trade profitably regardless of its MAE.

The plan's approach to producing probabilities is:
- Quantile regression (5th/50th/95th percentiles)
- Distributional regression (mixture density or Gaussian NLL)
- Conformal prediction

These are listed as options with no analysis of which suits the trading problem. Critical issues:

**Quantile regression with 3 quantiles is grossly insufficient.** Kalshi KXHIGHNY markets use discrete temperature buckets (e.g., "≥80°F and <85°F"). You need P(a ≤ TMAX < b) for ~8-12 buckets spanning the plausible range. Three quantiles cannot reconstruct this. You need either:
- Many quantiles (e.g., every 1°F from 20°F to 105°F)
- A parametric distributional output (Gaussian, mixture of Gaussians, logistic)
- A non-parametric CDF approach (quantile function network)

**Conformal prediction gives marginal coverage guarantees, not conditional ones.** It guarantees that over all test days, 95% fall in the interval. It does NOT guarantee calibration on hot days, cold days, or any specific regime. For trading, you need conditional calibration — accurate probabilities for specific temperature ranges on specific types of days.

**No post-hoc calibration step.** Even well-trained probabilistic models are typically miscalibrated. The standard approach is to apply isotonic regression or Platt scaling to the model's raw probabilities using a held-out calibration set. The plan omits this entirely.

### Recommendation
1. Use a distributional output (Gaussian NLL at minimum; mixture density network preferred)
2. Train on CRPS or log-score, not MAE/MSE
3. Add a post-hoc calibration layer (isotonic regression on validation set)
4. Evaluate calibration PER BUCKET, not just globally
5. Require conditional calibration: by season, by forecast uncertainty, by temperature regime

---

## 4. The Market Microstructure Analysis Is Absent

The plan talks about "EV computation" and "Kelly sizing" as if the market were a simple coin flip. Prediction markets have microstructure that determines whether theoretical edge translates to realized profit.

**Missing analysis:**

- **Liquidity.** Kalshi weather markets are relatively thin. What is the typical orderbook depth at each temperature bucket? If you can only fill 5-10 contracts per bucket, your maximum daily profit is capped at a few dollars. No amount of model sophistication matters if you can't get filled.

- **Bid-ask spread.** On thin markets, the spread can be 5-15 cents on a $1 contract. That is a 5-15% round-trip cost. Your model needs to generate >5-15% probability divergence from the market AFTER accounting for its own miscalibration just to break even.

- **Market impact.** If your orders move the market, your realized fill price is worse than the quoted price. On thin books, placing even $20 of contracts can eat through multiple levels.

- **Adverse selection.** When you get filled, it may be because a more informed participant (e.g., a professional meteorologist) is taking the other side. The fills you get are disproportionately the ones where you're wrong.

- **Timing.** When should you trade? Markets for tomorrow's temperature open in the evening. Forecasts update overnight. The market is most efficient right before close. Trading too early means stale information; trading too late means thin liquidity.

### Recommendation
1. Conduct a liquidity analysis of KXHIGHNY: average depth, spread, daily volume per bucket
2. Compute the minimum probability divergence needed to overcome the spread
3. Model the expected P&L per trade accounting for fill rates and market impact
4. Define time-of-day trading windows based on data freshness vs. liquidity
5. Set a firm minimum bet threshold: do not trade when expected edge < 2× transaction cost

---

## 5. Kelly Criterion Without Honest Uncertainty Is Ruin

The plan proposes "fractional Kelly with caps." This sounds prudent but masks a fundamental problem: **Kelly assumes your probability estimates are correct.** If they're wrong, Kelly overbets systematically, and fractional Kelly only delays the damage.

Your model has 3.95°F MAE. Converting this to probability space: for a bucket like "≥80°F and <85°F," a 4°F error in the point forecast translates to massive probability errors. If the true temperature is 82°F and your model says 78°F, your model assigns near-zero probability to the 80-85 bucket when it should be ~80%+. Kelly would tell you to bet heavily on the wrong side.

**The plan proposes no mechanism for estimating how wrong the model's probabilities might be.** Without this, any Kelly-derived bet size is meaningless.

### Recommendation
1. Never use more than 1/4 Kelly. 1/8 Kelly is safer for a new model.
2. Implement a meta-model that estimates the reliability of each prediction (e.g., confidence in the confidence). Flag low-reliability days (high ensemble spread, transitional weather, rapid changes).
3. Set a maximum position size per day (e.g., $10-20 per market) until you have 100+ days of live P&L data confirming edge.
4. Track your calibration in real-time. If your model is miscalibrated over any 30-day window, halt trading and retrain.

---

## 6. Where Does Edge Actually Come From?

The plan never answers the most important question: **why would your model know something the market doesn't?**

In an efficient prediction market, the price already reflects:
- All publicly available NWP forecasts (GFS, ECMWF, NAM, HRRR)
- The NWS official forecast
- Historical base rates
- Current conditions

Your model uses **none of these** except historical observations. It's fighting with one hand tied behind its back.

Realistic sources of edge in weather prediction markets:

1. **Better calibration than the consensus.** NWP models have known systematic biases (warm bias in winter, cold bias in summer for NYC). If you can correct these better than the market consensus does, you have edge.

2. **Faster processing.** If you ingest the latest model run (e.g., 06Z GFS) and update your forecast before the market adjusts, you have a brief edge. This requires low-latency infrastructure.

3. **Tail events.** Markets tend to underprice extreme outcomes (behavioral bias toward the base rate). If your model correctly identifies that tomorrow has a 15% chance of hitting 95°F when the market prices it at 5%, that single trade can generate outsized returns. This requires excellent tail calibration, which is the hardest calibration problem.

4. **Regime transitions.** Markets are often slow to adjust when the weather pattern changes (e.g., a sudden cold front not well-captured by the previous NWP run). Station observations can detect this faster than the market adjusts.

5. **Market microstructure.** On thin markets, prices can be stale or distorted by a single large participant. You can sometimes pick off mispriced contracts simply by monitoring the orderbook.

**The current model architecture addresses none of these.** The plan's Phase C (residual learning on NWP) is closest to #1, but it's buried as a later step rather than the core strategy.

### Recommendation
Choose 1-2 edge sources and design the entire system around them. My ranking:
1. **NWP bias correction + calibration** (highest expected edge, requires NWP data pipeline)
2. **Tail event detection** (highest per-trade profit, requires excellent extreme-event model)
3. **Regime transition detection** (this is where your station observation network adds unique value)

---

## 7. The Execution Sequence Is Wrong

The plan proposes: Phase A (baseline from historical) → Phase B (add forecasts) → Phase C (synthesis) → Phase D (trading simulator) → Phase E (operationalize).

This sequence front-loads the least valuable work and defers the most critical.

**Problems:**
- Phase A builds a "minimal probability model from historical data" — this is the 3.95°F MAE model. It has no trading value. Building it first wastes time and creates anchoring bias.
- Phase B adds NWP forecasts, which should be step 1, not step 2.
- Phase D (trading simulator) comes after model building, but you need the trading simulator FIRST to establish your profitability requirements (minimum edge, position sizing constraints, break-even accuracy).
- Phase E (operationalization) is vaguely defined. In practice, this is the hardest part.

### Recommended Sequence
1. **Market analysis first:** Study KXHIGHNY contract structure, liquidity, spreads, historical pricing. Determine if the market is even tradeable given its microstructure.
2. **Competitive baseline:** Download NWS/GFS/ECMWF forecasts for NYC TMAX. Evaluate their accuracy. Establish the bar to beat.
3. **Trading simulator:** Build the EV/Kelly/backtest framework. Determine the minimum probability edge needed to profit after fees.
4. **NWP bias-correction model:** Build a model that corrects NWP forecasts using station observations, recent errors, and regime indicators. This is your alpha.
5. **Calibration layer:** Ensure the corrected forecast produces well-calibrated probabilities for each Kalshi bucket.
6. **Paper trading:** Run the system for 30-60 days without real money. Track P&L, calibration, and edge.
7. **Live trading:** Start with minimal position sizes. Scale only after confirming edge.

---

## 8. Missing: NWP Data Pipeline (The Actual Hard Part)

The plan hand-waves about "GFS/GEFS, ECMWF (if accessible), HRRR" without acknowledging that building a reliable NWP data ingestion pipeline is a major engineering effort.

- **GFS:** Free, available via NOAA NOMADS. ~4 runs/day, ~3-4 hour delay. You need to parse GRIB2 files and extract the NYC grid point.
- **ECMWF:** The high-resolution deterministic model is behind a paywall for real-time access (ECMWF requires licensing). The open data initiative provides some products with delay.
- **HRRR:** High-resolution, hourly updates. Free but massive data volume. Extremely valuable for same-day forecasts.
- **MOS:** Statistical post-processing of NWP. Available from NWS. Arguably the single best operational forecast for specific stations.

Building a pipeline that reliably ingests, parses, and stores these data products every day is non-trivial. The plan allocates zero effort to this.

### Recommendation
1. Start with MOS and GFS ensemble (both freely available, well-documented formats)
2. Build a daily cron job that downloads the latest 00Z and 12Z runs
3. Store the NYC TMAX forecast, ensemble spread, and key upper-air variables
4. Only add ECMWF/HRRR if the basic system shows promise

---

## 9. The Existing Codebase Is a Liability, Not an Asset, for Trading

The current codebase (Phase 1-4) was designed for a different problem: understanding how well station observations predict NYC temperature. It has:
- Excellent test coverage (652 tests)
- Clean data pipeline for historical GHCN data
- Good baseline comparisons
- Solid training infrastructure

But for trading, most of this is irrelevant or needs fundamental restructuring:
- The entire feature engineering pipeline assumes only station observations. Adding NWP features requires a new preprocessing module.
- The training loop optimizes for MAE/MSE, not CRPS/log-score. This must change.
- The evaluation framework has no probabilistic scoring (no Brier score, no CRPS, no calibration plots).
- There is no mechanism to produce a full predictive CDF, only point predictions.
- There is no operational (daily) pipeline — everything runs in batch on historical data.

**What IS salvageable:**
- The station observation pipeline (useful for regime detection and NWP bias correction)
- The delta-T target formulation (good for NWP residual modeling)
- The general training infrastructure (train/val/test split logic, early stopping)
- The station attention model architecture (potentially useful for learning which stations provide correction signal)

### Recommendation
Do not attempt to bolt the trading system onto the existing codebase. Instead:
1. Build a new `src/forecasting/` module for the NWP-based system
2. Use the existing station pipeline as a supplementary input
3. Build a new `src/trading/` module for the market interface
4. Reuse the existing infrastructure where it fits, but don't let it dictate architecture

---

## 10. Specific Technical Gaps

### 10.1 No wind data
The model uses only temperatures. Wind direction is the single most important variable for temperature advection — it tells you WHICH station to weight highest. Without wind, the model must learn a static average weighting that is wrong ~50% of the time. The GHCN dataset doesn't include wind, but ASOS/METAR stations do, and this data is freely available.

### 10.2 No upper-air data
Surface temperatures are a lagging indicator. The 850mb temperature (available from GFS within hours) is a leading indicator that directly predicts surface TMAX with high skill. Ignoring upper-air data is leaving the most valuable signal on the table.

### 10.3 No morning observation updates
For a prediction market that closes in the afternoon, the morning's TMIN and current temperature reading are enormously valuable. The current model uses only t-1 data. A real trading system should update its forecast intraday as new obs arrive.

### 10.4 No cloud/radiation model
On clear days, TMAX is primarily determined by solar heating and advection. On cloudy days, solar heating is suppressed. Cloud cover (available from NWP and satellite) explains a large fraction of the residual error that station temperatures alone cannot capture.

### 10.5 No snow cover data
Snow cover dramatically alters the surface energy budget. A model trained on 5 years of data has very few snow events. This is a tail-event risk that the plan ignores.

---

## 11. The Training Data Problem

With 1,277 training samples and a current best of 3.95°F MAE, more data is the clearest path to improvement. The plan acknowledges this (Phase 6: scale to 25 years). But for trading:

- **NWP forecast archives are the rate-limiting data.** Historical GFS reanalysis (CFSR/CFSv2) goes back to 1979 but at coarser resolution than operational forecasts. Historical MOS data is harder to obtain.
- **Climate non-stationarity.** A model trained on 1985-2024 data may not reflect the current climate. NYC has warmed ~2°F since 1985. The most recent data is most relevant for trading.
- **Market data is recent.** Kalshi weather markets launched around 2021-2022. You have at most ~4 years of market data for backtesting. This is barely enough for statistical significance.

### Recommendation
1. Scale the station data to 25 years as planned (Phase 6)
2. Prioritize NWP archive access: GFS reforecast dataset, MOS archive
3. Use a rolling/weighted training window that emphasizes recent years
4. Accept that backtesting against market prices will have limited statistical power

---

## 12. What a Profitable System Actually Looks Like

Based on the analysis above, here is what a realistic profitable weather prediction market system looks like:

```
Daily Pipeline (runs at ~5 AM ET):
  1. Ingest latest GFS 00Z, NAM 00Z, HRRR 00Z runs
  2. Extract NYC-area 2m TMAX forecast, 850mb temp, wind, clouds
  3. Ingest latest MOS TMAX forecast for Central Park
  4. Pull yesterday's station observations (GHCN quick data or ASOS)
  5. Compute NWP ensemble mean, spread, and recent bias
  6. Feed to bias-correction model → get corrected TMAX distribution
  7. Apply calibration layer → get calibrated bucket probabilities
  8. Pull current Kalshi KXHIGHNY orderbooks
  9. Compute EV for each bucket; flag any bucket where |edge| > threshold
  10. If edge exists and liquidity sufficient → execute trade
  11. Log everything for monitoring

Morning Update (runs at ~10 AM ET):
  1. Ingest HRRR 06Z/09Z, morning ASOS obs (current temp, cloud cover)
  2. Update forecast distribution
  3. Check if new edge has appeared or previous edge has evaporated
  4. Adjust positions if needed

Model Architecture:
  - Input: [MOS_forecast, GFS_mean, GFS_spread, HRRR_TMAX,
            sector_obs_gradients, obs_trend_features,
            recent_model_bias, sin_day, cos_day]
  - Output: Gaussian mixture (mu1, sigma1, mu2, sigma2, pi)
  - Loss: CRPS or negative log-likelihood
  - Post-hoc: Isotonic regression calibration per bucket

Trading Logic:
  - Only trade when calibrated edge > 2× spread
  - Position size: min(1/4 Kelly, $20 per bucket, orderbook depth × 50%)
  - Maximum daily exposure: $100
  - Halt if 30-day calibration degrades below threshold
```

This is substantially different from what the plan proposes. The key differences:
- NWP is the primary input, not station observations
- The model learns to correct NWP, not predict temperature from scratch
- Calibration is a separate, explicit layer
- Trading logic is integrated from the start, not bolted on later
- Position sizing is conservative by default
- There are circuit breakers for when the model fails

---

## 13. Summary Scorecard

| Plan Element | Rating | Reasoning |
|---|---|---|
| High-level intent (EV-first, probabilistic) | B+ | Correct instincts, just needs follow-through |
| Market definition/alignment (Sec 2) | A- | Good; should also cover bucket structure |
| Data sources (Sec 3) | D | Station obs are tertiary; NWP should be primary |
| Feature engineering (Sec 4) | C- | Station features are solid but insufficient alone |
| Model outputs (Sec 5) | C | Right idea (distribution), wrong implementation (3 quantiles) |
| Training/evaluation protocol (Sec 6) | C+ | Missing CRPS, conditional calibration, NWP baseline |
| Trading layer (Sec 7) | D+ | Superficial; missing microstructure, liquidity, timing |
| Risk controls (Sec 8) | C | Decent checklist but no quantitative thresholds |
| Execution sequence (Sec 9) | D | Completely inverted; market analysis should be first |
| Winning definition (Sec 10) | B | Correct criteria, but bar is set too low |

**Overall: C-** — The plan identifies the right questions but answers most of them incorrectly or incompletely. Following it as-is would produce a system that loses money.

---

## 14. Top 5 Changes for Maximum Profitability

Ranked by expected impact on P&L:

1. **Make NWP the primary input.** Build an NWP ingestion pipeline (GFS + MOS at minimum). Use the station observation network for bias correction, not primary prediction. This alone would roughly halve the MAE and bring the model into the competitive range.

2. **Build the trading simulator first.** Before improving the model, quantify how much edge you need. Pull Kalshi orderbooks for 30 days, compute spreads and depths, and calculate the minimum probability divergence needed for positive EV. If the market is too thin or the spreads too wide, save yourself the effort.

3. **Replace quantile regression with a full distributional output.** Train a model that outputs a parametric distribution (Gaussian mixture with 2-3 components). Evaluate using CRPS and conditional calibration per bucket. Apply isotonic regression post-hoc.

4. **Focus on tails.** The largest edge in weather markets is on extreme events. Build a separate extreme-event detector (e.g., P(TMAX ≥ 90°F) or P(TMAX ≤ 25°F)) trained specifically on those regimes. The market typically underprices tails.

5. **Add wind and upper-air data.** Even if you don't use NWP directly, adding surface wind direction/speed from nearby ASOS stations would dramatically improve the model's ability to detect advection events. The 850mb temperature from GFS is the single most predictive variable for surface TMAX that you're currently ignoring.

---

*This review is intentionally harsh. The underlying project — using station observations to predict NYC temperature — is well-executed and scientifically sound. But a prediction market is not a science project. It is a competition against other forecasters where the bar is set by NWP models that cost hundreds of millions of dollars to develop. Winning requires either correcting those models' errors or exploiting market inefficiencies. The current plan does neither.*
