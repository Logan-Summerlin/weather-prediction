# NYC Daily Max Temperature Prediction Market Plan

## 1) Goal and Success Criteria (Trading-First)

**Primary objective:** maximize **expected trading value (EV)** in a prediction market on NYC’s daily maximum temperature (TMAX), not just minimize point error.

**Core success metrics:**
- **EV vs. market odds** (simulated backtest with slippage).
- **Proper probabilistic scores** (log score, CRPS, Brier on thresholds).
- **Calibration** (reliability diagram, PIT histogram), with special attention to **tail events**.
- **Regime robustness** (performance on high-gradient and rapid-change days).

**Secondary metrics (still tracked):** MAE, RMSE, bias, R².

**Decision rule principle:** only bet when **model-implied probability** meaningfully diverges from the market-implied probability by a risk-adjusted threshold.

---

## 2) Market Definition and Alignment (Non-Negotiable)

**Why this comes first:** Even a strong model fails if the market’s “daily high” definition is misaligned.

- Confirm **market contract definition** of “daily maximum temperature” (station, time zone, cutoff time).
- Use **Central Park (USW00094728)** as the canonical target unless the market specifies another station.
- Normalize all timestamps to NYC local time.
- Record any rounding, reporting, or official NOAA/NWS finalization rules.

Deliverable:
- A short “contract spec” doc (station, time window, rounding, source).

---

## 3) Data Sources (Forecast-Forward + Observational)

**3.1 Observational (historical ground truth)**
- NOAA GHCN-Daily station data for NYC + surrounding stations.
- This is still the **training truth** for model evaluation.

**3.2 Forecast-forward signals (for edge)**
- **Numerical weather prediction (NWP):** GFS/GEFS, ECMWF (if accessible), HRRR.
- **MOS / NWS forecast products** for NYC TMAX.
- **Ensemble spreads** (uncertainty proxy).
- Optional: **nowcasting signals** (morning updates, cloud cover, wind direction).

**3.3 Market data**
- Historical market prices and volumes for NYC TMAX contracts.
- Compute market-implied probabilities.

Deliverables:
- A unified dataset joining **observations + forecasts + market prices**.

---

## 4) Feature Engineering (Synthesis, Not Copying)

**Base physics/observation features:**
- Lagged station TMAX/TMIN, diurnal range.
- Sector gradients (WNW vs. coastal, SW vs. NW).
- Trend features (Δ1, Δ2) for front timing.
- Cyclical day-of-year encodings.

**Forecast features:**
- Forecasted TMAX for NYC (from multiple models).
- Forecasted 2m temperature, 850mb temperature, cloud cover, wind.
- Ensemble mean and spread.

**Market features (optional for execution):**
- Market-implied probability and its change over time.

Synthesis principle:
- Forecasts are **inputs**, not a final target. The model learns when to trust each signal.

---

## 5) Model Outputs (Probability-First)

**Primary outputs:** predictive distribution of NYC TMAX.

Recommended approaches:
- **Quantile regression** (e.g., 5th/50th/95th percentiles).
- **Distributional regression** (mixture density or Gaussian NLL).
- **Conformal prediction** for calibrated intervals.

**Point forecasts** are still produced, but trading uses **probabilities**.

---

## 6) Training and Evaluation Protocol

**Splits:** strict chronological train/val/test.

**Metrics:**
- Log score / CRPS (probability quality)
- Brier score on threshold events (e.g., TMAX ≥ 90°F)
- MAE/RMSE for baseline comparison
- Calibration plots (reliability, PIT)
- Tail/regime slices (top 5% hottest days, high-gradient days)

**Model comparison:**
- Against persistence, climatology, ridge.
- Against each external forecast source alone.
- Against market-implied probabilities alone.

Demonstrate synthesis value by showing:
- Higher log score / lower CRPS than any single source.
- Better calibration in tails and regime shifts.

---

## 7) Trading Layer (From Forecasts to Bets)

**7.1 Market probability alignment**
- Convert model distribution to P(TMAX ≥ threshold) or contract-specific probabilities.

**7.2 EV computation**
- EV = model_prob × payout − (1 − model_prob) × cost

**7.3 Bet sizing**
- Fractional Kelly with caps.
- Avoid correlated risk across adjacent thresholds.

**7.4 Backtesting**
- Include transaction costs, slippage, liquidity constraints.
- Evaluate Sharpe-like metrics on EV series.

Deliverable:
- Backtest report with “bet vs. no bet” thresholds.

---

## 8) Risk and Robustness Controls

- **Bias monitoring:** warm vs. cold bias by season.
- **Drift checks:** recency-weighted training and rolling retrain.
- **Outlier handling:** extreme heat days scored separately.
- **Adversarial case review:** days with largest losses.

---

## 9) Execution Sequence (Trading-Optimized)

**Phase A: Market Alignment + Baseline**
- Lock the contract spec.
- Build minimal probability model from historical data.
- Compare to market prices.

**Phase B: Add Forecast Inputs**
- Integrate NWP/MOS features.
- Train probabilistic model; verify calibration improvements.

**Phase C: Synthesis + Residual Learning**
- Stacking: let external forecasts handle the “easy part,” model corrects residuals.
- Confirm EV improvement vs. single-source forecasts.

**Phase D: Trading Simulator**
- Build bet sizing and EV backtest.
- Tune decision thresholds.

**Phase E: Operationalization**
- Daily ingest → forecast distribution → EV vs market → bet decision.

---

## 10) Definition of “Winning”

A model “wins” if, over out-of-sample evaluation:
- It is **better calibrated** than the market-implied probabilities.
- It delivers **positive EV** after realistic slippage.
- It performs well on **tail days** (where edge is largest).

This is stricter than MAE, but directly aligned with profit.
