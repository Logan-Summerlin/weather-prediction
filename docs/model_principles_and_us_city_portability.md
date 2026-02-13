# High-Level Modeling Principles and How to Rebuild This Stack for Any U.S. City

## 1) Core principles used in this project

1. **Contract-first target definition**
   - Align target variable to settlement measurement (station, day boundary, bucket rules, inclusivity).

2. **Time-safe operational design**
   - Use only inputs available before the daily decision cutoff in live mode.
   - Keep training-only archives isolated from operational feature generation.

3. **Distribution-first forecasting**
   - Predict uncertainty-aware distributions (`mu`, `sigma`, or richer forms), not just point values.
   - Optimize proper probabilistic scoring (Brier/CRPS/NLL), with MAE as secondary.

4. **Calibration is mandatory**
   - Apply post-hoc calibration (isotonic, conditional, Platt+isotonic) before trading.
   - Validate mass conservation and reliability.

5. **Market-aware synthesis beats single-source forecasts**
   - Blend model, NWS/MOS, and market state features when out-of-sample evidence supports it.

6. **Trading must be cost-aware and risk-limited**
   - Compute EV net of fees, spread/slippage assumptions.
   - Use capped/fractional sizing with hard halts.

## 2) Portable city-agnostic architecture

### Layer A — City contract alignment
- Select target station used by the contract (e.g., airport/site ID).
- Encode exact local day boundary and bucket thresholds.

### Layer B — Data sources
- Operational: local+regional station obs, MOS/NWP previews, market snapshots.
- Training-only: long archives/reanalysis for robust historical fit and diagnostics.

### Layer C — Feature template
- Persistence and lag deltas of target and neighbor stations.
- Seasonality (harmonics), frontal/regime volatility proxies.
- Optional market-state features (spread, depth, staleness) for synthesis layer.

### Layer D — Modeling template
- Start with baseline distributional model.
- Add calibration variants.
- Add synthesis models only if they improve OOS calibration + Brier.

### Layer E — Trading template
- Bucketize calibrated CDF.
- Compute EV vs market-implied probabilities.
- Gate trades by edge, liquidity, and confidence.

## 3) Step-by-step migration to a new U.S. city

1. **Define contract spec:** settle station, timezone/day cutoff, bucket endpoints.
2. **Build station registry:** target + surrounding stations with completeness thresholds.
3. **Implement time-safe ingestion:** only sources available pre-cutoff in live mode.
4. **Recreate feature pipeline:** same transforms used in training and inference.
5. **Train baseline distributional model:** chronological split with holdout and calibration windows.
6. **Calibrate + reliability check:** PIT/reliability/interval coverage by season/regime.
7. **Run benchmark suite:** compare against NWS/MOS and pre-settlement market.
8. **Simulate trading conservatively:** fees, spread, slippage, fill uncertainty.
9. **Paper trade:** require stable edge and bounded drawdowns before scaling.

## 4) City-specific challenges and mitigations

- **Sparse station network (Mountain West/rural):** use stronger regularization and broader spatial composites.
- **Coastal regimes (marine influence):** include wind-direction and humidity/cloud proxies.
- **Extreme seasonality (Upper Midwest):** use season-specific calibration, regime-conditioned sigma.
- **Frequent convective volatility (Southeast):** favor wider uncertainty modeling + stronger gating thresholds.

## 5) What should remain invariant across cities

- Chronological validation only.
- Strict train/inference parity.
- Calibration before any trading.
- EV net of costs and risk-managed sizing.
- Full logging/audit artifacts for every decision day.
