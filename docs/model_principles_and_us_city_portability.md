# High-Level Modeling Principles + How to Port This Stack to Any U.S. City

## A. Core principles used in the current stack

1. **Contract-definition-first modeling**
   - Target definition, day boundary, bucket thresholds, and inclusivity rules must match settlement logic exactly before any model training.

2. **Time-safe operational parity**
   - Live features must be available by the decision cutoff.
   - Training pipelines must mimic live feature construction to minimize train/inference mismatch.

3. **Distribution-first forecast output**
   - The stack is built to output calibrated probabilities per bucket, not just point temperature.
   - Bucket probabilities are created from `mu/sigma` CDF mass or directly from contract-level classifiers.

4. **Calibration as a first-class layer**
   - Isotonic, Platt+isotonic, and regime-aware recalibration are deeply integrated.
   - Model promotion is tied to reliability/ECE and OOS Brier, not raw fit metrics alone.

5. **Model-combination over single-source dependence**
   - Best variants combine multiple signals: flat model, WGA model, NWS, and market-state context.
   - Cross-model disagreement features improve robustness in regime shifts.

6. **Trading discipline > score chasing**
   - EV-gated deployment, fee/slippage awareness, exposure controls, and promotion gates are required.
   - No assumption that a Brier win automatically translates to tradable edge.

## B. Architecture principles by layer

### 1) Ingestion layer
- Keep operational and training-only data sources separated.
- Maintain schema/version checks and source provenance tags for each feature group.

### 2) Feature layer
- Build physically plausible, low-leakage features:
  - persistence/trend,
  - bucket geometry (quantile, width, distance in sigma-space),
  - market-state diagnostics (spread, depth, staleness),
  - regime indicators (volatility/uncertainty normalization).

### 3) Modeling layer
- Maintain families with complementary biases:
  - E-series calibrated synthesis,
  - WGA V2 attention-driven spatial synthesis,
  - Unified U-series cross-model stackers.
- Prefer compact models unless complexity gives repeatable OOS gains.

### 4) Calibration + bucketization layer
- Enforce monotonicity and clipping safeguards.
- Validate daily probability mass and reliability before downstream trading steps.

### 5) Trading simulation/execution layer
- Use cost-adjusted EV thresholds.
- Apply gating by quality diagnostics (OOS Brier, ECE, seasonal stress, slippage sensitivity).

## C. How to build this for any U.S. city (implementation recipe)

1. **Contract + observation alignment (hard prerequisite)**
   - Identify settlement station/site and official measurement convention.
   - Encode timezone/day roll and bucket boundaries exactly.

2. **City-specific data source mapping**
   - Operational by cutoff: local station observations, forecast proxies, market snapshots.
   - Training-only: archives/reanalysis for historical fit and diagnostics.

3. **Station network design**
   - Build a target-centered station registry with directional sectors and availability scores.
   - Add fallback logic for sparse/missing stations.

4. **Time-safe feature pipeline**
   - Recreate the same transformations in training and live paths.
   - Persist scalers/feature schema and block post-cutoff columns.

5. **Baseline + probabilistic model build**
   - Start with simple distributional baseline (flat model, isotonic calibration).
   - Add WGA/synthesis only if OOS Brier + reliability improve over baseline.

6. **Unified synthesis and calibration sweep**
   - Add cross-model stackers and contract-level Brier-MLP variants.
   - Compare calibration windows (e.g., single year vs multi-year) and regime-conditioned features.

7. **Backtest with conservative execution assumptions**
   - Include fees, spread/slippage, and realistic fill constraints.
   - Evaluate per-regime, per-season, and OOS-only slices.

8. **Paper-trade promotion gates**
   - Require pass on reliability + edge quality + drawdown constraints before scaling.
   - Keep kill-switch rules for missing data/schema/calibration drift.

## D. City-specific adaptation guidelines

- **Coastal cities (e.g., BOS, SFO):** emphasize wind-direction, marine layer/cloud proxies.
- **Continental extremes (e.g., DEN, MSP):** stronger regime/season-conditioned sigma modeling.
- **Convective regimes (e.g., ATL, MIA):** wider uncertainty treatment and stricter EV thresholds.
- **Sparse station regions:** stronger regularization + robust fallback to broader regional aggregates.

## E. Invariants that should not change across cities

- Chronological evaluation only.
- Calibrated probabilities before trading decisions.
- Strict train/inference parity and cutoff-time safety.
- Cost-aware EV and risk-limited sizing.
- Complete logging/audit artifacts for every daily run.
