# High-Level Modeling Principles + Multi-City Portability Guide

**Last updated:** 2026-02-15

## A. Core principles used in the current stack

1. **Contract-definition-first modeling**
   - Target definition, day boundary, bucket thresholds, and inclusivity rules must match settlement logic exactly before any model training.

2. **Time-safe operational parity**
   - Live features must be available by the decision cutoff.
   - Training pipelines must mimic live feature construction to minimize train/inference mismatch.

3. **Distribution-first forecast output**
   - The stack outputs calibrated probabilities per bucket, not just point temperature.
   - Bucket probabilities are created from mu/sigma CDF mass or directly from contract-level classifiers.

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

## C. Multi-city expansion recipe

### Step 1: Contract + observation alignment (hard prerequisite)
- Identify settlement station/site and official measurement convention.
- Encode timezone/day roll and bucket boundaries exactly.
- Confirm contract ticker and bucket structure on Kalshi.

### Step 2: City-specific data source mapping
- **Operational by cutoff:** local ASOS stations, NWS forecast point, Kalshi market snapshots.
- **Training-only:** GHCN-Daily archives, reanalysis for historical fit and diagnostics.

### Step 3: Station network design
- Use `src/station_discovery.py` to build a target-centered station registry.
- Classify by distance rings and compass sectors (same schema as NYC `config_expanded.py`).
- Create city-specific config file (e.g., `config_chicago.py`, `config_philadelphia.py`).
- Validate completeness (>= 80% TMAX coverage over study period).

### Step 4: Time-safe feature pipeline
- Recreate the same transformations in training and live paths.
- Persist scalers/feature schema and block post-cutoff columns.
- Reuse `src/operational_features.py` with city-specific station metadata.

### Step 5: Baseline + probabilistic model build
- Start with simple distributional baseline (flat model, isotonic calibration).
- Port flat feedforward model (`src/model.py`) with city-specific inputs.
- Add WGA/synthesis only if OOS Brier + reliability improve over baseline.

### Step 6: Unified synthesis and calibration sweep
- Add cross-model stackers and contract-level Brier-MLP variants.
- Compare calibration windows and regime-conditioned features.
- Leverage NYC model architecture decisions as strong priors.

### Step 7: Backtest with conservative execution assumptions
- Include fees, spread/slippage, and realistic fill constraints.
- Evaluate per-regime, per-season, and OOS-only slices.

### Step 8: Paper-trade promotion gates
- Require pass on reliability + edge quality + drawdown constraints before scaling.
- Keep kill-switch rules for missing data/schema/calibration drift.

## D. City-specific adaptation notes

### Chicago (KXHIGHCHI)
- **Settlement station:** O'Hare International (USW00094846)
- **Climate regime:** Continental, high seasonal variance, lake-effect modulation
- **Key considerations:**
  - Strong cold-air outbreaks from NW (Arctic intrusions) create tail events
  - Lake Michigan moderates near-shore temperatures — station network must capture lake vs inland gradient
  - Larger sigma (temperature variance) than NYC, especially winter — wider bucket uncertainty
  - Wind direction is critical: onshore (NE/E) vs offshore (W/SW) drives 5-10F swings
  - IGRA station: Davenport (DVN) or Lincoln (ILX) for upper-air context
  - NWS forecast office: LOT (Chicago)

### Philadelphia (KXHIGHPHL)
- **Settlement station:** PHL International (USW00013739)
- **Climate regime:** Mid-Atlantic transitional, moderate marine influence
- **Key considerations:**
  - Already in NYC's station network (Ring 2, 91 mi SW) — significant shared signal
  - Delaware Valley urban heat island effect
  - Summer convective uncertainty (thunderstorm events) adds tail risk
  - Strong correlation with NYC — cross-city model sharing opportunities
  - IGRA station: Sterling VA (IAD) or Upton (OKX, shared with NYC)
  - NWS forecast office: PHI (Mt Holly)

### General coastal cities (BOS, SFO)
- Emphasize wind-direction, marine layer/cloud proxies.

### Continental extremes (DEN, MSP)
- Stronger regime/season-conditioned sigma modeling.

### Convective regimes (ATL, MIA)
- Wider uncertainty treatment and stricter EV thresholds.

### Sparse station regions
- Stronger regularization + robust fallback to broader regional aggregates.

## E. Invariants that must not change across cities

- Chronological evaluation only.
- Calibrated probabilities before trading decisions.
- Strict train/inference parity and cutoff-time safety.
- Cost-aware EV and risk-limited sizing.
- Complete logging/audit artifacts for every daily run.
- Contract bucket definitions must exactly match Kalshi settlement spec.
