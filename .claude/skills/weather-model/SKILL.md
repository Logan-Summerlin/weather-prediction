---
name: weather-model
description: Build and productionize a new city weather probabilistic forecasting pipeline aligned to market contracts.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Weather Model Skill (New City Rollout)

Use this skill when adding a new city model to the repo and integrating it with the existing forecasting + trading pipeline.

## Objective
Ship a contract-aligned, calibrated distribution model for a new city, then connect it to bucketization, EV logic, backtesting, and promotion gates.

## 1) Define contract alignment first
- Add/verify city config in `src/city_config.py` (station, timezone/day boundary, bucket thresholds, semantics).
- Confirm target definition exactly matches settlement logic.

## 2) Create pipeline entrypoints under `scripts/`
Implement or maintain:
- `run_<city>_data_collection.py`
- `run_<city>_preprocessing.py`
- `run_<city>_benchmark.py`
- `run_<city>_synthesis_calibration.py`
- `run_<city>_backtest.py`
- `run_<city>_promotion_evaluation.py`

These should reuse shared modules and follow the same stage order as existing city pipelines.

## 3) Enforce data and artifact conventions
- Raw station files: `data/<city>/raw/<station_id>.csv`
- Processed splits: `data/<city>/processed/features_{train,val,test}.csv`
- Targets: `data/<city>/processed/target_{train,val,test}.csv`
- Model outputs: `models/<city>/...`
- Reports/results: `results/<city>/...`

## 4) Data source requirements: ASOS over GHCN for training
- **Use ASOS (IEM hourly) data as the primary training data source**, not GHCN-Daily.
- ASOS data matches the operational inference data source, ensuring training/inference parity.
- GHCN-Daily TMAX can differ from ASOS-derived TMAX by 1–3°F due to observation time conventions, sensor siting, and temporal aggregation — training on GHCN creates a systematic bias in live performance.
- Collect full ASOS history for the city's station network using `src/asos_collection.py`.
- Aggregate to daily features using `src/asos_preprocessing.py` (TMAX, TMIN, dewpoint, wind, pressure, clouds).
- Run ASOS vs GHCN cross-validation on the overlap period (`compare_asos_ghcn_tmax()`) to document any station-specific offsets.
- GHCN data may still be used as a secondary validation source but must not be the primary training input.

## 5) Modeling + calibration requirements
- Use chronological splits only; no random shuffle.
- Fit preprocessors/scalers on train partition only.
- Train distributional models (not point-only) and benchmark against persistence/climatology/linear baselines.
- Calibrate probabilities on held-out calibration data before bucketization.
- Convert calibrated CDF to contract bucket probabilities with exact boundary semantics.

## 6) Trading/backtest integration
- Compute EV vs market-implied probabilities including fees/slippage.
- Apply risk controls (sizing caps, exposure limits, kill switch conditions).
- Backtests must be time-faithful and include realistic execution assumptions.

## 7) Promotion gate (must pass)
- Contract alignment verified.
- Model trained on ASOS-derived features (not GHCN-only) with verified training/inference parity.
- Calibration diagnostics acceptable out-of-sample.
- Contract-level probabilistic performance beats required baselines.
- Positive EV after conservative costs.
- Risk limits and halts validated.

## Canonical command sequence
```bash
python scripts/run_<city>_data_collection.py
python scripts/run_<city>_preprocessing.py
python scripts/run_<city>_benchmark.py
python scripts/run_city_nws_kalshi_template_benchmark.py --city <city>
python scripts/run_<city>_synthesis_calibration.py
python scripts/run_<city>_backtest.py
python scripts/run_<city>_promotion_evaluation.py
```
