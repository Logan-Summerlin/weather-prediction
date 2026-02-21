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

## 4) Modeling + calibration requirements
- Use chronological splits only; no random shuffle.
- Fit preprocessors/scalers on train partition only.
- Train distributional models (not point-only) and benchmark against persistence/climatology/linear baselines.
- Calibrate probabilities on held-out calibration data before bucketization.
- Convert calibrated CDF to contract bucket probabilities with exact boundary semantics.

## 5) Trading/backtest integration
- Compute EV vs market-implied probabilities including fees/slippage.
- Apply risk controls (sizing caps, exposure limits, kill switch conditions).
- Backtests must be time-faithful and include realistic execution assumptions.

## 6) Promotion gate (must pass)
- Contract alignment verified.
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
