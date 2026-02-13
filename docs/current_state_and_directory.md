# Current Codebase State and Directory Guide

## Executive State Summary
This repository currently operates as a **probabilistic weather-to-market pipeline** with five active modules:
1. Data ingestion and preprocessing (station + operational feeds)
2. Feature generation (time-safe weather/regime features)
3. Distributional forecasting (baseline, NN, WGA, synthesis)
4. Calibration + bucketization into Kalshi contracts
5. EV-gated trading simulation and benchmark analytics

The active benchmark center of gravity is the E0–E22 pipeline plus unified outperformance experiments.

## High-Level Directory Map

```text
weather-prediction/
├── src/                         # Reusable core modules
├── scripts/                     # Experiment and benchmark runners
├── tests/                       # Unit/integration tests
├── data/                        # Input datasets and model prediction artifacts
├── results/                     # Benchmark outputs, model artifacts, diagnostics
├── reports/                     # Narrative reports and strategy write-ups
├── docs/                        # Current project documentation (new)
├── ARCHIVE/                     # Explicitly archived legacy code
├── .claude/rules/MEMORY.md      # Project memory for agent workflow
└── nyc_temp_prediction_project_plan.md
```

## `src/` module overview
- `data_collection.py`, `asos_collection.py`, `nwp_collection.py`, `soundings_collection.py`: ingestion paths for weather inputs.
- `data_preprocessing.py`, `asos_preprocessing.py`, `nwp_preprocessing.py`, `soundings_preprocessing.py`: deterministic transforms from raw to model-ready inputs.
- `model.py`, `wind_gated_attention.py`, `synthesis_model.py`: model families for temperature distribution and meta-synthesis.
- `calibration.py`, `evaluate.py`, `baselines.py`, `crps_loss.py`: probabilistic training/evaluation tooling.
- `kalshi_client.py`, `kalshi_backtester.py`, `trading.py`: market mapping, contract scoring, and EV/risk simulation.
- `operational_features.py`, `station_registry.py`: station metadata and production-safe feature orchestration.

## `scripts/` usage pattern
- **Primary benchmark runner:** `run_e0_e8_best_model_benchmark.py` (E0–E22 variants).
- **Unified synthesis benchmark:** `run_unified_outperformance_benchmark.py`.
- **WGA model development:** `train_wga_mdn.py`, `train_wga_v2.py`, and associated benchmark scripts.
- **Data prep/utility scripts:** MOS download/validation, feature engineering, retraining, and diagnostic generation.

## `results/` interpretation
- `results/prediction_market_benchmark/` is the most decision-relevant output tree.
- Contains per-variant Brier/log metrics, calibration tables, EV-gating outputs, and paper-trading reports.
- `unified_outperformance/` and `e0_e8_best_model_base/` contain the strongest recent comparative artifacts.

## Archived code
- `ARCHIVE/legacy_runners/run_kalshi_real_backtest.py` was moved out of active workflow to reduce confusion with newer OOS and E/U benchmark pipelines.
