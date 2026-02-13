# Current State of the Codebase + Directory Guide

## Current system state (as of latest benchmark artifacts)
The repository is now centered on a **contract-aligned probabilistic forecasting and trading-research pipeline** for NYC KXHIGHNY buckets, with model lineage extended through **E42** and unified synthesis through **U9**.

### Active benchmark families
1. **E-series core benchmark (E0–E22)**
   - Runner: `scripts/run_e0_e8_best_model_benchmark.py`
   - Summary: `results/prediction_market_benchmark/e0_e8_best_model_base/e0_e22_benchmark_summary.csv`
2. **WGA V2 benchmark extensions (E38–E42)**
   - Runner: `scripts/run_wga_v2_benchmark.py`
   - Summary: `results/prediction_market_benchmark/wga_v2_model/benchmark_summary.csv`
3. **Unified cross-model synthesis (U0–U9)**
   - Runner: `scripts/run_unified_outperformance_benchmark.py`
   - Summary: `results/prediction_market_benchmark/unified_outperformance/benchmark_summary.csv`

### Headline benchmark position
- Best overall Brier among model variants in current artifacts: **U7_regime_conditional**.
- Strong cluster immediately behind U7 includes **E40 (lag2 contract-brier)**, **U6**, **E17**, and **E42/U9-range** variants.

## Directory map

```text
weather-prediction/
├── src/                              # Core modules (ingestion, features, modeling, calibration, trading)
├── scripts/                          # Active benchmark/training/utility runners
├── tests/                            # Unit tests
├── data/                             # Input datasets + generated prediction artifacts
├── results/                          # Benchmark outputs and diagnostics
├── reports/                          # Narrative strategy + analysis reports
├── docs/                             # Current operational/project docs
├── ARCHIVE/                          # Archived legacy code (not active path)
│   ├── legacy_runners/
│   └── legacy_experiments/
├── .claude/rules/MEMORY.md           # Project memory and active status
└── nyc_temp_prediction_project_plan.md
```

## Active module responsibilities

### `src/`
- Ingestion: `data_collection.py`, `asos_collection.py`, `nwp_collection.py`, `soundings_collection.py`
- Preprocessing: `data_preprocessing.py`, `asos_preprocessing.py`, `nwp_preprocessing.py`, `soundings_preprocessing.py`
- Modeling: `model.py`, `wind_gated_attention.py`, `synthesis_model.py`
- Probabilistic eval/calibration: `calibration.py`, `evaluate.py`, `crps_loss.py`, `baselines.py`
- Trading/market: `kalshi_client.py`, `kalshi_backtester.py`, `trading.py`, `market_proxy.py`, `mos_market_proxy.py`, `enhanced_market_proxy.py`

### `scripts/` (active high-value runners)
- `run_e0_e8_best_model_benchmark.py` → E0–E22 lineage.
- `run_wga_v2_benchmark.py` → WGA V2 variants and E38–E42 synthesis stack.
- `run_unified_outperformance_benchmark.py` → U0–U9 unified variants + EV gating + promotion checks.
- `run_extended_val_benchmark.py`, `run_gfs_residual_no_nam_benchmark.py` → targeted benchmark branches.
- MOS/data utilities remain active for input generation and diagnostics.

## What was archived in this update
To reduce confusion and prevent accidental use of superseded pipelines, legacy exploratory phase-1 scripts were moved from `scripts/` to `ARCHIVE/legacy_experiments/`.

Archived files:
- `advanced_models_eval.py`
- `enhanced_nn_pipeline.py`
- `phase1_architecture_temporal.py`
- `phase1_combined_best.py`
- `phase1_ensemble_training.py`
- `phase1_feature_engineering.py`
- `phase1_probabilistic_output.py`

These are preserved for historical reproducibility only.
