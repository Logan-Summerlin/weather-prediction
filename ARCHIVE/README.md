# ARCHIVE

This folder stores legacy code kept for reproducibility but excluded from the active daily benchmark/trading workflow.

## Archived items

### `legacy_runners/`
- `run_kalshi_real_backtest.py`
  - Older in-sample-heavy runner superseded by the OOS-oriented benchmark stack.

### `legacy_experiments/`
The following scripts were moved from `scripts/` because they are phase-1/early exploration pipelines that are now redundant with the E0–E42 and U0–U9 benchmark runners:
- `advanced_models_eval.py`
- `enhanced_nn_pipeline.py`
- `phase1_architecture_temporal.py`
- `phase1_combined_best.py`
- `phase1_ensemble_training.py`
- `phase1_feature_engineering.py`
- `phase1_probabilistic_output.py`

Active model-comparison paths are now:
- `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22 core)
- `scripts/run_wga_v2_benchmark.py` (E38–E42)
- `scripts/run_unified_outperformance_benchmark.py` (U0–U9)
