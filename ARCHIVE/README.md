# ARCHIVE

This folder stores legacy code and documents kept for reproducibility but excluded from the active daily benchmark/trading workflow.

## Active model-comparison paths (current)

- `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22 core)
- `scripts/run_wga_v2_benchmark.py` (E38–E42)
- `scripts/run_unified_outperformance_benchmark.py` (U0–U9)

## Archived items

### `legacy_runners/` (original)
- `run_kalshi_real_backtest.py` — Older in-sample-heavy runner superseded by the OOS-oriented benchmark stack.

### `legacy_experiments/` (original)
Phase-1/early exploration pipelines now redundant with E0–E42 and U0–U9 benchmark runners:
- `advanced_models_eval.py`
- `enhanced_nn_pipeline.py`
- `phase1_architecture_temporal.py`
- `phase1_combined_best.py`
- `phase1_ensemble_training.py`
- `phase1_feature_engineering.py`
- `phase1_probabilistic_output.py`

### `legacy_root_runners/`
Root-level runner scripts from early project phases, now superseded by the `scripts/` benchmark stack:
- `run_baselines.py` — Phase 2 baseline evaluation (persistence, climatology, ridge).
- `run_nn.py` — Phase 3 feedforward NN training runner.
- `run_phase0.py` — Phase 1 data pipeline validation runner.
- `run_kalshi_backtest.py` — Early Kalshi backtest runner, superseded by E-series benchmarks.
- `run_kalshi_oos_validation.py` — Early OOS validation, superseded by unified benchmark.
- `run_kalshi_real_oos.py` — Early real OOS runner, superseded by unified benchmark.
- `run_collect_all_stations.py` — One-time station collection utility; station list now in `config_expanded.py`.

### `legacy_scripts/`
Scripts from `scripts/` that were intermediate benchmark/experiment runners, now superseded:
- `run_e0_e1_e2_benchmark.py` — Early E0/E1/E2 benchmark, superseded by `run_e0_e8_best_model_benchmark.py`.
- `run_best_model_lineage_top2_benchmark.py` — Top-2 lineage audit, results captured.
- `run_top3_adjusted_benchmarks.py` — Top-3 adjusted model benchmark, results captured.
- `run_wga_benchmark.py` — WGA V1 benchmark (E34–E37), superseded by `run_wga_v2_benchmark.py`.
- `run_gfs_residual_no_nam_benchmark.py` — One-off GFS residual experiment, results captured.
- `architecture_sweep.py` — Phase 1 architecture exploration, results captured.
- `probabilistic_ensemble_experiments_v2.py` — Probabilistic ensemble experiments, superseded by E-series synthesis.

### `legacy_docs/`
Early-phase research documents superseded by current operational documentation:
- `Temperature_Forecasting_ML_LLM_Report.md` — Initial ML/LLM research report.
- `weather_patterns_and_prediction_research.md` — Early weather pattern research notes.
