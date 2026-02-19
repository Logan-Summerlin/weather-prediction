# ARCHIVE

This folder stores legacy code, documents, and reports kept for reproducibility but excluded from the active daily benchmark/trading workflow.

## Active model-comparison paths (current)

- `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22 core, NYC)
- `scripts/run_wga_v2_benchmark.py` (E38–E42, NYC)
- `scripts/run_unified_outperformance_benchmark.py` (U0–U9, NYC)
- `scripts/run_city_nws_kalshi_template_benchmark.py` (multi-city NWS template)
- `scripts/run_chi_phl_unified_benchmark.py` (CHI+PHL unified)
- `scripts/run_<city>_benchmark.py` (per-city: chi, phl, atl, aus)
- `scripts/run_<city>_backtest.py` (per-city backtests)
- `scripts/run_<city>_promotion_evaluation.py` (per-city promotion gates)

## Archived items

### `legacy_scripts_v2/` (archived 2026-02-19)
Scripts from `scripts/` that are superseded, one-off analyses, or no longer in the active pipeline:
- `train_wga_mdn.py` — WGA-MDN V1 training, superseded by WGA V2 benchmark.
- `train_wga_v2.py` — WGA V2 prototyping, results captured in benchmark.
- `generate_max_training_predictions.py` — Legacy max-training NN forecast generation.
- `run_max_train_backtest.py` — Old max-train backtest, superseded by real Kalshi backtest.
- `run_mos_backtest.py` — MOS proxy backtest, replaced by real Kalshi backtest.
- `run_honest_benchmark.py` — One-off leakage audit benchmark.
- `run_extended_val_benchmark.py` — Extended validation gap analysis.
- `run_mos_residual_benchmark.py` — MOS residual correction experiment.
- `test_model_vs_benchmarks.py` — Supporting library, not standalone runner.
- `run_phl_nws_kalshi_benchmark.py` — PHL-specific NWS, use template instead.
- `build_extended_mos.py` — One-time MOS data prep.
- `run_aus_e_series_benchmark.py` — AUS E-series one-off eval.
- `download_all_dly.py` — Old GHCN batch downloader, replaced by city-specific collection.
- `download_real_ghcn.py` — NYC GHCN downloader, replaced by city pipeline.
- `retrain_extended_mos.py` — One-time MOS retrain experiment.
- `retrain_extended_validation.py` — One-time extended validation retrain.
- `mos_ensemble_pipeline.py` — MOS ensemble experiments, not in pipeline.
- `mos_sufficiency_analysis.py` — One-time MOS data quality audit.
- `airport_mos_proxy_analysis.py` — One-time airport MOS harmonization study.
- `airport_mos_similarity_analysis.py` — Exploratory analysis.
- `validate_mos_quality.py` — One-off quality check.
- `generate_mos_comparison_report.py` — One-time comparative analysis.
- `audit_cross_city_brier_scale.py` — One-time compliance check.

### `legacy_tests/` (archived 2026-02-19)
Test files for deprecated/removed functionality:
- `test_kalshi_backtest.py` — Superseded by `test_kalshi_backtester.py`.
- `test_nn_integration.py` — Phase 3 NN integration tests, superseded by synthesis models.
- `test_nn_v1_128_64.py` — Experimental NN layer sizing tests.
- `test_probabilistic_ensemble_experiments_v2.py` — Legacy probabilistic ensemble tests.
- `test_run_kalshi_real_oos.py` — Tests deprecated `run_kalshi_real_oos.py` runner.
- `test_run_nn.py` — Tests deprecated `run_nn.py` runner.

### `legacy_docs_v2/` (archived 2026-02-19)
All previous docs/ contents, now consolidated into 3 files in `docs/`:
- `current_state_and_directory.md` — Previous codebase directory reference.
- `top15_models_brier_function_reference.md` — Previous model family reference.
- `model_principles_and_us_city_portability.md` — Previous principles doc.
- `comprehensive_pipeline_and_model_report.md` — Previous comprehensive report.
- `brier_score_metrics.md` — Brier score aggregation best practices.
- `chicago_contract_spec.md` — CHI contract specification (draft).
- `philadelphia_contract_spec.md` — PHL contract specification (draft).
- `chi_phl_remaining_tasks.md` — CHI/PHL gap analysis.
- `new_city_model_end_to_end_guide.md` — New city implementation guide.

### `legacy_reports/` (archived 2026-02-19)
All previous reports/ contents:
- `README.md` — Reports directory description.
- `archived_models_catalog.md` — Historical model catalog.
- `benchmark_audit_report.md` — Benchmark audit findings.
- `benchmark_audit_rerun_report.md` — Audit rerun results.
- `chi_phl_nyc_template_gap_to_003.md` — Template gap analysis.
- `chi_phl_nyc_template_model_upgrade.md` — Template model upgrade plan.
- `e0_e8_kalshi_presettlement_brier_improvement_plan.md` — E-series improvement plan.
- `kalshi_nws_outperformance_strategy.md` — NWS outperformance strategy.
- `master_improvement_plan.md` — Master improvement plan.
- `probabilistic_ensemble_experiment_report.md` — Ensemble experiment results.

### `legacy_root_docs/` (archived 2026-02-19)
Root-level documents that are historical artifacts:
- `AUDIT_cross_city_brier_integrity.md` — Cross-city Brier metric-scale audit (2026-02-16).
- `AUDIT_model_cheating_investigation.md` — Settlement-price leakage audit (2026-02-16).
- `prediction_market_expansion.md` — Original CHI/PHL expansion plan.

### `legacy_runners/` (original)
- `run_kalshi_real_backtest.py` — Older in-sample-heavy runner.

### `legacy_experiments/` (original)
Phase-1/early exploration pipelines:
- `advanced_models_eval.py`, `enhanced_nn_pipeline.py`
- `phase1_architecture_temporal.py`, `phase1_combined_best.py`
- `phase1_ensemble_training.py`, `phase1_feature_engineering.py`
- `phase1_probabilistic_output.py`

### `legacy_root_runners/` (original)
Root-level runner scripts from early project phases:
- `run_baselines.py`, `run_nn.py`, `run_phase0.py`
- `run_kalshi_backtest.py`, `run_kalshi_oos_validation.py`
- `run_kalshi_real_oos.py`, `run_collect_all_stations.py`

### `legacy_scripts/` (original)
Early benchmark/experiment runners:
- `run_e0_e1_e2_benchmark.py`, `run_best_model_lineage_top2_benchmark.py`
- `run_top3_adjusted_benchmarks.py`, `run_wga_benchmark.py`
- `run_gfs_residual_no_nam_benchmark.py`, `architecture_sweep.py`
- `probabilistic_ensemble_experiments_v2.py`

### `legacy_docs/` (original)
Early-phase research documents:
- `Temperature_Forecasting_ML_LLM_Report.md`
- `weather_patterns_and_prediction_research.md`
