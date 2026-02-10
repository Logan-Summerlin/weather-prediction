# MEMORY.md — Project Working Memory (Source of Truth)

**Last updated:** 2026-02-10

## Current canonical facts

- Benchmark script: `scripts/test_model_vs_benchmarks.py`
- Default benchmark model inputs:
  - `data/best_model_predictions_2023_2024.csv`
  - `data/best_model_predictions_2025.csv`
- Legacy model available only via `--use-legacy-model` for controlled comparisons.

## Latest validated benchmark outputs

Generated in `results/prediction_market_benchmark/`:

- `full_benchmark_report.md`
- `presettlement_brier_scores.csv`
- `presettlement_calibration.csv`
- `trading_simulation_results.csv`
- `model_probability_mass_check.csv`
- `nws_probability_mass_check.csv`

## Methodology decisions locked in

1. Trading simulation must use bid/ask crossing costs (not midpoint-only execution).
2. Sharpe is reported as both raw per-trade and annualized.
3. OOS trading output must include uncertainty intervals (bootstrap CI).
4. Probability mass checks per date are mandatory artifacts.

## Known open issues

1. Full microstructure fill model (queue/depth/slippage) is not yet implemented.
2. Distribution bucketization is still Gaussian-based.
3. Calibration can still degrade in high-probability bins.
4. Operational train/infer parity for all live features is not fully complete.

## Documentation authority order

The following are documentation source-of-truth files and should stay synchronized:

1. `reports/benchmark_audit_report.md`
2. `reports/benchmark_audit_rerun_report.md`
3. `nyc_temp_prediction_project_plan.md`
4. `MEMORY.md`
5. `reports/master_improvement_plan.md`
