# Project Memory

> **RULE:** Keep this file under 500 lines. Keep entries concise and prune stale items.

---

## Project Overview
NYC daily max-temperature probabilistic forecasting for Kalshi KXHIGHNY bucket contracts, with calibration-first evaluation and EV-gated trading simulation.

## Current Phase Status
| Layer | Description | Status |
|---|---|---|
| 1 | Operational + historical ingestion | ACTIVE |
| 2 | Time-safe feature engineering | ACTIVE |
| 3 | Distributional modeling (E/WGA/U families) | ACTIVE |
| 4 | Post-hoc calibration + bucketization | ACTIVE |
| 5 | EV/risk trading simulation | ACTIVE |
| 6 | Daily production hardening | IN PROGRESS |
| 7 | Documentation/runbook synchronization | IN PROGRESS |

## Canonical Benchmark State (2026-02-13)
- Primary benchmark track: `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22).
- Unified synthesis track: `results/prediction_market_benchmark/unified_outperformance/benchmark_summary.csv`.
- Strongest E-family Brier in summary table: **E17_contract_brier_synthesis**.
- Top OOS trading P&L in E-family summary appears on **E19_platt_beta_calibration** (positive but small).
- U-series shows best overall Brier around **U7/U6 cluster** in current artifact.

## Operational Guardrails (locked)
1. No delayed training-grade data in live-time features.
2. Chronological splits only; no random shuffling.
3. Trade decisions require calibrated probabilities and cost-aware EV.
4. Preserve audit artifacts: probability mass checks, calibration tables, trading outputs.
5. Halt/kill-switch behavior required on critical data or calibration failures.

## Repository State Updates
- Archived legacy runner to: `ARCHIVE/legacy_runners/run_kalshi_real_backtest.py`.
- Added ARCHIVE index: `ARCHIVE/README.md`.
- Added project docs:
  - `docs/current_state_and_directory.md`
  - `docs/top15_models_brier_function_reference.md`
  - `docs/model_principles_and_us_city_portability.md`
- Updated planning doc: `nyc_temp_prediction_project_plan.md`.

## Active File Reference
| Domain | Key Files |
|---|---|
| Core benchmark logic | `scripts/run_e0_e8_best_model_benchmark.py`, `scripts/test_model_vs_benchmarks.py` |
| Unified synthesis benchmark | `scripts/run_unified_outperformance_benchmark.py` |
| Forecast modeling | `src/model.py`, `src/wind_gated_attention.py`, `src/synthesis_model.py` |
| Calibration + evaluation | `src/calibration.py`, `src/evaluate.py`, `src/kalshi_backtester.py` |
| Trading + market integration | `src/trading.py`, `src/kalshi_client.py`, `run_kalshi_real_oos.py` |
| Feature/ingestion | `src/data_collection.py`, `src/data_preprocessing.py`, `src/operational_features.py`, `src/asos_collection.py`, `src/nwp_collection.py` |

## Next Tactical Priorities
1. Formalize daily cutoff-time data availability checks as enforceable schema/tests.
2. Consolidate production-candidate model selection criteria across E/U/WGA families.
3. Improve trading microstructure realism (depth/queue/fill uncertainty).
4. Expand drift detection automation and documented incident runbook steps.
