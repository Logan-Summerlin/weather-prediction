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
| 3 | Distributional + synthesis modeling (E/WGA/U) | ACTIVE |
| 4 | Post-hoc calibration + bucketization | ACTIVE |
| 5 | EV/risk trading simulation + gating | ACTIVE |
| 6 | Daily production hardening (cutoff enforcement + kill switch) | IN PROGRESS |
| 7 | Documentation synchronization | ACTIVE |

## Canonical Benchmark State (2026-02-13)
- E-core benchmark: `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22).
- WGA extension benchmark: `scripts/run_wga_v2_benchmark.py` (E38–E42).
- Unified benchmark: `scripts/run_unified_outperformance_benchmark.py` (U0–U9).
- Current best overall model Brier in active artifacts: **U7_regime_conditional (0.1137)**.
- Other top cluster: **E40_lag2_contract_brier, U6_platt_on_u5, E17_contract_brier, E42_dual_attention, U9_kitchen_sink**.

## Operational Guardrails (locked)
1. No delayed training-grade source may appear in live feature computation.
2. Chronological splits only; no random shuffles.
3. Trade logic requires calibrated probabilities and cost-aware EV.
4. Persist audit artifacts for each run (mass checks, reliability metrics, trading diagnostics).
5. Trigger kill-switch on critical data/schema/calibration failures.

## Repo Hygiene State
- Legacy exploratory phase-1 scripts moved to `ARCHIVE/legacy_experiments/`.
- Legacy backtest runner retained in `ARCHIVE/legacy_runners/run_kalshi_real_backtest.py`.
- Current docs aligned to E42/U9 state:
  - `docs/current_state_and_directory.md`
  - `docs/top15_models_brier_function_reference.md`
  - `docs/model_principles_and_us_city_portability.md`
- Planning doc synchronized: `nyc_temp_prediction_project_plan.md`.

## Active File Reference
| Domain | Key Files |
|---|---|
| E0–E22 benchmark core | `scripts/run_e0_e8_best_model_benchmark.py` |
| WGA E38–E42 benchmark | `scripts/run_wga_v2_benchmark.py` |
| Unified U0–U9 benchmark | `scripts/run_unified_outperformance_benchmark.py` |
| Core modeling modules | `src/model.py`, `src/wind_gated_attention.py`, `src/synthesis_model.py` |
| Calibration + evaluation | `src/calibration.py`, `src/evaluate.py`, `src/kalshi_backtester.py` |
| Trading + market integration | `src/trading.py`, `src/kalshi_client.py` |
| Data + operational features | `src/data_collection.py`, `src/data_preprocessing.py`, `src/operational_features.py` |

## Immediate Priorities
1. Enforce cutoff-time feature availability checks via automated schema/contract tests.
2. Consolidate promotion rubric for production candidate selection across E/WGA/U families.
3. Improve execution microstructure assumptions (depth, queue, fill uncertainty).
4. Expand drift alerting + incident runbook automation.
