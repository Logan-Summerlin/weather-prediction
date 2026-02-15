# Project Memory

> **RULE:** Keep this file under 500 lines. Keep entries concise and prune stale items.

---

## Project Overview
Multi-city daily max-temperature probabilistic forecasting for Kalshi temperature bucket contracts, with calibration-first evaluation and EV-gated trading simulation. Currently operational for NYC (KXHIGHNY). Expansion to Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) in progress.

## Current Phase Status
| Layer | Description | Status |
|---|---|---|
| 1 | Operational + historical ingestion (NYC) | COMPLETE |
| 2 | Time-safe feature engineering (NYC) | COMPLETE |
| 3 | Distributional + synthesis modeling E/WGA/U (NYC) | COMPLETE |
| 4 | Post-hoc calibration + bucketization (NYC) | COMPLETE |
| 5 | EV/risk trading simulation + gating (NYC) | COMPLETE |
| 6 | Daily production hardening (cutoff enforcement + kill switch) | IN PROGRESS |
| 7 | Multi-city expansion (Chicago + Philadelphia) | IN PROGRESS |
| 8 | Operational dashboard | PLANNING |

## Canonical Benchmark State (2026-02-15)
- E-core benchmark: `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22).
- WGA extension benchmark: `scripts/run_wga_v2_benchmark.py` (E38–E42).
- Unified benchmark: `scripts/run_unified_outperformance_benchmark.py` (U0–U9).
- Current best overall model Brier: **U7_regime_conditional (0.1137)**.
- Top cluster: **E40_lag2_contract_brier (0.1138), U6_platt_on_u5 (0.1141), E17_contract_brier (0.1141), E42_dual_attention (0.1150)**.

## Operational Guardrails (locked)
1. No delayed training-grade source may appear in live feature computation.
2. Chronological splits only; no random shuffles.
3. Trade logic requires calibrated probabilities and cost-aware EV.
4. Persist audit artifacts for each run (mass checks, reliability metrics, trading diagnostics).
5. Trigger kill-switch on critical data/schema/calibration failures.

## Repo Hygiene State (2026-02-15)
- Legacy phase-1 experiments in `ARCHIVE/legacy_experiments/`.
- Legacy root-level runners in `ARCHIVE/legacy_root_runners/`.
- Legacy intermediate benchmark scripts in `ARCHIVE/legacy_scripts/`.
- Legacy early research docs in `ARCHIVE/legacy_docs/`.
- Legacy backtest runner in `ARCHIVE/legacy_runners/`.
- Current docs aligned to E42/U9 state:
  - `docs/current_state_and_directory.md`
  - `docs/top15_models_brier_function_reference.md`
  - `docs/model_principles_and_us_city_portability.md`
- Planning doc synchronized: `nyc_temp_prediction_project_plan.md`.
- Expansion plan: `prediction_market_expansion.md`.

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
| ASOS/NWP/Soundings | `src/asos_collection.py`, `src/nwp_collection.py`, `src/soundings_collection.py` |
| Station management | `src/station_registry.py`, `src/station_discovery.py`, `config_expanded.py` |
| Market proxies | `src/market_proxy.py`, `src/mos_market_proxy.py`, `src/enhanced_market_proxy.py` |
| Chicago pipeline | `config_chicago.py`, `scripts/run_chi_*.py`, `tests/test_chi_pipeline.py` |
| Philadelphia pipeline | `config_philadelphia.py`, `scripts/run_phl_*.py` |

## Immediate Priorities
1. Enforce cutoff-time feature availability checks via automated schema/contract tests.
2. Consolidate promotion rubric for production candidate selection across E/WGA/U families.
3. Execute multi-city expansion plan (Chicago + Philadelphia).
4. Build operational dashboard for model vs market monitoring.
5. Improve execution microstructure assumptions (depth, queue, fill uncertainty).

## Multi-City Expansion Notes

### Chicago (KXHIGHCHI) — Phase 2 Complete
- Target: O'Hare (USW00094846), 11 buckets (10°F floor)
- Station network: 45 stations across 4 rings (9 near / 12 regional / 16 extended / 8 far)
- Meteorological sectors: WNW (cold advection), Lake (Michigan moderation), SW (Gulf warm), NearField, NE_Lake
- Full pipeline: `config_chicago.py` + 6 scripts (data collection, preprocessing, benchmark, synthesis/calibration, backtest, promotion eval)
- Tests: 27 passing in `tests/test_chi_pipeline.py`
- Status: Pipeline code complete. Next: execute data collection → preprocessing → benchmark runs.

### Philadelphia (KXHIGHPHL)
- Target: PHL International (USW00013739), 10 buckets
- Station network: ~48 stations in `config_philadelphia.py`
- Full pipeline: 6 scripts in `scripts/run_phl_*.py`
- Status: Pipeline code complete. Data collection and preprocessing in progress.

# currentDate
Today's date is 2026-02-15.
