# Project Memory

> **RULE:** Keep this file under 500 lines. Keep entries concise and prune stale items.

---

## Project Overview
Multi-city daily max-temperature probabilistic forecasting for Kalshi temperature bucket contracts, with calibration-first evaluation and EV-gated trading simulation. Five cities: NYC (KXHIGHNY) fully operational, Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) pipelines complete through backtesting with promotion evaluations done, Atlanta (KXHIGHTATL) pipeline complete, Austin (KXHIGHAUS) pipeline complete but needs work.

## Current Phase Status
| Layer | Description | Status |
|---|---|---|
| 1 | Operational + historical ingestion (multi-city) | COMPLETE |
| 2 | Time-safe feature engineering (multi-city) | COMPLETE |
| 3 | Distributional + synthesis modeling E/WGA/U | COMPLETE |
| 4 | Post-hoc calibration + bucketization | COMPLETE |
| 5 | EV/risk trading simulation + gating | COMPLETE |
| 6 | Daily production hardening (cutoff + kill switch) | IN PROGRESS |
| 7 | Multi-city expansion (CHI/PHL/ATL/AUS) | BACKTEST COMPLETE |
| 8 | Operational dashboard | PLANNING |

## Canonical Benchmark State (2026-02-19)

### NYC (KXHIGHNY)
- E-core benchmark: `scripts/run_e0_e8_best_model_benchmark.py` (E0–E22).
- WGA extension benchmark: `scripts/run_wga_v2_benchmark.py` (E38–E42).
- Unified benchmark: `scripts/run_unified_outperformance_benchmark.py` (U0–U9).
- Best model: **U7_regime_conditional (contract Brier 0.1137)**.
- Kalshi market Brier: 0.1271 | Edge: **+0.0133**.
- Top cluster: E40 (0.1138), U6 (0.1141), E17 (0.1141), E42 (0.1150).

### Chicago (KXHIGHCHI)
- Best model (real Kalshi): **U7_extended (contract Brier 0.1091)**.
- Kalshi market Brier: 0.1253 | Edge: **+0.0162**.
- Real Kalshi backtest: +$2,406 (+241%), Sharpe 6.07, 71% win rate, 1144 trading days.
- Simulated market backtest: +$48 (+4.8%), Sharpe 1.17, 64% win rate, 317 days.
- Promotion eval: **10/10 gates PASS**.
- Cross-city best model: U8_cv_ensemble (contract Brier 0.1087).

### Philadelphia (KXHIGHPHL)
- Best model (real Kalshi): **U9_kitchen_sink (contract Brier 0.1060)**.
- Kalshi market Brier: 0.1099 | Edge: **+0.0039** (narrow).
- Real Kalshi backtest: +$340 (+34%), Sharpe 2.76, 46% win rate, 451 trading days.
- Promotion eval: **10/10 gates PASS**.
- **Key issue:** Narrow Brier edge; not yet profitable against simulated market.

### Atlanta (KXHIGHTATL)
- Target: Hartsfield-Jackson (USW00013874), ~50 stations.
- Promotion eval: **11/11 gates PASS**.
- Status: Pipeline complete, backtest done.

### Austin (KXHIGHAUS)
- Target: Austin-Bergstrom (USW00013904), ~56 stations.
- Promotion eval: **8/13 gates FAIL**.
- Status: Pipeline complete, needs model improvements.

## Key Lessons Learned
1. **Real Kalshi vs simulated market results diverge significantly.** Always report both.
2. **Winter (DJF) is the universal weak season** across all cities.
3. **Chicago has strongest expansion signal** — large Brier edge (+0.0162).
4. **Philadelphia edge is thin** (+0.0039) — needs calibration improvements.
5. **Contract Brier (not bucket-day Brier) is the correct metric.**
6. **Promotion evaluation thresholds are city-specific** — CHI uses 0.16, PHL uses 0.14.

## Operational Guardrails (locked)
1. No delayed training-grade source may appear in live feature computation.
2. Chronological splits only; no random shuffles.
3. Trade logic requires calibrated probabilities and cost-aware EV.
4. Persist audit artifacts for each run.
5. Trigger kill-switch on critical data/schema/calibration failures.
6. While subagents are working, enter 3-minute sleep/wake cycles to monitor progress.
7. Always use actual data — never use made-up or proxy data.
8. Always compare model predictions against Kalshi market prices as primary benchmark.

## Known Audit Findings
1. **Settlement-Price Leakage (CRITICAL):** CHI/PHL Unified models U2-U5 used settlement-time market_prob as feature. E-series unaffected. See `ARCHIVE/legacy_root_docs/AUDIT_model_cheating_investigation.md`.
2. **Brier Metric Scale Inconsistency:** NYC binary contract-row vs CHI/PHL multiclass bucket-day. Now standardized. See `ARCHIVE/legacy_root_docs/AUDIT_cross_city_brier_integrity.md`.

## Repo Hygiene State (2026-02-19)
- Major archive cleanup performed 2026-02-19: moved 23 superseded scripts, 6 legacy tests, all docs/ and reports/ contents, 3 root-level audit/planning docs to ARCHIVE/.
- ARCHIVE/ now has 10 subdirectories (legacy_scripts_v2, legacy_tests, legacy_docs_v2, legacy_reports, legacy_root_docs, plus 5 original legacy dirs).
- **Codebase simplification (2026-02-19):**
  - 24 per-city pipeline scripts consolidated into 6 unified scripts with `--city` flag.
  - Old per-city scripts converted to thin backward-compatible wrappers (11 lines each).
  - 3 per-city test files consolidated into `tests/test_city_pipeline.py` (parameterized).
  - 9 experimental scripts moved to `scripts/experiments/`.
  - Config system cleaned up: `src/operational_data.py` and `src/wga_data_pipeline.py` now use dynamic importlib-based config loading supporting all 5 cities.
  - See `SIMPLIFICATION_REPORT.md` for full analysis.
- Documentation consolidated into 3 files in `docs/`:
  - `docs/01_current_state_and_directory.md` — Codebase state and file directory.
  - `docs/02_model_families_and_methods.md` — Model families, functions, and methods.
  - `docs/03_principles_and_city_portability.md` — Principles and US city portability guide.
- Planning docs: `nyc_temp_prediction_project_plan.md` (updated 2026-02-19).

## Active File Reference
| Domain | Key Files |
|---|---|
| NYC E0–E22 benchmark | `scripts/run_e0_e8_best_model_benchmark.py` |
| NYC WGA E38–E42 | `scripts/run_wga_v2_benchmark.py` |
| NYC Unified U0–U9 | `scripts/run_unified_outperformance_benchmark.py` |
| Multi-city template | `scripts/run_city_nws_kalshi_template_benchmark.py` |
| Core modeling | `src/model.py`, `src/advanced_model.py`, `src/wind_gated_attention.py`, `src/synthesis_model.py` |
| Extended models | `src/extended_models.py`, `src/baselines.py`, `src/crps_loss.py` |
| Calibration + evaluation | `src/calibration.py`, `src/contract_brier.py`, `src/evaluate.py` |
| Trading + market | `src/trading.py`, `src/kalshi_client.py`, `src/kalshi_backtester.py`, `src/live_trading.py` |
| Data + features | `src/data_collection.py`, `src/data_preprocessing.py`, `src/operational_features.py` |
| ASOS/NWP/Soundings | `src/asos_collection.py`, `src/nwp_collection.py`, `src/soundings_collection.py` |
| Preprocessing | `src/asos_preprocessing.py`, `src/nwp_preprocessing.py`, `src/soundings_preprocessing.py` |
| Station management | `src/station_registry.py`, `src/station_discovery.py`, `src/city_config.py` |
| Market proxies | `src/market_proxy.py`, `src/mos_market_proxy.py`, `src/enhanced_market_proxy.py` |
| City configs | `config.py`, `config_expanded.py`, `config_chicago.py`, `config_philadelphia.py`, `config_atlanta.py`, `config_austin.py` |
| Unified city pipeline | `scripts/run_data_collection.py`, `scripts/run_preprocessing.py`, `scripts/run_benchmark.py`, `scripts/run_synthesis_calibration.py`, `scripts/run_backtest.py`, `scripts/run_promotion_evaluation.py` (all accept `--city` flag) |
| Per-city wrappers | `scripts/run_chi_*.py`, `scripts/run_phl_*.py`, `scripts/run_atl_*.py`, `scripts/run_aus_*.py` (thin wrappers delegating to unified scripts) |
| City pipeline tests | `tests/test_city_pipeline.py` (parameterized for CHI/PHL/ATL) |
| Experimental scripts | `scripts/experiments/` (non-pipeline exploratory scripts) |
| CHI results | `results/chicago/` |
| PHL results | `results/philadelphia/` |
| ATL results | `results/atlanta/` |
| AUS results | `results/austin/` |
| Cross-city comparison | `results/cross_city_comparison/best_models_summary.json` |

## Immediate Priorities
1. **Improve DJF winter calibration** across all cities — biggest source of losses.
2. **Validate CHI for live paper trading** — strongest expansion candidate (10/10 gates, +0.0162 edge).
3. **Improve PHL Brier edge** before promoting to live (currently only +0.0039).
4. **Improve Austin model** to pass remaining 5 promotion gates.
5. Enforce cutoff-time feature availability checks via automated schema/contract tests.
6. Build operational dashboard for model vs market monitoring.
7. Improve execution microstructure assumptions (depth, queue, fill uncertainty).

## Multi-City Expansion Notes

### Chicago (KXHIGHCHI) — Promotion Ready
- Target: O'Hare (USW00094846), 62 buckets (-10°F floor)
- Station network: 55 stations, 4 rings, lake-effect sectors (WNW, Lake, SW, NE_Lake)
- Tests: 27 passing in `tests/test_chi_pipeline.py`
- **Next:** Live paper trading validation.

### Philadelphia (KXHIGHPHL) — Needs Calibration Work
- Target: PHL International (USW00013739), 57 buckets
- Station network: ~50 stations, 4 rings
- **Next:** Improve winter calibration, widen Brier edge.

### Atlanta (KXHIGHTATL) — Pipeline Complete
- Target: Hartsfield-Jackson (USW00013874), 57 buckets
- Station network: ~49 stations, Piedmont/mountain sectors
- **Next:** Backtest refinement, Kalshi data integration.

### Austin (KXHIGHAUS) — Needs Work
- Target: Austin-Bergstrom (USW00013904), 57 buckets
- Station network: ~56 stations
- **Next:** Model improvements to pass promotion gates.
