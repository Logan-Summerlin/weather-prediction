# Project Memory

> **RULE:** Keep this file under 500 lines. Keep entries concise and prune stale items.

---

## Project Overview
Multi-city daily max-temperature probabilistic forecasting for Kalshi temperature bucket contracts, with calibration-first evaluation and EV-gated trading simulation. NYC (KXHIGHNY) fully operational. Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) pipelines complete through backtesting with promotion evaluations done.

## Current Phase Status
| Layer | Description | Status |
|---|---|---|
| 1 | Operational + historical ingestion (NYC) | COMPLETE |
| 2 | Time-safe feature engineering (NYC) | COMPLETE |
| 3 | Distributional + synthesis modeling E/WGA/U (NYC) | COMPLETE |
| 4 | Post-hoc calibration + bucketization (NYC) | COMPLETE |
| 5 | EV/risk trading simulation + gating (NYC) | COMPLETE |
| 6 | Daily production hardening (cutoff enforcement + kill switch) | IN PROGRESS |
| 7 | Multi-city expansion (Chicago + Philadelphia) | BACKTEST COMPLETE |
| 8 | Operational dashboard | PLANNING |

## Canonical Benchmark State (2026-02-17)

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
- Seasonal weakness: DJF winter (simulated -$92, real Kalshi still +$564).
- Cross-city best model: U8_cv_ensemble (contract Brier 0.1087).

### Philadelphia (KXHIGHPHL)
- Best model (real Kalshi): **U9_kitchen_sink (contract Brier 0.1060)**.
- Kalshi market Brier: 0.1099 | Edge: **+0.0039** (narrow).
- Real Kalshi backtest: +$340 (+34%), Sharpe 2.76, 46% win rate, 451 trading days.
- Simulated market backtest: -$22 (-2.2%), Sharpe -1.57, 63% win rate, 110 days.
- Promotion eval: **10/10 gates PASS**.
- Seasonal weakness: DJF winter (simulated -$62, real Kalshi mixed).
- Cross-city best model: U2_contract_ridge (contract Brier 0.1058).
- **Key issue:** Narrow Brier edge vs Kalshi; not yet profitable against simulated market. More calibration work needed for live readiness.

## Key Lessons Learned (2026-02-17)
1. **Real Kalshi vs simulated market results diverge significantly.** Simulated market proxy is much harder to beat. Always report both.
2. **Winter (DJF) is the universal weak season** across all three cities. Cold-season volatility, storm tracks, and regime shifts degrade model skill.
3. **Chicago has strongest expansion signal** — large Brier edge (+0.0162), profitable in real Kalshi backtest across all seasons.
4. **Philadelphia edge is thin** (+0.0039) — needs winter calibration improvements before live deployment.
5. **Contract Brier (not bucket-day Brier) is the correct metric** for Kalshi settlement logic comparison. All benchmarks now use contract Brier.
6. **Promotion evaluation thresholds are city-specific** — CHI uses 0.16, PHL uses 0.14 Brier thresholds.

## Operational Guardrails (locked)
1. No delayed training-grade source may appear in live feature computation.
2. Chronological splits only; no random shuffles.
3. Trade logic requires calibrated probabilities and cost-aware EV.
4. Persist audit artifacts for each run (mass checks, reliability metrics, trading diagnostics).
5. Trigger kill-switch on critical data/schema/calibration failures.
6. While subagents are working, enter 3-minute sleep/wake cycles to monitor progress without burning context.
7. Always use actual data as the foundation for analysis — never use made-up, template, or "proxy" data.
8. Our models aim to beat Kalshi prediction markets; always compare model predictions against Kalshi market prices from ~24 hours before settlement as the primary benchmark.

## Repo Hygiene State (2026-02-17)
- Legacy files in `ARCHIVE/` (legacy_experiments, legacy_root_runners, legacy_scripts, legacy_docs, legacy_runners).
- Current docs: `docs/current_state_and_directory.md`, `docs/top15_models_brier_function_reference.md`, `docs/model_principles_and_us_city_portability.md`.
- Planning docs: `nyc_temp_prediction_project_plan.md`, `prediction_market_expansion.md`.

## Active File Reference
| Domain | Key Files |
|---|---|
| NYC E0–E22 benchmark | `scripts/run_e0_e8_best_model_benchmark.py` |
| NYC WGA E38–E42 | `scripts/run_wga_v2_benchmark.py` |
| NYC Unified U0–U9 | `scripts/run_unified_outperformance_benchmark.py` |
| Core modeling | `src/model.py`, `src/wind_gated_attention.py`, `src/synthesis_model.py` |
| Calibration + evaluation | `src/calibration.py`, `src/evaluate.py`, `src/kalshi_backtester.py` |
| Trading + market | `src/trading.py`, `src/kalshi_client.py` |
| Data + features | `src/data_collection.py`, `src/data_preprocessing.py`, `src/operational_features.py` |
| ASOS/NWP/Soundings | `src/asos_collection.py`, `src/nwp_collection.py`, `src/soundings_collection.py` |
| Station management | `src/station_registry.py`, `src/station_discovery.py`, `config_expanded.py` |
| Market proxies | `src/market_proxy.py`, `src/mos_market_proxy.py`, `src/enhanced_market_proxy.py` |
| Chicago pipeline | `config_chicago.py`, `scripts/run_chi_*.py`, `tests/test_chi_pipeline.py` |
| Philadelphia pipeline | `config_philadelphia.py`, `scripts/run_phl_*.py` |
| CHI results | `results/chicago/` (backtest, promotion_report_v2, unified_benchmark_results) |
| PHL results | `results/philadelphia/` (backtest, promotion_report_v2, unified_benchmark_results) |
| Cross-city comparison | `results/cross_city_comparison/best_models_summary.json` |

## Immediate Priorities
1. **Improve DJF winter calibration** across all cities — biggest source of losses.
2. **Validate CHI for live paper trading** — strongest expansion candidate (10/10 gates, +0.0162 edge).
3. Enforce cutoff-time feature availability checks via automated schema/contract tests.
4. Build operational dashboard for model vs market monitoring.
5. Improve PHL Brier edge before promoting to live (currently only +0.0039).
6. Improve execution microstructure assumptions (depth, queue, fill uncertainty).

## Multi-City Expansion Notes

### Chicago (KXHIGHCHI) — Backtest Complete, Promotion Ready
- Target: O'Hare (USW00094846), 11 buckets (10°F floor)
- Station network: 45 stations across 4 rings (9 near / 12 regional / 16 extended / 8 far)
- Meteorological sectors: WNW (cold advection), Lake (Michigan moderation), SW (Gulf warm), NearField, NE_Lake
- Full pipeline: `config_chicago.py` + 6 scripts
- Tests: 27 passing in `tests/test_chi_pipeline.py`
- Promotion: 10/10 gates pass. Best real Kalshi model: U7_extended (Brier 0.1091, Sharpe 6.07).
- **Next:** Live paper trading validation, then production deployment.

### Philadelphia (KXHIGHPHL) — Backtest Complete, Needs Calibration Work
- Target: PHL International (USW00013739), 10 buckets
- Station network: ~48 stations in `config_philadelphia.py`
- Full pipeline: 6 scripts in `scripts/run_phl_*.py`
- Promotion: 10/10 gates pass, but narrow real Kalshi edge (+0.0039) and negative simulated market P&L.
- **Next:** Improve winter calibration, widen Brier edge before live deployment.

# currentDate
Today's date is 2026-02-17.
