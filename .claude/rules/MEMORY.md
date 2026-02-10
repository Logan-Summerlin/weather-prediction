# Project Memory

> **RULE:** This file must remain under 500 lines. No single write/edit may add more than 50 lines. Prune older or redundant entries before appending.

---

## Project Overview
NYC daily max-temperature prediction using surrounding NOAA weather stations. Target: Central Park (USW00094728). Predict day-t TMAX from surrounding stations' day t-1 observations.

## Phase Status
| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline | COMPLETE |
| 2 | Baseline Models | COMPLETE |
| 3 | Neural Network V1 | COMPLETE |
| 4 | Enhancements | COMPLETE |
| 5 | Confidence Intervals | NOT STARTED |
| 6 | Scale Up (25 yr) | **COMPLETE** |
| 7 | Documentation | IN PROGRESS |

## NN Pipeline Optimization (2026-02-10) — MAJOR MILESTONE
- **70+ model configurations tested** with real NOAA data (25yr, 48 stations)
- **Best test MAE: 1.959°F** (seasonal warm-season Ridge+MOS), **2.086°F** (full-year Hybrid NN)
- **Best OOS MAE: 2.056°F** (MOS Correction NN), **2.093°F** (MOS Correction NN tiny)
- **MOS integration is the key lever**: reduces MAE from ~4.3°F to ~2.1°F
- **Recommended production model**: C_Correction_NN_tiny (2.090 test / 2.093 OOS)
- Full report: `reports/nn_pipeline_optimization_report.md`
- Scripts: `scripts/enhanced_nn_pipeline.py`, `scripts/architecture_sweep.py`, `scripts/mos_ensemble_pipeline.py`, `scripts/advanced_models_eval.py`

## Key Implementation Lessons
- No NOAA API token needed — bulk .dly downloads from `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/` work reliably
- GHCN .dly format: 21-char header + 31 × 8-char daily values; temps in tenths °C; -9999 = missing; non-blank qflag = failed QC
- Temperature conversion: `(value / 10) * 9/5 + 32`
- Scaler fit on training data only — no leakage
- Chronological split (no shuffling) to avoid temporal leakage
- Forward-fill ≤3 days, then impute remaining NaNs with training-set column means
- Delta-T target: biggest single improvement for station-only models
- Without MOS, station-only NN plateaus at ~4.0°F MAE regardless of architecture
- MOS residual correction is the most robust approach (smallest IS/OOS gap)

## Key Results Summary
### Baselines (5yr data, 14 stations)
- Persistence: MAE=5.06°F | Climatology: MAE=6.15°F | Ridge: MAE=4.33°F

### Phase 4 Best (5yr, enhanced features)
- NN Delta+Huber+AR: MAE=3.95°F

### Optimization Best (25yr, 48 stations, MOS)
- C_Correction_NN_tiny: 2.090°F test / 2.093°F OOS (production recommendation)
- B_Hybrid_NN: 2.086°F test / 2.177°F OOS
- E_warm_Ridge (seasonal): 1.959°F test / 2.244°F OOS

## Station Network
- 48 qualifying stations (≥80% completeness over 1998-2024) from ~52 candidates
- Config: `config_expanded.py`, registry: `src/station_registry.py`
- Geography gap analysis: `reports/station_geography_report.md`
- ESE gap (LI south shore) confirmed unfillable; SSE ocean gap irreducible

## Active File Reference
| Category | Key Files |
|----------|-----------|
| Config | `config.py`, `config_expanded.py` |
| Data pipeline | `src/data_collection.py`, `src/data_preprocessing.py` |
| Models | `src/model.py` (TempPredictorV1) |
| Training | `src/train.py` |
| Evaluation | `src/evaluate.py`, `src/baselines.py` |
| Optimization scripts | `scripts/enhanced_nn_pipeline.py`, `scripts/architecture_sweep.py`, `scripts/mos_ensemble_pipeline.py`, `scripts/advanced_models_eval.py` |
| Kalshi/Trading | `src/trading.py`, `src/kalshi_client.py`, `src/kalshi_backtester.py` |
| Market proxy | `src/market_proxy.py`, `src/enhanced_market_proxy.py`, `src/mos_market_proxy.py` |
| Backtest scripts | `scripts/run_max_train_backtest.py`, `scripts/run_mos_backtest.py` |
| MOS data | `scripts/download_iem_mos.py`, `scripts/validate_mos_quality.py` |
| Reports | `reports/nn_pipeline_optimization_report.md`, `reports/mos_integration_report.md` |

## Cleanup Log (2026-02-10)
Deleted outdated/superseded files from Phases 1-4:
- 12 results directories (baselines, nn_v1, phase3, phase4, phase4_expanded, experiments, station_sensitivity, 5 old kalshi dirs)
- 6 runner scripts (run_experiments, run_hp_tuning, run_phase3, run_phase4, run_phase4_expanded, run_station_sensitivity)
- 5 src modules (experiments, models_v2, train_v2, data_preprocessing_v2, data_preprocessing_expanded)
- 5 test files (matching deleted src modules)
- 7 reports (phase1-4 PM reports, phase3_kalshi, nn_pipeline_audit)
- 3 scripts (generate_expanded_predictions, generate_real_predictions, run_expanded_backtest)
