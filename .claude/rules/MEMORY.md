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

## Phase 1 Improvement (2026-02-10) — NEW BEST
- **100+ model configurations** across 5 workstreams (features, probabilistic, ensemble, architecture, synthesis)
- **Best test MAE: 1.987°F** (5-seed ensemble with combined features) — SUB-2°F!
- **Best OOS MAE: 2.010°F** (single A_NN_64_32 seed 42), **2.018°F** (5-seed ensemble)
- Key new features: MOS error memory (7/14/30d rolling bias), MOS×station interactions, temporal (day length, solar elevation, anomaly), spatial (gradients, frontal proxy)
- Probabilistic output: NLL→CRPS training, 95% PI coverage ≈ 95%, CRPS ≈ 1.48°F
- **Recommended**: A_NN_64_32 combined features (production), 5-seed ensemble (max accuracy), D_Probabilistic (trading)
- Report: `reports/phase1_improvement_report.md`
- Scripts: `scripts/phase1_*.py` (5 scripts), Results: `results/phase1_*/` (5 dirs)

## Prior: NN Pipeline Optimization (2026-02-10)
- 70+ configs tested. Prior best: C_Correction_NN_tiny (2.090 test / 2.093 OOS)
- MOS integration key lever: 4.3→2.1°F. Scripts: `scripts/mos_ensemble_pipeline.py` etc.

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

## WGA-MDN Model (2026-02-13) — Prediction Market Benchmark
- **Wind-Gated Attention + MDN**: `scripts/train_wga_mdn.py`, 47 stations, 13,245 params/seed
- Test MAE: 2.062°F (ensemble), 95% PI coverage: 95.2%
- E36_wga_contract_brier: Overall Brier 0.1137, **ECE 0.0088** (best ever), IS Brier 0.1186
- E36 ties E17 (0.1136) as best overall Brier; attention provides complementary signal to flat model
- OOS gap to PreSettlement remains ~0.004 (0.1027 vs 0.0988); likely fundamental information limit
- Remaining levers: station ablation, multi-head attention, end-to-end Brier training
- Results: `results/wga_mdn_model/`, `results/prediction_market_benchmark/wga_mdn_model/`
- Strategy doc: `reports/kalshi_nws_outperformance_strategy.md`

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
| WGA-MDN | `scripts/train_wga_mdn.py`, `scripts/run_wga_benchmark.py`, `src/wind_gated_attention.py` |
| Reports | `reports/nn_pipeline_optimization_report.md`, `reports/mos_integration_report.md` |

## Cleanup Log (2026-02-10)
Deleted outdated/superseded files from Phases 1-4:
- 12 results directories (baselines, nn_v1, phase3, phase4, phase4_expanded, experiments, station_sensitivity, 5 old kalshi dirs)
- 6 runner scripts (run_experiments, run_hp_tuning, run_phase3, run_phase4, run_phase4_expanded, run_station_sensitivity)
- 5 src modules (experiments, models_v2, train_v2, data_preprocessing_v2, data_preprocessing_expanded)
- 5 test files (matching deleted src modules)
- 7 reports (phase1-4 PM reports, phase3_kalshi, nn_pipeline_audit)
- 3 scripts (generate_expanded_predictions, generate_real_predictions, run_expanded_backtest)
