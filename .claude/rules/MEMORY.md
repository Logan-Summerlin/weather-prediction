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

## Phase 1 — Data Pipeline (COMPLETE)

### Files Delivered
- `config.py` — 14 surrounding stations + Central Park, 2018-2022, all hyperparams
- `src/data_collection.py` — GHCN .dly bulk download + fixed-width parser
- `src/data_preprocessing.py` — merge, lag-1 features, cyclical dates, split, scale
- `tests/test_data_collection.py` — 29 tests
- `tests/test_data_preprocessing.py` — 32 tests
- `data/stations.csv` — 15 stations with coordinates, distances, directions
- `reports/phase1_pm_report.md` — PM completion report
- `.gitignore` — excludes raw data, caches, model checkpoints

### Data Summary
- 15 stations (1 target + 14 surrounding), all >99% completeness
- 30 features: 28 lagged station TMAX/TMIN + sin_day + cos_day
- Train: 1277 rows (2018-01-02 to 2021-07-01)
- Val: 274 rows (2021-07-02 to 2022-04-01)
- Test: 274 rows (2022-04-02 to 2022-12-31)
- Target mean: 62.6°F, std: 17.8°F, range: 13.1–98.1°F

### Implementation Lessons
- No NOAA API token needed — bulk .dly downloads from `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/` work reliably
- GHCN .dly format: 21-char header + 31 × 8-char daily values; temps in tenths °C; -9999 = missing; non-blank qflag = failed QC
- Temperature conversion: `(value / 10) * 9/5 + 32`
- Scaler fit on training data only — no leakage
- Chronological split (no shuffling) to avoid temporal leakage
- Forward-fill ≤3 days, then impute remaining NaNs with training-set column means
- All 61 tests pass; pipeline is end-to-end operational

## Phase 2 — Baseline Models (COMPLETE)

### Files Delivered
- `src/baselines.py` — 4 models: Persistence, Climatology, Linear Regression, Ridge
- `src/evaluate.py` — Full evaluation framework: metrics, seasonal breakdown, 5 plot types, report generation
- `tests/test_baselines.py` — 61 tests
- `tests/test_evaluate.py` — 55 tests
- `run_baselines.py` — End-to-end evaluation script
- `results/baselines/` — 17 PNG plots + evaluation report
- `reports/phase2_pm_report.md` — PM completion report

### Baseline Results (Test Set, n=274)
- Persistence: MAE=5.06°F, R²=0.799
- Climatology: MAE=6.15°F, R²=0.747
- Linear Regression: MAE=4.35°F, R²=0.875
- **Ridge (alpha=1.0): MAE=4.33°F, R²=0.876** (best baseline)

### Key Findings
- Ridge ≈ OLS: multicollinearity not causing instability
- Surrounding stations add ~0.7°F MAE improvement over persistence
- Winter/spring are hardest seasons (MAE 1-2°F higher)
- Stretch goal (≤2°F) requires roughly halving best baseline error
- 177 total tests pass (61 baselines + 55 evaluate + 61 Phase 1)

## Phase 3 — Neural Network V1 (COMPLETE)

### Key Results
- NN V1 [64,32]: Test MAE=4.29°F, RMSE=5.69, R²=0.869
- Beats Ridge by 0.04°F (0.9%) — marginal improvement
- Dropout=0.0 optimal; model has 4,097 parameters
- 318 total tests pass

## Phase 4 — Enhancements (COMPLETE)

### Files Delivered
- `src/data_preprocessing_v2.py` — Delta-T target, autoregressive input, sector features, trends
- `src/train_v2.py` — Huber/MAE/MSE loss, delta-T reconstruction
- `src/models_v2.py` — EnhancedMLP, MultiLagMLP, LSTMPredictor, StationAttentionModel
- `src/experiments.py` — Sensitivity experiment framework
- `run_phase4.py`, `run_experiments.py` — Runner scripts
- Tests: 200 new tests (55+42+69+34), 525 total pass

### Best Results (Test Set, n=274)
- **NN Delta+Huber+AR: MAE=3.95°F, RMSE=5.33, R²=0.885** (best overall)
- Delta-T target: biggest single improvement (-0.27°F over raw)
- Autoregressive NYC TMAX(t-1): adds 0.07°F improvement
- Full enhanced features (79 feat) overfits with current data size

### Sensitivity Findings
- Huber loss best for delta-T; MSE better for raw TMAX
- TMAX features dominate; TMIN alone is weakest
- Date features contribute ~0.2°F; low dropout optimal
- Larger architectures don't help with limited data

### Phase 4.3 — Station Expansion (COMPLETE)
- Expanded from 14 to 50 surrounding stations (51 total incl. target)
- `config_expanded.py`, `src/station_registry.py`, `src/station_discovery.py`
- `src/data_preprocessing_expanded.py` with missingness masking
- Station count sensitivity: optimal ~10-14 stations with 5-year data
- More stations degrade performance due to overfitting (202+ features / 1277 samples)
- Infrastructure ready for Phase 6 (25-year data) to exploit expanded stations
- 652 total tests pass
- **Geography gap analysis completed** (`reports/station_geography_report.md`):
  - S near-field gap filled (~50mi McGuire-Dix-Lakehurst NJ); SW Ring3 gap filled (~130mi Dover AFB DE)
  - ESE gap (Farmingdale-JFK, 96-132 deg) confirmed unfillable — no active GHCN station on LI south shore
  - SSE ocean gap accepted as irreducible; total ~52 surrounding stations post-gap-fill
