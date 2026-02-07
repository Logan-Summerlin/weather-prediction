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
| 3 | Neural Network V1 | NOT STARTED |
| 4 | Enhancements | NOT STARTED |
| 5 | Confidence Intervals | NOT STARTED |
| 6 | Scale Up (25 yr) | NOT STARTED |
| 7 | Documentation | NOT STARTED |

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
