# Phase 1 — Data Pipeline: Project Manager Report

## Status: COMPLETE

## Summary
Phase 1 (Data Pipeline) has been fully implemented, tested, and validated. All 15 NOAA weather stations have been successfully downloaded and preprocessed into train/val/test splits ready for model training.

## Deliverables

### 1.1 Project Structure
- Created full directory layout: `src/`, `tests/`, `data/`, `models/`, `results/`, `notebooks/`
- `config.py` — Centralized configuration with station IDs, date ranges, paths, hyperparameters
- `requirements.txt` — pandas, numpy, scikit-learn, torch, matplotlib, seaborn, requests, pytest
- `.gitignore` — Excludes raw data, caches, model checkpoints (large/re-downloadable files)

### 1.2 Station Selection
- **Target:** USW00094728 (Central Park, NYC)
- **Surrounding stations:** 14 NOAA ASOS/AWOS airport stations, 5–140 miles from Central Park
- Directional coverage: N (Albany, Poughkeepsie, White Plains), NE (Hartford, Bridgeport), E (Islip, LaGuardia), SE (JFK), S (Atlantic City), SW (Trenton, Philadelphia), W (Allentown, Scranton, Newark)
- All stations stored in `data/stations.csv` with coordinates, distances, and directions

### 1.3 Data Collection (`src/data_collection.py`)
- Downloads `.dly` files from NOAA GHCN bulk server (no API token required)
- Parses fixed-width GHCN format (21-char header + 31 x 8-char daily values)
- Handles missing values (-9999), quality flag filtering, leap years
- Converts tenths-of-°C to °F: `(value / 10) * 9/5 + 32`
- Filters to date range 2018-01-01 to 2022-12-31 (5 years)
- **All 15 stations downloaded successfully** — data quality is excellent

### 1.4 Data Preprocessing (`src/data_preprocessing.py`)
- Merges station CSVs into wide DataFrame (1826 rows × 30 columns)
- Creates target: NYC TMAX at day t
- Creates lag-1 features: surrounding station TMAX/TMIN from day t-1
- Adds cyclical date encoding: sin(2π·doy/365.25), cos(2π·doy/365.25)
- Forward-fills gaps ≤ 3 days; imputes remaining NaNs with training-set means
- Chronological split (no shuffling): Train 1277 rows / Val 274 / Test 274
- StandardScaler fit on training data only — no data leakage
- Generates `preprocessing_report.txt` with full data quality summary

### 1.5 Testing
- **61 tests total — all passing**
- `test_data_collection.py` (29 tests): temperature conversion, .dly parsing, quality flags, date filtering, download mocking, end-to-end
- `test_data_preprocessing.py` (32 tests): merging, lag shift, cyclical encoding, chronological split, scaler, missing data, completeness, target isolation, save/load, integration

## Data Quality Summary
| Station | TMAX Completeness | TMIN Completeness |
|---------|------------------|------------------|
| Central Park (target) | 100.0% | 100.0% |
| Most surrounding stations | 100.0% | 100.0% |
| Scranton (lowest) | 99.8% | 99.4% |

All stations well above the 90% completeness threshold. Zero stations dropped.

## Output Artifacts
- `data/processed/features_{train,val,test}.csv` — Scaled feature matrices (30 features each)
- `data/processed/target_{train,val,test}.csv` — Target vectors (NYC TMAX in °F)
- `data/processed/scaler.pkl` — Fitted StandardScaler for inverse transforms
- `data/processed/preprocessing_report.txt` — Data quality report

## Training Set Target Statistics
- Mean: 62.6°F | Std: 17.8°F | Range: 13.1°F – 98.1°F | Median: 63.0°F

## Risks and Notes
- Data quality was excellent for this 5-year period — no stations needed to be dropped
- The pipeline is designed to handle missing data gracefully if quality degrades in Phase 6 (25-year extension)
- Raw `.dly` files are ~1-6 MB each; excluded from git via `.gitignore`

## Next Phase
Phase 2: Baseline Models — Implement persistence, climatological average, and linear/ridge regression benchmarks.
