# Phase 4.3 -- Station Expansion: Project Manager Report

**Date:** 2026-02-08
**Phase:** 4.3 -- Station Expansion
**Status:** COMPLETE

---

## Objective

Expand the surrounding station set from 14 to ~50 stations organized by distance ring and compass sector, implement missingness masking, and run station-count sensitivity experiments.

## Deliverables

### Source Code

| File | Purpose | Size |
|------|---------|------|
| `config_expanded.py` | 50-station config with ring/sector metadata, imports from config.py | 23 KB |
| `src/station_discovery.py` | GHCN inventory download, Haversine filtering, station selection | 17 KB |
| `src/station_registry.py` | Station query API: by count, radius, ring, sector, predefined subsets | 12 KB |
| `src/data_preprocessing_expanded.py` | Expanded preprocessing with missingness masking, variable station counts | 34 KB |
| `run_phase4_expanded.py` | Rerun Phase 4 with all 50 stations | 19 KB |
| `run_station_sensitivity.py` | Station count sensitivity: 5, 10, 14, 20, 30, 40, 50 stations | 16 KB |
| `data/stations_expanded.csv` | Expanded station metadata (51 rows incl. target) | CSV |

### Tests

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_station_registry.py` | 93 | PASSED |
| `tests/test_data_preprocessing_expanded.py` | 34 | PASSED |
| **New Phase 4.3 total** | **127** | **ALL PASSED** |
| **Full project total** | **652** | **ALL PASSED** |

---

## Station Expansion Summary

### Geographic Coverage
- **51 total stations** (1 target + 50 surrounding)
- **Ring 1 (0-50 mi):** 12 stations — near-field urban/suburban
- **Ring 2 (50-100 mi):** 20 stations — regional coverage
- **Ring 3 (100-150 mi):** 11 stations — extended regional
- **Ring 4 (150-250 mi):** 7 stations — far-field
- **All 14 original stations included** in expanded set

### Sector Coverage
- **N/NE:** Albany, Hartford, Bridgeport, Danbury, New Haven, Springfield, etc.
- **E/SE:** Islip, JFK, Shirley, Farmingdale, Atlantic City, etc.
- **S/SW:** Philadelphia, Trenton, Doylestown, Reading, Allentown, etc.
- **W/NW:** Scranton, Allentown, Caldwell, Sussex, Aeroflex-Andover, etc.

---

## Phase 4 Rerun Results (50 Stations)

### Expanded vs 14-Station Comparison

| Model | 14 Stations MAE | 50 Stations MAE | Change |
|-------|----------------|----------------|--------|
| NN Raw+MSE | 4.30 | 8.61 | +4.31 (worse) |
| NN Raw+Huber | 4.36 | 8.22 | +3.86 (worse) |
| NN Delta+Huber (no AR) | 4.03 | 5.10 | +1.07 (worse) |
| **NN Delta+Huber+AR** | **3.95** | **4.85** | **+0.90 (worse)** |
| NN Delta+Huber+Full | 4.15 | N/A | Failed (363 features) |

### Station Count Sensitivity

| n_stations | MAE (F) | RMSE | R2 | n_features | n_params |
|-----------|---------|------|-----|-----------|----------|
| 5 | 4.87 | 6.39 | 0.834 | 23 | 3,649 |
| **10** | **4.62** | **6.08** | **0.850** | **43** | **4,929** |
| 14 | 4.67 | 6.09 | 0.849 | 59 | 16,001 |
| 20 | 4.69 | 6.04 | 0.852 | 83 | 19,073 |
| 30 | 4.67 | 5.92 | 0.858 | 123 | 64,769 |
| 40 | 4.86 | 6.27 | 0.841 | 163 | 75,009 |
| 50 | 4.91 | 6.48 | 0.830 | 203 | 85,249 |

---

## Key Findings

### 1. More stations hurt with limited data
With only 1,277 training samples, increasing from 14 to 50 stations degrades all models. The feature-to-sample ratio becomes unfavorable: 202 features / 1,277 samples = 0.16 features per sample.

### 2. Optimal station count is ~10-14
The sensitivity analysis shows MAE is minimized at 10 stations (MAE=4.62). Performance degrades beyond ~20 stations. This is consistent with the Phase 4 finding that 79 enhanced features overfitted.

### 3. Delta-T models are most robust to expansion
Raw TMAX models collapsed with 50 stations (MAE > 8.0), while delta-T models degraded more gracefully (MAE ~4.9-5.1). The delta-T formulation provides inherent regularization.

### 4. Phase 6 (25 years) is critical
The expanded station infrastructure is in place and ready. With ~7,000+ training samples from 25 years of data, the expanded stations should become beneficial. The sensitivity curve should shift: more stations should help rather than hurt.

### 5. The 14-station MAE=3.95 result stands as current best
The original Phase 4 NN Delta+Huber+AR with 14 stations remains the best configuration.

---

## Infrastructure Ready for Phase 6

The following is now in place for the 25-year scale-up:
- `config_expanded.py` — 50 stations with full metadata
- `src/station_registry.py` — flexible station subsetting
- `src/data_preprocessing_expanded.py` — handles variable station counts with missingness masking
- `run_station_sensitivity.py` — automated sensitivity sweeps
- All data downloaded and validated

---

## Risks & Recommendations

1. **Rerun sensitivity after Phase 6** — with 25 years of data, the optimal station count will likely shift upward
2. **Consider regularization** — L2 penalty, dropout, or feature selection may help with high station counts
3. **Station attention model** — the StationAttentionModel from Phase 4 may be particularly suited to variable station counts
4. **Feature selection** — sector averages and PCA could compress the expanded station features more effectively than raw per-station inputs
