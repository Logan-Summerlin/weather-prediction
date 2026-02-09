# Phase 0 Data Scale-Up (Multi-Source Ingestion)

This report documents the Phase 0 implementation status, artifacts, and how to run the full data scale-up pipeline.

## Scope (Phase 0 Tasks)

1. **ASOS hourly downloads (1998–2024)** for all mapped stations.
2. **Daily ASOS aggregation** for TMAX/TMIN/mean temp, dewpoint, wind, SLP, and cloud fraction.
3. **ASOS vs GHCN TMAX cross-check** for overlapping dates.
4. **IGRA soundings (00Z/12Z, 2000–2024)** for OKX/Upton.
5. **NWP downloads** (GEFSv12 reforecast 2000–2019 + operational GFS/GEFS 2021–present).
6. **Expanded GHCN parsing** with PRCP/SNOW/SNWD/AWND elements.

## Implementation Summary

### ASOS data download
- **Script:** `src/asos_collection.py`
- **Runner:** `run_phase0.py` (`--skip-asos` to disable)
- **Raw output:** `data/raw/asos/`

### ASOS daily aggregation
- **Script:** `src/asos_preprocessing.py`
- **Runner:** `run_phase0.py` (`--skip-asos-aggregate` to disable)
- **Output:** `data/processed/asos_daily/{station_id}_asos_daily.csv`
- **Computed fields:** `tmax_f`, `tmin_f`, `tmean_f`, `dewpoint_mean_f`,
  `dewpoint_afternoon_f` (18–23Z), `wind_speed_mean_mph`, `wind_speed_max_mph`,
  `wind_dir_mean_deg`, `wind_dir_evening_deg`, `slp_00z_mb`, `slp_12z_mb`,
  `slp_tendency_24h_mb`, `cloud_fraction_low`, `obs_count`.

### ASOS vs GHCN cross-check
- **Script:** `src/asos_preprocessing.py`
- **Runner:** `run_phase0.py` (use `--skip-asos-ghcn-report` to disable)
- **Outputs:**
  - `reports/asos_ghcn_tmax_comparison.csv`
  - `reports/asos_ghcn_tmax_comparison.md`

### IGRA soundings
- **Script:** `src/soundings_collection.py`
- **Runner:** `run_phase0.py` (`--skip-igra` to disable)
- **Raw output:** `data/raw/igra/`

### NWP downloads
- **Script:** `src/nwp_collection.py`
- **Runner:** `run_phase0.py` (`--skip-nwp` to disable)
- **Raw output:** `data/raw/nwp/`

### GHCN expanded elements
- **Script:** `src/data_collection.py`
- **Elements:** `TMAX`, `TMIN`, `PRCP`, `SNOW`, `SNWD`, `AWND`.

## How to run Phase 0 end-to-end

```bash
python run_phase0.py \
  --asos-chunk-years 1 \
  --nwp-model gfs \
  --nwp-fxx 24
```

Run the GEFS reforecast separately if needed:

```bash
python run_phase0.py \
  --skip-asos --skip-igra \
  --nwp-model gefs_reforecast \
  --nwp-member 0
```

## Notes & Assumptions

- ASOS daily aggregation currently uses **UTC day boundaries** and the 18–23Z window for afternoon/evening features, matching the plan’s UTC-based feature definitions.
- GHCN data is training-only and used here solely for **cross-checking ASOS-derived TMAX** (no live-time leakage).
- NWP downloads require `herbie-data`; IGRA downloads require `siphon`.
