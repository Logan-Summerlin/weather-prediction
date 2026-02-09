# Kalshi Real-Data Backtesting: Diagnostic Report & Fixes

**Date:** 2026-02-09
**Author:** Project Manager
**Status:** Issues identified, fixes pending

---

## 1. Issue: Model Trained on Only 3.5 Years of Data

### What Happened

The analyst hard-coded `start_date="2018-01-01"` in `scripts/generate_real_predictions.py:68` and `run_kalshi_real_oos.py:257`, ignoring the project's `config.py` which specifies `START_DATE = "1985-01-01"` (40 years). The actual training splits used were:

| Split | Date Range | Samples | Duration |
|-------|-----------|---------|----------|
| **Train** | 2018-01-02 to 2021-12-31 | ~1,460 | 4 years |
| **Validation** | 2022-01-01 to 2022-12-31 | ~365 | 1 year |
| **IS Prediction** | 2023-01-01 to 2024-12-31 | 731 | 2 years |
| **OOS Prediction** | 2025-01-01 to 2025-12-31 | 365 | 1 year |

### Available Data (Not Used)

The raw GHCN `.dly` files in `data/raw/` already contain far deeper history:

| Station | Available Range | Years |
|---------|----------------|-------|
| Central Park (USW00094728) | 1900 - 2026 | 126 |
| Newark (USW00014734) | 1900 - 2026 | 126 |
| Albany (USW00014735) | 1938 - 2026 | 88 |
| Islip (USW00014732) | 1939 - 2026 | 87 |
| Hartford (USW00014740) | 1949 - 2026 | 77 |
| Atlantic City (USW00093730) | 1958 - 2026 | 68 |
| Poughkeepsie (USW00014757) | 1957 - 2026 | 69 |

All 14 surrounding stations have 60+ years of data. The bottleneck station determines the common window start.

### Impact

- **Samples per feature:** With 1,460 train samples and 30 features, the ratio is ~49:1. With 25 years (1999-2024), it would be ~300:1. With 40 years (1985-2024), ~480:1.
- **Missed climate variability:** The 2018-2021 window is only 4 years — it may not capture El Nino/La Nina cycles, rare cold outbreaks, or multi-year warming trends.
- **Sigma estimates unreliable:** The monthly sigma values were estimated from only 1 year of validation data (~30 samples per month). With 5+ years of validation, these become much more stable.
- **Phase 6 was skipped:** The project plan explicitly requires "Scale Up — Extend data to 25 years" before running real-money strategies. This was never done.

### Fix Required

In `scripts/generate_real_predictions.py`:

```python
# Line 68: CHANGE FROM
def build_features_and_targets(raw_dir, start_date="2018-01-01", end_date="2024-12-31"):

# TO (use config.py)
def build_features_and_targets(raw_dir, start_date=None, end_date="2024-12-31"):
    if start_date is None:
        import config
        start_date = config.START_DATE  # "1985-01-01"
```

In `scripts/generate_real_predictions.py`:

```python
# Lines 153-154: CHANGE FROM
train_end = date(2021, 12, 31)
val_end = date(2022, 12, 31)

# TO (use proportional splits over the full date range, or explicit)
train_end = date(2019, 12, 31)   # 35 years of training
val_end = date(2022, 12, 31)     # 3 years of validation
# prediction: 2023-2024 as before
```

In `run_kalshi_real_oos.py`:

```python
# Line 257: CHANGE FROM
all_data, feature_cols = build_station_features("2018-01-01", "2025-12-31")

# TO
all_data, feature_cols = build_station_features(config.START_DATE, "2025-12-31")
```

The train/val split should also be adjusted for the longer window. A reasonable split for a 1985-2024 dataset:

| Split | Date Range | Samples | Purpose |
|-------|-----------|---------|---------|
| Train | 1985-01-01 to 2019-12-31 | ~12,775 | 35 years |
| Validation | 2020-01-01 to 2022-12-31 | ~1,095 | 3 years (for sigma + early stopping) |
| IS Prediction | 2023-01-01 to 2024-12-31 | 731 | In-sample backtest |
| OOS Prediction | 2025-01-01 to 2025-12-31 | 365 | Out-of-sample test |

---

## 2. Issue: Market Probability Proxy Is Too Naive

### What Happened

The Kalshi public API for settled markets returns only settlement prices (0 or 100 cents), not the historical pre-settlement trading prices. To fill this gap, the analysts used a **climatological + persistence Gaussian model** as the market proxy:

```
Market proxy: 40% × yesterday's TMAX + 60% × monthly climatological mean
```

This proxy is far less informed than what Kalshi markets actually price. Real Kalshi participants have access to NWS forecasts, ensemble model output, and professional weather services. Comparing our model against this naive proxy overstates our edge.

### Evidence of Overstated Edge

| Metric | Against Naive Proxy | Against Settlement Prices |
|--------|-------------------|--------------------------|
| Model Brier | 0.0243 | 0.1785 |
| Market Brier | 0.0331 | 0.0250 |
| Brier Delta | -0.0088 (model better) | +0.1535 (market better) |
| Profitable strategies | Many | 0 out of 448 |

The naive proxy makes the model look good because the proxy is weak. Against actual market pricing, the model shows no edge.

### What We Should Use Instead

The Kalshi KXHIGHNY market is almost certainly priced off public weather forecasts. The best proxies for market pricing, in order of quality:

#### Tier 1: IEM MOS Archive for KNYC (Primary)

The Iowa Environmental Mesonet archives NWS Model Output Statistics for Central Park (station code: KNYC). These are post-processed NWP model forecasts calibrated against local observations — exactly what market participants use.

- **Models available:** GFS MOS, NAM MOS (since June 2007), NBS/NBE (since 2020)
- **API (free, no auth):**
  ```
  https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2024-01-15%2012:00Z
  ```
  ```
  https://mesonet.agron.iastate.edu/mos/fe.phtml  # Bulk download interface
  ```
- **Fields:** Max temperature forecast (`N/X`), min temp, hourly temps, precip prob
- **Archive depth:** June 2007+ for GFS/NAM MOS; 2020+ for NBS/NBE
- **How to use:** Pull the max-temp forecast from the 12Z or 13Z model run the day before (matches when Kalshi markets open at 10 AM ET). Compute historical forecast error distribution by season. Assume `MaxT ~ N(MOS_forecast, sigma_seasonal)` and compute bracket probabilities.

#### Tier 2: NBM Probabilistic Products (Best for Distributions)

The NWS National Blend of Models provides actual probability distributions for MaxT — percentiles (10th, 25th, 50th, 75th, 90th) and standard deviation.

- **Archive:** AWS S3 `s3://noaa-nbm-grib2-pds` (GRIB2, May 2020+)
- **Text products:** IEM archives NBS/NBP bulletins for KNYC going back to 2020
- **How to use:**
  ```python
  # From NBM percentiles, fit a distribution:
  mean = P50
  sigma = (P90 - P10) / 2.56
  # For each Kalshi bracket [L, U]:
  P(L <= MaxT < U) = norm.cdf(U, mean, sigma) - norm.cdf(L, mean, sigma)
  ```
- **Access via Herbie:**
  ```python
  from herbie import Herbie
  H = Herbie("2024-01-15 12:00", model="nbm", fxx=24, product="co")
  ```

#### Tier 3: Open-Meteo Previous Runs + Ensemble API (2024+ only)

For recent data, Open-Meteo archives what models predicted before the fact.

- **Previous Runs API (point forecasts):**
  ```
  https://previous-runs-api.open-meteo.com/v1/forecast?latitude=40.78&longitude=-73.97&daily=temperature_2m_max&previous_day=1&models=gfs_seamless&timezone=America/New_York&temperature_unit=fahrenheit
  ```
- **Ensemble API (probability distributions):**
  ```
  https://ensemble-api.open-meteo.com/v1/ensemble?latitude=40.78&longitude=-73.97&hourly=temperature_2m&models=gfs_seamless&temperature_unit=fahrenheit
  ```
  Returns 31 GFS ensemble members; compute mean/spread for bracket probabilities.
- **Limitation:** GFS temp data from March 2021+; most models from January 2024+.

#### Sources NOT Useful

| Source | Why Not |
|--------|---------|
| NWS `api.weather.gov` | No historical archive; current forecasts only |
| NDFD | Superseded by NBM; no temperature probability fields |
| CPC Outlooks | Multi-day/seasonal scale, not daily |
| Weather Underground | Observations only; no forecast archive |

### Recommended Market Proxy Architecture

For the 2023-2024 in-sample backtest period:

```
Layer 1: IEM MOS point forecasts (GFS MOS + NAM MOS + NBS for KNYC)
    ↓ Average or regression-weight the point forecasts
Layer 2: Estimate sigma from historical MOS errors (by month/season)
    ↓ Fit N(mu, sigma) per day
Layer 3: Compute P(L <= MaxT < U) for each Kalshi bracket
    ↓ These are the "market-implied probabilities"
Layer 4: Compare our model's bracket probs vs market-implied probs
    ↓ If delta > EV threshold, trade
```

For 2025 OOS period (if NBM available):

```
Use NBM probabilistic MaxT percentiles directly
    ↓ Fit distribution from (P10, P25, P50, P75, P90)
    ↓ Compute bracket probabilities
    ↓ Compare against our model
```

### Fix Required

1. **Download IEM MOS archive** for KNYC (GFS MOS + NAM MOS, 2007-2025; NBS 2020-2025)
2. **Download NBM probabilistic archive** for KNYC (2020-2025) from AWS S3 or IEM text bulletins
3. **Build a new market proxy** from these real forecast data sources
4. **Re-run the Kalshi backtest** using the NWS-based market proxy instead of the naive climatological proxy
5. **Compare:** Our model's bracket probabilities vs NWS forecast-derived bracket probabilities

---

## 3. Summary of All Required Fixes

| # | Fix | Priority | Estimated Impact |
|---|-----|----------|-----------------|
| 1 | Expand training data from 4 years to 25-40 years | **HIGH** | 5-6x more training samples; better generalization |
| 2 | Replace naive market proxy with IEM MOS / NBM forecasts | **HIGH** | Honest assessment of model edge vs actual market information |
| 3 | Re-estimate monthly sigma with larger validation set (3 years) | MEDIUM | More stable uncertainty estimates |
| 4 | Consider adding more surrounding stations (from expanded config) | LOW | May improve with more training data to support them |
| 5 | Re-run full backtesting pipeline after fixes 1-2 | **HIGH** | Only results after these fixes are trustworthy |

### Expected Outcome After Fixes

- If the model still shows a Brier advantage over NWS MOS forecasts, it has **genuine edge** worth trading
- If the model's Brier score matches or is worse than NWS MOS, the model adds no value over free public forecasts
- The honest answer may be sobering — NWS MOS forecasts are highly calibrated and hard to beat for 1-day-ahead MaxT
- Even a small edge (0.01 Brier improvement) can be profitable if sizing is correct

---

## 4. File Reference

| File | Issue | Action |
|------|-------|--------|
| `scripts/generate_real_predictions.py:68` | Hard-coded `start_date="2018-01-01"` | Change to `config.START_DATE` |
| `scripts/generate_real_predictions.py:153-154` | Hard-coded `train_end=2021, val_end=2022` | Use proportional split over full date range |
| `run_kalshi_real_oos.py:257` | Hard-coded `"2018-01-01"` | Change to `config.START_DATE` |
| `run_kalshi_real_backtest.py` | Uses naive climatological market proxy | Replace with IEM MOS / NBM proxy |
| `run_kalshi_real_oos.py` | Uses naive climatological market proxy | Replace with IEM MOS / NBM proxy |
| `config.py:45-46` | `START_DATE = "1985-01-01"` | Verify station completeness for this range |

---

## 5. Data Sources for NWS Market Proxy

### IEM MOS API Examples

```bash
# GFS MOS for KNYC, specific runtime
curl "https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2024-01-15%2012:00Z"

# Bulk download page for CSV export
# https://mesonet.agron.iastate.edu/mos/fe.phtml
# Select: Station=KNYC, Model=GFS/NAM/NBS, Start=2007-06-01, End=2025-12-31
```

### NBM on AWS S3

```bash
# List available NBM data
aws s3 ls --no-sign-request s3://noaa-nbm-grib2-pds/

# Download a specific NBM run (GRIB2)
aws s3 cp --no-sign-request s3://noaa-nbm-grib2-pds/blend.20240115/12/core/blend_nbmtx.t12z.grib2 .
```

### Open-Meteo (for 2024+ supplementary data)

```bash
# Historical GFS forecast for NYC from 1 day before
curl "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=40.78&longitude=-73.97&daily=temperature_2m_max&previous_day=1&models=gfs_seamless&timezone=America/New_York&temperature_unit=fahrenheit&start_date=2024-01-01&end_date=2024-12-31"

# GFS ensemble spread
curl "https://ensemble-api.open-meteo.com/v1/ensemble?latitude=40.78&longitude=-73.97&daily=temperature_2m_max&models=gfs_seamless&temperature_unit=fahrenheit&timezone=America/New_York"
```
