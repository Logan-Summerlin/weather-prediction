# NYC Daily Max Temperature Prediction — Market-Beating Project Plan

## 1. Research Concept Summary

**Objective:** Build a **calibrated probabilistic forecast** for NYC’s daily maximum temperature (°F) and use it to beat the **Kalshi KXHIGHNY** prediction market. The system must generate bucket probabilities that are better calibrated than the market consensus and actionable for trading.

**Core hypothesis:** A hyper-local, station-based model trained on decades of NYC-area observations can detect local biases and regimes better than raw NWP. **Synthesizing** that station model with NWP forecasts yields a calibrated distribution that beats either source alone.

**Primary success metrics (forecasting):**
- **MAE ≤ 2.0–2.3°F** on a 2020–2024 holdout.
- **CRPS ≤ 1.8–2.0°F** for distribution quality.
- **Calibration:** PIT histogram ≈ uniform; 90% interval coverage 88–92%.
- **Kalshi bucket performance:** Brier score per bucket < 0.15 and positive expected value (EV) decisions after fees.

**Operational constraint:** All data used for day-*t* prediction must be available by **6:00 AM ET** on day *t*. This drives the data-source and feature selection.

---

## 2. Operational Data Availability (6 AM ET Constraint)

**Key constraint:** GHCN-Daily and ERA5 are **not** available by 6 AM ET and are therefore **training-only**. Operational inference must use **IEM ASOS hourly data**, **IGRA soundings**, and **GFS/GEFS** forecasts that are available overnight.

**Operationally available by 6 AM ET:**
- IEM ASOS hourly surface observations (5–10 minute latency).
- IGRA 00Z sounding for OKX/Upton (available late evening).
- GFS 00Z forecast (F024) for day-*t* TMAX and ancillary variables.
- GEFS ensemble spread (00Z) for uncertainty estimation.

**Training-only sources:**
- GHCN-Daily `.dly` (1–2 day lag).
- ERA5/ERA5T (multi-day lag).

---

## 3. Data Sources & Roles

### 3.1 Station Observations (Primary, Operational)
**IEM ASOS hourly data** (Iowa Environmental Mesonet) is the **primary operational observation source**.
- **URL:** `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- **Stations:** 14 core NYC-area airports (Central Park + surrounding).
- **Key variables:** temp, dewpoint, wind direction/speed, sea-level pressure, cloud cover/ceilings, visibility.
- **Daily aggregation (t−1 features):**
  - TMAX/TMIN/mean temperature
  - Prevailing wind direction (vector mean), evening wind direction (18–00Z)
  - Mean/max wind speed
  - Mean dewpoint, afternoon dewpoint
  - SLP at 00Z/12Z and 24h tendency
  - Cloud fraction (hours with ceiling < 5000 ft)

**ASOS station coverage for the 52-station network:**
- Use the ICAO mapping in `data/asos_station_mapping.csv` and `config_expanded.ASOS_STATION_MAP`.
- Filter the training set to **ASOS-available stations only** and document any replacements for non-ASOS sites.

### 3.2 GHCN-Daily (Training-only, Supplementary)
**GHCN-Daily** remains valuable for long historical temperature records and snow/precipitation signals.
- Expand `.dly` parsing to include **PRCP, SNOW, SNWD, AWND**.
- Use GHCN for **training features** and **cross-checking** IEM-derived TMAX.

### 3.3 Upper-Air Soundings (Operational)
**IGRA soundings** via Siphon for OKX/Upton (USM00072501).
- Extract **850mb temperature, wind, stability** (T850 − surface).
- 00Z sounding available before 6 AM ET (use for day-*t* forecast).

### 3.4 NWP Forecasts (Synthesis Layer)
Use **GEFSv12 Reforecast (2000–2019)** for training and **operational GFS/GEFS (2021–present)** for live inference.
- Variables: TMAX, TMIN, T850, 10m wind, cloud cover, MSLP, precipitation, ensemble spread.
- Use **Herbie** or AWS S3 for partial downloads of NYC grid point.

### 3.5 ERA5 Reanalysis (Research-Only)
ERA5 is valuable for exploratory analysis but **must not be used for operational inference** due to its multi-day latency and analysis/forecast mismatch. Use it only for diagnostics or feature validation.

---

## 4. Data Source: NOAA GHCN-Daily (Training-only Details)

The **Global Historical Climatology Network — Daily (GHCNd)** is the ideal **training** data source. It is free, quality-controlled, and provides daily TMAX, TMIN, and other variables for thousands of U.S. stations going back over 100 years. **It is not available by 6 AM ET**, so it must not be used for operational inference.

### Key Details

- **Variables available:** TMAX (daily max temp), TMIN (daily min temp), TAVG (where available), PRCP, SNOW, SNWD, AWND, WDF2/WSF2 (variable availability)
- **Format:** Values in tenths of °C (divide by 10 to get °C, then convert to °F)
- **Access methods:**
  - Bulk download: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/`
  - NOAA CDO API v2: `https://www.ncei.noaa.gov/cdo-web/api/v2/` (requires free API token)
  - Python library: `noaa_sdk` or direct HTTP requests
- **Station metadata file:** `ghcnd-stations.txt` — lists station ID, lat, lon, elevation, name

### API Token

Sign up at `https://www.ncdc.noaa.gov/cdo-web/token` for a free NOAA CDO API token. Rate limit: 5 requests/second, 10,000 requests/day.

---

## 5. Station Selection Strategy

### Target Station (NYC)

Use the **Central Park** weather station, which is the official NYC climate station and has one of the longest continuous records in the U.S.

- **Station ID:** `USW00094728` (NY CITY CENTRAL PARK)
- **Coordinates:** 40.7789°N, 73.9692°W
- **Record:** 1869–present

### Surrounding Input Stations

Select 15–25 stations within approximately 50–200 miles of Central Park, covering all compass directions. This captures incoming weather systems from all sides. Prioritize stations with long, complete records.

> **Implementation status:** Station geography analysis is complete. The base network of 50 surrounding stations has been expanded to ~52 surrounding stations (~53 total including the target) based on gap-filling recommendations:
> - **S near-field gap (Priority 1):** Added a station near McGuire-Dix-Lakehurst, NJ (~50mi, bearing ~195 deg) to provide Ring1/Ring2 coverage in the previously empty S sector near-field.
> - **SW Ring3 gap (Priority 2):** Added a station near Dover AFB, DE (~130mi, bearing ~210 deg) to bridge the 90mi gap between Philadelphia (91mi) and Ocean City (181mi) in the SW meteorological corridor.
> - **ESE transition zone (Priority 3):** Research confirmed no suitable GHCN station exists. The 36-degree gap between Farmingdale (96 deg) and JFK (132 deg) on Long Island's south shore cannot be filled — all historical stations in the Babylon/Lindenhurst/Wantagh area (e.g., Mineola USC00305377, Wantagh Cedar Creek USC00308946) ceased TMAX reporting by 2011 or earlier. No USW airport stations are active in this zone.
> - **SSE ocean gap (Priority 4):** Accepted as an irreducible geographic constraint (62-degree gap over the Atlantic).
>
> See `config_expanded.py`, `src/station_registry.py`, and `reports/station_geography_report.md` for the full station list and analysis.

**Directional coverage guidance:** Keep stations from all sides, but **intentionally include a few extra candidates to the W/NW and SW** (interior/upstream sectors in the Mid-Atlantic) while retaining a small set of coastal/onshore proxies to capture Atlantic moderation regimes. This is *not* hard-coding weights—just ensuring the candidate set has enough upstream coverage and redundancy.


**Recommended stations (to be confirmed during data exploration):**

| Direction | Station / Area | Approx. Distance |
|-----------|---------------|-------------------|
| North | Poughkeepsie, NY | ~75 mi |
| North | Albany, NY | ~140 mi |
| NE | Hartford, CT | ~110 mi |
| NE | Bridgeport, CT | ~55 mi |
| East | Islip/Long Island, NY | ~45 mi |
| SE | Atlantic City, NJ | ~120 mi |
| South | Trenton, NJ | ~70 mi |
| SW | Philadelphia, PA | ~95 mi |
| SW | Allentown, PA | ~85 mi |
| West | Scranton, PA | ~120 mi |
| West | Morristown, NJ | ~30 mi |
| NW | Danbury, CT | ~65 mi |
| NW | Monticello, NY | ~90 mi |
| North | Westchester/White Plains, NY | ~25 mi |
| South | Sandy Hook, NJ | ~25 mi |
| East | Bridgehampton, NY | ~95 mi |

**Add-on upstream candidates (optional, for redundancy):** consider adding 2–4 more stations from these sectors (to be confirmed in exploration):
- **W/NW (upstream cold-air advection candidates):** Binghamton NY; Wilkes-Barre/Scranton PA area; State College PA; Harrisburg PA.
- **SW (warm-advection / pre-frontal candidates):** Harrisburg PA; Baltimore–Washington corridor; Lancaster/York PA.
- **Coastal/onshore proxies (Atlantic moderation):** JFK/LaGuardia/Newark area airports; coastal NJ / Long Island shore stations.

When you evaluate feature importance later, group stations by sector (W/NW, SW, coastal, near-field) to see if learned signal matches expectations.


**Selection criteria:**
1. Station has TMAX data for ≥ 90% of days in the study period
2. Station is within 200 miles of Central Park
3. Stations distributed across compass directions (not clustered)
4. Prefer ASOS/AWOS airport stations for data quality

**Operational ASOS alignment (required):**
- Every station in the expanded 52-station set must map to an **ASOS/AWOS ICAO** identifier for operational training/inference.
- The authoritative mapping lives in:
  - `data/asos_station_mapping.csv`
  - `config_expanded.ASOS_STATION_MAP`
- Stations without confirmed ASOS/AWOS coverage (e.g., Millbrook 3 W, Kingston 1 W, Atlantic City Marina, Heritage Field) must be **excluded from the operational training set** or replaced with the nearest ASOS station.

---

## 6. Data Collection Period

### Phase 1 (Proof of Concept): 5 years
- **Period:** 2018-01-01 to 2022-12-31
- **Purpose:** Validate the pipeline, train an initial model, assess feasibility
- **Approx. rows per station:** ~1,826

### Phase 2 (Station Model Scale-Up): 26 years
- **Period:** 1998-01-01 to 2024-12-31 (IEM ASOS availability)
- **Purpose:** Train a robust station-based model with full operational feature parity
- **Approx. rows per station:** ~9,500

### Phase 3 (Synthesis Training Window): 24 years
- **Period:** 2000-01-01 to 2024-12-31
- **Purpose:** Train synthesis layer using GEFSv12 reforecast (2000–2019) and operational GFS/GEFS (2021–2024)

### Data Completeness Findings (GHCN Training-Only)
- **15 original stations:** Mostly 99–100% complete over the full 40-year period.
- **30 expanded stations:** Below 80% completeness (many started reporting circa 1997–2000).
- **21 stations total** meet the ≥80% completeness threshold over the 40-year range.
- Use these stats for **training-only** station selection; operational features come from ASOS.

---

## 7. Neural Network Architecture

### 7.1 Baseline: Simple Feedforward Network

Start simple. This is essentially a learned weighted-average problem.

```
Input Layer:  N neurons (one per surrounding station's TMAX at t-1)
     |
Hidden Layer 1: 32 neurons, ReLU activation
     |
Hidden Layer 2: 16 neurons, ReLU activation
     |
Output Layer: 1 neuron (predicted NYC TMAX at day t), linear activation
```

**Loss function:** Mean Squared Error (MSE)
**Optimizer:** Adam (lr=0.001)

### 7.2 Enhanced: Multi-Feature Input and Target Formulation

#### 7.2a ΔT (Delta) Target — Highest-Leverage Change

Switch the prediction target from raw TMAX(t) to the daily change:

```
ΔT(t) = TMAX_NYC(t) − TMAX_NYC(t−1)
```

Then reconstruct the absolute prediction:

```
TMAX_NYC_pred(t) = TMAX_NYC(t−1) + ΔT_pred(t)
```

**Why:** Persistence is already strong (MAE ≈ 5°F). Predicting deltas forces the model to learn the hard part — regime shifts and advection events — rather than re-learning climatology and persistence. Empirically, delta targets narrow the output range and stabilize training.

**Implementation:**
- Create target column `y = TMAX_NYC[t] − TMAX_NYC[t−1]`
- Include `TMAX_NYC[t−1]` as an input feature
- Train model to predict ΔT
- Evaluate MAE on reconstructed TMAX (not on ΔT itself)

**Training loss:** Use **Huber (SmoothL1)** or **MAE** loss (often better than MSE for MAE optimization). Track MAE for early stopping.

#### 7.2b Per-Station Feature Expansion

Instead of one feature per station, use multiple features per station:

```
Per station (at t-1):
  - TMAX (max temperature)
  - TMIN (min temperature)
  - TMAX - TMIN (diurnal range)

Total input features: N_stations × 3

Additional inputs:
  - Day of year (encoded as sin/cos for cyclical nature)
  - NYC's own TMAX at t-1 (autoregressive term)
```

#### 7.2c Sector Averages and Gradients (Physics-Shaped Features)

Group stations into directional sectors and compute aggregate features that proxy frontal passages and air-mass advection:

**Sector definitions:**
- W/NW (upstream cold-air advection): Scranton, Allentown, Albany, Poughkeepsie
- SW (warm advection / pre-frontal): Philadelphia, Trenton
- Coastal/E/SE (Atlantic moderation): Islip, JFK, Atlantic City, Bridgeport
- Near-field (urban/local): Newark, LaGuardia, White Plains

**Sector features:**
- `mean_WNW(t−1)`, `mean_SW(t−1)`, `mean_coast(t−1)` — sector mean temperatures
- `grad_upstream_vs_coast = mean_WNW(t−1) − mean_coast(t−1)` — cold-air advection proxy
- `grad_SW_vs_NW = mean_SW(t−1) − mean_WNW(t−1)` — warm-front proxy

**Why:** Gradients proxy "which air mass is arriving," especially in complex coastal terrain. They reduce dimensionality and increase signal-to-noise when scaling to many stations.

#### 7.2d Trend Features (Front-Timing Proxy)

Compute short differences per sector or per top stations:

```
Δ1 = T(t−1) − T(t−2)   # yesterday's change
Δ2 = T(t−2) − T(t−3)   # day-before-yesterday's change
```

These capture direction and momentum of temperature changes, helping detect approaching fronts.

#### 7.2e Static Station Metadata

Add per-station metadata as features (especially useful with attention-based architectures):
- Elevation (meters)
- Distance to Central Park (km)
- Bearing from Central Park (degrees)
- Station type flag (airport vs. other)

#### 7.2f Operational ASOS Feature Set (Highest Priority)

From **IEM ASOS hourly data**, aggregate daily features for t−1:
- Wind direction (vector mean), wind direction persistence, evening wind direction (18–00Z)
- Wind speed: mean, max, evening mean
- Dewpoint: mean, afternoon value, dewpoint depression (T − Td)
- Sea-level pressure: 00Z, 12Z, 24h tendency
- Cloud fraction proxy: fraction of hours with ceiling < 5000 ft

These features are both **high-impact** and **operationally available** by 6 AM ET.

#### 7.2g Wind-Conditioned Physics Features

Use wind direction to compute physically meaningful composites:
- **Upwind temperature:** weighted average of stations aligned with wind direction.
- **Crosswind temperature:** stations perpendicular to wind.
- **Downwind temperature:** stations opposite the wind.
- **Upwind gradient:** upwind_temp − NYC_TMAX(t−1).
- **Advection rate:** wind_speed × upwind_gradient / upwind_distance.

These features replace static station weights with **dynamic, regime-aware signals**.

#### 7.2h Upper-Air Features (IGRA Soundings)

From OKX/Upton 00Z soundings:
- **850mb temperature** (T850) — strongest single predictor of surface TMAX.
- **850mb wind direction/speed** — large-scale advection signal.
- **Stability indicator:** T850 − surface temp.

#### 7.2i Precipitation & Snow (Training-Only → Operational Bridge)

From GHCN (training) and ASOS (operations):
- PRCP(t−1), snow depth, snow depth change
- Days since last precipitation
- Snow presence binary (albedo effect)

### 7.3 Advanced: Architecture Upgrades (Attention Pooling and Temporal Models)

Test these architectures in order, stopping when gains plateau:

#### 7.3a Wind-Gated Station Attention (Primary Architecture)

Replace static station attention with wind-conditioned gating:

```
Input per station i:
  [TMAX_i, TMIN_i, diurnal_range_i, PRCP_i,
   dewpoint_i, wind_speed_i, ΔT_i(t-1 vs t-2)]

Station metadata:
  [bearing_i, distance_i, elevation_i, sector_one_hot_i]

Global context:
  [wind_dir_prevailing, wind_speed_mean, SLP, SLP_tendency,
   sin_day, cos_day, day_length, snow_depth_nyc,
   NYC_TMAX(t-1), NYC_dewpoint(t-1)]

Attention logit:
  dot(Q, K_i) + α × cos(wind_dir − bearing_i)
```

This biases attention toward **upwind** stations while allowing the model to override when the data says otherwise. It is the highest-leverage architecture upgrade after scaling to 40+ years of data.

#### 7.3b Station Embeddings + Attention Pooling (recommended for 20+ stations)

**Motivation:** With many correlated stations, flattened MLPs learn fragile weights. Attention pooling lets the model learn "which stations matter today."

```
Architecture:
1) Per-station encoder (shared weights):
   - Inputs per station: TMAX, TMIN, diurnal, trend, station metadata
   - Small MLP → station embedding (dim 16–64)
2) Attention pooling across station embeddings:
   - Mask missing stations (variable station count per day)
   - Add layer norm for stability
3) Output head: predict ΔT (or TMAX)
```

**Practical notes:**
- Keep the model small enough to avoid overfitting
- Attention pooling naturally handles missing stations (mask them out)
- Evaluate with permutation importance grouped by sector

#### 7.3c k-Day Window MLP

Concatenate features from days [t−1, …, t−k] into a single flat input vector:

```
Input: k × (N_stations × features_per_station + engineered_features)
     |
MLP with 2–3 hidden layers
     |
Output: 1 neuron (ΔT or TMAX prediction)
```

Often surprisingly strong when combined with engineered gradient/trend features. Test before adding sequence model complexity.

#### 7.3d 1D Temporal Convolution

Lightweight and stable temporal modeling:

```
Input shape: (k_days, pooled_station_features)
     |
1D Conv layers (kernel size 2–3)
     |
Dense output head
```

#### 7.3e LSTM/GRU (use only if k-window models plateau)

```
Input shape: (k_days, N_stations × features_per_station)
     |
LSTM/GRU Layer: 64 units
     |
Dense Layer: 32 neurons, ReLU
     |
Output: 1 neuron (predicted ΔT or TMAX at day t)
```

Predict ΔT with sequences of station-pooled features.

#### 7.3f Station Embeddings + Temporal Attention (Transformer-style)

Instead of treating each station as a fixed input column, learn a small **embedding per station** and use a **temporal attention / Transformer-style encoder** over the last *k* days. This can capture time-varying "which stations matter today" behavior without requiring a spatial grid.

- Inputs: (station_id, features at t-1..t-k)
- Model: station embedding + per-day feature projection → temporal attention → pooled representation → prediction head
- Evaluation: keep the same headline comparisons and permutation-importance sector checks.

### 7.4 Probabilistic Outputs & Confidence Intervals

For trading, the model must output a **full predictive distribution**, not just a point forecast.

**Preferred output heads (in order):**
1. **Gaussian Mixture (2–3 components)** — handles bimodal outcomes (front arrives vs. stalls).
2. **Heteroscedastic Gaussian** — outputs μ and σ (fastest to train).
3. **Quantile network** — 5th–95th percentiles (pinball loss).

**Training loss:** use **CRPS** or Gaussian NLL (for Gaussian outputs). CRPS is preferred because it rewards both accuracy and calibration.

**Calibration layer (post-hoc):**
- Hold out a calibration set from late years.
- Fit isotonic regression on CDF outputs (by season or regime).
- Validate PIT histogram and reliability diagram.

### 7.5 Residual Learning / Stacking Ensemble

**Core idea:** Let simple models handle the "easy majority" and let a NN learn the residual.

**Base models (fast priors):**
- Persistence: `TMAX_NYC(t−1)`
- Ridge/elastic net on station features

**Meta model (residual corrector):**
Train a small NN on:
- Base model predictions
- Engineered gradients and trends
- Cyclical day features

**Target:** TMAX residual or ΔT residual. This approach frequently reduces MAE by improving robustness, especially when the NN is tempted to overfit.

### 7.6 Season/Regime Specialization

If seasonal MAE disparity remains large after other enhancements, try specialization:

**Option A — Two seasonal models:**
- Cool season: DJF + MAM
- Warm season: JJA + SON

**Option B — Mixture-of-experts gate (single unified training):**
- Small gating network uses day-of-year sin/cos + sector gradients
- Gate blends two small expert sub-networks

**Report:** Season-wise MAE and "transition months" performance (Apr/May, Sep/Oct).

### 7.7 Synthesis Layer (Station Model + NWP = Market Edge)

Build a **separate meta-learner** that combines the station model’s distribution with NWP forecasts:

**Inputs:**
- `station_mu`, `station_sigma` (from station model)
- `nwp_tmax` (GFS/GEFS F024)
- `nwp_t850`, `nwp_wind_dir/speed`, `nwp_cloud_cover`, `nwp_mslp`
- `nwp_ensemble_spread`
- `station_nwp_gap` and `abs_station_nwp_gap`
- Recent bias features (last 7 days of station vs NWP MAE)
- Season encodings (sin/cos day-of-year)

**Output:** calibrated distribution (μ, σ or mixture parameters).

**Training data:** GEFSv12 reforecast (2000–2019) + operational GFS/GEFS (2021–present).  
**Loss:** CRPS or NLL + post-hoc isotonic calibration.

---

## 8. Implementation Plan — Step-by-Step

### Step 1: Environment Setup

```bash
pip install pandas numpy scikit-learn torch matplotlib requests
```

**Files to create:**
- `config.py` — station IDs, date ranges, API token, hyperparameters
- `data_collection.py` — download GHCN-Daily `.dly` files (training)
- `asos_collection.py` — download IEM ASOS hourly data (operational + training)
- `asos_preprocessing.py` — aggregate ASOS hourly → daily features
- `soundings_collection.py` — download IGRA soundings (OKX)
- `nwp_collection.py` — download GEFSv12 reforecast + operational GFS
- `nwp_preprocessing.py` — extract NYC grid point features
- `data_preprocessing.py` — clean, align, feature-engineer, and merge all sources
- `model.py` — PyTorch model definition
- `train.py` — training loop
- `evaluate.py` — evaluation metrics, plotting
- `predict.py` — make predictions on new data
- `synthesis_model.py` — meta-learner combining station + NWP
- `calibration.py` — isotonic calibration + PIT/reliability plots
- `kalshi_client.py` — Kalshi market data utilities
- `trading.py` — EV/Kelly sizing + execution logic

### Step 2: Data Collection (Multi-Source)

**2A) GHCN-Daily (training-only):**
> **Implemented method:** Bulk `.dly` file downloads from `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly` are used (no NOAA API token required). The fixed-width `.dly` format is parsed directly. The `run_collect_all_stations.py` script handles downloading and parsing for all 51 stations. This approach avoids all API rate-limit issues.

**2B) IEM ASOS (operational + training):**
- Download hourly observations for the 14 core airport stations (1998–present).
- Aggregate to daily features (TMAX, wind dir/speed, dewpoint, SLP, cloud fraction).
- Store raw hourly CSVs and daily aggregates.

**2C) IGRA Soundings (operational + training):**
- Download 00Z/12Z soundings for OKX/Upton (USM00072501).
- Extract T850, wind, and stability features.

**2D) NWP Forecasts for Synthesis (training + operations):**
- GEFSv12 reforecast (2000–2019) for training the synthesis model.
- Operational GFS/GEFS (2021–present) for inference and bridge years.
- Extract NYC grid-point forecasts: TMAX, T850, 10m wind, cloud cover, MSLP, precipitation, ensemble spread.

```python
"""
Pseudocode for data collection (CDO API method — not used in practice).

For each station in STATION_LIST:
    Query NOAA CDO API for GHCND dataset
    Request datatypes: TMAX, TMIN
    Date range: START_DATE to END_DATE
    Store results as CSV: data/raw/{station_id}.csv

API endpoint:
  https://www.ncei.noaa.gov/cdo-web/api/v2/data
  ?datasetid=GHCND
  &stationid={station_id}
  &datatypeid=TMAX,TMIN
  &startdate={start}
  &enddate={end}
  &units=standard        # Returns °F directly
  &limit=1000
  &offset={offset}       # Paginate through results

Headers: {"token": YOUR_API_TOKEN}

Note: API returns max 1000 records per request.
      Must paginate and also chunk by year (max 1-year date range per request).
"""
```

**Alternative (faster for bulk):** Download the per-station `.dly` files directly from `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly` and parse the fixed-width format. This avoids API rate limits entirely.

### Step 3: Data Preprocessing (`data_preprocessing.py`)

```python
"""
Pseudocode for preprocessing.

1. Load all station data sources (GHCN, ASOS, IGRA, NWP).
2. For each station (GHCN):
   a. Pivot so each row is one date, columns are TMAX, TMIN, PRCP, SNOW, SNWD, AWND.
   b. Convert from tenths of °C to °F:
      temp_f = (value / 10) * 9/5 + 32
   c. Flag/remove quality-flagged observations.
3. For each station (ASOS hourly):
   a. Compute daily aggregates (TMAX, TMIN, wind, dewpoint, SLP, cloud fraction).
   b. Compute wind-conditioned composites (upwind temp, advection rate).
4. Merge IGRA T850 features by date (00Z/12Z).
5. Merge NWP features (GEFS reforecast / GFS operational) for synthesis training.
6. Quantify GHCN vs ASOS TMAX differences; store offsets for monitoring.
7. Merge all stations into one DataFrame:
   - Index: DATE
   - Columns: {station_id}_TMAX, {station_id}_TMIN, ... for each station
8. Create target column: NYC_TMAX (Central Park TMAX on day t)
9. Create feature columns: all other stations' values shifted by +1 day
   (i.e., feature row for day t uses surrounding stations' data from day t-1)
10. Add cyclical date features:
   day_of_year = date.timetuple().tm_yday
   sin_day = sin(2π × day_of_year / 365.25)
   cos_day = cos(2π × day_of_year / 365.25)
11. Handle missing data:
   - Drop days where NYC target is missing.
   - For input stations: forward-fill gaps ≤ 3 days,
     then use column mean for remaining NaNs (or drop row).
12. Train/val/test split:
   - Train: first 70% of dates (chronological!)
   - Validation: next 15%
   - Test: final 15%
   IMPORTANT: Do NOT shuffle. Must respect temporal order.
13. Normalize features:
   - Fit StandardScaler on training set only.
   - Apply to val and test sets.
"""
```

### Step 4: Model Definition (`model.py`)

```python
import torch
import torch.nn as nn

class TempPredictorV1(nn.Module):
    """Simple feedforward network."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class TempPredictorQuantile(nn.Module):
    """Feedforward with 3 output heads for quantile regression."""
    def __init__(self, n_features):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head_median = nn.Linear(32, 1)   # 50th percentile
        self.head_lower = nn.Linear(32, 1)    # 2.5th percentile
        self.head_upper = nn.Linear(32, 1)    # 97.5th percentile

    def forward(self, x):
        h = self.shared(x)
        return self.head_lower(h), self.head_median(h), self.head_upper(h)
```

### Step 5: Training Loop (`train.py`)

```python
"""
Pseudocode for training.

1. Load preprocessed train/val tensors.
2. Create DataLoaders (batch_size=64).
3. Initialize model, optimizer (Adam, lr=1e-3), scheduler (ReduceLROnPlateau).
4. For each epoch (max 200, with early stopping patience=15):
   a. Train step: forward pass, compute MSE loss, backprop, update.
   b. Val step: compute val MSE and val MAE.
   c. If val MAE improved, save model checkpoint.
   d. If no improvement for 15 epochs, stop.
5. Load best checkpoint.
6. Report final val MAE.

For quantile regression, use pinball loss:
  def pinball_loss(pred, target, quantile):
      error = target - pred
      return torch.max(quantile * error, (quantile - 1) * error).mean()

  total_loss = (pinball_loss(lower, y, 0.025)
              + pinball_loss(median, y, 0.50)
              + pinball_loss(upper, y, 0.975))

For Gaussian/mixture outputs, use Gaussian NLL or CRPS and track calibration metrics.
"""
```

### Step 6A: Required Headline Comparisons

Always report these **two primary model variants** (in addition to the persistence baseline):

1. **No autoregressive NYC term:** predict NYC TMAX(t) using only surrounding-station features (and date encodings).
2. **With autoregressive NYC term:** include NYC TMAX(t-1) as an input.

Interpret the performance gap as the **incremental value of the surrounding-station network** beyond persistence.

### Step 6: Evaluation (`evaluate.py`)

```python
"""
Metrics to compute on the test set:

1. MAE (Mean Absolute Error) — point accuracy
2. RMSE (Root Mean Squared Error)
3. CRPS — distribution quality (primary probabilistic metric)
4. Brier score per Kalshi bucket — calibration for trading
5. PIT histogram + reliability diagram — distribution calibration
6. Percentage of predictions within ±1°F, ±2°F, ±3°F
7. R² score
8. Bias (mean error, to detect systematic over/under-prediction)
9. Seasonal breakdown: compute MAE and CRPS for each season
   (winter=DJF, spring=MAM, summer=JJA, fall=SON)

Plots to generate:
- Actual vs. Predicted scatter plot (with y=x line)
- Time series plot of actual vs. predicted for a sample month
- Residual histogram
- Residuals by month (box plot)
- If using probabilistic outputs: coverage plots for 50/90/95% intervals
- PIT histogram and reliability diagram for calibration
- Kalshi bucket probability comparison (model vs. market implied)
"""
```

### Step 7: Synthesis Model (`synthesis_model.py`)

Combine the optimized station model with NWP features:

- Train on GEFSv12 reforecast (2000–2019) + operational GFS/GEFS (2021–present).
- Inputs: station μ/σ, NWP TMAX/T850, wind, cloud, MSLP, ensemble spread.
- Output: calibrated distribution (μ, σ or mixture parameters).
- Compare against station-only and NWP-only baselines.

### Step 8: Calibration + Kalshi Mapping (`calibration.py`, `kalshi_client.py`)

- Fit isotonic regression on CDF outputs (seasonal or regime-specific).
- Convert calibrated CDFs into **Kalshi bucket probabilities**.
- Pull KXHIGHNY markets via Kalshi API and compare model vs market implied probabilities.
- Compute Brier score and expected value (EV) deltas for trading decisions.

### Step 9: Sensitivity Analysis

Run experiments varying these dimensions:

| Experiment | Variations |
|-----------|-----------|
| **Target formulation** | Raw TMAX vs. ΔT (daily change) |
| **Loss function** | MSE, Huber (SmoothL1), MAE |
| Input temperature type | TMAX only, TMIN only, both TMAX+TMIN, average of TMAX/TMIN |
| **Sector gradients** | With/without sector averages and inter-sector gradients |
| **Trend features** | With/without Δ1/Δ2 short differences |
| Number of surrounding stations | 20, 30, 40, 50, 70 (with imputation + masking for expansion) |
| Radius of stations | 150mi, 200mi, 250mi |
| Lag structure | t-1 only, t-1 and t-2, t-1 through t-3 |
| Architecture | Ridge baseline, MLP, station-attention pooling, k-window MLP, 1D temporal conv, LSTM/GRU, Transformer-style attention |
| **Autoregressive input** | With/without NYC's own TMAX at t−1 |
| Cyclical date encoding | With/without sin/cos day-of-year features |
| Month/season indicator | With/without one-hot month encoding |
| **Stacking/residual** | Direct prediction vs. residual correction on base models |
| **Season specialization** | Single model vs. two-expert seasonal vs. mixture-of-experts |

**Reporting requirements** — for each experiment record:
- Model type, target formulation, input features, station config
- Training setup (loss, LR schedule, batch size, epochs, early stopping)
- Overall MAE, Winter MAE, Summer MAE
- MAE on "high-gradient days" (upstream-vs-coast gradient above 75th percentile)
- Failure modes, stability notes, outlier behavior

---

## 9. Benchmark / Baseline Models

To assess whether the neural network adds value, compare against these baselines:

1. **Persistence model:** Predict NYC TMAX(t) = NYC TMAX(t-1). This is the simplest possible baseline.
2. **Climatological average:** Predict NYC TMAX(t) = historical average TMAX for that calendar day.
3. **Linear regression:** Same inputs as the NN, but fit with ordinary least squares.
4. **Ridge/Lasso regression:** Regularized linear regression (guards against multicollinearity between nearby stations).
5. **NWP baseline:** Raw GFS/GEFS TMAX F024 forecast at NYC grid point.
6. **MOS baseline (if available):** NWS MAV/MEX forecast as a post-processed benchmark.

The neural network should meaningfully outperform these to justify its use.

---

## 10. Prediction Market Integration (Kalshi KXHIGHNY)

**Contract alignment checks (mandatory):**
1. Verify station = **Central Park** and the daily boundary uses NYC local time.
2. Confirm rounding rules for the market’s integer Fahrenheit outcomes.
3. Match model target definition to Kalshi contract definition.

**Kalshi public API (unauthenticated):**
- Base: `https://api.elections.kalshi.com/trade-api/v2`
- Series: `/series/KXHIGHNY`
- Markets: `/markets?series_ticker=KXHIGHNY&status=open`
- Orderbook: `/markets/{ticker}/orderbook`

**Market mapping workflow:**
1. Generate model CDF for day-*t*.
2. Convert bucket thresholds to probabilities (CDF differences).
3. Compare to market-implied probabilities (YES price / 100).
4. Compute EV, Brier score, and log score for each bucket.
5. Trade only when calibrated EV exceeds fees + threshold.

---

## 11. Project File Structure

```

### Step 6B: Interpretability / Sanity Checks

Add lightweight checks to ensure the model’s learned signal is physically plausible:

- **Permutation importance** on held-out data (feature-level and **grouped by station sector**: W/NW, SW, coastal, near-field).
- **Stability across folds/seasons:** do the top sectors remain top in winter vs summer?
- **Error slices by regime proxies:** e.g., days with large coastal–inland gradient vs small gradient.

(If you later use tree models like XGBoost/LightGBM, you can add SHAP as a second interpretability method; keep permutation importance as the common denominator across model types.)

nyc-temp-prediction/
├── config.py                # All configuration constants
├── data/
│   ├── raw/                 # Raw station + NWP files
│   ├── processed/           # Cleaned, merged DataFrames
│   └── stations.csv         # Selected station metadata
├── notebooks/
│   ├── 01_explore_stations.ipynb
│   ├── 02_data_quality.ipynb
│   └── 03_results_analysis.ipynb
├── src/
│   ├── data_collection.py
│   ├── asos_collection.py
│   ├── asos_preprocessing.py
│   ├── soundings_collection.py
│   ├── nwp_collection.py
│   ├── nwp_preprocessing.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── synthesis_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── calibration.py
│   ├── kalshi_client.py
│   └── trading.py
├── models/                  # Saved model checkpoints
├── results/                 # Plots, metric summaries
├── requirements.txt
└── README.md
```

---

## 12. Implementation Sequence and Timeline

| Phase | Task | Est. Effort | Status |
|-------|------|-------------|--------|
| **Phase 0** | **Multi-source data scale-up (prerequisite)** | | **NOT STARTED** |
| 0.1 | Download IEM ASOS hourly data (1998–2024) for stations | 6 hours | NOT STARTED |
| 0.2 | Download IGRA soundings (OKX/Upton, 2000–2024) | 2 hours | NOT STARTED |
| 0.3 | Download GEFSv12 reforecast (2000–2019) + operational GFS/GEFS (2021–2024) | 6 hours | NOT STARTED |
| 0.4 | Expand GHCN parsing to PRCP/SNOW/SNWD/AWND | 2 hours | NOT STARTED |
| **Phase 1** | **Optimize station model (IEM-based)** | | **NOT STARTED** |
| 1.1 | Wind-conditioned features (upwind temp, advection rate) | 2 hours | NOT STARTED |
| 1.2 | Add dewpoint, pressure, cloud fraction features | 2 hours | NOT STARTED |
| 1.3 | Add 850mb sounding features | 2 hours | NOT STARTED |
| 1.4 | Train wind-gated attention model + CRPS loss | 4 hours | NOT STARTED |
| **Phase 2** | **Synthesis layer (station + NWP)** | | **NOT STARTED** |
| 2.1 | Train meta-learner on station μ/σ + GFS features | 3 hours | NOT STARTED |
| 2.2 | Calibrate CDF outputs (isotonic) | 2 hours | NOT STARTED |
| **Phase 3** | **Kalshi evaluation + trading tools** | | **NOT STARTED** |
| 3.1 | Kalshi KXHIGHNY market ingestion + bucket mapping | 2 hours | NOT STARTED |
| 3.2 | Backtest EV/Kelly sizing vs historical markets | 4 hours | NOT STARTED |
| **Phase 4** | **Operationalization** | | **NOT STARTED** |
| 4.1 | 6 AM ET daily pipeline + monitoring | 3 hours | NOT STARTED |
| 4.2 | Paper trading + go-live criteria | 3 hours | NOT STARTED |

**Total estimated effort: ~43 hours (post-data scale-up)**

---

## 13. Potential Challenges and Mitigations

**Missing data:** Some stations may have gaps. Mitigation: select stations with ≥90% completeness; use forward-fill for short gaps; for remaining NaNs, experiment with imputation vs. dropping rows.

**Training–inference mismatch:** GHCN-Daily vs ASOS definitions differ (daily boundaries, rounding). Mitigation: train primarily on IEM ASOS-derived TMAX and quantify offsets against GHCN for monitoring.

**Operational latency:** Ensure all features are available by 6 AM ET. Mitigation: rely on IEM ASOS, IGRA 00Z soundings, and 00Z GFS/GEFS.

**NWP model version shifts:** GEFSv12 reforecast (GFSv15.1) vs operational GFSv16+. Mitigation: add a binary feature for operational data and re-calibrate on 2021–2024.

**Multicollinearity:** Nearby stations are highly correlated. Mitigation: Ridge/Lasso baselines will reveal this; the NN should handle it implicitly, but monitor weight magnitudes.

**Non-stationarity / Climate trends:** Long-term warming may affect older data. Mitigation: include year or decade as a feature, or detrend temperatures before training.

**Seasonal heterogeneity:** Model may perform differently across seasons (e.g., worse in spring when weather is more volatile). Mitigation: evaluate per-season; consider training separate seasonal models if performance varies widely.

**Overfitting:** With many input features relative to the signal. Mitigation: dropout, early stopping, regularization, and a clean temporal train/val/test split.

**API rate limits:** NOAA CDO API limits to 10,000 requests/day. Mitigation: use bulk `.dly` file downloads from the GHCN FTP server instead, which is faster and unlimited.

---

## 14. Getting Started — First Commands

```bash
# 1. Create project directory
mkdir -p nyc-temp-prediction/{data/{raw,processed},src,models,results,notebooks}
cd nyc-temp-prediction

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install pandas numpy scikit-learn torch matplotlib seaborn requests jupyter

# 4. Get your NOAA API token
#    Visit: https://www.ncdc.noaa.gov/cdo-web/token
#    Save the token in config.py

# 5. Run data collection
python src/data_collection.py
python src/asos_collection.py
python src/soundings_collection.py
python src/nwp_collection.py

# 6. Run preprocessing
python src/data_preprocessing.py

# 7. Train baseline models
python src/train.py --model baseline

# 8. Train station model + synthesis
python src/train.py --model nn_v1
python src/synthesis_model.py
```

---

## 15. Starter Code: `config.py`

```python
"""
Configuration for NYC Temperature Prediction Project.
"""

# NOAA API
NOAA_API_TOKEN = "YOUR_TOKEN_HERE"
NOAA_BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# Target station
TARGET_STATION = "USW00094728"  # NY City Central Park
TARGET_VARIABLE = "TMAX"

# Date range
START_DATE = "2018-01-01"
END_DATE = "2022-12-31"

# Surrounding stations (to be confirmed during exploration)
# Format: {station_id: "description"}
SURROUNDING_STATIONS = {
    "USW00014757": "Poughkeepsie, NY (Hudson Valley Regional Airport)",
    "USW00014735": "Albany, NY (Albany Airport)",
    "USW00014740": "Hartford, CT (Bradley Airport)",
    "USW00094702": "Bridgeport, CT (Sikorsky Airport)",
    "USW00014732": "Islip, NY (Long Island)",
    "USW00093730": "Atlantic City, NJ (Airport)",
    "USW00014792": "Trenton, NJ (Mercer Airport)",
    "USW00013739": "Philadelphia, PA (Airport)",
    "USW00014737": "Allentown, PA (ABE Airport)",
    "USW00014777": "Scranton, PA (Wilkes-Barre)",
    "USW00014734": "Newark, NJ (Airport)",
    "USW00094789": "JFK Airport, NY",
    "USW00014739": "LaGuardia Airport, NY",
    "USW00014771": "White Plains, NY (Westchester)",
    # Add more as discovered
}

# ASOS station mapping (GHCN -> ICAO)
ASOS_STATIONS = {
    "USW00094728": "KNYC",
    "USW00014735": "KALB",
    "USW00014740": "KBDL",
    "USW00094702": "KBDR",
    "USW00014732": "KISP",
    "USW00093730": "KACY",
    "USW00014792": "KTTN",
    "USW00013739": "KPHL",
    "USW00014737": "KABE",
    "USW00014777": "KAVP",
    "USW00014734": "KEWR",
    "USW00094789": "KJFK",
    "USW00014739": "KLGA",
    "USW00014771": "KHPN",
    "USW00014757": "KPOU",
}

# NWP settings
GFS_MODEL = "gfs"
GEFS_REFORECAST_MODEL = "gefs_reforecast"
NWP_FXX_HOURS = 24

# Input features per station
INPUT_VARIABLES = ["TMAX", "TMIN"]

# Training config
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZES = [64, 32]
DROPOUT = 0.1
```

---

*This plan provides a complete roadmap from data acquisition through model training, evaluation, and confidence interval estimation. Begin with Phase 1 (data pipeline) and iterate from there.*
