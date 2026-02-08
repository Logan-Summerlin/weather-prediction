# Addendum: Maximizing the Station-Based Model, Then Synthesizing for Market Edge

**Date:** 2026-02-08
**Context:** The original review (Section 1) recommended making NWP the primary input. The project owner's position is different: **push the station-observation model to its theoretical maximum first**, then synthesize it with NWP as a supplementary signal — creating a model that beats any single NWP forecast and the market consensus. This addendum provides the roadmap.

**Why this approach can work:** NWP models are general-purpose — they forecast the entire globe. A model hyper-specialized on one station (Central Park), trained on decades of local observation data with physics-informed features, can discover systematic local biases that NWP cannot resolve. MOS (Model Output Statistics) already proves this principle: statistical post-processing of NWP routinely beats raw NWP. The ambition here is to build something *better* than MOS by combining a rich observation-derived model with NWP signals in a learned synthesis layer.

---

## CRITICAL: Operational Data Availability at 6 AM ET

**Constraint:** The model must have all day t-1 data by 6:00 AM Eastern Time on day t in order to predict that day's high temperature. This section audits every data source.

### Availability Summary

| Source | Day t-1 Available by 6 AM ET? | Typical Latency | Role |
|---|---|---|---|
| **GHCN-Daily (.dly files)** | **NO** | 1-2 days for US stations | Training only |
| **IEM ASOS (hourly METAR)** | **YES** | 5-10 minutes | **Primary operational obs source** |
| **NOAA ISD (hourly obs)** | **YES** | 1-2 hours | Backup operational obs source |
| **IGRA Soundings (via UWyo/Siphon)** | **YES** | 2-3 hours (00Z and 12Z) | Operational upper-air input |
| **ERA5 / ERA5T Reanalysis** | **NO** | ~5 days (ERA5T) | Training only |
| **GFS (NOAA NOMADS)** | **YES** | 00Z run: fully available; 06Z: partial | Operational NWP input |
| **HRRR (NOAA NOMADS)** | **YES** | 50-90 min per cycle; 09Z available | Operational NWP input |
| **NWS MOS (MAV/MEX)** | **YES** | 00Z products: available; 06Z: borderline | Operational forecast input |
| **NWS API (api.weather.gov)** | **YES** | Minutes | Real-time spot obs |

### Two Critical Problems

**Problem 1: GHCN-Daily is not a real-time data source.** The entire model is trained on GHCN-Daily TMAX, but the .dly bulk files are updated with a 1-2 day lag for US stations. Yesterday's TMAX will NOT be in the .dly file by 6 AM today. The operational pipeline MUST use IEM ASOS hourly data instead, computing TMAX from the hourly METAR observations.

**Problem 2: ERA5 is not a real-time data source.** ERA5T (the near-real-time preliminary product) has a ~5-day lag. ERA5 is invaluable for training (complete atmospheric state back to 1979) but CANNOT be used for operational predictions. For real-time operations, the model must use GFS/HRRR/MOS output as the NWP supplement instead.

### Training-Inference Mismatch Risk

If we train on GHCN-Daily TMAX but predict using IEM-derived TMAX, there will be small systematic differences:
- **Observation time conventions:** GHCN uses the station's "observation time" (often 7 AM local) to define the daily boundary. IEM reports in UTC. A TMAX that occurs at 11:50 PM might fall on different calendar days in each system.
- **Rounding:** ASOS reports temperature in whole-degree Celsius; GHCN stores tenths of °C. Conversion to °F amplifies rounding differences by 1.8×.
- **Quality control:** GHCN applies post-hoc QC flags; ASOS METAR data is raw.

**Mitigation strategy:**
1. During training, compute TMAX from IEM hourly data for the training period alongside GHCN-Daily TMAX
2. Quantify the systematic difference (expected: <0.5°F mean, <1°F std)
3. Train a small calibration offset or, better, train the model on IEM-derived TMAX from the start so training and inference use the same data source
4. Use GHCN-Daily only as a validation cross-check

### What's Available at 6 AM ET (Detailed)

**From last night / overnight:**
- 00Z GFS run (init 7 PM ET yesterday): FULLY available, all forecast hours posted by ~1 AM ET
- 00Z HRRR extended run (init 7 PM ET yesterday): Fully available
- 00Z MOS (MAV and MEX): Available by ~midnight-2 AM ET
- 00Z radiosonde from OKX/Upton (launched ~7 PM ET yesterday): Available by ~10 PM ET yesterday
- All hourly ASOS/METAR observations through 05:00 local (10Z): Available within minutes

**From this morning:**
- 06Z GFS run (init 1 AM ET): Early forecast hours (F000-F048) likely available; full run borderline
- 06Z HRRR extended: Available by ~3-3:30 AM ET
- 09Z HRRR: Available by ~5:30-6:00 AM ET (latest high-res cycle)
- 06Z MOS (MAV): Borderline — may be arriving right at 6 AM

**NOT available:**
- Yesterday's GHCN-Daily record (1-2 day lag)
- ERA5 for yesterday (~5-day lag)
- 12Z GFS/HRRR for today (hasn't run yet — 12Z = 7 AM ET)

---

## Part I: Maximizing the Station-Based Model

### 1. The Current Model's Bottlenecks (Diagnosis Before Treatment)

The best current model (NN Delta+Huber+AR) achieves MAE=3.95°F. Before adding complexity, identify where the errors come from:

| Error Source | Estimated Contribution | Evidence | Fix |
|---|---|---|---|
| **Data volume** (1,277 train samples) | ~0.5-1.0°F | 79-feature model overfits; optimal at only 10 stations | Scale to 40 years (Phase 6) |
| **Missing wind information** | ~0.3-0.5°F | Model can't distinguish advection direction; static station weights | Add ASOS wind data |
| **Missing humidity/cloud signal** | ~0.2-0.4°F | Clear-sky vs. overcast TMAX differs by 5-10°F; model is blind to this | Add dewpoint, cloud cover |
| **Missing pressure/synoptic signal** | ~0.2-0.3°F | Can't detect approaching fronts or pressure systems | Add SLP, pressure tendency |
| **No upper-air information** | ~0.2-0.3°F | 850mb temp is the best single predictor of surface TMAX | Add radiosonde/ERA5 data |
| **Architecture limitations** | ~0.1-0.2°F | Attention model underperforms MLP due to small data | Will resolve with more data |
| **Temporal resolution** | ~0.1-0.2°F | Daily aggregates lose intra-day trajectory information | Add sub-daily features |

**Conservative estimate:** fixing these could reduce MAE from 3.95°F to ~2.0-2.5°F — competitive with NWS day-1 forecasts.

---

### 2. New Data Sources (Priority-Ordered)

#### 2.1 ASOS/METAR Hourly Observations (HIGHEST PRIORITY — ALSO THE OPERATIONAL DATA SOURCE)

This is the single highest-value data addition AND it solves the operational data availability problem. ASOS stations report hourly surface observations including variables the GHCN completely lacks. Critically, **IEM ASOS data is available within 5-10 minutes of observation**, making it the only viable real-time source for station observations.

**Source:** Iowa Environmental Mesonet (IEM) ASOS download service
**URL:** `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
**Format:** CSV, hourly/sub-hourly resolution
**Coverage:** All 14 surrounding stations have co-located ASOS stations (they are airports). Data back to the 1990s for most.
**Cost:** Free, no API key needed.
**Latency:** 5-10 minutes (vs. 1-2 days for GHCN-Daily). All of yesterday's data is available by 6 AM ET.

**Key variables to extract:**

| Variable | IEM Code | Why It Matters | Expected MAE Impact |
|---|---|---|---|
| Wind direction | `drct` | Tells you WHICH stations are upwind — the single most important missing signal | 0.2-0.4°F |
| Wind speed | `sknt` | Strength of advection; strong wind = faster temperature change | 0.1-0.2°F |
| Dewpoint | `dwpf` | Moisture content; controls overnight cooling and daytime heating ceiling | 0.1-0.2°F |
| Sea-level pressure | `mslp` | Synoptic pattern indicator; falling pressure = approaching system | 0.1-0.2°F |
| Cloud cover/ceiling | `skyc1-4`, `skyl1-4` | Clear vs. overcast controls solar heating; 5-10°F TMAX difference | 0.2-0.3°F |
| Visibility | `vsby` | Proxy for fog, haze, precipitation — secondary signal | 0.05°F |
| Feels-like/heat index | `feel` | Derived; less useful than raw components | Skip |

**Daily aggregation strategy for t-1 features:**
- Wind direction: compute the *prevailing wind direction* (vector-mean of hourly u,v components) and *directional persistence* (consistency of direction). Also compute the *evening wind direction* (18Z-00Z) separately — this is most predictive of overnight/next-day advection.
- Wind speed: daily mean, max, and evening mean.
- Dewpoint: daily mean and afternoon (18Z) value.
- Pressure: value at 00Z (midnight), 12Z (morning), and *24-hour pressure change* (tendency).
- Cloud cover: fraction of hours with ceiling < 5000ft (overcast proxy).

**Station mapping (GHCN → ICAO for IEM):**

| GHCN Station | ICAO | Location |
|---|---|---|
| USW00094728 | KNYC | Central Park (target) |
| USW00014735 | KALB | Albany |
| USW00014740 | KBDL | Hartford/Bradley |
| USW00094702 | KBDR | Bridgeport/Sikorsky |
| USW00014732 | KISP | Islip/MacArthur |
| USW00093730 | KACY | Atlantic City |
| USW00014792 | KTTN | Trenton/Mercer |
| USW00013739 | KPHL | Philadelphia |
| USW00014737 | KABE | Allentown/Lehigh Valley |
| USW00014777 | KAVP | Scranton/Wilkes-Barre |
| USW00014734 | KEWR | Newark |
| USW00094789 | KJFK | JFK |
| USW00014739 | KLGA | LaGuardia |
| USW00014771 | KHPN | White Plains/Westchester |
| USW00014757 | KPOU | Poughkeepsie |

**Implementation approach:**
```
src/asos_collection.py    — Download hourly ASOS data from IEM for all stations
src/asos_preprocessing.py — Aggregate to daily features, merge with GHCN pipeline
```

#### 2.2 GHCN Additional Variables (MEDIUM PRIORITY, LOW EFFORT)

The existing `.dly` file parser only extracts TMAX and TMIN. The same files already contain additional variables that require zero new downloads — just expanding the parser.

**Variables to add (available in existing .dly files):**

| Element | Description | Units | Usefulness |
|---|---|---|---|
| PRCP | Precipitation | tenths of mm | Evaporative cooling indicator; post-precip days are often cooler |
| SNOW | Snowfall | mm | Snow cover proxy; albedo effect on next-day TMAX |
| SNWD | Snow depth | mm | Persistent albedo/thermal effect; snow-covered ground suppresses TMAX |
| AWND | Average daily wind speed | tenths of m/s | Advection strength (but less complete than ASOS) |
| WDF2 | Direction of fastest 2-min wind | degrees | Crude wind direction (check availability) |
| WSF2 | Fastest 2-minute wind speed | tenths of m/s | Gust/front passage indicator |
| TAVG | Average temperature | tenths of °C | Mean temperature proxy (where available) |

**Implementation:** Modify `ELEMENTS_OF_INTEREST` in `data_collection.py` from `{"TMAX", "TMIN"}` to include PRCP, SNOW, SNWD, AWND. Minimal code change — the parser already handles any element code.

**Caveat:** AWND completeness at airport stations varies. Check `ghcnd-inventory.txt` for each station's AWND coverage before relying on it. ASOS wind data (Section 2.1) is more reliable and higher resolution.

#### 2.3 Upper-Air Sounding Data (MEDIUM PRIORITY)

The 850mb temperature is arguably the single most predictive upper-air variable for surface TMAX. It represents the "free atmosphere" temperature that surface stations will trend toward through mixing.

**Source:** Integrated Global Radiosonde Archive (IGRA) via the Siphon Python library
**Nearest station:** Upton/Brookhaven, NY (WMO 72501, IGRA ID `USM00072501`), ~60 miles east of Central Park
**Temporal resolution:** 2x/day (00Z and 12Z soundings)
**Coverage:** Back to 1957; reliable digital data from ~1973+

**Key variables to extract:**
- 850mb temperature (°C) — primary predictor
- 850mb wind direction and speed — upper-level advection direction
- 850mb-surface temperature difference — stability indicator
- 500mb temperature — deep-layer thermal indicator (secondary)
- Precipitable water (derived) — moisture/cloud potential

**Python access:**
```python
from siphon.simplewebservice.igra2 import IGRAUpperAir
from datetime import datetime
date = datetime(2022, 7, 15, 12)  # 12Z sounding
df, header = IGRAUpperAir.request_data(date, 'USM00072501')
# Extract 850mb level
t850 = df[df['pressure'] == 850.0]['temperature'].values[0]
```

**Daily feature engineering:**
- Use the 12Z (morning) sounding for day-t prediction (available by ~8 AM ET)
- Use the 00Z (evening) sounding from day t-1 for overnight context
- Compute T850 anomaly from climatological mean
- Compute T850 - T_surface (stability: positive = warm air aloft, inversion)

#### 2.4 ERA5 Reanalysis (TRAINING ONLY — NOT AVAILABLE IN REAL-TIME)

ERA5 provides gridded atmospheric variables at 0.25° resolution, hourly, back to 1979. This is the gold standard for historical atmospheric state — essentially what NWP "would have forecast" for historical dates.

**Source:** Copernicus Climate Data Store (CDS)
**Access:** `cdsapi` Python library; requires free CDS account
**Resolution:** 0.25° × 0.25° (~28 km), hourly
**Latency:** ~5 days behind real-time. **ERA5 CANNOT be used for operational predictions.** Yesterday's ERA5 data will not be available until ~5 days from now. Use GFS/HRRR instead for real-time operations (see Part II synthesis architecture).

**Key variables for NYC grid point (~40.75°N, 74.0°W):**
- 2m temperature, 2m dewpoint
- 10m u/v wind components
- Mean sea-level pressure
- Total cloud cover
- 850mb temperature, 850mb u/v wind
- 500mb geopotential height
- Total precipitation
- Surface solar radiation downwards

**Why ERA5 is valuable for training (not operations):**
ERA5 provides a *complete, gap-free atmospheric state* for every hour back to 1979. Training on ERA5 features means your model learns from a complete picture — wind direction, upper-air temperature, cloud cover — for every historical day. When you go operational, you substitute ERA5 with real-time equivalents (ASOS for surface obs, GFS/HRRR for upper-air). The features are the same; only the source changes.

**Bounding box for download:**
```python
'area': [42.0, -75.0, 39.5, -72.5]  # Covers full station network
```

---

### 3. Physics-Informed Feature Engineering

The current model treats station temperatures as interchangeable numbers. The following features encode physical knowledge about *how* temperature changes propagate.

#### 3.1 Wind-Conditioned Station Weighting (THE KEY INNOVATION)

**The problem:** The current model assigns static learned weights to each station. But a station's predictive value depends entirely on whether it is upwind or downwind on that specific day. Albany (NW, 136 miles) is highly predictive when wind blows from the NW (cold front approaching); it is nearly irrelevant when wind blows from the SW.

**Feature design — "Upwind Temperature":**

Given prevailing wind direction θ on day t-1, compute:
```
For each station i with bearing β_i and distance d_i from Central Park:
    angular_alignment_i = cos(θ - β_i)  # +1 if directly upwind, -1 if downwind
    upwind_weight_i = max(0, angular_alignment_i) / d_i^0.5  # distance-decayed upwind weighting

upwind_temp = Σ(upwind_weight_i × TMAX_i) / Σ(upwind_weight_i)
crosswind_temp = weighted average of stations perpendicular to wind
downwind_temp = weighted average of stations directly downwind
```

This gives the model three physically meaningful composite features instead of 14 individual station temperatures:
- **Upwind temperature:** what air mass is arriving
- **Crosswind temperature:** lateral gradient (front orientation)
- **Downwind temperature:** what air mass is leaving

**Additional wind-derived features:**
- `upwind_gradient = upwind_temp - NYC_TMAX(t-1)` — magnitude of incoming temperature change
- `upwind_distance_weighted_temp` — emphasizes closer upwind stations
- `max_upwind_station_temp` and `min_upwind_station_temp` — extremes of incoming air

These features can be computed at multiple levels:
- Using daily-mean wind direction (from ASOS)
- Using evening wind direction (more predictive of overnight advection)
- Using 850mb wind direction (from soundings; better represents large-scale flow)

#### 3.2 Advection Rate Feature

Estimate the actual temperature advection rate:
```
advection_rate = wind_speed × upwind_gradient / upwind_distance
```
This approximates the physical advection equation ΔT/Δt ≈ -V × (ΔT/Δd) and directly estimates how fast temperature is changing due to horizontal transport.

#### 3.3 Frontal Passage Detection

Fronts cause the most dramatic day-to-day temperature changes and are where forecast errors concentrate. Detect them from station observations:

```
front_indicator features:
- max_station_24h_change: largest |T(t-1) - T(t-2)| across all stations
- WNW_to_coast_timing: hours between when WNW stations cool and when coast cools
  (if WNW cooled 8°F yesterday while coast didn't, a front passed inland but hasn't reached NYC)
- pressure_tendency: 24h SLP change at nearest station (falling = approaching low)
- wind_shift: change in prevailing wind direction from t-2 to t-1 (>90° shift = frontal passage)
- dewpoint_drop: sudden dewpoint decrease = dry air mass arrival (post-cold-front)
```

#### 3.4 Enhanced Temporal Features

Beyond the current sin/cos day-of-year:
```
- day_length: hours of daylight (computed from latitude + day-of-year; controls solar heating budget)
- solar_elevation_noon: maximum sun angle (determines peak heating potential)
- days_since_solstice: linear lag feature (temperature lags solar forcing by ~30 days)
- heating_degree_days_trailing_7: recent cold anomaly (cold ground = lower TMAX)
- cooling_degree_days_trailing_7: recent heat anomaly (warm ground = higher TMAX)
- TMAX_7day_rolling_mean: recent thermal regime
- TMAX_anomaly_from_climo: how unusual is the current temperature? (regime indicator)
```

#### 3.5 Precipitation and Snow Features

From GHCN (zero new downloads needed):
```
- precip_yesterday: PRCP(t-1) at Central Park and sector averages
- precip_binary: did it rain? (evaporative cooling + cloud cover proxy)
- snow_depth: SNWD at Central Park (albedo effect: snow reflects solar radiation, suppresses TMAX)
- snow_depth_change: SNWD(t-1) - SNWD(t-2) (fresh snow vs. melting)
- days_since_last_precip: drought duration (dry ground heats faster)
```

#### 3.6 Dewpoint and Humidity Features

From ASOS:
```
- dewpoint_nyc: afternoon dewpoint at Central Park (moisture content)
- dewpoint_depression: T - Td (large depression = dry air = greater diurnal range)
- sector_dewpoint_gradient: moisture contrast between sectors (moisture advection)
- dewpoint_trend: 24h dewpoint change (increasing = warm/moist air arriving)
```

#### 3.7 Cloud Cover Proxy

From ASOS:
```
- cloud_fraction_yesterday: fraction of hours with ceiling < 5000ft
- clear_sky_yesterday: binary (was it mostly clear?)
- cloud_trend: change in cloud fraction from t-2 to t-1
- predicted_clear_sky: if yesterday was clear AND pressure rising AND dewpoint low,
  today likely clear → expect full solar heating → higher TMAX
```

---

### 4. Architecture Upgrades

#### 4.1 Wind-Gated Station Attention (PRIMARY ARCHITECTURE)

The existing `StationAttentionModel` uses a learned query vector to compute static attention weights. Replace this with a **wind-conditioned attention** mechanism:

```
Architecture: WindGatedStationAttention

Input per station i:
  [TMAX_i(t-1), TMIN_i(t-1), diurnal_range_i, PRCP_i,
   dewpoint_i, wind_speed_i, ΔT_i(t-1 vs t-2)]

Station metadata (static, embedded):
  [bearing_i, distance_i, elevation_i, sector_one_hot_i]

Global context (shared across all stations):
  [wind_dir_prevailing, wind_speed_mean, SLP, SLP_tendency,
   sin_day, cos_day, day_length, snow_depth_nyc,
   NYC_TMAX(t-1), NYC_dewpoint(t-1)]

Architecture:
  1. Station encoder (shared weights):
     station_embed_i = MLP([station_features_i, station_metadata_i]) → dim 64

  2. Wind-conditioned attention:
     - Compute angular_alignment_i = cos(wind_dir - bearing_i)
     - Key = station_embed_i
     - Query = MLP([global_context]) → dim 64
     - Value = station_embed_i
     - Attention logit = dot(Q, K_i) + α × angular_alignment_i
       (α is a learned scalar that controls how much wind direction biases attention)
     - Attention weights = softmax(logits)
     - Pooled = Σ(attention_weight_i × V_i)

  3. Output head:
     combined = [pooled, global_context, upwind_temp, advection_rate]
     prediction = MLP(combined) → ΔT (or distribution parameters)
```

**Why this works:** The attention mechanism learns to dynamically weight stations based on the day's wind regime. On NW-wind days, it upweights Albany, Scranton, Poughkeepsie. On SW-wind days, it upweights Philadelphia, Trenton. The angular alignment bias gives it an inductive prior from physics, but the network can override it when the data says otherwise.

#### 4.2 Multi-Scale Temporal Model

Instead of single-lag (t-1), use a 3-day window with temporal attention:

```
For each day in [t-1, t-2, t-3]:
  - Compute station-pooled embedding (using wind-gated attention above)
  - Compute global context features

Stack: (3, embed_dim) temporal sequence

Temporal processing (choose one):
  Option A: 1D Conv (kernel=2, stride=1) → captures 2-day transitions
  Option B: Temporal attention (3 time steps is too short for LSTM to help)
  Option C: Simple concatenation + MLP (often best for k ≤ 3)

Output: ΔT prediction or distribution parameters
```

**Why 3 days:** Weather systems in the mid-latitudes have a characteristic timescale of 3-7 days. A 3-day window captures the onset, peak, and exit of frontal passages. Beyond 3 days, the marginal signal decays rapidly for day-ahead TMAX prediction.

#### 4.3 Mixture Density Network Output (REQUIRED FOR TRADING)

Replace the single-value output with a **parametric distributional output:**

```
Output head options (in order of preference):

Option 1: Gaussian Mixture (2-3 components)
  - Parameters: [μ₁, σ₁, μ₂, σ₂, π₁] (5 params for 2-component)
  - Loss: Negative log-likelihood
  - Why: Handles bimodal forecasts (e.g., cold front may or may not arrive)

Option 2: Heteroscedastic Gaussian
  - Parameters: [μ, σ]
  - Loss: Gaussian NLL = 0.5 × [log(σ²) + (y-μ)²/σ²]
  - Why: Simplest distributional output; learned σ varies by day

Option 3: Quantile Function Network
  - Output: 19 quantiles (5th through 95th, every 5th percentile)
  - Loss: Pinball loss summed across quantiles
  - Why: Non-parametric; captures any distribution shape
  - Con: Quantile crossing requires post-hoc sorting
```

**For Kalshi bucket probabilities:** Given the output distribution, compute:
```
P(a ≤ TMAX < b) = CDF(b) - CDF(a)
```
For Gaussian mixture: `CDF(x) = Σ πₖ × Φ((x - μₖ)/σₖ)`

**Training loss: CRPS (Continuous Ranked Probability Score)**

CRPS is the proper scoring rule for distributional forecasts. For a Gaussian:
```
CRPS(μ, σ, y) = σ × [z × (2Φ(z) - 1) + 2φ(z) - 1/√π]
where z = (y - μ)/σ, Φ = CDF, φ = PDF of standard normal
```

CRPS rewards both accuracy (correct μ) and calibration (correct σ). It is strictly proper — the only way to minimize CRPS is to output the true predictive distribution.

---

### 5. Training Protocol Changes

#### 5.1 Loss Function: CRPS, Not MSE

| Loss | What It Optimizes | Suitable For |
|---|---|---|
| MSE | Mean prediction accuracy | Point forecasts only |
| MAE/Huber | Median prediction accuracy | Point forecasts only |
| Gaussian NLL | Distributional accuracy | Probabilistic output |
| **CRPS** | **Full distribution calibration** | **Probabilistic output (preferred)** |
| Pinball (quantile) | Individual quantile accuracy | Quantile regression |

#### 5.2 Expanded Evaluation Metrics

| Metric | Purpose | Target |
|---|---|---|
| MAE | Point forecast accuracy | ≤ 2.5°F |
| CRPS | Distributional sharpness + calibration | ≤ 2.0°F |
| Brier score (per bucket) | Binary calibration for each Kalshi-style bucket | < 0.15 |
| PIT histogram | Overall distributional calibration | Uniform |
| Reliability diagram | Conditional calibration | Diagonal |
| Interval coverage (90%) | Does the 90% interval contain 90% of outcomes? | 88-92% |
| Tail calibration | P(TMAX ≥ 90°F) accuracy on hot days | Within 10% |
| Log score | Information-theoretic scoring | Minimize |

#### 5.3 Calibration Layer (Post-Hoc)

Even a CRPS-trained model will be miscalibrated in practice. Add a post-hoc calibration step:

1. **Hold out a calibration set** (separate from validation): the last 20% of the validation period.
2. **Compute raw model CDF** at each integer temperature from 0-110°F for each calibration day.
3. **Apply isotonic regression** to each CDF level: maps raw P(TMAX ≤ t) to calibrated P(TMAX ≤ t).
4. **Evaluate calibration** on the test set using PIT histogram and reliability diagram.

For conditional calibration (by season, by regime):
- Fit separate isotonic regression models for winter (DJF) vs. summer (JJA) vs. transition (MAM/SON).
- Or use a small calibration network that takes (raw_probability, season, regime_indicator) → calibrated_probability.

#### 5.4 Training with 40 Years of Data

With Phase 6 data (1985-2024, ~14,000 samples after lag):

| Configuration | n_train | n_val | n_test | n_calib |
|---|---|---|---|---|
| **Recommended split** | 10,950 (1985-2014) | 1,825 (2015-2019) | 1,460 (2020-2023) | 365 (2024) |

With 10,950 training samples (8.6× more than current):
- The 79-feature enhanced model should no longer overfit
- Station attention model becomes viable
- 50-station network becomes beneficial
- Multi-lag (t-1, t-2, t-3) inputs become viable
- Mixture density outputs become trainable

**Climate non-stationarity mitigation:**
- Apply a recency weight: samples from 2010-2024 weighted 2× vs. 1985-2009
- Or use a rolling window of 20 years for training (retrain annually)

---

### 6. Expected Performance After Full Optimization

**Conservative estimates based on literature and analysis of error sources:**

| Enhancement | Expected MAE Reduction | Cumulative MAE |
|---|---|---|
| Current best (Delta+Huber+AR, 5yr) | — | 3.95°F |
| Scale to 40yr data | -0.4 to -0.6°F | ~3.4-3.5°F |
| Add ASOS wind direction + speed | -0.3 to -0.5°F | ~3.0-3.2°F |
| Wind-gated station attention | -0.1 to -0.2°F | ~2.8-3.1°F |
| Add dewpoint + pressure + cloud proxy | -0.2 to -0.3°F | ~2.6-2.8°F |
| Add PRCP/SNOW/SNWD from GHCN | -0.05 to -0.1°F | ~2.5-2.7°F |
| Add 850mb temperature (soundings) | -0.1 to -0.2°F | ~2.4-2.6°F |
| Multi-lag temporal features (3-day) | -0.05 to -0.1°F | ~2.3-2.5°F |
| Physics-informed features (advection rate, fronts) | -0.1 to -0.2°F | ~2.2-2.4°F |
| **Optimized station-based model** | | **~2.2-2.5°F** |

This brings the station-based model into the competitive range with NWS day-1 forecasts (~2.0-2.5°F MAE). It will be better than NWS on some day-types (regime transitions, advection events with good station coverage) and worse on others (clear-sky radiative days, tropical moisture events).

---

## Part II: Synthesis Architecture (Your Model + NWP = Market Edge)

### 7. The Synthesis Philosophy

The goal is not to replace NWP. The goal is to build a synthesis model where:

1. **Your station-based model** provides local ground truth, regime detection, and bias estimation
2. **NWP forecasts** provide the physics-based atmospheric evolution
3. **The synthesis layer** learns when each source is more reliable and produces a calibrated distribution that beats either one alone

This is methodologically similar to **Ensemble Model Output Statistics (EMOS)** and **Bayesian Model Averaging (BMA)**, but with a neural network meta-learner instead of linear/Gaussian assumptions.

### 8. NWP Data Sources: The Training-Inference Problem (Deep Analysis)

The synthesis layer needs NWP forecast features for both training and operations. The critical question: **what NWP data do you train on?** There are three options, each with real tradeoffs. This section analyzes them in depth.

#### Option A: GEFSv12 Reforecast (RECOMMENDED)

The **GEFSv12 Reforecast** is a dataset of retrospective GFS forecasts run with a single frozen model version (GFSv15.1/FV3) for every day from 2000-2019. It's the cleanest solution.

**What it is:** NOAA ran the GFS model backwards through 20 years of historical dates, using a single frozen model version throughout. This means the biases and error characteristics are **consistent** across all 20 years — the model doesn't change, only the weather changes. This is exactly what you need for ML training.

**Key specs:**
| Property | Value |
|---|---|
| Period | 2000-2019 (Phase 2; Phase 1 1989-1999 not easily accessible) |
| Model version | Frozen GFSv15.1 (FV3 dynamical core) |
| Resolution | 0.25° (Days 1-10), 0.5° (Days 10-16) |
| Init time | 00Z daily (one run/day) |
| Ensemble members | 5 daily (c00 + p01-p04), 11 on Wednesdays |
| Forecast hours | Every 3h out to Day 10 |
| Available on | AWS S3: `s3://noaa-gefs-retrospective/GEFSv12/reforecast/` |
| Format | GRIB2, one file per variable per member per day |
| Cost | Free, no sign-up |
| Python access | **Herbie** library (`model="gefs_reforecast"`) |

**Variables available (all the ones we need):**
| Variable | File name | Notes |
|---|---|---|
| 2m TMAX | `tmax_2m` | Dedicated file — exactly what we need |
| 2m TMIN | `tmin_2m` | Dedicated file |
| 2m Temperature | `tmp_2m` | Instantaneous T at each forecast hour |
| 850mb Temperature | `tmp_pres` | On pressure levels (includes 850mb) |
| 10m U/V Wind | `ugrd_10m`, `vgrd_10m` | Wind components |
| Total Cloud Cover | `tcdc_eatm` | Entire atmosphere |
| MSLP | `pres_msl` | Mean sea-level pressure |
| Precipitation | `apcp_sfc` | Accumulated |
| 500mb Height | `hgt_pres` | On pressure levels |
| Precipitable Water | `pwat` | Column moisture |
| Wind Gust | `gust` | Surface |
| Ensemble spread | Computed from 5 members | Standard deviation across c00, p01-p04 |

**Download example via Herbie:**
```python
from herbie import Herbie
H = Herbie("2017-03-14", model="gefs_reforecast", fxx=24, member=0, variable_level="tmax_2m")
ds = H.xarray(":24 hour fcst:")
nyc = ds.sel(latitude=40.78, longitude=360-73.97, method="nearest")
```

**Why this is the right choice:**
1. **Zero training-inference mismatch for NWP features.** The reforecast uses the same GFS model version as operational GFS (v15.1). Biases are similar. When you switch to real-time GFS at inference, the model sees the same kind of input.
2. **Consistent biases across 20 years.** Unlike stitching together operational GFS archives (which span 6+ model versions), the reforecast has uniform error characteristics. The ML model can learn a single bias correction pattern.
3. **Ensemble spread available.** With 5 members, you can compute forecast uncertainty as a feature — something ERA5 cannot provide.
4. **Known cold bias.** The GFSv15.1 has a documented cold bias of ~1.5-1.8°C at 00Z over CONUS. This is a learnable, consistent signal — not a problem, but a feature your model can exploit.
5. **Data is a real forecast.** Unlike ERA5 (which is an analysis that already knows the answer), the reforecast represents what GFS would have predicted before the event. This is the honest input for a synthesis model.

**Known limitations:**
- Only 00Z initialization (one run/day). Operational GFS runs 4x/day.
- Only 5 ensemble members (operational GEFS runs 31). Spread estimates are noisier.
- Frozen at GFSv15.1, while operational GFS is now v16. There is a small model-version gap.
- Phase 1 (1989-1999) is not easily accessible on AWS.

**Operational bridging:** For 2020-2024, the reforecast doesn't exist. Use operational GFS data:
- **2021-present:** Available on AWS (`s3://noaa-gfs-bdp-pds`), 0.25°, Herbie-accessible
- **2020:** Available from NCAR RDA (d084001), 0.25°

This gives a combined training set of **2000-2024 (24 years)** with only one discontinuity (GEFSv12 reforecast → operational GFSv16 at ~2021).

---

#### Option B: ERA5 + Synthetic Noise (NOT RECOMMENDED)

This approach trains on ERA5 features and adds random noise to simulate GFS forecast error. **The research conclusively shows this doesn't work.**

**Why ERA5 and GFS are not "the same thing plus noise":**

1. **ERA5 is an analysis; GFS is a forecast.** ERA5 knows the answer because it assimilates observations after the fact. GFS must predict the future. These are fundamentally different tasks with fundamentally different error structures.

2. **GFS errors are structured, not random.** Per NCEP Office Note 520 (2024):
   - GFS has a **cold bias of 1.5-1.8°C at 00Z** over most of CONUS
   - ERA5 has **~0.2°C bias** at the same time
   - GFS biases have a **strong diurnal cycle** (largest cold bias at 00Z, different at 12Z)
   - GFS biases are **spatially structured** (cold over Appalachians, warm over Great Plains)
   - ERA5 biases are spatially much more uniform and smaller

3. **GFS errors are temporally autocorrelated.** GFS forecast errors persist day-to-day because they depend on weather regime (blocking patterns, storm tracks, El Niño). NCEP's "decaying average" bias correction method explicitly exploits this persistence. Random noise has zero autocorrelation — a model trained on noise-injected ERA5 would never learn to exploit this persistence.

4. **GFS errors are correlated across variables.** When GFS overestimates temperature, it may also overestimate wind or underestimate cloud cover, depending on the synoptic pattern. Gaussian noise applied independently to each variable cannot reproduce these cross-variable correlations.

5. **ERA5 assimilates your station data.** ERA5 ingests the same ASOS observations you use as model features. This means ERA5 and your station features are not independent — ERA5 has already "seen" the station temperatures. A synthesis model trained on ERA5 might learn to just trust ERA5 (since it already contains the station signal). At inference, GFS hasn't seen today's stations, so this learned relationship breaks.

**The literature confirms this is problematic.** Training on reanalysis and deploying on forecasts is a known domain-adaptation problem in NWP post-processing. Studies show that models trained on ERA5 analysis fields underperform when deployed on GFS forecast fields without explicit adaptation. The error characteristics are too different for simple noise injection to bridge.

**Verdict: Do not use this approach.** The structural differences between ERA5 and GFS are too large for noise injection to resolve. Option A (GEFSv12 Reforecast) eliminates this problem entirely.

---

#### Option C: Hybrid Architecture (RECOMMENDED ARCHITECTURE, USE WITH OPTION A DATA)

This is the recommended overall architecture. The station model stands completely alone with zero NWP dependency. The synthesis layer is a separate, lightweight model.

**Architecture:**
```
┌──────────────────────────────────────────────────────────────┐
│  STATION MODEL (Part I)                                      │
│  Training data: IEM ASOS 1998-2024 (26 years)                │
│  NWP dependency: NONE                                        │
│  Inputs: station temps, wind, dewpoint, pressure, clouds,    │
│          850mb from IGRA, PRCP/SNOW from GHCN               │
│  Output: μ_station, σ_station (distributional)              │
│  Training-inference mismatch: NONE (IEM ASOS in both)       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  SYNTHESIS LAYER (Part II)                                    │
│  Training data: GEFSv12 Reforecast 2000-2019                 │
│                 + Operational GFS 2021-2024                   │
│  Inputs: station_μ, station_σ,                               │
│          GFS_TMAX_F024, GFS_T850, GFS_wind, GFS_clouds,     │
│          GFS_ensemble_spread, station-GFS disagreement,      │
│          recent GFS bias, season                             │
│  Output: μ_synth, σ_synth (calibrated distribution)         │
│  Training-inference mismatch: MINIMAL (reforecast ≈ op GFS) │
└──────────────────────────────────────────────────────────────┘
```

**Why this is the safest and most effective approach:**

1. **The station model has ZERO NWP dependency.** It trains on 26 years of IEM ASOS data. The same IEM ASOS data is available operationally at 6 AM ET. There is no training-inference mismatch at all. If the NWP pipeline breaks, the station model still produces a forecast.

2. **The synthesis layer trains on actual GFS forecasts.** Using GEFSv12 reforecast data (2000-2019), the synthesis model sees the same kind of GFS output it will receive in operations. The biases, noise, and error correlations are all realistic. No approximation needed.

3. **The one discontinuity is manageable.** The transition from GEFSv12 reforecast (frozen GFSv15.1) to operational GFS (v16, 2021+) introduces one model-version change. You can handle this by:
   - Including a binary `is_operational_gfs` feature
   - Training a small bias-correction offset on the 2021-2024 overlap
   - Or simply accepting the minor discontinuity (GFSv15.1 and v16 are architecturally similar)

4. **Ensemble spread is available.** The 5-member reforecast ensemble provides a physically-based uncertainty estimate. This is a powerful feature for the synthesis model: when GFS ensemble spread is large, the synthesis model should widen its output σ and weight the station model more heavily.

5. **Degradation is graceful.** If GFS data is delayed or missing on a given day, you can fall back to the station model alone. The synthesis layer is an enhancement, not a requirement.

**The training-vs-operations data source mapping (UPDATED — no ERA5):**

| Feature | Training Source | Operational Source (6 AM ET) | Mismatch? |
|---|---|---|---|
| Station TMAX/TMIN at t-1 | IEM ASOS hourly → daily | IEM ASOS hourly → daily | **None** |
| Station wind/dewpoint/pressure | IEM ASOS hourly | IEM ASOS hourly | **None** |
| Station precip/snow | GHCN-Daily | IEM ASOS (precip) / GHCN (snow, 1-2 day lag) | Minor |
| 850mb temp at t-1 | IGRA sounding (00Z, 12Z) | IGRA sounding (00Z) | **None** |
| NWP TMAX forecast for day t | GEFSv12 reforecast F024 (2000-2019) | GFS 00Z F024 (available ~1 AM ET) | **Minimal** |
| NWP T850 forecast | GEFSv12 reforecast (tmp_pres) | GFS 00Z (available ~1 AM ET) | **Minimal** |
| NWP ensemble spread | GEFSv12 5-member spread | GEFS 00Z spread (available ~2 AM ET) | Minor* |
| NWP cloud/wind forecast | GEFSv12 reforecast | GFS/HRRR 00Z or 06Z | **Minimal** |

*Operational GEFS has 31 members vs. reforecast's 5 — spread magnitude will differ. Normalize by number of members.

---

#### Option Comparison Summary

| Criterion | A: GEFSv12 Reforecast | B: ERA5 + Noise | C: Hybrid (A's data) |
|---|---|---|---|
| Training-inference mismatch | Minimal (frozen GFSv15.1 ≈ operational GFSv16) | **Severe** (analysis ≠ forecast) | Minimal for both layers |
| Historical coverage | 2000-2019 (20 years) | 1979-present (45 years) | Station: 26yr; NWP: 24yr |
| Ensemble spread available | Yes (5 members) | No | Yes |
| Consistent error structure | Yes (single model version) | N/A (ERA5 has no forecast error) | Yes |
| Captures GFS cold bias | Yes (same bias as operational) | No (ERA5 bias is ~0.2°C) | Yes |
| Captures error autocorrelation | Yes (real forecast errors) | No (random noise has zero autocorr.) | Yes |
| Fallback if NWP unavailable | No fallback | No fallback | **Station model runs independently** |
| Implementation complexity | Medium | Low but flawed | Medium (two-phase training) |
| **Verdict** | **Good data source** | **Do not use** | **Best architecture** |

**Final recommendation: Use Option C (Hybrid architecture) with Option A (GEFSv12 Reforecast) as the NWP data source for the synthesis layer. Eliminate ERA5 from the production pipeline entirely.** ERA5 remains useful as a research/validation tool but is not needed for training or operations.

### 9. Synthesis Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHESIS MODEL                          │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │  STATION MODEL       │    │  NWP FEATURES        │        │
│  │  (your optimized     │    │  (GEFSv12 reforecast  │        │
│  │   model from Part I) │    │   train, GFS for ops) │        │
│  │                      │    │                       │        │
│  │  Outputs:            │    │  Features:            │        │
│  │  - μ_station         │    │  - NWP_TMAX_forecast  │        │
│  │  - σ_station         │    │  - NWP_T850           │        │
│  │  - attention_weights │    │  - NWP_cloud_cover    │        │
│  │  - regime_indicators │    │  - NWP_ensemble_spread│        │
│  └──────────┬──────────┘    │  - NWP_recent_bias    │        │
│             │               │  - NWP_wind_dir/speed │        │
│             │               └──────────┬────────────┘        │
│             │                          │                     │
│             └──────────┬───────────────┘                     │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │   META-LEARNER     │                          │
│              │                    │                          │
│              │  Inputs:           │                          │
│              │  - station_μ, σ    │                          │
│              │  - NWP_TMAX        │                          │
│              │  - NWP_spread      │                          │
│              │  - NWP_recent_bias │                          │
│              │  - station_NWP_gap │                          │
│              │  - regime features │                          │
│              │  - day-of-year     │                          │
│              │                    │                          │
│              │  Architecture:     │                          │
│              │  Small MLP [64,32] │                          │
│              │  or GBM (XGBoost)  │                          │
│              │                    │                          │
│              │  Output:           │                          │
│              │  μ_synth, σ_synth  │                          │
│              │  (or mixture params│                          │
│              └─────────┬─────────┘                          │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │  CALIBRATION LAYER │                          │
│              │  (isotonic regr.)  │                          │
│              │                    │                          │
│              │  Input: raw CDF    │                          │
│              │  Output: calibrated│                          │
│              │  bucket probs      │                          │
│              └─────────┬─────────┘                          │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │  TRADING LAYER     │                          │
│              │  P(bucket) vs.     │                          │
│              │  market price →    │                          │
│              │  EV → Kelly → bet  │                          │
│              └───────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 10. Key Meta-Learner Features

The meta-learner's job is to learn **when to trust the station model vs. NWP**, and to produce a distribution better than either alone.

| Feature | Purpose |
|---|---|
| `station_mu` | Station model's point forecast |
| `station_sigma` | Station model's uncertainty estimate |
| `nwp_tmax` | NWP point forecast for NYC TMAX |
| `nwp_spread` | NWP ensemble spread (uncertainty) |
| `station_nwp_gap` | `station_mu - nwp_tmax` (disagreement signal — large gaps suggest one source is wrong) |
| `nwp_recent_bias` | NWP's MAE over the last 7 days (is NWP running hot or cold?) |
| `station_recent_bias` | Station model's MAE over last 7 days |
| `nwp_t850` | 850mb temperature forecast (thermal advection signal) |
| `nwp_cloud_cover` | Forecast cloud cover (radiation budget) |
| `nwp_wind_dir`, `nwp_wind_speed` | Forecast wind (advection direction for next day) |
| `abs_station_nwp_gap` | |station_mu - nwp_tmax| — large disagreement → increase σ |
| `sin_day`, `cos_day` | Season (NWP bias is seasonal) |
| `is_extreme` | Is either forecast near a climatological extreme? |
| `front_detected` | Station model's frontal passage indicator |

**Architecture choices for the meta-learner:**

| Option | Pros | Cons |
|---|---|---|
| **Small MLP [64, 32]** | Learns nonlinear interactions | Needs careful regularization |
| **XGBoost/LightGBM** | Handles feature interactions natively, resistant to overfitting | Harder to output distribution parameters |
| **Bayesian linear regression** | Closed-form uncertainty, interpretable weights | Can't capture nonlinear interactions |
| **Stacking with isotonic calibration** | Simple, robust, well-studied | Limited expressiveness |

**Recommended:** Start with a Bayesian linear combination (BMA-style) to establish a baseline, then try a small MLP.

### 11. Where the Synthesis Creates Edge

The synthesis model can beat both individual sources because:

1. **NWP is biased, station model detects bias.** NWP has known systematic errors (urban heat island underestimation, coastal temperature gradient smoothing). The station model sees the actual local temperature and can estimate how much NWP is wrong on this specific day type.

2. **Station model captures local regime, NWP captures large-scale evolution.** The station model knows "NYC temperatures have been anomalously warm for 3 days" (persistence/regime). NWP knows "a cold front will pass through tonight" (physics). Neither alone has both signals.

3. **Disagreement = information.** When the station model and NWP disagree strongly, this is itself a signal. The synthesis model learns that large disagreements correspond to higher forecast uncertainty (wider σ) and that the direction of the resolution depends on the weather regime.

4. **Calibration beats accuracy.** For trading, you don't need to predict the exact temperature — you need accurate probabilities. Even if NWP has lower MAE, your synthesis model can produce better-calibrated tail probabilities because it's trained on CRPS with post-hoc calibration. Many NWP products have poor calibration despite good MAE.

**Expected synthesis performance:** If the optimized station model achieves ~2.3°F MAE and GFS achieves ~2.5°F MAE individually, the synthesis should achieve ~1.8-2.1°F MAE with substantially better CRPS and calibration than either source alone. This is based on the literature finding that multi-model synthesis typically reduces CRPS by 15-25% vs. the best individual source.

---

## Part III: Revised Execution Plan

### Phase 0: Scale Up Data (Prerequisite)
- Download IEM ASOS hourly data for all 14 core stations (2000-2024) — this becomes the PRIMARY data source for both training and operations
- Compute daily TMAX, TMIN, mean wind dir/speed, mean dewpoint, mean SLP, cloud fraction from IEM hourly
- Cross-validate IEM-derived TMAX against GHCN-Daily TMAX; quantify and document systematic differences
- Download GHCN-Daily with expanded elements (PRCP, SNOW, SNWD) for training period — used for precip/snow features and as a cross-check
- Download IGRA sounding data for OKX/Upton (2000-2024) via University of Wyoming or Siphon
- Download GEFSv12 Reforecast (2000-2019) from AWS S3 for NYC grid point: tmax_2m, tmp_pres (850mb), ugrd_10m, vgrd_10m, tcdc_eatm, pres_msl — all 5 ensemble members for spread computation. Use Herbie library for efficient partial downloads.
- Download operational GFS (2021-2024) from AWS for same variables to bridge the gap
- **Deliverable:** Unified multi-source dataset with ~9,000 days (24 years) of complete features, all derived from the same sources that will be used operationally at 6 AM ET. Zero ERA5 dependency.

### Phase 1: Optimize Station Model
- Implement wind-conditioned features (upwind temperature, advection rate) from IEM ASOS
- Implement dewpoint, pressure, cloud cover features from IEM ASOS
- Implement precipitation and snow features from GHCN (training) / IEM (operations)
- Implement 850mb temperature features from IGRA soundings (available by 6 AM ET)
- Train wind-gated attention model on full 26-year IEM-based dataset
- Add mixture density output head (2-component Gaussian)
- Train with CRPS loss
- Verify all features can be computed from data available by 6 AM ET
- **Target:** MAE ≤ 2.5°F, well-calibrated PIT histogram

### Phase 2: Build Synthesis Layer
- Prepare GEFSv12 reforecast features (2000-2019) + operational GFS features (2021-2024) for synthesis training
- Train meta-learner on (station_model_output, GFS_features) → TMAX distribution
- Evaluate synthesis vs. station-only and GFS-only
- Apply isotonic calibration per Kalshi bucket
- Validate on held-out 2024 test set
- **Target:** MAE ≤ 2.0°F, CRPS ≤ 1.8°F, calibrated bucket probabilities

### Phase 3: Build Trading Infrastructure
- Study KXHIGHNY market structure (buckets, liquidity, spreads, timing)
- Build EV computation + Kelly sizing module
- Build backtesting framework against historical market data
- Compute minimum edge required for profitability
- Paper trade for 30-60 days
- **Target:** Positive EV after fees on >30% of trading days

### Phase 4: Operationalize
- No data source switching needed — synthesis layer already trains on GFS, deploys on GFS
- Build daily cron pipeline running at 6:00 AM ET:
  ```
  6:00 AM ET Daily Pipeline:
  1. Pull IEM ASOS hourly data for all stations through 05:00 local (10Z)
     → compute yesterday's TMAX, wind, dewpoint, pressure, cloud fraction
  2. Pull 00Z IGRA sounding from OKX (available since ~10 PM yesterday)
     → extract 850mb temp, wind, stability
  3. Pull 00Z GFS F024 TMAX forecast for NYC grid point (available since ~1 AM)
     → extract NWP TMAX, wind, cloud, ensemble spread (GEFS)
  4. Optionally pull 06Z HRRR or 09Z HRRR for latest mesoscale update
  5. Compute all features; run station model → μ_station, σ_station
  6. Run synthesis model (station output + GFS features) → calibrated distribution
  7. Convert to Kalshi bucket probabilities
  8. Pull current KXHIGHNY orderbooks; compute EV per bucket
  9. Execute trades where edge > threshold
  10. Log all inputs, predictions, and trades
  ```
- Implement monitoring, drift detection, and automatic halt conditions
- Go live with minimal position sizes
- **Target:** Sustained positive P&L over 90-day rolling window

---

## Part IV: Sensitivity Experiments to Run

Once the data and infrastructure are in place, run these experiments systematically to find the optimal configuration:

### Feature Ablation (Most Important)

| Experiment | Features | Expected Insight |
|---|---|---|
| Temperature only (current) | TMAX, TMIN, date | Baseline for comparison |
| + ASOS wind | Add wind dir/speed | Quantify wind signal value |
| + Wind-conditioned features | Add upwind_temp, advection_rate | Quantify physics-informed engineering value |
| + Dewpoint/humidity | Add dewpoint, depression | Quantify moisture signal |
| + Pressure/SLP | Add SLP, tendency | Quantify synoptic signal |
| + Cloud proxy | Add cloud fraction | Quantify radiation signal |
| + PRCP/SNOW | Add precipitation, snow depth | Quantify precip/albedo signal |
| + 850mb temp | Add sounding data | Quantify upper-air signal |
| Full observation model | All of the above | Maximum station-based performance |

### Architecture Comparison

| Architecture | Features | Expected Insight |
|---|---|---|
| Ridge regression | Full features | Linear baseline with rich features |
| MLP [128, 64] | Full features | Nonlinear capacity without attention |
| Wind-gated attention | Full features + metadata | Does attention mechanism help? |
| Multi-lag attention (3-day) | Full features, 3 days | Does temporal depth help? |
| Mixture density MLP | Full features, GMM output | Distributional vs. point output |

### Synthesis Experiments

| Experiment | Inputs | Expected Insight |
|---|---|---|
| Station model only | Station features | Part I maximum |
| GFS reforecast only (linear) | GFS TMAX, T850, wind, clouds | NWP-only baseline |
| GFS reforecast only (MLP) | Same | Nonlinear NWP post-processing |
| Synthesis (BMA) | Station μ + GFS TMAX | Linear combination baseline |
| Synthesis (MLP) | Station output + GFS features | Full synthesis capacity |
| Synthesis + calibration | Same + isotonic | Calibrated final product |

---

## Summary: Why This Path Can Win

| Advantage | Detail |
|---|---|
| **Hyper-local specialization** | NWP forecasts the globe; your model forecasts one station with 40 years of local ground truth |
| **Wind-conditioned physics** | The explicit wind → station weighting mechanism is something neither NWP nor generic ML approaches do well for individual station forecasting |
| **Multi-source synthesis** | Combining independent information sources (station obs + NWP) provably outperforms either alone |
| **Calibration focus** | Training on CRPS + post-hoc isotonic calibration produces better bucket probabilities than any point-forecast-based approach |
| **Tail event detection** | Station observations can detect approaching extreme events (large upstream gradients, rapid pressure drops) that NWP may under-predict |
| **Local bias correction** | 40 years of data at one station reveals systematic NWP biases that operational MOS may not fully correct |

**The key insight:** You're not trying to outforecast NWP in general. You're trying to produce a better **probability distribution for one specific station on one specific day** by combining NWP's physics with your network's local knowledge. That's a much more tractable problem, and the academic literature strongly supports that it's achievable.
