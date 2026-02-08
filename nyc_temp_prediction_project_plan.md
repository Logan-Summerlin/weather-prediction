# NYC Daily Max Temperature Prediction — Neural Network Project Plan

## 1. Research Concept Summary

**Objective:** Build a neural network that predicts the daily maximum temperature (°F) in New York City on day *t*, using daily temperature observations from surrounding cities/stations on day *t−1*.

**Core hypothesis:** Weather patterns propagate geographically. Temperatures observed yesterday at stations surrounding NYC contain predictive signal for NYC's temperature today. A neural network can learn optimal weightings of these surrounding-station inputs to minimize prediction error.

**Target metric:** Mean Absolute Error (MAE) in °F on a held-out test set, with a stretch goal of ≤ 2°F MAE. Secondary metric: percentage of predictions within ±1°F of actual.

**Extension goal:** Produce a 95% prediction interval for each forecast.

---

## 2. Data Source: NOAA GHCN-Daily

The **Global Historical Climatology Network — Daily (GHCNd)** is the ideal data source. It is free, quality-controlled, and provides daily TMAX, TMIN, and other variables for thousands of U.S. stations going back over 100 years.

### Key Details

- **Variables available:** TMAX (daily max temp), TMIN (daily min temp), TAVG (where available), PRCP, SNOW, SNWD
- **Format:** Values in tenths of °C (divide by 10 to get °C, then convert to °F)
- **Access methods:**
  - Bulk download: `https://www.ncei.noaa.gov/pub/data/ghcn/daily/`
  - NOAA CDO API v2: `https://www.ncei.noaa.gov/cdo-web/api/v2/` (requires free API token)
  - Python library: `noaa_sdk` or direct HTTP requests
- **Station metadata file:** `ghcnd-stations.txt` — lists station ID, lat, lon, elevation, name

### API Token

Sign up at `https://www.ncdc.noaa.gov/cdo-web/token` for a free NOAA CDO API token. Rate limit: 5 requests/second, 10,000 requests/day.

---

## 3. Station Selection Strategy

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

---

## 4. Data Collection Period

### Phase 1 (Proof of Concept): 5 years
- **Period:** 2018-01-01 to 2022-12-31
- **Purpose:** Validate the pipeline, train an initial model, assess feasibility
- **Approx. rows per station:** ~1,826

### Phase 2 (Full Model): 40 years
- **Period:** 1985-01-01 to 2024-12-31
- **Purpose:** Train a robust model with more seasonal cycles and weather variability
- **Approx. rows per station:** ~14,610
- **Status:** Data collection complete for all 51 stations. See data completeness findings below.

### Data Completeness Findings (Phase 2)
- **15 original stations:** Mostly 99–100% complete over the full 40-year period.
- **30 expanded stations:** Below 80% completeness (many started reporting circa 1997–2000).
- **21 stations total** meet the ≥80% completeness threshold over the 40-year range.
- Station completeness should be re-evaluated when selecting the final station set for Phase 6 model training.

---

## 5. Neural Network Architecture

### 5.1 Baseline: Simple Feedforward Network

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

### 5.2 Enhanced: Multi-Feature Input and Target Formulation

#### 5.2a ΔT (Delta) Target — Highest-Leverage Change

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

#### 5.2b Per-Station Feature Expansion

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

#### 5.2c Sector Averages and Gradients (Physics-Shaped Features)

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

#### 5.2d Trend Features (Front-Timing Proxy)

Compute short differences per sector or per top stations:

```
Δ1 = T(t−1) − T(t−2)   # yesterday's change
Δ2 = T(t−2) − T(t−3)   # day-before-yesterday's change
```

These capture direction and momentum of temperature changes, helping detect approaching fronts.

#### 5.2e Static Station Metadata

Add per-station metadata as features (especially useful with attention-based architectures):
- Elevation (meters)
- Distance to Central Park (km)
- Bearing from Central Park (degrees)
- Station type flag (airport vs. other)

### 5.3 Advanced: Architecture Upgrades (Attention Pooling and Temporal Models)

Test these architectures in order, stopping when gains plateau:

#### 5.3a Station Embeddings + Attention Pooling (recommended for 20+ stations)

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

#### 5.3b k-Day Window MLP

Concatenate features from days [t−1, …, t−k] into a single flat input vector:

```
Input: k × (N_stations × features_per_station + engineered_features)
     |
MLP with 2–3 hidden layers
     |
Output: 1 neuron (ΔT or TMAX prediction)
```

Often surprisingly strong when combined with engineered gradient/trend features. Test before adding sequence model complexity.

#### 5.3c 1D Temporal Convolution

Lightweight and stable temporal modeling:

```
Input shape: (k_days, pooled_station_features)
     |
1D Conv layers (kernel size 2–3)
     |
Dense output head
```

#### 5.3d LSTM/GRU (use only if k-window models plateau)

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

#### 5.3e Station Embeddings + Temporal Attention (Transformer-style)

Instead of treating each station as a fixed input column, learn a small **embedding per station** and use a **temporal attention / Transformer-style encoder** over the last *k* days. This can capture time-varying "which stations matter today" behavior without requiring a spatial grid.

- Inputs: (station_id, features at t-1..t-k)
- Model: station embedding + per-day feature projection → temporal attention → pooled representation → prediction head
- Evaluation: keep the same headline comparisons and permutation-importance sector checks.

### 5.4 Confidence Intervals

Two approaches:

**Approach A — Quantile Regression:**
Train two additional output heads that predict the 2.5th and 97.5th percentiles using pinball (quantile) loss. This directly gives a 95% prediction interval.

**Approach B — MC Dropout:**
Use dropout layers during both training and inference. At prediction time, run 100 forward passes with dropout active, producing a distribution of predictions. Use the 2.5th and 97.5th percentiles of that distribution.

### 5.5 Residual Learning / Stacking Ensemble

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

### 5.6 Season/Regime Specialization

If seasonal MAE disparity remains large after other enhancements, try specialization:

**Option A — Two seasonal models:**
- Cool season: DJF + MAM
- Warm season: JJA + SON

**Option B — Mixture-of-experts gate (single unified training):**
- Small gating network uses day-of-year sin/cos + sector gradients
- Gate blends two small expert sub-networks

**Report:** Season-wise MAE and "transition months" performance (Apr/May, Sep/Oct).

---

## 6. Implementation Plan — Step-by-Step

### Step 1: Environment Setup

```bash
pip install pandas numpy scikit-learn torch matplotlib requests
```

**Files to create:**
- `config.py` — station IDs, date ranges, API token, hyperparameters
- `data_collection.py` — download data from NOAA
- `data_preprocessing.py` — clean, align, feature-engineer
- `model.py` — PyTorch model definition
- `train.py` — training loop
- `evaluate.py` — evaluation metrics, plotting
- `predict.py` — make predictions on new data
- `confidence_intervals.py` — quantile regression or MC dropout

### Step 2: Data Collection (`data_collection.py`)

> **Implemented method:** Bulk `.dly` file downloads from `https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly` are used (no NOAA API token required). The fixed-width `.dly` format is parsed directly. The `run_collect_all_stations.py` script handles downloading and parsing for all 51 stations. This approach avoids all API rate-limit issues.

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

1. Load all station CSVs.
2. For each station:
   a. Pivot so each row is one date, columns are TMAX, TMIN.
   b. Convert from tenths of °C to °F if using raw .dly files:
      temp_f = (value / 10) * 9/5 + 32
   c. Flag/remove quality-flagged observations.
3. Merge all stations into one DataFrame:
   - Index: DATE
   - Columns: {station_id}_TMAX, {station_id}_TMIN, ... for each station
4. Create target column: NYC_TMAX (Central Park TMAX on day t)
5. Create feature columns: all other stations' values shifted by +1 day
   (i.e., feature row for day t uses surrounding stations' data from day t-1)
6. Add cyclical date features:
   day_of_year = date.timetuple().tm_yday
   sin_day = sin(2π × day_of_year / 365.25)
   cos_day = cos(2π × day_of_year / 365.25)
7. Handle missing data:
   - Drop days where NYC target is missing.
   - For input stations: forward-fill gaps ≤ 3 days,
     then use column mean for remaining NaNs (or drop row).
8. Train/val/test split:
   - Train: first 70% of dates (chronological!)
   - Validation: next 15%
   - Test: final 15%
   IMPORTANT: Do NOT shuffle. Must respect temporal order.
9. Normalize features:
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

1. MAE (Mean Absolute Error) — primary metric
2. RMSE (Root Mean Squared Error)
3. Percentage of predictions within ±1°F
4. Percentage of predictions within ±2°F
5. Percentage of predictions within ±3°F
6. R² score
7. Bias (mean error, to detect systematic over/under-prediction)
8. Seasonal breakdown: compute MAE for each season
   (winter=DJF, spring=MAM, summer=JJA, fall=SON)

Plots to generate:
- Actual vs. Predicted scatter plot (with y=x line)
- Time series plot of actual vs. predicted for a sample month
- Residual histogram
- Residuals by month (box plot)
- If using confidence intervals: coverage plot showing
  what % of actuals fall within the 95% interval
"""
```

### Step 7: Sensitivity Analysis

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

## 7. Benchmark / Baseline Models

To assess whether the neural network adds value, compare against these baselines:

1. **Persistence model:** Predict NYC TMAX(t) = NYC TMAX(t-1). This is the simplest possible baseline.
2. **Climatological average:** Predict NYC TMAX(t) = historical average TMAX for that calendar day.
3. **Linear regression:** Same inputs as the NN, but fit with ordinary least squares.
4. **Ridge/Lasso regression:** Regularized linear regression (guards against multicollinearity between nearby stations).

The neural network should meaningfully outperform these to justify its use.

---

## 8. Project File Structure

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
│   ├── raw/                 # Raw downloaded station files
│   ├── processed/           # Cleaned, merged DataFrames
│   └── stations.csv         # Selected station metadata
├── notebooks/
│   ├── 01_explore_stations.ipynb
│   ├── 02_data_quality.ipynb
│   └── 03_results_analysis.ipynb
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── confidence_intervals.py
├── models/                  # Saved model checkpoints
├── results/                 # Plots, metric summaries
├── requirements.txt
└── README.md
```

---

## 9. Implementation Sequence and Timeline

| Phase | Task | Est. Effort | Status |
|-------|------|-------------|--------|
| **Phase 1** | **Data Pipeline** | | **COMPLETE** |
| 1.1 | Set up environment and project structure | 1 hour | COMPLETE |
| 1.2 | Get NOAA API token, identify target + surrounding stations | 2 hours | COMPLETE |
| 1.3 | Write data collection script (download 5 years for ~20 stations) | 3 hours | COMPLETE |
| 1.4 | Write preprocessing: merge, align, feature-engineer, split | 4 hours | COMPLETE |
| 1.5 | Exploratory data analysis notebook (completeness, correlations) | 2 hours | COMPLETE |
| **Phase 2** | **Baseline Models** | | **COMPLETE** |
| 2.1 | Implement persistence and climatology baselines | 1 hour | COMPLETE |
| 2.2 | Implement linear/ridge regression baselines | 2 hours | COMPLETE |
| 2.3 | Evaluate baselines, establish benchmarks | 1 hour | COMPLETE |
| **Phase 3** | **Neural Network — V1** | | **COMPLETE** |
| 3.1 | Build feedforward NN (simple: TMAX-only inputs) | 2 hours | COMPLETE |
| 3.2 | Train, tune hyperparameters on validation set | 3 hours | COMPLETE |
| 3.3 | Evaluate on test set, compare to baselines | 1 hour | COMPLETE |
| **Phase 4** | **Enhancements** (run in order; stop when gains plateau) | | **4.1–4.3 COMPLETE** |
| 4.1 | Switch to ΔT target + include NYC TMAX(t−1) as input; use Huber loss; evaluate MAE on reconstructed TMAX | 3 hours | COMPLETE |
| 4.2 | Feature engineering: add sector averages/gradients, trend features (Δ1/Δ2), TMIN, diurnal range, station metadata | 4 hours | COMPLETE |
| 4.3 | Station expansion: add ~50 stations (rings × sectors), implement imputation + missingness masking. Geography-driven gap analysis completed; 2 gap-filling stations added (S near-field, SW Ring3); ESE gap confirmed unfillable. | 4 hours | COMPLETE |
| 4.4 | Sensitivity experiments: vary station count (20–70), radius (150–250 mi), lag (t−1 … t−3), input type, loss function, autoregressive input, date encoding | 4 hours | NOT STARTED |
| 4.5 | Architecture upgrades: station embeddings + attention pooling → k-window MLP → 1D temporal conv → LSTM/GRU (only if earlier models plateau) | 5 hours | NOT STARTED |
| 4.6 | Residual learning / stacking: train NN residual corrector on base-model predictions + engineered features | 3 hours | NOT STARTED |
| 4.7 | Season/regime specialization: two-expert seasonal models or mixture-of-experts gate (only if seasonal MAE disparity remains large) | 2 hours | NOT STARTED |
| **Phase 5** | **Confidence Intervals** | | **NOT STARTED** |
| 5.1 | Implement quantile regression model | 2 hours | NOT STARTED |
| 5.2 | Evaluate coverage (does 95% interval capture ~95% of actuals?) | 1 hour | NOT STARTED |
| **Phase 6** | **Scale Up** | | **Data collection COMPLETE; model retraining NOT STARTED** |
| 6.1 | Extend to 40 years of data (1985–2024) | 2 hours | COMPLETE (data collected for all 51 stations) |
| 6.2 | Retrain best model, compare to 5-year version | 2 hours | NOT STARTED |
| **Phase 7** | **Documentation and Reporting** | | **NOT STARTED** |
| 7.1 | Write up results, generate final plots | 3 hours | NOT STARTED |

**Total estimated effort: ~56 hours**

---

## 10. Potential Challenges and Mitigations

**Missing data:** Some stations may have gaps. Mitigation: select stations with ≥90% completeness; use forward-fill for short gaps; for remaining NaNs, experiment with imputation vs. dropping rows.

**Multicollinearity:** Nearby stations are highly correlated. Mitigation: Ridge/Lasso baselines will reveal this; the NN should handle it implicitly, but monitor weight magnitudes.

**Non-stationarity / Climate trends:** Long-term warming may affect older data. Mitigation: include year or decade as a feature, or detrend temperatures before training.

**Seasonal heterogeneity:** Model may perform differently across seasons (e.g., worse in spring when weather is more volatile). Mitigation: evaluate per-season; consider training separate seasonal models if performance varies widely.

**Overfitting:** With many input features relative to the signal. Mitigation: dropout, early stopping, regularization, and a clean temporal train/val/test split.

**API rate limits:** NOAA CDO API limits to 10,000 requests/day. Mitigation: use bulk `.dly` file downloads from the GHCN FTP server instead, which is faster and unlimited.

---

## 11. Getting Started — First Commands

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

# 6. Run preprocessing
python src/data_preprocessing.py

# 7. Train baseline models
python src/train.py --model baseline

# 8. Train neural network
python src/train.py --model nn_v1
```

---

## 12. Starter Code: `config.py`

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
