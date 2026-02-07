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

### Phase 2 (Full Model): 25+ years
- **Period:** 2000-01-01 to 2024-12-31
- **Purpose:** Train a robust model with more seasonal cycles and weather variability
- **Approx. rows per station:** ~9,131

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

### 5.2 Enhanced: Multi-Feature Input

Instead of one feature per station, use multiple features per station:

```
Per station (at t-1):
  - TMAX (max temperature)
  - TMIN (min temperature)
  - TMAX - TMIN (diurnal range)

Total input features: N_stations × 3

Optional additional inputs:
  - Day of year (encoded as sin/cos for cyclical nature)
  - NYC's own TMAX at t-1 (autoregressive term)
```

This gives the network more information to work with.

### 5.3 Advanced: Sequence Model (LSTM/GRU)

For a later phase, feed the network a window of *k* days (e.g., t-3 through t-1) for each station, using an LSTM or GRU to capture temporal trends.

```
Input shape: (k_days, N_stations × features_per_station)
     |
LSTM/GRU Layer: 64 units
     |
Dense Layer: 32 neurons, ReLU
     |
Output: 1 neuron (predicted NYC TMAX at day t)
```


**Additional advanced option: station embeddings + temporal attention**

Instead of treating each station as a fixed input column, learn a small **embedding per station** and use a **temporal attention / Transformer-style encoder** over the last *k* days. This can capture time-varying “which stations matter today” behavior without requiring a spatial grid.

- Inputs: (station_id, features at t-1..t-k)
- Model: station embedding + per-day feature projection → temporal attention → pooled representation → prediction head
- Evaluation: keep the same headline comparisons and permutation-importance sector checks.

### 5.4 Confidence Intervals

Two approaches:

**Approach A — Quantile Regression:**
Train two additional output heads that predict the 2.5th and 97.5th percentiles using pinball (quantile) loss. This directly gives a 95% prediction interval.

**Approach B — MC Dropout:**
Use dropout layers during both training and inference. At prediction time, run 100 forward passes with dropout active, producing a distribution of predictions. Use the 2.5th and 97.5th percentiles of that distribution.

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

```python
"""
Pseudocode for data collection.

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
| Input temperature type | TMAX only, TMIN only, both TMAX+TMIN, average of TMAX/TMIN |
| Number of surrounding stations | 5, 10, 15, 20, 25 (assess marginal value of adding stations) |
| Radius of stations | 50mi only, 100mi, 150mi, 200mi |
| Lag structure | t-1 only, t-1 and t-2, t-1 through t-3 |
| Architecture | Linear regression baseline, 1-hidden-layer NN, 2-hidden-layer NN, LSTM/GRU, temporal attention (Transformer-style) |
| Cyclical date encoding | With/without sin/cos day-of-year features |
| Month/season indicator | With/without one-hot month encoding |

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

| Phase | Task | Est. Effort |
|-------|------|-------------|
| **Phase 1** | **Data Pipeline** | |
| 1.1 | Set up environment and project structure | 1 hour |
| 1.2 | Get NOAA API token, identify target + surrounding stations | 2 hours |
| 1.3 | Write data collection script (download 5 years for ~20 stations) | 3 hours |
| 1.4 | Write preprocessing: merge, align, feature-engineer, split | 4 hours |
| 1.5 | Exploratory data analysis notebook (completeness, correlations) | 2 hours |
| **Phase 2** | **Baseline Models** | |
| 2.1 | Implement persistence and climatology baselines | 1 hour |
| 2.2 | Implement linear/ridge regression baselines | 2 hours |
| 2.3 | Evaluate baselines, establish benchmarks | 1 hour |
| **Phase 3** | **Neural Network — V1** | |
| 3.1 | Build feedforward NN (simple: TMAX-only inputs) | 2 hours |
| 3.2 | Train, tune hyperparameters on validation set | 3 hours |
| 3.3 | Evaluate on test set, compare to baselines | 1 hour |
| **Phase 4** | **Enhancements** | |
| 4.1 | Add multi-feature inputs (TMAX + TMIN + date encoding) | 2 hours |
| 4.2 | Run sensitivity experiments (station count, radius, lag) | 4 hours |
| 4.3 | Try LSTM/sequence model if warranted | 3 hours |
| **Phase 5** | **Confidence Intervals** | |
| 5.1 | Implement quantile regression model | 2 hours |
| 5.2 | Evaluate coverage (does 95% interval capture ~95% of actuals?) | 1 hour |
| **Phase 6** | **Scale Up** | |
| 6.1 | Extend to 25 years of data | 2 hours |
| 6.2 | Retrain best model, compare to 5-year version | 2 hours |
| **Phase 7** | **Documentation and Reporting** | |
| 7.1 | Write up results, generate final plots | 3 hours |

**Total estimated effort: ~40 hours**

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
