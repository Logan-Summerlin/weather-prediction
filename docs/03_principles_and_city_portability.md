# High-Level Principles and US City Portability Guide

> **Last Updated:** 2026-02-19

---

## Part 1: Core Modeling Principles

### Principle 1: Contract-Definition-First

Every modeling decision starts from the Kalshi contract specification, not from general weather forecasting conventions.

**What this means:**
- The settlement station, day boundary, timezone, units, and rounding must be confirmed before any data is collected.
- Bucket edges come from the actual Kalshi contract listings, not from arbitrary choices.
- The scoring metric (Contract Brier) evaluates only actually-listed contracts, not a uniform grid.
- Open-tail semantics (below/above buckets) must match the exchange's settlement logic.

**Why it matters:** A model that is well-calibrated in general but misaligned with contract definitions will lose money. Contract alignment is checked first and continuously verified.

### Principle 2: Time-Safe Parity

Training and inference must use identical information sets. No feature may use data unavailable at prediction time.

**What this means:**
- All features use lagged values (day t-1 or earlier) for predicting day t.
- The operational cutoff (e.g., 6 AM ET) defines what data is available.
- Sources with delayed availability (e.g., finalized GHCN records) cannot appear in live inference unless the delay is modeled.
- Feature timestamps are explicitly tracked and audited.
- Chronological train/val/test splits enforce no future information leakage.

**Split Convention:**
| Partition | Range | Purpose |
|-----------|-------|---------|
| Train | 2000-01-01 to 2021-12-31 | Model fitting |
| Calibration | 2022-01-01 to 2023-12-31 | Post-hoc calibration fitting |
| Test | 2024-01-01 to 2025-12-31 | Final evaluation (never touched during development) |

### Principle 3: Distribution-First Forecasting

Models must output full probability distributions, not point predictions.

**What this means:**
- All models produce heteroscedastic Gaussian outputs: (mu, sigma) per day.
- Sigma varies by input conditions (weather regime, season, volatility).
- The Gaussian CDF maps (mu, sigma) to bucket probabilities.
- Point-only models are never shipped to production.
- The training objective is CRPS or NLL (distributional losses), not MAE alone.

**Distributional pipeline:**
```
Raw features → Neural network → (mu, sigma)
→ Gaussian CDF → bucket probabilities [p1, p2, ..., pN]
→ Calibration → adjusted probabilities
→ Renormalize (sum = 1)
→ Contract-row evaluation
```

### Principle 4: Calibration-First Evaluation

Calibration is mandatory before any trading decision. Uncalibrated probabilities are never traded.

**What this means:**
- The base model is frozen before the calibrator is fit.
- Calibrators are fit only on the calibration partition (2022-2023).
- Multiple calibration methods are compared: isotonic, Platt, Platt+isotonic, regime-conditional.
- Diagnostics include PIT histograms, reliability diagrams, ECE, coverage, and sharpness.
- Calibration drift is monitored in production.

**Calibration hierarchy:**
1. Raw model probabilities (typically overconfident)
2. Isotonic regression (nonparametric, flexible)
3. Platt scaling (smooth parametric)
4. Platt + isotonic (two-stage)
5. Regime-conditional (separate calibrators per season × volatility)

### Principle 5: Model Combination via Synthesis

Individual models are combined through meta-learners that leverage model diversity.

**What this means:**
- Multiple base models (E-series, WGA, baselines) each produce (mu, sigma) predictions.
- A synthesis layer (U-series) learns optimal combination weights.
- The synthesis layer can incorporate regime context to weight models differently in different conditions.
- The combination is fit on the calibration partition, preserving the separation from the test set.

**Synthesis hierarchy:**
- U0: Simple average
- U1-U3: Weighted/Ridge combinations
- U4-U6: Regime-conditional combinations with calibration
- U7: RegimeConditionalNet on ensemble outputs (best NYC model)
- U8-U9: Cross-validated and kitchen-sink variants

### Principle 6: Trading Discipline

Trades are placed only when expected value is positive after all costs.

**What this means:**
- Market prices are converted to implied probabilities.
- EV is computed for both YES and NO directions.
- Fees (Kalshi ~7%), bid-ask spread, slippage, and execution uncertainty are included.
- Position sizing uses fractional Kelly (0.25× Kelly) with per-contract and per-day caps.
- Kill switches halt trading on data gaps, schema mismatches, calibration drift, or risk limit breaches.
- Backtests use historical listed contracts, actual pre-settlement prices, and conservative slippage.

---

## Part 2: Five-Layer Architecture

The system is organized into five independently testable layers:

### Layer 1: Ingestion
**Purpose:** Collect and persist raw data from external sources.

**Sources:**
| Source | Module | Data | Availability |
|--------|--------|------|-------------|
| GHCN-Daily | `data_collection.py` | Historical station TMAX/TMIN/PRCP | Daily, ~1 day lag |
| ASOS Hourly | `asos_collection.py` | Hourly temperature, wind, dewpoint, pressure | Hourly, real-time |
| NWP (GFS/GEFS) | `nwp_collection.py` | Gridded numerical weather predictions | 4× daily, 0-6h lag |
| IGRA Soundings | `soundings_collection.py` | Upper-air profiles (00Z, 12Z) | Twice daily |
| Kalshi API | `kalshi_client.py` | Market prices, orderbooks, settlements | Real-time |

**Conventions:**
- Raw data persisted as immutable snapshots in `data/<city>/raw/`
- Schema and units validated on ingestion
- Source latency and completeness tracked
- Missing target station triggers a hard fail

### Layer 2: Feature Engineering
**Purpose:** Transform raw data into time-safe model inputs.

**Key Design Rules:**
- All predictors for day t use only data from day t-1 or earlier
- Cyclical time encoding: sin(2π·day/365), cos(2π·day/365)
- Persistence features: TMAX at lag-1, lag-2, lag-3
- Trend features: 3-day, 7-day rolling mean and standard deviation
- Seasonal climatology: monthly TMAX mean and std
- Station features: per-station TMAX with sector/ring aggregation
- Wind-conditioned features: upwind/downwind station composites (when wind data available)
- Missingness represented explicitly (indicator columns, not silent fills)

**Split Enforcement:**
- Scalers (StandardScaler) fit only on train partition
- No overlap between train, calibration, and test
- Split boundaries persisted in artifacts

### Layer 3: Forecasting
**Purpose:** Produce distributional predictions from features.

**Model Families:**
- **Baselines:** Persistence, climatology, Ridge regression (always included)
- **E-Series (E0-E22):** Feedforward NNs of increasing complexity
- **WGA (E38-E42):** Wind-gated attention with spatial station reasoning
- **U-Series (U0-U9):** Synthesis meta-learners combining multiple base models

**Training:**
- Loss: CRPS (primary), MAE (secondary), or combined
- Optimizer: Adam with learning rate scheduling
- Early stopping on validation loss
- 5-seed ensemble for uncertainty estimation
- Gradient clipping for attention models

### Layer 4: Calibration + Bucketization
**Purpose:** Convert model outputs to well-calibrated bucket probabilities.

**Steps:**
1. Freeze base model
2. Generate (mu, sigma) predictions on calibration set
3. Convert to bucket probabilities via Gaussian CDF
4. Fit calibrator (isotonic/Platt/regime-conditional) on calibration set
5. Apply calibrator to transform probabilities
6. Renormalize daily bucket probabilities to sum to 1
7. Validate: PIT histograms, reliability diagrams, ECE, coverage

### Layer 5: Trading + Risk
**Purpose:** Identify and execute profitable trading opportunities.

**Steps:**
1. Fetch live market orderbooks
2. Compute cost-adjusted EV for each contract
3. Apply sizing (fractional Kelly) with exposure caps
4. Check kill-switch conditions
5. Place qualified orders
6. Persist all decisions and audit trail

---

## Part 3: Creating a Model for Any US City

### Step 0: Verify Kalshi Contract Availability

Before any work, confirm that Kalshi lists temperature contracts for the target city:
- Identify the ticker family (e.g., KXHIGHNY, KXHIGHCHI, KXHIGHPHL)
- Confirm the settlement station (which specific GHCN station is used)
- Confirm bucket edges, open-tail semantics, day boundary, timezone
- Document in a contract alignment artifact

### Step 1: Build City Configuration

Create `config_<city>.py` with:

```python
# Required components:
TARGET_STATION = "USW000XXXXX"  # GHCN station ID
TARGET_LAT = XX.XXXX
TARGET_LON = -XX.XXXX
TIMEZONE = "America/<timezone>"

# Station network: ~40-60 GHCN stations in 4 rings
STATION_RINGS = {
    "ring1": [...],  # 0-50 miles (8-12 stations)
    "ring2": [...],  # 50-100 miles (10-15 stations)
    "ring3": [...],  # 100-150 miles (12-16 stations)
    "ring4": [...],  # 150-250 miles (10-18 stations)
}

# Meteorological sectors (directionally meaningful)
METEOROLOGICAL_SECTORS = {
    "cold_advection": [...],  # Direction of typical cold air
    "warm_advection": [...],  # Direction of typical warm air
    "local_effects":  [...],  # Water bodies, terrain
    "near_field":     [...],  # Immediate urban area
}

# Monthly climatology
MONTHLY_TMAX_MEAN = {1: XX.X, 2: XX.X, ..., 12: XX.X}
MONTHLY_TMAX_STD = {1: XX.X, 2: XX.X, ..., 12: XX.X}
```

Register the city in `src/city_config.py`:
- Add CityConfig entry with all fields
- Set appropriate bucket grid (floor/ceiling for local climate range)
- Set IGRA sounding station and NWP grid point

### Step 2: Station Discovery

Use `src/station_discovery.py` to find GHCN stations near the target:

1. Download GHCN station inventory
2. Filter by distance from target (haversine)
3. Classify into rings and sectors
4. Verify data availability (need 20+ years of TMAX records)
5. Select ~40-60 stations with good coverage

**Key considerations:**
- Cover all compass directions for advection signals
- Include nearby airports (often have best data quality)
- Identify city-specific meteorological features:
  - Lake-effect zones (Chicago, Cleveland, Buffalo)
  - Coastal moderation (NYC, Boston, Miami)
  - Mountain/valley effects (Denver, Phoenix)
  - Gulf moisture corridors (Houston, Atlanta)

### Step 3: Run Data Collection Pipeline

```bash
python scripts/run_<city>_data_collection.py
```

This invokes `src/data_collection.py` to:
1. Download GHCN-Daily .dly files for all stations
2. Parse fixed-width format to CSV
3. Validate schema, units, and completeness
4. Store in `data/<city>/raw/`

### Step 4: Run Preprocessing Pipeline

```bash
python scripts/run_<city>_preprocessing.py
```

This invokes `src/data_preprocessing.py` to:
1. Load station CSVs and merge on date
2. Apply QC filtering (remove bad flags, outliers)
3. Create lagged features (TMAX at t-1, t-2, t-3)
4. Add rolling statistics (3-day, 7-day mean/std)
5. Add cyclical time encoding
6. Add climatological features (monthly mean/std)
7. Apply chronological splits (train/cal/test)
8. Fit scaler on train partition only
9. Save features and targets to `data/<city>/processed/`

### Step 5: Run Benchmark Models

```bash
python scripts/run_<city>_benchmark.py
# Or use the multi-city template:
python scripts/run_city_nws_kalshi_template_benchmark.py --city <city>
```

This trains and evaluates:
1. Baselines: persistence, climatology, Ridge
2. E-series models: E0 (simple NN), E1-E5 (variants)
3. Advanced models: FeatureAttentionNet, MOSCorrectionNet, RegimeConditionalNet
4. Evaluate all on Contract Brier metric

### Step 6: Run Synthesis and Calibration

```bash
python scripts/run_<city>_synthesis_calibration.py
```

This:
1. Collects predictions from all base models
2. Trains U-series synthesis models (U0-U9)
3. Fits calibrators (isotonic, Platt, regime-conditional)
4. Generates calibration diagnostics
5. Selects best calibration method per model

### Step 7: Run Backtest

```bash
python scripts/run_<city>_backtest.py
```

This:
1. Loads calibrated model predictions on test set
2. Simulates Kalshi market using market proxy or real data
3. Computes EV for each contract on each day
4. Applies trading strategy with sizing and risk limits
5. Generates P&L curves, Sharpe ratio, drawdown analysis

### Step 8: Run Promotion Evaluation

```bash
python scripts/run_<city>_promotion_evaluation.py
```

This checks all promotion gates:

| Gate | Requirement |
|------|-------------|
| 1. Contract alignment | Verified against Kalshi specification |
| 2. Time-safety audit | No future leakage in features |
| 3. Calibration diagnostics | PIT, reliability, ECE acceptable |
| 4. Contract Brier vs baselines | Must beat persistence, climatology |
| 5. Contract Brier vs market | Competitive with Kalshi market |
| 6. Positive EV after costs | Net positive after fees, spread, slippage |
| 7. Drawdown within limits | Max drawdown < threshold |
| 8. Exposure within limits | Per-contract and per-day caps |
| 9. Kill switch validated | All triggers tested |
| 10. Reproducibility | Artifacts complete, pipeline deterministic |

All gates must pass for live deployment.

---

## Part 4: City-Specific Adaptations

### General Climate Considerations

When adapting to a new city, the key meteorological factors to consider:

**Temperature Variance:**
- Continental cities (Chicago, Denver) have 2-3× the variance of coastal cities (Miami, San Diego)
- Higher variance = wider bucket grids, harder calibration, higher potential edge

**Seasonal Regime Shifts:**
- Northern cities: Arctic outbreaks in winter create fat-tailed cold extremes
- Southern cities: Summer heat domes create persistent above-normal regimes
- Transitional cities (NYC, Philadelphia): Both cold and warm extremes matter

**Local Effects:**
- Lake-effect (Great Lakes cities): Lake temperature modulates nearby station temperatures by 5-15°F
- Coastal moderation (Atlantic/Pacific): Maritime air stabilizes temperatures
- Urban heat island: Airport stations may run 2-5°F warmer than rural
- Mountain/valley: Diurnal drainage flows, inversions

### NYC (Reference City)

- **Climate:** Mid-Atlantic maritime influence, moderate variance
- **Key Feature:** Dense station network within 100 miles provides strong interpolation
- **Challenge:** Occasional tropical influence (hurricanes, remnant moisture)
- **Bucket Grid:** 0°F to 110°F (57 buckets), same as PHL/ATL/AUS

### Chicago Adaptations

- **Climate:** Strong continental, lake-modulated
- **Bucket Grid:** -10°F to 110°F (62 buckets) — lower floor for Arctic outbreaks
- **Key Adaptations:**
  - WNW sector stations are critical for cold advection signals
  - Lake Michigan moderation (NE_Lake sector) can differ from O'Hare by 10°F+
  - Winter sigma calibration needs special attention (extreme cold tails)
  - Regime conditioning (U7) is high-value due to winter/summer divergence

### Philadelphia Adaptations

- **Climate:** Mid-Atlantic transitional, similar to NYC
- **Key Adaptations:**
  - Strong correlation with NYC allows transfer learning
  - Airport (USW00013739) may have urban heat island bias
  - Summer convective uncertainty (pop-up thunderstorms)
  - Delaware Valley corridor creates channeled wind effects
- **Challenge:** Narrow Brier edge (+0.0039) suggests market is already efficient for PHL

### Atlanta Adaptations

- **Climate:** Subtropical, Piedmont elevation (~1,050 ft)
- **Key Adaptations:**
  - Gulf moisture corridor from SW brings warm/humid air
  - Appalachian cold air damming from NW creates sharp frontal transitions
  - Summer convective maximum (afternoon thunderstorms reduce TMAX)
  - Less winter variance than northern cities

### Austin Adaptations

- **Climate:** Subtropical, Hill Country terrain
- **Key Adaptations:**
  - Gulf of Mexico moisture dominant in spring/summer
  - Blue norther events (dramatic cold fronts) in winter
  - Summer heat is persistent and less variable
  - Terrain effects from Texas Hill Country to the west

---

## Part 5: Invariants Across All Cities

These rules must hold regardless of city:

1. **Chronological evaluation only.** No random shuffles, no future leakage.
2. **Calibration is mandatory before trading.** Fitted only on calibration partition.
3. **Contract Brier is the primary metric.** Not bucket-day Brier, not MAE alone.
4. **Cost-aware EV is required.** Include fees, spread, slippage in all trade decisions.
5. **Kill switches are always armed.** Missing data, schema changes, or calibration drift halt trading.
6. **Complete audit logging.** Every prediction, decision, and order is persisted.
7. **Baselines must be beaten.** Complexity only justified by OOS improvement.
8. **Real data only.** Never fabricate, simulate, or proxy settlement data.
9. **Listed contracts only.** Score and trade only on contracts that actually exist.
10. **Risk limits always active.** Per-contract caps, per-day caps, drawdown limits.

---

## Part 6: Acceptance Testing

For any new city pipeline, run the following verification sequence:

```bash
# 1. Configuration tests
python -m pytest tests/test_city_config.py

# 2. Data ingestion
python scripts/run_<city>_data_collection.py

# 3. Preprocessing
python scripts/run_<city>_preprocessing.py

# 4. Kalshi data
python scripts/fetch_kalshi_presettlement_multi.py --city <city>

# 5. Benchmark
python scripts/run_city_nws_kalshi_template_benchmark.py --city <city>

# 6. Synthesis + calibration
python scripts/run_<city>_synthesis_calibration.py

# 7. Backtest
python scripts/run_<city>_backtest.py

# 8. Promotion evaluation
python scripts/run_<city>_promotion_evaluation.py
```

Each step must complete without errors. The promotion evaluation provides a pass/fail verdict for live deployment readiness.
