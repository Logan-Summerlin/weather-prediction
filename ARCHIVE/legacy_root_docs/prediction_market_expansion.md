# Prediction Market Expansion Plan: Chicago + Philadelphia + Dashboard

**Created:** 2026-02-15
**Scope:** Extend the NYC probabilistic temperature forecasting and trading pipeline to Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) Kalshi contracts, and build a unified operational dashboard.

---

## Executive Summary

The NYC pipeline (E0–E42, U0–U9) has demonstrated calibrated probabilistic forecasting with Brier scores competitive against NWS and Kalshi market-implied probabilities. This plan extends that proven architecture to two new cities and adds a real-time monitoring dashboard. The approach reuses the existing modular codebase, adapting city-specific elements (station networks, climate regimes, contract definitions) while keeping the core modeling, calibration, and trading infrastructure shared.

---

## Part 1: Chicago Model (KXHIGHCHI)

### 1.1 Contract Specification Research
- **Task:** Confirm KXHIGHCHI contract details on Kalshi
  - Settlement station and measurement standard
  - Bucket thresholds and boundary conventions (inclusive/exclusive)
  - Day boundary (local time vs UTC)
  - Settlement timing and data source
- **Deliverable:** `docs/chicago_contract_spec.md`

### 1.2 Station Network Design
- **Target station:** O'Hare International Airport (USW00094846)
  - Lat: 41.9742, Lon: -87.9073
  - Elevation: 662 ft
- **Surrounding station search:**
  - Use `src/station_discovery.py` centered on O'Hare
  - Distance rings: 0–50 mi, 50–100 mi, 100–150 mi, 150–250 mi
  - Compass sectors: N, NE, E, SE, S, SW, W, NW
  - Target: 40–55 surrounding stations with >= 80% TMAX completeness
- **Key stations to include (expected):**
  - Midway Airport (USW00014819) — 12 mi S, urban reference
  - Rockford (USW00094822) — 75 mi NW, continental interior
  - Milwaukee (USW00014839) — 80 mi N, lake-effect comparison
  - South Bend (USW00014848) — 80 mi E, lake-effect leeward
  - Peoria (USW00014842) — 140 mi SW, interior Illinois
  - Madison (USW00014837) — 120 mi NW, Wisconsin interior
  - Indianapolis (USW00093819) — 165 mi SE, Ohio Valley
  - Moline/Quad Cities (USW00014923) — 160 mi W
- **Lake Michigan stations:** Critical for capturing onshore/offshore gradient
- **ASOS mapping:** Build GHCN-to-ICAO map for all selected stations
- **Deliverable:** `config_chicago.py` with full station metadata, rings, sectors

### 1.3 Data Collection
- **GHCN-Daily:** Download .dly files for all Chicago-area stations (1985–2024)
- **ASOS/AWOS:** Hourly observations for operational stations (1998–2024)
- **IGRA soundings:** Davenport (DVN, USM00074455) or Lincoln (ILX) — 12Z runs
- **NWP data:** GFS/NAM grids centered on Chicago (41.97, -87.91)
- **Kalshi market data:** KXHIGHCHI historical pre-settlement and settled prices
- **NWS forecasts:** LOT (Chicago) point forecast for benchmark
- **Deliverable:** `data/raw/chicago/` populated, completeness validated

### 1.4 Feature Engineering
- Reuse `src/data_preprocessing.py` and `src/operational_features.py` with Chicago config
- **City-specific features:**
  - Lake Michigan onshore/offshore wind composite (critical for Chicago)
  - Lake-vs-inland station temperature gradient
  - Arctic outbreak regime indicator (850mb temperature, wind direction)
  - Seasonal amplitude features (Chicago has larger annual swing than NYC)
- **Chronological splits:** 70/15/15 on Chicago data
- **Deliverable:** Processed feature matrices in `data/processed/chicago/`

### 1.5 Baseline Models
- Persistence (yesterday's Chicago TMAX)
- Climatological average (day-of-year)
- Ridge regression on surrounding station TMAX/TMIN
- NWS LOT forecast (benchmark)
- **Deliverable:** Baseline MAE and Brier scores for Chicago

### 1.6 Core Model Training
- **Phase A: Flat feedforward model**
  - Port `src/model.py` with Chicago inputs
  - Heteroscedastic Gaussian output (mu, sigma)
  - Train on Chicago feature set, evaluate OOS
- **Phase B: WGA model**
  - Port `src/wind_gated_attention.py` with Chicago station structure
  - Wind-direction gating is especially important for Chicago (lake effect)
  - CRPS-trained distributional model
- **Phase C: Synthesis variants**
  - Port E17-style contract Brier MLP
  - Port U7-style regime-conditional synthesis
  - Calibrate with isotonic + Platt + regime-aware layers
- **Deliverable:** Trained models in `models/chicago/`, benchmark comparison

### 1.7 Calibration + Bucketization
- Fit calibration on Chicago calibration set (not training set)
- Convert mu/sigma to KXHIGHCHI bucket probabilities
- Validate probability mass sums to 1.0 per day
- Reliability/ECE evaluation on holdout
- **Deliverable:** Calibration artifacts, reliability curves

### 1.8 Backtest + Trading Simulation
- Run EV-gated backtest against KXHIGHCHI historical market data
- Include fees, spread/slippage, conservative fill assumptions
- Report: Brier vs market, EV, realized P&L, drawdowns, seasonal slices
- **Deliverable:** `results/prediction_market_benchmark/chicago/`

---

## Part 2: Philadelphia Model (KXHIGHPHL)

### 2.1 Contract Specification Research
- **Task:** Confirm KXHIGHPHL contract details on Kalshi
  - Same deliverable format as Chicago
- **Deliverable:** `docs/philadelphia_contract_spec.md`

### 2.2 Station Network Design
- **Target station:** Philadelphia International Airport (USW00013739)
  - Lat: 39.8733, Lon: -75.2269
  - Elevation: 10 ft
  - Note: Already in NYC's expanded station network (Ring 2, 91 mi SW)
- **Surrounding station search:**
  - Centered on PHL; expect significant overlap with NYC's network
  - Distance rings: 0–50 mi, 50–100 mi, 100–150 mi, 150–250 mi
  - Target: 35–50 surrounding stations
- **Key stations to include (expected):**
  - NE Philadelphia (USW00094732) — 7 mi NE
  - Trenton-Mercer (USW00014792) — 30 mi NE
  - Wilmington (USW00013781) — 25 mi SW
  - Atlantic City (USW00093730) — 55 mi SE
  - Allentown (USW00014737) — 50 mi NW
  - Reading (USW00014712) — 55 mi W
  - Dover AFB (USW00013707) — 70 mi S
  - Baltimore (USW00093721) — 90 mi SW
  - Newark/NYC stations — 80-100 mi NE (shared with NYC network)
- **Advantage:** Many stations already downloaded for NYC — data reuse opportunity
- **Deliverable:** `config_philadelphia.py` with full station metadata

### 2.3 Data Collection
- **GHCN-Daily:** Reuse NYC-area .dly files where stations overlap; download additional southern/western stations
- **ASOS/AWOS:** Hourly observations for PHL-centered stations
- **IGRA soundings:** Sterling VA (IAD, USM00072403) or share NYC's Upton (OKX)
- **NWP data:** GFS/NAM grids centered on PHL (39.87, -75.23)
- **Kalshi market data:** KXHIGHPHL historical prices
- **NWS forecasts:** PHI (Mt Holly) point forecast for benchmark
- **Deliverable:** `data/raw/philadelphia/` populated

### 2.4 Feature Engineering
- Reuse NYC preprocessing pipeline with Philadelphia config
- **City-specific features:**
  - Delaware Valley urban heat island gradient
  - Coastal/marine influence from Atlantic (summer sea breeze)
  - NYC-PHL temperature correlation as feature (cross-city signal)
  - Appalachian ridgetop vs valley station gradient
- **Deliverable:** Processed feature matrices in `data/processed/philadelphia/`

### 2.5 Baseline Models
- Same baseline suite as Chicago
- Additionally: NYC model as a transfer-learning baseline (PHL is in NYC's network)
- **Deliverable:** Baseline metrics for Philadelphia

### 2.6 Core Model Training
- Same three-phase approach as Chicago (flat, WGA, synthesis)
- **Philadelphia-specific consideration:** Given strong NYC correlation, evaluate:
  - Transfer learning from NYC model (fine-tune on PHL data)
  - Cross-city ensemble (NYC prediction as feature for PHL model)
  - Shared-backbone architecture with city-specific heads
- **Deliverable:** Trained models in `models/philadelphia/`

### 2.7 Calibration + Bucketization
- Same approach as Chicago, adapted to KXHIGHPHL bucket definitions
- **Deliverable:** Calibration artifacts, reliability curves

### 2.8 Backtest + Trading Simulation
- Same approach as Chicago against KXHIGHPHL market data
- **Deliverable:** `results/prediction_market_benchmark/philadelphia/`

---

## Part 3: Operational Dashboard

### 3.1 Requirements
The dashboard provides a unified real-time view across all three cities for informed manual trading decisions.

**Core displays:**
1. **Current model predictions** — Today's mu/sigma, bucket probabilities for each city
2. **Market comparison** — Model prob vs Kalshi market-implied prob per bucket, highlighted edge
3. **EV signals** — Cost-adjusted EV per bucket, flagged tradable opportunities
4. **Historical performance** — Rolling Brier, calibration reliability, P&L curves
5. **Data health** — Station completeness, feature availability, last-update timestamps
6. **Kill-switch status** — Green/yellow/red indicators for each city's operational health

### 3.2 Architecture
- **Backend:** Python (Flask or FastAPI)
  - Reuse existing `src/` modules for predictions and market data
  - Scheduled jobs for data refresh (aligned to operational cutoff)
  - REST API endpoints for dashboard data
- **Frontend:** Lightweight web UI
  - Option A: Streamlit (fastest to build, single-file deployment)
  - Option B: React + Chart.js (more polished, more work)
  - Recommendation: Start with Streamlit, upgrade later if needed
- **Data refresh cycle:**
  - Market data: every 5 minutes during trading hours
  - Model predictions: once daily after cutoff computation
  - Historical metrics: daily rollup

### 3.3 Dashboard Pages

#### Page 1: Trading Overview (default)
- Three-city summary cards (NYC, CHI, PHL)
- Each card shows: today's predicted high, model confidence, top EV bucket, market vs model delta
- Color-coded by trading signal strength

#### Page 2: City Detail View (per city)
- Full bucket probability bar chart (model vs market)
- EV waterfall chart (gross edge, fees, slippage, net EV)
- Reliability diagram (recent 30-day window)
- Historical Brier score trend line
- Recent trade log (if paper trading)

#### Page 3: Cross-City Correlation
- NYC vs CHI vs PHL model predictions scatter
- Cross-city regime indicator (are all cities in same weather pattern?)
- Correlated exposure warning (avoid over-betting same weather event)

#### Page 4: Operations Health
- Data pipeline status per city (last successful run, missing stations, staleness)
- Model version and calibration window info
- Kill-switch status and recent alerts
- Feature drift indicators

### 3.4 Implementation Plan
1. Create `src/dashboard/` module directory
2. Build data aggregation layer (`src/dashboard/data_service.py`)
   - Wraps existing prediction, market, and evaluation modules
   - Provides unified JSON API for dashboard consumption
3. Build Streamlit app (`src/dashboard/app.py`)
   - Multi-page layout with sidebar navigation
   - Auto-refresh on configurable interval
4. Add Kalshi real-time market polling (`src/dashboard/market_poller.py`)
   - Fetch current order book for all three cities
   - Compute market-implied probabilities
   - Cache with configurable TTL
5. Add operational health checks (`src/dashboard/health_checks.py`)
   - Validate data freshness, model outputs, calibration drift
6. Deploy configuration
   - Local development: `streamlit run src/dashboard/app.py`
   - Production: Docker container with scheduled refresh jobs

### 3.5 Dashboard Deliverables
- `src/dashboard/app.py` — Main Streamlit application
- `src/dashboard/data_service.py` — Data aggregation layer
- `src/dashboard/market_poller.py` — Real-time market data
- `src/dashboard/health_checks.py` — Operational monitoring
- `src/dashboard/charts.py` — Visualization components
- `tests/test_dashboard.py` — Dashboard unit tests

---

## Part 4: Shared Infrastructure Work

### 4.1 Multi-City Config Framework
- Refactor config loading to support city-specific config files
- Pattern: `config.py` (shared defaults) + `config_{city}.py` (city overrides)
- Add city registry with contract ticker, target station, timezone, bucket definitions

### 4.2 Generalized Station Discovery
- Parameterize `src/station_discovery.py` to accept any target lat/lon
- Output standardized config format matching `config_expanded.py` schema

### 4.3 City-Agnostic Training Pipeline
- Parameterize training scripts to accept city config as argument
- Shared model architectures with city-specific input dimensions
- City-specific model checkpoints in `models/{city}/`

### 4.4 Unified Benchmark Framework
- Extend benchmark runners to accept city parameter
- Shared reporting format across cities
- Cross-city comparison summaries

### 4.5 Test Coverage
- Extend existing test suite for multi-city support
- Add integration tests for each city's end-to-end pipeline
- Add dashboard-specific tests

---

## Part 5: Implementation Phases

### Phase 1: Foundation (Infrastructure + Philadelphia)
Philadelphia is the easier city to start with because:
- Many stations overlap with NYC's existing network (data already downloaded)
- PHL is already in NYC's station list — strong prior signal
- Mid-Atlantic climate is similar to NYC — model transfer likely effective

**Tasks:**
1. Confirm KXHIGHPHL and KXHIGHCHI contract specifications on Kalshi
2. Build multi-city config framework
3. Generalize station discovery for any target
4. Design Philadelphia station network
5. Download additional PHL-area data (non-overlapping stations)
6. Build PHL feature pipeline and baselines
7. Train flat model + WGA for Philadelphia
8. Run synthesis and calibration sweep
9. Backtest against KXHIGHPHL market
10. Evaluate promotion readiness

### Phase 2: Chicago Expansion
Chicago is more complex due to:
- No station overlap with NYC — all new data download
- Lake Michigan creates unique microclimate dynamics
- Higher temperature variance — different sigma modeling needed

**Tasks:**
1. Design Chicago station network (full discovery run)
2. Download all Chicago-area GHCN/ASOS/NWP/IGRA data
3. Build CHI feature pipeline with lake-effect features
4. Train flat model + WGA for Chicago
5. Run synthesis and calibration sweep
6. Backtest against KXHIGHCHI market
7. Evaluate promotion readiness

### Phase 3: Dashboard
Build after at least one new city model is producing predictions.

**Tasks:**
1. Design dashboard data service API
2. Build Streamlit multi-page app with city overview
3. Add real-time Kalshi market polling
4. Build city detail views with model vs market comparison
5. Add operations health monitoring
6. Add cross-city correlation view
7. Deploy and test in paper-trading mode

### Phase 4: Cross-City Optimization
Once all three cities are operational:
1. Evaluate cross-city ensemble features
2. Test shared-backbone architectures
3. Implement correlated-exposure risk management
4. Run unified promotion gate across all cities
5. Begin cautious live trading on highest-confidence signals

---

## Part 6: Success Criteria

### Per-City Model Quality
- OOS Brier score beats NWS baseline for the target city
- Calibration reliability within 5% of nominal across all probability bins
- Seasonal stress tests show no catastrophic failure modes
- Positive EV in paper-trading simulation after fees/slippage

### Dashboard
- Displays current predictions, market data, and EV signals for all three cities
- Auto-refreshes market data during trading hours
- Operational health checks catch and display data/model issues
- Load time < 3 seconds for all pages

### System-Level
- Shared infrastructure supports adding new cities with minimal code changes
- Full test coverage for multi-city pipelines
- Audit trail for all predictions, market reads, and trading signals
- Kill-switch operational for each city independently

---

## Part 7: Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| KXHIGHCHI/PHL contract specs differ from KXHIGHNY | High | Research contract specs first; parameterize bucket logic |
| Chicago lake-effect creates non-Gaussian tails | Medium | Use mixture density or wider sigma modeling; evaluate quantile networks |
| Philadelphia too correlated with NYC | Low | Validate independence; cross-city features may help both models |
| Insufficient Kalshi market data for backtest | High | Use NWS as benchmark proxy until market data accumulates |
| Dashboard latency with 3 cities | Low | Cache predictions; only poll market data on configurable interval |
| Station data gaps for new cities | Medium | Validate completeness before model training; build fallback features |

---

## Appendix: Kalshi Temperature Contract Reference

### Known contract families
- **KXHIGHNY** — NYC daily high temperature (Central Park)
- **KXHIGHCHI** — Chicago daily high temperature (expected: O'Hare)
- **KXHIGHPHL** — Philadelphia daily high temperature (expected: PHL International)

### Typical bucket structure
Contracts typically offer 5F-wide buckets spanning the plausible temperature range for the season. Exact thresholds must be confirmed per contract per day, as Kalshi may adjust ranges seasonally.

### Settlement
Contracts settle based on the official daily maximum temperature recorded at the designated weather station, as reported by the National Weather Service.
