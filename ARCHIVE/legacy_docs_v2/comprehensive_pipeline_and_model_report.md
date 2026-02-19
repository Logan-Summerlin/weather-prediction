# Comprehensive Pipeline & Model Report: NYC, CHI, PHL

> **Date:** 2026-02-17 | **Best Model:** U7_regime_conditional (Brier 0.1137)

---

## 1. System Overview

This project produces calibrated daily probability distributions for daily max temperature (TMAX) across three cities, converts them to Kalshi contract bucket probabilities, and trades when expected value is positive after costs.

| City | Ticker | Target Station | Station Network | Buckets | Floor/Ceiling | Status |
|------|--------|---------------|-----------------|---------|---------------|--------|
| **NYC** | KXHIGHNY | Central Park (USW00094728) | 52 stations, 4 rings | 57 (2°F) | 0°F / 110°F | **Fully operational** |
| **Chicago** | KXHIGHCHI | O'Hare (USW00094846) | 55 stations, 4 rings | 62 (2°F) | -10°F / 110°F | Pipeline code complete |
| **Philadelphia** | KXHIGHPHL | PHL Intl (USW00013739) | 50 stations, 4 rings | 57 (2°F) | 0°F / 110°F | Pipeline code complete |

---

## 2. Kalshi Contract Buckets & Probability Conversion

### 2.1 Bucket Definitions

All bucket definitions use **2°F resolution** matching actual Kalshi contracts. Buckets are generated programmatically via `_make_2f_bucket_grid()` in `src/city_config.py`.

**NYC & Philadelphia (57 buckets, 0–110°F):**
- Below 0°F (sentinel: -999 to 0)
- Interior: 0–2, 2–4, 4–6, ... 108–110 (55 buckets at 2°F each)
- Above 110°F (sentinel: 110 to 999)

**Chicago (62 buckets, -10–110°F):**
- Below -10°F (sentinel: -999 to -10)
- Interior: -10–(-8), -8–(-6), ... 108–110 (60 buckets at 2°F each)
- Above 110°F (sentinel: 110 to 999)

Chicago's lower floor (-10°F vs 0°F) accommodates significantly colder winter temperatures.

### 2.2 Gaussian-to-Bucket Probability Conversion

Models output a calibrated Gaussian prediction **(mu, sigma)** per forecast day. The CDF-to-bucket conversion (`src/calibration.py`, `cdf_to_kalshi_buckets()`) computes:

```
P(bucket) = CDF(upper + 0.5) - CDF(lower - 0.5)
```

Where CDF = Phi((x - mu) / sigma) is the standard normal CDF. The +/-0.5 offsets account for integer rounding of observed temperatures.

For the three contract types:
- **"below" contracts:** P(TMAX < threshold_high) = CDF(threshold_high)
- **"above" contracts:** P(TMAX > threshold_low) = 1 - CDF(threshold_low)
- **"between" contracts:** P(low <= TMAX <= high) = CDF(high) - CDF(low)

All probabilities are clipped to [0.0001, 0.9999] to avoid numerical edge effects.

### 2.3 Contract-Level Brier MLP

The top-performing models (E17, E40, U5, U7) bypass the raw Gaussian-to-CDF path and instead train an **MLP classifier** directly on contract-level (date, bucket) outcome pairs to predict P(contract outcome = 1), optimized against Brier score.

**U7 (best model) uses 36 input features per contract row:**

| Category | Features | Count |
|----------|----------|-------|
| Base probabilities | wga_prob, flat_prob, nws_prob, presettlement_prob | 4 |
| Pairwise differences | 3-way differences among model/nws/market probs | 6 |
| Market state | spread, sigma_norm, depth, stale_norm | 4 |
| WGA bucket geometry | quantile, width, distance_sigma, direction, neighbor_sum | 6 |
| Flat bucket geometry | quantile, width, distance_sigma, direction, neighbor_sum | 6 |
| Cross-model | disagreement, sigma_ratio, interaction terms | 4 |
| Regime features | seasonal sin/cos, WGA variance regime, forecast drift, sigma_norm, model disagreement | 6 |

**Training protocol:**
- 60/20/20 chronological split (train/validation/calibration)
- MLP architectures searched: (32,), (64,32), (128,64), (128,64,32)
- Hyperparameter grid: alpha in {3e-3, 5e-3, 8e-3, 1e-2}, lr in {1e-3, 8e-4, 6e-4, 5e-4}
- Selection criterion: minimize val_brier + 0.15 * val_ece
- Post-hoc isotonic calibration on held-out calibration set

### 2.4 Per-Day Renormalization

After calibration/synthesis, bucket probabilities are renormalized per date to ensure they sum to 1.0:

```python
for each date d:
    probs[d] = probs[d] / sum(probs[d])
```

This enforces logical consistency: Kalshi bucket outcomes for a given day are mutually exclusive and exhaustive.

---

## 3. Data Pipelines

### 3.1 Data Sources (Shared Across All Cities)

| Source | Type | Coverage | Use |
|--------|------|----------|-----|
| **NOAA GHCN-Daily** | Historical archive | 1985-2024 | Training features (TMAX, TMIN, PRCP, SNOW, AWND) |
| **IEM ASOS/AWOS** | Operational hourly | 1998-present | Live features (temp, dewpoint, wind, pressure, clouds) |
| **IGRA Soundings** | Upper-air | 2000-present | 850mb/500mb temp, wind, stability, lapse rate |
| **GFS/GEFS (Herbie)** | NWP forecasts | 2000-present | Optional day-ahead forecasts |
| **Kalshi API** | Market data | Live | Contract prices for EV comparison |

### 3.2 NYC Pipeline (KXHIGHNY) — Fully Operational

**Station Network:** 52 stations across 4 distance rings from Central Park:
- **Ring 1 (0-50mi):** 12 stations — LaGuardia, Teterboro, Newark, JFK, Caldwell, Westchester, Farmingdale, Somerset, Aeroflex-Andover, Sussex, Islip, Danbury
- **Ring 2 (50-100mi):** 21 stations — Bridgeport, Trenton, Poughkeepsie, McGuire, New Haven, Allentown, Philadelphia, Atlantic City, etc.
- **Ring 3 (100-150mi):** 12 stations — Hartford, Groton, Reading, Montauk, Pittsfield, Albany, Dover AFB, etc.
- **Ring 4 (150-250mi):** 7 stations — Bennington, Orange, Selinsgrove, Glens Falls, Ocean City, Boston, Syracuse

**Meteorological Sectors:**
- **WNW** — Cold advection from W/NW (Appalachian)
- **SW** — Warm advection from S/SW (Gulf track)
- **Coastal** — Atlantic moderation (E/SE)
- **NearField** — All Ring 1 stations (urban/local)
- **NE** — New England influence (N/NE)

**Unique NYC Features:**
- Wind-conditioned composites: upwind/crosswind/downwind temperature, advection rate
- IGRA sounding station: Upton/Brookhaven (USM00072501)
- ASOS coverage: 48 of 52 stations have operational ASOS data
- Most mature pipeline — all E0-E42 and U0-U9 models trained and evaluated

### 3.3 Chicago Pipeline (KXHIGHCHI) — Code Complete

**Station Network:** 55 stations across 4 rings from O'Hare:
- **Ring 1 (0-50mi):** 9 stations — Midway, Palwaukee, Lansing, Gary, DuPage, Aurora, Waukegan, Joliet, Valparaiso
- **Ring 2 (50-100mi):** 12 stations — Kenosha, Michigan City, Kankakee, DeKalb, Burlington WI, Janesville, Rochelle, Rockford, Milwaukee, South Bend, Pontiac, Bloomington
- **Ring 3 (100-150mi):** 16 stations — Lafayette, Danville, Battle Creek, Madison, Champaign, Fort Wayne, Oshkosh, Kalamazoo, Muskegon, Peoria, Grand Rapids, La Crosse, etc.
- **Ring 4 (150-250mi):** 18 stations — Moline, Cedar Rapids, Indianapolis, Springfield, Waterloo, Mason City, Saginaw, Traverse City, etc.

**Meteorological Sectors (Chicago-specific):**
- **WNW** — Continental cold-air advection (Arctic outbreaks)
- **Lake** — Lake Michigan moderation (N/NE/E sectors)
- **SW** — Gulf warm advection (southerly flow)
- **NearField** — Urban O'Hare microclimate (9 nearest)
- **NE_Lake** — Critical lake-shore dynamics (5 stations: Kenosha, Michigan City, Waukegan, Milwaukee, Muskegon)

**Pipeline Scripts (8 total):**
1. `run_chi_data_collection.py` — GHCN download for 55 stations
2. `run_chi_preprocessing.py` — Feature matrix creation (lag, scale, split)
3. `run_chi_benchmark.py` — Baselines (persistence, climatology, ridge, heteroscedastic NN)
4. `run_chi_advanced_benchmark.py` — Advanced models (FeatureAttention, MOSCorrection, RegimeConditional, CalibratedEnsemble)
5. `run_chi_synthesis_calibration.py` — Synthesis MLP + calibration sweep
6. `run_chi_backtest.py` — EV-gated trading simulation
7. `run_chi_promotion_evaluation.py` — Promotion gate evaluation
8. `run_chi_phl_unified_benchmark.py` — Joint CHI+PHL comparison

**Tests:** 27 passing in `tests/test_chi_pipeline.py` (config loads, bucket indexing, script existence, imports)

### 3.4 Philadelphia Pipeline (KXHIGHPHL) — Code Complete

**Station Network:** 50 stations across 4 rings from PHL International:
- **Ring 1 (0-50mi):** 7 stations — NE Philadelphia, Wilmington, McGuire AFB, Trenton-Mercer, Millville, Atlantic City Intl, Lakehurst
- **Ring 2 (50-100mi):** 13 stations — Reading, Dover AFB, Atlantic City Marina, Allentown, Aberdeen, Newark, Georgetown DE, Harrisburg, Annapolis, Central Park, BWI, Ft Meade
- **Ring 3 (100-150mi):** 14 stations — Wilkes-Barre, Salisbury, Andrews AFB, Reagan National, Frederick, Patuxent River, Islip, Williamsport, Wallops, Dulles, Quantico, Martinsburg, etc.
- **Ring 4 (150-250mi):** 16 stations — Binghamton, Altoona, Front Royal, Elmira, Hartford, Langley, Norfolk, Albany, DuBois, Bradford, Syracuse, Morgantown, etc.

**Meteorological Sectors:**
- **WNW** — Appalachian cold-air advection (W/NW, 13 stations)
- **SW** — Chesapeake/Gulf warm advection (S/SW, 21 stations)
- **Coastal** — Atlantic moderation (E/SE, 4 stations)
- **NearField** — Local effects (Ring 1, 7 stations)
- **NE** — NYC corridor correlation (N/NE, 12 stations)

**Pipeline Scripts (8 total):**
1. `run_phl_data_collection.py` — GHCN download for 51 stations
2. `run_phl_preprocessing.py` — Feature matrix creation
3. `run_phl_benchmark.py` — Baseline models
4. `run_phl_advanced_benchmark.py` — Advanced architectures
5. `run_phl_synthesis_calibration.py` — Synthesis + calibration sweep
6. `run_phl_backtest.py` — EV-gated trading backtest
7. `run_phl_promotion_evaluation.py` — Promotion gate evaluation
8. `run_phl_nws_kalshi_benchmark.py` — Model vs NWS vs Kalshi comparison

### 3.5 Shared Preprocessing Pipeline (All Cities)

All cities use the same 10-step pipeline parameterized by city config:

1. **Load** station CSVs from `data/{city}/raw/`
2. **Merge** into wide-format DataFrame (1 row/date, columns = station x variable)
3. **QC filter** — Drop stations with <80% TMAX completeness (target always kept)
4. **Lag features** — Surrounding stations shifted +1 day (respects 6 AM ET cutoff)
5. **Cyclical date encoding** — sin(2pi x DOY/365.25), cos(2pi x DOY/365.25)
6. **Missing data** — Forward-fill 3 days max, then training-mean imputation
7. **Chronological split** — 70% train / 15% val / 15% test (no shuffling)
8. **StandardScaler** — Fit on training data only, applied to val/test
9. **Save** — features_*.csv, target_*.csv, scaler.pkl, col_means.pkl
10. **Audit** — preprocessing_report.txt with completeness stats and split dates

---

## 4. Model Architectures

### 4.1 TempPredictorV1 — Flat Feedforward NN (`src/model.py`)

- **Architecture:** Input -> [64, 32] hidden (ReLU + Dropout 0.1) -> linear output
- **Output:** Point prediction (mu) or Gaussian (mu, sigma)
- **Use:** E0-E22 base model family
- **Parameters:** ~2,500-4,000

### 4.2 WindGatedAttentionModel (`src/wind_gated_attention.py`)

- **Per-station encoder:** Shared-weight MLP -> 32-dim embeddings
- **Attention:** Scaled dot-product with **wind bias** — learnable alpha x cos(wind_dir - bearing_i) upweights stations aligned with prevailing wind
- **Missing-station masking:** Attention logits set to negative infinity before softmax
- **Output head:** Gaussian (mu, log_sigma clamped [-10, 5])
- **Use:** E38-E42 (WGA V2) family

### 4.3 Synthesis Model — Meta-Learner (`src/synthesis_model.py`)

- **Input features (15):** model mu/sigma, NWP features, ensemble spread, station-NWP gap, seasonal encodings
- **Architecture:** [128, 64, 32] hidden with BatchNorm + ReLU + Dropout 0.15
- **Loss:** Combined CRPS+MAE (70%/30%) or Gaussian NLL
- **LR schedule:** ReduceLROnPlateau (patience 5, factor 0.5)
- **Use:** U0-U9 unified family

### 4.4 Advanced Architectures (`src/advanced_model.py`)

- **FeatureAttentionNet:** Dynamic feature importance via learned attention weights
- **MOSCorrectionNet:** Learns residuals (actual - MOS baseline)
- **RegimeConditionalNet:** Season x volatility regime-aware heteroscedastic predictor
- **CalibratedEnsemble:** Stacked meta-learner with isotonic + Platt calibration

### 4.5 Loss Functions (`src/crps_loss.py`)

| Loss | Formula | Use |
|------|---------|-----|
| **Gaussian CRPS** | Closed-form (Gneiting & Raftery 2007) | Synthesis training |
| **Combined CRPS+MAE** | 0.7 x CRPS + 0.3 x MAE | Default synthesis loss |
| **Gaussian NLL** | -log N(y; mu, sigma) | Heteroscedastic NN training |
| **Pinball** | Quantile regression loss | Quantile output mode |
| **Contract Brier** | Direct Brier on bucket outcomes | E17, E40, U5, U7 |

---

## 5. Model Families & Benchmark Results (NYC)

### 5.1 E-Series (E0-E22) — Flat NN Variants

| Model | Brier | Method |
|-------|-------|--------|
| E0 | ~0.122 | Raw flat NN predictions |
| E1-E8 | 0.115-0.120 | Various isotonic/offset calibrations |
| **E17** | **0.1141** | Contract-level Brier MLP + isotonic |
| E21 | 0.1144 | Platt recalibration on E17 |

### 5.2 WGA V2 Series (E38-E42) — Wind-Gated Attention

| Model | Brier | Method |
|-------|-------|--------|
| E38_full | ~0.116 | Full WGA V2 raw output |
| E39 variants | ~0.115 | WGA + logistic synthesis |
| **E40_lag2** | **0.1138** | Contract Brier MLP on WGA |
| E41 | ~0.115 | Ensemble blend (E40 + E0) |
| E42 | 0.1150 | Dual attention synthesis |

### 5.3 Unified Series (U0-U9) — Cross-Model Blends

| Model | Brier | Method |
|-------|-------|--------|
| U0 | ~0.122 | Flat NN raw bucket probs |
| U1 | ~0.116 | WGA V2 raw bucket probs |
| U4 | 0.1149 | Logistic synthesis stacker |
| U5 | 0.1154 | Contract-level Brier MLP (30 features) |
| U6 | 0.1141 | Platt recalibration on U5 |
| **U7** | **0.1137** | Regime-conditional contract MLP (36 features) |
| U8 | 0.1152 | 2023-only cal comparison |
| U9 | 0.1145 | Kitchen sink (max features) |

### 5.4 Top 10 Models (OOS 2025)

| Rank | Model | Brier | Family |
|-----:|-------|------:|--------|
| 1 | **U7_regime_conditional** | **0.1137** | Unified |
| 2 | E40_lag2_contract_brier | 0.1138 | WGA |
| 3 | U6_platt_on_u5 | 0.1141 | Unified |
| 4 | E17_contract_brier_synthesis | 0.1141 | E-series |
| 5 | E40_multihead_only | 0.1142 | WGA |
| 6 | E21_platt_recalibrated_e17 | 0.1144 | E-series |
| 7 | U9_kitchen_sink | 0.1145 | Unified |
| 8 | U4_extended_cal_synthesis | 0.1149 | Unified |
| 9 | E40_deep_only | 0.1149 | WGA |
| 10 | E42_dual_attention | 0.1150 | WGA |

### 5.5 Comparison vs Baselines

| Source | Brier | Delta vs U7 |
|--------|-------|-------------|
| **NWS baseline** | ~0.145 | U7 is ~21% better |
| **Kalshi pre-settlement (~24h)** | ~0.135 | U7 is ~16% better |
| **Climatology** | ~0.16+ | U7 is ~29% better |
| **Persistence** | ~0.18+ | U7 is ~37% better |
| **U7_regime_conditional** | **0.1137** | — |

---

## 6. Calibration Pipeline

### 6.1 Methods (`src/calibration.py`)

| Method | Approach | Use |
|--------|----------|-----|
| **Isotonic (global)** | Monotonic CDF mapping via PIT values | Default post-calibration |
| **Isotonic (seasonal)** | Separate models for DJF/MAM/JJA/SON | Regime-aware calibration |
| **Platt scaling** | Logistic sigmoid on log-odds | U6 recalibration |
| **PIT analysis** | KS test for CDF uniformity | Diagnostic |
| **Reliability diagrams** | Nominal vs observed coverage | Visual audit |

### 6.2 Calibration Quality Metrics

- **ECE (Expected Calibration Error):** <0.05 required for production
- **PIT uniformity:** KS p-value > 0.05
- **Coverage:** 50/80/90/95% intervals should match nominal levels

---

## 7. Trading & Backtesting

### 7.1 EV Computation (`src/trading.py`)

```
EV_yes = model_prob x (1 - fee_rate) - market_ask_price
EV_no  = (1 - model_prob) x (1 - fee_rate) - (1 - market_bid_price)
```

### 7.2 Position Sizing

| Method | Formula | Usage |
|--------|---------|-------|
| **Fixed** | constant $ per trade | Risk control baseline |
| **Proportional** | $ = portfolio x EV x scale | Simple scaling |
| **Full Kelly** | f = (p*b - q) / b | Maximize log utility |
| **Fractional Kelly** | f x (0.25 to 0.5) | Conservative sizing |
| **Capped Kelly** | min(Kelly, max_position) | Risk limits |

### 7.3 Backtest Parameters (All Cities)

| Parameter | Value |
|-----------|-------|
| EV threshold | 0.02 (2 cents) |
| Fee rate | 0.07 (7%) |
| Kelly fraction | 0.25 |
| Max contracts/bucket/day | 10 |
| Initial bankroll | $1,000 |

### 7.4 Kill Switch Triggers

- Data staleness > cutoff
- Schema mismatch (unexpected feature count)
- NaN flood > 10%
- Calibration failure
- ECE degradation > 0.15

---

## 8. Promotion Gates (Per City)

Each city must pass all gates before live deployment:

| Gate | Criterion | NYC | CHI | PHL |
|------|-----------|-----|-----|-----|
| OOS Brier | Absolute | <0.15 | <0.16 | <0.15 |
| Beats NWS | Relative | <0.13 | <0.14 | <0.13 |
| Seasonal max | Stress test | <0.20 | <0.22 | <0.20 |
| Min OOS days | Coverage | >=200 | >=200 | >=200 |
| ECE | Calibration | <0.05 | <0.05 | <0.05 |
| Paper P&L | Trading | >=$0 | >=$0 | >=$0 |
| Max drawdown | Risk | >=-30% | >=-30% | >=-30% |
| Data artifacts | Operations | All present | All present | All present |

---

## 9. Key Files Reference

| Domain | Key Files |
|--------|-----------|
| Bucket/city config | `src/city_config.py` |
| E0-E22 benchmark core | `scripts/run_e0_e8_best_model_benchmark.py` |
| WGA E38-E42 benchmark | `scripts/run_wga_v2_benchmark.py` |
| Unified U0-U9 benchmark | `scripts/run_unified_outperformance_benchmark.py` |
| Core modeling modules | `src/model.py`, `src/wind_gated_attention.py`, `src/synthesis_model.py` |
| Advanced models | `src/advanced_model.py` |
| Calibration + evaluation | `src/calibration.py`, `src/evaluate.py`, `src/kalshi_backtester.py` |
| Trading + market integration | `src/trading.py`, `src/kalshi_client.py` |
| Data + operational features | `src/data_collection.py`, `src/data_preprocessing.py`, `src/operational_features.py` |
| ASOS/NWP/Soundings | `src/asos_collection.py`, `src/nwp_collection.py`, `src/soundings_collection.py` |
| Station management | `src/station_registry.py`, `src/station_discovery.py`, `config_expanded.py` |
| Market proxies | `src/market_proxy.py`, `src/mos_market_proxy.py`, `src/enhanced_market_proxy.py` |
| Chicago pipeline | `config_chicago.py`, `scripts/run_chi_*.py`, `tests/test_chi_pipeline.py` |
| Philadelphia pipeline | `config_philadelphia.py`, `scripts/run_phl_*.py` |
| Loss functions | `src/crps_loss.py` |

---

## 10. Observations & Gaps

### Strengths

1. **NYC is mature and production-ready** — 30+ model variants benchmarked, top cluster within 0.04% Brier of each other, comprehensive calibration pipeline.
2. **City-agnostic architecture** — All `src/` modules are parameterized by city config; CHI and PHL pipelines are full clones with city-specific station networks and bucket definitions.
3. **Time-safety is enforced** — All features lagged +1 day, scaler fit on training only, chronological splits throughout.
4. **Wind-gated attention** captures physically meaningful advection patterns (upwind weighting).
5. **Uniform 2°F bucket resolution** across all cities simplifies the CDF-to-bucket and contract-level MLP code — only the floor/ceiling differs.

### Gaps & Risks

1. **CHI and PHL have no benchmark results yet** — Pipeline code is complete but no models have been trained or evaluated. All Brier scores in this report are NYC-only.
2. **No PHL test suite** — Chicago has 27 tests; Philadelphia has none. Risk of config bugs going undetected.
3. **Backtest uses simulated market data** — CHI and PHL backtests generate synthetic market prices with noise, not actual Kalshi order book data. Per our guardrails, we should be comparing against actual Kalshi prices ~24h before settlement.
4. **Kalshi pre-settlement benchmark is proxy-based** — The `enhanced_market_proxy.py` uses Ridge regression to approximate market prices, not actual historical Kalshi order book snapshots.
5. **Top model cluster is extremely tight** — U7 (0.1137) through E17 (0.1141) differ by only 0.04% Brier. Statistical significance of these differences has not been formally tested.
6. **Chicago has 5 more buckets than NYC/PHL** — 62 vs 57 buckets means more sparse outcome data per bucket, potentially harder to calibrate with limited training data.
