# Current State and File Directory

> **Last Updated:** 2026-02-19

---

## Project Overview

Multi-city daily maximum temperature (Tmax) probabilistic forecasting system for Kalshi weather prediction market contracts. The system produces calibrated probability distributions over temperature buckets, compares them against market-implied probabilities, and executes trades when expected value is positive after costs.

---

## City Pipeline Status

| City | Ticker | Station | Brier | Market Brier | Edge | Backtest P&L | Promotion | Status |
|------|--------|---------|-------|--------------|------|-------------|-----------|--------|
| **NYC** | KXHIGHNY | USW00094728 (Central Park) | 0.1137 (U7) | 0.1271 | +0.0133 | — | — | Fully Operational |
| **Chicago** | KXHIGHCHI | USW00094846 (O'Hare) | 0.1091 (U7_ext) | 0.1253 | +0.0162 | +$2,406 (241%) | 10/10 PASS | Backtest Complete |
| **Philadelphia** | KXHIGHPHL | USW00013739 (PHL Intl) | 0.1060 (U9) | 0.1099 | +0.0039 | +$340 (34%) | 10/10 PASS | Backtest Complete |
| **Atlanta** | KXHIGHTATL | USW00013874 (Hartsfield) | — | — | — | — | 11/11 PASS | Pipeline Complete |
| **Austin** | KXHIGHAUS | USW00013904 (Bergstrom) | — | — | — | — | 8/13 FAIL | Pipeline Complete |

---

## Architecture Layers

| Layer | Purpose | Key Modules |
|-------|---------|-------------|
| 1 - Ingestion | Raw data collection (GHCN, ASOS, NWP, Soundings) | `data_collection.py`, `asos_collection.py`, `nwp_collection.py`, `soundings_collection.py` |
| 2 - Features | Time-safe feature engineering with lagged inputs | `data_preprocessing.py`, `operational_features.py`, `asos_preprocessing.py`, `nwp_preprocessing.py`, `soundings_preprocessing.py` |
| 3 - Forecasting | Distributional models (E-series, WGA, Unified) | `model.py`, `advanced_model.py`, `wind_gated_attention.py`, `synthesis_model.py`, `extended_models.py` |
| 4 - Calibration | Post-hoc calibration + bucket probability conversion | `calibration.py`, `contract_brier.py`, `evaluate.py` |
| 5 - Trading | EV-gated execution with risk management | `trading.py`, `kalshi_client.py`, `kalshi_backtester.py`, `live_trading.py` |

---

## File Directory

### Root Configuration

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Master system prompt: rules, architecture contract, city anchors, operational discipline |
| `AGENTS.md` | Technical agent role definition and operational constraints |
| `config.py` | NYC base configuration (station list, parameters) |
| `config_expanded.py` | NYC expanded station network (~50 stations, 4 rings, sectors) |
| `config_chicago.py` | Chicago station network (~55 stations, lake-effect sectors) |
| `config_philadelphia.py` | Philadelphia station network (~50 stations, 4 rings) |
| `config_atlanta.py` | Atlanta station network (~50 stations, Piedmont/mountain sectors) |
| `config_austin.py` | Austin station network (~50 stations) |
| `nyc_temp_prediction_project_plan.md` | Master project plan with multi-city status |
| `requirements.txt` | Python dependencies |
| `.claude/rules/MEMORY.md` | Project memory: status, metrics, priorities, lessons learned |

### `src/` — Core Source Modules

**Layer 1: Ingestion**

| File | Purpose |
|------|---------|
| `city_config.py` | Multi-city registry: CityConfig dataclass, bucket generation, station/ticker mappings |
| `data_collection.py` | NOAA GHCN-Daily .dly parser and CSV conversion for all cities |
| `asos_collection.py` | IEM ASOS hourly data downloader (temperature, wind, dewpoint, pressure) |
| `nwp_collection.py` | GFS/GEFS downloader via Herbie library |
| `soundings_collection.py` | IGRA upper-air sounding downloader via Siphon |
| `station_registry.py` | NYC station querying: by count, radius, ring classification |
| `station_discovery.py` | GHCN station inventory search with haversine distance and sector classification |
| `data_integrity.py` | Data quality validation gates (no synthetic data, schema checks) |
| `operational_data.py` | City-aware ASOS/NWP/IGRA configuration wrapper |

**Layer 2: Feature Engineering**

| File | Purpose |
|------|---------|
| `data_preprocessing.py` | Multi-city preprocessing: station merge, QC, chronological splits, scaling |
| `operational_features.py` | NYC advanced features: wind-conditioned composites, ASOS/IGRA integration |
| `asos_preprocessing.py` | ASOS daily aggregation (TMAX/TMIN, dewpoint, wind, SLP) |
| `nwp_preprocessing.py` | NWP GRIB parsing, wind derivation, bias computation |
| `soundings_preprocessing.py` | IGRA sounding parsing, stability indices, lapse rates |
| `wga_data_pipeline.py` | Converts flat CSVs to 3D tensors for WindGatedAttentionModel |

**Layer 3: Forecasting**

| File | Purpose |
|------|---------|
| `model.py` | TempPredictorV1: feedforward heteroscedastic NN baseline |
| `advanced_model.py` | FeatureAttentionNet, MOSCorrectionNet, RegimeConditionalNet, EnsembleStacker |
| `wind_gated_attention.py` | WindGatedAttentionModel: direction-aware station aggregation |
| `synthesis_model.py` | SynthesisModel: meta-learner combining station + NWP forecasts |
| `extended_models.py` | E6-E22 model variants: contract Brier MLP, synthesis stackers, neural synthesis |
| `baselines.py` | Persistence, Climatology, LinearRegression, Ridge baselines |
| `crps_loss.py` | GaussianCRPSLoss, EnergyCRPSLoss, PinballLoss, CombinedCRPSMAELoss |
| `train.py` | PyTorch training loop with early stopping for TempPredictorV1 |
| `train_phase1.py` | Training loop for structured-input attention models |
| `model_checkpoint.py` | Checkpoint persistence utilities (PyTorch + sklearn) |

**Layer 4: Calibration & Evaluation**

| File | Purpose |
|------|---------|
| `calibration.py` | Post-hoc calibration suite: isotonic, Platt+isotonic, regime-conditional, PIT analysis |
| `contract_brier.py` | Kalshi contract-row Brier scoring (primary evaluation metric) |
| `evaluate.py` | MAE, RMSE, R², bias, seasonal breakdown, diagnostic plots |

**Layer 5: Trading & Execution**

| File | Purpose |
|------|---------|
| `trading.py` | TradingStrategy, BacktestEngine, Kelly sizing, EV computation, strategy grid search |
| `kalshi_client.py` | Kalshi public API client: market data, orderbook, probability parsing |
| `kalshi_backtester.py` | KalshiMarketSimulator, BacktestAnalyzer, CalibrationAnalyzer |
| `market_proxy.py` | Persistence-climatology blend market proxy for backtesting |
| `mos_market_proxy.py` | MOS-based market proxy using NWS day-ahead forecasts |
| `enhanced_market_proxy.py` | Sophisticated lag/rolling/Ridge market proxy with LOYO residual sigma |
| `live_trading.py` | Multi-city live trading harness with kill switches and audit logging |

**Dashboard (Planning)**

| File | Purpose |
|------|---------|
| `dashboard/dashboard_data.py` | DashboardData class for multi-city status aggregation |

### `scripts/` — Pipeline Runners

**Per-City Pipeline (6-step pattern)**

Each city follows: data_collection → preprocessing → benchmark → synthesis_calibration → backtest → promotion_evaluation.

| Step | CHI | PHL | ATL | AUS |
|------|-----|-----|-----|-----|
| 1. Data | `run_chi_data_collection.py` | `run_phl_data_collection.py` | `run_atl_data_collection.py` | `run_aus_data_collection.py` |
| 2. Preprocess | `run_chi_preprocessing.py` | `run_phl_preprocessing.py` | `run_atl_preprocessing.py` | `run_aus_preprocessing.py` |
| 3. Benchmark | `run_chi_benchmark.py` | `run_phl_benchmark.py` | `run_atl_benchmark.py` | `run_aus_benchmark.py` |
| 4. Calibrate | `run_chi_synthesis_calibration.py` | `run_phl_synthesis_calibration.py` | `run_atl_synthesis_calibration.py` | `run_aus_synthesis_calibration.py` |
| 5. Backtest | `run_chi_backtest.py` | `run_phl_backtest.py` | `run_atl_backtest.py` | `run_aus_backtest.py` |
| 6. Promote | `run_chi_promotion_evaluation.py` | `run_phl_promotion_evaluation.py` | `run_atl_promotion_evaluation.py` | `run_aus_promotion_evaluation.py` |

**NYC Canonical Benchmarks**

| Script | Models |
|--------|--------|
| `run_e0_e8_best_model_benchmark.py` | E0–E22 core models |
| `run_wga_v2_benchmark.py` | E38–E42 wind-gated attention |
| `run_unified_outperformance_benchmark.py` | U0–U9 unified synthesis |

**Multi-City Benchmarks**

| Script | Purpose |
|--------|---------|
| `run_city_nws_kalshi_template_benchmark.py` | NWS/Kalshi template for any city |
| `run_chi_phl_unified_benchmark.py` | CHI+PHL unified benchmark |
| `run_chi_advanced_benchmark.py` | CHI advanced model variants |
| `run_phl_advanced_benchmark.py` | PHL advanced model variants |
| `run_cross_city_benchmark_comparison.py` | Cross-city summary comparison |

**Kalshi Data & Market**

| Script | Purpose |
|--------|---------|
| `fetch_kalshi_presettlement.py` | NYC pre-settlement fetch |
| `fetch_kalshi_presettlement_multi.py` | Multi-city pre-settlement fetch |
| `fetch_kalshi_markets.py` | Real Kalshi settled markets |
| `fetch_kalshi_multi_city.py` | Multi-city Kalshi settlements |

**Trading & Backtesting**

| Script | Purpose |
|--------|---------|
| `run_real_kalshi_backtest.py` | Real Kalshi backtest (CHI/PHL) |
| `run_unified_trading_backtest.py` | Unified model trading backtest |
| `run_trading_strategy_sweep.py` | Multi-city strategy grid search |
| `run_promotion_evaluation_v2.py` | Multi-city promotion gates |

**Data Collection Utilities**

| Script | Purpose |
|--------|---------|
| `download_iem_mos.py` | NYC MOS download |
| `download_iem_mos_kord.py` | CHI MOS download |
| `download_iem_mos_kphl.py` | PHL MOS download |
| `build_nws_benchmark.py` | NWS distribution benchmark |
| `run_real_data_benchmark.py` | Real NWS MOS benchmark |

### `tests/` — Active Test Suite

| File | Tests | Module Covered |
|------|-------|----------------|
| `test_city_config.py` | Core | Multi-city configuration registry |
| `test_data_collection.py` | Core | GHCN-Daily parsing |
| `test_data_preprocessing.py` | Core | Time-safe preprocessing, splits |
| `test_operational_features.py` | 61 | Feature engineering |
| `test_model.py` | Core | TempPredictorV1 NN |
| `test_calibration.py` | 77 | Calibration suite |
| `test_contract_brier.py` | Core | Contract-row Brier scoring |
| `test_evaluate.py` | 55 | Evaluation metrics |
| `test_synthesis_model.py` | 71 | Synthesis meta-models |
| `test_wind_gated_attention.py` | 52 | WGA model + CRPS loss |
| `test_trading.py` | 101 | Trading strategy + backtest |
| `test_kalshi_backtester.py` | 57 | Market simulation + analysis |
| `test_kalshi_client.py` | 98 | Kalshi API client |
| `test_mos_market_proxy.py` | 78 | MOS market proxy |
| `test_station_registry.py` | 69 | Station discovery |
| `test_baselines.py` | Core | Baseline models |
| `test_train.py` | 54 | PyTorch training loop |
| `test_asos_collection.py` | Core | ASOS data collection |
| `test_asos_preprocessing.py` | Core | ASOS preprocessing |
| `test_nwp_collection.py` | Core | NWP data collection |
| `test_nwp_preprocessing.py` | 57 | NWP preprocessing |
| `test_soundings_collection.py` | Core | Sounding data |
| `test_soundings_preprocessing.py` | 47 | Sounding preprocessing |
| `test_chi_pipeline.py` | 27 | Chicago end-to-end pipeline |
| `test_phl_pipeline.py` | Core | Philadelphia pipeline |
| `test_atl_pipeline.py` | Core | Atlanta pipeline |
| `test_chi_phl_tasks.py` | Core | Checkpoint persistence |

### `data/` — Data Artifacts

| Directory | Contents |
|-----------|----------|
| `data/chicago/raw/` | 45 station CSV/DLY files |
| `data/chicago/processed/` | features_train/val/test.csv, target_train/val/test.csv, scaler.pkl |
| `data/chicago/mos/` | GFS/NAM/combined MOS for KORD |
| `data/philadelphia/raw/` | 43 station CSV/DLY files |
| `data/philadelphia/processed/` | features_train/val/test.csv, target_train/val/test.csv, scaler.pkl |
| `data/philadelphia/mos/` | GFS/NAM/combined MOS for KPHL |
| `data/atlanta/raw/` | 49 station CSV/DLY files |
| `data/atlanta/processed/` | features_train/val/test.csv, target_train/val/test.csv, scaler.pkl |
| `data/austin/raw/` | 56 station CSV/DLY files |
| `data/austin/processed/` | features_train/val/test.csv, target_train/val/test.csv, scaler.pkl |
| `data/airport_mos/` | 6 airport MOS CSV files for NYC area |
| Root-level CSVs | Kalshi presettlement/settlement, model predictions, historical weather (NYC legacy) |

### `results/` — Benchmark & Backtest Results

| Directory | Key Artifacts |
|-----------|---------------|
| `results/chicago/` | Unified benchmark (U0-U9), promotion report (10/10 PASS), real Kalshi backtest (+$2,406) |
| `results/philadelphia/` | Unified benchmark, promotion report (10/10 PASS), real Kalshi backtest (+$340) |
| `results/atlanta/` | Benchmark, promotion report (11/11 PASS), backtest |
| `results/austin/` | Benchmark, promotion report (8/13 FAIL), backtest |
| `results/cross_city_comparison/` | `best_models_summary.json` |
| Legacy dirs | `results/phase1_*`, `results/advanced_models/`, etc. (historical NYC experiments) |

### `ARCHIVE/` — Legacy Code & Docs

All superseded scripts, tests, documentation, and reports are organized in timestamped subdirectories. See `ARCHIVE/README.md` for full inventory.

---

## Chronological Splits

| Partition | Date Range |
|-----------|------------|
| Train | 2000-01-01 to 2021-12-31 |
| Calibration/Validation | 2022-01-01 to 2023-12-31 |
| Test | 2024-01-01 to 2025-12-31 |

---

## Known Issues and Audit Findings

1. **Settlement-Price Leakage (CRITICAL):** CHI/PHL Unified models U2-U5 used settlement-time market_prob as a feature, causing near-zero Brier scores via target leakage. E-series models are unaffected. Documented in `ARCHIVE/legacy_root_docs/AUDIT_model_cheating_investigation.md`.

2. **Brier Metric Scale Inconsistency:** NYC uses binary contract-row Brier; CHI/PHL historically reported multiclass bucket-day Brier. Values are not directly comparable without normalization. Documented in `ARCHIVE/legacy_root_docs/AUDIT_cross_city_brier_integrity.md`.

3. **Winter (DJF) Weakness:** All cities show degraded performance in winter due to cold-season volatility, storm tracks, and regime shifts.

4. **Simulated vs Real Kalshi Divergence:** Simulated market proxy is significantly harder to beat than real Kalshi markets. Both metrics should be reported.
