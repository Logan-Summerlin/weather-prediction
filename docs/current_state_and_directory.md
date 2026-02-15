# Current State of the Codebase + Directory Guide

**Last updated:** 2026-02-15

## Current system state
The repository is a **multi-city contract-aligned probabilistic forecasting and trading-research pipeline** for Kalshi temperature bucket contracts. NYC (KXHIGHNY) is fully operational with model lineage through **E42** and unified synthesis through **U9**. Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) expansion is in planning.

### Active benchmark families
1. **E-series core benchmark (E0–E22)**
   - Runner: `scripts/run_e0_e8_best_model_benchmark.py`
   - Summary: `results/prediction_market_benchmark/e0_e8_best_model_base/e0_e22_benchmark_summary.csv`
2. **WGA V2 benchmark extensions (E38–E42)**
   - Runner: `scripts/run_wga_v2_benchmark.py`
   - Summary: `results/prediction_market_benchmark/wga_v2_model/benchmark_summary.csv`
3. **Unified cross-model synthesis (U0–U9)**
   - Runner: `scripts/run_unified_outperformance_benchmark.py`
   - Summary: `results/prediction_market_benchmark/unified_outperformance/benchmark_summary.csv`

### Headline benchmark position
- Best overall Brier: **U7_regime_conditional (0.1137)**.
- Strong cluster: **E40_lag2 (0.1138), U6_platt (0.1141), E17_contract_brier (0.1141), E42_dual_attention (0.1150)**.

## Directory map

```text
weather-prediction/
├── src/                              # Core modules
│   ├── data_collection.py            # GHCN-Daily download
│   ├── data_preprocessing.py         # GHCN cleaning, merging, splits
│   ├── asos_collection.py            # ASOS/AWOS hourly download
│   ├── asos_preprocessing.py         # ASOS hourly → daily
│   ├── nwp_collection.py             # GFS/NAM grid download
│   ├── nwp_preprocessing.py          # NWP grid → station features
│   ├── soundings_collection.py       # IGRA sounding download
│   ├── soundings_preprocessing.py    # IGRA → stability features
│   ├── operational_features.py       # Time-safe composite features
│   ├── station_registry.py           # Station metadata lookups
│   ├── station_discovery.py          # GHCN station inventory search
│   ├── model.py                      # Flat feedforward NN
│   ├── wind_gated_attention.py       # WGA model
│   ├── synthesis_model.py            # Cross-model synthesis stacker
│   ├── train.py                      # Training loop + early stopping
│   ├── train_phase1.py               # WGA training utilities
│   ├── baselines.py                  # Persistence, climatology, ridge
│   ├── crps_loss.py                  # CRPS loss function
│   ├── calibration.py                # Isotonic, Platt, regime calibration
│   ├── evaluate.py                   # Brier, CRPS, reliability metrics
│   ├── market_proxy.py               # Market-implied probability extraction
│   ├── mos_market_proxy.py           # MOS-based market proxy
│   ├── enhanced_market_proxy.py      # Enhanced market features
│   ├── trading.py                    # EV, Kelly sizing, risk limits
│   ├── kalshi_client.py              # Kalshi API client
│   └── kalshi_backtester.py          # Historical backtest simulation
├── scripts/                          # Active runners
│   ├── run_e0_e8_best_model_benchmark.py    # E0–E22
│   ├── run_wga_v2_benchmark.py              # E38–E42
│   ├── run_unified_outperformance_benchmark.py  # U0–U9
│   ├── run_extended_val_benchmark.py        # Extended val split
│   ├── train_wga_v2.py                      # WGA V2 training
│   ├── train_wga_mdn.py                     # WGA-MDN training
│   ├── download_all_dly.py                  # Bulk GHCN download
│   ├── download_real_ghcn.py                # Targeted GHCN download
│   ├── download_iem_mos.py                  # IEM MOS download
│   ├── fetch_kalshi_markets.py              # Kalshi market fetch
│   ├── fetch_kalshi_presettlement.py        # Pre-settlement fetch
│   ├── build_nws_benchmark.py               # NWS benchmark builder
│   ├── build_extended_mos.py                # Extended MOS builder
│   ├── generate_max_training_predictions.py # Max-train predictions
│   ├── test_model_vs_benchmarks.py          # Model comparison
│   └── (MOS analysis/validation scripts)    # Various MOS utilities
├── tests/                            # Unit tests for all src modules
├── data/                             # Input datasets + predictions
├── results/                          # Benchmark outputs + diagnostics
├── reports/                          # Strategy + analysis reports
├── docs/                             # Current operational docs
├── models/                           # Saved model weights + scalers
├── ARCHIVE/                          # Legacy code (not active)
│   ├── legacy_experiments/           # Phase-1 exploration scripts
│   ├── legacy_runners/               # Early backtest runners
│   ├── legacy_root_runners/          # Root-level phase-1/2/3 runners
│   ├── legacy_scripts/               # Superseded benchmark scripts
│   └── legacy_docs/                  # Early research documents
├── config.py                         # Core 14-station NYC config
├── config_expanded.py                # Full 52-station config + metadata
├── CLAUDE.md                         # Project manager system prompt
├── AGENTS.md                         # Analyst agent system prompt
├── prediction_market_expansion.md    # Multi-city expansion plan
├── nyc_temp_prediction_project_plan.md  # Operational project plan
└── .claude/rules/MEMORY.md           # Active project memory
```

## Active module responsibilities

### Ingestion
- `src/data_collection.py` — GHCN-Daily bulk .dly download and parsing
- `src/asos_collection.py` — ASOS/AWOS hourly observations from IEM
- `src/nwp_collection.py` — GFS/NAM numerical weather prediction grids
- `src/soundings_collection.py` — IGRA upper-air sounding download

### Preprocessing + Features
- `src/data_preprocessing.py` — GHCN cleaning, merging, chronological train/val/test splits
- `src/asos_preprocessing.py` — ASOS hourly aggregation to daily features
- `src/nwp_preprocessing.py` — NWP grid interpolation to station-level features
- `src/soundings_preprocessing.py` — IGRA data to stability/lapse-rate features
- `src/operational_features.py` — Time-safe composite feature construction
- `src/station_registry.py` — Station metadata, sector classification, distance lookups
- `src/station_discovery.py` — GHCN inventory search for station network design

### Modeling
- `src/model.py` — Flat feedforward NN with heteroscedastic Gaussian output
- `src/wind_gated_attention.py` — Wind-Gated Attention (WGA) model
- `src/synthesis_model.py` — Cross-model synthesis stacker
- `src/train.py` — General training loop with early stopping and LR scheduling
- `src/train_phase1.py` — WGA-specific structured tensor training utilities
- `src/baselines.py` — Persistence, climatological, ridge regression baselines
- `src/crps_loss.py` — CRPS loss for distributional model training

### Calibration + Evaluation
- `src/calibration.py` — Isotonic, Platt, and regime-stratified calibration
- `src/evaluate.py` — Brier score, CRPS, reliability curves, ECE metrics
- `src/market_proxy.py` — Market-implied probability extraction from Kalshi
- `src/mos_market_proxy.py` — MOS-based market proxy construction
- `src/enhanced_market_proxy.py` — Enhanced market-state feature engineering

### Trading + Execution
- `src/trading.py` — EV computation, Kelly sizing, risk limits, halt logic
- `src/kalshi_client.py` — Kalshi REST API client
- `src/kalshi_backtester.py` — Historical backtesting with fees/spread simulation

## What was archived (2026-02-15)

### `ARCHIVE/legacy_root_runners/`
Root-level runners from early project phases: `run_baselines.py`, `run_nn.py`, `run_phase0.py`, `run_kalshi_backtest.py`, `run_kalshi_oos_validation.py`, `run_kalshi_real_oos.py`, `run_collect_all_stations.py`.

### `ARCHIVE/legacy_scripts/`
Superseded intermediate benchmark scripts: `run_e0_e1_e2_benchmark.py`, `run_best_model_lineage_top2_benchmark.py`, `run_top3_adjusted_benchmarks.py`, `run_wga_benchmark.py`, `run_gfs_residual_no_nam_benchmark.py`, `architecture_sweep.py`, `probabilistic_ensemble_experiments_v2.py`.

### `ARCHIVE/legacy_docs/`
Early research documents: `Temperature_Forecasting_ML_LLM_Report.md`, `weather_patterns_and_prediction_research.md`.

### Previously archived
- `ARCHIVE/legacy_experiments/` — Phase-1 exploration scripts (7 files).
- `ARCHIVE/legacy_runners/` — `run_kalshi_real_backtest.py`.
