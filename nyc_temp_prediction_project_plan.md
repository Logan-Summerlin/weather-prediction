# NYC Temperature Prediction — Operational Project Plan

**Last updated:** 2026-02-15
**Scope:** Multi-city contract-aligned probabilistic forecasting, calibration, bucketization, EV-aware trading simulation for Kalshi temperature contracts.

## 1) Mission

Produce calibrated daily probability distributions for daily max temperature at contract resolution, convert to exact Kalshi bucket probabilities, compare against market-implied probabilities, and trade only when edge remains positive after costs. Currently operational for NYC (KXHIGHNY); expansion to Chicago and Philadelphia planned.

## 2) Current Implemented State

### 2a) NYC (KXHIGHNY) — Production

#### Contract and benchmark alignment
- Benchmarking aligned to KXHIGHNY contract buckets and pre-settlement/settled market references.
- Model lineage analyzed through three benchmark families:
  - **E0–E22** core synthesis (`scripts/run_e0_e8_best_model_benchmark.py`)
  - **E38–E42** WGA V2 extensions (`scripts/run_wga_v2_benchmark.py`)
  - **U0–U9** unified cross-model synthesis (`scripts/run_unified_outperformance_benchmark.py`)

#### Forecasting stack
- Base forecast streams: flat feedforward model + Wind-Gated Attention (WGA) model probability pipelines.
- Advanced variants: contract-level Brier-optimized MLP synthesis, Platt+isotonic recalibration, regime-conditional features.
- Unified variants (U-family) combine flat, WGA, NWS, and market-state features with gating diagnostics.

#### Top model variants (by overall Brier score)
| Rank | Variant | Brier | Family |
|---:|---|---:|---|
| 1 | U7_regime_conditional | 0.1137 | Unified |
| 2 | E40_lag2_only_contract_brier | 0.1138 | WGA V2 |
| 3 | U6_platt_on_u5 | 0.1141 | Unified |
| 4 | E17_contract_brier_synthesis | 0.1141 | E-core |
| 5 | E42_dual_attention_synthesis | 0.1150 | WGA V2 |

#### Calibration and diagnostics
- Multi-stage calibration: isotonic, Platt+isotonic, regime-stratified.
- Reliability/ECE and Brier decomposition in benchmark artifacts.
- Seasonal stress slices and OOS-focused diagnostics in unified reports.

#### Trading evaluation
- EV-gated simulation with threshold sweeps and paper-trading promotion checks.
- Fees/spread-aware assumptions integrated in benchmark outputs.

### 2b) Data infrastructure
- **Target station:** Central Park, NYC (USW00094728)
- **Surrounding stations:** 52 GHCN-Daily stations in 4 distance rings (0–250 mi), 8 compass sectors
- **Data window:** 1985–2024 (40 years GHCN); 1998–2024 (ASOS); 2000–2024 (NWP/IGRA)
- **Operational data:** ASOS hourly obs, IGRA soundings, GFS/NAM NWP grids, Kalshi market data
- **Station configs:** `config.py` (14 core stations), `config_expanded.py` (52 stations with full metadata)

## 3) Active Architecture

### Layer A — Ingestion
| Source | Module | Availability |
|---|---|---|
| GHCN-Daily archives | `src/data_collection.py` | Training-only (delayed QC) |
| ASOS/AWOS hourly | `src/asos_collection.py` | Operational (by cutoff) |
| IGRA soundings | `src/soundings_collection.py` | Operational (12Z available by morning) |
| GFS/NAM grids | `src/nwp_collection.py` | Operational (forecast cycles) |
| Kalshi markets | `src/kalshi_client.py` | Operational (live) |

### Layer B — Feature engineering
| Module | Role |
|---|---|
| `src/data_preprocessing.py` | GHCN cleaning, merging, feature splits |
| `src/asos_preprocessing.py` | ASOS hourly to daily aggregation |
| `src/nwp_preprocessing.py` | NWP grid to station-level features |
| `src/soundings_preprocessing.py` | IGRA to stability/lapse-rate features |
| `src/operational_features.py` | Time-safe composite feature builder |
| `src/station_registry.py` | Station metadata and sector lookups |

### Layer C — Modeling
| Module | Role |
|---|---|
| `src/model.py` | Flat feedforward NN (heteroscedastic Gaussian) |
| `src/wind_gated_attention.py` | Wind-Gated Attention model |
| `src/synthesis_model.py` | Cross-model synthesis stacker |
| `src/train.py` | Training loop with early stopping |
| `src/train_phase1.py` | WGA-specific training utilities |
| `src/baselines.py` | Persistence, climatology, ridge baselines |
| `src/crps_loss.py` | CRPS loss for distributional training |

### Layer D — Calibration + bucketization
| Module | Role |
|---|---|
| `src/calibration.py` | Isotonic, Platt, regime-stratified calibration |
| `src/evaluate.py` | Brier, CRPS, reliability metrics |
| `src/market_proxy.py` | Market-implied probability extraction |
| `src/mos_market_proxy.py` | MOS-based market proxy |
| `src/enhanced_market_proxy.py` | Enhanced market feature engineering |

### Layer E — Trading + risk
| Module | Role |
|---|---|
| `src/trading.py` | EV computation, Kelly sizing, risk limits |
| `src/kalshi_client.py` | Kalshi API integration |
| `src/kalshi_backtester.py` | Historical backtest simulation |

### Active benchmark runners (`scripts/`)
| Script | Scope |
|---|---|
| `run_e0_e8_best_model_benchmark.py` | E0–E22 core lineage |
| `run_wga_v2_benchmark.py` | E38–E42 WGA V2 variants |
| `run_unified_outperformance_benchmark.py` | U0–U9 unified synthesis |
| `run_extended_val_benchmark.py` | Extended validation split benchmark |
| `train_wga_v2.py` | WGA V2 model training |
| `train_wga_mdn.py` | WGA-MDN model training |

### Active utility scripts (`scripts/`)
| Script | Purpose |
|---|---|
| `download_all_dly.py` | Bulk GHCN .dly download |
| `download_real_ghcn.py` | Targeted GHCN download |
| `download_iem_mos.py` | IEM MOS data download |
| `fetch_kalshi_markets.py` | Kalshi market data fetch |
| `fetch_kalshi_presettlement.py` | Pre-settlement price fetch |
| `build_nws_benchmark.py` | NWS forecast benchmark builder |
| `build_extended_mos.py` | Extended MOS dataset builder |
| `generate_max_training_predictions.py` | Max-training prediction generator |
| `test_model_vs_benchmarks.py` | Model vs benchmark comparison |

## 4) High-Priority Gaps

1. Formalize hard-cutoff data availability manifest for every live feature.
2. Add automated kill-switch checks into daily orchestration path.
3. Increase execution realism for queue position/fill uncertainty in backtests.
4. Finalize production promotion rubric across E/WGA/U families.
5. Build multi-city expansion pipeline (Chicago, Philadelphia).
6. Build operational dashboard for real-time model vs market monitoring.

## 5) Promotion Gates (before live scaling)

### Forecast gate
- OOS Brier must consistently beat NWS baseline and remain stable across seasonal slices.

### Calibration gate
- Reliability/ECE and interval checks within configured tolerance bands.

### Trading gate
- Positive conservative paper-trading profile with acceptable drawdown behavior.

### Operations gate
- Complete daily run by cutoff with full audit artifacts and no critical validation failures.

## 6) Repository Structure

```
weather-prediction/
├── src/                    # Core modules (ingestion, features, modeling, calibration, trading)
├── scripts/                # Active benchmark/training/utility runners
├── tests/                  # Unit tests for all src modules
├── data/                   # Input datasets + generated prediction artifacts
├── results/                # Benchmark outputs and diagnostics
├── reports/                # Strategy and analysis reports
├── docs/                   # Operational documentation
├── models/                 # Saved model weights and scalers
├── ARCHIVE/                # Legacy code (5 subdirectories, not active)
│   ├── legacy_experiments/ # Phase-1 exploration scripts
│   ├── legacy_runners/     # Early backtest runners
│   ├── legacy_root_runners/# Root-level phase-1/2/3 runners
│   ├── legacy_scripts/     # Superseded benchmark scripts
│   └── legacy_docs/        # Early research documents
├── config.py               # Core 14-station configuration
├── config_expanded.py      # Full 52-station configuration with metadata
├── CLAUDE.md               # Project manager system prompt
├── AGENTS.md               # Analyst agent system prompt
└── .claude/rules/MEMORY.md # Active project memory
```

## 7) Tech Stack

- **Language:** Python 3
- **Deep learning:** PyTorch
- **Data handling:** pandas, NumPy
- **Classical ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Data sources:** NOAA GHCN-Daily, ASOS/AWOS (IEM), IGRA soundings, GFS/NAM NWP, Kalshi API
- **Market integration:** Kalshi REST API
