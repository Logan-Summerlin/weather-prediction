# Weather Prediction Market System — Operational Project Plan

**Last updated:** 2026-02-19
**Scope:** Multi-city contract-aligned probabilistic forecasting, calibration, bucketization, EV-aware trading simulation for Kalshi temperature contracts.

## 1) Mission

Produce calibrated daily probability distributions for daily max temperature at contract resolution, convert to exact Kalshi bucket probabilities, compare against market-implied probabilities, and trade only when edge remains positive after costs. NYC (KXHIGHNY) fully operational. Chicago and Philadelphia pipelines complete through backtesting with promotion evaluations done. Atlanta and Austin pipelines in development.

## 2) Current Implemented State

### 2a) NYC (KXHIGHNY) — Fully Operational

#### Contract and benchmark alignment
- Benchmarking aligned to KXHIGHNY contract buckets and pre-settlement/settled market references.
- Model lineage analyzed through three benchmark families:
  - **E0–E22** core synthesis (`scripts/run_e0_e8_best_model_benchmark.py`)
  - **E38–E42** WGA V2 extensions (`scripts/run_wga_v2_benchmark.py`)
  - **U0–U9** unified cross-model synthesis (`scripts/run_unified_outperformance_benchmark.py`)

#### Top model variants (by Contract Brier score)
| Rank | Variant | Brier | Family |
|---:|---|---:|---|
| 1 | U7_regime_conditional | 0.1137 | Unified |
| 2 | E40_lag2_only_contract_brier | 0.1138 | WGA V2 |
| 3 | U6_platt_on_u5 | 0.1141 | Unified |
| 4 | E17_contract_brier_synthesis | 0.1141 | E-core |
| 5 | E42_dual_attention_synthesis | 0.1150 | WGA V2 |

- **Kalshi market Brier:** 0.1271 | **Edge:** +0.0133

#### Calibration and diagnostics
- Multi-stage calibration: isotonic, Platt+isotonic, regime-stratified.
- Reliability/ECE and Brier decomposition in benchmark artifacts.
- Seasonal stress slices and OOS-focused diagnostics in unified reports.

### 2b) Chicago (KXHIGHCHI) — Backtest Complete, Promotion Ready

- **Target:** O'Hare International (USW00094846), 55 stations, 4 rings, lake-effect sectors
- **Best model (real Kalshi):** U7_extended (Contract Brier 0.1091)
- **Kalshi market Brier:** 0.1253 | **Edge:** +0.0162
- **Real Kalshi backtest:** +$2,406 (+241%), Sharpe 6.07, 71% win rate, 1144 trading days
- **Simulated market backtest:** +$48 (+4.8%), Sharpe 1.17, 64% win rate, 317 days
- **Promotion evaluation:** 10/10 gates PASS
- **Seasonal weakness:** DJF winter (simulated -$92, real Kalshi still +$564)
- **Cross-city best model:** U8_cv_ensemble (Contract Brier 0.1087)
- **Next steps:** Live paper trading validation, then production deployment

### 2c) Philadelphia (KXHIGHPHL) — Backtest Complete, Needs Calibration Work

- **Target:** PHL International (USW00013739), 50 stations, 4 rings
- **Best model (real Kalshi):** U9_kitchen_sink (Contract Brier 0.1060)
- **Kalshi market Brier:** 0.1099 | **Edge:** +0.0039 (narrow)
- **Real Kalshi backtest:** +$340 (+34%), Sharpe 2.76, 46% win rate, 451 trading days
- **Simulated market backtest:** -$22 (-2.2%), Sharpe -1.57, 63% win rate, 110 days
- **Promotion evaluation:** 10/10 gates PASS
- **Seasonal weakness:** DJF winter (simulated -$62, real Kalshi mixed)
- **Key issue:** Narrow Brier edge vs Kalshi; not yet profitable against simulated market
- **Next steps:** Improve winter calibration, widen Brier edge before live deployment

### 2d) Atlanta (KXHIGHTATL) — Pipeline Complete

- **Target:** Hartsfield-Jackson (USW00013874), ~50 stations
- **Status:** Full pipeline operational, promotion evaluation complete (11/11 gates PASS)
- **Next steps:** Backtest refinement, Kalshi data collection

### 2e) Austin (KXHIGHAUS) — Pipeline Complete, Needs Work

- **Target:** Austin-Bergstrom (USW00013904), ~56 stations
- **Status:** Full pipeline operational, promotion evaluation (8/13 gates FAIL)
- **Next steps:** Model improvements needed to pass remaining promotion gates

### 2f) Data Infrastructure

| City | Target Station | Stations | Data Window |
|------|---------------|----------|-------------|
| NYC | USW00094728 (Central Park) | 52 | 1985–2025 |
| Chicago | USW00094846 (O'Hare) | 55 | 2000–2025 |
| Philadelphia | USW00013739 (PHL Intl) | 50 | 2000–2025 |
| Atlanta | USW00013874 (Hartsfield) | 49 | 2000–2025 |
| Austin | USW00013904 (Bergstrom) | 56 | 2000–2025 |

**Operational data sources:** ASOS hourly obs, IGRA soundings, GFS/NAM NWP grids, Kalshi market data.

## 3) Active Architecture

### Layer 1 — Ingestion
| Source | Module | Availability |
|---|---|---|
| GHCN-Daily archives | `src/data_collection.py` | Training-only (delayed QC) |
| ASOS/AWOS hourly | `src/asos_collection.py` | Operational (by cutoff) |
| IGRA soundings | `src/soundings_collection.py` | Operational (12Z available by morning) |
| GFS/NAM grids | `src/nwp_collection.py` | Operational (forecast cycles) |
| Kalshi markets | `src/kalshi_client.py` | Operational (live) |

### Layer 2 — Feature Engineering
| Module | Role |
|---|---|
| `src/data_preprocessing.py` | GHCN cleaning, merging, feature splits |
| `src/asos_preprocessing.py` | ASOS hourly to daily aggregation |
| `src/nwp_preprocessing.py` | NWP grid to station-level features |
| `src/soundings_preprocessing.py` | IGRA to stability/lapse-rate features |
| `src/operational_features.py` | Time-safe composite feature builder |
| `src/station_registry.py` | Station metadata and sector lookups |
| `src/city_config.py` | Multi-city registry, bucket generation |

### Layer 3 — Modeling
| Module | Role |
|---|---|
| `src/model.py` | Flat feedforward NN (heteroscedastic Gaussian) |
| `src/advanced_model.py` | FeatureAttentionNet, MOSCorrectionNet, RegimeConditionalNet |
| `src/wind_gated_attention.py` | Wind-Gated Attention model |
| `src/synthesis_model.py` | Cross-model synthesis stacker |
| `src/extended_models.py` | E6–E22 city-agnostic variants |
| `src/train.py` | Training loop with early stopping |
| `src/train_phase1.py` | WGA-specific training utilities |
| `src/baselines.py` | Persistence, climatology, ridge baselines |
| `src/crps_loss.py` | CRPS loss for distributional training |

### Layer 4 — Calibration + Bucketization
| Module | Role |
|---|---|
| `src/calibration.py` | Isotonic, Platt, regime-stratified calibration |
| `src/contract_brier.py` | Contract-row Brier scoring (primary metric) |
| `src/evaluate.py` | Brier, CRPS, reliability metrics |
| `src/market_proxy.py` | Market-implied probability extraction |
| `src/mos_market_proxy.py` | MOS-based market proxy |
| `src/enhanced_market_proxy.py` | Enhanced market feature engineering |

### Layer 5 — Trading + Risk
| Module | Role |
|---|---|
| `src/trading.py` | EV computation, Kelly sizing, risk limits |
| `src/kalshi_client.py` | Kalshi API integration |
| `src/kalshi_backtester.py` | Historical backtest simulation |
| `src/live_trading.py` | Multi-city live trading harness |

### Per-City Pipeline Scripts
Each city follows a 6-step pipeline: data_collection → preprocessing → benchmark → synthesis_calibration → backtest → promotion_evaluation. Scripts are in `scripts/run_<city>_*.py`.

### NYC Canonical Benchmark Scripts
| Script | Scope |
|---|---|
| `run_e0_e8_best_model_benchmark.py` | E0–E22 core lineage |
| `run_wga_v2_benchmark.py` | E38–E42 WGA V2 variants |
| `run_unified_outperformance_benchmark.py` | U0–U9 unified synthesis |

### Multi-City Scripts
| Script | Purpose |
|---|---|
| `run_city_nws_kalshi_template_benchmark.py` | NWS/Kalshi template for any city |
| `run_chi_phl_unified_benchmark.py` | CHI+PHL unified benchmark |
| `fetch_kalshi_presettlement_multi.py` | Multi-city pre-settlement fetch |
| `run_cross_city_benchmark_comparison.py` | Cross-city comparison |
| `run_real_kalshi_backtest.py` | Real Kalshi backtest (CHI/PHL) |
| `run_promotion_evaluation_v2.py` | Multi-city promotion gates |

## 4) High-Priority Gaps and Next Steps

1. **Improve DJF winter calibration** across all cities — biggest source of losses.
2. **Validate CHI for live paper trading** — strongest expansion candidate (10/10 gates, +0.0162 edge).
3. **Improve PHL Brier edge** before promoting to live (currently only +0.0039).
4. Formalize hard-cutoff data availability manifest for every live feature.
5. Add automated kill-switch checks into daily orchestration path.
6. Increase execution realism for queue position/fill uncertainty in backtests.
7. Build operational dashboard for real-time model vs market monitoring.
8. Improve Austin model to pass remaining 5 promotion gates.

## 5) Known Audit Findings

1. **Settlement-Price Leakage (CRITICAL):** CHI/PHL Unified models U2-U5 used settlement-time `market_prob` as a feature, inflating Brier scores. E-series models unaffected. See `ARCHIVE/legacy_root_docs/AUDIT_model_cheating_investigation.md`.
2. **Brier Metric Scale Inconsistency:** NYC uses binary contract-row Brier; CHI/PHL historically reported multiclass bucket-day Brier. All benchmarks now standardized to Contract Brier. See `ARCHIVE/legacy_root_docs/AUDIT_cross_city_brier_integrity.md`.
3. **Real vs Simulated Kalshi Divergence:** Simulated market proxy is significantly harder to beat than real Kalshi. Both metrics should always be reported.

## 6) Promotion Gates (all must pass before live scaling)

| Gate | Requirement |
|------|-------------|
| Contract alignment | Verified against Kalshi specification |
| Time-safety audit | No future leakage in features |
| Calibration diagnostics | PIT, reliability, ECE acceptable |
| Contract Brier vs baselines | Must beat persistence, climatology |
| Contract Brier vs market | Competitive with Kalshi market baseline |
| Positive EV after costs | Net positive after fees, spread, slippage |
| Drawdown within limits | Max drawdown below threshold |
| Exposure within limits | Per-contract and per-day caps |
| Kill switch validated | All triggers tested |
| Reproducibility | Artifacts complete, pipeline deterministic |

## 7) Repository Structure

```
weather-prediction/
├── src/                    # Core modules (ingestion, features, modeling, calibration, trading)
├── scripts/                # Active benchmark/training/utility runners
├── tests/                  # Unit tests for all src modules
├── data/                   # Per-city raw + processed data artifacts
│   ├── chicago/            # CHI raw (45 stations) + processed
│   ├── philadelphia/       # PHL raw (43 stations) + processed
│   ├── atlanta/            # ATL raw (49 stations) + processed
│   └── austin/             # AUS raw (56 stations) + processed
├── results/                # Per-city benchmark outputs and backtest diagnostics
├── docs/                   # Consolidated documentation (3 files)
│   ├── 01_current_state_and_directory.md
│   ├── 02_model_families_and_methods.md
│   └── 03_principles_and_city_portability.md
├── ARCHIVE/                # Legacy code, docs, reports (10 subdirectories)
├── config.py               # NYC base station configuration
├── config_expanded.py      # NYC full 52-station configuration
├── config_chicago.py       # CHI 55-station configuration
├── config_philadelphia.py  # PHL 50-station configuration
├── config_atlanta.py       # ATL 50-station configuration
├── config_austin.py        # AUS 56-station configuration
├── CLAUDE.md               # Master system prompt
├── AGENTS.md               # Agent role definition
└── .claude/rules/MEMORY.md # Active project memory
```

## 8) Tech Stack

- **Language:** Python 3
- **Deep learning:** PyTorch
- **Data handling:** pandas, NumPy
- **Classical ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Data sources:** NOAA GHCN-Daily, ASOS/AWOS (IEM), IGRA soundings, GFS/NAM NWP, Kalshi API
- **Market integration:** Kalshi REST API

## 9) Key Lessons Learned

1. **Real Kalshi vs simulated market results diverge significantly.** Simulated market proxy is much harder to beat. Always report both.
2. **Winter (DJF) is the universal weak season** across all cities. Cold-season volatility and regime shifts degrade model skill.
3. **Chicago has strongest expansion signal** — large Brier edge (+0.0162), profitable across all seasons in real Kalshi backtest.
4. **Philadelphia edge is thin** (+0.0039) — needs winter calibration improvements before live deployment.
5. **Contract Brier (not bucket-day Brier) is the correct metric** for Kalshi settlement logic comparison.
6. **Promotion evaluation thresholds are city-specific** — CHI uses 0.16, PHL uses 0.14 Brier thresholds.
