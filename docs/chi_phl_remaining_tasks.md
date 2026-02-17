# CHI/PHL Remaining Tasks to Reach Full NYC Parity

**Created:** 2026-02-17
**Context:** CHI and PHL pipelines are ~90% complete. Core models (E0-E5, U0-U9) are trained, real-Kalshi backtests are running, and promotion evaluation infrastructure is in place. The items below represent the remaining gap vs NYC.

---

## 1. Wind-Gated Attention (WGA) Architecture for CHI/PHL

**Priority:** Medium (models already competitive without WGA)
**Effort:** Large

NYC uses a Wind-Gated Attention model (E38-E42 family) that dynamically weights surrounding stations based on wind direction. This architecture is defined in `src/wind_gated_attention.py` and requires 3D station×feature tensor input.

### Tasks:
- [ ] Create preprocessing pipeline that preserves per-station structure (currently flattened into wide feature vectors in `data/{city}/processed/features_*.csv`)
- [ ] Define station metadata tensors (bearing, distance, elevation, sector) for CHI and PHL networks
- [ ] Extract global context features (wind direction, SLP, date encoding) for CHI/PHL
- [ ] Train WGA model in Gaussian mode for CHI (55 stations) and PHL (51 stations)
- [ ] Add E38-E42 equivalent variants to CHI/PHL benchmark suite
- [ ] Integrate WGA predictions into U-series synthesis models (U10+ variants combining flat NN + WGA)

### Alternative (lighter-weight):
- [ ] Implement feature-group attention NN (E6-E8) that applies attention over feature groups within the existing flat feature framework, capturing the WGA concept without full 3D restructuring

### Notes:
- CHI U7 already achieves Brier 0.1091, PHL U9 achieves 0.1060 — both better than NYC's best (U7 at 0.1137). WGA may not improve scores further.
- Lake Michigan wind gating is especially relevant for CHI; Atlantic sea breeze gating relevant for PHL.

---

## 2. Extended E-Series Models (E6-E22)

**Priority:** Low (U-series already captures most of this value)
**Effort:** Medium

NYC has E6-E22 variants including contract-level Brier optimization, neural synthesis stackers, and attention variants. CHI/PHL only have E0-E5.

### Tasks:
- [ ] Port E6-E8 (advanced NN architectures with dropout/regularization sweeps)
- [ ] Port E9-E16 (synthesis stacker variants — Ridge, Lasso, ElasticNet on base model outputs)
- [ ] Port E17 (contract-level Brier-optimal MLP — key NYC model at 0.1141)
- [ ] Port E18-E22 (neural synthesis with attention, residual connections)

### Notes:
- Most of this value is already captured by U3 (contract MLP) and U7 (extended MLP) in the unified benchmark.
- Consider whether the marginal improvement justifies the complexity.

---

## 3. Model Checkpoint Persistence

**Priority:** Medium
**Effort:** Small

CHI/PHL models are trained inline within benchmark scripts and discarded after evaluation. NYC saves model checkpoints to `models/` for reuse.

### Tasks:
- [ ] Save best E3/E4/E5 PyTorch model state dicts to `models/{city}/`
- [ ] Save best Ridge model (E2) as pickle to `models/{city}/`
- [ ] Save scaler and column metadata alongside models
- [ ] Add model loading utility for inference without retraining
- [ ] Update `run_chi_phl_unified_benchmark.py` to optionally load pre-trained models

---

## 4. Live Trading Harness for CHI/PHL

**Priority:** High (required for production deployment)
**Effort:** Medium

NYC has live Kalshi API integration via `src/kalshi_client.py` and `src/trading.py`. CHI/PHL have no live trading capability.

### Tasks:
- [ ] Extend `src/trading.py` to accept city_code parameter and route to correct Kalshi ticker (KXHIGHCHI, KXHIGHPHL)
- [ ] Test Kalshi API connectivity for CHI/PHL contract tickers
- [ ] Build daily inference pipeline: data fetch → feature compute → model predict → calibrate → bucketize → EV gate → trade
- [ ] Add kill-switch per city (independent of NYC kill-switch)
- [ ] Set up paper-trading mode for CHI/PHL before going live

---

## 5. Operational Dashboard Integration

**Priority:** Low (Phase 3 per expansion plan)
**Effort:** Medium

### Tasks:
- [ ] Add CHI/PHL data sources to `src/dashboard/data_service.py`
- [ ] Add CHI/PHL cards to trading overview page
- [ ] Add city detail views for CHI and PHL
- [ ] Add cross-city correlation view (NYC vs CHI vs PHL)
- [ ] Add per-city operational health checks

---

## 6. ASOS/NWP/Sounding Data Integration

**Priority:** Medium (improves operational features)
**Effort:** Medium

NYC uses ASOS hourly observations, GFS/NAM NWP forecasts, and IGRA soundings as operational features. CHI/PHL configs reference these but the data may not be fully integrated.

### Tasks:
- [ ] Verify ASOS hourly data collection for CHI stations (KORD + network)
- [ ] Verify ASOS hourly data collection for PHL stations (KPHL + network)
- [ ] Set up GFS/NAM grid extraction centered on CHI (41.97, -87.91) and PHL (39.87, -75.23)
- [ ] Integrate IGRA soundings: Davenport (DVN) for CHI, Sterling (IAD) for PHL
- [ ] Add NWP-derived features to CHI/PHL preprocessing pipeline

---

## Current State Summary

| Dimension | NYC | Chicago | Philadelphia |
|-----------|-----|---------|--------------|
| E0-E5 base models | Done | Done | Done |
| E6-E22 advanced | Done | Not started | Not started |
| E38-E42 WGA | Done | Not started | Not started |
| U0-U9 synthesis | Done | Done | Done |
| Real Kalshi backtest | Done | Done | Done |
| Promotion evaluation | Done | Done | Done |
| Test coverage | 30+ tests | 27 tests | 29 tests |
| Model checkpoints | Saved | Not saved | Not saved |
| Live trading | Ready | Not ready | Not ready |
| Dashboard | Planning | Not started | Not started |
| Best Brier | 0.1137 (U7) | 0.1091 (U7) | 0.1060 (U9) |
| Beats market? | Yes | Yes (+0.0162) | Yes (+0.0039) |
