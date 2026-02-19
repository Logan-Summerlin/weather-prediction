# Model Families: Functions and Methods Reference

> **Last Updated:** 2026-02-19

---

## Overview

The system implements three hierarchical model families for daily Tmax probabilistic forecasting, each progressively more sophisticated:

1. **E-Series (E0–E22):** Core feedforward and advanced neural architectures
2. **WGA-Series (E38–E42):** Wind-gated attention with station-level spatial reasoning
3. **U-Series (U0–U9):** Unified synthesis models combining multiple base models

All models output distributional predictions (mu, sigma) that are converted to bucket probabilities via Gaussian CDF, calibrated, and scored using Contract Brier.

### Best Models by City

| City | Model | Contract Brier | Market Brier | Edge |
|------|-------|---------------|--------------|------|
| NYC | U7_regime_conditional | 0.1137 | 0.1271 | +0.0133 |
| Chicago | U7_extended | 0.1091 | 0.1253 | +0.0162 |
| Philadelphia | U9_kitchen_sink | 0.1060 | 0.1099 | +0.0039 |

---

## Part 1: Base Neural Network (`src/model.py`)

### TempPredictorV1
The foundational heteroscedastic feedforward neural network.

**Architecture:**
- Input: n_features (default 30: 28 station features + sin_day + cos_day)
- Hidden layers: [64, 32] (configurable)
- ReLU activation + Dropout (p=0.1) between hidden layers
- Single linear output head (mu prediction)

**Key Methods:**
| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward(x)` | `Tensor → Tensor` | Returns point prediction (batch, 1) |
| `create_model()` | `config → TempPredictorV1` | Factory with config defaults |
| `count_parameters()` | `→ int` | Trainable parameter count |
| `get_model_summary()` | `→ str` | Architecture string |

**Used As:** Baseline model in E0-E5 benchmarks; backbone for advanced variants.

---

## Part 2: Advanced Models (`src/advanced_model.py`)

Four core architectures for heteroscedastic probabilistic forecasting. All output (mu, sigma) pairs.

### 2.1 FeatureAttentionNet
Dynamic feature importance via learned attention gates.

**Architecture:**
- Context Encoder: 2-layer MLP + LayerNorm → encodes all features to understand weather regime
- Attention Gate: context → per-feature softmax importance scores (learnable temperature)
- Prediction Trunk: attended features → [256, 128, 64] with BatchNorm
- Separate mu and log_sigma output heads

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(x)` | Returns (mu, sigma, attention_weights) |
| `get_attention_weights(x)` | Extract interpretable feature importance |

**Innovation:** Attention weights are interpretable—show which stations matter most per forecast day.

### 2.2 MOSCorrectionNet
Residual correction on top of MOS/climatology baseline.

**Architecture:**
- Learns delta corrections: `TMAX = baseline_mu + NN_correction`
- Correction trunk: [128, 64, 32] → BatchNorm + ReLU + Dropout
- Separate delta_head and log_sigma_head

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(x, baseline_mu)` | Returns (baseline + delta, sigma) |

**Strategy:** Predicting residuals reduces target variance ~80%, improving optimization and generalization.

### 2.3 RegimeConditionalNet (Basis for U7)
Different uncertainty modeling per weather regime (season × volatility interaction).

**Architecture:**
- Main Trunk: (n_features + n_regime_features) → [256, 128, 64]
- Mu Head: standard prediction from trunk
- Sigma Trunk: separate MLP using (trunk_hidden + regime_features) → log_sigma
- Regime Features: 4 seasons + 3 volatility bins + 12 interactions = 16-19 features

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(x, regime_features)` | Returns (mu, regime-conditional sigma) |
| `compute_regime_features()` | Season onehot + volatility bins + interactions |

**Key Insight:** Winter forecasts have ~2-3× higher uncertainty than summer. Regime conditioning improves calibration significantly.

### 2.4 EnsembleStacker
Meta-learner combining multiple base models via Ridge regression.

**Architecture:**
- Feature matrix: [mu1, sigma1, mu2, sigma2, ..., mun, sigman] + optional regime features
- Ridge regression head → ensemble mu
- Sigma estimation: 50% avg(base sigmas) + 50% training residual sigma

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `fit(base_predictions, targets)` | Fit Ridge weights on calibration set |
| `predict(base_predictions)` | Returns (mu_ensemble, sigma_ensemble) |

---

## Part 3: Wind-Gated Attention (`src/wind_gated_attention.py`)

### WindGatedAttentionModel
Spatial-aware station aggregation using wind direction as attention bias.

**Architecture:**
- Station Feature Encoder: per-station MLP encoding local features
- Wind Gate: `attention_bias = α * cos(wind_dir - station_bearing)` (learnable α)
- Multi-head Attention: station embeddings with wind-biased attention scores
- Global Context: date/cyclical features concatenated after aggregation
- Output Heads: mu_head and log_sigma_head

**Input Format (from `wga_data_pipeline.py`):**
- `station_features`: (batch, n_stations, n_features) — 3D tensor
- `station_metadata`: (batch, n_stations, n_meta) — lat, lon, elevation, bearing
- `global_context`: (batch, n_global) — date, cyclical, wind direction
- `bearings`: (batch, n_stations) — compass bearing from target
- `wind_direction`: (batch, 1) — prevailing wind
- `station_mask`: (batch, n_stations) — binary mask for missing stations

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(station_features, metadata, global_ctx, bearings, wind_dir, mask)` | Returns (mu, sigma) |
| `get_attention_maps()` | Extract per-head attention weights for visualization |

**Key Innovation:** Wind direction physically biases which upwind stations receive more attention weight. On NW wind days, stations to the NW (cold advection corridor) are upweighted.

### E38–E42 Model Variants (WGA V2 Benchmark)

| Model | Key Variation |
|-------|---------------|
| E38 | Base WGA with CRPS loss |
| E39 | WGA + wind-gated ring attention |
| E40_lag2 | WGA + 2-day lag features (Brier 0.1138, near-best) |
| E41 | WGA + multi-head (4 heads) |
| E42 | WGA + residual connections + dropout refinement |

---

## Part 4: Extended Models (`src/extended_models.py`)

City-agnostic E6–E22 variants. Key implementations:

### E6–E8: Advanced NN Variants
- **E6:** Deeper architecture [128, 64, 32] + CRPS loss
- **E7:** E6 + batch normalization
- **E8:** E6 + dropout scheduling (0.3 → 0.1)

### E9–E16: Synthesis Stackers
- **E9:** Linear stacker on E0-E8 mu/sigma
- **E10:** Ridge stacker with L2 regularization
- **E11–E16:** Progressively complex synthesis with regime features

### E17: Contract Brier MLP (NYC Reference Model)
Direct optimization on contract-row Brier score. Takes 36 features:
- Base probabilities from 4 models
- Pairwise probability differences (6)
- Market state features (4)
- Bucket geometry features (12)
- Cross-model agreement features (4)
- Regime indicators (6)

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `fit(contract_features, outcomes)` | Train MLP on contract-level data |
| `predict_proba(contract_features)` | Returns calibrated bucket probabilities |

### E18–E22: Neural Synthesis with Attention
- **E18:** Attention-based synthesis of E0-E17 outputs
- **E19:** E18 + temporal context (day-of-year, trend)
- **E20:** E18 + regime-conditional combination weights
- **E21_platt_e17:** Platt scaling applied to E17 outputs (Brier 0.1144)
- **E22:** Full neural synthesis with residual connections

---

## Part 5: Synthesis Model (`src/synthesis_model.py`)

### SynthesisModel
Meta-learner fusing station model + NWP forecasts into combined distributional output.

**Architecture:**
- Station Branch: encodes station model (mu, sigma) pairs
- NWP Branch: encodes GFS/GEFS features (temperature, wind, moisture)
- Fusion Layer: concatenated branches → [128, 64]
- Output: (mu_synthesis, sigma_synthesis)

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `forward(station_features, nwp_features)` | Returns fused (mu, sigma) |
| `train_synthesis(train_loader, val_loader, epochs)` | Full training with early stopping |

---

## Part 6: Unified Series U0–U9 (`scripts/run_unified_outperformance_benchmark.py`)

The Unified series combines E-series and WGA outputs into calibrated ensemble predictions.

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| U0 | Simple average of top-5 E-series | Baseline ensemble |
| U1 | Weighted average (inverse Brier weights) | Skill-weighted |
| U2 | Contract-level Ridge regression | Direct contract optimization |
| U3 | U2 + isotonic calibration | Post-hoc calibration |
| U4 | Regime-conditional stacking | Season × volatility conditioning |
| U5 | U4 + Platt scaling | Two-stage calibration |
| U6_platt | Platt-calibrated weighted ensemble | Brier 0.1141 |
| **U7_regime_conditional** | **RegimeConditionalNet on ensemble** | **Best NYC: Brier 0.1137** |
| U8_cv_ensemble | Cross-validated ensemble | Best CHI cross-city: 0.1087 |
| **U9_kitchen_sink** | **All features + all calibration** | **Best PHL: Brier 0.1060** |

---

## Part 7: Calibration (`src/calibration.py`)

Mandatory post-processing before trading. Fit only on calibration partition (2022–2023).

### Calibration Methods

| Method | Implementation | Use Case |
|--------|----------------|----------|
| Isotonic Regression | `IsotonicRegression(out_of_bounds='clip')` | Default, nonparametric |
| Platt Scaling | `LogisticRegression` on logit(p) | Smooth parametric adjustment |
| Platt + Isotonic | Platt first, then isotonic | Two-stage refinement |
| Regime-Conditional | Separate calibrators per season × volatility | Seasonal adaptation |

### Diagnostic Methods

| Method | Purpose |
|--------|---------|
| `compute_pit_values()` | Probability Integral Transform for bias detection |
| `plot_reliability_diagram()` | Observed vs predicted frequency curves |
| `compute_ece()` | Expected Calibration Error |
| `compute_coverage()` | Prediction interval coverage (50%, 90%, 95%) |
| `compute_sharpness()` | Average prediction interval width |

---

## Part 8: Contract Brier Scoring (`src/contract_brier.py`)

Primary evaluation metric. Scores only actually-listed Kalshi contracts per date.

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_contract_brier(predictions, outcomes, contracts)` | Per-contract Brier score |
| `compute_contract_brier_aggregate(...)` | Weighted/unweighted mean across contracts |
| `compare_model_vs_market(model_probs, market_probs, outcomes)` | Head-to-head on same rows |
| `stratified_brier(predictions, outcomes, groups)` | Brier by season, bucket, regime |

### Brier Score Definitions

- **Contract Brier:** Average over actually-listed contract rows. Typical range: 0.10–0.19. This is the primary metric.
- **Bucket-Day Brier:** Average over ALL (day × bucket) pairs including trivial buckets. Typical range: 0.014–0.016. Dominated by easy predictions.

---

## Part 9: Bucketization (`src/city_config.py`)

Converting Gaussian (mu, sigma) to Kalshi bucket probabilities.

### Bucket Probability Computation

```
For bucket with edges [lo, hi]:
  - Between bucket: P = F(hi) - F(lo)     where F = Gaussian CDF
  - Below bucket:   P = F(hi)             (open lower tail, lo = -999)
  - Above bucket:   P = 1 - F(lo)         (open upper tail, hi = 999)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `cdf_to_kalshi_buckets(mu, sigma, edges)` | Full bucket probability vector |
| `get_bucket_index(tmax, edges)` | Map actual temp to realized bucket |
| `_make_2f_bucket_grid(floor, ceiling)` | Generate 2°F-width bucket edges |

### Bucket Grids by City

| City | Floor | Ceiling | Buckets | Open Tails |
|------|-------|---------|---------|------------|
| NYC | 0°F | 110°F | 57 | Below 0°F, Above 110°F |
| Chicago | -10°F | 110°F | 62 | Below -10°F, Above 110°F |
| Philadelphia | 0°F | 110°F | 57 | Below 0°F, Above 110°F |
| Atlanta | 0°F | 110°F | 57 | Below 0°F, Above 110°F |
| Austin | 0°F | 110°F | 57 | Below 0°F, Above 110°F |

---

## Part 10: Training Infrastructure

### `src/train.py` — Standard Training Loop

| Function | Purpose |
|----------|---------|
| `train_model(model, train_loader, val_loader, ...)` | Full training with early stopping, LR scheduling |
| `train_epoch(model, loader, optimizer, criterion)` | Single epoch forward/backward |
| `validate_epoch(model, loader, criterion)` | Single epoch validation |
| `save_training_history(history, path)` | Persist loss/metric curves |

### `src/train_phase1.py` — Attention Model Training

| Function | Purpose |
|----------|---------|
| `train_attention_model(model, train_loader, val_loader, ...)` | Training for WindGatedAttentionModel |
| `AttentionDataset` | Custom Dataset for 3D station tensor inputs |

### `src/crps_loss.py` — Loss Functions

| Class | Formula | Use |
|-------|---------|-----|
| `GaussianCRPSLoss` | Closed-form CRPS (Gneiting & Raftery 2007) | Primary distributional loss |
| `EnergyCRPSLoss` | Sample-based energy score | Alternative for non-Gaussian |
| `PinballLoss` | Quantile regression loss | Quantile forecasts |
| `CombinedCRPSMAELoss` | α·CRPS + (1-α)·MAE | Balanced training objective |

---

## Part 11: Trading Framework (`src/trading.py`)

### EV Computation

| Function | Purpose |
|----------|---------|
| `compute_ev_yes(model_prob, market_price, fee)` | EV for buying YES contract |
| `compute_ev_no(model_prob, market_price, fee)` | EV for buying NO contract |
| `compute_ev_best(model_prob, market_price, fee)` | Best of YES/NO |

### Position Sizing

| Function | Purpose |
|----------|---------|
| `kelly_fraction(model_prob, market_price)` | Full Kelly optimal fraction |
| `fractional_kelly(model_prob, market_price, fraction=0.25)` | Conservative Kelly |
| `capped_kelly(model_prob, market_price, cap=0.10)` | Kelly with max cap |

### TradingStrategy Class

Configurable parameters: `ev_threshold`, `sizing_method`, `max_position`, `max_daily_exposure`, `fee_rate`.

### BacktestEngine

| Method | Purpose |
|--------|---------|
| `run_backtest(predictions, market_data, outcomes)` | Day-by-day simulation |
| `generate_strategy_grid()` | 6,000+ parameter permutations |
| `run_comprehensive_backtest()` | Evaluate all strategies |

### BacktestResult Metrics

Sharpe ratio, ROI, win rate, max drawdown, profit factor, VaR, expected shortfall, monthly P&L breakdown, seasonal performance.
