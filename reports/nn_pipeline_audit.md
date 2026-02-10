# Neural Network Pipeline Audit Report

**Date:** 2026-02-10
**Scope:** End-to-end audit of the current NN pipeline — data, features, architecture, training, output
**Audited by:** Project Manager + Analyst subagent

---

## Executive Summary

The project contains three tiers of model architecture (simple NN, wind-gated attention, synthesis meta-learner), but **only the simplest tier is operational**. The synthesis model and wind-gated attention model are fully implemented but never invoked by any runner script. The actual prediction pipeline uses `TempPredictorV1` (a 2-layer feedforward net) alongside Ridge regression. Several design issues constrain current performance.

| Finding | Severity | Section |
|---------|----------|---------|
| Synthesis model + wind-gated attention are dead code (no runner) | CRITICAL | 1 |
| 11 of 15 synthesis features are phantom (NWP data never downloaded) | CRITICAL | 2 |
| Station ID labeling errors in config_expanded.py | HIGH | 3 |
| Actual NN uses TempPredictorV1 [128,64], not synthesis model | HIGH | 4 |
| Architecture overparameterized for available data | MEDIUM | 5 |
| Sigma is monthly-constant, not heteroscedastic | MEDIUM | 6 |
| Wind direction hardcoded to 0 in train_phase1.py | MEDIUM | 7 |
| No saved model checkpoints in repo | MEDIUM | 8 |

---

## 1. What the Pipeline ACTUALLY Does (vs. What's Documented)

### The operational prediction pipeline

The script that generates the predictions used in Kalshi backtesting is `scripts/generate_max_training_predictions.py`. It:

1. **Imports `config_expanded`** (52 surrounding stations)
2. **Downloads GHCN .dly files** for all stations, filters by >=80% completeness
3. **Builds TMAX-only lag-1 features** + NYC autoregressive lag-1 + sin/cos day encoding
4. **Trains two models** side-by-side:
   - `TempPredictorV1` [128, 64] with Huber loss, dropout=0.0
   - Ridge regression (alpha=1.0)
5. **Estimates monthly sigma** from validation residuals (12 values, one per month)
6. **Outputs CSV files**: `max_train_nn_predictions_{is,oos}.csv` with columns: date, model_mu, model_sigma, actual_tmax

### What is NOT operational

| Module | Lines of Code | Status |
|--------|---------------|--------|
| `src/synthesis_model.py` (SynthesisModel + SynthesisTrainer) | 1,498 | **Dead code** — no runner imports it |
| `src/wind_gated_attention.py` (WindGatedAttentionModel) | 359 | **Dead code** — no runner imports it |
| `src/train_phase1.py` (wind-gated training loop) | 982 | **Dead code** — no runner imports it |
| `src/crps_loss.py` (CRPS, pinball, combined losses) | 404 | **Dead code** — only imported by synthesis/wind-gated |
| `src/calibration.py` | 1,370 | **Dead code** — no runner imports it |

**Verification:** `grep -r 'SynthesisModel\|SynthesisTrainer\|synthesis_model' run_*.py scripts/*.py` returns zero matches. Same for WindGatedAttention and train_phase1.

---

## 2. Phantom Features: 11 of 15 Synthesis Inputs Don't Exist

The synthesis model (`src/synthesis_model.py:73-98`) defines 15 input features:

| Feature | Source | Data Exists? |
|---------|--------|-------------|
| station_mu | Wind-gated attention model output | NO — model never trained |
| station_sigma | Wind-gated attention model output | NO — model never trained |
| nwp_tmax | GFS/GEFS download | NO — no NWP data in data/ |
| nwp_t850 | GFS/GEFS download | NO |
| nwp_wind_speed | GFS/GEFS download | NO |
| nwp_wind_dir | GFS/GEFS download | NO |
| nwp_cloud_cover | GFS/GEFS download | NO |
| nwp_mslp | GFS/GEFS download | NO |
| nwp_precip | GFS/GEFS download | NO |
| nwp_ensemble_spread | Derived from NWP | NO |
| station_nwp_gap | station_mu - nwp_tmax | NO (both inputs missing) |
| abs_station_nwp_gap | abs(above) | NO |
| nwp_bias_7d | Rolling NWP bias | NO |
| sin_day | Cyclical date encoding | YES |
| cos_day | Cyclical date encoding | YES |

No `data/raw/nwp/`, `data/raw/asos/`, or `data/raw/igra/` directories exist. The NWP preprocessing module (`src/nwp_preprocessing.py:504`) explicitly sets ensemble_spread to NaN. If `prepare_synthesis_data()` were called, 11 features would be imputed to training-column-mean (0.0 for all-NaN columns), then StandardScaler would map them to near-zero — contributing only noise.

### What the operational NN ACTUALLY uses

`scripts/generate_max_training_predictions.py:197-251` builds features from:
- `nyc_tmax_lag1` (autoregressive)
- `{station_id}_tmax_lag1` for each qualifying surrounding station
- `sin_day`, `cos_day`

With ~47 qualifying stations from `config_expanded`, this produces approximately **49 features** (1 NYC lag + 46 surrounding lag + 2 date encodings). These are all real, populated features.

---

## 3. Station ID Labeling Errors in config_expanded.py

`config_expanded.py` line 59 lists:
```
"USW00014732": "LAGUARDIA AP, NY (5mi E)"
```

But in `config.py` (the original validated config), `USW00014732` is:
```
"USW00014732": "Islip, NY (Long Island MacArthur Airport)"
```

And separately in the expanded config (line 70):
```
"USW00004781": "ISLIP-LI MACARTHUR AP, NY (45mi E)"
```

**Impact:** The .dly file downloaded for `USW00014732` contains whichever station GHCN assigns to that ID. If this is actually Islip (45 mi away), calling it "LaGuardia (5mi E)" misrepresents its geographic contribution. The model trains on the correct data (from the .dly file), but any geographic analysis (distance weighting, sector assignment) would be wrong for this station.

**Recommendation:** Verify all station IDs in config_expanded.py against the GHCN station inventory (`ghcnd-stations.txt`). Cross-check coordinates.

---

## 4. The Actual NN Architecture

### What's trained: TempPredictorV1

Defined in `src/model.py:42`. Architecture:

```
Input(~49 features)
  → Linear(49, 128) → ReLU → [no dropout]
  → Linear(128, 64) → ReLU → [no dropout]
  → Linear(64, 1)   [linear output, no activation]
```

**Hyperparameters** (from `scripts/generate_max_training_predictions.py:96-102`):
- Hidden sizes: [128, 64]
- Dropout: 0.0
- Learning rate: 0.001 (Adam)
- Batch size: 128
- Max epochs: 300
- Early stopping patience: 20
- LR scheduler: ReduceLROnPlateau(patience=7, factor=0.5)
- Loss: HuberLoss(delta=1.0)

**Training data:** GHCN TMAX 1998-2020 (Model A) or 1998-2022 (Model B)
**Validation:** 2021-2022 (Model A) or 2023-2024 (Model B)
**Prediction:** 2023-2024 IS or 2025 OOS

### Parameter count

With 49 input features:
- Layer 1: 49×128 + 128 = 6,400
- Layer 2: 128×64 + 64 = 8,256
- Output: 64×1 + 1 = 65
- **Total: ~14,721 parameters**

Training set (1998-2020) has ~8,400 rows. Parameter-to-sample ratio: ~1:570. This is acceptable (not overparameterized) with the expanded date range.

---

## 5. Architecture Issues & Recommendations

### 5.1 Dropout = 0.0 (No Regularization)

The NN trains with zero dropout. Phase 4 sensitivity experiments found dropout=0.0 was optimal for the 5-year (1,277 sample) dataset. But with the expanded 22-year dataset (~8,400 training samples), modest dropout (0.05-0.1) may help generalization, especially for winter/spring where the model struggles most.

**Recommendation:** Re-run with dropout={0.0, 0.05, 0.1} on the 22-year data to verify.

### 5.2 Hidden Layers [128, 64] May Be Suboptimal

The current architecture was chosen in Phase 3 with 30 features and 1,277 samples. With ~49 features and ~8,400 samples, a wider or deeper network may extract more from the expanded station set:

| Architecture | Parameters | Ratio (8400 samples) |
|---|---|---|
| [128, 64] (current) | ~14,700 | 1:571 |
| [256, 128, 64] | ~46,000 | 1:183 |
| [64, 32] (smaller) | ~5,300 | 1:1585 |

The current architecture is reasonably sized. However, the single biggest limitation is likely **feature engineering** (TMAX-only, lag-1 only), not architecture depth.

**Recommendation:** Before tuning hidden layers, first try adding TMIN lag-1 features and lag-2 features. These were shown in Phase 4 experiments to have more impact than architecture changes.

### 5.3 Huber Loss vs. MAE/MSE

HuberLoss(delta=1.0) is a good choice — it's robust to outliers while maintaining gradient flow near zero. Phase 4 experiments showed Huber was optimal for the delta-T target formulation.

However, the operational pipeline predicts **raw TMAX**, not delta-T. For raw TMAX targets, MSE loss was slightly better in Phase 4 experiments.

**Recommendation:** The delta-T target formulation (predict the change, reconstruct TMAX) reduced MAE by ~0.27°F in Phase 4. This should be integrated into `generate_max_training_predictions.py`. It is the single highest-leverage modeling change available.

### 5.4 No Batch Normalization

The TempPredictorV1 uses no batch normalization. The synthesis model (if it were operational) uses BatchNorm throughout. For a 2-layer network with ReLU, BatchNorm is not strictly necessary but can help training stability.

**Recommendation:** Low priority. Skip unless other changes are exhausted.

---

## 6. Sigma is Monthly-Constant, Not Heteroscedastic

The prediction CSVs contain 12 unique sigma values (one per month), estimated from validation residuals:

```
Model A (IS) sigma range: 3.87 - 7.87°F
Model B (OOS) sigma range: 3.76 - 8.34°F
```

This means the probability distribution for every day in January is the same width, every day in July is the same width, etc. The model cannot express "I'm more uncertain today because stations disagree" or "I'm more confident because the weather pattern is stable."

**Impact on Kalshi:** A fixed monthly sigma produces bracket probabilities that don't respond to day-to-day forecast confidence variation. This is a significant handicap against MOS, which implicitly benefits from ensemble spread information.

**Recommendation:** Either:
1. Train the synthesis model (Gaussian mode) to produce per-sample sigma — this is what it was designed for
2. Or at minimum, add a second-stage sigma model (e.g., predict |residual| from features to get conditional sigma)

---

## 7. Wind Direction Hardcoded to Zero

In `src/train_phase1.py:295`:
```python
wd = np.zeros(n_samples, dtype=np.float32)
```

The wind-gated attention model's signature feature — weighting stations by upwind/downwind relationship — is completely bypassed because wind_direction is always 0.0. The attention mechanism degenerates to `cos(0 - bearing) = cos(bearing)`, which is a static geographic bias, not a weather-responsive signal.

**Impact:** If the wind-gated model were ever trained, this bug would prevent it from learning wind-dependent station weighting.

**Recommendation:** Source wind direction from ASOS data (already parsed in `src/asos_preprocessing.py` if data existed) or from NWP forecasts.

---

## 8. No Saved Model Checkpoints

The `models/` directory is empty. No `.pt` checkpoint files exist in the repository.

**Impact:** Previously trained models cannot be reloaded for inference or continued training. Every run starts from scratch.

**Recommendation:** Add checkpoint saving to the training pipeline and ensure `.gitignore` does not exclude them (or save to a tracked `models/` directory with clear naming).

---

## 9. Features That Should Be Added or Removed

### Features currently missing (high expected impact):

| Feature | Expected Impact | Source |
|---------|----------------|--------|
| **Delta-T target** (predict change, not raw TMAX) | -0.27°F MAE (Phase 4 result) | Code exists in `data_preprocessing_v2.py` |
| **TMIN lag-1** for surrounding stations | -0.1-0.2°F MAE (Phase 4 result) | GHCN .dly files already have TMIN |
| **Lag-2 TMAX** | Small improvement | Trivial to add |
| **MOS forecasts as input** | Potentially large (MOS MAE=2.51°F) | `data/mos/` files exist |

### Features to consider removing:

| Feature | Concern |
|---------|---------|
| Stations with <90% completeness (expanded config allows 80%) | More imputation = more noise |
| Distant Ring 4 stations (>150mi) | Phase 4.3 found >14 stations degrades performance with small data; check with large data |

---

## 10. Summary of Recommended Actions (Priority Order)

1. **Integrate delta-T target** into `generate_max_training_predictions.py` — biggest single MAE improvement available from existing code
2. **Add MOS forecasts as input features** — if the NN can learn to correct MOS biases, this bridges the 2.51 vs 4.3°F gap
3. **Build a runner for the synthesis model** — connect the wind-gated attention → synthesis pipeline with real data
4. **Download ASOS/NWP data** so that synthesis features are real, not phantom
5. **Fix station ID labeling** in `config_expanded.py`
6. **Add per-sample sigma** estimation (either via synthesis model or a second-stage model)
7. **Re-evaluate architecture** (dropout, hidden sizes) only after feature improvements are in place
