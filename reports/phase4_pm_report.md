# Phase 4 -- Enhancements: Project Manager Report

**Date:** 2026-02-08
**Phase:** 4 -- Enhancements (Steps 4.1, 4.2, 4.4, 4.5)
**Status:** COMPLETE

---

## Objective

Enhance the Phase 3 feedforward NN through target reformulation (delta-T), feature engineering (sector gradients, trend features, autoregressive input), new model architectures (LSTM, attention pooling), and systematic sensitivity experiments.

## Deliverables

### Source Code (new files)

| File | Purpose | Size |
|------|---------|------|
| `src/data_preprocessing_v2.py` | Enhanced pipeline: delta-T target, autoregressive features, sector averages/gradients, trend features, diurnal range, multi-lag | 25 KB |
| `src/train_v2.py` | Extended training: Huber/MAE/MSE loss, delta-T with reconstruction, dual MAE tracking | 21 KB |
| `src/models_v2.py` | 4 new architectures: EnhancedMLP, MultiLagMLP, LSTMPredictor, StationAttentionModel | 22 KB |
| `src/experiments.py` | Sensitivity experiment framework: config, runner, reporting | 24 KB |
| `run_phase4.py` | End-to-end Phase 4 runner (5 model variants) | 18 KB |
| `run_experiments.py` | Sensitivity experiment runner (15 experiments) | 10 KB |

### Tests (new files)

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/test_data_preprocessing_v2.py` | 55 | PASSED |
| `tests/test_train_v2.py` | 42 | PASSED |
| `tests/test_models_v2.py` | 69 | PASSED |
| `tests/test_experiments.py` | 34 | PASSED |
| **New Phase 4 total** | **200** | **ALL PASSED** |
| **Full project total** | **525** | **ALL PASSED** |

### Results

| Directory | Contents |
|-----------|----------|
| `results/phase4/` | 23 files: metrics JSON, report, 5x training curves, 5x scatter plots, 5x residual histograms, comparison bar chart |
| `results/experiments/` | 4 files: results CSV, report, architecture comparison chart, feature ablation chart |

---

## Phase 4 Results (Test Set, n=274)

### Step 4.1/4.2: Delta-T Target & Feature Engineering

| Model | MAE (F) | RMSE (F) | R2 | Key Change |
|-------|---------|----------|----|------------|
| Persistence (ref) | 5.06 | 6.39 | 0.799 | -- |
| Ridge (ref) | 4.33 | 5.41 | 0.876 | -- |
| NN V1 Phase 3 (ref) | 4.29 | 5.69 | 0.869 | -- |
| NN Raw+MSE | 4.30 | 5.69 | 0.868 | Reproduces Phase 3 |
| NN Raw+Huber | 4.36 | 5.73 | 0.867 | Huber on raw target |
| NN Delta+Huber (no AR) | 4.03 | 5.40 | 0.882 | Delta-T target |
| **NN Delta+Huber+AR** | **3.95** | **5.33** | **0.885** | **+NYC TMAX(t-1)** |
| NN Delta+Huber+Full | 4.15 | 5.45 | 0.880 | +all enhanced features (79 feat) |

**Best model: NN Delta+Huber+AR (MAE = 3.95 F)**
- 0.34 F improvement over Phase 3 NN V1 (8.0%)
- 0.38 F improvement over Ridge baseline (8.8%)
- 1.11 F improvement over Persistence (21.9%)

### Step 4.4/4.5: Architecture & Sensitivity Experiments

15 experiments completed. Top results (those beating Phase 3 MAE of 4.29 F):

| Experiment | MAE (F) | RMSE (F) | R2 |
|-----------|---------|----------|----|
| MLP [128,64] dropout=0.0, Huber | 4.16 | 5.75 | 0.866 |
| MLP [128,64] dropout=0.1, Huber | 4.22 | 5.76 | 0.865 |
| MLP [128,64] BatchNorm, Huber | 4.22 | 5.56 | 0.875 |

---

## Key Findings

### 1. Delta-T is the single biggest improvement
Switching from predicting raw TMAX to predicting daily change (DeltaT) reduced MAE by 0.27 F (6.3%). This confirms the project plan hypothesis: persistence is already strong, so predicting the "hard part" (changes) is more effective.

### 2. Autoregressive input adds incremental value
Adding NYC TMAX(t-1) as a feature on top of delta-T improves MAE by 0.07 F. The gap between "with AR" and "without AR" quantifies the incremental value of surrounding stations beyond persistence.

### 3. More features can overfit
The full enhanced feature set (79 features: diurnal range, sectors, gradients, trends) actually performed worse (MAE 4.15) than the simpler delta-T+AR model (MAE 3.95). With only 1,277 training samples, the model cannot exploit 79 features effectively. This suggests Phase 6 (25-year data) will be critical for benefiting from richer features.

### 4. Huber loss is beneficial for delta-T but not raw targets
On raw TMAX prediction, Huber loss is slightly worse than MSE. But for delta-T prediction (narrower output range), Huber's robustness to outliers helps.

### 5. Architecture insights
- BatchNorm achieves the best RMSE (5.56) and R2 (0.875) among the architecture experiments
- LSTM (MAE 4.34) performs comparably to MLP despite the data being single-lag
- Larger MLPs don't help: [256,128,64] performs worse than [128,64]
- Low dropout (0.0) is optimal given the small dataset

### 6. Feature importance
- TMAX features alone (MAE 4.32) nearly match TMAX+TMIN combined (MAE 4.36)
- TMIN alone is the weakest (MAE 4.91), confirming TMAX is the primary signal
- Date features (sin/cos day-of-year) contribute ~0.2 F MAE reduction

---

## Stretch Goal Progress

| Target | Current Best | Gap |
|--------|-------------|-----|
| MAE <= 2.0 F | 3.95 F | 1.95 F remaining |

The stretch goal remains challenging but the trajectory is positive. Paths forward:
1. Phase 6 (25-year data) should enable richer features
2. Multi-lag inputs (t-1, t-2, t-3) with the MultiLagMLP architecture
3. Station attention pooling with more stations
4. Residual learning / stacking (Step 4.6)

---

## Architecture Reference

### New Models Implemented

1. **EnhancedMLP**: Configurable feedforward with optional batch normalization
2. **MultiLagMLP**: Concatenated multi-lag features as flat input
3. **LSTMPredictor**: LSTM/GRU sequence model (supports bidirectional, multi-layer)
4. **StationAttentionModel**: Per-station shared encoder + multi-head attention pooling

### New Features Implemented

1. **Delta-T target**: TMAX(t) - TMAX(t-1), with reconstruction for evaluation
2. **Autoregressive input**: NYC TMAX(t-1) as feature
3. **Diurnal range**: Per-station TMAX - TMIN at each lag
4. **Sector averages**: Mean TMAX for WNW, SW, Coastal, NearField sectors
5. **Sector gradients**: upstream-vs-coast, SW-vs-NW temperature differences
6. **Trend features**: Delta1 (T(t-1) - T(t-2)), Delta2 (T(t-2) - T(t-3))

---

## Risks & Notes for Phase 5

1. The full enhanced feature set overfits -- need more data (Phase 6) or stronger regularization
2. The delta-T model has a slight warm bias (+0.45 F) that could be addressed
3. Confidence intervals (Phase 5) should use the best delta-T+AR model as the base
4. Steps 4.3 (station expansion), 4.6 (residual stacking), and 4.7 (season specialization) are deferred to after Phase 5/6 when more data is available
