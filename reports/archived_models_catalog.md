# Archived & Previous Model Families Catalog (Top 10)

**Date:** 2026-02-10  
**Purpose:** High-level catalog of major model families tried in this repo, how they were implemented, and broad outcomes.

---

## Scope and framing

This document summarizes broad **model families** (not every hyperparameter run) spanning baseline, neural, probabilistic, ensemble, and market-proxy approaches. It is designed as a historical technical map for future model selection.

---

## Top 10 previously implemented model families

## 1) Persistence baseline (naive lag-1)

- **What it is:** Predict tomorrow’s NYC TMAX as yesterday’s observed NYC TMAX.
- **How implemented:** `PersistenceModel` in `src/baselines.py` with lag-1 shift logic and split-boundary handling.
- **Why used:** Hard baseline to beat; captures temporal inertia.
- **General outcome:** Useful benchmark floor; insufficient for market-grade edges alone.

## 2) Climatology baseline (day-of-year average)

- **What it is:** Predict from historical day-of-year means.
- **How implemented:** `ClimatologyModel` in `src/baselines.py`, learned from training-period day-of-year statistics only.
- **Why used:** Seasonality anchor and leakage-safe benchmark.
- **General outcome:** Better seasonal structure than naive constant models; still weak on synoptic regime shifts.

## 3) Linear feature models (Linear/Ridge/ElasticNet)

- **What it is:** Regularized linear models over engineered station/MOS features.
- **How implemented:** baseline and advanced scripts use ridge/elastic-net families (`run_baselines.py`, `scripts/advanced_models_eval.py`, `scripts/phase1_feature_engineering.py`, `scripts/phase1_combined_best.py`).
- **Why used:** Strong low-variance baseline and interpretability check.
- **General outcome:** Competitive and robust; eventually outperformed by MOS-corrected nonlinear models.

## 4) Vanilla station MLP (TempPredictorV1)

- **What it is:** Feedforward neural net over lagged station features plus seasonal encodings.
- **How implemented:** Early pipeline in `run_nn.py` and `src/model.py` (`TempPredictorV1` path).
- **Why used:** First nonlinear baseline beyond linear/ridge.
- **General outcome:** Established NN viability but later superseded by MOS-aware residual pipelines.

## 5) Multi-lag / enhanced MLPs

- **What it is:** MLPs with richer temporal context (multi-day windows) and larger hidden stacks.
- **How implemented:** `EnhancedMLP` / `MultiLagMLP` in `scripts/advanced_models_eval.py`; additional depth sweeps in phase-1 architecture scripts.
- **Why used:** Test whether more context and capacity improve regime handling.
- **General outcome:** Incremental gains in some slices; pure capacity increases alone were not the main lever.

## 6) Sequence RNNs (LSTM/GRU)

- **What it is:** Sequential neural models over short station-history windows.
- **How implemented:** `LSTMModel` and `GRUModel` in `scripts/advanced_models_eval.py` with chronological train/val/test.
- **Why used:** Hypothesis that explicit sequence modeling could beat static MLPs.
- **General outcome:** Mixed-to-underwhelming versus simpler MOS-residual MLP approaches at this horizon.

## 7) Temporal Conv1D sequence model

- **What it is:** 1D temporal convolution over short lag windows.
- **How implemented:** `TemporalConv1D` in `scripts/advanced_models_eval.py`.
- **Why used:** Lower-latency alternative to RNNs for local temporal pattern extraction.
- **General outcome:** Did not become dominant versus residual MLP + feature-engineering stacks.

## 8) MOS residual-correction neural models (core historical winner class)

- **What it is:** Predict residual error relative to MOS ensemble forecast, then add correction back to MOS baseline.
- **How implemented:** MOS-correction pipelines in `scripts/mos_ensemble_pipeline.py`, `scripts/phase1_feature_engineering.py`, and `scripts/phase1_architecture_temporal.py` including tiny NN, residual-connection NN, and skip variants.
- **Why used:** Align model objective with incremental value beyond public forecast signal.
- **General outcome:** Major step-change class; became the most consistently strong family prior to probabilistic/ensemble upgrades.

## 9) Tree-boosting residual models (GBR/HGB)

- **What it is:** Gradient-boosted tree models on MOS residual targets.
- **How implemented:** `GradientBoostingRegressor` / `HistGradientBoostingRegressor` comparison tracks in `scripts/mos_ensemble_pipeline.py`, `scripts/phase1_feature_engineering.py`, and `scripts/phase1_architecture_temporal.py`.
- **Why used:** Nonlinear tabular baseline with different bias-variance profile than NNs.
- **General outcome:** Often competitive and useful as challenger model; generally slightly behind best NN residual stacks.

## 10) Probabilistic + ensemble model family (current historical top tier)

- **What it is:**
  - **Probabilistic:** heteroscedastic Gaussian heads producing `(mu, sigma)` via NLL then CRPS fine-tuning.
  - **Ensemble:** multi-seed averaging to stabilize/generalize point and distributional outputs.
- **How implemented:**
  - probabilistic correction models in `scripts/phase1_probabilistic_output.py`;
  - 5-seed training/evaluation in `scripts/phase1_ensemble_training.py` and synthesis in `scripts/phase1_combined_best.py`.
- **Why used:** Required for bucket probabilities, calibration, and risk-aware market decisions.
- **General outcome:** Best historical family in repo-era reports; strong point accuracy and better uncertainty outputs than point-only models.

---

## Additional implemented but still pre-mainline families

### Wind-gated station attention

- **What it is:** Station attention with wind-direction gating to upweight upwind stations.
- **How implemented:** `WindGatedAttentionModel` in `src/wind_gated_attention.py` with missing-station masking and optional Gaussian output head.
- **Role:** Architecture pathway for stronger physics-conditioned aggregation; implemented but not yet the canonical benchmark default.

### Synthesis/meta model layer

- **What it is:** Meta-layer to combine station-model signal with forecast-model/market context.
- **How implemented:** `src/synthesis_model.py` and related integration tests.
- **Role:** Framework for reliability-weighted source blending; retained for iterative development.

### Market proxy models

- **What it is:** Proxy distributions to emulate market-implied probability structure.
- **How implemented:** `src/market_proxy.py`, `src/mos_market_proxy.py`, and `src/enhanced_market_proxy.py`.
- **Role:** Benchmarking and EV diagnostics when direct market state is partial/noisy.

---

## Broad historical lessons

1. **Biggest practical gains came from residualizing on MOS**, not from architecture depth alone.
2. **Probabilistic outputs are required** for bucketization/calibration workflows.
3. **Ensembling improved stability** and typically improved out-of-sample robustness.
4. **Tree boosters remain important challengers** but did not clearly dominate best residual NNs.
5. **Pure sequence complexity (LSTM/GRU/Conv1D) did not reliably beat simpler, well-engineered residual pipelines** at this prediction horizon.

---

## Implementation reference map

- Baselines: `src/baselines.py`, `run_baselines.py`
- Early NN path: `src/model.py`, `run_nn.py`
- Advanced architecture sweep: `scripts/advanced_models_eval.py`
- MOS ensemble/residual pipeline: `scripts/mos_ensemble_pipeline.py`
- Phase-1 feature/probabilistic/ensemble/combined:  
  `scripts/phase1_feature_engineering.py`,  
  `scripts/phase1_probabilistic_output.py`,  
  `scripts/phase1_ensemble_training.py`,  
  `scripts/phase1_combined_best.py`
- Attention and synthesis: `src/wind_gated_attention.py`, `src/synthesis_model.py`
- Market proxies: `src/market_proxy.py`, `src/mos_market_proxy.py`, `src/enhanced_market_proxy.py`
