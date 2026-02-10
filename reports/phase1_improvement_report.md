# Phase 1 Model Improvement Report — Project Manager Summary

**Date:** 2026-02-10
**Objective:** Implement Phase 1 "Quick Wins" from master_improvement_plan.md to improve NYC TMAX prediction accuracy for prediction market trading.

---

## Executive Summary

Four analyst workstreams ran in parallel to implement feature engineering, probabilistic output, ensemble training, and architecture improvements. A synthesis step combined the best findings into a unified model.

**Bottom line: OOS MAE improved from 2.093°F to 2.018°F (-3.6%), with test MAE breaking sub-2°F at 1.987°F (-4.9%).**

---

## Results Overview

### Combined Best Models (all features, all optimizations)

| Model | Test MAE | OOS MAE | Test RMSE | OOS RMSE | Notes |
|-------|----------|---------|-----------|----------|-------|
| **E_Ensemble_5seed** | **1.987** | **2.018** | 2.621 | 2.684 | Best test MAE (sub-2°F!) |
| A_NN_64_32 (seed 42) | 2.015 | **2.010** | 2.658 | 2.689 | Best OOS single model |
| E_seed_123 | 1.998 | 2.037 | 2.633 | 2.693 | |
| C_ResidualNN | 2.006 | 2.022 | 2.654 | 2.672 | Residual connections help |
| D_Probabilistic [64,32] | 2.029 | 2.083 | 2.655 | 2.736 | Calibrated 95% PI |
| G_HGB | 2.049 | 2.120 | 2.696 | 2.824 | Gradient boosting |
| F_Ridge | 2.102 | 2.231 | 2.710 | 2.862 | Linear baseline |
| **Prior baseline** | **2.090** | **2.093** | — | — | C_Correction_NN_tiny |

### Seasonal Performance (5-seed ensemble, combined features)

| Season | Test MAE | OOS MAE |
|--------|----------|---------|
| DJF (Winter) | 2.145 | 1.880 |
| MAM (Spring) | 2.383 | 2.851 |
| JJA (Summer) | 1.805 | 1.633 |
| SON (Fall) | 1.615 | 1.703 |

Spring remains the hardest season (MAM OOS ~2.85°F). Summer and fall predictions are excellent (<1.8°F).

---

## Analyst Workstream Results

### Analyst 1: Feature Engineering (32 configurations)

**Best:** Phase1A+1C NN[64,32] = test 2.009 / OOS 2.013

Key findings:
- **MOS error memory (Phase 1A)** is the single highest-impact feature group: -0.05 to -0.10°F
- Features: rolling 7/14/30-day MOS bias, yesterday's MOS error, GFS-NAM spread
- **MOS×Station interactions (Phase 1C)** add -0.02 to -0.03°F on top
- Features: MOS-station gap, sector gap, station-MOS agreement flag
- Semi-annual harmonics and spatial gradients (Phase 1B) provide modest but consistent improvement
- Feature ablation ranking: 1A+1C > AllPhase1 > 1A+1B > 1A > 1C > 1B > Baseline

### Analyst 2: Probabilistic Output (5 variants)

**Best point MAE:** D_NLL_CRPS_large = test 2.032 / OOS 2.150
**Best calibration:** 95% PI coverage = 94.7-95.6% (well calibrated)

Key findings:
- Two-stage NLL→CRPS training produces well-calibrated prediction intervals
- Sigma floor of 0.75°F prevents overconfidence
- Mean predicted sigma ~2.7°F across models
- CRPS: 1.48°F test, 1.52°F OOS
- Point MAE slightly worse than pure MAE-trained models (expected tradeoff)

### Analyst 3: Ensemble + Training Protocol (40+ configurations)

**Best:** Combined 5-seed ensemble = test 2.064 / OOS 2.133

Key findings:
- 5-seed ensemble reduces MAE by -0.01 to -0.03°F
- **Weight decay = 1e-4** is optimal (vs default 1e-5)
- ReduceLROnPlateau slightly outperforms CosineAnnealingWarmRestarts
- SWA provides marginal benefit; not always worth the training time
- Expanding-window CV confirms consistent performance across folds

### Analyst 4: Architecture + Temporal Features (16 configurations)

**Best:** C_64_32_16 enhanced = test 2.082 / OOS 2.083

Key findings:
- Enhanced temporal features (day length, solar elevation, anomalies) improve OOS generalization
- Enhanced spatial features (frontal proxy, gradients, station consensus) add modest value
- Residual connections help test MAE (-0.06°F)
- 3-layer [64,32,16] offers good capacity/generalization tradeoff
- HGB competitive but slightly worse than best NNs on OOS

---

## Feature Importance Ranking

Based on ablation studies across all analysts:

1. **MOS ensemble forecast** (existing) — dominant input
2. **MOS error memory** (7/14/30-day rolling bias) — NEW, highest impact
3. **NYC autoregressive lag** (TMAX t-1) — existing, key anchor
4. **MOS×Station interaction** (gap, agreement) — NEW
5. **Station spatial composite** (mean TMAX, sector means) — existing
6. **Solar/temporal encoding** (day length, anomaly from climo) — NEW
7. **Spatial gradients** (WNW-coast, NE-SW, ring gradient) — NEW
8. **Semi-annual harmonics** (sin/cos 2×year cycle) — NEW, modest

---

## Recommendations

### For Production Deployment
**Recommended model:** A_NN_64_32 with combined features (single seed 42)
- Test MAE: 2.015°F, OOS MAE: 2.010°F
- Simple, fast inference, deterministic
- Add probabilistic wrapper (D variant) for confidence intervals when needed

### For Maximum Accuracy
**Use:** 5-seed ensemble (E_Ensemble_5seed)
- Test MAE: 1.987°F, OOS MAE: 2.018°F
- 5× inference cost, but sub-2°F accuracy

### For Prediction Market Trading
**Use:** D_Probabilistic with combined features
- Point MAE: 2.029°F test, 2.083°F OOS
- Calibrated 95% prediction intervals
- Essential for Kalshi position sizing and risk management

### Next Steps (Phase 2+)
1. **ASOS integration** (wind shift, pressure tendency) — expected -0.1 to -0.2°F
2. **Seasonal-conditional calibration** for spring improvement (MAM is weakest)
3. **Regime-aware multi-head model** for front passage vs. stable patterns
4. **Retrain with 2025 data** as it becomes available for expanding training window

---

## Files Produced

| Analyst | Script | Results |
|---------|--------|---------|
| 1 (Features) | `scripts/phase1_feature_engineering.py` | `results/phase1_features/` |
| 2 (Probabilistic) | `scripts/phase1_probabilistic_output.py` | `results/phase1_probabilistic/` |
| 3 (Ensemble) | `scripts/phase1_ensemble_training.py` | `results/phase1_ensemble/` |
| 4 (Architecture) | `scripts/phase1_architecture_temporal.py` | `results/phase1_architecture/` |
| Synthesis | `scripts/phase1_combined_best.py` | `results/phase1_combined/` |

Total configurations tested: **100+ model variants** across 5 workstreams.
All results use real NOAA GHCN data (48 stations, 25 years, 2004-2024).
