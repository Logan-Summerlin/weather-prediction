# Extended vs Original Model: Full E0-E22 Benchmark Comparison

**Date:** 2026-02-13
**Purpose:** Determine whether extending training data to 2000 via airport MOS proxy was the right choice.

## 1. Setup Differences

| Parameter | Original Model | Extended Model |
|-----------|---------------|----------------|
| Training period | 2004-01 to 2020-12 | 2000-06 to 2021-12 |
| Validation period | 2021-2022 | 2022-2023 |
| Test period | 2023-2024 | 2024 |
| OOS period | 2025 | 2025 |
| Calibration window | 2023 only (~2,008 rows) | 2022-2023 (~4,000+ rows) |
| MOS source (pre-2004) | N/A | Airport proxy (KJFK+KLGA+KEWR avg with monthly offsets) |
| Features | 121 | 122 (+ mos_era binary) |
| Training samples | ~6,200 | ~7,862 |

**Important confounds:** The changes are not a pure data-extension ablation. Three things changed simultaneously:
1. More training data (2000-2003 via airport proxy)
2. Different validation window (2022-2023 vs 2021-2022)
3. Different calibration window for benchmark variants (2022-2023 vs 2023-only)
4. Additional mos_era feature

This means improvements or regressions could stem from any combination of these factors.

## 2. Full Variant-by-Variant Comparison

### Synthesis Tier (E11, E13, E17, E18, E19, E21, E22)

| Model | Orig Overall | Ext Overall | Delta | Orig OOS | Ext OOS | Delta | Verdict |
|-------|:-----------:|:----------:|:-----:|:--------:|:-------:|:-----:|---------|
| E17_contract_brier_synthesis | 0.11409 | 0.11396 | **-0.00013** | 0.10664 | 0.10555 | **-0.00109** | IMPROVED |
| E19_platt_beta_calibration | 0.11638 | 0.11464 | **-0.00174** | 0.10384 | 0.10581 | +0.00197 | MIXED (overall better, OOS worse) |
| E18_regime_adaptive_ensemble | 0.12386 | 0.11469 | **-0.00917** | 0.11308 | 0.10501 | **-0.00807** | STRONGLY IMPROVED |
| E11_synthesis_stacker | 0.11658 | 0.11493 | **-0.00165** | 0.10536 | 0.10267 | **-0.00269** | IMPROVED |
| E21_platt_recalibrated_e17 | 0.11441 | 0.11495 | +0.00054 | 0.10902 | 0.10767 | **-0.00135** | MIXED (overall worse, OOS better) |
| E13_neural_synthesis_mlp | 0.11620 | 0.11502 | **-0.00118** | 0.10358 | 0.10555 | +0.00197 | MIXED (overall better, OOS worse) |
| E22_expanded_platt_e13 | 0.11633 | 0.11584 | **-0.00049** | 0.10431 | 0.10734 | +0.00303 | MIXED (overall better, OOS worse) |

**Synthesis tier summary:**
- 3 variants clearly improved on OOS: E17, E18, E11
- 3 variants regressed on OOS: E19 (+0.0020), E13 (+0.0020), E22 (+0.0030)
- 1 mixed: E21 (slight OOS improvement but overall regression)
- **Pattern:** Linear/simple stackers (E11) and regime-aware methods (E18) benefited most from doubled calibration data. Complex neural methods (E13, E19) that were tightly fitted to 2023 regressed.

### Base Tier (E0-E8, E10, E12)

| Model | Orig Overall | Ext Overall | Delta | Orig OOS | Ext OOS | Delta | Verdict |
|-------|:-----------:|:----------:|:-----:|:--------:|:-------:|:-----:|---------|
| E0_baseline_ensemble | 0.13352 | 0.13360 | +0.00008 | 0.13019 | 0.13014 | **-0.00005** | NEUTRAL |
| E1_global_isotonic | 0.13337 | 0.13413 | +0.00076 | 0.13050 | 0.13165 | +0.00115 | REGRESSED |
| E2_seasonal_calibration | 0.13425 | 0.13526 | +0.00101 | 0.13159 | 0.13352 | +0.00194 | REGRESSED |
| E3_weighted_ensemble | 0.13331 | 0.13333 | +0.00002 | 0.13005 | 0.12992 | **-0.00013** | NEUTRAL |
| E4_uncertainty_decomp | 0.13339 | 0.13339 | +0.00000 | 0.13010 | 0.12996 | **-0.00013** | NEUTRAL |
| E5_mdn2 | 0.13344 | 0.13341 | **-0.00004** | 0.12993 | 0.12986 | **-0.00008** | NEUTRAL |
| E6_quantile | 0.13441 | 0.13416 | **-0.00025** | 0.13137 | 0.13100 | **-0.00036** | SLIGHTLY IMPROVED |
| E7_regularization_sweep | 0.13362 | 0.13395 | +0.00032 | 0.13054 | 0.13100 | +0.00046 | SLIGHTLY REGRESSED |
| E8_feature_pruning_sweep | 0.13635 | 0.13746 | +0.00111 | 0.13477 | 0.13681 | +0.00204 | REGRESSED |
| E10_wga_mdn_regime | 0.13358 | 0.13360 | +0.00002 | 0.13057 | 0.13044 | **-0.00013** | NEUTRAL |
| E12_capacity_sweep | 0.13377 | 0.13412 | +0.00035 | 0.13056 | 0.13073 | +0.00017 | NEUTRAL |

**Base tier summary:**
- Most variants barely moved (within noise range of +/-0.0005)
- E1, E2, E8 regressed moderately (these are calibration-heavy variants — isotonic, seasonal, feature-pruning — that are sensitive to the calibration window change)
- E3, E4, E5, E6, E10 essentially unchanged or slightly improved
- **Pattern:** Simple model-level transforms (E0, E3-E5) are robust to the data extension. Calibration-specific variants (E1, E2) regressed because they depend on the calibration window composition, which changed.

### Calibration-Specific Tier (E9, E15, E16)

| Model | Orig Overall | Ext Overall | Delta | Orig OOS | Ext OOS | Delta | Verdict |
|-------|:-----------:|:----------:|:-----:|:--------:|:-------:|:-----:|---------|
| E9_conditional_cal_grid | 0.13425 | 0.13526 | +0.00101 | 0.13159 | 0.13359 | +0.00201 | REGRESSED |
| E15_conditional_cal_spread | 0.13447 | 0.13549 | +0.00102 | 0.13239 | 0.13442 | +0.00204 | REGRESSED |
| E16_conditional_cal_shrunk | 0.13413 | 0.13512 | +0.00099 | 0.13179 | 0.13368 | +0.00189 | REGRESSED |

**Calibration tier summary:**
- All three regressed by a similar magnitude (~+0.001 overall, ~+0.002 OOS)
- **Pattern:** Conditional calibration methods split the calibration data into finer cells. Doubling the calibration window should help, but these cells now span 2022+2023 instead of 2023-only, and 2022's distribution may differ enough to add noise rather than signal to the cell estimates.

### Failed Tier (E14, E20)

| Model | Orig Overall | Ext Overall | Delta | Orig OOS | Ext OOS | Delta | Verdict |
|-------|:-----------:|:----------:|:-----:|:--------:|:-------:|:-----:|---------|
| E14_distributional_nll | 0.19439 | 0.20188 | +0.00749 | 0.19198 | 0.20338 | +0.01140 | WORSENED |
| E20_crps_distributional | 0.20480 | 0.20188 | -0.00292 | 0.20526 | 0.20338 | -0.00188 | SLIGHTLY BETTER |

- E14 and E20 are now identical in the extended model (0.20188 overall, 0.20338 OOS). This suggests both converged to the same degenerate solution — likely both selecting the same base model distribution without meaningful synthesis.
- These remain failed variants regardless.

## 3. Trading P&L Impact

| Model | Orig OOS P&L | Ext OOS P&L | Delta | Verdict |
|-------|:-----------:|:----------:|:-----:|---------|
| E11_synthesis_stacker | -$3.57 | **+$1.39** | **+$4.96** | FLIPPED POSITIVE |
| E18_regime_adaptive | -$20.69 | -$1.05 | +$19.64 | MASSIVE IMPROVEMENT |
| E19_platt_beta | **+$3.63** | -$5.90 | -$9.53 | FLIPPED NEGATIVE |
| E13_neural_synthesis | -$1.01 | -$4.72 | -$3.71 | WORSENED |
| E17_contract_brier | -$6.08 | -$7.15 | -$1.07 | SLIGHTLY WORSE |
| E21_platt_recalibrated_e17 | -$12.61 | -$6.15 | +$6.46 | IMPROVED |
| E22_expanded_platt_e13 | -$4.74 | -$6.96 | -$2.22 | WORSENED |
| E0_baseline_ensemble | -$19.52 | -$17.24 | +$2.28 | SLIGHTLY BETTER |
| E6_quantile | -$17.78 | -$13.30 | +$4.48 | IMPROVED |
| E12_capacity_sweep | -$18.55 | -$14.08 | +$4.47 | IMPROVED |

**Trading summary:**
- E11 is the big winner: from -$3.57 to +$1.39 (only variant with positive OOS P&L in extended model)
- E18 massively improved: -$20.69 to -$1.05 (near break-even)
- E19 is the big loser: from +$3.63 to -$5.90 (lost its positive P&L — fragile/overfit to 2023)
- **Key concern:** E19 was the only positive-P&L variant in the original model. Its regression suggests that result was fragile. E11's new positive P&L in the extended model is more trustworthy because it comes from a more robust calibration setup.

## 4. Paper-Trading Gate Comparison

| Gate | Original | Extended | Direction |
|------|----------|----------|-----------|
| OOS Brier ≤ PreSettlement | PASS (0.1066) | PASS (0.1056) | Improved |
| OOS gated P&L positive + CI | FAIL (-$3.79) | FAIL (-$3.86) | Same |
| ECE ≤ 0.03 | PASS (0.0129) | PASS (0.0153) | Slightly worse |
| Tail reliability ≤ 0.20 | PASS (0.181) | FAIL (0.255) | **REGRESSED** |
| **Total passing** | **3/4** | **2/4** | **Net worse** |

The tail reliability regression is the most concerning change. Root cause: the extended model's sigma is tighter (monthly mean ~2.79°F vs ~3.02°F), producing overconfident tail probabilities.

## 5. Analysis: What Drove the Changes?

### Winners (E17, E18, E11)
These variants share a common pattern: they use **simple, data-hungry calibration methods** (linear stacking, regime-conditioned MLP) that benefit directly from having 2x the calibration data (2022+2023 vs 2023-only).

- **E11** (linear stacker): simple logistic blend of model/NWS/market probs. With 2x calibration data, the blend weights are more stable.
- **E18** (regime-adaptive ensemble): MLP over 5 variant outputs conditioned on regime features. With only 2023 data, regime cells were too sparse. 2022+2023 data fills the cells.
- **E17** (contract-level brier synthesis): MLP trained directly on contract rows. More calibration data = better bucket-level training.

### Losers (E13, E19, E22)
These variants share a common pattern: they use **complex, overparameterized calibration** that may have overfit to 2023's specific distribution.

- **E13** (neural synthesis MLP): nonlinear synthesis with many interaction features. The neural MLP had enough capacity to memorize 2023 patterns that don't transfer to 2022 or 2025.
- **E19** (Platt + isotonic on E13): takes E13's output and applies additional Platt scaling. If E13 was tuned to 2023, Platt on top of it amplifies the overfitting.
- **E22** (expanded multi-feature Platt on E13): even more features on top of E13 = more overfitting channels.

### Calibration variants (E1, E2, E9, E15, E16)
All regressed by similar amounts (~+0.002 OOS). These variants apply isotonic/seasonal/conditional calibration directly to the base model probabilities. The calibration window shift from 2023 to 2022+2023 changes the calibration mapping. Since these methods are fitting monotonic transforms to histograms, a distribution shift between 2022 and 2023 (which is plausible — different weather patterns) introduces noise.

### Base tier (E0, E3-E5, E10)
Essentially unchanged. These apply simple, low-parameter transforms (identity, uncertainty weighting, MDN-style) that don't depend heavily on the calibration window.

## 6. Sigma Regression Investigation

The extended model's sigma_by_month (from test-set residuals):

| Month | Extended Sigma | Impact |
|-------|:-------------:|--------|
| Jan | 2.79 | - |
| Feb | 2.31 | Tighter than expected |
| Mar | 3.18 | OK |
| Apr | 3.52 | OK |
| May | 3.02 | OK |
| Jun | 1.83 | Very tight |
| Jul | 2.63 | OK |
| Aug | 2.12 | Tight |
| Sep | 1.96 | Tight |
| Oct | 2.06 | OK |
| Nov | 2.30 | OK |
| Dec | 2.86 | OK |

The sigma values are computed from test-set (2024) residuals only. The test set changed from 2023-2024 to 2024-only, which means sigma is estimated from 365 days instead of ~730. With fewer samples per month (~30 vs ~60), sigma estimates are noisier and can systematically underestimate if 2024 happened to be an easier-than-average year for the model.

## 7. Verdict: Was the Extension the Right Choice?

### Arguments FOR keeping the extended model:
1. **E11 OOS Brier improved significantly** (0.1054 → 0.1027) — closest to PreSettlement (0.0988)
2. **E18 showed massive improvement** (0.1131 → 0.1050) — regime conditioning now works
3. **E11 achieved positive OOS P&L** (+$1.39) — first variant to achieve this with robust calibration
4. **More training data is fundamentally good** — the model's point prediction improved (MAE 2.020 → 2.011)
5. **Doubled calibration window** should produce more stable calibration in production

### Arguments AGAINST (or for caution):
1. **Tail reliability gate regressed** (0.181 → 0.255) — lost a gate check
2. **E13 and E19 regressed** — the previous best OOS Brier (E13: 0.1036) no longer achieves that
3. **ECE slightly worse** (0.0129 → 0.0153) — still passes but less margin
4. **E19's positive P&L was fragile** — this weakens confidence in any single P&L result, including E11's new +$1.39
5. **Net gate regression** (3/4 → 2/4 passing)

### Recommendation:

**The extended model is the right foundation going forward**, but with caveats:

1. **Fix the sigma**: The tail reliability regression is a solvable problem (sigma recalibration on holdout, conformal calibration, or explicit widening). The underlying model quality is better.
2. **Prefer E11 over E13 for production**: E11's simpler stacking is more robust to calibration window shifts. E13's neural complexity was overfitting.
3. **Don't trust any single P&L result**: Both E19's prior +$3.63 and E11's new +$1.39 should be viewed as noise-level. The model needs to demonstrate consistently positive P&L across multiple evaluation windows.
4. **The calibration window expansion was likely the bigger driver than the extra training years**: E18's dramatic improvement (regime conditioning needing more data) and the calibration-variant regressions (distribution shift) both point to the calibration window as the key variable.

### Suggested ablation to confirm:
Run the original model (2004-2020 training) with 2022-2023 calibration (instead of 2023-only) to isolate the calibration window effect from the training data extension effect. If E11 and E18 still improve with just the calibration change, the training data extension is secondary.
