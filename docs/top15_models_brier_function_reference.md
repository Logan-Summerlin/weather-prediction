# Top 15 Model Variants by Overall Brier (E0–E42 + U0–U9 scope)

**Last updated:** 2026-02-15

## Ranking scope and method
- Scope explicitly includes the E-lineage through **E42** and Unified variants through **U9**.
- Rankings were built by combining model rows from:
  - `results/prediction_market_benchmark/e0_e8_best_model_base/e0_e22_benchmark_summary.csv`
  - `results/prediction_market_benchmark/wga_v2_model/benchmark_summary.csv`
  - `results/prediction_market_benchmark/unified_outperformance/benchmark_summary.csv`
- Baseline comparators (`NWS`, `Original_Model`, `Kalshi_*`) are excluded from ranking.

## Top 15 (lowest overall Brier first)

| Rank | Variant | Overall Brier | Family |
|---:|---|---:|---|
| 1 | U7_regime_conditional | 0.113719 | Unified |
| 2 | E40_lag2_only_contract_brier | 0.113810 | E38–E42/WGA V2 |
| 3 | U6_platt_on_u5 | 0.114081 | Unified |
| 4 | E17_contract_brier_synthesis | 0.114090 | E0–E22 |
| 5 | E40_multihead_only_contract_brier | 0.114242 | E38–E42/WGA V2 |
| 6 | E21_platt_recalibrated_e17 | 0.114406 | E0–E22 |
| 7 | U9_kitchen_sink | 0.114545 | Unified |
| 8 | U4_extended_cal_synthesis | 0.114857 | Unified |
| 9 | E40_deep_only_contract_brier | 0.114880 | E38–E42/WGA V2 |
| 10 | E42_dual_attention_synthesis | 0.114981 | E38–E42/WGA V2 |
| 11 | U8_2023only_cal_brier_mlp | 0.115162 | Unified |
| 12 | U5_extended_cal_brier_mlp | 0.115415 | Unified |
| 13 | E39_deep_only_synthesis | 0.115569 | E38–E42/WGA V2 |
| 14 | E39_full_synthesis | 0.115838 | E38–E42/WGA V2 |
| 15 | E39_multihead_only_synthesis | 0.116007 | E38–E42/WGA V2 |

---

## Functions and methods for the ranked top 15

### Unified-family variants (U4/U5/U6/U7/U8/U9)
All implemented in `scripts/run_unified_outperformance_benchmark.py`.

1. **U4_extended_cal_synthesis**
   - Fit: `fit_u4_synthesis_stacker(...)`
   - Apply: `apply_u4_synthesis_stacker(...)`
   - Method: logistic stacker blending flat + WGA + NWS with market state and isotonic post-calibration.

2. **U5_extended_cal_brier_mlp**
   - Feature build: `_build_u5_features(...)` + `build_bucket_features(...)` + `build_market_state_features(...)`
   - Fit: `fit_contract_brier_mlp(...)`
   - Apply: `apply_contract_brier_mlp(...)`
   - Method: contract-level Brier-optimized MLP on extended calibration (IS: 2023+2024).

3. **U6_platt_on_u5**
   - Fit: `fit_platt_recalibration(...)` on U5 outputs
   - Apply: `apply_platt_recalibration(...)` then `_per_day_renormalize(...)`
   - Method: Platt recalibration layer on top of U5.

4. **U7_regime_conditional**
   - Feature extension: `_build_regime_features(...)` appended to U5 features.
   - Fit/apply via the same contract-MLP path as U5.
   - Method: regime-aware contract-level MLP (best overall Brier in current artifacts).

5. **U8_2023only_cal_brier_mlp**
   - Same core machinery as U5 but calibration fit window restricted to 2023.
   - Method: calibration-window sensitivity check for robustness.

6. **U9_kitchen_sink**
   - Base fit: `fit_contract_brier_mlp(...)` with wider architecture sweep.
   - Recalibration: `fit_platt_recalibration(...)` + `apply_platt_recalibration(...)` + per-day renorm.
   - Method: maximal feature/architecture stack with extra post-calibration.

### E0–E22 synthesis variants in top 15
Implemented in `scripts/run_e0_e8_best_model_benchmark.py`.

7. **E17_contract_brier_synthesis**
   - Fit: `_fit_e17_contract_brier_synthesis(...)`
   - Apply dispatch: `_apply_variant(..., "E17_contract_brier_synthesis", ...)`
   - Method: contract-row MLP trained on model/NWS/market + bucket geometry + market-state features; isotonic post-calibration.

8. **E21_platt_recalibrated_e17**
   - Fit: `_fit_e21_platt_e17(...)`
   - Apply dispatch: `_apply_variant(..., "E21_platt_recalibrated_e17", ...)`
   - Method: two-stage recalibration over E17 (Platt on logits, then isotonic).

### WGA V2 lineage variants in top 15
Implemented in `scripts/run_wga_v2_benchmark.py`.

9. **E39_deep_only_synthesis / E39_full_synthesis / E39_multihead_only_synthesis**
   - Fit/apply family: `fit_e39_synthesis_logistic(...)`, `apply_e39_synthesis_logistic(...)`
   - Method: logistic synthesis blending WGA-V2 probs, original flat model probs, NWS, market state, and disagreement interactions.

10. **E40_lag2_only_contract_brier / E40_multihead_only_contract_brier / E40_deep_only_contract_brier**
    - Fit: `fit_e40_contract_brier_mlp(...)`
    - Apply: `apply_e40_contract_brier_mlp(...)`
    - Method: contract-level Brier-optimized MLP over WGA-V2 + flat + NWS + market + bucket-feature composites.

11. **E42_dual_attention_synthesis**
    - Fit: `fit_e42_dual_attention_synthesis(...)`
    - Apply: `apply_e42_dual_attention_synthesis(...)`
    - Method: dual-model feature stack using both WGA-V2 and original flat model bucket descriptors, cross-model disagreement, and market-state interactions.

---

## Shared implementation primitives used repeatedly
- Bucketization:
  - `compute_bucket_probs(...)`
  - `compute_bucket_probs_from_arrays(...)`
- Calibration helpers:
  - `fit_isotonic(...)`, `apply_isotonic(...)`
  - `fit_platt_recalibration(...)`, `apply_platt_recalibration(...)`
- Feature engines:
  - `build_bucket_features(...)`
  - `build_market_state_features(...)`
  - `_build_u5_features(...)`, `_build_regime_features(...)`
