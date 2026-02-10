# Probabilistic & Ensemble Model Improvement Experiment Report

**Date:** 2026-02-10  
**Author:** GPT-5.2-Codex  
**Scope:** Practical, low-leakage, time-safe experiments to improve the repository's probabilistic and ensemble weather forecast performance under a mostly fixed-data regime.

---

## 1) Objective and constraints

### Goal
Improve **distribution quality** and **decision usefulness** (for bucketized contracts and EV), not just point MAE.

### Constraints
- We likely will not get meaningfully more historical observations in the near term.
- Any live-facing improvements must preserve time-safety and cutoff availability assumptions.
- Complexity should be added only when it provides out-of-sample gains vs simpler baselines.

### Success criteria (ranked)
1. Better probabilistic score on holdout (CRPS, then NLL).
2. Better calibration on holdout (PIT/reliability/coverage).
3. Better bucket-probability quality (Brier / log score by bucket).
4. Stable or improved point MAE/RMSE as secondary diagnostics.

---

## 2) Current archived model families (single comparison table)

| Family | Trained on | Target | Training process | Input → output | Output form |
|---|---|---|---|---|---|
| Persistence baseline | Historical NYC target series | NYC TMAX(t) | No parameter learning; predicts day *t* as observed day *t-1* | Date index + previous actual values → prediction | Scalar point prediction |
| Climatology baseline | Train-period NYC target grouped by day-of-year | NYC TMAX(t) | Computes day-of-year mean from train only | Date (DOY) → climatological value | Scalar point prediction |
| Linear/Ridge/ElasticNet | Lagged station features + seasonal encoding; train-only scaling | Usually NYC TMAX(t) | Fit regularized linear models on chronological splits | Feature vector → prediction | Scalar point prediction |
| Vanilla station MLP (TempPredictorV1) | Processed station dataset (train/val/test) | NYC TMAX(t), unscaled °F | Feedforward MLP with early stopping/LR schedule | Scaled feature vector → NN output | Scalar point prediction |
| Enhanced/Multi-lag MLPs | Long NOAA station history + multi-day windows | NYC TMAX(t) / delta variants | Larger MLPs over lag windows, same chrono eval | Multi-lag feature vector → prediction | Scalar point prediction |
| LSTM/GRU sequence models | Same station history, sequence windows | NYC TMAX(t) | Sequence encoder + FC head, early stopping | (batch, seq, feat) → prediction | Scalar point prediction |
| Temporal Conv1D | Same sequence setup | NYC TMAX(t) | Conv1D temporal extraction + linear head | Sequence/reshaped tensor → prediction | Scalar point prediction |
| MOS residual-correction NNs | Station lag features + MOS features + NYC lag/date features | Residual = actual − MOS ensemble | Train residual model, then add correction to MOS baseline | Features → residual → corrected temp | Scalar corrected point prediction |
| Tree-boosting residual models (GBR/HGB) | Same MOS residual tabular data | Residual = actual − MOS ensemble | Gradient-boosted tree regressors as challengers | Features → residual → corrected temp | Scalar corrected point prediction |
| Probabilistic Gaussian residual model | Same MOS residual dataset | Residual distribution | Two-head NN predicts (mu, sigma); NLL and/or CRPS-centric training | Features → (mu_resid, sigma_resid), add MOS for mean | Gaussian distribution params (+ point mean) |
| Multi-seed ensemble | Same base dataset repeated with different seeds | Same as base member model | Train multiple seeds and average outputs | Features → per-seed outputs → aggregate | Ensemble mean (point and optionally sigma artifacts) |
| Wind-gated attention (implemented path) | Station tensors + metadata + global context | Delta-T and/or Gaussian params | Shared station encoder + wind-biased attention + masking | Structured tensors → attention-pooled prediction | Point or Gaussian output |
| Synthesis/meta model (implemented path) | Station model outputs + NWP + derived disagreement + seasonal features | Final distribution proxy | MLP meta-learner with Gaussian or quantile head | Combined feature vector → meta-output | Gaussian params or quantile vector |
| Market proxy family | Historical NYC TMAX history | Day-ahead (mu, sigma) and bracket probabilities | Climatology + lag regression + monthly sigma + CDF bucketization | Recent observed temps + date + bucket bounds | Bracket probabilities |

---

## 3) Why performance may have plateaued

With fixed data, three issues become dominant:
1. **Distribution mis-specification** (single Gaussian struggles on regime-mix days).
2. **Calibration drift** (good average CRPS but poor tail/bucket reliability by season).
3. **Model variance / instability** (seed sensitivity and feature overfit).

Accordingly, high-ROI changes should focus on:
- better uncertainty parameterization,
- better post-hoc calibration,
- smarter ensembling,
- tighter evaluation design.

---

## 4) Proposed experiment program (phased)

## Phase A — Calibration-first upgrades (low code risk, high expected ROI)

### A1. Add explicit calibration split + post-hoc CDF calibration
- **Change:** Reserve a dedicated calibration window (chronological, after model fit window, before final test/oos).
- **Method:** Isotonic calibration on CDF/PIT space, optionally stratified by season.
- **Why:** Often gives immediate reliability and bucket probability gains without changing base model complexity.

**Primary metrics:** CRPS, coverage error (50/90/95), PIT uniformity diagnostics, bucket Brier.

### A2. Regime-conditional calibration
- **Change:** Fit separate calibrators by season or simple regime bins (e.g., abs(MOS-station gap)).
- **Why:** Avoids one-size-fits-all sigma corrections.

**Guardrail:** Minimum sample threshold per bin; back off to global calibrator when sample is insufficient.

---

## Phase B — Distribution head improvements (moderate code risk)

### B1. Two-component Gaussian mixture head (MDN-lite)
- **Change:** Replace single Gaussian output with 2-component mixture `(w1, mu1, sigma1, mu2, sigma2)`.
- **Why:** Captures mild bimodality/front uncertainty better than inflating one sigma.
- **Constraint handling:** Keep parameter count small; strong regularization and entropy floor on weights.

**Primary metrics:** Holdout CRPS and tail calibration.

### B2. Quantile-head alternative (leveraging synthesis pathway style)
- **Change:** Train quantile output model for a fixed quantile set, with monotonicity enforcement/post-processing.
- **Why:** Robust nonparametric uncertainty without assuming Gaussian shape.

**Primary metrics:** Weighted pinball loss + interval coverage + bucket scoring.

---

## Phase C — Ensemble strategy improvements (moderate code risk)

### C1. Performance-weighted ensemble (with shrinkage)
- **Change:** Replace equal-weight seed averaging with weights from rolling validation CRPS/NLL.
- **Method:** `w_i ∝ exp(-score_i / tau)`, then shrink toward equal weights.
- **Why:** Preserves ensemble stability while exploiting consistently stronger members.

### C2. Explicit aleatoric + epistemic decomposition
- **Change:** Report/output:
  - `sigma_aleatoric` from model head,
  - `sigma_epistemic` from across-seed spread,
  - `sigma_total = sqrt(sigma_aleatoric^2 + sigma_epistemic^2)`.
- **Why:** Reduces overconfidence; improves downstream bucketization robustness.

---

## Phase D — Feature/control tuning for fixed-data regimes (low-to-moderate code risk)

### D1. Stronger regularization, not feature explosion
- Increase weight decay and structured dropout before introducing many new predictors.
- Use grouped dropout by feature families (station blocks, MOS blocks, date block).

### D2. Stability-driven feature selection
- Perform permutation-importance stability checks across years/seasons.
- Remove features with unstable sign/importance and marginal gain.

### D3. Optional compact physics composites (small additions only)
- Add a few high-signal composites (e.g., upwind-downwind spread, station gradient proxy) only if they beat baseline in ablation.

---

## 5) Detailed experiment matrix (what to run)

| ID | Variant | Changes | Expected benefit | Risk |
|---|---|---|---|---|
| E0 | Baseline reproduction | Re-run current best probabilistic + 5-seed ensemble | Reference anchor | Low |
| E1 | Global isotonic calibration | Post-hoc CDF calibration on dedicated calibration split | Better reliability, bucket quality | Low |
| E2 | Seasonal calibrators | Separate calibrators by season | Better seasonal tails | Low-Med |
| E3 | Weighted ensemble | CRPS/NLL-based seed weights + shrinkage | Better OOS stability | Med |
| E4 | Uncertainty decomposition | Aleatoric + epistemic combined sigma | Better uncertainty realism | Med |
| E5 | 2-component Gaussian | MDN-lite output head | Better multimodal days | Med-High |
| E6 | Quantile model | Quantile head + monotonic correction | Distribution robustness | Med-High |
| E7 | Regularization sweep | Weight decay/dropout/group dropout sweep | Better generalization | Low-Med |
| E8 | Feature pruning sweep | Remove unstable low-value features | Lower variance | Low-Med |

---

## 6) Evaluation protocol (time-safe and trading-relevant)

### Splits
- Keep strict chronological splits.
- Add a dedicated **calibration split** not used for fitting base model weights.
- Preserve a final holdout/OOS period for honest comparison.

### Metrics to report for every experiment
1. **Distribution quality:** CRPS, NLL.
2. **Calibration:** PIT diagnostics, reliability curves, 50/90/95 coverage and mean interval width.
3. **Point diagnostics:** MAE/RMSE/R² (secondary).
4. **Bucket diagnostics:** Brier/log score by contract bucket; sum-to-one and monotonicity checks.
5. **Stability:** season-sliced metrics and seed variance.

### Decision rule
Promote a model only if it:
- beats baseline on CRPS **and** calibration error,
- does not degrade materially on point MAE,
- improves bucket diagnostics on holdout.

---

## 7) Practical implementation notes

1. **Keep models small.** Data is limited; parameter efficiency matters more than architectural novelty.
2. **Calibrate after ensembling** (primary) and optionally compare to calibrate-then-ensemble.
3. **Use conservative assumptions** when translating probabilities to EV decisions.
4. **Log artifacts per run:** model config, seed, split dates, calibration mapping, metrics by season.

---

## 8) Recommended execution order (fastest ROI first)

1. Reproduce baseline (E0).  
2. Add global + seasonal calibration (E1/E2).  
3. Add weighted ensemble + uncertainty decomposition (E3/E4).  
4. Run regularization + pruning sweeps (E7/E8).  
5. Only then test richer distribution heads (E5/E6).

This order minimizes engineering risk while maximizing the chance of immediate performance gains.

---

## 9) Go / no-go checklist for promotion

A candidate is **Go** only if all are true:
- [ ] CRPS improves on holdout.
- [ ] Coverage error improves (especially 90/95 intervals).
- [ ] Bucket Brier/log score improves.
- [ ] No material instability across seasons.
- [ ] No leakage/time-safety violations.
- [ ] End-to-end runtime remains operationally practical.

If any fail, mark **No-Go** and archive the experiment with findings.

---

## 10) Final recommendation

Under a fixed-data constraint, the best expected return is:
1) **calibration and evaluation redesign**,  
2) **ensemble weighting and uncertainty decomposition**,  
3) **regularization/stability tuning**,  
4) then only selective distribution-head complexity.

This path typically yields better real-world probability quality and more reliable downstream decision-making than simply increasing network depth or feature count.


---

## 11) Implementation run results (executed)

Implemented and executed all experiments E0-E8 via `scripts/probabilistic_ensemble_experiments_v2.py`.
Outputs are stored under `results/probabilistic_ensemble_experiments/`.

### Test-period benchmark summary (2024 holdout)

| Experiment | CRPS | NLL | MAE | Bucket Brier | Bucket Log |
|---|---:|---:|---:|---:|---:|
| E8 Feature pruning sweep (best) | 2.579 | 3.186 | 2.204 | 0.0595 | 0.9942 |
| E3 Weighted ensemble + E4 uncertainty | 2.939 | 3.368 | 2.292 | 0.0667 | 1.1362 |
| E7 Regularization sweep | 2.959 | 3.358 | 2.465 | 0.0663 | 1.1333 |
| E0 Baseline 5-seed ensemble | 3.038 | 3.409 | 2.315 | 0.0683 | 1.1706 |
| E2 Seasonal calibration | 3.038 | 3.409 | 2.315 | 0.0347 | 0.5895 |
| E1 Global isotonic calibration | 3.038 | 3.409 | 2.315 | 0.0331 | 0.5417 |
| E6 Quantile model | 3.605 | 3.341 | 4.970 | 0.0676 | 1.1515 |
| E5 2-component Gaussian mixture | 8.446 | 4.185 | 11.831 | 0.0962 | 1.9373 |

### Key outcomes

- Best CRPS experiment: **E8 Feature pruning sweep**.
- CRPS improvement vs E0 baseline: **15.11%**.
- Calibration experiments (E1/E2) gave the strongest bucket-score gains.
- E5/E6 richer distribution heads underperformed in this fixed-data setup and are currently No-Go.

See `results/probabilistic_ensemble_experiments/summary.csv` and `benchmark_results.json` for exact machine-readable outputs.
