# E0–E8 Benchmark Review: How to Beat Kalshi PreSettlement Brier (Without More Training Rows)

## What the current benchmark is telling us

From the E0–E8 summary, every best-model-derived variant beats NWS, but none beats Kalshi pre-settlement overall or OOS in 2025. The top model (`E3_weighted_ensemble_E4_uncertainty`) is:

- Model overall Brier: **0.1333**
- Kalshi pre-settlement overall Brier: **0.1271**
- Model OOS Brier: **0.1300**
- Kalshi pre-settlement OOS Brier: **0.0988**

This means we are still materially behind market consensus where it matters most (forward OOS). The gap is especially large in OOS spring and in tail directions (`above`, `below`), suggesting distribution-shape and calibration misspecification, not just mean error.

## Why the E0–E8 family likely plateaued

The benchmark script applies lightweight transformations around a single Gaussian base (`mu`, `sigma`) and mostly tweaks offsets/sigma multipliers/calibration maps; it does not materially change information content or distribution family complexity.

Specific limitations in the current setup:

1. **Calibration window is fixed to 2023 only**, then evaluated through 2025; this is vulnerable to drift and regime shift.
2. **Most variants are post-hoc parameter nudges** (offsets, sigma scaling), which typically cannot close a large OOS market-information gap.
3. **Bucketization is Gaussian for most variants**; this can be mis-specified around fronts/transition regimes where tails or bimodality matter.
4. **Trading sim shows persistent negative P&L across thresholds**, so even when model beats NWS, edge quality is not robust after fees.

## Improvement strategy (ordered, overfitting-aware)

### 1) Rebuild calibration as the primary lever (highest ROI, lowest overfit risk)

If training-row count is constrained, improve **probability calibration** first before adding more features.

- Replace one-shot 2023 calibration with a **rolling-origin calibration** design:
  - Example: for each month in 2025 OOS eval, fit calibrator on prior 6–12 months only.
  - Keep calibrator simple: isotonic or beta calibration, per season if sample size supports it.
- Add **direction-conditional calibration** (`below`, `between`, `above`) with shrinkage toward global calibration when sample is sparse.
- Calibrate **CDF levels**, then derive bucket masses from calibrated CDF differences (not direct bucket-only map).

Why this is likely to help: current reliability metrics show nontrivial calibration error and high-probability bins with low counts/instability, so better temporal calibration hygiene can improve Brier without expanding feature dimensionality.

### 2) Upgrade output distribution family without increasing raw feature count

Keep feature set stable; increase expressivity in output head only.

- Move from single Gaussian to one of:
  - **2-component MDN** (shared trunk, tiny head).
  - **Monotone quantile head** with ~9–15 quantiles + monotonicity penalty.
- Train by **CRPS/NLL**, select by OOS Brier + calibration diagnostics.
- Constrain parameters aggressively (small hidden layers, dropout/L2, early stopping).

This directly targets bucket-shape errors while limiting overfitting risk versus adding many new predictors.

### 3) Add a tiny, regularized meta-learner that ingests market information

Kalshi pre-settlement is outperforming model OOS by a wide margin; the market carries extra information.

- Build a small synthesis model with inputs:
  - model distribution summary (`mu`, `sigma`, selected quantiles),
  - NWS summary,
  - market state (`presettlement_prob`, bid/ask spread, depth proxy if available).
- Output calibrated bucket probabilities.
- Use **strict time-safe splits** and **heavy regularization** (ridge / shallow MLP).

This is effectively “learn when to trust us vs market” and should be one of the highest-probability paths to beating pre-settlement Brier.

### 4) Reduce variance via feature grouping/priors, not feature explosion

Given limited rows, avoid adding many raw stations/features.

- Replace raw high-dimensional inputs with low-variance composites:
  - sector means/gradients,
  - persistence deltas,
  - wind-conditioned upwind/downwind aggregates,
  - pressure tendency indicators.
- Prefer **group-lasso / ridge** or small-network with grouped dropout.
- Keep a hard cap on effective dimensionality (e.g., <= 20–30 strong engineered features).

### 5) Optimize to the true objective: bucket Brier and calibration, not point MAE

- Tune model checkpoints by:
  1. OOS bucket Brier,
  2. calibration (ECE/reliability/PIT),
  3. log score / CRPS.
- Report by regime slice (season, direction, spread/volatility bins).
- Reject models that improve mean temp error but worsen bucket calibration.

### 6) Tighten market benchmark fairness and data leakage checks

Ensure benchmark comparison is apples-to-apples:

- Verify contract boundary rules and bucket inclusivity are exactly mirrored.
- Ensure no accidental lookahead in joined data and calibration windows.
- Add significance testing (paired/block bootstrap) for model-vs-market Brier deltas.

### 7) Trading policy: edge should be EV-after-cost and uncertainty-aware

Current thresholded edge trading is negative at all thresholds; convert to EV gating:

- Trade only when:
  - expected value after fees/slippage > 0,
  - and lower confidence bound on EV > 0 (conservative).
- Size with fractional Kelly capped by daily exposure and adjacent-bucket correlation.
- Add kill switches for calibration drift and data integrity failures.

## Practical experiment ladder (next 4 experiments)

1. **C1 Rolling calibration only** (no model retrain):
   - global vs seasonal vs direction-conditional calibrators.
2. **D1 Distribution head swap**:
   - baseline trunk + MDN2 or quantile head; no new features.
3. **S1 Small synthesis with market features**:
   - ridge and 1-hidden-layer MLP variants.
4. **T1 EV-aware execution policy**:
   - no threshold-only rules; include fees + slippage + uncertainty buffer.

Promote only if each step improves **OOS pre-settlement Brier** (and not just IS).

## Anti-overfitting guardrails (must enforce)

- Chronological splits only, with final untouched holdout.
- Distinct train / calibration / test windows.
- Hyperparameter budget limits and early stopping.
- Parsimony rule: if two models are statistically tied, keep the simpler one.
- Track parameter count and effective feature count alongside performance.

## Success criteria to claim “market-beating”

- Statistically significant improvement vs Kalshi pre-settlement on OOS bucket Brier.
- Stable or improved reliability/ECE across seasons and directions.
- Positive EV after conservative costs in paper-trade simulation.
- No degradation in operational robustness (time-safe inputs by cutoff).
