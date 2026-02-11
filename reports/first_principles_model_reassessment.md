# First-Principles Model Reassessment (Beginner-Friendly)

**Date:** 2026-02-11

## Why this document exists

You asked whether we may have over-committed to one architecture path while trying to beat Kalshi pre-settlement and NWS. This note steps back to first principles and reframes what matters most for prediction-market performance.

---

## 1) First principles (in plain English)

If the goal is **trading profit**, not just low forecast error, then a model must do 4 things well at the same time:

1. **Be directionally right** often enough (accuracy / Brier / CRPS).
2. **Be calibrated** (when it says 70%, the event happens near 70% over time).
3. **Produce usable bucket probabilities** that match contract settlement rules.
4. **Beat transaction costs** (fees + spread + slippage), not just midpoint prices.

A model that is “more accurate” but poorly calibrated or too weak at market edges can still lose money.

---

## 2) What the current benchmark evidence says

### Forecast quality
- Best-model stack currently beats NWS on Brier, but still trails Kalshi pre-settlement.
- This means the model is useful, but the market is still stronger overall.

### Trading quality after realistic costs
- Once bid/ask crossing is included, profitability becomes much more conservative and often negative.
- This is the most important operational reality check.

### Practical implication
- The bottleneck appears to be less about adding more hidden layers and more about:
  - calibration quality (especially tails),
  - distribution quality (Gaussian limits),
  - and execution realism.

---

## 3) What Phase-1 results suggest (architecture vs representation)

### Phase-1 architecture runs
- Architecture depth changes improved some metrics, but gains are incremental.
- Better/wider MLP variants help, but no dramatic “silver bullet” from depth alone.

### Phase-1 feature engineering runs
- Feature-combination variants (especially with Phase1A+1C + NN64x32) produced very strong OOS MAE.
- This suggests feature representation and residual framing matter a lot.

### Phase-1 combined runs
- The 5-seed ensemble and probabilistic variant are among strongest OOS performers.
- The probabilistic model also gives uncertainty outputs needed for bucketization.

### Phase-1 probabilistic runs
- Best CRPS models are not always best MAE models.
- This is expected: probabilistic quality and point error optimize different goals.

---

## 4) Model-family lessons (from archived catalog)

The historical catalog already points to an important pattern:

- Biggest gains came from **MOS residualization** (learning the correction to a strong public forecast), not from architecture complexity alone.
- Pure sequence complexity (LSTM/GRU/Conv1D) did not reliably beat simpler residual pipelines at this horizon.
- Probabilistic + ensemble methods are the strongest family so far.

So the evidence favors a **robust residual+probabilistic+calibration pipeline** over chasing ever-deeper neural architectures.

---

## 5) Are we too deep in one architecture pipeline?

**Short answer: yes, partially.**

Not because neural nets are wrong, but because market performance depends on more than raw architecture:

- Better execution realism changed conclusions.
- Tail calibration remains a gap.
- Gaussian bucketization is likely too restrictive for some weather regimes.

So the next performance jump likely comes from **distribution + calibration + execution upgrades**, not simply “bigger model.”

---

## 6) Most promising options (ranked)

### Tier 1 (highest expected return on effort)
1. **Calibration-first upgrade**
   - Stronger post-hoc calibration (global + seasonal/regime checks).
   - Reliability/PIT gates as hard deploy criteria.

2. **Non-Gaussian distribution outputs**
   - Quantile model or low-component MDN for multimodality/tails.
   - Select by CRPS + reliability, not MAE alone.

3. **EV thresholding with uncertainty buffers**
   - Trade only when edge clears fees/spread/slippage + model uncertainty margin.

### Tier 2 (likely helpful)
4. **Hybrid ensemble across model classes**
   - Blend residual NN + tree booster + probabilistic head.
   - Let ensemble reduce regime-specific failure risk.

5. **Regime-aware calibration/model routing**
   - Different calibrators or blend weights by season/synoptic regime.

### Tier 3 (longer-horizon exploration)
6. **Wind-gated attention / synthesis layer promotion**
   - Promising physically-informed path, but should “earn” mainline status with strict OOS CRPS+trading checks.

7. **Full microstructure simulator**
   - Queue/fill/depth modeling for truer EV and sizing.

---

## 7) Beginner-friendly mental model

Think of the pipeline like this:

- **Model = Weather brain** (predicts temperature distribution).
- **Calibration = Honesty filter** (makes probabilities trustworthy).
- **Bucketization = Translator** (turns weather distribution into contract probabilities).
- **Trading engine = Accountant + risk manager** (only acts when math stays positive after all costs).

Right now, the weather brain is decent. The biggest upside is improving the honesty filter + translator + accountant discipline.

---

## 8) Suggested next 30-day plan

1. Freeze a strong baseline (E3/E1 + current best data path).
2. Add quantile and MDN alternatives with same features/splits.
3. Run calibration bake-off (global isotonic vs seasonal/regime isotonic).
4. Evaluate by CRPS, Brier, PIT/reliability, and spread-aware trading outcomes.
5. Promote only if OOS improvements survive conservative execution assumptions.

---

## Final takeaway

The evidence does **not** suggest abandoning neural nets.

It suggests re-centering on:
- **residual framing**,
- **distributional quality**,
- **calibration robustness**, and
- **execution realism**.

That is the most likely path to beating benchmarks in a way that survives real market frictions.
