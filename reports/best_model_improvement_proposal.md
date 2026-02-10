# Proposal: Improving the Best NYC Tmax Forecast Model

## 1) What the current results say

### Strengths
- MOS integration is the dominant gain driver: station-only models plateau near ~4.0 MAE, while MOS+station models are near ~2.1 MAE.
- The most robust current model family is residual correction (predict `actual - MOS`) with relatively small NNs.
- Generalization stability is best for `C_Correction_NN_tiny` (test 2.090 vs OOS 2.093, essentially no degradation).

### Gaps still limiting performance
- The best test-only score (`E_warm_Ridge`, 1.959) degrades materially OOS (2.244), implying regime-specific overfit.
- Spring/winter remain materially harder than summer/fall in the top hybrid model.
- Current winner is still mostly a **point forecast** stack; trading needs calibrated **distribution** quality first, then bucket EV.
- Architecture sweeps show little gain from deeper/wider plain MLPs, so additional depth alone is unlikely to help.

## 2) Diagnosis (why we are stuck around ~2.05–2.20 OOS)

1. **Model objective mismatch**
   - Most experiments optimize point error (MAE/Huber), while the downstream objective is calibrated bucket probabilities.
   - This can produce good MAE but weak tails / miscalibrated bucket mass, especially near front-driven transitions.

2. **Residual model is right; residual distribution is under-modeled**
   - Residual correction works because MOS captures large-scale signal.
   - But residuals are likely heteroscedastic and sometimes bimodal (on mixed advection / marine influence days).

3. **Seasonality/regime not explicitly parameterized enough**
   - Warm-only specialists are excellent in-season but brittle out-of-regime.
   - A single global residual head underfits regime-dependent error structure.

4. **Station signal aggregation is still too static**
   - Existing flat feature stacks cannot fully adapt weights based on synoptic flow direction and stability.
   - The repo already has a wind-gated attention architecture that has not been integrated into the MOS-residual best path.

## 3) Architecture upgrade proposal (priority order)

## A. Replace point residual head with probabilistic residual head (highest ROI)
**Current:** `y = MOS + f_theta(X)` (point)

**Proposed:**
- `resid ~ N(mu(X), sigma(X))` (heteroscedastic Gaussian residual)
- Prediction distribution: `Tmax ~ N(MOS + mu, sigma)`
- Train with NLL (or CRPS approximation), not MAE.

Why first:
- Minimal pipeline disruption.
- Directly supports calibration + bucketization.
- Should improve spring/winter where residual variance is larger.

Implementation notes:
- Start from `C_Correction_NN_tiny` backbone.
- Two output heads (`mu`, `log_sigma`) with `sigma = softplus(log_sigma)+eps`.
- Clip sigma floor (e.g., 0.75F) and cap (e.g., 10F) for stability.

## B. Add mixture residual head for multi-regime days (second ROI)
Use a 2-component Gaussian mixture residual model:
- Outputs: `pi1, mu1, sigma1, mu2, sigma2`
- Final distribution: `MOS + mixture(resid)`

Why:
- Handles bimodality around frontal timing uncertainty where single Gaussian is too narrow/biased.
- Especially useful for shoulder seasons.

Guardrails:
- Keep tiny backbone; only enlarge output head.
- Penalize component collapse (entropy regularization on mixture weights).

## C. Integrate wind-gated station attention into MOS residual pathway
Build a synthesis model:
- Inputs: station tensor + station metadata + global context + MOS features.
- Base path: MOS forecast.
- Learned correction path: wind-gated attention residual.

Why:
- Gives dynamic upwind/downwind weighting by flow regime.
- Uses architecture already implemented in repo (`src/wind_gated_attention.py`).

Minimal variant:
- Attention output -> small MLP -> residual distribution heads.
- Fallback mode when station sparsity high: downweight correction and rely on MOS.

## D. Regime-aware multi-head residual model (instead of separate seasonal models)
Single shared encoder + 3 heads:
- head 1: warm stable regime
- head 2: cold advection regime
- head 3: transition/front regime

Gate via small classifier using only time-safe features (MOS spread, pressure tendency proxy, wind shift, DOY sin/cos).

Why:
- Better than hard season splits; avoids abrupt month-boundary behavior.

## 4) Parameter tuning proposal (targeted, not brute-force)

Use a constrained Bayesian search (~60–100 trials) over the **best two** architectures only.

### Backbone / optimization
- Hidden sizes: `[64,32]`, `[96,48]`, `[128,64]`
- Dropout: `{0.0, 0.05}` (strong prior on 0.0)
- Weight decay: `1e-6 ... 3e-4` (log-uniform)
- LR: `2e-4 ... 2e-3`
- Batch size: `{128, 256}`
- Scheduler: cosine decay with warmup **or** plateau (compare once)
- Gradient clip: `{2.0, 5.0, 10.0}`

### Loss / objective
- Gaussian NLL with optional robust clipping.
- CRPS proxy fine-tuning on final 15–30 epochs.
- Multi-objective early stop: weighted validation score
  - `0.5*CRPS + 0.3*MAE + 0.2*calibration_error`.

### Regularization / stability
- EMA weights for evaluation.
- Ensembling 3 seeds for final production model (small cost, better calibration stability).

## 5) Feature engineering improvements (time-safe and operational)

## A. MOS-derived uncertainty and disagreement features (low effort, high value)
Add:
- `|gfs_mos - nam_mos|`
- spread ratio vs climatological spread
- recent MOS bias features (rolling residual by month/season from training history)
- interaction terms with DOY and nyc_lag1

Rationale: disagreement is a regime/uncertainty proxy and should inform sigma/head gating.

## B. Station composites instead of raw-only expansion
Keep raw lag features, but add robust composites:
- sector means/medians (W/NW, SW, coastal, near-field)
- upwind-minus-downwind temperature gradient (wind-conditioned)
- coastal moderation index (coastal minus inland)
- lagged diurnal range anomalies

Use grouped normalization and missing masks to prevent leakage via imputation artifacts.

## C. Add operational meteorology features already supported in repo modules
From operational pipelines:
- ASOS hourly aggregates (overnight wind shift, cloud ceiling fraction, dewpoint depression, pressure tendency)
- sounding-derived stability proxies
- NWP spread / uncertainty signals

These are likely most helpful in spring/winter where current errors are largest.

## 6) Station network improvements

## A. Separate training pools by objective
- **Operational model pool:** ASOS-mapped stations only (strict live parity).
- **Research augmentation pool:** non-ASOS stations only for offline diagnostics, not live features.

## B. Improve upstream coverage quality, not just count
Current config includes many stations, but some have low long-window completeness.
Focus on:
- maintaining robust W/NW + SW upstream stations with high completeness,
- reducing influence of weak/spotty stations via learned masks/attention,
- explicitly tracking per-station availability and marginal contribution.

## C. Candidate additions (if available operationally)
Prioritize missing upstream/coastal ASOS candidates within 75–180 mi that increase sector redundancy rather than density near NYC core.

## 7) Calibration and market alignment upgrades (required before live scaling)

1. Fit post-hoc calibration on held-out calibration window (not train/val):
   - isotonic on PIT/CDF levels or conformalized quantiles by season/regime.
2. Validate by:
   - PIT histogram,
   - interval coverage (50/80/90%),
   - bucket reliability at Kalshi cutpoints.
3. Convert CDF -> exact contract buckets with strict endpoint logic and sum-to-one checks.

## 8) Proposed 6-week execution plan

## Week 1–2: Probabilistic residual baseline
- Implement Gaussian residual head on `C_Correction_NN_tiny` features.
- Add CRPS/NLL evaluation and calibration report artifacts.
- Deliverable: calibrated probabilistic baseline with bucket metrics.

## Week 3–4: Mixture + regime gating
- Add 2-component residual mixture.
- Add lightweight regime gate (shared encoder, multi-head residual).
- Deliverable: shoulder-season MAE/CRPS and reliability improvement vs Week 1 model.

## Week 5: Wind-gated attention synthesis
- Integrate `WindGatedAttentionModel` as correction module with MOS base.
- Add missingness-aware fallback behavior.
- Deliverable: ablation showing when attention beats flat residual model.

## Week 6: Trading-readiness hardening
- Final calibration freeze, bucket EV evaluation with conservative slippage.
- Risk constraints + kill-switch thresholds tied to calibration drift.
- Paper-trading checklist signoff.

## 9) Acceptance criteria for “new best model”

A candidate replaces current best only if it beats it on **all**:
1. OOS MAE improvement >= 0.08F over `C_Correction_NN_tiny`.
2. OOS CRPS improvement >= 5%.
3. Bucket reliability improvement (ECE/Brier) on holdout and 2025 OOS.
4. No degradation worse than 0.05F in any season slice.
5. Stable across >=3 random seeds and no calibration drift alerts in replay.

---

## Bottom line
The path to a materially better model is **not** a larger generic NN. The highest-probability win is: **probabilistic MOS-residual modeling + regime-aware dynamics + wind-aware station weighting + strict post-hoc calibration**. This aligns model optimization with trading objectives and targets the exact weaknesses visible in current results.
