# Phase 4 Enhancements Playbook (NYC Daily TMAX Prediction)

This playbook translates the Phase 4 “Enhancements” stage into a concrete, testable sequence of experiments designed to **reduce MAE** while keeping the workflow reproducible and the results interpretable.

**Phase 4 scope (from project plan):**
- **4.1 Multi-feature inputs** (add TMIN, diurnal range, cyclical date; optionally include NYC(t−1))
- **4.2 Sensitivity experiments** (vary station count, radius, weather-type bins)
- **4.3 LSTM / advanced model** (sequence modeling; possible station embeddings + attention)

---

## 0) Success Criteria and Guardrails

### Primary objective
- Reduce **MAE** on the held-out test set versus the best Phase 3 baseline.

### Secondary objectives
- Reduce MAE especially on **high-error days** (fronts/advection/rapid changes).
- Maintain stable performance across seasons (winter vs summer).
- Avoid brittle gains (overfit to specific station sets or time periods).

### Guardrails
- Keep a **frozen evaluation protocol**: same train/val/test split logic, same leakage rules.
- Always compare against:
  1) Persistence baseline (NYC(t−1))
  2) Best Phase 3 model
  3) “Simple regression” baseline (ridge/elastic net) for sanity

---

## 1) The Highest-Leverage Change: Predict **ΔTMAX** Instead of TMAX

### What to do
Switch the target from predicting NYC **TMAX(t)** to predicting the daily change:

\[
\Delta T(t) = T_{NYC}(t) - T_{NYC}(t-1)
\]

Then reconstruct:
\[
\hat{T}_{NYC}(t) = T_{NYC}(t-1) + \widehat{\Delta T}(t)
\]

### Why it reduces MAE
- Persistence is strong; deltas force the model to learn **the hard part** (regime shifts, advection events) rather than re-learning climatology and persistence.
- Empirically, delta targets often reduce MAE by stabilizing training and narrowing the output range.

### Implementation checklist
- [ ] Create target `y = TNYC[t] - TNYC[t-1]`
- [ ] Include `TNYC[t-1]` as an input feature (for reconstruction and stabilization)
- [ ] Train model to predict `ΔT`
- [ ] Reconstruct `TNYC_pred = TNYC_lag1 + ΔT_pred`
- [ ] Evaluate MAE on reconstructed `TNYC_pred` (not on `ΔT`)

### Training loss recommendation
- Start with **Huber (SmoothL1)** or **MAE** loss (often better for MAE optimization than MSE).
- Keep a tracked metric of MAE for early stopping.

---

## 2) Add Physics-Shaped Inputs Before Adding Model Complexity

Your Phase 4.1 adds: TMIN, diurnal range, cyclical date, and optionally NYC(t−1).  
This section adds **low-cost, high-signal features** that encode fronts and coastal gradients.

### 2.1 Sector averages and gradients (very high value)
**Concept:** Group stations into sectors (W/NW, SW, coastal/E/SE, N/NE, near-field) and feed:
- sector mean temperature(s) at t−1 (and t−2)
- sector gradients (differences between sectors)

**Examples**
- `mean_WNW(t-1)`, `mean_SW(t-1)`, `mean_coast(t-1)`
- `grad_upstream_vs_coast = mean_WNW(t-1) - mean_coast(t-1)`
- `grad_SW_vs_NW = mean_SW(t-1) - mean_WNW(t-1)`

**Why it helps**
- Gradients proxy “which air mass is arriving,” especially in coastal complex terrain.
- Reduces dimensionality and increases signal-to-noise when you add many stations.

### 2.2 Trend features (front-timing proxy)
Compute short differences (per sector or per top stations):
- `Δ1 = T(t-1) - T(t-2)`
- `Δ2 = T(t-2) - T(t-3)`

These help detect *direction and momentum* of change.

### 2.3 Static station metadata
Add simple station metadata (from NOAA station metadata):
- Elevation
- Distance to NYC (km)
- Bearing from NYC (degrees)
- Optional: station type flag (airport vs other)

Use these as station-level inputs for attention/embedding models, or as features for aggregated sector models.

---

## 3) Add Stations the Right Way: +50 Station Expansion Plan

The goal is to test whether expanding the station set improves performance, **without destroying sample size** or learning unstable weights.

### 3.1 Selection strategy: rings × sectors
Build the additional ~50 stations as a structured pool:

- **Near field (0–50 mi):** NYC metro, coastal moderation, urban effects
- **Mid field (50–150 mi):** strongest “upstream yesterday → NYC today” signal
- **Far field (150–250 mi):** air-mass source and frontal structure

Ensure sector coverage:
- W/NW (primary upstream)
- SW
- N/NE
- E/SE/coastal

### 3.2 Missingness strategy (critical)
Do **not** drop rows whenever any station is missing—this will collapse your dataset.

Preferred options:
1) **Impute + missingness mask**
   - Impute missing station-day values using:
     - a short rolling mean, OR
     - day-of-year climatology per station
   - Add a binary feature `is_missing` for each imputed value
   - (Best) Use attention pooling with station masking

2) **Variable station count per day (attention-friendly)**
   - For each day, pass only available station inputs
   - Pool with attention over the set of available stations

### 3.3 Sensitivity grid (Phase 4.2)
Run controlled experiments varying station count and radius:

- **Station count K:** {20, 30, 40, 50, 70}
- **Radius:** {150, 200, 250} (optional 300 if coverage requires)

Always report:
- overall MAE
- winter MAE vs summer MAE
- MAE on “high-gradient days” (e.g., upstream-vs-coast gradient above a percentile threshold)

---

## 4) Architecture Upgrades: MLP → Attention Pooling → Temporal Model

### 4.1 Baseline MLP (keep for benchmarking)
- Flattened station features + engineered gradients + cyclical day features
- Strong regularization (weight decay, dropout)
- Early stopping on validation MAE

### 4.2 Station embeddings + attention pooling (recommended when adding 50 stations)
**Motivation:** With many correlated stations, flattened MLPs can learn fragile weights.
Attention pooling lets the model learn “which stations matter today.”

**Template architecture**
1) Per-station encoder (shared weights):
   - Inputs per station: TMAX, TMIN, diurnal, trend, plus station metadata
   - Small MLP → station embedding
2) Attention pooling across station embeddings:
   - Mask missing stations
3) Optional: small temporal block (see 4.3)
4) Output: predict ΔT

**Practical notes**
- Use a modest embedding size (e.g., 16–64)
- Add layer norm to stabilize attention
- Keep the model small enough to avoid overfitting

### 4.3 Temporal model (only if it empirically helps)
Rather than jumping straight to an LSTM for everything, test these in order:

1) **k-day window with MLP**
   - Concatenate features from [t−1, …, t−k]
   - Often surprisingly strong with engineered features

2) **1D temporal convolution over k days**
   - Lightweight and stable

3) **GRU/LSTM**
   - Use only if k-window models plateau
   - Predict ΔT with sequences of station-pooled features

---

## 5) Residual Learning / Stacking Ensemble (Low Risk, Often High Return)

**Core idea:** Let a simple model handle the “easy majority,” and let a NN learn the residual.

### Base models (fast priors)
- Persistence: `TNYC(t-1)`
- Ridge/elastic net on station features
- A simple distance-weighted upstream composite (optional)

### Meta model (residual corrector)
Train a small NN on:
- base model predictions
- engineered gradients, trends
- cyclical day features
Target:
- either **TMAX residual** or **ΔT residual**

This approach frequently reduces MAE by improving robustness, especially when the NN is otherwise tempted to overfit.

---

## 6) Season/Regime Specialization Without Overkill

If summer MAE stays high, try **two-expert specialization**:

### Option A: Two seasonal models
- Cool season model: (DJF + MAM)
- Warm season model: (JJA + SON)

### Option B: Mixture-of-experts gate (single unified training)
- A small gating network uses day-of-year sin/cos + a couple gradients
- Gate blends two small expert models

Report:
- season-wise MAE
- “transition months” performance (Apr/May, Sep/Oct)

---

## 7) Training Details That Move MAE

### Loss and metrics
- Optimize for MAE using **Huber** or **MAE** loss
- Track MAE on validation for early stopping
- Consider a small weight on extreme changes if tails dominate errors

### Regularization
- Weight decay (L2)
- Dropout (modest; avoid huge dropout which can destabilize)
- Early stopping with patience

### Normalization
- Standardize station features using training-set statistics
- If mixing stations with different climates, standardize per-station using its own training history (optional) and provide station metadata to retain absolute context

### Learning-rate strategy
- Warmup + cosine decay, or ReduceLROnPlateau on validation MAE

---

## 8) Recommended Execution Order (Expected MAE Impact)

Run the following in order, stopping when gains plateau:

1) **ΔT target + include NYC(t−1)**; Huber loss  
2) Add **sector gradients + trend features**  
3) Add **+50 stations** with masking/imputation and rerun sensitivity grid  
4) Upgrade MLP → **station embedding + attention pooling**  
5) Add temporal modeling only if needed (k-window → conv → GRU/LSTM)  
6) Add **stacking/residual correction**  
7) If summer remains stubborn: **two-expert seasonal / mixture-of-experts**  

---

## 9) Reporting Template (Copy/Paste)

For each experiment, record:

- **Model:** (MLP / attention / GRU / stacking)
- **Target:** (TMAX or ΔT)
- **Inputs:** (TMAX, TMIN, diurnal, gradients, trends, cyclical, metadata)
- **Stations:** K, radius, selection method, missingness handling
- **Train setup:** loss, lr schedule, batch size, epochs, early stopping
- **Overall MAE:** ___
- **Winter MAE:** ___
- **Summer MAE:** ___
- **High-gradient days MAE:** ___
- **Notes:** failure modes, stability, outlier behavior

---

## 10) Minimal “Must-Do” Checklist (If You Only Do Three Things)

- [ ] Switch to **ΔT target** (learn changes, reconstruct TMAX)
- [ ] Add **sector gradients + trends** (front/advection proxies)
- [ ] Add stations using **masking/imputation** + (ideally) **attention pooling**

These three changes typically yield the largest MAE reductions with the lowest risk of overfitting.

