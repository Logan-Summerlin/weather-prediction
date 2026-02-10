# NYC Temperature Prediction Project Plan (Operational + Trading)

**Last updated:** 2026-02-10  
**Scope:** End-to-end probabilistic forecast and Kalshi bucket decision pipeline.

## 1) Mission

Build a calibrated daily distribution for NYC max temperature, convert it to Kalshi bucket probabilities, and trade only when EV remains positive after realistic costs.

## 2) What is implemented now

### Data + benchmark integration
- Kalshi settled + pre-settlement data pipeline is wired into benchmark evaluation.
- NWS probability benchmark is available via MOS-to-distribution conversion.
- Best-model prediction exports are the benchmark default inputs.

### Forecast evaluation
- Bucket-level Brier/log scoring by overall, period, season, and bucket direction.
- Reliability output with ECE and per-bin observed rates.
- Probability clipping for numeric stability.

### Trading evaluation
- Edge-based YES/NO strategy against pre-settlement prices.
- Fees included (7% on winnings).
- **Bid/ask crossing execution costs included** (ask for YES, 1-bid for NO).
- OOS bootstrap confidence intervals for net P&L.

## 3) What remains to complete

1. Strict train/infer parity on cutoff-safe operational features.
2. Post-hoc calibration layer (isotonic / regime-conditional) tied to held-out calibration split.
3. Non-Gaussian distribution option (quantile or mixture) with CRPS-first selection.
4. Live-ready risk controls: exposure caps, drawdown limits, kill switches.

## 4) Technical plan by layer

### Layer A — Ingestion
- Preserve split between operational sources and training-only archives.
- Add explicit source registry with cutoff availability flags.

### Layer B — Feature engineering
- Keep lag-safe persistence/regime features.
- Expand robust physical composites (wind-conditioned, pressure tendency, moisture/cloud).

### Layer C — Modeling
- Continue best-model ensemble baseline.
- Add alternative probabilistic heads and compare by CRPS + calibration, not MAE alone.

### Layer D — Calibration + bucketization
- Calibrate CDF on held-out calibration set.
- Validate per-date bucket probability mass and monotonicity.

### Layer E — Trading + risk
- Use spread/slippage-aware EV.
- Apply fractional Kelly with hard caps and daily halts.

## 5) Success gates

### Forecast gate
- OOS model Brier remains better than NWS and stable by season.

### Calibration gate
- PIT/reliability and interval coverage within acceptable tolerances.

### Trading gate
- Positive simulated returns under conservative execution assumptions and uncertainty bounds.

### Operations gate
- Daily run completes with full audit logs and no critical data-quality failures.
