# NYC Temperature Prediction Project Plan (Operational + Trading)

**Last updated:** 2026-02-13  
**Scope:** End-to-end probabilistic forecast, calibration, contract bucketization, and EV-gated Kalshi execution simulation.

## 1) Mission

Build a calibrated daily distribution for NYC (Central Park) max temperature, map that distribution to KXHIGHNY contract buckets, compare against market-implied probabilities, and only trade when expected value is positive after fees/spread/slippage assumptions.

## 2) Current state (implemented)

### Contract + benchmark alignment
- Contract benchmarking now centers on KXHIGHNY bucket definitions and pre-settlement prices.
- Main benchmark track includes E0–E22 model variants and unified U-series synthesis variants.
- Summary artifacts are produced with overall + OOS Brier and trading P&L diagnostics.

### Forecasting + calibration stack
- Canonical best-model temperature forecasts are used as the base distribution stream.
- Post-processing variants include isotonic, seasonal calibration, conditional calibration, neural/stacker synthesis, and Platt+isotonic hybrids.
- WGA/WGA-v2 and unified synthesis experiments are integrated into benchmark comparisons.

### Trading evaluation
- EV-gated strategy simulations are in place with fees and spread-aware assumptions.
- Paper-trading gate reports are generated for key model families.
- Seasonal and OOS slices are available for risk-aware diagnostics.

## 3) Immediate gaps (next priorities)

1. Harden strict cutoff-time feature provenance tracking for all live features.
2. Add explicit automated kill-switch checks in daily orchestration (missing data, schema drift, calibration drift).
3. Expand execution realism (depth/queue/fill uncertainty) in trading simulation.
4. Standardize a single production candidate from E/U model families with explicit promotion criteria.

## 4) Layered plan (current architecture)

### Layer A — Ingestion
- Operational ingestion: station observations, MOS/NWS proxies, Kalshi market snapshots.
- Training-only archives remain separate and must never leak into live-time features.
- Keep source/cutoff metadata attached to each feature group.

### Layer B — Feature engineering
- Continue lag-safe persistence, seasonal harmonics, and regime proxies.
- Preserve missingness handling and deterministic transformations.
- Maintain compatibility between training and live-time feature definitions.

### Layer C — Modeling
- Base model stream: best-model artifacts and learned variants (E-series).
- Supplemental stream: WGA/WGA-v2 and unified synthesis (U-series).
- Selection objective remains proper probabilistic scoring (Brier/CRPS/NLL) with OOS emphasis.

### Layer D — Calibration + bucketization
- Calibrate with held-out chronology-safe windows.
- Convert calibrated CDF/distribution outputs to exact contract bucket probabilities.
- Enforce probability mass checks and reliability diagnostics each run.

### Layer E — Trading + risk
- Use EV thresholding that includes fees + conservative execution costs.
- Apply capped/fractional Kelly and per-day exposure limits.
- Halt on data-quality failures, unavailable critical inputs, or calibration drift.

## 5) Success gates (must pass before live trading)

### Forecast gate
- OOS Brier beats NWS and remains stable by season/regime.

### Calibration gate
- Reliability/ECE and bucket mass checks remain within configured tolerances.

### Trading gate
- Conservative-simulation returns and drawdown profile pass pre-defined risk constraints.

### Operations gate
- Daily run completes by cutoff with auditable logs, artifacts, and no critical data validation failures.

## 6) Active documentation and governance updates

- Legacy runner moved to `ARCHIVE/legacy_runners/run_kalshi_real_backtest.py`.
- New docs added under `docs/`:
  1. `current_state_and_directory.md`
  2. `top15_models_brier_function_reference.md`
  3. `model_principles_and_us_city_portability.md`
- `.claude/rules/MEMORY.md` is now synchronized to current benchmark/model state.
