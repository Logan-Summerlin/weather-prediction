# Master Improvement Plan (Source of Truth)

**Date:** 2026-02-10

This is the canonical improvement plan for the NYC temperature prediction + Kalshi benchmark/trading pipeline.

## 1) Current state (validated)

### Forecasting
- Best production benchmark model family: **Phase-1 ensemble exports** in:
  - `data/best_model_predictions_2023_2024.csv`
  - `data/best_model_predictions_2025.csv`
- Current benchmark overall Brier (bucket-level):
  - Kalshi pre-settlement: **0.1271**
  - Model: **0.1335**
  - NWS: **0.1418**

### Trading simulation
- Execution costs now modeled with **bid/ask crossing**.
- Fees modeled at **7% on winnings**.
- OOS trading results include **bootstrap 95% P&L CI**.
- Annualized Sharpe is reported explicitly alongside per-trade Sharpe.

### Diagnostics now present
- Reliability table + ECE outputs.
- Per-date bucket probability mass checks:
  - `model_probability_mass_check.csv`
  - `nws_probability_mass_check.csv`

## 2) Gaps to close before live trading

1. **Calibration quality at high-probability bins remains weak** for both model/NWS tails.
2. **Distributional form is still Gaussian** (`mu`,`sigma`) for bucketization; no learned multimodality.
3. **Execution simulator is still simplified** (no queue/fill-depth model).
4. **Training/inference parity is incomplete** versus strict early-morning operational data constraints (ASOS-first live feature parity still in progress).

## 3) Priority roadmap

### Phase A — Calibration + distribution quality (highest priority)
- Add post-hoc CDF calibration (isotonic on held-out calibration window).
- Evaluate PIT, interval coverage, and seasonal reliability.
- Introduce quantile or mixture-density output path and compare CRPS/NLL.

### Phase B — Trading realism hardening
- Add explicit slippage/depth scenarios and conservative fill assumptions.
- Require edge > execution-cost budget + uncertainty buffer.
- Add policy-level kill switches for calibration drift and missing data.

### Phase C — Operational parity
- Ensure all live features are available by cutoff time.
- Separate training-only sources vs operational sources in config and feature registry.
- Add daily completeness/latency checks with hard failure behavior.

### Phase D — Regime robustness
- Target spring regime error reduction with wind/pressure/frontal proxies.
- Track seasonal performance and recalibration slices by regime.

## 4) Acceptance criteria for paper-trade promotion

Promote from benchmark-only to paper trading only when all are true:

1. Model beats NWS on OOS Brier with stable seasonal behavior.
2. Calibrated interval coverage is within tolerance bands.
3. Spread-aware trading sim remains positive under conservative slippage scenario.
4. Daily runbook checks pass for at least 30 consecutive paper-trade days.

## 5) Documentation policy

Repository documentation source-of-truth set:

1. `reports/benchmark_audit_report.md`
2. `reports/benchmark_audit_rerun_report.md`
3. `nyc_temp_prediction_project_plan.md`
4. `MEMORY.md`
5. `reports/master_improvement_plan.md`

All other reports should be considered historical context unless explicitly promoted into one of the files above.
