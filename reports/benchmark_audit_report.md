# Benchmark Audit Report — Forensic Findings and Resolution Status

**Date:** 2026-02-10  
**Purpose:** Canonical record of benchmark integrity issues identified during audit, plus current resolution status.

---

## Executive summary

The original benchmark conclusions were directionally compromised by two dominant flaws:

1. **Wrong model benchmarked** (legacy vanilla NN instead of the project best model).
2. **Unrealistic trading execution** (midpoint-like costs instead of crossing bid/ask).

Both issues have now been addressed in the benchmark engine and re-run outputs.

---

## Audit findings status matrix

| ID | Finding | Severity | Status |
|---|---|---:|---|
| 1 | Wrong model benchmarked (legacy max-train NN) | FATAL | **Resolved** |
| 2 | Sigma mischaracterized in report text | High | **Documented/clarified** |
| 3 | Calibration pathology understated by ECE-only summary | High | **Partially resolved** (full reliability tables retained; still Gaussian model) |
| 4 | Trading sim ignored bid/ask spread | High | **Resolved** |
| 5 | Sharpe convention inconsistency | High | **Resolved** (raw + annualized both reported) |
| 6 | OOS uncertainty underreported | High | **Resolved** (bootstrap CI added for OOS P&L) |
| 7 | Probability-mass sums not validated | Low | **Resolved** (per-date mass-check CSVs) |
| 8 | Pre-settlement snapshot realism limitations | Medium | **Open** (known scope limit) |
| 9 | Full microstructure fill simulation missing | Medium | **Open** (future enhancement) |

---

## Key validated facts

- Benchmark default model is now `best_model_predictions_2023_2024.csv` and `best_model_predictions_2025.csv`.
- Legacy benchmark path is preserved only behind `--use-legacy-model` for controlled comparisons.
- Trading costs now use pre-settlement orderbook crossing:
  - YES: ask
  - NO: 1 - bid
- OOS P&L is now reported with 95% bootstrap confidence intervals.
- Probability mass checks are exported for model and NWS per date.

---

## Why the conclusions changed

The forecast-quality ranking remains:

- Kalshi pre-settlement best,
- best model second,
- NWS third (by Brier).

But after spread-aware execution, simulated strategy P&L is much more conservative (and often negative), which materially changes deployability conclusions.

---

## Canonical next steps

1. Integrate calibrated non-Gaussian distribution output for bucketization.
2. Add fuller execution realism (fill assumptions, depth-aware slippage).
3. Promote threshold/risk tuning to strictly OOS walk-forward procedure.
4. Gate live trading behind calibration + execution diagnostics, not only Brier/MAE.

This file remains the forensic baseline; operationally current benchmark behavior is described in `benchmark_audit_rerun_report.md`.
