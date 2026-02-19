# Benchmark Audit Re-Run Report (Best Model + Market-Microstructure Corrections)

**Date:** 2026-02-10  
**Objective:** Verify that benchmark methodology in code matches claims from the original audit and re-run reports, then close key methodological gaps.

## 1) Implementation audit against prior reports

The two prior benchmark audit reports made four high-priority implementation claims:

1. Best model predictions should be the benchmark default (not legacy max-train NN).
2. Trading simulation should account for orderbook crossing costs (bid/ask), not midpoint-only execution.
3. Sharpe convention should be explicit and consistent.
4. Bucket probability mass should be validated per date.

### Current status in code

- ✅ **Best-model default is implemented** in `scripts/test_model_vs_benchmarks.py` via:
  - `DEFAULT_MODEL_IS = data/best_model_predictions_2023_2024.csv`
  - `DEFAULT_MODEL_OOS = data/best_model_predictions_2025.csv`
  - `--use-legacy-model` toggle retained for controlled regression checks.
- ✅ **Bid/ask crossing is now implemented** for trading simulation:
  - YES cost = ask price (`ask_cents / 100`)
  - NO cost = `1 - bid` (`1 - bid_cents / 100`)
  - midpoint fallback only when bid/ask is missing.
- ✅ **Sharpe is now explicit in both forms**:
  - per-trade Sharpe (`mean/std`)
  - annualized Sharpe (`* sqrt(252)`).
- ✅ **Probability-mass checks added**:
  - `model_probability_mass_check.csv`
  - `nws_probability_mass_check.csv`
  with per-date sum/deviation and tolerance flag.
- ✅ **OOS trading uncertainty now reported** via bootstrap 95% CI (date-block resampling).

## 2) Re-run command and dataset integrity

```bash
python scripts/test_model_vs_benchmarks.py \
  --model-is data/best_model_predictions_2023_2024.csv \
  --model-oos data/best_model_predictions_2025.csv
```

Re-run merge diagnostics:

- Final merged dataset: **6,204 bucket rows** across **1,089 dates**.
- Split: **IS 4,046** / **OOS 2,158**.

## 3) Re-run scoring results (unchanged scoring methodology)

Overall Brier (lower better):

- Kalshi pre-settlement: **0.1271**
- Best model: **0.1335**
- NWS: **0.1418**

Interpretation:

- Best model still beats NWS by **0.0083** Brier.
- Kalshi pre-settlement still beats best model by **0.0065** Brier.

## 4) Trading simulation impact after microstructure correction

The major behavioral change from midpoint simulation to bid/ask crossing is that previously positive backtest P&L is no longer positive under conservative execution.

Example (best threshold in summary):

- Model_All @ 0.20: **net P&L = -$121.53**, ROI **-13.6%**.
- NWS_All @ 0.20: **net P&L = -$158.34**, ROI **-15.8%**.

This confirms the original audit warning that ignoring spread materially overstated profitability.

## 5) Remaining caveats

Even after this pass, two caveats remain:

1. The benchmark still compares probabilities at pre-settlement snapshot time only; it is not a full intraday execution simulator with queue-position/fill modeling.
2. Model distribution remains Gaussian bucketization (`mu`, `sigma`) rather than a richer calibrated nonparametric CDF.

## 6) Produced artifacts (current truth)

- Main benchmark outputs:
  - `results/prediction_market_benchmark/full_benchmark_report.md`
  - `results/prediction_market_benchmark/presettlement_brier_scores.csv`
  - `results/prediction_market_benchmark/presettlement_calibration.csv`
  - `results/prediction_market_benchmark/trading_simulation_results.csv`
- New method-validation outputs:
  - `results/prediction_market_benchmark/model_probability_mass_check.csv`
  - `results/prediction_market_benchmark/nws_probability_mass_check.csv`

## 7) Conclusion

The benchmark stack now matches the core methodological requirements raised by the audit:

- correct model default,
- realistic spread-aware execution,
- explicit Sharpe conventions,
- uncertainty intervals for OOS trading,
- probability-mass diagnostics.

Forecast-quality conclusions are stable; trading-profitability conclusions are now materially more conservative and operationally credible.
