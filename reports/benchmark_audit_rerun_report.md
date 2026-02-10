# Benchmark Audit Re-Run Report (Best Model + Expanded OOS Pre-Settlement)

**Date:** 2026-02-10  
**Objective:** Re-run prediction-market benchmark using the current best model and the larger 2025 pre-settlement OOS sample; verify reported methodology against implementation.

## 1) What was implemented

1. Re-trained the Phase 1 combined pipeline (`scripts/phase1_combined_best.py`) and confirmed the best model family remains the 5-seed ensemble (`E_Ensemble_5seed`).
2. Added persistent artifact saving logic for local re-use and benchmark exports, while keeping binary checkpoints out of Git PRs:
   - `results/phase1_combined/best_model_sigma_by_month.json`
3. Added benchmark-ready prediction exports for direct consumption by benchmark code:
   - `data/best_model_predictions_2023_2024.csv`
   - `data/best_model_predictions_2025.csv`
4. Updated benchmark runner (`scripts/test_model_vs_benchmarks.py`) to default to the new best-model prediction files and to allow explicit legacy-vs-best toggling via CLI.

## 2) Re-training validation vs previously reported levels

Re-training results (from fresh run of `scripts/phase1_combined_best.py`):

- `E_Ensemble_5seed` **test MAE = 1.990°F**, **OOS MAE = 2.020°F**.
- This is consistent with previously reported levels (~1.99 test / ~2.02 OOS), i.e., the model reached historical performance.

## 3) Benchmark dataset size check (expanded OOS)

From re-run benchmark merge logs:

- Final merged dataset: **6,204 buckets across 1,089 dates**.
- Period split: **IS = 4,046**, **OOS = 2,158**.

This confirms the benchmark now evaluates a materially larger OOS bucket sample than the old small-sample setup.

## 4) Re-run benchmark results (best model)

Using:

```bash
python scripts/test_model_vs_benchmarks.py \
  --model-is data/best_model_predictions_2023_2024.csv \
  --model-oos data/best_model_predictions_2025.csv
```

Key overall Brier scores (lower is better):

- Kalshi pre-settlement: **0.1271**
- **Best model: 0.1335**
- NWS: **0.1418**

Interpretation:

- Best model now **beats NWS** on Brier by **0.0083**.
- Best model remains behind pre-settlement Kalshi by **0.0064**.

## 5) Legacy-vs-best audit check

Legacy run command:

```bash
python scripts/test_model_vs_benchmarks.py --use-legacy-model
```

Legacy model overall Brier:

- Model (legacy): **0.1816**

Best model overall Brier:

- Model (best): **0.1335**

Improvement from replacing benchmarked model:

- **Δ Brier = -0.0481** (substantial).

This directly validates the central audit claim that previous conclusions were confounded by benchmarking an obsolete model.

## 6) Method-implementation consistency checks

Verified in code and outputs:

1. **Model source selection is now explicit** (best vs legacy via CLI), removing hidden mismatch risk.
2. Probability conversion is still Gaussian CDF bucketization (below/between/above) and numerically clipped, consistent with original benchmark methodology.
3. Benchmarked rows come from strict merge of pre-settlement + settled contracts + daily model and NWS forecasts; dropped rows are explicitly logged.
4. Outputs are reproducible and persisted for both best-model and legacy runs:
   - `results/prediction_market_benchmark/best_model_run/*`
   - `results/prediction_market_benchmark/legacy_model_run/*`

## 7) Important caveats that remain (not yet fixed in this pass)

The following known issues from the audit still apply to the current benchmark engine:

1. Trading simulation still uses midpoint-like `presettlement_prob` execution instead of explicit bid/ask crossing.
2. Sharpe is still non-annualized in benchmark script.
3. No confidence intervals are reported for OOS P&L.
4. Probability-mass sum checks across mutually exclusive bucket sets are not explicitly validated.

So this run resolves the **wrong-model** problem and expands OOS sample usage, but does **not** yet address all market microstructure and uncertainty-reporting limitations.

---

## Produced artifacts

- Best model predictions for benchmark:
  - `data/best_model_predictions_2023_2024.csv`
  - `data/best_model_predictions_2025.csv`
- Reproducible non-binary artifact:
  - `results/phase1_combined/best_model_sigma_by_month.json`
- Benchmark outputs (best model):
  - `results/prediction_market_benchmark/full_benchmark_report.md`
  - `results/prediction_market_benchmark/presettlement_brier_scores.csv`
  - `results/prediction_market_benchmark/trading_simulation_results.csv`
- Snapshot folders for direct comparison:
  - `results/prediction_market_benchmark/best_model_run/`
  - `results/prediction_market_benchmark/legacy_model_run/`
