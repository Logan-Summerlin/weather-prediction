# Chicago + Philadelphia Upgrade Plan Using NYC Top Model Families

## What was studied
- Station expansion plan and city contract specs (`prediction_market_expansion.md`, `docs/chicago_contract_spec.md`, `docs/philadelphia_contract_spec.md`).
- NYC top model families and benchmark lineage (`docs/top15_models_brier_function_reference.md`, `scripts/run_e0_e8_best_model_benchmark.py`, `scripts/run_wga_v2_benchmark.py`, `scripts/run_unified_outperformance_benchmark.py`).
- NYC MOS-residual implementation patterns (`scripts/retrain_extended_mos.py`, `scripts/retrain_extended_validation.py`, `scripts/mos_ensemble_pipeline.py`, `scripts/train_wga_v2.py`).

## NYC template elements to port directly
1. **Residual learning over MOS Tmax**
   - Core target: `residual = actual_tmax - mos_tmax`.
   - Time-safe lagged memory features: `mos_error_lag1`, rolling 7/14-day error, rolling abs-error.
   - Small NN (regularized) to predict residual, then reconstruct `mu = mos_tmax + residual_pred`.
2. **Distributional output for bucket contracts**
   - Convert `(mu, sigma)` to contract probabilities via CDF differences aligned with contract thresholds.
3. **Benchmarking against market and NWS**
   - Brier on contract rows for model vs Kalshi pre-settlement vs NWS baseline.
4. **No synthetic-data dependence**
   - Training and benchmarking only from real observed temperatures + archived MOS + archived Kalshi snapshots.

## Implementation delivered
- Added `scripts/run_city_nws_kalshi_template_benchmark.py`.
- Script supports `--city phl` and `--city chi`.
- Uses actual city processed datasets (`data/<city>/processed`) and MOS files.
- Trains:
  - Ridge baseline on processed station features.
  - NYC-template MOS residual NN correction model.
  - NWS MOS baseline.
- Benchmarks against Kalshi pre-settlement only when local Kalshi archives include matching city contracts.
- Never generates synthetic training/evaluation data.

## Current data availability note
- Repository currently contains Philadelphia processed and MOS data, but no local Chicago processed/MOS artifacts and no archived CHI/PHL Kalshi contract rows in the included Kalshi CSV snapshots.
- The new benchmark runner handles this gracefully and writes explicit benchmark status metadata.

## Recommended next operational steps
1. Backfill CHI/PHL Kalshi pre-settlement + settled contract archives.
2. Add KORD MOS combined file (`data/chicago/mos/combined_mos_kord.csv`).
3. Run city preprocessing for Chicago to populate `data/chicago/processed`.
4. Execute:
   - `python scripts/run_city_nws_kalshi_template_benchmark.py --city phl`
   - `python scripts/run_city_nws_kalshi_template_benchmark.py --city chi`
5. Promote only variants that beat both NWS and Kalshi pre-settlement on contract-level OOS Brier.
