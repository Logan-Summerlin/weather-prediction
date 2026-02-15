# Chicago + Philadelphia NYC-Template Deep Dive and Brier-0.03 Roadmap

## Scope reviewed
- Station expansion and contract alignment plan for CHI/PHL rollout.
- NYC top model lineages (E-family + Unified U-family) and shared primitives.
- Existing CHI/PHL benchmark and synthesis scripts.

## What was changed
1. Upgraded `scripts/run_city_nws_kalshi_template_benchmark.py` from a basic ridge+MOS-NN check into a multi-family NYC-template benchmark runner.
2. Added model families directly inspired by NYC best performers:
   - station-feature `ridge` baseline,
   - `mos_residual_ridge` (MOS-error correction),
   - `mos_residual_nn` (NYC-style small NN on MOS residual),
   - `u7_style_regime_stacker` (contract-row regime-aware logistic stacker inspired by U7/U5/U6 layering),
   - `nws_mos` baseline.
3. Added validation-only isotonic CDF calibration for Gaussian families before bucketization.
4. Preserved strict real-data policy:
   - only processed city features,
   - archived MOS,
   - archived Kalshi files when available,
   - no synthetic training data generation.
5. Improved Kalshi benchmarking behavior:
   - still computes contract-Brier if city contract rows exist,
   - emits explicit metadata status when unavailable.

## Benchmark outputs produced (current repository data)
### Philadelphia (`results/philadelphia/phl_nyc_template_benchmark.csv`)
- `ridge`: **0.05398** daily bucket Brier (best in this run)
- `u7_style_regime_stacker`: **0.05567**
- `mos_residual_nn`: **0.08542**
- `nws_mos`: **0.08595**
- `mos_residual_ridge`: **0.09514**

Interpretation:
- Residual MOS correction did not beat the station-ridge family on this dataset split.
- Regime stacker improved over MOS families and is close to ridge, but still above 0.03 target.

### Chicago (`results/chicago/chi_nyc_template_benchmark.csv`)
- Missing required processed CHI artifacts in current repo snapshot (`data/chicago/processed/features_train.csv` absent).

### Kalshi pre-settlement contract benchmark
- Not currently available for CHI/PHL in included `data/kalshi_presettlement.csv` + settled archives.
- Script records `kalshi_contract_benchmark_status: unavailable` until city contract rows are backfilled.

## Why 0.03 is hard and what to do next (priority order)

### A) Data parity and leakage-safe live equivalence (highest priority)
1. Build CHI operational dataset parity first (same fields, lags, station-availability behavior as live run).
2. For both CHI/PHL, train on exactly the same representation used at inference (including missingness masks).
3. Track train-infer feature drift daily; halt promotion if drift exceeds thresholds.

### B) Port NYC top family mechanics more fully
1. Promote contract-row objective optimization (E17/E40/U5 style) over only daily Gaussian fitting.
2. Add two-stage recalibration on top of stacker outputs:
   - Platt on logits (U6/E21 idea), then
   - isotonic per bucket with per-day renormalization.
3. Add regime features from U7:
   - seasonal harmonics,
   - sigma-confidence bands,
   - spread/depth/staleness once CHI/PHL Kalshi snapshots are available.

### C) Improve residual learning quality
1. Replace single MOS source with robust blend (GFS/NAM + station persistence + lagged station gradients).
2. Move from point residual to heteroscedastic residual head (`mu_resid`, `sigma_resid`) and train with NLL/CRPS proxy.
3. Use month/regime-conditional sigma models rather than fixed monthly lookup when enough data exists.

### D) Add physically informed city-specific signals
1. Chicago: lake-breeze and lake-inland gradient composites, wind-direction-conditioned sectors.
2. Philadelphia: coastal marine influence + Delaware Valley urban heat gradient + NYC correlation transfer feature.
3. Use grouped permutation-importance to reject unstable features.

### E) Benchmark discipline to avoid false progress
1. Keep naive/persistence/climatology/NWS/Kalshi in every run.
2. Optimize on chronological split and reserve untouched recent holdout for promotion.
3. Report seasonal Brier and tail-bucket reliability, not just global mean.

## Promotion gate proposal toward 0.03
- **Gate 1 (data readiness):** CHI/PHL processed + MOS + Kalshi contracts all available with >=95% date coverage.
- **Gate 2 (probability quality):** holdout ECE and PIT diagnostics stable across seasons.
- **Gate 3 (benchmark):** beat NWS and Kalshi pre-settlement on contract Brier with conservative costs.
- **Gate 4 (stretch target):** sustained rolling Brier <=0.03 for at least one full seasonal cycle before capital scaling.

## Operational note
Current repo state supports a real-data Philly benchmark run and robust status reporting for missing CHI/Kalshi inputs, enabling clean PR integration while preventing silent synthetic fallbacks.
