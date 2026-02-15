# Cross-City Model Accuracy Audit Report

**Date:** 2026-02-15
**Scope:** NYC (KXHIGHNY), Chicago (KXHIGHCHI), Philadelphia (KXHIGHPHL)
**Trigger:** Suspicion that CHI/PHL models report artificially better Brier scores than NYC
**Verdict:** Suspicion CONFIRMED — scores are incomparable and CHI/PHL results are unreliable

---

## Executive Summary

The Chicago and Philadelphia models are **NOT** more accurate than NYC models. The apparent superiority is caused by **three independent problems** that each alone would invalidate cross-city comparison:

1. **Incomparable evaluation units** — NYC evaluates on 2°F Kalshi contract rows; PHL evaluates on 10°F bucket-day pairs. These produce fundamentally different Brier score scales.
2. **Chicago benchmark never ran** — A path error prevented execution. All downstream CHI synthesis/calibration/backtest results rely on synthetic data where predictions are artificially correlated with outcomes.
3. **Silent synthetic data fallback** — Both CHI and PHL synthesis scripts generate fake predictions with `model_mu = actual_tmax + N(0, 2.5)` when real benchmarks are missing, producing unrealistically good results with no warning beyond a log message.

**No trading decisions should be based on cross-city Brier comparisons until these issues are resolved.**

---

## Finding 1: Incomparable Evaluation Methodologies (CRITICAL)

### NYC Evaluation Unit
- **Source:** `scripts/run_unified_outperformance_benchmark.py:339-438`
- **Unit:** Kalshi contract rows — each row is a specific contract (above/below/between)
- **Contract width:** ~2°F for "between" contracts
- **Contracts per day:** ~6 (1 below + 3-4 between + 1 above)
- **Data:** `data/real_kalshi_2023_2024.csv` — 4,377 rows, 730 dates

Verified contract structure for 2023-01-01:
```
below    [     ,  50.0] outcome=0
between  [ 50.0,  52.0] outcome=0
between  [ 52.0,  54.0] outcome=0
between  [ 54.0,  56.0] outcome=1
between  [ 56.0,  58.0] outcome=0
above    [ 57.0,      ] outcome=0
```

### PHL Evaluation Unit
- **Source:** `scripts/run_phl_benchmark.py:179-237`
- **Unit:** Bucket-day pairs — 2D matrix (n_days × n_buckets)
- **Bucket width:** 10°F
- **Buckets per day:** 10 (Below 20, 20-29, 30-39, ..., Above 100)
- **Effect:** Tail buckets (Below 20, Above 100) are nearly always 0 and trivially predictable, pulling the mean Brier down substantially

### Why This Makes Scores Incomparable

With 10°F buckets, the per-bucket Brier for PHL shows (from `results/philadelphia/phl_benchmark_detail.json`):
- Below 20: **0.0016** (trivial — almost never happens)
- Above 100: **0.0007** (trivial — almost never happens)
- Core buckets (40-89°F): **0.074–0.101** (comparable to NYC contract-level difficulty)

The overall mean is dragged down to ~0.055 by the trivial tail buckets. NYC's 2°F contracts don't have this averaging-with-easy-tails effect.

**The PHL Brier of 0.055 and NYC Brier of 0.1137 measure completely different things.** They cannot be compared.

### Chicago Bucket Count
- **Source:** `src/city_config.py:123-135`
- CHI uses **11 buckets** (extra bucket at Below 10) vs 10 for NYC/PHL
- Further invalidates cross-city Brier comparison even if methodology were aligned

---

## Finding 2: Chicago Benchmark Never Ran (CRITICAL)

### Evidence
**File:** `results/chicago/chi_nyc_template_benchmark.json`
```json
{
  "city": "chi",
  "status": "missing_required_inputs",
  "error": "[Errno 2] No such file or directory: '/workspace/weather-prediction/data/chicago/processed/features_train.csv'",
  "output_csv": "/workspace/weather-prediction/results/chicago/chi_nyc_template_benchmark.csv"
}
```

The benchmark script references `/workspace/weather-prediction/` instead of the actual path `/home/user/weather-prediction/`. The processed data files DO exist at the correct path (`data/chicago/processed/features_train.csv` — 42MB), but the script never found them.

### Consequence
- No `results/chicago/base_predictions.csv` exists (verified)
- No `results/chicago/nn_predictions.csv` exists (verified)
- Any execution of `run_chi_synthesis_calibration.py` would hit the synthetic data fallback at line 127
- All downstream Chicago results (synthesis, calibration, backtest) are based on synthetic data

---

## Finding 3: Silent Synthetic Data Fallback (CRITICAL)

### Mechanism
Both CHI and PHL synthesis scripts contain identical fallback logic:

**Files:**
- `scripts/run_chi_synthesis_calibration.py:121-174`
- `scripts/run_phl_synthesis_calibration.py:121-174`

```python
# Line 155-158 (both files):
actual_tmax = clim_mean + rng.normal(0, clim_std)
model_mu = actual_tmax + rng.normal(0, 2.5)  # <-- prediction is actual + noise
```

The synthetic model prediction is centered on the synthetic actual temperature with only 2.5°F standard deviation noise. This is **unrealistically good** — it means the "model" knows the actual temperature to within ~5°F before the fact.

### Why This Is Dangerous
- The fallback emits only a `logger.warning()` — no exception, no failure
- Downstream scripts (backtest, promotion evaluation) have no way to distinguish synthetic from real results
- Calibration trained on this synthetic data will appear well-calibrated because the predictions were designed to be close to actuals
- Brier scores will be artificially excellent

### Backtest Also Affected
- `scripts/run_chi_backtest.py:850` — same synthetic fallback pattern
- `scripts/run_phl_backtest.py:850` — same synthetic fallback pattern
- Additionally, backtest market simulation gives the market the actual temperature: `market_mu = tmax + rng.normal(0, 2.5)` (line 147), biasing PnL toward the model

---

## Finding 4: NYC Calibration-on-Test-Period Data (HIGH)

### Mechanism
**File:** `scripts/run_unified_outperformance_benchmark.py:1498-1612`

```python
ext_cal_mask = df["period"] == "IS"       # All 2023-2024 data
cal_df_ext = df[ext_cal_mask].copy()      # Calibration includes all IS with actual outcomes
cal_outcomes = cal_df_ext["actual_outcome"].values  # Used for calibration fitting
```

Calibration variants U2-U9 are fit on the full IS period (2023-2024) including actual outcomes. The reported "overall Brier" includes evaluation on this same IS period.

### Impact
- **Sorted by `overall_brier`** (line 1262): `sorted_metrics = sorted(all_metrics, key=lambda m: m.get("overall_brier", 999))`
- **Best model selected by `overall_brier`** (line 1401): `f"Best overall Brier: {best['variant']} ({best['overall_brier']:.4f})"`
- The reported U7 Brier of 0.1137 is the **overall** (IS+OOS combined), not OOS-only
- IS portion is contaminated; the clean metric is `oos_brier` (2025 only), which is computed but not used for ranking

### Mitigation
The script does compute separate IS and OOS Brier scores. The OOS Brier (2025 only) is not contaminated by calibration fitting. However, the headline "best model" ranking uses the contaminated overall metric.

---

## Finding 5: PHL Benchmark Appears Legitimate But Unimpressive (MEDIUM)

### Evidence
Philadelphia has real benchmark results:
- `results/philadelphia/phl_benchmark_summary.csv` — real model scores
- `results/philadelphia/phl_benchmark_detail.json` — per-bucket breakdown
- `results/philadelphia/phl_nws_kalshi_benchmark.json` — NWS comparison

### PHL Model vs NWS (2024)
From `phl_nws_kalshi_benchmark.json`:
| Model | Brier (2024) | MAE |
|-------|-------------|-----|
| NWS MOS | **0.0325** | 2.38°F |
| Ridge (ours) | 0.0544 | 4.44°F |
| HeteroscedasticNN | 0.0575 | 4.97°F |
| Market Proxy | 0.0577 | 5.04°F |
| Climatology | 0.0664 | 6.61°F |

**Our PHL model loses to NWS by 68%.** This fails the CLAUDE.md promotion gate: "OOS Brier consistently beats NWS baseline across seasonal slices."

---

## Summary of Issues

| # | Finding | Severity | Files | Impact |
|---|---------|----------|-------|--------|
| 1 | Incomparable eval units (2°F contracts vs 10°F buckets) | **CRITICAL** | `run_unified_*.py` vs `run_phl_benchmark.py` | Cross-city Brier comparison is invalid |
| 2 | CHI benchmark never ran (path error) | **CRITICAL** | `results/chicago/chi_nyc_template_benchmark.json` | All CHI synthesis/calibration/backtest results are fake |
| 3 | Silent synthetic data fallback | **CRITICAL** | `run_chi_synthesis_calibration.py:121`, `run_phl_synthesis_calibration.py:121` | Corrupts results without failure |
| 4 | NYC calibration evaluated on calibration period | **HIGH** | `run_unified_outperformance_benchmark.py:1262,1401` | Headline Brier score is contaminated |
| 5 | PHL model loses to NWS by 68% | **MEDIUM** | `results/philadelphia/phl_nws_kalshi_benchmark.json` | Fails promotion gate |
| 6 | CHI has 11 buckets vs 10 elsewhere | **MEDIUM** | `src/city_config.py:123-135` | Even aligned methodology wouldn't be comparable |

---

## Recommendations

### Immediate (Block Trading)
1. **Do not make cross-city Brier score comparisons.** The evaluation units are different.
2. **Do not use any Chicago synthesis/calibration/backtest results.** They are based on synthetic data.
3. **Fix the Chicago benchmark path** from `/workspace/` to the correct project root and re-run.
4. **Remove or gate the synthetic data fallback** — replace with `raise RuntimeError(...)` so failures are loud.

### Short-Term (Before Any City Goes Live)
5. **Standardize evaluation methodology** — all cities should use the same contract structure and Brier computation for comparability.
6. **Rank NYC models by OOS Brier only** (`oos_brier`), not `overall_brier` which includes the calibration period.
7. **Re-run PHL promotion evaluation** using the same contract-level evaluation as NYC to get a fair comparison.
8. **Add automated data integrity tests** that verify:
   - No synthetic data is used in any evaluation pipeline
   - Calibration period does not overlap with test evaluation period
   - Bucket definitions are consistent across cities (or explicitly flagged as different)

### Medium-Term (Operational Hardening)
9. **Implement a universal `evaluate_model()` function** used by all city pipelines to prevent methodology drift.
10. **Add a schema check** at the start of every benchmark script that validates required input files exist (no silent fallbacks).
11. **Compute and report per-bucket Brier** alongside overall to make cross-city comparisons meaningful at the bucket level.
