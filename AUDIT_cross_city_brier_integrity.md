# Cross-City Brier Integrity Audit (NYC vs CHI/PHL)

**Date:** 2026-02-16  
**Scope:** NYC (KXHIGHNY), Chicago (KXHIGHCHI), Philadelphia (KXHIGHPHL)  
**Question:** "Are CHI/PHL truly much better than NYC, or is this a scoring artifact?"

## Verdict (Skeptical/Harsh)

The apparent CHI/PHL superiority is mostly a **metric-scale illusion**, not evidence of miraculous forecasting.  
A Brier around **0.013–0.014** for CHI/PHL is plausible under their current multiclass bucket-row metric and does **not** mean those models are dramatically better than NYC.

## What I verified

### 1) The three pipelines are scoring different row types

- NYC unified benchmark reports binary **contract-row** Brier (`overall_brier`) from `benchmark_summary.csv`, where each row is one Kalshi contract outcome.  
- PHL/CHI real-data benchmarks report multiclass **bucket-day row** Brier over all city bucket bins (`n_buckets` = 57 for PHL, 62 for CHI).

These are not directly comparable because row-level Brier scale shrinks with class count.

### 2) Why 0.014 can happen without leakage

For a K-class one-hot Brier averaged over class columns, even a uniform predictor has expected row Brier:

\[
\text{Brier}_{uniform}(K) = \frac{(1-1/K)^2 + (K-1)(1/K^2)}{K}
\]

- For **K=57 (PHL)**, uniform baseline is **0.017236**.
- For **K=62 (CHI)**, uniform baseline is **0.015869**.

So raw values in the 0.013–0.014 range are **only modestly better than uniform** on that scale; they are not magical.

### 3) Scale-normalized comparison removes the illusion

Using generated audit table (`results/audits/cross_city_brier_scale_audit.*`):

| city | evaluation unit | raw Brier | rows/day | daily aggregate proxy (`raw × rows/day`) |
|---|---|---:|---:|---:|
| NYC | binary contract-row | 0.113719 | 5.996 | 0.681847 |
| PHL | 57-way bucket-day row | 0.014284 | 57 | 0.814212 |
| CHI | 62-way bucket-day row | 0.013514 | 62 | 0.837852 |

After rough scale normalization, CHI/PHL are **not better** than NYC; if anything they look slightly worse in this proxy.

### 4) No evidence (from this pass) of impossible leakage-level skill

- CHI metadata reports MOS MAE ≈ 2.81°F and RMSE ≈ 3.74°F, not absurdly low.
- PHL metadata reports MOS MAE ≈ 2.49°F and RMSE ≈ 3.37°F, also plausible.

If there were direct target leakage, we would expect much tighter errors than this.

## Root cause of inconsistency

The inconsistency is predominantly from **mixing incompatible Brier definitions**:

- NYC: binary contract-level benchmark
- CHI/PHL: multiclass full-bucket matrix benchmark

As long as dashboard/summary compares these raw values directly, it will falsely imply CHI/PHL are "10x better."

## Required fixes

1. **Standardize one canonical cross-city evaluation unit** (recommended: exact Kalshi contract-row scoring per city/day).  
2. Report **both**:
   - raw row-level Brier,
   - and a normalized metric (e.g., skill vs row-level uniform baseline or daily-summed Brier).  
3. Block any ranking table that compares raw Brier across different row definitions.

## Repro artifacts

- Script: `scripts/audit_cross_city_brier_scale.py`  
- Outputs:
  - `results/audits/cross_city_brier_scale_audit.json`
  - `results/audits/cross_city_brier_scale_audit.md`

