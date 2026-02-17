# Brier Score Metrics: Bucket-Day vs Contract

This project uses two distinct Brier score aggregations. Both use the same core formula — `mean((predicted_prob - outcome)²)` — but differ in **which (day, bucket) pairs are included**.

## Bucket-Day Brier

**Used by:** _(Deprecated — all benchmarks now use Contract Brier.)_ Formerly used by Honest Benchmark and E-series base models.

Averages over **all** (day × bucket) pairs, including every bucket in the grid.

| Property | Value |
|----------|-------|
| Data points per day | All buckets (62 for CHI, 57 for PHL/NYC) |
| Includes trivial buckets | Yes — e.g. "Below -10°F" on a July day |
| Typical score range | 0.014 – 0.016 |
| Uniform-random baseline | ~(n-1)/n² ≈ 0.016 (CHI) / 0.017 (PHL) |

Because ~50 of 62 buckets on any given day have near-zero predicted probability and outcome=0, each contributing ≈0.000001 to the mean, the metric is dominated by these easy cells and the absolute value is very low.

## Contract Brier

**Used by:** All benchmarks — Honest Benchmark, E-series, CHI/PHL baselines, MOS Residual, Unified Benchmark, U-series synthesis models.

Averages only over (day × bucket) pairs **where Kalshi listed a tradeable contract** — typically the 10–15 buckets near the forecast temperature where there is genuine uncertainty.

| Property | Value |
|----------|-------|
| Data points per day | ~10–15 active Kalshi contracts |
| Includes trivial buckets | No — only traded temperature ranges |
| Typical score range | 0.10 – 0.19 |
| Kalshi market baseline | ~0.11 – 0.13 |

## Example: 70°F day in Chicago (62 buckets)

| Bucket | Model Pred | Outcome | (p-o)² | In Bucket-Day? | In Contract? |
|--------|-----------|---------|--------|----------------|-------------|
| Below -10°F | 0.001 | 0 | 0.000001 | Yes | No |
| … (~48 trivial buckets) | ~0.001 | 0 | ~0.000001 | Yes | No |
| 66–68°F | 0.15 | 0 | 0.0225 | Yes | Yes |
| 68–70°F | 0.22 | 0 | 0.0484 | Yes | Yes |
| **70–72°F** | **0.25** | **1** | **0.5625** | Yes | Yes |
| 72–74°F | 0.18 | 0 | 0.0324 | Yes | Yes |
| 74–76°F | 0.10 | 0 | 0.0100 | Yes | Yes |
| … (~8 trivial buckets) | ~0.001 | 0 | ~0.000001 | Yes | No |

**Bucket-Day Brier** averages all 62 cells → ~0.015 (diluted by trivial buckets).

**Contract Brier** averages only the ~5–12 contested cells → ~0.11 (harder test).

## Key takeaway

As of the latest update, all benchmarks have been standardized to use **Contract Brier** as the primary metric. This ensures all model scores are directly comparable across benchmark scripts. The bucket-day Brier computation is retained only internally for Ridge alpha search and calibration fitting, but is never reported as a final result.
