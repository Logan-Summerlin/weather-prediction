# MOS Era Dummy Variable: Current Design and Recommendations

**Date:** 2026-02-13
**Purpose:** Document how MOS eras are currently handled and recommend improvements.

## 1. The Three Distinct MOS Eras

The extended MOS dataset spans 2000-06 to 2026-02 and contains three distinct data regimes:

| Era | Date Range | GFS Source | NAM Source | Ensemble | Rows |
|-----|-----------|-----------|-----------|----------|------|
| **GFS-only proxy** | 2000-06 to 2002-04 | Airport avg (KJFK+KLGA+KEWR) with monthly offsets | **Missing (NaN → imputed)** | GFS only | 649 |
| **GFS+NAM proxy** | 2002-04 to 2003-12 | Airport avg with monthly offsets | Airport avg with monthly offsets | (GFS+NAM)/2 | 638 |
| **KNYC native** | 2004-01 to 2026-02 | KNYC direct | KNYC direct | (GFS+NAM)/2 | ~8,078 |

## 2. Current Implementation

### What happens now (in `scripts/retrain_extended_mos.py`)

**Feature:** A single binary `mos_era` (0 = airport_proxy, 1 = knyc_native), loaded from `data/mos/mos_era_indicator.csv` and appended to the feature vector.

**NAM imputation (line 557-558):**
```python
df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
```

For the 649 GFS-only proxy rows (2000-06 to 2002-04), this means:
- `nam_mos_tmax_f` is filled with `mos_ensemble_tmax_f` (which equals `gfs_mos_tmax_f`)
- **Result:** `gfs = nam = ensemble` for these rows
- **Downstream effect:** `gfs_nam_spread = 0` and `gfs_nam_sign = 0` for all 649 rows

**Scaling:** The binary `mos_era` feature is passed through `StandardScaler` along with all other features.

### Problems with the current approach

1. **No distinction between GFS-only proxy and GFS+NAM proxy.** The binary treats both proxy eras the same, but they have very different data quality:
   - GFS-only: no real NAM signal, ensemble is just GFS, `gfs_nam_spread` is artificially zero
   - GFS+NAM: real two-model ensemble, genuine spread signal

2. **StandardScaler on a binary is wasteful.** Scaling transforms `{0, 1}` to `{-mean/std, (1-mean)/std}`. Since ~87% of rows are era=1, the scaled values are approximately `{-2.6, 0.38}`. This isn't harmful but wastes the scaler on what should be a clean indicator.

3. **Fake NAM signal in GFS-only era.** Setting `nam = gfs` creates an artificial pattern: every GFS-only row has zero GFS-NAM spread. The model may learn "if spread=0 then era=0" rather than learning meaningful relationships. More importantly, the `gfs_nam_spread` and `gfs_nam_sign` features — which are among the most useful MOS error memory features — carry no real information for 649 training rows.

4. **No interaction effects.** The model must learn purely from the single binary that the MOS data has different bias/variance characteristics in different eras. It has no feature-level guidance to condition MOS-related features on era.

5. **Possible sigma impact.** The GFS-only era has different residual variance than the native era. If the model treats all eras equally during training, the blended residual distribution may be tighter than reality (because the proxy era has lower residual variance due to harmonized offsets centering the data). This could explain the tighter sigma (2.79 vs 3.02) and the tail reliability regression.

## 3. Recommended Improvements

### Option A: Three-level era encoding (minimal change)

Replace the single binary with two binary indicators:

| Feature | GFS-only proxy | GFS+NAM proxy | KNYC native |
|---------|:-:|:-:|:-:|
| `era_proxy` | 1 | 1 | 0 |
| `era_gfs_only` | 1 | 0 | 0 |

This lets the model distinguish:
- GFS-only proxy (both = 1): "don't trust the NAM features or spread features"
- GFS+NAM proxy (proxy=1, gfs_only=0): "MOS data is harmonized but real two-model ensemble"
- KNYC native (both = 0): "full trust in all MOS features"

Implementation change: add `era_gfs_only = 1 where nam_mos_tmax_f was NaN before imputation`.

### Option B: NAM-availability indicator (targeted)

Instead of era-based dummies, add a single `nam_available` feature:

| Feature | Value when NAM is real | Value when NAM is imputed |
|---------|:-----:|:------:|
| `nam_available` | 1 | 0 |

This is more precise because it directly flags the rows where `nam_mos_tmax_f` was imputed from GFS. The model can then learn to discount NAM-derived features (spread, sign) when `nam_available = 0`.

This also generalizes to future scenarios where NAM might be missing for operational reasons (delayed run, data outage).

### Option C: Era interaction features (higher capacity)

Add interaction terms between the era indicator and key MOS features:

```
era_x_gfs_mos      = mos_era * gfs_mos_tmax_f
era_x_nam_mos      = mos_era * nam_mos_tmax_f
era_x_ensemble     = mos_era * mos_ensemble_tmax_f
era_x_spread       = mos_era * gfs_nam_spread
era_x_error_7d     = mos_era * mos_error_7d
era_x_station_gap  = mos_era * mos_station_gap
```

This lets the model learn different MOS-feature slopes for proxy vs native eras without requiring it to discover the interaction purely from the nonlinear hidden layers.

**Risk:** with only ~1,287 proxy-era training rows (7,862 total), interaction features may overfit to the proxy era. This approach should only be used with regularization and careful validation.

### Option D: Don't scale the dummy (minor fix)

Exclude `mos_era` (and any other binary indicators) from `StandardScaler`. Pass them through unscaled. Most neural networks handle this fine, and it preserves the clean `{0, 1}` semantics.

Implementation: partition the feature list into `scaled_features` and `unscaled_features`, apply `StandardScaler` only to the former, then concatenate.

## 4. Recommendation

**Implement Options A + B + D together.** Specifically:

1. **Replace single `mos_era` with three features:** `era_proxy`, `era_gfs_only`, `nam_available`
2. **Don't scale these binary features** through StandardScaler
3. **Don't impute NAM with GFS for GFS-only rows** — instead, set `nam_mos_tmax_f = 0` (after scaling, this means the feature is zeroed out rather than carrying fake GFS signal) or use a dedicated missing-value indicator

Skip Option C (interactions) for now — the proxy era has too few rows to support reliable interaction learning.

### Expected impact

- **Sigma fix:** If the model can distinguish GFS-only proxy rows and assign them appropriately wider uncertainty, the sigma regression may partially resolve itself
- **Spread features:** The `gfs_nam_spread` feature will no longer carry an artificial zero signal for 649 rows, improving its information content
- **Minimal risk:** Adding 2 binary features to a 122-feature model is unlikely to hurt, and removing the fake NAM signal should help

## 5. Implementation Checklist

1. In `scripts/build_extended_mos.py`:
   - Add a `nam_available` column to `combined_mos_extended.csv` (1 where NAM was real, 0 where imputed or missing)

2. In `scripts/retrain_extended_mos.py`:
   - Load `nam_available` from the extended MOS CSV
   - Create `era_proxy` and `era_gfs_only` from `mos_source` column
   - Add all three as unscaled features
   - Remove the old `mos_era` single binary
   - **Change NAM imputation:** instead of `fillna(ensemble)`, use `fillna(0)` after scaling (or use a mask so the model sees the zeroed feature + the `nam_available=0` indicator)
   - Partition features into scaled and unscaled groups before `StandardScaler`

3. Retrain and re-benchmark to measure impact on:
   - Base model MAE/RMSE (should be similar or slightly better)
   - Sigma estimates (should widen, fixing tail reliability)
   - Benchmark Brier scores (should be similar or slightly better for E11/E17/E18)
