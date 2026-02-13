# Airport MOS Similarity Analysis Report

**Generated:** 2026-02-13 08:50

**Purpose:** Evaluate NYC-area airport MOS stations as proxies for KNYC to extend MOS training data back to 2000 (GFS/AVN) and 2002 (NAM/ETA).

## 1. Data Summary

### Airport MOS Data Availability

| Station | Model | Date Range | Total Days | Model Labels |
|---------|-------|------------|------------|--------------|
| KJFK | GFS | 2000-05-31 to 2026-02-15 | 9370 | AVN: 1273, GFS: 8097 |
| KJFK | NAM | 2002-04-03 to 2026-02-15 | 8718 | ETA: 2441, NAM: 6277 |
| KLGA | GFS | 2000-05-31 to 2026-02-15 | 9370 | AVN: 1273, GFS: 8097 |
| KLGA | NAM | 2002-04-03 to 2026-02-15 | 8718 | ETA: 2441, NAM: 6277 |
| KEWR | GFS | 2000-05-31 to 2026-02-15 | 9370 | AVN: 1273, GFS: 8097 |
| KEWR | NAM | 2002-04-03 to 2026-02-15 | 8718 | ETA: 2441, NAM: 6277 |

### KNYC MOS Data Availability

| Model | Date Range | Total Days |
|-------|------------|------------|
| GFS | 2004-01-01 to 2026-02-11 | 8078 |
| NAM | 2004-02-25 to 2026-02-11 | 8021 |

## 2. Similarity Analysis (Overlap Period 2004-2026)


### GFS MOS

| Metric | KJFK | KLGA | KEWR |
|--------|-----|-----|-----|
| N overlap days | 8072 | 8072 | 8072 |
| Correlation | 0.9895 | 0.9942 | 0.9954 |
| Mean bias (AP-KNYC) | -0.66 | 0.37 | 1.37 |
| MAE vs KNYC | 2.12 | 1.52 | 1.86 |
| RMSE vs KNYC | 2.73 | 1.98 | 2.28 |
| Airport MAE vs actual | 3.26 | 2.80 | 3.01 |
| KNYC MAE vs actual | 2.69 | 2.69 | 2.69 |
| Skill diff (AP-KNYC) | 0.57 | 0.11 | 0.32 |
| Bias stability (365d std) | 0.533 | 0.393 | 0.273 |
| Bias stability (range) | 2.86 | 2.39 | 3.03 |
| KS statistic | 0.0912 | 0.0671 | 0.1810 |
| KS p-value | 0.0000 | 0.0000 | 0.0000 |

#### Seasonal Bias (Airport - KNYC, GFS)

| Season | KJFK | KLGA | KEWR |
|--------|-----|-----|-----|
| DJF | -0.03 | 0.12 | 0.68 |
| MAM | -2.14 | -0.61 | 0.94 |
| JJA | -1.15 | 1.01 | 2.21 |
| SON | 0.67 | 0.96 | 1.65 |

#### Monthly Bias (Airport - KNYC, GFS)

| Month | KJFK | KLGA | KEWR |
|-------|-----|-----|-----|
| 01 | 0.04 | 0.12 | 0.69 |
| 02 | -0.48 | -0.05 | 0.82 |
| 03 | -1.44 | -0.27 | 1.26 |
| 04 | -2.32 | -1.00 | 0.65 |
| 05 | -2.67 | -0.58 | 0.89 |
| 06 | -1.97 | 0.47 | 1.95 |
| 07 | -1.30 | 1.15 | 2.33 |
| 08 | -0.20 | 1.40 | 2.33 |
| 09 | 0.47 | 1.17 | 1.91 |
| 10 | 0.76 | 1.07 | 1.84 |
| 11 | 0.79 | 0.63 | 1.18 |
| 12 | 0.32 | 0.27 | 0.53 |

### NAM MOS

| Metric | KJFK | KLGA | KEWR |
|--------|-----|-----|-----|
| N overlap days | 8015 | 8015 | 8015 |
| Correlation | 0.9901 | 0.9930 | 0.9942 |
| Mean bias (AP-KNYC) | -0.64 | 0.46 | 1.12 |
| MAE vs KNYC | 1.97 | 1.66 | 1.80 |
| RMSE vs KNYC | 2.59 | 2.18 | 2.29 |
| Airport MAE vs actual | 3.42 | 2.99 | 2.83 |
| KNYC MAE vs actual | 2.80 | 2.80 | 2.80 |
| Skill diff (AP-KNYC) | 0.61 | 0.19 | 0.03 |
| Bias stability (365d std) | 0.384 | 0.740 | 0.256 |
| Bias stability (range) | 3.84 | 3.37 | 1.41 |
| KS statistic | 0.0882 | 0.0735 | 0.1513 |
| KS p-value | 0.0000 | 0.0000 | 0.0000 |

#### Seasonal Bias (Airport - KNYC, NAM)

| Season | KJFK | KLGA | KEWR |
|--------|-----|-----|-----|
| DJF | -0.08 | 0.38 | 0.62 |
| MAM | -1.98 | -0.58 | 0.99 |
| JJA | -0.77 | 1.34 | 1.81 |
| SON | 0.31 | 0.70 | 1.03 |

#### Monthly Bias (Airport - KNYC, NAM)

| Month | KJFK | KLGA | KEWR |
|-------|-----|-----|-----|
| 01 | 0.10 | 0.57 | 0.77 |
| 02 | -0.68 | -0.10 | 0.57 |
| 03 | -1.55 | -0.77 | 0.77 |
| 04 | -2.22 | -1.00 | 0.93 |
| 05 | -2.18 | 0.02 | 1.27 |
| 06 | -1.58 | 0.84 | 1.77 |
| 07 | -0.75 | 1.61 | 1.92 |
| 08 | -0.01 | 1.55 | 1.75 |
| 09 | 0.26 | 1.08 | 1.33 |
| 10 | 0.24 | 0.36 | 0.95 |
| 11 | 0.44 | 0.68 | 0.82 |
| 12 | 0.27 | 0.61 | 0.51 |

## 3. Harmonization Strategy Evaluation

Train period: 2004-2019, Test period: 2020-2023


### GFS MOS Harmonization


#### KJFK (JFK International (~15 mi))

- Train: 5844 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 1.982 | 3.214 | Baseline |
| Constant offset (-0.74F) | 2.052 | 3.163 | Single bias correction |
| Seasonal offset | 1.912 | 2.920 | 4 seasonal offsets |
| Monthly offset | 1.880 | 2.907 | 12 monthly offsets |
| KNYC direct | - | 2.596 | Reference |

Seasonal offsets: DJF: -0.15F, JJA: -1.15F, MAM: -2.27F, SON: 0.62F

#### KLGA (LaGuardia (~8 mi))

- Train: 5844 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 1.578 | 2.627 | Baseline |
| Constant offset (0.32F) | 1.599 | 2.666 | Single bias correction |
| Seasonal offset | 1.541 | 2.581 | 4 seasonal offsets |
| Monthly offset | 1.536 | 2.576 | 12 monthly offsets |
| KNYC direct | - | 2.596 | Reference |

Seasonal offsets: DJF: -0.00F, JJA: 1.02F, MAM: -0.67F, SON: 0.92F

#### KEWR (Newark Liberty (~10 mi))

- Train: 5844 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 2.050 | 2.930 | Baseline |
| Constant offset (1.28F) | 1.512 | 2.811 | Single bias correction |
| Seasonal offset | 1.432 | 2.763 | 4 seasonal offsets |
| Monthly offset | 1.418 | 2.768 | 12 monthly offsets |
| KNYC direct | - | 2.596 | Reference |

Seasonal offsets: DJF: 0.62F, JJA: 2.12F, MAM: 0.88F, SON: 1.47F

#### Multi-Station Average (KJFK, KLGA, KEWR)

| Method | MAE vs KNYC | MAE vs Actual |
|--------|-------------|---------------|
| Raw average | 1.437 | - |
| Constant offset (0.28F) | 1.374 | 2.665 |
| Monthly offset | 1.275 | 2.536 |

### NAM MOS Harmonization


#### KJFK (JFK International (~15 mi))

- Train: 5787 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 1.787 | 3.329 | Baseline |
| Constant offset (-0.68F) | 1.802 | 3.197 | Single bias correction |
| Seasonal offset | 1.733 | 3.031 | 4 seasonal offsets |
| Monthly offset | 1.725 | 3.035 | 12 monthly offsets |
| KNYC direct | - | 2.774 | Reference |

Seasonal offsets: DJF: -0.12F, JJA: -0.79F, MAM: -2.16F, SON: 0.38F

#### KLGA (LaGuardia (~8 mi))

- Train: 5787 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 1.414 | 2.758 | Baseline |
| Constant offset (0.71F) | 1.588 | 3.047 | Single bias correction |
| Seasonal offset | 1.608 | 3.027 | 4 seasonal offsets |
| Monthly offset | 1.608 | 3.031 | 12 monthly offsets |
| KNYC direct | - | 2.774 | Reference |

Seasonal offsets: DJF: 0.58F, JJA: 1.62F, MAM: -0.54F, SON: 1.17F

#### KEWR (Newark Liberty (~10 mi))

- Train: 5787 days, Test: 1461 days

| Method | MAE vs KNYC | MAE vs Actual | Notes |
|--------|-------------|---------------|-------|
| Raw (no harmonization) | 1.755 | 2.624 | Baseline |
| Constant offset (1.17F) | 1.441 | 2.802 | Single bias correction |
| Seasonal offset | 1.445 | 2.837 | 4 seasonal offsets |
| Monthly offset | 1.445 | 2.828 | 12 monthly offsets |
| KNYC direct | - | 2.774 | Reference |

Seasonal offsets: DJF: 0.56F, JJA: 1.90F, MAM: 0.90F, SON: 1.29F

#### Multi-Station Average (KJFK, KLGA, KEWR)

| Method | MAE vs KNYC | MAE vs Actual |
|--------|-------------|---------------|
| Raw average | 1.264 | - |
| Constant offset (0.40F) | 1.267 | 2.834 |
| Monthly offset | 1.291 | 2.793 |

## 4. Pre-Overlap Data Quality (2000-2003)


### KJFK GFS

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2000 | 193 |  |
| 2001 | 365 |  |
| 2002 | 365 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 1288

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2000-2003) | 3.36 | -1.27 | 1288 |
| Overlap (2004+) | 3.26 | -0.98 | 8072 |

### KJFK NAM

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2002 | 273 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 638

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2002-2003) | 3.79 | -1.20 | 638 |
| Overlap (2004+) | 3.42 | -1.52 | 8070 |

### KLGA GFS

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2000 | 193 |  |
| 2001 | 365 |  |
| 2002 | 365 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 1288

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2000-2003) | 2.67 | 0.00 | 1288 |
| Overlap (2004+) | 2.80 | 0.05 | 8072 |

### KLGA NAM

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2002 | 273 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 638

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2002-2003) | 3.15 | 0.62 | 638 |
| Overlap (2004+) | 2.99 | -0.43 | 8070 |

### KEWR GFS

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2000 | 193 |  |
| 2001 | 365 |  |
| 2002 | 365 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 1288

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2000-2003) | 2.91 | 0.80 | 1288 |
| Overlap (2004+) | 3.01 | 1.05 | 8072 |

### KEWR NAM

| Year | Days Available | Model Label |
|------|---------------|-------------|
| 2002 | 273 |  |
| 2003 | 365 |  |

**Total pre-overlap days:** 638

| Period | MAE vs Actual | Bias vs Actual | N Days |
|--------|---------------|----------------|--------|
| Pre-overlap (2002-2003) | 3.39 | 1.42 | 638 |
| Overlap (2004+) | 2.83 | 0.23 | 8070 |

## 5. Recommendation

### Best Single Proxy: **KEWR** (Newark Liberty (~10 mi))

- **Correlation with KNYC:** 0.9954
- **Mean bias:** 1.37F
- **MAE vs KNYC:** 1.86F
- **Best harmonization method:** Monthly offset (MAE vs KNYC on test: 1.418F)

### Multi-Station Average
- Monthly offset MAE vs KNYC: 1.275F
- Monthly offset MAE vs actual: 2.536F

### Recommended Extended Date Ranges

| Model | Current Start | Extended Start | Added Years |
|-------|---------------|----------------|-------------|
| GFS/AVN | 2004-01 | 2000-06* | ~3.5 years |
| NAM/ETA | 2004-02 | 2002-06* | ~1.5 years |

*Exact start depends on data availability at each airport station.

### Data Quality Concerns & Caveats

1. **Model transitions:** AVN→GFS transition (2003/2004) and ETA→NAM transition (~2005-2009) may introduce subtle systematic shifts.
2. **MOS equation updates:** MOS equations are re-derived periodically, which can cause discontinuities.
3. **Airport microclimate:** Airport stations have different local climates than Central Park (urban heat island, proximity to water, runway effects).
4. **Pre-overlap verification:** The harmonization offsets are trained on 2004-2019 but applied to 2000-2003 data that used different NWP models.
5. **Recommendation:** Use monthly harmonization with the best proxy station. Consider adding a small noise term or widening prediction intervals for the pre-2004 extended period to account for increased uncertainty.