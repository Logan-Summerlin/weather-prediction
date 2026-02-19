# Airport MOS Proxy Harmonization Study

**Generated:** 2026-02-13 16:04:36

**Objective:** Assess feasibility of using airport MOS stations (KLGA, KJFK, KEWR)
as proxies for KNYC MOS to extend training data from 2004 back to 2000.

## 1. Data Summary

### Airport MOS Data

| Station | Distance | Model | Date Range | Total Days |
|---------|----------|-------|------------|------------|
| KLGA | 8 mi | GFS | 2000-05-31 to 2026-02-15 | 9370 |
| KLGA | 8 mi | NAM | 2002-04-03 to 2026-02-15 | 8718 |
| KJFK | 15 mi | GFS | 2000-05-31 to 2026-02-15 | 9370 |
| KJFK | 15 mi | NAM | 2002-04-03 to 2026-02-15 | 8718 |
| KEWR | 10 mi | GFS | 2000-05-31 to 2026-02-15 | 9370 |
| KEWR | 10 mi | NAM | 2002-04-03 to 2026-02-15 | 8718 |

### KNYC MOS Data

| Model | Date Range | Total Days |
|-------|------------|------------|
| GFS | 2004-01-01 to 2026-02-11 | 8078 |
| NAM | 2004-02-25 to 2026-02-11 | 8021 |

**Overlap period:** 2004 onwards

**Holdout validation years:** [2004, 2005]

**Harmonization training years:** 2006-2023

**Pre-overlap extension target:** 2000-2003

## 2. Bias Analysis (Overlap Period)


### GFS MOS

| Metric | KLGA | KJFK | KEWR |
|--------|--------|--------|--------|
| N overlap days | 8078 | 8078 | 8078 |
| Correlation | 0.9942 | 0.9895 | 0.9954 |
| Mean bias (AP-KNYC) | +0.37 | -0.66 | +1.37 |
| Std of differences | 1.95 | 2.65 | 1.82 |
| MAE vs KNYC | 1.52 | 2.12 | 1.86 |
| KS statistic | 0.0225 | 0.0314 | 0.0509 |
| KS p-value | 0.0331 | 0.0007 | 0.0000 |
| Rolling 365d bias range | 1.78 | 1.88 | 1.22 |
| Rolling 365d bias stability (std) | 0.396 | 0.521 | 0.268 |

#### Seasonal Bias (GFS, Airport - KNYC)

| Season | KLGA | KJFK | KEWR |
|--------|--------|--------|--------|
| DJF | +0.11 | -0.03 | +0.68 |
| MAM | -0.61 | -2.14 | +0.94 |
| JJA | +1.01 | -1.15 | +2.21 |
| SON | +0.96 | +0.67 | +1.65 |

### NAM MOS

| Metric | KLGA | KJFK | KEWR |
|--------|--------|--------|--------|
| N overlap days | 8021 | 8021 | 8021 |
| Correlation | 0.9930 | 0.9901 | 0.9942 |
| Mean bias (AP-KNYC) | +0.46 | -0.64 | +1.12 |
| Std of differences | 2.13 | 2.52 | 2.00 |
| MAE vs KNYC | 1.66 | 1.97 | 1.80 |
| KS statistic | 0.0300 | 0.0227 | 0.0425 |
| KS p-value | 0.0014 | 0.0322 | 0.0000 |
| Rolling 365d bias range | 2.25 | 1.94 | 1.27 |
| Rolling 365d bias stability (std) | 0.730 | 0.334 | 0.258 |

#### Seasonal Bias (NAM, Airport - KNYC)

| Season | KLGA | KJFK | KEWR |
|--------|--------|--------|--------|
| DJF | +0.38 | -0.07 | +0.62 |
| MAM | -0.58 | -1.98 | +0.99 |
| JJA | +1.34 | -0.77 | +1.81 |
| SON | +0.70 | +0.31 | +1.03 |

**KLGA is the closest proxy at ~8 miles from Central Park.** It generally shows the smallest MAE vs KNYC among the three airports.

## 3. Harmonization Layer

Harmonization parameters fitted on overlap years 2006-2023, excluding holdout years [2004, 2005].


### KLGA GFS

- **Global offset (KNYC - airport):** -0.357 F
- **Global variance ratio (std KNYC / std airport):** 0.9902
- **Training days:** 6574

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | -0.155 F | 1.0252 |
| MAM | +0.619 F | 1.0334 |
| JJA | -0.982 F | 0.9839 |
| SON | -0.911 F | 1.0096 |

| Month | Offset |
|-------|--------|
| 01 | -0.099 F |
| 02 | -0.033 F |
| 03 | +0.138 F |
| 04 | +1.144 F |
| 05 | +0.591 F |
| 06 | -0.478 F |
| 07 | -1.176 F |
| 08 | -1.278 F |
| 09 | -0.939 F |
| 10 | -1.043 F |
| 11 | -0.746 F |
| 12 | -0.323 F |

### KLGA NAM

- **Global offset (KNYC - airport):** -0.552 F
- **Global variance ratio (std KNYC / std airport):** 0.9827
- **Training days:** 6572

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | -0.496 F | 1.0390 |
| MAM | +0.562 F | 0.9951 |
| JJA | -1.457 F | 0.9755 |
| SON | -0.819 F | 0.9984 |

| Month | Offset |
|-------|--------|
| 01 | -0.728 F |
| 02 | +0.091 F |
| 03 | +0.871 F |
| 04 | +0.913 F |
| 05 | -0.088 F |
| 06 | -0.957 F |
| 07 | -1.706 F |
| 08 | -1.690 F |
| 09 | -1.113 F |
| 10 | -0.498 F |
| 11 | -0.856 F |
| 12 | -0.797 F |

### KJFK GFS

- **Global offset (KNYC - airport):** +0.645 F
- **Global variance ratio (std KNYC / std airport):** 1.0442
- **Training days:** 6574

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | -0.009 F | 1.0954 |
| MAM | +2.136 F | 1.1301 |
| JJA | +1.133 F | 1.0892 |
| SON | -0.708 F | 1.0697 |

| Month | Offset |
|-------|--------|
| 01 | +0.022 F |
| 02 | +0.364 F |
| 03 | +1.244 F |
| 04 | +2.385 F |
| 05 | +2.789 F |
| 06 | +1.954 F |
| 07 | +1.287 F |
| 08 | +0.186 F |
| 09 | -0.411 F |
| 10 | -0.776 F |
| 11 | -0.933 F |
| 12 | -0.378 F |

### KJFK NAM

- **Global offset (KNYC - airport):** +0.618 F
- **Global variance ratio (std KNYC / std airport):** 1.0332
- **Training days:** 6572

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | +0.041 F | 1.1257 |
| MAM | +2.007 F | 1.0957 |
| JJA | +0.749 F | 1.0714 |
| SON | -0.350 F | 1.0600 |

| Month | Offset |
|-------|--------|
| 01 | -0.127 F |
| 02 | +0.644 F |
| 03 | +1.643 F |
| 04 | +2.167 F |
| 05 | +2.217 F |
| 06 | +1.496 F |
| 07 | +0.783 F |
| 08 | -0.009 F |
| 09 | -0.215 F |
| 10 | -0.255 F |
| 11 | -0.581 F |
| 12 | -0.339 F |

### KEWR GFS

- **Global offset (KNYC - airport):** -1.346 F
- **Global variance ratio (std KNYC / std airport):** 0.9718
- **Training days:** 6574

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | -0.680 F | 0.9917 |
| MAM | -0.878 F | 1.0093 |
| JJA | -2.184 F | 0.9581 |
| SON | -1.631 F | 0.9760 |

| Month | Offset |
|-------|--------|
| 01 | -0.651 F |
| 02 | -0.837 F |
| 03 | -1.281 F |
| 04 | -0.548 F |
| 05 | -0.794 F |
| 06 | -1.900 F |
| 07 | -2.337 F |
| 08 | -2.306 F |
| 09 | -1.839 F |
| 10 | -1.808 F |
| 11 | -1.239 F |
| 12 | -0.568 F |

### KEWR NAM

- **Global offset (KNYC - airport):** -1.127 F
- **Global variance ratio (std KNYC / std airport):** 0.9726
- **Training days:** 6572

| Season | Bias Offset | Variance Ratio |
|--------|-------------|----------------|
| DJF | -0.667 F | 0.9943 |
| MAM | -0.949 F | 0.9652 |
| JJA | -1.842 F | 0.9460 |
| SON | -1.040 F | 0.9794 |

| Month | Offset |
|-------|--------|
| 01 | -0.824 F |
| 02 | -0.524 F |
| 03 | -0.616 F |
| 04 | -0.974 F |
| 05 | -1.256 F |
| 06 | -1.822 F |
| 07 | -1.953 F |
| 08 | -1.751 F |
| 09 | -1.257 F |
| 10 | -0.991 F |
| 11 | -0.874 F |
| 12 | -0.642 F |

## 4. Holdout Validation (Years [2004, 2005])

These years were excluded from harmonization training to test out-of-sample performance.


### KLGA GFS

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.390 | +0.211 | 1.840 | 0.9952 |
| global_offset | 1.435 | -0.146 | 1.833 | 0.9952 |
| seasonal_offset | 1.468 | -0.146 | 1.887 | 0.9948 |
| monthly_offset | 1.458 | -0.146 | 1.875 | 0.9949 |
| seasonal_var_correction | 1.468 | -0.178 | 1.884 | 0.9949 |

### KLGA NAM

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.463 | +0.315 | 1.929 | 0.9942 |
| global_offset | 1.488 | -0.237 | 1.918 | 0.9942 |
| seasonal_offset | 1.404 | -0.241 | 1.816 | 0.9945 |
| monthly_offset | 1.378 | -0.251 | 1.793 | 0.9946 |
| seasonal_var_correction | 1.393 | -0.239 | 1.805 | 0.9946 |

### KJFK GFS

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.904 | -0.648 | 2.411 | 0.9924 |
| global_offset | 1.852 | -0.003 | 2.322 | 0.9924 |
| seasonal_offset | 1.765 | -0.004 | 2.271 | 0.9926 |
| monthly_offset | 1.790 | -0.003 | 2.288 | 0.9924 |
| seasonal_var_correction | 1.769 | -0.182 | 2.293 | 0.9926 |

### KJFK NAM

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.936 | -0.904 | 2.513 | 0.9906 |
| global_offset | 1.871 | -0.286 | 2.362 | 0.9906 |
| seasonal_offset | 1.757 | -0.240 | 2.215 | 0.9917 |
| monthly_offset | 1.733 | -0.253 | 2.166 | 0.9921 |
| seasonal_var_correction | 1.812 | -0.348 | 2.251 | 0.9918 |

### KEWR GFS

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.993 | +1.298 | 2.466 | 0.9941 |
| global_offset | 1.671 | -0.048 | 2.097 | 0.9941 |
| seasonal_offset | 1.659 | -0.047 | 2.105 | 0.9936 |
| monthly_offset | 1.611 | -0.047 | 2.047 | 0.9939 |
| seasonal_var_correction | 1.618 | -0.024 | 2.045 | 0.9939 |

### KEWR NAM

| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |
|--------|-------------|--------------|--------------|-------------|
| raw_no_harmonization | 1.676 | +1.235 | 2.059 | 0.9960 |
| global_offset | 1.296 | +0.108 | 1.650 | 0.9960 |
| seasonal_offset | 1.222 | +0.071 | 1.596 | 0.9958 |
| monthly_offset | 1.223 | +0.073 | 1.586 | 0.9958 |
| seasonal_var_correction | 1.212 | +0.101 | 1.544 | 0.9960 |

## 5. Feasibility Assessment

### Quality Gates

- Mean absolute bias (holdout MAE vs KNYC) <= 1.0 F
- Correlation >= 0.95
- Seasonal drift std <= 0.50 F

### Results by Station

| Station | Model | Pre-overlap Days | Holdout MAE | Correlation | Seasonal Drift | All Gates |
|---------|-------|-----------------|-------------|-------------|----------------|-----------|
| KLGA | GFS | 1288 | 1.458 (N) | 0.9949 (Y) | 0.668 (N) | **FAIL** |
| KLGA | NAM | 638 | 1.378 (N) | 0.9946 (Y) | 0.691 (N) | **FAIL** |
| KJFK | GFS | 1288 | 1.790 (N) | 0.9924 (Y) | 1.073 (N) | **FAIL** |
| KJFK | NAM | 638 | 1.733 (N) | 0.9921 (Y) | 0.872 (N) | **FAIL** |
| KEWR | GFS | 1288 | 1.611 (N) | 0.9939 (Y) | 0.600 (N) | **FAIL** |
| KEWR | NAM | 638 | 1.223 (N) | 0.9958 (Y) | 0.434 (Y) | **FAIL** |

### Pre-overlap Data Completeness (2000-2003)


**KLGA GFS:** 2000-05-31 to 2003-12-31
  - 2000: 193 days (52.7%)
  - 2001: 365 days (100.0%)
  - 2002: 365 days (100.0%)
  - 2003: 365 days (100.0%)

**KLGA NAM:** 2002-04-03 to 2003-12-31
  - 2002: 273 days (74.8%)
  - 2003: 365 days (100.0%)

**KJFK GFS:** 2000-05-31 to 2003-12-31
  - 2000: 193 days (52.7%)
  - 2001: 365 days (100.0%)
  - 2002: 365 days (100.0%)
  - 2003: 365 days (100.0%)

**KJFK NAM:** 2002-04-03 to 2003-12-31
  - 2002: 273 days (74.8%)
  - 2003: 365 days (100.0%)

**KEWR GFS:** 2000-05-31 to 2003-12-31
  - 2000: 193 days (52.7%)
  - 2001: 365 days (100.0%)
  - 2002: 365 days (100.0%)
  - 2003: 365 days (100.0%)

**KEWR NAM:** 2002-04-03 to 2003-12-31
  - 2002: 273 days (74.8%)
  - 2003: 365 days (100.0%)

### Additional Training Days Available

- GFS/AVN: up to 1288 days (best single station)
- NAM/ETA: up to 638 days (best single station)

## 6. Recommendation

**Best GFS proxy:** KLGA (LaGuardia, ~8 mi) -- holdout MAE: 1.458 F
**Best NAM proxy:** KEWR (Newark Liberty, ~10 mi) -- holdout MAE: 1.223 F

### Verdict: FEASIBLE WITH RESERVATIONS

No station/model passes all strict quality gates. However, the harmonized airport MOS may still be useful for training with appropriate uncertainty handling:

**Caveats:**
1. Holdout MAE represents the noise floor added by using harmonized proxy vs native KNYC MOS.
2. Pre-2004 data used AVN/ETA models (predecessors to GFS/NAM) with different error characteristics.
3. MOS equations are updated periodically; pre-2004 equations differ from current ones.
4. Consider using an `mos_era` indicator feature to let the model learn era-specific corrections.
5. The variance correction helps match the spread of KNYC but does not fix conditional biases.

**Practical recommendation:**
- Use monthly-offset harmonization with the best single-station proxy (KLGA for proximity).
- Add an `mos_era` binary indicator (0 = pre-2004 proxy, 1 = native KNYC) as a model feature.
- Widen prediction intervals for the pre-2004 extended period proportional to harmonization MAE.
- The ~1,200 additional GFS training days (2000-2003) represent ~15-18% more data, which may help in tail/extreme temperature regimes where sample size matters most.