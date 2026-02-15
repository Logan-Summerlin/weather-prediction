# Cross-City Brier Scale Audit

Raw Brier numbers are not directly comparable when each row means something different.

| city | benchmark | evaluation unit | rows/day | raw Brier | daily aggregate proxy | row-level uniform baseline | skill vs uniform |
|---|---|---|---:|---:|---:|---:|---:|
| nyc | U7_regime_conditional | binary contract-row | 5.996 | 0.113719 | 0.681847 | 0.250000 | 54.51% |
| phl | Real_NWS_MOS | multiclass bucket-day row | 57.000 | 0.014284 | 0.814212 | 0.017236 | 17.12% |
| chi | Real_NWS_MOS | multiclass bucket-day row | 62.000 | 0.013514 | 0.837852 | 0.015869 | 14.84% |

## Interpretation
- NYC row-level Brier is binary-contract scale (baseline 0.25).
- PHL/CHI row-level Brier is 57/62-class scale (uniform baselines ~0.017/0.016).
- Multiplying row-level Brier by rows/day produces a rough daily aggregate proxy showing NYC/PHL/CHI are all in the same ballpark (~0.68-0.84), not an order-of-magnitude apart.
