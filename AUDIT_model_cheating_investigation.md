# Model Cheating Investigation Report

**Date:** 2026-02-16
**Investigator:** Claude (automated audit)
**Trigger:** Suspicion that contract-level Brier scores of 0.03 or lower are artificially inflated

---

## Executive Summary

**The models are NOT using fake or synthetic weather data.** The raw GHCN station data is real, the preprocessing is legitimate, and the E-series models (E0-E5) produce honest bucket-day Brier scores in the 0.014-0.016 range. The weather forecasting pipeline itself is sound.

**However, the Unified synthesis models (U2-U5) ARE cheating** — not through synthetic data, but through a severe **target leakage** problem in the contract-level evaluation pipeline. The Kalshi "market_prob" data used as a training feature is effectively the settlement outcome, making the contract-level Brier scores meaningless.

---

## Finding 1: Kalshi Market Data Contains Settlement-Time Prices (CRITICAL)

### The Problem

The files `data/real_kalshi_chi_all.csv` and `data/real_kalshi_phl_all.csv` contain `market_prob` values that are **settlement-time prices**, not genuine pre-event forecasting prices.

### Evidence

**Philadelphia:**
- 97.1% of `market_prob` values are at extremes (<=0.05 or >=0.95)
- Correlation between `market_prob` and `actual_outcome`: **0.9951** (near-perfect)
- Directional accuracy: **99.89%** (only 3 mismatches in 2,687 rows)
- Brier score of `market_prob` alone: **0.0015** (effectively zero)
- Median bid-ask spread: **1.00** (no active two-sided market)

**Chicago:**
- 86.9% of `market_prob` values are at extremes
- Correlation: **0.93**
- Directional accuracy: **97.5%** (172 mismatches in 6,870 rows)
- Brier score of `market_prob` alone: **0.0186**

### Root Cause

The `market_prob` column appears to contain the **last traded price or final settlement price**, not a meaningful pre-event market probability. At settlement time, contracts converge to 0 or 1 (or very near). These are not forecasts — they are outcomes.

### Affected Files
- `data/real_kalshi_chi_all.csv`
- `data/real_kalshi_phl_all.csv`

---

## Finding 2: Contract-Level Models Use Settlement Prices as Features (CRITICAL)

### The Problem

The `build_contract_features()` function in `scripts/run_chi_phl_unified_benchmark.py` (line 458) uses `market_prob` as a **direct input feature** when training the U2-U5 models to predict `actual_outcome`.

### How It Cheats

```
Feature: market_prob (essentially = actual_outcome since it's settlement data)
Target:  actual_outcome

Result: Model "learns" to copy market_prob → gets near-zero Brier
```

The feature matrix includes:
- `prob` (model probability)
- **`market`** (Kalshi market_prob — near-settlement prices)
- **`prob - market`** (edge — dominated by near-settlement market)
- `volume`
- Various bucket geometry features

Since `market_prob ≈ actual_outcome` in this data, the model trivially learns to predict the outcome by copying the market feature.

### Resulting Fake Brier Scores

| Model | Contract Brier | OOS Brier | Assessment |
|-------|---------------|-----------|------------|
| U2 (CHI) | 0.0203 | **0.0016** | Fake — impossible without leakage |
| U3 (CHI) | 0.0155 | **0.0032** | Fake |
| U4 (CHI) | 0.0154 | **0.0040** | Fake |
| U5 (CHI) | 0.0158 | **0.0033** | Fake |
| U2 (PHL) | **0.0008** | **0.00005** | Absurd — near-zero |
| U3 (PHL) | 0.0047 | **0.0008** | Fake |
| U4 (PHL) | 0.0052 | **0.0005** | Fake |
| U5 (PHL) | 0.0043 | **0.0025** | Fake |

For reference, the U0 (raw Gaussian, no market features) has contract Brier of ~0.19 for both cities, and U1 (isotonic calibration only) has ~0.13-0.14. These are the legitimate scores.

### Affected Scripts
- `scripts/run_chi_phl_unified_benchmark.py`: `build_contract_features()` at line 458
- `scripts/run_real_data_benchmark.py`: `load_city_kalshi_contract_rows()` uses same data

---

## Finding 3: "Kalshi_Settled_Market" Benchmark Uses Settlement Data Against Itself

In `scripts/run_real_data_benchmark.py` (lines 713-717), the "Kalshi_Settled_Market" benchmark computes Brier score using `market_prob` directly as the prediction against `actual_outcome`. Since `market_prob` IS the settlement price:

- CHI: Brier = 0.019 (appears excellent but is just settlement-price variance)
- PHL: Brier = **0.006** (near-zero because PHL settlement data is even more extreme)

This makes all other models appear to have poor Brier scores in comparison, which is misleading.

---

## Finding 4: What Is NOT Cheating

The following components are legitimate:

1. **Raw GHCN data**: Real `.dly` and `.csv` files from NOAA (verified file sizes and station IDs match known GHCN format)
2. **Preprocessing**: Chronological splits (1985-2012 train / 2013-2018 val / 2019-2024 test), column completeness filtering, standard scaling fit on training only
3. **E-series models (E0-E5)**: Bucket-day Brier scores of 0.014-0.016 are reasonable for distributional temperature forecasting across 57-62 buckets
4. **MOS data**: Real GFS/NAM MOS forecasts from IEM archive
5. **MarketProxy / EnhancedMarketProxy**: These are legitimate climatology+persistence regression proxies, not fake data generators
6. **Data collection scripts**: Download from real NOAA/IEM endpoints

---

## Recommendations

### Immediate Actions

1. **Remove `market_prob` from contract-level features** in `build_contract_features()`. Models U2-U5 must not see any Kalshi price data as input features.

2. **Replace settlement Kalshi data** with genuine pre-event market prices (e.g., closing prices 24 hours before event settlement). The current data is useless for evaluating forecasting skill.

3. **Retract all contract-level Brier claims** for U2-U5 models. The only legitimate metrics are:
   - E-series bucket-day Brier (0.014-0.016 range)
   - U0/U1 contract Brier (~0.13-0.19 range)

4. **Remove the "Kalshi_Settled_Market" benchmark** from reports — it's comparing settlement prices against settlement outcomes, which is circular.

### Honest Benchmark Going Forward

The E-series bucket-day Brier scores represent the actual model skill:
- **E2 Ridge**: ~0.0145 (CHI), ~0.0157 (PHL)
- **E3 NN**: ~0.0147 (CHI), ~0.0159 (PHL)
- **E5 Ensemble**: ~0.0146 (CHI), ~0.0159 (PHL)
- **E0 Persistence**: ~0.0150 (CHI), ~0.0161 (PHL)
- **E1 Climatology**: ~0.0152 (CHI), ~0.0163 (PHL)

The models DO beat baselines, but the improvements are modest (~2-5% over persistence), which is realistic for temperature forecasting.

---

## Finding 5: Backtest Scripts Generate Entirely Synthetic Market Data (HIGH)

Both backtest scripts fabricate market data rather than using real Kalshi prices:

- **`scripts/run_chi_backtest.py`**: `generate_chi_market_data()` (line 85) creates synthetic market prices from actual temperatures with added noise (`rng.normal(0, 2.5)`), synthetic spreads, and Poisson-generated volumes.
- **`scripts/run_phl_backtest.py`**: `generate_phl_market_data()` (line 85) does the same for Philadelphia.

These functions:
1. Start from the actual observed TMAX (information the model shouldn't have at forecast time)
2. Add Gaussian noise to create "market" mu/sigma
3. Generate fake bid/ask prices with synthetic spreads
4. Fabricate volume via Poisson distribution

All trading P&L metrics, win rates, and drawdown figures from these backtests are based on fake market data. They cannot be used to validate a trading strategy.

---

## Severity Assessment

| Finding | Severity | Impact |
|---------|----------|--------|
| Settlement prices as features | **CRITICAL** | All U2-U5 contract Brier scores are meaningless |
| Synthetic backtest market data | **HIGH** | All trading P&L/drawdown metrics are fabricated |
| Settlement data as benchmark | **HIGH** | Cross-model comparisons are misleading |
| PHL data quality | **HIGH** | 97% of prices are at extremes; data is nearly useless |
| CHI data quality | **MEDIUM** | 87% at extremes; slightly better but still problematic |
