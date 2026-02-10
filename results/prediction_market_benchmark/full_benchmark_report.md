# Model vs Benchmarks: Comprehensive Comparison Report

Generated: 2026-02-10 17:17

## Overview

This report compares our neural network temperature prediction model against two benchmarks:
1. **Kalshi pre-settlement market prices** - Real prediction market consensus captured before settlement
2. **NWS/MOS forecast** - National Weather Service operational forecast distribution

- Total bucket-level observations: **4,261**
- Unique dates: **765**
- Date range: 2023-01-01 to 2025-02-09
- IS period (2023-2024): 4,046 observations
- OOS period (2025): 215 observations

### Kalshi Market Schedule & Snapshot Timing

The KXHIGHNY market for a given observation day t is open from **10:00 AM ET on day t-1** through **11:59 PM ET on day t** (~38 hours). Settlement occurs the following morning (~10:00 AM ET, day t+1) against the official NWS Daily Climate Report.

Our "pre-settlement" snapshots capture the **last available candlestick from the evening/night of day t-1**, specifically from the window 5:00 PM ET (day t-1) through 12:30 AM ET (day t). This is 6-18 hours before the peak daytime temperature is observed on day t.

**Verified snapshot distribution (4,286 rows with timestamps):**

| ET Time (evening of day t-1) | Rows | % |
|------------------------------|------|---|
| 5:00-6:00 PM | 132 | 3.1% |
| 7:00-9:00 PM | 486 | 11.3% |
| 10:00-11:00 PM | 1,247 | 29.1% |
| 12:00 AM (midnight) | 2,421 | 56.5% |

**Zero snapshots come from daytime on day t.** A dead zone from ~3am-8am ET (no trading activity) separates our overnight snapshots from the next day's trading. This confirms the pre-settlement prices are genuine overnight forecasts with no contamination from day-of temperature observations.

## 1. Brier Score Comparison (lower = better)

### Overall

| Source | Brier Score | Log Score | N |
|--------|-------------|-----------|---|
| Kalshi_Settled | 0.0253 | 0.0956 | 4,261 |
| Kalshi_PreSettlement | 0.1384 | 0.4200 | 4,261 |
| NWS | 0.1407 | 0.4478 | 4,261 |
| Model | 0.1793 | 0.5729 | 4,261 |

### By Period

| Period | Source | Brier Score | Log Score | N |
|--------|--------|-------------|-----------|---|
| IS | Kalshi_Settled | 0.0266 | 0.0999 | 4,046 |
| IS | Kalshi_PreSettlement | 0.1421 | 0.4304 | 4,046 |
| IS | NWS | 0.1431 | 0.4546 | 4,046 |
| IS | Model | 0.1825 | 0.5828 | 4,046 |
| OOS | Kalshi_Settled | 0.0003 | 0.0130 | 215 |
| OOS | Kalshi_PreSettlement | 0.0679 | 0.2246 | 215 |
| OOS | NWS | 0.0962 | 0.3188 | 215 |
| OOS | Model | 0.1187 | 0.3877 | 215 |

### By Season

| Season | Source | Brier Score | Log Score | N |
|--------|--------|-------------|-----------|---|
| Winter | Kalshi_Settled | 0.0203 | 0.0753 | 1,211 |
| Winter | Kalshi_PreSettlement | 0.1207 | 0.3723 | 1,211 |
| Winter | NWS | 0.1395 | 0.4447 | 1,211 |
| Winter | Model | 0.1766 | 0.5568 | 1,211 |
| Spring | Kalshi_Settled | 0.0160 | 0.0708 | 1,054 |
| Spring | NWS | 0.1385 | 0.4434 | 1,054 |
| Spring | Kalshi_PreSettlement | 0.1516 | 0.4535 | 1,054 |
| Spring | Model | 0.1852 | 0.5994 | 1,054 |
| Summer | Kalshi_Settled | 0.0263 | 0.1037 | 1,051 |
| Summer | NWS | 0.1417 | 0.4515 | 1,051 |
| Summer | Kalshi_PreSettlement | 0.1449 | 0.4434 | 1,051 |
| Summer | Model | 0.1734 | 0.5506 | 1,051 |
| Fall | Kalshi_Settled | 0.0410 | 0.1400 | 945 |
| Fall | Kalshi_PreSettlement | 0.1391 | 0.4177 | 945 |
| Fall | NWS | 0.1438 | 0.4524 | 945 |
| Fall | Model | 0.1828 | 0.5891 | 945 |

### By Bucket Direction

| Direction | Source | Brier Score | Log Score | N |
|-----------|--------|-------------|-----------|---|
| below | Kalshi_Settled | 0.0067 | 0.0357 | 608 |
| below | Kalshi_PreSettlement | 0.0586 | 0.2002 | 608 |
| below | NWS | 0.0610 | 0.2401 | 608 |
| below | Model | 0.1338 | 0.4232 | 608 |
| between | Kalshi_Settled | 0.0317 | 0.1134 | 2,952 |
| between | NWS | 0.1613 | 0.5014 | 2,952 |
| between | Kalshi_PreSettlement | 0.1614 | 0.4821 | 2,952 |
| between | Model | 0.1762 | 0.5758 | 2,952 |
| above | Kalshi_Settled | 0.0147 | 0.0722 | 701 |
| above | Kalshi_PreSettlement | 0.1104 | 0.3491 | 701 |
| above | NWS | 0.1232 | 0.4021 | 701 |
| above | Model | 0.2317 | 0.6908 | 701 |

## 2. Calibration Analysis

### Expected Calibration Error (ECE)

| Source | ECE |
|--------|-----|
| Kalshi_Settled | 0.0197 |
| NWS | 0.0245 |
| Kalshi_PreSettlement | 0.0768 |
| Model | 0.1397 |

### Reliability Diagram Data (10 bins)

**Model**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.059 | 0.162 | 1606 |
| 0.15 | 0.136 | 0.209 | 1834 |
| 0.25 | 0.238 | 0.112 | 214 |
| 0.35 | 0.350 | 0.123 | 155 |
| 0.45 | 0.448 | 0.079 | 127 |
| 0.55 | 0.550 | 0.089 | 101 |
| 0.65 | 0.645 | 0.054 | 93 |
| 0.75 | 0.746 | 0.088 | 57 |
| 0.85 | 0.849 | 0.133 | 45 |
| 0.95 | 0.945 | 0.414 | 29 |

**Kalshi_PreSettlement**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.043 | 0.020 | 1262 |
| 0.15 | 0.144 | 0.108 | 752 |
| 0.25 | 0.251 | 0.218 | 666 |
| 0.35 | 0.346 | 0.317 | 539 |
| 0.45 | 0.454 | 0.293 | 426 |
| 0.55 | 0.530 | 0.239 | 464 |
| 0.65 | 0.640 | 0.417 | 108 |
| 0.75 | 0.746 | 0.643 | 28 |
| 0.85 | 0.831 | 0.667 | 9 |
| 0.95 | 0.942 | 1.000 | 7 |

**NWS**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.055 | 0.062 | 928 |
| 0.15 | 0.158 | 0.157 | 1769 |
| 0.25 | 0.240 | 0.262 | 1223 |
| 0.35 | 0.329 | 0.201 | 179 |
| 0.45 | 0.441 | 0.236 | 72 |
| 0.55 | 0.540 | 0.184 | 38 |
| 0.65 | 0.654 | 0.367 | 30 |
| 0.75 | 0.746 | 0.143 | 14 |
| 0.85 | 0.839 | 0.714 | 7 |
| 0.95 | 0.992 | 0.000 | 1 |

**Kalshi_Settled**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.016 | 0.003 | 3128 |
| 0.15 | 0.133 | 0.054 | 202 |
| 0.25 | 0.245 | 0.153 | 111 |
| 0.35 | 0.347 | 0.377 | 53 |
| 0.45 | 0.445 | 0.442 | 52 |
| 0.55 | 0.532 | 0.627 | 83 |
| 0.65 | 0.665 | 0.743 | 35 |
| 0.75 | 0.752 | 0.714 | 21 |
| 0.85 | 0.837 | 0.760 | 25 |
| 0.95 | 0.981 | 0.984 | 551 |

## 3. Trading Simulation: Model vs Pre-Settlement Market

Fee rate: 7% on winnings

### Model as Signal

| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |
|--------|-----------|--------|----------|---------|------|--------|
| All | 0.02 | 3906 | 51.9% | $54.34 | 3.0% | 0.040 |
| IS | 0.02 | 3713 | 53.0% | $49.43 | 2.8% | 0.037 |
| OOS | 0.02 | 193 | 32.1% | $4.91 | 9.3% | 0.104 |
| All | 0.05 | 3468 | 52.0% | $67.52 | 4.2% | 0.053 |
| IS | 0.05 | 3298 | 52.9% | $63.00 | 4.0% | 0.052 |
| OOS | 0.05 | 170 | 34.7% | $4.51 | 9.0% | 0.105 |
| All | 0.10 | 2780 | 53.3% | $90.29 | 7.0% | 0.084 |
| IS | 0.10 | 2653 | 54.0% | $85.30 | 6.8% | 0.083 |
| OOS | 0.10 | 127 | 38.6% | $4.99 | 12.3% | 0.141 |
| All | 0.15 | 2327 | 53.1% | $106.95 | 10.3% | 0.118 |
| IS | 0.15 | 2224 | 53.5% | $101.53 | 10.1% | 0.116 |
| OOS | 0.15 | 103 | 43.7% | $5.42 | 14.9% | 0.177 |
| All | 0.20 | 1927 | 52.4% | $115.49 | 14.0% | 0.152 |
| IS | 0.20 | 1847 | 52.9% | $111.42 | 14.0% | 0.151 |
| OOS | 0.20 | 80 | 40.0% | $4.07 | 15.9% | 0.170 |

### NWS as Signal

| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |
|--------|-----------|--------|----------|---------|------|--------|
| All | 0.02 | 3791 | 48.2% | $125.98 | 8.0% | 0.095 |
| IS | 0.02 | 3608 | 49.2% | $123.58 | 8.1% | 0.097 |
| OOS | 0.02 | 183 | 29.0% | $2.40 | 5.1% | 0.053 |
| All | 0.05 | 3167 | 49.5% | $139.13 | 10.6% | 0.121 |
| IS | 0.05 | 3019 | 50.3% | $136.22 | 10.7% | 0.123 |
| OOS | 0.05 | 148 | 32.4% | $2.91 | 7.0% | 0.073 |
| All | 0.10 | 2330 | 52.6% | $173.93 | 18.0% | 0.201 |
| IS | 0.10 | 2223 | 53.5% | $170.06 | 18.2% | 0.204 |
| OOS | 0.10 | 107 | 33.6% | $3.87 | 13.1% | 0.125 |
| All | 0.15 | 1702 | 57.2% | $182.00 | 25.1% | 0.279 |
| IS | 0.15 | 1626 | 58.1% | $178.15 | 25.5% | 0.283 |
| OOS | 0.15 | 76 | 39.5% | $3.85 | 16.0% | 0.166 |
| All | 0.20 | 1255 | 60.4% | $173.96 | 32.8% | 0.354 |
| IS | 0.20 | 1196 | 61.4% | $170.62 | 33.3% | 0.362 |
| OOS | 0.20 | 59 | 40.7% | $3.34 | 17.6% | 0.176 |

## 4. Key Findings

- **Best overall Brier score**: Kalshi_Settled (0.0253)
- Model Brier: 0.1793
- NWS Brier: 0.1407
- Pre-settlement market Brier: 0.1384

- NWS BEATS Model by 0.0386 Brier points
- Pre-settlement market BEATS Model by 0.0409 Brier points

- Best Model trading: threshold=0.20, P&L=$115.49, ROI=14.0%, 1927 trades (Model_All)
- Best NWS trading: threshold=0.15, P&L=$182.00, ROI=25.1%, 1702 trades (NWS_All)

## 5. Pre-Settlement vs Settled Market Comparison

The pre-settlement prices are the market consensus BEFORE the event resolves.
The settled prices reflect the final market state at settlement.

- Pre-settlement market Brier: 0.1384
- Settled market Brier: 0.0253
- Difference: 0.1131 (pre-settlement is worse)

Pre-settlement prices reflect genuine forecasting uncertainty, while settled prices
often approach 0 or 1 as the outcome becomes known. This makes pre-settlement the
more meaningful benchmark for comparing forecast quality.
