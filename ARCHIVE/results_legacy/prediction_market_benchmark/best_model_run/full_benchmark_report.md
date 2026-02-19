# Model vs Benchmarks: Comprehensive Comparison Report

Generated: 2026-02-10 20:23

## Overview

This report compares our neural network temperature prediction model against two benchmarks:
1. **Kalshi pre-settlement market prices** - Real prediction market consensus captured before settlement
2. **NWS/MOS forecast** - National Weather Service operational forecast distribution

- Total bucket-level observations: **6,204**
- Unique dates: **1089**
- Date range: 2023-01-01 to 2025-12-31
- IS period (2023-2024): 4,046 observations
- OOS period (2025): 2,158 observations

## 1. Brier Score Comparison (lower = better)

### Overall

| Source | Brier Score | Log Score | N |
|--------|-------------|-----------|---|
| Kalshi_Settled | 0.0181 | 0.0711 | 6,204 |
| Kalshi_PreSettlement | 0.1271 | 0.3882 | 6,204 |
| Model | 0.1335 | 0.4228 | 6,204 |
| NWS | 0.1418 | 0.4499 | 6,204 |

### By Period

| Period | Source | Brier Score | Log Score | N |
|--------|--------|-------------|-----------|---|
| IS | Kalshi_Settled | 0.0266 | 0.0999 | 4,046 |
| IS | Model | 0.1353 | 0.4276 | 4,046 |
| IS | Kalshi_PreSettlement | 0.1421 | 0.4304 | 4,046 |
| IS | NWS | 0.1431 | 0.4546 | 4,046 |
| OOS | Kalshi_Settled | 0.0021 | 0.0171 | 2,158 |
| OOS | Kalshi_PreSettlement | 0.0988 | 0.3093 | 2,158 |
| OOS | Model | 0.1302 | 0.4139 | 2,158 |
| OOS | NWS | 0.1393 | 0.4411 | 2,158 |

### By Season

| Season | Source | Brier Score | Log Score | N |
|--------|--------|-------------|-----------|---|
| Winter | Kalshi_Settled | 0.0180 | 0.0668 | 1,511 |
| Winter | Kalshi_PreSettlement | 0.1173 | 0.3633 | 1,511 |
| Winter | Model | 0.1307 | 0.4148 | 1,511 |
| Winter | NWS | 0.1409 | 0.4474 | 1,511 |
| Spring | Kalshi_Settled | 0.0107 | 0.0513 | 1,605 |
| Spring | Kalshi_PreSettlement | 0.1358 | 0.4113 | 1,605 |
| Spring | Model | 0.1392 | 0.4468 | 1,605 |
| Spring | NWS | 0.1422 | 0.4545 | 1,605 |
| Summer | Kalshi_Settled | 0.0174 | 0.0725 | 1,603 |
| Summer | Model | 0.1302 | 0.4080 | 1,603 |
| Summer | Kalshi_PreSettlement | 0.1319 | 0.4058 | 1,603 |
| Summer | NWS | 0.1398 | 0.4421 | 1,603 |
| Fall | Kalshi_Settled | 0.0270 | 0.0956 | 1,485 |
| Fall | Kalshi_PreSettlement | 0.1222 | 0.3698 | 1,485 |
| Fall | Model | 0.1338 | 0.4210 | 1,485 |
| Fall | NWS | 0.1444 | 0.4560 | 1,485 |

### By Bucket Direction

| Direction | Source | Brier Score | Log Score | N |
|-----------|--------|-------------|-----------|---|
| below | Kalshi_Settled | 0.0045 | 0.0276 | 931 |
| below | Kalshi_PreSettlement | 0.0511 | 0.1788 | 931 |
| below | Model | 0.0593 | 0.2169 | 931 |
| below | NWS | 0.0810 | 0.2906 | 931 |
| between | Kalshi_Settled | 0.0228 | 0.0844 | 4,248 |
| between | Kalshi_PreSettlement | 0.1549 | 0.4645 | 4,248 |
| between | Model | 0.1594 | 0.4952 | 4,248 |
| between | NWS | 0.1650 | 0.5126 | 4,248 |
| above | Kalshi_Settled | 0.0109 | 0.0557 | 1,025 |
| above | Kalshi_PreSettlement | 0.0809 | 0.2625 | 1,025 |
| above | Model | 0.0936 | 0.3099 | 1,025 |
| above | NWS | 0.1007 | 0.3349 | 1,025 |

## 2. Calibration Analysis

### Expected Calibration Error (ECE)

| Source | ECE |
|--------|-----|
| Kalshi_Settled | 0.0174 |
| Model | 0.0230 |
| NWS | 0.0324 |
| Kalshi_PreSettlement | 0.0557 |

### Reliability Diagram Data (10 bins)

**Model**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.041 | 0.050 | 2027 |
| 0.15 | 0.150 | 0.135 | 1471 |
| 0.25 | 0.249 | 0.259 | 1816 |
| 0.35 | 0.345 | 0.329 | 611 |
| 0.45 | 0.435 | 0.257 | 109 |
| 0.55 | 0.548 | 0.361 | 72 |
| 0.65 | 0.649 | 0.114 | 35 |
| 0.75 | 0.742 | 0.387 | 31 |
| 0.85 | 0.844 | 0.500 | 20 |
| 0.95 | 0.945 | 0.583 | 12 |

**Kalshi_PreSettlement**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.038 | 0.017 | 2242 |
| 0.15 | 0.144 | 0.115 | 999 |
| 0.25 | 0.251 | 0.222 | 874 |
| 0.35 | 0.346 | 0.329 | 733 |
| 0.45 | 0.450 | 0.327 | 621 |
| 0.55 | 0.531 | 0.291 | 540 |
| 0.65 | 0.639 | 0.481 | 131 |
| 0.75 | 0.747 | 0.632 | 38 |
| 0.85 | 0.831 | 0.786 | 14 |
| 0.95 | 0.938 | 0.917 | 12 |

**NWS**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.051 | 0.064 | 1493 |
| 0.15 | 0.158 | 0.162 | 2460 |
| 0.25 | 0.240 | 0.266 | 1734 |
| 0.35 | 0.331 | 0.179 | 262 |
| 0.45 | 0.442 | 0.220 | 109 |
| 0.55 | 0.542 | 0.141 | 64 |
| 0.65 | 0.653 | 0.302 | 43 |
| 0.75 | 0.747 | 0.080 | 25 |
| 0.85 | 0.834 | 0.583 | 12 |
| 0.95 | 0.980 | 0.000 | 2 |

**Kalshi_Settled**

| Bin | Mean Predicted | Mean Observed | Count |
|-----|---------------|---------------|-------|
| 0.05 | 0.015 | 0.002 | 4740 |
| 0.15 | 0.133 | 0.059 | 205 |
| 0.25 | 0.245 | 0.167 | 114 |
| 0.35 | 0.347 | 0.377 | 53 |
| 0.45 | 0.444 | 0.434 | 53 |
| 0.55 | 0.532 | 0.627 | 83 |
| 0.65 | 0.666 | 0.750 | 36 |
| 0.75 | 0.755 | 0.625 | 24 |
| 0.85 | 0.839 | 0.778 | 27 |
| 0.95 | 0.984 | 0.990 | 869 |

## 3. Trading Simulation: Model vs Pre-Settlement Market

Fee rate: 7% on winnings

### Model as Signal

| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |
|--------|-----------|--------|----------|---------|------|--------|
| All | 0.02 | 5183 | 48.0% | $104.31 | 4.7% | 0.059 |
| IS | 0.02 | 3472 | 54.9% | $152.92 | 9.4% | 0.124 |
| OOS | 0.02 | 1711 | 34.0% | $-48.61 | -8.2% | -0.090 |
| All | 0.05 | 4215 | 48.1% | $145.25 | 8.3% | 0.097 |
| IS | 0.05 | 2819 | 55.6% | $180.67 | 14.2% | 0.174 |
| OOS | 0.05 | 1396 | 33.1% | $-35.42 | -7.6% | -0.078 |
| All | 0.10 | 3013 | 50.0% | $170.02 | 13.8% | 0.150 |
| IS | 0.10 | 1989 | 58.7% | $201.58 | 22.8% | 0.267 |
| OOS | 0.10 | 1024 | 33.3% | $-31.56 | -9.1% | -0.088 |
| All | 0.15 | 2109 | 52.9% | $178.02 | 20.7% | 0.220 |
| IS | 0.15 | 1446 | 62.0% | $195.46 | 30.6% | 0.354 |
| OOS | 0.15 | 663 | 33.0% | $-17.45 | -7.9% | -0.073 |
| All | 0.20 | 1478 | 56.2% | $174.06 | 29.1% | 0.305 |
| IS | 0.20 | 1065 | 65.5% | $185.06 | 39.9% | 0.457 |
| OOS | 0.20 | 413 | 32.0% | $-11.01 | -8.2% | -0.073 |

### NWS as Signal

| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |
|--------|-----------|--------|----------|---------|------|--------|
| All | 0.02 | 5433 | 43.1% | $76.64 | 3.6% | 0.042 |
| IS | 0.02 | 3608 | 49.2% | $123.58 | 8.1% | 0.097 |
| OOS | 0.02 | 1825 | 31.2% | $-46.94 | -8.1% | -0.083 |
| All | 0.05 | 4533 | 44.1% | $98.84 | 5.6% | 0.062 |
| IS | 0.05 | 3019 | 50.3% | $136.22 | 10.7% | 0.123 |
| OOS | 0.05 | 1514 | 31.6% | $-37.38 | -7.7% | -0.077 |
| All | 0.10 | 3341 | 46.3% | $141.62 | 10.9% | 0.116 |
| IS | 0.10 | 2223 | 53.5% | $170.06 | 18.2% | 0.204 |
| OOS | 0.10 | 1118 | 32.1% | $-28.44 | -7.8% | -0.075 |
| All | 0.15 | 2420 | 49.8% | $155.59 | 16.1% | 0.169 |
| IS | 0.15 | 1626 | 58.1% | $178.15 | 25.5% | 0.283 |
| OOS | 0.15 | 794 | 32.9% | $-22.57 | -8.5% | -0.081 |
| All | 0.20 | 1753 | 51.6% | $149.13 | 21.5% | 0.217 |
| IS | 0.20 | 1196 | 61.4% | $170.62 | 33.3% | 0.362 |
| OOS | 0.20 | 557 | 30.7% | $-21.49 | -11.9% | -0.108 |

## 4. Key Findings

- **Best overall Brier score**: Kalshi_Settled (0.0181)
- Model Brier: 0.1335
- NWS Brier: 0.1418
- Pre-settlement market Brier: 0.1271

- Model BEATS NWS by 0.0083 Brier points
- Pre-settlement market BEATS Model by 0.0065 Brier points

- Best Model trading: threshold=0.10, P&L=$201.58, ROI=22.8%, 1989 trades (Model_IS)
- Best NWS trading: threshold=0.15, P&L=$178.15, ROI=25.5%, 1626 trades (NWS_IS)

## 5. Pre-Settlement vs Settled Market Comparison

The pre-settlement prices are the market consensus BEFORE the event resolves.
The settled prices reflect the final market state at settlement.

- Pre-settlement market Brier: 0.1271
- Settled market Brier: 0.0181
- Difference: 0.1090 (pre-settlement is worse)

Pre-settlement prices reflect genuine forecasting uncertainty, while settled prices
often approach 0 or 1 as the outcome becomes known. This makes pre-settlement the
more meaningful benchmark for comparing forecast quality.
