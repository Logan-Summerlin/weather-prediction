# Unified Outperformance Benchmark Report

Generated: 2026-06-24 14:06

## Overview

Cross-model synthesis combining the flat NN model and WGA V2 attention model
with extended calibration (full IS period 2023+2024 vs 2023-only baseline).

### Variant Definitions

| Variant | Description |
|---------|-------------|
| U0 | Flat model raw Gaussian bucket probs (baseline) |
| U1 | WGA V2 raw Gaussian bucket probs (baseline) |
| U2 | Extended-cal isotonic on flat model |
| U3 | Extended-cal isotonic on WGA V2 |
| U4 | Extended-cal logistic synthesis stacker |
| U5 | Extended-cal contract-level Brier-optimal MLP |
| U6 | Platt recalibration on U5 |
| U7 | Regime-conditional variance + U5 features |
| U8 | 2023-only cal contract-level Brier MLP (comparison) |
| U9 | Kitchen sink (all features + Platt + regime) |

- Total bucket-level observations: **6,204**
- Unique dates: **1089**
- Date range: 2023-01-01 to 2025-12-31
- IS period (2023-2024): 4,046 rows
- OOS period (2025): 2,158 rows
- Rows with WGA V2: 0

## 1. Brier Score Comparison (lower = better)

| Variant | Overall | IS | OOS | LogScore | ECE | OOS ECE |
|---------|---------|-----|-----|----------|-----|---------|
| Kalshi_Settled | 0.0181 | 0.0266 | 0.0021 | 0.0711 | 0.0174 | 0.0141 |
| Kalshi_PreSettlement | 0.1271 | 0.1421 | 0.0988 | 0.3882 | 0.0557 | 0.0249 |
| U4_extended_cal_synthesis | 0.1146 | 0.1215 | 0.1018 | 0.3626 | 0.0285 | 0.0342 |
| U7_regime_conditional | 0.1119 | 0.1171 | 0.1021 | 0.3512 | 0.0115 | 0.0221 |
| U6_platt_on_u5 | 0.1140 | 0.1198 | 0.1032 | 0.3572 | 0.0082 | 0.0179 |
| U5_extended_cal_brier_mlp | 0.1153 | 0.1215 | 0.1037 | 0.3604 | 0.0185 | 0.0220 |
| U9_kitchen_sink | 0.1127 | 0.1169 | 0.1049 | 0.3533 | 0.0132 | 0.0265 |
| U8_2023only_cal_brier_mlp | 0.1154 | 0.1204 | 0.1059 | 0.3615 | 0.0106 | 0.0194 |
| U2_extended_cal_iso_flat | 0.1296 | 0.1322 | 0.1247 | 0.4111 | 0.0081 | 0.0232 |
| U3_extended_cal_iso_wga | 0.1296 | 0.1322 | 0.1247 | 0.4111 | 0.0081 | 0.0232 |
| U0_flat_raw | 0.1336 | 0.1354 | 0.1301 | 0.4254 | 0.0242 | 0.0337 |
| U1_wga_raw | 0.1336 | 0.1354 | 0.1301 | 0.4254 | 0.0242 | 0.0337 |
| NWS | 0.1418 | 0.1431 | 0.1393 | 0.4499 | 0.0324 | 0.0475 |

### By Season (Overall Brier)

| Variant | Winter | Spring | Summer | Fall |
|---------|--------|--------|--------|--------|
| Kalshi_Settled | 0.0180 | 0.0107 | 0.0174 | 0.0270 |
| Kalshi_PreSettlement | 0.1173 | 0.1358 | 0.1319 | 0.1222 |
| U4_extended_cal_synthesis | 0.1072 | 0.1197 | 0.1195 | 0.1115 |
| U7_regime_conditional | 0.1063 | 0.1142 | 0.1183 | 0.1082 |
| U6_platt_on_u5 | 0.1111 | 0.1182 | 0.1171 | 0.1092 |
| U5_extended_cal_brier_mlp | 0.1122 | 0.1198 | 0.1187 | 0.1098 |
| U9_kitchen_sink | 0.1073 | 0.1162 | 0.1178 | 0.1091 |
| U8_2023only_cal_brier_mlp | 0.1104 | 0.1199 | 0.1190 | 0.1116 |
| U2_extended_cal_iso_flat | 0.1231 | 0.1354 | 0.1285 | 0.1312 |
| U3_extended_cal_iso_wga | 0.1231 | 0.1354 | 0.1285 | 0.1312 |

### Brier Decomposition (Top 5)

| Variant | Brier | Reliability | Resolution | Uncertainty |
|---------|-------|------------|------------|-------------|
| Kalshi_Settled | 0.018106 | 0.000679 | 0.124213 | 0.141453 |
| Kalshi_PreSettlement | 0.127061 | 0.007612 | 0.021273 | 0.141453 |
| U4_extended_cal_synthesis | 0.11464 | 0.001201 | 0.027794 | 0.141453 |
| U7_regime_conditional | 0.111887 | 0.000468 | 0.028914 | 0.141453 |
| U6_platt_on_u5 | 0.114035 | 0.000532 | 0.027423 | 0.141453 |

## 2. Extended Calibration Impact (U5 vs U8)

- U5 (extended cal) OOS Brier: 0.1037
- U8 (2023-only cal) OOS Brier: 0.1059
- Extended calibration IMPROVES OOS by 0.0022 Brier points

## 3. Trading Simulation

Fee rate: 7% on winnings

### OOS Trading Summary (best threshold per variant)

| Variant | Threshold | Trades | Win Rate | Net P&L | ROI% | Ann. Sharpe | CI Low | CI High |
|---------|-----------|--------|----------|---------|------|-------------|--------|---------|
| U4_extended_cal_synthesis | 0.15 | 77 | 39.0% | $2.03 | 7.8% | 1.060 | $-4.85 | $9.11 |
| U7_regime_conditional | 0.15 | 54 | 40.7% | $-4.13 | -16.8% | -3.356 | $-9.63 | $0.97 |
| U9_kitchen_sink | 0.20 | 35 | 34.3% | $-4.23 | -27.5% | -5.112 | $-8.77 | $-0.22 |
| U6_platt_on_u5 | 0.20 | 37 | 27.0% | $-6.13 | -39.7% | -7.543 | $-10.51 | $-1.92 |
| U5_extended_cal_brier_mlp | 0.15 | 84 | 39.3% | $-7.20 | -19.0% | -3.486 | $-14.81 | $0.69 |
| U8_2023only_cal_brier_mlp | 0.20 | 42 | 16.7% | $-7.75 | -54.4% | -10.212 | $-11.70 | $-3.98 |
| U2_extended_cal_iso_flat | 0.20 | 416 | 33.2% | $-16.12 | -11.2% | -1.710 | $-28.97 | $-2.99 |
| U3_extended_cal_iso_wga | 0.20 | 416 | 33.2% | $-16.12 | -11.2% | -1.710 | $-28.97 | $-2.99 |
| U0_flat_raw | 0.20 | 431 | 32.5% | $-17.24 | -11.7% | -1.768 | $-30.33 | $-3.67 |
| U1_wga_raw | 0.20 | 431 | 32.5% | $-17.24 | -11.7% | -1.768 | $-30.33 | $-3.67 |
| NWS | 0.20 | 557 | 30.7% | $-32.16 | -16.8% | -2.571 | $-46.42 | $-17.03 |

## 4. EV-Aware Quality Gating (OOS)

### U0_flat_raw

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 1039 | $-45.12 | -11.6% | 35.5% | -0.1242 | $-63.65 | $-24.93 |
| 0.03 | 892 | $-37.45 | -11.1% | 36.2% | -0.1166 | $-56.59 | $-17.80 |
| 0.04 | 763 | $-31.37 | -10.9% | 36.3% | -0.1117 | $-50.07 | $-11.80 |
| 0.05 | 648 | $-26.14 | -10.6% | 36.7% | -0.1089 | $-44.23 | $-8.69 |
| 0.06 | 556 | $-20.52 | -9.8% | 36.5% | -0.0992 | $-37.34 | $-4.50 |

### U1_wga_raw

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 1039 | $-45.12 | -11.6% | 35.5% | -0.1242 | $-63.65 | $-24.93 |
| 0.03 | 892 | $-37.45 | -11.1% | 36.2% | -0.1166 | $-56.59 | $-17.80 |
| 0.04 | 763 | $-31.37 | -10.9% | 36.3% | -0.1117 | $-50.07 | $-11.80 |
| 0.05 | 648 | $-26.14 | -10.6% | 36.7% | -0.1089 | $-44.23 | $-8.69 |
| 0.06 | 556 | $-20.52 | -9.8% | 36.5% | -0.0992 | $-37.34 | $-4.50 |

### U2_extended_cal_iso_flat

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 1028 | $-42.70 | -11.0% | 36.1% | -0.1174 | $-62.11 | $-19.66 |
| 0.03 | 883 | $-36.67 | -10.8% | 36.9% | -0.1140 | $-56.26 | $-15.86 |
| 0.04 | 757 | $-33.32 | -11.4% | 36.9% | -0.1185 | $-52.05 | $-14.35 |
| 0.05 | 625 | $-28.72 | -11.7% | 37.4% | -0.1204 | $-46.80 | $-10.98 |
| 0.06 | 535 | $-22.41 | -10.7% | 37.6% | -0.1108 | $-39.22 | $-5.07 |

### U3_extended_cal_iso_wga

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 1028 | $-42.70 | -11.0% | 36.1% | -0.1174 | $-62.11 | $-19.66 |
| 0.03 | 883 | $-36.67 | -10.8% | 36.9% | -0.1140 | $-56.26 | $-15.86 |
| 0.04 | 757 | $-33.32 | -11.4% | 36.9% | -0.1185 | $-52.05 | $-14.35 |
| 0.05 | 625 | $-28.72 | -11.7% | 37.4% | -0.1204 | $-46.80 | $-10.98 |
| 0.06 | 535 | $-22.41 | -10.7% | 37.6% | -0.1108 | $-39.22 | $-5.07 |

### U4_extended_cal_synthesis

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 433 | $-15.62 | -8.9% | 39.5% | -0.0920 | $-29.86 | $-1.94 |
| 0.03 | 326 | $-12.52 | -10.4% | 35.6% | -0.0981 | $-24.50 | $-1.30 |
| 0.04 | 254 | $-9.63 | -10.6% | 34.2% | -0.0974 | $-20.53 | $0.81 |
| 0.05 | 193 | $-5.22 | -7.7% | 34.7% | -0.0689 | $-15.70 | $4.67 |
| 0.06 | 156 | $-4.01 | -7.4% | 34.6% | -0.0650 | $-13.97 | $5.65 |

### U5_extended_cal_brier_mlp

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 436 | $-16.38 | -6.3% | 59.9% | -0.0989 | $-34.61 | $3.17 |
| 0.03 | 304 | $-4.96 | -2.9% | 59.9% | -0.0425 | $-19.87 | $9.29 |
| 0.04 | 226 | $-4.43 | -3.4% | 59.7% | -0.0518 | $-17.40 | $8.96 |
| 0.05 | 166 | $-5.23 | -5.6% | 57.2% | -0.0821 | $-16.06 | $5.44 |
| 0.06 | 127 | $-4.00 | -5.9% | 54.3% | -0.0819 | $-13.01 | $4.97 |

### U6_platt_on_u5

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 377 | $-12.91 | -6.2% | 56.0% | -0.0901 | $-28.47 | $3.14 |
| 0.03 | 256 | $-9.36 | -6.5% | 56.6% | -0.0987 | $-21.51 | $3.50 |
| 0.04 | 191 | $-5.25 | -4.8% | 58.1% | -0.0734 | $-16.85 | $6.07 |
| 0.05 | 146 | $-2.67 | -3.4% | 56.2% | -0.0456 | $-13.21 | $6.93 |
| 0.06 | 105 | $-2.70 | -4.9% | 53.3% | -0.0669 | $-10.53 | $5.75 |

### U7_regime_conditional

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 418 | $-12.72 | -5.8% | 52.9% | -0.0785 | $-29.87 | $3.32 |
| 0.03 | 300 | $-12.06 | -7.8% | 51.0% | -0.1028 | $-27.40 | $2.81 |
| 0.04 | 208 | $-9.73 | -9.3% | 49.0% | -0.1213 | $-20.98 | $2.57 |
| 0.05 | 162 | $-7.01 | -8.7% | 48.8% | -0.1157 | $-17.28 | $2.99 |
| 0.06 | 117 | $-6.45 | -10.7% | 49.6% | -0.1396 | $-15.39 | $2.78 |

### U8_2023only_cal_brier_mlp

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 478 | $-40.21 | -16.9% | 44.4% | -0.2283 | $-56.55 | $-23.54 |
| 0.03 | 331 | $-25.62 | -15.4% | 45.6% | -0.2135 | $-38.68 | $-10.91 |
| 0.04 | 241 | $-16.75 | -14.2% | 45.2% | -0.1870 | $-28.52 | $-4.76 |
| 0.05 | 181 | $-13.93 | -15.6% | 44.8% | -0.2041 | $-23.18 | $-4.22 |
| 0.06 | 140 | $-5.80 | -8.3% | 49.3% | -0.1095 | $-14.75 | $2.66 |

### U9_kitchen_sink

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 442 | $-33.07 | -13.4% | 51.8% | -0.1920 | $-51.63 | $-14.96 |
| 0.03 | 314 | $-26.39 | -15.3% | 50.0% | -0.2115 | $-41.38 | $-10.74 |
| 0.04 | 241 | $-21.28 | -16.1% | 49.4% | -0.2223 | $-35.30 | $-8.69 |
| 0.05 | 183 | $-12.72 | -12.7% | 51.4% | -0.1737 | $-23.74 | $-1.52 |
| 0.06 | 140 | $-8.19 | -10.5% | 53.6% | -0.1470 | $-17.83 | $1.47 |

## 5. Paper-Trading Promotion Gate

PreSettlement OOS Brier: 0.098839

| Variant | OOS Brier <= Pre | Gated P&L Pos CI | ECE <= 0.03 | Tail Gap <= 0.20 | ALL PASS |
|---------|-----------------|------------------|------------|-----------------|----------|
| Kalshi_PreSettlement | FAIL | FAIL | PASS | FAIL | **FAIL** |
| NWS | FAIL | FAIL | FAIL | PASS | **FAIL** |
| Kalshi_Settled | PASS | FAIL | PASS | FAIL | **FAIL** |
| U0_flat_raw | FAIL | FAIL | FAIL | PASS | **FAIL** |
| U1_wga_raw | FAIL | FAIL | FAIL | PASS | **FAIL** |
| U2_extended_cal_iso_flat | FAIL | FAIL | PASS | PASS | **FAIL** |
| U3_extended_cal_iso_wga | FAIL | FAIL | PASS | PASS | **FAIL** |
| U4_extended_cal_synthesis | FAIL | FAIL | FAIL | PASS | **FAIL** |
| U5_extended_cal_brier_mlp | FAIL | FAIL | PASS | PASS | **FAIL** |
| U6_platt_on_u5 | FAIL | FAIL | PASS | PASS | **FAIL** |
| U7_regime_conditional | FAIL | FAIL | PASS | PASS | **FAIL** |
| U8_2023only_cal_brier_mlp | FAIL | FAIL | PASS | PASS | **FAIL** |
| U9_kitchen_sink | FAIL | FAIL | PASS | PASS | **FAIL** |

## 6. Key Findings

- **Best OOS Brier**: Kalshi_Settled (0.0021)
- **Best OOS Brier**: Kalshi_Settled (0.0021)
- No variants pass all promotion gates
