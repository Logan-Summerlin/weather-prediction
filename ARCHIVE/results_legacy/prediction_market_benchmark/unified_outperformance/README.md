# Unified Outperformance Benchmark Report

Generated: 2026-02-13 16:04

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
- Rows with WGA V2: 4,046

## 1. Brier Score Comparison (lower = better)

| Variant | Overall | IS | OOS | LogScore | ECE | OOS ECE |
|---------|---------|-----|-----|----------|-----|---------|
| Kalshi_Settled | 0.0181 | 0.0266 | 0.0021 | 0.0711 | 0.0174 | 0.0141 |
| U7_regime_conditional | 0.1137 | 0.1185 | 0.1047 | 0.3559 | 0.0115 | 0.0229 |
| U6_platt_on_u5 | 0.1141 | 0.1191 | 0.1046 | 0.3567 | 0.0085 | 0.0147 |
| U9_kitchen_sink | 0.1145 | 0.1192 | 0.1059 | 0.3580 | 0.0114 | 0.0198 |
| U4_extended_cal_synthesis | 0.1149 | 0.1218 | 0.1019 | 0.3627 | 0.0278 | 0.0313 |
| U8_2023only_cal_brier_mlp | 0.1152 | 0.1205 | 0.1051 | 0.3615 | 0.0111 | 0.0234 |
| U5_extended_cal_brier_mlp | 0.1154 | 0.1208 | 0.1053 | 0.3611 | 0.0153 | 0.0190 |
| Kalshi_PreSettlement | 0.1271 | 0.1421 | 0.0988 | 0.3882 | 0.0557 | 0.0249 |
| U2_extended_cal_iso_flat | 0.1296 | 0.1322 | 0.1247 | 0.4111 | 0.0081 | 0.0232 |
| U3_extended_cal_iso_wga | 0.1314 | 0.1350 | 0.1246 | 0.4175 | 0.0103 | 0.0297 |
| U0_flat_raw | 0.1336 | 0.1354 | 0.1301 | 0.4254 | 0.0242 | 0.0337 |
| U1_wga_raw | 0.1351 | 0.1377 | 0.1301 | 0.4294 | 0.0205 | 0.0337 |
| NWS | 0.1418 | 0.1431 | 0.1393 | 0.4499 | 0.0324 | 0.0475 |

### By Season (Overall Brier)

| Variant | Winter | Spring | Summer | Fall |
|---------|--------|--------|--------|--------|
| Kalshi_Settled | 0.0180 | 0.0107 | 0.0174 | 0.0270 |
| U7_regime_conditional | 0.1084 | 0.1174 | 0.1187 | 0.1097 |
| U6_platt_on_u5 | 0.1077 | 0.1190 | 0.1176 | 0.1114 |
| U9_kitchen_sink | 0.1071 | 0.1205 | 0.1193 | 0.1106 |
| U4_extended_cal_synthesis | 0.1079 | 0.1187 | 0.1203 | 0.1118 |
| U8_2023only_cal_brier_mlp | 0.1091 | 0.1196 | 0.1188 | 0.1126 |
| U5_extended_cal_brier_mlp | 0.1083 | 0.1202 | 0.1198 | 0.1127 |
| Kalshi_PreSettlement | 0.1173 | 0.1358 | 0.1319 | 0.1222 |
| U2_extended_cal_iso_flat | 0.1231 | 0.1354 | 0.1285 | 0.1312 |
| U3_extended_cal_iso_wga | 0.1270 | 0.1354 | 0.1294 | 0.1336 |

### Brier Decomposition (Top 5)

| Variant | Brier | Reliability | Resolution | Uncertainty |
|---------|-------|------------|------------|-------------|
| Kalshi_Settled | 0.018106 | 0.000679 | 0.124213 | 0.141453 |
| U7_regime_conditional | 0.113719 | 0.000712 | 0.027844 | 0.141453 |
| U6_platt_on_u5 | 0.114081 | 0.000151 | 0.026919 | 0.141453 |
| U9_kitchen_sink | 0.114545 | 0.000407 | 0.026557 | 0.141453 |
| U4_extended_cal_synthesis | 0.114857 | 0.001023 | 0.027225 | 0.141453 |

## 2. Extended Calibration Impact (U5 vs U8)

- U5 (extended cal) OOS Brier: 0.1053
- U8 (2023-only cal) OOS Brier: 0.1051
- 2023-only calibration IMPROVES OOS by 0.0002 Brier points

## 3. Trading Simulation

Fee rate: 7% on winnings

### OOS Trading Summary (best threshold per variant)

| Variant | Threshold | Trades | Win Rate | Net P&L | ROI% | Ann. Sharpe | CI Low | CI High |
|---------|-----------|--------|----------|---------|------|-------------|--------|---------|
| U4_extended_cal_synthesis | 0.20 | 27 | 44.4% | $-0.19 | -1.7% | -0.285 | $-4.29 | $3.96 |
| U5_extended_cal_brier_mlp | 0.20 | 48 | 33.3% | $-4.16 | -21.9% | -3.935 | $-9.52 | $0.74 |
| U8_2023only_cal_brier_mlp | 0.20 | 37 | 24.3% | $-4.26 | -33.7% | -5.621 | $-8.61 | $-0.12 |
| U9_kitchen_sink | 0.20 | 58 | 36.2% | $-5.42 | -21.7% | -4.042 | $-11.50 | $0.42 |
| U7_regime_conditional | 0.20 | 41 | 31.7% | $-6.61 | -35.4% | -7.930 | $-11.42 | $-2.16 |
| U6_platt_on_u5 | 0.20 | 53 | 24.5% | $-6.77 | -35.9% | -6.322 | $-12.06 | $-1.89 |
| U2_extended_cal_iso_flat | 0.20 | 416 | 33.2% | $-16.12 | -11.2% | -1.710 | $-28.97 | $-2.99 |
| U3_extended_cal_iso_wga | 0.20 | 421 | 32.8% | $-16.23 | -11.2% | -1.704 | $-28.99 | $-2.94 |
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
| 0.02 | 1038 | $-41.95 | -10.9% | 35.6% | -0.1154 | $-61.44 | $-19.44 |
| 0.03 | 905 | $-38.28 | -11.3% | 35.8% | -0.1177 | $-57.17 | $-19.19 |
| 0.04 | 773 | $-31.93 | -10.8% | 36.9% | -0.1114 | $-51.32 | $-12.93 |
| 0.05 | 645 | $-28.53 | -11.3% | 37.2% | -0.1189 | $-44.72 | $-11.62 |
| 0.06 | 546 | $-19.66 | -9.4% | 37.4% | -0.0975 | $-35.27 | $-3.67 |

### U4_extended_cal_synthesis

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 394 | $-21.43 | -13.3% | 38.1% | -0.1424 | $-34.26 | $-7.10 |
| 0.03 | 295 | $-13.62 | -12.4% | 34.9% | -0.1217 | $-26.19 | $-1.30 |
| 0.04 | 229 | $-14.27 | -17.4% | 31.9% | -0.1671 | $-23.69 | $-3.72 |
| 0.05 | 172 | $-9.25 | -14.9% | 33.1% | -0.1423 | $-19.55 | $0.36 |
| 0.06 | 137 | $-7.12 | -15.1% | 31.4% | -0.1372 | $-16.17 | $2.54 |

### U5_extended_cal_brier_mlp

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 537 | $-31.85 | -12.0% | 46.7% | -0.1533 | $-51.07 | $-9.62 |
| 0.03 | 390 | $-23.61 | -12.3% | 46.4% | -0.1560 | $-41.31 | $-6.29 |
| 0.04 | 296 | $-11.69 | -8.0% | 48.6% | -0.1035 | $-27.13 | $3.67 |
| 0.05 | 248 | $-9.56 | -7.9% | 48.4% | -0.1008 | $-22.57 | $3.33 |
| 0.06 | 180 | $-8.01 | -9.1% | 47.8% | -0.1171 | $-18.78 | $2.59 |

### U6_platt_on_u5

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 564 | $-33.73 | -12.9% | 43.4% | -0.1577 | $-54.23 | $-13.53 |
| 0.03 | 410 | $-22.18 | -11.9% | 43.2% | -0.1406 | $-38.97 | $-5.87 |
| 0.04 | 299 | $-10.91 | -7.7% | 46.8% | -0.0940 | $-23.99 | $3.32 |
| 0.05 | 230 | $-8.32 | -7.9% | 45.2% | -0.0947 | $-19.13 | $3.95 |
| 0.06 | 154 | $-8.44 | -12.4% | 41.6% | -0.1468 | $-17.64 | $0.69 |

### U7_regime_conditional

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 450 | $-27.26 | -11.5% | 50.2% | -0.1602 | $-43.46 | $-9.79 |
| 0.03 | 309 | $-17.89 | -11.5% | 47.9% | -0.1502 | $-32.86 | $-3.44 |
| 0.04 | 238 | $-15.64 | -12.9% | 47.9% | -0.1729 | $-27.88 | $-2.79 |
| 0.05 | 189 | $-10.54 | -10.9% | 49.2% | -0.1507 | $-21.99 | $0.23 |
| 0.06 | 149 | $-10.99 | -13.9% | 49.0% | -0.1981 | $-20.27 | $-1.40 |

### U8_2023only_cal_brier_mlp

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 502 | $-39.70 | -17.4% | 40.2% | -0.2196 | $-56.38 | $-23.71 |
| 0.03 | 372 | $-32.76 | -18.9% | 40.6% | -0.2425 | $-46.35 | $-18.42 |
| 0.04 | 276 | $-23.09 | -17.8% | 41.7% | -0.2216 | $-35.33 | $-9.69 |
| 0.05 | 191 | $-11.11 | -12.1% | 45.6% | -0.1537 | $-21.09 | $-0.38 |
| 0.06 | 136 | $-2.70 | -4.2% | 49.3% | -0.0543 | $-10.93 | $6.14 |

### U9_kitchen_sink

| Base Cut | Trades | Net P&L | ROI% | Win Rate | Sharpe | CI Low | CI High |
|----------|--------|---------|------|----------|--------|--------|---------|
| 0.02 | 567 | $-35.37 | -12.6% | 46.6% | -0.1673 | $-54.69 | $-15.27 |
| 0.03 | 405 | $-27.44 | -13.6% | 46.4% | -0.1772 | $-43.81 | $-10.61 |
| 0.04 | 298 | $-13.81 | -9.4% | 48.0% | -0.1220 | $-28.05 | $0.41 |
| 0.05 | 242 | $-14.73 | -12.5% | 45.9% | -0.1580 | $-27.97 | $-1.37 |
| 0.06 | 189 | $-16.66 | -18.1% | 42.9% | -0.2329 | $-28.13 | $-4.90 |

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

- **Best overall Brier**: Kalshi_Settled (0.0181)
- **Best OOS Brier**: Kalshi_Settled (0.0021)
- No variants pass all promotion gates
