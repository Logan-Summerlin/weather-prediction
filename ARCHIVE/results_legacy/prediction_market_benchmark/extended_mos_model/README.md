# Extended-MOS Model E0-E22 Benchmark Results

**Date:** 2026-02-13
**Model:** Extended-MOS retrained model (trained on 2000-2021, calibrated on 2022, IS=2023-2024, OOS=2025)
**Comparison baseline:** Original model (trained on 2000-2022, calibrated on 2023, IS=2023-2024, OOS=2025)

## Key Metrics

| Metric | Value |
|--------|-------|
| PreSettlement Brier | 0.1271 |
| NWS OOS Brier | 0.1393 |

## Top Variants Comparison (Original vs Extended)

| Model | Orig Overall | Ext Overall | Orig OOS | Ext OOS | Direction |
|-------|-------------|------------|----------|---------|-----------|
| E17_contract_brier_synthesis | 0.1141 | 0.1140 | 0.1066 | 0.1056 | OOS BETTER |
| E18_regime_adaptive_ensemble | 0.1239 | 0.1147 | 0.1131 | 0.1050 | OOS BETTER |
| E11_synthesis_stacker_market_aware | 0.1166 | 0.1149 | 0.1054 | 0.1027 | OOS BETTER |
| E13_neural_synthesis_mlp | 0.1162 | 0.1150 | 0.1036 | 0.1055 | OOS WORSE |
| E19_platt_beta_calibration | 0.1164 | 0.1146 | 0.1038 | 0.1058 | OOS WORSE |

## Paper Trading Gate Report

- **Top model:** E17_contract_brier_synthesis
- **Promotion ready:** No
- **OOS Brier beats PreSettlement:** Yes (0.1056 < 0.1271)
- **OOS Gated P&L positive:** No (best: -$3.86 at 0.05 quality cut)
- **Calibration ECE:** 0.0153 (PASS, threshold 0.03)
- **Tail reliability:** 0.255 max bin gap (FAIL, threshold 0.2)

## Summary

The extended-MOS model shows mixed results vs the original:
- Overall Brier scores improve across nearly all top variants (E17, E18, E11, E13, E19)
- OOS Brier improves for E17 (+0.0011), E18 (+0.0081), E11 (+0.0027)
- OOS Brier regresses for E13 (-0.0020), E19 (-0.0020)
- E18 shows the largest improvement: overall 0.1239->0.1147, OOS 0.1131->0.1050
- Trading P&L remains negative for most variants; E11 is the only top variant with positive OOS P&L (+$1.39)
- The extended model's tighter sigma (2.79 vs 3.02) hurts tail reliability calibration
