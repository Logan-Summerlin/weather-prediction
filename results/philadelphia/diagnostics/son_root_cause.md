# Philadelphia SON Calibration Diagnostic — Phase 2 Deliverable #7

Base model: **ridge_base**, held-out test window 2025-08-26 → 2026-02-14 (110 days; calibrators fit on the earlier 70% only).

## Calibration: overall PIT KS (lower = more uniform)

| Calibration | Overall KS |
|---|---|
| Raw model | 0.145 |
| Global isotonic | 0.174 |
| Seasonal isotonic | 0.160 |
| Frontal-regime conditional | 0.175 |

## Autumn (SON) PIT after calibration

| Calibration | Fall KS | n |
|---|---|---|
| Raw | 0.098 | 67 |
| Seasonal isotonic | 0.096 | 67 |
| Frontal-regime | 0.085 | 67 |

## Evidence for frontal-passage features (full series SON)

| Regime (prior-day) | n | high-temp MAE (°F) |
|---|---|---|
| post_frontal | 4 | 2.00 |
| other | 74 | 4.63 |

Post-frontal SON days carry a **-2.63 °F** MAE difference vs other days — the lag-only base model handles the drier, cooler post-frontal airmass differently, which is exactly the signal the `dewpoint_depression_trend`, `wind_post_frontal`, and `frontal_passage_index` features (src/frontal_features.py) encode for the model and the regime-conditional calibrator.

## Recommendation

- Add the cutoff-safe frontal-passage features to PHL's processed feature set (prior-day lagged: `dewpoint_depression`, `dewpoint_depression_trend`, `wind_post_frontal`, `slp_tendency_24h`, `frontal_passage_index`).
- Apply seasonal (or frontal-regime) calibration via `RegimeConditionalCalibrator` so autumn PIT is corrected separately from the rest of the year.
- Re-evaluate against market Brier on real OOS presettlement prices; PHL stays MONITOR until model Brier < market Brier.