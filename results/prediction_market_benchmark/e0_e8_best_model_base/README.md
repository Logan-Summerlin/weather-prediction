# E0-E8 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

                              model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17
                 E1_global_isotonic             0.133372           0.141775                     0.127061         0.130500       0.139298                     -123.32                      -18.24
       E4_uncertainty_decomposition             0.133388           0.141775                     0.127061         0.130096       0.139298                     -121.76                      -20.19
                            E5_mdn2             0.133444           0.141775                     0.127061         0.129932       0.139298                     -123.22                      -19.49
               E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
            E7_regularization_sweep             0.133623           0.141775                     0.127061         0.130537       0.139298                     -126.19                      -21.41
            E2_seasonal_calibration             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
    E9_conditional_calibration_grid             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                        E6_quantile             0.134413           0.141775                     0.127061         0.131367       0.139298                     -121.45                      -17.78
           E8_feature_pruning_sweep             0.136353           0.141775                     0.127061         0.134766       0.139298                     -139.56                      -24.47

## Top 2

                              model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17
                 E1_global_isotonic             0.133372           0.141775                     0.127061         0.130500       0.139298                     -123.32                      -18.24

## EV-aware dynamic edge gating (best-Brier model)

                              model period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples
E3_weighted_ensemble_E4_uncertainty    All         0.02      2223       0.25   -24.41   -11.77    0.3540        -31.73         -16.53        -15.36          -8.04               1000
E3_weighted_ensemble_E4_uncertainty    All         0.03      1883       0.25   -21.37   -12.16    0.3526        -28.28         -13.90        -16.10          -7.95               1000
E3_weighted_ensemble_E4_uncertainty    All         0.04      1580       0.25   -15.95   -10.84    0.3570        -22.60          -9.01        -15.24          -6.23               1000
E3_weighted_ensemble_E4_uncertainty    All         0.05      1321       0.25   -14.95   -12.41    0.3437        -21.41          -8.72        -17.42          -7.22               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.02       915       0.25   -10.81   -11.84    0.3781        -16.11          -5.56        -17.57          -6.10               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.03       697       0.25    -8.38   -12.25    0.3702        -13.02          -3.97        -19.10          -5.74               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.04       527       0.25    -4.85    -9.57    0.3738         -8.92          -0.58        -17.45          -1.20               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.05       411       0.25    -3.26    -8.77    0.3552         -6.88           0.25        -18.38           0.65               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.02      1308       0.25   -13.61   -11.72    0.3372        -18.58          -8.43        -15.99          -7.28               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.03      1186       0.25   -12.99   -12.10    0.3423        -18.21          -7.13        -17.02          -6.75               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.04      1053       0.25   -11.10   -11.51    0.3485        -16.74          -5.95        -17.31          -6.20               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.05       910       0.25   -11.69   -14.03    0.3385        -16.36          -6.80        -19.37          -8.35               1000

## Contract/time-safe audit

{
  "contract_alignment": {
    "directions_seen": [
      "above",
      "below",
      "between"
    ],
    "rows_with_invalid_threshold_order": 0,
    "rows_with_missing_between_bounds": 0,
    "rows_with_missing_above_low": 0,
    "rows_with_missing_below_high": 0,
    "days_with_non_unit_probability_mass": 1086
  },
  "time_safety": {
    "decision_cutoff_utc_hour": 5,
    "snapshot_rows_total": 6204,
    "snapshot_rows_after_cutoff": 0,
    "snapshot_rows_after_cutoff_pct": 0.0,
    "snapshot_lag_hours_p10": 0.0,
    "snapshot_lag_hours_p50": 0.0,
    "snapshot_lag_hours_p90": 3.0
  }
}
