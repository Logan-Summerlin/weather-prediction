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

                              model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples
E3_weighted_ensemble_E4_uncertainty          All         0.02      2130       0.25   -30.66   -14.54    0.3638        -38.17         -22.85        -18.04         -10.90               1000
E3_weighted_ensemble_E4_uncertainty          All         0.03      1779       0.25   -24.88   -14.02    0.3687        -31.33         -18.09        -17.74         -10.11               1000
E3_weighted_ensemble_E4_uncertainty          All         0.04      1468       0.25   -19.27   -13.29    0.3685        -25.70         -12.13        -17.84          -8.46               1000
E3_weighted_ensemble_E4_uncertainty          All         0.05      1213       0.25   -17.76   -15.03    0.3561        -23.66         -11.35        -19.74          -9.65               1000
E3_weighted_ensemble_E4_uncertainty           IS         0.02       867       0.25   -14.12   -15.30    0.3875        -19.26          -8.80        -21.10          -9.58               1000
E3_weighted_ensemble_E4_uncertainty           IS         0.03       649       0.25    -9.91   -14.42    0.3898        -14.49          -5.33        -20.97          -7.86               1000
E3_weighted_ensemble_E4_uncertainty           IS         0.04       488       0.25    -5.83   -11.61    0.3914         -9.81          -1.63        -19.69          -3.35               1000
E3_weighted_ensemble_E4_uncertainty           IS         0.05       372       0.25    -4.43   -11.98    0.3763         -8.02          -0.99        -21.18          -2.75               1000
E3_weighted_ensemble_E4_uncertainty          OOS         0.02      1263       0.25   -16.55   -13.95    0.3476        -21.44         -11.39        -18.09          -9.62               1000
E3_weighted_ensemble_E4_uncertainty          OOS         0.03      1130       0.25   -14.97   -13.77    0.3566        -20.08          -9.67        -18.52          -9.04               1000
E3_weighted_ensemble_E4_uncertainty          OOS         0.04       980       0.25   -13.44   -14.17    0.3571        -18.35          -7.98        -19.14          -8.47               1000
E3_weighted_ensemble_E4_uncertainty          OOS         0.05       841       0.25   -13.34   -16.42    0.3472        -17.88          -8.37        -22.14         -10.39               1000
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.02       289       0.25    -1.94    -7.83    0.3391         -4.72           0.64        -18.55           2.63               1000
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.03       256       0.25    -1.71    -7.56    0.3516         -4.05           0.67        -17.88           3.10               1000
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.04       224       0.25    -1.62    -8.22    0.3482         -4.11           1.01        -21.10           5.15               1000
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.05       189       0.25    -1.33    -7.87    0.3545         -3.65           1.03        -21.60           6.49               1000
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.02       316       0.25    -5.00   -15.99    0.3576         -7.11          -2.57        -22.64          -8.28               1000
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.03       269       0.25    -4.45   -16.21    0.3680         -6.58          -2.19        -24.25          -8.19               1000
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.04       226       0.25    -4.01   -17.37    0.3628         -6.14          -1.71        -26.69          -7.59               1000
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.05       188       0.25    -3.87   -20.39    0.3457         -5.91          -1.85        -30.47          -9.90               1000
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.02       318       0.25    -3.06    -9.19    0.4088         -6.17           0.31        -18.75           0.93               1000
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.03       287       0.25    -2.31    -7.49    0.4286         -5.24           0.98        -17.14           3.19               1000
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.04       246       0.25    -2.05    -7.69    0.4309         -5.16           1.36        -19.85           4.92               1000
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.05       211       0.25    -2.89   -12.64    0.4076         -5.53          -0.17        -24.48          -0.74               1000
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.02       340       0.25    -6.55   -22.33    0.2882         -8.85          -3.98        -30.05         -13.72               1000
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.03       318       0.25    -6.49   -23.48    0.2862         -8.71          -4.00        -31.35         -14.80               1000
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.04       284       0.25    -5.75   -22.75    0.2958         -8.01          -3.32        -31.29         -13.54               1000
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.05       253       0.25    -5.24   -23.35    0.2925         -7.55          -3.03        -33.31         -13.81               1000
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.02       380       0.25    -3.51   -10.33    0.3447         -6.08          -0.61        -18.39          -1.80               1000
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.03       329       0.25    -3.03   -10.02    0.3556         -5.80          -0.03        -19.28          -0.10               1000
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.04       282       0.25    -2.95   -11.45    0.3475         -5.60          -0.13        -21.61          -0.51               1000
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.05       235       0.25    -2.74   -12.83    0.3404         -5.27          -0.25        -24.92          -1.10               1000

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
    "days_with_non_unit_probability_mass": 1086,
    "rows_with_unparseable_ticker": 0,
    "rows_with_ticker_date_mismatch": 0,
    "rows_with_ticker_strike_mismatch": 0,
    "rows_with_unexpected_ticker_kind": 0,
    "rows_with_outcome_rule_mismatch": 95
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

## Paper-trading gate report

{
  "top_model": "E3_weighted_ensemble_E4_uncertainty",
  "promotion_ready": false,
  "checks": {
    "oos_brier_beats_presettlement": {
      "pass": false,
      "oos_model_brier": 0.13004572811498297,
      "presettlement_brier": 0.12706111379754997
    },
    "oos_gated_pnl_positive_with_positive_ci": {
      "pass": false,
      "best_oos_gated": {
        "quality_cut": 0.05,
        "trades": 841,
        "net_pnl": -13.34,
        "roi_pct": -16.42,
        "pnl_ci95_low": -17.88,
        "pnl_ci95_high": -8.37
      }
    },
    "calibration_ece_gate": {
      "pass": true,
      "ece": 0.02284814226776027,
      "threshold": 0.03
    },
    "tail_reliability_gate": {
      "pass": false,
      "max_abs_bin_gap": 0.5133113877132489,
      "threshold": 0.2
    }
  }
}
