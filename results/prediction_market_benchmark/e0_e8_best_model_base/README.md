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

                              model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E3_weighted_ensemble_E4_uncertainty          All         0.02      2043       0.25   -29.46   -14.69    0.3603        -36.23         -21.78        -17.95         -10.93               1000              0.0438            0.1589                21.10
E3_weighted_ensemble_E4_uncertainty          All         0.03      1714       0.25   -22.93   -13.44    0.3705        -29.35         -15.68        -17.16          -9.11               1000              0.0344            0.1380                20.00
E3_weighted_ensemble_E4_uncertainty          All         0.04      1404       0.25   -18.51   -13.50    0.3632        -25.00         -11.89        -18.10          -8.67               1000              0.0288            0.1222                19.28
E3_weighted_ensemble_E4_uncertainty          All         0.05      1156       0.25   -17.01   -15.23    0.3521        -22.60         -11.21        -20.14         -10.17               1000              0.0243            0.1092                18.70
E3_weighted_ensemble_E4_uncertainty          All         0.06       953       0.25   -12.88   -14.09    0.3547        -18.06          -7.81        -19.72          -8.56               1000              0.0193            0.0965                18.15
E3_weighted_ensemble_E4_uncertainty           IS         0.02       793       0.25   -13.38   -16.05    0.3796        -18.43          -8.30        -22.29          -9.99               1000              0.0825            0.3202                27.42
E3_weighted_ensemble_E4_uncertainty           IS         0.03       589       0.25    -8.22   -13.22    0.3939        -12.66          -3.75        -20.26          -6.02               1000              0.0668            0.2953                25.76
E3_weighted_ensemble_E4_uncertainty           IS         0.04       433       0.25    -4.97   -11.46    0.3811         -8.65          -1.12        -19.81          -2.61               1000              0.0563            0.2735                24.66
E3_weighted_ensemble_E4_uncertainty           IS         0.05       328       0.25    -4.17   -13.20    0.3598         -7.47          -0.75        -24.06          -2.49               1000              0.0481            0.2485                23.55
E3_weighted_ensemble_E4_uncertainty           IS         0.06       247       0.25    -2.43   -10.63    0.3563         -5.10           0.34        -22.83           1.53               1000              0.0374            0.2264                22.50
E3_weighted_ensemble_E4_uncertainty          OOS         0.02      1250       0.25   -16.08   -13.72    0.3480        -21.38         -10.61        -18.14          -9.18               1000              0.0192            0.0565                17.10
E3_weighted_ensemble_E4_uncertainty          OOS         0.03      1125       0.25   -14.71   -13.57    0.3582        -19.65          -9.31        -18.23          -8.60               1000              0.0175            0.0556                16.99
E3_weighted_ensemble_E4_uncertainty          OOS         0.04       971       0.25   -13.54   -14.44    0.3553        -18.55          -8.20        -19.75          -8.78               1000              0.0165            0.0547                16.88
E3_weighted_ensemble_E4_uncertainty          OOS         0.05       828       0.25   -12.84   -16.04    0.3490        -17.82          -8.00        -22.02         -10.04               1000              0.0149            0.0540                16.78
E3_weighted_ensemble_E4_uncertainty          OOS         0.06       706       0.25   -10.45   -15.24    0.3541        -14.99          -5.90        -21.48          -8.59               1000              0.0129            0.0510                16.63
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.02       284       0.25    -1.86    -7.61    0.3415         -4.38           0.96        -17.87           3.96               1000              0.0202            0.0706                17.52
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.03       255       0.25    -1.67    -7.41    0.3529         -4.10           1.11        -18.07           5.04               1000              0.0189            0.0712                17.45
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.04       220       0.25    -1.54    -7.91    0.3500         -3.95           0.96        -19.48           5.16               1000              0.0180            0.0704                17.40
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.05       186       0.25    -1.54    -9.23    0.3495         -3.71           0.97        -21.51           5.66               1000              0.0175            0.0706                17.29
E3_weighted_ensemble_E4_uncertainty      OOS_DJF         0.06       155       0.25    -1.39   -10.16    0.3419         -3.68           0.82        -26.74           6.13               1000              0.0150            0.0654                17.07
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.02       310       0.25    -4.90   -16.08    0.3548         -7.17          -2.43        -23.45          -7.96               1000              0.0207            0.0640                17.44
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.03       267       0.25    -4.36   -15.92    0.3708         -6.50          -2.04        -24.32          -7.44               1000              0.0179            0.0620                17.26
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.04       224       0.25    -4.05   -17.90    0.3571         -6.19          -1.80        -27.53          -7.97               1000              0.0173            0.0587                16.99
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.05       182       0.25    -3.29   -17.86    0.3571         -5.14          -1.23        -27.77          -6.68               1000              0.0157            0.0569                16.89
E3_weighted_ensemble_E4_uncertainty      OOS_MAM         0.06       151       0.25    -2.15   -13.97    0.3775         -4.13          -0.35        -26.36          -2.25               1000              0.0133            0.0552                16.76
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.02       316       0.25    -2.81    -8.50    0.4114         -6.02           0.62        -18.19           1.91               1000              0.0216            0.0487                16.95
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.03       286       0.25    -2.26    -7.34    0.4301         -5.45           1.05        -17.37           3.45               1000              0.0199            0.0491                16.88
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.04       244       0.25    -2.25    -8.50    0.4262         -5.17           0.93        -19.40           3.59               1000              0.0189            0.0486                16.82
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.05       209       0.25    -2.88   -12.72    0.4067         -5.66          -0.19        -25.31          -0.77               1000              0.0164            0.0467                16.67
E3_weighted_ensemble_E4_uncertainty      OOS_JJA         0.06       181       0.25    -2.31   -11.58    0.4199         -4.78           0.19        -23.86           0.92               1000              0.0146            0.0437                16.53
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.02       340       0.25    -6.51   -22.23    0.2882         -8.87          -4.09        -30.09         -14.34               1000              0.0148            0.0452                16.57
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.03       317       0.25    -6.41   -23.24    0.2871         -8.56          -3.99        -30.87         -14.88               1000              0.0139            0.0436                16.50
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.04       283       0.25    -5.70   -22.60    0.2968         -7.96          -3.35        -31.15         -13.41               1000              0.0125            0.0446                16.45
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.05       251       0.25    -5.13   -22.97    0.2948         -7.35          -2.76        -32.44         -12.67               1000              0.0110            0.0457                16.41
E3_weighted_ensemble_E4_uncertainty      OOS_SON         0.06       219       0.25    -4.59   -23.58    0.2922         -6.79          -2.45        -34.23         -12.82               1000              0.0099            0.0440                16.33
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.02       372       0.25    -3.46   -10.35    0.3468         -6.23          -0.61        -18.47          -1.84               1000              0.0222            0.0770                17.88
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.03       326       0.25    -2.92    -9.70    0.3589         -5.70          -0.24        -19.35          -0.80               1000              0.0198            0.0766                17.73
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.04       280       0.25    -2.90   -11.41    0.3464         -5.69          -0.08        -22.46          -0.32               1000              0.0194            0.0746                17.54
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.05       228       0.25    -2.33   -11.26    0.3465         -4.78          -0.09        -23.28          -0.44               1000              0.0189            0.0736                17.44
E3_weighted_ensemble_E4_uncertainty OOS_volatile         0.06       186       0.25    -1.60    -9.71    0.3441         -3.84           0.45        -22.83           2.66               1000              0.0165            0.0690                17.24

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
        "quality_cut": 0.06,
        "trades": 706,
        "net_pnl": -10.45,
        "roi_pct": -15.24,
        "pnl_ci95_low": -14.99,
        "pnl_ci95_high": -5.9
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
