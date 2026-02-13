# E0-E12 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

                                model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
   E11_synthesis_stacker_market_aware             0.121762           0.141775                     0.127061         0.110530       0.139298                      -70.75                       -6.98
  E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17
                   E1_global_isotonic             0.133372           0.141775                     0.127061         0.130500       0.139298                     -123.32                      -18.24
         E4_uncertainty_decomposition             0.133388           0.141775                     0.127061         0.130096       0.139298                     -121.76                      -20.19
                              E5_mdn2             0.133444           0.141775                     0.127061         0.129932       0.139298                     -123.22                      -19.49
                 E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
           E10_wga_mdn_regime_mixture             0.133584           0.141775                     0.127061         0.130573       0.139298                     -127.21                      -19.09
              E7_regularization_sweep             0.133623           0.141775                     0.127061         0.130537       0.139298                     -126.19                      -21.41
E12_capacity_sweep_residual_synthesis             0.133770           0.141775                     0.127061         0.130561       0.139298                     -122.27                      -18.55
              E2_seasonal_calibration             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
      E9_conditional_calibration_grid             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                          E6_quantile             0.134413           0.141775                     0.127061         0.131367       0.139298                     -121.45                      -17.78
             E8_feature_pruning_sweep             0.136353           0.141775                     0.127061         0.134766       0.139298                     -139.56                      -24.47

## Top 2

                              model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
 E11_synthesis_stacker_market_aware             0.121762           0.141775                     0.127061         0.110530       0.139298                      -70.75                       -6.98
E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17

## EV-aware dynamic edge gating (best-Brier model)

                             model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E11_synthesis_stacker_market_aware          All         0.02      1515       0.25   -20.56   -14.14    0.3545        -27.09         -13.68        -18.44          -9.32               1000              0.0344            0.1326                19.87
E11_synthesis_stacker_market_aware          All         0.03      1190       0.25   -17.12   -14.73    0.3580        -22.65         -11.26        -19.41          -9.90               1000              0.0257            0.1105                18.79
E11_synthesis_stacker_market_aware          All         0.04       865       0.25   -11.22   -13.28    0.3642        -16.51          -6.48        -19.42          -7.62               1000              0.0192            0.0985                18.15
E11_synthesis_stacker_market_aware          All         0.05       646       0.25   -10.63   -17.40    0.3359        -15.16          -6.16        -24.55         -10.18               1000              0.0162            0.0855                17.64
E11_synthesis_stacker_market_aware          All         0.06       468       0.25    -8.86   -20.81    0.3098        -13.00          -4.87        -30.45         -11.34               1000              0.0131            0.0746                17.19
E11_synthesis_stacker_market_aware           IS         0.02       502       0.25    -6.11   -11.93    0.3865         -9.97          -1.88        -19.44          -3.84               1000              0.0689            0.2922                25.85
E11_synthesis_stacker_market_aware           IS         0.03       349       0.25    -4.59   -13.18    0.3725         -7.93          -1.29        -22.60          -3.87               1000              0.0517            0.2474                23.65
E11_synthesis_stacker_market_aware           IS         0.04       227       0.25    -2.24   -10.41    0.3656         -4.78           0.40        -22.01           1.89               1000              0.0370            0.2306                22.45
E11_synthesis_stacker_market_aware           IS         0.05       153       0.25    -1.94   -13.82    0.3399         -4.17           0.40        -29.62           2.72               1000              0.0309            0.1979                21.15
E11_synthesis_stacker_market_aware           IS         0.06       100       0.25    -1.00   -11.83    0.3200         -2.83           0.93        -33.56          10.46               1000              0.0257            0.1612                19.86
E11_synthesis_stacker_market_aware          OOS         0.02      1013       0.25   -14.45   -15.34    0.3386        -19.37          -9.69        -20.54         -10.34               1000              0.0174            0.0536                16.91
E11_synthesis_stacker_market_aware          OOS         0.03       841       0.25   -12.53   -15.40    0.3520        -17.30          -7.44        -21.13          -9.31               1000              0.0148            0.0537                16.77
E11_synthesis_stacker_market_aware          OOS         0.04       638       0.25    -8.98   -14.27    0.3636        -13.03          -5.14        -20.93          -8.15               1000              0.0128            0.0515                16.62
E11_synthesis_stacker_market_aware          OOS         0.05       493       0.25    -8.69   -18.47    0.3347        -12.61          -4.81        -26.87         -10.24               1000              0.0116            0.0506                16.55
E11_synthesis_stacker_market_aware          OOS         0.06       368       0.25    -7.86   -23.03    0.3071        -11.31          -3.99        -33.11         -11.78               1000              0.0097            0.0511                16.47
E11_synthesis_stacker_market_aware      OOS_DJF         0.02       220       0.25    -1.85   -10.06    0.3227         -4.26           0.72        -22.22           4.18               1000              0.0189            0.0698                17.45
E11_synthesis_stacker_market_aware      OOS_DJF         0.03       188       0.25    -1.35    -8.00    0.3564         -3.54           1.10        -21.09           6.40               1000              0.0165            0.0707                17.25
E11_synthesis_stacker_market_aware      OOS_DJF         0.04       135       0.25    -1.29   -10.34    0.3556         -3.37           0.89        -27.26           7.52               1000              0.0155            0.0670                17.07
E11_synthesis_stacker_market_aware      OOS_DJF         0.05        97       0.25    -1.57   -18.33    0.3093         -3.23           0.32        -37.65           3.73               1000              0.0132            0.0649                16.94
E11_synthesis_stacker_market_aware      OOS_DJF         0.06        70       0.25    -1.85   -29.53    0.2714         -3.28          -0.31        -51.31          -4.96               1000              0.0120            0.0709                16.96
E11_synthesis_stacker_market_aware      OOS_MAM         0.02       225       0.25    -4.40   -19.51    0.3467         -6.51          -2.09        -29.01          -9.11               1000              0.0183            0.0568                17.02
E11_synthesis_stacker_market_aware      OOS_MAM         0.03       173       0.25    -2.58   -14.38    0.3815         -4.45          -0.66        -24.32          -3.73               1000              0.0153            0.0571                16.88
E11_synthesis_stacker_market_aware      OOS_MAM         0.04       121       0.25    -2.21   -17.75    0.3636         -4.00          -0.45        -31.66          -3.79               1000              0.0122            0.0577                16.74
E11_synthesis_stacker_market_aware      OOS_MAM         0.05        85       0.25    -1.24   -14.66    0.3647         -2.58           0.24        -30.70           2.88               1000              0.0123            0.0588                16.78
E11_synthesis_stacker_market_aware      OOS_MAM         0.06        59       0.25    -1.21   -22.38    0.3051         -2.57           0.24        -48.55           4.51               1000              0.0113            0.0551                16.69
E11_synthesis_stacker_market_aware      OOS_JJA         0.02       255       0.25    -2.29    -8.58    0.4118         -5.32           0.61        -19.90           2.39               1000              0.0198            0.0486                16.86
E11_synthesis_stacker_market_aware      OOS_JJA         0.03       218       0.25    -2.84   -12.19    0.4037         -5.56           0.03        -23.47           0.12               1000              0.0173            0.0480                16.74
E11_synthesis_stacker_market_aware      OOS_JJA         0.04       167       0.25    -1.19    -6.40    0.4491         -3.49           1.17        -19.21           6.49               1000              0.0151            0.0449                16.58
E11_synthesis_stacker_market_aware      OOS_JJA         0.05       131       0.25    -1.73   -12.10    0.4122         -3.87           0.41        -27.15           2.94               1000              0.0144            0.0443                16.53
E11_synthesis_stacker_market_aware      OOS_JJA         0.06        91       0.25    -1.26   -13.08    0.3956         -3.04           0.76        -32.34           8.16               1000              0.0099            0.0442                16.33
E11_synthesis_stacker_market_aware      OOS_SON         0.02       313       0.25    -5.92   -22.24    0.2843         -8.13          -3.44        -30.27         -13.03               1000              0.0136            0.0438                16.49
E11_synthesis_stacker_market_aware      OOS_SON         0.03       262       0.25    -5.76   -24.82    0.2863         -7.95          -3.26        -34.35         -14.25               1000              0.0113            0.0441                16.39
E11_synthesis_stacker_market_aware      OOS_SON         0.04       215       0.25    -4.29   -22.11    0.3023         -6.36          -2.07        -33.53         -10.95               1000              0.0097            0.0435                16.31
E11_synthesis_stacker_market_aware      OOS_SON         0.05       180       0.25    -4.16   -26.34    0.2778         -6.15          -2.03        -39.01         -12.93               1000              0.0084            0.0435                16.25
E11_synthesis_stacker_market_aware      OOS_SON         0.06       148       0.25    -3.54   -27.60    0.2703         -5.72          -1.39        -44.04         -11.37               1000              0.0079            0.0443                16.24
E11_synthesis_stacker_market_aware OOS_volatile         0.02       272       0.25    -3.48   -14.68    0.3199         -6.09          -0.79        -26.01          -3.43               1000              0.0201            0.0726                17.56
E11_synthesis_stacker_market_aware OOS_volatile         0.03       216       0.25    -1.57    -7.97    0.3611         -3.86           0.76        -19.84           3.72               1000              0.0185            0.0750                17.45
E11_synthesis_stacker_market_aware OOS_volatile         0.04       146       0.25    -1.87   -14.08    0.3356         -3.75           0.06        -28.55           0.48               1000              0.0165            0.0747                17.30
E11_synthesis_stacker_market_aware OOS_volatile         0.05        98       0.25    -1.15   -13.73    0.3163         -2.74           0.38        -34.26           4.53               1000              0.0154            0.0765                17.31
E11_synthesis_stacker_market_aware OOS_volatile         0.06        65       0.25    -1.12   -23.12    0.2462         -2.32           0.08        -48.66           1.50               1000              0.0146            0.0790                17.31

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
  "top_model": "E11_synthesis_stacker_market_aware",
  "promotion_ready": false,
  "checks": {
    "oos_brier_beats_presettlement": {
      "pass": true,
      "oos_model_brier": 0.11053013878012687,
      "presettlement_brier": 0.12706111379754997
    },
    "oos_gated_pnl_positive_with_positive_ci": {
      "pass": false,
      "best_oos_gated": {
        "quality_cut": 0.06,
        "trades": 368,
        "net_pnl": -7.86,
        "roi_pct": -23.03,
        "pnl_ci95_low": -11.31,
        "pnl_ci95_high": -3.99
      }
    },
    "calibration_ece_gate": {
      "pass": false,
      "ece": 0.03143197451024529,
      "threshold": 0.03
    },
    "tail_reliability_gate": {
      "pass": true,
      "max_abs_bin_gap": 0.12860274838532854,
      "threshold": 0.2
    }
  }
}
