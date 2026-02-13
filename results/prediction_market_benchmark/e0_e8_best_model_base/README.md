# E0-E16 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

                                    model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
       E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57
                 E13_neural_synthesis_mlp             0.116842           0.141775                     0.127061         0.104891       0.139298                      -68.79                       -4.55
      E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17
                       E1_global_isotonic             0.133372           0.141775                     0.127061         0.130500       0.139298                     -123.32                      -18.24
             E4_uncertainty_decomposition             0.133388           0.141775                     0.127061         0.130096       0.139298                     -121.76                      -20.19
                                  E5_mdn2             0.133444           0.141775                     0.127061         0.129932       0.139298                     -123.22                      -19.49
                     E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
               E10_wga_mdn_regime_mixture             0.133584           0.141775                     0.127061         0.130573       0.139298                     -127.21                      -19.09
                  E7_regularization_sweep             0.133623           0.141775                     0.127061         0.130537       0.139298                     -126.19                      -21.41
    E12_capacity_sweep_residual_synthesis             0.133770           0.141775                     0.127061         0.130561       0.139298                     -122.27                      -18.55
       E16_conditional_calibration_shrunk             0.134133           0.141775                     0.127061         0.131791       0.139298                     -125.14                      -18.84
          E9_conditional_calibration_grid             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                  E2_seasonal_calibration             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                              E6_quantile             0.134413           0.141775                     0.127061         0.131367       0.139298                     -121.45                      -17.78
E15_conditional_calibration_spread_regime             0.134469           0.141775                     0.127061         0.132387       0.139298                     -124.05                      -18.44
                 E8_feature_pruning_sweep             0.136353           0.141775                     0.127061         0.134766       0.139298                     -139.56                      -24.47
            E14_distributional_neural_nll             0.194393           0.141775                     0.127061         0.191978       0.139298                     -334.78                      -52.97

## Top 2

                             model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57
          E13_neural_synthesis_mlp             0.116842           0.141775                     0.127061         0.104891       0.139298                      -68.79                       -4.55

## EV-aware dynamic edge gating (best-Brier model)

                             model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E11_synthesis_stacker_market_aware          All         0.02      1073       0.25    -9.10   -12.26    0.2610        -13.43          -4.59        -17.81          -6.39               1000              0.0221            0.0893                18.37
E11_synthesis_stacker_market_aware          All         0.03       783       0.25    -9.00   -16.02    0.2593        -13.10          -4.95        -22.99          -8.81               1000              0.0136            0.0748                17.48
E11_synthesis_stacker_market_aware          All         0.04       541       0.25    -6.18   -15.76    0.2625         -9.29          -2.79        -24.20          -7.12               1000              0.0115            0.0722                17.24
E11_synthesis_stacker_market_aware          All         0.05       407       0.25    -4.07   -12.92    0.2899         -7.10          -1.09        -22.45          -3.47               1000              0.0103            0.0718                17.08
E11_synthesis_stacker_market_aware          All         0.06       304       0.25    -3.94   -16.15    0.2895         -6.45          -1.30        -26.86          -5.31               1000              0.0099            0.0667                16.95
E11_synthesis_stacker_market_aware           IS         0.02       291       0.25    -2.41    -9.75    0.3299         -5.07           0.21        -20.31           0.86               1000              0.0497            0.1844                22.97
E11_synthesis_stacker_market_aware           IS         0.03       200       0.25    -2.43   -16.99    0.2550         -4.65          -0.48        -32.27          -3.60               1000              0.0252            0.1444                20.38
E11_synthesis_stacker_market_aware           IS         0.04       129       0.25    -1.03   -11.28    0.2713         -2.90           0.75        -31.18           7.66               1000              0.0231            0.1459                20.04
E11_synthesis_stacker_market_aware           IS         0.05        95       0.25    -0.85   -12.37    0.2737         -2.64           0.68        -37.66           9.72               1000              0.0206            0.1603                19.92
E11_synthesis_stacker_market_aware           IS         0.06        56       0.25    -1.16   -29.34    0.2143         -2.45           0.20        -61.50           5.60               1000              0.0218            0.1689                20.30
E11_synthesis_stacker_market_aware          OOS         0.02       782       0.25    -6.68   -13.51    0.2353        -10.56          -3.27        -21.29          -6.69               1000              0.0118            0.0539                16.65
E11_synthesis_stacker_market_aware          OOS         0.03       583       0.25    -6.58   -15.69    0.2607         -9.86          -3.19        -23.63          -7.51               1000              0.0097            0.0509                16.49
E11_synthesis_stacker_market_aware          OOS         0.04       412       0.25    -5.14   -17.13    0.2597         -7.78          -2.54        -25.84          -8.39               1000              0.0079            0.0491                16.36
E11_synthesis_stacker_market_aware          OOS         0.05       312       0.25    -3.22   -13.07    0.2949         -5.72          -0.85        -23.25          -3.41               1000              0.0071            0.0449                16.22
E11_synthesis_stacker_market_aware          OOS         0.06       248       0.25    -2.78   -13.60    0.3065         -5.19          -0.49        -25.20          -2.38               1000              0.0072            0.0436                16.20
E11_synthesis_stacker_market_aware      OOS_DJF         0.02       161       0.25    -0.28    -3.24    0.2236         -1.56           0.93        -18.32          10.63               1000              0.0081            0.0752                17.03
E11_synthesis_stacker_market_aware      OOS_DJF         0.03       120       0.25    -0.55    -7.60    0.2417         -1.95           0.76        -27.11          10.63               1000              0.0070            0.0739                16.94
E11_synthesis_stacker_market_aware      OOS_DJF         0.04        82       0.25    -0.04    -0.74    0.2683         -1.15           1.03        -22.23          20.63               1000              0.0058            0.0732                16.83
E11_synthesis_stacker_market_aware      OOS_DJF         0.05        52       0.25     0.53    14.64    0.3462         -0.39           1.49        -10.70          42.55               1000              0.0044            0.0568                16.33
E11_synthesis_stacker_market_aware      OOS_DJF         0.06        29       0.25     0.45    24.10    0.3448         -0.39           1.27        -21.46          69.81               1000              0.0025            0.0504                16.12
E11_synthesis_stacker_market_aware      OOS_MAM         0.02        94       0.25    -1.49   -28.53    0.1702         -2.77          -0.15        -52.42          -2.80               1000              0.0104            0.0722                17.00
E11_synthesis_stacker_market_aware      OOS_MAM         0.03        48       0.25    -1.70   -59.41    0.1042         -2.62          -0.72        -90.67         -27.71               1000              0.0087            0.0592                16.67
E11_synthesis_stacker_market_aware      OOS_MAM         0.04        26       0.25    -0.97   -80.66    0.0385         -1.55          -0.27       -100.00         -26.86               1000              0.0044            0.0569                16.33
E11_synthesis_stacker_market_aware      OOS_MAM         0.05         5       0.25    -0.39  -100.00    0.0000         -0.53          -0.24       -100.00        -100.00               1000              0.0032            0.0433                16.01
E11_synthesis_stacker_market_aware      OOS_MAM         0.06         3       0.25    -0.28  -100.00    0.0000         -0.41          -0.11       -100.00        -100.00               1000              0.0044            0.0420                16.04
E11_synthesis_stacker_market_aware      OOS_JJA         0.02       208       0.25    -0.13    -0.90    0.2981         -2.15           1.79        -14.68          12.33               1000              0.0151            0.0460                16.60
E11_synthesis_stacker_market_aware      OOS_JJA         0.03       159       0.25    -0.45    -3.52    0.3333         -2.42           1.47        -18.83          11.95               1000              0.0124            0.0441                16.44
E11_synthesis_stacker_market_aware      OOS_JJA         0.04        99       0.25    -1.19   -14.59    0.3030         -2.48          -0.02        -29.95          -0.20               1000              0.0110            0.0438                16.37
E11_synthesis_stacker_market_aware      OOS_JJA         0.05        83       0.25    -1.17   -16.21    0.3133         -2.48           0.13        -33.28           2.00               1000              0.0105            0.0456                16.38
E11_synthesis_stacker_market_aware      OOS_JJA         0.06        70       0.25    -1.04   -15.72    0.3429         -2.43           0.19        -36.95           3.12               1000              0.0109            0.0471                16.43
E11_synthesis_stacker_market_aware      OOS_SON         0.02       319       0.25    -4.79   -22.73    0.2194         -7.33          -2.58        -34.70         -12.37               1000              0.0119            0.0429                16.39
E11_synthesis_stacker_market_aware      OOS_SON         0.03       256       0.25    -3.87   -20.39    0.2539         -5.98          -1.71        -31.87          -9.32               1000              0.0094            0.0428                16.28
E11_synthesis_stacker_market_aware      OOS_SON         0.04       205       0.25    -2.94   -18.98    0.2634         -4.94          -1.09        -31.23          -7.07               1000              0.0076            0.0410                16.16
E11_synthesis_stacker_market_aware      OOS_SON         0.05       172       0.25    -2.20   -16.44    0.2791         -3.92          -0.43        -29.15          -3.39               1000              0.0065            0.0409                16.11
E11_synthesis_stacker_market_aware      OOS_SON         0.06       146       0.25    -1.92   -16.40    0.2877         -3.54          -0.28        -31.10          -2.32               1000              0.0065            0.0406                16.10
E11_synthesis_stacker_market_aware OOS_volatile         0.02       100       0.25    -0.69   -17.51    0.1400         -1.68           0.35        -45.48           8.71               1000              0.0069            0.1110                17.88
E11_synthesis_stacker_market_aware OOS_volatile         0.03        58       0.25    -0.60   -27.08    0.1207         -1.45           0.25        -69.94          11.13               1000              0.0037            0.1094                17.73
E11_synthesis_stacker_market_aware OOS_volatile         0.04        26       0.25    -0.22   -49.15    0.0385         -0.61           0.07       -100.00          14.73               1000              0.0006            0.1390                18.14
E11_synthesis_stacker_market_aware OOS_volatile         0.05         8       0.25    -0.10   -29.93    0.1250         -0.47           0.21       -100.00          42.98               1000              0.0007            0.1222                17.47
E11_synthesis_stacker_market_aware OOS_volatile         0.06         3       0.25    -0.01  -100.00    0.0000         -0.01          -0.01       -100.00        -100.00               1000              0.0009            0.1515                18.07

## EV-aware dynamic edge gating (E16 shrunk-conditional-calibration challenger)

                             model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E16_conditional_calibration_shrunk          All         0.02      1965       0.25   -27.28   -14.08    0.3644        -35.00         -19.61        -18.11         -10.22               1000              0.0426            0.1308                20.47
E16_conditional_calibration_shrunk          All         0.03      1631       0.25   -21.47   -13.17    0.3734        -28.10         -14.69        -17.15          -8.96               1000              0.0336            0.1134                19.52
E16_conditional_calibration_shrunk          All         0.04      1358       0.25   -16.96   -12.47    0.3770        -22.48         -10.54        -16.84          -7.95               1000              0.0281            0.0990                18.77
E16_conditional_calibration_shrunk          All         0.05      1113       0.25   -15.00   -13.62    0.3675        -20.69          -9.06        -18.68          -8.20               1000              0.0236            0.0903                18.29
E16_conditional_calibration_shrunk          All         0.06       919       0.25   -11.31   -12.78    0.3613        -16.82          -6.15        -18.92          -7.02               1000              0.0197            0.0825                17.87
E16_conditional_calibration_shrunk           IS         0.02       727       0.25   -11.37   -14.82    0.3865        -15.72          -6.37        -20.57          -8.48               1000              0.0825            0.2584                26.24
E16_conditional_calibration_shrunk           IS         0.03       530       0.25    -8.18   -14.84    0.3811        -12.15          -4.31        -22.09          -8.01               1000              0.0665            0.2354                24.84
E16_conditional_calibration_shrunk           IS         0.04       398       0.25    -5.06   -12.59    0.3794         -8.44          -1.72        -20.87          -4.22               1000              0.0565            0.2074                23.43
E16_conditional_calibration_shrunk           IS         0.05       300       0.25    -4.13   -14.23    0.3567         -7.06          -0.94        -24.19          -3.23               1000              0.0468            0.1898                22.43
E16_conditional_calibration_shrunk           IS         0.06       231       0.25    -2.13   -10.04    0.3550         -4.85           0.82        -22.91           4.06               1000              0.0377            0.1726                21.44
E16_conditional_calibration_shrunk          OOS         0.02      1238       0.25   -15.91   -13.60    0.3514        -21.37         -10.10        -18.20          -8.45               1000              0.0191            0.0558                17.08
E16_conditional_calibration_shrunk          OOS         0.03      1101       0.25   -13.29   -12.32    0.3697        -18.64          -7.73        -17.25          -7.16               1000              0.0177            0.0547                16.96
E16_conditional_calibration_shrunk          OOS         0.04       960       0.25   -11.90   -12.42    0.3760        -17.38          -6.74        -17.93          -7.06               1000              0.0163            0.0541                16.85
E16_conditional_calibration_shrunk          OOS         0.05       813       0.25   -10.87   -13.40    0.3715        -15.95          -6.01        -19.72          -7.30               1000              0.0150            0.0535                16.77
E16_conditional_calibration_shrunk          OOS         0.06       688       0.25    -9.19   -13.65    0.3634        -13.87          -4.57        -20.46          -6.88               1000              0.0136            0.0523                16.68
E16_conditional_calibration_shrunk      OOS_DJF         0.02       289       0.25    -1.50    -6.05    0.3460         -3.97           1.18        -15.74           4.99               1000              0.0205            0.0693                17.55
E16_conditional_calibration_shrunk      OOS_DJF         0.03       249       0.25    -1.28    -5.70    0.3655         -3.80           1.45        -16.78           6.62               1000              0.0191            0.0688                17.41
E16_conditional_calibration_shrunk      OOS_DJF         0.04       216       0.25    -1.07    -5.18    0.3889         -3.39           1.32        -16.07           6.82               1000              0.0184            0.0680                17.27
E16_conditional_calibration_shrunk      OOS_DJF         0.05       186       0.25    -0.63    -3.45    0.4086         -3.18           1.73        -16.91           9.84               1000              0.0181            0.0688                17.24
E16_conditional_calibration_shrunk      OOS_DJF         0.06       154       0.25    -1.04    -7.30    0.3701         -3.37           1.37        -23.52           9.53               1000              0.0165            0.0665                17.10
E16_conditional_calibration_shrunk      OOS_MAM         0.02       299       0.25    -5.03   -16.19    0.3746         -7.40          -2.56        -24.29          -7.99               1000              0.0204            0.0625                17.36
E16_conditional_calibration_shrunk      OOS_MAM         0.03       265       0.25    -3.73   -13.26    0.3962         -6.10          -1.40        -21.15          -5.08               1000              0.0187            0.0600                17.17
E16_conditional_calibration_shrunk      OOS_MAM         0.04       220       0.25    -2.76   -11.78    0.4045         -4.94          -0.21        -20.92          -0.90               1000              0.0172            0.0595                17.00
E16_conditional_calibration_shrunk      OOS_MAM         0.05       179       0.25    -3.00   -16.34    0.3687         -5.22          -0.76        -27.89          -4.48               1000              0.0158            0.0568                16.90
E16_conditional_calibration_shrunk      OOS_MAM         0.06       143       0.25    -1.89   -13.30    0.3706         -3.77           0.07        -26.94           0.44               1000              0.0141            0.0574                16.84
E16_conditional_calibration_shrunk      OOS_JJA         0.02       309       0.25    -1.97    -6.13    0.4207         -4.95           1.60        -15.47           5.17               1000              0.0213            0.0488                16.94
E16_conditional_calibration_shrunk      OOS_JJA         0.03       276       0.25    -1.88    -6.23    0.4420         -5.02           1.67        -16.67           5.38               1000              0.0207            0.0496                16.92
E16_conditional_calibration_shrunk      OOS_JJA         0.04       237       0.25    -1.98    -7.57    0.4388         -5.24           1.39        -20.13           5.05               1000              0.0188            0.0484                16.81
E16_conditional_calibration_shrunk      OOS_JJA         0.05       207       0.25    -1.74    -7.53    0.4444         -4.79           1.63        -20.56           7.09               1000              0.0173            0.0462                16.70
E16_conditional_calibration_shrunk      OOS_JJA         0.06       181       0.25    -1.26    -6.20    0.4530         -4.29           1.95        -21.15           9.52               1000              0.0158            0.0452                16.62
E16_conditional_calibration_shrunk      OOS_SON         0.02       341       0.25    -7.41   -25.53    0.2727         -9.61          -4.93        -32.74         -17.14               1000              0.0150            0.0450                16.57
E16_conditional_calibration_shrunk      OOS_SON         0.03       311       0.25    -6.40   -23.61    0.2862         -8.76          -4.00        -32.12         -15.14               1000              0.0132            0.0435                16.46
E16_conditional_calibration_shrunk      OOS_SON         0.04       287       0.25    -6.09   -23.77    0.2927         -8.44          -3.73        -32.56         -14.44               1000              0.0121            0.0443                16.43
E16_conditional_calibration_shrunk      OOS_SON         0.05       241       0.25    -5.50   -25.81    0.2822         -7.96          -2.91        -36.53         -14.13               1000              0.0100            0.0456                16.36
E16_conditional_calibration_shrunk      OOS_SON         0.06       210       0.25    -4.99   -27.02    0.2762         -7.29          -2.65        -39.24         -14.27               1000              0.0093            0.0444                16.31
E16_conditional_calibration_shrunk OOS_volatile         0.02       365       0.25    -3.72   -10.82    0.3616         -6.51          -0.75        -19.09          -2.22               1000              0.0221            0.0755                17.85
E16_conditional_calibration_shrunk OOS_volatile         0.03       322       0.25    -2.78    -9.14    0.3696         -5.48          -0.03        -17.92          -0.12               1000              0.0204            0.0735                17.63
E16_conditional_calibration_shrunk OOS_volatile         0.04       270       0.25    -1.94    -7.41    0.3852         -4.42           0.66        -16.62           2.63               1000              0.0193            0.0737                17.44
E16_conditional_calibration_shrunk OOS_volatile         0.05       225       0.25    -1.90    -8.96    0.3689         -4.24           0.59        -20.21           2.94               1000              0.0195            0.0728                17.41
E16_conditional_calibration_shrunk OOS_volatile         0.06       179       0.25    -1.14    -7.21    0.3520         -3.26           1.36        -21.30           8.21               1000              0.0178            0.0726                17.33

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
      "oos_model_brier": 0.10536368272790153,
      "presettlement_brier": 0.12706111379754997
    },
    "oos_gated_pnl_positive_with_positive_ci": {
      "pass": false,
      "best_oos_gated": {
        "quality_cut": 0.06,
        "trades": 248,
        "net_pnl": -2.78,
        "roi_pct": -13.6,
        "pnl_ci95_low": -5.19,
        "pnl_ci95_high": -0.49
      }
    },
    "calibration_ece_gate": {
      "pass": false,
      "ece": 0.03585406273617443,
      "threshold": 0.03
    },
    "tail_reliability_gate": {
      "pass": true,
      "max_abs_bin_gap": 0.15451372532079088,
      "threshold": 0.2
    }
  }
}
