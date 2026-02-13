# E0-E13 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

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
      E9_conditional_calibration_grid             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
              E2_seasonal_calibration             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                          E6_quantile             0.134413           0.141775                     0.127061         0.131367       0.139298                     -121.45                      -17.78
             E8_feature_pruning_sweep             0.136353           0.141775                     0.127061         0.134766       0.139298                     -139.56                      -24.47

## Top 2

                             model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57
          E13_neural_synthesis_mlp             0.116842           0.141775                     0.127061         0.104891       0.139298                      -68.79                       -4.55

## EV-aware dynamic edge gating (best-Brier model)

                             model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E11_synthesis_stacker_market_aware          All         0.02      1099       0.25   -10.29   -13.32    0.2621        -15.04          -5.66        -19.31          -7.31               1000              0.0237            0.1102                18.93
E11_synthesis_stacker_market_aware          All         0.03       794       0.25    -9.80   -17.06    0.2582        -13.57          -6.01        -23.56         -10.56               1000              0.0142            0.0870                17.77
E11_synthesis_stacker_market_aware          All         0.04       548       0.25    -6.73   -16.83    0.2609        -10.07          -3.31        -25.02          -8.37               1000              0.0118            0.0834                17.49
E11_synthesis_stacker_market_aware          All         0.05       412       0.25    -4.47   -13.90    0.2888         -7.63          -1.59        -23.20          -5.12               1000              0.0106            0.0823                17.31
E11_synthesis_stacker_market_aware          All         0.06       309       0.25    -4.34   -17.32    0.2880         -6.88          -1.40        -27.62          -5.74               1000              0.0104            0.0807                17.25
E11_synthesis_stacker_market_aware           IS         0.02       317       0.25    -3.61   -12.98    0.3281         -6.49          -0.76        -23.11          -2.80               1000              0.0532            0.2492                24.53
E11_synthesis_stacker_market_aware           IS         0.03       211       0.25    -3.23   -20.75    0.2512         -5.45          -1.05        -34.60          -7.26               1000              0.0268            0.1869                21.31
E11_synthesis_stacker_market_aware           IS         0.04       136       0.25    -1.59   -15.94    0.2647         -3.37           0.19        -34.40           1.90               1000              0.0238            0.1873                20.91
E11_synthesis_stacker_market_aware           IS         0.05       100       0.25    -1.25   -16.60    0.2700         -2.97           0.48        -38.17           6.75               1000              0.0216            0.1991                20.70
E11_synthesis_stacker_market_aware           IS         0.06        61       0.25    -1.55   -33.96    0.2131         -2.81          -0.14        -60.41          -3.27               1000              0.0232            0.2318                21.54
E11_synthesis_stacker_market_aware          OOS         0.02       782       0.25    -6.68   -13.51    0.2353        -10.33          -2.83        -20.93          -5.80               1000              0.0118            0.0539                16.65
E11_synthesis_stacker_market_aware          OOS         0.03       583       0.25    -6.58   -15.69    0.2607        -10.16          -3.34        -24.00          -8.07               1000              0.0097            0.0509                16.49
E11_synthesis_stacker_market_aware          OOS         0.04       412       0.25    -5.14   -17.13    0.2597         -7.85          -2.44        -26.20          -8.28               1000              0.0079            0.0491                16.36
E11_synthesis_stacker_market_aware          OOS         0.05       312       0.25    -3.22   -13.07    0.2949         -5.74          -0.90        -23.59          -3.73               1000              0.0071            0.0449                16.22
E11_synthesis_stacker_market_aware          OOS         0.06       248       0.25    -2.78   -13.60    0.3065         -5.14          -0.57        -25.54          -2.78               1000              0.0072            0.0436                16.20
E11_synthesis_stacker_market_aware      OOS_DJF         0.02       161       0.25    -0.28    -3.24    0.2236         -1.58           0.97        -18.16          11.56               1000              0.0081            0.0752                17.03
E11_synthesis_stacker_market_aware      OOS_DJF         0.03       120       0.25    -0.55    -7.60    0.2417         -1.92           0.85        -26.91          11.46               1000              0.0070            0.0739                16.94
E11_synthesis_stacker_market_aware      OOS_DJF         0.04        82       0.25    -0.04    -0.74    0.2683         -1.09           1.04        -21.15          21.34               1000              0.0058            0.0732                16.83
E11_synthesis_stacker_market_aware      OOS_DJF         0.05        52       0.25     0.53    14.64    0.3462         -0.43           1.47        -12.20          40.78               1000              0.0044            0.0568                16.33
E11_synthesis_stacker_market_aware      OOS_DJF         0.06        29       0.25     0.45    24.10    0.3448         -0.43           1.27        -23.35          73.34               1000              0.0025            0.0504                16.12
E11_synthesis_stacker_market_aware      OOS_MAM         0.02        94       0.25    -1.49   -28.53    0.1702         -2.84          -0.34        -53.91          -5.99               1000              0.0104            0.0722                17.00
E11_synthesis_stacker_market_aware      OOS_MAM         0.03        48       0.25    -1.70   -59.41    0.1042         -2.62          -0.71        -88.11         -26.21               1000              0.0087            0.0592                16.67
E11_synthesis_stacker_market_aware      OOS_MAM         0.04        26       0.25    -0.97   -80.66    0.0385         -1.58          -0.35       -100.00         -33.40               1000              0.0044            0.0569                16.33
E11_synthesis_stacker_market_aware      OOS_MAM         0.05         5       0.25    -0.39  -100.00    0.0000         -0.53          -0.24       -100.00        -100.00               1000              0.0032            0.0433                16.01
E11_synthesis_stacker_market_aware      OOS_MAM         0.06         3       0.25    -0.28  -100.00    0.0000         -0.41          -0.11       -100.00        -100.00               1000              0.0044            0.0420                16.04
E11_synthesis_stacker_market_aware      OOS_JJA         0.02       208       0.25    -0.13    -0.90    0.2981         -2.13           1.87        -14.43          13.13               1000              0.0151            0.0460                16.60
E11_synthesis_stacker_market_aware      OOS_JJA         0.03       159       0.25    -0.45    -3.52    0.3333         -2.36           1.55        -18.37          12.31               1000              0.0124            0.0441                16.44
E11_synthesis_stacker_market_aware      OOS_JJA         0.04        99       0.25    -1.19   -14.59    0.3030         -2.49           0.06        -30.55           0.72               1000              0.0110            0.0438                16.37
E11_synthesis_stacker_market_aware      OOS_JJA         0.05        83       0.25    -1.17   -16.21    0.3133         -2.60           0.07        -36.45           1.05               1000              0.0105            0.0456                16.38
E11_synthesis_stacker_market_aware      OOS_JJA         0.06        70       0.25    -1.04   -15.72    0.3429         -2.39           0.26        -36.50           3.82               1000              0.0109            0.0471                16.43
E11_synthesis_stacker_market_aware      OOS_SON         0.02       319       0.25    -4.79   -22.73    0.2194         -7.20          -2.57        -33.99         -12.28               1000              0.0119            0.0429                16.39
E11_synthesis_stacker_market_aware      OOS_SON         0.03       256       0.25    -3.87   -20.39    0.2539         -6.11          -1.72        -31.76          -9.06               1000              0.0094            0.0428                16.28
E11_synthesis_stacker_market_aware      OOS_SON         0.04       205       0.25    -2.94   -18.98    0.2634         -5.05          -1.10        -32.16          -7.29               1000              0.0076            0.0410                16.16
E11_synthesis_stacker_market_aware      OOS_SON         0.05       172       0.25    -2.20   -16.44    0.2791         -3.99          -0.41        -30.10          -3.19               1000              0.0065            0.0409                16.11
E11_synthesis_stacker_market_aware      OOS_SON         0.06       146       0.25    -1.92   -16.40    0.2877         -3.39          -0.25        -29.69          -2.23               1000              0.0065            0.0406                16.10
E11_synthesis_stacker_market_aware OOS_volatile         0.02       100       0.25    -0.69   -17.51    0.1400         -1.68           0.43        -45.05          10.15               1000              0.0069            0.1110                17.88
E11_synthesis_stacker_market_aware OOS_volatile         0.03        58       0.25    -0.60   -27.08    0.1207         -1.39           0.30        -66.97          12.57               1000              0.0037            0.1094                17.73
E11_synthesis_stacker_market_aware OOS_volatile         0.04        26       0.25    -0.22   -49.15    0.0385         -0.60           0.10       -100.00          16.37               1000              0.0006            0.1390                18.14
E11_synthesis_stacker_market_aware OOS_volatile         0.05         8       0.25    -0.10   -29.93    0.1250         -0.47           0.21       -100.00          42.60               1000              0.0007            0.1222                17.47
E11_synthesis_stacker_market_aware OOS_volatile         0.06         3       0.25    -0.01  -100.00    0.0000         -0.01          -0.01       -100.00        -100.00               1000              0.0009            0.1515                18.07

## EV-aware dynamic edge gating (E13 neural synthesis challenger)

                   model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E13_neural_synthesis_mlp          All         0.02      1315       0.25   -14.75    -8.00    0.5551        -20.90          -8.63        -11.26          -4.64               1000              0.0371            0.1441                20.52
E13_neural_synthesis_mlp          All         0.03       957       0.25   -11.61    -8.90    0.5340        -16.86          -6.32        -13.06          -4.86               1000              0.0229            0.1101                18.93
E13_neural_synthesis_mlp          All         0.04       664       0.25    -6.18    -7.22    0.5151        -10.96          -2.00        -12.73          -2.36               1000              0.0167            0.0987                18.26
E13_neural_synthesis_mlp          All         0.05       461       0.25    -5.40    -9.43    0.4837         -9.08          -1.51        -16.03          -2.81               1000              0.0124            0.0813                17.66
E13_neural_synthesis_mlp          All         0.06       334       0.25    -4.11   -10.13    0.4701         -7.14          -0.86        -17.72          -2.16               1000              0.0099            0.0679                17.08
E13_neural_synthesis_mlp           IS         0.02       479       0.25    -4.67    -7.02    0.5553         -8.53          -0.97        -12.96          -1.46               1000              0.0735            0.3069                26.98
E13_neural_synthesis_mlp           IS         0.03       271       0.25    -2.68    -7.27    0.5424         -5.72           0.24        -15.38           0.68               1000              0.0465            0.2646                24.77
E13_neural_synthesis_mlp           IS         0.04       157       0.25    -0.53    -2.75    0.5159         -3.08           1.92        -15.61           9.64               1000              0.0338            0.2622                23.94
E13_neural_synthesis_mlp           IS         0.05        93       0.25    -0.10    -0.95    0.5054         -1.97           1.70        -17.79          16.04               1000              0.0261            0.2189                22.77
E13_neural_synthesis_mlp           IS         0.06        53       0.25    -0.24    -4.10    0.4528         -1.51           1.02        -25.84          17.55               1000              0.0196            0.1852                21.32
E13_neural_synthesis_mlp          OOS         0.02       836       0.25   -10.08    -8.55    0.5550        -14.90          -5.33        -12.77          -4.50               1000              0.0163            0.0508                16.81
E13_neural_synthesis_mlp          OOS         0.03       686       0.25    -8.93    -9.55    0.5306        -13.20          -4.57        -14.24          -4.88               1000              0.0136            0.0490                16.62
E13_neural_synthesis_mlp          OOS         0.04       507       0.25    -5.65    -8.52    0.5148         -9.64          -1.61        -14.46          -2.44               1000              0.0114            0.0480                16.50
E13_neural_synthesis_mlp          OOS         0.05       368       0.25    -5.30   -11.46    0.4783         -8.44          -1.86        -18.31          -3.91               1000              0.0089            0.0465                16.36
E13_neural_synthesis_mlp          OOS         0.06       281       0.25    -3.87   -11.14    0.4733         -6.66          -0.91        -19.72          -2.52               1000              0.0081            0.0458                16.28
E13_neural_synthesis_mlp      OOS_DJF         0.02       159       0.25    -2.72   -12.49    0.5157         -5.20          -0.52        -25.00          -2.43               1000              0.0202            0.0629                17.28
E13_neural_synthesis_mlp      OOS_DJF         0.03       125       0.25    -2.48   -14.88    0.4880         -4.61          -0.36        -29.18          -2.29               1000              0.0149            0.0629                17.07
E13_neural_synthesis_mlp      OOS_DJF         0.04        80       0.25    -1.20   -11.95    0.4750         -2.95           0.78        -29.66           7.38               1000              0.0122            0.0642                16.94
E13_neural_synthesis_mlp      OOS_DJF         0.05        52       0.25    -1.23   -21.77    0.3654         -2.59           0.26        -47.82           4.43               1000              0.0075            0.0601                16.71
E13_neural_synthesis_mlp      OOS_DJF         0.06        38       0.25    -0.51   -13.59    0.3684         -1.65           0.69        -47.02          19.20               1000              0.0065            0.0639                16.57
E13_neural_synthesis_mlp      OOS_MAM         0.02       192       0.25    -1.05    -4.19    0.5365         -3.35           1.18        -13.54           5.08               1000              0.0174            0.0561                17.09
E13_neural_synthesis_mlp      OOS_MAM         0.03       144       0.25    -0.17    -0.99    0.5069         -2.16           1.92        -13.02          11.72               1000              0.0127            0.0514                16.63
E13_neural_synthesis_mlp      OOS_MAM         0.04       100       0.25     0.48     4.40    0.4900         -1.39           2.26        -12.43          20.76               1000              0.0100            0.0499                16.49
E13_neural_synthesis_mlp      OOS_MAM         0.05        59       0.25    -0.65   -11.73    0.3559         -2.12           0.73        -37.43          13.03               1000              0.0096            0.0501                16.51
E13_neural_synthesis_mlp      OOS_MAM         0.06        39       0.25    -0.24    -6.36    0.3846         -1.36           0.86        -35.64          24.30               1000              0.0084            0.0513                16.40
E13_neural_synthesis_mlp      OOS_JJA         0.02       234       0.25    -2.88    -9.14    0.5256         -5.58          -0.28        -17.85          -0.90               1000              0.0189            0.0463                16.77
E13_neural_synthesis_mlp      OOS_JJA         0.03       199       0.25    -2.58    -9.97    0.5025         -4.89          -0.25        -18.75          -0.96               1000              0.0170            0.0457                16.68
E13_neural_synthesis_mlp      OOS_JJA         0.04       148       0.25    -1.97   -10.81    0.4730         -4.13           0.51        -21.95           2.88               1000              0.0151            0.0452                16.58
E13_neural_synthesis_mlp      OOS_JJA         0.05       111       0.25    -1.27    -9.35    0.4775         -3.08           0.59        -22.88           4.65               1000              0.0105            0.0445                16.36
E13_neural_synthesis_mlp      OOS_JJA         0.06        83       0.25    -0.89    -9.11    0.4578         -2.47           0.74        -24.31           8.06               1000              0.0096            0.0423                16.28
E13_neural_synthesis_mlp      OOS_SON         0.02       251       0.25    -3.44    -8.66    0.6215         -5.78          -0.84        -14.88          -2.04               1000              0.0107            0.0432                16.34
E13_neural_synthesis_mlp      OOS_SON         0.03       218       0.25    -3.71   -10.93    0.5963         -5.74          -1.50        -17.66          -4.33               1000              0.0101            0.0425                16.30
E13_neural_synthesis_mlp      OOS_SON         0.04       179       0.25    -2.96   -10.91    0.5810         -5.05          -0.97        -19.53          -3.42               1000              0.0088            0.0420                16.24
E13_neural_synthesis_mlp      OOS_SON         0.05       146       0.25    -2.15   -10.02    0.5685         -3.92          -0.33        -18.82          -1.56               1000              0.0078            0.0417                16.18
E13_neural_synthesis_mlp      OOS_SON         0.06       121       0.25    -2.24   -12.74    0.5455         -3.93          -0.53        -23.30          -2.89               1000              0.0075            0.0408                16.15
E13_neural_synthesis_mlp OOS_volatile         0.02       208       0.25    -1.61    -5.97    0.5240         -4.16           0.90        -15.68           3.46               1000              0.0192            0.0645                17.41
E13_neural_synthesis_mlp OOS_volatile         0.03       151       0.25    -1.18    -6.48    0.4834         -3.50           1.30        -19.78           6.93               1000              0.0150            0.0619                17.06
E13_neural_synthesis_mlp OOS_volatile         0.04        99       0.25    -0.31    -3.05    0.4343         -2.50           1.88        -24.55          18.30               1000              0.0116            0.0601                16.86
E13_neural_synthesis_mlp OOS_volatile         0.05        57       0.25    -0.75   -15.24    0.3158         -2.40           0.82        -50.64          15.91               1000              0.0087            0.0599                16.82
E13_neural_synthesis_mlp OOS_volatile         0.06        38       0.25    -0.02    -0.79    0.3421         -1.33           1.30        -42.51          43.59               1000              0.0066            0.0638                16.57

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
        "pnl_ci95_low": -5.14,
        "pnl_ci95_high": -0.57
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
