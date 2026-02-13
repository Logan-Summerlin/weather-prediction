# E0-E14 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

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
        E14_distributional_neural_nll             0.194393           0.141775                     0.127061         0.191978       0.139298                     -334.78                      -52.97

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

## EV-aware dynamic edge gating (E14 distributional neural challenger)

                        model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E14_distributional_neural_nll          All         0.02      3054       0.25   -48.59   -14.30    0.4103        -55.20         -42.08        -16.25         -12.37               1000              0.0569            0.1938                23.65
E14_distributional_neural_nll          All         0.03      2645       0.25   -40.26   -14.39    0.3894        -46.23         -33.89        -16.47         -12.08               1000              0.0465            0.1726                22.46
E14_distributional_neural_nll          All         0.04      2240       0.25   -33.73   -14.67    0.3768        -39.32         -27.80        -17.09         -12.18               1000              0.0373            0.1510                21.24
E14_distributional_neural_nll          All         0.05      1917       0.25   -26.40   -13.61    0.3761        -32.08         -20.87        -16.48         -10.86               1000              0.0308            0.1301                20.19
E14_distributional_neural_nll          All         0.06      1633       0.25   -23.01   -14.06    0.3705        -28.49         -17.76        -17.37         -10.97               1000              0.0249            0.1150                19.41
E14_distributional_neural_nll           IS         0.02      1388       0.25   -24.69   -15.36    0.4215        -29.81         -19.86        -18.45         -12.35               1000              0.0949            0.3592                30.87
E14_distributional_neural_nll           IS         0.03      1104       0.25   -19.38   -15.65    0.4067        -23.69         -14.48        -18.95         -11.76               1000              0.0772            0.3365                29.28
E14_distributional_neural_nll           IS         0.04       830       0.25   -14.57   -15.96    0.3976        -18.50         -10.41        -20.19         -11.34               1000              0.0632            0.3155                27.79
E14_distributional_neural_nll           IS         0.05       640       0.25    -9.90   -14.27    0.4000        -13.86          -5.99        -19.58          -8.40               1000              0.0531            0.2829                26.15
E14_distributional_neural_nll           IS         0.06       492       0.25    -7.28   -13.96    0.3923        -10.89          -3.83        -20.74          -7.44               1000              0.0432            0.2569                24.82
E14_distributional_neural_nll          OOS         0.02      1666       0.25   -23.90   -13.34    0.4010        -27.35         -19.87        -15.30         -11.18               1000              0.0252            0.0560                17.63
E14_distributional_neural_nll          OOS         0.03      1541       0.25   -20.88   -13.39    0.3770        -24.90         -16.88        -15.98         -10.93               1000              0.0246            0.0551                17.57
E14_distributional_neural_nll          OOS         0.04      1410       0.25   -19.16   -13.82    0.3645        -23.31         -15.14        -16.74         -10.87               1000              0.0221            0.0543                17.39
E14_distributional_neural_nll          OOS         0.05      1277       0.25   -16.50   -13.24    0.3641        -20.41         -12.34        -16.44         -10.02               1000              0.0197            0.0536                17.20
E14_distributional_neural_nll          OOS         0.06      1141       0.25   -15.73   -14.10    0.3611        -19.90         -11.79        -17.98         -10.51               1000              0.0170            0.0539                17.07
E14_distributional_neural_nll      OOS_DJF         0.02       397       0.25    -4.75   -11.20    0.4081         -6.41          -2.72        -14.86          -6.74               1000              0.0256            0.0690                18.67
E14_distributional_neural_nll      OOS_DJF         0.03       357       0.25    -4.07   -11.32    0.3838         -5.82          -2.14        -15.96          -6.29               1000              0.0252            0.0685                18.62
E14_distributional_neural_nll      OOS_DJF         0.04       327       0.25    -3.81   -12.28    0.3578         -5.86          -2.00        -18.50          -6.48               1000              0.0234            0.0668                18.41
E14_distributional_neural_nll      OOS_DJF         0.05       299       0.25    -3.40   -12.04    0.3579         -4.99          -1.38        -17.38          -4.91               1000              0.0213            0.0635                17.98
E14_distributional_neural_nll      OOS_DJF         0.06       274       0.25    -2.81   -11.19    0.3504         -4.55          -0.92        -17.39          -4.05               1000              0.0194            0.0641                17.92
E14_distributional_neural_nll      OOS_MAM         0.02       427       0.25    -5.13   -10.41    0.4450         -7.50          -2.31        -15.20          -4.68               1000              0.0263            0.0610                17.83
E14_distributional_neural_nll      OOS_MAM         0.03       396       0.25    -4.20    -9.65    0.4268         -6.64          -1.72        -15.14          -4.03               1000              0.0250            0.0599                17.73
E14_distributional_neural_nll      OOS_MAM         0.04       363       0.25    -4.03   -10.49    0.4077         -6.72          -1.33        -17.46          -3.49               1000              0.0234            0.0587                17.51
E14_distributional_neural_nll      OOS_MAM         0.05       321       0.25    -2.63    -7.79    0.4174         -5.21           0.06        -15.19           0.17               1000              0.0195            0.0592                17.33
E14_distributional_neural_nll      OOS_MAM         0.06       278       0.25    -2.56    -8.59    0.4209         -5.20           0.14        -17.74           0.52               1000              0.0164            0.0596                17.12
E14_distributional_neural_nll      OOS_JJA         0.02       427       0.25    -6.88   -14.75    0.4005         -8.97          -4.60        -18.99          -9.90               1000              0.0281            0.0482                17.23
E14_distributional_neural_nll      OOS_JJA         0.03       401       0.25    -5.86   -14.30    0.3766         -7.78          -3.62        -18.78          -8.84               1000              0.0271            0.0479                17.18
E14_distributional_neural_nll      OOS_JJA         0.04       368       0.25    -5.32   -13.78    0.3886         -7.36          -3.14        -18.88          -8.22               1000              0.0229            0.0475                16.98
E14_distributional_neural_nll      OOS_JJA         0.05       336       0.25    -4.70   -13.27    0.3929         -6.77          -2.33        -19.24          -6.52               1000              0.0211            0.0469                16.89
E14_distributional_neural_nll      OOS_JJA         0.06       298       0.25    -4.28   -13.29    0.4027         -6.54          -1.92        -20.25          -6.11               1000              0.0179            0.0464                16.73
E14_distributional_neural_nll      OOS_SON         0.02       415       0.25    -7.14   -17.48    0.3494         -7.94          -6.22        -19.64         -15.30               1000              0.0207            0.0464                16.86
E14_distributional_neural_nll      OOS_SON         0.03       387       0.25    -6.76   -18.99    0.3204         -7.77          -5.61        -21.93         -15.85               1000              0.0210            0.0454                16.85
E14_distributional_neural_nll      OOS_SON         0.04       352       0.25    -6.00   -19.59    0.3011         -7.25          -4.70        -23.86         -15.47               1000              0.0186            0.0452                16.74
E14_distributional_neural_nll      OOS_SON         0.05       321       0.25    -5.77   -21.23    0.2866         -7.11          -4.21        -26.07         -15.75               1000              0.0170            0.0457                16.68
E14_distributional_neural_nll      OOS_SON         0.06       291       0.25    -6.08   -24.87    0.2715         -7.41          -4.69        -30.09         -19.28               1000              0.0145            0.0465                16.58
E14_distributional_neural_nll OOS_volatile         0.02       244       0.25    -3.80   -14.17    0.4057         -5.39          -1.97        -19.93          -7.47               1000              0.0222            0.0568                17.46
E14_distributional_neural_nll OOS_volatile         0.03       222       0.25    -2.67   -11.65    0.3919         -4.25          -0.83        -18.26          -3.80               1000              0.0210            0.0542                17.29
E14_distributional_neural_nll OOS_volatile         0.04       194       0.25    -2.33   -12.05    0.3763         -3.99          -0.72        -20.87          -3.69               1000              0.0177            0.0534                17.00
E14_distributional_neural_nll OOS_volatile         0.05       165       0.25    -1.70   -10.24    0.3879         -3.40           0.14        -20.17           0.79               1000              0.0145            0.0543                16.90
E14_distributional_neural_nll OOS_volatile         0.06       139       0.25    -1.57   -11.14    0.3885         -3.16           0.08        -22.86           0.59               1000              0.0115            0.0553                16.72

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
