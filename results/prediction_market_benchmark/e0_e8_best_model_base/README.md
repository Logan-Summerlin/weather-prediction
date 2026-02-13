# E0-E15 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

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

## EV-aware dynamic edge gating (E14 distributional neural challenger)

                        model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E14_distributional_neural_nll          All         0.02      2901       0.25   -45.33   -14.03    0.4119        -51.23         -38.18        -15.87         -11.82               1000              0.0541            0.1538                22.51
E14_distributional_neural_nll          All         0.03      2537       0.25   -38.44   -14.34    0.3894        -44.15         -32.26        -16.49         -12.00               1000              0.0450            0.1394                21.57
E14_distributional_neural_nll          All         0.04      2166       0.25   -32.38   -14.61    0.3758        -37.94         -26.88        -17.10         -12.15               1000              0.0361            0.1236                20.51
E14_distributional_neural_nll          All         0.05      1870       0.25   -25.62   -13.60    0.3743        -31.18         -19.58        -16.55         -10.36               1000              0.0302            0.1093                19.67
E14_distributional_neural_nll          All         0.06      1600       0.25   -22.44   -14.06    0.3688        -27.52         -17.50        -17.18         -11.02               1000              0.0243            0.0977                19.00
E14_distributional_neural_nll           IS         0.02      1235       0.25   -21.43   -14.88    0.4267        -25.99         -16.58        -17.99         -11.65               1000              0.0931            0.2856                29.09
E14_distributional_neural_nll           IS         0.03       996       0.25   -17.56   -15.65    0.4086        -21.83         -13.32        -19.40         -11.86               1000              0.0766            0.2697                27.75
E14_distributional_neural_nll           IS         0.04       756       0.25   -13.22   -15.94    0.3968        -16.75          -9.26        -20.32         -11.21               1000              0.0622            0.2530                26.34
E14_distributional_neural_nll           IS         0.05       593       0.25    -9.12   -14.31    0.3963        -12.70          -5.45        -20.13          -8.64               1000              0.0527            0.2293                25.00
E14_distributional_neural_nll           IS         0.06       459       0.25    -6.72   -13.96    0.3878        -10.03          -3.25        -20.71          -6.92               1000              0.0425            0.2065                23.80
E14_distributional_neural_nll          OOS         0.02      1666       0.25   -23.90   -13.34    0.4010        -27.85         -19.83        -15.35         -11.16               1000              0.0252            0.0560                17.63
E14_distributional_neural_nll          OOS         0.03      1541       0.25   -20.88   -13.39    0.3770        -24.81         -16.96        -15.76         -10.98               1000              0.0246            0.0551                17.57
E14_distributional_neural_nll          OOS         0.04      1410       0.25   -19.16   -13.82    0.3645        -23.11         -15.10        -16.67         -10.82               1000              0.0221            0.0543                17.39
E14_distributional_neural_nll          OOS         0.05      1277       0.25   -16.50   -13.24    0.3641        -20.54         -12.15        -16.36          -9.90               1000              0.0197            0.0536                17.20
E14_distributional_neural_nll          OOS         0.06      1141       0.25   -15.73   -14.10    0.3611        -20.04         -11.25        -17.82         -10.16               1000              0.0170            0.0539                17.07
E14_distributional_neural_nll      OOS_DJF         0.02       397       0.25    -4.75   -11.20    0.4081         -6.39          -2.70        -14.92          -6.67               1000              0.0256            0.0690                18.67
E14_distributional_neural_nll      OOS_DJF         0.03       357       0.25    -4.07   -11.32    0.3838         -5.87          -2.03        -15.79          -5.80               1000              0.0252            0.0685                18.62
E14_distributional_neural_nll      OOS_DJF         0.04       327       0.25    -3.81   -12.28    0.3578         -5.49          -1.95        -17.68          -6.61               1000              0.0234            0.0668                18.41
E14_distributional_neural_nll      OOS_DJF         0.05       299       0.25    -3.40   -12.04    0.3579         -5.14          -1.52        -17.81          -5.67               1000              0.0213            0.0635                17.98
E14_distributional_neural_nll      OOS_DJF         0.06       274       0.25    -2.81   -11.19    0.3504         -4.41          -0.94        -17.25          -3.85               1000              0.0194            0.0641                17.92
E14_distributional_neural_nll      OOS_MAM         0.02       427       0.25    -5.13   -10.41    0.4450         -7.53          -2.28        -15.19          -4.59               1000              0.0263            0.0610                17.83
E14_distributional_neural_nll      OOS_MAM         0.03       396       0.25    -4.20    -9.65    0.4268         -6.70          -1.38        -15.33          -3.36               1000              0.0250            0.0599                17.73
E14_distributional_neural_nll      OOS_MAM         0.04       363       0.25    -4.03   -10.49    0.4077         -6.43          -1.29        -16.76          -3.39               1000              0.0234            0.0587                17.51
E14_distributional_neural_nll      OOS_MAM         0.05       321       0.25    -2.63    -7.79    0.4174         -4.97           0.19        -14.57           0.56               1000              0.0195            0.0592                17.33
E14_distributional_neural_nll      OOS_MAM         0.06       278       0.25    -2.56    -8.59    0.4209         -4.97          -0.02        -17.01          -0.06               1000              0.0164            0.0596                17.12
E14_distributional_neural_nll      OOS_JJA         0.02       427       0.25    -6.88   -14.75    0.4005         -8.71          -4.63        -18.84         -10.11               1000              0.0281            0.0482                17.23
E14_distributional_neural_nll      OOS_JJA         0.03       401       0.25    -5.86   -14.30    0.3766         -7.92          -3.58        -19.30          -8.72               1000              0.0271            0.0479                17.18
E14_distributional_neural_nll      OOS_JJA         0.04       368       0.25    -5.32   -13.78    0.3886         -7.39          -2.86        -18.77          -7.27               1000              0.0229            0.0475                16.98
E14_distributional_neural_nll      OOS_JJA         0.05       336       0.25    -4.70   -13.27    0.3929         -6.62          -2.35        -18.79          -6.58               1000              0.0211            0.0469                16.89
E14_distributional_neural_nll      OOS_JJA         0.06       298       0.25    -4.28   -13.29    0.4027         -6.26          -2.02        -19.80          -6.13               1000              0.0179            0.0464                16.73
E14_distributional_neural_nll      OOS_SON         0.02       415       0.25    -7.14   -17.48    0.3494         -7.98          -6.14        -19.57         -15.25               1000              0.0207            0.0464                16.86
E14_distributional_neural_nll      OOS_SON         0.03       387       0.25    -6.76   -18.99    0.3204         -7.70          -5.67        -21.85         -15.98               1000              0.0210            0.0454                16.85
E14_distributional_neural_nll      OOS_SON         0.04       352       0.25    -6.00   -19.59    0.3011         -7.20          -4.67        -23.73         -15.56               1000              0.0186            0.0452                16.74
E14_distributional_neural_nll      OOS_SON         0.05       321       0.25    -5.77   -21.23    0.2866         -7.12          -4.38        -25.97         -16.25               1000              0.0170            0.0457                16.68
E14_distributional_neural_nll      OOS_SON         0.06       291       0.25    -6.08   -24.87    0.2715         -7.39          -4.75        -30.44         -19.61               1000              0.0145            0.0465                16.58
E14_distributional_neural_nll OOS_volatile         0.02       244       0.25    -3.80   -14.17    0.4057         -5.37          -1.93        -19.69          -7.30               1000              0.0222            0.0568                17.46
E14_distributional_neural_nll OOS_volatile         0.03       222       0.25    -2.67   -11.65    0.3919         -4.33          -0.92        -18.35          -4.13               1000              0.0210            0.0542                17.29
E14_distributional_neural_nll OOS_volatile         0.04       194       0.25    -2.33   -12.05    0.3763         -3.95          -0.59        -19.97          -3.08               1000              0.0177            0.0534                17.00
E14_distributional_neural_nll OOS_volatile         0.05       165       0.25    -1.70   -10.24    0.3879         -3.27           0.21        -19.86           1.17               1000              0.0145            0.0543                16.90
E14_distributional_neural_nll OOS_volatile         0.06       139       0.25    -1.57   -11.14    0.3885         -3.27           0.23        -22.93           1.68               1000              0.0115            0.0553                16.72

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
