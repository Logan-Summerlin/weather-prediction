# E0-E22 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

                                    model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
             E17_contract_brier_synthesis             0.114090           0.141775                     0.127061         0.106640       0.139298                      -79.90                       -6.08
               E21_platt_recalibrated_e17             0.114406           0.141775                     0.127061         0.109023       0.139298                      -76.84                      -12.61
                 E13_neural_synthesis_mlp             0.116196           0.141775                     0.127061         0.103582       0.139298                      -85.10                       -1.01
                   E22_expanded_platt_e13             0.116332           0.141775                     0.127061         0.104314       0.139298                      -90.82                       -4.74
               E19_platt_beta_calibration             0.116378           0.141775                     0.127061         0.103835       0.139298                      -83.32                        3.63
       E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57
             E18_regime_adaptive_ensemble             0.123860           0.141775                     0.127061         0.113080       0.139298                     -131.85                      -20.69
      E3_weighted_ensemble_E4_uncertainty             0.133306           0.141775                     0.127061         0.130046       0.139298                     -124.66                      -21.17
                       E1_global_isotonic             0.133372           0.141775                     0.127061         0.130500       0.139298                     -123.32                      -18.24
             E4_uncertainty_decomposition             0.133388           0.141775                     0.127061         0.130096       0.139298                     -121.76                      -20.19
                                  E5_mdn2             0.133444           0.141775                     0.127061         0.129932       0.139298                     -123.22                      -19.49
                     E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
               E10_wga_mdn_regime_mixture             0.133584           0.141775                     0.127061         0.130573       0.139298                     -127.21                      -19.09
                  E7_regularization_sweep             0.133623           0.141775                     0.127061         0.130537       0.139298                     -126.19                      -21.41
    E12_capacity_sweep_residual_synthesis             0.133770           0.141775                     0.127061         0.130561       0.139298                     -122.27                      -18.55
       E16_conditional_calibration_shrunk             0.134133           0.141775                     0.127061         0.131791       0.139298                     -125.14                      -18.84
                  E2_seasonal_calibration             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
          E9_conditional_calibration_grid             0.134247           0.141775                     0.127061         0.131587       0.139298                     -127.25                      -17.05
                              E6_quantile             0.134413           0.141775                     0.127061         0.131367       0.139298                     -121.45                      -17.78
E15_conditional_calibration_spread_regime             0.134469           0.141775                     0.127061         0.132387       0.139298                     -124.05                      -18.44
                 E8_feature_pruning_sweep             0.136353           0.141775                     0.127061         0.134766       0.139298                     -139.56                      -24.47
            E14_distributional_neural_nll             0.194393           0.141775                     0.127061         0.191978       0.139298                     -334.78                      -52.97
        E20_crps_distributional_synthesis             0.204795           0.141775                     0.127061         0.205264       0.139298                     -356.66                      -58.85

## Top 2

                       model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
E17_contract_brier_synthesis             0.114090           0.141775                     0.127061         0.106640       0.139298                      -79.90                       -6.08
  E21_platt_recalibrated_e17             0.114406           0.141775                     0.127061         0.109023       0.139298                      -76.84                      -12.61

## EV-aware dynamic edge gating (best-Brier model)

                       model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E17_contract_brier_synthesis          All         0.02      1022       0.25   -12.18    -9.40    0.4941        -18.87          -5.17        -14.38          -4.00               1000              0.0382            0.1275                20.47
E17_contract_brier_synthesis          All         0.03       738       0.25   -11.92   -12.78    0.4743        -17.41          -6.54        -18.46          -7.03               1000              0.0282            0.1072                19.18
E17_contract_brier_synthesis          All         0.04       480       0.25    -8.13   -14.15    0.4417        -12.90          -3.70        -22.43          -6.46               1000              0.0229            0.0849                18.18
E17_contract_brier_synthesis          All         0.05       329       0.25    -4.26   -11.16    0.4438         -8.17          -0.01        -21.50          -0.02               1000              0.0211            0.0761                17.87
E17_contract_brier_synthesis          All         0.06       232       0.25    -3.31   -13.04    0.4095         -6.71           0.15        -25.91           0.59               1000              0.0162            0.0629                17.33
E17_contract_brier_synthesis           IS         0.02       348       0.25     0.14     0.32    0.5517         -3.92           4.47         -8.89           9.86               1000              0.0820            0.2810                27.69
E17_contract_brier_synthesis           IS         0.03       217       0.25    -1.54    -5.59    0.5161         -4.67           1.65        -16.88           6.25               1000              0.0631            0.2540                25.46
E17_contract_brier_synthesis           IS         0.04       114       0.25    -0.21    -1.59    0.4912         -2.48           1.86        -18.53          14.31               1000              0.0576            0.2172                23.85
E17_contract_brier_synthesis           IS         0.05        76       0.25     0.49     5.43    0.5395         -1.42           2.47        -15.22          28.03               1000              0.0573            0.1876                23.06
E17_contract_brier_synthesis           IS         0.06        42       0.25     0.48    10.31    0.5238         -0.99           1.90        -20.08          43.63               1000              0.0436            0.1557                21.95
E17_contract_brier_synthesis          OOS         0.02       674       0.25   -12.32   -14.48    0.4644        -17.71          -6.77        -20.79          -7.95               1000              0.0156            0.0482                16.74
E17_contract_brier_synthesis          OOS         0.03       521       0.25   -10.38   -15.79    0.4568        -15.64          -5.77        -23.36          -9.04               1000              0.0136            0.0461                16.57
E17_contract_brier_synthesis          OOS         0.04       366       0.25    -7.91   -17.91    0.4262        -11.89          -3.70        -26.83          -8.43               1000              0.0121            0.0436                16.42
E17_contract_brier_synthesis          OOS         0.05       253       0.25    -4.75   -16.30    0.4150         -8.07          -1.07        -27.74          -3.65               1000              0.0103            0.0426                16.31
E17_contract_brier_synthesis          OOS         0.06       190       0.25    -3.79   -18.26    0.3842         -6.41          -0.99        -32.04          -4.67               1000              0.0101            0.0424                16.30
E17_contract_brier_synthesis      OOS_DJF         0.02       144       0.25    -2.41   -13.75    0.4514         -4.79          -0.17        -28.24          -0.88               1000              0.0172            0.0605                17.23
E17_contract_brier_synthesis      OOS_DJF         0.03       108       0.25    -1.63   -11.90    0.4815         -3.64           0.24        -26.96           1.80               1000              0.0141            0.0580                16.95
E17_contract_brier_synthesis      OOS_DJF         0.04        69       0.25    -2.12   -25.95    0.3768         -3.78          -0.41        -48.64          -5.13               1000              0.0132            0.0552                16.70
E17_contract_brier_synthesis      OOS_DJF         0.05        42       0.25    -1.82   -41.57    0.2619         -3.13          -0.56        -75.30         -12.05               1000              0.0105            0.0599                16.67
E17_contract_brier_synthesis      OOS_DJF         0.06        30       0.25    -1.09   -36.88    0.2667         -1.97          -0.05        -78.80          -1.63               1000              0.0108            0.0662                16.81
E17_contract_brier_synthesis      OOS_MAM         0.02       121       0.25    -2.09   -13.03    0.4959         -4.42           0.28        -27.41           1.83               1000              0.0164            0.0530                16.94
E17_contract_brier_synthesis      OOS_MAM         0.03        78       0.25    -2.18   -21.16    0.4487         -4.24          -0.15        -40.21          -1.50               1000              0.0124            0.0471                16.50
E17_contract_brier_synthesis      OOS_MAM         0.04        39       0.25    -0.61   -12.79    0.4615         -2.05           0.81        -44.53          15.89               1000              0.0083            0.0475                16.32
E17_contract_brier_synthesis      OOS_MAM         0.05        22       0.25    -0.20    -8.10    0.4545         -1.20           0.95        -51.89          35.09               1000              0.0075            0.0433                16.21
E17_contract_brier_synthesis      OOS_MAM         0.06        11       0.25    -0.31   -24.82    0.3636         -1.01           0.49       -100.00          34.11               1000              0.0075            0.0416                16.17
E17_contract_brier_synthesis      OOS_JJA         0.02       180       0.25    -2.01    -8.03    0.5500         -5.13           0.73        -20.98           2.95               1000              0.0185            0.0432                16.70
E17_contract_brier_synthesis      OOS_JJA         0.03       150       0.25    -2.16   -10.52    0.5267         -4.84           0.65        -23.46           3.17               1000              0.0176            0.0439                16.67
E17_contract_brier_synthesis      OOS_JJA         0.04       102       0.25    -0.92    -7.05    0.5098         -3.06           1.38        -24.09          10.18               1000              0.0159            0.0429                16.58
E17_contract_brier_synthesis      OOS_JJA         0.05        71       0.25     0.07     0.82    0.5493         -1.77           2.04        -20.48          22.03               1000              0.0129            0.0403                16.39
E17_contract_brier_synthesis      OOS_JJA         0.06        52       0.25     0.02     0.37    0.5385         -1.54           1.66        -25.49          24.37               1000              0.0136            0.0413                16.44
E17_contract_brier_synthesis      OOS_SON         0.02       229       0.25    -5.81   -21.93    0.3886         -8.68          -2.73        -32.39         -10.15               1000              0.0118            0.0419                16.37
E17_contract_brier_synthesis      OOS_SON         0.03       185       0.25    -4.40   -20.82    0.3892         -6.99          -1.76        -32.84          -8.26               1000              0.0107            0.0406                16.29
E17_contract_brier_synthesis      OOS_SON         0.04       156       0.25    -4.27   -23.42    0.3846         -6.58          -1.75        -36.45          -9.37               1000              0.0100            0.0381                16.21
E17_contract_brier_synthesis      OOS_SON         0.05       118       0.25    -2.80   -21.13    0.3814         -4.95          -0.62        -37.03          -4.61               1000              0.0091            0.0378                16.16
E17_contract_brier_synthesis      OOS_SON         0.06        97       0.25    -2.42   -23.99    0.3402         -4.32          -0.34        -43.06          -3.21               1000              0.0084            0.0357                16.09
E17_contract_brier_synthesis OOS_volatile         0.02       143       0.25    -2.38   -14.80    0.4126         -4.67           0.03        -28.87           0.16               1000              0.0181            0.0676                17.53
E17_contract_brier_synthesis OOS_volatile         0.03        99       0.25    -2.15   -18.42    0.4141         -4.16          -0.07        -35.35          -0.71               1000              0.0153            0.0623                17.11
E17_contract_brier_synthesis OOS_volatile         0.04        55       0.25    -1.71   -27.93    0.3455         -3.16          -0.31        -51.17          -4.73               1000              0.0141            0.0636                16.91
E17_contract_brier_synthesis OOS_volatile         0.05        34       0.25    -1.31   -38.45    0.2647         -2.30          -0.14        -75.55          -3.45               1000              0.0121            0.0668                16.88
E17_contract_brier_synthesis OOS_volatile         0.06        24       0.25    -1.27   -57.73    0.1667         -1.97          -0.40       -100.00         -18.19               1000              0.0122            0.0738                17.02

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
  "top_model": "E17_contract_brier_synthesis",
  "promotion_ready": false,
  "checks": {
    "oos_brier_beats_presettlement": {
      "pass": true,
      "oos_model_brier": 0.1066402510782998,
      "presettlement_brier": 0.12706111379754997
    },
    "oos_gated_pnl_positive_with_positive_ci": {
      "pass": false,
      "best_oos_gated": {
        "quality_cut": 0.06,
        "trades": 190,
        "net_pnl": -3.79,
        "roi_pct": -18.26,
        "pnl_ci95_low": -6.41,
        "pnl_ci95_high": -0.99
      }
    },
    "calibration_ece_gate": {
      "pass": true,
      "ece": 0.012940019299335196,
      "threshold": 0.03
    },
    "tail_reliability_gate": {
      "pass": true,
      "max_abs_bin_gap": 0.1808606663737995,
      "threshold": 0.2
    }
  }
}
