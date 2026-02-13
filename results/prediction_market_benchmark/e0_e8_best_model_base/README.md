# E0-E16 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement

                                    model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
                 E13_neural_synthesis_mlp             0.116196           0.141775                     0.127061         0.103582       0.139298                      -85.10                       -1.01
       E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57
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
          E13_neural_synthesis_mlp             0.116196           0.141775                     0.127061         0.103582       0.139298                      -85.10                       -1.01
E11_synthesis_stacker_market_aware             0.116579           0.141775                     0.127061         0.105364       0.139298                      -88.47                       -3.57

## EV-aware dynamic edge gating (best-Brier model)

                   model       period  quality_cut  n_trades  avg_stake  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples  avg_queue_pressure  avg_cancel_proxy  avg_latency_seconds
E13_neural_synthesis_mlp          All         0.02      1041       0.25   -11.54   -11.02    0.3852        -17.49          -5.96        -16.66          -5.63               1000              0.0377            0.1192                19.80
E13_neural_synthesis_mlp          All         0.03       816       0.25    -7.65    -9.88    0.3676        -12.74          -2.86        -16.57          -3.69               1000              0.0285            0.1028                18.97
E13_neural_synthesis_mlp          All         0.04       585       0.25    -5.62   -10.64    0.3470        -10.35          -0.93        -19.75          -1.78               1000              0.0221            0.0945                18.44
E13_neural_synthesis_mlp          All         0.05       435       0.25    -2.36    -6.41    0.3402         -6.49           1.73        -17.21           4.96               1000              0.0189            0.0857                18.05
E13_neural_synthesis_mlp          All         0.06       332       0.25    -3.00   -10.67    0.3253         -6.49           0.35        -23.28           1.30               1000              0.0171            0.0851                17.88
E13_neural_synthesis_mlp           IS         0.02       365       0.25    -2.59    -6.75    0.4219         -5.96           0.93        -15.78           2.46               1000              0.0808            0.2521                25.68
E13_neural_synthesis_mlp           IS         0.03       262       0.25    -1.58    -6.23    0.3893         -4.70           1.39        -18.11           5.60               1000              0.0626            0.2221                24.16
E13_neural_synthesis_mlp           IS         0.04       166       0.25    -1.23    -8.10    0.3614         -3.75           1.58        -25.02          10.25               1000              0.0480            0.2167                23.45
E13_neural_synthesis_mlp           IS         0.05       112       0.25    -0.77    -8.02    0.3393         -3.00           1.57        -31.07          16.65               1000              0.0419            0.2004                22.73
E13_neural_synthesis_mlp           IS         0.06        77       0.25    -2.31   -35.55    0.2338         -3.76          -0.77        -57.17         -12.03               1000              0.0395            0.2154                22.85
E13_neural_synthesis_mlp          OOS         0.02       676       0.25    -8.95   -13.49    0.3654        -13.39          -4.35        -20.12          -6.65               1000              0.0145            0.0474                16.62
E13_neural_synthesis_mlp          OOS         0.03       554       0.25    -6.07   -11.66    0.3574        -10.26          -1.79        -19.26          -3.64               1000              0.0124            0.0463                16.51
E13_neural_synthesis_mlp          OOS         0.04       419       0.25    -4.39   -11.67    0.3413         -8.39          -0.59        -22.11          -1.57               1000              0.0118            0.0460                16.46
E13_neural_synthesis_mlp          OOS         0.05       323       0.25    -1.59    -5.84    0.3406         -5.06           2.02        -18.64           7.68               1000              0.0110            0.0459                16.42
E13_neural_synthesis_mlp          OOS         0.06       255       0.25    -0.69    -3.20    0.3529         -3.91           2.57        -18.08          11.85               1000              0.0104            0.0457                16.38
E13_neural_synthesis_mlp      OOS_DJF         0.02       127       0.25    -1.68   -14.69    0.3307         -3.82           0.46        -33.41           4.12               1000              0.0154            0.0552                16.94
E13_neural_synthesis_mlp      OOS_DJF         0.03       102       0.25    -0.94   -10.87    0.3235         -2.81           0.97        -31.99          11.21               1000              0.0137            0.0543                16.83
E13_neural_synthesis_mlp      OOS_DJF         0.04        71       0.25    -0.32    -6.36    0.2817         -1.86           1.41        -38.81          27.54               1000              0.0139            0.0553                16.79
E13_neural_synthesis_mlp      OOS_DJF         0.05        57       0.25     0.14     3.77    0.2982         -1.20           1.76        -30.30          45.46               1000              0.0122            0.0531                16.69
E13_neural_synthesis_mlp      OOS_DJF         0.06        46       0.25    -0.15    -4.75    0.2826         -1.29           1.09        -43.08          35.25               1000              0.0099            0.0538                16.52
E13_neural_synthesis_mlp      OOS_MAM         0.02       118       0.25    -0.81    -7.32    0.3729         -2.71           1.13        -24.93           9.92               1000              0.0163            0.0504                16.74
E13_neural_synthesis_mlp      OOS_MAM         0.03        85       0.25    -0.62    -8.18    0.3529         -2.10           0.91        -27.54          12.21               1000              0.0128            0.0510                16.60
E13_neural_synthesis_mlp      OOS_MAM         0.04        62       0.25     0.17     3.19    0.3871         -1.09           1.48        -20.21          28.04               1000              0.0123            0.0498                16.55
E13_neural_synthesis_mlp      OOS_MAM         0.05        45       0.25     0.03     0.69    0.3556         -0.98           1.06        -28.45          28.58               1000              0.0106            0.0475                16.43
E13_neural_synthesis_mlp      OOS_MAM         0.06        28       0.25     0.32    14.24    0.3929         -0.67           1.30        -30.28          58.76               1000              0.0118            0.0453                16.44
E13_neural_synthesis_mlp      OOS_JJA         0.02       193       0.25    -2.47   -12.56    0.3834         -5.04           0.08        -24.87           0.42               1000              0.0163            0.0438                16.61
E13_neural_synthesis_mlp      OOS_JJA         0.03       166       0.25    -1.47    -8.85    0.3916         -3.72           1.02        -21.93           6.23               1000              0.0139            0.0420                16.46
E13_neural_synthesis_mlp      OOS_JJA         0.04       129       0.25    -1.57   -12.82    0.3566         -3.67           0.70        -29.02           6.10               1000              0.0139            0.0421                16.47
E13_neural_synthesis_mlp      OOS_JJA         0.05        96       0.25    -0.24    -2.76    0.3750         -2.37           2.08        -27.15          25.74               1000              0.0139            0.0425                16.47
E13_neural_synthesis_mlp      OOS_JJA         0.06        75       0.25     0.10     1.58    0.3867         -1.76           2.09        -26.50          34.67               1000              0.0128            0.0423                16.42
E13_neural_synthesis_mlp      OOS_SON         0.02       238       0.25    -3.99   -16.48    0.3655         -6.71          -1.23        -27.22          -5.15               1000              0.0115            0.0446                16.41
E13_neural_synthesis_mlp      OOS_SON         0.03       201       0.25    -3.05   -15.78    0.3483         -5.55          -0.37        -28.42          -1.95               1000              0.0102            0.0439                16.34
E13_neural_synthesis_mlp      OOS_SON         0.04       157       0.25    -2.68   -17.84    0.3376         -4.94          -0.52        -33.39          -3.65               1000              0.0089            0.0435                16.27
E13_neural_synthesis_mlp      OOS_SON         0.05       125       0.25    -1.52   -13.73    0.3280         -3.58           0.65        -32.65           5.40               1000              0.0083            0.0446                16.26
E13_neural_synthesis_mlp      OOS_SON         0.06       106       0.25    -0.96   -10.08    0.3491         -2.82           1.11        -29.11          11.65               1000              0.0084            0.0447                16.27
E13_neural_synthesis_mlp OOS_volatile         0.02       118       0.25    -0.79    -6.37    0.4237         -2.71           1.14        -21.86           9.14               1000              0.0152            0.0588                17.01
E13_neural_synthesis_mlp OOS_volatile         0.03        78       0.25    -0.22    -2.85    0.4103         -1.63           1.32        -22.07          18.19               1000              0.0127            0.0612                16.96
E13_neural_synthesis_mlp OOS_volatile         0.04        47       0.25    -0.12    -3.01    0.3617         -1.29           1.18        -31.65          28.32               1000              0.0123            0.0653                16.95
E13_neural_synthesis_mlp OOS_volatile         0.05        28       0.25     0.12     5.48    0.3571         -0.71           1.10        -34.78          51.56               1000              0.0131            0.0710                17.17
E13_neural_synthesis_mlp OOS_volatile         0.06        20       0.25     0.14     7.93    0.4000         -0.55           0.86        -33.83          48.18               1000              0.0118            0.0787                17.10

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
  "top_model": "E13_neural_synthesis_mlp",
  "promotion_ready": false,
  "checks": {
    "oos_brier_beats_presettlement": {
      "pass": true,
      "oos_model_brier": 0.10358207606243564,
      "presettlement_brier": 0.12706111379754997
    },
    "oos_gated_pnl_positive_with_positive_ci": {
      "pass": false,
      "best_oos_gated": {
        "quality_cut": 0.06,
        "trades": 255,
        "net_pnl": -0.69,
        "roi_pct": -3.2,
        "pnl_ci95_low": -3.91,
        "pnl_ci95_high": 2.57
      }
    },
    "calibration_ece_gate": {
      "pass": true,
      "ece": 0.017574989695171104,
      "threshold": 0.03
    },
    "tail_reliability_gate": {
      "pass": true,
      "max_abs_bin_gap": 0.11875364797919996,
      "threshold": 0.2
    }
  }
}
