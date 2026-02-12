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

                              model period  quality_cut  n_trades  net_pnl  roi_pct  win_rate  pnl_ci95_low  pnl_ci95_high  roi_ci95_low  roi_ci95_high  bootstrap_samples
E3_weighted_ensemble_E4_uncertainty    All         0.02      2584  -113.58   -12.09    0.3437       -143.64         -79.76        -15.30          -8.58               1000
E3_weighted_ensemble_E4_uncertainty    All         0.03      2460  -107.15   -12.02    0.3427       -138.38         -75.39        -15.56          -8.50               1000
E3_weighted_ensemble_E4_uncertainty    All         0.04      2114   -91.43   -11.96    0.3425       -120.74         -61.44        -15.70          -8.03               1000
E3_weighted_ensemble_E4_uncertainty    All         0.05      1813   -69.84   -10.76    0.3436        -97.19         -42.89        -14.85          -6.62               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.02      1200   -56.99   -12.42    0.3600        -81.21         -33.27        -17.65          -7.35               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.03      1129   -51.15   -11.85    0.3623        -74.27         -29.06        -17.42          -6.81               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.04       915   -41.30   -12.02    0.3552        -62.01         -20.13        -17.92          -5.94               1000
E3_weighted_ensemble_E4_uncertainty     IS         0.05       738   -22.34    -8.06    0.3713        -40.85          -4.81        -14.53          -1.71               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.02      1384   -56.59   -11.77    0.3295        -77.36         -36.00        -15.86          -7.48               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.03      1331   -56.00   -12.18    0.3261        -76.35         -34.19        -16.72          -7.49               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.04      1199   -50.13   -11.90    0.3328        -70.41         -29.04        -16.74          -6.85               1000
E3_weighted_ensemble_E4_uncertainty    OOS         0.05      1075   -47.50   -12.77    0.3247        -67.30         -26.59        -17.97          -7.11               1000
