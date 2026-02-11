# E0/E1/E2 Benchmarks vs NWS + Kalshi Pre-Settlement

E0 in this run is explicitly the canonical benchmark model from data/best_model_predictions_*.

                  model  overall_model_brier  overall_nws_brier  overall_presettlement_brier  oos_model_brier  oos_nws_brier  best_model_all_trading_pnl  best_model_oos_trading_pnl
     E1_global_isotonic             0.133412           0.141775                     0.127061         0.130606       0.139298                     -121.42                      -16.80
   E0_baseline_ensemble             0.133517           0.141775                     0.127061         0.130190       0.139298                     -121.53                      -19.52
E2_seasonal_calibration             0.134256           0.141775                     0.127061         0.131565       0.139298                     -126.52                      -16.64
