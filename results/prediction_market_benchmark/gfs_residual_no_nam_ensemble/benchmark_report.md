# GFS Residual Model Benchmark (NAM/Ensemble MOS Removed)

## Setup
- Base forecast: `gfs_mos_tmax_f` only.
- Removed inputs: `nam_mos_tmax_f`, `mos_ensemble_tmax_f`.
- Residual target: `residual_gfs = nyc_tmax - gfs_mos_tmax_f`.
- Features: gfs_mos_tmax_f, lag1, lag2, gfs_resid_lag1, gfs_resid_3d, gfs_resid_7d, gfs_resid_14d, gfs_abs_resid_7d, knyc_mos_wind_speed_mph, knyc_mos_wind_dir_deg, knyc_mos_cloud_cover_code, knyc_mos_dewpoint_f, knyc_mos_rel_humidity_pct, other_station_avg_wind_speed_mph, other_station_avg_precip_prob, other_station_avg_snow_indicator, sin_doy, cos_doy

## Residual model metrics
           crps       nll       mae
val    1.773344  2.574300  2.447292
calib  1.709117  2.553986  2.377247
test   1.593179  2.456301  2.190114
oos    1.646564  2.481605  2.257017

## Benchmark (Brier, lower is better)
      slice               source  brier_score  log_score  n_buckets
    Overall                Model     0.143070   0.451671       6204
    Overall Kalshi_PreSettlement     0.127061   0.388250       6204
    Overall                  NWS     0.141775   0.449920       6204
    Overall       Kalshi_Settled     0.018106   0.071145       6204
Period: OOS                Model     0.138151   0.438211       2158
Period: OOS Kalshi_PreSettlement     0.098839   0.309298       2158
Period: OOS                  NWS     0.139298   0.441113       2158
Period: OOS       Kalshi_Settled     0.002100   0.017147       2158
