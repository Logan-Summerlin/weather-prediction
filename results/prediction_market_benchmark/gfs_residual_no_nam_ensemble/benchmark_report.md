# GFS Residual Model Benchmark (NAM/Ensemble MOS Removed)

## Setup
- Base forecast: `gfs_mos_tmax_f` only.
- Removed inputs: `nam_mos_tmax_f`, `mos_ensemble_tmax_f`.
- Residual target: `residual_gfs = nyc_tmax - gfs_mos_tmax_f`.
- Features: gfs_mos_tmax_f, lag1, lag2, gfs_resid_lag1, gfs_resid_3d, gfs_resid_7d, gfs_resid_14d, gfs_abs_resid_7d, sin_doy, cos_doy

## Residual model metrics
           crps       nll       mae
val    1.765067  2.586383  2.394076
calib  1.698891  2.545568  2.358909
test   1.573661  2.440946  2.164427
oos    1.636753  2.467317  2.265434

## Benchmark (Brier, lower is better)
      slice               source  brier_score  log_score  n_buckets
    Overall                Model     0.142323   0.450140       6204
    Overall Kalshi_PreSettlement     0.127061   0.388250       6204
    Overall                  NWS     0.141775   0.449920       6204
    Overall       Kalshi_Settled     0.018106   0.071145       6204
Period: OOS                Model     0.137755   0.436526       2158
Period: OOS Kalshi_PreSettlement     0.098839   0.309298       2158
Period: OOS                  NWS     0.139298   0.441113       2158
Period: OOS       Kalshi_Settled     0.002100   0.017147       2158
