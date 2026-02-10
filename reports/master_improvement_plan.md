# Master Improvement Plan: Beating the NYC Temperature Prediction Market

**Date:** 2026-02-10
**Synthesized from:** `best_model_improvement_proposal.md`, `model_improvement_proposal.md`, `prediction_market_plan_review_addendum.md`
**Current best model:** C_Correction_NN_tiny — 2.090°F test / 2.093°F OOS MAE
**End goal:** Calibrated probabilistic forecasts that consistently beat the Kalshi KXHIGHNY market

---

## Strategic Assessment

### What the Three Proposals Agree On

All three proposals converge on the same diagnosis and direction. The areas of consensus represent the highest-confidence bets:

1. **MOS residual correction is the right foundation.** Station-only models plateau at ~4.0°F regardless of architecture. MOS integration is a 2°F lever. Build on C_Correction_NN_tiny, don't replace it.

2. **The bottleneck is features, not architecture.** Deeper/wider NNs, LSTMs, GRUs, and temporal convolutions have all been tested and failed to move the needle without MOS. The next gains come from new data and physics-informed feature engineering.

3. **Spring (MAM) is the dominant error source.** At 2.575°F test / 2.863°F OOS, spring is 0.5-0.8°F worse than summer/fall. Spring errors stem from regime transitions, frontal passages, and sea-breeze onset — all detectable with wind and pressure data the model currently lacks.

4. **Probabilistic output is mandatory for trading.** The downstream objective is calibrated bucket probabilities, not point MAE. Training with NLL/CRPS and adding post-hoc calibration directly aligns model optimization with trading objectives.

5. **Wind-conditioned station weighting is the key missing physics.** The model treats 48 stations with fixed weights. On any given day, only the upwind stations carry meaningful signal. Wind direction data collapses effective dimensionality and provides a physics-based attention prior.

6. **Training-inference consistency is critical.** GHCN-Daily has a 1-2 day lag and cannot be used operationally. The pipeline must train on IEM ASOS-derived features so that training and inference see identical data sources.

### Where the Proposals Diverge — and the Resolution

| Question | Proposal 1 (Architecture) | Proposal 2 (Features) | Addendum (Operations) | **This Plan** |
|----------|--------------------------|----------------------|----------------------|---------------|
| First priority | Probabilistic residual head | MOS error memory features | IEM ASOS data pipeline | **Features first, then probabilistic output** (features are faster to implement and validate) |
| Architecture upgrade | Mixture residual + regime multi-head | Attention pooling + MoE | Wind-gated attention + MDN | **Wind-gated attention into MOS-residual path** (already implemented in repo; MoE as Phase 4) |
| NWP synthesis | Not addressed | GFS/GEFS synthesis layer (Tier 4) | GEFSv12 Reforecast + hybrid architecture | **GEFSv12 Reforecast per addendum** (eliminates ERA5, minimizes training-inference mismatch) |
| Calibration approach | Isotonic on PIT/CDF + conformalized quantiles | Post-hoc isotonic by season | Isotonic + seasonal conditional calibration | **Season-conditional isotonic on held-out calibration window** |
| Data volume | Not addressed | 50+ years considered | 26 years IEM-based | **26 years IEM-based** (climate non-stationarity makes 50yr risky without detrending) |

### What Won't Help (Confirmed Dead Ends)

The optimization work has already ruled these out. They should not be revisited:

- Deeper/wider station-only NNs (plateau at ~4.0°F)
- LSTM/GRU for temperature sequences (worse than flat MLPs)
- Temporal 1D convolution (worse than flat MLP)
- Stacking with NN meta-learner (massive overfitting)
- Dropout > 0.1 (optimal at 0.0 for station-only)
- Large batch sizes >256 (minimal impact)
- ERA5 for operational features (multi-day lag, analysis ≠ forecast)

---

## Irreducible Error Floor

Before committing to a target, it's important to bound what's achievable:

| Component | Estimated | Addressable? |
|-----------|-----------|--------------|
| Fundamental weather chaos (24h TMAX) | 1.2-1.5°F | No |
| MOS systematic biases (seasonal, regime) | 0.3-0.5°F | Yes — MOS error memory + regime features |
| Local effects not in MOS (sea breeze, UHI, elevation) | 0.2-0.4°F | Yes — station observations + wind conditioning |
| Data noise / measurement error | 0.1-0.2°F | Partially — ensembling, SWA |

**Theoretical floor: ~1.5-1.8°F.** Realistic target with all improvements: **1.8-1.95°F full-year MAE** with well-calibrated probability distributions.

---

## Phased Execution Plan

### Phase 1: Quick Wins — Feature Additions + Probabilistic Output
**Goal:** Reduce OOS MAE to ~1.95-2.05°F and produce a calibrated distribution
**Dependencies:** None (uses existing data + model infrastructure)

#### 1A. MOS Error Memory Features
*Source: Proposal 2 §2.2 — Low effort, highest expected ROI*

Add features that make MOS autocorrelated biases explicit:

```
mos_error_yesterday  = actual(t-1) - MOS_ensemble(t-1)
mos_error_7d         = rolling_mean(actual - MOS, 7 days, lag-1)
mos_error_14d        = rolling_mean(actual - MOS, 14 days, lag-1)
mos_error_30d        = rolling_mean(actual - MOS, 30 days, lag-1)
mos_abs_error_7d     = rolling_mean(|actual - MOS|, 7 days, lag-1)
gfs_nam_spread       = |GFS_MOS_TMAX - NAM_MOS_TMAX|
gfs_nam_sign         = sign(GFS_MOS_TMAX - NAM_MOS_TMAX)
```

**Critical:** All rolling features use strictly lag-1+ data. Verify with a shuffled-label sanity check.

**Expected impact:** -0.05 to -0.15°F MAE

#### 1B. Semi-Annual Harmonics + Gradient Features
*Source: Proposal 2 §5.2-5.3 — Low effort*

```
sin_day_semi = sin(4π × doy / 365.25)
cos_day_semi = cos(4π × doy / 365.25)
max_gradient = max over station pairs of |T_i - T_j| / distance(i,j)
gradient_bearing = bearing of maximum gradient
```

**Expected impact:** -0.02 to -0.05°F MAE

#### 1C. MOS × Station Interaction Features
*Source: Proposal 2 §5.1*

```
mos_station_gap    = MOS_ensemble - NYC_TMAX(t-1)
mos_sector_gap     = MOS_ensemble - mean_WNW_TMAX(t-1)
station_mos_agree  = sign(delta_T_stations) == sign(mos_station_gap)
```

**Expected impact:** -0.01 to -0.03°F MAE

#### 1D. Convert to Probabilistic Output
*Source: Proposal 1 §A — Highest ROI for trading objective*

Replace the point residual head on C_Correction_NN_tiny with a heteroscedastic Gaussian:

- Two output heads: `mu` and `log_sigma`
- `sigma = softplus(log_sigma) + 0.75` (floor at 0.75°F, cap at 10°F)
- Train with Gaussian NLL, then fine-tune final 15-30 epochs with CRPS
- Existing `src/crps_loss.py` already implements CRPS

**Implementation:** Start from C_Correction_NN_tiny backbone. Only change the output layer and loss function. Keep everything else identical for clean comparison.

#### 1E. Multi-Seed Ensemble
*Source: Proposal 2 §3.3 — Low effort, reliable*

Train 5 copies of the probabilistic correction model with different random seeds. Average the mu predictions; combine sigma via `sqrt(mean(sigma_i^2) + var(mu_i))` (mixture of Gaussians variance).

**Expected impact:** -0.02 to -0.05°F MAE, improved calibration stability

#### Phase 1 Validation
- Compare test and OOS MAE against 2.090/2.093°F baseline
- Evaluate CRPS, PIT histogram, 90% interval coverage
- Seasonal breakdown (especially spring MAE improvement from MOS error features)
- **Gate:** Proceed to Phase 2 only if OOS MAE ≤ 2.05°F or calibration metrics improve meaningfully

---

### Phase 2: ASOS Integration + Wind Physics
**Goal:** Reduce OOS MAE to ~1.85-1.95°F, especially in spring
**Dependencies:** IEM ASOS data download (asos_collection.py already exists)

#### 2A. IEM ASOS Data Pipeline
*Source: Addendum §2.1 — asos_collection.py already implemented*

Download hourly ASOS data for all mapped stations (14 core airports + extended network where available). Compute daily aggregates:

```
Per station:
  wind_dir_prevailing   (vector mean of hourly u,v)
  wind_dir_evening      (18-00Z vector mean)
  wind_speed_mean, wind_speed_max
  wind_dir_persistence  (1 - circular variance)
  dewpoint_mean, dewpoint_afternoon
  slp_00z, slp_12z, slp_24h_tendency
  cloud_fraction        (hours with ceiling < 5000ft)
```

**Training-inference consistency:** Train on IEM ASOS-derived TMAX (not GHCN). Cross-validate against GHCN to quantify systematic differences. If IEM-GHCN bias > 0.5°F, train a calibration offset.

#### 2B. Wind-Conditioned Station Composites
*Source: Addendum §3.1, Proposal 2 §2.1 — The key innovation*

```
For each station i with bearing β_i and distance d_i:
  angular_alignment_i = cos(wind_dir - β_i)
  upwind_weight_i = max(0, angular_alignment_i) / sqrt(d_i)

upwind_temp      = Σ(upwind_weight_i × TMAX_i) / Σ(upwind_weight_i)
crosswind_temp   = weighted avg of perpendicular stations
downwind_temp    = weighted avg of downwind stations
upwind_gradient  = upwind_temp - NYC_TMAX(t-1)
advection_proxy  = wind_speed × upwind_gradient / mean_upwind_distance
```

Compute at two levels:
- Surface wind (IEM ASOS daily mean)
- Evening wind (18-00Z, more predictive of overnight advection)

**Expected impact:** -0.10 to -0.25°F MAE (largest single feature addition)

#### 2C. Frontal Passage Detection Features
*Source: Addendum §3.3*

```
max_station_24h_change  = max(|T_i(t-1) - T_i(t-2)|) across all stations
pressure_tendency       = SLP(00Z, t-1) - SLP(00Z, t-2)
wind_shift              = angular distance between prevailing wind(t-1) and wind(t-2)
dewpoint_drop           = dewpoint(t-1) - dewpoint(t-2)
wn_to_coast_gradient_change = (WNW_mean - coastal_mean)(t-1) - same(t-2)
```

These features directly address the spring error problem: frontal passages are the dominant cause of large spring forecast errors.

#### 2D. Dewpoint, Pressure, Cloud Features
*Source: Proposal 2 §2.1, Addendum §3.6-3.7*

```
dewpoint_nyc             = afternoon dewpoint at Central Park
dewpoint_depression      = T - Td (dry air → greater diurnal range)
sector_dewpoint_gradient = moisture contrast between sectors
cloud_fraction_yesterday = fraction of hours with ceiling < 5000ft
clear_sky_indicator      = binary (was it mostly clear?)
```

#### Phase 2 Validation
- Feature ablation: test each group (wind, pressure, cloud, dewpoint) independently AND combined
- Seasonal breakdown with focus on spring MAE
- Compare wind-conditioned composites vs. raw station features
- Verify all features are computable from data available at 6 AM ET
- **Gate:** Proceed to Phase 3 if spring MAE improves by ≥0.2°F or overall OOS MAE ≤ 1.95°F

---

### Phase 3: Upper-Air + Precipitation + Enhanced Temporal
**Goal:** Reduce OOS MAE to ~1.80-1.90°F
**Dependencies:** IGRA sounding download, GHCN precip parsing

#### 3A. 850mb Temperature from IGRA Soundings
*Source: Proposal 2 §2.3, Addendum §2.3 — soundings_collection.py already exists*

```
T850_00z         = 850mb temperature from 00Z sounding (°F)
wind_850_dir     = 850mb wind direction
wind_850_spd     = 850mb wind speed (kt)
stability        = T850_00z - surface_temp_00z
T850_delta       = T850_00z(t-1) - T850_00z(t-2)
```

Use the 00Z sounding from day t-1 (available by ~10 PM ET). This is the strongest single predictor of free-atmosphere temperature and can correct NWP bias where NWP's T850 diverges from observations.

Compute upwind temperature using 850mb wind direction (in addition to surface wind from Phase 2). 850mb wind better represents large-scale advection.

#### 3B. Precipitation and Snow from GHCN
*Source: Proposal 2 §2.4, Addendum §3.5*

Expand the existing .dly parser (modify `ELEMENTS_OF_INTEREST` in `data_collection.py`):

```
prcp_nyc_t1      = NYC precipitation (t-1), inches
snow_depth_nyc   = NYC snow depth (t-1), inches
snow_binary      = 1 if snow_depth > 0 (albedo effect)
days_since_rain  = consecutive dry days ending at t-1
prcp_regional    = mean(surrounding_station_PRCP(t-1))
```

**Operational note:** For live inference, derive precipitation from IEM ASOS (precip field). Snow depth may require GHCN with 1-2 day lag — flag this as a known operational gap and test impact of using stale snow depth.

#### 3C. Enhanced Temporal Features
*Source: Addendum §3.4*

```
day_length              = hours of daylight (from latitude + DOY)
solar_elevation_noon    = maximum sun angle
days_since_solstice     = linear lag feature
TMAX_7day_rolling_mean  = recent thermal regime
TMAX_anomaly_from_climo = deviation from climatological mean
heating_degree_days_7d  = trailing cold anomaly
```

#### Phase 3 Validation
- Test T850 impact on high-error spring days specifically
- Test snow/precip impact on winter shoulder-season transitions
- Full error decomposition by season and regime
- **Gate:** Proceed to Phase 4 if OOS MAE ≤ 1.90°F or clear evidence of remaining architectural limitations

---

### Phase 4: Architecture Upgrade
**Goal:** Reduce OOS MAE to ~1.75-1.85°F with improved regime handling
**Dependencies:** Feature pipeline from Phases 2-3 stable and validated

#### 4A. Wind-Gated Attention in MOS-Residual Path
*Source: Proposal 1 §C, Addendum §4.1 — already implemented in src/wind_gated_attention.py*

Integrate the existing `WindGatedAttentionModel` into the MOS correction pathway:

```
Station tensor + metadata + global context → wind-gated attention pooling
Attention output + MOS features + NYC lag → correction head → mu, sigma
Final prediction: MOS_ensemble + mu, with uncertainty sigma
```

Key design decisions from Proposal 2:
- Keep per-station encoder small (2 layers, dim 32)
- Multi-head attention (2-4 heads)
- Wind bias term: `alpha × cos(wind_dir - bearing_i)` (learned alpha)
- Missing-station masking (attention logit → -inf)

**Fallback mode:** When station sparsity is high, downweight the correction and rely more heavily on MOS. Implement as a learned confidence gate.

#### 4B. Mixture Residual Head for Multi-Regime Days
*Source: Proposal 1 §B, Addendum §4.3*

Replace single Gaussian with 2-component Gaussian mixture:

```
Outputs: pi1, mu1, sigma1, mu2, sigma2
Loss: mixture NLL
Final CDF: MOS + mixture(pi1*N(mu1,sigma1) + (1-pi1)*N(mu2,sigma2))
```

Guardrails:
- Keep the backbone tiny; only enlarge the output head
- Entropy regularization on mixture weights to prevent component collapse
- Clip pi1 to [0.1, 0.9] to prevent degeneracy

This directly addresses bimodal forecast scenarios (front arrival timing uncertainty, marine layer onset).

#### 4C. Regime-Aware Multi-Head (MoE Variant)
*Source: Proposal 1 §D, Proposal 2 §3.2*

Single shared encoder + 3 expert heads:
- Expert 1: warm stable regime
- Expert 2: cold advection regime
- Expert 3: transition/front regime

Gating network inputs (all time-safe):
```
sin_day, cos_day, NYC_lag1, sector_gradient_WNW_coastal,
pressure_tendency, mos_error_7d, wind_shift, dewpoint_trend
```

Soft routing avoids the data-halving problem of hard seasonal splits and the month-boundary artifacts of hard season cutoffs.

#### Phase 4 Validation
- Ablation: attention vs. flat MLP on identical features
- Attention weight visualization: do weights match physical expectations on known weather regimes?
- Mixture vs. single Gaussian on frontal-passage days
- MoE gating patterns: do experts specialize as expected?
- **Gate:** Proceed to Phase 5 if architecture provides ≥0.05°F improvement over flat MLP with same features

---

### Phase 5: NWP Synthesis Layer
**Goal:** Reduce OOS MAE to ~1.70-1.80°F with calibrated bucket probabilities
**Dependencies:** Optimized station model from Phases 1-4, GEFSv12 Reforecast data

#### 5A. GEFSv12 Reforecast Data Collection
*Source: Addendum §Option A — nwp_collection.py already exists*

Download GEFSv12 Reforecast (2000-2019) from AWS S3 for NYC grid point using Herbie:
- `tmax_2m` — NWP TMAX forecast (F024)
- `tmp_pres` — 850mb temperature
- `ugrd_10m`, `vgrd_10m` — 10m wind
- `tcdc_eatm` — total cloud cover
- `pres_msl` — MSLP
- `apcp_sfc` — precipitation
- All 5 ensemble members (c00 + p01-p04) for spread computation

Bridge the gap with operational GFS (2021-2024) from AWS.

**Why not ERA5:** ERA5 is an analysis (it knows the answer). GFS is a forecast. Training on actual GFS forecasts eliminates the training-inference mismatch that ERA5 would introduce. The reforecast uses a single frozen model version (GFSv15.1), ensuring consistent bias characteristics across all 20 years.

#### 5B. Synthesis Meta-Learner
*Source: Addendum §9, synthesis_model.py already exists*

```
Inputs:
  station_mu, station_sigma    (from optimized station model)
  nwp_tmax, nwp_t850           (GFS forecasts)
  nwp_wind_dir, nwp_wind_speed
  nwp_cloud_cover
  nwp_ensemble_spread          (GEFS 5-member std)
  station_nwp_gap              (station_mu - nwp_tmax)
  abs_station_nwp_gap
  nwp_recent_bias_7d           (NWP rolling MAE)
  station_recent_bias_7d
  sin_day, cos_day
  front_detected               (from station model)

Architecture: Start with Bayesian linear (BMA-style), then small MLP [64,32]
Output: mu_synth, sigma_synth (calibrated distribution)
Loss: CRPS
```

**Key principle from the Addendum:** The station model has ZERO NWP dependency. If the NWP pipeline breaks, the station model still produces a forecast. The synthesis layer is an enhancement, not a requirement.

#### 5C. Post-Hoc Calibration Layer
*Source: Proposal 1 §7, Addendum §5.3 — calibration.py already exists*

1. Hold out a calibration window (separate from validation): last 20% of validation period
2. Compute raw model CDF at each integer temperature (0-110°F)
3. Fit season-conditional isotonic regression: separate models for DJF, MAM, JJA, SON
4. Evaluate with PIT histogram, interval coverage (50/80/90%), bucket reliability at Kalshi cutpoints
5. Convert CDF → exact Kalshi contract buckets with strict endpoint logic and sum-to-one normalization

#### Phase 5 Validation
- Compare synthesis vs. station-only and NWP-only on all metrics: MAE, CRPS, PIT, interval coverage
- Verify that ensemble spread feature improves calibration (wider spread → wider sigma)
- Bucket reliability evaluation at Kalshi cutpoints
- Verify no regression in any season slice > 0.05°F vs. station-only
- **Gate:** Proceed to Phase 6 if synthesis improves CRPS by ≥5% and bucket Brier score improves

---

### Phase 6: Trading Hardening + Go-Live
**Goal:** Positive expected value after fees on >30% of trading days
**Dependencies:** Calibrated probabilistic model from Phase 5

#### 6A. Kalshi Bucket Evaluation
- Convert calibrated CDF → KXHIGHNY bucket probabilities
- Compare model probabilities vs. historical market prices
- Compute EV per bucket per day on backtest period
- Identify systematic market biases (e.g., does market consistently underprice tail events?)

#### 6B. Trading Strategy
- Kelly criterion with fractional sizing (quarter-Kelly for safety)
- Minimum edge threshold: only trade when |model_prob - market_prob| > slippage + fee margin
- Conservative slippage model from historical orderbook data
- Position limits per contract and per day

#### 6C. Risk Management
- Kill-switch triggers:
  - Calibration drift: if rolling 30-day PIT histogram deviates from uniform by >0.15 (KS test)
  - P&L drawdown: halt if 14-day rolling P&L < -$X
  - Data quality: halt if >5 station observations missing in a single day
- Monthly model re-evaluation against most recent 90-day performance

#### 6D. Operational Pipeline (6:00 AM ET Daily)
```
6:00 AM: Pull IEM ASOS through 10Z → compute day t-1 features
         Pull 00Z IGRA sounding → extract T850, stability
         Pull GFS 00Z F024 for NYC → extract NWP features
         Compute all features → run station model → mu_station, sigma_station
         Run synthesis → mu_synth, sigma_synth
         Apply isotonic calibration → calibrated CDF
         Convert CDF → Kalshi bucket probabilities
         Pull KXHIGHNY orderbooks → compute EV
         Execute trades where edge > threshold
         Log everything
```

#### 6E. Paper Trading
- Run the full pipeline for 30-60 days with no real capital
- Track hypothetical P&L, calibration metrics, and data availability
- Validate that 6 AM pipeline completes reliably
- Document edge cases (missing data, API outages, extreme weather)

---

## Cross-Cutting Concerns

### Training Protocol
- **Loss:** NLL → CRPS (NLL for initial convergence, CRPS fine-tuning for calibration)
- **LR schedule:** Cosine annealing with warm restarts (escape local minima)
- **Weight decay:** Sweep [1e-5, 5e-5, 1e-4, 5e-4] for small correction model
- **EMA weights:** Maintain exponential moving average for evaluation
- **SWA:** After convergence, cyclical LR for N epochs and average weights (flatter minima)

### Two-Stage Training (Phases 2+)
*Source: Proposal 2 §4.2*

1. **Stage 1:** Pre-train the station encoder on 1998-2003 data (station-only, no MOS) to learn spatial temperature relationships
2. **Stage 2:** Fine-tune the full correction model on 2004-2024 (where MOS is available)

This uses 6 years of station data currently wasted.

### Expanding-Window Cross-Validation
*Source: Proposal 2 §4.2*

For hyperparameter selection, use time-series CV instead of single train/val split:
- Fold 1: Train 2004-2015, Val 2016-2017
- Fold 2: Train 2004-2017, Val 2018-2019
- Fold 3: Train 2004-2019, Val 2020-2021
- Average metrics across folds

### Bayesian Hyperparameter Search (Phase 4+)
*Source: Proposal 1 §4*

Constrained search (60-100 trials) over the best 1-2 architectures:
- Hidden sizes: [64,32], [96,48], [128,64]
- Weight decay: 1e-6 to 3e-4 (log-uniform)
- LR: 2e-4 to 2e-3
- Batch size: {128, 256}
- Gradient clip: {2.0, 5.0, 10.0}
- Multi-objective early stop: 0.5×CRPS + 0.3×MAE + 0.2×calibration_error

### Training-Inference Data Source Parity
*Source: Addendum §CRITICAL*

| Feature | Training Source | Operational Source | Mismatch |
|---------|---------------|-------------------|----------|
| Station TMAX/TMIN | IEM ASOS hourly → daily | IEM ASOS hourly → daily | None |
| Wind/dewpoint/pressure | IEM ASOS hourly | IEM ASOS hourly | None |
| Precip (rain) | IEM ASOS | IEM ASOS | None |
| Snow depth | GHCN-Daily | GHCN-Daily (1-2 day lag) | Minor — use stale or impute |
| 850mb temperature | IGRA sounding | IGRA sounding | None |
| NWP forecasts | GEFSv12 reforecast | GFS 00Z operational | Minimal (frozen v15.1 ≈ op v16) |

---

## Acceptance Criteria

A new candidate model replaces C_Correction_NN_tiny only if it beats it on **all** of the following:

1. OOS MAE improvement ≥ 0.08°F (i.e., ≤ 2.01°F)
2. OOS CRPS improvement ≥ 5%
3. Bucket reliability improvement (ECE/Brier) on holdout and 2025 OOS
4. No degradation worse than 0.05°F in any season slice
5. Stable across ≥3 random seeds
6. No calibration drift in replay

---

## Expected Performance Trajectory

| Phase | Changes | Expected OOS MAE | Expected OOS CRPS |
|-------|---------|------------------|--------------------|
| Current | C_Correction_NN_tiny | 2.093°F | — |
| Phase 1 | MOS memory + probabilistic + ensemble | 1.95-2.05°F | ~2.0°F |
| Phase 2 | ASOS wind + pressure + frontal detection | 1.85-1.95°F | ~1.8-1.9°F |
| Phase 3 | T850 + precip/snow + temporal | 1.80-1.90°F | ~1.7-1.8°F |
| Phase 4 | Wind-gated attention + MoE + mixture | 1.75-1.85°F | ~1.6-1.7°F |
| Phase 5 | NWP synthesis + calibration | 1.70-1.80°F | ~1.5-1.7°F |

**Important:** These assume improvements are partially additive. Realistic expectation for full pipeline: **1.80-1.90°F OOS MAE** with well-calibrated distributions. Getting to 1.70°F would be a strong outcome. The theoretical floor is ~1.5°F.

**Conservative seasonal targets (after Phase 5):**
- DJF (Winter): ~1.9-2.1°F
- MAM (Spring): ~2.0-2.3°F (remains hardest)
- JJA (Summer): ~1.4-1.6°F
- SON (Fall): ~1.5-1.7°F

---

## Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Diminishing returns from feature stacking | High | Medium | Test each group in isolation and combination; prune non-contributors |
| ASOS data quality (missing hours, sensor errors) | Medium | Medium | QC filtering; fall back to daily GHCN when hourly missing |
| Feature leakage from rolling MOS error stats | Low | High | Strict lag-1+ construction; shuffled-label sanity check |
| Attention model overfitting | Medium | Medium | Keep encoder small; regularize attention; compare vs flat MLP baseline |
| IGRA sounding gaps (missing 00Z) | Low | Low | Forward-fill ≤2 days; graceful degradation to MOS-only features |
| GEFSv12 → operational GFS model version gap | Low | Low | Include `is_operational_gfs` binary feature; train bias offset |
| IEM-GHCN TMAX systematic difference | Medium | Medium | Cross-validate; train calibration offset or train on IEM exclusively |
| Overfitting to 2025 OOS year | Medium | High | Expanding-window CV; multi-year OOS validation when possible |

---

## Implementation Notes

### Existing Code Assets (Ready to Integrate)
These are already implemented in the codebase and need integration, not creation:

| Module | Status | Integration Needed |
|--------|--------|--------------------|
| `src/wind_gated_attention.py` | Implemented | Connect to MOS-residual path |
| `src/asos_collection.py` | Implemented | Run data download, build daily aggregation |
| `src/asos_preprocessing.py` | Implemented | Connect to feature pipeline |
| `src/operational_features.py` | Implemented | Wind composites, sounding features, synoptic flow |
| `src/nwp_collection.py` | Implemented | GEFSv12 reforecast download |
| `src/nwp_preprocessing.py` | Implemented | NWP feature extraction |
| `src/synthesis_model.py` | Implemented | Station + NWP fusion |
| `src/crps_loss.py` | Implemented | CRPS training loss |
| `src/calibration.py` | Implemented | Isotonic calibration |
| `src/soundings_collection.py` | Implemented | IGRA data download |

### New Code Required
| Task | Phase | Effort |
|------|-------|--------|
| MOS error memory feature engineering | 1 | Low |
| Semi-annual harmonics + gradient features | 1 | Low |
| Heteroscedastic output head for C_Correction_NN_tiny | 1 | Low-Medium |
| IEM ASOS daily aggregation pipeline | 2 | Medium |
| Wind-conditioned composite computation | 2 | Medium |
| Frontal detection features | 2 | Medium |
| GHCN precip/snow parser expansion | 3 | Low |
| MoE gating network | 4 | Medium |
| Mixture density output head | 4 | Medium |
| GEFSv12 download automation | 5 | Medium |
| Synthesis meta-learner training loop | 5 | Medium |
| Season-conditional isotonic calibration | 5 | Medium |
| Kalshi bucket probability converter | 6 | Low-Medium |

---

## Summary

The path to beating the market is not a single breakthrough but a systematic campaign:

1. **Foundation:** Make MOS biases explicit and switch to probabilistic output (Phase 1)
2. **Physics:** Add the atmospheric state information the model is blind to — wind, moisture, pressure, clouds, upper-air (Phases 2-3)
3. **Architecture:** Let the model dynamically weight stations by wind regime and specialize by weather pattern (Phase 4)
4. **Synthesis:** Combine station ground truth with NWP physics for a distribution better than either source alone (Phase 5)
5. **Calibration + Trading:** Translate the distribution into calibrated bucket probabilities and trade where edge exceeds costs (Phase 6)

Each phase is independently valuable and testable. Each has clear acceptance criteria and can be shipped as soon as it passes. The infrastructure for most of this already exists in the codebase — the primary work is integration, feature engineering, and rigorous validation.
