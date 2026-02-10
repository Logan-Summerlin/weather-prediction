# Model Improvement Proposal: Pushing Below 2.0°F MAE

**Date:** 2026-02-10
**Current Best:** C_Correction_NN_tiny — 2.090°F test / 2.093°F OOS
**Goal:** Consistent sub-2.0°F full-year MAE with robust OOS generalization

---

## 1. Diagnosis: Where the Remaining Error Lives

Before proposing solutions, it is critical to understand the structure of the remaining ~2.1°F error.

### 1.1 Seasonal Error Distribution (C_Correction_NN_tiny)

| Season | Test MAE | OOS MAE | Share of Error |
|--------|----------|---------|----------------|
| DJF (Winter) | 2.300°F | 1.938°F | ~27% |
| MAM (Spring) | 2.575°F | 2.863°F | **~31%** |
| JJA (Summer) | 1.782°F | 1.745°F | ~21% |
| SON (Fall) | 1.703°F | 1.820°F | ~21% |

**Spring (MAM) is the dominant error source** — consistently 0.5–0.8°F worse than summer/fall across all models. Spring in the NYC region features:
- Rapid regime transitions (arctic air retreating, subtropical air advancing)
- Frequent frontal passages with large 24-hour temperature swings
- Sea-breeze onset (coastal stations decouple from inland)
- Late-season snow events (albedo shocks)

**Winter OOS is anomalously good** (1.938°F) compared to test (2.300°F), suggesting year-to-year variability rather than model robustness. In contrast, spring OOS (2.863°F) is significantly worse than test (2.575°F), indicating the model struggles with spring regime diversity.

### 1.2 What the Current Model Uses vs. What It Doesn't

**Currently used:**
- Station TMAX and TMIN at lag-1 (48 stations)
- Diurnal range (TMAX-TMIN) per station
- Sector mean temperatures and inter-sector gradients
- 1-day trend features per sector
- MOS ensemble forecast (GFS+NAM average)
- NYC TMAX lag-1 (autoregressive)
- sin/cos day-of-year

**Not used but available or obtainable:**
- Wind direction and speed (from ASOS hourly data)
- Dewpoint / humidity (from ASOS)
- Sea-level pressure and 24h tendency (from ASOS)
- Cloud cover / ceiling (from ASOS)
- 850mb temperature and upper-air wind (from IGRA soundings)
- Precipitation, snowfall, snow depth (from GHCN .dly, already parseable)
- GFS/NAM disagreement (GFS-NAM spread)
- Recent MOS bias/error statistics (rolling windows)
- NWP ensemble spread (GEFS)

### 1.3 Irreducible Error Floor Estimate

MOS ensemble alone achieves 2.51°F MAE. Our best correction model reduces this by ~0.42°F. The question is how much further correction is physically possible.

Estimated error budget:
- **Irreducible weather chaos:** ~1.2–1.5°F (fundamental limit for 24h ahead TMAX)
- **MOS systematic biases:** ~0.3–0.5°F (seasonal, regime-dependent, correctable)
- **Local effects not captured by MOS:** ~0.2–0.4°F (sea breeze, urban heat, elevation gradients)
- **Data noise / measurement error:** ~0.1–0.2°F

This suggests a floor of roughly **1.5–1.8°F** is theoretically achievable with perfect features and correction. Getting from 2.09°F to ~1.8°F requires capturing the systematic MOS biases and local effects that the current model misses.

---

## 2. Highest-Impact Improvements (Tier 1)

These changes target the largest gaps in the current pipeline and are expected to provide the most improvement per unit effort.

### 2.1 Add Atmospheric State Features from ASOS Hourly Data

**Expected impact: 0.10–0.25°F MAE reduction**
**Effort: Medium (data pipeline + feature engineering)**

The project plan identifies IEM ASOS data as "highest priority" but it has not been integrated. Surface observations beyond temperature carry strong predictive signal:

**Wind direction and speed (day t-1):**
- Prevailing wind direction determines which air mass is advecting toward NYC
- Wind direction persistence (variance of direction over 24h) indicates frontal activity
- Evening wind direction (18–00Z) is the best proxy for "what air mass will be over NYC tomorrow"
- **Wind-conditioned upwind temperature:** Instead of treating all 48 stations equally, weight stations by alignment with prevailing wind. On a NW-wind day, Albany and Scranton temperatures are far more predictive than Atlantic City

**Specific features to compute from ASOS hourly:**
```
Per station (14 core ASOS airports):
  - wind_dir_prevailing (vector mean, 0-360°)
  - wind_dir_evening (18-00Z vector mean)
  - wind_speed_mean, wind_speed_max
  - wind_dir_persistence (1 - circular variance; high = steady direction)

Central Park / nearest station:
  - dewpoint_mean, dewpoint_afternoon (proxy for air mass moisture)
  - dewpoint_depression (T - Td; proxy for cloud/mixing potential)
  - slp_00z, slp_12z, slp_24h_tendency (frontal passage signal)
  - cloud_fraction (hours with ceiling < 5000 ft)
```

**Wind-conditioned composite features:**
```
upwind_temp = weighted_avg(station_temps, weight=cos(wind_dir - bearing_to_station))
crosswind_temp = weighted_avg(station_temps, weight=|sin(wind_dir - bearing_to_station)|)
upwind_gradient = upwind_temp - NYC_TMAX(t-1)
advection_proxy = wind_speed × upwind_gradient / mean_upwind_distance
```

**Why this helps:** The current model treats all 48 stations with fixed (learned) weights. But on a NW-wind day, the NW stations carry nearly all the signal while coastal stations are noise. Wind-conditioning collapses the effective dimensionality and provides a physics-based attention signal.

**Data source:** IEM ASOS (Iowa Environmental Mesonet), free, available 1998-present for all ASOS stations.

### 2.2 MOS Error Memory Features

**Expected impact: 0.05–0.15°F MAE reduction**
**Effort: Low (feature engineering only, no new data needed)**

The MOS forecast has regime-dependent biases. For example, it may consistently over-predict during spring sea-breeze events or under-predict during arctic outbreaks. The current correction model must learn these biases implicitly from station observations. We can make them explicit:

**Features to add:**
```
mos_error_7d  = rolling 7-day mean(actual - MOS_ensemble)   # recent bias
mos_error_14d = rolling 14-day mean(actual - MOS_ensemble)  # medium-term bias
mos_error_30d = rolling 30-day mean(actual - MOS_ensemble)  # seasonal bias
mos_abs_error_7d = rolling 7-day mean(|actual - MOS_ensemble|)  # recent skill
mos_error_yesterday = actual(t-1) - MOS_ensemble(t-1)       # most recent error

gfs_nam_spread = |GFS_MOS_TMAX - NAM_MOS_TMAX|              # model disagreement
gfs_nam_sign   = sign(GFS_MOS_TMAX - NAM_MOS_TMAX)          # which model is warmer
```

**Why this helps:** MOS errors are autocorrelated — if MOS was 3°F too warm yesterday, it's likely still biased today (similar synoptic pattern persists for 2-5 days). Making this explicit gives the correction model a "head start." The GFS-NAM spread signals forecast confidence; when models disagree, the correction should be larger or more cautious.

**Implementation note:** Rolling features must use strictly lag-1+ data to avoid leakage. The 7-day rolling mean at day t uses errors from days t-7 through t-1.

### 2.3 Upper-Air Features from IGRA Soundings

**Expected impact: 0.05–0.15°F MAE reduction**
**Effort: Medium (new data source, but well-documented)**

The 850mb temperature (T850) from the 00Z sounding at OKX/Upton (Long Island) is widely considered the strongest single predictor of next-day surface TMAX. It represents the free-atmosphere temperature above the boundary layer and is less affected by local surface effects.

**Features to extract:**
```
T850_00z    = 850mb temperature at 00Z (°F)
wind_850_dir = 850mb wind direction (°)
wind_850_spd = 850mb wind speed (kt)
stability   = T850_00z - surface_temp_00z  # positive = stable (inversion)
T850_delta  = T850_00z(t-1) - T850_00z(t-2)  # 850mb trend
```

**Why this helps:** Surface TMAX is strongly controlled by the temperature of the air mass aloft. T850 captures this directly, independent of local surface heating/cooling artifacts. The stability indicator helps predict whether surface temps will decouple from the free atmosphere (strong inversions trap cold surface air). The MOS itself is partially derived from NWP T850, but the sounding is the actual observation — any NWP bias in T850 propagates into MOS, and the sounding observation can correct it.

**Data source:** IGRA via Siphon or direct download. Station USM00072501 (OKX/Upton). Available 1946-present.

### 2.4 Precipitation and Snow Features from GHCN

**Expected impact: 0.03–0.08°F MAE reduction**
**Effort: Low (data already downloadable via existing .dly parser)**

Snow cover and recent precipitation affect next-day TMAX through albedo, soil moisture, and latent heat effects. These are already parseable from the .dly files using the existing data_collection infrastructure.

**Features to add:**
```
prcp_nyc_t1     = NYC precipitation (t-1), inches
snow_depth_nyc  = NYC snow depth (t-1), inches
snow_binary     = 1 if snow_depth > 0  # albedo effect indicator
days_since_rain = count of consecutive dry days ending at t-1
prcp_regional   = mean(surrounding_station_PRCP(t-1))
snow_regional   = mean(surrounding_station_SNWD(t-1))
```

**Why this helps:** Fresh snow cover raises albedo dramatically, suppressing TMAX. Recent heavy rain increases soil moisture, favoring latent heat flux over sensible heat, which suppresses TMAX. The MOS partially accounts for this, but the observation-based snow/precip features provide ground truth that the NWP may miss (e.g., NWP underestimates snow cover after a surprise storm).

---

## 3. Architecture Improvements (Tier 2)

These changes address structural limitations in the model architecture. They have high potential but carry more implementation risk.

### 3.1 Attention-Based Station Pooling

**Expected impact: 0.05–0.15°F MAE reduction (especially with wind conditioning)**
**Effort: Medium-High**

The current model flattens all 48 stations into a single vector and feeds it through an MLP. This treats stations as independent features with fixed importance. An attention-based architecture learns dynamic, per-day station weighting:

```
Architecture:
  1) Per-station encoder (shared weights across all stations):
     Input: [TMAX_i, TMIN_i, diurnal_i, distance_i, bearing_i, elevation_i]
     Output: station_embedding_i (dim 32)

  2) Attention pooling:
     Query: global_context = [NYC_lag1, MOS_forecast, wind_dir, sin_day, cos_day]
     Keys/Values: station_embeddings
     Attention logit: dot(W_q * query, W_k * embedding_i)
                      + α * cos(wind_dir - bearing_i)  # wind bias term
     Output: weighted sum of station embeddings

  3) Correction head:
     Input: [attention_output, MOS_forecast, NYC_lag1, mos_error_features, ...]
     Output: MOS residual correction (scalar)
```

**Why this helps:**
- **Dynamic station weighting:** On NW-wind days, the model naturally up-weights NW stations. On coastal-flow days, it up-weights Long Island/NJ shore stations.
- **Missing station handling:** Attention naturally masks out missing stations (set attention logit to -inf).
- **Interpretability:** Attention weights reveal which stations the model relies on, enabling physical validation.
- **Wind-gated bias term:** The `α * cos(wind_dir - bearing_i)` term provides a physics-informed inductive bias. The model can learn to override it when data warrants, but it starts from a physically sensible prior.

**Key design decisions:**
- Keep the per-station encoder small (2 layers, dim 32) to avoid overfitting with 48 stations × ~6 features each
- Use multi-head attention (2-4 heads) to capture different spatial patterns simultaneously
- Train with the same MOS correction objective (predict residual)

### 3.2 Mixture of Experts for Season/Regime Specialization

**Expected impact: 0.05–0.10°F MAE reduction**
**Effort: Medium**

The seasonal results show a persistent gap: spring MAE is 0.5–0.8°F worse than fall/summer. Rather than training separate seasonal models (which halve the training data), use a Mixture of Experts (MoE) with a learned gating network:

```
Architecture:
  Expert_1: [32, 16] NN (cold-season specialist)
  Expert_2: [32, 16] NN (warm-season specialist)
  Expert_3: [32, 16] NN (transition specialist)

  Gating network:
    Input: [sin_day, cos_day, NYC_lag1, sector_gradient_WNW_coastal,
            slp_tendency, mos_error_7d]
    Output: softmax weights over 3 experts

  Final prediction: Σ gate_i * expert_i(features)
```

**Why this helps:**
- Spring errors are disproportionate because spring weather patterns are fundamentally different (competing air masses, sea-breeze onset). A dedicated "transition expert" can specialize.
- The gating network learns when to activate each expert, so it's not a hard seasonal cutoff (which fails in anomalous years).
- All experts share the same training data (with soft routing), avoiding the data-halving problem of separate seasonal models.

**Evidence from existing results:**
- E_warm_Ridge (1.959°F) and E_cold_NN (2.503°F) show that season-specific models work for warm season
- But the cold model degrades significantly, and blended models (E_Blended_Ridge: 2.187°F OOS) don't match the warm specialist
- MoE with soft gating avoids the hard boundary problem

### 3.3 Stochastic Weight Averaging (SWA) and Multi-Seed Ensemble

**Expected impact: 0.02–0.05°F MAE reduction**
**Effort: Low**

Neural networks have inherent stochasticity from random initialization and mini-batch sampling. Simple ensembling reduces this variance:

**SWA:** After standard training converges, continue training with a cyclical or constant learning rate and average the weights from the last N checkpoints. This provably finds flatter minima with better generalization.

**Multi-seed ensemble:** Train 5 copies of C_Correction_NN_tiny with different random seeds, average predictions. This is the simplest possible ensemble and consistently reduces MAE by 1-3% in regression problems.

**Why this helps:** The C_Correction_NN_tiny has a 0.003°F IS/OOS gap, suggesting low variance. But the model is small (32+16 neurons) and may have multiple local minima. Averaging smooths out per-instance prediction noise.

---

## 4. Parameter & Training Improvements (Tier 3)

### 4.1 Training Configuration Refinements

**Learning rate schedule:** Replace ReduceLROnPlateau with cosine annealing with warm restarts (CosineAnnealingWarmRestarts). This avoids getting stuck in local minima early and provides multiple opportunities to escape.

**Batch size:** The architecture sweep found minimal batch-size impact (64-256). However, for the small correction model, a smaller batch size (32) may help by injecting more noise during training, acting as implicit regularization.

**Weight decay tuning:** Current weight_decay=1e-5. For the small correction model, this may be too low. Sweep [1e-5, 5e-5, 1e-4, 5e-4]. Higher weight decay can improve generalization for small models.

**Gradient clipping:** Currently 5.0. For residual correction (small targets), this is likely never triggered. Could tighten to 1.0 for more stable training.

**Loss function:** The correction model uses Huber or L1 loss. For spring (high-error season), consider a weighted loss that upweights spring samples by 1.2-1.5x. This trades off summer/fall MAE for spring MAE improvement, potentially reducing overall MAE since spring is the bottleneck.

### 4.2 Expanded Training Window Strategies

**Two-stage training for the correction model:**
1. Stage 1: Pre-train the station encoder on 1998-2003 data (station-only, no MOS) to learn spatial temperature relationships
2. Stage 2: Fine-tune the full correction model on 2004-2020 (where MOS is available)

This uses 6 additional years of station data that are currently wasted (1998-2003 have no MOS).

**Time-series cross-validation:** Instead of a single train/val split, use expanding-window CV:
- Fold 1: Train 2004-2015, Val 2016-2017
- Fold 2: Train 2004-2017, Val 2018-2019
- Fold 3: Train 2004-2019, Val 2020-2021
- Average metrics across folds for hyperparameter selection

This provides more reliable estimates and uses validation data more efficiently.

---

## 5. Feature Interaction & Nonlinear Feature Engineering (Tier 3)

### 5.1 MOS × Station Interaction Features

```
mos_station_gap = MOS_ensemble - NYC_TMAX(t-1)     # MOS change signal
mos_sector_gap  = MOS_ensemble - mean_WNW_TMAX(t-1) # MOS vs upstream
station_mos_agree = sign(delta_T_stations) == sign(mos_station_gap)  # agreement
```

When MOS and station observations agree on direction, confidence is high. When they disagree, the model should be more cautious.

### 5.2 Gradient Magnitude and Direction Features

The current sector gradients (WNW-coastal, SW-WNW) are useful but static. Add:

```
max_gradient     = max over all station pairs of |T_i - T_j| / distance(i,j)
gradient_bearing = bearing of the maximum gradient (points toward warmer air)
gradient_persistence = cos(gradient_bearing(t-1) - gradient_bearing(t-2))
```

Large gradients with persistent direction indicate a strong frontal zone. The gradient bearing indicates from which direction the temperature change is arriving.

### 5.3 Semi-Annual and Higher Harmonics

Currently only the annual cycle is encoded (sin/cos with period 365.25 days). Add:

```
sin_day_semi = sin(4π × doy / 365.25)  # semi-annual (captures equinox effects)
cos_day_semi = cos(4π × doy / 365.25)
```

This captures the fact that spring and fall transitions have different character from deep winter and deep summer, which a single annual harmonic cannot represent.

---

## 6. Additional Data Sources (Tier 4 — Higher Effort)

### 6.1 GFS/GEFS Ensemble Forecasts (Synthesis Layer)

**Expected impact: 0.10–0.20°F MAE reduction beyond current MOS correction**
**Effort: High (new data pipeline)**

The MOS forecast is derived from NWP, but it's a lossy compression. The raw GFS grid-point forecast contains additional information (T850, wind fields, moisture, cloud cover, precipitation type) that MOS discards. A synthesis layer that ingests raw NWP alongside the station model could capture:
- NWP systematic biases that MOS doesn't fully correct
- NWP ensemble spread as a confidence signal
- NWP fields (cloud cover, precipitation) that the station model doesn't see

**Implementation:** Use Herbie or AWS S3 to download GFS 00Z F024 for the NYC grid point. Extract TMAX, T850, 10m wind, total cloud cover, MSLP, CAPE, precipitation. Train a meta-learner on station model output + NWP features.

### 6.2 Extended Historical Data (50+ Years)

The current model trains on 1998-2020 for stations (23 years) and 2004-2020 for MOS-based models (17 years). Extending to 50+ years would provide:
- More extreme event examples (rare cold outbreaks, heat waves)
- Better climatological baselines
- More training data for the attention/MoE models

**Caveat:** Climate non-stationarity. Temperatures have trended upward ~2°F over 50 years. Need to either detrend or include a year/decade feature.

---

## 7. Prioritized Implementation Roadmap

Ranked by expected impact per unit effort:

| Priority | Improvement | Expected Δ MAE | Effort | Cumulative |
|----------|------------|----------------|--------|------------|
| **P0** | MOS error memory features (§2.2) | -0.05 to -0.15°F | Low | ~1.95–2.04°F |
| **P1** | Multi-seed ensemble (§3.3) | -0.02 to -0.05°F | Low | ~1.93–1.99°F |
| **P2** | Semi-annual harmonics + gradient features (§5) | -0.02 to -0.05°F | Low | ~1.91–1.94°F |
| **P3** | ASOS wind + pressure features (§2.1) | -0.10 to -0.25°F | Medium | ~1.80–1.84°F |
| **P4** | Precipitation & snow from GHCN (§2.4) | -0.03 to -0.08°F | Low | ~1.77–1.81°F |
| **P5** | Upper-air T850 from IGRA (§2.3) | -0.05 to -0.15°F | Medium | ~1.72–1.76°F |
| **P6** | Attention-based station pooling (§3.1) | -0.05 to -0.15°F | Medium-High | ~1.67–1.71°F |
| **P7** | Mixture of Experts (§3.2) | -0.05 to -0.10°F | Medium | ~1.62–1.66°F |
| **P8** | GFS/GEFS synthesis layer (§6.1) | -0.10 to -0.20°F | High | ~1.52–1.56°F |

**Conservative realistic estimate with P0–P5:** ~1.80°F full-year MAE
**Optimistic estimate with all tiers:** ~1.60°F full-year MAE
**Important:** These estimates assume improvements are additive, which they won't fully be. Realistic expectation with P0–P5 is **1.85–1.95°F**.

---

## 8. What Won't Help (Based on Evidence)

The optimization work has already ruled out several approaches. These should **not** be revisited:

1. **Deeper/wider station-only NNs:** Without MOS, station-only models plateau at ~4.0°F regardless of architecture. Adding layers or neurons won't break this ceiling.

2. **LSTM/GRU for temperature sequences:** Tested extensively (§Analyst 4). LSTM (4.450°F) and GRU (4.674°F) are worse than flat MLPs. The temporal structure in daily TMAX is better captured by lag features than recurrent architectures.

3. **Temporal 1D convolution:** 4.497°F — worse than flat MLP. The 2-3 day temporal patterns don't have the local structure that convolutions exploit.

4. **Stacking with NN meta-learner:** D_Stack_NN (4.269°F test) massively overfits compared to D_Stack_Ridge (2.158°F). Ridge or simple averaging is better for meta-learning.

5. **Dropout > 0.1:** Architecture sweep found dropout=0.0 optimal for station-only models. For the small correction model, 0.2 is used, which seems appropriate given the small architecture.

6. **Large batch sizes (>256):** Minimal impact in all experiments.

---

## 9. Recommended Experiment Plan

### Phase A: Quick Wins (P0 + P1 + P2)
- Add MOS error memory features to existing C_Correction_NN_tiny pipeline
- Train 5-seed ensemble
- Add semi-annual harmonics and gradient magnitude features
- **Validation:** Compare test and OOS MAE against current 2.090/2.093°F baseline
- **Expected result:** 1.93–2.00°F

### Phase B: ASOS Integration (P3)
- Download IEM ASOS hourly data for 14 core airports (1998-present)
- Implement daily aggregation (wind, dewpoint, pressure, cloud)
- Compute wind-conditioned upwind/crosswind features
- Retrain C_Correction_NN_tiny with expanded feature set
- **Validation:** Seasonal breakdown, especially spring MAE improvement
- **Expected result:** 1.80–1.90°F

### Phase C: Additional Observations (P4 + P5)
- Parse PRCP/SNOW/SNWD from existing GHCN .dly files
- Download IGRA soundings for OKX/Upton
- Add snow/precip and T850 features
- **Validation:** Full seasonal + extreme-day error analysis
- **Expected result:** 1.75–1.85°F

### Phase D: Architecture Upgrade (P6 + P7)
- Implement attention-based station pooling with wind gating
- Implement Mixture of Experts with season/regime gating
- Compare against flat MLP correction on same features
- **Validation:** Attention weight analysis (do weights match physical expectations?)
- **Expected result:** 1.70–1.80°F

### Phase E: NWP Synthesis (P8)
- Download GFS/GEFS forecasts for NYC grid point
- Build synthesis meta-learner (station model + NWP features)
- **Validation:** Compare against MOS-only and station-only baselines
- **Expected result:** 1.60–1.75°F

---

## 10. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ASOS data quality issues (missing hours, sensor errors) | Medium | Medium | QC filtering, fall back to daily GHCN when hourly missing |
| Feature leakage from rolling MOS error stats | Low | High | Strict lag-1+ construction; verify with shuffled-label test |
| Attention model overfitting (small data relative to params) | Medium | Medium | Keep per-station encoder small; regularize attention weights |
| IGRA sounding gaps (missing 00Z obs) | Low | Low | Forward-fill ≤2 days, graceful degradation to MOS-only |
| Diminishing returns from feature stacking | High | Medium | Test each feature group in isolation and in combination; prune non-contributing features |
| Climate non-stationarity with extended data | Medium | Low | Include year as feature; test with 25yr vs 50yr window |

---

## 11. Summary

The current C_Correction_NN_tiny model at 2.09°F MAE is a strong baseline that exploits the single most important lever (MOS integration). The remaining error is concentrated in **spring (MAM)** and on **high-gradient days** where the model lacks the atmospheric state information to predict regime transitions.

The three highest-leverage improvements are:
1. **MOS error memory features** (low effort, immediate impact from autocorrelated MOS biases)
2. **ASOS wind and pressure features** (medium effort, enables wind-conditioned station weighting and frontal detection)
3. **Upper-air T850 observations** (medium effort, strongest single predictor of free-atmosphere temperature)

Together with architectural refinements (attention pooling, MoE), these changes target the **physical mechanisms** driving the remaining error — air mass advection, frontal passages, and regime transitions — rather than simply adding more of the same type of data.

A realistic full-year target with P0–P5 is **1.85–1.95°F MAE**, with the warm season (JJA+SON) likely reaching **1.5–1.6°F** and spring remaining the hardest at **2.0–2.3°F**.
