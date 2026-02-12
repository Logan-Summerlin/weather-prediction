# ML & Neural Networks for Operational Temperature Forecasting (NYC-focused)
_1000-line LLM reference summary derived from the provided meteorological research report (compiled 2026-02-11)._
## How to use this document
- Each bullet is intentionally short (one idea per line) so an LLM can retrieve and stitch guidance reliably.
- Focus: daily temperature (T2m) forecasting for NYC / coastal-urban midlatitude settings, but methods generalize.
- Emphasis: operational realism (latency, updates, QC, model-change drift), not just offline ML benchmarks.

## Table of contents (high-level)
- 1. Forecasting as a decision workflow
- 2. Temperature physics you must encode implicitly or explicitly
- 3. NWP model ecosystem and guidance products
- 4. Downscaling and local effects in NYC
- 5. Observations, quality control, and data assimilation context
- 6. Ensembles, post-processing, and uncertainty
- 7. Verification, monitoring, and model-change management
- 8. Reproducible public-data pipeline (labels + predictors)
- 9. ML formulations for temperature forecasting
- 10. Neural architectures and feature design patterns
- 11. Training, evaluation, and deployment in operations
- 12. Failure modes, guardrails, and interpretability
- 13. Glossary and operational checklists

## 1) Forecasting as a decision workflow (what ops forecasters actually do)
- Temperature forecasting is not one model; it is a workflow that blends guidance, local knowledge, and last-minute updates.
- Operational meteorologists treat NWP output as a first guess, then adjust using bias knowledge, obs trends, and ensemble spread.
- Daily max/min temperature forecasts are often produced as point values plus probabilistic ranges (e.g., 10th/50th/90th).
- Core idea for ML: emulate the workflow (baseline + corrections + uncertainty), not just a single deterministic mapping.
- Operational cadence matters: some inputs update hourly (e.g., HRRR), others 6-hourly (e.g., global models), others lag (URMA).
- Latency constraints define what is usable at forecast time; training must respect ‘information sets’ by valid-time.
- NYC is a ‘hard’ site: coastal gradients, urban heat island, sea-breeze timing, and boundary-layer errors dominate.
- Forecasters segment the problem into regimes (clear vs cloudy, windy vs calm, frontal passages, marine flow).
- ML should learn regime-dependent corrections, or at least be flexible enough to represent them.
- Typical ops decomposition: (1) synoptic setup, (2) mesoscale details, (3) local station bias, (4) nowcast adjustments.
- An ML system can be layered similarly: global-context module + local-downscaling module + bias-correction module.
- Never evaluate ML in a way that ignores operations (e.g., using future observations as predictors via leakage).
- Forecast targets should be explicit: T2m hourly, daily Tmin, daily Tmax, or daily mean, each with different error sources.
- Stakeholders care about thresholds (freeze, heat advisory) as much as MAE; include event metrics.
- Deliverables in ops: grids (for maps), points (for airports), and text guidance; ML outputs should match these formats.

## 2) Temperature drivers you must model (physics/meteorology in ML terms)
- Temperature at 2 m is controlled by surface energy balance + boundary-layer mixing + advection + radiative forcing.
- Key forcings: shortwave (sun), longwave (clouds, humidity), sensible heat flux (wind, stability), latent heat (evaporation).
- Cloud cover errors are a first-order driver of Tmax errors; nocturnal mixing and clouds drive Tmin errors.
- Wind speed/direction changes both advection (airmass) and mixing (decoupling at night).
- Boundary-layer depth errors cause large temperature biases, especially during morning transition and evening collapse.
- Urban surfaces (NYC) store heat; UHI signal is strongest at night and under calm, clear conditions.
- Coastal proximity creates marine moderation: sea surface temperature and onshore winds cap daytime warming.
- Sea-breeze timing is a classic NYC failure mode; a 1–2 hour timing error can shift Tmax several degrees.
- Frontal passages create discontinuities; ML should treat them as change-points, not smooth trends.
- Cold-air damming and shallow cold pools can persist; models may scour them too fast; ML can learn persistence signals.
- Snow cover dramatically changes albedo and surface fluxes; include snow/ice predictors when relevant.
- Soil moisture influences heat extremes via evaporative cooling; include land-surface state predictors when possible.
- Terrain is mild in NYC but still matters via elevation and land/water masks across the metro area.
- Radiation fog/low stratus can limit warming; satellite-derived cloud/IR features can help nowcasts.
- Precipitation impacts temperature via evaporative cooling and cloud shading; even ‘temperature-only’ tasks benefit from precip proxies.
- Synoptic patterns (ridge/trough, cyclones) set the airmass and cloud/wind regimes; global models capture this well.
- Seasonality is not just a sine wave; it interacts with regimes (e.g., winter radiative cooling vs summer convection).
- Diurnal cycle is strong; represent time-of-day explicitly (harmonics) or via sequence models.
- Observation representativeness matters: airports vs park stations vs rooftop sensors differ systematically.
- Biases are not constant: they vary by regime, season, lead time, and upstream model version changes.

## 3) NWP ecosystem and guidance products (what to ingest and why)
- Numerical Weather Prediction (NWP) provides gridded forecasts from dynamical models; ML commonly post-processes them.
- Operational temperature forecasts often start from short-range convection-permitting models for near-term details (hours–18h).
- For NYC, a common predictor stack includes: HRRR/RAP (short range), NAM (mesoscale), GFS/ECMWF (synoptic), ensembles (GEFS/ENS), and blends (NBM).
- HRRR: high-resolution, rapid-update, strong for boundary-layer evolution and near-term temperature trends.
- RAP: hourly-updating mesoscale analysis/forecast that provides broader context and often feeds short-range guidance.
- NAM: mesoscale model suite; can offer alternative physics/solutions and is useful as an additional ensemble member.
- GFS: global deterministic model; strong for synoptic-scale temperature advection and fronts at 1–10 day scales.
- ECMWF: high-performing global system (HRES/ENS); valuable for medium-range temperature distributions.
- GEFS: global ensemble; use spread and quantiles as uncertainty features for Tmin/Tmax ranges.
- NBM: National Blend of Models; effectively an operationally-calibrated blend and a strong baseline for ML residual learning.
- MOS (Model Output Statistics): legacy statistical post-processing; conceptually similar to modern ML calibration layers.
- LAMP-like updates: short-fuse statistical/ML updates that incorporate recent obs trends to adjust near-term guidance.
- RTMA/URMA: high-resolution analysis products useful as ‘truth’ labels for gridded temperature (RTMA near-real-time; URMA best estimate with lag).
- NDFD/point forecasts: operational gridded products; useful for baselines and for matching what end users consume.
- Model cycles and run times are part of the data schema; encode initialization time and forecast lead explicitly.
- Model upgrades cause non-stationarity; track model version/change dates and allow re-training or adaptation.
- Operational blending logic: prefer higher-res short-range for day-0 details, global ensembles for day-3+ distributions.
- Do not treat NWP as ground truth; treat it as a structured prior that ML can correct and calibrate.
- Gridded predictors are multi-field: not just T2m, but winds, clouds, radiation, boundary-layer height, and surface flux proxies.
- Even if you forecast temperature only, you should ingest multi-variable NWP fields to let the network infer regimes.

## 4) Downscaling and local modeling (NYC coastal-urban specifics)
- Downscaling is mapping coarse guidance to local truth; it can be dynamical (nested models) or statistical/ML.
- NYC challenges: land–water contrasts (Atlantic/Long Island Sound), complex coastline, and heterogeneous urban surfaces.
- Local gradients mean a single ‘NYC temperature’ is ambiguous; decide on targets (e.g., Central Park, JFK, LGA, EWR, or a grid).
- Station choice changes the label distribution; airports often have different exposure and microclimate than park stations.
- Use station metadata (lat, lon, elevation, land/water fraction, urban index) as static features.
- Include coastal distance and upwind water path length as engineered features for onshore-flow regimes.
- Include wind-direction-dependent coastal influence (e.g., onshore vs offshore sectors) via trig transforms of wind direction.
- Sea-breeze detection features: low-level wind shift, pressure tendency, coastal temperature gradient, and time-of-day.
- ML downscaling can be framed as: (a) bias correction of a baseline (NBM/HRRR), or (b) direct prediction from multiscale inputs.
- For gridded downscaling, CNNs can learn coastline-aligned patterns; for station networks, GNNs can learn spatial correlations.
- Statistical downscaling must preserve physical constraints (e.g., Tmax ≥ Tmin); enforce via model structure or post-processing.
- Quantile mapping is a simple bias correction; ML can generalize it with covariate-dependent quantile transforms.
- Urban heat island is regime-dependent; include stability proxies (wind speed, cloud cover) to learn when UHI amplifies.
- Boundary-layer height and mixing proxies help nocturnal Tmin; if not available, use wind + cloud + recent cooling rate.
- Land-surface state (soil moisture, snow cover) affects extremes; include if accessible via NWP or reanalysis.
- Downscaling should be trained separately by season or with explicit seasonal conditioning to reduce confounding.
- High-frequency near-term updates should emphasize recent observations (persistence + trends) more than distant synoptic predictors.

## 5) Observations, QC, and assimilation context (what ‘truth’ means)
- Operational systems fuse observations into analyses via data assimilation (DA); ML users typically consume the resulting analyses.
- Observation types relevant to temperature: METAR (airports), surface mesonets, buoys, satellite radiances, and reanalyses.
- METAR stations (e.g., NYC-area airports) are high-quality, frequent, and operationally central for point temperature verification.
- Gridded ‘truth’ options: RTMA (near-real-time analysis) and URMA (more complete, lagged analysis) for hourly T2m.
- Label selection is crucial: if you train on RTMA, you learn near-real-time noise; if you train on URMA, you accept latency but better QC.
- Observation QC steps: range checks, step checks, climatology checks, buddy checks (spatial consistency), and sensor status flags.
- Missing data is common; decide whether to impute, mask, or drop examples; for sequence models, masking is preferred.
- Representativeness error: a station measures a micro-site, while a model grid is an average; learnable offsets may be needed.
- DA changes over time (new satellites, new QC); this introduces non-stationarity into your labels.
- If you use reanalysis or analysis fields as predictors, ensure their availability at forecast issuance time to avoid leakage.
- Time alignment: always use valid times; never join by ‘date’ without specifying timezone and UTC conversion.
- NYC area is sensitive to marine layer and boundary-layer structure; if you can ingest sounding-derived features, do so.
- Even without upper-air obs, NWP-derived stability indices (e.g., lapse rates) can proxy boundary-layer behavior.
- Keep raw obs and QC’d obs; store QC flags so ML can learn when sensors are unreliable or outliers.

## 6) Ensembles, post-processing, and uncertainty (from guidance to calibrated distributions)
- Ensembles approximate forecast uncertainty via multiple model members or perturbed initial conditions.
- Operationally, ensemble spread is informative but not perfectly calibrated; post-processing is required.
- NBM is an example of systematic blending + post-processing; it often yields strong baseline probabilistic temperature guidance.
- MOS is the canonical post-processing concept: map model outputs to station truth using statistical relationships.
- LAMP-like approaches update MOS using very recent observations to improve short-range temperature guidance.
- Bias correction can be static (mean bias per station/lead) or dynamic (state-dependent, e.g., Kalman filter).
- Kalman-filter bias correction idea: treat bias as a latent state that evolves; update with new errors over time.
- Modern ML generalization: learn bias as a function of covariates (regime) and allow time-varying components.
- Probabilistic ML outputs are preferred: predict quantiles (e.g., 10/50/90) or a parametric distribution (e.g., Gaussian).
- For quantile prediction, use pinball loss; for full distribution, use negative log-likelihood (NLL).
- Calibrate probabilistic outputs with reliability diagrams and CRPS; adjust via isotonic regression or learned calibration layers.
- Ensemble model output statistics (EMOS) is a strong baseline: fit mean/variance as functions of ensemble mean/spread.
- Combine ML with ensembles by learning: (a) corrected member-wise outputs, (b) corrected ensemble moments, or (c) direct quantiles.
- Sharpness vs reliability tradeoff: avoid overconfident distributions that look good on MAE but fail in tail events.
- Provide decision-friendly uncertainty: probability of exceeding thresholds (e.g., Tmax ≥ 90F), plus prediction intervals.
- Communicate uncertainty differently by lead time: day-1 should be sharper; day-5 should be broader and regime-driven.

## 7) Verification metrics and performance monitoring (what to optimize and how to trust it)
- Use multiple error metrics: MAE (robust), RMSE (penalizes outliers), bias (systematic), and skill vs baseline (NBM).
- Stratify verification by season, time-of-day, lead time, regime (clear/cloudy), wind regime, and coastal/onshore flow.
- For probabilistic forecasts, use CRPS, pinball loss (quantiles), Brier score (events), and reliability diagrams.
- Track coverage: e.g., fraction of observations within 80% interval should be ~0.8 if calibrated.
- Monitor drift: model upgrades, new sensors, and seasonal shifts will change error distributions.
- Operational monitoring should be automated and daily: dashboards for bias, MAE, and reliability by lead time.
- Use rolling windows (e.g., last 30/90 days) to detect regressions quickly without overreacting to noise.
- Backtesting must simulate real availability: re-create ‘as-of’ data snapshots; don’t use reprocessed archives naively.
- Perform ablations: quantify value added from each predictor family (HRRR, GFS/GEFS, satellite, NBM baseline).
- Treat NBM as a competitive baseline; aim for incremental skill, not raw error minimization in isolation.
- Confirm robustness to rare events: heat waves, Arctic outbreaks, rapid frontal passages, snow cover transitions.
- Include ‘human factors’: produce explanations of large errors (e.g., sea-breeze timing) to guide future feature improvements.

## 8) Reproducible public-data pipeline for ML daily forecasts (NYC)
- Core principle: build a fully reproducible pipeline using public datasets with known cadences and versioning.
- Define the label first, then match predictors: labels can be (a) point METAR temperatures, or (b) gridded RTMA/URMA T2m.
- Point-label pipeline is simplest for daily Tmax/Tmin at airports; gridded labels enable map-like outputs.
- Recommended labels (gridded): RTMA/URMA hourly 2.5 km; choose URMA for best quality but accept time lag.
- Recommended labels (point): METAR / ISD station series (e.g., JFK/LGA/EWR) for verification and training targets.
- Short-range predictors: HRRR (3 km) provides near-term temperature evolution; pull hourly cycles for 0–18 h guidance.
- Additional short-range predictors: RAP provides hourly analyses and forecasts; useful for broader mesoscale context.
- Synoptic predictors: GFS deterministic + GEFS ensemble for airmass and fronts; update 6-hourly.
- Blended baseline: NBM hourly grids are a strong operational baseline; use as direct input and/or residual target.
- Nowcast context: GOES-East satellite products provide cloud and radiative context; helpful when clouds dominate T errors.
- Ingestion cadence design: hourly loop (HRRR/RAP/RTMA), 6-hour loop (GFS/GEFS), lagged backfill loop (URMA).
- Store all datasets with explicit init_time, valid_time, lead_time, and source_model identifiers.
- Normalize units and grids: standardize temperature units (K vs C vs F) and coordinate conventions (lat/lon, grid spacing).
- Subsetting strategy: for NYC, use a bounding box large enough to capture upstream flow (e.g., Mid-Atlantic + New England context).
- Keep raw GRIB2/NetCDF alongside processed tensors; reprocessing is often needed after bug fixes.
- Track missingness: each variable should have availability flags so the model can degrade gracefully when inputs are missing.
- Keep station metadata separate and versioned; station moves/changes can alter time series.
- Document everything: dataset URLs, update schedules, and any regridding/interpolation methods.

## 9) ML formulations for temperature forecasting (turn ops into supervised learning)
- Choose a target: hourly T2m sequence, or daily aggregates (Tmin/Tmax) derived from the hourly truth series.
- Forecast horizons: nowcast (0–6h), short-range (6–48h), and medium-range (2–10d) require different input emphasis.
- Most operationally effective pattern: residual learning on a strong baseline (e.g., y = NBM + ML_correction).
- Residual learning reduces the burden on the network and improves stability across model upgrades.
- Alternative: direct prediction from multi-model predictors; use when baselines are unavailable or you want grid outputs.
- Multi-task learning helps: predict Tmax, Tmin, and hourly T together; shared representations capture regimes.
- Quantile regression formulation: output q10, q50, q90 for each target and lead; optimize pinball loss.
- Distributional formulation: output mean and scale (and possibly skew) for each lead; optimize NLL + regularization.
- Classification heads: probability of exceedance for thresholds (e.g., freeze, 90F) can be trained jointly.
- Hierarchical outputs: (1) synoptic-scale latent, (2) local adjustment; encourages generalization across nearby sites.
- Spatial formulation for grids: input multi-channel gridded fields; output a temperature grid (downscaled/corrected).
- Graph formulation for stations: nodes are stations; edges encode distance/upwind similarity; output station temperatures.
- Sequence-to-sequence formulation: input past obs + past model errors; output future correction trajectory.
- Online learning formulation: update bias states daily (Kalman-like) while keeping a slower neural backbone fixed.
- Ensembling ML models often beats a single model; use diverse architectures or training seeds for robustness.

## 10) Neural architectures and feature patterns (practical recipes)
- Start simple: linear regression or gradient boosting on NBM/HRRR predictors + station metadata is a strong baseline.
- Then add neural nets where they provide structure: sequences, grids, multi-scale context, and probabilistic outputs.
- MLP (tabular): best for point forecasts with engineered features; fast and stable; easy to regularize.
- Temporal models: 1D CNNs or Transformers over time capture diurnal patterns and regime persistence.
- ConvLSTM/Temporal CNN: useful when you have gridded inputs over time (e.g., HRRR fields + GOES cloud fields).
- 2D CNN encoder: learns coastline/urban patterns from grids; pair with a decoder for station points or output grids.
- U-Net style: good for spatial downscaling where you want fine-resolution corrected temperature grids.
- Vision Transformer (ViT)-like encoders: can ingest multi-channel meteorological fields; consider patching by km-scale blocks.
- GNNs: represent station networks; edge features can include distance, bearing, and climatological correlation.
- Hybrid models: CNN for grids + GNN for stations (grid-to-station attention) can be strong for metro-area forecasts.
- Attention mechanisms can focus on upstream regions based on wind direction; implement via learnable spatial attention.
- Use embeddings for categorical features: month, hour, model_source, station_id, and forecast lead bucket.
- Encode cyclical time with sin/cos of hour-of-day and day-of-year; avoid discontinuities at boundaries.
- Engineer regime indicators: cloud fraction, wind speed, stability proxy, precipitation flag; these reduce data needs.
- Use ensemble features: mean, spread, percentiles of GEFS/ENS for temperature and key drivers (wind, clouds).
- Loss functions: MAE for deterministic; pinball for quantiles; CRPS-like surrogates for distributional outputs.
- Regularize: weight decay, dropout, and early stopping; operational datasets are large but non-stationary.
- Calibration layers: temperature distributions often need post-hoc calibration; integrate a learned scaling or isotonic step.
- Constraint handling: enforce physical bounds (e.g., plausible Tmin/Tmax ranges) via output transforms or penalties.
- Data augmentation is limited; but you can augment by adding noise to predictors within sensor uncertainty ranges.

## 11) Training, evaluation, and deployment (operational ML engineering)
- Split data by time, not random rows: use rolling-origin evaluation to mimic forecasting.
- Create ‘as-of’ datasets: freeze predictor availability exactly as it would have been at issuance time.
- Handle model upgrades: include change-point indicators, or re-train frequently, or use domain adaptation.
- Retraining cadence: daily/weekly for bias layers; monthly/seasonal for full neural backbones (depends on drift rate).
- Hyperparameters should be selected on recent validation windows to reflect current model configurations.
- Use baselines: persistence, climatology, NBM, and simple MOS; ML must beat these consistently.
- Evaluate by lead time and by product (hourly, Tmin/Tmax); a model can help day-0 but hurt day-5.
- Quantify compute/latency: can you run inference within the time budget between data arrival and forecast issuance?
- Design for missing inputs: robust models accept masks; operational data gaps happen (satellite outage, delayed model run).
- Use feature stores with versioning; store both raw gridded data and derived features (e.g., spatial averages).
- Make inference deterministic and reproducible (fixed preprocessing, fixed weights, explicit time zones).
- Log everything: model version, input checksums, and runtime; debugging operational issues requires audit trails.
- Provide fallbacks: if ML fails, output baseline NBM; but track the failure and alert.
- Post-processing: convert outputs to user products (point forecasts, grids, intervals, and threshold probabilities).
- Human-in-the-loop option: allow forecaster edits; log edits to learn systematic gaps or to train a ‘human correction’ model.

## 12) Failure modes, guardrails, and interpretability (how to avoid embarrassing ops failures)
- Data leakage: using future analyses/obs in predictors is the #1 silent failure in weather ML pipelines.
- Non-stationarity: NWP upgrades and new DA sources can shift predictor distributions overnight.
- Label drift: station moves or instrumentation changes create step shifts in ‘truth’.
- Regime sparsity: rare extremes are underrepresented; use weighted losses or oversampling for tail events.
- Overfitting to seasons: a model trained heavily on summer patterns can fail in winter transitions; use seasonal conditioning.
- Coastal timing errors: sea-breeze and marine stratus onset are hard; add targeted features and specialized evaluation slices.
- Under-calibrated uncertainty: quantile models can be too narrow; monitor coverage and recalibrate.
- Spatial mismatch: gridded models may not resolve microclimates; don’t promise street-level accuracy without dense sensors.
- Explainability tools: permutation importance (tabular), SHAP (careful with correlated features), and saliency maps (grids).
- Use case-based diagnostics: store top-N worst-error cases with context fields for rapid human review.
- Guardrails: enforce monotonic relationships where appropriate (e.g., warmer baseline tends to imply warmer corrected mean).
- Sanity checks: corrected temperatures should not jump unrealistically between adjacent hours without a frontal signal.
- Consistency checks: daily Tmax should equal max of hourly forecast series if both are produced by the model.
- Alerting: trigger alerts when bias spikes, data gaps occur, or uncertainty becomes implausibly small/large.

## 13) Glossary (operational + ML terms)
- **T2m**: air temperature at 2 meters above ground.
- **Tmin/Tmax**: daily minimum/maximum temperature (defined over a local-day window).
- **NWP**: Numerical Weather Prediction (physics-based dynamical forecasting).
- **DA**: Data Assimilation (combining observations with model to form analyses).
- **HRRR**: High-Resolution Rapid Refresh (convection-permitting short-range model).
- **RAP**: Rapid Refresh (hourly-updating mesoscale analysis/forecast model).
- **NAM**: North American Mesoscale model suite.
- **GFS**: Global Forecast System (global deterministic model).
- **GEFS**: Global Ensemble Forecast System (global ensemble).
- **ECMWF HRES/ENS**: ECMWF high-resolution deterministic / ensemble system.
- **NBM**: National Blend of Models (blended, bias-corrected guidance).
- **MOS**: Model Output Statistics (statistical post-processing of model guidance).
- **LAMP**: Localized Aviation MOS Program (short-range updates to MOS concepts).
- **RTMA**: Real-Time Mesoscale Analysis (near-real-time gridded analysis).
- **URMA**: Unrestricted Mesoscale Analysis (lagged but more complete analysis).
- **Pinball loss**: quantile regression loss function.
- **CRPS**: Continuous Ranked Probability Score (probabilistic forecast metric).
- **EMOS**: Ensemble Model Output Statistics (calibrating ensembles).
- **UHI**: Urban Heat Island.
- **Sea breeze**: coastal circulation bringing cooler marine air inland.
- **Lead time**: forecast horizon from initialization to valid time.
- **Valid time**: the time the forecast applies to.
- **Init time**: the model run start time.

## Appendix A) Minimal feature set (start here before deep nets)
- Baseline temperature guidance: NBM T2m (or HRRR T2m if focusing on <18h).
- Baseline dew point / humidity proxy (affects clouds and radiative cooling).
- 10 m wind speed and direction (mixing + advection).
- Cloud cover proxy (total cloud fraction or downward shortwave).
- Precipitation flag/proxy (evaporative cooling, cloud shading).
- Recent observed temperature trend (last 1–6 hours) for nowcasts.
- Station metadata: lat, lon, elevation, coastal distance, urban index.
- Time encodings: hour-of-day and day-of-year (cyclical).
- Model lead time encoding (continuous hours and/or bucket embedding).
- Ensemble spread feature (if available) to represent uncertainty.

## Appendix B) NYC-specific engineered features (high leverage)
- Onshore-flow indicator: dot product of wind with a coastline-normal vector.
- Upwind water path length: distance traveled over water along low-level wind direction.
- Coastal distance (km) and land/water fraction in a radius (e.g., 5–20 km).
- Sea-breeze likelihood score: (warm inland vs cool coast gradient) × (weak synoptic wind) × (daytime).
- Urban heat island index: impervious surface fraction or night-light proxy (static).
- Wind-direction bins (e.g., 16 sectors) as embeddings to capture directional regimes.
- Pressure tendency or geopotential height gradient proxy (front approach).
- Cloud-radiation mismatch features: difference between forecast shortwave and climatological clear-sky shortwave.
- Boundary-layer mixing proxy: wind speed × (1 - cloud fraction) for nocturnal coupling likelihood.
- Persistence-corrected residual: last observed (obs - baseline) error carried forward with decay.

## Appendix C) Target definitions (avoid ambiguity)
- Hourly T2m: forecast each hour’s temperature (best for products and for consistency checks).
- Daily Tmax: max over local-day window (specify 00–23 local or 06–06, etc.).
- Daily Tmin: min over local-day window (specify window).
- Daily mean: average over window; less sensitive to timing errors than Tmax/Tmin.
- Airport point targets: use METAR temperatures; specify station IDs and QC rules.
- Gridcell targets: use RTMA/URMA at the gridpoint nearest the location or bilinear interpolation.
- Area-mean targets: average T2m over a metro polygon; useful for ‘NYC average’ products.
- Probabilistic targets: quantiles of temperature distribution or exceedance probabilities.

## Appendix D) Common leakage traps checklist
- Using URMA values as predictors for a time before URMA is available operationally.
- Using analysis fields that are produced with future observations relative to forecast issuance time.
- Merging datasets by date only (dropping hour) and inadvertently aligning to the wrong valid time.
- Using station climatology computed with future years included (for strict backtests).
- Training on reforecast datasets but evaluating on operational data without matching processing.
- Accidentally including the observation target as an input via ‘latest obs’ fields when forecasting the same valid time.

## Appendix E) Model selection ladder (what to try in order)
- 1) Persistence + diurnal climatology baseline (fast sanity check).
- 2) NBM-only bias correction (linear / ridge) per station and lead.
- 3) Gradient boosting on baseline + meteorological drivers (tabular).
- 4) Quantile boosting / quantile regression (probabilistic).
- 5) Sequence model (1D CNN / Transformer) using recent obs + baseline errors.
- 6) Multi-model fusion network: ingest HRRR/RAP + GFS/GEFS + NBM.
- 7) Spatial CNN/U-Net for gridded correction (if delivering maps).
- 8) GNN for station network + attention to upwind grid context.
- 9) Hybrid: CNN encoder (grids) + temporal attention (time) + probabilistic head (quantiles).

## Appendix F) Evaluation slices (where models usually break)
- Clear, calm nights (Tmin errors via decoupling and UHI).
- Overcast nights with light wind (cloud LW forcing errors).
- Sunny days with weak gradient wind (sea-breeze onset drives Tmax).
- Post-frontal cold advection days (mixing depth and advection rate).
- Warm-sector humid days (cloud/convection timing affects Tmax).
- Snow-covered periods (albedo and flux changes).
- Rapid cyclogenesis / coastal storms (strong gradients, marine layer).
- Transition seasons (March–April, Oct–Nov) where climatology is least informative.

## Appendix G) Predictor variable catalog (NWP fields that matter for T2m)
- Use this as a menu; start with a small subset, then expand based on error attribution and ablations.

### Thermodynamics
- T2m / near-surface temperature: include because it is directly or indirectly coupled to surface temperature errors.
- T850 / T925 (lower-troposphere temperature): include because it is directly or indirectly coupled to surface temperature errors.
- Dew point / specific humidity at 2m: include because it is directly or indirectly coupled to surface temperature errors.
- Relative humidity at 2m: include because it is directly or indirectly coupled to surface temperature errors.
- Temperature advection proxy (wind · ∇T) from model: include because it is directly or indirectly coupled to surface temperature errors.
- Thickness or geopotential heights (e.g., 1000–500 hPa): include because it is directly or indirectly coupled to surface temperature errors.

### Kinematics
- 10 m wind speed: include because it is directly or indirectly coupled to surface temperature errors.
- 10 m wind direction: include because it is directly or indirectly coupled to surface temperature errors.
- U/V wind at 925/850 hPa: include because it is directly or indirectly coupled to surface temperature errors.
- Gusts (if available): include because it is directly or indirectly coupled to surface temperature errors.
- Vertical velocity proxy (omega) at 700 hPa: include because it is directly or indirectly coupled to surface temperature errors.
- Boundary-layer shear proxy (|V850 - V10|): include because it is directly or indirectly coupled to surface temperature errors.

### Boundary layer
- PBL height (HPBL): include because it is directly or indirectly coupled to surface temperature errors.
- Surface friction velocity / mixing parameter (if available): include because it is directly or indirectly coupled to surface temperature errors.
- Richardson number / stability index proxies: include because it is directly or indirectly coupled to surface temperature errors.
- Turbulent kinetic energy (TKE) (if available): include because it is directly or indirectly coupled to surface temperature errors.
- Surface layer stability (Monin-Obukhov length proxy): include because it is directly or indirectly coupled to surface temperature errors.

### Clouds & moisture
- Total cloud fraction: include because it is directly or indirectly coupled to surface temperature errors.
- Low/mid/high cloud fraction: include because it is directly or indirectly coupled to surface temperature errors.
- Cloud base height (aviation-relevant): include because it is directly or indirectly coupled to surface temperature errors.
- Integrated water vapor / precipitable water: include because it is directly or indirectly coupled to surface temperature errors.
- Radar reflectivity proxy (from HRRR) for convective shading/cooling: include because it is directly or indirectly coupled to surface temperature errors.

### Radiation
- Downward shortwave radiation at surface (SWDOWN): include because it is directly or indirectly coupled to surface temperature errors.
- Downward longwave radiation at surface (LWDOWN): include because it is directly or indirectly coupled to surface temperature errors.
- Net radiation at surface: include because it is directly or indirectly coupled to surface temperature errors.
- Clear-sky radiation (if available) or astronomical clear-sky computed feature: include because it is directly or indirectly coupled to surface temperature errors.

### Precipitation & convection
- Accumulated precipitation (APCP): include because it is directly or indirectly coupled to surface temperature errors.
- Convective precipitation (if separated): include because it is directly or indirectly coupled to surface temperature errors.
- Convective available potential energy (CAPE): include because it is directly or indirectly coupled to surface temperature errors.
- Convective inhibition (CIN): include because it is directly or indirectly coupled to surface temperature errors.
- Thunderstorm probability proxies (if available): include because it is directly or indirectly coupled to surface temperature errors.

### Land & surface
- Surface pressure (MSLP / SLP): include because it is directly or indirectly coupled to surface temperature errors.
- Soil moisture (top layers): include because it is directly or indirectly coupled to surface temperature errors.
- Soil temperature (top layers): include because it is directly or indirectly coupled to surface temperature errors.
- Snow depth / snow cover fraction: include because it is directly or indirectly coupled to surface temperature errors.
- Sea surface temperature (SST) or near-shore water temperature proxy: include because it is directly or indirectly coupled to surface temperature errors.
- Land-use category / roughness length (static): include because it is directly or indirectly coupled to surface temperature errors.
- Albedo (dynamic with snow): include because it is directly or indirectly coupled to surface temperature errors.
- Vegetation fraction / leaf area index (seasonal): include because it is directly or indirectly coupled to surface temperature errors.

### Large-scale pattern
- 500 hPa geopotential height: include because it is directly or indirectly coupled to surface temperature errors.
- 850 hPa temperature: include because it is directly or indirectly coupled to surface temperature errors.
- Sea-level pressure pattern (for fronts): include because it is directly or indirectly coupled to surface temperature errors.
- Vorticity at 500/700 hPa: include because it is directly or indirectly coupled to surface temperature errors.
- Jet stream proxy (wind at 250/300 hPa): include because it is directly or indirectly coupled to surface temperature errors.

### Ensemble descriptors
- Ensemble mean T2m: include because it is directly or indirectly coupled to surface temperature errors.
- Ensemble spread (std) T2m: include because it is directly or indirectly coupled to surface temperature errors.
- Ensemble percentiles (p10/p50/p90) for T2m: include because it is directly or indirectly coupled to surface temperature errors.
- Ensemble mean winds/clouds: include because it is directly or indirectly coupled to surface temperature errors.
- Ensemble spread of key drivers (clouds, winds): include because it is directly or indirectly coupled to surface temperature errors.

### Derived local features
- Temporal tendency: ΔT2m over last 1–3 forecast hours: include because it is directly or indirectly coupled to surface temperature errors.
- Diurnal anomaly: T2m minus hourly climatology: include because it is directly or indirectly coupled to surface temperature errors.
- Bias history: last k days of (obs - baseline) at same lead: include because it is directly or indirectly coupled to surface temperature errors.
- Upwind average of T2m over a wind-aligned corridor: include because it is directly or indirectly coupled to surface temperature errors.
- Land-water gradient across coastline normal direction: include because it is directly or indirectly coupled to surface temperature errors.

## Appendix H) Preprocessing and data engineering steps (operationally safe)
- Define a canonical time axis in UTC for storage; convert to local time only for target window definitions (Tmin/Tmax).
- Represent each forecast example by (init_time, valid_time, lead_time).
- Never infer lead_time from file names alone; compute it and store as an explicit numeric feature.
- Standardize units at ingest (K/C/F, m/s vs kt, Pa vs hPa) and keep unit metadata.
- Regrid all gridded predictors to a common grid when using CNNs; prefer conservative methods for fluxes.
- For point forecasts, interpolate gridded predictors to station points (bilinear) and store the interpolation method used.
- Store masks for missing channels; do not silently fill missing grids with zeros without a mask.
- Clip or winsorize physically implausible values caused by corrupt files; log anomalies.
- Compute derived features consistently in one library/module; avoid duplicated feature logic between training and inference.
- Normalize continuous predictors using robust scalers (median/IQR) computed on training windows; update periodically.
- Encode cyclical time with sin/cos transforms; avoid raw hour/day integers.
- Add embeddings for categorical sources (model id, station id) instead of one-hot when categories are many.
- Create separate datasets for each horizon band (nowcast vs short-range vs medium-range) if cadences differ.
- Balance datasets across seasons; if one season dominates, use sampling weights.
- Handle daylight saving time explicitly for local-day Tmax/Tmin windows; define windows in UTC to avoid ambiguity.
- Keep two label versions if needed: near-real-time (RTMA) for rapid evaluation and lagged (URMA) for high-quality training.
- Backfill late-arriving labels into the training store; support incremental training updates.
- Version datasets with immutable identifiers so experiments are reproducible.
- Implement strict schema validation (expected variables, dimensions, coordinate ranges) at ingest time.

### QC checklist for station observations (point labels)
- Range check: temperature within plausible limits for region/season (loose bounds).
- Step check: reject sudden jumps inconsistent with nearby stations unless corroborated by fronts.
- Temporal consistency: flag repeated identical readings over long periods (stuck sensor).
- Spatial buddy check: compare to neighboring stations weighted by distance and climatological correlation.
- Metadata check: station elevation and location consistent with known; detect station relocations.
- Time stamp integrity: ensure observation time is monotonic and in correct timezone/UTC.
- Reporting frequency: detect gaps and downweight recent-trend features when obs are missing.
- Outlier persistence: if flagged outlier persists, treat as possible regime change and inspect neighbors.

### QC checklist for gridded predictors (GRIB/NetCDF ingestion)
- File completeness: all expected forecast hours present for each init cycle.
- Coordinate consistency: lat/lon ordering, grid resolution, and projection metadata match expectations.
- NaN/Inf scan: detect corrupted fields; mark channel as missing rather than filling silently.
- Unit sanity: validate typical ranges (e.g., SWDOWN not negative in daytime, RH between 0–100).
- Physical bounds: enforce basic constraints (e.g., cloud fraction 0–1, snow cover 0–1).
- Temporal alignment: ensure valid_time computed correctly for each forecast hour.
- Model cycle changes: detect unexpected new lead ranges or missing hours (e.g., occasional HRRR extended products).

## Appendix I) Gridding, interpolation, and spatial context design
- Decide the spatial context window: too small misses upstream advection; too large wastes compute and adds noise.
- For NYC, include upstream land areas based on prevailing westerlies as well as coastal waters for marine regimes.
- Use multi-scale windows: small high-res crop (metro) + larger coarse crop (synoptic) to capture both scales efficiently.
- Coastlines create sharp gradients; CNNs benefit from high-res land/water masks as separate channels.
- If using station targets, consider ‘grid-to-point’ models: encode grids then sample at station coordinates via differentiable interpolation.
- Wind-aligned sampling: rotate or sample along the low-level wind direction to focus on upstream influence.
- Spatial pooling features: compute mean/max T2m, cloud, and wind over sectors (N/E/S/W) around station.
- Topography is mild but not zero; include elevation channels and lapse-rate-based adjustments where appropriate.
- Urban morphology: if available, add static urban fraction or impervious surface; otherwise learn station_id biases.
- Beware edge effects in convolutions: use padding strategies or limit predictions to interior regions.

## Appendix J) Uncertainty products to output (operationally useful)
- Deterministic point estimate (mean or median).
- Prediction interval (e.g., 10th–90th percentile) for each lead.
- Full quantile set (e.g., q05, q10, q25, q50, q75, q90, q95) for flexible products.
- Event probabilities: P(Tmax ≥ threshold), P(Tmin ≤ threshold), P(freeze), P(heat advisory).
- Uncertainty decomposition: aleatoric (weather chaos) vs epistemic (model uncertainty) when possible.
- Spread-skill diagnostics: compare predicted interval width vs realized absolute error.
- Reliability calibration tables by season and lead time.
- Spatial uncertainty maps (for gridded outputs): uncertainty larger near coastal gradients and fronts.
- ‘Scenario’ samples: draw synthetic temperature trajectories consistent with quantiles for downstream users (energy, transit).

## Appendix K) Deployment checklist (from data arrival to forecast product)
- Define the operational schedule: when do inputs arrive and when must outputs be published?
- Implement data fetchers for each source (HRRR/RAP hourly; GFS/GEFS 6-hourly; NBM as available; RTMA hourly; URMA lagged).
- Validate each fetch with checksums/size and expected variable inventory.
- Run preprocessing pipeline to generate model-ready tensors/features.
- Run inference with fixed model version; record runtime and memory usage.
- Apply post-processing: enforce bounds, compute intervals, compute threshold probabilities.
- Write outputs to both machine formats (JSON/NetCDF/GRIB-like) and human formats (tables/text).
- Publish to storage and/or API; include metadata: model_version, issue_time, and data_sources used.
- Run verification job when truth arrives; update dashboards and alerting.
- Trigger retraining jobs according to schedule; promote new models only after passing backtests.
- Keep a rollback mechanism: if new model underperforms, revert quickly.
- Log all exceptions; a missing satellite file should not crash the whole forecast.
- Document runbook for on-call: common failures, data outages, and manual override steps.

## Appendix L) Research directions that matter for temperature (practical, not hype)
- Neural post-processing of NBM/ensemble outputs with regime conditioning to improve calibration.
- Graph-based station networks with wind-conditioned edges to capture directional dependence.
- Multimodal fusion: combine satellite cloud fields with NWP drivers for better cloud-driven temperature corrections.
- Continual learning with drift detection to handle frequent NWP upgrades.
- Probabilistic sequence models that output coherent temperature trajectories (hour-to-hour smoothness with change-points).
- Physics-informed losses: penalize violations of energy-balance-inspired constraints (light-touch, avoid over-constraining).
- Extreme-aware training: tail-weighted losses for heat/cold extremes and threshold events.
- Interpretable regime clustering: learn latent weather regimes and attach human-readable labels for debugging.
- Multi-location transfer: train a backbone across many coastal-urban stations, then fine-tune to NYC to reduce overfitting.
- Reforecast-based training: use consistent historical model runs to reduce non-stationarity, then adapt to operations.

## Appendix M) Canonical data schema (recommended columns/keys for every record)
- issue_time_utc: time forecast product is issued (ops).
- init_time_utc: model initialization time for each predictor source.
- valid_time_utc: time the forecast applies to (target time).
- lead_hours: (valid_time - init_time) in hours (float).
- source_id: identifier for predictor source (HRRR, RAP, GFS, GEFS, NBM, etc.).
- variable_id: standardized variable name (e.g., t2m, u10, v10, t850, cldfrac).
- level: vertical level or surface (e.g., 2m, 10m, 850hPa).
- grid_id: spatial grid definition (projection/resolution/domain).
- lat, lon: coordinate for point features (stations) or grid indices for gridded tensors.
- station_id: station identifier if point target/predictors are station-based.
- station_meta_version: version of station metadata used.
- qc_flags: bitmask or structured flags for observation quality.
- availability_mask: per-variable mask indicating if the value was present at issuance.
- model_version: ML model version used for inference.
- training_window: date range used to train the deployed model.
- baseline_id: baseline used (e.g., NBM) if doing residual learning.
- baseline_value: baseline forecast value at the same valid_time and location.
- target_value: observed/analysis truth value (available later for training/verification).
- target_source: truth dataset used (METAR, RTMA, URMA).

## Appendix N) Weather regimes for stratified learning and evaluation (NYC)
- Clear-sky, calm (radiative cooling dominates; strong UHI at night).
- Clear-sky, windy (mixing dominates; smaller nocturnal cooling).
- Overcast, light wind (enhanced longwave; muted diurnal range).
- Overcast, windy (advection dominates; clouds reduce radiation).
- Onshore marine flow (cooler day, higher night temps; sea-breeze possible).
- Offshore flow (larger diurnal range; inland warming, cooler coast).
- Pre-frontal warm sector (humid; clouds/convection timing matters).
- Post-frontal cold advection (rapid cooling; mixing depth errors).
- Cold-air damming / shallow cold pool (persistence; scouring errors).
- Heat wave ridge (subsidence, clear skies; UHI strong; soil moisture matters).
- Arctic outbreak (very cold; wind chill; boundary-layer mixing critical).
- Coastal cyclone (strong gradients; marine layer; precipitation-driven cooling).
- Nor'easter with snow cover (albedo + cold air + strong winds).
- Warm advection over cold surface (fog/stratus; Tmax suppressed).
- Backdoor cold front (marine air from NE; timing critical in NYC).
- Sea-breeze day (weak synoptic wind; abrupt afternoon cooling near coast).
- Smoke/haze (radiation reduction; can lower Tmax).
- Persistent rain (evaporative cooling; low diurnal amplitude).
- Convective afternoon storms (outflow boundaries; sharp temp drops).

### Regime detection feature hints (turn raw inputs into regime tags)
- Clear vs cloudy: use cloud fraction + SWDOWN relative to clear-sky estimate.
- Calm vs windy: use 10 m wind speed thresholds and gustiness.
- Onshore vs offshore: use wind direction sectors and coastal-normal dot product.
- Frontal proximity: use SLP gradients, temperature gradients, and wind shifts.
- Precip regime: use accumulated precip and radar reflectivity proxies.
- Heat/cold extremes: use anomalies vs climatology at 850 hPa and surface.
- Snow regime: use snow cover fraction and surface albedo.
- Backdoor front: detect NE low-level winds with coastal cooling signature.

## Appendix O) Training recipes (step-by-step playbooks)
### O1) Residual MOS-style model (tabular, fast, strong baseline)
- Inputs: baseline forecast (NBM or HRRR), plus key drivers (wind, cloud, humidity), time encodings, station metadata.
- Target: observed temperature (or analysis) at valid_time.
- Label transform: residual = obs - baseline.
- Model: ridge regression, gradient boosting, or small MLP.
- Loss: MAE on residual; optionally Huber to balance outliers.
- Train separate models by lead bucket (e.g., 0–6h, 6–18h, 18–36h, 36–48h) for stability.
- Add regime features or train a mixture-of-experts by regime.
- Post-process: corrected = baseline + predicted_residual; clip to plausible bounds.
- Evaluate skill vs baseline and bias by regime and season.

### O2) Quantile residual model (probabilistic intervals on top of baseline)
- Same as O1 but predict multiple quantiles of residual (q10/q50/q90).
- Loss: pinball loss summed over quantiles; optionally weight tails for extremes.
- Ensure quantile monotonicity: sort outputs or use monotone quantile layers.
- Compute corrected quantiles: baseline + residual_quantile.
- Calibrate: check empirical coverage; apply post-hoc calibration if needed.

### O3) Sequence model for near-term updates (LAMP-like)
- Inputs: recent observed temperatures (last 6–24h), recent baseline errors, and current NWP guidance trajectory.
- Architecture: 1D CNN or Transformer over time, with static embeddings (station, month).
- Target: correction trajectory for the next N hours (multi-step).
- Training: teacher forcing for multi-step; use scheduled sampling if needed.
- Loss: MAE per hour + smoothness penalty to avoid jitter, except allow change-points.
- Mask missing observations; do not impute aggressively.

### O4) Grid-based downscaling with CNN/U-Net (map outputs)
- Inputs: multi-channel gridded predictors (NBM, HRRR, winds, clouds, radiation, land/water mask).
- Target: RTMA/URMA temperature grid at high resolution.
- Architecture: U-Net with skip connections; optionally multi-scale encoder with synoptic context.
- Loss: MAE + gradient loss to preserve fronts/coastal gradients; consider SSIM-like term cautiously.
- Evaluation: spatial MAE and structure (front position) + point extraction at key stations.
- Operational note: large grids are expensive; crop to domain of interest or use tiling.

### O5) Station-network model with GNN (spatial correlations)
- Nodes: stations; node features include baseline forecasts + local drivers + metadata.
- Edges: distance-based and/or wind-direction-conditioned; include edge attributes (bearing, distance).
- Architecture: message passing network + temporal encoder if using sequences.
- Target: station temperatures or residuals; optionally multi-task across stations.
- Benefits: learns coherent spatial patterns and can borrow strength from nearby stations.

### O6) Universal training tips (apply to all weather ML models)
- Use time-based cross-validation: train on years A–B, validate on later months, test on most recent season.
- Include a ‘recentness’ weighting: weight recent examples more to adapt to drifting NWP systems.
- Standardize by season: either train per season or use seasonal embeddings; reduces distribution shift.
- Normalize per lead time: some predictors have lead-dependent biases; consider lead-wise normalization.
- Use early stopping on a rolling validation window that matches deployment recency.
- Keep an experiment registry: config, dataset hash, commit id, and results.
- Prefer simpler models if they match skill: operational robustness beats marginal offline gains.
- Check that improvements persist across multiple months; avoid one-off wins from favorable regimes.
- Perform sensitivity tests: remove a major input (e.g., GOES) and ensure the model degrades gracefully.
- Test cold-start scenarios: what happens when a new station is added or a station goes offline?

## Appendix P) Monitoring dashboard metrics (what to plot daily/weekly)
### Deterministic metrics (per station, per lead, per season)
- MAE of temperature (overall).
- RMSE of temperature (overall).
- Mean bias (forecast - truth).
- Median bias (robust).
- 95th percentile absolute error (tail risk).
- Max absolute error (worst case).
- Skill vs baseline MAE (NBM).
- Skill vs persistence MAE (nowcast).
- Error autocorrelation (indicates systematic drift).
- Diurnal error cycle (bias by hour).
- Seasonal error cycle (bias by month).
- Regime-conditional MAE (clear/calm, onshore, post-frontal, etc.).
- Spatial error map (for gridded outputs).

### Probabilistic metrics (for quantiles/distributions)
- CRPS (overall and by lead).
- Pinball loss per quantile.
- Reliability: empirical coverage of 50%, 80%, 90% intervals.
- Sharpness: average interval width by lead and season.
- Probability integral transform (PIT) histogram for distribution forecasts.
- Rank histograms for ensemble-based outputs.
- Brier score for key thresholds (freeze, 90F, 95F).
- Reliability diagrams for threshold events.
- ROC-AUC / PR-AUC for rare event thresholds (useful but not sufficient).
- Spread-skill: interval width vs realized absolute error.

### Operational health metrics (pipeline, data, runtime)
- Data latency per source (minutes from expected availability).
- Missing channel rate per source and cycle.
- File corruption rate / QC failure counts.
- Inference runtime distribution (p50/p95).
- Memory usage and GPU utilization (if used).
- Forecast publication success rate and timestamps.
- Fallback rate (how often baseline used due to ML failure).
- Model version distribution (ensure consistent deployment).

### Alert rules (examples)
- Bias magnitude exceeds threshold for N consecutive runs (e.g., |bias| > 2F for 24h).
- Coverage drops below target (e.g., 80% interval covers <70% of cases over last 7 days).
- MAE regression vs baseline exceeds threshold (e.g., ML worse than NBM by >0.5F for 3 days).
- Data source missing for >X cycles (e.g., HRRR missing 2 consecutive hours).
- Inference runtime exceeds SLA (e.g., p95 runtime > 2 minutes).

### Lead-time tracking grid (template: compute each metric for each lead bucket)
- Track **MAE** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **0–1h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **1–3h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **3–6h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **6–12h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **12–18h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **18–24h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **24–36h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **bias** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **36–48h** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **bias** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **2–3d** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **bias** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **3–5d** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **bias** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **5–7d** (overall + by season + by key regimes).
- Track **MAE** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **bias** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **RMSE** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **q10-90 width** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **80% coverage** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **freeze Brier** for lead bucket **7–10d** (overall + by season + by key regimes).
- Track **90F Brier** for lead bucket **7–10d** (overall + by season + by key regimes).

## Appendix Q) LLM prompting patterns for weather ML (so an LLM can act as an engineer)
- Use these as reusable instruction blocks when asking an LLM to build parts of the pipeline.

- Ask for an ‘as-of’ data join plan: explicitly list which datasets are available at issue_time and which are lagged.
- Ask for a strict schema: insist the LLM defines init_time, valid_time, lead_time for every record and forbids date-only merges.
- Ask for baseline-first modeling: require NBM residual learning before any deep net proposal.
- Ask for a regime slice analysis: require evaluation splits for clear/calm, onshore flow, and post-frontal regimes.
- Ask for leakage checks: require the LLM to list potential leakage vectors and how to test for each.
- Ask for a retraining plan: specify what is retrained daily (bias) vs monthly (backbone) and what triggers rollbacks.
- Ask for probabilistic outputs: require quantiles and calibration verification, not just point forecasts.
- Ask for operational constraints: require runtime budgets and missing-data behavior.
- Ask for ablation tables: require a table showing incremental skill from adding HRRR, then GEFS, then GOES, etc.
- Ask for failure-case logging: require a design that stores worst-case forecast contexts for inspection.

### Example ‘system prompt’ fragments (single-line templates)
- ‘Design a temperature-forecast ML model that corrects NBM using only information available by <ISSUE_TIME>, and output q10/q50/q90.’
- ‘Produce a data schema with explicit init_time/valid_time/lead_time for HRRR, RAP, GFS, GEFS, NBM, RTMA, URMA.’
- ‘Write a backtest plan using rolling-origin evaluation from 2018–2026 with no leakage from URMA availability.’
- ‘List top 20 features for NYC Tmax errors on sea-breeze days and how to compute them.’
- ‘Propose monitoring alerts for MAE, bias, and coverage regressions and suggest rollback thresholds.’

## Appendix R) Variable naming conventions (avoid ambiguity across sources)
- Standardize variable names across HRRR/RAP/GFS/NBM so your feature store is coherent.

- **t2m** = 2 m temperature (canonical unit: K).
- **d2m** = 2 m dewpoint (canonical unit: K).
- **rh2m** = 2 m relative humidity (canonical unit: %).
- **u10** = 10 m u-wind (canonical unit: m/s).
- **v10** = 10 m v-wind (canonical unit: m/s).
- **wind10** = 10 m wind speed (canonical unit: m/s).
- **wdir10** = 10 m wind direction (canonical unit: deg).
- **mslp** = mean sea-level pressure (canonical unit: Pa).
- **t850** = 850 hPa temperature (canonical unit: K).
- **u850** = 850 hPa u-wind (canonical unit: m/s).
- **v850** = 850 hPa v-wind (canonical unit: m/s).
- **z500** = 500 hPa geopotential height (canonical unit: m^2/s^2 or m).
- **cldfrac** = total cloud fraction (canonical unit: 0-1).
- **lcld** = low cloud fraction (canonical unit: 0-1).
- **mcld** = mid cloud fraction (canonical unit: 0-1).
- **hcld** = high cloud fraction (canonical unit: 0-1).
- **swdown** = surface downward shortwave (canonical unit: W/m^2).
- **lwdown** = surface downward longwave (canonical unit: W/m^2).
- **pblh** = boundary-layer height (canonical unit: m).
- **apcp** = accumulated precipitation (canonical unit: mm).
- **snowc** = snow cover fraction (canonical unit: 0-1).
- **snod** = snow depth (canonical unit: m).
- **sst** = sea surface temperature (canonical unit: K).

## Appendix S) Ablation study plan (prove what adds skill)
- Start with baseline-only (NBM) model and measure MAE/bias by lead and season.
- Add station metadata and time encodings; measure incremental skill.
- Add wind features; measure skill on nocturnal Tmin and onshore regimes.
- Add cloud/radiation features; measure skill on sunny vs cloudy slices.
- Add HRRR predictors (if baseline is NBM) for day-0; measure nowcast improvement.
- Add RAP predictors for broader context; measure robustness during fronts.
- Add GFS/GEFS predictors for day-2+; measure medium-range skill and interval quality.
- Add GOES cloud features; measure improvement on cloud-driven Tmax errors.
- Add snow/land-surface state features; measure winter performance changes.
- Switch from deterministic to quantile outputs; measure CRPS and coverage calibration.
- Try sequence model vs static model; measure improvement in rapidly changing regimes.
- Try spatial CNN vs point model; measure map-quality metrics and station extraction skill.
- Report all deltas vs baseline with confidence intervals (block bootstrap over days).

- Ablation row 01: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 02: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 03: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 04: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 05: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 06: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 07: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 08: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 09: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 10: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 11: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 12: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).
- Ablation row 13: record (features_added, train_window, test_window, MAE, bias, CRPS, coverage, notes).

## Appendix T) Error attribution playbook (turn big misses into actionable fixes)
- When error is large, first classify by regime: clear/cloudy, calm/windy, onshore/offshore, frontal/non-frontal.
- Check baseline vs corrected: did ML make it better or worse? If worse, suspect regime misclassification or overfitting.
- Inspect cloud fields: if Tmax is too high under unexpected cloudiness, focus on cloud/radiation predictors.
- Inspect wind: if Tmin is too warm under calm clear nights, model likely over-mixed; add stability/mixing proxies.
- Inspect sea-breeze timing: if coastal stations cooled early/late, add wind-shift and coastal-gradient features.
- Inspect frontal timing: if temperature jumps occur at wrong hour, add synoptic gradient and tendency features.
- Inspect snow cover: if model too warm in snow, ensure snowc/snod features are present and correctly aligned.
- Inspect station anomalies: compare to neighbors to detect sensor issues or representativeness quirks.
- Check input availability: missing HRRR cycle or delayed data can silently degrade forecasts if not handled.
- Check drift: sudden bias across many stations may coincide with an upstream NWP model upgrade.
- Maintain a library of ‘canonical failures’ (sea-breeze day, backdoor front, cold pool) for regression testing.

- Case card fields: issue_time, station, lead, truth, baseline, ml, error, regime_tag, key_predictors_snapshot, notes.
- Plot set: (truth vs baseline vs ml), (cloud fraction), (wind), (radiation), (pressure), (ensemble spread).
- Decision: add feature, adjust training weights, recalibrate uncertainty, or add guardrail.

## Appendix U) Dataset quick reference (what each source is good for)
- **METAR**: Point observations at airports; high-frequency truth and nowcast anchoring.
- **ISD**: Historical station archive for long backtests and climatology.
- **RTMA**: Near-real-time gridded analysis for hourly temperature labels and verification.
- **URMA**: Lagged, higher-quality gridded analysis for training labels and post-event verification.
- **HRRR**: High-res short-range model; strong for hour-to-hour temperature evolution and mesoscale features.
- **RAP**: Hourly mesoscale model/analysis; good for broader short-range context and stability proxies.
- **NAM**: Mesoscale guidance; useful as additional model diversity and alternative physics.
- **GFS**: Global deterministic; best for synoptic setups and longer leads.
- **GEFS**: Global ensemble; uncertainty features and probabilistic context.
- **ECMWF ENS/HRES**: High-performing global guidance; valuable when accessible.
- **NBM**: Operational blend; excellent baseline and calibrated guidance source.
- **GOES-East**: Satellite cloud and radiation context; helpful when clouds drive temperature errors.

- METAR storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- METAR storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- METAR storage tip: Store provenance: download timestamps and checksum hashes.
- ISD storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- ISD storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- ISD storage tip: Store provenance: download timestamps and checksum hashes.
- RTMA storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- RTMA storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- RTMA storage tip: Store provenance: download timestamps and checksum hashes.
- URMA storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- URMA storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- URMA storage tip: Store provenance: download timestamps and checksum hashes.
- HRRR storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- HRRR storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- HRRR storage tip: Store provenance: download timestamps and checksum hashes.
- RAP storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- RAP storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- RAP storage tip: Store provenance: download timestamps and checksum hashes.
- NAM storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- NAM storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- NAM storage tip: Store provenance: download timestamps and checksum hashes.
- GFS storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- GFS storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- GFS storage tip: Store provenance: download timestamps and checksum hashes.
- GEFS storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- GEFS storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- GEFS storage tip: Store provenance: download timestamps and checksum hashes.
- ECMWF ENS/HRES storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- ECMWF ENS/HRES storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- ECMWF ENS/HRES storage tip: Store provenance: download timestamps and checksum hashes.
- NBM storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- NBM storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- NBM storage tip: Store provenance: download timestamps and checksum hashes.
- GOES-East storage tip: Store raw files (GRIB2/NetCDF) with immutable naming, plus parsed arrays/tensors for model ingestion.
- GOES-East storage tip: Store metadata: run times, forecast hours, grid definitions, and variable inventories.
- GOES-East storage tip: Store provenance: download timestamps and checksum hashes.

## Appendix V) End-to-end pipeline blueprint (pseudocode as bullets, no code)
### Ingestion module
- Input: source_id, init_time, expected_variables, domain bbox.
- Fetch latest files; if missing, retry with backoff; if still missing, mark channel unavailable.
- Validate file integrity and variable inventory; emit QC report.
- Parse to arrays; store raw and parsed artifacts with immutable keys.

### Preprocessing module
- Align all predictors to canonical grid and time axis (valid_time).
- Interpolate to station points if producing point forecasts.
- Compute derived features (tendencies, anomalies, regime flags).
- Attach static metadata features (station, land/water, urban).
- Build masks for missing variables and missing obs history.
- Write model-ready batch records keyed by (issue_time, station/gridcell, lead).

### Training module
- Construct time-based splits (train/val/test) with rolling-origin evaluation.
- Train baseline residual model first; log metrics and diagnostics.
- Train probabilistic extension (quantiles) and evaluate calibration.
- Optionally train sequence/spatial models; compare skill vs simpler models.
- Select best model by multi-metric score (MAE + CRPS + coverage + robustness).
- Package model artifacts and preprocessing config as a single versioned bundle.

### Inference module
- On each issue_time, assemble latest available predictors and masks.
- Run model inference; if failure, fall back to baseline and alert.
- Apply post-processing constraints and produce forecast products.
- Publish outputs and metadata; store for later verification.

### Verification module
- When truth arrives (METAR/RTMA/URMA), compute metrics by lead/regime/station.
- Update dashboards and alerting; store case cards for worst errors.
- Feed recent errors into bias-updater (Kalman-like) if used.

### Reproducibility checklist (make future-you happy)
- Pin dataset versions and store a manifest of every file used for training.
- Pin preprocessing code version (git commit) inside the model bundle.
- Save feature normalization parameters and any categorical vocabularies/embeddings mappings.
- Record random seeds and deterministic compute settings.
- Store training config (hyperparameters, loss, early stopping) in a machine-readable file.
- Store evaluation outputs (predictions + truth) for audit and for recalibration experiments.
- Keep a changelog of model deployments with dates and reasons.
- Run regression tests on canonical failure cases before each deployment.

### Suggested unit/integration tests (minimum set)
- Test that valid_time = init_time + lead_hours for every record.
- Test that no predictor has timestamp later than issue_time (leakage guard).
- Test that unit conversions are correct (K↔C↔F) on known values.
- Test that missing-file behavior sets masks and does not crash inference.
- Test that quantiles are monotone (q10 ≤ q50 ≤ q90).
- Test that Tmin ≤ Tmax for daily outputs.
- Test that outputs stay within physical bounds for season/location (loose bounds).
- Test that model bundle includes preprocessing config and can run standalone.
- Integration test: run a full ‘yesterday’ issue cycle end-to-end and compare to stored golden outputs.

## Appendix W) ML term glossary (weather-forecasting specific usage)
- **Residual learning**: predicting (truth - baseline) instead of truth directly.
- **Rolling-origin evaluation**: evaluating forecasts on a moving time window to mimic real operations.
- **Pinball loss**: loss for quantile forecasts; asymmetric linear penalty.
- **CRPS**: integrates quantile loss over all thresholds; lower is better.
- **Calibration**: agreement between predicted probabilities/intervals and observed frequencies.
- **Sharpness**: concentration of the predictive distribution; should increase only when reliable.
- **Ensembling**: averaging multiple models to reduce variance and improve robustness.
- **Epistemic uncertainty**: uncertainty due to model ignorance; reducible with data/model improvements.
- **Aleatoric uncertainty**: intrinsic weather variability; irreducible noise.
- **Teacher forcing**: training sequence models using true previous steps as inputs.
- **Scheduled sampling**: gradually replacing teacher forcing with model-generated inputs.
- **Masking**: indicating missing inputs explicitly so the model can ignore them.
- **Domain adaptation**: adjusting a model trained on one distribution (reforecast) to another (operations).
- **Concept drift**: change in relationship between predictors and target over time (e.g., NWP upgrade).
- **Covariate shift**: change in predictor distribution over time without changing conditional relationship.
- **Mixture of experts**: multiple sub-models specialized to regimes, combined by a gating network.
- **Monotone quantiles**: enforcing ordered quantile outputs to avoid crossings.
- **Isotonic regression**: nonparametric calibration mapping to fix reliability issues.
- **Temperature trajectory coherence**: ensuring hourly forecasts are smooth except at physical change-points.
- **Change-point**: time when regime shifts abruptly (front, sea breeze).
- **Feature store**: centralized system to compute, store, and serve features consistently.
- **Model registry**: system to store model bundles, metadata, and deployment status.
- **Fallback**: operational baseline used when ML cannot run or fails QC.
- **Backfill**: adding late-arriving truth data to the training store after the fact.
- **Reforecast**: historical model runs with fixed model version for consistent training data.

- **Bias correction**: Adjusting systematic errors of a forecast model.
- **Downscaling**: Mapping coarse predictions to local scales.
- **Post-processing**: Any statistical/ML adjustment applied after NWP.
- **Verification**: Comparing forecasts to truth to measure skill.
- **Skill score**: Performance relative to a baseline.
- **Reliability diagram**: Plot comparing forecast probability to observed frequency.
- **PIT histogram**: Uniformity diagnostic for probabilistic forecasts.
- **Rank histogram**: Ensemble calibration diagnostic.
- **EMOS**: Statistical calibration of ensemble mean/spread.
- **MOS**: Statistical mapping from model output to observed truth.

## Appendix X) Quick-start checklist (27 steps from zero to operational baseline)
- Pick target: station-based hourly T or daily Tmax/Tmin for JFK/LGA/EWR.
- Pick truth source: METAR for point; RTMA/URMA for gridded.
- Pick baseline: NBM for most leads; HRRR for very short-range.
- Define issue times you will emulate (e.g., 00Z, 06Z, 12Z, 18Z).
- Build an ‘as-of’ join: only predictors available by issue time.
- Ingest baseline forecasts and store init/valid/lead metadata.
- Ingest key drivers: wind, cloud, humidity, radiation (start small).
- Ingest recent observations for nowcast features (if allowed at issue time).
- Attach station metadata and time encodings (hour/day-of-year).
- Compute residual labels (truth - baseline).
- Split data by time; reserve the most recent season as test.
- Train ridge regression on residuals; evaluate MAE and bias by lead.
- Add gradient boosting; compare skill vs ridge.
- Add quantile regression (q10/q50/q90); check coverage.
- Set up daily verification job when truth arrives.
- Create dashboard: MAE, bias, coverage, and skill vs baseline.
- Add drift alerts (bias spikes, MAE regressions, coverage drops).
- Add regime slices: clear/calm and onshore flow at minimum.
- Log worst-error case cards for inspection.
- Only then consider deeper nets (sequence or spatial).
- If using sequences, add masking for missing obs history.
- If using grids, add land/water mask channel and multi-scale crops.
- Re-run ablations: confirm each new input adds consistent skill.
- Package model + preprocessing as a versioned bundle.
- Deploy with fallback to baseline and full logging.
- Schedule periodic recalibration/retraining (bias fast, backbone slower).
- Document the runbook and regression-test canonical failure cases.