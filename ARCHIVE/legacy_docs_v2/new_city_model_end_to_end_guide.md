# End-to-End Guide: Building a New U.S. City Weather Market Model (NYC/CHI/PHL Template)

**Audience:** LLM agents and engineers implementing a full city pipeline (data → probabilistic forecast → calibration → bucket probabilities → EV/backtest/trading gating).  
**Scope:** Practical implementation playbook based on this repository’s NYC/CHI/PHL architecture.

---

## 0) What “done” means

A new city implementation is complete only when all of the following exist and pass checks:

1. **Contract-aligned target spec** (station, timezone/day boundary, bucket logic, inclusivity).  
2. **City config + station network config** wired into ingestion, preprocessing, benchmark, calibration, and backtest scripts.  
3. **Chronological train/val/test datasets** built from city raw data with time-safe feature construction.  
4. **Probabilistic models + baselines** that output distributions (mu/sigma or bucket probabilities).  
5. **Post-hoc calibration layer** evaluated with reliability diagnostics.  
6. **Bucketization + contract-level scoring** against observed outcomes and (when available) Kalshi pre-settlement rows.  
7. **Cost-aware EV/backtest paper-trading gate** with risk limits and halt logic.

---

## 1) System blueprint in this repo (copy this shape)

Use this 5-layer split and keep interfaces explicit:

1. **Ingestion**  
   - Historical daily station data (`src/data_collection.py`)  
   - Optional operational feeds (ASOS/NWP/soundings) (`src/asos_collection.py`, `src/nwp_collection.py`, `src/soundings_collection.py`)
2. **Preprocessing + features**  
   - Merge station files + completeness filtering + chronological splits (`src/data_preprocessing.py`)  
   - Time-safe feature engineering (`src/operational_features.py`)
3. **Modeling**  
   - Baselines (`src/baselines.py`)  
   - Flat heteroscedastic NN (`src/model.py`)  
   - WGA / advanced model families (`src/wind_gated_attention.py`, `src/advanced_model.py`)
4. **Calibration + evaluation**  
   - Isotonic/Platt/regime-aware calibration (`src/calibration.py`)  
   - Brier/CRPS/reliability diagnostics (`src/evaluate.py`, `src/contract_brier.py`)
5. **Trading/backtest**  
   - EV/sizing/risk controls (`src/trading.py`)  
   - Backtesting and Kalshi integration (`src/kalshi_backtester.py`, `src/kalshi_client.py`)

---

## 2) First-principles requirements before any coding

### 2.1 Contract alignment checklist (must be explicit)

Before creating any city file, capture:

- Kalshi ticker and contract family naming pattern.
- Official settlement station identifier (GHCN + ICAO/NWS mapping).
- Local-time day boundary used for settlement (including DST handling).
- Bucket definitions currently listed, including open tails and boundary inclusivity/exclusivity.
- Rounding/measurement conventions used by settlement.

**Implementation rule:** the target variable in preprocessing must represent exactly what the contract settles on.

### 2.2 Operational cutoff and source taxonomy

For every feature, label it:

- **Operational** (available by inference cutoff) or
- **Training-only** (never used directly in live inference).

If a feature is training-only, do not include it in live-time model inputs.

---

## 3) Files you must create or modify for a new city

## 3.1 City registry (required)

Edit `src/city_config.py`:

1. Add new `CityConfig` instance (`_XXX_CONFIG`) with:
   - `city_code`, `city_name`, `kalshi_ticker`
   - target station + lat/lon + timezone
   - IGRA station + NWP anchor
   - bucket edges + labels
   - monthly climatology mean/std
   - city-specific `data_dir`, `models_dir`, `results_dir`
2. Register in `_CITY_REGISTRY`.
3. Ensure bucket conventions are represented with sentinel tails (`-999`, `999`) and `get_bucket_index` logic remains valid.

## 3.2 City station network config (required)

Create `config_<city>.py` following CHI/PHL pattern:

- `CITY_CONFIG = get_city_config("<code>")`
- `TARGET_STATION`, `TARGET_LAT`, `TARGET_LON`, `TARGET_VARIABLE`
- `SURROUNDING_STATIONS` grouped by distance ring and direction
- `ALL_STATIONS` including target + surrounding
- `ASOS_STATION_MAP`, `NON_ASOS_STATIONS`
- `MIN_COMPLETENESS`, `STATION_RINGS`, `STATION_SECTORS`
- split parameters (`TRAIN_RATIO`, `VAL_RATIO`) and date windows

## 3.3 City runners (recommended minimum)

Clone CHI/PHL script family with the new city code:

1. `scripts/run_<city>_data_collection.py`  
2. `scripts/run_<city>_preprocessing.py`  
3. `scripts/run_<city>_benchmark.py`  
4. `scripts/run_<city>_synthesis_calibration.py`  
5. `scripts/run_<city>_backtest.py`  
6. `scripts/run_<city>_promotion_evaluation.py`

Also update multi-city runners when needed:

- `scripts/run_chi_phl_unified_benchmark.py` (or a generalized replacement)
- `scripts/fetch_kalshi_multi_city.py`
- `scripts/run_city_nws_kalshi_template_benchmark.py` (`--city` choices and MOS path map)

---

## 4) End-to-end implementation flow (copy/paste runbook)

## Phase A — Data setup

1. Build station list (target + 40-55 nearby stations) using climate/terrain reasoning.
2. Download/parse station histories into `data/<city>/raw/*.csv`.
3. Run preprocessing to produce:
   - `features_train.csv`, `features_val.csv`, `features_test.csv`
   - `target_train.csv`, `target_val.csv`, `target_test.csv`
   - scaler and metadata artifacts.

**Expected behavior from existing pipeline templates:**
- station completeness filtering,
- lagged feature construction (to avoid same-day leakage),
- cyclical day-of-year features,
- train-only scaler fitting,
- chronological split (no random shuffle).

## Phase B — Baselines and probabilistic modeling

1. Fit baseline families first:
   - persistence,
   - climatology,
   - ridge regression.
2. Fit at least one distributional model (mu/sigma output):
   - flat NN and/or advanced/WGA variant.
3. Convert model distributions to bucket probabilities via CDF mass differences.
4. Score with Brier/CRPS and keep a model only if it beats simpler baselines OOS.

## Phase C — Calibration and reliability

1. Reserve non-training calibration slice (validation or dedicated split).
2. Apply isotonic or Platt+isotonic calibration to bucket/CDF outputs.
3. Validate reliability:
   - PIT behavior,
   - calibration curve slope/intercept,
   - interval coverage by season/regime.

## Phase D — Contract benchmark and EV gating

1. Load Kalshi **pre-settlement** rows for the city when available.
2. Compare model probabilities vs market-implied probabilities.
3. Compute contract-level Brier and EV after costs (fees + conservative slippage).
4. Run backtest with risk caps and kill-switch logic; do not auto-promote on Brier alone.

## Phase E — Promotion decision

Promote only if all hold:

- OOS reliability is stable,
- contract-level Brier beats baselines and market proxy benchmarks,
- EV remains positive under conservative cost assumptions,
- drawdown and exposure limits remain within policy.

---

## 5) Canonical command sequence (new city template)

> Replace `<city>` and script names with your actual files.

```bash
# 1) Data collection + preprocessing
python scripts/run_<city>_data_collection.py
python scripts/run_<city>_preprocessing.py

# 2) Baseline/probabilistic benchmark
python scripts/run_<city>_benchmark.py
python scripts/run_city_nws_kalshi_template_benchmark.py --city <city>

# 3) Calibration/synthesis
python scripts/run_<city>_synthesis_calibration.py

# 4) Backtest + promotion gate
python scripts/run_<city>_backtest.py
python scripts/run_<city>_promotion_evaluation.py
```

---

## 6) Data contract and artifact conventions

Keep these conventions so multi-city tooling works:

- Raw station files: `data/<city>/raw/<station_id>.csv`
- Processed split files: `data/<city>/processed/features_{train,val,test}.csv`
- Targets: `data/<city>/processed/target_{train,val,test}.csv`
- MOS (if used): `data/<city>/mos/combined_mos_<icao>.csv`
- Models: `models/<city>/...`
- Results: `results/<city>/...`

Every run should emit:

- benchmark summary CSV/JSON,
- calibration diagnostics,
- backtest metrics + PnL curve,
- explicit status metadata when inputs are missing (never silently fabricate data).

---

## 7) Modeling guidance distilled from NYC/CHI/PHL

1. **Earn complexity**: keep ridge/persistence/climatology as permanent baselines.
2. **Prefer calibrated probabilities over MAE wins** for trading decisions.
3. **Use city physics** in feature design:
   - coastal moderation, lake effects, upslope/downslope regimes, urban heat effects.
4. **Handle missingness explicitly** (masking/ffill strategy) and preserve train/live parity.
5. **Check seasonal slices**; average metrics can hide winter/summer failure modes.
6. **Treat sigma modeling as first-class** (monthly or regime-conditioned uncertainty).

---

## 8) Common failure modes + safeguards

1. **Bucket spec drift** between docs/code/market listing  
   - Safeguard: nightly contract metadata verification and alert.
2. **Train/inference mismatch** (using delayed data in offline but not live)  
   - Safeguard: feature provenance tags + live-availability tests.
3. **Probability miscalibration** in tail buckets  
   - Safeguard: per-season reliability checks and recalibration cadence.
4. **Kalshi data gaps** (missing city rows)  
   - Safeguard: emit benchmark status as unavailable; block trading deployment.
5. **Overexposure across adjacent buckets**  
   - Safeguard: portfolio-level correlated exposure caps + kill switch.

---

## 9) Minimum test plan for any new city PR

1. **Unit-level checks**
   - bucket index logic and boundary inclusivity,
   - preprocessing leakage tests (lag correctness),
   - probability normalization (sums to 1).
2. **Integration check**
   - run data_collection → preprocessing → benchmark end to end.
3. **Backtest simulation check**
   - run one historical period with fees/slippage enabled.
4. **Sanity metrics gate**
   - no catastrophic reliability failure (e.g., strong PIT skew, interval undercoverage).

---

## 10) LLM implementation prompt skeleton (for future automation)

Use this structure when asking an LLM to spin up a city:

1. “Implement `<city>` in `src/city_config.py` with contract-aligned station/timezone/buckets.”
2. “Create `config_<city>.py` with station rings/sectors, ASOS mapping, and completeness threshold.”
3. “Create city scripts by adapting CHI/PHL runners for collection, preprocessing, benchmark, calibration, and backtest.”
4. “Ensure chronological splits, no leakage, and train-only scaler fitting.”
5. “Run baseline + probabilistic benchmark and produce `results/<city>/` artifacts.”
6. “Run calibration diagnostics and backtest with cost assumptions.”
7. “Return a model card + data dictionary + trading gate decision with explicit risks.”

---

## 11) Final acceptance checklist (ship/no-ship)

- [ ] Contract definition and settlement station verified against current market listing.  
- [ ] City config added and discoverable via `get_city_config` / `list_cities`.  
- [ ] Data/preprocessing scripts run successfully and produce expected files.  
- [ ] Baselines + distributional model benchmarked on chronological holdout.  
- [ ] Calibration layer applied and reliability diagnostics acceptable.  
- [ ] Bucket probabilities sum to 1 and align to contract boundaries exactly.  
- [ ] EV/backtest positive under conservative costs with acceptable drawdown profile.  
- [ ] Kill-switch and data-failure halts wired before any live-trading path.

