# Project Memory

> **RULE:** Keep this file concise, current, and operationally useful.

## Current Project State (2026-03)
- Repository focus: multi-city probabilistic weather forecasting for Kalshi daily max-temperature contracts.
- Primary objective: calibrated predictive distributions converted into contract bucket probabilities, then EV-filtered trading decisions with risk controls.
- Core city support: NYC, Chicago, Philadelphia, Atlanta, Austin.
- **Phase 4 expansion registered (2026-06-24):** Denver (`KXHIGHDEN`), Washington
  DC (`KXHIGHTDC`), Los Angeles (`KXHIGHLAX`), Miami (`KXHIGHMIA`), Phoenix
  (`KXHIGHTPHX`) are now registered cities, chosen by live-API contract
  verification (`results/expansion/contract_verification.json`). Houston is
  BLOCKED (series exists but no markets). All five expansion cities are
  **MONITOR** until they have >= 1 full year of real-price backtest; their
  rollout data collection + pipeline run is the remaining operational step.
- **Ticker correction:** Philadelphia's real Kalshi series is `KXHIGHPHIL`
  (not `KXHIGHPHL`); fixed across config/live-trading/benchmark/tests.
- Architecture is modularized across ingestion, feature engineering, modeling, calibration/bucketization, and trading simulation.
- Multi-city script flow has been unified (city passed via `--city`) with thin compatibility wrappers for legacy per-city commands.
- Promotion evaluations are implemented; city readiness differs by market edge and calibration robustness.
- **Promotion status (2026-06-24):** PHL and CHI **PASS all 14 gates (READY)**; NYC
  passes 11/14 (the 3 trading gates fail). NYC's model is well-calibrated and
  competitive (contract Brier 0.110, beats benchmark market 0.127, ECE 0.018,
  all seasons < 0.11) but its real OOS pre-settlement market (Brier **0.0988**)
  is too efficient to beat: the generic unified model, MOS-anchoring, AND the
  full NYC WGA/NWS/market-state stack (best OOS Brier 0.1018) all trade negative
  even under accurate fees. Optimal model+market blend weight is ~0 — the market
  already prices all available forecast signal. This is a legitimate "no genuine
  edge -> do not trade" outcome (Central Park is the most-traded, best-forecast
  station), not a pipeline defect.
- **Kalshi fee model corrected (2026-06-24):** backtests previously charged a
  flat 7c/contract (payout*(1-0.07)); Kalshi's real general-markets fee is the
  curved `ceil(0.07*P*(1-P))` per contract (~1.3-1.75c), charged on entry. See
  `src.trading.kalshi_fee_per_contract`. The overcharge had been masking CHI's
  genuine (directionally-correct) edge; with accurate costs CHI/PHL trade
  positive. Live/paper EV functions still use the flat rate — migrate them next.
- **seasonal_brier.json now reflects the promoted unified model** (via
  `scripts/build_unified_seasonal_brier.py`), not the weaker base benchmark, so
  the seasonal gate is consistent with the overall-Brier gate.
- **Critical data gap:** expansion cities (CHI, PHL, ATL, AUS) train on GHCN data but infer with ASOS-derived features. ASOS migration is the top data quality priority.

## Known Strategic Status by City
- **NYC:** mature benchmark stack (E/WGA/U families), strongest reference implementation. Pipeline now runs end-to-end via `--city nyc`; promoted as a calibrated *forecaster* (11/14) — market too efficient to trade (see status above). WGA V2: the `wga_v2_benchmark` import bug is fixed (it pulled SEASON_MAP from a deleted script; now imports from `src.seasons`), but the V2 *training* step that produced `results/wga_v2_model/.../predictions_{val,test}.csv` is gone. `scripts/train_wga_predictions.py` reconstructs it via `src.wga_data_pipeline.train_wga_city`, but `WGADataBuilder` only extracts 2 station features from the current processed layout and the model trains to garbage (NYC test MAE ~60 F). WGA needs a `WGADataBuilder` feature-extraction fix before it's usable; until then the unified stack falls back to the flat model. Even a working WGA would be bounded by the same efficient-market ceiling.
- **Chicago:** READY — all 14 gates pass; genuine directionally-correct OOS edge, real-Kalshi backtest positive under accurate fees.
- **Philadelphia:** weaker edge; calibration and simulated-market robustness remain key concerns.
- **Atlanta:** pipeline complete and promotion gates passed.
- **Austin:** pipeline complete but promotion criteria not yet consistently met.

## Operational Non-Negotiables
1. Contract alignment first (station, day boundary/timezone, units, bucket semantics, settlement rules).
2. Strict cutoff-time safety for live inputs (no delayed/training-only leakage into inference).
3. Chronological splits only; never random shuffle time-series data.
4. Train/inference feature parity is mandatory; quantify and correct mismatch when unavoidable.
5. **Train on ASOS (IEM hourly) data, not GHCN-Daily** — ASOS matches the operational inference source. GHCN may be used for secondary validation only.
6. Trading requires calibrated probabilities and full-cost EV accounting (fees + slippage + execution uncertainty).
7. Kill switch required for missing critical inputs, schema drift, calibration drift, or execution anomalies.

## Quality and Evaluation Expectations
- Use proper probabilistic metrics (CRPS/NLL) and contract-level bucket metrics (including Brier).
- Benchmark against persistence, climatology, linear/ridge, and market-implied probabilities when available.
- Evaluate by regime/season; avoid aggregate-only reporting.
- Keep model complexity only when it materially improves out-of-sample calibration and contract performance.

## Repo Working Memory
- City/contract definitions and bucket logic: `src/city_config.py`.
- Unified city pipeline scripts live under `scripts/` and should be preferred over bespoke one-off scripts.
- Artifacts should remain organized under `data/<city>/`, `models/<city>/`, and `results/<city>/`.

## LLM Workflow Reminder
- For coding and research tasks, **always delegate implementation/research execution to the `analyst` subagent first**, then synthesize and review results before finalizing.
