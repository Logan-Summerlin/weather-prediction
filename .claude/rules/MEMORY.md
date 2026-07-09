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
- **Promotion status (2026-07-09):** PHL, CHI and now **NYC PASS all 14 gates**.
  NYC's path was NOT meteorological: the pre-settlement market (2025 OOS Brier
  0.0988) beats every weather variant in every slice (optimal blend weight
  0.00 everywhere — 2026-06-24 conclusion confirmed and final). The trading
  gates are cleared by **U10_market_debias** (`src/market_debias.py`): a
  walk-forward longshot-discount + book-coherence repricing of the market
  itself (favorite-longshot bias: ≤10c buckets realized at ~0.5x price in
  2023/24/25), traded as NO-side-only on mids in (0.05,0.25] at REAL bids
  with curved fees. +$28.10 over 2025-01→2026-07 (249 trades, 92.4% win,
  max DD -3.2%); best 2025 OOS Brier of any variant (0.0990). Full audit:
  `results/audits/nyc_u10_market_debias_audit.md`.
- **⚠️ NYC edge decay / kill switch (2026-07-09):** the longshot bias decayed
  monotonically (-5c 2023/24 → -2.5c 2025 → **-0.6c in the 2026-05→07 true-OOS
  fetch**, where the strategy lost -$14.20 over 44 trades). 2026 books are
  tight (1c median spread, overround 4.4c). U10's checkpoint
  (`models/nyc/u10_market_debias.json`) embeds a kill switch — trade only if
  trailing 120-day zone bias ≤ -1.5c — which is currently **HALTED**. NYC is
  promoted as a validated pipeline; do NOT deploy capital until the bias
  signal re-widens. Kalshi purged pre-2026-May market data from the public
  API (old tickers 404); `scripts/fetch_kalshi_nyc_2026_oos.py` re-fetches
  the retained window (candlesticks now use `close_dollars` string schema).
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
