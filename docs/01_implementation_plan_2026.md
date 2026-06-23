# Implementation Plan: Positive-EV Models, 10-City Portfolio, Real-Time EV Dashboard

> Status: Phase 0 COMPLETE (2026-06-12). Phase 1 COMPLETE. Phase 2 COMPLETE
> (2026-06-22). Phase 3 COMPLETE (2026-06-23). Phases 4-5 pending.
> Authoritative per-city metrics: `results/baseline_ledger.json`
> (regenerate with `python scripts/build_baseline_ledger.py`).
>
> **Phase 1 (data parity) — DONE:** ASOS migration for CHI/ATL/PHL/AUS
> (`results/<city>/asos_migration/`), 7am-ET cutoff manifest
> (`results/cutoff_manifest.json` + `src/data_sla.py`) with a freshness
> kill-switch wired into `src/schema_validation.py` and
> `src/operational_features.py`, and the NYC reorg under `data/nyc/`.
>
> **Phase 2 (model optimization) — DONE:** all seven deliverables landed as
> tested code + real artifacts under `results/<city>/diagnostics/`:
> 1. Diagnostics — `src/model_diagnostics.py` + `scripts/run_model_diagnostics.py`
>    (residual/sigma/PIT/model-vs-market). Austin's broken constant-54.6F sigma
>    surfaced here.
> 2. NWP/MOS lever — `src/mos_features.py` + `scripts/run_mos_residual_benchmark.py`
>    (MOSCorrectionNet residual model; Chicago OOS CRPS 2.88 station-only ->
>    1.81 with MOS). Operational parity via `add_operational_mos_features`;
>    download CLI extended to KPHL/KATL/KAUS.
> 3. Austin deep-dive — `results/austin/diagnostics/root_cause.md` (model
>    quality, not station alignment; ASOS/GHCN offset at KAUS is -0.29F).
> 4. Hparam search — `scripts/run_hparam_search.py` (rolling-origin 3-fold
>    chronological CV, single OOS eval).
> 5. Ensembling — `src/ensembling.py` + mixture bucket semantics
>    (inverse-val-CRPS weighted Gaussian mixtures).
> 6. Distribution heads — `src/distribution_heads.py` +
>    `scripts/run_distribution_head_comparison.py` (Gaussian/quantile/mixture
>    by OOS CRPS + contract Brier).
> 7. Philadelphia SON — `src/frontal_features.py` + `RegimeConditionalCalibrator`
>    + `scripts/run_son_calibration_diagnostic.py`.
>
> **Phase 2 gate status (unchanged conclusion):** no city yet beats its market
> Brier on real OOS presettlement prices, so all remain **MONITOR**. The MOS
> lever is the path forward (full multi-city MOS collection + retrain is Phase
> 3 work); thresholds were not tuned to manufacture EV.
>
> **Phase 3 (trading/EV decision quality) — DONE:** three deliverables landed
> as tested, importable `src/` cores behind thin CLIs (94 new tests; full suite
> green):
> 1. Real-price strategy refit — `scripts/run_real_strategy_sweep.py` +
>    `src/strategy_selection.py`. Drives `trading.generate_strategy_grid` over
>    real OOS presettlement rows, joined to real `bid_cents`/`ask_cents` for a
>    per-row half-spread (not flat slippage); 7% fee;
>    `trading.compute_conservative_ev` as the gate. Fits on an earlier
>    chronological OOS slice, scores once on an untouched later holdout, and
>    persists per-city params to `results/<city>/strategy.json` (read back by
>    `LiveTradingHarness`, which now also covers ATL/AUS). NOTE: the plan's
>    literal "fit 2023-24 / validate 2025" cannot be honoured where the OOS
>    window is entirely 2025+ (e.g. CHI, whose 2022-24 rows are in-sample and
>    trading them would be leakage), so the split is chronological within OOS.
> 2. Promotion decisions — `strategy_selection.decide_promotion` enforces the
>    Phase 3.2 bar on the real-price holdout (model Brier < market Brier; P&L >
>    0; Sharpe >= 1.0; >= 50 trades; DD >= -30%).
> 3. Daily inference + paper loop — `scripts/run_daily_inference.py` +
>    `src/daily_inference.py` (enforces the 7am-ET cutoff manifest, kill-switch
>    on stale/leaking critical inputs, writes `results/<city>/live/signals_<date>.json`)
>    and `scripts/run_paper_trading.py` + `src/paper_trading.py` (prices the
>    model against the day's *actual* re-struck Kalshi buckets via
>    `bucket_semantics`, settles, audits; read-only paper mode only). Kill-switch
>    scenario tests + a losing-week replay are in `tests/test_paper_trading.py`.
>
> **Phase 3 gate status (honest):** the refit holdouts confirm Phase 2's
> conclusion — **CHI and PHL both remain MONITOR** (model Brier does not beat
> market Brier on the untouched holdout; no positive verified edge). No
> thresholds were tuned to manufacture EV. `strategy.json` records each city's
> selected strategy and MONITOR decision; PROMOTED stays empty until a city's
> model genuinely beats the market (the MOS-retrain lever from Phase 2/4).

## Goals

1. More accurate probabilistic temperature models and positive-EV trading
   with proper bet sizing (capped/fractional Kelly, full-cost EV).
2. Positive-EV models in all 5 current cities (NYC, CHI, PHL, ATL, AUS),
   plus expansion to 5 new cities: Denver + Washington DC + 3 chosen by
   verified Kalshi liquidity.
3. A Streamlit dashboard for real-time identification of positive-EV trades
   in Kalshi daily-high-temperature markets.

**Decisions:** paper trading only (no authenticated order placement —
`src/kalshi_client.py` stays read-only); Streamlit; foundation-first
sequencing.

**Honesty principle (core of "proper betting decision making"):** deliver
positive EV where it genuinely exists, honest no-go elsewhere. Every city
ends in one of three states: **PROMOTED** (verified real-price edge,
paper-trade signals with sizing), **MONITOR** (forecasts shown, "no verified
edge" badge, no sizing), or **BLOCKED** (data/contract problem). Never tune
EV thresholds to manufacture P&L.

## Ground rules (all phases)

- **7am ET inference cutoff (hard requirement):** every feature used at
  inference must be verifiably available by 7:00 AM Eastern Time on the day
  of the market, for all cities regardless of local timezone. This is the
  reason for the ASOS migration: ASOS/IEM publishes hourly (GHCN-Daily lags
  days and can never feed live inference). Deliverable: a cutoff manifest
  extending `src/data_sla.py` listing, per inference feature: source,
  publication schedule/latency, latest usable timestamp at 7am ET, fallback
  behavior. Training features must use only data that would have been
  available by 7am ET on each historical day. Violations are kill-switch
  events.
- Canonical evaluation = real Kalshi presettlement prices
  (`data/kalshi_presettlement_<city>.csv`), out-of-sample rows only.
  Simulated-market backtests are smoke tests.
- Chronological splits only; calibrators/scalers fit on train; >= 200 OOS days.
- All bucket probability/outcome code routes through `src/bucket_semantics.py`
  (verified settlement semantics: Kalshi settles on rounded integer deg-F,
  `lo <= round(TMAX) < hi`).

---

## Phase 0 — Truth & unblocking — COMPLETE (2026-06-12)

What was found and fixed (see commits 2f34211, 95f118f):

1. **Drawdown bug fixed**: stakes capped at available bankroll, bankruptcy
   halts the backtest (`busted` flag, hard gate failure), drawdown percent
   bounded in [-100%, 0] via `src/trading.py::compute_drawdown_metrics`
   (Atlanta previously reported an impossible -134.3%).
2. **Artifact filename mismatch fixed**: canonical real-price metrics file is
   `backtest/real_kalshi_metrics.json`; promotion reads both schemas
   (variants + flat) and no longer fabricates a -100% sentinel.
3. **Promotion frameworks unified** into `src/promotion_report.py` (14 gates):
   ported `beats_nws` and `real_kalshi_pnl`; added `trading_source_real`
   (simulated-only trading evidence fails promotion when presettlement data
   exists). `scripts/run_promotion_evaluation.py` is a thin delegator.
4. **Settlement semantics verified and centralized**: 100% agreement with all
   14,116 settled contract rows; half-degree rounding shift applied to every
   CDF->bucket conversion (`src/bucket_semantics.py`). Austin's settlement
   station verified CORRECT (Bergstrom — Camp Mabry hypothesis disproven).
5. **In-sample leakage removed from backtests**: real-price backtests now
   filter to OOS rows only (experiments script + synthesis predictions carry
   a `period` column).
6. **Synthesis MLP convergence pathology fixed** (sigma absorbed the initial
   mu residual; mu never trained — test MAE was ~63F garbage). mu head now
   initializes at target mean; log_sigma clamped.
7. **Schema validation fixed** to match real artifacts (z-scored features,
   city-prefixed TMAX targets) — benchmark stages were silently aborting for
   every city.
8. **Checkpoints saved** to `models/<city>/` (benchmark NN + synthesis);
   `model_checkpoints` gate now satisfiable.
9. **Docs reconciled**: Project_Plan Section 1 rewritten from regenerated
   artifacts; stale "+240%/6.07 Sharpe" claims removed.

**Honest baseline (results/baseline_ledger.json, 2026-06-12):** with leakage
removed, real prices, capped stakes, and rounding-aware probabilities, NO
city currently shows a verified positive-EV edge:

| City | Real OOS P&L | Sharpe | Model vs market Brier (tradeable universe) | Gates |
|------|-------------|--------|---------------------------------------------|-------|
| CHI (U9) | -$44.50 | -0.42 | 0.1046 vs 0.1008 (worse) | 12/14 |
| PHL (U9) | +$8.75 | 0.18 | 0.1062 vs 0.1080 (noise-level edge) | 13/14 |
| ATL | -$19.70 (53 trades) | -4.46 | 0.165 vs 0.125 (worse) | 8/14 |
| AUS | bust (-$1000) | -6.08 | 0.236 vs 0.145 (much worse) | 8/14 |
| NYC | not yet evaluated under unified gates (Phase G) | — | — | — |

**Diagnosed root cause for the missing edge:** the pipeline's synthesis stage
uses NO NWP/MOS forecast data (it is a 6-feature recalibration of the base
model's own mu/sigma; the richer `src/synthesis_model.py` station+NWP fusion
is not wired in). The market is NWP-informed; station lags alone cannot beat
it. This is the Phase 2 priority.

---

## Phase 1 — Data parity: ASOS migration + cutoff manifest + NYC reorg

1. **ASOS migration** (Project_Plan Phase E) for CHI -> ATL -> PHL -> AUS:
   `python scripts/run_asos_migration.py --city <c>` (IEM collection,
   hourly->daily, ASOS-vs-GHCN cross-validation report, rebuild
   `data/<city>/processed/` with ASOS TMAX primary, KS-test parity), then
   full pipeline rerun. ASOS is required for the 7am ET cutoff, not just
   train/inference parity.
2. **Cutoff manifest**: author the 7am-ET availability manifest for every
   operational feature source (ASOS, MOS/NWP runs, soundings, prior-day
   settlements) extending `src/data_sla.py`; wire a freshness validator into
   `src/schema_validation.py` and `src/operational_features.py`.
3. **NYC reorganization** (Phase G): move root-level NYC data into
   `data/nyc/{raw,processed}/`, update references
   (`src/nyc_benchmark_registry.py`, `config.py`), generate NYC's first
   unified promotion report.

**Acceptance:** offset report per city; KS parity; post-migration OOS Brier
<= ledger value + 0.002; manifest covers every inference feature; NYC runs
from `data/nyc/`.

## Phase 2 — Model optimization (the accuracy core)

1. **Diagnostics first** — new `scripts/run_model_diagnostics.py`: residual
   bias by season/regime, sigma vs realized error, PIT, per-bucket Brier vs
   market, model-vs-market disagreement -> `results/<city>/diagnostics/`.
2. **NWP/MOS integration (primary lever)**: extend
   `scripts/download_iem_mos_data.py` to KPHL/KATL/KAUS (KORD + NYC exist in
   `data/chicago/mos/`, `data/mos/`, `data/airport_mos/`); add cutoff-safe
   MOS features (MOS TMAX morning run published before 7am ET, MOS-climo
   anomaly, GFS/NAM disagreement); wire `MOSCorrectionNet`
   (TMAX - MOS_TMAX residual distribution, already in
   `src/advanced_model.py`) into `run_benchmark.py`; mirror features in
   `src/operational_features.py`. Replace the weak 6-feature synthesis stage.
3. **Austin deep-dive** (station alignment already verified correct):
   remaining suspects are model quality (no NWP) and the GHCN/ASOS offset
   (Phase 1 report). Deliverable: `results/austin/diagnostics/root_cause.md`.
4. **Hyperparameter search** — new `scripts/run_hparam_search.py --city <c>
   --family <f> --budget 50`: rolling-origin 3-fold chronological CV, select
   by val CRPS, single final OOS eval.
5. **Ensembling**: inverse-val-CRPS weighted mixture of top families (reuse
   `EnsembleCalibrator`); extend `gaussian_to_bucket_probs` to mixtures.
6. **Distribution heads**: Gaussian vs 7-quantile vs 2-component mixture per
   city by OOS CRPS + contract Brier.
7. **Philadelphia SON failure**: seasonal-regime calibration + fall
   frontal-passage features (dewpoint depression trend, wind-direction regime
   from `src/wind_gated_attention.py`).

**Gate to Phase 3 (per city):** improves ledger CRPS + contract Brier; Brier
< NWS baseline AND < market Brier on real OOS presettlement prices.
**No-go rule:** model Brier >= market Brier after the above -> MONITOR; do
not tune thresholds to manufacture EV.

## Phase 3 — Trading/EV decision quality

1. **Real-price strategy refit** — new `scripts/run_real_strategy_sweep.py`
   driving `trading.generate_strategy_grid` over real presettlement rows:
   fit on 2023-2024, validate untouched on 2025. Cost realism: half-spread
   from `bid_cents`/`ask_cents` columns instead of flat 2% slippage; 7% fee;
   `trading.compute_conservative_ev` as gate-side EV. Persist per-city params
   to `results/<city>/strategy.json`, read by `LiveTradingHarness`.
2. **Promotion decisions**: PROMOTED requires (all on real prices) model
   Brier < market Brier; 2025-slice P&L > 0, Sharpe >= 1.0, >= 50 trades;
   DD >= -30%; all gates pass.
3. **Daily inference + paper loop** (required by dashboard):
   - New `scripts/run_daily_inference.py --city <c>`: validate inputs against
     the 7am ET cutoff manifest (abort + kill-switch event if stale) ->
     cutoff-safe features -> promoted checkpoint -> calibrator -> write
     `results/<city>/live/signals_<date>.json`.
   - New `scripts/run_paper_trading.py`: read signals, poll prices
     (`KalshiClient.get_markets` + `parse_market_buckets`),
     `harness.evaluate_trades`, settle next day, `save_audit_log`.
   - Kill-switch scenario tests + historical losing-week replay.

## Phase 4 — Expansion: Denver, DC, +3 by verified liquidity

1. **Contract verification first** — new `scripts/verify_kalshi_contracts.py`
   for candidates {DEN, DC, MIA, LA, HOU, PHX}: discover real series tickers
   via the public API (naming is irregular — ATL is `KXHIGHTATL`; resolve the
   PHL `KXHIGHPHL` vs `KXHIGHPHIL` discrepancy between `src/city_config.py`
   and `scripts/fetch_kalshi_multi_city.py`), settlement station from market
   rules, bucket structure (`parse_market_buckets`), 7-day liquidity sample
   (volume, OI, spread). Output
   `results/expansion/contract_verification.json`; pick top 3 by
   spread-adjusted liquidity. No city config hardcoded before this artifact
   exists.
2. **Rollout per city** (DEN, DC first — checklists in Project_Plan Section
   10; <= 2 in flight), per `.claude/skills/weather-model/SKILL.md`,
   ASOS-first from day one: register `CityConfig` +
   `city_config_runtime_data` + `SUPPORTED_CITIES` + `CITY_THRESHOLDS`
   (entropy-derived) + fetch-script city lists; collect ASOS/MOS/Kalshi data;
   run the full pipeline; same gates. New cities need >= 1 full year of
   real-price backtest before PROMOTED; otherwise MONITOR.

## Phase 5 — Streamlit dashboard

Files: `dashboard/app.py` (multi-city EV overview),
`dashboard/pages/2_market_detail.py` (orderbook depth, model-vs-market),
`dashboard/pages/3_calibration_health.py` (PIT drift, ECE trend, kill-switch,
freshness), `dashboard/pages/4_paper_trading.py` (trade log, cumulative P&L,
predicted-EV vs realized-P&L scatter), plus pure-logic
`src/dashboard/live_signals.py` (signals + market prices -> EV table).
Add `streamlit>=1.30`, `plotly>=5` to requirements.

- Model side: latest `results/<city>/live/signals_<date>.json`; stale ->
  warning, no EV computed.
- Market side: `KalshiClient.get_markets(status="open")` +
  `parse_market_buckets`; `st.cache_data(ttl=60)`; client rate limiter
  respected (1 call/city/refresh on overview).
- EV table: reuse `trading.compute_ev_best`, `compute_conservative_ev`,
  `capped_kelly` + per-city `strategy.json`. Kelly-sized recommendations for
  PROMOTED cities only; MONITOR cities show probabilities with a "no
  verified edge" badge.
- Health page: `DashboardData.get_multi_city_status()` from
  `src/dashboard/dashboard_data.py` as-is.
- Tests: `tests/test_live_signals.py` covers the EV join and bucket-label
  matching between `parse_market_buckets` output and `cfg.bucket_labels`
  (riskiest seam) — no network in tests. Zero authenticated/write Kalshi
  endpoints anywhere.

## Risk register

| Risk | Detection | Response |
|---|---|---|
| A city has no edge (current state: ALL cities) | model Brier >= market Brier on real OOS | MONITOR; never manufacture P&L |
| Synthesis/NWP inputs remain unwired | Phase 2 diagnostics | MOS-residual family replaces synthesis stage |
| Strategy overfits 2023-24 prices | 2025 holdout slice | demote strategy, widen EV threshold |
| API instability across 10 cities | freshness checks + dashboard health page | stage-level run-fail via `schema_validation.enforce_preconditions` |
| New-city ticker/bucket misassumption | verification artifact required before config merge | BLOCKED until verified |

## Sequencing

Phase 0 (done) -> Phase 1 (per-city serial) -> Phase 2 (MOS integration
first, then hparam/ensemble) -> Phase 3 -> Phase 4 (verification script can
run during Phase 2) -> Phase 5 (can start once `run_daily_inference.py`
exists; MVP overview page may ship early on MONITOR data).
