# Locally Run Real-Time Positive-EV Dashboard Plan: NYC / CHI / PHL

> Status: Proposed implementation plan  
> Scope: local-only dashboard for read-only Kalshi market monitoring and paper-trading opportunity review across NYC, Chicago, and Philadelphia daily-high-temperature markets.  
> Primary goal: surface *honest* positive expected value (EV) opportunities after fees, spread/slippage, data freshness, calibration, and promotion/monitor gates.  
> Non-goal for this phase: authenticated live order placement.

## 1. Current repository assessment

### 1.1 Existing assets to reuse

The repository already contains most of the forecasting, market-data, and trading primitives needed for a local dashboard:

| Capability | Current implementation | Dashboard use |
|---|---|---|
| City registry and contract metadata | `src/city_config.py` exposes `CityConfig`, registered city lookup, Kalshi ticker, target station, time zone, bucket edges, and bucket labels. | Drive city selectors, ticker routing, target-station display, bucket labels, and model artifact paths. |
| Settlement-aware bucket math | `src/bucket_semantics.py` is the centralized settlement semantics layer referenced by inference, paper trading, and live harness code. | Convert model distributions to contract probabilities without duplicating contract logic. |
| Public Kalshi market client | `src/kalshi_client.py` provides a read-only API client with retries, rate limiting, pagination helpers, market parsing, and implied probabilities. | Poll open markets, quotes/order books, and series metadata. |
| Live/paper trading harness | `src/live_trading.py` defines city-aware ticker routing, `DailyPrediction`, `TradeRecord`, `KillSwitch`, `LiveTradingHarness`, and EV evaluation through `TradingStrategy`. | Reuse EV, sizing, audit, and kill-switch behavior rather than inventing a dashboard-only trading path. |
| Paper trading utilities | `src/paper_trading.py` prices model distributions against each day’s actual Kalshi contracts and writes audit logs. | Use the same contract-by-contract pricing path for live open markets. |
| Dashboard data skeleton | `src/dashboard/dashboard_data.py` already aggregates per-city data freshness, model availability, benchmark status, calibration artifacts, trading audits, and cross-city health. | Extend this into a real-time opportunity feed and dashboard state service. |
| Daily inference signals | `src/daily_inference.py` and `scripts/run_daily_inference.py` write daily city signal JSON under `results/<city>/live/`. | Dashboard should consume latest generated signals first, then optionally trigger local refresh commands. |
| Strategy selection | `src/strategy_selection.py` and per-city `results/<city>/strategy.json` define conservative strategy parameters and promotion decisions. | Show whether a city is PROMOTED/MONITOR/BLOCKED and prevent sizing on unverified edges. |
| Historical validation artifacts | `results/<city>/diagnostics/`, `results/<city>/backtest/`, `results/<city>/trading/`, and `results/baseline_ledger.json`. | Populate model-quality, calibration, and “why no trade” panels. |

### 1.2 Important current constraints

1. **Read-only Kalshi posture.** The codebase is explicitly oriented around public/read-only Kalshi access and paper mode for this phase. The dashboard should not submit orders until a separate authenticated-execution project is approved.
2. **7am ET cutoff discipline.** The daily forecast inputs must be available by the repository’s hard 7:00 AM America/New_York cutoff; stale or post-cutoff feature data is a kill-switch event.
3. **No false positive-EV claims.** Current documentation says CHI and PHL remain MONITOR because model Brier has not beaten market Brier on untouched real-price holdouts. The dashboard can display mathematical live edges, but sizing/action badges must distinguish “verified tradable edge” from “monitor-only discrepancy.”
4. **Contract grids are re-struck.** The dashboard must price each live Kalshi contract’s own parsed thresholds rather than assuming the static city bucket grid is identical to the day’s market contracts.
5. **Fee model consistency.** Existing trading code contains both older flat `fee_rate` EV functions and a more accurate Kalshi per-contract fee helper. The dashboard plan should standardize the opportunity feed on a single explicit cost model and label it in the UI.

## 2. Product definition

### 2.1 User stories

1. As a local operator, I can run one command and open a browser page showing NYC, CHI, and PHL market opportunities.
2. As a trader, I can see the latest calibrated model probability, market YES/NO prices, spread, estimated fees, conservative slippage, EV, suggested paper size, and reason codes for every contract.
3. As a risk manager, I can see kill-switch status, data freshness, model/calibration availability, and whether each city is PROMOTED, MONITOR, or BLOCKED.
4. As a model owner, I can inspect which model artifact/signals produced the probabilities, when the forecast was generated, and whether inputs respected the 7am ET cutoff.
5. As an auditor, I can export the current opportunity snapshot and later reconcile it against paper trades and settlements.

### 2.2 Dashboard operating modes

| Mode | Purpose | Market data | Forecast data | Trading behavior |
|---|---|---|---|---|
| `offline` | Development and demo | cached JSON/CSV fixtures only | latest local signal files | no network, no trades |
| `monitor` | Normal local operation | public Kalshi polling | latest signals, optionally refreshed by CLI | no orders; shows opportunity feed |
| `paper` | Local paper-trading review | public Kalshi polling | latest signals | writes paper-trade audit logs through existing harness |
| `live` | Future only | authenticated Kalshi | latest signals | explicitly out of scope for this plan |

Default launch should be `monitor`.

## 3. Target architecture

### 3.1 Components

```text
scripts/run_ev_dashboard.py
        |
        v
Streamlit UI (or FastAPI + lightweight UI if Streamlit polling proves limiting)
        |
        v
src/dashboard/opportunity_service.py
        |-- City registry: src.city_config
        |-- Forecast signal loader: results/<city>/live/signals_YYYY-MM-DD.json
        |-- Market polling: src.kalshi_client.KalshiClient
        |-- Contract parsing: src.kalshi_client.parse_market_buckets
        |-- Probability mapping: src.paper_trading.model_prob_for_contract or bucket_semantics
        |-- EV/cost model: src.trading
        |-- Strategy/risk: src.live_trading.load_city_strategy + KillSwitch
        |-- Health summary: src.dashboard.dashboard_data.DashboardData
        v
results/dashboard/opportunities_<timestamp>.json
results/<city>/trading/trading_audit_<date>.json (paper mode only)
```

### 3.2 New files proposed

| File | Responsibility |
|---|---|
| `src/dashboard/opportunity_service.py` | Pure-Python service that builds the real-time opportunity table. No UI dependencies. |
| `src/dashboard/app.py` | Streamlit app: controls, tables, charts, refresh loop, export buttons. |
| `scripts/run_ev_dashboard.py` | Thin CLI wrapper to launch Streamlit with safe defaults. |
| `config/dashboard.yaml` | Local settings: city list, refresh interval, mode, EV thresholds, slippage assumptions, artifact retention. |
| `tests/test_dashboard_opportunity_service.py` | Unit tests using mocked Kalshi responses and local signal fixtures. |
| `tests/test_dashboard_app_smoke.py` | Optional smoke test for importability/config validation, not visual rendering. |

### 3.3 Existing files to modify

| File | Change |
|---|---|
| `src/dashboard/dashboard_data.py` | Add opportunity snapshot loading and expose latest opportunity counts by city. |
| `src/trading.py` | Add or clarify one canonical dashboard EV function using explicit fee + slippage parameters and bid/ask side semantics. |
| `src/kalshi_client.py` | Ensure market parser emits bid/ask/last/spread/ticker/date enough for dashboard tables. |
| `scripts/README.md` | Document dashboard command and modes. |
| `README.md` | Add local dashboard quickstart link. |
| `docs/00_canonical_docs_index.md` | Add this plan once accepted. |

## 4. Data flow

### 4.1 Forecast side

1. For each city in `nyc`, `chi`, `phl`, read the most recent `results/<city>/live/signals_YYYY-MM-DD.json` generated by `scripts/run_daily_inference.py`.
2. Validate:
   - signal date matches today’s market date or is explicitly flagged stale;
   - `mu`, `sigma`, and bucket probabilities are finite;
   - source feature SLA status is present and non-critical;
   - calibration identifier/artifact is present;
   - city strategy file and promotion decision are loaded when available.
3. If signal is missing or stale, set city state to `BLOCKED_FOR_TODAY` in the dashboard and show the exact missing artifact path and refresh command.
4. Optional controlled refresh: provide a “run inference locally” button only if it executes `python scripts/run_daily_inference.py --city <city>` in a subprocess and captures logs; never silently recompute with post-cutoff data.

### 4.2 Market side

1. Poll `KalshiClient.get_markets(series_ticker=<city ticker>, status="open")` for each city.
2. Parse markets into contract rows with ticker, title, event ticker, threshold low/high, direction, yes bid/ask, no bid/ask, last price, volume/open interest when present, and computed mid/implied probability.
3. Fetch order books only for top candidate contracts or on row expansion to avoid excessive calls.
4. Cache raw API responses to `results/dashboard/raw_kalshi/<timestamp>_<city>.json` with a retention policy.
5. If API polling fails, keep the last-good snapshot but mark all EV values stale and disable paper-trade buttons.

### 4.3 Probability and EV side

For every parsed contract:

1. Compute model YES probability against the contract’s parsed `(lo, hi, direction)` using the same settlement-aware probability path used by paper trading.
2. Choose executable price assumptions:
   - buying YES uses `yes_ask / 100`;
   - buying NO uses `no_ask / 100` or `1 - yes_bid / 100`, depending on available fields and consistency checks;
   - mid prices may be displayed but must not be used for executable EV by default.
3. Apply a conservative cost model:
   - Kalshi trading fee per contract;
   - configurable slippage buffer in cents or bps;
   - optional minimum spread/liquidity guard.
4. Compute:
   - executable EV per YES contract;
   - executable EV per NO contract;
   - best direction;
   - edge in probability points;
   - maximum size under strategy/risk caps;
   - reason code if rejected.
5. Sort opportunities by conservative EV after costs, then by liquidity and strategy eligibility.

### 4.4 State persistence

Each refresh writes an append-only snapshot:

```json
{
  "generated_at": "2026-07-09T11:05:00Z",
  "mode": "monitor",
  "cities": {
    "nyc": {
      "status": "ok",
      "signal_path": "results/live/signals_2026-07-09.json",
      "market_poll_age_seconds": 4.2,
      "opportunities": []
    }
  }
}
```

The exact path should be `results/dashboard/opportunities_<YYYYMMDD_HHMMSS>.json`, plus a symlink or copied latest file at `results/dashboard/latest_opportunities.json` for downstream tools.

## 5. UI specification

### 5.1 Page layout

1. **Header:** current UTC time, local ET time, mode, refresh interval, last Kalshi poll, and “paper/live disabled” badge.
2. **Cross-city summary cards:**
   - number of healthy cities;
   - number of stale/blocked cities;
   - count of positive-EV executable opportunities;
   - total suggested paper exposure;
   - max EV opportunity after costs.
3. **City status row:** NYC / CHI / PHL cards showing model date, calibration status, data SLA, promotion status, kill switch, best benchmark Brier, latest paper P&L.
4. **Opportunity table:** one row per contract with city, ticker, bucket/threshold, model probability, executable YES/NO prices, spread, fee, slippage, conservative EV, direction, suggested size, status, reason code.
5. **Detail drawer:** selected row shows raw market fields, distribution parameters, bucket probability calculation, strategy thresholds, order book depth if fetched, and audit trace.
6. **Diagnostics tabs:**
   - Calibration: latest PIT/reliability artifacts when present.
   - Backtest: latest real-price metrics and promotion gate failures.
   - Logs: dashboard poll logs, inference logs, paper-trade audit links.
7. **Controls:** city filter, min EV, min liquidity, include MONITOR cities toggle, refresh now, export snapshot, run paper evaluation.

### 5.2 Visual semantics

| Badge | Meaning |
|---|---|
| Green `TRADEABLE-PAPER` | City promoted, data/calibration healthy, EV clears conservative threshold and risk limits. |
| Yellow `MONITOR-ONLY` | Positive mathematical EV displayed, but city lacks verified market-beating promotion. No sizing by default. |
| Red `BLOCKED` | Critical data, calibration, contract, or market polling failure. |
| Gray `NO EDGE` | EV does not clear costs/thresholds. |
| Purple `STALE MARKET` | Using last-good market snapshot because polling failed or exceeded freshness SLA. |

## 6. Implementation phases

### Phase A — Repository-grounded design and config (0.5-1 day)

**Tasks**

1. Add `config/dashboard.yaml` with cities `nyc`, `chi`, `phl`, mode `monitor`, refresh interval, max market age, min EV, slippage, order-book polling cap, and export directory.
2. Define `OpportunityRow`, `CityDashboardState`, and `DashboardSnapshot` dataclasses in `src/dashboard/opportunity_service.py`.
3. Add config loader with environment-variable overrides for local paths, but no secrets.
4. Add tests for config defaults and city filtering.

**Acceptance criteria**

- Service can instantiate without network.
- Invalid city code or unsafe mode fails fast.
- Defaults do not enable live trading.

### Phase B — Signal and health ingestion (1 day)

**Tasks**

1. Implement latest signal discovery per city.
2. Parse `DailyPrediction` compatible fields: date, mu, sigma, model name, bucket probabilities, bucket labels.
3. Integrate `DashboardData.get_multi_city_status()` for freshness/model/calibration checks.
4. Load per-city strategy and promotion files where available.
5. Produce reason codes for missing signal, stale signal, missing calibration, missing strategy, and kill switch.

**Acceptance criteria**

- Unit tests cover present/missing/stale signals.
- Snapshot includes health state for NYC/CHI/PHL even when artifacts are missing.
- No network calls are made in signal-only tests.

### Phase C — Kalshi market polling and parsing (1-2 days)

**Tasks**

1. Wrap `KalshiClient` in a dashboard poller with timeout, retry logging, and last-good cache.
2. Normalize market rows across NYC/CHI/PHL into a single schema.
3. Extend parser if needed to retain all executable quote fields.
4. Add fixtures using mocked Kalshi markets for above/below/between contracts.
5. Add freshness metadata and stale-market guard.

**Acceptance criteria**

- Tests use mocked HTTP only.
- Market rows include enough fields to compute executable YES and NO EV.
- Failed poll never produces a false fresh opportunity.

### Phase D — Cost-aware opportunity engine (1-2 days)

**Tasks**

1. Implement canonical `compute_dashboard_ev()` or equivalent service method that accepts model probability, yes bid/ask, no bid/ask, fee model, slippage, and direction.
2. Use settlement-aware contract probabilities from `paper_trading` / `bucket_semantics` rather than static bucket labels.
3. Apply strategy thresholds from `TradingStrategy` and promotion status.
4. Add reason codes:
   - `ev_below_threshold`;
   - `spread_too_wide`;
   - `liquidity_missing`;
   - `city_not_promoted`;
   - `calibration_missing`;
   - `data_sla_failed`;
   - `kill_switch_active`.
5. Persist `latest_opportunities.json` and timestamped snapshots.

**Acceptance criteria**

- Tests prove YES and NO side EV use executable ask prices, not mid prices.
- Positive EV after mid but negative after spread/costs is rejected.
- MONITOR cities show opportunities but `suggested_size=0` unless an explicit `show_monitor_sizing` dev flag is set.

### Phase E — Streamlit dashboard app (1-2 days)

**Tasks**

1. Create `src/dashboard/app.py` with config sidebar, auto-refresh, status cards, tables, and detail panel.
2. Add `scripts/run_ev_dashboard.py` to invoke Streamlit with repo-root path handling.
3. Use `st.cache_data` only for low-risk static artifacts; never cache market data beyond configured freshness.
4. Add export button for current JSON/CSV snapshot.
5. Add optional paper-trade button that calls existing paper harness in paper mode only.

**Acceptance criteria**

- `python scripts/run_ev_dashboard.py --mode monitor` launches locally.
- Dashboard remains usable when one city is blocked or Kalshi polling fails.
- UI clearly differentiates mathematical edge from verified tradeable edge.

### Phase F — Audit, paper workflow, and runbook (1 day)

**Tasks**

1. Add snapshot-to-paper-trade workflow for selected rows or all eligible rows.
2. Write audit artifacts under existing `results/<city>/trading/` conventions.
3. Add docs for daily operation, troubleshooting, and incident response.
4. Add retention cleanup for dashboard snapshots.
5. Add “known limitations” panel in the UI.

**Acceptance criteria**

- Paper actions are reproducible from saved snapshot IDs.
- No code path places authenticated orders.
- Runbook includes exact commands and failure modes.

### Phase G — End-to-end validation (1 day)

**Tasks**

1. Add integration test with mocked market data and local signal fixture for all three cities.
2. Run relevant existing tests: Kalshi client, trading, paper trading, dashboard service, daily inference smoke if fixtures permit.
3. Manually run dashboard in offline fixture mode and monitor mode.
4. Verify all persisted JSON schemas are deterministic and documented.

**Acceptance criteria**

- Full dashboard opportunity snapshot generated deterministically under mocked data.
- No random shuffling or post-cutoff data access introduced.
- All positive-EV rows include explicit cost assumptions and rejection/eligibility reason.

## 7. Proposed data schemas

### 7.1 `OpportunityRow`

| Field | Type | Notes |
|---|---|---|
| `city_code` | string | `nyc`, `chi`, `phl` |
| `city_name` | string | From `CityConfig` |
| `market_date` | string | Contract market date when parsed |
| `ticker` | string | Kalshi contract ticker |
| `event_ticker` | string | Daily event ticker |
| `label` | string | Display label / title |
| `lo`, `hi`, `direction` | number/string | Parsed contract semantics |
| `model_mu`, `model_sigma` | number | Forecast distribution parameters |
| `model_yes_prob` | number | Settlement-aware probability |
| `yes_bid`, `yes_ask`, `no_bid`, `no_ask` | number | Dollars, not cents |
| `executable_price` | number | Price used for best-direction EV |
| `spread` | number | Best available estimate |
| `fee_per_contract` | number | Explicit fee assumption |
| `slippage_per_contract` | number | Explicit slippage assumption |
| `ev_yes`, `ev_no`, `ev_best` | number | After costs |
| `best_direction` | string | `YES`, `NO`, or `NONE` |
| `suggested_size` | integer | Zero unless eligible |
| `promotion_status` | string | PROMOTED / MONITOR / BLOCKED / unknown |
| `eligibility` | string | TRADEABLE-PAPER / MONITOR-ONLY / BLOCKED / NO EDGE |
| `reason_codes` | list[string] | Machine-readable rejection/guardrail reasons |
| `market_age_seconds` | number | From poll timestamp |
| `signal_age_seconds` | number | From signal timestamp |

### 7.2 `DashboardSnapshot`

| Field | Type | Notes |
|---|---|---|
| `generated_at` | ISO timestamp | UTC |
| `mode` | string | offline / monitor / paper |
| `config_hash` | string | Hash of dashboard config |
| `cities` | object | keyed by city code |
| `opportunities` | list | flattened rows for table display |
| `summary` | object | counts and max EV |
| `warnings` | list | cross-city warnings |

## 8. Risk controls and guardrails

1. **No live orders:** dashboard code must not import or call authenticated order placement until a future project explicitly enables it.
2. **Promotion-aware sizing:** do not show nonzero size for MONITOR/BLOCKED cities by default, even if a row has positive mathematical EV.
3. **Executable price only:** default EV uses asks plus costs, never mid-only prices.
4. **Freshness gating:** stale model signals or stale Kalshi snapshots turn all opportunities non-actionable.
5. **Calibration gating:** missing calibration artifacts convert city to MONITOR/BLOCKED depending on severity.
6. **Contract parser confidence:** if a contract threshold cannot be parsed, it is hidden from EV ranking and listed in a parser-warning panel.
7. **Auditability:** every opportunity snapshot includes model date, market poll timestamp, fee/slippage assumptions, and reason codes.
8. **Local secrets hygiene:** no API keys are required for public polling; any future credentials must be env vars and hidden from logs.

## 9. Testing plan

| Test file | Coverage |
|---|---|
| `tests/test_dashboard_opportunity_service.py` | config, signal loading, market normalization, EV/reason codes, snapshot persistence |
| `tests/test_dashboard_app_smoke.py` | app imports and initializes with mocked service |
| Existing `tests/test_kalshi_client.py` | API client behavior, parser, rate limit, retry |
| Existing `tests/test_trading.py` | EV, sizing, backtesting invariants |
| Existing `tests/test_paper_trading.py` | actual-contract probability pricing and audit behavior |
| Existing `tests/test_daily_inference.py` | cutoff and live signal generation behavior |
| Existing `tests/test_data_sla.py` | 7am ET availability guardrails |

Minimum commands before merging:

```bash
python -m pytest tests/test_dashboard_opportunity_service.py tests/test_kalshi_client.py tests/test_trading.py tests/test_paper_trading.py tests/test_data_sla.py
python scripts/run_ev_dashboard.py --mode offline --once
```

## 10. Milestone estimate

| Milestone | Duration | Output |
|---|---:|---|
| A. Config and dataclasses | 0.5-1 day | service skeleton and config validation |
| B. Signal/health ingestion | 1 day | city dashboard states without live markets |
| C. Market polling/parser | 1-2 days | normalized live/cached market table |
| D. Opportunity engine | 1-2 days | cost-aware EV snapshot JSON |
| E. Streamlit UI | 1-2 days | local interactive dashboard |
| F. Audit/runbook | 1 day | paper workflow and operating docs |
| G. End-to-end validation | 1 day | mocked integration test and launch validation |

Expected total: **6.5-10 development days** for a robust local monitor/paper dashboard.

## 11. Open questions before implementation

1. Should `dashboard.yaml` allow ATL/AUS and expansion cities behind a feature flag, or should this project strictly hard-code NYC/CHI/PHL in config defaults only?
2. Which Kalshi quote should be authoritative when `yes_ask`, `no_bid`, and order-book top of book disagree?
3. Should the first version require an existing daily inference signal, or may it run inference on demand from the dashboard?
4. What exact promotion-status source should be canonical when `results/<city>/strategy.json`, promotion reports, and baseline ledger disagree?
5. Should Streamlit be mandatory, or should the service be UI-framework agnostic with Streamlit as only one frontend?

## 12. Recommended first implementation slice

Implement the dashboard in this order:

1. `src/dashboard/opportunity_service.py` with mocked market tests and local signal loading.
2. Snapshot JSON output with no UI.
3. Streamlit table around the service.
4. Paper-mode audit actions.
5. Order-book depth and richer diagnostics.

This sequence creates a useful local CLI-generated opportunity feed before investing in UI polish, and it preserves the repository’s core guardrails: calibrated probabilities, contract-aligned bucketization, cutoff safety, conservative costs, and honest no-trade decisions.
