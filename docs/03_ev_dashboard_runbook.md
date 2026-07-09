# EV Dashboard Runbook: Daily Operation, Troubleshooting, Incident Response

> Status: Active reference  
> Scope: local real-time positive-EV dashboard (`docs/02_realtime_ev_dashboard_plan.md`)  
> Hard rule: the dashboard is read-only against Kalshi. There is no code path
> that places authenticated orders; `live` mode is rejected at config load.

## 1. Components

| Piece | Path |
|---|---|
| Opportunity service (no UI deps) | `src/dashboard/opportunity_service.py` |
| Streamlit app | `src/dashboard/app.py` |
| Launcher CLI | `scripts/run_ev_dashboard.py` |
| Settings | `config/dashboard.yaml` (env overrides: `EV_DASHBOARD_MODE`, `EV_DASHBOARD_CITIES`, `EV_DASHBOARD_CONFIG`, `EV_DASHBOARD_EXPORT_DIR`, `EV_DASHBOARD_FIXTURE_DIR`) |
| Offline fixtures | `data/fixtures/dashboard/` (synthetic; demo only) |
| Snapshots | `results/dashboard/opportunities_<ts>.json` + `latest_opportunities.json` |
| Raw market cache | `results/dashboard/raw_kalshi/` (last-good reuse on poll failure) |
| Paper audits | `results/<city>/trading/trading_audit_<date>.json` (existing convention) |

## 2. Daily operation

1. **Generate today's cutoff-safe signals** (must respect the 7am ET cutoff —
   the signal generator trips its own kill switch on stale inputs):

   ```bash
   python scripts/run_daily_inference.py --city nyc
   python scripts/run_daily_inference.py --city chi
   python scripts/run_daily_inference.py --city phl
   ```

2. **Launch the dashboard** (default monitor mode, public polling only):

   ```bash
   python scripts/run_ev_dashboard.py --mode monitor
   ```

   Open http://localhost:8501. For paper-trade audit logging use
   `--mode paper`; for a no-network demo use `--mode offline`.

3. **Headless snapshot** (cron-friendly; no Streamlit required):

   ```bash
   python scripts/run_ev_dashboard.py --mode monitor --once
   ```

## 3. Reading the dashboard honestly

- **TRADEABLE-PAPER (green):** city PROMOTED, signal fresh, market fresh, EV
  clears `max(config.min_ev, strategy.ev_threshold)` at executable ask +
  curved Kalshi fee + slippage buffer. Sized with capped fractional Kelly.
- **MONITOR-ONLY (yellow):** positive *mathematical* EV but the city has not
  passed market-beating promotion. `suggested_size` is always 0 (the
  `show_monitor_sizing` dev flag exists but must not be enabled routinely).
- **BLOCKED (red):** missing/stale signal, kill switch, or unparseable
  contract. The city card shows the exact refresh command.
- **NO EDGE (gray):** EV does not clear costs/thresholds.
- **STALE MARKET (purple):** last-good cached quotes in use; nothing is
  actionable regardless of EV.

Mid prices are shown for context only; EV is never computed from mids.

## 4. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| City BLOCKED, `signal_missing` | No `results/<city>/live/signals_*.json` | Run the refresh command shown on the city card. |
| City BLOCKED, `signal_stale` | Signal date != today (ET) | Re-run daily inference after the 7am ET cutoff. Never backfill with post-cutoff data. |
| City BLOCKED, `kill_switch_active` | Daily inference tripped the freshness kill switch | Fix the upstream feed (`kill_switch_reasons` in the signal file), then re-run inference. Do not hand-edit signals. |
| STALE MARKET everywhere | Kalshi polling failed | Check network / `results/dashboard/raw_kalshi/`; the dashboard keeps serving last-good quotes but disables actions. Retry with "Refresh now". |
| Contracts in "Parser warnings" | Threshold text unparseable | Excluded from EV ranking by design. Extend `parse_market_threshold` with a fixture-backed test before trusting them. |
| Nothing is TRADEABLE-PAPER | Cities are MONITOR, or no edge after costs | Expected: this is the honest default. Do not lower thresholds to force activity. |

## 5. Incident response

1. **Suspected bad pricing/probabilities:** stop using the dashboard, keep the
   snapshot (`results/dashboard/latest_opportunities.json` embeds model
   inputs, quotes, fee/slippage assumptions, and reason codes), and reconcile
   against `tests/test_dashboard_opportunity_service.py` expectations.
2. **Calibration/model drift:** treat as a promotion problem, not a dashboard
   problem — rerun promotion evaluation; the city drops out of
   TRADEABLE-PAPER automatically once `strategy.json` demotes it.
3. **Anything hinting at order placement:** there is none by design; any
   diff adding authenticated Kalshi calls to `src/dashboard/` must be
   rejected in review.

## 6. Retention

Snapshots and raw poll caches are pruned automatically to
`snapshot_retention` / `raw_cache_retention` (config). Paper audits under
`results/<city>/trading/` follow the existing repository conventions and are
never pruned by the dashboard.

## 7. Validation commands

```bash
python -m pytest tests/test_dashboard_opportunity_service.py \
  tests/test_dashboard_app_smoke.py tests/test_kalshi_client.py \
  tests/test_trading.py tests/test_paper_trading.py tests/test_data_sla.py
python scripts/run_ev_dashboard.py --mode offline --once
```
