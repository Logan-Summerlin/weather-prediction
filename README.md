# weather-prediction

Probabilistic daily temperature forecasting and Kalshi contract trading pipeline.

## Repository map

- Core code: `src/`
- Pipeline entrypoints: `scripts/`
- Tests: `tests/`
- Technical docs: `docs/`

## Local EV dashboard quickstart

Local read-only dashboard for NYC/CHI/PHL Kalshi opportunity monitoring
(see `docs/02_realtime_ev_dashboard_plan.md` and `docs/03_ev_dashboard_runbook.md`):

```bash
pip install streamlit  # plus the repo's usual scientific deps

# Offline demo from bundled fixtures (no network)
python scripts/run_ev_dashboard.py --mode offline

# Live read-only monitoring of open Kalshi markets
python scripts/run_ev_dashboard.py --mode monitor

# Headless snapshot only (no Streamlit required)
python scripts/run_ev_dashboard.py --mode offline --once
```

The dashboard never places orders; `paper` mode writes paper-trade audit
logs through the existing harness. Snapshots persist under `results/dashboard/`.

## Canonical docs

- Documentation index: `docs/00_canonical_docs_index.md`
- NYC project plan: `Project_Plan`
