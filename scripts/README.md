# Scripts Directory Guide

This directory contains runnable entrypoints for data ingestion, modeling,
calibration, backtesting, and promotion evaluation.

## Canonical Pipeline Entrypoint

Use `run_city_pipeline.py` as the default orchestrator:

```bash
python scripts/run_city_pipeline.py --city nyc --stage all
python scripts/run_city_pipeline.py --city chi --stage all
python scripts/run_city_pipeline.py --city phl --stage benchmark
python scripts/run_city_pipeline.py --city atl --stage all --dry-run
```

Supported cities: `nyc`, `chi`, `phl`, `atl`, `aus`.

Supported stages:
- `data_collection`
- `preprocessing`
- `benchmark`
- `synthesis_calibration`
- `backtest`
- `promotion_evaluation`
- `all` (ordered full pipeline)

## Real-Time EV Dashboard

`run_ev_dashboard.py` launches the local read-only positive-EV dashboard
(NYC/CHI/PHL) from `docs/02_realtime_ev_dashboard_plan.md`:

```bash
python scripts/run_ev_dashboard.py --mode offline          # bundled fixtures, no network
python scripts/run_ev_dashboard.py --mode monitor          # public Kalshi polling, no trades
python scripts/run_ev_dashboard.py --mode paper            # + paper-trade audit logging
python scripts/run_ev_dashboard.py --mode offline --once   # headless snapshot, no Streamlit
```

Modes: `offline` | `monitor` | `paper` — `live` is rejected by design (no
authenticated order placement). Settings live in `config/dashboard.yaml`;
snapshots are written to `results/dashboard/opportunities_<ts>.json` with a
`latest_opportunities.json` copy. Daily signals come from
`run_daily_inference.py --city <city>`. Operating runbook:
`docs/03_ev_dashboard_runbook.md`.

## Stage Entrypoints (Unified)

Each stage still has a direct script for ad hoc runs:

- `run_data_collection.py`
- `run_preprocessing.py`
- `run_benchmark.py`
- `run_synthesis_calibration.py`
- `run_backtest.py`
- `run_promotion_evaluation.py`

All accept `--city {nyc,chi,phl,atl,aus}` and preserve existing artifact
locations under `data/<city>/`, `models/<city>/`, and `results/<city>/`.
NYC uses root-level `data/`, `models/`, `results/` for backward compatibility.

## Legacy Wrappers Removed

City-specific wrapper scripts (`run_<city>_<stage>.py`) were removed because
they duplicated the unified stage CLIs without adding functionality.

Use either:

- `run_city_pipeline.py --city <city> --stage <stage>`
- `run_<stage>.py --city <city>`

## Parameterized Template

`run_city_nws_kalshi_template_benchmark.py` remains available for cross-city
NWS/Kalshi benchmark evaluation and template validation.

## Experiments and Utilities

- `experiments/benchmarking/`: model training, evaluation, and cross-city comparisons.
- `experiments/trading/`: backtest and trading strategy experiments.
- `fetch_*` / `download_*`: market and external data utilities.

See `experiments/README.md` for a detailed index.

Keep these separate from production pipeline stage entrypoints to avoid
accidental coupling.
