#!/usr/bin/env python
"""
Launch the local real-time positive-EV dashboard (NYC / CHI / PHL).

Usage
-----
    # Interactive Streamlit UI (default: monitor mode, public read-only Kalshi)
    python scripts/run_ev_dashboard.py --mode monitor

    # Offline demo from bundled fixtures — no network at all
    python scripts/run_ev_dashboard.py --mode offline

    # One headless refresh: build + persist a snapshot, print a summary, exit.
    # Works without Streamlit installed.
    python scripts/run_ev_dashboard.py --mode offline --once

Modes: offline | monitor | paper.  "live" is rejected by design — this
dashboard never places authenticated orders.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default=None,
                   choices=["offline", "monitor", "paper"],
                   help="Dashboard mode (default: config file value, else monitor)")
    p.add_argument("--config", default=None,
                   help="Path to dashboard YAML (default: config/dashboard.yaml)")
    p.add_argument("--cities", default=None,
                   help="Comma-separated city codes override (e.g. nyc,chi)")
    p.add_argument("--once", action="store_true",
                   help="Build and persist one snapshot headlessly, then exit")
    p.add_argument("--port", type=int, default=8501, help="Streamlit port")
    return p.parse_args()


def run_once(args) -> int:
    from src.dashboard.opportunity_service import OpportunityService, load_config

    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.cities:
        overrides["cities"] = [c for c in args.cities.split(",") if c.strip()]
    config = load_config(path=args.config, overrides=overrides)
    service = OpportunityService(config)
    snapshot = service.refresh()

    s = snapshot.summary
    print(f"\nEV dashboard snapshot ({snapshot.mode}) @ {snapshot.generated_at}")
    print(f"  cities healthy/stale/blocked: {s['n_healthy_cities']}/"
          f"{s['n_stale_cities']}/{s['n_blocked_cities']}")
    print(f"  contracts: {s['n_contracts']}  positive-EV: {s['n_positive_ev']}  "
          f"tradeable-paper: {s['n_tradeable_paper']}  "
          f"monitor-only: {s['n_monitor_only']}")
    if s.get("max_ev_after_costs") is not None:
        print(f"  max EV after costs: {100 * s['max_ev_after_costs']:.1f}c "
              f"(fee model: {s['fee_model']}, slippage {s['slippage_cents']}c)")
    print(f"  snapshot: {os.path.join(config.export_dir, 'latest_opportunities.json')}")

    top = [r for r in snapshot.opportunities if r["ev_best"] is not None][:10]
    if top:
        print("\n  Top contracts by conservative EV:")
        for r in top:
            print(f"    {r['city_code'].upper():4s} {r['ticker']:28s} "
                  f"P={r['model_yes_prob']:.3f} {r['best_direction']:>3s} "
                  f"EV={100 * r['ev_best']:+6.1f}c size={r['suggested_size']:3d} "
                  f"[{r['eligibility']}] {','.join(r['reason_codes'])}")
    for w in snapshot.warnings:
        print(f"  WARNING: {w}")
    return 0


def run_streamlit(args) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit is not installed. Install it (pip install streamlit) "
              "or use --once for a headless snapshot.", file=sys.stderr)
        return 1

    env = dict(os.environ)
    if args.mode:
        env["EV_DASHBOARD_MODE"] = args.mode
    if args.config:
        env["EV_DASHBOARD_CONFIG"] = os.path.abspath(args.config)
    if args.cities:
        env["EV_DASHBOARD_CITIES"] = args.cities

    app_path = os.path.join(REPO_ROOT, "src", "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path,
           "--server.port", str(args.port),
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    print(f"Launching dashboard: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT, env=env)


def main() -> int:
    args = parse_args()
    if args.once:
        return run_once(args)
    return run_streamlit(args)


if __name__ == "__main__":
    raise SystemExit(main())
