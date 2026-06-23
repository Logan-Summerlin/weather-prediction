#!/usr/bin/env python3
"""
Real-price trading-strategy sweep (Phase 3.1 / 3.2).

Refits Kalshi trading-strategy parameters on **real** pre-settlement prices and
emits a per-city promotion decision.  Drives ``trading.generate_strategy_grid``
over the out-of-sample rows of ``results/<city>/unified_predictions.csv``,
joined to the real ``bid_cents`` / ``ask_cents`` quotes from
``data/kalshi_presettlement_<city>.csv`` for cost realism.  The grid is fit on
an earlier chronological slice and the single best configuration is scored once
on an untouched later holdout.

Per-city outputs land at ``results/<city>/strategy.json`` (read back by
``src.live_trading.LiveTradingHarness``) plus a sweep table under
``results/<city>/trading/``.

Promotion (Phase 3.2): PROMOTED requires, on the real-price holdout, model
Brier < market Brier, P&L > 0, Sharpe >= 1.0, >= 50 trades, and drawdown
>= -30%.  Otherwise the city is MONITOR — never tune thresholds to manufacture
EV.

Usage:
    python scripts/run_real_strategy_sweep.py --city chi
    python scripts/run_real_strategy_sweep.py --all
    python scripts/run_real_strategy_sweep.py --city phl --val-frac 0.4
    python scripts/run_real_strategy_sweep.py --city chi --val-start 2025-09-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy_selection import (  # noqa: E402
    _metrics_summary,
    run_city_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_real_strategy_sweep")

# Cities that have both unified predictions and presettlement quotes today.
DEFAULT_CITIES = ["chi", "phl"]


def _has_inputs(city_code: str) -> bool:
    from src.city_config import get_city_config

    cfg = get_city_config(city_code)
    upath = Path(cfg.results_dir) / "unified_predictions.csv"
    ppath = PROJECT_ROOT / "data" / f"kalshi_presettlement_{city_code}.csv"
    return upath.is_file() and ppath.is_file()


def run_one(city_code: str, args) -> Optional[dict]:
    if not _has_inputs(city_code):
        logger.warning(
            "Skipping %s: missing unified_predictions.csv or presettlement CSV.",
            city_code,
        )
        return None

    logger.info("=" * 70)
    logger.info("Real-price strategy sweep — %s", city_code.upper())
    logger.info("=" * 70)

    result = run_city_sweep(
        city_code,
        model_col=args.model_col,
        val_frac=args.val_frac,
        val_start=args.val_start,
        write=True,
    )

    strat = result["strategy"]
    hm = result["holdout_metrics"]
    tm = result["train_metrics"]
    promo = result["promotion"]

    # Persist the full sweep table for auditing.
    cfg_results = Path(result["strategy_json_path"]).parent
    trading_dir = cfg_results / "trading"
    trading_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["sweep_rows"]).to_csv(
        trading_dir / "strategy_sweep.csv", index=False
    )
    with open(trading_dir / "strategy_sweep_holdout.json", "w") as f:
        json.dump(
            {
                "city_code": city_code,
                "model_col": result["model_col"],
                "selected_strategy": strat.name,
                "train_metrics": _metrics_summary(tm),
                "holdout_metrics": _metrics_summary(hm),
                "promotion": promo,
            },
            f, indent=2, default=str,
        )

    logger.info(
        "%s: selected %s | model_col=%s", city_code.upper(),
        strat.name, result["model_col"],
    )
    logger.info(
        "  TRAIN : pnl=$%.2f sharpe=%.2f trades=%d",
        tm["total_pnl"], tm["sharpe_ratio"], tm["n_trades"],
    )
    logger.info(
        "  HOLDOUT: pnl=$%.2f sharpe=%.2f trades=%d dd=%.1f%% "
        "model_brier=%.4f market_brier=%.4f edge=%+.4f",
        hm["total_pnl"], hm["sharpe_ratio"], hm["n_trades"],
        hm["max_drawdown_pct"], hm["model_brier"], hm["market_brier"],
        hm["brier_edge"],
    )
    logger.info("  PROMOTION: %s (failed: %s)", promo["status"], promo["failed_checks"])
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", help="City code (chi, phl, ...)")
    parser.add_argument("--all", action="store_true", help="Run all default cities")
    parser.add_argument(
        "--model-col", default=None,
        help="Model probability column (default: lowest-Brier variant).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.4,
        help="Holdout fraction of OOS dates (most recent). Default 0.4.",
    )
    parser.add_argument(
        "--val-start", default=None,
        help="Explicit holdout start date (YYYY-MM-DD); overrides --val-frac.",
    )
    args = parser.parse_args(argv)

    if args.all:
        cities = DEFAULT_CITIES
    elif args.city:
        cities = [args.city.strip().lower()]
    else:
        parser.error("Provide --city <code> or --all")
        return 2

    summary = []
    for city in cities:
        try:
            result = run_one(city, args)
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.exception("Sweep failed for %s: %s", city, exc)
            continue
        if result is not None:
            summary.append((city, result["promotion"]["status"],
                            result["holdout_metrics"]))

    if summary:
        print("\n" + "=" * 78)
        print(f"{'City':<8}{'Status':<12}{'Holdout P&L':>14}{'Sharpe':>9}"
              f"{'Trades':>9}{'Brier edge':>12}")
        print("-" * 78)
        for city, status, hm in summary:
            print(f"{city.upper():<8}{status:<12}{hm['total_pnl']:>13.2f}"
                  f"{hm['sharpe_ratio']:>9.2f}{hm['n_trades']:>9d}"
                  f"{hm['brier_edge']:>+12.4f}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
