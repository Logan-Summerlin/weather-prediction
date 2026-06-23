#!/usr/bin/env python3
"""
Paper-trading loop (Phase 3.3).

Reads a daily signal (``results/<city>/live/signals_<date>.json``), joins it to
Kalshi market prices, runs the EV/Kelly gate through the read-only
``LiveTradingHarness`` (paper mode only — no authenticated order placement),
settles against the realised TMAX, and writes a per-day audit log under
``results/<city>/trading/``.

Market-price sources:
  * ``--source presettlement`` (default) — historical replay from
    ``data/kalshi_presettlement_<city>.csv`` (no network).
  * ``--source live`` — one read-only ``KalshiClient.get_markets`` call.

A multi-day historical replay (e.g. a losing-week stress test) is available via
``--replay-start`` / ``--replay-end``.

Usage:
    python scripts/run_paper_trading.py --city chi --date 2025-06-10 \
        --actual-tmax 88.0
    python scripts/run_paper_trading.py --city chi --date 2025-06-10 --source live
    python scripts/run_paper_trading.py --city chi --replay-start 2025-01-01 \
        --replay-end 2025-01-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.daily_inference import load_signal, signals_path  # noqa: E402
from src.live_trading import LiveTradingHarness, load_city_strategy  # noqa: E402
from src.paper_trading import (  # noqa: E402
    build_market_prediction,
    contracts_from_live,
    contracts_from_presettlement,
    replay_period,
    run_paper_cycle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_paper_trading")


def _run_single_day(args, city: str) -> int:
    path = args.signal or signals_path(city, args.date)
    if not Path(path).is_file():
        logger.error(
            "No signal at %s — run scripts/run_daily_inference.py first.", path
        )
        return 1
    signal = load_signal(path)
    if signal.get("status") != "OK":
        logger.error(
            "Signal status is %s (kill switch) — no trading for %s/%s.",
            signal.get("status"), city, args.date,
        )
        return 1

    mu = float(signal.get("mu") or 0.0)
    sigma = float(signal.get("sigma") or 1.0)

    if args.source == "live":
        contracts = contracts_from_live(city)
    else:
        contracts = contracts_from_presettlement(city, args.date)

    if not contracts:
        logger.warning("No market contracts found for %s/%s.", city, args.date)

    prediction = build_market_prediction(
        city, args.date, mu, sigma, contracts,
        model_name=signal.get("model_name", ""),
    )

    harness = LiveTradingHarness(city, mode="paper",
                                 strategy=load_city_strategy(city))
    summary = run_paper_cycle(
        harness, prediction, contracts,
        actual_tmax=args.actual_tmax, save_audit=not args.no_audit,
    )

    print(f"\n{city.upper()} {args.date}  signals={summary['n_signals']} "
          f"day_pnl=${summary['day_pnl']:.2f} "
          f"kill_switch={summary['kill_switch_active']}")
    for t in summary["trades"]:
        print(f"  {t['direction']:<3} {t['bucket_label']:<14} "
              f"size={t['size']:<3} @ {t['market_price']:.3f} EV={t['ev']:+.4f}")
    if summary["audit_path"]:
        print(f"  audit -> {summary['audit_path']}")
    return 0


def _run_replay(args, city: str) -> int:
    result = replay_period(
        city, args.replay_start, args.replay_end,
        strategy=load_city_strategy(city), save_audit=not args.no_audit,
    )
    print(f"\n{city.upper()} REPLAY {args.replay_start} -> {args.replay_end}")
    print(f"  days={result['n_days']} trades={result['n_trades']} "
          f"settled={result['n_settled']} total_pnl=${result['total_pnl']:.2f}")
    print(f"  kill_switch_active={result['kill_switch_active']} "
          f"({result['kill_switch_reason']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {k: v for k, v in result.items() if k != "days"},
                f, indent=2, default=str,
            )
        print(f"  summary -> {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City code")
    parser.add_argument("--date", default=None, help="Market date YYYY-MM-DD")
    parser.add_argument("--signal", default=None, help="Path to a signals JSON.")
    parser.add_argument(
        "--source", choices=["presettlement", "live"], default="presettlement",
        help="Market-price source (default: presettlement replay, no network).",
    )
    parser.add_argument("--actual-tmax", type=float, default=None,
                        help="Realised TMAX to settle the day's trades.")
    parser.add_argument("--no-audit", action="store_true",
                        help="Do not write the audit log.")
    parser.add_argument("--replay-start", default=None, help="Replay start date.")
    parser.add_argument("--replay-end", default=None, help="Replay end date.")
    parser.add_argument("--out", default=None, help="Replay summary output path.")
    args = parser.parse_args(argv)

    city = args.city.strip().lower()

    if args.replay_start and args.replay_end:
        return _run_replay(args, city)
    if args.date:
        return _run_single_day(args, city)
    parser.error("Provide --date <YYYY-MM-DD> or --replay-start/--replay-end")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
