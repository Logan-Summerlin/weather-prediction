#!/usr/bin/env python3
"""
Daily cutoff-safe inference for one city-day (Phase 3.3).

Validates every operational inference feature against the 7am-ET cutoff
manifest (kill switch on a critical staleness/leakage violation), runs the
promoted model's calibrated forecast, converts it to contract-bucket
probabilities through the verified settlement semantics, and writes
``results/<city>/live/signals_<date>.json`` for the paper-trading loop and
dashboard.

Freshness inputs:
  * ``--available-timestamps <json>`` — a ``{feature: ISO8601|null}`` map of the
    freshest available record per cutoff feature (live/operational use).
  * default — assume each required source is fresh at its latest cutoff-safe
    instant (historical backfill/replay only).
  * ``--stale-feature <name>`` — force one feature's freshest record to a stale
    value, to exercise the kill switch.

Usage:
    python scripts/run_daily_inference.py --city chi --date 2025-06-10
    python scripts/run_daily_inference.py --city phl --date 2025-06-10 \
        --available-timestamps results/phl/live/freshness_2025-06-10.json
    python scripts/run_daily_inference.py --city chi --date 2025-06-10 \
        --stale-feature mos_tmax_morning   # demonstrates the kill switch
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.daily_inference import (  # noqa: E402
    DEFAULT_REQUIRED_FEATURES,
    assumed_fresh_timestamps,
    load_available_timestamps,
    run_daily_inference,
    signals_path,
)
from src.data_sla import cutoff_instant_utc, get_cutoff_spec  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_daily_inference")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City code (chi, phl, ...)")
    parser.add_argument("--date", required=True, help="Market date YYYY-MM-DD")
    parser.add_argument(
        "--available-timestamps", default=None,
        help="JSON file mapping cutoff feature -> freshest record timestamp.",
    )
    parser.add_argument(
        "--stale-feature", default=None,
        help="Force a feature stale to trip the kill switch (demo/testing).",
    )
    parser.add_argument("--mu", type=float, default=None, help="Override forecast mu.")
    parser.add_argument("--sigma", type=float, default=None, help="Override forecast sigma.")
    parser.add_argument(
        "--no-write", action="store_true", help="Do not persist the signal JSON.",
    )
    args = parser.parse_args(argv)

    city = args.city.strip().lower()
    date = args.date.strip()
    require = list(DEFAULT_REQUIRED_FEATURES)

    if args.available_timestamps:
        timestamps = load_available_timestamps(args.available_timestamps)
    else:
        timestamps = assumed_fresh_timestamps(date, require)

    if args.stale_feature:
        spec = get_cutoff_spec(args.stale_feature)
        cutoff = cutoff_instant_utc(date)
        # Push the freshest record well past its staleness tolerance.
        timestamps[args.stale_feature] = cutoff - timedelta(
            hours=spec.max_staleness_hours + 24.0
        )
        if args.stale_feature not in require:
            require.append(args.stale_feature)

    signal = run_daily_inference(
        city, date,
        available_timestamps=timestamps,
        require_features=require,
        mu=args.mu, sigma=args.sigma,
        write=not args.no_write,
    )

    if signal.status == "KILL_SWITCH":
        logger.error(
            "KILL SWITCH for %s/%s — no signal produced. Reasons:", city, date
        )
        for r in signal.kill_switch_reasons:
            logger.error("  - %s", r)
        return 1

    print(f"\n{city.upper()} {date}  status={signal.status} "
          f"promotion={signal.promotion_status}")
    print(f"  mu={signal.mu:.2f}  sigma={signal.sigma:.2f}  model={signal.model_name}")
    top = sorted(
        zip(signal.bucket_labels, signal.bucket_probs),
        key=lambda kv: kv[1], reverse=True,
    )[:5]
    print("  top buckets: " + ", ".join(f"{lbl}={p:.3f}" for lbl, p in top))
    if not args.no_write:
        print(f"  signal -> {signals_path(city, date)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
