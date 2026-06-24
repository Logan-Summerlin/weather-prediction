#!/usr/bin/env python3
"""
Verify Kalshi weather contracts for portfolio expansion (Phase 4.1).

Probes the live Kalshi public API for the candidate expansion cities
({DEN, DC, MIA, LAX, HOU, PHX}), resolving each irregular series ticker,
its settlement station, bucket structure, and a 7-day liquidity sample,
then ranks the discretionary pool by spread-adjusted liquidity and writes
the recommendation artifact.

    results/expansion/contract_verification.json

No city config may be hard-coded before this artifact exists. All API
access is read-only (unauthenticated public endpoints); no order-placing
or authenticated endpoints are touched.

Usage:
    python scripts/verify_kalshi_contracts.py
    python scripts/verify_kalshi_contracts.py --top-n 3 \
        --output results/expansion/contract_verification.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.expansion_verification import (  # noqa: E402
    CANDIDATE_CITIES,
    DEFAULT_TOP_N,
    run_verification,
)
from src.kalshi_client import KalshiClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = os.path.join("results", "expansion", "contract_verification.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help="number of discretionary cities to recommend by liquidity",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="path to write the verification artifact",
    )
    args = parser.parse_args()

    client = KalshiClient()
    logger.info(
        "Verifying %d candidate cities: %s",
        len(CANDIDATE_CITIES), ", ".join(CANDIDATE_CITIES),
    )

    payload = run_verification(client, top_n=args.top_n)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["source"] = "kalshi_public_api"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Wrote %s", args.output)
    print("\n=== Kalshi contract verification ===")
    for res in payload["results"]:
        liq = res.get("liquidity", {}) or {}
        print(
            f"  {res['code']:4s} {res.get('series_ticker', '-'):14s} "
            f"{res['status']:8s} "
            f"station={res.get('settlement_station')!r} "
            f"vol={liq.get('total_volume', 0):.0f} "
            f"oi={liq.get('total_open_interest', 0):.0f} "
            f"spread_c={liq.get('mean_spread_cents')} "
            f"score={res.get('spread_adjusted_liquidity', 0):.1f}"
            + ("  [mandated]" if res.get("mandated") else "")
        )
    print(f"\nRecommended expansion set: {payload['recommended']}")
    print(f"Blocked: {payload['blocked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
