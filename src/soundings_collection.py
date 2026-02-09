"""
IGRA Soundings Collection.

Downloads upper-air soundings for the OKX/Upton station via Siphon/IGRA.
These soundings are available operationally for 00Z and 12Z.
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _require_siphon():
    try:
        from siphon.simplewebservice.igra2 import IGRAUpperAir  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "The 'siphon' package is required for IGRA soundings collection. "
            "Install it via pip install siphon."
        ) from exc


def _daterange(start_date: str, end_date: str) -> Iterable[datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_soundings(
    station_id: str = config.IGRA_STATION_ID,
    start_date: str = config.IGRA_START_DATE,
    end_date: str = config.IGRA_END_DATE,
    hours: Iterable[int] = (0, 12),
    output_dir: str = config.IGRA_RAW_DIR,
) -> list[str]:
    """Download IGRA soundings for the specified station and date range."""
    _require_siphon()
    from siphon.simplewebservice.igra2 import IGRAUpperAir

    os.makedirs(output_dir, exist_ok=True)
    saved_files: list[str] = []

    for day in _daterange(start_date, end_date):
        for hour in hours:
            timestamp = day.replace(hour=hour)
            output_path = os.path.join(
                output_dir,
                f"{station_id}_{timestamp:%Y%m%d%H}.csv",
            )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                saved_files.append(output_path)
                continue

            try:
                df, header = IGRAUpperAir.request_data(timestamp, station_id)
                df.to_csv(output_path, index=False)
                header_path = output_path.replace(".csv", ".header.json")
                pd.Series(header).to_json(header_path)
                saved_files.append(output_path)
                logger.info("Saved sounding %s", output_path)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning(
                    "Failed to download sounding %s %s: %s",
                    station_id, timestamp.strftime("%Y-%m-%d %H"), exc,
                )

    logger.info("Soundings collection complete: %d files", len(saved_files))
    return saved_files


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download IGRA soundings.")
    parser.add_argument("--station-id", default=config.IGRA_STATION_ID)
    parser.add_argument("--start-date", default=config.IGRA_START_DATE)
    parser.add_argument("--end-date", default=config.IGRA_END_DATE)
    parser.add_argument(
        "--hours",
        nargs="*",
        type=int,
        default=[0, 12],
        help="UTC hours to download (e.g., 0 12).",
    )
    parser.add_argument(
        "--output-dir",
        default=config.IGRA_RAW_DIR,
        help="Directory to store IGRA CSVs.",
    )
    args = parser.parse_args()

    download_soundings(
        station_id=args.station_id,
        start_date=args.start_date,
        end_date=args.end_date,
        hours=args.hours,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
