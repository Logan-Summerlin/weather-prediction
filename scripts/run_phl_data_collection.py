#!/usr/bin/env python3
"""
Philadelphia Station Data Collection Script.

Downloads GHCN-Daily .dly files for all stations in the Philadelphia
configuration, parses them into wide-format CSVs (TMAX, TMIN per day),
and saves the output to data/philadelphia/raw/.

Uses the same download/parse pipeline as the NYC data collection
(src/data_collection.py) but with Philadelphia-specific station config.

Usage:
    python scripts/run_phl_data_collection.py
"""

import os
import sys
import logging

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, ensure_city_dirs
from src.data_collection import process_station
import config_philadelphia as phl_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Download and process GHCN-Daily data for all Philadelphia-area stations."""
    phl = get_city_config("phl")
    ensure_city_dirs(phl)

    raw_dir = os.path.join(phl.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_stations = phl_config.ALL_STATIONS
    total = len(all_stations)

    logger.info("=" * 60)
    logger.info("Philadelphia GHCN-Daily Data Collection")
    logger.info("=" * 60)
    logger.info("Date range: %s to %s", phl_config.START_DATE, phl_config.END_DATE)
    logger.info("Target station: %s (%s)", phl_config.TARGET_STATION,
                phl_config.ALL_STATIONS.get(phl_config.TARGET_STATION, ""))
    logger.info("Total stations: %d", total)
    logger.info("Output directory: %s", raw_dir)
    logger.info("=" * 60)

    success = 0
    failed = []

    for i, (station_id, description) in enumerate(all_stations.items(), 1):
        csv_path = os.path.join(raw_dir, f"{station_id}.csv")

        # Skip if already downloaded and parsed
        if os.path.exists(csv_path):
            logger.info("[%d/%d] [CACHED] %s: %s", i, total, station_id, description)
            success += 1
            continue

        logger.info("[%d/%d] Processing %s: %s", i, total, station_id, description)

        try:
            df = process_station(
                station_id=station_id,
                raw_dir=raw_dir,
                start_date=phl_config.START_DATE,
                end_date=phl_config.END_DATE,
            )
            if not df.empty:
                success += 1
                logger.info("  -> %d rows for %s", len(df), station_id)
            else:
                logger.warning("  -> EMPTY dataset for %s -- skipping", station_id)
                failed.append(station_id)
        except Exception as e:
            logger.error("  -> FAILED %s: %s", station_id, e)
            failed.append(station_id)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Collection Complete")
    logger.info("=" * 60)
    logger.info("Success: %d / %d stations", success, total)
    if failed:
        logger.warning("Failed stations (%d): %s", len(failed), failed)
    else:
        logger.info("All stations downloaded successfully.")
    logger.info("Raw data saved to: %s", raw_dir)
    logger.info("Next step: run scripts/run_phl_preprocessing.py")


if __name__ == "__main__":
    main()
