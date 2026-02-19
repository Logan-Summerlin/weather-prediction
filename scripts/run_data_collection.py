#!/usr/bin/env python3
"""
Unified Multi-City Station Data Collection Script.

Downloads GHCN-Daily .dly files for all stations in the specified city's
configuration, parses them into wide-format CSVs (TMAX, TMIN per day),
and saves the output to data/<city>/raw/.

Uses the same download/parse pipeline as the NYC data collection
(src/data_collection.py) but with city-specific station config.

Replaces the per-city scripts:
  - run_chi_data_collection.py
  - run_phl_data_collection.py
  - run_atl_data_collection.py
  - run_aus_data_collection.py

Usage:
    python scripts/run_data_collection.py --city chi
    python scripts/run_data_collection.py --city phl
    python scripts/run_data_collection.py --city atl
    python scripts/run_data_collection.py --city aus
"""

import argparse
import importlib
import os
import sys
import logging

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, ensure_city_dirs
from src.data_collection import process_station

# ---------------------------------------------------------------------------
# City code -> config module mapping
# ---------------------------------------------------------------------------
CITY_CONFIG_MODULES = {
    "chi": "config_chicago",
    "phl": "config_philadelphia",
    "atl": "config_atlanta",
    "aus": "config_austin",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Download and process GHCN-Daily data for a specified city."""
    parser = argparse.ArgumentParser(
        description="Download GHCN-Daily station data for a city."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=sorted(CITY_CONFIG_MODULES.keys()),
        help="City code (chi, phl, atl, aus)",
    )
    args = parser.parse_args()
    city_code = args.city

    # Load city configs
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    city_config = importlib.import_module(CITY_CONFIG_MODULES[city_code])

    raw_dir = os.path.join(cfg.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    all_stations = city_config.ALL_STATIONS
    total = len(all_stations)

    logger.info("=" * 60)
    logger.info("%s GHCN-Daily Data Collection", cfg.city_name)
    logger.info("=" * 60)
    logger.info("Date range: %s to %s", city_config.START_DATE, city_config.END_DATE)
    logger.info("Target station: %s (%s)", city_config.TARGET_STATION,
                city_config.ALL_STATIONS.get(city_config.TARGET_STATION, ""))
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
                start_date=city_config.START_DATE,
                end_date=city_config.END_DATE,
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
    logger.info("Next step: run scripts/run_preprocessing.py --city %s", city_code)


if __name__ == "__main__":
    main()
