#!/usr/bin/env python3
"""
Download real GHCN temperature data for Central Park and surrounding stations.

Downloads .dly files from NOAA GHCN bulk server, parses TMAX for the target
station (USW00094728 = Central Park) and all surrounding stations for
2022-01-01 through 2025-12-31.

Outputs:
  - data/real_central_park_tmax_2023_2025.csv  (date, actual_tmax)
  - data/raw/{station_id}.csv  (full station data)
"""

import os
import sys
import logging

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
from src.data_collection import (
    download_dly_file,
    parse_dly_file,
    pivot_station_data,
    tenths_c_to_fahrenheit,
)
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_and_parse_station(station_id, raw_dir, start_date, end_date):
    """Download .dly file and parse to wide-format CSV."""
    dly_path = os.path.join(raw_dir, f"{station_id}.dly")

    # Download if not cached
    if not os.path.exists(dly_path):
        logger.info("Downloading %s...", station_id)
        download_dly_file(station_id, raw_dir)
    else:
        logger.info("Using cached .dly file for %s", station_id)

    # Parse
    df_long = parse_dly_file(dly_path, start_date, end_date)
    if df_long.empty:
        logger.warning("No data for %s in %s to %s", station_id, start_date, end_date)
        return pd.DataFrame()

    # Pivot to wide
    df_wide = pivot_station_data(df_long)

    # Save CSV
    csv_path = os.path.join(raw_dir, f"{station_id}.csv")
    df_wide.to_csv(csv_path)
    logger.info("Saved %s: %d rows", csv_path, len(df_wide))

    return df_wide


def main():
    raw_dir = config.RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)
    data_dir = config.DATA_DIR

    # We need data from 2022 (for lag features) through 2025
    start_date = "2022-01-01"
    end_date = "2025-12-31"

    # ---- 1. Download Central Park (target station) ----
    logger.info("=" * 60)
    logger.info("Downloading Central Park (USW00094728) data...")
    logger.info("=" * 60)

    cp_df = download_and_parse_station(
        config.TARGET_STATION, raw_dir, start_date, end_date
    )

    if cp_df.empty or "TMAX" not in cp_df.columns:
        logger.error("Failed to get TMAX data for Central Park!")
        sys.exit(1)

    # Extract TMAX for 2023-2025
    tmax_series = cp_df["TMAX"].dropna()
    # Index may be datetime.date objects, convert for comparison
    from datetime import date
    tmax_2023_2025 = tmax_series[tmax_series.index >= date(2023, 1, 1)]

    # Save Central Park TMAX CSV
    cp_tmax_df = pd.DataFrame({
        "date": tmax_2023_2025.index,
        "actual_tmax": tmax_2023_2025.values,
    })
    cp_tmax_path = os.path.join(data_dir, "real_central_park_tmax_2023_2025.csv")
    cp_tmax_df.to_csv(cp_tmax_path, index=False)
    logger.info("Saved Central Park TMAX: %s (%d days)", cp_tmax_path, len(cp_tmax_df))
    logger.info("  Date range: %s to %s",
                cp_tmax_df["date"].min(), cp_tmax_df["date"].max())
    logger.info("  TMAX range: %.1f to %.1f F",
                cp_tmax_df["actual_tmax"].min(), cp_tmax_df["actual_tmax"].max())
    logger.info("  TMAX mean: %.1f F", cp_tmax_df["actual_tmax"].mean())

    # ---- 2. Download all surrounding stations ----
    logger.info("=" * 60)
    logger.info("Downloading %d surrounding stations...", len(config.SURROUNDING_STATIONS))
    logger.info("=" * 60)

    station_results = {}
    for sid, desc in config.SURROUNDING_STATIONS.items():
        logger.info("Processing %s (%s)...", sid, desc)
        try:
            df = download_and_parse_station(sid, raw_dir, start_date, end_date)
            if not df.empty:
                station_results[sid] = df
                n_tmax = df["TMAX"].notna().sum() if "TMAX" in df.columns else 0
                logger.info("  -> %d TMAX records", n_tmax)
            else:
                logger.warning("  -> EMPTY")
        except Exception as e:
            logger.error("  -> FAILED: %s", e)

    logger.info("=" * 60)
    logger.info("Download complete: %d / %d surrounding stations",
                len(station_results), len(config.SURROUNDING_STATIONS))
    logger.info("Central Park TMAX saved to %s", cp_tmax_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
