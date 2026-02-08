#!/usr/bin/env python3
"""
Data Collection Script for ALL Stations (Original 15 + Expanded 50).

Downloads and parses NOAA GHCN-Daily .dly files for all stations defined
in both config.py (15 stations) and config_expanded.py (51 stations).
The merged set contains all unique stations across both configurations.

Features:
- Merges stations from both configs (deduplicates automatically)
- Retries failed downloads with exponential backoff
- Reports data completeness for each station
- Flags stations with < 80% TMAX completeness
- Saves a quality summary CSV

Usage:
    python run_collect_all_stations.py
"""

import os
import sys
import time
import logging
from datetime import date, datetime

import pandas as pd
import requests

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import config_expanded
from src.data_collection import (
    download_dly_file,
    parse_dly_file,
    pivot_station_data,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_all_stations")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 5
COMPLETENESS_THRESHOLD = 0.80


def merge_all_stations() -> dict:
    """Merge stations from config.py and config_expanded.py (deduplicated).

    Returns
    -------
    dict
        Mapping of station_id -> description for all unique stations.
    """
    merged = {}

    # Add target station
    merged[config.TARGET_STATION] = "Central Park, NYC (Target)"

    # Add original 14 surrounding stations from config.py
    for sid, desc in config.SURROUNDING_STATIONS.items():
        merged[sid] = desc

    # Add expanded surrounding stations from config_expanded.py
    for sid, desc in config_expanded.SURROUNDING_STATIONS.items():
        if sid not in merged:
            merged[sid] = desc

    return merged


def download_with_retries(station_id: str, raw_dir: str,
                          max_retries: int = MAX_RETRIES) -> str:
    """Download a .dly file with exponential backoff retries.

    Parameters
    ----------
    station_id : str
        GHCN station identifier.
    raw_dir : str
        Directory to save .dly files.
    max_retries : int
        Maximum number of retry attempts.

    Returns
    -------
    str
        Path to the downloaded .dly file.

    Raises
    ------
    Exception
        If all retries are exhausted.
    """
    dly_path = os.path.join(raw_dir, f"{station_id}.dly")

    # Use cached file if it exists
    if os.path.exists(dly_path):
        file_size = os.path.getsize(dly_path)
        if file_size > 0:
            logger.info("  Using cached .dly file for %s (%.1f KB)",
                        station_id, file_size / 1024)
            return dly_path
        else:
            logger.warning("  Cached .dly file for %s is empty, re-downloading",
                           station_id)
            os.remove(dly_path)

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            path = download_dly_file(station_id, raw_dir, timeout=180)
            return path
        except (requests.exceptions.RequestException, IOError) as e:
            last_exception = e
            backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "  Download attempt %d/%d failed for %s: %s. "
                "Retrying in %ds...",
                attempt, max_retries, station_id, e, backoff,
            )
            if attempt < max_retries:
                time.sleep(backoff)

    raise last_exception


def process_station_with_retries(station_id: str, raw_dir: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
    """Download (with retries), parse, and save a single station's data.

    Parameters
    ----------
    station_id : str
        GHCN station identifier.
    raw_dir : str
        Directory for .dly files and output CSVs.
    start_date : str
        Start date (inclusive) in "YYYY-MM-DD" format.
    end_date : str
        End date (inclusive) in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with TMAX, TMIN per date.
    """
    # Step 1: Ensure .dly file exists (download with retries)
    dly_path = download_with_retries(station_id, raw_dir)

    # Step 2: Parse .dly file
    logger.info("  Parsing %s.dly for date range %s to %s ...",
                station_id, start_date, end_date)
    df_long = parse_dly_file(dly_path, start_date, end_date)

    if df_long.empty:
        logger.warning("  No data found for %s in range %s to %s",
                        station_id, start_date, end_date)
        return pd.DataFrame()

    # Step 3: Pivot to wide format
    df_wide = pivot_station_data(df_long)

    # Step 4: Save as CSV
    csv_path = os.path.join(raw_dir, f"{station_id}.csv")
    df_wide.to_csv(csv_path)
    logger.info("  Saved %s: %d rows, columns: %s",
                csv_path, len(df_wide), list(df_wide.columns))

    return df_wide


def compute_completeness(df: pd.DataFrame, start_date: str,
                         end_date: str) -> dict:
    """Compute data completeness statistics for a station.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame indexed by date.
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    dict
        Completeness statistics.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1

    if df.empty:
        return {
            "total_expected_days": total_days,
            "actual_rows": 0,
            "tmax_count": 0,
            "tmin_count": 0,
            "tmax_completeness": 0.0,
            "tmin_completeness": 0.0,
            "first_date": None,
            "last_date": None,
        }

    tmax_count = df["TMAX"].notna().sum() if "TMAX" in df.columns else 0
    tmin_count = df["TMIN"].notna().sum() if "TMIN" in df.columns else 0

    # Get date range
    dates = df.index
    first_date = min(dates) if len(dates) > 0 else None
    last_date = max(dates) if len(dates) > 0 else None

    return {
        "total_expected_days": total_days,
        "actual_rows": len(df),
        "tmax_count": int(tmax_count),
        "tmin_count": int(tmin_count),
        "tmax_completeness": tmax_count / total_days if total_days > 0 else 0.0,
        "tmin_completeness": tmin_count / total_days if total_days > 0 else 0.0,
        "first_date": str(first_date) if first_date else None,
        "last_date": str(last_date) if last_date else None,
    }


def main():
    """Main collection pipeline for all stations."""
    start_time = time.time()

    # Merge all stations from both configs
    all_stations = merge_all_stations()
    total = len(all_stations)

    logger.info("=" * 70)
    logger.info("NOAA GHCN-Daily Data Collection for ALL Stations")
    logger.info("=" * 70)
    logger.info("Date range: %s to %s", config.START_DATE, config.END_DATE)
    logger.info("Total stations: %d", total)
    logger.info("Target station: %s", config.TARGET_STATION)
    logger.info("Output directory: %s", config.RAW_DATA_DIR)
    logger.info("=" * 70)

    raw_dir = config.RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)

    # Delete existing CSV files to force re-parse with new date range
    csv_count = 0
    for f in os.listdir(raw_dir):
        if f.endswith(".csv"):
            os.remove(os.path.join(raw_dir, f))
            csv_count += 1
    if csv_count > 0:
        logger.info("Deleted %d existing CSV files for re-parsing", csv_count)

    # Process each station
    results = {}
    failed_stations = []
    quality_records = []

    for i, (station_id, description) in enumerate(all_stations.items(), 1):
        logger.info("-" * 70)
        logger.info("[%d/%d] %s — %s", i, total, station_id, description)
        logger.info("-" * 70)

        try:
            df = process_station_with_retries(
                station_id, raw_dir, config.START_DATE, config.END_DATE
            )

            if not df.empty:
                results[station_id] = df
                stats = compute_completeness(df, config.START_DATE, config.END_DATE)
                stats["station_id"] = station_id
                stats["description"] = description
                stats["status"] = "OK"
                quality_records.append(stats)

                logger.info(
                    "  TMAX: %d/%d days (%.1f%%), TMIN: %d/%d days (%.1f%%)",
                    stats["tmax_count"], stats["total_expected_days"],
                    stats["tmax_completeness"] * 100,
                    stats["tmin_count"], stats["total_expected_days"],
                    stats["tmin_completeness"] * 100,
                )
            else:
                failed_stations.append((station_id, description, "Empty dataset"))
                quality_records.append({
                    "station_id": station_id,
                    "description": description,
                    "status": "EMPTY",
                    "total_expected_days": 0,
                    "actual_rows": 0,
                    "tmax_count": 0,
                    "tmin_count": 0,
                    "tmax_completeness": 0.0,
                    "tmin_completeness": 0.0,
                    "first_date": None,
                    "last_date": None,
                })

        except Exception as e:
            logger.error("  FAILED for %s: %s", station_id, e)
            failed_stations.append((station_id, description, str(e)))
            quality_records.append({
                "station_id": station_id,
                "description": description,
                "status": f"FAILED: {e}",
                "total_expected_days": 0,
                "actual_rows": 0,
                "tmax_count": 0,
                "tmin_count": 0,
                "tmax_completeness": 0.0,
                "tmin_completeness": 0.0,
                "first_date": None,
                "last_date": None,
            })

    elapsed = time.time() - start_time

    # Build quality summary DataFrame
    quality_df = pd.DataFrame(quality_records)

    # Save quality summary
    quality_path = os.path.join(config.DATA_DIR, "data_quality_summary.csv")
    quality_df.to_csv(quality_path, index=False)
    logger.info("Saved data quality summary to %s", quality_path)

    # Print final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info("Total stations attempted: %d", total)
    logger.info("Successful: %d", len(results))
    logger.info("Failed: %d", len(failed_stations))
    logger.info("Elapsed time: %.1f minutes", elapsed / 60)
    logger.info("")

    # Check target station
    if config.TARGET_STATION in results:
        target_df = results[config.TARGET_STATION]
        target_stats = compute_completeness(
            target_df, config.START_DATE, config.END_DATE
        )
        logger.info("TARGET STATION (%s - Central Park):", config.TARGET_STATION)
        logger.info("  Date range in data: %s to %s",
                     target_stats["first_date"], target_stats["last_date"])
        logger.info("  TMAX: %d days (%.1f%% complete)",
                     target_stats["tmax_count"],
                     target_stats["tmax_completeness"] * 100)
        logger.info("  TMIN: %d days (%.1f%% complete)",
                     target_stats["tmin_count"],
                     target_stats["tmin_completeness"] * 100)
    else:
        logger.error("TARGET STATION (%s) FAILED - this is critical!",
                      config.TARGET_STATION)

    # Flag stations below completeness threshold
    logger.info("")
    logger.info("STATIONS BELOW %.0f%% TMAX COMPLETENESS:",
                COMPLETENESS_THRESHOLD * 100)
    low_quality = quality_df[
        (quality_df["tmax_completeness"] < COMPLETENESS_THRESHOLD) &
        (quality_df["status"] == "OK")
    ]
    if len(low_quality) > 0:
        for _, row in low_quality.iterrows():
            logger.info("  %s (%s): %.1f%% TMAX completeness",
                         row["station_id"], row["description"],
                         row["tmax_completeness"] * 100)
    else:
        logger.info("  None - all successful stations meet the threshold.")

    # Report failed stations
    if failed_stations:
        logger.info("")
        logger.info("FAILED STATIONS:")
        for sid, desc, reason in failed_stations:
            logger.info("  %s (%s): %s", sid, desc, reason)

    # Per-station record counts
    logger.info("")
    logger.info("PER-STATION RECORD COUNTS:")
    logger.info("%-15s %-40s %8s %8s %8s %8s",
                "Station ID", "Description", "Rows", "TMAX", "TMIN", "TMAX%")
    logger.info("-" * 95)
    for _, row in quality_df.sort_values("station_id").iterrows():
        if row["status"] == "OK":
            logger.info(
                "%-15s %-40s %8d %8d %8d %7.1f%%",
                row["station_id"],
                row["description"][:40],
                row["actual_rows"],
                row["tmax_count"],
                row["tmin_count"],
                row["tmax_completeness"] * 100,
            )

    # Verify CSVs exist
    logger.info("")
    logger.info("CSV FILE VERIFICATION:")
    expected_csvs = set(all_stations.keys())
    found_csvs = set()
    for f in os.listdir(raw_dir):
        if f.endswith(".csv"):
            sid = f.replace(".csv", "")
            found_csvs.add(sid)

    missing_csvs = expected_csvs - found_csvs
    if missing_csvs:
        logger.warning("  Missing CSV files for %d stations: %s",
                        len(missing_csvs), sorted(missing_csvs))
    else:
        logger.info("  All %d station CSV files present.", len(expected_csvs))

    return results, quality_df


if __name__ == "__main__":
    results, quality_df = main()
