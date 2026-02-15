#!/usr/bin/env python3
"""
Download MOS (Model Output Statistics) archive from Iowa Environmental Mesonet
for KPHL (Philadelphia International Airport).

Downloads GFS MOS and NAM MOS historical forecasts from IEM,
extracts day-ahead max temperature forecasts, and saves processed data.

Data source: https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py

Adapted from scripts/download_iem_mos.py (KNYC version).

Usage:
    python scripts/download_iem_mos_kphl.py
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MOS_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "philadelphia", "mos")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATION = "KPHL"
MODELS = ["GFS", "NAM"]
START_YEAR = 2004  # Earliest available GFS MOS data
END_YEAR = 2026    # Up to current year
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 120  # seconds


def download_mos_chunk(
    station: str,
    model: str,
    year: int,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[str]:
    """Download one year of MOS data from IEM with exponential backoff."""
    params = {
        "station": station,
        "model": model,
        "year1": year,
        "month1": 1,
        "day1": 1,
        "hour1": 0,
        "year2": year,
        "month2": 12,
        "day2": 31,
        "hour2": 23,
    }

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading %s %s year=%d (attempt %d/%d)",
                station, model, year, attempt, MAX_RETRIES,
            )
            resp = requests.get(IEM_MOS_URL, params=params, timeout=timeout)
            resp.raise_for_status()

            text = resp.text.strip()
            if not text or not text.startswith("runtime"):
                logger.warning(
                    "Empty or invalid response for %s %s %d",
                    station, model, year,
                )
                return None

            lines = text.split("\n")
            logger.info(
                "  Got %d lines (%.1f KB) for %s %s %d",
                len(lines), len(text) / 1024, station, model, year,
            )
            return text

        except (requests.RequestException, ConnectionError) as exc:
            logger.warning(
                "  Attempt %d failed: %s. Retrying in %.1fs...",
                attempt, exc, backoff,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "FAILED to download %s %s %d after %d retries",
                    station, model, year, MAX_RETRIES,
                )
                return None

    return None


def parse_mos_csv(raw_csv: str) -> pd.DataFrame:
    """Parse raw IEM MOS CSV into a DataFrame."""
    df = pd.read_csv(StringIO(raw_csv), parse_dates=["runtime", "ftime"])
    return df


def extract_tmax_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day-ahead TMAX forecasts from parsed MOS data.

    MOS N/X field convention:
    - n_x at ftime with hour == 0 (00Z) = MAX temperature (TMAX)
      for the PREVIOUS calendar day (ftime.date() - 1 day)
    - n_x at ftime with hour == 12 (12Z) = MIN temperature (TMIN)

    For each target date, we prefer the most recent 12Z runtime from
    the day before (day-ahead forecast), falling back to 00Z same day.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])

    # Filter to rows with valid n_x at 00Z ftime (TMAX values)
    mask = df["n_x"].notna() & (df["ftime"].dt.hour == 0)
    tmax_df = df.loc[mask].copy()

    if tmax_df.empty:
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])

    # The target date for a TMAX forecast is ftime.date() - 1 day
    tmax_df["target_date"] = tmax_df["ftime"].dt.date - pd.Timedelta(days=1)
    tmax_df["runtime_hour"] = tmax_df["runtime"].dt.hour
    tmax_df["runtime_date"] = tmax_df["runtime"].dt.date

    records = []
    for target_date, group in tmax_df.groupby("target_date"):
        group = group.copy()
        target_dt = pd.Timestamp(target_date)
        day_before = (target_dt - pd.Timedelta(days=1)).date()

        def _priority(row):
            rd = row["runtime_date"]
            rh = row["runtime_hour"]
            if rd == day_before and rh == 12:
                return 100
            elif rd == day_before and rh == 18:
                return 90
            elif rd == day_before and rh == 6:
                return 80
            elif rd == day_before and rh == 0:
                return 70
            elif rd == target_date and rh == 0:
                return 60
            elif rd == target_date and rh == 6:
                return 50
            else:
                return 10

        group["priority"] = group.apply(_priority, axis=1)
        best = group.sort_values("priority", ascending=False).iloc[0]

        records.append({
            "date": target_date,
            "tmax_forecast_f": float(best["n_x"]),
            "runtime": best["runtime"],
            "model": best["model"],
        })

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result = result.sort_values("date").reset_index(drop=True)
    return result


def download_model_full(station: str, model: str) -> pd.DataFrame:
    """Download full historical MOS archive for a single model."""
    all_chunks = []

    for year in range(START_YEAR, END_YEAR + 1):
        raw = download_mos_chunk(station, model, year)
        if raw is None:
            logger.warning("No data for %s %s %d, skipping.", station, model, year)
            continue

        try:
            chunk_df = parse_mos_csv(raw)
            all_chunks.append(chunk_df)
        except Exception as exc:
            logger.error(
                "Failed to parse %s %s %d: %s", station, model, year, exc,
            )

        # Be polite to IEM servers
        time.sleep(0.5)

    if not all_chunks:
        logger.error("No data downloaded for %s %s", station, model)
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])

    combined = pd.concat(all_chunks, ignore_index=True)
    logger.info(
        "Downloaded %d total rows for %s %s across %d years",
        len(combined), station, model, len(all_chunks),
    )

    tmax_df = extract_tmax_forecasts(combined)
    logger.info(
        "Extracted %d TMAX forecasts for %s %s (date range: %s to %s)",
        len(tmax_df), station, model,
        tmax_df["date"].min() if not tmax_df.empty else "N/A",
        tmax_df["date"].max() if not tmax_df.empty else "N/A",
    )
    return tmax_df


def build_combined_mos(
    gfs_df: pd.DataFrame,
    nam_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine GFS and NAM MOS forecasts into an ensemble."""
    # Prepare GFS
    gfs_slim = gfs_df[["date", "tmax_forecast_f", "runtime"]].copy()
    gfs_slim.columns = ["date", "gfs_mos_tmax_f", "gfs_runtime"]
    gfs_slim["date"] = pd.to_datetime(gfs_slim["date"])

    # Prepare NAM
    nam_slim = nam_df[["date", "tmax_forecast_f", "runtime"]].copy()
    nam_slim.columns = ["date", "nam_mos_tmax_f", "nam_runtime"]
    nam_slim["date"] = pd.to_datetime(nam_slim["date"])

    # Merge on date
    combined = pd.merge(gfs_slim, nam_slim, on="date", how="outer")
    combined = combined.sort_values("date").reset_index(drop=True)

    # Ensemble: average where both available, single model otherwise
    combined["mos_ensemble_tmax_f"] = combined[
        ["gfs_mos_tmax_f", "nam_mos_tmax_f"]
    ].mean(axis=1)

    return combined


def validate_data(df: pd.DataFrame, name: str) -> None:
    """Run basic data quality checks and print summary."""
    if df.empty:
        logger.error("VALIDATION FAILED: %s DataFrame is empty!", name)
        return

    logger.info("=" * 60)
    logger.info("Data Quality Report: %s", name)
    logger.info("=" * 60)

    date_col = "date"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        logger.info("  Date range: %s to %s", dates.min().date(), dates.max().date())
        logger.info("  Total days: %d", len(df))

    tmax_cols = [c for c in df.columns if "tmax" in c.lower()]
    for col in tmax_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            logger.warning("  %s: ALL values are NaN!", col)
            continue

        logger.info(
            "  %s: count=%d, mean=%.1f, std=%.1f, min=%.1f, max=%.1f, missing=%d (%.1f%%)",
            col, len(vals), vals.mean(), vals.std(), vals.min(), vals.max(),
            df[col].isna().sum(), 100 * df[col].isna().sum() / len(df),
        )

        unrealistic = vals[(vals < -20) | (vals > 120)]
        if len(unrealistic) > 0:
            logger.warning(
                "  WARNING: %d unrealistic values in %s (outside -20 to 120F)",
                len(unrealistic), col,
            )


def analyze_gaps(df: pd.DataFrame, name: str) -> None:
    """Analyze gaps in the date coverage."""
    if df.empty:
        return

    dates = pd.to_datetime(df["date"])
    date_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
    missing = date_range.difference(dates)

    print(f"\n--- Gap Analysis: {name} ---")
    print(f"  Expected days (min to max): {len(date_range):,}")
    print(f"  Actual days: {len(df):,}")
    print(f"  Missing days: {len(missing):,} ({100*len(missing)/len(date_range):.1f}%)")

    if len(missing) > 0:
        # Find contiguous gaps
        gaps = []
        gap_start = missing[0]
        gap_end = missing[0]
        for d in missing[1:]:
            if d - gap_end == timedelta(days=1):
                gap_end = d
            else:
                gaps.append((gap_start, gap_end, (gap_end - gap_start).days + 1))
                gap_start = d
                gap_end = d
        gaps.append((gap_start, gap_end, (gap_end - gap_start).days + 1))

        # Show largest gaps
        gaps.sort(key=lambda x: x[2], reverse=True)
        print(f"  Largest gaps (top 10):")
        for start, end, length in gaps[:10]:
            print(f"    {start.date()} to {end.date()} ({length} days)")

        # Coverage by year
        print(f"\n  Coverage by year:")
        for year in range(dates.min().year, dates.max().year + 1):
            year_mask = dates.dt.year == year
            year_days = year_mask.sum()
            expected = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            # Adjust for partial first/last years
            year_start = max(pd.Timestamp(f"{year}-01-01"), dates.min())
            year_end = min(pd.Timestamp(f"{year}-12-31"), dates.max())
            expected_partial = (year_end - year_start).days + 1
            print(f"    {year}: {year_days:4d} / {expected_partial:3d} days ({100*year_days/expected_partial:.1f}%)")


def main() -> None:
    """Main entry point: download, process, and save MOS data for KPHL."""
    os.makedirs(MOS_DATA_DIR, exist_ok=True)

    # Download GFS MOS
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading GFS MOS archive for %s", STATION)
    logger.info("=" * 60)
    gfs_df = download_model_full(STATION, "GFS")

    # Download NAM MOS
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading NAM MOS archive for %s", STATION)
    logger.info("=" * 60)
    nam_df = download_model_full(STATION, "NAM")

    if gfs_df.empty and nam_df.empty:
        logger.error("FATAL: No MOS data downloaded. Check network connectivity.")
        sys.exit(1)

    # Save individual model files
    gfs_out_path = os.path.join(MOS_DATA_DIR, "gfs_mos_kphl.csv")
    nam_out_path = os.path.join(MOS_DATA_DIR, "nam_mos_kphl.csv")
    combined_out_path = os.path.join(MOS_DATA_DIR, "combined_mos_kphl.csv")

    if not gfs_df.empty:
        gfs_save = gfs_df.rename(columns={"tmax_forecast_f": "gfs_mos_tmax_f"})
        gfs_save[["date", "gfs_mos_tmax_f", "runtime"]].to_csv(
            gfs_out_path, index=False,
        )
        logger.info("Saved GFS MOS to %s", gfs_out_path)

    if not nam_df.empty:
        nam_save = nam_df.rename(columns={"tmax_forecast_f": "nam_mos_tmax_f"})
        nam_save[["date", "nam_mos_tmax_f", "runtime"]].to_csv(
            nam_out_path, index=False,
        )
        logger.info("Saved NAM MOS to %s", nam_out_path)

    # Build combined / ensemble
    logger.info("=" * 60)
    logger.info("STEP 3: Building combined MOS ensemble")
    logger.info("=" * 60)
    combined_df = build_combined_mos(gfs_df, nam_df)
    combined_df.to_csv(combined_out_path, index=False)
    logger.info("Saved combined MOS to %s", combined_out_path)

    # Validate
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Data validation")
    logger.info("=" * 60)
    if not gfs_df.empty:
        validate_data(
            gfs_df.rename(columns={"tmax_forecast_f": "gfs_mos_tmax_f"}),
            "GFS MOS",
        )
    if not nam_df.empty:
        validate_data(
            nam_df.rename(columns={"tmax_forecast_f": "nam_mos_tmax_f"}),
            "NAM MOS",
        )
    validate_data(combined_df, "Combined MOS Ensemble")

    # Gap analysis
    if not gfs_df.empty:
        analyze_gaps(
            gfs_df.rename(columns={"tmax_forecast_f": "gfs_mos_tmax_f"}),
            "GFS MOS",
        )
    if not nam_df.empty:
        analyze_gaps(
            nam_df.rename(columns={"tmax_forecast_f": "nam_mos_tmax_f"}),
            "NAM MOS",
        )
    analyze_gaps(combined_df, "Combined MOS Ensemble")

    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    if not gfs_df.empty:
        gfs_dates = pd.to_datetime(gfs_df["date"])
        print(f"GFS MOS:     {len(gfs_df):,} days  ({gfs_dates.min().date()} to {gfs_dates.max().date()})")
    else:
        print("GFS MOS:     NO DATA")

    if not nam_df.empty:
        nam_dates = pd.to_datetime(nam_df["date"])
        print(f"NAM MOS:     {len(nam_df):,} days  ({nam_dates.min().date()} to {nam_dates.max().date()})")
    else:
        print("NAM MOS:     NO DATA")

    comb_dates = pd.to_datetime(combined_df["date"])
    gfs_count = combined_df["gfs_mos_tmax_f"].notna().sum()
    nam_count = combined_df["nam_mos_tmax_f"].notna().sum()
    both_count = (
        combined_df["gfs_mos_tmax_f"].notna() & combined_df["nam_mos_tmax_f"].notna()
    ).sum()
    print(f"Combined:    {len(combined_df):,} days  ({comb_dates.min().date()} to {comb_dates.max().date()})")
    print(f"  GFS available:  {gfs_count:,} days ({100*gfs_count/len(combined_df):.1f}%)")
    print(f"  NAM available:  {nam_count:,} days ({100*nam_count/len(combined_df):.1f}%)")
    print(f"  Both available: {both_count:,} days ({100*both_count/len(combined_df):.1f}%)")
    print(f"\nFiles saved to: {MOS_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
