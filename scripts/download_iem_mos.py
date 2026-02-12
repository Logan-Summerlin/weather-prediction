#!/usr/bin/env python3
"""
Download MOS (Model Output Statistics) archive from Iowa Environmental Mesonet.

Downloads GFS MOS and NAM MOS historical forecasts for KNYC from IEM,
extracts day-ahead max temperature forecasts, and saves processed data.

Data source: https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py

Usage:
    python scripts/download_iem_mos.py
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
import numpy as np

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MOS_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mos")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATION = "KNYC"
MODELS = ["GFS", "NAM"]
START_YEAR = 2003  # Earliest reliable GFS-like MOS data for KNYC in IEM archive
END_YEAR = 2026    # Up to current year
NETWORK_START_YEAR = 2018  # Practical range for multi-station aggregate features
AVN_START_YEAR = 1998
AVN_END_YEAR = 2003
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 120  # seconds


def download_mos_chunk(
    station: str,
    model: str,
    year: int,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[str]:
    """Download one year of MOS data from IEM with exponential backoff.

    Parameters
    ----------
    station : str
        4-character station identifier (e.g., KNYC).
    model : str
        Model name (GFS or NAM).
    year : int
        Calendar year to download.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str or None
        Raw CSV text, or None if download failed after retries.
    """
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
    """Parse raw IEM MOS CSV into a DataFrame.

    Parameters
    ----------
    raw_csv : str
        Raw CSV text from IEM.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with columns: runtime, ftime, model, n_x, tmp, ...
    """
    df = pd.read_csv(StringIO(raw_csv))
    for col in ["runtime", "ftime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def extract_tmax_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day-ahead TMAX forecasts from parsed MOS data.

    MOS N/X field convention:
    - n_x at ftime with hour == 0 (00Z) = MAX temperature (TMAX)
      for the PREVIOUS calendar day (ftime.date() - 1 day)
    - n_x at ftime with hour == 12 (12Z) = MIN temperature (TMIN)
      for the overnight period

    For each target date, we prefer the most recent 12Z runtime from
    the day before (day-ahead forecast), falling back to 00Z same day.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed MOS DataFrame with runtime, ftime, n_x columns.

    Returns
    -------
    pd.DataFrame
        Columns: date, tmax_forecast_f, runtime, model
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])

    # Filter to rows with valid n_x at 00Z ftime (TMAX values)
    mask = df["n_x"].notna() & (df["ftime"].dt.hour == 0)
    tmax_df = df.loc[mask].copy()

    if tmax_df.empty:
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])

    # The target date for a TMAX forecast is ftime.date() - 1 day
    # (because the daytime high occurs before midnight UTC)
    tmax_df["target_date"] = tmax_df["ftime"].dt.date - pd.Timedelta(days=1)
    tmax_df["runtime_hour"] = tmax_df["runtime"].dt.hour
    tmax_df["runtime_date"] = tmax_df["runtime"].dt.date

    # For day-ahead forecasts, prefer:
    # 1. 12Z run from the day BEFORE the target date (standard day-ahead)
    # 2. 00Z run from the target date itself
    # 3. Any other runtime, preferring most recent
    #
    # We compute a "priority" score: higher = better
    # day_before_12z gets highest priority, then day_before_00z, etc.
    records = []
    for target_date, group in tmax_df.groupby("target_date"):
        # Sort by preference: latest runtime from day before target, then same day
        group = group.copy()
        target_dt = pd.Timestamp(target_date)
        day_before = (target_dt - pd.Timedelta(days=1)).date()

        # Assign priority
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
                # Fallback: prefer more recent runtimes
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


def _compute_relative_humidity_from_f(tmp_f: pd.Series, dpt_f: pd.Series) -> pd.Series:
    """Compute relative humidity (%) from temperature and dewpoint in Fahrenheit."""
    tmp_c = (tmp_f - 32.0) * (5.0 / 9.0)
    dpt_c = (dpt_f - 32.0) * (5.0 / 9.0)
    es = 6.112 * np.exp((17.67 * tmp_c) / (tmp_c + 243.5))
    e = 6.112 * np.exp((17.67 * dpt_c) / (dpt_c + 243.5))
    rh = 100.0 * (e / es)
    return rh.clip(lower=0.0, upper=100.0)


def extract_daily_feature_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day-ahead MOS daily features aligned to target-date Tmax.

    Uses same target-date alignment + runtime-priority logic as TMAX extraction.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date", "tmp", "dpt", "cld", "wdr", "wsp", "p06", "snw", "runtime", "model",
            ]
        )

    mask = df["n_x"].notna() & (df["ftime"].dt.hour == 0)
    base = df.loc[mask].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "date", "tmp", "dpt", "cld", "wdr", "wsp", "p06", "snw", "runtime", "model",
            ]
        )

    base["target_date"] = base["ftime"].dt.date - pd.Timedelta(days=1)
    base["runtime_hour"] = base["runtime"].dt.hour
    base["runtime_date"] = base["runtime"].dt.date

    rows = []
    for target_date, group in base.groupby("target_date"):
        target_dt = pd.Timestamp(target_date)
        day_before = (target_dt - pd.Timedelta(days=1)).date()

        def _priority(row):
            rd = row["runtime_date"]
            rh = row["runtime_hour"]
            if rd == day_before and rh == 12:
                return 100
            if rd == day_before and rh == 18:
                return 90
            if rd == day_before and rh == 6:
                return 80
            if rd == day_before and rh == 0:
                return 70
            if rd == target_date and rh == 0:
                return 60
            if rd == target_date and rh == 6:
                return 50
            return 10

        group = group.copy()
        group["priority"] = group.apply(_priority, axis=1)
        best = group.sort_values("priority", ascending=False).iloc[0]
        rows.append(
            {
                "date": target_date,
                "tmp": best.get("tmp", np.nan),
                "dpt": best.get("dpt", np.nan),
                "cld": best.get("cld", np.nan),
                "wdr": best.get("wdr", np.nan),
                "wsp": best.get("wsp", np.nan),
                "p06": best.get("p06", np.nan),
                "snw": best.get("snw", np.nan),
                "runtime": best.get("runtime"),
                "model": best.get("model"),
            }
        )

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def download_model_full(station: str, model: str) -> pd.DataFrame:
    """Download full historical MOS archive for a single model.

    Downloads year by year from START_YEAR to END_YEAR, concatenates,
    and extracts TMAX forecasts.

    Parameters
    ----------
    station : str
        Station identifier.
    model : str
        Model name (GFS or NAM).

    Returns
    -------
    pd.DataFrame
        TMAX forecasts with columns: date, tmax_forecast_f, runtime, model
    """
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


def download_model_daily_features_full(station: str, model: str, start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """Download MOS archive and extract daily feature forecasts."""
    all_chunks = []
    for year in range(start_year, end_year + 1):
        raw = download_mos_chunk(station, model, year)
        if raw is None:
            continue
        try:
            all_chunks.append(parse_mos_csv(raw))
        except Exception as exc:
            logger.error("Failed to parse daily-feature chunk %s %s %d: %s", station, model, year, exc)
        time.sleep(0.35)

    if not all_chunks:
        return pd.DataFrame(columns=["date", "tmp", "dpt", "cld", "wdr", "wsp", "p06", "snw", "runtime", "model"])

    combined = pd.concat(all_chunks, ignore_index=True)
    return extract_daily_feature_forecasts(combined)


def build_station_aggregate_features() -> pd.DataFrame:
    """Build daily average MOS features across non-KNYC nearby stations."""
    mapping_path = os.path.join(PROJECT_ROOT, "data", "asos_station_mapping.csv")
    expanded_path = os.path.join(PROJECT_ROOT, "data", "stations_expanded.csv")
    mapping = pd.read_csv(mapping_path)
    expanded = pd.read_csv(expanded_path)

    map_lut = mapping.set_index("station_id")["icao"].to_dict()
    prioritized = expanded.sort_values(["priority", "distance_mi"], ascending=[False, True])["station_id"].tolist()
    station_codes = []
    for sid in prioritized:
        icao = str(map_lut.get(sid, "")).upper()
        if not icao or icao == "NAN" or icao == "KNYC":
            continue
        row = mapping[mapping["station_id"] == sid]
        if row.empty or str(row.iloc[0]["asos_available"]).lower() != "yes":
            continue
        station_codes.append(icao)
        if len(station_codes) >= 8:
            break

    station_daily = []
    for station in station_codes:
        daily = download_model_daily_features_full(station, "GFS", start_year=NETWORK_START_YEAR, end_year=END_YEAR)
        if daily.empty:
            continue
        slim = daily[["date", "wsp", "p06", "snw"]].copy()
        slim["station"] = station
        station_daily.append(slim)

    if not station_daily:
        return pd.DataFrame(columns=["date", "other_station_avg_wind_speed_mph", "other_station_avg_precip_prob", "other_station_avg_snow_indicator"])

    all_daily = pd.concat(station_daily, ignore_index=True)
    agg = all_daily.groupby("date", as_index=False).agg(
        other_station_avg_wind_speed_mph=("wsp", "mean"),
        other_station_avg_precip_prob=("p06", "mean"),
        other_station_avg_snow_indicator=("snw", "mean"),
    )
    return agg




def download_model_range(station: str, model: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Download model archive for a custom year range and extract Tmax forecasts."""
    all_chunks = []
    for year in range(start_year, end_year + 1):
        raw = download_mos_chunk(station, model, year)
        if raw is None:
            continue
        try:
            all_chunks.append(parse_mos_csv(raw))
        except Exception as exc:
            logger.error("Failed to parse %s %s %d in range download: %s", station, model, year, exc)
        time.sleep(0.35)
    if not all_chunks:
        return pd.DataFrame(columns=["date", "tmax_forecast_f", "runtime", "model"])
    return extract_tmax_forecasts(pd.concat(all_chunks, ignore_index=True))

def build_combined_mos(
    gfs_df: pd.DataFrame,
    nam_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine GFS and NAM MOS forecasts into an ensemble.

    Parameters
    ----------
    gfs_df : pd.DataFrame
        GFS TMAX forecasts with columns: date, tmax_forecast_f, runtime, model
    nam_df : pd.DataFrame
        NAM TMAX forecasts with columns: date, tmax_forecast_f, runtime, model

    Returns
    -------
    pd.DataFrame
        Columns: date, gfs_mos_tmax_f, nam_mos_tmax_f, mos_ensemble_tmax_f
    """
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
    """Run basic data quality checks and print summary.

    Parameters
    ----------
    df : pd.DataFrame
        The TMAX forecast DataFrame.
    name : str
        Descriptive name for logging.
    """
    if df.empty:
        logger.error("VALIDATION FAILED: %s DataFrame is empty!", name)
        return

    logger.info("=" * 60)
    logger.info("Data Quality Report: %s", name)
    logger.info("=" * 60)

    # Date range
    date_col = "date"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        logger.info("  Date range: %s to %s", dates.min().date(), dates.max().date())
        logger.info("  Total days: %d", len(df))

    # TMAX columns
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

        # Check for unrealistic values
        unrealistic = vals[(vals < -20) | (vals > 120)]
        if len(unrealistic) > 0:
            logger.warning(
                "  WARNING: %d unrealistic values in %s (outside -20 to 120F)",
                len(unrealistic), col,
            )


def print_sample_rows(df: pd.DataFrame, name: str, n: int = 5) -> None:
    """Print first and last N rows of a DataFrame."""
    print(f"\n{'='*60}")
    print(f"Sample data: {name}")
    print(f"{'='*60}")
    print(f"\nFirst {n} rows:")
    print(df.head(n).to_string(index=False))
    print(f"\nLast {n} rows:")
    print(df.tail(n).to_string(index=False))
    print()


def main() -> None:
    """Main entry point: download, process, and save MOS data."""
    os.makedirs(MOS_DATA_DIR, exist_ok=True)

    # Download GFS MOS
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading GFS MOS archive for %s", STATION)
    logger.info("=" * 60)
    gfs_df = download_model_full(STATION, "GFS")

    # Attempt AVN MOS back-extension (legacy GFS predecessor)
    logger.info("=" * 60)
    logger.info("STEP 1b: Downloading AVN MOS archive for %s", STATION)
    logger.info("=" * 60)
    avn_df = download_model_range(STATION, "AVN", AVN_START_YEAR, AVN_END_YEAR)
    if not avn_df.empty:
        avn_for_merge = avn_df.rename(columns={"tmax_forecast_f": "avn_mos_tmax_f", "runtime": "avn_runtime"})
        gfs_df = gfs_df.merge(avn_for_merge[["date", "avn_mos_tmax_f", "avn_runtime"]], on="date", how="outer")
        if "gfs_mos_tmax_f" not in gfs_df.columns:
            gfs_df["gfs_mos_tmax_f"] = np.nan
        if "runtime" not in gfs_df.columns:
            gfs_df["runtime"] = pd.NaT
        gfs_df["gfs_mos_tmax_f"] = gfs_df["gfs_mos_tmax_f"].combine_first(gfs_df["avn_mos_tmax_f"])
        gfs_df["runtime"] = gfs_df["runtime"].combine_first(gfs_df["avn_runtime"])
        gfs_df = gfs_df.drop(columns=["avn_mos_tmax_f", "avn_runtime"]).sort_values("date").reset_index(drop=True)

    # Download NAM MOS
    logger.info("=" * 60)
    logger.info("STEP 2: Downloading NAM MOS archive for %s", STATION)
    logger.info("=" * 60)
    nam_df = download_model_full(STATION, "NAM")

    if gfs_df.empty and nam_df.empty:
        logger.error("FATAL: No MOS data downloaded. Check network connectivity.")
        sys.exit(1)

    # Download KNYC GFS daily MOS feature forecasts (wind/cloud/dewpoint/precip/snow)
    logger.info("=" * 60)
    logger.info("STEP 2b: Downloading KNYC GFS daily feature MOS")
    logger.info("=" * 60)
    gfs_knyc_daily = download_model_daily_features_full(STATION, "GFS")
    if not gfs_knyc_daily.empty:
        gfs_knyc_daily = gfs_knyc_daily.rename(
            columns={
                "wsp": "knyc_mos_wind_speed_mph",
                "wdr": "knyc_mos_wind_dir_deg",
                "cld": "knyc_mos_cloud_cover_code",
                "dpt": "knyc_mos_dewpoint_f",
                "tmp": "knyc_mos_tmp_f",
                "p06": "knyc_mos_precip_prob",
                "snw": "knyc_mos_snow_indicator",
            }
        )
        gfs_knyc_daily["knyc_mos_rel_humidity_pct"] = _compute_relative_humidity_from_f(
            gfs_knyc_daily["knyc_mos_tmp_f"], gfs_knyc_daily["knyc_mos_dewpoint_f"]
        )

    logger.info("=" * 60)
    logger.info("STEP 2c: Downloading nearby-station GFS MOS daily features for network averages")
    logger.info("=" * 60)
    station_agg_daily = build_station_aggregate_features()

    # Save individual model files
    gfs_out_path = os.path.join(MOS_DATA_DIR, "gfs_mos_knyc.csv")
    nam_out_path = os.path.join(MOS_DATA_DIR, "nam_mos_knyc.csv")
    combined_out_path = os.path.join(MOS_DATA_DIR, "combined_mos_knyc.csv")

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
    if not gfs_knyc_daily.empty:
        merged_cols = [
            "date",
            "knyc_mos_wind_speed_mph",
            "knyc_mos_wind_dir_deg",
            "knyc_mos_cloud_cover_code",
            "knyc_mos_dewpoint_f",
            "knyc_mos_rel_humidity_pct",
            "knyc_mos_precip_prob",
            "knyc_mos_snow_indicator",
        ]
        combined_df = combined_df.merge(gfs_knyc_daily[merged_cols], on="date", how="left")
    if not station_agg_daily.empty:
        combined_df = combined_df.merge(station_agg_daily, on="date", how="left")
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

    # Print samples
    if not gfs_df.empty:
        print_sample_rows(
            gfs_df.rename(columns={"tmax_forecast_f": "gfs_mos_tmax_f"}),
            "GFS MOS",
        )
    if not nam_df.empty:
        print_sample_rows(
            nam_df.rename(columns={"tmax_forecast_f": "nam_mos_tmax_f"}),
            "NAM MOS",
        )
    print_sample_rows(combined_df, "Combined MOS Ensemble")

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
