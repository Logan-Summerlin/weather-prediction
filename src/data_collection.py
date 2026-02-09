"""
NOAA GHCN-Daily Data Collection Module.

Downloads raw .dly files from the NOAA GHCN bulk server and parses
them into clean CSV files with daily temperature, precipitation,
snow, and wind observations in operational units.

GHCN .dly Fixed-Width Format
-----------------------------
Each record line contains:
  - station_id : chars  0-10  (11 chars)
  - year       : chars 11-14  ( 4 chars)
  - month      : chars 15-16  ( 2 chars)
  - element    : chars 17-20  ( 4 chars)
Then 31 daily-value groups, each 8 chars wide:
  - value      : 5 chars (integer, tenths of degree C; -9999 = missing)
  - mflag      : 1 char  (measurement flag)
  - qflag      : 1 char  (quality flag; blank = passed all checks)
  - sflag      : 1 char  (source flag)
Total line length = 21 + 31*8 = 269 chars.

Temperature Conversion
----------------------
Raw values are in tenths of degrees Celsius.
  temp_fahrenheit = (value / 10) * 9/5 + 32

Quality Flags
-------------
If the qflag is anything other than a blank space, the observation
failed one or more quality-assurance checks and should be excluded.
"""

import os
import sys
import logging
import calendar
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests

# Add project root to path so config is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISSING_VALUE = -9999
RECORD_HEADER_LEN = 21  # station(11) + year(4) + month(2) + element(4)
DAILY_FIELD_WIDTH = 8    # value(5) + mflag(1) + qflag(1) + sflag(1)
VALUE_WIDTH = 5
ELEMENTS_OF_INTEREST = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"}
ELEMENT_UNITS = {
    "TMAX": "degF",
    "TMIN": "degF",
    "PRCP": "in",
    "SNOW": "in",
    "SNWD": "in",
    "AWND": "mph",
}


# ===========================================================================
# Core Functions
# ===========================================================================

def tenths_c_to_fahrenheit(value_tenths: int) -> float:
    """Convert a temperature from tenths of degrees Celsius to Fahrenheit.

    Parameters
    ----------
    value_tenths : int
        Temperature in tenths of degrees Celsius (e.g., 215 means 21.5 C).

    Returns
    -------
    float
        Temperature in degrees Fahrenheit.

    Examples
    --------
    >>> tenths_c_to_fahrenheit(0)
    32.0
    >>> tenths_c_to_fahrenheit(1000)
    212.0
    >>> tenths_c_to_fahrenheit(-400)
    -40.0
    """
    return (value_tenths / 10) * 9 / 5 + 32


def tenths_mm_to_inches(value_tenths: int) -> float:
    """Convert tenths of millimeters to inches.

    GHCN stores PRCP in tenths of mm. SNOW and SNWD are stored in mm.
    This helper expects a value in tenths of mm and returns inches.
    """
    return (value_tenths / 10) / 25.4


def mm_to_inches(value_mm: int) -> float:
    """Convert millimeters to inches."""
    return value_mm / 25.4


def tenths_ms_to_mph(value_tenths: int) -> float:
    """Convert tenths of meters per second to miles per hour."""
    return (value_tenths / 10) * 2.236936


def convert_element_value(element: str, value_raw: int) -> tuple[float, str]:
    """Convert raw GHCN element values into operational units.

    Parameters
    ----------
    element : str
        GHCN element name (TMAX, TMIN, PRCP, SNOW, SNWD, AWND).
    value_raw : int
        Raw value from the .dly file.

    Returns
    -------
    tuple[float, str]
        Converted value and unit label.
    """
    if element in ("TMAX", "TMIN"):
        return tenths_c_to_fahrenheit(value_raw), ELEMENT_UNITS[element]
    if element == "PRCP":
        return tenths_mm_to_inches(value_raw), ELEMENT_UNITS[element]
    if element in ("SNOW", "SNWD"):
        return mm_to_inches(value_raw), ELEMENT_UNITS[element]
    if element == "AWND":
        return tenths_ms_to_mph(value_raw), ELEMENT_UNITS[element]
    raise ValueError(f"Unsupported element for conversion: {element}")


def parse_dly_line(line: str) -> list[dict]:
    """Parse a single line from a GHCN .dly file.

    Parameters
    ----------
    line : str
        A single record line from a .dly file (269 chars).

    Returns
    -------
    list[dict]
        A list of observation dictionaries, one per valid day in the line.
        Each dict has keys: station_id, date, element, value_raw,
        value, units, mflag, qflag, sflag.
        Days with missing values (-9999) or failed quality flags are excluded.
    """
    # Strip trailing newline/whitespace but pad if too short
    line = line.rstrip("\n\r")

    if len(line) < RECORD_HEADER_LEN:
        return []

    station_id = line[0:11]
    year = int(line[11:15])
    month = int(line[15:17])
    element = line[17:21]

    # Only parse elements we care about
    if element not in ELEMENTS_OF_INTEREST:
        return []

    # Determine the number of days in this month
    days_in_month = calendar.monthrange(year, month)[1]

    observations = []
    for day_index in range(31):
        offset = RECORD_HEADER_LEN + day_index * DAILY_FIELD_WIDTH

        # If the line is shorter than expected, remaining days are missing
        if offset + VALUE_WIDTH > len(line):
            break

        # Skip days beyond the actual month length (e.g., day 31 of February)
        if day_index + 1 > days_in_month:
            break

        value_str = line[offset : offset + VALUE_WIDTH]

        try:
            value_raw = int(value_str)
        except ValueError:
            continue

        # Skip missing values
        if value_raw == MISSING_VALUE:
            continue

        # Extract flags
        mflag = line[offset + 5] if offset + 6 <= len(line) else " "
        qflag = line[offset + 6] if offset + 7 <= len(line) else " "
        sflag = line[offset + 7] if offset + 8 <= len(line) else " "

        # Exclude observations that failed quality checks
        # A blank (space) or empty qflag means the observation passed
        if qflag not in (" ", ""):
            continue

        day_num = day_index + 1
        try:
            obs_date = date(year, month, day_num)
        except ValueError:
            # Shouldn't happen since we checked days_in_month, but be safe
            continue

        value_converted, units = convert_element_value(element, value_raw)

        observations.append({
            "station_id": station_id,
            "date": obs_date,
            "element": element,
            "value_raw": value_raw,
            "value": value_converted,
            "units": units,
            "mflag": mflag,
            "qflag": qflag,
            "sflag": sflag,
        })

    return observations


def parse_dly_file(filepath: str, start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """Parse an entire .dly file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .dly file on disk.
    start_date : str, optional
        Start date for filtering (inclusive), format "YYYY-MM-DD".
    end_date : str, optional
        End date for filtering (inclusive), format "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station_id, date, element, value_raw,
        value, units, mflag, qflag, sflag.
        Filtered to the specified date range and containing only configured
        elements.
    """
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start_dt = None

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_dt = None

    all_observations = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Quick pre-filter: check if the year is even plausible
            # before full parsing (performance optimization for large files)
            if start_dt or end_dt:
                try:
                    year = int(line[11:15])
                except (ValueError, IndexError):
                    continue
                if start_dt and year < start_dt.year:
                    continue
                if end_dt and year > end_dt.year:
                    continue

            observations = parse_dly_line(line)
            all_observations.extend(observations)

    if not all_observations:
        return pd.DataFrame(columns=[
            "station_id", "date", "element", "value_raw",
            "value", "units", "mflag", "qflag", "sflag"
        ])

    df = pd.DataFrame(all_observations)

    # Apply precise date filtering
    if start_dt:
        df = df[df["date"] >= start_dt]
    if end_dt:
        df = df[df["date"] <= end_dt]

    df = df.sort_values("date").reset_index(drop=True)
    return df


def pivot_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot parsed .dly data from long to wide format.

    Converts from one row per (date, element) to one row per date with
    separate element columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from parse_dly_file.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with 'date' as index and columns for each
        configured element.
    """
    if df.empty:
        empty = pd.DataFrame(columns=["date", *sorted(ELEMENTS_OF_INTEREST)])
        return empty.set_index("date")

    pivot = df.pivot_table(
        index="date",
        columns="element",
        values="value",
        aggfunc="first",  # If duplicates, take the first
    )
    pivot.index.name = "date"
    return pivot.reindex(columns=sorted(ELEMENTS_OF_INTEREST))


def download_dly_file(station_id: str, output_dir: str,
                      base_url: Optional[str] = None,
                      timeout: int = 120) -> str:
    """Download a .dly file from the NOAA GHCN bulk server.

    Parameters
    ----------
    station_id : str
        The GHCN station identifier (e.g., "USW00094728").
    output_dir : str
        Directory to save the downloaded file.
    base_url : str, optional
        Base URL for bulk downloads. Defaults to config.NOAA_BULK_BASE_URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Path to the downloaded .dly file.

    Raises
    ------
    requests.HTTPError
        If the download fails with an HTTP error.
    requests.ConnectionError
        If network is unavailable.
    """
    if base_url is None:
        base_url = config.NOAA_BULK_BASE_URL

    url = f"{base_url}{station_id}.dly"
    output_path = os.path.join(output_dir, f"{station_id}.dly")

    logger.info("Downloading %s from %s", station_id, url)

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)

    file_size = os.path.getsize(output_path)
    logger.info("Saved %s (%.1f KB)", output_path, file_size / 1024)

    return output_path


def process_station(station_id: str, raw_dir: str,
                    start_date: str, end_date: str) -> pd.DataFrame:
    """Download, parse, and save a single station's data.

    Parameters
    ----------
    station_id : str
        The GHCN station identifier.
    raw_dir : str
        Directory for raw .dly files and output CSVs.
    start_date : str
        Start date (inclusive) in "YYYY-MM-DD" format.
    end_date : str
        End date (inclusive) in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with TMAX, TMIN per date.
    """
    dly_path = os.path.join(raw_dir, f"{station_id}.dly")

    # Download if not already present
    if not os.path.exists(dly_path):
        download_dly_file(station_id, raw_dir)
    else:
        logger.info("Using cached .dly file for %s", station_id)

    # Parse
    logger.info("Parsing %s.dly ...", station_id)
    df_long = parse_dly_file(dly_path, start_date, end_date)

    if df_long.empty:
        logger.warning("No data found for station %s in range %s to %s",
                        station_id, start_date, end_date)
        return pd.DataFrame()

    # Pivot to wide format
    df_wide = pivot_station_data(df_long)

    # Save as CSV
    csv_path = os.path.join(raw_dir, f"{station_id}.csv")
    df_wide.to_csv(csv_path)
    logger.info("Saved %s with %d rows, columns: %s",
                csv_path, len(df_wide), list(df_wide.columns))

    return df_wide


def collect_all_stations() -> dict[str, pd.DataFrame]:
    """Download and process data for all configured stations.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of station_id -> wide-format DataFrame.
    """
    raw_dir = config.RAW_DATA_DIR
    os.makedirs(raw_dir, exist_ok=True)

    station_data = {}
    all_stations = config.ALL_STATIONS
    total = len(all_stations)

    for i, (station_id, description) in enumerate(all_stations.items(), 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] Processing: %s — %s", i, total, station_id, description)
        logger.info("=" * 60)

        try:
            df = process_station(
                station_id, raw_dir, config.START_DATE, config.END_DATE
            )
            if not df.empty:
                station_data[station_id] = df
                logger.info("  -> %d records for %s", len(df), station_id)
            else:
                logger.warning("  -> EMPTY dataset for %s — skipping", station_id)
        except requests.exceptions.RequestException as e:
            logger.error("  -> DOWNLOAD FAILED for %s: %s", station_id, e)
        except Exception as e:
            logger.error("  -> ERROR processing %s: %s", station_id, e, exc_info=True)

    logger.info("=" * 60)
    logger.info("Collection complete: %d / %d stations succeeded",
                len(station_data), total)
    logger.info("=" * 60)

    return station_data


# ===========================================================================
# Main Entry Point
# ===========================================================================

def main():
    """Main entry point for data collection.

    Downloads .dly files for all configured stations, parses them,
    and saves station CSVs to data/raw/.
    """
    logger.info("Starting GHCN-Daily data collection")
    logger.info("Date range: %s to %s", config.START_DATE, config.END_DATE)
    logger.info("Target station: %s", config.TARGET_STATION)
    logger.info("Surrounding stations: %d", len(config.SURROUNDING_STATIONS))

    station_data = collect_all_stations()

    # Summary report
    logger.info("\n--- Data Collection Summary ---")
    for sid, df in station_data.items():
        desc = config.ALL_STATIONS.get(sid, "Unknown")
        n_tmax = df["TMAX"].notna().sum() if "TMAX" in df.columns else 0
        n_tmin = df["TMIN"].notna().sum() if "TMIN" in df.columns else 0
        logger.info("  %s (%s): TMAX=%d days, TMIN=%d days",
                     sid, desc, n_tmax, n_tmin)

    return station_data


if __name__ == "__main__":
    main()
