"""
IGRA Soundings Daily Preprocessing.

Extracts daily upper-air features from raw IGRA sounding CSV files
downloaded by soundings_collection.py. Features include pressure-level
temperatures, wind, stability indices, and lapse rates.

Key outputs per sounding:
  - Temperature at specified pressure levels (C and F)
  - Wind direction and speed at each level
  - Geopotential height at each level
  - Dewpoint at each level
  - Stability index (T850 - T_surface)
  - 850-500mb lapse rate
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

import config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Siphon IGRA CSV column names (as observed from IGRAUpperAir.request_data)
COL_PRESSURE = "pressure"
COL_HEIGHT = "height"
COL_TEMPERATURE = "temperature"
COL_DEWPOINT = "dewpoint"
COL_DIRECTION = "direction"
COL_SPEED = "speed"
COL_U_WIND = "u_wind"
COL_V_WIND = "v_wind"
COL_STATION = "station"
COL_STATION_NUMBER = "station_number"
COL_LATITUDE = "latitude"
COL_LONGITUDE = "longitude"
COL_ELEVATION = "elevation"
COL_TIME = "time"

# Surface level is typically reported near 1000-1013 hPa.
# We use this threshold to identify surface observations.
SURFACE_PRESSURE_MIN_MB = 900.0

# Default filename pattern: {station_id}_{YYYYMMDDhh}.csv
SOUNDING_FILENAME_PATTERN = re.compile(
    r"^(?P<station>[A-Z0-9]+)_(?P<date>\d{8})(?P<hour>\d{2})\.csv$"
)


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    if np.isnan(temp_c):
        return np.nan
    return temp_c * 9.0 / 5.0 + 32.0


def parse_sounding_csv(path: str) -> pd.DataFrame:
    """Load a single IGRA sounding CSV file as saved by soundings_collection.py.

    Parameters
    ----------
    path : str
        Path to the CSV file produced by Siphon's IGRAUpperAir.request_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: pressure, height, temperature, dewpoint,
        direction, speed, u_wind, v_wind, station, station_number,
        latitude, longitude, elevation, time.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV is empty or missing required columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sounding CSV not found: {path}")

    df = pd.read_csv(path, na_values=["", "NA", "NaN"])

    if df.empty:
        raise ValueError(f"Sounding CSV is empty: {path}")

    required_cols = {COL_PRESSURE, COL_TEMPERATURE}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Sounding CSV missing required columns {missing}: {path}"
        )

    # Ensure numeric types for key columns
    numeric_cols = [
        COL_PRESSURE, COL_HEIGHT, COL_TEMPERATURE, COL_DEWPOINT,
        COL_DIRECTION, COL_SPEED, COL_U_WIND, COL_V_WIND,
        COL_LATITUDE, COL_LONGITUDE, COL_ELEVATION,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse time column if present
    if COL_TIME in df.columns:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")

    return df


def _interpolate_to_level(
    df: pd.DataFrame,
    target_pressure: float,
    column: str,
) -> float:
    """Linearly interpolate a column value to a target pressure level.

    Pressure decreases with altitude, so we sort descending (surface first)
    and interpolate in log-pressure space for physical accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        Sounding data with at least 'pressure' and the target column.
    target_pressure : float
        Target pressure level in mb (hPa).
    column : str
        Column name to interpolate.

    Returns
    -------
    float
        Interpolated value, or NaN if interpolation is not possible.
    """
    valid = df.dropna(subset=[COL_PRESSURE, column]).copy()
    if valid.empty:
        return np.nan

    valid = valid.sort_values(COL_PRESSURE, ascending=False)
    pressures = valid[COL_PRESSURE].values
    values = valid[column].values

    # Check if target is outside the range of available data
    if target_pressure > pressures.max() or target_pressure < pressures.min():
        return np.nan

    # Interpolate in log-pressure space
    log_pressures = np.log(pressures)
    log_target = np.log(target_pressure)

    return float(np.interp(log_target, log_pressures[::-1], values[::-1]))


def extract_level_features(df: pd.DataFrame, level_mb: float) -> dict:
    """Extract meteorological features at a given pressure level.

    If the exact pressure level is not in the sounding, values are
    linearly interpolated in log-pressure space.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed sounding DataFrame from parse_sounding_csv().
    level_mb : float
        Target pressure level in millibars (hPa).

    Returns
    -------
    dict
        Feature dictionary with keys:
        - t{level}_c: temperature (Celsius)
        - t{level}_f: temperature (Fahrenheit)
        - dewpoint_{level}_c: dewpoint (Celsius)
        - wind_dir_{level}: wind direction (degrees)
        - wind_speed_{level}: wind speed (m/s or kt)
        - height_{level}_m: geopotential height (meters)
    """
    level_int = int(level_mb)
    features: dict = {}

    # Check for exact pressure match (within 0.5 hPa tolerance)
    exact = df.loc[(df[COL_PRESSURE] - level_mb).abs() < 0.5]

    if not exact.empty:
        row = exact.iloc[0]
        features[f"t{level_int}_c"] = (
            float(row[COL_TEMPERATURE]) if pd.notna(row.get(COL_TEMPERATURE)) else np.nan
        )
        features[f"dewpoint_{level_int}_c"] = (
            float(row[COL_DEWPOINT]) if COL_DEWPOINT in row.index and pd.notna(row.get(COL_DEWPOINT)) else np.nan
        )
        features[f"wind_dir_{level_int}"] = (
            float(row[COL_DIRECTION]) if COL_DIRECTION in row.index and pd.notna(row.get(COL_DIRECTION)) else np.nan
        )
        features[f"wind_speed_{level_int}"] = (
            float(row[COL_SPEED]) if COL_SPEED in row.index and pd.notna(row.get(COL_SPEED)) else np.nan
        )
        features[f"height_{level_int}_m"] = (
            float(row[COL_HEIGHT]) if COL_HEIGHT in row.index and pd.notna(row.get(COL_HEIGHT)) else np.nan
        )
    else:
        # Interpolate to target level
        features[f"t{level_int}_c"] = _interpolate_to_level(
            df, level_mb, COL_TEMPERATURE
        )
        features[f"dewpoint_{level_int}_c"] = (
            _interpolate_to_level(df, level_mb, COL_DEWPOINT)
            if COL_DEWPOINT in df.columns
            else np.nan
        )
        features[f"wind_dir_{level_int}"] = (
            _interpolate_to_level(df, level_mb, COL_DIRECTION)
            if COL_DIRECTION in df.columns
            else np.nan
        )
        features[f"wind_speed_{level_int}"] = (
            _interpolate_to_level(df, level_mb, COL_SPEED)
            if COL_SPEED in df.columns
            else np.nan
        )
        features[f"height_{level_int}_m"] = (
            _interpolate_to_level(df, level_mb, COL_HEIGHT)
            if COL_HEIGHT in df.columns
            else np.nan
        )

    # Add Fahrenheit conversion for temperature
    features[f"t{level_int}_f"] = celsius_to_fahrenheit(
        features[f"t{level_int}_c"]
    )

    return features


def _get_surface_temperature(df: pd.DataFrame) -> float:
    """Extract surface temperature from the sounding.

    Surface is defined as the observation with the highest pressure
    above SURFACE_PRESSURE_MIN_MB.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed sounding data.

    Returns
    -------
    float
        Surface temperature in Celsius, or NaN if not available.
    """
    valid = df.dropna(subset=[COL_PRESSURE, COL_TEMPERATURE])
    surface = valid.loc[valid[COL_PRESSURE] >= SURFACE_PRESSURE_MIN_MB]
    if surface.empty:
        return np.nan
    # Highest pressure = closest to surface
    idx = surface[COL_PRESSURE].idxmax()
    return float(surface.loc[idx, COL_TEMPERATURE])


def compute_stability_features(df: pd.DataFrame) -> dict:
    """Compute atmospheric stability indicators from a sounding profile.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed sounding DataFrame with pressure and temperature columns.

    Returns
    -------
    dict
        - t_surface_c: surface temperature (Celsius)
        - t_surface_f: surface temperature (Fahrenheit)
        - stability_index: T850 - T_surface (negative = unstable)
        - lapse_rate_850_500: (T850 - T500) / (height500 - height850)
          in C/km. Positive = temperature decreasing with height.
    """
    features: dict = {}

    t_surface = _get_surface_temperature(df)
    features["t_surface_c"] = t_surface
    features["t_surface_f"] = celsius_to_fahrenheit(t_surface)

    # T850
    t850 = _interpolate_to_level(df, 850.0, COL_TEMPERATURE)

    # Stability index: T850 - T_surface
    if np.isnan(t850) or np.isnan(t_surface):
        features["stability_index"] = np.nan
    else:
        features["stability_index"] = t850 - t_surface

    # 850-500 mb lapse rate
    t500 = _interpolate_to_level(df, 500.0, COL_TEMPERATURE)
    h850 = _interpolate_to_level(df, 850.0, COL_HEIGHT) if COL_HEIGHT in df.columns else np.nan
    h500 = _interpolate_to_level(df, 500.0, COL_HEIGHT) if COL_HEIGHT in df.columns else np.nan

    if any(np.isnan(v) for v in [t850, t500, h850, h500]):
        features["lapse_rate_850_500"] = np.nan
    else:
        dz_km = (h500 - h850) / 1000.0
        if abs(dz_km) < 0.001:
            features["lapse_rate_850_500"] = np.nan
        else:
            # Positive lapse rate = temperature decreasing with height
            features["lapse_rate_850_500"] = (t850 - t500) / dz_km

    return features


def process_single_sounding(
    path: str,
    levels_mb: list[float],
) -> dict:
    """Process one sounding CSV file into a flat feature dictionary.

    Combines level-specific features and stability features into a
    single dict suitable for building a DataFrame row.

    Parameters
    ----------
    path : str
        Path to a sounding CSV file.
    levels_mb : list[float]
        Pressure levels to extract features for (e.g., [850.0, 500.0]).

    Returns
    -------
    dict
        Flat dictionary of all extracted features, including:
        - Per-level: t{level}_c, t{level}_f, dewpoint_{level}_c,
          wind_dir_{level}, wind_speed_{level}, height_{level}_m
        - Stability: t_surface_c, t_surface_f, stability_index,
          lapse_rate_850_500
        - Metadata: file_path
    """
    df = parse_sounding_csv(path)
    features: dict = {"file_path": path}

    # Extract features at each requested pressure level
    for level in levels_mb:
        level_feats = extract_level_features(df, level)
        features.update(level_feats)

    # Compute stability features
    stability_feats = compute_stability_features(df)
    features.update(stability_feats)

    return features


def _parse_filename_metadata(filename: str) -> dict:
    """Extract station ID, date, and hour from sounding filename.

    Expected format: {station_id}_{YYYYMMDDhh}.csv

    Parameters
    ----------
    filename : str
        Basename of the sounding CSV file.

    Returns
    -------
    dict
        Keys: station_id, date (datetime.date), hour (int).
        Empty dict if filename doesn't match expected pattern.
    """
    match = SOUNDING_FILENAME_PATTERN.match(filename)
    if not match:
        return {}

    date_str = match.group("date")
    hour_str = match.group("hour")
    try:
        date = pd.Timestamp(date_str).date()
        hour = int(hour_str)
    except (ValueError, TypeError):
        return {}

    return {
        "station_id": match.group("station"),
        "date": date,
        "hour": hour,
    }


def aggregate_soundings_daily(
    sounding_dir: str,
    station_id: str,
    levels_mb: list[float],
) -> pd.DataFrame:
    """Process all sounding CSVs in a directory into a daily DataFrame.

    Scans for files matching {station_id}_{YYYYMMDDhh}.csv, processes
    each sounding, and assembles a DataFrame indexed by (date, hour).

    Parameters
    ----------
    sounding_dir : str
        Directory containing raw sounding CSV files.
    station_id : str
        IGRA station identifier (e.g., 'USM00072501').
    levels_mb : list[float]
        Pressure levels to extract (e.g., [850.0, 500.0]).

    Returns
    -------
    pd.DataFrame
        Daily sounding features with columns:
        date, hour, t850_c, t500_c, t_surface_c, wind_dir_850,
        wind_speed_850, height_850_m, height_500_m, stability_index,
        lapse_rate_850_500, etc.
    """
    pattern = os.path.join(sounding_dir, f"{station_id}_*.csv")
    # Exclude header JSON files
    csv_files = [
        f for f in sorted(glob.glob(pattern))
        if not f.endswith(".header.json")
    ]

    if not csv_files:
        LOGGER.warning(
            "No sounding files found for station %s in %s",
            station_id, sounding_dir,
        )
        return pd.DataFrame()

    rows: list[dict] = []
    processed = 0
    errors = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        metadata = _parse_filename_metadata(filename)
        if not metadata:
            LOGGER.warning("Skipping file with unrecognized name: %s", filename)
            continue

        try:
            features = process_single_sounding(filepath, levels_mb)
            features["date"] = metadata["date"]
            features["hour"] = metadata["hour"]
            features["station_id"] = metadata["station_id"]
            rows.append(features)
            processed += 1
        except (ValueError, FileNotFoundError) as exc:
            LOGGER.warning("Failed to process %s: %s", filepath, exc)
            errors += 1

    LOGGER.info(
        "Processed %d soundings (%d errors) for station %s",
        processed, errors, station_id,
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date and hour
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    # Drop the file_path column from output
    if "file_path" in df.columns:
        df = df.drop(columns=["file_path"])

    return df


def run_soundings_preprocessing(
    sounding_dir: str = config.IGRA_RAW_DIR,
    output_dir: str = config.IGRA_DAILY_DIR,
    station_id: str = config.IGRA_STATION_ID,
    levels_mb: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Main pipeline: process raw IGRA sounding files into daily features.

    Parameters
    ----------
    sounding_dir : str
        Directory containing raw sounding CSV files.
    output_dir : str
        Directory to write the processed daily CSV output.
    station_id : str
        IGRA station identifier.
    levels_mb : list[float], optional
        Pressure levels to extract. Defaults to config.IGRA_LEVELS_MB.

    Returns
    -------
    pd.DataFrame
        Aggregated daily sounding features.
    """
    if levels_mb is None:
        levels_mb = list(config.IGRA_LEVELS_MB)

    LOGGER.info(
        "Starting soundings preprocessing: station=%s, levels=%s",
        station_id, levels_mb,
    )

    daily = aggregate_soundings_daily(
        sounding_dir=sounding_dir,
        station_id=station_id,
        levels_mb=levels_mb,
    )

    if daily.empty:
        LOGGER.warning("No sounding data processed. Output is empty.")
        return daily

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{station_id}_soundings_daily.csv")
    daily.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved %d sounding records to %s", len(daily), output_path,
    )

    return daily


def main() -> None:
    """CLI entry point for soundings preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess IGRA sounding data into daily features."
    )
    parser.add_argument(
        "--sounding-dir", default=config.IGRA_RAW_DIR,
        help="Directory containing raw IGRA sounding CSVs.",
    )
    parser.add_argument(
        "--output-dir", default=config.IGRA_DAILY_DIR,
        help="Directory for processed daily output.",
    )
    parser.add_argument(
        "--station-id", default=config.IGRA_STATION_ID,
        help="IGRA station identifier.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=float,
        default=config.IGRA_LEVELS_MB,
        help="Pressure levels in mb to extract (e.g., 850 500).",
    )
    args = parser.parse_args()

    run_soundings_preprocessing(
        sounding_dir=args.sounding_dir,
        output_dir=args.output_dir,
        station_id=args.station_id,
        levels_mb=args.levels,
    )


if __name__ == "__main__":
    main()
