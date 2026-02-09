"""
NWP Preprocessing Pipeline.

Extracts daily NYC grid-point features from downloaded NWP GRIB files
(GFS/GEFS), computes derived meteorological variables, and aligns
NWP forecasts with surface observations for the synthesis layer.

Key outputs:
  - Parsed GRIB variables at the nearest grid point to Central Park
  - Derived wind speed/direction from U/V components
  - NWP bias relative to observations (rolling 7-day window)
  - Ensemble spread placeholder for GEFS
"""

from __future__ import annotations

import glob
import logging
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

import config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LAT = config.TARGET_LAT   # 40.7789
DEFAULT_LON = config.TARGET_LON   # -73.9692

# Conversion factors
MS_TO_KNOTS = 1.94384
K_TO_C_OFFSET = 273.15

# GRIB variable short-name -> friendly column mapping
GRIB_VAR_MAP = {
    "tmax": "tmax_2m_k",       # 2-m TMAX (Kelvin)
    "t": "tmp_850_k",          # 850-mb temperature (Kelvin)
    "u10": "ugrd_10m",         # 10-m U-wind (m/s)
    "v10": "vgrd_10m",         # 10-m V-wind (m/s)
    "tcc": "tcdc_pct",         # Total cloud cover (0-100%)
    "prmsl": "mslp_pa",        # Mean sea-level pressure (Pa)
    "tp": "apcp_m",            # Total precipitation (m)
    # Alternative short names used by different GRIB editions
    "2t": "tmp_2m_k",
    "mx2t": "tmax_2m_k",
    "10u": "ugrd_10m",
    "10v": "vgrd_10m",
    "msl": "mslp_pa",
}


def _require_cfgrib():
    """Check that cfgrib and xarray are available; raise ImportError if not."""
    try:
        import cfgrib  # noqa: F401
        import xarray  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'cfgrib' and 'xarray' packages are required for GRIB parsing. "
            "Install via: pip install cfgrib xarray"
        ) from exc


def _kelvin_to_fahrenheit(k: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (k - K_TO_C_OFFSET) * 9.0 / 5.0 + 32.0


def _kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - K_TO_C_OFFSET


def _wind_speed(u: float, v: float) -> float:
    """Compute wind speed from U and V components (same units)."""
    return math.sqrt(u * u + v * v)


def _wind_direction(u: float, v: float) -> float:
    """Compute meteorological wind direction from U/V in degrees.

    Returns the direction the wind is coming FROM (meteorological convention).
    Calm winds (u == v == 0) return 0.0.
    """
    if u == 0.0 and v == 0.0:
        return 0.0
    return (270.0 - math.degrees(math.atan2(v, u))) % 360.0


def _find_nearest_index(coords: np.ndarray, target: float) -> int:
    """Return the index of the value in *coords* nearest to *target*."""
    return int(np.argmin(np.abs(coords - target)))


# ---------------------------------------------------------------------------
# Primary GRIB parsing (cfgrib + xarray)
# ---------------------------------------------------------------------------


def parse_grib_file(
    path: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """Extract NWP variables from a GRIB file at the nearest grid point.

    Parameters
    ----------
    path : str
        Path to a GRIB/GRIB2 file.
    lat : float
        Target latitude (positive north).
    lon : float
        Target longitude (negative west).

    Returns
    -------
    dict
        Keys: tmax_2m_f, tmp_850_c, ugrd_10m, vgrd_10m,
              wind_speed_10m_kt, wind_dir_10m_deg, cloud_cover_pct,
              mslp_mb, precip_mm, forecast_hour.
        Values may be ``np.nan`` for missing fields.
    """
    _require_cfgrib()
    import xarray as xr

    result = {
        "tmax_2m_f": np.nan,
        "tmp_850_c": np.nan,
        "ugrd_10m": np.nan,
        "vgrd_10m": np.nan,
        "wind_speed_10m_kt": np.nan,
        "wind_dir_10m_deg": np.nan,
        "cloud_cover_pct": np.nan,
        "mslp_mb": np.nan,
        "precip_mm": np.nan,
        "forecast_hour": np.nan,
    }

    # cfgrib may return multiple datasets for different level types
    try:
        datasets = xr.open_datasets(path, engine="cfgrib")
    except Exception:
        # Fallback: try opening as a single dataset
        try:
            datasets = [xr.open_dataset(path, engine="cfgrib")]
        except Exception as exc:
            LOGGER.warning("Cannot open GRIB file %s: %s", path, exc)
            return result

    # Normalize longitude for the lookup (GRIB files often use 0-360)
    lon_lookup = lon % 360 if lon < 0 else lon

    for ds in datasets:
        # Determine the nearest grid point
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"

        if lat_dim not in ds.dims or lon_dim not in ds.dims:
            continue

        lat_vals = ds[lat_dim].values
        lon_vals = ds[lon_dim].values

        # If GRIB uses 0-360 but our target is negative, convert
        if lon_vals.min() >= 0 and lon < 0:
            actual_lon = lon_lookup
        else:
            actual_lon = lon

        lat_idx = _find_nearest_index(lat_vals, lat)
        lon_idx = _find_nearest_index(lon_vals, actual_lon)

        sel = {lat_dim: lat_idx, lon_dim: lon_idx}

        for var_name in ds.data_vars:
            short = str(var_name).lower()

            if short in ("tmax", "mx2t"):
                val = float(ds[var_name].isel(**sel).values)
                result["tmax_2m_f"] = _kelvin_to_fahrenheit(val)
            elif short == "t" and "isobaricInhPa" in ds[var_name].dims:
                # 850-mb temperature
                pressure_vals = ds["isobaricInhPa"].values
                p_idx = _find_nearest_index(pressure_vals, 850.0)
                val = float(
                    ds[var_name].isel(**sel, isobaricInhPa=p_idx).values
                )
                result["tmp_850_c"] = _kelvin_to_celsius(val)
            elif short in ("u10", "10u"):
                result["ugrd_10m"] = float(ds[var_name].isel(**sel).values)
            elif short in ("v10", "10v"):
                result["vgrd_10m"] = float(ds[var_name].isel(**sel).values)
            elif short == "tcc":
                result["cloud_cover_pct"] = float(
                    ds[var_name].isel(**sel).values
                )
            elif short in ("prmsl", "msl"):
                val = float(ds[var_name].isel(**sel).values)
                result["mslp_mb"] = val / 100.0  # Pa -> mb (hPa)
            elif short == "tp":
                val = float(ds[var_name].isel(**sel).values)
                result["precip_mm"] = val * 1000.0  # m -> mm

        # Extract forecast hour from dataset attributes or step variable
        if "step" in ds.coords:
            step = ds["step"].values
            if hasattr(step, "astype"):
                # timedelta64 -> hours
                hours = step.astype("timedelta64[h]").astype(float)
                result["forecast_hour"] = float(hours)
            else:
                result["forecast_hour"] = float(step)

    # Compute derived wind fields from U/V
    u = result["ugrd_10m"]
    v = result["vgrd_10m"]
    if not (np.isnan(u) or np.isnan(v)):
        result["wind_speed_10m_kt"] = _wind_speed(u, v) * MS_TO_KNOTS
        result["wind_dir_10m_deg"] = _wind_direction(u, v)

    return result


# ---------------------------------------------------------------------------
# Fallback GRIB parsing (Herbie JSON/CSV sidecar files)
# ---------------------------------------------------------------------------


def parse_grib_fallback(path: str) -> dict:
    """Attempt to parse NWP data from Herbie sidecar files.

    Herbie sometimes saves a .json or .csv file alongside the GRIB.
    This fallback looks for those files and extracts available fields.

    Returns
    -------
    dict
        Same key structure as :func:`parse_grib_file`, with np.nan for
        any field not found.
    """
    result = {
        "tmax_2m_f": np.nan,
        "tmp_850_c": np.nan,
        "ugrd_10m": np.nan,
        "vgrd_10m": np.nan,
        "wind_speed_10m_kt": np.nan,
        "wind_dir_10m_deg": np.nan,
        "cloud_cover_pct": np.nan,
        "mslp_mb": np.nan,
        "precip_mm": np.nan,
        "forecast_hour": np.nan,
    }

    import json

    # Try JSON sidecar
    json_path = path.replace(".grib2", ".json").replace(".grb2", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "tmax_2m" in meta:
                result["tmax_2m_f"] = _kelvin_to_fahrenheit(
                    float(meta["tmax_2m"])
                )
            if "tmp_850" in meta:
                result["tmp_850_c"] = _kelvin_to_celsius(
                    float(meta["tmp_850"])
                )
            if "ugrd_10m" in meta:
                result["ugrd_10m"] = float(meta["ugrd_10m"])
            if "vgrd_10m" in meta:
                result["vgrd_10m"] = float(meta["vgrd_10m"])
            if "tcdc" in meta:
                result["cloud_cover_pct"] = float(meta["tcdc"])
            if "mslp" in meta:
                result["mslp_mb"] = float(meta["mslp"]) / 100.0
            if "apcp" in meta:
                result["precip_mm"] = float(meta["apcp"]) * 1000.0
            if "forecast_hour" in meta:
                result["forecast_hour"] = float(meta["forecast_hour"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            LOGGER.warning("Failed to parse JSON sidecar %s: %s", json_path, exc)

    # Try CSV sidecar
    csv_path = path.replace(".grib2", ".csv").replace(".grb2", ".csv")
    if os.path.exists(csv_path) and np.isnan(result["tmax_2m_f"]):
        try:
            df = pd.read_csv(csv_path)
            if "tmax_2m" in df.columns and len(df) > 0:
                result["tmax_2m_f"] = _kelvin_to_fahrenheit(
                    float(df["tmax_2m"].iloc[0])
                )
            if "tmp_850" in df.columns and len(df) > 0:
                result["tmp_850_c"] = _kelvin_to_celsius(
                    float(df["tmp_850"].iloc[0])
                )
        except Exception as exc:
            LOGGER.warning("Failed to parse CSV sidecar %s: %s", csv_path, exc)

    # Derive wind from components
    u = result["ugrd_10m"]
    v = result["vgrd_10m"]
    if not (np.isnan(u) or np.isnan(v)):
        result["wind_speed_10m_kt"] = _wind_speed(u, v) * MS_TO_KNOTS
        result["wind_dir_10m_deg"] = _wind_direction(u, v)

    return result


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


def _extract_date_from_filename(filename: str) -> Optional[pd.Timestamp]:
    """Try to extract a date from a GRIB filename.

    Expects patterns like ``gfs_20220101_f024.grib2`` or ``20220101.grb2``.
    """
    import re

    basename = os.path.basename(filename)
    match = re.search(r"(\d{8})", basename)
    if match:
        try:
            return pd.Timestamp(match.group(1))
        except ValueError:
            return None
    return None


def _extract_fxx_from_filename(filename: str) -> int:
    """Try to extract the forecast hour from a GRIB filename.

    Looks for ``f024``, ``fxx024``, etc. Defaults to 24 if not found.
    """
    import re

    basename = os.path.basename(filename)
    match = re.search(r"f(?:xx)?(\d{2,3})", basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 24


def process_nwp_directory(
    nwp_dir: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """Scan a directory of GRIB files and combine into a DataFrame.

    Parameters
    ----------
    nwp_dir : str
        Directory containing GRIB/GRIB2 files.
    lat : float
        Target latitude.
    lon : float
        Target longitude.

    Returns
    -------
    pd.DataFrame
        Columns: date, fxx, tmax_2m_f, tmp_850_c, wind_speed_10m_kt,
        wind_dir_10m_deg, cloud_cover_pct, mslp_mb, precip_mm.
    """
    columns = [
        "date",
        "fxx",
        "tmax_2m_f",
        "tmp_850_c",
        "wind_speed_10m_kt",
        "wind_dir_10m_deg",
        "cloud_cover_pct",
        "mslp_mb",
        "precip_mm",
    ]

    if not os.path.isdir(nwp_dir):
        LOGGER.warning("NWP directory does not exist: %s", nwp_dir)
        return pd.DataFrame(columns=columns)

    # Collect GRIB files
    patterns = ["*.grib2", "*.grb2", "*.grib", "*.grb"]
    grib_files: list[str] = []
    for pat in patterns:
        grib_files.extend(glob.glob(os.path.join(nwp_dir, pat)))
        grib_files.extend(glob.glob(os.path.join(nwp_dir, "**", pat), recursive=True))
    grib_files = sorted(set(grib_files))

    if not grib_files:
        LOGGER.info("No GRIB files found in %s", nwp_dir)
        return pd.DataFrame(columns=columns)

    rows = []
    for filepath in grib_files:
        date = _extract_date_from_filename(filepath)
        fxx = _extract_fxx_from_filename(filepath)

        try:
            parsed = parse_grib_file(filepath, lat=lat, lon=lon)
        except ImportError:
            # cfgrib not available, try fallback
            LOGGER.info("cfgrib unavailable; using fallback for %s", filepath)
            parsed = parse_grib_fallback(filepath)
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", filepath, exc)
            continue

        # Use forecast_hour from parsed data if filename didn't provide it
        if parsed.get("forecast_hour") is not None and not np.isnan(
            parsed["forecast_hour"]
        ):
            fxx = int(parsed["forecast_hour"])

        row = {
            "date": date,
            "fxx": fxx,
            "tmax_2m_f": parsed.get("tmax_2m_f", np.nan),
            "tmp_850_c": parsed.get("tmp_850_c", np.nan),
            "wind_speed_10m_kt": parsed.get("wind_speed_10m_kt", np.nan),
            "wind_dir_10m_deg": parsed.get("wind_dir_10m_deg", np.nan),
            "cloud_cover_pct": parsed.get("cloud_cover_pct", np.nan),
            "mslp_mb": parsed.get("mslp_mb", np.nan),
            "precip_mm": parsed.get("precip_mm", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty and "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------


def compute_nwp_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived NWP features to the DataFrame.

    New columns:
      - nwp_tmax_change: TMAX change from previous day's forecast
      - nwp_wind_chill: wind chill index (°F) when applicable
      - nwp_ensemble_spread: placeholder (NaN for single-model GFS)

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`process_nwp_directory`.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional derived columns.
    """
    if df.empty:
        df["nwp_tmax_change"] = pd.Series(dtype=float)
        df["nwp_wind_chill"] = pd.Series(dtype=float)
        df["nwp_ensemble_spread"] = pd.Series(dtype=float)
        return df

    out = df.copy()

    # Day-over-day TMAX change
    if "tmax_2m_f" in out.columns:
        out["nwp_tmax_change"] = out["tmax_2m_f"].diff()
    else:
        out["nwp_tmax_change"] = np.nan

    # Wind chill (NWS formula, valid for T <= 50°F and wind >= 3 mph)
    # Using knots for wind speed; convert to mph for formula: 1 kt = 1.15078 mph
    kt_to_mph = 1.15078
    if "tmax_2m_f" in out.columns and "wind_speed_10m_kt" in out.columns:
        t = out["tmax_2m_f"]
        v_mph = out["wind_speed_10m_kt"] * kt_to_mph
        wc = (
            35.74
            + 0.6215 * t
            - 35.75 * v_mph.pow(0.16)
            + 0.4275 * t * v_mph.pow(0.16)
        )
        # Wind chill only valid when T <= 50°F and wind >= 3 mph
        valid_mask = (t <= 50.0) & (v_mph >= 3.0)
        out["nwp_wind_chill"] = np.where(valid_mask, wc, np.nan)
    else:
        out["nwp_wind_chill"] = np.nan

    # Ensemble spread: NaN placeholder for single-model GFS data
    out["nwp_ensemble_spread"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Alignment with observations
# ---------------------------------------------------------------------------


def align_nwp_with_observations(
    nwp_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    target_col: str = "TMAX",
) -> pd.DataFrame:
    """Merge NWP forecasts with observation data and compute bias metrics.

    Parameters
    ----------
    nwp_df : pd.DataFrame
        NWP forecast DataFrame with a ``date`` column.
    obs_df : pd.DataFrame
        Observation DataFrame with a ``date`` column and *target_col*.
    target_col : str
        Name of the observed TMAX column in *obs_df*.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with additional columns:
        - ``obs_tmax``: observed TMAX
        - ``nwp_bias``: nwp_tmax - observed_tmax
        - ``nwp_bias_7d``: rolling 7-day mean bias
    """
    if nwp_df.empty or obs_df.empty:
        out = nwp_df.copy()
        out["obs_tmax"] = pd.Series(dtype=float)
        out["nwp_bias"] = pd.Series(dtype=float)
        out["nwp_bias_7d"] = pd.Series(dtype=float)
        return out

    # Ensure date columns are datetime
    nwp = nwp_df.copy()
    obs = obs_df.copy()
    nwp["date"] = pd.to_datetime(nwp["date"])
    obs["date"] = pd.to_datetime(obs["date"])

    merged = nwp.merge(
        obs[["date", target_col]].rename(columns={target_col: "obs_tmax"}),
        on="date",
        how="left",
    )

    # NWP bias: forecast minus observed
    if "tmax_2m_f" in merged.columns:
        merged["nwp_bias"] = merged["tmax_2m_f"] - merged["obs_tmax"]
    else:
        merged["nwp_bias"] = np.nan

    # Rolling 7-day mean bias
    merged["nwp_bias_7d"] = (
        merged["nwp_bias"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_nwp_preprocessing(
    nwp_dir: str = config.NWP_RAW_DIR,
    output_dir: Optional[str] = None,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """Run the full NWP preprocessing pipeline.

    Steps:
      1. Scan *nwp_dir* for GRIB files and extract grid-point values.
      2. Compute derived features (TMAX change, wind chill, spread).
      3. Save processed CSV to *output_dir*.

    Parameters
    ----------
    nwp_dir : str
        Directory containing raw GRIB files.
    output_dir : str or None
        Directory for output CSV. Defaults to ``config.NWP_DAILY_DIR``
        (set in config.py).
    lat : float
        Target latitude.
    lon : float
        Target longitude.

    Returns
    -------
    pd.DataFrame
        Processed NWP feature DataFrame.
    """
    if output_dir is None:
        output_dir = getattr(config, "NWP_DAILY_DIR", os.path.join(
            config.PROCESSED_DATA_DIR, "nwp_daily"
        ))

    LOGGER.info("Starting NWP preprocessing from %s", nwp_dir)

    # Step 1: Parse GRIB files
    df = process_nwp_directory(nwp_dir, lat=lat, lon=lon)
    LOGGER.info("Parsed %d GRIB files", len(df))

    if df.empty:
        LOGGER.warning("No NWP data extracted — skipping derived features")
        return df

    # Step 2: Derived features
    df = compute_nwp_derived_features(df)
    LOGGER.info("Added derived features: nwp_tmax_change, nwp_wind_chill, nwp_ensemble_spread")

    # Step 3: Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nwp_daily_features.csv")
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved NWP daily features to %s (%d rows)", output_path, len(df))

    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess NWP GRIB files into daily features."
    )
    parser.add_argument(
        "--nwp-dir",
        default=config.NWP_RAW_DIR,
        help="Directory containing raw GRIB files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for processed CSV.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=DEFAULT_LAT,
        help="Target latitude.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=DEFAULT_LON,
        help="Target longitude.",
    )
    args = parser.parse_args()

    run_nwp_preprocessing(
        nwp_dir=args.nwp_dir,
        output_dir=args.output_dir,
        lat=args.lat,
        lon=args.lon,
    )


if __name__ == "__main__":
    main()
