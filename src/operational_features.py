"""
Operational Feature Engineering Module for NYC Temperature Prediction.

Combines GHCN station temperatures, ASOS hourly-derived daily features,
IGRA upper-air sounding features, and wind-conditioned composites into a
unified training-ready feature matrix.

Feature groups:
  1. Wind-conditioned composites (upwind/crosswind/downwind temperature,
     advection gradient, advection rate)
  2. Per-station ASOS operational features (dewpoint, pressure, clouds,
     wind persistence, evening wind direction)
  3. 850 mb sounding features (T850, 850 mb wind, stability, lapse rate)
  4. Existing V2 features (lagged TMAX/TMIN, cyclical dates, sectors, etc.)

All features respect the no-leakage constraint: only data from t-1 or
earlier is used for predicting at time t.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# ASOS daily columns that we pull per-station
ASOS_FEATURE_COLS = [
    "dewpoint_mean_f",
    "dewpoint_afternoon_f",
    "wind_speed_mean_mph",
    "wind_speed_max_mph",
    "wind_dir_mean_deg",
    "wind_dir_evening_deg",
    "slp_00z_mb",
    "slp_12z_mb",
    "slp_tendency_24h_mb",
    "cloud_fraction_low",
]

# Sounding columns we merge from the IGRA daily file
SOUNDING_FEATURE_COLS = [
    "t850_f",
    "t500_f",
    "wind_dir_850",
    "wind_speed_850",
    "stability_index",
    "lapse_rate_850_500",
    "t_surface_f",
]


# ============================================================================
# Station Metadata Helpers
# ============================================================================

def load_station_metadata(
    stations_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Load station metadata (bearing, distance) from stations.csv.

    Parameters
    ----------
    stations_csv : str, optional
        Path to stations.csv. Defaults to config.STATIONS_FILE.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station_id, name, latitude, longitude,
        distance_miles, direction, bearing_deg.
    """
    if stations_csv is None:
        stations_csv = config.STATIONS_FILE

    df = pd.read_csv(stations_csv)

    # Convert cardinal direction to numeric bearing (degrees)
    direction_to_bearing = {
        "N": 0.0,
        "NNE": 22.5,
        "NE": 45.0,
        "ENE": 67.5,
        "E": 90.0,
        "ESE": 112.5,
        "SE": 135.0,
        "SSE": 157.5,
        "S": 180.0,
        "SSW": 202.5,
        "SW": 225.0,
        "WSW": 247.5,
        "W": 270.0,
        "WNW": 292.5,
        "NW": 315.0,
        "NNW": 337.5,
        "Target": np.nan,
    }
    df["bearing_deg"] = df["direction"].map(direction_to_bearing)
    return df


def load_asos_station_mapping(
    mapping_csv: Optional[str] = None,
) -> dict[str, str]:
    """Load GHCN station_id -> ICAO mapping from asos_station_mapping.csv.

    Parameters
    ----------
    mapping_csv : str, optional
        Path to the mapping CSV. Defaults to data/asos_station_mapping.csv.

    Returns
    -------
    dict[str, str]
        Mapping of GHCN station_id to ICAO code for stations that have ASOS.
    """
    if mapping_csv is None:
        mapping_csv = os.path.join(config.DATA_DIR, "asos_station_mapping.csv")

    df = pd.read_csv(mapping_csv)
    available = df[df["asos_available"] == "yes"].dropna(subset=["icao"])
    return dict(zip(available["station_id"], available["icao"]))


# ============================================================================
# Wind-Conditioned Feature Computation
# ============================================================================

def compute_upwind_temperature(
    station_tmax: pd.DataFrame,
    wind_dir_deg: pd.Series,
    station_bearings: dict[str, float],
) -> pd.Series:
    """Compute upwind-weighted average temperature.

    For each day, weights stations by cos similarity between the prevailing
    wind direction and the station's bearing from Central Park. Stations
    that are upwind (bearing ~ wind direction) get higher weight.

    Parameters
    ----------
    station_tmax : pd.DataFrame
        DataFrame where each column is a station's TMAX (station_id as
        column name), indexed by date.
    wind_dir_deg : pd.Series
        Daily mean wind direction in degrees (0=N, 90=E, etc.), same index.
    station_bearings : dict[str, float]
        Mapping of station_id to bearing (degrees) from Central Park.

    Returns
    -------
    pd.Series
        Upwind-weighted temperature for each day. NaN if no valid weights.
    """
    result = pd.Series(np.nan, index=station_tmax.index, dtype=float)

    for idx in station_tmax.index:
        wd = wind_dir_deg.get(idx, np.nan)
        if pd.isna(wd):
            continue

        wd_rad = np.deg2rad(wd)
        weights = {}
        for sid, bearing in station_bearings.items():
            if sid not in station_tmax.columns:
                continue
            val = station_tmax.at[idx, sid]
            if pd.isna(val) or pd.isna(bearing):
                continue
            b_rad = np.deg2rad(bearing)
            w = max(0.0, np.cos(wd_rad - b_rad))
            if w > 0:
                weights[sid] = (w, val)

        if weights:
            total_w = sum(w for w, _ in weights.values())
            if total_w > 0:
                result.at[idx] = sum(w * v for w, v in weights.values()) / total_w

    return result


def compute_crosswind_temperature(
    station_tmax: pd.DataFrame,
    wind_dir_deg: pd.Series,
    station_bearings: dict[str, float],
) -> pd.Series:
    """Compute crosswind-weighted average temperature.

    Weights stations by |sin(wind_dir - bearing)|. Stations perpendicular
    to the wind direction get the highest weight.

    Parameters
    ----------
    station_tmax : pd.DataFrame
        DataFrame of station TMAX values, indexed by date.
    wind_dir_deg : pd.Series
        Daily mean wind direction in degrees.
    station_bearings : dict[str, float]
        Station bearing mapping.

    Returns
    -------
    pd.Series
        Crosswind-weighted temperature for each day.
    """
    result = pd.Series(np.nan, index=station_tmax.index, dtype=float)

    for idx in station_tmax.index:
        wd = wind_dir_deg.get(idx, np.nan)
        if pd.isna(wd):
            continue

        wd_rad = np.deg2rad(wd)
        weights = {}
        for sid, bearing in station_bearings.items():
            if sid not in station_tmax.columns:
                continue
            val = station_tmax.at[idx, sid]
            if pd.isna(val) or pd.isna(bearing):
                continue
            b_rad = np.deg2rad(bearing)
            w = abs(np.sin(wd_rad - b_rad))
            if w > 1e-10:
                weights[sid] = (w, val)

        if weights:
            total_w = sum(w for w, _ in weights.values())
            if total_w > 0:
                result.at[idx] = sum(w * v for w, v in weights.values()) / total_w

    return result


def compute_downwind_temperature(
    station_tmax: pd.DataFrame,
    wind_dir_deg: pd.Series,
    station_bearings: dict[str, float],
) -> pd.Series:
    """Compute downwind-weighted average temperature.

    Weights stations by max(0, -cos(wind_dir - bearing)). Stations that
    are downwind (bearing opposite to wind direction) get higher weight.

    Parameters
    ----------
    station_tmax : pd.DataFrame
        DataFrame of station TMAX values, indexed by date.
    wind_dir_deg : pd.Series
        Daily mean wind direction in degrees.
    station_bearings : dict[str, float]
        Station bearing mapping.

    Returns
    -------
    pd.Series
        Downwind-weighted temperature for each day.
    """
    result = pd.Series(np.nan, index=station_tmax.index, dtype=float)

    for idx in station_tmax.index:
        wd = wind_dir_deg.get(idx, np.nan)
        if pd.isna(wd):
            continue

        wd_rad = np.deg2rad(wd)
        weights = {}
        for sid, bearing in station_bearings.items():
            if sid not in station_tmax.columns:
                continue
            val = station_tmax.at[idx, sid]
            if pd.isna(val) or pd.isna(bearing):
                continue
            b_rad = np.deg2rad(bearing)
            w = max(0.0, -np.cos(wd_rad - b_rad))
            if w > 0:
                weights[sid] = (w, val)

        if weights:
            total_w = sum(w for w, _ in weights.values())
            if total_w > 0:
                result.at[idx] = sum(w * v for w, v in weights.values()) / total_w

    return result


def compute_upwind_gradient(
    upwind_temp: pd.Series,
    nyc_tmax_prev: pd.Series,
) -> pd.Series:
    """Compute upwind temperature gradient (advection signal).

    Parameters
    ----------
    upwind_temp : pd.Series
        Upwind-weighted temperature.
    nyc_tmax_prev : pd.Series
        NYC TMAX from the previous day (t-1).

    Returns
    -------
    pd.Series
        upwind_temp - nyc_tmax_prev. Positive means warmer air incoming.
    """
    return upwind_temp - nyc_tmax_prev


def compute_advection_rate(
    wind_speed_mean: pd.Series,
    upwind_gradient: pd.Series,
    mean_upwind_distance_mi: float,
) -> pd.Series:
    """Compute estimated temperature advection rate.

    advection_rate = wind_speed * upwind_gradient / mean_distance

    Parameters
    ----------
    wind_speed_mean : pd.Series
        Mean wind speed (mph).
    upwind_gradient : pd.Series
        Upwind temperature gradient (F).
    mean_upwind_distance_mi : float
        Mean distance of upwind stations (miles).

    Returns
    -------
    pd.Series
        Advection rate (F * mph / mi). NaN if distance is zero.
    """
    if mean_upwind_distance_mi <= 0:
        return pd.Series(np.nan, index=wind_speed_mean.index, dtype=float)
    return wind_speed_mean * upwind_gradient / mean_upwind_distance_mi


def compute_wind_conditioned_features(
    station_tmax: pd.DataFrame,
    wind_dir_deg: pd.Series,
    wind_speed_mean: pd.Series,
    nyc_tmax_prev: pd.Series,
    station_bearings: dict[str, float],
    station_distances: dict[str, float],
) -> pd.DataFrame:
    """Compute all wind-conditioned composite features.

    Parameters
    ----------
    station_tmax : pd.DataFrame
        Surrounding station TMAX values indexed by date.
    wind_dir_deg : pd.Series
        Daily mean wind direction in degrees.
    wind_speed_mean : pd.Series
        Daily mean wind speed in mph.
    nyc_tmax_prev : pd.Series
        NYC TMAX from the previous day.
    station_bearings : dict[str, float]
        Station ID to bearing (degrees from Central Park).
    station_distances : dict[str, float]
        Station ID to distance (miles from Central Park).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: upwind_temp, crosswind_temp, downwind_temp,
        upwind_gradient, advection_rate.
    """
    upwind = compute_upwind_temperature(station_tmax, wind_dir_deg, station_bearings)
    crosswind = compute_crosswind_temperature(station_tmax, wind_dir_deg, station_bearings)
    downwind = compute_downwind_temperature(station_tmax, wind_dir_deg, station_bearings)
    gradient = compute_upwind_gradient(upwind, nyc_tmax_prev)

    # Compute mean distance of stations with valid bearings
    valid_distances = [
        d for sid, d in station_distances.items()
        if sid in station_bearings and not pd.isna(station_bearings.get(sid))
        and d > 0
    ]
    mean_dist = float(np.mean(valid_distances)) if valid_distances else 0.0

    adv_rate = compute_advection_rate(wind_speed_mean, gradient, mean_dist)

    result = pd.DataFrame({
        "upwind_temp": upwind,
        "crosswind_temp": crosswind,
        "downwind_temp": downwind,
        "upwind_gradient": gradient,
        "advection_rate": adv_rate,
    }, index=station_tmax.index)

    return result


# ============================================================================
# ASOS Feature Loading
# ============================================================================

def load_asos_daily_for_station(
    station_id: str,
    asos_daily_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load ASOS daily aggregated features for a single station.

    Parameters
    ----------
    station_id : str
        GHCN station ID (e.g., 'USW00014734').
    asos_daily_dir : str, optional
        Directory containing {station_id}_asos_daily.csv files.
        Defaults to config.ASOS_DAILY_DIR.

    Returns
    -------
    pd.DataFrame or None
        DataFrame indexed by date with ASOS feature columns, or None
        if the file does not exist.
    """
    if asos_daily_dir is None:
        asos_daily_dir = config.ASOS_DAILY_DIR

    path = os.path.join(asos_daily_dir, f"{station_id}_asos_daily.csv")
    if not os.path.exists(path):
        logger.debug("No ASOS daily file for station %s at %s", station_id, path)
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def compute_dewpoint_depression(
    tmax_f: pd.Series,
    dewpoint_mean_f: pd.Series,
) -> pd.Series:
    """Compute dewpoint depression: T - Td.

    Parameters
    ----------
    tmax_f : pd.Series
        Daily maximum temperature (F).
    dewpoint_mean_f : pd.Series
        Daily mean dewpoint (F).

    Returns
    -------
    pd.Series
        Dewpoint depression (F). Higher = drier air.
    """
    return tmax_f - dewpoint_mean_f


def extract_asos_features_for_station(
    asos_daily: pd.DataFrame,
    station_id: str,
) -> pd.DataFrame:
    """Extract operational ASOS features for a single station.

    Extracts dewpoint, pressure, cloud, and wind features from the ASOS
    daily aggregated data. Also computes dewpoint depression.

    Parameters
    ----------
    asos_daily : pd.DataFrame
        ASOS daily DataFrame (output of asos_preprocessing), indexed by date.
    station_id : str
        Station identifier, used for column naming.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns named
        ``{station_id}_asos_{feature}``.
    """
    result = pd.DataFrame(index=asos_daily.index)

    # Core ASOS features
    feature_map = {
        "dewpoint_mean_f": "dewpoint_mean",
        "dewpoint_afternoon_f": "dewpoint_afternoon",
        "slp_00z_mb": "slp_00z",
        "slp_12z_mb": "slp_12z",
        "slp_tendency_24h_mb": "slp_tendency",
        "cloud_fraction_low": "cloud_fraction",
        "wind_dir_evening_deg": "wind_dir_evening",
    }

    for src_col, feat_name in feature_map.items():
        if src_col in asos_daily.columns:
            result[f"{station_id}_asos_{feat_name}"] = asos_daily[src_col]
        else:
            result[f"{station_id}_asos_{feat_name}"] = np.nan

    # Dewpoint depression
    if "tmax_f" in asos_daily.columns and "dewpoint_mean_f" in asos_daily.columns:
        result[f"{station_id}_asos_dewpoint_depression"] = compute_dewpoint_depression(
            asos_daily["tmax_f"], asos_daily["dewpoint_mean_f"]
        )
    else:
        result[f"{station_id}_asos_dewpoint_depression"] = np.nan

    return result


def load_all_asos_features(
    station_ids: list[str],
    asos_daily_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load and combine ASOS features for all specified stations.

    Parameters
    ----------
    station_ids : list[str]
        List of GHCN station IDs to load ASOS data for.
    asos_daily_dir : str, optional
        Directory containing ASOS daily CSVs.

    Returns
    -------
    pd.DataFrame
        Combined ASOS features indexed by date. Stations without ASOS
        data will have NaN columns.
    """
    frames = []
    loaded_count = 0

    for sid in station_ids:
        asos_df = load_asos_daily_for_station(sid, asos_daily_dir)
        if asos_df is not None:
            features = extract_asos_features_for_station(asos_df, sid)
            frames.append(features)
            loaded_count += 1
        else:
            logger.debug("No ASOS data for station %s — columns will be NaN", sid)

    if not frames:
        logger.warning("No ASOS data loaded for any station")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()

    logger.info(
        "Loaded ASOS features for %d/%d stations, %d columns",
        loaded_count, len(station_ids), combined.shape[1],
    )
    return combined


# ============================================================================
# Sounding Feature Loading
# ============================================================================

def load_sounding_daily(
    station_id: Optional[str] = None,
    igra_daily_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load daily sounding features from the IGRA preprocessed CSV.

    Parameters
    ----------
    station_id : str, optional
        IGRA station ID. Defaults to config.IGRA_STATION_ID.
    igra_daily_dir : str, optional
        Directory containing the daily sounding CSV.
        Defaults to config.IGRA_DAILY_DIR.

    Returns
    -------
    pd.DataFrame or None
        DataFrame indexed by date with sounding feature columns, or None
        if the file does not exist.
    """
    if station_id is None:
        station_id = config.IGRA_STATION_ID
    if igra_daily_dir is None:
        igra_daily_dir = config.IGRA_DAILY_DIR

    path = os.path.join(igra_daily_dir, f"{station_id}_soundings_daily.csv")
    if not os.path.exists(path):
        logger.warning("No sounding daily file at %s", path)
        return None

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def extract_sounding_features(
    sounding_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Extract relevant sounding features for model input.

    Selects the 12Z sounding (most representative for daytime convection
    and temperature forecasting). If 12Z is missing, falls back to 00Z.

    Parameters
    ----------
    sounding_daily : pd.DataFrame
        Raw sounding daily DataFrame with 'hour' column and feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with sounding features prefixed with
        ``sounding_``.
    """
    # Column mapping from sounding file to output names
    col_map = {
        "t850_f": "sounding_t850_f",
        "t500_f": "sounding_t500_f",
        "wind_dir_850": "sounding_wind_dir_850",
        "wind_speed_850": "sounding_wind_speed_850",
        "stability_index": "sounding_stability_index",
        "lapse_rate_850_500": "sounding_lapse_rate",
        "t_surface_f": "sounding_t_surface_f",
    }

    # Prefer 12Z soundings for daytime prediction
    if "hour" in sounding_daily.columns:
        sounding_12z = sounding_daily[sounding_daily["hour"] == 12].copy()
        sounding_00z = sounding_daily[sounding_daily["hour"] == 0].copy()

        # Use 12Z where available, fall back to 00Z
        if not sounding_12z.empty and not sounding_00z.empty:
            combined = sounding_12z.combine_first(sounding_00z)
        elif not sounding_12z.empty:
            combined = sounding_12z
        else:
            combined = sounding_00z
    else:
        # No hour column — use data as-is (deduplicate dates)
        combined = sounding_daily.copy()
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="first")]

    result = pd.DataFrame(index=combined.index)
    for src_col, out_col in col_map.items():
        if src_col in combined.columns:
            result[out_col] = combined[src_col]
        else:
            result[out_col] = np.nan

    return result


# ============================================================================
# Feature Matrix Builder
# ============================================================================

def _get_nyc_wind_from_asos(
    asos_daily_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load NYC Central Park ASOS wind data.

    Parameters
    ----------
    asos_daily_dir : str, optional
        Directory with ASOS daily files.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with wind_dir_mean_deg and wind_speed_mean_mph columns.
    """
    nyc_asos = load_asos_daily_for_station(config.TARGET_STATION, asos_daily_dir)
    if nyc_asos is None:
        # Try nearby LaGuardia as proxy
        nyc_asos = load_asos_daily_for_station("USW00014739", asos_daily_dir)
    if nyc_asos is None:
        logger.warning("No ASOS wind data available for NYC or LaGuardia")
        return None
    return nyc_asos


def build_operational_feature_matrix(
    merged_station_df: Optional[pd.DataFrame] = None,
    asos_daily_dir: Optional[str] = None,
    igra_daily_dir: Optional[str] = None,
    stations_csv: Optional[str] = None,
    include_wind_conditioned: bool = True,
    include_asos: bool = True,
    include_sounding: bool = True,
    station_ids: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full operational feature matrix.

    Combines existing GHCN-based station temperatures with ASOS
    operational features, sounding features, and wind-conditioned
    composites. All features are lagged to t-1 for prediction at t.

    Parameters
    ----------
    merged_station_df : pd.DataFrame, optional
        Pre-merged wide-format station TMAX DataFrame (indexed by date,
        columns are station IDs with TMAX values). If None, attempts
        to load from the existing preprocessing pipeline.
    asos_daily_dir : str, optional
        Directory containing ASOS daily CSVs.
    igra_daily_dir : str, optional
        Directory containing IGRA daily CSV.
    stations_csv : str, optional
        Path to stations.csv for metadata.
    include_wind_conditioned : bool
        Whether to include wind-conditioned composite features.
    include_asos : bool
        Whether to include per-station ASOS features.
    include_sounding : bool
        Whether to include 850mb sounding features.
    station_ids : list[str], optional
        List of surrounding station IDs. Defaults to config.SURROUNDING_STATIONS.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (feature_matrix, target) both indexed by date. All features use
        data from t-1 or earlier. Target is NYC TMAX at t.
    """
    if station_ids is None:
        station_ids = list(config.SURROUNDING_STATIONS.keys())

    # Load station metadata for bearings and distances
    station_meta = load_station_metadata(stations_csv)
    station_bearings: dict[str, float] = {}
    station_distances: dict[str, float] = {}
    for _, row in station_meta.iterrows():
        sid = row["station_id"]
        if sid != config.TARGET_STATION and not pd.isna(row.get("bearing_deg")):
            station_bearings[sid] = row["bearing_deg"]
            station_distances[sid] = row["distance_miles"]

    # Prepare the merged station DataFrame
    if merged_station_df is None:
        from src.data_preprocessing import load_all_stations, merge_stations
        station_data = load_all_stations()
        merged_station_df = merge_stations(station_data)

    # Extract target (NYC TMAX at day t)
    target_col = f"{config.TARGET_STATION}_TMAX"
    if target_col not in merged_station_df.columns:
        raise ValueError(f"Target column '{target_col}' not in data")
    target = merged_station_df[target_col].copy()
    target.name = "NYC_TMAX"

    # NYC TMAX at t-1 (for gradient and autoregressive)
    nyc_tmax_prev = target.shift(1)
    nyc_tmax_prev.name = "NYC_TMAX_prev"

    # Build station TMAX DataFrame for wind-conditioned features
    tmax_cols = {
        sid: f"{sid}_TMAX" for sid in station_ids
        if f"{sid}_TMAX" in merged_station_df.columns
    }
    station_tmax = merged_station_df[list(tmax_cols.values())].rename(
        columns={v: k for k, v in tmax_cols.items()}
    )

    feature_parts: list[pd.DataFrame] = []

    # --- Wind-conditioned features ---
    if include_wind_conditioned:
        nyc_asos = _get_nyc_wind_from_asos(asos_daily_dir)
        if nyc_asos is not None:
            wind_dir = nyc_asos["wind_dir_mean_deg"] if "wind_dir_mean_deg" in nyc_asos.columns else pd.Series(dtype=float)
            wind_speed = nyc_asos["wind_speed_mean_mph"] if "wind_speed_mean_mph" in nyc_asos.columns else pd.Series(dtype=float)

            # Align to station_tmax index
            wind_dir = wind_dir.reindex(station_tmax.index)
            wind_speed = wind_speed.reindex(station_tmax.index)
            nyc_prev_aligned = nyc_tmax_prev.reindex(station_tmax.index)

            wind_features = compute_wind_conditioned_features(
                station_tmax=station_tmax,
                wind_dir_deg=wind_dir,
                wind_speed_mean=wind_speed,
                nyc_tmax_prev=nyc_prev_aligned,
                station_bearings=station_bearings,
                station_distances=station_distances,
            )
            # Lag wind features by 1 day (use t-1 values at t)
            wind_features_lagged = wind_features.shift(1)
            wind_features_lagged.columns = [f"{c}_lag1" for c in wind_features_lagged.columns]
            feature_parts.append(wind_features_lagged)
            logger.info("Added %d wind-conditioned features", wind_features_lagged.shape[1])
        else:
            logger.warning("Skipping wind-conditioned features (no ASOS wind data)")

    # --- Per-station ASOS features ---
    if include_asos:
        asos_features = load_all_asos_features(station_ids, asos_daily_dir)
        if not asos_features.empty:
            # Align to main index
            asos_features = asos_features.reindex(merged_station_df.index)
            # Lag by 1 day
            asos_features_lagged = asos_features.shift(1)
            asos_features_lagged.columns = [f"{c}_lag1" for c in asos_features_lagged.columns]
            feature_parts.append(asos_features_lagged)
            logger.info("Added %d ASOS features", asos_features_lagged.shape[1])
        else:
            logger.warning("Skipping ASOS features (no data loaded)")

    # --- Sounding features ---
    if include_sounding:
        sounding_raw = load_sounding_daily(igra_daily_dir=igra_daily_dir)
        if sounding_raw is not None:
            sounding_features = extract_sounding_features(sounding_raw)
            # Align to main index
            sounding_features = sounding_features.reindex(merged_station_df.index)
            # Lag by 1 day
            sounding_features_lagged = sounding_features.shift(1)
            sounding_features_lagged.columns = [f"{c}_lag1" for c in sounding_features_lagged.columns]
            feature_parts.append(sounding_features_lagged)
            logger.info("Added %d sounding features", sounding_features_lagged.shape[1])
        else:
            logger.warning("Skipping sounding features (no data loaded)")

    # Combine all operational features
    if feature_parts:
        operational_features = pd.concat(feature_parts, axis=1)
        operational_features = operational_features.reindex(merged_station_df.index)
    else:
        operational_features = pd.DataFrame(index=merged_station_df.index)
        logger.warning("No operational features were added")

    # Drop rows where target is NaN or in the initial lag window
    valid_mask = target.notna()
    valid_mask.iloc[0] = False  # First row always invalid due to lag
    operational_features = operational_features[valid_mask]
    target = target[valid_mask]

    logger.info(
        "Built operational feature matrix: %d rows x %d columns",
        operational_features.shape[0], operational_features.shape[1],
    )

    return operational_features, target


# ============================================================================
# Feature Name and Group Registry
# ============================================================================

def get_feature_names(feature_df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature column names.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The feature matrix.

    Returns
    -------
    list[str]
        Ordered list of column names.
    """
    return list(feature_df.columns)


def get_feature_groups(feature_df: pd.DataFrame) -> dict[str, str]:
    """Return mapping of feature name to feature group.

    Groups are assigned based on column name patterns:
      - ``upwind_*``, ``crosswind_*``, ``downwind_*``,
        ``advection_*``, ``upwind_gradient*`` -> "wind_conditioned"
      - ``*_asos_*`` -> "asos_operational"
      - ``sounding_*`` -> "sounding"
      - ``*_TMAX_lag*``, ``*_TMIN_lag*`` -> "station_temperature"
      - ``sin_day``, ``cos_day`` -> "date_encoding"
      - ``NYC_TMAX_lag*`` -> "autoregressive"
      - ``sector_*`` -> "sector"
      - ``grad_*`` -> "sector_gradient"
      - ``diurnal_*`` -> "diurnal_range"
      - ``trend_*`` -> "trend"
      - everything else -> "other"

    Parameters
    ----------
    feature_df : pd.DataFrame
        The feature matrix.

    Returns
    -------
    dict[str, str]
        Mapping of feature column name to group name.
    """
    groups: dict[str, str] = {}
    for col in feature_df.columns:
        if any(col.startswith(p) for p in [
            "upwind_temp", "crosswind_temp", "downwind_temp",
            "upwind_gradient", "advection_rate",
        ]):
            groups[col] = "wind_conditioned"
        elif "_asos_" in col:
            groups[col] = "asos_operational"
        elif col.startswith("sounding_") or col.lstrip("_").startswith("sounding_"):
            groups[col] = "sounding"
        elif col.startswith("NYC_TMAX_lag"):
            groups[col] = "autoregressive"
        elif "sin_day" in col or "cos_day" in col:
            groups[col] = "date_encoding"
        elif col.startswith("sector_"):
            groups[col] = "sector"
        elif col.startswith("grad_"):
            groups[col] = "sector_gradient"
        elif col.startswith("diurnal_"):
            groups[col] = "diurnal_range"
        elif col.startswith("trend_"):
            groups[col] = "trend"
        elif "_TMAX_lag" in col or "_TMIN_lag" in col:
            groups[col] = "station_temperature"
        else:
            groups[col] = "other"
    return groups


# ============================================================================
# Convenience: merge operational features with V2 features
# ============================================================================

def merge_with_v2_features(
    v2_features: pd.DataFrame,
    operational_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge V2 preprocessed features with operational features.

    Joins on the date index, keeping all rows from v2_features. Columns
    from operational_features that are missing dates will be NaN.

    Parameters
    ----------
    v2_features : pd.DataFrame
        Features from data_preprocessing_v2 (already lagged and aligned).
    operational_features : pd.DataFrame
        Operational features from build_operational_feature_matrix().

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with all columns from both inputs.
    """
    # Avoid duplicate columns
    overlap = set(v2_features.columns) & set(operational_features.columns)
    if overlap:
        logger.warning(
            "Dropping %d overlapping columns from operational features: %s",
            len(overlap), sorted(overlap),
        )
        operational_features = operational_features.drop(columns=list(overlap))

    combined = v2_features.join(operational_features, how="left")
    logger.info(
        "Merged features: %d V2 + %d operational = %d total columns",
        v2_features.shape[1],
        operational_features.shape[1],
        combined.shape[1],
    )
    return combined
