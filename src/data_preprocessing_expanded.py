"""
Expanded Data Preprocessing Module for NYC Temperature Prediction.

Extends the V2 preprocessing pipeline to support:
  - Variable station counts (any subset of available stations)
  - Missingness masking (binary 0/1 per station per lag, BEFORE imputation)
  - Auto-selection of N closest stations with directional diversity
  - Graceful handling of missing station CSVs

This module does NOT modify any existing source files. It imports helpers
from data_preprocessing.py and data_preprocessing_v2.py as needed.

Key functions:
  - discover_available_stations()   -- find CSVs in data/raw/
  - compute_station_distances()     -- haversine distances to Central Park
  - assign_station_sectors()        -- compass-direction sectors
  - select_stations_by_count()      -- auto-select N closest with diversity
  - create_missingness_mask()       -- binary mask BEFORE imputation
  - run_expanded_preprocessing()    -- full pipeline with configurable stations
"""

import glob
import logging
import math
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.data_preprocessing import (
    load_station_csv,
    merge_stations,
    handle_missing_data,
    chronological_split,
    fill_remaining_nans_with_train_means,
    fit_and_apply_scaler,
)
from src.data_preprocessing_v2 import (
    create_lagged_features,
    create_autoregressive_feature,
    create_diurnal_range_features,
    create_trend_features,
    create_target_columns,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------
PROCESSED_EXPANDED_DIR = os.path.join(config.DATA_DIR, "processed_expanded")


# ===========================================================================
# Station Discovery & Metadata
# ===========================================================================

def discover_available_stations(raw_dir: Optional[str] = None) -> list[str]:
    """Find all station IDs that have CSV files in the raw data directory.

    Parameters
    ----------
    raw_dir : str, optional
        Directory to scan. Defaults to config.RAW_DATA_DIR.

    Returns
    -------
    list[str]
        Sorted list of station IDs (e.g., ['USW00013739', 'USW00014732', ...]).
    """
    if raw_dir is None:
        raw_dir = config.RAW_DATA_DIR

    csv_files = glob.glob(os.path.join(raw_dir, "USW*.csv"))
    station_ids = sorted(
        os.path.splitext(os.path.basename(f))[0] for f in csv_files
    )

    logger.info("Discovered %d station CSVs in %s", len(station_ids), raw_dir)
    return station_ids


def haversine_miles(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Compute great-circle distance in miles between two lat/lon points.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Distance in statute miles.
    """
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def compass_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Compute initial compass bearing from (lat1,lon1) to (lat2,lon2).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Bearing in degrees [0, 360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    x = math.sin(dl) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dl))
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def bearing_to_sector(bearing: float) -> str:
    """Map a compass bearing to one of 8 directional sectors.

    Sectors: N, NE, E, SE, S, SW, W, NW (each spanning 45 degrees).

    Parameters
    ----------
    bearing : float
        Bearing in degrees [0, 360).

    Returns
    -------
    str
        Sector name.
    """
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) % 360 / 45)
    return sectors[idx]


def compute_station_metadata(
    station_ids: list[str],
    stations_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute distance, bearing, and sector for each station relative to
    Central Park.

    If a stations.csv file exists and contains the station, its coordinates
    are used. Otherwise, a fallback lookup table of known NOAA station
    coordinates is used. Stations without known coordinates are assigned
    distance=999 and sector='UNK'.

    Parameters
    ----------
    station_ids : list[str]
        Station IDs to compute metadata for.
    stations_csv_path : str, optional
        Path to stations.csv. Defaults to config.STATIONS_FILE.

    Returns
    -------
    pd.DataFrame
        Columns: station_id, latitude, longitude, distance_miles, bearing,
        sector. Sorted by distance_miles ascending.
    """
    if stations_csv_path is None:
        stations_csv_path = config.STATIONS_FILE

    # Load known station coordinates from multiple sources
    known_coords = {}

    # Source 1: Original stations.csv
    if os.path.exists(stations_csv_path):
        try:
            df = pd.read_csv(stations_csv_path)
            for _, row in df.iterrows():
                sid = row["station_id"]
                known_coords[sid] = (row["latitude"], row["longitude"])
        except Exception:
            logger.warning("Could not parse stations CSV at %s",
                           stations_csv_path)

    # Source 2: Expanded stations CSV (generated by Analyst 1)
    expanded_csv = os.path.join(config.DATA_DIR, "stations_expanded.csv")
    if os.path.exists(expanded_csv):
        try:
            df_exp = pd.read_csv(expanded_csv)
            for _, row in df_exp.iterrows():
                sid = row["station_id"]
                known_coords[sid] = (row["latitude"], row["longitude"])
        except Exception:
            logger.warning("Could not parse expanded stations CSV at %s",
                           expanded_csv)

    # Source 3: config_expanded.py module (if it exists)
    try:
        import config_expanded
        if hasattr(config_expanded, "EXPANDED_STATION_COORDS"):
            known_coords.update(config_expanded.EXPANDED_STATION_COORDS)
    except ImportError:
        pass

    target_lat = config.TARGET_LAT
    target_lon = config.TARGET_LON

    records = []
    for sid in station_ids:
        if sid == config.TARGET_STATION:
            records.append({
                "station_id": sid,
                "latitude": target_lat,
                "longitude": target_lon,
                "distance_miles": 0.0,
                "bearing": 0.0,
                "sector": "Target",
            })
            continue

        if sid in known_coords:
            lat, lon = known_coords[sid]
            dist = haversine_miles(target_lat, target_lon, lat, lon)
            bear = compass_bearing(target_lat, target_lon, lat, lon)
            sector = bearing_to_sector(bear)
        else:
            lat, lon = 0.0, 0.0
            dist = 999.0
            bear = 0.0
            sector = "UNK"
            logger.warning(
                "No coordinates found for station %s; assigning dist=999",
                sid,
            )

        records.append({
            "station_id": sid,
            "latitude": lat,
            "longitude": lon,
            "distance_miles": dist,
            "bearing": bear,
            "sector": sector,
        })

    meta = pd.DataFrame(records)
    meta = meta.sort_values("distance_miles").reset_index(drop=True)
    return meta


def select_stations_by_count(
    available_ids: list[str],
    n_stations: int,
    stations_csv_path: Optional[str] = None,
) -> list[str]:
    """Auto-select the N closest surrounding stations with directional diversity.

    Strategy:
      1. Compute metadata (distance, sector) for all available stations.
      2. Exclude the target station from selection.
      3. Allocate slots proportionally across represented sectors.
      4. Fill each sector's slots with the closest stations in that sector.
      5. If any slots remain, fill with the globally closest unfilled stations.

    Parameters
    ----------
    available_ids : list[str]
        All station IDs with CSV files.
    n_stations : int
        Number of surrounding stations to select.
    stations_csv_path : str, optional
        Path to stations.csv.

    Returns
    -------
    list[str]
        Selected station IDs (excluding the target station).
    """
    # Remove target station from candidates
    candidates = [s for s in available_ids if s != config.TARGET_STATION]

    if n_stations >= len(candidates):
        logger.info(
            "Requested %d stations but only %d available; using all",
            n_stations, len(candidates),
        )
        return candidates

    meta = compute_station_metadata(candidates, stations_csv_path)
    # Exclude target row if it snuck in
    meta = meta[meta["station_id"] != config.TARGET_STATION].copy()

    # Group by sector
    sectors = meta["sector"].unique()
    sectors = [s for s in sectors if s not in ("Target", "UNK")]

    selected = []

    if len(sectors) > 0:
        # Proportional allocation
        sector_counts = meta[meta["sector"].isin(sectors)].groupby("sector").size()
        total_in_sectors = sector_counts.sum()

        # Allocate at least 1 per sector, then proportional remainder
        base_per_sector = max(1, n_stations // len(sectors))
        remainder = n_stations - base_per_sector * len(sectors)

        allocation = {}
        for sec in sectors:
            allocation[sec] = base_per_sector

        # Give remainder to sectors with most stations (largest pools)
        sorted_secs = sector_counts.sort_values(ascending=False).index.tolist()
        for i in range(max(0, remainder)):
            allocation[sorted_secs[i % len(sorted_secs)]] += 1

        # Select closest within each sector
        for sec, alloc in allocation.items():
            sec_stations = (
                meta[meta["sector"] == sec]
                .sort_values("distance_miles")["station_id"]
                .tolist()
            )
            selected.extend(sec_stations[:alloc])

        # Fill any remaining slots with closest overall not yet selected
        if len(selected) < n_stations:
            remaining_meta = meta[~meta["station_id"].isin(selected)]
            remaining_sorted = (
                remaining_meta.sort_values("distance_miles")["station_id"]
                .tolist()
            )
            for sid in remaining_sorted:
                if len(selected) >= n_stations:
                    break
                selected.append(sid)
    else:
        # No sector info -- just pick closest
        selected = (
            meta.sort_values("distance_miles")["station_id"]
            .head(n_stations)
            .tolist()
        )

    # Trim if over-allocated
    selected = selected[:n_stations]

    logger.info("Selected %d stations with sector diversity: %s",
                len(selected), selected)
    return selected


# ===========================================================================
# Sector Assignments for Expanded Station Sets
# ===========================================================================

def get_expanded_sector_assignments(
    station_ids: list[str],
    stations_csv_path: Optional[str] = None,
) -> dict[str, list[str]]:
    """Compute sector assignments for an arbitrary set of stations.

    Unlike the hardcoded V2 sectors, this function dynamically assigns
    stations to compass sectors based on their bearing from Central Park.

    Sectors used: N, NE, E, SE, S, SW, W, NW.

    Parameters
    ----------
    station_ids : list[str]
        Surrounding station IDs (excluding target).
    stations_csv_path : str, optional
        Path to stations.csv.

    Returns
    -------
    dict[str, list[str]]
        Mapping of sector name -> list of station IDs in that sector.
    """
    meta = compute_station_metadata(station_ids, stations_csv_path)
    meta = meta[meta["station_id"] != config.TARGET_STATION]

    sectors: dict[str, list[str]] = {}
    for _, row in meta.iterrows():
        sec = row["sector"]
        if sec in ("Target", "UNK"):
            sec = "UNK"
        sectors.setdefault(sec, []).append(row["station_id"])

    logger.info("Expanded sector assignments: %s",
                {k: len(v) for k, v in sectors.items()})
    return sectors


# ===========================================================================
# Missingness Masking
# ===========================================================================

def create_missingness_mask(
    merged_df: pd.DataFrame,
    station_ids: list[str],
    variables: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Create binary missingness mask features BEFORE imputation.

    For each station, variable, and lag, creates a column:
      - 1 = original data was present (not NaN)
      - 0 = original data was missing (will be imputed)

    The mask is computed from the raw (pre-imputation) merged DataFrame,
    after applying the lag shift.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame with raw (pre-imputed) station data.
    station_ids : list[str]
        Surrounding station IDs to create masks for.
    variables : list[str]
        Variable names (e.g., ["TMAX", "TMIN"]).
    lags : list[int]
        Lag values (e.g., [1]).

    Returns
    -------
    pd.DataFrame
        DataFrame with binary mask columns named
        ``{station_id}_mask_{var}_lag{k}``.
    """
    frames = []
    for lag in sorted(lags):
        for sid in station_ids:
            for var in variables:
                col = f"{sid}_{var}"
                if col in merged_df.columns:
                    # Apply same lag shift as features
                    lagged = merged_df[col].shift(lag)
                    # 1 = present, 0 = missing
                    mask = lagged.notna().astype(int)
                    mask.name = f"{sid}_mask_{var}_lag{lag}"
                    frames.append(mask)

    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


# ===========================================================================
# Load Stations (Expanded)
# ===========================================================================

def load_expanded_stations(
    station_ids: list[str],
    raw_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Load station CSVs for the given station IDs.

    Gracefully skips stations whose CSV files do not exist.

    Parameters
    ----------
    station_ids : list[str]
        Station IDs to load.
    raw_dir : str, optional
        Directory containing station CSVs. Defaults to config.RAW_DATA_DIR.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of station_id -> DataFrame. Only includes stations
        with available CSV files.
    """
    if raw_dir is None:
        raw_dir = config.RAW_DATA_DIR

    station_data = {}
    missing = []

    for sid in station_ids:
        csv_path = os.path.join(raw_dir, f"{sid}.csv")
        if not os.path.exists(csv_path):
            missing.append(sid)
            continue
        try:
            df = load_station_csv(csv_path)
            station_data[sid] = df
        except Exception as e:
            logger.warning("Error loading %s: %s", sid, e)
            missing.append(sid)

    if missing:
        logger.warning(
            "Skipped %d stations without CSV files: %s",
            len(missing), missing,
        )
    logger.info("Loaded %d / %d station CSVs", len(station_data), len(station_ids))
    return station_data


# ===========================================================================
# Completeness Filtering for Expanded Sets
# ===========================================================================

def filter_expanded_by_completeness(
    merged_df: pd.DataFrame,
    min_completeness: float = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove stations below minimum TMAX completeness.

    Same logic as data_preprocessing.filter_stations_by_completeness
    but works with any set of stations (not just config.ALL_STATIONS).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame.
    min_completeness : float, optional
        Minimum fraction. Defaults to config.MIN_COMPLETENESS.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (filtered DataFrame, list of dropped station IDs)
    """
    if min_completeness is None:
        min_completeness = config.MIN_COMPLETENESS

    # Identify all station IDs from column names
    all_sids = set()
    for col in merged_df.columns:
        # Columns are like USW00014735_TMAX
        if "_TMAX" in col or "_TMIN" in col:
            sid = col.rsplit("_", 1)[0]
            all_sids.add(sid)

    total_days = len(merged_df)
    dropped = []
    cols_to_drop = []

    for sid in all_sids:
        tmax_col = f"{sid}_TMAX"
        if tmax_col in merged_df.columns:
            completeness = merged_df[tmax_col].notna().sum() / max(total_days, 1)
        else:
            completeness = 0.0

        if completeness < min_completeness:
            if sid == config.TARGET_STATION:
                logger.warning(
                    "Target station %s has %.1f%% completeness, keeping",
                    sid, completeness * 100,
                )
                continue
            dropped.append(sid)
            cols_to_drop.extend(
                c for c in merged_df.columns if c.startswith(f"{sid}_")
            )

    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)

    logger.info(
        "Completeness filter: kept %d, dropped %d stations",
        len(all_sids) - len(dropped), len(dropped),
    )
    return merged_df, dropped


# ===========================================================================
# Enhanced Feature Creation with Missingness Mask
# ===========================================================================

def create_expanded_features(
    merged_df: pd.DataFrame,
    surrounding_ids: list[str],
    include_missingness_mask: bool = True,
    include_autoregressive: bool = True,
    include_diurnal: bool = True,
    include_sectors: bool = True,
    include_trends: bool = True,
    lags: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create feature matrix from merged station data with missingness mask.

    Similar to data_preprocessing_v2.create_enhanced_features, but:
      - Uses the provided surrounding_ids instead of config.SURROUNDING_STATIONS
      - Adds missingness mask features before imputation
      - Uses dynamic sector assignments

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame (raw, pre-imputation).
    surrounding_ids : list[str]
        Surrounding station IDs to use as features.
    include_missingness_mask : bool
        If True, add binary missingness mask per station/variable/lag.
    include_autoregressive : bool
        Include NYC TMAX(t-1) as a feature.
    include_diurnal : bool
        Include per-station diurnal range features.
    include_sectors : bool
        Include sector average and gradient features.
    include_trends : bool
        Include trend (day-over-day change) features.
    lags : list[int], optional
        Lag values. Defaults to [1].

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
        (features, raw_target, delta_target, nyc_tmax_prev)
    """
    if lags is None:
        lags = [1]

    feature_parts = []

    # 1. Targets (always computed from target station)
    raw_target, delta_target, nyc_tmax_prev = create_target_columns(merged_df)

    # 2. Missingness mask (MUST be created BEFORE any imputation/fill)
    if include_missingness_mask:
        mask_features = create_missingness_mask(
            merged_df, surrounding_ids, config.INPUT_VARIABLES, lags,
        )
        feature_parts.append(mask_features)

    # 3. Lagged TMAX/TMIN from surrounding stations
    lagged_features = create_lagged_features(
        merged_df, lags, surrounding_ids, config.INPUT_VARIABLES,
    )
    feature_parts.append(lagged_features)

    # 4. Cyclical date encoding
    date_features = pd.DataFrame(index=merged_df.index)
    day_of_year = merged_df.index.dayofyear
    date_features["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    date_features["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    feature_parts.append(date_features)

    # 5. Autoregressive NYC TMAX
    if include_autoregressive:
        ar_features = create_autoregressive_feature(merged_df, lags)
        feature_parts.append(ar_features)

    # 6. Diurnal range
    if include_diurnal:
        diurnal_features = create_diurnal_range_features(
            merged_df, lags, surrounding_ids,
        )
        feature_parts.append(diurnal_features)

    # 7. Sector features (dynamic assignment)
    if include_sectors:
        sector_assignments = get_expanded_sector_assignments(surrounding_ids)
        # Sector averages
        from src.data_preprocessing_v2 import (
            compute_sector_features,
            compute_sector_gradients,
        )
        sector_avg = compute_sector_features(
            merged_df, sector_assignments, lags,
        )
        feature_parts.append(sector_avg)
        # Sector gradients (only if relevant sectors exist)
        if any(s in sector_assignments for s in ("W", "NW", "WNW")):
            # Compute gradients using whatever sectors are available
            sector_grad = compute_sector_gradients(
                merged_df, sector_assignments, lags,
            )
            feature_parts.append(sector_grad)

    # 8. Trend features
    if include_trends:
        trend_features = create_trend_features(merged_df, surrounding_ids)
        feature_parts.append(trend_features)

    # Combine all features
    features = pd.concat(feature_parts, axis=1)

    # Determine how many initial rows to drop due to lagging
    max_lag = max(lags)
    min_valid_start = max(max_lag, 3) if include_trends else max_lag

    # Validity mask
    valid_mask = raw_target.notna() & delta_target.notna()
    valid_mask.iloc[:min_valid_start] = False

    features = features[valid_mask]
    raw_target = raw_target[valid_mask]
    delta_target = delta_target[valid_mask]
    nyc_tmax_prev = nyc_tmax_prev[valid_mask]

    logger.info(
        "Created expanded features: %d rows x %d columns "
        "(stations=%d, mask=%s, AR=%s, diurnal=%s, sectors=%s, trends=%s)",
        features.shape[0], features.shape[1],
        len(surrounding_ids), include_missingness_mask,
        include_autoregressive, include_diurnal, include_sectors,
        include_trends,
    )

    return features, raw_target, delta_target, nyc_tmax_prev


# ===========================================================================
# Save Expanded Data
# ===========================================================================

def save_expanded_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: pd.Series,
    y_val_raw: pd.Series,
    y_test_raw: pd.Series,
    y_train_delta: pd.Series,
    y_val_delta: pd.Series,
    y_test_delta: pd.Series,
    nyc_prev_train: pd.Series,
    nyc_prev_val: pd.Series,
    nyc_prev_test: pd.Series,
    scaler: StandardScaler,
    output_dir: Optional[str] = None,
) -> None:
    """Save all expanded processed data and artifacts to disk.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Scaled feature matrices.
    y_train_raw, y_val_raw, y_test_raw : pd.Series
        Raw TMAX targets.
    y_train_delta, y_val_delta, y_test_delta : pd.Series
        Delta-T targets.
    nyc_prev_train, nyc_prev_val, nyc_prev_test : pd.Series
        NYC TMAX(t-1) for delta reconstruction.
    scaler : StandardScaler
        Fitted scaler.
    output_dir : str, optional
        Output directory. Defaults to PROCESSED_EXPANDED_DIR.
    """
    if output_dir is None:
        output_dir = PROCESSED_EXPANDED_DIR

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "features_train.csv"))
    X_val.to_csv(os.path.join(output_dir, "features_val.csv"))
    X_test.to_csv(os.path.join(output_dir, "features_test.csv"))

    y_train_raw.to_csv(os.path.join(output_dir, "target_train.csv"), header=True)
    y_val_raw.to_csv(os.path.join(output_dir, "target_val.csv"), header=True)
    y_test_raw.to_csv(os.path.join(output_dir, "target_test.csv"), header=True)

    y_train_delta.to_csv(os.path.join(output_dir, "target_delta_train.csv"), header=True)
    y_val_delta.to_csv(os.path.join(output_dir, "target_delta_val.csv"), header=True)
    y_test_delta.to_csv(os.path.join(output_dir, "target_delta_test.csv"), header=True)

    nyc_prev_train.to_csv(os.path.join(output_dir, "nyc_prev_train.csv"), header=True)
    nyc_prev_val.to_csv(os.path.join(output_dir, "nyc_prev_val.csv"), header=True)
    nyc_prev_test.to_csv(os.path.join(output_dir, "nyc_prev_test.csv"), header=True)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Saved expanded processed data to %s", output_dir)
    logger.info("  features_train: %s", X_train.shape)
    logger.info("  features_val:   %s", X_val.shape)
    logger.info("  features_test:  %s", X_test.shape)


# ===========================================================================
# Main Expanded Pipeline
# ===========================================================================

def run_expanded_preprocessing(
    station_list: Optional[list[str]] = None,
    n_stations: Optional[int] = None,
    include_missingness_mask: bool = True,
    include_autoregressive: bool = True,
    include_diurnal: bool = True,
    include_sectors: bool = True,
    include_trends: bool = True,
    lags: Optional[list[int]] = None,
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run enhanced preprocessing with configurable station set.

    Parameters
    ----------
    station_list : list[str], optional
        Explicit list of surrounding station IDs to use.
        If None, auto-discover from data/raw/ and use n_stations.
    n_stations : int, optional
        Auto-select N closest stations with sector diversity.
        Only used if station_list is None. If both are None,
        all available surrounding stations are used.
    include_missingness_mask : bool
        Add binary missingness mask features.
    include_autoregressive : bool
        Include NYC TMAX(t-1).
    include_diurnal : bool
        Include diurnal range features.
    include_sectors : bool
        Include sector averages and gradients.
    include_trends : bool
        Include trend features.
    lags : list[int], optional
        Lag values. Defaults to [1].
    raw_dir : str, optional
        Raw data directory. Defaults to config.RAW_DATA_DIR.
    output_dir : str, optional
        Output directory. Defaults to PROCESSED_EXPANDED_DIR.

    Returns
    -------
    dict
        Dictionary containing:
        - X_train, X_val, X_test (scaled features)
        - y_train, y_val, y_test (raw TMAX targets)
        - y_train_delta, y_val_delta, y_test_delta (delta-T targets)
        - nyc_prev_train, nyc_prev_val, nyc_prev_test
        - scaler, feature_names, n_features
        - surrounding_ids (list of station IDs used)
        - dropped_stations
        - available_stations (list of all discovered station IDs)
    """
    if lags is None:
        lags = [1]

    logger.info("=" * 60)
    logger.info("Starting Expanded Preprocessing Pipeline")
    logger.info("=" * 60)

    # 1. Discover available stations
    available = discover_available_stations(raw_dir)
    logger.info("Available stations: %d", len(available))

    # Ensure target station is included
    if config.TARGET_STATION not in available:
        raise FileNotFoundError(
            f"Target station {config.TARGET_STATION} CSV not found in "
            f"{raw_dir or config.RAW_DATA_DIR}. Run data_collection.py first."
        )

    # 2. Determine surrounding station list
    if station_list is not None:
        # Use explicit list, filtering to available CSVs
        surrounding_ids = [s for s in station_list
                           if s != config.TARGET_STATION and s in available]
        unavailable = [s for s in station_list
                       if s != config.TARGET_STATION and s not in available]
        if unavailable:
            logger.warning(
                "Requested but unavailable stations (skipped): %s",
                unavailable,
            )
    elif n_stations is not None:
        surrounding_ids = select_stations_by_count(
            available, n_stations,
        )
    else:
        # Use all available except target
        surrounding_ids = [s for s in available
                           if s != config.TARGET_STATION]

    if not surrounding_ids:
        raise ValueError("No surrounding stations available for preprocessing")

    logger.info("Using %d surrounding stations: %s",
                len(surrounding_ids), surrounding_ids)

    # 3. Load station data (target + selected surrounding)
    all_needed = [config.TARGET_STATION] + surrounding_ids
    station_data = load_expanded_stations(all_needed, raw_dir)

    if config.TARGET_STATION not in station_data:
        raise FileNotFoundError(
            f"Target station {config.TARGET_STATION} could not be loaded"
        )

    # 4. Merge into wide DataFrame
    merged_df = merge_stations(station_data)

    # 5. Filter by completeness
    merged_filtered, dropped_stations = filter_expanded_by_completeness(
        merged_df,
    )

    # Update surrounding_ids to exclude dropped stations
    surrounding_ids = [
        s for s in surrounding_ids if s not in dropped_stations
    ]

    # 6. Create features with missingness mask
    features, raw_target, delta_target, nyc_tmax_prev = create_expanded_features(
        merged_filtered,
        surrounding_ids,
        include_missingness_mask=include_missingness_mask,
        include_autoregressive=include_autoregressive,
        include_diurnal=include_diurnal,
        include_sectors=include_sectors,
        include_trends=include_trends,
        lags=lags,
    )

    # 7. Handle missing data (forward-fill, drop NaN targets)
    features, raw_target = handle_missing_data(features, raw_target)
    delta_target = delta_target.reindex(features.index)
    nyc_tmax_prev = nyc_tmax_prev.reindex(features.index)

    # 8. Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        features, raw_target,
    )

    y_train_delta = delta_target.reindex(X_train.index)
    y_val_delta = delta_target.reindex(X_val.index)
    y_test_delta = delta_target.reindex(X_test.index)

    nyc_prev_train = nyc_tmax_prev.reindex(X_train.index)
    nyc_prev_val = nyc_tmax_prev.reindex(X_val.index)
    nyc_prev_test = nyc_tmax_prev.reindex(X_test.index)

    # 9. Fill remaining NaNs with training-set means
    X_train, X_val, X_test, col_means = fill_remaining_nans_with_train_means(
        X_train, X_val, X_test,
    )

    # 10. Fit scaler on training data only
    X_train_s, X_val_s, X_test_s, scaler = fit_and_apply_scaler(
        X_train, X_val, X_test,
    )

    # 11. Save
    save_expanded_data(
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        y_train_delta, y_val_delta, y_test_delta,
        nyc_prev_train, nyc_prev_val, nyc_prev_test,
        scaler, output_dir,
    )

    feature_names = list(X_train_s.columns)
    n_features = len(feature_names)

    logger.info("Expanded preprocessing complete.")
    logger.info("  %d features, %d surrounding stations",
                n_features, len(surrounding_ids))
    logger.info("  Train: %d, Val: %d, Test: %d",
                len(X_train_s), len(X_val_s), len(X_test_s))

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_delta": y_train_delta,
        "y_val_delta": y_val_delta,
        "y_test_delta": y_test_delta,
        "nyc_prev_train": nyc_prev_train,
        "nyc_prev_val": nyc_prev_val,
        "nyc_prev_test": nyc_prev_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": n_features,
        "surrounding_ids": surrounding_ids,
        "dropped_stations": dropped_stations,
        "available_stations": available,
    }


def main():
    """Main entry point for expanded preprocessing."""
    result = run_expanded_preprocessing()
    logger.info("Expanded preprocessing pipeline complete!")
    return result


if __name__ == "__main__":
    main()
