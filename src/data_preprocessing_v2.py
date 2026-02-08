"""
Enhanced Data Preprocessing Module (V2) for NYC Temperature Prediction.

Extends the Phase 1 preprocessing pipeline with:
  - Delta-T target: DeltaT(t) = TMAX_NYC(t) - TMAX_NYC(t-1)
  - Autoregressive feature: NYC TMAX(t-1)
  - Per-station diurnal range: TMAX - TMIN (lagged)
  - Sector average features: mean TMAX per directional sector
  - Sector gradient features: grad_upstream_vs_coast, grad_SW_vs_NW
  - Trend features: Delta1 = T(t-1) - T(t-2), Delta2 = T(t-2) - T(t-3)
  - Multi-lag support (t-1, t-2, t-3)

All features respect the no-leakage constraint: only past data is used.
Scaler is fit on training data only.
Splits are strictly chronological.
"""

import os
import sys
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.data_preprocessing import (
    load_all_stations,
    merge_stations,
    filter_stations_by_completeness,
    add_cyclical_date_features,
    handle_missing_data,
    chronological_split,
    fill_remaining_nans_with_train_means,
    fit_and_apply_scaler,
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
# Default output directory for V2 processed data
# ---------------------------------------------------------------------------
PROCESSED_V2_DIR = os.path.join(config.DATA_DIR, "processed_v2")


# ===========================================================================
# Sector Definitions
# ===========================================================================

def get_sector_assignments() -> dict[str, list[str]]:
    """Return directional sector assignments for surrounding stations.

    Sectors group stations by compass direction relative to Central Park:
      - WNW (upstream cold-air advection): Scranton, Allentown, Albany, Poughkeepsie
      - SW  (warm advection): Philadelphia, Trenton
      - Coastal (Atlantic moderation): Islip, JFK, Atlantic City, Bridgeport
      - Near-field (urban/local): Newark, LaGuardia, White Plains

    Returns
    -------
    dict[str, list[str]]
        Mapping of sector name to list of NOAA station IDs.
    """
    return {
        "WNW": [
            "USW00014777",  # Scranton, PA
            "USW00014737",  # Allentown, PA
            "USW00014735",  # Albany, NY
            "USW00014757",  # Poughkeepsie, NY
        ],
        "SW": [
            "USW00013739",  # Philadelphia, PA
            "USW00014792",  # Trenton, NJ
        ],
        "Coastal": [
            "USW00014732",  # Islip, NY
            "USW00094789",  # JFK Airport, NY
            "USW00093730",  # Atlantic City, NJ
            "USW00094702",  # Bridgeport, CT
        ],
        "NearField": [
            "USW00014734",  # Newark, NJ
            "USW00014739",  # LaGuardia Airport, NY
            "USW00014771",  # White Plains, NY
        ],
    }


# ===========================================================================
# Enhanced Feature Engineering
# ===========================================================================

def create_lagged_features(
    merged_df: pd.DataFrame,
    lags: list[int],
    station_ids: list[str],
    variables: list[str],
) -> pd.DataFrame:
    """Create lagged features for multiple lag values.

    For each station, variable, and lag, creates a column with the value
    from ``lag`` days prior. E.g., lag=1 means the column at date t
    contains the value from date t-1.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame with columns like ``{station_id}_{var}``.
    lags : list[int]
        List of lag values (e.g., [1, 2, 3]).
    station_ids : list[str]
        Station IDs to create lagged features for.
    variables : list[str]
        Variable names (e.g., ["TMAX", "TMIN"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged feature columns named
        ``{station_id}_{var}_lag{k}``.
    """
    frames = []
    for lag in sorted(lags):
        for sid in station_ids:
            for var in variables:
                col = f"{sid}_{var}"
                if col in merged_df.columns:
                    lagged = merged_df[col].shift(lag)
                    lagged.name = f"{sid}_{var}_lag{lag}"
                    frames.append(lagged)
    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


def create_autoregressive_feature(
    merged_df: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    """Create NYC TMAX autoregressive (lag) features.

    Adds the target station's own TMAX from previous days as input features.
    This allows the model to leverage persistence-like information directly.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.
    lags : list[int]
        Lag values (e.g., [1] or [1, 2, 3]).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like ``NYC_TMAX_lag1``, ``NYC_TMAX_lag2``, etc.
    """
    target_col = f"{config.TARGET_STATION}_TMAX"
    if target_col not in merged_df.columns:
        raise ValueError(f"Target column '{target_col}' not in merged_df")

    frames = []
    for lag in sorted(lags):
        lagged = merged_df[target_col].shift(lag)
        lagged.name = f"NYC_TMAX_lag{lag}"
        frames.append(lagged)
    return pd.concat(frames, axis=1)


def create_diurnal_range_features(
    merged_df: pd.DataFrame,
    lags: list[int],
    station_ids: list[str],
) -> pd.DataFrame:
    """Compute per-station diurnal range (TMAX - TMIN) as lagged features.

    The diurnal range is a proxy for cloud cover and weather-front activity.
    Large ranges indicate clear skies; small ranges suggest clouds or fronts.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.
    lags : list[int]
        Lag values.
    station_ids : list[str]
        Station IDs to compute diurnal range for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like ``diurnal_{station_id}_lag{k}``.
    """
    frames = []
    for sid in station_ids:
        tmax_col = f"{sid}_TMAX"
        tmin_col = f"{sid}_TMIN"
        if tmax_col in merged_df.columns and tmin_col in merged_df.columns:
            diurnal = merged_df[tmax_col] - merged_df[tmin_col]
            for lag in sorted(lags):
                lagged = diurnal.shift(lag)
                lagged.name = f"diurnal_{sid}_lag{lag}"
                frames.append(lagged)
    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


def compute_sector_features(
    merged_df: pd.DataFrame,
    sector_assignments: dict[str, list[str]],
    lags: list[int],
) -> pd.DataFrame:
    """Compute mean TMAX per directional sector as lagged features.

    For each sector (e.g., WNW, SW, Coastal, NearField), averages the
    TMAX values across all stations in that sector, then lags the result.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.
    sector_assignments : dict[str, list[str]]
        Mapping of sector name to list of station IDs.
    lags : list[int]
        Lag values.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like ``sector_WNW_mean_lag1``.
    """
    frames = []
    for sector_name, station_ids in sector_assignments.items():
        tmax_cols = [f"{sid}_TMAX" for sid in station_ids
                     if f"{sid}_TMAX" in merged_df.columns]
        if not tmax_cols:
            logger.warning("No TMAX columns found for sector %s", sector_name)
            continue
        sector_mean = merged_df[tmax_cols].mean(axis=1)
        for lag in sorted(lags):
            lagged = sector_mean.shift(lag)
            lagged.name = f"sector_{sector_name}_mean_lag{lag}"
            frames.append(lagged)
    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


def compute_sector_gradients(
    merged_df: pd.DataFrame,
    sector_assignments: dict[str, list[str]],
    lags: list[int],
) -> pd.DataFrame:
    """Compute temperature gradients between directional sectors.

    Two gradient features:
      1. grad_upstream_vs_coast: WNW mean - Coastal mean
         (positive = cold air upstream, negative = warm coast relative to inland)
      2. grad_SW_vs_NW: SW mean - WNW mean
         (positive = warm advection from south, negative = cold air dominance)

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.
    sector_assignments : dict[str, list[str]]
        Sector assignments (must include 'WNW', 'SW', 'Coastal').
    lags : list[int]
        Lag values.

    Returns
    -------
    pd.DataFrame
        DataFrame with gradient columns.
    """
    frames = []

    # Helper: compute sector mean TMAX
    def _sector_mean(sector_name: str) -> pd.Series:
        station_ids = sector_assignments.get(sector_name, [])
        tmax_cols = [f"{sid}_TMAX" for sid in station_ids
                     if f"{sid}_TMAX" in merged_df.columns]
        if not tmax_cols:
            return pd.Series(np.nan, index=merged_df.index)
        return merged_df[tmax_cols].mean(axis=1)

    wnw_mean = _sector_mean("WNW")
    sw_mean = _sector_mean("SW")
    coastal_mean = _sector_mean("Coastal")

    # Gradient 1: upstream vs coast
    grad_upstream_coast = wnw_mean - coastal_mean
    for lag in sorted(lags):
        lagged = grad_upstream_coast.shift(lag)
        lagged.name = f"grad_upstream_vs_coast_lag{lag}"
        frames.append(lagged)

    # Gradient 2: SW vs NW
    grad_sw_nw = sw_mean - wnw_mean
    for lag in sorted(lags):
        lagged = grad_sw_nw.shift(lag)
        lagged.name = f"grad_SW_vs_NW_lag{lag}"
        frames.append(lagged)

    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


def create_trend_features(
    merged_df: pd.DataFrame,
    station_ids: list[str],
) -> pd.DataFrame:
    """Create trend (day-over-day change) features per station.

    Two trend features per station:
      - trend_delta1_{sid}: T(t-1) - T(t-2)  (recent 1-day change)
      - trend_delta2_{sid}: T(t-2) - T(t-3)  (prior 1-day change)

    These capture whether temperatures are rising or falling.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.
    station_ids : list[str]
        Station IDs to compute trends for.

    Returns
    -------
    pd.DataFrame
        DataFrame with trend columns.
    """
    frames = []
    for sid in station_ids:
        tmax_col = f"{sid}_TMAX"
        if tmax_col not in merged_df.columns:
            continue
        tmax = merged_df[tmax_col]
        # Delta1: T(t-1) - T(t-2)
        delta1 = tmax.shift(1) - tmax.shift(2)
        delta1.name = f"trend_delta1_{sid}"
        frames.append(delta1)
        # Delta2: T(t-2) - T(t-3)
        delta2 = tmax.shift(2) - tmax.shift(3)
        delta2.name = f"trend_delta2_{sid}"
        frames.append(delta2)
    if not frames:
        return pd.DataFrame(index=merged_df.index)
    return pd.concat(frames, axis=1)


def create_target_columns(
    merged_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Create both raw TMAX and delta-T target columns.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (raw_target, delta_target, nyc_tmax_prev)
        - raw_target: NYC TMAX on day t
        - delta_target: TMAX_NYC(t) - TMAX_NYC(t-1)
        - nyc_tmax_prev: TMAX_NYC(t-1), needed for delta reconstruction
    """
    target_col = f"{config.TARGET_STATION}_TMAX"
    if target_col not in merged_df.columns:
        raise ValueError(f"Target column '{target_col}' not in merged_df")

    raw_target = merged_df[target_col].copy()
    raw_target.name = "NYC_TMAX"

    nyc_tmax_prev = merged_df[target_col].shift(1).copy()
    nyc_tmax_prev.name = "NYC_TMAX_prev"

    delta_target = raw_target - nyc_tmax_prev
    delta_target.name = "NYC_DELTA_T"

    return raw_target, delta_target, nyc_tmax_prev


def create_enhanced_features(
    merged_df: pd.DataFrame,
    include_autoregressive: bool = True,
    include_diurnal: bool = True,
    include_sectors: bool = True,
    include_trends: bool = True,
    lags: Optional[list[int]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create the full enhanced feature set from merged station data.

    Assembles all feature types controlled by boolean flags:
      1. Lagged TMAX/TMIN from surrounding stations (always included)
      2. Cyclical date encoding (always included)
      3. NYC TMAX autoregressive term (if include_autoregressive)
      4. Per-station diurnal range (if include_diurnal)
      5. Sector averages (if include_sectors)
      6. Sector gradients (if include_sectors)
      7. Trend features (if include_trends)

    Parameters
    ----------
    merged_df : pd.DataFrame
        Wide-format merged DataFrame with all station columns.
    include_autoregressive : bool
        Whether to include NYC TMAX(t-1) as a feature.
    include_diurnal : bool
        Whether to include per-station diurnal range features.
    include_sectors : bool
        Whether to include sector average and gradient features.
    include_trends : bool
        Whether to include trend (day-over-day change) features.
    lags : list[int], optional
        Lag values for features. Defaults to [1].

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
        (features, raw_target, delta_target, nyc_tmax_prev)
        All aligned by date index. Contains NaNs at the boundaries
        due to lagging; these should be dropped before splitting.
    """
    if lags is None:
        lags = [1]

    # Identify surrounding station IDs (exclude the target station)
    surrounding_ids = [sid for sid in config.SURROUNDING_STATIONS.keys()]

    # 1. Create targets
    raw_target, delta_target, nyc_tmax_prev = create_target_columns(merged_df)

    # 2. Lagged TMAX/TMIN from surrounding stations
    feature_parts = []
    lagged_features = create_lagged_features(
        merged_df, lags, surrounding_ids, config.INPUT_VARIABLES,
    )
    feature_parts.append(lagged_features)

    # 3. Cyclical date encoding
    date_features = pd.DataFrame(index=merged_df.index)
    day_of_year = merged_df.index.dayofyear
    date_features["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    date_features["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    feature_parts.append(date_features)

    # 4. Autoregressive NYC TMAX
    if include_autoregressive:
        ar_features = create_autoregressive_feature(merged_df, lags)
        feature_parts.append(ar_features)

    # 5. Diurnal range
    if include_diurnal:
        diurnal_features = create_diurnal_range_features(
            merged_df, lags, surrounding_ids,
        )
        feature_parts.append(diurnal_features)

    # 6. Sector features
    if include_sectors:
        sector_assignments = get_sector_assignments()
        sector_avg_features = compute_sector_features(
            merged_df, sector_assignments, lags,
        )
        feature_parts.append(sector_avg_features)

        sector_grad_features = compute_sector_gradients(
            merged_df, sector_assignments, lags,
        )
        feature_parts.append(sector_grad_features)

    # 7. Trend features
    if include_trends:
        trend_features = create_trend_features(merged_df, surrounding_ids)
        feature_parts.append(trend_features)

    # Combine all features
    features = pd.concat(feature_parts, axis=1)

    # Determine how many initial rows to drop due to lagging
    max_lag = max(lags)
    # Trends need up to shift(3) regardless, so minimum valid start is row 3
    min_valid_start = max(max_lag, 3) if include_trends else max_lag

    # Create a validity mask: target not NaN, and not in the initial lag window
    valid_mask = raw_target.notna() & delta_target.notna()
    valid_mask.iloc[:min_valid_start] = False

    features = features[valid_mask]
    raw_target = raw_target[valid_mask]
    delta_target = delta_target[valid_mask]
    nyc_tmax_prev = nyc_tmax_prev[valid_mask]

    logger.info(
        "Created enhanced features: %d rows x %d columns "
        "(autoregressive=%s, diurnal=%s, sectors=%s, trends=%s, lags=%s)",
        features.shape[0], features.shape[1],
        include_autoregressive, include_diurnal, include_sectors,
        include_trends, lags,
    )

    return features, raw_target, delta_target, nyc_tmax_prev


# ===========================================================================
# Saving Enhanced Data
# ===========================================================================

def save_enhanced_data(
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
    """Save all enhanced processed data and artifacts to disk.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Scaled feature matrices.
    y_train_raw, y_val_raw, y_test_raw : pd.Series
        Raw TMAX target vectors (degF).
    y_train_delta, y_val_delta, y_test_delta : pd.Series
        Delta-T target vectors (degF).
    nyc_prev_train, nyc_prev_val, nyc_prev_test : pd.Series
        NYC TMAX(t-1) values for delta reconstruction.
    scaler : StandardScaler
        The fitted scaler.
    output_dir : str, optional
        Output directory. Defaults to PROCESSED_V2_DIR.
    """
    if output_dir is None:
        output_dir = PROCESSED_V2_DIR

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "features_train.csv"))
    X_val.to_csv(os.path.join(output_dir, "features_val.csv"))
    X_test.to_csv(os.path.join(output_dir, "features_test.csv"))

    y_train_raw.to_csv(os.path.join(output_dir, "target_train.csv"), header=True)
    y_val_raw.to_csv(os.path.join(output_dir, "target_val.csv"), header=True)
    y_test_raw.to_csv(os.path.join(output_dir, "target_test.csv"), header=True)

    y_train_delta.to_csv(
        os.path.join(output_dir, "target_delta_train.csv"), header=True,
    )
    y_val_delta.to_csv(
        os.path.join(output_dir, "target_delta_val.csv"), header=True,
    )
    y_test_delta.to_csv(
        os.path.join(output_dir, "target_delta_test.csv"), header=True,
    )

    nyc_prev_train.to_csv(
        os.path.join(output_dir, "nyc_prev_train.csv"), header=True,
    )
    nyc_prev_val.to_csv(
        os.path.join(output_dir, "nyc_prev_val.csv"), header=True,
    )
    nyc_prev_test.to_csv(
        os.path.join(output_dir, "nyc_prev_test.csv"), header=True,
    )

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Saved enhanced processed data to %s", output_dir)
    logger.info("  features_train: %s", X_train.shape)
    logger.info("  features_val:   %s", X_val.shape)
    logger.info("  features_test:  %s", X_test.shape)


# ===========================================================================
# Main Enhanced Pipeline
# ===========================================================================

def run_enhanced_preprocessing(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_autoregressive: bool = True,
    include_diurnal: bool = True,
    include_sectors: bool = True,
    include_trends: bool = True,
    lags: Optional[list[int]] = None,
) -> dict:
    """Execute the full enhanced preprocessing pipeline.

    Loads raw data, computes enhanced features, handles missing data,
    splits chronologically, scales features, and saves results.

    Parameters
    ----------
    raw_dir : str, optional
        Directory with station CSVs. Defaults to config.RAW_DATA_DIR.
    output_dir : str, optional
        Directory for processed outputs. Defaults to PROCESSED_V2_DIR.
    include_autoregressive : bool
        Include NYC TMAX(t-1) autoregressive feature.
    include_diurnal : bool
        Include per-station diurnal range features.
    include_sectors : bool
        Include sector average and gradient features.
    include_trends : bool
        Include trend (day-over-day change) features.
    lags : list[int], optional
        Lag values for features. Defaults to [1].

    Returns
    -------
    dict
        Dictionary containing all processed data and metadata:
        X_train, X_val, X_test (scaled features),
        y_train, y_val, y_test (raw TMAX targets),
        y_train_delta, y_val_delta, y_test_delta (delta-T targets),
        nyc_prev_train, nyc_prev_val, nyc_prev_test (for reconstruction),
        scaler, feature_names, n_features.
    """
    if lags is None:
        lags = [1]

    logger.info("=" * 60)
    logger.info("Starting Enhanced Preprocessing Pipeline (V2)")
    logger.info("=" * 60)

    # 1. Load station data
    station_data = load_all_stations(raw_dir)
    if not station_data:
        raise FileNotFoundError(
            f"No station CSVs found in {raw_dir or config.RAW_DATA_DIR}. "
            "Run data_collection.py first."
        )

    # 2. Merge into single DataFrame
    merged_df = merge_stations(station_data)

    # 3. Filter out low-completeness stations
    merged_filtered, dropped_stations = filter_stations_by_completeness(merged_df)

    # 4. Create enhanced features and targets
    features, raw_target, delta_target, nyc_tmax_prev = create_enhanced_features(
        merged_filtered,
        include_autoregressive=include_autoregressive,
        include_diurnal=include_diurnal,
        include_sectors=include_sectors,
        include_trends=include_trends,
        lags=lags,
    )

    # 5. Handle missing data (forward-fill, then drop remaining NaN targets)
    features, raw_target = handle_missing_data(features, raw_target)
    # Re-align delta_target and nyc_tmax_prev to the same index
    delta_target = delta_target.reindex(features.index)
    nyc_tmax_prev = nyc_tmax_prev.reindex(features.index)

    # 6. Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        features, raw_target,
    )

    # Split delta targets
    y_train_delta = delta_target.reindex(X_train.index)
    y_val_delta = delta_target.reindex(X_val.index)
    y_test_delta = delta_target.reindex(X_test.index)

    # Split nyc_prev
    nyc_prev_train = nyc_tmax_prev.reindex(X_train.index)
    nyc_prev_val = nyc_tmax_prev.reindex(X_val.index)
    nyc_prev_test = nyc_tmax_prev.reindex(X_test.index)

    # 7. Fill remaining NaNs with training-set means
    X_train, X_val, X_test, col_means = fill_remaining_nans_with_train_means(
        X_train, X_val, X_test,
    )

    # 8. Fit scaler on training data, apply to all splits
    X_train_s, X_val_s, X_test_s, scaler = fit_and_apply_scaler(
        X_train, X_val, X_test,
    )

    # 9. Save
    save_enhanced_data(
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        y_train_delta, y_val_delta, y_test_delta,
        nyc_prev_train, nyc_prev_val, nyc_prev_test,
        scaler, output_dir,
    )

    feature_names = list(X_train_s.columns)
    n_features = len(feature_names)

    logger.info("Enhanced preprocessing complete.")
    logger.info("  %d features: %s", n_features, feature_names[:5])
    logger.info("  Train: %d rows, Val: %d rows, Test: %d rows",
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
        "dropped_stations": dropped_stations,
    }


def main():
    """Main entry point for enhanced data preprocessing."""
    result = run_enhanced_preprocessing()
    logger.info("Enhanced preprocessing pipeline complete!")
    return result


if __name__ == "__main__":
    main()
