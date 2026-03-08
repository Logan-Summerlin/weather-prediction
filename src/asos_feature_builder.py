"""
ASOS-Based Feature Builder for Training Data.

Builds training-ready feature matrices from ASOS daily aggregated data,
replacing GHCN-Daily as the primary temperature source. This resolves
the training/inference mismatch identified in Phase E.

The feature matrix mirrors the structure produced by run_preprocessing.py
but uses ASOS-derived TMAX/TMIN as the primary temperature values.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.city_config import get_city_config, CityConfig
from src.data_preprocessing import add_cyclical_date_features

logger = logging.getLogger(__name__)

# Mapping from ASOS daily CSV column names to GHCN-style variable names
_ASOS_TO_GHCN = {
    "tmax_f": "TMAX",
    "tmin_f": "TMIN",
}

# City code -> target name prefix (mirrors run_preprocessing.py convention)
_CITY_TARGET_NAMES = {
    "nyc": "NYC_TMAX",
    "chi": "CHI_TMAX",
    "phl": "PHL_TMAX",
    "atl": "ATL_TMAX",
    "aus": "AUS_TMAX",
}


def write_city_asos_mapping_csv(city_code: str, output_path: str) -> str:
    """Write a city-specific ASOS station mapping CSV.

    Uses the city config's asos_station_map to create a mapping CSV
    compatible with asos_collection.py's load_asos_station_map().

    Returns the output path.
    """
    cfg = get_city_config(city_code)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for station_id, icao in cfg.asos_station_map.items():
        station_name = cfg.all_stations.get(station_id, station_id)
        rows.append({
            "station_id": station_id,
            "station_name": station_name,
            "icao": icao,
            "asos_available": "yes",
        })

    df = pd.DataFrame(rows, columns=["station_id", "station_name", "icao", "asos_available"])
    df.to_csv(output_path, index=False)

    logger.info(
        "Wrote ASOS station mapping CSV for %s: %d stations -> %s",
        city_code.upper(), len(rows), output_path,
    )
    return output_path


def load_asos_daily_for_city(
    city_code: str,
    asos_daily_dir: str,
) -> dict[str, pd.DataFrame]:
    """Load all ASOS daily CSVs for a city's station network.

    Returns dict of station_id -> DataFrame with date index.
    Only loads stations that are in the city's asos_station_map.
    """
    cfg = get_city_config(city_code)
    station_data: dict[str, pd.DataFrame] = {}

    for station_id in cfg.asos_station_map:
        csv_path = os.path.join(asos_daily_dir, f"{station_id}_asos_daily.csv")
        if not os.path.exists(csv_path):
            logger.warning(
                "ASOS daily CSV not found for station %s at %s -- skipping",
                station_id, csv_path,
            )
            continue

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        # Filter to configured date range if specified
        if cfg.start_date:
            df = df[df.index >= cfg.start_date]
        if cfg.end_date:
            df = df[df.index <= cfg.end_date]

        station_data[station_id] = df
        logger.info(
            "Loaded ASOS daily for %s: %d rows (%s to %s)",
            station_id, len(df),
            df.index.min().strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
            df.index.max().strftime("%Y-%m-%d") if len(df) > 0 else "N/A",
        )

    logger.info(
        "Loaded %d / %d ASOS daily datasets for %s",
        len(station_data), len(cfg.asos_station_map), city_code.upper(),
    )
    return station_data


def merge_asos_stations(
    station_data: dict[str, pd.DataFrame],
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Merge multiple station ASOS daily DataFrames into wide format.

    Creates columns like {station_id}_TMAX, {station_id}_TMIN for each station.
    Variables default to ["TMAX", "TMIN"] mapped from asos tmax_f, tmin_f.

    Returns a wide-format DataFrame indexed by date.
    """
    if variables is None:
        variables = ["TMAX", "TMIN"]

    # Build reverse mapping: GHCN name -> ASOS column name
    ghcn_to_asos = {v: k for k, v in _ASOS_TO_GHCN.items()}

    frames = []
    for station_id, df in station_data.items():
        renamed_cols = {}
        for var in variables:
            asos_col = ghcn_to_asos.get(var)
            if asos_col and asos_col in df.columns:
                renamed_cols[asos_col] = f"{station_id}_{var}"

        if renamed_cols:
            subset = df[list(renamed_cols.keys())].rename(columns=renamed_cols)
            frames.append(subset)

    if not frames:
        raise ValueError("No station data to merge")

    merged = pd.concat(frames, axis=1).sort_index()
    logger.info("Merged ASOS DataFrame: %d rows x %d columns", *merged.shape)
    return merged


def compute_asos_completeness(
    merged: pd.DataFrame,
    station_ids: list[str],
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-station completeness metrics for ASOS data.

    Returns DataFrame with columns: station_id, variable, non_missing,
    total_days, completeness_pct
    """
    if variables is None:
        variables = ["TMAX", "TMIN"]

    total_days = len(merged)
    records = []

    for sid in station_ids:
        for var in variables:
            col = f"{sid}_{var}"
            if col in merged.columns:
                non_missing = int(merged[col].notna().sum())
                pct = non_missing / total_days if total_days > 0 else 0.0
            else:
                non_missing = 0
                pct = 0.0

            records.append({
                "station_id": sid,
                "variable": var,
                "non_missing": non_missing,
                "total_days": total_days,
                "completeness_pct": pct,
            })

    return pd.DataFrame(records)


def _filter_by_completeness(
    merged: pd.DataFrame,
    target_station: str,
    min_completeness: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove stations below the completeness threshold.

    The target station is never dropped regardless of completeness.

    Returns (filtered DataFrame, list of dropped station IDs).
    """
    # Extract unique station IDs from column names
    all_station_ids = set()
    for col in merged.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            all_station_ids.add(parts[0])

    total_days = len(merged)
    dropped_stations = []
    columns_to_drop = []

    for sid in all_station_ids:
        tmax_col = f"{sid}_TMAX"
        if tmax_col in merged.columns:
            completeness = merged[tmax_col].notna().sum() / total_days
        else:
            completeness = 0.0

        if completeness < min_completeness:
            if sid == target_station:
                logger.warning(
                    "Target station %s has only %.1f%% TMAX completeness "
                    "(below threshold %.0f%%), keeping anyway",
                    sid, completeness * 100, min_completeness * 100,
                )
                continue

            dropped_stations.append(sid)
            for col in merged.columns:
                if col.startswith(f"{sid}_"):
                    columns_to_drop.append(col)

            logger.warning(
                "Dropping station %s: TMAX completeness %.1f%% < %.0f%%",
                sid, completeness * 100, min_completeness * 100,
            )

    if columns_to_drop:
        merged = merged.drop(columns=columns_to_drop)

    logger.info(
        "Kept %d stations, dropped %d: %s",
        len(all_station_ids) - len(dropped_stations),
        len(dropped_stations), dropped_stations,
    )
    return merged, dropped_stations


def _create_target_and_features(
    merged: pd.DataFrame,
    target_station: str,
    city_code: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create target variable and lagged surrounding-station features.

    Target: target_station TMAX on day t.
    Features: surrounding station values from day t-1 (shift +1).

    Returns (features DataFrame, target Series).
    """
    target_col = f"{target_station}_TMAX"
    target_name = _CITY_TARGET_NAMES.get(city_code, f"{city_code.upper()}_TMAX")

    if target_col not in merged.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(merged.columns)[:10]}..."
        )

    # Separate target station columns from surrounding station columns
    target_prefix = f"{target_station}_"
    surrounding_cols = [
        c for c in merged.columns if not c.startswith(target_prefix)
    ]

    # Target: TMAX on day t
    target = merged[target_col].copy()
    target.name = target_name

    # Features: surrounding stations shifted by +1 day (lag = 1)
    features = merged[surrounding_cols].shift(1).copy()
    features.columns = [f"{c}_lag1" for c in features.columns]

    # Drop first row (NaN from shift) and rows where target is missing
    valid_mask = target.notna() & features.notna().any(axis=1)
    valid_mask.iloc[0] = False

    features = features[valid_mask]
    target = target[valid_mask]

    logger.info(
        "Created %s ASOS features: %d rows x %d columns",
        city_code.upper(), *features.shape,
    )
    logger.info(
        "Target (%s): %d non-missing values",
        target_name, target.notna().sum(),
    )
    return features, target


def _chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    cfg: CityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Chronological train/val/test split.

    Austin uses fixed date boundaries; other cities use ratio-based splits.
    """
    if cfg.city_code == "aus":
        # Austin: date-based split
        TRAIN_START, TRAIN_END = "2000-01-01", "2021-12-31"
        CAL_START, CAL_END = "2022-01-01", "2023-12-31"
        TEST_START, TEST_END = "2024-01-01", "2025-12-31"

        idx = features.index
        m_train = (idx >= TRAIN_START) & (idx <= TRAIN_END)
        m_cal = (idx >= CAL_START) & (idx <= CAL_END)
        m_test = (idx >= TEST_START) & (idx <= TEST_END)

        X_train = features.loc[m_train]
        X_val = features.loc[m_cal]
        X_test = features.loc[m_test]
        y_train = target.loc[m_train]
        y_val = target.loc[m_cal]
        y_test = target.loc[m_test]

        logger.info("Chronological split (date-based for Austin):")
    else:
        # Ratio-based split
        n = len(features)
        train_end = int(n * cfg.train_ratio)
        val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

        X_train = features.iloc[:train_end]
        X_val = features.iloc[train_end:val_end]
        X_test = features.iloc[val_end:]
        y_train = target.iloc[:train_end]
        y_val = target.iloc[train_end:val_end]
        y_test = target.iloc[val_end:]

        logger.info("Chronological split (ratio-based):")

    for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        if len(X) > 0:
            logger.info(
                "  %s: %d rows (%s to %s)",
                name, len(X),
                X.index.min().strftime("%Y-%m-%d"),
                X.index.max().strftime("%Y-%m-%d"),
            )
        else:
            logger.info("  %s: 0 rows", name)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_asos_features(
    city_code: str,
    asos_daily_dir: str,
    output_dir: str,
    variables: list[str] | None = None,
) -> dict:
    """Full ASOS-based feature building pipeline for a city.

    Steps:
    1. Load ASOS daily data for all city stations
    2. Merge into wide format with TMAX/TMIN columns
    3. Filter by completeness
    4. Create target (target_station TMAX) and lagged features
    5. Add cyclical date features (sin_day, cos_day)
    6. Handle missing data (forward-fill, then training-mean)
    7. Chronological split
    8. Scale features
    9. Save to output_dir

    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
    scaler, col_means
    """
    if variables is None:
        variables = ["TMAX", "TMIN"]

    cfg = get_city_config(city_code)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load ASOS daily data
    logger.info("Step 1: Loading ASOS daily data for %s from %s", city_code.upper(), asos_daily_dir)
    station_data = load_asos_daily_for_city(city_code, asos_daily_dir)
    if not station_data:
        raise FileNotFoundError(
            f"No ASOS daily data found for {city_code.upper()} in {asos_daily_dir}. "
            "Run ASOS collection and preprocessing first."
        )

    # Step 2: Merge into wide format
    logger.info("Step 2: Merging ASOS station data into wide format")
    merged = merge_asos_stations(station_data, variables=variables)

    # Step 3: Compute completeness and filter
    logger.info("Step 3: Computing completeness and filtering stations")
    station_ids = list(station_data.keys())
    completeness_df = compute_asos_completeness(merged, station_ids, variables=variables)
    logger.info("\nASOS completeness report:\n%s", completeness_df.to_string())

    merged_filtered, dropped_stations = _filter_by_completeness(
        merged, cfg.target_station, cfg.min_completeness,
    )

    # Step 4: Create target and lagged features
    logger.info("Step 4: Creating target and lagged features")
    features, target = _create_target_and_features(
        merged_filtered, cfg.target_station, city_code,
    )

    # Step 5: Add cyclical date features
    logger.info("Step 5: Adding cyclical date features")
    features = add_cyclical_date_features(features)

    # Step 6: Handle missing data -- forward-fill with city config limit
    logger.info("Step 6: Handling missing data (forward-fill limit=%d)", cfg.max_forward_fill_days)
    # Drop rows where target is missing
    valid = target.notna()
    features = features[valid].copy()
    target = target[valid].copy()
    # Forward-fill feature gaps
    features = features.ffill(limit=cfg.max_forward_fill_days)
    remaining_nans = features.isna().sum().sum()
    logger.info("After forward-fill: %d remaining NaN values", remaining_nans)

    # Step 7: Chronological split
    logger.info("Step 7: Chronological split")
    X_train, X_val, X_test, y_train, y_val, y_test = _chronological_split(
        features, target, cfg,
    )

    # Step 8a: Fill remaining NaNs with training-set means (no leakage)
    logger.info("Step 8a: Filling remaining NaNs with training means")
    col_means = X_train.mean()

    train_nans = X_train.isna().sum().sum()
    val_nans = X_val.isna().sum().sum()
    test_nans = X_test.isna().sum().sum()

    X_train = X_train.fillna(col_means)
    X_val = X_val.fillna(col_means)
    X_test = X_test.fillna(col_means)

    logger.info(
        "Filled remaining NaNs with training means: train=%d, val=%d, test=%d",
        train_nans, val_nans, test_nans,
    )

    # Step 8b: Scale features (fit on training only)
    logger.info("Step 8b: Scaling features (StandardScaler fit on training data)")
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )

    logger.info("StandardScaler fit on %d training features", X_train.shape[1])

    # Step 9: Save outputs
    logger.info("Step 9: Saving processed ASOS features to %s", output_dir)
    X_train_scaled.to_csv(os.path.join(output_dir, "features_train.csv"))
    X_val_scaled.to_csv(os.path.join(output_dir, "features_val.csv"))
    X_test_scaled.to_csv(os.path.join(output_dir, "features_test.csv"))
    y_train.to_csv(os.path.join(output_dir, "target_train.csv"), header=True)
    y_val.to_csv(os.path.join(output_dir, "target_val.csv"), header=True)
    y_test.to_csv(os.path.join(output_dir, "target_test.csv"), header=True)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "col_means.pkl"), "wb") as f:
        pickle.dump(col_means, f)

    # Save completeness report
    completeness_df.to_csv(
        os.path.join(output_dir, "asos_completeness.csv"), index=False,
    )

    logger.info("=" * 60)
    logger.info("ASOS Feature Building Complete for %s", city_code.upper())
    logger.info("=" * 60)
    logger.info("Features: %d columns", X_train_scaled.shape[1])
    logger.info(
        "Train: %d rows, Val: %d rows, Test: %d rows",
        len(X_train_scaled), len(X_val_scaled), len(X_test_scaled),
    )
    logger.info("Output saved to: %s", output_dir)

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "col_means": col_means,
    }
