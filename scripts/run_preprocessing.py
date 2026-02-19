#!/usr/bin/env python3
"""
Unified Multi-City Preprocessing Pipeline.

Processes raw GHCN-Daily station CSVs into training-ready feature matrices
for temperature prediction models.

Pipeline steps:
  1. Load all station CSVs from data/<city>/raw/
  2. Merge into a wide DataFrame (station_id_TMAX, station_id_TMIN, ...)
  3. Filter stations below completeness threshold
  4. Create target (TMAX) and lagged surrounding-station features
  5. Add cyclical date features (sin_day, cos_day)
  6. Handle missing data (forward-fill, then training-mean imputation)
  7. Chronological split (ratio-based or date-based depending on city config)
  8. Scale features (StandardScaler fit on training set only)
  9. Save processed outputs to data/<city>/processed/

Replaces the per-city scripts:
  - run_chi_preprocessing.py
  - run_phl_preprocessing.py
  - run_atl_preprocessing.py
  - run_aus_preprocessing.py

Usage:
    python scripts/run_preprocessing.py --city chi
    python scripts/run_preprocessing.py --city phl
    python scripts/run_preprocessing.py --city atl
    python scripts/run_preprocessing.py --city aus
"""

import argparse
import importlib
import os
import sys
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, ensure_city_dirs
from src.data_preprocessing import (
    load_station_csv,
    merge_stations,
    compute_completeness,
    add_cyclical_date_features,
    handle_missing_data,
    fill_remaining_nans_with_train_means,
    fit_and_apply_scaler,
)

# ---------------------------------------------------------------------------
# City code -> config module mapping
# ---------------------------------------------------------------------------
CITY_CONFIG_MODULES = {
    "nyc": "config_expanded",
    "chi": "config_chicago",
    "phl": "config_philadelphia",
    "atl": "config_atlanta",
    "aus": "config_austin",
}

# City code -> target name prefix (used for target Series naming)
CITY_TARGET_NAMES = {
    "chi": "CHI_TMAX",
    "phl": "PHL_TMAX",
    "atl": "ATL_TMAX",
    "aus": "AUS_TMAX",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Generic Functions (parameterized by city config)
# ===========================================================================

def load_stations(raw_dir: str, city_config, city_code: str) -> dict[str, pd.DataFrame]:
    """Load all station CSVs from the raw data directory.

    Parameters
    ----------
    raw_dir : str
        Directory containing per-station CSV files.
    city_config : module
        City-specific config module with ALL_STATIONS.
    city_code : str
        Short city code for logging.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of station_id -> DataFrame with DatetimeIndex.
    """
    station_data = {}
    for station_id in city_config.ALL_STATIONS:
        csv_path = os.path.join(raw_dir, f"{station_id}.csv")
        if not os.path.exists(csv_path):
            logger.warning("CSV not found for %s at %s -- skipping", station_id, csv_path)
            continue
        df = load_station_csv(csv_path)
        station_data[station_id] = df
        logger.info("Loaded %s: %d rows, columns=%s",
                     station_id, len(df), list(df.columns))
    logger.info("Loaded %d / %d %s station datasets",
                len(station_data), len(city_config.ALL_STATIONS), city_code.upper())
    return station_data


def filter_stations_by_completeness(
    merged_df: pd.DataFrame,
    city_config,
    min_completeness: float = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove stations that fall below the minimum completeness threshold.

    The target station is never dropped regardless of completeness.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame.
    city_config : module
        City-specific config module with TARGET_STATION, MIN_COMPLETENESS.
    min_completeness : float, optional
        Override threshold. Defaults to city_config.MIN_COMPLETENESS.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (filtered DataFrame, list of dropped station IDs)
    """
    if min_completeness is None:
        min_completeness = city_config.MIN_COMPLETENESS

    target_station = city_config.TARGET_STATION

    # Extract all unique station IDs from column names
    all_station_ids = set()
    for col in merged_df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            all_station_ids.add(parts[0])

    total_days = len(merged_df)
    dropped_stations = []
    columns_to_drop = []

    for sid in all_station_ids:
        tmax_col = f"{sid}_TMAX"
        if tmax_col in merged_df.columns:
            completeness = merged_df[tmax_col].notna().sum() / total_days
        else:
            completeness = 0.0

        if completeness < min_completeness:
            # Never drop the target station
            if sid == target_station:
                logger.warning(
                    "Target station %s has only %.1f%% TMAX completeness "
                    "(below threshold %.0f%%), keeping anyway",
                    sid, completeness * 100, min_completeness * 100,
                )
                continue

            dropped_stations.append(sid)
            for col in merged_df.columns:
                if col.startswith(f"{sid}_"):
                    columns_to_drop.append(col)

            logger.warning(
                "Dropping station %s: TMAX completeness %.1f%% < %.0f%%",
                sid, completeness * 100, min_completeness * 100,
            )

    if columns_to_drop:
        merged_df = merged_df.drop(columns=columns_to_drop)

    logger.info("Kept %d stations, dropped %d: %s",
                len(all_station_ids) - len(dropped_stations),
                len(dropped_stations), dropped_stations)
    return merged_df, dropped_stations


def create_target_and_features(
    merged_df: pd.DataFrame,
    city_config,
    city_code: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the target variable and lagged feature columns.

    Target: city TMAX on day t.
    Features: All surrounding-station values shifted by +1 (so row for day t
    contains surrounding-station data from day t-1).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame with all stations.
    city_config : module
        City-specific config module with TARGET_STATION.
    city_code : str
        Short city code for target naming.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features DataFrame, target Series), both indexed by date.
    """
    target_station = city_config.TARGET_STATION
    target_col = f"{target_station}_TMAX"
    target_name = CITY_TARGET_NAMES.get(city_code, f"{city_code.upper()}_TMAX")

    if target_col not in merged_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(merged_df.columns)[:10]}..."
        )

    # Identify surrounding-station columns (everything except target station)
    target_prefix = f"{target_station}_"
    surrounding_cols = [
        c for c in merged_df.columns if not c.startswith(target_prefix)
    ]

    # Target: city TMAX on day t
    target = merged_df[target_col].copy()
    target.name = target_name

    # Features: surrounding stations shifted by +1 day (lag = 1)
    features = merged_df[surrounding_cols].shift(1).copy()
    features.columns = [f"{c}_lag1" for c in features.columns]

    # Drop first row (NaN from shift) and rows where target is missing
    valid_mask = target.notna() & features.notna().any(axis=1)
    valid_mask.iloc[0] = False

    features = features[valid_mask]
    target = target[valid_mask]

    logger.info("Created %s features: %d rows x %d columns",
                city_code.upper(), *features.shape)
    logger.info("Target (%s): %d non-missing values", target_name, target.notna().sum())
    return features, target


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    city_config,
    city_code: str,
) -> tuple:
    """Split data chronologically into train/val/test sets.

    Uses date-based splitting if the city config has SPLIT_BY_DATE=True
    (Austin uses fixed date boundaries). Otherwise uses ratio-based splitting.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix sorted by date.
    target : pd.Series
        Target variable sorted by date.
    city_config : module
        City-specific config module with TRAIN_RATIO, VAL_RATIO.
    city_code : str
        Short city code for logging.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Check if this city uses date-based splitting
    use_date_split = getattr(city_config, 'SPLIT_BY_DATE', False)

    # Austin uses date-based splits by default
    if city_code == "aus" or use_date_split:
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

        logger.info("Chronological split (date-based):")
    else:
        train_ratio = city_config.TRAIN_RATIO
        val_ratio = city_config.VAL_RATIO

        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = features.iloc[:train_end]
        X_val = features.iloc[train_end:val_end]
        X_test = features.iloc[val_end:]

        y_train = target.iloc[:train_end]
        y_val = target.iloc[train_end:val_end]
        y_test = target.iloc[val_end:]

        logger.info("Chronological split (ratio-based):")

    logger.info("  Train: %d rows (%s to %s)",
                len(X_train),
                X_train.index.min().strftime("%Y-%m-%d") if len(X_train) > 0 else "N/A",
                X_train.index.max().strftime("%Y-%m-%d") if len(X_train) > 0 else "N/A")
    logger.info("  Val:   %d rows (%s to %s)",
                len(X_val),
                X_val.index.min().strftime("%Y-%m-%d") if len(X_val) > 0 else "N/A",
                X_val.index.max().strftime("%Y-%m-%d") if len(X_val) > 0 else "N/A")
    logger.info("  Test:  %d rows (%s to %s)",
                len(X_test),
                X_test.index.min().strftime("%Y-%m-%d") if len(X_test) > 0 else "N/A",
                X_test.index.max().strftime("%Y-%m-%d") if len(X_test) > 0 else "N/A")

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_preprocessing_report(
    city_name: str,
    city_config,
    merged_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
    dropped_stations: list[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> str:
    """Generate a human-readable preprocessing report.

    Parameters
    ----------
    city_name : str
        Human-readable city name.
    city_config : module
        City-specific config module.
    merged_df : pd.DataFrame
        The merged wide-format DataFrame (before feature engineering).
    completeness_df : pd.DataFrame
        Per-station completeness metrics.
    dropped_stations : list[str]
        Station IDs removed for low completeness.
    X_train, X_val, X_test : pd.DataFrame
        Final feature matrices.
    y_train, y_val, y_test : pd.Series
        Final target vectors.
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        The report text.
    """
    lines = [
        "=" * 70,
        f"{city_name} Temperature Prediction -- Preprocessing Report",
        "=" * 70,
        "",
        f"Date range: {city_config.START_DATE} to {city_config.END_DATE}",
        f"Target station: {city_config.TARGET_STATION}",
        f"Minimum completeness threshold: {city_config.MIN_COMPLETENESS * 100:.0f}%",
        "",
        "--- Station Data Completeness ---",
        "",
    ]

    for _, row in completeness_df.iterrows():
        status = "KEPT" if row["station_id"] not in dropped_stations else "DROPPED"
        lines.append(
            f"  {row['station_id']} | {row['variable']:4s} | "
            f"{row['non_missing']:5.0f}/{row['total_days']:.0f} days | "
            f"{row['completeness_pct'] * 100:5.1f}% | {status}"
        )

    lines.extend([
        "",
        f"Stations dropped for low completeness: {dropped_stations or 'None'}",
        "",
        "--- Merged Data ---",
        "",
        f"Total date range rows: {len(merged_df)}",
        f"Total columns: {merged_df.shape[1]}",
        "",
        "--- Train / Validation / Test Split ---",
        "",
        f"Train: {len(X_train)} rows "
        f"({X_train.index.min().strftime('%Y-%m-%d') if len(X_train) else 'N/A'} to "
        f"{X_train.index.max().strftime('%Y-%m-%d') if len(X_train) else 'N/A'})",
        f"Val:   {len(X_val)} rows "
        f"({X_val.index.min().strftime('%Y-%m-%d') if len(X_val) else 'N/A'} to "
        f"{X_val.index.max().strftime('%Y-%m-%d') if len(X_val) else 'N/A'})",
        f"Test:  {len(X_test)} rows "
        f"({X_test.index.min().strftime('%Y-%m-%d') if len(X_test) else 'N/A'} to "
        f"{X_test.index.max().strftime('%Y-%m-%d') if len(X_test) else 'N/A'})",
        "",
        f"Feature columns: {X_train.shape[1]}",
        "",
        "--- Target Statistics (training set, unscaled deg F) ---",
        "",
        f"Mean:   {y_train.mean():.1f}",
        f"Std:    {y_train.std():.1f}",
        f"Min:    {y_train.min():.1f}",
        f"Max:    {y_train.max():.1f}",
        f"Median: {y_train.median():.1f}",
        "",
        "=" * 70,
    ])

    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "preprocessing_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("Preprocessing report saved to %s", report_path)
    return report


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    """Execute the full preprocessing pipeline for a specified city."""
    parser = argparse.ArgumentParser(
        description="Preprocess GHCN-Daily station data for a city."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=sorted(CITY_CONFIG_MODULES.keys()),
        help="City code (chi, phl, atl, aus)",
    )
    args = parser.parse_args()
    city_code = args.city

    # Load configs
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)
    city_config = importlib.import_module(CITY_CONFIG_MODULES[city_code])

    logger.info("=" * 60)
    logger.info("%s Preprocessing Pipeline", cfg.city_name)
    logger.info("=" * 60)

    raw_dir = os.path.join(cfg.data_dir, "raw")
    processed_dir = os.path.join(cfg.data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Austin also needs models dir
    if city_code == "aus":
        os.makedirs(cfg.models_dir, exist_ok=True)

    # 1. Load station data
    logger.info("Step 1: Loading %s station data from %s", city_code.upper(), raw_dir)
    station_data = load_stations(raw_dir, city_config, city_code)
    if not station_data:
        logger.error("No station data found. Run run_data_collection.py --city %s first.",
                      city_code)
        return

    # 2. Merge into single DataFrame
    logger.info("Step 2: Merging station data")
    merged = merge_stations(station_data)

    # 3. Compute completeness report
    logger.info("Step 3: Computing completeness")
    all_station_ids = list(station_data.keys())
    completeness_df = compute_completeness(
        merged, all_station_ids, city_config.INPUT_VARIABLES
    )
    logger.info("\nCompleteness report:\n%s", completeness_df.to_string())

    # 4. Filter out low-completeness stations
    logger.info("Step 4: Filtering by completeness (threshold=%.0f%%)",
                city_config.MIN_COMPLETENESS * 100)
    merged_filtered, dropped_stations = filter_stations_by_completeness(
        merged, city_config
    )

    # 5. Create target and lagged features
    logger.info("Step 5: Creating target and lagged features")
    features, target = create_target_and_features(merged_filtered, city_config, city_code)

    # 6. Add cyclical date features
    logger.info("Step 6: Adding cyclical date features")
    features = add_cyclical_date_features(features)

    # 7. Handle missing data (forward-fill with city config limit)
    logger.info("Step 7: Handling missing data")
    features, target = handle_missing_data(
        features, target, max_fill_days=city_config.MAX_FORWARD_FILL_DAYS
    )

    # 8. Chronological split
    logger.info("Step 8: Chronological split")
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        features, target, city_config, city_code
    )

    # 9. Fill remaining NaNs with training-set means (no leakage)
    logger.info("Step 9: Filling remaining NaNs with training means")
    X_train, X_val, X_test, col_means = fill_remaining_nans_with_train_means(
        X_train, X_val, X_test
    )

    # 10. Fit scaler on training data, apply to all
    logger.info("Step 10: Scaling features")
    X_train_s, X_val_s, X_test_s, scaler = fit_and_apply_scaler(
        X_train, X_val, X_test
    )

    # 11. Save everything
    logger.info("Step 11: Saving processed data to %s", processed_dir)
    X_train_s.to_csv(os.path.join(processed_dir, "features_train.csv"))
    X_val_s.to_csv(os.path.join(processed_dir, "features_val.csv"))
    X_test_s.to_csv(os.path.join(processed_dir, "features_test.csv"))
    y_train.to_csv(os.path.join(processed_dir, "target_train.csv"), header=True)
    y_val.to_csv(os.path.join(processed_dir, "target_val.csv"), header=True)
    y_test.to_csv(os.path.join(processed_dir, "target_test.csv"), header=True)

    with open(os.path.join(processed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(processed_dir, "col_means.pkl"), "wb") as f:
        pickle.dump(col_means, f)

    # 12. Generate report
    logger.info("Step 12: Generating preprocessing report")
    report = generate_preprocessing_report(
        cfg.city_name, city_config,
        merged, completeness_df, dropped_stations,
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        processed_dir,
    )
    logger.info("\n%s", report)

    # 13. Summary
    logger.info("=" * 60)
    logger.info("%s Preprocessing Complete", cfg.city_name)
    logger.info("=" * 60)
    logger.info("Features: %d columns", X_train_s.shape[1])
    logger.info("Train: %d rows, Val: %d rows, Test: %d rows",
                len(X_train_s), len(X_val_s), len(X_test_s))
    logger.info("Processed data saved to: %s", processed_dir)
    logger.info("Next step: run scripts/run_benchmark.py --city %s", city_code)


if __name__ == "__main__":
    main()
