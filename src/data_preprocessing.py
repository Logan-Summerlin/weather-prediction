"""
Data Preprocessing Module for NYC Temperature Prediction.

Loads per-station CSVs from data/raw/, merges into a unified DataFrame,
creates lag features, handles missing data, performs chronological
train/val/test split, scales features, and saves processed outputs.

Key design decisions:
  - Features for day t use surrounding-station data from day t-1 (shift +1).
  - Cyclical date encoding: sin(2*pi*doy/365.25) and cos(2*pi*doy/365.25).
  - Missing data: forward-fill gaps <= 3 days, then impute with training-set
    column means. Stations with < 90% completeness are dropped.
  - Scaler is fit on training data ONLY, then applied to val/test.
  - Splits are strictly chronological — no shuffling.
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Loading
# ===========================================================================

def load_station_csv(csv_path: str) -> pd.DataFrame:
    """Load a single station CSV (output of data_collection).

    Parameters
    ----------
    csv_path : str
        Path to the station CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and TMAX/TMIN columns.
    """
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    return df


def load_all_stations(raw_dir: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Load all station CSVs from the raw data directory.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing station CSVs. Defaults to config.RAW_DATA_DIR.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of station_id -> DataFrame.
    """
    if raw_dir is None:
        raw_dir = config.RAW_DATA_DIR

    station_data = {}

    for station_id in config.ALL_STATIONS:
        csv_path = os.path.join(raw_dir, f"{station_id}.csv")
        if not os.path.exists(csv_path):
            logger.warning("CSV not found for station %s at %s — skipping",
                           station_id, csv_path)
            continue

        df = load_station_csv(csv_path)
        station_data[station_id] = df
        logger.info("Loaded %s: %d rows, columns=%s",
                     station_id, len(df), list(df.columns))

    logger.info("Loaded %d station datasets", len(station_data))
    return station_data


# ===========================================================================
# Merging
# ===========================================================================

def merge_stations(station_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all station DataFrames into a single wide DataFrame.

    Creates columns like {station_id}_TMAX and {station_id}_TMIN for each
    station. The index is the date.

    Parameters
    ----------
    station_data : dict[str, pd.DataFrame]
        Mapping of station_id -> DataFrame with TMAX/TMIN columns.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by date.
    """
    frames = []

    for station_id, df in station_data.items():
        # Rename columns to include station ID
        renamed = df.rename(
            columns={col: f"{station_id}_{col}" for col in df.columns}
        )
        frames.append(renamed)

    if not frames:
        raise ValueError("No station data to merge")

    merged = pd.concat(frames, axis=1)
    merged = merged.sort_index()

    logger.info("Merged DataFrame: %d rows x %d columns", *merged.shape)
    return merged


# ===========================================================================
# Data Quality
# ===========================================================================

def compute_completeness(merged_df: pd.DataFrame,
                         station_ids: list[str],
                         variables: list[str]) -> pd.DataFrame:
    """Compute data completeness per station.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame.
    station_ids : list[str]
        List of station IDs to check.
    variables : list[str]
        List of variable names (e.g., ["TMAX", "TMIN"]).

    Returns
    -------
    pd.DataFrame
        Completeness report with columns: station_id, variable, total_days,
        non_missing, completeness_pct.
    """
    total_days = len(merged_df)
    records = []

    for sid in station_ids:
        for var in variables:
            col = f"{sid}_{var}"
            if col in merged_df.columns:
                non_missing = merged_df[col].notna().sum()
                pct = non_missing / total_days if total_days > 0 else 0.0
            else:
                non_missing = 0
                pct = 0.0

            records.append({
                "station_id": sid,
                "variable": var,
                "total_days": total_days,
                "non_missing": non_missing,
                "completeness_pct": pct,
            })

    return pd.DataFrame(records)


def filter_stations_by_completeness(
    merged_df: pd.DataFrame,
    min_completeness: float = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove stations that fall below the minimum completeness threshold.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame.
    min_completeness : float, optional
        Minimum fraction of non-missing days required. Defaults to
        config.MIN_COMPLETENESS.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (filtered DataFrame, list of dropped station IDs)
    """
    if min_completeness is None:
        min_completeness = config.MIN_COMPLETENESS

    all_station_ids = set()
    for col in merged_df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            all_station_ids.add(parts[0])

    total_days = len(merged_df)
    dropped_stations = []
    columns_to_drop = []

    for sid in all_station_ids:
        # Check TMAX completeness (primary variable)
        tmax_col = f"{sid}_TMAX"
        if tmax_col in merged_df.columns:
            completeness = merged_df[tmax_col].notna().sum() / total_days
        else:
            completeness = 0.0

        if completeness < min_completeness:
            # Don't drop the target station regardless of completeness
            if sid == config.TARGET_STATION:
                logger.warning(
                    "Target station %s has only %.1f%% TMAX completeness "
                    "(below threshold %.0f%%), keeping anyway",
                    sid, completeness * 100, min_completeness * 100
                )
                continue

            dropped_stations.append(sid)
            # Find all columns for this station
            for col in merged_df.columns:
                if col.startswith(f"{sid}_"):
                    columns_to_drop.append(col)

            logger.warning(
                "Dropping station %s: TMAX completeness %.1f%% < %.0f%%",
                sid, completeness * 100, min_completeness * 100
            )

    if columns_to_drop:
        merged_df = merged_df.drop(columns=columns_to_drop)

    logger.info("Kept %d stations, dropped %d: %s",
                len(all_station_ids) - len(dropped_stations),
                len(dropped_stations), dropped_stations)

    return merged_df, dropped_stations


# ===========================================================================
# Feature Engineering
# ===========================================================================

def create_target_and_features(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the target variable and lagged feature columns.

    Target: NYC TMAX on day t (the Central Park value).
    Features: All surrounding-station values shifted by +1 (so row for day t
    contains surrounding-station data from day t-1).

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged wide-format DataFrame with all stations.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features DataFrame, target Series), both indexed by date.
        Rows where the target is missing are dropped.
    """
    target_col = f"{config.TARGET_STATION}_TMAX"

    if target_col not in merged_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(merged_df.columns)[:10]}..."
        )

    # Identify surrounding-station columns (all columns except target station's)
    target_prefix = f"{config.TARGET_STATION}_"
    surrounding_cols = [
        c for c in merged_df.columns if not c.startswith(target_prefix)
    ]

    # The target: NYC TMAX on day t
    target = merged_df[target_col].copy()
    target.name = "NYC_TMAX"

    # Features: surrounding stations shifted by +1 day (lag = 1)
    # shift(1) means: row at date t gets the value from date t-1
    features = merged_df[surrounding_cols].shift(1).copy()

    # Rename feature columns to indicate they are lagged
    features.columns = [f"{c}_lag1" for c in features.columns]

    # Drop the first row (which has NaN features due to shift)
    # and rows where target is missing
    valid_mask = target.notna() & features.notna().any(axis=1)
    # After shift, the first row is always NaN
    valid_mask.iloc[0] = False

    features = features[valid_mask]
    target = target[valid_mask]

    logger.info("Created features: %d rows x %d columns", *features.shape)
    logger.info("Target: %d non-missing values", target.notna().sum())

    return features, target


def add_cyclical_date_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encoding of the day of year.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Features with added 'sin_day' and 'cos_day' columns.
    """
    day_of_year = features.index.dayofyear
    features = features.copy()
    features["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)

    logger.info("Added cyclical date features (sin_day, cos_day)")
    return features


# ===========================================================================
# Missing Data Handling
# ===========================================================================

def handle_missing_data(
    features: pd.DataFrame,
    target: pd.Series,
    max_fill_days: int = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Handle missing values in the feature matrix.

    Strategy:
      1. Drop rows where the target (NYC TMAX) is missing.
      2. Forward-fill feature gaps of <= max_fill_days consecutive NaNs.
      3. Remaining NaNs will be filled with training-set column means later
         (after the split, to avoid data leakage).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Target variable.
    max_fill_days : int, optional
        Maximum gap length for forward-fill. Defaults to
        config.MAX_FORWARD_FILL_DAYS.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (cleaned features, cleaned target) with aligned indices.
    """
    if max_fill_days is None:
        max_fill_days = config.MAX_FORWARD_FILL_DAYS

    # 1. Drop rows where target is missing
    valid = target.notna()
    features = features[valid].copy()
    target = target[valid].copy()

    # 2. Forward-fill gaps <= max_fill_days
    # ffill with limit fills up to `limit` consecutive NaNs
    features = features.ffill(limit=max_fill_days)

    missing_before = features.isna().sum().sum()
    logger.info("After forward-fill (limit=%d): %d remaining NaN values",
                max_fill_days, missing_before)

    return features, target


def fill_remaining_nans_with_train_means(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fill remaining NaNs using training-set column means.

    The means are computed from the training set only and applied to all
    three splits, avoiding data leakage.

    Parameters
    ----------
    train_features : pd.DataFrame
    val_features : pd.DataFrame
    test_features : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]
        (train, val, test, column_means) with NaNs filled.
    """
    col_means = train_features.mean()

    train_nans = train_features.isna().sum().sum()
    val_nans = val_features.isna().sum().sum()
    test_nans = test_features.isna().sum().sum()

    train_filled = train_features.fillna(col_means)
    val_filled = val_features.fillna(col_means)
    test_filled = test_features.fillna(col_means)

    logger.info("Filled remaining NaNs with training means: "
                "train=%d, val=%d, test=%d", train_nans, val_nans, test_nans)

    return train_filled, val_filled, test_filled, col_means


# ===========================================================================
# Splitting
# ===========================================================================

def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = None,
    val_ratio: float = None,
) -> tuple:
    """Split data chronologically into train/val/test sets.

    IMPORTANT: No shuffling. The first train_ratio fraction is training,
    the next val_ratio is validation, and the remainder is test.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix sorted by date.
    target : pd.Series
        Target variable sorted by date.
    train_ratio : float, optional
        Fraction for training. Defaults to config.TRAIN_RATIO.
    val_ratio : float, optional
        Fraction for validation. Defaults to config.VAL_RATIO.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO

    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = features.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]

    y_train = target.iloc[:train_end]
    y_val = target.iloc[train_end:val_end]
    y_test = target.iloc[val_end:]

    logger.info("Chronological split:")
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


# ===========================================================================
# Scaling
# ===========================================================================

def fit_and_apply_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and transform all splits.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_val : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]
        (scaled_train, scaled_val, scaled_test, fitted_scaler)
    """
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

    logger.info("StandardScaler fit on training data (%d features)", X_train.shape[1])
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ===========================================================================
# Saving
# ===========================================================================

def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    output_dir: Optional[str] = None,
) -> None:
    """Save all processed data and artifacts to disk.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Scaled feature matrices.
    y_train, y_val, y_test : pd.Series
        Target vectors.
    scaler : StandardScaler
        The fitted scaler (for inverse transforms later).
    output_dir : str, optional
        Output directory. Defaults to config.PROCESSED_DATA_DIR.
    """
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "features_train.csv"))
    X_val.to_csv(os.path.join(output_dir, "features_val.csv"))
    X_test.to_csv(os.path.join(output_dir, "features_test.csv"))
    y_train.to_csv(os.path.join(output_dir, "target_train.csv"), header=True)
    y_val.to_csv(os.path.join(output_dir, "target_val.csv"), header=True)
    y_test.to_csv(os.path.join(output_dir, "target_test.csv"), header=True)

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Saved processed data to %s", output_dir)
    logger.info("  features_train.csv: %s", X_train.shape)
    logger.info("  features_val.csv:   %s", X_val.shape)
    logger.info("  features_test.csv:  %s", X_test.shape)
    logger.info("  target_train.csv:   %d rows", len(y_train))
    logger.info("  target_val.csv:     %d rows", len(y_val))
    logger.info("  target_test.csv:    %d rows", len(y_test))
    logger.info("  scaler.pkl saved")


def generate_preprocessing_report(
    merged_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
    dropped_stations: list[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[str] = None,
) -> str:
    """Generate a human-readable data quality and preprocessing report.

    Parameters
    ----------
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
    output_dir : str, optional
        Output directory. Defaults to config.PROCESSED_DATA_DIR.

    Returns
    -------
    str
        The report text.
    """
    if output_dir is None:
        output_dir = config.PROCESSED_DATA_DIR

    lines = [
        "=" * 70,
        "NYC Temperature Prediction — Preprocessing Report",
        "=" * 70,
        "",
        f"Date range: {config.START_DATE} to {config.END_DATE}",
        f"Target station: {config.TARGET_STATION}",
        f"Minimum completeness threshold: {config.MIN_COMPLETENESS * 100:.0f}%",
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
        f"Feature names: {list(X_train.columns)}",
        "",
        "--- Target Statistics (training set, unscaled °F) ---",
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

def run_preprocessing_pipeline(raw_dir: Optional[str] = None,
                                output_dir: Optional[str] = None) -> dict:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    raw_dir : str, optional
        Directory with station CSVs. Defaults to config.RAW_DATA_DIR.
    output_dir : str, optional
        Directory for processed outputs. Defaults to config.PROCESSED_DATA_DIR.

    Returns
    -------
    dict
        Dictionary containing all processed data and metadata.
    """
    logger.info("=" * 60)
    logger.info("Starting Preprocessing Pipeline")
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

    # 3. Compute completeness report
    all_station_ids = list(station_data.keys())
    completeness_df = compute_completeness(
        merged_df, all_station_ids, config.INPUT_VARIABLES
    )
    logger.info("\nCompleteness report:\n%s", completeness_df.to_string())

    # 4. Filter out low-completeness stations
    merged_filtered, dropped_stations = filter_stations_by_completeness(merged_df)

    # 5. Create target and lagged features
    features, target = create_target_and_features(merged_filtered)

    # 6. Add cyclical date features
    features = add_cyclical_date_features(features)

    # 7. Handle missing data (forward-fill)
    features, target = handle_missing_data(features, target)

    # 8. Chronological split
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        features, target
    )

    # 9. Fill remaining NaNs with training-set means (no leakage)
    X_train, X_val, X_test, col_means = fill_remaining_nans_with_train_means(
        X_train, X_val, X_test
    )

    # 10. Fit scaler on training data, apply to all
    X_train_s, X_val_s, X_test_s, scaler = fit_and_apply_scaler(
        X_train, X_val, X_test
    )

    # 11. Save everything
    save_processed_data(
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        scaler, output_dir
    )

    # 12. Generate report
    report = generate_preprocessing_report(
        merged_df, completeness_df, dropped_stations,
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        output_dir,
    )
    logger.info("\n%s", report)

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "completeness": completeness_df,
        "dropped_stations": dropped_stations,
    }


def main():
    """Main entry point for data preprocessing."""
    result = run_preprocessing_pipeline()
    logger.info("Preprocessing complete!")
    return result


if __name__ == "__main__":
    main()
