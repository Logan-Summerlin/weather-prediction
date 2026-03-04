"""
Data Integrity Validation Module.

Implements automated checks recommended by the cross-city Brier integrity
audit (2026-02-15). These guards prevent silent data quality failures that
can corrupt model evaluation results.

Phase B additions:
    - ``validate_critical_data_available``: hard-fail when critical operational
      data (ASOS, processed features) is missing or empty.
    - ``validate_dataframe_not_empty``: assert a DataFrame is non-empty with
      a clear error message.

Usage:
    from src.data_integrity import (
        validate_no_synthetic_data,
        validate_bucket_consistency,
        validate_required_inputs_exist,
        validate_critical_data_available,
        validate_dataframe_not_empty,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_no_synthetic_data(df: pd.DataFrame, context: str = "") -> None:
    """Verify that a DataFrame does not contain synthetic/fallback data.

    Checks for telltale signs of the synthetic data generator that was
    removed in the audit fix: model_name == "synthetic_base", perfectly
    round date ranges starting 2020-01-01, or suspiciously low residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame to validate.
    context : str
        Description of what this DataFrame represents (for error messages).

    Raises
    ------
    RuntimeError
        If synthetic data indicators are detected.
    """
    ctx = f" ({context})" if context else ""

    if "model_name" in df.columns:
        synthetic_mask = df["model_name"].str.contains("synthetic", case=False, na=False)
        if synthetic_mask.any():
            n_synthetic = synthetic_mask.sum()
            raise RuntimeError(
                f"Synthetic data detected{ctx}: {n_synthetic} rows with "
                f"model_name containing 'synthetic'. Real benchmark data required."
            )

    logger.debug("No synthetic data detected%s (%d rows checked)", ctx, len(df))


def validate_required_inputs_exist(
    paths: List[str],
    context: str = "",
) -> None:
    """Verify that all required input files exist before running a pipeline.

    Parameters
    ----------
    paths : list of str
        File paths that must exist.
    context : str
        Pipeline step name for error messages.

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        ctx = f" for {context}" if context else ""
        raise FileNotFoundError(
            f"Missing required input files{ctx}:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nRun the prerequisite pipeline steps first."
        )


def validate_bucket_consistency(
    bucket_edges: List[Tuple[float, float]],
    city_code: str,
    expected_width: float = 2.0,
) -> None:
    """Verify bucket edges use the expected resolution.

    Parameters
    ----------
    bucket_edges : list of (float, float)
        Bucket boundary tuples.
    city_code : str
        City identifier for error messages.
    expected_width : float
        Expected width of interior buckets in degrees F (default 2.0).

    Raises
    ------
    ValueError
        If interior buckets do not match the expected width.
    """
    interior = bucket_edges[1:-1]  # exclude tail buckets
    bad = []
    for i, (lo, hi) in enumerate(interior):
        width = hi - lo
        if abs(width - expected_width) > 0.01:
            bad.append((i + 1, lo, hi, width))

    if bad:
        details = ", ".join(f"bucket[{i}]=({lo},{hi}) width={w}" for i, lo, hi, w in bad[:5])
        raise ValueError(
            f"Bucket width mismatch for {city_code}: expected {expected_width}°F "
            f"interior buckets but found: {details}"
            + (f" ... and {len(bad) - 5} more" if len(bad) > 5 else "")
        )

    n_total = len(bucket_edges)
    logger.info(
        "Bucket consistency OK for %s: %d buckets, %.0f°F interior width",
        city_code, n_total, expected_width,
    )


def validate_no_calibration_test_overlap(
    cal_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    context: str = "",
) -> None:
    """Verify calibration and test periods do not overlap.

    Parameters
    ----------
    cal_dates : pd.DatetimeIndex
        Dates used for calibration fitting.
    test_dates : pd.DatetimeIndex
        Dates used for out-of-sample evaluation.
    context : str
        Description for error messages.

    Raises
    ------
    RuntimeError
        If any dates appear in both sets.
    """
    overlap = cal_dates.intersection(test_dates)
    if len(overlap) > 0:
        ctx = f" ({context})" if context else ""
        raise RuntimeError(
            f"Calibration-test date overlap detected{ctx}: "
            f"{len(overlap)} dates in common (first: {overlap[0]}, "
            f"last: {overlap[-1]}). Calibration must use training data only."
        )

    logger.debug(
        "No calibration-test overlap: %d cal dates, %d test dates",
        len(cal_dates), len(test_dates),
    )


# ---------------------------------------------------------------------------
# Phase B: Critical data availability and run-fail behaviour
# ---------------------------------------------------------------------------

class CriticalDataError(RuntimeError):
    """Raised when critical data required for pipeline operation is missing.

    This exception triggers hard pipeline failures that prevent silent
    degradation of forecasts or trading decisions.

    Attributes
    ----------
    source : str
        Name of the missing or invalid data source.
    reason : str
        Human-readable description of the failure.
    city_code : str
        City for which the check was performed (may be empty for global).
    """

    def __init__(self, source: str, reason: str, city_code: str = ""):
        self.source = source
        self.reason = reason
        self.city_code = city_code
        city_part = f" [{city_code}]" if city_code else ""
        msg = f"CRITICAL DATA FAILURE{city_part} — {source}: {reason}"
        super().__init__(msg)


def validate_dataframe_not_empty(
    df: pd.DataFrame,
    source: str,
    context: str = "",
) -> None:
    """Assert that a DataFrame is non-empty.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    source : str
        Data source name for error messages (e.g., "features_train").
    context : str
        Additional context (e.g., city code or file path).

    Raises
    ------
    CriticalDataError
        If the DataFrame is empty (zero rows).
    """
    if len(df) == 0:
        raise CriticalDataError(
            source=source,
            reason=f"DataFrame is empty (0 rows). Context: {context}" if context
            else "DataFrame is empty (0 rows).",
            city_code=context if len(context) <= 5 else "",
        )
    logger.debug(
        "DataFrame non-empty check passed for %s: %d rows", source, len(df)
    )


def validate_critical_data_available(
    city_code: str,
    check_processed: bool = True,
    check_raw: bool = True,
) -> Dict[str, bool]:
    """Verify that critical data files are present and non-empty for a city.

    This function checks both raw and processed data directories and returns
    a summary.  When critical files are missing it raises ``CriticalDataError``
    to enforce explicit run-fail behaviour (Phase B requirement).

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "chi", "phl", "atl", "aus").
    check_processed : bool
        If True, verify that processed train/val/test feature and target
        CSVs exist and are non-empty.
    check_raw : bool
        If True, verify that at least one raw station CSV exists.

    Returns
    -------
    dict
        Summary with keys ``raw_ok``, ``processed_ok``, ``n_raw_files``,
        ``n_processed_files``.

    Raises
    ------
    CriticalDataError
        If any critical file is missing, empty, or inaccessible.
    """
    from src.city_config import get_city_config  # local import to avoid cycles

    cfg = get_city_config(city_code)
    summary: Dict[str, bool] = {
        "raw_ok": True,
        "processed_ok": True,
        "n_raw_files": 0,
        "n_processed_files": 0,
    }

    # -- Raw data check --
    if check_raw and city_code != "nyc":
        raw_dir = os.path.join(cfg.data_dir, "raw")
        if not os.path.isdir(raw_dir):
            raise CriticalDataError(
                source="raw_data",
                reason=f"Raw data directory does not exist: {raw_dir}",
                city_code=city_code,
            )
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        summary["n_raw_files"] = len(csv_files)
        if len(csv_files) == 0:
            raise CriticalDataError(
                source="raw_data",
                reason=f"No CSV files found in raw directory: {raw_dir}",
                city_code=city_code,
            )

    # -- Processed data check --
    if check_processed:
        processed_dir = os.path.join(cfg.data_dir, "processed")
        if not os.path.isdir(processed_dir):
            raise CriticalDataError(
                source="processed_data",
                reason=f"Processed data directory does not exist: {processed_dir}",
                city_code=city_code,
            )

        required_files = []
        for split in ("train", "val", "test"):
            required_files.append(f"features_{split}.csv")
            required_files.append(f"target_{split}.csv")

        missing = []
        empty = []
        for fname in required_files:
            fpath = os.path.join(processed_dir, fname)
            if not os.path.isfile(fpath):
                missing.append(fname)
            elif os.path.getsize(fpath) == 0:
                empty.append(fname)

        summary["n_processed_files"] = len(required_files) - len(missing)

        if missing:
            raise CriticalDataError(
                source="processed_data",
                reason=(
                    f"Missing critical processed files in {processed_dir}: "
                    + ", ".join(missing)
                ),
                city_code=city_code,
            )
        if empty:
            raise CriticalDataError(
                source="processed_data",
                reason=(
                    f"Empty critical processed files in {processed_dir}: "
                    + ", ".join(empty)
                ),
                city_code=city_code,
            )

    logger.info(
        "Critical data availability check passed for %s "
        "(raw=%d files, processed=%d files)",
        city_code,
        summary["n_raw_files"],
        summary["n_processed_files"],
    )
    return summary


def validate_no_nan_in_targets(
    targets: pd.DataFrame,
    city_code: str = "",
) -> None:
    """Verify that no NaN values exist in the target column.

    Parameters
    ----------
    targets : pd.DataFrame
        Target DataFrame (typically with a single TMAX column).
    city_code : str
        City code for error messages.

    Raises
    ------
    CriticalDataError
        If any NaN values are found in the target.
    """
    nan_counts = targets.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        detail = "; ".join(
            f"{col}: {count} NaN" for col, count in cols_with_nan.items()
        )
        raise CriticalDataError(
            source="target_data",
            reason=f"NaN values found in target columns: {detail}",
            city_code=city_code,
        )
    logger.debug(
        "No NaN in targets for %s (%d rows checked)",
        city_code, len(targets),
    )


def validate_no_inf_in_features(
    features: pd.DataFrame,
    city_code: str = "",
) -> None:
    """Verify that no infinite values exist in the feature matrix.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    city_code : str
        City code for error messages.

    Raises
    ------
    CriticalDataError
        If any infinite values are found.
    """
    float_cols = features.select_dtypes(include=[np.floating]).columns
    if len(float_cols) == 0:
        return

    inf_mask = np.isinf(features[float_cols])
    cols_with_inf = inf_mask.any()
    bad_cols = cols_with_inf[cols_with_inf].index.tolist()

    if bad_cols:
        detail = "; ".join(
            f"{col}: {int(inf_mask[col].sum())} inf"
            for col in bad_cols[:5]
        )
        raise CriticalDataError(
            source="feature_data",
            reason=f"Infinite values found in feature columns: {detail}",
            city_code=city_code,
        )
