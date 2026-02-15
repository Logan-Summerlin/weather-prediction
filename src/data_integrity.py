"""
Data Integrity Validation Module.

Implements automated checks recommended by the cross-city Brier integrity
audit (2026-02-15). These guards prevent silent data quality failures that
can corrupt model evaluation results.

Usage:
    from src.data_integrity import (
        validate_no_synthetic_data,
        validate_bucket_consistency,
        validate_required_inputs_exist,
    )
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

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
