"""
Schema Validation Module for the Weather Prediction Pipeline.

Validates DataFrames against SLA (Service Level Agreement) specifications and
provides pipeline-stage-specific precondition checks. This module is part of
Phase B (Reliability Hardening) and enforces data quality contracts at every
stage of the pipeline.

Key capabilities:
    - DataFrame-level schema validation against SLA definitions
    - Pipeline stage precondition verification (file existence + schema checks)
    - Infinity and chronological index guards
    - Enforceable preconditions that hard-fail pipelines on violations

Usage:
    from src.schema_validation import (
        validate_dataframe_schema,
        validate_pipeline_preconditions,
        enforce_preconditions,
        validate_no_infinities,
        validate_chronological_index,
        ValidationResult,
        PipelineValidationError,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_sla import (
    DataSourceSLA,
    ColumnSpec,
    get_sla,
    SLA_MANIFEST_VERSION,
    CutoffFeatureSpec,
    get_cutoff_spec,
    get_cutoff_manifest_version,
    get_critical_cutoff_features,
    cutoff_instant_utc,
    latest_usable_timestamp,
)
from src.city_config import get_city_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid pipeline stages (ordered by dependency)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "data_collection",
    "preprocessing",
    "benchmark",
    "synthesis_calibration",
    "backtest",
    "promotion_evaluation",
]


# ---------------------------------------------------------------------------
# Result and exception types
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Result of a schema validation check.

    Attributes:
        valid: True if the validation passed with no critical errors.
        source_name: Name of the data source or SLA that was validated.
        sla_version: Version of the SLA used for validation.
        errors: List of critical issues that cause validation failure.
        warnings: List of non-critical issues (logged but do not fail).
        stats: Summary statistics collected during validation (row count,
               per-column completeness fractions, etc.).
    """

    valid: bool
    source_name: str
    sla_version: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class KillSwitchError(RuntimeError):
    """Raised when a critical inference input violates the 7am-ET cutoff.

    Per the project ground rules, a critical operational feature that cannot be
    proven available (and not stale) by the 7:00 AM Eastern cutoff is a
    kill-switch event: live inference / trading for that city-day must halt
    rather than run on leaked or stale data.

    Attributes:
        city_code: City the inference run was for (``""`` if not city-scoped).
        market_date: ISO date string of the affected market day.
        violations: Human-readable descriptions of each cutoff violation.
    """

    def __init__(self, city_code: str, market_date: str, violations: List[str]):
        self.city_code = city_code
        self.market_date = market_date
        self.violations = violations
        scope = f"{city_code}/" if city_code else ""
        msg = (
            f"KILL SWITCH — cutoff freshness violation for {scope}{market_date}:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
        super().__init__(msg)


class PipelineValidationError(RuntimeError):
    """Raised when pipeline preconditions are not met.

    Carries structured information about which stage and city failed, and
    the list of specific validation errors encountered.

    Attributes:
        stage: Pipeline stage name that failed validation.
        city_code: City code for which the validation was run.
        errors: List of human-readable error descriptions.
    """

    def __init__(self, stage: str, city_code: str, errors: List[str]):
        self.stage = stage
        self.city_code = city_code
        self.errors = errors
        msg = (
            f"Pipeline precondition check failed for {city_code}/{stage}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Core DataFrame validation
# ---------------------------------------------------------------------------
def validate_dataframe_schema(
    df: pd.DataFrame,
    sla: DataSourceSLA,
    context: str = "",
) -> ValidationResult:
    """Validate a DataFrame against an SLA specification.

    Checks performed:
        1. All required columns are present.
        2. No unexpected infinities in float columns.
        3. Value ranges for columns with min/max defined.
        4. Minimum completeness (non-null fraction) per required column.
        5. Minimum row count.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    sla : DataSourceSLA
        The SLA definition to validate against.
    context : str, optional
        Human-readable context string for log messages (e.g., "Chicago raw
        station USW00094846").

    Returns
    -------
    ValidationResult
        Object containing validation outcome, errors, warnings, and stats.
    """
    ctx = f" ({context})" if context else ""
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "sla_name": sla.name,
    }

    # -- 1. Minimum row count --
    if len(df) < sla.min_rows:
        errors.append(
            f"Row count {len(df)} is below minimum {sla.min_rows}{ctx}"
        )

    # -- 2. Required columns present --
    df_columns = set(df.columns)
    completeness_stats: Dict[str, float] = {}

    for col_spec in sla.columns:
        if col_spec.required and col_spec.name not in df_columns:
            errors.append(
                f"Required column '{col_spec.name}' is missing{ctx}"
            )
            continue

        if col_spec.name not in df_columns:
            # Optional column not present -- skip remaining checks for it
            continue

        col_series = df[col_spec.name]

        # -- 3. Completeness check for required columns --
        if col_spec.required:
            non_null_count = col_series.notna().sum()
            total_count = len(col_series)
            completeness = non_null_count / total_count if total_count > 0 else 0.0
            completeness_stats[col_spec.name] = round(completeness, 4)

            if completeness < sla.min_completeness:
                errors.append(
                    f"Column '{col_spec.name}' completeness {completeness:.2%} "
                    f"is below minimum {sla.min_completeness:.2%}{ctx}"
                )

        # -- 4. Infinity check for float columns --
        if col_spec.dtype == "float" and col_spec.name in df_columns:
            try:
                numeric_col = pd.to_numeric(col_series, errors="coerce")
                inf_count = np.isinf(numeric_col).sum()
                if inf_count > 0:
                    errors.append(
                        f"Column '{col_spec.name}' contains {inf_count} "
                        f"infinite value(s){ctx}"
                    )
            except (TypeError, ValueError):
                warnings.append(
                    f"Column '{col_spec.name}' could not be checked for "
                    f"infinities (non-numeric data){ctx}"
                )

        # -- 5. Value range checks --
        if col_spec.dtype in ("float", "int") and col_spec.name in df_columns:
            try:
                numeric_col = pd.to_numeric(col_series, errors="coerce").dropna()
                if len(numeric_col) > 0:
                    col_min = float(numeric_col.min())
                    col_max = float(numeric_col.max())
                    stats[f"{col_spec.name}_min"] = col_min
                    stats[f"{col_spec.name}_max"] = col_max

                    if col_spec.min_value is not None and col_min < col_spec.min_value:
                        n_violations = int((numeric_col < col_spec.min_value).sum())
                        errors.append(
                            f"Column '{col_spec.name}' has {n_violations} value(s) "
                            f"below minimum {col_spec.min_value} "
                            f"(actual min: {col_min:.2f}){ctx}"
                        )

                    if col_spec.max_value is not None and col_max > col_spec.max_value:
                        n_violations = int((numeric_col > col_spec.max_value).sum())
                        errors.append(
                            f"Column '{col_spec.name}' has {n_violations} value(s) "
                            f"above maximum {col_spec.max_value} "
                            f"(actual max: {col_max:.2f}){ctx}"
                        )
            except (TypeError, ValueError):
                warnings.append(
                    f"Column '{col_spec.name}' could not be range-checked "
                    f"(non-numeric data){ctx}"
                )

    stats["completeness"] = completeness_stats

    # -- Also check all float-like columns in the DataFrame for infinities
    #    beyond those explicitly listed in the SLA --
    sla_col_names = {cs.name for cs in sla.columns}
    for col_name in df.columns:
        if col_name in sla_col_names:
            continue  # already checked above
        if df[col_name].dtype.kind == "f":
            inf_count = int(np.isinf(df[col_name]).sum())
            if inf_count > 0:
                warnings.append(
                    f"Non-SLA float column '{col_name}' contains {inf_count} "
                    f"infinite value(s){ctx}"
                )

    valid = len(errors) == 0

    # Log results
    if valid:
        logger.info(
            "Schema validation PASSED for %s%s (%d rows)",
            sla.name, ctx, len(df),
        )
        for w in warnings:
            logger.warning("Validation warning: %s", w)
    else:
        logger.error(
            "Schema validation FAILED for %s%s: %d error(s)",
            sla.name, ctx, len(errors),
        )
        for e in errors:
            logger.error("  %s", e)
        for w in warnings:
            logger.warning("  %s", w)

    return ValidationResult(
        valid=valid,
        source_name=sla.name,
        sla_version=sla.version,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Quick infinity check
# ---------------------------------------------------------------------------
def validate_no_infinities(
    df: pd.DataFrame,
    context: str = "",
) -> List[str]:
    """Check that no float columns contain inf/-inf values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    context : str, optional
        Human-readable context string for log messages.

    Returns
    -------
    list of str
        Column names that contain at least one infinite value. An empty list
        means no infinities were found.
    """
    ctx = f" ({context})" if context else ""
    cols_with_inf: List[str] = []

    for col_name in df.columns:
        if df[col_name].dtype.kind == "f":
            inf_count = int(np.isinf(df[col_name]).sum())
            if inf_count > 0:
                cols_with_inf.append(col_name)
                logger.warning(
                    "Column '%s' contains %d infinite value(s)%s",
                    col_name, inf_count, ctx,
                )

    if not cols_with_inf:
        logger.debug("No infinities found in %d float columns%s", len(df.columns), ctx)
    else:
        logger.warning(
            "Found infinities in %d column(s)%s: %s",
            len(cols_with_inf), ctx, ", ".join(cols_with_inf),
        )

    return cols_with_inf


# ---------------------------------------------------------------------------
# Chronological index validation
# ---------------------------------------------------------------------------
def validate_chronological_index(
    df: pd.DataFrame,
    context: str = "",
) -> ValidationResult:
    """Verify a DataFrame's DatetimeIndex is strictly chronologically sorted
    with no duplicate timestamps.

    If the DataFrame does not have a DatetimeIndex, a 'date' column is sought
    and used instead.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate. Should have a DatetimeIndex or a 'date'
        column that can be parsed as datetime.
    context : str, optional
        Human-readable context string for log messages.

    Returns
    -------
    ValidationResult
        Validation result with errors if the index is unsorted or has
        duplicates.
    """
    ctx = f" ({context})" if context else ""
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {"row_count": len(df)}

    # Resolve the datetime series to check
    if isinstance(df.index, pd.DatetimeIndex):
        dt_index = df.index
    elif "date" in df.columns:
        try:
            dt_index = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        except Exception as exc:
            errors.append(
                f"Could not parse 'date' column as datetime{ctx}: {exc}"
            )
            return ValidationResult(
                valid=False,
                source_name="chronological_index",
                sla_version=SLA_MANIFEST_VERSION,
                errors=errors,
                warnings=warnings,
                stats=stats,
            )
    else:
        errors.append(
            f"DataFrame has neither a DatetimeIndex nor a 'date' column{ctx}"
        )
        return ValidationResult(
            valid=False,
            source_name="chronological_index",
            sla_version=SLA_MANIFEST_VERSION,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # Check for duplicates
    n_duplicates = int(dt_index.duplicated().sum())
    if n_duplicates > 0:
        errors.append(
            f"DatetimeIndex contains {n_duplicates} duplicate timestamp(s){ctx}"
        )
    stats["n_duplicates"] = n_duplicates

    # Check strict chronological ordering
    if len(dt_index) > 1:
        is_sorted = (dt_index[1:] > dt_index[:-1]).all()
        if not is_sorted:
            # Count how many positions are out of order
            out_of_order = int((dt_index[1:] <= dt_index[:-1]).sum())
            errors.append(
                f"DatetimeIndex is not strictly chronologically sorted: "
                f"{out_of_order} position(s) out of order{ctx}"
            )
        stats["is_sorted"] = bool(is_sorted)
    else:
        stats["is_sorted"] = True

    if len(dt_index) > 0:
        stats["min_date"] = str(dt_index.min())
        stats["max_date"] = str(dt_index.max())

    valid = len(errors) == 0

    if valid:
        logger.info(
            "Chronological index validation PASSED%s (%d rows, %s to %s)",
            ctx, len(df),
            stats.get("min_date", "N/A"),
            stats.get("max_date", "N/A"),
        )
    else:
        logger.error(
            "Chronological index validation FAILED%s: %d error(s)",
            ctx, len(errors),
        )
        for e in errors:
            logger.error("  %s", e)

    return ValidationResult(
        valid=valid,
        source_name="chronological_index",
        sla_version=SLA_MANIFEST_VERSION,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Pipeline stage precondition validation
# ---------------------------------------------------------------------------
def validate_pipeline_preconditions(
    city_code: str,
    stage: str,
) -> ValidationResult:
    """Validate preconditions for a specific pipeline stage.

    Given a city and a pipeline stage name, checks that all required input
    files exist and (where feasible) pass schema validation.

    Supported stages:
        - data_collection: City config must be valid.
        - preprocessing: Raw data CSVs must exist with at least 1 station file.
        - benchmark: Processed feature and target files must exist.
        - synthesis_calibration: Benchmark results must exist.
        - backtest: Synthesis/calibration results must exist.
        - promotion_evaluation: Backtest results must exist.

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "nyc", "chi", "phl", "atl", "aus").
    stage : str
        Pipeline stage name (one of PIPELINE_STAGES).

    Returns
    -------
    ValidationResult
        Validation result with errors for any missing or invalid inputs.
    """
    if stage not in PIPELINE_STAGES:
        return ValidationResult(
            valid=False,
            source_name=f"precondition/{stage}",
            sla_version=SLA_MANIFEST_VERSION,
            errors=[
                f"Unknown pipeline stage '{stage}'. "
                f"Valid stages: {', '.join(PIPELINE_STAGES)}"
            ],
        )

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {"city_code": city_code, "stage": stage}

    # Validate city config first (common to all stages)
    try:
        cfg = get_city_config(city_code)
        stats["city_name"] = cfg.city_name
    except (ValueError, KeyError) as exc:
        errors.append(f"Invalid city config for '{city_code}': {exc}")
        return ValidationResult(
            valid=False,
            source_name=f"precondition/{stage}",
            sla_version=SLA_MANIFEST_VERSION,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    # Dispatch to stage-specific checks
    if stage == "data_collection":
        _check_data_collection(cfg, errors, warnings, stats)
    elif stage == "preprocessing":
        _check_preprocessing(cfg, city_code, errors, warnings, stats)
    elif stage == "benchmark":
        _check_benchmark(cfg, errors, warnings, stats)
    elif stage == "synthesis_calibration":
        _check_synthesis_calibration(cfg, errors, warnings, stats)
    elif stage == "backtest":
        _check_backtest(cfg, errors, warnings, stats)
    elif stage == "promotion_evaluation":
        _check_promotion_evaluation(cfg, errors, warnings, stats)

    valid = len(errors) == 0

    if valid:
        logger.info(
            "Precondition check PASSED for %s/%s",
            city_code, stage,
        )
    else:
        logger.error(
            "Precondition check FAILED for %s/%s: %d error(s)",
            city_code, stage, len(errors),
        )
        for e in errors:
            logger.error("  %s", e)

    for w in warnings:
        logger.warning("  %s", w)

    return ValidationResult(
        valid=valid,
        source_name=f"precondition/{stage}",
        sla_version=SLA_MANIFEST_VERSION,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Stage-specific precondition helpers
# ---------------------------------------------------------------------------
def _check_data_collection(
    cfg,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """data_collection: City config must be valid (already checked above)."""
    # The city config was already validated by the caller; if we got here,
    # the config is valid. We can do additional sanity checks.
    if not cfg.target_station:
        errors.append("City config has no target_station defined")
    if not cfg.bucket_edges:
        errors.append("City config has no bucket_edges defined")
    stats["target_station"] = cfg.target_station
    stats["n_buckets"] = len(cfg.bucket_edges)


def _check_preprocessing(
    cfg,
    city_code: str,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """preprocessing: Raw data CSVs must exist with at least 1 station file."""
    # NYC has a different layout -- skip raw dir check for it
    if city_code == "nyc":
        stats["skipped_raw_check"] = True
        warnings.append(
            "NYC uses a legacy data layout; skipping raw directory check"
        )
        return

    raw_dir = os.path.join(cfg.data_dir, "raw")
    if not os.path.isdir(raw_dir):
        errors.append(
            f"Raw data directory does not exist: {raw_dir}"
        )
        return

    # Find station CSV files in raw directory
    station_files = [
        f for f in os.listdir(raw_dir)
        if f.endswith(".csv") and f.startswith("USW")
    ]
    stats["raw_station_file_count"] = len(station_files)

    if len(station_files) == 0:
        errors.append(
            f"No station CSV files found in raw directory: {raw_dir}"
        )
        return

    # Validate each station file against the GHCN SLA
    ghcn_sla = get_sla("ghcn_daily_raw")
    files_checked = 0
    files_failed = 0

    for station_file in station_files:
        file_path = os.path.join(raw_dir, station_file)
        try:
            df = pd.read_csv(file_path)
            result = validate_dataframe_schema(
                df, ghcn_sla,
                context=f"raw station {station_file}",
            )
            files_checked += 1
            if not result.valid:
                files_failed += 1
                for e in result.errors:
                    errors.append(f"[{station_file}] {e}")
        except Exception as exc:
            files_failed += 1
            errors.append(
                f"Failed to read/validate {station_file}: {exc}"
            )

    stats["raw_files_checked"] = files_checked
    stats["raw_files_failed"] = files_failed


def _scaled_features_sla(features_sla: DataSourceSLA) -> DataSourceSLA:
    """Return a copy of the features SLA with value bounds removed.

    Processed feature files on disk are standardized (z-scored) by
    data_preprocessing.py, including the cyclical sin_day/cos_day columns,
    so the raw [-1, 1] bounds do not apply. Presence, dtype, NaN/inf and
    completeness checks still do.
    """
    relaxed_columns = [
        replace(col, min_value=None, max_value=None)
        for col in features_sla.columns
    ]
    return replace(features_sla, columns=relaxed_columns)


def _resolve_target_column(df: pd.DataFrame, context: str, errors: List[str]) -> pd.DataFrame:
    """Normalize the city-prefixed target column to the SLA name 'TMAX'.

    Target files contain exactly one column matching ``TMAX`` or
    ``*_TMAX`` (e.g. CHI_TMAX); the SLA spec validates it as 'TMAX'.
    """
    candidates = [
        c for c in df.columns if c == "TMAX" or c.endswith("_TMAX")
    ]
    if len(candidates) == 1:
        return df.rename(columns={candidates[0]: "TMAX"})
    errors.append(
        f"[{context}] Expected exactly one TMAX/*_TMAX target column, "
        f"found {candidates or list(df.columns)}"
    )
    return df


def _check_benchmark(
    cfg,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """benchmark: Processed feature and target files must exist."""
    processed_dir = os.path.join(cfg.data_dir, "processed")

    if not os.path.isdir(processed_dir):
        errors.append(
            f"Processed data directory does not exist: {processed_dir}"
        )
        return

    required_splits = ["train", "val", "test"]
    features_sla = _scaled_features_sla(get_sla("processed_features"))
    target_sla = get_sla("processed_targets")

    for split in required_splits:
        # Check features file
        features_path = os.path.join(processed_dir, f"features_{split}.csv")
        if not os.path.isfile(features_path):
            errors.append(f"Missing processed features file: {features_path}")
        else:
            try:
                df = pd.read_csv(features_path)
                result = validate_dataframe_schema(
                    df, features_sla,
                    context=f"features_{split}",
                )
                if not result.valid:
                    for e in result.errors:
                        errors.append(f"[features_{split}] {e}")
                stats[f"features_{split}_rows"] = len(df)
            except Exception as exc:
                errors.append(
                    f"Failed to read/validate features_{split}.csv: {exc}"
                )

        # Check target file
        target_path = os.path.join(processed_dir, f"target_{split}.csv")
        if not os.path.isfile(target_path):
            errors.append(f"Missing processed target file: {target_path}")
        else:
            try:
                df = pd.read_csv(target_path)
                df = _resolve_target_column(df, f"target_{split}", errors)
                result = validate_dataframe_schema(
                    df, target_sla,
                    context=f"target_{split}",
                )
                if not result.valid:
                    for e in result.errors:
                        errors.append(f"[target_{split}] {e}")
                stats[f"target_{split}_rows"] = len(df)
            except Exception as exc:
                errors.append(
                    f"Failed to read/validate target_{split}.csv: {exc}"
                )


def _check_synthesis_calibration(
    cfg,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """synthesis_calibration: Benchmark results must exist."""
    results_dir = cfg.results_dir

    if not os.path.isdir(results_dir):
        errors.append(
            f"Results directory does not exist: {results_dir}"
        )
        return

    # Look for benchmark result files (JSON or CSV)
    result_files = [
        f for f in os.listdir(results_dir)
        if ("benchmark" in f.lower() or "base_predictions" in f.lower())
        and (f.endswith(".json") or f.endswith(".csv"))
    ]
    stats["benchmark_result_files"] = len(result_files)

    if len(result_files) == 0:
        errors.append(
            f"No benchmark result files found in {results_dir}. "
            f"Run the benchmark stage first."
        )


def _check_backtest(
    cfg,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """backtest: Synthesis/calibration results must exist."""
    results_dir = cfg.results_dir

    if not os.path.isdir(results_dir):
        errors.append(
            f"Results directory does not exist: {results_dir}"
        )
        return

    # Look for synthesis/calibration outputs -- could be in results dir
    # or a calibration subdirectory
    synthesis_markers = [
        f for f in os.listdir(results_dir)
        if ("synth" in f.lower() or "calibrat" in f.lower() or "brier" in f.lower())
        and (f.endswith(".json") or f.endswith(".csv") or f.endswith(".png"))
    ]
    stats["synthesis_calibration_files"] = len(synthesis_markers)

    if len(synthesis_markers) == 0:
        errors.append(
            f"No synthesis/calibration result files found in {results_dir}. "
            f"Run the synthesis_calibration stage first."
        )


def _check_promotion_evaluation(
    cfg,
    errors: List[str],
    warnings: List[str],
    stats: Dict[str, Any],
) -> None:
    """promotion_evaluation: Backtest results must exist."""
    results_dir = cfg.results_dir

    if not os.path.isdir(results_dir):
        errors.append(
            f"Results directory does not exist: {results_dir}"
        )
        return

    # Look for backtest outputs
    backtest_dir = os.path.join(results_dir, "backtest")
    backtest_files_in_subdir = []
    backtest_files_in_results = []

    if os.path.isdir(backtest_dir):
        backtest_files_in_subdir = [
            f for f in os.listdir(backtest_dir)
            if f.endswith(".json") or f.endswith(".csv")
        ]

    backtest_files_in_results = [
        f for f in os.listdir(results_dir)
        if "backtest" in f.lower()
        and (f.endswith(".json") or f.endswith(".csv"))
    ]

    total_backtest_files = len(backtest_files_in_subdir) + len(backtest_files_in_results)
    stats["backtest_files"] = total_backtest_files

    if total_backtest_files == 0:
        errors.append(
            f"No backtest result files found in {results_dir} or "
            f"{backtest_dir}. Run the backtest stage first."
        )


# ---------------------------------------------------------------------------
# Enforcement wrapper
# ---------------------------------------------------------------------------
def enforce_preconditions(city_code: str, stage: str) -> None:
    """Validate pipeline preconditions and raise on failure.

    This is the function that pipeline scripts should call to hard-fail if
    critical preconditions are not met. It wraps validate_pipeline_preconditions
    and raises PipelineValidationError on any critical errors.

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "nyc", "chi").
    stage : str
        Pipeline stage name (one of PIPELINE_STAGES).

    Raises
    ------
    PipelineValidationError
        If the precondition check produces any errors.
    """
    result = validate_pipeline_preconditions(city_code, stage)

    if not result.valid:
        raise PipelineValidationError(
            stage=stage,
            city_code=city_code,
            errors=result.errors,
        )

    logger.info(
        "Preconditions enforced successfully for %s/%s",
        city_code, stage,
    )


# ---------------------------------------------------------------------------
# 7am-ET cutoff freshness validation
# ---------------------------------------------------------------------------
def _to_utc_datetime(value):
    """Coerce a timestamp-like value to a timezone-aware UTC datetime.

    Accepts ``datetime`` (naive assumed UTC), ISO strings, ``date`` objects,
    and pandas ``Timestamp``.  Returns ``None`` for missing values (``None``,
    NaN/NaT) so callers can treat them as "feature absent".
    """
    from datetime import datetime, date, timezone

    if value is None:
        return None
    # pandas NaT / float NaN
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    elif isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    raise TypeError(f"Unsupported timestamp type for freshness check: {type(value)!r}")


def validate_cutoff_freshness(
    feature: str,
    latest_available,
    market_date,
    context: str = "",
) -> ValidationResult:
    """Validate one inference feature against the 7am-ET cutoff manifest.

    Two failure modes are enforced for every feature, using its
    :class:`CutoffFeatureSpec`:

    1. **Leakage** — the freshest record's valid/issue time is *after* the
       feature's ``latest_usable_timestamp`` for the market day.  Using it
       would leak post-cutoff information; this is always an error.
    2. **Staleness** — the freshest record is older than
       ``max_staleness_hours`` measured back from the cutoff, i.e. the source
       has gone missing.  This is an error for the feature regardless of
       criticality; the *aggregate* validator decides whether it trips the
       kill switch (critical) or merely degrades (recommended).

    A missing timestamp (``None``/NaT) is reported as an error ("feature
    absent at cutoff").

    Parameters
    ----------
    feature : str
        Registered cutoff feature identifier (see
        :func:`src.data_sla.list_cutoff_features`).
    latest_available : datetime-like or None
        Valid/issue time of the freshest record the operational path has for
        this feature.  Naive datetimes are assumed UTC.
    market_date : date-like
        The market day the inference is for.
    context : str, optional
        Extra context for log messages.

    Returns
    -------
    ValidationResult
        ``source_name`` is ``"cutoff/<feature>"``; ``stats`` carries the
        resolved timestamps and the feature criticality.
    """
    ctx = f" ({context})" if context else ""
    spec: CutoffFeatureSpec = get_cutoff_spec(feature)
    cutoff = cutoff_instant_utc(market_date)
    newest_usable = latest_usable_timestamp(feature, market_date)
    oldest_usable = cutoff - _timedelta_hours(spec.max_staleness_hours)

    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "feature": feature,
        "criticality": spec.criticality,
        "cutoff_utc": cutoff.isoformat(),
        "latest_usable_utc": newest_usable.isoformat(),
        "max_staleness_hours": spec.max_staleness_hours,
    }

    ts = _to_utc_datetime(latest_available)
    if ts is None:
        errors.append(
            f"Feature '{feature}' has no available record at the 7am ET "
            f"cutoff for {market_date}{ctx}. Fallback: {spec.fallback_behavior}"
        )
        return ValidationResult(
            valid=False,
            source_name=f"cutoff/{feature}",
            sla_version=get_cutoff_manifest_version(),
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    stats["latest_available_utc"] = ts.isoformat()

    # 1. Leakage: record is newer than the cutoff-safe horizon.
    if ts > newest_usable:
        errors.append(
            f"Feature '{feature}' freshest record {ts.isoformat()} is after "
            f"the latest cutoff-safe time {newest_usable.isoformat()} "
            f"(post-cutoff leakage){ctx}."
        )

    # 2. Staleness: source has gone missing relative to the cutoff.
    if ts < oldest_usable:
        age_h = (cutoff - ts).total_seconds() / 3600.0
        errors.append(
            f"Feature '{feature}' is stale: freshest record {ts.isoformat()} "
            f"is {age_h:.1f}h before the cutoff, exceeding the "
            f"{spec.max_staleness_hours:.1f}h limit{ctx}. "
            f"Fallback: {spec.fallback_behavior}"
        )

    valid = len(errors) == 0
    if valid:
        logger.info(
            "Cutoff freshness PASSED for %s%s (record %s <= %s)",
            feature, ctx, ts.isoformat(), newest_usable.isoformat(),
        )
    else:
        log = logger.error if spec.criticality == "critical" else logger.warning
        log("Cutoff freshness FAILED for %s%s: %d issue(s)", feature, ctx, len(errors))
        for e in errors:
            log("  %s", e)

    return ValidationResult(
        valid=valid,
        source_name=f"cutoff/{feature}",
        sla_version=get_cutoff_manifest_version(),
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def _timedelta_hours(hours: float):
    from datetime import timedelta
    return timedelta(hours=hours)


def validate_inference_freshness(
    city_code: str,
    market_date,
    available_timestamps: Dict[str, Any],
    require_features: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate all inference features for a city-day against the cutoff manifest.

    Aggregates :func:`validate_cutoff_freshness` over every registered cutoff
    feature.  Criticality decides escalation:

    * A **critical** feature that fails (absent, leaking, or stale) produces a
      hard error -> the aggregate result is ``valid=False`` and
      :func:`enforce_inference_freshness` will trip the kill switch.
    * A **recommended** feature that fails produces a warning -> the run may
      proceed in a degraded mode.

    Parameters
    ----------
    city_code : str
        City the inference is for (used only for messaging / stats).
    market_date : date-like
        The market day.
    available_timestamps : dict
        Mapping of cutoff-feature identifier -> freshest available record
        timestamp (datetime-like or ``None``).  Features registered in the
        manifest but absent from this mapping are treated as missing.
    require_features : list of str, optional
        Restrict validation to this subset of features (e.g. only those a
        given model actually consumes).  Defaults to every registered feature.

    Returns
    -------
    ValidationResult
        ``valid`` is False iff at least one *critical* feature failed.
        ``stats['feature_results']`` holds the per-feature stats and
        ``stats['kill_switch']`` mirrors ``not valid``.
    """
    from src.data_sla import list_cutoff_features

    features = require_features or list_cutoff_features()
    critical = set(get_critical_cutoff_features())

    errors: List[str] = []
    warnings: List[str] = []
    feature_results: Dict[str, Any] = {}

    for feature in features:
        result = validate_cutoff_freshness(
            feature,
            available_timestamps.get(feature),
            market_date,
            context=f"{city_code} {market_date}",
        )
        feature_results[feature] = result.stats
        if not result.valid:
            if feature in critical:
                errors.extend(result.errors)
            else:
                warnings.extend(result.errors)

    valid = len(errors) == 0
    stats: Dict[str, Any] = {
        "city_code": city_code,
        "market_date": str(market_date),
        "kill_switch": not valid,
        "n_features_checked": len(features),
        "feature_results": feature_results,
    }

    if valid and not warnings:
        logger.info(
            "Inference freshness PASSED for %s/%s (%d features)",
            city_code, market_date, len(features),
        )
    elif valid:
        logger.warning(
            "Inference freshness DEGRADED for %s/%s: %d recommended-feature "
            "warning(s); proceeding without kill switch",
            city_code, market_date, len(warnings),
        )
    else:
        logger.error(
            "Inference freshness FAILED for %s/%s: %d critical violation(s) "
            "— KILL SWITCH",
            city_code, market_date, len(errors),
        )

    return ValidationResult(
        valid=valid,
        source_name=f"cutoff_freshness/{city_code}",
        sla_version=get_cutoff_manifest_version(),
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def enforce_inference_freshness(
    city_code: str,
    market_date,
    available_timestamps: Dict[str, Any],
    require_features: Optional[List[str]] = None,
) -> ValidationResult:
    """Validate cutoff freshness and trip the kill switch on critical failure.

    Operational inference scripts should call this immediately before running
    a model for a city-day.  Returns the (valid) :class:`ValidationResult` on
    success so callers can inspect warnings / degraded features.

    Raises
    ------
    KillSwitchError
        If any critical inference feature is absent, leaking, or stale at the
        7am ET cutoff.
    """
    result = validate_inference_freshness(
        city_code, market_date, available_timestamps, require_features,
    )
    if not result.valid:
        raise KillSwitchError(
            city_code=city_code,
            market_date=str(market_date),
            violations=result.errors,
        )
    return result
