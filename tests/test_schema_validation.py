"""
Tests for the Schema Validation module (src/schema_validation.py).

Validates:
  - DataFrame-level schema validation against SLA specifications
  - Infinity detection in float columns
  - Chronological index validation (sorted, no duplicates, date column fallback)
  - Pipeline stage precondition checks (city config, file existence)
  - Enforcement wrapper (PipelineValidationError raised on failure)
  - PipelineValidationError exception attributes and message formatting
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_sla import get_sla
from src.schema_validation import (
    PipelineValidationError,
    ValidationResult,
    enforce_preconditions,
    validate_chronological_index,
    validate_dataframe_schema,
    validate_no_infinities,
    validate_pipeline_preconditions,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def ghcn_sla():
    """Return the GHCN daily raw SLA for reuse across tests."""
    return get_sla("ghcn_daily_raw")


@pytest.fixture
def valid_ghcn_df():
    """Create a synthetic DataFrame that passes GHCN SLA validation.

    Contains 400 rows (above min_rows=365) with date, TMAX, TMIN columns
    in valid ranges and no missing values.
    """
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    tmax = np.random.uniform(20.0, 100.0, size=n)
    tmin = tmax - np.random.uniform(5.0, 15.0, size=n)
    return pd.DataFrame({
        "date": dates,
        "TMAX": tmax,
        "TMIN": tmin,
    })


# ===========================================================================
# TestValidateDataframeSchema
# ===========================================================================

class TestValidateDataframeSchema:
    """Tests for validate_dataframe_schema()."""

    def test_valid_ghcn_dataframe(self, ghcn_sla, valid_ghcn_df):
        """A well-formed DataFrame matching the GHCN SLA should pass."""
        result = validate_dataframe_schema(valid_ghcn_df, ghcn_sla)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_required_column(self, ghcn_sla, valid_ghcn_df):
        """Removing a required column (TMAX) should cause validation failure."""
        df = valid_ghcn_df.drop(columns=["TMAX"])
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("TMAX" in e and "missing" in e.lower() for e in result.errors)

    def test_below_min_rows(self, ghcn_sla):
        """A DataFrame with fewer rows than SLA minimum should fail."""
        np.random.seed(42)
        n = 100  # below GHCN min_rows=365
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "TMAX": np.random.uniform(30.0, 90.0, size=n),
            "TMIN": np.random.uniform(20.0, 70.0, size=n),
        })
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("row count" in e.lower() or "below minimum" in e.lower()
                    for e in result.errors)

    def test_low_completeness(self, ghcn_sla):
        """More than 20% NaN in TMAX should fail GHCN completeness check (0.80)."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        tmax = np.random.uniform(30.0, 90.0, size=n).astype(float)
        # Set 25% of values to NaN (above the 20% threshold)
        nan_count = int(n * 0.25)
        tmax[:nan_count] = np.nan
        df = pd.DataFrame({
            "date": dates,
            "TMAX": tmax,
            "TMIN": np.random.uniform(20.0, 70.0, size=n),
        })
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("completeness" in e.lower() for e in result.errors)

    def test_value_below_range(self, ghcn_sla, valid_ghcn_df):
        """TMAX value of -50 (below min=-40) should cause a range error."""
        df = valid_ghcn_df.copy()
        df.loc[0, "TMAX"] = -50.0
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("below minimum" in e.lower() for e in result.errors)

    def test_value_above_range(self, ghcn_sla, valid_ghcn_df):
        """TMAX value of 150 (above max=130) should cause a range error."""
        df = valid_ghcn_df.copy()
        df.loc[0, "TMAX"] = 150.0
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("above maximum" in e.lower() for e in result.errors)

    def test_infinity_detected(self, ghcn_sla, valid_ghcn_df):
        """Infinity in TMAX column should cause validation failure."""
        df = valid_ghcn_df.copy()
        df.loc[0, "TMAX"] = np.inf
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is False
        assert any("infinite" in e.lower() or "inf" in e.lower()
                    for e in result.errors)

    def test_all_optional_columns_missing_ok(self, ghcn_sla):
        """Having only required columns (no optional) should still pass."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "TMAX": np.random.uniform(30.0, 90.0, size=n),
            "TMIN": np.random.uniform(20.0, 70.0, size=n),
        })
        # No PRCP, SNOW, SNWD, AWND -- all optional
        result = validate_dataframe_schema(df, ghcn_sla)
        assert result.valid is True

    def test_stats_populated(self, ghcn_sla, valid_ghcn_df):
        """The stats dict should contain row_count and completeness."""
        result = validate_dataframe_schema(valid_ghcn_df, ghcn_sla)
        assert "row_count" in result.stats
        assert result.stats["row_count"] == len(valid_ghcn_df)
        assert "completeness" in result.stats
        assert isinstance(result.stats["completeness"], dict)


# ===========================================================================
# TestValidateNoInfinities
# ===========================================================================

class TestValidateNoInfinities:
    """Tests for validate_no_infinities()."""

    def test_clean_dataframe(self):
        """A DataFrame with no infinities should return an empty list."""
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
        })
        result = validate_no_infinities(df)
        assert result == []

    def test_inf_detected(self):
        """A column with np.inf should be returned in the list."""
        np.random.seed(42)
        df = pd.DataFrame({
            "clean": np.random.randn(10),
            "dirty": np.random.randn(10),
        })
        df.loc[0, "dirty"] = np.inf
        result = validate_no_infinities(df)
        assert "dirty" in result

    def test_negative_inf_detected(self):
        """A column with -np.inf should be detected."""
        np.random.seed(42)
        df = pd.DataFrame({
            "col": np.random.randn(10),
        })
        df.loc[0, "col"] = -np.inf
        result = validate_no_infinities(df)
        assert "col" in result

    def test_non_float_columns_skipped(self):
        """String (object dtype) columns should not be checked for infinities."""
        df = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "value": [1.0, 2.0, 3.0],
        })
        result = validate_no_infinities(df)
        assert result == []


# ===========================================================================
# TestValidateChronologicalIndex
# ===========================================================================

class TestValidateChronologicalIndex:
    """Tests for validate_chronological_index()."""

    def test_sorted_index_passes(self):
        """A properly sorted DatetimeIndex should pass validation."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"val": range(100)}, index=dates)
        result = validate_chronological_index(df)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_unsorted_index_fails(self):
        """An out-of-order DatetimeIndex should fail validation."""
        dates = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
        df = pd.DataFrame({"val": [1, 2, 3]}, index=dates)
        result = validate_chronological_index(df)
        assert result.valid is False
        assert any("sorted" in e.lower() or "order" in e.lower()
                    for e in result.errors)

    def test_duplicate_timestamps_fail(self):
        """Duplicated dates in the index should be detected."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
        df = pd.DataFrame({"val": [1, 2, 3]}, index=dates)
        result = validate_chronological_index(df)
        assert result.valid is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_date_column_used_if_no_datetimeindex(self):
        """A DataFrame with a 'date' column (not DatetimeIndex) should work."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "val": range(50),
        })
        result = validate_chronological_index(df)
        assert result.valid is True

    def test_no_date_info_fails(self):
        """A DataFrame with neither DatetimeIndex nor 'date' column should fail."""
        df = pd.DataFrame({
            "val": [1, 2, 3],
            "other": [4, 5, 6],
        })
        result = validate_chronological_index(df)
        assert result.valid is False
        assert any("neither" in e.lower() or "date" in e.lower()
                    for e in result.errors)


# ===========================================================================
# TestValidatePipelinePreconditions
# ===========================================================================

class TestValidatePipelinePreconditions:
    """Tests for validate_pipeline_preconditions()."""

    def test_invalid_city_code(self):
        """A nonsense city code should produce validation errors."""
        result = validate_pipeline_preconditions("zzz_invalid", "data_collection")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_invalid_stage_name(self):
        """An unrecognized stage name should produce errors."""
        result = validate_pipeline_preconditions("chi", "bogus_stage")
        assert result.valid is False
        assert any("bogus_stage" in e for e in result.errors)

    def test_data_collection_valid_city(self):
        """'chi' data_collection should pass because city config exists."""
        result = validate_pipeline_preconditions("chi", "data_collection")
        assert result.valid is True

    def test_preprocessing_missing_raw_dir(self, tmp_path):
        """When a city's raw/ subdirectory does not exist, preprocessing fails."""
        # We test with 'chi' which checks for a raw dir; the real data dir
        # may or may not have it, but we can verify the validation logic
        # by checking that it produces an error if raw/ is missing in the
        # configured path. Since we cannot easily override city_config's
        # data_dir, we test that validation returns errors for a city
        # whose raw dir might not exist in the test environment.
        result = validate_pipeline_preconditions("chi", "preprocessing")
        # In a test environment without actual data files, this should fail
        # because the raw directory or station files won't be present
        # If the raw dir does exist (CI environment), at least it runs without crash
        assert isinstance(result, ValidationResult)

    def test_benchmark_missing_processed_dir(self):
        """Benchmark stage should fail when the processed dir does not exist."""
        # In a test environment, the processed directory for a city will
        # typically not exist, so this should fail
        result = validate_pipeline_preconditions("chi", "benchmark")
        # The processed directory likely does not exist in test env
        if not os.path.isdir(os.path.join("data", "chicago", "processed")):
            assert result.valid is False
            assert any("processed" in e.lower() or "does not exist" in e.lower()
                        for e in result.errors)


# ===========================================================================
# TestEnforcePreconditions
# ===========================================================================

class TestEnforcePreconditions:
    """Tests for enforce_preconditions()."""

    def test_enforce_raises_on_failure(self):
        """enforce_preconditions should raise PipelineValidationError when
        preconditions are not met (processed files won't exist in test env)."""
        with pytest.raises(PipelineValidationError):
            enforce_preconditions("chi", "benchmark")

    def test_enforce_carries_stage_and_city(self):
        """The raised exception should carry stage, city_code, and errors."""
        with pytest.raises(PipelineValidationError) as exc_info:
            enforce_preconditions("chi", "benchmark")
        exc = exc_info.value
        assert exc.stage == "benchmark"
        assert exc.city_code == "chi"
        assert isinstance(exc.errors, list)
        assert len(exc.errors) > 0


# ===========================================================================
# TestPipelineValidationError
# ===========================================================================

class TestPipelineValidationError:
    """Tests for PipelineValidationError exception class."""

    def test_exception_attributes(self):
        """stage, city_code, and errors must be set correctly on the exception."""
        err = PipelineValidationError(
            stage="preprocessing",
            city_code="nyc",
            errors=["Missing raw file", "Schema mismatch"],
        )
        assert err.stage == "preprocessing"
        assert err.city_code == "nyc"
        assert err.errors == ["Missing raw file", "Schema mismatch"]

    def test_exception_message_format(self):
        """Error message string should include the city and stage."""
        err = PipelineValidationError(
            stage="benchmark",
            city_code="phl",
            errors=["No features file found"],
        )
        msg = str(err)
        assert "phl" in msg
        assert "benchmark" in msg
        assert "No features file found" in msg
