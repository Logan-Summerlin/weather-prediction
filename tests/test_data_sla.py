"""
Tests for the Data SLA (Service Level Agreement) Manifest module (src/data_sla.py).

Validates:
  - SLA manifest version follows semver
  - Registry completeness and ordering
  - GHCN-Daily SLA fields and column specifications
  - ASOS hourly and daily SLA fields
  - Processed features and targets SLA fields
  - Recommended (NWP, sounding) source criticality
  - Convenience helpers (get_required_columns, get_column_spec, is_critical)
  - Frozen dataclass immutability for ColumnSpec and DataSourceSLA
"""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_sla import (
    SLA_MANIFEST_VERSION,
    ColumnSpec,
    DataSourceSLA,
    get_column_spec,
    get_required_columns,
    get_sla,
    get_sla_manifest_version,
    is_critical,
    list_sla_sources,
)


# ===========================================================================
# TestSLAManifestVersion
# ===========================================================================

class TestSLAManifestVersion:
    """Tests for the SLA manifest version constant and accessor."""

    def test_manifest_version_is_semver(self):
        """SLA_MANIFEST_VERSION must match the X.Y.Z semver pattern."""
        pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(pattern, SLA_MANIFEST_VERSION), (
            f"SLA_MANIFEST_VERSION '{SLA_MANIFEST_VERSION}' does not match semver X.Y.Z"
        )

    def test_get_sla_manifest_version_matches_constant(self):
        """get_sla_manifest_version() should return the same value as the constant."""
        assert get_sla_manifest_version() == SLA_MANIFEST_VERSION


# ===========================================================================
# TestSLARegistry
# ===========================================================================

class TestSLARegistry:
    """Tests for the SLA registry: list_sla_sources() and get_sla()."""

    def test_list_sla_sources_returns_all_seven(self):
        """The registry must contain exactly 7 data sources."""
        sources = list_sla_sources()
        assert len(sources) == 7

    def test_list_sla_sources_sorted(self):
        """Returned source list must be alphabetically sorted."""
        sources = list_sla_sources()
        assert sources == sorted(sources)

    def test_known_sources_exist(self):
        """Each known source name must be retrievable from the registry."""
        expected = [
            "ghcn_daily_raw",
            "asos_hourly",
            "asos_daily",
            "processed_features",
            "processed_targets",
            "nwp_data",
            "sounding_data",
        ]
        for name in expected:
            sla = get_sla(name)
            assert sla.name == name

    def test_unknown_source_raises_key_error(self):
        """get_sla() must raise KeyError for a source not in the registry."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_sla("nonexistent")


# ===========================================================================
# TestGHCNDailySLA
# ===========================================================================

class TestGHCNDailySLA:
    """Tests for the ghcn_daily_raw SLA definition."""

    @pytest.fixture
    def ghcn(self):
        return get_sla("ghcn_daily_raw")

    def test_ghcn_name_and_criticality(self, ghcn):
        """Name must be 'ghcn_daily_raw' and criticality must be 'critical'."""
        assert ghcn.name == "ghcn_daily_raw"
        assert ghcn.criticality == "critical"

    def test_ghcn_required_columns(self, ghcn):
        """Required columns must include date, TMAX, and TMIN."""
        required = [col.name for col in ghcn.columns if col.required]
        assert "date" in required
        assert "TMAX" in required
        assert "TMIN" in required

    def test_ghcn_tmax_range(self, ghcn):
        """TMAX column spec must have min_value=-40.0 and max_value=130.0."""
        tmax_spec = None
        for col in ghcn.columns:
            if col.name == "TMAX":
                tmax_spec = col
                break
        assert tmax_spec is not None
        assert tmax_spec.min_value == pytest.approx(-40.0)
        assert tmax_spec.max_value == pytest.approx(130.0)

    def test_ghcn_min_completeness(self, ghcn):
        """min_completeness must be 0.80."""
        assert ghcn.min_completeness == pytest.approx(0.80)

    def test_ghcn_min_rows(self, ghcn):
        """min_rows must be 365."""
        assert ghcn.min_rows == 365


# ===========================================================================
# TestASOSSLAs
# ===========================================================================

class TestASOSSLAs:
    """Tests for ASOS hourly and daily SLA definitions."""

    @pytest.fixture
    def asos_hourly(self):
        return get_sla("asos_hourly")

    @pytest.fixture
    def asos_daily(self):
        return get_sla("asos_daily")

    def test_asos_hourly_criticality(self, asos_hourly):
        """ASOS hourly criticality must be 'critical'."""
        assert asos_hourly.criticality == "critical"

    def test_asos_hourly_staleness(self, asos_hourly):
        """ASOS hourly max_staleness_hours must be 6.0."""
        assert asos_hourly.max_staleness_hours == pytest.approx(6.0)

    def test_asos_daily_obs_count_range(self, asos_daily):
        """ASOS daily obs_count column must have min=4 and max=48."""
        obs_spec = None
        for col in asos_daily.columns:
            if col.name == "obs_count":
                obs_spec = col
                break
        assert obs_spec is not None
        assert obs_spec.min_value == pytest.approx(4.0)
        assert obs_spec.max_value == pytest.approx(48.0)


# ===========================================================================
# TestProcessedSLAs
# ===========================================================================

class TestProcessedSLAs:
    """Tests for processed features and targets SLA definitions."""

    @pytest.fixture
    def features(self):
        return get_sla("processed_features")

    @pytest.fixture
    def targets(self):
        return get_sla("processed_targets")

    def test_processed_features_has_sin_cos(self, features):
        """sin_day and cos_day must be required columns in processed_features."""
        required = [col.name for col in features.columns if col.required]
        assert "sin_day" in required
        assert "cos_day" in required

    def test_processed_targets_completeness_is_100_pct(self, targets):
        """Processed targets min_completeness must be 1.0 (no missing targets)."""
        assert targets.min_completeness == pytest.approx(1.0)

    def test_processed_features_no_inf_columns(self, features):
        """no_inf_columns must contain sin_day and cos_day."""
        assert "sin_day" in features.no_inf_columns
        assert "cos_day" in features.no_inf_columns


# ===========================================================================
# TestRecommendedSources
# ===========================================================================

class TestRecommendedSources:
    """Tests for recommended (non-critical) data sources."""

    def test_nwp_is_recommended(self):
        """nwp_data criticality must be 'recommended'."""
        nwp = get_sla("nwp_data")
        assert nwp.criticality == "recommended"

    def test_sounding_is_recommended(self):
        """sounding_data criticality must be 'recommended'."""
        sounding = get_sla("sounding_data")
        assert sounding.criticality == "recommended"


# ===========================================================================
# TestConvenienceHelpers
# ===========================================================================

class TestConvenienceHelpers:
    """Tests for get_required_columns, get_column_spec, is_critical."""

    def test_get_required_columns(self):
        """get_required_columns returns only required column names."""
        required = get_required_columns("ghcn_daily_raw")
        assert isinstance(required, list)
        assert all(isinstance(c, str) for c in required)
        # date, TMAX, TMIN are required; PRCP, SNOW, etc. are optional
        assert "date" in required
        assert "TMAX" in required
        assert "TMIN" in required
        # PRCP is optional so it should not be in the list
        assert "PRCP" not in required

    def test_get_column_spec_found(self):
        """get_column_spec returns a ColumnSpec for an existing column."""
        spec = get_column_spec("ghcn_daily_raw", "TMAX")
        assert spec is not None
        assert isinstance(spec, ColumnSpec)
        assert spec.name == "TMAX"

    def test_get_column_spec_not_found(self):
        """get_column_spec returns None for a non-existing column."""
        spec = get_column_spec("ghcn_daily_raw", "nonexistent_column")
        assert spec is None

    def test_is_critical_true(self):
        """is_critical returns True for a critical source."""
        assert is_critical("ghcn_daily_raw") is True

    def test_is_critical_false(self):
        """is_critical returns False for a recommended source."""
        assert is_critical("nwp_data") is False


# ===========================================================================
# TestColumnSpecImmutability
# ===========================================================================

class TestColumnSpecImmutability:
    """Tests that frozen dataclasses cannot be mutated."""

    def test_column_spec_frozen(self):
        """ColumnSpec instances should be frozen (immutable)."""
        spec = ColumnSpec(name="test_col", required=True, dtype="float")
        with pytest.raises(AttributeError):
            spec.name = "mutated"

    def test_data_source_sla_frozen(self):
        """DataSourceSLA instances should be frozen (immutable)."""
        sla = get_sla("ghcn_daily_raw")
        with pytest.raises(AttributeError):
            sla.name = "mutated"
