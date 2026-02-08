"""
Tests for the enhanced data preprocessing module (src/data_preprocessing_v2.py).

Validates:
  - Sector assignments: correctness, completeness, no overlap
  - Lagged feature creation: single and multi-lag
  - Autoregressive feature creation: NYC TMAX(t-1) values
  - Diurnal range features: TMAX - TMIN
  - Sector average features: mean TMAX per sector
  - Sector gradient features: upstream vs coast, SW vs NW
  - Trend features: delta1, delta2
  - Delta-T target computation: TMAX(t) - TMAX(t-1)
  - create_enhanced_features: various flag combinations
  - Full pipeline: run_enhanced_preprocessing with synthetic data
  - No data leakage: scaler fit on train only, chronological splits
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.data_preprocessing_v2 import (
    get_sector_assignments,
    create_lagged_features,
    create_autoregressive_feature,
    create_diurnal_range_features,
    compute_sector_features,
    compute_sector_gradients,
    create_trend_features,
    create_target_columns,
    create_enhanced_features,
    run_enhanced_preprocessing,
    save_enhanced_data,
    PROCESSED_V2_DIR,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory cleaned up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_merged_df():
    """Create a synthetic merged DataFrame mimicking real station data.

    Includes target + 4 surrounding stations with TMAX and TMIN columns.
    100 days of data with controlled values for easy verification.
    """
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    data = {}
    # Target station (Central Park)
    target_id = config.TARGET_STATION
    data[f"{target_id}_TMAX"] = 50 + 20 * np.sin(
        2 * np.pi * np.arange(n_days) / 365
    ) + np.random.normal(0, 2, n_days)
    data[f"{target_id}_TMIN"] = data[f"{target_id}_TMAX"] - 10 - np.random.uniform(
        0, 5, n_days
    )

    # Use a subset of real surrounding station IDs for realistic testing
    test_stations = [
        "USW00014777",  # Scranton (WNW)
        "USW00014737",  # Allentown (WNW)
        "USW00013739",  # Philadelphia (SW)
        "USW00014792",  # Trenton (SW)
        "USW00014732",  # Islip (Coastal)
        "USW00094789",  # JFK (Coastal)
        "USW00093730",  # Atlantic City (Coastal)
        "USW00094702",  # Bridgeport (Coastal)
        "USW00014734",  # Newark (NearField)
        "USW00014739",  # LaGuardia (NearField)
        "USW00014771",  # White Plains (NearField)
        "USW00014735",  # Albany (WNW)
        "USW00014757",  # Poughkeepsie (WNW)
        "USW00014740",  # Hartford
    ]

    for i, sid in enumerate(test_stations):
        offset = (i - 5) * 2  # Some stations warmer, some cooler
        data[f"{sid}_TMAX"] = (
            48 + offset + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
            + np.random.normal(0, 2, n_days)
        )
        data[f"{sid}_TMIN"] = (
            data[f"{sid}_TMAX"] - 12 - np.random.uniform(0, 4, n_days)
        )

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def small_merged_df():
    """Create a minimal merged DataFrame for quick unit tests.

    5 days, 1 target + 2 surrounding stations.
    Values are deterministic for easy hand-checking.
    """
    dates = pd.date_range("2020-06-01", periods=10, freq="D")
    target_id = config.TARGET_STATION

    data = {
        f"{target_id}_TMAX": [70, 72, 68, 75, 80, 78, 73, 71, 69, 74],
        f"{target_id}_TMIN": [55, 57, 53, 60, 65, 63, 58, 56, 54, 59],
        "USW00014777_TMAX": [65, 67, 63, 70, 75, 73, 68, 66, 64, 69],
        "USW00014777_TMIN": [50, 52, 48, 55, 60, 58, 53, 51, 49, 54],
        "USW00013739_TMAX": [72, 74, 70, 77, 82, 80, 75, 73, 71, 76],
        "USW00013739_TMIN": [58, 60, 56, 63, 68, 66, 61, 59, 57, 62],
        "USW00014732_TMAX": [68, 70, 66, 73, 78, 76, 71, 69, 67, 72],
        "USW00014732_TMIN": [53, 55, 51, 58, 63, 61, 56, 54, 52, 57],
        "USW00014734_TMAX": [71, 73, 69, 76, 81, 79, 74, 72, 70, 75],
        "USW00014734_TMIN": [56, 58, 54, 61, 66, 64, 59, 57, 55, 60],
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ===========================================================================
# Sector Assignment Tests
# ===========================================================================

class TestGetSectorAssignments:
    """Tests for get_sector_assignments."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        sectors = get_sector_assignments()
        assert isinstance(sectors, dict)

    def test_has_four_sectors(self):
        """Should define exactly four sectors."""
        sectors = get_sector_assignments()
        assert len(sectors) == 4

    def test_sector_names(self):
        """Should have the correct sector names."""
        sectors = get_sector_assignments()
        expected = {"WNW", "SW", "Coastal", "NearField"}
        assert set(sectors.keys()) == expected

    def test_wnw_stations(self):
        """WNW sector should contain Scranton, Allentown, Albany, Poughkeepsie."""
        sectors = get_sector_assignments()
        expected = {"USW00014777", "USW00014737", "USW00014735", "USW00014757"}
        assert set(sectors["WNW"]) == expected

    def test_sw_stations(self):
        """SW sector should contain Philadelphia and Trenton."""
        sectors = get_sector_assignments()
        expected = {"USW00013739", "USW00014792"}
        assert set(sectors["SW"]) == expected

    def test_coastal_stations(self):
        """Coastal sector should contain Islip, JFK, Atlantic City, Bridgeport."""
        sectors = get_sector_assignments()
        expected = {"USW00014732", "USW00094789", "USW00093730", "USW00094702"}
        assert set(sectors["Coastal"]) == expected

    def test_nearfield_stations(self):
        """NearField sector should contain Newark, LaGuardia, White Plains."""
        sectors = get_sector_assignments()
        expected = {"USW00014734", "USW00014739", "USW00014771"}
        assert set(sectors["NearField"]) == expected

    def test_no_overlapping_stations(self):
        """No station should appear in more than one sector."""
        sectors = get_sector_assignments()
        all_ids = []
        for ids in sectors.values():
            all_ids.extend(ids)
        assert len(all_ids) == len(set(all_ids)), "Duplicate station in sectors"

    def test_all_stations_are_valid(self):
        """All station IDs in sectors should be in config.SURROUNDING_STATIONS."""
        sectors = get_sector_assignments()
        for sector_name, ids in sectors.items():
            for sid in ids:
                assert sid in config.SURROUNDING_STATIONS, (
                    f"{sid} in sector {sector_name} is not in SURROUNDING_STATIONS"
                )

    def test_target_station_not_in_sectors(self):
        """Target station should not be in any sector."""
        sectors = get_sector_assignments()
        for sector_name, ids in sectors.items():
            assert config.TARGET_STATION not in ids, (
                f"Target station found in sector {sector_name}"
            )


# ===========================================================================
# Lagged Feature Tests
# ===========================================================================

class TestCreateLaggedFeatures:
    """Tests for create_lagged_features."""

    def test_single_lag(self, small_merged_df):
        """Single lag should create one column per station-variable pair."""
        result = create_lagged_features(
            small_merged_df, [1],
            ["USW00014777"], ["TMAX"],
        )
        assert "USW00014777_TMAX_lag1" in result.columns
        assert len(result.columns) == 1

    def test_multi_lag(self, small_merged_df):
        """Multiple lags should create one column per lag per variable."""
        result = create_lagged_features(
            small_merged_df, [1, 2, 3],
            ["USW00014777"], ["TMAX"],
        )
        expected_cols = {
            "USW00014777_TMAX_lag1",
            "USW00014777_TMAX_lag2",
            "USW00014777_TMAX_lag3",
        }
        assert set(result.columns) == expected_cols

    def test_lag_values_are_shifted(self, small_merged_df):
        """Lag-1 value at date t should be the original value at t-1."""
        result = create_lagged_features(
            small_merged_df, [1],
            ["USW00014777"], ["TMAX"],
        )
        # Row at index 1 (2020-06-02) should have value from index 0 (2020-06-01)
        original = small_merged_df["USW00014777_TMAX"].iloc[0]
        lagged = result["USW00014777_TMAX_lag1"].iloc[1]
        assert lagged == original

    def test_first_row_is_nan_for_lag1(self, small_merged_df):
        """First row should be NaN for lag=1."""
        result = create_lagged_features(
            small_merged_df, [1],
            ["USW00014777"], ["TMAX"],
        )
        assert pd.isna(result["USW00014777_TMAX_lag1"].iloc[0])

    def test_multiple_variables(self, small_merged_df):
        """Should create columns for both TMAX and TMIN."""
        result = create_lagged_features(
            small_merged_df, [1],
            ["USW00014777"], ["TMAX", "TMIN"],
        )
        assert "USW00014777_TMAX_lag1" in result.columns
        assert "USW00014777_TMIN_lag1" in result.columns

    def test_preserves_index(self, small_merged_df):
        """Result should have the same index as input."""
        result = create_lagged_features(
            small_merged_df, [1],
            ["USW00014777"], ["TMAX"],
        )
        pd.testing.assert_index_equal(result.index, small_merged_df.index)


# ===========================================================================
# Autoregressive Feature Tests
# ===========================================================================

class TestCreateAutoRegressiveFeature:
    """Tests for create_autoregressive_feature."""

    def test_creates_nyc_tmax_lag1(self, small_merged_df):
        """Should create NYC_TMAX_lag1 column."""
        result = create_autoregressive_feature(small_merged_df, [1])
        assert "NYC_TMAX_lag1" in result.columns

    def test_lag_value_is_previous_day(self, small_merged_df):
        """NYC_TMAX_lag1 at day t should be TMAX from day t-1."""
        result = create_autoregressive_feature(small_merged_df, [1])
        target_col = f"{config.TARGET_STATION}_TMAX"
        expected = small_merged_df[target_col].iloc[0]
        actual = result["NYC_TMAX_lag1"].iloc[1]
        assert actual == expected

    def test_multi_lag(self, small_merged_df):
        """Should create columns for multiple lags."""
        result = create_autoregressive_feature(small_merged_df, [1, 2])
        assert "NYC_TMAX_lag1" in result.columns
        assert "NYC_TMAX_lag2" in result.columns

    def test_no_current_day_leakage(self, small_merged_df):
        """Lag-1 should NOT contain same-day target value."""
        result = create_autoregressive_feature(small_merged_df, [1])
        target_col = f"{config.TARGET_STATION}_TMAX"
        for i in range(1, len(result)):
            lag_val = result["NYC_TMAX_lag1"].iloc[i]
            prev_val = small_merged_df[target_col].iloc[i - 1]
            assert lag_val == prev_val


# ===========================================================================
# Diurnal Range Tests
# ===========================================================================

class TestCreateDiurnalRangeFeatures:
    """Tests for create_diurnal_range_features."""

    def test_creates_diurnal_columns(self, small_merged_df):
        """Should create diurnal_{station}_lag{k} columns."""
        result = create_diurnal_range_features(
            small_merged_df, [1], ["USW00014777"],
        )
        assert "diurnal_USW00014777_lag1" in result.columns

    def test_diurnal_value_is_tmax_minus_tmin(self, small_merged_df):
        """Diurnal range should be TMAX - TMIN from the lagged day."""
        result = create_diurnal_range_features(
            small_merged_df, [1], ["USW00014777"],
        )
        # Check lag-1 value at day 1 = (TMAX - TMIN) at day 0
        tmax_day0 = small_merged_df["USW00014777_TMAX"].iloc[0]
        tmin_day0 = small_merged_df["USW00014777_TMIN"].iloc[0]
        expected = tmax_day0 - tmin_day0
        actual = result["diurnal_USW00014777_lag1"].iloc[1]
        assert abs(actual - expected) < 1e-10

    def test_multi_station_diurnal(self, small_merged_df):
        """Should compute diurnal for multiple stations."""
        result = create_diurnal_range_features(
            small_merged_df, [1], ["USW00014777", "USW00013739"],
        )
        assert "diurnal_USW00014777_lag1" in result.columns
        assert "diurnal_USW00013739_lag1" in result.columns


# ===========================================================================
# Sector Feature Tests
# ===========================================================================

class TestComputeSectorFeatures:
    """Tests for compute_sector_features."""

    def test_creates_sector_columns(self, synthetic_merged_df):
        """Should create sector mean columns for each sector and lag."""
        sectors = get_sector_assignments()
        result = compute_sector_features(synthetic_merged_df, sectors, [1])
        for sector_name in sectors:
            assert f"sector_{sector_name}_mean_lag1" in result.columns

    def test_sector_mean_is_average_of_stations(self, small_merged_df):
        """Sector mean should be the average TMAX of stations in that sector."""
        # Small df has USW00014777 (WNW) and USW00013739 (SW)
        sectors = {"WNW": ["USW00014777"], "SW": ["USW00013739"]}
        result = compute_sector_features(small_merged_df, sectors, [1])

        # For WNW with one station, sector mean == that station's TMAX
        wnw_day0 = small_merged_df["USW00014777_TMAX"].iloc[0]
        wnw_lag1_day1 = result["sector_WNW_mean_lag1"].iloc[1]
        assert abs(wnw_lag1_day1 - wnw_day0) < 1e-10

    def test_sector_mean_multiple_stations(self, synthetic_merged_df):
        """Sector mean with multiple stations should be their average."""
        sectors = {
            "WNW": ["USW00014777", "USW00014737"],
        }
        result = compute_sector_features(synthetic_merged_df, sectors, [1])

        # Check value at day 1 (lag1 = day 0 values)
        t1 = synthetic_merged_df["USW00014777_TMAX"].iloc[0]
        t2 = synthetic_merged_df["USW00014737_TMAX"].iloc[0]
        expected = (t1 + t2) / 2
        actual = result["sector_WNW_mean_lag1"].iloc[1]
        assert abs(actual - expected) < 1e-10


# ===========================================================================
# Sector Gradient Tests
# ===========================================================================

class TestComputeSectorGradients:
    """Tests for compute_sector_gradients."""

    def test_creates_gradient_columns(self, synthetic_merged_df):
        """Should create upstream_vs_coast and SW_vs_NW gradient columns."""
        sectors = get_sector_assignments()
        result = compute_sector_gradients(synthetic_merged_df, sectors, [1])
        assert "grad_upstream_vs_coast_lag1" in result.columns
        assert "grad_SW_vs_NW_lag1" in result.columns

    def test_gradient_values(self, small_merged_df):
        """Gradient should be sector_A_mean - sector_B_mean."""
        # WNW = USW00014777 only, Coastal = USW00014732 only
        sectors = {
            "WNW": ["USW00014777"],
            "SW": ["USW00013739"],
            "Coastal": ["USW00014732"],
        }
        result = compute_sector_gradients(small_merged_df, sectors, [1])

        # At day 1, lag1 = day 0 values
        wnw = small_merged_df["USW00014777_TMAX"].iloc[0]  # 65
        coastal = small_merged_df["USW00014732_TMAX"].iloc[0]  # 68
        expected_grad1 = wnw - coastal  # 65 - 68 = -3
        actual_grad1 = result["grad_upstream_vs_coast_lag1"].iloc[1]
        assert abs(actual_grad1 - expected_grad1) < 1e-10

        sw = small_merged_df["USW00013739_TMAX"].iloc[0]  # 72
        expected_grad2 = sw - wnw  # 72 - 65 = 7
        actual_grad2 = result["grad_SW_vs_NW_lag1"].iloc[1]
        assert abs(actual_grad2 - expected_grad2) < 1e-10

    def test_multi_lag_gradients(self, synthetic_merged_df):
        """Should create gradient columns for multiple lags."""
        sectors = get_sector_assignments()
        result = compute_sector_gradients(synthetic_merged_df, sectors, [1, 2])
        assert "grad_upstream_vs_coast_lag1" in result.columns
        assert "grad_upstream_vs_coast_lag2" in result.columns
        assert "grad_SW_vs_NW_lag1" in result.columns
        assert "grad_SW_vs_NW_lag2" in result.columns


# ===========================================================================
# Trend Feature Tests
# ===========================================================================

class TestCreateTrendFeatures:
    """Tests for create_trend_features."""

    def test_creates_delta1_and_delta2(self, small_merged_df):
        """Should create trend_delta1 and trend_delta2 for each station."""
        result = create_trend_features(small_merged_df, ["USW00014777"])
        assert "trend_delta1_USW00014777" in result.columns
        assert "trend_delta2_USW00014777" in result.columns

    def test_delta1_value(self, small_merged_df):
        """trend_delta1 = T(t-1) - T(t-2)."""
        result = create_trend_features(small_merged_df, ["USW00014777"])
        # At index 2: delta1 = T(1) - T(0) = 67 - 65 = 2
        tmax = small_merged_df["USW00014777_TMAX"]
        expected = tmax.iloc[1] - tmax.iloc[0]
        actual = result["trend_delta1_USW00014777"].iloc[2]
        assert abs(actual - expected) < 1e-10

    def test_delta2_value(self, small_merged_df):
        """trend_delta2 = T(t-2) - T(t-3)."""
        result = create_trend_features(small_merged_df, ["USW00014777"])
        # At index 3: delta2 = T(1) - T(0) = 67 - 65 = 2
        tmax = small_merged_df["USW00014777_TMAX"]
        expected = tmax.iloc[1] - tmax.iloc[0]
        actual = result["trend_delta2_USW00014777"].iloc[3]
        assert abs(actual - expected) < 1e-10

    def test_first_rows_are_nan(self, small_merged_df):
        """Initial rows should be NaN due to shifting."""
        result = create_trend_features(small_merged_df, ["USW00014777"])
        # delta1 needs shift(1) and shift(2), so first two rows are NaN
        assert pd.isna(result["trend_delta1_USW00014777"].iloc[0])
        assert pd.isna(result["trend_delta1_USW00014777"].iloc[1])


# ===========================================================================
# Delta-T Target Tests
# ===========================================================================

class TestCreateTargetColumns:
    """Tests for create_target_columns."""

    def test_returns_three_series(self, small_merged_df):
        """Should return raw_target, delta_target, nyc_tmax_prev."""
        raw, delta, prev = create_target_columns(small_merged_df)
        assert isinstance(raw, pd.Series)
        assert isinstance(delta, pd.Series)
        assert isinstance(prev, pd.Series)

    def test_raw_target_values(self, small_merged_df):
        """Raw target should match target station TMAX."""
        raw, _, _ = create_target_columns(small_merged_df)
        target_col = f"{config.TARGET_STATION}_TMAX"
        pd.testing.assert_series_equal(
            raw, small_merged_df[target_col].rename("NYC_TMAX"),
        )

    def test_delta_target_values(self, small_merged_df):
        """Delta target should be TMAX(t) - TMAX(t-1)."""
        raw, delta, _ = create_target_columns(small_merged_df)
        # At index 1: delta = TMAX[1] - TMAX[0] = 72 - 70 = 2
        expected = small_merged_df[f"{config.TARGET_STATION}_TMAX"].iloc[1] - \
                   small_merged_df[f"{config.TARGET_STATION}_TMAX"].iloc[0]
        assert abs(delta.iloc[1] - expected) < 1e-10

    def test_delta_first_row_is_nan(self, small_merged_df):
        """Delta target at first row should be NaN (no previous day)."""
        _, delta, _ = create_target_columns(small_merged_df)
        assert pd.isna(delta.iloc[0])

    def test_reconstruction_identity(self, small_merged_df):
        """prev + delta should reconstruct raw target."""
        raw, delta, prev = create_target_columns(small_merged_df)
        reconstructed = prev + delta
        # Compare from index 1 onward (index 0 has NaN delta)
        np.testing.assert_allclose(
            reconstructed.iloc[1:].values,
            raw.iloc[1:].values,
            atol=1e-10,
        )

    def test_nyc_prev_is_shifted(self, small_merged_df):
        """nyc_tmax_prev at index i should be TMAX at index i-1."""
        _, _, prev = create_target_columns(small_merged_df)
        target_col = f"{config.TARGET_STATION}_TMAX"
        for i in range(1, len(prev)):
            assert prev.iloc[i] == small_merged_df[target_col].iloc[i - 1]


# ===========================================================================
# Enhanced Feature Creation Tests
# ===========================================================================

class TestCreateEnhancedFeatures:
    """Tests for create_enhanced_features."""

    def test_minimal_features(self, synthetic_merged_df):
        """With all extras disabled, should create lagged + cyclical features."""
        features, raw, delta, prev = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
        )
        assert "sin_day" in features.columns
        assert "cos_day" in features.columns
        # Should NOT have autoregressive, diurnal, sector, or trend columns
        assert not any("NYC_TMAX_lag" in c for c in features.columns)
        assert not any("diurnal_" in c for c in features.columns)
        assert not any("sector_" in c for c in features.columns)
        assert not any("trend_" in c for c in features.columns)

    def test_autoregressive_flag(self, synthetic_merged_df):
        """With include_autoregressive=True, should add NYC_TMAX_lag1."""
        features, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        assert "NYC_TMAX_lag1" in features.columns

    def test_diurnal_flag(self, synthetic_merged_df):
        """With include_diurnal=True, should add diurnal range columns."""
        features, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=False,
            include_diurnal=True,
            include_sectors=False,
            include_trends=False,
        )
        diurnal_cols = [c for c in features.columns if c.startswith("diurnal_")]
        assert len(diurnal_cols) > 0

    def test_sectors_flag(self, synthetic_merged_df):
        """With include_sectors=True, should add sector and gradient columns."""
        features, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=True,
            include_trends=False,
        )
        sector_cols = [c for c in features.columns if c.startswith("sector_")]
        grad_cols = [c for c in features.columns if c.startswith("grad_")]
        assert len(sector_cols) > 0
        assert len(grad_cols) > 0

    def test_trends_flag(self, synthetic_merged_df):
        """With include_trends=True, should add trend columns."""
        features, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=True,
        )
        trend_cols = [c for c in features.columns if c.startswith("trend_")]
        assert len(trend_cols) > 0

    def test_all_features_enabled(self, synthetic_merged_df):
        """With all flags True, should have all feature types."""
        features, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=True,
            include_sectors=True,
            include_trends=True,
        )
        assert any("NYC_TMAX_lag" in c for c in features.columns)
        assert any("diurnal_" in c for c in features.columns)
        assert any("sector_" in c for c in features.columns)
        assert any("grad_" in c for c in features.columns)
        assert any("trend_" in c for c in features.columns)

    def test_no_nan_in_targets_after_trimming(self, synthetic_merged_df):
        """After trimming, raw and delta targets should have no NaNs."""
        _, raw, delta, prev = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=True,
            include_sectors=True,
            include_trends=True,
        )
        assert raw.isna().sum() == 0
        assert delta.isna().sum() == 0
        assert prev.isna().sum() == 0

    def test_feature_count_increases_with_flags(self, synthetic_merged_df):
        """More feature flags should result in more columns."""
        feat_min, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        feat_max, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=True,
            include_sectors=True,
            include_trends=True,
        )
        assert feat_max.shape[1] > feat_min.shape[1]

    def test_multi_lag_creates_more_features(self, synthetic_merged_df):
        """Multi-lag should create more features than single-lag."""
        feat_1, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
        )
        feat_3, _, _, _ = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1, 2, 3],
        )
        assert feat_3.shape[1] > feat_1.shape[1]

    def test_reconstruction_identity_after_trim(self, synthetic_merged_df):
        """prev + delta should reconstruct raw target (after trimming)."""
        _, raw, delta, prev = create_enhanced_features(
            synthetic_merged_df,
            include_autoregressive=True,
        )
        reconstructed = prev + delta
        np.testing.assert_allclose(
            reconstructed.values, raw.values, atol=1e-10,
        )


# ===========================================================================
# Full Pipeline Tests
# ===========================================================================

class TestRunEnhancedPreprocessing:
    """Tests for run_enhanced_preprocessing with real data (if available)."""

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_pipeline_runs_end_to_end(self, tmp_dir):
        """Full pipeline should run without error on real data."""
        result = run_enhanced_preprocessing(
            output_dir=tmp_dir,
            include_autoregressive=True,
            include_diurnal=True,
            include_sectors=True,
            include_trends=True,
        )
        assert "X_train" in result
        assert "y_train" in result
        assert "y_train_delta" in result
        assert "nyc_prev_train" in result
        assert result["X_train"].shape[0] > 0

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_pipeline_saves_files(self, tmp_dir):
        """Pipeline should save all expected files."""
        run_enhanced_preprocessing(
            output_dir=tmp_dir,
            include_autoregressive=True,
        )
        expected_files = [
            "features_train.csv", "features_val.csv", "features_test.csv",
            "target_train.csv", "target_val.csv", "target_test.csv",
            "target_delta_train.csv", "target_delta_val.csv",
            "target_delta_test.csv",
            "nyc_prev_train.csv", "nyc_prev_val.csv", "nyc_prev_test.csv",
            "scaler.pkl",
        ]
        for fname in expected_files:
            fpath = os.path.join(tmp_dir, fname)
            assert os.path.isfile(fpath), f"Missing file: {fname}"

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_scaler_fit_on_train_only(self, tmp_dir):
        """Scaler should be fit on training data only (no leakage)."""
        result = run_enhanced_preprocessing(
            output_dir=tmp_dir,
            include_autoregressive=True,
        )
        # Training features should have mean ~0 and std ~1
        train_means = result["X_train"].mean()
        for col in train_means.index:
            assert abs(train_means[col]) < 0.1, (
                f"Train mean for {col} is {train_means[col]:.4f}, "
                "expected ~0 if scaler is correct"
            )

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_chronological_split_order(self, tmp_dir):
        """Train dates should precede val dates, which precede test dates."""
        result = run_enhanced_preprocessing(
            output_dir=tmp_dir,
            include_autoregressive=True,
        )
        assert result["X_train"].index.max() < result["X_val"].index.min()
        assert result["X_val"].index.max() < result["X_test"].index.min()

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_delta_reconstruction_on_real_data(self, tmp_dir):
        """Delta reconstruction should match raw target on real data."""
        result = run_enhanced_preprocessing(
            output_dir=tmp_dir,
            include_autoregressive=True,
        )
        for split in ["train", "val", "test"]:
            nyc_prev = result[f"nyc_prev_{split}"]
            delta = result[f"y_{split}_delta"]
            raw = result[f"y_{split}"]
            reconstructed = nyc_prev + delta
            np.testing.assert_allclose(
                reconstructed.values, raw.values, atol=1e-6,
                err_msg=f"Reconstruction failed for {split} split",
            )

    @pytest.mark.skipif(
        not os.path.isdir(config.RAW_DATA_DIR)
        or len(os.listdir(config.RAW_DATA_DIR)) < 5,
        reason="Raw data not available",
    )
    def test_no_autoregressive_feature_count(self, tmp_dir):
        """Without autoregressive, should have fewer features."""
        result_no = run_enhanced_preprocessing(
            output_dir=os.path.join(tmp_dir, "no_ar"),
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        result_ar = run_enhanced_preprocessing(
            output_dir=os.path.join(tmp_dir, "ar"),
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        assert result_ar["n_features"] > result_no["n_features"]
