"""
Tests for the operational feature engineering module.

Covers:
  - Wind-conditioned features: upwind, crosswind, downwind, gradient, advection
  - ASOS feature extraction: dewpoint depression, per-station extraction
  - Sounding feature merging: 12Z preference, fallback to 00Z
  - Full feature matrix builder integration
  - Feature name/group registry
  - Edge cases: missing wind, missing ASOS, missing sounding, all-NaN
  - No-leakage verification: features at row t use only data from t-1
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.operational_features import (
    compute_upwind_temperature,
    compute_crosswind_temperature,
    compute_downwind_temperature,
    compute_upwind_gradient,
    compute_advection_rate,
    compute_wind_conditioned_features,
    compute_dewpoint_depression,
    extract_asos_features_for_station,
    load_all_asos_features,
    load_asos_daily_for_station,
    extract_sounding_features,
    load_sounding_daily,
    load_station_metadata,
    load_asos_station_mapping,
    build_operational_feature_matrix,
    get_feature_names,
    get_feature_groups,
    merge_with_v2_features,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_dates():
    """5-day date range for testing."""
    return pd.date_range("2022-01-01", periods=5, freq="D")


@pytest.fixture
def station_bearings():
    """Station bearings in degrees from Central Park."""
    return {
        "STN_N": 0.0,      # Due north
        "STN_E": 90.0,     # Due east
        "STN_S": 180.0,    # Due south
        "STN_W": 270.0,    # Due west
    }


@pytest.fixture
def station_distances():
    """Station distances in miles from Central Park."""
    return {
        "STN_N": 50.0,
        "STN_E": 60.0,
        "STN_S": 80.0,
        "STN_W": 100.0,
    }


@pytest.fixture
def station_tmax(sample_dates):
    """Station TMAX DataFrame for 4 cardinal-direction stations."""
    return pd.DataFrame(
        {
            "STN_N": [30.0, 32.0, 28.0, 35.0, 33.0],
            "STN_E": [40.0, 42.0, 38.0, 45.0, 43.0],
            "STN_S": [50.0, 52.0, 48.0, 55.0, 53.0],
            "STN_W": [35.0, 37.0, 33.0, 40.0, 38.0],
        },
        index=sample_dates,
    )


@pytest.fixture
def stations_csv_path(tmp_path):
    """Create a minimal stations.csv."""
    csv = tmp_path / "stations.csv"
    csv.write_text(
        "station_id,name,latitude,longitude,distance_miles,direction\n"
        "USW00094728,Central Park NYC (Target),40.7789,-73.9692,0,Target\n"
        "USW00014735,Albany NY,42.7483,-73.8017,137,N\n"
        "USW00014734,Newark NJ,40.6831,-74.1694,12,W\n"
        "USW00094789,JFK Airport NY,40.6413,-73.7781,13,SE\n"
    )
    return str(csv)


@pytest.fixture
def asos_mapping_csv(tmp_path):
    """Create a minimal asos_station_mapping.csv."""
    csv = tmp_path / "asos_station_mapping.csv"
    csv.write_text(
        "station_id,station_name,icao,asos_available,notes\n"
        "USW00094728,CENTRAL PARK,KNYC,yes,ASOS\n"
        "USW00014734,NEWARK,KEWR,yes,ASOS\n"
        "USW00099999,MISSING,KXYZ,no,no ASOS\n"
    )
    return str(csv)


@pytest.fixture
def asos_daily_df():
    """Sample ASOS daily DataFrame for one station."""
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "tmax_f": [45.0, 48.0, 42.0, 50.0, 47.0],
            "tmin_f": [30.0, 32.0, 28.0, 35.0, 33.0],
            "tmean_f": [37.5, 40.0, 35.0, 42.5, 40.0],
            "dewpoint_mean_f": [25.0, 28.0, 22.0, 30.0, 27.0],
            "dewpoint_afternoon_f": [23.0, 26.0, 20.0, 28.0, 25.0],
            "wind_speed_mean_mph": [10.0, 12.0, 8.0, 15.0, 11.0],
            "wind_speed_max_mph": [20.0, 25.0, 15.0, 30.0, 22.0],
            "wind_dir_mean_deg": [180.0, 270.0, 0.0, 90.0, 225.0],
            "wind_dir_evening_deg": [190.0, 260.0, 10.0, 100.0, 230.0],
            "slp_00z_mb": [1015.0, 1012.0, 1018.0, 1010.0, 1014.0],
            "slp_12z_mb": [1014.0, 1011.0, 1017.0, 1009.0, 1013.0],
            "slp_tendency_24h_mb": [np.nan, -3.0, 6.0, -8.0, 4.0],
            "cloud_fraction_low": [0.3, 0.5, 0.1, 0.8, 0.4],
        },
        index=dates,
    )


@pytest.fixture
def sounding_daily_df():
    """Sample sounding daily DataFrame."""
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    # Create both 00Z and 12Z entries
    rows = []
    for i, dt in enumerate(dates):
        rows.append({
            "date": dt,
            "hour": 0,
            "t850_f": 25.0 + i,
            "t850_c": -3.9 + i * 0.5,
            "t500_f": -15.0 + i,
            "t500_c": -26.1 + i * 0.5,
            "wind_dir_850": 250.0 + i * 5,
            "wind_speed_850": 20.0 + i,
            "stability_index": -5.0 + i * 0.3,
            "lapse_rate_850_500": 6.5 + i * 0.1,
            "t_surface_f": 35.0 + i,
            "t_surface_c": 1.7 + i * 0.5,
        })
        rows.append({
            "date": dt,
            "hour": 12,
            "t850_f": 28.0 + i,
            "t850_c": -2.2 + i * 0.5,
            "t500_f": -13.0 + i,
            "t500_c": -25.0 + i * 0.5,
            "wind_dir_850": 260.0 + i * 5,
            "wind_speed_850": 22.0 + i,
            "stability_index": -4.0 + i * 0.3,
            "lapse_rate_850_500": 6.8 + i * 0.1,
            "t_surface_f": 38.0 + i,
            "t_surface_c": 3.3 + i * 0.5,
        })
    df = pd.DataFrame(rows)
    df = df.set_index("date")
    return df


# ===========================================================================
# Wind-Conditioned Feature Tests
# ===========================================================================

class TestComputeUpwindTemperature:

    def test_south_wind_upwind_is_south_station(self, station_tmax, station_bearings):
        """With 180-deg (S) wind, station at 180 deg bearing should dominate."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        result = compute_upwind_temperature(station_tmax, wind_dir, station_bearings)

        # Only STN_S (180 deg) has cos(0)=1; STN_E, STN_W have cos(90)=0; STN_N has cos(180)=-1 (clamped)
        # So upwind_temp should be exactly STN_S's TMAX
        for idx in dates:
            assert result.at[idx] == pytest.approx(station_tmax.at[idx, "STN_S"], abs=1e-6)

    def test_north_wind_upwind_is_north_station(self, station_tmax, station_bearings):
        """With 0-deg (N) wind, station at 0 deg bearing dominates."""
        dates = station_tmax.index
        wind_dir = pd.Series([0.0] * len(dates), index=dates)
        result = compute_upwind_temperature(station_tmax, wind_dir, station_bearings)

        for idx in dates:
            assert result.at[idx] == pytest.approx(station_tmax.at[idx, "STN_N"], abs=1e-6)

    def test_northeast_wind_blends_north_and_east(self, station_tmax, station_bearings):
        """With 45-deg (NE) wind, N and E stations share weight equally."""
        dates = station_tmax.index
        wind_dir = pd.Series([45.0] * len(dates), index=dates)
        result = compute_upwind_temperature(station_tmax, wind_dir, station_bearings)

        # cos(45-0) = cos(45) ~ 0.707 (N station)
        # cos(45-90) = cos(-45) ~ 0.707 (E station)
        # cos(45-180) = cos(-135) ~ -0.707 -> clamped to 0 (S station)
        # cos(45-270) = cos(-225) ~ -0.707 -> clamped to 0 (W station)
        w_n = np.cos(np.deg2rad(45))
        w_e = np.cos(np.deg2rad(45))
        for idx in dates:
            expected = (w_n * station_tmax.at[idx, "STN_N"] + w_e * station_tmax.at[idx, "STN_E"]) / (w_n + w_e)
            assert result.at[idx] == pytest.approx(expected, abs=1e-6)

    def test_nan_wind_direction_returns_nan(self, station_tmax, station_bearings):
        """NaN wind direction should give NaN result."""
        dates = station_tmax.index
        wind_dir = pd.Series([np.nan] * len(dates), index=dates)
        result = compute_upwind_temperature(station_tmax, wind_dir, station_bearings)
        assert result.isna().all()

    def test_all_nan_tmax_returns_nan(self, station_bearings, sample_dates):
        """All-NaN station TMAX should give NaN result."""
        tmax = pd.DataFrame(
            {sid: [np.nan] * 5 for sid in station_bearings},
            index=sample_dates,
        )
        wind_dir = pd.Series([180.0] * 5, index=sample_dates)
        result = compute_upwind_temperature(tmax, wind_dir, station_bearings)
        assert result.isna().all()

    def test_empty_bearings_returns_nan(self, station_tmax):
        """No station bearings -> all NaN."""
        wind_dir = pd.Series([180.0] * len(station_tmax), index=station_tmax.index)
        result = compute_upwind_temperature(station_tmax, wind_dir, {})
        assert result.isna().all()


class TestComputeCrosswindTemperature:

    def test_south_wind_crosswind_is_east_west(self, station_tmax, station_bearings):
        """With 180-deg wind, E (90) and W (270) are crosswind."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        result = compute_crosswind_temperature(station_tmax, wind_dir, station_bearings)

        # |sin(180-0)| = 0 (N), |sin(180-90)| = 1 (E), |sin(180-180)| = 0 (S), |sin(180-270)| = 1 (W)
        for idx in dates:
            expected = (station_tmax.at[idx, "STN_E"] + station_tmax.at[idx, "STN_W"]) / 2.0
            assert result.at[idx] == pytest.approx(expected, abs=1e-6)

    def test_east_wind_crosswind_is_north_south(self, station_tmax, station_bearings):
        """With 90-deg wind, N (0) and S (180) are crosswind."""
        dates = station_tmax.index
        wind_dir = pd.Series([90.0] * len(dates), index=dates)
        result = compute_crosswind_temperature(station_tmax, wind_dir, station_bearings)

        for idx in dates:
            expected = (station_tmax.at[idx, "STN_N"] + station_tmax.at[idx, "STN_S"]) / 2.0
            assert result.at[idx] == pytest.approx(expected, abs=1e-6)

    def test_nan_wind_returns_nan(self, station_tmax, station_bearings):
        """NaN wind direction -> NaN crosswind."""
        wind_dir = pd.Series([np.nan] * len(station_tmax), index=station_tmax.index)
        result = compute_crosswind_temperature(station_tmax, wind_dir, station_bearings)
        assert result.isna().all()


class TestComputeDownwindTemperature:

    def test_south_wind_downwind_is_north(self, station_tmax, station_bearings):
        """With 180-deg wind, station at 0 deg bearing (N) is downwind."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        result = compute_downwind_temperature(station_tmax, wind_dir, station_bearings)

        # max(0, -cos(180-0)) = max(0, -cos(180)) = max(0, 1) = 1 (N)
        # max(0, -cos(180-90)) = max(0, 0) = 0 (E)
        # max(0, -cos(180-180)) = max(0, -1) = 0 (S)
        # max(0, -cos(180-270)) = max(0, 0) = 0 (W)
        for idx in dates:
            assert result.at[idx] == pytest.approx(station_tmax.at[idx, "STN_N"], abs=1e-6)

    def test_north_wind_downwind_is_south(self, station_tmax, station_bearings):
        """With 0-deg wind, station at 180 deg bearing (S) is downwind."""
        dates = station_tmax.index
        wind_dir = pd.Series([0.0] * len(dates), index=dates)
        result = compute_downwind_temperature(station_tmax, wind_dir, station_bearings)

        for idx in dates:
            assert result.at[idx] == pytest.approx(station_tmax.at[idx, "STN_S"], abs=1e-6)

    def test_nan_wind_returns_nan(self, station_tmax, station_bearings):
        """NaN wind -> NaN downwind."""
        wind_dir = pd.Series([np.nan] * len(station_tmax), index=station_tmax.index)
        result = compute_downwind_temperature(station_tmax, wind_dir, station_bearings)
        assert result.isna().all()


class TestUpwindGradient:

    def test_positive_gradient(self):
        """Upwind warmer than NYC -> positive gradient."""
        upwind = pd.Series([60.0, 65.0, 55.0])
        nyc_prev = pd.Series([50.0, 55.0, 50.0])
        result = compute_upwind_gradient(upwind, nyc_prev)
        expected = pd.Series([10.0, 10.0, 5.0])
        pd.testing.assert_series_equal(result, expected)

    def test_negative_gradient(self):
        """Upwind cooler than NYC -> negative gradient."""
        upwind = pd.Series([40.0, 45.0])
        nyc_prev = pd.Series([50.0, 55.0])
        result = compute_upwind_gradient(upwind, nyc_prev)
        expected = pd.Series([-10.0, -10.0])
        pd.testing.assert_series_equal(result, expected)

    def test_nan_propagation(self):
        """NaN in either input -> NaN result."""
        upwind = pd.Series([60.0, np.nan, 55.0])
        nyc_prev = pd.Series([50.0, 55.0, np.nan])
        result = compute_upwind_gradient(upwind, nyc_prev)
        assert result.iloc[0] == pytest.approx(10.0)
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestAdvectionRate:

    def test_basic_advection_rate(self):
        """Simple advection rate calculation."""
        wind_speed = pd.Series([10.0, 20.0, 15.0])
        gradient = pd.Series([5.0, -3.0, 0.0])
        mean_dist = 50.0
        result = compute_advection_rate(wind_speed, gradient, mean_dist)
        expected = pd.Series([10.0 * 5.0 / 50.0, 20.0 * -3.0 / 50.0, 15.0 * 0.0 / 50.0])
        pd.testing.assert_series_equal(result, expected)

    def test_zero_distance_returns_nan(self):
        """Zero mean distance should return NaN to avoid division by zero."""
        wind_speed = pd.Series([10.0, 20.0])
        gradient = pd.Series([5.0, -3.0])
        result = compute_advection_rate(wind_speed, gradient, 0.0)
        assert result.isna().all()

    def test_negative_distance_returns_nan(self):
        """Negative mean distance should return NaN."""
        wind_speed = pd.Series([10.0])
        gradient = pd.Series([5.0])
        result = compute_advection_rate(wind_speed, gradient, -10.0)
        assert result.isna().all()


class TestComputeWindConditionedFeatures:

    def test_all_features_present(self, station_tmax, station_bearings, station_distances):
        """Should produce all 5 wind-conditioned columns."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        wind_speed = pd.Series([10.0] * len(dates), index=dates)
        nyc_prev = pd.Series([45.0] * len(dates), index=dates)

        result = compute_wind_conditioned_features(
            station_tmax, wind_dir, wind_speed, nyc_prev,
            station_bearings, station_distances,
        )
        assert set(result.columns) == {
            "upwind_temp", "crosswind_temp", "downwind_temp",
            "upwind_gradient", "advection_rate",
        }
        assert len(result) == len(dates)

    def test_consistent_with_individual_functions(
        self, station_tmax, station_bearings, station_distances
    ):
        """Composite output should match individual function outputs."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        wind_speed = pd.Series([10.0] * len(dates), index=dates)
        nyc_prev = pd.Series([45.0] * len(dates), index=dates)

        result = compute_wind_conditioned_features(
            station_tmax, wind_dir, wind_speed, nyc_prev,
            station_bearings, station_distances,
        )

        upwind = compute_upwind_temperature(station_tmax, wind_dir, station_bearings)
        pd.testing.assert_series_equal(
            result["upwind_temp"], upwind, check_names=False
        )


# ===========================================================================
# ASOS Feature Tests
# ===========================================================================

class TestDewpointDepression:

    def test_basic_depression(self):
        """Depression = T - Td."""
        tmax = pd.Series([50.0, 60.0, 70.0])
        dewpoint = pd.Series([40.0, 45.0, 55.0])
        result = compute_dewpoint_depression(tmax, dewpoint)
        expected = pd.Series([10.0, 15.0, 15.0])
        pd.testing.assert_series_equal(result, expected)

    def test_nan_propagation(self):
        """NaN in either input -> NaN output."""
        tmax = pd.Series([50.0, np.nan, 70.0])
        dewpoint = pd.Series([40.0, 45.0, np.nan])
        result = compute_dewpoint_depression(tmax, dewpoint)
        assert result.iloc[0] == pytest.approx(10.0)
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestExtractAsosFeatures:

    def test_basic_extraction(self, asos_daily_df):
        """Should produce named columns for all expected features."""
        result = extract_asos_features_for_station(asos_daily_df, "STN001")
        assert len(result) == len(asos_daily_df)
        assert f"STN001_asos_dewpoint_mean" in result.columns
        assert f"STN001_asos_dewpoint_afternoon" in result.columns
        assert f"STN001_asos_slp_00z" in result.columns
        assert f"STN001_asos_slp_12z" in result.columns
        assert f"STN001_asos_slp_tendency" in result.columns
        assert f"STN001_asos_cloud_fraction" in result.columns
        assert f"STN001_asos_wind_dir_evening" in result.columns
        assert f"STN001_asos_dewpoint_depression" in result.columns

    def test_dewpoint_depression_computed(self, asos_daily_df):
        """Dewpoint depression should be tmax_f - dewpoint_mean_f."""
        result = extract_asos_features_for_station(asos_daily_df, "STN001")
        expected = asos_daily_df["tmax_f"] - asos_daily_df["dewpoint_mean_f"]
        pd.testing.assert_series_equal(
            result["STN001_asos_dewpoint_depression"],
            expected,
            check_names=False,
        )

    def test_missing_columns_produce_nan(self):
        """Missing ASOS columns should become NaN features."""
        dates = pd.date_range("2022-01-01", periods=3, freq="D")
        sparse_df = pd.DataFrame({"tmax_f": [50, 55, 60]}, index=dates)
        result = extract_asos_features_for_station(sparse_df, "STN002")
        # dewpoint_mean column doesn't exist so mapped feature is NaN
        assert result["STN002_asos_dewpoint_mean"].isna().all()

    def test_total_feature_count(self, asos_daily_df):
        """Should produce exactly 8 features per station."""
        result = extract_asos_features_for_station(asos_daily_df, "STN001")
        assert result.shape[1] == 8


class TestLoadAsosDailyForStation:

    def test_file_exists(self, tmp_path, asos_daily_df):
        """Should load a properly formatted ASOS daily CSV."""
        asos_daily_df_with_date = asos_daily_df.copy()
        asos_daily_df_with_date.index.name = "date"
        csv_path = tmp_path / "STN001_asos_daily.csv"
        asos_daily_df_with_date.to_csv(csv_path)

        result = load_asos_daily_for_station("STN001", str(tmp_path))
        assert result is not None
        assert len(result) == 5

    def test_file_missing_returns_none(self, tmp_path):
        """Should return None when file doesn't exist."""
        result = load_asos_daily_for_station("MISSING_STN", str(tmp_path))
        assert result is None


class TestLoadAllAsosFeatures:

    def test_loads_multiple_stations(self, tmp_path, asos_daily_df):
        """Should combine features from multiple stations."""
        for sid in ["STN_A", "STN_B"]:
            df = asos_daily_df.copy()
            df.index.name = "date"
            df.to_csv(tmp_path / f"{sid}_asos_daily.csv")

        result = load_all_asos_features(["STN_A", "STN_B"], str(tmp_path))
        assert not result.empty
        assert any("STN_A_asos_" in c for c in result.columns)
        assert any("STN_B_asos_" in c for c in result.columns)

    def test_missing_station_skipped(self, tmp_path, asos_daily_df):
        """Missing stations should be skipped, not error."""
        df = asos_daily_df.copy()
        df.index.name = "date"
        df.to_csv(tmp_path / "STN_A_asos_daily.csv")

        result = load_all_asos_features(["STN_A", "MISSING"], str(tmp_path))
        assert not result.empty
        assert any("STN_A_asos_" in c for c in result.columns)

    def test_no_stations_returns_empty(self, tmp_path):
        """All missing stations -> empty DataFrame."""
        result = load_all_asos_features(["X", "Y"], str(tmp_path))
        assert result.empty


# ===========================================================================
# Sounding Feature Tests
# ===========================================================================

class TestExtractSoundingFeatures:

    def test_prefers_12z(self, sounding_daily_df):
        """Should prefer 12Z sounding data when both 00Z and 12Z exist."""
        result = extract_sounding_features(sounding_daily_df)
        # 12Z t850_f for day 0 is 28.0
        assert result["sounding_t850_f"].iloc[0] == pytest.approx(28.0)

    def test_fallback_to_00z(self):
        """Should use 00Z when 12Z is missing."""
        dates = pd.date_range("2022-01-01", periods=3, freq="D")
        rows = []
        for i, dt in enumerate(dates):
            rows.append({
                "date": dt, "hour": 0,
                "t850_f": 25.0 + i, "t500_f": -15.0,
                "wind_dir_850": 250.0, "wind_speed_850": 20.0,
                "stability_index": -5.0, "lapse_rate_850_500": 6.5,
                "t_surface_f": 35.0,
            })
        df = pd.DataFrame(rows).set_index("date")
        result = extract_sounding_features(df)
        assert result["sounding_t850_f"].iloc[0] == pytest.approx(25.0)

    def test_output_columns(self, sounding_daily_df):
        """Should produce the 7 expected sounding columns."""
        result = extract_sounding_features(sounding_daily_df)
        expected_cols = {
            "sounding_t850_f", "sounding_t500_f",
            "sounding_wind_dir_850", "sounding_wind_speed_850",
            "sounding_stability_index", "sounding_lapse_rate",
            "sounding_t_surface_f",
        }
        assert set(result.columns) == expected_cols

    def test_missing_columns_produce_nan(self):
        """Missing sounding columns should produce NaN features."""
        dates = pd.date_range("2022-01-01", periods=2, freq="D")
        df = pd.DataFrame(
            {"hour": [12, 12], "t850_f": [28.0, 30.0]},
            index=dates,
        )
        result = extract_sounding_features(df)
        assert "sounding_t850_f" in result.columns
        assert result["sounding_lapse_rate"].isna().all()

    def test_no_hour_column(self):
        """Should work even without hour column."""
        dates = pd.date_range("2022-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"t850_f": [28.0, 30.0, 32.0], "stability_index": [-4, -3, -2]},
            index=dates,
        )
        result = extract_sounding_features(df)
        assert len(result) == 3
        assert result["sounding_t850_f"].iloc[0] == pytest.approx(28.0)


class TestLoadSoundingDaily:

    def test_file_exists(self, tmp_path, sounding_daily_df):
        """Should load a properly saved sounding daily file."""
        sounding_daily_df_out = sounding_daily_df.copy()
        sounding_daily_df_out.index.name = "date"
        csv_path = tmp_path / "USM00072501_soundings_daily.csv"
        sounding_daily_df_out.to_csv(csv_path)

        result = load_sounding_daily(
            station_id="USM00072501", igra_daily_dir=str(tmp_path)
        )
        assert result is not None
        assert len(result) == len(sounding_daily_df)

    def test_file_missing_returns_none(self, tmp_path):
        """Missing file -> None."""
        result = load_sounding_daily(
            station_id="USM00072501", igra_daily_dir=str(tmp_path)
        )
        assert result is None


# ===========================================================================
# Station Metadata Tests
# ===========================================================================

class TestLoadStationMetadata:

    def test_loads_bearings(self, stations_csv_path):
        """Should parse cardinal directions to numeric bearings."""
        df = load_station_metadata(stations_csv_path)
        assert len(df) == 4

        # Albany is N -> 0 degrees
        albany = df[df["station_id"] == "USW00014735"]
        assert albany["bearing_deg"].iloc[0] == pytest.approx(0.0)

        # Newark is W -> 270 degrees
        newark = df[df["station_id"] == "USW00014734"]
        assert newark["bearing_deg"].iloc[0] == pytest.approx(270.0)

        # JFK is SE -> 135 degrees
        jfk = df[df["station_id"] == "USW00094789"]
        assert jfk["bearing_deg"].iloc[0] == pytest.approx(135.0)

        # Target station has NaN bearing
        target = df[df["station_id"] == "USW00094728"]
        assert pd.isna(target["bearing_deg"].iloc[0])


class TestLoadAsosStationMapping:

    def test_filters_available(self, asos_mapping_csv):
        """Should only return stations with asos_available='yes'."""
        mapping = load_asos_station_mapping(asos_mapping_csv)
        assert "USW00094728" in mapping
        assert "USW00014734" in mapping
        assert "USW00099999" not in mapping  # asos_available = 'no'

    def test_returns_icao(self, asos_mapping_csv):
        """Should map station_id to icao code."""
        mapping = load_asos_station_mapping(asos_mapping_csv)
        assert mapping["USW00094728"] == "KNYC"
        assert mapping["USW00014734"] == "KEWR"


# ===========================================================================
# Feature Registry Tests
# ===========================================================================

class TestGetFeatureNames:

    def test_returns_ordered_list(self):
        """Should return column names in order."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = get_feature_names(df)
        assert result == ["a", "b", "c"]


class TestGetFeatureGroups:

    def test_wind_conditioned_group(self):
        """Wind-conditioned columns should be grouped correctly."""
        df = pd.DataFrame({
            "upwind_temp_lag1": [1],
            "crosswind_temp_lag1": [2],
            "downwind_temp_lag1": [3],
            "upwind_gradient_lag1": [4],
            "advection_rate_lag1": [5],
        })
        groups = get_feature_groups(df)
        for col in df.columns:
            assert groups[col] == "wind_conditioned"

    def test_asos_group(self):
        """ASOS columns should be grouped as asos_operational."""
        df = pd.DataFrame({
            "STN_A_asos_dewpoint_mean_lag1": [1],
            "STN_B_asos_slp_00z_lag1": [2],
        })
        groups = get_feature_groups(df)
        for col in df.columns:
            assert groups[col] == "asos_operational"

    def test_sounding_group(self):
        """Sounding columns should be grouped as sounding."""
        df = pd.DataFrame({
            "sounding_t850_f_lag1": [1],
            "sounding_stability_index_lag1": [2],
        })
        groups = get_feature_groups(df)
        for col in df.columns:
            assert groups[col] == "sounding"

    def test_station_temp_group(self):
        """Station temp columns should be grouped correctly."""
        df = pd.DataFrame({
            "USW00014734_TMAX_lag1": [1],
            "USW00014734_TMIN_lag1": [2],
        })
        groups = get_feature_groups(df)
        for col in df.columns:
            assert groups[col] == "station_temperature"

    def test_mixed_groups(self):
        """Should assign correct groups to mixed columns."""
        df = pd.DataFrame({
            "upwind_temp_lag1": [1],
            "STN_A_asos_slp_00z_lag1": [2],
            "sounding_t850_f_lag1": [3],
            "USW00014734_TMAX_lag1": [4],
            "sin_day": [5],
            "cos_day": [6],
            "NYC_TMAX_lag1": [7],
            "sector_WNW_mean_lag1": [8],
            "grad_upstream_vs_coast_lag1": [9],
            "diurnal_USW00014734_lag1": [10],
            "trend_delta1_USW00014734": [11],
            "unknown_column": [12],
        })
        groups = get_feature_groups(df)
        assert groups["upwind_temp_lag1"] == "wind_conditioned"
        assert groups["STN_A_asos_slp_00z_lag1"] == "asos_operational"
        assert groups["sounding_t850_f_lag1"] == "sounding"
        assert groups["USW00014734_TMAX_lag1"] == "station_temperature"
        assert groups["sin_day"] == "date_encoding"
        assert groups["cos_day"] == "date_encoding"
        assert groups["NYC_TMAX_lag1"] == "autoregressive"
        assert groups["sector_WNW_mean_lag1"] == "sector"
        assert groups["grad_upstream_vs_coast_lag1"] == "sector_gradient"
        assert groups["diurnal_USW00014734_lag1"] == "diurnal_range"
        assert groups["trend_delta1_USW00014734"] == "trend"
        assert groups["unknown_column"] == "other"


# ===========================================================================
# Merge with V2 Features Tests
# ===========================================================================

class TestMergeWithV2Features:

    def test_basic_merge(self):
        """Should combine V2 and operational features."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        v2 = pd.DataFrame({"v2_feat": [1, 2, 3, 4, 5]}, index=dates)
        ops = pd.DataFrame({"ops_feat": [10, 20, 30, 40, 50]}, index=dates)
        result = merge_with_v2_features(v2, ops)
        assert "v2_feat" in result.columns
        assert "ops_feat" in result.columns
        assert len(result) == 5

    def test_overlapping_columns_removed(self):
        """Overlapping columns from operational should be dropped."""
        dates = pd.date_range("2022-01-01", periods=3, freq="D")
        v2 = pd.DataFrame({"shared": [1, 2, 3], "v2_only": [4, 5, 6]}, index=dates)
        ops = pd.DataFrame({"shared": [10, 20, 30], "ops_only": [7, 8, 9]}, index=dates)
        result = merge_with_v2_features(v2, ops)
        assert "v2_only" in result.columns
        assert "ops_only" in result.columns
        # 'shared' should come from v2 (not duplicated)
        assert (result["shared"] == v2["shared"]).all()
        assert result.shape[1] == 3  # shared + v2_only + ops_only

    def test_partial_date_overlap(self):
        """Operational features with fewer dates should produce NaN."""
        dates_v2 = pd.date_range("2022-01-01", periods=5, freq="D")
        dates_ops = pd.date_range("2022-01-02", periods=3, freq="D")
        v2 = pd.DataFrame({"v2_feat": range(5)}, index=dates_v2)
        ops = pd.DataFrame({"ops_feat": [10, 20, 30]}, index=dates_ops)
        result = merge_with_v2_features(v2, ops)
        assert len(result) == 5
        assert pd.isna(result.loc[dates_v2[0], "ops_feat"])
        assert result.loc[dates_v2[1], "ops_feat"] == 10


# ===========================================================================
# Integration Test: Build Feature Matrix
# ===========================================================================

class TestBuildOperationalFeatureMatrix:

    def test_with_no_external_data(self, tmp_path):
        """Should produce feature matrix even with no ASOS/sounding data.

        Uses only wind_conditioned=False, asos=False, sounding=False.
        """
        # Create a minimal merged station DataFrame
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        merged = pd.DataFrame(
            {
                "USW00094728_TMAX": np.random.uniform(30, 70, 10),
                "USW00014734_TMAX": np.random.uniform(30, 70, 10),
                "USW00014735_TMAX": np.random.uniform(25, 65, 10),
            },
            index=dates,
        )

        features, target = build_operational_feature_matrix(
            merged_station_df=merged,
            asos_daily_dir=str(tmp_path),
            igra_daily_dir=str(tmp_path),
            include_wind_conditioned=False,
            include_asos=False,
            include_sounding=False,
            station_ids=["USW00014734", "USW00014735"],
        )

        # Should have target (NYC TMAX) with 9 rows (first row dropped)
        assert len(target) == 9
        assert target.name == "NYC_TMAX"

    def test_with_asos_data(self, tmp_path, asos_daily_df):
        """Should include ASOS features when available."""
        # Write ASOS daily file
        df = asos_daily_df.copy()
        df.index.name = "date"
        df.to_csv(tmp_path / "USW00014734_asos_daily.csv")

        # Create merged station DataFrame matching ASOS dates
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        merged = pd.DataFrame(
            {
                "USW00094728_TMAX": [50, 55, 48, 60, 53],
                "USW00014734_TMAX": [45, 48, 42, 50, 47],
            },
            index=dates,
        )

        features, target = build_operational_feature_matrix(
            merged_station_df=merged,
            asos_daily_dir=str(tmp_path),
            igra_daily_dir=str(tmp_path),
            include_wind_conditioned=False,
            include_asos=True,
            include_sounding=False,
            station_ids=["USW00014734"],
        )

        # ASOS features should be present and lagged
        asos_cols = [c for c in features.columns if "_asos_" in c]
        assert len(asos_cols) > 0
        # All ASOS columns should end with _lag1
        assert all(c.endswith("_lag1") for c in asos_cols)

    def test_with_sounding_data(self, tmp_path, sounding_daily_df):
        """Should include sounding features when available."""
        sounding_daily_df_out = sounding_daily_df.copy()
        sounding_daily_df_out.index.name = "date"
        csv_path = tmp_path / "USM00072501_soundings_daily.csv"
        sounding_daily_df_out.to_csv(csv_path)

        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        merged = pd.DataFrame(
            {
                "USW00094728_TMAX": [50, 55, 48, 60, 53],
                "USW00014734_TMAX": [45, 48, 42, 50, 47],
            },
            index=dates,
        )

        features, target = build_operational_feature_matrix(
            merged_station_df=merged,
            asos_daily_dir=str(tmp_path),
            igra_daily_dir=str(tmp_path),
            include_wind_conditioned=False,
            include_asos=False,
            include_sounding=True,
            station_ids=["USW00014734"],
        )

        sounding_cols = [c for c in features.columns if "sounding_" in c]
        assert len(sounding_cols) > 0
        assert all(c.endswith("_lag1") for c in sounding_cols)

    def test_no_data_leakage(self, tmp_path, asos_daily_df):
        """Features at row t must use only data from t-1 or earlier."""
        # Write ASOS daily file
        df = asos_daily_df.copy()
        df.index.name = "date"
        df.to_csv(tmp_path / "USW00014734_asos_daily.csv")

        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        merged = pd.DataFrame(
            {
                "USW00094728_TMAX": [50, 55, 48, 60, 53],
                "USW00014734_TMAX": [45, 48, 42, 50, 47],
            },
            index=dates,
        )

        features, target = build_operational_feature_matrix(
            merged_station_df=merged,
            asos_daily_dir=str(tmp_path),
            igra_daily_dir=str(tmp_path),
            include_wind_conditioned=False,
            include_asos=True,
            include_sounding=False,
            station_ids=["USW00014734"],
        )

        # Check that ASOS features are lagged: value at t should be from t-1
        # For date 2022-01-02 (index 0 in features after lag drop),
        # the ASOS feature should contain t-1 = 2022-01-01 ASOS value
        dewp_col = [c for c in features.columns if "dewpoint_mean" in c]
        if dewp_col:
            # features index[0] = 2022-01-02, lag1 value should be from 2022-01-01
            feat_val = features.iloc[0][dewp_col[0]]
            # Original ASOS dewpoint_mean on 2022-01-01 is 25.0
            assert feat_val == pytest.approx(25.0)

    def test_first_row_dropped_due_to_lag(self, tmp_path):
        """First row should always be dropped because lag creates NaN."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        merged = pd.DataFrame(
            {
                "USW00094728_TMAX": [50, 55, 48, 60, 53],
                "USW00014734_TMAX": [45, 48, 42, 50, 47],
            },
            index=dates,
        )

        features, target = build_operational_feature_matrix(
            merged_station_df=merged,
            asos_daily_dir=str(tmp_path),
            igra_daily_dir=str(tmp_path),
            include_wind_conditioned=False,
            include_asos=False,
            include_sounding=False,
            station_ids=["USW00014734"],
        )

        # First date (2022-01-01) should not appear
        assert dates[0] not in features.index
        assert dates[0] not in target.index

    def test_missing_target_column_raises(self, tmp_path):
        """Should raise ValueError if target column is missing."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        merged = pd.DataFrame(
            {"USW00014734_TMAX": [45, 48, 42, 50, 47]},
            index=dates,
        )

        with pytest.raises(ValueError, match="Target column"):
            build_operational_feature_matrix(
                merged_station_df=merged,
                asos_daily_dir=str(tmp_path),
                igra_daily_dir=str(tmp_path),
                include_wind_conditioned=False,
                include_asos=False,
                include_sounding=False,
                station_ids=["USW00014734"],
            )


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases:

    def test_single_station_wind_features(self, sample_dates):
        """Should work with just one station."""
        tmax = pd.DataFrame({"STN_A": [50, 55, 48, 60, 53]}, index=sample_dates)
        bearings = {"STN_A": 180.0}
        distances = {"STN_A": 50.0}
        wind_dir = pd.Series([180.0] * 5, index=sample_dates)
        wind_speed = pd.Series([10.0] * 5, index=sample_dates)
        nyc_prev = pd.Series([45.0] * 5, index=sample_dates)

        result = compute_wind_conditioned_features(
            tmax, wind_dir, wind_speed, nyc_prev, bearings, distances,
        )
        assert not result["upwind_temp"].isna().all()

    def test_varying_wind_direction(self, station_tmax, station_bearings, station_distances):
        """Different wind direction each day should produce varying weights."""
        dates = station_tmax.index
        wind_dir = pd.Series([0, 90, 180, 270, 45], index=dates, dtype=float)
        wind_speed = pd.Series([10.0] * 5, index=dates)
        nyc_prev = pd.Series([45.0] * 5, index=dates)

        result = compute_wind_conditioned_features(
            station_tmax, wind_dir, wind_speed, nyc_prev,
            station_bearings, station_distances,
        )
        # Each day should have a different upwind temp
        upwind_vals = result["upwind_temp"].dropna().values
        assert len(set(upwind_vals)) > 1  # Not all same

    def test_wind_direction_360_same_as_0(self, station_tmax, station_bearings):
        """Wind direction 360 should be equivalent to 0."""
        dates = station_tmax.index
        wind_0 = pd.Series([0.0] * len(dates), index=dates)
        wind_360 = pd.Series([360.0] * len(dates), index=dates)

        result_0 = compute_upwind_temperature(station_tmax, wind_0, station_bearings)
        result_360 = compute_upwind_temperature(station_tmax, wind_360, station_bearings)

        pd.testing.assert_series_equal(result_0, result_360, atol=1e-10)

    def test_all_zero_wind_speed(self, station_tmax, station_bearings, station_distances):
        """Zero wind speed should produce zero advection rate."""
        dates = station_tmax.index
        wind_dir = pd.Series([180.0] * len(dates), index=dates)
        wind_speed = pd.Series([0.0] * len(dates), index=dates)
        nyc_prev = pd.Series([45.0] * len(dates), index=dates)

        result = compute_wind_conditioned_features(
            station_tmax, wind_dir, wind_speed, nyc_prev,
            station_bearings, station_distances,
        )
        # Advection rate should be 0 (wind_speed=0 * gradient / dist)
        assert (result["advection_rate"].dropna() == 0.0).all()

    def test_sounding_with_duplicated_dates(self):
        """Should handle duplicated date indices gracefully."""
        dates = pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02"])
        df = pd.DataFrame(
            {
                "t850_f": [28.0, 30.0, 32.0],
                "stability_index": [-4.0, -3.0, -2.0],
            },
            index=dates,
        )
        result = extract_sounding_features(df)
        # Should deduplicate
        assert not result.index.duplicated().any()
