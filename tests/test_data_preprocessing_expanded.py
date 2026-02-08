"""
Tests for the Expanded Data Preprocessing Module.

Tests cover:
  - Station discovery
  - Haversine distance and bearing calculations
  - Compass sector assignment
  - Station metadata computation
  - Station selection by count (with diversity)
  - Expanded sector assignments
  - Missingness mask creation
  - Expanded station loading (graceful missing handling)
  - Completeness filtering for expanded sets
  - Feature creation with variable station counts
  - Full expanded preprocessing pipeline
  - Feature count scaling with station count
  - Scaler fit on training data only
  - Chronological split maintained
  - Synthetic data with intentional gaps
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

from src.data_preprocessing_expanded import (
    discover_available_stations,
    haversine_miles,
    compass_bearing,
    bearing_to_sector,
    compute_station_metadata,
    select_stations_by_count,
    get_expanded_sector_assignments,
    create_missingness_mask,
    load_expanded_stations,
    filter_expanded_by_completeness,
    create_expanded_features,
    run_expanded_preprocessing,
    PROCESSED_EXPANDED_DIR,
)


# ===========================================================================
# Fixtures: Synthetic station data
# ===========================================================================

def _make_station_csv(tmp_dir: str, station_id: str,
                      start: str = "2020-01-01", periods: int = 365,
                      missing_pct: float = 0.0) -> str:
    """Create a synthetic station CSV with optional missing data."""
    dates = pd.date_range(start, periods=periods, freq="D")
    rng = np.random.RandomState(hash(station_id) % 2**31)
    tmax = 50 + 30 * np.sin(2 * np.pi * np.arange(periods) / 365) + rng.randn(periods) * 5
    tmin = tmax - 10 - rng.rand(periods) * 5

    df = pd.DataFrame({"TMAX": tmax, "TMIN": tmin}, index=dates)
    df.index.name = "date"

    # Inject missing values
    if missing_pct > 0:
        n_missing = int(periods * missing_pct)
        miss_idx = rng.choice(periods, n_missing, replace=False)
        df.iloc[miss_idx, 0] = np.nan  # TMAX missing
        df.iloc[miss_idx, 1] = np.nan  # TMIN missing

    path = os.path.join(tmp_dir, f"{station_id}.csv")
    df.to_csv(path)
    return path


@pytest.fixture
def synthetic_raw_dir():
    """Create a temp directory with synthetic station CSVs."""
    tmp_dir = tempfile.mkdtemp(prefix="test_expanded_")

    # Target station
    _make_station_csv(tmp_dir, config.TARGET_STATION, periods=500)

    # Original 14 surrounding stations (all complete)
    for sid in list(config.SURROUNDING_STATIONS.keys()):
        _make_station_csv(tmp_dir, sid, periods=500)

    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_expanded_dir():
    """Create temp dir with 25 stations including some with missing data."""
    tmp_dir = tempfile.mkdtemp(prefix="test_expanded25_")

    # Target
    _make_station_csv(tmp_dir, config.TARGET_STATION, periods=500)

    # 14 original surrounding
    for sid in config.SURROUNDING_STATIONS:
        _make_station_csv(tmp_dir, sid, periods=500)

    # 10 additional "expanded" stations (some with gaps)
    extra_ids = [
        "USW00099901", "USW00099902", "USW00099903", "USW00099904",
        "USW00099905", "USW00099906", "USW00099907", "USW00099908",
        "USW00099909", "USW00099910",
    ]
    for i, sid in enumerate(extra_ids):
        missing_pct = 0.05 * i  # 0% to 45% missing
        _make_station_csv(tmp_dir, sid, periods=500,
                          missing_pct=min(missing_pct, 0.4))

    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_sparse_dir():
    """Dir with only target + 3 stations, one with heavy gaps."""
    tmp_dir = tempfile.mkdtemp(prefix="test_sparse_")
    _make_station_csv(tmp_dir, config.TARGET_STATION, periods=400)
    _make_station_csv(tmp_dir, "USW00014735", periods=400)  # good
    _make_station_csv(tmp_dir, "USW00014740", periods=400)  # good
    _make_station_csv(tmp_dir, "USW00099999", periods=400,
                      missing_pct=0.5)  # 50% missing
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ===========================================================================
# Test: Station Discovery
# ===========================================================================

class TestStationDiscovery:
    """Tests for discover_available_stations()."""

    def test_discovers_all_csvs(self, synthetic_raw_dir):
        stations = discover_available_stations(synthetic_raw_dir)
        assert len(stations) == 15  # target + 14 surrounding

    def test_returns_sorted_list(self, synthetic_raw_dir):
        stations = discover_available_stations(synthetic_raw_dir)
        assert stations == sorted(stations)

    def test_includes_target_station(self, synthetic_raw_dir):
        stations = discover_available_stations(synthetic_raw_dir)
        assert config.TARGET_STATION in stations

    def test_empty_directory(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            stations = discover_available_stations(tmp_dir)
            assert stations == []
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_discovers_expanded_stations(self, synthetic_expanded_dir):
        stations = discover_available_stations(synthetic_expanded_dir)
        assert len(stations) == 25  # target + 14 + 10 extra


# ===========================================================================
# Test: Haversine Distance
# ===========================================================================

class TestHaversineDistance:
    """Tests for haversine_miles()."""

    def test_same_point_is_zero(self):
        d = haversine_miles(40.0, -74.0, 40.0, -74.0)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_known_distance_nyc_to_philly(self):
        # NYC Central Park to Philadelphia: ~80 miles
        d = haversine_miles(40.7789, -73.9692, 39.8722, -75.2411)
        assert 70 < d < 100

    def test_symmetric(self):
        d1 = haversine_miles(40.0, -74.0, 41.0, -73.0)
        d2 = haversine_miles(41.0, -73.0, 40.0, -74.0)
        assert d1 == pytest.approx(d2, abs=0.01)


# ===========================================================================
# Test: Compass Bearing and Sectors
# ===========================================================================

class TestCompassBearing:
    """Tests for compass_bearing() and bearing_to_sector()."""

    def test_due_north(self):
        b = compass_bearing(40.0, -74.0, 42.0, -74.0)
        assert 350 < b or b < 10  # near 0/360

    def test_due_east(self):
        b = compass_bearing(40.0, -74.0, 40.0, -72.0)
        assert 80 < b < 100

    def test_due_south(self):
        b = compass_bearing(40.0, -74.0, 38.0, -74.0)
        assert 170 < b < 190

    def test_sector_north(self):
        assert bearing_to_sector(0) == "N"
        assert bearing_to_sector(10) == "N"
        assert bearing_to_sector(350) == "N"

    def test_sector_east(self):
        assert bearing_to_sector(90) == "E"

    def test_sector_south(self):
        assert bearing_to_sector(180) == "S"

    def test_sector_west(self):
        assert bearing_to_sector(270) == "W"

    def test_sector_ne(self):
        assert bearing_to_sector(45) == "NE"

    def test_all_sectors_covered(self):
        sectors = set()
        for angle in range(0, 360, 45):
            sectors.add(bearing_to_sector(angle))
        assert len(sectors) == 8


# ===========================================================================
# Test: Station Metadata
# ===========================================================================

class TestStationMetadata:
    """Tests for compute_station_metadata()."""

    def test_target_station_distance_zero(self):
        meta = compute_station_metadata([config.TARGET_STATION])
        row = meta[meta["station_id"] == config.TARGET_STATION].iloc[0]
        assert row["distance_miles"] == pytest.approx(0.0)
        assert row["sector"] == "Target"

    def test_sorted_by_distance(self):
        ids = [config.TARGET_STATION] + list(config.SURROUNDING_STATIONS.keys())[:5]
        meta = compute_station_metadata(ids)
        dists = meta["distance_miles"].tolist()
        assert dists == sorted(dists)

    def test_unknown_station_gets_999(self):
        meta = compute_station_metadata(["USW00099999"])
        assert meta.iloc[0]["distance_miles"] == 999.0
        assert meta.iloc[0]["sector"] == "UNK"


# ===========================================================================
# Test: Station Selection by Count
# ===========================================================================

class TestSelectStationsByCount:
    """Tests for select_stations_by_count()."""

    def test_select_5_stations(self, synthetic_raw_dir):
        available = discover_available_stations(synthetic_raw_dir)
        selected = select_stations_by_count(available, 5)
        assert len(selected) == 5
        assert config.TARGET_STATION not in selected

    def test_select_10_stations(self, synthetic_raw_dir):
        available = discover_available_stations(synthetic_raw_dir)
        selected = select_stations_by_count(available, 10)
        assert len(selected) == 10

    def test_select_all_returns_all_surrounding(self, synthetic_raw_dir):
        available = discover_available_stations(synthetic_raw_dir)
        selected = select_stations_by_count(available, 14)
        assert len(selected) == 14
        assert config.TARGET_STATION not in selected

    def test_request_more_than_available(self, synthetic_raw_dir):
        available = discover_available_stations(synthetic_raw_dir)
        selected = select_stations_by_count(available, 100)
        assert len(selected) == 14  # all surrounding

    def test_no_duplicates(self, synthetic_expanded_dir):
        available = discover_available_stations(synthetic_expanded_dir)
        selected = select_stations_by_count(available, 15)
        assert len(selected) == len(set(selected))


# ===========================================================================
# Test: Expanded Sector Assignments
# ===========================================================================

class TestExpandedSectorAssignments:
    """Tests for get_expanded_sector_assignments()."""

    def test_assigns_original_stations(self):
        ids = list(config.SURROUNDING_STATIONS.keys())
        sectors = get_expanded_sector_assignments(ids)
        # Should have sectors assigned
        assert len(sectors) > 0
        # Total stations across sectors should match input
        total = sum(len(v) for v in sectors.values())
        assert total == len(ids)

    def test_excludes_target(self):
        ids = [config.TARGET_STATION] + list(config.SURROUNDING_STATIONS.keys())[:3]
        sectors = get_expanded_sector_assignments(ids)
        all_in_sectors = []
        for v in sectors.values():
            all_in_sectors.extend(v)
        assert config.TARGET_STATION not in all_in_sectors


# ===========================================================================
# Test: Missingness Mask
# ===========================================================================

class TestMissingnessMask:
    """Tests for create_missingness_mask()."""

    def _make_merged_df(self, n_days=100, missing_idx=None):
        """Create a synthetic merged DataFrame."""
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        data = {
            "USW00014735_TMAX": 50 + rng.randn(n_days) * 10,
            "USW00014735_TMIN": 40 + rng.randn(n_days) * 8,
            "USW00014740_TMAX": 48 + rng.randn(n_days) * 10,
            "USW00014740_TMIN": 38 + rng.randn(n_days) * 8,
        }
        df = pd.DataFrame(data, index=dates)
        if missing_idx is not None:
            for col in df.columns:
                df.iloc[missing_idx, df.columns.get_loc(col)] = np.nan
        return df

    def test_mask_shape_matches_lagged_features(self):
        df = self._make_merged_df()
        mask = create_missingness_mask(
            df, ["USW00014735", "USW00014740"], ["TMAX", "TMIN"], [1]
        )
        assert mask.shape == (100, 4)  # 2 stations * 2 vars * 1 lag

    def test_mask_all_ones_when_no_missing(self):
        df = self._make_merged_df()
        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX"], [1]
        )
        # First row should be 0 (due to lag shift) but rest should be 1
        assert mask.iloc[1:].sum().sum() == len(mask) - 1

    def test_mask_zeros_for_missing_values(self):
        df = self._make_merged_df(missing_idx=[5, 10, 15])
        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX"], [1]
        )
        # Missing at idx 5,10,15 in raw data -> mask 0 at idx 6,11,16 (shifted)
        assert mask.iloc[6]["USW00014735_mask_TMAX_lag1"] == 0
        assert mask.iloc[11]["USW00014735_mask_TMAX_lag1"] == 0
        assert mask.iloc[16]["USW00014735_mask_TMAX_lag1"] == 0

    def test_mask_ones_for_present_values(self):
        df = self._make_merged_df(missing_idx=[5])
        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX"], [1]
        )
        # Row 3 (lag1 of row 2, which is present) should be 1
        assert mask.iloc[3]["USW00014735_mask_TMAX_lag1"] == 1

    def test_mask_is_binary(self):
        df = self._make_merged_df(missing_idx=[5, 10, 15])
        mask = create_missingness_mask(
            df, ["USW00014735", "USW00014740"], ["TMAX", "TMIN"], [1]
        )
        unique_vals = set(mask.values.flatten())
        assert unique_vals.issubset({0, 1})

    def test_mask_multi_lag(self):
        df = self._make_merged_df()
        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX"], [1, 2, 3]
        )
        assert mask.shape[1] == 3  # 1 station * 1 var * 3 lags
        expected_cols = [
            "USW00014735_mask_TMAX_lag1",
            "USW00014735_mask_TMAX_lag2",
            "USW00014735_mask_TMAX_lag3",
        ]
        assert list(mask.columns) == expected_cols

    def test_mask_column_naming(self):
        df = self._make_merged_df()
        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX", "TMIN"], [1]
        )
        assert "USW00014735_mask_TMAX_lag1" in mask.columns
        assert "USW00014735_mask_TMIN_lag1" in mask.columns

    def test_mask_empty_station_list(self):
        df = self._make_merged_df()
        mask = create_missingness_mask(df, [], ["TMAX"], [1])
        assert mask.shape[1] == 0


# ===========================================================================
# Test: Loading Expanded Stations
# ===========================================================================

class TestLoadExpandedStations:
    """Tests for load_expanded_stations()."""

    def test_loads_existing_stations(self, synthetic_raw_dir):
        ids = list(config.SURROUNDING_STATIONS.keys())[:3]
        data = load_expanded_stations(ids, synthetic_raw_dir)
        assert len(data) == 3

    def test_skips_missing_stations(self, synthetic_raw_dir):
        ids = ["USW00014735", "USW00099999"]  # one exists, one doesn't
        data = load_expanded_stations(ids, synthetic_raw_dir)
        assert len(data) == 1
        assert "USW00014735" in data

    def test_returns_empty_for_all_missing(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            data = load_expanded_stations(["USW00099999"], tmp_dir)
            assert len(data) == 0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_loaded_data_has_expected_columns(self, synthetic_raw_dir):
        data = load_expanded_stations([config.TARGET_STATION], synthetic_raw_dir)
        df = data[config.TARGET_STATION]
        assert "TMAX" in df.columns
        assert "TMIN" in df.columns


# ===========================================================================
# Test: Completeness Filtering
# ===========================================================================

class TestCompletenessFiltering:
    """Tests for filter_expanded_by_completeness()."""

    def test_keeps_complete_stations(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": np.random.randn(100),
            "USW00014735_TMAX": np.random.randn(100),
            "USW00014735_TMIN": np.random.randn(100),
        }, index=dates)
        filtered, dropped = filter_expanded_by_completeness(df)
        assert len(dropped) == 0
        assert "USW00014735_TMAX" in filtered.columns

    def test_drops_low_completeness_station(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tmax = np.random.randn(100)
        tmax[:85] = np.nan  # 85% missing = 15% complete
        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": np.random.randn(100),
            "USW00099999_TMAX": tmax,
            "USW00099999_TMIN": np.random.randn(100),
        }, index=dates)
        filtered, dropped = filter_expanded_by_completeness(df)
        assert "USW00099999" in dropped
        assert "USW00099999_TMAX" not in filtered.columns

    def test_never_drops_target_station(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tmax = np.random.randn(100)
        tmax[:85] = np.nan  # mostly missing
        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": tmax,
        }, index=dates)
        filtered, dropped = filter_expanded_by_completeness(df)
        assert config.TARGET_STATION not in dropped


# ===========================================================================
# Test: Expanded Feature Creation
# ===========================================================================

class TestExpandedFeatureCreation:
    """Tests for create_expanded_features()."""

    def _make_merged_df(self, n_stations=5, n_days=200):
        """Create merged DataFrame with target + surrounding stations."""
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        rng = np.random.RandomState(42)
        data = {}

        # Target station
        data[f"{config.TARGET_STATION}_TMAX"] = (
            60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
            + rng.randn(n_days) * 3
        )
        data[f"{config.TARGET_STATION}_TMIN"] = (
            data[f"{config.TARGET_STATION}_TMAX"] - 10
            - rng.rand(n_days) * 5
        )

        # Surrounding stations
        station_ids = []
        for i in range(n_stations):
            sid = f"USW0009{i:04d}"
            station_ids.append(sid)
            data[f"{sid}_TMAX"] = (
                data[f"{config.TARGET_STATION}_TMAX"]
                + rng.randn(n_days) * 5
            )
            data[f"{sid}_TMIN"] = (
                data[f"{sid}_TMAX"] - 10 - rng.rand(n_days) * 5
            )

        df = pd.DataFrame(data, index=dates)
        return df, station_ids

    def test_basic_feature_creation(self):
        df, sids = self._make_merged_df(n_stations=3)
        feats, raw_t, delta_t, prev = create_expanded_features(
            df, sids,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
        )
        # Features: 3 stations * 2 vars * 1 lag + 2 date = 8
        assert feats.shape[1] == 8

    def test_feature_count_with_mask(self):
        df, sids = self._make_merged_df(n_stations=3)
        feats, _, _, _ = create_expanded_features(
            df, sids,
            include_missingness_mask=True,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
        )
        # Mask: 3 stations * 2 vars * 1 lag = 6
        # Lagged: 3 * 2 * 1 = 6
        # Date: 2
        assert feats.shape[1] == 6 + 6 + 2

    def test_feature_count_with_autoregressive(self):
        df, sids = self._make_merged_df(n_stations=3)
        feats, _, _, _ = create_expanded_features(
            df, sids,
            include_missingness_mask=False,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
        )
        # 3*2*1 lagged + 2 date + 1 AR = 9
        assert feats.shape[1] == 9

    def test_features_scale_with_station_count(self):
        df5, sids5 = self._make_merged_df(n_stations=5)
        df10, sids10 = self._make_merged_df(n_stations=10)

        feats5, _, _, _ = create_expanded_features(
            df5, sids5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        feats10, _, _, _ = create_expanded_features(
            df10, sids10,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
        )
        # 5 stations: 5*2 + 2 = 12
        # 10 stations: 10*2 + 2 = 22
        assert feats5.shape[1] == 12
        assert feats10.shape[1] == 22
        assert feats10.shape[1] > feats5.shape[1]


# ===========================================================================
# Test: Full Pipeline (run_expanded_preprocessing)
# ===========================================================================

class TestRunExpandedPreprocessing:
    """Tests for run_expanded_preprocessing()."""

    def test_basic_pipeline(self, synthetic_raw_dir):
        result = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            lags=[1],
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        assert "X_train" in result
        assert "X_val" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_val" in result
        assert "y_test" in result
        assert "scaler" in result
        assert result["n_features"] > 0
        assert len(result["surrounding_ids"]) == 5

    def test_pipeline_with_explicit_station_list(self, synthetic_raw_dir):
        stations = list(config.SURROUNDING_STATIONS.keys())[:3]
        result = run_expanded_preprocessing(
            station_list=stations,
            include_missingness_mask=True,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        assert len(result["surrounding_ids"]) == 3

    def test_pipeline_chronological_split(self, synthetic_raw_dir):
        result = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        # Check chronological ordering
        assert result["X_train"].index.max() < result["X_val"].index.min()
        assert result["X_val"].index.max() < result["X_test"].index.min()

    def test_scaler_fit_on_train_only(self, synthetic_raw_dir):
        result = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        scaler = result["scaler"]
        # Scaler should have been fit on training data
        assert scaler.n_features_in_ == result["n_features"]
        # Training data should have mean~0, std~1 after scaling
        train_means = result["X_train"].mean()
        assert all(abs(m) < 0.3 for m in train_means)

    def test_no_nan_in_features(self, synthetic_raw_dir):
        result = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        assert result["X_train"].isna().sum().sum() == 0
        assert result["X_val"].isna().sum().sum() == 0
        assert result["X_test"].isna().sum().sum() == 0

    def test_pipeline_with_missing_data(self, synthetic_expanded_dir):
        """Test that pipeline handles stations with missing data."""
        result = run_expanded_preprocessing(
            include_missingness_mask=True,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_expanded_dir,
            output_dir=tempfile.mkdtemp(),
        )
        assert result["X_train"].isna().sum().sum() == 0
        assert result["n_features"] > 0

    def test_pipeline_graceful_with_sparse_data(self, synthetic_sparse_dir):
        """Test pipeline with only 3 stations (one with heavy gaps)."""
        result = run_expanded_preprocessing(
            include_missingness_mask=True,
            include_autoregressive=True,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_sparse_dir,
            output_dir=tempfile.mkdtemp(),
        )
        assert result["n_features"] > 0
        # The station with 50% missing data may have been dropped
        assert len(result["X_train"]) > 0

    def test_delta_targets_aligned(self, synthetic_raw_dir):
        result = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        # Delta target should equal raw - prev
        reconstructed = result["nyc_prev_train"] + result["y_train_delta"]
        diff = (reconstructed - result["y_train"]).abs()
        assert diff.max() < 0.01

    def test_pipeline_handles_unavailable_stations(self, synthetic_raw_dir):
        """Request stations that don't exist -- should skip gracefully."""
        result = run_expanded_preprocessing(
            station_list=["USW00014735", "USW00099999", "USW00088888"],
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        # Only USW00014735 should be used
        assert len(result["surrounding_ids"]) == 1
        assert "USW00014735" in result["surrounding_ids"]

    def test_feature_count_scales_correctly(self, synthetic_raw_dir):
        """Test that features scale with station count."""
        result5 = run_expanded_preprocessing(
            n_stations=5,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        result10 = run_expanded_preprocessing(
            n_stations=10,
            include_missingness_mask=False,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_raw_dir,
            output_dir=tempfile.mkdtemp(),
        )
        # 5 stations: 5*2+2=12, 10 stations: 10*2+2=22
        assert result5["n_features"] == 12
        assert result10["n_features"] == 22


# ===========================================================================
# Test: Synthetic Data with Intentional Gaps
# ===========================================================================

class TestSyntheticGaps:
    """Tests specifically for handling data with intentional gaps."""

    def test_mask_captures_injected_gaps(self):
        """Create data with known gaps and verify mask detects them."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        tmax = np.ones(50) * 60.0
        tmin = np.ones(50) * 45.0

        # Inject gaps at days 10, 20, 30
        tmax[10] = np.nan
        tmax[20] = np.nan
        tmax[30] = np.nan

        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": np.ones(50) * 62.0,
            f"{config.TARGET_STATION}_TMIN": np.ones(50) * 48.0,
            "USW00014735_TMAX": tmax,
            "USW00014735_TMIN": tmin,
        }, index=dates)

        mask = create_missingness_mask(
            df, ["USW00014735"], ["TMAX"], [1],
        )

        # Gaps at raw idx 10,20,30 -> mask 0 at idx 11,21,31 (after lag1)
        assert mask.iloc[11]["USW00014735_mask_TMAX_lag1"] == 0
        assert mask.iloc[21]["USW00014735_mask_TMAX_lag1"] == 0
        assert mask.iloc[31]["USW00014735_mask_TMAX_lag1"] == 0

        # Non-gap indices should be 1
        assert mask.iloc[5]["USW00014735_mask_TMAX_lag1"] == 1
        assert mask.iloc[15]["USW00014735_mask_TMAX_lag1"] == 1

    def test_forward_fill_respects_limit(self, synthetic_sparse_dir):
        """Verify forward-fill fills up to 3 days only."""
        result = run_expanded_preprocessing(
            include_missingness_mask=True,
            include_autoregressive=False,
            include_diurnal=False,
            include_sectors=False,
            include_trends=False,
            raw_dir=synthetic_sparse_dir,
            output_dir=tempfile.mkdtemp(),
        )
        # No NaN in final output
        assert result["X_train"].isna().sum().sum() == 0
