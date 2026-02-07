"""
Tests for the data preprocessing module.

Validates:
  - Lag shift correctness (features at t use data from t-1)
  - Cyclical date encoding (sin/cos)
  - Chronological train/val/test splitting (no future leakage)
  - StandardScaler fit on training data only
  - Missing data handling (forward-fill, column mean imputation)
  - Target column isolation
  - Data completeness filtering
"""

import os
import sys
import tempfile
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    merge_stations,
    compute_completeness,
    filter_stations_by_completeness,
    create_target_and_features,
    add_cyclical_date_features,
    handle_missing_data,
    chronological_split,
    fit_and_apply_scaler,
    fill_remaining_nans_with_train_means,
    save_processed_data,
    load_station_csv,
)
import config


# ===========================================================================
# Fixtures: synthetic station data
# ===========================================================================

def make_date_range(start="2020-01-01", periods=100):
    """Create a DatetimeIndex for testing."""
    return pd.date_range(start=start, periods=periods, freq="D")


def make_station_df(dates, tmax_values=None, tmin_values=None):
    """Create a synthetic station DataFrame.

    Parameters
    ----------
    dates : pd.DatetimeIndex
    tmax_values : array-like, optional
        If None, generates random values around 70 F.
    tmin_values : array-like, optional
        If None, generates random values around 50 F.
    """
    n = len(dates)
    rng = np.random.RandomState(42)

    if tmax_values is None:
        tmax_values = 60 + 20 * np.sin(
            2 * np.pi * np.arange(n) / 365.25
        ) + rng.normal(0, 3, n)
    if tmin_values is None:
        tmin_values = tmax_values - 15 + rng.normal(0, 2, n)

    return pd.DataFrame(
        {"TMAX": tmax_values, "TMIN": tmin_values},
        index=dates,
    )


@pytest.fixture
def sample_station_data():
    """Create a dict of synthetic station DataFrames for testing."""
    dates = make_date_range("2020-01-01", 365)

    # Target station
    target = make_station_df(dates)
    # Two surrounding stations
    station_a = make_station_df(dates)
    station_b = make_station_df(dates)

    return {
        config.TARGET_STATION: target,
        "USW00014735": station_a,
        "USW00014740": station_b,
    }


@pytest.fixture
def merged_df(sample_station_data):
    """Create a merged DataFrame from sample station data."""
    return merge_stations(sample_station_data)


# ===========================================================================
# Merge Tests
# ===========================================================================

class TestMergeStations:
    """Test merging station DataFrames."""

    def test_column_naming(self, sample_station_data):
        """Columns should be {station_id}_{variable}."""
        merged = merge_stations(sample_station_data)

        expected_cols = {
            f"{config.TARGET_STATION}_TMAX",
            f"{config.TARGET_STATION}_TMIN",
            "USW00014735_TMAX",
            "USW00014735_TMIN",
            "USW00014740_TMAX",
            "USW00014740_TMIN",
        }
        assert set(merged.columns) == expected_cols

    def test_row_count(self, sample_station_data):
        """All dates should be preserved in the merge."""
        merged = merge_stations(sample_station_data)
        assert len(merged) == 365

    def test_sorted_index(self, sample_station_data):
        """Output should be sorted by date."""
        merged = merge_stations(sample_station_data)
        assert merged.index.is_monotonic_increasing

    def test_empty_raises(self):
        """Merging empty dict should raise ValueError."""
        with pytest.raises(ValueError, match="No station data"):
            merge_stations({})


# ===========================================================================
# Lag Shift Tests
# ===========================================================================

class TestLagShift:
    """Verify that features at day t use surrounding-station data from day t-1."""

    def test_lag_shift_correctness(self, sample_station_data):
        """Feature value at date t should equal the source value at date t-1."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        # Pick a specific surrounding station column
        src_col = "USW00014735_TMAX"
        lag_col = "USW00014735_TMAX_lag1"

        assert lag_col in features.columns

        # For dates where both exist, verify the shift
        for dt in features.index[1:10]:  # Check a few dates
            # The feature at date dt should equal the source at dt - 1 day
            prev_date = dt - pd.Timedelta(days=1)
            if prev_date in merged.index:
                expected = merged.loc[prev_date, src_col]
                actual = features.loc[dt, lag_col]
                if pd.notna(expected) and pd.notna(actual):
                    assert actual == expected, (
                        f"At {dt}: feature={actual}, expected source at "
                        f"{prev_date}={expected}"
                    )

    def test_first_row_dropped(self, sample_station_data):
        """The very first date should be dropped (no t-1 data available)."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        # First date in merged should NOT be in features
        first_date = merged.index.min()
        assert first_date not in features.index

    def test_target_not_lagged(self, sample_station_data):
        """Target column (NYC TMAX at t) should NOT be lagged."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        # Target should match the merged data's target column directly
        target_col = f"{config.TARGET_STATION}_TMAX"
        for dt in target.index[:10]:
            assert target.loc[dt] == merged.loc[dt, target_col]

    def test_target_station_columns_excluded_from_features(self, sample_station_data):
        """Feature columns should NOT include the target station's data."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        target_prefix = f"{config.TARGET_STATION}_"
        for col in features.columns:
            assert not col.startswith(target_prefix), (
                f"Target station column {col} found in features"
            )


# ===========================================================================
# Cyclical Date Encoding Tests
# ===========================================================================

class TestCyclicalDateEncoding:
    """Test sin/cos day-of-year encoding."""

    def test_jan1_values(self):
        """Jan 1 (day 1): sin ~ 0.017, cos ~ 1.0."""
        dates = pd.DatetimeIndex(["2020-01-01"])
        features = pd.DataFrame({"x": [1.0]}, index=dates)
        result = add_cyclical_date_features(features)

        sin_val = result.loc["2020-01-01", "sin_day"]
        cos_val = result.loc["2020-01-01", "cos_day"]

        # Day 1 of 365.25: sin(2*pi*1/365.25) ~ 0.0172
        expected_sin = np.sin(2 * np.pi * 1 / 365.25)
        expected_cos = np.cos(2 * np.pi * 1 / 365.25)

        assert abs(sin_val - expected_sin) < 0.001
        assert abs(cos_val - expected_cos) < 0.001

    def test_jul1_values(self):
        """Jul 1 (~ day 182): sin ~ close to 0, cos ~ -1."""
        dates = pd.DatetimeIndex(["2020-07-01"])
        features = pd.DataFrame({"x": [1.0]}, index=dates)
        result = add_cyclical_date_features(features)

        sin_val = result.loc["2020-07-01", "sin_day"]
        cos_val = result.loc["2020-07-01", "cos_day"]

        # Day 183 of 365.25: roughly pi, so sin ~ small, cos ~ -1
        day_of_year = 183  # Jul 1 in a leap year
        expected_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        expected_cos = np.cos(2 * np.pi * day_of_year / 365.25)

        assert abs(sin_val - expected_sin) < 0.001
        assert abs(cos_val - expected_cos) < 0.001

    def test_symmetry_equinox(self):
        """Spring and fall equinox dates should have opposite sin values."""
        dates = pd.DatetimeIndex(["2020-03-20", "2020-09-22"])
        features = pd.DataFrame({"x": [1.0, 1.0]}, index=dates)
        result = add_cyclical_date_features(features)

        spring_sin = result.iloc[0]["sin_day"]
        fall_sin = result.iloc[1]["sin_day"]

        # They should be roughly opposite in sign
        # (not exactly, but approximately)
        assert spring_sin > 0
        assert fall_sin < 0

    def test_full_year_range(self):
        """Sin and cos values should be in [-1, 1] for all days."""
        dates = pd.date_range("2020-01-01", periods=366, freq="D")
        features = pd.DataFrame({"x": np.ones(366)}, index=dates)
        result = add_cyclical_date_features(features)

        assert result["sin_day"].min() >= -1.0
        assert result["sin_day"].max() <= 1.0
        assert result["cos_day"].min() >= -1.0
        assert result["cos_day"].max() <= 1.0

    def test_columns_added(self):
        """sin_day and cos_day should be added as new columns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        features = pd.DataFrame({"x": np.ones(10)}, index=dates)
        result = add_cyclical_date_features(features)

        assert "sin_day" in result.columns
        assert "cos_day" in result.columns
        assert "x" in result.columns  # original column preserved


# ===========================================================================
# Chronological Split Tests
# ===========================================================================

class TestChronologicalSplit:
    """Test that the train/val/test split is strictly chronological."""

    def test_no_overlap(self):
        """Train, val, and test date ranges must not overlap."""
        dates = make_date_range("2020-01-01", 1000)
        features = pd.DataFrame({"x": np.arange(1000)}, index=dates)
        target = pd.Series(np.arange(1000), index=dates, name="y")

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target, train_ratio=0.7, val_ratio=0.15
        )

        # No date overlap
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()

    def test_correct_proportions(self):
        """Split sizes should approximately match the given ratios."""
        n = 1000
        dates = make_date_range("2020-01-01", n)
        features = pd.DataFrame({"x": np.arange(n)}, index=dates)
        target = pd.Series(np.arange(n), index=dates, name="y")

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target, train_ratio=0.7, val_ratio=0.15
        )

        assert len(X_train) == 700
        assert len(X_val) == 150
        assert len(X_test) == 150

    def test_no_shuffling(self):
        """Values in each split should be in the original order."""
        n = 100
        dates = make_date_range("2020-01-01", n)
        features = pd.DataFrame({"x": np.arange(n)}, index=dates)
        target = pd.Series(np.arange(n), index=dates, name="y")

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target, train_ratio=0.7, val_ratio=0.15
        )

        # Train should be the first 70 values
        np.testing.assert_array_equal(X_train["x"].values, np.arange(70))
        # Val should be 70-84
        np.testing.assert_array_equal(X_val["x"].values, np.arange(70, 85))
        # Test should be 85-99
        np.testing.assert_array_equal(X_test["x"].values, np.arange(85, 100))

    def test_temporal_ordering(self):
        """All train dates < all val dates < all test dates."""
        n = 365
        dates = make_date_range("2020-01-01", n)
        features = pd.DataFrame({"x": np.arange(n)}, index=dates)
        target = pd.Series(np.arange(n), index=dates, name="y")

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target
        )

        # All train dates before all val dates
        assert all(X_train.index < X_val.index.min())
        # All val dates before all test dates
        assert all(X_val.index < X_test.index.min())

    def test_target_aligned_with_features(self):
        """Target indices must match feature indices in each split."""
        n = 200
        dates = make_date_range("2020-01-01", n)
        features = pd.DataFrame({"x": np.arange(n)}, index=dates)
        target = pd.Series(np.arange(n) * 10, index=dates, name="y")

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target
        )

        pd.testing.assert_index_equal(X_train.index, y_train.index)
        pd.testing.assert_index_equal(X_val.index, y_val.index)
        pd.testing.assert_index_equal(X_test.index, y_test.index)


# ===========================================================================
# Scaler Tests
# ===========================================================================

class TestScaler:
    """Test that StandardScaler is fit on training data only."""

    def test_scaler_fit_on_train_only(self):
        """Scaler mean/std should match training set statistics."""
        n_train, n_val, n_test = 100, 20, 20

        rng = np.random.RandomState(42)
        train_data = rng.normal(50, 10, (n_train, 3))
        val_data = rng.normal(80, 5, (n_val, 3))  # Different distribution
        test_data = rng.normal(20, 15, (n_test, 3))  # Different distribution

        cols = ["a", "b", "c"]
        X_train = pd.DataFrame(train_data, columns=cols)
        X_val = pd.DataFrame(val_data, columns=cols)
        X_test = pd.DataFrame(test_data, columns=cols)

        X_tr_s, X_val_s, X_te_s, scaler = fit_and_apply_scaler(
            X_train, X_val, X_test
        )

        # Scaler's mean should match training data mean
        np.testing.assert_array_almost_equal(
            scaler.mean_, X_train.mean().values, decimal=10
        )

        # Training scaled data should have mean ~ 0, std ~ 1
        assert abs(X_tr_s.mean().mean()) < 0.1
        assert abs(X_tr_s.std().mean() - 1.0) < 0.1

        # Val/test should NOT have mean=0 (they come from different distributions)
        # This proves the scaler was NOT fit on val/test
        assert abs(X_val_s.mean().mean()) > 1.0  # far from 0

    def test_scaler_preserves_index(self):
        """Scaled DataFrames should retain their original index."""
        dates_train = make_date_range("2020-01-01", 50)
        dates_val = make_date_range("2020-03-01", 10)
        dates_test = make_date_range("2020-04-01", 10)

        X_train = pd.DataFrame({"x": np.arange(50, dtype=float)}, index=dates_train)
        X_val = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=dates_val)
        X_test = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=dates_test)

        X_tr_s, X_val_s, X_te_s, scaler = fit_and_apply_scaler(
            X_train, X_val, X_test
        )

        pd.testing.assert_index_equal(X_tr_s.index, dates_train)
        pd.testing.assert_index_equal(X_val_s.index, dates_val)
        pd.testing.assert_index_equal(X_te_s.index, dates_test)

    def test_scaler_preserves_columns(self):
        """Column names should be preserved after scaling."""
        cols = ["feature_a", "feature_b"]
        X_train = pd.DataFrame(np.ones((10, 2)), columns=cols)
        X_val = pd.DataFrame(np.ones((5, 2)), columns=cols)
        X_test = pd.DataFrame(np.ones((5, 2)), columns=cols)

        X_tr_s, _, _, _ = fit_and_apply_scaler(X_train, X_val, X_test)
        assert list(X_tr_s.columns) == cols


# ===========================================================================
# Missing Data Tests
# ===========================================================================

class TestMissingData:
    """Test missing data handling strategies."""

    def test_target_nan_rows_dropped(self):
        """Rows where the target is NaN should be dropped."""
        dates = make_date_range("2020-01-01", 10)
        features = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=dates)
        target = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
                           index=dates, name="y")

        features_out, target_out = handle_missing_data(features, target)

        assert len(features_out) == 8
        assert target_out.notna().all()

    def test_forward_fill_limit(self):
        """Forward-fill should fill gaps up to max_fill_days consecutive NaNs."""
        dates = make_date_range("2020-01-01", 10)
        target = pd.Series(np.ones(10), index=dates, name="y")

        # Column with a gap of 2 (should be filled with limit=3)
        col_values = [1.0, np.nan, np.nan, 4.0, 5.0,
                      6.0, np.nan, np.nan, np.nan, 10.0]
        features = pd.DataFrame({"x": col_values}, index=dates)

        features_out, target_out = handle_missing_data(
            features, target, max_fill_days=3
        )

        # Gap of 2 (indices 1,2) should be filled
        assert features_out.loc[dates[1], "x"] == 1.0
        assert features_out.loc[dates[2], "x"] == 1.0

        # Gap of 3 (indices 6,7,8) should also be filled (limit=3)
        assert features_out.loc[dates[6], "x"] == 6.0
        assert features_out.loc[dates[7], "x"] == 6.0
        assert features_out.loc[dates[8], "x"] == 6.0

    def test_forward_fill_respects_limit(self):
        """Gaps longer than max_fill_days should NOT be fully filled."""
        dates = make_date_range("2020-01-01", 10)
        target = pd.Series(np.ones(10), index=dates, name="y")

        # Gap of 5 with limit=2: only first 2 should be filled
        col_values = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
                      7.0, 8.0, 9.0, 10.0]
        features = pd.DataFrame({"x": col_values}, index=dates)

        features_out, _ = handle_missing_data(features, target, max_fill_days=2)

        # First 2 NaNs filled, remaining 3 still NaN
        assert features_out.loc[dates[1], "x"] == 1.0
        assert features_out.loc[dates[2], "x"] == 1.0
        assert pd.isna(features_out.loc[dates[3], "x"])
        assert pd.isna(features_out.loc[dates[4], "x"])
        assert pd.isna(features_out.loc[dates[5], "x"])

    def test_train_mean_imputation(self):
        """Remaining NaNs should be filled with training-set column means."""
        train = pd.DataFrame({"a": [10.0, 20.0, np.nan, 40.0],
                               "b": [1.0, 2.0, 3.0, 4.0]})
        val = pd.DataFrame({"a": [np.nan, 50.0],
                             "b": [5.0, np.nan]})
        test = pd.DataFrame({"a": [60.0], "b": [np.nan]})

        train_out, val_out, test_out, means = fill_remaining_nans_with_train_means(
            train, val, test
        )

        # Training mean of "a": (10+20+40)/3 = 23.333...
        expected_mean_a = np.nanmean([10.0, 20.0, 40.0])
        assert abs(train_out.loc[2, "a"] - expected_mean_a) < 0.01
        assert abs(val_out.loc[0, "a"] - expected_mean_a) < 0.01

        # Training mean of "b": (1+2+3+4)/4 = 2.5
        expected_mean_b = 2.5
        assert abs(val_out.loc[1, "b"] - expected_mean_b) < 0.01
        assert abs(test_out.loc[0, "b"] - expected_mean_b) < 0.01


# ===========================================================================
# Data Completeness Tests
# ===========================================================================

class TestCompleteness:
    """Test station completeness checking and filtering."""

    def test_completeness_computation(self):
        """Completeness should be correctly calculated."""
        dates = make_date_range("2020-01-01", 100)
        tmax_vals = np.ones(100)
        tmax_vals[80:] = np.nan  # 80% complete

        df = pd.DataFrame({
            "STATION_A_TMAX": tmax_vals,
            "STATION_A_TMIN": np.ones(100),
        }, index=dates)

        report = compute_completeness(df, ["STATION_A"], ["TMAX", "TMIN"])

        tmax_row = report[(report["station_id"] == "STATION_A") &
                          (report["variable"] == "TMAX")]
        assert abs(tmax_row.iloc[0]["completeness_pct"] - 0.80) < 0.01

        tmin_row = report[(report["station_id"] == "STATION_A") &
                          (report["variable"] == "TMIN")]
        assert abs(tmin_row.iloc[0]["completeness_pct"] - 1.0) < 0.01

    def test_filter_drops_low_completeness(self):
        """Stations below the threshold should be dropped."""
        dates = make_date_range("2020-01-01", 100)

        good_tmax = np.ones(100)  # 100% complete
        bad_tmax = np.ones(100)
        bad_tmax[15:] = np.nan  # 15% complete

        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": good_tmax,
            f"{config.TARGET_STATION}_TMIN": good_tmax,
            "BAD_STATION_TMAX": bad_tmax,
            "BAD_STATION_TMIN": bad_tmax,
        }, index=dates)

        filtered, dropped = filter_stations_by_completeness(df, min_completeness=0.5)

        assert "BAD_STATION" in dropped
        assert "BAD_STATION_TMAX" not in filtered.columns
        assert f"{config.TARGET_STATION}_TMAX" in filtered.columns

    def test_target_station_not_dropped(self):
        """Target station should never be dropped even if below threshold."""
        dates = make_date_range("2020-01-01", 100)

        low_completeness = np.ones(100)
        low_completeness[50:] = np.nan  # 50% complete

        df = pd.DataFrame({
            f"{config.TARGET_STATION}_TMAX": low_completeness,
            f"{config.TARGET_STATION}_TMIN": low_completeness,
        }, index=dates)

        filtered, dropped = filter_stations_by_completeness(df, min_completeness=0.9)

        assert config.TARGET_STATION not in dropped
        assert f"{config.TARGET_STATION}_TMAX" in filtered.columns


# ===========================================================================
# Target Isolation Tests
# ===========================================================================

class TestTargetIsolation:
    """Test that the target column is correctly extracted."""

    def test_target_is_nyc_tmax(self, sample_station_data):
        """Target should be Central Park's TMAX."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        assert target.name == "NYC_TMAX"
        assert len(target) > 0
        assert target.notna().all()

    def test_target_not_in_features(self, sample_station_data):
        """The target variable should not appear in the feature matrix."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)

        for col in features.columns:
            assert "NYC_TMAX" not in col
            assert config.TARGET_STATION not in col


# ===========================================================================
# Save/Load Tests
# ===========================================================================

class TestSaveLoad:
    """Test saving and loading processed data."""

    def test_save_and_reload(self):
        """Saved files should be loadable and match original data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dates_tr = make_date_range("2020-01-01", 50)
            dates_val = make_date_range("2020-03-01", 10)
            dates_te = make_date_range("2020-04-01", 10)

            X_train = pd.DataFrame({"a": np.arange(50, dtype=float)}, index=dates_tr)
            X_val = pd.DataFrame({"a": np.arange(10, dtype=float)}, index=dates_val)
            X_test = pd.DataFrame({"a": np.arange(10, dtype=float)}, index=dates_te)
            y_train = pd.Series(np.arange(50, dtype=float), index=dates_tr, name="NYC_TMAX")
            y_val = pd.Series(np.arange(10, dtype=float), index=dates_val, name="NYC_TMAX")
            y_test = pd.Series(np.arange(10, dtype=float), index=dates_te, name="NYC_TMAX")

            scaler = StandardScaler()
            scaler.fit(X_train)

            save_processed_data(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                scaler, output_dir=tmpdir
            )

            # Check all files exist
            assert os.path.exists(os.path.join(tmpdir, "features_train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "features_val.csv"))
            assert os.path.exists(os.path.join(tmpdir, "features_test.csv"))
            assert os.path.exists(os.path.join(tmpdir, "target_train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "target_val.csv"))
            assert os.path.exists(os.path.join(tmpdir, "target_test.csv"))
            assert os.path.exists(os.path.join(tmpdir, "scaler.pkl"))

            # Reload and verify
            loaded_train = pd.read_csv(
                os.path.join(tmpdir, "features_train.csv"),
                index_col=0, parse_dates=True
            )
            assert loaded_train.shape == X_train.shape

            with open(os.path.join(tmpdir, "scaler.pkl"), "rb") as f:
                loaded_scaler = pickle.load(f)
            np.testing.assert_array_almost_equal(
                loaded_scaler.mean_, scaler.mean_
            )


# ===========================================================================
# Integration Test
# ===========================================================================

class TestIntegration:
    """End-to-end integration test of the preprocessing pipeline."""

    def test_full_pipeline(self, sample_station_data):
        """Run the full pipeline on synthetic data and verify outputs."""
        merged = merge_stations(sample_station_data)
        features, target = create_target_and_features(merged)
        features = add_cyclical_date_features(features)
        features, target = handle_missing_data(features, target)

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
            features, target
        )

        X_train, X_val, X_test, _ = fill_remaining_nans_with_train_means(
            X_train, X_val, X_test
        )

        X_tr_s, X_val_s, X_te_s, scaler = fit_and_apply_scaler(
            X_train, X_val, X_test
        )

        # Verify no NaNs in final output
        assert X_tr_s.notna().all().all()
        assert X_val_s.notna().all().all()
        assert X_te_s.notna().all().all()
        assert y_train.notna().all()
        assert y_val.notna().all()
        assert y_test.notna().all()

        # Verify chronological ordering
        assert X_tr_s.index.max() < X_val_s.index.min()
        assert X_val_s.index.max() < X_te_s.index.min()

        # Verify sin_day and cos_day are present
        assert "sin_day" in X_tr_s.columns
        assert "cos_day" in X_tr_s.columns

        # Verify no target station columns in features
        for col in X_tr_s.columns:
            assert config.TARGET_STATION not in col

        # Verify total row count is reasonable
        total = len(X_tr_s) + len(X_val_s) + len(X_te_s)
        assert total == len(features)
