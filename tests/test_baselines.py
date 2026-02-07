"""
Tests for the baseline models module.

Validates:
  - PersistenceModel: lag-1 prediction, boundary handling, fit/predict
  - ClimatologyModel: day-of-year averages, fallback for unseen DOY
  - LinearRegressionModel: fit/predict, coefficient recovery, shape checks
  - RidgeRegressionModel: alpha effects, collinear handling, shrinkage
  - run_all_baselines: result structure, all models present, metrics
  - compute_metrics: known errors, NaN handling, edge cases
  - Edge cases: single-row, empty, NaN, date gaps
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines import (
    PersistenceModel,
    ClimatologyModel,
    LinearRegressionModel,
    RidgeRegressionModel,
    run_all_baselines,
    compute_metrics,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def make_date_index(start="2020-01-01", periods=100):
    """Create a DatetimeIndex for testing."""
    return pd.date_range(start=start, periods=periods, freq="D")


@pytest.fixture
def synthetic_data():
    """Create synthetic train/val/test data with DatetimeIndex.

    300 days split chronologically: 210 train / 45 val / 45 test.
    Features are random; target follows a seasonal sinusoidal pattern
    plus Gaussian noise.
    """
    rng = np.random.RandomState(42)

    dates = make_date_index("2020-01-01", 300)
    n = len(dates)
    n_features = 5

    X = pd.DataFrame(
        rng.randn(n, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    doy = dates.dayofyear
    y = pd.Series(
        50 + 20 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 3, n),
        index=dates,
        name="NYC_TMAX",
    )

    train_end = 210
    val_end = 255

    return {
        "X_train": X.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_train": y.iloc[:train_end],
        "y_val": y.iloc[train_end:val_end],
        "y_test": y.iloc[val_end:],
    }


@pytest.fixture
def simple_persistence_data():
    """Create minimal data for persistence testing with known values.

    Train: 5 days [10, 20, 30, 40, 50]
    Val:   3 days [60, 70, 80]
    Test:  3 days [90, 100, 110]
    """
    dates_train = pd.date_range("2020-01-01", periods=5, freq="D")
    dates_val = pd.date_range("2020-01-06", periods=3, freq="D")
    dates_test = pd.date_range("2020-01-09", periods=3, freq="D")

    y_train = pd.Series([10, 20, 30, 40, 50], index=dates_train, name="NYC_TMAX")
    y_val = pd.Series([60, 70, 80], index=dates_val, name="NYC_TMAX")
    y_test = pd.Series([90, 100, 110], index=dates_test, name="NYC_TMAX")

    X_train = pd.DataFrame({"f1": range(5)}, index=dates_train)
    X_val = pd.DataFrame({"f1": range(3)}, index=dates_val)
    X_test = pd.DataFrame({"f1": range(3)}, index=dates_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


@pytest.fixture
def linear_data():
    """Create data with a perfect linear relationship.

    y = 2*x0 + 3*x1 - 1*x2 + 50
    OLS should recover these coefficients exactly.
    """
    rng = np.random.RandomState(42)
    dates = make_date_index("2020-01-01", 200)
    n = len(dates)
    n_features = 3

    X = pd.DataFrame(
        rng.randn(n, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    true_coefs = np.array([2.0, 3.0, -1.0])
    y = pd.Series(
        X.values @ true_coefs + 50.0,
        index=dates,
        name="NYC_TMAX",
    )

    train_end = 140
    val_end = 170

    return {
        "X_train": X.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_train": y.iloc[:train_end],
        "y_val": y.iloc[train_end:val_end],
        "y_test": y.iloc[val_end:],
        "true_coefs": true_coefs,
    }


# ===========================================================================
# PersistenceModel Tests
# ===========================================================================

class TestPersistenceModel:
    """Tests for the PersistenceModel baseline."""

    def test_name(self):
        """Verify model name is 'Persistence'."""
        model = PersistenceModel()
        assert model.name == "Persistence"

    def test_fit_stores_training_data(self, synthetic_data):
        """Verify fit() stores the training target series."""
        d = synthetic_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        assert hasattr(model, "_y_train")
        assert len(model._y_train) == len(d["y_train"])
        pd.testing.assert_series_equal(model._y_train, d["y_train"])

    def test_predict_without_fit_raises(self, synthetic_data):
        """Verify predicting before fitting raises RuntimeError."""
        model = PersistenceModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(synthetic_data["X_val"])

    def test_in_sample_predictions(self, simple_persistence_data):
        """Test in-sample predictions: each equals the previous day's actual.

        For y_train = [10, 20, 30, 40, 50]:
          - prediction[0] = NaN (no prior value)
          - prediction[1] = 10
          - prediction[2] = 20
          - prediction[3] = 30
          - prediction[4] = 40
        """
        d = simple_persistence_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        preds = model.predict(d["X_train"])

        assert np.isnan(preds[0]), "First in-sample prediction should be NaN"
        np.testing.assert_array_equal(preds[1:], [10, 20, 30, 40])

    def test_val_predictions_with_boundary(self, simple_persistence_data):
        """Test validation predictions using concatenated y_prev.

        First val prediction should use last training actual (50).
        """
        d = simple_persistence_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        y_prev = pd.concat([d["y_train"], d["y_val"]])
        preds = model.predict(d["X_val"], y_prev=y_prev)

        assert preds[0] == 50, "First val prediction should be last train value"
        assert preds[1] == 60, "Second val prediction should be first val actual"
        assert preds[2] == 70, "Third val prediction should be second val actual"

    def test_test_predictions_with_boundary(self, simple_persistence_data):
        """Test test-set predictions using full concatenated y_prev.

        First test prediction should use last validation actual (80).
        """
        d = simple_persistence_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        y_prev = pd.concat([d["y_train"], d["y_val"], d["y_test"]])
        preds = model.predict(d["X_test"], y_prev=y_prev)

        assert preds[0] == 80, "First test prediction should be last val value"
        assert preds[1] == 90
        assert preds[2] == 100

    def test_predictions_match_shifted_actuals(self, synthetic_data):
        """Verify each prediction equals the actual from the preceding day."""
        d = synthetic_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        y_prev = pd.concat([d["y_train"], d["y_val"]])
        preds = model.predict(d["X_val"], y_prev=y_prev)

        val_dates = d["y_val"].index
        for i, date in enumerate(val_dates):
            expected = y_prev[y_prev.index < date].iloc[-1]
            np.testing.assert_almost_equal(
                preds[i], expected,
                err_msg=f"Prediction for {date} should equal prior actual",
            )

    def test_fit_rejects_non_series(self, synthetic_data):
        """Verify fit() raises TypeError for non-Series y_train."""
        model = PersistenceModel()
        with pytest.raises(TypeError, match="pandas Series"):
            model.fit(synthetic_data["X_train"], [1, 2, 3])

    def test_empty_input(self, simple_persistence_data):
        """Predict with empty DataFrame returns empty array."""
        d = simple_persistence_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        empty_X = pd.DataFrame(columns=d["X_train"].columns)
        empty_X.index = pd.DatetimeIndex([])
        preds = model.predict(empty_X)
        assert len(preds) == 0

    def test_method_chaining(self, synthetic_data):
        """Verify fit() returns self for method chaining."""
        model = PersistenceModel()
        result = model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert result is model

    def test_prediction_length_matches_input(self, synthetic_data):
        """Verify prediction array length matches input DataFrame rows."""
        d = synthetic_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        y_prev = pd.concat([d["y_train"], d["y_val"]])
        preds = model.predict(d["X_val"], y_prev=y_prev)
        assert len(preds) == len(d["X_val"])


# ===========================================================================
# ClimatologyModel Tests
# ===========================================================================

class TestClimatologyModel:
    """Tests for the ClimatologyModel baseline."""

    def test_name(self):
        """Verify model name is 'Climatology'."""
        model = ClimatologyModel()
        assert model.name == "Climatology"

    def test_predict_without_fit_raises(self, synthetic_data):
        """Verify predicting before fitting raises RuntimeError."""
        model = ClimatologyModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(synthetic_data["X_val"])

    def test_doy_averages_computed(self, synthetic_data):
        """Verify day-of-year averages are stored after fit."""
        d = synthetic_data
        model = ClimatologyModel()
        model.fit(d["X_train"], d["y_train"])

        assert hasattr(model, "_doy_means")
        assert len(model._doy_means) > 0
        assert hasattr(model, "_overall_mean")

    def test_predictions_match_doy_means(self):
        """Verify predictions match day-of-year averages from training data."""
        # Two full years so each DOY appears twice
        dates_train = pd.date_range("2019-01-01", "2020-12-31", freq="D")
        n = len(dates_train)

        y_train = pd.Series(
            50 + 20 * np.sin(2 * np.pi * dates_train.dayofyear / 365.25),
            index=dates_train,
            name="NYC_TMAX",
        )
        X_train = pd.DataFrame({"f1": range(n)}, index=dates_train)

        # Predict for dates in a different year
        dates_pred = pd.date_range("2021-01-01", periods=10, freq="D")
        X_pred = pd.DataFrame({"f1": range(10)}, index=dates_pred)

        model = ClimatologyModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_pred)

        for i, date in enumerate(dates_pred):
            doy = date.dayofyear
            expected = y_train[y_train.index.dayofyear == doy].mean()
            np.testing.assert_almost_equal(
                preds[i], expected,
                err_msg=f"Prediction for DOY {doy} should match training mean",
            )

    def test_fallback_for_unseen_doy(self):
        """Verify overall-mean fallback for DOY 366 unseen in training.

        Training data from a non-leap year (2019) has only 365 days,
        so DOY 366 is never observed.  The prediction for Dec 31, 2020
        (a leap year, DOY = 366) should fall back to the overall training mean.
        """
        dates_train = pd.date_range("2019-01-01", "2019-12-31", freq="D")
        y_train = pd.Series(
            np.full(len(dates_train), 65.0),
            index=dates_train,
            name="NYC_TMAX",
        )
        X_train = pd.DataFrame({"f1": range(len(dates_train))}, index=dates_train)

        model = ClimatologyModel()
        model.fit(X_train, y_train)

        # Dec 31, 2020 is DOY 366 (leap year)
        date_366 = pd.Timestamp("2020-12-31")
        assert date_366.dayofyear == 366, "Sanity check: 2020-12-31 is DOY 366"

        X_pred = pd.DataFrame({"f1": [0]}, index=pd.DatetimeIndex([date_366]))
        preds = model.predict(X_pred)

        assert preds[0] == pytest.approx(65.0), \
            "DOY 366 should use overall training mean as fallback"

    def test_seasonal_pattern_preserved(self):
        """Verify climatology captures seasonal structure in the data."""
        dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
        doy = dates.dayofyear
        y = pd.Series(
            50 + 30 * np.sin(2 * np.pi * (doy - 80) / 365.25),
            index=dates,
            name="NYC_TMAX",
        )
        X = pd.DataFrame({"f1": range(len(dates))}, index=dates)

        model = ClimatologyModel()
        model.fit(X, y)

        summer_dates = pd.date_range("2021-07-01", periods=5, freq="D")
        winter_dates = pd.date_range("2021-01-15", periods=5, freq="D")
        X_summer = pd.DataFrame({"f1": range(5)}, index=summer_dates)
        X_winter = pd.DataFrame({"f1": range(5)}, index=winter_dates)

        summer_preds = model.predict(X_summer)
        winter_preds = model.predict(X_winter)

        assert np.mean(summer_preds) > np.mean(winter_preds), \
            "Summer predictions should be warmer than winter predictions"

    def test_fit_rejects_non_series(self, synthetic_data):
        """Verify fit() raises TypeError for non-Series input."""
        model = ClimatologyModel()
        with pytest.raises(TypeError, match="pandas Series"):
            model.fit(synthetic_data["X_train"], [1, 2, 3])

    def test_empty_input(self, synthetic_data):
        """Predict with empty DataFrame returns empty array."""
        d = synthetic_data
        model = ClimatologyModel()
        model.fit(d["X_train"], d["y_train"])

        empty_X = pd.DataFrame(columns=d["X_train"].columns)
        empty_X.index = pd.DatetimeIndex([])
        preds = model.predict(empty_X)
        assert len(preds) == 0

    def test_method_chaining(self, synthetic_data):
        """Verify fit() returns self."""
        model = ClimatologyModel()
        result = model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert result is model

    def test_single_year_training(self):
        """With one year of training, each DOY average equals the single value."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        values = np.arange(len(dates), dtype=float)
        y = pd.Series(values, index=dates, name="NYC_TMAX")
        X = pd.DataFrame({"f1": range(len(dates))}, index=dates)

        model = ClimatologyModel()
        model.fit(X, y)

        preds = model.predict(X)
        np.testing.assert_array_almost_equal(preds, values)


# ===========================================================================
# LinearRegressionModel Tests
# ===========================================================================

class TestLinearRegressionModel:
    """Tests for the LinearRegressionModel baseline."""

    def test_name(self):
        """Verify model name."""
        model = LinearRegressionModel()
        assert model.name == "Linear Regression"

    def test_predict_without_fit_raises(self, synthetic_data):
        """Verify predicting before fitting raises RuntimeError."""
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(synthetic_data["X_val"])

    def test_fit_predict_works(self, synthetic_data):
        """Test that fit/predict runs without errors and returns finite values."""
        d = synthetic_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_val"])

        assert len(preds) == len(d["X_val"])
        assert np.all(np.isfinite(preds))

    def test_predictions_reasonable_range(self, synthetic_data):
        """Verify predictions are in a plausible temperature range."""
        d = synthetic_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_val"])

        assert np.all(preds > -50), "No absurdly low predictions"
        assert np.all(preds < 150), "No absurdly high predictions"

    def test_perfect_linear_data(self, linear_data):
        """With perfectly linear data, OLS should achieve near-zero MAE."""
        d = linear_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_val"])

        mae = float(np.mean(np.abs(preds - d["y_val"].values)))
        assert mae < 1e-10, f"MAE should be near zero for perfect linear data, got {mae}"

    def test_coefficient_shapes(self, linear_data):
        """Verify coefficients have the correct shape."""
        d = linear_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        assert len(model.coefficients) == d["X_train"].shape[1]
        assert isinstance(model.intercept, float)

    def test_recovers_true_coefficients(self, linear_data):
        """With perfect linear data, OLS should recover the true coefficients."""
        d = linear_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        np.testing.assert_array_almost_equal(
            model.coefficients, d["true_coefs"], decimal=10
        )
        np.testing.assert_almost_equal(model.intercept, 50.0, decimal=10)

    def test_empty_input(self, linear_data):
        """Predict with empty DataFrame returns empty array."""
        d = linear_data
        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        empty_X = pd.DataFrame(columns=d["X_train"].columns)
        preds = model.predict(empty_X)
        assert len(preds) == 0

    def test_numpy_array_input(self, synthetic_data):
        """Verify that raw numpy arrays work as input."""
        d = synthetic_data
        model = LinearRegressionModel()
        model.fit(d["X_train"].values, d["y_train"].values)
        preds = model.predict(d["X_val"].values)

        assert len(preds) == len(d["X_val"])
        assert np.all(np.isfinite(preds))

    def test_method_chaining(self, synthetic_data):
        """Verify fit() returns self."""
        model = LinearRegressionModel()
        result = model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert result is model

    def test_coefficients_before_fit_raises(self):
        """Verify accessing coefficients before fit raises RuntimeError."""
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.coefficients
        with pytest.raises(RuntimeError, match="fitted"):
            _ = model.intercept


# ===========================================================================
# RidgeRegressionModel Tests
# ===========================================================================

class TestRidgeRegressionModel:
    """Tests for the RidgeRegressionModel baseline."""

    def test_name_includes_alpha(self):
        """Verify model name includes the alpha value."""
        model = RidgeRegressionModel(alpha=0.5)
        assert "0.5" in model.name
        assert "Ridge" in model.name

    def test_default_alpha(self):
        """Verify default alpha is 1.0."""
        model = RidgeRegressionModel()
        assert "1.0" in model.name

    def test_invalid_alpha_raises(self):
        """Verify non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RidgeRegressionModel(alpha=-1.0)
        with pytest.raises(ValueError, match="positive"):
            RidgeRegressionModel(alpha=0.0)

    def test_predict_without_fit_raises(self, synthetic_data):
        """Verify predicting before fitting raises RuntimeError."""
        model = RidgeRegressionModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(synthetic_data["X_val"])

    def test_fit_predict_works(self, synthetic_data):
        """Test that fit/predict runs without errors and returns finite values."""
        d = synthetic_data
        model = RidgeRegressionModel()
        model.fit(d["X_train"], d["y_train"])
        preds = model.predict(d["X_val"])

        assert len(preds) == len(d["X_val"])
        assert np.all(np.isfinite(preds))

    def test_different_alphas_different_results(self, synthetic_data):
        """Verify different alpha values produce different predictions."""
        d = synthetic_data

        model_low = RidgeRegressionModel(alpha=0.01)
        model_high = RidgeRegressionModel(alpha=100.0)

        model_low.fit(d["X_train"], d["y_train"])
        model_high.fit(d["X_train"], d["y_train"])

        preds_low = model_low.predict(d["X_val"])
        preds_high = model_high.predict(d["X_val"])

        assert not np.allclose(preds_low, preds_high, atol=1e-5), \
            "Different alpha values should produce different predictions"

    def test_high_alpha_shrinks_coefficients(self, synthetic_data):
        """Higher alpha should produce smaller coefficient magnitudes."""
        d = synthetic_data

        model_low = RidgeRegressionModel(alpha=0.01)
        model_high = RidgeRegressionModel(alpha=100.0)

        model_low.fit(d["X_train"], d["y_train"])
        model_high.fit(d["X_train"], d["y_train"])

        l2_low = float(np.sum(model_low.coefficients ** 2))
        l2_high = float(np.sum(model_high.coefficients ** 2))

        assert l2_high < l2_low, \
            "Higher alpha should shrink coefficients (L2 norm)"

    def test_collinear_features(self):
        """Ridge should handle nearly collinear features gracefully."""
        rng = np.random.RandomState(42)
        dates = make_date_index("2020-01-01", 100)

        x1 = rng.randn(100)
        x2 = x1 + rng.normal(0, 0.01, 100)  # Nearly identical to x1
        X = pd.DataFrame(
            {"f1": x1, "f2": x2, "f3": rng.randn(100)}, index=dates
        )
        y = pd.Series(
            x1 * 2 + 50 + rng.normal(0, 0.5, 100), index=dates, name="NYC_TMAX"
        )

        model = RidgeRegressionModel(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)

        assert np.all(np.isfinite(preds)), "Predictions should be finite"
        assert len(model.coefficients) == 3

    def test_coefficient_shapes(self, linear_data):
        """Verify coefficients have the correct shape."""
        d = linear_data
        model = RidgeRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        assert len(model.coefficients) == d["X_train"].shape[1]
        assert isinstance(model.intercept, float)

    def test_empty_input(self, linear_data):
        """Predict with empty DataFrame returns empty array."""
        d = linear_data
        model = RidgeRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        empty_X = pd.DataFrame(columns=d["X_train"].columns)
        preds = model.predict(empty_X)
        assert len(preds) == 0

    def test_method_chaining(self, synthetic_data):
        """Verify fit() returns self."""
        model = RidgeRegressionModel()
        result = model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
        assert result is model


# ===========================================================================
# run_all_baselines Tests
# ===========================================================================

class TestRunAllBaselines:
    """Tests for the run_all_baselines convenience function."""

    def test_returns_all_models(self, synthetic_data):
        """Verify results contain all 4 baseline models."""
        d = synthetic_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )

        assert "Persistence" in results
        assert "Climatology" in results
        assert "Linear Regression" in results
        assert "Ridge (alpha=1.0)" in results
        assert len(results) == 4

    def test_result_structure(self, synthetic_data):
        """Verify each result has the expected keys."""
        d = synthetic_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )

        expected_keys = {
            "model", "val_predictions", "test_predictions",
            "val_metrics", "test_metrics",
        }
        for name, res in results.items():
            assert set(res.keys()) == expected_keys, \
                f"{name}: unexpected result keys {set(res.keys())}"

    def test_prediction_lengths(self, synthetic_data):
        """Verify predictions have correct lengths for val and test."""
        d = synthetic_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )

        for name, res in results.items():
            assert len(res["val_predictions"]) == len(d["X_val"]), \
                f"{name}: val predictions length mismatch"
            assert len(res["test_predictions"]) == len(d["X_test"]), \
                f"{name}: test predictions length mismatch"

    def test_metrics_keys(self, synthetic_data):
        """Verify metrics contain all expected keys."""
        d = synthetic_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )

        expected_metric_keys = {"mae", "rmse", "mbe", "r_squared"}
        for name, res in results.items():
            assert set(res["val_metrics"].keys()) == expected_metric_keys, \
                f"{name}: missing val metric keys"
            assert set(res["test_metrics"].keys()) == expected_metric_keys, \
                f"{name}: missing test metric keys"

    def test_metrics_finite(self, synthetic_data):
        """Verify all metrics are finite numbers."""
        d = synthetic_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )

        for name, res in results.items():
            for split_label in ["val_metrics", "test_metrics"]:
                for key, value in res[split_label].items():
                    assert np.isfinite(value), \
                        f"{name} {split_label} {key} is not finite: {value}"

    def test_with_simple_data(self, simple_persistence_data):
        """Verify run_all_baselines works with minimal data."""
        d = simple_persistence_data
        results = run_all_baselines(
            d["X_train"], d["X_val"], d["X_test"],
            d["y_train"], d["y_val"], d["y_test"],
        )
        assert len(results) == 4
        for name, res in results.items():
            assert len(res["val_predictions"]) == 3
            assert len(res["test_predictions"]) == 3


# ===========================================================================
# compute_metrics Tests
# ===========================================================================

class TestComputeMetrics:
    """Tests for the compute_metrics helper function."""

    def test_perfect_predictions(self):
        """Perfect predictions should give zero MAE/RMSE/MBE and R2=1."""
        y = np.array([50.0, 60.0, 70.0, 80.0])
        metrics = compute_metrics(y, y)

        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mbe"] == pytest.approx(0.0)
        assert metrics["r_squared"] == pytest.approx(1.0)

    def test_known_errors(self):
        """Test with known symmetric errors: [+2, -2, +2, -2]."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(2.0)
        assert metrics["rmse"] == pytest.approx(2.0)
        assert metrics["mbe"] == pytest.approx(0.0)  # Symmetric errors cancel

    def test_positive_bias(self):
        """Consistent over-prediction should give positive MBE."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([13.0, 23.0, 33.0])  # All +3

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mbe"] == pytest.approx(3.0)
        assert metrics["mae"] == pytest.approx(3.0)

    def test_handles_nan_in_predictions(self):
        """NaN predictions should be excluded from metric computation."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([np.nan, 20.0, 30.0, 40.0])

        metrics = compute_metrics(y_true, y_pred)

        # Only 3 valid predictions (all perfect)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_all_nan_returns_nan(self):
        """All-NaN predictions should return NaN metrics."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([np.nan, np.nan, np.nan])

        metrics = compute_metrics(y_true, y_pred)
        assert np.isnan(metrics["mae"])
        assert np.isnan(metrics["rmse"])
        assert np.isnan(metrics["mbe"])
        assert np.isnan(metrics["r_squared"])

    def test_empty_arrays(self):
        """Empty arrays should return NaN metrics."""
        metrics = compute_metrics(np.array([]), np.array([]))
        assert np.isnan(metrics["mae"])


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for all models."""

    def test_single_row_train_ridge(self):
        """Ridge should handle single-row training data."""
        date = pd.date_range("2020-06-15", periods=1, freq="D")
        X = pd.DataFrame({"f1": [1.0], "f2": [2.0]}, index=date)
        y = pd.Series([75.0], index=date, name="NYC_TMAX")

        model = RidgeRegressionModel(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 1
        assert np.isfinite(preds[0])

    def test_single_row_train_linear(self):
        """LinearRegression should handle single-row training data."""
        date = pd.date_range("2020-06-15", periods=1, freq="D")
        X = pd.DataFrame({"f1": [1.0], "f2": [2.0]}, index=date)
        y = pd.Series([75.0], index=date, name="NYC_TMAX")

        model = LinearRegressionModel()
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 1

    def test_nan_in_features_raises(self, synthetic_data):
        """NaN in features should raise ValueError in sklearn linear models.

        sklearn's LinearRegression does not accept NaN inputs natively.
        The correct upstream fix is data imputation (handled by the
        preprocessing pipeline), not model-level NaN propagation.
        """
        d = synthetic_data
        X_with_nan = d["X_val"].copy()
        X_with_nan.iloc[0, 0] = np.nan

        model = LinearRegressionModel()
        model.fit(d["X_train"], d["y_train"])

        with pytest.raises(ValueError, match="NaN"):
            model.predict(X_with_nan)

    def test_persistence_with_date_gaps(self):
        """Persistence handles non-consecutive dates (gaps in the series).

        Dates have a 3-day gap between train and val; shift(1) still
        produces the correct lag-1 predictions based on row position.
        """
        dates_train = pd.DatetimeIndex([
            "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06",
        ])
        dates_val = pd.DatetimeIndex(["2020-01-07", "2020-01-08"])

        y_train = pd.Series(
            [50, 60, 70, 80], index=dates_train, name="NYC_TMAX"
        )
        y_val = pd.Series([90, 100], index=dates_val, name="NYC_TMAX")

        X_train = pd.DataFrame({"f1": range(4)}, index=dates_train)
        X_val = pd.DataFrame({"f1": range(2)}, index=dates_val)

        model = PersistenceModel()
        model.fit(X_train, y_train)

        y_prev = pd.concat([y_train, y_val])
        preds = model.predict(X_val, y_prev=y_prev)

        # Jan 7 prediction = shifted value at Jan 7 = y_prev at Jan 6 = 80
        assert preds[0] == 80
        # Jan 8 prediction = shifted value at Jan 8 = y_prev at Jan 7 = 90
        assert preds[1] == 90

    def test_climatology_single_year_no_feb29(self):
        """Climatology from a single non-leap year handles leap-year prediction."""
        dates_train = pd.date_range("2019-01-01", "2019-12-31", freq="D")
        y_train = pd.Series(
            np.ones(len(dates_train)) * 50.0,
            index=dates_train,
            name="NYC_TMAX",
        )
        X_train = pd.DataFrame({"f1": range(len(dates_train))}, index=dates_train)

        model = ClimatologyModel()
        model.fit(X_train, y_train)

        # Predict for all of 2020 (leap year — 366 days)
        dates_pred = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        X_pred = pd.DataFrame({"f1": range(len(dates_pred))}, index=dates_pred)
        preds = model.predict(X_pred)

        # All predictions should be 50.0 (known DOYs) or 50.0 (fallback)
        np.testing.assert_array_almost_equal(preds, 50.0)

    def test_persistence_predict_none_uses_stored(self, simple_persistence_data):
        """When y_prev is None, persistence uses stored y_train."""
        d = simple_persistence_data
        model = PersistenceModel()
        model.fit(d["X_train"], d["y_train"])

        # Predict on training data with y_prev=None
        preds = model.predict(d["X_train"])

        # Should be same as predict(X_train, y_prev=y_train)
        preds_explicit = model.predict(d["X_train"], y_prev=d["y_train"])
        np.testing.assert_array_equal(preds, preds_explicit)
