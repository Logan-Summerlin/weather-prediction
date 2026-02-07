"""
Tests for the evaluation framework (src/evaluate.py).

Validates:
  - Metric computation (MAE, RMSE, R2, bias, thresholds, max error)
  - Edge cases (empty arrays, NaN values, single point, constant values)
  - Seasonal breakdown correctness
  - Metrics comparison table formatting
  - All plot functions produce output files without error
  - Convenience function (evaluate_predictions) end-to-end
  - Report generation
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import (
    compute_metrics,
    compute_seasonal_metrics,
    format_metrics_table,
    plot_actual_vs_predicted,
    plot_time_series,
    plot_residual_histogram,
    plot_residuals_by_month,
    plot_baseline_comparison,
    generate_baseline_report,
    evaluate_predictions,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def perfect_data():
    """Data where predictions exactly match actuals."""
    np.random.seed(42)
    actual = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 55.0, 65.0, 75.0])
    pred = actual.copy()
    return actual, pred


@pytest.fixture
def constant_bias_data():
    """Predictions are always exactly 2 degF higher than actuals."""
    actual = np.array([40.0, 50.0, 60.0, 70.0, 80.0, 45.0, 55.0, 65.0, 75.0, 85.0])
    pred = actual + 2.0
    return actual, pred


@pytest.fixture
def known_error_data():
    """Data with known error pattern for threshold verification.

    Errors: [0.5, -1.5, 2.5, -0.5, 3.5, 1.0, -2.0, 0.0]
    Abs errors: [0.5, 1.5, 2.5, 0.5, 3.5, 1.0, 2.0, 0.0]
    Within 1: 0.5, 0.5, 1.0, 0.0 -> 4/8 = 50%
    Within 2: 0.5, 1.5, 0.5, 1.0, 2.0, 0.0 -> 6/8 = 75%
    Within 3: 0.5, 1.5, 2.5, 0.5, 1.0, 2.0, 0.0 -> 7/8 = 87.5%
    """
    actual = np.array([60.0, 70.0, 50.0, 80.0, 40.0, 55.0, 65.0, 75.0])
    errors = np.array([0.5, -1.5, 2.5, -0.5, 3.5, 1.0, -2.0, 0.0])
    pred = actual + errors
    return actual, pred, errors


@pytest.fixture
def synthetic_seasonal_data():
    """One full year of daily data (2021) for seasonal testing.

    Creates realistic temperature-like patterns per season.
    """
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    n = len(dates)
    np.random.seed(99)

    # Simulate a temperature cycle (convert to numpy to allow mutation)
    day_of_year = dates.dayofyear.to_numpy(dtype=np.float64)
    actual = 55 + 25 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    actual += np.random.normal(0, 2, n)

    # Predicted values with season-dependent bias
    pred = actual.copy()  # numpy array, supports item assignment
    for i, d in enumerate(dates):
        if d.month in (12, 1, 2):
            pred[i] += 3.0   # Over-predict in winter
        elif d.month in (6, 7, 8):
            pred[i] -= 2.0   # Under-predict in summer
        else:
            pred[i] += np.random.normal(0, 1)

    return actual, pred, dates


@pytest.fixture
def multi_model_results():
    """Pre-computed metrics for three models, for table/comparison tests."""
    return {
        "Persistence": {
            "n": 274,
            "mae": 4.32,
            "rmse": 5.61,
            "r2": 0.91,
            "bias": -0.15,
            "within_1f": 12.5,
            "within_2f": 28.3,
            "within_3f": 44.1,
            "max_abs_error": 22.1,
        },
        "Climatology": {
            "n": 274,
            "mae": 6.10,
            "rmse": 7.83,
            "r2": 0.82,
            "bias": 0.42,
            "within_1f": 7.1,
            "within_2f": 15.3,
            "within_3f": 24.5,
            "max_abs_error": 28.4,
        },
        "Ridge Regression": {
            "n": 274,
            "mae": 3.05,
            "rmse": 3.92,
            "r2": 0.95,
            "bias": 0.08,
            "within_1f": 18.2,
            "within_2f": 38.7,
            "within_3f": 56.9,
            "max_abs_error": 15.3,
        },
    }


# ===========================================================================
# Tests: compute_metrics
# ===========================================================================

class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self, perfect_data):
        """Perfect predictions should yield MAE=0, RMSE=0, R2=1."""
        actual, pred = perfect_data
        m = compute_metrics(actual, pred, model_name="Perfect")

        assert m["model_name"] == "Perfect"
        assert m["n"] == len(actual)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"] == pytest.approx(1.0, abs=1e-10)
        assert m["bias"] == pytest.approx(0.0, abs=1e-10)
        assert m["within_1f"] == pytest.approx(100.0, abs=1e-10)
        assert m["within_2f"] == pytest.approx(100.0, abs=1e-10)
        assert m["within_3f"] == pytest.approx(100.0, abs=1e-10)
        assert m["max_abs_error"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_bias(self, constant_bias_data):
        """Constant +2 degF bias: MAE=2, bias=+2, within_2f includes boundary."""
        actual, pred = constant_bias_data
        m = compute_metrics(actual, pred)

        assert m["mae"] == pytest.approx(2.0, abs=1e-10)
        assert m["rmse"] == pytest.approx(2.0, abs=1e-10)
        assert m["bias"] == pytest.approx(2.0, abs=1e-10)
        assert m["max_abs_error"] == pytest.approx(2.0, abs=1e-10)
        # All errors are exactly 2 — should be within +/-2
        assert m["within_2f"] == pytest.approx(100.0, abs=1e-10)
        # None within +/-1
        assert m["within_1f"] == pytest.approx(0.0, abs=1e-10)

    def test_known_threshold_percentages(self, known_error_data):
        """Verify within-threshold percentages with hand-computed values."""
        actual, pred, errors = known_error_data
        m = compute_metrics(actual, pred)

        abs_errors = np.abs(errors)
        n = len(errors)
        expected_within_1 = np.sum(abs_errors <= 1.0) / n * 100
        expected_within_2 = np.sum(abs_errors <= 2.0) / n * 100
        expected_within_3 = np.sum(abs_errors <= 3.0) / n * 100

        assert m["within_1f"] == pytest.approx(expected_within_1, abs=0.01)
        assert m["within_2f"] == pytest.approx(expected_within_2, abs=0.01)
        assert m["within_3f"] == pytest.approx(expected_within_3, abs=0.01)

    def test_known_mae_rmse(self, known_error_data):
        """Verify MAE and RMSE with hand-computed values."""
        actual, pred, errors = known_error_data
        m = compute_metrics(actual, pred)

        expected_mae = float(np.mean(np.abs(errors)))
        expected_rmse = float(np.sqrt(np.mean(errors ** 2)))

        assert m["mae"] == pytest.approx(expected_mae, abs=1e-10)
        assert m["rmse"] == pytest.approx(expected_rmse, abs=1e-10)

    def test_max_abs_error(self, known_error_data):
        """Max absolute error should be the largest single error."""
        actual, pred, errors = known_error_data
        m = compute_metrics(actual, pred)
        assert m["max_abs_error"] == pytest.approx(3.5, abs=1e-10)

    def test_r2_in_valid_range(self):
        """R2 for reasonable synthetic data should be between -1 and 1."""
        np.random.seed(123)
        actual = np.linspace(30, 100, 200)
        pred = actual + np.random.normal(0, 3, 200)
        m = compute_metrics(actual, pred)
        assert -1.0 <= m["r2"] <= 1.0

    def test_empty_arrays(self):
        """Empty inputs should return NaN metrics, not crash."""
        m = compute_metrics([], [])
        assert m["n"] == 0
        assert math.isnan(m["mae"])
        assert math.isnan(m["rmse"])
        assert math.isnan(m["r2"])
        assert math.isnan(m["bias"])
        assert math.isnan(m["within_1f"])
        assert math.isnan(m["max_abs_error"])

    def test_nan_values_stripped(self):
        """NaN entries should be silently dropped before computing metrics."""
        actual = np.array([50.0, np.nan, 70.0, 80.0])
        pred = np.array([50.0, 60.0, np.nan, 80.0])
        m = compute_metrics(actual, pred)

        # Only indices 0 and 3 survive — both are perfect
        assert m["n"] == 2
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_all_nan_values(self):
        """All-NaN inputs should behave like empty arrays."""
        actual = np.array([np.nan, np.nan])
        pred = np.array([np.nan, np.nan])
        m = compute_metrics(actual, pred)
        assert m["n"] == 0
        assert math.isnan(m["mae"])

    def test_model_name_omitted(self):
        """When model_name is None, the key should not appear in the dict."""
        m = compute_metrics([50, 60], [51, 61])
        assert "model_name" not in m

    def test_model_name_included(self):
        """When model_name is given, it should appear in the dict."""
        m = compute_metrics([50, 60], [51, 61], model_name="Test")
        assert m["model_name"] == "Test"

    def test_length_mismatch_raises(self):
        """Mismatched input lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_metrics([1, 2, 3], [1, 2])

    def test_all_identical_actuals(self):
        """When all actuals are the same, R2 should be NaN (undefined)."""
        actual = np.array([60.0, 60.0, 60.0, 60.0])
        pred = np.array([61.0, 59.0, 60.5, 60.0])
        m = compute_metrics(actual, pred)
        assert math.isnan(m["r2"])
        # MAE should still be valid
        assert m["mae"] == pytest.approx(np.mean([1.0, 1.0, 0.5, 0.0]), abs=1e-10)

    def test_pandas_series_input(self):
        """Function should accept pandas Series as well as numpy arrays."""
        actual = pd.Series([50.0, 60.0, 70.0])
        pred = pd.Series([51.0, 59.0, 72.0])
        m = compute_metrics(actual, pred)
        assert m["n"] == 3
        assert m["mae"] == pytest.approx(np.mean([1.0, 1.0, 2.0]), abs=1e-10)

    def test_list_input(self):
        """Function should accept plain Python lists."""
        m = compute_metrics([50, 60, 70], [50, 60, 70])
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_negative_bias(self):
        """Consistent under-prediction should yield negative bias."""
        actual = np.array([50.0, 60.0, 70.0])
        pred = np.array([48.0, 58.0, 68.0])
        m = compute_metrics(actual, pred)
        assert m["bias"] == pytest.approx(-2.0, abs=1e-10)

    def test_single_prediction(self):
        """Single data point should not crash."""
        m = compute_metrics([60.0], [62.0])
        assert m["n"] == 1
        assert m["mae"] == pytest.approx(2.0, abs=1e-10)
        assert m["rmse"] == pytest.approx(2.0, abs=1e-10)
        assert m["bias"] == pytest.approx(2.0, abs=1e-10)
        assert m["max_abs_error"] == pytest.approx(2.0, abs=1e-10)

    def test_very_large_errors(self):
        """Extreme errors should be computed without overflow."""
        actual = np.array([0.0])
        pred = np.array([1e6])
        m = compute_metrics(actual, pred)
        assert m["mae"] == pytest.approx(1e6, rel=1e-6)
        assert m["rmse"] == pytest.approx(1e6, rel=1e-6)
        assert m["max_abs_error"] == pytest.approx(1e6, rel=1e-6)


# ===========================================================================
# Tests: compute_seasonal_metrics
# ===========================================================================

class TestComputeSeasonalMetrics:
    """Tests for the seasonal breakdown function."""

    def test_all_four_seasons_present(self, synthetic_seasonal_data):
        """Full-year data should produce entries for all four seasons."""
        actual, pred, dates = synthetic_seasonal_data
        seasonal = compute_seasonal_metrics(actual, pred, dates)

        assert "Winter (DJF)" in seasonal
        assert "Spring (MAM)" in seasonal
        assert "Summer (JJA)" in seasonal
        assert "Fall (SON)" in seasonal

    def test_seasonal_counts_add_up(self, synthetic_seasonal_data):
        """Sum of per-season sample sizes should equal the total."""
        actual, pred, dates = synthetic_seasonal_data
        seasonal = compute_seasonal_metrics(actual, pred, dates)
        total_n = sum(s["n"] for s in seasonal.values())
        assert total_n == len(actual)

    def test_winter_bias_positive(self, synthetic_seasonal_data):
        """Winter should have positive bias (we added +3 in the fixture)."""
        actual, pred, dates = synthetic_seasonal_data
        seasonal = compute_seasonal_metrics(actual, pred, dates)
        # Winter bias should be around +3.0 (fixture adds exactly +3)
        assert seasonal["Winter (DJF)"]["bias"] > 2.0

    def test_summer_bias_negative(self, synthetic_seasonal_data):
        """Summer should have negative bias (we subtracted 2 in the fixture)."""
        actual, pred, dates = synthetic_seasonal_data
        seasonal = compute_seasonal_metrics(actual, pred, dates)
        assert seasonal["Summer (JJA)"]["bias"] < -1.0

    def test_partial_year_coverage(self):
        """Data covering only some months should still compute partial seasons."""
        dates = pd.date_range("2021-06-01", "2021-08-31", freq="D")
        n = len(dates)
        actual = np.ones(n) * 85.0
        pred = np.ones(n) * 86.0

        seasonal = compute_seasonal_metrics(actual, pred, dates)

        # Only summer should be present
        assert "Summer (JJA)" in seasonal
        assert len(seasonal) == 1
        assert seasonal["Summer (JJA)"]["mae"] == pytest.approx(1.0, abs=1e-10)
        assert seasonal["Summer (JJA)"]["bias"] == pytest.approx(1.0, abs=1e-10)

    def test_seasonal_mae_rmse_nonnegative(self, synthetic_seasonal_data):
        """MAE and RMSE should always be non-negative."""
        actual, pred, dates = synthetic_seasonal_data
        seasonal = compute_seasonal_metrics(actual, pred, dates)

        for season, sm in seasonal.items():
            assert sm["mae"] >= 0.0, f"{season} MAE is negative"
            assert sm["rmse"] >= 0.0, f"{season} RMSE is negative"

    def test_dates_length_mismatch_after_nan_removal(self):
        """Dates misaligned with data after NaN removal should raise ValueError."""
        actual = np.array([50.0, np.nan, 70.0])
        pred = np.array([50.0, 60.0, 70.0])
        dates = pd.date_range("2021-01-01", periods=3, freq="D")

        # After NaN removal, 2 data points but 3 dates -> error
        with pytest.raises(ValueError, match="does not match"):
            compute_seasonal_metrics(actual, pred, dates)

    def test_december_is_winter(self):
        """December observations must be assigned to Winter (DJF)."""
        dates = pd.date_range("2021-12-01", "2021-12-31", freq="D")
        n = len(dates)
        actual = np.ones(n) * 30.0
        pred = np.ones(n) * 32.0
        seasonal = compute_seasonal_metrics(actual, pred, dates)

        assert "Winter (DJF)" in seasonal
        assert seasonal["Winter (DJF)"]["n"] == n


# ===========================================================================
# Tests: format_metrics_table
# ===========================================================================

class TestFormatMetricsTable:
    """Tests for the comparison table formatter."""

    def test_output_is_nonempty_string(self, multi_model_results):
        """Output should be a non-empty string."""
        table = format_metrics_table(multi_model_results)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_all_model_names_in_table(self, multi_model_results):
        """Every model name should appear somewhere in the table."""
        table = format_metrics_table(multi_model_results)
        for name in multi_model_results:
            assert name in table, f"Model name '{name}' missing from table"

    def test_single_model(self):
        """Table should work correctly with a single model."""
        results = {
            "Only Model": {
                "n": 100, "mae": 3.0, "rmse": 4.0, "r2": 0.9,
                "bias": 0.1, "within_1f": 20.0, "within_2f": 40.0,
                "within_3f": 60.0, "max_abs_error": 12.0,
            }
        }
        table = format_metrics_table(results)
        assert "Only Model" in table
        assert "3.00" in table  # MAE

    def test_empty_results_dict(self):
        """Empty dict should return a placeholder message, not crash."""
        table = format_metrics_table({})
        assert isinstance(table, str)
        assert len(table) > 0

    def test_table_contains_metric_values(self, multi_model_results):
        """Specific metric values should appear in the table."""
        table = format_metrics_table(multi_model_results)
        # Ridge Regression MAE = 3.05
        assert "3.05" in table
        # Persistence RMSE = 5.61
        assert "5.61" in table


# ===========================================================================
# Tests: Plot Functions (file creation only)
# ===========================================================================

class TestPlots:
    """Verify that each plot function runs without error and creates a file."""

    @pytest.fixture
    def plot_data(self):
        """Generate synthetic data for plot tests."""
        np.random.seed(42)
        dates = pd.date_range("2022-04-01", periods=100, freq="D")
        actual = 55 + 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25) \
            + np.random.normal(0, 2, 100)
        pred = actual + np.random.normal(0, 3, 100)
        return actual, pred, dates

    def test_plot_actual_vs_predicted(self, plot_data, tmp_path):
        """Scatter plot should create a .png file."""
        actual, pred, _ = plot_data
        save_path = str(tmp_path / "scatter.png")
        plot_actual_vs_predicted(actual, pred, "TestModel", save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_time_series(self, plot_data, tmp_path):
        """Time-series plot should create a .png file."""
        actual, pred, dates = plot_data
        save_path = str(tmp_path / "timeseries.png")
        plot_time_series(actual, pred, dates, "TestModel", save_path, n_days=30)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_time_series_full(self, plot_data, tmp_path):
        """Time-series with n_days=0 should show the entire series."""
        actual, pred, dates = plot_data
        save_path = str(tmp_path / "timeseries_full.png")
        plot_time_series(actual, pred, dates, "TestModel", save_path, n_days=0)
        assert os.path.isfile(save_path)

    def test_plot_residual_histogram(self, plot_data, tmp_path):
        """Residual histogram should create a .png file."""
        actual, pred, _ = plot_data
        save_path = str(tmp_path / "residual_hist.png")
        plot_residual_histogram(actual, pred, "TestModel", save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_residuals_by_month(self, plot_data, tmp_path):
        """Monthly residual box plot should create a .png file."""
        actual, pred, dates = plot_data
        save_path = str(tmp_path / "residuals_month.png")
        plot_residuals_by_month(actual, pred, dates, "TestModel", save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_baseline_comparison(self, multi_model_results, tmp_path):
        """Bar chart comparison should create a .png file."""
        save_path = str(tmp_path / "comparison.png")
        plot_baseline_comparison(multi_model_results, metric="mae",
                                 save_path=save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_baseline_comparison_rmse(self, multi_model_results, tmp_path):
        """Comparison bar chart should work with different metrics."""
        save_path = str(tmp_path / "comparison_rmse.png")
        plot_baseline_comparison(multi_model_results, metric="rmse",
                                 save_path=save_path)
        assert os.path.isfile(save_path)

    def test_plot_baseline_comparison_empty_raises(self):
        """Empty results dict should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            plot_baseline_comparison({}, metric="mae")

    def test_plot_baseline_comparison_no_save(self, multi_model_results):
        """Comparison plot with save_path=None should not crash."""
        # Should not raise
        plot_baseline_comparison(multi_model_results, metric="mae",
                                 save_path=None)


# ===========================================================================
# Tests: evaluate_predictions
# ===========================================================================

class TestEvaluatePredictions:
    """Tests for the convenience evaluate_predictions function."""

    def test_returns_metrics_dict(self):
        """Should return a dict with all expected metric keys."""
        actual = np.array([50.0, 60.0, 70.0, 80.0])
        pred = np.array([52.0, 58.0, 71.0, 79.0])
        m = evaluate_predictions(actual, pred, model_name="Quick")

        assert isinstance(m, dict)
        assert "mae" in m
        assert "rmse" in m
        assert "r2" in m
        assert "bias" in m
        assert "within_1f" in m
        assert "within_2f" in m
        assert "within_3f" in m
        assert "max_abs_error" in m
        assert m["model_name"] == "Quick"

    def test_creates_plots_with_output_dir(self, tmp_path):
        """When output_dir is given, plot files should be created."""
        np.random.seed(10)
        dates = pd.date_range("2022-01-01", periods=50, freq="D")
        actual = np.random.normal(60, 10, 50)
        pred = actual + np.random.normal(0, 3, 50)

        out = str(tmp_path / "eval_output")
        m = evaluate_predictions(actual, pred, dates=dates,
                                 model_name="PlotTest", output_dir=out)

        assert os.path.isdir(out)
        # Scatter and residual hist should be present
        assert os.path.isfile(os.path.join(out, "plottest_scatter.png"))
        assert os.path.isfile(os.path.join(out, "plottest_residual_hist.png"))
        # Date-dependent plots should also be present
        assert os.path.isfile(os.path.join(out, "plottest_timeseries.png"))
        assert os.path.isfile(os.path.join(out, "plottest_residuals_month.png"))

        # Seasonal breakdown should be attached
        assert "seasonal" in m

    def test_no_plots_without_output_dir(self):
        """Without output_dir, should return metrics only (no side effects)."""
        m = evaluate_predictions([50, 60], [51, 61])
        assert "mae" in m
        # No seasonal key when dates not provided
        assert "seasonal" not in m

    def test_no_dates_skips_time_plots(self, tmp_path):
        """Without dates, date-dependent plots should be skipped."""
        out = str(tmp_path / "no_dates")
        actual = np.array([50.0, 60.0, 70.0])
        pred = np.array([51.0, 59.0, 72.0])
        evaluate_predictions(actual, pred, dates=None,
                             model_name="NoDate", output_dir=out)
        # Scatter and histogram should be present
        assert os.path.isfile(os.path.join(out, "nodate_scatter.png"))
        assert os.path.isfile(os.path.join(out, "nodate_residual_hist.png"))
        # Time-series and monthly should NOT exist
        assert not os.path.isfile(os.path.join(out, "nodate_timeseries.png"))
        assert not os.path.isfile(os.path.join(out, "nodate_residuals_month.png"))


# ===========================================================================
# Tests: generate_baseline_report
# ===========================================================================

class TestGenerateBaselineReport:
    """Tests for the report generation function."""

    def test_report_creates_files(self, multi_model_results, tmp_path):
        """Report function should create a text file and comparison plot."""
        out = str(tmp_path / "report")
        report_text = generate_baseline_report(multi_model_results, out)

        assert isinstance(report_text, str)
        assert len(report_text) > 0
        assert os.path.isfile(os.path.join(out, "baseline_evaluation_report.txt"))
        assert os.path.isfile(os.path.join(out, "baseline_comparison_mae.png"))

    def test_report_with_raw_data(self, tmp_path):
        """Report with actuals/preds should generate per-model plots."""
        np.random.seed(55)
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        actual = np.random.normal(50, 10, 60)
        pred_a = actual + np.random.normal(0, 2, 60)
        pred_b = actual + np.random.normal(1, 4, 60)

        results = {
            "ModelA": compute_metrics(actual, pred_a, "ModelA"),
            "ModelB": compute_metrics(actual, pred_b, "ModelB"),
        }
        out = str(tmp_path / "report_full")
        generate_baseline_report(
            results, out,
            dates_dict={"ModelA": dates, "ModelB": dates},
            actuals_dict={"ModelA": actual, "ModelB": actual},
            preds_dict={"ModelA": pred_a, "ModelB": pred_b},
        )

        # Per-model plots should exist
        assert os.path.isfile(os.path.join(out, "modela_scatter.png"))
        assert os.path.isfile(os.path.join(out, "modelb_scatter.png"))
        assert os.path.isfile(os.path.join(out, "modela_timeseries.png"))
        assert os.path.isfile(os.path.join(out, "modela_residuals_month.png"))

    def test_report_contains_model_names(self, multi_model_results, tmp_path):
        """Report text should include all model names."""
        out = str(tmp_path / "report_names")
        report_text = generate_baseline_report(multi_model_results, out)
        for name in multi_model_results:
            assert name in report_text


# ===========================================================================
# Tests: Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Additional edge-case and robustness tests."""

    def test_single_point_metrics(self):
        """A single data point should yield valid metrics (R2 is NaN)."""
        m = compute_metrics([70.0], [73.0])
        assert m["n"] == 1
        assert m["mae"] == pytest.approx(3.0, abs=1e-10)
        assert m["bias"] == pytest.approx(3.0, abs=1e-10)
        # R2 undefined for single point (ss_tot = 0)
        assert math.isnan(m["r2"])

    def test_all_identical_predictions(self):
        """All predictions the same value (constant model)."""
        actual = np.array([30.0, 50.0, 70.0, 90.0])
        pred = np.array([60.0, 60.0, 60.0, 60.0])
        m = compute_metrics(actual, pred)
        assert m["mae"] == pytest.approx(20.0, abs=1e-10)
        assert m["bias"] == pytest.approx(0.0, abs=1e-10)

    def test_very_large_dataset(self):
        """Should handle large arrays without issue."""
        np.random.seed(7)
        n = 100_000
        actual = np.random.normal(60, 15, n)
        pred = actual + np.random.normal(0, 3, n)
        m = compute_metrics(actual, pred)
        assert m["n"] == n
        assert m["mae"] < 5.0  # Should be around 2.4 for normal(0,3)

    def test_plot_with_single_point(self, tmp_path):
        """Plot functions should not crash with a single data point."""
        actual = np.array([60.0])
        pred = np.array([62.0])
        save_path = str(tmp_path / "single_scatter.png")
        plot_actual_vs_predicted(actual, pred, "SinglePt", save_path)
        assert os.path.isfile(save_path)

    def test_plot_histogram_single_point(self, tmp_path):
        """Histogram should handle a single data point gracefully."""
        actual = np.array([60.0])
        pred = np.array([63.0])
        save_path = str(tmp_path / "single_hist.png")
        plot_residual_histogram(actual, pred, "SinglePt", save_path)
        assert os.path.isfile(save_path)

    def test_seasonal_perfect_predictions(self):
        """Perfect seasonal predictions should have MAE=0 everywhere."""
        dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
        n = len(dates)
        actual = np.random.RandomState(0).normal(60, 10, n)
        pred = actual.copy()
        seasonal = compute_seasonal_metrics(actual, pred, dates)

        for season, sm in seasonal.items():
            assert sm["mae"] == pytest.approx(0.0, abs=1e-10)
            assert sm["rmse"] == pytest.approx(0.0, abs=1e-10)
            assert sm["bias"] == pytest.approx(0.0, abs=1e-10)

    def test_negative_r2_for_bad_model(self):
        """A model worse than predicting the mean should have R2 < 0."""
        actual = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
        # Predictions inversely correlated
        pred = np.array([90.0, 80.0, 70.0, 60.0, 50.0])
        m = compute_metrics(actual, pred)
        assert m["r2"] < 0.0

    def test_format_table_with_nan_values(self):
        """Table formatter should handle NaN metric values gracefully."""
        results = {
            "EmptyModel": {
                "n": 0, "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "bias": float("nan"),
                "within_1f": float("nan"), "within_2f": float("nan"),
                "within_3f": float("nan"), "max_abs_error": float("nan"),
            }
        }
        table = format_metrics_table(results)
        assert "EmptyModel" in table
        assert "N/A" in table
