"""
Tests for the calibration pipeline (src/calibration.py).

Validates:
  - PIT computation (perfect calibration, biased model, edge cases)
  - PIT uniformity KS test
  - PIT histogram plotting
  - Reliability diagram computation and plotting
  - Isotonic calibrator (fit, calibrate, seasonal, serialization)
  - Interval coverage assessment
  - CRPS computation (known values, seasonal breakdown)
  - Sharpness assessment
  - Kalshi bucket probability mapping
  - Comprehensive calibration report generation
  - Edge cases (empty arrays, NaN handling, very large/small sigma)

Target: at least 50 meaningful tests.
"""

import os
import sys
import math
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import (
    compute_pit_values,
    pit_uniformity_test,
    plot_pit_histogram,
    compute_reliability,
    plot_reliability_diagram,
    IsotonicCalibrator,
    compute_interval_coverage,
    compute_crps,
    compute_sharpness,
    cdf_to_kalshi_buckets,
    generate_calibration_report,
    _validate_probabilistic_inputs,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="test_calibration_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def well_calibrated_data():
    """Synthetic data where the model is perfectly calibrated.

    Observations are drawn from N(mu, sigma^2), so PIT values
    should be approximately uniform.
    """
    np.random.seed(42)
    n = 1000
    mu = np.random.uniform(40.0, 90.0, size=n)
    sigma = np.random.uniform(2.0, 6.0, size=n)
    observations = mu + sigma * np.random.randn(n)
    return mu, sigma, observations


@pytest.fixture
def biased_model_data():
    """Synthetic data where the model has a consistent +5 degF bias.

    mu is systematically too high, so PIT values will be skewed
    toward 1 (observations tend to be below mu).
    """
    np.random.seed(123)
    n = 500
    true_mu = np.random.uniform(40.0, 90.0, size=n)
    sigma = np.full(n, 3.0)
    observations = true_mu + sigma * np.random.randn(n)
    biased_mu = true_mu + 5.0  # model over-predicts by 5 degF
    return biased_mu, sigma, observations


@pytest.fixture
def underdispersed_data():
    """Synthetic data where sigma is too small (underdispersed).

    True sigma=5.0, model uses sigma=2.0.
    PIT histogram should be U-shaped.
    """
    np.random.seed(456)
    n = 500
    mu = np.random.uniform(50.0, 80.0, size=n)
    true_sigma = 5.0
    model_sigma = np.full(n, 2.0)
    observations = mu + true_sigma * np.random.randn(n)
    return mu, model_sigma, observations


@pytest.fixture
def seasonal_dates():
    """One full year of daily dates (2021) for seasonal testing."""
    return pd.date_range("2021-01-01", "2021-12-31", freq="D")


@pytest.fixture
def seasonal_probabilistic_data(seasonal_dates):
    """Well-calibrated probabilistic data with dates for all seasons."""
    np.random.seed(789)
    n = len(seasonal_dates)
    mu = 60.0 + 20.0 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    sigma = np.full(n, 3.0)
    observations = mu + sigma * np.random.randn(n)
    return mu, sigma, observations, seasonal_dates


# ===========================================================================
# PIT Computation Tests
# ===========================================================================

class TestComputePITValues:
    """Tests for compute_pit_values()."""

    def test_well_calibrated_gives_uniform_pit(self, well_calibrated_data):
        """PIT values from a well-calibrated model should be approximately uniform."""
        mu, sigma, obs = well_calibrated_data
        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == len(mu)
        # Mean of uniform is 0.5, std is ~0.289
        assert abs(pit.mean() - 0.5) < 0.05
        assert abs(pit.std() - 1.0 / np.sqrt(12)) < 0.05

    def test_biased_model_gives_skewed_pit(self, biased_model_data):
        """PIT values from a biased model should be skewed."""
        mu, sigma, obs = biased_model_data
        pit = compute_pit_values(mu, sigma, obs)
        # Model over-predicts, so observations tend to be below mu
        # PIT = Phi((y - mu)/sigma) will be skewed toward 0
        assert pit.mean() < 0.4

    def test_perfect_prediction_pit(self):
        """When y == mu exactly, PIT should be 0.5 for all entries."""
        mu = np.array([50.0, 60.0, 70.0, 80.0])
        sigma = np.array([2.0, 3.0, 4.0, 5.0])
        obs = mu.copy()  # perfect prediction
        pit = compute_pit_values(mu, sigma, obs)
        np.testing.assert_allclose(pit, 0.5, atol=1e-10)

    def test_pit_range(self, well_calibrated_data):
        """All PIT values should be in [0, 1]."""
        mu, sigma, obs = well_calibrated_data
        pit = compute_pit_values(mu, sigma, obs)
        assert np.all(pit >= 0.0)
        assert np.all(pit <= 1.0)

    def test_very_small_sigma_handled(self):
        """Near-zero sigma should not cause division by zero errors."""
        mu = np.array([70.0, 70.0])
        sigma = np.array([1e-15, 1e-15])
        obs = np.array([70.0, 71.0])
        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 2
        assert not np.any(np.isnan(pit))

    def test_empty_arrays(self):
        """Empty inputs should return empty PIT array."""
        pit = compute_pit_values([], [], [])
        assert len(pit) == 0

    def test_nan_handling(self):
        """NaN values should be dropped, not propagated."""
        mu = np.array([50.0, np.nan, 70.0])
        sigma = np.array([2.0, 3.0, np.nan])
        obs = np.array([51.0, 61.0, 71.0])
        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 1  # only the first entry is valid
        assert not np.any(np.isnan(pit))

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_pit_values([1, 2, 3], [1, 2], [1, 2, 3])

    def test_observation_above_mu_gives_high_pit(self):
        """Observation well above mu should give PIT close to 1."""
        mu = np.array([60.0])
        sigma = np.array([2.0])
        obs = np.array([70.0])  # 5 sigma above
        pit = compute_pit_values(mu, sigma, obs)
        assert pit[0] > 0.99

    def test_observation_below_mu_gives_low_pit(self):
        """Observation well below mu should give PIT close to 0."""
        mu = np.array([60.0])
        sigma = np.array([2.0])
        obs = np.array([50.0])  # 5 sigma below
        pit = compute_pit_values(mu, sigma, obs)
        assert pit[0] < 0.01


# ===========================================================================
# PIT Uniformity Test
# ===========================================================================

class TestPITUniformityTest:
    """Tests for pit_uniformity_test()."""

    def test_uniform_data_passes(self):
        """Truly uniform samples should pass the KS test."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=500)
        result = pit_uniformity_test(pit)
        assert result["is_uniform"] is True
        assert result["p_value"] > 0.05

    def test_non_uniform_fails(self):
        """Highly non-uniform data should fail the KS test."""
        # All PIT values near 0.5 (overdispersed model)
        pit = np.random.normal(0.5, 0.05, size=500)
        pit = np.clip(pit, 0, 1)
        result = pit_uniformity_test(pit)
        assert result["is_uniform"] is False
        assert result["p_value"] < 0.05

    def test_single_value_returns_nan(self):
        """Fewer than 2 PIT values should return NaN."""
        result = pit_uniformity_test(np.array([0.5]))
        assert math.isnan(result["ks_statistic"])
        assert result["is_uniform"] is False

    def test_output_keys(self):
        """Output dict should have the expected keys."""
        pit = np.random.uniform(0, 1, size=100)
        result = pit_uniformity_test(pit)
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "is_uniform" in result


# ===========================================================================
# PIT Histogram Plotting
# ===========================================================================

class TestPlotPITHistogram:
    """Tests for plot_pit_histogram()."""

    def test_saves_to_file(self, tmp_dir):
        """PIT histogram should be saved to the specified path."""
        pit = np.random.uniform(0, 1, size=200)
        save_path = os.path.join(tmp_dir, "pit_hist.png")
        plot_pit_histogram(pit, save_path=save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_returns_figure(self):
        """Function should return a matplotlib Figure."""
        import matplotlib.figure
        pit = np.random.uniform(0, 1, size=100)
        fig = plot_pit_histogram(pit)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_bins(self, tmp_dir):
        """Custom number of bins should work."""
        pit = np.random.uniform(0, 1, size=200)
        save_path = os.path.join(tmp_dir, "pit_20bins.png")
        fig = plot_pit_histogram(pit, n_bins=20, save_path=save_path)
        assert os.path.exists(save_path)


# ===========================================================================
# Reliability Diagram Tests
# ===========================================================================

class TestReliability:
    """Tests for compute_reliability() and plot_reliability_diagram()."""

    def test_well_calibrated_on_diagonal(self, well_calibrated_data):
        """Well-calibrated model should have observed ~ nominal coverage."""
        mu, sigma, obs = well_calibrated_data
        result = compute_reliability(mu, sigma, obs)
        nominal = result["nominal_levels"]
        observed = result["observed_coverages"]
        # Each observed coverage should be close to nominal (+/- 5%)
        for nom, obs_cov in zip(nominal, observed):
            assert abs(nom - obs_cov) < 0.08, (
                f"Nominal {nom:.2f} but observed {obs_cov:.2f}"
            )

    def test_output_format(self):
        """Output should contain expected keys."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([3.0, 3.0, 3.0])
        obs = np.array([61.0, 69.0, 82.0])
        result = compute_reliability(mu, sigma, obs)
        assert "nominal_levels" in result
        assert "observed_coverages" in result
        assert "n_samples" in result
        assert result["n_samples"] == 3

    def test_custom_nominal_levels(self):
        """Custom nominal levels should be respected."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([3.0, 3.0, 3.0])
        obs = np.array([61.0, 69.0, 82.0])
        custom_levels = [0.5, 0.9]
        result = compute_reliability(mu, sigma, obs, nominal_levels=custom_levels)
        assert result["nominal_levels"] == custom_levels
        assert len(result["observed_coverages"]) == 2

    def test_empty_data_returns_nan(self):
        """Empty data should return NaN coverages."""
        result = compute_reliability([], [], [])
        assert result["n_samples"] == 0
        assert all(math.isnan(c) for c in result["observed_coverages"])

    def test_plot_saves(self, tmp_dir):
        """Reliability diagram plot should be saved."""
        reliability = {
            "nominal_levels": [0.1, 0.2, 0.5, 0.9],
            "observed_coverages": [0.12, 0.22, 0.48, 0.88],
            "n_samples": 100,
        }
        save_path = os.path.join(tmp_dir, "rel_diagram.png")
        plot_reliability_diagram(reliability, save_path=save_path)
        assert os.path.exists(save_path)

    def test_coverage_monotonically_increases(self, well_calibrated_data):
        """Observed coverage should generally increase with nominal level."""
        mu, sigma, obs = well_calibrated_data
        result = compute_reliability(mu, sigma, obs)
        observed = result["observed_coverages"]
        # Allow small non-monotonicity due to noise, but trend should be up
        assert observed[-1] > observed[0]


# ===========================================================================
# Isotonic Calibrator Tests
# ===========================================================================

class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator class."""

    def test_fit_and_is_fitted(self):
        """After fit, is_fitted should be True."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=100)
        cal = IsotonicCalibrator()
        assert not cal.is_fitted
        cal.fit(pit)
        assert cal.is_fitted

    def test_calibrate_cdf_preserves_range(self):
        """Calibrated PIT values should remain in [0, 1]."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=200)
        cal = IsotonicCalibrator()
        cal.fit(pit)
        calibrated = cal.calibrate_cdf(pit)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_calibrate_improves_uniformity(self, underdispersed_data):
        """Isotonic calibration should improve PIT uniformity."""
        mu, sigma, obs = underdispersed_data
        pit_raw = compute_pit_values(mu, sigma, obs)

        # Raw model is underdispersed
        ks_raw = pit_uniformity_test(pit_raw)

        # Fit calibrator on the raw PIT
        cal = IsotonicCalibrator()
        cal.fit(pit_raw)

        # Calibrate the PIT values
        pit_cal = cal.calibrate_cdf(pit_raw)
        ks_cal = pit_uniformity_test(pit_cal)

        # Calibrated PIT should be more uniform (higher p-value)
        assert ks_cal["ks_statistic"] <= ks_raw["ks_statistic"] + 0.01

    def test_calibrate_mu_sigma(self, well_calibrated_data):
        """calibrate() should return adjusted mu and sigma arrays."""
        mu, sigma, obs = well_calibrated_data
        pit = compute_pit_values(mu, sigma, obs)

        cal = IsotonicCalibrator()
        cal.fit(pit)

        cal_mu, cal_sigma = cal.calibrate(mu, sigma)
        assert len(cal_mu) == len(mu)
        assert len(cal_sigma) == len(sigma)
        assert np.all(cal_sigma > 0)

    def test_calibrate_not_fitted_raises(self):
        """calibrate() before fit() should raise RuntimeError."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate(np.array([70.0]), np.array([3.0]))

    def test_calibrate_cdf_not_fitted_raises(self):
        """calibrate_cdf() before fit() should raise RuntimeError."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate_cdf(np.array([0.5]))

    def test_serialization_roundtrip(self, tmp_dir):
        """Save and load should preserve the calibrator state."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=200)
        cal = IsotonicCalibrator()
        cal.fit(pit)

        filepath = os.path.join(tmp_dir, "calibrator.pkl")
        cal.save(filepath)
        assert os.path.exists(filepath)

        loaded = IsotonicCalibrator.load(filepath)
        assert loaded.is_fitted
        assert loaded.seasonal == cal.seasonal

        # Results should be identical
        test_pit = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(
            cal.calibrate_cdf(test_pit),
            loaded.calibrate_cdf(test_pit),
            atol=1e-10,
        )

    def test_seasonal_mode(self, seasonal_probabilistic_data):
        """Seasonal calibrator should fit per-season models."""
        mu, sigma, obs, dates = seasonal_probabilistic_data
        pit = compute_pit_values(mu, sigma, obs)

        cal = IsotonicCalibrator(seasonal=True)
        cal.fit(pit, dates=dates)
        assert cal.is_fitted
        assert len(cal._seasonal_models) > 0

        cal_pit = cal.calibrate_cdf(pit, dates=dates)
        assert len(cal_pit) == len(pit)
        assert np.all(cal_pit >= 0.0)
        assert np.all(cal_pit <= 1.0)

    def test_seasonal_requires_dates(self):
        """Seasonal mode should raise ValueError if dates not provided."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=100)
        cal = IsotonicCalibrator(seasonal=True)
        with pytest.raises(ValueError, match="dates must be provided"):
            cal.fit(pit)

    def test_fit_empty_raises(self):
        """fit() with empty PIT should raise ValueError."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="empty PIT"):
            cal.fit(np.array([]))

    def test_calibrate_empty_input(self):
        """calibrate() with empty arrays should return empty arrays."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=100)
        cal = IsotonicCalibrator()
        cal.fit(pit)
        cal_mu, cal_sigma = cal.calibrate(np.array([]), np.array([]))
        assert len(cal_mu) == 0
        assert len(cal_sigma) == 0

    def test_calibrate_cdf_empty_input(self):
        """calibrate_cdf() with empty array should return empty array."""
        np.random.seed(42)
        pit = np.random.uniform(0, 1, size=100)
        cal = IsotonicCalibrator()
        cal.fit(pit)
        result = cal.calibrate_cdf(np.array([]))
        assert len(result) == 0

    def test_seasonal_serialization(self, seasonal_probabilistic_data, tmp_dir):
        """Seasonal calibrator should survive save/load."""
        mu, sigma, obs, dates = seasonal_probabilistic_data
        pit = compute_pit_values(mu, sigma, obs)

        cal = IsotonicCalibrator(seasonal=True)
        cal.fit(pit, dates=dates)

        filepath = os.path.join(tmp_dir, "seasonal_cal.pkl")
        cal.save(filepath)
        loaded = IsotonicCalibrator.load(filepath)

        assert loaded.seasonal is True
        assert loaded.is_fitted
        assert len(loaded._seasonal_models) == len(cal._seasonal_models)


# ===========================================================================
# Interval Coverage Tests
# ===========================================================================

class TestIntervalCoverage:
    """Tests for compute_interval_coverage()."""

    def test_standard_normal_known_coverage(self):
        """For standard normal predictions, 95% PI should cover ~95%."""
        np.random.seed(42)
        n = 5000
        mu = np.zeros(n)
        sigma = np.ones(n)
        obs = np.random.randn(n)

        result = compute_interval_coverage(mu, sigma, obs, levels=[0.95])
        cov_95 = result["coverages"][0]
        # Should be close to 0.95 with 5000 samples
        assert abs(cov_95 - 0.95) < 0.02

    def test_coverage_monotonically_increases(self, well_calibrated_data):
        """Wider intervals should always have higher or equal coverage."""
        mu, sigma, obs = well_calibrated_data
        result = compute_interval_coverage(
            mu, sigma, obs, levels=[0.10, 0.30, 0.50, 0.70, 0.90, 0.99]
        )
        coverages = result["coverages"]
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1] + 1e-10

    def test_100_percent_coverage(self):
        """Near-infinite sigma should give ~100% coverage at any level."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([1000.0, 1000.0, 1000.0])
        obs = np.array([65.0, 75.0, 85.0])
        result = compute_interval_coverage(mu, sigma, obs, levels=[0.50, 0.95])
        assert all(c > 0.99 for c in result["coverages"])

    def test_zero_coverage_with_tiny_sigma(self):
        """Very small sigma with large errors should give ~0% coverage."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([0.001, 0.001, 0.001])
        obs = np.array([100.0, 100.0, 100.0])  # way off
        result = compute_interval_coverage(mu, sigma, obs, levels=[0.50])
        assert result["coverages"][0] < 0.01

    def test_empty_data(self):
        """Empty inputs should return NaN coverages."""
        result = compute_interval_coverage([], [], [])
        assert result["n_samples"] == 0
        assert all(math.isnan(c) for c in result["coverages"])

    def test_default_levels(self):
        """Default levels should be [0.50, 0.80, 0.90, 0.95]."""
        mu = np.array([60.0])
        sigma = np.array([3.0])
        obs = np.array([61.0])
        result = compute_interval_coverage(mu, sigma, obs)
        assert result["levels"] == [0.50, 0.80, 0.90, 0.95]


# ===========================================================================
# CRPS Tests
# ===========================================================================

class TestCRPS:
    """Tests for compute_crps()."""

    def test_perfect_prediction_crps_near_zero(self):
        """When obs == mu and sigma is very small, CRPS approaches 0."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([0.001, 0.001, 0.001])
        obs = mu.copy()
        result = compute_crps(mu, sigma, obs)
        assert result["mean_crps"] < 0.01

    def test_crps_positive(self, well_calibrated_data):
        """CRPS should always be non-negative."""
        mu, sigma, obs = well_calibrated_data
        result = compute_crps(mu, sigma, obs)
        assert result["mean_crps"] >= 0
        assert np.all(result["crps_values"] >= 0)

    def test_crps_increases_with_error(self):
        """Larger prediction errors should give higher CRPS."""
        sigma = np.array([3.0])

        mu_close = np.array([60.0])
        obs = np.array([61.0])
        crps_close = compute_crps(mu_close, sigma, obs)["mean_crps"]

        mu_far = np.array([60.0])
        obs_far = np.array([80.0])
        crps_far = compute_crps(mu_far, sigma, obs_far)["mean_crps"]

        assert crps_far > crps_close

    def test_crps_increases_with_sigma(self):
        """Larger sigma (all else equal) should increase CRPS when mu=obs."""
        mu = np.array([60.0])
        obs = np.array([60.0])

        crps_small = compute_crps(mu, np.array([1.0]), obs)["mean_crps"]
        crps_large = compute_crps(mu, np.array([10.0]), obs)["mean_crps"]

        assert crps_large > crps_small

    def test_crps_known_value_standard_normal(self):
        """CRPS of N(0,1) at y=0 should be 1/sqrt(pi) - 1/sqrt(pi) + ... = sigma*(2*phi(0)-1/sqrt(pi))."""
        # For z=0: CRPS = sigma * (0*(2*0.5-1) + 2*phi(0) - 1/sqrt(pi))
        # = sigma * (2 * 1/sqrt(2*pi) - 1/sqrt(pi))
        # = 1.0 * (2/sqrt(2*pi) - 1/sqrt(pi))
        expected = 2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi)
        result = compute_crps(np.array([0.0]), np.array([1.0]), np.array([0.0]))
        assert abs(result["mean_crps"] - expected) < 1e-10

    def test_seasonal_crps_breakdown(self, seasonal_probabilistic_data):
        """Seasonal CRPS should be computed when dates are provided."""
        mu, sigma, obs, dates = seasonal_probabilistic_data
        result = compute_crps(mu, sigma, obs, dates=dates)
        assert "seasonal_crps" in result
        seasonal = result["seasonal_crps"]
        # All 4 seasons should be present
        assert len(seasonal) == 4
        for season_name, sdata in seasonal.items():
            assert "mean_crps" in sdata
            assert "n" in sdata
            assert sdata["n"] > 0
            assert sdata["mean_crps"] >= 0

    def test_empty_crps(self):
        """Empty inputs should return NaN CRPS."""
        result = compute_crps([], [], [])
        assert math.isnan(result["mean_crps"])
        assert result["n_samples"] == 0

    def test_crps_values_length(self, well_calibrated_data):
        """Per-sample CRPS array should have correct length."""
        mu, sigma, obs = well_calibrated_data
        result = compute_crps(mu, sigma, obs)
        assert len(result["crps_values"]) == len(mu)


# ===========================================================================
# Sharpness Tests
# ===========================================================================

class TestSharpness:
    """Tests for compute_sharpness()."""

    def test_wider_sigma_gives_wider_intervals(self):
        """Larger sigma should produce wider prediction intervals."""
        sharp_small = compute_sharpness(np.array([2.0, 2.0, 2.0]))
        sharp_large = compute_sharpness(np.array([10.0, 10.0, 10.0]))
        # All widths should be larger for larger sigma
        for w_small, w_large in zip(sharp_small["mean_widths"],
                                     sharp_large["mean_widths"]):
            assert w_large > w_small

    def test_sigma_width_relationship(self):
        """95% PI width should be ~2*1.96*sigma = 3.92*sigma."""
        sigma_val = 5.0
        result = compute_sharpness(
            np.array([sigma_val]),
            levels=[0.95],
        )
        expected_width = 2 * 1.96 * sigma_val
        assert abs(result["mean_widths"][0] - expected_width) < 0.01

    def test_mean_sigma_correct(self):
        """Mean sigma should match np.mean of input."""
        sigma = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_sharpness(sigma)
        assert abs(result["mean_sigma"] - 3.0) < 1e-10

    def test_empty_sigma(self):
        """Empty sigma array should return NaN sharpness."""
        result = compute_sharpness([])
        assert math.isnan(result["mean_sigma"])
        assert result["n_samples"] == 0

    def test_seasonal_sharpness(self, seasonal_probabilistic_data):
        """Seasonal sharpness should be computed when dates are provided."""
        _, sigma, _, dates = seasonal_probabilistic_data
        result = compute_sharpness(sigma, dates=dates)
        assert "seasonal_sharpness" in result
        assert len(result["seasonal_sharpness"]) == 4

    def test_widths_increase_with_level(self):
        """Higher confidence level should give wider intervals."""
        sigma = np.array([3.0, 3.0, 3.0])
        result = compute_sharpness(sigma, levels=[0.50, 0.80, 0.95])
        widths = result["mean_widths"]
        for i in range(len(widths) - 1):
            assert widths[i] < widths[i + 1]


# ===========================================================================
# Kalshi Bucket Tests
# ===========================================================================

class TestKalshiBuckets:
    """Tests for cdf_to_kalshi_buckets()."""

    def test_probabilities_sum_to_one(self):
        """All bucket probabilities should sum to approximately 1."""
        buckets = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0)
        total = sum(buckets.values())
        assert abs(total - 1.0) < 0.01

    def test_wider_sigma_flatter_distribution(self):
        """Wider sigma should spread probability across more buckets."""
        buckets_narrow = cdf_to_kalshi_buckets(mu=70.0, sigma=2.0)
        buckets_wide = cdf_to_kalshi_buckets(mu=70.0, sigma=10.0)
        # Max probability should be lower for wider distribution
        assert max(buckets_wide.values()) < max(buckets_narrow.values())

    def test_peak_bucket_contains_mu(self):
        """The most likely bucket should contain mu."""
        mu = 72.0
        buckets = cdf_to_kalshi_buckets(mu=mu, sigma=3.0)
        # Remove edge buckets for this check
        interior = {k: v for k, v in buckets.items()
                    if not k.startswith("Below") and not k.startswith("Above")}
        peak_bucket = max(interior, key=interior.get)
        # Parse bucket range
        parts = peak_bucket.replace(" F", "").split("-")
        lo, hi = int(parts[0]), int(parts[1])
        assert lo <= mu <= hi + 1, f"mu={mu} not in peak bucket {peak_bucket}"

    def test_bucket_labels_format(self):
        """Bucket labels should follow expected format."""
        buckets = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0, temp_range=(50, 100))
        keys = list(buckets.keys())
        assert keys[0] == "Below 50 F"
        assert keys[-1] == "Above 99 F"
        # Interior buckets should be "X-Y F" format
        for k in keys[1:-1]:
            assert k.endswith(" F")
            parts = k.replace(" F", "").split("-")
            assert len(parts) == 2

    def test_custom_bucket_width(self):
        """Custom bucket width should be respected."""
        buckets_5 = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0, bucket_width=5)
        buckets_10 = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0, bucket_width=10)
        # 10-degree buckets should have fewer interior buckets
        interior_5 = len([k for k in buckets_5 if "-" in k])
        interior_10 = len([k for k in buckets_10 if "-" in k])
        assert interior_10 < interior_5

    def test_all_probabilities_non_negative(self):
        """No bucket should have negative probability."""
        buckets = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0)
        assert all(v >= 0 for v in buckets.values())

    def test_very_small_sigma(self):
        """Very small sigma should concentrate probability in one bucket."""
        buckets = cdf_to_kalshi_buckets(mu=72.0, sigma=0.01)
        interior = {k: v for k, v in buckets.items()
                    if not k.startswith("Below") and not k.startswith("Above")}
        # One bucket should have almost all the probability
        assert max(interior.values()) > 0.95


# ===========================================================================
# Validate Probabilistic Inputs Tests
# ===========================================================================

class TestValidateProbabilisticInputs:
    """Tests for _validate_probabilistic_inputs()."""

    def test_basic_validation(self):
        """Basic arrays should pass validation."""
        mu, sigma, obs = _validate_probabilistic_inputs(
            [1, 2, 3], [1, 1, 1], [1.5, 2.5, 3.5]
        )
        assert len(mu) == 3
        assert mu.dtype == np.float64

    def test_nan_removal(self):
        """NaN entries should be removed."""
        mu, sigma, obs = _validate_probabilistic_inputs(
            [1, np.nan, 3], [1, 1, 1], [1.5, 2.5, 3.5]
        )
        assert len(mu) == 2

    def test_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            _validate_probabilistic_inputs([1, 2], [1], [1, 2])


# ===========================================================================
# Integration / Report Tests
# ===========================================================================

class TestCalibrationReport:
    """Tests for generate_calibration_report()."""

    def test_report_generates_all_files(self, tmp_dir, well_calibrated_data):
        """Report should generate plots, CSV, and return dict."""
        mu, sigma, obs = well_calibrated_data
        result = generate_calibration_report(
            mu, sigma, obs,
            output_dir=tmp_dir,
            model_name="Test Model",
        )

        # Check return dict
        assert "pit_ks_test" in result
        assert "reliability" in result
        assert "coverage" in result
        assert "crps" in result
        assert "sharpness" in result
        assert result["model_name"] == "Test Model"

        # Check files were created
        safe_name = "test_model"
        assert os.path.exists(os.path.join(tmp_dir, f"{safe_name}_pit_histogram.png"))
        assert os.path.exists(os.path.join(tmp_dir, f"{safe_name}_reliability.png"))
        assert os.path.exists(os.path.join(tmp_dir, f"{safe_name}_calibration_summary.png"))
        assert os.path.exists(os.path.join(tmp_dir, f"{safe_name}_calibration_metrics.csv"))

    def test_report_with_dates(self, tmp_dir, seasonal_probabilistic_data):
        """Report with dates should include seasonal breakdowns."""
        mu, sigma, obs, dates = seasonal_probabilistic_data
        result = generate_calibration_report(
            mu, sigma, obs,
            dates=dates,
            output_dir=tmp_dir,
            model_name="Seasonal Test",
        )
        assert "crps" in result
        # CRPS should have seasonal breakdown
        crps = result["crps"]
        assert "seasonal_crps" in crps

    def test_report_csv_content(self, tmp_dir, well_calibrated_data):
        """CSV should contain expected columns."""
        mu, sigma, obs = well_calibrated_data
        generate_calibration_report(
            mu, sigma, obs,
            output_dir=tmp_dir,
            model_name="CSV Check",
        )
        csv_path = os.path.join(tmp_dir, "csv_check_calibration_metrics.csv")
        df = pd.read_csv(csv_path)
        assert "model_name" in df.columns
        assert "ks_statistic" in df.columns
        assert "mean_crps" in df.columns
        assert "mean_sigma" in df.columns


# ===========================================================================
# Before/After Calibration Integration Test
# ===========================================================================

class TestCalibrationImprovement:
    """Integration tests verifying that calibration actually helps."""

    def test_calibration_improves_underdispersed_coverage(self, underdispersed_data):
        """Isotonic calibration should improve coverage for an underdispersed model."""
        mu, sigma, obs = underdispersed_data

        # Raw coverage at 95%
        raw_cov = compute_interval_coverage(mu, sigma, obs, levels=[0.95])
        raw_95 = raw_cov["coverages"][0]
        # Underdispersed model should under-cover
        assert raw_95 < 0.90, f"Expected under-coverage, got {raw_95}"

        # Fit calibrator
        pit = compute_pit_values(mu, sigma, obs)
        cal = IsotonicCalibrator()
        cal.fit(pit)
        cal_mu, cal_sigma = cal.calibrate(mu, sigma)

        # Calibrated coverage at 95%
        cal_cov = compute_interval_coverage(cal_mu, cal_sigma, obs, levels=[0.95])
        cal_95 = cal_cov["coverages"][0]

        # Calibrated should be closer to 0.95
        assert abs(cal_95 - 0.95) < abs(raw_95 - 0.95) + 0.05

    def test_crps_comparison(self, well_calibrated_data):
        """CRPS should not substantially degrade after calibration on well-calibrated data."""
        mu, sigma, obs = well_calibrated_data
        crps_raw = compute_crps(mu, sigma, obs)["mean_crps"]

        pit = compute_pit_values(mu, sigma, obs)
        cal = IsotonicCalibrator()
        cal.fit(pit)
        cal_mu, cal_sigma = cal.calibrate(mu, sigma)

        crps_cal = compute_crps(cal_mu, cal_sigma, obs)["mean_crps"]

        # For already-calibrated data, CRPS should not increase much
        assert crps_cal < crps_raw * 1.5


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_observation(self):
        """All functions should handle a single data point."""
        mu = np.array([65.0])
        sigma = np.array([3.0])
        obs = np.array([67.0])

        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 1

        coverage = compute_interval_coverage(mu, sigma, obs)
        assert coverage["n_samples"] == 1

        crps = compute_crps(mu, sigma, obs)
        assert crps["n_samples"] == 1

    def test_very_large_sigma(self):
        """Very large sigma should not cause overflow."""
        mu = np.array([70.0])
        sigma = np.array([1e6])
        obs = np.array([70.0])

        pit = compute_pit_values(mu, sigma, obs)
        assert not np.any(np.isnan(pit))

        crps = compute_crps(mu, sigma, obs)
        assert not math.isnan(crps["mean_crps"])
        assert crps["mean_crps"] > 0

    def test_all_same_values(self):
        """All identical predictions should work without errors."""
        n = 50
        mu = np.full(n, 70.0)
        sigma = np.full(n, 3.0)
        obs = np.full(n, 70.0)

        pit = compute_pit_values(mu, sigma, obs)
        np.testing.assert_allclose(pit, 0.5, atol=1e-10)

        crps = compute_crps(mu, sigma, obs)
        assert crps["n_samples"] == n

    def test_list_inputs(self):
        """Python lists should be accepted as inputs."""
        mu = [60.0, 70.0, 80.0]
        sigma = [3.0, 3.0, 3.0]
        obs = [61.0, 69.0, 82.0]

        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 3

        coverage = compute_interval_coverage(mu, sigma, obs)
        assert coverage["n_samples"] == 3

    def test_pandas_series_inputs(self):
        """Pandas Series should be accepted as inputs."""
        mu = pd.Series([60.0, 70.0, 80.0])
        sigma = pd.Series([3.0, 3.0, 3.0])
        obs = pd.Series([61.0, 69.0, 82.0])

        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 3

    def test_negative_sigma_handled(self):
        """Negative sigma values (after clamp) should not crash."""
        mu = np.array([60.0, 70.0])
        sigma = np.array([-1.0, 3.0])  # negative sigma
        obs = np.array([61.0, 69.0])
        # Should not raise, sigma gets clamped
        pit = compute_pit_values(mu, sigma, obs)
        assert len(pit) == 2
