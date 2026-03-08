"""
Tests for Phase C calibration enhancements (src/calibration.py).

Validates:
  - compute_seasonal_reliability() function
  - validate_bucket_probabilities() function
  - NLL computation in generate_calibration_report()
  - Seasonal reliability in generate_calibration_report()
  - Bucket validation wired into cdf_to_kalshi_buckets()
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import (
    compute_seasonal_reliability,
    validate_bucket_probabilities,
    cdf_to_kalshi_buckets,
    generate_calibration_report,
    compute_reliability,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_phase_c_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def seasonal_data():
    """Full year of well-calibrated data with seasonal pattern."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    n = len(dates)
    mu = 60.0 + 20.0 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    sigma = np.full(n, 3.0)
    observations = mu + sigma * np.random.randn(n)
    return mu, sigma, observations, dates


@pytest.fixture
def two_year_data():
    """Two years of data for robust seasonal testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    n = len(dates)
    mu = 65.0 + 15.0 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    sigma = np.full(n, 4.0)
    observations = mu + sigma * np.random.randn(n)
    return mu, sigma, observations, dates


# ===========================================================================
# compute_seasonal_reliability() Tests
# ===========================================================================

class TestComputeSeasonalReliability:
    """Tests for compute_seasonal_reliability()."""

    def test_returns_all_four_seasons(self, seasonal_data):
        """With a full year of data, all 4 seasons should be present."""
        mu, sigma, obs, dates = seasonal_data
        result = compute_seasonal_reliability(mu, sigma, obs, dates)
        assert len(result) == 4

    def test_each_season_has_reliability_format(self, seasonal_data):
        """Each season's output should have reliability diagram keys."""
        mu, sigma, obs, dates = seasonal_data
        result = compute_seasonal_reliability(mu, sigma, obs, dates)
        for season_name, rel in result.items():
            assert "nominal_levels" in rel
            assert "observed_coverages" in rel
            assert "n_samples" in rel
            assert rel["n_samples"] > 0

    def test_skips_season_below_min_samples(self):
        """Season with fewer than min_samples should be skipped."""
        np.random.seed(42)
        # Only January dates (Winter) - 10 days
        dates = pd.date_range("2021-01-01", "2021-01-10", freq="D")
        n = len(dates)
        mu = np.full(n, 40.0)
        sigma = np.full(n, 3.0)
        obs = mu + sigma * np.random.randn(n)
        # min_samples=20 should skip Winter (only 10 samples)
        result = compute_seasonal_reliability(mu, sigma, obs, dates, min_samples=20)
        assert len(result) == 0

    def test_custom_nominal_levels(self, seasonal_data):
        """Custom nominal levels should be passed through."""
        mu, sigma, obs, dates = seasonal_data
        custom_levels = [0.5, 0.9]
        result = compute_seasonal_reliability(
            mu, sigma, obs, dates, nominal_levels=custom_levels,
        )
        for season_name, rel in result.items():
            assert rel["nominal_levels"] == custom_levels
            assert len(rel["observed_coverages"]) == 2

    def test_empty_data_returns_empty(self):
        """Empty inputs should return empty dict."""
        result = compute_seasonal_reliability([], [], [], pd.DatetimeIndex([]))
        assert result == {}

    def test_mismatched_date_length_returns_empty(self):
        """Mismatched date array length should return empty dict."""
        mu = np.array([60.0, 70.0, 80.0])
        sigma = np.array([3.0, 3.0, 3.0])
        obs = np.array([61.0, 69.0, 82.0])
        dates = pd.date_range("2021-01-01", periods=5, freq="D")  # mismatch
        result = compute_seasonal_reliability(mu, sigma, obs, dates)
        assert result == {}

    def test_well_calibrated_close_to_diagonal(self, two_year_data):
        """Well-calibrated data should have observed ~ nominal for each season."""
        mu, sigma, obs, dates = two_year_data
        result = compute_seasonal_reliability(mu, sigma, obs, dates)
        for season_name, rel in result.items():
            for nom, obs_cov in zip(rel["nominal_levels"], rel["observed_coverages"]):
                assert abs(nom - obs_cov) < 0.15, (
                    f"{season_name}: nominal {nom:.2f} but observed {obs_cov:.2f}"
                )


# ===========================================================================
# validate_bucket_probabilities() Tests
# ===========================================================================

class TestValidateBucketProbabilities:
    """Tests for validate_bucket_probabilities()."""

    def test_valid_buckets_pass(self):
        """Properly normalized buckets should pass validation."""
        buckets = {"A": 0.3, "B": 0.5, "C": 0.2}
        result = validate_bucket_probabilities(buckets)
        assert result["is_valid"] is True
        assert len(result["warnings"]) == 0
        assert abs(result["prob_sum"] - 1.0) < 0.01
        assert result["max_negative"] == 0.0
        assert result["monotonicity_violations"] == 0

    def test_negative_probability_detected(self):
        """Negative probabilities should be flagged."""
        buckets = {"A": 0.6, "B": -0.1, "C": 0.5}
        result = validate_bucket_probabilities(buckets)
        assert result["is_valid"] is False
        assert result["max_negative"] < 0
        assert any("negative" in w.lower() for w in result["warnings"])

    def test_sum_deviation_detected(self):
        """Sum significantly different from 1.0 should be flagged."""
        buckets = {"A": 0.3, "B": 0.3, "C": 0.3}  # sum = 0.9
        result = validate_bucket_probabilities(buckets)
        assert result["is_valid"] is False
        assert result["sum_deviation"] > 0.05
        assert any("sum" in w.lower() for w in result["warnings"])

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        buckets = {"A": 0.33, "B": 0.33, "C": 0.33}  # sum = 0.99
        result_strict = validate_bucket_probabilities(buckets, tolerance=0.001)
        result_loose = validate_bucket_probabilities(buckets, tolerance=0.02)
        assert result_strict["is_valid"] is False
        assert result_loose["is_valid"] is True

    def test_perfect_sum(self):
        """Probabilities summing to exactly 1.0 should pass."""
        buckets = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        result = validate_bucket_probabilities(buckets)
        assert result["is_valid"] is True
        assert result["prob_sum"] == 1.0

    def test_monotonicity_always_passes_for_positive_probs(self):
        """CDF cumsum of non-negative values is always non-decreasing."""
        buckets = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
        result = validate_bucket_probabilities(buckets)
        assert result["monotonicity_violations"] == 0

    def test_empty_buckets(self):
        """Empty bucket dict should fail sum check."""
        result = validate_bucket_probabilities({})
        # Sum is 0, deviation from 1.0 is 1.0
        assert result["is_valid"] is False

    def test_single_bucket(self):
        """Single bucket with prob=1.0 should pass."""
        result = validate_bucket_probabilities({"Only": 1.0})
        assert result["is_valid"] is True

    def test_output_keys(self):
        """Output should have all expected keys."""
        result = validate_bucket_probabilities({"A": 0.5, "B": 0.5})
        for key in ["is_valid", "prob_sum", "max_negative", "sum_deviation",
                     "monotonicity_violations", "warnings"]:
            assert key in result


# ===========================================================================
# cdf_to_kalshi_buckets() Validation Integration
# ===========================================================================

class TestBucketValidationIntegration:
    """Tests that cdf_to_kalshi_buckets() runs validation."""

    def test_normal_buckets_pass_validation(self):
        """Standard call should produce valid buckets."""
        buckets = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0)
        result = validate_bucket_probabilities(buckets)
        assert result["is_valid"] is True

    def test_buckets_sum_near_one(self):
        """Bucket probabilities should sum very close to 1.0."""
        buckets = cdf_to_kalshi_buckets(mu=70.0, sigma=5.0)
        total = sum(buckets.values())
        assert abs(total - 1.0) < 0.001

    def test_extreme_mu_still_valid(self):
        """Extreme mu should still produce valid buckets."""
        for mu in [0.0, 120.0, -10.0, 150.0]:
            buckets = cdf_to_kalshi_buckets(mu=mu, sigma=5.0)
            result = validate_bucket_probabilities(buckets)
            # Sum should still be ~1.0 (edge buckets capture mass)
            assert abs(result["prob_sum"] - 1.0) < 0.01


# ===========================================================================
# NLL in generate_calibration_report() Tests
# ===========================================================================

class TestCalibrationReportNLL:
    """Tests for NLL computation in generate_calibration_report()."""

    def test_report_contains_nll(self, tmp_dir):
        """Report should contain NLL results."""
        np.random.seed(42)
        n = 100
        mu = np.random.uniform(60, 80, n)
        sigma = np.full(n, 3.0)
        obs = mu + sigma * np.random.randn(n)
        report = generate_calibration_report(
            mu, sigma, obs, output_dir=tmp_dir, model_name="NLLTest",
        )
        assert "nll" in report
        assert "mean_nll" in report["nll"]
        assert "median_nll" in report["nll"]
        assert report["nll"]["n_samples"] == n
        assert report["nll"]["mean_nll"] > 0

    def test_report_nll_seasonal_with_dates(self, tmp_dir, seasonal_data):
        """NLL should have seasonal breakdown when dates are provided."""
        mu, sigma, obs, dates = seasonal_data
        report = generate_calibration_report(
            mu, sigma, obs, dates=dates,
            output_dir=tmp_dir, model_name="NLLSeasonal",
        )
        nll = report["nll"]
        assert "seasonal" in nll
        assert len(nll["seasonal"]) == 4  # all 4 seasons

    def test_nll_known_value(self, tmp_dir):
        """NLL for obs==mu and sigma=1 should equal 0.5*log(2*pi)."""
        n = 50
        mu = np.full(n, 70.0)
        sigma = np.ones(n)
        obs = mu.copy()
        expected_nll = 0.5 * np.log(2 * np.pi)
        report = generate_calibration_report(
            mu, sigma, obs, output_dir=tmp_dir, model_name="NLLKnown",
        )
        assert abs(report["nll"]["mean_nll"] - expected_nll) < 1e-10

    def test_csv_contains_nll_columns(self, tmp_dir):
        """CSV output should contain NLL columns."""
        np.random.seed(42)
        n = 100
        mu = np.random.uniform(60, 80, n)
        sigma = np.full(n, 3.0)
        obs = mu + sigma * np.random.randn(n)
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        generate_calibration_report(
            mu, sigma, obs, dates=dates,
            output_dir=tmp_dir, model_name="CSVNll",
        )
        csv_path = os.path.join(tmp_dir, "csvnll_calibration_metrics.csv")
        df = pd.read_csv(csv_path)
        assert "mean_nll" in df.columns
        assert "median_nll" in df.columns


# ===========================================================================
# Seasonal reliability in generate_calibration_report()
# ===========================================================================

class TestCalibrationReportSeasonalReliability:
    """Tests for seasonal reliability in generate_calibration_report()."""

    def test_report_contains_seasonal_reliability(self, tmp_dir, seasonal_data):
        """Report with dates should include seasonal_reliability."""
        mu, sigma, obs, dates = seasonal_data
        report = generate_calibration_report(
            mu, sigma, obs, dates=dates,
            output_dir=tmp_dir, model_name="SeasonalRel",
        )
        assert "seasonal_reliability" in report
        assert len(report["seasonal_reliability"]) == 4

    def test_report_without_dates_no_seasonal_reliability(self, tmp_dir):
        """Report without dates should not include seasonal_reliability."""
        np.random.seed(42)
        n = 100
        mu = np.random.uniform(60, 80, n)
        sigma = np.full(n, 3.0)
        obs = mu + sigma * np.random.randn(n)
        report = generate_calibration_report(
            mu, sigma, obs, output_dir=tmp_dir, model_name="NoSeason",
        )
        assert "seasonal_reliability" not in report
