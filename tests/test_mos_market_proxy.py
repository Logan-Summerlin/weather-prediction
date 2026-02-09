"""
Comprehensive tests for MOSMarketProxy.

Tests cover initialization, fit, predict, bracket probabilities,
fallback behavior, diagnostics, and integration with real MOS data.
"""

import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.mos_market_proxy import MOSMarketProxy


# =========================================================================
# Fixtures
# =========================================================================

def _make_mos_df(
    start_date: str = "2020-01-01",
    n_days: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic MOS forecast DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # Simulate a seasonal temperature pattern
    doy = dates.dayofyear
    seasonal = 35 + 25 * np.sin(2 * np.pi * (doy - 100) / 365)

    # GFS and NAM forecasts with small noise
    gfs = seasonal + rng.normal(0, 3.0, n_days)
    nam = seasonal + rng.normal(0, 3.5, n_days)

    # Randomly drop a few NAM values
    nam_mask = rng.random(n_days) > 0.05
    nam_vals = np.where(nam_mask, nam, np.nan)

    return pd.DataFrame({
        "date": dates,
        "gfs_mos_tmax_f": gfs.round(1),
        "nam_mos_tmax_f": pd.array(nam_vals.round(1), dtype=pd.Float64Dtype()),
        "mos_ensemble_tmax_f": np.nanmean(
            np.column_stack([gfs, np.where(nam_mask, nam, np.nan)]), axis=1
        ).round(1),
    })


def _make_actual_df(
    start_date: str = "2020-01-01",
    n_days: int = 1000,
    seed: int = 99,
) -> pd.DataFrame:
    """Create a synthetic actual TMAX DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    doy = dates.dayofyear
    seasonal = 35 + 25 * np.sin(2 * np.pi * (doy - 100) / 365)
    actual = seasonal + rng.normal(0, 4.0, n_days)

    return pd.DataFrame({
        "date": dates,
        "tmax_f": actual.round(1),
    })


@pytest.fixture
def mos_df():
    return _make_mos_df()


@pytest.fixture
def actual_df():
    return _make_actual_df()


@pytest.fixture
def fitted_proxy(mos_df, actual_df):
    proxy = MOSMarketProxy(mos_df, actual_df)
    proxy.fit(train_end_date="2021-12-31")
    return proxy


# =========================================================================
# Test: Initialization
# =========================================================================

class TestInitialization:
    """Tests for MOSMarketProxy.__init__."""

    def test_basic_init(self, mos_df, actual_df):
        proxy = MOSMarketProxy(mos_df, actual_df)
        assert not proxy._is_fitted
        assert proxy.overall_sigma is None

    def test_init_with_ensemble_column(self, actual_df):
        """Test that initialization works when mos_ensemble_tmax_f is present."""
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "mos_ensemble_tmax_f": np.random.uniform(30, 90, 100),
        })
        proxy = MOSMarketProxy(mos, actual_df)
        assert len(proxy._mos_lookup) > 0

    def test_init_without_ensemble_builds_from_models(self, actual_df):
        """Test that ensemble is auto-computed from GFS + NAM columns."""
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "gfs_mos_tmax_f": np.random.uniform(30, 90, 100),
            "nam_mos_tmax_f": np.random.uniform(30, 90, 100),
        })
        proxy = MOSMarketProxy(mos, actual_df)
        assert len(proxy._mos_lookup) == 100

    def test_init_gfs_only(self, actual_df):
        """Test initialization with only GFS data (no NAM)."""
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "gfs_mos_tmax_f": np.random.uniform(30, 90, 100),
        })
        proxy = MOSMarketProxy(mos, actual_df)
        assert len(proxy._gfs_lookup) == 100

    def test_init_empty_mos_raises(self, actual_df):
        with pytest.raises(ValueError, match="non-empty"):
            MOSMarketProxy(pd.DataFrame(), actual_df)

    def test_init_none_mos_raises(self, actual_df):
        with pytest.raises(ValueError, match="non-empty"):
            MOSMarketProxy(None, actual_df)

    def test_init_empty_actual_raises(self, mos_df):
        with pytest.raises(ValueError, match="non-empty"):
            MOSMarketProxy(mos_df, pd.DataFrame())

    def test_init_none_actual_raises(self, mos_df):
        with pytest.raises(ValueError, match="non-empty"):
            MOSMarketProxy(mos_df, None)

    def test_init_missing_tmax_col_raises(self, mos_df):
        bad_actual = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "temperature": [50] * 10,
        })
        with pytest.raises(ValueError, match="tmax_f"):
            MOSMarketProxy(mos_df, bad_actual)

    def test_init_missing_all_forecast_cols_raises(self):
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "some_column": [50] * 10,
        })
        actual = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "tmax_f": [50] * 10,
        })
        with pytest.raises(ValueError, match="mos_ensemble_tmax_f"):
            MOSMarketProxy(mos, actual)

    def test_duplicate_dates_handled(self, actual_df):
        """Test that duplicate dates are deduplicated."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        mos = pd.DataFrame({
            "date": list(dates) + list(dates[:10]),  # duplicates
            "mos_ensemble_tmax_f": list(range(50)) + list(range(10)),
        })
        proxy = MOSMarketProxy(mos, actual_df)
        assert len(proxy._mos_lookup) == 50


# =========================================================================
# Test: Fit
# =========================================================================

class TestFit:
    """Tests for MOSMarketProxy.fit()."""

    def test_fit_basic(self, mos_df, actual_df):
        proxy = MOSMarketProxy(mos_df, actual_df)
        result = proxy.fit(train_end_date="2021-12-31")
        assert result is proxy  # method chaining
        assert proxy._is_fitted
        assert proxy.n_train_days > 0

    def test_fit_computes_overall_stats(self, fitted_proxy):
        assert fitted_proxy.overall_mae is not None
        assert fitted_proxy.overall_rmse is not None
        assert fitted_proxy.overall_bias is not None
        assert fitted_proxy.overall_sigma is not None

    def test_overall_mae_positive(self, fitted_proxy):
        assert fitted_proxy.overall_mae > 0

    def test_overall_rmse_gte_mae(self, fitted_proxy):
        assert fitted_proxy.overall_rmse >= fitted_proxy.overall_mae

    def test_overall_sigma_positive(self, fitted_proxy):
        assert fitted_proxy.overall_sigma > 0

    def test_monthly_sigma_all_12_months(self, fitted_proxy):
        assert len(fitted_proxy.monthly_sigma) == 12
        for month in range(1, 13):
            assert month in fitted_proxy.monthly_sigma
            assert fitted_proxy.monthly_sigma[month] > 0

    def test_monthly_bias_all_12_months(self, fitted_proxy):
        assert len(fitted_proxy.monthly_bias) == 12

    def test_monthly_mae_all_12_months(self, fitted_proxy):
        assert len(fitted_proxy.monthly_mae) == 12
        for month in range(1, 13):
            assert fitted_proxy.monthly_mae[month] > 0

    def test_fit_no_overlap_raises(self):
        mos = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100, freq="D"),
            "mos_ensemble_tmax_f": np.random.uniform(30, 90, 100),
        })
        actual = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "tmax_f": np.random.uniform(30, 90, 100),
        })
        proxy = MOSMarketProxy(mos, actual)
        with pytest.raises(ValueError, match="No overlapping"):
            proxy.fit(train_end_date="2024-12-31")

    def test_fit_respects_cutoff(self, mos_df, actual_df):
        """Test that no data after train_end_date is used."""
        proxy = MOSMarketProxy(mos_df, actual_df)
        proxy.fit(train_end_date="2020-06-30")
        # Should have used ~182 days (Jan-Jun 2020)
        assert proxy.n_train_days <= 182

    def test_fit_train_end_date_stored(self, fitted_proxy):
        assert fitted_proxy._train_end_date == date(2021, 12, 31)


# =========================================================================
# Test: predict_mu_sigma
# =========================================================================

class TestPredictMuSigma:
    """Tests for MOSMarketProxy.predict_mu_sigma()."""

    def test_returns_tuple(self, fitted_proxy):
        result = fitted_proxy.predict_mu_sigma(date(2022, 7, 15))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mu_is_mos_forecast(self, fitted_proxy):
        """Verify that mu equals the MOS ensemble forecast."""
        target = date(2022, 7, 15)
        mu, sigma = fitted_proxy.predict_mu_sigma(target)
        expected_mu = fitted_proxy._mos_lookup.get(target)
        if expected_mu is not None:
            assert abs(mu - expected_mu) < 0.01

    def test_sigma_is_monthly(self, fitted_proxy):
        """Verify that sigma matches the monthly value."""
        target = date(2022, 7, 15)
        mu, sigma = fitted_proxy.predict_mu_sigma(target)
        expected_sigma = fitted_proxy.monthly_sigma[7]
        assert abs(sigma - expected_sigma) < 0.01

    def test_accepts_string_date(self, fitted_proxy):
        mu, sigma = fitted_proxy.predict_mu_sigma("2022-07-15")
        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_accepts_kwargs_for_compatibility(self, fitted_proxy):
        """Test that extra kwargs are accepted without error."""
        mu, sigma = fitted_proxy.predict_mu_sigma(
            date(2022, 7, 15),
            yesterday_tmax=85.0,
            day_before_tmax=82.0,
            rolling_7d_mean=83.0,
        )
        assert isinstance(mu, float)

    def test_not_fitted_raises(self, mos_df, actual_df):
        proxy = MOSMarketProxy(mos_df, actual_df)
        with pytest.raises(RuntimeError, match="not fitted"):
            proxy.predict_mu_sigma(date(2022, 7, 15))

    def test_reasonable_summer_values(self, fitted_proxy):
        mu, sigma = fitted_proxy.predict_mu_sigma(date(2022, 7, 15))
        assert 50 < mu < 110  # Reasonable summer NYC temp
        assert 0 < sigma < 20

    def test_reasonable_winter_values(self, fitted_proxy):
        mu, sigma = fitted_proxy.predict_mu_sigma(date(2022, 1, 15))
        assert 0 < mu < 80  # Reasonable winter NYC temp
        assert 0 < sigma < 20

    def test_fallback_to_yesterday_tmax(self):
        """Test fallback when MOS forecast is missing for a date."""
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "mos_ensemble_tmax_f": np.random.uniform(30, 60, 100),
        })
        actual = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "tmax_f": np.random.uniform(30, 60, 100),
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-04-09")

        # Date with no MOS data
        mu, sigma = proxy.predict_mu_sigma(
            date(2025, 6, 15), yesterday_tmax=75.0,
        )
        assert mu == 75.0

    def test_fallback_to_nearest_mos(self):
        """Test fallback to nearest MOS when no yesterday_tmax."""
        mos = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "mos_ensemble_tmax_f": [50.0] * 100,
        })
        actual = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            "tmax_f": [50.0] * 100,
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-04-09")

        mu, sigma = proxy.predict_mu_sigma(date(2025, 6, 15))
        assert isinstance(mu, float)


# =========================================================================
# Test: compute_bracket_prob
# =========================================================================

class TestComputeBracketProb:
    """Tests for MOSMarketProxy.compute_bracket_prob()."""

    def test_above_high_threshold(self, fitted_proxy):
        """P(TMAX > very_high) should be low."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 1, 15), 100.0, None, "above",
        )
        assert 0.02 <= prob <= 0.10

    def test_above_low_threshold(self, fitted_proxy):
        """P(TMAX > very_low) should be high."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 30.0, None, "above",
        )
        assert 0.90 <= prob <= 0.98

    def test_below_low_threshold(self, fitted_proxy):
        """P(TMAX < very_low) should be low."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), None, 30.0, "below",
        )
        assert 0.02 <= prob <= 0.10

    def test_below_high_threshold(self, fitted_proxy):
        """P(TMAX < very_high) should be high."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 1, 15), None, 100.0, "below",
        )
        assert 0.90 <= prob <= 0.98

    def test_between_wide_bracket(self, fitted_proxy):
        """P(0 < TMAX < 120) should be very high."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 0.0, 120.0, "between",
        )
        assert prob >= 0.95

    def test_between_narrow_bracket(self, fitted_proxy):
        """Narrow bracket around forecast should have moderate probability."""
        mu, sigma = fitted_proxy.predict_mu_sigma(date(2022, 7, 15))
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), mu - 2, mu + 2, "between",
        )
        assert 0.10 < prob < 0.90

    def test_prob_clipped_min(self, fitted_proxy):
        """Probability should never be below 0.02."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 150.0, None, "above",
        )
        assert prob >= 0.02

    def test_prob_clipped_max(self, fitted_proxy):
        """Probability should never exceed 0.98."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), -50.0, None, "above",
        )
        assert prob <= 0.98

    def test_unknown_direction_returns_half(self, fitted_proxy):
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 50.0, 60.0, "unknown_direction",
        )
        assert prob == 0.5

    def test_accepts_kwargs(self, fitted_proxy):
        """Test that extra kwargs are accepted for interface compat."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 50.0, None, "above",
            yesterday_tmax=80.0,
            day_before_tmax=78.0,
        )
        assert 0.02 <= prob <= 0.98

    def test_string_date(self, fitted_proxy):
        prob = fitted_proxy.compute_bracket_prob(
            "2022-07-15", 50.0, None, "above",
        )
        assert 0.02 <= prob <= 0.98

    def test_above_plus_below_sums_roughly_one(self, fitted_proxy):
        """P(above T) + P(below T) should be close to 1.0."""
        threshold = 70.0
        p_above = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), threshold, None, "above",
        )
        p_below = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), None, threshold, "below",
        )
        # Allow some tolerance due to clipping
        assert abs(p_above + p_below - 1.0) < 0.10

    def test_between_none_low(self, fitted_proxy):
        """Test between with None low threshold."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), None, 80.0, "between",
        )
        assert 0.02 <= prob <= 0.98

    def test_between_none_high(self, fitted_proxy):
        """Test between with None high threshold."""
        prob = fitted_proxy.compute_bracket_prob(
            date(2022, 7, 15), 50.0, None, "between",
        )
        assert 0.02 <= prob <= 0.98


# =========================================================================
# Test: Fallback behavior
# =========================================================================

class TestFallback:
    """Tests for fallback when one MOS model is missing."""

    def test_gfs_only_ensemble(self):
        """Ensemble should work with GFS only (no NAM)."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        mos = pd.DataFrame({
            "date": dates,
            "gfs_mos_tmax_f": np.random.uniform(30, 90, 200),
            "nam_mos_tmax_f": [np.nan] * 200,
            "mos_ensemble_tmax_f": np.random.uniform(30, 90, 200),
        })
        actual = pd.DataFrame({
            "date": dates,
            "tmax_f": np.random.uniform(30, 90, 200),
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-07-18")
        mu, sigma = proxy.predict_mu_sigma(date(2020, 5, 15))
        assert isinstance(mu, float)
        assert isinstance(sigma, float)

    def test_nam_only_fallback(self):
        """Should fall back to NAM when GFS is missing for a date."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        mos = pd.DataFrame({
            "date": dates,
            "gfs_mos_tmax_f": [np.nan] * 200,
            "nam_mos_tmax_f": [55.0] * 200,
            "mos_ensemble_tmax_f": [55.0] * 200,
        })
        actual = pd.DataFrame({
            "date": dates,
            "tmax_f": [55.0] * 200,
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-07-18")
        mu, sigma = proxy.predict_mu_sigma(date(2020, 5, 15))
        assert abs(mu - 55.0) < 0.1

    def test_missing_date_uses_nearest(self):
        """When a date has no MOS data, should use nearest available."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        mos = pd.DataFrame({
            "date": dates,
            "mos_ensemble_tmax_f": [50.0] * 100,
        })
        actual = pd.DataFrame({
            "date": dates,
            "tmax_f": [50.0] * 100,
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-04-09")

        # Date far outside MOS range
        mu, sigma = proxy.predict_mu_sigma(date(2030, 1, 1))
        assert isinstance(mu, float)


# =========================================================================
# Test: Monthly sigma computation
# =========================================================================

class TestMonthlySigma:
    """Tests for monthly error sigma computation."""

    def test_sigma_varies_by_month(self, fitted_proxy):
        """Different months should potentially have different sigmas."""
        sigmas = list(fitted_proxy.monthly_sigma.values())
        # With enough data, not all months should be identical
        assert len(set(round(s, 2) for s in sigmas)) > 1

    def test_sigma_reasonable_range(self, fitted_proxy):
        """Monthly sigmas should be in a reasonable range for temp forecasts."""
        for month, sigma in fitted_proxy.monthly_sigma.items():
            assert 0.5 < sigma < 20.0, f"Month {month}: sigma={sigma} out of range"

    def test_sparse_month_falls_back_to_overall(self):
        """Months with <10 data points should use overall sigma."""
        # Create data only for January
        dates = pd.date_range("2020-01-01", periods=31, freq="D")
        mos = pd.DataFrame({
            "date": dates,
            "mos_ensemble_tmax_f": np.random.uniform(30, 50, 31),
        })
        actual = pd.DataFrame({
            "date": dates,
            "tmax_f": np.random.uniform(30, 50, 31),
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-01-31")

        # July (month=7) should fall back since no data exists
        assert proxy.monthly_sigma[7] == proxy.overall_sigma


# =========================================================================
# Test: get_diagnostics
# =========================================================================

class TestGetDiagnostics:
    """Tests for MOSMarketProxy.get_diagnostics()."""

    def test_unfitted_returns_not_fitted(self, mos_df, actual_df):
        proxy = MOSMarketProxy(mos_df, actual_df)
        diag = proxy.get_diagnostics()
        assert diag["fitted"] is False

    def test_fitted_returns_all_keys(self, fitted_proxy):
        diag = fitted_proxy.get_diagnostics()
        assert diag["fitted"] is True
        expected_keys = [
            "fitted", "train_end_date", "n_train_days",
            "overall_mae", "overall_rmse", "overall_bias",
            "overall_sigma", "monthly_sigma", "monthly_bias",
            "monthly_mae", "mos_date_range", "actual_date_range",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing key: {key}"

    def test_diagnostics_n_train_days(self, fitted_proxy):
        diag = fitted_proxy.get_diagnostics()
        assert diag["n_train_days"] > 0

    def test_diagnostics_train_end_date(self, fitted_proxy):
        diag = fitted_proxy.get_diagnostics()
        assert diag["train_end_date"] == "2021-12-31"

    def test_diagnostics_monthly_sigma_has_12(self, fitted_proxy):
        diag = fitted_proxy.get_diagnostics()
        assert len(diag["monthly_sigma"]) == 12


# =========================================================================
# Test: generate_proxy_forecasts
# =========================================================================

class TestGenerateProxyForecasts:
    """Tests for MOSMarketProxy.generate_proxy_forecasts()."""

    def test_returns_dataframe(self, fitted_proxy):
        result = fitted_proxy.generate_proxy_forecasts("2022-01-01", "2022-01-31")
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_proxy):
        result = fitted_proxy.generate_proxy_forecasts("2022-01-01", "2022-01-31")
        assert "date" in result.columns
        assert "proxy_mu" in result.columns
        assert "proxy_sigma" in result.columns

    def test_correct_date_range(self, fitted_proxy):
        result = fitted_proxy.generate_proxy_forecasts("2022-01-01", "2022-01-31")
        if not result.empty:
            dates = pd.to_datetime(result["date"])
            assert dates.min().date() >= date(2022, 1, 1)
            assert dates.max().date() <= date(2022, 1, 31)

    def test_not_fitted_raises(self, mos_df, actual_df):
        proxy = MOSMarketProxy(mos_df, actual_df)
        with pytest.raises(RuntimeError):
            proxy.generate_proxy_forecasts("2022-01-01", "2022-01-31")


# =========================================================================
# Test: Integration with real downloaded MOS data
# =========================================================================

class TestIntegrationWithRealData:
    """Integration tests using actual downloaded MOS data from IEM."""

    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        """Skip tests if real data files are not present."""
        combined_path = os.path.join(
            PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv"
        )
        actual_path = os.path.join(
            PROJECT_ROOT, "data", "central_park_tmax_full_history.csv"
        )
        if not os.path.exists(combined_path):
            pytest.skip("Combined MOS data not found. Run download_iem_mos.py first.")
        if not os.path.exists(actual_path):
            pytest.skip("Central Park TMAX data not found.")

    @pytest.fixture
    def real_mos_df(self):
        path = os.path.join(
            PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def real_actual_df(self):
        path = os.path.join(
            PROJECT_ROOT, "data", "central_park_tmax_full_history.csv"
        )
        return pd.read_csv(path)

    @pytest.fixture
    def real_proxy(self, real_mos_df, real_actual_df):
        proxy = MOSMarketProxy(real_mos_df, real_actual_df)
        proxy.fit(train_end_date="2022-12-31")
        return proxy

    def test_real_data_loads(self, real_mos_df, real_actual_df):
        assert len(real_mos_df) > 1000
        assert len(real_actual_df) > 1000

    def test_real_data_fit(self, real_proxy):
        assert real_proxy._is_fitted
        assert real_proxy.n_train_days > 1000

    def test_real_mae_reasonable(self, real_proxy):
        """MOS MAE should be around 3-5 degrees F."""
        assert 1.0 < real_proxy.overall_mae < 8.0

    def test_real_rmse_reasonable(self, real_proxy):
        assert 1.0 < real_proxy.overall_rmse < 10.0

    def test_real_sigma_reasonable(self, real_proxy):
        assert 2.0 < real_proxy.overall_sigma < 10.0

    def test_real_monthly_sigma_range(self, real_proxy):
        for month, sigma in real_proxy.monthly_sigma.items():
            assert 1.0 < sigma < 12.0, (
                f"Month {month}: sigma={sigma:.2f} seems unreasonable"
            )

    def test_real_predict_summer(self, real_proxy):
        mu, sigma = real_proxy.predict_mu_sigma(date(2023, 7, 15))
        assert 60 < mu < 105, f"Summer mu={mu:.1f} out of range"
        assert 1 < sigma < 12

    def test_real_predict_winter(self, real_proxy):
        mu, sigma = real_proxy.predict_mu_sigma(date(2023, 1, 15))
        assert 10 < mu < 60, f"Winter mu={mu:.1f} out of range"
        assert 1 < sigma < 12

    def test_real_bracket_prob(self, real_proxy):
        # Summer day: P(TMAX > 80) should be moderate-to-high
        prob_above_80 = real_proxy.compute_bracket_prob(
            date(2023, 7, 15), 80.0, None, "above",
        )
        assert 0.02 <= prob_above_80 <= 0.98

        # Winter day: P(TMAX > 80) should be very low
        prob_above_80_winter = real_proxy.compute_bracket_prob(
            date(2023, 1, 15), 80.0, None, "above",
        )
        assert prob_above_80_winter < 0.10

    def test_real_diagnostics(self, real_proxy):
        diag = real_proxy.get_diagnostics()
        assert diag["fitted"] is True
        assert diag["n_train_days"] > 1000
        assert diag["overall_mae"] is not None

    def test_real_generate_forecasts(self, real_proxy):
        result = real_proxy.generate_proxy_forecasts("2023-01-01", "2023-12-31")
        assert len(result) > 300  # Should have most of the year
        assert result["proxy_mu"].min() > -10
        assert result["proxy_mu"].max() < 120


# =========================================================================
# Test: Download script utilities
# =========================================================================

class TestDownloadScript:
    """Tests for download script utility functions."""

    def test_parse_mos_csv(self):
        """Test that parse_mos_csv correctly parses IEM format."""
        from scripts.download_iem_mos import parse_mos_csv

        sample_csv = (
            "runtime,ftime,model,n_x,tmp,dpt,cld,wdr,wsp,p06,p12,q06,q12,"
            "t06_1,t06_2,t12_1,t12_2,snw,cig,vis,obv,poz,pos,typ,station,t06,t12\n"
            "2024-01-01 12:00:00,2024-01-02 12:00:00,GFS,32.0,33,18,CL,320,5,"
            "0.0,2.0,0.0,0.0,0.0,3.0,,,0.0,8,7,N ,4,89,S ,KNYC,0/3,\n"
            "2024-01-01 12:00:00,2024-01-03 00:00:00,GFS,45.0,41,19,CL,290,4,"
            "0.0,0.0,0.0,0.0,0.0,0.0,,,,8,7,N ,3,10,R ,KNYC,0/0,\n"
        )

        df = parse_mos_csv(sample_csv)
        assert len(df) == 2
        assert df.iloc[0]["n_x"] == 32.0
        assert df.iloc[1]["n_x"] == 45.0
        assert df.iloc[0]["model"] == "GFS"

    def test_extract_tmax_forecasts(self):
        """Test TMAX extraction from parsed MOS data."""
        from scripts.download_iem_mos import extract_tmax_forecasts

        # Create sample parsed data with n_x at 00Z (TMAX) and 12Z (TMIN)
        data = {
            "runtime": pd.to_datetime([
                "2024-01-01 12:00", "2024-01-01 12:00",
                "2024-01-01 12:00", "2024-01-01 12:00",
            ]),
            "ftime": pd.to_datetime([
                "2024-01-02 12:00", "2024-01-03 00:00",
                "2024-01-03 12:00", "2024-01-04 00:00",
            ]),
            "model": ["GFS"] * 4,
            "n_x": [32.0, 45.0, 35.0, 44.0],
        }
        df = pd.DataFrame(data)

        result = extract_tmax_forecasts(df)

        # Should extract only the 00Z ftime entries (TMAX)
        assert len(result) == 2
        # First TMAX: ftime Jan 3 00Z -> target date Jan 2
        assert result.iloc[0]["tmax_forecast_f"] == 45.0
        # Second TMAX: ftime Jan 4 00Z -> target date Jan 3
        assert result.iloc[1]["tmax_forecast_f"] == 44.0

    def test_build_combined_mos(self):
        """Test combining GFS and NAM forecasts."""
        from scripts.download_iem_mos import build_combined_mos

        gfs = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "tmax_forecast_f": [40.0, 42.0, 44.0],
            "runtime": pd.to_datetime(["2024-01-01"] * 3),
            "model": ["GFS"] * 3,
        })
        nam = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "tmax_forecast_f": [38.0, 44.0],
            "runtime": pd.to_datetime(["2024-01-01"] * 2),
            "model": ["NAM"] * 2,
        })

        result = build_combined_mos(gfs, nam)

        assert len(result) == 3
        assert "gfs_mos_tmax_f" in result.columns
        assert "nam_mos_tmax_f" in result.columns
        assert "mos_ensemble_tmax_f" in result.columns

        # Check ensemble averaging
        row1 = result.iloc[0]
        assert row1["mos_ensemble_tmax_f"] == 39.0  # (40+38)/2

        # Check that NAM-missing date still has ensemble (GFS only)
        row3 = result.iloc[2]
        assert row3["mos_ensemble_tmax_f"] == 44.0  # GFS only


# =========================================================================
# Test: Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_day_data(self):
        """Test with minimal data (1 day)."""
        mos = pd.DataFrame({
            "date": [date(2020, 6, 15)],
            "mos_ensemble_tmax_f": [85.0],
        })
        actual = pd.DataFrame({
            "date": [date(2020, 6, 15)],
            "tmax_f": [83.0],
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-12-31")
        mu, sigma = proxy.predict_mu_sigma(date(2020, 6, 15))
        assert mu == 85.0

    def test_constant_forecasts(self):
        """Test with constant forecast and actuals."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        mos = pd.DataFrame({
            "date": dates,
            "mos_ensemble_tmax_f": [70.0] * 365,
        })
        actual = pd.DataFrame({
            "date": dates,
            "tmax_f": [72.0] * 365,
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-12-31")

        # Bias should be +2.0 (actual - forecast = 72 - 70)
        assert abs(proxy.overall_bias - 2.0) < 0.1

        # Sigma should be ~0 since errors are constant
        assert proxy.overall_sigma < 0.5

    def test_large_date_gap(self):
        """Test with a gap in MOS data."""
        dates_before = pd.date_range("2020-01-01", periods=100, freq="D")
        dates_after = pd.date_range("2020-06-01", periods=100, freq="D")
        all_dates = dates_before.append(dates_after)

        mos = pd.DataFrame({
            "date": all_dates,
            "mos_ensemble_tmax_f": np.random.uniform(30, 90, 200),
        })
        actual = pd.DataFrame({
            "date": all_dates,
            "tmax_f": np.random.uniform(30, 90, 200),
        })
        proxy = MOSMarketProxy(mos, actual)
        proxy.fit(train_end_date="2020-09-08")

        # Date in the gap
        mu, sigma = proxy.predict_mu_sigma(date(2020, 4, 15))
        assert isinstance(mu, float)  # Should use nearest forecast
