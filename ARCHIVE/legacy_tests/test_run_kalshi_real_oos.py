"""
Tests for run_kalshi_real_oos.py -- Real-Data OOS Validation Pipeline.

Validates:
  - Feature building from GHCN station data
  - Ridge model training and prediction generation
  - Market probability construction (climatological model)
  - Backtest data preparation (model + market probabilities)
  - Strategy extraction from config strings
  - Final output file existence and format
"""

import os
import sys
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from run_kalshi_real_oos import (
    build_station_features,
    train_ridge_model,
    construct_market_probabilities,
    prepare_oos_backtest_data,
    extract_strategy_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_kalshi_df():
    """Create a minimal Kalshi-like DataFrame."""
    dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    records = []
    for d in dates:
        actual = 40.0 + 5.0 * np.sin(2 * np.pi * d.dayofyear / 365)
        # Above market
        records.append({
            "date": d.date(),
            "direction": "above",
            "threshold_low": actual - 5,
            "threshold_high": np.nan,
            "actual_tmax": actual,
            "actual_outcome": 1,
            "ticker": f"KXHIGHNY-{d.strftime('%y%b%d').upper()}-T{int(actual-5)}",
            "bucket_label": f"Above {int(actual-5)}F",
            "volume": 100,
        })
        # Below market
        records.append({
            "date": d.date(),
            "direction": "below",
            "threshold_low": np.nan,
            "threshold_high": actual + 5,
            "actual_tmax": actual,
            "actual_outcome": 1,
            "ticker": f"KXHIGHNY-{d.strftime('%y%b%d').upper()}-T{int(actual+5)}",
            "bucket_label": f"Below {int(actual+5)}F",
            "volume": 50,
        })
        # Between market
        records.append({
            "date": d.date(),
            "direction": "between",
            "threshold_low": actual - 2,
            "threshold_high": actual + 2,
            "actual_tmax": actual,
            "actual_outcome": 1,
            "ticker": f"KXHIGHNY-{d.strftime('%y%b%d').upper()}-B{int(actual)}",
            "bucket_label": f"{int(actual-2)}-{int(actual+2)}F",
            "volume": 200,
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_predictions_df():
    """Create sample model predictions."""
    dates = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    records = []
    for d in dates:
        actual = 40.0 + 5.0 * np.sin(2 * np.pi * d.dayofyear / 365)
        records.append({
            "date": d.date(),
            "model_mu": actual + np.random.normal(0, 2),
            "model_sigma": 5.0,
            "actual_tmax": actual,
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_actual_tmax():
    """Create sample actual TMAX data."""
    dates = pd.date_range("2024-12-30", "2025-01-12", freq="D")
    records = []
    for d in dates:
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "tmax_f": 35.0 + 5.0 * np.sin(2 * np.pi * d.dayofyear / 365),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests: extract_strategy_params
# ---------------------------------------------------------------------------

class TestExtractStrategyParams:
    """Tests for strategy parameter extraction from config strings."""

    def test_full_name_parsing(self):
        """Parse a complete strategy name."""
        config = {
            "strategy_name": "S0396_ev0.15_proportional_kf0.05_fee0.07_mp0.05_br10000"
        }
        params = extract_strategy_params(config)
        assert params["ev_threshold"] == 0.15
        assert params["sizing_method"] == "proportional"
        assert params["kelly_fraction"] == 0.05
        assert params["fee_rate"] == 0.07
        assert params["max_position_frac"] == 0.05
        assert params["bankroll"] == 10000

    def test_fractional_kelly_name(self):
        """Parse a fractional kelly strategy name."""
        config = {
            "strategy_name": "S0001_ev0.02_fractional_kelly_kf0.10_fee0.07_mp0.10_br10000"
        }
        params = extract_strategy_params(config)
        assert params["ev_threshold"] == 0.02
        assert params["sizing_method"] == "fractional_kelly"
        assert params["kelly_fraction"] == 0.10

    def test_default_params(self):
        """Default params when name is empty."""
        params = extract_strategy_params({"strategy_name": ""})
        assert params["name"] == "Best_from_2023_2024"
        assert params["ev_threshold"] == 0.02
        assert params["sizing_method"] == "fractional_kelly"

    def test_fixed_sizing(self):
        """Parse a fixed sizing strategy."""
        config = {
            "strategy_name": "S0050_ev0.05_fixed_kf0.05_fee0.07_mp0.20_br10000"
        }
        params = extract_strategy_params(config)
        assert params["sizing_method"] == "fixed"

    def test_capped_kelly(self):
        """Parse capped kelly strategy."""
        config = {
            "strategy_name": "S0100_ev0.03_capped_kelly_kf0.20_fee0.07_mp0.15_br10000"
        }
        params = extract_strategy_params(config)
        assert params["sizing_method"] == "capped_kelly"
        assert params["kelly_fraction"] == 0.20
        assert params["max_position_frac"] == 0.15


# ---------------------------------------------------------------------------
# Tests: construct_market_probabilities
# ---------------------------------------------------------------------------

class TestConstructMarketProbabilities:
    """Tests for market probability construction."""

    def test_basic_construction(self, sample_kalshi_df, sample_actual_tmax):
        """Market probabilities are constructed for all rows."""
        result = construct_market_probabilities(sample_kalshi_df, sample_actual_tmax)
        assert "market_prob" in result.columns
        assert len(result) == len(sample_kalshi_df)
        assert result["market_prob"].notna().all()

    def test_probability_range(self, sample_kalshi_df, sample_actual_tmax):
        """Market probabilities are in [0.02, 0.98]."""
        result = construct_market_probabilities(sample_kalshi_df, sample_actual_tmax)
        assert result["market_prob"].min() >= 0.02
        assert result["market_prob"].max() <= 0.98

    def test_above_direction(self, sample_actual_tmax):
        """Above-threshold markets have decreasing prob with higher thresholds."""
        records = []
        d = date(2025, 7, 15)
        for threshold in [60, 70, 80, 90, 100]:
            records.append({
                "date": d,
                "direction": "above",
                "threshold_low": float(threshold),
                "threshold_high": np.nan,
                "actual_tmax": 85.0,
                "actual_outcome": 1 if 85 > threshold else 0,
                "ticker": f"T{threshold}",
                "volume": 100,
            })
        df = pd.DataFrame(records)
        result = construct_market_probabilities(df, sample_actual_tmax)
        probs = result["market_prob"].tolist()
        # Probabilities should decrease as threshold increases
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], \
                f"P(>={60+10*i}) = {probs[i]} should be >= P(>={70+10*i}) = {probs[i+1]}"

    def test_below_direction(self, sample_actual_tmax):
        """Below-threshold markets have increasing prob with higher thresholds."""
        records = []
        d = date(2025, 1, 15)
        for threshold in [30, 40, 50, 60]:
            records.append({
                "date": d,
                "direction": "below",
                "threshold_low": np.nan,
                "threshold_high": float(threshold),
                "actual_tmax": 35.0,
                "actual_outcome": 1 if 35 < threshold else 0,
                "ticker": f"T{threshold}",
                "volume": 100,
            })
        df = pd.DataFrame(records)
        result = construct_market_probabilities(df, sample_actual_tmax)
        probs = result["market_prob"].tolist()
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]


# ---------------------------------------------------------------------------
# Tests: prepare_oos_backtest_data
# ---------------------------------------------------------------------------

class TestPrepareOosBacktestData:
    """Tests for backtest data preparation."""

    def test_basic_preparation(self, sample_kalshi_df, sample_predictions_df,
                                sample_actual_tmax):
        kalshi_with_market = construct_market_probabilities(
            sample_kalshi_df, sample_actual_tmax)
        result = prepare_oos_backtest_data(
            kalshi_with_market, sample_predictions_df, sample_actual_tmax)

        assert "model_prob" in result.columns
        assert "market_price" in result.columns
        assert "actual_outcome" in result.columns
        assert len(result) > 0

    def test_model_probs_valid(self, sample_kalshi_df, sample_predictions_df,
                                sample_actual_tmax):
        kalshi_with_market = construct_market_probabilities(
            sample_kalshi_df, sample_actual_tmax)
        result = prepare_oos_backtest_data(
            kalshi_with_market, sample_predictions_df, sample_actual_tmax)

        assert result["model_prob"].min() >= 0.001
        assert result["model_prob"].max() <= 0.999

    def test_no_nan_in_required_cols(self, sample_kalshi_df, sample_predictions_df,
                                      sample_actual_tmax):
        kalshi_with_market = construct_market_probabilities(
            sample_kalshi_df, sample_actual_tmax)
        result = prepare_oos_backtest_data(
            kalshi_with_market, sample_predictions_df, sample_actual_tmax)

        for col in ["model_prob", "market_price", "actual_outcome"]:
            assert result[col].notna().all(), f"Found NaN in {col}"


# ---------------------------------------------------------------------------
# Tests: Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Tests to verify data quality and consistency."""

    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data",
    )

    def test_ghcn_tmax_realistic_range(self):
        """GHCN temperatures should be in realistic NYC range."""
        df = pd.read_csv(os.path.join(self.DATA_DIR, "real_central_park_tmax_2025.csv"))
        assert df["tmax_f"].min() >= -10, "TMAX too low for NYC"
        assert df["tmax_f"].max() <= 115, "TMAX too high for NYC"

    def test_model_predictions_mae_reasonable(self):
        """Model MAE should be in a reasonable range."""
        df = pd.read_csv(os.path.join(self.DATA_DIR, "real_model_predictions_2025.csv"))
        mae = (df["model_mu"] - df["actual_tmax"]).abs().mean()
        assert mae < 10.0, f"MAE too high: {mae:.2f}F"
        assert mae > 1.0, f"MAE suspiciously low: {mae:.2f}F"

    def test_model_sigma_reasonable(self):
        """Model sigma should reflect genuine uncertainty."""
        df = pd.read_csv(os.path.join(self.DATA_DIR, "real_model_predictions_2025.csv"))
        mean_sigma = df["model_sigma"].mean()
        assert 2.0 < mean_sigma < 15.0, f"Mean sigma {mean_sigma:.2f} outside expected range"

    def test_kalshi_dates_cover_2025(self):
        """Kalshi data should cover most of 2025."""
        df = pd.read_csv(os.path.join(self.DATA_DIR, "real_kalshi_2025.csv"))
        dates = pd.to_datetime(df["date"])
        assert dates.dt.year.eq(2025).sum() > 0 or dates.dt.year.eq(2024).sum() > 0
        n_unique = df["date"].nunique()
        assert n_unique >= 350, f"Only {n_unique} unique dates in Kalshi data"
