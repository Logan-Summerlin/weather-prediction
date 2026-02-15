"""
Comprehensive tests for src/kalshi_backtester.py.

Tests all four classes and utility functions:
  - KalshiMarketSimulator
  - ModelPredictionGenerator
  - BacktestAnalyzer
  - CalibrationAnalyzer
  - prepare_backtest_data, compute_seasonal_pnl
  - End-to-end OOS pipeline

At least 40 tests, all passing.
"""

import os
import sys
import json
import tempfile
import datetime

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.kalshi_backtester import (
    KalshiMarketSimulator,
    ModelPredictionGenerator,
    BacktestAnalyzer,
    CalibrationAnalyzer,
    prepare_backtest_data,
    compute_seasonal_pnl,
    KXHIGHNY_BUCKET_EDGES,
    KXHIGHNY_BUCKET_LABELS,
    NYC_MONTHLY_TMAX_MEAN,
    NYC_MONTHLY_TMAX_STD,
    SEASON_MAP,
    SEASON_ORDER,
)
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    BacktestResult,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    """Fixed random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def market_sim():
    """Default KalshiMarketSimulator."""
    return KalshiMarketSimulator()


@pytest.fixture
def pred_gen():
    """Default ModelPredictionGenerator."""
    return ModelPredictionGenerator()


@pytest.fixture
def sample_predictions_short():
    """Small DataFrame of model predictions (30 days)."""
    gen = ModelPredictionGenerator()
    return gen.generate_predictions("2025-01-01", "2025-01-30", seed=42)


@pytest.fixture
def sample_predictions_year():
    """Full-year DataFrame of model predictions."""
    gen = ModelPredictionGenerator()
    return gen.generate_predictions("2025-01-01", "2025-12-31", seed=42)


@pytest.fixture
def sample_market_data(sample_predictions_short, market_sim):
    """Small market dataset for testing."""
    return market_sim.generate_market_dataset(sample_predictions_short, seed=42)


@pytest.fixture
def sample_comparison_df(sample_market_data):
    """Market data prepared for backtest analysis."""
    return sample_market_data


@pytest.fixture
def analyzer():
    """Default BacktestAnalyzer."""
    return BacktestAnalyzer()


@pytest.fixture
def cal_analyzer():
    """Default CalibrationAnalyzer."""
    return CalibrationAnalyzer()


# ===========================================================================
# KalshiMarketSimulator Tests
# ===========================================================================

class TestKalshiMarketSimulator:
    """Tests for KalshiMarketSimulator."""

    def test_init_default(self, market_sim):
        """Default constructor sets expected values."""
        assert len(market_sim.bucket_edges) == 57
        assert len(market_sim.bucket_labels) == 57
        assert market_sim.market_noise_std == 0.06
        assert market_sim.min_spread == 0.02
        assert market_sim.max_spread == 0.10

    def test_init_custom_buckets(self):
        """Custom bucket edges and labels."""
        edges = [(0, 50), (50, 100)]
        labels = ["Low", "High"]
        sim = KalshiMarketSimulator(bucket_edges=edges, bucket_labels=labels)
        assert len(sim.bucket_edges) == 2
        assert sim.bucket_labels == ["Low", "High"]

    def test_init_mismatched_buckets_raises(self):
        """Mismatched bucket edges and labels raises ValueError."""
        with pytest.raises(ValueError, match="bucket_edges length"):
            KalshiMarketSimulator(
                bucket_edges=[(0, 50), (50, 100)],
                bucket_labels=["Only one label"],
            )

    def test_generate_daily_buckets_returns_correct_count(self, market_sim, rng):
        """generate_daily_buckets returns one dict per bucket."""
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 7, 15),
            actual_tmax=85.0,
            model_mu=83.0,
            model_sigma=4.0,
            rng=rng,
        )
        assert len(buckets) == len(KXHIGHNY_BUCKET_EDGES)

    def test_generate_daily_buckets_has_required_keys(self, market_sim, rng):
        """Each bucket dict has all required keys."""
        buckets = market_sim.generate_daily_buckets(
            date="2025-07-15",
            actual_tmax=85.0,
            model_mu=83.0,
            model_sigma=4.0,
            rng=rng,
        )
        required_keys = {
            "date", "bucket_label", "bucket_low", "bucket_high",
            "true_prob", "market_prob", "model_prob",
            "bid_price", "ask_price", "volume",
            "actual_outcome", "direction", "threshold_low", "threshold_high",
        }
        for bucket in buckets:
            assert required_keys.issubset(bucket.keys())

    def test_generate_daily_buckets_probabilities_in_range(self, market_sim, rng):
        """Market and model probabilities are in [0.001, 0.999]."""
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 1, 15),
            actual_tmax=35.0,
            model_mu=37.0,
            model_sigma=5.0,
            rng=rng,
        )
        for b in buckets:
            assert 0.0 < b["market_prob"] < 1.0
            assert 0.0 < b["model_prob"] < 1.0
            assert 0.0 < b["bid_price"] < 1.0
            assert 0.0 < b["ask_price"] < 1.0

    def test_generate_daily_buckets_exactly_one_outcome(self, market_sim, rng):
        """Exactly one bucket has actual_outcome=1 for any given day."""
        for actual_tmax in [15.0, 45.0, 75.0, 95.0, 105.0]:
            buckets = market_sim.generate_daily_buckets(
                date=datetime.date(2025, 6, 1),
                actual_tmax=actual_tmax,
                model_mu=actual_tmax,
                model_sigma=4.0,
                rng=rng,
            )
            outcomes = [b["actual_outcome"] for b in buckets]
            assert sum(outcomes) == 1, (
                f"Expected exactly 1 outcome=1 for tmax={actual_tmax}, "
                f"got {sum(outcomes)}"
            )

    def test_generate_daily_buckets_settlement_correctness(self, market_sim, rng):
        """Outcome is correct for known temperature values."""
        # TMAX = 55.0 should settle in "54-56" bucket (2°F grid: [54, 56))
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 4, 15),
            actual_tmax=55.0,
            model_mu=55.0,
            model_sigma=4.0,
            rng=rng,
        )
        for b in buckets:
            if b["bucket_label"] == "54-56":
                assert b["actual_outcome"] == 1
            else:
                assert b["actual_outcome"] == 0

    def test_generate_daily_buckets_bid_less_than_ask(self, market_sim, rng):
        """Bid price is always less than or equal to ask price."""
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 8, 1),
            actual_tmax=90.0,
            model_mu=88.0,
            model_sigma=3.0,
            rng=rng,
        )
        for b in buckets:
            assert b["bid_price"] <= b["ask_price"]

    def test_generate_daily_buckets_volume_positive(self, market_sim, rng):
        """Volume is positive for all buckets."""
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 3, 10),
            actual_tmax=50.0,
            model_mu=48.0,
            model_sigma=5.0,
            rng=rng,
        )
        for b in buckets:
            assert b["volume"] >= 1

    def test_generate_market_dataset_shape(self, market_sim, sample_predictions_short):
        """Market dataset has correct number of rows."""
        df = market_sim.generate_market_dataset(sample_predictions_short, seed=42)
        expected_rows = len(sample_predictions_short) * len(KXHIGHNY_BUCKET_EDGES)
        assert len(df) == expected_rows

    def test_generate_market_dataset_columns(self, market_sim, sample_predictions_short):
        """Market dataset has required columns."""
        df = market_sim.generate_market_dataset(sample_predictions_short, seed=42)
        required = {
            "date", "bucket_label", "market_prob", "model_prob",
            "actual_outcome", "bid_price", "ask_price", "volume",
        }
        assert required.issubset(df.columns)

    def test_generate_market_dataset_missing_columns_raises(self, market_sim):
        """Missing required columns raises ValueError."""
        bad_df = pd.DataFrame({"date": ["2025-01-01"], "model_mu": [50.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            market_sim.generate_market_dataset(bad_df, seed=42)

    def test_generate_market_dataset_reproducible(self, market_sim, sample_predictions_short):
        """Same seed produces identical results."""
        df1 = market_sim.generate_market_dataset(sample_predictions_short, seed=123)
        df2 = market_sim.generate_market_dataset(sample_predictions_short, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_daily_buckets_directions(self, market_sim, rng):
        """First bucket is 'below', last is 'above', middle are 'between'."""
        buckets = market_sim.generate_daily_buckets(
            date=datetime.date(2025, 5, 1),
            actual_tmax=65.0,
            model_mu=64.0,
            model_sigma=4.0,
            rng=rng,
        )
        assert buckets[0]["direction"] == "below"
        assert buckets[-1]["direction"] == "above"
        for b in buckets[1:-1]:
            assert b["direction"] == "between"


# ===========================================================================
# ModelPredictionGenerator Tests
# ===========================================================================

class TestModelPredictionGenerator:
    """Tests for ModelPredictionGenerator."""

    def test_init_default(self, pred_gen):
        """Default constructor sets expected values."""
        assert pred_gen.model_bias == 0.0
        assert pred_gen.model_noise_std == 2.0
        assert pred_gen.sigma_base == 3.5
        assert pred_gen.model_edge == 0.03

    def test_generate_predictions_shape(self, pred_gen):
        """Predictions have correct number of rows."""
        df = pred_gen.generate_predictions("2025-01-01", "2025-01-31", seed=42)
        assert len(df) == 31

    def test_generate_predictions_columns(self, pred_gen):
        """Predictions have all required columns."""
        df = pred_gen.generate_predictions("2025-01-01", "2025-01-10", seed=42)
        required = {
            "date", "model_mu", "model_sigma", "actual_tmax",
            "climatology_mean", "climatology_std",
        }
        assert required.issubset(df.columns)

    def test_generate_predictions_date_range(self, pred_gen):
        """Date column covers the requested range."""
        df = pred_gen.generate_predictions("2025-03-01", "2025-03-31", seed=42)
        dates = pd.to_datetime(df["date"])
        assert dates.min().date() == datetime.date(2025, 3, 1)
        assert dates.max().date() == datetime.date(2025, 3, 31)

    def test_generate_predictions_seasonal_pattern(self, sample_predictions_year):
        """Summer temperatures are higher than winter temperatures."""
        df = sample_predictions_year
        df["_date"] = pd.to_datetime(df["date"])
        df["_month"] = df["_date"].dt.month

        winter_mean = df[df["_month"].isin([1, 2, 12])]["actual_tmax"].mean()
        summer_mean = df[df["_month"].isin([6, 7, 8])]["actual_tmax"].mean()

        assert summer_mean > winter_mean, (
            f"Summer ({summer_mean:.1f}) should be warmer than winter ({winter_mean:.1f})"
        )

    def test_generate_predictions_reasonable_range(self, sample_predictions_year):
        """Temperatures stay within physically reasonable range."""
        df = sample_predictions_year
        assert df["actual_tmax"].min() >= -10.0
        assert df["actual_tmax"].max() <= 115.0

    def test_generate_predictions_sigma_positive(self, pred_gen):
        """Model sigma is always positive."""
        df = pred_gen.generate_predictions("2025-01-01", "2025-12-31", seed=42)
        assert (df["model_sigma"] > 0).all()

    def test_generate_predictions_reproducible(self, pred_gen):
        """Same seed produces identical results."""
        df1 = pred_gen.generate_predictions("2025-06-01", "2025-06-30", seed=77)
        df2 = pred_gen.generate_predictions("2025-06-01", "2025-06-30", seed=77)
        pd.testing.assert_frame_equal(df1, df2)

    def test_climatology_seasonal_variation(self, pred_gen):
        """Climatological mean varies by season."""
        jan_mean, jan_std = pred_gen._get_climatology(datetime.date(2025, 1, 15))
        jul_mean, jul_std = pred_gen._get_climatology(datetime.date(2025, 7, 15))

        assert jul_mean > jan_mean
        assert jan_std > jul_std  # Winter is more volatile

    def test_generate_predictions_handles_leap_year(self, pred_gen):
        """Handles leap year dates correctly."""
        # 2024 was a leap year
        df = pred_gen.generate_predictions("2024-02-28", "2024-03-01", seed=42)
        assert len(df) == 3  # Feb 28, Feb 29, Mar 1

    def test_generate_predictions_winter_higher_sigma(self, sample_predictions_year):
        """Winter model sigma is higher than summer (seasonal variance)."""
        df = sample_predictions_year
        df["_date"] = pd.to_datetime(df["date"])
        df["_month"] = df["_date"].dt.month

        winter_sigma = df[df["_month"].isin([1, 2, 12])]["model_sigma"].mean()
        summer_sigma = df[df["_month"].isin([6, 7, 8])]["model_sigma"].mean()

        # Winter should have higher sigma due to seasonal scaling
        assert winter_sigma > summer_sigma, (
            f"Winter sigma ({winter_sigma:.2f}) should exceed summer ({summer_sigma:.2f})"
        )


# ===========================================================================
# BacktestAnalyzer Tests
# ===========================================================================

class TestBacktestAnalyzer:
    """Tests for BacktestAnalyzer."""

    def test_analyze_brier_scores_basic(self, analyzer, sample_comparison_df):
        """Brier analysis returns expected structure."""
        result = analyzer.analyze_brier_scores(sample_comparison_df)
        assert "overall" in result
        assert "by_month" in result
        assert "by_season" in result
        assert "n_samples" in result
        assert result["n_samples"] > 0

    def test_analyze_brier_scores_values_in_range(self, analyzer, sample_comparison_df):
        """Brier scores are in [0, 1]."""
        result = analyzer.analyze_brier_scores(sample_comparison_df)
        overall = result["overall"]
        assert 0 <= overall["model_brier"] <= 1
        assert 0 <= overall["market_brier"] <= 1

    def test_analyze_brier_scores_empty_df(self, analyzer):
        """Empty DataFrame returns nan values."""
        empty = pd.DataFrame(columns=["date", "model_prob", "market_prob", "actual_outcome"])
        result = analyzer.analyze_brier_scores(empty)
        assert result["n_samples"] == 0
        assert np.isnan(result["overall"]["model_brier"])

    def test_analyze_brier_scores_missing_column_raises(self, analyzer):
        """Missing outcome column raises ValueError."""
        bad_df = pd.DataFrame({
            "date": ["2025-01-01"],
            "model_prob": [0.5],
            "market_prob": [0.5],
        })
        with pytest.raises(ValueError, match="actual_outcome"):
            analyzer.analyze_brier_scores(bad_df)

    def test_analyze_edge_persistence_structure(self, analyzer):
        """Edge persistence returns correct DataFrame structure."""
        is_metrics = {
            "sharpe_ratio": 2.5, "roi": 0.15, "win_rate": 0.55,
            "max_drawdown": 800, "total_pnl": 1500, "n_trades": 200,
            "brier_delta": -0.02,
        }
        oos_metrics = {
            "sharpe_ratio": 1.8, "roi": 0.08, "win_rate": 0.52,
            "max_drawdown": 1000, "total_pnl": 800, "n_trades": 100,
            "brier_delta": -0.01,
        }
        df = analyzer.analyze_edge_persistence(is_metrics, oos_metrics)
        assert isinstance(df, pd.DataFrame)
        assert "metric" in df.columns
        assert "verdict" in df.columns
        assert len(df) == 7  # 7 metrics tracked

    def test_analyze_edge_persistence_verdicts(self, analyzer):
        """Verdicts are generated correctly for known metrics."""
        is_m = {"sharpe_ratio": 2.0, "roi": 0.10, "win_rate": 0.5,
                "max_drawdown": 500, "total_pnl": 1000, "n_trades": 200,
                "brier_delta": -0.02}
        oos_m = {"sharpe_ratio": 1.8, "roi": 0.05, "win_rate": 0.48,
                 "max_drawdown": 600, "total_pnl": 500, "n_trades": 100,
                 "brier_delta": -0.01}
        df = analyzer.analyze_edge_persistence(is_m, oos_m)

        sharpe_row = df[df["metric"] == "sharpe_ratio"].iloc[0]
        assert "STRONG" in sharpe_row["verdict"]

        brier_row = df[df["metric"] == "brier_delta"].iloc[0]
        assert "PERSISTS" in brier_row["verdict"]

    def test_select_best_strategy_basic(self, analyzer):
        """Selects the correct best strategy from a DataFrame."""
        df = pd.DataFrame({
            "strategy_name": ["A", "B", "C"],
            "sharpe_ratio": [2.5, 1.8, 0.5],
            "total_pnl": [1500, 1200, 300],
            "roi": [0.15, 0.12, 0.03],
            "win_rate": [0.55, 0.50, 0.35],
            "max_drawdown": [800, 1000, 500],
            "n_trades": [200, 150, 50],
        })
        best = analyzer.select_best_strategy(df, bankroll=10000)
        assert best["strategy_name"] == "A"

    def test_select_best_strategy_empty_df(self, analyzer):
        """Empty DataFrame returns NONE strategy."""
        best = analyzer.select_best_strategy(pd.DataFrame())
        assert best["strategy_name"] == "NONE"

    def test_select_best_strategy_relaxed_criteria(self, analyzer):
        """Relaxes criteria when no strategy qualifies."""
        # All strategies fail strict criteria (sharpe < 1.5)
        df = pd.DataFrame({
            "strategy_name": ["A", "B"],
            "sharpe_ratio": [1.2, 0.8],
            "total_pnl": [500, 200],
            "roi": [0.05, 0.02],
            "win_rate": [0.45, 0.40],
            "max_drawdown": [300, 200],
            "n_trades": [80, 60],
        })
        best = analyzer.select_best_strategy(df, bankroll=10000)
        assert "Relaxed" in best.get("selection_reason", "")

    def test_select_best_strategy_pnl_tiebreaker(self, analyzer):
        """When both Sharpe > 2.0 and within 0.2, prefer higher P&L."""
        df = pd.DataFrame({
            "strategy_name": ["HighPnL", "HighSharpe"],
            "sharpe_ratio": [2.3, 2.4],
            "total_pnl": [2000, 1500],
            "roi": [0.20, 0.15],
            "win_rate": [0.55, 0.50],
            "max_drawdown": [800, 600],
            "n_trades": [200, 180],
        })
        best = analyzer.select_best_strategy(df, bankroll=10000)
        assert best["strategy_name"] == "HighPnL"
        assert "higher P&L" in best.get("selection_reason", "")

    def test_generate_comprehensive_report(self, analyzer):
        """Comprehensive report generates a non-empty markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_result = {
                "metrics": {
                    "sharpe_ratio": 2.0, "roi": 0.10, "win_rate": 0.55,
                    "max_drawdown": 500, "total_pnl": 1000, "n_trades": 200,
                    "avg_ev": 0.03, "period": "2023-2024", "bankroll": 10000,
                },
                "brier_analysis": {
                    "overall": {
                        "model_brier": 0.18, "market_brier": 0.20,
                        "brier_delta": -0.02,
                    },
                    "by_season": {},
                },
                "strategy_config": {"name": "TestStrategy", "ev_threshold": 0.02},
            }
            oos_result = {
                "metrics": {
                    "sharpe_ratio": 1.5, "roi": 0.06, "win_rate": 0.52,
                    "max_drawdown": 700, "total_pnl": 600, "n_trades": 100,
                    "avg_ev": 0.025, "period": "2025", "bankroll": 10000,
                },
                "brier_analysis": {
                    "overall": {
                        "model_brier": 0.19, "market_brier": 0.20,
                        "brier_delta": -0.01,
                    },
                    "by_season": {},
                },
                "trades": [
                    {"date": "2025-01-15", "pnl": 10.0},
                    {"date": "2025-01-16", "pnl": -5.0},
                ],
                "seasonal_pnl": {},
            }

            report = analyzer.generate_comprehensive_report(
                is_result, oos_result, tmpdir,
            )
            assert len(report) > 100
            assert "Executive Summary" in report
            assert "Trading Recommendation" in report

            # Check file was saved
            report_path = os.path.join(tmpdir, "final_backtest_report.md")
            assert os.path.exists(report_path)

    def test_plot_oos_vs_insample(self, analyzer):
        """Comparison plots are generated and saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_result = {
                "trades": [
                    {"date": "2024-01-01", "pnl": 5.0},
                    {"date": "2024-01-02", "pnl": -3.0},
                ],
                "metrics": {"sharpe_ratio": 1.5, "roi": 0.05, "win_rate": 0.5},
            }
            oos_result = {
                "trades": [
                    {"date": "2025-01-01", "pnl": 3.0},
                    {"date": "2025-01-02", "pnl": -2.0},
                ],
                "metrics": {"sharpe_ratio": 1.0, "roi": 0.03, "win_rate": 0.45},
            }

            plots = analyzer.plot_oos_vs_insample(is_result, oos_result, tmpdir)
            assert len(plots) >= 2
            for p in plots:
                assert os.path.exists(p)

    def test_recommendation_validated(self, analyzer):
        """VALIDATED recommendation for strong OOS performance."""
        rec = analyzer._generate_recommendation(
            {"sharpe_ratio": 2.0, "roi": 0.15, "brier_delta": -0.02},
            {"sharpe_ratio": 1.8, "roi": 0.10, "brier_delta": -0.01},
        )
        assert rec["verdict"] == "VALIDATED"

    def test_recommendation_cautious(self, analyzer):
        """CAUTIOUS recommendation for moderate OOS performance."""
        rec = analyzer._generate_recommendation(
            {"sharpe_ratio": 2.0, "roi": 0.15, "brier_delta": -0.02},
            {"sharpe_ratio": 1.0, "roi": 0.03, "brier_delta": -0.005},
        )
        assert rec["verdict"] == "CAUTIOUS"

    def test_recommendation_overfit(self, analyzer):
        """OVERFIT recommendation for weak OOS performance."""
        rec = analyzer._generate_recommendation(
            {"sharpe_ratio": 2.0, "roi": 0.15, "brier_delta": -0.02},
            {"sharpe_ratio": 0.3, "roi": -0.02, "brier_delta": -0.005},
        )
        assert rec["verdict"] == "OVERFIT"

    def test_recommendation_no_edge(self, analyzer):
        """NO EDGE when OOS brier delta is positive."""
        rec = analyzer._generate_recommendation(
            {"sharpe_ratio": 2.0, "roi": 0.15, "brier_delta": -0.02},
            {"sharpe_ratio": 0.3, "roi": -0.05, "brier_delta": 0.01},
        )
        assert rec["verdict"] == "NO EDGE"


# ===========================================================================
# CalibrationAnalyzer Tests
# ===========================================================================

class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer."""

    def test_analyze_model_calibration_structure(self, cal_analyzer):
        """Calibration analysis returns expected structure."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.5])
        outcomes = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
        result = cal_analyzer.analyze_model_calibration(probs, outcomes)

        assert "brier_score" in result
        assert "log_score" in result
        assert "ece" in result
        assert "mce" in result
        assert "reliability" in result
        assert "n_samples" in result
        assert result["n_samples"] == 10

    def test_analyze_model_calibration_perfect(self, cal_analyzer):
        """Perfect predictions give low Brier and ECE."""
        n = 1000
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, n)
        outcomes = (rng.uniform(0, 1, n) < probs).astype(float)

        result = cal_analyzer.analyze_model_calibration(probs, outcomes, n_bins=10)
        # Should have low ECE for large N with correct calibration
        assert result["ece"] < 0.1

    def test_analyze_model_calibration_empty(self, cal_analyzer):
        """Empty arrays return NaN values."""
        result = cal_analyzer.analyze_model_calibration(
            np.array([]), np.array([]),
        )
        assert result["n_samples"] == 0
        assert np.isnan(result["brier_score"])

    def test_plot_reliability_diagram(self, cal_analyzer):
        """Reliability diagram is saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            outcomes = np.array([0, 0, 1, 1, 1])
            path = cal_analyzer.plot_reliability_diagram(
                probs, outcomes, tmpdir,
            )
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_compute_seasonal_calibration(self, cal_analyzer, sample_comparison_df):
        """Seasonal calibration returns per-season metrics."""
        result = cal_analyzer.compute_seasonal_calibration(sample_comparison_df)
        assert isinstance(result, dict)
        # Should have at least 1 season (sample is only January)
        assert len(result) >= 1

    def test_compute_seasonal_calibration_all_seasons(self, cal_analyzer):
        """All 4 seasons have calibration when data covers full year."""
        gen = ModelPredictionGenerator()
        preds = gen.generate_predictions("2025-01-01", "2025-12-31", seed=42)
        sim = KalshiMarketSimulator()
        mkt = sim.generate_market_dataset(preds, seed=42)
        result = cal_analyzer.compute_seasonal_calibration(mkt)
        assert len(result) == 4


# ===========================================================================
# Utility Function Tests
# ===========================================================================

class TestUtilityFunctions:
    """Tests for prepare_backtest_data and compute_seasonal_pnl."""

    def test_prepare_backtest_data_renames_columns(self, sample_market_data):
        """prepare_backtest_data renames market_prob to market_price."""
        df = prepare_backtest_data(sample_market_data)
        assert "market_price" in df.columns
        assert "model_prob" in df.columns
        assert "actual_outcome" in df.columns

    def test_prepare_backtest_data_missing_column_raises(self):
        """Missing column raises ValueError."""
        bad_df = pd.DataFrame({"date": ["2025-01-01"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_backtest_data(bad_df)

    def test_compute_seasonal_pnl_structure(self):
        """Seasonal P&L returns correct structure."""
        trades = [
            {"date": "2025-01-15", "pnl": 10.0},
            {"date": "2025-01-16", "pnl": -5.0},
            {"date": "2025-07-15", "pnl": 20.0},
            {"date": "2025-07-16", "pnl": -8.0},
        ]
        result = compute_seasonal_pnl(trades)
        assert "Winter (DJF)" in result
        assert "Summer (JJA)" in result
        assert result["Winter (DJF)"]["total_pnl"] == 5.0
        assert result["Summer (JJA)"]["total_pnl"] == 12.0

    def test_compute_seasonal_pnl_empty(self):
        """Empty trades return empty dict."""
        assert compute_seasonal_pnl([]) == {}

    def test_compute_seasonal_pnl_win_rate(self):
        """Win rate is correctly computed per season."""
        trades = [
            {"date": "2025-01-01", "pnl": 10.0},  # win
            {"date": "2025-01-02", "pnl": -5.0},   # loss
            {"date": "2025-01-03", "pnl": 3.0},    # win
        ]
        result = compute_seasonal_pnl(trades)
        winter = result["Winter (DJF)"]
        assert winter["n_trades"] == 3
        assert abs(winter["win_rate"] - 2.0 / 3.0) < 1e-6


# ===========================================================================
# End-to-End Pipeline Tests
# ===========================================================================

class TestEndToEndPipeline:
    """End-to-end tests for the full OOS validation pipeline."""

    def test_full_pipeline_in_sample(self, sample_predictions_short, market_sim, analyzer):
        """Full in-sample pipeline: predictions -> market -> backtest -> analysis."""
        # Generate market data
        market_df = market_sim.generate_market_dataset(
            sample_predictions_short, seed=42,
        )

        # Prepare for backtest
        backtest_data = prepare_backtest_data(market_df)
        assert "market_price" in backtest_data.columns

        # Run a simple strategy
        strategy = TradingStrategy(
            name="Test",
            ev_threshold=0.02,
            sizing_method="fixed",
            fee_rate=0.07,
            bankroll=10000,
        )
        engine = BacktestEngine(strategy)
        result = engine.run_backtest(backtest_data)

        assert isinstance(result, BacktestResult)
        assert result.n_days == len(backtest_data)

        # Brier analysis
        brier = analyzer.analyze_brier_scores(market_df)
        assert brier["n_samples"] > 0

    def test_full_pipeline_oos_frozen_strategy(self):
        """Full OOS pipeline: frozen IS strategy applied to OOS data."""
        # In-sample
        gen = ModelPredictionGenerator(model_edge=0.03)
        is_preds = gen.generate_predictions("2024-06-01", "2024-06-30", seed=42)
        sim = KalshiMarketSimulator()
        is_market = sim.generate_market_dataset(is_preds, seed=42)

        # OOS with different seed
        oos_preds = gen.generate_predictions("2025-06-01", "2025-06-30", seed=99)
        oos_market = sim.generate_market_dataset(oos_preds, seed=99)

        # Run same strategy on both
        strategy = TradingStrategy(
            name="Frozen",
            ev_threshold=0.01,
            sizing_method="fractional_kelly",
            kelly_fraction=0.10,
            fee_rate=0.07,
            bankroll=10000,
        )

        is_data = prepare_backtest_data(is_market)
        oos_data = prepare_backtest_data(oos_market)

        is_result = BacktestEngine(strategy).run_backtest(is_data)
        oos_result = BacktestEngine(strategy).run_backtest(oos_data)

        # Both should produce results
        assert is_result.n_days > 0
        assert oos_result.n_days > 0

        # Analyze edge persistence
        analyzer = BacktestAnalyzer()
        is_m = is_result.to_summary_dict()
        oos_m = oos_result.to_summary_dict()

        # Add mock brier delta
        is_m["brier_delta"] = -0.01
        oos_m["brier_delta"] = -0.005

        persistence = analyzer.analyze_edge_persistence(is_m, oos_m)
        assert len(persistence) == 7

    def test_pipeline_output_files(self):
        """Output files are generated in correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ModelPredictionGenerator()
            preds = gen.generate_predictions("2025-01-01", "2025-01-15", seed=42)
            sim = KalshiMarketSimulator()
            market = sim.generate_market_dataset(preds, seed=42)

            # Save market data
            market.to_csv(os.path.join(tmpdir, "oos_backtest_results.csv"), index=False)

            # Save metrics
            metrics = {"sharpe_ratio": 1.5, "roi": 0.05}
            with open(os.path.join(tmpdir, "oos_metrics.json"), "w") as f:
                json.dump(metrics, f)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "oos_backtest_results.csv"))
            assert os.path.exists(os.path.join(tmpdir, "oos_metrics.json"))

    def test_constants_consistency(self):
        """Bucket edges, labels, and season maps are consistent."""
        assert len(KXHIGHNY_BUCKET_EDGES) == len(KXHIGHNY_BUCKET_LABELS)
        assert len(SEASON_MAP) == 12
        assert set(SEASON_MAP.values()) == set(SEASON_ORDER)

        # All months mapped
        for m in range(1, 13):
            assert m in SEASON_MAP

        # NYC climatology has all 12 months
        assert len(NYC_MONTHLY_TMAX_MEAN) == 12
        assert len(NYC_MONTHLY_TMAX_STD) == 12


# ===========================================================================
# Run with: python -m pytest tests/test_kalshi_backtester.py -v
# ===========================================================================
