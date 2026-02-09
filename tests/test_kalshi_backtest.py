"""
Comprehensive tests for run_kalshi_backtest.py.

Tests cover:
  - Market data generation (buckets, settlement, dates, ranges)
  - Model prediction generation (seasonal patterns, sigma values)
  - Alignment between model and market data
  - Strategy selection logic
  - Report generation
  - Edge cases (NaN, empty data, extreme temperatures)
  - Full pipeline end-to-end
"""

import json
import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from run_kalshi_backtest import (
    generate_realistic_market_data,
    align_model_and_market,
    run_strategy_grid_search,
    analyze_results,
    generate_all_plots,
    generate_backtest_report,
    _get_season,
    _get_buckets_for_month,
    _select_best_strategy,
    _config_from_strategy_row,
    NYC_MONTHLY_TMAX_MEAN,
    NYC_MONTHLY_TMAX_STD,
    WINTER_BUCKETS,
    SPRING_FALL_BUCKETS,
    SUMMER_BUCKETS,
)
from src.kalshi_client import build_historical_comparison, compute_brier_scores
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    generate_strategy_grid,
    run_comprehensive_backtest,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def market_and_predictions():
    """Generate a short (3-month) dataset for testing."""
    market_df, predictions_df = generate_realistic_market_data(
        start_date="2023-01-01",
        end_date="2023-03-31",
        seed=42,
    )
    return market_df, predictions_df


@pytest.fixture(scope="module")
def full_year_data():
    """Generate a full year of data for comprehensive tests."""
    market_df, predictions_df = generate_realistic_market_data(
        start_date="2023-01-01",
        end_date="2023-12-31",
        seed=99,
    )
    return market_df, predictions_df


@pytest.fixture(scope="module")
def comparison_data(market_and_predictions):
    """Generate aligned comparison data."""
    market_df, predictions_df = market_and_predictions
    comparison_df = align_model_and_market(market_df, predictions_df)
    return comparison_df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for output files."""
    tmpdir = tempfile.mkdtemp(prefix="kalshi_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# 1. Market Data Generation Tests
# ============================================================================

class TestMarketDataGeneration:
    """Test the generate_realistic_market_data function."""

    def test_market_data_has_required_columns(self, market_and_predictions):
        """Market DataFrame must have all required columns."""
        market_df, _ = market_and_predictions
        required = [
            "date", "ticker", "bucket", "threshold_low", "threshold_high",
            "direction", "market_prob", "actual_outcome", "bid_price",
            "ask_price", "volume", "actual_tmax",
        ]
        for col in required:
            assert col in market_df.columns, f"Missing column: {col}"

    def test_predictions_have_required_columns(self, market_and_predictions):
        """Predictions DataFrame must have all required columns."""
        _, predictions_df = market_and_predictions
        required = ["date", "model_mu", "model_sigma", "actual_tmax"]
        for col in required:
            assert col in predictions_df.columns, f"Missing column: {col}"

    def test_market_date_coverage(self, market_and_predictions):
        """Market data should cover the full date range."""
        market_df, _ = market_and_predictions
        unique_dates = market_df["date"].dt.date.nunique()
        # Jan-Mar = 90 days
        assert unique_dates == 90, f"Expected 90 unique dates, got {unique_dates}"

    def test_predictions_date_coverage(self, market_and_predictions):
        """Predictions should cover 1 entry per day."""
        _, predictions_df = market_and_predictions
        assert len(predictions_df) == 90

    def test_buckets_per_day(self, market_and_predictions):
        """Each day should have 7 bucket contracts."""
        market_df, _ = market_and_predictions
        buckets_per_day = market_df.groupby(market_df["date"].dt.date).size()
        assert (buckets_per_day == 7).all(), (
            f"Expected 7 buckets per day, got: {buckets_per_day.unique()}"
        )

    def test_market_prob_range(self, market_and_predictions):
        """Market probabilities should be in (0, 1)."""
        market_df, _ = market_and_predictions
        assert (market_df["market_prob"] > 0).all()
        assert (market_df["market_prob"] < 1).all()

    def test_actual_outcome_binary(self, market_and_predictions):
        """Actual outcomes should be 0 or 1."""
        market_df, _ = market_and_predictions
        assert set(market_df["actual_outcome"].unique()).issubset({0, 1})

    def test_exactly_one_yes_per_day(self, market_and_predictions):
        """Exactly one bucket per day should have actual_outcome=1.

        Uses half-open intervals [low, high) for between-buckets, so the
        entire temperature axis is partitioned without gaps or overlaps.
        """
        market_df, _ = market_and_predictions
        yes_per_day = market_df.groupby(market_df["date"].dt.date)["actual_outcome"].sum()
        # With contiguous [low, high) buckets, exactly 1 bucket should settle YES
        assert (yes_per_day == 1).all(), (
            f"Expected exactly 1 YES per day; got min={yes_per_day.min()}, "
            f"max={yes_per_day.max()}, "
            f"days with !=1: {(yes_per_day != 1).sum()}"
        )

    def test_bid_ask_spread(self, market_and_predictions):
        """Bid should be less than or equal to ask."""
        market_df, _ = market_and_predictions
        assert (market_df["bid_price"] <= market_df["ask_price"]).all()

    def test_volume_positive(self, market_and_predictions):
        """Volume should be positive for all contracts."""
        market_df, _ = market_and_predictions
        assert (market_df["volume"] > 0).all()

    def test_actual_tmax_realistic_range(self, market_and_predictions):
        """Actual temperatures should be in realistic NYC range."""
        _, predictions_df = market_and_predictions
        tmax = predictions_df["actual_tmax"]
        assert tmax.min() > 0, f"Min temp {tmax.min()} below 0F"
        assert tmax.max() < 115, f"Max temp {tmax.max()} above 115F"

    def test_direction_values(self, market_and_predictions):
        """Direction should be one of above, below, between."""
        market_df, _ = market_and_predictions
        valid_directions = {"above", "below", "between"}
        actual_directions = set(market_df["direction"].unique())
        assert actual_directions.issubset(valid_directions), (
            f"Invalid directions: {actual_directions - valid_directions}"
        )

    def test_reproducibility(self):
        """Same seed should produce identical data."""
        m1, p1 = generate_realistic_market_data("2023-06-01", "2023-06-30", seed=42)
        m2, p2 = generate_realistic_market_data("2023-06-01", "2023-06-30", seed=42)
        pd.testing.assert_frame_equal(m1, m2)
        pd.testing.assert_frame_equal(p1, p2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        _, p1 = generate_realistic_market_data("2023-06-01", "2023-06-30", seed=42)
        _, p2 = generate_realistic_market_data("2023-06-01", "2023-06-30", seed=99)
        assert not np.allclose(p1["model_mu"].values, p2["model_mu"].values)


# ============================================================================
# 2. Model Prediction Tests
# ============================================================================

class TestModelPredictions:
    """Test model prediction generation."""

    def test_model_sigma_positive(self, market_and_predictions):
        """Model sigma must be positive for all days."""
        _, predictions_df = market_and_predictions
        assert (predictions_df["model_sigma"] > 0).all()

    def test_model_sigma_range(self, market_and_predictions):
        """Model sigma should be in a realistic range (2-10 deg F)."""
        _, predictions_df = market_and_predictions
        assert predictions_df["model_sigma"].min() >= 2.0
        assert predictions_df["model_sigma"].max() <= 15.0

    def test_model_mu_tracks_seasonal(self, full_year_data):
        """Model mu should follow seasonal temperature patterns."""
        _, predictions_df = full_year_data
        predictions_df = predictions_df.copy()
        predictions_df["month"] = pd.to_datetime(predictions_df["date"]).dt.month

        # Winter should be colder than summer
        winter_mu = predictions_df[predictions_df["month"].isin([12, 1, 2])]["model_mu"].mean()
        summer_mu = predictions_df[predictions_df["month"].isin([6, 7, 8])]["model_mu"].mean()
        assert summer_mu > winter_mu + 20, (
            f"Summer mu ({summer_mu:.1f}) should be much higher than "
            f"winter mu ({winter_mu:.1f})"
        )

    def test_no_nan_in_predictions(self, market_and_predictions):
        """Predictions should have no NaN values."""
        _, predictions_df = market_and_predictions
        assert not predictions_df["model_mu"].isna().any()
        assert not predictions_df["model_sigma"].isna().any()
        assert not predictions_df["actual_tmax"].isna().any()


# ============================================================================
# 3. Season and Bucket Helper Tests
# ============================================================================

class TestHelpers:
    """Test helper functions."""

    def test_get_season(self):
        """Test season classification for all months."""
        assert _get_season(1) == "Winter"
        assert _get_season(2) == "Winter"
        assert _get_season(3) == "Spring"
        assert _get_season(6) == "Summer"
        assert _get_season(9) == "Fall"
        assert _get_season(12) == "Winter"

    def test_bucket_definitions(self):
        """Bucket definitions should have 7 entries each."""
        assert len(WINTER_BUCKETS) == 7
        assert len(SPRING_FALL_BUCKETS) == 7
        assert len(SUMMER_BUCKETS) == 7

    def test_winter_buckets_for_january(self):
        """January should use winter buckets."""
        buckets = _get_buckets_for_month(1)
        assert buckets == WINTER_BUCKETS

    def test_summer_buckets_for_july(self):
        """July should use summer buckets."""
        buckets = _get_buckets_for_month(7)
        assert buckets == SUMMER_BUCKETS

    def test_spring_fall_buckets_for_april(self):
        """April should use spring/fall buckets."""
        buckets = _get_buckets_for_month(4)
        assert buckets == SPRING_FALL_BUCKETS

    def test_nyc_climatology_constants(self):
        """NYC monthly mean temps should cover all 12 months."""
        assert len(NYC_MONTHLY_TMAX_MEAN) == 12
        assert len(NYC_MONTHLY_TMAX_STD) == 12
        # July should be hottest
        assert NYC_MONTHLY_TMAX_MEAN[7] == max(NYC_MONTHLY_TMAX_MEAN.values())
        # January should be coldest
        assert NYC_MONTHLY_TMAX_MEAN[1] == min(NYC_MONTHLY_TMAX_MEAN.values())


# ============================================================================
# 4. Alignment Tests
# ============================================================================

class TestAlignment:
    """Test model-market alignment."""

    def test_alignment_produces_model_prob(self, comparison_data):
        """Aligned data should have model_prob column."""
        assert "model_prob" in comparison_data.columns

    def test_alignment_produces_outcome(self, comparison_data):
        """Aligned data should have outcome column."""
        assert "outcome" in comparison_data.columns

    def test_alignment_produces_prob_delta(self, comparison_data):
        """Aligned data should have prob_delta column."""
        assert "prob_delta" in comparison_data.columns

    def test_aligned_model_prob_range(self, comparison_data):
        """Model probabilities should be in [0, 1]."""
        valid = comparison_data["model_prob"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_aligned_outcome_binary(self, comparison_data):
        """Outcomes should be 0 or 1."""
        valid = comparison_data["outcome"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_alignment_date_coverage(self, comparison_data, market_and_predictions):
        """Aligned data should cover all market dates."""
        market_df, _ = market_and_predictions
        assert comparison_data["date"].nunique() == market_df["date"].dt.date.nunique()

    def test_empty_market_raises_or_returns_empty(self):
        """Alignment with empty market data should handle gracefully."""
        empty_market = pd.DataFrame(columns=[
            "date", "bucket", "threshold_low", "threshold_high",
            "direction", "market_prob",
        ])
        predictions = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "model_mu": [50.0] * 5,
            "model_sigma": [4.0] * 5,
        })
        result = build_historical_comparison(predictions, empty_market)
        assert result.empty


# ============================================================================
# 5. Strategy Selection Tests
# ============================================================================

class TestStrategySelection:
    """Test strategy selection logic."""

    def test_select_best_with_ideal_candidates(self):
        """Should select highest Sharpe when candidates meet all criteria."""
        df = pd.DataFrame({
            "strategy_name": [
                "S0001_ev0.02_fractional_kelly_kf0.10_fee0.07_mp0.10_br10000",
                "S0002_ev0.03_capped_kelly_kf0.15_fee0.07_mp0.10_br10000",
            ],
            "n_trades": [150, 200],
            "win_rate": [0.45, 0.42],
            "sharpe_ratio": [2.5, 2.8],
            "max_drawdown": [800.0, 1200.0],
            "total_pnl": [1500.0, 1200.0],
            "roi": [0.15, 0.12],
            "avg_ev": [0.03, 0.04],
            "n_days": [500, 500],
        })
        config = _select_best_strategy(df, [])

        # Both have Sharpe > 2.0 and within 0.3 of each other,
        # so the one with best P&L within 0.2 Sharpe of top should win
        assert "strategy_name" in config
        # Since S0002 has higher Sharpe but S0001 is within 0.2
        # of 2.8 (2.6 threshold), S0001 has better P&L
        # 2.5 >= 2.8 - 0.2 = 2.6? No, 2.5 < 2.6.
        # So S0002 should be selected as it has the best P&L among
        # those within 0.2 of top Sharpe (only itself qualifies)
        assert config["strategy_name"] == "S0002_ev0.03_capped_kelly_kf0.15_fee0.07_mp0.10_br10000"

    def test_select_falls_back_on_no_criteria(self):
        """When no strategy meets criteria, should fall back gracefully."""
        df = pd.DataFrame({
            "strategy_name": ["S0001_ev0.10_fixed_kf0.10_fee0.07_mp0.10_br10000"],
            "n_trades": [10],  # Too few
            "win_rate": [0.20],  # Too low
            "sharpe_ratio": [0.3],  # Too low
            "max_drawdown": [500.0],
            "total_pnl": [50.0],
            "roi": [0.005],
            "avg_ev": [0.01],
            "n_days": [100],
        })
        config = _select_best_strategy(df, [])
        assert "strategy_name" in config
        # Should fall back to best overall
        assert config.get("selection_reason") in (
            "fallback_best_sharpe", "highest_sharpe",
        )

    def test_select_empty_df(self):
        """Empty strategy DataFrame should return error."""
        config = _select_best_strategy(pd.DataFrame(), [])
        assert "error" in config

    def test_config_from_strategy_row(self):
        """Should parse strategy parameters from name."""
        row = pd.Series({
            "strategy_name": "S0042_ev0.03_fractional_kelly_kf0.15_fee0.07_mp0.10_br10000",
            "total_pnl": 1500.0,
            "roi": 0.15,
            "sharpe_ratio": 2.5,
            "max_drawdown": 800.0,
            "win_rate": 0.45,
            "n_trades": 200,
        })
        config = _config_from_strategy_row(row, reason="test")
        assert config["ev_threshold"] == 0.03
        assert config["sizing_method"] == "fractional_kelly"
        assert config["kelly_fraction"] == 0.15
        assert config["fee_rate"] == 0.07
        assert config["max_position_frac"] == 0.10
        assert config["bankroll"] == 10000
        assert config["selection_reason"] == "test"


# ============================================================================
# 6. Brier Score Tests
# ============================================================================

class TestBrierScores:
    """Test Brier score computation on aligned data."""

    def test_brier_scores_computed(self, comparison_data):
        """Brier scores should be computable on aligned data."""
        valid = comparison_data.dropna(subset=["model_prob", "market_prob", "outcome"])
        brier = compute_brier_scores(
            valid["model_prob"], valid["market_prob"], valid["outcome"],
        )
        assert "model_brier" in brier
        assert "market_brier" in brier
        assert "brier_delta" in brier
        assert brier["n_samples"] > 0

    def test_brier_in_valid_range(self, comparison_data):
        """Brier scores should be between 0 and 1."""
        valid = comparison_data.dropna(subset=["model_prob", "market_prob", "outcome"])
        brier = compute_brier_scores(
            valid["model_prob"], valid["market_prob"], valid["outcome"],
        )
        assert 0 <= brier["model_brier"] <= 1
        assert 0 <= brier["market_brier"] <= 1

    def test_model_has_edge(self, comparison_data):
        """Model should have slightly better Brier score than market."""
        valid = comparison_data.dropna(subset=["model_prob", "market_prob", "outcome"])
        brier = compute_brier_scores(
            valid["model_prob"], valid["market_prob"], valid["outcome"],
        )
        # Model should be better (lower Brier = better, so delta should be negative)
        # The edge should be small but real
        assert brier["brier_delta"] < 0, (
            f"Model Brier ({brier['model_brier']:.4f}) should be lower than "
            f"market Brier ({brier['market_brier']:.4f})"
        )


# ============================================================================
# 7. Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_day_market_data(self):
        """Should handle generating just one day of data."""
        market_df, predictions_df = generate_realistic_market_data(
            start_date="2023-06-15",
            end_date="2023-06-15",
            seed=42,
        )
        assert len(predictions_df) == 1
        assert len(market_df) == 7  # 7 buckets

    def test_market_data_no_nan_in_critical_columns(self, market_and_predictions):
        """Critical columns should have no NaN values."""
        market_df, _ = market_and_predictions
        for col in ["date", "market_prob", "actual_outcome", "direction"]:
            assert not market_df[col].isna().any(), f"NaN found in {col}"

    def test_ticker_format(self, market_and_predictions):
        """Tickers should follow KXHIGHNY format."""
        market_df, _ = market_and_predictions
        for ticker in market_df["ticker"].head(20):
            assert ticker.startswith("KXHIGHNY-"), f"Invalid ticker: {ticker}"

    def test_two_year_data_coverage(self):
        """Two-year data should have ~730 unique dates."""
        market_df, predictions_df = generate_realistic_market_data(
            start_date="2023-01-01",
            end_date="2024-12-31",
            seed=42,
        )
        assert predictions_df.shape[0] == 731  # 2023 (365) + 2024 (366 leap)
        assert market_df["date"].dt.date.nunique() == 731


# ============================================================================
# 8. Full Pipeline Integration Test
# ============================================================================

class TestFullPipeline:
    """Integration test for the full pipeline."""

    def test_full_pipeline_short(self, temp_dir):
        """Run a short version of the full pipeline and verify outputs."""
        # Step 1-2: Generate data (short period)
        market_df, predictions_df = generate_realistic_market_data(
            start_date="2023-06-01",
            end_date="2023-08-31",
            seed=42,
        )

        # Save data
        market_df.to_csv(os.path.join(temp_dir, "market.csv"), index=False)
        predictions_df.to_csv(os.path.join(temp_dir, "predictions.csv"), index=False)

        # Step 3-4: Align
        comparison_df = align_model_and_market(market_df, predictions_df)
        assert len(comparison_df) > 0

        # Step 5: Strategy grid (small grid for speed)
        backtest_data = comparison_df.copy()
        if "actual_outcome" in backtest_data.columns and "outcome" in backtest_data.columns:
            backtest_data = backtest_data.drop(columns=["actual_outcome"])
        backtest_data = backtest_data.rename(columns={
            "market_prob": "market_price",
            "outcome": "actual_outcome",
        })
        backtest_data = backtest_data.dropna(
            subset=["date", "model_prob", "market_price", "actual_outcome"]
        )

        strategies = generate_strategy_grid(
            ev_thresholds=[0.02, 0.05],
            sizing_methods=["fixed", "fractional_kelly"],
            kelly_fractions=[0.10],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )

        results = run_comprehensive_backtest(
            backtest_data,
            output_dir=temp_dir,
            strategies=strategies,
            max_strategies=50,
        )

        assert "all_results" in results
        assert "comparison_df" in results
        assert len(results["all_results"]) > 0

        # Step 6: Analysis
        analysis = analyze_results(comparison_df, results, temp_dir)

        assert "brier" in analysis
        assert "best_strategy_config" in analysis

        # Verify output files
        assert os.path.exists(os.path.join(temp_dir, "brier_comparison.json"))
        assert os.path.exists(os.path.join(temp_dir, "best_strategy_config.json"))
        assert os.path.exists(os.path.join(temp_dir, "seasonal_analysis.csv"))

    def test_report_generation(self, temp_dir):
        """Report generation should produce a non-empty text file."""
        market_df, predictions_df = generate_realistic_market_data(
            start_date="2023-06-01",
            end_date="2023-06-30",
            seed=42,
        )
        comparison_df = align_model_and_market(market_df, predictions_df)

        backtest_data = comparison_df.copy()
        if "actual_outcome" in backtest_data.columns and "outcome" in backtest_data.columns:
            backtest_data = backtest_data.drop(columns=["actual_outcome"])
        backtest_data = backtest_data.rename(columns={
            "market_prob": "market_price",
            "outcome": "actual_outcome",
        })
        backtest_data = backtest_data.dropna(
            subset=["date", "model_prob", "market_price", "actual_outcome"]
        )

        strategies = generate_strategy_grid(
            ev_thresholds=[0.03],
            sizing_methods=["fixed"],
            kelly_fractions=[0.10],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )

        results = run_comprehensive_backtest(
            backtest_data, output_dir=temp_dir,
            strategies=strategies, max_strategies=10,
        )

        analysis = analyze_results(comparison_df, results, temp_dir)
        report = generate_backtest_report(comparison_df, analysis, results, temp_dir)

        assert len(report) > 100  # Non-trivial report
        assert "BRIER SCORE" in report
        assert os.path.exists(os.path.join(temp_dir, "backtest_report.txt"))

    def test_plot_generation(self, temp_dir):
        """Plot generation should create PNG files without errors."""
        market_df, predictions_df = generate_realistic_market_data(
            start_date="2023-06-01",
            end_date="2023-06-30",
            seed=42,
        )
        comparison_df = align_model_and_market(market_df, predictions_df)

        backtest_data = comparison_df.copy()
        if "actual_outcome" in backtest_data.columns and "outcome" in backtest_data.columns:
            backtest_data = backtest_data.drop(columns=["actual_outcome"])
        backtest_data = backtest_data.rename(columns={
            "market_prob": "market_price",
            "outcome": "actual_outcome",
        })
        backtest_data = backtest_data.dropna(
            subset=["date", "model_prob", "market_price", "actual_outcome"]
        )

        strategies = generate_strategy_grid(
            ev_thresholds=[0.03],
            sizing_methods=["fixed"],
            kelly_fractions=[0.10],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )

        results = run_comprehensive_backtest(
            backtest_data, output_dir=temp_dir,
            strategies=strategies, max_strategies=10,
        )

        analysis = analyze_results(comparison_df, results, temp_dir)
        saved = generate_all_plots(comparison_df, analysis, results, temp_dir)

        assert len(saved) >= 1  # At least scatter plot
        for path in saved:
            assert os.path.exists(path), f"Plot file missing: {path}"
            assert os.path.getsize(path) > 0, f"Plot file empty: {path}"


# ============================================================================
# 9. 2025 OOS Data Generation Tests
# ============================================================================

class TestOOSDataGeneration:
    """Test out-of-sample 2025 data generation."""

    def test_2025_data_different_from_2023(self):
        """2025 data (different seed) should differ from 2023."""
        _, p1 = generate_realistic_market_data(
            "2023-01-01", "2023-12-31", seed=42,
        )
        _, p2 = generate_realistic_market_data(
            "2025-01-01", "2025-12-31", seed=123,
        )
        # Different mu values
        assert not np.allclose(
            p1["model_mu"].values[:365], p2["model_mu"].values[:365]
        )

    def test_2025_data_has_correct_dates(self):
        """2025 data should have 2025 dates."""
        market_df, predictions_df = generate_realistic_market_data(
            "2025-01-01", "2025-12-31", seed=123,
        )
        dates = pd.to_datetime(predictions_df["date"])
        assert (dates.dt.year == 2025).all()
        assert len(predictions_df) == 365  # 2025 is not a leap year


# ============================================================================
# 10. Data Quality and Consistency Tests
# ============================================================================

class TestDataQuality:
    """Test data quality and internal consistency."""

    def test_settlement_consistency(self, market_and_predictions):
        """Settlement outcomes should be consistent with actual temperatures."""
        market_df, _ = market_and_predictions
        for _, row in market_df.head(100).iterrows():
            actual = row["actual_tmax"]
            outcome = row["actual_outcome"]
            direction = row["direction"]
            low = row["threshold_low"]
            high = row["threshold_high"]

            if direction == "below":
                expected = 1 if actual < high else 0
                assert outcome == expected, (
                    f"Below {high}: actual={actual}, expected={expected}, got={outcome}"
                )
            elif direction == "above":
                expected = 1 if actual >= low else 0
                assert outcome == expected, (
                    f"Above {low}: actual={actual}, expected={expected}, got={outcome}"
                )
            elif direction == "between":
                # Convention: between uses half-open interval [low, high)
                expected = 1 if (low <= actual < high) else 0
                assert outcome == expected, (
                    f"Between [{low},{high}): actual={actual}, expected={expected}, got={outcome}"
                )

    def test_model_sigma_seasonal_variation(self, full_year_data):
        """Model sigma should vary by season (wider in winter, narrower in summer)."""
        _, predictions_df = full_year_data
        predictions_df = predictions_df.copy()
        predictions_df["month"] = pd.to_datetime(predictions_df["date"]).dt.month

        winter_sigma = predictions_df[predictions_df["month"].isin([12, 1, 2])]["model_sigma"].mean()
        summer_sigma = predictions_df[predictions_df["month"].isin([6, 7, 8])]["model_sigma"].mean()

        # Winter uncertainty should be higher
        assert winter_sigma > summer_sigma, (
            f"Winter sigma ({winter_sigma:.2f}) should be > summer sigma ({summer_sigma:.2f})"
        )

    def test_market_prob_not_degenerate(self, market_and_predictions):
        """No market prob should be exactly 0 or 1 (degenerate)."""
        market_df, _ = market_and_predictions
        assert (market_df["market_prob"] > 0.01).all()
        assert (market_df["market_prob"] < 0.99).all()
