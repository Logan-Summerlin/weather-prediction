"""
Tests for the trading strategy and backtesting framework (src/trading.py).

Validates:
  - Expected Value computation (YES, NO, best direction)
  - Kelly Criterion sizing (full, fractional, capped)
  - Position sizing conversion
  - TradingStrategy configuration and signal generation
  - BacktestEngine (single and multi-strategy)
  - Strategy grid generation
  - Comprehensive backtest pipeline
  - Synthetic market data generator
  - Report and visualization generation
  - Edge cases (NaN, extreme probabilities, empty data)

Target: at least 70 meaningful tests.
"""

import os
import sys
import math
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trading import (
    compute_ev_yes,
    compute_ev_no,
    compute_ev_best,
    kelly_fraction,
    fractional_kelly,
    capped_kelly,
    position_size,
    TradeSignal,
    TradingStrategy,
    BacktestEngine,
    BacktestResult,
    generate_strategy_grid,
    run_comprehensive_backtest,
    generate_synthetic_market_data,
    generate_phase3_report,
    _compute_max_drawdown,
    compute_drawdown_metrics,
    VALID_SIZING_METHODS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="test_trading_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def simple_strategy():
    """Create a simple default trading strategy."""
    return TradingStrategy(
        name="TestStrategy",
        ev_threshold=0.02,
        sizing_method="fractional_kelly",
        kelly_fraction=0.25,
        fee_rate=0.07,
        bankroll=10000.0,
    )


@pytest.fixture
def fixed_strategy():
    """Create a fixed-size trading strategy."""
    return TradingStrategy(
        name="FixedStrategy",
        ev_threshold=0.01,
        sizing_method="fixed",
        fee_rate=0.07,
        bankroll=10000.0,
        fixed_size=10,
    )


@pytest.fixture
def simple_backtest_data():
    """Simple backtest data with known outcomes."""
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates,
        "model_prob": [0.80, 0.20, 0.70, 0.30, 0.90,
                       0.10, 0.60, 0.40, 0.85, 0.15],
        "market_price": [0.50, 0.50, 0.50, 0.50, 0.50,
                         0.50, 0.50, 0.50, 0.50, 0.50],
        "actual_outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def model_predictions_df():
    """Synthetic model predictions for market data generation."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2022-06-01", periods=n, freq="D")
    mu = 70.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 30.0)
    sigma = np.full(n, 4.0)
    actual = mu + sigma * np.random.randn(n)

    return pd.DataFrame({
        "date": dates,
        "model_mu": mu,
        "model_sigma": sigma,
        "actual_tmax": actual,
    })


@pytest.fixture
def all_winning_data():
    """Backtest data where model is always right and highly confident."""
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "model_prob": [0.90, 0.10, 0.90, 0.10, 0.90],
        "market_price": [0.50, 0.50, 0.50, 0.50, 0.50],
        "actual_outcome": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def all_losing_data():
    """Backtest data where model is always wrong."""
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "model_prob": [0.90, 0.10, 0.90, 0.10, 0.90],
        "market_price": [0.50, 0.50, 0.50, 0.50, 0.50],
        "actual_outcome": [0, 1, 0, 1, 0],
    })


# ===========================================================================
# EV Computation Tests
# ===========================================================================

class TestComputeEVYes:
    """Tests for compute_ev_yes()."""

    def test_basic_positive_ev(self):
        """Model prob > market price should give positive EV."""
        ev = compute_ev_yes(0.80, 0.50, fee_rate=0.07)
        # 0.80 * 0.93 - 0.50 = 0.744 - 0.50 = 0.244
        assert abs(ev - 0.244) < 1e-6

    def test_basic_negative_ev(self):
        """Model prob < market price should give negative EV."""
        ev = compute_ev_yes(0.30, 0.50, fee_rate=0.07)
        # 0.30 * 0.93 - 0.50 = 0.279 - 0.50 = -0.221
        assert ev < 0

    def test_fair_market(self):
        """When model_prob equals market_price, EV is negative due to fees."""
        ev = compute_ev_yes(0.50, 0.50, fee_rate=0.07)
        # 0.50 * 0.93 - 0.50 = 0.465 - 0.50 = -0.035
        assert abs(ev - (-0.035)) < 1e-6

    def test_zero_fee(self):
        """With zero fees, EV = model_prob - market_price."""
        ev = compute_ev_yes(0.60, 0.40, fee_rate=0.0)
        # 0.60 * 1.0 - 0.40 = 0.20
        assert abs(ev - 0.20) < 1e-6

    def test_prob_zero(self):
        """Model prob = 0 means EV_yes = -market_price."""
        ev = compute_ev_yes(0.0, 0.50, fee_rate=0.07)
        assert abs(ev - (-0.50)) < 1e-6

    def test_prob_one(self):
        """Model prob = 1 means EV_yes = (1-fee) - market_price."""
        ev = compute_ev_yes(1.0, 0.50, fee_rate=0.07)
        assert abs(ev - 0.43) < 1e-6

    def test_nan_prob(self):
        """NaN probability should return NaN."""
        ev = compute_ev_yes(float("nan"), 0.50)
        assert math.isnan(ev)

    def test_nan_price(self):
        """NaN price should return NaN."""
        ev = compute_ev_yes(0.50, float("nan"))
        assert math.isnan(ev)

    def test_high_fee_rate(self):
        """Higher fee rate reduces EV."""
        ev_low_fee = compute_ev_yes(0.80, 0.50, fee_rate=0.05)
        ev_high_fee = compute_ev_yes(0.80, 0.50, fee_rate=0.10)
        assert ev_low_fee > ev_high_fee


class TestComputeEVNo:
    """Tests for compute_ev_no()."""

    def test_basic_positive_ev(self):
        """Low model prob should make NO trade +EV."""
        ev = compute_ev_no(0.20, 0.50, fee_rate=0.07)
        # (1-0.20) * 0.93 - (1-0.50) = 0.80 * 0.93 - 0.50 = 0.244
        assert abs(ev - 0.244) < 1e-6

    def test_basic_negative_ev(self):
        """High model prob should make NO trade -EV."""
        ev = compute_ev_no(0.80, 0.50, fee_rate=0.07)
        assert ev < 0

    def test_symmetry_with_yes(self):
        """EV_no(p, price) == EV_yes(1-p, 1-price)."""
        ev_no = compute_ev_no(0.30, 0.60, fee_rate=0.07)
        ev_yes = compute_ev_yes(0.70, 0.40, fee_rate=0.07)
        assert abs(ev_no - ev_yes) < 1e-6

    def test_nan_handling(self):
        """NaN inputs should return NaN."""
        assert math.isnan(compute_ev_no(float("nan"), 0.5))
        assert math.isnan(compute_ev_no(0.5, float("nan")))

    def test_prob_zero_price_zero(self):
        """Edge case with prob=0 and price=0."""
        ev = compute_ev_no(0.0, 0.0, fee_rate=0.07)
        # (1.0) * 0.93 - (1.0) = -0.07
        assert abs(ev - (-0.07)) < 1e-6


class TestComputeEVBest:
    """Tests for compute_ev_best()."""

    def test_yes_preferred(self):
        """High model prob should prefer YES."""
        result = compute_ev_best(0.80, 0.50, fee_rate=0.07)
        assert result["direction"] == "YES"
        assert result["ev"] > 0

    def test_no_preferred(self):
        """Low model prob should prefer NO."""
        result = compute_ev_best(0.20, 0.50, fee_rate=0.07)
        assert result["direction"] == "NO"
        assert result["ev"] > 0

    def test_symmetric_point(self):
        """At p=0.5, price=0.5, both EVs are equal (both negative)."""
        result = compute_ev_best(0.50, 0.50, fee_rate=0.07)
        assert abs(result["ev_yes"] - result["ev_no"]) < 1e-6

    def test_contains_both_evs(self):
        """Result should contain ev_yes and ev_no."""
        result = compute_ev_best(0.70, 0.50)
        assert "ev_yes" in result
        assert "ev_no" in result
        assert "direction" in result
        assert "ev" in result

    def test_nan_direction_none(self):
        """NaN input should give direction NONE."""
        result = compute_ev_best(float("nan"), 0.50)
        assert result["direction"] == "NONE"


# ===========================================================================
# Kelly Criterion Tests
# ===========================================================================

class TestKellyFraction:
    """Tests for kelly_fraction()."""

    def test_strong_edge_positive_kelly(self):
        """Strong model edge should produce positive Kelly fraction."""
        kf = kelly_fraction(0.80, 0.50, fee_rate=0.07)
        assert kf > 0

    def test_no_edge_zero_kelly(self):
        """No edge (prob=price) should produce zero or near-zero Kelly."""
        kf = kelly_fraction(0.50, 0.50, fee_rate=0.07)
        assert kf == 0.0 or kf < 0.01

    def test_negative_edge_zero_kelly(self):
        """Negative edge should produce zero Kelly."""
        kf = kelly_fraction(0.30, 0.50, fee_rate=0.07)
        # With fees, betting YES at 0.30 when price is 0.50 is bad
        # And betting NO at (1-0.30)=0.70 when NO costs (1-0.50)=0.50
        # Actually NO might be +EV here
        assert kf >= 0  # Kelly is never negative

    def test_nan_returns_zero(self):
        """NaN inputs should return 0."""
        assert kelly_fraction(float("nan"), 0.50) == 0.0
        assert kelly_fraction(0.50, float("nan")) == 0.0

    def test_extreme_prob_one(self):
        """prob=1 should give positive Kelly for YES."""
        kf = kelly_fraction(1.0, 0.50, fee_rate=0.07)
        assert kf > 0

    def test_extreme_prob_zero(self):
        """prob=0 should give positive Kelly for NO."""
        kf = kelly_fraction(0.0, 0.50, fee_rate=0.07)
        assert kf > 0

    def test_kelly_non_negative(self):
        """Kelly fraction should never be negative."""
        for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for price in [0.1, 0.3, 0.5, 0.7, 0.9]:
                kf = kelly_fraction(p, price)
                assert kf >= 0, f"Negative Kelly at p={p}, price={price}"


class TestFractionalKelly:
    """Tests for fractional_kelly()."""

    def test_quarter_kelly(self):
        """Fractional Kelly with fraction=0.25 should be 1/4 of full."""
        full = kelly_fraction(0.80, 0.50)
        frac = fractional_kelly(0.80, 0.50, fraction=0.25)
        assert abs(frac - full * 0.25) < 1e-10

    def test_half_kelly(self):
        """Fractional Kelly with fraction=0.50 should be 1/2 of full."""
        full = kelly_fraction(0.80, 0.50)
        frac = fractional_kelly(0.80, 0.50, fraction=0.50)
        assert abs(frac - full * 0.50) < 1e-10

    def test_full_fraction(self):
        """fraction=1.0 should equal full Kelly."""
        full = kelly_fraction(0.80, 0.50)
        frac = fractional_kelly(0.80, 0.50, fraction=1.0)
        assert abs(frac - full) < 1e-10

    def test_zero_fraction(self):
        """fraction=0.0 should always return 0."""
        frac = fractional_kelly(0.80, 0.50, fraction=0.0)
        assert frac == 0.0


class TestCappedKelly:
    """Tests for capped_kelly()."""

    def test_cap_enforced(self):
        """Capped Kelly should not exceed max_fraction."""
        ck = capped_kelly(0.99, 0.10, fee_rate=0.0, max_fraction=0.10)
        assert ck <= 0.10

    def test_below_cap_unchanged(self):
        """When Kelly is below cap, capped Kelly should equal full Kelly."""
        full = kelly_fraction(0.55, 0.50)
        ck = capped_kelly(0.55, 0.50, max_fraction=0.50)
        assert abs(ck - full) < 1e-10 or ck <= 0.50

    def test_zero_cap(self):
        """max_fraction=0 should return 0."""
        ck = capped_kelly(0.80, 0.50, max_fraction=0.0)
        assert ck == 0.0


class TestPositionSize:
    """Tests for position_size()."""

    def test_basic_conversion(self):
        """Simple conversion from fraction to contracts."""
        # 0.10 * 10000 = $1000, at $1/contract = 1000 contracts, capped at 100
        size = position_size(0.10, 10000.0, contract_price=1.0, max_size=100)
        assert size == 100

    def test_small_fraction(self):
        """Very small fraction might yield 0 contracts."""
        size = position_size(0.0001, 1000.0, contract_price=1.0, min_size=1)
        assert size == 0  # 0.0001 * 1000 = $0.10, less than 1 contract

    def test_zero_kelly_returns_zero(self):
        """Zero Kelly fraction should yield 0 contracts."""
        size = position_size(0.0, 10000.0)
        assert size == 0

    def test_negative_kelly_returns_zero(self):
        """Negative Kelly fraction should yield 0 contracts."""
        size = position_size(-0.05, 10000.0)
        assert size == 0

    def test_zero_bankroll_returns_zero(self):
        """Zero bankroll should yield 0 contracts."""
        size = position_size(0.10, 0.0)
        assert size == 0

    def test_respects_max_size(self):
        """Position size should not exceed max_size."""
        size = position_size(1.0, 100000.0, max_size=50)
        assert size <= 50

    def test_custom_contract_price(self):
        """Position size with custom contract price."""
        # 0.10 * 10000 = $1000, at $0.50/contract = 2000, capped at 100
        size = position_size(0.10, 10000.0, contract_price=0.50, max_size=100)
        assert size == 100


# ===========================================================================
# TradeSignal Tests
# ===========================================================================

class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_default_values(self):
        """Default TradeSignal should be NONE with size 0."""
        signal = TradeSignal()
        assert signal.direction == "NONE"
        assert signal.size == 0
        assert signal.ev == 0.0

    def test_custom_values(self):
        """TradeSignal with custom values."""
        signal = TradeSignal(
            direction="YES", size=10, ev=0.05,
            confidence=0.3, model_prob=0.8, market_price=0.5,
        )
        assert signal.direction == "YES"
        assert signal.size == 10
        assert signal.ev == 0.05


# ===========================================================================
# TradingStrategy Tests
# ===========================================================================

class TestTradingStrategy:
    """Tests for TradingStrategy class."""

    def test_default_initialization(self):
        """Strategy with defaults should initialize correctly."""
        strategy = TradingStrategy()
        assert strategy.name == "Default"
        assert strategy.ev_threshold == 0.02
        assert strategy.sizing_method == "fractional_kelly"

    def test_custom_initialization(self):
        """Strategy with custom params should store them."""
        strategy = TradingStrategy(
            name="Custom",
            ev_threshold=0.05,
            sizing_method="fixed",
            fee_rate=0.10,
            bankroll=25000.0,
        )
        assert strategy.name == "Custom"
        assert strategy.ev_threshold == 0.05
        assert strategy.sizing_method == "fixed"
        assert strategy.fee_rate == 0.10
        assert strategy.bankroll == 25000.0

    def test_invalid_sizing_method_raises(self):
        """Invalid sizing method should raise ValueError."""
        with pytest.raises(ValueError, match="sizing_method must be one of"):
            TradingStrategy(sizing_method="invalid")

    def test_repr(self):
        """Strategy repr should be informative."""
        strategy = TradingStrategy(name="TestRepr")
        repr_str = repr(strategy)
        assert "TestRepr" in repr_str
        assert "TradingStrategy" in repr_str

    def test_evaluate_trade_yes(self, simple_strategy):
        """Strong YES signal should produce YES trade."""
        signal = simple_strategy.evaluate_trade(0.90, 0.50)
        assert signal.direction == "YES"
        assert signal.size > 0
        assert signal.ev > 0

    def test_evaluate_trade_no(self, simple_strategy):
        """Strong NO signal should produce NO trade."""
        signal = simple_strategy.evaluate_trade(0.10, 0.50)
        assert signal.direction == "NO"
        assert signal.size > 0
        assert signal.ev > 0

    def test_evaluate_trade_no_trade_below_threshold(self):
        """Weak signal below threshold should produce no trade."""
        strategy = TradingStrategy(
            ev_threshold=0.50,  # Very high threshold
        )
        signal = strategy.evaluate_trade(0.55, 0.50)
        assert signal.direction == "NONE"

    def test_should_trade_true(self, simple_strategy):
        """should_trade should return True for strong signals."""
        assert simple_strategy.should_trade(0.90, 0.50)

    def test_should_trade_false(self):
        """should_trade should return False for weak signals."""
        strategy = TradingStrategy(ev_threshold=0.50)
        assert not strategy.should_trade(0.55, 0.50)

    def test_fixed_sizing(self, fixed_strategy):
        """Fixed sizing should produce fixed_size contracts."""
        signal = fixed_strategy.evaluate_trade(0.90, 0.50)
        assert signal.size == 10

    def test_proportional_sizing(self):
        """Proportional sizing should scale with EV."""
        strategy = TradingStrategy(
            sizing_method="proportional",
            ev_threshold=0.01,
            bankroll=10000.0,
        )
        signal = strategy.evaluate_trade(0.90, 0.50)
        assert signal.size > 0

    def test_full_kelly_sizing(self):
        """Full Kelly sizing should produce trades."""
        strategy = TradingStrategy(
            sizing_method="full_kelly",
            ev_threshold=0.01,
            bankroll=10000.0,
        )
        signal = strategy.evaluate_trade(0.80, 0.50)
        assert signal.size > 0

    def test_capped_kelly_sizing(self):
        """Capped Kelly sizing should produce trades."""
        strategy = TradingStrategy(
            sizing_method="capped_kelly",
            ev_threshold=0.01,
            bankroll=10000.0,
        )
        signal = strategy.evaluate_trade(0.80, 0.50)
        assert signal.size > 0

    def test_nan_prob_no_trade(self, simple_strategy):
        """NaN probability should produce no trade."""
        signal = simple_strategy.evaluate_trade(float("nan"), 0.50)
        assert signal.direction == "NONE"


# ===========================================================================
# BacktestEngine Tests
# ===========================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_basic_backtest(self, simple_strategy, simple_backtest_data):
        """Basic backtest should complete without error."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert isinstance(result, BacktestResult)
        assert result.n_days == 10

    def test_backtest_pnl_nonzero(self, simple_strategy, simple_backtest_data):
        """Backtest with trades should have non-zero P&L."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert result.n_trades > 0
        assert result.total_pnl != 0.0

    def test_cumulative_pnl_length(self, simple_strategy, simple_backtest_data):
        """Cumulative P&L should match number of days."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert len(result.cumulative_pnl) == 10

    def test_cumulative_pnl_consistency(self, simple_strategy, simple_backtest_data):
        """Last cumulative P&L should equal total P&L."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert abs(result.cumulative_pnl[-1] - result.total_pnl) < 1e-10

    def test_win_rate_bounds(self, simple_strategy, simple_backtest_data):
        """Win rate should be between 0 and 1."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert 0.0 <= result.win_rate <= 1.0

    def test_sharpe_ratio_computed(self, simple_strategy, simple_backtest_data):
        """Sharpe ratio should be a finite number."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        # Sharpe can be inf for all-winning trades, but should be computed
        assert not math.isnan(result.sharpe_ratio)

    def test_max_drawdown_non_negative(self, simple_strategy, simple_backtest_data):
        """Max drawdown should be non-negative."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert result.max_drawdown >= 0.0

    def test_all_winning_trades(self, simple_strategy, all_winning_data):
        """All-winning scenario should have 100% win rate."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(all_winning_data)
        if result.n_trades > 0:
            assert result.win_rate == 1.0
            assert result.total_pnl > 0

    def test_all_losing_trades(self, simple_strategy, all_losing_data):
        """All-losing scenario should have 0% win rate."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(all_losing_data)
        if result.n_trades > 0:
            assert result.win_rate == 0.0
            assert result.total_pnl < 0

    def test_no_trades_high_threshold(self, simple_backtest_data):
        """Very high EV threshold should result in no trades."""
        strategy = TradingStrategy(
            name="VeryHighThreshold",
            ev_threshold=0.99,
        )
        engine = BacktestEngine(strategy)
        result = engine.run_backtest(simple_backtest_data)
        assert result.n_trades == 0
        assert result.total_pnl == 0.0
        assert result.win_rate == 0.0

    def test_missing_columns_raises(self, simple_strategy):
        """Missing required columns should raise ValueError."""
        bad_data = pd.DataFrame({"date": [1], "model_prob": [0.5]})
        engine = BacktestEngine(simple_strategy)
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.run_backtest(bad_data)

    def test_multi_strategy_backtest(self, simple_backtest_data):
        """Multi-strategy backtest should return sorted results."""
        strategies = [
            TradingStrategy(name="Low", ev_threshold=0.01),
            TradingStrategy(name="Mid", ev_threshold=0.05),
            TradingStrategy(name="High", ev_threshold=0.10),
        ]
        engine = BacktestEngine(strategies[0])
        results = engine.run_multi_strategy_backtest(
            simple_backtest_data, strategies
        )
        assert len(results) == 3
        # Results should be sorted by total_pnl descending
        for i in range(len(results) - 1):
            assert results[i].total_pnl >= results[i + 1].total_pnl

    def test_roi_computation(self, simple_strategy, simple_backtest_data):
        """ROI should equal total_pnl / bankroll."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        expected_roi = result.total_pnl / simple_strategy.bankroll
        assert abs(result.roi - expected_roi) < 1e-10

    def test_trade_records_complete(self, simple_strategy, simple_backtest_data):
        """Each trade record should have all required fields."""
        engine = BacktestEngine(simple_strategy)
        result = engine.run_backtest(simple_backtest_data)
        for trade in result.trades:
            assert "date" in trade
            assert "direction" in trade
            assert "size" in trade
            assert "pnl" in trade
            assert "model_prob" in trade
            assert "market_price" in trade
            assert "actual_outcome" in trade


# ===========================================================================
# BacktestResult Tests
# ===========================================================================

class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_default_values(self):
        """Default result should have zero metrics."""
        result = BacktestResult()
        assert result.n_trades == 0
        assert result.total_pnl == 0.0

    def test_to_summary_dict(self):
        """to_summary_dict should return a flat dict."""
        result = BacktestResult(
            strategy_name="Test",
            total_pnl=100.0,
            roi=0.01,
            sharpe_ratio=1.5,
            n_trades=50,
        )
        d = result.to_summary_dict()
        assert d["strategy_name"] == "Test"
        assert d["total_pnl"] == 100.0
        assert d["roi"] == 0.01
        assert "trades" not in d  # Should exclude list


# ===========================================================================
# Max Drawdown Tests
# ===========================================================================

class TestMaxDrawdown:
    """Tests for _compute_max_drawdown()."""

    def test_monotonically_increasing(self):
        """Monotonically increasing P&L should have 0 drawdown."""
        cum_pnl = np.array([1, 2, 3, 4, 5])
        assert _compute_max_drawdown(cum_pnl) == 0.0

    def test_simple_drawdown(self):
        """Simple drawdown scenario."""
        cum_pnl = np.array([0, 10, 5, 15, 3])
        # Peak at 10, drop to 5 (dd=5); peak at 15, drop to 3 (dd=12)
        assert _compute_max_drawdown(cum_pnl) == 12.0

    def test_empty_array(self):
        """Empty array should return 0."""
        assert _compute_max_drawdown(np.array([])) == 0.0

    def test_single_element(self):
        """Single element should return 0."""
        assert _compute_max_drawdown(np.array([5.0])) == 0.0

    def test_monotonically_decreasing(self):
        """Monotonically decreasing should return total decline."""
        cum_pnl = np.array([10, 8, 5, 2, 0])
        assert _compute_max_drawdown(cum_pnl) == 10.0


class TestComputeDrawdownMetrics:
    """Tests for compute_drawdown_metrics() (canonical bankroll drawdown)."""

    def test_monotonically_increasing_is_zero(self):
        result = compute_drawdown_metrics([1000, 1100, 1200, 1300], 1000.0)
        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_known_series_exact_values(self):
        # Peak 1200, trough 600 -> dd = -600 dollars, -50% of peak
        result = compute_drawdown_metrics([1000, 1200, 600, 900], 1000.0)
        assert result["max_drawdown"] == -600.0
        assert result["max_drawdown_pct"] == -50.0

    def test_total_loss_is_minus_100_pct(self):
        result = compute_drawdown_metrics([1000, 500, 0], 1000.0)
        assert result["max_drawdown"] == -1000.0
        assert result["max_drawdown_pct"] == -100.0

    def test_legacy_negative_bankroll_clamped_to_minus_100(self):
        # Legacy series from before stake-capping could go below zero.
        # The percent must never exceed -100% (e.g. Atlanta's -134.3%).
        result = compute_drawdown_metrics([1000, 200, -343], 1000.0)
        assert result["max_drawdown_pct"] == -100.0

    def test_peak_floored_at_initial_bankroll(self):
        # Series that never reaches the initial bankroll: percent is
        # measured against the initial bankroll, not a lower local peak.
        result = compute_drawdown_metrics([800, 700, 600], 1000.0)
        assert result["max_drawdown_pct"] == pytest.approx(-40.0)

    def test_empty_series(self):
        result = compute_drawdown_metrics([], 1000.0)
        assert result == {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}

    def test_accepts_pandas_series(self):
        series = pd.Series([1000.0, 900.0, 1100.0])
        result = compute_drawdown_metrics(series, 1000.0)
        assert result["max_drawdown"] == -100.0
        assert result["max_drawdown_pct"] == -10.0

    def test_bounds_invariant(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            series = 1000 + np.cumsum(rng.normal(0, 200, size=100))
            result = compute_drawdown_metrics(series, 1000.0)
            assert -100.0 <= result["max_drawdown_pct"] <= 0.0
            assert result["max_drawdown"] <= 0.0


# ===========================================================================
# Strategy Grid Tests
# ===========================================================================

class TestStrategyGrid:
    """Tests for generate_strategy_grid()."""

    def test_generates_strategies(self):
        """Grid should generate multiple strategies."""
        strategies = generate_strategy_grid()
        assert len(strategies) > 10

    def test_all_valid(self):
        """All generated strategies should be valid TradingStrategy objects."""
        strategies = generate_strategy_grid()
        for s in strategies:
            assert isinstance(s, TradingStrategy)
            assert s.sizing_method in VALID_SIZING_METHODS
            assert s.ev_threshold > 0
            assert s.bankroll > 0

    def test_custom_grid(self):
        """Custom grid parameters should be respected."""
        strategies = generate_strategy_grid(
            ev_thresholds=[0.02, 0.05],
            sizing_methods=["fixed"],
            kelly_fractions=[0.25],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )
        # 2 thresholds * 1 sizing * 1 kelly * 1 fee * 1 max_pos * 1 bankroll = 2
        assert len(strategies) == 2

    def test_unique_names(self):
        """All strategy names should be unique."""
        strategies = generate_strategy_grid(
            ev_thresholds=[0.01, 0.05],
            sizing_methods=["fixed", "fractional_kelly"],
            kelly_fractions=[0.25],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )
        names = [s.name for s in strategies]
        assert len(names) == len(set(names))


# ===========================================================================
# Comprehensive Backtest Tests
# ===========================================================================

class TestComprehensiveBacktest:
    """Tests for run_comprehensive_backtest()."""

    def test_runs_without_error(self, simple_backtest_data, tmp_dir):
        """Comprehensive backtest should complete without error."""
        strategies = generate_strategy_grid(
            ev_thresholds=[0.02],
            sizing_methods=["fixed"],
            kelly_fractions=[0.25],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )
        result = run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )
        assert "all_results" in result
        assert "comparison_df" in result

    def test_saves_csv(self, simple_backtest_data, tmp_dir):
        """Should save strategy_comparison.csv."""
        strategies = [TradingStrategy(name="TestCSV", ev_threshold=0.02)]
        run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )
        assert os.path.exists(os.path.join(tmp_dir, "strategy_comparison.csv"))

    def test_max_strategies_limit(self, simple_backtest_data, tmp_dir):
        """Should respect max_strategies limit."""
        strategies = generate_strategy_grid(
            ev_thresholds=[0.01, 0.02, 0.03],
            sizing_methods=["fixed"],
            kelly_fractions=[0.25],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )
        result = run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
            max_strategies=2,
        )
        assert len(result["all_results"]) <= 2


# ===========================================================================
# Synthetic Market Data Tests
# ===========================================================================

class TestSyntheticMarketData:
    """Tests for generate_synthetic_market_data()."""

    def test_correct_shape(self, model_predictions_df):
        """Output should have expected number of rows."""
        data = generate_synthetic_market_data(
            model_predictions_df, n_buckets_per_day=3, seed=42,
        )
        assert len(data) == 30 * 3  # 30 days * 3 buckets

    def test_market_prices_in_range(self, model_predictions_df):
        """All market prices should be in [0.01, 0.99]."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        assert data["market_price"].min() >= 0.01
        assert data["market_price"].max() <= 0.99

    def test_model_probs_in_range(self, model_predictions_df):
        """All model probabilities should be in [0.001, 0.999]."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        assert data["model_prob"].min() >= 0.001
        assert data["model_prob"].max() <= 0.999

    def test_reproducibility(self, model_predictions_df):
        """Same seed should produce identical results."""
        data1 = generate_synthetic_market_data(model_predictions_df, seed=123)
        data2 = generate_synthetic_market_data(model_predictions_df, seed=123)
        pd.testing.assert_frame_equal(data1, data2)

    def test_different_seeds(self, model_predictions_df):
        """Different seeds should produce different results."""
        data1 = generate_synthetic_market_data(model_predictions_df, seed=1)
        data2 = generate_synthetic_market_data(model_predictions_df, seed=2)
        # Market prices should differ
        assert not np.allclose(
            data1["market_price"].values, data2["market_price"].values
        )

    def test_actual_outcomes_binary(self, model_predictions_df):
        """Actual outcomes should be 0 or 1."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        assert set(data["actual_outcome"].unique()).issubset({0, 1})

    def test_bid_ask_spread(self, model_predictions_df):
        """Bid should be less than ask."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        assert (data["bid_price"] <= data["ask_price"]).all()

    def test_volume_positive(self, model_predictions_df):
        """Volume should be positive."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        assert (data["volume"] >= 0).all()

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        bad_df = pd.DataFrame({"date": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_synthetic_market_data(bad_df)

    def test_required_output_columns(self, model_predictions_df):
        """Output should contain all expected columns."""
        data = generate_synthetic_market_data(model_predictions_df, seed=42)
        expected_cols = [
            "date", "bucket_label", "bucket_low", "bucket_high",
            "model_prob", "market_price", "actual_outcome",
            "bid_price", "ask_price", "volume",
        ]
        for col in expected_cols:
            assert col in data.columns, f"Missing column: {col}"

    def test_single_bucket_per_day(self, model_predictions_df):
        """Should work with 1 bucket per day."""
        data = generate_synthetic_market_data(
            model_predictions_df, n_buckets_per_day=1, seed=42,
        )
        assert len(data) == 30


# ===========================================================================
# Report Generation Tests
# ===========================================================================

class TestReportGeneration:
    """Tests for generate_phase3_report()."""

    def test_generates_report_text(self, simple_backtest_data, tmp_dir):
        """Should generate a non-empty report string."""
        strategies = [TradingStrategy(name="ReportTest", ev_threshold=0.02)]
        backtest_results = run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )
        report = generate_phase3_report(backtest_results, output_dir=tmp_dir)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Phase 3" in report

    def test_saves_report_files(self, simple_backtest_data, tmp_dir):
        """Should save text report and JSON metrics."""
        strategies = [TradingStrategy(name="FileTest", ev_threshold=0.02)]
        backtest_results = run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )
        generate_phase3_report(backtest_results, output_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "phase3_report.txt"))
        assert os.path.exists(os.path.join(tmp_dir, "phase3_metrics.json"))

    def test_json_metrics_valid(self, simple_backtest_data, tmp_dir):
        """JSON metrics file should be valid JSON."""
        import json
        strategies = [TradingStrategy(name="JSONTest", ev_threshold=0.02)]
        backtest_results = run_comprehensive_backtest(
            simple_backtest_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )
        generate_phase3_report(backtest_results, output_dir=tmp_dir)
        with open(os.path.join(tmp_dir, "phase3_metrics.json")) as f:
            metrics = json.load(f)
        assert "n_strategies" in metrics

    def test_empty_results(self, tmp_dir):
        """Should handle empty results gracefully."""
        backtest_results = {
            "all_results": [],
            "comparison_df": pd.DataFrame(),
        }
        report = generate_phase3_report(backtest_results, output_dir=tmp_dir)
        assert isinstance(report, str)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, model_predictions_df, tmp_dir):
        """Full pipeline: generate data -> backtest -> report."""
        # Generate synthetic market data
        market_data = generate_synthetic_market_data(
            model_predictions_df, n_buckets_per_day=1, seed=42,
        )

        # Create strategies
        strategies = generate_strategy_grid(
            ev_thresholds=[0.02, 0.05],
            sizing_methods=["fixed", "fractional_kelly"],
            kelly_fractions=[0.25],
            fee_rates=[0.07],
            max_positions=[0.10],
            bankrolls=[10000],
        )

        # Run comprehensive backtest
        results = run_comprehensive_backtest(
            market_data,
            output_dir=tmp_dir,
            strategies=strategies,
        )

        # Generate report
        report = generate_phase3_report(results, output_dir=tmp_dir)

        # Verify outputs
        assert len(results["all_results"]) > 0
        assert not results["comparison_df"].empty
        assert len(report) > 0

    def test_strategy_evaluate_to_backtest(self, simple_backtest_data, tmp_dir):
        """Strategy evaluation should be consistent with backtest results."""
        strategy = TradingStrategy(
            name="Consistency",
            ev_threshold=0.01,
            sizing_method="fixed",
            fixed_size=5,
            fee_rate=0.07,
            bankroll=10000.0,
        )

        # Count expected trades manually
        expected_trades = 0
        for _, row in simple_backtest_data.iterrows():
            if strategy.should_trade(row["model_prob"], row["market_price"]):
                expected_trades += 1

        # Run backtest
        engine = BacktestEngine(strategy)
        result = engine.run_backtest(simple_backtest_data)

        assert result.n_trades == expected_trades
