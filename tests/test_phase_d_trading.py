"""
Tests for Phase D trading quality controls.

Validates:
  - compute_conservative_ev() function
  - validate_kelly_drawdown_policy() function
  - create_conservative_strategy() factory
  - validate_audit_log() function
  - validate_audit_log_file() function
  - validate_run_audit_completeness() function
  - Conservative constants
"""

import os
import sys
import json
import math
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trading import (
    compute_conservative_ev,
    validate_kelly_drawdown_policy,
    create_conservative_strategy,
    compute_ev_best,
    TradingStrategy,
    CONSERVATIVE_FEE_RATE,
    CONSERVATIVE_FEE_TOTAL,
    CONSERVATIVE_EV_THRESHOLD,
    CONSERVATIVE_MIN_EV,
    SLIPPAGE_BPS,
)
from src.live_trading import (
    validate_audit_log,
    validate_audit_log_file,
    validate_run_audit_completeness,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_phase_d_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def conservative_strategy():
    return create_conservative_strategy()


@pytest.fixture
def valid_audit():
    """A complete, valid audit log dict."""
    return {
        "city_code": "chi",
        "kalshi_ticker": "KXHIGHCHI",
        "date": "2025-03-08",
        "mode": "paper",
        "n_trades": 2,
        "total_pnl": 15.50,
        "kill_switch": {
            "city_code": "chi",
            "is_active": False,
            "reason": "",
            "activated_at": "",
            "max_daily_loss": 500.0,
            "max_consecutive_losses": 10,
            "current_daily_loss": 0.0,
            "consecutive_losses": 0,
        },
        "strategy": {
            "name": "test",
            "ev_threshold": 0.03,
            "sizing_method": "fractional_kelly",
            "fee_rate": 0.07,
        },
        "trades": [
            {
                "city_code": "chi",
                "date": "2025-03-08",
                "ticker": "KXHIGHCHI-2025-03-08",
                "bucket_label": "70-74 F",
                "direction": "YES",
                "size": 5,
                "model_prob": 0.65,
                "market_price": 0.45,
                "ev": 0.04,
                "mode": "paper",
                "pnl": 10.0,
                "settled": True,
            },
            {
                "city_code": "chi",
                "date": "2025-03-08",
                "ticker": "KXHIGHCHI-2025-03-08",
                "bucket_label": "75-79 F",
                "direction": "NO",
                "size": 3,
                "model_prob": 0.20,
                "market_price": 0.35,
                "ev": 0.03,
                "mode": "paper",
                "pnl": 5.50,
                "settled": True,
            },
        ],
    }


# ===========================================================================
# Conservative Constants Tests
# ===========================================================================

class TestConservativeConstants:
    """Tests for Phase D conservative constants."""

    def test_fee_rate(self):
        assert CONSERVATIVE_FEE_RATE == 0.07

    def test_slippage(self):
        assert SLIPPAGE_BPS == 0.02

    def test_total_cost(self):
        assert abs(CONSERVATIVE_FEE_TOTAL - 0.09) < 1e-10

    def test_ev_threshold(self):
        assert CONSERVATIVE_EV_THRESHOLD == 0.03

    def test_min_ev(self):
        assert CONSERVATIVE_MIN_EV == 0.015


# ===========================================================================
# compute_conservative_ev() Tests
# ===========================================================================

class TestComputeConservativeEV:
    """Tests for compute_conservative_ev()."""

    def test_more_conservative_than_standard(self):
        """Conservative EV should be lower than standard EV."""
        standard = compute_ev_best(0.80, 0.50, fee_rate=0.07)
        conservative = compute_conservative_ev(0.80, 0.50, fee_rate=0.07, slippage=0.02)
        assert conservative["ev"] < standard["ev"]

    def test_total_cost_in_result(self):
        """Result should include total_cost field."""
        result = compute_conservative_ev(0.80, 0.50)
        assert "total_cost" in result
        assert abs(result["total_cost"] - 0.09) < 1e-10

    def test_direction_matches_standard(self):
        """Direction should match standard EV for strong signals."""
        result = compute_conservative_ev(0.90, 0.50)
        assert result["direction"] == "YES"

        result = compute_conservative_ev(0.10, 0.50)
        assert result["direction"] == "NO"

    def test_nan_handling(self):
        """NaN inputs should return NONE direction."""
        result = compute_conservative_ev(float("nan"), 0.50)
        assert result["direction"] == "NONE"

    def test_custom_slippage(self):
        """Custom slippage should be applied."""
        low_slip = compute_conservative_ev(0.80, 0.50, slippage=0.01)
        high_slip = compute_conservative_ev(0.80, 0.50, slippage=0.05)
        assert low_slip["ev"] > high_slip["ev"]

    def test_zero_slippage_matches_standard(self):
        """Zero slippage should match standard EV."""
        standard = compute_ev_best(0.80, 0.50, fee_rate=0.07)
        conservative = compute_conservative_ev(0.80, 0.50, fee_rate=0.07, slippage=0.0)
        assert abs(conservative["ev"] - standard["ev"]) < 1e-10

    def test_result_keys(self):
        """Result should have all expected keys."""
        result = compute_conservative_ev(0.70, 0.50)
        for key in ["direction", "ev", "ev_yes", "ev_no", "total_cost"]:
            assert key in result


# ===========================================================================
# validate_kelly_drawdown_policy() Tests
# ===========================================================================

class TestValidateKellyDrawdownPolicy:
    """Tests for validate_kelly_drawdown_policy()."""

    def test_conservative_strategy_passes(self, conservative_strategy):
        """Conservative strategy should pass all checks."""
        result = validate_kelly_drawdown_policy(conservative_strategy)
        assert result["is_valid"] is True
        assert len(result["warnings"]) == 0

    def test_aggressive_strategy_fails(self):
        """Very aggressive strategy should fail validation."""
        aggressive = TradingStrategy(
            name="Aggressive",
            max_position_frac=0.50,  # 50% of bankroll per trade
            kelly_fraction=1.0,  # full Kelly
            sizing_method="fractional_kelly",
            bankroll=10000.0,
        )
        result = validate_kelly_drawdown_policy(aggressive, max_acceptable_drawdown=0.20)
        assert result["is_valid"] is False
        assert len(result["warnings"]) > 0

    def test_max_position_exceeds_drawdown(self):
        """max_position_frac > max_acceptable_drawdown should fail."""
        strategy = TradingStrategy(
            name="BigPos",
            max_position_frac=0.30,
            sizing_method="fixed",
            bankroll=10000.0,
        )
        result = validate_kelly_drawdown_policy(strategy, max_acceptable_drawdown=0.20)
        assert not result["checks"][0]["passed"]
        assert "max_position_frac" in result["recommended_adjustments"]

    def test_recommended_adjustments_provided(self):
        """Failing strategies should get recommended adjustments."""
        strategy = TradingStrategy(
            name="TooAggressive",
            max_position_frac=0.50,
            kelly_fraction=1.0,
            sizing_method="fractional_kelly",
            bankroll=10000.0,
        )
        result = validate_kelly_drawdown_policy(strategy, max_acceptable_drawdown=0.10)
        assert len(result["recommended_adjustments"]) > 0

    def test_output_keys(self, conservative_strategy):
        """Output should have all expected keys."""
        result = validate_kelly_drawdown_policy(conservative_strategy)
        for key in ["is_valid", "checks", "warnings", "recommended_adjustments"]:
            assert key in result

    def test_checks_have_correct_structure(self, conservative_strategy):
        """Each check should have name, passed, detail."""
        result = validate_kelly_drawdown_policy(conservative_strategy)
        for check in result["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "detail" in check

    def test_fixed_sizing_method_skips_var_check(self):
        """Non-Kelly sizing should skip the VaR check."""
        strategy = TradingStrategy(
            name="FixedSize",
            max_position_frac=0.10,
            sizing_method="fixed",
            bankroll=10000.0,
        )
        result = validate_kelly_drawdown_policy(strategy)
        check_names = [c["name"] for c in result["checks"]]
        assert "fractional_kelly_daily_var" not in check_names


# ===========================================================================
# create_conservative_strategy() Tests
# ===========================================================================

class TestCreateConservativeStrategy:
    """Tests for create_conservative_strategy()."""

    def test_creates_valid_strategy(self):
        """Should create a valid TradingStrategy."""
        strategy = create_conservative_strategy()
        assert isinstance(strategy, TradingStrategy)

    def test_conservative_defaults(self):
        """Strategy should use Phase D conservative defaults."""
        strategy = create_conservative_strategy()
        assert strategy.fee_rate == CONSERVATIVE_FEE_TOTAL
        assert strategy.ev_threshold == CONSERVATIVE_EV_THRESHOLD
        assert strategy.min_ev == CONSERVATIVE_MIN_EV
        assert strategy.sizing_method == "fractional_kelly"
        assert strategy.kelly_fraction_param == 0.20
        assert strategy.max_position_frac == 0.08
        assert strategy.max_contracts == 25

    def test_custom_name_and_bankroll(self):
        """Custom name and bankroll should be applied."""
        strategy = create_conservative_strategy(name="MyStrat", bankroll=50000.0)
        assert strategy.name == "MyStrat"
        assert strategy.bankroll == 50000.0

    def test_passes_drawdown_validation(self):
        """Conservative strategy should pass drawdown validation."""
        strategy = create_conservative_strategy()
        result = validate_kelly_drawdown_policy(strategy)
        assert result["is_valid"] is True


# ===========================================================================
# validate_audit_log() Tests
# ===========================================================================

class TestValidateAuditLog:
    """Tests for validate_audit_log()."""

    def test_valid_audit_passes(self, valid_audit):
        """Complete valid audit should pass."""
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is True
        assert len(result["missing_fields"]) == 0
        assert len(result["invalid_trades"]) == 0

    def test_missing_top_level_key(self, valid_audit):
        """Missing top-level key should be detected."""
        del valid_audit["city_code"]
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False
        assert "city_code" in result["missing_fields"]

    def test_missing_trade_key(self, valid_audit):
        """Missing trade key should be detected."""
        del valid_audit["trades"][0]["direction"]
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False
        assert len(result["invalid_trades"]) == 1
        assert result["invalid_trades"][0]["trade_index"] == 0

    def test_invalid_direction(self, valid_audit):
        """Invalid direction should be flagged."""
        valid_audit["trades"][0]["direction"] = "MAYBE"
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False
        issues = result["invalid_trades"][0]["issues"]
        assert any("direction" in i for i in issues)

    def test_invalid_size(self, valid_audit):
        """Non-positive size should be flagged."""
        valid_audit["trades"][0]["size"] = 0
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False

    def test_model_prob_out_of_range(self, valid_audit):
        """model_prob outside [0,1] should be flagged."""
        valid_audit["trades"][0]["model_prob"] = 1.5
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False

    def test_market_price_out_of_range(self, valid_audit):
        """market_price outside [0,1] should be flagged."""
        valid_audit["trades"][0]["market_price"] = -0.1
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False

    def test_invalid_mode(self, valid_audit):
        """Invalid mode should be flagged."""
        valid_audit["trades"][0]["mode"] = "demo"
        result = validate_audit_log(valid_audit)
        assert result["is_valid"] is False

    def test_n_trades_mismatch_warning(self, valid_audit):
        """n_trades mismatch should produce warning."""
        valid_audit["n_trades"] = 99
        result = validate_audit_log(valid_audit)
        assert len(result["warnings"]) > 0
        assert any("n_trades" in w for w in result["warnings"])

    def test_empty_trades_valid(self):
        """Audit with no trades should be valid if all keys present."""
        audit = {
            "city_code": "chi",
            "kalshi_ticker": "KXHIGHCHI",
            "date": "2025-03-08",
            "mode": "paper",
            "n_trades": 0,
            "total_pnl": 0.0,
            "kill_switch": {},
            "strategy": {},
            "trades": [],
        }
        result = validate_audit_log(audit)
        assert result["is_valid"] is True

    def test_output_keys(self, valid_audit):
        """Output should have all expected keys."""
        result = validate_audit_log(valid_audit)
        for key in ["is_valid", "missing_fields", "invalid_trades", "warnings"]:
            assert key in result


# ===========================================================================
# validate_audit_log_file() Tests
# ===========================================================================

class TestValidateAuditLogFile:
    """Tests for validate_audit_log_file()."""

    def test_valid_file(self, tmp_dir, valid_audit):
        """Valid JSON audit file should pass."""
        path = os.path.join(tmp_dir, "audit.json")
        with open(path, "w") as f:
            json.dump(valid_audit, f)
        result = validate_audit_log_file(path)
        assert result["is_valid"] is True
        assert result["file_path"] == path

    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_audit_log_file("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_dir):
        """Invalid JSON should raise JSONDecodeError."""
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("not json {{{")
        with pytest.raises(json.JSONDecodeError):
            validate_audit_log_file(path)


# ===========================================================================
# validate_run_audit_completeness() Tests
# ===========================================================================

class TestValidateRunAuditCompleteness:
    """Tests for validate_run_audit_completeness()."""

    def test_all_present_and_valid(self, tmp_dir, valid_audit):
        """All dates present and valid should give 100% completeness."""
        for day in range(1, 4):
            date_str = f"2025-03-0{day}"
            audit = dict(valid_audit)
            audit["date"] = date_str
            path = os.path.join(tmp_dir, f"trading_audit_chi_{date_str}.json")
            with open(path, "w") as f:
                json.dump(audit, f)

        result = validate_run_audit_completeness(
            tmp_dir, "chi", "2025-03-01", "2025-03-03",
        )
        assert result["total_expected"] == 3
        assert result["total_found"] == 3
        assert result["total_valid"] == 3
        assert result["completeness_pct"] == 100.0
        assert len(result["missing_dates"]) == 0
        assert len(result["invalid_logs"]) == 0

    def test_missing_dates_detected(self, tmp_dir, valid_audit):
        """Missing audit files should be reported."""
        # Only create file for day 1
        path = os.path.join(tmp_dir, "trading_audit_chi_2025-03-01.json")
        with open(path, "w") as f:
            json.dump(valid_audit, f)

        result = validate_run_audit_completeness(
            tmp_dir, "chi", "2025-03-01", "2025-03-03",
        )
        assert result["total_expected"] == 3
        assert result["total_found"] == 1
        assert len(result["missing_dates"]) == 2
        assert "2025-03-02" in result["missing_dates"]
        assert "2025-03-03" in result["missing_dates"]

    def test_invalid_log_detected(self, tmp_dir):
        """Invalid audit content should be reported."""
        # Create file with missing required keys
        bad_audit = {"trades": []}
        path = os.path.join(tmp_dir, "trading_audit_chi_2025-03-01.json")
        with open(path, "w") as f:
            json.dump(bad_audit, f)

        result = validate_run_audit_completeness(
            tmp_dir, "chi", "2025-03-01", "2025-03-01",
        )
        assert result["total_expected"] == 1
        assert result["total_found"] == 1
        assert result["total_valid"] == 0
        assert len(result["invalid_logs"]) == 1

    def test_empty_directory(self, tmp_dir):
        """Empty directory should report all dates missing."""
        result = validate_run_audit_completeness(
            tmp_dir, "chi", "2025-03-01", "2025-03-05",
        )
        assert result["total_expected"] == 5
        assert result["total_found"] == 0
        assert result["completeness_pct"] == 0.0
        assert len(result["missing_dates"]) == 5

    def test_output_keys(self, tmp_dir):
        """Output should have all expected keys."""
        result = validate_run_audit_completeness(
            tmp_dir, "chi", "2025-03-01", "2025-03-01",
        )
        for key in ["total_expected", "total_found", "total_valid",
                     "missing_dates", "invalid_logs", "completeness_pct"]:
            assert key in result
