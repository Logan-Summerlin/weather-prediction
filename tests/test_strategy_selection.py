"""Tests for the Phase 3 real-price strategy sweep + promotion decision."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import strategy_selection as ss
from src.trading import TradingStrategy


# ---------------------------------------------------------------------------
# Synthetic frame helpers
# ---------------------------------------------------------------------------

def _frame(n_days=120, edge=True, seed=0):
    """Build a synthetic real-price frame.

    When *edge* is True the model is systematically better-calibrated than the
    market (so a +EV strategy makes money); when False the model is noise.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        outcome = int(rng.rand() < 0.4)
        if edge:
            model_prob = 0.4 + (0.25 if outcome else -0.2) * rng.rand()
            market_mid = 0.4 + rng.normal(0, 0.03)
        else:
            model_prob = np.clip(0.4 + rng.normal(0, 0.2), 0.05, 0.95)
            market_mid = np.clip(0.4 + rng.normal(0, 0.05), 0.05, 0.95)
        rows.append({
            "date": d, "ticker": f"T-{d.date()}", "bucket": "B",
            "model_prob": np.clip(model_prob, 0.02, 0.98),
            "market_mid": np.clip(market_mid, 0.02, 0.98),
            "bid": market_mid - 0.02, "ask": market_mid + 0.02,
            "half_spread": 0.02, "actual_outcome": outcome, "season": "DJF",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def test_chronological_split_is_ordered_and_disjoint():
    frame = _frame(100)
    train, holdout = ss.chronological_split(frame, val_frac=0.4)
    assert len(train) + len(holdout) == len(frame)
    assert train["date"].max() < holdout["date"].min()
    # ~40% of dates in holdout
    assert 0.3 < len(holdout) / len(frame) < 0.5


def test_chronological_split_explicit_val_start():
    frame = _frame(100)
    train, holdout = ss.chronological_split(frame, val_start="2025-03-01")
    assert train["date"].max() < pd.Timestamp("2025-03-01")
    assert holdout["date"].min() >= pd.Timestamp("2025-03-01")


# ---------------------------------------------------------------------------
# Backtest + Brier
# ---------------------------------------------------------------------------

def test_backtest_metrics_shape_and_no_leakage():
    frame = _frame(60)
    strat = TradingStrategy(name="t", ev_threshold=0.03, fee_rate=0.07,
                            bankroll=1000.0)
    m = ss.backtest_strategy(frame, strat)
    for key in ("total_pnl", "sharpe_ratio", "n_trades", "max_drawdown_pct",
                "model_brier", "market_brier", "brier_edge"):
        assert key in m
    assert m["n_trades"] >= 0
    assert -100.0 <= m["max_drawdown_pct"] <= 0.0


def test_backtest_bankruptcy_halts():
    # Tiny bankroll fully consumed by a single losing fill -> bust next day.
    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    frame = pd.DataFrame({
        "date": dates, "ticker": "T", "bucket": "B",
        "model_prob": 0.95, "market_mid": 0.5, "bid": 0.5, "ask": 0.5,
        "half_spread": 0.0, "actual_outcome": 0, "season": "DJF",
    })
    strat = TradingStrategy(name="t", ev_threshold=0.02, fee_rate=0.07,
                            sizing_method="fixed", fixed_size=10,
                            max_contracts=10, bankroll=1.0)
    m = ss.backtest_strategy(frame, strat, initial_bankroll=1.0)
    assert m["busted"] is True
    assert m["bust_date"] is not None


def test_backtest_never_goes_negative():
    # Capped stakes (Phase 0 fix): bankroll/drawdown stay bounded even on a
    # relentlessly losing stream.
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    frame = pd.DataFrame({
        "date": dates, "ticker": "T", "bucket": "B",
        "model_prob": 0.95, "market_mid": 0.5, "bid": 0.48, "ask": 0.52,
        "half_spread": 0.02, "actual_outcome": 0, "season": "DJF",
    })
    strat = TradingStrategy(name="t", ev_threshold=0.02, fee_rate=0.07,
                            sizing_method="full_kelly", max_position_frac=0.5,
                            bankroll=100.0)
    m = ss.backtest_strategy(frame, strat, initial_bankroll=100.0)
    assert m["final_bankroll"] >= 0.0
    assert -100.0 <= m["max_drawdown_pct"] <= 0.0


def test_compute_brier_edge_sign():
    frame = _frame(80, edge=True, seed=3)
    b = ss.compute_brier(frame)
    # edge frame: model should not be worse than market by much; sign defined
    assert np.isfinite(b["model_brier"]) and np.isfinite(b["market_brier"])
    assert b["brier_edge"] == pytest.approx(b["market_brier"] - b["model_brier"])


def test_select_model_column_picks_lowest_brier():
    df = pd.DataFrame({
        "actual_outcome": [1, 0, 1, 0],
        "u9_kitchen_prob": [0.9, 0.1, 0.8, 0.2],   # good
        "u3_mlp_prob": [0.5, 0.5, 0.5, 0.5],        # uninformative
    })
    col = ss.select_model_column(df, candidates=["u3_mlp_prob", "u9_kitchen_prob"])
    assert col == "u9_kitchen_prob"


# ---------------------------------------------------------------------------
# Sweep selection
# ---------------------------------------------------------------------------

def test_sweep_selects_a_strategy():
    train = _frame(120, edge=True, seed=1)
    grid = ss.default_strategy_grid(bankroll=1000.0)
    best, metrics, rows = ss.sweep_strategies(train, grid, min_train_trades=5)
    assert isinstance(best, TradingStrategy)
    assert len(rows) == len(grid)
    assert metrics["n_trades"] >= 0


# ---------------------------------------------------------------------------
# Promotion decision (Phase 3.2)
# ---------------------------------------------------------------------------

def test_decide_promotion_promoted():
    metrics = {
        "model_brier": 0.10, "market_brier": 0.11, "total_pnl": 120.0,
        "sharpe_ratio": 1.4, "n_trades": 80, "max_drawdown_pct": -12.0,
        "busted": False,
    }
    decision = ss.decide_promotion(metrics)
    assert decision["status"] == "PROMOTED"
    assert decision["failed_checks"] == []


@pytest.mark.parametrize("override,failed", [
    ({"model_brier": 0.12}, "model_beats_market_brier"),
    ({"total_pnl": -5.0}, "positive_pnl"),
    ({"sharpe_ratio": 0.5}, "sharpe_ok"),
    ({"n_trades": 10}, "min_trades"),
    ({"max_drawdown_pct": -45.0}, "drawdown_ok"),
])
def test_decide_promotion_monitor(override, failed):
    metrics = {
        "model_brier": 0.10, "market_brier": 0.11, "total_pnl": 120.0,
        "sharpe_ratio": 1.4, "n_trades": 80, "max_drawdown_pct": -12.0,
        "busted": False,
    }
    metrics.update(override)
    decision = ss.decide_promotion(metrics)
    assert decision["status"] == "MONITOR"
    assert failed in decision["failed_checks"]


# ---------------------------------------------------------------------------
# strategy.json round-trip
# ---------------------------------------------------------------------------

def test_strategy_config_roundtrip():
    strat = TradingStrategy(
        name="s", ev_threshold=0.05, sizing_method="capped_kelly",
        kelly_fraction=0.2, max_position_frac=0.05, fee_rate=0.07,
        max_contracts=10, bankroll=1000.0,
    )
    cfg = ss.strategy_to_config(strat)
    rebuilt = ss.strategy_from_config(cfg)
    assert rebuilt.ev_threshold == strat.ev_threshold
    assert rebuilt.sizing_method == strat.sizing_method
    assert rebuilt.kelly_fraction_param == strat.kelly_fraction_param
    assert rebuilt.max_position_frac == strat.max_position_frac


def test_save_strategy_json_writes_file(tmp_path):
    frame = _frame(60)
    train, holdout = ss.chronological_split(frame)
    strat = TradingStrategy(name="s", ev_threshold=0.03, fee_rate=0.07,
                            bankroll=1000.0)
    tm = ss.backtest_strategy(train, strat)
    hm = ss.backtest_strategy(holdout, strat)
    promo = ss.decide_promotion(hm)
    path = tmp_path / "strategy.json"
    ss.save_strategy_json("chi", strat, "u9_kitchen_prob", train, holdout,
                          tm, hm, promo, path=str(path))
    data = json.loads(path.read_text())
    assert data["city_code"] == "chi"
    assert data["promotion_status"] in ("PROMOTED", "MONITOR")
    assert "trades" not in data["holdout_metrics"]  # trade list stripped
    assert data["strategy"]["ev_threshold"] == 0.03
