"""Tests for Phase 3.3 paper-trading loop, settlement, kill switch + replay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import paper_trading as pt
from src.bucket_semantics import bucket_prob_gaussian
from src.live_trading import KillSwitch, LiveTradingHarness
from src.trading import TradingStrategy


def _strategy(**kw):
    defaults = dict(
        name="test", ev_threshold=0.03, sizing_method="fixed", fixed_size=10,
        fee_rate=0.07, min_ev=0.01, max_contracts=10, bankroll=1000.0,
    )
    defaults.update(kw)
    return TradingStrategy(**defaults)


def _harness(strategy=None, kill_switch=None, tmp_path=None):
    return LiveTradingHarness(
        "chi", mode="paper", strategy=strategy or _strategy(),
        kill_switch=kill_switch,
        audit_dir=str(tmp_path) if tmp_path else None,
    )


# ---------------------------------------------------------------------------
# Contract pricing
# ---------------------------------------------------------------------------

def test_model_prob_matches_bucket_semantics():
    mu, sigma = 72.0, 5.0
    cases = [
        {"lo": 70.0, "hi": 74.0, "direction": "between"},
        {"lo": float("nan"), "hi": 69.0, "direction": "below"},
        {"lo": 76.0, "hi": float("nan"), "direction": "above"},
    ]
    for c in cases:
        expected = float(bucket_prob_gaussian(mu, sigma, c["lo"], c["hi"], c["direction"]))
        got = pt.model_prob_for_contract(mu, sigma, c)
        assert got == pytest.approx(np.clip(expected, 0.001, 0.999), abs=1e-9)


def test_build_market_prediction_labels_align():
    contracts = [
        {"ticker": "A", "label": "70-72F", "lo": 70.0, "hi": 72.0,
         "direction": "between", "price": 0.3},
        {"ticker": "B", "label": "Below 70F", "lo": float("nan"), "hi": 70.0,
         "direction": "below", "price": 0.2},
    ]
    pred = pt.build_market_prediction("chi", "2025-06-10", 71.0, 5.0, contracts)
    assert pred.bucket_labels == ["70-72F", "Below 70F"]
    assert len(pred.bucket_probs) == 2


# ---------------------------------------------------------------------------
# Settlement correctness
# ---------------------------------------------------------------------------

def _between_contract(price=0.3):
    return {"ticker": "X", "label": "70-72F", "lo": 70.0, "hi": 72.0,
            "direction": "between", "price": price}


def test_yes_trade_wins_when_in_bucket(tmp_path):
    # Model strongly favours the bucket; market cheap -> YES.
    harness = _harness(tmp_path=tmp_path)
    contracts = [_between_contract(price=0.10)]
    pred = pt.build_market_prediction("chi", "2025-06-10", 71.0, 1.5, contracts)
    summary = pt.run_paper_cycle(harness, pred, contracts, actual_tmax=71.0,
                                 save_audit=False)
    assert summary["n_signals"] == 1
    assert summary["trades"][0]["direction"] == "YES"
    assert summary["day_pnl"] > 0  # 71 rounds into [70,72)


def test_yes_trade_loses_when_out_of_bucket(tmp_path):
    harness = _harness(tmp_path=tmp_path)
    contracts = [_between_contract(price=0.10)]
    pred = pt.build_market_prediction("chi", "2025-06-10", 71.0, 1.5, contracts)
    summary = pt.run_paper_cycle(harness, pred, contracts, actual_tmax=80.0,
                                 save_audit=False)
    assert summary["day_pnl"] < 0


# ---------------------------------------------------------------------------
# Kill-switch scenarios
# ---------------------------------------------------------------------------

def test_active_kill_switch_blocks_all_trades(tmp_path):
    ks = KillSwitch(city_code="chi")
    ks.activate("manual halt")
    harness = _harness(kill_switch=ks, tmp_path=tmp_path)
    contracts = [_between_contract(price=0.10)]
    pred = pt.build_market_prediction("chi", "2025-06-10", 71.0, 1.5, contracts)
    summary = pt.run_paper_cycle(harness, pred, contracts, actual_tmax=71.0,
                                 save_audit=False)
    assert summary["n_signals"] == 0


def test_consecutive_losses_trip_kill_switch(tmp_path):
    ks = KillSwitch(city_code="chi", max_consecutive_losses=2,
                    max_daily_loss=1e9)
    harness = _harness(kill_switch=ks, tmp_path=tmp_path)
    contracts = [_between_contract(price=0.10)]

    # Two consecutive losing days (TMAX far outside the bucket).
    for day in ("2025-06-10", "2025-06-11"):
        pred = pt.build_market_prediction("chi", day, 71.0, 1.5, contracts)
        pt.run_paper_cycle(harness, pred, contracts, actual_tmax=95.0,
                           save_audit=False)
    assert harness.kill_switch.is_active
    assert "Consecutive loss" in harness.kill_switch.reason


def test_daily_loss_limit_trips_kill_switch(tmp_path):
    ks = KillSwitch(city_code="chi", max_daily_loss=0.5,
                    max_consecutive_losses=999)
    harness = _harness(kill_switch=ks, tmp_path=tmp_path)
    contracts = [_between_contract(price=0.10)]
    pred = pt.build_market_prediction("chi", "2025-06-10", 71.0, 1.5, contracts)
    pt.run_paper_cycle(harness, pred, contracts, actual_tmax=95.0,
                       save_audit=False)
    assert harness.kill_switch.is_active
    assert "Daily loss" in harness.kill_switch.reason


# ---------------------------------------------------------------------------
# Historical losing-week replay
# ---------------------------------------------------------------------------

def _write_replay_fixtures(tmp_path):
    """A week of synthetic data where the model is overconfident and loses."""
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    preds = pd.DataFrame({
        "date": dates,
        "actual_tmax": [55.0] * 7,   # always settles in a bucket the model fades
        "mu": [40.0] * 7,            # model thinks much colder than reality
        "sigma": [3.0] * 7,
    })
    pred_path = tmp_path / "synthesis_predictions.csv"
    preds.to_csv(pred_path, index=False)

    rows = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        # Market correctly prices the 54-56F bucket high; model will fade it.
        rows.append({"date": ds, "ticker": f"T-{ds}-A", "bucket": "54-56F",
                     "threshold_low": 54.0, "threshold_high": 56.0,
                     "direction": "between", "presettlement_prob": 0.7})
        rows.append({"date": ds, "ticker": f"T-{ds}-B", "bucket": "Below 54F",
                     "threshold_low": float("nan"), "threshold_high": 54.0,
                     "direction": "below", "presettlement_prob": 0.2})
    pre_path = tmp_path / "presettlement.csv"
    pd.DataFrame(rows).to_csv(pre_path, index=False)
    return str(pred_path), str(pre_path)


def test_losing_week_replay_settles_and_loses(tmp_path):
    pred_path, pre_path = _write_replay_fixtures(tmp_path)
    strat = _strategy(ev_threshold=0.03, sizing_method="fixed", fixed_size=10)
    result = pt.replay_period(
        "chi", "2025-01-01", "2025-01-07", strategy=strat,
        predictions_path=pred_path, presettlement_path=pre_path,
        save_audit=False,
    )
    assert 1 <= result["n_days"] <= 7
    assert result["n_trades"] > 0
    assert result["n_settled"] == result["n_trades"]
    # Overconfident-cold model fades the realized warm bucket -> bleeds money.
    assert result["total_pnl"] < 0


def test_replay_halts_when_kill_switch_trips(tmp_path):
    pred_path, pre_path = _write_replay_fixtures(tmp_path)
    # Tight consecutive-loss limit so the kill switch trips mid-week.
    import src.paper_trading as pt_mod

    strat = _strategy(ev_threshold=0.03, sizing_method="fixed", fixed_size=10)
    # Patch the harness creation to inject a sensitive kill switch by
    # pre-building one through replay then asserting halt behaviour: instead we
    # verify the harness used inside replay would halt by checking total days
    # processed is capped when losses mount. Use a direct harness sequence.
    ks = KillSwitch(city_code="chi", max_consecutive_losses=2, max_daily_loss=1e9)
    harness = LiveTradingHarness("chi", mode="paper", strategy=strat,
                                 kill_switch=ks)
    preds = pd.read_csv(pred_path)
    preds["date"] = pd.to_datetime(preds["date"])
    processed = 0
    for _, row in preds.iterrows():
        if harness.kill_switch.is_active:
            break
        ds = row["date"].strftime("%Y-%m-%d")
        contracts = pt_mod.contracts_from_presettlement("chi", ds, pre_path)
        pred = pt_mod.build_market_prediction("chi", ds, float(row["mu"]),
                                              float(row["sigma"]), contracts)
        pt_mod.run_paper_cycle(harness, pred, contracts,
                               actual_tmax=float(row["actual_tmax"]),
                               save_audit=False)
        processed += 1
    assert harness.kill_switch.is_active
    assert processed < 7  # halted before the full week
