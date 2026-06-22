"""Tests for src.model_diagnostics (Phase 2 diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_diagnostics import (
    disagreement_table,
    load_base_predictions,
    model_vs_market,
    pit_diagnostics,
    residual_diagnostics,
    select_model,
    sigma_calibration,
)


@pytest.fixture
def calibrated_df():
    """A well-calibrated heteroscedastic Gaussian sample over two years."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    doy = dates.dayofyear.to_numpy()
    mu = 60 + 25 * np.sin(2 * np.pi * (doy - 100) / 365)
    sigma = np.full(len(dates), 6.0)
    actual = mu + rng.normal(0, 6.0, size=len(dates))
    return pd.DataFrame({"date": dates, "mu": mu, "sigma": sigma, "actual_tmax": actual})


def test_select_model_priority():
    assert select_model(["Persistence", "HeteroscedasticNN"]) == "HeteroscedasticNN"
    assert select_model(["Climatology", "ridge_base"]) == "ridge_base"
    # Unknown set falls back to sorted-first, never raises.
    assert select_model(["zeta", "alpha"]) == "alpha"


def test_load_base_predictions_selects_model(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "model_name": ["Persistence", "ridge_base", "ridge_base"],
            "mu": [50, 51, 52],
            "sigma": [6, 6, 6],
            "actual_tmax": [49, 49, 53],
        }
    )
    path = tmp_path / "base_predictions.csv"
    df.to_csv(path, index=False)
    out, model = load_base_predictions(path)
    assert model == "ridge_base"
    # De-dupes on date, keeps ridge_base rows only.
    assert len(out) == 2
    assert list(out.columns) == ["date", "mu", "sigma", "actual_tmax"]


def test_load_base_predictions_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2023-01-01"], "mu": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_base_predictions(path)


def test_residual_diagnostics_structure(calibrated_df):
    out = residual_diagnostics(calibrated_df)
    assert out["overall"]["n"] == 730
    assert abs(out["overall"]["bias"]) < 1.0  # unbiased by construction
    assert set(out["by_season"]) == {"Winter", "Spring", "Summer", "Fall"}
    assert set(out["by_regime"]) == {"cold", "normal", "hot"}
    # Regime mu ranges are ordered cold < hot.
    assert out["by_regime"]["cold"]["mu_range"][1] <= out["by_regime"]["hot"]["mu_range"][0] + 1e-6


def test_residual_bias_detects_warm_model():
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    mu = np.full(len(dates), 50.0)
    actual = mu + 5.0  # model is 5F too cold (actual warmer)
    df = pd.DataFrame({"date": dates, "mu": mu, "sigma": 6.0, "actual_tmax": actual})
    out = residual_diagnostics(df)
    assert out["overall"]["bias"] == pytest.approx(5.0, abs=1e-6)


def test_sigma_calibration_well_calibrated(calibrated_df):
    out = sigma_calibration(calibrated_df)
    assert out["calibration_ratio"] == pytest.approx(1.0, abs=0.15)
    # Fixture uses a constant 6F sigma: flagged constant, but not a pathology
    # (a 6F width is plausible, unlike the blown-up 54.6F Austin case).
    assert out["constant_sigma"]
    assert not out["sigma_pathology"]


def test_sigma_calibration_detects_pathology():
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    df = pd.DataFrame(
        {"date": dates, "mu": 60.0, "sigma": 54.6, "actual_tmax": 60.0 + np.zeros(300)}
    )
    out = sigma_calibration(df)
    assert out["constant_sigma"]
    assert out["sigma_pathology"]
    assert out["calibration_ratio"] < 0.2


def test_pit_diagnostics_calibrated_is_uniform(calibrated_df):
    summary, pit = pit_diagnostics(calibrated_df)
    assert pit.size == 730
    assert summary["overall"]["is_uniform"]
    assert summary["mean_crps"] > 0


def test_pit_diagnostics_underdispersed_not_uniform():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    actual = rng.normal(60, 10, size=500)
    # sigma far too small => PIT piles up at the tails => non-uniform.
    df = pd.DataFrame({"date": dates, "mu": 60.0, "sigma": 2.0, "actual_tmax": actual})
    summary, _ = pit_diagnostics(df)
    assert not summary["overall"]["is_uniform"]


def _contracts_frame(dates, low, high, market_prob):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": ["T"] * len(dates),
            "bucket": ["b"] * len(dates),
            "threshold_low": low,
            "threshold_high": high,
            "direction": ["between"] * len(dates),
            "strike_type": ["between"] * len(dates),
            "presettlement_prob": market_prob,
        }
    )


def test_model_vs_market_basic():
    dates = pd.date_range("2023-06-01", periods=50, freq="D")
    pred = pd.DataFrame({"date": dates, "mu": 80.0, "sigma": 4.0, "actual_tmax": 80.0})
    contracts = _contracts_frame(dates, [79.0] * 50, [81.0] * 50, [0.4] * 50)
    out = model_vs_market(pred, contracts)
    assert out["n_contracts"] == 50
    assert out["n_days"] == 50
    assert "model_brier" in out and "market_brier" in out
    assert out["verdict"] in {"BEATS_MARKET", "NO_EDGE_MONITOR"}
    # brier_edge sign matches verdict.
    if out["verdict"] == "BEATS_MARKET":
        assert out["brier_edge"] > 0


def test_model_vs_market_no_overlap():
    pred = pd.DataFrame(
        {"date": pd.to_datetime(["2023-01-01"]), "mu": [50.0], "sigma": [5.0], "actual_tmax": [50.0]}
    )
    contracts = _contracts_frame(["2024-01-01"], [49.0], [51.0], [0.5])
    out = model_vs_market(pred, contracts)
    assert out["n_contracts"] == 0
    assert out["verdict"] == "INSUFFICIENT_DATA"


def test_model_vs_market_oos_filter():
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    pred = pd.DataFrame({"date": dates, "mu": 50.0, "sigma": 5.0, "actual_tmax": 50.0})
    contracts = _contracts_frame(dates, [49.0] * 40, [51.0] * 40, [0.5] * 40)
    out = model_vs_market(pred, contracts, oos_start=pd.Timestamp("2023-01-21"))
    assert out["n_days"] == 20


def test_disagreement_table():
    dates = pd.date_range("2023-06-01", periods=10, freq="D")
    pred = pd.DataFrame({"date": dates, "mu": 80.0, "sigma": 4.0, "actual_tmax": 80.0})
    contracts = _contracts_frame(dates, [79.0] * 10, [81.0] * 10, [0.3] * 10)
    tbl = disagreement_table(pred, contracts)
    assert len(tbl) == 10
    assert {"model_prob", "market_prob", "disagreement", "outcome"} <= set(tbl.columns)
    assert (tbl["disagreement"] == tbl["model_prob"] - tbl["market_prob"]).all()
