"""Tests for src.mos_features (MOS feature builders + residual forecaster)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mos_features import (
    CITY_MOS_STATION,
    build_mos_features,
    doy_climatology,
    gfs_nam_disagreement,
)


def test_city_mos_station_map_covers_portfolio():
    assert CITY_MOS_STATION["chi"] == "KORD"
    assert CITY_MOS_STATION["phl"] == "KPHL"
    assert CITY_MOS_STATION["atl"] == "KATL"
    assert CITY_MOS_STATION["aus"] == "KAUS"


def test_gfs_nam_disagreement_handles_missing():
    out = gfs_nam_disagreement([70.0, 80.0, np.nan], [73.0, np.nan, 60.0])
    np.testing.assert_allclose(out, [3.0, 0.0, 0.0])


def test_doy_climatology_indexed_1_to_366():
    dates = pd.date_range("2020-01-01", periods=731, freq="D")
    # Peaks near day ~200 (mid-summer) and troughs in winter.
    tmax = 60 + 20 * np.sin(2 * np.pi * (dates.dayofyear - 110) / 365)
    clim = doy_climatology(tmax.to_numpy(), dates)
    assert clim.index.min() == 1 and clim.index.max() == 366
    assert clim.notna().all()
    # Summer climatology warmer than winter.
    assert clim.loc[200] > clim.loc[15]


def test_build_mos_features_columns_and_anomaly():
    dates = pd.date_range("2023-06-01", periods=5, freq="D")
    mos = pd.DataFrame(
        {
            "date": dates,
            "gfs_mos_tmax_f": [80, 82, 79, np.nan, 85],
            "nam_mos_tmax_f": [78, 83, 81, 75, np.nan],
            "mos_ensemble_tmax_f": [79, 82.5, 80, 75, 85],
        }
    )
    climo = pd.Series(75.0, index=np.arange(1, 367))  # flat 75F climo
    feats = build_mos_features(mos, climo)
    assert list(feats.columns) == [
        "mos_ensemble_tmax", "mos_climo_anomaly", "gfs_nam_disagreement"
    ]
    assert len(feats) == 5
    # anomaly = ensemble - 75
    np.testing.assert_allclose(feats["mos_climo_anomaly"].to_numpy(),
                               np.array([79, 82.5, 80, 75, 85]) - 75.0)
    # disagreement row 0 = |80-78| = 2
    assert feats["gfs_nam_disagreement"].iloc[0] == pytest.approx(2.0)


def test_build_mos_features_drops_missing_ensemble():
    dates = pd.date_range("2023-06-01", periods=3, freq="D")
    mos = pd.DataFrame(
        {
            "date": dates,
            "gfs_mos_tmax_f": [80, np.nan, 79],
            "nam_mos_tmax_f": [78, np.nan, 81],
            "mos_ensemble_tmax_f": [79, np.nan, 80],
        }
    )
    feats = build_mos_features(mos, pd.Series(75.0, index=np.arange(1, 367)))
    assert len(feats) == 2  # the all-NaN ensemble day is dropped


# ---------------------------------------------------------------------------
# Residual forecaster (torch)
# ---------------------------------------------------------------------------
def test_mos_residual_trains_and_beats_baseline():
    pytest.importorskip("torch")
    from src.mos_features import predict_mos_residual, train_mos_residual

    rng = np.random.default_rng(0)
    n = 600
    # MOS baseline is good but biased +2F and misses a feature-linear signal.
    feat = rng.normal(0, 1, (n, 3)).astype(np.float32)
    truth = 70 + 3.0 * feat[:, 0] + rng.normal(0, 1.5, n)
    mos = truth + 2.0 - 3.0 * feat[:, 0]  # baseline error correlated with feat[0]
    X = np.column_stack([feat, mos]).astype(np.float32)

    tr, va, te = slice(0, 400), slice(400, 500), slice(500, n)
    net = train_mos_residual(
        X[tr], mos[tr], truth[tr], X[va], mos[va], truth[va], epochs=120, seed=0
    )
    mu, sigma = predict_mos_residual(net, X[te], mos[te])
    model_rmse = np.sqrt(np.mean((mu - truth[te]) ** 2))
    base_rmse = np.sqrt(np.mean((mos[te] - truth[te]) ** 2))
    assert np.all(sigma > 0)
    assert model_rmse < base_rmse  # the correction model improves on raw MOS
