"""Tests for src.frontal_features and the regime-conditional calibrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration import RegimeConditionalCalibrator
from src.frontal_features import (
    build_frontal_features,
    dewpoint_depression,
    frontal_passage_index,
    is_post_frontal_wind,
    wind_direction_sector,
)


def test_dewpoint_depression():
    np.testing.assert_allclose(dewpoint_depression([70, 60], [50, 58]), [20, 2])


def test_wind_direction_sector_labels():
    assert wind_direction_sector(0) == "N"
    assert wind_direction_sector(90) == "E"
    assert wind_direction_sector(225) == "SW"
    assert wind_direction_sector(315) == "NW"
    assert wind_direction_sector(350) == "N"  # wraps to N
    assert wind_direction_sector(np.nan) is None


def test_is_post_frontal_wind():
    out = is_post_frontal_wind([315, 180, 0, 90])  # NW, S, N, E
    assert list(out) == [True, False, True, False]


def test_frontal_passage_index_combines_signals():
    # dry-out + rising pressure + NW wind => score 3
    score = frontal_passage_index(
        dewpoint_depression_change=[5.0], slp_tendency_24h=[2.0], wind_deg=[315]
    )
    assert score[0] == 3.0
    # none of the signals => 0
    score0 = frontal_passage_index([0.0], [0.0], [180])
    assert score0[0] == 0.0


def test_build_frontal_features_is_lagged_and_cutoff_safe():
    dates = pd.date_range("2023-10-01", periods=5, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "tmean_f": [70, 68, 55, 52, 60],
            "dewpoint_mean_f": [60, 58, 38, 36, 45],
            "wind_dir_evening_deg": [200, 210, 320, 330, 250],
            "slp_tendency_24h_mb": [-1, -0.5, 3.0, 2.0, -1.0],
        }
    )
    feats = build_frontal_features(daily, lag=1)
    # First row is NaN because of the 1-day lag (cutoff safety: D uses D-1).
    assert feats.iloc[0].isna().all()
    # Day index 3 (Oct 4) sees Oct 3's post-frontal NW wind + pressure rise.
    assert feats["wind_post_frontal"].iloc[3] == 1.0
    assert feats["frontal_passage_index"].iloc[3] >= 2.0
    assert set(feats.columns) == {
        "dewpoint_depression", "dewpoint_depression_trend",
        "wind_post_frontal", "slp_tendency_24h", "frontal_passage_index",
    }


def test_build_frontal_features_requires_core_columns():
    with pytest.raises(ValueError):
        build_frontal_features(pd.DataFrame({"date": ["2023-10-01"], "tmean_f": [70]}))


# ---------------------------------------------------------------------------
# RegimeConditionalCalibrator
# ---------------------------------------------------------------------------
def test_regime_calibrator_fits_per_regime_with_fallback():
    rng = np.random.default_rng(0)
    n = 400
    mu = rng.normal(60, 5, n)
    sigma = np.full(n, 5.0)
    obs = mu + rng.normal(0, 5, n)
    regimes = np.where(np.arange(n) % 2 == 0, "frontal", "calm")
    # Add a sparse regime that should fall back to global.
    regimes[:5] = "rare"
    cal = RegimeConditionalCalibrator(min_samples=30).fit(mu, sigma, obs, regimes)
    assert "frontal" in cal.regimes and "calm" in cal.regimes
    assert "rare" not in cal.regimes  # too few -> global fallback
    cmu, csig = cal.calibrate(mu, sigma, regimes)
    assert cmu.shape == mu.shape and np.all(csig > 0)


def test_regime_calibrator_requires_fit_before_calibrate():
    cal = RegimeConditionalCalibrator()
    with pytest.raises(RuntimeError):
        cal.calibrate([60.0], [5.0], ["frontal"])


def test_regime_calibrator_length_validation():
    cal = RegimeConditionalCalibrator()
    with pytest.raises(ValueError):
        cal.fit([60.0, 61.0], [5.0], [60.0], ["a", "b"])
