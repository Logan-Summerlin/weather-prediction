"""Tests for mixture bucket semantics and inverse-CRPS ensembling."""

from __future__ import annotations

import numpy as np
import pytest

from src.bucket_semantics import (
    bucket_prob_from_edges,
    bucket_prob_gaussian,
    mixture_bucket_prob_from_edges,
    mixture_bucket_prob_gaussian,
)
from src.ensembling import (
    build_mixture,
    ensemble_forecast,
    fit_ensemble_weights,
    inverse_crps_weights,
)


# ---------------------------------------------------------------------------
# Mixture bucket semantics
# ---------------------------------------------------------------------------
def test_single_component_mixture_equals_gaussian():
    mu = np.array([70.0, 80.0])
    sigma = np.array([5.0, 6.0])
    single = bucket_prob_from_edges(mu, sigma, 68.0, 72.0)
    mix = mixture_bucket_prob_from_edges(
        np.array([1.0]), mu[:, None], sigma[:, None], 68.0, 72.0
    )
    np.testing.assert_allclose(single, mix, rtol=1e-10)


def test_mixture_is_weighted_average_of_components():
    mu = np.array([[70.0, 80.0]])
    sigma = np.array([[5.0, 5.0]])
    w = np.array([0.3, 0.7])
    p0 = bucket_prob_from_edges(np.array([70.0]), np.array([5.0]), 72.0, 76.0)
    p1 = bucket_prob_from_edges(np.array([80.0]), np.array([5.0]), 72.0, 76.0)
    expected = 0.3 * p0 + 0.7 * p1
    got = mixture_bucket_prob_from_edges(w, mu, sigma, 72.0, 76.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_mixture_weights_renormalized():
    mu = np.array([[70.0, 80.0]])
    sigma = np.array([[5.0, 5.0]])
    a = mixture_bucket_prob_from_edges(np.array([1.0, 1.0]), mu, sigma, 70.0, 74.0)
    b = mixture_bucket_prob_from_edges(np.array([2.0, 2.0]), mu, sigma, 70.0, 74.0)
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_mixture_gaussian_contract_matches_single():
    mu = np.array([75.0])
    sigma = np.array([4.0])
    single = bucket_prob_gaussian(mu, sigma, 73.0, 77.0, np.array(["between"]))
    mix = mixture_bucket_prob_gaussian(
        np.array([1.0]), mu[:, None], sigma[:, None], 73.0, 77.0, np.array(["between"])
    )
    np.testing.assert_allclose(single, mix, rtol=1e-10)


def test_mixture_shape_validation():
    with pytest.raises(ValueError):
        mixture_bucket_prob_from_edges(
            np.array([1.0, 1.0]), np.array([[70.0]]), np.array([[5.0]]), 68.0, 72.0
        )


# ---------------------------------------------------------------------------
# Inverse-CRPS weights
# ---------------------------------------------------------------------------
def test_inverse_crps_weights_favor_lower_crps():
    w = inverse_crps_weights({"good": 1.0, "bad": 4.0})
    assert w["good"] > w["bad"]
    assert w["good"] + w["bad"] == pytest.approx(1.0)


def test_inverse_crps_temperature_sharpens():
    soft = inverse_crps_weights({"a": 1.0, "b": 2.0}, temperature=1.0)
    sharp = inverse_crps_weights({"a": 1.0, "b": 2.0}, temperature=4.0)
    assert sharp["a"] > soft["a"]


def test_inverse_crps_drops_invalid_scores():
    w = inverse_crps_weights({"a": 2.0, "b": float("nan"), "c": -1.0})
    assert w["b"] == 0.0 and w["c"] == 0.0
    assert w["a"] == pytest.approx(1.0)


def test_inverse_crps_all_invalid_equal_weights():
    w = inverse_crps_weights({"a": float("nan"), "b": float("nan")})
    assert w["a"] == pytest.approx(0.5)
    assert w["b"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Ensemble assembly + moment collapse
# ---------------------------------------------------------------------------
def _families():
    rng = np.random.default_rng(0)
    n = 300
    actual = rng.normal(60, 8, size=n)
    good = (actual + rng.normal(0, 3, size=n), np.full(n, 5.0))
    bad = (actual + rng.normal(0, 9, size=n), np.full(n, 10.0))
    return {"good": good, "bad": bad}, actual


def test_fit_ensemble_weights_prefers_better_family():
    fams, actual = _families()
    w = fit_ensemble_weights(fams, actual)
    assert w["good"] > w["bad"]


def test_build_mixture_drops_zero_weight():
    preds = {
        "a": (np.array([70.0, 71.0]), np.array([5.0, 5.0])),
        "b": (np.array([60.0, 61.0]), np.array([5.0, 5.0])),
    }
    mix = build_mixture(preds, {"a": 1.0, "b": 0.0})
    assert mix.names == ["a"]
    assert mix.mus.shape == (2, 1)


def test_moment_collapse_widens_with_disagreement():
    # Two far-apart components, equal weight: collapsed sigma exceeds component sigma.
    preds = {
        "lo": (np.array([50.0]), np.array([3.0])),
        "hi": (np.array([70.0]), np.array([3.0])),
    }
    mix = build_mixture(preds, {"lo": 0.5, "hi": 0.5})
    mu, sigma = mix.moment_collapse()
    assert mu[0] == pytest.approx(60.0)
    assert sigma[0] > 3.0  # between-component spread inflates the variance


def test_ensemble_forecast_end_to_end():
    fams, actual = _families()
    mixture, weights = ensemble_forecast(fams, actual, fams)
    assert set(weights) == {"good", "bad"}
    prob = mixture.bucket_prob_from_edges(58.0, 62.0)
    assert prob.shape == (len(actual),)
    assert np.all((prob >= 0) & (prob <= 1))
