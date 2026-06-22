"""Tests for src.distribution_heads (Gaussian / quantile / mixture)."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import compute_crps
from src.distribution_heads import (
    DEFAULT_QUANTILE_LEVELS,
    HeadScore,
    gaussian_crps,
    gaussian_mixture_crps,
    pinball_loss,
    quantile_bucket_prob_from_edges,
    quantile_cdf_at,
    quantile_crps,
    select_best_head,
)


def test_gaussian_crps_matches_calibration_module():
    rng = np.random.default_rng(0)
    mu = rng.normal(60, 5, 200)
    sigma = np.full(200, 6.0)
    y = mu + rng.normal(0, 6, 200)
    mine = float(np.mean(gaussian_crps(mu, sigma, y)))
    ref = compute_crps(mu, sigma, y)["mean_crps"]
    assert mine == pytest.approx(ref, rel=1e-9)


def test_mixture_crps_single_component_equals_gaussian():
    rng = np.random.default_rng(1)
    mu = rng.normal(60, 5, 100)
    sigma = np.full(100, 5.0)
    y = mu + rng.normal(0, 5, 100)
    g = gaussian_crps(mu, sigma, y)
    m = gaussian_mixture_crps(np.array([1.0]), mu[:, None], sigma[:, None], y)
    np.testing.assert_allclose(g, m, rtol=1e-9)


def test_mixture_crps_two_identical_components_equals_single():
    mu = np.array([60.0, 70.0])
    sigma = np.array([5.0, 5.0])
    y = np.array([62.0, 68.0])
    single = gaussian_crps(mu, sigma, y)
    mus = np.column_stack([mu, mu])
    sig = np.column_stack([sigma, sigma])
    mix = gaussian_mixture_crps(np.array([0.5, 0.5]), mus, sig, y)
    np.testing.assert_allclose(single, mix, rtol=1e-9)


def test_mixture_crps_nonnegative():
    rng = np.random.default_rng(2)
    n = 50
    mus = rng.normal(60, 8, (n, 2))
    sig = np.full((n, 2), 5.0)
    y = rng.normal(60, 8, n)
    assert np.all(gaussian_mixture_crps(np.array([0.4, 0.6]), mus, sig, y) >= 0)


def test_pinball_and_quantile_crps_reward_better_forecasts():
    levels = np.array(DEFAULT_QUANTILE_LEVELS)
    y = np.full(100, 60.0)
    # Tight quantiles centered on y vs wide ones.
    good = 60.0 + (levels - 0.5)[None, :] * 4.0 * np.ones((100, len(levels)))
    bad = 60.0 + (levels - 0.5)[None, :] * 40.0 * np.ones((100, len(levels)))
    assert pinball_loss(levels, good, y).mean() < pinball_loss(levels, bad, y).mean()
    assert quantile_crps(levels, good, y).mean() < quantile_crps(levels, bad, y).mean()


def test_quantile_cdf_monotone_and_bounded():
    levels = np.array(DEFAULT_QUANTILE_LEVELS)
    qvals = np.array([[50, 52, 56, 60, 64, 68, 70]], dtype=float)
    lo = quantile_cdf_at(levels, qvals, 48.0)
    mid = quantile_cdf_at(levels, qvals, 60.0)
    hi = quantile_cdf_at(levels, qvals, 72.0)
    assert lo[0] == 0.0 and hi[0] == 1.0
    assert lo[0] <= mid[0] <= hi[0]
    assert mid[0] == pytest.approx(0.5, abs=1e-9)


def test_quantile_bucket_prob_in_unit_interval_and_shift():
    levels = np.array(DEFAULT_QUANTILE_LEVELS)
    qvals = np.array([[50, 52, 56, 60, 64, 68, 70]], dtype=float)
    p = quantile_bucket_prob_from_edges(levels, qvals, 58.0, 62.0)
    assert 0.0 <= p[0] <= 1.0
    # Open-ended high edge => P(round(T) >= lo).
    p_open = quantile_bucket_prob_from_edges(levels, qvals, 60.0, 900.0)
    assert 0.0 <= p_open[0] <= 1.0


def test_select_best_head_by_crps():
    scores = [
        HeadScore("gaussian", 2.0, 0.1, 100),
        HeadScore("quantile", 1.5, 0.12, 100),
        HeadScore("mixture", 1.8, 0.09, 100),
    ]
    assert select_best_head(scores) == "quantile"


# ---------------------------------------------------------------------------
# Torch heads
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")


def test_quantile_net_monotone_output():
    from src.distribution_heads import build_quantile_net
    net = build_quantile_net(n_features=5)
    x = torch.randn(8, 5)
    out = net(x)
    assert out.shape == (8, len(DEFAULT_QUANTILE_LEVELS))
    diffs = out[:, 1:] - out[:, :-1]
    assert torch.all(diffs >= -1e-5)  # monotone non-decreasing quantiles


def test_mixture_net_shapes_and_weights_sum_to_one():
    from src.distribution_heads import build_mixture_net
    net = build_mixture_net(n_features=5, n_components=2)
    x = torch.randn(8, 5)
    w, mu, sigma = net(x)
    assert w.shape == (8, 2) and mu.shape == (8, 2) and sigma.shape == (8, 2)
    assert torch.allclose(w.sum(dim=1), torch.ones(8), atol=1e-5)
    assert torch.all(sigma > 0)


def test_quantile_net_trains_down():
    from src.distribution_heads import build_quantile_net, pinball_loss_torch
    torch.manual_seed(0)
    levels = DEFAULT_QUANTILE_LEVELS
    x = torch.randn(256, 4)
    y = (x[:, 0] * 3.0 + 60.0)
    net = build_quantile_net(n_features=4, levels=levels)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = pinball_loss_torch(levels, net(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
