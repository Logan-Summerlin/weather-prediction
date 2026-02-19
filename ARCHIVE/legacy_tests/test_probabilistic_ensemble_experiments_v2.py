import numpy as np
import torch

from scripts.probabilistic_ensemble_experiments_v2 import (
    BUCKET_EDGES,
    QUANTILES,
    QuantileNN,
    bucket_probs_from_gaussian,
)


def test_bucket_probs_sum_to_one():
    mu = np.array([50.0, 75.0], dtype=float)
    sigma = np.array([5.0, 8.0], dtype=float)
    probs = bucket_probs_from_gaussian(mu, sigma)
    assert probs.shape == (2, len(BUCKET_EDGES) - 1)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probs >= 0)


def test_quantile_head_monotonic_outputs():
    model = QuantileNN(d_in=4, n_q=len(QUANTILES))
    x = torch.randn(16, 4)
    with torch.no_grad():
        q = model(x)
    diffs = q[:, 1:] - q[:, :-1]
    assert torch.all(diffs >= 0)
