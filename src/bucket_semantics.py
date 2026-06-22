"""
Canonical Kalshi bucket semantics: settlement rounding and CDF conversion.

Empirically verified against every settled contract row in
data/real_kalshi_{chi,phl,atl,aus}_all.csv (100% agreement, n=14,116):

  - Kalshi settles daily-high contracts on the *integer* Fahrenheit value
    reported by the NWS climate report, i.e. ``round(TMAX)``. GHCN/ASOS
    archives store fractional values (Celsius-converted), so e.g. a GHCN
    TMAX of 84.92F settles as 85F.
  - A "between" contract with thresholds (lo, hi) pays YES iff
    ``lo <= round(TMAX) < hi``.
  - A "below"/"less" contract pays YES iff ``round(TMAX) < hi``.
  - An "above" contract pays YES iff ``round(TMAX) > lo``.

For a continuous temperature model, ``round(T) in [lo, hi)`` is the event
``T in [lo - 0.5, hi - 0.5)``, so every CDF evaluation must shift the
threshold by -0.5 ("below"/"between" edges) or +0.5 ("above" edges).
Omitting the shift mis-prices every bucket by half a degree.

All bucket-probability and settlement code must go through this module;
do not reimplement edge arithmetic at call sites.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]

# Sentinel magnitude used by city_config for open-ended buckets.
_OPEN_EDGE = 900.0


def settle_tmax(tmax: ArrayLike) -> ArrayLike:
    """Settlement temperature: integer deg-F as reported by the NWS climate
    report (round-half-up matches observed Kalshi settlements, e.g.
    100.94 -> 101, 75.92 -> 76)."""
    return np.floor(np.asarray(tmax, dtype=float) + 0.5)


def bucket_outcome(
    tmax: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    direction: ArrayLike,
) -> np.ndarray:
    """YES/NO settlement outcome (1/0) for contract rows.

    Parameters
    ----------
    tmax : float or array
        Observed TMAX (fractional deg F is fine; rounded internally).
    lo, hi : float or array
        Contract thresholds. NaN/open edges allowed where unused.
    direction : str or array
        One of "between", "below", "less", "above".
    """
    t = settle_tmax(tmax)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    direction = np.asarray(direction, dtype=object)

    below = (direction == "below") | (direction == "less")
    above = direction == "above"

    outcome = np.where(
        below, t < hi,
        np.where(above, t > lo, (lo <= t) & (t < hi)),
    )
    return outcome.astype(int)


def bucket_prob_gaussian(
    mu: ArrayLike,
    sigma: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    direction: ArrayLike,
) -> np.ndarray:
    """P(contract settles YES) under a Gaussian TMAX model, with the
    half-degree settlement-rounding shift applied.

    between: P(lo <= round(T) < hi) = Phi(hi-0.5) - Phi(lo-0.5)
    below:   P(round(T) < hi)       = Phi(hi-0.5)
    above:   P(round(T) > lo)       = 1 - Phi(lo+0.5)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    direction = np.asarray(direction, dtype=object)

    below = (direction == "below") | (direction == "less")
    above = direction == "above"

    cdf_hi = norm.cdf(hi - 0.5, loc=mu, scale=sigma)
    cdf_lo = norm.cdf(lo - 0.5, loc=mu, scale=sigma)
    sf_lo = 1.0 - norm.cdf(lo + 0.5, loc=mu, scale=sigma)

    probs = np.where(below, cdf_hi, np.where(above, sf_lo, cdf_hi - cdf_lo))
    return np.clip(probs, 0.0, 1.0)


def bucket_prob_from_edges(
    mu: ArrayLike,
    sigma: ArrayLike,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Bucket probability for a city_config (lo, hi) edge pair.

    Open-ended edges use the +/-900 sentinel convention: a bucket whose
    low edge is <= -900 is "everything below hi"; high edge >= 900 is
    "everything above lo" (i.e. round(T) >= lo, hence the -0.5 shift on
    the low edge).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)

    if lo <= -_OPEN_EDGE:
        return np.clip(norm.cdf(hi - 0.5, loc=mu, scale=sigma), 0.0, 1.0)
    if hi >= _OPEN_EDGE:
        return np.clip(1.0 - norm.cdf(lo - 0.5, loc=mu, scale=sigma), 0.0, 1.0)
    probs = (
        norm.cdf(hi - 0.5, loc=mu, scale=sigma)
        - norm.cdf(lo - 0.5, loc=mu, scale=sigma)
    )
    return np.clip(probs, 0.0, 1.0)


def bucket_outcome_from_edges(
    tmax: ArrayLike,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Settlement outcome for a city_config (lo, hi) edge pair
    (round(T) in [lo, hi), with the +/-900 open-edge sentinel)."""
    t = settle_tmax(tmax)
    if lo <= -_OPEN_EDGE:
        return (t < hi).astype(int)
    if hi >= _OPEN_EDGE:
        return (t >= lo).astype(int)
    return ((t >= lo) & (t < hi)).astype(int)


# ---------------------------------------------------------------------------
# Mixture distributions
# ---------------------------------------------------------------------------
# A K-component Gaussian mixture prices a contract by the weighted sum of its
# component bucket probabilities: because the bucket probability is linear in
# the component CDFs, P_mix(bucket) = sum_k w_k * P_k(bucket).  The
# settlement-rounding shift is inherited from the single-Gaussian helpers, so
# mixtures stay contract-faithful with no extra edge arithmetic.

def _normalize_mixture(weights, mus, sigmas):
    """Validate/broadcast mixture parameters to (n_days, K) arrays."""
    mus = np.atleast_2d(np.asarray(mus, dtype=float))
    sigmas = np.atleast_2d(np.asarray(sigmas, dtype=float))
    if mus.shape != sigmas.shape:
        raise ValueError(
            f"mus shape {mus.shape} != sigmas shape {sigmas.shape}"
        )
    n_days, k = mus.shape
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        if w.shape[0] != k:
            raise ValueError(f"weights length {w.shape[0]} != n_components {k}")
        w = np.broadcast_to(w, (n_days, k)).copy()
    elif w.shape != (n_days, k):
        raise ValueError(f"weights shape {w.shape} != {(n_days, k)}")
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum <= 0, 1.0, w_sum)
    return w / w_sum, mus, np.maximum(sigmas, 1e-10)


def mixture_bucket_prob_from_edges(
    weights,
    mus,
    sigmas,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Mixture bucket probability for a city_config (lo, hi) edge pair.

    Parameters
    ----------
    weights : array
        Component weights, shape ``(K,)`` (shared across days) or
        ``(n_days, K)``.  Renormalized to sum to 1 per day.
    mus, sigmas : array
        Component parameters, shape ``(n_days, K)`` (a single day may be
        passed as ``(K,)``).
    """
    w, mus, sigmas = _normalize_mixture(weights, mus, sigmas)
    n_days, k = mus.shape
    prob = np.zeros(n_days)
    for j in range(k):
        prob += w[:, j] * bucket_prob_from_edges(mus[:, j], sigmas[:, j], lo, hi)
    return np.clip(prob, 0.0, 1.0)


def mixture_bucket_prob_gaussian(
    weights,
    mus,
    sigmas,
    lo: ArrayLike,
    hi: ArrayLike,
    direction: ArrayLike,
) -> np.ndarray:
    """Mixture P(contract settles YES) with the settlement-rounding shift.

    ``lo``, ``hi``, ``direction`` are per-day contract terms (shape
    ``(n_days,)``); ``mus``/``sigmas`` are ``(n_days, K)``.
    """
    w, mus, sigmas = _normalize_mixture(weights, mus, sigmas)
    n_days, k = mus.shape
    prob = np.zeros(n_days)
    for j in range(k):
        prob += w[:, j] * bucket_prob_gaussian(
            mus[:, j], sigmas[:, j], lo, hi, direction
        )
    return np.clip(prob, 0.0, 1.0)
