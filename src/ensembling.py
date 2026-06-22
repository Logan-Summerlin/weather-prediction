"""Inverse-validation-CRPS ensembling of Gaussian forecast families.

Phase 2 deliverable #5.  Combines the per-day ``(mu, sigma)`` outputs of
several model families into a single calibrated forecast in one of two ways:

* **mixture** (default) — a K-component Gaussian mixture whose weights are
  proportional to ``1 / val_CRPS`` of each family.  Preserves multi-modality
  and naturally widens when families disagree.  Contract probabilities route
  through :func:`src.bucket_semantics.mixture_bucket_prob_*`.

* **moment** — collapse the mixture to its first two moments (a single
  Gaussian with the mixture mean and mixture standard deviation, the latter
  including the between-component spread).  Convenient when a downstream
  consumer requires a single ``(mu, sigma)``.

Weights are fit on a validation slice only (never the test/OOS slice), matching
the chronological-split discipline used everywhere else in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bucket_semantics import (
    mixture_bucket_prob_from_edges,
    mixture_bucket_prob_gaussian,
)
from src.calibration import compute_crps


def inverse_crps_weights(
    val_crps: dict[str, float], temperature: float = 1.0, eps: float = 1e-9
) -> dict[str, float]:
    """Weights proportional to ``(1 / CRPS) ** temperature``, normalized.

    Lower CRPS earns higher weight.  ``temperature > 1`` sharpens toward the
    best family; ``temperature -> 0`` approaches equal weighting.  NaN/non-
    positive CRPS values are dropped (a family with no valid score earns no
    weight).
    """
    valid = {k: v for k, v in val_crps.items() if np.isfinite(v) and v > 0}
    if not valid:
        # Degenerate: fall back to equal weights over all named families.
        n = len(val_crps) or 1
        return {k: 1.0 / n for k in val_crps}
    raw = {k: (1.0 / (v + eps)) ** temperature for k, v in valid.items()}
    total = sum(raw.values())
    weights = {k: w / total for k, w in raw.items()}
    # Families without a valid score get zero weight, but keep them in the map.
    for k in val_crps:
        weights.setdefault(k, 0.0)
    return weights


@dataclass
class MixtureForecast:
    """A per-day Gaussian mixture: ``weights`` (K,), ``mus``/``sigmas`` (n, K)."""

    names: list[str]
    weights: np.ndarray  # (K,)
    mus: np.ndarray      # (n_days, K)
    sigmas: np.ndarray   # (n_days, K)

    def moment_collapse(self) -> tuple[np.ndarray, np.ndarray]:
        """Collapse to a single Gaussian (mixture mean, mixture std).

        The mixture variance is the law-of-total-variance sum of the
        within-component variance and the between-component mean spread::

            var = sum_k w_k (sigma_k^2 + mu_k^2) - mu_mix^2
        """
        w = self.weights[None, :]
        mu_mix = np.sum(w * self.mus, axis=1)
        second = np.sum(w * (self.sigmas ** 2 + self.mus ** 2), axis=1)
        var = np.maximum(second - mu_mix ** 2, 1e-10)
        return mu_mix, np.sqrt(var)

    def bucket_prob_from_edges(self, lo: float, hi: float) -> np.ndarray:
        return mixture_bucket_prob_from_edges(self.weights, self.mus, self.sigmas, lo, hi)

    def contract_prob_gaussian(self, lo, hi, direction) -> np.ndarray:
        return mixture_bucket_prob_gaussian(
            self.weights, self.mus, self.sigmas, lo, hi, direction
        )


def fit_ensemble_weights(
    val_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    val_actual: np.ndarray,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Fit inverse-CRPS ensemble weights on a validation slice.

    Parameters
    ----------
    val_predictions : dict
        ``{family_name: (mu, sigma)}`` on the validation split.
    val_actual : np.ndarray
        Observed validation TMAX.
    """
    val_crps = {}
    for name, (mu, sigma) in val_predictions.items():
        val_crps[name] = float(compute_crps(mu, sigma, val_actual)["mean_crps"])
    return inverse_crps_weights(val_crps, temperature=temperature)


def build_mixture(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
) -> MixtureForecast:
    """Assemble a :class:`MixtureForecast` from per-family predictions/weights.

    Families with zero weight are dropped.  Weights are renormalized over the
    retained families.
    """
    names = [n for n in sorted(predictions) if weights.get(n, 0.0) > 0]
    if not names:
        raise ValueError("No families with positive weight to ensemble.")
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    mus = np.column_stack([np.asarray(predictions[n][0], dtype=float) for n in names])
    sigmas = np.column_stack([np.asarray(predictions[n][1], dtype=float) for n in names])
    return MixtureForecast(names=names, weights=w, mus=mus, sigmas=sigmas)


def ensemble_forecast(
    val_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    val_actual: np.ndarray,
    test_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    temperature: float = 1.0,
) -> tuple[MixtureForecast, dict[str, float]]:
    """End-to-end: fit weights on validation, build the test-slice mixture.

    Returns the test :class:`MixtureForecast` and the fitted weight map.
    """
    weights = fit_ensemble_weights(val_predictions, val_actual, temperature=temperature)
    mixture = build_mixture(test_predictions, weights)
    return mixture, weights
