"""Utilities for Kalshi contract-row Brier evaluation.

These helpers enforce a canonical evaluation unit for cross-city comparisons:
binary contract rows from real Kalshi contracts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4

CITY_TICKER_PATTERNS = {
    "chi": ("HIGHCHI", "KXHIGHCHI"),
    "phl": ("HIGHPHIL", "KXHIGHPHIL"),
}


def load_city_kalshi_contract_rows(
    city_code: str,
    valid_dates: pd.DatetimeIndex,
    settled_paths: list[Path],
) -> pd.DataFrame:
    """Load settled Kalshi contract rows for a city and date subset."""
    if city_code not in CITY_TICKER_PATTERNS:
        raise ValueError(f"Unsupported city_code={city_code!r}")

    frames = []
    for path in settled_paths:
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            "No Kalshi settled datasets found. Expected one of: "
            + ", ".join(str(p) for p in settled_paths)
        )

    all_rows = pd.concat(frames, ignore_index=True)
    patterns = CITY_TICKER_PATTERNS[city_code]
    ticker_mask = all_rows["ticker"].astype(str).str.contains("|".join(patterns), na=False)
    city_rows = all_rows[ticker_mask].copy()

    if city_rows.empty:
        raise RuntimeError(
            f"No settled Kalshi contracts found for city={city_code} using patterns={patterns}."
        )

    city_rows["date"] = pd.to_datetime(city_rows["date"]).dt.normalize()
    date_mask = city_rows["date"].isin(pd.to_datetime(valid_dates).normalize())
    city_rows = city_rows[date_mask].copy()
    if city_rows.empty:
        raise RuntimeError(
            f"No settled Kalshi contracts overlap requested evaluation dates for city={city_code}."
        )

    return city_rows


def contract_probabilities_from_gaussian(
    contract_rows: pd.DataFrame,
    mu_by_date: pd.Series,
    sigma_by_date: pd.Series,
) -> np.ndarray:
    """Map Gaussian daily forecasts to binary Kalshi contract probabilities."""
    mu = contract_rows["date"].map(mu_by_date).values
    sigma = contract_rows["date"].map(sigma_by_date).values

    missing = np.isnan(mu) | np.isnan(sigma)
    if np.any(missing):
        raise ValueError(
            "Forecast parameters missing for some contract dates "
            f"({int(np.sum(missing))} rows)."
        )

    probs = np.zeros(len(contract_rows), dtype=float)
    below = contract_rows["direction"] == "below"
    above = contract_rows["direction"] == "above"
    between = contract_rows["direction"] == "between"

    probs[below] = norm.cdf(
        contract_rows.loc[below, "threshold_high"].values,
        loc=mu[below],
        scale=sigma[below],
    )
    probs[above] = 1.0 - norm.cdf(
        contract_rows.loc[above, "threshold_low"].values,
        loc=mu[above],
        scale=sigma[above],
    )
    probs[between] = norm.cdf(
        contract_rows.loc[between, "threshold_high"].values,
        loc=mu[between],
        scale=sigma[between],
    ) - norm.cdf(
        contract_rows.loc[between, "threshold_low"].values,
        loc=mu[between],
        scale=sigma[between],
    )
    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)


def contract_brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute mean binary Brier score over contract rows."""
    return float(np.mean((probs.astype(float) - outcomes.astype(float)) ** 2))

