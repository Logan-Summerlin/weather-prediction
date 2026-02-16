import numpy as np
import pandas as pd

from src.contract_brier import (
    contract_brier_score,
    contract_probabilities_from_gaussian,
)


def test_contract_probabilities_from_gaussian_handles_directions():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "direction": ["below", "between", "above"],
            "threshold_low": [np.nan, 70.0, 72.0],
            "threshold_high": [70.0, 72.0, np.nan],
        }
    )
    mu = pd.Series([71.0], index=pd.to_datetime(["2024-01-01"]))
    sigma = pd.Series([2.0], index=pd.to_datetime(["2024-01-01"]))

    probs = contract_probabilities_from_gaussian(rows, mu, sigma)

    assert probs.shape == (3,)
    assert np.all((probs > 0.0) & (probs < 1.0))
    assert probs[1] > 0.0


def test_contract_brier_score_matches_manual_mean_squared_error():
    probs = np.array([0.2, 0.9, 0.4, 0.6])
    outcomes = np.array([0, 1, 0, 1])

    got = contract_brier_score(probs, outcomes)
    expected = float(np.mean((probs - outcomes) ** 2))

    assert abs(got - expected) < 1e-12
