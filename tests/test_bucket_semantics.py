"""
Tests for src/bucket_semantics.py — canonical Kalshi settlement semantics.

The integration tests validate the settlement model against every settled
contract row in data/real_kalshi_<city>_all.csv: the rounded-integer,
lo <= round(T) < hi convention must reproduce Kalshi's actual YES/NO
results exactly.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bucket_semantics import (
    bucket_outcome,
    bucket_outcome_from_edges,
    bucket_prob_from_edges,
    bucket_prob_gaussian,
    settle_tmax,
)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class TestSettleTmax:
    def test_rounds_half_up(self):
        # Observed Kalshi behavior: 100.94 -> 101, 75.92 -> 76, 84.92 -> 85
        assert settle_tmax(100.94) == 101
        assert settle_tmax(75.92) == 76
        assert settle_tmax(84.92) == 85
        assert settle_tmax(84.5) == 85
        assert settle_tmax(84.49) == 84

    def test_array_input(self):
        out = settle_tmax(np.array([75.92, 84.92]))
        assert list(out) == [76.0, 85.0]


class TestBucketOutcome:
    def test_between_inclusive_low_exclusive_high(self):
        # round(84.92)=85 belongs to [85, 87), not [83, 85)
        assert bucket_outcome(84.92, 85, 87, "between") == 1
        assert bucket_outcome(84.92, 83, 85, "between") == 0

    def test_below_and_above(self):
        assert bucket_outcome(75.92, np.nan, 76, "below") == 0  # round=76, not < 76
        assert bucket_outcome(75.4, np.nan, 76, "below") == 1   # round=75 < 76
        assert bucket_outcome(90.6, 90, np.nan, "above") == 1   # round=91 > 90
        assert bucket_outcome(89.6, 90, np.nan, "above") == 0   # round=90, not > 90


class TestBucketProbGaussian:
    def test_probabilities_shifted_by_half_degree(self):
        # With mu exactly at lo-0.5, half the mass is below the bucket.
        from scipy.stats import norm
        p = bucket_prob_gaussian(mu=84.5, sigma=2.0, lo=85, hi=87, direction="between")
        expected = norm.cdf(86.5, 84.5, 2.0) - norm.cdf(84.5, 84.5, 2.0)
        assert p == pytest.approx(expected)
        assert p == pytest.approx(norm.cdf(1.0) - 0.5)

    def test_directions_partition_probability(self):
        # below hi=85, between [85,87), above 86 (i.e. round(T)>86 == >=87)
        below = bucket_prob_gaussian(85.0, 3.0, np.nan, 85, "below")
        between = bucket_prob_gaussian(85.0, 3.0, 85, 87, "between")
        above = bucket_prob_gaussian(85.0, 3.0, 86, np.nan, "above")
        assert below + between + above == pytest.approx(1.0, abs=1e-9)

    def test_vectorized(self):
        p = bucket_prob_gaussian(
            mu=np.array([80.0, 90.0]),
            sigma=np.array([3.0, 3.0]),
            lo=np.array([85.0, 85.0]),
            hi=np.array([87.0, 87.0]),
            direction=np.array(["between", "between"]),
        )
        assert p.shape == (2,)
        assert 0 < p[0] < p[1] < 1


class TestEdgePairConventions:
    def test_open_low_edge(self):
        p_open = bucket_prob_from_edges(70.0, 3.0, -999.0, 60.0)
        from scipy.stats import norm
        assert p_open == pytest.approx(norm.cdf(59.5, 70, 3))

    def test_open_high_edge_uses_geq(self):
        # high edge open: round(T) >= lo, so continuous T >= lo - 0.5
        from scipy.stats import norm
        p = bucket_prob_from_edges(70.0, 3.0, 80.0, 999.0)
        assert p == pytest.approx(1.0 - norm.cdf(79.5, 70, 3))

    def test_edge_buckets_partition(self):
        edges = [(-999.0, 60.0)] + [(float(a), float(a + 2)) for a in range(60, 80, 2)] + [(80.0, 999.0)]
        total = sum(bucket_prob_from_edges(68.0, 4.0, lo, hi) for lo, hi in edges)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_outcome_from_edges(self):
        assert bucket_outcome_from_edges(59.7, -999.0, 60.0) == 0  # round=60
        assert bucket_outcome_from_edges(59.4, -999.0, 60.0) == 1
        assert bucket_outcome_from_edges(79.6, 80.0, 999.0) == 1   # round=80 >= 80
        assert bucket_outcome_from_edges(60.0, 60.0, 62.0) == 1


@pytest.mark.parametrize("city", ["chi", "phl", "atl", "aus"])
def test_reproduces_real_kalshi_settlements(city):
    """The settlement model must match Kalshi's actual results exactly."""
    path = os.path.join(PROJECT_ROOT, "data", f"real_kalshi_{city}_all.csv")
    if not os.path.exists(path):
        pytest.skip(f"no settled data for {city}")
    df = pd.read_csv(path).dropna(subset=["actual_tmax", "result"])
    predicted = bucket_outcome(
        df["actual_tmax"].values,
        df["threshold_low"].values,
        df["threshold_high"].values,
        df["direction"].values,
    )
    actual = df["result"].str.lower().eq("yes").astype(int).values
    agreement = (predicted == actual).mean()
    assert agreement == 1.0, (
        f"{city}: settlement model disagrees with Kalshi on "
        f"{(predicted != actual).sum()} of {len(df)} rows"
    )
