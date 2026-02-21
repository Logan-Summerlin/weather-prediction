"""High-signal invariants for time safety, bucketization, and EV economics."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.live_trading import gaussian_to_bucket_probs
from src.city_config import get_city_config, list_cities
from src.data_preprocessing import chronological_split
from src.trading import compute_ev_yes


@pytest.mark.parametrize("city_code", list_cities())
def test_bucketization_probabilities_sum_to_one_and_cover_all_buckets(city_code):
    """Bucketization must stay numerically valid for every registered city."""
    cfg = get_city_config(city_code)
    probs = gaussian_to_bucket_probs(mu=70.0, sigma=6.0, bucket_edges=cfg.bucket_edges)

    assert len(probs) == len(cfg.bucket_edges)
    assert probs.sum() == pytest.approx(1.0, abs=0.01)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_chronological_split_has_no_temporal_leakage():
    """Train/val/test boundaries must be strictly chronological."""
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    features = pd.DataFrame({"feature": range(12)}, index=index)
    target = pd.Series(range(12), index=index)

    X_train, X_val, X_test, _, _, _ = chronological_split(
        features, target, train_ratio=0.5, val_ratio=0.25
    )

    train_end = X_train.index.max()
    val_start = X_val.index.min()
    val_end = X_val.index.max()
    test_start = X_test.index.min()

    assert train_end < val_start
    assert val_end < test_start


def test_expected_value_decreases_when_fees_increase():
    """Trading edges must be fee-aware and conservative."""
    edge_low_fee = compute_ev_yes(model_prob=0.60, market_price=0.54, fee_rate=0.02)
    edge_high_fee = compute_ev_yes(model_prob=0.60, market_price=0.54, fee_rate=0.12)

    assert edge_low_fee > edge_high_fee
    assert edge_low_fee > 0.0
    assert edge_high_fee < edge_low_fee

