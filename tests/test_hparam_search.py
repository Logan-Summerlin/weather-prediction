"""Tests for the rolling-origin hyperparameter search helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("torch")
hps = importlib.import_module("scripts.run_hparam_search")


def test_rolling_origin_folds_are_forward_and_expanding():
    folds = hps.rolling_origin_folds(100, n_folds=3)
    assert len(folds) == 3
    prev_train_end = 0
    for tr, va in folds:
        # train is a contiguous prefix starting at 0
        assert tr[0] == 0
        # validation strictly follows training (no overlap, forward only)
        assert va[0] == tr[-1] + 1
        # expanding window: each fold's train extends the previous
        assert tr[-1] >= prev_train_end
        prev_train_end = tr[-1]
        # no index leakage between train and val
        assert len(set(tr.tolist()) & set(va.tolist())) == 0


def test_rolling_origin_no_future_leakage():
    folds = hps.rolling_origin_folds(60, n_folds=3)
    for tr, va in folds:
        assert tr.max() < va.min()  # every train index precedes every val index


def test_rolling_origin_rejects_tiny_series():
    with pytest.raises(ValueError):
        hps.rolling_origin_folds(5, n_folds=3)


def test_sample_config_within_space():
    rng = np.random.default_rng(0)
    for _ in range(20):
        cfg = hps.sample_config(rng)
        assert cfg["hidden"] in hps.SEARCH_SPACE["hidden"]
        assert cfg["depth"] in hps.SEARCH_SPACE["depth"]
        assert cfg["lr"] in hps.SEARCH_SPACE["lr"]
        assert cfg["weight_decay"] in hps.SEARCH_SPACE["weight_decay"]
        assert cfg["batch_size"] in hps.SEARCH_SPACE["batch_size"]
