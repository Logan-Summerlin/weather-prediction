"""Seam tests for scripts/run_distribution_head_comparison.py helpers."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
mod = importlib.import_module("scripts.run_distribution_head_comparison")


def test_align_columns_reindexes_val_test():
    X_tr = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_va = pd.DataFrame({"b": [5], "a": [6], "c": [7]})  # extra/reordered
    X_te = pd.DataFrame({"a": [8]})  # missing b
    tr, va, te = mod._align_columns(X_tr, X_va, X_te)
    assert list(va.columns) == ["a", "b"]
    assert list(te.columns) == ["a", "b"]
    assert te["b"].iloc[0] == 0.0  # missing column filled with 0


def test_contract_brier_matches_manual():
    # Two contract groups; deterministic probability function.
    groups = [
        (60.0, 64.0, np.array([0, 1]), np.array([1.0, 0.0])),
        (64.0, 68.0, np.array([2]), np.array([1.0])),
    ]

    def prob_fn(lo, hi, pos):
        # constant 0.5 everywhere
        return np.full(len(pos), 0.5)

    brier = mod._contract_brier(prob_fn, groups)
    # three rows, each (0.5 - outcome)^2 = 0.25
    assert brier == pytest.approx(0.25)


def test_contract_brier_empty_is_nan():
    assert np.isnan(mod._contract_brier(lambda lo, hi, pos: np.array([]), None))
