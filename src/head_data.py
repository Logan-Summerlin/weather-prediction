"""Shared data loading for the distribution-head / hparam-search drivers.

Loads a city's processed (z-scored) feature splits, centers the target so the
Gaussian mu head starts near the mean (the Phase 0 variance-collapse fix), and
prepares the real-Kalshi contract groups used for contract-Brier scoring.

Kept torch-optional: :func:`load_city_arrays` returns numpy; the torch-tensor
convenience (:func:`to_tensor`) is only imported lazily by callers that have
torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.bucket_semantics import bucket_outcome_from_edges
from src.city_config import get_city_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Real Kalshi presettlement CSVs by city code (canonical evaluation source).
PRESETTLEMENT = {
    "chi": "kalshi_presettlement_chi.csv",
    "phl": "kalshi_presettlement_phl.csv",
    "atl": "kalshi_presettlement_atl.csv",
    "aus": "kalshi_presettlement_aus.csv",
    "nyc": "kalshi_presettlement.csv",
}


def _load_split(processed: Path, split: str):
    X = pd.read_csv(processed / f"features_{split}.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(processed / f"target_{split}.csv", index_col=0, parse_dates=True)
    X = X.dropna(axis=1, how="all").fillna(0.0)
    return X, y.iloc[:, 0]


def align_columns(X_train, X_val, X_test):
    """Reindex val/test to the train columns (fill missing with 0.0)."""
    cols = X_train.columns
    return (
        X_train,
        X_val.reindex(columns=cols, fill_value=0.0),
        X_test.reindex(columns=cols, fill_value=0.0),
    )


@dataclass
class CitySplits:
    """Numpy feature/target splits plus metadata for one city."""

    city_code: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray            # centered (mean subtracted)
    y_val: np.ndarray              # centered
    y_test_raw: np.ndarray         # NOT centered (for scoring/outcomes)
    y_mean: float
    feature_names: list[str]
    test_dates: pd.DatetimeIndex
    n_features: int = field(init=False)

    def __post_init__(self):
        self.n_features = self.X_train.shape[1]


def load_city_arrays(city_code: str) -> CitySplits:
    """Load and align a city's processed splits; center the target on train mean."""
    cfg = get_city_config(city_code)
    processed = Path(cfg.data_dir) / "processed"
    if not processed.is_dir():
        raise FileNotFoundError(f"No processed dir for {city_code}: {processed}")

    X_tr, y_tr = _load_split(processed, "train")
    X_va, y_va = _load_split(processed, "val")
    X_te, y_te = _load_split(processed, "test")
    X_tr, X_va, X_te = align_columns(X_tr, X_va, X_te)

    y_mean = float(y_tr.mean())
    return CitySplits(
        city_code=city_code,
        X_train=X_tr.to_numpy(dtype=np.float32),
        X_val=X_va.to_numpy(dtype=np.float32),
        X_test=X_te.to_numpy(dtype=np.float32),
        y_train=(y_tr.to_numpy(dtype=np.float32) - y_mean),
        y_val=(y_va.to_numpy(dtype=np.float32) - y_mean),
        y_test_raw=y_te.to_numpy(dtype=float),
        y_mean=y_mean,
        feature_names=list(X_tr.columns),
        test_dates=pd.DatetimeIndex(X_te.index),
    )


def contract_groups(city_code: str, test_dates: pd.DatetimeIndex, y_test: np.ndarray):
    """Group test-window Kalshi rows by (lo, hi) for vectorized Brier scoring.

    Returns a list of ``(lo, hi, positions, outcomes)`` (positions index into
    the test-day arrays; outcomes are settled YES/NO via bucket semantics), or
    ``None`` when no presettlement file overlaps the test window.
    """
    name = PRESETTLEMENT.get(city_code, "")
    pre_path = PROJECT_ROOT / "data" / name
    if not name or not pre_path.exists():
        return None
    contracts = pd.read_csv(pre_path)
    contracts["date"] = pd.to_datetime(contracts["date"]).dt.normalize()
    date_to_pos = {d.normalize(): i for i, d in enumerate(test_dates)}
    rows = contracts[contracts["date"].isin(set(date_to_pos))]
    if rows.empty:
        return None
    groups = []
    for (lo, hi), grp in rows.groupby(["threshold_low", "threshold_high"]):
        pos = grp["date"].map(date_to_pos).to_numpy()
        outcomes = bucket_outcome_from_edges(y_test[pos], float(lo), float(hi))
        groups.append((float(lo), float(hi), pos, outcomes.astype(float)))
    return groups


def contract_brier(prob_fn, groups) -> float:
    """Mean contract Brier over all rows; ``prob_fn(lo, hi, pos) -> probs``."""
    if not groups:
        return float("nan")
    probs, outs = [], []
    for lo, hi, pos, outcome in groups:
        probs.append(prob_fn(lo, hi, pos))
        outs.append(outcome)
    return float(np.mean((np.concatenate(probs) - np.concatenate(outs)) ** 2))
