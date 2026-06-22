#!/usr/bin/env python3
"""
Rolling-origin chronological hyperparameter search (Phase 2 deliverable #4).

For a city and distribution-head family, randomly samples ``budget``
configurations and scores each by mean validation CRPS over a 3-fold
*rolling-origin* chronological cross-validation (no shuffling, no leakage:
each fold trains on a contiguous past prefix and validates on the next block).
The best config is then retrained on the full development set (train+val) and
evaluated exactly once on the held-out OOS test split (CRPS + real-contract
Brier).

Output: results/<city>/diagnostics/hparam_search_<family>.json

Usage:
    python scripts/run_hparam_search.py --city chi --family quantile --budget 50
    python scripts/run_hparam_search.py --city aus --family gaussian --budget 30 --cv-epochs 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bucket_semantics import (  # noqa: E402
    bucket_prob_from_edges,
    mixture_bucket_prob_from_edges,
)
from src.city_config import get_city_config  # noqa: E402
from src.distribution_heads import (  # noqa: E402
    HEAD_FAMILIES,
    build_head,
    head_crps,
    head_loss_fn,
    predict_head,
    quantile_bucket_prob_from_edges,
    train_head_net,
)
from src.head_data import contract_brier, contract_groups, load_city_arrays  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALL_CITIES = ["chi", "phl", "atl", "aus"]

#: Random search space (sampled uniformly per trial).
SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "depth": [1, 2, 3],
    "lr": [3e-3, 1e-2, 3e-2],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [128, 256],
}


def sample_config(rng: np.random.Generator) -> dict:
    return {k: (float(rng.choice(v)) if k in ("lr", "weight_decay") else int(rng.choice(v)))
            for k, v in SEARCH_SPACE.items()}


def rolling_origin_folds(n: int, n_folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin chronological folds over ``n`` ordered samples.

    Splits the series into ``n_folds + 1`` contiguous blocks; fold ``k`` trains
    on blocks ``0..k`` and validates on block ``k+1``.  Always expanding-window
    and strictly forward — no future data ever enters a fold's training set.
    """
    if n < (n_folds + 1) * 2:
        raise ValueError(f"Not enough samples ({n}) for {n_folds} rolling folds")
    bounds = np.linspace(0, n, n_folds + 2, dtype=int)
    folds = []
    for k in range(n_folds):
        train_idx = np.arange(0, bounds[k + 1])
        val_idx = np.arange(bounds[k + 1], bounds[k + 2])
        folds.append((train_idx, val_idx))
    return folds


def cv_score(family, X_dev, y_dev_raw, config, n_folds, epochs, seed) -> float:
    """Mean validation CRPS over rolling-origin folds (lower is better)."""
    folds = rolling_origin_folds(len(X_dev), n_folds)
    crps_vals = []
    for tr_idx, va_idx in folds:
        # Validation early-stopping target must be the real fold-val loss, so we
        # pass the fold-val features/targets into train via a closure-free path:
        # train uses its own val arg; here we hold out by scoring CRPS directly.
        pred = _fit_predict_val(
            family, X_dev[tr_idx], y_dev_raw[tr_idx],
            X_dev[va_idx], y_dev_raw[va_idx], config, epochs, seed,
        )
        crps_vals.append(head_crps(family, pred, y_dev_raw[va_idx]))
    return float(np.mean(crps_vals))


def _fit_predict_val(family, X_tr, y_tr_raw, X_va, y_va_raw, config, epochs, seed):
    """Train with the fold's own validation set for early stopping; predict val."""
    y_mean = float(np.mean(y_tr_raw))
    net = build_head(family, X_tr.shape[1], hidden=config["hidden"], depth=config["depth"])
    net, _ = train_head_net(
        net, head_loss_fn(family),
        torch.tensor(X_tr), torch.tensor((y_tr_raw - y_mean).astype(np.float32)),
        torch.tensor(X_va), torch.tensor((y_va_raw - y_mean).astype(np.float32)),
        epochs=epochs, lr=config["lr"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], seed=seed,
    )
    return predict_head(family, net, torch.tensor(X_va), y_mean)


def _final_brier(family, pred, groups) -> float:
    if family == "gaussian":
        fn = lambda lo, hi, pos: bucket_prob_from_edges(pred["mu"][pos], pred["sigma"][pos], lo, hi)
    elif family == "quantile":
        fn = lambda lo, hi, pos: quantile_bucket_prob_from_edges(pred["levels"], pred["qvals"][pos], lo, hi)
    else:
        fn = lambda lo, hi, pos: mixture_bucket_prob_from_edges(
            pred["weights"][pos], pred["mus"][pos], pred["sigmas"][pos], lo, hi
        )
    return contract_brier(fn, groups)


def run_search(city_code: str, family: str, budget: int = 50, n_folds: int = 3,
               cv_epochs: int = 80, final_epochs: int = 150, seed: int = 0) -> dict:
    if family not in HEAD_FAMILIES:
        raise ValueError(f"family must be one of {HEAD_FAMILIES}")
    rng = np.random.default_rng(seed)
    splits = load_city_arrays(city_code)

    # Development set = train + val, chronologically ordered (already so on disk).
    X_dev = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y_dev = np.concatenate([
        splits.y_train + splits.y_mean, splits.y_val + splits.y_mean
    ]).astype(np.float32)  # de-centered raw target

    trials = []
    seen = set()
    for t in range(budget):
        config = sample_config(rng)
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)
        score = cv_score(family, X_dev, y_dev, config, n_folds, cv_epochs, seed)
        trials.append({"config": config, "cv_crps": score})
        logger.info("[%s/%s] trial %d/%d cv_crps=%.4f %s",
                    city_code, family, t + 1, budget, score, config)

    trials.sort(key=lambda r: r["cv_crps"])
    best = trials[0]

    # Retrain best on full dev (early-stop on the val tail), single OOS eval.
    final_net = _retrain_full(
        family, X_dev, y_dev, splits.X_val, splits.y_val + splits.y_mean,
        best["config"], final_epochs, seed,
    )
    pred_test = predict_head(
        family, final_net, torch.tensor(splits.X_test), float(np.mean(y_dev))
    )
    groups = contract_groups(city_code, splits.test_dates, splits.y_test_raw)
    oos_crps = head_crps(family, pred_test, splits.y_test_raw)
    oos_brier = _final_brier(family, pred_test, groups)

    summary = {
        "city_code": city_code,
        "family": family,
        "budget": budget,
        "n_folds": n_folds,
        "n_dev_days": int(len(X_dev)),
        "n_test_days": int(len(splits.y_test_raw)),
        "best_config": best["config"],
        "best_cv_crps": best["cv_crps"],
        "oos_crps": oos_crps,
        "oos_contract_brier": oos_brier,
        "trials": trials,
    }
    out_dir = Path(get_city_config(city_code).results_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"hparam_search_{family}.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("[%s/%s] best cv_crps=%.4f -> OOS crps=%.4f brier=%.4f config=%s",
                city_code, family, best["cv_crps"], oos_crps, oos_brier, best["config"])
    return summary


def _retrain_full(family, X_dev, y_dev_raw, X_val, y_val_raw, config, epochs, seed):
    """Retrain the chosen config on full dev, early-stopping on the val tail."""
    y_mean = float(np.mean(y_dev_raw))
    net = build_head(family, X_dev.shape[1], hidden=config["hidden"], depth=config["depth"])
    net, _ = train_head_net(
        net, head_loss_fn(family),
        torch.tensor(X_dev), torch.tensor((y_dev_raw - y_mean).astype(np.float32)),
        torch.tensor(X_val), torch.tensor((y_val_raw - y_mean).astype(np.float32)),
        epochs=epochs, lr=config["lr"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], seed=seed,
    )
    return net


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", required=True, choices=ALL_CITIES)
    p.add_argument("--family", required=True, choices=list(HEAD_FAMILIES))
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--cv-epochs", type=int, default=80)
    p.add_argument("--final-epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_search(args.city, args.family, budget=args.budget, n_folds=args.folds,
               cv_epochs=args.cv_epochs, final_epochs=args.final_epochs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
