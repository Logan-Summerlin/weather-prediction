#!/usr/bin/env python3
"""
Per-city distribution-head comparison (Phase 2 deliverable #6).

Trains three forecast heads on a city's processed (z-scored) features —
Gaussian, 7-quantile, and 2-component Gaussian mixture — and reports the
held-out OOS CRPS and (where real Kalshi contracts overlap the test window)
contract Brier for each, selecting the best by OOS CRPS.

Output: results/<city>/diagnostics/distribution_heads.json

This is a model-selection diagnostic, not a promotion gate: it answers "which
output parameterization fits this city's residual structure best?" before the
hparam/ensemble work commits to one.

Usage:
    python scripts/run_distribution_head_comparison.py --city chi
    python scripts/run_distribution_head_comparison.py --all --epochs 150
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
    DEFAULT_QUANTILE_LEVELS,
    HEAD_FAMILIES,
    HeadScore,
    build_head,
    head_crps,
    head_loss_fn,
    predict_head,
    quantile_bucket_prob_from_edges,
    select_best_head,
    train_head_net,
)
from src.head_data import contract_brier, contract_groups, load_city_arrays  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALL_CITIES = ["chi", "phl", "atl", "aus"]


def _head_brier(family: str, pred: dict, groups) -> float:
    """Contract Brier for a trained head's prediction dict."""
    if family == "gaussian":
        fn = lambda lo, hi, pos: bucket_prob_from_edges(
            pred["mu"][pos], pred["sigma"][pos], lo, hi
        )
    elif family == "quantile":
        fn = lambda lo, hi, pos: quantile_bucket_prob_from_edges(
            pred["levels"], pred["qvals"][pos], lo, hi
        )
    else:  # mixture
        fn = lambda lo, hi, pos: mixture_bucket_prob_from_edges(
            pred["weights"][pos], pred["mus"][pos], pred["sigmas"][pos], lo, hi
        )
    return contract_brier(fn, groups)


def run_city(city_code: str, epochs: int = 150, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_city_arrays(city_code)
    groups = contract_groups(city_code, splits.test_dates, splits.y_test_raw)

    Xt = torch.tensor(splits.X_train)
    Xv = torch.tensor(splits.X_val)
    Xte = torch.tensor(splits.X_test)
    yt = torch.tensor(splits.y_train)
    yv = torch.tensor(splits.y_val)
    y_test = splits.y_test_raw

    results, scores = {}, []
    for family in HEAD_FAMILIES:
        net = build_head(family, splits.n_features)
        net, _ = train_head_net(
            net, head_loss_fn(family), Xt, yt, Xv, yv, epochs=epochs, seed=seed
        )
        pred = predict_head(family, net, Xte, splits.y_mean)
        crps = head_crps(family, pred, y_test)
        brier = _head_brier(family, pred, groups)
        results[family] = {"oos_crps": crps, "contract_brier": brier}
        scores.append(HeadScore(family, crps, brier, len(y_test)))

    best = select_best_head(scores)
    summary = {
        "city_code": city_code,
        "n_features": int(splits.n_features),
        "n_test_days": int(len(y_test)),
        "test_date_min": str(splits.test_dates.min().date()),
        "test_date_max": str(splits.test_dates.max().date()),
        "heads": results,
        "best_head_by_oos_crps": best,
    }

    out_dir = Path(get_city_config(city_code).results_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distribution_heads.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "[%s] CRPS gaussian=%.3f quantile=%.3f mixture=%.3f -> best=%s",
        city_code, results["gaussian"]["oos_crps"], results["quantile"]["oos_crps"],
        results["mixture"]["oos_crps"], best,
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=ALL_CITIES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.city and not args.all:
        p.error("provide --city or --all")
    targets = ALL_CITIES if args.all else [args.city]
    failures = []
    for c in targets:
        try:
            run_city(c, epochs=args.epochs, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] head comparison failed: %s", c, exc)
            failures.append(c)
    return 1 if failures and len(failures) == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
