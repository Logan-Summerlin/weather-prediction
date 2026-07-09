#!/usr/bin/env python3
"""Register the NYC U10 market-debias variant in the benchmark artifacts.

Adds the walk-forward U10 probabilities (see ``src.market_debias``) to:

  1. ``results/unified_predictions.csv``  — new ``u10_debias_prob`` column,
     computed with year-boundary walk-forward refits (2023 rows receive the
     raw market mid; every value is honestly out-of-sample).
  2. ``results/unified_benchmark_results.json`` — ``U10_market_debias``
     entry with contract/IS/OOS Brier in the same convention as U0-U9.

Run this BEFORE ``scripts/build_unified_seasonal_brier.py`` (which selects
the promoted variant by OOS contract Brier) and the promotion evaluation.

Usage:
    python scripts/build_u10_market_debias_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_debias import walk_forward_debias  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROB_CLIP = (0.001, 0.999)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(p, *PROB_CLIP) - y) ** 2))


def main() -> None:
    pred_path = PROJECT_ROOT / "results" / "unified_predictions.csv"
    up = pd.read_csv(pred_path)

    wf, _ = walk_forward_debias(up)
    up["u10_debias_prob"] = wf["u10_debias_prob"]

    y = up["actual_outcome"].astype(float).values
    p = up["u10_debias_prob"].values
    is_mask = (up["period"].astype(str).str.upper() == "IS").values
    oos_mask = ~is_mask

    entry = {
        "contract_brier": brier(p, y),
        "is_brier": brier(p[is_mask], y[is_mask]),
        "oos_brier": brier(p[oos_mask], y[oos_mask]),
        "note": (
            "Walk-forward market-debias (longshot discount + book coherence); "
            "every probability strictly out-of-sample (2023 rows = raw market "
            "mid, later years use curves fit on prior years only), unlike the "
            "U2-U9 calibrations whose IS rows are in-sample."
        ),
    }
    logger.info("U10 contract_brier=%.5f is=%.5f oos=%.5f",
                entry["contract_brier"], entry["is_brier"], entry["oos_brier"])

    up.to_csv(pred_path, index=False)
    logger.info("Wrote u10_debias_prob column to %s", pred_path)

    bench_path = PROJECT_ROOT / "results" / "unified_benchmark_results.json"
    with open(bench_path) as f:
        bench = json.load(f)
    bench["U10_market_debias"] = entry
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    logger.info("Registered U10_market_debias in %s", bench_path)


if __name__ == "__main__":
    main()
