#!/usr/bin/env python3
"""Regenerate per-season contract Brier from the *promoted* unified model.

Why this exists
---------------
The promotion ``overall_brier`` gate scores the best **unified** U-series
variant (``results/<city>/unified_benchmark_results.json``), but the
``seasonal_brier`` gate historically read ``seasonal_brier.json`` as written by
``run_benchmark.py`` — which evaluates the much weaker **base** E-series model.
That mismatch meant the seasonal gate judged a *different, worse* model than the
one actually being promoted, so a city could clear the overall Brier ceiling yet
fail the seasonal gate on a model it never ships.

This script removes the inconsistency: it computes per-meteorological-season
contract Brier from the same promoted unified predictions
(``unified_predictions.csv``) used everywhere else in the trading/backtest path,
evaluated strictly on the out-of-sample (OOS) contract rows, and writes the
result to ``results/<city>/seasonal_brier.json``.

Variant selection mirrors the deployable model: the unified variant with the
lowest OOS contract Brier among the columns present in the predictions file
(this is also the family the real-Kalshi backtest trades). The chosen variant
is recorded under ``_variant`` for auditability.

Usage
-----
    python scripts/build_unified_seasonal_brier.py --city phl
    python scripts/build_unified_seasonal_brier.py --city all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# Candidate unified-variant probability columns, in the order they appear in the
# unified benchmark. Only those present in a given predictions file are scored.
UNIFIED_VARIANT_COLS = [
    "u3_mlp_prob",
    "u4_platt_prob",
    "u5_regime_prob",
    "u6_ensemble_prob",
    "u7_extended_prob",
    "u8_cv_prob",
    "u9_kitchen_prob",
]

PROB_CLIP = (1e-4, 1.0 - 1e-4)


def _contract_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs.astype(float), *PROB_CLIP)
    return float(np.mean((p - outcomes.astype(float)) ** 2))


def build_for_city(city_code: str) -> dict | None:
    cfg = get_city_config(city_code)
    pred_path = os.path.join(cfg.results_dir, "unified_predictions.csv")
    if not os.path.exists(pred_path):
        logger.warning("[%s] no unified_predictions.csv at %s — skipping",
                       city_code, pred_path)
        return None

    df = pd.read_csv(pred_path)
    if "period" not in df.columns:
        logger.warning("[%s] predictions lack a 'period' column — skipping", city_code)
        return None

    oos = df[df["period"].str.upper() == "OOS"].copy()
    oos = oos.dropna(subset=["actual_outcome"])
    if oos.empty:
        logger.warning("[%s] no OOS contract rows — skipping", city_code)
        return None

    oos["month"] = pd.to_datetime(oos["date"]).dt.month
    oos["season"] = oos["month"].map(SEASON_MAP)
    outcomes = oos["actual_outcome"].values

    # Pick the deployable promoted variant: lowest OOS contract Brier among
    # variant columns actually present.
    present = [c for c in UNIFIED_VARIANT_COLS if c in oos.columns]
    if not present:
        logger.warning("[%s] no unified variant columns present — skipping", city_code)
        return None
    variant_briers = {c: _contract_brier(oos[c].values, outcomes) for c in present}
    best_col = min(variant_briers, key=variant_briers.get)

    seasonal: dict[str, float] = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = (oos["season"] == season).values
        if mask.sum() == 0:
            continue
        seasonal[season] = _contract_brier(oos.loc[mask, best_col].values,
                                           outcomes[mask])

    # seasonal_brier.json must contain ONLY season -> Brier float entries: the
    # promotion gate runs float() over every value and takes the max, so any
    # string/count metadata would crash or corrupt it. Provenance goes to a
    # separate sidecar file.
    out_path = os.path.join(cfg.results_dir, "seasonal_brier.json")
    with open(out_path, "w") as f:
        json.dump(seasonal, f, indent=2)

    provenance = {
        "variant": best_col,
        "overall_oos_contract_brier": round(variant_briers[best_col], 6),
        "n_oos_rows": int(len(oos)),
        "all_variant_oos_brier": {c: round(b, 6) for c, b in variant_briers.items()},
        "source": "unified_predictions.csv (promoted U-series, OOS contract rows)",
    }
    with open(os.path.join(cfg.results_dir, "seasonal_brier_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    pretty = ", ".join(f"{s}={seasonal[s]:.4f}" for s in seasonal)
    logger.info("[%s] wrote %s from promoted variant '%s' (OOS Brier %.4f): %s",
                city_code, out_path, best_col, variant_briers[best_col], pretty)
    return seasonal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True,
                    help="City code (e.g. nyc, chi, phl) or 'all'.")
    args = ap.parse_args()

    if args.city == "all":
        cities = ["nyc", "chi", "phl", "atl", "aus"]
    else:
        cities = [args.city]

    for c in cities:
        build_for_city(c)


if __name__ == "__main__":
    main()
