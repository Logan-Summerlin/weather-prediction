#!/usr/bin/env python3
"""
MOS-residual benchmark (Phase 2 deliverable #2 — the primary accuracy lever).

Trains a MOSCorrectionNet that predicts ``TMAX - MOS_TMAX`` on each city's
processed station features augmented with cutoff-safe MOS features
(mos_ensemble_tmax, mos_climo_anomaly, gfs_nam_disagreement), and compares it
to:
  * the raw MOS ensemble baseline (a constant-sigma climatological-error model),
  * the real Kalshi market Brier on overlapping OOS presettlement contracts.

This replaces the weak 6-feature recalibration synthesis stage with an
NWP-informed residual model. Requires a combined MOS archive for the city
(scripts/download_iem_mos_data.py --city <c>); cities without one are skipped
with a clear message.

Output: results/<city>/diagnostics/mos_residual.json

Usage:
    python scripts/run_mos_residual_benchmark.py --city chi
    python scripts/run_mos_residual_benchmark.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bucket_semantics import bucket_prob_from_edges  # noqa: E402
from src.city_config import get_city_config  # noqa: E402
from src.distribution_heads import gaussian_crps  # noqa: E402
from src.head_data import contract_brier, contract_groups, load_city_arrays  # noqa: E402
from src.mos_features import (  # noqa: E402
    build_mos_features,
    doy_climatology,
    find_mos_path,
    load_mos,
    predict_mos_residual,
    train_mos_residual,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALL_CITIES = ["chi", "phl", "atl", "aus"]


def _join_mos(X: np.ndarray, dates: pd.DatetimeIndex, mos_feats: pd.DataFrame):
    """Return (X_aug, baseline, keep_mask) joining station features to MOS rows."""
    idx = pd.DatetimeIndex(dates).normalize()
    aligned = mos_feats.reindex(idx)
    keep = aligned["mos_ensemble_tmax"].notna().to_numpy()
    X_aug = np.concatenate([X[keep], aligned.to_numpy()[keep]], axis=1)
    baseline = aligned["mos_ensemble_tmax"].to_numpy()[keep]
    return X_aug, baseline, keep


def run_city(city_code: str, epochs: int = 200, seed: int = 0) -> Optional[dict]:
    if find_mos_path(city_code) is None:
        logger.warning("[%s] no combined MOS archive; skipping (download first).", city_code)
        return None

    splits = load_city_arrays(city_code)
    mos = load_mos(city_code)

    # Climatology from the (de-centered) training target; MOS anomaly uses it.
    y_tr_raw = splits.y_train + splits.y_mean
    tr_dates = _train_dates(city_code)
    va_dates = _val_dates(city_code)
    climo = doy_climatology(y_tr_raw, tr_dates)
    mos_feats = build_mos_features(mos, climo)

    # Build train/val/test arrays with MOS join.
    Xtr, btr, ktr = _join_mos(splits.X_train, tr_dates, mos_feats)
    Xva, bva, kva = _join_mos(splits.X_val, va_dates, mos_feats)
    Xte, bte, kte = _join_mos(splits.X_test, splits.test_dates, mos_feats)

    ytr = y_tr_raw[ktr]
    yva = (splits.y_val + splits.y_mean)[kva]
    yte = splits.y_test_raw[kte]
    test_dates = splits.test_dates[kte]

    if len(Xtr) < 50 or len(Xte) < 20:
        logger.warning("[%s] too few MOS-overlap rows (train=%d test=%d); skipping.",
                       city_code, len(Xtr), len(Xte))
        return None

    net = train_mos_residual(Xtr, btr, ytr, Xva, bva, yva, epochs=epochs, seed=seed)
    mu, sigma = predict_mos_residual(net, Xte, bte)

    # MOS-residual model scores.
    model_crps = float(np.mean(gaussian_crps(mu, sigma, yte)))
    # Raw MOS baseline scored as a Gaussian with constant sigma = its RMSE.
    base_rmse = float(np.sqrt(np.mean((yte - bte) ** 2)))
    base_crps = float(np.mean(gaussian_crps(bte, np.full_like(bte, base_rmse), yte)))

    groups = contract_groups(city_code, test_dates, yte)
    model_brier = contract_brier(
        lambda lo, hi, pos: bucket_prob_from_edges(mu[pos], sigma[pos], lo, hi), groups
    )
    base_brier = contract_brier(
        lambda lo, hi, pos: bucket_prob_from_edges(
            bte[pos], np.full(len(pos), base_rmse), lo, hi
        ),
        groups,
    )

    summary = {
        "city_code": city_code,
        "n_features": int(Xtr.shape[1]),
        "n_test_days": int(len(yte)),
        "test_date_min": str(test_dates.min().date()),
        "test_date_max": str(test_dates.max().date()),
        "mos_baseline": {"crps": base_crps, "contract_brier": base_brier, "rmse": base_rmse},
        "mos_residual_model": {"crps": model_crps, "contract_brier": model_brier},
        "crps_improvement_vs_baseline": base_crps - model_crps,
    }
    out_dir = Path(get_city_config(city_code).results_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mos_residual.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "[%s] MOS-residual CRPS %.3f vs baseline %.3f (improve %+.3f) | Brier model %.4f base %.4f",
        city_code, model_crps, base_crps, base_crps - model_crps,
        model_brier, base_brier,
    )
    return summary


# Helpers to recover per-split dates (processed CSV indexes carry them).
def _split_dates(city_code: str, split: str) -> pd.DatetimeIndex:
    cfg = get_city_config(city_code)
    path = Path(cfg.data_dir) / "processed" / f"features_{split}.csv"
    return pd.DatetimeIndex(pd.read_csv(path, index_col=0, parse_dates=True).index).normalize()


def _train_dates(city_code: str) -> pd.DatetimeIndex:
    return _split_dates(city_code, "train")


def _val_dates(city_code: str) -> pd.DatetimeIndex:
    return _split_dates(city_code, "val")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=ALL_CITIES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.city and not args.all:
        p.error("provide --city or --all")
    targets = ALL_CITIES if args.all else [args.city]
    ran = 0
    for c in targets:
        try:
            if run_city(c, epochs=args.epochs, seed=args.seed) is not None:
                ran += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] MOS-residual benchmark failed: %s", c, exc)
    if ran == 0:
        logger.warning("No city had a MOS archive to benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
