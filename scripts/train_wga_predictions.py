#!/usr/bin/env python3
"""Train a Wind-Gated-Attention forecaster for one city and export predictions.

Produces the val/test prediction files the NYC unified_outperformance stack
expects at:
    results/wga_v2_model/wga_v2_multihead_only/predictions_{val,test}.csv
with columns: date, model_mu, model_sigma_cal, regime.

This reconstructs the missing WGA-V2 prediction step using the existing
src.wga_data_pipeline.train_wga_city trainer on the city's processed data.
Regime is a simple per-row volatility label (low/medium/high) from the
predicted sigma terciles, matching how the stack consumes `regime`.

Usage:
    python scripts/train_wga_predictions.py --city nyc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config  # noqa: E402
from src.wga_data_pipeline import train_wga_city  # noqa: E402


def _load_split(processed_dir: Path, split: str):
    X = pd.read_csv(processed_dir / f"features_{split}.csv", index_col=0, parse_dates=True)
    y = pd.read_csv(processed_dir / f"target_{split}.csv", index_col=0, parse_dates=True)
    y = y.iloc[:, 0]
    return X, y


def _regime_labels(sigma: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(sigma, [1 / 3, 2 / 3])
    out = np.where(sigma <= lo, "low_var",
                   np.where(sigma >= hi, "high_var", "medium_var"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()

    cfg = get_city_config(args.city)
    processed_dir = Path(cfg.data_dir) / "processed"

    X_tr, y_tr = _load_split(processed_dir, "train")
    X_va, y_va = _load_split(processed_dir, "val")
    X_te, y_te = _load_split(processed_dir, "test")
    print(f"[{args.city}] train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    res = train_wga_city(
        args.city, X_tr, y_tr, X_va, y_va, X_te, y_te,
        output_mode="gaussian", max_epochs=args.max_epochs, patience=20,
    )

    out_dir = PROJECT_ROOT / "results" / "wga_v2_model" / "wga_v2_multihead_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, X, mu, sig in [
        ("val", X_va, res["mu_val"], res["sigma_val"]),
        ("test", X_te, res["mu_test"], res["sigma_test"]),
    ]:
        df = pd.DataFrame({
            "date": pd.to_datetime(X.index).strftime("%Y-%m-%d"),
            "model_mu": mu,
            "model_sigma_cal": sig,
            "regime": _regime_labels(np.asarray(sig)),
        })
        path = out_dir / f"predictions_{split}.csv"
        df.to_csv(path, index=False)
        print(f"[{args.city}] wrote {path} ({len(df)} rows)")

    # Report test MAE against truth for sanity.
    test_mae = float(np.mean(np.abs(res["mu_test"] - y_te.values)))
    print(f"[{args.city}] WGA test MAE = {test_mae:.2f} F "
          f"(best epoch {res['best_epoch']}, val loss {res['best_val_loss']:.4f})")


if __name__ == "__main__":
    main()
