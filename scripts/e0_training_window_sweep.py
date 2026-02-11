#!/usr/bin/env python3
"""Training-window sweep for E0 baseline benchmark.

Goal
----
Estimate how sensitive E0 benchmark performance is to training-history length
using a consistent NN forecasting pipeline and fixed benchmark years.

Windows tested: 4y, 8y, 12y, 16y.

Notes
-----
- This script uses the same max-train NN backbone used in
  ``scripts/generate_max_training_predictions.py`` as a practical testbed.
- It evaluates E0 only (raw Gaussian bucket mapping) to isolate the effect of
  training-history length before extra calibration variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "prediction_market_benchmark" / "e0_training_window_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


gen = _load_module("genmax", "scripts/generate_max_training_predictions.py")
bench = _load_module("bench", "scripts/test_model_vs_benchmarks.py")
e012 = _load_module("e012", "scripts/run_e0_e1_e2_benchmark.py")


def _year_start(train_end_year: int, window_years: int) -> str:
    return f"{train_end_year - window_years + 1}-01-01"


def _run_single_window(window_years: int, shared: dict) -> dict:
    label = f"{window_years}y"
    run_dir = OUT_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model A: predicts 2023-2024
    a_train_start = _year_start(2020, window_years)
    splits_a, _ = gen.prepare_model_splits(
        shared["X"], shared["y"], shared["dates"],
        train_start=a_train_start,
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2022-12-31",
        pred_start="2023-01-01",
        pred_end="2024-12-31",
        label=f"A_{label}",
    )
    nn_model_a, nn_hist_a, device = gen.train_nn(splits_a, shared["n_features"], label=f"A_{label}")
    sigma_a = gen.estimate_sigma(nn_model_a, splits_a, device, model_type="nn")
    pred_is, metrics_is = gen.generate_predictions(nn_model_a, splits_a, device, sigma_a, "pred", "nn")

    # Model B: predicts 2025
    b_train_start = _year_start(2022, window_years)
    splits_b, _ = gen.prepare_model_splits(
        shared["X"], shared["y"], shared["dates"],
        train_start=b_train_start,
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2024-12-31",
        pred_start="2025-01-01",
        pred_end="2025-12-31",
        label=f"B_{label}",
    )
    nn_model_b, nn_hist_b, _ = gen.train_nn(splits_b, shared["n_features"], label=f"B_{label}")
    sigma_b = gen.estimate_sigma(nn_model_b, splits_b, device, model_type="nn")
    pred_oos, metrics_oos = gen.generate_predictions(nn_model_b, splits_b, device, sigma_b, "pred", "nn")

    is_path = run_dir / "model_predictions_2023_2024.csv"
    oos_path = run_dir / "model_predictions_2025.csv"
    pred_is.to_csv(is_path, index=False)
    pred_oos.to_csv(oos_path, index=False)

    # E0 benchmark only
    base_df = e012.load_base_dataset(model_is_path=is_path, model_oos_path=oos_path)
    e0_df = e012.add_model_probs(base_df, "E0_baseline_ensemble", global_cal=None, seasonal_cal=None)
    scores_df = bench.compute_all_scores(e0_df)
    trading_df = bench.run_all_trading_sims(e0_df)

    scores_df.to_csv(run_dir / "presettlement_brier_scores.csv", index=False)
    trading_df.to_csv(run_dir / "trading_simulation_results.csv", index=False)

    overall = scores_df[scores_df["slice"] == "Overall"].set_index("source")
    oos_model_brier = float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "Model")]["brier_score"].iloc[0])

    model_all = trading_df[trading_df["signal"] == "Model_All"]
    model_oos = trading_df[trading_df["signal"] == "Model_OOS"]

    return {
        "window_years": window_years,
        "model_a_train_start": a_train_start,
        "model_b_train_start": b_train_start,
        "model_a_train_samples": int(len(splits_a["train"]["y"])),
        "model_b_train_samples": int(len(splits_b["train"]["y"])),
        "is_mae": float(metrics_is["mae"]),
        "oos_mae": float(metrics_oos["mae"]),
        "overall_model_brier": float(overall.loc["Model", "brier_score"]),
        "overall_nws_brier": float(overall.loc["NWS", "brier_score"]),
        "overall_presettlement_brier": float(overall.loc["Kalshi_PreSettlement", "brier_score"]),
        "oos_model_brier": oos_model_brier,
        "best_model_all_trading_pnl": float(model_all["net_pnl"].max()),
        "best_model_oos_trading_pnl": float(model_oos["net_pnl"].max()),
        "best_model_all_roi_pct": float(model_all.loc[model_all["net_pnl"].idxmax(), "roi_pct"]),
        "best_model_oos_roi_pct": float(model_oos.loc[model_oos["net_pnl"].idxmax(), "roi_pct"]),
        "model_a_best_epoch": int(nn_hist_a["best_epoch"]),
        "model_b_best_epoch": int(nn_hist_b["best_epoch"]),
    }


def main() -> None:
    # Speed-oriented settings for sweep (keeps method same, shortens runtime).
    gen.MAX_EPOCHS = 180
    gen.PATIENCE = 16

    station_data = gen.load_all_stations()
    qualifying_ids, _ = gen.check_completeness(station_data)
    station_data = {sid: df for sid, df in station_data.items() if sid in qualifying_ids}
    X, y, dates, feature_cols = gen.build_features(station_data, qualifying_ids)

    shared = {
        "X": X,
        "y": y,
        "dates": dates,
        "n_features": len(feature_cols),
    }

    rows = []
    for window in [4, 8, 12, 16]:
        rows.append(_run_single_window(window, shared))

    summary = pd.DataFrame(rows).sort_values("window_years").reset_index(drop=True)
    summary_path = OUT_DIR / "e0_training_window_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Simple markdown report
    md_path = OUT_DIR / "README.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# E0 Baseline Training-Window Sweep (4y/8y/12y/16y)\n\n")
        f.write("This sweep retrains the E0 testbed forecast model with varying history lengths and re-runs the E0 benchmark.\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "windows_tested": [4, 8, 12, 16],
                "benchmark_variant": "E0_baseline_ensemble",
                "prediction_periods": {
                    "is": "2023-2024",
                    "oos": "2025",
                },
            },
            f,
            indent=2,
        )

    print(f"Saved: {summary_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
