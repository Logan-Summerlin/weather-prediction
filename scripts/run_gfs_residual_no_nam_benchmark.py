#!/usr/bin/env python3
"""Benchmark a GFS-residual MOS model (without NAM MOS / MOS ensemble) vs NWS + Kalshi pre-settlement.

Modeling idea:
- Base forecast = GFS MOS tmax
- Learn residual r = y - gfs_mos_tmax_f from time-safe features
- Predict Gaussian residual distribution (mu_r, sigma_r)
- Convert back to temperature distribution:
    mu = gfs_mos_tmax_f + mu_r
    sigma = sigma_r
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "prediction_market_benchmark" / "gfs_residual_no_nam_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


exp = _load_module("probexp", "scripts/probabilistic_ensemble_experiments_v2.py")
bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    cp = pd.read_csv(ROOT / "data" / "central_park_tmax_full_history.csv", parse_dates=["date"])
    cp = cp.rename(columns={"tmax_f": "nyc_tmax"})[["date", "nyc_tmax"]].set_index("date").sort_index()

    mos = pd.read_csv(ROOT / "data" / "mos" / "combined_mos_knyc.csv", parse_dates=["date"])
    mos = mos[["date", "gfs_mos_tmax_f"]].set_index("date").sort_index()

    df = cp.join(mos, how="inner")
    df["residual_gfs"] = df["nyc_tmax"] - df["gfs_mos_tmax_f"]

    # Time-safe predictors only (shifted residual memory)
    df["lag1"] = df["nyc_tmax"].shift(1)
    df["lag2"] = df["nyc_tmax"].shift(2)
    df["gfs_resid_lag1"] = df["residual_gfs"].shift(1)
    df["gfs_resid_3d"] = df["residual_gfs"].shift(1).rolling(3, min_periods=2).mean()
    df["gfs_resid_7d"] = df["residual_gfs"].shift(1).rolling(7, min_periods=3).mean()
    df["gfs_resid_14d"] = df["residual_gfs"].shift(1).rolling(14, min_periods=5).mean()
    df["gfs_abs_resid_7d"] = df["residual_gfs"].abs().shift(1).rolling(7, min_periods=3).mean()

    doy = df.index.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    aux_cols = [
        "knyc_mos_wind_speed_mph",
        "knyc_mos_wind_dir_deg",
        "knyc_mos_cloud_cover_code",
        "knyc_mos_dewpoint_f",
        "knyc_mos_rel_humidity_pct",
        "other_station_avg_wind_speed_mph",
        "other_station_avg_precip_prob",
        "other_station_avg_snow_indicator",
    ]
    for col in aux_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[aux_cols] = df[aux_cols].sort_index().ffill(limit=3)

    features = [
        "gfs_mos_tmax_f",
        "lag1",
        "lag2",
        "gfs_resid_lag1",
        "gfs_resid_3d",
        "gfs_resid_7d",
        "gfs_resid_14d",
        "gfs_abs_resid_7d",
        "knyc_mos_wind_speed_mph",
        "knyc_mos_wind_dir_deg",
        "knyc_mos_cloud_cover_code",
        "knyc_mos_dewpoint_f",
        "knyc_mos_rel_humidity_pct",
        "other_station_avg_wind_speed_mph",
        "other_station_avg_precip_prob",
        "other_station_avg_snow_indicator",
        "sin_doy",
        "cos_doy",
    ]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    med = df.loc["2003-01-01":"2018-12-31", features].median(numeric_only=True)
    df[features] = df[features].fillna(med).fillna(0.0)
    df = df.dropna(subset=["residual_gfs", "nyc_tmax"]).copy()
    return df, features


def split_df(df: pd.DataFrame, features: list[str]):
    windows = {
        "train": ("2003-01-01", "2018-12-31"),
        "val": ("2019-01-01", "2022-12-31"),
        "calib": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-12-31"),
        "oos": ("2025-01-01", "2025-12-31"),
    }
    out = {}
    for name, (s, e) in windows.items():
        sub = df.loc[s:e]
        out[name] = exp.SplitData(
            X=sub[features].values.astype(np.float32),
            y=sub["residual_gfs"].values.astype(np.float32),
            dates=sub.index.values,
        )
    return out


def save_prediction_splits(pred_df: pd.DataFrame):
    is_df = pred_df[(pred_df["date"] >= "2023-01-01") & (pred_df["date"] <= "2024-12-31")].copy()
    oos_df = pred_df[(pred_df["date"] >= "2025-01-01") & (pred_df["date"] <= "2025-12-31")].copy()

    is_path = OUT_DIR / "gfs_residual_predictions_2023_2024.csv"
    oos_path = OUT_DIR / "gfs_residual_predictions_2025.csv"
    is_df.assign(date=is_df["date"].dt.strftime("%Y-%m-%d")).to_csv(is_path, index=False)
    oos_df.assign(date=oos_df["date"].dt.strftime("%Y-%m-%d")).to_csv(oos_path, index=False)
    return is_path, oos_path


def main() -> None:
    df, features = load_dataset()
    split = split_df(df, features)

    scaler = exp.fit_scaler(split["train"].X)
    for k in split:
        split[k].X[:] = scaler.transform(split[k].X)

    model = exp.train_gaussian(split, seed=314, dropout=0.1, weight_decay=1e-4)

    # Evaluate residual model on validation/test/oos windows
    metrics = {}
    for part in ["val", "calib", "test", "oos"]:
        mu_r, sig_r = exp.predict_gaussian(model, split[part].X)
        metrics[part] = exp.metrics_gaussian(mu_r, sig_r, split[part].y)

    # Full-history inference
    X_full = scaler.transform(df[features].values.astype(np.float32))
    mu_r_full, sig_r_full = exp.predict_gaussian(model, X_full)

    pred_df = pd.DataFrame(
        {
            "date": df.index,
            "model_mu": df["gfs_mos_tmax_f"].values + mu_r_full,
            "model_sigma": sig_r_full,
            "actual_tmax": df["nyc_tmax"].values,
            "gfs_mos_tmax_f": df["gfs_mos_tmax_f"].values,
            "predicted_residual_mu": mu_r_full,
            "predicted_residual_sigma": sig_r_full,
            "actual_residual": df["residual_gfs"].values,
        }
    )

    is_path, oos_path = save_prediction_splits(pred_df)

    pre, settled, model_df, nws = bench.load_all_data(str(is_path), str(oos_path))
    merged = bench.build_merged_dataset(pre, settled, model_df, nws)
    merged = bench.add_all_probabilities(merged)

    scores_df = bench.compute_all_scores(merged)
    cal_df = bench.compute_calibration(merged)
    trading_df = bench.run_all_trading_sims(merged)

    merged.to_csv(OUT_DIR / "full_benchmark_comparison.csv", index=False)
    scores_df.to_csv(OUT_DIR / "presettlement_brier_scores.csv", index=False)
    cal_df.to_csv(OUT_DIR / "presettlement_calibration.csv", index=False)
    trading_df.to_csv(OUT_DIR / "trading_simulation_results.csv", index=False)

    overall = scores_df[scores_df["slice"] == "Overall"].set_index("source")
    oos = scores_df[scores_df["slice"] == "Period: OOS"].set_index("source")

    summary = {
        "model": "gfs_residual_no_nam_ensemble",
        "features": features,
        "residual_target": "nyc_tmax - gfs_mos_tmax_f",
        "validation_metrics_residual": metrics,
        "benchmark": {
            "overall_model_brier": float(overall.loc["Model", "brier_score"]),
            "overall_nws_brier": float(overall.loc["NWS", "brier_score"]),
            "overall_presettlement_brier": float(overall.loc["Kalshi_PreSettlement", "brier_score"]),
            "oos_model_brier": float(oos.loc["Model", "brier_score"]),
            "oos_nws_brier": float(oos.loc["NWS", "brier_score"]),
            "oos_presettlement_brier": float(oos.loc["Kalshi_PreSettlement", "brier_score"]),
        },
    }

    with open(OUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# GFS Residual Model Benchmark (NAM/Ensemble MOS Removed)",
        "",
        "## Setup",
        "- Base forecast: `gfs_mos_tmax_f` only.",
        "- Removed inputs: `nam_mos_tmax_f`, `mos_ensemble_tmax_f`.",
        "- Residual target: `residual_gfs = nyc_tmax - gfs_mos_tmax_f`.",
        "- Features: " + ", ".join(features),
        "",
        "## Residual model metrics",
        pd.DataFrame(metrics).T.to_string(),
        "",
        "## Benchmark (Brier, lower is better)",
        scores_df[scores_df["slice"].isin(["Overall", "Period: OOS"])].to_string(index=False),
    ]
    (OUT_DIR / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved benchmark to", OUT_DIR)


if __name__ == "__main__":
    main()
