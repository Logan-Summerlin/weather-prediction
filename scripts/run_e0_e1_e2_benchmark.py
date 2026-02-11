#!/usr/bin/env python3
"""Benchmark E0/E1/E2 starting from the canonical best-model predictions.

E0 = raw best-model probabilities (data/best_model_predictions_*).
E1 = global isotonic CDF calibration on 2023 calibration slice.
E2 = seasonal isotonic CDF calibration on 2023 calibration slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "e0_e1_e2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")


def _cdf(x, mu, sigma):
    return norm.cdf(x, loc=mu, scale=sigma)


def _calibrate_global(mu_cal: np.ndarray, sig_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    q = np.linspace(0.01, 0.99, 99)
    z = (y_cal[:, None] - mu_cal[:, None]) / sig_cal[:, None]
    u = norm.cdf(z)
    preds = np.repeat(q[None, :], len(y_cal), axis=0).reshape(-1)
    obs = (u <= q[None, :]).astype(float).reshape(-1)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(preds, obs)
    return iso


def _month_to_season(months: np.ndarray) -> np.ndarray:
    return np.where(np.isin(months, [12, 1, 2]), "DJF", np.where(np.isin(months, [3, 4, 5]), "MAM", np.where(np.isin(months, [6, 7, 8]), "JJA", "SON")))


def _calibrate_seasonal(cal_df: pd.DataFrame) -> dict[str, IsotonicRegression]:
    season = _month_to_season(cal_df["date_dt"].dt.month.values)
    out: dict[str, IsotonicRegression] = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = season == s
        if np.any(mask):
            out[s] = _calibrate_global(
                cal_df.loc[mask, "model_mu"].values,
                cal_df.loc[mask, "model_sigma"].values,
                cal_df.loc[mask, "actual_tmax"].values,
            )
    out["global"] = _calibrate_global(
        cal_df["model_mu"].values,
        cal_df["model_sigma"].values,
        cal_df["actual_tmax"].values,
    )
    return out


def load_base_dataset(
    model_is_path: str | Path | None = None,
    model_oos_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the benchmark join dataset.

    Parameters
    ----------
    model_is_path : str | Path | None
        Path to model predictions for 2023-2024. If None, uses canonical
        best-model artifact in ``data/best_model_predictions_2023_2024.csv``.
    model_oos_path : str | Path | None
        Path to model predictions for 2025. If None, uses canonical
        best-model artifact in ``data/best_model_predictions_2025.csv``.
    """
    if model_is_path is None:
        model_is_path = ROOT / "data" / "best_model_predictions_2023_2024.csv"
    if model_oos_path is None:
        model_oos_path = ROOT / "data" / "best_model_predictions_2025.csv"

    model_is = pd.read_csv(model_is_path)
    model_oos = pd.read_csv(model_oos_path)
    model = pd.concat([model_is, model_oos], ignore_index=True)

    pre = pd.read_csv(ROOT / "data" / "kalshi_presettlement.csv")
    s23 = pd.read_csv(ROOT / "data" / "real_kalshi_2023_2024.csv")
    s25 = pd.read_csv(ROOT / "data" / "real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)
    nws = pd.read_csv(ROOT / "results" / "prediction_market_benchmark" / "nws_probability_forecasts.csv")

    merged = pre.merge(
        settled[["date", "ticker", "direction", "threshold_low", "threshold_high", "actual_outcome", "actual_tmax", "market_prob"]],
        on=["date", "ticker"],
        suffixes=("_pre", ""),
        how="inner",
    )
    merged = merged.rename(columns={"market_prob": "settled_market_prob"})
    merged = merged.dropna(subset=["presettlement_prob"])

    merged = merged.merge(model[["date", "model_mu", "model_sigma"]], on="date", how="inner")
    merged = merged.merge(nws[["date", "nws_mu", "nws_sigma"]], on="date", how="inner")

    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["period"] = np.where(merged["date_dt"].dt.year <= 2024, "IS", "OOS")
    merged["season"] = merged["date_dt"].dt.month.map(bench.SEASON_MAP)
    return merged


def add_model_probs(df: pd.DataFrame, model_name: str, global_cal: IsotonicRegression, seasonal_cal: dict[str, IsotonicRegression]) -> pd.DataFrame:
    out = df.copy()
    mu = out["model_mu"].values
    sigma = out["model_sigma"].values
    lo = out["threshold_low"].values
    hi = out["threshold_high"].values

    if model_name == "E0_baseline_ensemble":
        out["model_prob"] = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
    else:
        f_lo = np.where(np.isnan(lo), 0.0, _cdf(lo, mu, sigma))
        f_hi = np.where(np.isnan(hi), 1.0, _cdf(hi, mu, sigma))

        if model_name == "E1_global_isotonic":
            lo_cal = np.where(np.isnan(lo), 0.0, np.clip(global_cal.predict(f_lo), 1e-6, 1 - 1e-6))
            hi_cal = np.where(np.isnan(hi), 1.0, np.clip(global_cal.predict(f_hi), 1e-6, 1 - 1e-6))
            out["model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
        elif model_name == "E2_seasonal_calibration":
            out["model_prob"] = np.nan
            season_tag = _month_to_season(out["date_dt"].dt.month.values)
            for s in ["DJF", "MAM", "JJA", "SON"]:
                mask = season_tag == s
                if not np.any(mask):
                    continue
                cal = seasonal_cal.get(s, seasonal_cal["global"])
                lo_cal = np.where(np.isnan(lo[mask]), 0.0, np.clip(cal.predict(f_lo[mask]), 1e-6, 1 - 1e-6))
                hi_cal = np.where(np.isnan(hi[mask]), 1.0, np.clip(cal.predict(f_hi[mask]), 1e-6, 1 - 1e-6))
                out.loc[mask, "model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
        else:
            raise ValueError(model_name)

    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    return out


def run_benchmark_for_model(base_df: pd.DataFrame, model_name: str, global_cal, seasonal_cal) -> dict[str, float | str]:
    df = add_model_probs(base_df, model_name, global_cal, seasonal_cal)
    scores_df = bench.compute_all_scores(df)
    cal_df = bench.compute_calibration(df)
    trading_df = bench.run_all_trading_sims(df)

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "date", "ticker", "bucket", "direction", "threshold_low", "threshold_high",
        "actual_tmax", "actual_outcome", "model_mu", "model_sigma", "model_prob",
        "nws_mu", "nws_sigma", "nws_prob", "presettlement_prob", "settled_market_prob",
        "period", "season",
    ]
    df[[c for c in output_cols if c in df.columns]].to_csv(out_dir / "full_benchmark_comparison.csv", index=False)
    scores_df.to_csv(out_dir / "presettlement_brier_scores.csv", index=False)
    cal_df.to_csv(out_dir / "presettlement_calibration.csv", index=False)
    trading_df.to_csv(out_dir / "trading_simulation_results.csv", index=False)

    overall = scores_df[scores_df["slice"] == "Overall"].set_index("source")
    return {
        "model": model_name,
        "overall_model_brier": float(overall.loc["Model", "brier_score"]),
        "overall_nws_brier": float(overall.loc["NWS", "brier_score"]),
        "overall_presettlement_brier": float(overall.loc["Kalshi_PreSettlement", "brier_score"]),
        "oos_model_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "Model")]["brier_score"].iloc[0]),
        "oos_nws_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "NWS")]["brier_score"].iloc[0]),
        "best_model_all_trading_pnl": float(trading_df[trading_df["signal"] == "Model_All"]["net_pnl"].max()),
        "best_model_oos_trading_pnl": float(trading_df[trading_df["signal"] == "Model_OOS"]["net_pnl"].max()),
    }


def main():
    base_df = load_base_dataset()

    # Calibration slice intentionally mirrors chronological practice: calibrate on 2023.
    calib_df = base_df[base_df["date_dt"].dt.year == 2023][["date_dt", "model_mu", "model_sigma", "actual_tmax"]].drop_duplicates("date_dt")
    global_cal = _calibrate_global(calib_df["model_mu"].values, calib_df["model_sigma"].values, calib_df["actual_tmax"].values)
    seasonal_cal = _calibrate_seasonal(calib_df)

    models = ["E0_baseline_ensemble", "E1_global_isotonic", "E2_seasonal_calibration"]
    rows = [run_benchmark_for_model(base_df, m, global_cal, seasonal_cal) for m in models]
    summary = pd.DataFrame(rows).sort_values("overall_model_brier")
    summary.to_csv(OUT_ROOT / "e0_e1_e2_benchmark_summary.csv", index=False)

    with open(OUT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write("# E0/E1/E2 Benchmarks vs NWS + Kalshi Pre-Settlement\n\n")
        f.write("E0 in this run is explicitly the canonical benchmark model from data/best_model_predictions_*.\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    with open(OUT_ROOT / "benchmark_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": models,
            "source_experiment_notes": "results/probabilistic_ensemble_experiments/experiment_notes.md",
            "model_source": [
                "data/best_model_predictions_2023_2024.csv",
                "data/best_model_predictions_2025.csv",
            ],
            "calibration_period": "2023",
            "n_rows": int(len(base_df)),
            "n_dates": int(base_df["date"].nunique()),
        }, f, indent=2)

    print("Saved:")
    print(f"  - {OUT_ROOT / 'e0_e1_e2_benchmark_summary.csv'}")
    print(f"  - {OUT_ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
