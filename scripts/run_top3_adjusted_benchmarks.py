#!/usr/bin/env python3
"""Train top-3 adjusted models and benchmark each vs NWS + Kalshi pre-settlement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util

ROOT = Path(__file__).resolve().parents[1]

def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

exp = _load_module("probexp", "scripts/probabilistic_ensemble_experiments_v2.py")
bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")

OUT_ROOT = Path("results/prediction_market_benchmark/top3_adjusted")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_full_data():
    df, feat = exp.load_dataset()
    split = exp.split_df(df, feat)
    scaler = exp.fit_scaler(split["train"].X)
    for k in split:
        split[k].X[:] = scaler.transform(split[k].X)
    return df, feat, split, scaler


def fit_e8_feature_prune(df: pd.DataFrame):
    feat = [
        "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f",
        "lag1", "lag2", "mos_spread", "sin_doy", "cos_doy",
    ]
    family = {
        "all": feat,
        "no_mos_spread": [c for c in feat if c != "mos_spread"],
        "no_lags": [c for c in feat if c not in {"lag1", "lag2"}],
        "mos_only": ["gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f", "sin_doy", "cos_doy"],
    }

    # select feature set by 2022 validation CRPS, then retrain on <=2022 and infer 2023-2025
    best = None
    for name, cols in family.items():
        local = exp.split_df(df, cols)
        sc = exp.fit_scaler(local["train"].X)
        for k in local:
            local[k].X[:] = sc.transform(local[k].X)
        model = exp.train_gaussian(local, seed=77)
        mu_val, sig_val = exp.predict_gaussian(model, local["val"].X)
        crps = exp.metrics_gaussian(mu_val, sig_val, local["val"].y)["crps"]
        if best is None or crps < best["crps"]:
            best = {"name": name, "cols": cols, "scaler": sc, "seed": 77, "crps": crps}

    # train model on data through 2022 by using same train split definition
    local = exp.split_df(df, best["cols"])
    sc = exp.fit_scaler(local["train"].X)
    for k in local:
        local[k].X[:] = sc.transform(local[k].X)
    model = exp.train_gaussian(local, seed=best["seed"])

    full_x = sc.transform(df[best["cols"]].values.astype(np.float32))
    mu, sigma = exp.predict_gaussian(model, full_x)
    out = pd.DataFrame({"date": df.index, "model_mu": mu, "model_sigma": sigma, "actual_tmax": df["nyc_tmax"].values})
    return out, {"selected_feature_set": best["name"]}


def fit_e3_e4_weighted(df: pd.DataFrame, feat: list[str]):
    split = exp.split_df(df, feat)
    sc = exp.fit_scaler(split["train"].X)
    for k in split:
        split[k].X[:] = sc.transform(split[k].X)

    seeds = [42, 123, 456, 789, 2024]
    members = [exp.train_gaussian(split, seed=s) for s in seeds]

    val_scores = []
    for m in members:
        mu, sig = exp.predict_gaussian(m, split["val"].X)
        val_scores.append(exp.metrics_gaussian(mu, sig, split["val"].y)["crps"])
    val_scores = np.array(val_scores)
    tau = val_scores.std() + 1e-6
    w = np.exp(-(val_scores - val_scores.min()) / tau)
    w = w / w.sum()
    w = 0.5 * w + 0.5 / len(w)

    x_full = sc.transform(df[feat].values.astype(np.float32))
    mu_members = []
    sig_members = []
    for m in members:
        mu, sig = exp.predict_gaussian(m, x_full)
        mu_members.append(mu)
        sig_members.append(sig)
    mu_m = np.vstack(mu_members)
    sig_m = np.vstack(sig_members)

    mu_w = (w[:, None] * mu_m).sum(axis=0)
    sigma_ale = np.sqrt((w[:, None] * (sig_m ** 2)).sum(axis=0))
    sigma_epi = np.sqrt((w[:, None] * ((mu_m - mu_w[None, :]) ** 2)).sum(axis=0))
    sigma_total = np.sqrt(sigma_ale ** 2 + sigma_epi ** 2)

    out = pd.DataFrame({"date": df.index, "model_mu": mu_w, "model_sigma": sigma_total, "actual_tmax": df["nyc_tmax"].values})
    return out, {"ensemble_weights": w.tolist()}


def fit_e7_regularized(df: pd.DataFrame, feat: list[str]):
    split = exp.split_df(df, feat)
    sc = exp.fit_scaler(split["train"].X)
    for k in split:
        split[k].X[:] = sc.transform(split[k].X)

    reg_configs = [(0.05, 1e-4, False), (0.15, 1e-4, False), (0.15, 5e-4, True), (0.25, 1e-3, True)]
    best = None
    best_model = None
    for i, (drop, wd, gdrop) in enumerate(reg_configs):
        m = exp.train_gaussian(split, dropout=drop, weight_decay=wd, grouped_dropout=gdrop, seed=100 + i)
        mu_val, sig_val = exp.predict_gaussian(m, split["val"].X)
        score = exp.metrics_gaussian(mu_val, sig_val, split["val"].y)["crps"]
        if best is None or score < best["crps"]:
            best = {"dropout": drop, "weight_decay": wd, "grouped_dropout": gdrop, "crps": score}
            best_model = m

    full_x = sc.transform(df[feat].values.astype(np.float32))
    mu, sigma = exp.predict_gaussian(best_model, full_x)
    out = pd.DataFrame({"date": df.index, "model_mu": mu, "model_sigma": sigma, "actual_tmax": df["nyc_tmax"].values})
    return out, best


def save_prediction_splits(pred_df: pd.DataFrame, tag: str):
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    is_df = pred_df[(pred_df["date"] >= "2023-01-01") & (pred_df["date"] <= "2024-12-31")].copy()
    oos_df = pred_df[(pred_df["date"] >= "2025-01-01") & (pred_df["date"] <= "2025-12-31")].copy()
    is_path = OUT_ROOT / f"{tag}_predictions_2023_2024.csv"
    oos_path = OUT_ROOT / f"{tag}_predictions_2025.csv"
    is_df.assign(date=is_df["date"].dt.strftime("%Y-%m-%d")).to_csv(is_path, index=False)
    oos_df.assign(date=oos_df["date"].dt.strftime("%Y-%m-%d")).to_csv(oos_path, index=False)
    return is_path, oos_path


def run_benchmark(tag: str, model_is_path: Path, model_oos_path: Path):
    pre, settled, model, nws = bench.load_all_data(str(model_is_path), str(model_oos_path))
    df = bench.build_merged_dataset(pre, settled, model, nws)
    df = bench.add_all_probabilities(df)
    scores_df = bench.compute_all_scores(df)
    cal_df = bench.compute_calibration(df)
    trading_df = bench.run_all_trading_sims(df)

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # save canonical outputs for this model
    output_cols = [
        "date", "ticker", "bucket", "direction", "threshold_low", "threshold_high",
        "actual_tmax", "actual_outcome", "model_mu", "model_sigma", "model_prob",
        "nws_mu", "nws_sigma", "nws_prob", "presettlement_prob", "settled_market_prob",
        "period", "season",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(out_dir / "full_benchmark_comparison.csv", index=False)
    scores_df.to_csv(out_dir / "presettlement_brier_scores.csv", index=False)
    cal_df.to_csv(out_dir / "presettlement_calibration.csv", index=False)
    trading_df.to_csv(out_dir / "trading_simulation_results.csv", index=False)

    overall = scores_df[scores_df["slice"] == "Overall"].set_index("source")
    return {
        "model": tag,
        "overall_model_brier": float(overall.loc["Model", "brier_score"]),
        "overall_nws_brier": float(overall.loc["NWS", "brier_score"]),
        "overall_presettlement_brier": float(overall.loc["Kalshi_PreSettlement", "brier_score"]),
        "oos_model_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "Model")]["brier_score"].iloc[0]),
        "oos_nws_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "NWS")]["brier_score"].iloc[0]),
        "best_model_all_trading_pnl": float(trading_df[trading_df["signal"] == "Model_All"]["net_pnl"].max()),
        "best_model_oos_trading_pnl": float(trading_df[trading_df["signal"] == "Model_OOS"]["net_pnl"].max()),
    }


def main():
    df, feat, _, _ = load_full_data()

    model_builders = [
        ("E8_feature_pruning", lambda: fit_e8_feature_prune(df)),
        ("E3E4_weighted_uncertainty", lambda: fit_e3_e4_weighted(df, feat)),
        ("E7_regularized", lambda: fit_e7_regularized(df, feat)),
    ]

    rows = []
    diagnostics = {}
    for tag, builder in model_builders:
        print(f"\n=== Building {tag} ===")
        pred_df, diag = builder()
        diagnostics[tag] = diag
        is_path, oos_path = save_prediction_splits(pred_df, tag)
        rows.append(run_benchmark(tag, is_path, oos_path))

    summary = pd.DataFrame(rows).sort_values("overall_model_brier")
    summary.to_csv(OUT_ROOT / "top3_benchmark_summary.csv", index=False)

    with open(OUT_ROOT / "model_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    with open(OUT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write("# Top-3 Adjusted Model Benchmarks vs NWS + Kalshi Pre-Settlement\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print("\nSaved:")
    print(f"  - {OUT_ROOT / 'top3_benchmark_summary.csv'}")
    print(f"  - {OUT_ROOT / 'model_diagnostics.json'}")


if __name__ == "__main__":
    main()
