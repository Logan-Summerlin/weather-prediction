#!/usr/bin/env python3
"""Benchmark E0-E8 best-model-derived variants vs NWS + Kalshi pre-settlement.

All variants are defined as transformations/calibrations of canonical best-model
prediction artifacts (data/best_model_predictions_*).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "e0_e8_best_model_base"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


exp = _load_module("probexp", "scripts/probabilistic_ensemble_experiments_v2.py")
bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")
e012 = _load_module("e012", "scripts/run_e0_e1_e2_benchmark.py")


def _season_from_month(months: np.ndarray) -> np.ndarray:
    return np.where(np.isin(months, [12, 1, 2]), "DJF", np.where(np.isin(months, [3, 4, 5]), "MAM", np.where(np.isin(months, [6, 7, 8]), "JJA", "SON")))


def _fit_experiment_transforms(base_df: pd.DataFrame):
    # Date-level calibration frame from canonical best-model predictions.
    by_date = base_df[["date_dt", "model_mu", "model_sigma", "actual_tmax"]].drop_duplicates("date_dt").copy()
    by_date["season"] = _season_from_month(by_date["date_dt"].dt.month.values)

    calib = by_date[by_date["date_dt"].dt.year == 2023].copy()
    mu_cal = calib["model_mu"].values
    sig_cal = calib["model_sigma"].values
    y_cal = calib["actual_tmax"].values

    global_cal = exp.calibrate_global(mu_cal, sig_cal, y_cal)
    seasonal_cal = exp.calibrate_seasonal(calib["date_dt"].values, mu_cal, sig_cal, y_cal)

    sigma_mult_global = exp._fit_sigma_multiplier(mu_cal, sig_cal, y_cal)
    sigma_mult_season = exp._fit_seasonal_sigma_multiplier(
        calib.rename(columns={"date_dt": "date"})[["date", "season", "model_mu", "model_sigma", "actual_tmax"]]
    )

    resid = y_cal - mu_cal
    offset_global = float(np.mean(resid))
    offset_by_season = calib.groupby("season").apply(lambda x: float(np.mean(x["actual_tmax"] - x["model_mu"]))).to_dict()
    resid_scale = float(np.std(resid))

    return {
        "global_cal": global_cal,
        "seasonal_cal": seasonal_cal,
        "sigma_mult_global": sigma_mult_global,
        "sigma_mult_season": sigma_mult_season,
        "offset_global": offset_global,
        "offset_by_season": offset_by_season,
        "resid_scale": resid_scale,
        "conditional_cal": _fit_conditional_calibration(calib),
    }


def _fit_conditional_calibration(calib: pd.DataFrame) -> dict[str, object]:
    """Fit isotonic CDF calibrators on season x spread-bin x regime-bin cells."""
    cal = calib.sort_values("date_dt").copy()
    cal["season"] = _season_from_month(cal["date_dt"].dt.month.values)
    cal["spread_bin"] = pd.qcut(cal["model_sigma"], q=3, labels=["lo", "mid", "hi"], duplicates="drop").astype(str)
    cal["mu_change"] = cal["model_mu"].diff().abs().fillna(0.0)
    cal["regime_bin"] = pd.qcut(cal["mu_change"], q=3, labels=["stable", "transition", "volatile"], duplicates="drop").astype(str)

    calibrators: dict[str, object] = {}
    min_points = 35
    for (season, spread_bin, regime_bin), grp in cal.groupby(["season", "spread_bin", "regime_bin"]):
        if len(grp) < min_points:
            continue
        key = f"{season}|{spread_bin}|{regime_bin}"
        calibrators[key] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)

    # Hierarchical fallbacks.
    fallbacks = {
        "global": exp.calibrate_global(cal["model_mu"].values, cal["model_sigma"].values, cal["actual_tmax"].values)
    }
    for season, grp in cal.groupby("season"):
        if len(grp) >= min_points:
            fallbacks[f"season|{season}"] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)
    for spread_bin, grp in cal.groupby("spread_bin"):
        if len(grp) >= min_points:
            fallbacks[f"spread|{spread_bin}"] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)

    sigma_edges = np.quantile(cal["model_sigma"].values, [0.0, 1 / 3, 2 / 3, 1.0])
    sigma_edges = np.unique(sigma_edges)
    mu_change_edges = np.quantile(cal["mu_change"].values, [0.0, 1 / 3, 2 / 3, 1.0])
    mu_change_edges = np.unique(mu_change_edges)

    return {
        "calibrators": calibrators,
        "fallbacks": fallbacks,
        "sigma_edges": sigma_edges,
        "mu_change_edges": mu_change_edges,
    }


def _bin_labels(values: np.ndarray, edges: np.ndarray, labels: list[str]) -> np.ndarray:
    if len(edges) < 2:
        return np.array([labels[0]] * len(values))
    idx = np.digitize(values, edges[1:-1], right=True)
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array([labels[i] for i in idx])


def _apply_variant(df: pd.DataFrame, variant: str, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    season = _season_from_month(out["date_dt"].dt.month.values)

    mu = out["model_mu"].values.copy()
    sigma = out["model_sigma"].values.copy()

    if variant == "E0_baseline_ensemble":
        pass
    elif variant == "E1_global_isotonic":
        pass
    elif variant == "E2_seasonal_calibration":
        pass
    elif variant == "E3_weighted_ensemble_E4_uncertainty":
        sigma = np.clip(sigma * cfg["sigma_mult_global"], 0.5, 15.0)
    elif variant == "E4_uncertainty_decomposition":
        sigma = np.clip(np.sqrt((sigma * cfg["sigma_mult_global"]) ** 2 + (0.15 * cfg["resid_scale"]) ** 2), 0.5, 15.0)
    elif variant == "E5_mdn2":
        mu = mu + cfg["offset_global"]
    elif variant == "E6_quantile":
        offsets = np.array([cfg["offset_by_season"].get(s, cfg["offset_global"]) for s in season])
        mu = mu + offsets
    elif variant == "E7_regularization_sweep":
        mult = np.array([cfg["sigma_mult_season"].get(s, cfg["sigma_mult_season"]["global"]) for s in season])
        sigma = np.clip(sigma * mult, 0.5, 15.0)
    elif variant == "E8_feature_pruning_sweep":
        offsets = np.array([cfg["offset_by_season"].get(s, cfg["offset_global"]) for s in season])
        mult = np.array([cfg["sigma_mult_season"].get(s, cfg["sigma_mult_season"]["global"]) for s in season])
        mu = mu + offsets
        sigma = np.clip(sigma * mult, 0.5, 15.0)
    elif variant == "E9_conditional_calibration_grid":
        pass
    else:
        raise ValueError(variant)

    out["model_mu"] = mu
    out["model_sigma"] = sigma

    # model bucket probabilities
    if variant == "E1_global_isotonic":
        f_lo = np.where(np.isnan(out["threshold_low"].values), 0.0, e012._cdf(out["threshold_low"].values, mu, sigma))
        f_hi = np.where(np.isnan(out["threshold_high"].values), 1.0, e012._cdf(out["threshold_high"].values, mu, sigma))
        lo_cal = np.where(np.isnan(out["threshold_low"].values), 0.0, np.clip(cfg["global_cal"].predict(f_lo), 1e-6, 1 - 1e-6))
        hi_cal = np.where(np.isnan(out["threshold_high"].values), 1.0, np.clip(cfg["global_cal"].predict(f_hi), 1e-6, 1 - 1e-6))
        out["model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
    elif variant in {"E2_seasonal_calibration", "E8_feature_pruning_sweep"}:
        f_lo = np.where(np.isnan(out["threshold_low"].values), 0.0, e012._cdf(out["threshold_low"].values, mu, sigma))
        f_hi = np.where(np.isnan(out["threshold_high"].values), 1.0, e012._cdf(out["threshold_high"].values, mu, sigma))
        out["model_prob"] = np.nan
        for s in ["DJF", "MAM", "JJA", "SON"]:
            m = season == s
            if not np.any(m):
                continue
            cal = cfg["seasonal_cal"].get(s, cfg["seasonal_cal"]["global"])
            lo_cal = np.where(np.isnan(out.loc[m, "threshold_low"].values), 0.0, np.clip(cal.predict(f_lo[m]), 1e-6, 1 - 1e-6))
            hi_cal = np.where(np.isnan(out.loc[m, "threshold_high"].values), 1.0, np.clip(cal.predict(f_hi[m]), 1e-6, 1 - 1e-6))
            out.loc[m, "model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
    elif variant == "E9_conditional_calibration_grid":
        f_lo = np.where(np.isnan(out["threshold_low"].values), 0.0, e012._cdf(out["threshold_low"].values, mu, sigma))
        f_hi = np.where(np.isnan(out["threshold_high"].values), 1.0, e012._cdf(out["threshold_high"].values, mu, sigma))
        by_date = out[["date", "date_dt", "model_mu", "model_sigma"]].drop_duplicates("date").sort_values("date_dt").copy()
        by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
        sigma_bin = _bin_labels(
            by_date["model_sigma"].values,
            cfg["conditional_cal"]["sigma_edges"],
            ["lo", "mid", "hi"],
        )
        regime_bin = _bin_labels(
            by_date["mu_change"].values,
            cfg["conditional_cal"]["mu_change_edges"],
            ["stable", "transition", "volatile"],
        )
        by_date["season"] = _season_from_month(by_date["date_dt"].dt.month.values)
        by_date["cond_key"] = by_date["season"] + "|" + sigma_bin + "|" + regime_bin
        out = out.merge(by_date[["date", "cond_key", "season"]], on="date", how="left", suffixes=("", "_cond"))
        out["model_prob"] = np.nan

        calibrators = cfg["conditional_cal"]["calibrators"]
        fallbacks = cfg["conditional_cal"]["fallbacks"]

        for key in out["cond_key"].dropna().unique():
            m = out["cond_key"] == key
            season_key = key.split("|")[0]
            cal = calibrators.get(key, fallbacks.get(f"season|{season_key}", fallbacks["global"]))
            lo_cal = np.where(np.isnan(out.loc[m, "threshold_low"].values), 0.0, np.clip(cal.predict(f_lo[m]), 1e-6, 1 - 1e-6))
            hi_cal = np.where(np.isnan(out.loc[m, "threshold_high"].values), 1.0, np.clip(cal.predict(f_hi[m]), 1e-6, 1 - 1e-6))
            out.loc[m, "model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
        out.drop(columns=[c for c in ["cond_key", "season_cond"] if c in out.columns], inplace=True)
    else:
        out["model_prob"] = bench.compute_bucket_probs(out, "model_mu", "model_sigma")

    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)

    return out


def _run_edge_quality_gating(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """EV-aware dynamic threshold gating to reduce low-quality/high-cost trades."""
    rows: list[dict[str, float | str | int]] = []
    edge = df["model_prob"].values - df["presettlement_prob"].values
    spread = ((df["ask_cents"].fillna(df["presettlement_prob"] * 100) - df["bid_cents"].fillna(df["presettlement_prob"] * 100)).clip(lower=0) / 100.0).values
    sigma_low = np.percentile(df["model_sigma"].values, 5)
    sigma_high = np.percentile(df["model_sigma"].values, 95)
    sigma_norm = np.clip((df["model_sigma"].values - sigma_low) / (sigma_high - sigma_low + 1e-6), 0, 1)
    liquidity = np.clip(1.0 - spread / 0.20, 0.0, 1.0)
    quality = np.abs(edge) * liquidity * (1.0 - 0.5 * sigma_norm)

    ask_all = df["ask_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0
    bid_all = df["bid_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0
    outcome_all = df["actual_outcome"].values.astype(float)

    rng = np.random.default_rng(20260212)
    n_boot = 1000

    for period in ["All", "IS", "OOS"]:
        m = np.ones(len(df), dtype=bool) if period == "All" else df["period"].values == period
        sub = df.loc[m].copy()
        sub_edge = edge[m]
        sub_quality = quality[m]
        sub_spread = spread[m]
        sub_sigma_norm = sigma_norm[m]
        sub_ask = ask_all[m]
        sub_bid = bid_all[m]
        sub_outcome = outcome_all[m]

        for q_cut in [0.02, 0.03, 0.04, 0.05]:
            dyn_threshold = 0.01 + 0.5 * sub_spread + 0.04 * sub_sigma_norm
            buy_yes = (sub_edge > dyn_threshold) & (sub_quality > q_cut)
            buy_no = (sub_edge < -dyn_threshold) & (sub_quality > q_cut)

            yes_payout = np.where(sub_outcome[buy_yes] == 1, 1.0, 0.0)
            yes_net = yes_payout - yes_payout * bench.FEE_RATE - sub_ask[buy_yes]
            no_payout = np.where(sub_outcome[buy_no] == 0, 1.0, 0.0)
            no_net = no_payout - no_payout * bench.FEE_RATE - (1.0 - sub_bid[buy_no])

            all_net = np.concatenate([yes_net, no_net])
            all_cost = np.concatenate([sub_ask[buy_yes], 1.0 - sub_bid[buy_no]])
            all_wins = np.concatenate([sub_outcome[buy_yes] == 1, sub_outcome[buy_no] == 0])

            net_pnl = float(all_net.sum()) if len(all_net) else 0.0
            total_cost = float(all_cost.sum()) if len(all_cost) else 0.0
            trades = int(len(all_net))
            win_rate = float(np.mean(all_wins)) if len(all_wins) else 0.0

            if trades == 0:
                pnl_ci_low, pnl_ci_high = 0.0, 0.0
                roi_ci_low, roi_ci_high = 0.0, 0.0
            else:
                trades_df = pd.DataFrame({"date": np.concatenate([sub["date"].values[buy_yes], sub["date"].values[buy_no]]),
                                          "net": all_net,
                                          "cost": all_cost})
                by_date = trades_df.groupby("date", as_index=False)[["net", "cost"]].sum()
                day_net = by_date["net"].values
                day_cost = by_date["cost"].values
                n_days = len(day_net)

                sampled_idx = rng.integers(0, n_days, size=(n_boot, n_days))
                boot_net = day_net[sampled_idx].sum(axis=1)
                boot_cost = day_cost[sampled_idx].sum(axis=1)
                boot_roi = np.where(boot_cost > 0, 100.0 * boot_net / boot_cost, 0.0)

                pnl_ci_low, pnl_ci_high = np.percentile(boot_net, [2.5, 97.5]).tolist()
                roi_ci_low, roi_ci_high = np.percentile(boot_roi, [2.5, 97.5]).tolist()

            rows.append({
                "model": label,
                "period": period,
                "quality_cut": q_cut,
                "n_trades": trades,
                "net_pnl": round(net_pnl, 2),
                "roi_pct": round((100.0 * net_pnl / total_cost), 2) if total_cost > 0 else 0.0,
                "win_rate": round(win_rate, 4),
                "pnl_ci95_low": round(float(pnl_ci_low), 2),
                "pnl_ci95_high": round(float(pnl_ci_high), 2),
                "roi_ci95_low": round(float(roi_ci_low), 2),
                "roi_ci95_high": round(float(roi_ci_high), 2),
                "bootstrap_samples": n_boot,
            })
    return pd.DataFrame(rows)


def _run_variant(base_df: pd.DataFrame, variant: str, cfg: dict, save_artifacts: bool = False) -> dict:
    df = _apply_variant(base_df, variant, cfg)
    scores_df = bench.compute_all_scores(df)
    cal_df = bench.compute_calibration(df)
    trading_df = bench.run_all_trading_sims(df)

    if save_artifacts:
        out_dir = OUT_ROOT / variant
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
        "model": variant,
        "overall_model_brier": float(overall.loc["Model", "brier_score"]),
        "overall_nws_brier": float(overall.loc["NWS", "brier_score"]),
        "overall_presettlement_brier": float(overall.loc["Kalshi_PreSettlement", "brier_score"]),
        "oos_model_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "Model")]["brier_score"].iloc[0]),
        "oos_nws_brier": float(scores_df[(scores_df["slice"] == "Period: OOS") & (scores_df["source"] == "NWS")]["brier_score"].iloc[0]),
        "best_model_all_trading_pnl": float(trading_df[trading_df["signal"] == "Model_All"]["net_pnl"].max()),
        "best_model_oos_trading_pnl": float(trading_df[trading_df["signal"] == "Model_OOS"]["net_pnl"].max()),
    }


def main() -> None:
    # Refresh experiment summary first.
    exp.run()

    base_df = e012.load_base_dataset()
    cfg = _fit_experiment_transforms(base_df)

    variants = [
        "E0_baseline_ensemble",
        "E1_global_isotonic",
        "E2_seasonal_calibration",
        "E3_weighted_ensemble_E4_uncertainty",
        "E4_uncertainty_decomposition",
        "E5_mdn2",
        "E6_quantile",
        "E7_regularization_sweep",
        "E8_feature_pruning_sweep",
        "E9_conditional_calibration_grid",
    ]

    rows = [_run_variant(base_df, v, cfg, save_artifacts=False) for v in variants]
    summary = pd.DataFrame(rows).sort_values("overall_model_brier").reset_index(drop=True)
    summary.to_csv(OUT_ROOT / "e0_e8_benchmark_summary.csv", index=False)

    top_model_name = summary.iloc[0]["model"]
    top_df = _apply_variant(base_df, top_model_name, cfg)
    gating_df = _run_edge_quality_gating(top_df, top_model_name)
    gating_df.to_csv(OUT_ROOT / "ev_edge_quality_gating_results.csv", index=False)

    top2 = summary.head(2).copy()
    top2.to_csv(OUT_ROOT / "top2_benchmark_summary.csv", index=False)

    for model_name in top2["model"].tolist():
        _run_variant(base_df, model_name, cfg, save_artifacts=True)

    with open(OUT_ROOT / "benchmark_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lineage": "all_variants_best_model_based",
                "variants": variants,
                "top2": top2["model"].tolist(),
                "calibration_period": "2023",
                "benchmark_period": "2023-2025",
                "ev_gating_results": "ev_edge_quality_gating_results.csv",
            },
            f,
            indent=2,
        )

    with open(OUT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write("# E0-E8 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n## Top 2\n\n")
        f.write(top2.to_string(index=False))
        f.write("\n\n## EV-aware dynamic edge gating (best-Brier model)\n\n")
        f.write(gating_df.to_string(index=False))
        f.write("\n")

    print("Saved:")
    print(f"  - {OUT_ROOT / 'e0_e8_benchmark_summary.csv'}")
    print(f"  - {OUT_ROOT / 'top2_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
