#!/usr/bin/env python3
"""Benchmark E0-E8 best-model-derived variants vs NWS + Kalshi pre-settlement.

All variants are defined as transformations/calibrations of canonical best-model
prediction artifacts (data/best_model_predictions_*).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

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

    # Lightweight synthesis-stacker fit using calibration-year only data.
    stacker = _fit_synthesis_stacker(base_df[base_df["date_dt"].dt.year == 2023].copy())
    neural_stacker = _fit_neural_synthesis_stacker(base_df[base_df["date_dt"].dt.year == 2023].copy())
    distributional_neural = _fit_distributional_neural_synthesis(base_df[base_df["date_dt"].dt.year == 2023].copy())

    # Capacity sweep multipliers (small regularized search on calibration year).
    capacity = _fit_capacity_sweep(calib)

    return {
        "global_cal": global_cal,
        "seasonal_cal": seasonal_cal,
        "sigma_mult_global": sigma_mult_global,
        "sigma_mult_season": sigma_mult_season,
        "offset_global": offset_global,
        "offset_by_season": offset_by_season,
        "resid_scale": resid_scale,
        "conditional_cal": _fit_conditional_calibration(calib),
        "conditional_cal_v2": _fit_conditional_calibration_v2(base_df[base_df["date_dt"].dt.year == 2023].copy()),
        "synthesis_stacker": stacker,
        "neural_synthesis_stacker": neural_stacker,
        "distributional_neural": distributional_neural,
        "capacity_sweep": capacity,
        "contract_audit": _build_contract_and_timesafe_audit(base_df),
    }


def _build_market_state_features(frame: pd.DataFrame, sigma_p05: float | None = None, sigma_p95: float | None = None) -> pd.DataFrame:
    """Build chronology-safe market/state features used by synthesis and gating layers."""
    spread = ((
        frame["ask_cents"].fillna(frame["presettlement_prob"] * 100)
        - frame["bid_cents"].fillna(frame["presettlement_prob"] * 100)
    ).clip(lower=0) / 100.0).values

    sigma = frame["model_sigma"].values
    if sigma_p05 is None:
        sigma_p05 = float(np.percentile(sigma, 5))
    if sigma_p95 is None:
        sigma_p95 = float(np.percentile(sigma, 95))
    sigma_norm = np.clip((sigma - sigma_p05) / (sigma_p95 - sigma_p05 + 1e-6), 0.0, 1.0)

    volume = np.log1p(frame["volume"].fillna(0.0).values)
    oi = np.log1p(frame["open_interest"].fillna(0.0).values)
    depth = np.clip(
        0.6 * (volume / (np.nanpercentile(volume, 95) + 1e-6))
        + 0.4 * (oi / (np.nanpercentile(oi, 95) + 1e-6)),
        0.0,
        1.0,
    )

    snapshot_dt = pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="coerce")
    cutoff_dt = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=5)
    staleness_hours = ((cutoff_dt - snapshot_dt).dt.total_seconds() / 3600.0).clip(lower=0.0)
    stale_norm = np.clip(staleness_hours.values / 8.0, 0.0, 1.0)

    return pd.DataFrame(
        {
            "spread": spread,
            "sigma_norm": sigma_norm,
            "depth": depth,
            "stale_norm": stale_norm,
            "sigma_p05": sigma_p05,
            "sigma_p95": sigma_p95,
        }
    )


def _fit_synthesis_stacker(calib: pd.DataFrame) -> dict[str, object]:
    """Fit trainable chronology-safe logistic stacker over model/NWS/market + state features."""
    frame = calib.copy()
    frame["model_prob"] = bench.compute_bucket_probs(frame, "model_mu", "model_sigma")
    frame["nws_prob"] = bench.compute_bucket_probs(frame, "nws_mu", "nws_sigma")
    frame[["model_prob", "nws_prob", "presettlement_prob"]] = frame[[
        "model_prob",
        "nws_prob",
        "presettlement_prob",
    ]].clip(bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    y = frame["actual_outcome"].values.astype(float)

    state = _build_market_state_features(frame)
    m = frame["model_prob"].values
    n = frame["nws_prob"].values
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    X = np.column_stack([
        m,
        n,
        k,
        m - k,
        m - n,
        n - k,
        spread,
        s,
        depth,
        stale,
        (m - k) * (1.0 - spread),
        (m - k) * (1.0 - s),
        (m - n) * (1.0 - s),
    ])

    n_train = int(len(frame) * 0.75)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_train_z = (X_train - mu) / sd
    X_val_z = (X_val - mu) / sd

    best = None
    best_brier = float("inf")
    for c in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        clf.fit(X_train_z, y_train)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best = clf

    assert best is not None
    X_all_z = (X - mu) / sd
    train_brier = float(np.mean((np.clip(best.predict_proba(X_all_z)[:, 1], bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX) - y) ** 2))

    return {
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "coef": best.coef_[0].tolist(),
        "intercept": float(best.intercept_[0]),
        "features": [
            "model_prob",
            "nws_prob",
            "presettlement_prob",
            "model_minus_market",
            "model_minus_nws",
            "nws_minus_market",
            "spread",
            "sigma_norm",
            "depth",
            "stale_norm",
            "edge_liquidity",
            "edge_confidence",
            "model_nws_confidence",
        ],
        "sigma_p05": float(state["sigma_p05"].iloc[0]),
        "sigma_p95": float(state["sigma_p95"].iloc[0]),
        "validation_brier": best_brier,
        "calibration_year_brier": train_brier,
    }


def _fit_capacity_sweep(calib: pd.DataFrame) -> dict[str, float]:
    """Select residual/uncertainty scaling from small regularized calibration sweep."""
    y = calib["actual_tmax"].values
    mu0 = calib["model_mu"].values
    sig0 = calib["model_sigma"].values
    season = _season_from_month(calib["date_dt"].dt.month.values)
    season_resid = calib.assign(season=season).groupby("season").apply(
        lambda x: float(np.mean(x["actual_tmax"] - x["model_mu"]))
    ).to_dict()

    best = {"resid_gain": 0.0, "sigma_gain": 1.0, "global_scale": 1.0}
    best_nll = float("inf")
    for rg in np.linspace(0.0, 0.4, 9):
        for sg in np.linspace(0.9, 1.4, 11):
            for gs in [0.95, 1.0, 1.05]:
                offs = np.array([season_resid.get(s, 0.0) for s in season])
                mu = mu0 + rg * offs
                sigma = np.clip(sig0 * sg * gs, 0.5, 15.0)
                z = (y - mu) / sigma
                nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * z**2))
                if nll < best_nll:
                    best_nll = nll
                    best = {"resid_gain": float(rg), "sigma_gain": float(sg), "global_scale": float(gs)}
    return best



def _fit_neural_synthesis_stacker(calib: pd.DataFrame) -> dict[str, object]:
    """Fit chronology-safe MLP synthesis stacker with isotonic post-calibration."""
    frame = calib.copy()
    frame["model_prob"] = bench.compute_bucket_probs(frame, "model_mu", "model_sigma")
    frame["nws_prob"] = bench.compute_bucket_probs(frame, "nws_mu", "nws_sigma")
    frame[["model_prob", "nws_prob", "presettlement_prob"]] = frame[[
        "model_prob",
        "nws_prob",
        "presettlement_prob",
    ]].clip(bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    y = frame["actual_outcome"].values.astype(float)

    state = _build_market_state_features(frame)
    m = frame["model_prob"].values
    n = frame["nws_prob"].values
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    X = np.column_stack([
        m,
        n,
        k,
        m - k,
        m - n,
        n - k,
        spread,
        s,
        depth,
        stale,
        m * (1.0 - spread),
        m * (1.0 - s),
        (m - k) * depth,
    ])

    n_total = len(frame)
    n_train = int(n_total * 0.60)
    n_val = int(n_total * 0.20)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    cal_idx = slice(n_train + n_val, n_total)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_train_z = (X_train - mu) / sd
    X_val_z = (X_val - mu) / sd
    X_cal_z = (X_cal - mu) / sd

    best = None
    best_brier = float("inf")
    configs = [
        ((16,), 1e-3),
        ((32,), 1e-3),
        ((32, 16), 1e-3),
        ((64, 32), 3e-3),
    ]
    for hidden, alpha in configs:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=alpha,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=42,
            early_stopping=False,
        )
        clf.fit(X_train_z, y_train)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best = clf

    assert best is not None
    cal_raw = np.clip(best.predict_proba(X_cal_z)[:, 1], bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=bench.PROB_CLIP_MIN, y_max=bench.PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    cal_calibrated = np.clip(iso.predict(cal_raw), bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)

    return {
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "coefs": [w.tolist() for w in best.coefs_],
        "intercepts": [b.tolist() for b in best.intercepts_],
        "hidden_layers": list(best.hidden_layer_sizes),
        "activation": best.activation,
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "features": [
            "model_prob",
            "nws_prob",
            "presettlement_prob",
            "model_minus_market",
            "model_minus_nws",
            "nws_minus_market",
            "spread",
            "sigma_norm",
            "depth",
            "stale_norm",
            "model_liquidity",
            "model_confidence",
            "edge_depth",
        ],
        "sigma_p05": float(state["sigma_p05"].iloc[0]),
        "sigma_p95": float(state["sigma_p95"].iloc[0]),
        "validation_brier": best_brier,
        "calibration_window_brier": float(np.mean((cal_calibrated - y_cal) ** 2)),
    }


def _build_distributional_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct date-level, time-safe features for distributional synthesis."""
    d = frame.copy()
    d["bucket_mid"] = np.where(
        d["direction"] == "above",
        d["threshold_low"] + 2.0,
        np.where(d["direction"] == "below", d["threshold_high"] - 2.0, (d["threshold_low"] + d["threshold_high"]) / 2.0),
    )
    state = _build_market_state_features(d)
    d["spread"] = state["spread"].values
    d["depth"] = state["depth"].values
    d["stale_norm"] = state["stale_norm"].values

    by_date = d.groupby("date", as_index=False).agg(
        date_dt=("date_dt", "first"),
        actual_tmax=("actual_tmax", "first"),
        model_mu=("model_mu", "first"),
        model_sigma=("model_sigma", "first"),
        nws_mu=("nws_mu", "first"),
        nws_sigma=("nws_sigma", "first"),
        market_prob_sum=("presettlement_prob", "sum"),
        market_implied_mu=("bucket_mid", lambda x: float(np.mean(x))),
        spread=("spread", "mean"),
        depth=("depth", "mean"),
        stale_norm=("stale_norm", "mean"),
    )

    weighted = d.groupby("date").apply(
        lambda x: pd.Series(
            {
                "market_implied_mu": float(np.sum(x["bucket_mid"] * x["presettlement_prob"]) / (np.sum(x["presettlement_prob"]) + 1e-6)),
                "market_implied_sigma": float(
                    np.sqrt(
                        np.sum(((x["bucket_mid"] - (np.sum(x["bucket_mid"] * x["presettlement_prob"]) / (np.sum(x["presettlement_prob"]) + 1e-6))) ** 2) * x["presettlement_prob"])
                        / (np.sum(x["presettlement_prob"]) + 1e-6)
                    )
                ),
            }
        )
    ).reset_index()
    by_date = by_date.drop(columns=["market_implied_mu"]).merge(weighted, on="date", how="left")

    by_date["mu_spread_model_nws"] = by_date["model_mu"] - by_date["nws_mu"]
    by_date["mu_spread_model_market"] = by_date["model_mu"] - by_date["market_implied_mu"]
    by_date["sigma_ratio_model_nws"] = by_date["model_sigma"] / (by_date["nws_sigma"] + 1e-6)
    by_date["sigma_ratio_model_market"] = by_date["model_sigma"] / (by_date["market_implied_sigma"] + 1e-6)
    by_date["season_sin"] = np.sin(2.0 * np.pi * by_date["date_dt"].dt.dayofyear.values / 365.25)
    by_date["season_cos"] = np.cos(2.0 * np.pi * by_date["date_dt"].dt.dayofyear.values / 365.25)
    return by_date.sort_values("date_dt").reset_index(drop=True)


def _fit_distributional_neural_synthesis(calib: pd.DataFrame) -> dict[str, object]:
    """Fit date-level neural residual/sigma synthesis with NLL selection and isotonic CDF calibration."""
    by_date = _build_distributional_date_features(calib)
    feature_cols = [
        "model_mu", "model_sigma", "nws_mu", "nws_sigma", "market_implied_mu", "market_implied_sigma",
        "mu_spread_model_nws", "mu_spread_model_market", "sigma_ratio_model_nws", "sigma_ratio_model_market",
        "spread", "depth", "stale_norm", "season_sin", "season_cos",
    ]
    X = by_date[feature_cols].values
    y = by_date["actual_tmax"].values
    base_mu = by_date["model_mu"].values
    resid_target = y - base_mu
    sigma_target = np.log(np.clip(np.abs(resid_target), 0.35, None))

    n = len(by_date)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    cal_idx = slice(n_train + n_val, n)

    X_train, X_val, X_cal = X[train_idx], X[val_idx], X[cal_idx]
    y_train, y_val, y_cal = y[train_idx], y[val_idx], y[cal_idx]
    mu_train_base, mu_val_base, mu_cal_base = base_mu[train_idx], base_mu[val_idx], base_mu[cal_idx]

    x_mu = X_train.mean(axis=0)
    x_sd = np.where(X_train.std(axis=0) < 1e-6, 1.0, X_train.std(axis=0))
    X_train_z = (X_train - x_mu) / x_sd
    X_val_z = (X_val - x_mu) / x_sd
    X_cal_z = (X_cal - x_mu) / x_sd

    best = None
    best_nll = float("inf")
    configs = [((16,), 1e-3), ((32,), 2e-3), ((32, 16), 3e-3), ((64, 32), 5e-3)]
    for hidden, alpha in configs:
        resid_model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=alpha, random_state=42, max_iter=1200)
        sig_model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", alpha=alpha, random_state=24, max_iter=1200)
        resid_model.fit(X_train_z, resid_target[train_idx])
        sig_model.fit(X_train_z, sigma_target[train_idx])

        mu_hat = mu_val_base + resid_model.predict(X_val_z)
        sigma_hat = np.clip(np.exp(sig_model.predict(X_val_z)), 0.5, 12.0)
        z = (y_val - mu_hat) / sigma_hat
        nll = float(np.mean(0.5 * np.log(2.0 * np.pi * sigma_hat**2) + 0.5 * z**2))
        if nll < best_nll:
            best_nll = nll
            best = (resid_model, sig_model, hidden, alpha)

    assert best is not None
    resid_model, sig_model, hidden, alpha = best
    mu_cal = mu_cal_base + resid_model.predict(X_cal_z)
    sigma_cal = np.clip(np.exp(sig_model.predict(X_cal_z)), 0.5, 12.0)
    f_cal = np.clip(e012._cdf(y_cal, mu_cal, sigma_cal), 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(f_cal, np.linspace(0.0, 1.0, len(f_cal), endpoint=False) + 0.5 / max(len(f_cal), 1))

    return {
        "feature_cols": feature_cols,
        "feature_mean": x_mu.tolist(),
        "feature_std": x_sd.tolist(),
        "resid_coefs": [w.tolist() for w in resid_model.coefs_],
        "resid_intercepts": [b.tolist() for b in resid_model.intercepts_],
        "sigma_coefs": [w.tolist() for w in sig_model.coefs_],
        "sigma_intercepts": [b.tolist() for b in sig_model.intercepts_],
        "hidden_layers": list(hidden),
        "alpha": float(alpha),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "validation_nll": best_nll,
    }


def _mlp_forward(X: np.ndarray, coefs: list[list[list[float]]], intercepts: list[list[float]]) -> np.ndarray:
    out = X
    n_layers = len(coefs)
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        w_arr = np.array(w)
        b_arr = np.array(b)
        out = out @ w_arr + b_arr
        if i < n_layers - 1:
            out = np.maximum(out, 0.0)
    return out.squeeze()

def _build_contract_and_timesafe_audit(df: pd.DataFrame) -> dict[str, object]:
    """Build audit metadata for contract alignment + time-safe live data checks."""
    out: dict[str, object] = {}
    d = df.copy()
    d["snapshot_dt_utc"] = pd.to_datetime(d["snapshot_time_utc"], utc=True, errors="coerce")
    d["contract_date_utc"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    cutoff_hour_utc = 5
    d["decision_cutoff_utc"] = d["contract_date_utc"] + pd.Timedelta(hours=cutoff_hour_utc)
    d["snapshot_lag_hours"] = (d["decision_cutoff_utc"] - d["snapshot_dt_utc"]).dt.total_seconds() / 3600.0

    out["contract_alignment"] = {
        "directions_seen": sorted(d["direction"].dropna().unique().tolist()),
        "rows_with_invalid_threshold_order": int(
            ((d["direction"] == "between") & (d["threshold_low"] >= d["threshold_high"])).sum()
        ),
        "rows_with_missing_between_bounds": int(
            ((d["direction"] == "between") & (d[["threshold_low", "threshold_high"]].isna().any(axis=1))).sum()
        ),
        "rows_with_missing_above_low": int(((d["direction"] == "above") & (d["threshold_low"].isna())).sum()),
        "rows_with_missing_below_high": int(((d["direction"] == "below") & (d["threshold_high"].isna())).sum()),
        "days_with_non_unit_probability_mass": int(
            (d.groupby("date")["settled_market_prob"].sum().sub(1.0).abs() > 1e-4).sum()
        ),
    }

    ticker_re = re.compile(r"^(?:KX)?HIGHNY-(\d{2}[A-Z]{3}\d{2})-([BT])(\d+(?:\.\d+)?)$")
    parsed = d["ticker"].astype(str).str.extract(ticker_re)
    parsed.columns = ["ticker_day", "ticker_kind", "ticker_strike"]
    d = pd.concat([d, parsed], axis=1)
    d["ticker_strike"] = pd.to_numeric(d["ticker_strike"], errors="coerce")

    d["date_fmt"] = pd.to_datetime(d["date"]).dt.strftime("%y%b%d").str.upper()
    strike_target = np.where(
        d["direction"] == "above",
        d["threshold_low"],
        np.where(d["direction"] == "below", d["threshold_high"], d["threshold_low"] + 0.5),
    )

    out["contract_alignment"].update(
        {
            "rows_with_unparseable_ticker": int(d["ticker_day"].isna().sum()),
            "rows_with_ticker_date_mismatch": int((d["ticker_day"] != d["date_fmt"]).sum()),
            "rows_with_ticker_strike_mismatch": int(
                np.nansum(np.abs(d["ticker_strike"].values - strike_target) > 1e-6)
            ),
            "rows_with_unexpected_ticker_kind": int(
                (~d["ticker_kind"].isin(["B", "T"]))
                .fillna(True)
                .sum()
            ),
        }
    )

    actual_rounded = np.rint(d["actual_tmax"].values)
    above_ok = np.where(d["direction"] == "above", actual_rounded >= d["threshold_low"], True)
    below_ok = np.where(d["direction"] == "below", actual_rounded < d["threshold_high"], True)
    between_ok = np.where(
        d["direction"] == "between",
        (actual_rounded >= d["threshold_low"]) & (actual_rounded < d["threshold_high"]),
        True,
    )
    implied = (above_ok & below_ok & between_ok).astype(int)
    out["contract_alignment"]["rows_with_outcome_rule_mismatch"] = int((implied != d["actual_outcome"]).sum())

    lag = d["snapshot_lag_hours"].replace([np.inf, -np.inf], np.nan)
    late_mask = lag < 0
    out["time_safety"] = {
        "decision_cutoff_utc_hour": cutoff_hour_utc,
        "snapshot_rows_total": int(lag.notna().sum()),
        "snapshot_rows_after_cutoff": int(late_mask.sum()),
        "snapshot_rows_after_cutoff_pct": float(100.0 * late_mask.mean()),
        "snapshot_lag_hours_p10": float(np.nanpercentile(lag, 10)),
        "snapshot_lag_hours_p50": float(np.nanpercentile(lag, 50)),
        "snapshot_lag_hours_p90": float(np.nanpercentile(lag, 90)),
    }
    return out


def _build_paper_trading_gate_report(
    top_model: str,
    summary: pd.DataFrame,
    top_scores: pd.DataFrame,
    top_calibration: pd.DataFrame,
    gating_df: pd.DataFrame,
) -> dict[str, object]:
    """Automate Phase-D paper-trade promotion checks from benchmark artifacts."""
    top_row = summary.loc[summary["model"] == top_model].iloc[0]
    pre_brier = float(top_row["overall_presettlement_brier"])
    oos_brier = float(top_row["oos_model_brier"])

    model_cal = top_calibration[top_calibration["source"] == "Model"]
    ece = float(model_cal["ece"].iloc[0]) if len(model_cal) else float("nan")
    tail_calib_error = float(
        np.nanmax(np.abs(model_cal["mean_predicted"] - model_cal["mean_observed"]))
    ) if len(model_cal) else float("nan")

    oos_gated = gating_df[gating_df["period"] == "OOS"]
    if len(oos_gated):
        best_oos = oos_gated.sort_values("net_pnl", ascending=False).iloc[0]
        oos_positive = bool(best_oos["net_pnl"] > 0 and best_oos["pnl_ci95_low"] > 0)
        oos_best = {
            "quality_cut": float(best_oos["quality_cut"]),
            "trades": int(best_oos["n_trades"]),
            "net_pnl": float(best_oos["net_pnl"]),
            "roi_pct": float(best_oos["roi_pct"]),
            "pnl_ci95_low": float(best_oos["pnl_ci95_low"]),
            "pnl_ci95_high": float(best_oos["pnl_ci95_high"]),
        }
    else:
        oos_positive = False
        oos_best = {}

    checks = {
        "oos_brier_beats_presettlement": {
            "pass": bool(oos_brier <= pre_brier),
            "oos_model_brier": oos_brier,
            "presettlement_brier": pre_brier,
        },
        "oos_gated_pnl_positive_with_positive_ci": {
            "pass": oos_positive,
            "best_oos_gated": oos_best,
        },
        "calibration_ece_gate": {
            "pass": bool(np.isfinite(ece) and ece <= 0.03),
            "ece": ece,
            "threshold": 0.03,
        },
        "tail_reliability_gate": {
            "pass": bool(np.isfinite(tail_calib_error) and tail_calib_error <= 0.20),
            "max_abs_bin_gap": tail_calib_error,
            "threshold": 0.20,
        },
    }

    return {
        "top_model": top_model,
        "promotion_ready": bool(all(v["pass"] for v in checks.values())),
        "checks": checks,
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


def _fit_conditional_calibration_v2(calib: pd.DataFrame) -> dict[str, object]:
    """Second-pass conditional isotonic: season x spread-tercile x regime-tercile with min-count fallback."""
    cal = calib.sort_values("date_dt").copy()
    cal["season"] = _season_from_month(cal["date_dt"].dt.month.values)
    cal["spread"] = ((cal["ask_cents"].fillna(cal["presettlement_prob"] * 100) - cal["bid_cents"].fillna(cal["presettlement_prob"] * 100)).clip(lower=0) / 100.0)
    cal["spread_bin"] = pd.qcut(cal["spread"], q=3, labels=["tight", "mid", "wide"], duplicates="drop").astype(str)

    by_date = cal[["date", "date_dt", "model_mu"]].drop_duplicates("date").sort_values("date_dt").copy()
    by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
    reg_edges = np.quantile(by_date["mu_change"].values, [0.0, 1 / 3, 2 / 3, 1.0])
    reg_edges = np.unique(reg_edges)
    by_date["regime_bin"] = _bin_labels(by_date["mu_change"].values, reg_edges, ["stable", "transition", "volatile"])
    reg_map = dict(zip(by_date["date"], by_date["regime_bin"]))
    cal["regime_bin"] = cal["date"].map(reg_map).fillna("transition")

    min_points = 60
    calibrators: dict[str, object] = {}
    cell_sizes: dict[str, int] = {}
    for key, grp in cal.groupby(["season", "spread_bin", "regime_bin"]):
        k = "|".join(key)
        cell_sizes[k] = int(len(grp))
        if len(grp) < min_points:
            continue
        calibrators[k] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)

    fallbacks: dict[str, object] = {
        "global": exp.calibrate_global(cal["model_mu"].values, cal["model_sigma"].values, cal["actual_tmax"].values),
    }
    for season, grp in cal.groupby("season"):
        if len(grp) >= min_points:
            fallbacks[f"season|{season}"] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)
    for spread_bin, grp in cal.groupby("spread_bin"):
        if len(grp) >= min_points:
            fallbacks[f"spread|{spread_bin}"] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)
    for regime_bin, grp in cal.groupby("regime_bin"):
        if len(grp) >= min_points:
            fallbacks[f"regime|{regime_bin}"] = exp.calibrate_global(grp["model_mu"].values, grp["model_sigma"].values, grp["actual_tmax"].values)

    spread_edges = np.quantile(cal["spread"].values, [0.0, 1 / 3, 2 / 3, 1.0])
    spread_edges = np.unique(spread_edges)

    return {
        "calibrators": calibrators,
        "fallbacks": fallbacks,
        "spread_edges": spread_edges,
        "regime_edges": reg_edges,
        "cell_sizes": cell_sizes,
        "min_points": min_points,
    }


def _bin_labels(values: np.ndarray, edges: np.ndarray, labels: list[str]) -> np.ndarray:
    if len(edges) < 2:
        return np.array([labels[0]] * len(values))
    idx = np.digitize(values, edges[1:-1], right=True)
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array([labels[i] for i in idx])


def _mixture_bucket_probs(
    threshold_low: np.ndarray,
    threshold_high: np.ndarray,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    w1: np.ndarray,
) -> np.ndarray:
    f1_lo = np.where(np.isnan(threshold_low), 0.0, e012._cdf(threshold_low, mu1, sigma1))
    f1_hi = np.where(np.isnan(threshold_high), 1.0, e012._cdf(threshold_high, mu1, sigma1))
    f2_lo = np.where(np.isnan(threshold_low), 0.0, e012._cdf(threshold_low, mu2, sigma2))
    f2_hi = np.where(np.isnan(threshold_high), 1.0, e012._cdf(threshold_high, mu2, sigma2))
    p1 = np.clip(f1_hi - f1_lo, 1e-6, 1.0)
    p2 = np.clip(f2_hi - f2_lo, 1e-6, 1.0)
    return np.clip(w1 * p1 + (1.0 - w1) * p2, bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)


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
    elif variant == "E10_wga_mdn_regime_mixture":
        mu_change = out.sort_values("date_dt").drop_duplicates("date")["model_mu"].diff().abs().fillna(0.0)
        date_mu_change = dict(
            zip(
                out.sort_values("date_dt").drop_duplicates("date")["date"],
                mu_change,
            )
        )
        reg = np.array([date_mu_change.get(d, 0.0) for d in out["date"].values])
        reg_norm = np.clip(reg / (np.nanpercentile(reg, 90) + 1e-6), 0.0, 1.0)
        out["_mdn_regime_norm"] = reg_norm
    elif variant == "E11_synthesis_stacker_market_aware":
        pass
    elif variant == "E12_capacity_sweep_residual_synthesis":
        cap = cfg["capacity_sweep"]
        offs = np.array([cfg["offset_by_season"].get(s, cfg["offset_global"]) for s in season])
        mu = mu + cap["resid_gain"] * offs
        sigma = np.clip(sigma * cap["sigma_gain"] * cap["global_scale"], 0.5, 15.0)
    elif variant == "E13_neural_synthesis_mlp":
        pass
    elif variant == "E14_distributional_neural_nll":
        pass
    elif variant == "E15_conditional_calibration_spread_regime":
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
    elif variant == "E10_wga_mdn_regime_mixture":
        reg_norm = out.pop("_mdn_regime_norm").values
        season_offsets = np.array([cfg["offset_by_season"].get(s, cfg["offset_global"]) for s in season])
        mu1 = mu + 0.45 * season_offsets
        mu2 = mu - 0.35 * season_offsets
        sigma1 = np.clip(sigma * (0.90 - 0.20 * reg_norm), 0.5, 15.0)
        sigma2 = np.clip(sigma * (1.10 + 0.35 * reg_norm), 0.5, 15.0)
        w1 = np.clip(0.70 - 0.35 * reg_norm, 0.30, 0.85)
        out["model_prob"] = _mixture_bucket_probs(
            out["threshold_low"].values,
            out["threshold_high"].values,
            mu1,
            sigma1,
            mu2,
            sigma2,
            w1,
        )
        # Preserve a comparable sigma summary for downstream gating diagnostics.
        out["model_sigma"] = np.clip(w1 * sigma1 + (1.0 - w1) * sigma2, 0.5, 15.0)
    elif variant == "E11_synthesis_stacker_market_aware":
        stack = cfg["synthesis_stacker"]
        model_prob = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
        nws_prob = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
        market_prob = out["presettlement_prob"].values
        state = _build_market_state_features(out, sigma_p05=stack["sigma_p05"], sigma_p95=stack["sigma_p95"])
        s = state["sigma_norm"].values
        spread = state["spread"].values
        depth = state["depth"].values
        stale = state["stale_norm"].values

        x = np.column_stack([
            model_prob,
            nws_prob,
            market_prob,
            model_prob - market_prob,
            model_prob - nws_prob,
            nws_prob - market_prob,
            spread,
            s,
            depth,
            stale,
            (model_prob - market_prob) * (1.0 - spread),
            (model_prob - market_prob) * (1.0 - s),
            (model_prob - nws_prob) * (1.0 - s),
        ])
        mu_x = np.array(stack["feature_mean"])
        sd_x = np.array(stack["feature_std"])
        beta = np.array(stack["coef"])
        z = (x - mu_x) / sd_x
        logits = z @ beta + float(stack["intercept"])
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        out["model_prob"] = np.clip(pred, bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    elif variant == "E13_neural_synthesis_mlp":
        stack = cfg["neural_synthesis_stacker"]
        model_prob = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
        nws_prob = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
        market_prob = out["presettlement_prob"].values
        state = _build_market_state_features(out, sigma_p05=stack["sigma_p05"], sigma_p95=stack["sigma_p95"])
        spread = state["spread"].values
        s = state["sigma_norm"].values
        depth = state["depth"].values
        stale = state["stale_norm"].values

        x = np.column_stack([
            model_prob,
            nws_prob,
            market_prob,
            model_prob - market_prob,
            model_prob - nws_prob,
            nws_prob - market_prob,
            spread,
            s,
            depth,
            stale,
            model_prob * (1.0 - spread),
            model_prob * (1.0 - s),
            (model_prob - market_prob) * depth,
        ])
        x = (x - np.array(stack["feature_mean"])) / np.array(stack["feature_std"])

        acts = x
        for i, (w, b) in enumerate(zip(stack["coefs"], stack["intercepts"])):
            acts = acts @ np.array(w) + np.array(b)
            if i < len(stack["coefs"]) - 1:
                acts = np.maximum(acts, 0.0)
        raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

        iso_x = np.array(stack["isotonic_x"])
        iso_y = np.array(stack["isotonic_y"])
        out["model_prob"] = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)
        out["model_prob"] = np.clip(out["model_prob"], bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    elif variant == "E14_distributional_neural_nll":
        dist = cfg["distributional_neural"]
        by_date = _build_distributional_date_features(out)
        x = by_date[dist["feature_cols"]].values
        x = (x - np.array(dist["feature_mean"])) / np.array(dist["feature_std"])

        resid = _mlp_forward(x, dist["resid_coefs"], dist["resid_intercepts"])
        sigma_raw = np.exp(_mlp_forward(x, dist["sigma_coefs"], dist["sigma_intercepts"]))
        mu_adj = by_date["model_mu"].values + resid
        sigma_adj = np.clip(sigma_raw, 0.5, 12.0)

        map_mu = dict(zip(by_date["date"], mu_adj))
        map_sigma = dict(zip(by_date["date"], sigma_adj))
        out["model_mu"] = out["date"].map(map_mu).astype(float)
        out["model_sigma"] = out["date"].map(map_sigma).astype(float)

        f_lo = np.where(np.isnan(out["threshold_low"].values), 0.0, e012._cdf(out["threshold_low"].values, out["model_mu"].values, out["model_sigma"].values))
        f_hi = np.where(np.isnan(out["threshold_high"].values), 1.0, e012._cdf(out["threshold_high"].values, out["model_mu"].values, out["model_sigma"].values))
        iso_x = np.array(dist["isotonic_x"])
        iso_y = np.array(dist["isotonic_y"])
        lo_cal = np.where(np.isnan(out["threshold_low"].values), 0.0, np.interp(np.clip(f_lo, iso_x.min(), iso_x.max()), iso_x, iso_y))
        hi_cal = np.where(np.isnan(out["threshold_high"].values), 1.0, np.interp(np.clip(f_hi, iso_x.min(), iso_x.max()), iso_x, iso_y))
        out["model_prob"] = np.clip(hi_cal - lo_cal, bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
    elif variant == "E15_conditional_calibration_spread_regime":
        f_lo = np.where(np.isnan(out["threshold_low"].values), 0.0, e012._cdf(out["threshold_low"].values, mu, sigma))
        f_hi = np.where(np.isnan(out["threshold_high"].values), 1.0, e012._cdf(out["threshold_high"].values, mu, sigma))

        spread = ((out["ask_cents"].fillna(out["presettlement_prob"] * 100) - out["bid_cents"].fillna(out["presettlement_prob"] * 100)).clip(lower=0) / 100.0).values
        spread_bin = _bin_labels(spread, cfg["conditional_cal_v2"]["spread_edges"], ["tight", "mid", "wide"])

        by_date = out[["date", "date_dt", "model_mu"]].drop_duplicates("date").sort_values("date_dt").copy()
        by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
        reg_bin = _bin_labels(by_date["mu_change"].values, cfg["conditional_cal_v2"]["regime_edges"], ["stable", "transition", "volatile"])
        reg_map = dict(zip(by_date["date"].values, reg_bin))
        out["regime_bin"] = out["date"].map(reg_map).fillna("transition")
        out["spread_bin"] = spread_bin

        out["model_prob"] = np.nan
        season_labels = _season_from_month(out["date_dt"].dt.month.values)
        for idx, s, sb, rb in zip(out.index, season_labels, out["spread_bin"].values, out["regime_bin"].values):
            key = f"{s}|{sb}|{rb}"
            cal = cfg["conditional_cal_v2"]["calibrators"].get(key)
            if cal is None:
                cal = cfg["conditional_cal_v2"]["fallbacks"].get(f"season|{s}")
            if cal is None:
                cal = cfg["conditional_cal_v2"]["fallbacks"].get(f"spread|{sb}")
            if cal is None:
                cal = cfg["conditional_cal_v2"]["fallbacks"].get(f"regime|{rb}")
            if cal is None:
                cal = cfg["conditional_cal_v2"]["fallbacks"]["global"]
            lo = 0.0 if np.isnan(out.at[idx, "threshold_low"]) else float(np.clip(cal.predict([f_lo[idx]])[0], 1e-6, 1 - 1e-6))
            hi = 1.0 if np.isnan(out.at[idx, "threshold_high"]) else float(np.clip(cal.predict([f_hi[idx]])[0], 1e-6, 1 - 1e-6))
            out.at[idx, "model_prob"] = np.clip(hi - lo, bench.PROB_CLIP_MIN, bench.PROB_CLIP_MAX)
        out.drop(columns=[c for c in ["regime_bin", "spread_bin"] if c in out.columns], inplace=True)
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
    volume = np.log1p(df["volume"].fillna(0.0).values)
    oi = np.log1p(df["open_interest"].fillna(0.0).values)
    depth = np.clip(0.6 * (volume / (np.nanpercentile(volume, 95) + 1e-6)) + 0.4 * (oi / (np.nanpercentile(oi, 95) + 1e-6)), 0.0, 1.0)

    snapshot_dt = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    cutoff_dt = pd.to_datetime(df["date"], utc=True, errors="coerce") + pd.Timedelta(hours=5)
    staleness_hours = ((cutoff_dt - snapshot_dt).dt.total_seconds() / 3600.0).clip(lower=0.0)
    stale_norm = np.clip(staleness_hours.values / 8.0, 0.0, 1.0)

    liquidity = np.clip(1.0 - spread / 0.20, 0.0, 1.0)

    ask_all = df["ask_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0
    bid_all = df["bid_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0
    mid = np.clip(0.5 * (ask_all + bid_all), 1e-6, 1.0 - 1e-6)
    imbalance = np.clip(np.abs(df["presettlement_prob"].values - mid) / (spread + 1e-3), 0.0, 1.0)
    queue_pressure = np.clip((1.0 - depth) * (spread / 0.20) * (0.6 + 0.4 * imbalance), 0.0, 1.0)

    # Daily cancellation proxy from quote instability (high spread dispersion / tail spread).
    daily_spread = pd.DataFrame({"date": df["date"].values, "spread": spread}).groupby("date", as_index=False).agg(
        spread_std=("spread", "std"),
        spread_p90=("spread", lambda x: float(np.nanpercentile(x, 90))),
    )
    daily_spread = daily_spread.fillna(0.0)
    spread_std_norm = np.clip(
        daily_spread["spread_std"].values / (np.nanpercentile(daily_spread["spread_std"].values, 90) + 1e-6),
        0.0,
        1.0,
    )
    spread_p90_norm = np.clip(
        daily_spread["spread_p90"].values / (np.nanpercentile(daily_spread["spread_p90"].values, 90) + 1e-6),
        0.0,
        1.0,
    )
    daily_spread["cancel_proxy"] = np.clip(0.5 * spread_std_norm + 0.5 * spread_p90_norm, 0.0, 1.0)
    cancel_proxy_map = dict(zip(daily_spread["date"].values, daily_spread["cancel_proxy"].values))
    cancel_proxy = np.array([cancel_proxy_map.get(d, 0.5) for d in df["date"].values])

    # Execution latency proxy in seconds (queue/cancel pressure + staleness).
    latency_seconds = 15.0 + 35.0 * stale_norm + 45.0 * queue_pressure + 20.0 * cancel_proxy
    latency_norm = np.clip((latency_seconds - 15.0) / 120.0, 0.0, 1.0)

    # Calibration-confidence proxy from chronological calibration year only.
    cal = df[df["date_dt"].dt.year == 2023].copy()
    if len(cal):
        cal["season"] = _season_from_month(cal["date_dt"].dt.month.values)
        cal["sigma_bin"] = _bin_labels(
            cal["model_sigma"].values,
            np.quantile(cal["model_sigma"].values, [0.0, 0.5, 1.0]),
            ["lo", "hi"],
        )
        cell_stats = (
            cal.groupby(["season", "direction", "sigma_bin"], as_index=False)
            .agg(
                n=("actual_outcome", "size"),
                mean_pred=("model_prob", "mean"),
                mean_obs=("actual_outcome", "mean"),
            )
        )
        cell_stats["abs_gap"] = (cell_stats["mean_pred"] - cell_stats["mean_obs"]).abs()
        cell_stats["confidence"] = np.clip(
            (cell_stats["n"] / 90.0) * (1.0 - cell_stats["abs_gap"] / 0.20),
            0.0,
            1.0,
        )
        conf_map = {
            (r["season"], r["direction"], r["sigma_bin"]): float(r["confidence"])
            for _, r in cell_stats.iterrows()
        }
        global_conf = float(np.clip(cell_stats["confidence"].mean(), 0.1, 1.0))
    else:
        conf_map = {}
        global_conf = 0.5

    df = df.copy()
    df["season"] = _season_from_month(df["date_dt"].dt.month.values)
    df["sigma_bin"] = _bin_labels(
        df["model_sigma"].values,
        np.quantile(df["model_sigma"].values, [0.0, 0.5, 1.0]),
        ["lo", "hi"],
    )
    cal_conf = np.array([
        conf_map.get((s, d, b), global_conf)
        for s, d, b in zip(df["season"].values, df["direction"].values, df["sigma_bin"].values)
    ])

    quality = (
        np.abs(edge)
        * liquidity
        * (1.0 - 0.5 * sigma_norm)
        * (0.3 + 0.7 * depth)
        * (1.0 - 0.30 * stale_norm)
        * (1.0 - 0.20 * queue_pressure)
        * (1.0 - 0.15 * cancel_proxy)
        * (1.0 - 0.20 * latency_norm)
        * (0.5 + 0.5 * cal_conf)
    )

    outcome_all = df["actual_outcome"].values.astype(float)

    rng = np.random.default_rng(20260212)
    n_boot = 1000

    period_masks = {
        "All": np.ones(len(df), dtype=bool),
        "IS": df["period"].values == "IS",
        "OOS": df["period"].values == "OOS",
        "OOS_DJF": (df["period"].values == "OOS") & (df["season"].values == "DJF"),
        "OOS_MAM": (df["period"].values == "OOS") & (df["season"].values == "MAM"),
        "OOS_JJA": (df["period"].values == "OOS") & (df["season"].values == "JJA"),
        "OOS_SON": (df["period"].values == "OOS") & (df["season"].values == "SON"),
        "OOS_volatile": (df["period"].values == "OOS") & (sigma_norm > 0.67),
    }

    for period, m in period_masks.items():
        sub = df.loc[m].copy()
        sub_edge = edge[m]
        sub_quality = quality[m]
        sub_spread = spread[m]
        sub_sigma_norm = sigma_norm[m]
        sub_ask = ask_all[m]
        sub_bid = bid_all[m]
        sub_outcome = outcome_all[m]

        sub_depth = depth[m]
        sub_stale = stale_norm[m]
        sub_queue = queue_pressure[m]
        sub_cancel = cancel_proxy[m]
        sub_latency = latency_norm[m]
        sub_latency_seconds = latency_seconds[m]
        sub_model = sub["model_prob"].values
        sub_market = sub["presettlement_prob"].values

        for q_cut in [0.02, 0.03, 0.04, 0.05, 0.06]:
            dyn_threshold = (
                0.01
                + 0.5 * sub_spread
                + 0.04 * sub_sigma_norm
                + 0.02 * (1.0 - sub_depth)
                + 0.01 * sub_stale
                + 0.015 * sub_queue
                + 0.010 * sub_cancel
                + 0.010 * sub_latency
            )
            no_trade_mask = (sub_cancel > 0.85) | (sub_queue > 0.85) | (sub_latency > 0.85)
            buy_yes = (sub_edge > dyn_threshold) & (sub_quality >= q_cut)
            buy_no = (sub_edge < -dyn_threshold) & (sub_quality >= q_cut)
            buy_yes = buy_yes & ~no_trade_mask
            buy_no = buy_no & ~no_trade_mask

            # Contract cluster exposure control: max 2 positions per (date, neighboring strike neighborhood).
            strike = np.where(np.isnan(sub["threshold_low"].values), sub["threshold_high"].values, sub["threshold_low"].values)
            cluster_key = np.floor(strike / 2.0)
            risk_order = np.argsort(-sub_quality)
            keep = np.zeros(len(sub), dtype=bool)
            cluster_counts: dict[tuple[str, float], int] = {}
            for idx in risk_order:
                if not (buy_yes[idx] or buy_no[idx]):
                    continue
                k = (str(sub["date"].values[idx]), float(cluster_key[idx]))
                count = cluster_counts.get(k, 0)
                if count >= 2:
                    continue
                keep[idx] = True
                cluster_counts[k] = count + 1
            buy_yes = buy_yes & keep
            buy_no = buy_no & keep

            # Capped fractional Kelly sizing using model vs market probabilities.
            side_prob = np.where(buy_yes, sub_model, np.where(buy_no, 1.0 - sub_model, 0.0))
            side_price = np.where(buy_yes, sub_ask, np.where(buy_no, 1.0 - sub_bid, 0.0))
            edge_side = np.where(buy_yes | buy_no, side_prob - side_price, 0.0)
            kelly = np.where((buy_yes | buy_no) & (side_price > 1e-6) & (side_price < 1 - 1e-6), edge_side / (1.0 - side_price), 0.0)
            frac_kelly = np.clip(0.25 * kelly, 0.0, 0.30)
            stake = np.where(buy_yes | buy_no, np.maximum(0.25, frac_kelly), 0.0)

            slippage = np.clip(
                0.20 * sub_spread
                + 0.015 * (1.0 - sub_depth)
                + 0.01 * sub_stale
                + 0.008 * sub_queue
                + 0.006 * sub_cancel
                + 0.006 * sub_latency,
                0.0,
                0.025,
            )
            exec_yes_cost = np.clip(sub_ask + slippage, 0.0, 1.0)
            exec_no_cost = np.clip((1.0 - sub_bid) + slippage, 0.0, 1.0)

            yes_payout = np.where(sub_outcome[buy_yes] == 1, 1.0, 0.0)
            yes_stake = stake[buy_yes]
            no_stake = stake[buy_no]
            yes_net = yes_stake * (yes_payout - yes_payout * bench.FEE_RATE - exec_yes_cost[buy_yes])
            no_payout = np.where(sub_outcome[buy_no] == 0, 1.0, 0.0)
            no_net = no_stake * (no_payout - no_payout * bench.FEE_RATE - exec_no_cost[buy_no])

            all_net = np.concatenate([yes_net, no_net])
            all_cost = np.concatenate([yes_stake * exec_yes_cost[buy_yes], no_stake * exec_no_cost[buy_no]])
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
                "avg_stake": round(float(np.mean(stake[buy_yes | buy_no])) if trades else 0.0, 4),
                "net_pnl": round(net_pnl, 2),
                "roi_pct": round((100.0 * net_pnl / total_cost), 2) if total_cost > 0 else 0.0,
                "win_rate": round(win_rate, 4),
                "pnl_ci95_low": round(float(pnl_ci_low), 2),
                "pnl_ci95_high": round(float(pnl_ci_high), 2),
                "roi_ci95_low": round(float(roi_ci_low), 2),
                "roi_ci95_high": round(float(roi_ci_high), 2),
                "bootstrap_samples": n_boot,
                "avg_queue_pressure": round(float(np.mean(sub_queue[buy_yes | buy_no])) if trades else 0.0, 4),
                "avg_cancel_proxy": round(float(np.mean(sub_cancel[buy_yes | buy_no])) if trades else 0.0, 4),
                "avg_latency_seconds": round(float(np.mean(sub_latency_seconds[buy_yes | buy_no])) if trades else 0.0, 2),
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
        "scores_df": scores_df,
        "cal_df": cal_df,
        "trading_df": trading_df,
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
        "E10_wga_mdn_regime_mixture",
        "E11_synthesis_stacker_market_aware",
        "E12_capacity_sweep_residual_synthesis",
        "E13_neural_synthesis_mlp",
        "E14_distributional_neural_nll",
        "E15_conditional_calibration_spread_regime",
    ]

    full_rows = [_run_variant(base_df, v, cfg, save_artifacts=False) for v in variants]
    rows = [
        {k: v for k, v in row.items() if k not in {"scores_df", "cal_df", "trading_df"}}
        for row in full_rows
    ]
    by_model = {row["model"]: row for row in full_rows}

    summary = pd.DataFrame(rows).sort_values("overall_model_brier").reset_index(drop=True)
    summary.to_csv(OUT_ROOT / "e0_e14_benchmark_summary.csv", index=False)

    top_model_name = summary.iloc[0]["model"]
    top_df = _apply_variant(base_df, top_model_name, cfg)
    gating_df = _run_edge_quality_gating(top_df, top_model_name)
    gating_df.to_csv(OUT_ROOT / "ev_edge_quality_gating_results.csv", index=False)

    challenger_name = "E14_distributional_neural_nll"
    challenger_df = _apply_variant(base_df, challenger_name, cfg)
    challenger_gating_df = _run_edge_quality_gating(challenger_df, challenger_name)
    challenger_gating_df.to_csv(OUT_ROOT / "ev_edge_quality_gating_results_e14.csv", index=False)
    paper_gate = _build_paper_trading_gate_report(
        top_model_name,
        summary,
        by_model[top_model_name]["scores_df"],
        by_model[top_model_name]["cal_df"],
        gating_df,
    )

    with open(OUT_ROOT / "paper_trading_gate_report.json", "w", encoding="utf-8") as f:
        json.dump(paper_gate, f, indent=2)

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
                "ev_gating_results_e14": "ev_edge_quality_gating_results_e14.csv",
                "contract_timesafe_audit": "contract_and_timesafe_audit.json",
                "paper_trading_gate": "paper_trading_gate_report.json",
            },
            f,
            indent=2,
        )

    with open(OUT_ROOT / "contract_and_timesafe_audit.json", "w", encoding="utf-8") as f:
        json.dump(cfg["contract_audit"], f, indent=2)

    with open(OUT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write("# E0-E15 Best-Model-Based Benchmark vs NWS + Kalshi PreSettlement\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n## Top 2\n\n")
        f.write(top2.to_string(index=False))
        f.write("\n\n## EV-aware dynamic edge gating (best-Brier model)\n\n")
        f.write(gating_df.to_string(index=False))
        f.write("\n\n## EV-aware dynamic edge gating (E14 distributional neural challenger)\n\n")
        f.write(challenger_gating_df.to_string(index=False))
        f.write("\n\n## Contract/time-safe audit\n\n")
        f.write(json.dumps(cfg["contract_audit"], indent=2))
        f.write("\n\n## Paper-trading gate report\n\n")
        f.write(json.dumps(paper_gate, indent=2))
        f.write("\n")

    print("Saved:")
    print(f"  - {OUT_ROOT / 'e0_e14_benchmark_summary.csv'}")
    print(f"  - {OUT_ROOT / 'top2_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
