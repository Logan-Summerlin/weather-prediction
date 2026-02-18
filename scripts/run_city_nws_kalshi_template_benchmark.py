#!/usr/bin/env python3
"""Run NYC-template benchmark (PHL/CHI) against NWS MOS and Kalshi pre-settlement.

This script ports NYC best-practice families into city expansion workflows:
1) station-feature ridge baseline,
2) MOS-residual correction (ridge + NN),
3) U7-style regime-aware bucket synthesis stacker,
4) NWS MOS baseline,
5) Kalshi pre-settlement contract benchmark when archives are present.

All fitting uses real observed targets and archived inputs only.
No synthetic training/evaluation data are generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import ensure_city_dirs, get_city_config

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1 - 1e-4

CITY_TO_MOS = {
    "phl": Path("data/philadelphia/mos/combined_mos_kphl.csv"),
    "chi": Path("data/chicago/mos/combined_mos_kord.csv"),
    "aus": Path("data/austin/mos/combined_mos_kaus.csv"),
}
CITY_TICKER_PATTERNS = {
    "phl": ("HIGHPHL", "KXHIGHPHL"),
    "chi": ("HIGHCHI", "KXHIGHCHI"),
    "aus": ("HIGHAUS", "KXHIGHAUS"),
}


def _load_processed(city: str):
    cfg = get_city_config(city)
    processed = Path(cfg.data_dir) / "processed"
    X_train = pd.read_csv(processed / "features_train.csv", index_col=0, parse_dates=True)
    X_val = pd.read_csv(processed / "features_val.csv", index_col=0, parse_dates=True)
    X_test = pd.read_csv(processed / "features_test.csv", index_col=0, parse_dates=True)
    y_train = pd.read_csv(processed / "target_train.csv", index_col=0, parse_dates=True).iloc[:, 0]
    y_val = pd.read_csv(processed / "target_val.csv", index_col=0, parse_dates=True).iloc[:, 0]
    y_test = pd.read_csv(processed / "target_test.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return cfg, X_train, X_val, X_test, y_train, y_val, y_test


def _mos_column(df: pd.DataFrame) -> str:
    for c in ["mos_ensemble_tmax_f", "gfs_mos_tmax_f", "nam_mos_tmax_f"]:
        if c in df.columns:
            return c
    raise ValueError("No MOS Tmax column found.")


def _prepare_mos(city: str, y_all: pd.Series) -> pd.DataFrame:
    p = CITY_TO_MOS[city]
    if not p.exists():
        raise FileNotFoundError(f"MOS file not found: {p}")
    mos = pd.read_csv(p, parse_dates=["date"])
    mos["date"] = pd.to_datetime(mos["date"]).dt.normalize()
    col = _mos_column(mos)

    df = pd.DataFrame({"date": pd.to_datetime(y_all.index).normalize(), "actual": y_all.values})
    merged = df.merge(mos[["date", col]], on="date", how="left").rename(columns={col: "mos_tmax"})
    merged = merged.sort_values("date")
    merged["mos_error_lag1"] = (merged["actual"] - merged["mos_tmax"]).shift(1)
    merged["mos_error_7d"] = merged["mos_error_lag1"].rolling(7, min_periods=3).mean()
    merged["mos_error_14d"] = merged["mos_error_lag1"].rolling(14, min_periods=5).mean()
    merged["mos_abs_error_7d"] = merged["mos_error_lag1"].abs().rolling(7, min_periods=3).mean()
    merged["month"] = merged["date"].dt.month
    merged["doy"] = merged["date"].dt.dayofyear
    return merged


def _bucket_probs(mu: np.ndarray, sigma: np.ndarray, edges):
    out = np.zeros((len(mu), len(edges)))
    for i, (lo, hi) in enumerate(edges):
        c_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        c_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        out[:, i] = np.clip(c_hi - c_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _daily_outcomes(actual, edges):
    outcomes = np.zeros((len(actual), len(edges)))
    for d, t in enumerate(actual):
        for b, (lo, hi) in enumerate(edges):
            if (b == len(edges) - 1 and lo <= t <= hi) or (b < len(edges) - 1 and lo <= t < hi):
                outcomes[d, b] = 1.0
                break
    return outcomes


def _brier_daily(bucket_probs, actual, edges):
    outcomes = _daily_outcomes(actual, edges)
    return float(np.mean((bucket_probs - outcomes) ** 2))


def _fit_monthly_sigma(residuals: np.ndarray, months: np.ndarray):
    monthly = {}
    for m in range(1, 13):
        r = residuals[months == m]
        if len(r) >= 15:
            monthly[m] = float(np.std(r))
    global_sigma = max(1.0, float(np.std(residuals)))
    return monthly, global_sigma


def _sigma_for_months(monthly: dict, global_sigma: float, months: np.ndarray):
    return np.array([max(1.0, monthly.get(int(m), global_sigma)) for m in months], dtype=float)


def _fit_models(X_train, X_val, X_test, y_train, y_val, y_test, mos_df, bucket_edges):
    y_all = pd.concat([y_train, y_val, y_test])
    idx = pd.to_datetime(y_all.index).normalize()

    mos_indexed = mos_df.set_index("date").reindex(idx)
    mos_feats = ["mos_tmax", "mos_error_lag1", "mos_error_7d", "mos_error_14d", "mos_abs_error_7d", "month", "doy"]
    mos_indexed[mos_feats] = mos_indexed[mos_feats].ffill().bfill()

    n_train, n_val = len(y_train), len(y_val)
    y_arr = y_all.values
    months_all = mos_indexed["month"].values

    # Station-feature ridge
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train.values, y_train.values)
    mu_ridge_train = ridge.predict(X_train.values)
    mu_ridge_val = ridge.predict(X_val.values)
    mu_ridge_test = ridge.predict(X_test.values)
    resid_ridge_train = y_train.values - mu_ridge_train
    month_sigma_ridge, gsig_ridge = _fit_monthly_sigma(resid_ridge_train, months_all[:n_train])
    sigma_ridge_test = _sigma_for_months(month_sigma_ridge, gsig_ridge, months_all[n_train + n_val:])

    # MOS residual feature matrix
    X_res_all = mos_indexed[mos_feats].values
    y_res_all = y_arr - mos_indexed["mos_tmax"].values
    X_res_train, X_res_val, X_res_test = X_res_all[:n_train], X_res_all[n_train:n_train + n_val], X_res_all[n_train + n_val:]
    y_res_train, y_res_val = y_res_all[:n_train], y_res_all[n_train:n_train + n_val]

    # MOS residual ridge
    mos_ridge = Ridge(alpha=5.0)
    mos_ridge.fit(X_res_train, y_res_train)
    resid_ridge_val = mos_ridge.predict(X_res_val)
    resid_ridge_test = mos_ridge.predict(X_res_test)
    mu_mos_ridge_test = mos_indexed["mos_tmax"].values[n_train + n_val:] + resid_ridge_test

    # MOS residual NN (NYC template)
    mos_nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=5e-4,
        max_iter=700,
        random_state=42,
    )
    mos_nn.fit(np.vstack([X_res_train, X_res_val]), np.concatenate([y_res_train, y_res_val]))
    resid_nn_train = mos_nn.predict(X_res_train)
    resid_nn_val = mos_nn.predict(X_res_val)
    resid_nn_test = mos_nn.predict(X_res_test)
    mu_mos_nn_test = mos_indexed["mos_tmax"].values[n_train + n_val:] + resid_nn_test

    resid_cal = np.concatenate([y_res_train - resid_nn_train, y_res_val - resid_nn_val])
    months_cal = months_all[:n_train + n_val]
    month_sigma_mos, gsig_mos = _fit_monthly_sigma(resid_cal, months_cal)
    sigma_mos_test = _sigma_for_months(month_sigma_mos, gsig_mos, months_all[n_train + n_val:])

    # NWS MOS baseline
    mu_nws_test = mos_indexed["mos_tmax"].values[n_train + n_val:]

    # Isotonic calibration for gaussian models using val-only CDF PIT values.
    def calibrated_probs(mu_val, mu_test, sigma_test):
        val_sigma = _sigma_for_months(month_sigma_mos, gsig_mos, months_all[n_train:n_train + n_val])
        pit_raw = norm.cdf(y_val.values, loc=mu_val, scale=val_sigma)
        pit_raw = np.clip(pit_raw, PROB_CLIP_MIN, PROB_CLIP_MAX)
        iso = IsotonicRegression(out_of_bounds="clip")
        q = np.linspace(0.01, 0.99, 99)
        emp = np.array([(pit_raw <= qq).mean() for qq in q])
        iso.fit(q, emp)

        p_raw = _bucket_probs(mu_test, sigma_test, bucket_edges)
        p_cal = p_raw.copy()
        for i, (lo, hi) in enumerate(bucket_edges):
            c_lo = 0.0 if lo <= -900 else iso.predict(norm.cdf(lo, loc=mu_test, scale=sigma_test))
            c_hi = 1.0 if hi >= 900 else iso.predict(norm.cdf(hi, loc=mu_test, scale=sigma_test))
            p_cal[:, i] = np.clip(c_hi - c_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)
        return p_cal / p_cal.sum(axis=1, keepdims=True)

    mu_mos_nn_val = mos_indexed["mos_tmax"].values[n_train:n_train + n_val] + resid_nn_val
    mu_mos_ridge_val = mos_indexed["mos_tmax"].values[n_train:n_train + n_val] + resid_ridge_val

    model_outputs = {
        "ridge": (mu_ridge_test, sigma_ridge_test, _bucket_probs(mu_ridge_test, sigma_ridge_test, bucket_edges)),
        "mos_residual_ridge": (
            mu_mos_ridge_test,
            sigma_mos_test,
            calibrated_probs(mu_mos_ridge_val, mu_mos_ridge_test, sigma_mos_test),
        ),
        "mos_residual_nn": (
            mu_mos_nn_test,
            sigma_mos_test,
            calibrated_probs(mu_mos_nn_val, mu_mos_nn_test, sigma_mos_test),
        ),
        "nws_mos": (mu_nws_test, sigma_mos_test, _bucket_probs(mu_nws_test, sigma_mos_test, bucket_edges)),
    }

    # U7-style regime conditional synthesis (contract-row logistic stacker on val).
    probs_val = {
        "ridge": _bucket_probs(mu_ridge_val, _sigma_for_months(month_sigma_ridge, gsig_ridge, months_all[n_train:n_train + n_val]), bucket_edges),
        "mos": calibrated_probs(mu_mos_nn_val, mu_mos_nn_val, _sigma_for_months(month_sigma_mos, gsig_mos, months_all[n_train:n_train + n_val])),
        "nws": _bucket_probs(mos_indexed["mos_tmax"].values[n_train:n_train + n_val], _sigma_for_months(month_sigma_mos, gsig_mos, months_all[n_train:n_train + n_val]), bucket_edges),
    }
    probs_test = {
        "ridge": model_outputs["ridge"][2],
        "mos": model_outputs["mos_residual_nn"][2],
        "nws": model_outputs["nws_mos"][2],
    }

    def _stack_features(prob_map, dates, sigma):
        n_days, n_buckets = prob_map["mos"].shape
        centers = np.array([(lo + hi) / 2 for lo, hi in bucket_edges], dtype=float)
        m = prob_map["mos"].reshape(-1)
        r = prob_map["ridge"].reshape(-1)
        n = prob_map["nws"].reshape(-1)
        bc = np.tile(centers, n_days)
        month = np.repeat(pd.to_datetime(dates).month.values, n_buckets)
        sin_month = np.sin(2 * np.pi * month / 12.0)
        cos_month = np.cos(2 * np.pi * month / 12.0)
        s_norm = np.repeat((sigma - np.percentile(sigma, 5)) / (np.percentile(sigma, 95) - np.percentile(sigma, 5) + 1e-6), n_buckets)
        X = np.column_stack([
            m, r, n,
            m - r, m - n, r - n,
            m * (1 - s_norm),
            bc / 100.0,
            sin_month, cos_month,
        ])
        return np.nan_to_num(X, nan=0.0)

    val_dates = y_val.index
    test_dates = y_test.index
    sigma_val_for_stack = _sigma_for_months(month_sigma_mos, gsig_mos, months_all[n_train:n_train + n_val])
    X_stack_val = _stack_features(probs_val, val_dates, sigma_val_for_stack)
    X_stack_test = _stack_features(probs_test, test_dates, sigma_mos_test)
    y_stack_val = _daily_outcomes(y_val.values, bucket_edges).reshape(-1)

    stacker = LogisticRegression(C=0.5, max_iter=1500, solver="lbfgs")
    stacker.fit(X_stack_val, y_stack_val)
    p_stack = np.clip(stacker.predict_proba(X_stack_test)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    p_stack = p_stack.reshape(len(y_test), len(bucket_edges))
    p_stack /= p_stack.sum(axis=1, keepdims=True)

    model_outputs["u7_style_regime_stacker"] = (None, None, p_stack)
    return model_outputs


def _load_kalshi_contract_rows(city: str, valid_dates: pd.DatetimeIndex) -> pd.DataFrame:
    pre = pd.read_csv("data/kalshi_presettlement.csv")
    settled = pd.concat([
        pd.read_csv("data/real_kalshi_2023_2024.csv"),
        pd.read_csv("data/real_kalshi_2025.csv"),
    ], ignore_index=True)
    patterns = CITY_TICKER_PATTERNS[city]
    mask_pre = pre["ticker"].astype(str).str.contains("|".join(patterns), na=False)
    mask_set = settled["ticker"].astype(str).str.contains("|".join(patterns), na=False)
    pre = pre[mask_pre].copy()
    settled = settled[mask_set].copy()
    if pre.empty or settled.empty:
        return pd.DataFrame()
    merged = pre.merge(
        settled[["date", "ticker", "direction", "threshold_low", "threshold_high", "actual_outcome", "actual_tmax"]],
        on=["date", "ticker"], how="inner"
    )
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged[merged["date"].isin(pd.to_datetime(valid_dates).normalize())].copy()
    return merged


def _contract_probs(frame: pd.DataFrame, mu_by_date: pd.Series, sigma_by_date: pd.Series) -> np.ndarray:
    mu = frame["date"].map(mu_by_date).values
    sig = frame["date"].map(sigma_by_date).values
    probs = np.zeros(len(frame))
    below = frame["direction"] == "below"
    above = frame["direction"] == "above"
    between = frame["direction"] == "between"
    probs[below] = norm.cdf(frame.loc[below, "threshold_high"].values, loc=mu[below], scale=sig[below])
    probs[above] = 1.0 - norm.cdf(frame.loc[above, "threshold_low"].values, loc=mu[above], scale=sig[above])
    probs[between] = norm.cdf(frame.loc[between, "threshold_high"].values, loc=mu[between], scale=sig[between]) - norm.cdf(frame.loc[between, "threshold_low"].values, loc=mu[between], scale=sig[between])
    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["phl", "chi", "aus"], required=True)
    args = ap.parse_args()

    cfg = get_city_config(args.city)
    ensure_city_dirs(cfg)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.city}_nyc_template_benchmark.csv"
    out_json = out_dir / f"{args.city}_nyc_template_benchmark.json"

    try:
        cfg, X_train, X_val, X_test, y_train, y_val, y_test = _load_processed(args.city)
        y_all = pd.concat([y_train, y_val, y_test])
        mos_df = _prepare_mos(args.city, y_all)
        models = _fit_models(X_train, X_val, X_test, y_train, y_val, y_test, mos_df, cfg.bucket_edges)
    except Exception as exc:
        pd.DataFrame([{"source": "status", "message": str(exc)}]).to_csv(out_csv, index=False)
        out_json.write_text(json.dumps({
            "city": args.city,
            "status": "missing_required_inputs",
            "error": str(exc),
            "output_csv": str(out_csv),
        }, indent=2))
        print(f"Wrote {out_csv}")
        print(f"Wrote {out_json}")
        return

    rows = []
    for name, (mu, sigma, probs) in models.items():
        rows.append({"source": name, "daily_bucket_brier": _brier_daily(probs, y_test.values, cfg.bucket_edges)})

    kalshi = _load_kalshi_contract_rows(args.city, pd.to_datetime(y_test.index))
    kalshi_status = "unavailable"
    if not kalshi.empty:
        outcomes = kalshi["actual_outcome"].astype(float).values
        kalshi["presettlement_prob"] = kalshi["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
        rows.append({
            "source": "kalshi_presettlement",
            "contract_brier": float(np.mean((kalshi["presettlement_prob"].values - outcomes) ** 2)),
        })
        for name, (mu, sigma, _probs) in models.items():
            if mu is None or sigma is None:
                continue
            mu_s = pd.Series(mu, index=pd.to_datetime(y_test.index).normalize())
            sg_s = pd.Series(sigma, index=pd.to_datetime(y_test.index).normalize())
            p = _contract_probs(kalshi, mu_s, sg_s)
            rows.append({"source": f"{name}_on_kalshi_contracts", "contract_brier": float(np.mean((p - outcomes) ** 2))})
        kalshi_status = "available"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    metadata = {
        "city": args.city,
        "kalshi_contract_benchmark_status": kalshi_status,
        "notes": "Uses actual station outcomes and archived MOS/Kalshi files only. No synthetic data.",
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
