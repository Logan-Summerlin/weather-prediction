#!/usr/bin/env python3
"""Run NYC-template city benchmark (PHL/CHI) against NWS MOS and Kalshi pre-settlement.

Implements the NYC-style MOS residual correction idea:
1) train on actual station outcomes,
2) model MOS Tmax error (residual),
3) fit a small NN to correct residual,
4) convert calibrated Gaussian (mu, sigma) to bucket/contract probabilities.

No synthetic data are generated in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1 - 1e-4

CITY_TO_MOS = {
    "phl": Path("data/philadelphia/mos/combined_mos_kphl.csv"),
    "chi": Path("data/chicago/mos/combined_mos_kord.csv"),
}
CITY_TICKER_PATTERNS = {
    "phl": ("HIGHPHL", "KXHIGHPHL"),
    "chi": ("HIGHCHI", "KXHIGHCHI"),
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
    return merged


def _fit_models(X_train, X_val, X_test, y_train, y_val, y_test, mos_df):
    # Ridge baseline on station features
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train.values, y_train.values)
    mu_ridge = ridge.predict(X_test.values)
    sigma_ridge = max(1.0, float(np.std(y_train.values - ridge.predict(X_train.values))))

    # NYC-template MOS residual NN
    y_all = pd.concat([y_train, y_val, y_test])
    idx = pd.to_datetime(y_all.index).normalize()
    mos_indexed = mos_df.set_index("date").reindex(idx)
    mos_feats = ["mos_tmax", "mos_error_lag1", "mos_error_7d", "mos_error_14d", "mos_abs_error_7d", "month"]
    mos_indexed[mos_feats] = mos_indexed[mos_feats].ffill().bfill()

    n_train, n_val = len(y_train), len(y_val)
    X_res_all = mos_indexed[mos_feats].values
    y_res_all = (y_all.values - mos_indexed["mos_tmax"].values)

    X_res_train = X_res_all[:n_train]
    y_res_train = y_res_all[:n_train]
    X_res_val = X_res_all[n_train:n_train + n_val]
    y_res_val = y_res_all[n_train:n_train + n_val]
    X_res_test = X_res_all[n_train + n_val:]

    nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                      learning_rate_init=5e-4, max_iter=500, random_state=42)
    nn.fit(np.vstack([X_res_train, X_res_val]), np.concatenate([y_res_train, y_res_val]))
    resid_test = nn.predict(X_res_test)
    mu_mos_nn = mos_indexed["mos_tmax"].values[n_train + n_val:] + resid_test

    # sigma from train+val residuals, monthly where available
    resid_cal = np.concatenate([y_res_train - nn.predict(X_res_train), y_res_val - nn.predict(X_res_val)])
    months_cal = mos_indexed["month"].values[:n_train + n_val]
    month_sigma = {}
    for m in range(1, 13):
        r = resid_cal[months_cal == m]
        if len(r) >= 15:
            month_sigma[m] = float(np.std(r))
    global_sigma = max(1.0, float(np.std(resid_cal)))
    months_test = mos_indexed["month"].values[n_train + n_val:]
    sigma_mos_nn = np.array([max(1.0, month_sigma.get(int(m), global_sigma)) for m in months_test])

    # NWS MOS baseline (mu=mos)
    mos_test = mos_indexed["mos_tmax"].values[n_train + n_val:]
    nws_sigma = np.array([max(1.0, month_sigma.get(int(m), global_sigma)) for m in months_test])
    return {
        "ridge": (mu_ridge, np.full_like(mu_ridge, sigma_ridge, dtype=float)),
        "mos_residual_nn": (mu_mos_nn, sigma_mos_nn),
        "nws_mos": (mos_test, nws_sigma),
    }


def _bucket_probs(mu: np.ndarray, sigma: np.ndarray, edges):
    out = np.zeros((len(mu), len(edges)))
    for i, (lo, hi) in enumerate(edges):
        c_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        c_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        out[:, i] = np.clip(c_hi - c_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _brier_daily(bucket_probs, actual, edges):
    outcomes = np.zeros_like(bucket_probs)
    for d, t in enumerate(actual):
        for b, (lo, hi) in enumerate(edges):
            if (b == len(edges) - 1 and lo <= t <= hi) or (b < len(edges) - 1 and lo <= t < hi):
                outcomes[d, b] = 1.0
                break
    return float(np.mean((bucket_probs - outcomes) ** 2))


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
    ap.add_argument("--city", choices=["phl", "chi"], required=True)
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
        models = _fit_models(X_train, X_val, X_test, y_train, y_val, y_test, mos_df)
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
    for name, (mu, sigma) in models.items():
        bp = _bucket_probs(mu, sigma, cfg.bucket_edges)
        rows.append({"source": name, "daily_bucket_brier": _brier_daily(bp, y_test.values, cfg.bucket_edges)})

    # Kalshi pre-settlement benchmark when actual city rows exist in local data.
    kalshi = _load_kalshi_contract_rows(args.city, pd.to_datetime(y_test.index))
    kalshi_status = "unavailable"
    if not kalshi.empty:
        outcomes = kalshi["actual_outcome"].astype(float).values
        kalshi["presettlement_prob"] = kalshi["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
        rows.append({
            "source": "kalshi_presettlement",
            "contract_brier": float(np.mean((kalshi["presettlement_prob"].values - outcomes) ** 2)),
        })
        for name, (mu, sigma) in models.items():
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
