#!/usr/bin/env python3
"""
run_wga_benchmark.py

Evaluate the WGA-MDN model predictions on the Kalshi prediction market benchmark.
Compares against Kalshi pre-settlement, NWS, original flat NN, and Kalshi settled.

Implements four WGA-specific variants:
  E34: Raw WGA-MDN bucket probabilities (no post-processing)
  E35: Market-aware synthesis stacker (logistic regression meta-model)
  E36: Contract-level Brier-optimal MLP
  E37: Simple and weighted probability blend of WGA and original model

Outputs to: results/prediction_market_benchmark/wga_mdn_model/
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "wga_mdn_model"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import existing benchmark machinery
# ---------------------------------------------------------------------------

def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999
FEE_RATE = 0.07
TRADING_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]
TRADING_DAYS_PER_YEAR = 252
N_CAL_BINS = 10
N_BOOTSTRAP = 50

SEASON_MAP = bench.SEASON_MAP


# ==============================================================================
# Part 1: Load and Prepare Data
# ==============================================================================

def load_wga_predictions() -> pd.DataFrame:
    """Load WGA-MDN predictions for test period (2023-2024)."""
    path = ROOT / "results" / "wga_mdn_model" / "predictions_test.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def load_original_predictions_is() -> pd.DataFrame:
    """Load original flat NN predictions for IS period (2023-2024)."""
    path = ROOT / "data" / "best_model_predictions_2023_2024.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def load_original_predictions_oos() -> pd.DataFrame:
    """Load original flat NN predictions for OOS period (2025)."""
    path = ROOT / "data" / "best_model_predictions_2025.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def build_merged_dataset() -> pd.DataFrame:
    """Build the master merged dataset with all data sources.

    Returns a contract-level DataFrame with columns for model predictions (both WGA
    and original), NWS, pre-settlement, and settled market probabilities.
    """
    print("=" * 70)
    print("PART 1: Loading and Preparing Data")
    print("=" * 70)

    # Load raw data
    pre = pd.read_csv(ROOT / "data" / "kalshi_presettlement.csv")
    s23 = pd.read_csv(ROOT / "data" / "real_kalshi_2023_2024.csv")
    s25 = pd.read_csv(ROOT / "data" / "real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)
    nws = pd.read_csv(ROOT / "results" / "prediction_market_benchmark" / "nws_probability_forecasts.csv")

    print(f"  Pre-settlement:    {len(pre)} rows, {pre['date'].nunique()} dates")
    print(f"  Settled:           {len(settled)} rows, {settled['date'].nunique()} dates")
    print(f"  NWS:               {len(nws)} rows")

    # Merge pre-settlement to settled on date+ticker
    merged = pre.merge(
        settled[["date", "ticker", "direction", "threshold_low", "threshold_high",
                 "actual_outcome", "actual_tmax", "market_prob"]],
        on=["date", "ticker"],
        suffixes=("_pre", ""),
        how="inner",
    )
    merged = merged.rename(columns={"market_prob": "settled_market_prob"})
    n_before = len(merged)
    merged = merged.dropna(subset=["presettlement_prob"])
    print(f"  Pre+Settled merge: {n_before} -> {len(merged)} (dropped {n_before - len(merged)} NaN presettlement)")

    # Load WGA-MDN predictions (2023-2024 only)
    wga = load_wga_predictions()
    wga_model = wga[["date", "model_mu", "model_sigma_base",
                      "model_sigma_monthly_cal", "model_sigma_regime_conditional"]].copy()
    wga_model = wga_model.rename(columns={
        "model_mu": "wga_mu",
        "model_sigma_base": "wga_sigma_base",
        "model_sigma_monthly_cal": "wga_sigma_monthly",
        "model_sigma_regime_conditional": "wga_sigma_regime",
    })
    print(f"  WGA predictions:   {len(wga_model)} dates ({wga['date'].min()} to {wga['date'].max()})")

    # Load original flat model predictions (IS + OOS)
    orig_is = load_original_predictions_is()
    orig_oos = load_original_predictions_oos()
    orig_all = pd.concat([
        orig_is[["date", "model_mu", "model_sigma"]],
        orig_oos[["date", "model_mu", "model_sigma"]],
    ], ignore_index=True).drop_duplicates(subset="date", keep="first")
    orig_all = orig_all.rename(columns={
        "model_mu": "orig_mu",
        "model_sigma": "orig_sigma",
    })
    print(f"  Original preds:    {len(orig_all)} dates ({orig_all['date'].min()} to {orig_all['date'].max()})")

    # Merge model predictions
    merged = merged.merge(wga_model, on="date", how="left")
    merged = merged.merge(orig_all, on="date", how="inner")
    merged = merged.merge(nws[["date", "nws_mu", "nws_sigma"]], on="date", how="inner")

    # Add period and season
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["period"] = np.where(merged["date_dt"].dt.year <= 2024, "IS", "OOS")
    merged["season"] = merged["date_dt"].dt.month.map(SEASON_MAP)
    merged["month"] = merged["date_dt"].dt.month

    # Flag rows that have WGA predictions (IS period only)
    merged["has_wga"] = merged["wga_mu"].notna()

    print(f"\n  Final merged: {len(merged)} rows, {merged['date'].nunique()} dates")
    print(f"    IS:  {(merged['period'] == 'IS').sum()} rows ({(merged[merged['period']=='IS']['date'].nunique())} dates)")
    print(f"    OOS: {(merged['period'] == 'OOS').sum()} rows ({(merged[merged['period']=='OOS']['date'].nunique())} dates)")
    print(f"    Rows with WGA: {merged['has_wga'].sum()}")

    return merged


# ==============================================================================
# Probability Computation
# ==============================================================================

def compute_bucket_probs(df, mu_col, sigma_col):
    """Compute P(TMAX in bucket) from N(mu, sigma), vectorized."""
    mu = df[mu_col].values.astype(float)
    sigma = df[sigma_col].values.astype(float)

    probs = np.full(len(df), np.nan)

    below = df["direction"].values == "below"
    above = df["direction"].values == "above"
    between = df["direction"].values == "between"

    # Handle NaN thresholds for above/below directions
    th_low = df["threshold_low"].values.astype(float)
    th_high = df["threshold_high"].values.astype(float)

    if below.any():
        probs[below] = norm.cdf(th_high[below], mu[below], sigma[below])
    if above.any():
        probs[above] = 1.0 - norm.cdf(th_low[above], mu[above], sigma[above])
    if between.any():
        probs[between] = (
            norm.cdf(th_high[between], mu[between], sigma[between])
            - norm.cdf(th_low[between], mu[between], sigma[between])
        )

    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)


# ==============================================================================
# Scoring Metrics
# ==============================================================================

def brier_score(probs, outcomes):
    return float(np.mean((probs - outcomes) ** 2))


def log_score(probs, outcomes):
    p = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def expected_calibration_error(probs, outcomes, n_bins=N_CAL_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            avg_pred = float(probs[mask].mean())
            avg_outcome = float(outcomes[mask].mean())
            ece += (count / n) * abs(avg_pred - avg_outcome)
    return float(ece)


def reliability_diagram_data(probs, outcomes, n_bins=N_CAL_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count > 0:
            rows.append({
                "bin_center": round(float((bins[i] + bins[i + 1]) / 2), 4),
                "mean_predicted": round(float(probs[mask].mean()), 6),
                "mean_observed": round(float(outcomes[mask].mean()), 6),
                "count": count,
            })
        else:
            rows.append({
                "bin_center": round(float((bins[i] + bins[i + 1]) / 2), 4),
                "mean_predicted": round(float((bins[i] + bins[i + 1]) / 2), 6),
                "mean_observed": None,
                "count": 0,
            })
    return rows


def compute_comprehensive_metrics(df, variant, prob_col="model_prob"):
    """Compute all quality metrics for a benchmark variant."""
    probs = df[prob_col].values.astype(float)
    outcomes = df["actual_outcome"].values.astype(float)

    overall_bs = brier_score(probs, outcomes)
    overall_ls = log_score(probs, outcomes)
    overall_ece = expected_calibration_error(probs, outcomes)

    is_mask = df["period"].values == "IS"
    oos_mask = df["period"].values == "OOS"
    is_bs = brier_score(probs[is_mask], outcomes[is_mask]) if is_mask.any() else None
    oos_bs = brier_score(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None
    is_ls = log_score(probs[is_mask], outcomes[is_mask]) if is_mask.any() else None
    oos_ls = log_score(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None
    oos_ece = expected_calibration_error(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None

    # By season
    season_bs = {}
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        m = df["season"].values == s
        if m.any():
            season_bs[s] = brier_score(probs[m], outcomes[m])

    # By direction
    dir_bs = {}
    for d in ["above", "below", "between"]:
        m = df["direction"].values == d
        if m.any():
            dir_bs[d] = brier_score(probs[m], outcomes[m])

    rel_data = reliability_diagram_data(probs, outcomes)

    return {
        "variant": variant,
        "overall_brier": round(overall_bs, 6),
        "is_brier": round(is_bs, 6) if is_bs is not None else None,
        "oos_brier": round(oos_bs, 6) if oos_bs is not None else None,
        "overall_log_score": round(overall_ls, 6),
        "is_log_score": round(is_ls, 6) if is_ls is not None else None,
        "oos_log_score": round(oos_ls, 6) if oos_ls is not None else None,
        "overall_ece": round(overall_ece, 6),
        "oos_ece": round(oos_ece, 6) if oos_ece is not None else None,
        "season_brier": {k: round(v, 6) for k, v in season_bs.items()},
        "direction_brier": {k: round(v, 6) for k, v in dir_bs.items()},
        "reliability_diagram": rel_data,
        "n_total": int(len(df)),
        "n_is": int(is_mask.sum()),
        "n_oos": int(oos_mask.sum()),
    }


# ==============================================================================
# Market State Features (reused from benchmark)
# ==============================================================================

def build_market_state_features(frame, sigma_col="wga_sigma_base",
                                 sigma_p05=None, sigma_p95=None):
    """Build market/state features used by synthesis layers."""
    spread = ((
        frame["ask_cents"].fillna(frame["presettlement_prob"] * 100)
        - frame["bid_cents"].fillna(frame["presettlement_prob"] * 100)
    ).clip(lower=0) / 100.0).values

    sigma = frame[sigma_col].values.astype(float)
    if sigma_p05 is None:
        sigma_p05 = float(np.nanpercentile(sigma, 5))
    if sigma_p95 is None:
        sigma_p95 = float(np.nanpercentile(sigma, 95))
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

    return pd.DataFrame({
        "spread": spread,
        "sigma_norm": sigma_norm,
        "depth": depth,
        "stale_norm": stale_norm,
        "sigma_p05": sigma_p05,
        "sigma_p95": sigma_p95,
    })


# ==============================================================================
# Part 2: E34 - Base WGA-MDN Model
# ==============================================================================

def apply_e34_wga_base(df):
    """E34: Raw WGA-MDN bucket probabilities, no post-processing.

    For IS period (2023-2024): use WGA-MDN mu/sigma.
    For OOS period (2025): use original model (WGA not available).
    """
    out = df.copy()

    # WGA bucket probs for IS period
    is_mask = out["has_wga"].values
    if is_mask.any():
        is_probs = compute_bucket_probs(out[is_mask], "wga_mu", "wga_sigma_regime")
        out.loc[is_mask, "model_prob"] = is_probs

    # Original model bucket probs for OOS period (fallback)
    oos_mask = ~out["has_wga"].values
    if oos_mask.any():
        oos_probs = compute_bucket_probs(out[oos_mask], "orig_mu", "orig_sigma")
        out.loc[oos_mask, "model_prob"] = oos_probs

    out["model_prob"] = out["model_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ==============================================================================
# Part 3: E35 - Market-Aware Synthesis Stacker
# ==============================================================================

def fit_e35_synthesis_stacker(cal_df):
    """E35: Logistic regression meta-model trained on 2023 calibration year.

    Features: wga_prob, orig_prob, nws_prob, presettlement_prob, model-market
    differences, spread, sigma_norm, depth, staleness.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    frame = cal_df.copy()
    frame["wga_prob"] = compute_bucket_probs(frame, "wga_mu", "wga_sigma_regime")
    frame["orig_prob"] = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    frame["nws_prob"] = compute_bucket_probs(frame, "nws_mu", "nws_sigma")

    for col in ["wga_prob", "orig_prob", "nws_prob", "presettlement_prob"]:
        frame[col] = frame[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    y = frame["actual_outcome"].values.astype(float)

    state = build_market_state_features(frame, sigma_col="wga_sigma_regime")
    w = frame["wga_prob"].values
    o = frame["orig_prob"].values
    n = frame["nws_prob"].values
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    X = np.column_stack([
        w, o, n, k,
        w - k, w - n, o - k, o - n, n - k,
        spread, s, depth, stale,
        (w - k) * (1.0 - spread),
        (w - k) * (1.0 - s),
        (w - n) * (1.0 - s),
    ])

    # 75/25 chronological split for train/calibration
    n_train = int(len(frame) * 0.75)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_train_z = (X_train - mu) / sd
    X_val_z = (X_val - mu) / sd

    best_clf = None
    best_brier = float("inf")
    for c in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        clf.fit(X_train_z, y_train)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best_clf = clf

    assert best_clf is not None

    # Isotonic post-calibration on validation set
    val_raw = np.clip(best_clf.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(val_raw, y_val)

    return {
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "coef": best_clf.coef_[0].tolist(),
        "intercept": float(best_clf.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": float(state["sigma_p05"].iloc[0]),
        "sigma_p95": float(state["sigma_p95"].iloc[0]),
        "validation_brier": best_brier,
    }


def apply_e35_synthesis_stacker(df, e35):
    """Apply E35 synthesis stacker to full dataset."""
    out = df.copy()

    # Compute base probs -- use WGA where available, else original
    has_wga = out["has_wga"].values
    wga_prob = np.zeros(len(out))
    if has_wga.any():
        wga_prob[has_wga] = compute_bucket_probs(out[has_wga], "wga_mu", "wga_sigma_regime")
    # For OOS, use original model probs as proxy for WGA
    if (~has_wga).any():
        wga_prob[~has_wga] = compute_bucket_probs(out[~has_wga], "orig_mu", "orig_sigma")

    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    for arr in [wga_prob, orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    state = build_market_state_features(out, sigma_col="orig_sigma",
                                         sigma_p05=e35["sigma_p05"],
                                         sigma_p95=e35["sigma_p95"])
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    x = np.column_stack([
        wga_prob, orig_prob, nws_prob, market_prob,
        wga_prob - market_prob, wga_prob - nws_prob,
        orig_prob - market_prob, orig_prob - nws_prob, nws_prob - market_prob,
        spread, s, depth, stale,
        (wga_prob - market_prob) * (1.0 - spread),
        (wga_prob - market_prob) * (1.0 - s),
        (wga_prob - nws_prob) * (1.0 - s),
    ])
    mu_x = np.array(e35["feature_mean"])
    sd_x = np.array(e35["feature_std"])
    z = (x - mu_x) / sd_x
    logits = z @ np.array(e35["coef"]) + float(e35["intercept"])
    pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e35["isotonic_x"])
    iso_y = np.array(e35["isotonic_y"])
    pred = np.interp(np.clip(pred, iso_x.min(), iso_x.max()), iso_x, iso_y)

    out["model_prob"] = np.clip(pred, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ==============================================================================
# Part 3: E36 - Contract-Level Brier-Optimal MLP
# ==============================================================================

def fit_e36_contract_brier_mlp(cal_df):
    """E36: Contract-level Brier-optimal MLP with WGA-specific features.

    Same structure as E17 but with WGA model as base + WGA-specific features.
    60/20/20 train/val/cal split on 2023.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neural_network import MLPClassifier

    frame = cal_df.copy()
    frame["wga_prob"] = compute_bucket_probs(frame, "wga_mu", "wga_sigma_regime")
    frame["orig_prob"] = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    frame["nws_prob"] = compute_bucket_probs(frame, "nws_mu", "nws_sigma")

    for col in ["wga_prob", "orig_prob", "nws_prob", "presettlement_prob"]:
        frame[col] = frame[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    y = frame["actual_outcome"].values.astype(float)

    state = build_market_state_features(frame, sigma_col="wga_sigma_regime")
    w = frame["wga_prob"].values
    o = frame["orig_prob"].values
    n = frame["nws_prob"].values
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # Bucket-specific features
    bucket_mid = np.where(
        frame["direction"] == "above",
        frame["threshold_low"].values + 2.0,
        np.where(
            frame["direction"] == "below",
            frame["threshold_high"].values - 2.0,
            (frame["threshold_low"].values + frame["threshold_high"].values) / 2.0,
        ),
    )
    mu_vals = frame["wga_mu"].values.astype(float)
    sig_vals = frame["wga_sigma_regime"].values.astype(float)
    bucket_quantile = norm.cdf(bucket_mid, mu_vals, sig_vals)
    bucket_width = np.where(
        frame["direction"] == "between",
        (frame["threshold_high"].values - frame["threshold_low"].values) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )
    bucket_distance_sigma = np.abs(bucket_mid - mu_vals) / (sig_vals + 1e-6)
    direction_above = (frame["direction"].values == "above").astype(float)
    direction_below = (frame["direction"].values == "below").astype(float)

    # Neighboring bucket sum
    frame["_wga_prob_tmp"] = w
    date_sum = frame.groupby("date")["_wga_prob_tmp"].transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - w, 0.0, None)

    # WGA-specific features
    wga_orig_diff = w - o  # WGA vs original model disagreement

    X = np.column_stack([
        w, o, n, k,
        w - k, w - n, o - k, n - k,
        spread, s, depth, stale,
        bucket_quantile, bucket_width, bucket_distance_sigma,
        direction_above, direction_below,
        neighboring_bucket_sum,
        wga_orig_diff,
    ])

    # 3-way chronological split: 60/20/20
    n_total = len(frame)
    n_train = int(n_total * 0.60)
    n_val = int(n_total * 0.20)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    cal_idx = slice(n_train + n_val, n_total)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x = np.where(sd_x < 1e-6, 1.0, sd_x)
    X_train_z = (X_train - mu_x) / sd_x
    X_val_z = (X_val - mu_x) / sd_x
    X_cal_z = (X_cal - mu_x) / sd_x

    best = None
    best_score = float("inf")
    ece_lambda = 0.15
    configs = [
        ((32,), 3e-3, 1e-3),
        ((64, 32), 5e-3, 8e-4),
        ((128, 64), 8e-3, 6e-4),
        ((128, 64, 32), 1e-2, 5e-4),
    ]
    for hidden, alpha, lr in configs:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
            learning_rate_init=lr, max_iter=1200, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        )
        clf.fit(X_train_z, y_train)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        val_ece = expected_calibration_error(val_pred, y_val)
        score = val_brier + ece_lambda * val_ece
        if score < best_score:
            best_score = score
            best = (clf, hidden, alpha, lr)

    assert best is not None
    clf, hidden, alpha, lr = best
    cal_raw = np.clip(clf.predict_proba(X_cal_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    return {
        "feature_mean": mu_x.tolist(),
        "feature_std": sd_x.tolist(),
        "coefs": [ww.tolist() for ww in clf.coefs_],
        "intercepts": [bb.tolist() for bb in clf.intercepts_],
        "hidden_layers": list(hidden),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": float(state["sigma_p05"].iloc[0]),
        "sigma_p95": float(state["sigma_p95"].iloc[0]),
        "validation_score": best_score,
    }


def apply_e36_contract_brier_mlp(df, e36):
    """Apply E36 contract-level Brier MLP to full dataset."""
    out = df.copy()

    has_wga = out["has_wga"].values

    # Compute WGA probs (use orig for OOS fallback)
    wga_prob = np.zeros(len(out))
    if has_wga.any():
        wga_prob[has_wga] = compute_bucket_probs(out[has_wga], "wga_mu", "wga_sigma_regime")
    if (~has_wga).any():
        wga_prob[~has_wga] = compute_bucket_probs(out[~has_wga], "orig_mu", "orig_sigma")

    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    for arr in [wga_prob, orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    state = build_market_state_features(out, sigma_col="orig_sigma",
                                         sigma_p05=e36["sigma_p05"],
                                         sigma_p95=e36["sigma_p95"])
    spread = state["spread"].values
    s = state["sigma_norm"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # Use orig_mu/sigma for bucket features on OOS (where WGA not available)
    mu_vals = np.where(has_wga, out["wga_mu"].fillna(0).values, out["orig_mu"].values).astype(float)
    sig_vals = np.where(has_wga, out["wga_sigma_regime"].fillna(1).values, out["orig_sigma"].values).astype(float)

    bucket_mid = np.where(
        out["direction"] == "above",
        out["threshold_low"].values + 2.0,
        np.where(
            out["direction"] == "below",
            out["threshold_high"].values - 2.0,
            (out["threshold_low"].values + out["threshold_high"].values) / 2.0,
        ),
    )
    bucket_quantile = norm.cdf(bucket_mid.astype(float), mu_vals, sig_vals)
    bucket_width = np.where(
        out["direction"] == "between",
        (out["threshold_high"].values - out["threshold_low"].values).astype(float) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )
    bucket_distance_sigma = np.abs(bucket_mid.astype(float) - mu_vals) / (sig_vals + 1e-6)
    direction_above = (out["direction"].values == "above").astype(float)
    direction_below = (out["direction"].values == "below").astype(float)

    date_sum = pd.Series(wga_prob).groupby(out["date"].values).transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - wga_prob, 0.0, None)

    wga_orig_diff = wga_prob - orig_prob

    x = np.column_stack([
        wga_prob, orig_prob, nws_prob, market_prob,
        wga_prob - market_prob, wga_prob - nws_prob,
        orig_prob - market_prob, nws_prob - market_prob,
        spread, s, depth, stale,
        bucket_quantile, bucket_width, bucket_distance_sigma,
        direction_above, direction_below,
        neighboring_bucket_sum,
        wga_orig_diff,
    ])
    x = (x - np.array(e36["feature_mean"])) / np.array(e36["feature_std"])

    # MLP forward pass
    acts = x
    for i, (wt, b) in enumerate(zip(e36["coefs"], e36["intercepts"])):
        acts = acts @ np.array(wt) + np.array(b)
        if i < len(e36["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)  # ReLU
    raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e36["isotonic_x"])
    iso_y = np.array(e36["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ==============================================================================
# Part 3: E37 - Ensemble Blend
# ==============================================================================

def apply_e37_ensemble_blend(df, weight_wga=0.5):
    """E37: Simple probability average/weighted blend of WGA and original model."""
    out = df.copy()

    has_wga = out["has_wga"].values

    wga_prob = np.zeros(len(out))
    if has_wga.any():
        wga_prob[has_wga] = compute_bucket_probs(out[has_wga], "wga_mu", "wga_sigma_regime")
    if (~has_wga).any():
        wga_prob[~has_wga] = compute_bucket_probs(out[~has_wga], "orig_mu", "orig_sigma")

    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")

    for arr in [wga_prob, orig_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    blended = weight_wga * wga_prob + (1.0 - weight_wga) * orig_prob
    out["model_prob"] = np.clip(blended, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def fit_e37_optimal_weights(cal_df):
    """Find optimal WGA weight on calibration data for E37 blend."""
    frame = cal_df.copy()
    wga_prob = compute_bucket_probs(frame, "wga_mu", "wga_sigma_regime")
    orig_prob = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    np.clip(wga_prob, PROB_CLIP_MIN, PROB_CLIP_MAX, out=wga_prob)
    np.clip(orig_prob, PROB_CLIP_MIN, PROB_CLIP_MAX, out=orig_prob)

    outcomes = frame["actual_outcome"].values.astype(float)

    best_w = 0.5
    best_bs = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        blended = w * wga_prob + (1.0 - w) * orig_prob
        blended = np.clip(blended, PROB_CLIP_MIN, PROB_CLIP_MAX)
        bs = float(np.mean((blended - outcomes) ** 2))
        if bs < best_bs:
            best_bs = bs
            best_w = w

    return {"optimal_weight": float(best_w), "calibration_brier": float(best_bs)}


# ==============================================================================
# Part 4: Trading Simulation
# ==============================================================================

def run_trading_sim(df, signal_col, market_col, threshold, label):
    """Simulate trading: buy YES when signal > market + threshold, buy NO when opposite."""
    signal = df[signal_col].values
    market = df[market_col].values
    outcome = df["actual_outcome"].values.astype(float)
    edge = signal - market

    buy_yes = edge > threshold
    buy_no = edge < -threshold
    total_trades = buy_yes.sum() + buy_no.sum()

    if total_trades == 0:
        return {
            "signal": label, "market": "Kalshi_PreSettlement", "threshold": threshold,
            "n_trades": 0, "n_yes_trades": 0, "n_no_trades": 0,
            "total_cost": 0, "gross_payout": 0, "fees": 0, "net_pnl": 0,
            "roi_pct": 0, "win_rate": 0, "avg_edge": 0,
            "sharpe": 0, "annualized_sharpe": 0,
        }

    ask = df["ask_cents"].fillna(df[market_col] * 100).values / 100.0
    bid = df["bid_cents"].fillna(df[market_col] * 100).values / 100.0

    # YES trades
    yes_cost = ask[buy_yes]
    yes_wins = outcome[buy_yes] == 1
    yes_payout = np.where(yes_wins, 1.0, 0.0)
    yes_fees = yes_payout * FEE_RATE
    yes_net = yes_payout - yes_fees - yes_cost

    # NO trades
    no_cost = 1.0 - bid[buy_no]
    no_wins = outcome[buy_no] == 0
    no_payout = np.where(no_wins, 1.0, 0.0)
    no_fees = no_payout * FEE_RATE
    no_net = no_payout - no_fees - no_cost

    all_net = np.concatenate([yes_net, no_net])
    all_cost = np.concatenate([yes_cost, no_cost])
    all_wins = np.concatenate([yes_wins, no_wins])

    total_cost = all_cost.sum()
    gross_payout = np.concatenate([yes_payout, no_payout]).sum()
    fees = np.concatenate([yes_fees, no_fees]).sum()
    net_pnl = all_net.sum()
    roi = (net_pnl / total_cost * 100) if total_cost > 0 else 0
    win_rate = all_wins.mean() if len(all_wins) > 0 else 0
    avg_edge = np.abs(edge[buy_yes | buy_no]).mean()
    sharpe = (all_net.mean() / all_net.std()) if all_net.std() > 0 else 0
    annualized_sharpe = sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "signal": label, "market": "Kalshi_PreSettlement", "threshold": threshold,
        "n_trades": int(total_trades), "n_yes_trades": int(buy_yes.sum()),
        "n_no_trades": int(buy_no.sum()),
        "total_cost": round(total_cost, 2), "gross_payout": round(gross_payout, 2),
        "fees": round(fees, 2), "net_pnl": round(net_pnl, 2),
        "roi_pct": round(roi, 2), "win_rate": round(win_rate, 4),
        "avg_edge": round(avg_edge, 4), "sharpe": round(sharpe, 4),
        "annualized_sharpe": round(annualized_sharpe, 4),
    }


def bootstrap_pnl_ci(df, signal_col, market_col, threshold, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap 95% CI for net P&L using date-level block resampling."""
    if len(df) == 0:
        return (0.0, 0.0)

    signal = df[signal_col].values
    market = df[market_col].values
    outcome = df["actual_outcome"].values.astype(float)
    ask = df["ask_cents"].fillna(df[market_col] * 100).values / 100.0
    bid = df["bid_cents"].fillna(df[market_col] * 100).values / 100.0

    edge = signal - market
    buy_yes = edge > threshold
    buy_no = edge < -threshold

    if not (buy_yes.any() or buy_no.any()):
        return (0.0, 0.0)

    pnl = np.zeros(len(df), dtype=float)
    pnl[buy_yes] = np.where(outcome[buy_yes] == 1, 1.0 - FEE_RATE, 0.0) - ask[buy_yes]
    pnl[buy_no] = np.where(outcome[buy_no] == 0, 1.0 - FEE_RATE, 0.0) - (1.0 - bid[buy_no])

    daily_pnl = pd.DataFrame({"date": df["date"].values, "pnl": pnl}).groupby("date")["pnl"].sum().values
    if len(daily_pnl) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(42)
    sampled_sums = [rng.choice(daily_pnl, size=len(daily_pnl), replace=True).sum()
                    for _ in range(n_bootstrap)]
    return float(np.percentile(sampled_sums, 2.5)), float(np.percentile(sampled_sums, 97.5))


def run_all_trading_sims(df, variant_prob_cols):
    """Run trading simulations for each variant vs pre-settlement market."""
    print("\n  Running trading simulations...")
    results = []

    for variant_name, prob_col in variant_prob_cols.items():
        for threshold in TRADING_THRESHOLDS:
            for period_label, period_mask in [("All", df.index == df.index),
                                               ("IS", df["period"] == "IS"),
                                               ("OOS", df["period"] == "OOS")]:
                sub = df[period_mask]
                if len(sub) == 0 or sub[prob_col].isna().all():
                    continue

                r = run_trading_sim(sub, prob_col, "presettlement_prob",
                                    threshold, f"{variant_name}_{period_label}")
                if period_label == "OOS":
                    ci_lo, ci_hi = bootstrap_pnl_ci(sub, prob_col, "presettlement_prob", threshold)
                    r["pnl_ci95_low"] = round(ci_lo, 2)
                    r["pnl_ci95_high"] = round(ci_hi, 2)
                results.append(r)

    return pd.DataFrame(results)


# ==============================================================================
# Part 5: Report Generation
# ==============================================================================

def generate_report(df, all_metrics, cal_df, trading_df, e37_weights):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# WGA-MDN Model: Prediction Market Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report evaluates the WGA-MDN (Weighted Gaussian Attention Mixture Density Network)")
    lines.append("model on the Kalshi NYC temperature prediction market benchmark.")
    lines.append("")
    lines.append("### Variants Evaluated")
    lines.append("")
    lines.append("| Variant | Description |")
    lines.append("|---------|-------------|")
    lines.append("| E34_wga_base | Raw WGA-MDN bucket probabilities (regime-conditional sigma) |")
    lines.append("| E35_wga_synthesis_stacker | Market-aware logistic regression meta-model |")
    lines.append("| E36_wga_contract_brier | Contract-level Brier-optimal MLP with WGA features |")
    lines.append("| E37_wga_blend_50_50 | 50/50 blend of WGA and original model |")
    lines.append(f"| E37_wga_blend_optimal | Optimally weighted blend (w_wga={e37_weights['optimal_weight']:.2f}) |")
    lines.append("")
    lines.append(f"- Total bucket-level observations: **{len(df):,}**")
    lines.append(f"- Unique dates: **{df['date'].nunique()}**")
    lines.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"- IS period (2023-2024): {(df['period'] == 'IS').sum():,} rows, {df[df['period']=='IS']['date'].nunique()} dates")
    lines.append(f"- OOS period (2025): {(df['period'] == 'OOS').sum():,} rows, {df[df['period']=='OOS']['date'].nunique()} dates")
    lines.append(f"- Rows with WGA predictions: {df['has_wga'].sum():,}")
    lines.append("")

    # --- Brier Scores ---
    lines.append("## 1. Brier Score Comparison (lower = better)")
    lines.append("")

    # Build summary table
    lines.append("### Overall and By Period")
    lines.append("")
    lines.append("| Variant | Overall Brier | IS Brier | OOS Brier | Overall LogScore | Overall ECE |")
    lines.append("|---------|--------------|----------|----------|-----------------|-------------|")
    for m in all_metrics:
        v = m["variant"]
        obs = m.get("overall_brier", None)
        isb = m.get("is_brier", None)
        oosb = m.get("oos_brier", None)
        ols = m.get("overall_log_score", None)
        oece = m.get("overall_ece", None)
        lines.append(f"| {v} | {obs:.4f} | {isb:.4f} | "
                     f"{oosb if oosb is not None else 'N/A'} | "
                     f"{ols:.4f} | {oece:.4f} |")
    lines.append("")

    # By season
    lines.append("### By Season (Overall Brier)")
    lines.append("")
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    header = "| Variant | " + " | ".join(seasons) + " |"
    lines.append(header)
    lines.append("|---------|" + "|".join(["------" for _ in seasons]) + "|")
    for m in all_metrics:
        v = m["variant"]
        sb = m.get("season_brier", {})
        vals = " | ".join([f"{sb.get(s, 'N/A')}" for s in seasons])
        lines.append(f"| {v} | {vals} |")
    lines.append("")

    # By direction
    lines.append("### By Direction (Overall Brier)")
    lines.append("")
    directions = ["above", "below", "between"]
    header = "| Variant | " + " | ".join(directions) + " |"
    lines.append(header)
    lines.append("|---------|" + "|".join(["------" for _ in directions]) + "|")
    for m in all_metrics:
        v = m["variant"]
        db = m.get("direction_brier", {})
        vals = " | ".join([f"{db.get(d, 'N/A')}" for d in directions])
        lines.append(f"| {v} | {vals} |")
    lines.append("")

    # --- Calibration ---
    lines.append("## 2. Calibration Analysis")
    lines.append("")
    lines.append("### Expected Calibration Error (ECE)")
    lines.append("")
    lines.append("| Variant | ECE |")
    lines.append("|---------|-----|")
    for m in all_metrics:
        lines.append(f"| {m['variant']} | {m.get('overall_ece', 'N/A')} |")
    lines.append("")

    # Reliability diagram data
    lines.append("### Reliability Diagram Data (10 bins)")
    lines.append("")
    if cal_df is not None and len(cal_df) > 0:
        for source in cal_df["source"].unique():
            src_data = cal_df[cal_df["source"] == source]
            lines.append(f"**{source}**")
            lines.append("")
            lines.append("| Bin | Mean Predicted | Mean Observed | Count |")
            lines.append("|-----|---------------|---------------|-------|")
            for _, row in src_data.iterrows():
                obs = f"{row['mean_observed']:.3f}" if pd.notna(row["mean_observed"]) else "N/A"
                lines.append(f"| {row['bin_center']:.2f} | {row['mean_predicted']:.3f} | {obs} | {int(row['count'])} |")
            lines.append("")
    lines.append("")

    # --- Trading Simulation ---
    lines.append("## 3. Trading Simulation")
    lines.append("")
    lines.append(f"Fee rate: {FEE_RATE*100:.0f}% on winnings")
    lines.append("")

    if trading_df is not None and len(trading_df) > 0:
        # Get unique variant prefixes
        variant_signals = sorted(trading_df["signal"].unique())

        # Group by variant
        seen_variants = set()
        for sig in variant_signals:
            # Extract variant name (everything before _All, _IS, _OOS)
            for suffix in ["_All", "_IS", "_OOS"]:
                if sig.endswith(suffix):
                    variant_base = sig[:-len(suffix)]
                    if variant_base not in seen_variants:
                        seen_variants.add(variant_base)
                        lines.append(f"### {variant_base}")
                        lines.append("")
                        lines.append("| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Ann. Sharpe |")
                        lines.append("|--------|-----------|--------|----------|---------|------|-------------|")
                        for period_suffix in ["_All", "_IS", "_OOS"]:
                            period_label = period_suffix[1:]
                            period_rows = trading_df[trading_df["signal"] == f"{variant_base}{period_suffix}"]
                            for _, row in period_rows.iterrows():
                                lines.append(
                                    f"| {period_label} | {row['threshold']:.2f} | {row['n_trades']} | "
                                    f"{row['win_rate']:.1%} | ${row['net_pnl']:.2f} | "
                                    f"{row['roi_pct']:.1f}% | {row['annualized_sharpe']:.3f} |"
                                )
                        lines.append("")
                    break

    # --- Key Findings ---
    lines.append("## 4. Key Findings")
    lines.append("")

    # Sort by overall Brier
    sorted_metrics = sorted(all_metrics, key=lambda m: m.get("overall_brier", 999))
    best = sorted_metrics[0]
    lines.append(f"- **Best overall Brier**: {best['variant']} ({best['overall_brier']:.4f})")

    # Compare WGA base vs benchmarks
    wga_base = next((m for m in all_metrics if m["variant"] == "E34_wga_base"), None)
    orig_base = next((m for m in all_metrics if m["variant"] == "Original_Model"), None)
    pre_base = next((m for m in all_metrics if m["variant"] == "Kalshi_PreSettlement"), None)
    nws_base = next((m for m in all_metrics if m["variant"] == "NWS"), None)

    if wga_base and orig_base:
        diff = wga_base["overall_brier"] - orig_base["overall_brier"]
        if diff < 0:
            lines.append(f"- E34_wga_base BEATS Original_Model by {abs(diff):.4f} Brier points overall")
        else:
            lines.append(f"- Original_Model BEATS E34_wga_base by {abs(diff):.4f} Brier points overall")

    if wga_base and pre_base:
        diff = wga_base["overall_brier"] - pre_base["overall_brier"]
        if diff < 0:
            lines.append(f"- E34_wga_base BEATS Kalshi PreSettlement by {abs(diff):.4f} Brier points overall")
        else:
            lines.append(f"- Kalshi PreSettlement BEATS E34_wga_base by {abs(diff):.4f} Brier points overall")

    if wga_base and nws_base:
        diff = wga_base["overall_brier"] - nws_base["overall_brier"]
        if diff < 0:
            lines.append(f"- E34_wga_base BEATS NWS by {abs(diff):.4f} Brier points overall")
        else:
            lines.append(f"- NWS BEATS E34_wga_base by {abs(diff):.4f} Brier points overall")

    lines.append("")
    lines.append(f"- E37 optimal WGA weight: {e37_weights['optimal_weight']:.2f}")
    lines.append(f"  (1.0 = pure WGA, 0.0 = pure original model)")
    lines.append("")

    # Best trading result
    if trading_df is not None and len(trading_df) > 0 and trading_df["n_trades"].sum() > 0:
        best_trade = trading_df.loc[trading_df["net_pnl"].idxmax()]
        lines.append(f"- Best trading result: {best_trade['signal']}, "
                     f"threshold={best_trade['threshold']:.2f}, "
                     f"P&L=${best_trade['net_pnl']:.2f}, "
                     f"ROI={best_trade['roi_pct']:.1f}%, "
                     f"{int(best_trade['n_trades'])} trades")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    np.random.seed(42)

    # ---- Part 1: Load and prepare data ----
    df = build_merged_dataset()

    # ---- Part 2: Compute baseline probabilities for benchmarks ----
    print("\n" + "=" * 70)
    print("PART 2: Computing Baseline Benchmark Probabilities")
    print("=" * 70)

    # Original model probs
    df["orig_prob"] = compute_bucket_probs(df, "orig_mu", "orig_sigma")
    df["nws_prob"] = compute_bucket_probs(df, "nws_mu", "nws_sigma")

    # Clip all benchmark probs
    for col in ["orig_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        df[col] = df[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    outcomes = df["actual_outcome"].values.astype(float)

    # Compute benchmark metrics
    all_metrics = []

    # Kalshi PreSettlement
    pre_metrics = compute_comprehensive_metrics(df, "Kalshi_PreSettlement", "presettlement_prob")
    all_metrics.append(pre_metrics)
    print(f"  Kalshi PreSettlement: Brier={pre_metrics['overall_brier']:.4f}")

    # NWS
    nws_metrics = compute_comprehensive_metrics(df, "NWS", "nws_prob")
    all_metrics.append(nws_metrics)
    print(f"  NWS:                 Brier={nws_metrics['overall_brier']:.4f}")

    # Original Model
    orig_metrics = compute_comprehensive_metrics(df, "Original_Model", "orig_prob")
    all_metrics.append(orig_metrics)
    print(f"  Original Model:      Brier={orig_metrics['overall_brier']:.4f}")

    # Kalshi Settled
    settled_metrics = compute_comprehensive_metrics(df, "Kalshi_Settled", "settled_market_prob")
    all_metrics.append(settled_metrics)
    print(f"  Kalshi Settled:      Brier={settled_metrics['overall_brier']:.4f}")

    # ---- Part 3: WGA-MDN Variants ----
    print("\n" + "=" * 70)
    print("PART 3: Evaluating WGA-MDN Variants (E34-E37)")
    print("=" * 70)

    # E34: Base WGA-MDN
    print("\n  E34: WGA Base...")
    df_e34 = apply_e34_wga_base(df)
    e34_metrics = compute_comprehensive_metrics(df_e34, "E34_wga_base")
    all_metrics.append(e34_metrics)
    df["e34_prob"] = df_e34["model_prob"]
    print(f"    Overall Brier={e34_metrics['overall_brier']:.4f}, "
          f"IS={e34_metrics['is_brier']:.4f}, OOS={e34_metrics['oos_brier']}")

    # E35: Synthesis stacker (train on 2023)
    print("\n  E35: WGA Synthesis Stacker...")
    cal_mask = df["has_wga"] & (df["date_dt"].dt.year == 2023)
    cal_df_2023 = df[cal_mask].copy()
    if len(cal_df_2023) > 100:
        e35_params = fit_e35_synthesis_stacker(cal_df_2023)
        df_e35 = apply_e35_synthesis_stacker(df, e35_params)
        e35_metrics = compute_comprehensive_metrics(df_e35, "E35_wga_synthesis_stacker")
        all_metrics.append(e35_metrics)
        df["e35_prob"] = df_e35["model_prob"]
        print(f"    Overall Brier={e35_metrics['overall_brier']:.4f}, "
              f"IS={e35_metrics['is_brier']:.4f}, OOS={e35_metrics['oos_brier']}")
        print(f"    Val Brier (fitting)={e35_params['validation_brier']:.4f}")
    else:
        print("    SKIPPED: Not enough calibration data")
        e35_params = None

    # E36: Contract-level Brier MLP (train on 2023)
    print("\n  E36: WGA Contract Brier MLP...")
    if len(cal_df_2023) > 200:
        e36_params = fit_e36_contract_brier_mlp(cal_df_2023)
        df_e36 = apply_e36_contract_brier_mlp(df, e36_params)
        e36_metrics = compute_comprehensive_metrics(df_e36, "E36_wga_contract_brier")
        all_metrics.append(e36_metrics)
        df["e36_prob"] = df_e36["model_prob"]
        print(f"    Overall Brier={e36_metrics['overall_brier']:.4f}, "
              f"IS={e36_metrics['is_brier']:.4f}, OOS={e36_metrics['oos_brier']}")
        print(f"    Val Score (fitting)={e36_params['validation_score']:.4f}")
    else:
        print("    SKIPPED: Not enough calibration data")
        e36_params = None

    # E37: Ensemble blend
    print("\n  E37: WGA Ensemble Blend...")

    # 50/50 blend
    df_e37_50 = apply_e37_ensemble_blend(df, weight_wga=0.5)
    e37_50_metrics = compute_comprehensive_metrics(df_e37_50, "E37_wga_blend_50_50")
    all_metrics.append(e37_50_metrics)
    df["e37_50_prob"] = df_e37_50["model_prob"]
    print(f"    50/50 blend: Brier={e37_50_metrics['overall_brier']:.4f}, "
          f"IS={e37_50_metrics['is_brier']:.4f}, OOS={e37_50_metrics['oos_brier']}")

    # Optimal weight (fit on 2023)
    e37_weights = fit_e37_optimal_weights(cal_df_2023)
    optimal_w = e37_weights["optimal_weight"]
    df_e37_opt = apply_e37_ensemble_blend(df, weight_wga=optimal_w)
    e37_opt_metrics = compute_comprehensive_metrics(df_e37_opt, "E37_wga_blend_optimal")
    all_metrics.append(e37_opt_metrics)
    df["e37_opt_prob"] = df_e37_opt["model_prob"]
    print(f"    Optimal blend (w={optimal_w:.2f}): Brier={e37_opt_metrics['overall_brier']:.4f}, "
          f"IS={e37_opt_metrics['is_brier']:.4f}, OOS={e37_opt_metrics['oos_brier']}")

    # ---- Calibration reliability data ----
    print("\n  Computing calibration reliability data...")
    cal_rows = []
    prob_sources = {
        "E34_wga_base": "e34_prob",
        "Original_Model": "orig_prob",
        "Kalshi_PreSettlement": "presettlement_prob",
        "NWS": "nws_prob",
        "Kalshi_Settled": "settled_market_prob",
    }
    if "e35_prob" in df.columns:
        prob_sources["E35_wga_synthesis_stacker"] = "e35_prob"
    if "e36_prob" in df.columns:
        prob_sources["E36_wga_contract_brier"] = "e36_prob"
    prob_sources["E37_wga_blend_50_50"] = "e37_50_prob"
    prob_sources["E37_wga_blend_optimal"] = "e37_opt_prob"

    for source_name, col in prob_sources.items():
        if col not in df.columns:
            continue
        probs = df[col].values.astype(float)
        rel = reliability_diagram_data(probs, outcomes)
        ece = expected_calibration_error(probs, outcomes)
        for row in rel:
            row["source"] = source_name
            row["ece"] = round(ece, 6)
            cal_rows.append(row)

    cal_reliability_df = pd.DataFrame(cal_rows)

    # ---- Part 4: Trading Simulation ----
    print("\n" + "=" * 70)
    print("PART 4: Trading Simulation")
    print("=" * 70)

    # Build prob columns for trading
    variant_prob_cols = {"E34_wga_base": "e34_prob"}
    if "e35_prob" in df.columns:
        variant_prob_cols["E35_wga_synthesis_stacker"] = "e35_prob"
    if "e36_prob" in df.columns:
        variant_prob_cols["E36_wga_contract_brier"] = "e36_prob"
    variant_prob_cols["E37_wga_blend_50_50"] = "e37_50_prob"
    variant_prob_cols["E37_wga_blend_optimal"] = "e37_opt_prob"
    variant_prob_cols["Original_Model"] = "orig_prob"
    variant_prob_cols["NWS"] = "nws_prob"

    trading_df = run_all_trading_sims(df, variant_prob_cols)

    # ---- Part 5: Output ----
    print("\n" + "=" * 70)
    print("PART 5: Saving Outputs")
    print("=" * 70)

    # Benchmark summary
    summary_rows = []
    for m in all_metrics:
        row = {
            "variant": m["variant"],
            "overall_brier": m.get("overall_brier"),
            "is_brier": m.get("is_brier"),
            "oos_brier": m.get("oos_brier"),
            "overall_log_score": m.get("overall_log_score"),
            "overall_ece": m.get("overall_ece"),
            "oos_ece": m.get("oos_ece"),
            "n_total": m.get("n_total"),
            "n_is": m.get("n_is"),
            "n_oos": m.get("n_oos"),
        }
        # Add season brier
        for s in ["Winter", "Spring", "Summer", "Fall"]:
            row[f"brier_{s.lower()}"] = m.get("season_brier", {}).get(s)
        # Add direction brier
        for d in ["above", "below", "between"]:
            row[f"brier_{d}"] = m.get("direction_brier", {}).get(d)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("overall_brier").reset_index(drop=True)
    summary_df.to_csv(OUT_ROOT / "benchmark_summary.csv", index=False)
    print(f"  Saved benchmark_summary.csv ({len(summary_df)} rows)")

    # Calibration reliability
    cal_reliability_df.to_csv(OUT_ROOT / "calibration_reliability.csv", index=False)
    print(f"  Saved calibration_reliability.csv ({len(cal_reliability_df)} rows)")

    # Trading simulation
    trading_df.to_csv(OUT_ROOT / "trading_simulation.csv", index=False)
    print(f"  Saved trading_simulation.csv ({len(trading_df)} rows)")

    # Full benchmark report
    report = generate_report(df, all_metrics, cal_reliability_df, trading_df, e37_weights)
    report_path = OUT_ROOT / "full_benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved full_benchmark_report.md")

    # ---- Print Summary ----
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(df):,} bucket observations across {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print(f"\n{'Variant':<35} {'Overall':>8} {'IS':>8} {'OOS':>8} {'ECE':>8}")
    print("-" * 75)
    for _, row in summary_df.iterrows():
        oos = f"{row['oos_brier']:.4f}" if pd.notna(row['oos_brier']) else "N/A"
        print(f"{row['variant']:<35} {row['overall_brier']:>8.4f} "
              f"{row['is_brier']:>8.4f} {oos:>8} {row['overall_ece']:>8.4f}")

    # Trading highlights
    if len(trading_df) > 0 and trading_df["n_trades"].sum() > 0:
        print("\n--- TRADING HIGHLIGHTS ---")
        for variant_base in ["E34_wga_base", "E35_wga_synthesis_stacker",
                              "E36_wga_contract_brier", "E37_wga_blend_optimal"]:
            all_rows = trading_df[trading_df["signal"].str.startswith(f"{variant_base}_All")]
            if len(all_rows) > 0:
                has_trades = all_rows[all_rows["n_trades"] > 0]
                if len(has_trades) > 0:
                    best = has_trades.loc[has_trades["net_pnl"].idxmax()]
                    print(f"\n  {variant_base} (best threshold={best['threshold']:.2f}):")
                    print(f"    Trades: {int(best['n_trades'])}, Win rate: {best['win_rate']:.1%}")
                    print(f"    Net P&L: ${best['net_pnl']:.2f}, ROI: {best['roi_pct']:.1f}%, "
                          f"Ann.Sharpe: {best['annualized_sharpe']:.3f}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {OUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
