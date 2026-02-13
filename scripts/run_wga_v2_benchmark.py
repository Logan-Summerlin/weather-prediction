#!/usr/bin/env python3
"""
run_wga_v2_benchmark.py

Evaluate Enhanced WGA-MDN V2 model predictions on the Kalshi prediction market
benchmark. Compares against Kalshi pre-settlement, NWS, original flat NN, and
Kalshi settled prices.

Implements variants across four WGA V2 architecture configs
(full, multihead_only, deep_only, lag2_only):

  E38_{config}_base:             Raw WGA V2 bucket probabilities (regime-conditional sigma)
  E39_{config}_synthesis:        Market-aware logistic regression synthesis stacker
  E40_{config}_contract_brier:   Contract-level Brier-optimal MLP

Plus cross-model blends:
  E41_wga_v2_flat_ensemble:      Optimal blend of best WGA V2 + original flat model
  E42_dual_attention_synthesis:   Contract-level MLP using features from BOTH WGA V2
                                  and original flat model simultaneously

Outputs to: results/prediction_market_benchmark/wga_v2_model/
"""

from __future__ import annotations

import importlib.util
import json
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

OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "wga_v2_model"
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999
FEE_RATE = 0.07
TRADING_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]
TRADING_DAYS_PER_YEAR = 252
N_CAL_BINS = 10
N_BOOTSTRAP = 1000
EV_QUALITY_CUTS = [0.02, 0.03, 0.04, 0.05, 0.06]

SEASON_MAP = bench.SEASON_MAP

# Architecture configs for WGA V2
WGA_V2_CONFIGS = ["wga_v2_full", "wga_v2_multihead_only",
                   "wga_v2_deep_only", "wga_v2_lag2_only"]


# ==============================================================================
# Part 1: Load and Prepare Data
# ==============================================================================

def load_wga_v2_predictions(config: str) -> pd.DataFrame | None:
    """Load WGA V2 predictions for a specific architecture config.

    Returns None if the file does not exist (config not yet trained).
    """
    path = ROOT / "results" / "wga_v2_model" / config / "predictions_test.csv"
    if not path.exists():
        # Fallback to old naming convention
        path = ROOT / "results" / "wga_v2_model" / f"predictions_test_{config}.csv"
        if not path.exists():
            print(f"  [WARN] WGA V2 predictions not found for {config}")
            return None
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

    Returns a contract-level DataFrame with columns for original model predictions,
    NWS, pre-settlement, and settled market probabilities.
    """
    print("=" * 70)
    print("PART 1: Loading and Preparing Data")
    print("=" * 70)

    # Load raw data
    pre = pd.read_csv(ROOT / "data" / "kalshi_presettlement.csv")
    s23 = pd.read_csv(ROOT / "data" / "real_kalshi_2023_2024.csv")
    s25 = pd.read_csv(ROOT / "data" / "real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)
    nws = pd.read_csv(
        ROOT / "results" / "prediction_market_benchmark" / "nws_probability_forecasts.csv"
    )

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
    print(f"  Pre+Settled merge: {n_before} -> {len(merged)} "
          f"(dropped {n_before - len(merged)} NaN presettlement)")

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
    print(f"  Original preds:    {len(orig_all)} dates "
          f"({orig_all['date'].min()} to {orig_all['date'].max()})")

    # Merge model predictions
    merged = merged.merge(orig_all, on="date", how="inner")
    merged = merged.merge(nws[["date", "nws_mu", "nws_sigma"]], on="date", how="inner")

    # Add period and season
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["period"] = np.where(merged["date_dt"].dt.year <= 2024, "IS", "OOS")
    merged["season"] = merged["date_dt"].dt.month.map(SEASON_MAP)
    merged["month"] = merged["date_dt"].dt.month

    print(f"\n  Final merged: {len(merged)} rows, {merged['date'].nunique()} dates")
    n_is = (merged["period"] == "IS").sum()
    n_oos = (merged["period"] == "OOS").sum()
    dates_is = merged[merged["period"] == "IS"]["date"].nunique()
    dates_oos = merged[merged["period"] == "OOS"]["date"].nunique()
    print(f"    IS:  {n_is} rows ({dates_is} dates)")
    print(f"    OOS: {n_oos} rows ({dates_oos} dates)")

    return merged


def attach_wga_v2_predictions(merged: pd.DataFrame, config: str,
                               wga_df: pd.DataFrame) -> pd.DataFrame:
    """Attach WGA V2 predictions for a specific config to the merged dataset.

    WGA V2 predictions are only available for IS period (2023-2024).
    For OOS (2025), we fall back to original flat model.
    """
    prefix = config.replace("wga_v2_", "v2_")

    # Select relevant columns from WGA V2 predictions
    cols_to_use = ["date"]
    col_map = {}
    # Map model_sigma_cal -> model_sigma_regime_conditional for downstream compatibility
    sigma_aliases = {
        "model_sigma_cal": "model_sigma_regime_conditional",
    }
    for col in ["model_mu", "model_sigma_regime_conditional", "model_sigma_monthly_cal",
                "model_sigma_regime_cal", "model_sigma_cal", "ensemble_std", "regime"]:
        if col in wga_df.columns:
            canonical = sigma_aliases.get(col, col)
            new_name = f"{prefix}_{canonical}"
            cols_to_use.append(col)
            col_map[col] = new_name

    wga_subset = wga_df[cols_to_use].copy().rename(columns=col_map)

    out = merged.merge(wga_subset, on="date", how="left")
    out[f"has_{prefix}"] = out[f"{prefix}_model_mu"].notna()

    return out


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


def compute_bucket_probs_from_arrays(mu, sigma, direction, th_low, th_high):
    """Compute bucket probabilities from numpy arrays directly."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    probs = np.full(len(mu), np.nan)

    below = direction == "below"
    above = direction == "above"
    between = direction == "between"

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


def brier_decomposition(probs, outcomes, n_bins=N_CAL_BINS):
    """Decompose Brier score into reliability, resolution, uncertainty."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n = len(probs)
    climo = float(np.mean(outcomes))
    uncertainty = climo * (1 - climo)

    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        nk = mask.sum()
        if nk > 0:
            fk = float(probs[mask].mean())
            ok = float(outcomes[mask].mean())
            reliability += (nk / n) * (fk - ok) ** 2
            resolution += (nk / n) * (ok - climo) ** 2

    return {
        "brier": round(float(np.mean((probs - outcomes) ** 2)), 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
    }


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


def _season_from_month(months):
    return np.where(np.isin(months, [12, 1, 2]), "DJF",
           np.where(np.isin(months, [3, 4, 5]), "MAM",
           np.where(np.isin(months, [6, 7, 8]), "JJA", "SON")))


def compute_comprehensive_metrics(df, variant, prob_col="model_prob"):
    """Compute all quality metrics for a benchmark variant."""
    probs = df[prob_col].values.astype(float)
    outcomes = df["actual_outcome"].values.astype(float)

    overall_bs = brier_score(probs, outcomes)
    overall_ls = log_score(probs, outcomes)
    overall_ece = expected_calibration_error(probs, outcomes)
    overall_decomp = brier_decomposition(probs, outcomes)

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
        "brier_decomposition": overall_decomp,
        "season_brier": {k: round(v, 6) for k, v in season_bs.items()},
        "direction_brier": {k: round(v, 6) for k, v in dir_bs.items()},
        "reliability_diagram": rel_data,
        "n_total": int(len(df)),
        "n_is": int(is_mask.sum()),
        "n_oos": int(oos_mask.sum()),
    }


# ==============================================================================
# Market State Features
# ==============================================================================

def build_market_state_features(frame, sigma_col="orig_sigma",
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
# Bucket-Specific Feature Builder
# ==============================================================================

def build_bucket_features(df, mu_vals, sig_vals, prob_vals):
    """Build standard bucket-specific features for contract-level models."""
    bucket_mid = np.where(
        df["direction"] == "above",
        df["threshold_low"].values + 2.0,
        np.where(
            df["direction"] == "below",
            df["threshold_high"].values - 2.0,
            (df["threshold_low"].values + df["threshold_high"].values) / 2.0,
        ),
    )
    bucket_quantile = norm.cdf(bucket_mid.astype(float), mu_vals, sig_vals)
    bucket_width = np.where(
        df["direction"] == "between",
        (df["threshold_high"].values - df["threshold_low"].values).astype(float) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )
    bucket_distance_sigma = np.abs(bucket_mid.astype(float) - mu_vals) / (sig_vals + 1e-6)
    direction_above = (df["direction"].values == "above").astype(float)
    direction_below = (df["direction"].values == "below").astype(float)

    # Neighboring bucket sum
    date_sum = pd.Series(prob_vals).groupby(df["date"].values).transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - prob_vals, 0.0, None)

    return {
        "bucket_quantile": bucket_quantile,
        "bucket_width": bucket_width,
        "bucket_distance_sigma": bucket_distance_sigma,
        "direction_above": direction_above,
        "direction_below": direction_below,
        "neighboring_bucket_sum": neighboring_bucket_sum,
    }


# ==============================================================================
# Part 2: E38 - Base WGA V2 Model
# ==============================================================================

def get_wga_v2_probs(df, config, sigma_type="regime_conditional"):
    """Compute WGA V2 bucket probabilities for a specific config.

    Uses WGA V2 mu/sigma for IS period (where available),
    falls back to original model for OOS period.

    Returns (probs, has_wga_mask).
    """
    prefix = config.replace("wga_v2_", "v2_")
    mu_col = f"{prefix}_model_mu"
    sigma_col = f"{prefix}_model_sigma_{sigma_type}"
    has_col = f"has_{prefix}"

    has_wga = df[has_col].values if has_col in df.columns else np.zeros(len(df), dtype=bool)

    probs = np.zeros(len(df))

    if has_wga.any():
        wga_mu = df.loc[has_wga, mu_col].values.astype(float)
        wga_sig = df.loc[has_wga, sigma_col].values.astype(float)
        direction = df.loc[has_wga, "direction"].values
        th_low = df.loc[has_wga, "threshold_low"].values.astype(float)
        th_high = df.loc[has_wga, "threshold_high"].values.astype(float)
        probs[has_wga] = compute_bucket_probs_from_arrays(
            wga_mu, wga_sig, direction, th_low, th_high
        )

    # OOS fallback to original model
    if (~has_wga).any():
        oos_probs = compute_bucket_probs(df[~has_wga], "orig_mu", "orig_sigma")
        probs[~has_wga] = oos_probs

    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX), has_wga


def apply_e38_wga_v2_base(df, config):
    """E38: Raw WGA V2 bucket probabilities, no post-processing."""
    out = df.copy()
    probs, _ = get_wga_v2_probs(df, config, sigma_type="regime_conditional")
    out["model_prob"] = probs
    return out


# ==============================================================================
# Part 3: E39 - Market-Aware Synthesis Stacker
# ==============================================================================

def fit_e39_synthesis_stacker(cal_df, config):
    """E39: Logistic regression meta-model trained on 2023 calibration year.

    Features: v2_prob, orig_prob, nws_prob, presettlement_prob, model-market
    differences, spread, sigma_norm, depth, staleness.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    frame = cal_df.copy()

    v2_prob, _ = get_wga_v2_probs(frame, config)
    orig_prob = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(frame, "nws_mu", "nws_sigma")

    for arr in [orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    frame["_v2_prob"] = v2_prob
    frame["_orig_prob"] = orig_prob
    frame["_nws_prob"] = nws_prob

    y = frame["actual_outcome"].values.astype(float)

    state = build_market_state_features(frame, sigma_col="orig_sigma")
    w = v2_prob
    o = orig_prob
    n = nws_prob
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


def apply_e39_synthesis_stacker(df, e39, config):
    """Apply E39 synthesis stacker to full dataset."""
    out = df.copy()

    v2_prob, _ = get_wga_v2_probs(out, config)
    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    for arr in [v2_prob, orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    state = build_market_state_features(out, sigma_col="orig_sigma",
                                         sigma_p05=e39["sigma_p05"],
                                         sigma_p95=e39["sigma_p95"])
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    x = np.column_stack([
        v2_prob, orig_prob, nws_prob, market_prob,
        v2_prob - market_prob, v2_prob - nws_prob,
        orig_prob - market_prob, orig_prob - nws_prob, nws_prob - market_prob,
        spread, s, depth, stale,
        (v2_prob - market_prob) * (1.0 - spread),
        (v2_prob - market_prob) * (1.0 - s),
        (v2_prob - nws_prob) * (1.0 - s),
    ])
    mu_x = np.array(e39["feature_mean"])
    sd_x = np.array(e39["feature_std"])
    z = (x - mu_x) / sd_x
    logits = z @ np.array(e39["coef"]) + float(e39["intercept"])
    pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e39["isotonic_x"])
    iso_y = np.array(e39["isotonic_y"])
    pred = np.interp(np.clip(pred, iso_x.min(), iso_x.max()), iso_x, iso_y)

    out["model_prob"] = np.clip(pred, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ==============================================================================
# Part 4: E40 - Contract-Level Brier-Optimal MLP
# ==============================================================================

def fit_e40_contract_brier_mlp(cal_df, config):
    """E40: Contract-level Brier-optimal MLP with WGA V2 specific features.

    60/20/20 train/val/cal split on 2023 calibration data.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neural_network import MLPClassifier

    frame = cal_df.copy()
    prefix = config.replace("wga_v2_", "v2_")

    v2_prob, _ = get_wga_v2_probs(frame, config)
    orig_prob = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(frame, "nws_mu", "nws_sigma")

    for arr in [orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    y = frame["actual_outcome"].values.astype(float)

    state = build_market_state_features(frame, sigma_col="orig_sigma")
    w = v2_prob
    o = orig_prob
    n = nws_prob
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # Use WGA V2 mu/sigma for bucket features where available, else original
    has_col = f"has_{prefix}"
    has_wga = frame[has_col].values if has_col in frame.columns else np.zeros(len(frame), dtype=bool)

    mu_col = f"{prefix}_model_mu"
    sig_col = f"{prefix}_model_sigma_regime_conditional"
    mu_vals = np.where(
        has_wga, frame[mu_col].fillna(0).values, frame["orig_mu"].values
    ).astype(float)
    sig_vals = np.where(
        has_wga, frame[sig_col].fillna(1).values, frame["orig_sigma"].values
    ).astype(float)

    bf = build_bucket_features(frame, mu_vals, sig_vals, w)

    # V2 vs original disagreement
    v2_orig_diff = w - o

    X = np.column_stack([
        w, o, n, k,
        w - k, w - n, o - k, n - k,
        spread, s, depth, stale,
        bf["bucket_quantile"], bf["bucket_width"], bf["bucket_distance_sigma"],
        bf["direction_above"], bf["direction_below"],
        bf["neighboring_bucket_sum"],
        v2_orig_diff,
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


def apply_e40_contract_brier_mlp(df, e40, config):
    """Apply E40 contract-level Brier MLP to full dataset."""
    out = df.copy()
    prefix = config.replace("wga_v2_", "v2_")

    v2_prob, has_wga = get_wga_v2_probs(out, config)
    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    for arr in [v2_prob, orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    state = build_market_state_features(out, sigma_col="orig_sigma",
                                         sigma_p05=e40["sigma_p05"],
                                         sigma_p95=e40["sigma_p95"])
    spread = state["spread"].values
    s = state["sigma_norm"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # Use WGA V2 mu/sigma where available, else original
    mu_col = f"{prefix}_model_mu"
    sig_col = f"{prefix}_model_sigma_regime_conditional"
    has_col = f"has_{prefix}"
    has_v2 = out[has_col].values if has_col in out.columns else np.zeros(len(out), dtype=bool)

    mu_vals = np.where(has_v2, out[mu_col].fillna(0).values, out["orig_mu"].values).astype(float)
    sig_vals = np.where(has_v2, out[sig_col].fillna(1).values, out["orig_sigma"].values).astype(float)

    bf = build_bucket_features(out, mu_vals, sig_vals, v2_prob)
    v2_orig_diff = v2_prob - orig_prob

    x = np.column_stack([
        v2_prob, orig_prob, nws_prob, market_prob,
        v2_prob - market_prob, v2_prob - nws_prob,
        orig_prob - market_prob, nws_prob - market_prob,
        spread, s, depth, stale,
        bf["bucket_quantile"], bf["bucket_width"], bf["bucket_distance_sigma"],
        bf["direction_above"], bf["direction_below"],
        bf["neighboring_bucket_sum"],
        v2_orig_diff,
    ])
    x = (x - np.array(e40["feature_mean"])) / np.array(e40["feature_std"])

    # MLP forward pass
    acts = x
    for i, (wt, b) in enumerate(zip(e40["coefs"], e40["intercepts"])):
        acts = acts @ np.array(wt) + np.array(b)
        if i < len(e40["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)  # ReLU
    raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e40["isotonic_x"])
    iso_y = np.array(e40["isotonic_y"])
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
# Part 5: E41 - WGA V2 + Flat Ensemble Blend
# ==============================================================================

def fit_e41_optimal_weights(cal_df, config):
    """Find optimal blend weight between best WGA V2 config and original model."""
    v2_prob, _ = get_wga_v2_probs(cal_df, config)
    orig_prob = compute_bucket_probs(cal_df, "orig_mu", "orig_sigma")
    np.clip(v2_prob, PROB_CLIP_MIN, PROB_CLIP_MAX, out=v2_prob)
    np.clip(orig_prob, PROB_CLIP_MIN, PROB_CLIP_MAX, out=orig_prob)

    outcomes = cal_df["actual_outcome"].values.astype(float)

    best_w = 0.5
    best_bs = float("inf")
    for w in np.linspace(0.0, 1.0, 41):
        blended = w * v2_prob + (1.0 - w) * orig_prob
        blended = np.clip(blended, PROB_CLIP_MIN, PROB_CLIP_MAX)
        bs = float(np.mean((blended - outcomes) ** 2))
        if bs < best_bs:
            best_bs = bs
            best_w = float(w)

    return {"optimal_weight": best_w, "calibration_brier": best_bs}


def apply_e41_ensemble_blend(df, config, weight_v2=0.5):
    """E41: Weighted blend of WGA V2 and original model."""
    out = df.copy()

    v2_prob, _ = get_wga_v2_probs(out, config)
    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")

    for arr in [v2_prob, orig_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    blended = weight_v2 * v2_prob + (1.0 - weight_v2) * orig_prob
    out["model_prob"] = np.clip(blended, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ==============================================================================
# Part 6: E42 - Dual Attention Synthesis
# ==============================================================================

def fit_e42_dual_attention_synthesis(cal_df, best_v2_config):
    """E42: Contract-level Brier-optimal MLP using features from BOTH
    WGA V2 and original flat model simultaneously.

    This is the key experiment: combining attention-based and flat representations.

    60/20/20 train/val/cal split on 2023 calibration data.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neural_network import MLPClassifier

    frame = cal_df.copy()
    prefix = best_v2_config.replace("wga_v2_", "v2_")

    # V2 model predictions
    v2_prob, _ = get_wga_v2_probs(frame, best_v2_config)

    # Original model predictions
    orig_prob = compute_bucket_probs(frame, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(frame, "nws_mu", "nws_sigma")

    for arr in [orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    y = frame["actual_outcome"].values.astype(float)

    state = build_market_state_features(frame, sigma_col="orig_sigma")
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # --- WGA V2 bucket features ---
    has_col = f"has_{prefix}"
    has_v2 = frame[has_col].values if has_col in frame.columns else np.zeros(len(frame), dtype=bool)
    mu_col = f"{prefix}_model_mu"
    sig_col = f"{prefix}_model_sigma_regime_conditional"
    v2_mu = np.where(has_v2, frame[mu_col].fillna(0).values, frame["orig_mu"].values).astype(float)
    v2_sig = np.where(has_v2, frame[sig_col].fillna(1).values, frame["orig_sigma"].values).astype(float)
    v2_bf = build_bucket_features(frame, v2_mu, v2_sig, v2_prob)

    # --- Original model bucket features ---
    orig_mu = frame["orig_mu"].values.astype(float)
    orig_sig = frame["orig_sigma"].values.astype(float)
    orig_bf = build_bucket_features(frame, orig_mu, orig_sig, orig_prob)

    # --- Cross-model disagreement features ---
    v2_flat_disagreement = np.abs(v2_prob - orig_prob)
    v2_flat_sigma_ratio = v2_sig / (orig_sig + 1e-6)

    # Feature matrix: 19+ features from each model + cross-model + market state
    X = np.column_stack([
        # V2 model features (7)
        v2_prob,
        v2_bf["bucket_quantile"],
        v2_bf["bucket_width"],
        v2_bf["bucket_distance_sigma"],
        v2_bf["direction_above"],
        v2_bf["direction_below"],
        v2_bf["neighboring_bucket_sum"],
        # Original model features (7)
        orig_prob,
        orig_bf["bucket_quantile"],
        orig_bf["bucket_width"],
        orig_bf["bucket_distance_sigma"],
        orig_bf["direction_above"],
        orig_bf["direction_below"],
        orig_bf["neighboring_bucket_sum"],
        # NWS + market (2)
        nws_prob, k,
        # Differences (6)
        v2_prob - k, v2_prob - nws_prob,
        orig_prob - k, orig_prob - nws_prob,
        v2_prob - orig_prob, nws_prob - k,
        # Market state (4)
        spread, s, depth, stale,
        # Cross-model features (2)
        v2_flat_disagreement,
        v2_flat_sigma_ratio,
        # Interactions (2)
        (v2_prob - k) * (1.0 - spread),
        v2_flat_disagreement * s,
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
        "best_v2_config": best_v2_config,
    }


def apply_e42_dual_attention_synthesis(df, e42):
    """Apply E42 dual attention synthesis MLP to full dataset."""
    out = df.copy()
    best_v2_config = e42["best_v2_config"]
    prefix = best_v2_config.replace("wga_v2_", "v2_")

    v2_prob, has_wga = get_wga_v2_probs(out, best_v2_config)
    orig_prob = compute_bucket_probs(out, "orig_mu", "orig_sigma")
    nws_prob = compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    for arr in [v2_prob, orig_prob, nws_prob]:
        np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX, out=arr)

    state = build_market_state_features(out, sigma_col="orig_sigma",
                                         sigma_p05=e42["sigma_p05"],
                                         sigma_p95=e42["sigma_p95"])
    spread = state["spread"].values
    s = state["sigma_norm"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # WGA V2 mu/sigma
    has_col = f"has_{prefix}"
    has_v2 = out[has_col].values if has_col in out.columns else np.zeros(len(out), dtype=bool)
    mu_col = f"{prefix}_model_mu"
    sig_col = f"{prefix}_model_sigma_regime_conditional"
    v2_mu = np.where(has_v2, out[mu_col].fillna(0).values, out["orig_mu"].values).astype(float)
    v2_sig = np.where(has_v2, out[sig_col].fillna(1).values, out["orig_sigma"].values).astype(float)
    v2_bf = build_bucket_features(out, v2_mu, v2_sig, v2_prob)

    # Original model bucket features
    orig_mu = out["orig_mu"].values.astype(float)
    orig_sig = out["orig_sigma"].values.astype(float)
    orig_bf = build_bucket_features(out, orig_mu, orig_sig, orig_prob)

    # Cross-model features
    v2_flat_disagreement = np.abs(v2_prob - orig_prob)
    v2_flat_sigma_ratio = v2_sig / (orig_sig + 1e-6)

    x = np.column_stack([
        # V2 model features (7)
        v2_prob,
        v2_bf["bucket_quantile"],
        v2_bf["bucket_width"],
        v2_bf["bucket_distance_sigma"],
        v2_bf["direction_above"],
        v2_bf["direction_below"],
        v2_bf["neighboring_bucket_sum"],
        # Original model features (7)
        orig_prob,
        orig_bf["bucket_quantile"],
        orig_bf["bucket_width"],
        orig_bf["bucket_distance_sigma"],
        orig_bf["direction_above"],
        orig_bf["direction_below"],
        orig_bf["neighboring_bucket_sum"],
        # NWS + market (2)
        nws_prob, market_prob,
        # Differences (6)
        v2_prob - market_prob, v2_prob - nws_prob,
        orig_prob - market_prob, orig_prob - nws_prob,
        v2_prob - orig_prob, nws_prob - market_prob,
        # Market state (4)
        spread, s, depth, stale,
        # Cross-model features (2)
        v2_flat_disagreement,
        v2_flat_sigma_ratio,
        # Interactions (2)
        (v2_prob - market_prob) * (1.0 - spread),
        v2_flat_disagreement * s,
    ])
    x = (x - np.array(e42["feature_mean"])) / np.array(e42["feature_std"])

    # MLP forward pass
    acts = x
    for i, (wt, b) in enumerate(zip(e42["coefs"], e42["intercepts"])):
        acts = acts @ np.array(wt) + np.array(b)
        if i < len(e42["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)  # ReLU
    raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e42["isotonic_x"])
    iso_y = np.array(e42["isotonic_y"])
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
# Trading Simulation
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

    daily_pnl = (pd.DataFrame({"date": df["date"].values, "pnl": pnl})
                 .groupby("date")["pnl"].sum().values)
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
                    ci_lo, ci_hi = bootstrap_pnl_ci(sub, prob_col, "presettlement_prob",
                                                     threshold)
                    r["pnl_ci95_low"] = round(ci_lo, 2)
                    r["pnl_ci95_high"] = round(ci_hi, 2)
                results.append(r)

    return pd.DataFrame(results)


# ==============================================================================
# EV-Aware Gating
# ==============================================================================

def compute_ev_gating(df, prob_col, variant_name):
    """Compute EV-aware quality gating for a variant.

    Dynamic threshold with spread + sigma + depth + staleness penalties.
    Bootstrap CIs (n=1000, date-block) for OOS.
    Seasonal stress slices.
    """
    results = []

    signal = df[prob_col].values
    market = df["presettlement_prob"].values
    outcome = df["actual_outcome"].values.astype(float)
    edge = np.abs(signal - market)

    # Market state for dynamic penalties
    spread = ((
        df["ask_cents"].fillna(df["presettlement_prob"] * 100)
        - df["bid_cents"].fillna(df["presettlement_prob"] * 100)
    ).clip(lower=0) / 100.0).values

    sigma = df["orig_sigma"].values.astype(float)
    sigma_norm = np.clip((sigma - np.nanpercentile(sigma, 5)) /
                         (np.nanpercentile(sigma, 95) - np.nanpercentile(sigma, 5) + 1e-6),
                         0.0, 1.0)

    volume = np.log1p(df["volume"].fillna(0.0).values)
    oi = np.log1p(df["open_interest"].fillna(0.0).values)
    depth = np.clip(
        0.6 * (volume / (np.nanpercentile(volume, 95) + 1e-6))
        + 0.4 * (oi / (np.nanpercentile(oi, 95) + 1e-6)),
        0.0, 1.0,
    )

    snapshot_dt = pd.to_datetime(df["snapshot_time_utc"], utc=True, errors="coerce")
    cutoff_dt = pd.to_datetime(df["date"], utc=True, errors="coerce") + pd.Timedelta(hours=5)
    staleness_hours = ((cutoff_dt - snapshot_dt).dt.total_seconds() / 3600.0).clip(lower=0.0)
    stale_norm = np.clip(staleness_hours.values / 8.0, 0.0, 1.0)

    # Dynamic threshold: base_cut + penalties
    for base_cut in EV_QUALITY_CUTS:
        dynamic_threshold = (
            base_cut
            + 0.02 * spread
            + 0.01 * sigma_norm
            + 0.015 * (1.0 - depth)
            + 0.01 * stale_norm
        )

        trade_mask = edge > dynamic_threshold
        n_trades = int(trade_mask.sum())

        if n_trades == 0:
            results.append({
                "variant": variant_name, "base_cut": base_cut,
                "n_trades": 0, "net_pnl": 0.0, "roi_pct": 0.0,
                "win_rate": 0.0, "sharpe": 0.0, "pnl_ci95_low": 0.0,
                "pnl_ci95_high": 0.0,
            })
            continue

        # Compute P&L for gated trades
        buy_yes = (signal - market > dynamic_threshold)
        buy_no = (market - signal > dynamic_threshold)

        ask = df["ask_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0
        bid = df["bid_cents"].fillna(df["presettlement_prob"] * 100).values / 100.0

        pnl = np.zeros(len(df), dtype=float)
        if buy_yes.any():
            pnl[buy_yes] = np.where(outcome[buy_yes] == 1, 1.0 - FEE_RATE, 0.0) - ask[buy_yes]
        if buy_no.any():
            pnl[buy_no] = np.where(outcome[buy_no] == 0, 1.0 - FEE_RATE, 0.0) - (1.0 - bid[buy_no])

        gated_pnl = pnl[buy_yes | buy_no]
        net_pnl = float(gated_pnl.sum())
        gated_cost = np.zeros(len(df), dtype=float)
        gated_cost[buy_yes] = ask[buy_yes]
        gated_cost[buy_no] = 1.0 - bid[buy_no]
        total_cost = float(gated_cost[buy_yes | buy_no].sum())
        roi = (net_pnl / total_cost * 100) if total_cost > 0 else 0.0
        win_mask = np.concatenate([outcome[buy_yes] == 1, outcome[buy_no] == 0])
        win_rate = float(win_mask.mean()) if len(win_mask) > 0 else 0.0
        sharpe = float(gated_pnl.mean() / gated_pnl.std()) if gated_pnl.std() > 0 else 0.0

        # Bootstrap CI on date-block PnL
        daily_pnl = (pd.DataFrame({"date": df["date"].values, "pnl": pnl})
                     .groupby("date")["pnl"].sum().values)
        daily_pnl = daily_pnl[daily_pnl != 0]  # only days with trades
        if len(daily_pnl) >= 2:
            rng = np.random.default_rng(42)
            sampled_sums = [rng.choice(daily_pnl, size=len(daily_pnl), replace=True).sum()
                            for _ in range(N_BOOTSTRAP)]
            ci_lo = float(np.percentile(sampled_sums, 2.5))
            ci_hi = float(np.percentile(sampled_sums, 97.5))
        else:
            ci_lo, ci_hi = 0.0, 0.0

        results.append({
            "variant": variant_name, "base_cut": base_cut,
            "n_trades": n_trades, "net_pnl": round(net_pnl, 2),
            "roi_pct": round(roi, 2), "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "pnl_ci95_low": round(ci_lo, 2), "pnl_ci95_high": round(ci_hi, 2),
        })

    return results


def compute_seasonal_stress_slices(df, prob_col, variant_name):
    """Compute OOS seasonal stress slices (DJF, MAM, JJA, SON, volatile)."""
    oos = df[df["period"] == "OOS"].copy()
    if len(oos) == 0:
        return []

    probs = oos[prob_col].values.astype(float)
    outcomes = oos["actual_outcome"].values.astype(float)
    months = oos["date_dt"].dt.month.values
    seasons_djf = _season_from_month(months)

    results = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons_djf == season
        if mask.sum() > 0:
            results.append({
                "variant": variant_name,
                "slice": season,
                "n": int(mask.sum()),
                "brier": round(brier_score(probs[mask], outcomes[mask]), 6),
                "ece": round(expected_calibration_error(probs[mask], outcomes[mask]), 6),
            })

    # Volatile slice: days where actual temp deviated > 10F from model mu
    if "orig_mu" in oos.columns:
        actual = oos["actual_tmax"].values.astype(float)
        model_mu = oos["orig_mu"].values.astype(float)
        # Map date-level volatility to contract level
        date_volatile = {}
        for date_val in oos["date"].unique():
            dm = oos["date"].values == date_val
            if dm.any():
                dev = np.abs(actual[dm][0] - model_mu[dm][0])
                date_volatile[date_val] = dev > 10.0

        volatile_mask = np.array([date_volatile.get(d, False) for d in oos["date"].values])
        if volatile_mask.sum() > 0:
            results.append({
                "variant": variant_name,
                "slice": "volatile",
                "n": int(volatile_mask.sum()),
                "brier": round(brier_score(probs[volatile_mask], outcomes[volatile_mask]), 6),
                "ece": round(expected_calibration_error(probs[volatile_mask], outcomes[volatile_mask]), 6),
            })

    return results


# ==============================================================================
# Paper-Trading Gate Evaluation
# ==============================================================================

def evaluate_paper_trading_gate(df, all_metrics, variant_prob_cols, ev_results):
    """Evaluate paper-trading promotion gates for each variant.

    Gates:
    1. OOS Brier <= PreSettlement OOS Brier
    2. OOS gated P&L positive with positive CI lower bound
    3. ECE <= 0.03
    4. Tail reliability <= 0.20
    """
    # Compute PreSettlement OOS Brier as reference
    oos_mask = df["period"] == "OOS"
    oos_df = df[oos_mask]
    pre_oos_brier = brier_score(
        oos_df["presettlement_prob"].values.astype(float),
        oos_df["actual_outcome"].values.astype(float),
    ) if len(oos_df) > 0 else None

    gate_report = {
        "presettlement_oos_brier": round(pre_oos_brier, 6) if pre_oos_brier is not None else None,
        "variants": {},
    }

    for m in all_metrics:
        variant = m["variant"]
        oos_brier = m.get("oos_brier")
        oos_ece = m.get("oos_ece")

        # Gate 1: OOS Brier <= PreSettlement
        gate1 = (oos_brier is not None and pre_oos_brier is not None
                 and oos_brier <= pre_oos_brier)

        # Gate 3: ECE <= 0.03
        gate3 = oos_ece is not None and oos_ece <= 0.03

        # Gate 4: Tail reliability
        # Compute tail reliability: reliability for probs > 0.4
        prob_col_name = None
        for vn, pc in variant_prob_cols.items():
            if vn == variant:
                prob_col_name = pc
                break

        gate4 = False
        tail_reliability = None
        if prob_col_name is not None and prob_col_name in df.columns:
            oos_probs = oos_df[prob_col_name].values.astype(float)
            oos_outcomes = oos_df["actual_outcome"].values.astype(float)
            tail_mask = oos_probs > 0.4
            if tail_mask.sum() > 10:
                tail_reliability = float(np.mean(np.abs(
                    oos_probs[tail_mask] - oos_outcomes[tail_mask]
                )))
                gate4 = tail_reliability <= 0.20

        # Gate 2: OOS gated P&L positive with positive CI
        gate2 = False
        best_gated_pnl = 0.0
        best_ci_low = 0.0
        for ev_row in ev_results:
            if ev_row.get("variant") == variant:
                # Check if any EV gating config yields positive P&L with positive CI
                if ev_row.get("net_pnl", 0) > 0 and ev_row.get("pnl_ci95_low", 0) > 0:
                    gate2 = True
                    if ev_row["net_pnl"] > best_gated_pnl:
                        best_gated_pnl = ev_row["net_pnl"]
                        best_ci_low = ev_row.get("pnl_ci95_low", 0.0)

        all_pass = gate1 and gate2 and gate3 and gate4

        gate_report["variants"][variant] = {
            "oos_brier": round(oos_brier, 6) if oos_brier is not None else None,
            "oos_ece": round(oos_ece, 6) if oos_ece is not None else None,
            "tail_reliability": round(tail_reliability, 6) if tail_reliability is not None else None,
            "best_gated_pnl": round(best_gated_pnl, 2),
            "best_ci_low": round(best_ci_low, 2),
            "gates": {
                "oos_brier_beats_presettlement": gate1,
                "oos_gated_pnl_positive_ci": gate2,
                "ece_below_0.03": gate3,
                "tail_reliability_below_0.20": gate4,
            },
            "all_gates_pass": all_pass,
        }

    return gate_report


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(df, all_metrics, trading_df, ev_results_all,
                    gate_report, e41_weights, best_v2_config):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# WGA-MDN V2 Enhanced Model: Prediction Market Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report evaluates the Enhanced WGA-MDN V2 (Wind-Gated Attention Mixture")
    lines.append("Density Network V2) model variants on the Kalshi NYC temperature prediction market.")
    lines.append("")
    lines.append("### Architecture Configs Tested")
    lines.append("")
    lines.append("| Config | Description |")
    lines.append("|--------|-------------|")
    lines.append("| wga_v2_full | Full enhanced model (multi-head + deep + lag2) |")
    lines.append("| wga_v2_multihead_only | Multi-head attention only |")
    lines.append("| wga_v2_deep_only | Deep MLP backbone only |")
    lines.append("| wga_v2_lag2_only | Lag-2 temporal features only |")
    lines.append("")
    lines.append("### Variant Definitions")
    lines.append("")
    lines.append("| Variant | Description |")
    lines.append("|---------|-------------|")
    lines.append("| E38_{config}_base | Raw WGA V2 bucket probabilities (regime-conditional sigma) |")
    lines.append("| E39_{config}_synthesis | Market-aware logistic regression synthesis stacker |")
    lines.append("| E40_{config}_contract_brier | Contract-level Brier-optimal MLP |")
    lines.append("| E41_wga_v2_flat_ensemble | Optimally weighted WGA V2 + flat model blend |")
    lines.append("| E42_dual_attention_synthesis | Dual-model contract-level MLP (key experiment) |")
    lines.append("")
    lines.append(f"- Total bucket-level observations: **{len(df):,}**")
    lines.append(f"- Unique dates: **{df['date'].nunique()}**")
    lines.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    n_is = (df["period"] == "IS").sum()
    n_oos = (df["period"] == "OOS").sum()
    lines.append(f"- IS period (2023-2024): {n_is:,} rows")
    lines.append(f"- OOS period (2025): {n_oos:,} rows")
    lines.append("")

    # --- Brier Scores ---
    lines.append("## 1. Brier Score Comparison (lower = better)")
    lines.append("")
    lines.append("### Overall and By Period")
    lines.append("")
    lines.append("| Variant | Overall | IS | OOS | LogScore | ECE | OOS ECE |")
    lines.append("|---------|---------|-----|-----|----------|-----|---------|")
    sorted_metrics = sorted(all_metrics, key=lambda m: m.get("overall_brier", 999))
    for m in sorted_metrics:
        v = m["variant"]
        obs = m.get("overall_brier")
        isb = m.get("is_brier")
        oosb = m.get("oos_brier")
        ols = m.get("overall_log_score")
        oece = m.get("overall_ece")
        oos_ece = m.get("oos_ece")
        fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
        lines.append(f"| {v} | {fmt(obs)} | {fmt(isb)} | {fmt(oosb)} | "
                     f"{fmt(ols)} | {fmt(oece)} | {fmt(oos_ece)} |")
    lines.append("")

    # By season
    lines.append("### By Season (Overall Brier)")
    lines.append("")
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    header = "| Variant | " + " | ".join(seasons) + " |"
    lines.append(header)
    lines.append("|---------|" + "|".join(["--------" for _ in seasons]) + "|")
    for m in sorted_metrics[:10]:  # top 10 only
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
    lines.append("|---------|" + "|".join(["--------" for _ in directions]) + "|")
    for m in sorted_metrics[:10]:
        v = m["variant"]
        db = m.get("direction_brier", {})
        vals = " | ".join([f"{db.get(d, 'N/A')}" for d in directions])
        lines.append(f"| {v} | {vals} |")
    lines.append("")

    # Brier Decomposition
    lines.append("### Brier Decomposition (Top 5)")
    lines.append("")
    lines.append("| Variant | Brier | Reliability | Resolution | Uncertainty |")
    lines.append("|---------|-------|------------|------------|-------------|")
    for m in sorted_metrics[:5]:
        decomp = m.get("brier_decomposition", {})
        lines.append(f"| {m['variant']} | {decomp.get('brier', 'N/A')} | "
                     f"{decomp.get('reliability', 'N/A')} | "
                     f"{decomp.get('resolution', 'N/A')} | "
                     f"{decomp.get('uncertainty', 'N/A')} |")
    lines.append("")

    # --- Trading Simulation ---
    lines.append("## 2. Trading Simulation")
    lines.append("")
    lines.append(f"Fee rate: {FEE_RATE*100:.0f}% on winnings")
    lines.append("")

    if trading_df is not None and len(trading_df) > 0:
        seen_variants = set()
        for sig in sorted(trading_df["signal"].unique()):
            for suffix in ["_All", "_IS", "_OOS"]:
                if sig.endswith(suffix):
                    variant_base = sig[:-len(suffix)]
                    if variant_base not in seen_variants:
                        seen_variants.add(variant_base)
                        lines.append(f"### {variant_base}")
                        lines.append("")
                        lines.append("| Period | Threshold | Trades | Win Rate | "
                                     "Net P&L | ROI% | Ann. Sharpe | CI Low | CI High |")
                        lines.append("|--------|-----------|--------|----------|"
                                     "---------|------|-------------|--------|---------|")
                        for period_suffix in ["_All", "_IS", "_OOS"]:
                            period_label = period_suffix[1:]
                            period_rows = trading_df[
                                trading_df["signal"] == f"{variant_base}{period_suffix}"
                            ]
                            for _, row in period_rows.iterrows():
                                ci_lo = row.get("pnl_ci95_low", "")
                                ci_hi = row.get("pnl_ci95_high", "")
                                ci_lo_str = f"${ci_lo:.2f}" if pd.notna(ci_lo) and ci_lo != "" else ""
                                ci_hi_str = f"${ci_hi:.2f}" if pd.notna(ci_hi) and ci_hi != "" else ""
                                lines.append(
                                    f"| {period_label} | {row['threshold']:.2f} | "
                                    f"{row['n_trades']} | {row['win_rate']:.1%} | "
                                    f"${row['net_pnl']:.2f} | {row['roi_pct']:.1f}% | "
                                    f"{row['annualized_sharpe']:.3f} | "
                                    f"{ci_lo_str} | {ci_hi_str} |"
                                )
                        lines.append("")
                    break

    # --- EV Gating ---
    lines.append("## 3. EV-Aware Quality Gating")
    lines.append("")
    if ev_results_all:
        ev_df = pd.DataFrame(ev_results_all)
        for variant in ev_df["variant"].unique():
            vd = ev_df[ev_df["variant"] == variant]
            lines.append(f"### {variant}")
            lines.append("")
            lines.append("| Base Cut | Trades | Net P&L | ROI% | Win Rate | "
                         "Sharpe | CI Low | CI High |")
            lines.append("|----------|--------|---------|------|----------|"
                         "--------|--------|---------|")
            for _, row in vd.iterrows():
                lines.append(
                    f"| {row['base_cut']:.2f} | {row['n_trades']} | "
                    f"${row['net_pnl']:.2f} | {row['roi_pct']:.1f}% | "
                    f"{row['win_rate']:.1%} | {row['sharpe']:.4f} | "
                    f"${row['pnl_ci95_low']:.2f} | ${row['pnl_ci95_high']:.2f} |"
                )
            lines.append("")

    # --- Paper Trading Gate ---
    lines.append("## 4. Paper-Trading Promotion Gate")
    lines.append("")
    lines.append(f"PreSettlement OOS Brier: {gate_report.get('presettlement_oos_brier', 'N/A')}")
    lines.append("")
    lines.append("| Variant | OOS Brier <= Pre | Gated P&L Pos CI | ECE <= 0.03 | "
                 "Tail Rel <= 0.20 | ALL PASS |")
    lines.append("|---------|-----------------|------------------|------------|"
                 "-----------------|----------|")
    for variant, info in gate_report.get("variants", {}).items():
        gates = info.get("gates", {})
        check = lambda b: "PASS" if b else "FAIL"
        lines.append(
            f"| {variant} | {check(gates.get('oos_brier_beats_presettlement'))} | "
            f"{check(gates.get('oos_gated_pnl_positive_ci'))} | "
            f"{check(gates.get('ece_below_0.03'))} | "
            f"{check(gates.get('tail_reliability_below_0.20'))} | "
            f"**{check(info.get('all_gates_pass'))}** |"
        )
    lines.append("")

    # --- Key Findings ---
    lines.append("## 5. Key Findings")
    lines.append("")

    best = sorted_metrics[0]
    lines.append(f"- **Best overall Brier**: {best['variant']} ({best['overall_brier']:.4f})")

    # E41 weights
    if e41_weights:
        lines.append(f"- E41 optimal V2 weight: {e41_weights.get('optimal_weight', 'N/A'):.2f}")
        lines.append("  (1.0 = pure WGA V2, 0.0 = pure original model)")

    # V2 vs original V1 comparison
    e38_variants = [m for m in all_metrics if m["variant"].startswith("E38_")]
    orig_m = next((m for m in all_metrics if m["variant"] == "Original_Model"), None)
    pre_m = next((m for m in all_metrics if m["variant"] == "Kalshi_PreSettlement"), None)

    if e38_variants and orig_m:
        best_e38 = min(e38_variants, key=lambda m: m.get("overall_brier", 999))
        diff = best_e38["overall_brier"] - orig_m["overall_brier"]
        if diff < 0:
            lines.append(f"- Best E38 ({best_e38['variant']}) BEATS Original by "
                         f"{abs(diff):.4f} Brier points")
        else:
            lines.append(f"- Original BEATS best E38 ({best_e38['variant']}) by "
                         f"{abs(diff):.4f} Brier points")

    if e38_variants and pre_m:
        best_e38 = min(e38_variants, key=lambda m: m.get("overall_brier", 999))
        diff = best_e38["overall_brier"] - pre_m["overall_brier"]
        if diff < 0:
            lines.append(f"- Best E38 BEATS PreSettlement by {abs(diff):.4f}")
        else:
            lines.append(f"- PreSettlement BEATS best E38 by {abs(diff):.4f}")

    # E42 dual attention vs E36 original WGA
    e42_m = next((m for m in all_metrics if m["variant"] == "E42_dual_attention_synthesis"), None)
    if e42_m:
        lines.append(f"- E42 Dual Attention Synthesis: Overall={e42_m['overall_brier']:.4f}, "
                     f"OOS={e42_m.get('oos_brier', 'N/A')}")

    # Best trading
    if trading_df is not None and len(trading_df) > 0 and trading_df["n_trades"].sum() > 0:
        best_trade = trading_df.loc[trading_df["net_pnl"].idxmax()]
        lines.append(f"- Best trading: {best_trade['signal']}, "
                     f"threshold={best_trade['threshold']:.2f}, "
                     f"P&L=${best_trade['net_pnl']:.2f}, "
                     f"ROI={best_trade['roi_pct']:.1f}%, "
                     f"{int(best_trade['n_trades'])} trades")

    # Promotion gate summary
    promoted = [v for v, info in gate_report.get("variants", {}).items()
                if info.get("all_gates_pass")]
    if promoted:
        lines.append(f"- **Variants passing all promotion gates**: {', '.join(promoted)}")
    else:
        lines.append("- No variants pass all promotion gates")

    lines.append("")
    lines.append(f"- Best WGA V2 config: **{best_v2_config}**")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    np.random.seed(42)

    # ---- Part 1: Load and prepare data ----
    df = build_merged_dataset()

    # ---- Load WGA V2 predictions for each config ----
    print("\n" + "=" * 70)
    print("Loading WGA V2 Predictions")
    print("=" * 70)

    available_configs = []
    for config in WGA_V2_CONFIGS:
        wga_df = load_wga_v2_predictions(config)
        if wga_df is not None:
            df = attach_wga_v2_predictions(df, config, wga_df)
            available_configs.append(config)
            prefix = config.replace("wga_v2_", "v2_")
            n_avail = df[f"has_{prefix}"].sum()
            print(f"  {config}: {n_avail} rows with predictions")

    if not available_configs:
        print("\n  [ERROR] No WGA V2 prediction files found. Cannot proceed.")
        print("  Expected files in results/wga_v2_model/predictions_test_{config}.csv")
        print("  Run the WGA V2 training script first.")
        return

    print(f"\n  Available configs: {available_configs}")

    # ---- Part 2: Compute baseline benchmark probabilities ----
    print("\n" + "=" * 70)
    print("PART 2: Computing Baseline Benchmark Probabilities")
    print("=" * 70)

    df["orig_prob"] = compute_bucket_probs(df, "orig_mu", "orig_sigma")
    df["nws_prob"] = compute_bucket_probs(df, "nws_mu", "nws_sigma")

    for col in ["orig_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        df[col] = df[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

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

    # ---- Part 3: Per-config WGA V2 variants (E38-E40) ----
    print("\n" + "=" * 70)
    print("PART 3: Evaluating WGA V2 Per-Config Variants (E38-E40)")
    print("=" * 70)

    # Track per-config IS Brier to select best config for cross-model blends
    config_is_brier = {}
    variant_prob_cols = {}  # map variant_name -> prob_col in df

    # Calibration data: 2023 IS rows with WGA V2 predictions
    cal_year_mask = df["date_dt"].dt.year == 2023

    for config in available_configs:
        prefix = config.replace("wga_v2_", "v2_")
        has_col = f"has_{prefix}"
        config_label = config.replace("wga_v2_", "")

        print(f"\n  --- Config: {config} ---")

        # E38: Base WGA V2
        variant_name = f"E38_{config_label}_base"
        print(f"    E38: {variant_name}")
        try:
            df_e38 = apply_e38_wga_v2_base(df, config)
            prob_col = f"e38_{config_label}_prob"
            df[prob_col] = df_e38["model_prob"]
            e38_metrics = compute_comprehensive_metrics(df, variant_name, prob_col)
            all_metrics.append(e38_metrics)
            variant_prob_cols[variant_name] = prob_col
            config_is_brier[config] = e38_metrics.get("is_brier", 999)
            print(f"      Brier: overall={e38_metrics['overall_brier']:.4f}, "
                  f"IS={e38_metrics['is_brier']:.4f}, OOS={e38_metrics['oos_brier']}")
        except Exception as exc:
            print(f"      [WARN] E38 failed for {config}: {exc}")

        # E39: Synthesis stacker (train on 2023)
        variant_name = f"E39_{config_label}_synthesis"
        print(f"    E39: {variant_name}")
        cal_mask = cal_year_mask & df[has_col]
        cal_df_2023 = df[cal_mask].copy()
        if len(cal_df_2023) > 100:
            try:
                e39_params = fit_e39_synthesis_stacker(cal_df_2023, config)
                df_e39 = apply_e39_synthesis_stacker(df, e39_params, config)
                prob_col = f"e39_{config_label}_prob"
                df[prob_col] = df_e39["model_prob"]
                e39_metrics = compute_comprehensive_metrics(df, variant_name, prob_col)
                all_metrics.append(e39_metrics)
                variant_prob_cols[variant_name] = prob_col
                print(f"      Brier: overall={e39_metrics['overall_brier']:.4f}, "
                      f"IS={e39_metrics['is_brier']:.4f}, OOS={e39_metrics['oos_brier']}")
                print(f"      Val Brier (fitting)={e39_params['validation_brier']:.4f}")
            except Exception as exc:
                print(f"      [WARN] E39 failed for {config}: {exc}")
        else:
            print(f"      SKIPPED: insufficient calibration data ({len(cal_df_2023)} rows)")

        # E40: Contract-level Brier MLP (train on 2023)
        variant_name = f"E40_{config_label}_contract_brier"
        print(f"    E40: {variant_name}")
        if len(cal_df_2023) > 200:
            try:
                e40_params = fit_e40_contract_brier_mlp(cal_df_2023, config)
                df_e40 = apply_e40_contract_brier_mlp(df, e40_params, config)
                prob_col = f"e40_{config_label}_prob"
                df[prob_col] = df_e40["model_prob"]
                e40_metrics = compute_comprehensive_metrics(df, variant_name, prob_col)
                all_metrics.append(e40_metrics)
                variant_prob_cols[variant_name] = prob_col
                print(f"      Brier: overall={e40_metrics['overall_brier']:.4f}, "
                      f"IS={e40_metrics['is_brier']:.4f}, OOS={e40_metrics['oos_brier']}")
                print(f"      Val Score (fitting)={e40_params['validation_score']:.4f}")
            except Exception as exc:
                print(f"      [WARN] E40 failed for {config}: {exc}")
        else:
            print(f"      SKIPPED: insufficient calibration data ({len(cal_df_2023)} rows)")

    # ---- Part 4: Cross-model blends (E41-E42) ----
    print("\n" + "=" * 70)
    print("PART 4: Cross-Model Blends (E41-E42)")
    print("=" * 70)

    # Select best WGA V2 config by IS Brier
    if config_is_brier:
        best_v2_config = min(config_is_brier, key=config_is_brier.get)
    else:
        best_v2_config = available_configs[0]
    print(f"\n  Best V2 config (by IS Brier): {best_v2_config}")

    best_prefix = best_v2_config.replace("wga_v2_", "v2_")
    best_has_col = f"has_{best_prefix}"

    # E41: WGA V2 + Flat ensemble blend
    print("\n  E41: WGA V2 + Flat Ensemble Blend")
    e41_weights = None
    cal_mask_e41 = cal_year_mask & df[best_has_col]
    cal_df_e41 = df[cal_mask_e41].copy()
    if len(cal_df_e41) > 50:
        try:
            e41_weights = fit_e41_optimal_weights(cal_df_e41, best_v2_config)
            optimal_w = e41_weights["optimal_weight"]
            df_e41 = apply_e41_ensemble_blend(df, best_v2_config, weight_v2=optimal_w)
            prob_col = "e41_prob"
            df[prob_col] = df_e41["model_prob"]
            e41_metrics = compute_comprehensive_metrics(
                df, "E41_wga_v2_flat_ensemble", prob_col
            )
            all_metrics.append(e41_metrics)
            variant_prob_cols["E41_wga_v2_flat_ensemble"] = prob_col
            print(f"    Optimal weight (V2): {optimal_w:.2f}")
            print(f"    Brier: overall={e41_metrics['overall_brier']:.4f}, "
                  f"IS={e41_metrics['is_brier']:.4f}, OOS={e41_metrics['oos_brier']}")
        except Exception as exc:
            print(f"    [WARN] E41 failed: {exc}")
    else:
        print("    SKIPPED: insufficient calibration data")

    # E42: Dual attention synthesis
    print("\n  E42: Dual Attention Synthesis")
    if len(cal_df_e41) > 200:
        try:
            e42_params = fit_e42_dual_attention_synthesis(cal_df_e41, best_v2_config)
            df_e42 = apply_e42_dual_attention_synthesis(df, e42_params)
            prob_col = "e42_prob"
            df[prob_col] = df_e42["model_prob"]
            e42_metrics = compute_comprehensive_metrics(
                df, "E42_dual_attention_synthesis", prob_col
            )
            all_metrics.append(e42_metrics)
            variant_prob_cols["E42_dual_attention_synthesis"] = prob_col
            print(f"    Brier: overall={e42_metrics['overall_brier']:.4f}, "
                  f"IS={e42_metrics['is_brier']:.4f}, OOS={e42_metrics['oos_brier']}")
            print(f"    Val Score (fitting)={e42_params['validation_score']:.4f}")
        except Exception as exc:
            print(f"    [WARN] E42 failed: {exc}")
    else:
        print("    SKIPPED: insufficient calibration data")

    # Add baseline prob cols to variant_prob_cols for trading
    variant_prob_cols["Original_Model"] = "orig_prob"
    variant_prob_cols["NWS"] = "nws_prob"

    # ---- Part 5: Trading Simulation ----
    print("\n" + "=" * 70)
    print("PART 5: Trading Simulation")
    print("=" * 70)

    trading_df = run_all_trading_sims(df, variant_prob_cols)

    # ---- Part 6: EV-Aware Gating ----
    print("\n" + "=" * 70)
    print("PART 6: EV-Aware Quality Gating")
    print("=" * 70)

    ev_results_all = []
    seasonal_stress_all = []

    # Run EV gating on OOS period for each variant
    oos_df = df[df["period"] == "OOS"]
    for variant_name, prob_col in variant_prob_cols.items():
        if prob_col not in df.columns or variant_name in ["Original_Model", "NWS"]:
            continue
        if oos_df[prob_col].isna().all():
            continue

        print(f"  EV gating: {variant_name}...")
        try:
            ev_results = compute_ev_gating(oos_df, prob_col, variant_name)
            ev_results_all.extend(ev_results)

            # Seasonal stress slices
            stress_results = compute_seasonal_stress_slices(df, prob_col, variant_name)
            seasonal_stress_all.extend(stress_results)
        except Exception as exc:
            print(f"    [WARN] EV gating failed for {variant_name}: {exc}")

    # Save per-variant EV gating results
    for variant_name in variant_prob_cols:
        variant_ev = [r for r in ev_results_all if r.get("variant") == variant_name]
        if variant_ev:
            ev_df = pd.DataFrame(variant_ev)
            safe_name = variant_name.replace("/", "_").replace(" ", "_")
            ev_df.to_csv(OUT_ROOT / f"ev_edge_quality_gating_results_{safe_name}.csv",
                         index=False)

    # ---- Part 7: Paper-Trading Gate ----
    print("\n" + "=" * 70)
    print("PART 7: Paper-Trading Promotion Gate")
    print("=" * 70)

    gate_report = evaluate_paper_trading_gate(df, all_metrics, variant_prob_cols, ev_results_all)

    for variant, info in gate_report.get("variants", {}).items():
        gates = info.get("gates", {})
        passed = sum(1 for v in gates.values() if v)
        total = len(gates)
        status = "ALL PASS" if info.get("all_gates_pass") else f"{passed}/{total}"
        print(f"  {variant}: {status}")

    # ---- Part 8: Save Outputs ----
    print("\n" + "=" * 70)
    print("PART 8: Saving Outputs")
    print("=" * 70)

    # Benchmark summary CSV
    summary_rows = []
    for m in all_metrics:
        row = {
            "variant": m["variant"],
            "overall_brier": m.get("overall_brier"),
            "is_brier": m.get("is_brier"),
            "oos_brier": m.get("oos_brier"),
            "overall_log_score": m.get("overall_log_score"),
            "is_log_score": m.get("is_log_score"),
            "oos_log_score": m.get("oos_log_score"),
            "overall_ece": m.get("overall_ece"),
            "oos_ece": m.get("oos_ece"),
            "n_total": m.get("n_total"),
            "n_is": m.get("n_is"),
            "n_oos": m.get("n_oos"),
        }
        decomp = m.get("brier_decomposition", {})
        row["brier_reliability"] = decomp.get("reliability")
        row["brier_resolution"] = decomp.get("resolution")
        row["brier_uncertainty"] = decomp.get("uncertainty")
        for s in ["Winter", "Spring", "Summer", "Fall"]:
            row[f"brier_{s.lower()}"] = m.get("season_brier", {}).get(s)
        for d in ["above", "below", "between"]:
            row[f"brier_{d}"] = m.get("direction_brier", {}).get(d)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("overall_brier").reset_index(drop=True)
    summary_df.to_csv(OUT_ROOT / "benchmark_summary.csv", index=False)
    print(f"  Saved benchmark_summary.csv ({len(summary_df)} rows)")

    # Trading simulation
    trading_df.to_csv(OUT_ROOT / "trading_simulation_results.csv", index=False)
    print(f"  Saved trading_simulation_results.csv ({len(trading_df)} rows)")

    # Paper trading gate report
    with open(OUT_ROOT / "paper_trading_gate_report.json", "w") as f:
        json.dump(gate_report, f, indent=2)
    print("  Saved paper_trading_gate_report.json")

    # Seasonal stress slices
    if seasonal_stress_all:
        stress_df = pd.DataFrame(seasonal_stress_all)
        stress_df.to_csv(OUT_ROOT / "seasonal_stress_slices.csv", index=False)
        print(f"  Saved seasonal_stress_slices.csv ({len(stress_df)} rows)")

    # Full benchmark report
    report = generate_report(
        df, all_metrics, trading_df, ev_results_all,
        gate_report, e41_weights or {}, best_v2_config
    )
    with open(OUT_ROOT / "README.md", "w") as f:
        f.write(report)
    print("  Saved README.md")

    # ---- Print Summary ----
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(df):,} bucket observations across {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Available WGA V2 configs: {available_configs}")
    print(f"Best V2 config: {best_v2_config}")

    print(f"\n{'Variant':<45} {'Overall':>8} {'IS':>8} {'OOS':>8} {'ECE':>8}")
    print("-" * 85)
    for _, row in summary_df.iterrows():
        oos = f"{row['oos_brier']:.4f}" if pd.notna(row['oos_brier']) else "N/A"
        is_b = f"{row['is_brier']:.4f}" if pd.notna(row['is_brier']) else "N/A"
        ece = f"{row['overall_ece']:.4f}" if pd.notna(row['overall_ece']) else "N/A"
        overall = f"{row['overall_brier']:.4f}" if pd.notna(row['overall_brier']) else "N/A"
        print(f"{str(row['variant']):<45} {overall:>8} {is_b:>8} {oos:>8} {ece:>8}")

    # Trading highlights
    if len(trading_df) > 0 and trading_df["n_trades"].sum() > 0:
        print("\n--- TRADING HIGHLIGHTS ---")
        wga_v2_variants = [v for v in variant_prob_cols if v.startswith("E3") or
                           v.startswith("E4")]
        for variant_base in wga_v2_variants:
            all_rows = trading_df[trading_df["signal"].str.startswith(f"{variant_base}_All")]
            if len(all_rows) > 0:
                has_trades = all_rows[all_rows["n_trades"] > 0]
                if len(has_trades) > 0:
                    best_row = has_trades.loc[has_trades["net_pnl"].idxmax()]
                    print(f"\n  {variant_base} (best threshold={best_row['threshold']:.2f}):")
                    print(f"    Trades: {int(best_row['n_trades'])}, "
                          f"Win rate: {best_row['win_rate']:.1%}")
                    print(f"    Net P&L: ${best_row['net_pnl']:.2f}, "
                          f"ROI: {best_row['roi_pct']:.1f}%, "
                          f"Ann.Sharpe: {best_row['annualized_sharpe']:.3f}")

    # Promotion gate summary
    promoted = [v for v, info in gate_report.get("variants", {}).items()
                if info.get("all_gates_pass")]
    if promoted:
        print(f"\n--- PROMOTION GATE PASSES: {', '.join(promoted)} ---")
    else:
        print("\n--- NO VARIANTS PASS ALL PROMOTION GATES ---")

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {OUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
