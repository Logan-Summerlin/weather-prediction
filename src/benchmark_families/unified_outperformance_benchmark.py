#!/usr/bin/env python3
"""
Unified Outperformance Benchmark: Cross-model synthesis with extended calibration.

Combines flat NN model + WGA V2 attention model with 2022+2023 extended calibration
to implement remaining strategy items from kalshi_nws_outperformance_strategy.md.

NOTE: Kalshi contract data exists only for 2023-2025.  "Extended calibration"
therefore uses the full IS period (2023+2024) rather than just one year, and the
"2023-only" comparison calibrates on 2023 alone -- doubling the calibration data
is the operational goal.  WGA V2 *validation* predictions (2020-2022) supply model
mu/sigma for the 2022 period but there are no Kalshi contracts for 2022, so those
rows do not appear in the contract-level dataset.

Variants:
  U0: Flat model raw Gaussian bucket probs (baseline)
  U1: WGA V2 raw Gaussian bucket probs (baseline)
  U2: Extended-cal isotonic on flat model (2023+2024 cal)
  U3: Extended-cal isotonic on WGA V2 (2023+2024 cal)
  U4: Extended-cal logistic synthesis stacker (flat + WGA V2 + NWS + market)
  U5: Extended-cal contract-level Brier-optimal MLP (dual model features)
  U6: Extended-cal Platt recalibration on U5
  U7: Regime-conditional variance (season x volatility interaction features)
  U8: 2023-only cal contract-level Brier MLP (comparison to show extended-cal benefit)
  U9: Kitchen sink (all features, extended cal, contract-level Brier + Platt + regime)

Output: results/prediction_market_benchmark/unified_outperformance/
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "unified_outperformance"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

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

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}


# ============================================================================
# Part 1: Probability and Metric Helpers
# ============================================================================

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


# ============================================================================
# Scoring Metrics
# ============================================================================

def brier_score(probs, outcomes):
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - o) ** 2))


def log_score(probs, outcomes):
    p = np.clip(np.asarray(probs, dtype=float), PROB_CLIP_MIN, PROB_CLIP_MAX)
    o = np.asarray(outcomes, dtype=float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def expected_calibration_error(probs, outcomes, n_bins=N_CAL_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
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
    bin_indices = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
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
    bin_indices = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        center = round(float((bins[i] + bins[i + 1]) / 2), 4)
        if count > 0:
            rows.append({
                "bin_center": center,
                "mean_predicted": round(float(probs[mask].mean()), 6),
                "mean_observed": round(float(outcomes[mask].mean()), 6),
                "count": count,
            })
        else:
            rows.append({
                "bin_center": center,
                "mean_predicted": center,
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

    season_bs = {}
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        m = df["season"].values == s
        if m.any():
            season_bs[s] = brier_score(probs[m], outcomes[m])

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


# ============================================================================
# Feature Builders
# ============================================================================

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


def build_market_state_features(frame, sigma_col="flat_sigma",
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
        0.0, 1.0,
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


# ============================================================================
# Part 2: Data Loading and Merge
# ============================================================================

def build_merged_dataset() -> pd.DataFrame:
    """Build master merged dataset with all data sources.

    Returns contract-level DataFrame with columns for flat model, WGA V2,
    NWS, pre-settlement and settled market probabilities.
    """
    print("=" * 70)
    print("PART 1: Loading and Preparing Data")
    print("=" * 70)

    # --- Kalshi data ---
    pre = pd.read_csv(ROOT / "data" / "kalshi_presettlement.csv")
    s23 = pd.read_csv(ROOT / "data" / "real_kalshi_2023_2024.csv")
    s25 = pd.read_csv(ROOT / "data" / "real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)

    print(f"  Pre-settlement:    {len(pre)} rows, {pre['date'].nunique()} dates")
    print(f"  Settled:           {len(settled)} rows, {settled['date'].nunique()} dates")

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

    # --- Flat model predictions (extended 2022-2024 + 2025) ---
    flat_is = pd.read_csv(ROOT / "data" / "best_model_predictions_extended_2022_2024.csv")
    flat_oos = pd.read_csv(ROOT / "data" / "best_model_predictions_extended_2025.csv")
    flat_all = pd.concat([
        flat_is[["date", "model_mu", "model_sigma"]],
        flat_oos[["date", "model_mu", "model_sigma"]],
    ], ignore_index=True).drop_duplicates(subset="date", keep="first")
    flat_all = flat_all.rename(columns={"model_mu": "flat_mu", "model_sigma": "flat_sigma"})
    print(f"  Flat model preds:  {len(flat_all)} dates "
          f"({flat_all['date'].min()} to {flat_all['date'].max()})")

    merged = merged.merge(flat_all, on="date", how="inner")

    # --- WGA V2 predictions (val for 2020-2022, test for 2023-2024) ---
    wga_val_path = ROOT / "results" / "wga_v2_model" / "wga_v2_multihead_only" / "predictions_val.csv"
    wga_test_path = ROOT / "results" / "wga_v2_model" / "wga_v2_multihead_only" / "predictions_test.csv"

    wga_parts = []
    if wga_val_path.exists():
        wga_val = pd.read_csv(wga_val_path)
        wga_val["date"] = pd.to_datetime(wga_val["date"]).dt.strftime("%Y-%m-%d")
        wga_parts.append(wga_val)
        print(f"  WGA V2 val:        {len(wga_val)} dates "
              f"({wga_val['date'].min()} to {wga_val['date'].max()})")
    if wga_test_path.exists():
        wga_test = pd.read_csv(wga_test_path)
        wga_test["date"] = pd.to_datetime(wga_test["date"]).dt.strftime("%Y-%m-%d")
        wga_parts.append(wga_test)
        print(f"  WGA V2 test:       {len(wga_test)} dates "
              f"({wga_test['date'].min()} to {wga_test['date'].max()})")

    if wga_parts:
        wga_all = pd.concat(wga_parts, ignore_index=True).drop_duplicates(subset="date", keep="last")
        wga_all = wga_all.rename(columns={
            "model_mu": "wga_mu",
            "model_sigma_cal": "wga_sigma",
            "regime": "wga_regime",
        })
        wga_cols = ["date", "wga_mu", "wga_sigma"]
        if "wga_regime" in wga_all.columns:
            wga_cols.append("wga_regime")
        merged = merged.merge(wga_all[wga_cols], on="date", how="left")
        merged["has_wga"] = merged["wga_mu"].notna()
        n_wga = merged["has_wga"].sum()
        print(f"  WGA V2 attached:   {n_wga} / {len(merged)} rows have WGA V2")
    else:
        merged["wga_mu"] = np.nan
        merged["wga_sigma"] = np.nan
        merged["wga_regime"] = "medium_var"
        merged["has_wga"] = False
        print("  WGA V2:            NO PREDICTIONS FOUND")

    # --- NWS forecasts ---
    nws = pd.read_csv(
        ROOT / "results" / "prediction_market_benchmark" / "nws_probability_forecasts.csv"
    )
    merged = merged.merge(nws[["date", "nws_mu", "nws_sigma"]], on="date", how="inner")
    print(f"  NWS attached:      {merged['nws_mu'].notna().sum()} rows")

    # --- Derived columns ---
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
    print(f"    Has WGA: {merged['has_wga'].sum()} rows")

    return merged


# ============================================================================
# Part 3: Variant Implementations
# ============================================================================

# --------------- U0: Flat model raw Gaussian bucket probs ---------------

def apply_u0_flat_raw(df):
    """U0: Raw flat model Gaussian bucket probabilities."""
    return compute_bucket_probs(df, "flat_mu", "flat_sigma")


# --------------- U1: WGA V2 raw Gaussian bucket probs ------------------

def apply_u1_wga_raw(df):
    """U1: Raw WGA V2 bucket probs; fall back to flat where WGA unavailable."""
    probs = np.full(len(df), np.nan)
    has_wga = df["has_wga"].values

    if has_wga.any():
        sub = df[has_wga]
        probs[has_wga] = compute_bucket_probs_from_arrays(
            sub["wga_mu"].values, sub["wga_sigma"].values,
            sub["direction"].values,
            sub["threshold_low"].values.astype(float),
            sub["threshold_high"].values.astype(float),
        )
    if (~has_wga).any():
        probs[~has_wga] = compute_bucket_probs(df[~has_wga], "flat_mu", "flat_sigma")

    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)


# --------------- U2: Extended-cal isotonic on flat model ----------------

def fit_isotonic(cal_probs, cal_outcomes):
    """Fit IsotonicRegression and return fitted model."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                             out_of_bounds="clip")
    iso.fit(cal_probs, cal_outcomes)
    return iso


def apply_isotonic(iso, probs):
    """Apply a fitted IsotonicRegression to an array of probabilities."""
    return np.clip(
        np.interp(
            np.clip(probs, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
            iso.X_thresholds_, iso.y_thresholds_
        ),
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )


# --------------- U4: Extended-cal logistic synthesis stacker -------------

def _build_u4_features(df, flat_prob, wga_prob, nws_prob, state):
    """Build feature matrix for the U4 logistic synthesis stacker."""
    k = df["presettlement_prob"].values
    spread = state["spread"].values
    sigma_norm = state["sigma_norm"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    X = np.column_stack([
        flat_prob, wga_prob, nws_prob, k,
        flat_prob - k, wga_prob - k, nws_prob - k,
        wga_prob - flat_prob, wga_prob - nws_prob,
        spread, sigma_norm, depth, stale,
        (wga_prob - k) * (1.0 - spread),
        (flat_prob - k) * (1.0 - sigma_norm),
    ])
    return X


def fit_u4_synthesis_stacker(cal_df, flat_prob_cal, wga_prob_cal, nws_prob_cal):
    """Fit logistic regression synthesis stacker on extended calibration data."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y = cal_df["actual_outcome"].values.astype(float)
    state = build_market_state_features(cal_df, sigma_col="flat_sigma")
    X = _build_u4_features(cal_df, flat_prob_cal, wga_prob_cal, nws_prob_cal, state)

    # 75/25 chrono split within cal period for train/isotonic
    n_train = int(len(cal_df) * 0.75)
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
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=42)
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


def apply_u4_synthesis_stacker(df, u4, flat_prob, wga_prob, nws_prob):
    """Apply U4 synthesis stacker to full dataset."""
    state = build_market_state_features(
        df, sigma_col="flat_sigma",
        sigma_p05=u4["sigma_p05"], sigma_p95=u4["sigma_p95"],
    )
    X = _build_u4_features(df, flat_prob, wga_prob, nws_prob, state)
    mu_x = np.array(u4["feature_mean"])
    sd_x = np.array(u4["feature_std"])
    z = (X - mu_x) / sd_x
    logits = z @ np.array(u4["coef"]) + float(u4["intercept"])
    pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(u4["isotonic_x"])
    iso_y = np.array(u4["isotonic_y"])
    pred = np.interp(np.clip(pred, iso_x.min(), iso_x.max()), iso_x, iso_y)
    return np.clip(pred, PROB_CLIP_MIN, PROB_CLIP_MAX)


# --------------- U5: Extended-cal contract-level Brier-optimal MLP -------

def _build_u5_features(df, wga_prob, flat_prob, nws_prob, state,
                       wga_mu, wga_sig, flat_mu, flat_sig):
    """Build the 30-feature matrix for U5/U7/U8/U9 contract-level MLP."""
    k = df["presettlement_prob"].values
    spread = state["spread"].values
    sigma_norm = state["sigma_norm"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # WGA bucket features
    wga_bf = build_bucket_features(df, wga_mu, wga_sig, wga_prob)
    # Flat bucket features
    flat_bf = build_bucket_features(df, flat_mu, flat_sig, flat_prob)

    # Cross-model disagreement
    cross_disagree = np.abs(wga_prob - flat_prob)
    sigma_ratio = wga_sig / (flat_sig + 1e-6)

    X = np.column_stack([
        # 4 base probs
        wga_prob, flat_prob, nws_prob, k,
        # 6 differences
        wga_prob - k, flat_prob - k, nws_prob - k,
        wga_prob - flat_prob, wga_prob - nws_prob, nws_prob - k,
        # 4 market state
        spread, sigma_norm, depth, stale,
        # 6 wga bucket features
        wga_bf["bucket_quantile"], wga_bf["bucket_width"],
        wga_bf["bucket_distance_sigma"], wga_bf["direction_above"],
        wga_bf["direction_below"], wga_bf["neighboring_bucket_sum"],
        # 6 flat bucket features
        flat_bf["bucket_quantile"], flat_bf["bucket_width"],
        flat_bf["bucket_distance_sigma"], flat_bf["direction_above"],
        flat_bf["direction_below"], flat_bf["neighboring_bucket_sum"],
        # 4 cross-model features
        cross_disagree,
        sigma_ratio,
        (wga_prob - k) * (1.0 - spread),
        cross_disagree * sigma_norm,
    ])
    return X


def _mlp_forward(X, coefs, intercepts):
    """Pure numpy MLP forward pass (ReLU hidden, sigmoid output)."""
    acts = X
    for i, (wt, b) in enumerate(zip(coefs, intercepts)):
        acts = acts @ np.array(wt) + np.array(b)
        if i < len(coefs) - 1:
            acts = np.maximum(acts, 0.0)  # ReLU
    return 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))


def _per_day_renormalize(probs, date_vals):
    """Renormalize probabilities so they sum to 1 within each date."""
    out = probs.copy()
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = out[mask].sum()
        if day_sum > 0:
            out[mask] = out[mask] / day_sum
    return out


def fit_contract_brier_mlp(cal_df, X, y, configs, ece_lambda=0.15,
                           train_frac=0.60, val_frac=0.20):
    """Fit contract-level Brier-optimal MLP with 3-way chrono split.

    Returns (best_clf, mu, sd, iso, best_config_info).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neural_network import MLPClassifier

    n_total = len(cal_df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_cal, y_cal = X[n_train + n_val:], y[n_train + n_val:]

    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x = np.where(sd_x < 1e-6, 1.0, sd_x)
    X_train_z = (X_train - mu_x) / sd_x
    X_val_z = (X_val - mu_x) / sd_x
    X_cal_z = (X_cal - mu_x) / sd_x

    best = None
    best_score = float("inf")
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
        "validation_score": best_score,
    }


def apply_contract_brier_mlp(df, params, X):
    """Apply fitted contract-level MLP to full dataset."""
    mu_x = np.array(params["feature_mean"])
    sd_x = np.array(params["feature_std"])
    z = (X - mu_x) / sd_x

    raw = _mlp_forward(z, params["coefs"], params["intercepts"])

    # Isotonic post-calibration
    iso_x = np.array(params["isotonic_x"])
    iso_y = np.array(params["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    calibrated = _per_day_renormalize(calibrated, df["date"].values)
    return np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)


# --------------- U6: Platt recalibration on U5 -------------------------

def fit_platt_recalibration(cal_probs, cal_outcomes):
    """Fit Platt scaling (logistic on logit) + isotonic."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    logit_probs = np.log(np.clip(cal_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
                         / (1 - np.clip(cal_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)))
    X = logit_probs.reshape(-1, 1)
    y = cal_outcomes.astype(float)

    # 50/50 chrono split
    n_half = len(X) // 2
    X_train, y_train = X[:n_half], y[:n_half]
    X_val, y_val = X[n_half:], y[n_half:]

    best_clf = None
    best_brier = float("inf")
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train)
        val_pred = np.clip(clf.predict_proba(X_val)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best_clf = clf

    assert best_clf is not None

    val_raw = np.clip(best_clf.predict_proba(X_val)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(val_raw, y_val)

    return {
        "coef": float(best_clf.coef_[0][0]),
        "intercept": float(best_clf.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "validation_brier": best_brier,
    }


def apply_platt_recalibration(probs, platt):
    """Apply Platt scaling + isotonic to probabilities."""
    logit_p = np.log(np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
                     / (1 - np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)))
    logit_out = logit_p * platt["coef"] + platt["intercept"]
    raw = 1.0 / (1.0 + np.exp(-np.clip(logit_out, -30.0, 30.0)))

    iso_x = np.array(platt["isotonic_x"])
    iso_y = np.array(platt["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)
    return np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)


# --------------- U7: Regime-conditional variance features ----------------

def _build_regime_features(df, flat_mu_arr, wga_mu_arr, wga_sigma_arr,
                           flat_sigma_arr, sigma_norm_arr):
    """Build regime-conditional features to append to U5 feature set."""
    day_of_year = df["date_dt"].dt.day_of_year.values.astype(float)
    season_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    season_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    # Encode wga_regime: low_var=0, medium_var=0.5, high_var=1.0
    regime_raw = df["wga_regime"].fillna("medium_var").values
    regime_encoded = np.where(regime_raw == "low_var", 0.0,
                     np.where(regime_raw == "high_var", 1.0, 0.5))

    # mu_change: day-over-day absolute change in flat_mu
    # Group by date and take first flat_mu per date, then map back
    date_vals = df["date"].values
    unique_dates = sorted(np.unique(date_vals))
    date_to_mu = {}
    for d in unique_dates:
        mask = date_vals == d
        date_to_mu[d] = float(flat_mu_arr[mask][0])

    mu_change = np.zeros(len(df), dtype=float)
    prev_mu = None
    for d in unique_dates:
        current_mu = date_to_mu[d]
        if prev_mu is not None:
            change = abs(current_mu - prev_mu)
        else:
            change = 0.0
        mask = date_vals == d
        mu_change[mask] = change
        prev_mu = current_mu

    # model_disagreement = |wga_mu - flat_mu| / flat_sigma
    model_disagreement = np.abs(wga_mu_arr - flat_mu_arr) / (flat_sigma_arr + 1e-6)

    return np.column_stack([
        season_sin,
        season_cos,
        regime_encoded,
        mu_change,
        sigma_norm_arr,
        model_disagreement,
    ])


# ============================================================================
# Trading Simulation
# ============================================================================

def run_trading_sim(df, signal_col, market_col, threshold, label):
    """Simulate trading: buy YES when signal > market + threshold, NO opposite."""
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
            "total_cost": 0.0, "gross_payout": 0.0, "fees": 0.0, "net_pnl": 0.0,
            "roi_pct": 0.0, "win_rate": 0.0, "avg_edge": 0.0,
            "sharpe": 0.0, "annualized_sharpe": 0.0,
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
    roi = (net_pnl / total_cost * 100) if total_cost > 0 else 0.0
    win_rate = all_wins.mean() if len(all_wins) > 0 else 0.0
    avg_edge = np.abs(edge[buy_yes | buy_no]).mean()
    sharpe = (all_net.mean() / all_net.std()) if all_net.std() > 0 else 0.0
    annualized_sharpe = sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "signal": label, "market": "Kalshi_PreSettlement", "threshold": threshold,
        "n_trades": int(total_trades), "n_yes_trades": int(buy_yes.sum()),
        "n_no_trades": int(buy_no.sum()),
        "total_cost": round(float(total_cost), 2),
        "gross_payout": round(float(gross_payout), 2),
        "fees": round(float(fees), 2), "net_pnl": round(float(net_pnl), 2),
        "roi_pct": round(float(roi), 2), "win_rate": round(float(win_rate), 4),
        "avg_edge": round(float(avg_edge), 4), "sharpe": round(float(sharpe), 4),
        "annualized_sharpe": round(float(annualized_sharpe), 4),
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


# ============================================================================
# EV-Aware Gating
# ============================================================================

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

    spread = ((
        df["ask_cents"].fillna(df["presettlement_prob"] * 100)
        - df["bid_cents"].fillna(df["presettlement_prob"] * 100)
    ).clip(lower=0) / 100.0).values

    sigma = df["flat_sigma"].values.astype(float)
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

    # Quality score
    quality_score = edge * (1.0 - spread) * (1.0 - sigma_norm) * depth

    for base_cut in EV_QUALITY_CUTS:
        dynamic_threshold = (
            base_cut
            + 0.5 * spread
            + 0.04 * sigma_norm
            + 0.02 * (1.0 - depth)
            + 0.01 * stale_norm
        )

        # Also filter by quality score
        trade_mask = (edge > dynamic_threshold) & (quality_score > base_cut)
        n_trades = int(trade_mask.sum())

        if n_trades == 0:
            results.append({
                "variant": variant_name, "base_cut": base_cut,
                "n_trades": 0, "net_pnl": 0.0, "roi_pct": 0.0,
                "win_rate": 0.0, "sharpe": 0.0,
                "pnl_ci95_low": 0.0, "pnl_ci95_high": 0.0,
            })
            continue

        buy_yes = ((signal - market) > dynamic_threshold) & (quality_score > base_cut)
        buy_no = ((market - signal) > dynamic_threshold) & (quality_score > base_cut)

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

        # Bootstrap CI
        daily_pnl_df = pd.DataFrame({"date": df["date"].values, "pnl": pnl})
        daily_pnl = daily_pnl_df.groupby("date")["pnl"].sum().values
        daily_pnl = daily_pnl[daily_pnl != 0]
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
                "variant": variant_name, "slice": season,
                "n": int(mask.sum()),
                "brier": round(brier_score(probs[mask], outcomes[mask]), 6),
                "ece": round(expected_calibration_error(probs[mask], outcomes[mask]), 6),
            })

    # Volatile slice: days where actual temp deviated > 10F from model mu
    if "flat_mu" in oos.columns:
        actual = oos["actual_tmax"].values.astype(float)
        model_mu = oos["flat_mu"].values.astype(float)
        date_volatile = {}
        for date_val in oos["date"].unique():
            dm = oos["date"].values == date_val
            if dm.any():
                dev = np.abs(actual[dm][0] - model_mu[dm][0])
                date_volatile[date_val] = dev > 10.0
        volatile_mask = np.array([date_volatile.get(d, False) for d in oos["date"].values])
        if volatile_mask.sum() > 0:
            results.append({
                "variant": variant_name, "slice": "volatile",
                "n": int(volatile_mask.sum()),
                "brier": round(brier_score(probs[volatile_mask], outcomes[volatile_mask]), 6),
                "ece": round(expected_calibration_error(probs[volatile_mask], outcomes[volatile_mask]), 6),
            })

    return results


# ============================================================================
# Paper-Trading Gate
# ============================================================================

def evaluate_paper_trading_gate(df, all_metrics, variant_prob_cols, ev_results):
    """Evaluate paper-trading promotion gates for each variant.

    Gates:
    1. OOS Brier <= PreSettlement OOS Brier
    2. Best OOS gated P&L positive with positive CI lower bound
    3. ECE <= 0.03
    4. Max tail reliability bin gap <= 0.20
    """
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
        prob_col_name = variant_prob_cols.get(variant)
        gate4 = False
        tail_gap = None
        if prob_col_name is not None and prob_col_name in df.columns:
            oos_probs = oos_df[prob_col_name].values.astype(float)
            oos_outcomes = oos_df["actual_outcome"].values.astype(float)
            # Look at reliability in tail bins (>0.7 or <0.3)
            rel_data = reliability_diagram_data(oos_probs, oos_outcomes)
            max_gap = 0.0
            for rd in rel_data:
                if rd["mean_observed"] is not None and rd["count"] > 5:
                    bc = rd["bin_center"]
                    if bc < 0.15 or bc > 0.85:
                        gap = abs(rd["mean_predicted"] - rd["mean_observed"])
                        max_gap = max(max_gap, gap)
            tail_gap = max_gap
            gate4 = tail_gap <= 0.20

        # Gate 2: OOS gated P&L positive with positive CI
        gate2 = False
        best_gated_pnl = 0.0
        best_ci_low = 0.0
        for ev_row in ev_results:
            if ev_row.get("variant") == variant:
                if ev_row.get("net_pnl", 0) > 0 and ev_row.get("pnl_ci95_low", 0) > 0:
                    gate2 = True
                    if ev_row["net_pnl"] > best_gated_pnl:
                        best_gated_pnl = ev_row["net_pnl"]
                        best_ci_low = ev_row.get("pnl_ci95_low", 0.0)

        all_pass = gate1 and gate2 and gate3 and gate4

        gate_report["variants"][variant] = {
            "oos_brier": round(oos_brier, 6) if oos_brier is not None else None,
            "oos_ece": round(oos_ece, 6) if oos_ece is not None else None,
            "tail_gap": round(tail_gap, 6) if tail_gap is not None else None,
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


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(df, all_metrics, trading_df, ev_results_all,
                    gate_report, variant_prob_cols):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# Unified Outperformance Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Cross-model synthesis combining the flat NN model and WGA V2 attention model")
    lines.append("with extended calibration (full IS period 2023+2024 vs 2023-only baseline).")
    lines.append("")
    lines.append("### Variant Definitions")
    lines.append("")
    lines.append("| Variant | Description |")
    lines.append("|---------|-------------|")
    lines.append("| U0 | Flat model raw Gaussian bucket probs (baseline) |")
    lines.append("| U1 | WGA V2 raw Gaussian bucket probs (baseline) |")
    lines.append("| U2 | Extended-cal isotonic on flat model |")
    lines.append("| U3 | Extended-cal isotonic on WGA V2 |")
    lines.append("| U4 | Extended-cal logistic synthesis stacker |")
    lines.append("| U5 | Extended-cal contract-level Brier-optimal MLP |")
    lines.append("| U6 | Platt recalibration on U5 |")
    lines.append("| U7 | Regime-conditional variance + U5 features |")
    lines.append("| U8 | 2023-only cal contract-level Brier MLP (comparison) |")
    lines.append("| U9 | Kitchen sink (all features + Platt + regime) |")
    lines.append("")
    lines.append(f"- Total bucket-level observations: **{len(df):,}**")
    lines.append(f"- Unique dates: **{df['date'].nunique()}**")
    lines.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    n_is = (df["period"] == "IS").sum()
    n_oos = (df["period"] == "OOS").sum()
    lines.append(f"- IS period (2023-2024): {n_is:,} rows")
    lines.append(f"- OOS period (2025): {n_oos:,} rows")
    lines.append(f"- Rows with WGA V2: {df['has_wga'].sum():,}")
    lines.append("")

    # --- Brier Scores ---
    lines.append("## 1. Brier Score Comparison (lower = better)")
    lines.append("")
    lines.append("| Variant | Overall | IS | OOS | LogScore | ECE | OOS ECE |")
    lines.append("|---------|---------|-----|-----|----------|-----|---------|")
    sorted_metrics = sorted(all_metrics, key=lambda m: m.get("oos_brier", 999))
    fmt = lambda x: f"{x:.4f}" if x is not None else "N/A"
    for m in sorted_metrics:
        v = m["variant"]
        lines.append(
            f"| {v} | {fmt(m.get('overall_brier'))} | {fmt(m.get('is_brier'))} | "
            f"{fmt(m.get('oos_brier'))} | {fmt(m.get('overall_log_score'))} | "
            f"{fmt(m.get('overall_ece'))} | {fmt(m.get('oos_ece'))} |"
        )
    lines.append("")

    # By season
    lines.append("### By Season (Overall Brier)")
    lines.append("")
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    header = "| Variant | " + " | ".join(seasons) + " |"
    lines.append(header)
    lines.append("|---------|" + "|".join(["--------" for _ in seasons]) + "|")
    for m in sorted_metrics[:10]:
        v = m["variant"]
        sb = m.get("season_brier", {})
        vals = " | ".join([fmt(sb.get(s)) for s in seasons])
        lines.append(f"| {v} | {vals} |")
    lines.append("")

    # Brier Decomposition
    lines.append("### Brier Decomposition (Top 5)")
    lines.append("")
    lines.append("| Variant | Brier | Reliability | Resolution | Uncertainty |")
    lines.append("|---------|-------|------------|------------|-------------|")
    for m in sorted_metrics[:5]:
        decomp = m.get("brier_decomposition", {})
        lines.append(
            f"| {m['variant']} | {decomp.get('brier', 'N/A')} | "
            f"{decomp.get('reliability', 'N/A')} | "
            f"{decomp.get('resolution', 'N/A')} | "
            f"{decomp.get('uncertainty', 'N/A')} |"
        )
    lines.append("")

    # --- Extended vs 2023-only comparison ---
    lines.append("## 2. Extended Calibration Impact (U5 vs U8)")
    lines.append("")
    u5_m = next((m for m in all_metrics if m["variant"] == "U5_extended_cal_brier_mlp"), None)
    u8_m = next((m for m in all_metrics if m["variant"] == "U8_2023only_cal_brier_mlp"), None)
    if u5_m and u8_m:
        lines.append(f"- U5 (extended cal) OOS Brier: {fmt(u5_m.get('oos_brier'))}")
        lines.append(f"- U8 (2023-only cal) OOS Brier: {fmt(u8_m.get('oos_brier'))}")
        diff = ((u8_m.get("oos_brier") or 0) - (u5_m.get("oos_brier") or 0))
        if diff > 0:
            lines.append(f"- Extended calibration IMPROVES OOS by {diff:.4f} Brier points")
        elif diff < 0:
            lines.append(f"- 2023-only calibration IMPROVES OOS by {-diff:.4f} Brier points")
        else:
            lines.append("- No difference observed")
    else:
        lines.append("One or both variants not available for comparison.")
    lines.append("")

    # --- Trading Simulation ---
    lines.append("## 3. Trading Simulation")
    lines.append("")
    lines.append(f"Fee rate: {FEE_RATE*100:.0f}% on winnings")
    lines.append("")

    if trading_df is not None and len(trading_df) > 0:
        # Show OOS only, best threshold per variant
        oos_trades = trading_df[trading_df["signal"].str.endswith("_OOS")]
        if len(oos_trades) > 0:
            lines.append("### OOS Trading Summary (best threshold per variant)")
            lines.append("")
            lines.append("| Variant | Threshold | Trades | Win Rate | "
                         "Net P&L | ROI% | Ann. Sharpe | CI Low | CI High |")
            lines.append("|---------|-----------|--------|----------|"
                         "---------|------|-------------|--------|---------|")
            seen = set()
            for _, row in oos_trades.sort_values("net_pnl", ascending=False).iterrows():
                sig = row["signal"]
                variant_base = sig.replace("_OOS", "")
                if variant_base in seen:
                    continue
                seen.add(variant_base)
                ci_lo_str = f"${row.get('pnl_ci95_low', 0):.2f}" if pd.notna(row.get("pnl_ci95_low")) else ""
                ci_hi_str = f"${row.get('pnl_ci95_high', 0):.2f}" if pd.notna(row.get("pnl_ci95_high")) else ""
                lines.append(
                    f"| {variant_base} | {row['threshold']:.2f} | "
                    f"{row['n_trades']} | {row['win_rate']:.1%} | "
                    f"${row['net_pnl']:.2f} | {row['roi_pct']:.1f}% | "
                    f"{row['annualized_sharpe']:.3f} | {ci_lo_str} | {ci_hi_str} |"
                )
            lines.append("")

    # --- EV Gating ---
    lines.append("## 4. EV-Aware Quality Gating (OOS)")
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
    lines.append("## 5. Paper-Trading Promotion Gate")
    lines.append("")
    lines.append(f"PreSettlement OOS Brier: {gate_report.get('presettlement_oos_brier', 'N/A')}")
    lines.append("")
    lines.append("| Variant | OOS Brier <= Pre | Gated P&L Pos CI | ECE <= 0.03 | "
                 "Tail Gap <= 0.20 | ALL PASS |")
    lines.append("|---------|-----------------|------------------|------------|"
                 "-----------------|----------|")
    check = lambda b: "PASS" if b else "FAIL"
    for variant, info in gate_report.get("variants", {}).items():
        gates = info.get("gates", {})
        lines.append(
            f"| {variant} | {check(gates.get('oos_brier_beats_presettlement'))} | "
            f"{check(gates.get('oos_gated_pnl_positive_ci'))} | "
            f"{check(gates.get('ece_below_0.03'))} | "
            f"{check(gates.get('tail_reliability_below_0.20'))} | "
            f"**{check(info.get('all_gates_pass'))}** |"
        )
    lines.append("")

    # --- Key Findings ---
    lines.append("## 6. Key Findings")
    lines.append("")
    best = sorted_metrics[0]
    lines.append(f"- **Best OOS Brier**: {best['variant']} ({best.get('oos_brier', float('nan')):.4f})")

    # Best OOS
    oos_sorted = sorted([m for m in all_metrics if m.get("oos_brier") is not None],
                        key=lambda m: m["oos_brier"])
    if oos_sorted:
        best_oos = oos_sorted[0]
        lines.append(f"- **Best OOS Brier**: {best_oos['variant']} ({best_oos['oos_brier']:.4f})")

    # Promoted variants
    promoted = [v for v, info in gate_report.get("variants", {}).items()
                if info.get("all_gates_pass")]
    if promoted:
        lines.append(f"- **Variants passing all promotion gates**: {', '.join(promoted)}")
    else:
        lines.append("- No variants pass all promotion gates")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(42)

    # ---- Part 1: Load and prepare data ----
    df = build_merged_dataset()

    # ---- Part 2: Compute base bucket probs ----
    print("\n" + "=" * 70)
    print("PART 2: Computing Base Bucket Probabilities")
    print("=" * 70)

    flat_prob_all = apply_u0_flat_raw(df)
    df["flat_prob"] = flat_prob_all
    print(f"  Flat model probs computed: mean={flat_prob_all.mean():.4f}")

    wga_prob_all = apply_u1_wga_raw(df)
    df["wga_prob"] = wga_prob_all
    print(f"  WGA V2 probs computed: mean={wga_prob_all.mean():.4f}")

    nws_prob_all = compute_bucket_probs(df, "nws_mu", "nws_sigma")
    df["nws_prob"] = np.clip(nws_prob_all, PROB_CLIP_MIN, PROB_CLIP_MAX)
    print(f"  NWS probs computed: mean={nws_prob_all.mean():.4f}")

    for col in ["presettlement_prob", "settled_market_prob"]:
        df[col] = df[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    # ---- Prepare mu/sigma arrays for features ----
    has_wga = df["has_wga"].values
    wga_mu_arr = np.where(has_wga, df["wga_mu"].fillna(0).values,
                          df["flat_mu"].values).astype(float)
    wga_sig_arr = np.where(has_wga, df["wga_sigma"].fillna(1).values,
                           df["flat_sigma"].values).astype(float)
    flat_mu_arr = df["flat_mu"].values.astype(float)
    flat_sig_arr = df["flat_sigma"].values.astype(float)

    # ---- Baselines ----
    all_metrics = []
    variant_prob_cols = {}

    # Kalshi PreSettlement
    pre_metrics = compute_comprehensive_metrics(df, "Kalshi_PreSettlement", "presettlement_prob")
    all_metrics.append(pre_metrics)
    print(f"\n  Kalshi PreSettlement: Brier={pre_metrics['overall_brier']:.4f}")

    # NWS
    nws_metrics = compute_comprehensive_metrics(df, "NWS", "nws_prob")
    all_metrics.append(nws_metrics)
    variant_prob_cols["NWS"] = "nws_prob"
    print(f"  NWS:                 Brier={nws_metrics['overall_brier']:.4f}")

    # Kalshi Settled
    settled_metrics = compute_comprehensive_metrics(df, "Kalshi_Settled", "settled_market_prob")
    all_metrics.append(settled_metrics)
    print(f"  Kalshi Settled:      Brier={settled_metrics['overall_brier']:.4f}")

    # ---- U0: Flat model raw ----
    u0_metrics = compute_comprehensive_metrics(df, "U0_flat_raw", "flat_prob")
    all_metrics.append(u0_metrics)
    variant_prob_cols["U0_flat_raw"] = "flat_prob"
    print(f"  U0 Flat Raw:         Brier={u0_metrics['overall_brier']:.4f}")

    # ---- U1: WGA V2 raw ----
    u1_metrics = compute_comprehensive_metrics(df, "U1_wga_raw", "wga_prob")
    all_metrics.append(u1_metrics)
    variant_prob_cols["U1_wga_raw"] = "wga_prob"
    print(f"  U1 WGA Raw:          Brier={u1_metrics['overall_brier']:.4f}")

    # ---- Part 3: Calibration-based variants ----
    print("\n" + "=" * 70)
    print("PART 3: Calibration-Based Variants (Extended Cal = 2023+2024)")
    print("=" * 70)

    # Extended calibration mask: all IS data (2023+2024)
    ext_cal_mask = df["period"] == "IS"
    cal_df_ext = df[ext_cal_mask].copy().reset_index(drop=True)
    print(f"  Extended cal data: {len(cal_df_ext)} rows, "
          f"{cal_df_ext['date'].nunique()} dates "
          f"({cal_df_ext['date'].min()} to {cal_df_ext['date'].max()})")

    # 2023-only calibration mask
    cal_2023_mask = df["date_dt"].dt.year == 2023
    cal_df_2023 = df[cal_2023_mask].copy().reset_index(drop=True)
    print(f"  2023-only cal data: {len(cal_df_2023)} rows, "
          f"{cal_df_2023['date'].nunique()} dates")

    # ---- U2: Extended-cal isotonic on flat model ----
    print("\n  U2: Extended-cal isotonic on flat model...")
    try:
        cal_flat_probs = cal_df_ext["flat_prob"].values
        cal_outcomes = cal_df_ext["actual_outcome"].values.astype(float)
        iso_flat = fit_isotonic(cal_flat_probs, cal_outcomes)
        df["u2_prob"] = apply_isotonic(iso_flat, df["flat_prob"].values)
        u2_metrics = compute_comprehensive_metrics(df, "U2_extended_cal_iso_flat", "u2_prob")
        all_metrics.append(u2_metrics)
        variant_prob_cols["U2_extended_cal_iso_flat"] = "u2_prob"
        print(f"    Brier: overall={u2_metrics['overall_brier']:.4f}, "
              f"IS={u2_metrics['is_brier']:.4f}, OOS={u2_metrics['oos_brier']}")
    except Exception as exc:
        print(f"    [WARN] U2 failed: {exc}")

    # ---- U3: Extended-cal isotonic on WGA V2 ----
    print("\n  U3: Extended-cal isotonic on WGA V2...")
    try:
        cal_wga_probs = cal_df_ext["wga_prob"].values
        iso_wga = fit_isotonic(cal_wga_probs, cal_outcomes)
        df["u3_prob"] = apply_isotonic(iso_wga, df["wga_prob"].values)
        u3_metrics = compute_comprehensive_metrics(df, "U3_extended_cal_iso_wga", "u3_prob")
        all_metrics.append(u3_metrics)
        variant_prob_cols["U3_extended_cal_iso_wga"] = "u3_prob"
        print(f"    Brier: overall={u3_metrics['overall_brier']:.4f}, "
              f"IS={u3_metrics['is_brier']:.4f}, OOS={u3_metrics['oos_brier']}")
    except Exception as exc:
        print(f"    [WARN] U3 failed: {exc}")

    # ---- U4: Extended-cal logistic synthesis stacker ----
    print("\n  U4: Extended-cal logistic synthesis stacker...")
    try:
        u4_params = fit_u4_synthesis_stacker(
            cal_df_ext,
            cal_df_ext["flat_prob"].values,
            cal_df_ext["wga_prob"].values,
            cal_df_ext["nws_prob"].values,
        )
        df["u4_prob"] = apply_u4_synthesis_stacker(
            df, u4_params,
            df["flat_prob"].values, df["wga_prob"].values, df["nws_prob"].values,
        )
        u4_metrics = compute_comprehensive_metrics(df, "U4_extended_cal_synthesis", "u4_prob")
        all_metrics.append(u4_metrics)
        variant_prob_cols["U4_extended_cal_synthesis"] = "u4_prob"
        print(f"    Brier: overall={u4_metrics['overall_brier']:.4f}, "
              f"IS={u4_metrics['is_brier']:.4f}, OOS={u4_metrics['oos_brier']}")
        print(f"    Val Brier (fitting)={u4_params['validation_brier']:.4f}")
    except Exception as exc:
        print(f"    [WARN] U4 failed: {exc}")

    # ---- U5: Extended-cal contract-level Brier-optimal MLP ----
    print("\n  U5: Extended-cal contract-level Brier-optimal MLP...")
    try:
        # Build features for calibration data
        cal_state = build_market_state_features(cal_df_ext, sigma_col="flat_sigma")
        cal_wga_mu = np.where(
            cal_df_ext["has_wga"].values,
            cal_df_ext["wga_mu"].fillna(0).values,
            cal_df_ext["flat_mu"].values,
        ).astype(float)
        cal_wga_sig = np.where(
            cal_df_ext["has_wga"].values,
            cal_df_ext["wga_sigma"].fillna(1).values,
            cal_df_ext["flat_sigma"].values,
        ).astype(float)

        X_cal_u5 = _build_u5_features(
            cal_df_ext,
            cal_df_ext["wga_prob"].values, cal_df_ext["flat_prob"].values,
            cal_df_ext["nws_prob"].values, cal_state,
            cal_wga_mu, cal_wga_sig,
            cal_df_ext["flat_mu"].values.astype(float),
            cal_df_ext["flat_sigma"].values.astype(float),
        )
        y_cal_u5 = cal_df_ext["actual_outcome"].values.astype(float)

        mlp_configs_u5 = [
            ((32,), 3e-3, 1e-3),
            ((64, 32), 5e-3, 8e-4),
            ((128, 64), 8e-3, 6e-4),
            ((128, 64, 32), 1e-2, 5e-4),
        ]
        u5_params = fit_contract_brier_mlp(
            cal_df_ext, X_cal_u5, y_cal_u5, mlp_configs_u5,
            ece_lambda=0.15,
        )
        u5_params["sigma_p05"] = float(cal_state["sigma_p05"].iloc[0])
        u5_params["sigma_p95"] = float(cal_state["sigma_p95"].iloc[0])

        # Apply to full dataset
        full_state = build_market_state_features(
            df, sigma_col="flat_sigma",
            sigma_p05=u5_params["sigma_p05"], sigma_p95=u5_params["sigma_p95"],
        )
        X_full_u5 = _build_u5_features(
            df,
            df["wga_prob"].values, df["flat_prob"].values,
            df["nws_prob"].values, full_state,
            wga_mu_arr, wga_sig_arr, flat_mu_arr, flat_sig_arr,
        )
        df["u5_prob"] = apply_contract_brier_mlp(df, u5_params, X_full_u5)

        u5_metrics = compute_comprehensive_metrics(df, "U5_extended_cal_brier_mlp", "u5_prob")
        all_metrics.append(u5_metrics)
        variant_prob_cols["U5_extended_cal_brier_mlp"] = "u5_prob"
        print(f"    Brier: overall={u5_metrics['overall_brier']:.4f}, "
              f"IS={u5_metrics['is_brier']:.4f}, OOS={u5_metrics['oos_brier']}")
        print(f"    Val Score (fitting)={u5_params['validation_score']:.4f}")
        print(f"    Best arch: {u5_params['hidden_layers']}")
    except Exception as exc:
        print(f"    [WARN] U5 failed: {exc}")
        import traceback
        traceback.print_exc()

    # ---- U6: Platt recalibration on U5 ----
    print("\n  U6: Platt recalibration on U5...")
    try:
        if "u5_prob" in df.columns:
            cal_u5_probs = df.loc[ext_cal_mask, "u5_prob"].values
            cal_u5_outcomes = df.loc[ext_cal_mask, "actual_outcome"].values.astype(float)
            platt_params = fit_platt_recalibration(cal_u5_probs, cal_u5_outcomes)
            df["u6_prob"] = apply_platt_recalibration(df["u5_prob"].values, platt_params)
            # Per-day renormalization
            df["u6_prob"] = _per_day_renormalize(df["u6_prob"].values, df["date"].values)
            df["u6_prob"] = df["u6_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

            u6_metrics = compute_comprehensive_metrics(df, "U6_platt_on_u5", "u6_prob")
            all_metrics.append(u6_metrics)
            variant_prob_cols["U6_platt_on_u5"] = "u6_prob"
            print(f"    Brier: overall={u6_metrics['overall_brier']:.4f}, "
                  f"IS={u6_metrics['is_brier']:.4f}, OOS={u6_metrics['oos_brier']}")
            print(f"    Platt val Brier={platt_params['validation_brier']:.4f}")
        else:
            print("    SKIPPED: U5 not available")
    except Exception as exc:
        print(f"    [WARN] U6 failed: {exc}")

    # ---- U7: Regime-conditional variance ----
    print("\n  U7: Regime-conditional variance features...")
    try:
        # Build U5 features + regime features for calibration data
        cal_sigma_norm = cal_state["sigma_norm"].values
        regime_feats_cal = _build_regime_features(
            cal_df_ext,
            cal_df_ext["flat_mu"].values.astype(float),
            cal_wga_mu, cal_wga_sig,
            cal_df_ext["flat_sigma"].values.astype(float),
            cal_sigma_norm,
        )
        X_cal_u7 = np.column_stack([X_cal_u5, regime_feats_cal])

        mlp_configs_u7 = [
            ((32,), 3e-3, 1e-3),
            ((64, 32), 5e-3, 8e-4),
            ((128, 64), 8e-3, 6e-4),
            ((128, 64, 32), 1e-2, 5e-4),
        ]
        u7_params = fit_contract_brier_mlp(
            cal_df_ext, X_cal_u7, y_cal_u5, mlp_configs_u7,
            ece_lambda=0.15,
        )
        u7_params["sigma_p05"] = u5_params.get("sigma_p05", 2.0)
        u7_params["sigma_p95"] = u5_params.get("sigma_p95", 4.0)

        # Apply to full dataset
        full_sigma_norm = full_state["sigma_norm"].values
        regime_feats_full = _build_regime_features(
            df, flat_mu_arr, wga_mu_arr, wga_sig_arr, flat_sig_arr, full_sigma_norm,
        )
        X_full_u7 = np.column_stack([X_full_u5, regime_feats_full])
        df["u7_prob"] = apply_contract_brier_mlp(df, u7_params, X_full_u7)

        u7_metrics = compute_comprehensive_metrics(df, "U7_regime_conditional", "u7_prob")
        all_metrics.append(u7_metrics)
        variant_prob_cols["U7_regime_conditional"] = "u7_prob"
        print(f"    Brier: overall={u7_metrics['overall_brier']:.4f}, "
              f"IS={u7_metrics['is_brier']:.4f}, OOS={u7_metrics['oos_brier']}")
        print(f"    Val Score (fitting)={u7_params['validation_score']:.4f}")
        print(f"    Best arch: {u7_params['hidden_layers']}")
    except Exception as exc:
        print(f"    [WARN] U7 failed: {exc}")
        import traceback
        traceback.print_exc()

    # ---- U8: 2023-only cal contract-level Brier MLP (comparison) ----
    print("\n  U8: 2023-only cal contract-level Brier MLP...")
    try:
        cal_state_2023 = build_market_state_features(cal_df_2023, sigma_col="flat_sigma")
        cal_wga_mu_2023 = np.where(
            cal_df_2023["has_wga"].values,
            cal_df_2023["wga_mu"].fillna(0).values,
            cal_df_2023["flat_mu"].values,
        ).astype(float)
        cal_wga_sig_2023 = np.where(
            cal_df_2023["has_wga"].values,
            cal_df_2023["wga_sigma"].fillna(1).values,
            cal_df_2023["flat_sigma"].values,
        ).astype(float)

        X_cal_u8 = _build_u5_features(
            cal_df_2023,
            cal_df_2023["wga_prob"].values, cal_df_2023["flat_prob"].values,
            cal_df_2023["nws_prob"].values, cal_state_2023,
            cal_wga_mu_2023, cal_wga_sig_2023,
            cal_df_2023["flat_mu"].values.astype(float),
            cal_df_2023["flat_sigma"].values.astype(float),
        )
        y_cal_u8 = cal_df_2023["actual_outcome"].values.astype(float)

        u8_params = fit_contract_brier_mlp(
            cal_df_2023, X_cal_u8, y_cal_u8, mlp_configs_u5,
            ece_lambda=0.15,
        )
        u8_params["sigma_p05"] = float(cal_state_2023["sigma_p05"].iloc[0])
        u8_params["sigma_p95"] = float(cal_state_2023["sigma_p95"].iloc[0])

        # Apply to full dataset using same full features
        full_state_u8 = build_market_state_features(
            df, sigma_col="flat_sigma",
            sigma_p05=u8_params["sigma_p05"], sigma_p95=u8_params["sigma_p95"],
        )
        X_full_u8 = _build_u5_features(
            df,
            df["wga_prob"].values, df["flat_prob"].values,
            df["nws_prob"].values, full_state_u8,
            wga_mu_arr, wga_sig_arr, flat_mu_arr, flat_sig_arr,
        )
        df["u8_prob"] = apply_contract_brier_mlp(df, u8_params, X_full_u8)

        u8_metrics = compute_comprehensive_metrics(df, "U8_2023only_cal_brier_mlp", "u8_prob")
        all_metrics.append(u8_metrics)
        variant_prob_cols["U8_2023only_cal_brier_mlp"] = "u8_prob"
        print(f"    Brier: overall={u8_metrics['overall_brier']:.4f}, "
              f"IS={u8_metrics['is_brier']:.4f}, OOS={u8_metrics['oos_brier']}")
        print(f"    Val Score (fitting)={u8_params['validation_score']:.4f}")
        print(f"    Best arch: {u8_params['hidden_layers']}")
    except Exception as exc:
        print(f"    [WARN] U8 failed: {exc}")
        import traceback
        traceback.print_exc()

    # ---- U9: Kitchen sink ----
    print("\n  U9: Kitchen sink (all features + Platt + regime)...")
    try:
        # Start with U7 full feature set (U5 + regime)
        # Use wider architecture sweep and tighter ece_lambda
        mlp_configs_u9 = [
            ((64, 32), 5e-3, 8e-4),
            ((128, 64), 8e-3, 6e-4),
            ((128, 64, 32), 1e-2, 5e-4),
            ((256, 128, 64), 1.5e-2, 4e-4),
        ]
        u9_params = fit_contract_brier_mlp(
            cal_df_ext, X_cal_u7, y_cal_u5, mlp_configs_u9,
            ece_lambda=0.20,
        )
        u9_params["sigma_p05"] = u5_params.get("sigma_p05", 2.0)
        u9_params["sigma_p95"] = u5_params.get("sigma_p95", 4.0)

        # Apply MLP
        u9_raw = apply_contract_brier_mlp(df, u9_params, X_full_u7)

        # Platt + isotonic on top
        cal_u9_raw = u9_raw[ext_cal_mask.values]
        cal_u9_outcomes = df.loc[ext_cal_mask, "actual_outcome"].values.astype(float)
        platt_u9 = fit_platt_recalibration(cal_u9_raw, cal_u9_outcomes)
        u9_calibrated = apply_platt_recalibration(u9_raw, platt_u9)
        u9_calibrated = _per_day_renormalize(u9_calibrated, df["date"].values)
        df["u9_prob"] = np.clip(u9_calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)

        u9_metrics = compute_comprehensive_metrics(df, "U9_kitchen_sink", "u9_prob")
        all_metrics.append(u9_metrics)
        variant_prob_cols["U9_kitchen_sink"] = "u9_prob"
        print(f"    Brier: overall={u9_metrics['overall_brier']:.4f}, "
              f"IS={u9_metrics['is_brier']:.4f}, OOS={u9_metrics['oos_brier']}")
        print(f"    MLP Val Score={u9_params['validation_score']:.4f}")
        print(f"    Platt Val Brier={platt_u9['validation_brier']:.4f}")
        print(f"    Best arch: {u9_params['hidden_layers']}")
    except Exception as exc:
        print(f"    [WARN] U9 failed: {exc}")
        import traceback
        traceback.print_exc()

    # ---- Part 4: Trading Simulation ----
    print("\n" + "=" * 70)
    print("PART 4: Trading Simulation")
    print("=" * 70)

    trading_df = run_all_trading_sims(df, variant_prob_cols)
    print(f"  Total simulation rows: {len(trading_df)}")

    # ---- Part 5: EV-Aware Gating ----
    print("\n" + "=" * 70)
    print("PART 5: EV-Aware Quality Gating (OOS)")
    print("=" * 70)

    ev_results_all = []
    seasonal_stress_all = []

    oos_df = df[df["period"] == "OOS"].copy()
    print(f"  OOS data: {len(oos_df)} rows, {oos_df['date'].nunique()} dates")

    # Run EV gating for model variants (skip baselines)
    skip_variants = {"NWS", "Kalshi_PreSettlement", "Kalshi_Settled"}
    for variant_name, prob_col in variant_prob_cols.items():
        if variant_name in skip_variants:
            continue
        if prob_col not in df.columns or oos_df[prob_col].isna().all():
            continue

        print(f"  EV gating: {variant_name}...")
        try:
            ev_results = compute_ev_gating(oos_df, prob_col, variant_name)
            ev_results_all.extend(ev_results)

            stress_results = compute_seasonal_stress_slices(df, prob_col, variant_name)
            seasonal_stress_all.extend(stress_results)
        except Exception as exc:
            print(f"    [WARN] EV gating failed for {variant_name}: {exc}")

    # ---- Part 6: Paper-Trading Gate ----
    print("\n" + "=" * 70)
    print("PART 6: Paper-Trading Promotion Gate")
    print("=" * 70)

    gate_report = evaluate_paper_trading_gate(df, all_metrics, variant_prob_cols, ev_results_all)

    for variant, info in gate_report.get("variants", {}).items():
        gates = info.get("gates", {})
        passed = sum(1 for v in gates.values() if v)
        total = len(gates)
        status = "ALL PASS" if info.get("all_gates_pass") else f"{passed}/{total}"
        print(f"  {variant}: {status}")

    # ---- Part 7: Save Outputs ----
    print("\n" + "=" * 70)
    print("PART 7: Saving Outputs")
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
    summary_df = summary_df.sort_values("oos_brier").reset_index(drop=True)
    summary_df.to_csv(OUT_ROOT / "benchmark_summary.csv", index=False)
    print(f"  Saved benchmark_summary.csv ({len(summary_df)} rows)")

    # Trading simulation results
    trading_df.to_csv(OUT_ROOT / "trading_simulation_results.csv", index=False)
    print(f"  Saved trading_simulation_results.csv ({len(trading_df)} rows)")

    # EV gating results
    if ev_results_all:
        ev_df = pd.DataFrame(ev_results_all)
        ev_df.to_csv(OUT_ROOT / "ev_gating_results.csv", index=False)
        print(f"  Saved ev_gating_results.csv ({len(ev_df)} rows)")

    # Paper trading gate report
    with open(OUT_ROOT / "paper_trading_gate_report.json", "w") as f:
        json.dump(gate_report, f, indent=2)
    print("  Saved paper_trading_gate_report.json")

    # Seasonal stress slices
    if seasonal_stress_all:
        stress_df = pd.DataFrame(seasonal_stress_all)
        stress_df.to_csv(OUT_ROOT / "seasonal_stress_slices.csv", index=False)
        print(f"  Saved seasonal_stress_slices.csv ({len(stress_df)} rows)")

    # Per-row predictions export (all variant probabilities + market + outcome),
    # so downstream trading analyses can re-score under accurate cost models.
    export_cols = (
        ["date", "ticker", "period", "season", "direction",
         "threshold_low", "threshold_high", "actual_outcome", "actual_tmax",
         "presettlement_prob", "bid_cents", "ask_cents"]
        + sorted(set(variant_prob_cols.values()))
    )
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(OUT_ROOT / "per_row_predictions.csv", index=False)
    print(f"  Saved per_row_predictions.csv ({len(df)} rows, "
          f"{len(export_cols)} cols)")

    # Full benchmark report
    report = generate_report(
        df, all_metrics, trading_df, ev_results_all,
        gate_report, variant_prob_cols,
    )
    with open(OUT_ROOT / "README.md", "w") as f:
        f.write(report)
    print("  Saved README.md")

    # ---- Print Summary ----
    print("\n" + "=" * 70)
    print("UNIFIED OUTPERFORMANCE BENCHMARK - RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(df):,} bucket observations across {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"WGA V2 coverage: {df['has_wga'].sum():,} / {len(df):,} rows")
    print(f"Extended cal period: 2023+2024 ({(df['period']=='IS').sum():,} rows)")

    print(f"\n{'Variant':<45} {'Overall':>8} {'IS':>8} {'OOS':>8} {'ECE':>8} {'OOS ECE':>8}")
    print("-" * 95)
    for _, row in summary_df.iterrows():
        fmt_v = lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        print(f"{str(row['variant']):<45} "
              f"{fmt_v(row['overall_brier']):>8} "
              f"{fmt_v(row['is_brier']):>8} "
              f"{fmt_v(row['oos_brier']):>8} "
              f"{fmt_v(row['overall_ece']):>8} "
              f"{fmt_v(row['oos_ece']):>8}")

    # Extended cal impact
    u5_row = summary_df[summary_df["variant"] == "U5_extended_cal_brier_mlp"]
    u8_row = summary_df[summary_df["variant"] == "U8_2023only_cal_brier_mlp"]
    if len(u5_row) > 0 and len(u8_row) > 0:
        u5_oos = u5_row.iloc[0]["oos_brier"]
        u8_oos = u8_row.iloc[0]["oos_brier"]
        if pd.notna(u5_oos) and pd.notna(u8_oos):
            diff = u8_oos - u5_oos
            print(f"\n  Extended cal impact (U5 vs U8 OOS): {diff:+.4f} "
                  f"({'improvement' if diff > 0 else 'regression'})")

    # Promotion gate summary
    print("\n--- PROMOTION GATE SUMMARY ---")
    promoted = [v for v, info in gate_report.get("variants", {}).items()
                if info.get("all_gates_pass")]
    if promoted:
        print(f"  Variants passing all gates: {', '.join(promoted)}")
    else:
        print("  No variants pass all promotion gates")

    # Trading highlights
    if len(trading_df) > 0 and trading_df["n_trades"].sum() > 0:
        print("\n--- OOS TRADING HIGHLIGHTS ---")
        oos_trading = trading_df[
            trading_df["signal"].str.endswith("_OOS") & (trading_df["n_trades"] > 0)
        ].copy()
        if len(oos_trading) > 0:
            best_oos = oos_trading.sort_values("net_pnl", ascending=False).iloc[0]
            print(f"  Best OOS P&L: {best_oos['signal']}, "
                  f"threshold={best_oos['threshold']:.2f}, "
                  f"${best_oos['net_pnl']:.2f} "
                  f"({int(best_oos['n_trades'])} trades, "
                  f"{best_oos['roi_pct']:.1f}% ROI)")

    # EV gating highlights
    if ev_results_all:
        print("\n--- EV GATING HIGHLIGHTS (OOS) ---")
        ev_df = pd.DataFrame(ev_results_all)
        positive_ev = ev_df[ev_df["net_pnl"] > 0]
        if len(positive_ev) > 0:
            best_ev = positive_ev.sort_values("net_pnl", ascending=False).iloc[0]
            print(f"  Best EV-gated: {best_ev['variant']}, "
                  f"cut={best_ev['base_cut']:.2f}, "
                  f"${best_ev['net_pnl']:.2f} "
                  f"({int(best_ev['n_trades'])} trades, "
                  f"CI [{best_ev['pnl_ci95_low']:.2f}, {best_ev['pnl_ci95_high']:.2f}])")
        else:
            print("  No EV-gated configurations show positive P&L on OOS")

    print(f"\n  All results saved to: {OUT_ROOT}/")
    print("  Done.")


if __name__ == "__main__":
    main()
