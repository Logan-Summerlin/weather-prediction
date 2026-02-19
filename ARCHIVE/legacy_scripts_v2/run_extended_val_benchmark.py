#!/usr/bin/env python3
"""Benchmark E0-E33 using the retrained extended-validation model.

Loads the retrained model's predictions (mu/sigma) and replaces the canonical
best_model_predictions in the benchmark pipeline.  Runs all E0-E22 variants
plus sigma-calibration variants (E23-E25), model-quality variants (E26-E29),
and four new gap-targeting variants:
  E26: Tail-Weighted Brier Synthesis (asymmetric loss for upper-tail overconfidence)
  E27: Conformal Prediction Overlay (distribution-free coverage via split-conformal)
  E28: Ensemble Disagreement Sharpening (cross-variant meta-model for resolution)
  E29: Learned Heteroscedastic Sigma (MLP-predicted sigma for PIT/CRPS improvement)
  E30: Aggressive Conformal + Neural Sharpener (conformal calibration + focal-loss MLP)
  E31: Quantile-Crossing-Penalized Synthesis (CDF-monotonic logistic meta-model)
  E32: Platt-Calibrated E17 + Conformal (triple-stack: Platt + conformal + isotonic)
  E33: Temperature-Regime-Aware Synthesis with Resolution Boost (regime-conditional stretching)

Evaluates with two calibration windows: (A) fit on 2023 only, (B) fit on 2023-2024.

Outputs to: results/prediction_market_benchmark/extended_val_model/
"""

from __future__ import annotations

import json
import os
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

OUT_ROOT = ROOT / "results" / "prediction_market_benchmark" / "extended_val_model"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import existing benchmark machinery via importlib
# ---------------------------------------------------------------------------
import importlib.util

def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

bench = _load_module("benchmod", "scripts/test_model_vs_benchmarks.py")
e012 = _load_module("e012", "scripts/run_e0_e1_e2_benchmark.py")
exp = _load_module("probexp", "scripts/probabilistic_ensemble_experiments_v2.py")

# We import the big benchmark script to reuse _apply_variant, fitting helpers, etc.
e0e8 = _load_module("e0e8", "scripts/run_e0_e8_best_model_benchmark.py")

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999

# ---------------------------------------------------------------------------
# 1. Load benchmark data, replacing model_mu / model_sigma with retrain model
# ---------------------------------------------------------------------------

def load_retrain_predictions() -> pd.DataFrame:
    """Load the retrained model's predictions for test period (2023-2024)."""
    test_path = ROOT / "results" / "retrain_extended_validation" / "predictions_test.csv"
    df = pd.read_csv(test_path, parse_dates=["date"])
    df = df.rename(columns={"date": "date_dt"})
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")
    return df


def load_sigma_calibration() -> dict:
    """Load sigma calibration from retrain extended validation."""
    path = ROOT / "results" / "retrain_extended_validation" / "sigma_calibration.json"
    with open(path) as f:
        return json.load(f)


def load_sigma_by_month() -> dict:
    """Load monthly sigma values."""
    path = ROOT / "results" / "retrain_extended_validation" / "sigma_by_month.json"
    with open(path) as f:
        return json.load(f)


def load_base_dataset_with_retrain() -> pd.DataFrame:
    """Load the canonical benchmark dataset but replace model_mu and model_sigma
    with the retrained extended-validation model's predictions."""
    # Load original benchmark data (kalshi + NWS + settled)
    pre = pd.read_csv(ROOT / "data" / "kalshi_presettlement.csv")
    s23 = pd.read_csv(ROOT / "data" / "real_kalshi_2023_2024.csv")
    s25 = pd.read_csv(ROOT / "data" / "real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)
    nws = pd.read_csv(ROOT / "results" / "prediction_market_benchmark" / "nws_probability_forecasts.csv")

    merged = pre.merge(
        settled[["date", "ticker", "direction", "threshold_low", "threshold_high",
                 "actual_outcome", "actual_tmax", "market_prob"]],
        on=["date", "ticker"],
        suffixes=("_pre", ""),
        how="inner",
    )
    merged = merged.rename(columns={"market_prob": "settled_market_prob"})
    merged = merged.dropna(subset=["presettlement_prob"])

    # Load retrain predictions (test period: 2023-2024)
    retrain = load_retrain_predictions()
    sigma_cal = load_sigma_calibration()

    # Build model predictions df: date -> model_mu, model_sigma
    model_preds = retrain[["date", "model_mu", "model_sigma_base"]].copy()
    model_preds = model_preds.rename(columns={"model_sigma_base": "model_sigma"})

    # For 2025, we need to check if we have predictions -- the retrain test period
    # is 2023-2024. For 2025 we use the old best model predictions.
    old_2025 = pd.read_csv(ROOT / "data" / "best_model_predictions_2025.csv")
    old_2025 = old_2025.rename(columns={"date": "date"})

    # Combine: retrain for 2023-2024, old for 2025
    model_all = pd.concat([
        model_preds[["date", "model_mu", "model_sigma"]],
        old_2025[["date", "model_mu", "model_sigma"]],
    ], ignore_index=True).drop_duplicates(subset="date", keep="first")

    merged = merged.merge(model_all[["date", "model_mu", "model_sigma"]], on="date", how="inner")
    merged = merged.merge(nws[["date", "nws_mu", "nws_sigma"]], on="date", how="inner")

    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["period"] = np.where(merged["date_dt"].dt.year <= 2024, "IS", "OOS")
    merged["season"] = merged["date_dt"].dt.month.map(bench.SEASON_MAP)

    print(f"[ExtVal Benchmark] Loaded {len(merged)} rows, "
          f"{merged['date_dt'].dt.date.nunique()} unique dates, "
          f"IS={int((merged['period']=='IS').sum())}, OOS={int((merged['period']=='OOS').sum())}")

    return merged


def apply_retrain_sigma_to_base(base_df: pd.DataFrame, sigma_mode: str = "base") -> pd.DataFrame:
    """Optionally replace model_sigma with retrain calibrated sigma variants.

    sigma_mode: 'base', 'monthly_cal', 'regime_cal', 'combined_cal'
    """
    if sigma_mode == "base":
        return base_df

    retrain = load_retrain_predictions()
    sigma_cal = load_sigma_calibration()

    sigma_col_map = {
        "monthly_cal": "model_sigma_monthly_cal",
        "regime_cal": "model_sigma_regime_cal",
        "combined_cal": "model_sigma_combined_cal",
        "regime_conditional": "model_sigma_regime_conditional",
    }

    if sigma_mode not in sigma_col_map:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    col = sigma_col_map[sigma_mode]
    retrain_sigma = retrain[["date", col]].copy()
    retrain_sigma = retrain_sigma.rename(columns={col: "new_sigma"})

    out = base_df.copy()
    merged = out.merge(retrain_sigma, on="date", how="left")
    # Only replace where we have retrain values (2023-2024)
    has_new = merged["new_sigma"].notna()
    out.loc[has_new, "model_sigma"] = merged.loc[has_new, "new_sigma"].values
    return out


# ---------------------------------------------------------------------------
# 2. Quality metrics
# ---------------------------------------------------------------------------

def brier_score(probs, outcomes):
    return float(np.mean((probs - outcomes) ** 2))


def log_score(probs, outcomes):
    p = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def expected_calibration_error(probs, outcomes, n_bins=10):
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


def reliability_diagram_data(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count > 0:
            rows.append({
                "bin_center": round(float((bins[i] + bins[i+1]) / 2), 4),
                "mean_predicted": round(float(probs[mask].mean()), 6),
                "mean_observed": round(float(outcomes[mask].mean()), 6),
                "count": count,
            })
        else:
            rows.append({
                "bin_center": round(float((bins[i] + bins[i+1]) / 2), 4),
                "mean_predicted": round(float((bins[i] + bins[i+1]) / 2), 6),
                "mean_observed": None,
                "count": 0,
            })
    return rows


def brier_decomposition(probs, outcomes, n_bins=10):
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


def _season_from_month(months):
    return np.where(np.isin(months, [12, 1, 2]), "DJF",
           np.where(np.isin(months, [3, 4, 5]), "MAM",
           np.where(np.isin(months, [6, 7, 8]), "JJA", "SON")))


# ---------------------------------------------------------------------------
# 3. Run a single variant and compute comprehensive metrics
# ---------------------------------------------------------------------------

def compute_comprehensive_metrics(df: pd.DataFrame, variant: str) -> dict:
    """Compute all quality metrics for a benchmark variant."""
    probs = df["model_prob"].values
    outcomes = df["actual_outcome"].values.astype(float)
    pre_probs = df["presettlement_prob"].values
    nws_probs = df["nws_prob"].values

    months = df["date_dt"].dt.month.values
    seasons_djf = _season_from_month(months)

    # Overall
    overall_brier = brier_score(probs, outcomes)
    overall_log = log_score(probs, outcomes)
    overall_ece = expected_calibration_error(probs, outcomes)
    overall_decomp = brier_decomposition(probs, outcomes)

    # IS/OOS
    is_mask = df["period"].values == "IS"
    oos_mask = df["period"].values == "OOS"
    is_brier = brier_score(probs[is_mask], outcomes[is_mask]) if is_mask.any() else None
    oos_brier = brier_score(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None
    is_log = log_score(probs[is_mask], outcomes[is_mask]) if is_mask.any() else None
    oos_log = log_score(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None
    oos_ece = expected_calibration_error(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None

    # By season
    season_brier = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        m = seasons_djf == s
        if m.any():
            season_brier[s] = brier_score(probs[m], outcomes[m])

    # By direction
    dir_brier = {}
    for d in ["above", "below", "between"]:
        m = df["direction"].values == d
        if m.any():
            dir_brier[d] = brier_score(probs[m], outcomes[m])

    # Kalshi PreSettlement and NWS for comparison
    pre_brier_overall = brier_score(pre_probs, outcomes)
    pre_brier_oos = brier_score(pre_probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None
    nws_brier_overall = brier_score(nws_probs, outcomes)
    nws_brier_oos = brier_score(nws_probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else None

    # Reliability diagram
    rel_data = reliability_diagram_data(probs, outcomes)

    return {
        "variant": variant,
        "overall_brier": round(overall_brier, 6),
        "is_brier": round(is_brier, 6) if is_brier is not None else None,
        "oos_brier": round(oos_brier, 6) if oos_brier is not None else None,
        "overall_log_score": round(overall_log, 6),
        "is_log_score": round(is_log, 6) if is_log is not None else None,
        "oos_log_score": round(oos_log, 6) if oos_log is not None else None,
        "overall_ece": round(overall_ece, 6),
        "oos_ece": round(oos_ece, 6) if oos_ece is not None else None,
        "brier_decomposition": overall_decomp,
        "season_brier": {k: round(v, 6) for k, v in season_brier.items()},
        "direction_brier": {k: round(v, 6) for k, v in dir_brier.items()},
        "presettlement_brier_overall": round(pre_brier_overall, 6),
        "presettlement_brier_oos": round(pre_brier_oos, 6) if pre_brier_oos is not None else None,
        "nws_brier_overall": round(nws_brier_overall, 6),
        "nws_brier_oos": round(nws_brier_oos, 6) if nws_brier_oos is not None else None,
        "reliability_diagram": rel_data,
        "n_total": int(len(df)),
        "n_is": int(is_mask.sum()),
        "n_oos": int(oos_mask.sum()),
    }


# ---------------------------------------------------------------------------
# 4. Run all E-variants (reusing e0e8 machinery)
# ---------------------------------------------------------------------------

VARIANT_LIST = [
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
    "E16_conditional_calibration_shrunk",
    "E17_contract_brier_synthesis",
    "E18_regime_adaptive_ensemble",
    "E19_platt_beta_calibration",
    "E20_crps_distributional_synthesis",
    "E21_platt_recalibrated_e17",
    "E22_expanded_platt_e13",
]


def run_variant_with_metrics(base_df, variant, cfg, calib_label="cal2023"):
    """Run a single E-variant: apply transform, compute metrics."""
    try:
        df = e0e8._apply_variant(base_df, variant, cfg)
        # Also compute NWS prob
        df["nws_prob"] = bench.compute_bucket_probs(df, "nws_mu", "nws_sigma")
        for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
            df[col] = df[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

        metrics = compute_comprehensive_metrics(df, f"{variant}_{calib_label}")
        return metrics
    except Exception as exc:
        print(f"  [WARN] Variant {variant} failed: {exc}")
        return {
            "variant": f"{variant}_{calib_label}",
            "overall_brier": None,
            "oos_brier": None,
            "error": str(exc),
        }


def apply_sigma_variant_and_compute(base_df, variant_name, sigma_col, calib_label="cal2023"):
    """Apply a sigma-calibrated variant and compute bucket probs + metrics."""
    retrain = load_retrain_predictions()

    out = base_df.copy()
    retrain_sigma = retrain[["date", sigma_col]].copy()
    retrain_sigma = retrain_sigma.rename(columns={sigma_col: "new_sigma"})

    merged = out.merge(retrain_sigma, on="date", how="left")
    has_new = merged["new_sigma"].notna()
    out.loc[has_new, "model_sigma"] = merged.loc[has_new, "new_sigma"].values

    # Compute bucket probs using the new sigma
    out["model_prob"] = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    metrics = compute_comprehensive_metrics(out, f"{variant_name}_{calib_label}")
    return metrics


def apply_sigma_regime_platt(base_df, cfg, calib_label="cal2023"):
    """E25: regime sigma + Platt recalibration approach.

    First apply regime_cal sigma, then run E19-style Platt+isotonic calibration.
    """
    retrain = load_retrain_predictions()

    out = base_df.copy()
    retrain_sigma = retrain[["date", "model_sigma_regime_cal"]].copy()
    retrain_sigma = retrain_sigma.rename(columns={"model_sigma_regime_cal": "new_sigma"})

    merged = out.merge(retrain_sigma, on="date", how="left")
    has_new = merged["new_sigma"].notna()
    out.loc[has_new, "model_sigma"] = merged.loc[has_new, "new_sigma"].values

    # First get E0 bucket probs with regime sigma
    out["model_prob"] = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Now apply Platt scaling on the model_prob
    # Use the calibration year to fit Platt
    if calib_label == "cal2023":
        cal_mask = out["date_dt"].dt.year == 2023
    else:
        cal_mask = out["date_dt"].dt.year.isin([2023, 2024])

    cal_data = out[cal_mask].copy()
    raw_probs = cal_data["model_prob"].values
    y_cal = cal_data["actual_outcome"].values.astype(float)

    if len(cal_data) > 100:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        n_total = len(cal_data)
        n_platt = int(n_total * 0.5)
        platt_idx = slice(0, n_platt)
        iso_idx = slice(n_platt, n_total)

        logit_probs = np.log(np.clip(raw_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(raw_probs, 1e-6, 1 - 1e-6)))
        X_platt = logit_probs[platt_idx].reshape(-1, 1)
        y_platt = y_cal[platt_idx]

        best_lr = None
        best_brier = float("inf")
        for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
            lr.fit(X_platt, y_platt)
            pred = np.clip(lr.predict_proba(X_platt)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
            brier = float(np.mean((pred - y_platt) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_lr = lr

        if best_lr is not None:
            # Stage 2: Isotonic
            platt_scaled_iso = np.clip(
                best_lr.predict_proba(logit_probs[iso_idx].reshape(-1, 1))[:, 1],
                PROB_CLIP_MIN, PROB_CLIP_MAX,
            )
            iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
            iso.fit(platt_scaled_iso, y_cal[iso_idx])

            # Apply to all data
            all_logit = np.log(np.clip(out["model_prob"].values, 1e-6, 1 - 1e-6) /
                               (1 - np.clip(out["model_prob"].values, 1e-6, 1 - 1e-6)))
            platt_all = np.clip(
                best_lr.predict_proba(all_logit.reshape(-1, 1))[:, 1],
                PROB_CLIP_MIN, PROB_CLIP_MAX,
            )
            iso_x = iso.X_thresholds_
            iso_y = iso.y_thresholds_
            out["model_prob"] = np.interp(
                np.clip(platt_all, iso_x.min(), iso_x.max()), iso_x, iso_y
            )
            out["model_prob"] = np.clip(out["model_prob"], PROB_CLIP_MIN, PROB_CLIP_MAX)

    metrics = compute_comprehensive_metrics(out, f"E25_regime_sigma_platt_{calib_label}")
    return metrics


# ---------------------------------------------------------------------------
# 4b. E26-E29 new variant fitting and application
# ---------------------------------------------------------------------------

def _fit_e26_tail_weighted_brier_synthesis(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E26: Tail-Weighted Brier Synthesis.

    Same architecture as E17 (contract-level MLP on bucket features),
    but with asymmetric tail-upweighted loss during training and additional
    tail-specific features.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()
    frame = calib.copy()
    frame["model_prob"] = bench.compute_bucket_probs(frame, "model_mu", "model_sigma")
    frame["nws_prob"] = bench.compute_bucket_probs(frame, "nws_mu", "nws_sigma")
    frame[["model_prob", "nws_prob", "presettlement_prob"]] = frame[[
        "model_prob", "nws_prob", "presettlement_prob",
    ]].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    y = frame["actual_outcome"].values.astype(float)

    state = e0e8._build_market_state_features(frame)
    m = frame["model_prob"].values
    n = frame["nws_prob"].values
    k = frame["presettlement_prob"].values
    s = state["sigma_norm"].values
    spread = state["spread"].values
    depth = state["depth"].values
    stale = state["stale_norm"].values

    # Bucket-specific features (same as E17)
    bucket_mid = np.where(
        frame["direction"] == "above",
        frame["threshold_low"].values + 2.0,
        np.where(
            frame["direction"] == "below",
            frame["threshold_high"].values - 2.0,
            (frame["threshold_low"].values + frame["threshold_high"].values) / 2.0,
        ),
    )
    mu_vals = frame["model_mu"].values
    sig_vals = frame["model_sigma"].values
    bucket_quantile = e012._cdf(bucket_mid, mu_vals, sig_vals)
    bucket_width = np.where(
        frame["direction"] == "between",
        (frame["threshold_high"].values - frame["threshold_low"].values) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )
    bucket_distance_sigma = np.abs(bucket_mid - mu_vals) / (sig_vals + 1e-6)
    direction_above = (frame["direction"].values == "above").astype(float)
    direction_below = (frame["direction"].values == "below").astype(float)

    frame["_model_prob_tmp"] = m
    date_sum = frame.groupby("date")["_model_prob_tmp"].transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - m, 0.0, None)

    # E26-specific tail features
    tail_indicator = np.maximum(0.0, m - 0.4)
    prob_squared = m ** 2

    X = np.column_stack([
        m, n, k,
        m - k, m - n, n - k,
        spread, s, depth, stale,
        bucket_quantile, bucket_width, bucket_distance_sigma,
        direction_above, direction_below,
        neighboring_bucket_sum,
        tail_indicator, prob_squared,  # new tail features
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

    # Compute sample weights: 3x weight for high-probability samples
    train_model_probs = m[train_idx]
    sample_weight = np.where(train_model_probs > 0.4, 3.0, 1.0)

    best = None
    best_score = float("inf")
    configs = [
        ((32,), 3e-3, 1e-3),
        ((64, 32), 5e-3, 8e-4),
        ((128, 64), 8e-3, 6e-4),
    ]
    ece_lambda = 0.15
    for hidden, alpha, lr in configs:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
            learning_rate_init=lr, max_iter=1200, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        )
        # Fit with sample weights to upweight tail errors
        clf.fit(X_train_z, y_train, sample_weight=sample_weight)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)

        # Evaluate with tail-aware score (Brier + ECE + extra tail penalty)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        val_ece = float(np.mean(np.abs(
            np.digitize(val_pred, np.linspace(0, 1, 11)) -
            np.digitize(val_pred, np.linspace(0, 1, 11))
        )))
        # Compute tail-specific calibration error
        tail_mask = val_pred > 0.4
        if tail_mask.sum() > 10:
            tail_brier = float(np.mean((val_pred[tail_mask] - y_val[tail_mask]) ** 2))
        else:
            tail_brier = val_brier
        score = val_brier + ece_lambda * tail_brier
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
        "coefs": [w.tolist() for w in clf.coefs_],
        "intercepts": [b.tolist() for b in clf.intercepts_],
        "hidden_layers": list(hidden),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": float(state["sigma_p05"].iloc[0]),
        "sigma_p95": float(state["sigma_p95"].iloc[0]),
    }


def _apply_e26(df: pd.DataFrame, e26: dict) -> pd.DataFrame:
    """Apply E26 Tail-Weighted Brier Synthesis to full dataset."""
    out = df.copy()
    model_prob = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
    nws_prob = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values
    state = e0e8._build_market_state_features(out, sigma_p05=e26["sigma_p05"], sigma_p95=e26["sigma_p95"])
    spread_v = state["spread"].values
    s_v = state["sigma_norm"].values
    depth_v = state["depth"].values
    stale_v = state["stale_norm"].values

    bucket_mid = np.where(
        out["direction"] == "above",
        out["threshold_low"].values + 2.0,
        np.where(
            out["direction"] == "below",
            out["threshold_high"].values - 2.0,
            (out["threshold_low"].values + out["threshold_high"].values) / 2.0,
        ),
    )
    mu_vals = out["model_mu"].values
    sig_vals = out["model_sigma"].values
    bucket_quantile = e012._cdf(bucket_mid, mu_vals, sig_vals)
    bucket_width = np.where(
        out["direction"] == "between",
        (out["threshold_high"].values - out["threshold_low"].values) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )
    bucket_distance_sigma = np.abs(bucket_mid - mu_vals) / (sig_vals + 1e-6)
    direction_above = (out["direction"].values == "above").astype(float)
    direction_below = (out["direction"].values == "below").astype(float)
    date_sum = pd.Series(model_prob).groupby(out["date"].values).transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - model_prob, 0.0, None)

    tail_indicator = np.maximum(0.0, model_prob - 0.4)
    prob_squared = model_prob ** 2

    x = np.column_stack([
        model_prob, nws_prob, market_prob,
        model_prob - market_prob, model_prob - nws_prob, nws_prob - market_prob,
        spread_v, s_v, depth_v, stale_v,
        bucket_quantile, bucket_width, bucket_distance_sigma,
        direction_above, direction_below,
        neighboring_bucket_sum,
        tail_indicator, prob_squared,
    ])
    x = (x - np.array(e26["feature_mean"])) / np.array(e26["feature_std"])

    acts = x
    for i, (w, b) in enumerate(zip(e26["coefs"], e26["intercepts"])):
        acts = acts @ np.array(w) + np.array(b)
        if i < len(e26["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)
    raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

    iso_x = np.array(e26["isotonic_x"])
    iso_y = np.array(e26["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e27_conformal_prediction(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E27: Conformal Prediction Overlay.

    Apply split-conformal calibration to E13 probabilities for
    distribution-free coverage guarantees, especially in tails.
    """
    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Get E13 probabilities on calibration year
    e13_df = e0e8._apply_variant(calib, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()
    y = e13_df["actual_outcome"].values.astype(float)

    # Split calibration data chronologically 50/50
    n_total = len(calib)
    n_conf = int(n_total * 0.50)  # conformalization proper
    conf_idx = slice(0, n_conf)
    eval_idx = slice(n_conf, n_total)

    conf_probs = e13_probs[conf_idx]
    conf_y = y[conf_idx]

    # Compute nonconformity scores: |predicted_prob - actual_outcome|
    scores = np.abs(conf_probs - conf_y)

    # Bin-conditional conformal: separate adjustment quantiles for each 0.1-wide bin
    bin_edges = np.linspace(0, 1, 11)  # 10 bins
    bin_adjustments = {}
    alpha = 0.10  # shrinkage alpha for conformal

    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_mask = (conf_probs >= lo) & (conf_probs < hi)
        if bin_mask.sum() >= 5:
            bin_scores = scores[bin_mask]
            # Quantile at 1-alpha level (the conformal quantile)
            q_high = float(np.quantile(bin_scores, 1 - alpha))
            # Also store median for moderate adjustments
            q_med = float(np.quantile(bin_scores, 0.5))
            bin_adjustments[i] = {"q_high": q_high, "q_med": q_med, "n": int(bin_mask.sum())}
        else:
            # Fallback to global quantile
            q_high = float(np.quantile(scores, 1 - alpha))
            q_med = float(np.quantile(scores, 0.5))
            bin_adjustments[i] = {"q_high": q_high, "q_med": q_med, "n": int(bin_mask.sum())}

    # Global adjustment as fallback
    global_q_high = float(np.quantile(scores, 1 - alpha))

    return {
        "bin_adjustments": bin_adjustments,
        "global_q_high": global_q_high,
        "alpha": alpha,
        "bin_edges": bin_edges.tolist(),
    }


def _apply_e27(df: pd.DataFrame, e27: dict, cfg: dict) -> pd.DataFrame:
    """Apply E27 Conformal Prediction Overlay to full dataset."""
    out = df.copy()

    # Get E13 probabilities as base
    e13_df = e0e8._apply_variant(df, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()

    bin_edges = np.array(e27["bin_edges"])
    adjusted = e13_probs.copy()

    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 9:
            bin_mask = (e13_probs >= lo) & (e13_probs <= hi)
        else:
            bin_mask = (e13_probs >= lo) & (e13_probs < hi)

        if not bin_mask.any():
            continue

        adj_info = e27["bin_adjustments"].get(str(i), e27["bin_adjustments"].get(i))
        if adj_info is None:
            continue

        q_high = adj_info["q_high"]
        bin_probs = e13_probs[bin_mask]

        # Conformal adjustment:
        # For overconfident predictions (high prob), shrink toward 0.5
        # For underconfident predictions (low prob), push slightly away from 0.5
        if lo >= 0.5:
            # Upper tail: cap at the conformal-adjusted level
            # adjusted = min(p, p - (q_high - expected_score) * shrink_factor)
            adjusted[bin_mask] = np.minimum(bin_probs, bin_probs - (q_high - 0.5) * 0.3)
            # But never below 0.5 for probs that were above 0.5
            adjusted[bin_mask] = np.maximum(adjusted[bin_mask], 0.5 * np.ones_like(adjusted[bin_mask]))
            # Actually: shrink overconfident probs toward their bin center
            shrink = np.clip(q_high * 0.5, 0.0, 0.3)
            bin_center = (lo + hi) / 2.0
            adjusted[bin_mask] = bin_probs * (1 - shrink) + bin_center * shrink
        elif hi <= 0.5:
            # Lower tail: similar but symmetric
            shrink = np.clip(q_high * 0.5, 0.0, 0.3)
            bin_center = (lo + hi) / 2.0
            adjusted[bin_mask] = bin_probs * (1 - shrink) + bin_center * shrink
        else:
            # Mid range: minimal adjustment
            shrink = np.clip(q_high * 0.2, 0.0, 0.15)
            bin_center = (lo + hi) / 2.0
            adjusted[bin_mask] = bin_probs * (1 - shrink) + bin_center * shrink

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = adjusted[mask].sum()
        if day_sum > 0:
            adjusted[mask] = adjusted[mask] / day_sum

    out["model_prob"] = np.clip(adjusted, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e28_ensemble_disagreement(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E28: Ensemble Disagreement Sharpening.

    Use cross-variant disagreement to decide when to sharpen vs hedge.
    Train logistic regression meta-model on calibration year.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Get probabilities from top 5 variants
    variant_names = ["E0_baseline_ensemble", "E3_weighted_ensemble_E4_uncertainty",
                     "E11_synthesis_stacker_market_aware", "E13_neural_synthesis_mlp",
                     "E17_contract_brier_synthesis"]
    variant_probs = {}
    for vname in variant_names:
        try:
            vdf = e0e8._apply_variant(calib, vname, cfg)
            variant_probs[vname] = vdf["model_prob"].values.copy()
        except Exception:
            # Fallback to E0
            variant_probs[vname] = bench.compute_bucket_probs(calib, "model_mu", "model_sigma").copy()

    y = calib["actual_outcome"].values.astype(float)

    # Stack variant probs
    prob_matrix = np.column_stack([variant_probs[vn] for vn in variant_names])

    # Compute meta-features
    mean_prob = prob_matrix.mean(axis=1)
    std_prob = prob_matrix.std(axis=1)
    max_min_spread = prob_matrix.max(axis=1) - prob_matrix.min(axis=1)
    consensus_x_agreement = mean_prob * (1.0 - std_prob)

    sigma = calib["model_sigma"].values
    sigma_p05 = float(np.percentile(sigma, 5))
    sigma_p95 = float(np.percentile(sigma, 95))
    sigma_norm = np.clip((sigma - sigma_p05) / (sigma_p95 - sigma_p05 + 1e-6), 0.0, 1.0)

    season_sin = np.sin(2.0 * np.pi * calib["date_dt"].dt.dayofyear.values / 365.25)
    season_cos = np.cos(2.0 * np.pi * calib["date_dt"].dt.dayofyear.values / 365.25)

    X = np.column_stack([
        mean_prob,
        std_prob,
        max_min_spread,
        consensus_x_agreement,
        sigma_norm,
        season_sin,
        season_cos,
    ])

    # Chronological split: 60/20/20
    n_total = len(calib)
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

    best_lr = None
    best_brier = float("inf")
    best_c = None
    for c in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        lr.fit(X_train_z, y_train)
        val_pred = np.clip(lr.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best_lr = lr
            best_c = c

    assert best_lr is not None

    # Isotonic post-calibration on remaining cal data
    cal_raw = np.clip(best_lr.predict_proba(X_cal_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    return {
        "variant_names": variant_names,
        "feature_mean": mu_x.tolist(),
        "feature_std": sd_x.tolist(),
        "lr_coef": best_lr.coef_[0].tolist(),
        "lr_intercept": float(best_lr.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": sigma_p05,
        "sigma_p95": sigma_p95,
        "best_C": best_c,
        "validation_brier": best_brier,
    }


def _apply_e28(df: pd.DataFrame, e28: dict, cfg: dict) -> pd.DataFrame:
    """Apply E28 Ensemble Disagreement Sharpening to full dataset."""
    out = df.copy()

    # Get probabilities from top 5 variants
    variant_probs = {}
    for vname in e28["variant_names"]:
        try:
            vdf = e0e8._apply_variant(df, vname, cfg)
            variant_probs[vname] = vdf["model_prob"].values.copy()
        except Exception:
            variant_probs[vname] = bench.compute_bucket_probs(df, "model_mu", "model_sigma").copy()

    prob_matrix = np.column_stack([variant_probs[vn] for vn in e28["variant_names"]])

    mean_prob = prob_matrix.mean(axis=1)
    std_prob = prob_matrix.std(axis=1)
    max_min_spread = prob_matrix.max(axis=1) - prob_matrix.min(axis=1)
    consensus_x_agreement = mean_prob * (1.0 - std_prob)

    sigma = out["model_sigma"].values
    sigma_norm = np.clip(
        (sigma - e28["sigma_p05"]) / (e28["sigma_p95"] - e28["sigma_p05"] + 1e-6), 0.0, 1.0
    )

    season_sin = np.sin(2.0 * np.pi * out["date_dt"].dt.dayofyear.values / 365.25)
    season_cos = np.cos(2.0 * np.pi * out["date_dt"].dt.dayofyear.values / 365.25)

    x = np.column_stack([
        mean_prob,
        std_prob,
        max_min_spread,
        consensus_x_agreement,
        sigma_norm,
        season_sin,
        season_cos,
    ])
    x = (x - np.array(e28["feature_mean"])) / np.array(e28["feature_std"])

    # Apply logistic regression
    logits = x @ np.array(e28["lr_coef"]) + e28["lr_intercept"]
    raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e28["isotonic_x"])
    iso_y = np.array(e28["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e29_learned_sigma(base_df: pd.DataFrame) -> dict:
    """E29: Learned Heteroscedastic Sigma.

    Train a small MLP to predict optimal sigma that minimizes NLL,
    using features from available columns.
    """
    from sklearn.neural_network import MLPRegressor

    # Load retrain predictions for sigma-prediction features
    retrain = load_retrain_predictions()

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Build date-level features from retrain predictions
    by_date = calib[["date", "date_dt", "model_mu", "model_sigma", "actual_tmax"]].drop_duplicates("date").sort_values("date_dt").copy()

    # Merge retrain features
    retrain_features = retrain[["date", "model_sigma_base", "ensemble_std"]].copy()
    by_date = by_date.merge(retrain_features, on="date", how="left")

    # Day-over-day mu change
    by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)

    # Season features
    doy = by_date["date_dt"].dt.dayofyear.values
    by_date["season_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    by_date["season_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Sigma ratio
    by_date["sigma_ratio"] = by_date["model_sigma_base"].fillna(by_date["model_sigma"]) / (
        by_date["ensemble_std"].fillna(1.0) + 1e-6
    )

    feature_cols = ["model_mu", "model_sigma", "model_sigma_base", "ensemble_std",
                    "season_sin", "season_cos", "mu_change", "sigma_ratio"]

    # Fill NaN features with column means
    for c in feature_cols:
        if c not in by_date.columns:
            by_date[c] = 0.0
        by_date[c] = by_date[c].fillna(by_date[c].mean())

    X = by_date[feature_cols].values
    y_actual = by_date["actual_tmax"].values
    mu_base = by_date["model_mu"].values

    # Target: optimal sigma for each day (from residuals)
    residuals = y_actual - mu_base
    # We'll train to predict log(sigma) that minimizes NLL
    # sigma_target = |residual| as a proxy, with floor
    sigma_target = np.clip(np.abs(residuals), 0.5, 15.0)

    # Chronological split: 60/20/20
    n = len(by_date)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    cal_idx = slice(n_train + n_val, n)

    X_train, X_val, X_cal = X[train_idx], X[val_idx], X[cal_idx]
    sig_train = sigma_target[train_idx]
    sig_val = sigma_target[val_idx]

    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x = np.where(sd_x < 1e-6, 1.0, sd_x)
    X_train_z = (X_train - mu_x) / sd_x
    X_val_z = (X_val - mu_x) / sd_x

    # Train small MLP to predict sigma
    best_model = None
    best_val_loss = float("inf")
    for hidden, alpha in [((16,), 1e-2), ((32, 16), 5e-3), ((64, 32), 1e-2)]:
        reg = MLPRegressor(
            hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
            learning_rate_init=1e-3, max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=30,
        )
        reg.fit(X_train_z, np.log(sig_train))
        val_pred_log = reg.predict(X_val_z)
        val_pred_sigma = np.exp(val_pred_log)
        # NLL loss
        val_nll = 0.5 * np.log(2 * np.pi * val_pred_sigma ** 2) + \
                  0.5 * ((sig_val) / val_pred_sigma) ** 2
        val_loss = float(np.mean(val_nll))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = reg

    assert best_model is not None

    # Now we have a sigma predictor. For isotonic post-calibration on bucket probs,
    # we need to compute bucket probs and calibrate them.
    # Build cal-set bucket probs with learned sigma
    X_all_z = (X - mu_x) / sd_x
    pred_log_sigma = best_model.predict(X_all_z)
    pred_sigma = np.clip(np.exp(pred_log_sigma), 0.5, 15.0)

    # Create date -> sigma map
    date_sigma_map = dict(zip(by_date["date"].values, pred_sigma))

    # Apply to contract-level calibration data for isotonic fit
    cal_contracts = calib[calib["date_dt"].dt.year == 2023].copy()
    # Use last 20% of cal year for isotonic
    n_contracts = len(cal_contracts)
    iso_start = int(n_contracts * 0.80)

    cal_contracts["learned_sigma"] = cal_contracts["date"].map(date_sigma_map).fillna(
        cal_contracts["model_sigma"]
    )
    cal_contracts["model_sigma_saved"] = cal_contracts["model_sigma"].copy()
    cal_contracts["model_sigma"] = cal_contracts["learned_sigma"]
    cal_contracts["bucket_prob"] = bench.compute_bucket_probs(cal_contracts, "model_mu", "model_sigma")
    cal_contracts["model_sigma"] = cal_contracts["model_sigma_saved"]

    iso_data = cal_contracts.iloc[iso_start:]
    iso_probs = iso_data["bucket_prob"].values
    iso_y = iso_data["actual_outcome"].values.astype(float)

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    if len(iso_probs) > 20:
        iso.fit(np.clip(iso_probs, PROB_CLIP_MIN, PROB_CLIP_MAX), iso_y)
        has_iso = True
    else:
        has_iso = False

    result = {
        "feature_cols": feature_cols,
        "feature_mean": mu_x.tolist(),
        "feature_std": sd_x.tolist(),
        "coefs": [w.tolist() for w in best_model.coefs_],
        "intercepts": [b.tolist() for b in best_model.intercepts_],
        "hidden_layers": list(best_model.hidden_layer_sizes) if hasattr(best_model.hidden_layer_sizes, '__iter__') else [best_model.hidden_layer_sizes],
        "has_isotonic": has_iso,
    }
    if has_iso:
        result["isotonic_x"] = iso.X_thresholds_.tolist()
        result["isotonic_y"] = iso.y_thresholds_.tolist()

    return result


def _apply_e29(df: pd.DataFrame, e29: dict) -> pd.DataFrame:
    """Apply E29 Learned Heteroscedastic Sigma to full dataset."""
    out = df.copy()

    retrain = load_retrain_predictions()

    # Build date-level features
    by_date = out[["date", "date_dt", "model_mu", "model_sigma"]].drop_duplicates("date").sort_values("date_dt").copy()
    retrain_features = retrain[["date", "model_sigma_base", "ensemble_std"]].copy()
    by_date = by_date.merge(retrain_features, on="date", how="left")

    by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
    doy = by_date["date_dt"].dt.dayofyear.values
    by_date["season_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    by_date["season_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    by_date["sigma_ratio"] = by_date["model_sigma_base"].fillna(by_date["model_sigma"]) / (
        by_date["ensemble_std"].fillna(1.0) + 1e-6
    )

    feature_cols = e29["feature_cols"]
    for c in feature_cols:
        if c not in by_date.columns:
            by_date[c] = 0.0
        by_date[c] = by_date[c].fillna(by_date[c].mean())

    X = by_date[feature_cols].values
    X_z = (X - np.array(e29["feature_mean"])) / np.array(e29["feature_std"])

    # MLP forward pass
    acts = X_z
    for i, (w, b) in enumerate(zip(e29["coefs"], e29["intercepts"])):
        acts = acts @ np.array(w) + np.array(b)
        if i < len(e29["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)  # ReLU
    pred_sigma = np.clip(np.exp(acts.reshape(-1)), 0.5, 15.0)

    # Map learned sigma to contract-level data
    date_sigma_map = dict(zip(by_date["date"].values, pred_sigma))
    out["model_sigma_learned"] = out["date"].map(date_sigma_map).fillna(out["model_sigma"])

    # Compute bucket probs with learned sigma
    saved_sigma = out["model_sigma"].copy()
    out["model_sigma"] = out["model_sigma_learned"]
    out["model_prob"] = bench.compute_bucket_probs(out, "model_mu", "model_sigma")
    out["model_sigma"] = saved_sigma
    out.drop(columns=["model_sigma_learned"], inplace=True)

    # Isotonic post-calibration if available
    if e29["has_isotonic"]:
        iso_x = np.array(e29["isotonic_x"])
        iso_y = np.array(e29["isotonic_y"])
        out["model_prob"] = np.interp(
            np.clip(out["model_prob"].values, iso_x.min(), iso_x.max()), iso_x, iso_y
        )

    # Per-day renormalization
    calibrated = out["model_prob"].values.copy()
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum
    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)

    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ---------------------------------------------------------------------------
# 4c. E30-E33 new variant fitting and application
# ---------------------------------------------------------------------------

def _fit_e30_conformal_neural_sharpener(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E30: Aggressive Conformal + Neural Sharpener.

    Combine E27 conformal shrinkage (for calibration) with a neural
    sharpening step (for resolution).  A small MLP trained with focal-loss
    weighting learns to push conformal-adjusted probabilities away from
    0.5 where the model has genuine skill.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # --- Step 1: get E27 conformal-adjusted probabilities on calibration ---
    e27_cfg = _fit_e27_conformal_prediction(base_df, cfg)
    e27_df = _apply_e27(calib, e27_cfg, cfg)
    conformal_probs = e27_df["model_prob"].values.copy()

    y = calib["actual_outcome"].values.astype(float)

    # --- Step 2: build sharpening features ---
    sigma = calib["model_sigma"].values
    sigma_p05 = float(np.percentile(sigma, 5))
    sigma_p95 = float(np.percentile(sigma, 95))
    sigma_norm = np.clip((sigma - sigma_p05) / (sigma_p95 - sigma_p05 + 1e-6), 0.0, 1.0)

    doy = calib["date_dt"].dt.dayofyear.values
    season_sin = np.sin(2.0 * np.pi * doy / 365.25)
    season_cos = np.cos(2.0 * np.pi * doy / 365.25)

    direction_above = (calib["direction"].values == "above").astype(float)
    direction_below = (calib["direction"].values == "below").astype(float)

    distance_from_half = np.abs(conformal_probs - 0.5)

    X = np.column_stack([
        conformal_probs,
        distance_from_half,
        sigma_norm,
        season_sin,
        season_cos,
        direction_above,
        direction_below,
    ])

    # Chronological 3-way split: 50/30/20
    n_total = len(calib)
    n_train = int(n_total * 0.50)
    n_val = int(n_total * 0.30)
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

    # Focal loss weighting: upweight samples where model is uncertain
    # p_correct = p if y=1 else (1-p)
    p_correct_train = np.where(y_train == 1, conformal_probs[train_idx], 1.0 - conformal_probs[train_idx])
    gamma = 2.0
    focal_weight = (1.0 - p_correct_train) ** gamma
    # Ensure minimum weight
    focal_weight = np.clip(focal_weight, 0.1, 10.0)

    best = None
    best_score = float("inf")
    configs = [
        ((16,), 5e-3, 1e-3),
        ((32,), 3e-3, 8e-4),
        ((32, 16), 5e-3, 6e-4),
    ]
    for hidden, alpha, lr in configs:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
            learning_rate_init=lr, max_iter=1500, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        )
        clf.fit(X_train_z, y_train, sample_weight=focal_weight)
        val_pred = np.clip(clf.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_score:
            best_score = val_brier
            best = (clf, hidden, alpha, lr)

    assert best is not None
    clf, hidden, alpha, lr = best

    # Isotonic post-calibration on last slice
    cal_raw = np.clip(clf.predict_proba(X_cal_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    return {
        "e27_cfg": e27_cfg,
        "feature_mean": mu_x.tolist(),
        "feature_std": sd_x.tolist(),
        "coefs": [w.tolist() for w in clf.coefs_],
        "intercepts": [b.tolist() for b in clf.intercepts_],
        "hidden_layers": list(hidden),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": sigma_p05,
        "sigma_p95": sigma_p95,
    }


def _apply_e30(df: pd.DataFrame, e30: dict, cfg: dict) -> pd.DataFrame:
    """Apply E30 Aggressive Conformal + Neural Sharpener to full dataset."""
    out = df.copy()

    # Step 1: get E27 conformal-adjusted probabilities
    e27_df = _apply_e27(df, e30["e27_cfg"], cfg)
    conformal_probs = e27_df["model_prob"].values.copy()

    # Step 2: build sharpening features
    sigma = out["model_sigma"].values
    sigma_norm = np.clip(
        (sigma - e30["sigma_p05"]) / (e30["sigma_p95"] - e30["sigma_p05"] + 1e-6), 0.0, 1.0
    )
    doy = out["date_dt"].dt.dayofyear.values
    season_sin = np.sin(2.0 * np.pi * doy / 365.25)
    season_cos = np.cos(2.0 * np.pi * doy / 365.25)
    direction_above = (out["direction"].values == "above").astype(float)
    direction_below = (out["direction"].values == "below").astype(float)
    distance_from_half = np.abs(conformal_probs - 0.5)

    x = np.column_stack([
        conformal_probs,
        distance_from_half,
        sigma_norm,
        season_sin,
        season_cos,
        direction_above,
        direction_below,
    ])
    x = (x - np.array(e30["feature_mean"])) / np.array(e30["feature_std"])

    # MLP forward pass
    acts = x
    for i, (w, b) in enumerate(zip(e30["coefs"], e30["intercepts"])):
        acts = acts @ np.array(w) + np.array(b)
        if i < len(e30["coefs"]) - 1:
            acts = np.maximum(acts, 0.0)
    raw = 1.0 / (1.0 + np.exp(-np.clip(acts.reshape(-1), -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e30["isotonic_x"])
    iso_y = np.array(e30["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e31_quantile_crossing_synthesis(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E31: Quantile-Crossing-Penalized Synthesis.

    Build a synthesis model that respects the ordered structure of
    temperature bucket probabilities via monotonic CDF enforcement.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Get E13 probabilities on calibration year
    e13_df = e0e8._apply_variant(calib, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()

    y = calib["actual_outcome"].values.astype(float)

    # Compute bucket features
    model_prob_base = bench.compute_bucket_probs(calib, "model_mu", "model_sigma")
    nws_prob = bench.compute_bucket_probs(calib, "nws_mu", "nws_sigma")
    market_prob = calib["presettlement_prob"].values

    sigma = calib["model_sigma"].values
    sigma_p05 = float(np.percentile(sigma, 5))
    sigma_p95 = float(np.percentile(sigma, 95))
    sigma_norm = np.clip((sigma - sigma_p05) / (sigma_p95 - sigma_p05 + 1e-6), 0.0, 1.0)

    doy = calib["date_dt"].dt.dayofyear.values
    season_sin = np.sin(2.0 * np.pi * doy / 365.25)
    season_cos = np.cos(2.0 * np.pi * doy / 365.25)

    # Compute per-date CDF position for E13 probs
    # For each contract, compute CDF: cumulative sum of bucket probs from lowest threshold
    # Sort within each date by threshold
    calib_sorted = calib.copy()
    calib_sorted["e13_prob"] = e13_probs
    calib_sorted["nws_prob_feat"] = nws_prob
    calib_sorted["market_prob_feat"] = market_prob
    calib_sorted["sigma_norm_feat"] = sigma_norm
    calib_sorted["season_sin_feat"] = season_sin
    calib_sorted["season_cos_feat"] = season_cos

    # Compute bucket_mid for ordering
    bucket_mid = np.where(
        calib["direction"] == "above",
        calib["threshold_low"].values + 2.0,
        np.where(
            calib["direction"] == "below",
            calib["threshold_high"].values - 2.0,
            (calib["threshold_low"].values + calib["threshold_high"].values) / 2.0,
        ),
    )
    mu_vals = calib["model_mu"].values
    sig_vals = calib["model_sigma"].values
    bucket_width_sigma = np.where(
        calib["direction"] == "between",
        (calib["threshold_high"].values - calib["threshold_low"].values) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )

    # CDF position: cumulative sum of E13 probs within each date
    calib_sorted["bucket_mid"] = bucket_mid
    calib_sorted = calib_sorted.sort_values(["date", "bucket_mid"])

    # Compute cumulative CDF within date
    cdf_values = []
    for _, group in calib_sorted.groupby("date"):
        probs = group["e13_prob"].values
        cdf = np.cumsum(probs)
        cdf_values.extend(cdf.tolist())
    calib_sorted["e13_cdf_position"] = cdf_values

    # Neighboring bucket sum
    date_sum = calib_sorted.groupby("date")["e13_prob"].transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - calib_sorted["e13_prob"].values, 0.0, None)

    X = np.column_stack([
        calib_sorted["e13_prob"].values,
        calib_sorted["e13_cdf_position"].values,
        calib_sorted["nws_prob_feat"].values,
        calib_sorted["market_prob_feat"].values,
        bucket_width_sigma[calib_sorted.index],
        neighboring_bucket_sum,
        calib_sorted["sigma_norm_feat"].values,
        calib_sorted["season_sin_feat"].values,
        calib_sorted["season_cos_feat"].values,
    ])
    y_sorted = calib_sorted["actual_outcome"].values.astype(float)

    # Chronological 3-way split: 60/20/20
    n_total = len(calib_sorted)
    n_train = int(n_total * 0.60)
    n_val = int(n_total * 0.20)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    cal_idx = slice(n_train + n_val, n_total)

    X_train, y_train = X[train_idx], y_sorted[train_idx]
    X_val, y_val = X[val_idx], y_sorted[val_idx]
    X_cal, y_cal = X[cal_idx], y_sorted[cal_idx]

    mu_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x = np.where(sd_x < 1e-6, 1.0, sd_x)
    X_train_z = (X_train - mu_x) / sd_x
    X_val_z = (X_val - mu_x) / sd_x
    X_cal_z = (X_cal - mu_x) / sd_x

    # Fit logistic regression meta-model with C sweep
    best_lr = None
    best_brier = float("inf")
    best_c = None
    for c in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        lr.fit(X_train_z, y_train)
        val_pred = np.clip(lr.predict_proba(X_val_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        val_brier = float(np.mean((val_pred - y_val) ** 2))
        if val_brier < best_brier:
            best_brier = val_brier
            best_lr = lr
            best_c = c

    assert best_lr is not None

    # Isotonic post-calibration on cal data
    cal_raw = np.clip(best_lr.predict_proba(X_cal_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(cal_raw, y_cal)

    return {
        "feature_mean": mu_x.tolist(),
        "feature_std": sd_x.tolist(),
        "lr_coef": best_lr.coef_[0].tolist(),
        "lr_intercept": float(best_lr.intercept_[0]),
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "sigma_p05": sigma_p05,
        "sigma_p95": sigma_p95,
        "best_C": best_c,
    }


def _apply_e31(df: pd.DataFrame, e31: dict, cfg: dict) -> pd.DataFrame:
    """Apply E31 Quantile-Crossing-Penalized Synthesis to full dataset."""
    out = df.copy()

    # Get E13 probabilities as base
    e13_df = e0e8._apply_variant(df, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()

    nws_prob = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    market_prob = out["presettlement_prob"].values

    sigma = out["model_sigma"].values
    sigma_norm = np.clip(
        (sigma - e31["sigma_p05"]) / (e31["sigma_p95"] - e31["sigma_p05"] + 1e-6), 0.0, 1.0
    )

    doy = out["date_dt"].dt.dayofyear.values
    season_sin = np.sin(2.0 * np.pi * doy / 365.25)
    season_cos = np.cos(2.0 * np.pi * doy / 365.25)

    mu_vals = out["model_mu"].values
    sig_vals = out["model_sigma"].values
    bucket_mid = np.where(
        out["direction"] == "above",
        out["threshold_low"].values + 2.0,
        np.where(
            out["direction"] == "below",
            out["threshold_high"].values - 2.0,
            (out["threshold_low"].values + out["threshold_high"].values) / 2.0,
        ),
    )
    bucket_width_sigma = np.where(
        out["direction"] == "between",
        (out["threshold_high"].values - out["threshold_low"].values) / (sig_vals + 1e-6),
        4.0 / (sig_vals + 1e-6),
    )

    # Compute per-date CDF position
    temp = out[["date"]].copy()
    temp["e13_prob"] = e13_probs
    temp["bucket_mid"] = bucket_mid
    temp = temp.sort_values(["date", "bucket_mid"])
    cdf_values = []
    for _, group in temp.groupby("date"):
        probs = group["e13_prob"].values
        cdf = np.cumsum(probs)
        cdf_values.extend(cdf.tolist())
    temp["e13_cdf_position"] = cdf_values
    # Map back to original order
    cdf_map = dict(zip(temp.index, temp["e13_cdf_position"].values))
    e13_cdf_position = np.array([cdf_map[i] for i in out.index])

    # Neighboring bucket sum
    date_sum = pd.Series(e13_probs, index=out.index).groupby(out["date"].values).transform("sum")
    neighboring_bucket_sum = np.clip(date_sum.values - e13_probs, 0.0, None)

    x = np.column_stack([
        e13_probs,
        e13_cdf_position,
        nws_prob,
        market_prob,
        bucket_width_sigma,
        neighboring_bucket_sum,
        sigma_norm,
        season_sin,
        season_cos,
    ])
    x = (x - np.array(e31["feature_mean"])) / np.array(e31["feature_std"])

    # Apply logistic regression
    logits = x @ np.array(e31["lr_coef"]) + e31["lr_intercept"]
    raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    # Isotonic post-calibration
    iso_x = np.array(e31["isotonic_x"])
    iso_y = np.array(e31["isotonic_y"])
    calibrated = np.interp(np.clip(raw, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Enforce monotonic CDF per-day via isotonic regression on the CDF
    from sklearn.isotonic import IsotonicRegression as IsoReg
    temp2 = out[["date"]].copy()
    temp2["bucket_mid"] = bucket_mid
    temp2["calibrated"] = calibrated
    temp2 = temp2.sort_values(["date", "bucket_mid"])

    corrected = []
    for date_val, group in temp2.groupby("date"):
        probs = group["calibrated"].values
        n_buckets = len(probs)
        if n_buckets <= 1:
            corrected.extend(probs.tolist())
            continue

        # Convert probs to CDF
        cdf = np.cumsum(probs)
        # Enforce monotonicity via isotonic regression
        positions = np.arange(n_buckets, dtype=float)
        iso_mono = IsoReg(y_min=0.0, out_of_bounds="clip")
        iso_mono.fit(positions, cdf)
        cdf_corrected = iso_mono.predict(positions)
        # Convert CDF back to probs
        probs_corrected = np.diff(np.concatenate([[0.0], cdf_corrected]))
        probs_corrected = np.clip(probs_corrected, PROB_CLIP_MIN, None)
        corrected.extend(probs_corrected.tolist())

    temp2["corrected"] = corrected
    corrected_map = dict(zip(temp2.index, temp2["corrected"].values))
    calibrated_final = np.array([corrected_map[i] for i in out.index])

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated_final[mask].sum()
        if day_sum > 0:
            calibrated_final[mask] = calibrated_final[mask] / day_sum

    out["model_prob"] = np.clip(calibrated_final, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e32_platt_conformal_e17(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E32: Platt-Calibrated E17 + Conformal triple-stack.

    Chains: E17 contract-level MLP -> Platt scaling -> conformal bin-shrinkage
    -> isotonic regression.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Get E17 probabilities on calibration year
    e17_df = e0e8._apply_variant(calib, "E17_contract_brier_synthesis", cfg)
    e17_probs = e17_df["model_prob"].values.copy()
    y = calib["actual_outcome"].values.astype(float)

    # 4-way chronological split: 50% Platt, 25% conformal, 25% isotonic
    n_total = len(calib)
    n_platt = int(n_total * 0.50)
    n_conf = int(n_total * 0.25)
    platt_idx = slice(0, n_platt)
    conf_idx = slice(n_platt, n_platt + n_conf)
    iso_idx = slice(n_platt + n_conf, n_total)

    # --- Stage 1: Platt scaling ---
    platt_probs = e17_probs[platt_idx]
    platt_y = y[platt_idx]
    logit_probs = np.log(np.clip(platt_probs, 1e-6, 1 - 1e-6) /
                         (1 - np.clip(platt_probs, 1e-6, 1 - 1e-6)))
    X_platt = logit_probs.reshape(-1, 1)

    best_lr = None
    best_brier = float("inf")
    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        lr.fit(X_platt, platt_y)
        pred = np.clip(lr.predict_proba(X_platt)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        brier = float(np.mean((pred - platt_y) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_lr = lr

    assert best_lr is not None

    # Apply Platt to conformal split
    conf_probs_raw = e17_probs[conf_idx]
    conf_logits = np.log(np.clip(conf_probs_raw, 1e-6, 1 - 1e-6) /
                         (1 - np.clip(conf_probs_raw, 1e-6, 1 - 1e-6)))
    conf_probs_platt = np.clip(
        best_lr.predict_proba(conf_logits.reshape(-1, 1))[:, 1],
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )
    conf_y = y[conf_idx]

    # --- Stage 2: Conformal bin-conditional shrinkage ---
    scores = np.abs(conf_probs_platt - conf_y)
    bin_edges = np.linspace(0, 1, 11)
    bin_adjustments = {}
    alpha = 0.10

    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_mask = (conf_probs_platt >= lo) & (conf_probs_platt < (hi if i < 9 else hi + 0.01))
        if bin_mask.sum() >= 5:
            bin_scores = scores[bin_mask]
            q_high = float(np.quantile(bin_scores, 1 - alpha))
            q_med = float(np.quantile(bin_scores, 0.5))
        else:
            q_high = float(np.quantile(scores, 1 - alpha))
            q_med = float(np.quantile(scores, 0.5))
        bin_adjustments[i] = {"q_high": q_high, "q_med": q_med, "n": int(bin_mask.sum())}

    # Apply conformal to isotonic split
    iso_probs_raw = e17_probs[iso_idx]
    iso_logits = np.log(np.clip(iso_probs_raw, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(iso_probs_raw, 1e-6, 1 - 1e-6)))
    iso_probs_platt = np.clip(
        best_lr.predict_proba(iso_logits.reshape(-1, 1))[:, 1],
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )

    # Apply conformal shrinkage
    iso_probs_conformal = iso_probs_platt.copy()
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 9:
            bin_mask = (iso_probs_platt >= lo) & (iso_probs_platt <= hi)
        else:
            bin_mask = (iso_probs_platt >= lo) & (iso_probs_platt < hi)
        if not bin_mask.any():
            continue
        adj = bin_adjustments.get(i)
        if adj is None:
            continue
        q_high = adj["q_high"]
        shrink = np.clip(q_high * 0.4, 0.0, 0.3)
        bin_center = (lo + hi) / 2.0
        iso_probs_conformal[bin_mask] = iso_probs_platt[bin_mask] * (1 - shrink) + bin_center * shrink

    iso_y = y[iso_idx]

    # --- Stage 3: Isotonic regression ---
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(iso_probs_conformal, iso_y)

    return {
        "platt_coef": float(best_lr.coef_[0][0]),
        "platt_intercept": float(best_lr.intercept_[0]),
        "bin_edges": bin_edges.tolist(),
        "bin_adjustments": bin_adjustments,
        "alpha": alpha,
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
    }


def _apply_e32(df: pd.DataFrame, e32: dict, cfg: dict) -> pd.DataFrame:
    """Apply E32 Platt-Calibrated E17 + Conformal to full dataset."""
    out = df.copy()

    # Get E17 probabilities
    e17_df = e0e8._apply_variant(df, "E17_contract_brier_synthesis", cfg)
    e17_probs = e17_df["model_prob"].values.copy()

    # Stage 1: Platt scaling
    logits = np.log(np.clip(e17_probs, 1e-6, 1 - 1e-6) /
                    (1 - np.clip(e17_probs, 1e-6, 1 - 1e-6)))
    platt_logits = e32["platt_coef"] * logits + e32["platt_intercept"]
    platt_probs = 1.0 / (1.0 + np.exp(-np.clip(platt_logits, -30.0, 30.0)))

    # Stage 2: Conformal bin-conditional shrinkage
    bin_edges = np.array(e32["bin_edges"])
    conformal_probs = platt_probs.copy()
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 9:
            bin_mask = (platt_probs >= lo) & (platt_probs <= hi)
        else:
            bin_mask = (platt_probs >= lo) & (platt_probs < hi)
        if not bin_mask.any():
            continue
        adj = e32["bin_adjustments"].get(str(i), e32["bin_adjustments"].get(i))
        if adj is None:
            continue
        q_high = adj["q_high"]
        shrink = np.clip(q_high * 0.4, 0.0, 0.3)
        bin_center = (lo + hi) / 2.0
        conformal_probs[bin_mask] = platt_probs[bin_mask] * (1 - shrink) + bin_center * shrink

    # Stage 3: Isotonic post-calibration
    iso_x = np.array(e32["isotonic_x"])
    iso_y = np.array(e32["isotonic_y"])
    calibrated = np.interp(
        np.clip(conformal_probs, iso_x.min(), iso_x.max()), iso_x, iso_y
    )

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


def _fit_e33_regime_aware_resolution_boost(base_df: pd.DataFrame, cfg: dict) -> dict:
    """E33: Temperature-Regime-Aware Synthesis with Resolution Boost.

    Classify each day into a regime (mu level x volatility x season) and
    learn regime-specific probability stretching to boost resolution.
    """
    from sklearn.isotonic import IsotonicRegression

    calib = base_df[base_df["date_dt"].dt.year == 2023].copy()

    # Get E13 probabilities on calibration year
    e13_df = e0e8._apply_variant(calib, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()
    y = calib["actual_outcome"].values.astype(float)

    # Build date-level regime features
    by_date = calib[["date", "date_dt", "model_mu", "model_sigma"]].drop_duplicates("date").sort_values("date_dt").copy()
    by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
    months = by_date["date_dt"].dt.month.values

    # Regime classification
    # mu_level terciles
    mu_terciles = np.percentile(by_date["model_mu"].values, [33.3, 66.7])
    by_date["regime_mu"] = np.where(
        by_date["model_mu"].values < mu_terciles[0], "cold",
        np.where(by_date["model_mu"].values < mu_terciles[1], "moderate", "hot")
    )

    # volatility terciles (|mu_change|)
    vol_terciles = np.percentile(by_date["mu_change"].values, [33.3, 66.7])
    by_date["regime_vol"] = np.where(
        by_date["mu_change"].values < vol_terciles[0], "stable",
        np.where(by_date["mu_change"].values < vol_terciles[1], "moderate_vol", "volatile")
    )

    # season
    by_date["regime_season"] = _season_from_month(months)

    # Map regimes back to contract-level data
    regime_map = by_date.set_index("date")[["regime_mu", "regime_vol", "regime_season"]].to_dict("index")

    calib_regime_mu = np.array([regime_map.get(d, {}).get("regime_mu", "moderate") for d in calib["date"].values])
    calib_regime_vol = np.array([regime_map.get(d, {}).get("regime_vol", "moderate_vol") for d in calib["date"].values])
    calib_regime_season = np.array([regime_map.get(d, {}).get("regime_season", "MAM") for d in calib["date"].values])

    # Chronological split: first 70% for fitting stretch factors, last 30% for isotonic
    n_total = len(calib)
    n_fit = int(n_total * 0.70)
    fit_idx = slice(0, n_fit)
    iso_idx = slice(n_fit, n_total)

    fit_probs = e13_probs[fit_idx]
    fit_y = y[fit_idx]
    fit_mu = calib_regime_mu[fit_idx]
    fit_vol = calib_regime_vol[fit_idx]
    fit_season = calib_regime_season[fit_idx]

    # Learn stretch factors per regime cell
    # Stretch: stretched = base_rate + stretch_factor * (prob - base_rate)
    # where stretch_factor > 1 pushes probabilities away from base_rate (sharpening)
    # Optimize stretch_factor per cell to minimize Brier score
    regime_params = {}
    global_base_rate = float(np.mean(fit_y))

    # Iterate over all regime cells + fallbacks
    for mu_label in ["cold", "moderate", "hot"]:
        for vol_label in ["stable", "moderate_vol", "volatile"]:
            for season_label in ["DJF", "MAM", "JJA", "SON"]:
                cell_key = f"{mu_label}|{vol_label}|{season_label}"
                cell_mask = (fit_mu == mu_label) & (fit_vol == vol_label) & (fit_season == season_label)

                if cell_mask.sum() >= 30:
                    cell_probs = fit_probs[cell_mask]
                    cell_y = fit_y[cell_mask]
                    cell_base_rate = float(np.mean(cell_y))
                    # Optimize stretch_factor
                    best_sf = 1.0
                    best_brier = float(np.mean((cell_probs - cell_y) ** 2))
                    for sf in np.arange(0.5, 2.5, 0.05):
                        stretched = np.clip(cell_base_rate + sf * (cell_probs - cell_base_rate),
                                            PROB_CLIP_MIN, PROB_CLIP_MAX)
                        brier = float(np.mean((stretched - cell_y) ** 2))
                        if brier < best_brier:
                            best_brier = brier
                            best_sf = sf
                    regime_params[cell_key] = {"base_rate": cell_base_rate, "stretch_factor": float(best_sf)}
                # else: will use fallback hierarchy

    # Fallback: per-season
    season_params = {}
    for season_label in ["DJF", "MAM", "JJA", "SON"]:
        season_mask = fit_season == season_label
        if season_mask.sum() >= 30:
            cell_probs = fit_probs[season_mask]
            cell_y = fit_y[season_mask]
            cell_base_rate = float(np.mean(cell_y))
            best_sf = 1.0
            best_brier = float(np.mean((cell_probs - cell_y) ** 2))
            for sf in np.arange(0.5, 2.5, 0.05):
                stretched = np.clip(cell_base_rate + sf * (cell_probs - cell_base_rate),
                                    PROB_CLIP_MIN, PROB_CLIP_MAX)
                brier = float(np.mean((stretched - cell_y) ** 2))
                if brier < best_brier:
                    best_brier = brier
                    best_sf = sf
            season_params[season_label] = {"base_rate": cell_base_rate, "stretch_factor": float(best_sf)}

    # Fallback: per mu_level
    mu_params = {}
    for mu_label in ["cold", "moderate", "hot"]:
        mu_mask = fit_mu == mu_label
        if mu_mask.sum() >= 30:
            cell_probs = fit_probs[mu_mask]
            cell_y = fit_y[mu_mask]
            cell_base_rate = float(np.mean(cell_y))
            best_sf = 1.0
            best_brier = float(np.mean((cell_probs - cell_y) ** 2))
            for sf in np.arange(0.5, 2.5, 0.05):
                stretched = np.clip(cell_base_rate + sf * (cell_probs - cell_base_rate),
                                    PROB_CLIP_MIN, PROB_CLIP_MAX)
                brier = float(np.mean((stretched - cell_y) ** 2))
                if brier < best_brier:
                    best_brier = brier
                    best_sf = sf
            mu_params[mu_label] = {"base_rate": cell_base_rate, "stretch_factor": float(best_sf)}

    # Global fallback
    global_best_sf = 1.0
    global_best_brier = float(np.mean((fit_probs - fit_y) ** 2))
    for sf in np.arange(0.5, 2.5, 0.05):
        stretched = np.clip(global_base_rate + sf * (fit_probs - global_base_rate),
                            PROB_CLIP_MIN, PROB_CLIP_MAX)
        brier = float(np.mean((stretched - fit_y) ** 2))
        if brier < global_best_brier:
            global_best_brier = brier
            global_best_sf = sf
    global_params = {"base_rate": global_base_rate, "stretch_factor": float(global_best_sf)}

    # Apply stretching to isotonic split for post-calibration
    iso_probs = e13_probs[iso_idx]
    iso_y = y[iso_idx]
    iso_mu = calib_regime_mu[iso_idx]
    iso_vol = calib_regime_vol[iso_idx]
    iso_season = calib_regime_season[iso_idx]

    stretched_iso = iso_probs.copy()
    for j in range(len(iso_probs)):
        cell_key = f"{iso_mu[j]}|{iso_vol[j]}|{iso_season[j]}"
        if cell_key in regime_params:
            params = regime_params[cell_key]
        elif iso_season[j] in season_params:
            params = season_params[iso_season[j]]
        elif iso_mu[j] in mu_params:
            params = mu_params[iso_mu[j]]
        else:
            params = global_params
        stretched_iso[j] = np.clip(
            params["base_rate"] + params["stretch_factor"] * (iso_probs[j] - params["base_rate"]),
            PROB_CLIP_MIN, PROB_CLIP_MAX,
        )

    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(stretched_iso, iso_y)

    return {
        "mu_terciles": mu_terciles.tolist(),
        "vol_terciles": vol_terciles.tolist(),
        "regime_params": regime_params,
        "season_params": season_params,
        "mu_params": mu_params,
        "global_params": global_params,
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
    }


def _apply_e33(df: pd.DataFrame, e33: dict, cfg: dict) -> pd.DataFrame:
    """Apply E33 Temperature-Regime-Aware Synthesis with Resolution Boost."""
    out = df.copy()

    # Get E13 probabilities
    e13_df = e0e8._apply_variant(df, "E13_neural_synthesis_mlp", cfg)
    e13_probs = e13_df["model_prob"].values.copy()

    # Build date-level regime features
    by_date = out[["date", "date_dt", "model_mu"]].drop_duplicates("date").sort_values("date_dt").copy()
    by_date["mu_change"] = by_date["model_mu"].diff().abs().fillna(0.0)
    months = by_date["date_dt"].dt.month.values

    mu_terciles = np.array(e33["mu_terciles"])
    vol_terciles = np.array(e33["vol_terciles"])

    by_date["regime_mu"] = np.where(
        by_date["model_mu"].values < mu_terciles[0], "cold",
        np.where(by_date["model_mu"].values < mu_terciles[1], "moderate", "hot")
    )
    by_date["regime_vol"] = np.where(
        by_date["mu_change"].values < vol_terciles[0], "stable",
        np.where(by_date["mu_change"].values < vol_terciles[1], "moderate_vol", "volatile")
    )
    by_date["regime_season"] = _season_from_month(months)

    regime_map = by_date.set_index("date")[["regime_mu", "regime_vol", "regime_season"]].to_dict("index")

    # Apply stretching per contract
    stretched = e13_probs.copy()
    for j in range(len(e13_probs)):
        d = out["date"].values[j]
        info = regime_map.get(d, {})
        mu_label = info.get("regime_mu", "moderate")
        vol_label = info.get("regime_vol", "moderate_vol")
        season_label = info.get("regime_season", "MAM")

        cell_key = f"{mu_label}|{vol_label}|{season_label}"
        if cell_key in e33["regime_params"]:
            params = e33["regime_params"][cell_key]
        elif season_label in e33["season_params"]:
            params = e33["season_params"][season_label]
        elif mu_label in e33["mu_params"]:
            params = e33["mu_params"][mu_label]
        else:
            params = e33["global_params"]

        stretched[j] = np.clip(
            params["base_rate"] + params["stretch_factor"] * (e13_probs[j] - params["base_rate"]),
            PROB_CLIP_MIN, PROB_CLIP_MAX,
        )

    # Isotonic post-calibration
    iso_x = np.array(e33["isotonic_x"])
    iso_y = np.array(e33["isotonic_y"])
    calibrated = np.interp(np.clip(stretched, iso_x.min(), iso_x.max()), iso_x, iso_y)

    # Per-day renormalization
    date_vals = out["date"].values
    for d in np.unique(date_vals):
        mask = date_vals == d
        day_sum = calibrated[mask].sum()
        if day_sum > 0:
            calibrated[mask] = calibrated[mask] / day_sum

    out["model_prob"] = np.clip(calibrated, PROB_CLIP_MIN, PROB_CLIP_MAX)
    out["nws_prob"] = bench.compute_bucket_probs(out, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out[col] = out[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    return out


# ---------------------------------------------------------------------------
# 5. Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmark_suite(base_df, calib_label, calib_year_filter):
    """Run all variants for a given calibration window."""
    print(f"\n{'='*70}")
    print(f"Running benchmark suite: calibration={calib_label}")
    print(f"{'='*70}")

    # Fit calibration transforms using the specified calibration window
    print(f"  Fitting E0-E22 transforms on {calib_label} ...")
    try:
        cfg = e0e8._fit_experiment_transforms(base_df)
    except Exception as exc:
        print(f"  [ERROR] Failed to fit transforms: {exc}")
        # Fallback: fit with limited transforms
        cfg = {}

    all_metrics = []

    # Run E0-E22
    for variant in VARIANT_LIST:
        print(f"  Running {variant} ...")
        m = run_variant_with_metrics(base_df, variant, cfg, calib_label)
        all_metrics.append(m)

    # E23: regime sigma
    print("  Running E23_regime_sigma ...")
    m23 = apply_sigma_variant_and_compute(base_df, "E23_regime_sigma", "model_sigma_regime_cal", calib_label)
    all_metrics.append(m23)

    # E24: combined (month x regime) sigma
    print("  Running E24_combined_sigma ...")
    m24 = apply_sigma_variant_and_compute(base_df, "E24_combined_sigma", "model_sigma_combined_cal", calib_label)
    all_metrics.append(m24)

    # E25: regime sigma + Platt recalibration
    print("  Running E25_regime_sigma_platt ...")
    m25 = apply_sigma_regime_platt(base_df, cfg, calib_label)
    all_metrics.append(m25)

    # E26: Tail-Weighted Brier Synthesis
    print("  Fitting + Running E26_tail_weighted_brier_synthesis ...")
    try:
        e26_cfg = _fit_e26_tail_weighted_brier_synthesis(base_df, cfg)
        out_e26 = _apply_e26(base_df, e26_cfg)
        m26 = compute_comprehensive_metrics(out_e26, f"E26_tail_weighted_brier_synthesis_{calib_label}")
        all_metrics.append(m26)
    except Exception as exc:
        print(f"  [WARN] E26 failed: {exc}")
        all_metrics.append({"variant": f"E26_tail_weighted_brier_synthesis_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E27: Conformal Prediction Overlay
    print("  Fitting + Running E27_conformal_prediction ...")
    try:
        e27_cfg = _fit_e27_conformal_prediction(base_df, cfg)
        out_e27 = _apply_e27(base_df, e27_cfg, cfg)
        m27 = compute_comprehensive_metrics(out_e27, f"E27_conformal_prediction_{calib_label}")
        all_metrics.append(m27)
    except Exception as exc:
        print(f"  [WARN] E27 failed: {exc}")
        all_metrics.append({"variant": f"E27_conformal_prediction_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E28: Ensemble Disagreement Sharpening
    print("  Fitting + Running E28_ensemble_disagreement ...")
    try:
        e28_cfg = _fit_e28_ensemble_disagreement(base_df, cfg)
        out_e28 = _apply_e28(base_df, e28_cfg, cfg)
        m28 = compute_comprehensive_metrics(out_e28, f"E28_ensemble_disagreement_{calib_label}")
        all_metrics.append(m28)
    except Exception as exc:
        print(f"  [WARN] E28 failed: {exc}")
        all_metrics.append({"variant": f"E28_ensemble_disagreement_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E29: Learned Heteroscedastic Sigma
    print("  Fitting + Running E29_learned_sigma ...")
    try:
        e29_cfg = _fit_e29_learned_sigma(base_df)
        out_e29 = _apply_e29(base_df, e29_cfg)
        m29 = compute_comprehensive_metrics(out_e29, f"E29_learned_sigma_{calib_label}")
        all_metrics.append(m29)
    except Exception as exc:
        print(f"  [WARN] E29 failed: {exc}")
        all_metrics.append({"variant": f"E29_learned_sigma_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E30: Aggressive Conformal + Neural Sharpener
    print("  Fitting + Running E30_conformal_neural_sharpener ...")
    try:
        e30_cfg = _fit_e30_conformal_neural_sharpener(base_df, cfg)
        out_e30 = _apply_e30(base_df, e30_cfg, cfg)
        m30 = compute_comprehensive_metrics(out_e30, f"E30_conformal_neural_sharpener_{calib_label}")
        all_metrics.append(m30)
    except Exception as exc:
        print(f"  [WARN] E30 failed: {exc}")
        import traceback; traceback.print_exc()
        all_metrics.append({"variant": f"E30_conformal_neural_sharpener_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E31: Quantile-Crossing-Penalized Synthesis
    print("  Fitting + Running E31_quantile_crossing_synthesis ...")
    try:
        e31_cfg = _fit_e31_quantile_crossing_synthesis(base_df, cfg)
        out_e31 = _apply_e31(base_df, e31_cfg, cfg)
        m31 = compute_comprehensive_metrics(out_e31, f"E31_quantile_crossing_synthesis_{calib_label}")
        all_metrics.append(m31)
    except Exception as exc:
        print(f"  [WARN] E31 failed: {exc}")
        import traceback; traceback.print_exc()
        all_metrics.append({"variant": f"E31_quantile_crossing_synthesis_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E32: Platt-Calibrated E17 + Conformal
    print("  Fitting + Running E32_platt_conformal_e17 ...")
    try:
        e32_cfg = _fit_e32_platt_conformal_e17(base_df, cfg)
        out_e32 = _apply_e32(base_df, e32_cfg, cfg)
        m32 = compute_comprehensive_metrics(out_e32, f"E32_platt_conformal_e17_{calib_label}")
        all_metrics.append(m32)
    except Exception as exc:
        print(f"  [WARN] E32 failed: {exc}")
        import traceback; traceback.print_exc()
        all_metrics.append({"variant": f"E32_platt_conformal_e17_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    # E33: Temperature-Regime-Aware Synthesis with Resolution Boost
    print("  Fitting + Running E33_regime_resolution_boost ...")
    try:
        e33_cfg = _fit_e33_regime_aware_resolution_boost(base_df, cfg)
        out_e33 = _apply_e33(base_df, e33_cfg, cfg)
        m33 = compute_comprehensive_metrics(out_e33, f"E33_regime_resolution_boost_{calib_label}")
        all_metrics.append(m33)
    except Exception as exc:
        print(f"  [WARN] E33 failed: {exc}")
        import traceback; traceback.print_exc()
        all_metrics.append({"variant": f"E33_regime_resolution_boost_{calib_label}",
                            "overall_brier": None, "oos_brier": None, "error": str(exc)})

    return all_metrics


def main():
    print("=" * 70)
    print("Extended Validation Model Benchmark: E0-E33")
    print("=" * 70)

    # Load base dataset with retrain model predictions
    base_df = load_base_dataset_with_retrain()

    # Refresh experiment summary (needed by e0e8 transforms)
    print("\nRefreshing probabilistic experiment summary ...")
    try:
        exp.run()
    except Exception as exc:
        print(f"  [WARN] exp.run() failed: {exc}, continuing without refresh ...")

    # ---- Window A: Calibration on 2023 only (same as original benchmark) ----
    metrics_cal2023 = run_benchmark_suite(base_df, "cal2023", 2023)

    # ---- Window B: Calibration on 2023-2024 ----
    # For this, we need to re-fit the transforms with 2023-2024 as calibration
    # The e0e8 fitting functions use year==2023 by default internally.
    # We'll create a modified base_df where we relabel periods so the fit
    # functions see 2023-2024 as "calibration" year.
    # Actually, the simplest approach: modify the base_df to pretend
    # dates in [2023, 2024] are all "year 2023" for fitting, but keep true period labels.
    # However that's hacky. Instead, let's manually run the variants with a
    # wider calibration window by calling the fit functions directly.

    # For Window B, we re-run only E0 (baseline), E1, E23-E25 with the wider cal window,
    # since re-fitting all stacker/neural models requires the full pipeline and
    # the calibrators would not recognize a different year.
    print(f"\n{'='*70}")
    print("Running Window B analysis: cal2023_2024 (E0 + sigma variants)")
    print(f"{'='*70}")

    metrics_cal2024 = []

    # E0 baseline with base sigma (same model, no recalibration needed)
    out_e0 = base_df.copy()
    out_e0["model_prob"] = bench.compute_bucket_probs(out_e0, "model_mu", "model_sigma")
    out_e0["nws_prob"] = bench.compute_bucket_probs(out_e0, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out_e0[col] = out_e0[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    m_e0_b = compute_comprehensive_metrics(out_e0, "E0_baseline_ensemble_cal2023_2024")
    metrics_cal2024.append(m_e0_b)

    # E1: global isotonic calibrated on 2023-2024
    cal_df_wide = base_df[base_df["date_dt"].dt.year.isin([2023, 2024])][
        ["date_dt", "model_mu", "model_sigma", "actual_tmax"]
    ].drop_duplicates("date_dt")
    global_cal_wide = e012._calibrate_global(
        cal_df_wide["model_mu"].values,
        cal_df_wide["model_sigma"].values,
        cal_df_wide["actual_tmax"].values,
    )
    out_e1 = base_df.copy()
    mu = out_e1["model_mu"].values
    sigma = out_e1["model_sigma"].values
    f_lo = np.where(np.isnan(out_e1["threshold_low"].values), 0.0,
                    e012._cdf(out_e1["threshold_low"].values, mu, sigma))
    f_hi = np.where(np.isnan(out_e1["threshold_high"].values), 1.0,
                    e012._cdf(out_e1["threshold_high"].values, mu, sigma))
    lo_cal = np.where(np.isnan(out_e1["threshold_low"].values), 0.0,
                      np.clip(global_cal_wide.predict(f_lo), 1e-6, 1 - 1e-6))
    hi_cal = np.where(np.isnan(out_e1["threshold_high"].values), 1.0,
                      np.clip(global_cal_wide.predict(f_hi), 1e-6, 1 - 1e-6))
    out_e1["model_prob"] = np.clip(hi_cal - lo_cal, 1e-6, 1.0)
    out_e1["nws_prob"] = bench.compute_bucket_probs(out_e1, "nws_mu", "nws_sigma")
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        out_e1[col] = out_e1[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    m_e1_b = compute_comprehensive_metrics(out_e1, "E1_global_isotonic_cal2023_2024")
    metrics_cal2024.append(m_e1_b)

    # E23-E25 with wider calibration window
    print("  Running E23_regime_sigma_cal2023_2024 ...")
    m23b = apply_sigma_variant_and_compute(base_df, "E23_regime_sigma", "model_sigma_regime_cal", "cal2023_2024")
    metrics_cal2024.append(m23b)

    print("  Running E24_combined_sigma_cal2023_2024 ...")
    m24b = apply_sigma_variant_and_compute(base_df, "E24_combined_sigma", "model_sigma_combined_cal", "cal2023_2024")
    metrics_cal2024.append(m24b)

    print("  Running E25_regime_sigma_platt_cal2023_2024 ...")
    m25b = apply_sigma_regime_platt(base_df, {}, "cal2023_2024")
    metrics_cal2024.append(m25b)

    # ---- Combine all metrics ----
    all_metrics = metrics_cal2023 + metrics_cal2024

    # Build summary DataFrame
    summary_rows = []
    for m in all_metrics:
        if m.get("overall_brier") is None:
            continue
        row = {
            "model": m["variant"],
            "overall_model_brier": m["overall_brier"],
            "is_model_brier": m.get("is_brier"),
            "oos_model_brier": m.get("oos_brier"),
            "overall_log_score": m.get("overall_log_score"),
            "oos_log_score": m.get("oos_log_score"),
            "overall_ece": m.get("overall_ece"),
            "oos_ece": m.get("oos_ece"),
            "overall_presettlement_brier": m.get("presettlement_brier_overall"),
            "oos_presettlement_brier": m.get("presettlement_brier_oos"),
            "overall_nws_brier": m.get("nws_brier_overall"),
            "oos_nws_brier": m.get("nws_brier_oos"),
            "brier_reliability": m.get("brier_decomposition", {}).get("reliability"),
            "brier_resolution": m.get("brier_decomposition", {}).get("resolution"),
            "brier_uncertainty": m.get("brier_decomposition", {}).get("uncertainty"),
            "n_total": m.get("n_total"),
            "n_oos": m.get("n_oos"),
        }
        # Season brier
        for s in ["DJF", "MAM", "JJA", "SON"]:
            row[f"brier_{s}"] = m.get("season_brier", {}).get(s)
        # Direction brier
        for d in ["above", "below", "between"]:
            row[f"brier_{d}"] = m.get("direction_brier", {}).get(d)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if len(summary) > 0:
        summary = summary.sort_values("oos_model_brier", na_position="last").reset_index(drop=True)

    # Save summary
    summary.to_csv(OUT_ROOT / "benchmark_summary.csv", index=False)
    print(f"\nSaved benchmark_summary.csv ({len(summary)} variants)")

    # Save calibration detail for top variants
    cal_detail = {}
    for m in all_metrics:
        if m.get("overall_brier") is not None and m.get("oos_brier") is not None:
            cal_detail[m["variant"]] = {
                "reliability_diagram": m.get("reliability_diagram"),
                "brier_decomposition": m.get("brier_decomposition"),
                "season_brier": m.get("season_brier"),
                "direction_brier": m.get("direction_brier"),
            }

    with open(OUT_ROOT / "calibration_detail.json", "w") as f:
        json.dump(cal_detail, f, indent=2)

    # Paper trading gate report (lightweight version)
    if len(summary) > 0:
        top = summary.iloc[0]
        oos_model = float(top["oos_model_brier"]) if pd.notna(top["oos_model_brier"]) else None
        oos_pre = float(top["oos_presettlement_brier"]) if pd.notna(top["oos_presettlement_brier"]) else None
        oos_ece_val = float(top["oos_ece"]) if pd.notna(top["oos_ece"]) else None
        gate_report = {
            "top_model": str(top["model"]),
            "oos_model_brier": oos_model,
            "oos_presettlement_brier": oos_pre,
            "overall_presettlement_brier": float(top["overall_presettlement_brier"]) if pd.notna(top["overall_presettlement_brier"]) else None,
            "oos_ece": oos_ece_val,
            "checks": {
                "oos_brier_beats_presettlement": bool(
                    oos_model is not None and oos_pre is not None
                    and oos_model <= oos_pre
                ),
                "overall_brier_beats_presettlement": bool(
                    pd.notna(top["overall_model_brier"]) and pd.notna(top["overall_presettlement_brier"])
                    and top["overall_model_brier"] <= top["overall_presettlement_brier"]
                ),
                "ece_below_threshold": bool(oos_ece_val is not None and oos_ece_val <= 0.03),
            }
        }
        with open(OUT_ROOT / "paper_trading_gate_report.json", "w") as f:
            json.dump(gate_report, f, indent=2)

    # Quality metrics report
    _write_quality_report(summary, all_metrics)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 10 VARIANTS BY OOS BRIER SCORE")
    print("=" * 70)
    if len(summary) > 0:
        top10 = summary.head(10)[["model", "overall_model_brier", "oos_model_brier",
                                   "overall_ece", "oos_ece",
                                   "overall_presettlement_brier"]].copy()
        # Format to 6 decimal places
        for col in top10.columns:
            if col != "model":
                top10[col] = top10[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
        print(top10.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_ROOT}")
    return summary


def _write_quality_report(summary: pd.DataFrame, all_metrics: list):
    """Write a comprehensive markdown quality metrics report."""
    lines = []
    lines.append("# Extended Validation Model Benchmark Report")
    lines.append(f"\nGenerated with retrained model from results/retrain_extended_validation/")
    lines.append(f"Calibration windows: cal2023 (2023 only), cal2023_2024 (2023-2024)")
    lines.append("")

    if len(summary) == 0:
        lines.append("No valid results.")
        with open(OUT_ROOT / "quality_metrics_report.md", "w") as f:
            f.write("\n".join(lines))
        return

    # Reference benchmarks
    pre_brier = summary["overall_presettlement_brier"].iloc[0]
    pre_oos = summary["oos_presettlement_brier"].iloc[0]
    nws_brier = summary["overall_nws_brier"].iloc[0]
    nws_oos = summary["oos_nws_brier"].iloc[0]

    lines.append("## Reference Benchmarks")
    lines.append(f"| Source | Overall Brier | OOS Brier |")
    lines.append(f"|--------|--------------|-----------|")
    lines.append(f"| Kalshi PreSettlement | {pre_brier:.6f} | {pre_oos:.6f} |")
    lines.append(f"| NWS | {nws_brier:.6f} | {nws_oos:.6f} |")
    lines.append("")

    # Top variants
    lines.append("## Top 15 Variants by OOS Brier")
    lines.append(f"| Rank | Variant | Overall Brier | OOS Brier | ECE | OOS ECE |")
    lines.append(f"|------|---------|--------------|-----------|-----|---------|")
    for i, row in summary.head(15).iterrows():
        oos_b = f"{row['oos_model_brier']:.6f}" if pd.notna(row["oos_model_brier"]) else "N/A"
        ov_b = f"{row['overall_model_brier']:.6f}" if pd.notna(row["overall_model_brier"]) else "N/A"
        ece = f"{row['overall_ece']:.6f}" if pd.notna(row["overall_ece"]) else "N/A"
        oos_ece = f"{row['oos_ece']:.6f}" if pd.notna(row["oos_ece"]) else "N/A"
        lines.append(f"| {i+1} | {row['model']} | {ov_b} | {oos_b} | {ece} | {oos_ece} |")
    lines.append("")

    # Brier decomposition for top 5
    lines.append("## Brier Decomposition (Top 5)")
    lines.append("| Variant | Brier | Reliability | Resolution | Uncertainty |")
    lines.append("|---------|-------|------------|------------|-------------|")
    for _, row in summary.head(5).iterrows():
        rel = f"{row['brier_reliability']:.6f}" if pd.notna(row["brier_reliability"]) else "N/A"
        res = f"{row['brier_resolution']:.6f}" if pd.notna(row["brier_resolution"]) else "N/A"
        unc = f"{row['brier_uncertainty']:.6f}" if pd.notna(row["brier_uncertainty"]) else "N/A"
        brier = f"{row['overall_model_brier']:.6f}" if pd.notna(row["overall_model_brier"]) else "N/A"
        lines.append(f"| {row['model']} | {brier} | {rel} | {res} | {unc} |")
    lines.append("")

    # Seasonal analysis for top 3
    lines.append("## Seasonal Brier (Top 3)")
    lines.append("| Variant | DJF | MAM | JJA | SON |")
    lines.append("|---------|-----|-----|-----|-----|")
    for _, row in summary.head(3).iterrows():
        djf = f"{row['brier_DJF']:.6f}" if pd.notna(row.get("brier_DJF")) else "N/A"
        mam = f"{row['brier_MAM']:.6f}" if pd.notna(row.get("brier_MAM")) else "N/A"
        jja = f"{row['brier_JJA']:.6f}" if pd.notna(row.get("brier_JJA")) else "N/A"
        son = f"{row['brier_SON']:.6f}" if pd.notna(row.get("brier_SON")) else "N/A"
        lines.append(f"| {row['model']} | {djf} | {mam} | {jja} | {son} |")
    lines.append("")

    # New variants comparison (E23-E29)
    sigma_variants = summary[summary["model"].str.contains("E23|E24|E25|E26|E27|E28|E29|E30|E31|E32|E33")]
    if len(sigma_variants) > 0:
        lines.append("## New Variants (E23-E33)")
        lines.append("| Variant | Overall Brier | OOS Brier | ECE |")
        lines.append("|---------|--------------|-----------|-----|")
        for _, row in sigma_variants.iterrows():
            oos_b = f"{row['oos_model_brier']:.6f}" if pd.notna(row["oos_model_brier"]) else "N/A"
            ov_b = f"{row['overall_model_brier']:.6f}" if pd.notna(row["overall_model_brier"]) else "N/A"
            ece = f"{row['overall_ece']:.6f}" if pd.notna(row["overall_ece"]) else "N/A"
            lines.append(f"| {row['model']} | {ov_b} | {oos_b} | {ece} |")
        lines.append("")

    # vs Previous best
    lines.append("## Model vs Benchmarks Summary")
    best = summary.iloc[0]
    lines.append(f"- **Best variant**: {best['model']}")
    if pd.notna(best["overall_model_brier"]):
        lines.append(f"- **Overall Brier**: {best['overall_model_brier']:.6f}")
    if pd.notna(best["oos_model_brier"]):
        lines.append(f"- **OOS Brier**: {best['oos_model_brier']:.6f}")
    if pd.notna(best["oos_model_brier"]) and pd.notna(pre_oos):
        delta = best["oos_model_brier"] - pre_oos
        lines.append(f"- **vs Kalshi PreSettlement OOS** ({pre_oos:.6f}): {delta:+.6f} ({'better' if delta < 0 else 'worse'})")
    if pd.notna(best["overall_model_brier"]) and pd.notna(pre_brier):
        delta = best["overall_model_brier"] - pre_brier
        lines.append(f"- **vs Kalshi PreSettlement Overall** ({pre_brier:.6f}): {delta:+.6f} ({'better' if delta < 0 else 'worse'})")
    if pd.notna(best["oos_model_brier"]) and pd.notna(nws_oos):
        delta = best["oos_model_brier"] - nws_oos
        lines.append(f"- **vs NWS OOS** ({nws_oos:.6f}): {delta:+.6f} ({'better' if delta < 0 else 'worse'})")
    if pd.notna(best["overall_model_brier"]) and pd.notna(nws_brier):
        delta = best["overall_model_brier"] - nws_brier
        lines.append(f"- **vs NWS Overall** ({nws_brier:.6f}): {delta:+.6f} ({'better' if delta < 0 else 'worse'})")

    with open(OUT_ROOT / "quality_metrics_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved quality_metrics_report.md")


if __name__ == "__main__":
    main()
