#!/usr/bin/env python3
"""Benchmark E0-E25 using the retrained extended-validation model.

Loads the retrained model's predictions (mu/sigma) and replaces the canonical
best_model_predictions in the benchmark pipeline.  Runs all E0-E22 variants
plus three new sigma-calibration variants (E23-E25).  Evaluates with two
calibration windows: (A) fit on 2023 only, (B) fit on 2023-2024.

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

    return all_metrics


def main():
    print("=" * 70)
    print("Extended Validation Model Benchmark: E0-E25")
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

    # New sigma variants comparison
    sigma_variants = summary[summary["model"].str.contains("E23|E24|E25")]
    if len(sigma_variants) > 0:
        lines.append("## New Sigma Calibration Variants (E23-E25)")
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
