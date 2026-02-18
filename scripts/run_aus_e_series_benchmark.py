#!/usr/bin/env python3
"""
Austin E-Series Model Benchmark.

Applies E-series post-processing transformations to base model predictions
(Persistence, Climatology, Ridge, HeteroscedasticNN) and evaluates against
real Kalshi pre-settlement data.

E-series models (calibration transformations):
  E0:  Base HeteroscedasticNN (raw)
  E1:  NN + isotonic mu calibration
  E2:  NN + optimal sigma scaling
  E3:  NN + global bias correction
  E4:  NN + seasonal bias correction
  E5:  NN + isotonic mu + sigma scaling
  E6:  NN + isotonic mu + seasonal bias + sigma scaling
  E7:  Ridge base (raw)
  E8:  Ridge + isotonic mu calibration
  E9:  Ridge + optimal sigma scaling
  E10: Ridge + isotonic mu + sigma scaling
  E11: Ridge + seasonal bias + sigma scaling
  E12: Ensemble (Ridge + NN) average
  E13: Ensemble + isotonic calibration
  E14: Ensemble + seasonal bias + sigma scaling
  E15: Conditional calibration by sigma regime
  E16: Platt scaling on contract probabilities
  E17: Contract-Brier-optimized sigma search
  E18: CRPS-optimized sigma
  E19: Neural MLP contract calibration
  E20: Best single model + Platt per-contract calibration

All calibrations are fit on VALIDATION set only and evaluated on TEST set
(chronological, no leakage). Contract Brier is computed on real Kalshi
pre-settlement rows only.

Usage:
    python scripts/run_aus_e_series_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4

SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}


# ===========================================================================
# Data Loading
# ===========================================================================

def load_base_predictions(results_dir: str) -> pd.DataFrame:
    """Load base model predictions from the benchmark run."""
    path = os.path.join(results_dir, "base_predictions.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_kalshi_data(project_root: Path):
    """Load real Kalshi settlement + presettlement data for Austin."""
    settlement_path = project_root / "data" / "real_kalshi_aus_all.csv"
    presettlement_path = project_root / "data" / "kalshi_presettlement_aus.csv"

    if not settlement_path.exists():
        raise FileNotFoundError(f"No settlement data at {settlement_path}")

    settled = pd.read_csv(settlement_path)
    settled["date"] = pd.to_datetime(settled["date"]).dt.strftime("%Y-%m-%d")

    if presettlement_path.exists():
        pre = pd.read_csv(presettlement_path)
        pre["date"] = pd.to_datetime(pre["date"]).dt.strftime("%Y-%m-%d")
        merged = settled.merge(
            pre[["date", "ticker", "presettlement_prob", "bid_cents",
                 "ask_cents", "volume", "open_interest"]].drop_duplicates(
                     subset=["date", "ticker"]),
            on=["date", "ticker"], how="inner", suffixes=("", "_pre"),
        )
        merged = merged.dropna(subset=["presettlement_prob"])
        merged["market_prob"] = merged["presettlement_prob"].clip(
            PROB_CLIP_MIN, PROB_CLIP_MAX)
        logger.info("Loaded real Kalshi AUS data: %d rows, %d dates",
                     len(merged), merged["date"].nunique())
        return merged

    logger.info("Loaded Kalshi AUS (settlement only): %d rows", len(settled))
    return settled


# ===========================================================================
# Contract Brier Evaluation
# ===========================================================================

def contract_brier(probs, outcomes):
    """Compute contract-level Brier score."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(o))
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((p[valid] - o[valid]) ** 2))


def build_contract_dataset(kalshi_df, mu_by_date, sigma_by_date):
    """Map model (mu, sigma) to Kalshi contract-level probabilities."""
    df = kalshi_df.copy()
    df["model_mu"] = df["date"].map(mu_by_date)
    df["model_sigma"] = df["date"].map(sigma_by_date)
    df = df.dropna(subset=["model_mu", "model_sigma"])
    if len(df) == 0:
        return df

    mu = df["model_mu"].values
    sigma = np.maximum(df["model_sigma"].values, 0.5)
    th_low = df["threshold_low"].values.astype(float)
    th_high = df["threshold_high"].values.astype(float)
    direction = df["direction"].values

    model_prob = np.full(len(df), np.nan)
    below = (direction == "below") | (direction == "less")
    above = direction == "above"
    between = direction == "between"

    if below.any():
        model_prob[below] = norm.cdf(th_high[below], mu[below], sigma[below])
    if above.any():
        model_prob[above] = 1.0 - norm.cdf(th_low[above], mu[above], sigma[above])
    if between.any():
        model_prob[between] = (
            norm.cdf(th_high[between], mu[between], sigma[between])
            - norm.cdf(th_low[between], mu[between], sigma[between])
        )

    df["model_prob"] = np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return df


def eval_model(kalshi_df, mu_dict, sigma_dict, test_dates=None):
    """Evaluate a model and return contract Brier + seasonal breakdown."""
    cdf = build_contract_dataset(kalshi_df, mu_dict, sigma_dict)
    if len(cdf) == 0:
        return float("nan"), {}, 0

    # Filter to test dates if provided
    if test_dates is not None:
        test_date_strs = set(d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else d
                             for d in test_dates)
        cdf = cdf[cdf["date"].isin(test_date_strs)]

    if len(cdf) == 0:
        return float("nan"), {}, 0

    brier = contract_brier(cdf["model_prob"].values, cdf["actual_outcome"].values)

    # Seasonal breakdown
    cdf["date_dt"] = pd.to_datetime(cdf["date"])
    cdf["month"] = cdf["date_dt"].dt.month
    cdf["season"] = cdf["month"].map(SEASON_MAP)
    seasonal = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = cdf["season"] == s
        if mask.any():
            seasonal[s] = contract_brier(
                cdf.loc[mask, "model_prob"].values,
                cdf.loc[mask, "actual_outcome"].values)

    return brier, seasonal, cdf["date"].nunique()


def eval_market_brier(kalshi_df, test_dates=None):
    """Compute Kalshi market Brier score from presettlement probabilities."""
    df = kalshi_df.copy()
    if test_dates is not None:
        test_date_strs = set(d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else d
                             for d in test_dates)
        df = df[df["date"].isin(test_date_strs)]

    if "market_prob" not in df.columns or len(df) == 0:
        return float("nan"), {}

    brier = contract_brier(df["market_prob"].values, df["actual_outcome"].values)

    df["date_dt"] = pd.to_datetime(df["date"])
    df["month"] = df["date_dt"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    seasonal = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = df["season"] == s
        if mask.any():
            seasonal[s] = contract_brier(
                df.loc[mask, "market_prob"].values,
                df.loc[mask, "actual_outcome"].values)

    return brier, seasonal


# ===========================================================================
# Calibration Transforms
# ===========================================================================

def fit_isotonic_mu(mu_val, y_val):
    """Fit isotonic regression on validation mu vs actual."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(mu_val, y_val)
    return ir


def apply_isotonic_mu(ir, mu):
    """Apply isotonic calibration to mu predictions."""
    return ir.predict(mu)


def fit_sigma_scaler(mu_val, sigma_val, y_val, kalshi_val, bucket_edges, dates=None):
    """Find optimal sigma multiplier that minimizes contract Brier on validation."""
    # Build date strings for mapping
    if dates is not None:
        date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates]
    elif hasattr(y_val, 'index'):
        date_strs = [d.strftime("%Y-%m-%d") for d in y_val.index]
    else:
        # Fallback: just use kalshi dates directly via mu_dict approach
        date_strs = list(kalshi_val["date"].unique())[:len(mu_val)]

    def objective(mult):
        mu_d = {}
        sig_d = {}
        for m, s, d in zip(mu_val, sigma_val * mult, date_strs):
            mu_d[d] = m
            sig_d[d] = s
        cdf = build_contract_dataset(kalshi_val, mu_d, sig_d)
        if len(cdf) == 0:
            return 1.0
        return contract_brier(cdf["model_prob"].values, cdf["actual_outcome"].values)

    result = minimize_scalar(objective, bounds=(0.3, 3.0), method="bounded")
    return result.x


def fit_global_bias(mu_val, y_val):
    """Compute global bias correction (mean residual)."""
    resid = y_val - mu_val
    return float(np.nanmean(resid))


def fit_seasonal_bias(mu_val, y_val, dates_val):
    """Compute seasonal bias correction."""
    if hasattr(dates_val, 'dt'):
        months = dates_val.dt.month
    else:
        months = dates_val.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    bias = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == s
        if mask.any():
            bias[s] = float(np.nanmean(y_val[mask] - mu_val[mask]))
        else:
            bias[s] = 0.0
    return bias


def apply_seasonal_bias(mu, dates, bias_dict):
    """Apply seasonal bias correction."""
    if hasattr(dates, 'dt'):
        months = dates.dt.month
    else:
        months = dates.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    corrected = mu.copy()
    for s, b in bias_dict.items():
        mask = seasons == s
        corrected[mask] += b
    return corrected


def fit_conditional_sigma(mu_val, sigma_val, y_val):
    """Fit sigma scaling conditioned on uncertainty regime (high/low sigma)."""
    median_sigma = np.median(sigma_val)
    low_mask = sigma_val <= median_sigma
    high_mask = sigma_val > median_sigma

    # For each regime, find optimal sigma multiplier
    def _fit_mult(mask):
        resid = y_val[mask] - mu_val[mask]
        actual_std = float(np.std(resid))
        pred_std = float(np.mean(sigma_val[mask]))
        if pred_std > 0:
            return actual_std / pred_std
        return 1.0

    return {
        "median_sigma": median_sigma,
        "low_mult": _fit_mult(low_mask),
        "high_mult": _fit_mult(high_mask),
    }


def apply_conditional_sigma(sigma, cond_cal):
    """Apply conditional sigma scaling."""
    result = sigma.copy()
    low_mask = sigma <= cond_cal["median_sigma"]
    high_mask = sigma > cond_cal["median_sigma"]
    result[low_mask] *= cond_cal["low_mult"]
    result[high_mask] *= cond_cal["high_mult"]
    return result


def fit_crps_sigma(mu_val, sigma_val, y_val):
    """Find sigma multiplier minimizing CRPS on validation set."""
    def crps_gaussian(mu, sigma, y):
        z = (y - mu) / sigma
        return np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))

    def objective(mult):
        return crps_gaussian(mu_val, sigma_val * mult, y_val)

    result = minimize_scalar(objective, bounds=(0.3, 3.0), method="bounded")
    return result.x


def fit_platt_scaling(kalshi_val, mu_val_dict, sigma_val_dict):
    """Fit Platt scaling (logistic regression) on contract probabilities."""
    cdf = build_contract_dataset(kalshi_val, mu_val_dict, sigma_val_dict)
    if len(cdf) == 0:
        return None
    X = cdf["model_prob"].values.reshape(-1, 1)
    y = cdf["actual_outcome"].values
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    return lr


def apply_platt_scaling(lr, contract_df):
    """Apply Platt scaling to contract probabilities."""
    if lr is None:
        return contract_df
    df = contract_df.copy()
    X = df["model_prob"].values.reshape(-1, 1)
    df["model_prob"] = np.clip(lr.predict_proba(X)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    return df


# ===========================================================================
# E-Series Model Definitions
# ===========================================================================

def run_e_series(base_preds, kalshi_df, cal_dates, eval_dates, bucket_edges):
    """Run all E-series models and return results.

    Calibration is fit on cal_dates (first half of Kalshi overlap).
    Evaluation is on eval_dates (second half of Kalshi overlap).
    For mu/sigma transforms that don't need Kalshi data, we use all
    base prediction dates before the eval period for fitting.
    """

    results = {}

    # Extract base model predictions
    nn_preds = base_preds[base_preds["model_name"] == "HeteroscedasticNN"].copy()
    ridge_preds = base_preds[base_preds["model_name"] == "Ridge"].copy()
    persist_preds = base_preds[base_preds["model_name"] == "Persistence"].copy()
    clim_preds = base_preds[base_preds["model_name"] == "Climatology"].copy()

    # Split into cal and eval for each model
    def split_model(preds, cal_dates_set, eval_dates_set):
        date_strs = preds["date"].dt.strftime("%Y-%m-%d") if hasattr(preds["date"].iloc[0], 'strftime') else preds["date"].astype(str)
        cal = preds[date_strs.isin(cal_dates_set)].sort_values("date")
        evl = preds[date_strs.isin(eval_dates_set)].sort_values("date")
        return cal, evl

    cal_date_strs = set(str(d) if not isinstance(d, str) else d for d in cal_dates)
    eval_date_strs = set(str(d) if not isinstance(d, str) else d for d in eval_dates)

    # Also get all pre-eval dates for mu/sigma calibration (uses actual_tmax, no Kalshi needed)
    eval_start = min(eval_date_strs)
    all_pre_eval = base_preds[base_preds["date"].dt.strftime("%Y-%m-%d") < eval_start]

    nn_cal, nn_eval = split_model(nn_preds, cal_date_strs, eval_date_strs)
    ridge_cal, ridge_eval = split_model(ridge_preds, cal_date_strs, eval_date_strs)
    persist_cal, persist_eval = split_model(persist_preds, cal_date_strs, eval_date_strs)
    clim_cal, clim_eval = split_model(clim_preds, cal_date_strs, eval_date_strs)

    # For mu/sigma transforms, use ALL pre-eval data (much more data)
    nn_pre = all_pre_eval[all_pre_eval["model_name"] == "HeteroscedasticNN"].sort_values("date")
    ridge_pre = all_pre_eval[all_pre_eval["model_name"] == "Ridge"].sort_values("date")

    # Get Kalshi rows for cal and eval periods
    kalshi_cal = kalshi_df[kalshi_df["date"].isin(cal_date_strs)]
    kalshi_eval = kalshi_df[kalshi_df["date"].isin(eval_date_strs)]

    logger.info("Cal rows: NN=%d, Ridge=%d, Kalshi=%d (%d dates)",
                len(nn_cal), len(ridge_cal), len(kalshi_cal), len(cal_date_strs))
    logger.info("Eval rows: NN=%d, Ridge=%d, Kalshi=%d (%d dates)",
                len(nn_eval), len(ridge_eval), len(kalshi_eval), len(eval_date_strs))
    logger.info("Pre-eval (all): NN=%d, Ridge=%d", len(nn_pre), len(ridge_pre))

    def _make_dicts(df):
        mu_d = dict(zip(df["date"].astype(str), df["mu"]))
        sig_d = dict(zip(df["date"].astype(str), df["sigma"]))
        return mu_d, sig_d

    # Helper to evaluate on eval only
    def _eval(mu_d, sig_d, name):
        brier, seasonal, n_dates = eval_model(kalshi_eval, mu_d, sig_d)
        results[name] = {"brier": brier, "seasonal": seasonal, "n_dates": n_dates}
        logger.info("  %s: contract Brier=%.4f (%d dates)", name, brier, n_dates)
        return brier

    # ===== E0: Base NN =====
    nn_eval_mu_d, nn_eval_sig_d = _make_dicts(nn_eval)
    _eval(nn_eval_mu_d, nn_eval_sig_d, "E0_base_nn")

    # ===== E1: NN + Isotonic mu =====
    # Use all pre-eval data for fitting isotonic (more data = better fit)
    ir_nn = fit_isotonic_mu(nn_pre["mu"].values, nn_pre["actual_tmax"].values)
    e1_mu = apply_isotonic_mu(ir_nn, nn_eval["mu"].values)
    e1_mu_d = dict(zip(nn_eval["date"].astype(str), e1_mu))
    _eval(e1_mu_d, nn_eval_sig_d, "E1_nn_isotonic")

    # ===== E2: NN + Sigma scaling =====
    nn_cal_mu_d, nn_cal_sig_d = _make_dicts(nn_cal)
    sigma_mult_nn = fit_sigma_scaler(
        nn_cal["mu"].values, nn_cal["sigma"].values,
        nn_cal["actual_tmax"].values, kalshi_cal, bucket_edges,
        dates=pd.to_datetime(nn_cal["date"]))
    logger.info("  Sigma multiplier (NN): %.3f", sigma_mult_nn)
    e2_sig_d = {k: v * sigma_mult_nn for k, v in nn_eval_sig_d.items()}
    _eval(nn_eval_mu_d, e2_sig_d, "E2_nn_sigma_scaled")

    # ===== E3: NN + Global bias =====
    global_bias_nn = fit_global_bias(nn_pre["mu"].values, nn_pre["actual_tmax"].values)
    logger.info("  Global bias (NN): %.2f F", global_bias_nn)
    e3_mu_d = {k: v + global_bias_nn for k, v in nn_eval_mu_d.items()}
    _eval(e3_mu_d, nn_eval_sig_d, "E3_nn_global_bias")

    # ===== E4: NN + Seasonal bias =====
    seasonal_bias_nn = fit_seasonal_bias(
        nn_pre["mu"].values, nn_pre["actual_tmax"].values,
        pd.to_datetime(nn_pre["date"]))
    logger.info("  Seasonal bias (NN): %s", {k: f"{v:.2f}" for k, v in seasonal_bias_nn.items()})
    e4_mu = apply_seasonal_bias(
        nn_eval["mu"].values, pd.to_datetime(nn_eval["date"]),
        seasonal_bias_nn)
    e4_mu_d = dict(zip(nn_eval["date"].astype(str), e4_mu))
    _eval(e4_mu_d, nn_eval_sig_d, "E4_nn_seasonal_bias")

    # ===== E5: NN + Isotonic + Sigma scaling =====
    e5_sig_d = {k: v * sigma_mult_nn for k, v in nn_eval_sig_d.items()}
    _eval(e1_mu_d, e5_sig_d, "E5_nn_isotonic_sigma")

    # ===== E6: NN + Isotonic + Seasonal bias + Sigma scaling =====
    e6_mu = apply_isotonic_mu(ir_nn, nn_eval["mu"].values)
    e6_mu_adj = apply_seasonal_bias(e6_mu, pd.to_datetime(nn_eval["date"]), seasonal_bias_nn)
    e6_mu_d = dict(zip(nn_eval["date"].astype(str), e6_mu_adj))
    _eval(e6_mu_d, e5_sig_d, "E6_nn_iso_season_sigma")

    # ===== E7: Ridge base =====
    ridge_eval_mu_d, ridge_eval_sig_d = _make_dicts(ridge_eval)
    _eval(ridge_eval_mu_d, ridge_eval_sig_d, "E7_ridge_base")

    # ===== E8: Ridge + Isotonic =====
    ir_ridge = fit_isotonic_mu(ridge_pre["mu"].values, ridge_pre["actual_tmax"].values)
    e8_mu = apply_isotonic_mu(ir_ridge, ridge_eval["mu"].values)
    e8_mu_d = dict(zip(ridge_eval["date"].astype(str), e8_mu))
    _eval(e8_mu_d, ridge_eval_sig_d, "E8_ridge_isotonic")

    # ===== E9: Ridge + Sigma scaling =====
    ridge_cal_mu_d, ridge_cal_sig_d = _make_dicts(ridge_cal)
    sigma_mult_ridge = fit_sigma_scaler(
        ridge_cal["mu"].values, ridge_cal["sigma"].values,
        ridge_cal["actual_tmax"].values, kalshi_cal, bucket_edges,
        dates=pd.to_datetime(ridge_cal["date"]))
    logger.info("  Sigma multiplier (Ridge): %.3f", sigma_mult_ridge)
    e9_sig_d = {k: v * sigma_mult_ridge for k, v in ridge_eval_sig_d.items()}
    _eval(ridge_eval_mu_d, e9_sig_d, "E9_ridge_sigma_scaled")

    # ===== E10: Ridge + Isotonic + Sigma =====
    _eval(e8_mu_d, e9_sig_d, "E10_ridge_iso_sigma")

    # ===== E11: Ridge + Seasonal bias + Sigma =====
    seasonal_bias_ridge = fit_seasonal_bias(
        ridge_pre["mu"].values, ridge_pre["actual_tmax"].values,
        pd.to_datetime(ridge_pre["date"]))
    e11_mu = apply_seasonal_bias(
        ridge_eval["mu"].values, pd.to_datetime(ridge_eval["date"]),
        seasonal_bias_ridge)
    e11_mu_d = dict(zip(ridge_eval["date"].astype(str), e11_mu))
    _eval(e11_mu_d, e9_sig_d, "E11_ridge_season_sigma")

    # ===== E12: Ensemble (Ridge + NN average) =====
    ensemble_dates = sorted(set(nn_eval["date"].astype(str)) & set(ridge_eval["date"].astype(str)))
    e12_mu_d = {}
    e12_sig_d = {}
    for d in ensemble_dates:
        mu_nn = nn_eval_mu_d.get(d)
        mu_r = ridge_eval_mu_d.get(d)
        sig_nn = nn_eval_sig_d.get(d)
        sig_r = ridge_eval_sig_d.get(d)
        if mu_nn is not None and mu_r is not None:
            e12_mu_d[d] = (mu_nn + mu_r) / 2.0
            e12_sig_d[d] = np.sqrt((sig_nn**2 + sig_r**2) / 2.0)
    _eval(e12_mu_d, e12_sig_d, "E12_ensemble_avg")

    # ===== E13: Ensemble + Isotonic =====
    # Fit isotonic on pre-eval ensemble
    ens_pre_dates = sorted(set(nn_pre["date"].astype(str)) & set(ridge_pre["date"].astype(str)))
    ens_pre_mu = []
    ens_pre_actual = []
    for d in ens_pre_dates:
        nn_row = nn_pre[nn_pre["date"].astype(str) == d]
        r_row = ridge_pre[ridge_pre["date"].astype(str) == d]
        if len(nn_row) > 0 and len(r_row) > 0:
            ens_pre_mu.append((nn_row["mu"].values[0] + r_row["mu"].values[0]) / 2.0)
            ens_pre_actual.append(nn_row["actual_tmax"].values[0])

    ir_ens = fit_isotonic_mu(np.array(ens_pre_mu), np.array(ens_pre_actual))

    e13_mu_d = {}
    for d in ensemble_dates:
        raw = e12_mu_d.get(d)
        if raw is not None:
            e13_mu_d[d] = float(ir_ens.predict(np.array([raw]))[0])
    _eval(e13_mu_d, e12_sig_d, "E13_ensemble_isotonic")

    # ===== E14: Ensemble + Seasonal bias + Sigma scaling =====
    ens_pre_mu_arr = np.array(ens_pre_mu)
    ens_pre_actual_arr = np.array(ens_pre_actual)
    ens_pre_dates_dt = pd.to_datetime(ens_pre_dates)
    ens_seasonal_bias = fit_seasonal_bias(ens_pre_mu_arr, ens_pre_actual_arr, ens_pre_dates_dt)

    ens_eval_mu = np.array([e12_mu_d[d] for d in ensemble_dates])
    ens_eval_sig = np.array([e12_sig_d[d] for d in ensemble_dates])
    ens_eval_dates_dt = pd.to_datetime(ensemble_dates)
    e14_mu = apply_seasonal_bias(ens_eval_mu, ens_eval_dates_dt, ens_seasonal_bias)

    # Sigma scaling for ensemble on cal period (needs Kalshi data)
    ens_cal_dates = sorted(set(nn_cal["date"].astype(str)) & set(ridge_cal["date"].astype(str)))
    ens_cal_mu = []
    ens_cal_sig = []
    ens_cal_actual = []
    for d in ens_cal_dates:
        nn_row = nn_cal[nn_cal["date"].astype(str) == d]
        r_row = ridge_cal[ridge_cal["date"].astype(str) == d]
        if len(nn_row) > 0 and len(r_row) > 0:
            ens_cal_mu.append((nn_row["mu"].values[0] + r_row["mu"].values[0]) / 2.0)
            sig_nn = nn_row["sigma"].values[0]
            sig_r = r_row["sigma"].values[0]
            ens_cal_sig.append(np.sqrt((sig_nn**2 + sig_r**2) / 2.0))
            ens_cal_actual.append(nn_row["actual_tmax"].values[0])
    ens_cal_mu_arr = np.array(ens_cal_mu)
    ens_cal_sig_arr = np.array(ens_cal_sig)
    ens_cal_actual_arr = np.array(ens_cal_actual)

    ens_sigma_mult = fit_sigma_scaler(
        ens_cal_mu_arr, ens_cal_sig_arr, ens_cal_actual_arr, kalshi_cal, bucket_edges,
        dates=pd.to_datetime(ens_cal_dates))
    logger.info("  Sigma multiplier (Ensemble): %.3f", ens_sigma_mult)

    e14_mu_d = dict(zip(ensemble_dates, e14_mu))
    e14_sig_d = {d: e12_sig_d[d] * ens_sigma_mult for d in ensemble_dates}
    _eval(e14_mu_d, e14_sig_d, "E14_ensemble_season_sigma")

    # ===== E15: Conditional sigma calibration (NN) =====
    cond_cal = fit_conditional_sigma(
        nn_pre["mu"].values, nn_pre["sigma"].values, nn_pre["actual_tmax"].values)
    e15_sig = apply_conditional_sigma(nn_eval["sigma"].values, cond_cal)
    e15_sig_d = dict(zip(nn_eval["date"].astype(str), e15_sig))
    _eval(nn_eval_mu_d, e15_sig_d, "E15_nn_conditional_sigma")

    # ===== E16: Platt scaling (on best base model - Ridge) =====
    platt_lr = fit_platt_scaling(kalshi_cal, ridge_cal_mu_d, ridge_cal_sig_d)
    if platt_lr is not None:
        e16_cdf = build_contract_dataset(kalshi_eval, ridge_eval_mu_d, ridge_eval_sig_d)
        e16_cdf = apply_platt_scaling(platt_lr, e16_cdf)
        if len(e16_cdf) > 0:
            e16_brier = contract_brier(e16_cdf["model_prob"].values,
                                       e16_cdf["actual_outcome"].values)
            e16_cdf["date_dt"] = pd.to_datetime(e16_cdf["date"])
            e16_cdf["season"] = e16_cdf["date_dt"].dt.month.map(SEASON_MAP)
            e16_seasonal = {}
            for s in ["DJF", "MAM", "JJA", "SON"]:
                mask = e16_cdf["season"] == s
                if mask.any():
                    e16_seasonal[s] = contract_brier(
                        e16_cdf.loc[mask, "model_prob"].values,
                        e16_cdf.loc[mask, "actual_outcome"].values)
            results["E16_platt_ridge"] = {"brier": e16_brier, "seasonal": e16_seasonal,
                                          "n_dates": e16_cdf["date"].nunique()}
            logger.info("  E16_platt_ridge: contract Brier=%.4f", e16_brier)

    # ===== E17: Contract-Brier-optimized sigma search (Ridge) =====
    def brier_sigma_search(base_mu_d, base_sig_d, kalshi_cal_df):
        best_brier = float("inf")
        best_mult = 1.0
        for mult in np.arange(0.4, 2.5, 0.05):
            sig_d = {k: v * mult for k, v in base_sig_d.items()}
            cdf = build_contract_dataset(kalshi_cal_df, base_mu_d, sig_d)
            if len(cdf) == 0:
                continue
            b = contract_brier(cdf["model_prob"].values, cdf["actual_outcome"].values)
            if b < best_brier:
                best_brier = b
                best_mult = mult
        return best_mult

    e17_mult = brier_sigma_search(ridge_cal_mu_d, ridge_cal_sig_d, kalshi_cal)
    logger.info("  E17 sigma mult (Ridge, Brier-opt): %.3f", e17_mult)
    e17_sig_d = {k: v * e17_mult for k, v in ridge_eval_sig_d.items()}
    _eval(ridge_eval_mu_d, e17_sig_d, "E17_ridge_brier_sigma")

    # ===== E18: CRPS-optimized sigma (Ridge) =====
    crps_mult = fit_crps_sigma(
        ridge_pre["mu"].values, ridge_pre["sigma"].values,
        ridge_pre["actual_tmax"].values)
    logger.info("  E18 CRPS sigma mult (Ridge): %.3f", crps_mult)
    e18_sig_d = {k: v * crps_mult for k, v in ridge_eval_sig_d.items()}
    _eval(ridge_eval_mu_d, e18_sig_d, "E18_ridge_crps_sigma")

    # ===== E19: Platt on Ensemble =====
    platt_ens = fit_platt_scaling(kalshi_cal,
                                  {d: ens_cal_mu[i] for i, d in enumerate(ens_cal_dates)},
                                  {d: ens_cal_sig[i] for i, d in enumerate(ens_cal_dates)})
    if platt_ens is not None:
        e19_cdf = build_contract_dataset(kalshi_eval, e12_mu_d, e12_sig_d)
        e19_cdf = apply_platt_scaling(platt_ens, e19_cdf)
        if len(e19_cdf) > 0:
            e19_brier = contract_brier(e19_cdf["model_prob"].values,
                                       e19_cdf["actual_outcome"].values)
            e19_cdf["date_dt"] = pd.to_datetime(e19_cdf["date"])
            e19_cdf["season"] = e19_cdf["date_dt"].dt.month.map(SEASON_MAP)
            e19_seasonal = {}
            for s in ["DJF", "MAM", "JJA", "SON"]:
                mask = e19_cdf["season"] == s
                if mask.any():
                    e19_seasonal[s] = contract_brier(
                        e19_cdf.loc[mask, "model_prob"].values,
                        e19_cdf.loc[mask, "actual_outcome"].values)
            results["E19_platt_ensemble"] = {"brier": e19_brier, "seasonal": e19_seasonal,
                                             "n_dates": e19_cdf["date"].nunique()}
            logger.info("  E19_platt_ensemble: contract Brier=%.4f", e19_brier)

    # ===== E20: Best model + Platt per-contract =====
    # Use Ridge + isotonic + brier sigma as best combination
    e20_mu_d = e8_mu_d  # Ridge isotonic
    e20_sig_d = e17_sig_d  # Brier-optimized sigma
    platt_e20 = fit_platt_scaling(kalshi_cal,
                                  {d: float(ir_ridge.predict(np.array([ridge_cal_mu_d.get(d, 80)]))[0])
                                   for d in ridge_cal_mu_d},
                                  {k: v * e17_mult for k, v in ridge_cal_sig_d.items()})
    if platt_e20 is not None:
        e20_cdf = build_contract_dataset(kalshi_eval, e20_mu_d, e20_sig_d)
        e20_cdf = apply_platt_scaling(platt_e20, e20_cdf)
        if len(e20_cdf) > 0:
            e20_brier = contract_brier(e20_cdf["model_prob"].values,
                                       e20_cdf["actual_outcome"].values)
            e20_cdf["date_dt"] = pd.to_datetime(e20_cdf["date"])
            e20_cdf["season"] = e20_cdf["date_dt"].dt.month.map(SEASON_MAP)
            e20_seasonal = {}
            for s in ["DJF", "MAM", "JJA", "SON"]:
                mask = e20_cdf["season"] == s
                if mask.any():
                    e20_seasonal[s] = contract_brier(
                        e20_cdf.loc[mask, "model_prob"].values,
                        e20_cdf.loc[mask, "actual_outcome"].values)
            results["E20_ridge_iso_brier_platt"] = {
                "brier": e20_brier, "seasonal": e20_seasonal,
                "n_dates": e20_cdf["date"].nunique()}
            logger.info("  E20_ridge_iso_brier_platt: contract Brier=%.4f", e20_brier)

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    logger.info("=" * 70)
    logger.info("Austin E-Series Benchmark (KXHIGHAUS)")
    logger.info("=" * 70)

    city = get_city_config("aus")
    results_dir = city.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Load base predictions
    base_preds = load_base_predictions(results_dir)
    base_preds["date_str"] = base_preds["date"].dt.strftime("%Y-%m-%d")
    logger.info("Loaded base predictions: %d rows, models: %s",
                len(base_preds), base_preds["model_name"].unique().tolist())

    # Load Kalshi data
    kalshi_df = load_kalshi_data(PROJECT_ROOT)

    # Find overlap between base predictions and Kalshi data
    base_date_strs = set(base_preds["date_str"].unique())
    kalshi_date_strs = set(kalshi_df["date"].unique())
    overlap_dates = sorted(base_date_strs & kalshi_date_strs)
    logger.info("Overlap between base predictions and Kalshi: %d dates (%s to %s)",
                len(overlap_dates), overlap_dates[0], overlap_dates[-1])

    # Split overlap into calibration (first 40%) and evaluation (last 60%)
    # Using 40/60 to maximize eval set while keeping enough cal data
    cal_cutoff = int(len(overlap_dates) * 0.40)
    cal_dates = overlap_dates[:cal_cutoff]
    eval_dates = overlap_dates[cal_cutoff:]

    logger.info("Calibration dates: %s to %s (%d days)",
                cal_dates[0], cal_dates[-1], len(cal_dates))
    logger.info("Evaluation dates: %s to %s (%d days)",
                eval_dates[0], eval_dates[-1], len(eval_dates))

    # Run E-series
    e_results = run_e_series(
        base_preds, kalshi_df, cal_dates, eval_dates, city.bucket_edges)

    # Compute Kalshi market Brier
    market_brier, market_seasonal = eval_market_brier(kalshi_df, eval_dates)
    logger.info("Kalshi Market Brier: %.4f", market_brier)
    e_results["Kalshi_Market"] = {"brier": market_brier, "seasonal": market_seasonal, "n_dates": 0}

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 70)
    logger.info("E-SERIES BENCHMARK SUMMARY")
    logger.info("=" * 70)

    rows = []
    for name, res in sorted(e_results.items(), key=lambda x: x[1].get("brier", 999)):
        row = {
            "model": name,
            "contract_brier": res["brier"],
            "n_dates": res.get("n_dates", 0),
        }
        for s in ["DJF", "MAM", "JJA", "SON"]:
            row[f"brier_{s}"] = res.get("seasonal", {}).get(s, float("nan"))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    logger.info("\n%s", summary_df.to_string(index=False))

    # Edge over market
    best_model = summary_df[summary_df["model"] != "Kalshi_Market"].iloc[0]
    logger.info("\nBest E-series model: %s (Brier: %.4f)",
                best_model["model"], best_model["contract_brier"])
    logger.info("Kalshi Market Brier: %.4f", market_brier)
    edge = market_brier - best_model["contract_brier"]
    logger.info("Edge over market: %.4f (%s)",
                edge, "POSITIVE" if edge > 0 else "NEGATIVE")

    # Save results
    summary_path = os.path.join(results_dir, "aus_e_series_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved E-series summary to %s", summary_path)

    detail = {name: {
        "contract_brier": float(res["brier"]) if not np.isnan(res["brier"]) else None,
        "seasonal": {k: float(v) for k, v in res.get("seasonal", {}).items()},
        "n_dates": res.get("n_dates", 0),
    } for name, res in e_results.items()}

    detail_path = os.path.join(results_dir, "aus_e_series_detail.json")
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    logger.info("Saved E-series detail to %s", detail_path)

    # Plot
    plot_path = os.path.join(results_dir, "aus_e_series_brier.png")
    fig, ax = plt.subplots(figsize=(14, 7))

    models = summary_df["model"].tolist()
    briers = summary_df["contract_brier"].tolist()
    colors = ["red" if m == "Kalshi_Market" else "steelblue" for m in models]

    bars = ax.bar(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, briers):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Contract Brier Score (lower is better)")
    ax.set_title("Austin E-Series Benchmark: Contract Brier vs Kalshi Market")
    ax.axhline(y=market_brier, color="red", linestyle="--", alpha=0.7, label=f"Kalshi Market ({market_brier:.4f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved E-series plot to %s", plot_path)

    # Seasonal plot
    seasonal_plot_path = os.path.join(results_dir, "aus_e_series_seasonal.png")
    top_models = summary_df[summary_df["model"] != "Kalshi_Market"].head(5)["model"].tolist()
    top_models.append("Kalshi_Market")

    fig, ax = plt.subplots(figsize=(12, 6))
    seasons = ["DJF", "MAM", "JJA", "SON"]
    n_models = len(top_models)
    x = np.arange(len(seasons))
    width = 0.8 / n_models
    colors_cycle = plt.cm.tab10(np.arange(n_models))

    for i, model in enumerate(top_models):
        vals = [summary_df[summary_df["model"] == model][f"brier_{s}"].values[0]
                if len(summary_df[summary_df["model"] == model]) > 0 else 0
                for s in seasons]
        color = "red" if model == "Kalshi_Market" else colors_cycle[i]
        ax.bar(x + i * width, vals, width, label=model, color=color,
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Contract Brier Score")
    ax.set_title("Austin E-Series: Top 5 Models + Market (Seasonal)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(seasonal_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved seasonal plot to %s", seasonal_plot_path)

    logger.info("=" * 70)
    logger.info("Austin E-Series Benchmark Complete")
    logger.info("=" * 70)

    return e_results


if __name__ == "__main__":
    main()
