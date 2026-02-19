#!/usr/bin/env python3
"""
Chicago + Philadelphia Unified Benchmark with Real Kalshi Pre-Settlement Data.

Trains E-series base models and Unified synthesis variants for both cities,
using 100% of available Kalshi settlement data for contract-level Brier scoring.

Model families:
  E0: Persistence baseline
  E1: Climatology baseline
  E2: Ridge regression (alpha search)
  E3: Heteroscedastic NN [128,64]
  E4: Deep Heteroscedastic NN [256,128,64]
  E5: Ensemble of 5 NNs (average mu/sigma)
  U0: Raw Gaussian bucket probs from best NN
  U1: Isotonic calibration
  U2: Contract-level Ridge
  U3: Contract-level Brier-optimal MLP
  U4: Platt recalibration on U3
  U5: Regime-conditional (season x volatility)
  U6: Calibrated ensemble (avg of U1,U4,U5)

Usage:
    python scripts/run_chi_phl_unified_benchmark.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}


# ============================================================================
# Data Loading
# ============================================================================

def load_processed_data(processed_dir: str, city_code: str):
    """Load preprocessed CSV files with NaN imputation."""
    X_train = pd.read_csv(os.path.join(processed_dir, "features_train.csv"),
                          index_col=0, parse_dates=True)
    X_val = pd.read_csv(os.path.join(processed_dir, "features_val.csv"),
                        index_col=0, parse_dates=True)
    X_test = pd.read_csv(os.path.join(processed_dir, "features_test.csv"),
                         index_col=0, parse_dates=True)
    y_train = pd.read_csv(os.path.join(processed_dir, "target_train.csv"),
                          index_col=0, parse_dates=True).iloc[:, 0]
    y_val = pd.read_csv(os.path.join(processed_dir, "target_val.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(processed_dir, "target_test.csv"),
                         index_col=0, parse_dates=True).iloc[:, 0]

    # Drop columns that are entirely NaN, then impute remaining NaNs
    all_nan_cols = X_train.columns[X_train.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info("Dropping %d all-NaN columns: %s", len(all_nan_cols),
                     list(all_nan_cols))
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    # Impute remaining NaNs with column mean from training set
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_val = X_val.fillna(train_means)
    X_test = X_test.fillna(train_means)

    for s in (y_train, y_val, y_test):
        s.name = f"{city_code.upper()}_TMAX"
    logger.info("Loaded %s data: train=%s val=%s test=%s",
                city_code, X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_kalshi_data(city_code: str) -> pd.DataFrame:
    """Load Kalshi data: merge pre-settlement prices with settlement outcomes.

    Pre-settlement data provides market prices ~24 hours before close.
    Settlement data provides ground truth (actual_outcome, actual_tmax)
    and verified bucket definitions.
    """
    if city_code == "chi":
        settlement_path = PROJECT_ROOT / "data" / "real_kalshi_chi_all.csv"
        presettlement_path = PROJECT_ROOT / "data" / "kalshi_presettlement_chi.csv"
    elif city_code == "phl":
        settlement_path = PROJECT_ROOT / "data" / "real_kalshi_phl_all.csv"
        presettlement_path = PROJECT_ROOT / "data" / "kalshi_presettlement_phl.csv"
    else:
        raise ValueError(f"Unknown city: {city_code}")

    settled = pd.read_csv(settlement_path)
    settled["date"] = pd.to_datetime(settled["date"]).dt.strftime("%Y-%m-%d")

    if presettlement_path.exists():
        pre = pd.read_csv(presettlement_path)
        pre["date"] = pd.to_datetime(pre["date"]).dt.strftime("%Y-%m-%d")

        merged = settled.merge(
            pre[["date", "ticker", "presettlement_prob", "bid_cents",
                 "ask_cents", "volume", "open_interest", "snapshot_time_utc"]],
            on=["date", "ticker"],
            how="inner",
            suffixes=("", "_pre"),
        )
        n_before = len(merged)
        merged = merged.dropna(subset=["presettlement_prob"])
        n_dropped = n_before - len(merged)

        # CRITICAL: Replace settlement market_prob with pre-settlement prices
        merged["market_prob"] = merged["presettlement_prob"].clip(
            PROB_CLIP_MIN, PROB_CLIP_MAX)

        logger.info("Loaded Kalshi %s (pre-settlement): %d rows, %d dates "
                     "(dropped %d missing presettlement_prob)",
                     city_code, len(merged), merged["date"].nunique(), n_dropped)

        extreme = ((merged["presettlement_prob"] <= 0.02) |
                   (merged["presettlement_prob"] >= 0.98)).mean()
        if extreme > 0.5:
            logger.warning("  WARNING: %.1f%% extreme — may be settlement data!",
                           extreme * 100)
        else:
            logger.info("  Extreme presettlement_prob: %.1f%%", extreme * 100)

        return merged
    else:
        logger.warning("No presettlement data at %s — using settlement only!",
                        presettlement_path)
        df = settled
        logger.info("Loaded Kalshi %s: %d rows, %d dates (%s to %s)",
                    city_code, len(df), df["date"].nunique(),
                    df["date"].min(), df["date"].max())
        return df


# ============================================================================
# Bucket Probability Helpers
# ============================================================================

def gaussian_to_bucket_probs(mu, sigma, bucket_edges):
    """Convert N(mu, sigma) to bucket probabilities. Returns (n_days, n_buckets)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 0.5)
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    probs = np.zeros((n_days, n_buckets))
    for b, (lo, hi) in enumerate(bucket_edges):
        cdf_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        cdf_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        probs[:, b] = np.clip(cdf_hi - cdf_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sums, 1e-10)
    return probs


def compute_brier_score(bucket_probs, actual_tmax, bucket_edges):
    """Compute overall contract-level Brier score."""
    n_days, n_buckets = bucket_probs.shape
    outcomes = np.zeros((n_days, n_buckets))
    for d in range(n_days):
        t = actual_tmax[d]
        if np.isnan(t):
            continue
        for b, (lo, hi) in enumerate(bucket_edges):
            if b == n_buckets - 1:
                if lo <= t <= hi:
                    outcomes[d, b] = 1.0
                    break
            else:
                if lo <= t < hi:
                    outcomes[d, b] = 1.0
                    break
    brier_components = (bucket_probs - outcomes) ** 2
    return float(np.mean(brier_components))


def compute_seasonal_brier(bucket_probs, actual_tmax, dates, bucket_edges):
    """Compute Brier per season."""
    months = dates.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    results = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == s
        if mask.any():
            results[s] = compute_brier_score(
                bucket_probs[mask], actual_tmax[mask], bucket_edges)
    return results


# ============================================================================
# E-Series Base Models
# ============================================================================

def run_persistence(y_train, y_val, y_test):
    all_y = pd.concat([y_train, y_val, y_test])
    y_prev = all_y.shift(1)
    train_mask = all_y.index.isin(y_train.index)
    sigma = float((all_y[train_mask] - y_prev[train_mask]).dropna().std())
    sigma = max(sigma, 3.0)
    mu_val = np.where(np.isnan(y_prev.reindex(y_val.index).values),
                      float(y_train.mean()), y_prev.reindex(y_val.index).values)
    mu_test = np.where(np.isnan(y_prev.reindex(y_test.index).values),
                       float(y_train.mean()), y_prev.reindex(y_test.index).values)
    return {"mu_val": mu_val, "sigma_val": np.full_like(mu_val, sigma),
            "mu_test": mu_test, "sigma_test": np.full_like(mu_test, sigma)}


def run_climatology(y_train, y_val, y_test):
    doy = y_train.index.dayofyear
    clim_mean = y_train.groupby(doy).mean()
    clim_std = y_train.groupby(doy).std().clip(lower=3.0)
    all_doys = np.arange(1, 367)
    clim_mean = clim_mean.reindex(all_doys).interpolate().bfill().ffill()
    clim_std = clim_std.reindex(all_doys).interpolate().bfill().ffill()
    return {
        "mu_val": clim_mean.reindex(y_val.index.dayofyear).values,
        "sigma_val": clim_std.reindex(y_val.index.dayofyear).values,
        "mu_test": clim_mean.reindex(y_test.index.dayofyear).values,
        "sigma_test": clim_std.reindex(y_test.index.dayofyear).values,
    }


def run_ridge(X_train, y_train, X_val, y_val, X_test, y_test, bucket_edges):
    best_alpha, best_brier = None, float("inf")
    best_res = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train.values, y_train.values)
        mu_val = model.predict(X_val.values)
        sigma = max(float(np.std(y_train.values - model.predict(X_train.values))), 3.0)
        probs = gaussian_to_bucket_probs(mu_val, np.full_like(mu_val, sigma), bucket_edges)
        brier = compute_brier_score(probs, y_val.values, bucket_edges)
        if brier < best_brier:
            best_brier = brier
            best_alpha = alpha
            mu_test = model.predict(X_test.values)
            best_res = {"mu_val": mu_val, "sigma_val": np.full_like(mu_val, sigma),
                        "mu_test": mu_test, "sigma_test": np.full_like(mu_test, sigma),
                        "alpha": alpha}
    logger.info("Ridge best alpha=%.2f, val Brier=%.4f", best_alpha, best_brier)
    return best_res


# ============================================================================
# Heteroscedastic Neural Network
# ============================================================================

class HeteroscedasticNet(nn.Module):
    def __init__(self, n_features, hidden_sizes=None, dropout=0.1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 4.0)
        return mu, torch.exp(log_sigma)


def gaussian_nll_loss(mu, sigma, target):
    var = sigma ** 2
    return (0.5 * (torch.log(2 * torch.pi * var) + ((target - mu) ** 2) / var)).mean()


def train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
             hidden_sizes=None, dropout=0.1, lr=0.001,
             max_epochs=200, patience=20, batch_size=64):
    """Train heteroscedastic NN, return mu/sigma for val and test."""
    n_feat = X_train.shape[1]
    model = HeteroscedasticNet(n_feat, hidden_sizes, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    def make_loader(X, y, shuffle=False):
        Xt = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        yt = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            mu, sigma = model(Xb)
            loss = gaussian_nll_loss(mu, sigma, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                mu, sigma = model(Xb)
                val_losses.append(gaussian_nll_loss(mu, sigma, yb).item())
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    logger.info("NN best epoch %d, val NLL %.4f", best_epoch, best_val_loss)

    model.eval()
    def predict(X):
        Xt = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            mu, sigma = model(Xt)
        return mu.cpu().numpy().ravel(), sigma.cpu().numpy().ravel()

    mu_val, sig_val = predict(X_val)
    mu_test, sig_test = predict(X_test)
    return {"mu_val": mu_val, "sigma_val": sig_val,
            "mu_test": mu_test, "sigma_test": sig_test,
            "model": model, "best_epoch": best_epoch}


def train_ensemble_nn(X_train, y_train, X_val, y_val, X_test, y_test, n_models=5):
    """Train ensemble of NNs, average predictions."""
    configs = [
        ([128, 64], 0.1, 0.001),
        ([256, 128], 0.15, 0.0008),
        ([128, 64, 32], 0.1, 0.001),
        ([256, 128, 64], 0.12, 0.0005),
        ([192, 96], 0.1, 0.0007),
    ]
    mu_vals, sig_vals = [], []
    mu_tests, sig_tests = [], []
    for i, (hs, do, lr) in enumerate(configs[:n_models]):
        logger.info("  Ensemble member %d/%d: hidden=%s", i+1, n_models, hs)
        res = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                       hidden_sizes=hs, dropout=do, lr=lr, max_epochs=150, patience=15)
        mu_vals.append(res["mu_val"])
        sig_vals.append(res["sigma_val"])
        mu_tests.append(res["mu_test"])
        sig_tests.append(res["sigma_test"])
    return {
        "mu_val": np.mean(mu_vals, axis=0),
        "sigma_val": np.mean(sig_vals, axis=0),
        "mu_test": np.mean(mu_tests, axis=0),
        "sigma_test": np.mean(sig_tests, axis=0),
    }


# ============================================================================
# Contract-Level Dataset Builder
# ============================================================================

def build_contract_dataset(kalshi_df, mu_series, sigma_series, bucket_edges, bucket_labels):
    """Build contract-level dataset merging model predictions with Kalshi data.

    mu_series, sigma_series: pd.Series indexed by date string.
    Returns DataFrame with one row per (date, bucket).
    """
    kalshi = kalshi_df.copy()

    # Ensure consistent date format
    mu_df = pd.DataFrame({"date": mu_series.index.astype(str),
                          "model_mu": mu_series.values,
                          "model_sigma": sigma_series.values})
    mu_df["date"] = pd.to_datetime(mu_df["date"]).dt.strftime("%Y-%m-%d")
    mu_df = mu_df.drop_duplicates(subset="date", keep="first")

    # Merge model predictions to Kalshi data
    merged = kalshi.merge(mu_df, on="date", how="inner")
    logger.info("Contract dataset: %d rows (%d dates) after merge",
                len(merged), merged["date"].nunique())

    if len(merged) == 0:
        logger.warning("No overlap between model predictions and Kalshi dates!")
        return pd.DataFrame()

    # Compute model bucket probabilities
    mu = merged["model_mu"].values
    sigma = np.maximum(merged["model_sigma"].values, 0.5)

    # For each contract row, compute the Gaussian probability for that bucket
    th_low = merged["threshold_low"].values.astype(float)
    th_high = merged["threshold_high"].values.astype(float)
    direction = merged["direction"].values

    model_prob = np.full(len(merged), np.nan)
    below = direction == "below"
    above = direction == "above"
    between = direction == "between"

    # Handle 'less' as 'below'
    less = direction == "less"
    below = below | less

    if below.any():
        model_prob[below] = norm.cdf(th_high[below], mu[below], sigma[below])
    if above.any():
        model_prob[above] = 1.0 - norm.cdf(th_low[above], mu[above], sigma[above])
    if between.any():
        model_prob[between] = (norm.cdf(th_high[between], mu[between], sigma[between])
                               - norm.cdf(th_low[between], mu[between], sigma[between]))

    model_prob = np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
    merged["model_prob"] = model_prob

    # Period assignment
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].map(SEASON_MAP)
    merged["year"] = merged["date_dt"].dt.year

    return merged


# ============================================================================
# Unified Synthesis Variants
# ============================================================================

def apply_u0_raw(contract_df):
    """U0: Raw Gaussian bucket probabilities."""
    return contract_df["model_prob"].values.copy()


def apply_u1_isotonic(contract_df, cal_frac=0.6):
    """U1: Isotonic calibration on model probabilities."""
    probs = contract_df["model_prob"].values.copy()
    outcomes = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 50:
        return probs

    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                             out_of_bounds="clip")
    iso.fit(probs[:n_cal], outcomes[:n_cal])
    calibrated = np.clip(
        np.interp(np.clip(probs, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)
    return calibrated


def build_contract_features(df):
    """Build contract-level feature matrix for MLP/Ridge models."""
    mu = df["model_mu"].values.astype(float)
    sigma = np.maximum(df["model_sigma"].values.astype(float), 0.5)
    prob = df["model_prob"].values.astype(float)
    market = df["market_prob"].values.astype(float)
    th_low = df["threshold_low"].values.astype(float)
    th_high = df["threshold_high"].values.astype(float)
    direction = df["direction"].values

    # Bucket midpoint
    bucket_mid = np.where(
        (direction == "above"),
        th_low + 2.0,
        np.where((direction == "below") | (direction == "less"),
                 th_high - 2.0,
                 (th_low + th_high) / 2.0))

    bucket_quantile = norm.cdf(bucket_mid.astype(float), mu, sigma)
    bucket_width = np.where(
        (direction == "between"),
        (th_high - th_low).astype(float) / (sigma + 1e-6),
        4.0 / (sigma + 1e-6))
    bucket_dist = np.abs(bucket_mid.astype(float) - mu) / (sigma + 1e-6)
    dir_above = ((direction == "above")).astype(float)
    dir_below = ((direction == "below") | (direction == "less")).astype(float)

    # Neighboring bucket sum
    date_sum = pd.Series(prob).groupby(df["date"].values).transform("sum")
    neighbor_sum = np.clip(date_sum.values - prob, 0.0, None)

    # Season features
    month = df["month"].values.astype(float)
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # Volatility features
    sigma_norm = (sigma - np.nanpercentile(sigma, 5)) / (np.nanpercentile(sigma, 95) - np.nanpercentile(sigma, 5) + 1e-6)
    sigma_norm = np.clip(sigma_norm, 0, 1)

    # Volume features
    vol = np.log1p(df["volume"].fillna(0).values.astype(float))

    X = np.column_stack([
        prob, market, prob - market,  # 3: base probs + edge
        bucket_quantile, bucket_width, bucket_dist,  # 3: bucket geometry
        dir_above, dir_below, neighbor_sum,  # 3: structure
        mu, sigma, sigma_norm,  # 3: distributional
        sin_month, cos_month,  # 2: seasonal
        vol,  # 1: liquidity
        prob * sigma_norm,  # 1: interaction
        (prob - market) * (1 - dir_above - dir_below),  # 1: between-edge
    ])
    return X


def apply_u2_ridge(contract_df, cal_frac=0.6):
    """U2: Ridge regression on contract-level features."""
    X = build_contract_features(contract_df)
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 100:
        return contract_df["model_prob"].values.copy()

    X_cal, y_cal = X[:n_cal], y[:n_cal]
    n_train = int(n_cal * 0.75)
    X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
    X_va, y_va = X_cal[n_train:], y_cal[n_train:]

    mu_x = X_tr.mean(axis=0)
    sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

    best_clf, best_brier = None, float("inf")
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit((X_tr - mu_x) / sd_x, y_tr)
        pred = np.clip(clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                       PROB_CLIP_MIN, PROB_CLIP_MAX)
        brier = float(np.mean((pred - y_va) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_clf = clf

    # Apply to all data
    X_z = (X - mu_x) / sd_x
    raw = np.clip(best_clf.predict_proba(X_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Isotonic post-cal on validation portion
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    raw_va = np.clip(best_clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                     PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso.fit(raw_va, y_va)
    calibrated = np.clip(
        np.interp(np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)
    return calibrated


def apply_u3_mlp(contract_df, cal_frac=0.6):
    """U3: Contract-level Brier-optimal MLP."""
    X = build_contract_features(contract_df)
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 100:
        return contract_df["model_prob"].values.copy()

    X_cal, y_cal = X[:n_cal], y[:n_cal]
    n_train = int(n_cal * 0.60)
    n_val = int(n_cal * 0.20)
    X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
    X_va, y_va = X_cal[n_train:n_train+n_val], y_cal[n_train:n_train+n_val]
    X_iso, y_iso = X_cal[n_train+n_val:], y_cal[n_train+n_val:]

    mu_x = X_tr.mean(axis=0)
    sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))
    X_tr_z = (X_tr - mu_x) / sd_x
    X_va_z = (X_va - mu_x) / sd_x
    X_iso_z = (X_iso - mu_x) / sd_x

    configs = [
        ((64, 32), 0.001, 0.001),
        ((128, 64), 0.001, 0.001),
        ((128, 64, 32), 0.001, 0.0005),
        ((256, 128), 0.0001, 0.001),
        ((64, 32), 0.01, 0.001),
        ((128, 64), 0.0001, 0.0005),
    ]

    best_clf, best_score = None, float("inf")
    for hidden, alpha, lr in configs:
        try:
            clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                alpha=alpha, learning_rate_init=lr, max_iter=1200,
                                random_state=42, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=30)
            clf.fit(X_tr_z, y_tr)
            pred = np.clip(clf.predict_proba(X_va_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
            brier = float(np.mean((pred - y_va) ** 2))
            ece = _ece(pred, y_va)
            score = brier + 0.15 * ece
            if score < best_score:
                best_score = score
                best_clf = clf
        except Exception as e:
            logger.warning("MLP config %s failed: %s", hidden, e)

    if best_clf is None:
        return contract_df["model_prob"].values.copy()

    # Isotonic post-cal
    iso_raw = np.clip(best_clf.predict_proba(X_iso_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(iso_raw, y_iso)

    # Apply to all
    X_all_z = (X - mu_x) / sd_x
    raw = np.clip(best_clf.predict_proba(X_all_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    calibrated = np.clip(
        np.interp(np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Per-day renormalize
    return _per_day_renorm(calibrated, contract_df["date"].values)


def apply_u4_platt(contract_df, u3_probs, cal_frac=0.6):
    """U4: Platt scaling on U3 output."""
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 50:
        return u3_probs.copy()

    probs_cal = u3_probs[:n_cal].reshape(-1, 1)
    y_cal = y[:n_cal]

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(probs_cal, y_cal)

    all_probs = u3_probs.reshape(-1, 1)
    platt = np.clip(clf.predict_proba(all_probs)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    return _per_day_renorm(platt, contract_df["date"].values)


def apply_u5_regime(contract_df, cal_frac=0.6):
    """U5: Regime-conditional (season x volatility interaction features)."""
    X_base = build_contract_features(contract_df)
    y = contract_df["actual_outcome"].values.astype(float)

    # Add regime interaction features
    month = contract_df["month"].values.astype(float)
    sigma = np.maximum(contract_df["model_sigma"].values.astype(float), 0.5)
    sigma_norm = (sigma - np.nanpercentile(sigma, 5)) / (np.nanpercentile(sigma, 95) - np.nanpercentile(sigma, 5) + 1e-6)
    sigma_norm = np.clip(sigma_norm, 0, 1)

    # Season dummies
    is_winter = np.isin(month, [12, 1, 2]).astype(float)
    is_spring = np.isin(month, [3, 4, 5]).astype(float)
    is_summer = np.isin(month, [6, 7, 8]).astype(float)
    is_fall = np.isin(month, [9, 10, 11]).astype(float)

    prob = contract_df["model_prob"].values.astype(float)
    market = contract_df["market_prob"].values.astype(float)
    edge = prob - market

    # Regime interaction features
    X_regime = np.column_stack([
        X_base,
        is_winter, is_spring, is_summer, is_fall,
        is_winter * sigma_norm, is_summer * sigma_norm,
        is_winter * edge, is_summer * edge,
        sigma_norm * edge,
        sigma_norm ** 2,
    ])

    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 100:
        return contract_df["model_prob"].values.copy()

    X_cal, y_cal = X_regime[:n_cal], y[:n_cal]
    n_train = int(n_cal * 0.60)
    n_val = int(n_cal * 0.20)
    X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
    X_va, y_va = X_cal[n_train:n_train+n_val], y_cal[n_train:n_train+n_val]
    X_iso, y_iso = X_cal[n_train+n_val:], y_cal[n_train+n_val:]

    mu_x = X_tr.mean(axis=0)
    sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

    configs = [
        ((128, 64), 0.001, 0.001),
        ((128, 64, 32), 0.001, 0.0005),
        ((256, 128), 0.0001, 0.001),
        ((256, 128, 64), 0.0001, 0.0005),
    ]

    best_clf, best_score = None, float("inf")
    for hidden, alpha, lr in configs:
        try:
            clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                alpha=alpha, learning_rate_init=lr, max_iter=1200,
                                random_state=42, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=30)
            clf.fit((X_tr - mu_x) / sd_x, y_tr)
            pred = np.clip(clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                           PROB_CLIP_MIN, PROB_CLIP_MAX)
            brier = float(np.mean((pred - y_va) ** 2))
            ece = _ece(pred, y_va)
            score = brier + 0.15 * ece
            if score < best_score:
                best_score = score
                best_clf = clf
        except Exception as e:
            logger.warning("U5 config %s failed: %s", hidden, e)

    if best_clf is None:
        return contract_df["model_prob"].values.copy()

    # Isotonic post-cal
    iso_raw = np.clip(best_clf.predict_proba((X_iso - mu_x) / sd_x)[:, 1],
                      PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip")
    iso.fit(iso_raw, y_iso)

    X_all_z = (X_regime - mu_x) / sd_x
    raw = np.clip(best_clf.predict_proba(X_all_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    calibrated = np.clip(
        np.interp(np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)
    return _per_day_renorm(calibrated, contract_df["date"].values)


def apply_u6_ensemble(u1_probs, u4_probs, u5_probs):
    """U6: Simple average of calibrated variants."""
    return np.clip((u1_probs + u4_probs + u5_probs) / 3.0, PROB_CLIP_MIN, PROB_CLIP_MAX)


def build_extended_features(df):
    """Build extended feature matrix with market microstructure features."""
    X_base = build_contract_features(df)
    market = df["market_prob"].values.astype(float)
    prob = df["model_prob"].values.astype(float)

    bid = df["bid_cents"].values.astype(float) if "bid_cents" in df.columns else np.full(len(df), np.nan)
    ask = df["ask_cents"].values.astype(float) if "ask_cents" in df.columns else np.full(len(df), np.nan)
    bid_filled = np.where(np.isnan(bid), market * 100, bid)
    ask_filled = np.where(np.isnan(ask), market * 100, ask)
    spread = np.clip((ask_filled - bid_filled) / 100.0, 0.0, 1.0)

    oi_raw = df["open_interest"].values.astype(float) if "open_interest" in df.columns else np.zeros(len(df))
    oi = np.log1p(np.nan_to_num(oi_raw, nan=0.0))
    oi_norm = np.clip(oi / (np.nanpercentile(oi, 95) + 1e-6), 0, 1)

    rank = pd.Series(prob).groupby(df["date"].values).rank(method="average")
    rank_norm = rank.values / rank.groupby(df["date"].values).transform("max").values
    cum_prob = pd.Series(prob).groupby(df["date"].values).cumsum().values

    log_odds_model = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / (1 - np.clip(prob, 1e-4, 1 - 1e-4)))
    log_odds_market = np.log(np.clip(market, 1e-4, 1 - 1e-4) / (1 - np.clip(market, 1e-4, 1 - 1e-4)))
    logit_diff = log_odds_model - log_odds_market

    return np.column_stack([
        X_base, spread, oi_norm, rank_norm, cum_prob,
        log_odds_model, log_odds_market, logit_diff,
        prob ** 2, market ** 2,
    ])


def apply_u7_extended_mlp(contract_df, cal_frac=0.6):
    """U7: Extended-feature contract-level MLP."""
    X = build_extended_features(contract_df)
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 100:
        return contract_df["model_prob"].values.copy()

    X_cal, y_cal = X[:n_cal], y[:n_cal]
    n_train = int(n_cal * 0.55)
    n_val = int(n_cal * 0.20)
    X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
    X_va, y_va = X_cal[n_train:n_train + n_val], y_cal[n_train:n_train + n_val]
    X_iso, y_iso = X_cal[n_train + n_val:], y_cal[n_train + n_val:]

    mu_x = X_tr.mean(axis=0)
    sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

    configs = [
        ((128, 64), 0.001, 0.001),
        ((128, 64, 32), 0.001, 0.0005),
        ((256, 128), 0.0001, 0.001),
        ((256, 128, 64), 0.0001, 0.0005),
        ((128, 64), 0.0001, 0.0005),
        ((64, 32, 16), 0.01, 0.001),
    ]
    best_clf, best_score = None, float("inf")
    for hidden, alpha, lr in configs:
        try:
            clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                alpha=alpha, learning_rate_init=lr, max_iter=1500,
                                random_state=42, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=40)
            clf.fit((X_tr - mu_x) / sd_x, y_tr)
            pred = np.clip(clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                           PROB_CLIP_MIN, PROB_CLIP_MAX)
            brier = float(np.mean((pred - y_va) ** 2))
            ece = _ece(pred, y_va)
            score = brier + 0.12 * ece
            if score < best_score:
                best_score = score
                best_clf = clf
        except Exception as e:
            logger.warning("U7 config %s failed: %s", hidden, e)

    if best_clf is None:
        return contract_df["model_prob"].values.copy()

    iso_raw = np.clip(best_clf.predict_proba((X_iso - mu_x) / sd_x)[:, 1],
                      PROB_CLIP_MIN, PROB_CLIP_MAX)
    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                             out_of_bounds="clip")
    iso.fit(iso_raw, y_iso)

    X_all_z = (X - mu_x) / sd_x
    raw = np.clip(best_clf.predict_proba(X_all_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
    calibrated = np.clip(
        np.interp(np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)
    return _per_day_renorm(calibrated, contract_df["date"].values)


def apply_u8_cv_ensemble(contract_df, n_folds=3, cal_frac=0.6):
    """U8: Cross-validated ensemble of MLPs with different seeds."""
    X = build_extended_features(contract_df)
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 150:
        return contract_df["model_prob"].values.copy()

    fold_preds = []
    for fold in range(n_folds):
        seed = 42 + fold * 7
        X_cal, y_cal = X[:n_cal], y[:n_cal]
        n_train = int(n_cal * 0.55)
        n_val = int(n_cal * 0.20)
        X_tr, y_tr = X_cal[:n_train], y_cal[:n_train]
        X_va, y_va = X_cal[n_train:n_train + n_val], y_cal[n_train:n_train + n_val]
        X_iso, y_iso = X_cal[n_train + n_val:], y_cal[n_train + n_val:]

        mu_x = X_tr.mean(axis=0)
        sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

        configs = [
            ((128, 64), 0.001, 0.001),
            ((128, 64, 32), 0.001, 0.0005),
            ((256, 128), 0.0001, 0.001),
        ]
        best_clf, best_score = None, float("inf")
        for hidden, alpha, lr in configs:
            try:
                clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                    alpha=alpha, learning_rate_init=lr, max_iter=1200,
                                    random_state=seed, early_stopping=True,
                                    validation_fraction=0.15, n_iter_no_change=30)
                clf.fit((X_tr - mu_x) / sd_x, y_tr)
                pred = np.clip(clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                               PROB_CLIP_MIN, PROB_CLIP_MAX)
                brier = float(np.mean((pred - y_va) ** 2))
                if brier < best_score:
                    best_score = brier
                    best_clf = clf
            except Exception:
                pass

        if best_clf is None:
            continue

        iso_raw = np.clip(best_clf.predict_proba((X_iso - mu_x) / sd_x)[:, 1],
                          PROB_CLIP_MIN, PROB_CLIP_MAX)
        iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                                 out_of_bounds="clip")
        iso.fit(iso_raw, y_iso)

        X_all_z = (X - mu_x) / sd_x
        raw = np.clip(best_clf.predict_proba(X_all_z)[:, 1], PROB_CLIP_MIN, PROB_CLIP_MAX)
        calibrated = np.clip(
            np.interp(np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                      iso.X_thresholds_, iso.y_thresholds_),
            PROB_CLIP_MIN, PROB_CLIP_MAX)
        fold_preds.append(calibrated)

    if not fold_preds:
        return contract_df["model_prob"].values.copy()

    ensemble = np.mean(fold_preds, axis=0)
    return _per_day_renorm(
        np.clip(ensemble, PROB_CLIP_MIN, PROB_CLIP_MAX),
        contract_df["date"].values)


def apply_u9_kitchen_sink(contract_df, u2_probs, u5_probs, u7_probs, u8_probs,
                           cal_frac=0.6):
    """U9: Kitchen-sink weighted ensemble of all calibrated variants."""
    y = contract_df["actual_outcome"].values.astype(float)
    n = len(contract_df)
    n_cal = int(n * cal_frac)
    if n_cal < 50:
        return np.mean([u2_probs, u5_probs, u7_probs, u8_probs], axis=0)

    cal_outcomes = y[:n_cal]
    variants = {"u2": u2_probs, "u5": u5_probs, "u7": u7_probs, "u8": u8_probs}

    cal_briers = {}
    for name, probs in variants.items():
        cal_briers[name] = float(np.mean((probs[:n_cal] - cal_outcomes) ** 2))

    inv_briers = {k: 1.0 / (v + 1e-6) for k, v in cal_briers.items()}
    total = sum(inv_briers.values())
    weights = {k: v / total for k, v in inv_briers.items()}

    blended = sum(w * variants[k] for k, w in weights.items())

    n_iso_train = int(n_cal * 0.7)
    blend_iso_val = blended[n_iso_train:n_cal]
    y_iso_val = cal_outcomes[n_iso_train:]

    iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                             out_of_bounds="clip")
    iso.fit(blend_iso_val, y_iso_val)
    calibrated = np.clip(
        np.interp(np.clip(blended, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
                  iso.X_thresholds_, iso.y_thresholds_),
        PROB_CLIP_MIN, PROB_CLIP_MAX)
    return _per_day_renorm(calibrated, contract_df["date"].values)


# ============================================================================
# Helpers
# ============================================================================

def _ece(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bi = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        m = bi == i
        c = m.sum()
        if c > 0:
            ece += (c / n) * abs(float(probs[m].mean()) - float(outcomes[m].mean()))
    return ece


def _per_day_renorm(probs, dates):
    out = probs.copy()
    for d in np.unique(dates):
        m = dates == d
        s = out[m].sum()
        if s > 0:
            out[m] = out[m] / s
    return np.clip(out, PROB_CLIP_MIN, PROB_CLIP_MAX)


def contract_brier(probs, outcomes):
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(o))
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((p[valid] - o[valid]) ** 2))


# ============================================================================
# Main Pipeline for One City
# ============================================================================

def run_city_benchmark(city_code: str):
    """Run full benchmark for one city."""
    city = get_city_config(city_code)
    ensure_city_dirs(city)
    bucket_edges = city.bucket_edges
    bucket_labels = city.bucket_labels

    print("\n" + "=" * 70)
    print(f"  {city.city_name} ({city.kalshi_ticker}) Unified Benchmark")
    print("=" * 70)

    # Load data
    processed_dir = os.path.join(city.data_dir, "processed")
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(
        processed_dir, city_code)
    kalshi = load_kalshi_data(city_code)

    test_actual = y_test.values
    test_dates = y_test.index
    val_actual = y_val.values

    results = {}

    # ---- E0: Persistence ----
    print("\n--- E0: Persistence Baseline ---")
    e0 = run_persistence(y_train, y_val, y_test)
    e0_probs = gaussian_to_bucket_probs(e0["mu_test"], e0["sigma_test"], bucket_edges)
    e0_brier = compute_brier_score(e0_probs, test_actual, bucket_edges)
    results["E0_persistence"] = {"test_brier": e0_brier}
    print(f"  E0 test Brier: {e0_brier:.4f}")

    # ---- E1: Climatology ----
    print("\n--- E1: Climatology Baseline ---")
    e1 = run_climatology(y_train, y_val, y_test)
    e1_probs = gaussian_to_bucket_probs(e1["mu_test"], e1["sigma_test"], bucket_edges)
    e1_brier = compute_brier_score(e1_probs, test_actual, bucket_edges)
    results["E1_climatology"] = {"test_brier": e1_brier}
    print(f"  E1 test Brier: {e1_brier:.4f}")

    # ---- E2: Ridge ----
    print("\n--- E2: Ridge Regression ---")
    e2 = run_ridge(X_train, y_train, X_val, y_val, X_test, y_test, bucket_edges)
    e2_probs = gaussian_to_bucket_probs(e2["mu_test"], e2["sigma_test"], bucket_edges)
    e2_brier = compute_brier_score(e2_probs, test_actual, bucket_edges)
    results["E2_ridge"] = {"test_brier": e2_brier, "alpha": e2["alpha"]}
    print(f"  E2 test Brier: {e2_brier:.4f}")

    # ---- E3: Heteroscedastic NN [128, 64] ----
    print("\n--- E3: Heteroscedastic NN [128, 64] ---")
    e3 = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                  hidden_sizes=[128, 64], dropout=0.1, lr=0.001, max_epochs=200, patience=20)
    e3_probs = gaussian_to_bucket_probs(e3["mu_test"], e3["sigma_test"], bucket_edges)
    e3_brier = compute_brier_score(e3_probs, test_actual, bucket_edges)
    results["E3_nn_128_64"] = {"test_brier": e3_brier}
    print(f"  E3 test Brier: {e3_brier:.4f}")

    # ---- E4: Deep NN [256, 128, 64] ----
    print("\n--- E4: Deep NN [256, 128, 64] ---")
    e4 = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                  hidden_sizes=[256, 128, 64], dropout=0.12, lr=0.0005, max_epochs=200, patience=20)
    e4_probs = gaussian_to_bucket_probs(e4["mu_test"], e4["sigma_test"], bucket_edges)
    e4_brier = compute_brier_score(e4_probs, test_actual, bucket_edges)
    results["E4_deep_nn"] = {"test_brier": e4_brier}
    print(f"  E4 test Brier: {e4_brier:.4f}")

    # ---- E5: Ensemble NN ----
    print("\n--- E5: Ensemble NN (5 members) ---")
    e5 = train_ensemble_nn(X_train, y_train, X_val, y_val, X_test, y_test, n_models=5)
    e5_probs = gaussian_to_bucket_probs(e5["mu_test"], e5["sigma_test"], bucket_edges)
    e5_brier = compute_brier_score(e5_probs, test_actual, bucket_edges)
    results["E5_ensemble"] = {"test_brier": e5_brier}
    print(f"  E5 test Brier: {e5_brier:.4f}")

    # ---- Pick best base model for Unified variants ----
    best_base = min(["E3_nn_128_64", "E4_deep_nn", "E5_ensemble"],
                    key=lambda k: results[k]["test_brier"])
    print(f"\n  Best base model: {best_base} (Brier {results[best_base]['test_brier']:.4f})")

    # Use ensemble (E5) for unified variants as it's most robust
    # Build combined val+test predictions indexed by date
    all_dates = pd.DatetimeIndex(list(y_val.index) + list(y_test.index))
    all_mu = np.concatenate([e5["mu_val"], e5["mu_test"]])
    all_sigma = np.concatenate([e5["sigma_val"], e5["sigma_test"]])

    # Also generate climatology-adjusted NN predictions for dates beyond test set
    # For dates in 2025+ that aren't in the processed data, use climatology as fallback
    kalshi_dates = pd.to_datetime(kalshi["date"].unique())
    max_processed_date = y_test.index.max()
    future_dates = kalshi_dates[kalshi_dates > max_processed_date]
    if len(future_dates) > 0:
        logger.info("Extending predictions for %d dates beyond processed data (up to %s)",
                     len(future_dates), future_dates.max().strftime("%Y-%m-%d"))
        # Use climatology-enhanced NN prediction for future dates
        # Get DOY climatology from train+val+test
        all_y_combined = pd.concat([y_train, y_val, y_test])
        doy_mean = all_y_combined.groupby(all_y_combined.index.dayofyear).mean()
        doy_std = all_y_combined.groupby(all_y_combined.index.dayofyear).std().clip(lower=3.0)
        doy_mean = doy_mean.reindex(np.arange(1, 367)).interpolate().bfill().ffill()
        doy_std = doy_std.reindex(np.arange(1, 367)).interpolate().bfill().ffill()

        # For future dates, estimate mu from DOY climatology adjusted by recent model bias
        # Use last 90 days of test predictions to estimate model correction
        recent_mask = y_test.index >= (max_processed_date - pd.Timedelta(days=90))
        recent_idx = np.array(recent_mask)
        recent_actuals = y_test[recent_mask].values
        recent_mu = e5["mu_test"][recent_idx]
        recent_bias = float(np.mean(recent_mu - recent_actuals))
        recent_sigma_scale = float(np.mean(e5["sigma_test"][recent_idx]))

        future_doys = future_dates.dayofyear
        future_mu = doy_mean.reindex(future_doys).values - recent_bias * 0.5
        future_sigma = doy_std.reindex(future_doys).values * (recent_sigma_scale / np.mean(e5["sigma_test"]))
        future_sigma = np.maximum(future_sigma, 3.0)

        all_dates = pd.DatetimeIndex(list(all_dates) + list(future_dates))
        all_mu = np.concatenate([all_mu, future_mu])
        all_sigma = np.concatenate([all_sigma, future_sigma])

    mu_series = pd.Series(all_mu, index=all_dates.strftime("%Y-%m-%d"))
    sigma_series = pd.Series(all_sigma, index=all_dates.strftime("%Y-%m-%d"))

    # ---- Build contract-level dataset ----
    print("\n--- Building contract-level dataset with Kalshi data ---")
    cdf = build_contract_dataset(kalshi, mu_series, sigma_series,
                                 bucket_edges, bucket_labels)

    if len(cdf) == 0:
        print("  WARNING: No contract data available. Skipping Unified variants.")
        return results

    outcomes = cdf["actual_outcome"].values.astype(float)
    print(f"  Contract dataset: {len(cdf)} rows, {cdf['date'].nunique()} dates")
    print(f"  Date range: {cdf['date'].min()} to {cdf['date'].max()}")

    # ---- U0: Raw Gaussian ----
    print("\n--- U0: Raw Gaussian bucket probs ---")
    u0_probs = apply_u0_raw(cdf)
    u0_brier = contract_brier(u0_probs, outcomes)
    results["U0_raw_gaussian"] = {"contract_brier": u0_brier}
    print(f"  U0 contract Brier: {u0_brier:.4f}")

    # ---- U1: Isotonic ----
    print("\n--- U1: Isotonic calibration ---")
    u1_probs = apply_u1_isotonic(cdf)
    u1_brier = contract_brier(u1_probs, outcomes)
    results["U1_isotonic"] = {"contract_brier": u1_brier}
    print(f"  U1 contract Brier: {u1_brier:.4f}")

    # ---- U2: Contract Ridge ----
    print("\n--- U2: Contract-level Ridge ---")
    u2_probs = apply_u2_ridge(cdf)
    u2_brier = contract_brier(u2_probs, outcomes)
    results["U2_contract_ridge"] = {"contract_brier": u2_brier}
    print(f"  U2 contract Brier: {u2_brier:.4f}")

    # ---- U3: Contract MLP ----
    print("\n--- U3: Contract-level Brier MLP ---")
    u3_probs = apply_u3_mlp(cdf)
    u3_brier = contract_brier(u3_probs, outcomes)
    results["U3_contract_mlp"] = {"contract_brier": u3_brier}
    print(f"  U3 contract Brier: {u3_brier:.4f}")

    # ---- U4: Platt on U3 ----
    print("\n--- U4: Platt recalibration on U3 ---")
    u4_probs = apply_u4_platt(cdf, u3_probs)
    u4_brier = contract_brier(u4_probs, outcomes)
    results["U4_platt_on_u3"] = {"contract_brier": u4_brier}
    print(f"  U4 contract Brier: {u4_brier:.4f}")

    # ---- U5: Regime-conditional ----
    print("\n--- U5: Regime-conditional ---")
    u5_probs = apply_u5_regime(cdf)
    u5_brier = contract_brier(u5_probs, outcomes)
    results["U5_regime_conditional"] = {"contract_brier": u5_brier}
    print(f"  U5 contract Brier: {u5_brier:.4f}")

    # ---- U6: Calibrated ensemble ----
    print("\n--- U6: Calibrated ensemble (U1+U4+U5) ---")
    u6_probs = apply_u6_ensemble(u1_probs, u4_probs, u5_probs)
    u6_brier = contract_brier(u6_probs, outcomes)
    results["U6_calibrated_ensemble"] = {"contract_brier": u6_brier}
    print(f"  U6 contract Brier: {u6_brier:.4f}")

    # ---- U7: Extended-feature MLP ----
    print("\n--- U7: Extended-feature contract MLP ---")
    u7_probs = apply_u7_extended_mlp(cdf)
    u7_brier = contract_brier(u7_probs, outcomes)
    results["U7_extended_mlp"] = {"contract_brier": u7_brier}
    print(f"  U7 contract Brier: {u7_brier:.4f}")

    # ---- U8: Cross-validated ensemble ----
    print("\n--- U8: Cross-validated MLP ensemble ---")
    u8_probs = apply_u8_cv_ensemble(cdf)
    u8_brier = contract_brier(u8_probs, outcomes)
    results["U8_cv_ensemble"] = {"contract_brier": u8_brier}
    print(f"  U8 contract Brier: {u8_brier:.4f}")

    # ---- U9: Kitchen sink ----
    print("\n--- U9: Kitchen-sink weighted ensemble ---")
    u9_probs = apply_u9_kitchen_sink(cdf, u2_probs, u5_probs, u7_probs, u8_probs)
    u9_brier = contract_brier(u9_probs, outcomes)
    results["U9_kitchen_sink"] = {"contract_brier": u9_brier}
    print(f"  U9 contract Brier: {u9_brier:.4f}")

    # ---- IS/OOS Split for Contract Brier ----
    # For CHI: data starts 2022, use 2025+ as OOS
    # For PHL: data starts 2024-11, use last 30% as OOS
    cdf_dates = pd.to_datetime(cdf["date"])
    n_contract = len(cdf)
    cal_n = int(n_contract * 0.6)

    if city_code == "chi":
        oos_mask = (cdf_dates.dt.year >= 2025).values
    else:
        oos_mask = np.zeros(n_contract, dtype=bool)
        oos_mask[cal_n:] = True

    is_mask = ~oos_mask
    print(f"\n  IS/OOS split: IS={is_mask.sum()} rows, OOS={oos_mask.sum()} rows")

    # Report OOS Brier for all unified variants
    # Add presettlement market baseline
    market_probs = np.clip(cdf["market_prob"].values.astype(float),
                           PROB_CLIP_MIN, PROB_CLIP_MAX)
    mkt_brier = contract_brier(market_probs, outcomes)
    results["Kalshi_PreSettlement"] = {"contract_brier": mkt_brier}
    print(f"\n  Kalshi Pre-Settlement baseline Brier: {mkt_brier:.4f}")

    all_u_probs = {
        "U0_raw_gaussian": u0_probs,
        "U1_isotonic": u1_probs,
        "U2_contract_ridge": u2_probs,
        "U3_contract_mlp": u3_probs,
        "U4_platt_on_u3": u4_probs,
        "U5_regime_conditional": u5_probs,
        "U6_calibrated_ensemble": u6_probs,
        "U7_extended_mlp": u7_probs,
        "U8_cv_ensemble": u8_probs,
        "U9_kitchen_sink": u9_probs,
    }

    print(f"\n  {'Model':<30} {'Overall':>10} {'IS':>10} {'OOS':>10}")
    print("  " + "-" * 62)
    for name, probs in all_u_probs.items():
        overall = contract_brier(probs, outcomes)
        is_brier = contract_brier(probs[is_mask], outcomes[is_mask]) if is_mask.any() else float("nan")
        oos_brier = contract_brier(probs[oos_mask], outcomes[oos_mask]) if oos_mask.any() else float("nan")
        results[name]["is_brier"] = is_brier
        results[name]["oos_brier"] = oos_brier
        print(f"  {name:<28} {overall:>10.4f} {is_brier:>10.4f} {oos_brier:>10.4f}")

    # Best OOS model
    best_oos = min([k for k in results if k.startswith("U")],
                   key=lambda k: results[k].get("oos_brier", float("inf")))
    print(f"\n  Best OOS model: {best_oos} "
          f"(OOS Brier {results[best_oos]['oos_brier']:.4f})")

    # Seasonal breakdown for best OOS model
    best_oos_probs = all_u_probs[best_oos]
    print(f"\n  Seasonal breakdown for {best_oos} (OOS only):")
    for season in ["DJF", "MAM", "JJA", "SON"]:
        season_months = [m for m, s in SEASON_MAP.items() if s == season]
        s_mask = oos_mask & cdf_dates.dt.month.isin(season_months).values
        if s_mask.any():
            s_brier = contract_brier(best_oos_probs[s_mask], outcomes[s_mask])
            print(f"    {season}: {s_brier:.4f} (n={s_mask.sum()})")

    # ---- Save results ----
    os.makedirs(city.results_dir, exist_ok=True)
    results_path = os.path.join(city.results_dir, "unified_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Sanity check: no contract-level Brier < 0.03
    for name, res in results.items():
        if "contract_brier" not in res:
            continue
        brier = res["contract_brier"]
        if isinstance(brier, (int, float)) and brier < 0.03:
            logger.error("SANITY CHECK: %s Brier=%.4f < 0.03!", name, brier)

    # Save predictions CSV
    cols = ["date", "ticker", "direction", "threshold_low",
            "threshold_high", "actual_outcome", "actual_tmax",
            "market_prob", "model_mu", "model_sigma", "model_prob"]
    if "presettlement_prob" in cdf.columns:
        cols.append("presettlement_prob")
    pred_df = cdf[[c for c in cols if c in cdf.columns]].copy()
    if "bucket" in cdf.columns:
        pred_df.insert(2, "bucket", cdf["bucket"])
    pred_df["u3_mlp_prob"] = u3_probs
    pred_df["u4_platt_prob"] = u4_probs
    pred_df["u5_regime_prob"] = u5_probs
    pred_df["u6_ensemble_prob"] = u6_probs
    pred_df["u7_extended_prob"] = u7_probs
    pred_df["u8_cv_prob"] = u8_probs
    pred_df["u9_kitchen_prob"] = u9_probs
    pred_df["period"] = np.where(oos_mask, "OOS", "IS")
    pred_df["season"] = cdf["season"].values
    pred_path = os.path.join(city.results_dir, "unified_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print(f"  {city.city_name} BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<28} {'Brier':>10} {'OOS Brier':>12} {'Type':>12}")
    print("  " + "-" * 65)
    for name, res in sorted(results.items()):
        brier = res.get("contract_brier", res.get("test_brier", float("nan")))
        oos = res.get("oos_brier", "")
        btype = "contract" if "contract_brier" in res else "bucket-day"
        oos_str = f"{oos:.4f}" if isinstance(oos, float) and not np.isnan(oos) else ""
        print(f"  {name:<28} {brier:>10.4f} {oos_str:>12} {btype:>12}")
    print("=" * 70)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "#" * 70)
    print("#  CHI + PHL Unified Model Benchmark with Real Kalshi Data")
    print("#" * 70)

    all_results = {}

    for city_code in ["chi", "phl"]:
        try:
            results = run_city_benchmark(city_code)
            all_results[city_code] = results
        except Exception as e:
            logger.error("Failed to benchmark %s: %s", city_code, e, exc_info=True)
            all_results[city_code] = {"error": str(e)}

    # Final cross-city summary
    print("\n" + "#" * 70)
    print("#  CROSS-CITY SUMMARY")
    print("#" * 70)
    for city_code, results in all_results.items():
        if "error" in results:
            print(f"  {city_code.upper()}: ERROR - {results['error']}")
            continue
        best_e = min([k for k in results if k.startswith("E")],
                     key=lambda k: results[k].get("test_brier", float("inf")),
                     default="N/A")
        best_u = min([k for k in results if k.startswith("U")],
                     key=lambda k: results[k].get("contract_brier", float("inf")),
                     default="N/A")
        if best_e != "N/A":
            print(f"  {city_code.upper()} best E-series: {best_e} "
                  f"(bucket-day Brier {results[best_e]['test_brier']:.4f})")
        if best_u != "N/A":
            print(f"  {city_code.upper()} best Unified:  {best_u} "
                  f"(contract Brier {results[best_u]['contract_brier']:.4f})")

    # Save combined results
    combined_path = PROJECT_ROOT / "results" / "chi_phl_unified_benchmark_results.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results: {combined_path}")


if __name__ == "__main__":
    main()
