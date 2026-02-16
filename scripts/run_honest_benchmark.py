#!/usr/bin/env python3
"""
Honest Benchmark for Chicago and Philadelphia.

This script removes ALL sources of data leakage identified in the
model cheating investigation. Specifically:

1. NO Kalshi market_prob used as features (removes settlement leakage)
2. NO settlement-price benchmarks presented as forecasting baselines
3. Only bucket-day Brier computed from Gaussian (mu, sigma) predictions
4. Strict chronological splits with no information leakage

Models benchmarked:
  E0: Persistence baseline (yesterday's TMAX)
  E1: Climatology baseline (day-of-year mean/std)
  E2: Ridge regression (best alpha on val)
  E3: Heteroscedastic NN [128, 64]
  E4: Deep Heteroscedastic NN [256, 128, 64]
  E5: Ensemble of 3 NNs

All models output (mu, sigma) → Gaussian CDF → bucket probabilities.
Brier score is computed per (day, bucket) and averaged.

Usage:
    python scripts/run_honest_benchmark.py
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
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Data Loading (NO leakage)
# ============================================================================

def load_processed_data(processed_dir: str, city_code: str):
    """Load preprocessed CSV files with NaN imputation.

    Uses the original chronological splits from preprocessing.
    Training mean is used for NaN imputation — no leakage.
    """
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

    # Drop columns that are entirely NaN in training
    all_nan_cols = X_train.columns[X_train.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info("Dropping %d all-NaN columns", len(all_nan_cols))
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    # Impute NaNs with training column means only
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_val = X_val.fillna(train_means)
    X_test = X_test.fillna(train_means)

    for s in (y_train, y_val, y_test):
        s.name = f"{city_code.upper()}_TMAX"

    logger.info("Loaded %s: train=%s, val=%s, test=%s",
                city_code, X_train.shape, X_val.shape, X_test.shape)
    logger.info("  Train: %s to %s", X_train.index.min().date(), X_train.index.max().date())
    logger.info("  Val:   %s to %s", X_val.index.min().date(), X_val.index.max().date())
    logger.info("  Test:  %s to %s", X_test.index.min().date(), X_test.index.max().date())

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Bucket Probability Helpers (no leakage)
# ============================================================================

def gaussian_to_bucket_probs(mu, sigma, bucket_edges):
    """Convert N(mu, sigma) to bucket probabilities via CDF."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.5)
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
    """Compute overall bucket-day Brier score.

    For each (day, bucket), Brier = (predicted_prob - outcome)^2.
    Returns mean across all (day, bucket) pairs.
    """
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
    return float(np.mean((bucket_probs - outcomes) ** 2))


def compute_seasonal_brier(bucket_probs, actual_tmax, dates, bucket_edges):
    """Compute Brier per meteorological season."""
    months = dates.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    results = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == s
        if mask.any():
            results[s] = compute_brier_score(
                bucket_probs[mask], actual_tmax[mask], bucket_edges)
    return results


def compute_mae(actual, predicted):
    """Compute mean absolute error."""
    return float(np.mean(np.abs(actual - predicted)))


# ============================================================================
# Models (all produce mu, sigma — no market data used)
# ============================================================================

def run_persistence(y_train, y_val, y_test):
    """Persistence baseline: mu = yesterday's TMAX."""
    all_y = pd.concat([y_train, y_val, y_test])
    y_prev = all_y.shift(1)
    train_mask = all_y.index.isin(y_train.index)
    sigma = max(float((all_y[train_mask] - y_prev[train_mask]).dropna().std()), 3.0)
    mu_val = np.where(np.isnan(y_prev.reindex(y_val.index).values),
                      float(y_train.mean()), y_prev.reindex(y_val.index).values)
    mu_test = np.where(np.isnan(y_prev.reindex(y_test.index).values),
                       float(y_train.mean()), y_prev.reindex(y_test.index).values)
    return {"mu_val": mu_val, "sigma_val": np.full_like(mu_val, sigma),
            "mu_test": mu_test, "sigma_test": np.full_like(mu_test, sigma),
            "sigma": sigma}


def run_climatology(y_train, y_val, y_test):
    """Climatology baseline: DOY mean/std from training set only."""
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
    """Ridge regression: best alpha selected on validation Brier."""
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
                        "alpha": alpha, "sigma": sigma}
    logger.info("Ridge best alpha=%.2f, val Brier=%.6f", best_alpha, best_brier)
    return best_res


class HeteroscedasticNet(nn.Module):
    """Feedforward NN with heteroscedastic Gaussian output."""
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
    logger.info("NN (hidden=%s) best epoch %d, val NLL %.4f",
                hidden_sizes, best_epoch, best_val_loss)

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
            "best_epoch": best_epoch}


def train_ensemble_nn(X_train, y_train, X_val, y_val, X_test, y_test, n_models=3):
    """Train ensemble of NNs, average mu/sigma predictions."""
    configs = [
        ([128, 64], 0.1, 0.001),
        ([256, 128], 0.15, 0.0008),
        ([128, 64, 32], 0.1, 0.001),
    ]
    mu_vals, sig_vals = [], []
    mu_tests, sig_tests = [], []
    for i, (hs, do, lr_i) in enumerate(configs[:n_models]):
        logger.info("  Ensemble member %d/%d: hidden=%s", i+1, n_models, hs)
        res = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                       hidden_sizes=hs, dropout=do, lr=lr_i,
                       max_epochs=150, patience=15)
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
# Plotting
# ============================================================================

def plot_honest_comparison(results, save_path, city_name):
    """Bar chart of honest bucket-day Brier scores."""
    models = list(results.keys())
    briers = [results[m]["test_brier"] for m in models]

    colors = []
    for m in models:
        if "Persist" in m or "Climatology" in m:
            colors.append("#95a5a6")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, briers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00005,
                f"{val:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Bucket-Day Brier Score (lower is better)")
    ax.set_title(f"{city_name} HONEST Benchmark: No Market Data Leakage\n"
                 f"Gray = Baselines | Blue = Trained Models")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", save_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_city_benchmark(city_code: str):
    """Run honest benchmark for one city."""
    city = get_city_config(city_code)
    ensure_city_dirs(city)
    bucket_edges = city.bucket_edges
    bucket_labels = city.bucket_labels

    print("\n" + "=" * 70)
    print(f"  {city.city_name} ({city.kalshi_ticker}) HONEST Benchmark")
    print(f"  NO market data leakage. Bucket-day Brier only.")
    print("=" * 70)

    processed_dir = os.path.join(city.data_dir, "processed")
    results_dir = city.results_dir
    os.makedirs(results_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(
        processed_dir, city_code)

    test_actual = y_test.values
    test_dates = y_test.index
    val_actual = y_val.values

    results = {}

    # --- E0: Persistence ---
    print("\n--- E0: Persistence Baseline ---")
    e0 = run_persistence(y_train, y_val, y_test)
    e0_probs = gaussian_to_bucket_probs(e0["mu_test"], e0["sigma_test"], bucket_edges)
    e0_brier = compute_brier_score(e0_probs, test_actual, bucket_edges)
    e0_mae = compute_mae(test_actual, e0["mu_test"])
    e0_seasonal = compute_seasonal_brier(e0_probs, test_actual, test_dates, bucket_edges)
    results["E0_Persistence"] = {
        "test_brier": e0_brier, "test_mae": e0_mae,
        "sigma": e0["sigma"], "seasonal": e0_seasonal,
    }
    print(f"  Brier: {e0_brier:.6f}  |  MAE: {e0_mae:.2f}F  |  sigma: {e0['sigma']:.2f}")

    # --- E1: Climatology ---
    print("\n--- E1: Climatology Baseline ---")
    e1 = run_climatology(y_train, y_val, y_test)
    e1_probs = gaussian_to_bucket_probs(e1["mu_test"], e1["sigma_test"], bucket_edges)
    e1_brier = compute_brier_score(e1_probs, test_actual, bucket_edges)
    e1_mae = compute_mae(test_actual, e1["mu_test"])
    e1_seasonal = compute_seasonal_brier(e1_probs, test_actual, test_dates, bucket_edges)
    results["E1_Climatology"] = {
        "test_brier": e1_brier, "test_mae": e1_mae, "seasonal": e1_seasonal,
    }
    print(f"  Brier: {e1_brier:.6f}  |  MAE: {e1_mae:.2f}F")

    # --- E2: Ridge ---
    print("\n--- E2: Ridge Regression ---")
    e2 = run_ridge(X_train, y_train, X_val, y_val, X_test, y_test, bucket_edges)
    e2_probs = gaussian_to_bucket_probs(e2["mu_test"], e2["sigma_test"], bucket_edges)
    e2_brier = compute_brier_score(e2_probs, test_actual, bucket_edges)
    e2_mae = compute_mae(test_actual, e2["mu_test"])
    e2_seasonal = compute_seasonal_brier(e2_probs, test_actual, test_dates, bucket_edges)
    results[f"E2_Ridge(a={e2['alpha']})"] = {
        "test_brier": e2_brier, "test_mae": e2_mae,
        "alpha": e2["alpha"], "sigma": e2["sigma"],
        "seasonal": e2_seasonal,
    }
    print(f"  Brier: {e2_brier:.6f}  |  MAE: {e2_mae:.2f}F  |  alpha: {e2['alpha']}")

    # --- E3: Heteroscedastic NN [128, 64] ---
    print("\n--- E3: Heteroscedastic NN [128, 64] ---")
    e3 = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                  hidden_sizes=[128, 64], dropout=0.1, lr=0.001)
    e3_probs = gaussian_to_bucket_probs(e3["mu_test"], e3["sigma_test"], bucket_edges)
    e3_brier = compute_brier_score(e3_probs, test_actual, bucket_edges)
    e3_mae = compute_mae(test_actual, e3["mu_test"])
    e3_seasonal = compute_seasonal_brier(e3_probs, test_actual, test_dates, bucket_edges)
    results["E3_HeteroNN_128_64"] = {
        "test_brier": e3_brier, "test_mae": e3_mae,
        "best_epoch": e3["best_epoch"], "seasonal": e3_seasonal,
    }
    print(f"  Brier: {e3_brier:.6f}  |  MAE: {e3_mae:.2f}F  |  best_epoch: {e3['best_epoch']}")

    # --- E4: Deep NN [256, 128, 64] ---
    print("\n--- E4: Deep NN [256, 128, 64] ---")
    e4 = train_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                  hidden_sizes=[256, 128, 64], dropout=0.12, lr=0.0005)
    e4_probs = gaussian_to_bucket_probs(e4["mu_test"], e4["sigma_test"], bucket_edges)
    e4_brier = compute_brier_score(e4_probs, test_actual, bucket_edges)
    e4_mae = compute_mae(test_actual, e4["mu_test"])
    e4_seasonal = compute_seasonal_brier(e4_probs, test_actual, test_dates, bucket_edges)
    results["E4_DeepNN_256_128_64"] = {
        "test_brier": e4_brier, "test_mae": e4_mae,
        "best_epoch": e4["best_epoch"], "seasonal": e4_seasonal,
    }
    print(f"  Brier: {e4_brier:.6f}  |  MAE: {e4_mae:.2f}F  |  best_epoch: {e4['best_epoch']}")

    # --- E5: Ensemble NN ---
    print("\n--- E5: Ensemble NN (3 members) ---")
    e5 = train_ensemble_nn(X_train, y_train, X_val, y_val, X_test, y_test, n_models=3)
    e5_probs = gaussian_to_bucket_probs(e5["mu_test"], e5["sigma_test"], bucket_edges)
    e5_brier = compute_brier_score(e5_probs, test_actual, bucket_edges)
    e5_mae = compute_mae(test_actual, e5["mu_test"])
    e5_seasonal = compute_seasonal_brier(e5_probs, test_actual, test_dates, bucket_edges)
    results["E5_EnsembleNN"] = {
        "test_brier": e5_brier, "test_mae": e5_mae, "seasonal": e5_seasonal,
    }
    print(f"  Brier: {e5_brier:.6f}  |  MAE: {e5_mae:.2f}F")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"  {city.city_name} HONEST BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<28} {'Brier':>12} {'MAE (F)':>10} {'DJF':>10} {'MAM':>10} {'JJA':>10} {'SON':>10}")
    print("  " + "-" * 92)

    sorted_models = sorted(results.items(), key=lambda x: x[1]["test_brier"])
    for name, res in sorted_models:
        seasonal = res.get("seasonal", {})
        print(f"  {name:<28} {res['test_brier']:>12.6f} {res['test_mae']:>10.2f}"
              f" {seasonal.get('DJF', float('nan')):>10.6f}"
              f" {seasonal.get('MAM', float('nan')):>10.6f}"
              f" {seasonal.get('JJA', float('nan')):>10.6f}"
              f" {seasonal.get('SON', float('nan')):>10.6f}")

    # Improvement over baselines
    persist_brier = results["E0_Persistence"]["test_brier"]
    clim_brier = results["E1_Climatology"]["test_brier"]
    best_model_name = sorted_models[0][0]
    best_brier = sorted_models[0][1]["test_brier"]

    print(f"\n  Best model: {best_model_name}")
    print(f"  Improvement vs Persistence: {100*(persist_brier - best_brier)/persist_brier:.2f}%")
    print(f"  Improvement vs Climatology: {100*(clim_brier - best_brier)/clim_brier:.2f}%")
    print("=" * 70)

    # ---- Save ----
    os.makedirs(results_dir, exist_ok=True)

    # Clean results for JSON
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in res.items()
            if k != "seasonal"
        }
        if "seasonal" in res:
            json_results[name]["seasonal"] = {
                k: float(v) for k, v in res["seasonal"].items()
            }

    results_path = os.path.join(results_dir, f"honest_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Summary CSV
    summary_rows = []
    for name, res in sorted_models:
        row = {"model": name, "test_brier": res["test_brier"], "test_mae": res["test_mae"]}
        for s in ["DJF", "MAM", "JJA", "SON"]:
            row[f"brier_{s}"] = res.get("seasonal", {}).get(s)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, f"honest_benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")

    # Plot
    plot_honest_comparison(
        {name: res for name, res in sorted_models},
        os.path.join(results_dir, f"honest_benchmark_comparison.png"),
        city.city_name,
    )

    return results


def main():
    print("\n" + "#" * 70)
    print("#  HONEST BENCHMARK (No Market Data Leakage)")
    print("#  Chicago + Philadelphia")
    print("#" * 70)

    all_results = {}
    for city_code in ["chi", "phl"]:
        try:
            all_results[city_code] = run_city_benchmark(city_code)
        except Exception as e:
            logger.error("Failed %s: %s", city_code, e, exc_info=True)

    # Cross-city summary
    print("\n" + "#" * 70)
    print("#  CROSS-CITY HONEST SUMMARY")
    print("#" * 70)
    for city_code, results in all_results.items():
        sorted_models = sorted(results.items(), key=lambda x: x[1]["test_brier"])
        best_name = sorted_models[0][0]
        best_brier = sorted_models[0][1]["test_brier"]
        persist_brier = results["E0_Persistence"]["test_brier"]
        print(f"  {city_code.upper()}: Best = {best_name} (Brier {best_brier:.6f})")
        print(f"         Persistence baseline: {persist_brier:.6f}")
        print(f"         Improvement: {100*(persist_brier - best_brier)/persist_brier:.2f}%")


if __name__ == "__main__":
    main()
