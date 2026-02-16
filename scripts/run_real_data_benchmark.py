#!/usr/bin/env python3
"""
Real-Data Benchmark for Chicago and Philadelphia.

Trains and evaluates the full model suite against real NWS MOS forecasts
(not climatological proxies). Uses actual MOS ensemble Tmax predictions
from the IEM archive as the primary external benchmark.

Benchmarks:
  - Real NWS MOS (GFS+NAM ensemble) — from data/{city}/mos/combined_mos_*.csv
  - Climatology baseline — DOY mean/std from training data
  - Persistence baseline — yesterday's TMAX

Models:
  1. Ridge regression (best alpha on val Brier)
  2. Heteroscedastic NN (baseline)
  3. FeatureAttentionNet (dynamic feature importance)
  4. MOSCorrectionNet (error correction over MOS/climatology)
  5. RegimeConditionalNet (season x volatility regime)
  6. CalibratedEnsemble (stacking + isotonic + Platt)

Usage:
    python scripts/run_real_data_benchmark.py --city chi
    python scripts/run_real_data_benchmark.py --city phl
    python scripts/run_real_data_benchmark.py --city both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs
from src.advanced_model import (
    FeatureAttentionNet,
    MOSCorrectionNet,
    RegimeConditionalNet,
    IsotonicPlattCalibrator,
    EnsembleStacker,
    compute_regime_features,
    compute_mos_baseline,
    add_enhanced_features,
    gaussian_to_bucket_probs,
    compute_brier_score,
    compute_seasonal_brier,
    train_model,
    predict_model,
)
from src.mos_market_proxy import MOSMarketProxy
from src.contract_brier import (
    load_city_kalshi_contract_rows,
    contract_probabilities_from_gaussian,
    contract_brier_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KALSHI_SETTLED_PATHS = [
    PROJECT_ROOT / "data" / "real_kalshi_2023_2024.csv",
    PROJECT_ROOT / "data" / "real_kalshi_2025.csv",
]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_and_split_data(
    processed_dir: str,
    city_code: str,
) -> tuple:
    """Load preprocessed city data and split chronologically.

    Splits:
      - Train: before 2022-01-01
      - Val:   2022-01-01 to 2023-12-31
      - Test:  2024-01-01 onward
    """
    X_parts, y_parts = [], []
    tmax_col = {"chi": "CHI_TMAX", "phl": "PHL_TMAX"}[city_code]

    for split in ["train", "val", "test"]:
        feat_path = os.path.join(processed_dir, f"features_{split}.csv")
        targ_path = os.path.join(processed_dir, f"target_{split}.csv")
        if not os.path.exists(feat_path) or not os.path.exists(targ_path):
            raise FileNotFoundError(f"Missing {feat_path} or {targ_path}")
        X = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        y = pd.read_csv(targ_path, index_col=0, parse_dates=True).iloc[:, 0]
        X_parts.append(X)
        y_parts.append(y)

    X_full = pd.concat(X_parts, axis=0).sort_index()
    y_full = pd.concat(y_parts, axis=0).sort_index()
    y_full.name = tmax_col

    # Chronological splits
    train_mask = X_full.index < "2022-01-01"
    val_mask = (X_full.index >= "2022-01-01") & (X_full.index < "2024-01-01")
    test_mask = X_full.index >= "2024-01-01"

    X_train, X_val, X_test = X_full[train_mask], X_full[val_mask], X_full[test_mask]
    y_train, y_val, y_test = y_full[train_mask], y_full[val_mask], y_full[test_mask]

    # Filter training to start from 2000
    start = X_train.index >= "2000-01-01"
    X_train, y_train = X_train[start], y_train[start]

    logger.info(
        "Data splits: train=%d (%s to %s), val=%d (%s to %s), test=%d (%s to %s)",
        len(X_train), X_train.index.min().date(), X_train.index.max().date(),
        len(X_val), X_val.index.min().date(), X_val.index.max().date(),
        len(X_test), X_test.index.min().date(), X_test.index.max().date(),
    )

    # Drop columns that are all-NaN or constant in training set
    valid_cols = X_train.columns[X_train.std().fillna(0) > 0]
    X_train = X_train[valid_cols].fillna(0)
    X_val = X_val[valid_cols].fillna(0)
    X_test = X_test[valid_cols].fillna(0)

    # Re-scale: fit on training data only
    scaler = StandardScaler()
    cols = X_train.columns
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=cols)

    # Replace any remaining NaN from scaling
    X_train_s = X_train_s.fillna(0)
    X_val_s = X_val_s.fillna(0)
    X_test_s = X_test_s.fillna(0)

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test


# ===========================================================================
# Real NWS MOS Benchmark
# ===========================================================================

def compute_real_mos_benchmark(
    city_code: str,
    target_station_csv: str,
    mos_csv: str,
    test_dates: pd.DatetimeIndex,
    test_actual: np.ndarray,
    bucket_edges: list,
    train_end_date: str = "2021-12-31",
) -> dict:
    """Compute Brier score from real NWS MOS ensemble forecasts.

    Uses MOSMarketProxy with actual MOS data (not climatological proxy).
    """
    # Load target station TMAX
    station_df = pd.read_csv(target_station_csv, parse_dates=["date"])
    actual_df = station_df[["date", "TMAX"]].rename(columns={"TMAX": "tmax_f"}).dropna()

    # Load real MOS data
    mos_df = pd.read_csv(mos_csv, parse_dates=["date"])
    logger.info(
        "Loaded real MOS data: %d rows (%s to %s)",
        len(mos_df), mos_df["date"].min().date(), mos_df["date"].max().date(),
    )

    # Fit MOSMarketProxy on training period
    proxy = MOSMarketProxy(mos_df, actual_df)
    proxy.fit(train_end_date=train_end_date)

    diag = proxy.get_diagnostics()
    logger.info(
        "MOS proxy diagnostics: MAE=%.2f, RMSE=%.2f, bias=%.2f, sigma=%.2f, n_train=%d",
        diag["overall_mae"], diag["overall_rmse"],
        diag["overall_bias"], diag["overall_sigma"], diag["n_train_days"],
    )

    # Generate predictions for test dates
    n_test = len(test_dates)
    mu_mos = np.zeros(n_test)
    sigma_mos = np.zeros(n_test)
    n_available = 0

    for i, dt in enumerate(test_dates):
        d = dt.date()
        mu, sigma = proxy.predict_mu_sigma(d)
        mu_mos[i] = mu
        sigma_mos[i] = sigma
        if d in proxy._mos_lookup:
            n_available += 1

    logger.info(
        "MOS coverage on test set: %d/%d days have direct MOS forecasts (%.1f%%)",
        n_available, n_test, 100.0 * n_available / n_test,
    )

    probs = gaussian_to_bucket_probs(mu_mos, sigma_mos, bucket_edges)
    brier = compute_brier_score(probs, test_actual, bucket_edges)
    seasonal = compute_seasonal_brier(probs, test_actual, test_dates, bucket_edges)

    return {
        "mu": mu_mos,
        "sigma": sigma_mos,
        "probs": probs,
        "brier": brier,
        "seasonal": seasonal,
        "diagnostics": diag,
        "mos_coverage_pct": 100.0 * n_available / n_test,
    }


# ===========================================================================
# Baseline Models
# ===========================================================================

def gaussian_nll_loss(mu, sigma, target):
    var = sigma ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + ((target - mu) ** 2) / var)
    return nll.mean()


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


def run_persistence(y_train, y_val, y_test):
    all_y = pd.concat([y_train, y_val, y_test])
    y_prev = all_y.shift(1)
    train_resid = all_y[all_y.index.isin(y_train.index)] - y_prev[y_prev.index.isin(y_train.index)]
    sigma = max(float(train_resid.dropna().std()), 2.0)
    train_mean = float(y_train.mean())

    def _extract(idx):
        mu = y_prev[y_prev.index.isin(idx)].values
        return np.where(np.isnan(mu), train_mean, mu), np.full(len(mu), sigma)

    mu_v, sig_v = _extract(y_val.index)
    mu_t, sig_t = _extract(y_test.index)
    return {"mu_val": mu_v, "sigma_val": sig_v, "mu_test": mu_t, "sigma_test": sig_t}


def run_climatology(y_train, y_val, y_test):
    doy = y_train.index.dayofyear
    clim_mean = y_train.groupby(doy).mean()
    clim_std = y_train.groupby(doy).std()
    all_doys = np.arange(1, 367)
    clim_mean = clim_mean.reindex(all_doys).interpolate().bfill().ffill()
    clim_std = clim_std.reindex(all_doys).interpolate().bfill().ffill().clip(lower=3.0)

    return {
        "mu_val": clim_mean.reindex(y_val.index.dayofyear).values,
        "sigma_val": clim_std.reindex(y_val.index.dayofyear).values,
        "mu_test": clim_mean.reindex(y_test.index.dayofyear).values,
        "sigma_test": clim_std.reindex(y_test.index.dayofyear).values,
    }


def run_ridge(X_train, y_train, X_val, y_val, X_test, bucket_edges):
    best = {"brier": float("inf")}
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train.values, y_train.values)
        mu_v = model.predict(X_val.values)
        mu_t = model.predict(X_test.values)
        sig = max(float(np.std(y_train.values - model.predict(X_train.values))), 2.0)
        probs_v = gaussian_to_bucket_probs(mu_v, np.full(len(mu_v), sig), bucket_edges)
        brier_v = compute_brier_score(probs_v, y_val.values, bucket_edges)["overall_brier"]
        if brier_v < best["brier"]:
            best = {
                "brier": brier_v, "alpha": alpha,
                "mu_val": mu_v, "mu_test": mu_t, "sigma": sig,
            }
    return {
        "mu_val": best["mu_val"],
        "sigma_val": np.full(len(best["mu_val"]), best["sigma"]),
        "mu_test": best["mu_test"],
        "sigma_test": np.full(len(best["mu_test"]), best["sigma"]),
        "alpha": best["alpha"],
    }


def train_hetero_nn(X_train, y_train, X_val, y_val, X_test):
    n_feat = X_train.shape[1]
    model = HeteroscedasticNet(n_feat, [128, 64], 0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

    X_tr = torch.tensor(X_train.values, dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    X_vt = torch.tensor(X_val.values, dtype=torch.float32).to(DEVICE)
    y_vt = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    best_loss, best_state = float("inf"), None
    patience_count = 0
    for epoch in range(1, 201):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            mu, sig = model(xb)
            loss = gaussian_nll_loss(mu, sig, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, sig_v = model(X_vt)
            vl = gaussian_nll_loss(mu_v, sig_v, y_vt).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 20:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mu_v, sig_v = model(X_vt)
        X_tt = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
        mu_t, sig_t = model(X_tt)
    return {
        "mu_val": mu_v.cpu().numpy().ravel(),
        "sigma_val": sig_v.cpu().numpy().ravel(),
        "mu_test": mu_t.cpu().numpy().ravel(),
        "sigma_test": sig_t.cpu().numpy().ravel(),
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_brier_comparison(results, save_path, city_name):
    models = list(results.keys())
    briers = [results[m]["test_brier"] for m in models]

    # Color benchmarks differently
    colors = []
    for m in models:
        if "MOS" in m and "Net" not in m:
            colors.append("#e74c3c")  # red for real MOS benchmark
        elif "Persist" in m or "Climatology" in m:
            colors.append("#95a5a6")  # gray for simple baselines
        else:
            colors.append("#3498db")  # blue for our models

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, briers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title(f"{city_name} Real-Data Benchmark: Test-Set Brier Scores\n"
                 f"Red = Real NWS MOS | Blue = Our Models | Gray = Simple Baselines")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main Benchmark Runner
# ===========================================================================

def run_city_benchmark(city_code: str) -> dict:
    """Run the full real-data benchmark for a single city."""
    city = get_city_config(city_code)
    ensure_city_dirs(city)
    city_name = city.city_name
    bucket_edges = city.bucket_edges

    logger.info("=" * 70)
    logger.info("%s REAL-DATA BENCHMARK (%s)", city_name.upper(), city.kalshi_ticker)
    logger.info("Train: 2000-2021 | Val: 2022-2023 | Test: 2024+")
    logger.info("=" * 70)

    processed_dir = os.path.join(city.data_dir, "processed")
    results_dir = city.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # --- Determine file paths ---
    mos_map = {
        "chi": os.path.join(city.data_dir, "mos", "combined_mos_kord.csv"),
        "phl": os.path.join(city.data_dir, "mos", "combined_mos_kphl.csv"),
    }
    station_map = {
        "chi": os.path.join(city.data_dir, "raw", "USW00094846.csv"),
        "phl": os.path.join(city.data_dir, "raw", "USW00013739.csv"),
    }
    mos_csv = mos_map[city_code]
    station_csv = station_map[city_code]

    # --- Load data ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        processed_dir, city_code
    )
    logger.info("Base features: %d columns", X_train.shape[1])

    # --- Enhanced features ---
    logger.info("Adding enhanced features...")
    X_train_enh = add_enhanced_features(X_train, y_train, y_train.index)
    X_val_enh = add_enhanced_features(X_val, y_val, y_val.index)
    X_test_enh = add_enhanced_features(X_test, y_test, y_test.index)

    scaler_enh = StandardScaler()
    cols_enh = X_train_enh.columns
    X_train_enh_s = pd.DataFrame(scaler_enh.fit_transform(X_train_enh), index=X_train_enh.index, columns=cols_enh)
    X_val_enh_s = pd.DataFrame(scaler_enh.transform(X_val_enh), index=X_val_enh.index, columns=cols_enh)
    X_test_enh_s = pd.DataFrame(scaler_enh.transform(X_test_enh), index=X_test_enh.index, columns=cols_enh)
    n_feat_enh = X_train_enh_s.shape[1]
    logger.info("Enhanced features: %d columns", n_feat_enh)

    test_actual = y_test.values
    val_actual = y_val.values
    test_dates = y_test.index

    all_results = {}
    model_gaussian_test_params = {}
    seasonal_all = {}
    model_preds = {}

    # ===================================================================
    # BENCHMARK 1: Real NWS MOS (PRIMARY BENCHMARK)
    # ===================================================================
    logger.info("-" * 50)
    logger.info("BENCHMARK: Real NWS MOS Ensemble (GFS+NAM)")
    mos_result = compute_real_mos_benchmark(
        city_code=city_code,
        target_station_csv=station_csv,
        mos_csv=mos_csv,
        test_dates=test_dates,
        test_actual=test_actual,
        bucket_edges=bucket_edges,
        train_end_date="2021-12-31",
    )
    all_results["Real_NWS_MOS"] = {
        "bucket_day_brier": mos_result["brier"]["overall_brier"],
        "per_bucket_brier": mos_result["brier"]["per_bucket_brier"],
    }
    model_gaussian_test_params["Real_NWS_MOS"] = (mos_result["mu"], mos_result["sigma"])
    seasonal_all["Real_NWS_MOS"] = mos_result["seasonal"]
    logger.info("Real NWS MOS: test Brier=%.4f (coverage=%.1f%%)",
                mos_result["brier"]["overall_brier"], mos_result["mos_coverage_pct"])

    # ===================================================================
    # BASELINE 1: Persistence
    # ===================================================================
    logger.info("-" * 50)
    logger.info("BASELINE: Persistence")
    p = run_persistence(y_train, y_val, y_test)
    probs_t = gaussian_to_bucket_probs(p["mu_test"], p["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["Persistence"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["Persistence"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_gaussian_test_params["Persistence"] = (p["mu_test"], p["sigma_test"])
    logger.info("Persistence: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # BASELINE 2: Climatology
    # ===================================================================
    logger.info("-" * 50)
    logger.info("BASELINE: Climatology")
    c = run_climatology(y_train, y_val, y_test)
    probs_t = gaussian_to_bucket_probs(c["mu_test"], c["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["Climatology"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["Climatology"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_gaussian_test_params["Climatology"] = (c["mu_test"], c["sigma_test"])
    logger.info("Climatology: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # MODEL 1: Ridge Regression
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 1: Ridge Regression")
    r = run_ridge(X_train, y_train, X_val, y_val, X_test, bucket_edges)
    probs_t = gaussian_to_bucket_probs(r["mu_test"], r["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results[f"Ridge(a={r['alpha']})"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all[f"Ridge(a={r['alpha']})"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_preds["ridge"] = (r["mu_test"], r["sigma_test"])
    model_gaussian_test_params[f"Ridge(a={r['alpha']})"] = (r["mu_test"], r["sigma_test"])
    logger.info("Ridge(a=%s): test Brier=%.4f", r["alpha"], brier_t["overall_brier"])

    # ===================================================================
    # MODEL 2: Heteroscedastic NN
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 2: Heteroscedastic NN")
    nn_b = train_hetero_nn(X_train, y_train, X_val, y_val, X_test)
    probs_t = gaussian_to_bucket_probs(nn_b["mu_test"], nn_b["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["HeteroscedasticNN"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["HeteroscedasticNN"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_preds["hetero_nn"] = (nn_b["mu_test"], nn_b["sigma_test"])
    model_gaussian_test_params["HeteroscedasticNN"] = (nn_b["mu_test"], nn_b["sigma_test"])
    logger.info("HeteroscedasticNN: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # MODEL 3: FeatureAttentionNet
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 3: FeatureAttentionNet")
    fa_model = FeatureAttentionNet(
        n_features=n_feat_enh, context_dim=64,
        hidden_sizes=[256, 128, 64], dropout=0.15,
    )
    fa_result = train_model(
        fa_model, X_train_enh_s.values, y_train.values,
        X_val_enh_s.values, y_val.values,
        model_type="standard", lr=0.001, max_epochs=300,
        patience=25, batch_size=64, loss_type="crps_mae",
    )
    fa_mu_test, fa_sigma_test = predict_model(fa_result["model"], X_test_enh_s.values, model_type="standard")
    fa_mu_val, fa_sigma_val = predict_model(fa_result["model"], X_val_enh_s.values, model_type="standard")

    probs_t = gaussian_to_bucket_probs(fa_mu_test, fa_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["FeatureAttentionNet"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["FeatureAttentionNet"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_preds["feat_attn"] = (fa_mu_test, fa_sigma_test)
    model_gaussian_test_params["FeatureAttentionNet"] = (fa_mu_test, fa_sigma_test)
    logger.info("FeatureAttentionNet: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # MODEL 4: MOSCorrectionNet
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 4: MOSCorrectionNet")
    all_y = np.concatenate([y_train.values, y_val.values, y_test.values])
    all_dates = y_train.index.append(y_val.index).append(y_test.index)
    baseline_all, _ = compute_mos_baseline(y_train.values, y_train.index, all_y, all_dates)
    n_tr, n_v = len(y_train), len(y_val)
    baseline_train = baseline_all[:n_tr]
    baseline_val = baseline_all[n_tr:n_tr + n_v]
    baseline_test = baseline_all[n_tr + n_v:]

    mos_model = MOSCorrectionNet(n_features=n_feat_enh, hidden_sizes=[128, 64, 32], dropout=0.15)
    mos_nn_result = train_model(
        mos_model, X_train_enh_s.values, y_train.values,
        X_val_enh_s.values, y_val.values,
        model_type="mos_correction",
        baseline_train=baseline_train, baseline_val=baseline_val,
        lr=0.001, max_epochs=300, patience=25, batch_size=64, loss_type="crps_mae",
    )
    mos_nn_mu_test, mos_nn_sigma_test = predict_model(
        mos_nn_result["model"], X_test_enh_s.values,
        model_type="mos_correction", baseline=baseline_test,
    )
    mos_nn_mu_val, mos_nn_sigma_val = predict_model(
        mos_nn_result["model"], X_val_enh_s.values,
        model_type="mos_correction", baseline=baseline_val,
    )

    probs_t = gaussian_to_bucket_probs(mos_nn_mu_test, mos_nn_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["MOSCorrectionNet"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["MOSCorrectionNet"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_preds["mos_correction"] = (mos_nn_mu_test, mos_nn_sigma_test)
    model_gaussian_test_params["MOSCorrectionNet"] = (mos_nn_mu_test, mos_nn_sigma_test)
    logger.info("MOSCorrectionNet: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # MODEL 5: RegimeConditionalNet
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 5: RegimeConditionalNet")
    regime_train = compute_regime_features(y_train.index, y_train.values)
    regime_val = compute_regime_features(y_val.index, y_val.values)
    regime_test = compute_regime_features(y_test.index, y_test.values)
    n_regime = regime_train.shape[1]

    rc_model = RegimeConditionalNet(
        n_features=n_feat_enh, n_regime_features=n_regime,
        hidden_sizes=[256, 128, 64], dropout=0.15,
    )
    rc_result = train_model(
        rc_model, X_train_enh_s.values, y_train.values,
        X_val_enh_s.values, y_val.values,
        model_type="regime_conditional",
        regime_train=regime_train, regime_val=regime_val,
        lr=0.001, max_epochs=300, patience=25, batch_size=64, loss_type="crps_mae",
    )
    rc_mu_test, rc_sigma_test = predict_model(
        rc_result["model"], X_test_enh_s.values,
        model_type="regime_conditional", regime=regime_test,
    )
    probs_t = gaussian_to_bucket_probs(rc_mu_test, rc_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    all_results["RegimeConditionalNet"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["RegimeConditionalNet"] = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    model_preds["regime_cond"] = (rc_mu_test, rc_sigma_test)
    model_gaussian_test_params["RegimeConditionalNet"] = (rc_mu_test, rc_sigma_test)
    logger.info("RegimeConditionalNet: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # MODEL 6: Calibrated Ensemble
    # ===================================================================
    logger.info("-" * 50)
    logger.info("MODEL 6: CalibratedEnsemble")
    val_preds = {
        "ridge": (r["mu_val"], r["sigma_val"]),
        "hetero_nn": (nn_b["mu_val"], nn_b["sigma_val"]),
        "feat_attn": (fa_mu_val, fa_sigma_val),
        "mos_correction": (mos_nn_mu_val, mos_nn_sigma_val),
    }

    # Use same 4 models for test stacking as val (exclude regime_cond itself)
    test_preds_for_stacker = {
        k: v for k, v in model_preds.items() if k in val_preds
    }

    stacker = EnsembleStacker()
    stacker.fit(val_preds, y_val.values, regime_val)
    ens_mu_test, ens_sigma_test = stacker.predict(test_preds_for_stacker, regime_test)
    ens_mu_val, ens_sigma_val = stacker.predict(val_preds, regime_val)

    calibrator = IsotonicPlattCalibrator()
    calibrator.fit(ens_mu_val, ens_sigma_val, val_actual, bucket_edges)
    _, _, cal_probs = calibrator.calibrate(ens_mu_test, ens_sigma_test, bucket_edges)

    brier_t = compute_brier_score(cal_probs, test_actual, bucket_edges)
    all_results["CalibratedEnsemble"] = {"bucket_day_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["CalibratedEnsemble"] = compute_seasonal_brier(cal_probs, test_actual, test_dates, bucket_edges)
    model_gaussian_test_params["CalibratedEnsemble"] = (ens_mu_test, ens_sigma_test)
    logger.info("CalibratedEnsemble: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # Canonical metric for cross-city comparability: contract-row Brier
    # ===================================================================
    kalshi_contract_rows = load_city_kalshi_contract_rows(
        city_code=city_code,
        valid_dates=test_dates,
        settled_paths=KALSHI_SETTLED_PATHS,
    )
    contract_outcomes = kalshi_contract_rows["actual_outcome"].astype(float).values
    for name, (mu, sigma) in model_gaussian_test_params.items():
        mu_by_date = pd.Series(mu, index=pd.to_datetime(test_dates).normalize())
        sigma_by_date = pd.Series(np.maximum(np.asarray(sigma, dtype=float), 0.5), index=pd.to_datetime(test_dates).normalize())
        probs = contract_probabilities_from_gaussian(kalshi_contract_rows, mu_by_date, sigma_by_date)
        all_results[name]["contract_brier"] = contract_brier_score(probs, contract_outcomes)
        all_results[name]["test_brier"] = all_results[name]["contract_brier"]

    market_probs = kalshi_contract_rows["market_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).values
    all_results["Kalshi_Settled_Market"] = {
        "contract_brier": contract_brier_score(market_probs, contract_outcomes),
        "test_brier": contract_brier_score(market_probs, contract_outcomes),
    }

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 70)
    logger.info("%s REAL-DATA BENCHMARK SUMMARY", city_name.upper())
    logger.info("=" * 70)

    summary_rows = []
    for name, res in all_results.items():
        row = {
            "model": name,
            "test_brier": res["test_brier"],
            "contract_brier": res.get("contract_brier"),
            "bucket_day_brier": res.get("bucket_day_brier"),
        }
        if name in seasonal_all:
            for s, v in seasonal_all[name].items():
                row[f"brier_{s}"] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("test_brier")
    logger.info("\n%s", summary_df.to_string(index=False))

    # --- Compute improvement vs MOS ---
    mos_brier = all_results["Real_NWS_MOS"]["test_brier"]
    logger.info("\n--- Model improvement vs Real NWS MOS (Brier=%.4f) ---", mos_brier)
    for _, row in summary_df.iterrows():
        name = row["model"]
        b = row["test_brier"]
        if name != "Real_NWS_MOS":
            delta = b - mos_brier
            pct = 100.0 * delta / mos_brier if mos_brier > 0 else 0
            direction = "WORSE" if delta > 0 else "BETTER"
            logger.info("  %s: %.4f (%+.4f, %+.1f%% %s)", name, b, delta, pct, direction)

    # --- Save artifacts ---
    summary_df.to_csv(os.path.join(results_dir, f"{city_code}_real_data_benchmark_summary.csv"), index=False)

    detail = {}
    for name, res in all_results.items():
        detail[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in res.items()
            if k != "per_bucket_brier"
        }
        if "per_bucket_brier" in res:
            detail[name]["per_bucket_brier"] = [float(x) for x in res["per_bucket_brier"]]

    with open(os.path.join(results_dir, f"{city_code}_real_data_benchmark_detail.json"), "w") as f:
        json.dump(detail, f, indent=2, default=str)

    plot_brier_comparison(
        all_results,
        os.path.join(results_dir, f"{city_code}_real_data_brier_comparison.png"),
        city_name,
    )

    metadata = {
        "city": city_name,
        "city_code": city_code,
        "kalshi_ticker": city.kalshi_ticker,
        "target_station": city.target_station,
        "train_period": "2000-01-01 to 2021-12-31",
        "val_period": "2022-01-01 to 2023-12-31",
        "test_period": "2024-01-01+",
        "n_base_features": X_train.shape[1],
        "n_enhanced_features": n_feat_enh,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_buckets": len(bucket_edges),
        "benchmark_mos_brier": float(mos_brier),
        "benchmark_unit": "binary contract-row",
        "kalshi_contract_rows": int(len(kalshi_contract_rows)),
        "kalshi_contract_dates": int(kalshi_contract_rows["date"].nunique()),
        "mos_coverage_pct": mos_result["mos_coverage_pct"],
        "mos_diagnostics": {
            "mae": mos_result["diagnostics"]["overall_mae"],
            "rmse": mos_result["diagnostics"]["overall_rmse"],
            "bias": mos_result["diagnostics"]["overall_bias"],
            "sigma": mos_result["diagnostics"]["overall_sigma"],
        },
        "best_model": summary_df.iloc[0]["model"],
        "best_test_brier": float(summary_df.iloc[0]["test_brier"]),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(results_dir, f"{city_code}_real_data_benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info("%s Real-Data Benchmark Complete", city_name)
    logger.info("Best model: %s (Brier: %.4f) | MOS benchmark: %.4f",
                metadata["best_model"], metadata["best_test_brier"], mos_brier)
    logger.info("Results saved to: %s", results_dir)
    logger.info("=" * 70)

    return all_results


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-data benchmark for CHI/PHL")
    parser.add_argument(
        "--city", type=str, default="both",
        choices=["chi", "phl", "both"],
        help="City to benchmark (default: both)",
    )
    args = parser.parse_args()

    all_city_results = {}

    if args.city in ("chi", "both"):
        all_city_results["chi"] = run_city_benchmark("chi")

    if args.city in ("phl", "both"):
        all_city_results["phl"] = run_city_benchmark("phl")

    # --- Cross-city comparison ---
    if len(all_city_results) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-CITY COMPARISON")
        logger.info("=" * 70)
        for city_code, results in all_city_results.items():
            city_name = {"chi": "Chicago", "phl": "Philadelphia"}[city_code]
            sorted_models = sorted(results.items(), key=lambda x: x[1]["test_brier"])
            logger.info("\n%s:", city_name)
            for name, res in sorted_models:
                logger.info("  %-25s Brier=%.4f", name, res["test_brier"])

    return all_city_results


if __name__ == "__main__":
    main()
