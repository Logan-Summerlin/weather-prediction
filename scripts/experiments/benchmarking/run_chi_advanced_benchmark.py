#!/usr/bin/env python3
"""
Chicago Advanced Model Benchmark Script.

Trains and evaluates advanced neural network models for Chicago (KXHIGHCHI)
temperature prediction, applying best practices from NYC's top model families
(U7_regime_conditional, E17_contract_brier, E40_lag2, WindGatedAttention).

Models benchmarked (in addition to baselines):
  1. Persistence baseline
  2. Climatology baseline
  3. Ridge regression (best alpha on val Brier)
  4. Heteroscedastic NN (existing baseline)
  5. FeatureAttentionNet  -- dynamic feature importance via learned attention
  6. MOSCorrectionNet     -- NN that corrects MOS/climatology forecast errors
  7. RegimeConditionalNet -- regime-aware heteroscedastic (season x volatility)
  8. Calibrated Ensemble  -- stacked meta-learner with isotonic + Platt cal

All models use:
  - Training data:   2000-01-01 to 2021-12-31
  - Validation data:  2022-01-01 to 2023-12-31
  - Test data:        2024-01-01 to 2025-12-31
  - Actual GHCN station data (no synthetic data)
  - Up to 55 surrounding weather stations within 200mi of O'Hare

Benchmarked against:
  - NWS proxy (enhanced climatology approximating NWS skill)
  - Kalshi pre-settlement proxy (Ridge-based market expectation)

Results are saved to results/chicago/.

Usage:
    python scripts/run_chi_advanced_benchmark.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

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
    compute_nws_proxy_baseline,
    compute_kalshi_presettlement_proxy,
    train_model,
    predict_model,
)
from src.city_config import get_city_runtime_config

city_config = get_city_runtime_config("chi")

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


# ===========================================================================
# Data Loading with Date-Based Splits
# ===========================================================================

def load_and_split_chi_data(
    processed_dir: str,
    raw_dir: str | None = None,
) -> tuple:
    """Load Chicago data and split chronologically by year.

    Uses fixed date boundaries:
      - Train: 2000-01-01 to 2021-12-31
      - Val:   2022-01-01 to 2023-12-31
      - Test:  2024-01-01 to 2025-12-31

    First tries to load preprocessed data. If dates don't match the
    requested splits, re-splits from the full dataset.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Try loading preprocessed data first
    if os.path.isdir(processed_dir):
        try:
            X_full_parts = []
            y_full_parts = []
            for split in ["train", "val", "test"]:
                feat_path = os.path.join(processed_dir, f"features_{split}.csv")
                targ_path = os.path.join(processed_dir, f"target_{split}.csv")
                if os.path.exists(feat_path) and os.path.exists(targ_path):
                    X_part = pd.read_csv(feat_path, index_col=0, parse_dates=True)
                    y_part = pd.read_csv(targ_path, index_col=0, parse_dates=True).iloc[:, 0]
                    X_full_parts.append(X_part)
                    y_full_parts.append(y_part)

            if X_full_parts:
                X_full = pd.concat(X_full_parts, axis=0).sort_index()
                y_full = pd.concat(y_full_parts, axis=0).sort_index()
                y_full.name = "CHI_TMAX"

                # Re-split by date
                train_mask = X_full.index < "2022-01-01"
                val_mask = (X_full.index >= "2022-01-01") & (X_full.index < "2024-01-01")
                test_mask = X_full.index >= "2024-01-01"

                X_train = X_full[train_mask]
                X_val = X_full[val_mask]
                X_test = X_full[test_mask]
                y_train = y_full[train_mask]
                y_val = y_full[val_mask]
                y_test = y_full[test_mask]

                # Filter training to start from 2000
                train_start = X_train.index >= "2000-01-01"
                X_train = X_train[train_start]
                y_train = y_train[train_start]

                logger.info(
                    "Loaded and re-split CHI data: train=%d (%s to %s), "
                    "val=%d (%s to %s), test=%d (%s to %s)",
                    len(X_train), X_train.index.min().date(), X_train.index.max().date(),
                    len(X_val), X_val.index.min().date(), X_val.index.max().date(),
                    len(X_test), X_test.index.min().date(), X_test.index.max().date(),
                )

                # Re-scale: fit scaler on training data only
                scaler = StandardScaler()
                feature_cols = X_train.columns
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    index=X_train.index, columns=feature_cols,
                )
                X_val_scaled = pd.DataFrame(
                    scaler.transform(X_val),
                    index=X_val.index, columns=feature_cols,
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    index=X_test.index, columns=feature_cols,
                )

                return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

        except Exception as e:
            logger.warning("Could not load preprocessed data: %s", e)

    raise FileNotFoundError(
        f"Processed data not found at {processed_dir}. "
        "Run scripts/run_chi_preprocessing.py first."
    )


# ===========================================================================
# Baseline Models (same as existing benchmark)
# ===========================================================================

def gaussian_nll_loss(mu, sigma, target):
    """Gaussian negative log-likelihood."""
    var = sigma ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + ((target - mu) ** 2) / var)
    return nll.mean()


class HeteroscedasticNet(nn.Module):
    """Baseline heteroscedastic NN (for comparison)."""

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


def run_persistence_baseline(y_train, y_val, y_test):
    all_y = pd.concat([y_train, y_val, y_test])
    y_prev = all_y.shift(1)
    train_resid = all_y[all_y.index.isin(y_train.index)] - y_prev[y_prev.index.isin(y_train.index)]
    sigma = max(float(train_resid.dropna().std()), 2.0)
    train_mean = float(y_train.mean())

    def _extract(mask_idx):
        mu = y_prev[y_prev.index.isin(mask_idx)].values
        return np.where(np.isnan(mu), train_mean, mu), np.full(len(mu), sigma)

    mu_v, sig_v = _extract(y_val.index)
    mu_t, sig_t = _extract(y_test.index)
    return {"mu_val": mu_v, "sigma_val": sig_v, "mu_test": mu_t, "sigma_test": sig_t}


def run_climatology_baseline(y_train, y_val, y_test):
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


def run_ridge_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    best = {"brier": float("inf")}
    bucket_edges = get_city_config("chi").bucket_edges
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
                "mu_val": mu_v, "mu_test": mu_t,
                "sigma": sig, "model": model,
            }
    return {
        "mu_val": best["mu_val"],
        "sigma_val": np.full(len(best["mu_val"]), best["sigma"]),
        "mu_test": best["mu_test"],
        "sigma_test": np.full(len(best["mu_test"]), best["sigma"]),
        "alpha": best["alpha"],
    }


def train_baseline_nn(X_train, y_train, X_val, y_val, X_test, y_test):
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

def plot_brier_comparison(results, save_path, title="Chicago"):
    models = list(results.keys())
    briers = [results[m]["test_brier"] for m in models]
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(models)))
    bars = ax.bar(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, briers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title(f"{title} Advanced Benchmark: Test-Set Brier Scores")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(attn_weights, feature_names, save_path, top_n=30):
    """Plot top-N feature importance from attention weights."""
    mean_attn = attn_weights.mean(axis=0)
    top_idx = np.argsort(mean_attn)[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_idx)), mean_attn[top_idx][::-1])
    ax.set_yticks(range(len(top_idx)))
    labels = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_idx[::-1]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title(f"Top-{top_n} Feature Importance (FeatureAttentionNet)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main Benchmark
# ===========================================================================

def main():
    logger.info("=" * 70)
    logger.info("Chicago ADVANCED Model Benchmark (KXHIGHCHI)")
    logger.info("Train: 2000-2021 | Val: 2022-2023 | Test: 2024-2025")
    logger.info("=" * 70)

    chi = get_city_config("chi")
    ensure_city_dirs(chi)

    processed_dir = os.path.join(chi.data_dir, "processed")
    results_dir = chi.results_dir
    os.makedirs(results_dir, exist_ok=True)
    bucket_edges = chi.bucket_edges
    bucket_labels = chi.bucket_labels

    # --- Load data ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_chi_data(
        processed_dir
    )
    logger.info("Features: %d columns", X_train.shape[1])

    # --- Add enhanced features ---
    logger.info("Adding enhanced features (rolling stats, consensus, lag features)...")
    X_train_enh = add_enhanced_features(X_train, y_train, y_train.index)
    X_val_enh = add_enhanced_features(X_val, y_val, y_val.index)
    X_test_enh = add_enhanced_features(X_test, y_test, y_test.index)

    # Re-scale enhanced features (fit on train only)
    scaler_enh = StandardScaler()
    feature_cols_enh = X_train_enh.columns
    X_train_enh_scaled = pd.DataFrame(
        scaler_enh.fit_transform(X_train_enh),
        index=X_train_enh.index, columns=feature_cols_enh,
    )
    X_val_enh_scaled = pd.DataFrame(
        scaler_enh.transform(X_val_enh),
        index=X_val_enh.index, columns=feature_cols_enh,
    )
    X_test_enh_scaled = pd.DataFrame(
        scaler_enh.transform(X_test_enh),
        index=X_test_enh.index, columns=feature_cols_enh,
    )
    logger.info("Enhanced features: %d columns", X_train_enh_scaled.shape[1])

    test_actual = y_test.values
    val_actual = y_val.values
    test_dates = y_test.index

    all_results = {}
    seasonal_all = {}
    model_predictions = {}  # for ensemble stacking

    # ===================================================================
    # 1. Persistence Baseline
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 1: Persistence Baseline")
    p = run_persistence_baseline(y_train, y_val, y_test)
    probs_t = gaussian_to_bucket_probs(p["mu_test"], p["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["Persistence"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["Persistence"] = seas_t
    logger.info("Persistence: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # 2. Climatology Baseline
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 2: Climatology Baseline")
    c = run_climatology_baseline(y_train, y_val, y_test)
    probs_t = gaussian_to_bucket_probs(c["mu_test"], c["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["Climatology"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["Climatology"] = seas_t
    logger.info("Climatology: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # 3. Ridge Regression
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 3: Ridge Regression")
    r = run_ridge_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
    probs_t = gaussian_to_bucket_probs(r["mu_test"], r["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results[f"Ridge(a={r['alpha']})"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all[f"Ridge(a={r['alpha']})"] = seas_t
    model_predictions["ridge"] = (r["mu_test"], r["sigma_test"])
    logger.info("Ridge(a=%s): test Brier=%.4f", r["alpha"], brier_t["overall_brier"])

    # ===================================================================
    # 4. Baseline Heteroscedastic NN
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 4: Heteroscedastic NN (baseline)")
    nn_b = train_baseline_nn(X_train, y_train, X_val, y_val, X_test, y_test)
    probs_t = gaussian_to_bucket_probs(nn_b["mu_test"], nn_b["sigma_test"], bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["HeteroscedasticNN"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["HeteroscedasticNN"] = seas_t
    model_predictions["hetero_nn"] = (nn_b["mu_test"], nn_b["sigma_test"])
    logger.info("HeteroscedasticNN: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # 5. Feature Attention Network (KEY: dynamic feature importance)
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 5: FeatureAttentionNet (dynamic feature importance)")

    n_feat_enh = X_train_enh_scaled.shape[1]
    fa_model = FeatureAttentionNet(
        n_features=n_feat_enh,
        context_dim=64,
        hidden_sizes=[256, 128, 64],
        dropout=0.15,
    )

    fa_result = train_model(
        fa_model,
        X_train_enh_scaled.values, y_train.values,
        X_val_enh_scaled.values, y_val.values,
        model_type="standard",
        lr=0.001,
        max_epochs=300,
        patience=25,
        batch_size=64,
        loss_type="crps_mae",
    )

    fa_mu_test, fa_sigma_test = predict_model(
        fa_result["model"], X_test_enh_scaled.values, model_type="standard"
    )
    fa_mu_val, fa_sigma_val = predict_model(
        fa_result["model"], X_val_enh_scaled.values, model_type="standard"
    )

    probs_t = gaussian_to_bucket_probs(fa_mu_test, fa_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["FeatureAttentionNet"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["FeatureAttentionNet"] = seas_t
    model_predictions["feat_attn"] = (fa_mu_test, fa_sigma_test)
    logger.info("FeatureAttentionNet: test Brier=%.4f", brier_t["overall_brier"])

    # Extract attention weights for interpretability
    fa_result["model"].eval()
    with torch.no_grad():
        out = fa_result["model"](
            torch.tensor(X_test_enh_scaled.values, dtype=torch.float32).to(DEVICE)
        )
        attn_weights = out["attention_weights"].cpu().numpy()
    plot_feature_importance(
        attn_weights, list(X_test_enh_scaled.columns),
        os.path.join(results_dir, "chi_feature_importance.png"),
    )

    # ===================================================================
    # 6. MOS Correction Network
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 6: MOSCorrectionNet (error correction)")

    # Compute MOS baseline
    all_y = np.concatenate([y_train.values, y_val.values, y_test.values])
    all_dates = y_train.index.append(y_val.index).append(y_test.index)
    baseline_all, baseline_sigma = compute_mos_baseline(
        y_train.values, y_train.index, all_y, all_dates
    )
    n_tr = len(y_train)
    n_v = len(y_val)
    baseline_train = baseline_all[:n_tr]
    baseline_val = baseline_all[n_tr:n_tr+n_v]
    baseline_test = baseline_all[n_tr+n_v:]

    # Train NN on residuals
    mos_target_train = y_train.values  # still predict full TMAX, but model adds delta to baseline
    mos_target_val = y_val.values

    mos_model = MOSCorrectionNet(
        n_features=n_feat_enh,
        hidden_sizes=[128, 64, 32],
        dropout=0.15,
    )

    mos_result = train_model(
        mos_model,
        X_train_enh_scaled.values, mos_target_train,
        X_val_enh_scaled.values, mos_target_val,
        model_type="mos_correction",
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        lr=0.001,
        max_epochs=300,
        patience=25,
        batch_size=64,
        loss_type="crps_mae",
    )

    mos_mu_test, mos_sigma_test = predict_model(
        mos_result["model"], X_test_enh_scaled.values,
        model_type="mos_correction", baseline=baseline_test,
    )
    mos_mu_val, mos_sigma_val = predict_model(
        mos_result["model"], X_val_enh_scaled.values,
        model_type="mos_correction", baseline=baseline_val,
    )

    probs_t = gaussian_to_bucket_probs(mos_mu_test, mos_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["MOSCorrectionNet"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["MOSCorrectionNet"] = seas_t
    model_predictions["mos_correction"] = (mos_mu_test, mos_sigma_test)
    logger.info("MOSCorrectionNet: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # 7. Regime-Conditional Network
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 7: RegimeConditionalNet (season x volatility)")

    regime_train = compute_regime_features(y_train.index, y_train.values)
    regime_val = compute_regime_features(y_val.index, y_val.values)
    regime_test = compute_regime_features(y_test.index, y_test.values)

    n_regime = regime_train.shape[1]
    rc_model = RegimeConditionalNet(
        n_features=n_feat_enh,
        n_regime_features=n_regime,
        hidden_sizes=[256, 128, 64],
        dropout=0.15,
    )

    rc_result = train_model(
        rc_model,
        X_train_enh_scaled.values, y_train.values,
        X_val_enh_scaled.values, y_val.values,
        model_type="regime_conditional",
        regime_train=regime_train,
        regime_val=regime_val,
        lr=0.001,
        max_epochs=300,
        patience=25,
        batch_size=64,
        loss_type="crps_mae",
    )

    rc_mu_test, rc_sigma_test = predict_model(
        rc_result["model"], X_test_enh_scaled.values,
        model_type="regime_conditional", regime=regime_test,
    )

    probs_t = gaussian_to_bucket_probs(rc_mu_test, rc_sigma_test, bucket_edges)
    brier_t = compute_brier_score(probs_t, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(probs_t, test_actual, test_dates, bucket_edges)
    all_results["RegimeConditionalNet"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["RegimeConditionalNet"] = seas_t
    model_predictions["regime_cond"] = (rc_mu_test, rc_sigma_test)
    logger.info("RegimeConditionalNet: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # 8. Calibrated Ensemble (stacking + isotonic + Platt)
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 8: Calibrated Ensemble (stack + isotonic + Platt)")

    # Collect val predictions for stacking
    val_preds = {
        "ridge": (r["mu_val"], r["sigma_val"]),
        "hetero_nn": (nn_b["mu_val"], nn_b["sigma_val"]),
        "feat_attn": (fa_mu_val, fa_sigma_val),
        "mos_correction": (mos_mu_val, mos_sigma_val),
    }

    # Fit ensemble stacker on validation set
    stacker = EnsembleStacker()
    stacker.fit(val_preds, y_val.values, regime_val)

    # Predict on test
    ens_mu_test, ens_sigma_test = stacker.predict(model_predictions, regime_test)

    # Fit calibrator on validation predictions
    calibrator = IsotonicPlattCalibrator()
    calibrator.fit(ens_mu_test, ens_sigma_test, test_actual, bucket_edges)

    # Apply calibration (use val to fit, apply to test)
    # Actually fit on val first:
    ens_mu_val, ens_sigma_val = stacker.predict(val_preds, regime_val)
    calibrator = IsotonicPlattCalibrator()
    calibrator.fit(ens_mu_val, ens_sigma_val, val_actual, bucket_edges)

    _, _, cal_probs = calibrator.calibrate(ens_mu_test, ens_sigma_test, bucket_edges)
    brier_t = compute_brier_score(cal_probs, test_actual, bucket_edges)
    seas_t = compute_seasonal_brier(cal_probs, test_actual, test_dates, bucket_edges)
    all_results["CalibratedEnsemble"] = {"test_brier": brier_t["overall_brier"], "per_bucket_brier": brier_t["per_bucket_brier"]}
    seasonal_all["CalibratedEnsemble"] = seas_t
    logger.info("CalibratedEnsemble: test Brier=%.4f", brier_t["overall_brier"])

    # ===================================================================
    # NWS Proxy Benchmark
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Benchmark: NWS Proxy")
    nws_mu, nws_sigma = compute_nws_proxy_baseline(
        y_train.values, y_train.index, test_actual, test_dates
    )
    probs_nws = gaussian_to_bucket_probs(nws_mu, nws_sigma, bucket_edges)
    brier_nws = compute_brier_score(probs_nws, test_actual, bucket_edges)
    all_results["NWS_Proxy"] = {"test_brier": brier_nws["overall_brier"], "per_bucket_brier": brier_nws["per_bucket_brier"]}
    seasonal_all["NWS_Proxy"] = compute_seasonal_brier(probs_nws, test_actual, test_dates, bucket_edges)
    logger.info("NWS Proxy: test Brier=%.4f", brier_nws["overall_brier"])

    # ===================================================================
    # Kalshi Pre-Settlement Proxy Benchmark
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Benchmark: Kalshi PreSettlement Proxy")
    kal_mu, kal_sigma = compute_kalshi_presettlement_proxy(
        y_train.values, y_train.index, test_actual, test_dates,
        X_train.values, X_test.values,
    )
    probs_kal = gaussian_to_bucket_probs(kal_mu, kal_sigma, bucket_edges)
    brier_kal = compute_brier_score(probs_kal, test_actual, bucket_edges)
    all_results["Kalshi_PreSettlement"] = {"test_brier": brier_kal["overall_brier"], "per_bucket_brier": brier_kal["per_bucket_brier"]}
    seasonal_all["Kalshi_PreSettlement"] = compute_seasonal_brier(probs_kal, test_actual, test_dates, bucket_edges)
    logger.info("Kalshi PreSettlement: test Brier=%.4f", brier_kal["overall_brier"])

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 70)
    logger.info("ADVANCED BENCHMARK SUMMARY")
    logger.info("=" * 70)

    summary_rows = []
    for name, res in all_results.items():
        row = {"model": name, "test_brier": res["test_brier"]}
        if name in seasonal_all:
            for s, v in seasonal_all[name].items():
                row[f"brier_{s}"] = v
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("test_brier")
    logger.info("\n%s", summary_df.to_string(index=False))

    # --- Save artifacts ---
    summary_df.to_csv(os.path.join(results_dir, "chi_advanced_benchmark_summary.csv"), index=False)

    detail = {}
    for name, res in all_results.items():
        detail[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in res.items()
            if k != "per_bucket_brier"
        }
        if "per_bucket_brier" in res:
            detail[name]["per_bucket_brier"] = [float(x) for x in res["per_bucket_brier"]]

    with open(os.path.join(results_dir, "chi_advanced_benchmark_detail.json"), "w") as f:
        json.dump(detail, f, indent=2, default=str)

    # Plots
    plot_brier_comparison(
        all_results,
        os.path.join(results_dir, "chi_advanced_brier_comparison.png"),
        title="Chicago",
    )

    metadata = {
        "city": "Chicago",
        "kalshi_ticker": "KXHIGHCHI",
        "target_station": city_config.TARGET_STATION,
        "n_surrounding_stations": len(city_config.SURROUNDING_STATIONS),
        "train_period": "2000-01-01 to 2021-12-31",
        "val_period": "2022-01-01 to 2023-12-31",
        "test_period": "2024-01-01 to 2025-12-31",
        "n_base_features": X_train.shape[1],
        "n_enhanced_features": n_feat_enh,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_buckets": len(bucket_edges),
        "best_model": summary_df.iloc[0]["model"],
        "best_test_brier": float(summary_df.iloc[0]["test_brier"]),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(results_dir, "chi_advanced_benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info("Chicago Advanced Benchmark Complete")
    logger.info("Best model: %s (test Brier: %.4f)",
                metadata["best_model"], metadata["best_test_brier"])
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
