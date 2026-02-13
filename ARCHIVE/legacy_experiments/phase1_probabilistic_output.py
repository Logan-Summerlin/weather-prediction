#!/usr/bin/env python3
"""
Phase 1D: Probabilistic Output Conversion for NYC Temperature Prediction.

Converts the point-prediction MOS correction model (C_Correction_NN_tiny)
into a heteroscedastic Gaussian model that outputs both a predicted mean
(mu) and predicted uncertainty (sigma) for each day's temperature forecast.

Models trained:
  Variant A: NLL-only training [32,16]
  Variant B: NLL -> CRPS fine-tune [32,16]
  Variant C: Combined CRPS+MAE training [32,16]
  Variant D: NLL -> CRPS fine-tune [64,32] (larger)
  Variant E: NLL -> CRPS fine-tune [32,16] with dropout=0.1
  Baseline:  Point-prediction C_Correction_NN_tiny replica

All data is REAL -- downloaded from NOAA GHCN and loaded from existing MOS CSVs.
"""

import os
import sys
import json
import time
import copy
import math
import logging
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from mos_ensemble_pipeline import (
    download_all_stations,
    build_station_matrix,
    load_mos_data,
    load_central_park_tmax,
    DatasetBuilder,
    evaluate_model,
    FlexibleNN,
    train_nn,
    predict_nn,
    assign_season,
    DEVICE,
    SEED,
    RAW_DIR,
    DLY_START,
    DLY_END,
)
from src.crps_loss import GaussianCRPSLoss, CombinedCRPSMAELoss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger("phase1_probabilistic")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1_probabilistic")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Sigma constraints
SIGMA_FLOOR = 0.75   # degrees F
SIGMA_CAP = 10.0     # degrees F


# ============================================================================
# 1. PROBABILISTIC MODEL DEFINITION
# ============================================================================

class ProbabilisticCorrectionNN(nn.Module):
    """Heteroscedastic Gaussian correction model.

    Same backbone architecture as C_Correction_NN_tiny, but with two
    output heads from the last hidden layer:
      - mu_head:        Linear(last_hidden, 1) -- predicted residual mean
      - log_sigma_head: Linear(last_hidden, 1) -- predicted residual log-std

    sigma = softplus(log_sigma) + SIGMA_FLOOR, capped at SIGMA_CAP.
    Final prediction = MOS_ensemble + mu.
    """

    def __init__(self, n_features, hidden_sizes, dropout=0.2, use_batchnorm=True):
        super().__init__()
        self.backbone = nn.ModuleList()

        in_dim = n_features
        for h in hidden_sizes:
            block = nn.ModuleList()
            block.append(nn.Linear(in_dim, h))
            if use_batchnorm and h >= 16:
                block.append(nn.BatchNorm1d(h))
            block.append(nn.ReLU())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.backbone.append(block)
            in_dim = h

        # Two output heads from last hidden dimension
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

        # Initialize log_sigma_head bias so initial sigma ~ 2.0 F
        # softplus(x) + 0.75 = 2.0 => softplus(x) = 1.25 => x ~ 1.25 for large x
        nn.init.constant_(self.log_sigma_head.bias, 0.5)
        nn.init.zeros_(self.log_sigma_head.weight)

    def forward(self, x):
        """Forward pass returning (mu, sigma).

        Returns
        -------
        mu : Tensor of shape (batch, 1)
        sigma : Tensor of shape (batch, 1), strictly positive
        """
        h = x
        for block in self.backbone:
            for layer in block:
                h = layer(h)

        mu = self.mu_head(h)
        raw_log_sigma = self.log_sigma_head(h)

        # sigma = softplus(raw) + floor, capped
        sigma = F.softplus(raw_log_sigma) + SIGMA_FLOOR
        sigma = torch.clamp(sigma, max=SIGMA_CAP)

        return mu, sigma


# ============================================================================
# 2. LOSS FUNCTIONS
# ============================================================================

class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss.

    NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, target):
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1).clamp(min=1e-6)
        target = target.reshape(-1)

        nll = 0.5 * torch.log(2 * math.pi * sigma ** 2) + \
              (target - mu) ** 2 / (2 * sigma ** 2)
        return nll.mean()


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20, batch_size=128):
    """Stage 1: Train with Gaussian NLL loss."""
    model = model.to(DEVICE)
    criterion = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_nll = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = criterion(mu, sigma, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(X_val_t)
            val_nll = criterion(val_mu, val_sigma, y_val_t).item()

        scheduler.step(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  NLL training stopped at epoch %d (best NLL=%.4f)",
                            epoch + 1, best_val_nll)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    logger.info("  NLL training complete. Best val NLL=%.4f", best_val_nll)
    return best_val_nll


def train_probabilistic_crps(model, X_train, y_train, X_val, y_val,
                             lr=5e-4, epochs=50, patience=10, batch_size=128):
    """Stage 2: Fine-tune with Gaussian CRPS loss."""
    model = model.to(DEVICE)
    criterion = GaussianCRPSLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_crps = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = criterion(mu.squeeze(-1), sigma.squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(X_val_t)
            val_crps = criterion(val_mu.squeeze(-1), val_sigma.squeeze(-1), y_val_t).item()

        scheduler.step(val_crps)

        if val_crps < best_val_crps:
            best_val_crps = val_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  CRPS fine-tune stopped at epoch %d (best CRPS=%.4f)",
                            epoch + 1, best_val_crps)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    logger.info("  CRPS fine-tune complete. Best val CRPS=%.4f", best_val_crps)
    return best_val_crps


def train_probabilistic_combined(model, X_train, y_train, X_val, y_val,
                                 lr=1e-3, epochs=200, patience=20, batch_size=128,
                                 crps_weight=0.7, mae_weight=0.3):
    """Train with combined CRPS+MAE loss from scratch."""
    model = model.to(DEVICE)
    criterion = CombinedCRPSMAELoss(crps_weight=crps_weight, mae_weight=mae_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss_dict = criterion(mu.squeeze(-1), sigma.squeeze(-1), yb)
            loss = loss_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(X_val_t)
            val_loss_dict = criterion(val_mu.squeeze(-1), val_sigma.squeeze(-1), y_val_t)
            val_loss = val_loss_dict["loss"].item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Combined training stopped at epoch %d (best loss=%.4f)",
                            epoch + 1, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    logger.info("  Combined training complete. Best val loss=%.4f", best_val_loss)
    return best_val_loss


# ============================================================================
# 4. PREDICTION AND EVALUATION
# ============================================================================

def predict_probabilistic(model, X):
    """Get mu and sigma predictions from a probabilistic model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        mu, sigma = model(X_t)
        mu_np = mu.squeeze(-1).cpu().numpy()
        sigma_np = sigma.squeeze(-1).cpu().numpy()
    return mu_np, sigma_np


def compute_crps(y_true, mu, sigma):
    """Compute Gaussian CRPS using closed-form formula (numpy)."""
    z = (y_true - mu) / np.maximum(sigma, 1e-6)
    phi_z = scipy_norm.pdf(z)
    Phi_z = scipy_norm.cdf(z)
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / math.sqrt(math.pi))
    return float(np.mean(crps))


def compute_coverage(y_true, mu, sigma, level):
    """Compute prediction interval coverage at a given confidence level.

    Parameters
    ----------
    level : float
        Confidence level, e.g. 0.50, 0.80, 0.90, 0.95
    """
    alpha = 1 - level
    z_low = scipy_norm.ppf(alpha / 2)
    z_high = scipy_norm.ppf(1 - alpha / 2)
    lower = mu + z_low * sigma
    upper = mu + z_high * sigma
    inside = np.logical_and(y_true >= lower, y_true <= upper)
    coverage = float(np.mean(inside))
    mean_width = float(np.mean(upper - lower))
    return coverage, mean_width


def compute_pit(y_true, mu, sigma):
    """Compute Probability Integral Transform values.

    PIT_i = Phi((y_i - mu_i) / sigma_i)

    A well-calibrated model produces uniform PIT values.
    """
    z = (y_true - mu) / np.maximum(sigma, 1e-6)
    pit = scipy_norm.cdf(z)
    return pit


def evaluate_probabilistic(y_true, mu, sigma, mos_base, dates, label=""):
    """Full probabilistic evaluation of a model on one split.

    Returns a dict with all metrics.
    """
    # Point prediction: MOS + mu_residual
    y_pred = mos_base + mu

    # Point MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # CRPS (computed on residuals: target_resid vs predicted_resid mu, sigma)
    y_resid = y_true - mos_base
    crps = compute_crps(y_resid, mu, sigma)

    # Coverage at multiple levels
    coverages = {}
    widths = {}
    for level in [0.50, 0.80, 0.90, 0.95]:
        cov, width = compute_coverage(y_resid, mu, sigma, level)
        coverages[f"cov_{int(level*100)}"] = round(cov, 4)
        widths[f"width_{int(level*100)}"] = round(width, 2)

    # Mean sigma
    mean_sigma = float(np.mean(sigma))

    # Seasonal breakdown of MAE and CRPS
    seasonal = {}
    if dates is not None:
        seasons = assign_season(pd.DatetimeIndex(dates))
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = (seasons == s).values if hasattr(seasons, 'values') else (seasons == s)
            if np.sum(mask) > 0:
                s_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
                s_crps = compute_crps(y_resid[mask], mu[mask], sigma[mask])
                seasonal[f"mae_{s}"] = round(s_mae, 4)
                seasonal[f"crps_{s}"] = round(s_crps, 4)

    result = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "crps": round(crps, 4),
        "mean_sigma": round(mean_sigma, 4),
        **coverages,
        **widths,
        **seasonal,
    }

    if label:
        logger.info("  %s: MAE=%.3f  CRPS=%.3f  sigma=%.2f  cov50=%.1f%%  cov90=%.1f%%  cov95=%.1f%%",
                     label, mae, crps, mean_sigma,
                     coverages.get("cov_50", 0) * 100,
                     coverages.get("cov_90", 0) * 100,
                     coverages.get("cov_95", 0) * 100)

    return result


# ============================================================================
# 5. VARIANT TRAINING
# ============================================================================

def train_variant_a(n_features, X_train, y_train, X_val, y_val):
    """Variant A: NLL-only training [32, 16]."""
    logger.info("=" * 70)
    logger.info("VARIANT A: NLL-only [32, 16]")
    logger.info("=" * 70)

    model = ProbabilisticCorrectionNN(n_features, [32, 16], dropout=0.2)
    train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20)
    return model


def train_variant_b(n_features, X_train, y_train, X_val, y_val):
    """Variant B: NLL -> CRPS fine-tune [32, 16]."""
    logger.info("=" * 70)
    logger.info("VARIANT B: NLL -> CRPS fine-tune [32, 16]")
    logger.info("=" * 70)

    model = ProbabilisticCorrectionNN(n_features, [32, 16], dropout=0.2)
    # Stage 1: NLL
    train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20)
    # Stage 2: CRPS fine-tune
    train_probabilistic_crps(model, X_train, y_train, X_val, y_val,
                             lr=5e-4, epochs=50, patience=10)
    return model


def train_variant_c(n_features, X_train, y_train, X_val, y_val):
    """Variant C: Combined CRPS+MAE training [32, 16]."""
    logger.info("=" * 70)
    logger.info("VARIANT C: Combined CRPS+MAE [32, 16]")
    logger.info("=" * 70)

    model = ProbabilisticCorrectionNN(n_features, [32, 16], dropout=0.2)
    train_probabilistic_combined(model, X_train, y_train, X_val, y_val,
                                 lr=1e-3, epochs=200, patience=20,
                                 crps_weight=0.7, mae_weight=0.3)
    return model


def train_variant_d(n_features, X_train, y_train, X_val, y_val):
    """Variant D: NLL -> CRPS fine-tune [64, 32] (larger)."""
    logger.info("=" * 70)
    logger.info("VARIANT D: NLL -> CRPS fine-tune [64, 32]")
    logger.info("=" * 70)

    model = ProbabilisticCorrectionNN(n_features, [64, 32], dropout=0.2)
    train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20)
    train_probabilistic_crps(model, X_train, y_train, X_val, y_val,
                             lr=5e-4, epochs=50, patience=10)
    return model


def train_variant_e(n_features, X_train, y_train, X_val, y_val):
    """Variant E: NLL -> CRPS fine-tune [32, 16] with dropout=0.1."""
    logger.info("=" * 70)
    logger.info("VARIANT E: NLL -> CRPS fine-tune [32, 16] dropout=0.1")
    logger.info("=" * 70)

    model = ProbabilisticCorrectionNN(n_features, [32, 16], dropout=0.1)
    train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20)
    train_probabilistic_crps(model, X_train, y_train, X_val, y_val,
                             lr=5e-4, epochs=50, patience=10)
    return model


def train_baseline(n_features, X_train, y_train, X_val, y_val):
    """Baseline: Point-prediction C_Correction_NN_tiny replica."""
    logger.info("=" * 70)
    logger.info("BASELINE: C_Correction_NN_tiny (point prediction)")
    logger.info("=" * 70)

    model = FlexibleNN(n_features, [32, 16], dropout=0.2)
    train_nn(model, X_train, y_train, X_val, y_val,
             lr=1e-3, epochs=200, patience=15, loss_fn_name="mae")
    return model


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()

    # ----------------------------------------------------------------
    # Step 1: Download and prepare data
    # ----------------------------------------------------------------
    logger.info("STEP 1: Downloading station data ...")
    download_all_stations()

    logger.info("STEP 2: Building station matrix ...")
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    logger.info("STEP 3: Loading MOS and Central Park data ...")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # ----------------------------------------------------------------
    # Step 2: Build MOS correction dataset (same as C_Correction_NN_tiny)
    # ----------------------------------------------------------------
    logger.info("STEP 4: Building MOS correction dataset ...")
    builder = DatasetBuilder(station_matrix, mos_data, cp_data)
    data = builder.build_mos_correction()

    n_features = data["train"]["X"].shape[1]
    logger.info("Features: %d", n_features)
    logger.info("Train: %d, Val: %d, Test: %d, OOS: %d",
                len(data["train"]["y_resid"]),
                len(data["val"]["y_resid"]),
                len(data["test"]["y_resid"]),
                len(data["oos"]["y_resid"]))

    X_train = data["train"]["X"]
    y_train = data["train"]["y_resid"]
    X_val = data["val"]["X"]
    y_val = data["val"]["y_resid"]
    X_test = data["test"]["X"]
    y_test_resid = data["test"]["y_resid"]
    y_test_actual = data["test"]["y_actual"]
    mos_test = data["test"]["mos_base"]
    dates_test = data["test"]["dates"]
    X_oos = data["oos"]["X"]
    y_oos_resid = data["oos"]["y_resid"]
    y_oos_actual = data["oos"]["y_actual"]
    mos_oos = data["oos"]["mos_base"]
    dates_oos = data["oos"]["dates"]

    # Also need train/val for completeness reporting
    y_train_actual = data["train"]["y_actual"]
    mos_train = data["train"]["mos_base"]
    dates_train = data["train"]["dates"]
    y_val_actual = data["val"]["y_actual"]
    mos_val = data["val"]["mos_base"]
    dates_val = data["val"]["dates"]

    # ----------------------------------------------------------------
    # Step 3: Train all variants
    # ----------------------------------------------------------------
    all_results = {}
    all_models = {}
    pit_data = {}

    # --- Baseline ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    baseline_model = train_baseline(n_features, X_train, y_train, X_val, y_val)

    # Evaluate baseline
    for split_name, X_s, y_s_resid, y_s_actual, mos_s, dates_s in [
        ("train", X_train, y_train, y_train_actual, mos_train, dates_train),
        ("val", X_val, y_val, y_val_actual, mos_val, dates_val),
        ("test", X_test, y_test_resid, y_test_actual, mos_test, dates_test),
        ("oos", X_oos, y_oos_resid, y_oos_actual, mos_oos, dates_oos),
    ]:
        pred_resid = predict_nn(baseline_model, X_s)
        pred_actual = mos_s + pred_resid
        r = evaluate_model(y_s_actual, pred_actual, dates_s,
                           f"Baseline {split_name}")
        all_results[f"Baseline_{split_name}"] = r

    # --- Probabilistic variants ---
    variant_configs = {
        "A_NLL_only": train_variant_a,
        "B_NLL_CRPS": train_variant_b,
        "C_Combined": train_variant_c,
        "D_NLL_CRPS_large": train_variant_d,
        "E_NLL_CRPS_lowdrop": train_variant_e,
    }

    for variant_name, train_fn in variant_configs.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        try:
            model = train_fn(n_features, X_train, y_train, X_val, y_val)
            all_models[variant_name] = model

            # Evaluate on all splits
            for split_name, X_s, y_s_resid, y_s_actual, mos_s, dates_s in [
                ("train", X_train, y_train, y_train_actual, mos_train, dates_train),
                ("val", X_val, y_val, y_val_actual, mos_val, dates_val),
                ("test", X_test, y_test_resid, y_test_actual, mos_test, dates_test),
                ("oos", X_oos, y_oos_resid, y_oos_actual, mos_oos, dates_oos),
            ]:
                mu, sigma = predict_probabilistic(model, X_s)
                r = evaluate_probabilistic(y_s_actual, mu, sigma, mos_s, dates_s,
                                           f"{variant_name} {split_name}")
                all_results[f"{variant_name}_{split_name}"] = r

            # PIT histogram on test and OOS
            for pit_split, X_s, y_s_resid, sigma_label in [
                ("test", X_test, y_test_resid, "test"),
                ("oos", X_oos, y_oos_resid, "oos"),
            ]:
                mu, sigma = predict_probabilistic(model, X_s)
                pit_vals = compute_pit(y_s_resid, mu, sigma)
                hist_counts, hist_edges = np.histogram(pit_vals, bins=20, range=(0, 1))
                pit_data[f"{variant_name}_{pit_split}"] = {
                    "counts": hist_counts.tolist(),
                    "edges": hist_edges.tolist(),
                    "raw_pit": pit_vals.tolist(),
                }

        except Exception as e:
            logger.error("Variant %s FAILED: %s", variant_name, e, exc_info=True)

    # ----------------------------------------------------------------
    # Step 4: Build summary tables
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 110)
    logger.info("COMPREHENSIVE RESULTS COMPARISON")
    logger.info("=" * 110)

    # Test set comparison
    logger.info("\n--- TEST SET ---")
    logger.info("%-25s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %8s",
                "Model", "MAE", "RMSE", "CRPS", "Cov50", "Cov80", "Cov90", "Cov95", "MnSigma")
    logger.info("-" * 110)

    # Baseline first
    bk = "Baseline_test"
    if bk in all_results:
        r = all_results[bk]
        logger.info("%-25s  %6.3f  %6.3f  %6s  %6s  %6s  %6s  %6s  %8s",
                     "Baseline (point)", r["mae"], r["rmse"], "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

    for vname in variant_configs:
        key = f"{vname}_test"
        if key in all_results:
            r = all_results[key]
            logger.info("%-25s  %6.3f  %6.3f  %6.3f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %8.3f",
                         vname,
                         r["mae"], r["rmse"], r.get("crps", 0),
                         r.get("cov_50", 0) * 100,
                         r.get("cov_80", 0) * 100,
                         r.get("cov_90", 0) * 100,
                         r.get("cov_95", 0) * 100,
                         r.get("mean_sigma", 0))

    # OOS comparison
    logger.info("\n--- OOS SET ---")
    logger.info("%-25s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %8s",
                "Model", "MAE", "RMSE", "CRPS", "Cov50", "Cov80", "Cov90", "Cov95", "MnSigma")
    logger.info("-" * 110)

    bk = "Baseline_oos"
    if bk in all_results:
        r = all_results[bk]
        logger.info("%-25s  %6.3f  %6.3f  %6s  %6s  %6s  %6s  %6s  %8s",
                     "Baseline (point)", r["mae"], r["rmse"], "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

    for vname in variant_configs:
        key = f"{vname}_oos"
        if key in all_results:
            r = all_results[key]
            logger.info("%-25s  %6.3f  %6.3f  %6.3f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %8.3f",
                         vname,
                         r["mae"], r["rmse"], r.get("crps", 0),
                         r.get("cov_50", 0) * 100,
                         r.get("cov_80", 0) * 100,
                         r.get("cov_90", 0) * 100,
                         r.get("cov_95", 0) * 100,
                         r.get("mean_sigma", 0))

    # Seasonal breakdown for best model on test
    logger.info("\n--- SEASONAL BREAKDOWN (Test Set) ---")
    for vname in list(variant_configs.keys()) + ["Baseline"]:
        key = f"{vname}_test"
        if key in all_results:
            r = all_results[key]
            seasonal_str = ""
            for s in ["DJF", "MAM", "JJA", "SON"]:
                mae_s = r.get(f"mae_{s}", None)
                crps_s = r.get(f"crps_{s}", None)
                if mae_s is not None:
                    if crps_s is not None:
                        seasonal_str += f"  {s}: MAE={mae_s:.2f}/CRPS={crps_s:.2f}"
                    else:
                        seasonal_str += f"  {s}: MAE={mae_s:.2f}"
            if seasonal_str:
                logger.info("  %-25s%s", vname, seasonal_str)

    # ----------------------------------------------------------------
    # Step 5: Save results
    # ----------------------------------------------------------------
    logger.info("\nSaving results ...")

    # JSON results
    clean_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
        else:
            clean_results[k] = v

    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info("Saved %s", json_path)

    # Summary CSV
    summary_rows = []
    for key, metrics in all_results.items():
        if isinstance(metrics, dict) and "mae" in metrics:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_name, split = parts
            else:
                model_name, split = key, "unknown"
            row = {"model": model_name, "split": split, **metrics}
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved %s", summary_path)

    # PIT data JSON
    pit_path = os.path.join(RESULTS_DIR, "pit_data.json")
    with open(pit_path, "w") as f:
        json.dump(pit_data, f, indent=2)
    logger.info("Saved %s", pit_path)

    # Save best model
    # Find best test MAE among probabilistic variants
    best_name = None
    best_test_mae = float("inf")
    for vname in variant_configs:
        key = f"{vname}_test"
        if key in all_results and all_results[key]["mae"] < best_test_mae:
            best_test_mae = all_results[key]["mae"]
            best_name = vname

    if best_name and best_name in all_models:
        model_path = os.path.join(RESULTS_DIR, f"best_probabilistic_model_{best_name}.pt")
        torch.save(all_models[best_name].state_dict(), model_path)
        logger.info("Saved best probabilistic model (%s, test MAE=%.3f) to %s",
                     best_name, best_test_mae, model_path)

    elapsed = time.time() - start_time
    logger.info("\nTotal pipeline time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")

    return all_results


if __name__ == "__main__":
    main()
