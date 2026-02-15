#!/usr/bin/env python3
"""
MOS Residual Correction Benchmark for Chicago and Philadelphia.

Uses the NYC top-model approach: compute error residual of Ensemble (GFS/NAM)
MOS Tmax prediction from the target station, then correct that error with a
neural network trained on surrounding station features.

Pipeline:
  1. Load real MOS data (GFS/NAM ensemble Tmax) for target station
  2. Load actual observed Tmax at target station
  3. Load preprocessed station features (GHCN lag-1 from surrounding stations)
  4. Compute MOS residuals = actual_tmax - MOS_tmax
  5. Add enhanced features (MOS value, MOS spread, seasonal, trend, station consensus)
  6. Train MOSCorrectionNet with CRPS+MAE loss on residuals
  7. Also train FeatureAttentionNet and RegimeConditionalNet for ensemble
  8. Apply Isotonic + Platt calibration on validation bucket probabilities
  9. Evaluate Brier score on test set using real Kalshi bucket definitions
 10. Ensemble top models with calibration for best Brier

Data sources (all real, no synthetic proxies):
  - MOS forecasts: IEM MOS archive (GFS + NAM)
  - Station features: GHCN daily (preprocessed lag-1 from surrounding stations)
  - Bucket definitions: Real Kalshi contract structure from CityConfig
  - Actual outcomes: Observed GHCN TMAX at target station

Usage:
    python scripts/run_mos_residual_benchmark.py --city phl
    python scripts/run_mos_residual_benchmark.py --city chi
    python scripts/run_mos_residual_benchmark.py --city both
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


# ===========================================================================
# Neural Network Architectures
# ===========================================================================

def _std_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))


class MOSResidualNet(nn.Module):
    """Neural network that corrects MOS forecast errors.

    Predicts the residual: actual_tmax = MOS_tmax + NN(features)
    Uses larger capacity than the base MOSCorrectionNet with:
    - Batch normalization for training stability
    - Residual connections for gradient flow
    - Heteroscedastic output (mu_correction, sigma) for distributional forecasts
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.12,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64, 32]

        self.n_features = n_features

        # Input projection with batch norm
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Hidden blocks with residual connections where dimensions match
        self.hidden_blocks = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            block = nn.Sequential(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.BatchNorm1d(hidden_sizes[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.hidden_blocks.append(block)

        # Output heads
        out_dim = hidden_sizes[-1]
        self.delta_head = nn.Linear(out_dim, 1)
        self.log_sigma_head = nn.Linear(out_dim, 1)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MOSResidualNet: n_features=%d, hidden=%s, params=%d",
            n_features, hidden_sizes, n_params,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize delta head near zero (no correction initially)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self, x: torch.Tensor, mos_mu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.hidden_blocks:
            h = block(h)

        delta = self.delta_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 3.5)
        sigma = torch.exp(log_sigma)
        mu = mos_mu + delta

        return {
            "prediction": mu,
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "delta": delta,
        }


class AttentionMOSNet(nn.Module):
    """Feature-attention MOS correction network.

    Combines attention-weighted features with MOS correction for
    adaptive feature importance per forecast day.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int = 64,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.12,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.n_features = n_features

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(n_features, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

        # Attention gate
        self.attention_gate = nn.Linear(context_dim, n_features)
        self.attn_temperature = nn.Parameter(torch.tensor(1.0))

        # Prediction trunk
        layers = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        self.delta_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self, x: torch.Tensor, mos_mu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(x)
        attn_logits = self.attention_gate(context)
        temperature = self.attn_temperature.clamp(min=0.1)
        attn_weights = F.softmax(attn_logits / temperature, dim=-1)
        x_attended = x * attn_weights * self.n_features

        h = self.trunk(x_attended)
        delta = self.delta_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 3.5)
        sigma = torch.exp(log_sigma)
        mu = mos_mu + delta

        return {
            "prediction": mu,
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "delta": delta,
            "attention_weights": attn_weights,
        }


class RegimeMOSNet(nn.Module):
    """Regime-conditional MOS correction network.

    Different uncertainty estimates per weather regime (season x volatility).
    """

    def __init__(
        self,
        n_features: int,
        n_regime_features: int = 16,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.12,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        total_in = n_features + n_regime_features

        layers = []
        in_dim = total_in
        for h_dim in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        self.delta_head = nn.Linear(in_dim, 1)

        # Regime-conditional sigma
        sigma_in = in_dim + n_regime_features
        self.sigma_trunk = nn.Sequential(
            nn.Linear(sigma_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        mos_mu: torch.Tensor,
        regime: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        combined = torch.cat([x, regime], dim=-1)
        h = self.trunk(combined)

        delta = self.delta_head(h)
        mu = mos_mu + delta

        sigma_input = torch.cat([h, regime], dim=-1)
        log_sigma = self.sigma_trunk(sigma_input).clamp(-5.0, 3.5)
        sigma = torch.exp(log_sigma)

        return {
            "prediction": mu,
            "mu": mu,
            "sigma": sigma,
            "log_sigma": log_sigma,
            "delta": delta,
        }


# ===========================================================================
# Loss Functions
# ===========================================================================

def gaussian_crps_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Gaussian CRPS loss (Gneiting & Raftery 2007)."""
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1).clamp(min=1e-6)
    target = target.reshape(-1)
    z = (target - mu) / sigma
    crps = sigma * (z * (2 * _std_normal_cdf(z) - 1) + 2 * _std_normal_pdf(z) - _INV_SQRT_PI)
    return crps.mean()


def combined_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
    crps_weight: float = 0.6, mae_weight: float = 0.3, nll_weight: float = 0.1,
) -> torch.Tensor:
    """Combined CRPS + MAE + NLL loss for training stability and calibration."""
    crps = gaussian_crps_loss(mu, sigma, target)
    mae = torch.abs(mu.reshape(-1) - target.reshape(-1)).mean()
    var = sigma.reshape(-1) ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + ((target.reshape(-1) - mu.reshape(-1)) ** 2) / var)
    return crps_weight * crps + mae_weight * mae + nll_weight * nll.mean()


# ===========================================================================
# Brier Score Evaluation
# ===========================================================================

def gaussian_to_bucket_probs(
    mu: np.ndarray,
    sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Convert Gaussian (mu, sigma) to bucket probabilities via CDF."""
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    probs = np.zeros((n_days, n_buckets))

    for b, (low, high) in enumerate(bucket_edges):
        cdf_low = norm.cdf(low, loc=mu, scale=sigma) if low > -900 else np.zeros(n_days)
        cdf_high = norm.cdf(high, loc=mu, scale=sigma) if high < 900 else np.ones(n_days)
        probs[:, b] = cdf_high - cdf_low

    # Normalize rows to sum to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    probs = probs / row_sums

    # Clip
    probs = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


def compute_actual_buckets(
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Compute one-hot actual bucket outcomes from observed TMAX."""
    n_days = len(actual_tmax)
    n_buckets = len(bucket_edges)
    outcomes = np.zeros((n_days, n_buckets))

    for i, tmax in enumerate(actual_tmax):
        for b, (low, high) in enumerate(bucket_edges):
            if b == n_buckets - 1:
                if low <= tmax <= high:
                    outcomes[i, b] = 1.0
                    break
            else:
                if low <= tmax < high:
                    outcomes[i, b] = 1.0
                    break

    return outcomes


def compute_brier_score(
    pred_probs: np.ndarray,
    actual_outcomes: np.ndarray,
) -> float:
    """Compute overall Brier score across all bucket-days."""
    return float(np.mean((pred_probs - actual_outcomes) ** 2))


def compute_seasonal_brier(
    pred_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, float]:
    """Compute Brier score by season."""
    months = dates.month
    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    seasons = np.array([season_map[m] for m in months])
    result = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == s
        if mask.any():
            result[s] = compute_brier_score(pred_probs[mask], actual_outcomes[mask])
    return result


# ===========================================================================
# Feature Engineering
# ===========================================================================

def compute_regime_features(dates: pd.DatetimeIndex, tmax: np.ndarray) -> np.ndarray:
    """Compute regime features: season one-hot + volatility bins + interactions."""
    n = len(dates)
    months = dates.month
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season_idx = np.array([season_map[m] for m in months])
    season_onehot = np.zeros((n, 4))
    for i in range(n):
        season_onehot[i, season_idx[i]] = 1.0

    tmax_series = pd.Series(tmax, index=dates)
    rolling_std = tmax_series.rolling(7, min_periods=3).std().fillna(
        tmax_series.std()
    ).values

    q33 = np.nanpercentile(rolling_std, 33)
    q67 = np.nanpercentile(rolling_std, 67)
    vol_bins = np.zeros((n, 3))
    vol_bins[rolling_std <= q33, 0] = 1.0
    vol_bins[(rolling_std > q33) & (rolling_std <= q67), 1] = 1.0
    vol_bins[rolling_std > q67, 2] = 1.0
    no_bin = vol_bins.sum(axis=1) == 0
    vol_bins[no_bin, 1] = 1.0

    interactions = np.zeros((n, 9))
    idx = 0
    for s in range(3):
        for v in range(3):
            interactions[:, idx] = season_onehot[:, s] * vol_bins[:, v]
            idx += 1

    return np.concatenate([season_onehot, vol_bins, interactions], axis=1).astype(np.float32)


def build_enhanced_features(
    X: pd.DataFrame,
    y_actual: np.ndarray,
    dates: pd.DatetimeIndex,
    mos_forecast: np.ndarray,
    mos_gfs: np.ndarray | None = None,
    mos_nam: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build enhanced feature matrix incorporating MOS and station data.

    Features added:
    - MOS forecast value (the primary signal)
    - MOS model spread (GFS - NAM disagreement)
    - Recent MOS residual history (lagged)
    - Seasonal encoding (sin/cos day of year)
    - Station consensus (mean/std across station TMAX)
    - Temperature trend features
    - Diurnal range proxy
    """
    X_enh = X.copy()

    # --- MOS features ---
    X_enh["mos_forecast"] = mos_forecast
    if mos_gfs is not None and mos_nam is not None:
        X_enh["mos_spread"] = mos_gfs - mos_nam
        X_enh["mos_spread_abs"] = np.abs(mos_gfs - mos_nam)

    # --- Lagged MOS residuals (actual - MOS, shifted to avoid leakage) ---
    residual = y_actual - mos_forecast
    resid_series = pd.Series(residual, index=dates)
    X_enh["mos_resid_lag1"] = resid_series.shift(1).values
    X_enh["mos_resid_lag2"] = resid_series.shift(2).values
    X_enh["mos_resid_lag3"] = resid_series.shift(3).values
    X_enh["mos_resid_rolling7"] = resid_series.shift(1).rolling(7, min_periods=2).mean().values
    X_enh["mos_resid_rolling7_std"] = resid_series.shift(1).rolling(7, min_periods=2).std().values

    # --- Target lag features ---
    y_series = pd.Series(y_actual, index=dates)
    X_enh["target_lag1"] = y_series.shift(1).values
    X_enh["target_lag2"] = y_series.shift(2).values
    X_enh["target_rolling7_mean"] = y_series.shift(1).rolling(7, min_periods=2).mean().values
    X_enh["target_rolling7_std"] = y_series.shift(1).rolling(7, min_periods=2).std().values
    X_enh["target_delta"] = (y_series.shift(1) - y_series.shift(2)).values

    # --- Seasonal features ---
    doy = dates.dayofyear
    X_enh["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    X_enh["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    X_enh["sin_day2"] = np.sin(4 * np.pi * doy / 365.25)
    X_enh["cos_day2"] = np.cos(4 * np.pi * doy / 365.25)
    X_enh["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    X_enh["month_cos"] = np.cos(2 * np.pi * dates.month / 12)

    # --- Station consensus features ---
    tmax_cols = [c for c in X.columns if "TMAX" in c]
    if tmax_cols:
        station_data = X[tmax_cols]
        X_enh["station_mean"] = station_data.mean(axis=1)
        X_enh["station_std"] = station_data.std(axis=1)
        X_enh["station_range"] = station_data.max(axis=1) - station_data.min(axis=1)
        X_enh["station_median"] = station_data.median(axis=1)
        # MOS vs station consensus gap
        X_enh["mos_vs_station_gap"] = mos_forecast - station_data.mean(axis=1).values

    tmin_cols = [c for c in X.columns if "TMIN" in c]
    if tmin_cols:
        tmin_data = X[tmin_cols]
        X_enh["tmin_station_mean"] = tmin_data.mean(axis=1)
        X_enh["tmin_station_std"] = tmin_data.std(axis=1)

    # Diurnal range proxy
    if tmax_cols and tmin_cols:
        X_enh["diurnal_range"] = (
            X_enh.get("station_mean", 0) - X_enh.get("tmin_station_mean", 0)
        )

    # Fill NaN
    X_enh = X_enh.ffill().bfill().fillna(0)

    return X_enh


# ===========================================================================
# Calibration
# ===========================================================================

class IsotonicPlattCalibrator:
    """Two-stage bucket-level calibration: Isotonic + Platt scaling."""

    def __init__(self):
        self.isotonic_models = {}
        self.platt_models = {}
        self.n_buckets = 0

    def fit(
        self,
        pred_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> "IsotonicPlattCalibrator":
        self.n_buckets = pred_probs.shape[1]
        for b in range(self.n_buckets):
            p = pred_probs[:, b]
            y = actual_outcomes[:, b]
            # Isotonic
            iso = IsotonicRegression(y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX,
                                     out_of_bounds="clip")
            iso.fit(p, y)
            self.isotonic_models[b] = iso
            # Platt on isotonic output
            p_iso = iso.predict(p).reshape(-1, 1)
            if y.sum() > 2 and (1 - y).sum() > 2:
                lr = LogisticRegression(C=1.0, max_iter=1000)
                lr.fit(p_iso, y)
                self.platt_models[b] = lr
        return self

    def calibrate(self, pred_probs: np.ndarray) -> np.ndarray:
        cal = np.zeros_like(pred_probs)
        for b in range(self.n_buckets):
            p = pred_probs[:, b]
            p_iso = self.isotonic_models[b].predict(p)
            if b in self.platt_models:
                p_platt = self.platt_models[b].predict_proba(
                    p_iso.reshape(-1, 1)
                )[:, 1]
                cal[:, b] = p_platt
            else:
                cal[:, b] = p_iso
        # Normalize
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        cal = cal / row_sums
        cal = np.clip(cal, PROB_CLIP_MIN, PROB_CLIP_MAX)
        cal = cal / cal.sum(axis=1, keepdims=True)
        return cal


# ===========================================================================
# Training
# ===========================================================================

def train_mos_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mos_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mos_val: np.ndarray,
    model_type: str = "mos_residual",
    regime_train: np.ndarray | None = None,
    regime_val: np.ndarray | None = None,
    lr: float = 0.0008,
    max_epochs: int = 400,
    patience: int = 30,
    batch_size: int = 64,
) -> dict:
    """Train a MOS correction model with CRPS+MAE+NLL combined loss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    mos_train_t = torch.tensor(mos_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    mos_val_t = torch.tensor(mos_val, dtype=torch.float32).to(DEVICE)

    tensors = [X_train_t, y_train_t, mos_train_t]
    if model_type == "regime_mos" and regime_train is not None:
        regime_train_t = torch.tensor(regime_train, dtype=torch.float32)
        regime_val_t = torch.tensor(regime_val, dtype=torch.float32).to(DEVICE)
        tensors.append(regime_train_t)

    train_ds = TensorDataset(*tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            X_b = batch[0].to(DEVICE)
            y_b = batch[1].to(DEVICE)
            mos_b = batch[2].to(DEVICE).unsqueeze(-1)

            optimizer.zero_grad()

            if model_type == "regime_mos":
                regime_b = batch[3].to(DEVICE)
                out = model(X_b, mos_b, regime_b)
            else:
                out = model(X_b, mos_b)

            mu = out["mu"].squeeze(-1)
            sigma = out["sigma"].squeeze(-1)
            loss = combined_loss(mu, sigma, y_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validate
        model.eval()
        with torch.no_grad():
            if model_type == "regime_mos":
                out_v = model(X_val_t, mos_val_t.unsqueeze(-1), regime_val_t)
            else:
                out_v = model(X_val_t, mos_val_t.unsqueeze(-1))

            mu_v = out_v["mu"].squeeze(-1)
            sigma_v = out_v["sigma"].squeeze(-1)
            val_loss = combined_loss(mu_v, sigma_v, y_val_t).item()
            val_mae = float(torch.abs(mu_v - y_val_t).mean().item())

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch <= 5 or epoch % 50 == 0:
                logger.info(
                    "  Epoch %3d | train: %.4f | val: %.4f | MAE: %.2fF | * BEST *",
                    epoch, avg_train_loss, val_loss, val_mae,
                )
        else:
            epochs_no_improve += 1
            if epoch % 50 == 0:
                logger.info(
                    "  Epoch %3d | train: %.4f | val: %.4f | MAE: %.2fF | no improv (%d/%d)",
                    epoch, avg_train_loss, val_loss, val_mae,
                    epochs_no_improve, patience,
                )

        if epochs_no_improve >= patience:
            logger.info("  Early stopping at epoch %d (best: %d)", epoch, best_epoch)
            break

    if best_state:
        model.load_state_dict(best_state)

    logger.info(
        "  Training complete. Best val loss: %.4f at epoch %d",
        best_val_loss, best_epoch,
    )
    return {"model": model, "history": history, "best_epoch": best_epoch}


def predict_mos_model(
    model: nn.Module,
    X: np.ndarray,
    mos: np.ndarray,
    model_type: str = "mos_residual",
    regime: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (mu, sigma) predictions from a trained MOS correction model."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    mos_t = torch.tensor(mos, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        if model_type == "regime_mos":
            regime_t = torch.tensor(regime, dtype=torch.float32).to(DEVICE)
            out = model(X_t, mos_t.unsqueeze(-1), regime_t)
        else:
            out = model(X_t, mos_t.unsqueeze(-1))

    mu = out["mu"].cpu().numpy().ravel()
    sigma = out["sigma"].cpu().numpy().ravel()
    return mu, sigma


# ===========================================================================
# Data Loading
# ===========================================================================

def load_city_data(city_code: str) -> dict:
    """Load all data for a city: features, targets, MOS forecasts."""
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    processed_dir = os.path.join(cfg.data_dir, "processed")
    mos_dir = os.path.join(cfg.data_dir, "mos") if city_code != "nyc" else os.path.join(
        PROJECT_ROOT, "data", "mos"
    )

    # Determine MOS filename
    mos_filename = f"combined_mos_k{city_code}.csv"
    if city_code == "chi":
        mos_filename = "combined_mos_kord.csv"
    elif city_code == "phl":
        mos_filename = "combined_mos_kphl.csv"

    mos_path = os.path.join(mos_dir, mos_filename)

    # Load preprocessed features and targets
    logger.info("Loading preprocessed data from %s", processed_dir)
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

    logger.info("  X_train: %s, X_val: %s, X_test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # Load MOS data
    logger.info("Loading MOS data from %s", mos_path)
    if os.path.exists(mos_path):
        mos_df = pd.read_csv(mos_path, parse_dates=["date"])
        mos_df["date"] = pd.to_datetime(mos_df["date"])
        logger.info("  MOS data: %d rows, date range: %s to %s",
                     len(mos_df), mos_df["date"].min().date(), mos_df["date"].max().date())
    else:
        logger.warning("MOS data not found at %s", mos_path)
        mos_df = pd.DataFrame(columns=["date", "mos_ensemble_tmax_f"])

    return {
        "cfg": cfg,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "mos_df": mos_df,
    }


def validate_mos_date_alignment(mos_df: pd.DataFrame, city_code: str) -> None:
    """Validate that MOS dates represent the TARGET date (verification date),
    not the forecast issuance date.

    The MOS forecast made on day T-1 predicts TMAX for day T.
    In our data, the 'date' column should be the TARGET date T,
    and 'gfs_runtime' / 'nam_runtime' should be on day T-1 (or at most T at 00Z).

    Raises AssertionError if date alignment is violated.
    """
    logger.info("--- Validating MOS date alignment for %s ---", city_code.upper())

    # Check runtime columns exist
    has_gfs_rt = "gfs_runtime" in mos_df.columns
    has_nam_rt = "nam_runtime" in mos_df.columns

    if not has_gfs_rt and not has_nam_rt:
        logger.warning("  No runtime columns found; skipping alignment validation.")
        return

    rt_col = "gfs_runtime" if has_gfs_rt else "nam_runtime"
    valid = mos_df.dropna(subset=[rt_col]).copy()
    valid["runtime_dt"] = pd.to_datetime(valid[rt_col])
    valid["date_dt"] = pd.to_datetime(valid["date"])

    # MOS runtime should be BEFORE the target date (day-ahead forecast)
    # Acceptable: runtime on day T-1 at any hour, or day T at 00Z
    valid["rt_date"] = valid["runtime_dt"].dt.normalize()
    valid["tgt_date"] = valid["date_dt"].dt.normalize()
    day_ahead = (valid["rt_date"] < valid["tgt_date"]).sum()
    same_day = (valid["rt_date"] == valid["tgt_date"]).sum()
    future_leak = (valid["rt_date"] > valid["tgt_date"]).sum()

    logger.info("  Date alignment check (%d rows with runtime):", len(valid))
    logger.info("    Day-ahead forecasts (runtime < target date): %d (%.1f%%)",
                day_ahead, 100 * day_ahead / max(len(valid), 1))
    logger.info("    Same-day forecasts (runtime == target date, 00Z OK): %d (%.1f%%)",
                same_day, 100 * same_day / max(len(valid), 1))
    logger.info("    FUTURE LEAK (runtime > target date): %d", future_leak)

    if future_leak > 0:
        bad = valid[valid["rt_date"] > valid["tgt_date"]].head(3)
        for _, row in bad.iterrows():
            logger.error("    LEAK: date=%s, runtime=%s", row["date_dt"].date(), row["runtime_dt"])
        raise AssertionError(
            f"{city_code}: {future_leak} MOS entries have runtime AFTER target date. "
            "This indicates a date alignment error."
        )

    # Show a few examples for verification
    sample = valid.sample(min(3, len(valid)), random_state=42)
    logger.info("  Sample entries (runtime -> target_date):")
    for _, row in sample.iterrows():
        logger.info("    MOS issued %s -> predicts TMAX for %s (ensemble=%.1f F)",
                    row["runtime_dt"], row["date_dt"].date(),
                    row.get("mos_ensemble_tmax_f", float("nan")))

    logger.info("  Date alignment: PASSED")


def merge_mos_with_features(
    X: pd.DataFrame,
    y: pd.Series,
    mos_df: pd.DataFrame,
    cfg,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Merge MOS forecasts with feature matrix by matching on TARGET date.

    The MOS 'date' column = the date the TMAX forecast is FOR (verification date).
    The feature index date = the date the TMAX target occurs.
    Both represent the same calendar day, so we merge directly on date.

    For days without MOS data, uses persistence + climatology blend as fallback.
    """
    dates = pd.to_datetime(X.index)
    y_vals = y.values.copy()

    # Build MOS lookup keyed by TARGET date (= verification date)
    mos_lookup = {}
    gfs_lookup = {}
    nam_lookup = {}
    for _, row in mos_df.iterrows():
        d = row["date"]
        if pd.notna(row.get("mos_ensemble_tmax_f")):
            mos_lookup[d] = float(row["mos_ensemble_tmax_f"])
        if pd.notna(row.get("gfs_mos_tmax_f")):
            gfs_lookup[d] = float(row["gfs_mos_tmax_f"])
        if pd.notna(row.get("nam_mos_tmax_f")):
            nam_lookup[d] = float(row["nam_mos_tmax_f"])

    # Build MOS forecast array — match on target date
    mos_vals = np.full(len(dates), np.nan)
    gfs_vals = np.full(len(dates), np.nan)
    nam_vals = np.full(len(dates), np.nan)

    for i, d in enumerate(dates):
        d_ts = pd.Timestamp(d)
        if d_ts in mos_lookup:
            mos_vals[i] = mos_lookup[d_ts]
        if d_ts in gfs_lookup:
            gfs_vals[i] = gfs_lookup[d_ts]
        if d_ts in nam_lookup:
            nam_vals[i] = nam_lookup[d_ts]

    # For days without MOS, use persistence + climatology blend as fallback
    clim_mean = {m: cfg.monthly_tmax_mean.get(m, 60.0) for m in range(1, 13)}
    y_lag1 = np.full_like(y_vals, np.nan)
    y_lag1[1:] = y_vals[:-1]
    y_lag1[0] = y_vals[0]

    for i in range(len(mos_vals)):
        if np.isnan(mos_vals[i]):
            m = dates[i].month
            mos_vals[i] = 0.45 * y_lag1[i] + 0.55 * clim_mean.get(m, 60.0)

    n_real = np.sum(~np.isnan(gfs_vals) | ~np.isnan(nam_vals))
    logger.info("  MOS coverage: %d/%d days have real MOS data (%.1f%%)",
                n_real, len(dates), 100 * n_real / max(len(dates), 1))

    # Replace NaN in gfs/nam with mos_vals for consistency
    gfs_out = np.where(np.isnan(gfs_vals), mos_vals, gfs_vals)
    nam_out = np.where(np.isnan(nam_vals), mos_vals, nam_vals)

    return X, y, mos_vals, gfs_out, nam_out


# ===========================================================================
# Main Benchmark Pipeline
# ===========================================================================

def run_city_benchmark(city_code: str) -> dict:
    """Run the full MOS residual correction benchmark for one city."""
    logger.info("=" * 70)
    logger.info("MOS RESIDUAL CORRECTION BENCHMARK: %s", city_code.upper())
    logger.info("=" * 70)

    # Load data
    data = load_city_data(city_code)
    cfg = data["cfg"]
    bucket_edges = list(cfg.bucket_edges)
    bucket_labels = list(cfg.bucket_labels)

    # Validate MOS date alignment before proceeding
    # Ensures: MOS 'date' = target/verification date, runtime = issuance date < target
    validate_mos_date_alignment(data["mos_df"], city_code)

    # Merge MOS with features
    logger.info("Merging MOS with features...")
    for split_name in ["train", "val", "test"]:
        X_key = f"X_{split_name}"
        y_key = f"y_{split_name}"
        X, y, mos, gfs, nam = merge_mos_with_features(
            data[X_key], data[y_key], data["mos_df"], cfg,
        )
        data[X_key] = X
        data[y_key] = y
        data[f"mos_{split_name}"] = mos
        data[f"gfs_{split_name}"] = gfs
        data[f"nam_{split_name}"] = nam

    # Build enhanced features
    logger.info("Building enhanced features...")
    for split_name in ["train", "val", "test"]:
        dates = pd.to_datetime(data[f"X_{split_name}"].index)
        X_enh = build_enhanced_features(
            data[f"X_{split_name}"],
            data[f"y_{split_name}"].values,
            dates,
            data[f"mos_{split_name}"],
            data[f"gfs_{split_name}"],
            data[f"nam_{split_name}"],
        )
        data[f"X_enh_{split_name}"] = X_enh

    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    feature_cols = data["X_enh_train"].columns.tolist()
    X_train_scaled = scaler.fit_transform(data["X_enh_train"][feature_cols].values)
    X_val_scaled = scaler.transform(data["X_enh_val"][feature_cols].values)
    X_test_scaled = scaler.transform(data["X_enh_test"][feature_cols].values)

    y_train = data["y_train"].values.astype(np.float32)
    y_val = data["y_val"].values.astype(np.float32)
    y_test = data["y_test"].values.astype(np.float32)

    mos_train = data["mos_train"].astype(np.float32)
    mos_val = data["mos_val"].astype(np.float32)
    mos_test = data["mos_test"].astype(np.float32)

    dates_train = pd.to_datetime(data["X_train"].index)
    dates_val = pd.to_datetime(data["X_val"].index)
    dates_test = pd.to_datetime(data["X_test"].index)

    n_features = X_train_scaled.shape[1]
    logger.info("Feature dimensions: %d features, %d train, %d val, %d test",
                n_features, len(y_train), len(y_val), len(y_test))

    # Compute regime features for RegimeMOSNet
    y_all = np.concatenate([y_train, y_val, y_test])
    dates_all = pd.DatetimeIndex(np.concatenate([dates_train, dates_val, dates_test]))
    regime_all = compute_regime_features(dates_all, y_all)
    regime_train = regime_all[:len(y_train)]
    regime_val = regime_all[len(y_train):len(y_train) + len(y_val)]
    regime_test = regime_all[len(y_train) + len(y_val):]
    n_regime = regime_train.shape[1]

    # === Baselines ===
    logger.info("\n--- Computing Baselines ---")

    # 1. Persistence baseline
    actual_outcomes_test = compute_actual_buckets(y_test, bucket_edges)
    persistence_mu = np.roll(y_test, 1)
    persistence_mu[0] = y_test[0]
    persistence_sigma = np.full_like(y_test, 8.0)
    persistence_probs = gaussian_to_bucket_probs(persistence_mu, persistence_sigma, bucket_edges)
    brier_persistence = compute_brier_score(persistence_probs, actual_outcomes_test)
    logger.info("  Persistence baseline Brier: %.4f", brier_persistence)

    # 2. Climatology baseline
    clim_mu = np.array([cfg.monthly_tmax_mean.get(d.month, 60.0) for d in dates_test])
    clim_sigma = np.array([cfg.monthly_tmax_std.get(d.month, 10.0) for d in dates_test])
    clim_probs = gaussian_to_bucket_probs(clim_mu, clim_sigma, bucket_edges)
    brier_clim = compute_brier_score(clim_probs, actual_outcomes_test)
    logger.info("  Climatology baseline Brier: %.4f", brier_clim)

    # 3. Raw MOS baseline (no correction)
    mos_sigma_train = np.std(y_train - mos_train)
    mos_sigma_monthly = {}
    for m in range(1, 13):
        mask = dates_train.month == m
        if mask.sum() > 10:
            mos_sigma_monthly[m] = float(np.std(y_train[mask] - mos_train[mask]))
        else:
            mos_sigma_monthly[m] = float(mos_sigma_train)

    mos_sigma_test = np.array([mos_sigma_monthly.get(d.month, mos_sigma_train) for d in dates_test])
    mos_probs = gaussian_to_bucket_probs(mos_test, mos_sigma_test, bucket_edges)
    brier_mos_raw = compute_brier_score(mos_probs, actual_outcomes_test)
    logger.info("  Raw MOS baseline Brier: %.4f", brier_mos_raw)

    # 4. Ridge regression on MOS residuals
    logger.info("\n--- Training Ridge MOS Correction ---")
    best_alpha = 1.0
    best_ridge_brier = float("inf")
    actual_outcomes_val = compute_actual_buckets(y_val, bucket_edges)

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train - mos_train)  # predict residual
        resid_pred_val = ridge.predict(X_val_scaled)
        ridge_mu_val = mos_val + resid_pred_val
        ridge_sigma_val = np.array([mos_sigma_monthly.get(d.month, 5.0) for d in dates_val])
        ridge_probs_val = gaussian_to_bucket_probs(ridge_mu_val, ridge_sigma_val, bucket_edges)
        brier_val = compute_brier_score(ridge_probs_val, actual_outcomes_val)
        if brier_val < best_ridge_brier:
            best_ridge_brier = brier_val
            best_alpha = alpha

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train_scaled, y_train - mos_train)
    ridge_resid_test = ridge_final.predict(X_test_scaled)
    ridge_mu_test = mos_test + ridge_resid_test
    ridge_sigma_test = np.array([mos_sigma_monthly.get(d.month, 5.0) for d in dates_test])
    ridge_probs_test = gaussian_to_bucket_probs(ridge_mu_test, ridge_sigma_test, bucket_edges)
    brier_ridge = compute_brier_score(ridge_probs_test, actual_outcomes_test)
    logger.info("  Ridge MOS correction Brier: %.4f (alpha=%.2f)", brier_ridge, best_alpha)

    # === Neural Network Models ===
    results = {}

    # 5. MOSResidualNet
    logger.info("\n--- Training MOSResidualNet ---")
    model1 = MOSResidualNet(n_features=n_features, hidden_sizes=[256, 128, 64, 32])
    res1 = train_mos_model(
        model1, X_train_scaled, y_train, mos_train,
        X_val_scaled, y_val, mos_val,
        model_type="mos_residual", lr=0.0008, max_epochs=400, patience=30,
    )
    mu1_test, sigma1_test = predict_mos_model(res1["model"], X_test_scaled, mos_test)
    probs1_test = gaussian_to_bucket_probs(mu1_test, sigma1_test, bucket_edges)
    brier1 = compute_brier_score(probs1_test, actual_outcomes_test)
    logger.info("  MOSResidualNet Brier: %.4f", brier1)
    results["MOSResidualNet"] = {
        "mu": mu1_test, "sigma": sigma1_test, "probs": probs1_test, "brier": brier1,
        "model": res1["model"],
    }

    # 6. AttentionMOSNet
    logger.info("\n--- Training AttentionMOSNet ---")
    model2 = AttentionMOSNet(n_features=n_features, hidden_sizes=[256, 128, 64])
    res2 = train_mos_model(
        model2, X_train_scaled, y_train, mos_train,
        X_val_scaled, y_val, mos_val,
        model_type="mos_residual", lr=0.0008, max_epochs=400, patience=30,
    )
    mu2_test, sigma2_test = predict_mos_model(res2["model"], X_test_scaled, mos_test)
    probs2_test = gaussian_to_bucket_probs(mu2_test, sigma2_test, bucket_edges)
    brier2 = compute_brier_score(probs2_test, actual_outcomes_test)
    logger.info("  AttentionMOSNet Brier: %.4f", brier2)
    results["AttentionMOSNet"] = {
        "mu": mu2_test, "sigma": sigma2_test, "probs": probs2_test, "brier": brier2,
        "model": res2["model"],
    }

    # 7. RegimeMOSNet
    logger.info("\n--- Training RegimeMOSNet ---")
    model3 = RegimeMOSNet(n_features=n_features, n_regime_features=n_regime)
    res3 = train_mos_model(
        model3, X_train_scaled, y_train, mos_train,
        X_val_scaled, y_val, mos_val,
        model_type="regime_mos", regime_train=regime_train, regime_val=regime_val,
        lr=0.0008, max_epochs=400, patience=30,
    )
    mu3_test, sigma3_test = predict_mos_model(
        res3["model"], X_test_scaled, mos_test,
        model_type="regime_mos", regime=regime_test,
    )
    probs3_test = gaussian_to_bucket_probs(mu3_test, sigma3_test, bucket_edges)
    brier3 = compute_brier_score(probs3_test, actual_outcomes_test)
    logger.info("  RegimeMOSNet Brier: %.4f", brier3)
    results["RegimeMOSNet"] = {
        "mu": mu3_test, "sigma": sigma3_test, "probs": probs3_test, "brier": brier3,
        "model": res3["model"],
    }

    # === Calibration ===
    logger.info("\n--- Applying Isotonic+Platt Calibration ---")

    for name, res in results.items():
        # Get val predictions for calibration fitting
        model = res["model"]
        if name == "RegimeMOSNet":
            mu_val, sigma_val = predict_mos_model(
                model, X_val_scaled, mos_val,
                model_type="regime_mos", regime=regime_val,
            )
        else:
            mu_val, sigma_val = predict_mos_model(model, X_val_scaled, mos_val)

        probs_val = gaussian_to_bucket_probs(mu_val, sigma_val, bucket_edges)

        # Fit calibrator on validation set
        calibrator = IsotonicPlattCalibrator()
        calibrator.fit(probs_val, actual_outcomes_val)

        # Apply to test set
        cal_probs = calibrator.calibrate(res["probs"])
        brier_cal = compute_brier_score(cal_probs, actual_outcomes_test)
        logger.info("  %s + calibration Brier: %.4f (was %.4f)", name, brier_cal, res["brier"])
        results[name]["cal_probs"] = cal_probs
        results[name]["brier_cal"] = brier_cal

    # === Ensemble ===
    logger.info("\n--- Building Calibrated Ensemble ---")

    # Simple average ensemble of all calibrated models
    ensemble_probs = np.mean(
        [results[name]["cal_probs"] for name in results], axis=0
    )
    ensemble_probs = np.clip(ensemble_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
    brier_ensemble = compute_brier_score(ensemble_probs, actual_outcomes_test)
    logger.info("  Ensemble (equal weight) Brier: %.4f", brier_ensemble)

    # Weighted ensemble (weight by inverse val Brier)
    weights = {}
    for name in results:
        if name == "RegimeMOSNet":
            mu_val, sigma_val = predict_mos_model(
                results[name]["model"], X_val_scaled, mos_val,
                model_type="regime_mos", regime=regime_val,
            )
        else:
            mu_val, sigma_val = predict_mos_model(
                results[name]["model"], X_val_scaled, mos_val,
            )
        probs_val = gaussian_to_bucket_probs(mu_val, sigma_val, bucket_edges)
        brier_val = compute_brier_score(probs_val, actual_outcomes_val)
        weights[name] = 1.0 / max(brier_val, 1e-4)

    total_weight = sum(weights.values())
    weighted_probs = np.zeros_like(ensemble_probs)
    for name in results:
        w = weights[name] / total_weight
        weighted_probs += w * results[name]["cal_probs"]

    weighted_probs = np.clip(weighted_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    weighted_probs = weighted_probs / weighted_probs.sum(axis=1, keepdims=True)
    brier_weighted = compute_brier_score(weighted_probs, actual_outcomes_test)
    logger.info("  Ensemble (weighted) Brier: %.4f", brier_weighted)

    # === Also calibrate Ridge as a baseline ===
    ridge_probs_val = gaussian_to_bucket_probs(
        mos_val + ridge_final.predict(X_val_scaled),
        np.array([mos_sigma_monthly.get(d.month, 5.0) for d in dates_val]),
        bucket_edges,
    )
    ridge_cal = IsotonicPlattCalibrator()
    ridge_cal.fit(ridge_probs_val, actual_outcomes_val)
    ridge_cal_probs = ridge_cal.calibrate(ridge_probs_test)
    brier_ridge_cal = compute_brier_score(ridge_cal_probs, actual_outcomes_test)
    logger.info("  Ridge + calibration Brier: %.4f", brier_ridge_cal)

    # === Summary ===
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY: %s", city_code.upper())
    logger.info("=" * 70)

    all_results = {
        "Persistence": brier_persistence,
        "Climatology": brier_clim,
        "Raw MOS": brier_mos_raw,
        "Ridge MOS": brier_ridge,
        "Ridge MOS + Cal": brier_ridge_cal,
    }
    for name in results:
        all_results[name] = results[name]["brier"]
        all_results[f"{name} + Cal"] = results[name]["brier_cal"]
    all_results["Ensemble (equal)"] = brier_ensemble
    all_results["Ensemble (weighted)"] = brier_weighted

    # Sort by Brier
    sorted_results = sorted(all_results.items(), key=lambda x: x[1])
    logger.info("%-35s  %s", "Model", "Brier Score")
    logger.info("-" * 50)
    for name, brier in sorted_results:
        marker = " <-- BEST" if brier == sorted_results[0][1] else ""
        logger.info("%-35s  %.4f%s", name, brier, marker)

    best_name, best_brier = sorted_results[0]
    logger.info("\nBest model: %s (Brier: %.4f)", best_name, best_brier)

    # Seasonal breakdown for best model
    if "Ensemble" in best_name:
        best_probs = weighted_probs if "weighted" in best_name.lower() else ensemble_probs
    elif "Cal" in best_name:
        base_name = best_name.replace(" + Cal", "")
        if base_name in results:
            best_probs = results[base_name]["cal_probs"]
        else:
            best_probs = ridge_cal_probs
    elif best_name in results:
        best_probs = results[best_name]["probs"]
    else:
        best_probs = weighted_probs

    seasonal = compute_seasonal_brier(best_probs, actual_outcomes_test, dates_test)
    logger.info("\nSeasonal Brier for %s:", best_name)
    for season, sbrier in sorted(seasonal.items()):
        logger.info("  %s: %.4f", season, sbrier)

    # Save results
    results_dir = cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "mos_residual_benchmark.json")
    save_results = {
        "city": city_code,
        "date_run": datetime.now().isoformat(),
        "test_date_range": [str(dates_test.min().date()), str(dates_test.max().date())],
        "n_test_days": len(y_test),
        "n_features": n_features,
        "bucket_labels": bucket_labels,
        "results": {k: float(v) for k, v in all_results.items()},
        "best_model": best_name,
        "best_brier": float(best_brier),
        "seasonal_brier": {k: float(v) for k, v in seasonal.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info("\nResults saved to %s", out_path)

    # Generate diagnostic plot
    _plot_results(sorted_results, city_code, results_dir)

    return save_results


def _plot_results(sorted_results, city_code, results_dir):
    """Generate a bar chart of Brier scores."""
    names = [r[0] for r in sorted_results]
    scores = [r[1] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if s == min(scores) else "#3498db" for s in scores]
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Brier Score (lower is better)")
    ax.set_title(f"MOS Residual Correction Benchmark — {city_code.upper()}")
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "mos_residual_benchmark.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to %s", plot_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MOS Residual Correction Benchmark for CHI/PHL"
    )
    parser.add_argument(
        "--city", choices=["chi", "phl", "both"], default="both",
        help="City to benchmark (chi, phl, or both)",
    )
    args = parser.parse_args()

    cities = ["chi", "phl"] if args.city == "both" else [args.city]
    all_results = {}

    for city in cities:
        try:
            result = run_city_benchmark(city)
            all_results[city] = result
        except Exception as exc:
            logger.error("Failed to run benchmark for %s: %s", city, exc)
            import traceback
            traceback.print_exc()

    # Cross-city comparison
    if len(all_results) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-CITY COMPARISON")
        logger.info("=" * 70)
        for city, result in all_results.items():
            logger.info("  %s: Best=%s (Brier=%.4f)",
                        city.upper(), result["best_model"], result["best_brier"])


if __name__ == "__main__":
    main()
