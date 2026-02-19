#!/usr/bin/env python3
"""
Unified Multi-City Synthesis Model Training and Calibration Sweep.

Trains U-series synthesis models combining base model predictions with
seasonal context, then applies calibration methods to produce well-calibrated
bucket probabilities for Kalshi temperature contracts.

Calibration methods evaluated:
  1. Raw (uncalibrated) Gaussian -> bucket probs
  2. Isotonic-calibrated per bucket
  3. Platt-scaled per bucket
  4. Seasonal regime split: calibrate separately for DJF/MAM/JJA/SON

All splits are strictly chronological to avoid data leakage.

Replaces the per-city scripts:
  - run_chi_synthesis_calibration.py
  - run_phl_synthesis_calibration.py
  - run_atl_synthesis_calibration.py
  - run_aus_synthesis_calibration.py

Usage:
    python scripts/run_synthesis_calibration.py --city chi
    python scripts/run_synthesis_calibration.py --city phl
    python scripts/run_synthesis_calibration.py --city atl
    python scripts/run_synthesis_calibration.py --city aus
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs

# ---------------------------------------------------------------------------
# City code -> config module mapping
# ---------------------------------------------------------------------------
CITY_CONFIG_MODULES = {
    "nyc": "config_expanded",
    "chi": "config_chicago",
    "phl": "config_philadelphia",
    "atl": "config_atlanta",
    "aus": "config_austin",
}

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

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_base_predictions(results_dir: str, city_code: str = "") -> pd.DataFrame:
    """Load base model predictions from the city benchmark results.

    Looks for the benchmark summary JSON and per-model CSV prediction files
    produced by run_benchmark.py.  Falls back to checking NN prediction files.

    Parameters
    ----------
    results_dir : str
        Path to results/<city>/ directory.
    city_code : str
        City code for error messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, model_name, mu, sigma, actual_tmax.
    """
    predictions_csv = os.path.join(results_dir, "base_predictions.csv")

    if os.path.isfile(predictions_csv):
        logger.info("Loading base predictions from %s", predictions_csv)
        df = pd.read_csv(predictions_csv, parse_dates=["date"])
        required_cols = {"date", "model_name", "mu", "sigma", "actual_tmax"}
        if required_cols.issubset(df.columns):
            return df

    # Fallback: check for individual model prediction files
    nn_preds_path = os.path.join(results_dir, "nn_predictions.csv")
    if os.path.isfile(nn_preds_path):
        logger.info("Loading NN predictions from %s", nn_preds_path)
        df = pd.read_csv(nn_preds_path, parse_dates=["date"])
        if "model_name" not in df.columns:
            df["model_name"] = "flat_nn"
        return df

    # AUDIT FIX: Synthetic data fallback removed — must have real benchmark data.
    raise RuntimeError(
        f"No benchmark predictions found in {results_dir}. "
        f"Run the benchmark script (run_benchmark.py --city {city_code}) first to "
        f"generate real model predictions. Synthetic data fallback has been "
        f"removed to prevent silent corruption of evaluation results."
    )




def load_kalshi_contract_rows(city_code: str, valid_dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Load day-specific listed Kalshi contracts for a city from pre-settlement snapshots.

    Uses real pre-settlement rows (with threshold_low/high) joined to settled outcomes,
    then filters to dates present in the model predictions.
    """
    pre_path = PROJECT_ROOT / "data" / f"kalshi_presettlement_{city_code}.csv"
    settled_path = PROJECT_ROOT / "data" / f"real_kalshi_{city_code}_all.csv"
    if not pre_path.exists() or not settled_path.exists():
        raise FileNotFoundError(
            f"Missing Kalshi files for {city_code}: {pre_path} and/or {settled_path}"
        )

    pre = pd.read_csv(pre_path)
    settled = pd.read_csv(settled_path)
    pre["date"] = pd.to_datetime(pre["date"]).dt.normalize()
    settled["date"] = pd.to_datetime(settled["date"]).dt.normalize()

    merge_keys = ["date", "ticker"]
    if "bucket" in pre.columns and "bucket" in settled.columns:
        merge_keys.append("bucket")

    cols_pre = [c for c in ["date", "ticker", "bucket", "threshold_low", "threshold_high", "direction", "presettlement_prob"] if c in pre.columns]
    cols_set = [c for c in ["date", "ticker", "bucket", "actual_outcome", "actual_tmax"] if c in settled.columns]

    rows = pre[cols_pre].merge(settled[cols_set], on=merge_keys, how="inner")
    rows = rows.dropna(subset=["actual_outcome", "threshold_low", "threshold_high"])

    keep_dates = pd.to_datetime(valid_dates).dt.normalize().unique()
    rows = rows[rows["date"].isin(keep_dates)].copy()

    # Enforce canonical between/below/above semantics from thresholds
    rows["threshold_low"] = rows["threshold_low"].astype(float)
    rows["threshold_high"] = rows["threshold_high"].astype(float)
    if "direction" not in rows.columns:
        rows["direction"] = "between"

    rows["direction"] = rows["direction"].fillna("between").str.lower()
    rows.loc[rows["threshold_low"] <= -900, "direction"] = "below"
    rows.loc[rows["threshold_high"] >= 900, "direction"] = "above"

    rows = rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"No overlapping Kalshi contract rows found for {city_code}.")
    return rows

# ===========================================================================
# Bucket Probability Conversion
# ===========================================================================

def compute_bucket_probs_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Convert Gaussian (mu, sigma) predictions to bucket probabilities.

    For each day, computes P(low <= TMAX < high) = CDF(high) - CDF(low)
    for each bucket.  Open-ended sentinels (-999 / 999) are treated as
    -inf / +inf.

    Parameters
    ----------
    mu : np.ndarray
        Predicted means, shape (n_days,).
    sigma : np.ndarray
        Predicted standard deviations, shape (n_days,).
    bucket_edges : list of (low, high) tuples
        Kalshi contract bucket boundaries.

    Returns
    -------
    np.ndarray
        Shape (n_days, n_buckets) with probabilities summing to ~1.0 per row.
    """
    sigma = np.maximum(sigma, 1e-6)
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    probs = np.zeros((n_days, n_buckets))

    for b, (lo, hi) in enumerate(bucket_edges):
        cdf_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        cdf_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        probs[:, b] = np.clip(cdf_hi - cdf_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Normalize rows to sum to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sums, 1e-10)

    return probs


def compute_brier_score(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> dict:
    """Compute Brier score across all bucket-days.

    For each day and each bucket, the Brier score component is
    (predicted_prob - actual_outcome)^2, where actual_outcome is 1 if
    the observed TMAX falls in that bucket and 0 otherwise.

    Parameters
    ----------
    bucket_probs : np.ndarray
        Shape (n_days, n_buckets) of predicted bucket probabilities.
    actual_tmax : np.ndarray
        Shape (n_days,) of observed maximum temperatures (deg F).
    bucket_edges : list of (low, high) tuples
        Bucket boundary definitions.

    Returns
    -------
    dict
        Dictionary with keys: overall_brier, per_bucket_brier, n_days,
        n_buckets.
    """
    n_days, n_buckets = bucket_probs.shape
    assert len(actual_tmax) == n_days

    # Build outcome matrix
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
    overall_brier = float(np.mean(brier_components))
    per_bucket_brier = [
        float(np.mean(brier_components[:, b])) for b in range(n_buckets)
    ]

    return {
        "overall_brier": overall_brier,
        "per_bucket_brier": per_bucket_brier,
        "n_days": n_days,
        "n_buckets": n_buckets,
    }


# ===========================================================================
# Synthesis MLP
# ===========================================================================

class SynthesisMLP(nn.Module):
    """Small 3-layer MLP for synthesis model.

    Takes base model (mu, sigma), seasonal features (sin_day, cos_day),
    and interaction features as input.  Outputs calibrated (mu, sigma)
    for a heteroscedastic Gaussian distribution.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list[int]
        Sizes of the three hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden_sizes: Optional[list[int]] = None,
        dropout: float = 0.15,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_sizes[-1], 1)
        self.log_sigma_head = nn.Linear(hidden_sizes[-1], 1)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SynthesisMLP created: n_features=%d, hidden=%s, "
            "dropout=%.2f, params=%d",
            n_features, hidden_sizes, dropout, n_params,
        )

    def _init_weights(self) -> None:
        """Xavier-uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, n_features).

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: mu (batch, 1), log_sigma (batch, 1), sigma (batch, 1).
        """
        hidden = self.trunk(x)
        mu = self.mu_head(hidden)
        log_sigma = self.log_sigma_head(hidden)
        log_sigma = log_sigma.clamp(min=-10.0, max=5.0)
        sigma = torch.exp(log_sigma)

        return {"mu": mu, "sigma": sigma, "log_sigma": log_sigma}


def _build_synthesis_features(
    df: pd.DataFrame,
    cfg,
) -> np.ndarray:
    """Build synthesis feature matrix from prediction DataFrame.

    Features:
      0. mu               - base model predicted mean
      1. sigma             - base model predicted std
      2. sin_day           - sin(2*pi*dayofyear/365.25)
      3. cos_day           - cos(2*pi*dayofyear/365.25)
      4. mu_sigma_interaction  - mu * sigma
      5. mu_seasonal_anomaly   - (mu - seasonal_mean) / seasonal_std

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, mu, sigma.
    cfg : CityConfig
        City configuration with monthly_tmax_mean and monthly_tmax_std.

    Returns
    -------
    np.ndarray
        Shape (n_days, 6) feature matrix.
    """
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear.values.astype(float)
    months = dates.dt.month.values

    mu = df["mu"].values.astype(float)
    sigma = df["sigma"].values.astype(float)

    sin_day = np.sin(2.0 * np.pi * doy / 365.25)
    cos_day = np.cos(2.0 * np.pi * doy / 365.25)

    # Interaction: mu * sigma
    mu_sigma_interaction = mu * sigma

    # Seasonal anomaly: (mu - monthly_mean) / monthly_std
    seasonal_mean = np.array([cfg.monthly_tmax_mean[m] for m in months])
    seasonal_std = np.array([cfg.monthly_tmax_std[m] for m in months])
    seasonal_std = np.maximum(seasonal_std, 1.0)
    mu_seasonal_anomaly = (mu - seasonal_mean) / seasonal_std

    features = np.column_stack([
        mu, sigma, sin_day, cos_day,
        mu_sigma_interaction, mu_seasonal_anomaly,
    ])

    return features.astype(np.float32)


def train_synthesis_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    model_name: str = "U_chi_synthesis",
) -> dict:
    """Train a synthesis MLP with early stopping.

    Uses a Gaussian CRPS-inspired loss: NLL of heteroscedastic Gaussian.
    Loss = 0.5 * log(sigma^2) + 0.5 * ((y - mu) / sigma)^2

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, n_features).
    y_train : np.ndarray
        Training targets, shape (n_train,).
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation targets.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    model_name : str
        Name for logging.

    Returns
    -------
    dict
        Keys: model, scaler, best_epoch, best_val_loss, history.
    """
    n_features = X_train.shape[1]

    # Scale features (fit on training only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Build DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(y_train.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled),
        torch.tensor(y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SynthesisMLP(n_features=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state_dict = None
    history: list[dict] = []

    logger.info("Training synthesis MLP '%s' (features=%d, device=%s)",
                model_name, n_features, DEVICE)

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            out = model(X_batch)
            mu_pred = out["mu"].squeeze(-1)
            sigma_pred = out["sigma"].squeeze(-1)

            # Gaussian NLL loss
            loss = torch.mean(
                0.5 * torch.log(sigma_pred ** 2 + 1e-8)
                + 0.5 * ((y_batch - mu_pred) / (sigma_pred + 1e-8)) ** 2
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss_sum / max(n_train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        all_mu = []
        all_sigma = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                out = model(X_batch)
                mu_pred = out["mu"].squeeze(-1)
                sigma_pred = out["sigma"].squeeze(-1)

                loss = torch.mean(
                    0.5 * torch.log(sigma_pred ** 2 + 1e-8)
                    + 0.5 * ((y_batch - mu_pred) / (sigma_pred + 1e-8)) ** 2
                )

                val_loss_sum += loss.item()
                n_val_batches += 1

                all_mu.append(mu_pred.cpu().numpy())
                all_sigma.append(sigma_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)
        val_mu = np.concatenate(all_mu)
        val_targets = np.concatenate(all_targets)
        val_mae = float(np.mean(np.abs(val_mu - val_targets)))

        scheduler.step(avg_val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": val_mae,
        })

        # --- Early stopping check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch <= 5 or epoch % 20 == 0:
                logger.info(
                    "  Epoch %3d | Train: %.4f | Val: %.4f | MAE: %.2f F | * BEST *",
                    epoch, avg_train_loss, avg_val_loss, val_mae,
                )
        else:
            epochs_no_improve += 1
            if epoch <= 5 or epoch % 20 == 0:
                logger.info(
                    "  Epoch %3d | Train: %.4f | Val: %.4f | MAE: %.2f F | "
                    "No improvement (%d/%d)",
                    epoch, avg_train_loss, avg_val_loss, val_mae,
                    epochs_no_improve, patience,
                )

        if epochs_no_improve >= patience:
            logger.info("  Early stopping at epoch %d", epoch)
            break

    # Load best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    logger.info(
        "Synthesis MLP '%s' training complete. Best val loss: %.4f at epoch %d",
        model_name, best_val_loss, best_epoch,
    )

    return {
        "model": model,
        "scaler": scaler,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def predict_synthesis(
    model: nn.Module,
    scaler: StandardScaler,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a trained synthesis MLP.

    Parameters
    ----------
    model : nn.Module
        Trained SynthesisMLP.
    scaler : StandardScaler
        Fitted feature scaler.
    X : np.ndarray
        Raw features, shape (n_days, n_features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mu, sigma) arrays each of shape (n_days,).
    """
    model.eval()
    X_scaled = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        out = model(X_tensor)
        mu = out["mu"].squeeze(-1).cpu().numpy()
        sigma = out["sigma"].squeeze(-1).cpu().numpy()

    return mu, sigma


# ===========================================================================
# Calibration Methods
# ===========================================================================

def apply_isotonic_calibration(
    probs_train: np.ndarray,
    actual_train: np.ndarray,
    probs_val: np.ndarray,
) -> np.ndarray:
    """Apply isotonic regression calibration per bucket.

    Fits a separate isotonic regression for each bucket, mapping raw
    predicted probabilities to calibrated probabilities that better
    match observed frequencies.

    Parameters
    ----------
    probs_train : np.ndarray
        Training bucket probabilities, shape (n_train, n_buckets).
    actual_train : np.ndarray
        Training binary outcomes, shape (n_train, n_buckets).
    probs_val : np.ndarray
        Validation/test bucket probabilities to calibrate,
        shape (n_val, n_buckets).

    Returns
    -------
    np.ndarray
        Calibrated probabilities, shape (n_val, n_buckets).
    """
    n_buckets = probs_train.shape[1]
    calibrated = np.zeros_like(probs_val)

    for b in range(n_buckets):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs_train[:, b], actual_train[:, b])
        calibrated[:, b] = iso.transform(probs_val[:, b])

    # Re-normalize to sum to 1
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-10)

    return calibrated


def apply_platt_scaling(
    probs_train: np.ndarray,
    actual_train: np.ndarray,
    probs_val: np.ndarray,
) -> np.ndarray:
    """Apply Platt scaling (logistic regression) calibration per bucket.

    Fits a logistic regression on each bucket's predicted probability
    to produce calibrated probability estimates.

    Parameters
    ----------
    probs_train : np.ndarray
        Training bucket probabilities, shape (n_train, n_buckets).
    actual_train : np.ndarray
        Training binary outcomes, shape (n_train, n_buckets).
    probs_val : np.ndarray
        Validation/test bucket probabilities to calibrate.

    Returns
    -------
    np.ndarray
        Calibrated probabilities, shape (n_val, n_buckets).
    """
    n_buckets = probs_train.shape[1]
    calibrated = np.zeros_like(probs_val)

    for b in range(n_buckets):
        y_b = actual_train[:, b]
        # Platt scaling needs both classes represented
        if y_b.sum() < 2 or y_b.sum() > len(y_b) - 2:
            # Too few positives or negatives; fall back to raw
            calibrated[:, b] = probs_val[:, b]
            continue

        # Use log-odds as feature for Platt scaling
        p_train = np.clip(probs_train[:, b], 1e-6, 1 - 1e-6)
        log_odds_train = np.log(p_train / (1 - p_train)).reshape(-1, 1)

        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(log_odds_train, y_b)

        p_val = np.clip(probs_val[:, b], 1e-6, 1 - 1e-6)
        log_odds_val = np.log(p_val / (1 - p_val)).reshape(-1, 1)
        calibrated[:, b] = lr.predict_proba(log_odds_val)[:, 1]

    # Re-normalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-10)

    return calibrated


def apply_seasonal_calibration(
    probs_train: np.ndarray,
    actual_train: np.ndarray,
    dates_train: np.ndarray,
    probs_val: np.ndarray,
    dates_val: np.ndarray,
) -> np.ndarray:
    """Apply seasonal regime-conditional isotonic calibration.

    Fits separate isotonic calibrators for each meteorological season
    (DJF, MAM, JJA, SON).  Falls back to a global calibrator for
    seasons with insufficient training samples.

    Parameters
    ----------
    probs_train : np.ndarray
        Training bucket probabilities, shape (n_train, n_buckets).
    actual_train : np.ndarray
        Training binary outcomes, shape (n_train, n_buckets).
    dates_train : np.ndarray
        Training dates (datetime-like), shape (n_train,).
    probs_val : np.ndarray
        Validation bucket probabilities, shape (n_val, n_buckets).
    dates_val : np.ndarray
        Validation dates, shape (n_val,).

    Returns
    -------
    np.ndarray
        Calibrated probabilities, shape (n_val, n_buckets).
    """
    n_buckets = probs_train.shape[1]
    calibrated = np.zeros_like(probs_val)

    dates_train_pd = pd.to_datetime(dates_train)
    dates_val_pd = pd.to_datetime(dates_val)
    months_train = dates_train_pd.month.values
    months_val = dates_val_pd.month.values

    # Assign seasons
    seasons_train = np.array([SEASON_MAP[m] for m in months_train])
    seasons_val = np.array([SEASON_MAP[m] for m in months_val])

    # Fit global fallback calibrators per bucket
    global_calibrators = []
    for b in range(n_buckets):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs_train[:, b], actual_train[:, b])
        global_calibrators.append(iso)

    # Fit per-season calibrators
    for season in SEASON_ORDER:
        train_mask = seasons_train == season
        val_mask = seasons_val == season

        if val_mask.sum() == 0:
            continue

        for b in range(n_buckets):
            if train_mask.sum() < 20:
                # Not enough seasonal data; use global
                calibrated[val_mask, b] = global_calibrators[b].transform(
                    probs_val[val_mask, b]
                )
            else:
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                y_season = actual_train[train_mask, b]
                # Need at least some variance
                if y_season.std() < 1e-8:
                    calibrated[val_mask, b] = probs_val[val_mask, b]
                else:
                    iso.fit(probs_train[train_mask, b], y_season)
                    calibrated[val_mask, b] = iso.transform(
                        probs_val[val_mask, b]
                    )

    # Re-normalize
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = calibrated / np.maximum(row_sums, 1e-10)

    return calibrated


# ===========================================================================
# Reliability and ECE
# ===========================================================================

def compute_reliability_diagram(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data.

    Bins predictions into n_bins equally spaced intervals and computes
    the observed frequency in each bin.  Also computes the Expected
    Calibration Error (ECE).

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (N,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (N,).
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        Keys: bin_centers, bin_observed_freq, bin_predicted_mean, bin_counts,
        ece, n_samples.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_observed_freq = []
    bin_predicted_mean = []
    bin_counts = []

    total_ece = 0.0
    total_samples = len(predicted_probs)

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (predicted_probs >= lo) & (predicted_probs <= hi)
        else:
            mask = (predicted_probs >= lo) & (predicted_probs < hi)

        count = mask.sum()
        bin_counts.append(int(count))
        bin_centers.append((lo + hi) / 2.0)

        if count > 0:
            obs_freq = float(actual_outcomes[mask].mean())
            pred_mean = float(predicted_probs[mask].mean())
            bin_observed_freq.append(obs_freq)
            bin_predicted_mean.append(pred_mean)
            total_ece += count * abs(obs_freq - pred_mean)
        else:
            bin_observed_freq.append(float("nan"))
            bin_predicted_mean.append(float("nan"))

    ece = total_ece / max(total_samples, 1)

    return {
        "bin_centers": bin_centers,
        "bin_observed_freq": bin_observed_freq,
        "bin_predicted_mean": bin_predicted_mean,
        "bin_counts": bin_counts,
        "ece": float(ece),
        "n_samples": total_samples,
    }


def compute_ece(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b, acc is the observed
    accuracy (frequency), and conf is the mean predicted probability.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Predicted probabilities, shape (N,).
    actual_outcomes : np.ndarray
        Binary outcomes (0 or 1), shape (N,).
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        Expected Calibration Error in [0, 1].
    """
    rel = compute_reliability_diagram(predicted_probs, actual_outcomes, n_bins)
    return rel["ece"]


def _build_outcome_matrix(
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Build binary outcome matrix from actual temperatures.

    Parameters
    ----------
    actual_tmax : np.ndarray
        Observed TMAX values, shape (n_days,).
    bucket_edges : list of (low, high) tuples
        Bucket boundaries.

    Returns
    -------
    np.ndarray
        Binary outcome matrix, shape (n_days, n_buckets).
    """
    n_days = len(actual_tmax)
    n_buckets = len(bucket_edges)
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

    return outcomes


# ===========================================================================
# Visualization
# ===========================================================================

def plot_reliability(
    reliability_data: dict,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> None:
    """Plot a reliability diagram from computed reliability data.

    Parameters
    ----------
    reliability_data : dict
        Output from compute_reliability_diagram().
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    centers = reliability_data["bin_centers"]
    observed = reliability_data["bin_observed_freq"]
    counts = reliability_data["bin_counts"]

    # Filter out empty bins
    valid = [i for i, c in enumerate(counts) if c > 0]
    centers_v = [centers[i] for i in valid]
    observed_v = [observed[i] for i in valid]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(centers_v, observed_v, "o-", color="#d62728", linewidth=2,
            markersize=6, label="Model")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"{title}\nECE = {reliability_data['ece']:.4f}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved reliability diagram to %s", save_path)

    plt.close(fig)


def plot_brier_comparison(
    results: dict[str, float],
    title: str = "Calibration Sweep: Brier Scores",
    save_path: Optional[str] = None,
) -> None:
    """Plot a bar chart comparing Brier scores across calibration methods.

    Parameters
    ----------
    results : dict[str, float]
        Mapping of method name to overall Brier score.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure.
    """
    methods = list(results.keys())
    scores = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    bars = ax.bar(range(len(methods)), scores, color=bar_colors, edgecolor="black")
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight best
    best_idx = int(np.argmin(scores))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Brier comparison to %s", save_path)

    plt.close(fig)


# ===========================================================================
# Calibration Sweep
# ===========================================================================

def run_calibration_sweep(
    predictions_df: pd.DataFrame,
    bucket_edges: list[tuple[float, float]],
    output_dir: str,
    city_code: str = "chi",
    city_name: str = "Chicago",
    kalshi_ticker: str = "KXHIGHCHI",
) -> dict:
    """Run calibration sweep using day-specific listed Kalshi contracts."""
    os.makedirs(output_dir, exist_ok=True)

    df = predictions_df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    contracts = load_kalshi_contract_rows(city_code, df["date"])
    contracts = contracts.merge(
        df[["date", "mu", "sigma"]],
        on="date",
        how="inner",
    )
    contracts = contracts.dropna(subset=["mu", "sigma", "actual_outcome"])

    sigma = np.maximum(contracts["sigma"].values.astype(float), 1e-6)
    mu = contracts["mu"].values.astype(float)
    lo = contracts["threshold_low"].values.astype(float)
    hi = contracts["threshold_high"].values.astype(float)
    direction = contracts["direction"].astype(str).str.lower().values

    probs = np.full(len(contracts), np.nan, dtype=float)
    below = (direction == "below") | (direction == "less")
    above = direction == "above"
    between = ~(below | above)

    probs[below] = norm.cdf(hi[below], loc=mu[below], scale=sigma[below])
    probs[above] = 1.0 - norm.cdf(lo[above], loc=mu[above], scale=sigma[above])
    probs[between] = (
        norm.cdf(hi[between], loc=mu[between], scale=sigma[between])
        - norm.cdf(lo[between], loc=mu[between], scale=sigma[between])
    )
    contracts["raw_prob"] = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    contracts["outcome"] = contracts["actual_outcome"].astype(float)

    unique_dates = np.array(sorted(contracts["date"].unique()))
    n_dates = len(unique_dates)
    n_train = int(n_dates * 0.70)
    n_val = int(n_dates * 0.15)
    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train:n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val:])

    train_mask = contracts["date"].isin(train_dates).values
    val_mask = contracts["date"].isin(val_dates).values
    test_mask = contracts["date"].isin(test_dates).values

    logger.info(
        "Calibration sweep (contract rows): %d rows across %d dates (train=%d, val=%d, test=%d)",
        len(contracts), n_dates, len(train_dates), len(val_dates), len(test_dates),
    )

    cal_mask = train_mask | val_mask
    p_cal = contracts.loc[cal_mask, "raw_prob"].values
    y_cal = contracts.loc[cal_mask, "outcome"].values
    p_test_raw = contracts.loc[test_mask, "raw_prob"].values
    y_test = contracts.loc[test_mask, "outcome"].values

    results: dict = {}

    def _pack(name: str, p_test: np.ndarray):
        p_test = np.clip(p_test, PROB_CLIP_MIN, PROB_CLIP_MAX)
        brier = float(np.mean((p_test - y_test) ** 2))
        ece = compute_ece(p_test, y_test)
        rel = compute_reliability_diagram(p_test, y_test)
        results[name] = {"brier": brier, "ece": ece, "reliability": rel}

    _pack("1_raw_uncalibrated", p_test_raw)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    _pack("2_isotonic", iso.transform(p_test_raw))

    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_cal.reshape(-1, 1), y_cal)
        p_platt = lr.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
    except Exception:
        p_platt = p_test_raw.copy()
    _pack("3_platt_scaling", p_platt)

    season_cal = pd.to_datetime(contracts.loc[cal_mask, "date"]).dt.month.map(SEASON_MAP).values
    season_test = pd.to_datetime(contracts.loc[test_mask, "date"]).dt.month.map(SEASON_MAP).values
    p_seasonal = p_test_raw.copy()
    for season in SEASON_ORDER:
        m_cal = season_cal == season
        m_test = season_test == season
        if not m_test.any():
            continue
        if m_cal.sum() < 20:
            p_seasonal[m_test] = iso.transform(p_test_raw[m_test])
            continue
        iso_s = IsotonicRegression(out_of_bounds="clip")
        iso_s.fit(p_cal[m_cal], y_cal[m_cal])
        p_seasonal[m_test] = iso_s.transform(p_test_raw[m_test])
    _pack("4_seasonal_regime", p_seasonal)

    method_briers = {k: v["brier"] for k, v in results.items()}
    best_method = min(method_briers, key=method_briers.get)
    results["best_method"] = best_method

    for method_name, info in results.items():
        if method_name == "best_method":
            continue
        plot_reliability(
            info["reliability"],
            title=f"{city_code.upper()} {method_name}: Reliability (listed contracts)",
            save_path=os.path.join(output_dir, f"reliability_{method_name}.png"),
        )

    # Persist best-method reliability under canonical filename for promotion checks.
    plot_reliability(
        results[best_method]["reliability"],
        title=f"{city_code.upper()} best calibration ({best_method})",
        save_path=os.path.join(output_dir, "reliability_diagram.png"),
    )

    plot_brier_comparison(
        method_briers,
        title=f"{city_code.upper()} Calibration Sweep: Contract-row Brier",
        save_path=os.path.join(output_dir, "brier_comparison.png"),
    )

    summary = {
        "city": city_name,
        "ticker": kalshi_ticker,
        "evaluation_unit": "listed_contract_rows",
        "contract_granularity": "day-specific Kalshi pre-settlement rows (typically 2F increments)",
        "n_total_rows": int(len(contracts)),
        "n_total_dates": int(n_dates),
        "n_test_rows": int(test_mask.sum()),
        "n_test_dates": int(len(test_dates)),
        "n_buckets": len(bucket_edges),
        "methods": {k: {"brier": v["brier"], "ece": v["ece"]} for k, v in results.items() if k != "best_method"},
        "best_method": best_method,
        "best_brier": method_briers[best_method],
    }

    results["summary"] = summary
    summary_path = os.path.join(output_dir, "calibration_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved calibration sweep summary to %s", summary_path)

    return results


# ===========================================================================
# Main Orchestration
# ===========================================================================

def main() -> None:
    """Run the full synthesis model training and calibration sweep for a city."""
    parser = argparse.ArgumentParser(
        description="Synthesis model training and calibration sweep."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=sorted(CITY_CONFIG_MODULES.keys()),
        help="City code (chi, phl, atl, aus)",
    )
    args = parser.parse_args()
    city_code = args.city

    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    synthesis_dir = os.path.join(cfg.results_dir, "synthesis")
    os.makedirs(synthesis_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("%s Synthesis Model Training & Calibration Sweep", cfg.city_name)
    logger.info("  City:     %s", cfg.city_name)
    logger.info("  Ticker:   %s", cfg.kalshi_ticker)
    logger.info("  Station:  %s (%s)", cfg.target_station, cfg.target_station_name)
    logger.info("  Buckets:  %d", len(cfg.bucket_edges))
    logger.info("  Output:   %s", synthesis_dir)
    logger.info("=" * 70)

    # ---- Step 1: Load base predictions ----
    logger.info("Step 1: Loading base model predictions ...")
    base_df = load_base_predictions(cfg.results_dir, city_code)
    logger.info("  Loaded %d predictions (%s to %s)",
                len(base_df),
                base_df["date"].min(),
                base_df["date"].max())


    # Drop rows with missing actual_tmax — cannot train or evaluate without targets
    n_before = len(base_df)
    base_df = base_df.dropna(subset=["actual_tmax"]).reset_index(drop=True)
    n_dropped = n_before - len(base_df)
    if n_dropped > 0:
        logger.info("  Dropped %d rows with NaN actual_tmax (%d remaining)",
                     n_dropped, len(base_df))
    # ---- Step 2: Build synthesis features ----
    logger.info("Step 2: Building synthesis features ...")
    features = _build_synthesis_features(base_df, cfg)
    logger.info("  Feature matrix shape: %s", features.shape)

    # ---- Step 3: Chronological split ----
    n = len(base_df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    X_train = features[:n_train]
    X_val = features[n_train:n_train + n_val]
    X_test = features[n_train + n_val:]

    y_train = base_df["actual_tmax"].values[:n_train]
    y_val = base_df["actual_tmax"].values[n_train:n_train + n_val]
    y_test = base_df["actual_tmax"].values[n_train + n_val:]

    logger.info("  Train: %d, Val: %d, Test: %d", len(y_train), len(y_val), len(y_test))

    # ---- Step 4: Train synthesis MLP ----
    logger.info("Step 3: Training synthesis MLP ...")
    train_result = train_synthesis_mlp(
        X_train, y_train, X_val, y_val,
        max_epochs=200,
        patience=15,
        batch_size=64,
        lr=0.001,
        model_name=f"U_{city_code}_synthesis",
    )

    model = train_result["model"]
    scaler = train_result["scaler"]

    # Save model checkpoint
    model_path = os.path.join(synthesis_dir, "synthesis_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info("  Saved synthesis model to %s", model_path)

    scaler_path = os.path.join(synthesis_dir, "synthesis_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("  Saved scaler to %s", scaler_path)

    # ---- Step 5: Generate synthesis predictions ----
    logger.info("Step 4: Generating synthesis predictions ...")
    synth_mu_all, synth_sigma_all = predict_synthesis(model, scaler, features)

    # Build synthesis prediction DataFrame
    synth_df = base_df[["date", "actual_tmax"]].copy()
    synth_df["mu"] = synth_mu_all
    synth_df["sigma"] = synth_sigma_all

    # Compute MAE on test set
    test_mu = synth_mu_all[n_train + n_val:]
    test_mae = float(np.mean(np.abs(test_mu - y_test)))
    logger.info("  Synthesis test MAE: %.2f F", test_mae)

    # Save predictions
    preds_path = os.path.join(synthesis_dir, "synthesis_predictions.csv")
    synth_df.to_csv(preds_path, index=False)
    logger.info("  Saved synthesis predictions to %s", preds_path)

    # ---- Step 6: Calibration sweep ----
    logger.info("Step 5: Running calibration sweep ...")
    sweep_results = run_calibration_sweep(
        synth_df, cfg.bucket_edges, synthesis_dir,
        city_code=city_code, city_name=cfg.city_name,
        kalshi_ticker=cfg.kalshi_ticker,
    )

    # ---- Step 7: Save training curves ----
    logger.info("Step 6: Saving training curves ...")
    history = train_result["history"]
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_maes = [h["val_mae"] for h in history]

        axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
        axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (Gaussian NLL)")
        axes[0].set_title("Training Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_maes, label="Val MAE", linewidth=1.5, color="#2ca02c")
        best_idx = int(np.argmin(val_maes))
        axes[1].axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.7)
        axes[1].scatter([epochs[best_idx]], [val_maes[best_idx]],
                        color="red", zorder=5, s=50)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (deg F)")
        axes[1].set_title("Validation MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"{city_code.upper()} Synthesis Model: Training Curves", fontsize=14, fontweight="bold")
        fig.tight_layout()
        curves_path = os.path.join(synthesis_dir, "training_curves.png")
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved training curves to %s", curves_path)

    # ---- Final summary ----
    summary = sweep_results.get("summary", {})
    logger.info("=" * 70)
    logger.info("%s Synthesis Calibration Sweep Complete", city_code.upper())
    logger.info("  Best method:      %s", summary.get("best_method", "N/A"))
    logger.info("  Best Brier:       %.4f", summary.get("best_brier", float("nan")))
    logger.info("  Synthesis MAE:    %.2f F", test_mae)
    logger.info("  Test dates:       %d", summary.get("n_test_dates", summary.get("n_test_days", 0)))
    logger.info("  Output dir:       %s", synthesis_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
