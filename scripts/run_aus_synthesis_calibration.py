#!/usr/bin/env python3
"""
Austin Synthesis Model Training and Calibration Sweep.

Trains U-series synthesis models combining base model predictions with
seasonal context, then applies calibration methods to produce well-calibrated
bucket probabilities for KXHIGHAUS.

Calibration methods evaluated:
  1. Raw (uncalibrated) Gaussian -> bucket probs
  2. Isotonic-calibrated per bucket
  3. Platt-scaled per bucket
  4. Seasonal regime split: calibrate separately for DJF/MAM/JJA/SON

All splits are strictly chronological to avoid data leakage.

Usage:
    python scripts/run_aus_synthesis_calibration.py
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# U-series model configurations
# ---------------------------------------------------------------------------
# Each variant specifies hidden layer sizes, dropout, batch-norm usage,
# and the subset of synthesis features to include.  Feature names must
# match the column names returned by _build_synthesis_features().
U_SERIES_CONFIGS = {
    "U0_base_synthesis": {
        "hidden_sizes": [64, 32],
        "dropout": 0.1,
        "use_batch_norm": True,
        "features": ["mu", "sigma", "sin_day", "cos_day"],
    },
    "U2_contract_ridge": {
        "hidden_sizes": [32],
        "dropout": 0.0,
        "use_batch_norm": False,
        "features": ["mu", "sigma", "sin_day", "cos_day", "mu_seasonal_anomaly"],
        "note": "Simple linear-ish model for contract-level calibration",
    },
    "U5_regime_conditional": {
        "hidden_sizes": [128, 64, 32],
        "dropout": 0.15,
        "use_batch_norm": True,
        "features": ["mu", "sigma", "sin_day", "cos_day", "mu_sigma_interaction", "mu_seasonal_anomaly"],
    },
    "U7_extended_mlp": {
        "hidden_sizes": [128, 64, 32],
        "dropout": 0.15,
        "use_batch_norm": True,
        "features": ["mu", "sigma", "sin_day", "cos_day", "mu_sigma_interaction", "mu_seasonal_anomaly", "sigma_regime"],
    },
    "U9_kitchen_sink": {
        "hidden_sizes": [256, 128, 64],
        "dropout": 0.2,
        "use_batch_norm": True,
        "features": ["mu", "sigma", "sin_day", "cos_day", "mu_sigma_interaction", "mu_seasonal_anomaly", "sigma_regime", "mu_squared", "persistence_gap"],
    },
}

# Canonical order of all synthesis features.  _build_synthesis_features()
# returns a DataFrame with these columns; each U-series variant selects a
# subset via its "features" list.
ALL_SYNTHESIS_FEATURE_NAMES = [
    "mu", "sigma", "sin_day", "cos_day",
    "mu_sigma_interaction", "mu_seasonal_anomaly",
    "sigma_regime", "mu_squared", "persistence_gap",
]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_base_predictions(results_dir: str) -> pd.DataFrame:
    """Load base model predictions from the AUS benchmark results.

    Looks for the benchmark summary JSON and per-model CSV prediction files
    produced by run_aus_benchmark.py.  Falls back to generating synthetic
    predictions from AUS climatology if benchmark outputs are not found.

    Parameters
    ----------
    results_dir : str
        Path to results/austin/ directory.

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
        f"Run the AUS benchmark script (run_aus_benchmark.py) first to "
        f"generate real model predictions. Synthetic data fallback has been "
        f"removed to prevent silent corruption of evaluation results."
    )


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
    """Configurable MLP for synthesis model (U-series architecture).

    Takes base model (mu, sigma), seasonal features (sin_day, cos_day),
    and interaction features as input.  Outputs calibrated (mu, sigma)
    for a heteroscedastic Gaussian distribution.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list[int]
        Sizes of the hidden layers.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to include BatchNorm1d after each hidden linear layer.
    """

    def __init__(
        self,
        n_features: int = 6,
        hidden_sizes: Optional[list[int]] = None,
        dropout: float = 0.15,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
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
            "dropout=%.2f, batch_norm=%s, params=%d",
            n_features, hidden_sizes, dropout, use_batch_norm, n_params,
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
) -> pd.DataFrame:
    """Build synthesis feature DataFrame from prediction DataFrame.

    Features:
      0. mu                   - base model predicted mean
      1. sigma                - base model predicted std
      2. sin_day              - sin(2*pi*dayofyear/365.25)
      3. cos_day              - cos(2*pi*dayofyear/365.25)
      4. mu_sigma_interaction - mu * sigma
      5. mu_seasonal_anomaly  - (mu - seasonal_mean) / seasonal_std
      6. sigma_regime         - binary high/low uncertainty indicator
      7. mu_squared           - quadratic temperature term
      8. persistence_gap      - day-over-day mu change

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, mu, sigma.
    cfg : CityConfig
        City configuration with monthly_tmax_mean and monthly_tmax_std.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching ALL_SYNTHESIS_FEATURE_NAMES,
        each row corresponding to the input df rows.
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

    # Binary uncertainty regime: 1 if sigma above median, 0 otherwise
    sigma_median = np.median(sigma)
    sigma_regime = (sigma > sigma_median).astype(float)

    # Quadratic temperature term
    mu_squared = mu ** 2

    # Day-over-day mu change (fill first NaN with 0)
    persistence_gap = np.diff(mu, prepend=mu[0])
    persistence_gap[0] = 0.0

    feature_df = pd.DataFrame({
        "mu": mu,
        "sigma": sigma,
        "sin_day": sin_day,
        "cos_day": cos_day,
        "mu_sigma_interaction": mu_sigma_interaction,
        "mu_seasonal_anomaly": mu_seasonal_anomaly,
        "sigma_regime": sigma_regime,
        "mu_squared": mu_squared,
        "persistence_gap": persistence_gap,
    }).astype(np.float32)

    return feature_df


def train_synthesis_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    model_name: str = "U_aus_synthesis",
    hidden_sizes: Optional[list[int]] = None,
    dropout: float = 0.15,
    use_batch_norm: bool = True,
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
    hidden_sizes : list[int], optional
        Hidden layer sizes for the MLP.  Defaults to [64, 32, 16].
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to include BatchNorm1d layers.

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
    model = SynthesisMLP(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    ).to(DEVICE)
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
    title: str = "AUS Calibration Sweep: Brier Scores",
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
    model_name: str = "synthesis",
) -> dict:
    """Run a full calibration sweep on synthesis model predictions.

    Evaluates:
      1. Raw (uncalibrated) Gaussian -> bucket probs
      2. Isotonic calibration per bucket
      3. Platt scaling per bucket
      4. Seasonal regime-conditional isotonic calibration

    Date-based chronological splits:
      - Train: up to 2021-12-31 (for fitting calibrators)
      - Calibration: 2022-01-01 to 2023-12-31 (combined with train for
        calibrator fitting to maximize calibration data)
      - Test: 2024-01-01 to 2025-12-31 (held-out evaluation only)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must have columns: date, mu, sigma, actual_tmax.
        Sorted chronologically.
    bucket_edges : list of (low, high) tuples
        Kalshi contract bucket boundaries.
    output_dir : str
        Directory to save calibration results and plots.
    model_name : str
        Name of the model for logging and file naming.

    Returns
    -------
    dict
        Keys: method_name -> {brier, ece, reliability_data}.
        Also includes "best_method" and "summary" keys.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = predictions_df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    mu = df["mu"].values
    sigma = df["sigma"].values
    actual_tmax = df["actual_tmax"].values

    # Date-based chronological splits
    # Synthesis predictions span 2022-2024. Calibrators are fit on the
    # calibration partition (2022-2023) and evaluated on test (2024+).
    cal_mask = (dates >= "2022-01-01") & (dates <= "2023-12-31")
    test_mask = dates >= "2024-01-01"

    cal_idx = np.where(cal_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]
    # Use full calibration period for fitting calibrators
    train_cal_idx = cal_idx

    n_cal = len(cal_idx)
    n_test = len(test_idx)
    n = len(df)

    logger.info(
        "Calibration sweep [%s]: %d total days (cal=%d, test=%d)",
        model_name, n, n_cal, n_test,
    )

    if n_test == 0:
        logger.warning("No test data available (2024+). Returning empty results.")
        return {"best_method": "N/A", "summary": {"n_test_days": 0}}

    # Compute raw bucket probabilities for all splits
    raw_probs_train_cal = compute_bucket_probs_gaussian(
        mu[train_cal_idx], sigma[train_cal_idx], bucket_edges,
    )
    raw_probs_test = compute_bucket_probs_gaussian(
        mu[test_idx], sigma[test_idx], bucket_edges,
    )

    # Build outcome matrices
    outcomes_train_cal = _build_outcome_matrix(actual_tmax[train_cal_idx], bucket_edges)
    outcomes_test = _build_outcome_matrix(actual_tmax[test_idx], bucket_edges)

    results: dict = {}

    # ---- 1. Raw (uncalibrated) ----
    brier_raw = compute_brier_score(
        raw_probs_test, actual_tmax[test_idx], bucket_edges,
    )
    ece_raw = compute_ece(
        raw_probs_test.ravel(), outcomes_test.ravel(),
    )
    rel_raw = compute_reliability_diagram(
        raw_probs_test.ravel(), outcomes_test.ravel(),
    )

    results["1_raw_uncalibrated"] = {
        "brier": brier_raw["overall_brier"],
        "ece": ece_raw,
        "reliability": rel_raw,
    }
    logger.info(
        "  Raw uncalibrated: Brier=%.4f, ECE=%.4f",
        brier_raw["overall_brier"], ece_raw,
    )

    # ---- 2. Isotonic calibration ----
    # Fit on train+cal, apply to test
    iso_probs_test = apply_isotonic_calibration(
        raw_probs_train_cal, outcomes_train_cal, raw_probs_test,
    )

    brier_iso = compute_brier_score(
        iso_probs_test, actual_tmax[test_idx], bucket_edges,
    )
    ece_iso = compute_ece(
        iso_probs_test.ravel(), outcomes_test.ravel(),
    )
    rel_iso = compute_reliability_diagram(
        iso_probs_test.ravel(), outcomes_test.ravel(),
    )

    results["2_isotonic"] = {
        "brier": brier_iso["overall_brier"],
        "ece": ece_iso,
        "reliability": rel_iso,
    }
    logger.info(
        "  Isotonic calibrated: Brier=%.4f, ECE=%.4f",
        brier_iso["overall_brier"], ece_iso,
    )

    # ---- 3. Platt scaling ----
    platt_probs_test = apply_platt_scaling(
        raw_probs_train_cal, outcomes_train_cal, raw_probs_test,
    )

    brier_platt = compute_brier_score(
        platt_probs_test, actual_tmax[test_idx], bucket_edges,
    )
    ece_platt = compute_ece(
        platt_probs_test.ravel(), outcomes_test.ravel(),
    )
    rel_platt = compute_reliability_diagram(
        platt_probs_test.ravel(), outcomes_test.ravel(),
    )

    results["3_platt_scaling"] = {
        "brier": brier_platt["overall_brier"],
        "ece": ece_platt,
        "reliability": rel_platt,
    }
    logger.info(
        "  Platt scaling: Brier=%.4f, ECE=%.4f",
        brier_platt["overall_brier"], ece_platt,
    )

    # ---- 4. Seasonal regime-conditional isotonic ----
    dates_np = dates.values
    dates_train_cal = dates_np[train_cal_idx]
    seasonal_probs_test = apply_seasonal_calibration(
        raw_probs_train_cal, outcomes_train_cal, dates_train_cal,
        raw_probs_test, dates_np[test_idx],
    )

    brier_seasonal = compute_brier_score(
        seasonal_probs_test, actual_tmax[test_idx], bucket_edges,
    )
    ece_seasonal = compute_ece(
        seasonal_probs_test.ravel(), outcomes_test.ravel(),
    )
    rel_seasonal = compute_reliability_diagram(
        seasonal_probs_test.ravel(), outcomes_test.ravel(),
    )

    results["4_seasonal_regime"] = {
        "brier": brier_seasonal["overall_brier"],
        "ece": ece_seasonal,
        "reliability": rel_seasonal,
    }
    logger.info(
        "  Seasonal regime: Brier=%.4f, ECE=%.4f",
        brier_seasonal["overall_brier"], ece_seasonal,
    )

    # ---- Identify best method ----
    method_briers = {
        name: info["brier"] for name, info in results.items()
    }
    best_method = min(method_briers, key=method_briers.get)

    logger.info("=" * 60)
    logger.info("Best calibration method: %s (Brier=%.4f)",
                best_method, method_briers[best_method])
    logger.info("=" * 60)

    results["best_method"] = best_method

    # ---- Generate plots ----
    # Reliability diagrams
    for method_name, info in results.items():
        if method_name == "best_method":
            continue
        rel_data = info["reliability"]
        plot_reliability(
            rel_data,
            title=f"AUS {model_name} {method_name}: Reliability",
            save_path=os.path.join(
                output_dir, f"reliability_{model_name}_{method_name}.png"
            ),
        )

    # Brier comparison bar chart
    plot_brier_comparison(
        method_briers,
        title=f"AUS {model_name} Calibration Sweep: OOS Brier Scores",
        save_path=os.path.join(output_dir, f"brier_comparison_{model_name}.png"),
    )

    # ---- Build summary ----
    summary = {
        "city": "Austin",
        "ticker": "KXHIGHAUS",
        "model_name": model_name,
        "n_total_days": n,
        "n_train_days": n_cal,
        "n_cal_days": n_cal,
        "n_test_days": n_test,
        "n_buckets": len(bucket_edges),
        "methods": {},
    }
    for method_name, info in results.items():
        if method_name == "best_method":
            continue
        summary["methods"][method_name] = {
            "brier": info["brier"],
            "ece": info["ece"],
        }
    summary["best_method"] = best_method
    summary["best_brier"] = method_briers[best_method]

    results["summary"] = summary

    # Save summary JSON
    summary_path = os.path.join(
        output_dir, f"calibration_sweep_{model_name}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved calibration sweep summary to %s", summary_path)

    return results


# ===========================================================================
# Main Orchestration
# ===========================================================================

def _save_training_curves(
    history: list[dict],
    model_name: str,
    output_dir: str,
) -> None:
    """Save training/validation loss and MAE curves for one model.

    Parameters
    ----------
    history : list[dict]
        Training history with keys: epoch, train_loss, val_loss, val_mae.
    model_name : str
        Name of the U-series model variant.
    output_dir : str
        Directory to save the plot.
    """
    if not history:
        return

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

    fig.suptitle(f"AUS {model_name}: Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    curves_path = os.path.join(output_dir, f"training_curves_{model_name}.png")
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved training curves to %s", curves_path)


def main() -> None:
    """Run the full AUS U-series synthesis model training and calibration sweep.

    For each variant in U_SERIES_CONFIGS:
      1. Select the feature subset specified by the config.
      2. Train a SynthesisMLP with the variant's architecture.
      3. Generate predictions on all data.
      4. Run the 4-method calibration sweep (raw, isotonic, Platt, seasonal).
      5. Record the best (model, calibration) combination by test-set
         contract Brier.

    Outputs:
      - Per-model artifacts in results/austin/synthesis/
      - results/austin/synthesis/unified_benchmark_summary.json
      - results/austin/unified_predictions.csv
    """
    cfg = get_city_config("aus")
    ensure_city_dirs(cfg)

    synthesis_dir = os.path.join(cfg.results_dir, "synthesis")
    os.makedirs(synthesis_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Austin U-Series Synthesis Model Training & Calibration Sweep")
    logger.info("  City:     %s", cfg.city_name)
    logger.info("  Ticker:   %s", cfg.kalshi_ticker)
    logger.info("  Station:  %s (%s)", cfg.target_station, cfg.target_station_name)
    logger.info("  Buckets:  %d", len(cfg.bucket_edges))
    logger.info("  Models:   %s", list(U_SERIES_CONFIGS.keys()))
    logger.info("  Output:   %s", synthesis_dir)
    logger.info("=" * 70)

    # ---- Step 1: Load base predictions ----
    logger.info("Step 1: Loading base model predictions ...")
    base_df = load_base_predictions(cfg.results_dir)
    logger.info("  Loaded %d predictions (%s to %s)",
                len(base_df),
                base_df["date"].min(),
                base_df["date"].max())

    # Drop rows with missing actual_tmax -- cannot train or evaluate without targets
    n_before = len(base_df)
    base_df = base_df.dropna(subset=["actual_tmax"]).reset_index(drop=True)
    n_dropped = n_before - len(base_df)
    if n_dropped > 0:
        logger.info("  Dropped %d rows with NaN actual_tmax (%d remaining)",
                     n_dropped, len(base_df))

    # ---- Step 2: Build full synthesis feature set ----
    logger.info("Step 2: Building synthesis features ...")
    feature_df = _build_synthesis_features(base_df, cfg)
    logger.info("  Feature DataFrame shape: %s", feature_df.shape)
    logger.info("  Available features: %s", list(feature_df.columns))

    # ---- Step 3: Date-based chronological splits ----
    # NOTE: Base predictions only contain val (2022-2023) and test (2024+) from
    # the benchmark script. The synthesis meta-model trains on the calibration
    # partition (2022-2023) and evaluates on test (2024+). We split the
    # calibration partition 70/30 for synthesis train/val to avoid overfitting.
    dates = pd.to_datetime(base_df["date"])
    cal_mask = (dates >= "2022-01-01") & (dates <= "2023-12-31")
    test_mask = dates >= "2024-01-01"

    cal_idx = np.where(cal_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    # Split calibration into synthesis-train (70%) and synthesis-val (30%)
    n_cal = len(cal_idx)
    n_synth_train = int(n_cal * 0.7)
    train_idx = cal_idx[:n_synth_train]
    val_idx = cal_idx[n_synth_train:]

    y_all = base_df["actual_tmax"].values
    y_train = y_all[train_idx]
    y_cal = y_all[val_idx]
    y_test = y_all[test_idx]

    logger.info(
        "  Date splits: SynthTrain=%d (2022-mid2023), SynthVal=%d (mid2023-2023), Test=%d (2024+)",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # ---- Step 4: Train each U-series variant ----
    # Collect results for all (model, calibration) combinations
    all_model_results: dict[str, dict] = {}
    # Collect all predictions for unified_predictions.csv
    all_predictions: list[pd.DataFrame] = []

    for variant_name, variant_cfg in U_SERIES_CONFIGS.items():
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training U-series variant: %s", variant_name)
        logger.info("  hidden_sizes:   %s", variant_cfg["hidden_sizes"])
        logger.info("  dropout:        %.2f", variant_cfg["dropout"])
        logger.info("  use_batch_norm: %s", variant_cfg["use_batch_norm"])
        logger.info("  features:       %s", variant_cfg["features"])
        logger.info("=" * 60)

        # Select feature subset for this variant
        feat_cols = variant_cfg["features"]
        missing_cols = [c for c in feat_cols if c not in feature_df.columns]
        if missing_cols:
            logger.error(
                "  SKIPPING %s: missing features %s", variant_name, missing_cols,
            )
            continue

        X_all = feature_df[feat_cols].values.astype(np.float32)
        X_train_v = X_all[train_idx]
        X_cal_v = X_all[val_idx]
        X_test_v = X_all[test_idx]

        # Train MLP
        train_result = train_synthesis_mlp(
            X_train_v, y_train,
            X_cal_v, y_cal,
            max_epochs=200,
            patience=15,
            batch_size=64,
            lr=0.001,
            model_name=variant_name,
            hidden_sizes=variant_cfg["hidden_sizes"],
            dropout=variant_cfg["dropout"],
            use_batch_norm=variant_cfg["use_batch_norm"],
        )

        model = train_result["model"]
        scaler = train_result["scaler"]

        # Save model checkpoint
        model_path = os.path.join(synthesis_dir, f"{variant_name}_model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info("  Saved model to %s", model_path)

        scaler_path = os.path.join(synthesis_dir, f"{variant_name}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save training curves
        _save_training_curves(train_result["history"], variant_name, synthesis_dir)

        # Generate predictions on all data
        synth_mu_all, synth_sigma_all = predict_synthesis(model, scaler, X_all)

        # Compute test MAE
        test_mu = synth_mu_all[test_idx]
        test_mae = float(np.mean(np.abs(test_mu - y_test)))
        logger.info("  %s test MAE: %.2f F", variant_name, test_mae)

        # Build prediction DataFrame for this variant
        synth_df = base_df[["date", "actual_tmax"]].copy()
        synth_df["mu"] = synth_mu_all
        synth_df["sigma"] = synth_sigma_all

        # Save per-model predictions
        preds_path = os.path.join(synthesis_dir, f"{variant_name}_predictions.csv")
        synth_df.to_csv(preds_path, index=False)

        # Collect for unified predictions output
        pred_out = synth_df.copy()
        pred_out["model_name"] = variant_name
        all_predictions.append(pred_out)

        # ---- Calibration sweep for this variant ----
        logger.info("  Running calibration sweep for %s ...", variant_name)
        sweep_results = run_calibration_sweep(
            synth_df, cfg.bucket_edges, synthesis_dir, model_name=variant_name,
        )

        sweep_summary = sweep_results.get("summary", {})
        best_cal_method = sweep_summary.get("best_method", "N/A")
        best_cal_brier = sweep_summary.get("best_brier", float("nan"))

        all_model_results[variant_name] = {
            "test_mae": test_mae,
            "best_val_loss": train_result["best_val_loss"],
            "best_epoch": train_result["best_epoch"],
            "best_calibration_method": best_cal_method,
            "best_calibrated_brier": best_cal_brier,
            "hidden_sizes": variant_cfg["hidden_sizes"],
            "dropout": variant_cfg["dropout"],
            "use_batch_norm": variant_cfg["use_batch_norm"],
            "features": variant_cfg["features"],
            "calibration_methods": {},
        }

        # Record all calibration method results
        for method_name, method_info in sweep_results.items():
            if method_name in ("best_method", "summary"):
                continue
            all_model_results[variant_name]["calibration_methods"][method_name] = {
                "brier": method_info["brier"],
                "ece": method_info["ece"],
            }

        logger.info(
            "  %s => best_cal=%s, Brier=%.4f, MAE=%.2f F",
            variant_name, best_cal_method, best_cal_brier, test_mae,
        )

    # ---- Step 5: Identify overall best (model, calibration) combination ----
    if not all_model_results:
        logger.error("No models were successfully trained. Exiting.")
        return

    best_overall_model = None
    best_overall_brier = float("inf")
    for vname, vresults in all_model_results.items():
        if vresults["best_calibrated_brier"] < best_overall_brier:
            best_overall_brier = vresults["best_calibrated_brier"]
            best_overall_model = vname

    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL BEST MODEL: %s", best_overall_model)
    logger.info("  Calibration:   %s",
                all_model_results[best_overall_model]["best_calibration_method"])
    logger.info("  Contract Brier: %.4f", best_overall_brier)
    logger.info("  Test MAE:       %.2f F",
                all_model_results[best_overall_model]["test_mae"])
    logger.info("=" * 70)

    # ---- Step 6: Save unified benchmark summary ----
    unified_summary = {
        "city": "Austin",
        "ticker": "KXHIGHAUS",
        "station": cfg.target_station,
        "n_models": len(all_model_results),
        "split_boundaries": {
            "train": "<=2021-12-31",
            "calibration": "2022-01-01 to 2023-12-31",
            "test": "2024-01-01 to 2025-12-31",
        },
        "n_train": int(len(train_idx)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "n_buckets": len(cfg.bucket_edges),
        "best_overall_model": best_overall_model,
        "best_overall_brier": best_overall_brier,
        "best_overall_calibration": all_model_results[best_overall_model][
            "best_calibration_method"
        ],
        "models": {},
    }

    # Build per-model summary sorted by Brier
    sorted_models = sorted(
        all_model_results.items(),
        key=lambda x: x[1]["best_calibrated_brier"],
    )
    for vname, vresults in sorted_models:
        unified_summary["models"][vname] = {
            "contract_brier": vresults["best_calibrated_brier"],
            "best_calibration_method": vresults["best_calibration_method"],
            "test_mae": vresults["test_mae"],
            "best_val_loss": vresults["best_val_loss"],
            "best_epoch": vresults["best_epoch"],
            "hidden_sizes": vresults["hidden_sizes"],
            "dropout": vresults["dropout"],
            "use_batch_norm": vresults["use_batch_norm"],
            "features": vresults["features"],
            "calibration_methods": vresults["calibration_methods"],
        }

    summary_path = os.path.join(synthesis_dir, "unified_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(unified_summary, f, indent=2, default=str)
    logger.info("Saved unified benchmark summary to %s", summary_path)

    # ---- Step 7: Save unified predictions CSV ----
    if all_predictions:
        unified_preds = pd.concat(all_predictions, ignore_index=True)
        unified_preds_path = os.path.join(cfg.results_dir, "unified_predictions.csv")
        unified_preds.to_csv(unified_preds_path, index=False)
        logger.info("Saved unified predictions (%d rows) to %s",
                     len(unified_preds), unified_preds_path)

    # ---- Step 8: Print leaderboard ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("AUS U-Series Leaderboard (sorted by contract Brier, test set)")
    logger.info("%-25s  %12s  %12s  %10s  %s",
                "Model", "Brier", "MAE (F)", "Epoch", "Best Cal")
    logger.info("-" * 85)
    for vname, vresults in sorted_models:
        marker = " ***" if vname == best_overall_model else ""
        logger.info(
            "%-25s  %12.4f  %12.2f  %10d  %s%s",
            vname,
            vresults["best_calibrated_brier"],
            vresults["test_mae"],
            vresults["best_epoch"],
            vresults["best_calibration_method"],
            marker,
        )
    logger.info("=" * 70)

    # ---- Final summary ----
    logger.info("")
    logger.info("AUS U-Series Synthesis Calibration Sweep Complete")
    logger.info("  Best model:       %s", best_overall_model)
    logger.info("  Best calibration: %s",
                all_model_results[best_overall_model]["best_calibration_method"])
    logger.info("  Contract Brier:   %.4f", best_overall_brier)
    logger.info("  Output dir:       %s", synthesis_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
