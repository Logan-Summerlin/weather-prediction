#!/usr/bin/env python3
"""
Austin Model Benchmark Script.

Trains and evaluates a suite of models for Austin (KXHIGHAUS)
temperature prediction, computing Brier scores on held-out OOS data
using AUS bucket definitions from city_config.

Models benchmarked:
  - Persistence baseline   (yesterday's AUS TMAX)
  - Climatological baseline (day-of-year average from training set)
  - Ridge regression        (L2-regularized linear model)
  - Flat feedforward NN     (heteroscedastic Gaussian output: mu, sigma)

All models produce (or are converted to) distributional Gaussian output
(mu, sigma) which is mapped to Kalshi contract-level probabilities
via scipy.stats.norm.cdf.  Contract Brier score is computed over real
Kalshi contract rows only (not all bucket-days).

Results are saved to results/austin/.

Usage:
    python scripts/run_aus_benchmark.py
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
from scipy.stats import norm
from sklearn.linear_model import Ridge

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
from src.contract_brier import contract_brier_score
import config_austin as city_config

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


# ===========================================================================
# Data Loading
# ===========================================================================

def load_processed_aus_data(processed_dir: str) -> tuple:
    """Load preprocessed Austin CSV files from disk.

    Parameters
    ----------
    processed_dir : str
        Directory containing features_train.csv, target_train.csv, etc.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_dir}\n"
            "Run scripts/run_aus_preprocessing.py first."
        )

    X_train = pd.read_csv(
        os.path.join(processed_dir, "features_train.csv"),
        index_col=0, parse_dates=True,
    )
    X_val = pd.read_csv(
        os.path.join(processed_dir, "features_val.csv"),
        index_col=0, parse_dates=True,
    )
    X_test = pd.read_csv(
        os.path.join(processed_dir, "features_test.csv"),
        index_col=0, parse_dates=True,
    )
    y_train = pd.read_csv(
        os.path.join(processed_dir, "target_train.csv"),
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    y_val = pd.read_csv(
        os.path.join(processed_dir, "target_val.csv"),
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    y_test = pd.read_csv(
        os.path.join(processed_dir, "target_test.csv"),
        index_col=0, parse_dates=True,
    ).iloc[:, 0]

    # Ensure target series have a consistent name
    for s in (y_train, y_val, y_test):
        s.name = "AUS_TMAX"

    # Drop columns that are entirely NaN, then fill remaining NaN with 0
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping %d all-NaN columns: %s", len(all_nan_cols), all_nan_cols)
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    remaining_nans = X_train.isna().sum().sum()
    if remaining_nans > 0:
        logger.info("Filling %d remaining NaN values with 0", remaining_nans)
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)

    logger.info("Loaded AUS processed data from %s", processed_dir)
    logger.info("  X_train: %s, X_val: %s, X_test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ===========================================================================
# Bucket Probability Conversion
# ===========================================================================

def gaussian_to_bucket_probs(
    mu: np.ndarray,
    sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Convert Gaussian (mu, sigma) predictions to bucket probabilities.

    For each day, computes P(low <= TMAX < high) = CDF(high) - CDF(low)
    for each bucket.  Open-ended sentinels (-999 / 999) are treated as
    -infinity / +infinity.

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
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    probs = np.zeros((n_days, n_buckets))

    for b, (lo, hi) in enumerate(bucket_edges):
        cdf_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        cdf_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        probs[:, b] = np.clip(cdf_hi - cdf_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # Normalize rows to sum to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums

    return probs


# ===========================================================================
# Kalshi Contract-Level Evaluation
# ===========================================================================

SEASON_MAP_MONTH = {12: "DJF", 1: "DJF", 2: "DJF",
                    3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA",
                    9: "SON", 10: "SON", 11: "SON"}


def load_kalshi_data_aus(
    dates: pd.DatetimeIndex,
    actual_tmax: np.ndarray,
    bucket_edges: list,
) -> pd.DataFrame:
    """Load or generate Kalshi-format contract data for Austin.

    If real Kalshi settlement data exists, loads it. Otherwise, generates
    synthetic contract rows from the actual TMAX observations and bucket
    definitions, which is needed for initial city pipeline runs before
    real Kalshi market data becomes available.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates for which to generate contract rows.
    actual_tmax : np.ndarray
        Observed TMAX values for those dates.
    bucket_edges : list
        Kalshi bucket edge definitions.

    Returns
    -------
    pd.DataFrame
        Contract-format DataFrame with columns: date, ticker,
        threshold_low, threshold_high, direction, actual_outcome,
        actual_tmax.
    """
    settlement_path = PROJECT_ROOT / "data" / "real_kalshi_aus_all.csv"

    if settlement_path.exists():
        settled = pd.read_csv(settlement_path)
        settled["date"] = pd.to_datetime(settled["date"]).dt.strftime("%Y-%m-%d")
        presettlement_path = PROJECT_ROOT / "data" / "kalshi_presettlement_aus.csv"
        if presettlement_path.exists():
            pre = pd.read_csv(presettlement_path)
            pre["date"] = pd.to_datetime(pre["date"]).dt.strftime("%Y-%m-%d")
            merged = settled.merge(
                pre[["date", "ticker", "presettlement_prob", "bid_cents",
                     "ask_cents", "volume", "open_interest", "snapshot_time_utc"]],
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

    # Generate synthetic contract rows from actual outcomes
    logger.info("No real Kalshi data for Austin — generating synthetic contract rows")
    rows = []
    for i, (date, tmax) in enumerate(zip(dates, actual_tmax)):
        if np.isnan(tmax):
            continue
        ds = date.strftime("%Y-%m-%d")
        for b, (lo, hi) in enumerate(bucket_edges):
            # Determine direction and thresholds
            if lo <= -900:
                direction = "below"
                th_low = None
                th_high = hi
                outcome = 1 if tmax < hi else 0
            elif hi >= 900:
                direction = "above"
                th_low = lo
                th_high = None
                outcome = 1 if tmax >= lo else 0
            else:
                direction = "between"
                th_low = lo
                th_high = hi
                outcome = 1 if lo <= tmax < hi else 0

            rows.append({
                "date": ds,
                "ticker": f"KXHIGHAUS-{ds}-B{b}",
                "threshold_low": th_low if th_low is not None else float("nan"),
                "threshold_high": th_high if th_high is not None else float("nan"),
                "direction": direction,
                "actual_outcome": outcome,
                "actual_tmax": tmax,
            })

    df = pd.DataFrame(rows)
    logger.info("Generated %d synthetic AUS contract rows across %d dates",
                len(df), df["date"].nunique())
    return df


def build_contract_dataset(
    kalshi_df: pd.DataFrame,
    mu_by_date: dict,
    sigma_by_date: dict,
) -> pd.DataFrame:
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


def contract_brier(probs, outcomes):
    """Compute contract-level Brier score."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(o))
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((p[valid] - o[valid]) ** 2))


def compute_contract_seasonal_brier(contract_df, prob_col="model_prob"):
    """Compute contract Brier per meteorological season."""
    df = contract_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["month"] = df["date_dt"].dt.month
    df["season"] = df["month"].map(SEASON_MAP_MONTH)
    probs = df[prob_col].values.astype(float)
    outcomes = df["actual_outcome"].values.astype(float)
    results = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = (df["season"] == s).values
        if mask.any():
            results[s] = contract_brier(probs[mask], outcomes[mask])
    return results


def compute_brier_score(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> dict:
    """Compute bucket-day Brier score (kept for Ridge alpha search only).

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
        Dictionary with keys: overall_brier, n_days, n_buckets.
    """
    n_days, n_buckets = bucket_probs.shape
    assert len(actual_tmax) == n_days

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

    return {
        "overall_brier": overall_brier,
        "n_days": n_days,
        "n_buckets": n_buckets,
    }


# ===========================================================================
# Baseline Models
# ===========================================================================

def run_persistence_baseline(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Persistence baseline: mu = yesterday's TMAX, sigma from training residuals.

    Parameters
    ----------
    y_train : pd.Series
        Training target values.
    y_val : pd.Series
        Validation target values.
    y_test : pd.Series
        Test target values.

    Returns
    -------
    dict
        Dictionary with mu, sigma arrays for val and test sets.
    """
    # Combine all target data in order for lag-1 computation
    all_y = pd.concat([y_train, y_val, y_test])

    # Persistence: predict yesterday's value
    y_prev = all_y.shift(1)

    # Estimate sigma from training set residuals
    train_mask = all_y.index.isin(y_train.index)
    train_resid = all_y[train_mask] - y_prev[train_mask]
    sigma_train = float(train_resid.dropna().std())
    if sigma_train < 1.0:
        sigma_train = 5.0  # fallback

    # Extract val/test predictions
    val_mask = all_y.index.isin(y_val.index)
    test_mask = all_y.index.isin(y_test.index)

    mu_val = y_prev[val_mask].values
    mu_test = y_prev[test_mask].values

    # Handle first-day NaN: use training mean as fallback
    train_mean = float(y_train.mean())
    mu_val = np.where(np.isnan(mu_val), train_mean, mu_val)
    mu_test = np.where(np.isnan(mu_test), train_mean, mu_test)

    return {
        "mu_val": mu_val,
        "sigma_val": np.full_like(mu_val, sigma_train),
        "mu_test": mu_test,
        "sigma_test": np.full_like(mu_test, sigma_train),
        "sigma_estimate": sigma_train,
    }


def run_climatology_baseline(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Climatological baseline: day-of-year average and std from training set.

    Parameters
    ----------
    y_train : pd.Series
        Training target values with DatetimeIndex.
    y_val : pd.Series
        Validation target values with DatetimeIndex.
    y_test : pd.Series
        Test target values with DatetimeIndex.

    Returns
    -------
    dict
        Dictionary with mu, sigma arrays for val and test sets.
    """
    # Compute day-of-year climatology from training set
    doy = y_train.index.dayofyear
    clim_mean = y_train.groupby(doy).mean()
    clim_std = y_train.groupby(doy).std()

    # Fill any missing DOYs (e.g., Feb 29)
    all_doys = np.arange(1, 367)
    clim_mean = clim_mean.reindex(all_doys).interpolate(method="linear").bfill().ffill()
    clim_std = clim_std.reindex(all_doys).interpolate(method="linear").bfill().ffill()

    # Minimum sigma floor
    clim_std = clim_std.clip(lower=3.0)

    # Map to val/test
    mu_val = clim_mean.reindex(y_val.index.dayofyear).values
    sigma_val = clim_std.reindex(y_val.index.dayofyear).values
    mu_test = clim_mean.reindex(y_test.index.dayofyear).values
    sigma_test = clim_std.reindex(y_test.index.dayofyear).values

    return {
        "mu_val": mu_val,
        "sigma_val": sigma_val,
        "mu_test": mu_test,
        "sigma_test": sigma_test,
    }


def run_ridge_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 1.0,
) -> dict:
    """Ridge regression baseline with Gaussian residual-based sigma.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Scaled feature matrices.
    y_train, y_val, y_test : pd.Series
        Target values.
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    dict
        Dictionary with mu, sigma arrays for val and test sets.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train.values, y_train.values)

    mu_train_pred = model.predict(X_train.values)
    mu_val = model.predict(X_val.values)
    mu_test = model.predict(X_test.values)

    # Estimate sigma from training residuals
    train_residuals = y_train.values - mu_train_pred
    sigma_train = float(np.std(train_residuals))
    if sigma_train < 1.0:
        sigma_train = 5.0

    # Compute MAE for logging
    val_mae = float(np.mean(np.abs(y_val.values - mu_val)))
    test_mae = float(np.mean(np.abs(y_test.values - mu_test)))
    logger.info("Ridge (alpha=%.1f): val MAE=%.2f F, test MAE=%.2f F, sigma=%.2f",
                alpha, val_mae, test_mae, sigma_train)

    return {
        "mu_val": mu_val,
        "sigma_val": np.full_like(mu_val, sigma_train),
        "mu_test": mu_test,
        "sigma_test": np.full_like(mu_test, sigma_train),
        "sigma_estimate": sigma_train,
        "val_mae": val_mae,
        "test_mae": test_mae,
    }


# ===========================================================================
# Heteroscedastic Feedforward NN
# ===========================================================================

class HeteroscedasticNet(nn.Module):
    """Feedforward NN with heteroscedastic Gaussian output (mu, log_sigma).

    Architecture:
        Input(n_features)
          -> [Linear(h_i), ReLU, Dropout(p)] x len(hidden_sizes)
          -> mu_head:  Linear -> 1  (no activation)
          -> log_sigma_head: Linear -> 1  (clamped, then exp)

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list of int
        Widths of the hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

        self.n_features = n_features
        self.hidden_sizes = hidden_sizes

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "HeteroscedasticNet: n_features=%d, hidden=%s, dropout=%.2f, params=%d",
            n_features, hidden_sizes, dropout, total_params,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n_features).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (mu, sigma) each of shape (batch, 1).
        """
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        log_sigma = log_sigma.clamp(min=-5.0, max=4.0)
        sigma = torch.exp(log_sigma)
        return mu, sigma


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian predictions.

    NLL = 0.5 * [log(2*pi*sigma^2) + ((y - mu) / sigma)^2]

    Parameters
    ----------
    mu : torch.Tensor
        Predicted means, shape (batch, 1).
    sigma : torch.Tensor
        Predicted std devs (positive), shape (batch, 1).
    target : torch.Tensor
        Observed values, shape (batch, 1).

    Returns
    -------
    torch.Tensor
        Scalar mean NLL loss.
    """
    variance = sigma ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * variance) + ((target - mu) ** 2) / variance)
    return nll.mean()


def train_heteroscedastic_nn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hidden_sizes: list[int] | None = None,
    dropout: float = 0.1,
    lr: float = 0.001,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
) -> dict:
    """Train a heteroscedastic feedforward NN and return distributional predictions.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Scaled feature matrices.
    y_train, y_val, y_test : pd.Series
        Target values (unscaled deg F).
    hidden_sizes : list of int, optional
        Hidden layer widths. Default: [128, 64].
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    batch_size : int
        Training batch size.

    Returns
    -------
    dict
        Dictionary with mu/sigma arrays for val and test sets,
        plus training history.
    """
    n_features = X_train.shape[1]

    # Create model
    model = HeteroscedasticNet(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # Create DataLoaders
    def _to_loader(X, y, shuffle=False):
        X_t = torch.tensor(X.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(X_train, y_train, shuffle=True)
    val_loader = _to_loader(X_val, y_val, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    best_state = None

    logger.info("Training HeteroscedasticNet (max_epochs=%d, patience=%d)", max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            mu, sigma = model(X_batch)
            loss = gaussian_nll_loss(mu, sigma, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # --- Validate ---
        model.eval()
        val_losses = []
        val_mus = []
        val_sigmas = []
        val_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                mu, sigma = model(X_batch)
                loss = gaussian_nll_loss(mu, sigma, y_batch)
                val_losses.append(loss.item())
                val_mus.append(mu.cpu().numpy())
                val_sigmas.append(sigma.cpu().numpy())
                val_actuals.append(y_batch.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_mu_arr = np.concatenate(val_mus).ravel()
        val_actual_arr = np.concatenate(val_actuals).ravel()
        val_mae = float(np.mean(np.abs(val_actual_arr - val_mu_arr)))

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        history.append({
            "epoch": epoch,
            "train_nll": avg_train_loss,
            "val_nll": avg_val_loss,
            "val_mae": val_mae,
            "lr": current_lr,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch <= 5:
                logger.info(
                    "Epoch %3d | train NLL: %.4f | val NLL: %.4f | val MAE: %.2f F | * BEST *",
                    epoch, avg_train_loss, avg_val_loss, val_mae,
                )
        else:
            epochs_no_improve += 1
            if epoch % 20 == 0:
                logger.info(
                    "Epoch %3d | train NLL: %.4f | val NLL: %.4f | val MAE: %.2f F | no improv (%d/%d)",
                    epoch, avg_train_loss, avg_val_loss, val_mae, epochs_no_improve, patience,
                )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (best epoch: %d)", epoch, best_epoch)
            break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best epoch: %d, best val NLL: %.4f", best_epoch, best_val_loss)

    # --- Predict on val and test ---
    model.eval()

    def _predict(X_df):
        X_t = torch.tensor(X_df.values, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            mu, sigma = model(X_t)
        return mu.cpu().numpy().ravel(), sigma.cpu().numpy().ravel()

    mu_val, sigma_val = _predict(X_val)
    mu_test, sigma_test = _predict(X_test)

    val_mae = float(np.mean(np.abs(y_val.values - mu_val)))
    test_mae = float(np.mean(np.abs(y_test.values - mu_test)))
    logger.info("HeteroscedasticNet: val MAE=%.2f F, test MAE=%.2f F", val_mae, test_mae)

    return {
        "mu_val": mu_val,
        "sigma_val": sigma_val,
        "mu_test": mu_test,
        "sigma_test": sigma_test,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_loss,
        "history": history,
        "model": model,
    }


# ===========================================================================
# Seasonal Breakdown
# ===========================================================================

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def compute_seasonal_brier(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    dates: pd.DatetimeIndex,
    bucket_edges: list[tuple[float, float]],
) -> dict[str, float]:
    """Compute bucket-day Brier score per season (kept for backward compat)."""
    months = dates.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    results = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == season
        if not np.any(mask):
            continue
        score = compute_brier_score(
            bucket_probs[mask], actual_tmax[mask], bucket_edges
        )
        results[season] = score["overall_brier"]
    return results


# ===========================================================================
# Results Visualization
# ===========================================================================

def plot_brier_comparison(
    results: dict[str, dict],
    save_path: str,
) -> None:
    """Create a bar chart comparing model Brier scores.

    Parameters
    ----------
    results : dict[str, dict]
        Model name -> result dict (must have 'test_brier' key).
    save_path : str
        Path to save the figure.
    """
    models = list(results.keys())
    briers = [results[m]["test_brier"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, briers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Contract Brier Score (lower is better)")
    ax.set_title("Austin Model Benchmark: Contract Brier Scores")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Brier comparison chart to %s", save_path)


def plot_seasonal_brier(
    seasonal_results: dict[str, dict[str, float]],
    save_path: str,
) -> None:
    """Create a grouped bar chart of seasonal Brier scores by model.

    Parameters
    ----------
    seasonal_results : dict[str, dict[str, float]]
        Model name -> {season: brier_score}.
    save_path : str
        Path to save the figure.
    """
    models = list(seasonal_results.keys())
    seasons = ["DJF", "MAM", "JJA", "SON"]
    n_models = len(models)
    n_seasons = len(seasons)
    x = np.arange(n_seasons)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(n_models))

    for i, model_name in enumerate(models):
        vals = [seasonal_results[model_name].get(s, 0.0) for s in seasons]
        ax.bar(x + i * width, vals, width, label=model_name, color=colors[i],
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Contract Brier Score")
    ax.set_title("Austin Benchmark: Seasonal Contract Brier Scores")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved seasonal Brier chart to %s", save_path)


# ===========================================================================
# Main Benchmark
# ===========================================================================

def main():
    """Run the full Austin model benchmark with contract-level Brier."""
    logger.info("=" * 70)
    logger.info("Austin Model Benchmark (KXHIGHAUS) — Contract Brier")
    logger.info("=" * 70)

    # --- Setup ---
    chi = get_city_config("aus")
    ensure_city_dirs(chi)

    processed_dir = os.path.join(chi.data_dir, "processed")
    results_dir = chi.results_dir
    os.makedirs(results_dir, exist_ok=True)

    bucket_edges = chi.bucket_edges
    bucket_labels = chi.bucket_labels

    logger.info("Results directory: %s", results_dir)

    # --- Load data ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_aus_data(processed_dir)

    # Load Kalshi contract data for contract-level Brier
    # Combine val+test dates and actuals for contract generation
    all_eval_dates = y_val.index.append(y_test.index)
    all_eval_actuals = np.concatenate([y_val.values, y_test.values])
    kalshi = load_kalshi_data_aus(all_eval_dates, all_eval_actuals, bucket_edges)
    logger.info("Kalshi contract rows: %d, dates: %d",
                len(kalshi), kalshi["date"].nunique())

    test_actual = y_test.values
    test_dates = y_test.index
    val_actual = y_val.values

    # Helper: compute contract brier for a model's (mu, sigma) predictions
    def eval_contract_brier(mu_test, sigma_test, mu_val=None, sigma_val=None):
        date_mu = {}
        date_sigma = {}
        for i, d in enumerate(test_dates):
            ds = d.strftime("%Y-%m-%d")
            date_mu[ds] = mu_test[i]
            date_sigma[ds] = sigma_test[i]
        if mu_val is not None and sigma_val is not None:
            for i, d in enumerate(y_val.index):
                ds = d.strftime("%Y-%m-%d")
                date_mu[ds] = mu_val[i]
                date_sigma[ds] = sigma_val[i]
        cdf = build_contract_dataset(kalshi, date_mu, date_sigma)
        if len(cdf) == 0:
            return float("nan"), {}
        outcomes = cdf["actual_outcome"].values.astype(float)
        brier = contract_brier(cdf["model_prob"].values, outcomes)
        seasonal = compute_contract_seasonal_brier(cdf)
        return brier, seasonal

    all_results = {}
    seasonal_all = {}

    # ===================================================================
    # 1. Persistence Baseline
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 1: Persistence Baseline")
    logger.info("-" * 50)

    persist = run_persistence_baseline(y_train, y_val, y_test)
    persist_brier, persist_seasonal = eval_contract_brier(
        persist["mu_test"], persist["sigma_test"],
        persist["mu_val"], persist["sigma_val"])

    logger.info("Persistence: contract Brier=%.4f, sigma=%.2f",
                persist_brier, persist["sigma_estimate"])

    all_results["Persistence"] = {
        "test_brier": persist_brier,
    }
    seasonal_all["Persistence"] = persist_seasonal

    # ===================================================================
    # 2. Climatology Baseline
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 2: Climatology Baseline")
    logger.info("-" * 50)

    clim = run_climatology_baseline(y_train, y_val, y_test)
    clim_brier, clim_seasonal = eval_contract_brier(
        clim["mu_test"], clim["sigma_test"],
        clim["mu_val"], clim["sigma_val"])

    logger.info("Climatology: contract Brier=%.4f", clim_brier)

    all_results["Climatology"] = {
        "test_brier": clim_brier,
    }
    seasonal_all["Climatology"] = clim_seasonal

    # ===================================================================
    # 3. Ridge Regression
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 3: Ridge Regression")
    logger.info("-" * 50)

    # Try multiple alpha values, pick best on val bucket-day Brier
    best_ridge_result = None
    best_ridge_alpha = None
    best_ridge_val_brier = float("inf")

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        ridge_res = run_ridge_baseline(X_train, y_train, X_val, y_val, X_test, y_test, alpha=alpha)
        ridge_val_probs = gaussian_to_bucket_probs(
            ridge_res["mu_val"], ridge_res["sigma_val"], bucket_edges
        )
        ridge_val_brier = compute_brier_score(ridge_val_probs, val_actual, bucket_edges)
        logger.info("  Ridge alpha=%.1f: val Brier=%.4f", alpha, ridge_val_brier["overall_brier"])

        if ridge_val_brier["overall_brier"] < best_ridge_val_brier:
            best_ridge_val_brier = ridge_val_brier["overall_brier"]
            best_ridge_result = ridge_res
            best_ridge_alpha = alpha

    logger.info("Best Ridge alpha=%.1f", best_ridge_alpha)

    ridge_brier, ridge_seasonal = eval_contract_brier(
        best_ridge_result["mu_test"], best_ridge_result["sigma_test"],
        best_ridge_result["mu_val"], best_ridge_result["sigma_val"])

    logger.info("Ridge (alpha=%.1f): contract Brier=%.4f",
                best_ridge_alpha, ridge_brier)

    all_results[f"Ridge (a={best_ridge_alpha})"] = {
        "test_brier": ridge_brier,
        "val_mae": best_ridge_result["val_mae"],
        "test_mae": best_ridge_result["test_mae"],
        "alpha": best_ridge_alpha,
    }
    seasonal_all[f"Ridge (a={best_ridge_alpha})"] = ridge_seasonal

    # ===================================================================
    # 4. Heteroscedastic Feedforward NN
    # ===================================================================
    logger.info("-" * 50)
    logger.info("Model 4: Heteroscedastic Feedforward NN")
    logger.info("-" * 50)

    nn_res = train_heteroscedastic_nn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_sizes=[128, 64],
        dropout=0.1,
        lr=0.001,
        max_epochs=200,
        patience=20,
        batch_size=city_config.BATCH_SIZE,
    )

    nn_brier, nn_seasonal = eval_contract_brier(
        nn_res["mu_test"], nn_res["sigma_test"],
        nn_res["mu_val"], nn_res["sigma_val"])

    logger.info("HeteroscedasticNN: contract Brier=%.4f", nn_brier)

    all_results["HeteroscedasticNN"] = {
        "test_brier": nn_brier,
        "val_mae": nn_res["val_mae"],
        "test_mae": nn_res["test_mae"],
        "best_epoch": nn_res["best_epoch"],
    }
    seasonal_all["HeteroscedasticNN"] = nn_seasonal

    # Save NN model checkpoint
    nn_model_path = os.path.join(chi.models_dir, "heteroscedastic_nn_aus.pt")
    os.makedirs(chi.models_dir, exist_ok=True)
    torch.save(nn_res["model"].state_dict(), nn_model_path)
    logger.info("Saved NN checkpoint to %s", nn_model_path)

    # ===================================================================
    # Summary
    # ===================================================================
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY (Contract Brier)")
    logger.info("=" * 70)

    summary_rows = []
    for model_name, res in all_results.items():
        row = {
            "model": model_name,
            "contract_brier": res["test_brier"],
        }
        if "val_mae" in res:
            row["val_mae"] = res["val_mae"]
        if "test_mae" in res:
            row["test_mae"] = res["test_mae"]

        if model_name in seasonal_all:
            for season, score in seasonal_all[model_name].items():
                row[f"brier_{season}"] = score

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("contract_brier")

    logger.info("\n%s", summary_df.to_string(index=False))

    # --- Save results ---
    summary_path = os.path.join(results_dir, "aus_benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    # Detailed results JSON
    detail_path = os.path.join(results_dir, "aus_benchmark_detail.json")
    serializable_results = {}
    for model_name, res in all_results.items():
        ser_res = {}
        for k, v in res.items():
            if isinstance(v, np.floating):
                ser_res[k] = float(v)
            elif isinstance(v, np.integer):
                ser_res[k] = int(v)
            else:
                ser_res[k] = v
        serializable_results[model_name] = ser_res

    with open(detail_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    logger.info("Saved detailed results to %s", detail_path)

    # Training history for NN
    if nn_res.get("history"):
        history_df = pd.DataFrame(nn_res["history"])
        history_path = os.path.join(results_dir, "aus_nn_training_history.csv")
        history_df.to_csv(history_path, index=False)
        logger.info("Saved NN training history to %s", history_path)

    # --- Plots ---
    plot_brier_comparison(
        all_results,
        os.path.join(results_dir, "aus_brier_comparison.png"),
    )
    plot_seasonal_brier(
        seasonal_all,
        os.path.join(results_dir, "aus_seasonal_brier.png"),
    )

    # --- Metadata ---
    metadata = {
        "city": "Austin",
        "kalshi_ticker": "KXHIGHAUS",
        "scoring": "contract_brier",
        "target_station": city_config.TARGET_STATION,
        "n_surrounding_stations": len(city_config.SURROUNDING_STATIONS),
        "date_range": f"{city_config.START_DATE} to {city_config.END_DATE}",
        "n_features": X_train.shape[1],
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_buckets": len(bucket_edges),
        "best_model": summary_df.iloc[0]["model"],
        "best_contract_brier": float(summary_df.iloc[0]["contract_brier"]),
    }
    metadata_path = os.path.join(results_dir, "aus_benchmark_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    # Save benchmark_summary.json (used by promotion evaluation)
    benchmark_summary = {
        "best_brier": float(summary_df.iloc[0]["contract_brier"]),
        "best_model": summary_df.iloc[0]["model"],
        "n_oos_days": len(y_test),
        "all_models": {row["model"]: float(row["contract_brier"]) for _, row in summary_df.iterrows()},
    }
    bm_summary_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(bm_summary_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info("Saved benchmark summary to %s", bm_summary_path)

    # Save seasonal_brier.json (used by promotion evaluation)
    best_model_name = summary_df.iloc[0]["model"]
    if best_model_name in seasonal_all:
        seasonal_brier_path = os.path.join(results_dir, "seasonal_brier.json")
        with open(seasonal_brier_path, "w") as f:
            json.dump(seasonal_all[best_model_name], f, indent=2)
        logger.info("Saved seasonal Brier to %s", seasonal_brier_path)

    # Save aus_vs_kalshi_comparison.json — compare best model vs Kalshi market
    best_model_brier = float(summary_df.iloc[0]["contract_brier"])
    best_model_name_comp = summary_df.iloc[0]["model"]
    kalshi_comparison = {
        "best_model": best_model_name_comp,
        "best_model_contract_brier": best_model_brier,
        "all_model_briers": {
            row["model"]: float(row["contract_brier"])
            for _, row in summary_df.iterrows()
        },
    }
    if "market_prob" in kalshi.columns:
        market_valid = kalshi.dropna(subset=["market_prob", "actual_outcome"])
        if len(market_valid) > 0:
            market_brier = contract_brier(
                market_valid["market_prob"].values,
                market_valid["actual_outcome"].values,
            )
            kalshi_comparison["kalshi_market_brier"] = float(market_brier)
            kalshi_comparison["model_edge"] = float(market_brier - best_model_brier)
            kalshi_comparison["n_market_rows"] = int(len(market_valid))
            kalshi_comparison["n_market_dates"] = int(market_valid["date"].nunique())
            logger.info(
                "Kalshi comparison: best model (%s) Brier=%.4f, market Brier=%.4f, edge=+%.4f",
                best_model_name_comp, best_model_brier, market_brier,
                market_brier - best_model_brier,
            )
        else:
            kalshi_comparison["kalshi_market_brier"] = None
            kalshi_comparison["model_edge"] = None
            logger.info("No valid market_prob rows for Kalshi comparison")
    else:
        kalshi_comparison["kalshi_market_brier"] = None
        kalshi_comparison["model_edge"] = None
        logger.info("No market_prob column in Kalshi data — skipping market comparison")

    kalshi_comp_path = os.path.join(results_dir, "aus_vs_kalshi_comparison.json")
    with open(kalshi_comp_path, "w") as f:
        json.dump(kalshi_comparison, f, indent=2)
    logger.info("Saved Kalshi comparison to %s", kalshi_comp_path)

    # Save base predictions for synthesis/backtest pipeline (all models)
    all_model_preds = {
        "Persistence": persist,
        "Climatology": clim,
        "Ridge": best_ridge_result,
        "HeteroscedasticNN": nn_res,
    }

    preds_rows = []
    for model_name, model_preds in all_model_preds.items():
        for i, d in enumerate(y_val.index):
            preds_rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "model_name": model_name,
                "mu": float(model_preds["mu_val"][i]),
                "sigma": float(model_preds["sigma_val"][i]),
                "actual_tmax": float(y_val.values[i]),
            })
        for i, d in enumerate(y_test.index):
            preds_rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "model_name": model_name,
                "mu": float(model_preds["mu_test"][i]),
                "sigma": float(model_preds["sigma_test"][i]),
                "actual_tmax": float(y_test.values[i]),
            })
    preds_df = pd.DataFrame(preds_rows)
    preds_path = os.path.join(results_dir, "base_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    logger.info("Saved base predictions (%d rows, %d models) to %s",
                len(preds_df), len(all_model_preds), preds_path)

    logger.info("=" * 70)
    logger.info("Austin Benchmark Complete")
    logger.info("Best model: %s (contract Brier: %.4f)",
                metadata["best_model"], metadata["best_contract_brier"])
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
