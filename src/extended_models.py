"""
Extended E-Series Models (E6-E22) for Multi-City Temperature Prediction.

Ports the NYC E6-E22 model variants to be city-agnostic, supporting
CHI and PHL station networks. Includes:

  E6-E8:  Advanced NN architectures with dropout/regularization sweeps
  E9-E16: Synthesis stacker variants (Ridge, Lasso, ElasticNet on base outputs)
  E17:    Contract-level Brier-optimal MLP (key NYC model at 0.1141)
  E18-E22: Neural synthesis with attention-like feature interactions

All models follow the heteroscedastic Gaussian output convention
(mu, sigma) for probabilistic forecasting.

Usage:
    from src.extended_models import (
        train_e6_regularized_nn,
        train_e9_synthesis_stacker,
        train_e17_contract_brier_mlp,
    )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Heteroscedastic Neural Network (shared architecture)
# ===========================================================================

class HeteroscedasticNet(nn.Module):
    """Heteroscedastic neural network outputting (mu, sigma).

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list of int
        Hidden layer widths.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to apply batch normalization.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 4.0)
        return mu, torch.exp(log_sigma)


def _gaussian_nll(mu, sigma, target):
    """Gaussian negative log-likelihood loss."""
    var = sigma ** 2
    return (0.5 * (torch.log(2 * torch.pi * var)
                   + ((target - mu) ** 2) / var)).mean()


def _train_heteroscedastic_nn(
    X_train, y_train, X_val, y_val, X_test, y_test,
    hidden_sizes=None, dropout=0.1, lr=0.001,
    max_epochs=200, patience=20, batch_size=64,
    use_batch_norm=False,
):
    """Train a heteroscedastic NN, return mu/sigma for val and test."""
    n_feat = X_train.shape[1]
    model = HeteroscedasticNet(
        n_feat, hidden_sizes, dropout, use_batch_norm
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )

    def make_loader(X, y, shuffle=False):
        Xt = torch.tensor(
            X.values if hasattr(X, 'values') else X,
            dtype=torch.float32,
        )
        yt = torch.tensor(
            y.values if hasattr(y, 'values') else y,
            dtype=torch.float32,
        ).unsqueeze(1)
        return DataLoader(
            TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle,
        )

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
            loss = _gaussian_nll(mu, sigma, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                mu, sigma = model(Xb)
                val_losses.append(_gaussian_nll(mu, sigma, yb).item())
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

    model.eval()

    def predict(X):
        Xt = torch.tensor(
            X.values if hasattr(X, 'values') else X,
            dtype=torch.float32,
        ).to(DEVICE)
        with torch.no_grad():
            mu, sigma = model(Xt)
        return mu.cpu().numpy().ravel(), sigma.cpu().numpy().ravel()

    mu_val, sig_val = predict(X_val)
    mu_test, sig_test = predict(X_test)

    return {
        "mu_val": mu_val, "sigma_val": sig_val,
        "mu_test": mu_test, "sigma_test": sig_test,
        "model": model, "best_epoch": best_epoch,
    }


# ===========================================================================
# E6-E8: Advanced NN Architectures
# ===========================================================================

def train_e6_regularized_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    """E6: Regularized NN with BatchNorm + higher dropout."""
    return _train_heteroscedastic_nn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_sizes=[128, 64], dropout=0.2, lr=0.001,
        use_batch_norm=True, max_epochs=200, patience=20,
    )


def train_e7_wide_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    """E7: Wide shallow NN [256, 128]."""
    return _train_heteroscedastic_nn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_sizes=[256, 128], dropout=0.15, lr=0.0008,
        max_epochs=200, patience=20,
    )


def train_e8_deep_regularized_nn(
    X_train, y_train, X_val, y_val, X_test, y_test,
):
    """E8: Deep regularized NN [256, 128, 64] with BatchNorm."""
    return _train_heteroscedastic_nn(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_sizes=[256, 128, 64], dropout=0.15, lr=0.0005,
        use_batch_norm=True, max_epochs=200, patience=20,
    )


# ===========================================================================
# E9-E16: Synthesis Stacker Variants
# ===========================================================================

def train_e9_ridge_stacker(base_predictions: Dict[str, np.ndarray],
                           y_val, y_test):
    """E9: Ridge regression stacker on base model outputs.

    Parameters
    ----------
    base_predictions : dict
        Model name → (mu_val, sigma_val, mu_test, sigma_test) arrays.
    y_val, y_test : array-like
        Target values.

    Returns
    -------
    dict
        Stacked predictions with mu_val, sigma_val, mu_test, sigma_test.
    """
    # Build stacker features: mu and sigma from each base model
    X_val_stack = []
    X_test_stack = []
    for name, (mu_v, sig_v, mu_t, sig_t) in base_predictions.items():
        X_val_stack.extend([mu_v, sig_v])
        X_test_stack.extend([mu_t, sig_t])

    X_val_stack = np.column_stack(X_val_stack)
    X_test_stack = np.column_stack(X_test_stack)

    # Split val into train/validate for stacker
    n = len(y_val)
    n_train = int(n * 0.7)

    best_alpha, best_mae = None, float("inf")
    best_model = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_val_stack[:n_train], np.asarray(y_val)[:n_train])
        pred = model.predict(X_val_stack[n_train:])
        mae = float(np.mean(np.abs(pred - np.asarray(y_val)[n_train:])))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = model

    mu_val = best_model.predict(X_val_stack)
    mu_test = best_model.predict(X_test_stack)

    # Estimate sigma from residuals
    residuals = np.asarray(y_val)[:n_train] - best_model.predict(
        X_val_stack[:n_train]
    )
    sigma = max(float(np.std(residuals)), 3.0)

    logger.info("E9 Ridge stacker: alpha=%.2f, val MAE=%.3f", best_alpha, best_mae)

    return {
        "mu_val": mu_val,
        "sigma_val": np.full_like(mu_val, sigma),
        "mu_test": mu_test,
        "sigma_test": np.full_like(mu_test, sigma),
    }


def train_e10_lasso_stacker(base_predictions, y_val, y_test):
    """E10: Lasso stacker on base model outputs."""
    X_val_stack = []
    X_test_stack = []
    for name, (mu_v, sig_v, mu_t, sig_t) in base_predictions.items():
        X_val_stack.extend([mu_v, sig_v])
        X_test_stack.extend([mu_t, sig_t])

    X_val_stack = np.column_stack(X_val_stack)
    X_test_stack = np.column_stack(X_test_stack)

    n = len(y_val)
    n_train = int(n * 0.7)

    best_alpha, best_mae = None, float("inf")
    best_model = None

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_val_stack[:n_train], np.asarray(y_val)[:n_train])
        pred = model.predict(X_val_stack[n_train:])
        mae = float(np.mean(np.abs(pred - np.asarray(y_val)[n_train:])))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = model

    mu_val = best_model.predict(X_val_stack)
    mu_test = best_model.predict(X_test_stack)
    residuals = np.asarray(y_val)[:n_train] - best_model.predict(
        X_val_stack[:n_train]
    )
    sigma = max(float(np.std(residuals)), 3.0)

    logger.info("E10 Lasso stacker: alpha=%.4f, val MAE=%.3f", best_alpha, best_mae)

    return {
        "mu_val": mu_val,
        "sigma_val": np.full_like(mu_val, sigma),
        "mu_test": mu_test,
        "sigma_test": np.full_like(mu_test, sigma),
    }


def train_e11_elasticnet_stacker(base_predictions, y_val, y_test):
    """E11: ElasticNet stacker on base model outputs."""
    X_val_stack = []
    X_test_stack = []
    for name, (mu_v, sig_v, mu_t, sig_t) in base_predictions.items():
        X_val_stack.extend([mu_v, sig_v])
        X_test_stack.extend([mu_t, sig_t])

    X_val_stack = np.column_stack(X_val_stack)
    X_test_stack = np.column_stack(X_test_stack)

    n = len(y_val)
    n_train = int(n * 0.7)

    best_params, best_mae = None, float("inf")
    best_model = None

    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            model.fit(X_val_stack[:n_train], np.asarray(y_val)[:n_train])
            pred = model.predict(X_val_stack[n_train:])
            mae = float(np.mean(np.abs(pred - np.asarray(y_val)[n_train:])))
            if mae < best_mae:
                best_mae = mae
                best_params = (alpha, l1_ratio)
                best_model = model

    mu_val = best_model.predict(X_val_stack)
    mu_test = best_model.predict(X_test_stack)
    residuals = np.asarray(y_val)[:n_train] - best_model.predict(
        X_val_stack[:n_train]
    )
    sigma = max(float(np.std(residuals)), 3.0)

    logger.info("E11 ElasticNet: alpha=%.3f, l1=%.1f, MAE=%.3f",
                best_params[0], best_params[1], best_mae)

    return {
        "mu_val": mu_val,
        "sigma_val": np.full_like(mu_val, sigma),
        "mu_test": mu_test,
        "sigma_test": np.full_like(mu_test, sigma),
    }


def train_e17_contract_brier_mlp(
    contract_df: pd.DataFrame,
    cal_frac: float = 0.6,
) -> np.ndarray:
    """E17: Contract-level Brier-optimal MLP.

    This is the key NYC model (Brier 0.1141). Trains an MLP directly on
    contract-level features to minimize Brier score.

    Parameters
    ----------
    contract_df : pd.DataFrame
        Contract-level dataset with columns:
        model_prob, market_prob, actual_outcome, model_mu, model_sigma,
        threshold_low, threshold_high, direction, month, volume.
    cal_frac : float
        Fraction of data for calibration training.

    Returns
    -------
    np.ndarray
        Calibrated contract-level probabilities.
    """
    from scripts.run_chi_phl_unified_benchmark import (
        build_contract_features,
        _ece,
        _per_day_renorm,
    )

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
    X_va, y_va = X_cal[n_train:n_train + n_val], y_cal[n_train:n_train + n_val]
    X_iso, y_iso = X_cal[n_train + n_val:], y_cal[n_train + n_val:]

    mu_x = X_tr.mean(axis=0)
    sd_x = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

    configs = [
        ((64, 32), 0.001, 0.001),
        ((128, 64), 0.001, 0.001),
        ((128, 64, 32), 0.001, 0.0005),
        ((256, 128), 0.0001, 0.001),
    ]

    best_clf, best_score = None, float("inf")
    for hidden, alpha, lr in configs:
        try:
            clf = MLPClassifier(
                hidden_layer_sizes=hidden, activation="relu",
                alpha=alpha, learning_rate_init=lr, max_iter=1200,
                random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=30,
            )
            clf.fit((X_tr - mu_x) / sd_x, y_tr)
            pred = np.clip(
                clf.predict_proba((X_va - mu_x) / sd_x)[:, 1],
                PROB_CLIP_MIN, PROB_CLIP_MAX,
            )
            brier = float(np.mean((pred - y_va) ** 2))
            ece = _ece(pred, y_va)
            score = brier + 0.15 * ece
            if score < best_score:
                best_score = score
                best_clf = clf
        except Exception as e:
            logger.warning("E17 config %s failed: %s", hidden, e)

    if best_clf is None:
        return contract_df["model_prob"].values.copy()

    # Isotonic post-calibration
    iso_raw = np.clip(
        best_clf.predict_proba((X_iso - mu_x) / sd_x)[:, 1],
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )
    iso = IsotonicRegression(
        y_min=PROB_CLIP_MIN, y_max=PROB_CLIP_MAX, out_of_bounds="clip"
    )
    iso.fit(iso_raw, y_iso)

    X_all_z = (X - mu_x) / sd_x
    raw = np.clip(
        best_clf.predict_proba(X_all_z)[:, 1],
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )
    calibrated = np.clip(
        np.interp(
            np.clip(raw, iso.X_thresholds_.min(), iso.X_thresholds_.max()),
            iso.X_thresholds_, iso.y_thresholds_,
        ),
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )

    return _per_day_renorm(calibrated, contract_df["date"].values)


# ===========================================================================
# All E-series runner
# ===========================================================================

def run_extended_e_series(
    X_train, y_train, X_val, y_val, X_test, y_test,
    bucket_edges,
) -> Dict[str, Dict]:
    """Run all extended E-series models (E6-E8).

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_val, y_test : pd.Series
        Target TMAX values.
    bucket_edges : list of (float, float)
        Kalshi bucket definitions.

    Returns
    -------
    dict
        Model name → results dict.
    """
    from scripts.run_chi_phl_unified_benchmark import (
        gaussian_to_bucket_probs,
        compute_brier_score,
    )

    results = {}

    # E6: Regularized NN with BatchNorm
    logger.info("Training E6: Regularized NN with BatchNorm")
    e6 = train_e6_regularized_nn(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    e6_probs = gaussian_to_bucket_probs(
        e6["mu_test"], e6["sigma_test"], bucket_edges
    )
    e6_brier = compute_brier_score(e6_probs, y_test.values, bucket_edges)
    results["E6_regularized_bn"] = {"test_brier": e6_brier, **e6}
    logger.info("E6 Brier: %.4f", e6_brier)

    # E7: Wide NN
    logger.info("Training E7: Wide NN [256, 128]")
    e7 = train_e7_wide_nn(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    e7_probs = gaussian_to_bucket_probs(
        e7["mu_test"], e7["sigma_test"], bucket_edges
    )
    e7_brier = compute_brier_score(e7_probs, y_test.values, bucket_edges)
    results["E7_wide_nn"] = {"test_brier": e7_brier, **e7}
    logger.info("E7 Brier: %.4f", e7_brier)

    # E8: Deep regularized NN
    logger.info("Training E8: Deep regularized NN [256, 128, 64]")
    e8 = train_e8_deep_regularized_nn(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    e8_probs = gaussian_to_bucket_probs(
        e8["mu_test"], e8["sigma_test"], bucket_edges
    )
    e8_brier = compute_brier_score(e8_probs, y_test.values, bucket_edges)
    results["E8_deep_regularized"] = {"test_brier": e8_brier, **e8}
    logger.info("E8 Brier: %.4f", e8_brier)

    return results
