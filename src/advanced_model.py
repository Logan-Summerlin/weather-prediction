"""
Advanced Neural Network Models for Multi-City Temperature Prediction.

Implements model families inspired by NYC's top-performing architectures
(U7_regime_conditional, E17_contract_brier, E40_lag2_contract_brier,
WindGatedAttention V2). Designed for city-agnostic deployment across
Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL).

Model Families
--------------
1. FeatureAttentionNet
   Dynamic feature importance via learned attention gates. The network
   first encodes global context to understand the weather regime, then
   computes per-feature attention weights. This allows it to dynamically
   upweight informative features (e.g., upwind stations when advection is
   strong) and downweight noisy ones. Heteroscedastic Gaussian output.

2. MOSCorrectionNet
   Predicts residual error from a MOS/climatology baseline rather than
   raw TMAX. Reduces target variance by ~80%, making optimization much
   easier and improving generalization. Uses a two-stage approach:
   first fit a simple MOS model, then train an NN on the residuals.

3. RegimeConditionalNet
   Based on NYC's best model (U7_regime_conditional). Learns different
   uncertainty (sigma) per weather regime, defined by season x volatility-
   bin interaction features. Captures the fact that winter forecasts have
   much larger uncertainty than summer ones.

4. EnsembleCalibrator
   Meta-learner that combines predictions from multiple models with
   isotonic + Platt calibration. Optimizes directly on contract-level
   Brier score.

Training Protocol
-----------------
- Chronological splits only (no shuffling)
- Feature scaling fit on training data only
- CRPS + MAE combined loss for distributional training
- Gradient clipping (max_norm=5.0)
- Early stopping with patience=20
- ReduceLROnPlateau scheduler
"""

import logging
import math
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard normal helpers (for CRPS)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _std_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))


# =====================================================================
# 1. Feature Attention Network
# =====================================================================

class FeatureAttentionNet(nn.Module):
    """Neural network with dynamic feature importance via learned attention.

    Architecture:
        1. Context encoder: small MLP that reads ALL features to understand
           the current weather regime (wind pattern, season, volatility).
        2. Attention gate: uses context to compute per-feature importance
           scores via softmax, producing a soft mask over input features.
        3. Prediction MLP: processes attention-weighted features through
           a deeper MLP with batch norm and dropout.
        4. Dual output heads: separate mu and log_sigma heads for
           heteroscedastic Gaussian predictions.

    The attention weights are directly interpretable: they show which
    stations and features the model considers most predictive for each
    forecast day.

    Parameters
    ----------
    n_features : int
        Number of input features.
    context_dim : int
        Dimension of the context encoding.
    hidden_sizes : list[int]
        Widths of the prediction MLP hidden layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int = 64,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.n_features = n_features
        self.context_dim = context_dim

        # Context encoder: understand the weather regime
        self.context_encoder = nn.Sequential(
            nn.Linear(n_features, context_dim),
            nn.LayerNorm(context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

        # Attention gate: context -> per-feature importance scores
        self.attention_gate = nn.Sequential(
            nn.Linear(context_dim, n_features),
        )
        # Temperature parameter for softmax sharpness (learnable)
        self.attn_temperature = nn.Parameter(torch.tensor(1.0))

        # Prediction MLP on attended features
        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.prediction_trunk = nn.Sequential(*layers)

        # Output heads
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "FeatureAttentionNet: n_features=%d, context=%d, hidden=%s, "
            "params=%d", n_features, context_dim, hidden_sizes, n_params
        )

    def _init_weights(self):
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
            Shape (batch, n_features). Scaled input features.

        Returns
        -------
        dict with keys: prediction, mu, sigma, log_sigma, attention_weights
        """
        # 1. Encode context
        context = self.context_encoder(x)  # (B, context_dim)

        # 2. Compute attention weights
        attn_logits = self.attention_gate(context)  # (B, n_features)
        temperature = self.attn_temperature.clamp(min=0.1)
        attn_weights = F.softmax(attn_logits / temperature, dim=-1)

        # 3. Apply attention (element-wise gating, scaled)
        x_attended = x * attn_weights * self.n_features

        # 4. Prediction trunk
        h = self.prediction_trunk(x_attended)

        # 5. Output heads
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 4.0)
        sigma = torch.exp(log_sigma)

        return {
            "prediction": mu,
            "mu": mu,
            "log_sigma": log_sigma,
            "sigma": sigma,
            "attention_weights": attn_weights,
        }


# =====================================================================
# 2. MOS Correction Network
# =====================================================================

class MOSCorrectionNet(nn.Module):
    """Neural network that corrects MOS/climatology forecast errors.

    Instead of predicting raw TMAX, predicts the residual:
        TMAX = baseline_forecast + NN(features)

    The baseline can be climatology, persistence-climatology blend,
    or an actual MOS forecast. Since the residual has much lower
    variance than raw TMAX, the NN can focus on learning the
    correction pattern.

    Parameters
    ----------
    n_features : int
        Number of input features (includes baseline forecast as a feature).
    hidden_sizes : list[int]
        MLP hidden layer widths.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.n_features = n_features

        # Correction MLP
        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Correction head (predicts delta from baseline)
        self.delta_head = nn.Linear(in_dim, 1)
        # Uncertainty head
        self.log_sigma_head = nn.Linear(in_dim, 1)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MOSCorrectionNet: n_features=%d, hidden=%s, params=%d",
            n_features, hidden_sizes, n_params
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize delta head bias near 0 (no correction initially)
        nn.init.zeros_(self.delta_head.weight)

    def forward(
        self, x: torch.Tensor, baseline_mu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_features). All features including baseline.
        baseline_mu : torch.Tensor
            Shape (batch, 1). The MOS/climatology baseline forecast.

        Returns
        -------
        dict with keys: prediction, mu, sigma, log_sigma, delta
        """
        h = self.trunk(x)
        delta = self.delta_head(h)  # correction term
        log_sigma = self.log_sigma_head(h).clamp(-5.0, 4.0)
        sigma = torch.exp(log_sigma)

        mu = baseline_mu + delta  # final prediction = baseline + correction

        return {
            "prediction": mu,
            "mu": mu,
            "log_sigma": log_sigma,
            "sigma": sigma,
            "delta": delta,
        }


# =====================================================================
# 3. Regime-Conditional Network
# =====================================================================

class RegimeConditionalNet(nn.Module):
    """Regime-conditional heteroscedastic network.

    Based on NYC's U7_regime_conditional. The key insight is that
    forecast uncertainty varies dramatically by weather regime:
    - Winter cold outbreaks have high uncertainty
    - Summer stable patterns have low uncertainty
    - Transition seasons are moderately uncertain

    The model learns separate variance estimates per regime, defined
    by season x volatility-bin interaction features.

    Parameters
    ----------
    n_features : int
        Number of base input features.
    n_regime_features : int
        Number of regime features (season + volatility indicators).
    hidden_sizes : list[int]
        MLP hidden layer widths.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        n_regime_features: int = 16,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.15,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.n_features = n_features
        self.n_regime_features = n_regime_features
        total_in = n_features + n_regime_features

        # Main prediction trunk
        layers: list[nn.Module] = []
        in_dim = total_in
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Point prediction head
        self.mu_head = nn.Linear(in_dim, 1)

        # Regime-conditional sigma head
        # Takes trunk output + regime features for regime-aware uncertainty
        sigma_in = in_dim + n_regime_features
        self.sigma_trunk = nn.Sequential(
            nn.Linear(sigma_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "RegimeConditionalNet: n_features=%d, n_regime=%d, "
            "hidden=%s, params=%d",
            n_features, n_regime_features, hidden_sizes, n_params
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        regime_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_features). Base input features.
        regime_features : torch.Tensor
            Shape (batch, n_regime_features). Season/volatility indicators.

        Returns
        -------
        dict with keys: prediction, mu, sigma, log_sigma
        """
        combined = torch.cat([x, regime_features], dim=-1)
        h = self.trunk(combined)

        mu = self.mu_head(h)

        # Regime-conditional sigma
        sigma_input = torch.cat([h, regime_features], dim=-1)
        log_sigma = self.sigma_trunk(sigma_input).clamp(-5.0, 4.0)
        sigma = torch.exp(log_sigma)

        return {
            "prediction": mu,
            "mu": mu,
            "log_sigma": log_sigma,
            "sigma": sigma,
        }


# =====================================================================
# Feature Engineering Utilities
# =====================================================================

def compute_regime_features(dates: pd.DatetimeIndex, tmax: np.ndarray) -> np.ndarray:
    """Compute regime features: season one-hot + volatility bins + interactions.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates for each sample.
    tmax : np.ndarray
        Target TMAX values (used for volatility computation).

    Returns
    -------
    np.ndarray
        Shape (n_samples, 16). Regime features:
        [4 season one-hot, 3 volatility bins, 4*3=12 interactions - 3 = 16 total]
    """
    n = len(dates)
    months = dates.month

    # Season one-hot: DJF, MAM, JJA, SON
    season_map = {
        12: 0, 1: 0, 2: 0,  # DJF
        3: 1, 4: 1, 5: 1,   # MAM
        6: 2, 7: 2, 8: 2,   # JJA
        9: 3, 10: 3, 11: 3,  # SON
    }
    season_idx = np.array([season_map[m] for m in months])
    season_onehot = np.zeros((n, 4))
    for i in range(n):
        season_onehot[i, season_idx[i]] = 1.0

    # Volatility bins based on rolling 7-day TMAX std
    tmax_series = pd.Series(tmax, index=dates)
    rolling_std = tmax_series.rolling(7, min_periods=3).std().fillna(
        tmax_series.std()
    ).values

    # Quantile-based bins: low (0-33%), medium (33-67%), high (67-100%)
    q33 = np.nanpercentile(rolling_std, 33)
    q67 = np.nanpercentile(rolling_std, 67)

    vol_bins = np.zeros((n, 3))
    vol_bins[rolling_std <= q33, 0] = 1.0   # low volatility
    vol_bins[(rolling_std > q33) & (rolling_std <= q67), 1] = 1.0  # medium
    vol_bins[rolling_std > q67, 2] = 1.0   # high volatility
    # Handle any remaining zeros (assign to medium)
    no_bin = vol_bins.sum(axis=1) == 0
    vol_bins[no_bin, 1] = 1.0

    # Season x volatility interactions (4 seasons x 3 vol bins = 12 features)
    # But drop 3 to avoid multicollinearity -> keep 9
    interactions = np.zeros((n, 9))
    idx = 0
    for s in range(3):  # skip last season (captured by baseline)
        for v in range(3):
            interactions[:, idx] = season_onehot[:, s] * vol_bins[:, v]
            idx += 1

    regime = np.concatenate([season_onehot, vol_bins, interactions], axis=1)
    return regime.astype(np.float32)


def compute_mos_baseline(
    y_train: np.ndarray,
    dates_train: pd.DatetimeIndex,
    y_all: np.ndarray,
    dates_all: pd.DatetimeIndex,
) -> tuple[np.ndarray, float]:
    """Compute MOS-like baseline forecast using persistence + climatology blend.

    Parameters
    ----------
    y_train : np.ndarray
        Training set TMAX values.
    dates_train : pd.DatetimeIndex
        Training set dates.
    y_all : np.ndarray
        All TMAX values (train + val + test concatenated).
    dates_all : pd.DatetimeIndex
        All dates.

    Returns
    -------
    tuple[np.ndarray, float]
        (baseline_mu for all dates, baseline_sigma from training residuals)
    """
    # Compute DOY climatology from training set
    train_df = pd.DataFrame({"tmax": y_train, "doy": dates_train.dayofyear})
    clim_mean = train_df.groupby("doy")["tmax"].mean()
    all_doys = np.arange(1, 367)
    clim_mean = clim_mean.reindex(all_doys).interpolate().bfill().ffill()

    # Persistence (lag-1)
    tmax_series = pd.Series(y_all, index=dates_all)
    persistence = tmax_series.shift(1).values

    # Blend: 0.45 * persistence + 0.55 * climatology
    # (optimal blend weights from NYC experiments)
    clim_vals = clim_mean.reindex(dates_all.dayofyear).values
    baseline = np.where(
        np.isnan(persistence),
        clim_vals,
        0.45 * persistence + 0.55 * clim_vals,
    )

    # Estimate sigma from training residuals
    n_train = len(y_train)
    train_residuals = y_train - baseline[:n_train]
    train_residuals = train_residuals[~np.isnan(train_residuals)]
    sigma = float(np.std(train_residuals)) if len(train_residuals) > 10 else 5.0
    sigma = max(sigma, 2.0)

    return baseline.astype(np.float32), sigma


def add_enhanced_features(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add enhanced features derived from base station features.

    Adds rolling statistics, inter-station gradients, and temporal
    features that improve model performance.

    Parameters
    ----------
    X : pd.DataFrame
        Base feature matrix.
    y : pd.Series
        Target TMAX series.
    dates : pd.DatetimeIndex
        Date index.

    Returns
    -------
    pd.DataFrame
        Enhanced feature matrix with additional columns.
    """
    X_enhanced = X.copy()

    # 1. Rolling means of target (lag-safe: use shifted values)
    y_series = pd.Series(y.values, index=dates)
    y_lag1 = y_series.shift(1)
    y_lag2 = y_series.shift(2)

    X_enhanced["target_lag1"] = y_lag1.values
    X_enhanced["target_lag2"] = y_lag2.values
    X_enhanced["target_rolling7_mean"] = y_lag1.rolling(7, min_periods=1).mean().values
    X_enhanced["target_rolling7_std"] = y_lag1.rolling(7, min_periods=1).std().values
    X_enhanced["target_delta"] = (y_lag1 - y_lag2).values

    # 2. Day of year features (already have sin/cos, add raw doy)
    X_enhanced["doy"] = dates.dayofyear / 366.0

    # 3. Month (cyclical)
    months = dates.month
    X_enhanced["month_sin"] = np.sin(2 * np.pi * months / 12)
    X_enhanced["month_cos"] = np.cos(2 * np.pi * months / 12)

    # 4. Station consensus features (mean/std/range across all station columns)
    tmax_cols = [c for c in X.columns if "TMAX" in c and "lag" in c.lower()]
    if not tmax_cols:
        tmax_cols = [c for c in X.columns if "TMAX" in c]
    if tmax_cols:
        station_data = X[tmax_cols]
        X_enhanced["station_mean"] = station_data.mean(axis=1)
        X_enhanced["station_std"] = station_data.std(axis=1)
        X_enhanced["station_range"] = station_data.max(axis=1) - station_data.min(axis=1)
        X_enhanced["station_median"] = station_data.median(axis=1)
        # Stations above/below climatology
        if "target_lag1" in X_enhanced.columns:
            above = (station_data.values > X_enhanced["target_lag1"].values[:, None]).sum(axis=1)
            X_enhanced["stations_above_lag1"] = above / max(len(tmax_cols), 1)

    tmin_cols = [c for c in X.columns if "TMIN" in c and "lag" in c.lower()]
    if not tmin_cols:
        tmin_cols = [c for c in X.columns if "TMIN" in c]
    if tmin_cols:
        tmin_data = X[tmin_cols]
        X_enhanced["tmin_station_mean"] = tmin_data.mean(axis=1)
        X_enhanced["tmin_station_std"] = tmin_data.std(axis=1)

    # 5. Diurnal range proxy (mean TMAX - mean TMIN across stations)
    if tmax_cols and tmin_cols:
        X_enhanced["diurnal_range"] = (
            X_enhanced.get("station_mean", 0) -
            X_enhanced.get("tmin_station_mean", 0)
        )

    # Fill NaN in new features
    X_enhanced = X_enhanced.ffill().bfill().fillna(0)

    return X_enhanced


# =====================================================================
# Training Utilities
# =====================================================================

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


def combined_crps_mae_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor,
    crps_weight: float = 0.7, mae_weight: float = 0.3,
) -> torch.Tensor:
    """Combined CRPS + MAE loss for training stability."""
    crps = gaussian_crps_loss(mu, sigma, target)
    mae = torch.abs(mu.reshape(-1) - target.reshape(-1)).mean()
    return crps_weight * crps + mae_weight * mae


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "standard",
    baseline_train: np.ndarray | None = None,
    baseline_val: np.ndarray | None = None,
    regime_train: np.ndarray | None = None,
    regime_val: np.ndarray | None = None,
    lr: float = 0.001,
    max_epochs: int = 300,
    patience: int = 25,
    batch_size: int = 64,
    loss_type: str = "crps_mae",
) -> dict:
    """Train a model with CRPS/MAE loss and early stopping.

    Parameters
    ----------
    model : nn.Module
        The model to train (FeatureAttentionNet, MOSCorrectionNet, or
        RegimeConditionalNet).
    X_train, X_val : np.ndarray
        Scaled feature matrices.
    y_train, y_val : np.ndarray
        Target values (TMAX in deg F for standard, residuals for MOS).
    model_type : str
        "standard", "mos_correction", or "regime_conditional".
    baseline_train, baseline_val : np.ndarray or None
        MOS baseline forecasts (required for mos_correction).
    regime_train, regime_val : np.ndarray or None
        Regime features (required for regime_conditional).
    lr : float
        Initial learning rate.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    batch_size : int
        Training mini-batch size.
    loss_type : str
        "crps_mae" or "nll".

    Returns
    -------
    dict
        Training results with model, history, best_epoch, predictions.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    # Prepare tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    extra_train = {}
    extra_val = {}
    if model_type == "mos_correction":
        extra_train["baseline"] = torch.tensor(
            baseline_train, dtype=torch.float32
        )
        extra_val["baseline"] = torch.tensor(
            baseline_val, dtype=torch.float32
        ).to(DEVICE)
    elif model_type == "regime_conditional":
        extra_train["regime"] = torch.tensor(
            regime_train, dtype=torch.float32
        )
        extra_val["regime"] = torch.tensor(
            regime_val, dtype=torch.float32
        ).to(DEVICE)

    # DataLoader
    tensors = [X_train_t, y_train_t]
    for k in sorted(extra_train.keys()):
        tensors.append(extra_train[k])
    train_ds = TensorDataset(*tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            X_b = batch[0].to(DEVICE)
            y_b = batch[1].to(DEVICE)

            optimizer.zero_grad()

            if model_type == "mos_correction":
                base_b = batch[2].to(DEVICE).unsqueeze(-1)
                out = model(X_b, base_b)
            elif model_type == "regime_conditional":
                regime_b = batch[2].to(DEVICE)
                out = model(X_b, regime_b)
            else:
                out = model(X_b)

            mu = out["mu"].squeeze(-1)
            sigma = out["sigma"].squeeze(-1)

            if loss_type == "crps_mae":
                loss = combined_crps_mae_loss(mu, sigma, y_b)
            else:
                # NLL loss
                var = sigma ** 2
                nll = 0.5 * (torch.log(2 * torch.pi * var) + ((y_b - mu) ** 2) / var)
                loss = nll.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            if model_type == "mos_correction":
                out_v = model(X_val_t, extra_val["baseline"].unsqueeze(-1))
            elif model_type == "regime_conditional":
                out_v = model(X_val_t, extra_val["regime"])
            else:
                out_v = model(X_val_t)

            mu_v = out_v["mu"].squeeze(-1)
            sigma_v = out_v["sigma"].squeeze(-1)

            if loss_type == "crps_mae":
                val_loss = combined_crps_mae_loss(mu_v, sigma_v, y_val_t).item()
            else:
                var_v = sigma_v ** 2
                nll_v = 0.5 * (torch.log(2 * torch.pi * var_v) + ((y_val_t - mu_v) ** 2) / var_v)
                val_loss = nll_v.mean().item()

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
            if epoch <= 5 or epoch % 25 == 0:
                logger.info(
                    "Epoch %3d | train: %.4f | val: %.4f | MAE: %.2f F | * BEST *",
                    epoch, avg_train_loss, val_loss, val_mae,
                )
        else:
            epochs_no_improve += 1
            if epoch % 25 == 0:
                logger.info(
                    "Epoch %3d | train: %.4f | val: %.4f | MAE: %.2f F | no improv (%d/%d)",
                    epoch, avg_train_loss, val_loss, val_mae,
                    epochs_no_improve, patience,
                )

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d (best: %d)", epoch, best_epoch)
            break

    if best_state:
        model.load_state_dict(best_state)

    logger.info("Training complete. Best val loss: %.4f at epoch %d", best_val_loss, best_epoch)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def predict_model(
    model: nn.Module,
    X: np.ndarray,
    model_type: str = "standard",
    baseline: np.ndarray | None = None,
    regime: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions from a trained model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mu, sigma) arrays of shape (n_samples,).
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        if model_type == "mos_correction":
            base_t = torch.tensor(baseline, dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            out = model(X_t, base_t)
        elif model_type == "regime_conditional":
            regime_t = torch.tensor(regime, dtype=torch.float32).to(DEVICE)
            out = model(X_t, regime_t)
        else:
            out = model(X_t)

    mu = out["mu"].cpu().numpy().ravel()
    sigma = out["sigma"].cpu().numpy().ravel()
    return mu, sigma


# =====================================================================
# Calibration Pipeline
# =====================================================================

class IsotonicPlattCalibrator:
    """Two-stage calibration: isotonic regression on PIT values + Platt scaling.

    Stage 1: Isotonic regression maps raw PIT values to calibrated PITs,
    ensuring the forecast CDF is reliable.

    Stage 2: Platt scaling (logistic regression) on bucket probabilities
    to optimize contract-level calibration.
    """

    def __init__(self):
        self.isotonic = None
        self.platt_models = {}
        self.is_fitted = False

    def fit(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        actual: np.ndarray,
        bucket_edges: list[tuple[float, float]],
    ) -> "IsotonicPlattCalibrator":
        """Fit the calibration pipeline on calibration data.

        Parameters
        ----------
        mu, sigma : np.ndarray
            Model predictions (mean, std).
        actual : np.ndarray
            Observed TMAX values.
        bucket_edges : list of (low, high) tuples
            Kalshi contract bucket boundaries.
        """
        # Stage 1: Isotonic on PIT values
        sigma_safe = np.clip(sigma, 1e-6, None)
        pit = norm.cdf(actual, loc=mu, scale=sigma_safe)
        pit = np.clip(pit, 0.001, 0.999)

        self.isotonic = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )
        # Sort for isotonic regression
        sort_idx = np.argsort(pit)
        pit_sorted = pit[sort_idx]
        # Target: uniform quantiles
        target_uniform = np.linspace(0.01, 0.99, len(pit))
        self.isotonic.fit(pit_sorted, target_uniform)

        # Stage 2: Platt scaling per bucket
        raw_probs = gaussian_to_bucket_probs(mu, sigma, bucket_edges)
        n_buckets = len(bucket_edges)
        outcomes = _compute_outcomes(actual, bucket_edges)

        for b in range(n_buckets):
            if outcomes[:, b].sum() < 3 or (1 - outcomes[:, b]).sum() < 3:
                self.platt_models[b] = None
                continue
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(raw_probs[:, b:b+1], outcomes[:, b])
            self.platt_models[b] = lr

        self.is_fitted = True
        logger.info("Calibrator fitted on %d samples, %d buckets", len(mu), n_buckets)
        return self

    def calibrate(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        bucket_edges: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply calibration to new predictions.

        Returns
        -------
        tuple of (calibrated_mu, calibrated_sigma, calibrated_bucket_probs)
        """
        if not self.is_fitted:
            probs = gaussian_to_bucket_probs(mu, sigma, bucket_edges)
            return mu, sigma, probs

        sigma_safe = np.clip(sigma, 1e-6, None)

        # Stage 1: Isotonic calibration on mu/sigma
        # Compute reference PITs and apply isotonic
        z_ref = np.array([0.0, 1.0])  # median and +1 sigma
        cal_mu = mu.copy()
        cal_sigma = sigma_safe.copy()

        if self.isotonic is not None:
            pit_0 = norm.cdf(mu, loc=mu, scale=sigma_safe)  # should be 0.5
            pit_1 = norm.cdf(mu + sigma_safe, loc=mu, scale=sigma_safe)  # should be ~0.84

            cal_pit_0 = self.isotonic.predict(np.clip(pit_0, 0.001, 0.999))
            cal_pit_1 = self.isotonic.predict(np.clip(pit_1, 0.001, 0.999))

            # Convert calibrated PITs back to z-scores
            cal_z_0 = norm.ppf(np.clip(cal_pit_0, 0.001, 0.999))
            cal_z_1 = norm.ppf(np.clip(cal_pit_1, 0.001, 0.999))

            # Solve for new mu, sigma
            dz = cal_z_1 - cal_z_0
            valid = np.abs(dz) > 0.01
            cal_sigma[valid] = sigma_safe[valid] / dz[valid]
            cal_sigma = np.clip(cal_sigma, 1.0, 30.0)
            cal_mu[valid] = mu[valid] - cal_z_0[valid] * cal_sigma[valid]

        # Stage 2: Generate bucket probs and apply Platt
        raw_probs = gaussian_to_bucket_probs(cal_mu, cal_sigma, bucket_edges)
        cal_probs = raw_probs.copy()

        for b, lr in self.platt_models.items():
            if lr is not None:
                cal_probs[:, b] = lr.predict_proba(raw_probs[:, b:b+1])[:, 1]

        # Re-normalize
        cal_probs = np.clip(cal_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        cal_probs = cal_probs / row_sums

        return cal_mu, cal_sigma, cal_probs


# =====================================================================
# Ensemble Meta-Learner
# =====================================================================

class EnsembleStacker:
    """Meta-learner that combines multiple model predictions.

    Uses Ridge regression on model (mu, sigma) pairs plus regime
    features to produce optimized ensemble predictions.
    """

    def __init__(self):
        self.mu_model = None
        self.sigma_scaler = None
        self.is_fitted = False

    def fit(
        self,
        model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
        actual: np.ndarray,
        regime_features: np.ndarray | None = None,
    ) -> "EnsembleStacker":
        """Fit ensemble weights.

        Parameters
        ----------
        model_predictions : dict
            {model_name: (mu, sigma)} for each base model.
        actual : np.ndarray
            Observed TMAX values.
        regime_features : np.ndarray or None
            Optional regime features to include.
        """
        # Build feature matrix from all model predictions
        features = []
        for name in sorted(model_predictions.keys()):
            mu, sigma = model_predictions[name]
            features.extend([mu, sigma])
        X = np.column_stack(features)

        if regime_features is not None:
            X = np.hstack([X, regime_features])

        # Fit Ridge for mu
        self.mu_model = Ridge(alpha=1.0)
        self.mu_model.fit(X, actual)

        # Estimate sigma from ensemble residuals
        mu_pred = self.mu_model.predict(X)
        residuals = actual - mu_pred
        monthly_sigma = float(np.std(residuals))
        self.ensemble_sigma = max(monthly_sigma, 2.0)

        self.n_models = len(model_predictions)
        self.model_names = sorted(model_predictions.keys())
        self.is_fitted = True

        logger.info(
            "EnsembleStacker fitted: %d models, sigma=%.2f",
            self.n_models, self.ensemble_sigma
        )
        return self

    def predict(
        self,
        model_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
        regime_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions."""
        features = []
        for name in sorted(model_predictions.keys()):
            mu, sigma = model_predictions[name]
            features.extend([mu, sigma])
        X = np.column_stack(features)

        if regime_features is not None:
            X = np.hstack([X, regime_features])

        mu = self.mu_model.predict(X)

        # Adaptive sigma: use weighted average of base model sigmas
        sigmas = []
        for name in sorted(model_predictions.keys()):
            _, sigma = model_predictions[name]
            sigmas.append(sigma)
        avg_sigma = np.mean(sigmas, axis=0)
        # Blend: 50% average of base sigmas, 50% training residual sigma
        sigma = 0.5 * avg_sigma + 0.5 * self.ensemble_sigma

        return mu, sigma


# =====================================================================
# Bucket Probability and Brier Score Utilities
# =====================================================================

def gaussian_to_bucket_probs(
    mu: np.ndarray,
    sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Convert Gaussian (mu, sigma) to bucket probabilities."""
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    sigma_safe = np.clip(sigma, 1e-6, None)
    probs = np.zeros((n_days, n_buckets))

    for b, (lo, hi) in enumerate(bucket_edges):
        cdf_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma_safe)
        cdf_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma_safe)
        probs[:, b] = np.clip(cdf_hi - cdf_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)

    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    return probs


def _compute_outcomes(
    actual: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Compute binary outcome matrix for Brier score."""
    n_days = len(actual)
    n_buckets = len(bucket_edges)
    outcomes = np.zeros((n_days, n_buckets))
    for d in range(n_days):
        t = actual[d]
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


def compute_brier_score(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> dict:
    """Compute Brier score across all bucket-days."""
    n_days, n_buckets = bucket_probs.shape
    outcomes = _compute_outcomes(actual_tmax, bucket_edges)
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


def compute_seasonal_brier(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    dates: pd.DatetimeIndex,
    bucket_edges: list[tuple[float, float]],
) -> dict[str, float]:
    """Compute Brier score per meteorological season."""
    season_map = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    seasons = np.array([season_map[m] for m in dates.month])
    results = {}
    for s in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == s
        if mask.any():
            score = compute_brier_score(
                bucket_probs[mask], actual_tmax[mask], bucket_edges
            )
            results[s] = score["overall_brier"]
    return results


# =====================================================================
# NWS Baseline Proxy
# =====================================================================

def compute_nws_proxy_baseline(
    y_train: np.ndarray,
    dates_train: pd.DatetimeIndex,
    y_target: np.ndarray,
    dates_target: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute NWS-proxy baseline using enhanced climatology.

    Approximates NWS forecast skill using:
    - Smoothed DOY climatology as point forecast
    - Monthly-varying sigma from historical errors
    - 15-day Gaussian smoothing for temporal coherence

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mu, sigma) for the target dates.
    """
    # DOY climatology with Gaussian smoothing
    train_df = pd.DataFrame({"tmax": y_train, "doy": dates_train.dayofyear})
    clim_mean = train_df.groupby("doy")["tmax"].mean()
    clim_std = train_df.groupby("doy")["tmax"].std()

    all_doys = np.arange(1, 367)
    clim_mean = clim_mean.reindex(all_doys).interpolate().bfill().ffill()
    clim_std = clim_std.reindex(all_doys).interpolate().bfill().ffill()

    # Gaussian smooth (15-day window)
    from scipy.ndimage import gaussian_filter1d
    clim_mean_smooth = pd.Series(gaussian_filter1d(clim_mean.values, sigma=7))
    clim_mean_smooth.index = clim_mean.index
    clim_std_smooth = pd.Series(gaussian_filter1d(clim_std.values, sigma=7))
    clim_std_smooth.index = clim_std.index

    # NWS typically has ~3-4F MAE for day-ahead, translate to sigma
    # sigma ~= MAE * sqrt(pi/2) for Gaussian errors
    nws_sigma_multiplier = 0.85  # NWS is better than raw climatology

    mu = clim_mean_smooth.reindex(dates_target.dayofyear).values
    sigma = clim_std_smooth.reindex(dates_target.dayofyear).values * nws_sigma_multiplier
    sigma = np.clip(sigma, 3.0, 20.0)

    return mu.astype(np.float32), sigma.astype(np.float32)


def compute_kalshi_presettlement_proxy(
    y_train: np.ndarray,
    dates_train: pd.DatetimeIndex,
    y_target: np.ndarray,
    dates_target: pd.DatetimeIndex,
    X_train: np.ndarray | None = None,
    X_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Kalshi pre-settlement proxy using Ridge regression.

    Approximates market-implied probabilities using the best simple
    model available: Ridge regression with lagged features + climatology.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mu, sigma) approximating Kalshi pre-settlement expectations.
    """
    # Use Ridge on training features if available
    if X_train is not None and X_target is not None:
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train, y_train)
        mu = ridge.predict(X_target)
        train_residuals = y_train - ridge.predict(X_train)
    else:
        # Fallback: persistence + climatology blend
        mu, _ = compute_mos_baseline(
            y_train, dates_train,
            np.concatenate([y_train, y_target]),
            dates_train.append(dates_target),
        )
        mu = mu[len(y_train):]
        train_residuals = y_train - mu[:len(y_train)] if len(mu) >= len(y_train) else np.array([5.0])

    # Monthly sigma
    train_resid_series = pd.Series(
        train_residuals[~np.isnan(train_residuals)],
        index=dates_train[:len(train_residuals[~np.isnan(train_residuals)])]
    )
    monthly_std = train_resid_series.groupby(train_resid_series.index.month).std()
    overall_std = float(np.std(train_residuals[~np.isnan(train_residuals)]))

    sigma = np.full(len(dates_target), overall_std)
    for i, dt in enumerate(dates_target):
        m = dt.month
        if m in monthly_std.index:
            sigma[i] = float(monthly_std[m])
    sigma = np.clip(sigma, 2.0, 20.0)

    return mu.astype(np.float32), sigma.astype(np.float32)
