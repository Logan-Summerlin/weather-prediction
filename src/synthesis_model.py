"""
Synthesis Model: Meta-Learner Combining Station Model and NWP Forecasts.

Fuses probabilistic outputs from the wind-gated station attention model
with numerical weather prediction (NWP) forecasts to produce a superior
calibrated predictive distribution for NYC daily maximum temperature.

Architecture overview:
  1. Concatenate station model (mu, sigma), NWP features (tmax, t850,
     wind, cloud, mslp, precip), derived signals (station-NWP gap,
     bias, ensemble spread), and seasonal encoding (sin/cos day).
  2. Feed through a moderate MLP (2-3 hidden layers, 64-128 neurons)
     with batch normalisation and dropout.
  3. Output head:
     - Gaussian mode: predicted mu and log_sigma for a calibrated
       heteroscedastic Gaussian distribution.
     - Quantile mode: predicted quantile values (e.g., 0.025, 0.1,
       0.25, 0.5, 0.75, 0.9, 0.975) for non-parametric intervals.

The model learns to weight station-model vs NWP information adaptively,
increasing NWP reliance when station uncertainty is high or station-NWP
disagreement signals a regime where one source is more trustworthy.

Training utilities:
  - ``SynthesisTrainer``: full training loop with CRPS/pinball loss,
    early stopping, LR scheduling, checkpointing, and history logging.
  - ``prepare_synthesis_data()``: assembles the combined feature matrix
    from station predictions, NWP features, and observations.
  - ``evaluate_synthesis()``: computes MAE, RMSE, R-squared, CRPS, bias,
    coverage calibration, and generates comparison plots.
"""

import csv
import logging
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.crps_loss import GaussianCRPSLoss, PinballLoss, CombinedCRPSMAELoss

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

# Default synthesis feature names (order matters for the model input)
SYNTHESIS_NWP_FEATURES = [
    "nwp_tmax",
    "nwp_t850",
    "nwp_wind_speed",
    "nwp_wind_dir",
    "nwp_cloud_cover",
    "nwp_mslp",
    "nwp_precip",
]

SYNTHESIS_DERIVED_FEATURES = [
    "nwp_ensemble_spread",
    "station_nwp_gap",
    "abs_station_nwp_gap",
    "nwp_bias_7d",
]

SYNTHESIS_SEASON_FEATURES = [
    "sin_day",
    "cos_day",
]

SYNTHESIS_STATION_FEATURES = [
    "station_mu",
    "station_sigma",
]

DEFAULT_QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

# Total default input features = 2 (station) + 7 (NWP) + 4 (derived) + 2 (season) = 15
DEFAULT_N_FEATURES = (
    len(SYNTHESIS_STATION_FEATURES)
    + len(SYNTHESIS_NWP_FEATURES)
    + len(SYNTHESIS_DERIVED_FEATURES)
    + len(SYNTHESIS_SEASON_FEATURES)
)


# ===========================================================================
# SynthesisModel
# ===========================================================================

class SynthesisModel(nn.Module):
    """Meta-learner MLP that combines station model and NWP forecasts.

    Takes station model probabilistic outputs (mu, sigma), NWP features,
    derived disagreement signals, and seasonal encoding as input.
    Produces either a calibrated Gaussian (mu, sigma) or quantile
    predictions.

    Parameters
    ----------
    n_features : int
        Number of input features (station + NWP + derived + season).
    hidden_sizes : list[int]
        Sizes of hidden layers. Default ``[128, 64, 32]``.
    output_mode : str
        ``"gaussian"`` for heteroscedastic Gaussian output, or
        ``"quantile"`` for quantile regression output.
    quantiles : list[float] or None
        Quantile levels for quantile mode. Default
        ``[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]``.
    dropout : float
        Dropout probability. Default ``0.15``.
    use_batch_norm : bool
        Whether to use batch normalisation. Default ``True``.

    Examples
    --------
    >>> model = SynthesisModel(n_features=15, output_mode="gaussian")
    >>> x = torch.randn(32, 15)
    >>> out = model(x)
    >>> out["mu"].shape
    torch.Size([32, 1])
    >>> out["sigma"].shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        n_features: int = DEFAULT_N_FEATURES,
        hidden_sizes: Optional[list[int]] = None,
        output_mode: str = "gaussian",
        quantiles: Optional[list[float]] = None,
        dropout: float = 0.15,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if output_mode not in ("gaussian", "quantile"):
            raise ValueError(
                f"output_mode must be 'gaussian' or 'quantile', "
                f"got '{output_mode}'"
            )

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        if quantiles is None:
            quantiles = list(DEFAULT_QUANTILES)

        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.output_mode = output_mode
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        # ---- Build shared trunk ----
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

        # ---- Output head ----
        trunk_out_dim = hidden_sizes[-1] if hidden_sizes else n_features

        if output_mode == "gaussian":
            self.mu_head = nn.Linear(trunk_out_dim, 1)
            self.log_sigma_head = nn.Linear(trunk_out_dim, 1)
        else:
            self.quantile_head = nn.Linear(trunk_out_dim, self.n_quantiles)

        # ---- Initialise weights ----
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SynthesisModel created: n_features=%d, hidden=%s, "
            "mode=%s, dropout=%.2f, bn=%s, params=%d",
            n_features, hidden_sizes, output_mode, dropout,
            use_batch_norm, n_params,
        )

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        return (
            f"Synthesis(feat={self.n_features},"
            f"h={'x'.join(str(h) for h in self.hidden_sizes)},"
            f"mode={self.output_mode})"
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features. Shape ``(batch, n_features)``.

        Returns
        -------
        dict[str, torch.Tensor]
            For gaussian mode:
              - ``"prediction"``: ``(batch, 1)`` -- predicted mean (mu).
              - ``"mu"``: ``(batch, 1)`` -- same as prediction.
              - ``"log_sigma"``: ``(batch, 1)`` -- log standard deviation.
              - ``"sigma"``: ``(batch, 1)`` -- exp(log_sigma), clamped.
            For quantile mode:
              - ``"prediction"``: ``(batch, 1)`` -- median quantile.
              - ``"quantiles"``: ``(batch, n_quantiles)`` -- all quantiles.
        """
        hidden = self.trunk(x)  # (B, trunk_out_dim)

        result: dict[str, torch.Tensor] = {}

        if self.output_mode == "gaussian":
            mu = self.mu_head(hidden)  # (B, 1)
            log_sigma = self.log_sigma_head(hidden)  # (B, 1)
            log_sigma = log_sigma.clamp(min=-10.0, max=5.0)
            sigma = torch.exp(log_sigma)

            result["prediction"] = mu
            result["mu"] = mu
            result["log_sigma"] = log_sigma
            result["sigma"] = sigma
        else:
            quantile_preds = self.quantile_head(hidden)  # (B, Q)
            result["quantiles"] = quantile_preds
            # Prediction = median quantile
            median_idx = self._get_median_index()
            result["prediction"] = quantile_preds[:, median_idx].unsqueeze(1)

        return result

    def _get_median_index(self) -> int:
        """Return the index of the quantile closest to 0.5."""
        diffs = [abs(q - 0.5) for q in self.quantiles]
        return int(np.argmin(diffs))


# ===========================================================================
# SynthesisDataset
# ===========================================================================

class SynthesisDataset(Dataset):
    """PyTorch Dataset for synthesis model training.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix. Shape ``(N, n_features)``.
    targets : np.ndarray
        Target values. Shape ``(N,)``.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
        }


# ===========================================================================
# Data Preparation
# ===========================================================================

def prepare_synthesis_data(
    station_predictions: pd.DataFrame,
    nwp_features: pd.DataFrame,
    observations: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "date",
    target_col: str = "obs_tmax",
    station_mu_col: str = "station_mu",
    station_sigma_col: str = "station_sigma",
    nwp_tmax_col: str = "nwp_tmax",
) -> dict:
    """Prepare combined feature matrix for the synthesis model.

    Aligns station model predictions with NWP features and observed
    targets by date.  Computes derived features (station-NWP gap,
    absolute gap).  Applies chronological train/val/test splitting
    and StandardScaler (fit on training set only).

    Parameters
    ----------
    station_predictions : pd.DataFrame
        Must contain columns: ``date``, ``station_mu``, ``station_sigma``.
    nwp_features : pd.DataFrame
        Must contain columns: ``date``, plus NWP feature columns
        (``nwp_tmax``, ``nwp_t850``, ``nwp_wind_speed``, etc.).
    observations : pd.DataFrame
        Must contain columns: ``date``, ``obs_tmax`` (observed TMAX in F).
    train_ratio : float
        Fraction of data for training. Default 0.70.
    val_ratio : float
        Fraction of data for validation. Default 0.15.
    date_col : str
        Name of the date column. Default ``"date"``.
    target_col : str
        Name of the target column in observations. Default ``"obs_tmax"``.
    station_mu_col : str
        Column name for station model mu. Default ``"station_mu"``.
    station_sigma_col : str
        Column name for station model sigma. Default ``"station_sigma"``.
    nwp_tmax_col : str
        Column name for NWP TMAX forecast. Default ``"nwp_tmax"``.

    Returns
    -------
    dict
        Dictionary with keys:
          - ``"X_train"``, ``"X_val"``, ``"X_test"``: scaled feature
            arrays as np.ndarray, shape ``(N, n_features)``.
          - ``"y_train"``, ``"y_val"``, ``"y_test"``: target arrays.
          - ``"dates_train"``, ``"dates_val"``, ``"dates_test"``:
            date arrays for each split.
          - ``"feature_names"``: list of feature column names.
          - ``"scaler"``: fitted StandardScaler instance.
          - ``"n_features"``: int, number of features.
          - ``"n_train"``, ``"n_val"``, ``"n_test"``: split sizes.

    Raises
    ------
    ValueError
        If required columns are missing or no overlapping dates exist.
    """
    # --- Validate inputs ---
    _validate_columns(station_predictions, [date_col, station_mu_col, station_sigma_col],
                       "station_predictions")
    _validate_columns(observations, [date_col, target_col], "observations")

    # --- Convert dates ---
    station_df = station_predictions.copy()
    nwp_df = nwp_features.copy()
    obs_df = observations.copy()

    station_df[date_col] = pd.to_datetime(station_df[date_col])
    nwp_df[date_col] = pd.to_datetime(nwp_df[date_col])
    obs_df[date_col] = pd.to_datetime(obs_df[date_col])

    # --- Merge on date ---
    merged = station_df.merge(obs_df, on=date_col, how="inner")
    if not nwp_df.empty:
        merged = merged.merge(nwp_df, on=date_col, how="left")
    else:
        # Add NaN NWP columns if NWP data is empty
        for col in SYNTHESIS_NWP_FEATURES + SYNTHESIS_DERIVED_FEATURES:
            if col not in merged.columns:
                merged[col] = np.nan

    if len(merged) == 0:
        raise ValueError(
            "No overlapping dates between station predictions, "
            "NWP features, and observations."
        )

    # Sort chronologically
    merged = merged.sort_values(date_col).reset_index(drop=True)

    # --- Compute derived features ---
    if station_mu_col in merged.columns and nwp_tmax_col in merged.columns:
        merged["station_nwp_gap"] = (
            merged[station_mu_col] - merged[nwp_tmax_col]
        )
        merged["abs_station_nwp_gap"] = merged["station_nwp_gap"].abs()
    else:
        merged["station_nwp_gap"] = 0.0
        merged["abs_station_nwp_gap"] = 0.0

    # Ensure required columns exist (fill with NaN if missing)
    for col in (SYNTHESIS_NWP_FEATURES + SYNTHESIS_DERIVED_FEATURES
                + SYNTHESIS_SEASON_FEATURES):
        if col not in merged.columns:
            merged[col] = np.nan

    # --- Build feature matrix ---
    feature_cols = (
        SYNTHESIS_STATION_FEATURES
        + SYNTHESIS_NWP_FEATURES
        + SYNTHESIS_DERIVED_FEATURES
        + SYNTHESIS_SEASON_FEATURES
    )
    # Use only columns that exist
    feature_cols = [c for c in feature_cols if c in merged.columns]

    X = merged[feature_cols].values.astype(np.float32)
    y = merged[target_col].values.astype(np.float32)
    dates = merged[date_col].values

    # --- Handle missing NWP data: impute with column mean from training ---
    # (We compute the split first, impute using training means)

    # --- Chronological split ---
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Ensure at least 1 sample per split
    n_train = max(1, n_train)
    n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val
    n_test = max(1, n_test)

    X_train = X[:n_train]
    X_val = X[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]

    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    dates_train = dates[:n_train]
    dates_val = dates[n_train:n_train + n_val]
    dates_test = dates[n_train + n_val:]

    # --- Impute missing values using training-set column means ---
    train_means = np.nanmean(X_train, axis=0)
    # Replace NaN means with 0 (column entirely NaN)
    train_means = np.where(np.isnan(train_means), 0.0, train_means)

    for split_X in [X_train, X_val, X_test]:
        nan_mask = np.isnan(split_X)
        if nan_mask.any():
            # Broadcast train_means across rows
            imputed = np.where(nan_mask, np.tile(train_means, (split_X.shape[0], 1)), split_X)
            split_X[:] = imputed

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Prepared synthesis data: %d total rows, %d features, "
        "train=%d, val=%d, test=%d",
        n_total, len(feature_cols), n_train, n_val, n_test,
    )

    return {
        "X_train": X_train_scaled.astype(np.float32),
        "X_val": X_val_scaled.astype(np.float32),
        "X_test": X_test_scaled.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "dates_train": dates_train,
        "dates_val": dates_val,
        "dates_test": dates_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "n_features": len(feature_cols),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }


def _validate_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    """Check that all required columns are present in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : list[str]
        Required column names.
    name : str
        Name of the DataFrame (for error messages).

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# ===========================================================================
# SynthesisTrainer
# ===========================================================================

class SynthesisTrainer:
    """Training pipeline for the SynthesisModel.

    Supports CRPS loss (gaussian mode), pinball loss (quantile mode),
    early stopping, LR scheduling, model checkpointing, and CSV
    history logging.

    Parameters
    ----------
    model : SynthesisModel
        The synthesis model to train.
    output_dir : str
        Directory for checkpoints, history CSV, and training plots.
    learning_rate : float
        Initial learning rate. Default ``0.001``.
    max_epochs : int
        Maximum training epochs. Default ``200``.
    early_stopping_patience : int
        Epochs without improvement before stopping. Default ``15``.
    batch_size : int
        Mini-batch size. Default ``64``.
    loss_type : str
        Loss function: ``"crps"`` (Gaussian CRPS), ``"combined"``
        (CRPS + MAE), or ``"pinball"`` (quantile). Default ``"combined"``.
    crps_weight : float
        Weight for CRPS component in combined loss. Default ``0.7``.
    mae_weight : float
        Weight for MAE component in combined loss. Default ``0.3``.
    device : str
        Torch device. Default ``"cpu"``.
    model_name : str
        Name for logging and file naming. Default ``"synthesis"``.

    Examples
    --------
    >>> model = SynthesisModel(n_features=15, output_mode="gaussian")
    >>> trainer = SynthesisTrainer(model, output_dir="/tmp/synthesis")
    >>> result = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        model: SynthesisModel,
        output_dir: str,
        learning_rate: float = 0.001,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
        batch_size: int = 64,
        loss_type: str = "combined",
        crps_weight: float = 0.7,
        mae_weight: float = 0.3,
        device: str = "cpu",
        model_name: str = "synthesis",
    ):
        self.model = model.to(device)
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.device = device
        self.model_name = model_name

        os.makedirs(output_dir, exist_ok=True)

        # ---- Build loss function ----
        if model.output_mode == "quantile":
            self.loss_fn = PinballLoss(
                quantiles=model.quantiles, reduction="mean"
            )
        elif loss_type == "crps":
            self.loss_fn = GaussianCRPSLoss(reduction="mean")
        elif loss_type == "combined":
            self.loss_fn = CombinedCRPSMAELoss(
                crps_weight=crps_weight, mae_weight=mae_weight
            )
        else:
            # Default to combined for gaussian mode
            logger.warning(
                "loss_type='%s' not recognised for mode='%s'; "
                "falling back to combined CRPS+MAE",
                loss_type, model.output_mode,
            )
            self.loss_fn = CombinedCRPSMAELoss(
                crps_weight=crps_weight, mae_weight=mae_weight
            )

        # ---- Optimiser & scheduler ----
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5,
        )

        # ---- Checkpoint path ----
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        self.checkpoint_path = os.path.join(
            output_dir, f"best_{safe_name}.pt"
        )
        self.history_path = os.path.join(
            output_dir, f"history_{safe_name}.csv"
        )
        self.plot_path = os.path.join(
            output_dir, f"curves_{safe_name}.png"
        )

        logger.info(
            "SynthesisTrainer initialised: lr=%.6f, max_epochs=%d, "
            "patience=%d, batch=%d, loss=%s, device=%s",
            learning_rate, max_epochs, early_stopping_patience,
            batch_size, loss_type, device,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        X_train : np.ndarray
            Training features. Shape ``(N_train, n_features)``.
        y_train : np.ndarray
            Training targets. Shape ``(N_train,)``.
        X_val : np.ndarray
            Validation features.
        y_val : np.ndarray
            Validation targets.

        Returns
        -------
        dict
            Dictionary with keys:
              - ``"model"``: trained model (best checkpoint loaded).
              - ``"history"``: list of epoch dicts.
              - ``"best_epoch"``: int.
              - ``"best_val_loss"``: float.
              - ``"best_val_mae"``: float.
        """
        # ---- Create DataLoaders ----
        train_dataset = SynthesisDataset(X_train, y_train)
        val_dataset = SynthesisDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        # ---- Training loop ----
        history: list[dict] = []
        best_val_loss = float("inf")
        best_val_mae = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        logger.info("=" * 60)
        logger.info(
            "Training '%s' (%s mode, %s loss)",
            self.model_name, self.model.output_mode, self.loss_type,
        )
        logger.info("=" * 60)

        for epoch in range(1, self.max_epochs + 1):
            # -- Train one epoch --
            train_loss = self._train_epoch(train_loader)

            # -- Validate --
            val_loss, val_preds, val_actuals, val_sigmas = self._validate(
                val_loader
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # -- Compute MAE --
            val_mae = float(np.mean(np.abs(val_preds - val_actuals)))

            # -- Record history --
            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "lr": current_lr,
            }
            if val_sigmas is not None:
                entry["mean_sigma"] = float(np.mean(val_sigmas))
            history.append(entry)

            # -- Check improvement --
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                if epoch <= 5 or epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d | Train: %.4f | Val: %.4f | "
                        "MAE: %.3f F | * BEST *",
                        epoch, train_loss, val_loss, val_mae,
                    )
            else:
                epochs_without_improvement += 1
                if epoch <= 5 or epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d | Train: %.4f | Val: %.4f | "
                        "MAE: %.3f F | No improvement (%d/%d)",
                        epoch, train_loss, val_loss, val_mae,
                        epochs_without_improvement, self.patience,
                    )

            # -- Early stopping --
            if epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for "
                    "%d epochs)",
                    epoch, self.patience,
                )
                break

        # --- Load best checkpoint ---
        if os.path.isfile(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, weights_only=True)
            )
            logger.info(
                "Loaded best model from epoch %d (Val MAE: %.3f F)",
                best_epoch, best_val_mae,
            )

        # --- Save history ---
        self._save_history(history)

        # --- Plot training curves ---
        self._plot_curves(history)

        logger.info(
            "Training '%s' complete. Best MAE: %.3f F (epoch %d)",
            self.model_name, best_val_mae, best_epoch,
        )

        return {
            "model": self.model,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.

        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch["features"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(features)

            loss = self._compute_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(
        self,
        val_loader: DataLoader,
    ) -> tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Evaluate on the validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]
            (avg_loss, predictions, actuals, sigmas).
            sigmas is None for quantile mode.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[np.ndarray] = []
        all_actuals: list[np.ndarray] = []
        all_sigmas: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(features)
                loss = self._compute_loss(output, target)

                total_loss += loss.item()
                n_batches += 1

                all_preds.append(
                    output["prediction"].cpu().numpy().ravel()
                )
                all_actuals.append(target.cpu().numpy().ravel())

                if "sigma" in output:
                    all_sigmas.append(
                        output["sigma"].cpu().numpy().ravel()
                    )

        avg_loss = total_loss / max(n_batches, 1)
        preds = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)
        sigmas = np.concatenate(all_sigmas) if all_sigmas else None

        return avg_loss, preds, actuals, sigmas

    def _compute_loss(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on model output mode.

        Parameters
        ----------
        output : dict
            Model forward pass output.
        target : torch.Tensor
            Target values. Shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if self.model.output_mode == "gaussian":
            result = self.loss_fn(
                output["mu"].squeeze(-1),
                output["sigma"].squeeze(-1),
                target,
            )
            if isinstance(result, dict):
                return result["loss"]
            return result
        else:
            # Quantile mode
            return self.loss_fn(output["quantiles"], target)

    def predict(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Generate predictions for a feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix. Shape ``(N, n_features)``.

        Returns
        -------
        dict[str, np.ndarray]
            For gaussian mode:
              - ``"mu"``: predicted mean, shape ``(N,)``.
              - ``"sigma"``: predicted std, shape ``(N,)``.
            For quantile mode:
              - ``"quantiles"``: shape ``(N, n_quantiles)``.
              - ``"median"``: median prediction, shape ``(N,)``.
            Always includes:
              - ``"prediction"``: point prediction, shape ``(N,)``.
        """
        self.model.eval()
        dataset = SynthesisDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_outputs: dict[str, list[np.ndarray]] = {}

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                output = self.model(features)

                for key, val in output.items():
                    if key not in all_outputs:
                        all_outputs[key] = []
                    all_outputs[key].append(val.cpu().numpy())

        result: dict[str, np.ndarray] = {}
        for key, arrays in all_outputs.items():
            concatenated = np.concatenate(arrays, axis=0)
            if concatenated.ndim == 2 and concatenated.shape[1] == 1:
                concatenated = concatenated.ravel()
            result[key] = concatenated

        if self.model.output_mode == "quantile" and "quantiles" in result:
            median_idx = self.model._get_median_index()
            result["median"] = result["quantiles"][:, median_idx]

        return result

    def _save_history(self, history: list[dict]) -> None:
        """Save training history to CSV.

        Parameters
        ----------
        history : list[dict]
            Training history.
        """
        if not history:
            return

        fieldnames = list(history[0].keys())
        for entry in history:
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(self.history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        logger.info(
            "Saved training history (%d epochs) to %s",
            len(history), self.history_path,
        )

    def _plot_curves(self, history: list[dict]) -> None:
        """Plot training and validation curves.

        Parameters
        ----------
        history : list[dict]
            Training history.
        """
        if not history:
            return

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_maes = [h["val_mae"] for h in history]

        has_sigma = "mean_sigma" in history[0]
        n_cols = 3 if has_sigma else 2

        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        # Loss curves
        axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
        axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Val MAE
        axes[1].plot(
            epochs, val_maes, label="Val MAE",
            linewidth=1.5, color="#2ca02c",
        )
        best_idx = int(np.argmin(val_maes))
        axes[1].axvline(
            epochs[best_idx], color="red", linestyle="--", alpha=0.7
        )
        axes[1].scatter(
            [epochs[best_idx]], [val_maes[best_idx]],
            color="red", zorder=5, s=50,
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (degF)")
        axes[1].set_title("Validation MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Mean sigma
        if has_sigma:
            sigmas = [h.get("mean_sigma", 0) for h in history]
            axes[2].plot(
                epochs, sigmas, label="Mean sigma",
                linewidth=1.5, color="#ff7f0e",
            )
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("sigma (degF)")
            axes[2].set_title("Predicted Uncertainty")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Training Curves: {self.model_name}", fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved training curves to %s", self.plot_path)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_synthesis(
    trainer: SynthesisTrainer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dates_test: Optional[np.ndarray] = None,
    station_only_preds: Optional[np.ndarray] = None,
    nwp_only_preds: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Evaluate synthesis model performance with comprehensive metrics.

    Compares the synthesis model against station-only and NWP-only
    baselines (if provided).  Computes calibration metrics for
    probabilistic outputs.

    Parameters
    ----------
    trainer : SynthesisTrainer
        Trained synthesis trainer (with loaded best model).
    X_test : np.ndarray
        Test feature matrix. Shape ``(N, n_features)``.
    y_test : np.ndarray
        Test targets (observed TMAX). Shape ``(N,)``.
    dates_test : np.ndarray or None
        Test dates for seasonal breakdown.
    station_only_preds : np.ndarray or None
        Station-only model predictions for comparison.
    nwp_only_preds : np.ndarray or None
        NWP-only predictions for comparison.
    output_dir : str or None
        Directory to save plots.  Uses trainer's output_dir if None.

    Returns
    -------
    dict
        Dictionary with keys:
          - ``"synthesis_metrics"``: dict of MAE, RMSE, R2, bias.
          - ``"station_metrics"``: dict (if station_only_preds provided).
          - ``"nwp_metrics"``: dict (if nwp_only_preds provided).
          - ``"coverage"``: dict of coverage at various intervals
            (gaussian mode) or quantile calibration (quantile mode).
          - ``"seasonal"``: dict of seasonal MAE breakdown.
          - ``"predictions"``: synthesis model predictions dict.
    """
    if output_dir is None:
        output_dir = trainer.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # --- Generate predictions ---
    predictions = trainer.predict(X_test)
    synthesis_preds = predictions["prediction"]

    # --- Compute metrics ---
    result: dict = {}

    result["synthesis_metrics"] = _compute_metrics(
        synthesis_preds, y_test, name="Synthesis"
    )

    if station_only_preds is not None:
        result["station_metrics"] = _compute_metrics(
            station_only_preds, y_test, name="Station-Only"
        )

    if nwp_only_preds is not None:
        result["nwp_metrics"] = _compute_metrics(
            nwp_only_preds, y_test, name="NWP-Only"
        )

    # --- Coverage / calibration ---
    if trainer.model.output_mode == "gaussian" and "sigma" in predictions:
        result["coverage"] = _compute_gaussian_coverage(
            predictions["mu"], predictions["sigma"], y_test
        )
    elif trainer.model.output_mode == "quantile" and "quantiles" in predictions:
        result["coverage"] = _compute_quantile_calibration(
            predictions["quantiles"], y_test, trainer.model.quantiles
        )
    else:
        result["coverage"] = {}

    # --- Seasonal breakdown ---
    if dates_test is not None:
        result["seasonal"] = _compute_seasonal_metrics(
            synthesis_preds, y_test, dates_test
        )
    else:
        result["seasonal"] = {}

    result["predictions"] = predictions

    # --- Generate plots ---
    _plot_evaluation(
        synthesis_preds, y_test, predictions,
        trainer.model.output_mode,
        station_only_preds=station_only_preds,
        nwp_only_preds=nwp_only_preds,
        output_dir=output_dir,
    )

    # --- Log summary ---
    sm = result["synthesis_metrics"]
    logger.info("=" * 50)
    logger.info("Synthesis Model Evaluation")
    logger.info("  MAE:  %.3f F", sm["mae"])
    logger.info("  RMSE: %.3f F", sm["rmse"])
    logger.info("  R2:   %.3f", sm["r2"])
    logger.info("  Bias: %.3f F", sm["bias"])

    if "coverage" in result and result["coverage"]:
        logger.info("  Coverage: %s", result["coverage"])

    if station_only_preds is not None:
        stm = result["station_metrics"]
        logger.info("  Station-Only MAE: %.3f F", stm["mae"])
        improvement = stm["mae"] - sm["mae"]
        logger.info(
            "  Synthesis improvement over station: %.3f F (%.1f%%)",
            improvement, 100 * improvement / stm["mae"] if stm["mae"] > 0 else 0,
        )

    logger.info("=" * 50)

    return result


def _compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    name: str = "",
) -> dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values. Shape ``(N,)``.
    targets : np.ndarray
        Observed values. Shape ``(N,)``.
    name : str
        Label for logging.

    Returns
    -------
    dict[str, float]
        Keys: mae, rmse, r2, bias, max_error.
    """
    residuals = predictions - targets
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    max_error = float(np.max(np.abs(residuals)))

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "max_error": max_error,
        "name": name,
    }


def _compute_gaussian_coverage(
    mu: np.ndarray,
    sigma: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute coverage for Gaussian prediction intervals.

    Parameters
    ----------
    mu : np.ndarray
        Predicted means. Shape ``(N,)``.
    sigma : np.ndarray
        Predicted standard deviations. Shape ``(N,)``.
    targets : np.ndarray
        Observed values. Shape ``(N,)``.

    Returns
    -------
    dict[str, float]
        Coverage fractions at 50%, 90%, 95% nominal levels.
    """
    from scipy.stats import norm

    z_scores = {
        "50%": norm.ppf(0.75),    # ~0.674
        "90%": norm.ppf(0.95),    # ~1.645
        "95%": norm.ppf(0.975),   # ~1.960
    }

    coverage = {}
    for label, z in z_scores.items():
        lower = mu - z * sigma
        upper = mu + z * sigma
        within = np.sum((targets >= lower) & (targets <= upper))
        coverage[f"coverage_{label}"] = float(within / len(targets))

    return coverage


def _compute_quantile_calibration(
    quantile_preds: np.ndarray,
    targets: np.ndarray,
    quantiles: list[float],
) -> dict[str, float]:
    """Compute calibration for quantile predictions.

    For each predicted quantile q, the actual fraction of observations
    below the predicted quantile should be approximately q.

    Parameters
    ----------
    quantile_preds : np.ndarray
        Shape ``(N, n_quantiles)``.
    targets : np.ndarray
        Shape ``(N,)``.
    quantiles : list[float]
        Nominal quantile levels.

    Returns
    -------
    dict[str, float]
        Actual coverage for each nominal quantile.
    """
    calibration = {}
    for i, q in enumerate(quantiles):
        actual_below = np.mean(targets <= quantile_preds[:, i])
        calibration[f"q{q:.3f}_actual"] = float(actual_below)
        calibration[f"q{q:.3f}_nominal"] = q

    return calibration


def _compute_seasonal_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
) -> dict[str, float]:
    """Compute MAE by meteorological season.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values. Shape ``(N,)``.
    targets : np.ndarray
        Observed values. Shape ``(N,)``.
    dates : np.ndarray
        Dates corresponding to each prediction.

    Returns
    -------
    dict[str, float]
        MAE for each season.
    """
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }

    dates_pd = pd.to_datetime(dates)
    months = dates_pd.month

    result = {}
    for season_name in ["Winter", "Spring", "Summer", "Fall"]:
        mask = np.array([season_map.get(m, "Unknown") == season_name for m in months])
        if mask.any():
            result[f"mae_{season_name.lower()}"] = float(
                np.mean(np.abs(predictions[mask] - targets[mask]))
            )

    return result


def _plot_evaluation(
    synthesis_preds: np.ndarray,
    targets: np.ndarray,
    full_predictions: dict[str, np.ndarray],
    output_mode: str,
    station_only_preds: Optional[np.ndarray] = None,
    nwp_only_preds: Optional[np.ndarray] = None,
    output_dir: str = ".",
) -> None:
    """Generate evaluation comparison plots.

    Creates:
      1. Scatter plot: predicted vs actual
      2. Residual histogram
      3. Coverage plot (for probabilistic outputs)
      4. Model comparison bar chart (if baselines provided)

    Parameters
    ----------
    synthesis_preds : np.ndarray
        Synthesis model point predictions.
    targets : np.ndarray
        Observed values.
    full_predictions : dict
        Full prediction dict from the model.
    output_mode : str
        Model output mode ("gaussian" or "quantile").
    station_only_preds : np.ndarray or None
        Station-only baseline predictions.
    nwp_only_preds : np.ndarray or None
        NWP-only baseline predictions.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Scatter: Predicted vs Actual ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, synthesis_preds, alpha=0.5, s=15, label="Synthesis")
    min_val = min(targets.min(), synthesis_preds.min()) - 5
    max_val = max(targets.max(), synthesis_preds.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    ax.set_xlabel("Observed TMAX (F)")
    ax.set_ylabel("Predicted TMAX (F)")
    ax.set_title("Synthesis Model: Predicted vs Observed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "synthesis_scatter.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ---- 2. Residual histogram ----
    residuals = synthesis_preds - targets
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (F)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residuals (bias={np.mean(residuals):.2f}F, "
        f"MAE={np.mean(np.abs(residuals)):.2f}F)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "synthesis_residuals.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ---- 3. Coverage / uncertainty plot ----
    if output_mode == "gaussian" and "sigma" in full_predictions:
        mu = full_predictions["mu"]
        sigma = full_predictions["sigma"]
        sort_idx = np.argsort(targets)
        sorted_targets = targets[sort_idx]
        sorted_mu = mu[sort_idx]
        sorted_sigma = sigma[sort_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        x_axis = np.arange(len(sorted_targets))
        ax.plot(x_axis, sorted_targets, "k.", markersize=3, label="Observed")
        ax.plot(x_axis, sorted_mu, "b-", linewidth=0.8, label="Predicted mu")
        ax.fill_between(
            x_axis,
            sorted_mu - 1.96 * sorted_sigma,
            sorted_mu + 1.96 * sorted_sigma,
            alpha=0.2, color="blue", label="95% CI",
        )
        ax.set_xlabel("Sample (sorted by observed)")
        ax.set_ylabel("TMAX (F)")
        ax.set_title("Synthesis Model: Uncertainty Envelope")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "synthesis_uncertainty.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # ---- 4. Model comparison bar chart ----
    model_names = ["Synthesis"]
    mae_values = [float(np.mean(np.abs(synthesis_preds - targets)))]

    if station_only_preds is not None:
        model_names.append("Station-Only")
        mae_values.append(float(np.mean(np.abs(station_only_preds - targets))))
    if nwp_only_preds is not None:
        model_names.append("NWP-Only")
        mae_values.append(float(np.mean(np.abs(nwp_only_preds - targets))))

    if len(model_names) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e"][:len(model_names)]
        bars = ax.bar(model_names, mae_values, color=colors, edgecolor="black")
        for bar, val in zip(bars, mae_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=11,
            )
        ax.set_ylabel("MAE (F)")
        ax.set_title("Model Comparison: Test-Set MAE")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "synthesis_comparison.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    logger.info("Saved evaluation plots to %s", output_dir)
