"""
Phase 1 Training Pipeline for Wind-Gated Attention Model.

Provides utilities to:
  1. Reshape flat feature matrices into the structured tensors expected
     by WindGatedAttentionModel (station features, metadata, global
     context, bearings, wind direction, mask).
  2. Train the wind-gated attention model with CRPS or point-prediction
     losses, including delta-T target support, early stopping, LR
     scheduling, and model checkpointing.
  3. Evaluate on validation/test sets with CRPS, MAE, and reconstructed
     TMAX metrics.
  4. Generate training-curve plots.

This module mirrors the conventions of ``train_v2.py`` but adapts them
for the structured-input attention model and probabilistic losses.
"""

import os
import sys
import csv
import logging
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from src.wind_gated_attention import WindGatedAttentionModel
from src.crps_loss import GaussianCRPSLoss, CombinedCRPSMAELoss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Custom Dataset for Structured Attention Input
# ===========================================================================

class AttentionDataset(Dataset):
    """PyTorch Dataset that provides structured tensors for the model.

    Each sample contains station features, station metadata, global
    context, bearings, wind direction, station mask, and target.

    Parameters
    ----------
    station_features : np.ndarray
        Shape ``(N, n_stations, n_station_features)``.
    station_metadata : np.ndarray
        Shape ``(N, n_stations, n_metadata_features)``.
    global_context : np.ndarray
        Shape ``(N, n_global_features)``.
    station_bearings : np.ndarray
        Shape ``(N, n_stations)``.
    wind_direction : np.ndarray
        Shape ``(N,)``.
    station_mask : np.ndarray
        Shape ``(N, n_stations)``.
    targets : np.ndarray
        Shape ``(N,)``.
    """

    def __init__(
        self,
        station_features: np.ndarray,
        station_metadata: np.ndarray,
        global_context: np.ndarray,
        station_bearings: np.ndarray,
        wind_direction: np.ndarray,
        station_mask: np.ndarray,
        targets: np.ndarray,
    ):
        self.station_features = torch.tensor(
            station_features, dtype=torch.float32
        )
        self.station_metadata = torch.tensor(
            station_metadata, dtype=torch.float32
        )
        self.global_context = torch.tensor(
            global_context, dtype=torch.float32
        )
        self.station_bearings = torch.tensor(
            station_bearings, dtype=torch.float32
        )
        self.wind_direction = torch.tensor(
            wind_direction, dtype=torch.float32
        )
        self.station_mask = torch.tensor(
            station_mask, dtype=torch.float32
        )
        self.targets = torch.tensor(
            targets, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "station_features": self.station_features[idx],
            "station_metadata": self.station_metadata[idx],
            "global_context": self.global_context[idx],
            "station_bearings": self.station_bearings[idx],
            "wind_direction": self.wind_direction[idx],
            "station_mask": self.station_mask[idx],
            "target": self.targets[idx],
        }


# ===========================================================================
# Data Preparation
# ===========================================================================

def prepare_attention_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    station_metadata: pd.DataFrame,
) -> dict:
    """Reshape flat feature matrices into attention model input format.

    Parses feature column names to separate station-specific features,
    station metadata, and global context features.  Constructs the
    3-D station feature tensors and metadata tensors expected by
    ``WindGatedAttentionModel``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (scaled), shape ``(N_train, F)``.
    X_val : pd.DataFrame
        Validation feature matrix (scaled).
    X_test : pd.DataFrame
        Test feature matrix (scaled).
    feature_names : list[str]
        Column names for the feature matrices.
    station_metadata : pd.DataFrame
        Station information with columns: ``station_id``, ``latitude``,
        ``longitude``, ``distance_miles``, ``direction``.  Must include
        the surrounding stations.

    Returns
    -------
    dict
        Dictionary with keys:
          - ``"train"``, ``"val"``, ``"test"``: each a dict with
            numpy arrays for ``station_features``, ``station_metadata``,
            ``global_context``, ``station_bearings``, ``wind_direction``,
            ``station_mask``.
          - ``"station_ids"``: ordered list of surrounding station IDs.
          - ``"station_feature_names"``: names of per-station features.
          - ``"global_feature_names"``: names of global features.
          - ``"n_station_features"``: int.
          - ``"n_metadata_features"``: int.
          - ``"n_global_features"``: int.
          - ``"n_stations"``: int.
    """
    # Determine surrounding station IDs from config
    surrounding_ids = list(config.SURROUNDING_STATIONS.keys())
    n_stations = len(surrounding_ids)

    # --- Classify features ---
    # Station features: columns matching "{station_id}_*_lag*"
    # Global features: sin_day, cos_day, NYC_TMAX_lag*, sector_*, grad_*,
    #                  trend_*, diurnal_*, and anything not station-specific

    station_feature_cols: dict[str, list[str]] = {
        sid: [] for sid in surrounding_ids
    }
    global_feature_cols: list[str] = []

    for col in feature_names:
        matched = False
        for sid in surrounding_ids:
            if col.startswith(sid):
                station_feature_cols[sid].append(col)
                matched = True
                break
        if not matched:
            global_feature_cols.append(col)

    # Determine consistent station feature names (sorted, from first station)
    # Each station should have the same set of feature suffixes
    station_feature_suffixes: list[str] = []
    for sid in surrounding_ids:
        cols = station_feature_cols[sid]
        if cols:
            suffixes = sorted([c.replace(sid + "_", "", 1) for c in cols])
            if not station_feature_suffixes:
                station_feature_suffixes = suffixes
            break

    n_station_features = len(station_feature_suffixes)
    n_global_features = len(global_feature_cols)

    # --- Station metadata ---
    # Build per-station metadata: [bearing_rad, distance_norm, lat_norm, lon_norm]
    meta_df = station_metadata.copy()
    meta_df = meta_df.set_index("station_id")

    # Direction to bearing in radians
    direction_map = {
        "N": 0.0, "NNE": math.pi / 8, "NE": math.pi / 4,
        "ENE": 3 * math.pi / 8, "E": math.pi / 2,
        "ESE": 5 * math.pi / 8, "SE": 3 * math.pi / 4,
        "SSE": 7 * math.pi / 8, "S": math.pi,
        "SSW": 9 * math.pi / 8, "SW": 5 * math.pi / 4,
        "WSW": 11 * math.pi / 8, "W": 3 * math.pi / 2,
        "WNW": 13 * math.pi / 8, "NW": 7 * math.pi / 4,
        "NNW": 15 * math.pi / 8, "Target": 0.0,
    }

    metadata_arrays = []
    bearing_array = []
    for sid in surrounding_ids:
        if sid in meta_df.index:
            row = meta_df.loc[sid]
            direction = str(row.get("direction", "N")).strip()
            bearing = direction_map.get(direction, 0.0)
            distance = float(row.get("distance_miles", 50.0))
            lat = float(row.get("latitude", 40.7))
            lon = float(row.get("longitude", -73.9))
        else:
            bearing = 0.0
            distance = 50.0
            lat = 40.7
            lon = -73.9

        # Normalise distance (divide by 200 mi max)
        dist_norm = distance / 200.0
        # Normalise lat/lon relative to Central Park
        lat_norm = (lat - config.TARGET_LAT) / 2.0
        lon_norm = (lon - config.TARGET_LON) / 2.0

        metadata_arrays.append([bearing, dist_norm, lat_norm, lon_norm])
        bearing_array.append(bearing)

    station_meta_array = np.array(metadata_arrays, dtype=np.float32)  # (S, M)
    n_metadata_features = station_meta_array.shape[1]
    bearings_static = np.array(bearing_array, dtype=np.float32)  # (S,)

    # --- Helper: reshape one split ---
    def _reshape_split(X: pd.DataFrame) -> dict:
        n_samples = len(X)
        values = X.values if hasattr(X, "values") else np.asarray(X)

        # Station features: (N, S, F_s)
        sf = np.zeros(
            (n_samples, n_stations, n_station_features), dtype=np.float32
        )
        for s_idx, sid in enumerate(surrounding_ids):
            for f_idx, suffix in enumerate(station_feature_suffixes):
                col_name = f"{sid}_{suffix}"
                if col_name in feature_names:
                    col_idx = feature_names.index(col_name)
                    sf[:, s_idx, f_idx] = values[:, col_idx]

        # Global context: (N, G)
        gc = np.zeros(
            (n_samples, n_global_features), dtype=np.float32
        )
        for g_idx, col_name in enumerate(global_feature_cols):
            if col_name in feature_names:
                col_idx = feature_names.index(col_name)
                gc[:, g_idx] = values[:, col_idx]

        # Station metadata: broadcast static to (N, S, M)
        sm = np.tile(station_meta_array, (n_samples, 1, 1))

        # Bearings: broadcast to (N, S)
        sb = np.tile(bearings_static, (n_samples, 1))

        # Wind direction: use a default placeholder (0.0).
        # In production, this would come from ASOS data.
        # If global_context has wind-direction features, they're in gc.
        wd = np.zeros(n_samples, dtype=np.float32)

        # Station mask: 1 if station has any non-zero features
        # (after scaling, zero means imputed/missing in many cases)
        # A simple heuristic: mark as present unless all station
        # features are exactly 0.
        mask = np.ones((n_samples, n_stations), dtype=np.float32)
        all_zero = np.all(sf == 0, axis=2)  # (N, S)
        mask[all_zero] = 0.0
        # Ensure at least one station is present per sample
        any_present = mask.sum(axis=1) > 0
        if not np.all(any_present):
            # Fall back: set all stations to present for those samples
            mask[~any_present] = 1.0

        return {
            "station_features": sf,
            "station_metadata": sm,
            "global_context": gc,
            "station_bearings": sb,
            "wind_direction": wd,
            "station_mask": mask,
        }

    import math  # noqa: F811

    train_data = _reshape_split(X_train)
    val_data = _reshape_split(X_val)
    test_data = _reshape_split(X_test)

    logger.info(
        "Prepared attention data: %d stations, %d station feats, "
        "%d metadata feats, %d global feats",
        n_stations, n_station_features, n_metadata_features,
        n_global_features,
    )

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "station_ids": surrounding_ids,
        "station_feature_names": station_feature_suffixes,
        "global_feature_names": global_feature_cols,
        "n_station_features": n_station_features,
        "n_metadata_features": n_metadata_features,
        "n_global_features": n_global_features,
        "n_stations": n_stations,
    }


# ===========================================================================
# DataLoader Creation
# ===========================================================================

def create_attention_dataloaders(
    train_data: dict,
    val_data: dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders from structured attention data.

    Parameters
    ----------
    train_data : dict
        Output of ``prepare_attention_data()["train"]``.
    val_data : dict
        Output of ``prepare_attention_data()["val"]``.
    y_train : np.ndarray
        Training targets.
    y_val : np.ndarray
        Validation targets.
    batch_size : int, optional
        Batch size.  Defaults to ``config.BATCH_SIZE``.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    y_train_arr = (
        y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    )
    y_val_arr = (
        y_val.values if hasattr(y_val, "values") else np.asarray(y_val)
    )

    train_dataset = AttentionDataset(
        station_features=train_data["station_features"],
        station_metadata=train_data["station_metadata"],
        global_context=train_data["global_context"],
        station_bearings=train_data["station_bearings"],
        wind_direction=train_data["wind_direction"],
        station_mask=train_data["station_mask"],
        targets=y_train_arr,
    )
    val_dataset = AttentionDataset(
        station_features=val_data["station_features"],
        station_metadata=val_data["station_metadata"],
        global_context=val_data["global_context"],
        station_bearings=val_data["station_bearings"],
        wind_direction=val_data["wind_direction"],
        station_mask=val_data["station_mask"],
        targets=y_val_arr,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    logger.info(
        "Created attention DataLoaders: train=%d batches, "
        "val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader


# ===========================================================================
# Training & Validation Steps
# ===========================================================================

def _run_model_on_batch(
    model: WindGatedAttentionModel,
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    """Run the model on a single batch, returning the output dict."""
    return model(
        station_features=batch["station_features"].to(device),
        station_metadata=batch["station_metadata"].to(device),
        global_context=batch["global_context"].to(device),
        station_bearings=batch["station_bearings"].to(device),
        wind_direction=batch["wind_direction"].to(device),
        station_mask=batch["station_mask"].to(device),
    )


def train_one_epoch(
    model: WindGatedAttentionModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    output_mode: str,
    device: str = "cpu",
) -> float:
    """Train for one epoch.

    Parameters
    ----------
    model : WindGatedAttentionModel
        The model to train.
    train_loader : DataLoader
        Training data loader (yields dicts from AttentionDataset).
    optimizer : torch.optim.Optimizer
        Optimiser.
    loss_fn : nn.Module
        Loss function.  For ``output_mode="gaussian"``, should accept
        (mu, sigma, target).  For ``"point"``, should accept
        (prediction, target).
    output_mode : str
        ``"point"`` or ``"gaussian"``.
    device : str
        Device to use.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        target = batch["target"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        output = _run_model_on_batch(model, batch, device)

        if output_mode == "gaussian":
            loss_result = loss_fn(output["mu"], output["sigma"], target)
            if isinstance(loss_result, dict):
                loss = loss_result["loss"]
            else:
                loss = loss_result
        else:
            loss = loss_fn(output["prediction"], target)

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: WindGatedAttentionModel,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    output_mode: str,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluate model on the validation set.

    Parameters
    ----------
    model : WindGatedAttentionModel
        The model.
    val_loader : DataLoader
        Validation data loader.
    loss_fn : nn.Module
        Loss function.
    output_mode : str
        ``"point"`` or ``"gaussian"``.
    device : str
        Device.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]
        (avg_val_loss, predictions, actuals, sigmas)
        sigmas is None for point mode.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    all_sigmas: list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            target = batch["target"].to(device).unsqueeze(1)
            output = _run_model_on_batch(model, batch, device)

            if output_mode == "gaussian":
                loss_result = loss_fn(output["mu"], output["sigma"], target)
                if isinstance(loss_result, dict):
                    loss = loss_result["loss"]
                else:
                    loss = loss_result
                all_sigmas.append(
                    output["sigma"].cpu().numpy().ravel()
                )
            else:
                loss = loss_fn(output["prediction"], target)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(
                output["prediction"].cpu().numpy().ravel()
            )
            all_actuals.append(target.cpu().numpy().ravel())

    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    sigmas = (
        np.concatenate(all_sigmas) if all_sigmas else None
    )

    return avg_loss, preds, actuals, sigmas


# ===========================================================================
# Full Training Loop
# ===========================================================================

def train_wind_gated_model(
    train_data: dict,
    val_data: dict,
    model_config: dict,
    training_config: dict,
    output_dir: str,
) -> dict:
    """Train the wind-gated attention model.

    Parameters
    ----------
    train_data : dict
        Must contain:
          - ``"attention"`` : dict with structured numpy arrays from
            ``prepare_attention_data()["train"]``.
          - ``"targets"`` : np.ndarray of target values.
    val_data : dict
        Same structure as ``train_data`` but for validation.
    model_config : dict
        Model hyperparameters:
          - ``"n_station_features"`` : int
          - ``"n_metadata_features"`` : int
          - ``"n_global_features"`` : int
          - ``"n_stations"`` : int
          - ``"station_embed_dim"`` : int (default 32)
          - ``"attention_dim"`` : int (default 16)
          - ``"output_mode"`` : str (default "point")
          - ``"dropout"`` : float (default 0.1)
    training_config : dict
        Training hyperparameters:
          - ``"learning_rate"`` : float (default config.LEARNING_RATE)
          - ``"max_epochs"`` : int (default config.MAX_EPOCHS)
          - ``"early_stopping_patience"`` : int (default 15)
          - ``"batch_size"`` : int (default config.BATCH_SIZE)
          - ``"loss_type"`` : str ("crps", "combined_crps_mae",
            "mse", "huber", "mae"; default "combined_crps_mae")
          - ``"crps_weight"`` : float (default 0.7, for combined)
          - ``"mae_weight"`` : float (default 0.3, for combined)
          - ``"target_type"`` : str ("raw" or "delta"; default "delta")
          - ``"device"`` : str (default "cpu")
          - ``"model_name"`` : str (default "wind_gated_attn")
    output_dir : str
        Directory for checkpoints, plots, and history CSV.

    Returns
    -------
    dict
        Dictionary with:
          - ``"model"`` : trained model (best checkpoint loaded)
          - ``"history"`` : list of epoch dicts
          - ``"best_epoch"`` : int
          - ``"best_val_mae"`` : float
          - ``"output_mode"`` : str
          - ``"loss_type"`` : str
          - ``"target_type"`` : str

    Raises
    ------
    ValueError
        If required config keys are missing.
    """
    # --- Unpack config ---
    n_station_features = model_config["n_station_features"]
    n_metadata_features = model_config["n_metadata_features"]
    n_global_features = model_config["n_global_features"]
    n_stations = model_config["n_stations"]
    station_embed_dim = model_config.get("station_embed_dim", 32)
    attention_dim = model_config.get("attention_dim", 16)
    output_mode = model_config.get("output_mode", "point")
    dropout = model_config.get("dropout", 0.1)

    lr = training_config.get("learning_rate", config.LEARNING_RATE)
    max_epochs = training_config.get("max_epochs", config.MAX_EPOCHS)
    patience = training_config.get(
        "early_stopping_patience", config.EARLY_STOPPING_PATIENCE
    )
    batch_size = training_config.get("batch_size", config.BATCH_SIZE)
    loss_type = training_config.get("loss_type", "combined_crps_mae")
    crps_weight = training_config.get("crps_weight", 0.7)
    mae_weight = training_config.get("mae_weight", 0.3)
    target_type = training_config.get("target_type", "delta")
    device = training_config.get("device", "cpu")
    model_name = training_config.get("model_name", "wind_gated_attn")

    os.makedirs(output_dir, exist_ok=True)

    # --- Build model ---
    model = WindGatedAttentionModel(
        n_station_features=n_station_features,
        n_metadata_features=n_metadata_features,
        n_global_features=n_global_features,
        n_stations=n_stations,
        station_embed_dim=station_embed_dim,
        attention_dim=attention_dim,
        output_mode=output_mode,
        dropout=dropout,
    )
    model = model.to(device)

    # --- Build loss ---
    if output_mode == "gaussian":
        if loss_type == "crps":
            loss_fn = GaussianCRPSLoss(reduction="mean")
        elif loss_type == "combined_crps_mae":
            loss_fn = CombinedCRPSMAELoss(
                crps_weight=crps_weight, mae_weight=mae_weight
            )
        else:
            # Fall back to CRPS for gaussian mode
            logger.warning(
                "loss_type='%s' not suitable for gaussian mode; "
                "falling back to combined_crps_mae",
                loss_type,
            )
            loss_fn = CombinedCRPSMAELoss(
                crps_weight=crps_weight, mae_weight=mae_weight
            )
    else:
        # Point mode: standard losses
        loss_type_map = {
            "mse": nn.MSELoss(),
            "huber": nn.SmoothL1Loss(),
            "mae": nn.L1Loss(),
        }
        if loss_type in loss_type_map:
            loss_fn = loss_type_map[loss_type]
        else:
            # Default to Huber for point mode
            logger.warning(
                "loss_type='%s' not valid for point mode; "
                "falling back to huber",
                loss_type,
            )
            loss_fn = nn.SmoothL1Loss()

    # --- Build DataLoaders ---
    train_loader, val_loader = create_attention_dataloaders(
        train_data["attention"],
        val_data["attention"],
        train_data["targets"],
        val_data["targets"],
        batch_size=batch_size,
    )

    # --- Optimiser & Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    # --- Checkpoint path ---
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    checkpoint_path = os.path.join(output_dir, f"best_{safe_name}.pt")

    # --- Delta-T reconstruction data ---
    nyc_prev_val = val_data.get("nyc_prev")
    actual_tmax_val = val_data.get("actual_tmax")

    # --- Training loop ---
    history: list[dict] = []
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info("=" * 60)
    logger.info(
        "Training '%s' (output=%s, loss=%s, target=%s)",
        model_name, output_mode, loss_type, target_type,
    )
    logger.info(
        "  LR: %.6f | Max epochs: %d | Patience: %d | Batch: %d",
        lr, max_epochs, patience, batch_size,
    )
    logger.info("=" * 60)

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            output_mode, device,
        )

        # -- Validate --
        val_loss, val_preds, val_actuals, val_sigmas = validate(
            model, val_loader, loss_fn, output_mode, device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # -- Compute MAE --
        if target_type == "delta" and nyc_prev_val is not None:
            delta_mae = float(np.mean(np.abs(val_preds - val_actuals)))
            reconstructed = nyc_prev_val + val_preds
            if actual_tmax_val is not None:
                val_mae = float(
                    np.mean(np.abs(reconstructed - actual_tmax_val))
                )
            else:
                val_mae = delta_mae
        else:
            val_mae = float(np.mean(np.abs(val_preds - val_actuals)))
            delta_mae = None

        # -- Record history --
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": current_lr,
        }
        if delta_mae is not None:
            entry["delta_mae"] = delta_mae
        if val_sigmas is not None:
            entry["mean_sigma"] = float(np.mean(val_sigmas))
        history.append(entry)

        # -- Check improvement --
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            if epoch <= 5 or epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | Loss: %.4f | Val MAE: %.3f F | * BEST *",
                    epoch, val_loss, val_mae,
                )
        else:
            epochs_without_improvement += 1
            if epoch <= 5 or epoch % 10 == 0:
                logger.info(
                    "Epoch %3d | Loss: %.4f | Val MAE: %.3f F | "
                    "No improvement (%d/%d)",
                    epoch, val_loss, val_mae,
                    epochs_without_improvement, patience,
                )

        # -- Early stopping --
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, patience,
            )
            break

    # --- Load best checkpoint ---
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)
        )
        logger.info(
            "Loaded best model from epoch %d (Val MAE: %.3f F)",
            best_epoch, best_val_mae,
        )

    # --- Save history ---
    history_path = os.path.join(output_dir, f"history_{safe_name}.csv")
    save_training_history(history, history_path)

    # --- Plot training curves ---
    plot_path = os.path.join(output_dir, f"curves_{safe_name}.png")
    plot_training_curves(
        history, plot_path,
        title=f"Training Curves: {model_name}",
    )

    logger.info(
        "Training '%s' complete. Best MAE: %.3f F (epoch %d)",
        model_name, best_val_mae, best_epoch,
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "output_mode": output_mode,
        "loss_type": loss_type,
        "target_type": target_type,
    }


# ===========================================================================
# History & Plots
# ===========================================================================

def save_training_history(
    history: list[dict],
    output_path: str,
) -> None:
    """Save training history to CSV.

    Parameters
    ----------
    history : list[dict]
        List of epoch dictionaries.
    output_path : str
        File path for the CSV.
    """
    if not history:
        logger.warning("Empty history -- nothing to save")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = list(history[0].keys())
    # Include any extra keys that appear in later epochs
    for entry in history:
        for k in entry:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    logger.info(
        "Saved training history (%d epochs) to %s",
        len(history), output_path,
    )


def plot_training_curves(
    history: list[dict],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot training and validation curves.

    Parameters
    ----------
    history : list[dict]
        Training history from ``train_wind_gated_model()``.
    save_path : str
        File path for the saved figure.
    title : str
        Plot title.
    """
    if not history:
        logger.warning("Empty history -- cannot plot")
        return

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_maes = [h["val_mae"] for h in history]

    has_sigma = "mean_sigma" in history[0]
    n_cols = 3 if has_sigma else 2

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

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
    axes[1].axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.7)
    axes[1].scatter(
        [epochs[best_idx]], [val_maes[best_idx]],
        color="red", zorder=5, s=50,
    )
    axes[1].annotate(
        f"{val_maes[best_idx]:.2f} F",
        (epochs[best_idx], val_maes[best_idx]),
        textcoords="offset points", xytext=(10, 10),
        fontsize=9, color="red",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (degF)")
    axes[1].set_title("Validation MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mean sigma (if gaussian mode)
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

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)
