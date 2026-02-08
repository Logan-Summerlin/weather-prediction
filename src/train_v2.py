"""
Enhanced Training Pipeline (V2) for NYC Temperature Prediction.

Extends the Phase 3 training pipeline with:
  - Multiple loss functions: MSE, Huber (SmoothL1), MAE (L1)
  - Delta-T target training with TMAX reconstruction for evaluation
  - Dual metric tracking: delta MAE and reconstructed TMAX MAE
  - Configurable target type: 'raw' (direct TMAX) or 'delta' (DeltaT)

When target_type='delta', the model predicts DeltaT(t) = TMAX_NYC(t) - TMAX_NYC(t-1).
Validation MAE is always computed on RECONSTRUCTED TMAX values:
    pred_TMAX(t) = actual_TMAX_NYC(t-1) + pred_delta(t)

This ensures apple-to-apple comparison with raw-target models.
"""

import os
import sys
import logging
import csv
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Loss Function Factory
# ===========================================================================

def get_loss_function(loss_type: str) -> nn.Module:
    """Return the appropriate PyTorch loss function.

    Parameters
    ----------
    loss_type : str
        One of 'mse', 'huber', 'mae'.
        - 'mse': nn.MSELoss (standard squared error)
        - 'huber': nn.SmoothL1Loss (Huber loss, delta=1.0)
        - 'mae': nn.L1Loss (mean absolute error)

    Returns
    -------
    nn.Module
        The loss function instance.

    Raises
    ------
    ValueError
        If ``loss_type`` is not recognized.
    """
    loss_type = loss_type.lower().strip()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Must be one of: 'mse', 'huber', 'mae'."
        )


# ===========================================================================
# DataLoader Creation (enhanced)
# ===========================================================================

def create_enhanced_dataloaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Convert pandas data to PyTorch DataLoaders.

    Identical to the Phase 3 version but included here for self-containment
    and to allow future enhancements.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices (scaled).
    y_train, y_val : pd.Series
        Target values (raw TMAX or delta-T, depending on target_type).
    batch_size : int, optional
        Batch size for both loaders. Defaults to config.BATCH_SIZE.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    X_train_t = torch.tensor(
        X_train.values if hasattr(X_train, "values") else np.asarray(X_train),
        dtype=torch.float32,
    )
    y_train_t = torch.tensor(
        y_train.values if hasattr(y_train, "values") else np.asarray(y_train),
        dtype=torch.float32,
    ).unsqueeze(1)

    X_val_t = torch.tensor(
        X_val.values if hasattr(X_val, "values") else np.asarray(X_val),
        dtype=torch.float32,
    )
    y_val_t = torch.tensor(
        y_val.values if hasattr(y_val, "values") else np.asarray(y_val),
        dtype=torch.float32,
    ).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    logger.info(
        "Created DataLoaders: train=%d batches, val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader


# ===========================================================================
# Enhanced Training & Validation
# ===========================================================================

def train_one_epoch_v2(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    train_loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    criterion : nn.Module
        Loss function.
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

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_v2(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on validation set.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : str
        Device to use.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (avg_val_loss, predictions_1d, actuals_1d)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds, axis=0).ravel()
    actuals = np.concatenate(all_actuals, axis=0).ravel()

    return avg_loss, preds, actuals


def compute_reconstructed_mae(
    pred_delta: np.ndarray,
    nyc_prev: np.ndarray,
    actual_tmax: np.ndarray,
) -> float:
    """Compute MAE on reconstructed absolute TMAX from delta predictions.

    Reconstruction:
        pred_TMAX(t) = actual_TMAX_NYC(t-1) + pred_delta(t)

    Parameters
    ----------
    pred_delta : np.ndarray
        Predicted delta-T values.
    nyc_prev : np.ndarray
        Actual NYC TMAX(t-1) values.
    actual_tmax : np.ndarray
        Actual NYC TMAX(t) values.

    Returns
    -------
    float
        Mean absolute error on reconstructed TMAX.
    """
    reconstructed = nyc_prev + pred_delta
    return float(np.mean(np.abs(reconstructed - actual_tmax)))


# ===========================================================================
# Full Enhanced Training Loop
# ===========================================================================

def train_enhanced_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_type: str = "huber",
    target_type: str = "delta",
    nyc_prev_val: Optional[np.ndarray] = None,
    actual_tmax_val: Optional[np.ndarray] = None,
    config_dict: Optional[dict] = None,
    device: str = "cpu",
    models_dir: Optional[str] = None,
    model_name: str = "model",
) -> dict:
    """Train a model with configurable loss and target type.

    Supports both raw TMAX and delta-T targets. When target_type='delta',
    tracks both delta MAE and reconstructed TMAX MAE. Early stopping is
    always based on the reconstructed TMAX MAE (for delta) or the raw MAE
    (for raw target).

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    loss_type : str
        Loss function: 'mse', 'huber', or 'mae'.
    target_type : str
        Target type: 'raw' (direct TMAX) or 'delta' (DeltaT).
    nyc_prev_val : np.ndarray, optional
        NYC TMAX(t-1) for the validation set. Required when
        target_type='delta' for reconstruction.
    actual_tmax_val : np.ndarray, optional
        Actual NYC TMAX(t) for the validation set. Required when
        target_type='delta' for reconstructed MAE.
    config_dict : dict, optional
        Override training hyperparameters:
        'learning_rate', 'max_epochs', 'early_stopping_patience'.
    device : str
        Device ('cpu' or 'cuda').
    models_dir : str, optional
        Directory for model checkpoints.
    model_name : str
        Name for checkpoint file (e.g., 'nn_delta_huber').

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': model loaded with best checkpoint weights
        - 'history': list of epoch dicts
        - 'best_epoch': epoch with lowest validation MAE
        - 'best_val_mae': lowest validation MAE (reconstructed if delta)
        - 'loss_type': loss function used
        - 'target_type': target type used
    """
    if target_type == "delta":
        if nyc_prev_val is None or actual_tmax_val is None:
            raise ValueError(
                "nyc_prev_val and actual_tmax_val are required "
                "when target_type='delta'"
            )

    if config_dict is None:
        config_dict = {}
    if models_dir is None:
        models_dir = config.MODELS_DIR

    lr = config_dict.get("learning_rate", config.LEARNING_RATE)
    max_epochs = config_dict.get("max_epochs", config.MAX_EPOCHS)
    patience = config_dict.get("early_stopping_patience",
                               config.EARLY_STOPPING_PATIENCE)

    os.makedirs(models_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    checkpoint_path = os.path.join(models_dir, f"best_{safe_name}.pt")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = get_loss_function(loss_type)

    history = []
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info("=" * 60)
    logger.info("Training '%s' (loss=%s, target=%s)", model_name, loss_type,
                target_type)
    logger.info("  LR: %.6f | Max epochs: %d | Patience: %d",
                lr, max_epochs, patience)
    logger.info("=" * 60)

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss = train_one_epoch_v2(
            model, train_loader, optimizer, criterion, device,
        )

        # Validate
        val_loss, val_preds, val_actuals = validate_v2(
            model, val_loader, criterion, device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Compute MAE
        if target_type == "delta":
            delta_mae = float(np.mean(np.abs(val_preds - val_actuals)))
            reconstructed_mae = compute_reconstructed_mae(
                val_preds, nyc_prev_val, actual_tmax_val,
            )
            val_mae = reconstructed_mae  # early stopping on reconstructed
        else:
            val_mae = float(np.mean(np.abs(val_preds - val_actuals)))
            delta_mae = None
            reconstructed_mae = None

        # Record history
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": current_lr,
        }
        if target_type == "delta":
            entry["delta_mae"] = delta_mae
            entry["reconstructed_mae"] = reconstructed_mae
        history.append(entry)

        # Check for improvement
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

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, patience,
            )
            break

    # Load best model weights
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)
        )
        logger.info("Loaded best model from epoch %d (Val MAE: %.3f F)",
                    best_epoch, best_val_mae)

    logger.info("Training '%s' complete. Best MAE: %.3f F (epoch %d)",
                model_name, best_val_mae, best_epoch)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "loss_type": loss_type,
        "target_type": target_type,
    }


# ===========================================================================
# Test-Set Evaluation
# ===========================================================================

def evaluate_on_test(
    model: nn.Module,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_type: str = "raw",
    nyc_prev_test: Optional[np.ndarray] = None,
    actual_tmax_test: Optional[np.ndarray] = None,
    device: str = "cpu",
    batch_size: Optional[int] = None,
) -> dict:
    """Evaluate a trained model on the test set.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix (scaled).
    y_test : pd.Series
        Test target (raw TMAX or delta-T).
    target_type : str
        'raw' or 'delta'.
    nyc_prev_test : np.ndarray, optional
        NYC TMAX(t-1) for reconstruction (required for delta).
    actual_tmax_test : np.ndarray, optional
        Actual TMAX for comparison (required for delta).
    device : str
        Device.
    batch_size : int, optional
        Batch size for evaluation.

    Returns
    -------
    dict
        Dictionary with test metrics: mae, rmse, r2, bias, predictions,
        actuals, and (for delta) reconstructed predictions.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    X_test_t = torch.tensor(
        X_test.values if hasattr(X_test, "values") else np.asarray(X_test),
        dtype=torch.float32,
    )
    y_test_t = torch.tensor(
        y_test.values if hasattr(y_test, "values") else np.asarray(y_test),
        dtype=torch.float32,
    ).unsqueeze(1)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

    criterion = nn.MSELoss()  # Just for computing loss; MAE is computed manually
    _, preds, actuals = validate_v2(model, test_loader, criterion, device)

    if target_type == "delta":
        if nyc_prev_test is None or actual_tmax_test is None:
            raise ValueError(
                "nyc_prev_test and actual_tmax_test required for delta eval"
            )
        reconstructed = nyc_prev_test + preds
        eval_preds = reconstructed
        eval_actuals = actual_tmax_test
    else:
        eval_preds = preds
        eval_actuals = actuals

    errors = eval_preds - eval_actuals
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((eval_actuals - np.mean(eval_actuals)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "predictions": eval_preds,
        "actuals": eval_actuals,
        "raw_predictions": preds,
        "n": len(eval_actuals),
    }

    if target_type == "delta":
        result["delta_mae"] = float(np.mean(np.abs(preds - actuals)))
        result["reconstructed_predictions"] = reconstructed

    logger.info(
        "Test evaluation (target=%s): MAE=%.3f, RMSE=%.3f, R2=%.4f",
        target_type, mae, rmse, r2,
    )

    return result


# ===========================================================================
# History & Plots
# ===========================================================================

def save_training_history_v2(history: list, output_path: str) -> None:
    """Save training history to CSV.

    Parameters
    ----------
    history : list[dict]
        List of epoch dictionaries.
    output_path : str
        File path for the CSV output.
    """
    if not history:
        logger.warning("Empty history -- nothing to save")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = list(history[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    logger.info("Saved training history (%d epochs) to %s",
                len(history), output_path)


def plot_training_curves_v2(
    history: list,
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot training curves with optional delta/reconstructed MAE tracking.

    Parameters
    ----------
    history : list[dict]
        Training history.
    save_path : str
        File path to save the figure.
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

    has_delta = "delta_mae" in history[0]

    n_cols = 3 if has_delta else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # Loss curves
    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val MAE (reconstructed if delta)
    axes[1].plot(epochs, val_maes, label="Val MAE (TMAX)", linewidth=1.5,
                 color="#2ca02c")
    best_idx = int(np.argmin(val_maes))
    axes[1].axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.7)
    axes[1].scatter([epochs[best_idx]], [val_maes[best_idx]],
                    color="red", zorder=5, s=50)
    axes[1].annotate(
        f"{val_maes[best_idx]:.2f} F",
        (epochs[best_idx], val_maes[best_idx]),
        textcoords="offset points", xytext=(10, 10),
        fontsize=9, color="red",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (degF)")
    axes[1].set_title("Validation MAE (Reconstructed TMAX)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Delta MAE (only for delta target)
    if has_delta:
        delta_maes = [h.get("delta_mae", 0) for h in history]
        axes[2].plot(epochs, delta_maes, label="Delta MAE",
                     linewidth=1.5, color="#ff7f0e")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("MAE (degF)")
        axes[2].set_title("Delta-T MAE")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)
