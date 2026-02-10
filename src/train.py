"""
Training Pipeline for NYC Temperature Prediction Neural Network.

Provides the complete training loop for PyTorch models, including:
  - Data loading and DataLoader creation
  - Single-epoch training and validation
  - Full training with early stopping and learning-rate scheduling
  - Model checkpoint saving and training history export
  - Training curve visualization

The target variable (NYC TMAX) is NOT scaled -- it remains in original
degrees Fahrenheit.  Only the input features are standardized.  Therefore,
validation MAE is reported directly in degF.
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

# Use non-interactive backend before any other matplotlib import
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
# Data Loading
# ===========================================================================

def load_processed_data(processed_dir: Optional[str] = None) -> tuple:
    """Load preprocessed CSV files from disk.

    Reads the six CSV files saved by the preprocessing pipeline
    (features_train, features_val, features_test, target_train,
    target_val, target_test) and returns them as pandas objects.

    Parameters
    ----------
    processed_dir : str, optional
        Directory containing the processed CSV files.
        Defaults to config.PROCESSED_DATA_DIR.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
        Features are DataFrames; targets are Series named 'NYC_TMAX'.

    Raises
    ------
    FileNotFoundError
        If the processed data directory or any required file is missing.
    """
    if processed_dir is None:
        processed_dir = config.PROCESSED_DATA_DIR

    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_dir}\n"
            "Run the preprocessing pipeline first "
            "(python -m src.data_preprocessing)."
        )

    required_files = [
        "features_train.csv", "features_val.csv", "features_test.csv",
        "target_train.csv", "target_val.csv", "target_test.csv",
    ]
    for fname in required_files:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required file not found: {fpath}\n"
                "Run the preprocessing pipeline first."
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
    )["NYC_TMAX"]
    y_val = pd.read_csv(
        os.path.join(processed_dir, "target_val.csv"),
        index_col=0, parse_dates=True,
    )["NYC_TMAX"]
    y_test = pd.read_csv(
        os.path.join(processed_dir, "target_test.csv"),
        index_col=0, parse_dates=True,
    )["NYC_TMAX"]

    logger.info("Loaded processed data from %s", processed_dir)
    logger.info("  X_train: %s, X_val: %s, X_test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    logger.info("  y_train: %d, y_val: %d, y_test: %d",
                len(y_train), len(y_val), len(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===========================================================================
# DataLoader Creation
# ===========================================================================

def create_dataloaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    batch_size: Optional[int] = None,
) -> tuple:
    """Convert pandas data to PyTorch DataLoaders.

    Creates TensorDatasets from the feature matrices and target vectors,
    then wraps them in DataLoaders with appropriate settings.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (scaled).
    y_train : pd.Series
        Training target values (unscaled degF).
    X_val : pd.DataFrame
        Validation feature matrix (scaled).
    y_val : pd.Series
        Validation target values (unscaled degF).
    batch_size : int, optional
        Batch size for both loaders. Defaults to config.BATCH_SIZE.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Convert to numpy, then to float32 tensors
    X_train_t = torch.tensor(
        X_train.values if hasattr(X_train, "values") else np.asarray(X_train),
        dtype=torch.float32,
    )
    y_train_t = torch.tensor(
        y_train.values if hasattr(y_train, "values") else np.asarray(y_train),
        dtype=torch.float32,
    ).unsqueeze(1)  # Shape: (n, 1)

    X_val_t = torch.tensor(
        X_val.values if hasattr(X_val, "values") else np.asarray(X_val),
        dtype=torch.float32,
    )
    y_val_t = torch.tensor(
        y_val.values if hasattr(y_val, "values") else np.asarray(y_val),
        dtype=torch.float32,
    ).unsqueeze(1)  # Shape: (n, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(
        "Created DataLoaders: train=%d batches (batch_size=%d), "
        "val=%d batches",
        len(train_loader), batch_size, len(val_loader),
    )

    return train_loader, val_loader


# ===========================================================================
# Training & Validation
# ===========================================================================

def train_one_epoch(
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
        Optimizer (e.g., Adam).
    criterion : nn.Module
        Loss function (e.g., MSELoss).
    device : str
        Device to use ('cpu' or 'cuda').

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

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> tuple:
    """Evaluate the model on the validation set.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    val_loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function (e.g., MSELoss).
    device : str
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (average_val_loss, predictions_array, actuals_array)
        Both arrays are 1-D numpy arrays in original degF units.
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


# ===========================================================================
# Full Training Loop
# ===========================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_dict: Optional[dict] = None,
    device: str = "cpu",
    models_dir: Optional[str] = None,
) -> dict:
    """Train a model with early stopping and learning-rate scheduling.

    Uses Adam optimizer, MSE loss, ReduceLROnPlateau scheduler, and
    early stopping based on validation MAE.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config_dict : dict, optional
        Override training hyperparameters. Recognized keys:
        'learning_rate', 'max_epochs', 'early_stopping_patience'.
        Defaults to values from config module.
    device : str
        Device to use ('cpu' or 'cuda').
    models_dir : str, optional
        Directory to save model checkpoints.
        Defaults to config.MODELS_DIR.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': the model loaded with best checkpoint weights
        - 'history': list of dicts (epoch, train_loss, val_loss, val_mae, lr)
        - 'best_epoch': epoch number with lowest val MAE
        - 'best_val_mae': lowest validation MAE achieved (degF)
    """
    if config_dict is None:
        config_dict = {}
    if models_dir is None:
        models_dir = config.MODELS_DIR

    lr = config_dict.get("learning_rate", config.LEARNING_RATE)
    max_epochs = config_dict.get("max_epochs", config.MAX_EPOCHS)
    patience = config_dict.get("early_stopping_patience",
                               config.EARLY_STOPPING_PATIENCE)

    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "best_model.pt")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    criterion = nn.MSELoss()

    history = []
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("  Learning rate: %.6f", lr)
    logger.info("  Max epochs: %d", max_epochs)
    logger.info("  Early stopping patience: %d", patience)
    logger.info("  Device: %s", device)
    logger.info("  Checkpoint path: %s", checkpoint_path)
    logger.info("=" * 60)

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)

        # Validate
        val_loss, val_preds, val_actuals = validate(model, val_loader,
                                                    criterion, device)

        # Compute validation MAE in original degF
        val_mae = float(np.mean(np.abs(val_preds - val_actuals)))

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler (monitors val loss)
        scheduler.step(val_loss)

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "lr": current_lr,
        })

        # Check for improvement
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | "
                "Val MAE: %.3f F | LR: %.6f | * NEW BEST *",
                epoch, train_loss, val_loss, val_mae, current_lr,
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | "
                "Val MAE: %.3f F | LR: %.6f | No improvement (%d/%d)",
                epoch, train_loss, val_loss, val_mae, current_lr,
                epochs_without_improvement, patience,
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping triggered at epoch %d "
                "(no improvement for %d epochs)",
                epoch, patience,
            )
            break

    # Load best model weights
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        logger.info("Loaded best model from epoch %d (Val MAE: %.3f F)",
                    best_epoch, best_val_mae)

    logger.info("Training complete. Best epoch: %d, Best Val MAE: %.3f F",
                best_epoch, best_val_mae)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
    }


# ===========================================================================
# History Saving
# ===========================================================================

def save_training_history(history: list, output_path: str) -> None:
    """Save training history to a CSV file.

    Parameters
    ----------
    history : list[dict]
        List of epoch dictionaries, each containing:
        epoch, train_loss, val_loss, val_mae, lr.
    output_path : str
        File path for the CSV output.
    """
    if not history:
        logger.warning("Empty history — nothing to save")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = list(history[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    logger.info("Saved training history (%d epochs) to %s",
                len(history), output_path)


# ===========================================================================
# Training Curve Plots
# ===========================================================================

def plot_training_curves(history: list, save_path: str) -> None:
    """Plot training and validation loss curves, plus validation MAE.

    Creates a figure with two subplots:
      1. Train loss and val loss vs. epoch
      2. Validation MAE (degF) vs. epoch

    Parameters
    ----------
    history : list[dict]
        Training history (list of epoch dicts).
    save_path : str
        File path to save the figure (e.g., 'results/training_curves.png').
    """
    if not history:
        logger.warning("Empty history — cannot plot training curves")
        return

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_maes = [h["val_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=1.5,
             color="#1f77b4")
    ax1.plot(epochs, val_losses, label="Val Loss", linewidth=1.5,
             color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Right: Val MAE
    ax2 = axes[1]
    ax2.plot(epochs, val_maes, label="Val MAE", linewidth=1.5,
             color="#2ca02c")

    # Mark best epoch
    best_idx = int(np.argmin(val_maes))
    ax2.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.7,
                label=f"Best epoch ({epochs[best_idx]})")
    ax2.scatter([epochs[best_idx]], [val_maes[best_idx]],
                color="red", zorder=5, s=50)
    ax2.annotate(
        f"{val_maes[best_idx]:.2f}\u00b0F",
        (epochs[best_idx], val_maes[best_idx]),
        textcoords="offset points", xytext=(10, 10),
        fontsize=9, color="red",
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (\u00b0F)")
    ax2.set_title("Validation MAE")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved training curves to %s", save_path)
    plt.close(fig)
