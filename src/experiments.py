"""
Sensitivity Experiment Framework for NYC Temperature Prediction (Phase 4).

Provides a structured way to define, run, and compare experiments that
vary model architecture, loss function, feature subsets, and
regularisation.

Classes:
  - ExperimentConfig  -- dataclass describing a single experiment
  - ExperimentResult  -- dataclass collecting metrics from one run

Functions:
  - select_features()           -- subset feature columns
  - run_experiment()            -- train + evaluate one config
  - run_experiment_suite()      -- batch-run multiple configs
  - generate_experiment_report() -- comparison table, plots, text file
"""

import csv
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.models_v2 import (
    EnhancedMLP,
    MultiLagMLP,
    LSTMPredictor,
    StationAttentionModel,
    get_loss_function,
    create_model_v2,
)
from src.evaluate import compute_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class ExperimentConfig:
    """Defines a single experiment configuration.

    All fields have sensible defaults so that only the dimensions you
    want to vary need to be specified.

    Attributes
    ----------
    name : str
        Short human-readable identifier (used in tables/plots).
    model_class : str
        Model identifier: ``"mlp"``, ``"enhanced_mlp"``, ``"lstm"``,
        ``"gru"``, ``"attention"``.
    hidden_sizes : list[int]
        Hidden layer widths (for MLP variants).
    dropout : float
        Dropout probability.
    loss_type : str
        Loss function: ``"mse"``, ``"huber"``, ``"mae"``.
    learning_rate : float
        Initial Adam learning rate.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience.
    features : str
        Feature selection mode:
        ``"all"`` | ``"tmax_only"`` | ``"tmin_only"`` | ``"no_date"``.
    use_batch_norm : bool
        Use batch normalisation (EnhancedMLP only).
    model_kwargs : dict
        Extra keyword arguments forwarded to the model constructor.
    """

    name: str = "default"
    model_class: str = "enhanced_mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    loss_type: str = "mse"
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 15
    features: str = "all"
    use_batch_norm: bool = False
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Collects outputs from a single experiment run.

    Attributes
    ----------
    config_name : str
        Name from the corresponding ExperimentConfig.
    mae : float
        Test-set Mean Absolute Error (degF).
    rmse : float
        Test-set Root Mean Squared Error (degF).
    r2 : float
        Test-set R-squared.
    bias : float
        Test-set mean signed error (pred - actual).
    best_epoch : int
        Epoch at which early stopping selected the best model.
    best_val_mae : float
        Validation MAE at the best epoch.
    n_params : int
        Total trainable parameters.
    train_time_s : float
        Wall-clock training time in seconds.
    status : str
        ``"success"`` or an error message.
    """

    config_name: str = ""
    mae: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    bias: float = float("nan")
    best_epoch: int = 0
    best_val_mae: float = float("nan")
    n_params: int = 0
    train_time_s: float = 0.0
    status: str = "pending"


# ===========================================================================
# Feature Selection
# ===========================================================================

def select_features(
    X: pd.DataFrame,
    mode: str = "all",
) -> pd.DataFrame:
    """Subset feature columns according to *mode*.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (columns like ``<STATION>_TMAX_lag1``,
        ``<STATION>_TMIN_lag1``, ``sin_day``, ``cos_day``).
    mode : str
        One of:
        - ``"all"`` — keep every column.
        - ``"tmax_only"`` — keep only TMAX lag columns + date features.
        - ``"tmin_only"`` — keep only TMIN lag columns + date features.
        - ``"no_date"`` — drop ``sin_day`` and ``cos_day``.

    Returns
    -------
    pd.DataFrame
        Column-filtered copy of *X*.

    Raises
    ------
    ValueError
        If *mode* is not recognised.
    """
    mode = mode.lower().strip()

    if mode == "all":
        return X

    date_cols = [c for c in X.columns if c in ("sin_day", "cos_day")]
    tmax_cols = [c for c in X.columns if "TMAX" in c]
    tmin_cols = [c for c in X.columns if "TMIN" in c]

    if mode == "tmax_only":
        return X[tmax_cols + date_cols]
    elif mode == "tmin_only":
        return X[tmin_cols + date_cols]
    elif mode == "no_date":
        non_date = [c for c in X.columns if c not in ("sin_day", "cos_day")]
        return X[non_date]
    else:
        raise ValueError(
            f"Unknown feature mode '{mode}'. "
            "Choose from: all, tmax_only, tmin_only, no_date."
        )


# ===========================================================================
# Single Experiment Runner
# ===========================================================================

def _build_model(cfg: ExperimentConfig, n_features: int) -> nn.Module:
    """Instantiate a model from an ExperimentConfig."""
    mc = cfg.model_class.lower().strip()

    if mc in ("mlp", "enhanced_mlp"):
        return EnhancedMLP(
            n_features=n_features,
            hidden_sizes=list(cfg.hidden_sizes),
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        )
    elif mc == "lstm":
        hidden = cfg.hidden_sizes[0] if cfg.hidden_sizes else 64
        return LSTMPredictor(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=cfg.model_kwargs.get("num_layers", 1),
            dropout=cfg.dropout,
            cell_type="lstm",
            bidirectional=cfg.model_kwargs.get("bidirectional", False),
        )
    elif mc == "gru":
        hidden = cfg.hidden_sizes[0] if cfg.hidden_sizes else 64
        return LSTMPredictor(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=cfg.model_kwargs.get("num_layers", 1),
            dropout=cfg.dropout,
            cell_type="gru",
            bidirectional=cfg.model_kwargs.get("bidirectional", False),
        )
    elif mc == "attention":
        return StationAttentionModel(
            features_per_station=cfg.model_kwargs.get("features_per_station", 2),
            n_stations=cfg.model_kwargs.get("n_stations", 14),
            embed_dim=cfg.hidden_sizes[0] if cfg.hidden_sizes else 32,
            n_heads=cfg.model_kwargs.get("n_heads", 4),
            dropout=cfg.dropout,
            n_extra_features=cfg.model_kwargs.get("n_extra_features", 2),
        )
    else:
        raise ValueError(f"Unknown model_class '{mc}'")


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders from numpy arrays."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, val_loader


def run_experiment(
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    models_dir: Optional[str] = None,
) -> ExperimentResult:
    """Run a single experiment: build model, train, evaluate.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    X_train, y_train : np.ndarray
        Training data (features already column-subset).
    X_val, y_val : np.ndarray
        Validation data.
    X_test, y_test : np.ndarray
        Test data.
    device : str
        ``"cpu"`` or ``"cuda"``.
    models_dir : str, optional
        Directory for saving model checkpoints.  If ``None``, a
        temporary directory is used.

    Returns
    -------
    ExperimentResult
        Metrics and metadata for this experiment.
    """
    result = ExperimentResult(config_name=cfg.name)
    t0 = time.time()

    try:
        n_features = X_train.shape[1]
        logger.info("Experiment '%s': n_features=%d, model=%s",
                     cfg.name, n_features, cfg.model_class)

        # Build model
        model = _build_model(cfg, n_features)
        result.n_params = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)

        # Data loaders
        train_loader, val_loader = _make_loaders(
            X_train, y_train, X_val, y_val, cfg.batch_size,
        )

        # Loss, optimizer, scheduler
        criterion = get_loss_function(cfg.loss_type)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )

        # Checkpoint path
        if models_dir is None:
            import tempfile
            models_dir = tempfile.mkdtemp(prefix="exp_")
        os.makedirs(models_dir, exist_ok=True)
        ckpt_path = os.path.join(
            models_dir, f"{cfg.name.replace(' ', '_')}_best.pt"
        )

        # Training loop
        best_val_mae = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, cfg.max_epochs + 1):
            # Train
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_preds = []
            val_actuals = []
            val_loss_sum = 0.0
            n_batches = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    pred = model(X_b)
                    val_loss_sum += criterion(pred, y_b).item()
                    n_batches += 1
                    val_preds.append(pred.cpu().numpy())
                    val_actuals.append(y_b.cpu().numpy())

            val_preds = np.concatenate(val_preds).ravel()
            val_actuals = np.concatenate(val_actuals).ravel()
            val_mae = float(np.mean(np.abs(val_preds - val_actuals)))
            avg_val_loss = val_loss_sum / max(n_batches, 1)

            scheduler.step(avg_val_loss)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= cfg.patience:
                logger.info("  '%s' early stop at epoch %d", cfg.name, epoch)
                break

            # Check for NaN loss
            if not np.isfinite(avg_val_loss):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch}"
                )

        # Load best weights
        if os.path.isfile(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, weights_only=True, map_location=device)
            )

        # Evaluate on test set
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            test_preds = model(X_test_t).cpu().numpy().ravel()

        metrics = compute_metrics(y_test, test_preds)

        result.mae = metrics["mae"]
        result.rmse = metrics["rmse"]
        result.r2 = metrics["r2"]
        result.bias = metrics["bias"]
        result.best_epoch = best_epoch
        result.best_val_mae = best_val_mae
        result.train_time_s = time.time() - t0
        result.status = "success"

        logger.info(
            "  '%s' done: test MAE=%.3f, RMSE=%.3f, R2=%.4f, "
            "best_epoch=%d (%.1fs)",
            cfg.name, result.mae, result.rmse, result.r2,
            best_epoch, result.train_time_s,
        )

    except Exception as e:
        result.status = f"FAILED: {e}"
        result.train_time_s = time.time() - t0
        logger.error("Experiment '%s' failed: %s", cfg.name, e)
        logger.debug(traceback.format_exc())

    return result


# ===========================================================================
# Experiment Suite Runner
# ===========================================================================

def run_experiment_suite(
    configs: list[ExperimentConfig],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str = "cpu",
    models_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run all experiments and return a comparison DataFrame.

    Each config's ``features`` field is used to subset columns from
    the full feature matrices before training.

    Parameters
    ----------
    configs : list[ExperimentConfig]
        List of experiment configurations.
    X_train, y_train : pd.DataFrame / pd.Series
        Training data (full feature set).
    X_val, y_val : pd.DataFrame / pd.Series
        Validation data.
    X_test, y_test : pd.DataFrame / pd.Series
        Test data.
    device : str
        ``"cpu"`` or ``"cuda"``.
    models_dir : str, optional
        Base directory for model checkpoints.

    Returns
    -------
    pd.DataFrame
        One row per experiment with columns:
        name, model_class, mae, rmse, r2, bias, best_epoch,
        best_val_mae, n_params, train_time_s, status.
    """
    results: list[dict] = []

    for i, cfg in enumerate(configs, 1):
        logger.info(
            "=== Experiment %d/%d: '%s' ===", i, len(configs), cfg.name
        )

        # Feature selection
        Xtr = select_features(X_train, cfg.features)
        Xv = select_features(X_val, cfg.features)
        Xte = select_features(X_test, cfg.features)

        # Convert to numpy
        Xtr_np = Xtr.values if hasattr(Xtr, "values") else np.asarray(Xtr)
        ytr_np = (y_train.values if hasattr(y_train, "values")
                  else np.asarray(y_train))
        Xv_np = Xv.values if hasattr(Xv, "values") else np.asarray(Xv)
        yv_np = (y_val.values if hasattr(y_val, "values")
                 else np.asarray(y_val))
        Xte_np = Xte.values if hasattr(Xte, "values") else np.asarray(Xte)
        yte_np = (y_test.values if hasattr(y_test, "values")
                  else np.asarray(y_test))

        exp_models_dir = None
        if models_dir is not None:
            exp_models_dir = os.path.join(
                models_dir, cfg.name.replace(" ", "_")
            )

        r = run_experiment(
            cfg,
            Xtr_np, ytr_np,
            Xv_np, yv_np,
            Xte_np, yte_np,
            device=device,
            models_dir=exp_models_dir,
        )

        row = {
            "name": r.config_name,
            "model_class": cfg.model_class,
            "hidden_sizes": str(cfg.hidden_sizes),
            "dropout": cfg.dropout,
            "loss_type": cfg.loss_type,
            "features": cfg.features,
            "mae": r.mae,
            "rmse": r.rmse,
            "r2": r.r2,
            "bias": r.bias,
            "best_epoch": r.best_epoch,
            "best_val_mae": r.best_val_mae,
            "n_params": r.n_params,
            "train_time_s": round(r.train_time_s, 2),
            "status": r.status,
        }
        results.append(row)

    df = pd.DataFrame(results)
    return df


# ===========================================================================
# Report Generation
# ===========================================================================

def generate_experiment_report(
    results_df: pd.DataFrame,
    output_dir: str,
    reference_mae: float = 4.29,
) -> str:
    """Generate comparison tables, plots, and a text report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_experiment_suite()``.
    output_dir : str
        Directory to save report artefacts.
    reference_mae : float
        Phase 3 baseline MAE for comparison annotation.

    Returns
    -------
    str
        Full text of the report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw CSV
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved experiment results to %s", csv_path)

    # Only successful runs for plotting
    ok = results_df[results_df["status"] == "success"].copy()

    lines = [
        "=" * 70,
        "NYC Temperature Prediction — Sensitivity Experiment Report",
        "=" * 70,
        "",
        f"Total experiments: {len(results_df)}",
        f"Successful: {len(ok)}",
        f"Failed: {len(results_df) - len(ok)}",
        f"Reference MAE (NN V1): {reference_mae:.2f} F",
        "",
    ]

    if len(ok) == 0:
        lines.append("No successful experiments to report.")
        report_text = "\n".join(lines)
        report_path = os.path.join(output_dir, "experiment_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        return report_text

    # Sort by MAE
    ok_sorted = ok.sort_values("mae")

    # Text table
    lines.append("--- Results (sorted by MAE) ---")
    lines.append("")
    header = (
        f"{'Experiment':<35s} {'MAE':>7s} {'RMSE':>7s} {'R2':>8s} "
        f"{'Bias':>7s} {'Epoch':>6s} {'Params':>8s} {'Time':>6s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in ok_sorted.iterrows():
        beat = " *" if row["mae"] < reference_mae else ""
        lines.append(
            f"{row['name']:<35s} {row['mae']:>7.3f} {row['rmse']:>7.3f} "
            f"{row['r2']:>8.4f} {row['bias']:>7.3f} {row['best_epoch']:>6.0f} "
            f"{row['n_params']:>8.0f} {row['train_time_s']:>5.1f}s"
            f"{beat}"
        )
    lines.append("")
    lines.append("* = beats Phase 3 NN V1 reference MAE")
    lines.append("")

    # Best experiment
    best = ok_sorted.iloc[0]
    lines.append(f"Best experiment: {best['name']}")
    lines.append(f"  MAE:  {best['mae']:.3f} F")
    lines.append(f"  RMSE: {best['rmse']:.3f} F")
    lines.append(f"  R2:   {best['r2']:.4f}")
    lines.append("")

    # Failed experiments
    failed = results_df[results_df["status"] != "success"]
    if len(failed) > 0:
        lines.append("--- Failed Experiments ---")
        for _, row in failed.iterrows():
            lines.append(f"  {row['name']}: {row['status']}")
        lines.append("")

    lines.append("=" * 70)
    report_text = "\n".join(lines)

    # Save text report
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info("Saved experiment report to %s", report_path)

    # ---- Plots ----

    # 1. Architecture comparison bar chart
    _plot_comparison_bar(
        ok_sorted, "mae",
        title="Architecture Comparison: MAE",
        save_path=os.path.join(output_dir, "architecture_comparison.png"),
        reference_line=reference_mae,
        reference_label=f"NN V1 ({reference_mae:.2f})",
    )

    # 2. Feature ablation (if any feature-related experiments exist)
    feature_exps = ok[ok["features"] != "all"]
    if len(feature_exps) > 0:
        # Include the 'all' baseline too
        all_baseline = ok[ok["features"] == "all"]
        if len(all_baseline) > 0:
            feature_df = pd.concat(
                [all_baseline.head(1), feature_exps]
            ).sort_values("mae")
        else:
            feature_df = feature_exps.sort_values("mae")

        _plot_comparison_bar(
            feature_df, "mae",
            title="Feature Ablation: MAE",
            save_path=os.path.join(output_dir, "feature_ablation.png"),
            reference_line=reference_mae,
            reference_label=f"NN V1 ({reference_mae:.2f})",
        )

    return report_text


def _plot_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: str,
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
) -> None:
    """Bar chart comparing a metric across experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results (must have ``"name"`` and *metric* columns).
    metric : str
        Column name to plot.
    title : str
        Plot title.
    save_path : str
        Where to save the figure.
    reference_line : float, optional
        Horizontal reference line value.
    reference_label : str, optional
        Label for the reference line.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 6))

    names = df["name"].tolist()
    values = df[metric].tolist()

    # Colour bars that beat the reference differently
    colours = []
    for v in values:
        if reference_line is not None and v < reference_line:
            colours.append("#2ca02c")  # green
        else:
            colours.append("#4c72b0")  # blue

    bars = ax.bar(range(len(names)), values, color=colours, edgecolor="white")

    # Value annotations
    for bar_obj, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)

    if reference_line is not None:
        ax.axhline(
            reference_line, color="red", linestyle="--", linewidth=1.2,
            label=reference_label or f"Reference ({reference_line:.2f})",
        )
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", save_path)
    plt.close(fig)
