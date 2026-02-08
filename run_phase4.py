"""
Run Phase 4 Experiments: Delta-T Target, Autoregressive Input,
Huber Loss, and Feature Engineering.

Trains and evaluates five model configurations:
  (a) NN V1 with raw TMAX target + MSE loss   (reproduce Phase 3 baseline)
  (b) NN with raw TMAX + Huber loss
  (c) NN with delta-T target + Huber loss + NO autoregressive input
  (d) NN with delta-T target + Huber loss + WITH autoregressive NYC TMAX(t-1)
  (e) NN with delta-T + Huber + ALL enhanced features
      (diurnal, sectors, gradients, trends)

All models use the same architecture (TempPredictorV1) with hidden_sizes
adapted to the input dimension. Results are saved to results/phase4/.

Usage:
    python run_phase4.py
"""

import os
import sys
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import config
from src.model import TempPredictorV1, count_parameters
from src.data_preprocessing_v2 import (
    run_enhanced_preprocessing,
    get_sector_assignments,
    PROCESSED_V2_DIR,
)
from src.train_v2 import (
    create_enhanced_dataloaders,
    train_enhanced_model,
    evaluate_on_test,
    save_training_history_v2,
    plot_training_curves_v2,
)
from src.evaluate import (
    compute_metrics,
    format_metrics_table,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_time_series,
    plot_baseline_comparison,
)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "phase4")
MODELS_DIR = os.path.join(config.MODELS_DIR, "phase4")

# Phase 3 / baseline reference numbers
REFERENCE_RESULTS = {
    "Persistence": {"mae": 5.06, "rmse": 6.39, "r2": 0.799},
    "Ridge (alpha=1.0)": {"mae": 4.33, "rmse": 5.41, "r2": 0.876},
    "NN V1 (Phase 3)": {"mae": 4.29, "rmse": 5.69, "r2": 0.869},
}


def _print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def run_experiment(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    loss_type: str = "mse",
    target_type: str = "raw",
    nyc_prev_val: np.ndarray = None,
    nyc_prev_test: np.ndarray = None,
    actual_tmax_val: np.ndarray = None,
    actual_tmax_test: np.ndarray = None,
    max_epochs: int = 200,
) -> dict:
    """Run a single training experiment.

    Parameters
    ----------
    name : str
        Experiment name.
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_val, y_test : pd.Series
        Target vectors.
    loss_type : str
        Loss function type.
    target_type : str
        Target type: 'raw' or 'delta'.
    nyc_prev_val, nyc_prev_test : np.ndarray
        NYC TMAX(t-1) for reconstruction (delta only).
    actual_tmax_val, actual_tmax_test : np.ndarray
        Actual NYC TMAX for evaluation (delta only).
    max_epochs : int
        Maximum training epochs.

    Returns
    -------
    dict
        Experiment results including metrics and model info.
    """
    print(f"  Experiment: {name}")
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Loss: {loss_type} | Target: {target_type}")

    n_features = X_train.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model -- adapt hidden sizes for larger feature counts
    if n_features > 50:
        hidden_sizes = [128, 64]
    else:
        hidden_sizes = list(config.HIDDEN_SIZES)

    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=config.DROPOUT,
    )
    n_params = count_parameters(model)
    print(f"    Parameters: {n_params:,}")

    # Create DataLoaders
    train_loader, val_loader = create_enhanced_dataloaders(
        X_train, y_train, X_val, y_val,
    )

    # Train
    start_time = time.time()
    train_result = train_enhanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=loss_type,
        target_type=target_type,
        nyc_prev_val=nyc_prev_val,
        actual_tmax_val=actual_tmax_val,
        config_dict={
            "max_epochs": max_epochs,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        },
        device=device,
        models_dir=MODELS_DIR,
        model_name=name,
    )
    elapsed = time.time() - start_time

    print(f"    Training: {elapsed:.1f}s, best epoch {train_result['best_epoch']}, "
          f"val MAE {train_result['best_val_mae']:.3f} F")

    # Evaluate on test set
    test_result = evaluate_on_test(
        model=train_result["model"],
        X_test=X_test,
        y_test=y_test,
        target_type=target_type,
        nyc_prev_test=nyc_prev_test,
        actual_tmax_test=actual_tmax_test,
        device=device,
    )

    print(f"    Test MAE: {test_result['mae']:.3f} F | "
          f"RMSE: {test_result['rmse']:.3f} | R2: {test_result['r2']:.4f}")

    # Save training history and curves
    safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("/", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_training_history_v2(
        train_result["history"],
        os.path.join(RESULTS_DIR, f"{safe_name}_history.csv"),
    )
    plot_training_curves_v2(
        train_result["history"],
        os.path.join(RESULTS_DIR, f"{safe_name}_curves.png"),
        title=name,
    )

    return {
        "name": name,
        "mae": test_result["mae"],
        "rmse": test_result["rmse"],
        "r2": test_result["r2"],
        "bias": test_result["bias"],
        "n": test_result["n"],
        "best_epoch": train_result["best_epoch"],
        "best_val_mae": train_result["best_val_mae"],
        "n_features": n_features,
        "n_params": n_params,
        "loss_type": loss_type,
        "target_type": target_type,
        "elapsed": elapsed,
        "predictions": test_result["predictions"],
        "actuals": test_result["actuals"],
        "history": train_result["history"],
    }


def main():
    """Run all Phase 4 experiments."""
    _print_section("Phase 4: Delta-T, Autoregressive, Huber, Feature Engineering")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==================================================================
    # 1. Run enhanced preprocessing with different feature configurations
    # ==================================================================
    _print_section("Step 1: Preprocessing")

    # (a) and (b): Same features as Phase 3 (no autoregressive, no enhanced)
    print("  Preprocessing for experiments (a) and (b): baseline features...")
    data_baseline = run_enhanced_preprocessing(
        output_dir=os.path.join(PROCESSED_V2_DIR, "baseline"),
        include_autoregressive=False,
        include_diurnal=False,
        include_sectors=False,
        include_trends=False,
        lags=[1],
    )

    # (c): No autoregressive, no enhanced features (same base features)
    # Uses delta target from data_baseline (already computed)
    print("  Preprocessing for experiment (c): delta-T without autoregressive...")
    data_no_ar = data_baseline  # Same features, different target

    # (d): With autoregressive, no other enhanced features
    print("  Preprocessing for experiment (d): delta-T with autoregressive...")
    data_ar = run_enhanced_preprocessing(
        output_dir=os.path.join(PROCESSED_V2_DIR, "autoregressive"),
        include_autoregressive=True,
        include_diurnal=False,
        include_sectors=False,
        include_trends=False,
        lags=[1],
    )

    # (e): All enhanced features
    print("  Preprocessing for experiment (e): all enhanced features...")
    data_full = run_enhanced_preprocessing(
        output_dir=os.path.join(PROCESSED_V2_DIR, "full"),
        include_autoregressive=True,
        include_diurnal=True,
        include_sectors=True,
        include_trends=True,
        lags=[1],
    )

    # ==================================================================
    # 2. Run experiments
    # ==================================================================
    _print_section("Step 2: Training Experiments")

    all_results = {}

    # --- (a) NN V1 reproduced: raw TMAX + MSE ---
    result_a = run_experiment(
        name="NN Raw+MSE",
        X_train=data_baseline["X_train"],
        y_train=data_baseline["y_train"],
        X_val=data_baseline["X_val"],
        y_val=data_baseline["y_val"],
        X_test=data_baseline["X_test"],
        y_test=data_baseline["y_test"],
        loss_type="mse",
        target_type="raw",
    )
    all_results[result_a["name"]] = result_a
    print()

    # --- (b) Raw TMAX + Huber ---
    result_b = run_experiment(
        name="NN Raw+Huber",
        X_train=data_baseline["X_train"],
        y_train=data_baseline["y_train"],
        X_val=data_baseline["X_val"],
        y_val=data_baseline["y_val"],
        X_test=data_baseline["X_test"],
        y_test=data_baseline["y_test"],
        loss_type="huber",
        target_type="raw",
    )
    all_results[result_b["name"]] = result_b
    print()

    # --- (c) Delta-T + Huber + NO autoregressive ---
    result_c = run_experiment(
        name="NN Delta+Huber (no AR)",
        X_train=data_no_ar["X_train"],
        y_train=data_no_ar["y_train_delta"],
        X_val=data_no_ar["X_val"],
        y_val=data_no_ar["y_val_delta"],
        X_test=data_no_ar["X_test"],
        y_test=data_no_ar["y_test_delta"],
        loss_type="huber",
        target_type="delta",
        nyc_prev_val=data_no_ar["nyc_prev_val"].values,
        nyc_prev_test=data_no_ar["nyc_prev_test"].values,
        actual_tmax_val=data_no_ar["y_val"].values,
        actual_tmax_test=data_no_ar["y_test"].values,
    )
    all_results[result_c["name"]] = result_c
    print()

    # --- (d) Delta-T + Huber + WITH autoregressive ---
    result_d = run_experiment(
        name="NN Delta+Huber+AR",
        X_train=data_ar["X_train"],
        y_train=data_ar["y_train_delta"],
        X_val=data_ar["X_val"],
        y_val=data_ar["y_val_delta"],
        X_test=data_ar["X_test"],
        y_test=data_ar["y_test_delta"],
        loss_type="huber",
        target_type="delta",
        nyc_prev_val=data_ar["nyc_prev_val"].values,
        nyc_prev_test=data_ar["nyc_prev_test"].values,
        actual_tmax_val=data_ar["y_val"].values,
        actual_tmax_test=data_ar["y_test"].values,
    )
    all_results[result_d["name"]] = result_d
    print()

    # --- (e) Delta-T + Huber + ALL enhanced features ---
    result_e = run_experiment(
        name="NN Delta+Huber+Full",
        X_train=data_full["X_train"],
        y_train=data_full["y_train_delta"],
        X_val=data_full["X_val"],
        y_val=data_full["y_val_delta"],
        X_test=data_full["X_test"],
        y_test=data_full["y_test_delta"],
        loss_type="huber",
        target_type="delta",
        nyc_prev_val=data_full["nyc_prev_val"].values,
        nyc_prev_test=data_full["nyc_prev_test"].values,
        actual_tmax_val=data_full["y_val"].values,
        actual_tmax_test=data_full["y_test"].values,
    )
    all_results[result_e["name"]] = result_e
    print()

    # ==================================================================
    # 3. Results comparison
    # ==================================================================
    _print_section("Step 3: Results Comparison")

    # Build comparison dict for format_metrics_table
    comparison = {}

    # Add reference baselines
    for name, metrics in REFERENCE_RESULTS.items():
        comparison[name] = {
            "n": 274,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "bias": float("nan"),
            "within_1f": float("nan"),
            "within_2f": float("nan"),
            "within_3f": float("nan"),
            "max_abs_error": float("nan"),
        }

    # Add Phase 4 results
    for name, result in all_results.items():
        full_metrics = compute_metrics(
            result["actuals"], result["predictions"], model_name=name,
        )
        comparison[name] = full_metrics

    table = format_metrics_table(comparison)
    print(table)
    print()

    # Headline comparison: (c) vs (d) shows value of autoregressive input
    mae_no_ar = result_c["mae"]
    mae_ar = result_d["mae"]
    print(f"  Autoregressive comparison:")
    print(f"    Without AR (surrounding stations only): {mae_no_ar:.3f} F")
    print(f"    With AR (+ NYC TMAX t-1):               {mae_ar:.3f} F")
    print(f"    AR improvement:                         {mae_no_ar - mae_ar:.3f} F")
    print()

    # Best model vs best baseline
    best_phase4_name = min(all_results, key=lambda k: all_results[k]["mae"])
    best_phase4_mae = all_results[best_phase4_name]["mae"]
    best_baseline_mae = REFERENCE_RESULTS["Ridge (alpha=1.0)"]["mae"]

    print(f"  Best Phase 4 model: {best_phase4_name}")
    print(f"    Test MAE:  {best_phase4_mae:.3f} F")
    print(f"    vs Ridge:  improvement = {best_baseline_mae - best_phase4_mae:.3f} F")
    print(f"    vs NN V1:  improvement = {REFERENCE_RESULTS['NN V1 (Phase 3)']['mae'] - best_phase4_mae:.3f} F")
    if best_phase4_mae <= 2.0:
        print("    ** STRETCH GOAL ACHIEVED: MAE <= 2.0 F **")
    print()

    # ==================================================================
    # 4. Generate plots
    # ==================================================================
    _print_section("Step 4: Generating Plots")

    # Comparison bar chart
    phase4_comparison = {n: {"mae": r["mae"]} for n, r in all_results.items()}
    phase4_comparison.update(
        {n: {"mae": m["mae"]} for n, m in REFERENCE_RESULTS.items()}
    )
    plot_baseline_comparison(
        phase4_comparison, metric="mae",
        save_path=os.path.join(RESULTS_DIR, "phase4_mae_comparison.png"),
    )

    # Per-model scatter and residual plots for the best model
    for name, result in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("/", "_")
        safe_name = safe_name.replace("(", "").replace(")", "")

        plot_actual_vs_predicted(
            result["actuals"], result["predictions"],
            name,
            os.path.join(RESULTS_DIR, f"{safe_name}_scatter.png"),
        )
        plot_residual_histogram(
            result["actuals"], result["predictions"],
            name,
            os.path.join(RESULTS_DIR, f"{safe_name}_residual_hist.png"),
        )

    # ==================================================================
    # 5. Save results summary
    # ==================================================================
    _print_section("Step 5: Saving Results Summary")

    # Save comparison table
    report_path = os.path.join(RESULTS_DIR, "phase4_report.txt")
    report_lines = [
        "=" * 70,
        "NYC Temperature Prediction -- Phase 4 Results",
        "Delta-T Target, Autoregressive Input, Huber Loss, Feature Engineering",
        "=" * 70,
        "",
        table,
        "",
        "Autoregressive Comparison:",
        f"  Without AR: {mae_no_ar:.3f} F",
        f"  With AR:    {mae_ar:.3f} F",
        f"  Improvement: {mae_no_ar - mae_ar:.3f} F",
        "",
        f"Best Phase 4 model: {best_phase4_name} (MAE={best_phase4_mae:.3f} F)",
        "",
        "--- Experiment Details ---",
        "",
    ]

    for name, result in all_results.items():
        report_lines.extend([
            f"  {name}:",
            f"    Features: {result['n_features']}  |  Parameters: {result['n_params']:,}",
            f"    Loss: {result['loss_type']}  |  Target: {result['target_type']}",
            f"    Best epoch: {result['best_epoch']}  |  Training time: {result['elapsed']:.1f}s",
            f"    Test MAE: {result['mae']:.3f}  |  RMSE: {result['rmse']:.3f}  |  R2: {result['r2']:.4f}",
            "",
        ])

    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to {report_path}")

    # Save metrics as JSON
    metrics_json = {}
    for name, result in all_results.items():
        metrics_json[name] = {
            "mae": result["mae"],
            "rmse": result["rmse"],
            "r2": result["r2"],
            "bias": result["bias"],
            "n_features": result["n_features"],
            "n_params": result["n_params"],
            "best_epoch": result["best_epoch"],
            "loss_type": result["loss_type"],
            "target_type": result["target_type"],
        }

    json_path = os.path.join(RESULTS_DIR, "phase4_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics JSON saved to {json_path}")

    # List all saved files
    print()
    print("  Files in results/phase4/:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")

    _print_section("Phase 4 Complete")
    print(f"  Best model: {best_phase4_name} (MAE={best_phase4_mae:.3f} F)")
    print()


if __name__ == "__main__":
    main()
