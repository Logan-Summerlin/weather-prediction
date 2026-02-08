"""
Run Phase 4 Experiments with Expanded Station Set.

Reruns the full Phase 4 experiment suite using all available stations
(expanded from the original 14 surrounding stations to as many as are
available in data/raw/).

Trains and evaluates five model configurations:
  (a) NN with raw TMAX target + MSE loss
  (b) NN with raw TMAX + Huber loss
  (c) NN with delta-T target + Huber loss (no autoregressive)
  (d) NN with delta-T target + Huber + autoregressive NYC TMAX(t-1)
  (e) NN with delta-T + Huber + all enhanced features

Results are saved to results/phase4_expanded/ and compared against
the 14-station Phase 4 baselines.

Usage:
    python run_phase4_expanded.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import config
from src.model import TempPredictorV1, count_parameters
from src.data_preprocessing_expanded import (
    run_expanded_preprocessing,
    discover_available_stations,
    PROCESSED_EXPANDED_DIR,
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
    plot_baseline_comparison,
)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "phase4_expanded")
MODELS_DIR = os.path.join(config.MODELS_DIR, "phase4_expanded")

# Phase 4 reference results (14 stations)
PHASE4_14_STATION_RESULTS = {
    "14st: NN Raw+MSE": {"mae": 4.30, "rmse": 5.69, "r2": 0.869},
    "14st: NN Raw+Huber": {"mae": 4.36, "rmse": 5.69, "r2": 0.869},
    "14st: NN Delta+Huber (no AR)": {"mae": 4.03, "rmse": 5.41, "r2": 0.876},
    "14st: NN Delta+Huber+AR": {"mae": 3.95, "rmse": 5.33, "r2": 0.885},
    "14st: NN Delta+Huber+Full": {"mae": 4.15, "rmse": 5.50, "r2": 0.877},
}

# Earlier baselines
BASELINE_RESULTS = {
    "Persistence": {"mae": 5.06, "rmse": 6.39, "r2": 0.799},
    "Ridge (alpha=1.0)": {"mae": 4.33, "rmse": 5.41, "r2": 0.876},
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
    max_epochs: int = 100,
    patience: int = 15,
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
        'raw' or 'delta'.
    nyc_prev_val, nyc_prev_test : np.ndarray
        NYC TMAX(t-1) for reconstruction (delta only).
    actual_tmax_val, actual_tmax_test : np.ndarray
        Actual NYC TMAX for evaluation (delta only).
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.

    Returns
    -------
    dict
        Experiment results.
    """
    print(f"  Experiment: {name}")
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Loss: {loss_type} | Target: {target_type}")

    n_features = X_train.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adapt hidden sizes for larger feature counts
    if n_features > 100:
        hidden_sizes = [256, 128]
    elif n_features > 50:
        hidden_sizes = [128, 64]
    else:
        hidden_sizes = list(config.HIDDEN_SIZES)

    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=config.DROPOUT,
    )
    n_params = count_parameters(model)
    print(f"    Architecture: {hidden_sizes} | Parameters: {n_params:,}")

    train_loader, val_loader = create_enhanced_dataloaders(
        X_train, y_train, X_val, y_val,
    )

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
            "early_stopping_patience": patience,
        },
        device=device,
        models_dir=MODELS_DIR,
        model_name=name,
    )
    elapsed = time.time() - start_time

    print(f"    Training: {elapsed:.1f}s, best epoch {train_result['best_epoch']}, "
          f"val MAE {train_result['best_val_mae']:.3f} F")

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
    safe_name = safe_name.replace("(", "").replace(")", "")
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
    """Run all Phase 4 experiments with expanded stations."""
    _print_section("Phase 4 Expanded: All Available Stations")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==================================================================
    # 0. Check available stations
    # ==================================================================
    available = discover_available_stations()
    n_surrounding = len([s for s in available if s != config.TARGET_STATION])
    print(f"  Available stations: {len(available)} total, "
          f"{n_surrounding} surrounding")
    print(f"  Station IDs: {available}")
    print()

    if n_surrounding < 1:
        print("ERROR: No surrounding stations available. "
              "Run data_collection.py first.")
        return

    # ==================================================================
    # 1. Preprocessing with all available stations
    # ==================================================================
    _print_section("Step 1: Expanded Preprocessing")

    # (a) and (b): Baseline features (no AR, no enhanced)
    print("  Config (a,b): baseline features, all stations...")
    data_baseline = run_expanded_preprocessing(
        include_missingness_mask=True,
        include_autoregressive=False,
        include_diurnal=False,
        include_sectors=False,
        include_trends=False,
        lags=[1],
        output_dir=os.path.join(PROCESSED_EXPANDED_DIR, "baseline"),
    )

    # (c): Same as baseline (uses delta target)
    data_no_ar = data_baseline

    # (d): With autoregressive
    print("  Config (d): with autoregressive...")
    data_ar = run_expanded_preprocessing(
        include_missingness_mask=True,
        include_autoregressive=True,
        include_diurnal=False,
        include_sectors=False,
        include_trends=False,
        lags=[1],
        output_dir=os.path.join(PROCESSED_EXPANDED_DIR, "autoregressive"),
    )

    # (e): All enhanced features
    print("  Config (e): all enhanced features...")
    data_full = run_expanded_preprocessing(
        include_missingness_mask=True,
        include_autoregressive=True,
        include_diurnal=True,
        include_sectors=True,
        include_trends=True,
        lags=[1],
        output_dir=os.path.join(PROCESSED_EXPANDED_DIR, "full"),
    )

    # ==================================================================
    # 2. Run experiments
    # ==================================================================
    _print_section("Step 2: Training Experiments (Expanded Stations)")

    all_results = {}

    # --- (a) NN Raw+MSE ---
    result_a = run_experiment(
        name="Exp NN Raw+MSE",
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

    # --- (b) NN Raw+Huber ---
    result_b = run_experiment(
        name="Exp NN Raw+Huber",
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

    # --- (c) NN Delta+Huber (no AR) ---
    result_c = run_experiment(
        name="Exp NN Delta+Huber (no AR)",
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

    # --- (d) NN Delta+Huber+AR ---
    result_d = run_experiment(
        name="Exp NN Delta+Huber+AR",
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

    # --- (e) NN Delta+Huber+Full ---
    try:
        result_e = run_experiment(
            name="Exp NN Delta+Huber+Full",
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
    except Exception as e:
        print(f"  WARNING: Experiment (e) failed: {e}")
        import traceback
        traceback.print_exc()
    print()

    # ==================================================================
    # 3. Results comparison
    # ==================================================================
    _print_section("Step 3: Results Comparison (Expanded vs 14-Station)")

    # Build comparison table
    comparison = {}

    # Add baselines
    for name, metrics in BASELINE_RESULTS.items():
        comparison[name] = {
            "n": 274, "mae": metrics["mae"], "rmse": metrics["rmse"],
            "r2": metrics["r2"], "bias": float("nan"),
            "within_1f": float("nan"), "within_2f": float("nan"),
            "within_3f": float("nan"), "max_abs_error": float("nan"),
        }

    # Add 14-station references
    for name, metrics in PHASE4_14_STATION_RESULTS.items():
        comparison[name] = {
            "n": 274, "mae": metrics["mae"], "rmse": metrics["rmse"],
            "r2": metrics["r2"], "bias": float("nan"),
            "within_1f": float("nan"), "within_2f": float("nan"),
            "within_3f": float("nan"), "max_abs_error": float("nan"),
        }

    # Add expanded results
    for name, result in all_results.items():
        full_metrics = compute_metrics(
            result["actuals"], result["predictions"], model_name=name,
        )
        comparison[name] = full_metrics

    table = format_metrics_table(comparison)
    print(table)
    print()

    # Key comparison: expanded best vs 14-station best
    best_exp_name = min(all_results, key=lambda k: all_results[k]["mae"])
    best_exp_mae = all_results[best_exp_name]["mae"]
    best_14st_mae = 3.95  # NN Delta+Huber+AR from Phase 4

    print(f"  Best expanded model: {best_exp_name}")
    print(f"    MAE: {best_exp_mae:.3f} F")
    print(f"    vs 14-station best (3.95 F): "
          f"{'improvement' if best_exp_mae < best_14st_mae else 'regression'} "
          f"= {abs(best_14st_mae - best_exp_mae):.3f} F")
    print(f"    Stations used: {n_surrounding}")
    print()

    # ==================================================================
    # 4. Plots
    # ==================================================================
    _print_section("Step 4: Generating Plots")

    # Comparison bar chart
    bar_data = {}
    for name, metrics in PHASE4_14_STATION_RESULTS.items():
        bar_data[name] = {"mae": metrics["mae"]}
    for name, result in all_results.items():
        bar_data[name] = {"mae": result["mae"]}

    plot_baseline_comparison(
        bar_data, metric="mae",
        save_path=os.path.join(RESULTS_DIR, "expanded_vs_14station_comparison.png"),
    )

    # Per-model plots (skip experiments with no predictions)
    for name, result in all_results.items():
        if result.get("n", 0) == 0 or len(result.get("predictions", [])) == 0:
            print(f"  Skipping plots for '{name}' (no predictions)")
            continue

        safe_name = name.lower().replace(" ", "_").replace("+", "_").replace("/", "_")
        safe_name = safe_name.replace("(", "").replace(")", "")

        try:
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
        except Exception as e:
            print(f"  Warning: could not generate plot for '{name}': {e}")

    # ==================================================================
    # 5. Save results
    # ==================================================================
    _print_section("Step 5: Saving Results")

    # Report
    report_lines = [
        "=" * 70,
        "NYC Temperature Prediction -- Phase 4 Expanded Results",
        f"Stations: {n_surrounding} surrounding (expanded)",
        "=" * 70,
        "",
        table,
        "",
        f"Best expanded model: {best_exp_name} (MAE={best_exp_mae:.3f} F)",
        f"Best 14-station model: NN Delta+Huber+AR (MAE=3.95 F)",
        f"Difference: {best_14st_mae - best_exp_mae:+.3f} F",
        "",
        "--- Experiment Details ---",
        "",
    ]

    for name, result in all_results.items():
        report_lines.extend([
            f"  {name}:",
            f"    Features: {result['n_features']}  |  Parameters: {result['n_params']:,}",
            f"    Loss: {result['loss_type']}  |  Target: {result['target_type']}",
            f"    Best epoch: {result['best_epoch']}  |  Time: {result['elapsed']:.1f}s",
            f"    Test MAE: {result['mae']:.3f}  |  RMSE: {result['rmse']:.3f}  |  R2: {result['r2']:.4f}",
            "",
        ])

    report_lines.append("=" * 70)
    report_text = "\n".join(report_lines)

    report_path = os.path.join(RESULTS_DIR, "phase4_expanded_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to {report_path}")

    # JSON metrics
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
            "n_surrounding_stations": n_surrounding,
        }
    metrics_json["_metadata"] = {
        "n_stations_total": len(available),
        "n_surrounding": n_surrounding,
        "station_ids": available,
    }

    json_path = os.path.join(RESULTS_DIR, "phase4_expanded_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics saved to {json_path}")

    # List files
    print()
    print("  Files in results/phase4_expanded/:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")

    _print_section("Phase 4 Expanded Complete")
    print(f"  Best model: {best_exp_name} (MAE={best_exp_mae:.3f} F)")
    print(f"  Stations: {n_surrounding} surrounding")
    print()

    return all_results


if __name__ == "__main__":
    main()
