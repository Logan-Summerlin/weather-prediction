"""
Station Count Sensitivity Experiments for NYC Temperature Prediction.

Runs the best Phase 4 model (NN Delta+Huber+AR) with varying numbers
of surrounding stations to determine how station count affects
prediction accuracy.

Station counts tested: 5, 10, 14 (original), 20, 30, 40, 50
(or whatever max is available).

Results are saved to results/station_sensitivity/.

Usage:
    python run_station_sensitivity.py
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
    select_stations_by_count,
    compute_station_metadata,
    PROCESSED_EXPANDED_DIR,
)
from src.train_v2 import (
    create_enhanced_dataloaders,
    train_enhanced_model,
    evaluate_on_test,
    save_training_history_v2,
    plot_training_curves_v2,
)
from src.evaluate import compute_metrics

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "station_sensitivity")
MODELS_DIR = os.path.join(config.MODELS_DIR, "station_sensitivity")

# Default station counts to test
DEFAULT_COUNTS = [5, 10, 14, 20, 30, 40, 50]

# Training settings for sensitivity experiments
MAX_EPOCHS = 100
PATIENCE = 15


def _print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def train_and_evaluate_delta_ar(
    data: dict,
    name: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict:
    """Train NN Delta+Huber+AR model and evaluate on test set.

    This is the best Phase 4 configuration, used consistently across
    all station count experiments.

    Parameters
    ----------
    data : dict
        Output of run_expanded_preprocessing().
    name : str
        Experiment name.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.

    Returns
    -------
    dict
        Results with mae, rmse, r2, etc.
    """
    n_features = data["n_features"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adapt architecture to feature count
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

    train_loader, val_loader = create_enhanced_dataloaders(
        data["X_train"], data["y_train_delta"],
        data["X_val"], data["y_val_delta"],
    )

    start_time = time.time()
    train_result = train_enhanced_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type="huber",
        target_type="delta",
        nyc_prev_val=data["nyc_prev_val"].values,
        actual_tmax_val=data["y_val"].values,
        config_dict={
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
        },
        device=device,
        models_dir=MODELS_DIR,
        model_name=name,
    )
    elapsed = time.time() - start_time

    test_result = evaluate_on_test(
        model=train_result["model"],
        X_test=data["X_test"],
        y_test=data["y_test_delta"],
        target_type="delta",
        nyc_prev_test=data["nyc_prev_test"].values,
        actual_tmax_test=data["y_test"].values,
        device=device,
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
        "elapsed": elapsed,
        "history": train_result["history"],
    }


def plot_station_count_vs_mae(
    results: list[dict],
    save_path: str,
    reference_mae: float = 3.95,
) -> None:
    """Generate station-count vs MAE line plot.

    Parameters
    ----------
    results : list[dict]
        List of experiment results with 'n_stations' and 'mae' keys.
    save_path : str
        Path to save the figure.
    reference_mae : float
        14-station reference MAE for annotation.
    """
    counts = [r["n_stations"] for r in results]
    maes = [r["mae"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(counts, maes, "o-", linewidth=2, markersize=8,
            color="#2ca02c", label="NN Delta+Huber+AR")

    # Annotate each point
    for c, m in zip(counts, maes):
        ax.annotate(f"{m:.2f}", (c, m),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, fontweight="bold")

    # Reference line
    ax.axhline(reference_mae, color="red", linestyle="--", linewidth=1.5,
               label=f"14-station reference ({reference_mae:.2f} F)")

    ax.set_xlabel("Number of Surrounding Stations", fontsize=12)
    ax.set_ylabel("Test MAE (deg F)", fontsize=12)
    ax.set_title("Station Count Sensitivity: MAE vs Number of Stations",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks to actual counts
    ax.set_xticks(counts)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {save_path}")


def plot_station_count_vs_metrics(
    results: list[dict],
    save_path: str,
) -> None:
    """Generate multi-metric station-count comparison plot.

    Parameters
    ----------
    results : list[dict]
        List of experiment results.
    save_path : str
        Path to save the figure.
    """
    counts = [r["n_stations"] for r in results]
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    r2s = [r["r2"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE
    axes[0].plot(counts, maes, "o-", linewidth=2, markersize=7, color="#2ca02c")
    axes[0].set_xlabel("Stations")
    axes[0].set_ylabel("MAE (deg F)")
    axes[0].set_title("MAE")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(counts)

    # RMSE
    axes[1].plot(counts, rmses, "s-", linewidth=2, markersize=7, color="#ff7f0e")
    axes[1].set_xlabel("Stations")
    axes[1].set_ylabel("RMSE (deg F)")
    axes[1].set_title("RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(counts)

    # R2
    axes[2].plot(counts, r2s, "^-", linewidth=2, markersize=7, color="#4c72b0")
    axes[2].set_xlabel("Stations")
    axes[2].set_ylabel("R-squared")
    axes[2].set_title("R-squared")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(counts)

    fig.suptitle("Station Count Sensitivity: All Metrics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Multi-metric plot saved to {save_path}")


def main():
    """Run station count sensitivity experiments."""
    _print_section("Station Count Sensitivity Experiments")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==================================================================
    # 1. Discover available stations
    # ==================================================================
    available = discover_available_stations()
    n_surrounding = len([s for s in available if s != config.TARGET_STATION])
    print(f"  Available: {len(available)} total, {n_surrounding} surrounding")
    print(f"  Stations: {available}")
    print()

    if n_surrounding < 5:
        print("ERROR: Need at least 5 surrounding stations for sensitivity analysis.")
        return

    # Determine which counts to test
    counts_to_test = [c for c in DEFAULT_COUNTS if c <= n_surrounding]
    if n_surrounding not in counts_to_test:
        counts_to_test.append(n_surrounding)
    counts_to_test = sorted(set(counts_to_test))

    print(f"  Station counts to test: {counts_to_test}")
    print()

    # ==================================================================
    # 2. Run experiments for each station count
    # ==================================================================
    all_results = []

    for i, n in enumerate(counts_to_test, 1):
        _print_section(
            f"Experiment {i}/{len(counts_to_test)}: {n} Stations"
        )

        # Select stations
        if n >= n_surrounding:
            # Use all available
            selected = [s for s in available if s != config.TARGET_STATION]
        else:
            selected = select_stations_by_count(available, n)

        actual_n = len(selected)
        print(f"  Selected {actual_n} stations")

        # Preprocess with selected stations
        try:
            data = run_expanded_preprocessing(
                station_list=selected,
                include_missingness_mask=True,
                include_autoregressive=True,
                include_diurnal=False,
                include_sectors=False,
                include_trends=False,
                lags=[1],
                output_dir=os.path.join(
                    PROCESSED_EXPANDED_DIR, f"sensitivity_{actual_n}st"
                ),
            )
        except Exception as e:
            print(f"  ERROR in preprocessing: {e}")
            continue

        # Train and evaluate
        exp_name = f"{actual_n}st_Delta+Huber+AR"
        try:
            result = train_and_evaluate_delta_ar(data, exp_name)
            result["n_stations"] = actual_n
            result["station_ids"] = selected

            print(f"  Result: MAE={result['mae']:.3f} F, "
                  f"RMSE={result['rmse']:.3f}, R2={result['r2']:.4f}")

            # Save history
            safe_name = exp_name.lower().replace("+", "_")
            save_training_history_v2(
                result["history"],
                os.path.join(RESULTS_DIR, f"{safe_name}_history.csv"),
            )

            all_results.append(result)

        except Exception as e:
            print(f"  ERROR in training: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No successful experiments. Exiting.")
        return

    # ==================================================================
    # 3. Results comparison
    # ==================================================================
    _print_section("Results Comparison")

    # Print table
    print(f"  {'Stations':>10s}  {'MAE':>7s}  {'RMSE':>7s}  {'R2':>8s}  "
          f"{'Features':>10s}  {'Params':>8s}  {'Time':>6s}")
    print(f"  {'-' * 10}  {'-' * 7}  {'-' * 7}  {'-' * 8}  "
          f"{'-' * 10}  {'-' * 8}  {'-' * 6}")

    for r in sorted(all_results, key=lambda x: x["n_stations"]):
        print(f"  {r['n_stations']:>10d}  {r['mae']:>7.3f}  {r['rmse']:>7.3f}  "
              f"{r['r2']:>8.4f}  {r['n_features']:>10d}  "
              f"{r['n_params']:>8,d}  {r['elapsed']:>5.1f}s")

    print()

    # Find optimal station count
    best = min(all_results, key=lambda x: x["mae"])
    print(f"  Best station count: {best['n_stations']} stations "
          f"(MAE={best['mae']:.3f} F)")

    # Diminishing returns analysis
    sorted_results = sorted(all_results, key=lambda x: x["n_stations"])
    if len(sorted_results) >= 2:
        print()
        print("  Marginal improvement per additional station block:")
        for i in range(1, len(sorted_results)):
            prev = sorted_results[i - 1]
            curr = sorted_results[i]
            delta_mae = prev["mae"] - curr["mae"]
            delta_stations = curr["n_stations"] - prev["n_stations"]
            per_station = delta_mae / delta_stations if delta_stations > 0 else 0
            direction = "improvement" if delta_mae > 0 else "regression"
            print(f"    {prev['n_stations']} -> {curr['n_stations']} stations: "
                  f"{delta_mae:+.3f} F ({direction}, "
                  f"{per_station:+.4f} F/station)")

    # ==================================================================
    # 4. Generate plots
    # ==================================================================
    _print_section("Generating Plots")

    plot_station_count_vs_mae(
        all_results,
        os.path.join(RESULTS_DIR, "station_count_vs_mae.png"),
    )

    plot_station_count_vs_metrics(
        all_results,
        os.path.join(RESULTS_DIR, "station_count_all_metrics.png"),
    )

    # ==================================================================
    # 5. Save results
    # ==================================================================
    _print_section("Saving Results")

    # CSV table
    rows = []
    for r in sorted(all_results, key=lambda x: x["n_stations"]):
        rows.append({
            "n_stations": r["n_stations"],
            "mae": round(r["mae"], 3),
            "rmse": round(r["rmse"], 3),
            "r2": round(r["r2"], 4),
            "bias": round(r.get("bias", 0), 3),
            "n_features": r["n_features"],
            "n_params": r["n_params"],
            "best_epoch": r["best_epoch"],
            "train_time_s": round(r["elapsed"], 1),
        })

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "station_sensitivity_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV saved to {csv_path}")

    # JSON with station lists
    json_data = {
        "experiments": [],
        "best_n_stations": best["n_stations"],
        "best_mae": best["mae"],
    }
    for r in sorted(all_results, key=lambda x: x["n_stations"]):
        json_data["experiments"].append({
            "n_stations": r["n_stations"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "r2": r["r2"],
            "n_features": r["n_features"],
            "n_params": r["n_params"],
            "station_ids": r.get("station_ids", []),
        })

    json_path = os.path.join(RESULTS_DIR, "station_sensitivity_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved to {json_path}")

    # Text report
    report_lines = [
        "=" * 70,
        "Station Count Sensitivity Report",
        "=" * 70,
        "",
        f"Model: NN Delta+Huber+AR (best Phase 4 config)",
        f"Available stations: {n_surrounding}",
        f"Counts tested: {counts_to_test}",
        "",
        "Results:",
        "",
        results_df.to_string(index=False),
        "",
        f"Best: {best['n_stations']} stations, MAE={best['mae']:.3f} F",
        "",
        "=" * 70,
    ]
    report_path = os.path.join(RESULTS_DIR, "station_sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report saved to {report_path}")

    # List files
    print()
    print("  Files in results/station_sensitivity/:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")

    _print_section("Station Sensitivity Analysis Complete")
    print(f"  Best: {best['n_stations']} stations (MAE={best['mae']:.3f} F)")
    print()

    return all_results


if __name__ == "__main__":
    main()
