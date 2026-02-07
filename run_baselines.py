"""
Run all baseline models on the processed NYC temperature data.

Loads processed data from data/processed/, fits and evaluates all baseline
models (Persistence, Climatology, Linear Regression, Ridge), generates
comprehensive metrics, visualizations, and a full evaluation report.

Results are saved to results/baselines/.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from src.baselines import run_all_baselines
from src.evaluate import (
    compute_metrics,
    format_metrics_table,
    generate_baseline_report,
)


def load_processed_data():
    """Load all processed CSV files from data/processed/.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    processed_dir = config.PROCESSED_DATA_DIR

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

    print(f"Loaded processed data:")
    print(f"  X_train: {X_train.shape}  ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"  X_val:   {X_val.shape}  ({X_val.index.min().date()} to {X_val.index.max().date()})")
    print(f"  X_test:  {X_test.shape}  ({X_test.index.min().date()} to {X_test.index.max().date()})")
    print(f"  y_train: {len(y_train)} values, mean={y_train.mean():.1f} F")
    print(f"  y_val:   {len(y_val)} values, mean={y_val.mean():.1f} F")
    print(f"  y_test:  {len(y_test)} values, mean={y_test.mean():.1f} F")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Run the full baseline evaluation pipeline."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Baseline Evaluation")
    print("=" * 70)
    print()

    # 1. Load processed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

    # 2. Run all baseline models
    print("Fitting and evaluating baseline models...")
    print()
    results = run_all_baselines(X_train, X_val, X_test, y_train, y_val, y_test)

    # 3. Compute comprehensive metrics for test set using evaluate module
    output_dir = os.path.join(config.RESULTS_DIR, "baselines")
    os.makedirs(output_dir, exist_ok=True)

    # Build dicts that generate_baseline_report expects
    metrics_dict = {}
    actuals_dict = {}
    preds_dict = {}
    dates_dict = {}

    for model_name, res in results.items():
        # Compute comprehensive metrics using evaluate.compute_metrics
        test_metrics = compute_metrics(
            y_test.values,
            res["test_predictions"],
            model_name=model_name,
        )
        metrics_dict[model_name] = test_metrics
        actuals_dict[model_name] = y_test.values
        preds_dict[model_name] = res["test_predictions"]
        dates_dict[model_name] = X_test.index

    # 4. Generate full report with all plots
    print()
    print("Generating evaluation report and visualizations...")
    report_text = generate_baseline_report(
        results_dict=metrics_dict,
        output_dir=output_dir,
        dates_dict=dates_dict,
        actuals_dict=actuals_dict,
        preds_dict=preds_dict,
    )

    # 5. Print summary to console
    print()
    print(report_text)

    # 6. Print where outputs are saved
    print()
    print(f"All results saved to: {output_dir}")
    print()
    saved_files = sorted(os.listdir(output_dir))
    for f in saved_files:
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")

    print()
    print("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
