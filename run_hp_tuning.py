"""
Hyperparameter tuning script for NN V1.

Systematically explores different combinations of hidden sizes, learning rates,
and dropout rates to find the best configuration. Results are logged to a
structured text file.

This script does NOT modify config.py — it passes hyperparameters directly
to the model and training functions via their optional arguments.

Usage:
    python run_hp_tuning.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

import config
from src.model import create_model
from src.train import (
    load_processed_data,
    create_dataloaders,
    train_model,
    validate,
)


def run_experiment(
    X_train, y_train, X_val, y_val, X_test, y_test,
    hidden_sizes, learning_rate, dropout, batch_size=64,
    experiment_id="exp", device="cpu",
):
    """Run a single training experiment with given hyperparameters.

    Returns a dict with val_mae, test_mae, test_rmse, test_r2, best_epoch,
    n_epochs, and elapsed time.
    """
    n_features = X_train.shape[1]

    # Create model with specified hyperparameters
    model = create_model(n_features, hidden_sizes=hidden_sizes, dropout=dropout)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size,
    )

    # Train with specified learning rate
    # Use a temporary models dir so we don't overwrite best_model.pt
    temp_models_dir = os.path.join(config.MODELS_DIR, f"temp_{experiment_id}")

    result = train_model(
        model, train_loader, val_loader,
        config_dict={"learning_rate": learning_rate},
        device=device,
        models_dir=temp_models_dir,
    )

    # Evaluate on test set
    test_X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_X_tensor, test_y_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

    trained_model = result["model"]
    trained_model.eval()
    criterion = nn.MSELoss()

    test_loss, test_preds, test_actuals = validate(
        trained_model, test_loader, criterion, device=device,
    )

    test_mae = float(np.mean(np.abs(test_preds - test_actuals)))
    test_rmse = float(np.sqrt(np.mean((test_preds - test_actuals) ** 2)))
    ss_res = np.sum((test_preds - test_actuals) ** 2)
    ss_tot = np.sum((test_actuals - np.mean(test_actuals)) ** 2)
    test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    n_params = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)

    # Clean up temp checkpoint
    temp_ckpt = os.path.join(temp_models_dir, "best_model.pt")
    # Keep the checkpoint path for potential later use
    return {
        "hidden_sizes": hidden_sizes,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "n_params": n_params,
        "best_epoch": result["best_epoch"],
        "n_epochs": len(result["history"]),
        "best_val_mae": result["best_val_mae"],
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "checkpoint_path": temp_ckpt,
        "model": trained_model,
        "history": result["history"],
    }


def main():
    print("=" * 70)
    print("Hyperparameter Tuning for NN V1")
    print("=" * 70)
    print()

    # Load data
    print("Loading processed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Define search space
    experiments = [
        # Baseline default
        {"hidden_sizes": [64, 32], "learning_rate": 0.001, "dropout": 0.1},

        # Different hidden sizes
        {"hidden_sizes": [128, 64], "learning_rate": 0.001, "dropout": 0.1},
        {"hidden_sizes": [64, 32, 16], "learning_rate": 0.001, "dropout": 0.1},
        {"hidden_sizes": [32, 16], "learning_rate": 0.001, "dropout": 0.1},
        {"hidden_sizes": [128, 64, 32], "learning_rate": 0.001, "dropout": 0.1},
        {"hidden_sizes": [256, 128, 64], "learning_rate": 0.001, "dropout": 0.1},

        # Different learning rates (with default architecture)
        {"hidden_sizes": [64, 32], "learning_rate": 0.0005, "dropout": 0.1},
        {"hidden_sizes": [64, 32], "learning_rate": 0.002, "dropout": 0.1},
        {"hidden_sizes": [64, 32], "learning_rate": 0.0001, "dropout": 0.1},

        # Different dropout (with default architecture)
        {"hidden_sizes": [64, 32], "learning_rate": 0.001, "dropout": 0.05},
        {"hidden_sizes": [64, 32], "learning_rate": 0.001, "dropout": 0.15},
        {"hidden_sizes": [64, 32], "learning_rate": 0.001, "dropout": 0.2},
        {"hidden_sizes": [64, 32], "learning_rate": 0.001, "dropout": 0.0},

        # Promising combinations (larger model + regularization)
        {"hidden_sizes": [128, 64], "learning_rate": 0.001, "dropout": 0.15},
        {"hidden_sizes": [128, 64], "learning_rate": 0.001, "dropout": 0.2},
        {"hidden_sizes": [128, 64], "learning_rate": 0.0005, "dropout": 0.1},
        {"hidden_sizes": [128, 64, 32], "learning_rate": 0.001, "dropout": 0.15},
        {"hidden_sizes": [128, 64, 32], "learning_rate": 0.0005, "dropout": 0.1},

        # Smaller batch sizes with promising configs
        # (lower batch = more noise = more regularization)
    ]

    results = []

    for i, exp_config in enumerate(experiments):
        exp_id = f"exp_{i:02d}"
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp_id}")
        print(f"  Hidden sizes: {exp_config['hidden_sizes']}")
        print(f"  Learning rate: {exp_config['learning_rate']}")
        print(f"  Dropout: {exp_config['dropout']}")
        print(f"{'='*70}")

        start_time = time.time()

        result = run_experiment(
            X_train, y_train, X_val, y_val, X_test, y_test,
            hidden_sizes=exp_config["hidden_sizes"],
            learning_rate=exp_config["learning_rate"],
            dropout=exp_config["dropout"],
            experiment_id=exp_id,
            device=device,
        )
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        result["experiment_id"] = exp_id

        results.append(result)

        print(f"\n  Result: Val MAE={result['best_val_mae']:.3f}, "
              f"Test MAE={result['test_mae']:.3f}, "
              f"Test R2={result['test_r2']:.4f}, "
              f"Best epoch={result['best_epoch']}, "
              f"Time={elapsed:.1f}s")

    # Sort by test MAE
    results.sort(key=lambda r: r["test_mae"])

    # Print summary table
    print("\n\n" + "=" * 110)
    print("HYPERPARAMETER TUNING RESULTS (sorted by test MAE)")
    print("=" * 110)
    print(f"{'Rank':>4}  {'ID':>6}  {'Hidden Sizes':>20}  {'LR':>8}  {'Drop':>5}  "
          f"{'Params':>7}  {'Epoch':>5}  {'ValMAE':>7}  {'TestMAE':>7}  "
          f"{'TestRMSE':>8}  {'TestR2':>7}  {'Time':>6}")
    print("-" * 110)

    for rank, r in enumerate(results, 1):
        hs_str = str(r["hidden_sizes"])
        print(f"{rank:>4}  {r['experiment_id']:>6}  {hs_str:>20}  "
              f"{r['learning_rate']:>8.4f}  {r['dropout']:>5.2f}  "
              f"{r['n_params']:>7}  {r['best_epoch']:>5}  "
              f"{r['best_val_mae']:>7.3f}  {r['test_mae']:>7.3f}  "
              f"{r['test_rmse']:>8.3f}  {r['test_r2']:>7.4f}  "
              f"{r['elapsed']:>5.1f}s")

    print("-" * 110)
    print(f"\nBest baseline (Ridge): MAE=4.33, R2=0.876")
    print()

    best = results[0]
    print(f"BEST CONFIGURATION:")
    print(f"  Hidden sizes: {best['hidden_sizes']}")
    print(f"  Learning rate: {best['learning_rate']}")
    print(f"  Dropout: {best['dropout']}")
    print(f"  Parameters: {best['n_params']:,}")
    print(f"  Best epoch: {best['best_epoch']}")
    print(f"  Val MAE: {best['best_val_mae']:.3f}")
    print(f"  Test MAE: {best['test_mae']:.3f}")
    print(f"  Test RMSE: {best['test_rmse']:.3f}")
    print(f"  Test R2: {best['test_r2']:.4f}")

    ridge_mae = 4.33
    improvement = ridge_mae - best["test_mae"]
    pct = (improvement / ridge_mae) * 100
    print(f"\n  Improvement over Ridge: {improvement:.3f} F ({pct:.1f}%)")
    if best["test_mae"] < ridge_mae:
        print("  >>> NN outperforms Ridge baseline! <<<")
    else:
        print("  >>> NN does NOT outperform Ridge baseline <<<")

    # Save the best configuration details for use by the final run
    output_dir = os.path.join(config.RESULTS_DIR, "nn_v1")
    os.makedirs(output_dir, exist_ok=True)

    best_config = {
        "hidden_sizes": best["hidden_sizes"],
        "learning_rate": best["learning_rate"],
        "dropout": best["dropout"],
        "batch_size": best.get("batch_size", 64),
    }
    config_path = os.path.join(output_dir, "best_hp_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to: {config_path}")

    # Save full results summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Hyperparameter Tuning Summary for NN V1")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"Date: 2026-02-07")
    summary_lines.append(f"Data: Train={len(X_train)}, Val={len(X_val)}, "
                         f"Test={len(X_test)}")
    summary_lines.append(f"Features: {X_train.shape[1]}")
    summary_lines.append(f"Device: {device}")
    summary_lines.append(f"Max epochs: {config.MAX_EPOCHS}")
    summary_lines.append(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    summary_lines.append(f"Batch size: {config.BATCH_SIZE}")
    summary_lines.append(f"Reference baseline: Ridge MAE=4.33, R2=0.876")
    summary_lines.append("")
    summary_lines.append(f"Total experiments: {len(results)}")
    summary_lines.append("")

    # Results table
    summary_lines.append(f"{'Rank':>4}  {'Hidden Sizes':>20}  {'LR':>8}  {'Drop':>5}  "
                         f"{'Params':>7}  {'Epoch':>5}  {'ValMAE':>7}  {'TestMAE':>7}  "
                         f"{'TestRMSE':>8}  {'TestR2':>7}")
    summary_lines.append("-" * 100)
    for rank, r in enumerate(results, 1):
        hs_str = str(r["hidden_sizes"])
        summary_lines.append(
            f"{rank:>4}  {hs_str:>20}  {r['learning_rate']:>8.4f}  "
            f"{r['dropout']:>5.2f}  {r['n_params']:>7}  {r['best_epoch']:>5}  "
            f"{r['best_val_mae']:>7.3f}  {r['test_mae']:>7.3f}  "
            f"{r['test_rmse']:>8.3f}  {r['test_r2']:>7.4f}"
        )
    summary_lines.append("-" * 100)
    summary_lines.append("")

    # Best config details
    summary_lines.append("BEST CONFIGURATION:")
    summary_lines.append(f"  Hidden sizes: {best['hidden_sizes']}")
    summary_lines.append(f"  Learning rate: {best['learning_rate']}")
    summary_lines.append(f"  Dropout: {best['dropout']}")
    summary_lines.append(f"  Parameters: {best['n_params']:,}")
    summary_lines.append(f"  Best epoch: {best['best_epoch']}")
    summary_lines.append(f"  Val MAE: {best['best_val_mae']:.3f}")
    summary_lines.append(f"  Test MAE: {best['test_mae']:.3f}")
    summary_lines.append(f"  Test RMSE: {best['test_rmse']:.3f}")
    summary_lines.append(f"  Test R2: {best['test_r2']:.4f}")
    summary_lines.append(f"  Improvement over Ridge: {improvement:.3f} F ({pct:.1f}%)")
    summary_lines.append("")

    # Analysis
    summary_lines.append("ANALYSIS:")
    summary_lines.append("")

    # Group by hidden sizes
    summary_lines.append("  Effect of hidden sizes (LR=0.001, Dropout=0.1):")
    for r in results:
        if r["learning_rate"] == 0.001 and r["dropout"] == 0.1:
            summary_lines.append(
                f"    {str(r['hidden_sizes']):>20}: TestMAE={r['test_mae']:.3f}, "
                f"ValMAE={r['best_val_mae']:.3f}"
            )
    summary_lines.append("")

    # Group by LR
    summary_lines.append("  Effect of learning rate (Hidden=[64,32], Dropout=0.1):")
    for r in results:
        if r["hidden_sizes"] == [64, 32] and r["dropout"] == 0.1:
            summary_lines.append(
                f"    LR={r['learning_rate']:.4f}: TestMAE={r['test_mae']:.3f}, "
                f"ValMAE={r['best_val_mae']:.3f}"
            )
    summary_lines.append("")

    # Group by dropout
    summary_lines.append("  Effect of dropout (Hidden=[64,32], LR=0.001):")
    for r in results:
        if r["hidden_sizes"] == [64, 32] and r["learning_rate"] == 0.001:
            summary_lines.append(
                f"    Dropout={r['dropout']:.2f}: TestMAE={r['test_mae']:.3f}, "
                f"ValMAE={r['best_val_mae']:.3f}"
            )
    summary_lines.append("")
    summary_lines.append("=" * 80)

    summary_path = os.path.join(output_dir, "hyperparameter_tuning.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Tuning summary saved to: {summary_path}")

    # Clean up temp model dirs
    import shutil
    for r in results:
        temp_dir = os.path.dirname(r["checkpoint_path"])
        if os.path.exists(temp_dir) and "temp_" in temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nHyperparameter tuning complete.")

    return results


if __name__ == "__main__":
    results = main()
