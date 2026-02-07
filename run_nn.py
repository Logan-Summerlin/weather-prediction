"""
Run the Neural Network V1 model for NYC Temperature Prediction.

End-to-end script that:
  1. Loads preprocessed data from data/processed/
  2. Creates the TempPredictorV1 model
  3. Creates DataLoaders and trains the model
  4. Evaluates on the test set using the evaluation framework
  5. Compares results against baseline numbers
  6. Saves all results to results/nn_v1/
  7. Generates diagnostic plots

Usage:
    python run_nn.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import config
from src.train import (
    load_processed_data,
    create_dataloaders,
    train_model,
    validate,
    save_training_history,
    plot_training_curves,
)
from src.evaluate import (
    compute_metrics,
    format_metrics_table,
    evaluate_predictions,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_time_series,
    plot_residuals_by_month,
)


# Baseline reference numbers (from Phase 2)
BASELINE_RESULTS = {
    "Persistence": {"mae": 5.06, "rmse": 6.39, "r2": 0.799},
    "Climatology": {"mae": 6.15, "rmse": 7.72, "r2": 0.747},
    "Linear Regression": {"mae": 4.35, "rmse": 5.43, "r2": 0.875},
    "Ridge (alpha=1.0)": {"mae": 4.33, "rmse": 5.41, "r2": 0.876},
}


def main():
    """Run the full neural network training and evaluation pipeline."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Neural Network V1")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------------------
    print("Step 1: Loading processed data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run the data pipeline first:")
        print("  python -m src.data_collection")
        print("  python -m src.data_preprocessing")
        sys.exit(1)

    n_features = X_train.shape[1]
    print(f"  Features: {n_features}")
    print(f"  Train: {len(X_train)} rows  |  Val: {len(X_val)} rows  "
          f"|  Test: {len(X_test)} rows")
    print(f"  Target range: [{y_train.min():.1f}, {y_train.max():.1f}] F")
    print()

    # ------------------------------------------------------------------
    # 2. Create model
    # ------------------------------------------------------------------
    print("Step 2: Creating TempPredictorV1 model...")
    try:
        from src.model import create_model, count_parameters
        model = create_model(n_features)
        n_params = count_parameters(model)
    except ImportError:
        print("  WARNING: src.model not available yet. "
              "Using inline fallback model.")
        import torch.nn as nn

        class FallbackModel(nn.Module):
            def __init__(self, n_feat, hidden_sizes=None, dropout=None):
                super().__init__()
                if hidden_sizes is None:
                    hidden_sizes = config.HIDDEN_SIZES
                if dropout is None:
                    dropout = config.DROPOUT
                layers = []
                prev_size = n_feat
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev_size, h))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    prev_size = h
                layers.append(nn.Linear(prev_size, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        model = FallbackModel(n_features)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Architecture: {model}")
    print(f"  Trainable parameters: {n_params:,}")
    print()

    # ------------------------------------------------------------------
    # 3. Create DataLoaders
    # ------------------------------------------------------------------
    print("Step 3: Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
    )
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()

    # ------------------------------------------------------------------
    # 4. Train the model
    # ------------------------------------------------------------------
    print("Step 4: Training model...")
    print(f"  Max epochs: {config.MAX_EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print()

    start_time = time.time()
    result = train_model(
        model, train_loader, val_loader,
        device=device,
    )
    elapsed = time.time() - start_time

    print()
    print(f"  Training complete in {elapsed:.1f} seconds")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val MAE: {result['best_val_mae']:.3f} F")
    print()

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    print("Step 5: Evaluating on test set...")

    # Create a test DataLoader for evaluation
    test_X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    test_dataset = torch.utils.data.TensorDataset(test_X_tensor, test_y_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    )

    trained_model = result["model"]
    trained_model.eval()
    criterion = torch.nn.MSELoss()

    test_loss, test_preds, test_actuals = validate(
        trained_model, test_loader, criterion, device=device,
    )

    # Compute comprehensive metrics using evaluate module
    output_dir = os.path.join(config.RESULTS_DIR, "nn_v1")
    os.makedirs(output_dir, exist_ok=True)

    test_metrics = evaluate_predictions(
        y_actual=test_actuals,
        y_pred=test_preds,
        dates=X_test.index,
        model_name="NN V1",
        output_dir=output_dir,
    )

    print(f"  Test MAE:  {test_metrics['mae']:.3f} F")
    print(f"  Test RMSE: {test_metrics['rmse']:.3f} F")
    print(f"  Test R2:   {test_metrics['r2']:.4f}")
    print(f"  Test Bias: {test_metrics['bias']:.3f} F")
    print()

    # ------------------------------------------------------------------
    # 6. Compare against baselines
    # ------------------------------------------------------------------
    print("Step 6: Comparison with baselines...")
    print()

    all_results = {}
    for name, baseline_metrics in BASELINE_RESULTS.items():
        all_results[name] = baseline_metrics
    all_results["NN V1"] = test_metrics

    table = format_metrics_table(all_results)
    print(table)
    print()

    # Improvement over best baseline (Ridge)
    ridge_mae = BASELINE_RESULTS["Ridge (alpha=1.0)"]["mae"]
    nn_mae = test_metrics["mae"]
    improvement = ridge_mae - nn_mae
    pct_improvement = (improvement / ridge_mae) * 100

    print(f"  NN V1 MAE:           {nn_mae:.3f} F")
    print(f"  Best baseline MAE:   {ridge_mae:.2f} F (Ridge)")
    print(f"  Improvement:         {improvement:.3f} F ({pct_improvement:.1f}%)")
    if nn_mae <= 2.0:
        print("  ** STRETCH GOAL ACHIEVED: MAE <= 2.0 F **")
    elif nn_mae < ridge_mae:
        print("  NN outperforms best baseline.")
    else:
        print("  NOTE: NN does not outperform best baseline. "
              "Consider tuning hyperparameters.")
    print()

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print("Step 7: Saving results...")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.csv")
    save_training_history(result["history"], history_path)

    # Save training curves
    curves_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(result["history"], curves_path)

    # Save test predictions
    preds_df = pd.DataFrame({
        "date": X_test.index,
        "actual": test_actuals,
        "predicted": test_preds,
        "residual": test_preds - test_actuals,
    })
    preds_path = os.path.join(output_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    # Save summary report
    report_lines = [
        "=" * 70,
        "NYC Temperature Prediction -- Neural Network V1 Results",
        "=" * 70,
        "",
        f"Model: TempPredictorV1",
        f"Parameters: {n_params:,}",
        f"Hidden sizes: {config.HIDDEN_SIZES}",
        f"Dropout: {config.DROPOUT}",
        f"Learning rate: {config.LEARNING_RATE}",
        f"Batch size: {config.BATCH_SIZE}",
        f"Best epoch: {result['best_epoch']}",
        f"Training time: {elapsed:.1f} seconds",
        "",
        "--- Test Set Metrics ---",
        "",
        f"MAE:  {test_metrics['mae']:.3f} F",
        f"RMSE: {test_metrics['rmse']:.3f} F",
        f"R2:   {test_metrics['r2']:.4f}",
        f"Bias: {test_metrics['bias']:.3f} F",
        "",
        "--- Comparison with Baselines ---",
        "",
        table,
        "",
        f"Improvement over Ridge: {improvement:.3f} F ({pct_improvement:.1f}%)",
        "",
        "=" * 70,
    ]
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "nn_v1_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print()
    print(f"  All results saved to: {output_dir}")
    print()
    saved_files = sorted(os.listdir(output_dir))
    for fname in saved_files:
        fpath = os.path.join(output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")

    print()
    print("Neural Network V1 evaluation complete.")


if __name__ == "__main__":
    main()
