"""
Supplemental experiment: Train and evaluate NN V1 with hidden_sizes=[128, 64].

This test trains a TempPredictorV1 with a wider first hidden layer (128 vs 64)
to determine whether additional capacity improves prediction accuracy beyond
the default [64, 32] architecture.

Configuration:
  - hidden_sizes: [128, 64]
  - dropout: 0.0
  - learning_rate: 0.001
  - batch_size: 64
  - max_epochs: 200
  - early_stopping_patience: 15
  - seeds: torch=42, numpy=42

Reference baselines:
  - Ridge regression MAE: 4.33 degF
  - NN V1 [64, 32]  MAE: 4.291 degF
"""

import os
import sys
import math
import tempfile
import shutil

import numpy as np
import pytest

import torch
import torch.nn as nn

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import TempPredictorV1, count_parameters
from src.train import (
    load_processed_data,
    create_dataloaders,
    train_model,
    validate,
)
from src.evaluate import compute_metrics
import config


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

PROCESSED_DATA_EXISTS = os.path.isdir(config.PROCESSED_DATA_DIR) and all(
    os.path.isfile(os.path.join(config.PROCESSED_DATA_DIR, f))
    for f in [
        "features_train.csv", "features_val.csv", "features_test.csv",
        "target_train.csv", "target_val.csv", "target_test.csv",
    ]
)

SKIP_REASON = "Processed data files not found in data/processed/"


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

HIDDEN_SIZES = [128, 64]
DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def real_data():
    """Load the real processed data once for the module."""
    if not PROCESSED_DATA_EXISTS:
        pytest.skip(SKIP_REASON)
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


@pytest.fixture(scope="module")
def trained_result(real_data):
    """Train NN V1 [128, 64] on real data with full early-stopping run.

    Uses fixed seeds for reproducibility.  Saves checkpoints to a temp
    directory that is cleaned up after the module finishes.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    n_features = real_data["X_train"].shape[1]

    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    )

    train_loader, val_loader = create_dataloaders(
        real_data["X_train"], real_data["y_train"],
        real_data["X_val"], real_data["y_val"],
        batch_size=BATCH_SIZE,
    )

    tmp_dir = tempfile.mkdtemp(prefix="nn_v1_128_64_")

    result = train_model(
        model,
        train_loader,
        val_loader,
        config_dict={
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        models_dir=tmp_dir,
    )

    result["_tmp_dir"] = tmp_dir
    result["_train_loader"] = train_loader
    result["_val_loader"] = val_loader
    return result


# ===========================================================================
# Tests
# ===========================================================================

@pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
class TestNNV1_128_64:
    """Train and evaluate NN V1 with hidden_sizes=[128, 64], dropout=0.0."""

    def test_model_trains_without_error(self, trained_result):
        """Training should complete and return a result dict."""
        assert trained_result is not None
        assert "model" in trained_result
        assert "history" in trained_result
        assert "best_epoch" in trained_result
        assert "best_val_mae" in trained_result

    def test_mae_is_finite_and_positive(self, trained_result):
        """Best validation MAE should be finite and positive."""
        mae = trained_result["best_val_mae"]
        assert isinstance(mae, float)
        assert math.isfinite(mae)
        assert mae > 0.0

    def test_mae_less_than_50(self, real_data, trained_result):
        """Test-set MAE should be below 50 degF (sanity check)."""
        model = trained_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        mae = float(np.mean(np.abs(preds - actuals)))
        assert mae < 50.0, f"Test MAE of {mae:.2f} degF is unreasonably large"

    def test_predictions_are_finite(self, real_data, trained_result):
        """All test-set predictions should be finite (no NaN or Inf)."""
        model = trained_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf"

    def test_predictions_in_reasonable_range(self, real_data, trained_result):
        """Predictions should fall within -20 to 130 degF."""
        model = trained_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        assert preds.min() >= -20.0, (
            f"Min prediction ({preds.min():.1f}) below -20 degF"
        )
        assert preds.max() <= 130.0, (
            f"Max prediction ({preds.max():.1f}) above 130 degF"
        )

    def test_full_evaluation_and_report(self, real_data, trained_result):
        """Run full test-set evaluation, compute metrics, and print results.

        This is the primary result-reporting test.  It prints all metrics
        to stdout (visible with ``pytest -s``) and compares against the
        Ridge baseline and the default NN V1 [64, 32].
        """
        model = trained_result["model"]
        model.eval()

        # --- Generate test-set predictions ---
        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values

        # --- Compute metrics ---
        metrics = compute_metrics(actuals, preds, model_name="NN V1 [128,64]")
        n_params = count_parameters(model)
        best_epoch = trained_result["best_epoch"]
        best_val_mae = trained_result["best_val_mae"]
        total_epochs = len(trained_result["history"])

        # --- Reference baselines ---
        ridge_mae = 4.33
        nn_64_32_mae = 4.291

        # --- Print results ---
        print("\n")
        print("=" * 65)
        print("  NN V1 [128, 64] Experiment Results")
        print("=" * 65)
        print(f"  Architecture:       TempPredictorV1 hidden=[128, 64]")
        print(f"  Dropout:            {DROPOUT}")
        print(f"  Learning rate:      {LEARNING_RATE}")
        print(f"  Batch size:         {BATCH_SIZE}")
        print(f"  Parameter count:    {n_params:,}")
        print(f"  Total epochs run:   {total_epochs}")
        print(f"  Best epoch:         {best_epoch}")
        print(f"  Best val MAE:       {best_val_mae:.3f} degF")
        print("-" * 65)
        print(f"  TEST SET METRICS (n={metrics['n']})")
        print(f"    MAE:              {metrics['mae']:.3f} degF")
        print(f"    RMSE:             {metrics['rmse']:.3f} degF")
        print(f"    R-squared:        {metrics['r2']:.4f}")
        print(f"    Bias:             {metrics['bias']:+.3f} degF")
        print(f"    Within +/-1 degF: {metrics['within_1f']:.1f}%")
        print(f"    Within +/-2 degF: {metrics['within_2f']:.1f}%")
        print(f"    Within +/-3 degF: {metrics['within_3f']:.1f}%")
        print(f"    Max abs error:    {metrics['max_abs_error']:.2f} degF")
        print("-" * 65)
        print("  COMPARISON")
        print(f"    Ridge baseline MAE:     {ridge_mae:.3f} degF")
        print(f"    NN V1 [64,32] MAE:      {nn_64_32_mae:.3f} degF")
        print(f"    NN V1 [128,64] MAE:     {metrics['mae']:.3f} degF")

        diff_vs_ridge = metrics["mae"] - ridge_mae
        diff_vs_nn64 = metrics["mae"] - nn_64_32_mae

        if diff_vs_ridge < 0:
            print(f"    vs Ridge:  {abs(diff_vs_ridge):.3f} degF BETTER "
                  f"({abs(diff_vs_ridge)/ridge_mae*100:.1f}% improvement)")
        else:
            print(f"    vs Ridge:  {diff_vs_ridge:.3f} degF WORSE "
                  f"({diff_vs_ridge/ridge_mae*100:.1f}% degradation)")

        if diff_vs_nn64 < 0:
            print(f"    vs [64,32]: {abs(diff_vs_nn64):.3f} degF BETTER "
                  f"({abs(diff_vs_nn64)/nn_64_32_mae*100:.1f}% improvement)")
        else:
            print(f"    vs [64,32]: {diff_vs_nn64:.3f} degF WORSE "
                  f"({diff_vs_nn64/nn_64_32_mae*100:.1f}% degradation)")

        print("=" * 65)
        print()

        # --- Assertions ---
        assert math.isfinite(metrics["mae"])
        assert metrics["mae"] > 0.0
        assert metrics["mae"] < 50.0

    def test_cleanup(self, trained_result):
        """Clean up temp directory after all tests complete."""
        tmp_dir = trained_result.get("_tmp_dir")
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
