"""
Integration tests for the Neural Network V1 end-to-end pipeline.

Validates the complete Phase 3 pipeline from data loading through model
training, evaluation, and result persistence.  Tests cover:

  - Loading real processed data and verifying shapes/types
  - TempPredictorV1 creation with real feature count (30)
  - DataLoader creation from real processed data
  - Short training runs on real data
  - Model prediction quality (finite, reasonable range, beats mean predictor)
  - Evaluation framework integration (metrics, plots, report files)
  - Training history CSV round-trip
  - Model checkpoint save/load fidelity
  - Seasonal metrics integration

Tests that require processed data on disk are guarded with
``@pytest.mark.skipif`` so the suite remains green in CI environments
that lack data files.
"""

import os
import sys
import math
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import TempPredictorV1, create_model, count_parameters
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
    evaluate_predictions,
    compute_seasonal_metrics,
    format_metrics_table,
    plot_actual_vs_predicted,
    plot_residual_histogram,
    plot_time_series,
    plot_residuals_by_month,
)
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESSED_DATA_EXISTS = os.path.isdir(config.PROCESSED_DATA_DIR) and all(
    os.path.isfile(os.path.join(config.PROCESSED_DATA_DIR, f))
    for f in [
        "features_train.csv", "features_val.csv", "features_test.csv",
        "target_train.csv", "target_val.csv", "target_test.csv",
    ]
)

SKIP_REASON = "Processed data files not found in data/processed/"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def real_data():
    """Load the real processed data once per module (shared across tests).

    Returns a dict with X_train, X_val, X_test, y_train, y_val, y_test.
    """
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
def trained_model_result(real_data):
    """Train a TempPredictorV1 for a moderate run on real data.

    Uses a fixed seed and 30 epochs with a slightly elevated learning
    rate to ensure the model converges enough for meaningful quality
    checks while keeping the test suite fast.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_features = real_data["X_train"].shape[1]
    model = create_model(n_features)

    train_loader, val_loader = create_dataloaders(
        real_data["X_train"], real_data["y_train"],
        real_data["X_val"], real_data["y_val"],
        batch_size=config.BATCH_SIZE,
    )

    tmp_dir = tempfile.mkdtemp(prefix="nn_integration_")
    result = train_model(
        model, train_loader, val_loader,
        config_dict={
            "max_epochs": 30,
            "early_stopping_patience": 30,
            "learning_rate": 0.005,
        },
        models_dir=tmp_dir,
    )

    result["_tmp_dir"] = tmp_dir
    result["_train_loader"] = train_loader
    result["_val_loader"] = val_loader
    return result


@pytest.fixture
def tmp_dir():
    """Create and clean up a temporary directory."""
    d = tempfile.mkdtemp(prefix="nn_int_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# Data Loading Tests
# ===========================================================================

class TestDataLoading:
    """Tests for loading real processed data."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_load_returns_correct_types(self, real_data):
        """load_processed_data should return DataFrames and Series."""
        assert isinstance(real_data["X_train"], pd.DataFrame)
        assert isinstance(real_data["X_val"], pd.DataFrame)
        assert isinstance(real_data["X_test"], pd.DataFrame)
        assert isinstance(real_data["y_train"], pd.Series)
        assert isinstance(real_data["y_val"], pd.Series)
        assert isinstance(real_data["y_test"], pd.Series)

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_feature_count_is_30(self, real_data):
        """Real processed data should have 30 features."""
        assert real_data["X_train"].shape[1] == 30
        assert real_data["X_val"].shape[1] == 30
        assert real_data["X_test"].shape[1] == 30

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_train_val_test_row_counts(self, real_data):
        """Row counts should match the documented split sizes."""
        assert real_data["X_train"].shape[0] == 1277
        assert real_data["X_val"].shape[0] == 274
        assert real_data["X_test"].shape[0] == 274

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_target_aligned_with_features(self, real_data):
        """Target length should match feature row count for every split."""
        assert len(real_data["y_train"]) == real_data["X_train"].shape[0]
        assert len(real_data["y_val"]) == real_data["X_val"].shape[0]
        assert len(real_data["y_test"]) == real_data["X_test"].shape[0]

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_no_nan_in_features(self, real_data):
        """Processed features should contain no NaN values."""
        assert not real_data["X_train"].isnull().any().any()
        assert not real_data["X_val"].isnull().any().any()
        assert not real_data["X_test"].isnull().any().any()

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_no_nan_in_targets(self, real_data):
        """Processed targets should contain no NaN values."""
        assert not real_data["y_train"].isnull().any()
        assert not real_data["y_val"].isnull().any()
        assert not real_data["y_test"].isnull().any()

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_target_in_reasonable_temperature_range(self, real_data):
        """All target values should fall within 0-120 degF."""
        for split_name in ["y_train", "y_val", "y_test"]:
            y = real_data[split_name]
            assert y.min() >= 0.0, (
                f"{split_name} min ({y.min():.1f}) below 0 degF"
            )
            assert y.max() <= 120.0, (
                f"{split_name} max ({y.max():.1f}) above 120 degF"
            )


# ===========================================================================
# Model Creation Tests
# ===========================================================================

class TestModelCreation:
    """Tests for creating TempPredictorV1 with real feature counts."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_create_model_with_real_feature_count(self, real_data):
        """create_model should succeed with the actual feature count (30)."""
        n_features = real_data["X_train"].shape[1]
        model = create_model(n_features)
        assert isinstance(model, TempPredictorV1)
        assert model.n_features == 30

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_model_forward_pass_real_dimensions(self, real_data):
        """Model forward pass should work with a batch drawn from real data."""
        n_features = real_data["X_train"].shape[1]
        model = create_model(n_features)
        model.eval()

        sample = torch.tensor(
            real_data["X_train"].iloc[:16].values, dtype=torch.float32,
        )
        with torch.no_grad():
            out = model(sample)

        assert out.shape == (16, 1)
        assert torch.all(torch.isfinite(out))


# ===========================================================================
# DataLoader Integration Tests
# ===========================================================================

class TestDataLoaderIntegration:
    """Tests for creating DataLoaders from real processed data."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_dataloaders_created_successfully(self, real_data):
        """DataLoaders should be created without error from real data."""
        train_loader, val_loader = create_dataloaders(
            real_data["X_train"], real_data["y_train"],
            real_data["X_val"], real_data["y_val"],
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_dataloader_batch_shapes(self, real_data):
        """First batch from train loader should have correct shapes."""
        train_loader, _ = create_dataloaders(
            real_data["X_train"], real_data["y_train"],
            real_data["X_val"], real_data["y_val"],
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape[1] == 30
        assert y_batch.shape[1] == 1
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32


# ===========================================================================
# Training Integration Tests
# ===========================================================================

class TestTrainingIntegration:
    """Tests for short training runs on real data."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_short_training_completes(self, trained_model_result):
        """A 5-epoch training run on real data should complete."""
        assert trained_model_result is not None
        assert "model" in trained_model_result
        assert "history" in trained_model_result

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_training_history_has_expected_epochs(self, trained_model_result):
        """Training history should record exactly 30 epochs."""
        history = trained_model_result["history"]
        assert len(history) == 30

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_training_loss_decreases(self, trained_model_result):
        """Training loss should decrease over 5 epochs on real data."""
        history = trained_model_result["history"]
        first_loss = history[0]["train_loss"]
        last_loss = history[-1]["train_loss"]
        assert last_loss < first_loss, (
            f"Training loss did not decrease: first={first_loss:.4f}, "
            f"last={last_loss:.4f}"
        )

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_best_val_mae_is_finite_and_positive(self, trained_model_result):
        """Best validation MAE should be a finite positive number."""
        mae = trained_model_result["best_val_mae"]
        assert isinstance(mae, float)
        assert math.isfinite(mae)
        assert mae > 0.0


# ===========================================================================
# Prediction Quality Tests
# ===========================================================================

class TestPredictionQuality:
    """Tests for the quality and reasonableness of model predictions."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_predictions_are_finite(self, real_data, trained_model_result):
        """All test-set predictions should be finite (no NaN or Inf)."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        assert np.all(np.isfinite(preds)), "Test predictions contain NaN or Inf"

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_predictions_in_reasonable_range(self, real_data,
                                              trained_model_result):
        """Predictions should fall within a broadly plausible range (-20 to 130 degF).

        We use a slightly wider range than physical bounds because the
        model may slightly overshoot on edge cases, especially with
        limited training.  A well-trained model should be tighter, but
        this integration test verifies the model is not producing
        wildly divergent values.
        """
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        assert preds.min() >= -20.0, (
            f"Minimum prediction ({preds.min():.1f}) below -20 degF"
        )
        assert preds.max() <= 130.0, (
            f"Maximum prediction ({preds.max():.1f}) above 130 degF"
        )

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_mae_within_reasonable_bounds(self, real_data,
                                          trained_model_result):
        """MAE should be non-trivial: greater than 0 and less than 50 degF."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        mae = float(np.mean(np.abs(preds - actuals)))

        assert mae > 0.0, "MAE of exactly 0 is suspicious (overfitting or bug)"
        assert mae < 50.0, (
            f"MAE of {mae:.2f} degF is unreasonably large"
        )

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_model_beats_mean_predictor(self, real_data,
                                         trained_model_result):
        """NN V1 should outperform a trivial constant predictor (mean of training targets)."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        nn_mae = float(np.mean(np.abs(preds - actuals)))

        # Mean predictor: always predicts the training-set mean
        train_mean = float(real_data["y_train"].mean())
        mean_preds = np.full_like(actuals, train_mean)
        mean_mae = float(np.mean(np.abs(mean_preds - actuals)))

        assert nn_mae < mean_mae, (
            f"NN MAE ({nn_mae:.2f}) should be less than mean-predictor "
            f"MAE ({mean_mae:.2f})"
        )


# ===========================================================================
# Evaluation Integration Tests
# ===========================================================================

class TestEvaluationIntegration:
    """Tests for the evaluation framework on real model output."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_evaluate_predictions_returns_valid_metrics(
        self, real_data, trained_model_result,
    ):
        """evaluate_predictions should return a dict with all expected keys."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        metrics = compute_metrics(actuals, preds, model_name="NN V1")

        expected_keys = {"model_name", "n", "mae", "rmse", "r2", "bias",
                         "within_1f", "within_2f", "within_3f",
                         "max_abs_error"}
        assert expected_keys.issubset(metrics.keys()), (
            f"Missing keys: {expected_keys - metrics.keys()}"
        )
        assert metrics["n"] == len(actuals)
        assert math.isfinite(metrics["mae"])
        assert math.isfinite(metrics["rmse"])
        assert math.isfinite(metrics["r2"])

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_all_expected_plots_created(self, real_data,
                                         trained_model_result, tmp_dir):
        """evaluate_predictions should create scatter, residual, time-series, and monthly plots."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        dates = real_data["X_test"].index

        evaluate_predictions(
            y_actual=actuals,
            y_pred=preds,
            dates=dates,
            model_name="NN V1",
            output_dir=tmp_dir,
        )

        expected_files = [
            "nn_v1_scatter.png",
            "nn_v1_residual_hist.png",
            "nn_v1_timeseries.png",
            "nn_v1_residuals_month.png",
        ]
        for fname in expected_files:
            fpath = os.path.join(tmp_dir, fname)
            assert os.path.isfile(fpath), f"Expected plot not found: {fname}"
            assert os.path.getsize(fpath) > 0, f"Plot file is empty: {fname}"

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_seasonal_metrics_present(self, real_data, trained_model_result):
        """Seasonal metrics should be computable for test-set predictions."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        dates = real_data["X_test"].index

        seasonal = compute_seasonal_metrics(actuals, preds, dates)
        assert isinstance(seasonal, dict)
        assert len(seasonal) > 0, "Should have at least one season represented"

        for season_name, sm in seasonal.items():
            assert "mae" in sm
            assert "rmse" in sm
            assert "bias" in sm
            assert "n" in sm
            assert sm["n"] > 0


# ===========================================================================
# History and Checkpoint Persistence Tests
# ===========================================================================

class TestPersistence:
    """Tests for saving/loading training artifacts."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_training_history_csv_saved_correctly(self, trained_model_result,
                                                   tmp_dir):
        """Training history should round-trip through CSV correctly."""
        history = trained_model_result["history"]
        csv_path = os.path.join(tmp_dir, "training_history.csv")
        save_training_history(history, csv_path)

        assert os.path.isfile(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == len(history)
        assert set(df.columns) == set(history[0].keys())

        # Verify values round-trip accurately
        for i, entry in enumerate(history):
            assert df.iloc[i]["epoch"] == entry["epoch"]
            assert abs(df.iloc[i]["val_mae"] - entry["val_mae"]) < 1e-6

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_model_checkpoint_saved(self, trained_model_result):
        """A best_model.pt checkpoint should exist after training."""
        tmp_dir = trained_model_result["_tmp_dir"]
        checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
        assert os.path.isfile(checkpoint_path)
        assert os.path.getsize(checkpoint_path) > 0

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_checkpoint_load_produces_identical_predictions(
        self, real_data, trained_model_result,
    ):
        """Loading a checkpoint into a fresh model should give identical predictions."""
        tmp_dir = trained_model_result["_tmp_dir"]
        checkpoint_path = os.path.join(tmp_dir, "best_model.pt")

        n_features = real_data["X_train"].shape[1]

        # Original model predictions
        original_model = trained_model_result["model"]
        original_model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            original_preds = original_model(X_test_t).numpy().ravel()

        # Load into a fresh model
        fresh_model = TempPredictorV1(n_features=n_features)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        fresh_model.load_state_dict(state_dict)
        fresh_model.eval()

        with torch.no_grad():
            loaded_preds = fresh_model(X_test_t).numpy().ravel()

        np.testing.assert_allclose(
            original_preds, loaded_preds, atol=1e-6,
            err_msg="Loaded model predictions differ from original",
        )

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_training_curves_plot_created(self, trained_model_result, tmp_dir):
        """plot_training_curves should create a PNG file from real history."""
        history = trained_model_result["history"]
        plot_path = os.path.join(tmp_dir, "training_curves.png")
        plot_training_curves(history, plot_path)

        assert os.path.isfile(plot_path)
        assert os.path.getsize(plot_path) > 0


# ===========================================================================
# Metrics Table Integration Test
# ===========================================================================

class TestMetricsTableIntegration:
    """Tests for formatting a combined baseline + NN metrics table."""

    @pytest.mark.skipif(not PROCESSED_DATA_EXISTS, reason=SKIP_REASON)
    def test_format_combined_metrics_table(self, real_data,
                                            trained_model_result):
        """format_metrics_table should produce a valid table with NN + baselines."""
        model = trained_model_result["model"]
        model.eval()

        X_test_t = torch.tensor(
            real_data["X_test"].values, dtype=torch.float32,
        )
        with torch.no_grad():
            preds = model(X_test_t).numpy().ravel()

        actuals = real_data["y_test"].values
        nn_metrics = compute_metrics(actuals, preds, model_name="NN V1")

        # Combine with baseline references
        all_results = {
            "Ridge (alpha=1.0)": {"mae": 4.33, "rmse": 5.41, "r2": 0.876,
                                   "n": 274},
            "NN V1": nn_metrics,
        }

        table_str = format_metrics_table(all_results)
        assert isinstance(table_str, str)
        assert "Ridge" in table_str
        assert "NN V1" in table_str
        assert len(table_str) > 50  # Non-trivial content
