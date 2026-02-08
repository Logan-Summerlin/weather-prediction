"""
Tests for the enhanced training pipeline (src/train_v2.py).

Validates:
  - Loss function factory: MSE, Huber, MAE, invalid types
  - DataLoader creation: shapes, dtypes, batch sizes
  - train_one_epoch_v2: finite loss, parameter updates
  - validate_v2: correct shapes, finite values
  - compute_reconstructed_mae: hand-verified reconstruction
  - train_enhanced_model: full loop with raw and delta targets
  - evaluate_on_test: raw and delta evaluation
  - History saving and plot generation
  - Edge cases: single sample, large batches
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

from src.train_v2 import (
    get_loss_function,
    create_enhanced_dataloaders,
    train_one_epoch_v2,
    validate_v2,
    compute_reconstructed_mae,
    train_enhanced_model,
    evaluate_on_test,
    save_training_history_v2,
    plot_training_curves_v2,
)
import config


# ===========================================================================
# Simple model for testing
# ===========================================================================

class SimpleModel(nn.Module):
    """Minimal feedforward model for testing."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def n_features():
    return 10


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_data(n_features):
    """Create synthetic data for raw TMAX target."""
    rng = np.random.RandomState(42)
    n_train, n_val = 200, 50

    dates_train = pd.date_range("2020-01-01", periods=n_train, freq="D")
    dates_val = pd.date_range("2020-07-20", periods=n_val, freq="D")

    cols = [f"feature_{i}" for i in range(n_features)]

    X_train = pd.DataFrame(rng.randn(n_train, n_features),
                           index=dates_train, columns=cols)
    X_val = pd.DataFrame(rng.randn(n_val, n_features),
                         index=dates_val, columns=cols)

    doy_train = dates_train.dayofyear
    doy_val = dates_val.dayofyear
    y_train = pd.Series(
        60 + 20 * np.sin(2 * np.pi * doy_train / 365.25)
        + rng.normal(0, 3, n_train),
        index=dates_train, name="NYC_TMAX",
    )
    y_val = pd.Series(
        60 + 20 * np.sin(2 * np.pi * doy_val / 365.25)
        + rng.normal(0, 3, n_val),
        index=dates_val, name="NYC_TMAX",
    )

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
    }


@pytest.fixture
def synthetic_delta_data(n_features):
    """Create synthetic data for delta-T target."""
    rng = np.random.RandomState(42)
    n_train, n_val = 200, 50

    dates_train = pd.date_range("2020-01-01", periods=n_train, freq="D")
    dates_val = pd.date_range("2020-07-20", periods=n_val, freq="D")

    cols = [f"feature_{i}" for i in range(n_features)]

    X_train = pd.DataFrame(rng.randn(n_train, n_features),
                           index=dates_train, columns=cols)
    X_val = pd.DataFrame(rng.randn(n_val, n_features),
                         index=dates_val, columns=cols)

    # Actual TMAX values
    tmax_train = 60 + 20 * np.sin(2 * np.pi * dates_train.dayofyear / 365.25) \
                 + rng.normal(0, 3, n_train)
    tmax_val = 60 + 20 * np.sin(2 * np.pi * dates_val.dayofyear / 365.25) \
               + rng.normal(0, 3, n_val)

    # Previous-day TMAX
    nyc_prev_train = np.roll(tmax_train, 1)
    nyc_prev_train[0] = tmax_train[0] - 2  # Fake first value
    nyc_prev_val = np.roll(tmax_val, 1)
    nyc_prev_val[0] = tmax_val[0] - 1

    # Delta targets
    delta_train = tmax_train - nyc_prev_train
    delta_val = tmax_val - nyc_prev_val

    y_train_delta = pd.Series(delta_train, index=dates_train, name="NYC_DELTA_T")
    y_val_delta = pd.Series(delta_val, index=dates_val, name="NYC_DELTA_T")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train_delta": y_train_delta, "y_val_delta": y_val_delta,
        "nyc_prev_val": nyc_prev_val,
        "actual_tmax_val": tmax_val,
        "nyc_prev_train": nyc_prev_train,
        "actual_tmax_train": tmax_train,
    }


@pytest.fixture
def simple_model(n_features):
    return SimpleModel(n_features)


@pytest.fixture
def train_val_loaders(synthetic_data):
    d = synthetic_data
    return create_enhanced_dataloaders(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        batch_size=32,
    )


@pytest.fixture
def sample_history():
    return [
        {"epoch": 1, "train_loss": 100.0, "val_loss": 120.0,
         "val_mae": 8.5, "lr": 0.001},
        {"epoch": 2, "train_loss": 80.0, "val_loss": 95.0,
         "val_mae": 7.2, "lr": 0.001},
        {"epoch": 3, "train_loss": 65.0, "val_loss": 78.0,
         "val_mae": 6.1, "lr": 0.001},
    ]


@pytest.fixture
def sample_delta_history():
    return [
        {"epoch": 1, "train_loss": 10.0, "val_loss": 12.0,
         "val_mae": 5.5, "lr": 0.001, "delta_mae": 3.2,
         "reconstructed_mae": 5.5},
        {"epoch": 2, "train_loss": 8.0, "val_loss": 9.5,
         "val_mae": 4.8, "lr": 0.001, "delta_mae": 2.8,
         "reconstructed_mae": 4.8},
    ]


# ===========================================================================
# Loss Function Tests
# ===========================================================================

class TestGetLossFunction:
    """Tests for get_loss_function."""

    def test_mse(self):
        """'mse' should return MSELoss."""
        loss_fn = get_loss_function("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_huber(self):
        """'huber' should return SmoothL1Loss."""
        loss_fn = get_loss_function("huber")
        assert isinstance(loss_fn, nn.SmoothL1Loss)

    def test_mae(self):
        """'mae' should return L1Loss."""
        loss_fn = get_loss_function("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_case_insensitive(self):
        """Should be case-insensitive."""
        assert isinstance(get_loss_function("MSE"), nn.MSELoss)
        assert isinstance(get_loss_function("Huber"), nn.SmoothL1Loss)
        assert isinstance(get_loss_function("MAE"), nn.L1Loss)

    def test_invalid_type_raises(self):
        """Invalid loss type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            get_loss_function("invalid")

    def test_loss_functions_compute(self):
        """All loss functions should compute a scalar loss."""
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.5], [2.5], [2.0]])
        for loss_type in ["mse", "huber", "mae"]:
            loss_fn = get_loss_function(loss_type)
            loss = loss_fn(pred, target)
            assert loss.ndim == 0  # scalar
            assert loss.item() >= 0


# ===========================================================================
# DataLoader Tests
# ===========================================================================

class TestCreateEnhancedDataloaders:
    """Tests for create_enhanced_dataloaders."""

    def test_returns_two_dataloaders(self, synthetic_data):
        d = synthetic_data
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_custom_batch_size(self, synthetic_data):
        d = synthetic_data
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=16,
        )
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16

    def test_tensor_dtype(self, synthetic_data):
        d = synthetic_data
        train_loader, _ = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32

    def test_target_shape(self, synthetic_data):
        d = synthetic_data
        train_loader, _ = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        _, y_batch = next(iter(train_loader))
        assert y_batch.ndim == 2
        assert y_batch.shape[1] == 1


# ===========================================================================
# Training Tests
# ===========================================================================

class TestTrainOneEpochV2:
    """Tests for train_one_epoch_v2."""

    def test_returns_finite_loss(self, simple_model, train_val_loaders):
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch_v2(simple_model, train_loader, optimizer, criterion)
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_loss_with_huber(self, simple_model, train_val_loaders):
        """Should work with Huber loss."""
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = get_loss_function("huber")

        loss = train_one_epoch_v2(simple_model, train_loader, optimizer, criterion)
        assert math.isfinite(loss)
        assert loss >= 0

    def test_loss_with_mae(self, simple_model, train_val_loaders):
        """Should work with MAE loss."""
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = get_loss_function("mae")

        loss = train_one_epoch_v2(simple_model, train_loader, optimizer, criterion)
        assert math.isfinite(loss)

    def test_parameters_update(self, n_features, train_val_loaders):
        """Parameters should change after training."""
        model = SimpleModel(n_features)
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        train_one_epoch_v2(model, train_loader, optimizer, criterion)

        any_changed = False
        for name, p in model.named_parameters():
            if not torch.equal(params_before[name], p):
                any_changed = True
                break
        assert any_changed


# ===========================================================================
# Validation Tests
# ===========================================================================

class TestValidateV2:
    """Tests for validate_v2."""

    def test_returns_three_items(self, simple_model, train_val_loaders):
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()
        result = validate_v2(simple_model, val_loader, criterion)
        assert len(result) == 3

    def test_predictions_shape(self, simple_model, train_val_loaders,
                               synthetic_data):
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()
        _, preds, _ = validate_v2(simple_model, val_loader, criterion)
        assert len(preds) == len(synthetic_data["y_val"])

    def test_predictions_finite(self, simple_model, train_val_loaders):
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()
        _, preds, _ = validate_v2(simple_model, val_loader, criterion)
        assert np.all(np.isfinite(preds))


# ===========================================================================
# Reconstructed MAE Tests
# ===========================================================================

class TestComputeReconstructedMAE:
    """Tests for compute_reconstructed_mae."""

    def test_perfect_prediction(self):
        """Zero delta error should give zero reconstructed MAE."""
        pred_delta = np.array([2.0, -3.0, 1.0])
        nyc_prev = np.array([70.0, 72.0, 69.0])
        actual_tmax = np.array([72.0, 69.0, 70.0])  # prev + delta
        mae = compute_reconstructed_mae(pred_delta, nyc_prev, actual_tmax)
        assert abs(mae) < 1e-10

    def test_known_error(self):
        """Hand-computed reconstruction MAE."""
        pred_delta = np.array([3.0, -2.0])  # predictions
        nyc_prev = np.array([70.0, 73.0])   # previous day actuals
        actual_tmax = np.array([72.0, 69.0])  # actual current day

        # Reconstructed: [70+3=73, 73-2=71]
        # Errors: |73-72|=1, |71-69|=2
        # MAE = (1+2)/2 = 1.5
        mae = compute_reconstructed_mae(pred_delta, nyc_prev, actual_tmax)
        assert abs(mae - 1.5) < 1e-10

    def test_returns_float(self):
        """Should return a Python float."""
        pred_delta = np.array([1.0])
        nyc_prev = np.array([70.0])
        actual_tmax = np.array([71.0])
        mae = compute_reconstructed_mae(pred_delta, nyc_prev, actual_tmax)
        assert isinstance(mae, float)

    def test_nonnegative(self):
        """MAE should always be non-negative."""
        rng = np.random.RandomState(42)
        pred_delta = rng.randn(100)
        nyc_prev = 60 + rng.randn(100) * 5
        actual_tmax = nyc_prev + rng.randn(100) * 3
        mae = compute_reconstructed_mae(pred_delta, nyc_prev, actual_tmax)
        assert mae >= 0


# ===========================================================================
# Full Training Loop Tests
# ===========================================================================

class TestTrainEnhancedModel:
    """Tests for train_enhanced_model."""

    def test_raw_target_training(self, n_features, synthetic_data, tmp_dir):
        """Should train successfully with raw TMAX target."""
        d = synthetic_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="mse",
            target_type="raw",
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_raw",
        )

        assert "model" in result
        assert "history" in result
        assert result["best_val_mae"] >= 0
        assert result["loss_type"] == "mse"
        assert result["target_type"] == "raw"

    def test_delta_target_training(self, n_features, synthetic_delta_data,
                                    tmp_dir):
        """Should train successfully with delta-T target."""
        d = synthetic_delta_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train_delta"],
            d["X_val"], d["y_val_delta"],
            batch_size=32,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="huber",
            target_type="delta",
            nyc_prev_val=d["nyc_prev_val"],
            actual_tmax_val=d["actual_tmax_val"],
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_delta",
        )

        assert result["target_type"] == "delta"
        assert result["loss_type"] == "huber"
        assert result["best_val_mae"] >= 0

    def test_delta_history_has_extra_fields(self, n_features,
                                             synthetic_delta_data, tmp_dir):
        """Delta training history should include delta_mae and reconstructed_mae."""
        d = synthetic_delta_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train_delta"],
            d["X_val"], d["y_val_delta"],
            batch_size=32,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="huber",
            target_type="delta",
            nyc_prev_val=d["nyc_prev_val"],
            actual_tmax_val=d["actual_tmax_val"],
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_delta_hist",
        )

        for entry in result["history"]:
            assert "delta_mae" in entry
            assert "reconstructed_mae" in entry

    def test_huber_loss_training(self, n_features, synthetic_data, tmp_dir):
        """Should train with Huber loss without error."""
        d = synthetic_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="huber",
            target_type="raw",
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_huber",
        )
        assert result["loss_type"] == "huber"
        assert len(result["history"]) == 3

    def test_mae_loss_training(self, n_features, synthetic_data, tmp_dir):
        """Should train with MAE loss without error."""
        d = synthetic_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="mae",
            target_type="raw",
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_mae",
        )
        assert result["loss_type"] == "mae"

    def test_checkpoint_saved(self, n_features, synthetic_data, tmp_dir):
        """Should save a model checkpoint."""
        d = synthetic_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )

        train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="mse",
            target_type="raw",
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="test_ckpt",
        )

        checkpoint = os.path.join(tmp_dir, "best_test_ckpt.pt")
        assert os.path.isfile(checkpoint)

    def test_early_stopping(self, n_features, tmp_dir):
        """Early stopping should trigger with zero learning rate."""
        model = SimpleModel(n_features)
        rng = np.random.RandomState(99)
        X = pd.DataFrame(rng.randn(50, n_features),
                         columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series(rng.randn(50) * 10 + 60, name="NYC_TMAX")
        X_val = pd.DataFrame(rng.randn(20, n_features),
                             columns=[f"f{i}" for i in range(n_features)])
        y_val = pd.Series(rng.randn(20) * 10 + 60, name="NYC_TMAX")

        train_loader, val_loader = create_enhanced_dataloaders(
            X, y, X_val, y_val, batch_size=16,
        )

        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="mse",
            target_type="raw",
            config_dict={
                "max_epochs": 100,
                "early_stopping_patience": 3,
                "learning_rate": 0.0,
            },
            models_dir=tmp_dir,
            model_name="test_es",
        )

        assert len(result["history"]) <= 5

    def test_delta_requires_nyc_prev(self, n_features, synthetic_data, tmp_dir):
        """Delta training without nyc_prev should raise ValueError."""
        d = synthetic_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )

        with pytest.raises(ValueError, match="nyc_prev_val"):
            train_enhanced_model(
                model, train_loader, val_loader,
                loss_type="huber",
                target_type="delta",
                config_dict={"max_epochs": 1},
                models_dir=tmp_dir,
                model_name="test_fail",
            )


# ===========================================================================
# Test-Set Evaluation Tests
# ===========================================================================

class TestEvaluateOnTest:
    """Tests for evaluate_on_test."""

    def test_raw_evaluation(self, n_features, synthetic_data, tmp_dir):
        """Should evaluate correctly with raw TMAX target."""
        d = synthetic_data
        model = SimpleModel(n_features)
        # Quick train
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="mse", target_type="raw",
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir, model_name="eval_raw",
        )

        test_result = evaluate_on_test(
            result["model"], d["X_val"], d["y_val"],
            target_type="raw",
        )

        assert "mae" in test_result
        assert "rmse" in test_result
        assert "r2" in test_result
        assert test_result["mae"] >= 0
        assert len(test_result["predictions"]) == len(d["y_val"])

    def test_delta_evaluation(self, n_features, synthetic_delta_data, tmp_dir):
        """Should evaluate correctly with delta-T target."""
        d = synthetic_delta_data
        model = SimpleModel(n_features)
        train_loader, val_loader = create_enhanced_dataloaders(
            d["X_train"], d["y_train_delta"],
            d["X_val"], d["y_val_delta"],
            batch_size=32,
        )
        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="huber", target_type="delta",
            nyc_prev_val=d["nyc_prev_val"],
            actual_tmax_val=d["actual_tmax_val"],
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir, model_name="eval_delta",
        )

        test_result = evaluate_on_test(
            result["model"],
            d["X_val"], d["y_val_delta"],
            target_type="delta",
            nyc_prev_test=d["nyc_prev_val"],
            actual_tmax_test=d["actual_tmax_val"],
        )

        assert "mae" in test_result
        assert "delta_mae" in test_result
        assert "reconstructed_predictions" in test_result
        assert test_result["mae"] >= 0

    def test_delta_requires_prev(self, n_features, synthetic_data):
        """Delta evaluation without prev should raise ValueError."""
        model = SimpleModel(n_features)
        d = synthetic_data
        with pytest.raises(ValueError, match="nyc_prev_test"):
            evaluate_on_test(
                model, d["X_val"], d["y_val"],
                target_type="delta",
            )


# ===========================================================================
# History & Plot Tests
# ===========================================================================

class TestSaveTrainingHistoryV2:
    """Tests for save_training_history_v2."""

    def test_creates_csv(self, sample_history, tmp_dir):
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history_v2(sample_history, path)
        assert os.path.isfile(path)

    def test_csv_roundtrip(self, sample_history, tmp_dir):
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history_v2(sample_history, path)
        df = pd.read_csv(path)
        assert len(df) == len(sample_history)
        assert "epoch" in df.columns

    def test_empty_history(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.csv")
        save_training_history_v2([], path)
        # Should not crash

    def test_delta_history_roundtrip(self, sample_delta_history, tmp_dir):
        """Delta history with extra fields should save correctly."""
        path = os.path.join(tmp_dir, "delta_hist.csv")
        save_training_history_v2(sample_delta_history, path)
        df = pd.read_csv(path)
        assert "delta_mae" in df.columns
        assert "reconstructed_mae" in df.columns


class TestPlotTrainingCurvesV2:
    """Tests for plot_training_curves_v2."""

    def test_creates_plot(self, sample_history, tmp_dir):
        path = os.path.join(tmp_dir, "curves.png")
        plot_training_curves_v2(sample_history, path)
        assert os.path.isfile(path)

    def test_delta_plot(self, sample_delta_history, tmp_dir):
        """Should handle delta history with extra panels."""
        path = os.path.join(tmp_dir, "delta_curves.png")
        plot_training_curves_v2(sample_delta_history, path, title="Delta Test")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_empty_history_no_crash(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.png")
        plot_training_curves_v2([], path)


# ===========================================================================
# Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge case and integration tests."""

    def test_single_sample(self):
        """Should handle a single training sample."""
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        y = pd.Series([60.0], name="NYC_TMAX")
        X_v = pd.DataFrame({"a": [3.0], "b": [4.0]})
        y_v = pd.Series([65.0], name="NYC_TMAX")

        train_loader, val_loader = create_enhanced_dataloaders(
            X, y, X_v, y_v, batch_size=1,
        )
        model = SimpleModel(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch_v2(model, train_loader, optimizer, criterion)
        assert math.isfinite(loss)

    def test_large_batch_size(self):
        """Batch larger than data should work."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(5, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randn(5) * 10 + 60, name="NYC_TMAX")
        X_v = pd.DataFrame(rng.randn(3, 3), columns=["a", "b", "c"])
        y_v = pd.Series(rng.randn(3) * 10 + 60, name="NYC_TMAX")

        train_loader, _ = create_enhanced_dataloaders(
            X, y, X_v, y_v, batch_size=1000,
        )
        assert len(train_loader) == 1

    def test_full_pipeline_small(self, tmp_dir):
        """End-to-end small training pipeline with delta target."""
        n_features = 5
        rng = np.random.RandomState(42)
        n_train, n_val = 80, 20

        X_train = pd.DataFrame(rng.randn(n_train, n_features),
                               columns=[f"f{i}" for i in range(n_features)])
        X_val = pd.DataFrame(rng.randn(n_val, n_features),
                             columns=[f"f{i}" for i in range(n_features)])

        tmax_train = rng.randn(n_train) * 10 + 60
        tmax_val = rng.randn(n_val) * 10 + 60
        prev_train = np.roll(tmax_train, 1)
        prev_train[0] = tmax_train[0]
        prev_val = np.roll(tmax_val, 1)
        prev_val[0] = tmax_val[0]

        y_train = pd.Series(tmax_train - prev_train, name="NYC_DELTA_T")
        y_val = pd.Series(tmax_val - prev_val, name="NYC_DELTA_T")

        train_loader, val_loader = create_enhanced_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=16,
        )

        model = SimpleModel(n_features)
        result = train_enhanced_model(
            model, train_loader, val_loader,
            loss_type="huber",
            target_type="delta",
            nyc_prev_val=prev_val,
            actual_tmax_val=tmax_val,
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
            model_name="small_delta",
        )

        assert result["best_val_mae"] >= 0
        assert len(result["history"]) == 5
