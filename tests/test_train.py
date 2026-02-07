"""
Tests for the training pipeline (src/train.py).

Validates:
  - DataLoader creation: types, batch sizes, tensor dtypes, shapes
  - train_one_epoch: returns finite loss, model parameters update
  - validate: returns loss, predictions, actuals with correct shapes
  - train_model: full loop, early stopping, checkpoint saving, history
  - load_processed_data: correct loading (skipped if data absent)
  - save_training_history: CSV round-trip
  - plot_training_curves: file creation without error
  - Edge cases: single-sample data, tiny batch sizes
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

from src.train import (
    create_dataloaders,
    train_one_epoch,
    validate,
    train_model,
    load_processed_data,
    save_training_history,
    plot_training_curves,
)
import config


# ===========================================================================
# Simple inline model (avoids dependency on src.model)
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
    """Number of input features for synthetic data."""
    return 10


@pytest.fixture
def synthetic_pandas_data(n_features):
    """Create synthetic pandas DataFrames/Series mimicking preprocessed data.

    Returns dict with X_train, X_val, y_train, y_val as pandas objects.
    """
    rng = np.random.RandomState(42)

    n_train = 200
    n_val = 50

    dates_train = pd.date_range("2020-01-01", periods=n_train, freq="D")
    dates_val = pd.date_range("2020-07-20", periods=n_val, freq="D")

    cols = [f"feature_{i}" for i in range(n_features)]

    X_train = pd.DataFrame(rng.randn(n_train, n_features),
                           index=dates_train, columns=cols)
    X_val = pd.DataFrame(rng.randn(n_val, n_features),
                         index=dates_val, columns=cols)

    # Target: sinusoidal pattern + noise (realistic temp-like values)
    doy_train = dates_train.dayofyear
    doy_val = dates_val.dayofyear
    y_train = pd.Series(
        60 + 20 * np.sin(2 * np.pi * doy_train / 365.25) + rng.normal(0, 3, n_train),
        index=dates_train, name="NYC_TMAX",
    )
    y_val = pd.Series(
        60 + 20 * np.sin(2 * np.pi * doy_val / 365.25) + rng.normal(0, 3, n_val),
        index=dates_val, name="NYC_TMAX",
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


@pytest.fixture
def simple_model(n_features):
    """Create a simple feedforward model."""
    return SimpleModel(n_features)


@pytest.fixture
def train_val_loaders(synthetic_pandas_data):
    """Create DataLoaders from synthetic data."""
    d = synthetic_pandas_data
    return create_dataloaders(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        batch_size=32,
    )


@pytest.fixture
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_history():
    """Sample training history for save/plot tests."""
    return [
        {"epoch": 1, "train_loss": 100.0, "val_loss": 120.0,
         "val_mae": 8.5, "lr": 0.001},
        {"epoch": 2, "train_loss": 80.0, "val_loss": 95.0,
         "val_mae": 7.2, "lr": 0.001},
        {"epoch": 3, "train_loss": 65.0, "val_loss": 78.0,
         "val_mae": 6.1, "lr": 0.001},
        {"epoch": 4, "train_loss": 55.0, "val_loss": 70.0,
         "val_mae": 5.5, "lr": 0.0005},
        {"epoch": 5, "train_loss": 50.0, "val_loss": 68.0,
         "val_mae": 5.3, "lr": 0.0005},
    ]


# ===========================================================================
# DataLoader Tests
# ===========================================================================

class TestCreateDataloaders:
    """Tests for create_dataloaders."""

    def test_returns_tuple_of_dataloaders(self, synthetic_pandas_data):
        """create_dataloaders should return a tuple of two DataLoaders."""
        d = synthetic_pandas_data
        train_loader, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_default_batch_size(self, synthetic_pandas_data):
        """Default batch_size should come from config.BATCH_SIZE."""
        d = synthetic_pandas_data
        train_loader, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        )
        assert train_loader.batch_size == config.BATCH_SIZE

    def test_custom_batch_size(self, synthetic_pandas_data):
        """Custom batch_size should override the default."""
        d = synthetic_pandas_data
        custom_bs = 16
        train_loader, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=custom_bs,
        )
        assert train_loader.batch_size == custom_bs
        assert val_loader.batch_size == custom_bs

    def test_tensor_dtypes_are_float32(self, synthetic_pandas_data):
        """All tensors in the DataLoaders should be float32."""
        d = synthetic_pandas_data
        train_loader, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32

    def test_feature_shape(self, synthetic_pandas_data, n_features):
        """Feature tensor shape should be (batch, n_features)."""
        d = synthetic_pandas_data
        train_loader, _ = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        X_batch, _ = next(iter(train_loader))
        assert X_batch.shape[1] == n_features

    def test_target_shape(self, synthetic_pandas_data):
        """Target tensor should have shape (batch, 1)."""
        d = synthetic_pandas_data
        train_loader, _ = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        _, y_batch = next(iter(train_loader))
        assert y_batch.ndim == 2
        assert y_batch.shape[1] == 1

    def test_train_loader_number_of_batches(self, synthetic_pandas_data):
        """Number of training batches should match ceil(n_train / batch_size)."""
        d = synthetic_pandas_data
        bs = 32
        train_loader, _ = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=bs,
        )
        expected = math.ceil(len(d["X_train"]) / bs)
        assert len(train_loader) == expected

    def test_val_loader_number_of_batches(self, synthetic_pandas_data):
        """Number of validation batches should match ceil(n_val / batch_size)."""
        d = synthetic_pandas_data
        bs = 32
        _, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=bs,
        )
        expected = math.ceil(len(d["X_val"]) / bs)
        assert len(val_loader) == expected

    def test_total_samples_train(self, synthetic_pandas_data):
        """Total samples across all train batches should equal n_train."""
        d = synthetic_pandas_data
        train_loader, _ = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        total = sum(X.shape[0] for X, _ in train_loader)
        assert total == len(d["X_train"])

    def test_total_samples_val(self, synthetic_pandas_data):
        """Total samples across all val batches should equal n_val."""
        d = synthetic_pandas_data
        _, val_loader = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=32,
        )
        total = sum(X.shape[0] for X, _ in val_loader)
        assert total == len(d["X_val"])

    def test_batch_size_one(self, synthetic_pandas_data):
        """batch_size=1 should produce one sample per batch."""
        d = synthetic_pandas_data
        train_loader, _ = create_dataloaders(
            d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            batch_size=1,
        )
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape[0] == 1
        assert y_batch.shape[0] == 1


# ===========================================================================
# train_one_epoch Tests
# ===========================================================================

class TestTrainOneEpoch:
    """Tests for train_one_epoch."""

    def test_returns_finite_float(self, simple_model, train_val_loaders):
        """train_one_epoch should return a finite float loss."""
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch(simple_model, train_loader, optimizer, criterion)
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_loss_is_positive(self, simple_model, train_val_loaders):
        """MSE loss should be non-negative."""
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch(simple_model, train_loader, optimizer, criterion)
        assert loss >= 0.0

    def test_parameters_change_after_training(self, n_features, train_val_loaders):
        """Model parameters should change after one training epoch."""
        model = SimpleModel(n_features)
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Snapshot initial parameters
        params_before = {name: p.clone() for name, p in model.named_parameters()}

        train_one_epoch(model, train_loader, optimizer, criterion)

        # At least some parameters should have changed
        any_changed = False
        for name, p in model.named_parameters():
            if not torch.equal(params_before[name], p):
                any_changed = True
                break
        assert any_changed, "No parameters changed after training epoch"

    def test_loss_decreases_over_epochs(self, n_features, train_val_loaders):
        """Training loss should generally decrease over multiple epochs."""
        model = SimpleModel(n_features)
        train_loader, _ = train_val_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for _ in range(10):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            losses.append(loss)

        # Loss at epoch 10 should be less than loss at epoch 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, "
            f"last={losses[-1]:.4f}"
        )


# ===========================================================================
# validate Tests
# ===========================================================================

class TestValidate:
    """Tests for validate."""

    def test_returns_three_items(self, simple_model, train_val_loaders):
        """validate should return (loss, predictions, actuals)."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        result = validate(simple_model, val_loader, criterion)
        assert len(result) == 3

    def test_loss_is_finite_float(self, simple_model, train_val_loaders):
        """Validation loss should be a finite float."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        val_loss, _, _ = validate(simple_model, val_loader, criterion)
        assert isinstance(val_loss, float)
        assert math.isfinite(val_loss)

    def test_predictions_shape(self, simple_model, train_val_loaders,
                               synthetic_pandas_data):
        """Predictions should have same length as validation set."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        _, preds, _ = validate(simple_model, val_loader, criterion)
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1
        assert len(preds) == len(synthetic_pandas_data["y_val"])

    def test_actuals_shape(self, simple_model, train_val_loaders,
                           synthetic_pandas_data):
        """Actuals should have same length as validation set."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        _, _, actuals = validate(simple_model, val_loader, criterion)
        assert isinstance(actuals, np.ndarray)
        assert actuals.ndim == 1
        assert len(actuals) == len(synthetic_pandas_data["y_val"])

    def test_actuals_match_input(self, simple_model, train_val_loaders,
                                 synthetic_pandas_data):
        """Returned actuals should match the original validation targets."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        _, _, actuals = validate(simple_model, val_loader, criterion)
        expected = synthetic_pandas_data["y_val"].values
        np.testing.assert_allclose(actuals, expected, atol=1e-5)

    def test_predictions_are_finite(self, simple_model, train_val_loaders):
        """All predictions should be finite values."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        _, preds, _ = validate(simple_model, val_loader, criterion)
        assert np.all(np.isfinite(preds))

    def test_no_gradient_during_validation(self, simple_model, train_val_loaders):
        """Validate should not compute gradients (model.eval mode)."""
        _, val_loader = train_val_loaders
        criterion = nn.MSELoss()

        # Validate and check that no grad is accumulated
        validate(simple_model, val_loader, criterion)

        for param in simple_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


# ===========================================================================
# train_model Tests (Full Training Loop)
# ===========================================================================

class TestTrainModel:
    """Tests for train_model."""

    def test_runs_without_error(self, simple_model, train_val_loaders, tmp_dir):
        """train_model should complete without raising an exception."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 5, "early_stopping_patience": 3},
            models_dir=tmp_dir,
        )
        assert result is not None

    def test_returns_required_keys(self, simple_model, train_val_loaders,
                                   tmp_dir):
        """Return dict should contain model, history, best_epoch, best_val_mae."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        assert "model" in result
        assert "history" in result
        assert "best_epoch" in result
        assert "best_val_mae" in result

    def test_model_is_nn_module(self, simple_model, train_val_loaders, tmp_dir):
        """Returned model should be an nn.Module."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        assert isinstance(result["model"], nn.Module)

    def test_history_structure(self, simple_model, train_val_loaders, tmp_dir):
        """History should be a list of dicts with correct keys."""
        train_loader, val_loader = train_val_loaders
        n_epochs = 5
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": n_epochs,
                         "early_stopping_patience": n_epochs + 5},
            models_dir=tmp_dir,
        )
        history = result["history"]
        assert isinstance(history, list)
        assert len(history) == n_epochs

        required_keys = {"epoch", "train_loss", "val_loss", "val_mae", "lr"}
        for entry in history:
            assert isinstance(entry, dict)
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys: {required_keys - entry.keys()}"
            )

    def test_history_epochs_are_sequential(self, simple_model,
                                            train_val_loaders, tmp_dir):
        """Epoch numbers in history should be sequential starting from 1."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        epochs = [h["epoch"] for h in result["history"]]
        assert epochs == list(range(1, len(epochs) + 1))

    def test_best_epoch_is_positive(self, simple_model, train_val_loaders,
                                    tmp_dir):
        """best_epoch should be a positive integer."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        assert isinstance(result["best_epoch"], int)
        assert result["best_epoch"] >= 1

    def test_best_val_mae_is_finite(self, simple_model, train_val_loaders,
                                    tmp_dir):
        """best_val_mae should be a finite positive float."""
        train_loader, val_loader = train_val_loaders
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 5, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        assert isinstance(result["best_val_mae"], float)
        assert math.isfinite(result["best_val_mae"])
        assert result["best_val_mae"] >= 0.0

    def test_checkpoint_saved(self, simple_model, train_val_loaders, tmp_dir):
        """Best model checkpoint should be saved to disk."""
        train_loader, val_loader = train_val_loaders
        train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
        assert os.path.isfile(checkpoint_path)
        assert os.path.getsize(checkpoint_path) > 0

    def test_checkpoint_loadable(self, n_features, train_val_loaders, tmp_dir):
        """Saved checkpoint should be loadable into a new model."""
        model = SimpleModel(n_features)
        train_loader, val_loader = train_val_loaders
        train_model(
            model, train_loader, val_loader,
            config_dict={"max_epochs": 3, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )
        checkpoint_path = os.path.join(tmp_dir, "best_model.pt")

        new_model = SimpleModel(n_features)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        new_model.load_state_dict(state_dict)

    def test_early_stopping_triggers(self, n_features, tmp_dir):
        """Early stopping should trigger when validation loss stops improving.

        Use a model with zero learning rate so val MAE never improves.
        """
        model = SimpleModel(n_features)
        patience = 3

        # Create small synthetic data
        rng = np.random.RandomState(99)
        X = pd.DataFrame(rng.randn(50, n_features),
                         columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series(rng.randn(50) * 10 + 60, name="NYC_TMAX")
        X_val = pd.DataFrame(rng.randn(20, n_features),
                             columns=[f"f{i}" for i in range(n_features)])
        y_val = pd.Series(rng.randn(20) * 10 + 60, name="NYC_TMAX")

        train_loader, val_loader = create_dataloaders(X, y, X_val, y_val,
                                                      batch_size=16)

        result = train_model(
            model, train_loader, val_loader,
            config_dict={
                "max_epochs": 100,
                "early_stopping_patience": patience,
                "learning_rate": 0.0,  # no learning -> no improvement
            },
            models_dir=tmp_dir,
        )

        # Training should have stopped well before 100 epochs
        n_trained = len(result["history"])
        # With patience=3 and no improvement: epoch 1 is best, then
        # epochs 2, 3, 4 have no improvement -> stop at epoch 4
        assert n_trained <= patience + 2, (
            f"Expected early stopping by epoch {patience + 2}, "
            f"but trained for {n_trained} epochs"
        )

    def test_custom_learning_rate(self, simple_model, train_val_loaders,
                                  tmp_dir):
        """Custom learning_rate in config_dict should be used."""
        train_loader, val_loader = train_val_loaders
        custom_lr = 0.05
        result = train_model(
            simple_model, train_loader, val_loader,
            config_dict={"max_epochs": 2, "early_stopping_patience": 10,
                         "learning_rate": custom_lr},
            models_dir=tmp_dir,
        )
        # The first epoch should use the custom LR
        assert result["history"][0]["lr"] == pytest.approx(custom_lr)

    def test_val_mae_in_history_matches_validation(self, n_features,
                                                    train_val_loaders,
                                                    tmp_dir):
        """Val MAE in history should be consistent with validate() output."""
        model = SimpleModel(n_features)
        train_loader, val_loader = train_val_loaders

        result = train_model(
            model, train_loader, val_loader,
            config_dict={"max_epochs": 1, "early_stopping_patience": 10},
            models_dir=tmp_dir,
        )

        # Re-validate the returned model
        criterion = nn.MSELoss()
        _, preds, actuals = validate(result["model"], val_loader, criterion)
        expected_mae = float(np.mean(np.abs(preds - actuals)))

        # Should approximately match (model was loaded from best checkpoint)
        recorded_mae = result["best_val_mae"]
        assert abs(recorded_mae - expected_mae) < 0.5, (
            f"Recorded MAE ({recorded_mae:.3f}) diverges from "
            f"recomputed MAE ({expected_mae:.3f})"
        )


# ===========================================================================
# save_training_history Tests
# ===========================================================================

class TestSaveTrainingHistory:
    """Tests for save_training_history."""

    def test_creates_csv_file(self, sample_history, tmp_dir):
        """save_training_history should create a CSV file."""
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history(sample_history, path)
        assert os.path.isfile(path)

    def test_csv_has_correct_rows(self, sample_history, tmp_dir):
        """CSV should have one row per epoch plus header."""
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history(sample_history, path)

        df = pd.read_csv(path)
        assert len(df) == len(sample_history)

    def test_csv_has_correct_columns(self, sample_history, tmp_dir):
        """CSV columns should match history dict keys."""
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history(sample_history, path)

        df = pd.read_csv(path)
        expected_cols = set(sample_history[0].keys())
        assert set(df.columns) == expected_cols

    def test_csv_values_roundtrip(self, sample_history, tmp_dir):
        """Values should survive CSV write/read round-trip."""
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history(sample_history, path)

        df = pd.read_csv(path)
        for i, entry in enumerate(sample_history):
            assert df.iloc[i]["epoch"] == entry["epoch"]
            assert abs(df.iloc[i]["train_loss"] - entry["train_loss"]) < 1e-6
            assert abs(df.iloc[i]["val_mae"] - entry["val_mae"]) < 1e-6

    def test_empty_history_no_error(self, tmp_dir):
        """Saving empty history should not raise an error."""
        path = os.path.join(tmp_dir, "empty_history.csv")
        save_training_history([], path)
        # File may or may not be created, but no exception

    def test_creates_parent_directories(self, sample_history, tmp_dir):
        """Should create parent directories if they don't exist."""
        path = os.path.join(tmp_dir, "sub", "deep", "history.csv")
        save_training_history(sample_history, path)
        assert os.path.isfile(path)


# ===========================================================================
# plot_training_curves Tests
# ===========================================================================

class TestPlotTrainingCurves:
    """Tests for plot_training_curves."""

    def test_creates_plot_file(self, sample_history, tmp_dir):
        """plot_training_curves should create a PNG file."""
        path = os.path.join(tmp_dir, "curves.png")
        plot_training_curves(sample_history, path)
        assert os.path.isfile(path)

    def test_file_is_nonempty(self, sample_history, tmp_dir):
        """Generated plot file should not be empty."""
        path = os.path.join(tmp_dir, "curves.png")
        plot_training_curves(sample_history, path)
        assert os.path.getsize(path) > 0

    def test_creates_parent_directories(self, sample_history, tmp_dir):
        """Should create parent directories if they don't exist."""
        path = os.path.join(tmp_dir, "plots", "sub", "curves.png")
        plot_training_curves(sample_history, path)
        assert os.path.isfile(path)

    def test_empty_history_no_error(self, tmp_dir):
        """Empty history should not raise an error."""
        path = os.path.join(tmp_dir, "empty.png")
        plot_training_curves([], path)
        # Should not crash; file may or may not exist

    def test_single_epoch_history(self, tmp_dir):
        """Should handle a single-epoch history without error."""
        history = [{"epoch": 1, "train_loss": 50.0, "val_loss": 55.0,
                     "val_mae": 6.0, "lr": 0.001}]
        path = os.path.join(tmp_dir, "single.png")
        plot_training_curves(history, path)
        assert os.path.isfile(path)


# ===========================================================================
# load_processed_data Tests
# ===========================================================================

class TestLoadProcessedData:
    """Tests for load_processed_data."""

    def test_missing_directory_raises(self, tmp_dir):
        """Should raise FileNotFoundError if directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_processed_data(os.path.join(tmp_dir, "nonexistent"))

    def test_missing_files_raises(self, tmp_dir):
        """Should raise FileNotFoundError if required files are absent."""
        # Create directory but no files
        empty_dir = os.path.join(tmp_dir, "empty_processed")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError):
            load_processed_data(empty_dir)

    def test_loads_synthetic_processed_data(self, tmp_dir):
        """Should correctly load synthetic processed data from CSV files."""
        rng = np.random.RandomState(42)
        n_train, n_val, n_test = 100, 30, 30
        n_feat = 5

        dates_train = pd.date_range("2020-01-01", periods=n_train, freq="D")
        dates_val = pd.date_range("2020-04-11", periods=n_val, freq="D")
        dates_test = pd.date_range("2020-05-11", periods=n_test, freq="D")

        cols = [f"feat_{i}" for i in range(n_feat)]

        # Save synthetic data
        for dates, n, suffix in [(dates_train, n_train, "train"),
                                 (dates_val, n_val, "val"),
                                 (dates_test, n_test, "test")]:
            features = pd.DataFrame(rng.randn(n, n_feat),
                                    index=dates, columns=cols)
            features.index.name = "date"
            features.to_csv(os.path.join(tmp_dir, f"features_{suffix}.csv"))

            target = pd.Series(rng.randn(n) * 10 + 60,
                               index=dates, name="NYC_TMAX")
            target.index.name = "date"
            target.to_csv(os.path.join(tmp_dir, f"target_{suffix}.csv"),
                          header=True)

        X_train, X_val, X_test, y_train, y_val, y_test = \
            load_processed_data(tmp_dir)

        assert X_train.shape == (n_train, n_feat)
        assert X_val.shape == (n_val, n_feat)
        assert X_test.shape == (n_test, n_feat)
        assert len(y_train) == n_train
        assert len(y_val) == n_val
        assert len(y_test) == n_test

    @pytest.mark.skipif(
        not os.path.isdir(
            os.path.join(os.path.dirname(__file__), "..", "data", "processed")
        ),
        reason="Processed data not available",
    )
    def test_loads_real_processed_data(self):
        """Load actual processed data if it exists and verify shapes."""
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_val) == X_val.shape[0]
        assert len(y_test) == X_test.shape[0]


# ===========================================================================
# Integration / Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_single_sample_dataloaders(self):
        """DataLoaders should work with a single training sample."""
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        y = pd.Series([60.0], name="NYC_TMAX")
        X_val = pd.DataFrame({"a": [3.0], "b": [4.0]})
        y_val = pd.Series([65.0], name="NYC_TMAX")

        train_loader, val_loader = create_dataloaders(
            X, y, X_val, y_val, batch_size=1,
        )
        assert len(train_loader) == 1
        assert len(val_loader) == 1

    def test_single_sample_train_one_epoch(self):
        """train_one_epoch should work with a single sample."""
        model = SimpleModel(2)
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        y = torch.tensor([[60.0]], dtype=torch.float32)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = train_one_epoch(model, loader, optimizer, criterion)
        assert math.isfinite(loss)

    def test_single_sample_validate(self):
        """validate should work with a single sample."""
        model = SimpleModel(2)
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        y = torch.tensor([[60.0]], dtype=torch.float32)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=1)
        criterion = nn.MSELoss()

        val_loss, preds, actuals = validate(model, loader, criterion)
        assert math.isfinite(val_loss)
        assert len(preds) == 1
        assert len(actuals) == 1

    def test_full_pipeline_small(self, tmp_dir):
        """End-to-end small training pipeline."""
        n_features = 5
        rng = np.random.RandomState(42)

        X_train = pd.DataFrame(rng.randn(80, n_features),
                               columns=[f"f{i}" for i in range(n_features)])
        y_train = pd.Series(rng.randn(80) * 10 + 60, name="NYC_TMAX")
        X_val = pd.DataFrame(rng.randn(20, n_features),
                             columns=[f"f{i}" for i in range(n_features)])
        y_val = pd.Series(rng.randn(20) * 10 + 60, name="NYC_TMAX")

        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=16,
        )

        model = SimpleModel(n_features)
        result = train_model(
            model, train_loader, val_loader,
            config_dict={
                "max_epochs": 10,
                "early_stopping_patience": 20,
                "learning_rate": 0.01,
            },
            models_dir=tmp_dir,
        )

        assert result["best_val_mae"] >= 0
        assert len(result["history"]) == 10
        assert os.path.isfile(os.path.join(tmp_dir, "best_model.pt"))

        # Save and plot history
        hist_path = os.path.join(tmp_dir, "history.csv")
        save_training_history(result["history"], hist_path)
        assert os.path.isfile(hist_path)

        plot_path = os.path.join(tmp_dir, "curves.png")
        plot_training_curves(result["history"], plot_path)
        assert os.path.isfile(plot_path)

    def test_large_batch_size(self):
        """Batch size larger than dataset should produce a single batch."""
        rng = np.random.RandomState(42)
        n = 10
        n_feat = 3
        X = pd.DataFrame(rng.randn(n, n_feat), columns=["a", "b", "c"])
        y = pd.Series(rng.randn(n) * 10 + 60, name="NYC_TMAX")
        X_v = pd.DataFrame(rng.randn(5, n_feat), columns=["a", "b", "c"])
        y_v = pd.Series(rng.randn(5) * 10 + 60, name="NYC_TMAX")

        train_loader, _ = create_dataloaders(X, y, X_v, y_v, batch_size=1000)
        assert len(train_loader) == 1
        X_batch, _ = next(iter(train_loader))
        assert X_batch.shape[0] == n
