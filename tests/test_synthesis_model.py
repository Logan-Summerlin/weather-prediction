"""
Tests for SynthesisModel, SynthesisTrainer, data preparation, and evaluation.

Validates:
  - SynthesisModel: instantiation, forward pass shapes (gaussian + quantile),
    gradient flow, parameter counts, batch norm / dropout toggles, edge cases.
  - SynthesisDataset: shapes, indexing, dtype.
  - prepare_synthesis_data: feature alignment, missing NWP handling, scaler
    behaviour, chronological splits, derived feature computation, validation.
  - SynthesisTrainer: loss computation, early stopping, checkpoint save/load,
    training convergence on synthetic data, predict method.
  - evaluate_synthesis: metric correctness, coverage calculation, comparison
    output format, seasonal breakdown.
  - Integration: end-to-end pipeline with synthetic data.
  - Edge cases: all NWP missing, single-sample batches, extreme values,
    empty DataFrames.

Target: >= 55 tests.
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

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.synthesis_model import (
    SynthesisModel,
    SynthesisDataset,
    SynthesisTrainer,
    prepare_synthesis_data,
    evaluate_synthesis,
    _compute_metrics,
    _compute_gaussian_coverage,
    _compute_quantile_calibration,
    _compute_seasonal_metrics,
    _validate_columns,
    DEFAULT_N_FEATURES,
    DEFAULT_QUANTILES,
    SYNTHESIS_NWP_FEATURES,
    SYNTHESIS_DERIVED_FEATURES,
    SYNTHESIS_SEASON_FEATURES,
    SYNTHESIS_STATION_FEATURES,
)
from src.crps_loss import GaussianCRPSLoss, PinballLoss, CombinedCRPSMAELoss


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="test_synthesis_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def gaussian_model():
    """SynthesisModel in gaussian mode with default features."""
    return SynthesisModel(
        n_features=DEFAULT_N_FEATURES,
        hidden_sizes=[64, 32],
        output_mode="gaussian",
        dropout=0.1,
        use_batch_norm=True,
    )


@pytest.fixture
def quantile_model():
    """SynthesisModel in quantile mode."""
    return SynthesisModel(
        n_features=DEFAULT_N_FEATURES,
        hidden_sizes=[64, 32],
        output_mode="quantile",
        quantiles=list(DEFAULT_QUANTILES),
        dropout=0.1,
        use_batch_norm=True,
    )


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def synthetic_features(batch_size):
    """Synthetic input features for the model."""
    torch.manual_seed(42)
    return torch.randn(batch_size, DEFAULT_N_FEATURES)


@pytest.fixture
def synthetic_targets(batch_size):
    """Synthetic targets (temperature in F)."""
    torch.manual_seed(42)
    return torch.randn(batch_size) * 15 + 65  # ~65F +/- 15F


@pytest.fixture
def synthetic_dataframes():
    """Create synthetic DataFrames for data preparation testing."""
    np.random.seed(42)
    n_days = 200

    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Station predictions
    station_preds = pd.DataFrame({
        "date": dates,
        "station_mu": np.random.normal(65, 15, n_days),
        "station_sigma": np.abs(np.random.normal(3, 1, n_days)),
    })

    # NWP features
    nwp_features = pd.DataFrame({
        "date": dates,
        "nwp_tmax": np.random.normal(65, 15, n_days),
        "nwp_t850": np.random.normal(5, 10, n_days),
        "nwp_wind_speed": np.abs(np.random.normal(10, 5, n_days)),
        "nwp_wind_dir": np.random.uniform(0, 360, n_days),
        "nwp_cloud_cover": np.random.uniform(0, 100, n_days),
        "nwp_mslp": np.random.normal(1013, 10, n_days),
        "nwp_precip": np.abs(np.random.normal(2, 3, n_days)),
        "nwp_ensemble_spread": np.abs(np.random.normal(2, 1, n_days)),
        "nwp_bias_7d": np.random.normal(0, 2, n_days),
        "sin_day": np.sin(2 * np.pi * np.arange(n_days) / 365.25),
        "cos_day": np.cos(2 * np.pi * np.arange(n_days) / 365.25),
    })

    # Observations
    observations = pd.DataFrame({
        "date": dates,
        "obs_tmax": np.random.normal(65, 15, n_days),
    })

    return station_preds, nwp_features, observations


# ===========================================================================
# 1. Model Architecture Tests
# ===========================================================================

class TestSynthesisModelArchitecture:
    """Tests for SynthesisModel construction and forward pass."""

    def test_gaussian_instantiation(self, gaussian_model):
        """Gaussian model can be created with valid config."""
        assert gaussian_model.output_mode == "gaussian"
        assert gaussian_model.n_features == DEFAULT_N_FEATURES
        assert hasattr(gaussian_model, "mu_head")
        assert hasattr(gaussian_model, "log_sigma_head")

    def test_quantile_instantiation(self, quantile_model):
        """Quantile model can be created with valid config."""
        assert quantile_model.output_mode == "quantile"
        assert quantile_model.n_quantiles == len(DEFAULT_QUANTILES)
        assert hasattr(quantile_model, "quantile_head")

    def test_invalid_output_mode_raises(self):
        """Invalid output_mode raises ValueError."""
        with pytest.raises(ValueError, match="output_mode must be"):
            SynthesisModel(n_features=15, output_mode="invalid")

    def test_gaussian_forward_shapes(self, gaussian_model, synthetic_features):
        """Gaussian forward pass returns correct shapes."""
        gaussian_model.eval()
        output = gaussian_model(synthetic_features)

        assert "prediction" in output
        assert "mu" in output
        assert "sigma" in output
        assert "log_sigma" in output

        B = synthetic_features.shape[0]
        assert output["prediction"].shape == (B, 1)
        assert output["mu"].shape == (B, 1)
        assert output["sigma"].shape == (B, 1)
        assert output["log_sigma"].shape == (B, 1)

    def test_quantile_forward_shapes(self, quantile_model, synthetic_features):
        """Quantile forward pass returns correct shapes."""
        quantile_model.eval()
        output = quantile_model(synthetic_features)

        assert "prediction" in output
        assert "quantiles" in output

        B = synthetic_features.shape[0]
        Q = len(DEFAULT_QUANTILES)
        assert output["quantiles"].shape == (B, Q)
        assert output["prediction"].shape == (B, 1)

    def test_sigma_positive(self, gaussian_model, synthetic_features):
        """Gaussian sigma output is always positive."""
        gaussian_model.eval()
        output = gaussian_model(synthetic_features)
        assert (output["sigma"] > 0).all()

    def test_log_sigma_clamped(self, gaussian_model, synthetic_features):
        """Log sigma is clamped to [-10, 5]."""
        gaussian_model.eval()
        output = gaussian_model(synthetic_features)
        assert (output["log_sigma"] >= -10.0).all()
        assert (output["log_sigma"] <= 5.0).all()

    def test_gradient_flow_gaussian(self, gaussian_model, synthetic_features, synthetic_targets):
        """Gradients flow through gaussian model to all parameters."""
        gaussian_model.train()
        output = gaussian_model(synthetic_features)
        # Use both mu and sigma in the loss so gradients flow to all heads
        loss = (
            (output["mu"].squeeze() - synthetic_targets).pow(2).mean()
            + output["sigma"].mean()
        )
        loss.backward()

        for name, param in gaussian_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_flow_quantile(self, quantile_model, synthetic_features, synthetic_targets):
        """Gradients flow through quantile model."""
        quantile_model.train()
        output = quantile_model(synthetic_features)
        loss = output["quantiles"].mean()
        loss.backward()

        for name, param in quantile_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count_reasonable(self, gaussian_model):
        """Model parameter count is within expected range."""
        n_params = sum(p.numel() for p in gaussian_model.parameters())
        # With hidden [64, 32], ~15 inputs, gaussian head:
        # Layer 1: 15*64 + 64 = 1024; BN: 128; Layer 2: 64*32+32=2080; BN: 64
        # mu_head: 32+1=33; log_sigma_head: 33 -> ~3362 total (approximate)
        assert 1000 < n_params < 50000, f"Unexpected param count: {n_params}"

    def test_no_batch_norm_mode(self):
        """Model works without batch normalisation."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32, 16],
            output_mode="gaussian", use_batch_norm=False,
        )
        x = torch.randn(8, 10)
        model.eval()
        output = model(x)
        assert output["mu"].shape == (8, 1)

    def test_no_dropout_mode(self):
        """Model works with dropout=0."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32],
            output_mode="gaussian", dropout=0.0,
        )
        model.eval()
        x = torch.randn(8, 10)
        output = model(x)
        assert output["mu"].shape == (8, 1)

    def test_custom_quantiles(self):
        """Model with custom quantile levels."""
        quantiles = [0.1, 0.5, 0.9]
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32],
            output_mode="quantile", quantiles=quantiles,
        )
        model.eval()
        x = torch.randn(4, 10)
        output = model(x)
        assert output["quantiles"].shape == (4, 3)

    def test_median_index_selection(self):
        """Median index selects correct quantile."""
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        model = SynthesisModel(
            n_features=10, output_mode="quantile", quantiles=quantiles,
        )
        assert model._get_median_index() == 2  # 0.5 is at index 2

    def test_single_hidden_layer(self):
        """Model works with single hidden layer."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[64],
            output_mode="gaussian",
        )
        model.eval()
        x = torch.randn(8, 10)
        output = model(x)
        assert output["mu"].shape == (8, 1)

    def test_model_name_property(self, gaussian_model):
        """Name property returns a descriptive string."""
        name = gaussian_model.name
        assert "Synthesis" in name
        assert "gaussian" in name

    def test_default_hidden_sizes(self):
        """Default hidden sizes are [128, 64, 32]."""
        model = SynthesisModel(n_features=10, output_mode="gaussian")
        assert model.hidden_sizes == [128, 64, 32]

    def test_prediction_matches_mu_gaussian(self, gaussian_model, synthetic_features):
        """In gaussian mode, prediction equals mu."""
        gaussian_model.eval()
        output = gaussian_model(synthetic_features)
        assert torch.allclose(output["prediction"], output["mu"])


# ===========================================================================
# 2. SynthesisDataset Tests
# ===========================================================================

class TestSynthesisDataset:
    """Tests for the SynthesisDataset."""

    def test_dataset_length(self):
        """Dataset reports correct length."""
        features = np.random.randn(100, 15).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        ds = SynthesisDataset(features, targets)
        assert len(ds) == 100

    def test_dataset_item_shape(self):
        """Dataset returns correctly shaped items."""
        features = np.random.randn(50, 10).astype(np.float32)
        targets = np.random.randn(50).astype(np.float32)
        ds = SynthesisDataset(features, targets)
        item = ds[0]
        assert item["features"].shape == (10,)
        assert item["target"].shape == ()

    def test_dataset_dtype(self):
        """Dataset items are float32 tensors."""
        features = np.random.randn(10, 5).astype(np.float64)
        targets = np.random.randn(10).astype(np.float64)
        ds = SynthesisDataset(features, targets)
        item = ds[0]
        assert item["features"].dtype == torch.float32
        assert item["target"].dtype == torch.float32

    def test_dataset_values_correct(self):
        """Dataset returns correct values."""
        features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        targets = np.array([10.0, 20.0], dtype=np.float32)
        ds = SynthesisDataset(features, targets)
        item = ds[1]
        assert torch.allclose(item["features"], torch.tensor([3.0, 4.0]))
        assert torch.allclose(item["target"], torch.tensor(20.0))


# ===========================================================================
# 3. Data Preparation Tests
# ===========================================================================

class TestPrepareData:
    """Tests for prepare_synthesis_data."""

    def test_basic_preparation(self, synthetic_dataframes):
        """Basic data preparation succeeds."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        assert "X_train" in result
        assert "y_train" in result
        assert "scaler" in result
        assert "feature_names" in result

    def test_chronological_split(self, synthetic_dataframes):
        """Train/val/test splits are chronological."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        # Dates should be sorted and non-overlapping
        train_end = result["dates_train"][-1]
        val_start = result["dates_val"][0]
        assert val_start >= train_end

        val_end = result["dates_val"][-1]
        test_start = result["dates_test"][0]
        assert test_start >= val_end

    def test_split_sizes(self, synthetic_dataframes):
        """Split sizes are approximately correct."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(
            station_preds, nwp_features, observations,
            train_ratio=0.7, val_ratio=0.15,
        )
        n_total = result["n_train"] + result["n_val"] + result["n_test"]
        assert n_total == 200
        assert result["n_train"] == 140  # 0.7 * 200
        assert result["n_val"] == 30     # 0.15 * 200
        assert result["n_test"] == 30    # remainder

    def test_scaler_fit_on_train_only(self, synthetic_dataframes):
        """StandardScaler is fit on training data only."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        scaler = result["scaler"]
        X_train = result["X_train"]

        # Training data should be approximately zero-mean unit-variance
        train_means = X_train.mean(axis=0)
        train_stds = X_train.std(axis=0)

        assert np.allclose(train_means, 0.0, atol=0.1), \
            f"Train means not near zero: {train_means}"
        # Std should be approximately 1 (some tolerance for small sample)
        assert np.all(train_stds < 2.0), f"Train stds too large: {train_stds}"

    def test_derived_features_computed(self, synthetic_dataframes):
        """Station-NWP gap features are computed."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        assert "station_nwp_gap" in result["feature_names"]
        assert "abs_station_nwp_gap" in result["feature_names"]

    def test_missing_nwp_handled(self, synthetic_dataframes):
        """Missing NWP values are imputed without errors."""
        station_preds, nwp_features, observations = synthetic_dataframes

        # Introduce some NaN values in NWP
        nwp_features_with_nan = nwp_features.copy()
        nwp_features_with_nan.loc[:10, "nwp_tmax"] = np.nan
        nwp_features_with_nan.loc[:5, "nwp_t850"] = np.nan

        result = prepare_synthesis_data(
            station_preds, nwp_features_with_nan, observations
        )

        # Should have no NaNs in output
        assert not np.any(np.isnan(result["X_train"]))
        assert not np.any(np.isnan(result["X_val"]))
        assert not np.any(np.isnan(result["X_test"]))

    def test_empty_nwp_handled(self, synthetic_dataframes):
        """Empty NWP DataFrame is handled gracefully."""
        station_preds, _, observations = synthetic_dataframes

        empty_nwp = pd.DataFrame(columns=["date"])
        result = prepare_synthesis_data(station_preds, empty_nwp, observations)

        assert result["n_train"] > 0
        assert not np.any(np.isnan(result["X_train"]))

    def test_missing_required_columns_raises(self, synthetic_dataframes):
        """Missing required columns in station_predictions raises ValueError."""
        _, nwp_features, observations = synthetic_dataframes

        bad_station = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)})
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_synthesis_data(bad_station, nwp_features, observations)

    def test_no_overlapping_dates_raises(self, synthetic_dataframes):
        """No overlapping dates raises ValueError."""
        station_preds, nwp_features, observations = synthetic_dataframes

        # Shift observation dates to non-overlapping range
        observations_shifted = observations.copy()
        observations_shifted["date"] = pd.date_range(
            "2025-01-01", periods=len(observations_shifted)
        )

        with pytest.raises(ValueError, match="No overlapping dates"):
            prepare_synthesis_data(
                station_preds, nwp_features, observations_shifted
            )

    def test_feature_dimensions_correct(self, synthetic_dataframes):
        """Feature matrix dimensions match feature_names."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        n_features = result["n_features"]
        assert result["X_train"].shape[1] == n_features
        assert result["X_val"].shape[1] == n_features
        assert result["X_test"].shape[1] == n_features
        assert len(result["feature_names"]) == n_features

    def test_output_dtypes(self, synthetic_dataframes):
        """Output arrays are float32."""
        station_preds, nwp_features, observations = synthetic_dataframes
        result = prepare_synthesis_data(station_preds, nwp_features, observations)

        assert result["X_train"].dtype == np.float32
        assert result["y_train"].dtype == np.float32


# ===========================================================================
# 4. _validate_columns Tests
# ===========================================================================

class TestValidateColumns:
    """Tests for the _validate_columns helper."""

    def test_valid_columns_no_error(self):
        """No error when all required columns present."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        _validate_columns(df, ["a", "b"], "test_df")

    def test_missing_columns_raises(self):
        """Missing columns raise ValueError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_columns(df, ["a", "b", "c"], "test_df")


# ===========================================================================
# 5. Training Tests
# ===========================================================================

class TestSynthesisTrainer:
    """Tests for the SynthesisTrainer class."""

    def test_trainer_initialisation(self, gaussian_model, tmp_dir):
        """Trainer initialises without errors."""
        trainer = SynthesisTrainer(
            model=gaussian_model, output_dir=tmp_dir,
            learning_rate=0.001, max_epochs=5, batch_size=16,
        )
        assert trainer.model is gaussian_model
        assert trainer.max_epochs == 5

    def test_gaussian_training_smoke(self, tmp_dir):
        """Gaussian model training completes without errors."""
        np.random.seed(42)
        torch.manual_seed(42)

        model = SynthesisModel(
            n_features=10, hidden_sizes=[32, 16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=0.01, max_epochs=5,
            early_stopping_patience=10, batch_size=16,
            loss_type="crps",
        )

        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32) * 10 + 65
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32) * 10 + 65

        result = trainer.train(X_train, y_train, X_val, y_val)

        assert "model" in result
        assert "history" in result
        assert "best_epoch" in result
        assert len(result["history"]) == 5

    def test_quantile_training_smoke(self, tmp_dir):
        """Quantile model training completes without errors."""
        np.random.seed(42)
        torch.manual_seed(42)

        model = SynthesisModel(
            n_features=10, hidden_sizes=[32, 16],
            output_mode="quantile", quantiles=[0.1, 0.5, 0.9],
            dropout=0.0, use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=0.01, max_epochs=5,
            early_stopping_patience=10, batch_size=16,
        )

        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32) * 10 + 65
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32) * 10 + 65

        result = trainer.train(X_train, y_train, X_val, y_val)
        assert len(result["history"]) == 5

    def test_early_stopping_triggers(self, tmp_dir):
        """Early stopping triggers when no improvement."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=1e-8,  # Tiny LR so model cannot improve
            max_epochs=50,
            early_stopping_patience=3,
            batch_size=64, loss_type="crps",
        )

        # Use different train/val distributions to prevent lucky improvements
        X_train = np.random.randn(50, 5).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32) + 100
        X_val = np.random.randn(20, 5).astype(np.float32) * 5
        y_val = np.random.randn(20).astype(np.float32) - 100

        result = trainer.train(X_train, y_train, X_val, y_val)

        # Should stop well before 50 epochs due to patience=3
        n_epochs = len(result["history"])
        # First epoch sets best, then 3 more without improvement = 4 total
        assert n_epochs <= 10, f"Expected early stopping, got {n_epochs} epochs"

    def test_checkpoint_saved(self, tmp_dir):
        """Best model checkpoint is saved to disk."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        assert os.path.isfile(trainer.checkpoint_path)

    def test_history_csv_saved(self, tmp_dir):
        """Training history is saved as CSV."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        assert os.path.isfile(trainer.history_path)
        # Read CSV and verify content
        history_df = pd.read_csv(trainer.history_path)
        assert len(history_df) == 3
        assert "epoch" in history_df.columns
        assert "train_loss" in history_df.columns
        assert "val_mae" in history_df.columns

    def test_training_curves_saved(self, tmp_dir):
        """Training curve plot is saved to disk."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        assert os.path.isfile(trainer.plot_path)

    def test_predict_gaussian(self, tmp_dir):
        """Predict method works for gaussian model."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=2, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        predictions = trainer.predict(X)
        assert "prediction" in predictions
        assert "mu" in predictions
        assert "sigma" in predictions
        assert predictions["mu"].shape == (30,)
        assert predictions["sigma"].shape == (30,)

    def test_predict_quantile(self, tmp_dir):
        """Predict method works for quantile model."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="quantile", quantiles=[0.1, 0.5, 0.9],
            dropout=0.0, use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=2, batch_size=16,
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        predictions = trainer.predict(X)
        assert "quantiles" in predictions
        assert "median" in predictions
        assert predictions["quantiles"].shape == (30, 3)
        assert predictions["median"].shape == (30,)

    def test_combined_loss_gaussian(self, tmp_dir):
        """Combined CRPS+MAE loss works for gaussian model."""
        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="combined",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        result = trainer.train(X, y, X, y)
        assert len(result["history"]) == 3

    def test_convergence_on_simple_data(self, tmp_dir):
        """Model converges on a simple learnable pattern."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Simple linear relationship: y = 2*x0 + 3*x1 + noise
        n = 200
        X = np.random.randn(n, 5).astype(np.float32)
        y = (2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n) * 0.5).astype(np.float32)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[32, 16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=0.005, max_epochs=100,
            early_stopping_patience=20, batch_size=32,
            loss_type="crps",
        )

        X_train, X_val = X[:160], X[160:]
        y_train, y_val = y[:160], y[160:]

        result = trainer.train(X_train, y_train, X_val, y_val)

        # Final MAE should be reasonable for this simple task
        predictions = trainer.predict(X_val)
        mae = np.mean(np.abs(predictions["mu"] - y_val))
        assert mae < 3.0, f"MAE too high for simple linear data: {mae:.2f}"


# ===========================================================================
# 6. Metrics & Evaluation Tests
# ===========================================================================

class TestMetrics:
    """Tests for metric computation functions."""

    def test_compute_metrics_perfect(self):
        """Perfect predictions give zero errors and R2=1."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = _compute_metrics(preds, targets, "test")

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["bias"] == 0.0

    def test_compute_metrics_known_values(self):
        """Metric values match hand-computed results."""
        preds = np.array([10.0, 12.0, 14.0])
        targets = np.array([11.0, 11.0, 13.0])
        metrics = _compute_metrics(preds, targets)

        # residuals: [-1, 1, 1], MAE = 1.0, bias = 1/3
        assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["bias"] == pytest.approx(1.0 / 3, abs=1e-6)
        assert metrics["max_error"] == pytest.approx(1.0, abs=1e-6)

    def test_compute_metrics_has_all_keys(self):
        """Metrics dict contains all expected keys."""
        preds = np.array([1.0, 2.0])
        targets = np.array([1.5, 2.5])
        metrics = _compute_metrics(preds, targets, "test")

        for key in ["mae", "rmse", "r2", "bias", "max_error", "name"]:
            assert key in metrics

    def test_gaussian_coverage_perfect(self):
        """Perfect gaussian (sigma=0+) gives 100% coverage."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([5.0, 5.0, 5.0])  # Wide uncertainty
        targets = np.array([1.0, 2.0, 3.0])  # Targets at means

        coverage = _compute_gaussian_coverage(mu, sigma, targets)
        # With targets at mu, all coverage levels should be 100%
        assert coverage["coverage_50%"] == pytest.approx(1.0, abs=1e-6)
        assert coverage["coverage_90%"] == pytest.approx(1.0, abs=1e-6)
        assert coverage["coverage_95%"] == pytest.approx(1.0, abs=1e-6)

    def test_gaussian_coverage_narrow(self):
        """Very narrow sigma gives lower coverage for outliers."""
        mu = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        targets = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Last one is far

        coverage = _compute_gaussian_coverage(mu, sigma, targets)
        # 4/5 targets are at mu, 1 is far away
        assert coverage["coverage_95%"] == pytest.approx(0.8, abs=1e-6)

    def test_quantile_calibration_format(self):
        """Quantile calibration returns correct keys."""
        n = 100
        quantile_preds = np.random.randn(n, 3)
        targets = np.random.randn(n)
        quantiles = [0.1, 0.5, 0.9]

        calib = _compute_quantile_calibration(quantile_preds, targets, quantiles)

        for q in quantiles:
            assert f"q{q:.3f}_actual" in calib
            assert f"q{q:.3f}_nominal" in calib
            assert 0.0 <= calib[f"q{q:.3f}_actual"] <= 1.0

    def test_seasonal_metrics(self):
        """Seasonal MAE computation produces valid results."""
        n = 365
        dates = pd.date_range("2020-01-01", periods=n)
        preds = np.random.randn(n) * 10 + 65
        targets = np.random.randn(n) * 10 + 65

        seasonal = _compute_seasonal_metrics(preds, targets, dates.values)

        for season in ["winter", "spring", "summer", "fall"]:
            key = f"mae_{season}"
            assert key in seasonal
            assert seasonal[key] > 0


# ===========================================================================
# 7. Evaluation Integration Tests
# ===========================================================================

class TestEvaluateSynthesis:
    """Tests for the evaluate_synthesis function."""

    def test_evaluate_gaussian_basic(self, tmp_dir):
        """evaluate_synthesis works for gaussian model."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 10 + 65
        trainer.train(X, y, X[:20], y[:20])

        result = evaluate_synthesis(
            trainer, X[:20], y[:20], output_dir=tmp_dir,
        )

        assert "synthesis_metrics" in result
        assert "coverage" in result
        assert result["synthesis_metrics"]["mae"] >= 0

    def test_evaluate_with_baselines(self, tmp_dir):
        """evaluate_synthesis includes baseline comparisons."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 10 + 65
        trainer.train(X, y, X[:20], y[:20])

        station_preds = np.random.randn(20).astype(np.float32) * 10 + 65
        nwp_preds = np.random.randn(20).astype(np.float32) * 10 + 65

        result = evaluate_synthesis(
            trainer, X[:20], y[:20],
            station_only_preds=station_preds,
            nwp_only_preds=nwp_preds,
            output_dir=tmp_dir,
        )

        assert "station_metrics" in result
        assert "nwp_metrics" in result
        assert result["station_metrics"]["mae"] >= 0
        assert result["nwp_metrics"]["mae"] >= 0

    def test_evaluate_with_dates(self, tmp_dir):
        """evaluate_synthesis computes seasonal breakdown with dates."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        n = 100
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32) * 10 + 65
        dates = pd.date_range("2020-01-01", periods=n).values
        trainer.train(X, y, X[:20], y[:20])

        result = evaluate_synthesis(
            trainer, X, y, dates_test=dates, output_dir=tmp_dir,
        )

        assert "seasonal" in result
        assert len(result["seasonal"]) > 0

    def test_evaluate_quantile_mode(self, tmp_dir):
        """evaluate_synthesis works for quantile model."""
        torch.manual_seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="quantile", quantiles=[0.1, 0.5, 0.9],
            dropout=0.0, use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16,
        )

        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 10 + 65
        trainer.train(X, y, X[:20], y[:20])

        result = evaluate_synthesis(
            trainer, X[:20], y[:20], output_dir=tmp_dir,
        )

        assert "coverage" in result
        assert "synthesis_metrics" in result

    def test_evaluation_plots_saved(self, tmp_dir):
        """Evaluation plots are saved to output directory."""
        torch.manual_seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=3, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32) * 10 + 65
        trainer.train(X, y, X, y)

        evaluate_synthesis(trainer, X, y, output_dir=tmp_dir)

        assert os.path.isfile(os.path.join(tmp_dir, "synthesis_scatter.png"))
        assert os.path.isfile(os.path.join(tmp_dir, "synthesis_residuals.png"))
        assert os.path.isfile(os.path.join(tmp_dir, "synthesis_uncertainty.png"))


# ===========================================================================
# 8. Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_sample_batch(self):
        """Model handles single-sample batches."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,  # BN needs >1 sample in train mode
        )
        model.eval()
        x = torch.randn(1, 10)
        output = model(x)
        assert output["mu"].shape == (1, 1)
        assert output["sigma"].shape == (1, 1)

    def test_large_batch(self):
        """Model handles large batches."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32],
            output_mode="gaussian", use_batch_norm=True,
        )
        model.eval()
        x = torch.randn(1000, 10)
        output = model(x)
        assert output["mu"].shape == (1000, 1)

    def test_extreme_input_values(self):
        """Model handles extreme input values without NaN."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        model.eval()
        x = torch.randn(8, 10) * 100  # Large values
        output = model(x)
        assert not torch.isnan(output["mu"]).any()
        assert not torch.isnan(output["sigma"]).any()

    def test_all_zero_input(self):
        """Model handles all-zero input."""
        model = SynthesisModel(
            n_features=10, hidden_sizes=[32],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        model.eval()
        x = torch.zeros(4, 10)
        output = model(x)
        assert not torch.isnan(output["mu"]).any()
        assert output["sigma"].shape == (4, 1)

    def test_all_nwp_missing_in_prep(self, synthetic_dataframes):
        """All NWP values missing still produces valid output."""
        station_preds, nwp_features, observations = synthetic_dataframes

        # Set all NWP values to NaN
        nwp_all_nan = nwp_features.copy()
        for col in SYNTHESIS_NWP_FEATURES:
            if col in nwp_all_nan.columns:
                nwp_all_nan[col] = np.nan

        result = prepare_synthesis_data(
            station_preds, nwp_all_nan, observations
        )

        assert not np.any(np.isnan(result["X_train"]))
        assert result["n_train"] > 0

    def test_small_dataset(self):
        """Data preparation with very small dataset."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")

        station_preds = pd.DataFrame({
            "date": dates,
            "station_mu": np.random.randn(10),
            "station_sigma": np.abs(np.random.randn(10)),
        })
        nwp_features = pd.DataFrame({"date": dates})
        observations = pd.DataFrame({
            "date": dates,
            "obs_tmax": np.random.randn(10) * 10 + 65,
        })

        result = prepare_synthesis_data(
            station_preds, nwp_features, observations,
            train_ratio=0.6, val_ratio=0.2,
        )

        assert result["n_train"] >= 1
        assert result["n_val"] >= 1
        assert result["n_test"] >= 1


# ===========================================================================
# 9. Full Integration Test
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_gaussian(self, tmp_dir, synthetic_dataframes):
        """Full pipeline: prepare data -> train -> evaluate (gaussian)."""
        torch.manual_seed(42)
        np.random.seed(42)

        station_preds, nwp_features, observations = synthetic_dataframes

        # Step 1: Prepare data
        data = prepare_synthesis_data(
            station_preds, nwp_features, observations,
        )

        # Step 2: Build and train model
        model = SynthesisModel(
            n_features=data["n_features"],
            hidden_sizes=[32, 16],
            output_mode="gaussian",
            dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=0.005, max_epochs=10,
            early_stopping_patience=20, batch_size=32,
            loss_type="crps",
        )

        result = trainer.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        assert result["best_epoch"] > 0
        assert result["best_val_mae"] > 0

        # Step 3: Evaluate
        eval_result = evaluate_synthesis(
            trainer,
            data["X_test"], data["y_test"],
            dates_test=data["dates_test"],
            output_dir=tmp_dir,
        )

        assert eval_result["synthesis_metrics"]["mae"] > 0
        assert eval_result["synthesis_metrics"]["rmse"] > 0
        assert "predictions" in eval_result

    def test_full_pipeline_quantile(self, tmp_dir, synthetic_dataframes):
        """Full pipeline: prepare data -> train -> evaluate (quantile)."""
        torch.manual_seed(42)
        np.random.seed(42)

        station_preds, nwp_features, observations = synthetic_dataframes

        data = prepare_synthesis_data(
            station_preds, nwp_features, observations,
        )

        model = SynthesisModel(
            n_features=data["n_features"],
            hidden_sizes=[32, 16],
            output_mode="quantile",
            quantiles=[0.1, 0.5, 0.9],
            dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            learning_rate=0.005, max_epochs=10,
            early_stopping_patience=20, batch_size=32,
        )

        result = trainer.train(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
        )
        assert result["best_epoch"] > 0

        eval_result = evaluate_synthesis(
            trainer,
            data["X_test"], data["y_test"],
            output_dir=tmp_dir,
        )

        assert eval_result["synthesis_metrics"]["mae"] > 0

    def test_checkpoint_reload_predictions_match(self, tmp_dir):
        """Reloaded checkpoint produces identical predictions."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        trainer = SynthesisTrainer(
            model=model, output_dir=tmp_dir,
            max_epochs=5, batch_size=16, loss_type="crps",
        )

        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)
        trainer.train(X, y, X, y)

        # Get predictions from trained model
        preds_before = trainer.predict(X)

        # Reload from checkpoint into a new model
        model2 = SynthesisModel(
            n_features=5, hidden_sizes=[16],
            output_mode="gaussian", dropout=0.0,
            use_batch_norm=False,
        )
        model2.load_state_dict(
            torch.load(trainer.checkpoint_path, weights_only=True)
        )
        model2.eval()

        # Create new trainer with the loaded model
        trainer2 = SynthesisTrainer(
            model=model2, output_dir=tmp_dir,
            max_epochs=1, batch_size=16, loss_type="crps",
        )
        preds_after = trainer2.predict(X)

        np.testing.assert_allclose(
            preds_before["mu"], preds_after["mu"],
            rtol=1e-5, atol=1e-5,
        )

    def test_different_loss_types_work(self, tmp_dir):
        """Both crps and combined loss types train successfully."""
        np.random.seed(42)
        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30).astype(np.float32)

        for loss_type in ["crps", "combined"]:
            model = SynthesisModel(
                n_features=5, hidden_sizes=[16],
                output_mode="gaussian", dropout=0.0,
                use_batch_norm=False,
            )
            sub_dir = os.path.join(tmp_dir, loss_type)
            trainer = SynthesisTrainer(
                model=model, output_dir=sub_dir,
                max_epochs=3, batch_size=16, loss_type=loss_type,
            )
            result = trainer.train(X, y, X, y)
            assert len(result["history"]) == 3, \
                f"Loss type '{loss_type}' failed"


# ===========================================================================
# 10. Constants & Module-Level Tests
# ===========================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_default_n_features(self):
        """DEFAULT_N_FEATURES matches sum of feature groups."""
        expected = (
            len(SYNTHESIS_STATION_FEATURES)
            + len(SYNTHESIS_NWP_FEATURES)
            + len(SYNTHESIS_DERIVED_FEATURES)
            + len(SYNTHESIS_SEASON_FEATURES)
        )
        assert DEFAULT_N_FEATURES == expected

    def test_default_quantiles_sorted(self):
        """Default quantiles are sorted and in (0, 1)."""
        assert DEFAULT_QUANTILES == sorted(DEFAULT_QUANTILES)
        assert all(0 < q < 1 for q in DEFAULT_QUANTILES)

    def test_default_quantiles_include_bounds_and_median(self):
        """Default quantiles include 0.025, 0.5, 0.975."""
        assert 0.025 in DEFAULT_QUANTILES
        assert 0.5 in DEFAULT_QUANTILES
        assert 0.975 in DEFAULT_QUANTILES
