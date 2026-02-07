"""
Tests for the PyTorch neural network model (src/model.py).

Validates:
  - TempPredictorV1 initialization (defaults, custom args, edge cases)
  - Forward pass (output shape, finite outputs, gradient flow)
  - Factory function create_model (default and custom configurations)
  - Helper functions count_parameters and get_model_summary
  - Training vs. evaluation mode behavior (dropout differences)
  - Device placement (CPU)
  - Parameter trainability (requires_grad)
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    TempPredictorV1,
    create_model,
    count_parameters,
    get_model_summary,
)
import config


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def default_model():
    """Create a TempPredictorV1 with the default project configuration.

    Uses 30 input features (the actual project feature count) and
    config defaults for hidden_sizes and dropout.
    """
    return TempPredictorV1(n_features=30)


@pytest.fixture
def custom_model():
    """Create a TempPredictorV1 with custom hidden sizes and dropout."""
    return TempPredictorV1(
        n_features=10,
        hidden_sizes=[128, 64, 32],
        dropout=0.2,
    )


@pytest.fixture
def sample_input_30():
    """A batch of 16 samples with 30 features (standard project input)."""
    torch.manual_seed(42)
    return torch.randn(16, 30)


@pytest.fixture
def sample_input_10():
    """A batch of 16 samples with 10 features."""
    torch.manual_seed(42)
    return torch.randn(16, 10)


# ===========================================================================
# Initialization Tests
# ===========================================================================

class TestInitialization:
    """Tests for TempPredictorV1 __init__ behavior."""

    def test_default_hidden_sizes(self, default_model):
        """Default hidden_sizes should match config.HIDDEN_SIZES."""
        assert default_model.hidden_sizes == config.HIDDEN_SIZES

    def test_default_dropout(self, default_model):
        """Default dropout should match config.DROPOUT."""
        assert default_model.dropout_rate == config.DROPOUT

    def test_n_features_stored(self, default_model):
        """n_features should be stored as an attribute."""
        assert default_model.n_features == 30

    def test_custom_hidden_sizes(self, custom_model):
        """Custom hidden_sizes should be stored correctly."""
        assert custom_model.hidden_sizes == [128, 64, 32]

    def test_custom_dropout(self, custom_model):
        """Custom dropout should be stored correctly."""
        assert custom_model.dropout_rate == 0.2

    def test_custom_n_features(self, custom_model):
        """Custom n_features should be stored correctly."""
        assert custom_model.n_features == 10

    def test_single_hidden_layer(self):
        """Model should work with a single hidden layer."""
        model = TempPredictorV1(n_features=5, hidden_sizes=[16])
        assert model.hidden_sizes == [16]
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_many_hidden_layers(self):
        """Model should work with many hidden layers."""
        model = TempPredictorV1(
            n_features=5,
            hidden_sizes=[64, 32, 16, 8, 4],
        )
        assert model.hidden_sizes == [64, 32, 16, 8, 4]
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_no_dropout(self):
        """Model should work with dropout=0.0 (no dropout layers)."""
        model = TempPredictorV1(n_features=5, hidden_sizes=[16], dropout=0.0)
        assert model.dropout_rate == 0.0
        # Verify no Dropout modules are in the network
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model.network.modules()
        )
        assert not has_dropout, "No Dropout layers should exist when dropout=0.0"

    def test_dropout_present_when_nonzero(self):
        """Dropout layers should be present when dropout > 0."""
        model = TempPredictorV1(n_features=30, dropout=0.2)
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model.network.modules()
        )
        assert has_dropout, "Dropout layers should exist when dropout > 0"

    def test_network_is_sequential(self, default_model):
        """The internal network should be an nn.Sequential."""
        assert isinstance(default_model.network, nn.Sequential)

    def test_is_nn_module(self, default_model):
        """TempPredictorV1 should be a proper nn.Module subclass."""
        assert isinstance(default_model, nn.Module)

    def test_default_hidden_sizes_not_mutated(self):
        """Creating a model should not mutate config.HIDDEN_SIZES."""
        original = list(config.HIDDEN_SIZES)
        _ = TempPredictorV1(n_features=5)
        assert config.HIDDEN_SIZES == original


# ===========================================================================
# Forward Pass Tests
# ===========================================================================

class TestForwardPass:
    """Tests for the forward pass behavior."""

    def test_output_shape_default(self, default_model, sample_input_30):
        """Output shape should be (batch_size, 1)."""
        out = default_model(sample_input_30)
        assert out.shape == (16, 1)

    def test_output_shape_custom(self, custom_model, sample_input_10):
        """Output shape should be (batch_size, 1) for custom models."""
        out = custom_model(sample_input_10)
        assert out.shape == (16, 1)

    def test_output_shape_single_sample(self, default_model):
        """Output shape should be (1, 1) for a single sample."""
        x = torch.randn(1, 30)
        out = default_model(x)
        assert out.shape == (1, 1)

    def test_output_shape_large_batch(self, default_model):
        """Output shape should work for large batches."""
        x = torch.randn(512, 30)
        out = default_model(x)
        assert out.shape == (512, 1)

    def test_output_shape_various_batch_sizes(self, default_model):
        """Output first dim should match the batch size."""
        for batch_size in [1, 2, 8, 32, 64, 128]:
            x = torch.randn(batch_size, 30)
            out = default_model(x)
            assert out.shape == (batch_size, 1), (
                f"Failed for batch_size={batch_size}"
            )

    def test_output_finite(self, default_model, sample_input_30):
        """All output values should be finite (no NaN or Inf)."""
        default_model.eval()
        with torch.no_grad():
            out = default_model(sample_input_30)
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

    def test_output_finite_custom(self, custom_model, sample_input_10):
        """Output should be finite for custom model too."""
        custom_model.eval()
        with torch.no_grad():
            out = custom_model(sample_input_10)
        assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf"

    def test_gradient_flows_through_all_parameters(self, default_model):
        """Gradients should reach every trainable parameter after backward."""
        x = torch.randn(8, 30)
        out = default_model(x)
        loss = out.sum()
        loss.backward()

        for name, param in default_model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.any(param.grad != 0), (
                f"All-zero gradient for parameter: {name}"
            )

    def test_output_with_30_features(self):
        """Model should work with the actual project feature count (30)."""
        model = TempPredictorV1(n_features=30)
        x = torch.randn(32, 30)
        out = model(x)
        assert out.shape == (32, 1)
        assert torch.all(torch.isfinite(out))

    def test_output_dtype_float32(self, default_model, sample_input_30):
        """Output tensor should be float32."""
        out = default_model(sample_input_30)
        assert out.dtype == torch.float32

    def test_deterministic_in_eval_mode(self, default_model, sample_input_30):
        """In eval mode, outputs should be deterministic."""
        default_model.eval()
        with torch.no_grad():
            out1 = default_model(sample_input_30)
            out2 = default_model(sample_input_30)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_zero_input(self, default_model):
        """Model should handle all-zero input without errors."""
        x = torch.zeros(4, 30)
        out = default_model(x)
        assert out.shape == (4, 1)
        assert torch.all(torch.isfinite(out))


# ===========================================================================
# Helper Function Tests
# ===========================================================================

class TestHelperFunctions:
    """Tests for create_model, count_parameters, and get_model_summary."""

    def test_create_model_returns_correct_type(self):
        """create_model should return a TempPredictorV1 instance."""
        model = create_model(n_features=30)
        assert isinstance(model, TempPredictorV1)

    def test_create_model_default_config(self):
        """create_model with defaults should use config values."""
        model = create_model(n_features=30)
        assert model.hidden_sizes == config.HIDDEN_SIZES
        assert model.dropout_rate == config.DROPOUT
        assert model.n_features == 30

    def test_create_model_custom_args(self):
        """create_model should pass through custom hidden_sizes and dropout."""
        model = create_model(
            n_features=15,
            hidden_sizes=[256, 128],
            dropout=0.3,
        )
        assert model.hidden_sizes == [256, 128]
        assert model.dropout_rate == 0.3
        assert model.n_features == 15

    def test_count_parameters_correctness(self):
        """count_parameters should return the correct total for a known arch.

        Architecture: 5 -> 16 -> 1
        Layer 1: 5*16 weights + 16 biases = 96
        Layer 2: 16*1 weights + 1 bias = 17
        Total = 113
        """
        model = TempPredictorV1(n_features=5, hidden_sizes=[16], dropout=0.0)
        expected = (5 * 16 + 16) + (16 * 1 + 1)
        assert count_parameters(model) == expected

    def test_count_parameters_two_hidden(self):
        """count_parameters for a two-hidden-layer network.

        Architecture: 10 -> 32 -> 16 -> 1
        Layer 1: 10*32 + 32 = 352
        Layer 2: 32*16 + 16 = 528
        Output:  16*1  + 1  = 17
        Total = 897
        """
        model = TempPredictorV1(
            n_features=10, hidden_sizes=[32, 16], dropout=0.0,
        )
        expected = (10 * 32 + 32) + (32 * 16 + 16) + (16 * 1 + 1)
        assert count_parameters(model) == expected

    def test_count_parameters_default_model(self, default_model):
        """count_parameters should return a positive integer."""
        n_params = count_parameters(default_model)
        assert isinstance(n_params, int)
        assert n_params > 0

    def test_get_model_summary_returns_string(self, default_model):
        """get_model_summary should return a non-empty string."""
        summary = get_model_summary(default_model)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_model_summary_contains_class_name(self, default_model):
        """Summary should contain the model class name."""
        summary = get_model_summary(default_model)
        assert "TempPredictorV1" in summary

    def test_get_model_summary_contains_param_count(self, default_model):
        """Summary should contain parameter count information."""
        summary = get_model_summary(default_model)
        assert "trainable parameters" in summary.lower() or "parameters" in summary.lower()

    def test_get_model_summary_with_n_features(self, default_model):
        """Summary with n_features should include output shape info."""
        summary = get_model_summary(default_model, n_features=30)
        assert "Output shape" in summary or "output shape" in summary

    def test_get_model_summary_without_n_features(self, default_model):
        """Summary without n_features should still return valid string."""
        summary = get_model_summary(default_model, n_features=None)
        assert isinstance(summary, str)
        assert len(summary) > 0


# ===========================================================================
# Training / Evaluation Mode Tests
# ===========================================================================

class TestTrainingMode:
    """Tests for train vs. eval mode behavior."""

    def test_dropout_active_in_train_mode(self):
        """Model outputs should vary in train mode (due to dropout).

        We use a high dropout to make the stochastic effect detectable.
        """
        model = TempPredictorV1(n_features=10, hidden_sizes=[64], dropout=0.5)
        model.train()

        torch.manual_seed(0)
        x = torch.randn(64, 10)

        # Run forward twice in train mode with different seeds
        torch.manual_seed(1)
        out1 = model(x)
        torch.manual_seed(2)
        out2 = model(x)

        # With 50% dropout on 64 neurons over 64 samples, outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Outputs should differ in train mode due to dropout"
        )

    def test_dropout_inactive_in_eval_mode(self, default_model, sample_input_30):
        """In eval mode, dropout is disabled so outputs should be identical."""
        default_model.eval()
        with torch.no_grad():
            out1 = default_model(sample_input_30)
            out2 = default_model(sample_input_30)
        assert torch.allclose(out1, out2), "Eval mode outputs should be identical"

    def test_all_parameters_trainable(self, default_model):
        """Every parameter should have requires_grad=True."""
        for name, param in default_model.named_parameters():
            assert param.requires_grad, (
                f"Parameter {name} has requires_grad=False"
            )

    def test_model_can_compute_loss_and_backward(self, default_model):
        """Model should work in a standard training step (forward+loss+backward)."""
        criterion = nn.MSELoss()
        x = torch.randn(8, 30)
        target = torch.randn(8, 1)

        pred = default_model(x)
        loss = criterion(pred, target)
        loss.backward()

        # Verify loss is a scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_optimizer_can_step(self, default_model):
        """An optimizer should be able to update model parameters."""
        optimizer = torch.optim.Adam(default_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x = torch.randn(8, 30)
        target = torch.randn(8, 1)

        # Capture initial parameters
        initial_params = {
            name: param.clone() for name, param in default_model.named_parameters()
        }

        pred = default_model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        # At least some parameters should have changed
        any_changed = False
        for name, param in default_model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                any_changed = True
                break

        assert any_changed, "Optimizer step should update at least some parameters"


# ===========================================================================
# Device Tests
# ===========================================================================

class TestDevice:
    """Tests for device placement."""

    def test_model_on_cpu(self, default_model):
        """Model should work on CPU (always available)."""
        model = default_model.to("cpu")
        x = torch.randn(4, 30)
        out = model(x)
        assert out.device.type == "cpu"

    def test_parameters_on_cpu(self, default_model):
        """All parameters should be on CPU."""
        for name, param in default_model.named_parameters():
            assert param.device.type == "cpu", (
                f"Parameter {name} is on {param.device}, expected cpu"
            )

    def test_model_move_to_cpu(self):
        """Model should be movable to CPU via .to() method."""
        model = TempPredictorV1(n_features=10, hidden_sizes=[32])
        model = model.to(torch.device("cpu"))
        x = torch.randn(4, 10)
        out = model(x)
        assert out.device.type == "cpu"


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Tests for boundary and unusual configurations."""

    def test_single_feature_input(self):
        """Model should work with a single input feature."""
        model = TempPredictorV1(n_features=1, hidden_sizes=[8])
        x = torch.randn(4, 1)
        out = model(x)
        assert out.shape == (4, 1)
        assert torch.all(torch.isfinite(out))

    def test_large_feature_input(self):
        """Model should work with many input features."""
        model = TempPredictorV1(n_features=200, hidden_sizes=[64, 32])
        x = torch.randn(4, 200)
        out = model(x)
        assert out.shape == (4, 1)
        assert torch.all(torch.isfinite(out))

    def test_wide_hidden_layer(self):
        """Model should work with a very wide hidden layer."""
        model = TempPredictorV1(n_features=5, hidden_sizes=[512])
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_large_input_values(self):
        """Model should handle large but finite input values."""
        model = TempPredictorV1(n_features=5, hidden_sizes=[16])
        model.eval()
        x = torch.ones(4, 5) * 100.0
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)
        assert torch.all(torch.isfinite(out))

    def test_negative_input_values(self):
        """Model should handle negative input values."""
        model = TempPredictorV1(n_features=5, hidden_sizes=[16])
        model.eval()
        x = torch.ones(4, 5) * -50.0
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)
        assert torch.all(torch.isfinite(out))

    def test_model_repr(self, default_model):
        """Model should have a sensible string representation."""
        repr_str = repr(default_model)
        assert "TempPredictorV1" in repr_str
        assert "Sequential" in repr_str or "network" in repr_str
