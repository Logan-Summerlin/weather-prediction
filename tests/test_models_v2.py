"""
Tests for the Phase 4 model architectures (src/models_v2.py).

Validates:
  - EnhancedMLP: forward pass, batch norm, dropout, output shapes
  - MultiLagMLP: forward pass, different lag counts, shapes
  - LSTMPredictor: 3-D input, GRU variant, bidirectional, auto-reshape
  - StationAttentionModel: forward pass, attention weights, shapes
  - All models: batch_size=1, large batches, finite outputs, gradient flow
  - Model factory (create_model_v2) and loss factory (get_loss_function)
  - Reshape utilities
  - Model name properties
  - Parameter counts
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models_v2 import (
    EnhancedMLP,
    MultiLagMLP,
    LSTMPredictor,
    StationAttentionModel,
    create_model_v2,
    get_loss_function,
    reshape_for_lstm,
    reshape_for_attention,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def enhanced_mlp():
    """Create an EnhancedMLP with default settings."""
    return EnhancedMLP(n_features=30, hidden_sizes=[64, 32], dropout=0.1)


@pytest.fixture
def enhanced_mlp_bn():
    """Create an EnhancedMLP with batch normalisation."""
    return EnhancedMLP(
        n_features=30, hidden_sizes=[64, 32],
        dropout=0.1, use_batch_norm=True,
    )


@pytest.fixture
def multi_lag_mlp():
    """Create a MultiLagMLP for 3 lags of 30 features."""
    return MultiLagMLP(features_per_lag=30, n_lags=3, hidden_sizes=[128, 64])


@pytest.fixture
def lstm_model():
    """Create an LSTMPredictor."""
    return LSTMPredictor(input_size=30, hidden_size=32, num_layers=1)


@pytest.fixture
def gru_model():
    """Create a GRU-based LSTMPredictor."""
    return LSTMPredictor(
        input_size=30, hidden_size=32, num_layers=1, cell_type="gru",
    )


@pytest.fixture
def attention_model():
    """Create a StationAttentionModel for 14 stations x 2 features + 2 extra."""
    return StationAttentionModel(
        features_per_station=2, n_stations=14,
        embed_dim=32, n_heads=4, n_extra_features=2,
    )


@pytest.fixture
def batch_30():
    """Batch of 16 samples with 30 features."""
    torch.manual_seed(42)
    return torch.randn(16, 30)


@pytest.fixture
def batch_90():
    """Batch of 16 samples with 90 features (3 lags x 30)."""
    torch.manual_seed(42)
    return torch.randn(16, 90)


# ===========================================================================
# EnhancedMLP Tests
# ===========================================================================

class TestEnhancedMLP:
    """Tests for the EnhancedMLP model."""

    def test_forward_shape(self, enhanced_mlp, batch_30):
        """Output shape should be (batch, 1)."""
        out = enhanced_mlp(batch_30)
        assert out.shape == (16, 1)

    def test_forward_shape_batch1(self, enhanced_mlp):
        """Output shape for batch_size=1."""
        x = torch.randn(1, 30)
        out = enhanced_mlp(x)
        assert out.shape == (1, 1)

    def test_forward_shape_large_batch(self, enhanced_mlp):
        """Output shape for large batch."""
        x = torch.randn(256, 30)
        out = enhanced_mlp(x)
        assert out.shape == (256, 1)

    def test_finite_output(self, enhanced_mlp, batch_30):
        """All outputs should be finite."""
        enhanced_mlp.eval()
        with torch.no_grad():
            out = enhanced_mlp(batch_30)
        assert torch.all(torch.isfinite(out))

    def test_batch_norm_present(self, enhanced_mlp_bn):
        """BatchNorm1d should be present when use_batch_norm=True."""
        has_bn = any(
            isinstance(m, nn.BatchNorm1d)
            for m in enhanced_mlp_bn.network.modules()
        )
        assert has_bn

    def test_no_batch_norm_by_default(self, enhanced_mlp):
        """BatchNorm1d should NOT be present by default."""
        has_bn = any(
            isinstance(m, nn.BatchNorm1d)
            for m in enhanced_mlp.network.modules()
        )
        assert not has_bn

    def test_batch_norm_forward(self, enhanced_mlp_bn, batch_30):
        """Forward pass with batch norm should produce correct shapes."""
        out = enhanced_mlp_bn(batch_30)
        assert out.shape == (16, 1)
        assert torch.all(torch.isfinite(out))

    def test_dropout_present(self, enhanced_mlp):
        """Dropout should be present when dropout > 0."""
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in enhanced_mlp.network.modules()
        )
        assert has_dropout

    def test_no_dropout_when_zero(self):
        """Dropout should not be present when dropout=0.0."""
        model = EnhancedMLP(n_features=10, hidden_sizes=[16], dropout=0.0)
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model.network.modules()
        )
        assert not has_dropout

    def test_name_property(self, enhanced_mlp):
        """Name property should return a descriptive string."""
        assert "EnhancedMLP" in enhanced_mlp.name
        assert "[64, 32]" in enhanced_mlp.name

    def test_name_with_batch_norm(self, enhanced_mlp_bn):
        """Name should include BN indicator when batch norm is used."""
        assert "+BN" in enhanced_mlp_bn.name

    def test_gradient_flow(self, enhanced_mlp):
        """Gradients should flow to all parameters."""
        x = torch.randn(8, 30)
        out = enhanced_mlp(x)
        loss = out.sum()
        loss.backward()
        for name, param in enhanced_mlp.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_custom_hidden_sizes(self):
        """Model should work with various hidden size configs."""
        for sizes in [[16], [128, 64, 32], [256, 128, 64, 32]]:
            model = EnhancedMLP(n_features=10, hidden_sizes=sizes)
            out = model(torch.randn(4, 10))
            assert out.shape == (4, 1)

    def test_stored_attributes(self, enhanced_mlp):
        """Stored attributes should match constructor args."""
        assert enhanced_mlp.n_features == 30
        assert enhanced_mlp.hidden_sizes == [64, 32]
        assert enhanced_mlp.dropout_rate == 0.1
        assert enhanced_mlp.use_batch_norm is False

    def test_parameter_count(self):
        """Parameter count should match manual calculation.

        Architecture: 5 -> 16 -> 1  (no dropout, no BN)
        Layer 1: 5*16 + 16 = 96
        Output:  16*1 + 1  = 17
        Total = 113
        """
        model = EnhancedMLP(n_features=5, hidden_sizes=[16], dropout=0.0)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == (5 * 16 + 16) + (16 * 1 + 1)


# ===========================================================================
# MultiLagMLP Tests
# ===========================================================================

class TestMultiLagMLP:
    """Tests for the MultiLagMLP model."""

    def test_forward_shape(self, multi_lag_mlp, batch_90):
        """Output shape should be (batch, 1)."""
        out = multi_lag_mlp(batch_90)
        assert out.shape == (16, 1)

    def test_forward_shape_batch1(self, multi_lag_mlp):
        """Output shape for batch_size=1."""
        x = torch.randn(1, 90)
        out = multi_lag_mlp(x)
        assert out.shape == (1, 1)

    def test_finite_output(self, multi_lag_mlp, batch_90):
        """All outputs should be finite."""
        multi_lag_mlp.eval()
        with torch.no_grad():
            out = multi_lag_mlp(batch_90)
        assert torch.all(torch.isfinite(out))

    def test_different_lag_counts(self):
        """Model should work with different lag counts."""
        for n_lags in [1, 2, 5]:
            model = MultiLagMLP(features_per_lag=10, n_lags=n_lags)
            x = torch.randn(8, 10 * n_lags)
            out = model(x)
            assert out.shape == (8, 1)

    def test_n_features_property(self, multi_lag_mlp):
        """n_features should be features_per_lag * n_lags."""
        assert multi_lag_mlp.n_features == 30 * 3

    def test_name_property(self, multi_lag_mlp):
        """Name should include lag count."""
        assert "lags=3" in multi_lag_mlp.name

    def test_gradient_flow(self, multi_lag_mlp):
        """Gradients should flow to all parameters."""
        x = torch.randn(8, 90)
        out = multi_lag_mlp(x)
        out.sum().backward()
        for name, param in multi_lag_mlp.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ===========================================================================
# LSTMPredictor Tests
# ===========================================================================

class TestLSTMPredictor:
    """Tests for the LSTMPredictor model."""

    def test_forward_3d_input(self, lstm_model):
        """Forward pass with 3-D input (batch, seq_len, features)."""
        x = torch.randn(16, 5, 30)  # 5 time steps
        out = lstm_model(x)
        assert out.shape == (16, 1)

    def test_forward_2d_auto_reshape(self, lstm_model, batch_30):
        """2-D input should be auto-reshaped to (batch, 1, features)."""
        out = lstm_model(batch_30)
        assert out.shape == (16, 1)

    def test_forward_batch1(self, lstm_model):
        """Batch size 1 should work."""
        x = torch.randn(1, 3, 30)
        out = lstm_model(x)
        assert out.shape == (1, 1)

    def test_finite_output(self, lstm_model, batch_30):
        """Outputs should be finite."""
        lstm_model.eval()
        with torch.no_grad():
            out = lstm_model(batch_30)
        assert torch.all(torch.isfinite(out))

    def test_gru_cell_type(self, gru_model, batch_30):
        """GRU variant should work."""
        out = gru_model(batch_30)
        assert out.shape == (16, 1)
        assert gru_model.cell_type == "gru"

    def test_invalid_cell_type(self):
        """Invalid cell type should raise ValueError."""
        with pytest.raises(ValueError, match="cell_type"):
            LSTMPredictor(input_size=10, cell_type="rnn")

    def test_bidirectional(self):
        """Bidirectional LSTM should work."""
        model = LSTMPredictor(
            input_size=10, hidden_size=16, bidirectional=True,
        )
        x = torch.randn(8, 3, 10)
        out = model(x)
        assert out.shape == (8, 1)
        assert model.bidirectional is True

    def test_multi_layer(self):
        """Multi-layer LSTM should work."""
        model = LSTMPredictor(
            input_size=10, hidden_size=16, num_layers=2, dropout=0.1,
        )
        x = torch.randn(8, 3, 10)
        out = model(x)
        assert out.shape == (8, 1)

    def test_name_property(self, lstm_model):
        """Name should include cell type and hidden size."""
        assert "LSTM" in lstm_model.name
        assert "h=32" in lstm_model.name

    def test_gru_name_property(self, gru_model):
        """GRU name should say GRU."""
        assert "GRU" in gru_model.name

    def test_gradient_flow(self, lstm_model):
        """Gradients should flow to all parameters."""
        x = torch.randn(8, 3, 30)
        out = lstm_model(x)
        out.sum().backward()
        for name, param in lstm_model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ===========================================================================
# StationAttentionModel Tests
# ===========================================================================

class TestStationAttentionModel:
    """Tests for the StationAttentionModel."""

    def test_forward_shape(self, attention_model, batch_30):
        """Output shape should be (batch, 1) for 30-feature input."""
        out = attention_model(batch_30)
        assert out.shape == (16, 1)

    def test_forward_batch1(self, attention_model):
        """Batch size 1 should work."""
        x = torch.randn(1, 30)
        out = attention_model(x)
        assert out.shape == (1, 1)

    def test_finite_output(self, attention_model, batch_30):
        """Outputs should be finite."""
        attention_model.eval()
        with torch.no_grad():
            out = attention_model(batch_30)
        assert torch.all(torch.isfinite(out))

    def test_attention_weights_shape(self, attention_model, batch_30):
        """After forward pass, attention weights should have correct shape."""
        attention_model.eval()
        with torch.no_grad():
            _ = attention_model(batch_30)
        weights = attention_model.get_attention_weights()
        assert weights is not None
        # shape: (batch, 1, n_stations) = (16, 1, 14)
        assert weights.shape == (16, 1, 14)

    def test_attention_weights_sum(self, attention_model, batch_30):
        """Attention weights should sum to approximately 1."""
        attention_model.eval()
        with torch.no_grad():
            _ = attention_model(batch_30)
        weights = attention_model.get_attention_weights()
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_extra_features(self):
        """Model should work with n_extra_features=0."""
        model = StationAttentionModel(
            features_per_station=2, n_stations=14,
            embed_dim=16, n_heads=4, n_extra_features=0,
        )
        x = torch.randn(8, 28)  # 14 * 2 = 28
        out = model(x)
        assert out.shape == (8, 1)

    def test_name_property(self, attention_model):
        """Name should include station count and embed dim."""
        assert "StationAttn" in attention_model.name
        assert "s=14" in attention_model.name

    def test_gradient_flow(self, attention_model, batch_30):
        """Gradients should flow to all parameters."""
        out = attention_model(batch_30)
        out.sum().backward()
        for name, param in attention_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_different_station_configs(self):
        """Model should work with different station configurations."""
        for n_st, f_per_st in [(5, 3), (10, 2), (20, 1)]:
            embed = 16
            model = StationAttentionModel(
                features_per_station=f_per_st, n_stations=n_st,
                embed_dim=embed, n_heads=4, n_extra_features=0,
            )
            x = torch.randn(4, n_st * f_per_st)
            out = model(x)
            assert out.shape == (4, 1)


# ===========================================================================
# Loss Function Factory Tests
# ===========================================================================

class TestGetLossFunction:
    """Tests for the get_loss_function factory."""

    def test_mse(self):
        """MSE loss should be returned."""
        loss = get_loss_function("mse")
        assert isinstance(loss, nn.MSELoss)

    def test_huber(self):
        """Huber loss should be returned."""
        loss = get_loss_function("huber")
        assert isinstance(loss, nn.HuberLoss)

    def test_mae(self):
        """MAE loss should be returned."""
        loss = get_loss_function("mae")
        assert isinstance(loss, nn.L1Loss)

    def test_l1_alias(self):
        """'l1' should work as an alias for MAE."""
        loss = get_loss_function("l1")
        assert isinstance(loss, nn.L1Loss)

    def test_case_insensitive(self):
        """Loss type should be case-insensitive."""
        loss = get_loss_function("MSE")
        assert isinstance(loss, nn.MSELoss)

    def test_invalid_raises(self):
        """Unknown loss type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_function("unknown")


# ===========================================================================
# Reshape Utility Tests
# ===========================================================================

class TestReshapeUtilities:
    """Tests for reshape_for_lstm and reshape_for_attention."""

    def test_reshape_for_lstm_basic(self):
        """Basic reshape: (8, 60) -> (8, 3, 20)."""
        x = torch.randn(8, 60)
        out = reshape_for_lstm(x, seq_len=3)
        assert out.shape == (8, 3, 20)

    def test_reshape_for_lstm_seq1(self):
        """seq_len=1: (8, 30) -> (8, 1, 30)."""
        x = torch.randn(8, 30)
        out = reshape_for_lstm(x, seq_len=1)
        assert out.shape == (8, 1, 30)

    def test_reshape_for_lstm_invalid_dim(self):
        """3-D input should raise ValueError."""
        x = torch.randn(8, 3, 10)
        with pytest.raises(ValueError, match="2-D"):
            reshape_for_lstm(x, seq_len=3)

    def test_reshape_for_lstm_not_divisible(self):
        """Non-divisible total should raise ValueError."""
        x = torch.randn(8, 31)
        with pytest.raises(ValueError, match="not divisible"):
            reshape_for_lstm(x, seq_len=3)

    def test_reshape_for_attention_valid(self):
        """Valid shape should pass through."""
        x = torch.randn(8, 30)
        out = reshape_for_attention(x, n_stations=14, features_per_station=2,
                                     n_extra=2)
        assert out.shape == (8, 30)

    def test_reshape_for_attention_invalid(self):
        """Mismatched dimensions should raise ValueError."""
        x = torch.randn(8, 25)
        with pytest.raises(ValueError, match="Expected"):
            reshape_for_attention(x, n_stations=14, features_per_station=2,
                                   n_extra=2)


# ===========================================================================
# Model Factory Tests
# ===========================================================================

class TestCreateModelV2:
    """Tests for the create_model_v2 factory function."""

    def test_create_enhanced_mlp(self):
        """Should create an EnhancedMLP."""
        model = create_model_v2("enhanced_mlp", n_features=30)
        assert isinstance(model, EnhancedMLP)

    def test_create_lstm(self):
        """Should create an LSTMPredictor with LSTM cell."""
        model = create_model_v2("lstm", n_features=30, hidden_size=32)
        assert isinstance(model, LSTMPredictor)
        assert model.cell_type == "lstm"

    def test_create_gru(self):
        """Should create an LSTMPredictor with GRU cell."""
        model = create_model_v2("gru", n_features=30, hidden_size=32)
        assert isinstance(model, LSTMPredictor)
        assert model.cell_type == "gru"

    def test_create_attention(self):
        """Should create a StationAttentionModel."""
        model = create_model_v2(
            "attention", n_features=30,
            features_per_station=2, n_stations=14, n_extra_features=2,
        )
        assert isinstance(model, StationAttentionModel)

    def test_create_unknown_raises(self):
        """Unknown model class should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model_class"):
            create_model_v2("transformer", n_features=30)


# ===========================================================================
# Cross-Model Integration Tests
# ===========================================================================

class TestIntegration:
    """End-to-end integration tests across all model types."""

    @pytest.mark.parametrize("ModelClass,kwargs,input_shape", [
        (EnhancedMLP, {"n_features": 10, "hidden_sizes": [16, 8]}, (4, 10)),
        (MultiLagMLP, {"features_per_lag": 10, "n_lags": 2, "hidden_sizes": [16]}, (4, 20)),
        (LSTMPredictor, {"input_size": 10, "hidden_size": 16}, (4, 3, 10)),
        (StationAttentionModel, {"features_per_station": 2, "n_stations": 5, "embed_dim": 8, "n_heads": 2, "n_extra_features": 0}, (4, 10)),
    ])
    def test_training_step(self, ModelClass, kwargs, input_shape):
        """Each model should support a standard training step."""
        model = ModelClass(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x = torch.randn(*input_shape)
        target = torch.randn(input_shape[0], 1)

        model.train()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        assert pred.shape == (input_shape[0], 1)

    @pytest.mark.parametrize("ModelClass,kwargs,input_shape", [
        (EnhancedMLP, {"n_features": 10, "hidden_sizes": [16]}, (4, 10)),
        (MultiLagMLP, {"features_per_lag": 10, "n_lags": 1, "hidden_sizes": [16]}, (4, 10)),
        (LSTMPredictor, {"input_size": 10, "hidden_size": 16}, (4, 10)),
        (StationAttentionModel, {"features_per_station": 2, "n_stations": 5, "embed_dim": 8, "n_heads": 2, "n_extra_features": 0}, (4, 10)),
    ])
    def test_eval_deterministic(self, ModelClass, kwargs, input_shape):
        """In eval mode, outputs should be deterministic."""
        model = ModelClass(**kwargs)
        model.eval()
        x = torch.randn(*input_shape)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_all_models_param_count_positive(self):
        """All models should have positive parameter counts."""
        models = [
            EnhancedMLP(n_features=10, hidden_sizes=[8]),
            MultiLagMLP(features_per_lag=10, n_lags=1, hidden_sizes=[8]),
            LSTMPredictor(input_size=10, hidden_size=8),
            StationAttentionModel(
                features_per_station=2, n_stations=5,
                embed_dim=8, n_heads=2, n_extra_features=0,
            ),
        ]
        for model in models:
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params > 0, f"{type(model).__name__} has 0 params"

    def test_all_models_zero_input(self):
        """All models should handle zero input without error."""
        configs = [
            (EnhancedMLP(n_features=10, hidden_sizes=[8]), torch.zeros(4, 10)),
            (MultiLagMLP(features_per_lag=10, n_lags=1, hidden_sizes=[8]), torch.zeros(4, 10)),
            (LSTMPredictor(input_size=10, hidden_size=8), torch.zeros(4, 10)),
            (StationAttentionModel(
                features_per_station=2, n_stations=5,
                embed_dim=8, n_heads=2, n_extra_features=0,
            ), torch.zeros(4, 10)),
        ]
        for model, x in configs:
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (4, 1)
            assert torch.all(torch.isfinite(out))
