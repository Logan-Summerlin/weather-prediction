"""
Tests for Wind-Gated Attention Model, CRPS Loss, and Training Pipeline.

Validates:
  - WindGatedAttentionModel: instantiation, forward pass (point + gaussian),
    attention weights, wind bias, masking, gradient flow, edge cases.
  - GaussianCRPSLoss: correctness against known values, gradient flow,
    reduction modes.
  - CombinedCRPSMAELoss: weighting, gradient flow.
  - PinballLoss: correctness, gradient flow.
  - EnergyCRPSLoss: positive output, gradient flow.
  - AttentionDataset: shapes, indexing.
  - Training pipeline: smoke test, save/load round-trip.

Target: >= 40 tests.
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

from src.wind_gated_attention import WindGatedAttentionModel
from src.crps_loss import (
    GaussianCRPSLoss,
    EnergyCRPSLoss,
    PinballLoss,
    CombinedCRPSMAELoss,
    _standard_normal_pdf,
    _standard_normal_cdf,
)
from src.train_phase1 import (
    AttentionDataset,
    create_attention_dataloaders,
    train_one_epoch,
    validate,
    train_wind_gated_model,
    save_training_history,
    plot_training_curves,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def model_config():
    """Default model configuration."""
    return {
        "n_station_features": 6,
        "n_metadata_features": 4,
        "n_global_features": 7,
        "n_stations": 14,
        "station_embed_dim": 32,
        "attention_dim": 16,
        "output_mode": "point",
        "dropout": 0.1,
    }


@pytest.fixture
def point_model(model_config):
    """WindGatedAttentionModel in point mode."""
    return WindGatedAttentionModel(**model_config)


@pytest.fixture
def gaussian_model(model_config):
    """WindGatedAttentionModel in gaussian mode."""
    cfg = dict(model_config)
    cfg["output_mode"] = "gaussian"
    return WindGatedAttentionModel(**cfg)


@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def synthetic_batch(model_config, batch_size):
    """Synthetic batch of data for model forward pass."""
    torch.manual_seed(42)
    B = batch_size
    S = model_config["n_stations"]
    F_s = model_config["n_station_features"]
    F_m = model_config["n_metadata_features"]
    F_g = model_config["n_global_features"]

    return {
        "station_features": torch.randn(B, S, F_s),
        "station_metadata": torch.randn(B, S, F_m),
        "global_context": torch.randn(B, F_g),
        "station_bearings": torch.rand(B, S) * 2 * math.pi,
        "wind_direction": torch.rand(B) * 2 * math.pi,
        "station_mask": torch.ones(B, S),
    }


@pytest.fixture
def tmp_dir():
    """Temporary directory for outputs."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# WindGatedAttentionModel Tests
# ===========================================================================

class TestWindGatedAttentionModel:
    """Tests for the WindGatedAttentionModel class."""

    def test_instantiation_point(self, model_config):
        """Model can be instantiated in point mode."""
        model = WindGatedAttentionModel(**model_config)
        assert model.output_mode == "point"
        assert model.n_stations == 14

    def test_instantiation_gaussian(self, model_config):
        """Model can be instantiated in gaussian mode."""
        cfg = dict(model_config)
        cfg["output_mode"] = "gaussian"
        model = WindGatedAttentionModel(**cfg)
        assert model.output_mode == "gaussian"

    def test_invalid_output_mode(self, model_config):
        """Model raises ValueError for invalid output mode."""
        cfg = dict(model_config)
        cfg["output_mode"] = "invalid"
        with pytest.raises(ValueError, match="output_mode"):
            WindGatedAttentionModel(**cfg)

    def test_forward_point_shapes(self, point_model, synthetic_batch,
                                  batch_size, model_config):
        """Point mode forward pass produces correct output shapes."""
        output = point_model(**synthetic_batch)
        assert output["prediction"].shape == (batch_size, 1)
        assert output["attention_weights"].shape == (
            batch_size, model_config["n_stations"]
        )

    def test_forward_gaussian_shapes(self, gaussian_model, synthetic_batch,
                                     batch_size, model_config):
        """Gaussian mode forward pass produces correct output shapes."""
        output = gaussian_model(**synthetic_batch)
        assert output["prediction"].shape == (batch_size, 1)
        assert output["mu"].shape == (batch_size, 1)
        assert output["log_sigma"].shape == (batch_size, 1)
        assert output["sigma"].shape == (batch_size, 1)
        assert output["attention_weights"].shape == (
            batch_size, model_config["n_stations"]
        )

    def test_gaussian_sigma_positive(self, gaussian_model, synthetic_batch):
        """Gaussian mode sigma values are always positive."""
        output = gaussian_model(**synthetic_batch)
        assert (output["sigma"] > 0).all()

    def test_attention_weights_sum_to_one(self, point_model, synthetic_batch):
        """Attention weights sum to 1 across stations."""
        output = point_model(**synthetic_batch)
        weight_sums = output["attention_weights"].sum(dim=1)
        assert torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-5,
        )

    def test_attention_weights_non_negative(self, point_model, synthetic_batch):
        """Attention weights are non-negative."""
        output = point_model(**synthetic_batch)
        assert (output["attention_weights"] >= 0).all()

    def test_masked_stations_zero_weight(self, point_model, synthetic_batch):
        """Masked-out stations receive zero attention weight."""
        batch = dict(synthetic_batch)
        # Mask out stations 0, 3, 7
        batch["station_mask"] = synthetic_batch["station_mask"].clone()
        batch["station_mask"][:, 0] = 0.0
        batch["station_mask"][:, 3] = 0.0
        batch["station_mask"][:, 7] = 0.0

        output = point_model(**batch)
        weights = output["attention_weights"]

        assert torch.allclose(
            weights[:, 0], torch.zeros(weights.size(0)), atol=1e-6
        )
        assert torch.allclose(
            weights[:, 3], torch.zeros(weights.size(0)), atol=1e-6
        )
        assert torch.allclose(
            weights[:, 7], torch.zeros(weights.size(0)), atol=1e-6
        )
        # Remaining weights should still sum to ~1
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), atol=1e-5
        )

    def test_all_stations_masked_no_nan(self, point_model, model_config):
        """Model handles the edge case where all stations are masked."""
        torch.manual_seed(0)
        B = 4
        S = model_config["n_stations"]
        batch = {
            "station_features": torch.randn(B, S, model_config["n_station_features"]),
            "station_metadata": torch.randn(B, S, model_config["n_metadata_features"]),
            "global_context": torch.randn(B, model_config["n_global_features"]),
            "station_bearings": torch.rand(B, S) * 2 * math.pi,
            "wind_direction": torch.rand(B) * 2 * math.pi,
            "station_mask": torch.zeros(B, S),  # all masked
        }
        output = point_model(**batch)
        assert not torch.isnan(output["prediction"]).any()
        assert not torch.isinf(output["prediction"]).any()

    def test_wind_bias_effect(self, model_config):
        """Wind bias causes upwind stations to get higher attention.

        When wind_alpha is large and wind direction aligns with a
        station bearing, that station should get more attention.
        """
        torch.manual_seed(42)
        cfg = dict(model_config)
        cfg["n_stations"] = 4
        cfg["dropout"] = 0.0
        model = WindGatedAttentionModel(**cfg)

        # Set wind_alpha to a large value to amplify the effect
        with torch.no_grad():
            model.wind_alpha.fill_(10.0)

        B = 1
        S = 4
        sf = torch.randn(B, S, cfg["n_station_features"])
        sm = torch.zeros(B, S, cfg["n_metadata_features"])
        gc = torch.randn(B, cfg["n_global_features"])

        # Station bearings: [0, pi/2, pi, 3pi/2]
        bearings = torch.tensor([[0.0, math.pi / 2, math.pi, 3 * math.pi / 2]])
        # Wind direction: 0 (same as station 0's bearing)
        wd = torch.tensor([0.0])
        mask = torch.ones(B, S)

        output = model(sf, sm, gc, bearings, wd, mask)
        weights = output["attention_weights"][0]

        # Station 0 (bearing=0, wind=0) should have the highest weight
        # because cos(0 - 0) = 1 (max bias)
        # Station 2 (bearing=pi, wind=0) should have the lowest weight
        # because cos(0 - pi) = -1 (min bias)
        assert weights[0] > weights[2], (
            f"Upwind station weight ({weights[0]:.4f}) should exceed "
            f"downwind station weight ({weights[2]:.4f})"
        )

    def test_gradient_flow(self, point_model, synthetic_batch):
        """All model parameters receive gradients during backprop."""
        output = point_model(**synthetic_batch)
        loss = output["prediction"].sum()
        loss.backward()

        for name, param in point_model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient"
            )

    def test_gradient_flow_gaussian(self, gaussian_model, synthetic_batch):
        """Gradient flow works in gaussian mode through mu and sigma."""
        output = gaussian_model(**synthetic_batch)
        target = torch.randn(synthetic_batch["station_features"].size(0), 1)
        crps_fn = GaussianCRPSLoss()
        loss = crps_fn(output["mu"], output["sigma"], target)
        loss.backward()

        for name, param in gaussian_model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )

    def test_batch_size_one(self, point_model, model_config):
        """Model works with batch_size=1."""
        torch.manual_seed(0)
        S = model_config["n_stations"]
        batch = {
            "station_features": torch.randn(1, S, model_config["n_station_features"]),
            "station_metadata": torch.randn(1, S, model_config["n_metadata_features"]),
            "global_context": torch.randn(1, model_config["n_global_features"]),
            "station_bearings": torch.rand(1, S) * 2 * math.pi,
            "wind_direction": torch.rand(1) * 2 * math.pi,
            "station_mask": torch.ones(1, S),
        }
        output = point_model(**batch)
        assert output["prediction"].shape == (1, 1)

    def test_model_name_property(self, point_model, gaussian_model):
        """Model name property returns a descriptive string."""
        assert "WindGatedAttn" in point_model.name
        assert "point" in point_model.name
        assert "gaussian" in gaussian_model.name

    def test_finite_outputs(self, point_model, synthetic_batch):
        """Outputs are finite (no NaN or Inf)."""
        output = point_model(**synthetic_batch)
        assert torch.isfinite(output["prediction"]).all()
        assert torch.isfinite(output["attention_weights"]).all()

    def test_parameter_count(self, point_model):
        """Model has a reasonable number of parameters."""
        n_params = sum(p.numel() for p in point_model.parameters())
        assert n_params > 0
        # Should be manageable (< 100K for this config)
        assert n_params < 100_000

    def test_wind_alpha_learnable(self, point_model):
        """wind_alpha is a learnable parameter."""
        assert point_model.wind_alpha.requires_grad
        assert point_model.wind_alpha.item() == pytest.approx(1.0, abs=0.01)

    def test_variable_station_count(self, model_config):
        """Model works with different numbers of stations."""
        for n_stations in [3, 5, 10, 20]:
            cfg = dict(model_config)
            cfg["n_stations"] = n_stations
            model = WindGatedAttentionModel(**cfg)
            B = 4
            batch = {
                "station_features": torch.randn(B, n_stations, cfg["n_station_features"]),
                "station_metadata": torch.randn(B, n_stations, cfg["n_metadata_features"]),
                "global_context": torch.randn(B, cfg["n_global_features"]),
                "station_bearings": torch.rand(B, n_stations) * 2 * math.pi,
                "wind_direction": torch.rand(B) * 2 * math.pi,
                "station_mask": torch.ones(B, n_stations),
            }
            output = model(**batch)
            assert output["prediction"].shape == (B, 1)

    def test_partial_masking(self, point_model, model_config):
        """Model handles partial station masking (some samples have more
        stations masked than others)."""
        torch.manual_seed(0)
        B = 4
        S = model_config["n_stations"]
        mask = torch.ones(B, S)
        # Sample 0: 3 stations masked
        mask[0, :3] = 0.0
        # Sample 1: all present
        # Sample 2: only 1 present
        mask[2, 1:] = 0.0
        # Sample 3: half masked
        mask[3, ::2] = 0.0

        batch = {
            "station_features": torch.randn(B, S, model_config["n_station_features"]),
            "station_metadata": torch.randn(B, S, model_config["n_metadata_features"]),
            "global_context": torch.randn(B, model_config["n_global_features"]),
            "station_bearings": torch.rand(B, S) * 2 * math.pi,
            "wind_direction": torch.rand(B) * 2 * math.pi,
            "station_mask": mask,
        }
        output = point_model(**batch)
        assert torch.isfinite(output["prediction"]).all()

        # Sample 2 should have all weight on station 0
        w2 = output["attention_weights"][2]
        assert w2[0].item() == pytest.approx(1.0, abs=1e-5)
        assert w2[1:].sum().item() == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# CRPS Loss Tests
# ===========================================================================

class TestGaussianCRPSLoss:
    """Tests for the Gaussian CRPS loss function."""

    def test_perfect_prediction_zero_crps(self):
        """CRPS should be very small when mu = target and sigma is small."""
        loss_fn = GaussianCRPSLoss()
        mu = torch.tensor([70.0])
        sigma = torch.tensor([0.001])
        target = torch.tensor([70.0])
        crps = loss_fn(mu, sigma, target)
        # With sigma -> 0 and mu = target, CRPS -> 0
        assert crps.item() < 0.01

    def test_crps_positive(self):
        """CRPS is always non-negative."""
        loss_fn = GaussianCRPSLoss()
        torch.manual_seed(42)
        for _ in range(20):
            mu = torch.randn(10) * 10 + 60
            sigma = torch.rand(10) * 5 + 0.1
            target = torch.randn(10) * 10 + 60
            crps = loss_fn(mu, sigma, target)
            assert crps.item() >= 0, f"CRPS should be >= 0, got {crps.item()}"

    def test_crps_increases_with_error(self):
        """CRPS increases as the prediction error increases."""
        loss_fn = GaussianCRPSLoss()
        sigma = torch.tensor([2.0])
        target = torch.tensor([70.0])

        crps_good = loss_fn(torch.tensor([70.0]), sigma, target)
        crps_bad = loss_fn(torch.tensor([80.0]), sigma, target)
        assert crps_bad > crps_good

    def test_crps_increases_with_sigma(self):
        """CRPS increases with larger sigma (less confident prediction)."""
        loss_fn = GaussianCRPSLoss()
        mu = torch.tensor([70.0])
        target = torch.tensor([70.0])

        crps_sharp = loss_fn(mu, torch.tensor([1.0]), target)
        crps_wide = loss_fn(mu, torch.tensor([10.0]), target)
        assert crps_wide > crps_sharp

    def test_crps_known_value(self):
        """CRPS matches hand-computed value for z=0 (mu=target).

        When z=0: CRPS = sigma * (2*phi(0) - 1/sqrt(pi))
                       = sigma * (2*(1/sqrt(2pi)) - 1/sqrt(pi))
                       = sigma * (sqrt(2/pi) - 1/sqrt(pi))
                       = sigma * (1/sqrt(pi)) * (sqrt(2) - 1)
        """
        loss_fn = GaussianCRPSLoss()
        sigma_val = 3.0
        mu = torch.tensor([0.0])
        sigma = torch.tensor([sigma_val])
        target = torch.tensor([0.0])

        crps = loss_fn(mu, sigma, target)
        # z=0: phi(0) = 1/sqrt(2*pi), Phi(0) = 0.5
        # CRPS = sigma * [0*(2*0.5-1) + 2*phi(0) - 1/sqrt(pi)]
        #      = sigma * [0 + 2/sqrt(2*pi) - 1/sqrt(pi)]
        #      = sigma * [sqrt(2/pi) - 1/sqrt(pi)]
        expected = sigma_val * (
            math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
        )
        assert crps.item() == pytest.approx(expected, rel=1e-4)

    def test_crps_gradient_flow(self):
        """Gradients flow through CRPS loss."""
        loss_fn = GaussianCRPSLoss()
        mu = torch.tensor([70.0], requires_grad=True)
        sigma = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor([72.0])

        crps = loss_fn(mu, sigma, target)
        crps.backward()

        assert mu.grad is not None
        assert sigma.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(sigma.grad).all()

    def test_crps_reduction_none(self):
        """CRPS with reduction='none' returns per-sample values."""
        loss_fn = GaussianCRPSLoss(reduction="none")
        mu = torch.tensor([70.0, 65.0, 80.0])
        sigma = torch.tensor([2.0, 3.0, 1.0])
        target = torch.tensor([71.0, 63.0, 79.0])

        crps = loss_fn(mu, sigma, target)
        assert crps.shape == (3,)

    def test_crps_batch_consistency(self):
        """CRPS for a batch is the mean of individual CRPS values."""
        loss_fn_mean = GaussianCRPSLoss(reduction="mean")
        loss_fn_none = GaussianCRPSLoss(reduction="none")

        mu = torch.tensor([70.0, 65.0, 80.0])
        sigma = torch.tensor([2.0, 3.0, 1.0])
        target = torch.tensor([71.0, 63.0, 79.0])

        crps_mean = loss_fn_mean(mu, sigma, target)
        crps_none = loss_fn_none(mu, sigma, target)
        assert crps_mean.item() == pytest.approx(
            crps_none.mean().item(), rel=1e-5
        )

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            GaussianCRPSLoss(reduction="sum")


# ===========================================================================
# EnergyCRPSLoss Tests
# ===========================================================================

class TestEnergyCRPSLoss:
    """Tests for the energy-score CRPS approximation."""

    def test_energy_crps_positive(self):
        """Energy CRPS is non-negative."""
        loss_fn = EnergyCRPSLoss(n_samples=200)
        torch.manual_seed(42)
        mu = torch.tensor([70.0, 65.0])
        sigma = torch.tensor([2.0, 3.0])
        target = torch.tensor([71.0, 63.0])

        crps = loss_fn(mu, sigma, target)
        assert crps.item() >= 0

    def test_energy_crps_gradient_flow(self):
        """Gradients flow through energy CRPS."""
        loss_fn = EnergyCRPSLoss(n_samples=50)
        mu = torch.tensor([70.0], requires_grad=True)
        sigma = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor([72.0])

        crps = loss_fn(mu, sigma, target)
        crps.backward()

        assert mu.grad is not None
        assert sigma.grad is not None


# ===========================================================================
# PinballLoss Tests
# ===========================================================================

class TestPinballLoss:
    """Tests for the pinball (quantile) loss."""

    def test_pinball_median(self):
        """Pinball loss at tau=0.5 equals 0.5 * |y - q|."""
        loss_fn = PinballLoss(quantiles=[0.5])
        predictions = torch.tensor([[70.0]])
        target = torch.tensor([72.0])

        loss = loss_fn(predictions, target)
        expected = 0.5 * abs(72.0 - 70.0)
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_pinball_positive(self):
        """Pinball loss is non-negative."""
        loss_fn = PinballLoss()
        torch.manual_seed(42)
        predictions = torch.randn(16, 3) * 10 + 60
        target = torch.randn(16) * 10 + 60

        loss = loss_fn(predictions, target)
        assert loss.item() >= 0

    def test_pinball_gradient_flow(self):
        """Gradients flow through pinball loss."""
        loss_fn = PinballLoss(quantiles=[0.025, 0.5, 0.975])
        predictions = torch.randn(8, 3, requires_grad=True)
        target = torch.randn(8)

        loss = loss_fn(predictions, target)
        loss.backward()

        assert predictions.grad is not None
        assert torch.isfinite(predictions.grad).all()

    def test_pinball_asymmetry(self):
        """Pinball loss penalises differently above/below the quantile."""
        loss_fn_low = PinballLoss(quantiles=[0.1])
        loss_fn_high = PinballLoss(quantiles=[0.9])

        # Over-prediction (pred > target)
        pred_over = torch.tensor([[75.0]])
        target = torch.tensor([70.0])

        loss_low_over = loss_fn_low(pred_over, target)
        loss_high_over = loss_fn_high(pred_over, target)

        # For tau=0.1: over-predicting is heavily penalised
        # For tau=0.9: over-predicting is lightly penalised
        assert loss_low_over > loss_high_over


# ===========================================================================
# CombinedCRPSMAELoss Tests
# ===========================================================================

class TestCombinedCRPSMAELoss:
    """Tests for the combined CRPS + MAE loss."""

    def test_combined_loss_returns_dict(self):
        """Combined loss returns dict with loss, crps, mae keys."""
        loss_fn = CombinedCRPSMAELoss()
        mu = torch.tensor([70.0])
        sigma = torch.tensor([2.0])
        target = torch.tensor([72.0])

        result = loss_fn(mu, sigma, target)
        assert "loss" in result
        assert "crps" in result
        assert "mae" in result

    def test_combined_loss_weighting(self):
        """Combined loss is a weighted sum of CRPS and MAE."""
        crps_w, mae_w = 0.6, 0.4
        loss_fn = CombinedCRPSMAELoss(crps_weight=crps_w, mae_weight=mae_w)
        crps_alone = GaussianCRPSLoss()

        mu = torch.tensor([70.0])
        sigma = torch.tensor([2.0])
        target = torch.tensor([72.0])

        result = loss_fn(mu, sigma, target)
        crps_val = crps_alone(mu, sigma, target)
        mae_val = torch.abs(mu - target).mean()

        expected = crps_w * crps_val + mae_w * mae_val
        assert result["loss"].item() == pytest.approx(
            expected.item(), rel=1e-4
        )

    def test_combined_loss_gradient_flow(self):
        """Gradients flow through the combined loss."""
        loss_fn = CombinedCRPSMAELoss()
        mu = torch.tensor([70.0], requires_grad=True)
        sigma = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor([72.0])

        result = loss_fn(mu, sigma, target)
        result["loss"].backward()

        assert mu.grad is not None
        assert sigma.grad is not None


# ===========================================================================
# Standard Normal PDF/CDF Tests
# ===========================================================================

class TestNormalFunctions:
    """Tests for the standard normal PDF and CDF helpers."""

    def test_pdf_at_zero(self):
        """Standard normal PDF at z=0 is 1/sqrt(2*pi)."""
        z = torch.tensor([0.0])
        result = _standard_normal_pdf(z)
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert result.item() == pytest.approx(expected, rel=1e-6)

    def test_cdf_at_zero(self):
        """Standard normal CDF at z=0 is 0.5."""
        z = torch.tensor([0.0])
        result = _standard_normal_cdf(z)
        assert result.item() == pytest.approx(0.5, rel=1e-6)

    def test_cdf_monotonic(self):
        """CDF is monotonically increasing."""
        z = torch.linspace(-4, 4, 100)
        cdf_vals = _standard_normal_cdf(z)
        diffs = cdf_vals[1:] - cdf_vals[:-1]
        assert (diffs > 0).all()


# ===========================================================================
# AttentionDataset Tests
# ===========================================================================

class TestAttentionDataset:
    """Tests for the custom AttentionDataset."""

    def test_dataset_length(self):
        """Dataset reports correct length."""
        N = 50
        ds = AttentionDataset(
            station_features=np.random.randn(N, 5, 3).astype(np.float32),
            station_metadata=np.random.randn(N, 5, 2).astype(np.float32),
            global_context=np.random.randn(N, 4).astype(np.float32),
            station_bearings=np.random.randn(N, 5).astype(np.float32),
            wind_direction=np.random.randn(N).astype(np.float32),
            station_mask=np.ones((N, 5), dtype=np.float32),
            targets=np.random.randn(N).astype(np.float32),
        )
        assert len(ds) == N

    def test_dataset_item_shapes(self):
        """Each item from the dataset has correct shapes."""
        N, S, F_s, F_m, F_g = 10, 5, 3, 2, 4
        ds = AttentionDataset(
            station_features=np.random.randn(N, S, F_s).astype(np.float32),
            station_metadata=np.random.randn(N, S, F_m).astype(np.float32),
            global_context=np.random.randn(N, F_g).astype(np.float32),
            station_bearings=np.random.randn(N, S).astype(np.float32),
            wind_direction=np.random.randn(N).astype(np.float32),
            station_mask=np.ones((N, S), dtype=np.float32),
            targets=np.random.randn(N).astype(np.float32),
        )
        item = ds[0]
        assert item["station_features"].shape == (S, F_s)
        assert item["station_metadata"].shape == (S, F_m)
        assert item["global_context"].shape == (F_g,)
        assert item["station_bearings"].shape == (S,)
        assert item["wind_direction"].shape == ()
        assert item["station_mask"].shape == (S,)
        assert item["target"].shape == ()


# ===========================================================================
# Training Pipeline Tests
# ===========================================================================

class TestTrainingPipeline:
    """Smoke tests for the training pipeline functions."""

    def _make_synthetic_data(self, n_samples, n_stations, n_station_feats,
                             n_meta_feats, n_global_feats):
        """Create synthetic training data."""
        np.random.seed(42)
        return {
            "attention": {
                "station_features": np.random.randn(
                    n_samples, n_stations, n_station_feats
                ).astype(np.float32),
                "station_metadata": np.random.randn(
                    n_samples, n_stations, n_meta_feats
                ).astype(np.float32),
                "global_context": np.random.randn(
                    n_samples, n_global_feats
                ).astype(np.float32),
                "station_bearings": (
                    np.random.rand(n_samples, n_stations) * 2 * math.pi
                ).astype(np.float32),
                "wind_direction": (
                    np.random.rand(n_samples) * 2 * math.pi
                ).astype(np.float32),
                "station_mask": np.ones(
                    (n_samples, n_stations), dtype=np.float32
                ),
            },
            "targets": np.random.randn(n_samples).astype(np.float32),
        }

    def test_train_one_epoch_point(self, model_config, tmp_dir):
        """One epoch of training runs without error (point mode)."""
        S = model_config["n_stations"]
        train_data = self._make_synthetic_data(
            64, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )
        val_data = self._make_synthetic_data(
            16, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )

        model = WindGatedAttentionModel(**model_config)
        train_loader, val_loader = create_attention_dataloaders(
            train_data["attention"], val_data["attention"],
            train_data["targets"], val_data["targets"],
            batch_size=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.SmoothL1Loss()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, "point"
        )
        assert math.isfinite(train_loss)
        assert train_loss >= 0

    def test_validate_point(self, model_config):
        """Validation runs without error and produces correct shapes."""
        S = model_config["n_stations"]
        val_data = self._make_synthetic_data(
            32, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )

        model = WindGatedAttentionModel(**model_config)
        _, val_loader = create_attention_dataloaders(
            val_data["attention"], val_data["attention"],
            val_data["targets"], val_data["targets"],
            batch_size=16,
        )
        loss_fn = nn.SmoothL1Loss()

        val_loss, preds, actuals, sigmas = validate(
            model, val_loader, loss_fn, "point"
        )
        assert math.isfinite(val_loss)
        assert len(preds) == 32
        assert len(actuals) == 32
        assert sigmas is None

    def test_validate_gaussian(self, model_config):
        """Validation in gaussian mode returns sigma values."""
        cfg = dict(model_config)
        cfg["output_mode"] = "gaussian"
        S = cfg["n_stations"]
        val_data = self._make_synthetic_data(
            32, S, cfg["n_station_features"],
            cfg["n_metadata_features"],
            cfg["n_global_features"],
        )

        model = WindGatedAttentionModel(**cfg)
        _, val_loader = create_attention_dataloaders(
            val_data["attention"], val_data["attention"],
            val_data["targets"], val_data["targets"],
            batch_size=16,
        )
        loss_fn = GaussianCRPSLoss()

        val_loss, preds, actuals, sigmas = validate(
            model, val_loader, loss_fn, "gaussian"
        )
        assert math.isfinite(val_loss)
        assert sigmas is not None
        assert len(sigmas) == 32

    def test_full_training_smoke(self, model_config, tmp_dir):
        """Full training loop runs for a few epochs on synthetic data."""
        S = model_config["n_stations"]
        n_train = 64
        n_val = 16

        train_data = self._make_synthetic_data(
            n_train, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )
        val_data = self._make_synthetic_data(
            n_val, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )

        training_config = {
            "learning_rate": 0.01,
            "max_epochs": 3,
            "early_stopping_patience": 5,
            "batch_size": 16,
            "loss_type": "huber",
            "target_type": "raw",
            "device": "cpu",
            "model_name": "smoke_test",
        }

        result = train_wind_gated_model(
            train_data=train_data,
            val_data=val_data,
            model_config=model_config,
            training_config=training_config,
            output_dir=tmp_dir,
        )

        assert "model" in result
        assert "history" in result
        assert "best_epoch" in result
        assert "best_val_mae" in result
        assert len(result["history"]) == 3
        assert result["best_val_mae"] >= 0

    def test_full_training_gaussian_smoke(self, model_config, tmp_dir):
        """Full training loop with gaussian mode and CRPS loss."""
        cfg = dict(model_config)
        cfg["output_mode"] = "gaussian"
        S = cfg["n_stations"]

        train_data = self._make_synthetic_data(
            64, S, cfg["n_station_features"],
            cfg["n_metadata_features"],
            cfg["n_global_features"],
        )
        val_data = self._make_synthetic_data(
            16, S, cfg["n_station_features"],
            cfg["n_metadata_features"],
            cfg["n_global_features"],
        )

        training_config = {
            "learning_rate": 0.01,
            "max_epochs": 2,
            "early_stopping_patience": 5,
            "batch_size": 16,
            "loss_type": "combined_crps_mae",
            "target_type": "raw",
            "device": "cpu",
            "model_name": "gaussian_smoke",
        }

        result = train_wind_gated_model(
            train_data=train_data,
            val_data=val_data,
            model_config=cfg,
            training_config=training_config,
            output_dir=tmp_dir,
        )

        assert "model" in result
        assert len(result["history"]) == 2
        # Check that mean_sigma is tracked
        assert "mean_sigma" in result["history"][0]

    def test_model_save_load_roundtrip(self, model_config, tmp_dir):
        """Model can be saved and loaded with identical parameters."""
        model = WindGatedAttentionModel(**model_config)

        # Save
        save_path = os.path.join(tmp_dir, "test_model.pt")
        torch.save(model.state_dict(), save_path)

        # Load into new model
        model2 = WindGatedAttentionModel(**model_config)
        model2.load_state_dict(
            torch.load(save_path, weights_only=True)
        )

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs after load"

    def test_save_training_history(self, tmp_dir):
        """Training history can be saved to CSV."""
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_mae": 4.0, "lr": 0.001},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "val_mae": 3.5, "lr": 0.001},
        ]
        path = os.path.join(tmp_dir, "history.csv")
        save_training_history(history, path)
        assert os.path.isfile(path)

    def test_plot_training_curves(self, tmp_dir):
        """Training curves plot can be generated."""
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_mae": 4.0, "lr": 0.001},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "val_mae": 3.5, "lr": 0.001},
            {"epoch": 3, "train_loss": 0.3, "val_loss": 0.4, "val_mae": 3.0, "lr": 0.001},
        ]
        path = os.path.join(tmp_dir, "curves.png")
        plot_training_curves(history, path)
        assert os.path.isfile(path)

    def test_early_stopping(self, model_config, tmp_dir):
        """Training stops early when validation MAE stops improving."""
        S = model_config["n_stations"]

        train_data = self._make_synthetic_data(
            32, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )
        val_data = self._make_synthetic_data(
            16, S, model_config["n_station_features"],
            model_config["n_metadata_features"],
            model_config["n_global_features"],
        )

        training_config = {
            "learning_rate": 0.0,  # Zero LR -> no weight updates -> no improvement
            "max_epochs": 50,
            "early_stopping_patience": 3,
            "batch_size": 16,
            "loss_type": "mse",
            "target_type": "raw",
            "device": "cpu",
            "model_name": "early_stop_test",
        }

        result = train_wind_gated_model(
            train_data=train_data,
            val_data=val_data,
            model_config=model_config,
            training_config=training_config,
            output_dir=tmp_dir,
        )

        # Should stop well before 50 epochs
        n_epochs = len(result["history"])
        assert n_epochs < 50, (
            f"Expected early stopping, but ran {n_epochs} epochs"
        )
