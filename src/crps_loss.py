"""
CRPS (Continuous Ranked Probability Score) Loss Functions.

Provides differentiable loss functions for training probabilistic
weather forecast models:

  1. GaussianCRPSLoss   -- Closed-form CRPS for heteroscedastic
                           Gaussian (mu, sigma) outputs.
  2. EnergyCRPSLoss     -- Sample-based energy-score CRPS approximation
                           for non-Gaussian predictive distributions.
  3. PinballLoss        -- Quantile (pinball) loss for quantile
                           regression fallback.
  4. CombinedCRPSMAELoss -- Weighted sum of Gaussian CRPS and point MAE
                           for training stability.

The Gaussian CRPS formula (Gneiting & Raftery, 2007):
  CRPS(mu, sigma, y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
  where z = (y - mu) / sigma, Phi = standard normal CDF, phi = PDF.

Lower CRPS is better (0 = perfect deterministic forecast).
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
_SQRT_PI = math.sqrt(math.pi)
_INV_SQRT_PI = 1.0 / _SQRT_PI
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


# ===========================================================================
# Helper: Standard Normal PDF & CDF
# ===========================================================================

def _standard_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """Compute the standard normal probability density function.

    Parameters
    ----------
    z : torch.Tensor
        Input values.

    Returns
    -------
    torch.Tensor
        phi(z) = exp(-z^2/2) / sqrt(2*pi).
    """
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Compute the standard normal cumulative distribution function.

    Uses the error function for numerical stability.

    Parameters
    ----------
    z : torch.Tensor
        Input values.

    Returns
    -------
    torch.Tensor
        Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    """
    return 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))


# ===========================================================================
# 1. Gaussian CRPS Loss
# ===========================================================================

class GaussianCRPSLoss(nn.Module):
    """Closed-form CRPS for Gaussian predictive distributions.

    Given predicted mean ``mu``, predicted standard deviation ``sigma``,
    and observed target ``y``, the Gaussian CRPS is:

        CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]

    where ``z = (y - mu) / sigma``.

    Parameters
    ----------
    reduction : str
        ``"mean"`` (default) or ``"none"``.

    Examples
    --------
    >>> loss_fn = GaussianCRPSLoss()
    >>> mu = torch.tensor([70.0, 65.0])
    >>> sigma = torch.tensor([2.0, 3.0])
    >>> target = torch.tensor([71.0, 63.0])
    >>> loss = loss_fn(mu, sigma, target)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"reduction must be 'mean' or 'none', got '{reduction}'"
            )
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian CRPS.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted mean. Shape ``(batch,)`` or ``(batch, 1)``.
        sigma : torch.Tensor
            Predicted standard deviation (must be positive).
            Shape ``(batch,)`` or ``(batch, 1)``.
        target : torch.Tensor
            Observed values. Shape ``(batch,)`` or ``(batch, 1)``.

        Returns
        -------
        torch.Tensor
            Scalar (if reduction='mean') or per-sample CRPS
            (if reduction='none').
        """
        # Flatten to 1-D for computation
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        target = target.reshape(-1)

        # Clamp sigma to avoid division by zero
        sigma = sigma.clamp(min=1e-6)

        # Standardised residual
        z = (target - mu) / sigma

        # CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
        phi_z = _standard_normal_pdf(z)
        Phi_z = _standard_normal_cdf(z)

        crps = sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - _INV_SQRT_PI)

        if self.reduction == "mean":
            return crps.mean()
        return crps


# ===========================================================================
# 2. Energy Score CRPS Approximation
# ===========================================================================

class EnergyCRPSLoss(nn.Module):
    """Sample-based energy score CRPS approximation.

    Approximates CRPS by drawing samples from the predictive
    distribution and computing the energy score:

        CRPS ~= E[|X - y|] - 0.5 * E[|X - X'|]

    where X, X' are independent samples from the predictive
    distribution.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples to draw.
    reduction : str
        ``"mean"`` or ``"none"``.

    Notes
    -----
    This is useful when the predictive distribution is not Gaussian.
    For Gaussian outputs, prefer ``GaussianCRPSLoss`` (exact,
    gradient-friendly).
    """

    def __init__(self, n_samples: int = 100, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"reduction must be 'mean' or 'none', got '{reduction}'"
            )
        self.n_samples = n_samples
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy-score CRPS approximation.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted mean. Shape ``(batch,)`` or ``(batch, 1)``.
        sigma : torch.Tensor
            Predicted std. Shape ``(batch,)`` or ``(batch, 1)``.
        target : torch.Tensor
            Observed values. Shape ``(batch,)`` or ``(batch, 1)``.

        Returns
        -------
        torch.Tensor
            CRPS approximation.
        """
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1).clamp(min=1e-6)
        target = target.reshape(-1)

        # Draw samples: (n_samples, batch)
        eps = torch.randn(
            self.n_samples, mu.size(0),
            device=mu.device, dtype=mu.dtype,
        )
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B)

        # E[|X - y|]
        term1 = torch.abs(samples - target.unsqueeze(0)).mean(dim=0)  # (B,)

        # E[|X - X'|] -- use half the samples for X, half for X'
        half = self.n_samples // 2
        x1 = samples[:half]  # (S/2, B)
        x2 = samples[half:2 * half]  # (S/2, B)
        term2 = torch.abs(x1 - x2).mean(dim=0)  # (B,)

        crps = term1 - 0.5 * term2  # (B,)

        if self.reduction == "mean":
            return crps.mean()
        return crps


# ===========================================================================
# 3. Pinball (Quantile) Loss
# ===========================================================================

class PinballLoss(nn.Module):
    """Pinball (quantile) loss for quantile regression.

    For a quantile level tau:
        L(y, q, tau) = max(tau*(y - q), (tau - 1)*(y - q))
                     = (y - q) * (tau - 1_{y < q})

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels (e.g. [0.025, 0.5, 0.975] for 95% PI).
    reduction : str
        ``"mean"`` or ``"none"``.

    Examples
    --------
    >>> loss_fn = PinballLoss(quantiles=[0.025, 0.5, 0.975])
    >>> predictions = torch.randn(16, 3)  # 3 quantiles
    >>> target = torch.randn(16)
    >>> loss = loss_fn(predictions, target)
    """

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.025, 0.50, 0.975]
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"reduction must be 'mean' or 'none', got '{reduction}'"
            )
        self.quantiles = quantiles
        self.reduction = reduction
        # Register as buffer so they move with the model
        self.register_buffer(
            "tau",
            torch.tensor(quantiles, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pinball loss across all quantiles.

        Parameters
        ----------
        predictions : torch.Tensor
            Shape ``(batch, n_quantiles)`` -- predicted quantile values.
        target : torch.Tensor
            Shape ``(batch,)`` or ``(batch, 1)`` -- observed values.

        Returns
        -------
        torch.Tensor
            Averaged pinball loss across quantiles and samples.
        """
        target = target.reshape(-1, 1)  # (B, 1)
        # predictions: (B, Q), target: (B, 1) -> broadcast
        residual = target - predictions  # (B, Q)
        tau = self.tau.unsqueeze(0)  # (1, Q)

        loss = torch.max(tau * residual, (tau - 1.0) * residual)  # (B, Q)

        if self.reduction == "mean":
            return loss.mean()
        return loss.mean(dim=1)  # average over quantiles, per sample


# ===========================================================================
# 4. Combined CRPS + MAE Loss
# ===========================================================================

class CombinedCRPSMAELoss(nn.Module):
    """Weighted combination of Gaussian CRPS and point MAE.

    Combining CRPS with a point-prediction MAE term helps training
    stability: the MAE anchors the mean prediction while the CRPS
    shapes the uncertainty estimate.

    Total loss = crps_weight * CRPS + mae_weight * MAE(mu, target)

    Parameters
    ----------
    crps_weight : float
        Weight for the CRPS component.
    mae_weight : float
        Weight for the MAE component.

    Examples
    --------
    >>> loss_fn = CombinedCRPSMAELoss(crps_weight=0.7, mae_weight=0.3)
    >>> mu = torch.tensor([70.0])
    >>> sigma = torch.tensor([2.0])
    >>> target = torch.tensor([71.0])
    >>> result = loss_fn(mu, sigma, target)
    >>> result["loss"]  # total weighted loss
    """

    def __init__(
        self,
        crps_weight: float = 0.7,
        mae_weight: float = 0.3,
    ):
        super().__init__()
        self.crps_weight = crps_weight
        self.mae_weight = mae_weight
        self.crps_loss = GaussianCRPSLoss(reduction="mean")

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted mean.
        sigma : torch.Tensor
            Predicted standard deviation.
        target : torch.Tensor
            Observed values.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"loss"`` (total), ``"crps"`` (CRPS component),
            ``"mae"`` (MAE component).
        """
        crps = self.crps_loss(mu, sigma, target)

        mu_flat = mu.reshape(-1)
        target_flat = target.reshape(-1)
        mae = torch.abs(mu_flat - target_flat).mean()

        total = self.crps_weight * crps + self.mae_weight * mae

        return {
            "loss": total,
            "crps": crps.detach(),
            "mae": mae.detach(),
        }
