"""
PyTorch Neural Network Models for NYC Temperature Prediction.

Provides the feedforward neural network architecture used to predict
NYC's daily maximum temperature (TMAX) from surrounding weather-station
observations at lag t-1.

Classes:
  - TempPredictorV1 -- configurable feedforward network with ReLU/Dropout

Factory / helper functions:
  - create_model()       -- build a TempPredictorV1 with config defaults
  - count_parameters()   -- count trainable parameters
  - get_model_summary()  -- human-readable architecture string
"""

import os
import sys
import logging

import torch
import torch.nn as nn

# Add project root to path so config is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Feedforward Neural Network — V1
# ===========================================================================

class TempPredictorV1(nn.Module):
    """Feedforward neural network for daily maximum temperature prediction.

    Architecture
    ------------
    Input(n_features)
      -> [Hidden(h_i), ReLU, Dropout(p)]  x  len(hidden_sizes)
      -> Output(1)   (linear, no activation)

    The network learns to map scaled input features (lagged TMAX/TMIN
    from surrounding stations plus cyclical date encodings) to the
    unscaled NYC TMAX in degrees Fahrenheit.

    Parameters
    ----------
    n_features : int
        Number of input features.  For the default project configuration
        this is 30 (28 lagged station features + sin_day + cos_day).
    hidden_sizes : list[int], optional
        Widths of the hidden layers.  Defaults to ``config.HIDDEN_SIZES``
        ([64, 32]).
    dropout : float, optional
        Dropout probability applied after each hidden layer's ReLU.
        Defaults to ``config.DROPOUT`` (0.1).  Set to 0.0 to disable.

    Examples
    --------
    >>> model = TempPredictorV1(n_features=30)
    >>> x = torch.randn(16, 30)
    >>> out = model(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float | None = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = list(config.HIDDEN_SIZES)
        if dropout is None:
            dropout = config.DROPOUT

        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout

        # Build sequential stack: Input -> [Linear, ReLU, Dropout] x N -> Output
        layers: list[nn.Module] = []
        in_dim = n_features

        for i, h_dim in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        # Final output layer — single neuron, linear activation
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

        logger.info(
            "TempPredictorV1 created: n_features=%d, hidden_sizes=%s, "
            "dropout=%.2f, total_params=%d",
            n_features,
            hidden_sizes,
            dropout,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(batch_size, 1)``.
        """
        return self.network(x)


# ===========================================================================
# Factory Function
# ===========================================================================

def create_model(
    n_features: int,
    hidden_sizes: list[int] | None = None,
    dropout: float | None = None,
) -> TempPredictorV1:
    """Create a TempPredictorV1 with the specified or default configuration.

    This is the recommended entry point for model creation.  It
    instantiates the model, logs the architecture, and returns it
    ready for training.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list[int], optional
        Hidden layer widths.  Defaults to ``config.HIDDEN_SIZES``.
    dropout : float, optional
        Dropout probability.  Defaults to ``config.DROPOUT``.

    Returns
    -------
    TempPredictorV1
        An initialized (untrained) model instance.
    """
    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )
    logger.info(
        "Model created via create_model(): %d trainable parameters",
        count_parameters(model),
    )
    return model


# ===========================================================================
# Helper Utilities
# ===========================================================================

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.

    Returns
    -------
    int
        Total number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(
    model: nn.Module,
    n_features: int | None = None,
) -> str:
    """Return a human-readable summary of the model architecture.

    Includes layer names, shapes, parameter counts, and (optionally)
    a representative forward-pass output shape.

    Parameters
    ----------
    model : nn.Module
        The model to summarize.
    n_features : int, optional
        If provided, a dummy forward pass is performed to verify the
        output shape and include it in the summary.

    Returns
    -------
    str
        Multi-line string describing the model.
    """
    lines = [
        "=" * 60,
        "Model Summary",
        "=" * 60,
        f"  Class: {model.__class__.__name__}",
    ]

    if hasattr(model, "n_features"):
        lines.append(f"  Input features: {model.n_features}")
    if hasattr(model, "hidden_sizes"):
        lines.append(f"  Hidden sizes: {model.hidden_sizes}")
    if hasattr(model, "dropout_rate"):
        lines.append(f"  Dropout rate: {model.dropout_rate}")

    lines.append("")
    lines.append("  Layers:")

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        lines.append(
            f"    {name:30s}  shape={str(list(param.shape)):15s}  "
            f"params={param.numel():,}"
        )

    lines.append("")
    lines.append(f"  Total trainable parameters: {count_parameters(model):,}")
    lines.append(f"  Total parameters: {total_params:,}")

    # Optional: verify output shape with a dummy forward pass
    if n_features is not None:
        try:
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, n_features)
                out = model(dummy)
            lines.append(f"  Output shape (batch=1): {list(out.shape)}")
        except Exception as e:
            lines.append(f"  Output shape check failed: {e}")

    lines.append("=" * 60)
    return "\n".join(lines)
