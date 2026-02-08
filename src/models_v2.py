"""
Extended Model Architectures for NYC Temperature Prediction (Phase 4).

Provides four new architectures beyond the Phase 3 TempPredictorV1:

  1. EnhancedMLP       -- Flexible feedforward with optional batch-norm,
                          configurable width/depth, and activation choice.
  2. MultiLagMLP       -- MLP that consumes concatenated multi-lag features
                          as a flat input vector.
  3. LSTMPredictor     -- LSTM/GRU sequence model for time-series input.
  4. StationAttentionModel -- Per-station shared encoder with attention
                          pooling across stations.

Utility functions:
  - reshape_for_lstm()       -- reshape flat (batch, n_features) to 3-D
  - reshape_for_attention()  -- reshape flat data to (batch, n_stations, F)
  - get_loss_function()      -- factory for MSE / Huber / MAE loss
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Loss Function Factory
# ===========================================================================

def get_loss_function(loss_type: str = "mse") -> nn.Module:
    """Return a PyTorch loss function by name.

    Parameters
    ----------
    loss_type : str
        One of ``"mse"``, ``"huber"``, ``"mae"`` (case-insensitive).

    Returns
    -------
    nn.Module
        The corresponding loss function.

    Raises
    ------
    ValueError
        If *loss_type* is not recognised.
    """
    loss_type = loss_type.lower().strip()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_type in ("mae", "l1"):
        return nn.L1Loss()
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: mse, huber, mae."
        )


# ===========================================================================
# 1. Enhanced MLP
# ===========================================================================

class EnhancedMLP(nn.Module):
    """Flexible feedforward network with optional batch normalisation.

    Architecture
    ------------
    Input(n_features)
      -> [Linear(h_i), (BatchNorm1d)?, ReLU, Dropout(p)]  x  N
      -> Output(1)

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout probability after each activation.
    use_batch_norm : bool
        If ``True``, insert ``BatchNorm1d`` before each ReLU.

    Examples
    --------
    >>> model = EnhancedMLP(30, hidden_sizes=[128, 64, 32], dropout=0.1)
    >>> x = torch.randn(16, 30)
    >>> model(x).shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Optional[list[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.n_features = n_features
        self.hidden_sizes = list(hidden_sizes)
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        layers: list[nn.Module] = []
        in_dim = n_features

        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        logger.info(
            "EnhancedMLP created: n_features=%d, hidden_sizes=%s, "
            "dropout=%.2f, batch_norm=%s, params=%d",
            n_features, hidden_sizes, dropout, use_batch_norm,
            sum(p.numel() for p in self.parameters()),
        )

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        bn = "+BN" if self.use_batch_norm else ""
        return f"EnhancedMLP{self.hidden_sizes}{bn}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``.
        """
        return self.network(x)


# ===========================================================================
# 2. Multi-Lag MLP
# ===========================================================================

class MultiLagMLP(nn.Module):
    """MLP that takes concatenated multi-lag features as a flat input.

    This architecture is designed for inputs where *k* lags of
    *features_per_lag* features are stacked:
    ``x = [lag_1_features | lag_2_features | ... | lag_k_features]``.

    The total input dimension is ``n_lags * features_per_lag``.

    Parameters
    ----------
    features_per_lag : int
        Number of features at each time lag.
    n_lags : int
        Number of time lags concatenated.
    hidden_sizes : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout probability.

    Examples
    --------
    >>> model = MultiLagMLP(features_per_lag=30, n_lags=3)
    >>> x = torch.randn(16, 90)  # 3 lags * 30 features
    >>> model(x).shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        features_per_lag: int,
        n_lags: int = 3,
        hidden_sizes: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.features_per_lag = features_per_lag
        self.n_lags = n_lags
        self.hidden_sizes = list(hidden_sizes)
        self.dropout_rate = dropout
        self.n_features = features_per_lag * n_lags  # total input dim

        layers: list[nn.Module] = []
        in_dim = self.n_features

        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        logger.info(
            "MultiLagMLP created: features_per_lag=%d, n_lags=%d, "
            "total_input=%d, hidden=%s, params=%d",
            features_per_lag, n_lags, self.n_features, hidden_sizes,
            sum(p.numel() for p in self.parameters()),
        )

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return f"MultiLagMLP(lags={self.n_lags},h={self.hidden_sizes})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_lags * features_per_lag)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``.
        """
        return self.network(x)


# ===========================================================================
# 3. LSTM / GRU Sequence Model
# ===========================================================================

class LSTMPredictor(nn.Module):
    """LSTM/GRU sequence model for temperature prediction.

    Processes a sequence of *k* days of station features through a
    recurrent layer, then feeds the final hidden state through a
    dense output head.

    Input shape: ``(batch, seq_len, features_per_step)``

    For flat 2-D input ``(batch, n_features)`` the model automatically
    reshapes to ``(batch, 1, n_features)`` (single-step sequence).

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        Size of the recurrent hidden state.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout between stacked layers (only used if ``num_layers > 1``).
    bidirectional : bool
        If ``True``, use a bidirectional RNN.
    cell_type : str
        ``"lstm"`` or ``"gru"`` (case-insensitive).

    Examples
    --------
    >>> model = LSTMPredictor(input_size=30, hidden_size=64)
    >>> x = torch.randn(16, 5, 30)  # 5 time steps, 30 features each
    >>> model(x).shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        cell_type: str = "lstm",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower().strip()
        self.n_features = input_size  # for compatibility

        # Validate cell type
        if self.cell_type not in ("lstm", "gru"):
            raise ValueError(
                f"cell_type must be 'lstm' or 'gru', got '{cell_type}'"
            )

        # Recurrent layer
        rnn_dropout = dropout if num_layers > 1 else 0.0
        RNNClass = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        self.rnn = RNNClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )

        # Output head
        n_directions = 2 if bidirectional else 1
        fc_input = hidden_size * n_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_input // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(fc_input // 2, 1),
        )

        logger.info(
            "LSTMPredictor created: input_size=%d, hidden=%d, layers=%d, "
            "cell=%s, bidir=%s, params=%d",
            input_size, hidden_size, num_layers, self.cell_type,
            bidirectional,
            sum(p.numel() for p in self.parameters()),
        )

    @property
    def name(self) -> str:
        """Human-readable model name."""
        bidir = ",bidir" if self.bidirectional else ""
        return (
            f"{self.cell_type.upper()}(h={self.hidden_size},"
            f"L={self.num_layers}{bidir})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_size)`` or
            ``(batch, input_size)`` (auto-unsqueezed to seq_len=1).

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``.
        """
        # Auto-reshape 2D input to 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)

        # rnn_out: (batch, seq_len, hidden * n_dirs)
        rnn_out, _ = self.rnn(x)

        # Use the last time-step's output
        last_out = rnn_out[:, -1, :]  # (batch, hidden * n_dirs)

        return self.fc(last_out)


# ===========================================================================
# 4. Station Attention Pooling
# ===========================================================================

class StationAttentionModel(nn.Module):
    """Station-level attention pooling model.

    Each station's features are encoded independently using a shared
    encoder, producing *embed_dim*-dimensional embeddings.  Multi-head
    attention then aggregates embeddings across stations, and a dense
    head produces the final prediction.

    Non-station features (e.g. sin_day, cos_day) are concatenated to
    the pooled embedding before the output head.

    Parameters
    ----------
    features_per_station : int
        Number of features for each station (e.g., 2 for TMAX+TMIN).
    n_stations : int
        Number of weather stations.
    embed_dim : int
        Dimension of each station embedding.
    n_heads : int
        Number of attention heads.  Must evenly divide *embed_dim*.
    dropout : float
        Dropout probability.
    n_extra_features : int
        Number of additional non-station features (e.g., 2 for date
        encodings).  These bypass the attention and are concatenated
        to the pooled vector.

    Examples
    --------
    >>> model = StationAttentionModel(
    ...     features_per_station=2, n_stations=14,
    ...     embed_dim=32, n_heads=4, n_extra_features=2,
    ... )
    >>> x = torch.randn(16, 30)  # 14*2 + 2 = 30
    >>> model(x).shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        features_per_station: int,
        n_stations: int,
        embed_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_extra_features: int = 2,
    ):
        super().__init__()

        self.features_per_station = features_per_station
        self.n_stations = n_stations
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.n_extra_features = n_extra_features
        self.n_features = n_stations * features_per_station + n_extra_features

        # Shared station encoder
        self.station_encoder = nn.Sequential(
            nn.Linear(features_per_station, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Output head
        fc_input = embed_dim + n_extra_features
        self.output_head = nn.Sequential(
            nn.Linear(fc_input, fc_input),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(fc_input, 1),
        )

        # Initialize weights
        self._init_weights()

        logger.info(
            "StationAttentionModel created: %d stations x %d feats, "
            "embed=%d, heads=%d, extra=%d, params=%d",
            n_stations, features_per_station, embed_dim, n_heads,
            n_extra_features,
            sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self) -> None:
        """Xavier-uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return (
            f"StationAttn(s={self.n_stations},e={self.embed_dim},"
            f"h={self.n_heads})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_stations * features_per_station + n_extra)``.
            Station features come first, followed by extra features.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``.
        """
        batch_size = x.size(0)
        n_station_feats = self.n_stations * self.features_per_station

        # Split station features and extra features
        station_flat = x[:, :n_station_feats]
        extra = x[:, n_station_feats:]  # (batch, n_extra)

        # Reshape to (batch, n_stations, features_per_station)
        station_data = station_flat.view(
            batch_size, self.n_stations, self.features_per_station
        )

        # Encode each station (shared weights)
        # station_data: (batch, n_stations, features_per_station)
        # embeddings:   (batch, n_stations, embed_dim)
        embeddings = self.station_encoder(station_data)

        # Attention pooling using learnable query
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        attn_out, self._attn_weights = self.attention(
            query, embeddings, embeddings
        )
        # attn_out: (batch, 1, embed_dim)
        pooled = self.layer_norm(attn_out.squeeze(1))  # (batch, embed_dim)

        # Concatenate extra features (date encodings, etc.)
        if self.n_extra_features > 0:
            combined = torch.cat([pooled, extra], dim=1)
        else:
            combined = pooled

        return self.output_head(combined)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the most recent attention weights.

        Returns
        -------
        torch.Tensor or None
            Shape ``(batch, 1, n_stations)`` if a forward pass has been
            done, else ``None``.
        """
        return getattr(self, "_attn_weights", None)


# ===========================================================================
# Reshape Utilities
# ===========================================================================

def reshape_for_lstm(
    X: torch.Tensor,
    seq_len: int = 1,
) -> torch.Tensor:
    """Reshape flat 2-D features to 3-D for LSTM input.

    Parameters
    ----------
    X : torch.Tensor
        Shape ``(batch, total_features)`` where
        ``total_features = seq_len * features_per_step``.
    seq_len : int
        Number of time steps.  ``total_features`` must be divisible
        by *seq_len*.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, seq_len, features_per_step)``.

    Raises
    ------
    ValueError
        If ``total_features`` is not divisible by *seq_len*.
    """
    if X.dim() != 2:
        raise ValueError(f"Expected 2-D input, got {X.dim()}-D")

    total = X.size(1)
    if total % seq_len != 0:
        raise ValueError(
            f"total_features ({total}) not divisible by seq_len ({seq_len})"
        )

    features_per_step = total // seq_len
    return X.view(X.size(0), seq_len, features_per_step)


def reshape_for_attention(
    X: torch.Tensor,
    n_stations: int,
    features_per_station: int,
    n_extra: int = 0,
) -> torch.Tensor:
    """Reshape flat 2-D features for the StationAttentionModel.

    This is an identity operation — the StationAttentionModel handles
    reshaping internally.  The function validates shapes and returns
    *X* unchanged.

    Parameters
    ----------
    X : torch.Tensor
        Shape ``(batch, n_stations * features_per_station + n_extra)``.
    n_stations : int
        Number of stations.
    features_per_station : int
        Features per station.
    n_extra : int
        Number of extra (non-station) features.

    Returns
    -------
    torch.Tensor
        The same tensor (shape validated).

    Raises
    ------
    ValueError
        If feature dimension does not match expectations.
    """
    expected = n_stations * features_per_station + n_extra
    if X.size(-1) != expected:
        raise ValueError(
            f"Expected {expected} features "
            f"({n_stations}*{features_per_station}+{n_extra}), "
            f"got {X.size(-1)}"
        )
    return X


# ===========================================================================
# Model Factory
# ===========================================================================

def create_model_v2(
    model_class: str,
    n_features: int,
    **kwargs,
) -> nn.Module:
    """Factory function for Phase 4 models.

    Parameters
    ----------
    model_class : str
        One of ``"enhanced_mlp"``, ``"multi_lag_mlp"``, ``"lstm"``,
        ``"gru"``, ``"attention"``.
    n_features : int
        Total number of input features.
    **kwargs
        Additional keyword arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module
        An initialised (untrained) model.

    Raises
    ------
    ValueError
        If *model_class* is not recognised.
    """
    model_class = model_class.lower().strip()

    if model_class == "enhanced_mlp":
        return EnhancedMLP(n_features=n_features, **kwargs)

    elif model_class == "multi_lag_mlp":
        features_per_lag = kwargs.pop("features_per_lag", n_features)
        n_lags = kwargs.pop("n_lags", 1)
        return MultiLagMLP(
            features_per_lag=features_per_lag,
            n_lags=n_lags,
            **kwargs,
        )

    elif model_class in ("lstm", "gru"):
        cell = kwargs.pop("cell_type", model_class)
        return LSTMPredictor(
            input_size=n_features,
            cell_type=cell,
            **kwargs,
        )

    elif model_class == "attention":
        return StationAttentionModel(
            features_per_station=kwargs.pop("features_per_station", 2),
            n_stations=kwargs.pop("n_stations", 14),
            n_extra_features=kwargs.pop("n_extra_features", 2),
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown model_class '{model_class}'. "
            "Choose from: enhanced_mlp, multi_lag_mlp, lstm, gru, attention."
        )
