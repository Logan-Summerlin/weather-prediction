"""
Wind-Gated Station Attention Model for NYC Temperature Prediction.

Replaces static station attention with wind-conditioned gating.
Upwind stations receive higher attention weights when prevailing wind
direction aligns with the station bearing relative to Central Park.

Architecture overview:
  1. Per-station shared encoder: maps raw station features to embeddings.
  2. Station metadata (bearing, distance, elevation, sector) is concatenated
     with embeddings to form attention keys.
  3. Global context (wind, SLP, date, NYC prev TMAX) forms the query.
  4. Scaled dot-product attention with additive wind bias produces pooled
     station representation.
  5. Output head: point prediction (delta-T) or heteroscedastic Gaussian
     (mu, sigma) for probabilistic forecasting.

Key design choices:
  - Shared-weight encoder avoids overfitting with many stations.
  - Wind bias alpha * cos(wind_dir - bearing_i) is a learnable scalar
    that upweights stations along the prevailing wind direction.
  - Missing-station masking sets attention logits to -inf before softmax.
  - LayerNorm on station embeddings for training stability.
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


class WindGatedAttentionModel(nn.Module):
    """Station-level attention model with wind-direction gating.

    Each station's raw features are encoded through a shared MLP,
    then aggregated via scaled dot-product attention where the query
    is derived from global context and the keys incorporate station
    metadata.  An additive wind-bias term biases attention toward
    upwind stations.

    Parameters
    ----------
    n_station_features : int
        Number of raw features per station (e.g. TMAX, TMIN, diurnal
        range, dewpoint, wind speed, delta-T).
    n_metadata_features : int
        Number of metadata features per station (bearing, distance,
        elevation, sector one-hot, etc.).
    n_global_features : int
        Number of global context features (wind dir sin/cos, wind
        speed mean, SLP, SLP tendency, sin_day, cos_day, NYC prev
        TMAX).
    n_stations : int
        Maximum number of surrounding stations.
    station_embed_dim : int
        Dimension of station embeddings produced by the shared
        encoder.
    attention_dim : int
        Dimension of queries and keys in the attention mechanism
        (d_k).
    output_mode : str
        ``"point"`` for single delta-T prediction or ``"gaussian"``
        for heteroscedastic (mu, log_sigma) output.
    dropout : float
        Dropout probability applied in the shared encoder and output
        head.

    Examples
    --------
    >>> model = WindGatedAttentionModel(
    ...     n_station_features=6, n_metadata_features=4,
    ...     n_global_features=7, n_stations=14,
    ... )
    >>> batch = 16
    >>> sf = torch.randn(batch, 14, 6)
    >>> sm = torch.randn(batch, 14, 4)
    >>> gc = torch.randn(batch, 7)
    >>> sb = torch.randn(batch, 14)
    >>> wd = torch.randn(batch)
    >>> mask = torch.ones(batch, 14)
    >>> out = model(sf, sm, gc, sb, wd, mask)
    >>> out["prediction"].shape
    torch.Size([16, 1])
    """

    def __init__(
        self,
        n_station_features: int,
        n_metadata_features: int,
        n_global_features: int,
        n_stations: int,
        station_embed_dim: int = 32,
        attention_dim: int = 16,
        output_mode: str = "point",
        dropout: float = 0.1,
    ):
        super().__init__()

        if output_mode not in ("point", "gaussian"):
            raise ValueError(
                f"output_mode must be 'point' or 'gaussian', "
                f"got '{output_mode}'"
            )

        self.n_station_features = n_station_features
        self.n_metadata_features = n_metadata_features
        self.n_global_features = n_global_features
        self.n_stations = n_stations
        self.station_embed_dim = station_embed_dim
        self.attention_dim = attention_dim
        self.output_mode = output_mode
        self.dropout_rate = dropout

        # ---- Per-station shared encoder ----
        # Maps raw station features -> station_embed_dim
        self.station_encoder = nn.Sequential(
            nn.Linear(n_station_features, station_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(station_embed_dim, station_embed_dim),
        )

        # Layer norm for station embeddings (stability)
        self.station_layer_norm = nn.LayerNorm(station_embed_dim)

        # ---- Attention projections ----
        # Query from global context
        self.query_proj = nn.Linear(n_global_features, attention_dim)
        # Key from concat(station_embed, station_metadata)
        key_input_dim = station_embed_dim + n_metadata_features
        self.key_proj = nn.Linear(key_input_dim, attention_dim)
        # Value = station embedding (no projection needed; use station_embed_dim)

        # ---- Learnable wind-bias scaling ----
        # alpha * cos(wind_dir - bearing_i) is added to attention logits
        self.wind_alpha = nn.Parameter(torch.tensor(1.0))

        # ---- Output head ----
        output_input_dim = station_embed_dim + n_global_features
        if output_mode == "point":
            self.output_head = nn.Sequential(
                nn.Linear(output_input_dim, output_input_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(output_input_dim, 1),
            )
        else:  # gaussian
            # Shared hidden layer, then separate mu and log_sigma heads
            self.output_hidden = nn.Sequential(
                nn.Linear(output_input_dim, output_input_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )
            self.mu_head = nn.Linear(output_input_dim, 1)
            self.log_sigma_head = nn.Linear(output_input_dim, 1)

        # ---- Initialise weights ----
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "WindGatedAttentionModel created: %d stations, "
            "station_feats=%d, meta_feats=%d, global_feats=%d, "
            "embed=%d, attn_dim=%d, mode=%s, params=%d",
            n_stations, n_station_features, n_metadata_features,
            n_global_features, station_embed_dim, attention_dim,
            output_mode, n_params,
        )

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        return (
            f"WindGatedAttn(s={self.n_stations},"
            f"e={self.station_embed_dim},"
            f"dk={self.attention_dim},"
            f"mode={self.output_mode})"
        )

    def forward(
        self,
        station_features: torch.Tensor,
        station_metadata: torch.Tensor,
        global_context: torch.Tensor,
        station_bearings: torch.Tensor,
        wind_direction: torch.Tensor,
        station_mask: torch.Tensor,
    ) -> dict:
        """Forward pass.

        Parameters
        ----------
        station_features : torch.Tensor
            Shape ``(batch, n_stations, n_station_features)``.
            Raw weather observations per station.
        station_metadata : torch.Tensor
            Shape ``(batch, n_stations, n_metadata_features)``.
            Static station metadata (bearing, distance, etc.).
        global_context : torch.Tensor
            Shape ``(batch, n_global_features)``.
            Global features (wind, SLP, date encoding, NYC prev).
        station_bearings : torch.Tensor
            Shape ``(batch, n_stations)``.
            Bearing from Central Park to each station in radians.
        wind_direction : torch.Tensor
            Shape ``(batch,)``.
            Prevailing wind direction in radians.
        station_mask : torch.Tensor
            Shape ``(batch, n_stations)``.
            Binary mask: 1 = station present, 0 = missing.

        Returns
        -------
        dict
            Always contains:
              - ``"prediction"``: ``(batch, 1)`` point prediction
                (or mu for gaussian mode).
              - ``"attention_weights"``: ``(batch, n_stations)``
                normalised attention weights.
            In gaussian mode, additionally:
              - ``"mu"``: ``(batch, 1)``
              - ``"log_sigma"``: ``(batch, 1)``
              - ``"sigma"``: ``(batch, 1)`` (exp of log_sigma,
                clamped for numerical stability).
        """
        batch_size = station_features.size(0)

        # ---- Encode stations (shared weights) ----
        # station_features: (B, S, F_station)
        embeddings = self.station_encoder(station_features)  # (B, S, E)
        embeddings = self.station_layer_norm(embeddings)  # (B, S, E)

        # ---- Build keys and values ----
        # Keys = concat(embedding, metadata)
        key_input = torch.cat([embeddings, station_metadata], dim=-1)  # (B, S, E+M)
        keys = self.key_proj(key_input)  # (B, S, d_k)

        # Values = raw embeddings
        values = embeddings  # (B, S, E)

        # ---- Build query from global context ----
        query = self.query_proj(global_context)  # (B, d_k)
        query = query.unsqueeze(1)  # (B, 1, d_k)

        # ---- Scaled dot-product attention ----
        # logits: (B, 1, S) = Q @ K^T / sqrt(d_k)
        d_k = self.attention_dim
        attn_logits = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(d_k)
        attn_logits = attn_logits.squeeze(1)  # (B, S)

        # ---- Wind bias ----
        # cos(wind_dir - bearing_i) -> upwind gets positive bias
        # wind_direction: (B,) -> (B, 1) for broadcasting
        wind_bias = self.wind_alpha * torch.cos(
            wind_direction.unsqueeze(1) - station_bearings
        )  # (B, S)
        attn_logits = attn_logits + wind_bias

        # ---- Mask missing stations ----
        # Set logits to -inf where mask == 0
        mask_bool = station_mask.bool()

        # Edge case: if ALL stations are masked in a sample, avoid
        # NaN from softmax by leaving logits as-is (uniform).
        all_masked = ~mask_bool.any(dim=1)  # (B,)
        if all_masked.any():
            # For fully-masked samples, set all logits to 0 so
            # softmax produces uniform weights (graceful degradation).
            attn_logits = attn_logits.masked_fill(
                all_masked.unsqueeze(1).expand_as(attn_logits), 0.0
            )

        # Normal masking: -inf for absent stations
        attn_logits = attn_logits.masked_fill(~mask_bool, float("-inf"))

        # Re-handle all-masked case: replace -inf with 0
        if all_masked.any():
            all_inf = torch.isinf(attn_logits).all(dim=1)  # (B,)
            attn_logits = attn_logits.masked_fill(
                all_inf.unsqueeze(1).expand_as(attn_logits), 0.0
            )

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, S)

        # Zero out weights for masked stations (softmax may assign
        # residual weight due to numerical precision)
        attn_weights = attn_weights * station_mask

        # Renormalise after zeroing (avoid div by zero)
        weight_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / weight_sum

        # ---- Weighted pooling ----
        # (B, S) -> (B, 1, S) @ (B, S, E) -> (B, 1, E) -> (B, E)
        pooled = torch.bmm(
            attn_weights.unsqueeze(1), values
        ).squeeze(1)  # (B, E)

        # ---- Output ----
        combined = torch.cat([pooled, global_context], dim=-1)  # (B, E+G)

        result: dict[str, torch.Tensor] = {
            "attention_weights": attn_weights,
        }

        if self.output_mode == "point":
            prediction = self.output_head(combined)  # (B, 1)
            result["prediction"] = prediction
        else:
            hidden = self.output_hidden(combined)
            mu = self.mu_head(hidden)  # (B, 1)
            log_sigma = self.log_sigma_head(hidden)  # (B, 1)
            # Clamp log_sigma for numerical stability
            log_sigma = log_sigma.clamp(min=-10.0, max=5.0)
            sigma = torch.exp(log_sigma)

            result["prediction"] = mu
            result["mu"] = mu
            result["log_sigma"] = log_sigma
            result["sigma"] = sigma

        return result

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass.

        Returns
        -------
        torch.Tensor or None
            Shape ``(batch, n_stations)`` if a forward pass has been
            done, else ``None``.

        Notes
        -----
        This is a convenience method; attention weights are always
        returned in the forward pass output dict.
        """
        return getattr(self, "_last_attn_weights", None)
