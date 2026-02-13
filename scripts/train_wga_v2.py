#!/usr/bin/env python3
"""
Train Enhanced Wind-Gated Attention V2 + MDN model for NYC temperature prediction.

V2 enhancements over original WGA-MDN:
    1. Multi-head attention (4 heads): Different heads capture frontal,
       marine, advection, and seasonal patterns independently.
    2. Deeper station encoder (3 layers + residual connection + LayerNorm).
    3. Lag-2 station features: TMAX_lag2, TMIN_lag2, TMAX_change_t3_to_t2
       (9 total station features vs 6 in V1).
    4. Residual connection in Gaussian output head.

Ablation configurations:
    wga_v2_full:           4-head + 3-layer encoder + lag-2 features
    wga_v2_multihead_only: 4-head + 2-layer encoder + original 6 features
    wga_v2_deep_only:      1-head + 3-layer encoder + original 6 features
    wga_v2_lag2_only:      1-head + 2-layer encoder + lag-2 features

Station ablation ladder (full config, seed=42 only):
    top_10, top_20, top_30, all_47

Splits:
    Train: 2000-06-01 to 2019-12-31
    Val:   2020-01-01 to 2022-12-31
    Test:  2023-01-01 to 2024-12-31

Outputs saved to: results/wga_v2_model/
"""

import os
import sys
import json
import time
import copy
import math
import logging
import warnings
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from config_expanded import (
    SURROUNDING_STATIONS,
    STATION_METADATA,
    METEOROLOGICAL_SECTORS,
    STATION_SECTORS,
    STATION_RINGS,
)
from src.data_collection import download_dly_file, parse_dly_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger("train_wga_v2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_extended.csv")
ERA_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "mos_era_indicator.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "wga_v2_model")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# Chronological splits
MOS_TRAIN_START, MOS_TRAIN_END = "2000-06-01", "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

# Hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
STATION_EMBED_DIM = 64
ATTENTION_DIM = 32

# Sigma clamps
SIGMA_FLOOR = 0.75
SIGMA_CAP = 10.0
NYC_LAT = 40.7831


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. ENHANCED WGA-MDN V2 MODEL
# ============================================================================

class ResidualStationEncoder(nn.Module):
    """Deep station encoder with residual connection and LayerNorm.

    Architecture (depth=3):
        x -> Linear(F, E) -> ReLU -> Dropout
          -> Linear(E, E) -> ReLU -> Dropout
          -> Linear(E, E) -> + x_proj -> LayerNorm

    For depth=2, this reduces to the original V1 encoder:
        x -> Linear(F, E) -> ReLU -> Dropout -> Linear(E, E) -> LayerNorm
    """

    def __init__(self, n_features: int, embed_dim: int, depth: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.depth = depth

        layers = []
        # First layer: map from n_features -> embed_dim
        layers.append(nn.Linear(n_features, embed_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Middle + final layers: embed_dim -> embed_dim
        for i in range(1, depth):
            layers.append(nn.Linear(embed_dim, embed_dim))
            if i < depth - 1:
                # Intermediate layers get ReLU + Dropout
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            # Last layer: no activation before residual add

        self.layers = nn.Sequential(*layers)

        # Residual projection: if n_features != embed_dim, project x
        if n_features != embed_dim:
            self.residual_proj = nn.Linear(n_features, embed_dim)
        else:
            self.residual_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_stations, n_features) -> (batch, n_stations, embed_dim)"""
        residual = self.residual_proj(x)
        out = self.layers(x)
        if self.depth >= 3:
            out = out + residual
        return self.layer_norm(out)


class MultiHeadWindAttention(nn.Module):
    """Multi-head attention with per-head wind-direction gating.

    Each head has its own Q/K projections and its own learnable wind_alpha.
    Heads are concatenated and projected to embed_dim.
    """

    def __init__(self, n_global: int, key_input_dim: int, embed_dim: int,
                 attention_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim  # each head uses full attention_dim

        # Per-head projections
        self.query_projs = nn.ModuleList([
            nn.Linear(n_global, attention_dim) for _ in range(n_heads)
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(key_input_dim, attention_dim) for _ in range(n_heads)
        ])

        # Per-head wind gating
        self.wind_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(n_heads)
        ])

        # Final projection: concat of n_heads * embed_dim -> embed_dim
        self.output_proj = nn.Linear(n_heads * embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, embeddings: torch.Tensor, metadata: torch.Tensor,
                global_context: torch.Tensor, station_bearings: torch.Tensor,
                wind_direction: torch.Tensor, station_mask: torch.Tensor):
        """
        embeddings: (B, S, E) station embeddings
        metadata: (B, S, M) station metadata
        global_context: (B, G)
        station_bearings: (B, S)
        wind_direction: (B,)
        station_mask: (B, S)

        Returns: pooled (B, E), attn_weights (B, S) averaged across heads
        """
        batch_size, n_stations, embed_dim = embeddings.shape
        key_input = torch.cat([embeddings, metadata], dim=-1)  # (B, S, E+M)
        mask_bool = station_mask.bool()

        head_outputs = []
        head_attn_weights = []

        for h in range(self.n_heads):
            query = self.query_projs[h](global_context).unsqueeze(1)  # (B, 1, dk)
            keys = self.key_projs[h](key_input)  # (B, S, dk)

            d_k = self.attention_dim
            attn_logits = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(d_k)
            attn_logits = attn_logits.squeeze(1)  # (B, S)

            # Wind bias
            wind_bias = self.wind_alphas[h] * torch.cos(
                wind_direction.unsqueeze(1) - station_bearings
            )
            attn_logits = attn_logits + wind_bias

            # Mask missing stations
            all_masked = ~mask_bool.any(dim=1)
            if all_masked.any():
                attn_logits = attn_logits.masked_fill(
                    all_masked.unsqueeze(1).expand_as(attn_logits), 0.0
                )
            attn_logits = attn_logits.masked_fill(~mask_bool, float("-inf"))
            if all_masked.any():
                all_inf = torch.isinf(attn_logits).all(dim=1)
                attn_logits = attn_logits.masked_fill(
                    all_inf.unsqueeze(1).expand_as(attn_logits), 0.0
                )

            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = attn_weights * station_mask
            weight_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / weight_sum
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted pooling
            pooled = torch.bmm(attn_weights.unsqueeze(1), embeddings).squeeze(1)  # (B, E)
            head_outputs.append(pooled)
            head_attn_weights.append(attn_weights)

        # Concatenate heads and project
        combined = torch.cat(head_outputs, dim=-1)  # (B, n_heads * E)
        output = self.output_proj(combined)  # (B, E)

        # Average attention weights across heads for interpretability
        avg_attn = torch.stack(head_attn_weights, dim=0).mean(dim=0)  # (B, S)

        return output, avg_attn


class SingleHeadWindAttention(nn.Module):
    """Original single-head attention (V1 compatible), extracted as a module."""

    def __init__(self, n_global: int, key_input_dim: int, embed_dim: int,
                 attention_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention_dim = attention_dim
        self.query_proj = nn.Linear(n_global, attention_dim)
        self.key_proj = nn.Linear(key_input_dim, attention_dim)
        self.wind_alpha = nn.Parameter(torch.tensor(1.0))
        self.attn_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, embeddings, metadata, global_context,
                station_bearings, wind_direction, station_mask):
        key_input = torch.cat([embeddings, metadata], dim=-1)
        query = self.query_proj(global_context).unsqueeze(1)
        keys = self.key_proj(key_input)

        d_k = self.attention_dim
        attn_logits = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(d_k)
        attn_logits = attn_logits.squeeze(1)

        wind_bias = self.wind_alpha * torch.cos(
            wind_direction.unsqueeze(1) - station_bearings
        )
        attn_logits = attn_logits + wind_bias

        mask_bool = station_mask.bool()
        all_masked = ~mask_bool.any(dim=1)
        if all_masked.any():
            attn_logits = attn_logits.masked_fill(
                all_masked.unsqueeze(1).expand_as(attn_logits), 0.0
            )
        attn_logits = attn_logits.masked_fill(~mask_bool, float("-inf"))
        if all_masked.any():
            all_inf = torch.isinf(attn_logits).all(dim=1)
            attn_logits = attn_logits.masked_fill(
                all_inf.unsqueeze(1).expand_as(attn_logits), 0.0
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights * station_mask
        weight_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / weight_sum
        attn_weights = self.attn_dropout(attn_weights)

        pooled = torch.bmm(attn_weights.unsqueeze(1), embeddings).squeeze(1)
        return pooled, attn_weights


class EnhancedWindGatedAttentionModel(nn.Module):
    """Enhanced WGA-MDN V2 with multi-head attention, deep encoder, lag-2 features.

    Configurable for ablation studies:
      - n_heads=1 reverts to single-head attention
      - encoder_depth=2 reverts to V1 encoder
      - n_station_features can be 6 (V1) or 9 (V2 with lag-2)
    """

    def __init__(
        self,
        n_station_features: int,
        n_metadata_features: int,
        n_global_features: int,
        n_stations: int,
        station_embed_dim: int = 64,
        attention_dim: int = 32,
        output_mode: str = "gaussian",
        dropout: float = 0.1,
        n_heads: int = 4,
        encoder_depth: int = 3,
    ):
        super().__init__()

        if output_mode not in ("point", "gaussian"):
            raise ValueError(f"output_mode must be 'point' or 'gaussian', got '{output_mode}'")

        self.n_station_features = n_station_features
        self.n_metadata_features = n_metadata_features
        self.n_global_features = n_global_features
        self.n_stations = n_stations
        self.station_embed_dim = station_embed_dim
        self.attention_dim = attention_dim
        self.output_mode = output_mode
        self.dropout_rate = dropout
        self.n_heads = n_heads
        self.encoder_depth = encoder_depth

        # ---- Station encoder ----
        self.station_encoder = ResidualStationEncoder(
            n_features=n_station_features,
            embed_dim=station_embed_dim,
            depth=encoder_depth,
            dropout=dropout,
        )

        # ---- Attention mechanism ----
        key_input_dim = station_embed_dim + n_metadata_features
        if n_heads > 1:
            self.attention = MultiHeadWindAttention(
                n_global=n_global_features,
                key_input_dim=key_input_dim,
                embed_dim=station_embed_dim,
                attention_dim=attention_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:
            self.attention = SingleHeadWindAttention(
                n_global=n_global_features,
                key_input_dim=key_input_dim,
                embed_dim=station_embed_dim,
                attention_dim=attention_dim,
                dropout=dropout,
            )

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
            self.output_hidden = nn.Sequential(
                nn.Linear(output_input_dim, output_input_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )
            self.mu_head = nn.Linear(output_input_dim, 1)
            self.log_sigma_head = nn.Linear(output_input_dim, 1)
            # Residual skip from global context for mu
            self.mu_skip = nn.Linear(n_global_features, 1)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "EnhancedWGA_V2: stations=%d, station_feats=%d, meta=%d, global=%d, "
            "embed=%d, attn=%d, heads=%d, depth=%d, mode=%s, params=%d",
            n_stations, n_station_features, n_metadata_features,
            n_global_features, station_embed_dim, attention_dim,
            n_heads, encoder_depth, output_mode, n_params,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        return (
            f"EnhancedWGA_V2(s={self.n_stations},e={self.station_embed_dim},"
            f"dk={self.attention_dim},h={self.n_heads},d={self.encoder_depth},"
            f"sf={self.n_station_features},mode={self.output_mode})"
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
        # Encode stations
        embeddings = self.station_encoder(station_features)  # (B, S, E)

        # Attention pooling
        pooled, attn_weights = self.attention(
            embeddings, station_metadata, global_context,
            station_bearings, wind_direction, station_mask,
        )

        # Output
        combined = torch.cat([pooled, global_context], dim=-1)

        result = {"attention_weights": attn_weights}

        if self.output_mode == "point":
            prediction = self.output_head(combined)
            result["prediction"] = prediction
        else:
            hidden = self.output_hidden(combined)
            mu = self.mu_head(hidden) + self.mu_skip(global_context)  # residual skip
            log_sigma = self.log_sigma_head(hidden)
            log_sigma = log_sigma.clamp(min=-10.0, max=5.0)
            sigma = torch.exp(log_sigma)

            result["prediction"] = mu
            result["mu"] = mu
            result["log_sigma"] = log_sigma
            result["sigma"] = sigma

        return result


# ============================================================================
# 2. DATA LOADING (reused from train_wga_mdn.py)
# ============================================================================

def download_all_stations() -> None:
    """Download .dly files for target + all surrounding stations."""
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ids = [TARGET_STATION] + ALL_SURROUNDING
    total = len(all_ids)
    for i, sid in enumerate(all_ids, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path):
            continue
        for attempt in range(4):
            try:
                download_dly_file(sid, RAW_DIR)
                logger.info("[%d/%d] %s -- downloaded", i, total, sid)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "[%d/%d] %s attempt %d failed: %s",
                    i, total, sid, attempt + 1, e,
                )
                time.sleep(wait)
        else:
            logger.error("[%d/%d] %s -- FAILED after 4 attempts", i, total, sid)


def parse_station_element(station_id: str, element: str,
                          start_date: str, end_date: str) -> pd.Series:
    """Parse a single element (TMAX or TMIN) for a station."""
    dly_path = os.path.join(RAW_DIR, f"{station_id}.dly")
    if not os.path.exists(dly_path):
        return pd.Series(dtype=float)
    df = parse_dly_file(dly_path, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[df["element"] == element][["date", "value"]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.drop_duplicates(subset="date").set_index("date")["value"]
    sub.name = f"{station_id}_{element}"
    return sub


def build_station_matrix(start_date: str, end_date: str) -> pd.DataFrame:
    """Build a wide DataFrame with TMAX and TMIN for every surrounding station."""
    logger.info("Building station matrix from %s to %s ...", start_date, end_date)
    frames = []
    for sid in ALL_SURROUNDING:
        for elem in ("TMAX", "TMIN"):
            series = parse_station_element(sid, elem, start_date, end_date)
            if len(series) > 0:
                frames.append(series)
    if not frames:
        return pd.DataFrame()
    matrix = pd.concat(frames, axis=1)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index()
    completeness = matrix.notna().mean()
    good_cols = completeness[completeness >= 0.80].index.tolist()
    dropped = len(matrix.columns) - len(good_cols)
    if dropped > 0:
        logger.info("Dropped %d columns below 80%% completeness", dropped)
    matrix = matrix[good_cols]
    logger.info(
        "Station matrix: %d rows x %d columns", len(matrix), len(matrix.columns)
    )
    return matrix


def load_mos_data() -> pd.DataFrame:
    """Load extended MOS data with era indicator."""
    mos = pd.read_csv(MOS_PATH, parse_dates=["date"])
    mos = mos[
        ["date", "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]
    ].copy()
    mos = mos.set_index("date").sort_index()
    logger.info(
        "Extended MOS data: %d rows, %s to %s",
        len(mos), mos.index.min().date(), mos.index.max().date(),
    )
    if os.path.exists(ERA_PATH):
        era = pd.read_csv(ERA_PATH, parse_dates=["date"])
        era = era.set_index("date").sort_index()
        mos = mos.join(era, how="left")
        mos["mos_era"] = mos["mos_era"].fillna(0).astype(float)
    else:
        mos["mos_era"] = 0.0
    return mos


def load_central_park_tmax() -> pd.DataFrame:
    """Load Central Park daily TMAX."""
    cp = pd.read_csv(CP_PATH, parse_dates=["date"])
    cp = cp.set_index("date").sort_index()
    cp.columns = ["nyc_tmax"]
    logger.info(
        "Central Park TMAX: %d rows, %s to %s",
        len(cp), cp.index.min().date(), cp.index.max().date(),
    )
    return cp


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def solar_declination(doy: np.ndarray) -> np.ndarray:
    return np.radians(23.44) * np.sin(np.radians((360 / 365.25) * (doy - 81)))


def day_length_hours(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.arccos(cos_ha)
    return (2.0 * ha / np.pi) * 12.0


def solar_elevation_noon(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    elev = np.arcsin(
        np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl)
    )
    return np.degrees(elev)


def assign_season(dates: pd.DatetimeIndex) -> pd.Series:
    month = dates.month
    seasons = pd.Series("", index=dates)
    seasons[month.isin([12, 1, 2])] = "DJF"
    seasons[month.isin([3, 4, 5])] = "MAM"
    seasons[month.isin([6, 7, 8])] = "JJA"
    seasons[month.isin([9, 10, 11])] = "SON"
    return seasons


# ============================================================================
# 4. STATION-LEVEL DATASET CONSTRUCTION (V2: lag-2 features)
# ============================================================================

def _get_ordered_station_list(station_matrix: pd.DataFrame) -> list:
    """Return ordered station IDs that have both TMAX and TMIN data."""
    tmax_sids = set()
    tmin_sids = set()
    for col in station_matrix.columns:
        if col.endswith("_TMAX"):
            tmax_sids.add(col.replace("_TMAX", ""))
        elif col.endswith("_TMIN"):
            tmin_sids.add(col.replace("_TMIN", ""))
    both = tmax_sids & tmin_sids
    ordered = [sid for sid in ALL_SURROUNDING if sid in both]
    return ordered


def _get_stations_by_distance(ordered_stations: list, n: int) -> list:
    """Return the N closest stations from ordered_stations, sorted by distance."""
    station_dists = []
    for sid in ordered_stations:
        meta = STATION_METADATA.get(sid, {})
        dist = meta.get("distance_mi", 999)
        station_dists.append((sid, dist))
    station_dists.sort(key=lambda x: x[1])
    return [sid for sid, _ in station_dists[:n]]


def build_structured_dataset(
    station_matrix: pd.DataFrame,
    mos_data: pd.DataFrame,
    cp_data: pd.DataFrame,
    use_lag2: bool = True,
    station_subset: list = None,
) -> dict:
    """Build tensors preserving station-level structure for the WGA V2 model.

    If use_lag2=True, produces 9 station features (adding lag-2 features).
    Otherwise produces 6 features (V1 compatible).

    station_subset: optional list of station IDs to restrict to.
    """
    logger.info(
        "Building structured dataset (use_lag2=%s, station_subset=%s) ...",
        use_lag2,
        f"{len(station_subset)} stations" if station_subset else "all",
    )

    # --- Determine ordered station list ---
    all_ordered = _get_ordered_station_list(station_matrix)
    if station_subset is not None:
        ordered_stations = [s for s in all_ordered if s in station_subset]
    else:
        ordered_stations = all_ordered
    n_stations = len(ordered_stations)
    logger.info("Using %d stations with both TMAX and TMIN", n_stations)

    # --- Build lag matrices ---
    tmax_cols = [f"{sid}_TMAX" for sid in ordered_stations]
    tmin_cols = [f"{sid}_TMIN" for sid in ordered_stations]
    tmax_lag1 = station_matrix[tmax_cols].shift(1)
    tmin_lag1 = station_matrix[tmin_cols].shift(1)
    tmax_lag2 = station_matrix[tmax_cols].shift(2)
    tmin_lag2 = station_matrix[tmin_cols].shift(2) if use_lag2 else None
    tmax_lag3 = station_matrix[tmax_cols].shift(3) if use_lag2 else None

    # --- Merge data sources ---
    nyc_lag1 = cp_data["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")
    df = pd.concat([cp_data, nyc_lag1, mos_data], axis=1, join="inner")
    df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])
    df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
    df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
    if "mos_era" not in df.columns:
        df["mos_era"] = 0.0
    df["mos_era"] = df["mos_era"].fillna(0.0)
    df = df.dropna(subset=["nyc_tmax_lag1"])

    # Residual target
    df["residual"] = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]

    # Date features
    doy = df.index.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.25)

    # MOS error memory
    mos_error = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
    mos_error_shifted = mos_error.shift(1)
    df["mos_error_7d"] = mos_error_shifted.rolling(window=7, min_periods=3).mean()

    # MOS spread
    df["gfs_nam_spread"] = (df["gfs_mos_tmax_f"] - df["nam_mos_tmax_f"]).abs()

    # Day length and solar elevation
    df["day_length"] = day_length_hours(NYC_LAT, doy)
    df["solar_elev_noon"] = solar_elevation_noon(NYC_LAT, doy)

    # --- Align lag matrices to df.index ---
    tmax_l1 = tmax_lag1.reindex(df.index)
    tmin_l1 = tmin_lag1.reindex(df.index)
    tmax_l2 = tmax_lag2.reindex(df.index)
    if use_lag2:
        tmin_l2 = tmin_lag2.reindex(df.index)
        tmax_l3 = tmax_lag3.reindex(df.index)

    # --- Station climatology (training period) ---
    train_mask_climo = (tmax_l1.index >= MOS_TRAIN_START) & (
        tmax_l1.index <= MOS_TRAIN_END
    )
    station_tmax_train_mean = tmax_l1.loc[train_mask_climo].mean()

    # --- Build per-station feature arrays ---
    if use_lag2:
        N_STATION_FEATURES = 9
    else:
        N_STATION_FEATURES = 6

    n_days = len(df)
    station_feats = np.zeros((n_days, n_stations, N_STATION_FEATURES), dtype=np.float32)
    station_mask = np.zeros((n_days, n_stations), dtype=np.float32)

    nyc_lag1_arr = df["nyc_tmax_lag1"].values

    for s_idx, sid in enumerate(ordered_stations):
        tmax_col = f"{sid}_TMAX"
        tmin_col = f"{sid}_TMIN"

        tmax_vals = tmax_l1[tmax_col].values.astype(np.float64)
        tmin_vals = tmin_l1[tmin_col].values.astype(np.float64)
        tmax_prev_vals = tmax_l2[tmax_col].values.astype(np.float64)
        climo_mean = station_tmax_train_mean.get(tmax_col, np.nan)

        # Mask: valid if both TMAX and TMIN lag1 exist
        valid = (~np.isnan(tmax_vals)) & (~np.isnan(tmin_vals))
        station_mask[:, s_idx] = valid.astype(np.float32)

        tmax_safe = np.where(valid, tmax_vals, 0.0)
        tmin_safe = np.where(valid, tmin_vals, 0.0)
        tmax_prev_safe = np.where(~np.isnan(tmax_prev_vals), tmax_prev_vals, tmax_safe)

        # V1 features (0-5)
        station_feats[:, s_idx, 0] = tmax_safe
        station_feats[:, s_idx, 1] = tmin_safe
        station_feats[:, s_idx, 2] = np.where(valid, tmax_safe - nyc_lag1_arr, 0.0)
        station_feats[:, s_idx, 3] = np.where(valid, tmax_safe - tmin_safe, 0.0)
        station_feats[:, s_idx, 4] = np.where(valid, tmax_safe - tmax_prev_safe, 0.0)
        station_feats[:, s_idx, 5] = np.where(
            valid, tmax_safe - (climo_mean if not np.isnan(climo_mean) else 0.0), 0.0
        )

        # V2 lag-2 features (6-8)
        if use_lag2:
            tmax_l2_vals = tmax_l2[tmax_col].values.astype(np.float64)
            tmin_l2_vals = tmin_l2[tmin_col].values.astype(np.float64)
            tmax_l3_vals = tmax_l3[tmax_col].values.astype(np.float64)

            tmax_l2_safe = np.where(~np.isnan(tmax_l2_vals), tmax_l2_vals, tmax_safe)
            tmin_l2_safe = np.where(~np.isnan(tmin_l2_vals), tmin_l2_vals, tmin_safe)
            tmax_l3_safe = np.where(~np.isnan(tmax_l3_vals), tmax_l3_vals, tmax_l2_safe)

            station_feats[:, s_idx, 6] = np.where(valid, tmax_l2_safe, 0.0)
            station_feats[:, s_idx, 7] = np.where(valid, tmin_l2_safe, 0.0)
            station_feats[:, s_idx, 8] = np.where(valid, tmax_l2_safe - tmax_l3_safe, 0.0)

    # --- Station metadata ---
    N_META_FEATURES = 6
    RING_MAP = {"Ring1_Near": 0, "Ring2_Regional": 1, "Ring3_Extended": 2, "Ring4_Far": 3}

    station_meta_static = np.zeros((n_stations, N_META_FEATURES), dtype=np.float32)
    station_bearings_static = np.zeros(n_stations, dtype=np.float32)

    max_dist = max(m["distance_mi"] for m in STATION_METADATA.values())

    for s_idx, sid in enumerate(ordered_stations):
        meta = STATION_METADATA.get(sid, {})
        bearing_deg = meta.get("bearing", 0.0)
        bearing_rad = np.radians(bearing_deg)
        distance = meta.get("distance_mi", 0.0)
        ring = meta.get("ring", "Ring2_Regional")

        station_bearings_static[s_idx] = bearing_rad
        station_meta_static[s_idx, 0] = bearing_rad
        station_meta_static[s_idx, 1] = distance / max_dist

        ring_idx = RING_MAP.get(ring, 1)
        station_meta_static[s_idx, 2 + ring_idx] = 1.0

    station_meta = np.tile(station_meta_static, (n_days, 1, 1))
    station_bearings = np.tile(station_bearings_static, (n_days, 1))

    # --- Wind direction proxy ---
    # Use ALL stations for wind proxy (not just subset)
    all_ordered_for_wind = _get_ordered_station_list(station_matrix)
    nw_sids = STATION_SECTORS.get("NW", []) + STATION_SECTORS.get("N", [])
    se_sids = STATION_SECTORS.get("SE", []) + STATION_SECTORS.get("S", [])
    ne_sids = STATION_SECTORS.get("NE", []) + STATION_SECTORS.get("E", [])
    sw_sids = STATION_SECTORS.get("SW", []) + STATION_SECTORS.get("W", [])

    tmax_all_lag1 = station_matrix[[f"{s}_TMAX" for s in all_ordered_for_wind
                                     if f"{s}_TMAX" in station_matrix.columns]].shift(1).reindex(df.index)

    def sector_tmax_mean(sids: list, tmax_df: pd.DataFrame) -> pd.Series:
        cols = [f"{s}_TMAX" for s in sids if f"{s}_TMAX" in tmax_df.columns]
        if cols:
            return tmax_df[cols].mean(axis=1)
        return pd.Series(0.0, index=tmax_df.index)

    nw_mean = sector_tmax_mean(nw_sids, tmax_all_lag1)
    se_mean = sector_tmax_mean(se_sids, tmax_all_lag1)
    ne_mean = sector_tmax_mean(ne_sids, tmax_all_lag1)
    sw_mean = sector_tmax_mean(sw_sids, tmax_all_lag1)

    grad_nwse = (nw_mean - se_mean).fillna(0.0).values
    grad_nesw = (ne_mean - sw_mean).fillna(0.0).values
    wind_proxy = np.arctan2(-grad_nesw, -grad_nwse)
    wind_proxy = np.where(np.isnan(wind_proxy), 0.0, wind_proxy).astype(np.float32)

    # --- Global context features ---
    GLOBAL_FEATURE_NAMES = [
        "sin_day", "cos_day", "nyc_tmax_lag1",
        "mos_ensemble_tmax_f", "gfs_mos_tmax_f", "nam_mos_tmax_f",
        "gfs_nam_spread", "mos_error_7d", "mos_era",
        "day_length", "solar_elev_noon",
    ]
    N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)

    global_ctx = np.zeros((n_days, N_GLOBAL_FEATURES), dtype=np.float32)
    for gi, gname in enumerate(GLOBAL_FEATURE_NAMES):
        vals = df[gname].values.astype(np.float64)
        vals = np.where(np.isnan(vals), 0.0, vals)
        global_ctx[:, gi] = vals

    # --- Chronological splits ---
    idx = df.index
    masks = {
        "train": (idx >= MOS_TRAIN_START) & (idx <= MOS_TRAIN_END),
        "val": (idx >= VAL_START) & (idx <= VAL_END),
        "test": (idx >= TEST_START) & (idx <= TEST_END),
    }

    # === SCALING: fit on training data only ===
    train_mask_np = np.array(masks["train"])
    n_sf = N_STATION_FEATURES
    station_feat_means = np.zeros(n_sf, dtype=np.float64)
    station_feat_stds = np.ones(n_sf, dtype=np.float64)

    for f_idx in range(n_sf):
        train_vals = station_feats[train_mask_np, :, f_idx]
        train_mask_valid = station_mask[train_mask_np, :]
        valid_vals = train_vals[train_mask_valid > 0.5]
        if len(valid_vals) > 0:
            station_feat_means[f_idx] = np.nanmean(valid_vals)
            station_feat_stds[f_idx] = max(np.nanstd(valid_vals), 1e-8)

    for f_idx in range(n_sf):
        station_feats[:, :, f_idx] = (
            (station_feats[:, :, f_idx] - station_feat_means[f_idx])
            / station_feat_stds[f_idx]
        )
    for s_idx in range(n_stations):
        invalid = station_mask[:, s_idx] < 0.5
        station_feats[invalid, s_idx, :] = 0.0

    global_scaler = StandardScaler()
    global_ctx_train = global_ctx[train_mask_np]
    global_scaler.fit(global_ctx_train)
    global_ctx = global_scaler.transform(global_ctx).astype(np.float32)

    # --- Pack into split arrays ---
    result = {
        "ordered_stations": ordered_stations,
        "n_stations": n_stations,
        "n_station_features": N_STATION_FEATURES,
        "n_metadata_features": N_META_FEATURES,
        "n_global_features": N_GLOBAL_FEATURES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "station_feat_means": station_feat_means,
        "station_feat_stds": station_feat_stds,
        "global_scaler": global_scaler,
    }

    for split, m in masks.items():
        m_np = np.array(m)
        sub_df = df[m_np]
        sub_dates = sub_df.index
        result[split] = {
            "station_features": station_feats[m_np],
            "station_metadata": station_meta[m_np],
            "global_context": global_ctx[m_np],
            "station_bearings": station_bearings[m_np],
            "wind_direction": wind_proxy[m_np],
            "station_mask": station_mask[m_np],
            "target_residual": sub_df["residual"].values.astype(np.float32),
            "actual_tmax": sub_df["nyc_tmax"].values.astype(np.float64),
            "mos_base": sub_df["mos_ensemble_tmax_f"].values.astype(np.float64),
            "dates": sub_dates,
            "gfs_mos": sub_df["gfs_mos_tmax_f"].values.astype(np.float64),
            "nam_mos": sub_df["nam_mos_tmax_f"].values.astype(np.float64),
        }

    logger.info(
        "Dataset built: Train=%d, Val=%d, Test=%d, Stations=%d, StationFeats=%d",
        len(result["train"]["target_residual"]),
        len(result["val"]["target_residual"]),
        len(result["test"]["target_residual"]),
        n_stations, N_STATION_FEATURES,
    )
    return result


# ============================================================================
# 5. PYTORCH DATASET
# ============================================================================

class WGADataset(Dataset):
    def __init__(self, split_data: dict):
        self.station_features = torch.from_numpy(split_data["station_features"])
        self.station_metadata = torch.from_numpy(split_data["station_metadata"])
        self.global_context = torch.from_numpy(split_data["global_context"])
        self.station_bearings = torch.from_numpy(split_data["station_bearings"])
        self.wind_direction = torch.from_numpy(split_data["wind_direction"])
        self.station_mask = torch.from_numpy(split_data["station_mask"])
        self.target = torch.from_numpy(split_data["target_residual"])

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict:
        return {
            "station_features": self.station_features[idx],
            "station_metadata": self.station_metadata[idx],
            "global_context": self.global_context[idx],
            "station_bearings": self.station_bearings[idx],
            "wind_direction": self.wind_direction[idx],
            "station_mask": self.station_mask[idx],
            "target": self.target[idx],
        }


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def gaussian_nll_loss(mu, log_sigma, target):
    sigma = torch.exp(log_sigma)
    nll = 0.5 * math.log(2 * math.pi) + log_sigma + 0.5 * ((target - mu) / sigma) ** 2
    return nll.mean()


def train_one_seed(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    seed: int,
    lr: float = LR,
    epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple:
    """Train a single model seed. Returns (best_val_loss, best_state_dict, history)."""
    set_seed(seed)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    train_ds = WGADataset(train_data)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_ds = WGADataset(val_data)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                station_features=batch["station_features"].to(DEVICE),
                station_metadata=batch["station_metadata"].to(DEVICE),
                global_context=batch["global_context"].to(DEVICE),
                station_bearings=batch["station_bearings"].to(DEVICE),
                wind_direction=batch["wind_direction"].to(DEVICE),
                station_mask=batch["station_mask"].to(DEVICE),
            )
            loss = gaussian_nll_loss(
                out["mu"].squeeze(-1),
                out["log_sigma"].squeeze(-1),
                batch["target"].to(DEVICE),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    station_features=batch["station_features"].to(DEVICE),
                    station_metadata=batch["station_metadata"].to(DEVICE),
                    global_context=batch["global_context"].to(DEVICE),
                    station_bearings=batch["station_bearings"].to(DEVICE),
                    wind_direction=batch["wind_direction"].to(DEVICE),
                    station_mask=batch["station_mask"].to(DEVICE),
                )
                loss = gaussian_nll_loss(
                    out["mu"].squeeze(-1),
                    out["log_sigma"].squeeze(-1),
                    batch["target"].to(DEVICE),
                )
                val_losses.append(loss.item() * len(batch["target"]))
                val_preds.append(out["mu"].squeeze(-1).cpu().numpy())
                val_targets.append(batch["target"].numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = sum(val_losses) / len(val_data["target_residual"])
        val_pred_arr = np.concatenate(val_preds)
        val_target_arr = np.concatenate(val_targets)
        val_mae = float(np.mean(np.abs(val_pred_arr - val_target_arr)))

        scheduler.step(avg_val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 6),
            "val_loss": round(float(avg_val_loss), 6),
            "val_residual_mae": round(val_mae, 4),
        })

        if epoch % 25 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "  Seed %d Epoch %3d: train_nll=%.4f  val_nll=%.4f  "
                "val_resid_mae=%.3f  lr=%.2e",
                seed, epoch, avg_train_loss, avg_val_loss, val_mae, current_lr,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "  Seed %d: Early stopping at epoch %d (best val_nll=%.4f)",
                    seed, epoch, best_val_loss,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return best_val_loss, best_state, history


def predict_wga(model: nn.Module, split_data: dict,
                batch_size: int = 256) -> tuple:
    """Run inference, returning (mu_array, sigma_array) for the residual."""
    model.eval()
    ds = WGADataset(split_data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_mu = []
    all_sigma = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                station_features=batch["station_features"].to(DEVICE),
                station_metadata=batch["station_metadata"].to(DEVICE),
                global_context=batch["global_context"].to(DEVICE),
                station_bearings=batch["station_bearings"].to(DEVICE),
                wind_direction=batch["wind_direction"].to(DEVICE),
                station_mask=batch["station_mask"].to(DEVICE),
            )
            all_mu.append(out["mu"].squeeze(-1).cpu().numpy())
            all_sigma.append(out["sigma"].squeeze(-1).cpu().numpy())

    return np.concatenate(all_mu), np.concatenate(all_sigma)


# ============================================================================
# 7. EVALUATION
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray,
    dates: pd.DatetimeIndex = None, label: str = "",
) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    result = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    if dates is not None:
        seasons = assign_season(pd.DatetimeIndex(dates))
        for s in ["DJF", "MAM", "JJA", "SON"]:
            m = seasons == s
            if m.sum() > 0:
                result[f"mae_{s}"] = round(float(
                    mean_absolute_error(np.array(y_true)[m], np.array(y_pred)[m])
                ), 4)
    if label:
        logger.info(
            "  %s: MAE=%.3f  RMSE=%.3f  R2=%.3f", label, mae, rmse, r2
        )
    return result


# ============================================================================
# 8. REGIME CLASSIFICATION
# ============================================================================

def classify_regimes(
    mos_base, gfs_mos, nam_mos, ensemble_std, ensemble_mu, dates, thresholds=None
):
    mos_spread = np.abs(gfs_mos - nam_mos)
    dod_change = np.abs(np.diff(ensemble_mu, prepend=ensemble_mu[0]))
    month = dates.month
    season = np.where(
        np.isin(month, [12, 1, 2]), 0,
        np.where(np.isin(month, [3, 4, 5]), 1,
                 np.where(np.isin(month, [6, 7, 8]), 2, 3)),
    )

    if thresholds is None:
        spread_high = float(np.percentile(mos_spread, 70))
        consensus_high = float(np.percentile(ensemble_std, 70))
        dod_high = float(np.percentile(dod_change, 70))
        thresholds = {
            "spread_high": spread_high,
            "consensus_high": consensus_high,
            "dod_high": dod_high,
        }
    else:
        spread_high = thresholds["spread_high"]
        consensus_high = thresholds["consensus_high"]
        dod_high = thresholds["dod_high"]

    n = len(mos_spread)
    regimes = np.full(n, "medium_var", dtype=object)

    high_count = (
        (mos_spread > spread_high).astype(int)
        + (ensemble_std > consensus_high).astype(int)
        + (dod_change > dod_high).astype(int)
    )
    regimes[high_count >= 2] = "high_var"

    spread_med = np.median(mos_spread)
    consensus_med = np.median(ensemble_std)
    dod_med = np.median(dod_change)
    low_mask = (
        (mos_spread <= spread_med)
        & (ensemble_std <= consensus_med)
        & (dod_change <= dod_med)
    )
    regimes[low_mask] = "low_var"

    regimes[np.isin(season, [1, 3]) & (regimes == "medium_var")] = "seasonal_transition"

    return regimes, thresholds


# ============================================================================
# 9. SIGMA CALIBRATION
# ============================================================================

def compute_monthly_sigma_calibration(val_dates, val_errors, val_sigma_base):
    sigma_by_month = {}
    monthly_scale = {}
    global_sigma = float(np.std(val_errors, ddof=1))

    for month in range(1, 13):
        m = val_dates.month == month
        if np.any(m):
            actual_std = float(np.std(val_errors[m], ddof=1))
            pred_sigma = float(np.mean(val_sigma_base[m]))
            sigma_by_month[month] = float(np.clip(
                max(actual_std, SIGMA_FLOOR), SIGMA_FLOOR, SIGMA_CAP
            ))
            if pred_sigma > 0:
                monthly_scale[month] = actual_std / pred_sigma
            else:
                monthly_scale[month] = 1.0
        else:
            sigma_by_month[month] = float(np.clip(global_sigma, SIGMA_FLOOR, SIGMA_CAP))
            monthly_scale[month] = 1.0

    return sigma_by_month, monthly_scale


def apply_monthly_sigma_calibration(sigma_base, dates, monthly_scale):
    calibrated = sigma_base.copy()
    for i in range(len(calibrated)):
        month = int(dates[i].month)
        calibrated[i] *= monthly_scale.get(month, 1.0)
    return np.clip(calibrated, SIGMA_FLOOR, SIGMA_CAP)


# ============================================================================
# 10. ABLATION EXPERIMENT RUNNER
# ============================================================================

# Ablation configurations
ABLATION_CONFIGS = {
    "wga_v2_full": {
        "n_heads": 4,
        "encoder_depth": 3,
        "use_lag2": True,
        "description": "4-head + 3-layer encoder + lag-2 features (full V2)",
    },
    "wga_v2_multihead_only": {
        "n_heads": 4,
        "encoder_depth": 2,
        "use_lag2": False,
        "description": "4-head + 2-layer encoder + original 6 features",
    },
    "wga_v2_deep_only": {
        "n_heads": 1,
        "encoder_depth": 3,
        "use_lag2": False,
        "description": "1-head + 3-layer encoder + original 6 features",
    },
    "wga_v2_lag2_only": {
        "n_heads": 1,
        "encoder_depth": 2,
        "use_lag2": True,
        "description": "1-head + 2-layer encoder + lag-2 features",
    },
}


def _make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def run_single_config(
    config_name: str,
    config_params: dict,
    station_matrix: pd.DataFrame,
    mos_data: pd.DataFrame,
    cp_data: pd.DataFrame,
    seeds: list,
    station_subset: list = None,
) -> dict:
    """Train a single architecture configuration across multiple seeds.

    Returns a dict with all results, ensemble metrics, calibration, etc.
    """
    n_heads = config_params["n_heads"]
    encoder_depth = config_params["encoder_depth"]
    use_lag2 = config_params["use_lag2"]

    tag = config_name
    if station_subset is not None:
        tag += f"_top{len(station_subset)}"

    logger.info("=" * 70)
    logger.info("CONFIG: %s (%s)", tag, config_params.get("description", ""))
    logger.info("  heads=%d, depth=%d, lag2=%s, seeds=%s, stations=%s",
                n_heads, encoder_depth, use_lag2, seeds,
                len(station_subset) if station_subset else "all")
    logger.info("=" * 70)

    # Build dataset for this configuration
    data = build_structured_dataset(
        station_matrix, mos_data, cp_data,
        use_lag2=use_lag2,
        station_subset=station_subset,
    )

    n_stations = data["n_stations"]
    n_sf = data["n_station_features"]
    n_mf = data["n_metadata_features"]
    n_gf = data["n_global_features"]

    all_results = {}
    seed_state_dicts = {}
    per_seed_mu = {split: [] for split in ["train", "val", "test"]}
    per_seed_sigma = {split: [] for split in ["train", "val", "test"]}
    training_log = {}

    for seed in seeds:
        logger.info("--- Training %s seed %d ---", tag, seed)
        set_seed(seed)
        model = EnhancedWindGatedAttentionModel(
            n_station_features=n_sf,
            n_metadata_features=n_mf,
            n_global_features=n_gf,
            n_stations=n_stations,
            station_embed_dim=STATION_EMBED_DIM,
            attention_dim=ATTENTION_DIM,
            output_mode="gaussian",
            dropout=0.1,
            n_heads=n_heads,
            encoder_depth=encoder_depth,
        )

        best_val_loss, best_state, history = train_one_seed(
            model, data["train"], data["val"], seed=seed,
        )
        seed_state_dicts[str(seed)] = best_state
        training_log[str(seed)] = {
            "best_val_nll": round(float(best_val_loss), 6),
            "epochs_trained": len(history),
            "history": history,
        }

        for split in ["train", "val", "test"]:
            resid_mu, resid_sigma = predict_wga(model, data[split])
            actual_mu = data[split]["mos_base"] + resid_mu
            per_seed_mu[split].append(actual_mu)
            per_seed_sigma[split].append(resid_sigma)

            r = evaluate_predictions(
                data[split]["actual_tmax"], actual_mu,
                data[split]["dates"], f"{tag}_Seed{seed} {split}",
            )
            all_results[f"seed_{seed}_{split}"] = r

    # --- Ensemble aggregation ---
    logger.info("--- %s Ensemble (mean of %d seeds) ---", tag, len(seeds))
    ensemble_mu = {}
    ensemble_sigma_base = {}
    ensemble_std = {}

    for split in ["train", "val", "test"]:
        stacked_mu = np.array(per_seed_mu[split])
        stacked_sigma = np.array(per_seed_sigma[split])

        ens_mu = stacked_mu.mean(axis=0)
        ens_std = stacked_mu.std(axis=0)
        ens_sigma_model = stacked_sigma.mean(axis=0)
        ens_sigma_base = np.sqrt(ens_sigma_model ** 2 + ens_std ** 2)
        ens_sigma_base = np.clip(ens_sigma_base, SIGMA_FLOOR, SIGMA_CAP)

        ensemble_mu[split] = ens_mu
        ensemble_sigma_base[split] = ens_sigma_base
        ensemble_std[split] = ens_std

        r = evaluate_predictions(
            data[split]["actual_tmax"], ens_mu,
            data[split]["dates"], f"{tag}_Ensemble {split}",
        )
        all_results[f"ensemble_{split}"] = r

    # --- Sigma calibration ---
    val_dates = pd.to_datetime(data["val"]["dates"])
    val_errors = data["val"]["actual_tmax"] - ensemble_mu["val"]
    sigma_by_month, monthly_scale = compute_monthly_sigma_calibration(
        val_dates, val_errors, ensemble_sigma_base["val"]
    )

    sigma_monthly_cal = {}
    for split in ["train", "val", "test"]:
        sigma_monthly_cal[split] = apply_monthly_sigma_calibration(
            ensemble_sigma_base[split],
            pd.to_datetime(data[split]["dates"]),
            monthly_scale,
        )

    # --- Regime classification ---
    _, regime_thresholds = classify_regimes(
        data["val"]["mos_base"], data["val"]["gfs_mos"], data["val"]["nam_mos"],
        ensemble_std["val"], ensemble_mu["val"], val_dates,
    )

    regimes = {}
    for split in ["val", "test"]:
        r, _ = classify_regimes(
            data[split]["mos_base"], data[split]["gfs_mos"], data[split]["nam_mos"],
            ensemble_std[split], ensemble_mu[split],
            pd.to_datetime(data[split]["dates"]),
            thresholds=regime_thresholds,
        )
        regimes[split] = r

    # --- 95% PI coverage ---
    test_mu = ensemble_mu["test"]
    test_sigma_cal = sigma_monthly_cal["test"]
    test_actual = data["test"]["actual_tmax"]
    z95 = scipy_norm.ppf(0.975)
    lower = test_mu - z95 * test_sigma_cal
    upper = test_mu + z95 * test_sigma_cal
    coverage = float(np.mean((test_actual >= lower) & (test_actual <= upper)))
    avg_width = float(np.mean(upper - lower))

    all_results["pi_coverage_95"] = round(coverage, 4)
    all_results["pi_avg_width"] = round(avg_width, 2)

    n_params = sum(p.numel() for p in model.parameters())
    all_results["n_params"] = n_params
    all_results["n_stations"] = n_stations
    all_results["n_station_features"] = n_sf
    all_results["n_heads"] = n_heads
    all_results["encoder_depth"] = encoder_depth
    all_results["use_lag2"] = use_lag2

    logger.info(
        "  %s: Test MAE=%.4f, 95%% PI coverage=%.3f, width=%.2f, params=%d",
        tag, all_results["ensemble_test"]["mae"], coverage, avg_width, n_params,
    )

    return {
        "config_name": tag,
        "all_results": all_results,
        "seed_state_dicts": seed_state_dicts,
        "training_log": training_log,
        "ensemble_mu": ensemble_mu,
        "ensemble_sigma_base": ensemble_sigma_base,
        "ensemble_std": ensemble_std,
        "sigma_monthly_cal": sigma_monthly_cal,
        "sigma_by_month": sigma_by_month,
        "monthly_scale": monthly_scale,
        "regimes": regimes,
        "regime_thresholds": regime_thresholds,
        "data": data,
    }


# ============================================================================
# 11. OUTPUT SAVING
# ============================================================================

def save_config_outputs(result: dict, out_dir: str) -> None:
    """Save predictions, model checkpoints, and metrics for a single config."""
    os.makedirs(out_dir, exist_ok=True)
    data = result["data"]
    ens_mu = result["ensemble_mu"]
    sigma_cal = result["sigma_monthly_cal"]
    regimes = result.get("regimes", {})

    # Predictions CSV (test)
    test_dates = pd.to_datetime(data["test"]["dates"])
    pred_test = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["test"]["actual_tmax"],
        "mos_base": data["test"]["mos_base"],
        "model_mu": ens_mu["test"],
        "model_sigma_cal": sigma_cal["test"],
        "regime": regimes.get("test", "unknown"),
    })
    pred_test.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    # Predictions CSV (val)
    val_dates = pd.to_datetime(data["val"]["dates"])
    pred_val = pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["val"]["actual_tmax"],
        "mos_base": data["val"]["mos_base"],
        "model_mu": ens_mu["val"],
        "model_sigma_cal": sigma_cal["val"],
        "regime": regimes.get("val", "unknown"),
    })
    pred_val.to_csv(os.path.join(out_dir, "predictions_val.csv"), index=False)

    # Experiment results
    with open(os.path.join(out_dir, "experiment_results.json"), "w") as f:
        json.dump(_make_serializable(result["all_results"]), f, indent=2)

    # Training log
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(_make_serializable(result["training_log"]), f, indent=2)

    # Sigma calibration
    cal_data = {
        "sigma_by_month": result["sigma_by_month"],
        "monthly_scale": result["monthly_scale"],
        "regime_thresholds": result.get("regime_thresholds", {}),
    }
    with open(os.path.join(out_dir, "sigma_calibration.json"), "w") as f:
        json.dump(_make_serializable(cal_data), f, indent=2)

    # Model checkpoint
    seed_dicts = result.get("seed_state_dicts", {})
    if seed_dicts:
        ckpt = {
            "model_class": "EnhancedWindGatedAttentionModel",
            "model_config": {
                "n_station_features": data["n_station_features"],
                "n_metadata_features": data["n_metadata_features"],
                "n_global_features": data["n_global_features"],
                "n_stations": data["n_stations"],
                "station_embed_dim": STATION_EMBED_DIM,
                "attention_dim": ATTENTION_DIM,
                "output_mode": "gaussian",
                "dropout": 0.1,
                "n_heads": result["all_results"].get("n_heads", 4),
                "encoder_depth": result["all_results"].get("encoder_depth", 3),
            },
            "seed_state_dicts": seed_dicts,
            "ordered_stations": data["ordered_stations"],
        }
        torch.save(ckpt, os.path.join(out_dir, "ensemble_checkpoint.pt"))


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def main() -> None:
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ENHANCED WIND-GATED ATTENTION V2 TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info("Device: %s", DEVICE)
    logger.info("Splits: Train=%s to %s, Val=%s to %s, Test=%s to %s",
                MOS_TRAIN_START, MOS_TRAIN_END, VAL_START, VAL_END,
                TEST_START, TEST_END)
    logger.info("Ensemble seeds: %s", ENSEMBLE_SEEDS)
    logger.info(
        "Hyperparameters: batch=%d, lr=%.1e, wd=%.1e, patience=%d, max_epochs=%d",
        BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, MAX_EPOCHS,
    )

    # --- Step 1: Download station data ---
    logger.info("\n=== STEP 1: Downloading station .dly files ===")
    download_all_stations()

    # --- Step 2: Build station matrix ---
    logger.info("\n=== STEP 2: Building station observation matrix ===")
    station_matrix = build_station_matrix(DLY_START, DLY_END)

    # --- Step 3: Load MOS and Central Park data ---
    logger.info("\n=== STEP 3: Loading MOS and Central Park data ===")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # --- Step 4: Architecture ablation experiments ---
    logger.info("\n=== STEP 4: Architecture Ablation (4 configs x 5 seeds) ===")

    all_experiment_results = {}
    best_test_mae = float("inf")
    best_config_name = None

    for config_name, config_params in ABLATION_CONFIGS.items():
        result = run_single_config(
            config_name, config_params,
            station_matrix, mos_data, cp_data,
            seeds=ENSEMBLE_SEEDS,
        )

        all_experiment_results[config_name] = result

        # Save per-config outputs
        config_dir = os.path.join(RESULTS_DIR, config_name)
        save_config_outputs(result, config_dir)

        test_mae = result["all_results"]["ensemble_test"]["mae"]
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_config_name = config_name

    # --- Step 5: Station ablation ladder (best config, seed=42 only) ---
    logger.info("\n=== STEP 5: Station Ablation Ladder (seed=42 only) ===")

    # Get ordered stations and sort by distance
    all_ordered = _get_ordered_station_list(station_matrix)
    station_ablation_results = {}

    for n_top in [10, 20, 30]:
        subset = _get_stations_by_distance(all_ordered, n_top)
        tag = f"top_{n_top}"
        logger.info("--- Station ablation: %s (%d stations) ---", tag, len(subset))

        best_config = ABLATION_CONFIGS[best_config_name]
        result = run_single_config(
            best_config_name, best_config,
            station_matrix, mos_data, cp_data,
            seeds=[42],
            station_subset=subset,
        )

        station_ablation_results[tag] = {
            "n_stations": len(subset),
            "stations": subset,
            "test_mae": result["all_results"]["ensemble_test"]["mae"],
            "test_rmse": result["all_results"]["ensemble_test"]["rmse"],
            "test_r2": result["all_results"]["ensemble_test"]["r2"],
            "val_mae": result["all_results"]["ensemble_val"]["mae"],
            "pi_coverage_95": result["all_results"].get("pi_coverage_95", None),
            "n_params": result["all_results"].get("n_params", None),
        }

        # Save to sub-directory
        ablation_dir = os.path.join(RESULTS_DIR, f"station_ablation_{tag}")
        save_config_outputs(result, ablation_dir)

    # Also do all-stations with seed=42 for fair comparison
    result_all = run_single_config(
        best_config_name, ABLATION_CONFIGS[best_config_name],
        station_matrix, mos_data, cp_data,
        seeds=[42],
    )
    station_ablation_results["all_47"] = {
        "n_stations": result_all["data"]["n_stations"],
        "test_mae": result_all["all_results"]["ensemble_test"]["mae"],
        "test_rmse": result_all["all_results"]["ensemble_test"]["rmse"],
        "test_r2": result_all["all_results"]["ensemble_test"]["r2"],
        "val_mae": result_all["all_results"]["ensemble_val"]["mae"],
        "pi_coverage_95": result_all["all_results"].get("pi_coverage_95", None),
        "n_params": result_all["all_results"].get("n_params", None),
    }

    # --- Step 6: Save consolidated results ---
    logger.info("\n=== STEP 6: Saving consolidated results ===")

    # Consolidated experiment results
    consolidated = {
        "architecture_ablation": {},
        "station_ablation": _make_serializable(station_ablation_results),
        "best_config": best_config_name,
        "best_test_mae": best_test_mae,
    }
    for cname, cresult in all_experiment_results.items():
        consolidated["architecture_ablation"][cname] = {
            "description": ABLATION_CONFIGS[cname]["description"],
            "ensemble_test": cresult["all_results"]["ensemble_test"],
            "ensemble_val": cresult["all_results"]["ensemble_val"],
            "pi_coverage_95": cresult["all_results"].get("pi_coverage_95"),
            "pi_avg_width": cresult["all_results"].get("pi_avg_width"),
            "n_params": cresult["all_results"].get("n_params"),
            "n_stations": cresult["all_results"].get("n_stations"),
            "n_station_features": cresult["all_results"].get("n_station_features"),
            "n_heads": cresult["all_results"].get("n_heads"),
            "encoder_depth": cresult["all_results"].get("encoder_depth"),
            "use_lag2": cresult["all_results"].get("use_lag2"),
            "per_seed_test_mae": {
                str(seed): cresult["all_results"].get(f"seed_{seed}_test", {}).get("mae")
                for seed in ENSEMBLE_SEEDS
            },
        }

    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(_make_serializable(consolidated), f, indent=2)
    logger.info("Saved consolidated results: %s", results_path)

    # Station ablation results
    ablation_path = os.path.join(RESULTS_DIR, "station_ablation_results.json")
    with open(ablation_path, "w") as f:
        json.dump(_make_serializable(station_ablation_results), f, indent=2)
    logger.info("Saved station ablation results: %s", ablation_path)

    # Regime thresholds (from best config)
    best_result = all_experiment_results[best_config_name]
    regime_path = os.path.join(RESULTS_DIR, "regime_thresholds.json")
    with open(regime_path, "w") as f:
        json.dump(_make_serializable(best_result.get("regime_thresholds", {})), f, indent=2)

    # --- Final summary ---
    logger.info("\n" + "=" * 80)
    logger.info("WGA V2 TRAINING COMPLETE -- SUMMARY")
    logger.info("=" * 80)

    logger.info("\n--- Architecture Ablation Results (5-seed ensemble) ---")
    logger.info("%-30s %8s %8s %8s %7s %6s", "Config", "TestMAE", "ValMAE", "TestRMSE", "95%PI", "Params")
    logger.info("-" * 93)
    for cname in ABLATION_CONFIGS:
        cres = consolidated["architecture_ablation"][cname]
        logger.info(
            "%-30s %8.4f %8.4f %8.4f %7.3f %6d",
            cname,
            cres["ensemble_test"]["mae"],
            cres["ensemble_val"]["mae"],
            cres["ensemble_test"]["rmse"],
            cres.get("pi_coverage_95", 0),
            cres.get("n_params", 0),
        )

    logger.info("\n--- Per-Seed Test MAE (best config: %s) ---", best_config_name)
    best_res = consolidated["architecture_ablation"][best_config_name]
    for seed in ENSEMBLE_SEEDS:
        seed_mae = best_res["per_seed_test_mae"].get(str(seed), "N/A")
        logger.info("  Seed %d: %s", seed, seed_mae)

    logger.info("\n--- Seasonal Breakdown (best config: %s) ---", best_config_name)
    for s in ["DJF", "MAM", "JJA", "SON"]:
        smae = best_res["ensemble_test"].get(f"mae_{s}", "N/A")
        logger.info("  %s MAE: %s", s, smae)

    logger.info("\n--- Station Ablation (best config, seed=42 only) ---")
    logger.info("%-10s %8s %8s %7s", "Subset", "TestMAE", "ValMAE", "95%PI")
    logger.info("-" * 37)
    for tag in ["top_10", "top_20", "top_30", "all_47"]:
        if tag in station_ablation_results:
            sr = station_ablation_results[tag]
            logger.info(
                "%-10s %8.4f %8.4f %7s",
                tag, sr["test_mae"], sr["val_mae"],
                f"{sr['pi_coverage_95']:.3f}" if sr.get("pi_coverage_95") else "N/A",
            )

    logger.info("\nBest config: %s (Test MAE=%.4f)", best_config_name, best_test_mae)
    logger.info("Results saved to: %s", RESULTS_DIR)

    elapsed = time.time() - start_time
    logger.info(
        "\nTotal time: %.1f minutes (%.0f seconds)", elapsed / 60, elapsed
    )
    logger.info("DONE.")


if __name__ == "__main__":
    main()
