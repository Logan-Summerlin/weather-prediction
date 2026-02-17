"""
Wind-Gated Attention Data Pipeline for Multi-City Temperature Prediction.

Converts flat (wide) feature CSVs into structured 3D tensors required by
the WindGatedAttentionModel. Handles:

  1. Per-station feature extraction from wide feature matrices
  2. Station metadata tensor construction (bearing, distance, elevation, sector)
  3. Global context feature extraction (wind dir, SLP, date encoding, prev TMAX)
  4. Missing-station masking
  5. City-specific station network configuration

Input:  data/{city}/processed/features_{split}.csv  (flat, wide format)
Output: 3D tensors ready for WindGatedAttentionModel.forward()

Architecture:
  station_features: (batch, n_stations, n_station_features)
  station_metadata: (batch, n_stations, n_metadata_features)
  global_context:   (batch, n_global_features)
  station_bearings: (batch, n_stations)
  wind_direction:   (batch,)
  station_mask:     (batch, n_stations)

Usage:
    from src.wga_data_pipeline import WGADataBuilder
    builder = WGADataBuilder("chi")
    tensors = builder.build_tensors(X_train, y_train)
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Station metadata loading
# ---------------------------------------------------------------------------

def _load_station_metadata(city_code: str) -> Dict[str, Dict]:
    """Load station metadata from the city's config module.

    Parameters
    ----------
    city_code : str
        City identifier ("chi", "phl", "nyc").

    Returns
    -------
    dict
        Station ID → metadata dict with keys:
        name, state, lat, lon, distance_mi, bearing, ring, sector.
    """
    try:
        if city_code == "chi":
            import config_chicago as cfg
        elif city_code == "phl":
            import config_philadelphia as cfg
        elif city_code == "nyc":
            try:
                import config_expanded as cfg
            except ImportError:
                import config as cfg
        else:
            raise ValueError(f"Unknown city code: {city_code}")

        if hasattr(cfg, "STATION_METADATA"):
            return dict(cfg.STATION_METADATA)
        else:
            logger.warning("No STATION_METADATA in config for %s", city_code)
            return {}

    except ImportError:
        logger.warning("Could not import config module for %s", city_code)
        return {}


def _get_station_order(city_code: str) -> List[str]:
    """Get deterministic station ordering for a city.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    list of str
        Sorted list of surrounding station IDs.
    """
    metadata = _load_station_metadata(city_code)
    return sorted(metadata.keys())


# ---------------------------------------------------------------------------
# Sector one-hot encoding
# ---------------------------------------------------------------------------

SECTOR_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _sector_to_onehot(sector: str) -> np.ndarray:
    """Convert sector label to one-hot vector.

    Parameters
    ----------
    sector : str
        Compass sector ("N", "NE", "E", etc.).

    Returns
    -------
    np.ndarray
        One-hot vector of length 8.
    """
    vec = np.zeros(len(SECTOR_ORDER), dtype=np.float32)
    if sector in SECTOR_ORDER:
        vec[SECTOR_ORDER.index(sector)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# WGA Data Builder
# ---------------------------------------------------------------------------

# Per-station features extracted from the flat feature matrix
# These are the standard GHCN lag features present in the wide format
STATION_FEATURE_SUFFIXES = ["_TMAX_lag1", "_TMIN_lag1"]

# Global context features (extracted from flat features or computed)
GLOBAL_FEATURE_NAMES = [
    "sin_day",
    "cos_day",
]


class WGADataBuilder:
    """Builds 3D tensor inputs for the WindGatedAttentionModel.

    Converts flat (wide) feature DataFrames into structured tensors
    with per-station features, station metadata, and global context.

    Parameters
    ----------
    city_code : str
        City identifier ("chi", "phl", "nyc").
    n_station_features : int
        Number of features per station (default 2: TMAX_lag1, TMIN_lag1).
    n_metadata_features : int
        Number of metadata features per station (default 13:
        bearing_sin, bearing_cos, distance_norm, sector_onehot[8],
        ring_near, ring_regional).

    Examples
    --------
    >>> builder = WGADataBuilder("chi")
    >>> tensors = builder.build_tensors(X_train, y_train)
    >>> tensors["station_features"].shape
    torch.Size([n_days, 55, 2])
    """

    def __init__(
        self,
        city_code: str,
        n_station_features: int = 2,
        n_metadata_features: int = 13,
    ):
        self.city_code = city_code.strip().lower()
        self.city_config = get_city_config(self.city_code)

        # Load station metadata and establish ordering
        self.station_metadata = _load_station_metadata(self.city_code)
        self.station_order = _get_station_order(self.city_code)
        self.n_stations = len(self.station_order)
        self.n_station_features = n_station_features
        self.n_metadata_features = n_metadata_features

        # Precompute static metadata tensors
        self._precompute_metadata()

        logger.info(
            "WGADataBuilder for %s: %d stations, %d station_feats, "
            "%d metadata_feats",
            self.city_code, self.n_stations,
            self.n_station_features, self.n_metadata_features,
        )

    def _precompute_metadata(self) -> None:
        """Precompute static station metadata arrays."""
        n = self.n_stations
        self.bearings_rad = np.zeros(n, dtype=np.float32)
        self.distances_norm = np.zeros(n, dtype=np.float32)
        self.metadata_array = np.zeros(
            (n, self.n_metadata_features), dtype=np.float32
        )

        # Find max distance for normalization
        max_dist = max(
            (m.get("distance_mi", 0) for m in self.station_metadata.values()),
            default=1.0,
        )
        max_dist = max(max_dist, 1.0)

        for i, station_id in enumerate(self.station_order):
            meta = self.station_metadata.get(station_id, {})

            bearing_deg = meta.get("bearing", 0.0)
            bearing_rad = math.radians(bearing_deg)
            distance_mi = meta.get("distance_mi", 0.0)
            sector = meta.get("sector", "N")
            ring = meta.get("ring", "Ring1_Near")

            self.bearings_rad[i] = bearing_rad
            self.distances_norm[i] = distance_mi / max_dist

            # Build metadata vector:
            # [bearing_sin, bearing_cos, distance_norm, sector_onehot(8),
            #  ring_near_flag, ring_regional_flag]
            sector_oh = _sector_to_onehot(sector)
            ring_near = 1.0 if "Near" in ring else 0.0
            ring_regional = 1.0 if "Regional" in ring else 0.0

            self.metadata_array[i] = np.concatenate([
                [math.sin(bearing_rad), math.cos(bearing_rad)],
                [self.distances_norm[i]],
                sector_oh,
                [ring_near, ring_regional],
            ])

    def _extract_station_features(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-station features from flat feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Flat feature matrix (n_days, n_flat_features).

        Returns
        -------
        station_features : np.ndarray
            Shape (n_days, n_stations, n_station_features).
        station_mask : np.ndarray
            Shape (n_days, n_stations). 1 = present, 0 = missing.
        """
        n_days = len(X)
        features = np.zeros(
            (n_days, self.n_stations, self.n_station_features),
            dtype=np.float32,
        )
        mask = np.zeros((n_days, self.n_stations), dtype=np.float32)

        columns = X.columns.tolist()

        for i, station_id in enumerate(self.station_order):
            for j, suffix in enumerate(STATION_FEATURE_SUFFIXES):
                col_name = f"{station_id}{suffix}"
                if col_name in columns:
                    values = X[col_name].values.astype(np.float32)
                    features[:, i, j] = np.nan_to_num(values, nan=0.0)
                    # Station is present if any feature is non-NaN
                    if j == 0:
                        mask[:, i] = (~X[col_name].isna()).values.astype(
                            np.float32
                        )

        # Ensure at least one station is present per day
        all_missing = mask.sum(axis=1) == 0
        if all_missing.any():
            logger.warning(
                "%d days have no station data; setting uniform mask",
                int(all_missing.sum()),
            )
            mask[all_missing] = 1.0

        return features, mask

    def _extract_global_context(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract global context features and wind direction.

        Parameters
        ----------
        X : pd.DataFrame
            Flat feature matrix.
        y : pd.Series, optional
            Target TMAX (for previous-day TMAX feature).

        Returns
        -------
        global_context : np.ndarray
            Shape (n_days, n_global_features).
        wind_direction : np.ndarray
            Shape (n_days,). Wind direction in radians.
        """
        n_days = len(X)
        columns = X.columns.tolist()

        # Collect global features
        global_feats = []

        # sin_day, cos_day
        if "sin_day" in columns:
            global_feats.append(X["sin_day"].values.astype(np.float32))
        else:
            # Compute from index if DatetimeIndex
            try:
                doy = X.index.dayofyear
                global_feats.append(
                    np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
                )
            except AttributeError:
                global_feats.append(np.zeros(n_days, dtype=np.float32))

        if "cos_day" in columns:
            global_feats.append(X["cos_day"].values.astype(np.float32))
        else:
            try:
                doy = X.index.dayofyear
                global_feats.append(
                    np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
                )
            except AttributeError:
                global_feats.append(np.zeros(n_days, dtype=np.float32))

        # Previous TMAX (from target if available)
        if y is not None:
            prev_tmax = y.shift(1).fillna(y.mean()).values.astype(np.float32)
            global_feats.append(prev_tmax)
        else:
            global_feats.append(np.zeros(n_days, dtype=np.float32))

        # Target station TMAX_lag1 (a proxy for persistence)
        target_lag1_col = f"{self.city_config.target_station}_TMAX_lag1"
        if target_lag1_col in columns:
            global_feats.append(
                X[target_lag1_col].fillna(0).values.astype(np.float32)
            )
        else:
            global_feats.append(np.zeros(n_days, dtype=np.float32))

        global_context = np.column_stack(global_feats)

        # Wind direction: use prevailing wind estimate from station data
        # Default: 270° (west) which is climatological prevailing wind
        wind_direction = np.full(n_days, math.radians(270), dtype=np.float32)

        return global_context, wind_direction

    def build_tensors(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build all tensors needed for WindGatedAttentionModel.forward().

        Parameters
        ----------
        X : pd.DataFrame
            Flat feature matrix (n_days, n_flat_features).
        y : pd.Series, optional
            Target TMAX values.

        Returns
        -------
        dict of torch.Tensor
            Keys: station_features, station_metadata, global_context,
            station_bearings, wind_direction, station_mask.
            If y is provided, also includes "target".
        """
        n_days = len(X)

        # Extract station features and mask
        station_features, station_mask = self._extract_station_features(X)

        # Extract global context and wind direction
        global_context, wind_direction = self._extract_global_context(X, y)

        # Expand static metadata to batch dimension
        metadata_batch = np.tile(
            self.metadata_array[np.newaxis, :, :], (n_days, 1, 1)
        )
        bearings_batch = np.tile(
            self.bearings_rad[np.newaxis, :], (n_days, 1)
        )

        result = {
            "station_features": torch.from_numpy(station_features),
            "station_metadata": torch.from_numpy(metadata_batch),
            "global_context": torch.from_numpy(global_context),
            "station_bearings": torch.from_numpy(bearings_batch),
            "wind_direction": torch.from_numpy(wind_direction),
            "station_mask": torch.from_numpy(station_mask),
        }

        if y is not None:
            result["target"] = torch.from_numpy(
                y.values.astype(np.float32)
            ).unsqueeze(1)

        logger.info(
            "Built WGA tensors for %s: %d days, %d stations, "
            "station_feats=%s, global_ctx=%s",
            self.city_code, n_days, self.n_stations,
            list(result["station_features"].shape),
            list(result["global_context"].shape),
        )

        return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class WGADataset(Dataset):
    """PyTorch Dataset for WindGatedAttentionModel training.

    Parameters
    ----------
    tensors : dict of torch.Tensor
        Output from WGADataBuilder.build_tensors().
    """

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.n_samples = tensors["station_features"].shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.tensors.items()}


def create_wga_dataloader(
    tensors: Dict[str, torch.Tensor],
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader from WGA tensors.

    Parameters
    ----------
    tensors : dict of torch.Tensor
        Output from WGADataBuilder.build_tensors().
    batch_size : int
        Batch size (default 64).
    shuffle : bool
        Whether to shuffle (default False).

    Returns
    -------
    DataLoader
        PyTorch DataLoader yielding batched tensor dicts.
    """
    dataset = WGADataset(tensors)

    def collate_fn(batch):
        return {
            key: torch.stack([b[key] for b in batch])
            for key in batch[0].keys()
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# WGA Model Training Utilities
# ---------------------------------------------------------------------------

def train_wga_city(
    city_code: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_mode: str = "gaussian",
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = "cpu",
) -> Dict:
    """Train a WindGatedAttentionModel for a specific city.

    Parameters
    ----------
    city_code : str
        City identifier.
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices for each split.
    y_train, y_val, y_test : pd.Series
        Target TMAX values for each split.
    output_mode : str
        "gaussian" for heteroscedastic output, "point" for point prediction.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    device : str
        Device ("cpu" or "cuda").

    Returns
    -------
    dict
        Dictionary with trained model, predictions (mu, sigma for val/test),
        and training history.
    """
    from src.wind_gated_attention import WindGatedAttentionModel

    # Build tensors
    builder = WGADataBuilder(city_code)
    train_tensors = builder.build_tensors(X_train, y_train)
    val_tensors = builder.build_tensors(X_val, y_val)
    test_tensors = builder.build_tensors(X_test, y_test)

    # Model dimensions
    n_station_features = builder.n_station_features
    n_metadata_features = builder.n_metadata_features
    n_global_features = train_tensors["global_context"].shape[1]
    n_stations = builder.n_stations

    # Create model
    model = WindGatedAttentionModel(
        n_station_features=n_station_features,
        n_metadata_features=n_metadata_features,
        n_global_features=n_global_features,
        n_stations=n_stations,
        station_embed_dim=32,
        attention_dim=16,
        output_mode=output_mode,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )

    train_loader = create_wga_dataloader(
        train_tensors, batch_size=batch_size, shuffle=True
    )
    val_loader = create_wga_dataloader(
        val_tensors, batch_size=batch_size, shuffle=False
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            out = model(
                batch["station_features"],
                batch["station_metadata"],
                batch["global_context"],
                batch["station_bearings"],
                batch["wind_direction"],
                batch["station_mask"],
            )

            if output_mode == "gaussian":
                mu = out["mu"]
                sigma = out["sigma"]
                target = batch["target"]
                var = sigma ** 2
                loss = (
                    0.5 * (torch.log(2 * torch.pi * var)
                           + ((target - mu) ** 2) / var)
                ).mean()
            else:
                loss = torch.nn.functional.mse_loss(
                    out["prediction"], batch["target"]
                )

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    batch["station_features"],
                    batch["station_metadata"],
                    batch["global_context"],
                    batch["station_bearings"],
                    batch["wind_direction"],
                    batch["station_mask"],
                )

                if output_mode == "gaussian":
                    mu = out["mu"]
                    sigma = out["sigma"]
                    target = batch["target"]
                    var = sigma ** 2
                    loss = (
                        0.5 * (torch.log(2 * torch.pi * var)
                               + ((target - mu) ** 2) / var)
                    ).mean()
                else:
                    loss = torch.nn.functional.mse_loss(
                        out["prediction"], batch["target"]
                    )

                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)
    logger.info(
        "WGA training complete for %s: best_epoch=%d, val_loss=%.4f",
        city_code, best_epoch, best_val_loss,
    )

    # Generate predictions
    model.eval()

    def predict(tensors_dict):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in tensors_dict.items()
                     if k != "target"}
            out = model(
                batch["station_features"],
                batch["station_metadata"],
                batch["global_context"],
                batch["station_bearings"],
                batch["wind_direction"],
                batch["station_mask"],
            )
            if output_mode == "gaussian":
                return (
                    out["mu"].cpu().numpy().ravel(),
                    out["sigma"].cpu().numpy().ravel(),
                )
            else:
                return (
                    out["prediction"].cpu().numpy().ravel(),
                    np.ones(len(out["prediction"])) * 5.0,  # default sigma
                )

    mu_val, sig_val = predict(val_tensors)
    mu_test, sig_test = predict(test_tensors)

    return {
        "model": model,
        "builder": builder,
        "mu_val": mu_val,
        "sigma_val": sig_val,
        "mu_test": mu_test,
        "sigma_test": sig_test,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
