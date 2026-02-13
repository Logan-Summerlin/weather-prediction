#!/usr/bin/env python3
"""
Train Wind-Gated Attention + MDN model for NYC temperature prediction.

This script trains a WindGatedAttentionModel (from src/wind_gated_attention.py)
with a Gaussian (heteroscedastic) output head on structured station-level
features.  Unlike the flat feedforward pipeline, it preserves per-station
structure so that the attention mechanism can learn wind-conditioned station
weightings.

Architecture:
    - Per-station shared encoder maps raw station features to embeddings.
    - Scaled dot-product attention with wind-direction bias pools stations.
    - Gaussian output head predicts (mu, sigma) for delta-T residual.
    - Final prediction: mos_ensemble + predicted delta-T.

Splits:
    Train: 2000-06-01 to 2019-12-31
    Val:   2020-01-01 to 2022-12-31
    Test:  2023-01-01 to 2024-12-31

Outputs saved to: results/wga_mdn_model/
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
from src.wind_gated_attention import WindGatedAttentionModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger("train_wga_mdn")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_extended.csv")
ERA_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "mos_era_indicator.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "wga_mdn_model")

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
# 1. DATA LOADING  (reused from retrain_extended_validation.py)
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
    # Drop columns with < 80% completeness
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
# 2. FEATURE ENGINEERING HELPERS
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
# 3. STATION-LEVEL DATASET CONSTRUCTION
# ============================================================================

def _get_ordered_station_list(station_matrix: pd.DataFrame) -> list:
    """Return an ordered list of station IDs that have both TMAX and TMIN data."""
    tmax_sids = set()
    tmin_sids = set()
    for col in station_matrix.columns:
        if col.endswith("_TMAX"):
            tmax_sids.add(col.replace("_TMAX", ""))
        elif col.endswith("_TMIN"):
            tmin_sids.add(col.replace("_TMIN", ""))
    # Stations with both TMAX and TMIN
    both = tmax_sids & tmin_sids
    # Keep in the order they appear in ALL_SURROUNDING for determinism
    ordered = [sid for sid in ALL_SURROUNDING if sid in both]
    return ordered


def build_structured_dataset(
    station_matrix: pd.DataFrame,
    mos_data: pd.DataFrame,
    cp_data: pd.DataFrame,
) -> dict:
    """
    Build tensors preserving station-level structure for the WGA model.

    Returns a dict with keys: station_features, station_metadata, global_context,
    station_bearings, wind_direction, station_mask, target_residual,
    actual_tmax, mos_base, dates, gfs_mos, nam_mos, ordered_stations,
    n_station_features, n_metadata_features, n_global_features.

    Each value is a dict of {train, val, test} numpy arrays except for the
    scalar metadata fields.
    """
    logger.info("Building structured station-level dataset ...")

    # --- Determine ordered station list ---
    ordered_stations = _get_ordered_station_list(station_matrix)
    n_stations = len(ordered_stations)
    logger.info("Using %d stations with both TMAX and TMIN", n_stations)

    # --- Build lag-1 matrices for TMAX and TMIN ---
    tmax_cols = [f"{sid}_TMAX" for sid in ordered_stations]
    tmin_cols = [f"{sid}_TMIN" for sid in ordered_stations]
    tmax_lag1 = station_matrix[tmax_cols].shift(1)
    tmin_lag1 = station_matrix[tmin_cols].shift(1)
    tmax_lag2 = station_matrix[tmax_cols].shift(2)

    # --- Merge all data sources ---
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

    # MOS error memory (fitted only on available data up to each point;
    # we use shift(1) so only past information leaks in)
    mos_error = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
    mos_error_shifted = mos_error.shift(1)
    df["mos_error_7d"] = mos_error_shifted.rolling(window=7, min_periods=3).mean()

    # MOS spread
    df["gfs_nam_spread"] = (df["gfs_mos_tmax_f"] - df["nam_mos_tmax_f"]).abs()

    # Day length and solar elevation
    df["day_length"] = day_length_hours(NYC_LAT, doy)
    df["solar_elev_noon"] = solar_elevation_noon(NYC_LAT, doy)

    # --- Align all matrices to df.index ---
    tmax_l1 = tmax_lag1.reindex(df.index)
    tmin_l1 = tmin_lag1.reindex(df.index)
    tmax_l2 = tmax_lag2.reindex(df.index)

    # --- Station climatology (from training period only) ---
    train_mask_climo = (tmax_l1.index >= MOS_TRAIN_START) & (
        tmax_l1.index <= MOS_TRAIN_END
    )
    # Per-station mean TMAX over training set (used for anomaly features)
    station_tmax_train_mean = tmax_l1.loc[train_mask_climo].mean()

    # --- Build per-station feature arrays ---
    # Features per station:
    #   0: TMAX_lag1 (raw, will be scaled)
    #   1: TMIN_lag1
    #   2: delta_T = station_TMAX_lag1 - nyc_tmax_lag1
    #   3: diurnal_range = TMAX_lag1 - TMIN_lag1
    #   4: tmax_24h_change = TMAX_lag1 - TMAX_lag2
    #   5: station_anomaly = TMAX_lag1 - station_climatological_mean
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

        # Fill invalid with 0 (they will be masked in attention)
        tmax_safe = np.where(valid, tmax_vals, 0.0)
        tmin_safe = np.where(valid, tmin_vals, 0.0)
        tmax_prev_safe = np.where(~np.isnan(tmax_prev_vals), tmax_prev_vals, tmax_safe)

        station_feats[:, s_idx, 0] = tmax_safe
        station_feats[:, s_idx, 1] = tmin_safe
        station_feats[:, s_idx, 2] = np.where(valid, tmax_safe - nyc_lag1_arr, 0.0)
        station_feats[:, s_idx, 3] = np.where(valid, tmax_safe - tmin_safe, 0.0)
        station_feats[:, s_idx, 4] = np.where(valid, tmax_safe - tmax_prev_safe, 0.0)
        station_feats[:, s_idx, 5] = np.where(
            valid, tmax_safe - (climo_mean if not np.isnan(climo_mean) else 0.0), 0.0
        )

    # --- Per-station metadata (static, same every day) ---
    # Features:
    #   0: bearing (radians)
    #   1: distance_mi (raw, will be normalized)
    #   2-5: ring one-hot (Ring1, Ring2, Ring3, Ring4)
    N_META_FEATURES = 6
    RING_MAP = {"Ring1_Near": 0, "Ring2_Regional": 1, "Ring3_Extended": 2, "Ring4_Far": 3}

    station_meta_static = np.zeros((n_stations, N_META_FEATURES), dtype=np.float32)
    station_bearings_static = np.zeros(n_stations, dtype=np.float32)

    max_dist = max(
        m["distance_mi"] for m in STATION_METADATA.values()
    )

    for s_idx, sid in enumerate(ordered_stations):
        meta = STATION_METADATA.get(sid, {})
        bearing_deg = meta.get("bearing", 0.0)
        bearing_rad = np.radians(bearing_deg)
        distance = meta.get("distance_mi", 0.0)
        ring = meta.get("ring", "Ring2_Regional")

        station_bearings_static[s_idx] = bearing_rad
        station_meta_static[s_idx, 0] = bearing_rad
        station_meta_static[s_idx, 1] = distance / max_dist  # normalise to [0, 1]

        ring_idx = RING_MAP.get(ring, 1)
        station_meta_static[s_idx, 2 + ring_idx] = 1.0

    # Broadcast to (n_days, n_stations, N_META_FEATURES)
    station_meta = np.tile(station_meta_static, (n_days, 1, 1))
    # Broadcast bearings to (n_days, n_stations)
    station_bearings = np.tile(station_bearings_static, (n_days, 1))

    # --- Wind direction proxy ---
    # Compute NW-SE and NE-SW temperature gradient vectors across station network,
    # then estimate prevailing wind from the direction of maximum cooling.
    nw_sids = STATION_SECTORS.get("NW", []) + STATION_SECTORS.get("N", [])
    se_sids = STATION_SECTORS.get("SE", []) + STATION_SECTORS.get("S", [])
    ne_sids = STATION_SECTORS.get("NE", []) + STATION_SECTORS.get("E", [])
    sw_sids = STATION_SECTORS.get("SW", []) + STATION_SECTORS.get("W", [])

    def sector_tmax_mean(sids: list, tmax_df: pd.DataFrame) -> pd.Series:
        cols = [f"{s}_TMAX" for s in sids if f"{s}_TMAX" in tmax_df.columns]
        if cols:
            return tmax_df[cols].mean(axis=1)
        return pd.Series(0.0, index=tmax_df.index)

    nw_mean = sector_tmax_mean(nw_sids, tmax_l1)
    se_mean = sector_tmax_mean(se_sids, tmax_l1)
    ne_mean = sector_tmax_mean(ne_sids, tmax_l1)
    sw_mean = sector_tmax_mean(sw_sids, tmax_l1)

    # NW-SE gradient: positive means NW is warmer
    grad_nwse = (nw_mean - se_mean).fillna(0.0).values
    # NE-SW gradient: positive means NE is warmer
    grad_nesw = (ne_mean - sw_mean).fillna(0.0).values

    # Convert gradient vector to wind direction proxy (radians)
    # Wind blows FROM the cold side, so direction = atan2(-grad_y, -grad_x)
    # We use NW-SE as x-axis (bearing ~315 deg) and NE-SW as y-axis (~45 deg)
    # Simplified: atan2 of the two gradient components
    wind_proxy = np.arctan2(-grad_nesw, -grad_nwse)  # (n_days,)
    # Replace NaN with 0 (neutral)
    wind_proxy = np.where(np.isnan(wind_proxy), 0.0, wind_proxy).astype(np.float32)

    # --- Global context features ---
    # 0: sin_day
    # 1: cos_day
    # 2: nyc_tmax_lag1
    # 3: mos_ensemble_tmax_f
    # 4: gfs_mos_tmax_f
    # 5: nam_mos_tmax_f
    # 6: gfs_nam_spread
    # 7: mos_error_7d
    # 8: mos_era
    # 9: day_length
    # 10: solar_elev_noon
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
    # Station features: fit per-feature scaler across all stations
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

    # Apply scaling to station features (only to valid entries)
    for f_idx in range(n_sf):
        station_feats[:, :, f_idx] = (
            (station_feats[:, :, f_idx] - station_feat_means[f_idx])
            / station_feat_stds[f_idx]
        )
    # Zero out masked entries after scaling
    for s_idx in range(n_stations):
        invalid = station_mask[:, s_idx] < 0.5
        station_feats[invalid, s_idx, :] = 0.0

    # Global context scaler
    global_scaler = StandardScaler()
    global_ctx_train = global_ctx[train_mask_np]
    # Impute NaNs in train with 0 (they were already converted)
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
        "Structured dataset built: Train=%d, Val=%d, Test=%d, "
        "Stations=%d, StationFeats=%d, MetaFeats=%d, GlobalFeats=%d",
        len(result["train"]["target_residual"]),
        len(result["val"]["target_residual"]),
        len(result["test"]["target_residual"]),
        n_stations, N_STATION_FEATURES, N_META_FEATURES, N_GLOBAL_FEATURES,
    )
    return result


# ============================================================================
# 4. PYTORCH DATASET
# ============================================================================

class WGADataset(Dataset):
    """PyTorch Dataset that yields dicts for the WindGatedAttentionModel."""

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
# 5. TRAINING LOOP
# ============================================================================

def gaussian_nll_loss(mu: torch.Tensor, log_sigma: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.

    NLL = 0.5 * log(2*pi) + log_sigma + 0.5 * ((target - mu) / sigma)^2
    """
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
    """Train a single WGA model seed. Returns (best_val_loss, best_state_dict, history)."""
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

    # Preload validation tensors
    val_ds = WGADataset(val_data)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Validate ---
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
# 6. EVALUATION
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray,
    dates: pd.DatetimeIndex = None, label: str = "",
) -> dict:
    """Compute MAE, RMSE, R2 and seasonal MAE."""
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
# 7. REGIME CLASSIFICATION
# ============================================================================

def classify_regimes(
    mos_base: np.ndarray,
    gfs_mos: np.ndarray,
    nam_mos: np.ndarray,
    ensemble_std: np.ndarray,
    ensemble_mu: np.ndarray,
    dates: pd.DatetimeIndex,
    thresholds: dict = None,
) -> tuple:
    """Classify each day into variance regimes matching the existing predictions format.

    Returns (regimes_array, thresholds_dict).
    """
    mos_spread = np.abs(gfs_mos - nam_mos)
    dod_change = np.abs(np.diff(ensemble_mu, prepend=ensemble_mu[0]))
    month = dates.month
    season = np.where(
        np.isin(month, [12, 1, 2]), 0,
        np.where(
            np.isin(month, [3, 4, 5]), 1,
            np.where(np.isin(month, [6, 7, 8]), 2, 3),
        ),
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

    # High variance: 2+ indicators above 70th percentile
    high_count = (
        (mos_spread > spread_high).astype(int)
        + (ensemble_std > consensus_high).astype(int)
        + (dod_change > dod_high).astype(int)
    )
    regimes[high_count >= 2] = "high_var"

    # Low variance: all below median
    spread_med = np.median(mos_spread)
    consensus_med = np.median(ensemble_std)
    dod_med = np.median(dod_change)
    low_mask = (
        (mos_spread <= spread_med)
        & (ensemble_std <= consensus_med)
        & (dod_change <= dod_med)
    )
    regimes[low_mask] = "low_var"

    # Seasonal transition (MAM=1 and SON=3) where still medium_var
    regimes[np.isin(season, [1, 3]) & (regimes == "medium_var")] = "seasonal_transition"

    return regimes, thresholds


# ============================================================================
# 8. SIGMA CALIBRATION
# ============================================================================

def compute_monthly_sigma_calibration(
    val_dates: pd.DatetimeIndex,
    val_errors: np.ndarray,
    val_sigma_base: np.ndarray,
) -> tuple:
    """Compute monthly sigma from validation errors.

    Returns (sigma_by_month, monthly_scale_factors).
    """
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


def apply_monthly_sigma_calibration(
    sigma_base: np.ndarray,
    dates: pd.DatetimeIndex,
    monthly_scale: dict,
) -> np.ndarray:
    """Apply monthly scale factors to sigma."""
    calibrated = sigma_base.copy()
    for i in range(len(calibrated)):
        month = int(dates[i].month)
        calibrated[i] *= monthly_scale.get(month, 1.0)
    return np.clip(calibrated, SIGMA_FLOOR, SIGMA_CAP)


# ============================================================================
# 9. MAIN ENSEMBLE TRAINING PIPELINE
# ============================================================================

def run_pipeline(data: dict) -> dict:
    """Train 5-seed ensemble, evaluate, calibrate sigma, classify regimes, save outputs."""
    logger.info("=" * 70)
    logger.info("TRAINING 5-SEED WGA-MDN ENSEMBLE")
    logger.info("=" * 70)

    n_stations = data["n_stations"]
    n_sf = data["n_station_features"]
    n_mf = data["n_metadata_features"]
    n_gf = data["n_global_features"]

    logger.info(
        "Model config: stations=%d, station_feats=%d, meta_feats=%d, "
        "global_feats=%d, embed=%d, attn_dim=%d",
        n_stations, n_sf, n_mf, n_gf, STATION_EMBED_DIM, ATTENTION_DIM,
    )

    all_results = {}
    seed_state_dicts = {}
    per_seed_mu = {split: [] for split in ["train", "val", "test"]}
    per_seed_sigma = {split: [] for split in ["train", "val", "test"]}
    training_log = {}

    for seed in ENSEMBLE_SEEDS:
        logger.info("--- Training seed %d ---", seed)
        set_seed(seed)
        model = WindGatedAttentionModel(
            n_station_features=n_sf,
            n_metadata_features=n_mf,
            n_global_features=n_gf,
            n_stations=n_stations,
            station_embed_dim=STATION_EMBED_DIM,
            attention_dim=ATTENTION_DIM,
            output_mode="gaussian",
            dropout=0.1,
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

        # Predict on all splits
        for split in ["train", "val", "test"]:
            resid_mu, resid_sigma = predict_wga(model, data[split])
            actual_mu = data[split]["mos_base"] + resid_mu
            per_seed_mu[split].append(actual_mu)
            per_seed_sigma[split].append(resid_sigma)

            r = evaluate_predictions(
                data[split]["actual_tmax"], actual_mu,
                data[split]["dates"], f"Seed_{seed} {split}",
            )
            all_results[f"seed_{seed}_{split}"] = r

    # --- Ensemble aggregation ---
    logger.info("--- 5-Seed Ensemble (mean) ---")
    ensemble_mu = {}
    ensemble_sigma_base = {}
    ensemble_std = {}

    for split in ["train", "val", "test"]:
        stacked_mu = np.array(per_seed_mu[split])       # (5, N)
        stacked_sigma = np.array(per_seed_sigma[split])  # (5, N)

        ens_mu = stacked_mu.mean(axis=0)
        ens_std = stacked_mu.std(axis=0)
        # Combine model sigma (mean across seeds) with ensemble spread
        ens_sigma_model = stacked_sigma.mean(axis=0)
        ens_sigma_base = np.sqrt(ens_sigma_model ** 2 + ens_std ** 2)
        ens_sigma_base = np.clip(ens_sigma_base, SIGMA_FLOOR, SIGMA_CAP)

        ensemble_mu[split] = ens_mu
        ensemble_sigma_base[split] = ens_sigma_base
        ensemble_std[split] = ens_std

        r = evaluate_predictions(
            data[split]["actual_tmax"], ens_mu,
            data[split]["dates"], f"Ensemble_5seed {split}",
        )
        all_results[f"ensemble_{split}"] = r

    # --- Sigma calibration (fit on validation) ---
    logger.info("--- Sigma Calibration ---")
    val_dates = pd.to_datetime(data["val"]["dates"])
    val_errors = data["val"]["actual_tmax"] - ensemble_mu["val"]
    sigma_by_month, monthly_scale = compute_monthly_sigma_calibration(
        val_dates, val_errors, ensemble_sigma_base["val"]
    )

    for month in range(1, 13):
        logger.info(
            "  Month %2d: sigma=%.3f, scale=%.3f",
            month, sigma_by_month[month], monthly_scale[month],
        )

    # Apply calibration to all splits
    sigma_monthly_cal = {}
    for split in ["train", "val", "test"]:
        sigma_monthly_cal[split] = apply_monthly_sigma_calibration(
            ensemble_sigma_base[split],
            pd.to_datetime(data[split]["dates"]),
            monthly_scale,
        )

    # --- Regime classification ---
    logger.info("--- Regime Classification ---")
    # Fit thresholds on validation
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
        for regime_name in ["low_var", "medium_var", "high_var", "seasonal_transition"]:
            n_regime = (r == regime_name).sum()
            logger.info("  %s %s: %d days", split, regime_name, n_regime)

    # --- Build regime-conditional sigma from validation ---
    logger.info("--- Regime-Conditional Sigma ---")
    regime_sigma_models = {}
    for regime_name in ["low_var", "medium_var", "high_var", "seasonal_transition"]:
        m = regimes["val"] == regime_name
        if np.any(m):
            regime_std = float(np.std(val_errors[m], ddof=1))
            regime_sigma_models[regime_name] = max(regime_std, SIGMA_FLOOR)
            logger.info(
                "  %s: sigma=%.3f (n=%d)",
                regime_name, regime_sigma_models[regime_name], np.sum(m),
            )

    sigma_regime_cond = {}
    for split in ["val", "test"]:
        sig = np.full(len(regimes[split]), float(np.std(val_errors, ddof=1)))
        for i in range(len(sig)):
            r = regimes[split][i]
            if r in regime_sigma_models:
                sig[i] = regime_sigma_models[r]
        sigma_regime_cond[split] = np.clip(sig, SIGMA_FLOOR, SIGMA_CAP)

    return {
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
        "regime_sigma_models": regime_sigma_models,
        "sigma_regime_cond": sigma_regime_cond,
    }


# ============================================================================
# 10. OUTPUT GENERATION
# ============================================================================

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
    if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    return obj


def save_outputs(data: dict, pipeline_result: dict) -> None:
    """Save all outputs to results/wga_mdn_model/."""
    logger.info("=" * 70)
    logger.info("SAVING OUTPUTS to %s", RESULTS_DIR)
    logger.info("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ens_mu = pipeline_result["ensemble_mu"]
    ens_sigma_base = pipeline_result["ensemble_sigma_base"]
    ens_std = pipeline_result["ensemble_std"]
    sigma_monthly_cal = pipeline_result["sigma_monthly_cal"]
    sigma_regime_cond = pipeline_result["sigma_regime_cond"]
    regimes = pipeline_result["regimes"]

    # 1. Predictions CSV (test)
    test_dates = pd.to_datetime(data["test"]["dates"])
    pred_test_df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["test"]["actual_tmax"],
        "mos_base": data["test"]["mos_base"],
        "model_mu": ens_mu["test"],
        "model_sigma_base": ens_sigma_base["test"],
        "model_sigma_monthly_cal": sigma_monthly_cal["test"],
        "model_sigma_regime_conditional": sigma_regime_cond["test"],
        "ensemble_std": ens_std["test"],
        "regime": regimes["test"],
    })
    test_path = os.path.join(RESULTS_DIR, "predictions_test.csv")
    pred_test_df.to_csv(test_path, index=False)
    logger.info("Saved test predictions: %s (%d rows)", test_path, len(pred_test_df))

    # 2. Predictions CSV (val)
    val_dates = pd.to_datetime(data["val"]["dates"])
    pred_val_df = pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["val"]["actual_tmax"],
        "mos_base": data["val"]["mos_base"],
        "model_mu": ens_mu["val"],
        "model_sigma_base": ens_sigma_base["val"],
        "model_sigma_monthly_cal": sigma_monthly_cal["val"],
        "model_sigma_regime_conditional": sigma_regime_cond["val"],
        "ensemble_std": ens_std["val"],
        "regime": regimes["val"],
    })
    val_path = os.path.join(RESULTS_DIR, "predictions_val.csv")
    pred_val_df.to_csv(val_path, index=False)
    logger.info("Saved val predictions: %s (%d rows)", val_path, len(pred_val_df))

    # 3. Ensemble checkpoint
    ckpt_path = os.path.join(RESULTS_DIR, "ensemble_5seed.pt")
    torch.save({
        "model_class": "WindGatedAttentionModel",
        "model_config": {
            "n_station_features": data["n_station_features"],
            "n_metadata_features": data["n_metadata_features"],
            "n_global_features": data["n_global_features"],
            "n_stations": data["n_stations"],
            "station_embed_dim": STATION_EMBED_DIM,
            "attention_dim": ATTENTION_DIM,
            "output_mode": "gaussian",
            "dropout": 0.1,
        },
        "ordered_stations": data["ordered_stations"],
        "seed_state_dicts": pipeline_result["seed_state_dicts"],
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "splits": {
            "train": f"{MOS_TRAIN_START} to {MOS_TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
    }, ckpt_path)
    logger.info("Saved ensemble checkpoint: %s", ckpt_path)

    # 4. Scaler
    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    scaler_data = {
        "station_feat_means": data["station_feat_means"],
        "station_feat_stds": data["station_feat_stds"],
        "global_scaler": data["global_scaler"],
    }
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)
    logger.info("Saved scaler: %s", scaler_path)

    # 5. sigma_by_month
    sigma_path = os.path.join(RESULTS_DIR, "sigma_by_month.json")
    with open(sigma_path, "w") as f:
        json.dump(
            {str(k): round(v, 4) for k, v in pipeline_result["sigma_by_month"].items()},
            f, indent=2,
        )
    logger.info("Saved sigma_by_month: %s", sigma_path)

    # 6. Training log
    log_path = os.path.join(RESULTS_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(_make_serializable(pipeline_result["training_log"]), f, indent=2)
    logger.info("Saved training log: %s", log_path)

    # 7. Experiment results
    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(_make_serializable(pipeline_result["all_results"]), f, indent=2)
    logger.info("Saved experiment results: %s", results_path)

    # 8. Regime & calibration metadata
    meta_path = os.path.join(RESULTS_DIR, "calibration_metadata.json")
    cal_meta = {
        "regime_thresholds": pipeline_result["regime_thresholds"],
        "regime_sigma_models": pipeline_result["regime_sigma_models"],
        "monthly_scale_factors": pipeline_result["monthly_scale"],
        "sigma_by_month": pipeline_result["sigma_by_month"],
    }
    with open(meta_path, "w") as f:
        json.dump(_make_serializable(cal_meta), f, indent=2)
    logger.info("Saved calibration metadata: %s", meta_path)


# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

def print_summary(data: dict, pipeline_result: dict) -> None:
    """Print a concise summary of results."""
    logger.info("\n" + "=" * 80)
    logger.info("WGA-MDN TRAINING COMPLETE")
    logger.info("=" * 80)

    all_results = pipeline_result["all_results"]
    ens_mu = pipeline_result["ensemble_mu"]

    for split in ["val", "test"]:
        r = all_results.get(f"ensemble_{split}", {})
        logger.info(
            "  %s: MAE=%.4f  RMSE=%.4f  R2=%.4f",
            split.upper(), r.get("mae", 0), r.get("rmse", 0), r.get("r2", 0),
        )
        for s in ["DJF", "MAM", "JJA", "SON"]:
            smae = r.get(f"mae_{s}", "N/A")
            logger.info("    %s MAE: %s", s, smae)

    # Per-seed test MAE
    logger.info("\n  Per-seed test MAE:")
    for seed in ENSEMBLE_SEEDS:
        r = all_results.get(f"seed_{seed}_test", {})
        logger.info("    Seed %d: %.4f", seed, r.get("mae", 0))

    # Coverage check with monthly-calibrated sigma
    test_mu = ens_mu["test"]
    test_sigma = pipeline_result["sigma_monthly_cal"]["test"]
    test_actual = data["test"]["actual_tmax"]
    z95 = scipy_norm.ppf(0.975)
    lower = test_mu - z95 * test_sigma
    upper = test_mu + z95 * test_sigma
    coverage = float(np.mean((test_actual >= lower) & (test_actual <= upper)))
    avg_width = float(np.mean(upper - lower))
    logger.info(
        "\n  95%% PI coverage (monthly cal): %.3f (target: 0.950), avg width: %.2f F",
        coverage, avg_width,
    )

    logger.info("\n  Outputs saved to: %s", RESULTS_DIR)


# ============================================================================
# 12. MAIN
# ============================================================================

def main() -> None:
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("WIND-GATED ATTENTION + MDN TRAINING PIPELINE")
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
    logger.info(
        "Architecture: station_embed_dim=%d, attention_dim=%d, output=gaussian",
        STATION_EMBED_DIM, ATTENTION_DIM,
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

    # --- Step 4: Build structured dataset ---
    logger.info("\n=== STEP 4: Building structured station-level dataset ===")
    data = build_structured_dataset(station_matrix, mos_data, cp_data)

    # --- Step 5: Train ensemble ---
    logger.info("\n=== STEP 5: Training 5-seed WGA-MDN ensemble ===")
    pipeline_result = run_pipeline(data)

    # --- Step 6: Save all outputs ---
    logger.info("\n=== STEP 6: Saving outputs ===")
    save_outputs(data, pipeline_result)

    # --- Step 7: Print summary ---
    print_summary(data, pipeline_result)

    elapsed = time.time() - start_time
    logger.info(
        "\nTotal time: %.1f minutes (%.0f seconds)", elapsed / 60, elapsed
    )
    logger.info("DONE.")


if __name__ == "__main__":
    main()
