#!/usr/bin/env python3
"""
Retrain Extended MOS: Retrain the best model configuration (A_NN_64_32)
using extended MOS data back to 2000 with an expanded validation window.

Changes from phase1_combined_best.py:
  - MOS source: combined_mos_extended.csv (back to 2000, includes airport_proxy era)
  - Era indicator feature: mos_era (0=airport_proxy, 1=knyc_native)
  - Updated chronological splits:
      Train: 2000-06-01 to 2021-12-31
      Val:   2022-01-01 to 2023-12-31
      Test:  2024-01-01 to 2024-12-31
      OOS:   2025-01-01 to 2025-12-31
  - Only trains Model A (NN [64,32]) as 5-seed ensemble
  - Saves benchmark prediction files with mu/sigma
  - Compares old vs new metrics
"""

import os
import sys
import json
import time
import math
import copy
import logging
import warnings
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
    SURROUNDING_STATIONS, STATION_METADATA,
    METEOROLOGICAL_SECTORS, STATION_SECTORS, STATION_RINGS,
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
logger = logging.getLogger("retrain_extended_mos")

# ---------------------------------------------------------------------------
# Constants — UPDATED for extended data
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_extended.csv")
ERA_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "mos_era_indicator.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrain_extended_mos")
OLD_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "phase1_combined", "summary.csv")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# UPDATED chronological splits
MOS_TRAIN_START, MOS_TRAIN_END = "2000-06-01", "2021-12-31"
VAL_START, VAL_END = "2022-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_SEED = 42
DEFAULT_WD = 1e-4
SIGMA_FLOOR = 0.75
SIGMA_CAP = 10.0
NYC_LAT = 40.7831


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def download_all_stations():
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
                logger.warning("[%d/%d] %s attempt %d failed: %s", i, total, sid, attempt + 1, e)
                time.sleep(wait)
        else:
            logger.error("[%d/%d] %s -- FAILED", i, total, sid)


def parse_station_tmax(station_id, start_date, end_date):
    dly_path = os.path.join(RAW_DIR, f"{station_id}.dly")
    if not os.path.exists(dly_path):
        return pd.Series(dtype=float)
    df = parse_dly_file(dly_path, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    tmax = df[df["element"] == "TMAX"][["date", "value"]].copy()
    tmax["date"] = pd.to_datetime(tmax["date"])
    tmax = tmax.drop_duplicates(subset="date").set_index("date")["value"]
    tmax.name = f"{station_id}_TMAX"
    return tmax


def parse_station_tmin(station_id, start_date, end_date):
    dly_path = os.path.join(RAW_DIR, f"{station_id}.dly")
    if not os.path.exists(dly_path):
        return pd.Series(dtype=float)
    df = parse_dly_file(dly_path, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    tmin = df[df["element"] == "TMIN"][["date", "value"]].copy()
    tmin["date"] = pd.to_datetime(tmin["date"])
    tmin = tmin.drop_duplicates(subset="date").set_index("date")["value"]
    tmin.name = f"{station_id}_TMIN"
    return tmin


def build_station_matrix(start_date, end_date, include_tmin=True):
    logger.info("Building station matrix from %s to %s ...", start_date, end_date)
    frames = []
    for sid in ALL_SURROUNDING:
        tmax = parse_station_tmax(sid, start_date, end_date)
        if len(tmax) > 0:
            frames.append(tmax)
        if include_tmin:
            tmin = parse_station_tmin(sid, start_date, end_date)
            if len(tmin) > 0:
                frames.append(tmin)
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
    logger.info("Station matrix: %d rows x %d columns", len(matrix), len(matrix.columns))
    return matrix


def load_mos_data():
    """Load extended MOS data with era indicator."""
    mos = pd.read_csv(MOS_PATH, parse_dates=["date"])
    mos = mos[["date", "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]].copy()
    mos = mos.set_index("date").sort_index()
    logger.info("Extended MOS data: %d rows, %s to %s",
                len(mos), mos.index.min().date(), mos.index.max().date())

    # Load era indicator
    era = pd.read_csv(ERA_PATH, parse_dates=["date"])
    era = era.set_index("date").sort_index()
    mos = mos.join(era, how="left")
    mos["mos_era"] = mos["mos_era"].fillna(0).astype(float)
    logger.info("Era indicator loaded: %d airport_proxy (era=0), %d knyc_native (era=1)",
                (mos["mos_era"] == 0).sum(), (mos["mos_era"] == 1).sum())
    return mos


def load_central_park_tmax():
    cp = pd.read_csv(CP_PATH, parse_dates=["date"])
    cp = cp.set_index("date").sort_index()
    cp.columns = ["nyc_tmax"]
    logger.info("Central Park TMAX: %d rows, %s to %s",
                len(cp), cp.index.min().date(), cp.index.max().date())
    return cp


# ============================================================================
# 2. FEATURE ENGINEERING HELPERS
# ============================================================================

def add_date_features(df):
    doy = df.index.dayofyear
    df = df.copy()
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def create_lagged_features(station_matrix, lag=1):
    lagged = station_matrix.shift(lag)
    lagged.columns = [f"{c}_lag{lag}" for c in lagged.columns]
    return lagged


def assign_season(dates):
    month = dates.month
    seasons = pd.Series("", index=dates)
    seasons[month.isin([12, 1, 2])] = "DJF"
    seasons[month.isin([3, 4, 5])] = "MAM"
    seasons[month.isin([6, 7, 8])] = "JJA"
    seasons[month.isin([9, 10, 11])] = "SON"
    return seasons


# --- MOS Error Memory Features ---

def add_mos_error_memory_features(df):
    """MOS error memory features: rolling bias, abs error, GFS-NAM spread."""
    df = df.copy()
    mos_error = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
    df["mos_error_yesterday"] = mos_error.shift(1)
    mos_error_shifted = mos_error.shift(1)
    df["mos_error_7d"] = mos_error_shifted.rolling(window=7, min_periods=3).mean()
    df["mos_error_14d"] = mos_error_shifted.rolling(window=14, min_periods=5).mean()
    df["mos_error_30d"] = mos_error_shifted.rolling(window=30, min_periods=10).mean()
    abs_error_shifted = mos_error.abs().shift(1)
    df["mos_abs_error_7d"] = abs_error_shifted.rolling(window=7, min_periods=3).mean()
    gfs_nam_diff = df["gfs_mos_tmax_f"] - df["nam_mos_tmax_f"]
    df["gfs_nam_spread"] = gfs_nam_diff.abs()
    df["gfs_nam_sign"] = np.sign(gfs_nam_diff)
    return df


PHASE_1A_FEATURES = [
    "mos_error_yesterday", "mos_error_7d", "mos_error_14d", "mos_error_30d",
    "mos_abs_error_7d", "gfs_nam_spread", "gfs_nam_sign",
]


# --- MOS x Station Interaction Features ---

def add_mos_station_interaction_features(df, station_lag_tmax_cols):
    """MOS x Station interaction: gap, sector gap, trend agreement."""
    df = df.copy()
    df["mos_station_gap"] = df["mos_ensemble_tmax_f"] - df["nyc_tmax_lag1"]
    wnw_stations = METEOROLOGICAL_SECTORS.get("WNW", [])
    wnw_cols = [f"{sid}_TMAX_lag1" for sid in wnw_stations
                if f"{sid}_TMAX_lag1" in df.columns]
    if wnw_cols:
        wnw_mean = df[wnw_cols].mean(axis=1)
        df["mos_sector_gap"] = df["mos_ensemble_tmax_f"] - wnw_mean
    else:
        df["mos_sector_gap"] = 0.0
    if station_lag_tmax_cols:
        available = [c for c in station_lag_tmax_cols if c in df.columns]
        if available:
            station_mean = df[available].mean(axis=1)
            station_delta = station_mean.diff()
            mos_gap_sign = np.sign(df["mos_station_gap"])
            station_delta_sign = np.sign(station_delta)
            df["station_mos_agree"] = (mos_gap_sign == station_delta_sign).astype(float)
        else:
            df["station_mos_agree"] = 0.0
    else:
        df["station_mos_agree"] = 0.0
    return df


PHASE_1C_FEATURES = [
    "mos_station_gap", "mos_sector_gap", "station_mos_agree",
]


# --- Enhanced Temporal Features ---

def solar_declination(doy):
    return np.radians(23.44) * np.sin(np.radians((360 / 365.25) * (doy - 81)))


def day_length_hours(lat_deg, doy):
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.arccos(cos_ha)
    return (2.0 * ha / np.pi) * 12.0


def solar_elevation_noon(lat_deg, doy):
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    elev = np.arcsin(np.sin(lat_rad) * np.sin(decl) +
                     np.cos(lat_rad) * np.cos(decl))
    return np.degrees(elev)


def add_enhanced_temporal_features(df, cp_data, train_end):
    """Enhanced temporal features: day length, solar elevation, rolling stats, degree days."""
    df = df.copy()
    doy = df.index.dayofyear
    df["day_length"] = day_length_hours(NYC_LAT, doy)
    df["solar_elev_noon"] = solar_elevation_noon(NYC_LAT, doy)

    nyc_full = cp_data["nyc_tmax"].reindex(df.index)
    nyc_lag1 = nyc_full.shift(1)

    df["tmax_7d_rolling_mean"] = nyc_lag1.rolling(7, min_periods=4).mean()

    # Climatological mean from training data only
    train_mask = cp_data.index <= train_end
    train_cp = cp_data.loc[train_mask, "nyc_tmax"]
    climo = train_cp.groupby(train_cp.index.dayofyear).mean()
    doy_climo = doy.map(lambda d: climo.get(d, climo.get(min(d, 365), np.nan)))
    df["tmax_anomaly_from_climo"] = nyc_lag1 - doy_climo.values

    hdd_daily = np.maximum(0, 65 - nyc_lag1)
    df["hdd_7d"] = hdd_daily.rolling(7, min_periods=4).sum()
    cdd_daily = np.maximum(0, nyc_lag1 - 65)
    df["cdd_7d"] = cdd_daily.rolling(7, min_periods=4).sum()

    new_cols = [
        "day_length", "solar_elev_noon",
        "tmax_7d_rolling_mean", "tmax_anomaly_from_climo",
        "hdd_7d", "cdd_7d",
    ]
    return df, new_cols


TEMPORAL_FEATURES = [
    "day_length", "solar_elev_noon",
    "tmax_7d_rolling_mean", "tmax_anomaly_from_climo",
    "hdd_7d", "cdd_7d",
]


# --- Enhanced Spatial Features ---

def add_spatial_features(df, station_matrix):
    """Spatial features: station spread, consensus, gradients."""
    df = df.copy()
    tmax_cols = [c for c in station_matrix.columns if "TMAX" in c]
    if not tmax_cols:
        return df, []

    tmax_matrix = station_matrix[tmax_cols]
    tmax_lag1 = tmax_matrix.shift(1)
    tmax_lag2 = tmax_matrix.shift(2)
    tmax_l1 = tmax_lag1.reindex(df.index)
    tmax_l2 = tmax_lag2.reindex(df.index)

    change_24h = (tmax_l1 - tmax_l2).abs()
    df["max_station_24h_change"] = change_24h.max(axis=1)
    df["station_spread"] = tmax_l1.max(axis=1) - tmax_l1.min(axis=1)
    df["station_consensus"] = tmax_l1.std(axis=1)

    def sector_mean(sector_stations, tmax_df):
        cols = [f"{sid}_TMAX" for sid in sector_stations
                if f"{sid}_TMAX" in tmax_df.columns]
        if cols:
            return tmax_df[cols].mean(axis=1)
        return pd.Series(np.nan, index=tmax_df.index)

    wnw_stations = METEOROLOGICAL_SECTORS.get("WNW", [])
    coastal_stations = METEOROLOGICAL_SECTORS.get("Coastal", [])
    wnw_mean = sector_mean(wnw_stations, tmax_l1)
    coastal_mean = sector_mean(coastal_stations, tmax_l1)
    df["wn_to_coast_gradient"] = wnw_mean - coastal_mean

    ne_stations = METEOROLOGICAL_SECTORS.get("NE", [])
    sw_stations = METEOROLOGICAL_SECTORS.get("SW", [])
    ne_mean = sector_mean(ne_stations, tmax_l1)
    sw_mean = sector_mean(sw_stations, tmax_l1)
    df["ne_sw_gradient"] = ne_mean - sw_mean

    new_cols = [
        "max_station_24h_change", "station_spread", "station_consensus",
        "wn_to_coast_gradient", "ne_sw_gradient",
    ]
    return df, new_cols


SPATIAL_FEATURES = [
    "max_station_24h_change", "station_spread", "station_consensus",
    "wn_to_coast_gradient", "ne_sw_gradient",
]


# ============================================================================
# 3. MODEL DEFINITION
# ============================================================================

class FlexibleNN(nn.Module):
    """Configurable feedforward NN with optional BatchNorm."""
    def __init__(self, n_features, hidden_sizes, dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm and h >= 16:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================

def train_nn(model, X_train, y_train, X_val, y_val,
             lr=1e-3, epochs=300, patience=20, batch_size=128,
             loss_fn_name="huber", weight_decay=DEFAULT_WD):
    """Train point-prediction NN with early stopping. Returns best val MAE."""
    model = model.to(DEVICE)
    if loss_fn_name == "huber":
        criterion = nn.HuberLoss(delta=2.0)
    elif loss_fn_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze(-1)
            val_mae = torch.mean(torch.abs(val_pred - y_val_t)).item()

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return best_val_mae


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        preds = model(X_t).squeeze(-1).cpu().numpy()
    return preds


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, dates=None, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    result = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    if dates is not None:
        seasons = assign_season(pd.DatetimeIndex(dates))
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons == s
            if mask.sum() > 0:
                result[f"mae_{s}"] = round(mean_absolute_error(
                    np.array(y_true)[mask], np.array(y_pred)[mask]), 4)
    if label:
        logger.info("  %s: MAE=%.3f  RMSE=%.3f  R2=%.3f", label, mae, rmse, r2)
    return result


# ============================================================================
# 6. COMBINED DATASET BUILDER (with era indicator)
# ============================================================================

class CombinedDatasetBuilder:
    """Builds MOS-correction datasets with ALL combined features + era indicator."""

    def __init__(self, station_matrix, mos_data, cp_data):
        self.station_matrix = station_matrix
        self.mos = mos_data
        self.cp = cp_data
        self._full_df = None
        self._feature_cols = None
        self._station_lag_tmax_cols = None

    def _build_full_dataframe(self):
        if self._full_df is not None:
            return

        logger.info("Building full combined feature dataframe...")

        # Station lag-1
        lagged = create_lagged_features(self.station_matrix, lag=1)
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        # Merge
        df = pd.concat([lagged, self.mos, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])

        # Ensure mos_era is present (from MOS data join)
        if "mos_era" not in df.columns:
            df["mos_era"] = 0.0
        df["mos_era"] = df["mos_era"].fillna(0.0)

        # Base date features
        df = add_date_features(df)

        # Residual target
        df["residual"] = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
        df = df.dropna(subset=["nyc_tmax_lag1"])

        # Station mean and MOS-station diff
        station_lag_tmax = [c for c in lagged.columns if c in df.columns and "TMAX" in c]
        self._station_lag_tmax_cols = station_lag_tmax
        if station_lag_tmax:
            df["station_mean_lag1"] = df[station_lag_tmax].mean(axis=1)
            df["mos_station_diff"] = (df["station_mean_lag1"] - df["mos_ensemble_tmax_f"]).abs()
        else:
            df["station_mean_lag1"] = 0.0
            df["mos_station_diff"] = 0.0

        # MOS Error Memory
        df = add_mos_error_memory_features(df)

        # MOS x Station Interaction
        df = add_mos_station_interaction_features(df, station_lag_tmax)

        # Enhanced Temporal
        df, temporal_cols = add_enhanced_temporal_features(df, self.cp, train_end=MOS_TRAIN_END)

        # Enhanced Spatial
        df, spatial_cols = add_spatial_features(df, self.station_matrix)

        # Build feature column list
        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        base_features = station_lag_cols + [
            "mos_ensemble_tmax_f", "nyc_tmax_lag1",
            "station_mean_lag1", "mos_station_diff",
            "sin_day", "cos_day",
        ]

        # Add mos_era to feature list
        self._feature_cols = (
            base_features
            + PHASE_1A_FEATURES
            + PHASE_1C_FEATURES
            + TEMPORAL_FEATURES
            + SPATIAL_FEATURES
            + ["mos_era"]  # NEW: era indicator
        )
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in self._feature_cols:
            if c not in seen and c in df.columns:
                seen.add(c)
                unique.append(c)
        self._feature_cols = unique

        self._full_df = df
        logger.info("Full combined dataframe: %d rows, %d features",
                     len(df), len(self._feature_cols))
        logger.info("Feature groups: base=%d, 1A=%d, 1C=%d, temporal=%d, spatial=%d, era=1",
                     len(base_features),
                     len([f for f in PHASE_1A_FEATURES if f in df.columns]),
                     len([f for f in PHASE_1C_FEATURES if f in df.columns]),
                     len([f for f in TEMPORAL_FEATURES if f in df.columns]),
                     len([f for f in SPATIAL_FEATURES if f in df.columns]))
        logger.info("Date range in full df: %s to %s",
                     df.index.min().date(), df.index.max().date())

    def build_correction_data(self):
        """Build MOS correction dataset with all combined features."""
        self._build_full_dataframe()
        df = self._full_df
        feat_cols = self._feature_cols
        valid_cols = [c for c in feat_cols if c in df.columns]

        idx = df.index
        masks = {
            "train": (idx >= MOS_TRAIN_START) & (idx <= MOS_TRAIN_END),
            "val": (idx >= VAL_START) & (idx <= VAL_END),
            "test": (idx >= TEST_START) & (idx <= TEST_END),
            "oos": (idx >= OOS_START) & (idx <= OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y_resid": sub["residual"].values.astype(np.float64),
                "y_actual": sub["nyc_tmax"].values.astype(np.float64),
                "mos_base": sub["mos_ensemble_tmax_f"].values.astype(np.float64),
                "dates": sub.index,
            }

        # Impute NaNs with training mean, then StandardScaler
        X_tr = arrays["train"]["X"]
        X_v = arrays["val"]["X"]
        X_te = arrays["test"]["X"]
        X_oos = arrays["oos"]["X"]

        train_means = np.nanmean(X_tr, axis=0)
        train_means = np.where(np.isnan(train_means), 0.0, train_means)
        for arr in [X_tr, X_v, X_te, X_oos]:
            for j in range(arr.shape[1]):
                mask = np.isnan(arr[:, j])
                arr[mask, j] = train_means[j]

        scaler = StandardScaler()
        arrays["train"]["X"] = scaler.fit_transform(X_tr)
        arrays["val"]["X"] = scaler.transform(X_v)
        arrays["test"]["X"] = scaler.transform(X_te)
        arrays["oos"]["X"] = scaler.transform(X_oos)
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        arrays["n_features"] = len(valid_cols)

        logger.info("Dataset built: Train=%d, Val=%d, Test=%d, OOS=%d, Features=%d",
                     len(arrays["train"]["y_resid"]),
                     len(arrays["val"]["y_resid"]),
                     len(arrays["test"]["y_resid"]),
                     len(arrays["oos"]["y_resid"]),
                     arrays["n_features"])

        return arrays


# ============================================================================
# 7. MAIN: 5-SEED ENSEMBLE TRAINING + EVALUATION
# ============================================================================

def run_ensemble(data, all_results):
    """Train 5-seed ensemble of NN [64,32] and evaluate."""
    logger.info("=" * 70)
    logger.info("TRAINING: 5-seed Ensemble of A_NN_64_32 (extended MOS)")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    per_seed_preds = {split: [] for split in ["train", "val", "test", "oos"]}
    seed_state_dicts = {}

    for seed in ENSEMBLE_SEEDS:
        logger.info("--- Seed %d ---", seed)
        set_seed(seed)
        model = FlexibleNN(n_feat, [64, 32], dropout=0.15)
        val_mae = train_nn(
            model, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=1e-3, epochs=300, patience=20, loss_fn_name="huber",
            weight_decay=DEFAULT_WD,
        )
        logger.info("  Seed %d best val MAE (residual): %.4f", seed, val_mae)
        seed_state_dicts[str(seed)] = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }

        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            per_seed_preds[split].append(pred_actual)
            r = evaluate_model(
                data[split]["y_actual"], pred_actual,
                data[split]["dates"], f"Seed_{seed} {split}",
            )
            all_results[f"E_seed_{seed}_{split}"] = r

    # Ensemble mean
    logger.info("--- 5-Seed Ensemble (mean) ---")
    ensemble_preds = {}
    ensemble_stds = {}
    for split in ["train", "val", "test", "oos"]:
        stacked = np.array(per_seed_preds[split])  # (5, N)
        ensemble_pred = stacked.mean(axis=0)
        ensemble_std = stacked.std(axis=0)
        ensemble_preds[split] = ensemble_pred
        ensemble_stds[split] = ensemble_std
        r = evaluate_model(
            data[split]["y_actual"], ensemble_pred,
            data[split]["dates"], f"Ensemble_5seed {split}",
        )
        all_results[f"E_Ensemble_5seed_{split}"] = r

    return {
        "per_seed_preds": per_seed_preds,
        "seed_state_dicts": seed_state_dicts,
        "ensemble_preds": ensemble_preds,
        "ensemble_stds": ensemble_stds,
    }


def build_benchmark_files(data, ensemble_result):
    """Create benchmark-ready daily prediction files from the 5-seed ensemble."""
    per_seed_preds = ensemble_result["per_seed_preds"]
    ensemble_preds = ensemble_result["ensemble_preds"]
    ensemble_stds = ensemble_result["ensemble_stds"]

    # Compute month-specific sigma from test-set residuals
    test_dates = pd.to_datetime(data["test"]["dates"])
    test_actual = data["test"]["y_actual"]
    test_pred = ensemble_preds["test"]
    test_err = test_actual - test_pred

    sigma_by_month = {}
    for month in range(1, 13):
        m = test_dates.month == month
        if np.any(m):
            sigma_by_month[month] = float(np.std(test_err[m], ddof=1))

    global_sigma = float(np.std(test_err, ddof=1))
    sigma_by_month = {
        m: float(np.clip(sigma_by_month.get(m, global_sigma), SIGMA_FLOOR, SIGMA_CAP))
        for m in range(1, 13)
    }

    def _make_split_df(split):
        dates = pd.to_datetime(data[split]["dates"])
        mu = ensemble_preds[split]
        # Use max of: month-based sigma, ensemble spread
        sigma_month_arr = np.array([sigma_by_month[int(m)] for m in dates.month])
        sigma_ensemble_arr = ensemble_stds[split]
        sigma = np.maximum(sigma_month_arr, sigma_ensemble_arr)
        sigma = np.clip(sigma, SIGMA_FLOOR, SIGMA_CAP)
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "model_mu": mu,
            "model_sigma": sigma,
            "actual_tmax": data[split]["y_actual"],
        })

    # Test (2024) prediction file
    pred_test = _make_split_df("test")
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    is_path = os.path.join(PROJECT_ROOT, "data", "best_model_predictions_extended_2022_2024.csv")

    # Also include val predictions (2022-2023) in the IS file
    pred_val = _make_split_df("val")
    pred_is = pd.concat([pred_val, pred_test], ignore_index=True)
    pred_is.to_csv(is_path, index=False)
    logger.info("Saved IS predictions (val+test): %s (%d rows)", is_path, len(pred_is))

    # OOS (2025) prediction file
    pred_oos = _make_split_df("oos")
    oos_path = os.path.join(PROJECT_ROOT, "data", "best_model_predictions_extended_2025.csv")
    pred_oos.to_csv(oos_path, index=False)
    logger.info("Saved OOS predictions: %s (%d rows)", oos_path, len(pred_oos))

    # Save sigma-by-month
    sigma_path = os.path.join(RESULTS_DIR, "sigma_by_month.json")
    with open(sigma_path, "w") as f:
        json.dump({str(k): v for k, v in sigma_by_month.items()}, f, indent=2)
    logger.info("Saved sigma_by_month to %s", sigma_path)

    return sigma_by_month


def save_all_predictions(data, ensemble_result):
    """Save all predictions to a single CSV."""
    ensemble_preds = ensemble_result["ensemble_preds"]
    ensemble_stds = ensemble_result["ensemble_stds"]

    rows = []
    for split in ["train", "val", "test", "oos"]:
        dates = pd.to_datetime(data[split]["dates"])
        for i, d in enumerate(dates):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "split": split,
                "actual_tmax": data[split]["y_actual"][i],
                "mos_base": data[split]["mos_base"][i],
                "ensemble_pred": ensemble_preds[split][i],
                "ensemble_std": ensemble_stds[split][i],
                "residual_actual": data[split]["y_resid"][i],
                "residual_pred": ensemble_preds[split][i] - data[split]["mos_base"][i],
            })

    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(RESULTS_DIR, "predictions_all.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved all predictions: %s (%d rows)", pred_path, len(pred_df))
    return pred_df


def build_metrics_summary(all_results):
    """Build metrics summary CSV from results dict."""
    summary_rows = []
    for key, metrics in all_results.items():
        if isinstance(metrics, dict) and "mae" in metrics:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_name, split = parts
            else:
                model_name, split = key, "unknown"
            row = {"model": model_name, "split": split, **metrics}
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info("Saved metrics summary to %s", csv_path)
        return summary_df
    return pd.DataFrame()


def print_comparison(all_results):
    """Print comparison: old model vs new model metrics."""
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON: Old Model (phase1_combined) vs New Model (retrain_extended_mos)")
    logger.info("=" * 100)

    # Load old results
    old_results = {}
    if os.path.exists(OLD_RESULTS_PATH):
        old_df = pd.read_csv(OLD_RESULTS_PATH)
        for _, row in old_df.iterrows():
            key = f"{row['model']}_{row['split']}"
            old_results[key] = row.to_dict()
        logger.info("Loaded old results from %s", OLD_RESULTS_PATH)
    else:
        logger.warning("Old results not found at %s — using hardcoded baselines", OLD_RESULTS_PATH)
        # Hardcoded from MEMORY.md
        old_results = {
            "E_Ensemble_5seed_test": {"mae": 1.987, "rmse": 2.62, "r2": 0.974},
            "E_Ensemble_5seed_oos": {"mae": 2.018, "rmse": 2.70, "r2": 0.979},
            "A_NN_64_32_test": {"mae": 2.023, "rmse": 2.66, "r2": 0.973},
            "A_NN_64_32_oos": {"mae": 2.029, "rmse": 2.71, "r2": 0.979},
        }

    # Focus comparison on ensemble and key splits
    comparison_keys = [
        ("E_Ensemble_5seed_test", "test"),
        ("E_Ensemble_5seed_oos", "oos"),
        ("E_Ensemble_5seed_val", "val"),
        ("E_Ensemble_5seed_train", "train"),
    ]

    header = f"{'Split':<8} {'Old MAE':>8} {'New MAE':>8} {'Delta':>8}  {'Old RMSE':>9} {'New RMSE':>9}  {'Old R2':>7} {'New R2':>7}"
    logger.info(header)
    logger.info("-" * 90)

    for key, split_label in comparison_keys:
        new_r = all_results.get(key, {})
        old_r = old_results.get(key, {})

        new_mae = new_r.get("mae", float("nan"))
        old_mae = old_r.get("mae", float("nan"))
        delta_mae = new_mae - old_mae if not (np.isnan(new_mae) or np.isnan(old_mae)) else float("nan")

        new_rmse = new_r.get("rmse", float("nan"))
        old_rmse = old_r.get("rmse", float("nan"))

        new_r2 = new_r.get("r2", float("nan"))
        old_r2 = old_r.get("r2", float("nan"))

        old_mae_str = f"{old_mae:.3f}" if not np.isnan(old_mae) else "N/A"
        new_mae_str = f"{new_mae:.3f}" if not np.isnan(new_mae) else "N/A"
        delta_str = f"{delta_mae:+.3f}" if not np.isnan(delta_mae) else "N/A"
        old_rmse_str = f"{old_rmse:.3f}" if not np.isnan(old_rmse) else "N/A"
        new_rmse_str = f"{new_rmse:.3f}" if not np.isnan(new_rmse) else "N/A"
        old_r2_str = f"{old_r2:.3f}" if not np.isnan(old_r2) else "N/A"
        new_r2_str = f"{new_r2:.3f}" if not np.isnan(new_r2) else "N/A"

        logger.info("%-8s %8s %8s %8s  %9s %9s  %7s %7s",
                     split_label, old_mae_str, new_mae_str, delta_str,
                     old_rmse_str, new_rmse_str, old_r2_str, new_r2_str)

    # Seasonal comparison for test and OOS
    logger.info("\n--- SEASONAL BREAKDOWN ---")
    for split_label in ["test", "oos"]:
        new_key = f"E_Ensemble_5seed_{split_label}"
        old_key = new_key
        new_r = all_results.get(new_key, {})
        old_r = old_results.get(old_key, {})
        logger.info("  %s:", split_label.upper())
        for season in ["DJF", "MAM", "JJA", "SON"]:
            new_s = new_r.get(f"mae_{season}", float("nan"))
            old_s = old_r.get(f"mae_{season}", float("nan"))
            if isinstance(old_s, str):
                try:
                    old_s = float(old_s)
                except (ValueError, TypeError):
                    old_s = float("nan")
            delta_s = new_s - old_s if not (np.isnan(new_s) or np.isnan(old_s)) else float("nan")
            new_str = f"{new_s:.3f}" if not np.isnan(new_s) else "N/A"
            old_str = f"{old_s:.3f}" if not np.isnan(old_s) else "N/A"
            delta_str = f"{delta_s:+.3f}" if not np.isnan(delta_s) else "N/A"
            logger.info("    %s: Old=%-7s  New=%-7s  Delta=%s", season, old_str, new_str, delta_str)

    # Per-seed comparison for test
    logger.info("\n--- PER-SEED BREAKDOWN (Test) ---")
    for seed in ENSEMBLE_SEEDS:
        key = f"E_seed_{seed}_test"
        r = all_results.get(key, {})
        mae = r.get("mae", float("nan"))
        logger.info("  Seed %4d: Test MAE=%.3f", seed, mae)
    ens_test = all_results.get("E_Ensemble_5seed_test", {}).get("mae", float("nan"))
    logger.info("  Ensemble:  Test MAE=%.3f", ens_test)

    logger.info("\n--- PER-SEED BREAKDOWN (OOS) ---")
    for seed in ENSEMBLE_SEEDS:
        key = f"E_seed_{seed}_oos"
        r = all_results.get(key, {})
        mae = r.get("mae", float("nan"))
        logger.info("  Seed %4d: OOS  MAE=%.3f", seed, mae)
    ens_oos = all_results.get("E_Ensemble_5seed_oos", {}).get("mae", float("nan"))
    logger.info("  Ensemble:  OOS  MAE=%.3f", ens_oos)


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("RETRAIN EXTENDED MOS: A_NN_64_32 5-seed Ensemble")
    logger.info("=" * 80)
    logger.info("Device: %s", DEVICE)
    logger.info("MOS source: %s", MOS_PATH)
    logger.info("Era indicator: %s", ERA_PATH)
    logger.info("Splits: Train=%s to %s, Val=%s to %s, Test=%s to %s, OOS=%s to %s",
                MOS_TRAIN_START, MOS_TRAIN_END, VAL_START, VAL_END,
                TEST_START, TEST_END, OOS_START, OOS_END)
    logger.info("Ensemble seeds: %s", ENSEMBLE_SEEDS)
    logger.info("Weight decay: %s", DEFAULT_WD)

    # ---- Step 1: Download station data ----
    logger.info("\n=== STEP 1: Downloading station .dly files ===")
    download_all_stations()

    # ---- Step 2: Build station matrix ----
    logger.info("\n=== STEP 2: Building station observation matrix ===")
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # ---- Step 3: Load MOS and Central Park data ----
    logger.info("\n=== STEP 3: Loading extended MOS and Central Park data ===")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # ---- Step 4: Build combined feature dataset ----
    logger.info("\n=== STEP 4: Building combined feature dataset (with era indicator) ===")
    builder = CombinedDatasetBuilder(station_matrix, mos_data, cp_data)
    data = builder.build_correction_data()

    # ---- Step 5: Train 5-seed ensemble ----
    logger.info("\n=== STEP 5: Training 5-seed ensemble ===")
    all_results = {}
    ensemble_result = run_ensemble(data, all_results)

    # ---- Step 6: Save outputs ----
    logger.info("\n=== STEP 6: Saving outputs ===")

    # Save all predictions
    save_all_predictions(data, ensemble_result)

    # Save benchmark prediction files
    sigma_by_month = build_benchmark_files(data, ensemble_result)

    # Save metrics summary
    summary_df = build_metrics_summary(all_results)

    # Save full results JSON
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    clean_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: make_serializable(vv) for kk, vv in v.items()}
        else:
            clean_results[k] = make_serializable(v)

    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info("Saved full results to %s", json_path)

    # Save ensemble checkpoint
    ensemble_ckpt_path = os.path.join(RESULTS_DIR, "ensemble_5seed.pt")
    torch.save({
        "model_name": "A_NN_64_32_extended",
        "hidden_sizes": [64, 32],
        "dropout": 0.15,
        "n_features": data["n_features"],
        "feature_names": data["feature_names"],
        "seed_state_dicts": ensemble_result["seed_state_dicts"],
        "splits": {
            "train": f"{MOS_TRAIN_START} to {MOS_TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
            "oos": f"{OOS_START} to {OOS_END}",
        },
    }, ensemble_ckpt_path)
    logger.info("Saved ensemble checkpoint to %s", ensemble_ckpt_path)

    # Save scaler
    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(data.get("scaler"), f)
    logger.info("Saved scaler to %s", scaler_path)

    # ---- Step 7: Comparison ----
    print_comparison(all_results)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("Total pipeline time: %.1f minutes (%.0f seconds)", elapsed / 60, elapsed)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
