#!/usr/bin/env python3
"""
Retrain Extended Validation: Retrain A_NN_64_32 with extended validation split,
implement sigma recalibration, regime-conditional variance, and comprehensive
quality metrics.

Splits:
  Train: 2000-06-01 to 2019-12-31
  Val:   2020-01-01 to 2022-12-31  (3 years — extended from 2 years)
  Test:  2023-01-01 to 2024-12-31  (2 years — used for calibration in benchmark)

Parts:
  1. Retrain 5-seed ensemble with new splits
  2. Sigma recalibration (monthly, regime, combined)
  3. Regime-conditional variance modeling
  4. Comprehensive quality metrics (Brier decomposition, PIT, CRPS, ECE, etc.)

Outputs saved to: results/retrain_extended_validation/
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
from scipy.stats import norm as scipy_norm, kstest, uniform as scipy_uniform

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
logger = logging.getLogger("retrain_extended_validation")

# ---------------------------------------------------------------------------
# Constants — UPDATED SPLITS
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_extended.csv")
ERA_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "mos_era_indicator.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "retrain_extended_validation")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# NEW chronological splits
MOS_TRAIN_START, MOS_TRAIN_END = "2000-06-01", "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_SEED = 42
DEFAULT_WD = 1e-4
SIGMA_FLOOR = 0.75
SIGMA_CAP = 10.0
NYC_LAT = 40.7831
PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. DATA LOADING (identical to retrain_extended_mos.py)
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
# 2. FEATURE ENGINEERING HELPERS (identical to retrain_extended_mos.py)
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
# 3. MODEL DEFINITION (identical to retrain_extended_mos.py)
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
# 4. TRAINING UTILITIES (identical to retrain_extended_mos.py)
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
# 6. COMBINED DATASET BUILDER (with UPDATED splits)
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

        # Ensure mos_era is present
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

        self._feature_cols = (
            base_features
            + PHASE_1A_FEATURES
            + PHASE_1C_FEATURES
            + TEMPORAL_FEATURES
            + SPATIAL_FEATURES
            + ["mos_era"]
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
                "gfs_mos": sub["gfs_mos_tmax_f"].values.astype(np.float64),
                "nam_mos": sub["nam_mos_tmax_f"].values.astype(np.float64),
            }

        # Impute NaNs with training mean, then StandardScaler
        X_tr = arrays["train"]["X"]
        X_v = arrays["val"]["X"]
        X_te = arrays["test"]["X"]

        train_means = np.nanmean(X_tr, axis=0)
        train_means = np.where(np.isnan(train_means), 0.0, train_means)
        for arr in [X_tr, X_v, X_te]:
            for j in range(arr.shape[1]):
                mask = np.isnan(arr[:, j])
                arr[mask, j] = train_means[j]

        scaler = StandardScaler()
        arrays["train"]["X"] = scaler.fit_transform(X_tr)
        arrays["val"]["X"] = scaler.transform(X_v)
        arrays["test"]["X"] = scaler.transform(X_te)
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        arrays["n_features"] = len(valid_cols)

        logger.info("Dataset built: Train=%d, Val=%d, Test=%d, Features=%d",
                     len(arrays["train"]["y_resid"]),
                     len(arrays["val"]["y_resid"]),
                     len(arrays["test"]["y_resid"]),
                     arrays["n_features"])

        return arrays


# ============================================================================
# 7. PART 1: 5-SEED ENSEMBLE TRAINING
# ============================================================================

def run_ensemble(data, all_results):
    """Train 5-seed ensemble of NN [64,32] and evaluate."""
    logger.info("=" * 70)
    logger.info("PART 1: Training 5-seed Ensemble of A_NN_64_32 (extended val)")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    per_seed_preds = {split: [] for split in ["train", "val", "test"]}
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

        for split in ["train", "val", "test"]:
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
    for split in ["train", "val", "test"]:
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


# ============================================================================
# 8. PART 2: SIGMA RECALIBRATION
# ============================================================================

def compute_base_sigma(data, ensemble_preds, ensemble_stds, split):
    """Compute base sigma from ensemble spread and month-based residual std (fit on val)."""
    dates = pd.to_datetime(data[split]["dates"])
    actual = data[split]["y_actual"]
    pred = ensemble_preds[split]
    errors = actual - pred

    # Month-based sigma (fit on validation set)
    val_dates = pd.to_datetime(data["val"]["dates"])
    val_actual = data["val"]["y_actual"]
    val_pred = ensemble_preds["val"]
    val_errors = val_actual - val_pred

    sigma_by_month = {}
    for month in range(1, 13):
        m = val_dates.month == month
        if np.any(m):
            sigma_by_month[month] = float(np.std(val_errors[m], ddof=1))

    global_sigma = float(np.std(val_errors, ddof=1))
    sigma_by_month = {
        m: float(np.clip(sigma_by_month.get(m, global_sigma), SIGMA_FLOOR, SIGMA_CAP))
        for m in range(1, 13)
    }

    # Base sigma: max of month-based and ensemble spread
    sigma_month_arr = np.array([sigma_by_month[int(m)] for m in dates.month])
    sigma_ensemble_arr = ensemble_stds[split]
    sigma = np.maximum(sigma_month_arr, sigma_ensemble_arr)
    sigma = np.clip(sigma, SIGMA_FLOOR, SIGMA_CAP)
    return sigma, sigma_by_month


def sigma_recalibration(data, ensemble_preds, ensemble_stds):
    """
    Part 2: Sigma recalibration using validation data.

    Computes:
    1. Monthly sigma calibration factors
    2. Regime-aware sigma calibration factors
    3. Combined (month x regime) sigma calibration factors
    """
    logger.info("=" * 70)
    logger.info("PART 2: Sigma Recalibration")
    logger.info("=" * 70)

    val_dates = pd.to_datetime(data["val"]["dates"])
    val_actual = data["val"]["y_actual"]
    val_pred = ensemble_preds["val"]
    val_errors = val_actual - val_pred
    val_sigma_base, sigma_by_month = compute_base_sigma(
        data, ensemble_preds, ensemble_stds, "val"
    )

    # --- 1. Monthly sigma calibration ---
    logger.info("--- Monthly Sigma Calibration ---")
    monthly_cal = {}
    for month in range(1, 13):
        m = val_dates.month == month
        if np.any(m):
            actual_std = float(np.std(val_errors[m], ddof=1))
            predicted_sigma = float(np.mean(val_sigma_base[m]))
            if predicted_sigma > 0:
                scale = actual_std / predicted_sigma
            else:
                scale = 1.0
            monthly_cal[month] = {
                "actual_residual_std": round(actual_std, 4),
                "predicted_sigma_mean": round(predicted_sigma, 4),
                "scale_factor": round(scale, 4),
            }
            logger.info("  Month %2d: actual_std=%.3f  pred_sigma=%.3f  scale=%.3f",
                        month, actual_std, predicted_sigma, scale)
        else:
            monthly_cal[month] = {"actual_residual_std": 0, "predicted_sigma_mean": 0, "scale_factor": 1.0}

    # --- 2. Regime-based sigma calibration ---
    logger.info("--- Regime-Aware Sigma Calibration ---")

    # Classify regimes based on abs day-over-day mu change
    val_mu_change = np.abs(np.diff(val_pred, prepend=val_pred[0]))
    p33, p66 = np.percentile(val_mu_change, [33, 66])

    def classify_regime(mu_change_arr, p33_val, p66_val):
        regimes = np.full(len(mu_change_arr), "transition", dtype=object)
        regimes[mu_change_arr <= p33_val] = "stable"
        regimes[mu_change_arr >= p66_val] = "volatile"
        return regimes

    val_regimes = classify_regime(val_mu_change, p33, p66)

    regime_cal = {}
    for regime in ["stable", "transition", "volatile"]:
        m = val_regimes == regime
        if np.any(m):
            actual_std = float(np.std(val_errors[m], ddof=1))
            predicted_sigma = float(np.mean(val_sigma_base[m]))
            if predicted_sigma > 0:
                scale = actual_std / predicted_sigma
            else:
                scale = 1.0
            regime_cal[regime] = {
                "count": int(np.sum(m)),
                "actual_residual_std": round(actual_std, 4),
                "predicted_sigma_mean": round(predicted_sigma, 4),
                "scale_factor": round(scale, 4),
            }
            logger.info("  Regime %11s (n=%d): actual_std=%.3f  pred_sigma=%.3f  scale=%.3f",
                        regime, np.sum(m), actual_std, predicted_sigma, scale)

    # --- 3. Combined (month x regime) calibration ---
    logger.info("--- Combined Month x Regime Calibration ---")
    combined_cal = {}
    for month in range(1, 13):
        combined_cal[month] = {}
        for regime in ["stable", "transition", "volatile"]:
            m = (val_dates.month == month) & (val_regimes == regime)
            if np.sum(m) >= 5:
                actual_std = float(np.std(val_errors[m], ddof=1))
                predicted_sigma = float(np.mean(val_sigma_base[m]))
                if predicted_sigma > 0:
                    scale = actual_std / predicted_sigma
                else:
                    scale = 1.0
            else:
                # Fall back to monthly or regime
                m_scale = monthly_cal[month]["scale_factor"]
                r_scale = regime_cal.get(regime, {}).get("scale_factor", 1.0)
                scale = (m_scale + r_scale) / 2.0
            combined_cal[month][regime] = round(scale, 4)

    # Save calibration data
    calibration = {
        "sigma_by_month": {str(k): v for k, v in sigma_by_month.items()},
        "monthly_calibration": {str(k): v for k, v in monthly_cal.items()},
        "regime_calibration": regime_cal,
        "regime_thresholds": {"p33": round(float(p33), 4), "p66": round(float(p66), 4)},
        "combined_calibration": {str(k): v for k, v in combined_cal.items()},
    }

    return calibration, sigma_by_month, monthly_cal, regime_cal, combined_cal, (p33, p66)


def apply_sigma_calibration(sigma_base, dates, mu_preds, monthly_cal, regime_cal,
                             combined_cal, regime_thresholds, mode="combined"):
    """Apply sigma calibration to predictions."""
    dates = pd.to_datetime(dates)
    p33, p66 = regime_thresholds
    mu_change = np.abs(np.diff(mu_preds, prepend=mu_preds[0]))

    regimes = np.full(len(mu_change), "transition", dtype=object)
    regimes[mu_change <= p33] = "stable"
    regimes[mu_change >= p66] = "volatile"

    calibrated = np.copy(sigma_base)
    for i in range(len(calibrated)):
        month = int(dates[i].month)
        regime = regimes[i]
        if mode == "monthly":
            factor = monthly_cal[month]["scale_factor"]
        elif mode == "regime":
            factor = regime_cal.get(regime, {}).get("scale_factor", 1.0)
        elif mode == "combined":
            factor = combined_cal.get(month, {}).get(regime, 1.0)
        else:
            factor = 1.0
        calibrated[i] = calibrated[i] * factor

    calibrated = np.clip(calibrated, SIGMA_FLOOR, SIGMA_CAP)
    return calibrated


# ============================================================================
# 9. PART 3: REGIME-CONDITIONAL VARIANCE MODELING
# ============================================================================

def regime_conditional_variance(data, ensemble_preds, ensemble_stds, sigma_by_month):
    """
    Part 3: Regime-conditional variance modeling.

    Classifies days into regimes using multiple features, then fits separate
    sigma models per regime on validation data.
    """
    logger.info("=" * 70)
    logger.info("PART 3: Regime-Conditional Variance Modeling")
    logger.info("=" * 70)

    val_dates = pd.to_datetime(data["val"]["dates"])
    val_actual = data["val"]["y_actual"]
    val_pred = ensemble_preds["val"]
    val_errors = val_actual - val_pred
    val_gfs = data["val"]["gfs_mos"]
    val_nam = data["val"]["nam_mos"]

    # Regime features
    mos_spread = np.abs(val_gfs - val_nam)
    station_consensus = ensemble_stds["val"]
    dod_change = np.abs(np.diff(val_pred, prepend=val_pred[0]))
    month = val_dates.month
    season = np.where(np.isin(month, [12, 1, 2]), 0,
             np.where(np.isin(month, [3, 4, 5]), 1,
             np.where(np.isin(month, [6, 7, 8]), 2, 3)))

    # Multi-dimensional regime classification
    # Use percentile-based cutoffs
    spread_high = np.percentile(mos_spread, 70)
    consensus_high = np.percentile(station_consensus, 70)
    dod_high = np.percentile(dod_change, 70)

    def classify_variance_regime(mos_spread_arr, consensus_arr, dod_arr, season_arr):
        """Classify into regimes: low_var, medium_var, high_var, seasonal_transition."""
        n = len(mos_spread_arr)
        regimes = np.full(n, "medium_var", dtype=object)

        # High variance: any two of three indicators are high
        high_count = ((mos_spread_arr > spread_high).astype(int) +
                      (consensus_arr > consensus_high).astype(int) +
                      (dod_arr > dod_high).astype(int))
        regimes[high_count >= 2] = "high_var"

        # Low variance: all three below median
        spread_med = np.median(mos_spread_arr)
        consensus_med = np.median(consensus_arr)
        dod_med = np.median(dod_arr)
        low_mask = ((mos_spread_arr <= spread_med) &
                    (consensus_arr <= consensus_med) &
                    (dod_arr <= dod_med))
        regimes[low_mask] = "low_var"

        # Seasonal transition (MAM and SON)
        regimes[np.isin(season_arr, [1, 3]) & (regimes == "medium_var")] = "seasonal_transition"

        return regimes

    val_var_regimes = classify_variance_regime(mos_spread, station_consensus, dod_change, season)

    # Fit sigma per variance regime
    regime_sigma_models = {}
    for regime in ["low_var", "medium_var", "high_var", "seasonal_transition"]:
        m = val_var_regimes == regime
        if np.any(m):
            resid_std = float(np.std(val_errors[m], ddof=1))
            resid_mean = float(np.mean(val_errors[m]))
            resid_abs_mean = float(np.mean(np.abs(val_errors[m])))
            regime_sigma_models[regime] = {
                "count": int(np.sum(m)),
                "sigma": round(max(resid_std, SIGMA_FLOOR), 4),
                "mean_bias": round(resid_mean, 4),
                "mae": round(resid_abs_mean, 4),
            }
            logger.info("  Regime %-22s (n=%3d): sigma=%.3f, bias=%.3f, MAE=%.3f",
                        regime, np.sum(m), resid_std, resid_mean, resid_abs_mean)

    # Apply to test set
    test_dates = pd.to_datetime(data["test"]["dates"])
    test_actual = data["test"]["y_actual"]
    test_pred = ensemble_preds["test"]
    test_errors = test_actual - test_pred
    test_gfs = data["test"]["gfs_mos"]
    test_nam = data["test"]["nam_mos"]

    test_mos_spread = np.abs(test_gfs - test_nam)
    test_consensus = ensemble_stds["test"]
    test_dod = np.abs(np.diff(test_pred, prepend=test_pred[0]))
    test_month = test_dates.month
    test_season = np.where(np.isin(test_month, [12, 1, 2]), 0,
                  np.where(np.isin(test_month, [3, 4, 5]), 1,
                  np.where(np.isin(test_month, [6, 7, 8]), 2, 3)))

    test_var_regimes = classify_variance_regime(
        test_mos_spread, test_consensus, test_dod, test_season
    )

    # Build regime-conditional sigma for test
    test_regime_sigma = np.full(len(test_pred), float(np.std(val_errors, ddof=1)))
    for i in range(len(test_pred)):
        regime = test_var_regimes[i]
        if regime in regime_sigma_models:
            test_regime_sigma[i] = regime_sigma_models[regime]["sigma"]
    test_regime_sigma = np.clip(test_regime_sigma, SIGMA_FLOOR, SIGMA_CAP)

    # Evaluate regime-conditional vs base sigma
    test_sigma_base, _ = compute_base_sigma(data, ensemble_preds, ensemble_stds, "test")

    logger.info("\n--- Test Set Regime Distribution ---")
    for regime in ["low_var", "medium_var", "high_var", "seasonal_transition"]:
        m = test_var_regimes == regime
        if np.any(m):
            logger.info("  %s: n=%d, actual_std=%.3f, regime_sigma=%.3f, base_sigma=%.3f",
                        regime, np.sum(m),
                        float(np.std(test_errors[m], ddof=1)),
                        float(np.mean(test_regime_sigma[m])),
                        float(np.mean(test_sigma_base[m])))

    return {
        "regime_sigma_models": regime_sigma_models,
        "thresholds": {
            "spread_high": round(float(spread_high), 4),
            "consensus_high": round(float(consensus_high), 4),
            "dod_high": round(float(dod_high), 4),
        },
        "test_regime_sigma": test_regime_sigma,
        "test_var_regimes": test_var_regimes,
        "val_var_regimes": val_var_regimes,
    }


# ============================================================================
# 10. PART 4: COMPREHENSIVE QUALITY METRICS
# ============================================================================

def compute_crps_gaussian(mu, sigma, actual):
    """CRPS for Gaussian distribution."""
    z = (actual - mu) / sigma
    crps = sigma * (z * (2 * scipy_norm.cdf(z) - 1) +
                    2 * scipy_norm.pdf(z) - 1 / np.sqrt(np.pi))
    return crps


def compute_pit_values(mu, sigma, actual):
    """Probability Integral Transform values."""
    return scipy_norm.cdf((actual - mu) / sigma)


def compute_log_score(mu, sigma, actual):
    """Negative log-likelihood (log score)."""
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((actual - mu) / sigma)**2
    return nll


def compute_prediction_interval_coverage(mu, sigma, actual, levels=None):
    """Compute coverage for prediction intervals at various levels."""
    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95]
    results = {}
    for level in levels:
        alpha = 1 - level
        z = scipy_norm.ppf(1 - alpha / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        width = upper - lower
        covered = (actual >= lower) & (actual <= upper)
        coverage = float(np.mean(covered))
        avg_width = float(np.mean(width))
        results[f"{int(level*100)}%"] = {
            "target_coverage": level,
            "actual_coverage": round(coverage, 4),
            "coverage_error": round(coverage - level, 4),
            "avg_width_F": round(avg_width, 2),
            "median_width_F": round(float(np.median(width)), 2),
        }
    return results


def compute_reliability_diagram_data(mu, sigma, actual, n_bins=10):
    """Compute reliability diagram data for probabilistic forecasts.

    For each bin, compute predicted probability vs observed frequency
    of the event: actual > mu (i.e., above-median forecast).
    """
    # Use the CDF-based approach: P(actual <= threshold)
    # We compute P(actual <= actual_median) and check if it's calibrated
    predicted_cdf = scipy_norm.cdf((actual - mu) / sigma)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_freq = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (predicted_cdf >= lo) & (predicted_cdf < hi)
        if i == n_bins - 1:  # include right edge
            mask = (predicted_cdf >= lo) & (predicted_cdf <= hi)
        n_in_bin = int(np.sum(mask))
        if n_in_bin > 0:
            # For a well-calibrated forecast, the mean predicted CDF in each bin
            # should match the fraction of observations in the bin
            mean_predicted = float(np.mean(predicted_cdf[mask]))
            # Observed: what fraction of these points actually fell at or below
            # the predicted CDF value (a perfectly calibrated forecast has this
            # equal to mean_predicted)
            bin_centers.append(round(mean_predicted, 4))
            observed_freq.append(round(float(np.mean(mask)), 4))
            bin_counts.append(n_in_bin)
        else:
            bin_centers.append(round((lo + hi) / 2, 4))
            observed_freq.append(0.0)
            bin_counts.append(0)

    return {
        "bin_centers": bin_centers,
        "observed_frequency": observed_freq,
        "bin_counts": bin_counts,
    }


def compute_ece(mu, sigma, actual, n_bins=10):
    """Expected Calibration Error."""
    predicted_cdf = scipy_norm.cdf((actual - mu) / sigma)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(actual)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (predicted_cdf >= lo) & (predicted_cdf < hi)
        if i == n_bins - 1:
            mask = (predicted_cdf >= lo) & (predicted_cdf <= hi)
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            mean_predicted = np.mean(predicted_cdf[mask])
            observed = np.sum(predicted_cdf[mask] <= mean_predicted) / n_in_bin
            ece += (n_in_bin / total) * abs(mean_predicted - observed)
    return float(ece)


def compute_brier_decomposition(mu, sigma, actual, thresholds=None):
    """Brier score decomposition for selected temperature thresholds.

    Decomposes into reliability, resolution, and uncertainty.
    """
    if thresholds is None:
        # Use quartiles of actual as thresholds
        thresholds = [
            float(np.percentile(actual, 25)),
            float(np.percentile(actual, 50)),
            float(np.percentile(actual, 75)),
        ]

    results = {}
    for threshold in thresholds:
        # Predicted probability that actual exceeds threshold
        pred_prob = 1 - scipy_norm.cdf((threshold - mu) / sigma)
        pred_prob = np.clip(pred_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)

        # Observed outcome
        observed = (actual > threshold).astype(float)

        # Overall Brier score
        brier = float(np.mean((pred_prob - observed)**2))

        # Climatological frequency
        climo_freq = float(np.mean(observed))

        # Uncertainty component: climo_freq * (1 - climo_freq)
        uncertainty = climo_freq * (1 - climo_freq)

        # Binned reliability and resolution
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        reliability = 0.0
        resolution = 0.0
        total = len(actual)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (pred_prob >= lo) & (pred_prob < hi)
            if i == n_bins - 1:
                mask = (pred_prob >= lo) & (pred_prob <= hi)
            n_k = np.sum(mask)
            if n_k > 0:
                mean_pred_k = np.mean(pred_prob[mask])
                obs_freq_k = np.mean(observed[mask])
                reliability += (n_k / total) * (mean_pred_k - obs_freq_k)**2
                resolution += (n_k / total) * (obs_freq_k - climo_freq)**2

        results[f"threshold_{threshold:.0f}F"] = {
            "threshold_F": round(threshold, 1),
            "brier_score": round(brier, 6),
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
            "uncertainty": round(uncertainty, 6),
            "brier_skill_score": round(1 - brier / max(uncertainty, 1e-10), 4),
            "climatological_frequency": round(climo_freq, 4),
        }

    return results


def run_quality_metrics(data, ensemble_preds, ensemble_stds, calibration,
                        regime_result, sigma_by_month, monthly_cal, regime_cal,
                        combined_cal, regime_thresholds):
    """
    Part 4: Comprehensive quality metrics.
    """
    logger.info("=" * 70)
    logger.info("PART 4: Comprehensive Quality Metrics")
    logger.info("=" * 70)

    test_dates = pd.to_datetime(data["test"]["dates"])
    test_actual = data["test"]["y_actual"]
    test_mu = ensemble_preds["test"]
    test_errors = test_actual - test_mu

    # Compute different sigma variants
    test_sigma_base, _ = compute_base_sigma(data, ensemble_preds, ensemble_stds, "test")

    test_sigma_monthly = apply_sigma_calibration(
        test_sigma_base, test_dates, test_mu, monthly_cal, regime_cal,
        combined_cal, regime_thresholds, mode="monthly"
    )
    test_sigma_regime = apply_sigma_calibration(
        test_sigma_base, test_dates, test_mu, monthly_cal, regime_cal,
        combined_cal, regime_thresholds, mode="regime"
    )
    test_sigma_combined = apply_sigma_calibration(
        test_sigma_base, test_dates, test_mu, monthly_cal, regime_cal,
        combined_cal, regime_thresholds, mode="combined"
    )
    test_sigma_regime_cond = regime_result["test_regime_sigma"]

    sigma_variants = {
        "base": test_sigma_base,
        "monthly_cal": test_sigma_monthly,
        "regime_cal": test_sigma_regime,
        "combined_cal": test_sigma_combined,
        "regime_conditional": test_sigma_regime_cond,
    }

    all_metrics = {}

    for variant_name, sigma in sigma_variants.items():
        logger.info("\n--- Metrics for sigma variant: %s ---", variant_name)
        metrics = {}

        # Point prediction metrics
        metrics["mae"] = round(float(mean_absolute_error(test_actual, test_mu)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(test_actual, test_mu))), 4)
        metrics["r2"] = round(float(r2_score(test_actual, test_mu)), 4)
        metrics["mean_sigma"] = round(float(np.mean(sigma)), 4)
        metrics["median_sigma"] = round(float(np.median(sigma)), 4)

        # CRPS
        crps = compute_crps_gaussian(test_mu, sigma, test_actual)
        metrics["crps_mean"] = round(float(np.mean(crps)), 4)
        metrics["crps_median"] = round(float(np.median(crps)), 4)
        logger.info("  CRPS: mean=%.4f, median=%.4f", np.mean(crps), np.median(crps))

        # Log Score
        log_scores = compute_log_score(test_mu, sigma, test_actual)
        metrics["log_score_mean"] = round(float(np.mean(log_scores)), 4)
        logger.info("  Log Score (mean NLL): %.4f", np.mean(log_scores))

        # PIT histogram and uniformity test
        pit = compute_pit_values(test_mu, sigma, test_actual)
        ks_stat, ks_pvalue = kstest(pit, 'uniform')
        metrics["pit_ks_statistic"] = round(float(ks_stat), 4)
        metrics["pit_ks_pvalue"] = round(float(ks_pvalue), 4)
        # PIT histogram counts
        pit_hist, _ = np.histogram(pit, bins=10, range=(0, 1))
        metrics["pit_histogram"] = pit_hist.tolist()
        logger.info("  PIT KS test: stat=%.4f, p=%.4f", ks_stat, ks_pvalue)

        # Coverage analysis
        coverage = compute_prediction_interval_coverage(test_mu, sigma, test_actual)
        metrics["coverage"] = coverage
        for level_name, cov_data in coverage.items():
            logger.info("  %s PI: coverage=%.3f (target=%.2f), width=%.2f F",
                        level_name, cov_data["actual_coverage"],
                        cov_data["target_coverage"], cov_data["avg_width_F"])

        # Sharpness (average PI widths)
        metrics["sharpness_95pct_width"] = coverage.get("95%", {}).get("avg_width_F", 0)
        metrics["sharpness_90pct_width"] = coverage.get("90%", {}).get("avg_width_F", 0)

        # ECE
        ece = compute_ece(test_mu, sigma, test_actual)
        metrics["ece"] = round(ece, 4)
        logger.info("  ECE: %.4f", ece)

        # Reliability diagram data
        reliability_data = compute_reliability_diagram_data(test_mu, sigma, test_actual)
        metrics["reliability_diagram"] = reliability_data

        # Brier score decomposition
        brier_decomp = compute_brier_decomposition(test_mu, sigma, test_actual)
        metrics["brier_decomposition"] = brier_decomp
        for thresh_name, brier_data in brier_decomp.items():
            logger.info("  Brier %s: score=%.6f, reliability=%.6f, resolution=%.6f, BSS=%.4f",
                        thresh_name, brier_data["brier_score"],
                        brier_data["reliability"], brier_data["resolution"],
                        brier_data["brier_skill_score"])

        # Seasonal CRPS
        seasons = assign_season(test_dates)
        seasonal_crps = {}
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons == s
            if mask.sum() > 0:
                seasonal_crps[s] = round(float(np.mean(crps[mask.values])), 4)
        metrics["seasonal_crps"] = seasonal_crps

        all_metrics[variant_name] = metrics

    return all_metrics, sigma_variants


# ============================================================================
# 11. OUTPUT GENERATION
# ============================================================================

def generate_outputs(data, ensemble_result, calibration, regime_result,
                     all_metrics, sigma_variants, all_results, sigma_by_month):
    """Generate all required output files."""
    logger.info("=" * 70)
    logger.info("GENERATING OUTPUTS")
    logger.info("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ensemble_preds = ensemble_result["ensemble_preds"]
    ensemble_stds = ensemble_result["ensemble_stds"]

    # 1. Save ensemble checkpoint
    ensemble_ckpt_path = os.path.join(RESULTS_DIR, "ensemble_5seed.pt")
    torch.save({
        "model_name": "A_NN_64_32_extended_val",
        "hidden_sizes": [64, 32],
        "dropout": 0.15,
        "n_features": data["n_features"],
        "feature_names": data["feature_names"],
        "seed_state_dicts": ensemble_result["seed_state_dicts"],
        "splits": {
            "train": f"{MOS_TRAIN_START} to {MOS_TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
    }, ensemble_ckpt_path)
    logger.info("Saved ensemble checkpoint: %s", ensemble_ckpt_path)

    # 2. Save scaler
    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(data.get("scaler"), f)
    logger.info("Saved scaler: %s", scaler_path)

    # 3. Save sigma_by_month
    sigma_path = os.path.join(RESULTS_DIR, "sigma_by_month.json")
    with open(sigma_path, "w") as f:
        json.dump({str(k): v for k, v in sigma_by_month.items()}, f, indent=2)
    logger.info("Saved sigma_by_month: %s", sigma_path)

    # 4. Save sigma calibration
    cal_path = os.path.join(RESULTS_DIR, "sigma_calibration.json")
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info("Saved sigma calibration: %s", cal_path)

    # 5. Save predictions (test)
    test_dates = pd.to_datetime(data["test"]["dates"])
    best_sigma = sigma_variants["combined_cal"]

    pred_test_df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["test"]["y_actual"],
        "mos_base": data["test"]["mos_base"],
        "model_mu": ensemble_preds["test"],
        "model_sigma_base": sigma_variants["base"],
        "model_sigma_monthly_cal": sigma_variants["monthly_cal"],
        "model_sigma_regime_cal": sigma_variants["regime_cal"],
        "model_sigma_combined_cal": sigma_variants["combined_cal"],
        "model_sigma_regime_conditional": sigma_variants["regime_conditional"],
        "ensemble_std": ensemble_stds["test"],
        "regime": regime_result["test_var_regimes"],
    })
    pred_test_path = os.path.join(RESULTS_DIR, "predictions_test.csv")
    pred_test_df.to_csv(pred_test_path, index=False)
    logger.info("Saved test predictions: %s (%d rows)", pred_test_path, len(pred_test_df))

    # 6. Save predictions (val)
    val_dates = pd.to_datetime(data["val"]["dates"])
    val_sigma_base, _ = compute_base_sigma(data, ensemble_preds, ensemble_stds, "val")

    pred_val_df = pd.DataFrame({
        "date": val_dates.strftime("%Y-%m-%d"),
        "actual_tmax": data["val"]["y_actual"],
        "mos_base": data["val"]["mos_base"],
        "model_mu": ensemble_preds["val"],
        "model_sigma_base": val_sigma_base,
        "ensemble_std": ensemble_stds["val"],
        "regime": regime_result["val_var_regimes"],
    })
    pred_val_path = os.path.join(RESULTS_DIR, "predictions_val.csv")
    pred_val_df.to_csv(pred_val_path, index=False)
    logger.info("Saved val predictions: %s (%d rows)", pred_val_path, len(pred_val_df))

    # 7. Save quality metrics JSON
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

    def deep_serialize(obj):
        if isinstance(obj, dict):
            return {str(k): deep_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_serialize(v) for v in obj]
        return make_serializable(obj)

    metrics_json_path = os.path.join(RESULTS_DIR, "quality_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(deep_serialize(all_metrics), f, indent=2, default=str)
    logger.info("Saved quality metrics JSON: %s", metrics_json_path)

    # 8. Save regime model data
    regime_json_path = os.path.join(RESULTS_DIR, "regime_variance_models.json")
    regime_save = {
        "regime_sigma_models": regime_result["regime_sigma_models"],
        "thresholds": regime_result["thresholds"],
    }
    with open(regime_json_path, "w") as f:
        json.dump(deep_serialize(regime_save), f, indent=2, default=str)
    logger.info("Saved regime variance models: %s", regime_json_path)

    # 9. Save experiment results
    clean_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: make_serializable(vv) for kk, vv in v.items()}
        else:
            clean_results[k] = make_serializable(v)
    results_json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info("Saved experiment results: %s", results_json_path)

    # 10. Generate quality metrics markdown report
    generate_quality_report(all_metrics, all_results, calibration, regime_result, data,
                            ensemble_preds, ensemble_stds, sigma_by_month)


def generate_quality_report(all_metrics, all_results, calibration, regime_result,
                            data, ensemble_preds, ensemble_stds, sigma_by_month):
    """Generate comprehensive quality metrics markdown report."""
    lines = []
    lines.append("# Quality Metrics Report: Extended Validation Retrain")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Train Period | {MOS_TRAIN_START} to {MOS_TRAIN_END} |")
    lines.append(f"| Validation Period | {VAL_START} to {VAL_END} |")
    lines.append(f"| Test Period | {TEST_START} to {TEST_END} |")
    lines.append(f"| Architecture | A_NN_64_32 (FlexibleNN [64,32], BN, dropout=0.15) |")
    lines.append(f"| Ensemble Seeds | {ENSEMBLE_SEEDS} |")
    lines.append(f"| Train Size | {len(data['train']['y_resid'])} days |")
    lines.append(f"| Val Size | {len(data['val']['y_resid'])} days |")
    lines.append(f"| Test Size | {len(data['test']['y_resid'])} days |")
    lines.append(f"| Features | {data['n_features']} |")
    lines.append("")

    # Point prediction metrics
    lines.append("## Point Prediction Metrics (Test Set)")
    lines.append("")
    ens_test = all_results.get("E_Ensemble_5seed_test", {})
    lines.append(f"- **MAE**: {ens_test.get('mae', 'N/A')} F")
    lines.append(f"- **RMSE**: {ens_test.get('rmse', 'N/A')} F")
    lines.append(f"- **R2**: {ens_test.get('r2', 'N/A')}")
    lines.append("")

    # Per-seed breakdown
    lines.append("### Per-Seed Performance (Test)")
    lines.append("")
    lines.append("| Seed | MAE | RMSE | R2 |")
    lines.append("|------|-----|------|----|")
    for seed in ENSEMBLE_SEEDS:
        r = all_results.get(f"E_seed_{seed}_test", {})
        lines.append(f"| {seed} | {r.get('mae', 'N/A')} | {r.get('rmse', 'N/A')} | {r.get('r2', 'N/A')} |")
    lines.append(f"| **Ensemble** | **{ens_test.get('mae', 'N/A')}** | **{ens_test.get('rmse', 'N/A')}** | **{ens_test.get('r2', 'N/A')}** |")
    lines.append("")

    # Seasonal breakdown
    lines.append("### Seasonal MAE (Test)")
    lines.append("")
    lines.append("| Season | MAE |")
    lines.append("|--------|-----|")
    for s in ["DJF", "MAM", "JJA", "SON"]:
        val = ens_test.get(f"mae_{s}", "N/A")
        lines.append(f"| {s} | {val} |")
    lines.append("")

    # Sigma calibration results
    lines.append("## Sigma Recalibration Results")
    lines.append("")
    lines.append("### Monthly Sigma (from validation set)")
    lines.append("")
    lines.append("| Month | Base Sigma | Scale Factor |")
    lines.append("|-------|-----------|--------------|")
    monthly_cal = calibration.get("monthly_calibration", {})
    for m in range(1, 13):
        mc = monthly_cal.get(str(m), {})
        lines.append(f"| {m} | {mc.get('predicted_sigma_mean', 'N/A')} | {mc.get('scale_factor', 'N/A')} |")
    lines.append("")

    lines.append("### Regime Calibration")
    lines.append("")
    lines.append("| Regime | Count | Scale Factor | Actual Std |")
    lines.append("|--------|-------|--------------|------------|")
    regime_cal = calibration.get("regime_calibration", {})
    for regime in ["stable", "transition", "volatile"]:
        rc = regime_cal.get(regime, {})
        lines.append(f"| {regime} | {rc.get('count', 'N/A')} | {rc.get('scale_factor', 'N/A')} | {rc.get('actual_residual_std', 'N/A')} |")
    lines.append("")

    # Comprehensive metrics comparison
    lines.append("## Distributional Quality Metrics (Test Set)")
    lines.append("")
    lines.append("| Metric | Base | Monthly Cal | Regime Cal | Combined Cal | Regime-Cond |")
    lines.append("|--------|------|-------------|------------|--------------|-------------|")
    key_metrics = ["crps_mean", "log_score_mean", "pit_ks_statistic", "pit_ks_pvalue",
                   "ece", "mean_sigma", "sharpness_95pct_width"]
    metric_labels = ["CRPS (mean)", "Log Score (NLL)", "PIT KS stat", "PIT KS p-value",
                     "ECE", "Mean Sigma", "95% PI Width"]
    variants = ["base", "monthly_cal", "regime_cal", "combined_cal", "regime_conditional"]
    for label, key in zip(metric_labels, key_metrics):
        row = f"| {label} |"
        for var in variants:
            val = all_metrics.get(var, {}).get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        lines.append(row)
    lines.append("")

    # Coverage analysis
    lines.append("## Prediction Interval Coverage (Test Set)")
    lines.append("")
    lines.append("| Level | Target | Base | Monthly Cal | Combined Cal | Regime-Cond |")
    lines.append("|-------|--------|------|-------------|--------------|-------------|")
    for level in ["50%", "80%", "90%", "95%"]:
        target = float(level.replace("%", "")) / 100
        row = f"| {level} | {target:.2f} |"
        for var in ["base", "monthly_cal", "combined_cal", "regime_conditional"]:
            cov = all_metrics.get(var, {}).get("coverage", {}).get(level, {}).get("actual_coverage", "N/A")
            if isinstance(cov, float):
                row += f" {cov:.3f} |"
            else:
                row += f" {cov} |"
        lines.append(row)
    lines.append("")

    # Brier decomposition (for combined cal)
    lines.append("## Brier Score Decomposition (Combined Calibration)")
    lines.append("")
    brier = all_metrics.get("combined_cal", {}).get("brier_decomposition", {})
    if brier:
        lines.append("| Threshold | Brier | Reliability | Resolution | Uncertainty | BSS |")
        lines.append("|-----------|-------|-------------|------------|-------------|-----|")
        for thresh_name, bd in brier.items():
            lines.append(
                f"| {bd.get('threshold_F', 'N/A')} F | "
                f"{bd.get('brier_score', 'N/A')} | "
                f"{bd.get('reliability', 'N/A')} | "
                f"{bd.get('resolution', 'N/A')} | "
                f"{bd.get('uncertainty', 'N/A')} | "
                f"{bd.get('brier_skill_score', 'N/A')} |"
            )
    lines.append("")

    # Regime-conditional variance
    lines.append("## Regime-Conditional Variance Models")
    lines.append("")
    lines.append("| Regime | Count | Sigma | Bias | MAE |")
    lines.append("|--------|-------|-------|------|-----|")
    for regime, rm in regime_result.get("regime_sigma_models", {}).items():
        lines.append(f"| {regime} | {rm.get('count', 'N/A')} | {rm.get('sigma', 'N/A')} | {rm.get('mean_bias', 'N/A')} | {rm.get('mae', 'N/A')} |")
    lines.append("")

    # PIT histogram
    lines.append("## PIT Histogram (Combined Calibration)")
    lines.append("")
    pit_hist = all_metrics.get("combined_cal", {}).get("pit_histogram", [])
    if pit_hist:
        lines.append("| Bin | Count | Expected |")
        lines.append("|-----|-------|----------|")
        n_total = sum(pit_hist)
        expected = n_total / 10
        for i, count in enumerate(pit_hist):
            lines.append(f"| {i/10:.1f}-{(i+1)/10:.1f} | {count} | {expected:.0f} |")
    lines.append("")

    # Seasonal CRPS
    lines.append("## Seasonal CRPS (Combined Calibration)")
    lines.append("")
    seasonal = all_metrics.get("combined_cal", {}).get("seasonal_crps", {})
    if seasonal:
        lines.append("| Season | CRPS |")
        lines.append("|--------|------|")
        for s in ["DJF", "MAM", "JJA", "SON"]:
            lines.append(f"| {s} | {seasonal.get(s, 'N/A')} |")
    lines.append("")

    # Write report
    report_path = os.path.join(RESULTS_DIR, "quality_metrics_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved quality metrics report: %s", report_path)


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("RETRAIN EXTENDED VALIDATION: A_NN_64_32 5-seed Ensemble")
    logger.info("=" * 80)
    logger.info("Device: %s", DEVICE)
    logger.info("MOS source: %s", MOS_PATH)
    logger.info("Splits: Train=%s to %s, Val=%s to %s, Test=%s to %s",
                MOS_TRAIN_START, MOS_TRAIN_END, VAL_START, VAL_END,
                TEST_START, TEST_END)
    logger.info("Ensemble seeds: %s", ENSEMBLE_SEEDS)

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
    logger.info("\n=== STEP 4: Building combined feature dataset ===")
    builder = CombinedDatasetBuilder(station_matrix, mos_data, cp_data)
    data = builder.build_correction_data()

    # ---- Step 5: Part 1 - Train 5-seed ensemble ----
    logger.info("\n=== STEP 5: PART 1 - Training 5-seed ensemble ===")
    all_results = {}
    ensemble_result = run_ensemble(data, all_results)

    # ---- Step 6: Part 2 - Sigma recalibration ----
    logger.info("\n=== STEP 6: PART 2 - Sigma Recalibration ===")
    (calibration, sigma_by_month, monthly_cal, regime_cal,
     combined_cal, regime_thresholds) = sigma_recalibration(
        data, ensemble_result["ensemble_preds"], ensemble_result["ensemble_stds"]
    )

    # ---- Step 7: Part 3 - Regime-conditional variance ----
    logger.info("\n=== STEP 7: PART 3 - Regime-Conditional Variance ===")
    regime_result = regime_conditional_variance(
        data, ensemble_result["ensemble_preds"],
        ensemble_result["ensemble_stds"], sigma_by_month
    )

    # ---- Step 8: Part 4 - Quality metrics ----
    logger.info("\n=== STEP 8: PART 4 - Comprehensive Quality Metrics ===")
    all_metrics, sigma_variants = run_quality_metrics(
        data, ensemble_result["ensemble_preds"], ensemble_result["ensemble_stds"],
        calibration, regime_result, sigma_by_month, monthly_cal, regime_cal,
        combined_cal, regime_thresholds,
    )

    # ---- Step 9: Save all outputs ----
    logger.info("\n=== STEP 9: Saving all outputs ===")
    generate_outputs(
        data, ensemble_result, calibration, regime_result,
        all_metrics, sigma_variants, all_results, sigma_by_month
    )

    # ---- Final summary ----
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("Total time: %.1f minutes (%.0f seconds)", elapsed / 60, elapsed)

    # Print key results
    ens_test = all_results.get("E_Ensemble_5seed_test", {})
    ens_val = all_results.get("E_Ensemble_5seed_val", {})
    logger.info("KEY RESULTS:")
    logger.info("  Val  MAE: %.4f F", ens_val.get("mae", float("nan")))
    logger.info("  Test MAE: %.4f F", ens_test.get("mae", float("nan")))

    best_variant = min(all_metrics.keys(), key=lambda k: all_metrics[k].get("crps_mean", 999))
    best_crps = all_metrics[best_variant]["crps_mean"]
    logger.info("  Best CRPS variant: %s (%.4f)", best_variant, best_crps)

    best_coverage = all_metrics.get("combined_cal", {}).get("coverage", {}).get("95%", {})
    logger.info("  95%% PI coverage (combined_cal): %.3f (target: 0.950)",
                best_coverage.get("actual_coverage", float("nan")))

    logger.info("\nOutputs saved to: %s", RESULTS_DIR)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
