#!/usr/bin/env python3
"""
Phase 1 Architecture & Enhanced Temporal Features Experiment.

Tests 8 architecture variants (A-H) with both original and enhanced feature
sets under the MOS-correction (residual learning) framework.

Enhanced features include:
  - Temporal: day_length, solar_elevation_noon, days_since_solstice,
    TMAX_7day_rolling_mean, TMAX_anomaly_from_climo, HDD_7d, CDD_7d
  - Spatial: max_station_24h_change, station_spread, wn_to_coast_gradient,
    ne_sw_gradient, ring_gradient, station_consensus

Architecture experiments (all MOS-correction residual):
  A: Baseline [32,16] dropout=0.2
  B: [64,32] dropout=0.1
  C: [64,32,16] dropout=0.1  (3-layer)
  D: [128,64] dropout=0.15
  E: ResidualCorrectionNN with skip/residual connections
  F: SkipCorrectionNN with concatenated key features at output
  G: HistGradientBoosting on residuals
  H: Ridge on residuals

All data is REAL -- downloaded from NOAA GHCN and loaded from existing MOS CSVs.
"""

import os
import sys
import json
import time
import math
import logging
import warnings
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from config_expanded import (
    SURROUNDING_STATIONS, STATION_METADATA,
    METEOROLOGICAL_SECTORS, STATION_RINGS,
)
from src.data_collection import download_dly_file, parse_dly_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase1_arch_temporal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1_architecture")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

MOS_TRAIN_START, MOS_TRAIN_END = "2004-01-01", "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NYC_LAT = 40.7831  # Central Park latitude


# ============================================================================
# 1. DATA LOADING  (reused from mos_ensemble_pipeline)
# ============================================================================

def download_all_stations():
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ids = [TARGET_STATION] + ALL_SURROUNDING
    for i, sid in enumerate(all_ids, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path):
            logger.info("[%d/%d] %s -- cached", i, len(all_ids), sid)
            continue
        for attempt in range(4):
            try:
                download_dly_file(sid, RAW_DIR)
                logger.info("[%d/%d] %s -- downloaded", i, len(all_ids), sid)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("[%d/%d] %s attempt %d failed: %s. Retry in %ds",
                               i, len(all_ids), sid, attempt + 1, e, wait)
                time.sleep(wait)
        else:
            logger.error("[%d/%d] %s -- FAILED after 4 attempts", i, len(all_ids), sid)


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
    mos = pd.read_csv(MOS_PATH, parse_dates=["date"])
    mos = mos[["date", "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]]
    mos = mos.set_index("date").sort_index()
    logger.info("MOS data: %d rows, %s to %s",
                len(mos), mos.index.min().date(), mos.index.max().date())
    return mos


def load_central_park_tmax():
    cp = pd.read_csv(CP_PATH, parse_dates=["date"])
    cp = cp.set_index("date").sort_index()
    cp.columns = ["nyc_tmax"]
    logger.info("Central Park TMAX: %d rows, %s to %s",
                len(cp), cp.index.min().date(), cp.index.max().date())
    return cp


# ============================================================================
# 2. ENHANCED FEATURE ENGINEERING
# ============================================================================

def solar_declination(doy):
    """Solar declination angle in radians from day of year."""
    return np.radians(23.44) * np.sin(np.radians((360 / 365.25) * (doy - 81)))


def day_length_hours(lat_deg, doy):
    """Approximate hours of daylight using the sunrise equation."""
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    # cos(hour_angle) = -tan(lat)*tan(decl)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    # Clamp to [-1, 1] for polar day/night
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha = np.arccos(cos_ha)
    return (2.0 * ha / np.pi) * 12.0  # hours


def solar_elevation_noon(lat_deg, doy):
    """Maximum solar elevation angle at solar noon (degrees)."""
    lat_rad = np.radians(lat_deg)
    decl = solar_declination(doy)
    elev = np.arcsin(np.sin(lat_rad) * np.sin(decl) +
                     np.cos(lat_rad) * np.cos(decl))
    return np.degrees(elev)


def days_since_winter_solstice(doy):
    """Minimum circular distance from winter solstice (day ~355)."""
    winter_sol = 355
    d = np.abs(doy - winter_sol)
    return np.minimum(d, 365 - d)


def days_since_summer_solstice(doy):
    """Minimum circular distance from summer solstice (day ~172)."""
    summer_sol = 172
    d = np.abs(doy - summer_sol)
    return np.minimum(d, 365 - d)


def add_enhanced_temporal_features(df, cp_data, train_end=MOS_TRAIN_END):
    """Add all enhanced temporal features to the dataframe.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    cp_data : DataFrame with 'nyc_tmax' column
    train_end : str, end date for climatology computation (no leakage)

    Returns
    -------
    df : DataFrame with new temporal columns added
    new_cols : list of added column names
    """
    df = df.copy()
    doy = df.index.dayofyear

    # --- Astronomical / calendar features ---
    df["day_length"] = day_length_hours(NYC_LAT, doy)
    df["solar_elev_noon"] = solar_elevation_noon(NYC_LAT, doy)
    df["days_since_winter_sol"] = days_since_winter_solstice(doy)
    df["days_since_summer_sol"] = days_since_summer_solstice(doy)

    # --- NYC TMAX-derived temporal features ---
    # We need nyc_tmax aligned; use lag-1 to avoid leakage
    nyc_full = cp_data["nyc_tmax"].reindex(df.index)
    nyc_lag1 = nyc_full.shift(1)

    # Rolling 7-day mean of NYC TMAX (lag-1, so trailing 7 days ending yesterday)
    df["tmax_7d_rolling_mean"] = nyc_lag1.rolling(7, min_periods=4).mean()

    # Climatological mean for each DOY -- computed from training data only
    train_mask = cp_data.index <= train_end
    train_cp = cp_data.loc[train_mask, "nyc_tmax"]
    climo = train_cp.groupby(train_cp.index.dayofyear).mean()
    # Map DOY to climo (handle DOY 366 edge case)
    doy_climo = doy.map(lambda d: climo.get(d, climo.get(min(d, 365), np.nan)))
    df["tmax_anomaly_from_climo"] = nyc_lag1 - doy_climo.values

    # Heating degree days (trailing 7 days, lag-1)
    hdd_daily = np.maximum(0, 65 - nyc_lag1)
    df["hdd_7d"] = hdd_daily.rolling(7, min_periods=4).sum()

    # Cooling degree days (trailing 7 days, lag-1)
    cdd_daily = np.maximum(0, nyc_lag1 - 65)
    df["cdd_7d"] = cdd_daily.rolling(7, min_periods=4).sum()

    new_cols = [
        "day_length", "solar_elev_noon",
        "days_since_winter_sol", "days_since_summer_sol",
        "tmax_7d_rolling_mean", "tmax_anomaly_from_climo",
        "hdd_7d", "cdd_7d",
    ]
    return df, new_cols


def add_spatial_features(df, station_matrix):
    """Add station-derived spatial features.

    Uses lag-1 station data to compute spatial gradient and variability proxies.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    station_matrix : DataFrame of raw station TMAX (unlagged)

    Returns
    -------
    df : DataFrame with new spatial columns
    new_cols : list of added column names
    """
    df = df.copy()

    # Identify TMAX columns
    tmax_cols = [c for c in station_matrix.columns if "TMAX" in c]
    if not tmax_cols:
        return df, []

    tmax_matrix = station_matrix[tmax_cols]

    # Lag-1 values (yesterday's observations)
    tmax_lag1 = tmax_matrix.shift(1)
    tmax_lag2 = tmax_matrix.shift(2)

    # Align to df's index
    tmax_l1 = tmax_lag1.reindex(df.index)
    tmax_l2 = tmax_lag2.reindex(df.index)

    # 1. max_station_24h_change: max absolute change across all stations
    change_24h = (tmax_l1 - tmax_l2).abs()
    df["max_station_24h_change"] = change_24h.max(axis=1)

    # 2. station_spread: range of station temps at t-1
    df["station_spread"] = tmax_l1.max(axis=1) - tmax_l1.min(axis=1)

    # 3. station_consensus: std of station TMAX at t-1 (low = agreement)
    df["station_consensus"] = tmax_l1.std(axis=1)

    # --- Sector-based gradients ---
    # Helper: get lag-1 TMAX cols for stations in a sector
    def sector_mean(sector_stations, tmax_l1_df):
        cols = [f"{sid}_TMAX" for sid in sector_stations
                if f"{sid}_TMAX" in tmax_l1_df.columns]
        if cols:
            return tmax_l1_df[cols].mean(axis=1)
        return pd.Series(np.nan, index=tmax_l1_df.index)

    # WNW sector vs Coastal sector gradient
    wnw_stations = METEOROLOGICAL_SECTORS.get("WNW", [])
    coastal_stations = METEOROLOGICAL_SECTORS.get("Coastal", [])
    wnw_mean = sector_mean(wnw_stations, tmax_l1)
    coastal_mean = sector_mean(coastal_stations, tmax_l1)
    df["wn_to_coast_gradient"] = wnw_mean - coastal_mean

    # NE sector vs SW sector gradient
    ne_stations = METEOROLOGICAL_SECTORS.get("NE", [])
    sw_stations = METEOROLOGICAL_SECTORS.get("SW", [])
    ne_mean = sector_mean(ne_stations, tmax_l1)
    sw_mean = sector_mean(sw_stations, tmax_l1)
    df["ne_sw_gradient"] = ne_mean - sw_mean

    # Ring gradient: Ring1 mean minus Ring3 mean
    ring1_stations = STATION_RINGS.get("Ring1_Near", [])
    ring3_stations = STATION_RINGS.get("Ring3_Extended", [])
    ring1_mean = sector_mean(ring1_stations, tmax_l1)
    ring3_mean = sector_mean(ring3_stations, tmax_l1)
    df["ring_gradient"] = ring1_mean - ring3_mean

    new_cols = [
        "max_station_24h_change", "station_spread", "station_consensus",
        "wn_to_coast_gradient", "ne_sw_gradient", "ring_gradient",
    ]
    return df, new_cols


# ============================================================================
# 3. STANDARD HELPERS
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


def impute_and_scale(X_train, X_val, X_test, X_oos):
    """Impute NaNs with training mean, then StandardScaler."""
    train_means = np.nanmean(X_train, axis=0)
    train_means = np.where(np.isnan(train_means), 0.0, train_means)
    for arr in [X_train, X_val, X_test, X_oos]:
        for j in range(arr.shape[1]):
            mask = np.isnan(arr[:, j])
            arr[mask, j] = train_means[j]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_oos = scaler.transform(X_oos)
    return X_train, X_val, X_test, X_oos, scaler


# ============================================================================
# 4. MODEL DEFINITIONS
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


class ResidualCorrectionNN(nn.Module):
    """NN with residual (skip) connection from input to first hidden layer.

    Architecture:
      Layer 1: Linear(n_feat, 64) -> ReLU -> Dropout
      Residual projection: Linear(n_feat, 64)
      Layer 2: (layer1_out + residual_proj) -> Linear(64, 32) -> ReLU -> Dropout
      Output: Linear(32, 1)
    """
    def __init__(self, n_features, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Linear(n_features, 64)
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        h1 = self.layer1(x)
        res = self.residual_proj(x)
        h2 = self.layer2(h1 + res)
        return self.output(h2)


class SkipCorrectionNN(nn.Module):
    """NN that concatenates key input features at the output layer.

    Main path: Linear(n_feat, 64) -> ReLU -> Linear(64, 32) -> ReLU
    Skip: selects top-K important features (by index)
    Output: Linear(32 + K, 1) from concat(main_output, skip_features)
    """
    def __init__(self, n_features, skip_indices, dropout=0.1):
        super().__init__()
        self.skip_indices = skip_indices
        k = len(skip_indices)
        self.main = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(32 + k, 1)

    def forward(self, x):
        main_out = self.main(x)
        skip = x[:, self.skip_indices]
        combined = torch.cat([main_out, skip], dim=1)
        return self.output(combined)


# ============================================================================
# 5. TRAINING UTILITIES
# ============================================================================

def train_nn(model, X_train, y_train, X_val, y_val,
             lr=1e-3, epochs=300, patience=20, batch_size=128,
             loss_fn_name="huber"):
    model = model.to(DEVICE)
    if loss_fn_name == "huber":
        criterion = nn.HuberLoss(delta=2.0)
    elif loss_fn_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
# 6. DATASET BUILDER WITH ENHANCED FEATURES
# ============================================================================

class EnhancedDatasetBuilder:
    """Builds MOS-correction datasets with original and enhanced features."""

    def __init__(self, station_matrix, mos_data, cp_data):
        self.station_matrix = station_matrix
        self.mos = mos_data
        self.cp = cp_data

    def build_correction_dataset(self, use_enhanced=False):
        """Build MOS-correction dataset.

        Returns dict with train/val/test/oos splits containing:
          X, y_resid, y_actual, mos_base, dates
        Also returns feature_names and skip_feature_indices.
        """
        lagged = create_lagged_features(self.station_matrix, lag=1)
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        df = pd.concat([lagged, self.mos, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df = add_date_features(df)

        # Residual target
        df["residual"] = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
        df = df.dropna(subset=["nyc_tmax_lag1"])

        # Station mean
        station_lag_tmax = [c for c in lagged.columns
                           if c in df.columns and "TMAX" in c]
        if station_lag_tmax:
            df["station_mean_lag1"] = df[station_lag_tmax].mean(axis=1)
            df["mos_station_diff"] = (df["station_mean_lag1"] - df["mos_ensemble_tmax_f"]).abs()
        else:
            df["station_mean_lag1"] = 0.0
            df["mos_station_diff"] = 0.0

        # Base feature columns (original)
        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        base_feature_cols = station_lag_cols + [
            "mos_ensemble_tmax_f", "nyc_tmax_lag1",
            "station_mean_lag1", "mos_station_diff",
            "sin_day", "cos_day",
        ]

        enhanced_cols = []
        if use_enhanced:
            # Add enhanced temporal features
            df, temporal_cols = add_enhanced_temporal_features(
                df, self.cp, train_end=MOS_TRAIN_END
            )
            enhanced_cols.extend(temporal_cols)

            # Add spatial features
            df, spatial_cols = add_spatial_features(df, self.station_matrix)
            enhanced_cols.extend(spatial_cols)

        feature_cols = base_feature_cols + enhanced_cols
        valid_cols = [c for c in feature_cols if c in df.columns]

        # Identify skip-feature indices for SkipCorrectionNN
        skip_names = ["mos_ensemble_tmax_f", "nyc_tmax_lag1",
                      "sin_day", "cos_day", "station_mean_lag1"]
        skip_indices = [valid_cols.index(n) for n in skip_names if n in valid_cols]

        # Split
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

        X_tr, X_v, X_te, X_oos, scaler = impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["skip_indices"] = skip_indices
        arrays["scaler"] = scaler

        return arrays


# ============================================================================
# 7. EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(config_name, model_or_class, data, is_nn=True,
                          lr=1e-3, epochs=300, patience=20, batch_size=128,
                          loss_fn_name="huber"):
    """Run a single architecture experiment on the correction dataset.

    Returns dict of results keyed by split.
    """
    results = {}

    if is_nn:
        # Instantiate and train NN
        if isinstance(model_or_class, nn.Module):
            model = model_or_class
        else:
            model = model_or_class

        val_mae = train_nn(
            model, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=lr, epochs=epochs, patience=patience,
            batch_size=batch_size, loss_fn_name=loss_fn_name,
        )
        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            r = evaluate_model(
                data[split]["y_actual"], pred_actual,
                data[split]["dates"], f"{config_name} {split}",
            )
            results[split] = r
    else:
        # Sklearn model
        model = model_or_class
        model.fit(data["train"]["X"], data["train"]["y_resid"])
        for split in ["train", "val", "test", "oos"]:
            pred_resid = model.predict(data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            r = evaluate_model(
                data[split]["y_actual"], pred_actual,
                data[split]["dates"], f"{config_name} {split}",
            )
            results[split] = r

        # Feature importance for tree models
        if hasattr(model, "feature_importances_"):
            feat_names = data.get("feature_names", [])
            if feat_names:
                imp = model.feature_importances_
                fi = sorted(zip(feat_names, imp), key=lambda x: -x[1])
                results["feature_importance"] = [
                    {"feature": f, "importance": round(float(v), 6)}
                    for f, v in fi[:30]
                ]

    return results


def build_experiment_configs(n_features, skip_indices):
    """Return ordered dict of experiment name -> (model, is_nn, train_kwargs)."""
    configs = OrderedDict()

    # A: Baseline [32,16] dropout=0.2
    configs["A_baseline_32_16"] = {
        "model": FlexibleNN(n_features, [32, 16], dropout=0.2),
        "is_nn": True,
        "kwargs": {"lr": 1e-3, "epochs": 200, "patience": 15,
                   "batch_size": 128, "loss_fn_name": "mae"},
    }

    # B: [64,32] dropout=0.1
    configs["B_64_32"] = {
        "model": FlexibleNN(n_features, [64, 32], dropout=0.1),
        "is_nn": True,
        "kwargs": {"lr": 1e-3, "epochs": 300, "patience": 20,
                   "batch_size": 128, "loss_fn_name": "huber"},
    }

    # C: [64,32,16] dropout=0.1  (3-layer)
    configs["C_64_32_16"] = {
        "model": FlexibleNN(n_features, [64, 32, 16], dropout=0.1),
        "is_nn": True,
        "kwargs": {"lr": 1e-3, "epochs": 300, "patience": 20,
                   "batch_size": 128, "loss_fn_name": "huber"},
    }

    # D: [128,64] dropout=0.15
    configs["D_128_64"] = {
        "model": FlexibleNN(n_features, [128, 64], dropout=0.15),
        "is_nn": True,
        "kwargs": {"lr": 5e-4, "epochs": 300, "patience": 20,
                   "batch_size": 128, "loss_fn_name": "huber"},
    }

    # E: ResidualCorrectionNN
    configs["E_residual"] = {
        "model": ResidualCorrectionNN(n_features, dropout=0.1),
        "is_nn": True,
        "kwargs": {"lr": 1e-3, "epochs": 300, "patience": 20,
                   "batch_size": 128, "loss_fn_name": "huber"},
    }

    # F: SkipCorrectionNN
    configs["F_skip"] = {
        "model": SkipCorrectionNN(n_features, skip_indices, dropout=0.1),
        "is_nn": True,
        "kwargs": {"lr": 1e-3, "epochs": 300, "patience": 20,
                   "batch_size": 128, "loss_fn_name": "huber"},
    }

    # G: HistGradientBoosting
    configs["G_hgb"] = {
        "model": HistGradientBoostingRegressor(
            max_iter=500, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=SEED,
        ),
        "is_nn": False,
        "kwargs": {},
    }

    # H: Ridge
    configs["H_ridge"] = {
        "model": Ridge(alpha=1.0),
        "is_nn": False,
        "kwargs": {},
    }

    return configs


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Step 1: Download data ----
    logger.info("=" * 80)
    logger.info("STEP 1: Downloading station .dly files")
    logger.info("=" * 80)
    download_all_stations()

    # ---- Step 2: Build station matrix ----
    logger.info("=" * 80)
    logger.info("STEP 2: Building station observation matrix")
    logger.info("=" * 80)
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # ---- Step 3: Load MOS + CP data ----
    logger.info("=" * 80)
    logger.info("STEP 3: Loading MOS and Central Park data")
    logger.info("=" * 80)
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    builder = EnhancedDatasetBuilder(station_matrix, mos_data, cp_data)

    # ---- Step 4: Build datasets ----
    logger.info("=" * 80)
    logger.info("STEP 4: Building datasets (original + enhanced)")
    logger.info("=" * 80)

    data_orig = builder.build_correction_dataset(use_enhanced=False)
    data_enh = builder.build_correction_dataset(use_enhanced=True)

    n_feat_orig = data_orig["train"]["X"].shape[1]
    n_feat_enh = data_enh["train"]["X"].shape[1]
    skip_orig = data_orig["skip_indices"]
    skip_enh = data_enh["skip_indices"]

    logger.info("Original features: %d", n_feat_orig)
    logger.info("Enhanced features: %d  (+%d new)", n_feat_enh, n_feat_enh - n_feat_orig)
    logger.info("Feature names (enhanced): %s", data_enh["feature_names"][-20:])
    logger.info("Train: %d  Val: %d  Test: %d  OOS: %d",
                len(data_orig["train"]["y_resid"]),
                len(data_orig["val"]["y_resid"]),
                len(data_orig["test"]["y_resid"]),
                len(data_orig["oos"]["y_resid"]))

    # ---- Step 5: Run experiments ----
    logger.info("=" * 80)
    logger.info("STEP 5: Running architecture experiments")
    logger.info("=" * 80)

    all_results = {}
    summary_rows = []

    for feature_set_name, dataset, n_feat, skip_idx in [
        ("orig", data_orig, n_feat_orig, skip_orig),
        ("enh", data_enh, n_feat_enh, skip_enh),
    ]:
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE SET: %s (%d features)", feature_set_name, n_feat)
        logger.info("=" * 70)

        configs = build_experiment_configs(n_feat, skip_idx)

        for config_name, cfg in configs.items():
            full_name = f"{config_name}_{feature_set_name}"
            logger.info("\n--- Running %s ---", full_name)

            try:
                results = run_single_experiment(
                    full_name,
                    cfg["model"],
                    dataset,
                    is_nn=cfg["is_nn"],
                    **cfg["kwargs"],
                )
                all_results[full_name] = results

                # Collect summary
                for split in ["train", "val", "test", "oos"]:
                    if split in results:
                        row = {
                            "experiment": full_name,
                            "architecture": config_name,
                            "features": feature_set_name,
                            "split": split,
                            **results[split],
                        }
                        summary_rows.append(row)

            except Exception as e:
                logger.error("FAILED %s: %s", full_name, e, exc_info=True)

            # Re-instantiate model for next feature set (weights are consumed)
            # Models are already fresh per feature-set loop since
            # build_experiment_configs creates new instances

        # Need fresh models for the next feature_set since build_experiment_configs
        # was already called at the top of this loop

    # ---- Step 6: Save results ----
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving results")
    logger.info("=" * 80)

    # Clean results for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    # Save full results JSON
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    logger.info("Saved detailed results to %s", json_path)

    # Build and save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(RESULTS_DIR, "summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info("Saved summary CSV to %s", csv_path)

    # ---- Step 7: Print comprehensive comparison ----
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE RESULTS COMPARISON")
    logger.info("=" * 80)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Test set comparison
        logger.info("\n--- TEST SET (sorted by MAE) ---")
        test_df = summary_df[summary_df["split"] == "test"].sort_values("mae")
        header = f"{'Experiment':<40s} {'MAE':>6s} {'RMSE':>7s} {'R2':>6s}  {'DJF':>5s} {'MAM':>5s} {'JJA':>5s} {'SON':>5s}"
        logger.info(header)
        logger.info("-" * len(header))
        for _, row in test_df.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                val = row.get(s, np.nan)
                seasonal += f" {val:5.2f}" if pd.notna(val) else "   N/A"
            logger.info("%-40s %6.3f %7.3f %6.3f %s",
                        row["experiment"], row["mae"], row["rmse"], row["r2"], seasonal)

        # OOS comparison
        logger.info("\n--- OOS SET (sorted by MAE) ---")
        oos_df = summary_df[summary_df["split"] == "oos"].sort_values("mae")
        logger.info(header)
        logger.info("-" * len(header))
        for _, row in oos_df.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                val = row.get(s, np.nan)
                seasonal += f" {val:5.2f}" if pd.notna(val) else "   N/A"
            logger.info("%-40s %6.3f %7.3f %6.3f %s",
                        row["experiment"], row["mae"], row["rmse"], row["r2"], seasonal)

        # Comparison vs baseline
        logger.info("\n--- vs Baseline (C_Correction_NN_tiny: 2.090 test / 2.093 OOS) ---")
        baseline_test = 2.090
        baseline_oos = 2.093
        for _, row in test_df.iterrows():
            oos_row = oos_df[oos_df["experiment"] == row["experiment"]]
            oos_mae = oos_row["mae"].values[0] if len(oos_row) > 0 else np.nan
            test_delta = row["mae"] - baseline_test
            oos_delta = oos_mae - baseline_oos if not np.isnan(oos_mae) else np.nan
            test_sign = "+" if test_delta >= 0 else ""
            oos_sign = "+" if not np.isnan(oos_delta) and oos_delta >= 0 else ""
            oos_str = f"{oos_sign}{oos_delta:.3f}" if not np.isnan(oos_delta) else "N/A"
            logger.info("  %-40s test: %s%.3f  oos: %s",
                        row["experiment"], test_sign, test_delta, oos_str)

        # Feature importance for HGB models
        for name, res in all_results.items():
            if "hgb" in name.lower() and "feature_importance" in res:
                logger.info("\n--- Feature Importance: %s ---", name)
                for item in res["feature_importance"][:15]:
                    logger.info("  %-50s %.4f", item["feature"], item["importance"])

        # Best results summary
        logger.info("\n" + "=" * 80)
        logger.info("TOP 5 MODELS BY OOS MAE")
        logger.info("=" * 80)
        top5_oos = oos_df.head(5)
        for _, row in top5_oos.iterrows():
            test_row = test_df[test_df["experiment"] == row["experiment"]]
            test_mae = test_row["mae"].values[0] if len(test_row) > 0 else np.nan
            logger.info("  %-40s  Test=%.3f  OOS=%.3f  (gap=%.3f)",
                        row["experiment"], test_mae, row["mae"],
                        abs(row["mae"] - test_mae))

    elapsed = time.time() - start_time
    logger.info("\nTotal pipeline time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
