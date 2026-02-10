#!/usr/bin/env python3
"""
Phase 1 Feature Engineering for NYC Temperature Prediction.

Implements three feature groups and evaluates their impact on MOS correction models:
  1A) MOS Error Memory Features
  1B) Semi-Annual Harmonics + Gradient Features
  1C) MOS x Station Interaction Features

All data is REAL -- downloaded from NOAA GHCN and loaded from existing MOS CSVs.
Baseline: C_Correction_NN_tiny (2.090 test / 2.093 OOS MAE).
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime

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
    METEOROLOGICAL_SECTORS, STATION_SECTORS,
)
from src.data_collection import download_dly_file, parse_dly_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase1_features")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1_features")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# Chronological splits
MOS_TRAIN_START, MOS_TRAIN_END = "2004-01-01", "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 1. DATA LOADING (reused from mos_ensemble_pipeline)
# ============================================================================

def download_all_stations():
    """Download .dly files for target + all surrounding stations."""
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ids = [TARGET_STATION] + ALL_SURROUNDING
    total = len(all_ids)

    for i, sid in enumerate(all_ids, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path):
            logger.info("[%d/%d] %s -- cached", i, total, sid)
            continue

        for attempt in range(4):
            try:
                download_dly_file(sid, RAW_DIR)
                logger.info("[%d/%d] %s -- downloaded", i, total, sid)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("[%d/%d] %s attempt %d failed: %s. Retrying in %ds...",
                               i, total, sid, attempt + 1, e, wait)
                time.sleep(wait)
        else:
            logger.error("[%d/%d] %s -- FAILED after 4 attempts", i, total, sid)


def parse_station_tmax(station_id, start_date, end_date):
    """Parse a station .dly file and return a Series of daily TMAX (F)."""
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
    """Parse a station .dly file and return a Series of daily TMIN (F)."""
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
    """Build a DataFrame of daily TMAX (and optionally TMIN) for all surrounding stations."""
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
    """Load MOS forecast data."""
    mos = pd.read_csv(MOS_PATH, parse_dates=["date"])
    mos = mos[["date", "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]]
    mos = mos.set_index("date").sort_index()
    logger.info("MOS data: %d rows, date range %s to %s",
                len(mos), mos.index.min().date(), mos.index.max().date())
    return mos


def load_central_park_tmax():
    """Load Central Park actual TMAX."""
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
    """Add sin/cos day-of-year encoding (annual harmonics)."""
    doy = df.index.dayofyear
    df = df.copy()
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def create_lagged_features(station_matrix, lag=1):
    """Shift station observations by lag days."""
    lagged = station_matrix.shift(lag)
    lagged.columns = [f"{c}_lag{lag}" for c in lagged.columns]
    return lagged


def assign_season(dates):
    """Map dates to meteorological season labels."""
    month = dates.month
    seasons = pd.Series("", index=dates)
    seasons[month.isin([12, 1, 2])] = "DJF"
    seasons[month.isin([3, 4, 5])] = "MAM"
    seasons[month.isin([6, 7, 8])] = "JJA"
    seasons[month.isin([9, 10, 11])] = "SON"
    return seasons


# ============================================================================
# 3. PHASE 1A: MOS ERROR MEMORY FEATURES
# ============================================================================

def add_mos_error_memory_features(df):
    """
    Add MOS error memory features. All based on lag-1+ data to avoid leakage.

    Features:
      - mos_error_yesterday: actual(t-1) - MOS_ensemble(t-1)
      - mos_error_7d: rolling mean of (actual - MOS) over 7 days, lag-1
      - mos_error_14d: rolling mean over 14 days, lag-1
      - mos_error_30d: rolling mean over 30 days, lag-1
      - mos_abs_error_7d: rolling mean of |actual - MOS| over 7 days, lag-1
      - gfs_nam_spread: |GFS_MOS - NAM_MOS| (same day, no leakage since MOS is a forecast)
      - gfs_nam_sign: sign(GFS_MOS - NAM_MOS) (same day)
    """
    df = df.copy()

    # Compute MOS error series: actual - MOS_ensemble
    mos_error = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]

    # Lag-1 error (yesterday's MOS error)
    df["mos_error_yesterday"] = mos_error.shift(1)

    # Rolling mean of MOS error, shifted by 1 to avoid leakage
    # shift(1) before rolling so window ends at t-1
    mos_error_shifted = mos_error.shift(1)
    df["mos_error_7d"] = mos_error_shifted.rolling(window=7, min_periods=3).mean()
    df["mos_error_14d"] = mos_error_shifted.rolling(window=14, min_periods=5).mean()
    df["mos_error_30d"] = mos_error_shifted.rolling(window=30, min_periods=10).mean()

    # Rolling mean of absolute MOS error
    abs_error_shifted = mos_error.abs().shift(1)
    df["mos_abs_error_7d"] = abs_error_shifted.rolling(window=7, min_periods=3).mean()

    # GFS-NAM spread and sign (these are forecasts for day t, no leakage)
    gfs_nam_diff = df["gfs_mos_tmax_f"] - df["nam_mos_tmax_f"]
    df["gfs_nam_spread"] = gfs_nam_diff.abs()
    df["gfs_nam_sign"] = np.sign(gfs_nam_diff)

    return df


PHASE_1A_FEATURES = [
    "mos_error_yesterday", "mos_error_7d", "mos_error_14d", "mos_error_30d",
    "mos_abs_error_7d", "gfs_nam_spread", "gfs_nam_sign",
]


# ============================================================================
# 4. PHASE 1B: SEMI-ANNUAL HARMONICS + GRADIENT FEATURES
# ============================================================================

def add_semi_annual_harmonics(df):
    """Add semi-annual (2nd harmonic) sin/cos features."""
    df = df.copy()
    doy = df.index.dayofyear
    df["sin_day_semi"] = np.sin(4 * np.pi * doy / 365.25)
    df["cos_day_semi"] = np.cos(4 * np.pi * doy / 365.25)
    return df


def compute_gradient_features(df, station_lag_tmax_cols):
    """
    Compute temperature gradient features across station pairs.

    For each day, find the station pair with the maximum |T_i - T_j| / distance
    and record the magnitude and bearing of that gradient.
    """
    df = df.copy()

    # Build station metadata lookup (only for stations with lag-1 TMAX cols)
    station_pairs = []
    sid_to_col = {}
    for col in station_lag_tmax_cols:
        # Column name like: USW00014732_TMAX_lag1
        parts = col.split("_TMAX_lag")
        if len(parts) == 2:
            sid = parts[0]
            if sid in STATION_METADATA:
                sid_to_col[sid] = col

    # Build pairs with distance
    sids = list(sid_to_col.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            s1, s2 = sids[i], sids[j]
            m1, m2 = STATION_METADATA[s1], STATION_METADATA[s2]
            # Approximate distance between stations using their distances and bearings from CP
            # Use law of cosines: d12^2 = d1^2 + d2^2 - 2*d1*d2*cos(bearing_diff)
            d1, d2 = m1["distance_mi"], m2["distance_mi"]
            b1, b2 = np.radians(m1["bearing"]), np.radians(m2["bearing"])
            d12 = np.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * np.cos(b1 - b2))
            d12 = max(d12, 1.0)  # floor at 1 mile
            avg_bearing = np.degrees(np.arctan2(
                np.sin(b1) * d1 + np.sin(b2) * d2,
                np.cos(b1) * d1 + np.cos(b2) * d2
            )) % 360
            station_pairs.append((sid_to_col[s1], sid_to_col[s2], d12, avg_bearing))

    if not station_pairs:
        df["max_gradient"] = 0.0
        df["gradient_bearing"] = 0.0
        return df

    # For efficiency, sample a subset of pairs (top 100 by distance diversity)
    # Focus on pairs with meaningful distance (> 20 mi)
    station_pairs = [(c1, c2, d, b) for c1, c2, d, b in station_pairs if d > 20]
    if len(station_pairs) > 200:
        # Sample diverse pairs: sort by distance, take every nth
        station_pairs.sort(key=lambda x: x[2])
        step = max(1, len(station_pairs) // 200)
        station_pairs = station_pairs[::step]

    logger.info("Computing gradient features across %d station pairs...", len(station_pairs))

    max_gradient = np.zeros(len(df))
    gradient_bearing = np.zeros(len(df))

    # Vectorized computation
    for c1, c2, dist, bearing in station_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        temp_diff = (df[c1].values - df[c2].values)
        grad = np.abs(temp_diff) / dist
        # Use nan-aware comparison
        better = np.where(np.isnan(grad), False, grad > max_gradient)
        max_gradient = np.where(better, grad, max_gradient)
        gradient_bearing = np.where(better, bearing, gradient_bearing)

    df["max_gradient"] = max_gradient
    df["gradient_bearing"] = gradient_bearing

    return df


PHASE_1B_FEATURES = [
    "sin_day_semi", "cos_day_semi", "max_gradient", "gradient_bearing",
]


# ============================================================================
# 5. PHASE 1C: MOS x STATION INTERACTION FEATURES
# ============================================================================

def add_mos_station_interaction_features(df, station_lag_tmax_cols):
    """
    Add MOS x Station interaction features.

    Features:
      - mos_station_gap: MOS_ensemble - NYC_TMAX(t-1)
      - mos_sector_gap: MOS_ensemble - mean_WNW_TMAX(t-1)
      - station_mos_agree: sign(delta_T_stations) == sign(mos_station_gap)
    """
    df = df.copy()

    # MOS vs NYC lag-1 gap
    df["mos_station_gap"] = df["mos_ensemble_tmax_f"] - df["nyc_tmax_lag1"]

    # WNW sector mean (upstream cold-air advection stations)
    wnw_stations = METEOROLOGICAL_SECTORS.get("WNW", [])
    wnw_cols = [f"{sid}_TMAX_lag1" for sid in wnw_stations
                if f"{sid}_TMAX_lag1" in df.columns]
    if wnw_cols:
        wnw_mean = df[wnw_cols].mean(axis=1)
        df["mos_sector_gap"] = df["mos_ensemble_tmax_f"] - wnw_mean
    else:
        df["mos_sector_gap"] = 0.0

    # Station trend agreement with MOS
    # delta_T_stations: mean station TMAX(t-1) - mean station TMAX(t-2)
    # We need t-2 station data; approximate with station_mean_lag1 change
    if station_lag_tmax_cols:
        station_mean = df[station_lag_tmax_cols].mean(axis=1)
        station_delta = station_mean.diff()  # change from t-2 to t-1 station mean
        mos_gap_sign = np.sign(df["mos_station_gap"])
        station_delta_sign = np.sign(station_delta)
        df["station_mos_agree"] = (mos_gap_sign == station_delta_sign).astype(float)
    else:
        df["station_mos_agree"] = 0.0

    return df


PHASE_1C_FEATURES = [
    "mos_station_gap", "mos_sector_gap", "station_mos_agree",
]


# ============================================================================
# 6. MODEL DEFINITIONS
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


def train_nn(model, X_train, y_train, X_val, y_val,
             lr=1e-3, epochs=300, patience=20, batch_size=128,
             loss_fn_name="huber"):
    """Train a NN with early stopping. Returns best val MAE."""
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
    """Get predictions from a trained NN."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        preds = model(X_t).squeeze(-1).cpu().numpy()
    return preds


def evaluate_model(y_true, y_pred, dates=None, label=""):
    """Compute MAE, RMSE, R2 and seasonal breakdown."""
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
        logger.info("  %s: MAE=%.3f, RMSE=%.3f, R2=%.3f", label, mae, rmse, r2)
    return result


# ============================================================================
# 7. ENHANCED DATASET BUILDER
# ============================================================================

class EnhancedDatasetBuilder:
    """
    Builds MOS correction datasets with configurable feature groups.

    Feature groups:
      - 'base': Station lag-1, MOS, NYC lag-1, sin/cos date (same as original C model)
      - '1A': MOS error memory features
      - '1B': Semi-annual harmonics + gradient features
      - '1C': MOS x Station interaction features
    """

    def __init__(self, station_matrix, mos_data, cp_data):
        self.station_matrix = station_matrix
        self.mos = mos_data
        self.cp = cp_data
        self._full_df = None
        self._base_feature_cols = None
        self._all_feature_cols = None
        self._station_lag_tmax_cols = None

    def _build_full_dataframe(self):
        """Build the full merged dataframe with ALL features (computed once)."""
        if self._full_df is not None:
            return

        logger.info("Building full enhanced feature dataframe...")

        # Station lag-1
        lagged = create_lagged_features(self.station_matrix, lag=1)
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        # Merge
        df = pd.concat([lagged, self.mos, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])

        # Base date features
        df = add_date_features(df)

        # Residual target
        df["residual"] = df["nyc_tmax"] - df["mos_ensemble_tmax_f"]
        df = df.dropna(subset=["nyc_tmax_lag1"])

        # Station mean and MOS-station diff (base features)
        station_lag_tmax = [c for c in lagged.columns
                           if c in df.columns and "TMAX" in c]
        self._station_lag_tmax_cols = station_lag_tmax
        if station_lag_tmax:
            df["station_mean_lag1"] = df[station_lag_tmax].mean(axis=1)
            df["mos_station_diff"] = (df["station_mean_lag1"] - df["mos_ensemble_tmax_f"]).abs()
        else:
            df["station_mean_lag1"] = 0.0
            df["mos_station_diff"] = 0.0

        # ---- Phase 1A: MOS Error Memory ----
        df = add_mos_error_memory_features(df)

        # ---- Phase 1B: Semi-Annual Harmonics + Gradient ----
        df = add_semi_annual_harmonics(df)
        df = compute_gradient_features(df, station_lag_tmax)

        # ---- Phase 1C: MOS x Station Interaction ----
        df = add_mos_station_interaction_features(df, station_lag_tmax)

        # Define feature groups
        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        self._base_feature_cols = station_lag_cols + [
            "mos_ensemble_tmax_f", "nyc_tmax_lag1",
            "station_mean_lag1", "mos_station_diff",
            "sin_day", "cos_day",
        ]
        self._all_feature_cols = (
            self._base_feature_cols
            + PHASE_1A_FEATURES
            + PHASE_1B_FEATURES
            + PHASE_1C_FEATURES
        )

        self._full_df = df
        logger.info("Full dataframe: %d rows, %d potential features",
                     len(df), len(self._all_feature_cols))

    def get_feature_cols(self, groups):
        """
        Get feature column names for specified groups.

        Parameters
        ----------
        groups : list of str
            Any combination of 'base', '1A', '1B', '1C'.
        """
        self._build_full_dataframe()
        cols = list(self._base_feature_cols) if "base" in groups else []
        if "1A" in groups:
            cols += PHASE_1A_FEATURES
        if "1B" in groups:
            cols += PHASE_1B_FEATURES
        if "1C" in groups:
            cols += PHASE_1C_FEATURES
        # Deduplicate while preserving order
        seen = set()
        unique_cols = []
        for c in cols:
            if c not in seen and c in self._full_df.columns:
                seen.add(c)
                unique_cols.append(c)
        return unique_cols

    def build_correction_data(self, feature_groups):
        """
        Build MOS correction dataset with specified feature groups.

        Parameters
        ----------
        feature_groups : list of str
            Feature groups to include, e.g., ['base', '1A', '1B', '1C'].

        Returns
        -------
        dict
            Split data arrays with X, y_resid, y_actual, mos_base, dates, feature_names.
        """
        self._build_full_dataframe()
        df = self._full_df
        feat_cols = self.get_feature_cols(feature_groups)
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

        # Impute and scale
        X_tr, X_v, X_te, X_oos = (
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
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

        return arrays


# ============================================================================
# 8. EXPERIMENT RUNNERS
# ============================================================================

def run_ridge_correction(data, label):
    """Train Ridge regression on MOS residuals, evaluate on all splits."""
    logger.info("  Training Ridge: %s", label)
    ridge = Ridge(alpha=1.0)
    ridge.fit(data["train"]["X"], data["train"]["y_resid"])

    results = {}
    for split in ["train", "val", "test", "oos"]:
        pred_resid = ridge.predict(data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"{label} {split}")
        results[f"{label}_Ridge_{split}"] = r
    return results


def run_nn_correction(data, hidden_sizes, label, lr=1e-3, epochs=300,
                      patience=20, dropout=0.2, loss_fn="mae"):
    """Train NN on MOS residuals, evaluate on all splits."""
    logger.info("  Training NN %s: %s", hidden_sizes, label)
    n_feat = data["train"]["X"].shape[1]
    model = FlexibleNN(n_feat, hidden_sizes, dropout=dropout)
    train_nn(model, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=lr, epochs=epochs, patience=patience,
             batch_size=128, loss_fn_name=loss_fn)

    results = {}
    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"{label} {split}")
        results[f"{label}_NN{hidden_sizes}_{split}"] = r
    return results


def run_hgb_correction(data, label):
    """Train HistGradientBoosting on MOS residuals, evaluate on all splits."""
    logger.info("  Training HistGradientBoosting: %s", label)
    hgb = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )
    hgb.fit(data["train"]["X"], data["train"]["y_resid"])

    results = {}
    for split in ["train", "val", "test", "oos"]:
        pred_resid = hgb.predict(data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"{label} {split}")
        results[f"{label}_HGB_{split}"] = r

    # Feature importance
    feat_names = data.get("feature_names", [])
    if hasattr(hgb, "feature_importances_") and feat_names:
        importances = hgb.feature_importances_
        fi_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        results[f"{label}_HGB_feature_importance"] = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in fi_pairs[:30]
        ]
    return results


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Step 1: Download station data ----
    logger.info("=" * 70)
    logger.info("STEP 1: Downloading station .dly files ...")
    logger.info("=" * 70)
    download_all_stations()

    # ---- Step 2: Build station matrix ----
    logger.info("=" * 70)
    logger.info("STEP 2: Building station observation matrix ...")
    logger.info("=" * 70)
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # ---- Step 3: Load MOS and Central Park data ----
    logger.info("=" * 70)
    logger.info("STEP 3: Loading MOS and Central Park data ...")
    logger.info("=" * 70)
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # ---- Step 4: Build enhanced dataset ----
    logger.info("=" * 70)
    logger.info("STEP 4: Building enhanced feature dataset ...")
    logger.info("=" * 70)
    builder = EnhancedDatasetBuilder(station_matrix, mos_data, cp_data)

    # ---- Step 5: Run experiments ----
    all_results = {}

    # Define experiment configurations
    # Each config: (name, feature_groups)
    feature_configs = [
        ("Baseline",        ["base"]),
        ("Phase1A",         ["base", "1A"]),
        ("Phase1B",         ["base", "1B"]),
        ("Phase1C",         ["base", "1C"]),
        ("Phase1A+1B",      ["base", "1A", "1B"]),
        ("Phase1A+1C",      ["base", "1A", "1C"]),
        ("Phase1B+1C",      ["base", "1B", "1C"]),
        ("AllPhase1",       ["base", "1A", "1B", "1C"]),
    ]

    for config_name, groups in feature_configs:
        logger.info("=" * 70)
        logger.info("EXPERIMENT: %s (groups: %s)", config_name, groups)
        logger.info("=" * 70)

        data = builder.build_correction_data(groups)
        n_feat = data["train"]["X"].shape[1]
        logger.info("Features: %d, Train: %d, Val: %d, Test: %d, OOS: %d",
                     n_feat,
                     len(data["train"]["y_resid"]),
                     len(data["val"]["y_resid"]),
                     len(data["test"]["y_resid"]),
                     len(data["oos"]["y_resid"]))

        # 1. Ridge correction
        try:
            results = run_ridge_correction(data, config_name)
            all_results.update(results)
        except Exception as e:
            logger.error("Ridge failed for %s: %s", config_name, e, exc_info=True)

        # 2. NN [32, 16] correction (same as C_Correction_NN_tiny)
        try:
            results = run_nn_correction(data, [32, 16], config_name,
                                        lr=1e-3, epochs=200, patience=15,
                                        dropout=0.2, loss_fn="mae")
            all_results.update(results)
        except Exception as e:
            logger.error("NN[32,16] failed for %s: %s", config_name, e, exc_info=True)

        # 3. NN [64, 32] correction
        try:
            results = run_nn_correction(data, [64, 32], config_name,
                                        lr=1e-3, epochs=300, patience=20,
                                        dropout=0.15, loss_fn="huber")
            all_results.update(results)
        except Exception as e:
            logger.error("NN[64,32] failed for %s: %s", config_name, e, exc_info=True)

        # 4. HistGradientBoosting correction
        try:
            results = run_hgb_correction(data, config_name)
            all_results.update(results)
        except Exception as e:
            logger.error("HGB failed for %s: %s", config_name, e, exc_info=True)

    # ---- Step 6: Save results ----
    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    # Clean results for JSON serialization
    clean_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
        elif isinstance(v, list):
            clean_results[k] = v
        else:
            clean_results[k] = v

    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info("Saved results to %s", json_path)

    # Build summary CSV
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
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary to %s", summary_path)

    # ---- Step 7: Print comprehensive results ----
    logger.info("\n" + "=" * 120)
    logger.info("COMPREHENSIVE RESULTS TABLE")
    logger.info("=" * 120)

    if summary_rows:
        # Pivot: for each model, show test MAE, OOS MAE, seasonal
        test_df = summary_df[summary_df["split"] == "test"].copy()
        oos_df = summary_df[summary_df["split"] == "oos"].copy()

        # Merge test + oos by model name
        comparison = test_df[["model", "mae", "rmse", "r2"]].rename(
            columns={"mae": "test_mae", "rmse": "test_rmse", "r2": "test_r2"}
        )
        for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
            if s in test_df.columns:
                comparison[f"test_{s}"] = test_df[s].values

        oos_merge = oos_df[["model", "mae", "rmse", "r2"]].rename(
            columns={"mae": "oos_mae", "rmse": "oos_rmse", "r2": "oos_r2"}
        )
        for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
            if s in oos_df.columns:
                oos_merge[f"oos_{s}"] = oos_df[s].values

        comparison = comparison.merge(oos_merge, on="model", how="outer")
        comparison = comparison.sort_values("test_mae").reset_index(drop=True)

        logger.info("\n%-50s | %7s %7s %6s | %7s %7s %6s | Test Seasonal (DJF/MAM/JJA/SON)",
                    "Model", "TestMAE", "TestRMS", "TestR2", "OOS_MAE", "OOS_RMS", "OOS_R2")
        logger.info("-" * 140)

        baseline_test = None
        baseline_oos = None
        for _, row in comparison.iterrows():
            name = row["model"]
            t_mae = row.get("test_mae", float("nan"))
            t_rmse = row.get("test_rmse", float("nan"))
            t_r2 = row.get("test_r2", float("nan"))
            o_mae = row.get("oos_mae", float("nan"))
            o_rmse = row.get("oos_rmse", float("nan"))
            o_r2 = row.get("oos_r2", float("nan"))

            if baseline_test is None and "Baseline" in name:
                baseline_test = t_mae
                baseline_oos = o_mae

            seasonal = ""
            for s in ["test_mae_DJF", "test_mae_MAM", "test_mae_JJA", "test_mae_SON"]:
                val = row.get(s, float("nan"))
                if pd.notna(val):
                    seasonal += f"{val:.2f}/"
                else:
                    seasonal += "  -- /"
            seasonal = seasonal.rstrip("/")

            logger.info("%-50s | %7.3f %7.3f %6.3f | %7.3f %7.3f %6.3f | %s",
                        name[:50], t_mae, t_rmse, t_r2, o_mae, o_rmse, o_r2, seasonal)

        # Summary: best models
        logger.info("\n" + "=" * 80)
        logger.info("TOP 5 MODELS BY TEST MAE:")
        logger.info("=" * 80)
        top5_test = comparison.nsmallest(5, "test_mae")
        for i, (_, row) in enumerate(top5_test.iterrows(), 1):
            delta_t = ""
            if baseline_test:
                d = row["test_mae"] - baseline_test
                delta_t = f" (delta={d:+.3f} vs baseline)"
            delta_o = ""
            if baseline_oos and pd.notna(row.get("oos_mae")):
                d = row["oos_mae"] - baseline_oos
                delta_o = f" (delta={d:+.3f} vs baseline)"
            logger.info("  #%d: %-45s Test=%.3f%s  OOS=%.3f%s",
                        i, row["model"][:45],
                        row["test_mae"], delta_t,
                        row.get("oos_mae", float("nan")), delta_o)

        logger.info("\nTOP 5 MODELS BY OOS MAE:")
        logger.info("=" * 80)
        top5_oos = comparison.dropna(subset=["oos_mae"]).nsmallest(5, "oos_mae")
        for i, (_, row) in enumerate(top5_oos.iterrows(), 1):
            logger.info("  #%d: %-45s Test=%.3f  OOS=%.3f",
                        i, row["model"][:45], row["test_mae"], row["oos_mae"])

        # Ablation summary
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE GROUP ABLATION SUMMARY (NN [32,16] - C_tiny equivalent):")
        logger.info("=" * 80)
        logger.info("%-25s | %7s | %7s | %s",
                     "Feature Config", "TestMAE", "OOS_MAE", "Delta vs Baseline")
        logger.info("-" * 70)

        for config_name, _ in feature_configs:
            nn_key = f"{config_name}_NN[32, 16]"
            test_row = test_df[test_df["model"] == nn_key]
            oos_row = oos_df[oos_df["model"] == nn_key]
            if len(test_row) > 0 and len(oos_row) > 0:
                t_mae = test_row.iloc[0]["mae"]
                o_mae = oos_row.iloc[0]["mae"]
                # Get baseline NN[32,16] for comparison
                bl_test = test_df[test_df["model"] == "Baseline_NN[32, 16]"]
                bl_oos = oos_df[oos_df["model"] == "Baseline_NN[32, 16]"]
                delta_str = ""
                if len(bl_test) > 0 and len(bl_oos) > 0:
                    dt = t_mae - bl_test.iloc[0]["mae"]
                    do = o_mae - bl_oos.iloc[0]["mae"]
                    delta_str = f"Test:{dt:+.3f}, OOS:{do:+.3f}"
                logger.info("%-25s | %7.3f | %7.3f | %s",
                             config_name, t_mae, o_mae, delta_str)

    elapsed = time.time() - start_time
    logger.info("\nTotal pipeline time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
