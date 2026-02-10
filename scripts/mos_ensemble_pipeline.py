#!/usr/bin/env python3
"""
MOS Ensemble Pipeline for NYC Temperature Prediction.

Combines NOAA GHCN station observations with MOS (Model Output Statistics)
forecast data to build ensemble/stacking models that push MAE below 2.0F.

Models implemented:
  A) MOS-only baseline (Ridge + NN)
  B) Station + MOS hybrid NN
  C) MOS correction (residual learning)
  D) Stacking ensemble
  E) Season-specialized models
  F) Gradient Boosting comparison

All data is REAL -- downloaded from NOAA GHCN and loaded from existing MOS CSVs.
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
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from config_expanded import SURROUNDING_STATIONS, STATION_METADATA
from src.data_collection import download_dly_file, parse_dly_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mos_ensemble_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "mos_ensemble")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Date range for station data download (wide window)
DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# Splits -- MOS-based models
MOS_TRAIN_START, MOS_TRAIN_END = "2004-01-01", "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

# Splits -- station-only models
STATION_TRAIN_START = "1998-01-01"
STATION_TRAIN_END = "2020-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


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

    # Filter completeness >= 80%
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
# 2. FEATURE ENGINEERING
# ============================================================================

def add_date_features(df):
    """Add sin/cos day-of-year encoding."""
    doy = df.index.dayofyear
    df = df.copy()
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def create_lagged_features(station_matrix, lag=1):
    """Shift station observations by `lag` days (so row date t has station values from t-lag)."""
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
# 3. MODEL DEFINITIONS
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

        # Validate
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
# 4. DATASET BUILDER
# ============================================================================

class DatasetBuilder:
    """Builds aligned feature matrices and targets for all splits."""

    def __init__(self, station_matrix, mos_data, cp_data):
        self.station_matrix = station_matrix
        self.mos = mos_data
        self.cp = cp_data

    def build_mos_only(self):
        """Model A: MOS features + date encoding."""
        # Merge MOS + target
        df = self.mos.join(self.cp, how="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])

        # Fill individual MOS columns with ensemble if missing
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])

        df = add_date_features(df)

        feature_cols = ["gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f",
                        "sin_day", "cos_day"]
        return self._split_mos(df, feature_cols, "nyc_tmax")

    def build_station_mos_hybrid(self):
        """Model B: Station lag-1 + MOS + NYC lag-1 + date features. Delta-T target."""
        # Station lag-1
        lagged = create_lagged_features(self.station_matrix, lag=1)

        # NYC lag-1
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        # Merge everything
        df = pd.concat([lagged, self.mos, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax"])

        # Fill MOS NaNs
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])

        df = add_date_features(df)

        # Delta-T target: actual - yesterday's NYC TMAX
        df["delta_t"] = df["nyc_tmax"] - df["nyc_tmax_lag1"]
        df = df.dropna(subset=["delta_t", "nyc_tmax_lag1"])

        # Feature columns: all station lag cols + MOS + NYC lag + date
        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        feature_cols = station_lag_cols + [
            "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f",
            "nyc_tmax_lag1", "sin_day", "cos_day",
        ]

        # Impute remaining NaNs in station features with column training means
        return self._split_mos_delta(df, feature_cols, "delta_t", "nyc_tmax_lag1", "nyc_tmax")

    def build_mos_correction(self):
        """Model C: Residual = actual - MOS_ensemble. Station features predict residual."""
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

        # Compute station_mean (mean of all station lag-1 TMAX cols available)
        station_lag_tmax = [c for c in lagged.columns
                           if c in df.columns and "TMAX" in c]
        if station_lag_tmax:
            df["station_mean_lag1"] = df[station_lag_tmax].mean(axis=1)
            df["mos_station_diff"] = (df["station_mean_lag1"] - df["mos_ensemble_tmax_f"]).abs()
        else:
            df["station_mean_lag1"] = 0.0
            df["mos_station_diff"] = 0.0

        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        feature_cols = station_lag_cols + [
            "mos_ensemble_tmax_f", "nyc_tmax_lag1",
            "station_mean_lag1", "mos_station_diff",
            "sin_day", "cos_day",
        ]

        return self._split_mos_residual(df, feature_cols, "residual",
                                        "mos_ensemble_tmax_f", "nyc_tmax")

    def build_station_only(self):
        """Station-only features (no MOS). Longer training period 1998-2020."""
        lagged = create_lagged_features(self.station_matrix, lag=1)
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        df = pd.concat([lagged, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["nyc_tmax", "nyc_tmax_lag1"])
        df = add_date_features(df)

        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        feature_cols = station_lag_cols + ["nyc_tmax_lag1", "sin_day", "cos_day"]

        return self._split_station_only(df, feature_cols, "nyc_tmax")

    def build_seasonal(self):
        """Model E: Seasonal subsets of the hybrid dataset."""
        lagged = create_lagged_features(self.station_matrix, lag=1)
        nyc_lag1 = self.cp["nyc_tmax"].shift(1).rename("nyc_tmax_lag1")

        df = pd.concat([lagged, self.mos, self.cp, nyc_lag1], axis=1, join="inner")
        df = df.dropna(subset=["mos_ensemble_tmax_f", "nyc_tmax", "nyc_tmax_lag1"])
        df["gfs_mos_tmax_f"] = df["gfs_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df["nam_mos_tmax_f"] = df["nam_mos_tmax_f"].fillna(df["mos_ensemble_tmax_f"])
        df = add_date_features(df)

        station_lag_cols = [c for c in lagged.columns if c in df.columns]
        feature_cols = station_lag_cols + [
            "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f",
            "nyc_tmax_lag1", "sin_day", "cos_day",
        ]

        seasons_s = assign_season(df.index)
        result = {}
        for season_label, months in [("cold", [11, 12, 1, 2, 3]),
                                      ("warm", [4, 5, 6, 7, 8, 9, 10])]:
            mask = df.index.month.isin(months)
            sub = df[mask].copy()

            splits = self._split_mos_raw(sub, feature_cols, "nyc_tmax")
            result[season_label] = splits

        return result

    # ---- Internal split helpers ----

    def _get_mask(self, idx, start, end):
        return (idx >= start) & (idx <= end)

    def _impute_and_scale(self, X_train, X_val, X_test, X_oos=None):
        """Impute NaNs with training mean, then StandardScaler fit on train."""
        train_means = np.nanmean(X_train, axis=0)
        train_means = np.where(np.isnan(train_means), 0.0, train_means)

        for arr in [X_train, X_val, X_test] + ([X_oos] if X_oos is not None else []):
            for j in range(arr.shape[1]):
                mask = np.isnan(arr[:, j])
                arr[mask, j] = train_means[j]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        if X_oos is not None:
            X_oos = scaler.transform(X_oos)
            return X_train, X_val, X_test, X_oos, scaler
        return X_train, X_val, X_test, scaler

    def _split_mos(self, df, feat_cols, target_col):
        """Standard MOS-based splits returning (X,y,dates) for each split."""
        valid_cols = [c for c in feat_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, MOS_TRAIN_START, MOS_TRAIN_END),
            "val": self._get_mask(idx, VAL_START, VAL_END),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y": sub[target_col].values.astype(np.float64),
                "dates": sub.index,
            }

        X_tr, X_v, X_te, X_oos, scaler = self._impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        return arrays

    def _split_mos_delta(self, df, feat_cols, delta_col, lag_col, actual_col):
        """MOS split for delta-T model. Returns delta target + base for reconstruction."""
        valid_cols = [c for c in feat_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, MOS_TRAIN_START, MOS_TRAIN_END),
            "val": self._get_mask(idx, VAL_START, VAL_END),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y_delta": sub[delta_col].values.astype(np.float64),
                "y_actual": sub[actual_col].values.astype(np.float64),
                "base": sub[lag_col].values.astype(np.float64),
                "dates": sub.index,
            }

        X_tr, X_v, X_te, X_oos, scaler = self._impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        return arrays

    def _split_mos_residual(self, df, feat_cols, resid_col, mos_col, actual_col):
        """MOS split for residual correction model."""
        valid_cols = [c for c in feat_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, MOS_TRAIN_START, MOS_TRAIN_END),
            "val": self._get_mask(idx, VAL_START, VAL_END),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y_resid": sub[resid_col].values.astype(np.float64),
                "y_actual": sub[actual_col].values.astype(np.float64),
                "mos_base": sub[mos_col].values.astype(np.float64),
                "dates": sub.index,
            }

        X_tr, X_v, X_te, X_oos, scaler = self._impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        return arrays

    def _split_station_only(self, df, feat_cols, target_col):
        """Station-only splits: longer train 1998-2020."""
        valid_cols = [c for c in feat_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, STATION_TRAIN_START, STATION_TRAIN_END),
            "val": self._get_mask(idx, VAL_START, VAL_END),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y": sub[target_col].values.astype(np.float64),
                "dates": sub.index,
            }

        X_tr, X_v, X_te, X_oos, scaler = self._impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        return arrays

    def _split_mos_raw(self, df, feat_cols, target_col):
        """MOS split for raw target (used by seasonal models)."""
        valid_cols = [c for c in feat_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, MOS_TRAIN_START, MOS_TRAIN_END),
            "val": self._get_mask(idx, VAL_START, VAL_END),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
        }

        arrays = {}
        for split, m in masks.items():
            sub = df[m]
            arrays[split] = {
                "X": sub[valid_cols].values.astype(np.float64),
                "y": sub[target_col].values.astype(np.float64),
                "dates": sub.index,
            }

        # Only scale if we have training data
        if arrays["train"]["X"].shape[0] == 0:
            arrays["feature_names"] = valid_cols
            arrays["scaler"] = None
            return arrays

        X_tr, X_v, X_te, X_oos, scaler = self._impute_and_scale(
            arrays["train"]["X"], arrays["val"]["X"],
            arrays["test"]["X"], arrays["oos"]["X"],
        )
        arrays["train"]["X"] = X_tr
        arrays["val"]["X"] = X_v
        arrays["test"]["X"] = X_te
        arrays["oos"]["X"] = X_oos
        arrays["feature_names"] = valid_cols
        arrays["scaler"] = scaler
        return arrays


# ============================================================================
# 5. MODEL RUNNERS
# ============================================================================

def run_model_a(builder):
    """Model A: MOS-only baseline (Ridge + NN)."""
    logger.info("=" * 70)
    logger.info("MODEL A: MOS-Only Baseline")
    logger.info("=" * 70)

    data = builder.build_mos_only()
    n_feat = data["train"]["X"].shape[1]
    logger.info("Features: %d, Train: %d, Val: %d, Test: %d, OOS: %d",
                n_feat, len(data["train"]["y"]), len(data["val"]["y"]),
                len(data["test"]["y"]), len(data["oos"]["y"]))

    results = {}

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(data["train"]["X"], data["train"]["y"])
    for split in ["train", "val", "test", "oos"]:
        pred = ridge.predict(data[split]["X"])
        r = evaluate_model(data[split]["y"], pred, data[split]["dates"],
                          f"A-Ridge {split}")
        results[f"A_MOS_Ridge_{split}"] = r

    # NN
    model = FlexibleNN(n_feat, [64, 32], dropout=0.1)
    train_nn(model, data["train"]["X"], data["train"]["y"],
             data["val"]["X"], data["val"]["y"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber")
    for split in ["train", "val", "test", "oos"]:
        pred = predict_nn(model, data[split]["X"])
        r = evaluate_model(data[split]["y"], pred, data[split]["dates"],
                          f"A-NN {split}")
        results[f"A_MOS_NN_{split}"] = r

    return results


def run_model_b(builder):
    """Model B: Station + MOS hybrid NN with delta-T target."""
    logger.info("=" * 70)
    logger.info("MODEL B: Station + MOS Hybrid (Delta-T)")
    logger.info("=" * 70)

    data = builder.build_station_mos_hybrid()
    n_feat = data["train"]["X"].shape[1]
    logger.info("Features: %d, Train: %d, Val: %d, Test: %d, OOS: %d",
                n_feat, len(data["train"]["y_delta"]), len(data["val"]["y_delta"]),
                len(data["test"]["y_delta"]), len(data["oos"]["y_delta"]))

    results = {}

    # Ridge on delta-T
    ridge = Ridge(alpha=1.0)
    ridge.fit(data["train"]["X"], data["train"]["y_delta"])
    for split in ["train", "val", "test", "oos"]:
        pred_delta = ridge.predict(data[split]["X"])
        pred_actual = pred_delta + data[split]["base"]
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"B-Ridge {split}")
        results[f"B_Hybrid_Ridge_{split}"] = r

    # NN [256, 128, 64] on delta-T
    model = FlexibleNN(n_feat, [256, 128, 64], dropout=0.15)
    train_nn(model, data["train"]["X"], data["train"]["y_delta"],
             data["val"]["X"], data["val"]["y_delta"],
             lr=5e-4, epochs=400, patience=25, batch_size=128,
             loss_fn_name="huber")
    for split in ["train", "val", "test", "oos"]:
        pred_delta = predict_nn(model, data[split]["X"])
        pred_actual = pred_delta + data[split]["base"]
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"B-NN {split}")
        results[f"B_Hybrid_NN_{split}"] = r

    # Also try smaller NN
    model_small = FlexibleNN(n_feat, [128, 64], dropout=0.1)
    train_nn(model_small, data["train"]["X"], data["train"]["y_delta"],
             data["val"]["X"], data["val"]["y_delta"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber")
    for split in ["train", "val", "test", "oos"]:
        pred_delta = predict_nn(model_small, data[split]["X"])
        pred_actual = pred_delta + data[split]["base"]
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"B-NN-small {split}")
        results[f"B_Hybrid_NN_small_{split}"] = r

    return results, model


def run_model_c(builder):
    """Model C: MOS correction (residual learning)."""
    logger.info("=" * 70)
    logger.info("MODEL C: MOS Correction (Residual Learning)")
    logger.info("=" * 70)

    data = builder.build_mos_correction()
    n_feat = data["train"]["X"].shape[1]
    logger.info("Features: %d, Train: %d, Val: %d, Test: %d, OOS: %d",
                n_feat, len(data["train"]["y_resid"]), len(data["val"]["y_resid"]),
                len(data["test"]["y_resid"]), len(data["oos"]["y_resid"]))

    results = {}

    # Ridge residual correction
    ridge = Ridge(alpha=1.0)
    ridge.fit(data["train"]["X"], data["train"]["y_resid"])
    for split in ["train", "val", "test", "oos"]:
        pred_resid = ridge.predict(data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"C-Ridge {split}")
        results[f"C_Correction_Ridge_{split}"] = r

    # NN residual correction
    model = FlexibleNN(n_feat, [128, 64, 32], dropout=0.1)
    train_nn(model, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber")
    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"C-NN {split}")
        results[f"C_Correction_NN_{split}"] = r

    # Also try very small correction model (to avoid overfitting the residual)
    model_tiny = FlexibleNN(n_feat, [32, 16], dropout=0.2)
    train_nn(model_tiny, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=1e-3, epochs=200, patience=15, loss_fn_name="mae")
    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model_tiny, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual, data[split]["dates"],
                          f"C-NN-tiny {split}")
        results[f"C_Correction_NN_tiny_{split}"] = r

    return results


def run_model_d(builder):
    """Model D: Stacking ensemble."""
    logger.info("=" * 70)
    logger.info("MODEL D: Stacking Ensemble")
    logger.info("=" * 70)

    # Level 0 models need station-only and MOS data
    station_data = builder.build_station_only()
    mos_data = builder.build_mos_only()

    results = {}

    # --- Level 0: Train on train split ---
    # L0-1: Ridge on station features
    ridge_station = Ridge(alpha=1.0)
    ridge_station.fit(station_data["train"]["X"], station_data["train"]["y"])

    # L0-2: NN on station features
    n_station_feat = station_data["train"]["X"].shape[1]
    nn_station = FlexibleNN(n_station_feat, [128, 64], dropout=0.1)
    train_nn(nn_station, station_data["train"]["X"], station_data["train"]["y"],
             station_data["val"]["X"], station_data["val"]["y"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber")

    # L0-3: MOS ensemble (just pass-through -- use mos_ensemble_tmax_f directly)
    # We need the raw MOS values aligned with station data dates

    # Get L0 predictions on val set (for training L1)
    # We need to align dates across station_data and mos_data
    val_dates_station = station_data["val"]["dates"]
    val_dates_mos = mos_data["val"]["dates"]

    # Find common val dates
    common_val = val_dates_station.intersection(val_dates_mos)
    logger.info("Stacking: %d common val dates for L1 training", len(common_val))

    # Create L0 predictions for val set (aligned)
    # Station Ridge on val
    station_val_ridge = ridge_station.predict(station_data["val"]["X"])
    station_val_nn = predict_nn(nn_station, station_data["val"]["X"])

    # Build L0 prediction frame for val
    l0_val_df = pd.DataFrame({
        "ridge_station": station_val_ridge,
        "nn_station": station_val_nn,
    }, index=val_dates_station)

    # Add MOS ensemble for val dates
    mos_val_series = pd.Series(mos_data["val"]["y"], index=val_dates_mos, name="mos_raw")
    # Also get mos predictions from the mos data
    mos_raw_df = builder.mos.loc[val_dates_mos, "mos_ensemble_tmax_f"]

    l0_val_df = l0_val_df.join(mos_raw_df.rename("mos_ensemble"), how="inner")

    # Actual target for val
    actual_val = pd.Series(station_data["val"]["y"], index=val_dates_station, name="actual")
    l0_val_df = l0_val_df.join(actual_val, how="inner")
    l0_val_df = l0_val_df.dropna()

    # Add NYC lag-1 and date features for L1
    cp_data = builder.cp
    nyc_lag1_all = cp_data["nyc_tmax"].shift(1).rename("nyc_lag1")
    l0_val_df = l0_val_df.join(nyc_lag1_all, how="left")
    l0_val_df["sin_day"] = np.sin(2 * np.pi * l0_val_df.index.dayofyear / 365.25)
    l0_val_df["cos_day"] = np.cos(2 * np.pi * l0_val_df.index.dayofyear / 365.25)
    l0_val_df = l0_val_df.dropna()

    l1_features = ["ridge_station", "nn_station", "mos_ensemble", "nyc_lag1", "sin_day", "cos_day"]
    X_l1_train = l0_val_df[l1_features].values
    y_l1_train = l0_val_df["actual"].values

    # L1 model: Ridge
    l1_scaler = StandardScaler()
    X_l1_train_sc = l1_scaler.fit_transform(X_l1_train)
    l1_ridge = Ridge(alpha=0.5)
    l1_ridge.fit(X_l1_train_sc, y_l1_train)

    # L1 model: Small NN
    n_l1_feat = len(l1_features)
    l1_nn = FlexibleNN(n_l1_feat, [16, 8], dropout=0.1, use_batchnorm=False)

    # We need val for L1 -- use test dates as "val" for L1 training
    # But to be proper, we split the L1 training data 80/20
    n_l1 = len(X_l1_train_sc)
    n_l1_tr = int(0.8 * n_l1)
    train_nn(l1_nn,
             X_l1_train_sc[:n_l1_tr], y_l1_train[:n_l1_tr],
             X_l1_train_sc[n_l1_tr:], y_l1_train[n_l1_tr:],
             lr=1e-3, epochs=200, patience=15, batch_size=64, loss_fn_name="huber")

    # Evaluate on test and OOS
    for split in ["test", "oos"]:
        test_dates_station = station_data[split]["dates"]

        # L0 predictions
        test_ridge = ridge_station.predict(station_data[split]["X"])
        test_nn_pred = predict_nn(nn_station, station_data[split]["X"])

        l0_test_df = pd.DataFrame({
            "ridge_station": test_ridge,
            "nn_station": test_nn_pred,
        }, index=test_dates_station)

        # MOS ensemble
        mos_test_dates = mos_data[split]["dates"]
        mos_test_raw = builder.mos.reindex(test_dates_station)["mos_ensemble_tmax_f"]
        l0_test_df["mos_ensemble"] = mos_test_raw.values

        # NYC lag-1 + date
        l0_test_df["nyc_lag1"] = nyc_lag1_all.reindex(test_dates_station).values
        l0_test_df["sin_day"] = np.sin(2 * np.pi * l0_test_df.index.dayofyear / 365.25)
        l0_test_df["cos_day"] = np.cos(2 * np.pi * l0_test_df.index.dayofyear / 365.25)
        l0_test_df = l0_test_df.dropna()

        X_l1_eval = l1_scaler.transform(l0_test_df[l1_features].values)
        y_actual = station_data[split]["y"]

        # Align actuals with available dates
        actual_series = pd.Series(y_actual, index=test_dates_station)
        actual_aligned = actual_series.reindex(l0_test_df.index).values

        # L1 Ridge
        pred_l1_ridge = l1_ridge.predict(X_l1_eval)
        r = evaluate_model(actual_aligned, pred_l1_ridge, l0_test_df.index,
                          f"D-Stack-Ridge {split}")
        results[f"D_Stack_Ridge_{split}"] = r

        # L1 NN
        pred_l1_nn = predict_nn(l1_nn, X_l1_eval)
        r = evaluate_model(actual_aligned, pred_l1_nn, l0_test_df.index,
                          f"D-Stack-NN {split}")
        results[f"D_Stack_NN_{split}"] = r

        # Simple average of L0
        pred_avg = (l0_test_df["ridge_station"].values +
                    l0_test_df["nn_station"].values +
                    l0_test_df["mos_ensemble"].values) / 3
        r = evaluate_model(actual_aligned, pred_avg, l0_test_df.index,
                          f"D-SimpleAvg {split}")
        results[f"D_SimpleAvg_{split}"] = r

    return results


def run_model_e(builder):
    """Model E: Season-specialized models."""
    logger.info("=" * 70)
    logger.info("MODEL E: Season-Specialized Models")
    logger.info("=" * 70)

    seasonal_data = builder.build_seasonal()
    results = {}

    for season_label, data in seasonal_data.items():
        logger.info("--- Season: %s ---", season_label)
        n_feat = data["train"]["X"].shape[1]
        n_train = len(data["train"]["y"])
        logger.info("  Features: %d, Train samples: %d", n_feat, n_train)

        if n_train < 100:
            logger.warning("  Too few training samples for %s, skipping", season_label)
            continue

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(data["train"]["X"], data["train"]["y"])

        for split in ["test", "oos"]:
            if len(data[split]["y"]) == 0:
                continue
            pred = ridge.predict(data[split]["X"])
            r = evaluate_model(data[split]["y"], pred, data[split]["dates"],
                              f"E-{season_label}-Ridge {split}")
            results[f"E_{season_label}_Ridge_{split}"] = r

        # NN
        model = FlexibleNN(n_feat, [128, 64], dropout=0.1)
        train_nn(model, data["train"]["X"], data["train"]["y"],
                 data["val"]["X"], data["val"]["y"],
                 lr=1e-3, epochs=300, patience=20, loss_fn_name="huber")

        for split in ["test", "oos"]:
            if len(data[split]["y"]) == 0:
                continue
            pred = predict_nn(model, data[split]["X"])
            r = evaluate_model(data[split]["y"], pred, data[split]["dates"],
                              f"E-{season_label}-NN {split}")
            results[f"E_{season_label}_NN_{split}"] = r

    # Blend seasonal models: combine cold+warm test predictions
    # For blended result, we need to reassemble full test/oos predictions
    for split in ["test", "oos"]:
        all_dates = []
        all_actuals = []
        all_preds_ridge = []
        all_preds_nn = []

        for season_label, data in seasonal_data.items():
            if len(data[split]["y"]) == 0:
                continue
            n_feat = data["train"]["X"].shape[1]

            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(data["train"]["X"], data["train"]["y"])
            pred_ridge = ridge.predict(data[split]["X"])

            # NN
            model = FlexibleNN(n_feat, [128, 64], dropout=0.1)
            train_nn(model, data["train"]["X"], data["train"]["y"],
                     data["val"]["X"], data["val"]["y"],
                     lr=1e-3, epochs=200, patience=15, loss_fn_name="huber")
            pred_nn = predict_nn(model, data[split]["X"])

            all_dates.extend(data[split]["dates"])
            all_actuals.extend(data[split]["y"])
            all_preds_ridge.extend(pred_ridge)
            all_preds_nn.extend(pred_nn)

        if all_dates:
            r = evaluate_model(np.array(all_actuals), np.array(all_preds_ridge),
                              pd.DatetimeIndex(all_dates),
                              f"E-Blended-Ridge {split}")
            results[f"E_Blended_Ridge_{split}"] = r

            r = evaluate_model(np.array(all_actuals), np.array(all_preds_nn),
                              pd.DatetimeIndex(all_dates),
                              f"E-Blended-NN {split}")
            results[f"E_Blended_NN_{split}"] = r

    return results


def run_model_f(builder):
    """Model F: Gradient Boosting comparison."""
    logger.info("=" * 70)
    logger.info("MODEL F: Gradient Boosting")
    logger.info("=" * 70)

    # Use same features as Model B (hybrid), but with raw target for simplicity
    data_hybrid = builder.build_station_mos_hybrid()
    n_feat = data_hybrid["train"]["X"].shape[1]
    logger.info("Features: %d", n_feat)

    results = {}

    # --- HistGradientBoosting on delta-T ---
    logger.info("Training HistGradientBoostingRegressor (delta-T) ...")
    hgb_delta = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )
    hgb_delta.fit(data_hybrid["train"]["X"], data_hybrid["train"]["y_delta"])

    for split in ["train", "val", "test", "oos"]:
        pred_delta = hgb_delta.predict(data_hybrid[split]["X"])
        pred_actual = pred_delta + data_hybrid[split]["base"]
        r = evaluate_model(data_hybrid[split]["y_actual"], pred_actual,
                          data_hybrid[split]["dates"],
                          f"F-HGB-delta {split}")
        results[f"F_HGB_delta_{split}"] = r

    # --- HistGradientBoosting on raw target ---
    # We need a raw-target version of hybrid data
    # Reuse the same features but predict actual TMAX
    logger.info("Training HistGradientBoostingRegressor (raw TMAX) ...")
    hgb_raw = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )
    hgb_raw.fit(data_hybrid["train"]["X"], data_hybrid["train"]["y_actual"])

    for split in ["train", "val", "test", "oos"]:
        pred = hgb_raw.predict(data_hybrid[split]["X"])
        r = evaluate_model(data_hybrid[split]["y_actual"], pred,
                          data_hybrid[split]["dates"],
                          f"F-HGB-raw {split}")
        results[f"F_HGB_raw_{split}"] = r

    # --- GradientBoosting (sklearn, non-hist, for comparison) ---
    logger.info("Training GradientBoostingRegressor (delta-T) ...")
    gb_delta = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=SEED,
        loss="huber",
        alpha=0.9,
    )
    gb_delta.fit(data_hybrid["train"]["X"], data_hybrid["train"]["y_delta"])

    for split in ["train", "val", "test", "oos"]:
        pred_delta = gb_delta.predict(data_hybrid[split]["X"])
        pred_actual = pred_delta + data_hybrid[split]["base"]
        r = evaluate_model(data_hybrid[split]["y_actual"], pred_actual,
                          data_hybrid[split]["dates"],
                          f"F-GB-delta {split}")
        results[f"F_GB_delta_{split}"] = r

    # Feature importance for best GB model
    feat_names = data_hybrid.get("feature_names", [])
    if hasattr(hgb_delta, "feature_importances_") and feat_names:
        importances = hgb_delta.feature_importances_
        fi_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        results["F_feature_importance_top20"] = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in fi_pairs[:20]
        ]
        logger.info("Top 10 features (HGB):")
        for f, imp in fi_pairs[:10]:
            logger.info("  %s: %.4f", f, imp)

    return results


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Step 1: Download station data ----
    logger.info("STEP 1: Downloading station .dly files ...")
    download_all_stations()

    # ---- Step 2: Build station matrix ----
    logger.info("STEP 2: Building station observation matrix ...")
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # ---- Step 3: Load MOS and Central Park data ----
    logger.info("STEP 3: Loading MOS and Central Park data ...")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # ---- Step 4: Build datasets and run models ----
    builder = DatasetBuilder(station_matrix, mos_data, cp_data)
    all_results = {}

    # Model A: MOS-only baseline
    try:
        results_a = run_model_a(builder)
        all_results.update(results_a)
    except Exception as e:
        logger.error("Model A failed: %s", e, exc_info=True)

    # Model B: Station + MOS hybrid
    best_hybrid_model = None
    try:
        results_b, best_hybrid_model = run_model_b(builder)
        all_results.update(results_b)
    except Exception as e:
        logger.error("Model B failed: %s", e, exc_info=True)

    # Model C: MOS correction
    try:
        results_c = run_model_c(builder)
        all_results.update(results_c)
    except Exception as e:
        logger.error("Model C failed: %s", e, exc_info=True)

    # Model D: Stacking ensemble
    try:
        results_d = run_model_d(builder)
        all_results.update(results_d)
    except Exception as e:
        logger.error("Model D failed: %s", e, exc_info=True)

    # Model E: Seasonal specialists
    try:
        results_e = run_model_e(builder)
        all_results.update(results_e)
    except Exception as e:
        logger.error("Model E failed: %s", e, exc_info=True)

    # Model F: Gradient Boosting
    try:
        results_f = run_model_f(builder)
        all_results.update(results_f)
    except Exception as e:
        logger.error("Model F failed: %s", e, exc_info=True)

    # ---- Step 5: Save results ----
    logger.info("=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    # Save full results JSON
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    # Convert any non-serializable types
    clean_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
        elif isinstance(v, list):
            clean_results[k] = v
        else:
            clean_results[k] = v

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

        # Print summary table
        logger.info("\n" + "=" * 90)
        logger.info("RESULTS SUMMARY (Test Set)")
        logger.info("=" * 90)
        test_rows = summary_df[summary_df["split"] == "test"].sort_values("mae")
        for _, row in test_rows.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                if s in row and pd.notna(row[s]):
                    seasonal += f"  {s.split('_')[1]}={row[s]:.2f}"
            logger.info("  %-35s MAE=%.3f  RMSE=%.3f  R2=%.3f%s",
                        row["model"], row["mae"], row["rmse"], row["r2"], seasonal)

        logger.info("\n" + "=" * 90)
        logger.info("RESULTS SUMMARY (OOS Set)")
        logger.info("=" * 90)
        oos_rows = summary_df[summary_df["split"] == "oos"].sort_values("mae")
        for _, row in oos_rows.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                if s in row and pd.notna(row[s]):
                    seasonal += f"  {s.split('_')[1]}={row[s]:.2f}"
            logger.info("  %-35s MAE=%.3f  RMSE=%.3f  R2=%.3f%s",
                        row["model"], row["mae"], row["rmse"], row["r2"], seasonal)

    # Save best model
    if best_hybrid_model is not None:
        model_path = os.path.join(MODELS_DIR, "best_mos_ensemble.pt")
        torch.save(best_hybrid_model.state_dict(), model_path)
        logger.info("Saved best model to %s", model_path)

    elapsed = time.time() - start_time
    logger.info("\nTotal pipeline time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
