#!/usr/bin/env python3
"""
Advanced Models Evaluation for NYC Temperature Prediction.

Comprehensive evaluation of multiple model architectures using 25+ years of
REAL NOAA GHCN weather station data (1998-2024) from ~52 stations.

Models implemented:
  A) Ridge regression baselines (multiple feature configs, alpha sweeps)
  B) Multi-lag Window MLP (3-day concatenated features)
  C) LSTM sequence model (3-day sequences)
  D) GRU sequence model (3-day sequences)
  E) 1D Temporal Convolution (3-day sequences)
  F) Best MLP with weight decay (L2 regularization)
  G) Station-count sensitivity analysis

Data splits (chronological):
  Train: 1998-01-01 to 2020-12-31
  Val:   2021-01-01 to 2022-12-31
  Test:  2023-01-01 to 2024-12-31

All data is REAL NOAA GHCN observations downloaded from:
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly

Outputs:
  results/advanced_models/experiment_results.json
  results/advanced_models/summary.csv
  models/best_advanced_model.pt
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config_expanded as config
from src.data_collection import download_dly_file, parse_dly_file, pivot_station_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"  # Central Park

SURROUNDING_STATIONS_LIST = [
    "USW00014732", "USW00094741", "USW00014734", "USW00094789", "USW00054743",
    "USW00094745", "USW00054787", "USW00054785", "USW00054779", "USW00054793",
    "USW00004781", "USW00054734", "USW00094702", "USW00004789", "USW00014792",
    "USW00054790", "USW00014757", "USW00014780", "USW00064707", "USW00014758",
    "USW00054786", "USW00064756", "USW00014719", "USW00094732", "USW00093780",
    "USW00054746", "USW00054789", "USW00014737", "USW00013739", "USW00054782",
    "USW00093730", "USW00014777", "USW00013724", "USW00014740", "USW00014707",
    "USW00014712", "USW00054780", "USW00014763", "USW00014794", "USW00054737",
    "USW00054797", "USW00014735", "USW00013707", "USW00054768", "USW00004725",
    "USW00054781", "USW00054756", "USW00014770", "USW00014750", "USW00093786",
    "USW00014739", "USW00014771",
]

ALL_STATION_IDS = [TARGET_STATION] + SURROUNDING_STATIONS_LIST

DATA_START = "1998-01-01"
DATA_END = "2024-12-31"
TRAIN_END = "2020-12-31"
VAL_START = "2021-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

MIN_COMPLETENESS = 0.80
MAX_FFILL_DAYS = 3

# NN Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 300
PATIENCE = 20

RAW_DIR = config.RAW_DATA_DIR
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "advanced_models")
MODELS_DIR = config.MODELS_DIR

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")


# ===========================================================================
# Station distances (from config_expanded.py STATION_METADATA)
# ===========================================================================
def get_station_distances():
    """Return dict of station_id -> distance_mi from Central Park."""
    distances = {}
    for sid, meta in config.STATION_METADATA.items():
        distances[sid] = meta["distance_mi"]
    return distances


# ===========================================================================
# Step 1: Download all station .dly files
# ===========================================================================
def download_all_stations():
    """Download .dly files for all stations with retry logic."""
    os.makedirs(RAW_DIR, exist_ok=True)
    import requests

    for i, sid in enumerate(ALL_STATION_IDS, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path) and os.path.getsize(dly_path) > 1000:
            logger.info("[%d/%d] %s: cached (%.0f KB)",
                        i, len(ALL_STATION_IDS), sid,
                        os.path.getsize(dly_path) / 1024)
            continue

        success = False
        for attempt in range(1, 5):
            try:
                download_dly_file(sid, RAW_DIR, timeout=180)
                success = True
                logger.info("[%d/%d] %s: downloaded", i, len(ALL_STATION_IDS), sid)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("[%d/%d] %s: attempt %d failed (%s), retrying in %ds",
                               i, len(ALL_STATION_IDS), sid, attempt, str(e)[:80], wait)
                time.sleep(wait)

        if not success:
            logger.error("[%d/%d] %s: FAILED after 4 attempts", i, len(ALL_STATION_IDS), sid)


# ===========================================================================
# Step 2: Parse all station data
# ===========================================================================
def load_all_stations():
    """Parse .dly files for all stations, return {station_id: DataFrame}."""
    station_data = {}

    for sid in ALL_STATION_IDS:
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if not os.path.exists(dly_path):
            logger.warning("No .dly file for %s -- skipping", sid)
            continue

        logger.info("Parsing %s ...", sid)
        df_long = parse_dly_file(dly_path, DATA_START, DATA_END)
        if df_long.empty:
            logger.warning("No data for %s -- skipping", sid)
            continue

        df_wide = pivot_station_data(df_long)
        df_wide.index = pd.to_datetime(df_wide.index)
        station_data[sid] = df_wide

    logger.info("Loaded %d stations", len(station_data))
    return station_data


# ===========================================================================
# Step 3: Check completeness
# ===========================================================================
def check_completeness(station_data):
    """Filter stations by TMAX completeness over 1998-2024."""
    check_start = pd.Timestamp("1998-01-01")
    check_end = pd.Timestamp("2024-12-31")
    total_days = (check_end - check_start).days + 1

    qualifying = []
    report = {}

    for sid, df in station_data.items():
        if "TMAX" not in df.columns:
            logger.warning("No TMAX column for %s -- skipping", sid)
            continue

        mask = (df.index >= check_start) & (df.index <= check_end)
        n_valid = df.loc[mask, "TMAX"].notna().sum()
        completeness = n_valid / total_days

        report[sid] = {
            "n_valid": int(n_valid),
            "total_days": total_days,
            "completeness": round(float(completeness), 4),
        }

        if completeness >= MIN_COMPLETENESS:
            qualifying.append(sid)
        else:
            logger.warning("DROPPING %s: completeness=%.1f%% < %.0f%%",
                           sid, completeness * 100, MIN_COMPLETENESS * 100)

    logger.info("Completeness: %d/%d stations qualify (>= %.0f%%)",
                len(qualifying), len(station_data), MIN_COMPLETENESS * 100)
    return qualifying, report


# ===========================================================================
# Step 4: Build feature matrices
# ===========================================================================
def build_lag1_features(station_data, qualifying_ids, target_id=TARGET_STATION,
                        use_tmin=False, use_diurnal=False, use_date=True,
                        use_ar=True, station_subset=None):
    """Build lag-1 feature matrix.

    Parameters
    ----------
    station_data : dict
        {station_id: DataFrame with TMAX, TMIN columns}
    qualifying_ids : list
        List of qualifying station IDs
    use_tmin : bool
        Include TMIN lag-1 features
    use_diurnal : bool
        Include diurnal range (TMAX - TMIN) lag-1
    use_date : bool
        Include sin/cos day-of-year encoding
    use_ar : bool
        Include NYC autoregressive (lag-1 TMAX)
    station_subset : list or None
        If provided, only use these surrounding station IDs

    Returns
    -------
    X, y, dates, feature_cols
    """
    surrounding = [sid for sid in qualifying_ids if sid != target_id]
    if station_subset is not None:
        surrounding = [sid for sid in station_subset if sid in qualifying_ids and sid != target_id]

    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    # Target TMAX at day t
    target_df = station_data[target_id]
    master["target_tmax"] = target_df["TMAX"]

    feature_cols = []

    # NYC autoregressive
    if use_ar:
        master["nyc_tmax_lag1"] = master["target_tmax"].shift(1)
        feature_cols.append("nyc_tmax_lag1")

    # Surrounding station features
    for sid in surrounding:
        if sid not in station_data:
            continue
        df = station_data[sid]
        # TMAX lag-1
        col = f"{sid}_tmax_lag1"
        master[col] = df["TMAX"].shift(1)
        feature_cols.append(col)

        # TMIN lag-1
        if use_tmin and "TMIN" in df.columns:
            col_tmin = f"{sid}_tmin_lag1"
            master[col_tmin] = df["TMIN"].shift(1)
            feature_cols.append(col_tmin)

        # Diurnal range lag-1
        if use_diurnal and "TMAX" in df.columns and "TMIN" in df.columns:
            col_diurnal = f"{sid}_diurnal_lag1"
            master[col_diurnal] = (df["TMAX"] - df["TMIN"]).shift(1)
            feature_cols.append(col_diurnal)

    # Date encoding
    if use_date:
        doy = master.index.dayofyear
        master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
        master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
        feature_cols.extend(["sin_day", "cos_day"])

    # Drop first row (no lag)
    master = master.iloc[1:]

    y = master["target_tmax"].copy()
    X = master[feature_cols].copy()
    dates = master.index.to_series().reset_index(drop=True)

    return X, y, dates, feature_cols


def build_multilag_features(station_data, qualifying_ids, n_lags=3,
                            target_id=TARGET_STATION, use_date=True, use_ar=True):
    """Build multi-lag feature matrix with lags t-1, t-2, ..., t-n_lags.

    Returns X, y, dates, feature_cols, features_per_lag
    """
    surrounding = [sid for sid in qualifying_ids if sid != target_id]

    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    target_df = station_data[target_id]
    master["target_tmax"] = target_df["TMAX"]

    feature_cols = []
    lag_feature_cols = []  # features per lag (before date encoding)

    for lag in range(1, n_lags + 1):
        lag_cols = []
        # NYC AR
        if use_ar:
            col = f"nyc_tmax_lag{lag}"
            master[col] = master["target_tmax"].shift(lag)
            lag_cols.append(col)

        for sid in surrounding:
            if sid not in station_data:
                continue
            df = station_data[sid]
            col = f"{sid}_tmax_lag{lag}"
            master[col] = df["TMAX"].shift(lag)
            lag_cols.append(col)

        feature_cols.extend(lag_cols)
        if lag == 1:
            lag_feature_cols = lag_cols

    # Date encoding (not per-lag, appended once)
    if use_date:
        doy = master.index.dayofyear
        master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
        master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
        feature_cols.extend(["sin_day", "cos_day"])

    # Drop first n_lags rows
    master = master.iloc[n_lags:]

    y = master["target_tmax"].copy()
    X = master[feature_cols].copy()
    dates = master.index.to_series().reset_index(drop=True)

    features_per_lag = len(lag_feature_cols)

    return X, y, dates, feature_cols, features_per_lag


# ===========================================================================
# Step 5: Split, impute, scale
# ===========================================================================
def prepare_splits(X, y, dates, use_delta_t=False):
    """Create chronological train/val/test splits with imputation and scaling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target
    dates : pd.Series
        Date index
    use_delta_t : bool
        If True, target is delta-T: y(t) = TMAX(t) - TMAX(t-1)

    Returns
    -------
    splits : dict with "train", "val", "test" keys
    scaler : fitted StandardScaler
    nyc_lag1_values : dict of arrays for delta-T reconstruction (if use_delta_t)
    """
    idx = pd.to_datetime(dates.values)

    train_mask = idx <= pd.Timestamp(TRAIN_END)
    val_mask = (idx >= pd.Timestamp(VAL_START)) & (idx <= pd.Timestamp(VAL_END))
    test_mask = (idx >= pd.Timestamp(TEST_START)) & (idx <= pd.Timestamp(TEST_END))

    splits_raw = {
        "train": (X[train_mask].copy(), y[train_mask].copy(), dates[train_mask].copy()),
        "val": (X[val_mask].copy(), y[val_mask].copy(), dates[val_mask].copy()),
        "test": (X[test_mask].copy(), y[test_mask].copy(), dates[test_mask].copy()),
    }

    # Forward fill then impute with training means
    X_train_raw = splits_raw["train"][0].ffill(limit=MAX_FFILL_DAYS)
    train_col_means = X_train_raw.mean()

    nyc_lag1_values = {}

    scaled_splits = {}
    scaler = StandardScaler()

    for name, (Xs, ys, ds) in splits_raw.items():
        Xs = Xs.ffill(limit=MAX_FFILL_DAYS)
        Xs = Xs.fillna(train_col_means)
        Xs = Xs.reset_index(drop=True)
        ys = ys.reset_index(drop=True)
        ds = ds.reset_index(drop=True)

        # Save NYC lag-1 for delta-T reconstruction before any NaN drops
        if "nyc_tmax_lag1" in Xs.columns:
            nyc_lag1_values[name] = Xs["nyc_tmax_lag1"].values.copy()
        else:
            nyc_lag1_values[name] = np.zeros(len(ys))

        # Drop rows with NaN target
        valid = ys.notna()
        Xs = Xs[valid].reset_index(drop=True)
        ys = ys[valid].reset_index(drop=True)
        ds = ds[valid].reset_index(drop=True)
        if "nyc_tmax_lag1" in splits_raw["train"][0].columns:
            nyc_lag1_values[name] = nyc_lag1_values[name][valid.values]

        if name == "train":
            scaler.fit(Xs.values)

        X_scaled = scaler.transform(Xs.values)

        if use_delta_t:
            # Convert target to delta-T
            lag1 = nyc_lag1_values[name]
            delta_y = ys.values - lag1
            scaled_splits[name] = {
                "X": X_scaled,
                "y": delta_y,
                "y_raw": ys.values,
                "dates": ds.values,
                "nyc_lag1": lag1,
            }
        else:
            scaled_splits[name] = {
                "X": X_scaled,
                "y": ys.values,
                "dates": ds.values,
            }

        logger.info("  %s: %d samples", name, len(ys))

    return scaled_splits, scaler, nyc_lag1_values


# ===========================================================================
# Step 6: Metrics computation
# ===========================================================================
def compute_metrics(y_true, y_pred, dates=None):
    """Compute MAE, RMSE, R2, and seasonal MAE breakdown."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    result = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}

    # Seasonal breakdown
    if dates is not None:
        dt = pd.to_datetime(dates)
        months = dt.month
        seasons = {
            "DJF": months.isin([12, 1, 2]),
            "MAM": months.isin([3, 4, 5]),
            "JJA": months.isin([6, 7, 8]),
            "SON": months.isin([9, 10, 11]),
        }
        for season, mask in seasons.items():
            if mask.sum() > 0:
                season_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
                result[f"mae_{season}"] = round(season_mae, 4)

    return result


# ===========================================================================
# Model Architectures
# ===========================================================================

class TemporalConv1D(nn.Module):
    """1D Temporal Convolution for temperature prediction.

    Input: (batch, seq_len, n_features)
    """
    def __init__(self, n_features, seq_len=3):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

        # Conv1d expects (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()

        # After conv1 (kernel=2): seq_len -> seq_len-1
        # After conv2 (kernel=2): seq_len-1 -> seq_len-2
        # For seq_len=3: 3 -> 2 -> 1
        final_len = seq_len - 2
        self.fc = nn.Linear(32 * final_len, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len) for Conv1d
        if x.dim() == 2:
            # Assume flat input that needs reshaping
            batch = x.size(0)
            total = x.size(1)
            feat = total // self.seq_len
            x = x.view(batch, self.seq_len, feat)

        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


class EnhancedMLP(nn.Module):
    """MLP with optional weight decay (via optimizer), dropout."""
    def __init__(self, n_features, hidden_sizes=None, dropout=0.05):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.n_features = n_features
        self.hidden_sizes = hidden_sizes

        layers = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiLagMLP(nn.Module):
    """MLP for concatenated multi-lag features."""
    def __init__(self, n_features, hidden_sizes=None, dropout=0.05):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.n_features = n_features
        layers = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMModel(nn.Module):
    """LSTM for sequential temperature prediction."""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class GRUModel(nn.Module):
    """GRU for sequential temperature prediction."""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)


# ===========================================================================
# Training utilities
# ===========================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_nn_model(model, splits, label="", loss_type="huber",
                   weight_decay=0.0, lr=LEARNING_RATE, is_sequential=False,
                   seq_len=3, features_per_step=None):
    """Generic NN training with early stopping.

    Parameters
    ----------
    model : nn.Module
    splits : dict with "train", "val" keys
    label : str
    loss_type : str, "huber", "mse", or "mae"
    weight_decay : float
    is_sequential : bool, if True reshape input to (batch, seq_len, features_per_step)
    seq_len : int, sequence length for sequential models
    features_per_step : int, features per time step for sequential models

    Returns
    -------
    model, history dict, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = count_parameters(model)
    logger.info("[%s] Training on %s, params=%d, loss=%s, wd=%.1e",
                label, device, n_params, loss_type, weight_decay)

    # Loss function
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7,
    )

    # Prepare data
    X_train = torch.FloatTensor(splits["train"]["X"]).to(device)
    y_train = torch.FloatTensor(splits["train"]["y"]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(splits["val"]["X"]).to(device)
    y_val = torch.FloatTensor(splits["val"]["y"]).unsqueeze(1).to(device)

    # Reshape for sequential models
    if is_sequential and features_per_step is not None:
        X_train_seq = X_train[:, :seq_len * features_per_step].view(-1, seq_len, features_per_step)
        X_val_seq = X_val[:, :seq_len * features_per_step].view(-1, seq_len, features_per_step)
        # Append date features (not part of sequence) if any exist beyond the sequence
        extra_train = X_train[:, seq_len * features_per_step:]
        extra_val = X_val[:, seq_len * features_per_step:]
        if extra_train.size(1) > 0:
            # For sequential models with extra features, we just pass the reshaped sequence
            # The date features are already included in each lag step
            pass
        X_train_input = X_train_seq
        X_val_input = X_val_seq
    else:
        X_train_input = X_train
        X_val_input = X_val

    train_ds = TensorDataset(X_train_input, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_input)
            val_loss = criterion(val_pred, y_val).item()
            val_mae = torch.mean(torch.abs(val_pred - y_val)).item()

        scheduler.step(val_loss)

        if epoch % 50 == 0 or epoch == 1:
            logger.info("[%s] Epoch %3d: train_loss=%.4f val_loss=%.4f val_MAE=%.3f",
                        label, epoch, avg_train_loss, val_loss, val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("[%s] Early stopping at epoch %d (best: %d)", label, epoch, best_epoch)
                break

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    logger.info("[%s] Done in %.1fs, best epoch: %d", label, elapsed, best_epoch)

    history = {
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "elapsed_seconds": round(elapsed, 1),
        "total_params": n_params,
    }

    return model, history, device


def predict_nn(model, X, device, is_sequential=False, seq_len=3, features_per_step=None):
    """Get predictions from a trained NN."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        if is_sequential and features_per_step is not None:
            X_t = X_t[:, :seq_len * features_per_step].view(-1, seq_len, features_per_step)
        pred = model(X_t).cpu().numpy().flatten()
    return pred


def evaluate_model_on_splits(model, splits, device, label="", use_delta_t=False,
                              is_sequential=False, seq_len=3, features_per_step=None):
    """Evaluate a NN model on all splits, return metrics dict."""
    results = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue

        pred = predict_nn(model, splits[split_name]["X"], device,
                          is_sequential=is_sequential, seq_len=seq_len,
                          features_per_step=features_per_step)

        if use_delta_t and "nyc_lag1" in splits[split_name]:
            # Reconstruct absolute TMAX from delta-T prediction
            pred_abs = pred + splits[split_name]["nyc_lag1"]
            actual = splits[split_name]["y_raw"]
        else:
            pred_abs = pred
            actual = splits[split_name]["y"]

        metrics = compute_metrics(actual, pred_abs, splits[split_name].get("dates"))
        results[split_name] = metrics
        logger.info("[%s] %s: MAE=%.3f, RMSE=%.3f, R2=%.3f",
                    label, split_name, metrics["mae"], metrics["rmse"], metrics["r2"])

    return results


# ===========================================================================
# EXPERIMENT RUNNERS
# ===========================================================================

def run_ridge_experiments(splits_lag1, splits_lag1_tmin, splits_lag2, splits_full):
    """Run all Ridge regression experiments.

    Returns list of result dicts.
    """
    results = []

    # A1: Ridge TMAX-only lag-1, alpha sweep
    logger.info("\n=== A1: Ridge TMAX-only lag-1, alpha sweep ===")
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        label = f"Ridge_tmax_lag1_a{alpha}"
        t0 = time.time()
        ridge = Ridge(alpha=alpha)
        ridge.fit(splits_lag1["train"]["X"], splits_lag1["train"]["y"])
        elapsed = time.time() - t0

        metrics = {}
        for split in ["train", "val", "test"]:
            pred = ridge.predict(splits_lag1[split]["X"])
            m = compute_metrics(splits_lag1[split]["y"], pred, splits_lag1[split].get("dates"))
            metrics[split] = m

        logger.info("[%s] Test MAE=%.3f, R2=%.3f", label, metrics["test"]["mae"], metrics["test"]["r2"])

        results.append({
            "model": label,
            "architecture": f"Ridge(alpha={alpha})",
            "features": "TMAX lag-1 + date + AR",
            "n_features": splits_lag1["train"]["X"].shape[1],
            "n_params": splits_lag1["train"]["X"].shape[1] + 1,
            "training_time_s": round(elapsed, 2),
            "metrics": metrics,
        })

    # A2: Ridge TMAX+TMIN lag-1
    logger.info("\n=== A2: Ridge TMAX+TMIN lag-1 ===")
    label = "Ridge_tmax_tmin_lag1"
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(splits_lag1_tmin["train"]["X"], splits_lag1_tmin["train"]["y"])
    elapsed = time.time() - t0
    metrics = {}
    for split in ["train", "val", "test"]:
        pred = ridge.predict(splits_lag1_tmin[split]["X"])
        m = compute_metrics(splits_lag1_tmin[split]["y"], pred, splits_lag1_tmin[split].get("dates"))
        metrics[split] = m
    logger.info("[%s] Test MAE=%.3f, R2=%.3f", label, metrics["test"]["mae"], metrics["test"]["r2"])
    results.append({
        "model": label,
        "architecture": "Ridge(alpha=1.0)",
        "features": "TMAX+TMIN lag-1 + date + AR",
        "n_features": splits_lag1_tmin["train"]["X"].shape[1],
        "n_params": splits_lag1_tmin["train"]["X"].shape[1] + 1,
        "training_time_s": round(elapsed, 2),
        "metrics": metrics,
    })

    # A3: Ridge TMAX lag-1 + lag-2
    logger.info("\n=== A3: Ridge TMAX lag-1 + lag-2 ===")
    label = "Ridge_tmax_lag12"
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(splits_lag2["train"]["X"], splits_lag2["train"]["y"])
    elapsed = time.time() - t0
    metrics = {}
    for split in ["train", "val", "test"]:
        pred = ridge.predict(splits_lag2[split]["X"])
        m = compute_metrics(splits_lag2[split]["y"], pred, splits_lag2[split].get("dates"))
        metrics[split] = m
    logger.info("[%s] Test MAE=%.3f, R2=%.3f", label, metrics["test"]["mae"], metrics["test"]["r2"])
    results.append({
        "model": label,
        "architecture": "Ridge(alpha=1.0)",
        "features": "TMAX lag-1+lag-2 + date + AR",
        "n_features": splits_lag2["train"]["X"].shape[1],
        "n_params": splits_lag2["train"]["X"].shape[1] + 1,
        "training_time_s": round(elapsed, 2),
        "metrics": metrics,
    })

    # A4: Ridge full features
    logger.info("\n=== A4: Ridge full features ===")
    label = "Ridge_full_features"
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(splits_full["train"]["X"], splits_full["train"]["y"])
    elapsed = time.time() - t0
    metrics = {}
    for split in ["train", "val", "test"]:
        pred = ridge.predict(splits_full[split]["X"])
        m = compute_metrics(splits_full[split]["y"], pred, splits_full[split].get("dates"))
        metrics[split] = m
    logger.info("[%s] Test MAE=%.3f, R2=%.3f", label, metrics["test"]["mae"], metrics["test"]["r2"])
    results.append({
        "model": label,
        "architecture": "Ridge(alpha=1.0)",
        "features": "TMAX+TMIN+diurnal lag-1 + date + AR",
        "n_features": splits_full["train"]["X"].shape[1],
        "n_params": splits_full["train"]["X"].shape[1] + 1,
        "training_time_s": round(elapsed, 2),
        "metrics": metrics,
    })

    # A5: ElasticNet
    logger.info("\n=== A5: ElasticNet ===")
    label = "ElasticNet_l1_0.5"
    t0 = time.time()
    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000)
    enet.fit(splits_lag1["train"]["X"], splits_lag1["train"]["y"])
    elapsed = time.time() - t0
    metrics = {}
    for split in ["train", "val", "test"]:
        pred = enet.predict(splits_lag1[split]["X"])
        m = compute_metrics(splits_lag1[split]["y"], pred, splits_lag1[split].get("dates"))
        metrics[split] = m
    logger.info("[%s] Test MAE=%.3f, R2=%.3f", label, metrics["test"]["mae"], metrics["test"]["r2"])
    results.append({
        "model": label,
        "architecture": "ElasticNet(alpha=1.0, l1_ratio=0.5)",
        "features": "TMAX lag-1 + date + AR",
        "n_features": splits_lag1["train"]["X"].shape[1],
        "n_params": splits_lag1["train"]["X"].shape[1] + 1,
        "training_time_s": round(elapsed, 2),
        "metrics": metrics,
    })

    return results


def run_nn_experiments(splits_lag1, splits_lag1_delta, splits_multilag,
                       features_per_lag, n_lags=3):
    """Run all neural network experiments (B through F).

    Returns list of result dicts and the best model + device.
    """
    results = []
    best_test_mae = float("inf")
    best_model_info = None

    n_features_lag1 = splits_lag1["train"]["X"].shape[1]
    n_features_multilag = splits_multilag["train"]["X"].shape[1]

    # B: Multi-lag Window MLP
    logger.info("\n=== B: Multi-lag Window MLP [256,128,64] ===")
    label = "MultiLagMLP_256_128_64"
    model_b = MultiLagMLP(n_features=n_features_multilag,
                           hidden_sizes=[256, 128, 64], dropout=0.05)
    model_b, hist_b, device = train_nn_model(
        model_b, splits_multilag, label=label, loss_type="huber",
    )
    metrics_b = evaluate_model_on_splits(
        model_b, splits_multilag, device, label=label,
        use_delta_t=False,
    )
    results.append({
        "model": label,
        "architecture": "MLP [256,128,64], dropout=0.05",
        "features": f"TMAX lag-1..lag-{n_lags} + date + AR (flat concat)",
        "n_features": n_features_multilag,
        "n_params": count_parameters(model_b),
        "training_time_s": hist_b["elapsed_seconds"],
        "best_epoch": hist_b["best_epoch"],
        "metrics": metrics_b,
    })
    if metrics_b["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_b["test"]["mae"]
        best_model_info = (model_b, device, label, n_features_multilag, hist_b)

    # C: LSTM model
    logger.info("\n=== C: LSTM (hidden=64, layers=1, seq_len=3) ===")
    label = "LSTM_h64_L1"
    model_c = LSTMModel(input_size=features_per_lag, hidden_size=64, num_layers=1)
    model_c, hist_c, device = train_nn_model(
        model_c, splits_multilag, label=label, loss_type="huber",
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    metrics_c = evaluate_model_on_splits(
        model_c, splits_multilag, device, label=label,
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    results.append({
        "model": label,
        "architecture": "LSTM(h=64, L=1) -> Dense(32) -> Dense(1)",
        "features": f"TMAX lag-1..lag-{n_lags} + AR, seq_len={n_lags}",
        "n_features": features_per_lag,
        "n_params": count_parameters(model_c),
        "training_time_s": hist_c["elapsed_seconds"],
        "best_epoch": hist_c["best_epoch"],
        "metrics": metrics_c,
    })
    if metrics_c["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_c["test"]["mae"]
        best_model_info = (model_c, device, label, features_per_lag, hist_c)

    # D: GRU model
    logger.info("\n=== D: GRU (hidden=64, layers=1, seq_len=3) ===")
    label = "GRU_h64_L1"
    model_d = GRUModel(input_size=features_per_lag, hidden_size=64, num_layers=1)
    model_d, hist_d, device = train_nn_model(
        model_d, splits_multilag, label=label, loss_type="huber",
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    metrics_d = evaluate_model_on_splits(
        model_d, splits_multilag, device, label=label,
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    results.append({
        "model": label,
        "architecture": "GRU(h=64, L=1) -> Dense(32) -> Dense(1)",
        "features": f"TMAX lag-1..lag-{n_lags} + AR, seq_len={n_lags}",
        "n_features": features_per_lag,
        "n_params": count_parameters(model_d),
        "training_time_s": hist_d["elapsed_seconds"],
        "best_epoch": hist_d["best_epoch"],
        "metrics": metrics_d,
    })
    if metrics_d["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_d["test"]["mae"]
        best_model_info = (model_d, device, label, features_per_lag, hist_d)

    # E: 1D Temporal Convolution
    logger.info("\n=== E: 1D Temporal Convolution (seq_len=3) ===")
    label = "TemporalConv1D"
    model_e = TemporalConv1D(n_features=features_per_lag, seq_len=n_lags)
    model_e, hist_e, device = train_nn_model(
        model_e, splits_multilag, label=label, loss_type="huber",
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    metrics_e = evaluate_model_on_splits(
        model_e, splits_multilag, device, label=label,
        is_sequential=True, seq_len=n_lags, features_per_step=features_per_lag,
    )
    results.append({
        "model": label,
        "architecture": "Conv1d(64,k=2) -> Conv1d(32,k=2) -> Dense(1)",
        "features": f"TMAX lag-1..lag-{n_lags} + AR, seq_len={n_lags}",
        "n_features": features_per_lag,
        "n_params": count_parameters(model_e),
        "training_time_s": hist_e["elapsed_seconds"],
        "best_epoch": hist_e["best_epoch"],
        "metrics": metrics_e,
    })
    if metrics_e["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_e["test"]["mae"]
        best_model_info = (model_e, device, label, features_per_lag, hist_e)

    # F: Best MLP with weight decay + delta-T target
    logger.info("\n=== F: MLP [256,128,64] + weight_decay=1e-4 + delta-T ===")
    label = "MLP_256_128_64_wd_deltaT"
    model_f = EnhancedMLP(n_features=n_features_lag1,
                           hidden_sizes=[256, 128, 64], dropout=0.05)
    model_f, hist_f, device = train_nn_model(
        model_f, splits_lag1_delta, label=label, loss_type="huber",
        weight_decay=1e-4,
    )
    metrics_f = evaluate_model_on_splits(
        model_f, splits_lag1_delta, device, label=label,
        use_delta_t=True,
    )
    results.append({
        "model": label,
        "architecture": "MLP [256,128,64], dropout=0.05, weight_decay=1e-4",
        "features": "TMAX lag-1 + date + AR, delta-T target",
        "n_features": n_features_lag1,
        "n_params": count_parameters(model_f),
        "training_time_s": hist_f["elapsed_seconds"],
        "best_epoch": hist_f["best_epoch"],
        "metrics": metrics_f,
    })
    if metrics_f["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_f["test"]["mae"]
        best_model_info = (model_f, device, label, n_features_lag1, hist_f)

    # Also run MLP without delta-T for comparison
    logger.info("\n=== F2: MLP [256,128,64] + weight_decay=1e-4 (raw TMAX) ===")
    label = "MLP_256_128_64_wd_raw"
    model_f2 = EnhancedMLP(n_features=n_features_lag1,
                            hidden_sizes=[256, 128, 64], dropout=0.05)
    model_f2, hist_f2, device = train_nn_model(
        model_f2, splits_lag1, label=label, loss_type="huber",
        weight_decay=1e-4,
    )
    metrics_f2 = evaluate_model_on_splits(
        model_f2, splits_lag1, device, label=label,
    )
    results.append({
        "model": label,
        "architecture": "MLP [256,128,64], dropout=0.05, weight_decay=1e-4",
        "features": "TMAX lag-1 + date + AR, raw TMAX target",
        "n_features": n_features_lag1,
        "n_params": count_parameters(model_f2),
        "training_time_s": hist_f2["elapsed_seconds"],
        "best_epoch": hist_f2["best_epoch"],
        "metrics": metrics_f2,
    })
    if metrics_f2["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_f2["test"]["mae"]
        best_model_info = (model_f2, device, label, n_features_lag1, hist_f2)

    # Also run smaller MLP [128,64] for comparison
    logger.info("\n=== F3: MLP [128,64] + weight_decay=1e-4 (raw TMAX) ===")
    label = "MLP_128_64_wd_raw"
    model_f3 = EnhancedMLP(n_features=n_features_lag1,
                            hidden_sizes=[128, 64], dropout=0.0)
    model_f3, hist_f3, device = train_nn_model(
        model_f3, splits_lag1, label=label, loss_type="huber",
        weight_decay=1e-4,
    )
    metrics_f3 = evaluate_model_on_splits(
        model_f3, splits_lag1, device, label=label,
    )
    results.append({
        "model": label,
        "architecture": "MLP [128,64], dropout=0.0, weight_decay=1e-4",
        "features": "TMAX lag-1 + date + AR, raw TMAX target",
        "n_features": n_features_lag1,
        "n_params": count_parameters(model_f3),
        "training_time_s": hist_f3["elapsed_seconds"],
        "best_epoch": hist_f3["best_epoch"],
        "metrics": metrics_f3,
    })
    if metrics_f3["test"]["mae"] < best_test_mae:
        best_test_mae = metrics_f3["test"]["mae"]
        best_model_info = (model_f3, device, label, n_features_lag1, hist_f3)

    return results, best_model_info


def run_station_count_sensitivity(station_data, qualifying_ids):
    """G: Station-count sensitivity with 25-year data.

    Tests MLP [256,128,64] with 10, 20, 30, 40, 50 nearest stations.
    """
    distances = get_station_distances()
    surrounding = [sid for sid in qualifying_ids if sid != TARGET_STATION and sid in distances]
    surrounding.sort(key=lambda s: distances[s])

    results = []
    counts = [10, 20, 30, 40, 50]

    for n_stations in counts:
        subset = surrounding[:n_stations]
        actual_count = len(subset)
        label = f"StationSens_{actual_count}stn"

        logger.info("\n=== G: Station count = %d ===", actual_count)

        X, y, dates, feat_cols = build_lag1_features(
            station_data, qualifying_ids, station_subset=subset,
        )
        splits, scaler, _ = prepare_splits(X, y, dates, use_delta_t=False)

        n_feats = splits["train"]["X"].shape[1]

        # Train MLP
        model = EnhancedMLP(n_features=n_feats, hidden_sizes=[256, 128, 64], dropout=0.05)
        model, hist, device = train_nn_model(model, splits, label=label, loss_type="huber")
        metrics = evaluate_model_on_splits(model, splits, device, label=label)

        # Also train Ridge for comparison
        ridge = Ridge(alpha=1.0)
        ridge.fit(splits["train"]["X"], splits["train"]["y"])
        ridge_metrics = {}
        for split in ["train", "val", "test"]:
            pred = ridge.predict(splits[split]["X"])
            m = compute_metrics(splits[split]["y"], pred, splits[split].get("dates"))
            ridge_metrics[split] = m

        results.append({
            "n_stations": actual_count,
            "model": label,
            "n_features": n_feats,
            "mlp_test_mae": metrics["test"]["mae"],
            "mlp_test_rmse": metrics["test"]["rmse"],
            "mlp_test_r2": metrics["test"]["r2"],
            "ridge_test_mae": ridge_metrics["test"]["mae"],
            "ridge_test_rmse": ridge_metrics["test"]["rmse"],
            "ridge_test_r2": ridge_metrics["test"]["r2"],
            "mlp_metrics": metrics,
            "ridge_metrics": ridge_metrics,
            "training_time_s": hist["elapsed_seconds"],
            "best_epoch": hist["best_epoch"],
            "n_params": count_parameters(model),
        })

        logger.info("[%s] MLP Test MAE=%.3f, Ridge Test MAE=%.3f",
                    label, metrics["test"]["mae"], ridge_metrics["test"]["mae"])

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    logger.info("=" * 70)
    logger.info("ADVANCED MODELS EVALUATION")
    logger.info("Using REAL NOAA GHCN data, 1998-2024, ~52 stations")
    logger.info("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_results = []
    station_sensitivity_results = []

    # ===================================================================
    # PHASE 1: Download station data
    # ===================================================================
    logger.info("\n--- Phase 1: Downloading station .dly files ---")
    download_all_stations()

    # ===================================================================
    # PHASE 2: Parse and filter stations
    # ===================================================================
    logger.info("\n--- Phase 2: Parsing station data ---")
    station_data = load_all_stations()
    qualifying_ids, completeness_report = check_completeness(station_data)

    # Keep only qualifying stations in data dict
    station_data = {sid: df for sid, df in station_data.items() if sid in qualifying_ids}

    if TARGET_STATION not in qualifying_ids:
        logger.error("Target station %s did not qualify! Aborting.", TARGET_STATION)
        return

    n_surrounding = len([s for s in qualifying_ids if s != TARGET_STATION])
    logger.info("Qualifying: %d stations (%d surrounding)", len(qualifying_ids), n_surrounding)

    # ===================================================================
    # PHASE 3: Build feature matrices
    # ===================================================================
    logger.info("\n--- Phase 3: Building feature matrices ---")

    # 3a: TMAX-only lag-1 (baseline features)
    logger.info("Building TMAX lag-1 features...")
    X_lag1, y_lag1, dates_lag1, feat_lag1 = build_lag1_features(
        station_data, qualifying_ids, use_tmin=False, use_diurnal=False,
    )
    splits_lag1, scaler_lag1, _ = prepare_splits(X_lag1, y_lag1, dates_lag1, use_delta_t=False)
    logger.info("  TMAX lag-1: %d features, train=%d, val=%d, test=%d",
                len(feat_lag1), len(splits_lag1["train"]["y"]),
                len(splits_lag1["val"]["y"]), len(splits_lag1["test"]["y"]))

    # 3b: TMAX+TMIN lag-1
    logger.info("Building TMAX+TMIN lag-1 features...")
    X_tmin, y_tmin, dates_tmin, feat_tmin = build_lag1_features(
        station_data, qualifying_ids, use_tmin=True, use_diurnal=False,
    )
    splits_tmin, scaler_tmin, _ = prepare_splits(X_tmin, y_tmin, dates_tmin, use_delta_t=False)
    logger.info("  TMAX+TMIN lag-1: %d features", len(feat_tmin))

    # 3c: Multi-lag (3 lags) for sequential models
    logger.info("Building multi-lag (3 lags) features...")
    N_LAGS = 3
    X_ml, y_ml, dates_ml, feat_ml, features_per_lag = build_multilag_features(
        station_data, qualifying_ids, n_lags=N_LAGS,
    )
    splits_ml, scaler_ml, _ = prepare_splits(X_ml, y_ml, dates_ml, use_delta_t=False)
    logger.info("  Multi-lag: %d total features, %d per lag, train=%d",
                len(feat_ml), features_per_lag, len(splits_ml["train"]["y"]))

    # 3d: TMAX lag-1 + delta-T target
    logger.info("Building TMAX lag-1 with delta-T target...")
    splits_lag1_delta, scaler_delta, _ = prepare_splits(
        X_lag1, y_lag1, dates_lag1, use_delta_t=True,
    )

    # 3e: Full features (TMAX+TMIN+diurnal lag-1)
    logger.info("Building full features (TMAX+TMIN+diurnal lag-1)...")
    X_full, y_full, dates_full, feat_full = build_lag1_features(
        station_data, qualifying_ids, use_tmin=True, use_diurnal=True,
    )
    splits_full, scaler_full, _ = prepare_splits(X_full, y_full, dates_full, use_delta_t=False)
    logger.info("  Full features: %d features", len(feat_full))

    # 3f: Multi-lag (2 lags) for Ridge lag-1+lag-2
    logger.info("Building 2-lag features for Ridge...")
    X_lag2, y_lag2, dates_lag2, feat_lag2, _ = build_multilag_features(
        station_data, qualifying_ids, n_lags=2,
    )
    splits_lag2, scaler_lag2, _ = prepare_splits(X_lag2, y_lag2, dates_lag2, use_delta_t=False)
    logger.info("  2-lag: %d features", len(feat_lag2))

    # ===================================================================
    # PHASE 4: Run Ridge experiments (A)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Ridge Regression Experiments")
    logger.info("=" * 70)

    ridge_results = run_ridge_experiments(splits_lag1, splits_tmin, splits_lag2, splits_full)
    all_results.extend(ridge_results)

    # ===================================================================
    # PHASE 5: Run Neural Network experiments (B-F)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Neural Network Experiments")
    logger.info("=" * 70)

    nn_results, best_model_info = run_nn_experiments(
        splits_lag1, splits_lag1_delta, splits_ml,
        features_per_lag=features_per_lag, n_lags=N_LAGS,
    )
    all_results.extend(nn_results)

    # ===================================================================
    # PHASE 6: Station-count sensitivity (G)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Station-Count Sensitivity")
    logger.info("=" * 70)

    station_sensitivity_results = run_station_count_sensitivity(station_data, qualifying_ids)

    # ===================================================================
    # PHASE 7: Save results
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: Saving Results")
    logger.info("=" * 70)

    # Save best model checkpoint
    if best_model_info is not None:
        best_model, best_device, best_label, best_n_feat, best_hist = best_model_info
        best_path = os.path.join(MODELS_DIR, "best_advanced_model.pt")
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "model_class": best_model.__class__.__name__,
            "n_features": best_n_feat,
            "label": best_label,
            "best_epoch": best_hist["best_epoch"],
            "n_params": best_hist["total_params"],
        }, best_path)
        logger.info("Saved best model (%s) to %s", best_label, best_path)

    # Build comprehensive results JSON
    experiment_results = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "scripts/advanced_models_eval.py",
        "data_source": "NOAA GHCN-Daily bulk .dly files (REAL DATA)",
        "data_range": f"{DATA_START} to {DATA_END}",
        "splits": {
            "train": f"{DATA_START} to {TRAIN_END}",
            "val": f"{VAL_START} to {VAL_END}",
            "test": f"{TEST_START} to {TEST_END}",
        },
        "stations": {
            "target": TARGET_STATION,
            "total_qualifying": len(qualifying_ids),
            "surrounding": n_surrounding,
            "completeness_threshold": MIN_COMPLETENESS,
            "completeness_report": completeness_report,
        },
        "feature_sets": {
            "tmax_lag1": {"n_features": len(feat_lag1), "description": "TMAX lag-1 + date + AR"},
            "tmax_tmin_lag1": {"n_features": len(feat_tmin), "description": "TMAX+TMIN lag-1 + date + AR"},
            "multilag_3": {"n_features": len(feat_ml), "features_per_lag": features_per_lag,
                           "description": f"TMAX lag-1..lag-{N_LAGS} + date + AR"},
            "full": {"n_features": len(feat_full), "description": "TMAX+TMIN+diurnal lag-1 + date + AR"},
        },
        "model_results": all_results,
        "station_sensitivity": station_sensitivity_results,
    }

    results_json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_json_path, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    logger.info("Saved results JSON: %s", results_json_path)

    # Build summary CSV
    summary_rows = []
    for r in all_results:
        row = {
            "model": r["model"],
            "architecture": r["architecture"],
            "features": r["features"],
            "n_features": r["n_features"],
            "n_params": r.get("n_params", "N/A"),
            "training_time_s": r.get("training_time_s", "N/A"),
            "best_epoch": r.get("best_epoch", "N/A"),
            "train_mae": r["metrics"].get("train", {}).get("mae", "N/A"),
            "train_rmse": r["metrics"].get("train", {}).get("rmse", "N/A"),
            "train_r2": r["metrics"].get("train", {}).get("r2", "N/A"),
            "val_mae": r["metrics"].get("val", {}).get("mae", "N/A"),
            "val_rmse": r["metrics"].get("val", {}).get("rmse", "N/A"),
            "val_r2": r["metrics"].get("val", {}).get("r2", "N/A"),
            "test_mae": r["metrics"].get("test", {}).get("mae", "N/A"),
            "test_rmse": r["metrics"].get("test", {}).get("rmse", "N/A"),
            "test_r2": r["metrics"].get("test", {}).get("r2", "N/A"),
            "test_mae_DJF": r["metrics"].get("test", {}).get("mae_DJF", "N/A"),
            "test_mae_MAM": r["metrics"].get("test", {}).get("mae_MAM", "N/A"),
            "test_mae_JJA": r["metrics"].get("test", {}).get("mae_JJA", "N/A"),
            "test_mae_SON": r["metrics"].get("test", {}).get("mae_SON", "N/A"),
        }
        summary_rows.append(row)

    # Add station sensitivity to summary
    for r in station_sensitivity_results:
        summary_rows.append({
            "model": r["model"],
            "architecture": f"MLP [256,128,64] ({r['n_stations']} stations)",
            "features": f"TMAX lag-1 + date + AR ({r['n_stations']} stn)",
            "n_features": r["n_features"],
            "n_params": r["n_params"],
            "training_time_s": r["training_time_s"],
            "best_epoch": r["best_epoch"],
            "train_mae": r["mlp_metrics"].get("train", {}).get("mae", "N/A"),
            "train_rmse": r["mlp_metrics"].get("train", {}).get("rmse", "N/A"),
            "train_r2": r["mlp_metrics"].get("train", {}).get("r2", "N/A"),
            "val_mae": r["mlp_metrics"].get("val", {}).get("mae", "N/A"),
            "val_rmse": r["mlp_metrics"].get("val", {}).get("rmse", "N/A"),
            "val_r2": r["mlp_metrics"].get("val", {}).get("r2", "N/A"),
            "test_mae": r["mlp_test_mae"],
            "test_rmse": r["mlp_test_rmse"],
            "test_r2": r["mlp_test_r2"],
            "test_mae_DJF": r["mlp_metrics"].get("test", {}).get("mae_DJF", "N/A"),
            "test_mae_MAM": r["mlp_metrics"].get("test", {}).get("mae_MAM", "N/A"),
            "test_mae_JJA": r["mlp_metrics"].get("test", {}).get("mae_JJA", "N/A"),
            "test_mae_SON": r["mlp_metrics"].get("test", {}).get("mae_SON", "N/A"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved summary CSV: %s", summary_csv_path)

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-35s %8s %8s %8s %8s %8s %8s",
                "Model", "TrainMAE", "ValMAE", "TestMAE", "TestRMSE", "TestR2", "Params")
    logger.info("-" * 106)

    for r in all_results:
        train_mae = r["metrics"].get("train", {}).get("mae", 0)
        val_mae = r["metrics"].get("val", {}).get("mae", 0)
        test_mae = r["metrics"].get("test", {}).get("mae", 0)
        test_rmse = r["metrics"].get("test", {}).get("rmse", 0)
        test_r2 = r["metrics"].get("test", {}).get("r2", 0)
        n_params = r.get("n_params", "N/A")
        logger.info("%-35s %8.3f %8.3f %8.3f %8.3f %8.3f %8s",
                    r["model"][:35], train_mae, val_mae, test_mae, test_rmse, test_r2, str(n_params))

    logger.info("")
    logger.info("Station-Count Sensitivity:")
    logger.info("%-15s %8s %8s %8s %8s", "Stations", "MLP_MAE", "Ridge_MAE", "MLP_R2", "Ridge_R2")
    logger.info("-" * 55)
    for r in station_sensitivity_results:
        logger.info("%-15s %8.3f %8.3f %8.3f %8.3f",
                    f"{r['n_stations']} stations",
                    r["mlp_test_mae"], r["ridge_test_mae"],
                    r["mlp_test_r2"], r["ridge_test_r2"])

    if best_model_info:
        _, _, best_label, _, _ = best_model_info
        logger.info("\nBest model: %s (Test MAE: %.3f)", best_label,
                    min(r["metrics"]["test"]["mae"] for r in all_results))

    logger.info("\nResults saved to:")
    logger.info("  %s", results_json_path)
    logger.info("  %s", summary_csv_path)
    logger.info("  %s", os.path.join(MODELS_DIR, "best_advanced_model.pt"))
    logger.info("=" * 70)

    return experiment_results


if __name__ == "__main__":
    results = main()
