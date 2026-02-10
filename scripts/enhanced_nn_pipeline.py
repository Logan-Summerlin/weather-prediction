#!/usr/bin/env python3
"""
Enhanced Neural Network Pipeline for NYC Temperature Prediction.

Downloads real NOAA GHCN station data, builds enhanced feature matrices
with TMAX+TMIN, delta-T targets, diurnal range, sector gradients, and
trend features, then trains multiple model configurations.

Configurations:
  A: TMAX-only lag-1, raw target, [128,64], Huber
  B: TMAX-only lag-1, delta-T target, [128,64], Huber
  C: TMAX+TMIN lag-1, delta-T target, [128,64], Huber
  D: Full features (TMAX+TMIN+diurnal+gradients+trends), delta-T, [128,64], Huber
  E: Full features, delta-T, [256,128,64], Huber
  F: Full features, delta-T, [128,64], MAE loss
  G: Full features + lag-2 TMAX, delta-T, [256,128,64], Huber

Data splits (chronological, no shuffling):
  IS:  Train 1998-2020, Val 2021-2022, Test 2023-2024
  OOS: Train 1998-2022, Val 2023-2024, Test 2025

All data is REAL from NOAA GHCN-Daily bulk downloads.
"""

import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config_expanded as config
from src.data_collection import download_dly_file, parse_dly_file, pivot_station_data
from src.model import TempPredictorV1

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
TARGET_STATION = config.TARGET_STATION  # USW00094728
DATA_START = "1998-01-01"
DATA_END = "2025-12-31"
MIN_COMPLETENESS = 0.80
MAX_FFILL_DAYS = 3

# IS splits
IS_TRAIN_END = "2020-12-31"
IS_VAL_START = "2021-01-01"
IS_VAL_END = "2022-12-31"
IS_TEST_START = "2023-01-01"
IS_TEST_END = "2024-12-31"

# OOS splits
OOS_TRAIN_END = "2022-12-31"
OOS_VAL_START = "2023-01-01"
OOS_VAL_END = "2024-12-31"
OOS_TEST_START = "2025-01-01"
OOS_TEST_END = "2025-12-31"

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

# Paths
RAW_DIR = config.RAW_DATA_DIR
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "enhanced_pipeline")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Sector definitions for gradient features
SECTOR_WNW = [
    "USW00014777", "USW00014737", "USW00014735", "USW00014757",
    "USW00054789", "USW00014712", "USW00054737",
]
SECTOR_SW = [
    "USW00014734", "USW00014792", "USW00014780", "USW00013739", "USW00054786",
]
SECTOR_COASTAL = [
    "USW00094789", "USW00004781", "USW00054787", "USW00054790",
    "USW00014719", "USW00094702",
]
SECTOR_NEARFIELD = [
    "USW00094741", "USW00054743", "USW00094745", "USW00054734",
]


# ===========================================================================
# Step 1: Download all station .dly files
# ===========================================================================
def download_all_stations():
    """Download .dly files for all stations. Skip if already exists.
    Retry up to 4 times with exponential backoff on failure.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ids = list(config.ALL_STATIONS.keys())
    logger.info("Downloading .dly files for %d stations", len(all_ids))

    success = 0
    failed = []

    for i, sid in enumerate(all_ids, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path):
            file_size = os.path.getsize(dly_path)
            if file_size > 1000:  # Sanity check: file should be non-trivial
                logger.info("[%d/%d] %s already exists (%.1f KB) -- skipping",
                            i, len(all_ids), sid, file_size / 1024)
                success += 1
                continue
            else:
                logger.warning("[%d/%d] %s exists but is tiny (%.0f bytes) -- re-downloading",
                               i, len(all_ids), sid, file_size)

        # Download with retries
        max_retries = 4
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("[%d/%d] Downloading %s (attempt %d/%d)...",
                            i, len(all_ids), sid, attempt, max_retries)
                download_dly_file(sid, RAW_DIR, timeout=180)
                success += 1
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2, 4, 8, 16 seconds
                    logger.warning("  Download failed for %s: %s. Retrying in %ds...",
                                   sid, str(e)[:100], wait)
                    time.sleep(wait)
                else:
                    logger.error("  FAILED to download %s after %d attempts: %s",
                                 sid, max_retries, str(e)[:200])
                    failed.append(sid)

    logger.info("Download complete: %d/%d succeeded, %d failed",
                success, len(all_ids), len(failed))
    if failed:
        logger.warning("Failed stations: %s", failed)
    return failed


# ===========================================================================
# Step 2: Parse all station data (TMAX + TMIN)
# ===========================================================================
def load_all_stations():
    """Parse .dly files for target + all surrounding stations.
    Returns dict of {station_id: DataFrame with date index, TMAX and TMIN columns}.
    """
    all_ids = list(config.ALL_STATIONS.keys())
    station_data = {}

    for sid in all_ids:
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if not os.path.exists(dly_path):
            logger.warning("No .dly file for %s -- skipping", sid)
            continue

        logger.info("Parsing %s ...", sid)
        df_long = parse_dly_file(dly_path, DATA_START, DATA_END)
        if df_long.empty:
            logger.warning("No data for %s in date range -- skipping", sid)
            continue

        df_wide = pivot_station_data(df_long)

        # Need at least TMAX
        if "TMAX" not in df_wide.columns:
            logger.warning("No TMAX for %s -- skipping", sid)
            continue

        # Keep TMAX and TMIN
        cols_to_keep = [c for c in ["TMAX", "TMIN"] if c in df_wide.columns]
        df_out = df_wide[cols_to_keep].copy()
        df_out.index = pd.to_datetime(df_out.index)
        station_data[sid] = df_out

    logger.info("Loaded %d stations with data", len(station_data))
    return station_data


# ===========================================================================
# Step 3: Check data quality
# ===========================================================================
def check_completeness(station_data):
    """Check TMAX completeness for each station over 1998-2024."""
    check_start = pd.Timestamp("1998-01-01")
    check_end = pd.Timestamp("2024-12-31")
    total_days = (check_end - check_start).days + 1

    report = {}
    qualifying = []

    for sid, df in station_data.items():
        mask = (df.index >= check_start) & (df.index <= check_end)
        subset = df.loc[mask]

        n_tmax = subset["TMAX"].notna().sum() if "TMAX" in subset.columns else 0
        n_tmin = subset["TMIN"].notna().sum() if "TMIN" in subset.columns else 0
        completeness = n_tmax / total_days if total_days > 0 else 0.0

        report[sid] = {
            "name": config.ALL_STATIONS.get(sid, "Unknown"),
            "tmax_valid_days": int(n_tmax),
            "tmin_valid_days": int(n_tmin),
            "total_possible": total_days,
            "completeness": round(completeness, 4),
        }

        if completeness >= MIN_COMPLETENESS:
            qualifying.append(sid)
        else:
            logger.warning(
                "DROPPING %s (%s): completeness=%.1f%% < %.0f%%",
                sid, config.ALL_STATIONS.get(sid, "?"),
                completeness * 100, MIN_COMPLETENESS * 100,
            )

    logger.info("Completeness: %d/%d stations qualify (>= %.0f%%)",
                len(qualifying), len(station_data), MIN_COMPLETENESS * 100)
    return qualifying, report


# ===========================================================================
# Step 4: Feature builders for each configuration
# ===========================================================================
def _get_season(month):
    """Return season string for a month number."""
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    else:
        return "SON"


def build_features_config_a(station_data, qualifying_ids):
    """Config A: TMAX-only lag-1, raw target.
    Features: lag-1 TMAX per surrounding station + NYC AR + sin/cos day.
    Target: raw TMAX(t).
    """
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    # Target: raw TMAX
    target_df = station_data[TARGET_STATION]
    master["target"] = target_df["TMAX"]
    master["nyc_tmax_lag1"] = master["target"].shift(1)

    feature_cols = ["nyc_tmax_lag1"]
    for sid in surrounding:
        col = f"{sid}_tmax_lag1"
        master[col] = station_data[sid]["TMAX"].shift(1)
        feature_cols.append(col)

    # Date features
    doy = master.index.dayofyear
    master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    feature_cols.extend(["sin_day", "cos_day"])

    master = master.iloc[1:]  # drop first row (no lag)
    return master, feature_cols, "raw"


def build_features_config_b(station_data, qualifying_ids):
    """Config B: TMAX-only lag-1, delta-T target.
    Features: same as A.
    Target: delta = TMAX(t) - TMAX(t-1).
    """
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    target_df = station_data[TARGET_STATION]
    master["raw_target"] = target_df["TMAX"]
    master["nyc_tmax_lag1"] = master["raw_target"].shift(1)
    # Delta-T: TMAX(t) - TMAX(t-1)
    master["target"] = master["raw_target"] - master["nyc_tmax_lag1"]

    feature_cols = ["nyc_tmax_lag1"]
    for sid in surrounding:
        col = f"{sid}_tmax_lag1"
        master[col] = station_data[sid]["TMAX"].shift(1)
        feature_cols.append(col)

    doy = master.index.dayofyear
    master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    feature_cols.extend(["sin_day", "cos_day"])

    master = master.iloc[1:]
    return master, feature_cols, "delta"


def build_features_config_c(station_data, qualifying_ids):
    """Config C: TMAX+TMIN lag-1, delta-T target.
    Features: lag-1 TMAX + lag-1 TMIN per station + NYC AR + sin/cos.
    Target: delta = TMAX(t) - TMAX(t-1).
    """
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    target_df = station_data[TARGET_STATION]
    master["raw_target"] = target_df["TMAX"]
    master["nyc_tmax_lag1"] = master["raw_target"].shift(1)
    master["target"] = master["raw_target"] - master["nyc_tmax_lag1"]

    feature_cols = ["nyc_tmax_lag1"]

    for sid in surrounding:
        df = station_data[sid]
        col_tmax = f"{sid}_tmax_lag1"
        master[col_tmax] = df["TMAX"].shift(1)
        feature_cols.append(col_tmax)
        if "TMIN" in df.columns:
            col_tmin = f"{sid}_tmin_lag1"
            master[col_tmin] = df["TMIN"].shift(1)
            feature_cols.append(col_tmin)

    # NYC TMIN lag-1
    if "TMIN" in target_df.columns:
        master["nyc_tmin_lag1"] = target_df["TMIN"].shift(1)
        feature_cols.append("nyc_tmin_lag1")

    doy = master.index.dayofyear
    master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    feature_cols.extend(["sin_day", "cos_day"])

    master = master.iloc[1:]
    return master, feature_cols, "delta"


def _add_full_features(master, station_data, qualifying_ids, feature_cols, include_lag2=False):
    """Add diurnal range, sector gradients, and trend features.

    Modifies master in-place and appends to feature_cols.
    """
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]

    # Diurnal range per station: TMAX(t-1) - TMIN(t-1)
    for sid in [TARGET_STATION] + surrounding:
        df = station_data[sid]
        if "TMIN" in df.columns:
            prefix = "nyc" if sid == TARGET_STATION else sid
            col = f"{prefix}_diurnal_lag1"
            master[col] = (df["TMAX"] - df["TMIN"]).shift(1)
            feature_cols.append(col)

    # Sector average TMAX lag-1
    def _sector_avg(sector_ids, tag):
        valid_ids = [s for s in sector_ids if s in qualifying_ids and s in station_data]
        if not valid_ids:
            return
        cols = [f"{s}_tmax_lag1" for s in valid_ids if f"{s}_tmax_lag1" in master.columns]
        if cols:
            master[f"sector_{tag}_tmax_avg"] = master[cols].mean(axis=1)

    _sector_avg(SECTOR_WNW, "wnw")
    _sector_avg(SECTOR_SW, "sw")
    _sector_avg(SECTOR_COASTAL, "coastal")
    _sector_avg(SECTOR_NEARFIELD, "nearfield")

    # Gradient features
    if "sector_wnw_tmax_avg" in master.columns and "sector_coastal_tmax_avg" in master.columns:
        master["grad_wnw_coastal"] = master["sector_wnw_tmax_avg"] - master["sector_coastal_tmax_avg"]
        feature_cols.append("grad_wnw_coastal")
    if "sector_sw_tmax_avg" in master.columns and "sector_wnw_tmax_avg" in master.columns:
        master["grad_sw_wnw"] = master["sector_sw_tmax_avg"] - master["sector_wnw_tmax_avg"]
        feature_cols.append("grad_sw_wnw")
    if "sector_nearfield_tmax_avg" in master.columns and "sector_wnw_tmax_avg" in master.columns:
        master["grad_nearfield_wnw"] = master["sector_nearfield_tmax_avg"] - master["sector_wnw_tmax_avg"]
        feature_cols.append("grad_nearfield_wnw")

    # Trend features: 1-day delta per sector
    for tag in ["wnw", "sw", "coastal", "nearfield"]:
        col = f"sector_{tag}_tmax_avg"
        if col in master.columns:
            trend_col = f"trend_{tag}_1day"
            master[trend_col] = master[col] - master[col].shift(1)
            feature_cols.append(trend_col)

    # Lag-2 TMAX for surrounding stations (Config G)
    if include_lag2:
        for sid in surrounding:
            col = f"{sid}_tmax_lag2"
            master[col] = station_data[sid]["TMAX"].shift(2)
            feature_cols.append(col)
        # NYC lag-2
        master["nyc_tmax_lag2"] = station_data[TARGET_STATION]["TMAX"].shift(2)
        feature_cols.append("nyc_tmax_lag2")


def build_features_config_d(station_data, qualifying_ids):
    """Config D: Full features (TMAX+TMIN+diurnal+gradients+trends), delta-T, [128,64], Huber."""
    # Start from Config C base
    master, feature_cols, target_type = build_features_config_c(station_data, qualifying_ids)
    # Re-create master from scratch to get the full date range before iloc
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master_full = pd.DataFrame(index=date_range)
    master_full.index.name = "date"

    target_df = station_data[TARGET_STATION]
    master_full["raw_target"] = target_df["TMAX"]
    master_full["nyc_tmax_lag1"] = master_full["raw_target"].shift(1)
    master_full["target"] = master_full["raw_target"] - master_full["nyc_tmax_lag1"]

    feature_cols_d = ["nyc_tmax_lag1"]

    for sid in surrounding:
        df = station_data[sid]
        col_tmax = f"{sid}_tmax_lag1"
        master_full[col_tmax] = df["TMAX"].shift(1)
        feature_cols_d.append(col_tmax)
        if "TMIN" in df.columns:
            col_tmin = f"{sid}_tmin_lag1"
            master_full[col_tmin] = df["TMIN"].shift(1)
            feature_cols_d.append(col_tmin)

    if "TMIN" in target_df.columns:
        master_full["nyc_tmin_lag1"] = target_df["TMIN"].shift(1)
        feature_cols_d.append("nyc_tmin_lag1")

    doy = master_full.index.dayofyear
    master_full["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    master_full["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    feature_cols_d.extend(["sin_day", "cos_day"])

    # Add full features
    _add_full_features(master_full, station_data, qualifying_ids, feature_cols_d, include_lag2=False)

    master_full = master_full.iloc[1:]
    return master_full, feature_cols_d, "delta"


def build_features_config_e(station_data, qualifying_ids):
    """Config E: Same features as D, different architecture (handled externally)."""
    return build_features_config_d(station_data, qualifying_ids)


def build_features_config_f(station_data, qualifying_ids):
    """Config F: Same features as D, MAE loss (handled externally)."""
    return build_features_config_d(station_data, qualifying_ids)


def build_features_config_g(station_data, qualifying_ids):
    """Config G: Full features + lag-2 TMAX, delta-T, [256,128,64], Huber."""
    surrounding = [s for s in qualifying_ids if s != TARGET_STATION]
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    target_df = station_data[TARGET_STATION]
    master["raw_target"] = target_df["TMAX"]
    master["nyc_tmax_lag1"] = master["raw_target"].shift(1)
    master["target"] = master["raw_target"] - master["nyc_tmax_lag1"]

    feature_cols = ["nyc_tmax_lag1"]

    for sid in surrounding:
        df = station_data[sid]
        col_tmax = f"{sid}_tmax_lag1"
        master[col_tmax] = df["TMAX"].shift(1)
        feature_cols.append(col_tmax)
        if "TMIN" in df.columns:
            col_tmin = f"{sid}_tmin_lag1"
            master[col_tmin] = df["TMIN"].shift(1)
            feature_cols.append(col_tmin)

    if "TMIN" in target_df.columns:
        master["nyc_tmin_lag1"] = target_df["TMIN"].shift(1)
        feature_cols.append("nyc_tmin_lag1")

    doy = master.index.dayofyear
    master["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    master["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    feature_cols.extend(["sin_day", "cos_day"])

    # Add full features WITH lag-2
    _add_full_features(master, station_data, qualifying_ids, feature_cols, include_lag2=True)

    master = master.iloc[2:]  # Need 2 rows for lag-2
    return master, feature_cols, "delta"


# ===========================================================================
# Step 5: Prepare splits, impute, scale
# ===========================================================================
def prepare_splits(master, feature_cols, target_type,
                   train_end, val_start, val_end, test_start, test_end, label):
    """Create chronological train/val/test splits, impute, scale.

    For delta-T target, also extracts nyc_tmax_lag1 for reconstruction.
    Scaler is fit on training data ONLY.
    """
    idx = master.index

    train_mask = (idx >= pd.Timestamp(DATA_START)) & (idx <= pd.Timestamp(train_end))
    val_mask = (idx >= pd.Timestamp(val_start)) & (idx <= pd.Timestamp(val_end))
    test_mask = (idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(test_end))

    splits_raw = {
        "train": master[train_mask].copy(),
        "val": master[val_mask].copy(),
        "test": master[test_mask].copy(),
    }

    for name, df in splits_raw.items():
        logger.info("  %s %s: %d rows, target non-null: %d",
                     label, name, len(df), df["target"].notna().sum())

    # Forward-fill within each split
    for name in splits_raw:
        splits_raw[name][feature_cols] = splits_raw[name][feature_cols].ffill(limit=MAX_FFILL_DAYS)

    # Training column means for imputation
    train_means = splits_raw["train"][feature_cols].mean()

    # Impute remaining NaN with training means
    for name in splits_raw:
        splits_raw[name][feature_cols] = splits_raw[name][feature_cols].fillna(train_means)

    # Drop rows with NaN target
    for name in list(splits_raw.keys()):
        df = splits_raw[name]
        valid = df["target"].notna()
        n_dropped = (~valid).sum()
        splits_raw[name] = df[valid].copy()
        if n_dropped > 0:
            logger.info("  %s %s: dropped %d NaN-target rows", label, name, n_dropped)

    # Scale features
    scaler = StandardScaler()
    scaler.fit(splits_raw["train"][feature_cols].values)

    result = {}
    for name, df in splits_raw.items():
        X_scaled = scaler.transform(df[feature_cols].values)
        entry = {
            "X": X_scaled,
            "y": df["target"].values,
            "dates": df.index.values,
        }
        if target_type == "delta" and "nyc_tmax_lag1" in df.columns:
            entry["nyc_lag1"] = df["nyc_tmax_lag1"].values
        if "raw_target" in df.columns:
            entry["raw_target"] = df["raw_target"].values
        result[name] = entry

    for name, data in result.items():
        logger.info("  %s %s (final): %d samples, %d features",
                     label, name, len(data["y"]), data["X"].shape[1])

    return result, scaler


# ===========================================================================
# Step 6: Training loop
# ===========================================================================
def train_model(splits, n_features, hidden_sizes, loss_type="huber",
                dropout=0.0, label=""):
    """Train TempPredictorV1 with specified config.

    Returns: model, history dict, device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] Training on %s, features=%d, arch=%s, loss=%s, dropout=%.2f",
                label, device, n_features, hidden_sizes, loss_type, dropout)

    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Loss function
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
    )

    # Tensors
    X_train_t = torch.FloatTensor(splits["train"]["X"]).to(device)
    y_train_t = torch.FloatTensor(splits["train"]["y"]).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(splits["val"]["X"]).to(device)
    y_val_t = torch.FloatTensor(splits["val"]["y"]).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "lr": []}

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
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

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            train_pred_full = model(X_train_t)
            train_mae = torch.mean(torch.abs(train_pred_full - y_train_t)).item()
            val_mae = torch.mean(torch.abs(val_pred - y_val_t)).item()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        if epoch % 50 == 0 or epoch == 1:
            logger.info("[%s] Epoch %3d: train_loss=%.4f val_loss=%.4f "
                        "train_MAE=%.3f val_MAE=%.3f lr=%.6f",
                        label, epoch, avg_train_loss, val_loss,
                        train_mae, val_mae, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("[%s] Early stopping at epoch %d (best: %d)",
                            label, epoch, best_epoch)
                break

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    history["best_epoch"] = best_epoch
    history["total_epochs"] = epoch
    history["elapsed_seconds"] = elapsed
    history["total_params"] = total_params

    logger.info("[%s] Done in %.1fs, best epoch=%d, params=%d",
                label, elapsed, best_epoch, total_params)
    return model, history, device


# ===========================================================================
# Step 7: Evaluate model
# ===========================================================================
def evaluate_model(model, splits, device, target_type, label=""):
    """Compute MAE, RMSE, R^2 on train/val/test, plus seasonal MAE on test.

    For delta-T: reconstructs TMAX as nyc_lag1 + delta_pred.
    """
    metrics = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue

        X = splits[split_name]["X"]
        y_raw = splits[split_name]["y"]

        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            pred_raw = model(X_t).cpu().numpy().flatten()

        # For delta-T, reconstruct TMAX
        if target_type == "delta" and "nyc_lag1" in splits[split_name]:
            nyc_lag1 = splits[split_name]["nyc_lag1"]
            pred_tmax = nyc_lag1 + pred_raw
            if "raw_target" in splits[split_name]:
                actual_tmax = splits[split_name]["raw_target"]
            else:
                actual_tmax = nyc_lag1 + y_raw
        else:
            pred_tmax = pred_raw
            actual_tmax = y_raw

        # Filter NaN
        valid = np.isfinite(pred_tmax) & np.isfinite(actual_tmax)
        pred_tmax = pred_tmax[valid]
        actual_tmax = actual_tmax[valid]

        mae = float(np.mean(np.abs(pred_tmax - actual_tmax)))
        rmse = float(np.sqrt(np.mean((pred_tmax - actual_tmax) ** 2)))
        ss_res = np.sum((pred_tmax - actual_tmax) ** 2)
        ss_tot = np.sum((actual_tmax - np.mean(actual_tmax)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics[split_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "n": int(np.sum(valid)),
        }

        # Seasonal breakdown for test set
        if split_name == "test":
            dates = pd.to_datetime(splits[split_name]["dates"])[valid]
            months = dates.month
            seasonal_mae = {}
            for season, month_set in [("DJF", {12, 1, 2}), ("MAM", {3, 4, 5}),
                                       ("JJA", {6, 7, 8}), ("SON", {9, 10, 11})]:
                mask_s = np.array([m in month_set for m in months])
                if mask_s.sum() > 0:
                    seasonal_mae[season] = round(float(np.mean(np.abs(
                        pred_tmax[mask_s] - actual_tmax[mask_s]))), 4)
                else:
                    seasonal_mae[season] = None
            metrics["seasonal_mae"] = seasonal_mae

        logger.info("[%s] %s: MAE=%.3f, RMSE=%.3f, R2=%.3f (n=%d)",
                    label, split_name, mae, rmse, r2, int(np.sum(valid)))

    if "seasonal_mae" in metrics:
        logger.info("[%s] Seasonal test MAE: %s", label, metrics["seasonal_mae"])

    return metrics


# ===========================================================================
# Step 8: Run a single experiment configuration
# ===========================================================================
def run_experiment(config_name, station_data, qualifying_ids,
                   hidden_sizes, loss_type, dropout, description,
                   build_fn, split_type="IS"):
    """Run a single model configuration end-to-end.

    Returns dict with config, metrics, and training info.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT: %s (%s) -- %s", config_name, split_type, description)
    logger.info("=" * 70)

    # Build features
    master, feature_cols, target_type = build_fn(station_data, qualifying_ids)
    n_features = len(feature_cols)
    logger.info("Features: %d, target type: %s", n_features, target_type)

    # Set split boundaries
    if split_type == "IS":
        train_end, val_start, val_end = IS_TRAIN_END, IS_VAL_START, IS_VAL_END
        test_start, test_end = IS_TEST_START, IS_TEST_END
    else:
        train_end, val_start, val_end = OOS_TRAIN_END, OOS_VAL_START, OOS_VAL_END
        test_start, test_end = OOS_TEST_START, OOS_TEST_END

    label = f"{config_name}_{split_type}"

    # Prepare data
    splits, scaler = prepare_splits(
        master, feature_cols, target_type,
        train_end, val_start, val_end, test_start, test_end, label
    )

    # Train
    model, history, device = train_model(
        splits, n_features, hidden_sizes, loss_type, dropout, label
    )

    # Evaluate
    metrics = evaluate_model(model, splits, device, target_type, label)

    # Save model checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{config_name}_{split_type}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "feature_cols": feature_cols,
        "target_type": target_type,
        "config_name": config_name,
        "split_type": split_type,
    }, model_path)

    result = {
        "config_name": config_name,
        "split_type": split_type,
        "description": description,
        "target_type": target_type,
        "n_features": n_features,
        "hidden_sizes": hidden_sizes,
        "loss_type": loss_type,
        "dropout": dropout,
        "best_epoch": history["best_epoch"],
        "total_epochs": history["total_epochs"],
        "training_time_sec": round(history["elapsed_seconds"], 1),
        "total_params": history["total_params"],
        "metrics": metrics,
        "model_path": model_path,
    }

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    overall_start = time.time()
    logger.info("=" * 70)
    logger.info("ENHANCED NEURAL NETWORK PIPELINE")
    logger.info("Real NOAA GHCN data, multiple configurations, delta-T target")
    logger.info("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ===================================================================
    # PHASE 1: Download all station .dly files
    # ===================================================================
    logger.info("\n--- PHASE 1: Downloading station data ---")
    failed_downloads = download_all_stations()

    # ===================================================================
    # PHASE 2: Parse and quality-check
    # ===================================================================
    logger.info("\n--- PHASE 2: Parsing station data ---")
    station_data = load_all_stations()

    logger.info("\n--- PHASE 2b: Completeness check ---")
    qualifying_ids, completeness_report = check_completeness(station_data)

    # Keep only qualifying
    station_data = {sid: df for sid, df in station_data.items() if sid in qualifying_ids}

    if TARGET_STATION not in qualifying_ids:
        logger.error("Target station %s failed completeness check! Aborting.", TARGET_STATION)
        return

    n_surrounding = len([s for s in qualifying_ids if s != TARGET_STATION])
    logger.info("Qualifying stations: %d total (%d surrounding + 1 target)",
                len(qualifying_ids), n_surrounding)

    # Save completeness report
    completeness_path = os.path.join(RESULTS_DIR, "station_completeness.json")
    with open(completeness_path, "w") as f:
        json.dump(completeness_report, f, indent=2)
    logger.info("Saved completeness report: %s", completeness_path)

    # ===================================================================
    # PHASE 3: Run all experiment configurations
    # ===================================================================
    logger.info("\n--- PHASE 3: Running experiments ---")

    # Define experiment configurations
    experiments = [
        {
            "name": "Config_A",
            "description": "TMAX-only lag-1, raw target, [128,64], Huber",
            "build_fn": build_features_config_a,
            "hidden_sizes": [128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
        {
            "name": "Config_B",
            "description": "TMAX-only lag-1, delta-T target, [128,64], Huber",
            "build_fn": build_features_config_b,
            "hidden_sizes": [128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
        {
            "name": "Config_C",
            "description": "TMAX+TMIN lag-1, delta-T target, [128,64], Huber",
            "build_fn": build_features_config_c,
            "hidden_sizes": [128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
        {
            "name": "Config_D",
            "description": "Full features (TMAX+TMIN+diurnal+gradients+trends), delta-T, [128,64], Huber",
            "build_fn": build_features_config_d,
            "hidden_sizes": [128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
        {
            "name": "Config_E",
            "description": "Full features, delta-T, [256,128,64], Huber",
            "build_fn": build_features_config_e,
            "hidden_sizes": [256, 128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
        {
            "name": "Config_F",
            "description": "Full features, delta-T, [128,64], MAE loss",
            "build_fn": build_features_config_f,
            "hidden_sizes": [128, 64],
            "loss_type": "mae",
            "dropout": 0.0,
        },
        {
            "name": "Config_G",
            "description": "Full features + lag-2 TMAX, delta-T, [256,128,64], Huber",
            "build_fn": build_features_config_g,
            "hidden_sizes": [256, 128, 64],
            "loss_type": "huber",
            "dropout": 0.0,
        },
    ]

    all_results = []

    for exp in experiments:
        # Run IS split
        try:
            result_is = run_experiment(
                config_name=exp["name"],
                station_data=station_data,
                qualifying_ids=qualifying_ids,
                hidden_sizes=exp["hidden_sizes"],
                loss_type=exp["loss_type"],
                dropout=exp["dropout"],
                description=exp["description"],
                build_fn=exp["build_fn"],
                split_type="IS",
            )
            all_results.append(result_is)
        except Exception as e:
            logger.error("FAILED %s IS: %s", exp["name"], traceback.format_exc())
            all_results.append({
                "config_name": exp["name"],
                "split_type": "IS",
                "error": str(e),
            })

        # Run OOS split
        try:
            result_oos = run_experiment(
                config_name=exp["name"],
                station_data=station_data,
                qualifying_ids=qualifying_ids,
                hidden_sizes=exp["hidden_sizes"],
                loss_type=exp["loss_type"],
                dropout=exp["dropout"],
                description=exp["description"],
                build_fn=exp["build_fn"],
                split_type="OOS",
            )
            all_results.append(result_oos)
        except Exception as e:
            logger.error("FAILED %s OOS: %s", exp["name"], traceback.format_exc())
            all_results.append({
                "config_name": exp["name"],
                "split_type": "OOS",
                "error": str(e),
            })

    # ===================================================================
    # PHASE 4: Save comprehensive results
    # ===================================================================
    logger.info("\n--- PHASE 4: Saving results ---")

    results_json = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "enhanced_nn_pipeline.py",
        "data_source": "NOAA GHCN-Daily bulk downloads (REAL DATA)",
        "data_range": f"{DATA_START} to {DATA_END}",
        "target_station": TARGET_STATION,
        "qualifying_stations": len(qualifying_ids),
        "surrounding_stations": n_surrounding,
        "splits": {
            "IS": {
                "train": f"{DATA_START} to {IS_TRAIN_END}",
                "val": f"{IS_VAL_START} to {IS_VAL_END}",
                "test": f"{IS_TEST_START} to {IS_TEST_END}",
            },
            "OOS": {
                "train": f"{DATA_START} to {OOS_TRAIN_END}",
                "val": f"{OOS_VAL_START} to {OOS_VAL_END}",
                "test": f"{OOS_TEST_START} to {OOS_TEST_END}",
            },
        },
        "experiments": all_results,
    }

    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("Saved results: %s", results_path)

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    total_time = time.time() - overall_start
    logger.info("\n" + "=" * 90)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 90)
    logger.info("%-12s %-4s  %-6s  %-8s %-8s %-8s %-8s  %-5s %-6s %-5s",
                "Config", "Split", "Feats", "TrainMAE", "ValMAE", "TestMAE", "TestRMSE",
                "R2", "Epoch", "Time")
    logger.info("-" * 90)

    for r in all_results:
        if "error" in r:
            logger.info("%-12s %-4s  ERROR: %s", r["config_name"], r["split_type"], r["error"][:60])
            continue
        m = r["metrics"]
        logger.info(
            "%-12s %-4s  %-6d  %-8.3f %-8.3f %-8.3f %-8.3f  %-5.3f %-6d %-5.0fs",
            r["config_name"], r["split_type"], r["n_features"],
            m.get("train", {}).get("mae", 0),
            m.get("val", {}).get("mae", 0),
            m.get("test", {}).get("mae", 0),
            m.get("test", {}).get("rmse", 0),
            m.get("test", {}).get("r2", 0),
            r["best_epoch"],
            r["training_time_sec"],
        )

    logger.info("=" * 90)

    # Seasonal breakdown
    logger.info("\nSEASONAL TEST MAE BREAKDOWN (IS split only)")
    logger.info("%-12s  %-8s %-8s %-8s %-8s", "Config", "DJF", "MAM", "JJA", "SON")
    logger.info("-" * 52)
    for r in all_results:
        if "error" in r or r["split_type"] != "IS":
            continue
        sm = r["metrics"].get("seasonal_mae", {})
        logger.info("%-12s  %-8s %-8s %-8s %-8s",
                    r["config_name"],
                    f"{sm.get('DJF', 'N/A'):.3f}" if sm.get("DJF") else "N/A",
                    f"{sm.get('MAM', 'N/A'):.3f}" if sm.get("MAM") else "N/A",
                    f"{sm.get('JJA', 'N/A'):.3f}" if sm.get("JJA") else "N/A",
                    f"{sm.get('SON', 'N/A'):.3f}" if sm.get("SON") else "N/A")

    logger.info("\nTotal pipeline time: %.1f minutes", total_time / 60)
    logger.info("Results saved to: %s", RESULTS_DIR)

    return results_json


if __name__ == "__main__":
    results = main()
