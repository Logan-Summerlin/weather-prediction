#!/usr/bin/env python3
"""
Architecture Sweep & Hyperparameter Optimization for NYC Temperature Prediction.

Downloads REAL NOAA GHCN weather station data for 53 stations (1 target + 52
surrounding), builds a feature matrix with TMAX/TMIN lag-1, autoregressive
NYC TMAX lag-1, delta-T target, cyclical date features, and diurnal range,
then trains a systematic sweep of feedforward neural network configurations.

RESUMABLE: Saves results after every config. Re-running skips completed work.
CACHED: Feature matrix is cached to .npz for instant reload.

Usage:
    python scripts/architecture_sweep.py prep       # Download + build + cache features
    python scripts/architecture_sweep.py sweep      # Run sweep (resumable)
    python scripts/architecture_sweep.py stage2     # Full train top 5
    python scripts/architecture_sweep.py report     # Print results

Data splits (chronological, NO shuffling):
  Train: 1998-01-01 to 2020-12-31 (~8,400 rows)
  Val:   2021-01-01 to 2022-12-31 (~730 rows)
  Test:  2023-01-01 to 2024-12-31 (~730 rows)

Output:
  results/architecture_sweep/sweep_results.json
  results/architecture_sweep/sweep_summary.csv
"""

import os
import sys
import json
import time
import logging
import hashlib
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

import config_expanded as cfg
from src.data_collection import parse_dly_file

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
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "sweep_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "architecture_sweep")

TARGET_STATION = cfg.TARGET_STATION
ALL_STATIONS = cfg.ALL_STATIONS

START_DATE = "1998-01-01"
END_DATE = "2024-12-31"
TRAIN_END = "2020-12-31"
VAL_START = "2021-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

MIN_COMPLETENESS = 0.80
MAX_FORWARD_FILL_DAYS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep configurations
ARCHITECTURES = [
    [64, 32],
    [128, 64],
    [256, 128],
    [256, 128, 64],
    [512, 256, 128],
    [512, 256, 128, 64],
]
DROPOUTS = [0.0, 0.05, 0.1, 0.2]
LOSSES = ["huber", "mae", "mse"]
LEARNING_RATES = [0.001, 0.0005]
BATCH_SIZES = [64, 128, 256]
BATCH_NORMS = [False, True]

# Stage 1 screening: very aggressive early stopping for speed
STAGE1_MAX_EPOCHS = 25
STAGE1_PATIENCE = 4
STAGE1_SKIP_EPOCH = 8        # Check skip at this epoch
STAGE1_SKIP_THRESHOLD = 6.0  # Skip if val MAE > this at STAGE1_SKIP_EPOCH

# Stage 2 full training
STAGE2_MAX_EPOCHS = 300
STAGE2_PATIENCE = 20
STAGE2_TOP_K = 5


# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

def download_all_stations():
    """Download .dly files for all stations with retry logic."""
    import requests
    os.makedirs(RAW_DIR, exist_ok=True)
    all_ids = list(ALL_STATIONS.keys())
    logger.info("Checking %d station .dly files in %s", len(all_ids), RAW_DIR)

    for i, sid in enumerate(all_ids, 1):
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if os.path.exists(dly_path) and os.path.getsize(dly_path) > 1000:
            continue

        for attempt in range(1, 5):
            try:
                url = f"{cfg.NOAA_BULK_BASE_URL}{sid}.dly"
                logger.info("[%d/%d] Downloading %s (attempt %d)...",
                            i, len(all_ids), sid, attempt)
                resp = requests.get(url, timeout=180, stream=True)
                resp.raise_for_status()
                with open(dly_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                logger.info("  -> %s: %d KB", sid, os.path.getsize(dly_path) // 1024)
                break
            except Exception as e:
                logger.warning("  Attempt %d failed: %s", attempt, e)
                time.sleep(2 ** attempt)
        else:
            logger.error("  FAILED all retries for %s", sid)


# ---------------------------------------------------------------------------
# Data Loading & Feature Engineering (with caching)
# ---------------------------------------------------------------------------

def _load_cached_data():
    """Load cached numpy arrays if they exist."""
    cache_file = os.path.join(CACHE_DIR, "sweep_data.npz")
    meta_file = os.path.join(CACHE_DIR, "sweep_meta.json")
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        logger.info("Loading cached feature data from %s", cache_file)
        d = np.load(cache_file)
        with open(meta_file) as f:
            meta = json.load(f)
        return {
            "X_train": d["X_train"],
            "X_val": d["X_val"],
            "X_test": d["X_test"],
            "y_train_delta": d["y_train_delta"],
            "y_val_delta": d["y_val_delta"],
            "y_test_delta": d["y_test_delta"],
            "y_train_raw": d["y_train_raw"],
            "y_val_raw": d["y_val_raw"],
            "y_test_raw": d["y_test_raw"],
            "nyc_lag1_train": d["nyc_lag1_train"],
            "nyc_lag1_val": d["nyc_lag1_val"],
            "nyc_lag1_test": d["nyc_lag1_test"],
            "feature_names": meta["feature_names"],
            "n_features": meta["n_features"],
            "n_kept_stations": meta["n_kept_stations"],
            "n_dropped_stations": meta["n_dropped_stations"],
        }
    return None


def _save_cached_data(data):
    """Cache numpy arrays to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "sweep_data.npz")
    meta_file = os.path.join(CACHE_DIR, "sweep_meta.json")
    np.savez_compressed(
        cache_file,
        X_train=data["X_train"],
        X_val=data["X_val"],
        X_test=data["X_test"],
        y_train_delta=data["y_train_delta"],
        y_val_delta=data["y_val_delta"],
        y_test_delta=data["y_test_delta"],
        y_train_raw=data["y_train_raw"],
        y_val_raw=data["y_val_raw"],
        y_test_raw=data["y_test_raw"],
        nyc_lag1_train=data["nyc_lag1_train"],
        nyc_lag1_val=data["nyc_lag1_val"],
        nyc_lag1_test=data["nyc_lag1_test"],
    )
    meta = {
        "feature_names": data["feature_names"],
        "n_features": data["n_features"],
        "n_kept_stations": data["n_kept_stations"],
        "n_dropped_stations": data["n_dropped_stations"],
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Cached data to %s (%.1f MB)",
                cache_file, os.path.getsize(cache_file) / 1e6)


def load_and_build_features():
    """Load from cache or build from .dly files. Returns numpy arrays."""
    cached = _load_cached_data()
    if cached is not None:
        return cached

    logger.info("Building feature matrix from .dly files...")

    # Parse all stations
    station_series = {}
    for sid in ALL_STATIONS:
        dly_path = os.path.join(RAW_DIR, f"{sid}.dly")
        if not os.path.exists(dly_path):
            continue
        df = parse_dly_file(dly_path, START_DATE, END_DATE)
        if df.empty:
            continue
        for elem in ["TMAX", "TMIN"]:
            sub = df[df["element"] == elem][["date", "value"]].copy()
            if len(sub) > 0:
                sub["date"] = pd.to_datetime(sub["date"])
                sub = sub.set_index("date")["value"]
                station_series[f"{sid}_{elem}"] = sub

    logger.info("Parsed %d series from stations", len(station_series))

    # Build merged DataFrame
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    merged = pd.DataFrame(index=date_range)
    merged.index.name = "date"
    for name, series in station_series.items():
        merged[name] = series.reindex(merged.index)

    # Filter stations by completeness
    total_days = len(merged)
    surrounding_ids = [sid for sid in ALL_STATIONS if sid != TARGET_STATION]
    kept_stations = []
    dropped_stations = []

    for sid in surrounding_ids:
        tmax_col = f"{sid}_TMAX"
        if tmax_col not in merged.columns:
            dropped_stations.append(sid)
            continue
        completeness = merged[tmax_col].notna().sum() / total_days
        if completeness >= MIN_COMPLETENESS:
            kept_stations.append(sid)
        else:
            dropped_stations.append(sid)
            logger.info("Dropping %s: %.1f%% completeness", sid, completeness * 100)

    logger.info("Kept %d stations, dropped %d", len(kept_stations), len(dropped_stations))

    # Build feature columns
    target_tmax_col = f"{TARGET_STATION}_TMAX"
    nyc_tmax = merged[target_tmax_col].copy()
    nyc_tmax_lag1 = nyc_tmax.shift(1)
    delta_target = nyc_tmax - nyc_tmax_lag1

    feature_frames = []
    feature_names = []

    for sid in kept_stations:
        tmax_col = f"{sid}_TMAX"
        tmin_col = f"{sid}_TMIN"
        if tmax_col in merged.columns:
            s = merged[tmax_col].shift(1)
            s.name = f"{sid}_TMAX_lag1"
            feature_frames.append(s)
            feature_names.append(s.name)
        if tmin_col in merged.columns:
            s = merged[tmin_col].shift(1)
            s.name = f"{sid}_TMIN_lag1"
            feature_frames.append(s)
            feature_names.append(s.name)
        if tmax_col in merged.columns and tmin_col in merged.columns:
            s = (merged[tmax_col] - merged[tmin_col]).shift(1)
            s.name = f"{sid}_diurnal_lag1"
            feature_frames.append(s)
            feature_names.append(s.name)

    # NYC autoregressive
    ar = nyc_tmax_lag1.copy()
    ar.name = f"{TARGET_STATION}_TMAX_lag1"
    feature_frames.append(ar)
    feature_names.append(ar.name)

    # Cyclical date
    doy = merged.index.dayofyear
    sin_day = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=merged.index, name="sin_day")
    cos_day = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=merged.index, name="cos_day")
    feature_frames.extend([sin_day, cos_day])
    feature_names.extend(["sin_day", "cos_day"])

    features = pd.concat(feature_frames, axis=1)

    # Valid mask
    valid_mask = nyc_tmax.notna() & delta_target.notna() & nyc_tmax_lag1.notna()
    valid_mask.iloc[0] = False

    features = features[valid_mask]
    raw_target = nyc_tmax[valid_mask]
    delta_t = delta_target[valid_mask]
    ar_nyc = nyc_tmax_lag1[valid_mask]

    # Forward-fill then split
    features = features.ffill(limit=MAX_FORWARD_FILL_DAYS)

    train_mask = features.index <= TRAIN_END
    val_mask = (features.index >= VAL_START) & (features.index <= VAL_END)
    test_mask = features.index >= TEST_START

    X_train = features[train_mask]
    X_val = features[val_mask]
    X_test = features[test_mask]

    # Impute with train means
    train_means = X_train.mean()
    X_train = X_train.fillna(train_means)
    X_val = X_val.fillna(train_means)
    X_test = X_test.fillna(train_means)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    n_features = X_train_s.shape[1]
    logger.info("Features: %d, Train: %d, Val: %d, Test: %d",
                n_features, len(X_train_s), len(X_val_s), len(X_test_s))

    data = {
        "X_train": X_train_s.astype(np.float32),
        "X_val": X_val_s.astype(np.float32),
        "X_test": X_test_s.astype(np.float32),
        "y_train_delta": delta_t[train_mask].values.astype(np.float32),
        "y_val_delta": delta_t[val_mask].values.astype(np.float32),
        "y_test_delta": delta_t[test_mask].values.astype(np.float32),
        "y_train_raw": raw_target[train_mask].values.astype(np.float32),
        "y_val_raw": raw_target[val_mask].values.astype(np.float32),
        "y_test_raw": raw_target[test_mask].values.astype(np.float32),
        "nyc_lag1_train": ar_nyc[train_mask].values.astype(np.float32),
        "nyc_lag1_val": ar_nyc[val_mask].values.astype(np.float32),
        "nyc_lag1_test": ar_nyc[test_mask].values.astype(np.float32),
        "feature_names": feature_names,
        "n_features": n_features,
        "n_kept_stations": len(kept_stations),
        "n_dropped_stations": len(dropped_stations),
    }

    _save_cached_data(data)
    return data


# ---------------------------------------------------------------------------
# Neural Network Model
# ---------------------------------------------------------------------------

class SweepModel(nn.Module):
    def __init__(self, n_features, hidden_sizes, dropout=0.0, batch_norm=False):
        super().__init__()
        layers = []
        in_dim = n_features
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_loss_fn(loss_name):
    if loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def config_key(cfg_dict):
    """Create a unique hashable key for a config."""
    return json.dumps(cfg_dict, sort_keys=True)


def train_one_config(cfg_dict, tensors, max_epochs, patience,
                     skip_epoch=None, skip_threshold=None):
    """Train a single configuration. Returns result dict or None if skipped."""
    arch = cfg_dict["architecture"]
    dropout = cfg_dict["dropout"]
    loss_name = cfg_dict["loss"]
    lr = cfg_dict["lr"]
    batch_size = cfg_dict["batch_size"]
    batch_norm = cfg_dict["batch_norm"]

    X_train_t, y_train_t = tensors["X_train"], tensors["y_train_delta"]
    X_val_t, y_val_t = tensors["X_val"], tensors["y_val_delta"]
    X_test_t, y_test_t = tensors["X_test"], tensors["y_test_delta"]
    nyc_lag1_val = tensors["nyc_lag1_val"]
    nyc_lag1_test = tensors["nyc_lag1_test"]
    nyc_lag1_train = tensors["nyc_lag1_train"]
    y_raw_val = tensors["y_raw_val"]
    y_raw_test = tensors["y_raw_test"]
    y_raw_train = tensors["y_raw_train"]
    n_features = X_train_t.shape[1]

    model = SweepModel(n_features, arch, dropout, batch_norm).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = make_loss_fn(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched_patience = min(patience - 1, 5) if patience > 2 else 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=sched_patience, factor=0.5
    )

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_state = None
    start_time = time.time()
    last_epoch = 0

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        # Train
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        # Early skip check
        if skip_epoch and epoch == skip_epoch and skip_threshold:
            val_pred_np = val_pred.cpu().numpy().flatten()
            val_tmax = val_pred_np + nyc_lag1_val
            val_mae = float(np.mean(np.abs(y_raw_val - val_tmax)))
            if val_mae > skip_threshold:
                return None  # Skip

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    total_time = time.time() - start_time

    if best_state is None:
        return None

    # Evaluate with best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_pred = model(X_train_t).cpu().numpy().flatten()
        va_pred = model(X_val_t).cpu().numpy().flatten()
        te_pred = model(X_test_t).cpu().numpy().flatten()

    # Reconstruct TMAX
    tr_tmax = tr_pred + nyc_lag1_train
    va_tmax = va_pred + nyc_lag1_val
    te_tmax = te_pred + nyc_lag1_test

    return {
        "architecture": str(arch),
        "dropout": dropout,
        "loss": loss_name,
        "lr": lr,
        "batch_size": batch_size,
        "batch_norm": batch_norm,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "total_epochs": last_epoch,
        "training_time_s": round(total_time, 1),
        "train_mae": round(float(np.mean(np.abs(y_raw_train - tr_tmax))), 4),
        "val_mae": round(float(np.mean(np.abs(y_raw_val - va_tmax))), 4),
        "test_mae": round(float(np.mean(np.abs(y_raw_test - te_tmax))), 4),
        "train_rmse": round(float(np.sqrt(np.mean((y_raw_train - tr_tmax)**2))), 4),
        "val_rmse": round(float(np.sqrt(np.mean((y_raw_val - va_tmax)**2))), 4),
        "test_rmse": round(float(np.sqrt(np.mean((y_raw_test - te_tmax)**2))), 4),
        "train_r2": round(float(r2_score(y_raw_train, tr_tmax)), 4),
        "val_r2": round(float(r2_score(y_raw_val, va_tmax)), 4),
        "test_r2": round(float(r2_score(y_raw_test, te_tmax)), 4),
    }


# ---------------------------------------------------------------------------
# Config Generation
# ---------------------------------------------------------------------------

def generate_all_configs():
    configs = []
    for arch, dropout, loss, lr, bs, bn in itertools.product(
        ARCHITECTURES, DROPOUTS, LOSSES, LEARNING_RATES, BATCH_SIZES, BATCH_NORMS
    ):
        configs.append({
            "architecture": arch,
            "dropout": dropout,
            "loss": loss,
            "lr": lr,
            "batch_size": bs,
            "batch_norm": bn,
        })
    return configs


# ---------------------------------------------------------------------------
# Results Persistence
# ---------------------------------------------------------------------------

def _results_file():
    return os.path.join(RESULTS_DIR, "all_results.json")


def load_existing_results():
    path = _results_file()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _results_file()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def completed_keys(results):
    keys = set()
    for r in results:
        k = json.dumps({
            "architecture": r["architecture"],
            "dropout": r["dropout"],
            "loss": r["loss"],
            "lr": r["lr"],
            "batch_size": r["batch_size"],
            "batch_norm": r["batch_norm"],
        }, sort_keys=True)
        keys.add(k)
    return keys


# ---------------------------------------------------------------------------
# Prepare Tensors (once)
# ---------------------------------------------------------------------------

def prepare_tensors(data):
    """Convert numpy arrays to torch tensors once."""
    return {
        "X_train": torch.from_numpy(data["X_train"]).to(DEVICE),
        "y_train_delta": torch.from_numpy(data["y_train_delta"]).unsqueeze(1).to(DEVICE),
        "X_val": torch.from_numpy(data["X_val"]).to(DEVICE),
        "y_val_delta": torch.from_numpy(data["y_val_delta"]).unsqueeze(1).to(DEVICE),
        "X_test": torch.from_numpy(data["X_test"]).to(DEVICE),
        "y_test_delta": torch.from_numpy(data["y_test_delta"]).unsqueeze(1).to(DEVICE),
        "nyc_lag1_train": data["nyc_lag1_train"],
        "nyc_lag1_val": data["nyc_lag1_val"],
        "nyc_lag1_test": data["nyc_lag1_test"],
        "y_raw_train": data["y_train_raw"],
        "y_raw_val": data["y_val_raw"],
        "y_raw_test": data["y_test_raw"],
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_prep():
    """Download data and cache features."""
    download_all_stations()
    data = load_and_build_features()
    logger.info("Prep complete. %d features, train=%d, val=%d, test=%d",
                data["n_features"],
                len(data["X_train"]), len(data["X_val"]), len(data["X_test"]))
    logger.info("Target (raw) mean=%.1f std=%.1f", data["y_train_raw"].mean(),
                data["y_train_raw"].std())
    logger.info("Target (delta) mean=%.2f std=%.2f", data["y_train_delta"].mean(),
                data["y_train_delta"].std())


def cmd_sweep(time_limit=540):
    """Run sweep configs until time limit (seconds). Resumable."""
    data = load_and_build_features()
    tensors = prepare_tensors(data)

    all_configs = generate_all_configs()
    existing = load_existing_results()
    done_keys = completed_keys(existing)

    remaining = []
    for c in all_configs:
        k = json.dumps(c, sort_keys=True)
        if k not in done_keys:
            remaining.append(c)

    logger.info("Total configs: %d, completed: %d, remaining: %d",
                len(all_configs), len(done_keys), len(remaining))

    if not remaining:
        logger.info("All configs completed!")
        return

    results = list(existing)
    sweep_start = time.time()
    completed_this_run = 0
    skipped_this_run = 0

    for i, cfg_dict in enumerate(remaining):
        if time.time() - sweep_start > time_limit:
            logger.info("Time limit reached (%.0fs). Stopping.", time_limit)
            break

        config_str = (
            f"arch={cfg_dict['architecture']}, drop={cfg_dict['dropout']}, "
            f"loss={cfg_dict['loss']}, lr={cfg_dict['lr']}, "
            f"bs={cfg_dict['batch_size']}, bn={cfg_dict['batch_norm']}"
        )
        logger.info("[%d/%d remaining] %s",
                    i + 1, len(remaining), config_str)

        try:
            result = train_one_config(
                cfg_dict, tensors,
                max_epochs=STAGE1_MAX_EPOCHS,
                patience=STAGE1_PATIENCE,
                skip_epoch=STAGE1_SKIP_EPOCH,
                skip_threshold=STAGE1_SKIP_THRESHOLD,
            )
            if result is None:
                skipped_this_run += 1
                # Save a stub so we don't retry skipped configs
                stub = {
                    "architecture": str(cfg_dict["architecture"]),
                    "dropout": cfg_dict["dropout"],
                    "loss": cfg_dict["loss"],
                    "lr": cfg_dict["lr"],
                    "batch_size": cfg_dict["batch_size"],
                    "batch_norm": cfg_dict["batch_norm"],
                    "skipped": True,
                    "val_mae": 99.0,
                    "test_mae": 99.0,
                }
                results.append(stub)
                logger.info("  -> SKIPPED")
            else:
                result["stage"] = 1
                results.append(result)
                completed_this_run += 1
                logger.info("  -> Val MAE: %.3f, Test MAE: %.3f, Ep: %d, %.0fs",
                            result["val_mae"], result["test_mae"],
                            result["best_epoch"], result["training_time_s"])
        except Exception as e:
            logger.error("  -> ERROR: %s", e)
            # Save error stub
            stub = {
                "architecture": str(cfg_dict["architecture"]),
                "dropout": cfg_dict["dropout"],
                "loss": cfg_dict["loss"],
                "lr": cfg_dict["lr"],
                "batch_size": cfg_dict["batch_size"],
                "batch_norm": cfg_dict["batch_norm"],
                "skipped": True,
                "error": str(e),
                "val_mae": 99.0,
                "test_mae": 99.0,
            }
            results.append(stub)

        # Save after every config
        save_results(results)

    elapsed = time.time() - sweep_start
    logger.info("This run: %d completed, %d skipped in %.0fs (%.1f min)",
                completed_this_run, skipped_this_run, elapsed, elapsed / 60)

    total_done = len([r for r in results if not r.get("skipped")])
    total_skipped = len([r for r in results if r.get("skipped")])
    total_remaining = len(all_configs) - len(results)
    logger.info("Overall: %d completed, %d skipped, %d remaining",
                total_done, total_skipped, total_remaining)


def cmd_stage2():
    """Full training for top configs."""
    data = load_and_build_features()
    tensors = prepare_tensors(data)

    existing = load_existing_results()
    # Filter to non-skipped stage 1 results
    stage1 = [r for r in existing if not r.get("skipped") and r.get("stage") != 2]
    stage1.sort(key=lambda r: r["val_mae"])

    top = stage1[:STAGE2_TOP_K]
    logger.info("Stage 2: Retraining top %d configs with %d epochs, patience %d",
                len(top), STAGE2_MAX_EPOCHS, STAGE2_PATIENCE)

    stage2_results = []
    for i, r in enumerate(top, 1):
        cfg_dict = {
            "architecture": eval(r["architecture"]) if isinstance(r["architecture"], str) else r["architecture"],
            "dropout": r["dropout"],
            "loss": r["loss"],
            "lr": r["lr"],
            "batch_size": r["batch_size"],
            "batch_norm": r["batch_norm"],
        }
        logger.info("[Stage2 %d/%d] %s", i, len(top), cfg_dict)

        result = train_one_config(
            cfg_dict, tensors,
            max_epochs=STAGE2_MAX_EPOCHS,
            patience=STAGE2_PATIENCE,
        )
        if result:
            result["stage"] = 2
            stage2_results.append(result)
            logger.info("  -> Val MAE: %.4f, Test MAE: %.4f, Ep: %d, %.0fs",
                        result["val_mae"], result["test_mae"],
                        result["best_epoch"], result["training_time_s"])

    # Add stage 2 to existing results
    all_results = existing + stage2_results
    save_results(all_results)
    logger.info("Stage 2 complete. %d configs retrained.", len(stage2_results))


def cmd_report():
    """Generate final report and summary files."""
    existing = load_existing_results()
    real = [r for r in existing if not r.get("skipped")]

    if not real:
        logger.error("No completed results found!")
        return

    # Save sweep_results.json
    data = _load_cached_data()
    meta = {}
    if data:
        meta = {
            "n_features": data["n_features"],
            "n_kept_stations": data["n_kept_stations"],
            "n_dropped_stations": data["n_dropped_stations"],
            "feature_names": data["feature_names"],
        }

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "train_period": f"{START_DATE} to {TRAIN_END}",
            "val_period": f"{VAL_START} to {VAL_END}",
            "test_period": f"{TEST_START} to {TEST_END}",
            "total_configs": len(generate_all_configs()),
            "completed_configs": len(real),
            "skipped_configs": len([r for r in existing if r.get("skipped")]),
            **meta,
        },
        "results": real,
    }

    json_path = os.path.join(RESULTS_DIR, "sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved %s", json_path)

    # CSV summary
    df = pd.DataFrame(real)
    csv_path = os.path.join(RESULTS_DIR, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    # Print top 10
    sorted_by_test = sorted(real, key=lambda r: r["test_mae"])
    sorted_by_val = sorted(real, key=lambda r: r["val_mae"])

    print("\n" + "=" * 120)
    print(f"TOP 10 CONFIGURATIONS BY TEST MAE  ({len(real)} completed)")
    print("=" * 120)
    _print_table(sorted_by_test[:10])

    print(f"\nTOP 10 CONFIGURATIONS BY VAL MAE")
    print("=" * 120)
    _print_table(sorted_by_val[:10])

    # Summary
    test_maes = [r["test_mae"] for r in real]
    val_maes = [r["val_mae"] for r in real]
    print(f"\nSummary ({len(real)} configs):")
    print(f"  Test MAE — min: {min(test_maes):.4f}, median: {np.median(test_maes):.4f}, max: {max(test_maes):.4f}")
    print(f"  Val  MAE — min: {min(val_maes):.4f}, median: {np.median(val_maes):.4f}, max: {max(val_maes):.4f}")

    best = sorted_by_test[0]
    print(f"\nBEST CONFIG (by test MAE):")
    print(f"  Architecture: {best['architecture']}")
    print(f"  Dropout: {best['dropout']}")
    print(f"  Loss: {best['loss']}")
    print(f"  LR: {best['lr']}, Batch Size: {best['batch_size']}, BatchNorm: {best['batch_norm']}")
    print(f"  Params: {best['n_params']:,}")
    print(f"  Train MAE: {best['train_mae']:.4f}  Val MAE: {best['val_mae']:.4f}  Test MAE: {best['test_mae']:.4f}")
    print(f"  Test RMSE: {best['test_rmse']:.4f}  Test R2: {best['test_r2']:.4f}")
    print(f"  Best Epoch: {best['best_epoch']}, Stage: {best.get('stage', 1)}")


def _print_table(results):
    print(f"{'#':>3} {'Architecture':<22} {'Drop':>5} {'Loss':>5} {'LR':>7} "
          f"{'BS':>4} {'BN':>3} {'Stg':>3} {'Params':>8} "
          f"{'TrMAE':>7} {'VaMAE':>7} {'TeMAE':>7} {'TeRMSE':>8} {'TeR2':>7} "
          f"{'Ep':>4} {'Time':>5}")
    print("-" * 120)
    for i, r in enumerate(results, 1):
        print(
            f"{i:>3} {r['architecture']:<22} {r['dropout']:>5.2f} {r['loss']:>5} "
            f"{r['lr']:>7.4f} {r['batch_size']:>4} {'Y' if r['batch_norm'] else 'N':>3} "
            f"{r.get('stage', 1):>3} {r['n_params']:>8,} "
            f"{r['train_mae']:>7.3f} {r['val_mae']:>7.3f} {r['test_mae']:>7.3f} "
            f"{r['test_rmse']:>8.3f} {r['test_r2']:>7.4f} "
            f"{r['best_epoch']:>4} {r['training_time_s']:>4.0f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        # Default: run prep + sweep + stage2 + report
        print("Usage: python architecture_sweep.py [prep|sweep|stage2|report|all]")
        print("  prep   — Download data, build features, cache")
        print("  sweep  — Run sweep batch (resumable, ~9 min)")
        print("  stage2 — Full training on top 5")
        print("  report — Print/save results")
        print("  all    — Run everything sequentially")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "prep":
        cmd_prep()
    elif cmd == "sweep":
        time_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 540
        cmd_sweep(time_limit=time_limit)
    elif cmd == "stage2":
        cmd_stage2()
    elif cmd == "report":
        cmd_report()
    elif cmd == "all":
        cmd_prep()
        # Run sweep in a loop until all done
        all_configs = generate_all_configs()
        for batch in range(100):  # Safety limit
            existing = load_existing_results()
            done = completed_keys(existing)
            remaining = sum(1 for c in all_configs
                          if json.dumps(c, sort_keys=True) not in done)
            if remaining == 0:
                break
            logger.info("=== Sweep batch %d: %d remaining ===", batch + 1, remaining)
            cmd_sweep(time_limit=540)
        cmd_stage2()
        cmd_report()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
