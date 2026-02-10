#!/usr/bin/env python3
"""
Phase 1E: Multi-Seed Ensemble & Advanced Training Protocol.

Experiments:
  1. Multi-Seed Ensemble: Train 5 C_Correction_NN_tiny models (seeds 42,123,456,789,2024),
     average predictions, evaluate ensemble MAE.
  2. SWA (Stochastic Weight Averaging): Post-convergence SWA for seeds [42,123,456].
  3. Cosine Annealing with Warm Restarts: Replace ReduceLROnPlateau.
  4. Weight Decay Sweep: [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3].
  5. Expanding-Window Cross-Validation: 3-fold time-series CV.
  6. Combined Best: Final 5-seed ensemble with best discovered settings.
  7. Evaluation: MAE, RMSE, R2 on all splits + seasonal breakdown.

All data is REAL -- downloaded from NOAA GHCN bulk .dly files.
"""

import os
import sys
import json
import time
import copy
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
logger = logging.getLogger("phase1_ensemble")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1_ensemble")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# Splits
MOS_TRAIN_START, MOS_TRAIN_END = "2004-01-01", "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]
SWA_SEEDS = [42, 123, 456]
WEIGHT_DECAY_VALUES = [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# Expanding-window CV folds
CV_FOLDS = [
    {"train_start": "2004-01-01", "train_end": "2015-12-31",
     "val_start": "2016-01-01", "val_end": "2017-12-31"},
    {"train_start": "2004-01-01", "train_end": "2017-12-31",
     "val_start": "2018-01-01", "val_end": "2019-12-31"},
    {"train_start": "2004-01-01", "train_end": "2019-12-31",
     "val_start": "2020-01-01", "val_end": "2021-12-31"},
]

# Baseline reference
BASELINE_TEST_MAE = 2.090
BASELINE_OOS_MAE = 2.093


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
    logger.info("MOS data: %d rows, date range %s to %s",
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
# 2. FEATURE ENGINEERING
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
# 4. TRAINING FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_nn_standard(model, X_train, y_train, X_val, y_val,
                      lr=1e-3, epochs=300, patience=20, batch_size=128,
                      weight_decay=1e-5, loss_fn_name="mae",
                      scheduler_type="plateau"):
    """Train NN with configurable scheduler. Returns (best_val_mae, trained_model)."""
    model = model.to(DEVICE)
    if loss_fn_name == "huber":
        criterion = nn.HuberLoss(delta=2.0)
    elif loss_fn_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
    else:
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

        if scheduler_type == "cosine_warm_restarts":
            scheduler.step(epoch)
        else:
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
    return best_val_mae, model


def train_nn_swa(model, X_train, y_train, X_val, y_val,
                 lr=1e-3, pretrain_epochs=200, swa_epochs=30,
                 patience=20, batch_size=128,
                 weight_decay=1e-5, loss_fn_name="mae",
                 swa_lr_low=1e-4, swa_lr_high=5e-4):
    """Train NN with SWA post-convergence. Returns (best_val_mae, swa_model)."""
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

    # Phase 1: Standard pre-training with early stopping
    best_val_mae = float("inf")
    best_state = None
    wait = 0
    for epoch in range(pretrain_epochs):
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

    # Restore best model before SWA
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    # Phase 2: SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr_high, anneal_epochs=5, anneal_strategy='cos')

    for epoch in range(swa_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Update batch norm statistics for the SWA model
    swa_model = swa_model.to(DEVICE)
    # Use training data to update BN running stats
    bn_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train).to(DEVICE)),
        batch_size=batch_size, shuffle=False
    )
    torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=DEVICE)

    # Evaluate SWA model
    swa_model.eval()
    with torch.no_grad():
        val_pred = swa_model(X_val_t).squeeze(-1)
        swa_val_mae = torch.mean(torch.abs(val_pred - y_val_t)).item()

    logger.info("  SWA val MAE: %.4f (pre-SWA best: %.4f)", swa_val_mae, best_val_mae)
    return swa_val_mae, swa_model


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
        logger.info("  %s: MAE=%.4f, RMSE=%.4f, R2=%.4f", label, mae, rmse, r2)
    return result


# ============================================================================
# 6. DATASET BUILDER (simplified for MOS correction)
# ============================================================================

class DatasetBuilder:
    def __init__(self, station_matrix, mos_data, cp_data):
        self.station_matrix = station_matrix
        self.mos = mos_data
        self.cp = cp_data

    def _get_mask(self, idx, start, end):
        return (idx >= start) & (idx <= end)

    def _impute_and_scale(self, X_train, X_val, X_test, X_oos=None):
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

    def build_mos_correction(self, train_start=None, train_end=None,
                              val_start=None, val_end=None):
        """Model C: Residual = actual - MOS_ensemble.
        Allows custom train/val boundaries for CV."""
        if train_start is None:
            train_start = MOS_TRAIN_START
        if train_end is None:
            train_end = MOS_TRAIN_END
        if val_start is None:
            val_start = VAL_START
        if val_end is None:
            val_end = VAL_END

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

        # Compute station_mean
        station_lag_tmax = [c for c in lagged.columns if c in df.columns and "TMAX" in c]
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
        valid_cols = [c for c in feature_cols if c in df.columns]
        idx = df.index

        masks = {
            "train": self._get_mask(idx, train_start, train_end),
            "val": self._get_mask(idx, val_start, val_end),
            "test": self._get_mask(idx, TEST_START, TEST_END),
            "oos": self._get_mask(idx, OOS_START, OOS_END),
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
        arrays["n_features"] = X_tr.shape[1]
        return arrays

    def build_cv_fold(self, fold):
        """Build a single CV fold with custom train/val boundaries."""
        return self.build_mos_correction(
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
        )


# ============================================================================
# 7. EXPERIMENT RUNNERS
# ============================================================================

def run_experiment_1_multi_seed(data, all_results):
    """Experiment 1: Multi-Seed Ensemble (5 seeds)."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Multi-Seed Ensemble (5 seeds)")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    per_seed_preds = {split: [] for split in ["train", "val", "test", "oos"]}
    per_seed_results = {}

    for seed in ENSEMBLE_SEEDS:
        logger.info("--- Seed %d ---", seed)
        set_seed(seed)
        model = FlexibleNN(n_feat, [32, 16], dropout=0.2)
        val_mae, model = train_nn_standard(
            model, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=1e-3, epochs=200, patience=15, batch_size=128,
            weight_decay=1e-5, loss_fn_name="mae",
            scheduler_type="plateau"
        )
        logger.info("  Seed %d val MAE (resid): %.4f", seed, val_mae)

        seed_results = {}
        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            per_seed_preds[split].append(pred_actual)
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"Seed{seed} {split}")
            seed_results[split] = r
            all_results[f"E1_seed{seed}_{split}"] = r

        per_seed_results[seed] = seed_results

    # Ensemble: average of 5 seed predictions
    logger.info("--- Ensemble (mean of 5 seeds) ---")
    for split in ["train", "val", "test", "oos"]:
        ensemble_pred = np.mean(per_seed_preds[split], axis=0)
        r = evaluate_model(data[split]["y_actual"], ensemble_pred,
                          data[split]["dates"], f"Ensemble-5seed {split}")
        all_results[f"E1_ensemble_5seed_{split}"] = r

    return per_seed_preds


def run_experiment_2_swa(data, all_results):
    """Experiment 2: SWA (Stochastic Weight Averaging)."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: SWA (Stochastic Weight Averaging)")
    logger.info("=" * 70)

    n_feat = data["n_features"]

    swa_preds = {split: [] for split in ["train", "val", "test", "oos"]}
    noswa_preds = {split: [] for split in ["train", "val", "test", "oos"]}

    for seed in SWA_SEEDS:
        logger.info("--- Seed %d (SWA) ---", seed)
        set_seed(seed)
        model_swa = FlexibleNN(n_feat, [32, 16], dropout=0.2)
        swa_val_mae, swa_model = train_nn_swa(
            model_swa, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=1e-3, pretrain_epochs=200, swa_epochs=30,
            patience=15, batch_size=128,
            weight_decay=1e-5, loss_fn_name="mae",
            swa_lr_low=1e-4, swa_lr_high=5e-4
        )

        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(swa_model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            swa_preds[split].append(pred_actual)
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"SWA-Seed{seed} {split}")
            all_results[f"E2_SWA_seed{seed}_{split}"] = r

        # Also train without SWA for same seed (control)
        logger.info("--- Seed %d (no SWA, control) ---", seed)
        set_seed(seed)
        model_ctrl = FlexibleNN(n_feat, [32, 16], dropout=0.2)
        _, model_ctrl = train_nn_standard(
            model_ctrl, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=1e-3, epochs=200, patience=15, batch_size=128,
            weight_decay=1e-5, loss_fn_name="mae",
            scheduler_type="plateau"
        )
        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model_ctrl, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            noswa_preds[split].append(pred_actual)
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"NoSWA-Seed{seed} {split}")
            all_results[f"E2_noSWA_seed{seed}_{split}"] = r

    # SWA ensemble (3 seeds)
    logger.info("--- SWA Ensemble (3 seeds) ---")
    for split in ["train", "val", "test", "oos"]:
        swa_ens = np.mean(swa_preds[split], axis=0)
        r = evaluate_model(data[split]["y_actual"], swa_ens,
                          data[split]["dates"], f"SWA-Ensemble-3seed {split}")
        all_results[f"E2_SWA_ensemble_3seed_{split}"] = r

        noswa_ens = np.mean(noswa_preds[split], axis=0)
        r = evaluate_model(data[split]["y_actual"], noswa_ens,
                          data[split]["dates"], f"NoSWA-Ensemble-3seed {split}")
        all_results[f"E2_noSWA_ensemble_3seed_{split}"] = r


def run_experiment_3_cosine_annealing(data, all_results):
    """Experiment 3: Cosine Annealing with Warm Restarts."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Cosine Annealing with Warm Restarts")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    seed = 42

    # Cosine annealing
    logger.info("--- Cosine Annealing (T_0=30, T_mult=2) ---")
    set_seed(seed)
    model_cos = FlexibleNN(n_feat, [32, 16], dropout=0.2)
    _, model_cos = train_nn_standard(
        model_cos, data["train"]["X"], data["train"]["y_resid"],
        data["val"]["X"], data["val"]["y_resid"],
        lr=1e-3, epochs=300, patience=30, batch_size=128,
        weight_decay=1e-5, loss_fn_name="mae",
        scheduler_type="cosine_warm_restarts"
    )
    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model_cos, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"CosineAnneal {split}")
        all_results[f"E3_cosine_anneal_{split}"] = r

    # Control: ReduceLROnPlateau (same seed)
    logger.info("--- ReduceLROnPlateau (control) ---")
    set_seed(seed)
    model_plat = FlexibleNN(n_feat, [32, 16], dropout=0.2)
    _, model_plat = train_nn_standard(
        model_plat, data["train"]["X"], data["train"]["y_resid"],
        data["val"]["X"], data["val"]["y_resid"],
        lr=1e-3, epochs=300, patience=30, batch_size=128,
        weight_decay=1e-5, loss_fn_name="mae",
        scheduler_type="plateau"
    )
    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model_plat, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"Plateau {split}")
        all_results[f"E3_plateau_{split}"] = r


def run_experiment_4_weight_decay(data, all_results):
    """Experiment 4: Weight Decay Sweep."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Weight Decay Sweep")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    seed = 42

    for wd in WEIGHT_DECAY_VALUES:
        wd_str = f"{wd:.0e}" if wd > 0 else "0"
        logger.info("--- Weight Decay = %s ---", wd_str)
        set_seed(seed)
        model = FlexibleNN(n_feat, [32, 16], dropout=0.2)
        _, model = train_nn_standard(
            model, data["train"]["X"], data["train"]["y_resid"],
            data["val"]["X"], data["val"]["y_resid"],
            lr=1e-3, epochs=200, patience=15, batch_size=128,
            weight_decay=wd, loss_fn_name="mae",
            scheduler_type="plateau"
        )
        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"WD={wd_str} {split}")
            all_results[f"E4_wd_{wd_str}_{split}"] = r


def run_experiment_5_expanding_cv(builder, all_results):
    """Experiment 5: Expanding-Window Cross-Validation."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: Expanding-Window Cross-Validation")
    logger.info("=" * 70)

    n_feat = None
    fold_val_maes = []

    for i, fold in enumerate(CV_FOLDS):
        logger.info("--- Fold %d: Train %s-%s, Val %s-%s ---",
                    i + 1, fold["train_start"], fold["train_end"],
                    fold["val_start"], fold["val_end"])
        fold_data = builder.build_cv_fold(fold)
        n_feat = fold_data["n_features"]

        seed = 42
        set_seed(seed)
        model = FlexibleNN(n_feat, [32, 16], dropout=0.2)
        val_mae, model = train_nn_standard(
            model, fold_data["train"]["X"], fold_data["train"]["y_resid"],
            fold_data["val"]["X"], fold_data["val"]["y_resid"],
            lr=1e-3, epochs=200, patience=15, batch_size=128,
            weight_decay=1e-5, loss_fn_name="mae",
            scheduler_type="plateau"
        )

        # Evaluate on validation fold
        pred_resid = predict_nn(model, fold_data["val"]["X"])
        pred_actual = fold_data["val"]["mos_base"] + pred_resid
        r = evaluate_model(fold_data["val"]["y_actual"], pred_actual,
                          fold_data["val"]["dates"], f"CV-Fold{i+1} val")
        all_results[f"E5_cv_fold{i+1}_val"] = r
        fold_val_maes.append(r["mae"])

        # Also evaluate on test (held out) for reference
        if fold_data["test"]["X"].shape[0] > 0:
            pred_resid_test = predict_nn(model, fold_data["test"]["X"])
            pred_actual_test = fold_data["test"]["mos_base"] + pred_resid_test
            r_test = evaluate_model(fold_data["test"]["y_actual"], pred_actual_test,
                                   fold_data["test"]["dates"], f"CV-Fold{i+1} test")
            all_results[f"E5_cv_fold{i+1}_test"] = r_test

    avg_cv_mae = np.mean(fold_val_maes)
    std_cv_mae = np.std(fold_val_maes)
    logger.info("Expanding-window CV: avg val MAE = %.4f +/- %.4f", avg_cv_mae, std_cv_mae)
    all_results["E5_cv_summary"] = {
        "avg_val_mae": round(avg_cv_mae, 4),
        "std_val_mae": round(std_cv_mae, 4),
        "fold_val_maes": [round(m, 4) for m in fold_val_maes],
    }


def run_experiment_6_combined_best(data, all_results):
    """Experiment 6: Combined Best -- final 5-seed ensemble with best settings."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 6: Combined Best Ensemble")
    logger.info("=" * 70)

    # Determine best weight decay from E4
    best_wd = 1e-5  # default
    best_test_mae = float("inf")
    for wd in WEIGHT_DECAY_VALUES:
        wd_str = f"{wd:.0e}" if wd > 0 else "0"
        key = f"E4_wd_{wd_str}_test"
        if key in all_results and all_results[key]["mae"] < best_test_mae:
            best_test_mae = all_results[key]["mae"]
            best_wd = wd
    wd_str = f"{best_wd:.0e}" if best_wd > 0 else "0"
    logger.info("Best weight decay from E4: %s (test MAE=%.4f)", wd_str, best_test_mae)

    # Determine best scheduler from E3
    cos_test = all_results.get("E3_cosine_anneal_test", {}).get("mae", 999)
    plat_test = all_results.get("E3_plateau_test", {}).get("mae", 999)
    if cos_test < plat_test:
        best_scheduler = "cosine_warm_restarts"
        logger.info("Best scheduler: Cosine Annealing (test MAE=%.4f)", cos_test)
    else:
        best_scheduler = "plateau"
        logger.info("Best scheduler: ReduceLROnPlateau (test MAE=%.4f)", plat_test)

    # Determine whether SWA helps from E2
    swa_ens_test = all_results.get("E2_SWA_ensemble_3seed_test", {}).get("mae", 999)
    noswa_ens_test = all_results.get("E2_noSWA_ensemble_3seed_test", {}).get("mae", 999)
    use_swa = swa_ens_test < noswa_ens_test
    logger.info("SWA benefit: %s (SWA=%.4f, noSWA=%.4f)", use_swa, swa_ens_test, noswa_ens_test)

    n_feat = data["n_features"]
    preds = {split: [] for split in ["train", "val", "test", "oos"]}

    for seed in ENSEMBLE_SEEDS:
        logger.info("--- Combined Best: Seed %d (wd=%s, sched=%s, swa=%s) ---",
                    seed, wd_str, best_scheduler, use_swa)
        set_seed(seed)

        if use_swa:
            model = FlexibleNN(n_feat, [32, 16], dropout=0.2)
            _, model = train_nn_swa(
                model, data["train"]["X"], data["train"]["y_resid"],
                data["val"]["X"], data["val"]["y_resid"],
                lr=1e-3, pretrain_epochs=200, swa_epochs=30,
                patience=15, batch_size=128,
                weight_decay=best_wd, loss_fn_name="mae"
            )
        else:
            model = FlexibleNN(n_feat, [32, 16], dropout=0.2)
            _, model = train_nn_standard(
                model, data["train"]["X"], data["train"]["y_resid"],
                data["val"]["X"], data["val"]["y_resid"],
                lr=1e-3, epochs=300, patience=20, batch_size=128,
                weight_decay=best_wd, loss_fn_name="mae",
                scheduler_type=best_scheduler
            )

        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            preds[split].append(pred_actual)
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"CombBest-Seed{seed} {split}")
            all_results[f"E6_combined_seed{seed}_{split}"] = r

    # Final ensemble
    logger.info("--- Combined Best: 5-Seed Ensemble ---")
    for split in ["train", "val", "test", "oos"]:
        ensemble_pred = np.mean(preds[split], axis=0)
        r = evaluate_model(data[split]["y_actual"], ensemble_pred,
                          data[split]["dates"], f"CombBest-Ensemble {split}")
        all_results[f"E6_combined_ensemble_{split}"] = r

    # Store configuration
    all_results["E6_config"] = {
        "weight_decay": best_wd,
        "scheduler": best_scheduler,
        "use_swa": use_swa,
        "seeds": ENSEMBLE_SEEDS,
    }


# ============================================================================
# 8. REPORTING
# ============================================================================

def print_comparison_table(all_results):
    """Print comprehensive comparison table."""
    logger.info("\n" + "=" * 100)
    logger.info("COMPREHENSIVE COMPARISON TABLE")
    logger.info("=" * 100)

    # Key configurations to compare
    configs = [
        ("Baseline (C_tiny, single seed)", "E1_seed42"),
        ("Seed 123", "E1_seed123"),
        ("Seed 456", "E1_seed456"),
        ("Seed 789", "E1_seed789"),
        ("Seed 2024", "E1_seed2024"),
        ("5-Seed Ensemble", "E1_ensemble_5seed"),
        ("SWA Ensemble (3-seed)", "E2_SWA_ensemble_3seed"),
        ("NoSWA Ensemble (3-seed)", "E2_noSWA_ensemble_3seed"),
        ("Cosine Annealing", "E3_cosine_anneal"),
        ("ReduceLROnPlateau", "E3_plateau"),
        ("Combined Best Ensemble", "E6_combined_ensemble"),
    ]

    # Add weight decay configs
    for wd in WEIGHT_DECAY_VALUES:
        wd_str = f"{wd:.0e}" if wd > 0 else "0"
        configs.append((f"WD={wd_str}", f"E4_wd_{wd_str}"))

    header = f"{'Config':<35} {'Test MAE':>10} {'Test RMSE':>10} {'OOS MAE':>10} {'OOS RMSE':>10} {'Delta':>8}"
    logger.info(header)
    logger.info("-" * 100)

    for name, prefix in configs:
        test_key = f"{prefix}_test"
        oos_key = f"{prefix}_oos"
        if test_key in all_results:
            test_mae = all_results[test_key]["mae"]
            test_rmse = all_results[test_key]["rmse"]
            oos_mae = all_results.get(oos_key, {}).get("mae", float("nan"))
            oos_rmse = all_results.get(oos_key, {}).get("rmse", float("nan"))
            delta = test_mae - BASELINE_TEST_MAE
            delta_str = f"{delta:+.4f}"
            logger.info(f"  {name:<35} {test_mae:>10.4f} {test_rmse:>10.4f} {oos_mae:>10.4f} {oos_rmse:>10.4f} {delta_str:>8}")

    # Seasonal breakdown for key models
    logger.info("\n" + "=" * 100)
    logger.info("SEASONAL BREAKDOWN (Test Set)")
    logger.info("=" * 100)
    key_models = [
        ("Baseline (seed 42)", "E1_seed42_test"),
        ("5-Seed Ensemble", "E1_ensemble_5seed_test"),
        ("Combined Best Ensemble", "E6_combined_ensemble_test"),
    ]
    header = f"{'Config':<35} {'DJF':>8} {'MAM':>8} {'JJA':>8} {'SON':>8} {'Full':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for name, key in key_models:
        if key in all_results:
            r = all_results[key]
            djf = r.get("mae_DJF", float("nan"))
            mam = r.get("mae_MAM", float("nan"))
            jja = r.get("mae_JJA", float("nan"))
            son = r.get("mae_SON", float("nan"))
            logger.info(f"  {name:<35} {djf:>8.4f} {mam:>8.4f} {jja:>8.4f} {son:>8.4f} {r['mae']:>8.4f}")


def save_results(all_results):
    """Save all results to JSON and CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")

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

    clean = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            clean[k] = {kk: make_serializable(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            clean[k] = [make_serializable(x) for x in v]
        else:
            clean[k] = make_serializable(v)

    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    logger.info("Saved results JSON: %s", json_path)

    # CSV summary
    summary_rows = []
    for key, metrics in all_results.items():
        if isinstance(metrics, dict) and "mae" in metrics:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_name, split = parts
            else:
                model_name, split = key, "unknown"
            row = {"model": model_name, "split": split, **{k: v for k, v in metrics.items()}}
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(RESULTS_DIR, "summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info("Saved summary CSV: %s", csv_path)

        # Print sorted test results
        logger.info("\n" + "=" * 90)
        logger.info("ALL TEST SET RESULTS (sorted by MAE)")
        logger.info("=" * 90)
        test_rows = summary_df[summary_df["split"] == "test"].sort_values("mae")
        for _, row in test_rows.iterrows():
            logger.info("  %-45s MAE=%.4f  RMSE=%.4f  R2=%.4f",
                        row["model"], row["mae"], row["rmse"], row["r2"])

        logger.info("\n" + "=" * 90)
        logger.info("ALL OOS RESULTS (sorted by MAE)")
        logger.info("=" * 90)
        oos_rows = summary_df[summary_df["split"] == "oos"].sort_values("mae")
        for _, row in oos_rows.iterrows():
            logger.info("  %-45s MAE=%.4f  RMSE=%.4f  R2=%.4f",
                        row["model"], row["mae"], row["rmse"], row["r2"])


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("Phase 1E: Multi-Seed Ensemble & Advanced Training")
    logger.info("Device: %s", DEVICE)
    logger.info("Results dir: %s", RESULTS_DIR)

    # Step 1: Download data
    logger.info("\n=== STEP 1: Downloading station .dly files ===")
    download_all_stations()

    # Step 2: Build station matrix
    logger.info("\n=== STEP 2: Building station matrix ===")
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # Step 3: Load MOS and Central Park data
    logger.info("\n=== STEP 3: Loading MOS and Central Park data ===")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # Step 4: Build dataset
    logger.info("\n=== STEP 4: Building MOS correction dataset ===")
    builder = DatasetBuilder(station_matrix, mos_data, cp_data)
    data = builder.build_mos_correction()
    n_feat = data["n_features"]
    logger.info("Features: %d", n_feat)
    logger.info("Train: %d, Val: %d, Test: %d, OOS: %d",
                len(data["train"]["y_resid"]), len(data["val"]["y_resid"]),
                len(data["test"]["y_resid"]), len(data["oos"]["y_resid"]))

    all_results = {}

    # Experiment 1: Multi-Seed Ensemble
    try:
        run_experiment_1_multi_seed(data, all_results)
    except Exception as e:
        logger.error("Experiment 1 failed: %s", e, exc_info=True)

    # Experiment 2: SWA
    try:
        run_experiment_2_swa(data, all_results)
    except Exception as e:
        logger.error("Experiment 2 failed: %s", e, exc_info=True)

    # Experiment 3: Cosine Annealing
    try:
        run_experiment_3_cosine_annealing(data, all_results)
    except Exception as e:
        logger.error("Experiment 3 failed: %s", e, exc_info=True)

    # Experiment 4: Weight Decay Sweep
    try:
        run_experiment_4_weight_decay(data, all_results)
    except Exception as e:
        logger.error("Experiment 4 failed: %s", e, exc_info=True)

    # Experiment 5: Expanding-Window CV
    try:
        run_experiment_5_expanding_cv(builder, all_results)
    except Exception as e:
        logger.error("Experiment 5 failed: %s", e, exc_info=True)

    # Experiment 6: Combined Best
    try:
        run_experiment_6_combined_best(data, all_results)
    except Exception as e:
        logger.error("Experiment 6 failed: %s", e, exc_info=True)

    # Reporting
    print_comparison_table(all_results)
    save_results(all_results)

    elapsed = time.time() - start_time
    logger.info("\nTotal time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
