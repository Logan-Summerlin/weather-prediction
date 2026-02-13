#!/usr/bin/env python3
"""
Phase 1 Combined Best: Synthesis of all analyst findings.

Combines the BEST features and model findings from all 4 Phase 1 analysts:
  Analyst 1 (Features): MOS error memory + MOS x Station interactions
  Analyst 2 (Probabilistic): NLL -> CRPS two-stage training
  Analyst 3 (Ensemble): 5-seed ensemble, wd=1e-4, ReduceLROnPlateau
  Analyst 4 (Architecture): Enhanced temporal/spatial features, ResidualNN

Models trained with combined feature set:
  A: NN [64,32] MAE loss (point prediction, Analyst 1's best)
  B: NN [64,32,16] MAE loss (3-layer from Analyst 4)
  C: ResidualNN [64,32] MAE loss (from Analyst 4)
  D: NN [64,32] probabilistic NLL -> CRPS (from Analyst 2)
  E: 5-seed ensemble of A (seeds: 42, 123, 456, 789, 2024)
  F: Ridge regression (linear comparison)
  G: HistGradientBoosting (non-NN comparison)

All data is REAL -- downloaded from NOAA GHCN bulk .dly files.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
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
from src.crps_loss import GaussianCRPSLoss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger("phase1_combined_best")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_STATION = "USW00094728"
ALL_SURROUNDING = list(SURROUNDING_STATIONS.keys())
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
CP_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase1_combined")

DLY_START = "1998-01-01"
DLY_END = "2025-12-31"

# Chronological splits
MOS_TRAIN_START, MOS_TRAIN_END = "2004-01-01", "2020-12-31"
VAL_START, VAL_END = "2021-01-01", "2022-12-31"
TEST_START, TEST_END = "2023-01-01", "2024-12-31"
OOS_START, OOS_END = "2025-01-01", "2025-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_SEED = 42

# Analyst 3 findings: wd=1e-4, ReduceLROnPlateau optimal
DEFAULT_WD = 1e-4

# Analyst 2 findings: sigma floor for probabilistic models
SIGMA_FLOOR = 0.75
SIGMA_CAP = 10.0

NYC_LAT = 40.7831  # Central Park latitude


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 1. DATA LOADING (from mos_ensemble_pipeline)
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


# --- Phase 1A: MOS Error Memory Features (Analyst 1) ---

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


# --- Phase 1C: MOS x Station Interaction Features (Analyst 1) ---

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


# --- Enhanced Temporal Features (Analyst 4) ---

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


def add_enhanced_temporal_features(df, cp_data, train_end=MOS_TRAIN_END):
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


# --- Enhanced Spatial Features (Analyst 4) ---

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


class ResidualCorrectionNN(nn.Module):
    """NN with residual (skip) connection from input to first hidden layer."""
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


class ProbabilisticCorrectionNN(nn.Module):
    """Heteroscedastic Gaussian correction model with mu and sigma heads."""
    def __init__(self, n_features, hidden_sizes, dropout=0.2, use_batchnorm=True):
        super().__init__()
        self.backbone = nn.ModuleList()
        in_dim = n_features
        for h in hidden_sizes:
            block = nn.ModuleList()
            block.append(nn.Linear(in_dim, h))
            if use_batchnorm and h >= 16:
                block.append(nn.BatchNorm1d(h))
            block.append(nn.ReLU())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.backbone.append(block)
            in_dim = h
        self.mu_head = nn.Linear(in_dim, 1)
        self.log_sigma_head = nn.Linear(in_dim, 1)
        nn.init.constant_(self.log_sigma_head.bias, 0.5)
        nn.init.zeros_(self.log_sigma_head.weight)

    def forward(self, x):
        h = x
        for block in self.backbone:
            for layer in block:
                h = layer(h)
        mu = self.mu_head(h)
        raw_log_sigma = self.log_sigma_head(h)
        sigma = F.softplus(raw_log_sigma) + SIGMA_FLOOR
        sigma = torch.clamp(sigma, max=SIGMA_CAP)
        return mu, sigma


# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, target):
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1).clamp(min=1e-6)
        target = target.reshape(-1)
        nll = 0.5 * torch.log(2 * math.pi * sigma ** 2) + \
              (target - mu) ** 2 / (2 * sigma ** 2)
        return nll.mean()


def train_nn(model, X_train, y_train, X_val, y_val,
             lr=1e-3, epochs=300, patience=20, batch_size=128,
             loss_fn_name="mae", weight_decay=DEFAULT_WD):
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


def train_probabilistic_nll(model, X_train, y_train, X_val, y_val,
                            lr=1e-3, epochs=200, patience=20, batch_size=128):
    """Stage 1: Train with Gaussian NLL loss."""
    model = model.to(DEVICE)
    criterion = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULT_WD)
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

    best_val_nll = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = criterion(mu, sigma, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(X_val_t)
            val_nll = criterion(val_mu, val_sigma, y_val_t).item()

        scheduler.step(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    logger.info("  NLL training complete. Best val NLL=%.4f", best_val_nll)
    return best_val_nll


def train_probabilistic_crps(model, X_train, y_train, X_val, y_val,
                             lr=5e-4, epochs=50, patience=10, batch_size=128):
    """Stage 2: Fine-tune with Gaussian CRPS loss."""
    model = model.to(DEVICE)
    criterion = GaussianCRPSLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DEFAULT_WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    best_val_crps = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = criterion(mu.squeeze(-1), sigma.squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma = model(X_val_t)
            val_crps = criterion(val_mu.squeeze(-1), val_sigma.squeeze(-1), y_val_t).item()

        scheduler.step(val_crps)

        if val_crps < best_val_crps:
            best_val_crps = val_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    logger.info("  CRPS fine-tune complete. Best val CRPS=%.4f", best_val_crps)
    return best_val_crps


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        preds = model(X_t).squeeze(-1).cpu().numpy()
    return preds


def predict_probabilistic(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        mu, sigma = model(X_t)
        mu_np = mu.squeeze(-1).cpu().numpy()
        sigma_np = sigma.squeeze(-1).cpu().numpy()
    return mu_np, sigma_np


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


def compute_crps_numpy(y_true, mu, sigma):
    z = (y_true - mu) / np.maximum(sigma, 1e-6)
    phi_z = scipy_norm.pdf(z)
    Phi_z = scipy_norm.cdf(z)
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / math.sqrt(math.pi))
    return float(np.mean(crps))


def compute_coverage(y_true, mu, sigma, level):
    alpha = 1 - level
    z_low = scipy_norm.ppf(alpha / 2)
    z_high = scipy_norm.ppf(1 - alpha / 2)
    lower = mu + z_low * sigma
    upper = mu + z_high * sigma
    inside = np.logical_and(y_true >= lower, y_true <= upper)
    coverage = float(np.mean(inside))
    mean_width = float(np.mean(upper - lower))
    return coverage, mean_width


def evaluate_probabilistic(y_true, mu, sigma, mos_base, dates, label=""):
    y_pred = mos_base + mu
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    y_resid = y_true - mos_base
    crps = compute_crps_numpy(y_resid, mu, sigma)

    coverages = {}
    widths = {}
    for level in [0.50, 0.80, 0.90, 0.95]:
        cov, width = compute_coverage(y_resid, mu, sigma, level)
        coverages[f"cov_{int(level*100)}"] = round(cov, 4)
        widths[f"width_{int(level*100)}"] = round(width, 2)

    mean_sigma = float(np.mean(sigma))

    seasonal = {}
    if dates is not None:
        seasons = assign_season(pd.DatetimeIndex(dates))
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = (seasons == s).values if hasattr(seasons, 'values') else (seasons == s)
            if np.sum(mask) > 0:
                s_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
                seasonal[f"mae_{s}"] = round(s_mae, 4)

    result = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "crps": round(crps, 4),
        "mean_sigma": round(mean_sigma, 4),
        **coverages,
        **widths,
        **seasonal,
    }

    if label:
        logger.info("  %s: MAE=%.3f  CRPS=%.3f  sigma=%.2f  cov90=%.1f%%  cov95=%.1f%%",
                     label, mae, crps, mean_sigma,
                     coverages.get("cov_90", 0) * 100,
                     coverages.get("cov_95", 0) * 100)
    return result


# ============================================================================
# 6. COMBINED DATASET BUILDER
# ============================================================================

class CombinedDatasetBuilder:
    """Builds MOS-correction datasets with ALL combined features."""

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

        # Phase 1A: MOS Error Memory
        df = add_mos_error_memory_features(df)

        # Phase 1C: MOS x Station Interaction
        df = add_mos_station_interaction_features(df, station_lag_tmax)

        # Enhanced Temporal (Analyst 4)
        df, temporal_cols = add_enhanced_temporal_features(df, self.cp, train_end=MOS_TRAIN_END)

        # Enhanced Spatial (Analyst 4)
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
        logger.info("Feature groups: base=%d, 1A=%d, 1C=%d, temporal=%d, spatial=%d",
                     len(base_features),
                     len([f for f in PHASE_1A_FEATURES if f in df.columns]),
                     len([f for f in PHASE_1C_FEATURES if f in df.columns]),
                     len([f for f in TEMPORAL_FEATURES if f in df.columns]),
                     len([f for f in SPATIAL_FEATURES if f in df.columns]))

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
# 7. MODEL RUNNERS
# ============================================================================

def run_model_a(data, all_results):
    """Model A: NN [64,32] with MAE loss (Analyst 1's best)."""
    logger.info("=" * 70)
    logger.info("MODEL A: NN [64,32] MAE loss")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    set_seed(DEFAULT_SEED)
    model = FlexibleNN(n_feat, [64, 32], dropout=0.15)
    train_nn(model, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber",
             weight_decay=DEFAULT_WD)

    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"A_NN64_32 {split}")
        all_results[f"A_NN_64_32_{split}"] = r

    return model


def run_model_b(data, all_results):
    """Model B: NN [64,32,16] with MAE loss (3-layer, Analyst 4)."""
    logger.info("=" * 70)
    logger.info("MODEL B: NN [64,32,16] MAE loss")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    set_seed(DEFAULT_SEED)
    model = FlexibleNN(n_feat, [64, 32, 16], dropout=0.1)
    train_nn(model, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber",
             weight_decay=DEFAULT_WD)

    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"B_NN64_32_16 {split}")
        all_results[f"B_NN_64_32_16_{split}"] = r

    return model


def run_model_c(data, all_results):
    """Model C: ResidualNN [64,32] with MAE loss (Analyst 4)."""
    logger.info("=" * 70)
    logger.info("MODEL C: ResidualNN [64,32] MAE loss")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    set_seed(DEFAULT_SEED)
    model = ResidualCorrectionNN(n_feat, dropout=0.1)
    train_nn(model, data["train"]["X"], data["train"]["y_resid"],
             data["val"]["X"], data["val"]["y_resid"],
             lr=1e-3, epochs=300, patience=20, loss_fn_name="huber",
             weight_decay=DEFAULT_WD)

    for split in ["train", "val", "test", "oos"]:
        pred_resid = predict_nn(model, data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"C_ResidualNN {split}")
        all_results[f"C_ResidualNN_{split}"] = r

    return model


def run_model_d(data, all_results):
    """Model D: NN [64,32] probabilistic with NLL -> CRPS (Analyst 2)."""
    logger.info("=" * 70)
    logger.info("MODEL D: Probabilistic NN [64,32] NLL -> CRPS")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    set_seed(DEFAULT_SEED)
    model = ProbabilisticCorrectionNN(n_feat, [64, 32], dropout=0.2)

    # Stage 1: NLL
    train_probabilistic_nll(model, data["train"]["X"], data["train"]["y_resid"],
                            data["val"]["X"], data["val"]["y_resid"],
                            lr=1e-3, epochs=200, patience=20)
    # Stage 2: CRPS fine-tune
    train_probabilistic_crps(model, data["train"]["X"], data["train"]["y_resid"],
                             data["val"]["X"], data["val"]["y_resid"],
                             lr=5e-4, epochs=50, patience=10)

    for split in ["train", "val", "test", "oos"]:
        mu, sigma = predict_probabilistic(model, data[split]["X"])
        r = evaluate_probabilistic(data[split]["y_actual"], mu, sigma,
                                   data[split]["mos_base"], data[split]["dates"],
                                   f"D_Probabilistic {split}")
        all_results[f"D_Probabilistic_{split}"] = r

    return model


def run_model_e(data, all_results):
    """Model E: 5-seed ensemble of Model A (Analyst 3)."""
    logger.info("=" * 70)
    logger.info("MODEL E: 5-seed Ensemble of NN [64,32]")
    logger.info("=" * 70)

    n_feat = data["n_features"]
    per_seed_preds = {split: [] for split in ["train", "val", "test", "oos"]}
    seed_state_dicts = {}

    for seed in ENSEMBLE_SEEDS:
        logger.info("--- Seed %d ---", seed)
        set_seed(seed)
        model = FlexibleNN(n_feat, [64, 32], dropout=0.15)
        train_nn(model, data["train"]["X"], data["train"]["y_resid"],
                 data["val"]["X"], data["val"]["y_resid"],
                 lr=1e-3, epochs=300, patience=20, loss_fn_name="huber",
                 weight_decay=DEFAULT_WD)
        seed_state_dicts[str(seed)] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        for split in ["train", "val", "test", "oos"]:
            pred_resid = predict_nn(model, data[split]["X"])
            pred_actual = data[split]["mos_base"] + pred_resid
            per_seed_preds[split].append(pred_actual)
            r = evaluate_model(data[split]["y_actual"], pred_actual,
                              data[split]["dates"], f"E_seed{seed} {split}")
            all_results[f"E_seed_{seed}_{split}"] = r

    # Ensemble: average of 5 seed predictions
    logger.info("--- 5-Seed Ensemble (mean) ---")
    for split in ["train", "val", "test", "oos"]:
        ensemble_pred = np.mean(per_seed_preds[split], axis=0)
        r = evaluate_model(data[split]["y_actual"], ensemble_pred,
                          data[split]["dates"], f"E_Ensemble {split}")
        all_results[f"E_Ensemble_5seed_{split}"] = r

    return {
        "per_seed_preds": per_seed_preds,
        "seed_state_dicts": seed_state_dicts,
    }


def _build_benchmark_prediction_files(data, ensemble_result):
    """Create benchmark-ready daily prediction files from the 5-seed ensemble."""
    per_seed_preds = ensemble_result["per_seed_preds"]

    # Estimate month-specific sigma from IS residuals (2023-2024 test window).
    test_dates = pd.to_datetime(data["test"]["dates"])
    test_actual = data["test"]["y_actual"]
    test_pred = np.mean(per_seed_preds["test"], axis=0)
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
        mu = np.mean(per_seed_preds[split], axis=0)
        sigma = [sigma_by_month[int(m)] for m in dates.month]
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "model_mu": mu,
            "model_sigma": sigma,
            "actual_tmax": data[split]["y_actual"],
        })

    pred_is = _make_split_df("test")
    pred_oos = _make_split_df("oos")

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    is_path = os.path.join(PROJECT_ROOT, "data", "best_model_predictions_2023_2024.csv")
    oos_path = os.path.join(PROJECT_ROOT, "data", "best_model_predictions_2025.csv")
    pred_is.to_csv(is_path, index=False)
    pred_oos.to_csv(oos_path, index=False)
    logger.info("Saved benchmark prediction files: %s, %s", is_path, oos_path)

    sigma_path = os.path.join(RESULTS_DIR, "best_model_sigma_by_month.json")
    with open(sigma_path, "w") as f:
        json.dump({str(k): v for k, v in sigma_by_month.items()}, f, indent=2)
    logger.info("Saved month-wise sigma estimates to %s", sigma_path)


def run_model_f(data, all_results):
    """Model F: Ridge regression (linear comparison)."""
    logger.info("=" * 70)
    logger.info("MODEL F: Ridge Regression")
    logger.info("=" * 70)

    ridge = Ridge(alpha=1.0)
    ridge.fit(data["train"]["X"], data["train"]["y_resid"])

    for split in ["train", "val", "test", "oos"]:
        pred_resid = ridge.predict(data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"F_Ridge {split}")
        all_results[f"F_Ridge_{split}"] = r

    # Feature importance from Ridge coefficients
    feat_names = data.get("feature_names", [])
    if feat_names:
        coef_pairs = sorted(zip(feat_names, np.abs(ridge.coef_)), key=lambda x: -x[1])
        all_results["F_Ridge_top20_features"] = [
            {"feature": f, "abs_coef": round(float(c), 4)}
            for f, c in coef_pairs[:20]
        ]

    return ridge


def run_model_g(data, all_results):
    """Model G: HistGradientBoosting (non-NN comparison)."""
    logger.info("=" * 70)
    logger.info("MODEL G: HistGradientBoosting")
    logger.info("=" * 70)

    hgb = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=5,
        learning_rate=0.03,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=DEFAULT_SEED,
    )
    hgb.fit(data["train"]["X"], data["train"]["y_resid"])

    for split in ["train", "val", "test", "oos"]:
        pred_resid = hgb.predict(data[split]["X"])
        pred_actual = data[split]["mos_base"] + pred_resid
        r = evaluate_model(data[split]["y_actual"], pred_actual,
                          data[split]["dates"], f"G_HGB {split}")
        all_results[f"G_HGB_{split}"] = r

    # Feature importance
    feat_names = data.get("feature_names", [])
    if hasattr(hgb, "feature_importances_") and feat_names:
        importances = hgb.feature_importances_
        fi_pairs = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        all_results["G_HGB_top20_features"] = [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in fi_pairs[:20]
        ]
        logger.info("Top 10 HGB features:")
        for f, imp in fi_pairs[:10]:
            logger.info("  %-50s %.4f", f, imp)

    return hgb


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 1 COMBINED BEST: Synthesis of All Analyst Findings")
    logger.info("=" * 80)
    logger.info("Device: %s", DEVICE)
    logger.info("Default weight decay: %s (Analyst 3)", DEFAULT_WD)
    logger.info("Ensemble seeds: %s (Analyst 3)", ENSEMBLE_SEEDS)

    # ---- Step 1: Download station data ----
    logger.info("\n=== STEP 1: Downloading station .dly files ===")
    download_all_stations()

    # ---- Step 2: Build station matrix ----
    logger.info("\n=== STEP 2: Building station observation matrix ===")
    station_matrix = build_station_matrix(DLY_START, DLY_END, include_tmin=True)

    # ---- Step 3: Load MOS and Central Park data ----
    logger.info("\n=== STEP 3: Loading MOS and Central Park data ===")
    mos_data = load_mos_data()
    cp_data = load_central_park_tmax()

    # ---- Step 4: Build combined feature dataset ----
    logger.info("\n=== STEP 4: Building combined feature dataset ===")
    builder = CombinedDatasetBuilder(station_matrix, mos_data, cp_data)
    data = builder.build_correction_data()

    # ---- Step 5: Run all models ----
    all_results = {}
    all_models = {}

    # Model A: NN [64,32]
    try:
        model_a = run_model_a(data, all_results)
        all_models["A"] = model_a
    except Exception as e:
        logger.error("Model A failed: %s", e, exc_info=True)

    # Model B: NN [64,32,16]
    try:
        model_b = run_model_b(data, all_results)
        all_models["B"] = model_b
    except Exception as e:
        logger.error("Model B failed: %s", e, exc_info=True)

    # Model C: ResidualNN
    try:
        model_c = run_model_c(data, all_results)
        all_models["C"] = model_c
    except Exception as e:
        logger.error("Model C failed: %s", e, exc_info=True)

    # Model D: Probabilistic
    try:
        model_d = run_model_d(data, all_results)
        all_models["D"] = model_d
    except Exception as e:
        logger.error("Model D failed: %s", e, exc_info=True)

    # Model E: 5-seed Ensemble
    ensemble_artifacts = None
    try:
        ensemble_artifacts = run_model_e(data, all_results)
    except Exception as e:
        logger.error("Model E failed: %s", e, exc_info=True)

    # Model F: Ridge
    try:
        model_f = run_model_f(data, all_results)
    except Exception as e:
        logger.error("Model F failed: %s", e, exc_info=True)

    # Model G: HistGradientBoosting
    try:
        model_g = run_model_g(data, all_results)
    except Exception as e:
        logger.error("Model G failed: %s", e, exc_info=True)

    # ---- Step 6: Save results ----
    logger.info("\n=== STEP 6: Saving results ===")

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
        elif isinstance(v, list):
            clean_results[k] = [make_serializable(x) if isinstance(x, dict)
                                else {kkk: make_serializable(vvv) for kkk, vvv in x.items()}
                                if isinstance(x, dict) else make_serializable(x) for x in v]
        else:
            clean_results[k] = make_serializable(v)

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

    # Save best model checkpoint (prefer single-model, not ensemble)
    # Map from result key prefix to model dict key
    saveable_models = {
        "A_NN_64_32": all_models.get("A"),
        "B_NN_64_32_16": all_models.get("B"),
        "C_ResidualNN": all_models.get("C"),
        "D_Probabilistic": all_models.get("D"),
    }
    best_model_name = None
    best_test_mae = float("inf")
    for prefix, model_obj in saveable_models.items():
        key = f"{prefix}_test"
        if model_obj is not None and key in all_results:
            if all_results[key]["mae"] < best_test_mae:
                best_test_mae = all_results[key]["mae"]
                best_model_name = prefix

    if best_model_name:
        model_to_save = saveable_models[best_model_name]
        model_path = os.path.join(RESULTS_DIR, "best_model.pt")
        torch.save({
            "model_name": best_model_name,
            "test_mae": best_test_mae,
            "n_features": data["n_features"],
            "feature_names": data["feature_names"],
            "state_dict": model_to_save.state_dict(),
        }, model_path)
        logger.info("Saved best single model (%s, test MAE=%.3f) to %s",
                    best_model_name, best_test_mae, model_path)

    if ensemble_artifacts is not None:
        ensemble_ckpt_path = os.path.join(RESULTS_DIR, "best_ensemble_5seed.pt")
        torch.save({
            "model_name": "E_Ensemble_5seed",
            "hidden_sizes": [64, 32],
            "dropout": 0.15,
            "n_features": data["n_features"],
            "feature_names": data["feature_names"],
            "seed_state_dicts": ensemble_artifacts["seed_state_dicts"],
        }, ensemble_ckpt_path)
        logger.info("Saved reusable ensemble checkpoint to %s", ensemble_ckpt_path)

        scaler_path = os.path.join(RESULTS_DIR, "best_ensemble_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(data.get("scaler"), f)
        logger.info("Saved scaler to %s", scaler_path)

        _build_benchmark_prediction_files(data, ensemble_artifacts)

    # ---- Step 7: Print comprehensive comparison ----
    logger.info("\n" + "=" * 120)
    logger.info("COMPREHENSIVE RESULTS COMPARISON")
    logger.info("=" * 120)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        # Baseline reference
        baseline_test = 2.090
        baseline_oos = 2.093

        # Test set comparison
        logger.info("\n--- TEST SET (sorted by MAE) ---")
        test_df = summary_df[summary_df["split"] == "test"].sort_values("mae")
        header = f"{'Model':<35} {'MAE':>7} {'RMSE':>7} {'R2':>6}  {'DJF':>6} {'MAM':>6} {'JJA':>6} {'SON':>6}  {'Delta':>7}"
        logger.info(header)
        logger.info("-" * 110)
        for _, row in test_df.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                val = row.get(s, np.nan)
                seasonal += f" {val:6.2f}" if pd.notna(val) else "    N/A"
            delta = row["mae"] - baseline_test
            logger.info("%-35s %7.3f %7.3f %6.3f %s %+7.3f",
                        row["model"][:35], row["mae"], row["rmse"], row["r2"],
                        seasonal, delta)

        # OOS comparison
        logger.info("\n--- OOS SET (sorted by MAE) ---")
        oos_df = summary_df[summary_df["split"] == "oos"].sort_values("mae")
        logger.info(header)
        logger.info("-" * 110)
        for _, row in oos_df.iterrows():
            seasonal = ""
            for s in ["mae_DJF", "mae_MAM", "mae_JJA", "mae_SON"]:
                val = row.get(s, np.nan)
                seasonal += f" {val:6.2f}" if pd.notna(val) else "    N/A"
            delta = row["mae"] - baseline_oos
            logger.info("%-35s %7.3f %7.3f %6.3f %s %+7.3f",
                        row["model"][:35], row["mae"], row["rmse"], row["r2"],
                        seasonal, delta)

        # Probabilistic model coverage
        logger.info("\n--- PROBABILISTIC MODEL COVERAGE (Model D) ---")
        for split in ["test", "oos"]:
            key = f"D_Probabilistic_{split}"
            if key in all_results:
                r = all_results[key]
                logger.info("  %s: MAE=%.3f  CRPS=%.3f  sigma=%.2f  "
                            "cov50=%.1f%%  cov80=%.1f%%  cov90=%.1f%%  cov95=%.1f%%  "
                            "width95=%.1fF",
                            split,
                            r.get("mae", 0), r.get("crps", 0), r.get("mean_sigma", 0),
                            r.get("cov_50", 0) * 100,
                            r.get("cov_80", 0) * 100,
                            r.get("cov_90", 0) * 100,
                            r.get("cov_95", 0) * 100,
                            r.get("width_95", 0))

        # Top 5 summary
        logger.info("\n" + "=" * 80)
        logger.info("TOP 5 MODELS BY TEST MAE")
        logger.info("=" * 80)
        for i, (_, row) in enumerate(test_df.head(5).iterrows(), 1):
            oos_row = oos_df[oos_df["model"] == row["model"]]
            oos_mae = oos_row.iloc[0]["mae"] if len(oos_row) > 0 else float("nan")
            gap = abs(oos_mae - row["mae"]) if not np.isnan(oos_mae) else float("nan")
            logger.info("  #%d: %-30s  Test=%.3f  OOS=%.3f  Gap=%.3f  DeltaVsBaseline=%+.3f",
                        i, row["model"][:30], row["mae"], oos_mae, gap,
                        row["mae"] - baseline_test)

        logger.info("\nTOP 5 MODELS BY OOS MAE")
        logger.info("=" * 80)
        for i, (_, row) in enumerate(oos_df.head(5).iterrows(), 1):
            test_row = test_df[test_df["model"] == row["model"]]
            test_mae = test_row.iloc[0]["mae"] if len(test_row) > 0 else float("nan")
            gap = abs(row["mae"] - test_mae) if not np.isnan(test_mae) else float("nan")
            logger.info("  #%d: %-30s  OOS=%.3f  Test=%.3f  Gap=%.3f  DeltaVsBaseline=%+.3f",
                        i, row["model"][:30], row["mae"], test_mae, gap,
                        row["mae"] - baseline_oos)

        # Ensemble seed breakdown
        logger.info("\n--- ENSEMBLE SEED BREAKDOWN (Test) ---")
        for seed in ENSEMBLE_SEEDS:
            key = f"E_seed_{seed}_test"
            if key in all_results:
                logger.info("  Seed %4d: Test MAE=%.3f", seed, all_results[key]["mae"])
        ens_key = "E_Ensemble_5seed_test"
        if ens_key in all_results:
            logger.info("  Ensemble:  Test MAE=%.3f", all_results[ens_key]["mae"])

    elapsed = time.time() - start_time
    logger.info("\nTotal pipeline time: %.1f minutes", elapsed / 60)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
