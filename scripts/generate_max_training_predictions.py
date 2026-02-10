#!/usr/bin/env python3
"""
Generate Maximum-Training Predictions for NYC Temperature.

Trains TWO separate models, each using the MAXIMUM available training data
for its respective prediction period:

  Model A (IS predictions for 2023-2024):
    - Train:      1998-01-01 to 2020-12-31  (23 years)
    - Validation:  2021-01-01 to 2022-12-31  (2 years)
    - Predict:     2023-01-01 to 2024-12-31

  Model B (OOS predictions for 2025):
    - Train:      1998-01-01 to 2022-12-31  (25 years)
    - Validation:  2023-01-01 to 2024-12-31  (2 years)
    - Predict:     2025-01-01 to 2025-12-31

Features (50 total):
  - Lag-1 TMAX for ~47 qualifying surrounding stations
  - NYC autoregressive input (lag-1 TMAX)
  - Cyclical date encoding (sin_day, cos_day)

Models:
  - Neural network: TempPredictorV1 [128, 64], Huber loss, dropout=0.0
  - Ridge regression: alpha=1.0

Outputs (in data/ and models/):
  - max_train_nn_predictions_is.csv
  - max_train_nn_predictions_oos.csv
  - max_train_ridge_predictions_is.csv
  - max_train_ridge_predictions_oos.csv
  - max_train_nn_model_is.pt
  - max_train_nn_model_oos.pt
  - max_train_ridge_model_is.pkl
  - max_train_ridge_model_oos.pkl
  - max_train_sigma_estimates.json
  - max_train_report.json
"""

import os
import sys
import json
import logging
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config_expanded as config
from src.data_collection import parse_dly_file, pivot_station_data
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
DATA_START = "1998-01-01"
DATA_END = "2025-12-31"

# Model A splits (for IS prediction)
MODEL_A_TRAIN_END = "2020-12-31"
MODEL_A_VAL_START = "2021-01-01"
MODEL_A_VAL_END = "2022-12-31"
MODEL_A_PRED_START = "2023-01-01"
MODEL_A_PRED_END = "2024-12-31"

# Model B splits (for OOS prediction)
MODEL_B_TRAIN_END = "2022-12-31"
MODEL_B_VAL_START = "2023-01-01"
MODEL_B_VAL_END = "2024-12-31"
MODEL_B_PRED_START = "2025-01-01"
MODEL_B_PRED_END = "2025-12-31"

MIN_COMPLETENESS = 0.80
MAX_FFILL_DAYS = 3

# NN hyperparameters
HIDDEN_SIZES = [128, 64]
DROPOUT = 0.0
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20

# Paths
RAW_DIR = config.RAW_DATA_DIR
DATA_DIR = config.DATA_DIR
MODELS_DIR = config.MODELS_DIR


# ===========================================================================
# Step 1: Parse all station data
# ===========================================================================
def load_all_stations():
    """Parse .dly files for target + all surrounding stations.

    Returns dict of {station_id: DataFrame with date index, TMAX column}.
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
        if "TMAX" not in df_wide.columns:
            logger.warning("No TMAX for %s -- skipping", sid)
            continue

        # Keep only TMAX
        tmax = df_wide[["TMAX"]].copy()
        tmax.index = pd.to_datetime(tmax.index)
        station_data[sid] = tmax

    logger.info("Loaded %d stations with TMAX data", len(station_data))
    return station_data


# ===========================================================================
# Step 2: Check data quality and filter stations
# ===========================================================================
def check_completeness(station_data):
    """Check completeness of each station over 1998-2024.

    Returns list of qualifying station IDs and a report dict.
    """
    check_start = pd.Timestamp("1998-01-01")
    check_end = pd.Timestamp("2024-12-31")
    total_days = (check_end - check_start).days + 1

    report = {}
    qualifying = []

    for sid, df in station_data.items():
        mask = (df.index >= check_start) & (df.index <= check_end)
        subset = df.loc[mask, "TMAX"]
        n_valid = subset.notna().sum()
        completeness = n_valid / total_days if total_days > 0 else 0.0

        earliest = df.index.min() if not df.empty else pd.NaT

        report[sid] = {
            "name": config.ALL_STATIONS.get(sid, "Unknown"),
            "earliest_date": str(earliest.date()) if pd.notna(earliest) else "N/A",
            "valid_days_1998_2024": int(n_valid),
            "total_possible_days": total_days,
            "completeness": round(completeness, 4),
        }

        if completeness >= MIN_COMPLETENESS:
            qualifying.append(sid)
        else:
            logger.warning(
                "DROPPING %s (%s): completeness=%.1f%% < %.0f%% threshold",
                sid, config.ALL_STATIONS.get(sid, "?"),
                completeness * 100, MIN_COMPLETENESS * 100,
            )

    logger.info(
        "Completeness check: %d/%d stations qualify (>= %.0f%%)",
        len(qualifying), len(station_data), MIN_COMPLETENESS * 100,
    )
    return qualifying, report


# ===========================================================================
# Step 3: Build full feature matrix (entire date range)
# ===========================================================================
def build_features(station_data, qualifying_ids):
    """Build feature matrix with lag-1 TMAX + NYC autoregressive + date encoding.

    Returns: X DataFrame (features), y Series (target), dates Series, feature_cols list.
    """
    target_id = config.TARGET_STATION
    surrounding_ids = [sid for sid in qualifying_ids if sid != target_id]

    if target_id not in qualifying_ids:
        raise ValueError(f"Target station {target_id} did not pass completeness check!")

    logger.info("Building features with %d surrounding stations", len(surrounding_ids))

    # Create a master date range covering all data
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    master = pd.DataFrame(index=date_range)
    master.index.name = "date"

    # Add target TMAX (day t)
    target_tmax = station_data[target_id]["TMAX"].copy()
    target_tmax.index = pd.to_datetime(target_tmax.index)
    master["target_tmax"] = target_tmax

    # Add NYC autoregressive: lag-1 TMAX (day t-1)
    master["nyc_tmax_lag1"] = master["target_tmax"].shift(1)

    # Add lag-1 TMAX for each surrounding station
    feature_cols = ["nyc_tmax_lag1"]
    for sid in surrounding_ids:
        col_name = f"{sid}_tmax_lag1"
        df = station_data[sid]
        df.index = pd.to_datetime(df.index)
        master[col_name] = df["TMAX"].shift(1)  # lag-1
        feature_cols.append(col_name)

    # Add cyclical date encoding
    day_of_year = master.index.dayofyear
    master["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    master["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    feature_cols.extend(["sin_day", "cos_day"])

    # Drop the first row (no lag available for day 1)
    master = master.iloc[1:]

    # Separate target
    y = master["target_tmax"].copy()
    X = master[feature_cols].copy()
    dates = master.index.to_series().reset_index(drop=True)

    logger.info(
        "Feature matrix: %d rows x %d features, target has %d non-null",
        len(X), len(feature_cols), y.notna().sum(),
    )

    return X, y, dates, feature_cols


# ===========================================================================
# Step 4: Split, impute, and scale for a given model configuration
# ===========================================================================
def prepare_model_splits(X, y, dates, train_end, val_start, val_end, pred_start, pred_end, label):
    """Create train/val/pred splits, impute NaNs, and scale.

    Scaler is fit on training data only to prevent leakage.

    Parameters
    ----------
    X, y, dates : full feature matrix, target, and date series
    train_end, val_start, val_end, pred_start, pred_end : split boundaries
    label : str, descriptive label for logging (e.g., "Model_A_IS")

    Returns
    -------
    dict with "train", "val", "pred" keys, each containing X, y, dates arrays.
    Also returns the fitted scaler.
    """
    logger.info("Preparing splits for %s", label)
    idx = pd.to_datetime(dates.values)

    # Split masks
    train_mask = (idx >= pd.Timestamp(DATA_START)) & (idx <= pd.Timestamp(train_end))
    val_mask = (idx >= pd.Timestamp(val_start)) & (idx <= pd.Timestamp(val_end))
    pred_mask = (idx >= pd.Timestamp(pred_start)) & (idx <= pd.Timestamp(pred_end))

    splits = {
        "train": (X[train_mask].copy(), y[train_mask].copy(), dates[train_mask].copy()),
        "val": (X[val_mask].copy(), y[val_mask].copy(), dates[val_mask].copy()),
        "pred": (X[pred_mask].copy(), y[pred_mask].copy(), dates[pred_mask].copy()),
    }

    # Log sizes before processing
    for name, (Xs, ys, ds) in splits.items():
        logger.info("  %s %s: %d rows, y non-null: %d", label, name, len(Xs), ys.notna().sum())

    # Forward-fill up to MAX_FFILL_DAYS per split (no cross-split leakage)
    for name in splits:
        Xs, ys, ds = splits[name]
        Xs = Xs.ffill(limit=MAX_FFILL_DAYS)
        splits[name] = (Xs, ys, ds)

    # Compute training column means for imputation
    X_train = splits["train"][0]
    train_col_means = X_train.mean()

    # Impute remaining NaNs with training means
    for name in splits:
        Xs, ys, ds = splits[name]
        Xs = Xs.fillna(train_col_means)
        splits[name] = (Xs, ys, ds)

    # Drop rows where target is NaN
    for name in list(splits.keys()):
        Xs, ys, ds = splits[name]
        Xs = Xs.reset_index(drop=True)
        ys = ys.reset_index(drop=True)
        ds = ds.reset_index(drop=True)
        valid = ys.notna()
        n_dropped = (~valid).sum()
        splits[name] = (Xs[valid].reset_index(drop=True),
                        ys[valid].reset_index(drop=True),
                        ds[valid].reset_index(drop=True))
        if n_dropped > 0:
            logger.info("  %s %s: dropped %d rows with NaN target", label, name, n_dropped)

    # Scale features - fit on training data only
    scaler = StandardScaler()
    scaler.fit(splits["train"][0].values)

    scaled_splits = {}
    for name, (Xs, ys, ds) in splits.items():
        X_scaled = scaler.transform(Xs.values)
        scaled_splits[name] = {
            "X": X_scaled,
            "y": ys.values,
            "dates": ds.values,
        }

    # Log final sizes
    for name, data in scaled_splits.items():
        logger.info("  %s %s (final): %d samples", label, name, len(data["y"]))

    return scaled_splits, scaler


# ===========================================================================
# Step 5: Train Neural Network
# ===========================================================================
def train_nn(scaled_splits, n_features, label=""):
    """Train TempPredictorV1 with Huber loss and ReduceLROnPlateau.

    Returns: model, training_history dict, device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] Training NN on device: %s", label, device)

    # Build model
    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("[%s] Model params: %d", label, total_params)

    # Loss + optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7,
    )

    # DataLoaders
    X_train_t = torch.FloatTensor(scaled_splits["train"]["X"]).to(device)
    y_train_t = torch.FloatTensor(scaled_splits["train"]["y"]).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(scaled_splits["val"]["X"]).to(device)
    y_val_t = torch.FloatTensor(scaled_splits["val"]["y"]).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "lr": []}
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
        avg_train_loss = epoch_loss / n_batches

        # Validate
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

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "[%s] Epoch %3d: train_loss=%.4f val_loss=%.4f train_MAE=%.3f val_MAE=%.3f lr=%.6f",
                label, epoch, avg_train_loss, val_loss, train_mae, val_mae, current_lr,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("[%s] Early stopping at epoch %d (best epoch: %d)", label, epoch, best_epoch)
                break

    elapsed = time.time() - start_time
    logger.info("[%s] Training completed in %.1f seconds, best epoch: %d", label, elapsed, best_epoch)

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    history["best_epoch"] = best_epoch
    history["total_epochs"] = epoch
    history["elapsed_seconds"] = elapsed
    history["total_params"] = total_params

    return model, history, device


# ===========================================================================
# Step 6: Train Ridge Regression
# ===========================================================================
def train_ridge(scaled_splits, label=""):
    """Train Ridge regression. Returns model and metrics dict."""
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaled_splits["train"]["X"], scaled_splits["train"]["y"])

    metrics = {}
    for name in ["train", "val", "pred"]:
        if name not in scaled_splits:
            continue
        pred = ridge.predict(scaled_splits[name]["X"])
        actual = scaled_splits[name]["y"]
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        metrics[name] = {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}
        logger.info("[%s] Ridge %s: MAE=%.3f, RMSE=%.3f", label, name, mae, rmse)

    return ridge, metrics


# ===========================================================================
# Step 7: Estimate monthly sigma from validation residuals
# ===========================================================================
def estimate_sigma(model, scaled_splits, device, model_type="nn"):
    """Estimate monthly prediction uncertainty from validation residuals.

    Returns: dict of {month_int: sigma_value}
    """
    X_val = scaled_splits["val"]["X"]
    y_val = scaled_splits["val"]["y"]
    dates_val = pd.to_datetime(scaled_splits["val"]["dates"])

    if model_type == "nn":
        with torch.no_grad():
            X_t = torch.FloatTensor(X_val).to(device)
            pred = model(X_t).cpu().numpy().flatten()
    else:
        pred = model.predict(X_val)

    residuals = y_val - pred
    df_resid = pd.DataFrame({"month": dates_val.month, "residual": residuals})

    sigma = {}
    for month in range(1, 13):
        month_resid = df_resid[df_resid["month"] == month]["residual"]
        if len(month_resid) > 5:
            sigma[month] = round(float(month_resid.std()), 4)
        else:
            sigma[month] = round(float(df_resid["residual"].std()), 4)

    return sigma


# ===========================================================================
# Step 8: Generate predictions
# ===========================================================================
def generate_predictions(model, scaled_splits, device, sigma, split_name, model_type="nn"):
    """Generate predictions with sigma for a given split.

    Returns DataFrame with date, model_mu, model_sigma, actual_tmax.
    """
    X = scaled_splits[split_name]["X"]
    y = scaled_splits[split_name]["y"]
    dates = pd.to_datetime(scaled_splits[split_name]["dates"])

    if model_type == "nn":
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            pred = model(X_t).cpu().numpy().flatten()
    else:
        pred = model.predict(X)

    # Assign sigma based on month
    sigmas = np.array([sigma.get(d.month, sigma.get(1, 5.0)) for d in dates])

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "model_mu": np.round(pred, 4),
        "model_sigma": np.round(sigmas, 4),
        "actual_tmax": np.round(y, 4),
    })

    mae = np.mean(np.abs(pred - y))
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    logger.info(
        "%s %s predictions: n=%d, MAE=%.3f, RMSE=%.3f",
        model_type.upper(), split_name, len(df), mae, rmse,
    )

    return df, {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "n": len(df)}


# ===========================================================================
# Step 9: Compute NN metrics on all splits
# ===========================================================================
def compute_nn_metrics(nn_model, scaled_splits, device, label=""):
    """Compute MAE and RMSE for NN on all splits."""
    metrics = {}
    for split_name in ["train", "val", "pred"]:
        if split_name not in scaled_splits:
            continue
        with torch.no_grad():
            X_t = torch.FloatTensor(scaled_splits[split_name]["X"]).to(device)
            pred = nn_model(X_t).cpu().numpy().flatten()
        actual = scaled_splits[split_name]["y"]
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        metrics[split_name] = {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}
        logger.info("[%s] NN %s: MAE=%.3f, RMSE=%.3f", label, split_name, mae, rmse)
    return metrics


# ===========================================================================
# Main
# ===========================================================================
def main():
    logger.info("=" * 70)
    logger.info("MAXIMUM-TRAINING PREDICTION PIPELINE")
    logger.info("Two separate models for IS and OOS with max training data")
    logger.info("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ===================================================================
    # PHASE 1: Load and filter station data (shared across both models)
    # ===================================================================
    logger.info("\n--- Phase 1: Loading station data ---")
    station_data = load_all_stations()

    logger.info("\n--- Phase 1b: Checking data completeness ---")
    qualifying_ids, completeness_report = check_completeness(station_data)

    # Keep only qualifying stations
    station_data = {sid: df for sid, df in station_data.items() if sid in qualifying_ids}

    # ===================================================================
    # PHASE 2: Build full feature matrix (shared)
    # ===================================================================
    logger.info("\n--- Phase 2: Building features ---")
    X, y, dates, feature_cols = build_features(station_data, qualifying_ids)

    n_surrounding = len([c for c in feature_cols if c.endswith("_tmax_lag1") and c != "nyc_tmax_lag1"])
    n_features = len(feature_cols)
    logger.info("Feature summary:")
    logger.info("  Total features: %d", n_features)
    logger.info("  Surrounding station lag-1 TMAX: %d", n_surrounding)
    logger.info("  NYC autoregressive: 1")
    logger.info("  Date encoding: 2 (sin_day, cos_day)")

    # ===================================================================
    # PHASE 3: MODEL A -- IS predictions (train through 2020, val 2021-2022)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL A: For IS Prediction (2023-2024)")
    logger.info("  Train: 1998-01-01 to 2020-12-31 (23 years)")
    logger.info("  Val:   2021-01-01 to 2022-12-31 (2 years)")
    logger.info("  Pred:  2023-01-01 to 2024-12-31")
    logger.info("=" * 70)

    splits_a, scaler_a = prepare_model_splits(
        X, y, dates,
        train_end=MODEL_A_TRAIN_END,
        val_start=MODEL_A_VAL_START,
        val_end=MODEL_A_VAL_END,
        pred_start=MODEL_A_PRED_START,
        pred_end=MODEL_A_PRED_END,
        label="Model_A",
    )

    # Train NN for Model A
    logger.info("\n--- Model A: Training Neural Network ---")
    nn_model_a, nn_history_a, device = train_nn(splits_a, n_features, label="Model_A")
    nn_metrics_a = compute_nn_metrics(nn_model_a, splits_a, device, label="Model_A")

    # Train Ridge for Model A
    logger.info("\n--- Model A: Training Ridge Regression ---")
    ridge_model_a, ridge_metrics_a = train_ridge(splits_a, label="Model_A")

    # Estimate sigma for Model A (from 2021-2022 val residuals)
    logger.info("\n--- Model A: Estimating sigma ---")
    nn_sigma_a = estimate_sigma(nn_model_a, splits_a, device, model_type="nn")
    ridge_sigma_a = estimate_sigma(ridge_model_a, splits_a, device, model_type="ridge")
    logger.info("  NN sigma: %s", nn_sigma_a)
    logger.info("  Ridge sigma: %s", ridge_sigma_a)

    # Generate IS predictions
    logger.info("\n--- Model A: Generating IS predictions ---")
    nn_is_df, nn_is_metrics = generate_predictions(
        nn_model_a, splits_a, device, nn_sigma_a, "pred", "nn"
    )
    ridge_is_df, ridge_is_metrics = generate_predictions(
        ridge_model_a, splits_a, device, ridge_sigma_a, "pred", "ridge"
    )

    # ===================================================================
    # PHASE 4: MODEL B -- OOS predictions (train through 2022, val 2023-2024)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL B: For OOS Prediction (2025)")
    logger.info("  Train: 1998-01-01 to 2022-12-31 (25 years)")
    logger.info("  Val:   2023-01-01 to 2024-12-31 (2 years)")
    logger.info("  Pred:  2025-01-01 to 2025-12-31")
    logger.info("=" * 70)

    splits_b, scaler_b = prepare_model_splits(
        X, y, dates,
        train_end=MODEL_B_TRAIN_END,
        val_start=MODEL_B_VAL_START,
        val_end=MODEL_B_VAL_END,
        pred_start=MODEL_B_PRED_START,
        pred_end=MODEL_B_PRED_END,
        label="Model_B",
    )

    # Train NN for Model B
    logger.info("\n--- Model B: Training Neural Network ---")
    nn_model_b, nn_history_b, device = train_nn(splits_b, n_features, label="Model_B")
    nn_metrics_b = compute_nn_metrics(nn_model_b, splits_b, device, label="Model_B")

    # Train Ridge for Model B
    logger.info("\n--- Model B: Training Ridge Regression ---")
    ridge_model_b, ridge_metrics_b = train_ridge(splits_b, label="Model_B")

    # Estimate sigma for Model B (from 2023-2024 val residuals)
    logger.info("\n--- Model B: Estimating sigma ---")
    nn_sigma_b = estimate_sigma(nn_model_b, splits_b, device, model_type="nn")
    ridge_sigma_b = estimate_sigma(ridge_model_b, splits_b, device, model_type="ridge")
    logger.info("  NN sigma: %s", nn_sigma_b)
    logger.info("  Ridge sigma: %s", ridge_sigma_b)

    # Generate OOS predictions
    logger.info("\n--- Model B: Generating OOS predictions ---")
    nn_oos_df, nn_oos_metrics = generate_predictions(
        nn_model_b, splits_b, device, nn_sigma_b, "pred", "nn"
    )
    ridge_oos_df, ridge_oos_metrics = generate_predictions(
        ridge_model_b, splits_b, device, ridge_sigma_b, "pred", "ridge"
    )

    # ===================================================================
    # PHASE 5: Save all outputs
    # ===================================================================
    logger.info("\n--- Phase 5: Saving outputs ---")

    # -- Prediction CSVs --
    nn_is_path = os.path.join(DATA_DIR, "max_train_nn_predictions_is.csv")
    nn_oos_path = os.path.join(DATA_DIR, "max_train_nn_predictions_oos.csv")
    ridge_is_path = os.path.join(DATA_DIR, "max_train_ridge_predictions_is.csv")
    ridge_oos_path = os.path.join(DATA_DIR, "max_train_ridge_predictions_oos.csv")

    nn_is_df.to_csv(nn_is_path, index=False)
    nn_oos_df.to_csv(nn_oos_path, index=False)
    ridge_is_df.to_csv(ridge_is_path, index=False)
    ridge_oos_df.to_csv(ridge_oos_path, index=False)
    logger.info("Saved prediction CSVs to %s", DATA_DIR)

    # -- NN models --
    nn_model_is_path = os.path.join(MODELS_DIR, "max_train_nn_model_is.pt")
    torch.save({
        "model_state_dict": nn_model_a.state_dict(),
        "n_features": n_features,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "scaler_mean": scaler_a.mean_.tolist(),
        "scaler_scale": scaler_a.scale_.tolist(),
        "feature_cols": feature_cols,
        "train_end": MODEL_A_TRAIN_END,
        "val_range": f"{MODEL_A_VAL_START} to {MODEL_A_VAL_END}",
    }, nn_model_is_path)
    logger.info("Saved NN IS model: %s", nn_model_is_path)

    nn_model_oos_path = os.path.join(MODELS_DIR, "max_train_nn_model_oos.pt")
    torch.save({
        "model_state_dict": nn_model_b.state_dict(),
        "n_features": n_features,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "scaler_mean": scaler_b.mean_.tolist(),
        "scaler_scale": scaler_b.scale_.tolist(),
        "feature_cols": feature_cols,
        "train_end": MODEL_B_TRAIN_END,
        "val_range": f"{MODEL_B_VAL_START} to {MODEL_B_VAL_END}",
    }, nn_model_oos_path)
    logger.info("Saved NN OOS model: %s", nn_model_oos_path)

    # -- Ridge models --
    ridge_is_pkl_path = os.path.join(MODELS_DIR, "max_train_ridge_model_is.pkl")
    with open(ridge_is_pkl_path, "wb") as f:
        pickle.dump({
            "model": ridge_model_a,
            "scaler_mean": scaler_a.mean_.tolist(),
            "scaler_scale": scaler_a.scale_.tolist(),
            "feature_cols": feature_cols,
            "train_end": MODEL_A_TRAIN_END,
        }, f)
    logger.info("Saved Ridge IS model: %s", ridge_is_pkl_path)

    ridge_oos_pkl_path = os.path.join(MODELS_DIR, "max_train_ridge_model_oos.pkl")
    with open(ridge_oos_pkl_path, "wb") as f:
        pickle.dump({
            "model": ridge_model_b,
            "scaler_mean": scaler_b.mean_.tolist(),
            "scaler_scale": scaler_b.scale_.tolist(),
            "feature_cols": feature_cols,
            "train_end": MODEL_B_TRAIN_END,
        }, f)
    logger.info("Saved Ridge OOS model: %s", ridge_oos_pkl_path)

    # -- Sigma estimates --
    sigma_path = os.path.join(MODELS_DIR, "max_train_sigma_estimates.json")
    sigma_data = {
        "model_a_is": {
            "nn_sigma": {str(k): v for k, v in nn_sigma_a.items()},
            "ridge_sigma": {str(k): v for k, v in ridge_sigma_a.items()},
            "val_period": f"{MODEL_A_VAL_START} to {MODEL_A_VAL_END}",
        },
        "model_b_oos": {
            "nn_sigma": {str(k): v for k, v in nn_sigma_b.items()},
            "ridge_sigma": {str(k): v for k, v in ridge_sigma_b.items()},
            "val_period": f"{MODEL_B_VAL_START} to {MODEL_B_VAL_END}",
        },
    }
    with open(sigma_path, "w") as f:
        json.dump(sigma_data, f, indent=2)
    logger.info("Saved sigma estimates: %s", sigma_path)

    # -- Comprehensive training report --
    report_path = os.path.join(MODELS_DIR, "max_train_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "generate_max_training_predictions.py",
        "description": "Two separate models with maximum training data for IS and OOS",
        "features": {
            "total": n_features,
            "surrounding_stations": n_surrounding,
            "nyc_autoregressive": 1,
            "date_encoding": 2,
            "feature_names": feature_cols,
        },
        "stations": {
            "total_qualifying": len(qualifying_ids),
            "target": config.TARGET_STATION,
            "surrounding_count": len(qualifying_ids) - 1,
            "completeness_report": completeness_report,
        },
        "model_a_is": {
            "purpose": "IS predictions (2023-2024)",
            "train_range": f"{DATA_START} to {MODEL_A_TRAIN_END}",
            "train_samples": len(splits_a["train"]["y"]),
            "val_range": f"{MODEL_A_VAL_START} to {MODEL_A_VAL_END}",
            "val_samples": len(splits_a["val"]["y"]),
            "pred_range": f"{MODEL_A_PRED_START} to {MODEL_A_PRED_END}",
            "pred_samples": len(splits_a["pred"]["y"]),
            "nn": {
                "architecture": HIDDEN_SIZES,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "loss": "HuberLoss(delta=1.0)",
                "best_epoch": nn_history_a["best_epoch"],
                "total_epochs": nn_history_a["total_epochs"],
                "total_params": nn_history_a["total_params"],
                "elapsed_seconds": round(nn_history_a["elapsed_seconds"], 1),
                "metrics": nn_metrics_a,
                "pred_metrics": nn_is_metrics,
                "sigma": {str(k): v for k, v in nn_sigma_a.items()},
            },
            "ridge": {
                "alpha": 1.0,
                "metrics": ridge_metrics_a,
                "pred_metrics": ridge_is_metrics,
                "sigma": {str(k): v for k, v in ridge_sigma_a.items()},
            },
        },
        "model_b_oos": {
            "purpose": "OOS predictions (2025)",
            "train_range": f"{DATA_START} to {MODEL_B_TRAIN_END}",
            "train_samples": len(splits_b["train"]["y"]),
            "val_range": f"{MODEL_B_VAL_START} to {MODEL_B_VAL_END}",
            "val_samples": len(splits_b["val"]["y"]),
            "pred_range": f"{MODEL_B_PRED_START} to {MODEL_B_PRED_END}",
            "pred_samples": len(splits_b["pred"]["y"]),
            "nn": {
                "architecture": HIDDEN_SIZES,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "loss": "HuberLoss(delta=1.0)",
                "best_epoch": nn_history_b["best_epoch"],
                "total_epochs": nn_history_b["total_epochs"],
                "total_params": nn_history_b["total_params"],
                "elapsed_seconds": round(nn_history_b["elapsed_seconds"], 1),
                "metrics": nn_metrics_b,
                "pred_metrics": nn_oos_metrics,
                "sigma": {str(k): v for k, v in nn_sigma_b.items()},
            },
            "ridge": {
                "alpha": 1.0,
                "metrics": ridge_metrics_b,
                "pred_metrics": ridge_oos_metrics,
                "sigma": {str(k): v for k, v in ridge_sigma_b.items()},
            },
        },
        "sigma": sigma_data,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report: %s", report_path)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — MAXIMUM-TRAINING PREDICTIONS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Features: %d (%d surrounding stations + 1 NYC AR + 2 date)", n_features, n_surrounding)
    logger.info("")
    logger.info("%-28s  %-12s  %-12s", "", "MODEL A (IS)", "MODEL B (OOS)")
    logger.info("%-28s  %-12s  %-12s", "Training period",
                f"1998-{MODEL_A_TRAIN_END[:4]}", f"1998-{MODEL_B_TRAIN_END[:4]}")
    logger.info("%-28s  %-12d  %-12d", "Training samples",
                len(splits_a["train"]["y"]), len(splits_b["train"]["y"]))
    logger.info("%-28s  %-12s  %-12s", "Validation period",
                f"{MODEL_A_VAL_START[:4]}-{MODEL_A_VAL_END[:4]}",
                f"{MODEL_B_VAL_START[:4]}-{MODEL_B_VAL_END[:4]}")
    logger.info("%-28s  %-12d  %-12d", "Validation samples",
                len(splits_a["val"]["y"]), len(splits_b["val"]["y"]))
    logger.info("%-28s  %-12s  %-12s", "Prediction period",
                f"{MODEL_A_PRED_START[:4]}-{MODEL_A_PRED_END[:4]}",
                f"{MODEL_B_PRED_START[:4]}-{MODEL_B_PRED_END[:4]}")
    logger.info("%-28s  %-12d  %-12d", "Prediction samples",
                len(splits_a["pred"]["y"]), len(splits_b["pred"]["y"]))
    logger.info("")
    logger.info("Neural Network ([%s], Huber):", ", ".join(map(str, HIDDEN_SIZES)))
    logger.info("  %-26s  %-12s  %-12s", "Train MAE",
                f"{nn_metrics_a['train']['mae']:.3f}",
                f"{nn_metrics_b['train']['mae']:.3f}")
    logger.info("  %-26s  %-12s  %-12s", "Val MAE",
                f"{nn_metrics_a['val']['mae']:.3f}",
                f"{nn_metrics_b['val']['mae']:.3f}")
    logger.info("  %-26s  %-12s  %-12s", "Prediction MAE",
                f"{nn_is_metrics['mae']:.3f}",
                f"{nn_oos_metrics['mae']:.3f}")
    logger.info("  %-26s  %-12d  %-12d", "Best epoch",
                nn_history_a["best_epoch"], nn_history_b["best_epoch"])
    logger.info("")
    logger.info("Ridge Regression (alpha=1.0):")
    logger.info("  %-26s  %-12s  %-12s", "Train MAE",
                f"{ridge_metrics_a['train']['mae']:.3f}",
                f"{ridge_metrics_b['train']['mae']:.3f}")
    logger.info("  %-26s  %-12s  %-12s", "Val MAE",
                f"{ridge_metrics_a['val']['mae']:.3f}",
                f"{ridge_metrics_b['val']['mae']:.3f}")
    logger.info("  %-26s  %-12s  %-12s", "Prediction MAE",
                f"{ridge_is_metrics['mae']:.3f}",
                f"{ridge_oos_metrics['mae']:.3f}")
    logger.info("=" * 70)

    # List all saved files
    logger.info("\nSaved files:")
    saved_files = [
        nn_is_path, nn_oos_path, ridge_is_path, ridge_oos_path,
        nn_model_is_path, nn_model_oos_path, ridge_is_pkl_path, ridge_oos_pkl_path,
        sigma_path, report_path,
    ]
    for f in saved_files:
        size_kb = os.path.getsize(f) / 1024
        logger.info("  %s (%.1f KB)", f, size_kb)

    return report


if __name__ == "__main__":
    report = main()
