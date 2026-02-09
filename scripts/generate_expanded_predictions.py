#!/usr/bin/env python3
"""
Generate Expanded Model Predictions for NYC Temperature.

Uses ~52 surrounding GHCN stations over 1998-2025 with:
  - Lag-1 TMAX for all qualifying surrounding stations
  - NYC autoregressive input (lag-1 TMAX)
  - Cyclical date encoding (sin_day, cos_day)

Trains:
  - Neural network (TempPredictorV1 with [128, 64] architecture, Huber loss)
  - Ridge regression baseline

Splits:
  - Train:         1998-01-01 to 2019-12-31 (~22 years)
  - Validation:    2020-01-01 to 2022-12-31 (3 years)
  - IS Prediction: 2023-01-01 to 2024-12-31 (2 years)
  - OOS Prediction:2025-01-01 to 2025-12-31 (1 year)

Outputs:
  - data/expanded_model_predictions_2023_2024.csv
  - data/expanded_model_predictions_2025.csv
  - data/expanded_ridge_predictions_2023_2024.csv
  - data/expanded_ridge_predictions_2025.csv
  - models/expanded_nn_model.pt
  - models/expanded_ridge_model.pkl
  - models/expanded_sigma_estimates.json
  - models/expanded_training_report.json
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
START_DATE = "1998-01-01"
END_DATE = "2025-12-31"

TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2022-12-31"
IS_START = "2023-01-01"
IS_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"

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
            logger.warning("No .dly file for %s — skipping", sid)
            continue

        logger.info("Parsing %s ...", sid)
        df_long = parse_dly_file(dly_path, START_DATE, END_DATE)
        if df_long.empty:
            logger.warning("No data for %s in date range — skipping", sid)
            continue

        df_wide = pivot_station_data(df_long)
        if "TMAX" not in df_wide.columns:
            logger.warning("No TMAX for %s — skipping", sid)
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
    """Check completeness of each station over 1998-2024 (training+val+IS).

    Returns list of qualifying station IDs and a report dict.
    """
    # We check completeness over training+val period (1998-2024)
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

        # Also check earliest date
        if not df.empty:
            earliest = df.index.min()
        else:
            earliest = pd.NaT

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
# Step 3: Build features
# ===========================================================================
def build_features(station_data, qualifying_ids):
    """Build feature matrix with lag-1 TMAX + NYC autoregressive + date encoding.

    Returns: X DataFrame (features), y Series (target), dates Series.
    """
    target_id = config.TARGET_STATION
    surrounding_ids = [sid for sid in qualifying_ids if sid != target_id]

    if target_id not in qualifying_ids:
        raise ValueError(f"Target station {target_id} did not pass completeness check!")

    logger.info("Building features with %d surrounding stations", len(surrounding_ids))

    # Create a master date range
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
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
# Step 4: Split, impute, scale
# ===========================================================================
def split_and_prepare(X, y, dates):
    """Chronological split, forward-fill, impute, and scale.

    Returns dict with train/val/is/oos splits (X_scaled, y, dates).
    """
    # Convert dates to a comparable format
    idx = pd.to_datetime(dates.values)

    # Split masks
    train_mask = idx <= pd.Timestamp(TRAIN_END)
    val_mask = (idx >= pd.Timestamp(VAL_START)) & (idx <= pd.Timestamp(VAL_END))
    is_mask = (idx >= pd.Timestamp(IS_START)) & (idx <= pd.Timestamp(IS_END))
    oos_mask = (idx >= pd.Timestamp(OOS_START)) & (idx <= pd.Timestamp(OOS_END))

    splits = {
        "train": (X[train_mask].copy(), y[train_mask].copy(), dates[train_mask].copy()),
        "val": (X[val_mask].copy(), y[val_mask].copy(), dates[val_mask].copy()),
        "is": (X[is_mask].copy(), y[is_mask].copy(), dates[is_mask].copy()),
        "oos": (X[oos_mask].copy(), y[oos_mask].copy(), dates[oos_mask].copy()),
    }

    # Log sizes
    for name, (Xs, ys, ds) in splits.items():
        logger.info("  %s: %d rows, y non-null: %d", name, len(Xs), ys.notna().sum())

    # Forward-fill up to MAX_FFILL_DAYS on the full X before splitting
    # We do this per-split to avoid leakage
    for name in splits:
        Xs, ys, ds = splits[name]
        Xs = Xs.ffill(limit=MAX_FFILL_DAYS)
        splits[name] = (Xs, ys, ds)

    # Compute training column means for imputation (training data only)
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
        # Reset indices first to ensure alignment
        Xs = Xs.reset_index(drop=True)
        ys = ys.reset_index(drop=True)
        ds = ds.reset_index(drop=True)
        valid = ys.notna()
        n_dropped = (~valid).sum()
        splits[name] = (Xs[valid].reset_index(drop=True),
                        ys[valid].reset_index(drop=True),
                        ds[valid].reset_index(drop=True))
        if n_dropped > 0:
            logger.info("  %s: dropped %d rows with NaN target", name, n_dropped)

    # Scale features - fit on training data only
    scaler = StandardScaler()
    X_train_arr = splits["train"][0].values
    scaler.fit(X_train_arr)

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
        logger.info("  %s (final): %d samples", name, len(data["y"]))

    return scaled_splits, scaler


# ===========================================================================
# Step 5: Train Neural Network
# ===========================================================================
def train_nn(scaled_splits, n_features):
    """Train TempPredictorV1 with Huber loss and ReduceLROnPlateau.

    Returns: model, training_history dict.
    """
    from src.model import TempPredictorV1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training NN on device: %s", device)

    # Build model
    model = TempPredictorV1(
        n_features=n_features,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", total_params)

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
                "Epoch %3d: train_loss=%.4f val_loss=%.4f train_MAE=%.3f val_MAE=%.3f lr=%.6f",
                epoch, avg_train_loss, val_loss, train_mae, val_mae, current_lr,
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
                logger.info("Early stopping at epoch %d (best epoch: %d)", epoch, best_epoch)
                break

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds, best epoch: %d", elapsed, best_epoch)

    # Restore best model
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
def train_ridge(scaled_splits):
    """Train Ridge regression baseline. Returns model and metrics."""
    ridge = Ridge(alpha=1.0)
    ridge.fit(scaled_splits["train"]["X"], scaled_splits["train"]["y"])

    metrics = {}
    for name in ["train", "val", "is", "oos"]:
        pred = ridge.predict(scaled_splits[name]["X"])
        actual = scaled_splits[name]["y"]
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        metrics[name] = {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}
        logger.info("Ridge %s: MAE=%.3f, RMSE=%.3f", name, mae, rmse)

    return ridge, metrics


# ===========================================================================
# Step 7: Estimate monthly sigma from validation residuals
# ===========================================================================
def estimate_sigma(model, scaled_splits, device, model_type="nn"):
    """Estimate monthly prediction uncertainty from validation residuals.

    Returns: dict of {month: sigma_value}
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

    logger.info("Monthly sigma estimates: %s", sigma)
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
# Main
# ===========================================================================
def main():
    logger.info("=" * 70)
    logger.info("EXPANDED MODEL PREDICTION PIPELINE")
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Step 1: Load all station data ---
    logger.info("\n--- Step 1: Loading station data ---")
    station_data = load_all_stations()

    # --- Step 2: Check completeness ---
    logger.info("\n--- Step 2: Checking data completeness ---")
    qualifying_ids, completeness_report = check_completeness(station_data)

    # Keep only qualifying stations
    station_data = {sid: df for sid, df in station_data.items() if sid in qualifying_ids}

    # --- Step 3: Build features ---
    logger.info("\n--- Step 3: Building features ---")
    X, y, dates, feature_cols = build_features(station_data, qualifying_ids)

    n_surrounding = len([c for c in feature_cols if c.endswith("_tmax_lag1") and c != "nyc_tmax_lag1"])
    logger.info("Feature summary:")
    logger.info("  Total features: %d", len(feature_cols))
    logger.info("  Surrounding station lag-1 TMAX: %d", n_surrounding)
    logger.info("  NYC autoregressive: 1")
    logger.info("  Date encoding: 2 (sin_day, cos_day)")

    # --- Step 4: Split and prepare ---
    logger.info("\n--- Step 4: Splitting, imputing, scaling ---")
    scaled_splits, scaler = split_and_prepare(X, y, dates)

    # --- Step 5: Train NN ---
    logger.info("\n--- Step 5: Training Neural Network ---")
    n_features = scaled_splits["train"]["X"].shape[1]
    nn_model, nn_history, device = train_nn(scaled_splits, n_features)

    # Compute NN metrics on all splits
    nn_metrics = {}
    for split_name in ["train", "val", "is", "oos"]:
        with torch.no_grad():
            X_t = torch.FloatTensor(scaled_splits[split_name]["X"]).to(device)
            pred = nn_model(X_t).cpu().numpy().flatten()
        actual = scaled_splits[split_name]["y"]
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        nn_metrics[split_name] = {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}
        logger.info("NN %s: MAE=%.3f, RMSE=%.3f", split_name, mae, rmse)

    # --- Step 6: Train Ridge ---
    logger.info("\n--- Step 6: Training Ridge Regression ---")
    ridge_model, ridge_metrics = train_ridge(scaled_splits)

    # --- Step 7: Estimate sigma ---
    logger.info("\n--- Step 7: Estimating monthly sigma ---")
    nn_sigma = estimate_sigma(nn_model, scaled_splits, device, model_type="nn")
    ridge_sigma = estimate_sigma(ridge_model, scaled_splits, device, model_type="ridge")

    # --- Step 8: Generate predictions ---
    logger.info("\n--- Step 8: Generating predictions ---")

    # NN predictions
    nn_is_df, nn_is_metrics = generate_predictions(
        nn_model, scaled_splits, device, nn_sigma, "is", "nn"
    )
    nn_oos_df, nn_oos_metrics = generate_predictions(
        nn_model, scaled_splits, device, nn_sigma, "oos", "nn"
    )

    # Ridge predictions
    ridge_is_df, ridge_is_metrics = generate_predictions(
        ridge_model, scaled_splits, device, ridge_sigma, "is", "ridge"
    )
    ridge_oos_df, ridge_oos_metrics = generate_predictions(
        ridge_model, scaled_splits, device, ridge_sigma, "oos", "ridge"
    )

    # --- Step 9: Save everything ---
    logger.info("\n--- Step 9: Saving outputs ---")

    # Prediction CSVs
    nn_is_path = os.path.join(DATA_DIR, "expanded_model_predictions_2023_2024.csv")
    nn_oos_path = os.path.join(DATA_DIR, "expanded_model_predictions_2025.csv")
    ridge_is_path = os.path.join(DATA_DIR, "expanded_ridge_predictions_2023_2024.csv")
    ridge_oos_path = os.path.join(DATA_DIR, "expanded_ridge_predictions_2025.csv")

    nn_is_df.to_csv(nn_is_path, index=False)
    nn_oos_df.to_csv(nn_oos_path, index=False)
    ridge_is_df.to_csv(ridge_is_path, index=False)
    ridge_oos_df.to_csv(ridge_oos_path, index=False)
    logger.info("Saved prediction CSVs")

    # NN model
    nn_model_path = os.path.join(MODELS_DIR, "expanded_nn_model.pt")
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "n_features": n_features,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": feature_cols,
    }, nn_model_path)
    logger.info("Saved NN model: %s", nn_model_path)

    # Ridge model
    ridge_model_path = os.path.join(MODELS_DIR, "expanded_ridge_model.pkl")
    with open(ridge_model_path, "wb") as f:
        pickle.dump({
            "model": ridge_model,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_cols": feature_cols,
        }, f)
    logger.info("Saved Ridge model: %s", ridge_model_path)

    # Sigma estimates
    sigma_path = os.path.join(MODELS_DIR, "expanded_sigma_estimates.json")
    # Convert int keys to strings for JSON
    sigma_data = {
        "nn_sigma": {str(k): v for k, v in nn_sigma.items()},
        "ridge_sigma": {str(k): v for k, v in ridge_sigma.items()},
    }
    with open(sigma_path, "w") as f:
        json.dump(sigma_data, f, indent=2)
    logger.info("Saved sigma estimates: %s", sigma_path)

    # Training report
    report_path = os.path.join(MODELS_DIR, "expanded_training_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "date_range": {"start": START_DATE, "end": END_DATE},
        "splits": {
            "train": {"start": START_DATE, "end": TRAIN_END, "n_samples": len(scaled_splits["train"]["y"])},
            "val": {"start": VAL_START, "end": VAL_END, "n_samples": len(scaled_splits["val"]["y"])},
            "is": {"start": IS_START, "end": IS_END, "n_samples": len(scaled_splits["is"]["y"])},
            "oos": {"start": OOS_START, "end": OOS_END, "n_samples": len(scaled_splits["oos"]["y"])},
        },
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
        "nn": {
            "architecture": HIDDEN_SIZES,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "loss": "HuberLoss(delta=1.0)",
            "best_epoch": nn_history["best_epoch"],
            "total_epochs": nn_history["total_epochs"],
            "total_params": nn_history["total_params"],
            "elapsed_seconds": round(nn_history["elapsed_seconds"], 1),
            "metrics": nn_metrics,
            "is_metrics": nn_is_metrics,
            "oos_metrics": nn_oos_metrics,
        },
        "ridge": {
            "alpha": 1.0,
            "metrics": ridge_metrics,
            "is_metrics": ridge_is_metrics,
            "oos_metrics": ridge_oos_metrics,
        },
        "sigma": sigma_data,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved training report: %s", report_path)

    # --- Final Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Features: %d (%d surrounding stations + 1 NYC AR + 2 date)", n_features, n_surrounding)
    logger.info("Train: %d samples | Val: %d | IS: %d | OOS: %d",
                len(scaled_splits["train"]["y"]), len(scaled_splits["val"]["y"]),
                len(scaled_splits["is"]["y"]), len(scaled_splits["oos"]["y"]))
    logger.info("")
    logger.info("Neural Network:")
    logger.info("  Architecture: %s, params: %d", HIDDEN_SIZES, nn_history["total_params"])
    logger.info("  Train MAE: %.3f | Val MAE: %.3f", nn_metrics["train"]["mae"], nn_metrics["val"]["mae"])
    logger.info("  IS MAE:    %.3f | OOS MAE: %.3f", nn_is_metrics["mae"], nn_oos_metrics["mae"])
    logger.info("")
    logger.info("Ridge Regression:")
    logger.info("  Train MAE: %.3f | Val MAE: %.3f", ridge_metrics["train"]["mae"], ridge_metrics["val"]["mae"])
    logger.info("  IS MAE:    %.3f | OOS MAE: %.3f", ridge_is_metrics["mae"], ridge_oos_metrics["mae"])
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    report = main()
