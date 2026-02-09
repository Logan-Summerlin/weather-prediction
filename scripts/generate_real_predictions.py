#!/usr/bin/env python3
"""
Generate real model predictions for NYC TMAX (2023-2024).

Pipeline:
  1. Parse GHCN .dly files for all stations (2018-2024)
  2. Create lag-1 features + cyclical date encoding
  3. Train neural network on 2018-2021, validate on 2022
  4. Predict 2023-2024 as true out-of-sample
  5. Estimate prediction uncertainty (sigma) from validation residuals
  6. Save Gaussian (mu, sigma) predictions

Outputs:
  - data/real_model_predictions_2023_2024.csv
  - models/real_backtest_model.pt
"""

import os
import sys
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_collection import parse_dly_file, pivot_station_data
from src.model import TempPredictorV1
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Data Loading and Preprocessing (self-contained for this pipeline)
# ===========================================================================

def load_station_data(station_id, raw_dir, start_date, end_date):
    """Load and parse a single station's .dly file."""
    dly_path = os.path.join(raw_dir, f"{station_id}.dly")
    if not os.path.exists(dly_path):
        logger.warning("No .dly file for %s", station_id)
        return pd.DataFrame()

    df_long = parse_dly_file(dly_path, start_date, end_date)
    if df_long.empty:
        return pd.DataFrame()

    df_wide = pivot_station_data(df_long)
    return df_wide


def build_features_and_targets(raw_dir, start_date="2018-01-01", end_date="2024-12-31"):
    """Build feature matrix and target from all station .dly files.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (features, target) where features have lag-1 surrounding station
        values + cyclical date encoding, and target is NYC TMAX.
    """
    target_station = config.TARGET_STATION
    surrounding_ids = list(config.SURROUNDING_STATIONS.keys())

    # Load all stations
    station_data = {}
    for sid in [target_station] + surrounding_ids:
        df = load_station_data(sid, raw_dir, start_date, end_date)
        if df.empty:
            logger.warning("No data for %s, skipping", sid)
            continue
        station_data[sid] = df
        logger.info("Loaded %s: %d rows, cols=%s", sid, len(df), list(df.columns))

    if target_station not in station_data:
        raise ValueError(f"Target station {target_station} not found!")

    # Merge into wide format
    frames = []
    for sid, df in station_data.items():
        renamed = df.rename(columns={c: f"{sid}_{c}" for c in df.columns})
        frames.append(renamed)

    merged = pd.concat(frames, axis=1).sort_index()
    logger.info("Merged: %d rows x %d columns", *merged.shape)

    # Target: NYC TMAX on day t
    target_col = f"{target_station}_TMAX"
    target = merged[target_col].copy()
    target.name = "NYC_TMAX"

    # Features: surrounding stations lagged by 1 day
    # Only use TMAX and TMIN (not AWND, PRCP, SNOW, SNWD) to match project plan
    surrounding_cols = [
        c for c in merged.columns
        if not c.startswith(f"{target_station}_")
        and (c.endswith("_TMAX") or c.endswith("_TMIN"))
    ]
    logger.info("Using %d surrounding station columns: %s",
                len(surrounding_cols), surrounding_cols[:6])
    features = merged[surrounding_cols].shift(1)
    features.columns = [f"{c}_lag1" for c in features.columns]

    # Add cyclical date features
    # Index is datetime.date objects - convert to day-of-year
    doys = pd.Series(
        [d.timetuple().tm_yday for d in features.index],
        index=features.index,
    )
    features = features.copy()
    features["sin_day"] = np.sin(2 * np.pi * doys / 365.25)
    features["cos_day"] = np.cos(2 * np.pi * doys / 365.25)

    # Drop first row (NaN from shift) and rows where target is missing
    valid = target.notna() & features.notna().any(axis=1)
    valid.iloc[0] = False  # First row always NaN from shift
    features = features[valid]
    target = target[valid]

    # Forward-fill gaps <= 3 days
    features = features.ffill(limit=3)

    logger.info("Features: %d rows x %d columns", *features.shape)
    logger.info("Target: %d rows", len(target))

    return features, target


def prepare_splits(features, target):
    """Split data chronologically: 2018-2021 train, 2022 val, 2023-2024 prediction.

    Returns
    -------
    dict with X_train, y_train, X_val, y_val, X_pred, dates_pred, y_pred_actual
    """
    from datetime import date

    train_end = date(2021, 12, 31)
    val_end = date(2022, 12, 31)

    train_mask = features.index <= train_end
    val_mask = (features.index > train_end) & (features.index <= val_end)
    pred_mask = features.index > val_end

    X_train = features[train_mask]
    y_train = target[train_mask]
    X_val = features[val_mask]
    y_val = target[val_mask]
    X_pred = features[pred_mask]
    y_pred_actual = target[pred_mask]

    logger.info("Train: %d rows (%s to %s)", len(X_train),
                X_train.index.min(), X_train.index.max())
    logger.info("Val: %d rows (%s to %s)", len(X_val),
                X_val.index.min(), X_val.index.max())
    logger.info("Pred: %d rows (%s to %s)", len(X_pred),
                X_pred.index.min(), X_pred.index.max())

    # Drop columns that are ALL NaN in training
    all_nan_cols = X_train.columns[X_train.isna().all()]
    if len(all_nan_cols) > 0:
        logger.info("Dropping %d all-NaN columns: %s", len(all_nan_cols), list(all_nan_cols))
        X_train = X_train.drop(columns=all_nan_cols)
        X_val = X_val.drop(columns=all_nan_cols)
        X_pred = X_pred.drop(columns=all_nan_cols)

    # Fill NaNs with training means
    col_means = X_train.mean()
    X_train = X_train.fillna(col_means)
    X_val = X_val.fillna(col_means)
    X_pred = X_pred.fillna(col_means)

    # Replace any remaining NaN (from all-NaN columns) with 0
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_pred = X_pred.fillna(0)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_s = pd.DataFrame(scaler.transform(X_train),
                              index=X_train.index, columns=X_train.columns)
    X_val_s = pd.DataFrame(scaler.transform(X_val),
                            index=X_val.index, columns=X_val.columns)
    X_pred_s = pd.DataFrame(scaler.transform(X_pred),
                             index=X_pred.index, columns=X_pred.columns)

    return {
        "X_train": X_train_s,
        "y_train": y_train,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_pred": X_pred_s,
        "y_pred_actual": y_pred_actual,
        "scaler": scaler,
        "col_means": col_means,
    }


def train_model(X_train, y_train, X_val, y_val, models_dir, n_features,
                max_epochs=200, patience=15, lr=0.001):
    """Train a neural network and return the best model."""
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "real_backtest_model.pt")

    # Create model
    model = TempPredictorV1(n_features=n_features, hidden_sizes=[64, 32], dropout=0.0)

    # Convert to tensors
    X_tr = torch.tensor(X_train.values, dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val.values, dtype=torch.float32)
    y_v = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_v, y_v), batch_size=64, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    # Save initial state as fallback
    torch.save(model.state_dict(), checkpoint_path)

    logger.info("Training neural network: %d features, %d train, %d val",
                n_features, len(X_train), len(X_val))

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                all_preds.append(pred.numpy())
                all_actuals.append(yb.numpy())

        val_preds = np.concatenate(all_preds).ravel()
        val_actuals = np.concatenate(all_actuals).ravel()
        val_mae = np.mean(np.abs(val_preds - val_actuals))

        scheduler.step(total_loss / len(train_loader))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            if epoch % 10 == 0 or epoch <= 5:
                logger.info("Epoch %3d: Val MAE=%.3f F *BEST*", epoch, val_mae)
        else:
            epochs_no_improve += 1
            if epoch % 20 == 0:
                logger.info("Epoch %3d: Val MAE=%.3f F (no improve %d/%d)",
                            epoch, val_mae, epochs_no_improve, patience)

        if epochs_no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    logger.info("Best model: epoch %d, Val MAE=%.3f F", best_epoch, best_val_mae)

    return model, best_val_mae


def estimate_sigma(model, X_val, y_val):
    """Estimate prediction uncertainty from validation residuals.

    Uses a seasonal sigma model: different sigma for each month.
    Also estimates an overall sigma as fallback.

    Returns
    -------
    dict
        Mapping of month (1-12) -> sigma, plus "overall" -> sigma
    """
    model.eval()
    X_v = torch.tensor(X_val.values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_v).numpy().ravel()

    actuals = y_val.values
    residuals = actuals - preds

    overall_sigma = float(np.std(residuals))
    logger.info("Overall validation sigma: %.2f F", overall_sigma)
    logger.info("Overall validation MAE: %.2f F", np.mean(np.abs(residuals)))

    # Monthly sigma
    monthly_sigma = {}
    from datetime import date
    months = [d.month if isinstance(d, date) else pd.Timestamp(d).month
              for d in X_val.index]

    for m in range(1, 13):
        mask = [mo == m for mo in months]
        if sum(mask) > 5:
            m_resid = residuals[mask]
            monthly_sigma[m] = float(np.std(m_resid))
        else:
            monthly_sigma[m] = overall_sigma

    logger.info("Monthly sigma: %s",
                {m: f"{s:.2f}" for m, s in monthly_sigma.items()})

    return {"overall": overall_sigma, **{m: s for m, s in monthly_sigma.items()}}


def generate_predictions(model, X_pred, y_pred_actual, sigma_dict):
    """Generate Gaussian (mu, sigma) predictions for the prediction period.

    Returns
    -------
    pd.DataFrame
        Columns: date, model_mu, model_sigma, actual_tmax
    """
    model.eval()
    X_p = torch.tensor(X_pred.values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_p).numpy().ravel()

    dates = X_pred.index
    from datetime import date
    months = [d.month if isinstance(d, date) else pd.Timestamp(d).month
              for d in dates]

    sigmas = [sigma_dict.get(m, sigma_dict["overall"]) for m in months]

    pred_df = pd.DataFrame({
        "date": dates,
        "model_mu": preds,
        "model_sigma": sigmas,
        "actual_tmax": y_pred_actual.values,
    })

    # Quick validation
    mae = np.mean(np.abs(preds - y_pred_actual.values))
    logger.info("Prediction period (2023-2024) MAE: %.2f F", mae)
    logger.info("  mu range: %.1f to %.1f", preds.min(), preds.max())
    logger.info("  sigma range: %.2f to %.2f", min(sigmas), max(sigmas))

    return pred_df


def main():
    raw_dir = config.RAW_DATA_DIR
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    logger.info("=" * 60)
    logger.info("Building real model predictions for 2023-2024")
    logger.info("=" * 60)

    # Step 1: Build features and targets (2018-2024)
    logger.info("\n--- Step 1: Loading and merging station data ---")
    features, target = build_features_and_targets(
        raw_dir, start_date="2018-01-01", end_date="2024-12-31"
    )

    # Step 2: Prepare chronological splits
    logger.info("\n--- Step 2: Preparing train/val/prediction splits ---")
    splits = prepare_splits(features, target)

    # Step 3: Train model
    logger.info("\n--- Step 3: Training neural network ---")
    n_features = splits["X_train"].shape[1]
    model, best_val_mae = train_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        models_dir, n_features,
        max_epochs=200, patience=15,
    )

    # Step 4: Estimate prediction uncertainty
    logger.info("\n--- Step 4: Estimating prediction uncertainty ---")
    sigma_dict = estimate_sigma(model, splits["X_val"], splits["y_val"])

    # Step 5: Generate predictions
    logger.info("\n--- Step 5: Generating 2023-2024 predictions ---")
    pred_df = generate_predictions(
        model, splits["X_pred"], splits["y_pred_actual"], sigma_dict
    )

    # Save
    output_path = os.path.join(data_dir, "real_model_predictions_2023_2024.csv")
    pred_df.to_csv(output_path, index=False)
    logger.info("Saved predictions: %s (%d rows)", output_path, len(pred_df))

    # Also save sigma dict for reference
    import json
    sigma_path = os.path.join(models_dir, "sigma_estimates.json")
    sigma_json = {str(k): v for k, v in sigma_dict.items()}
    with open(sigma_path, "w") as f:
        json.dump(sigma_json, f, indent=2)
    logger.info("Saved sigma estimates: %s", sigma_path)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model Prediction Summary")
    logger.info("  Training period: 2018-2021 (%d samples)", len(splits["X_train"]))
    logger.info("  Validation period: 2022 (%d samples)", len(splits["X_val"]))
    logger.info("  Prediction period: 2023-2024 (%d samples)", len(pred_df))
    logger.info("  Val MAE: %.2f F", best_val_mae)
    pred_mae = np.mean(np.abs(pred_df["model_mu"] - pred_df["actual_tmax"]))
    logger.info("  Prediction MAE: %.2f F", pred_mae)
    logger.info("  Overall sigma: %.2f F", sigma_dict["overall"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
