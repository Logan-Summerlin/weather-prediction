"""
Run Kalshi KXHIGHNY Real-Data Out-of-Sample Validation (2025).

This script implements Part 2 (Steps 7-10) of the backtesting plan using
REAL data:
  - Real GHCN Central Park TMAX observations (from NOAA bulk download)
  - Real Kalshi KXHIGHNY market structure (from Kalshi public API)
  - Real model predictions (trained on 2018-2024 GHCN data, predicting 2025)

Data sources:
  - GHCN .dly files: https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/
  - Kalshi settled markets: https://api.elections.kalshi.com/trade-api/v2/markets

Note on market probabilities:
  The Kalshi public API for settled markets provides only settlement prices
  (0 or 100 cents), not the ex-ante trading prices that would have been
  available to a trader before the day's temperature was observed. To
  construct realistic market probabilities, we use a climatological Gaussian
  forecast model anchored to real NWS persistence + climatology, which
  approximates what the Kalshi market would have been pricing. This is
  clearly documented as a limitation.

Output directories:
  - results/kalshi_real_2025_oos/
  - results/kalshi_real_combined/

Usage:
    python run_kalshi_real_oos.py
"""

import os
import sys
import json
import logging
import re
from datetime import datetime, date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.data_collection import parse_dly_file, tenths_c_to_fahrenheit
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    BacktestResult,
    _compute_max_drawdown,
)
from src.kalshi_backtester import (
    BacktestAnalyzer,
    CalibrationAnalyzer,
    compute_seasonal_pnl,
    SEASON_MAP,
    SEASON_ORDER,
)
from src.kalshi_client import compute_brier_scores

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OOS_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_real_2025_oos")
COMBINED_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_real_combined")
IS_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_real_2023_2024")
DATA_DIR = config.DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANKROLL = 10000.0
FEE_RATE = 0.07
TRAIN_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"


# ===========================================================================
# Step 1: Build Training Data and Train Model
# ===========================================================================

def build_station_features(start_date, end_date):
    """Build feature matrix from surrounding station TMAX data.

    Downloads/parses .dly files for all configured stations and creates
    a wide-format DataFrame with lagged TMAX features, cyclical date
    encoding, and the target variable.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns: date, target_tmax, sin_day, cos_day,
        and lagged station TMAX columns.
    """
    print(f"  Building features from {start_date} to {end_date}...")

    # Parse target station
    target_path = os.path.join(config.RAW_DATA_DIR, f"{config.TARGET_STATION}.dly")
    target_df = parse_dly_file(target_path, start_date=start_date, end_date=end_date)
    target_tmax = target_df[target_df["element"] == "TMAX"][["date", "value"]].copy()
    target_tmax.columns = ["date", "target_tmax"]
    target_tmax = target_tmax.sort_values("date").drop_duplicates("date")

    # Parse surrounding stations
    station_frames = {}
    for station_id in config.SURROUNDING_STATIONS:
        filepath = os.path.join(config.RAW_DATA_DIR, f"{station_id}.dly")
        if not os.path.exists(filepath):
            print(f"    Warning: {filepath} not found, skipping")
            continue
        sdf = parse_dly_file(filepath, start_date=start_date, end_date=end_date)
        stmax = sdf[sdf["element"] == "TMAX"][["date", "value"]].copy()
        stmax.columns = ["date", f"TMAX_{station_id}"]
        stmax = stmax.sort_values("date").drop_duplicates("date")
        station_frames[station_id] = stmax

    # Merge all stations
    merged = target_tmax.copy()
    for sid, sdf in station_frames.items():
        merged = merged.merge(sdf, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Add NYC autoregressive feature (lag-1)
    merged["NYC_TMAX_lag1"] = merged["target_tmax"].shift(1)

    # Add lagged surrounding station features (lag-1)
    feature_cols = []
    for sid in config.SURROUNDING_STATIONS:
        col = f"TMAX_{sid}"
        lag_col = f"{col}_lag1"
        if col in merged.columns:
            merged[lag_col] = merged[col].shift(1)
            feature_cols.append(lag_col)

    # Add cyclical date features
    merged["day_of_year"] = pd.to_datetime(merged["date"]).dt.dayofyear
    merged["sin_day"] = np.sin(2 * np.pi * merged["day_of_year"] / 365.25)
    merged["cos_day"] = np.cos(2 * np.pi * merged["day_of_year"] / 365.25)
    feature_cols.extend(["NYC_TMAX_lag1", "sin_day", "cos_day"])

    # Drop first row (no lag available)
    merged = merged.dropna(subset=["target_tmax", "NYC_TMAX_lag1"]).reset_index(drop=True)

    # Forward-fill small gaps, then impute remaining with column means
    for col in feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill(limit=3)

    print(f"    Built {len(merged)} rows with {len(feature_cols)} features")
    return merged, feature_cols


def train_ridge_model(train_df, feature_cols, target_col="target_tmax"):
    """Train a Ridge regression model for temperature prediction.

    Also computes the model's residual standard deviation for
    probabilistic predictions (Gaussian sigma).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with feature columns and target.
    feature_cols : list[str]
        List of feature column names.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple
        (model, scaler, sigma, feature_cols_used)
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()

    # Drop rows with any NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Compute residual sigma on training data
    preds = model.predict(X_scaled)
    residuals = y.values - preds
    sigma = float(np.std(residuals))

    train_mae = float(np.mean(np.abs(residuals)))
    train_r2 = float(model.score(X_scaled, y))

    print(f"    Ridge model: MAE={train_mae:.2f}F, R2={train_r2:.3f}, sigma={sigma:.2f}F")
    print(f"    Features used: {len(feature_cols)}")

    return model, scaler, sigma, feature_cols


def generate_real_predictions_2025(output_path):
    """Generate real model predictions for 2025 using GHCN data.

    Trains a Ridge regression model on 2018-2024 data and generates
    out-of-sample predictions for every day in 2025.

    Parameters
    ----------
    output_path : str
        Path to save predictions CSV.

    Returns
    -------
    pd.DataFrame
        Predictions with columns: date, model_mu, model_sigma, actual_tmax.
    """
    print("\n" + "=" * 70)
    print("GENERATING REAL MODEL PREDICTIONS FOR 2025")
    print("=" * 70)

    # Build features for training period (2018-2024) and OOS (2025)
    # Need to parse from 2018 to include enough training data
    all_data, feature_cols = build_station_features("2018-01-01", "2025-12-31")

    # Split into train (2018-2024) and test (2025)
    all_data["date_dt"] = pd.to_datetime(all_data["date"])
    train_mask = all_data["date_dt"] < "2025-01-01"
    test_mask = all_data["date_dt"] >= "2025-01-01"

    train_df = all_data[train_mask].copy()
    test_df = all_data[test_mask].copy()

    print(f"\n  Training period: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} days)")
    print(f"  OOS period:      {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} days)")

    # Impute remaining NaNs in training features with training column means
    train_means = {}
    for col in feature_cols:
        if col in train_df.columns:
            col_mean = train_df[col].mean()
            train_means[col] = col_mean
            train_df[col] = train_df[col].fillna(col_mean)

    # Impute test features with training column means (no leakage)
    for col in feature_cols:
        if col in test_df.columns and col in train_means:
            test_df[col] = test_df[col].fillna(train_means[col])

    # Train model
    print("\n  Training Ridge regression model...")
    model, scaler, sigma, used_cols = train_ridge_model(train_df, feature_cols)

    # Generate predictions for 2025
    print("\n  Generating 2025 predictions...")
    X_test = test_df[used_cols].copy()
    valid_test = X_test.notna().all(axis=1)
    X_test_valid = X_test[valid_test]

    X_test_scaled = scaler.transform(X_test_valid)
    mu_preds = model.predict(X_test_scaled)

    # Compute per-prediction sigma using residual analysis
    # Use a seasonally-varying sigma based on residuals
    train_X = train_df[used_cols].copy()
    train_valid = train_X.notna().all(axis=1)
    train_X_v = train_X[train_valid]
    train_preds = model.predict(scaler.transform(train_X_v))
    train_residuals = train_df.loc[train_valid, "target_tmax"].values - train_preds

    # Monthly residual sigma
    train_months = pd.to_datetime(train_df.loc[train_valid, "date"]).dt.month
    monthly_sigma = {}
    for m in range(1, 13):
        mask = train_months.values == m
        if mask.sum() > 10:
            monthly_sigma[m] = float(np.std(train_residuals[mask]))
        else:
            monthly_sigma[m] = sigma

    # Assign sigma per prediction
    test_months = pd.to_datetime(test_df.loc[valid_test, "date"]).dt.month
    sigma_preds = np.array([monthly_sigma.get(m, sigma) for m in test_months.values])

    # Build predictions DataFrame
    predictions = pd.DataFrame({
        "date": test_df.loc[valid_test, "date"].values,
        "model_mu": mu_preds,
        "model_sigma": sigma_preds,
        "actual_tmax": test_df.loc[valid_test, "target_tmax"].values,
    })

    # Verify predictions are reasonable
    pred_mae = float(np.mean(np.abs(predictions["model_mu"] - predictions["actual_tmax"])))
    pred_corr = float(np.corrcoef(predictions["model_mu"], predictions["actual_tmax"])[0, 1])

    print(f"\n  2025 Prediction Quality:")
    print(f"    N predictions: {len(predictions)}")
    print(f"    MAE:          {pred_mae:.2f}F")
    print(f"    Correlation:  {pred_corr:.4f}")
    print(f"    Mean mu:      {predictions['model_mu'].mean():.1f}F")
    print(f"    Mean sigma:   {predictions['model_sigma'].mean():.2f}F")
    print(f"    Mean actual:  {predictions['actual_tmax'].mean():.1f}F")

    predictions.to_csv(output_path, index=False)
    print(f"\n  Saved predictions to {output_path}")

    return predictions


# ===========================================================================
# Step 2: Construct Market Probabilities from Climatological Model
# ===========================================================================

def construct_market_probabilities(kalshi_df, actual_tmax_df):
    """Construct market ex-ante probabilities for Kalshi buckets.

    Since the Kalshi public API only provides settlement prices (not
    historical trading prices), we construct realistic market probabilities
    using a climatological + persistence forecast model. This approximates
    what the Kalshi market would have been pricing.

    The market model uses:
    - Climatological mean for the day of year (from 30-year normals)
    - Yesterday's actual TMAX as persistence input
    - A weighted blend: 40% persistence + 60% climatology
    - Sigma from historical monthly variability

    Parameters
    ----------
    kalshi_df : pd.DataFrame
        Real Kalshi market data with columns: date, direction,
        threshold_low, threshold_high, actual_tmax, actual_outcome.
    actual_tmax_df : pd.DataFrame
        Real GHCN TMAX data with columns: date, tmax_f.

    Returns
    -------
    pd.DataFrame
        Market data with added 'market_prob' column representing
        realistic ex-ante probabilities.
    """
    # NYC Central Park monthly climatological TMAX (30-year normals, approximate)
    clim_mean = {
        1: 39.0, 2: 42.0, 3: 50.0, 4: 62.0, 5: 72.0, 6: 80.0,
        7: 85.0, 8: 84.0, 9: 76.0, 10: 65.0, 11: 54.0, 12: 43.0,
    }
    clim_std = {
        1: 10.0, 2: 10.0, 3: 10.0, 4: 9.0, 5: 8.0, 6: 6.5,
        7: 5.5, 8: 5.5, 9: 6.5, 10: 8.0, 11: 9.5, 12: 10.0,
    }

    # Prepare actual TMAX lookup
    tmax_lookup = {}
    tmax_sorted = actual_tmax_df.sort_values("date")
    for _, row in tmax_sorted.iterrows():
        d = pd.to_datetime(row["date"]).date() if not isinstance(row["date"], date) else row["date"]
        tmax_lookup[d] = row["tmax_f"]

    df = kalshi_df.copy()
    market_probs = []

    for _, row in df.iterrows():
        d = pd.to_datetime(row["date"]).date() if not isinstance(row["date"], date) else row["date"]
        month = d.month

        # Market's forecast: blend of climatology + persistence
        clim_mu = clim_mean[month]
        yesterday = d - timedelta(days=1)
        yesterday_tmax = tmax_lookup.get(yesterday, clim_mu)

        # Interpolate climatology within month for smoothness
        next_month = month % 12 + 1
        frac = (d.day - 1) / 30.0
        smooth_clim_mu = clim_mean[month] * (1 - frac) + clim_mean[next_month] * frac
        smooth_clim_std = clim_std[month] * (1 - frac) + clim_std[next_month] * frac

        # Market forecast: persistence-climatology blend
        market_mu = 0.4 * yesterday_tmax + 0.6 * smooth_clim_mu
        market_sigma = smooth_clim_std * 0.85  # Market is somewhat informed

        # Compute market probability for this bucket
        direction = row["direction"]
        tl = row.get("threshold_low")
        th = row.get("threshold_high")

        if direction == "above" and pd.notna(tl):
            # P(TMAX > threshold_low)
            mp = 1.0 - stats.norm.cdf(float(tl), loc=market_mu, scale=market_sigma)
        elif direction == "below" and pd.notna(th):
            # P(TMAX < threshold_high)
            mp = stats.norm.cdf(float(th), loc=market_mu, scale=market_sigma)
        elif direction == "between" and pd.notna(tl) and pd.notna(th):
            mp = (stats.norm.cdf(float(th), loc=market_mu, scale=market_sigma)
                  - stats.norm.cdf(float(tl), loc=market_mu, scale=market_sigma))
        else:
            mp = 0.5

        # Clip to realistic range
        mp = float(np.clip(mp, 0.02, 0.98))
        market_probs.append(mp)

    df["market_prob"] = market_probs
    return df


# ===========================================================================
# Step 3: Prepare Backtest Data
# ===========================================================================

def prepare_oos_backtest_data(kalshi_df, predictions_df, actual_tmax_df):
    """Prepare the 2025 data for backtesting by merging model predictions
    with Kalshi market structure.

    Parameters
    ----------
    kalshi_df : pd.DataFrame
        Real Kalshi market data with market_prob added.
    predictions_df : pd.DataFrame
        Model predictions with columns: date, model_mu, model_sigma, actual_tmax.
    actual_tmax_df : pd.DataFrame
        Real GHCN TMAX with columns: date, tmax_f.

    Returns
    -------
    pd.DataFrame
        Backtest-ready DataFrame with columns: date, model_prob, market_price,
        actual_outcome, plus metadata columns.
    """
    kalshi = kalshi_df.copy()
    preds = predictions_df.copy()

    # Normalize dates
    kalshi["date"] = pd.to_datetime(kalshi["date"]).dt.date
    preds["date"] = pd.to_datetime(preds["date"]).dt.date

    # Create prediction lookup
    pred_lookup = {}
    for _, row in preds.iterrows():
        pred_lookup[row["date"]] = (row["model_mu"], row["model_sigma"])

    # Compute model probabilities for each Kalshi bucket
    model_probs = []
    for _, row in kalshi.iterrows():
        d = row["date"]
        if d not in pred_lookup:
            model_probs.append(np.nan)
            continue

        mu, sigma = pred_lookup[d]
        sigma = max(sigma, 1e-10)
        direction = row["direction"]
        tl = row.get("threshold_low")
        th = row.get("threshold_high")

        if direction == "above" and pd.notna(tl):
            mp = 1.0 - stats.norm.cdf(float(tl), loc=mu, scale=sigma)
        elif direction == "below" and pd.notna(th):
            mp = stats.norm.cdf(float(th), loc=mu, scale=sigma)
        elif direction == "between" and pd.notna(tl) and pd.notna(th):
            mp = (stats.norm.cdf(float(th), loc=mu, scale=sigma)
                  - stats.norm.cdf(float(tl), loc=mu, scale=sigma))
        else:
            mp = np.nan

        model_probs.append(float(np.clip(mp, 0.001, 0.999)) if not np.isnan(mp) else np.nan)

    kalshi["model_prob"] = model_probs

    # Prepare for BacktestEngine
    # Determine the bucket label column name
    bucket_col = "bucket_label" if "bucket_label" in kalshi.columns else "bucket"
    select_cols = ["date", "model_prob", "market_prob", "actual_outcome",
                   "direction", "threshold_low", "threshold_high",
                   "ticker", "actual_tmax", "volume"]
    if bucket_col in kalshi.columns:
        select_cols.insert(8, bucket_col)
    bt_data = kalshi[select_cols].copy()
    if bucket_col != "bucket_label" and bucket_col in bt_data.columns:
        bt_data = bt_data.rename(columns={bucket_col: "bucket_label"})
    bt_data = bt_data.rename(columns={"market_prob": "market_price"})
    bt_data = bt_data.dropna(subset=["model_prob", "market_price", "actual_outcome"])
    bt_data = bt_data.sort_values("date").reset_index(drop=True)

    return bt_data


# ===========================================================================
# Step 4: Run OOS Backtest with Frozen Strategy
# ===========================================================================

def extract_strategy_params(config_dict):
    """Extract TradingStrategy constructor params from best strategy config."""
    name = config_dict.get("strategy_name", "")

    params = {
        "name": "Best_from_2023_2024",
        "ev_threshold": 0.02,
        "sizing_method": "fractional_kelly",
        "kelly_fraction": 0.10,
        "fee_rate": FEE_RATE,
        "max_position_frac": 0.10,
        "bankroll": BANKROLL,
    }

    ev_match = re.search(r"ev(\d+\.\d+)", name)
    if ev_match:
        params["ev_threshold"] = float(ev_match.group(1))

    for method in ["fractional_kelly", "capped_kelly", "full_kelly",
                    "proportional", "fixed"]:
        if method in name:
            params["sizing_method"] = method
            break

    kf_match = re.search(r"kf(\d+\.\d+)", name)
    if kf_match:
        params["kelly_fraction"] = float(kf_match.group(1))

    fee_match = re.search(r"fee(\d+\.\d+)", name)
    if fee_match:
        params["fee_rate"] = float(fee_match.group(1))

    mp_match = re.search(r"mp(\d+\.\d+)", name)
    if mp_match:
        params["max_position_frac"] = float(mp_match.group(1))

    br_match = re.search(r"br(\d+)", name)
    if br_match:
        params["bankroll"] = float(br_match.group(1))

    return params


def run_frozen_strategy_backtest(backtest_data, strategy_config, output_dir):
    """Run the frozen best strategy from in-sample on 2025 OOS data.

    Parameters
    ----------
    backtest_data : pd.DataFrame
        Prepared backtest data for 2025.
    strategy_config : dict
        Best strategy configuration from in-sample.
    output_dir : str
        Output directory.

    Returns
    -------
    BacktestResult
        Backtest results.
    """
    os.makedirs(output_dir, exist_ok=True)

    strategy_params = extract_strategy_params(strategy_config)
    print(f"\n  Frozen strategy params: {json.dumps(strategy_params, indent=2)}")

    frozen_strategy = TradingStrategy(**strategy_params)
    engine = BacktestEngine(frozen_strategy)
    result = engine.run_backtest(backtest_data)

    print(f"\n  OOS Backtest Results:")
    print(f"    Trades:       {result.n_trades}")
    print(f"    Total P&L:    ${result.total_pnl:.2f}")
    print(f"    ROI:          {result.roi * 100:.1f}%")
    print(f"    Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"    Win Rate:     {result.win_rate * 100:.1f}%")
    print(f"    Max Drawdown: ${result.max_drawdown:.2f}")
    print(f"    Avg EV:       {result.avg_ev:.4f}")

    # Save trade records
    if result.trades:
        pd.DataFrame(result.trades).to_csv(
            os.path.join(output_dir, "oos_backtest_results.csv"), index=False)

    return result


# ===========================================================================
# Step 5: Analysis and Visualization
# ===========================================================================

def generate_oos_plots(result, backtest_data, output_dir):
    """Generate all OOS-specific plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Cumulative P&L curve
    if len(result.cumulative_pnl) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result.cumulative_pnl, linewidth=1.5, color="#d62728")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(range(len(result.cumulative_pnl)),
                        result.cumulative_pnl, 0, alpha=0.1, color="#d62728")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("OOS 2025: Cumulative P&L (Real Data)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "oos_pnl_curve.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot 2: Drawdown
    if len(result.cumulative_pnl) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        running_max = np.maximum.accumulate(result.cumulative_pnl)
        drawdown = running_max - result.cumulative_pnl
        ax.fill_between(range(len(drawdown)), drawdown, alpha=0.5, color="#d62728")
        ax.set_xlabel("Day")
        ax.set_ylabel("Drawdown ($)")
        ax.set_title(f"OOS 2025: Drawdown (Max = ${result.max_drawdown:.2f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "oos_drawdown.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot 3: Monthly P&L
    if result.trades:
        fig, ax = plt.subplots(figsize=(12, 5))
        trade_df = pd.DataFrame(result.trades)
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["month"] = trade_df["date"].dt.to_period("M")
        monthly = trade_df.groupby("month")["pnl"].sum()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
        ax.bar(range(len(monthly)), monthly.values, color=colors)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels([str(m) for m in monthly.index],
                           rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("P&L ($)")
        ax.set_title("OOS 2025: Monthly P&L (Real Data)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "oos_monthly_pnl.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Plot 4: Model vs Market probability scatter
    if "model_prob" in backtest_data.columns and "market_price" in backtest_data.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(backtest_data["market_price"], backtest_data["model_prob"],
                   alpha=0.3, s=10, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Agreement line")
        ax.set_xlabel("Market Probability")
        ax.set_ylabel("Model Probability")
        ax.set_title("OOS 2025: Model vs Market Probability (Real Data)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "oos_model_vs_market_scatter.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Generated OOS plots in {output_dir}")


def generate_reliability_diagram(backtest_data, output_dir, n_bins=10):
    """Generate reliability diagram for model calibration."""
    os.makedirs(output_dir, exist_ok=True)
    cal = CalibrationAnalyzer()

    valid = backtest_data.dropna(subset=["model_prob", "actual_outcome"])
    if len(valid) < 20:
        print("  Warning: insufficient data for reliability diagram")
        return

    cal.plot_reliability_diagram(
        valid["model_prob"], valid["actual_outcome"],
        output_dir,
        title="OOS 2025: Model Calibration (Real Data)",
    )
    print(f"  Saved reliability diagram to {output_dir}")


# ===========================================================================
# Step 6: Combined IS vs OOS Analysis
# ===========================================================================

def run_combined_analysis(is_metrics, oos_metrics, oos_result, oos_backtest_data,
                          is_brier, oos_brier, strategy_config, output_dir):
    """Run Step 10: Combined IS vs OOS analysis.

    Parameters
    ----------
    is_metrics : dict
        In-sample performance metrics.
    oos_metrics : dict
        Out-of-sample performance metrics.
    oos_result : BacktestResult
        OOS backtest result object.
    oos_backtest_data : pd.DataFrame
        OOS backtest data.
    is_brier : dict
        In-sample Brier analysis.
    oos_brier : dict
        OOS Brier analysis.
    strategy_config : dict
        Best strategy configuration.
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to final report.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 10: Combined Analysis - In-Sample vs Out-of-Sample")
    print("=" * 70)

    analyzer = BacktestAnalyzer()

    # Add brier deltas to metrics
    is_m = {**is_metrics}
    oos_m = {**oos_metrics}
    is_m["brier_delta"] = is_brier.get("overall", {}).get("brier_delta",
                           is_metrics.get("brier_delta", float("nan")))
    oos_m["brier_delta"] = oos_brier.get("overall", {}).get("brier_delta",
                            oos_metrics.get("brier_delta", float("nan")))

    # Edge persistence analysis
    persistence_df = analyzer.analyze_edge_persistence(is_m, oos_m)
    persistence_df.to_csv(os.path.join(output_dir, "oos_vs_insample_comparison.csv"),
                          index=False)

    print("\n  Comparison Table:")
    print("  " + "-" * 90)
    print(f"  {'Metric':<18} {'In-Sample':>12} {'OOS':>12} {'Change':>12} {'Verdict'}")
    print("  " + "-" * 90)
    for _, row in persistence_df.iterrows():
        is_v = row["in_sample"]
        oos_v = row["oos"]
        ch = row["change"]
        if row["metric"] in ("sharpe_ratio",):
            is_str = f"{is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("roi", "win_rate"):
            is_str = f"{is_v*100:.1f}%" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v*100:.1f}%" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch*100:+.1f}%" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("total_pnl", "max_drawdown"):
            is_str = f"${is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"${oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"${ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] == "brier_delta":
            is_str = f"{is_v:.4f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.4f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.4f}" if not np.isnan(ch) else "N/A"
        else:
            is_str = f"{is_v}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch}" if not np.isnan(ch) else "N/A"
        print(f"  {row['metric']:<18} {is_str:>12} {oos_str:>12} {ch_str:>12}   {row['verdict']}")
    print("  " + "-" * 90)

    # Determine verdict
    recommendation = analyzer._generate_recommendation(is_m, oos_m)
    print(f"\n  VERDICT: {recommendation['verdict']}")
    print(f"  ACTION:  {recommendation['action']}")
    print(f"\n  {recommendation['summary']}")

    # Generate comparison plots
    _plot_is_vs_oos_comparison(is_metrics, oos_metrics, output_dir)

    # Combined P&L curves (if IS trades available)
    is_trades_path = os.path.join(IS_OUTPUT_DIR, "oos_backtest_results.csv")
    if os.path.exists(is_trades_path):
        is_trades_df = pd.read_csv(is_trades_path)
        _plot_combined_pnl(is_trades_df, oos_result, output_dir)

    # Monthly comparison
    _plot_combined_monthly(oos_result, output_dir)

    # Seasonal analysis
    seasonal = compute_seasonal_pnl(oos_result.trades)
    seasonal_rows = []
    for season in SEASON_ORDER:
        s = seasonal.get(season, {})
        seasonal_rows.append({
            "season": season,
            "oos_pnl": s.get("total_pnl", 0),
            "oos_trades": s.get("n_trades", 0),
            "oos_win_rate": s.get("win_rate", 0),
            "oos_mean_pnl": s.get("mean_pnl", 0),
        })
    seasonal_df = pd.DataFrame(seasonal_rows)
    seasonal_df.to_csv(os.path.join(output_dir, "combined_seasonal_analysis.csv"),
                       index=False)

    # Save trading recommendation
    rec_dict = {
        "verdict": recommendation["verdict"],
        "action": recommendation["action"],
        "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
        "oos_roi": oos_metrics.get("roi", 0),
        "oos_pnl": oos_metrics.get("total_pnl", 0),
        "strategy_config": strategy_config,
        "data_source": "Real GHCN + Kalshi API",
    }
    with open(os.path.join(output_dir, "trading_recommendation.json"), "w") as f:
        json.dump(rec_dict, f, indent=2, default=str)

    # Generate final comprehensive report
    report = generate_final_report(
        is_metrics, oos_metrics, is_brier, oos_brier,
        persistence_df, recommendation, seasonal, strategy_config,
        oos_result, oos_backtest_data, output_dir,
    )

    report_path = os.path.join(output_dir, "final_real_data_backtest_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Final report saved to {report_path}")

    return report_path


def _plot_is_vs_oos_comparison(is_metrics, oos_metrics, output_dir):
    """Plot IS vs OOS metric comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#4c72b0", "#d62728"]
    labels = ["In-Sample", "OOS"]

    # Sharpe
    ax = axes[0]
    vals = [is_metrics.get("sharpe_ratio", 0), oos_metrics.get("sharpe_ratio", 0)]
    ax.bar(labels, vals, color=colors, alpha=0.75)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio Comparison")
    ax.axhline(0, color="black", linewidth=0.5)

    # ROI
    ax = axes[1]
    vals = [is_metrics.get("roi", 0) * 100, oos_metrics.get("roi", 0) * 100]
    ax.bar(labels, vals, color=colors, alpha=0.75)
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI Comparison")
    ax.axhline(0, color="black", linewidth=0.5)

    # Win Rate
    ax = axes[2]
    vals = [is_metrics.get("win_rate", 0) * 100, oos_metrics.get("win_rate", 0) * 100]
    ax.bar(labels, vals, color=colors, alpha=0.75)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate Comparison")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "insample_vs_oos_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_pnl(is_trades_df, oos_result, output_dir):
    """Plot combined IS + OOS P&L curves."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if "pnl" in is_trades_df.columns and len(is_trades_df) > 0:
        is_pnls = np.cumsum(is_trades_df["pnl"].values)
        ax.plot(range(len(is_pnls)), is_pnls,
                label="In-Sample (2023-2024)", linewidth=1.5, color="#4c72b0")

    if oos_result.trades:
        oos_pnls = np.cumsum([t["pnl"] for t in oos_result.trades])
        offset = len(is_trades_df) if "pnl" in is_trades_df.columns else 0
        ax.plot(range(offset, offset + len(oos_pnls)), oos_pnls,
                label="Out-of-Sample (2025)", linewidth=1.5, color="#d62728")
        if offset > 0:
            ax.axvline(offset, color="gray", linestyle="--",
                       linewidth=0.8, label="IS/OOS boundary")

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("In-Sample vs Out-of-Sample: Cumulative P&L (Real Data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_pnl_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_monthly(oos_result, output_dir):
    """Plot OOS monthly P&L."""
    if not oos_result.trades:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    trade_df = pd.DataFrame(oos_result.trades)
    trade_df["date"] = pd.to_datetime(trade_df["date"])
    trade_df["month"] = trade_df["date"].dt.to_period("M")
    monthly = trade_df.groupby("month")["pnl"].sum()
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values, color=colors)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(m) for m in monthly.index],
                       rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("P&L ($)")
    ax.set_title("OOS 2025: Monthly P&L (Real Data)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_monthly_pnl.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Step 7: Final Report Generation
# ===========================================================================

def generate_final_report(is_metrics, oos_metrics, is_brier, oos_brier,
                          persistence_df, recommendation, seasonal,
                          strategy_config, oos_result, oos_backtest_data,
                          output_dir):
    """Generate the comprehensive final backtest report in markdown.

    Parameters
    ----------
    All parameters are analysis results from the pipeline.

    Returns
    -------
    str
        Full markdown report text.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Kalshi KXHIGHNY Real-Data Comprehensive Backtest Report",
        "",
        f"**Generated:** {now_str}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"**Strategy:** {strategy_config.get('strategy_name', 'Best Strategy')}",
        f"**In-Sample Period:** 2023-2024 (simulated market probabilities)",
        f"**Out-of-Sample Period:** 2025 (real GHCN data, real Kalshi structure)",
        "",
        f"**Overall Verdict:** {recommendation['verdict']}",
        "",
        recommendation["summary"],
        "",
        "---",
        "",
        "## 2. Data Sources and Quality Assessment",
        "",
        "| Data Element | Source | Type |",
        "|-------------|--------|------|",
        "| Temperature observations | NOAA GHCN-Daily (USW00094728) | Real |",
        "| Market structure/buckets | Kalshi API (KXHIGHNY settled) | Real |",
        "| Settlement outcomes | Kalshi API (result field) | Real |",
        "| Settlement temperatures | Kalshi API (expiration_value) | Real |",
        "| Model predictions | Ridge regression trained on 2018-2024 GHCN | Real (OOS) |",
        "| Market ex-ante probabilities | Climatological Gaussian model | Constructed |",
        "",
        "**Note on market probabilities:** The Kalshi public API for settled markets",
        "provides only settlement prices (0 or 100 cents), not the historical trading",
        "prices available to participants before the temperature was observed. Market",
        "probabilities are constructed from a climatological + persistence forecast",
        "model, which approximates the market's ex-ante pricing.",
        "",
    ]

    # Data quality
    n_days = oos_backtest_data["date"].nunique() if "date" in oos_backtest_data.columns else 0
    n_records = len(oos_backtest_data)
    lines.extend([
        f"**OOS data quality:** {n_days} trading days, {n_records} market records",
        "",
        "---",
        "",
        "## 3. Model Calibration",
        "",
        "### Brier Score Comparison",
        "",
        "| Period | Model Brier | Market Brier | Delta | Interpretation |",
        "|--------|-------------|--------------|-------|----------------|",
    ])

    is_overall = is_brier.get("overall", {})
    oos_overall = oos_brier.get("overall", {})

    is_mb = is_overall.get("model_brier", is_metrics.get("brier_model", float("nan")))
    is_mkb = is_overall.get("market_brier", is_metrics.get("brier_market", float("nan")))
    is_bd = is_overall.get("brier_delta", is_metrics.get("brier_delta", float("nan")))
    oos_mb = oos_overall.get("model_brier", oos_metrics.get("brier_model", float("nan")))
    oos_mkb = oos_overall.get("market_brier", oos_metrics.get("brier_market", float("nan")))
    oos_bd = oos_overall.get("brier_delta", oos_metrics.get("brier_delta", float("nan")))

    lines.append(
        f"| In-Sample | {is_mb:.4f} | {is_mkb:.4f} | {is_bd:.4f} "
        f"| {'Model better' if is_bd < 0 else 'Market better'} |"
    )
    lines.append(
        f"| OOS | {oos_mb:.4f} | {oos_mkb:.4f} | {oos_bd:.4f} "
        f"| {'Model better' if oos_bd < 0 else 'Market better'} |"
    )

    # Seasonal Brier
    lines.extend([
        "",
        "### Seasonal Brier Breakdown (OOS)",
        "",
        "| Season | Model Brier | Market Brier | Delta | N |",
        "|--------|-------------|--------------|-------|---|",
    ])
    for season in SEASON_ORDER:
        s_data = oos_brier.get("by_season", {}).get(season, {})
        if s_data:
            lines.append(
                f"| {season} | {s_data.get('model_brier', float('nan')):.4f} "
                f"| {s_data.get('market_brier', float('nan')):.4f} "
                f"| {s_data.get('brier_delta', float('nan')):.4f} "
                f"| {s_data.get('n', 0)} |"
            )

    # In-sample results
    lines.extend([
        "",
        "---",
        "",
        "## 4. In-Sample Results (2023-2024)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total P&L | ${is_metrics.get('total_pnl', 0):.2f} |",
        f"| ROI | {is_metrics.get('roi', 0) * 100:.1f}% |",
        f"| Sharpe Ratio | {is_metrics.get('sharpe_ratio', 0):.2f} |",
        f"| Win Rate | {is_metrics.get('win_rate', 0) * 100:.1f}% |",
        f"| Max Drawdown | ${is_metrics.get('max_drawdown', 0):.2f} |",
        f"| Trades | {is_metrics.get('n_trades', 0)} |",
        f"| Avg EV | {is_metrics.get('avg_ev', 0):.4f} |",
    ])

    # OOS results
    lines.extend([
        "",
        "---",
        "",
        "## 5. Out-of-Sample Results (2025)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total P&L | ${oos_metrics.get('total_pnl', 0):.2f} |",
        f"| ROI | {oos_metrics.get('roi', 0) * 100:.1f}% |",
        f"| Sharpe Ratio | {oos_metrics.get('sharpe_ratio', 0):.2f} |",
        f"| Win Rate | {oos_metrics.get('win_rate', 0) * 100:.1f}% |",
        f"| Max Drawdown | ${oos_metrics.get('max_drawdown', 0):.2f} |",
        f"| Trades | {oos_metrics.get('n_trades', 0)} |",
        f"| Avg EV | {oos_metrics.get('avg_ev', 0):.4f} |",
    ])

    # Stability analysis
    lines.extend([
        "",
        "---",
        "",
        "## 6. IS vs OOS Stability Analysis",
        "",
        "| Metric | In-Sample | OOS | Change | Verdict |",
        "|--------|-----------|-----|--------|---------|",
    ])
    for _, row in persistence_df.iterrows():
        is_v = row["in_sample"]
        oos_v = row["oos"]
        ch = row["change"]
        if row["metric"] in ("sharpe_ratio",):
            is_str = f"{is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("roi", "win_rate"):
            is_str = f"{is_v*100:.1f}%" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v*100:.1f}%" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch*100:+.1f}%" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("total_pnl", "max_drawdown"):
            is_str = f"${is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"${oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"${ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] == "brier_delta":
            is_str = f"{is_v:.4f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.4f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.4f}" if not np.isnan(ch) else "N/A"
        else:
            is_str = f"{is_v}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch}" if not np.isnan(ch) else "N/A"
        lines.append(
            f"| {row['metric']} | {is_str} | {oos_str} "
            f"| {ch_str} | {row['verdict']} |"
        )

    # Risk assessment
    lines.extend([
        "",
        "---",
        "",
        "## 7. Risk Assessment",
        "",
        f"- **Max Drawdown (IS):** ${is_metrics.get('max_drawdown', 0):.2f} "
        f"({is_metrics.get('max_drawdown', 0) / BANKROLL * 100:.1f}% of bankroll)",
        f"- **Max Drawdown (OOS):** ${oos_metrics.get('max_drawdown', 0):.2f} "
        f"({oos_metrics.get('max_drawdown', 0) / BANKROLL * 100:.1f}% of bankroll)",
    ])

    if oos_result.trades:
        pnls = np.array([t["pnl"] for t in oos_result.trades])
        var_5 = float(np.percentile(pnls, 5))
        losses = pnls[pnls <= var_5]
        es_5 = float(np.mean(losses)) if len(losses) > 0 else var_5
        lines.extend([
            f"- **Value at Risk (5%):** ${var_5:.2f}",
            f"- **Expected Shortfall (5%):** ${es_5:.2f}",
        ])

    # Seasonal edge analysis
    lines.extend([
        "",
        "---",
        "",
        "## 8. Seasonal Performance (OOS)",
        "",
        "| Season | P&L | Trades | Win Rate |",
        "|--------|-----|--------|----------|",
    ])
    for season in SEASON_ORDER:
        s = seasonal.get(season, {})
        if s:
            lines.append(
                f"| {season} | ${s.get('total_pnl', 0):.2f} "
                f"| {s.get('n_trades', 0)} "
                f"| {s.get('win_rate', 0) * 100:.1f}% |"
            )

    # Trading recommendation
    lines.extend([
        "",
        "---",
        "",
        "## 9. Trading Recommendation",
        "",
        f"**Recommendation:** {recommendation['action']}",
        "",
        recommendation["details"],
        "",
        "### Strategy Configuration",
        "",
        "```json",
        json.dumps(strategy_config, indent=2, default=str),
        "```",
        "",
        "---",
        "",
        "## 10. Methodology Notes",
        "",
        "### Model",
        "- Ridge regression (alpha=1.0) trained on GHCN 2018-2024 data",
        "- 14 surrounding station TMAX lag-1 features + NYC autoregressive + cyclical date",
        "- Seasonally-varying prediction sigma from training residuals",
        "",
        "### Market Probability Construction",
        "- Climatological + persistence Gaussian: 40% yesterday's TMAX + 60% climatology",
        "- Monthly-varying sigma from 30-year NYC climate normals",
        "- This is a limitation: real market prices would reflect more information",
        "",
        "### Backtest Mechanics",
        "- Each Kalshi market is treated as a binary contract",
        "- Model probability vs market probability determines trade direction and EV",
        "- Kelly criterion sizing with frozen parameters from IS optimization",
        "- Fee rate: 7% on winnings",
        "",
    ])

    return "\n".join(lines)


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    """Run the full real-data OOS validation pipeline."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Real-Data Kalshi OOS Validation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data sources: Real GHCN + Real Kalshi API")

    # === Step 1: Load/verify real data ===
    print("\n--- Step 1: Loading real data ---")

    # 1a: Real GHCN TMAX for 2025
    tmax_path = os.path.join(DATA_DIR, "real_central_park_tmax_2025.csv")
    if not os.path.exists(tmax_path):
        print("  ERROR: real_central_park_tmax_2025.csv not found. Run data download first.")
        return
    actual_tmax = pd.read_csv(tmax_path)
    print(f"  Loaded {len(actual_tmax)} GHCN TMAX records for 2025")
    print(f"    Range: {actual_tmax['tmax_f'].min():.1f}F to {actual_tmax['tmax_f'].max():.1f}F")

    # 1b: Real Kalshi market data for 2025
    kalshi_path = os.path.join(DATA_DIR, "real_kalshi_2025.csv")
    if not os.path.exists(kalshi_path):
        print("  ERROR: real_kalshi_2025.csv not found. Run Kalshi API fetch first.")
        return
    kalshi_raw = pd.read_csv(kalshi_path)
    print(f"  Loaded {len(kalshi_raw)} Kalshi market records for 2025")
    print(f"    Unique dates: {kalshi_raw['date'].nunique()}")

    # Filter to only 2025 dates and valid directions
    kalshi_raw["date_dt"] = pd.to_datetime(kalshi_raw["date"])
    kalshi_2025 = kalshi_raw[
        (kalshi_raw["date_dt"].dt.year == 2025) &
        (kalshi_raw["direction"] != "unknown")
    ].copy()
    print(f"    After filtering to 2025 + valid direction: {len(kalshi_2025)} records")

    # Use Kalshi expiration_value as actual_tmax (NWS integer F)
    # For rows missing expiration_value, merge from GHCN
    kalshi_2025["date_str"] = kalshi_2025["date_dt"].dt.strftime("%Y-%m-%d")
    actual_tmax["date_str"] = pd.to_datetime(actual_tmax["date"]).dt.strftime("%Y-%m-%d")
    tmax_map = dict(zip(actual_tmax["date_str"], actual_tmax["tmax_f"]))

    kalshi_2025["actual_tmax_ghcn"] = kalshi_2025["date_str"].map(tmax_map)
    # Use Kalshi settlement temperature where available, else GHCN
    kalshi_2025["actual_tmax"] = kalshi_2025["actual_tmax"].fillna(
        kalshi_2025["actual_tmax_ghcn"]
    )

    # Recompute actual_outcome based on real temperatures
    outcomes = []
    for _, row in kalshi_2025.iterrows():
        actual = row["actual_tmax"]
        if pd.isna(actual):
            outcomes.append(np.nan)
            continue
        direction = row["direction"]
        tl = row.get("threshold_low")
        th = row.get("threshold_high")
        if direction == "above" and pd.notna(tl):
            outcomes.append(1 if actual > float(tl) else 0)
        elif direction == "below" and pd.notna(th):
            outcomes.append(1 if actual < float(th) else 0)
        elif direction == "between" and pd.notna(tl) and pd.notna(th):
            outcomes.append(1 if float(tl) <= actual <= float(th) else 0)
        else:
            outcomes.append(np.nan)
    kalshi_2025["actual_outcome"] = outcomes

    # Drop rows with missing outcomes
    kalshi_2025 = kalshi_2025.dropna(subset=["actual_outcome"]).copy()
    kalshi_2025["actual_outcome"] = kalshi_2025["actual_outcome"].astype(int)
    print(f"    After outcome validation: {len(kalshi_2025)} records")

    # === Step 2: Generate real model predictions ===
    pred_path = os.path.join(DATA_DIR, "real_model_predictions_2025.csv")
    if os.path.exists(pred_path):
        predictions = pd.read_csv(pred_path)
        # Verify it has realistic precision (not synthetic)
        sample_actual = predictions["actual_tmax"].iloc[0]
        if len(str(sample_actual).split(".")[-1]) > 4:
            print("  WARNING: Existing predictions appear synthetic. Regenerating...")
            predictions = generate_real_predictions_2025(pred_path)
        else:
            print(f"  Loaded existing real predictions: {len(predictions)} days")
    else:
        predictions = generate_real_predictions_2025(pred_path)

    # === Step 3: Construct market probabilities ===
    print("\n--- Step 3: Constructing market probabilities ---")
    kalshi_with_market = construct_market_probabilities(kalshi_2025, actual_tmax)
    print(f"  Market prob range: {kalshi_with_market['market_prob'].min():.3f} to "
          f"{kalshi_with_market['market_prob'].max():.3f}")

    # Save processed market data
    kalshi_with_market.to_csv(
        os.path.join(OOS_OUTPUT_DIR, "market_data_2025.csv"), index=False)

    # === Step 4: Prepare backtest data ===
    print("\n--- Step 4: Preparing backtest data ---")
    os.makedirs(OOS_OUTPUT_DIR, exist_ok=True)
    backtest_data = prepare_oos_backtest_data(kalshi_with_market, predictions, actual_tmax)
    print(f"  Backtest data: {len(backtest_data)} records across "
          f"{backtest_data['date'].nunique()} days")

    # Save model predictions alongside market data
    predictions.to_csv(
        os.path.join(OOS_OUTPUT_DIR, "model_predictions_2025.csv"), index=False)

    # === Step 5: Load IS best strategy config ===
    print("\n--- Step 5: Loading in-sample best strategy ---")
    is_config_path = os.path.join(IS_OUTPUT_DIR, "best_strategy_config.json")
    if not os.path.exists(is_config_path):
        print(f"  WARNING: {is_config_path} not found.")
        print("  Using default strategy parameters.")
        strategy_config = {
            "strategy_name": "S0396_ev0.15_proportional_kf0.05_fee0.07_mp0.05_br10000",
            "ev_threshold": 0.15,
            "sizing_method": "proportional",
            "kelly_fraction": 0.05,
            "fee_rate": 0.07,
            "max_position_frac": 0.05,
            "bankroll": 10000,
        }
    else:
        with open(is_config_path) as f:
            strategy_config = json.load(f)
        print(f"  Loaded: {strategy_config.get('strategy_name', 'Unknown')}")
        print(f"  Selection reason: {strategy_config.get('selection_reason', 'N/A')}")

    # === Step 6: Run frozen strategy backtest ===
    print("\n--- Step 6: Running frozen strategy on 2025 OOS data ---")
    oos_result = run_frozen_strategy_backtest(backtest_data, strategy_config, OOS_OUTPUT_DIR)

    # === Step 7: Brier score analysis ===
    print("\n--- Step 7: Computing Brier score analysis ---")
    bt_analyzer = BacktestAnalyzer()

    # Prepare comparison data for Brier analysis
    comparison_for_brier = backtest_data.rename(columns={"market_price": "market_prob"})
    oos_brier = bt_analyzer.analyze_brier_scores(comparison_for_brier)

    brier_overall = oos_brier.get("overall", {})
    print(f"  Model Brier:  {brier_overall.get('model_brier', float('nan')):.4f}")
    print(f"  Market Brier: {brier_overall.get('market_brier', float('nan')):.4f}")
    print(f"  Delta:        {brier_overall.get('brier_delta', float('nan')):.4f}")

    with open(os.path.join(OOS_OUTPUT_DIR, "oos_brier_analysis.json"), "w") as f:
        json.dump(oos_brier, f, indent=2, default=str)

    # Seasonal calibration
    cal_analyzer = CalibrationAnalyzer()
    seasonal_cal = cal_analyzer.compute_seasonal_calibration(comparison_for_brier)
    with open(os.path.join(OOS_OUTPUT_DIR, "oos_seasonal_calibration.json"), "w") as f:
        json.dump(seasonal_cal, f, indent=2, default=str)

    # Save OOS metrics
    oos_metrics = oos_result.to_summary_dict()
    oos_metrics["period"] = "2025"
    oos_metrics["bankroll"] = BANKROLL
    oos_metrics["brier_model"] = brier_overall.get("model_brier", float("nan"))
    oos_metrics["brier_market"] = brier_overall.get("market_brier", float("nan"))
    oos_metrics["brier_delta"] = brier_overall.get("brier_delta", float("nan"))
    with open(os.path.join(OOS_OUTPUT_DIR, "oos_metrics.json"), "w") as f:
        json.dump(oos_metrics, f, indent=2, default=str)

    # === Step 8: Generate OOS plots ===
    print("\n--- Step 8: Generating OOS plots ---")
    generate_oos_plots(oos_result, backtest_data, OOS_OUTPUT_DIR)
    generate_reliability_diagram(comparison_for_brier, OOS_OUTPUT_DIR)

    # Seasonal P&L
    seasonal = compute_seasonal_pnl(oos_result.trades)
    if seasonal:
        seasonal_df = pd.DataFrame([
            {"season": k, **v} for k, v in seasonal.items()
        ])
        seasonal_df.to_csv(
            os.path.join(OOS_OUTPUT_DIR, "oos_seasonal_performance.csv"), index=False)

    # === Step 9: Load IS results for comparison ===
    print("\n--- Step 9: Loading in-sample results for comparison ---")
    is_metrics_path = os.path.join(IS_OUTPUT_DIR, "oos_metrics.json")
    is_brier_path = os.path.join(IS_OUTPUT_DIR, "brier_analysis.json")

    # Load IS metrics from the available files
    if os.path.exists(is_metrics_path):
        with open(is_metrics_path) as f:
            is_metrics = json.load(f)
        print(f"  Loaded IS metrics from oos_metrics.json")
    else:
        # Try the best strategy config as fallback
        is_metrics = {
            "sharpe_ratio": strategy_config.get("sharpe_ratio", 0),
            "roi": strategy_config.get("roi", 0),
            "total_pnl": strategy_config.get("total_pnl", 0),
            "win_rate": strategy_config.get("win_rate", 0),
            "max_drawdown": strategy_config.get("max_drawdown", 0),
            "n_trades": strategy_config.get("n_trades", 0),
            "avg_ev": strategy_config.get("avg_ev", 0),
            "period": "2023-2024",
            "bankroll": BANKROLL,
        }
        print(f"  Used best_strategy_config.json for IS metrics")

    if os.path.exists(is_brier_path):
        with open(is_brier_path) as f:
            is_brier = json.load(f)
    else:
        # Try brier_comparison.json
        alt_brier = os.path.join(IS_OUTPUT_DIR, "brier_comparison.json")
        if os.path.exists(alt_brier):
            with open(alt_brier) as f:
                is_brier = json.load(f)
        else:
            is_brier = {"overall": {}}

    # === Step 10: Combined IS vs OOS analysis ===
    report_path = run_combined_analysis(
        is_metrics, oos_metrics, oos_result, backtest_data,
        is_brier, oos_brier, strategy_config, COMBINED_OUTPUT_DIR,
    )

    # === Final summary ===
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE -- REAL DATA OOS VALIDATION")
    print("=" * 70)

    print("\nOutput directories:")
    for d in [OOS_OUTPUT_DIR, COMBINED_OUTPUT_DIR]:
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            print(f"\n  {d}/")
            for f_name in files:
                fpath = os.path.join(d, f_name)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    {f_name} ({size_kb:.1f} KB)")

    print(f"\n  Final report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
