#!/usr/bin/env python3
"""
Max-Training Kalshi KXHIGHNY Backtest with Enhanced Market Proxy.

Runs a comprehensive IS/OOS backtest using:
  1. The EnhancedMarketProxy (Ridge with lag1-3, rolling7d, climatology mean/std)
  2. Real Kalshi market data (2023-2024 IS, 2025 OOS)
  3. Max-training model predictions from Analyst 1:
       IS model: trained 1998-2020, validated 2021-2022, predicted 2023-2024
       OOS model: trained 1998-2022, validated 2023-2024, predicted 2025
  4. NN vs Ridge comparison
  5. Comprehensive Brier score analysis (model vs proxy vs Kalshi)
  6. Comprehensive strategy grid search with IS selection -> OOS validation

Key improvement: Enhanced proxy uses 6 features vs 3 in the base proxy.
  - lag1, lag2, lag3 TMAX
  - 7-day rolling mean
  - smooth climatology mean and std
  - Ridge regression with cross-validated alpha

Outputs:
  results/kalshi_max_train_backtest/ (all results, plots, reports)

Usage:
    python scripts/run_max_train_backtest.py
"""

import json
import os
import re
import sys
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from src.enhanced_market_proxy import EnhancedMarketProxy
from src.market_proxy import MarketProxy, NaiveMarketProxy
from src.kalshi_client import build_historical_comparison, compute_brier_scores
from src.kalshi_backtester import CalibrationAnalyzer, compute_seasonal_pnl
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    BacktestResult,
    generate_strategy_grid,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "kalshi_max_train_backtest")

# Max-training prediction files (from Analyst 1)
MAX_TRAIN_NN_IS = os.path.join(DATA_DIR, "max_train_nn_predictions_is.csv")
MAX_TRAIN_NN_OOS = os.path.join(DATA_DIR, "max_train_nn_predictions_oos.csv")
MAX_TRAIN_RIDGE_IS = os.path.join(DATA_DIR, "max_train_ridge_predictions_is.csv")
MAX_TRAIN_RIDGE_OOS = os.path.join(DATA_DIR, "max_train_ridge_predictions_oos.csv")

# Fallback prediction files
EXPANDED_PRED_IS = os.path.join(DATA_DIR, "expanded_model_predictions_2023_2024.csv")
EXPANDED_PRED_OOS = os.path.join(DATA_DIR, "expanded_model_predictions_2025.csv")

# Real Kalshi market data
KALSHI_IS = os.path.join(DATA_DIR, "real_kalshi_2023_2024.csv")
KALSHI_OOS = os.path.join(DATA_DIR, "real_kalshi_2025.csv")

# Central Park TMAX full history
TMAX_HISTORY = os.path.join(DATA_DIR, "central_park_tmax_full_history.csv")

# Strategy grid
STRATEGY_GRID = {
    "ev_thresholds": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "sizing_methods": ["fixed", "proportional", "fractional_kelly", "capped_kelly"],
    "kelly_fractions": [0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
    "fee_rates": [0.07],
    "max_positions": [0.05, 0.10, 0.15, 0.20],
    "bankrolls": [10000],
}

SELECTION_CRITERIA = {
    "min_trades": 100,
    "min_win_rate": 0.30,
    "min_sharpe": 1.5,
    "max_drawdown": 2000,
}


# ===========================================================================
# Helpers
# ===========================================================================
def _get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


SEASON_MAP_FULL = {
    12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
    3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
    6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
    9: "Fall (SON)", 10: "Fall (SON)", 11: "Fall (SON)",
}
SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]


# ===========================================================================
# Step 1: Check and Load Data
# ===========================================================================

def check_prediction_files():
    """Check if max-training prediction files exist.

    Returns True if at least NN IS + NN OOS are available, or
    if fallback expanded predictions are available.
    """
    nn_ready = os.path.exists(MAX_TRAIN_NN_IS) and os.path.exists(MAX_TRAIN_NN_OOS)
    fallback_ready = os.path.exists(EXPANDED_PRED_IS) and os.path.exists(EXPANDED_PRED_OOS)
    return nn_ready or fallback_ready


def load_nn_predictions(period="IS"):
    """Load NN predictions, falling back to expanded predictions."""
    if period == "IS":
        primary = MAX_TRAIN_NN_IS
        fallback = EXPANDED_PRED_IS
    else:
        primary = MAX_TRAIN_NN_OOS
        fallback = EXPANDED_PRED_OOS

    if os.path.exists(primary):
        df = pd.read_csv(primary)
        logger.info("Loaded max-train NN predictions (%s): %d rows from %s",
                     period, len(df), primary)
        return df, "max_train_nn"
    elif os.path.exists(fallback):
        df = pd.read_csv(fallback)
        logger.info("Fallback to expanded predictions (%s): %d rows from %s",
                     period, len(df), fallback)
        return df, "expanded_nn"
    else:
        return None, None


def load_ridge_predictions(period="IS"):
    """Load Ridge predictions if available."""
    if period == "IS":
        path = MAX_TRAIN_RIDGE_IS
    else:
        path = MAX_TRAIN_RIDGE_OOS

    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("Loaded max-train Ridge predictions (%s): %d rows from %s",
                     period, len(df), path)
        return df, "max_train_ridge"
    else:
        return None, None


def load_kalshi_data(period="IS"):
    """Load real Kalshi market data."""
    path = KALSHI_IS if period == "IS" else KALSHI_OOS
    if not os.path.exists(path):
        logger.error("Kalshi data not found: %s", path)
        return None
    df = pd.read_csv(path)
    logger.info("Loaded Kalshi %s: %d rows, %d days", period, len(df), df["date"].nunique())
    return df


def load_tmax_history():
    """Load Central Park full TMAX history."""
    if not os.path.exists(TMAX_HISTORY):
        logger.error("TMAX history not found: %s", TMAX_HISTORY)
        return None
    df = pd.read_csv(TMAX_HISTORY)
    logger.info("Loaded TMAX history: %d rows", len(df))
    return df


# ===========================================================================
# Step 2: Add Proxy Probabilities
# ===========================================================================

def add_enhanced_proxy_probabilities(kalshi_df, proxy, tmax_history_df):
    """Add enhanced proxy probabilities to Kalshi data.

    For each market row, looks up lag-1, lag-2, lag-3 TMAX and 7-day
    rolling mean from history, then computes the proxy bracket probability.
    """
    hist = tmax_history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    tmax_lookup = dict(zip(hist["date"], hist["tmax_f"]))

    result = kalshi_df.copy()
    result["date_obj"] = pd.to_datetime(result["date"]).dt.date

    proxy_probs = []
    for _, row in result.iterrows():
        target_date = row["date_obj"]
        d1 = target_date - timedelta(days=1)
        d2 = target_date - timedelta(days=2)
        d3 = target_date - timedelta(days=3)

        yt = tmax_lookup.get(d1)
        dbt = tmax_lookup.get(d2)
        tat = tmax_lookup.get(d3)

        if yt is None:
            proxy_probs.append(np.nan)
            continue

        # Compute 7-day rolling mean
        recent = []
        for i in range(1, 8):
            d = target_date - timedelta(days=i)
            v = tmax_lookup.get(d)
            if v is not None:
                recent.append(v)
        r7 = np.mean(recent) if len(recent) >= 3 else None

        try:
            prob = proxy.compute_bracket_prob(
                target_date=target_date,
                yesterday_tmax=yt,
                threshold_low=row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                threshold_high=row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                direction=row.get("direction", "between"),
                day_before_tmax=dbt,
                three_days_ago_tmax=tat,
                rolling_7d_mean=r7,
            )
            proxy_probs.append(prob)
        except Exception as e:
            logger.warning("Proxy error for %s: %s", target_date, e)
            proxy_probs.append(np.nan)

    result["proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])

    n_valid = sum(1 for p in proxy_probs if p is not None and not np.isnan(p))
    logger.info("Added enhanced proxy: %d/%d valid", n_valid, len(result))
    return result


def add_base_proxy_probabilities(kalshi_df, proxy, tmax_history_df):
    """Add base MarketProxy probabilities (for comparison)."""
    hist = tmax_history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    tmax_lookup = dict(zip(hist["date"], hist["tmax_f"]))

    result = kalshi_df.copy()
    result["date_obj"] = pd.to_datetime(result["date"]).dt.date

    proxy_probs = []
    for _, row in result.iterrows():
        target_date = row["date_obj"]
        d1 = target_date - timedelta(days=1)
        d2 = target_date - timedelta(days=2)

        yt = tmax_lookup.get(d1)
        dbt = tmax_lookup.get(d2)

        if yt is None:
            proxy_probs.append(np.nan)
            continue

        try:
            prob = proxy.compute_bracket_prob(
                target_date=target_date,
                yesterday_tmax=yt,
                threshold_low=row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                threshold_high=row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                direction=row.get("direction", "between"),
                day_before_tmax=dbt,
            )
            proxy_probs.append(prob)
        except Exception as e:
            logger.warning("Base proxy error for %s: %s", target_date, e)
            proxy_probs.append(np.nan)

    result["base_proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])
    return result


# ===========================================================================
# Step 3: Align Model Predictions
# ===========================================================================

def align_model_and_market(market_df, predictions_df):
    """Align model predictions with market data."""
    api_outcomes = market_df[["date", "ticker", "actual_outcome"]].copy()
    market_clean = market_df.drop(columns=["actual_tmax"], errors="ignore")

    comparison_df = build_historical_comparison(
        model_predictions_df=predictions_df,
        historical_markets_df=market_clean,
    )

    if comparison_df.empty:
        raise ValueError("No overlapping dates between model and market data")

    if "actual_outcome" in comparison_df.columns:
        comparison_df["outcome"] = comparison_df["actual_outcome"].astype(float)

    logger.info(
        "Aligned %d rows across %d dates",
        len(comparison_df), comparison_df["date"].nunique(),
    )
    return comparison_df


# ===========================================================================
# Step 4: Strategy Grid Search
# ===========================================================================

def run_strategy_grid_search(comparison_df, proxy_col="proxy_prob"):
    """Run strategy grid search on IS data."""
    backtest_data = comparison_df.copy()
    backtest_data["market_price"] = backtest_data[proxy_col]

    if "outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["outcome"].astype(float)

    required = ["date", "model_prob", "market_price", "actual_outcome"]
    missing = [c for c in required if c not in backtest_data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keep_cols = required + [c for c in backtest_data.columns
                            if c not in required and c in ["ticker", "bucket", "direction"]]
    before = len(backtest_data)
    backtest_data = backtest_data[keep_cols].dropna(subset=required).reset_index(drop=True)
    logger.info("Backtest data: %d rows (dropped %d NaN)", len(backtest_data), before - len(backtest_data))

    strategies = generate_strategy_grid(**STRATEGY_GRID)
    logger.info("Generated %d strategy permutations", len(strategies))

    all_results = []
    for strategy in strategies:
        engine = BacktestEngine(strategy)
        result = engine.run_backtest(backtest_data)
        all_results.append(result)

    all_results.sort(key=lambda r: r.total_pnl, reverse=True)
    rows = [r.to_summary_dict() for r in all_results]
    strategies_df = pd.DataFrame(rows)

    logger.info(
        "Grid search: %d strategies, %d with trades, %d profitable",
        len(all_results),
        sum(1 for r in all_results if r.n_trades > 0),
        sum(1 for r in all_results if r.total_pnl > 0),
    )
    return all_results, strategies_df, backtest_data


# ===========================================================================
# Step 5: Select Best Strategy
# ===========================================================================

def select_best_strategy(strategies_df):
    """Select best strategy from IS grid search."""
    if strategies_df.empty:
        return {"error": "No strategies"}, None

    crit = SELECTION_CRITERIA

    candidates = strategies_df[
        (strategies_df["n_trades"] >= crit["min_trades"])
        & (strategies_df["win_rate"] > crit["min_win_rate"])
        & (strategies_df["sharpe_ratio"] >= crit["min_sharpe"])
        & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        & (strategies_df["max_drawdown"] < crit["max_drawdown"])
    ].copy()

    reason = "primary_criteria"

    if candidates.empty:
        logger.warning("No strategies meet primary criteria. Relaxing...")
        candidates = strategies_df[
            (strategies_df["n_trades"] >= 50)
            & (strategies_df["win_rate"] > 0.25)
            & (strategies_df["sharpe_ratio"] >= 0.5)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            & (strategies_df["max_drawdown"] < 3000.0)
        ].copy()
        reason = "relaxed_criteria"

    if candidates.empty:
        valid = strategies_df[
            (strategies_df["n_trades"] > 0)
            & (strategies_df["total_pnl"] > 0)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        ]
        if valid.empty:
            valid = strategies_df[
                (strategies_df["n_trades"] > 0)
                & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            ]
        if valid.empty:
            return {"error": "No viable strategies"}, None
        best_row = valid.loc[valid["sharpe_ratio"].idxmax()]
        reason = "fallback_best_sharpe"
        return _config_from_row(best_row, reason), best_row

    top_sharpe = candidates["sharpe_ratio"].max()
    if top_sharpe > 2.0:
        near_top = candidates[candidates["sharpe_ratio"] >= top_sharpe - 0.2]
        if len(near_top) > 1:
            best_row = near_top.loc[near_top["total_pnl"].idxmax()]
            return _config_from_row(best_row, "best_pnl_near_top_sharpe"), best_row

    best_row = candidates.loc[candidates["sharpe_ratio"].idxmax()]
    return _config_from_row(best_row, reason), best_row


def _config_from_row(row, reason=""):
    """Extract strategy config from DataFrame row."""
    name = row["strategy_name"]
    config = {
        "strategy_name": name,
        "total_pnl": float(row["total_pnl"]),
        "roi": float(row["roi"]),
        "sharpe_ratio": float(row["sharpe_ratio"]),
        "max_drawdown": float(row["max_drawdown"]),
        "win_rate": float(row["win_rate"]),
        "n_trades": int(row["n_trades"]),
        "avg_ev": float(row.get("avg_ev", 0)),
        "selection_reason": reason,
    }

    for pat, key in [
        (r"ev(\d+\.\d+)", "ev_threshold"),
        (r"_(\w+)_kf", "sizing_method"),
        (r"kf(\d+\.\d+)", "kelly_fraction"),
        (r"fee(\d+\.\d+)", "fee_rate"),
        (r"mp(\d+\.\d+)", "max_position_frac"),
        (r"br(\d+)", "bankroll"),
    ]:
        m = re.search(pat, name)
        if m:
            val = m.group(1)
            try:
                config[key] = float(val) if "." in val else int(val)
            except ValueError:
                config[key] = val

    return config


def reconstruct_strategy(config):
    """Reconstruct TradingStrategy from config."""
    return TradingStrategy(
        name=config.get("strategy_name", "OOS_Frozen"),
        ev_threshold=config.get("ev_threshold", 0.02),
        sizing_method=config.get("sizing_method", "fractional_kelly"),
        kelly_fraction=config.get("kelly_fraction", 0.25),
        max_position_frac=config.get("max_position_frac", 0.10),
        fee_rate=config.get("fee_rate", 0.07),
        bankroll=config.get("bankroll", 10000),
    )


# ===========================================================================
# Step 6: OOS Backtest
# ===========================================================================

def run_oos_backtest(comparison_df, strategy_config, proxy_col="proxy_prob"):
    """Run frozen strategy on OOS data."""
    strategy = reconstruct_strategy(strategy_config)

    backtest_data = comparison_df.copy()
    backtest_data["market_price"] = backtest_data[proxy_col]

    if "outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["outcome"].astype(float)

    required = ["date", "model_prob", "market_price", "actual_outcome"]
    keep_cols = required + [c for c in backtest_data.columns
                            if c not in required and c in ["ticker", "bucket", "direction"]]
    backtest_data = backtest_data[keep_cols].dropna(subset=required).reset_index(drop=True)

    engine = BacktestEngine(strategy)
    result = engine.run_backtest(backtest_data)

    logger.info(
        "OOS backtest: %d trades, PnL=$%.2f, Sharpe=%.2f, WR=%.1f%%",
        result.n_trades, result.total_pnl, result.sharpe_ratio, result.win_rate * 100,
    )
    return result, backtest_data


# ===========================================================================
# Step 7: Comprehensive Brier Score Analysis
# ===========================================================================

def compute_all_brier_scores(comparison_df, proxy_col="proxy_prob",
                              base_proxy_col="base_proxy_prob",
                              naive_col="naive_proxy_prob"):
    """Compute comprehensive Brier score analysis.

    Compares: model vs enhanced proxy, model vs base proxy,
    model vs naive proxy, model vs Kalshi market.
    """
    df = comparison_df.dropna(subset=["model_prob", proxy_col, "outcome"]).copy()
    results = {}

    if len(df) > 0:
        results["model_vs_enhanced_proxy"] = compute_brier_scores(
            df["model_prob"], df[proxy_col], df["outcome"],
        )

    if base_proxy_col in df.columns:
        df_base = df.dropna(subset=[base_proxy_col])
        if len(df_base) > 0:
            results["model_vs_base_proxy"] = compute_brier_scores(
                df_base["model_prob"], df_base[base_proxy_col], df_base["outcome"],
            )

    if naive_col in df.columns:
        df_naive = df.dropna(subset=[naive_col])
        if len(df_naive) > 0:
            results["model_vs_naive_proxy"] = compute_brier_scores(
                df_naive["model_prob"], df_naive[naive_col], df_naive["outcome"],
            )

    if "market_prob" in df.columns:
        df_mkt = df.dropna(subset=["market_prob"])
        if len(df_mkt) > 0:
            results["model_vs_kalshi_market"] = compute_brier_scores(
                df_mkt["model_prob"], df_mkt["market_prob"], df_mkt["outcome"],
            )

    # Seasonal breakdown
    df["_date"] = pd.to_datetime(df["date"])
    df["_month"] = df["_date"].dt.month
    df["_season"] = df["_month"].apply(_get_season)

    seasonal = {}
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        mask = df["_season"] == season
        if mask.sum() < 10:
            continue
        s_df = df[mask]
        entry = {
            "model_brier": float(np.mean((s_df["model_prob"] - s_df["outcome"]) ** 2)),
            "enhanced_proxy_brier": float(np.mean((s_df[proxy_col] - s_df["outcome"]) ** 2)),
            "n": int(mask.sum()),
        }
        if "market_prob" in s_df.columns:
            s_mkt = s_df.dropna(subset=["market_prob"])
            if len(s_mkt) > 0:
                entry["kalshi_brier"] = float(np.mean((s_mkt["market_prob"] - s_mkt["outcome"]) ** 2))
        seasonal[season] = entry
    results["seasonal"] = seasonal

    # Monthly breakdown
    monthly = {}
    for month in range(1, 13):
        mask = df["_month"] == month
        if mask.sum() < 5:
            continue
        m_df = df[mask]
        entry = {
            "model_brier": float(np.mean((m_df["model_prob"] - m_df["outcome"]) ** 2)),
            "enhanced_proxy_brier": float(np.mean((m_df[proxy_col] - m_df["outcome"]) ** 2)),
            "n": int(mask.sum()),
        }
        if "market_prob" in m_df.columns:
            m_mkt = m_df.dropna(subset=["market_prob"])
            if len(m_mkt) > 0:
                entry["kalshi_brier"] = float(np.mean((m_mkt["market_prob"] - m_mkt["outcome"]) ** 2))
        monthly[str(month)] = entry
    results["monthly"] = monthly

    return results


# ===========================================================================
# Step 8: Generate Plots
# ===========================================================================

def generate_plots(comparison_df_is, comparison_df_oos, is_result, oos_result,
                   brier_is, brier_oos, output_dir,
                   nn_vs_ridge=None):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # ---- 1. Model vs Proxy Scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [
        (axes[0], comparison_df_is, "IS (2023-2024)"),
        (axes[1], comparison_df_oos, "OOS (2025)"),
    ]:
        if df is not None and "model_prob" in df.columns and "proxy_prob" in df.columns:
            valid = df.dropna(subset=["model_prob", "proxy_prob"])
            if len(valid) > 0:
                ax.scatter(valid["proxy_prob"], valid["model_prob"],
                           alpha=0.3, s=10, edgecolors="none", c="#1f77b4")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax.set_xlabel("Enhanced Proxy Probability")
                ax.set_ylabel("Model Probability")
                ax.set_title(f"Model vs Enhanced Proxy: {title}")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_vs_proxy_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. Combined PnL Curves ----
    fig, ax = plt.subplots(figsize=(14, 7))
    is_pnls = None
    if is_result and is_result.trades:
        is_pnls = np.cumsum([t["pnl"] for t in is_result.trades])
        ax.plot(range(len(is_pnls)), is_pnls,
                label=f"IS: PnL=${is_result.total_pnl:.0f}, Sharpe={is_result.sharpe_ratio:.2f}",
                linewidth=1.5, color="#4c72b0")
    if oos_result and oos_result.trades:
        oos_pnls = np.cumsum([t["pnl"] for t in oos_result.trades])
        offset = len(is_pnls) if is_pnls is not None else 0
        ax.plot(range(offset, offset + len(oos_pnls)), oos_pnls,
                label=f"OOS: PnL=${oos_result.total_pnl:.0f}, Sharpe={oos_result.sharpe_ratio:.2f}",
                linewidth=1.5, color="#d62728")
        if offset > 0:
            ax.axvline(offset, color="gray", linestyle="--", linewidth=0.8,
                       label="IS/OOS boundary")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("IS vs OOS: Cumulative P&L (Enhanced Proxy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "combined_pnl_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 3. Monthly PnL Breakdown ----
    for result, label, slug in [
        (is_result, "IS (2023-2024)", "is"),
        (oos_result, "OOS (2025)", "oos"),
    ]:
        if result and result.trades:
            trade_df = pd.DataFrame(result.trades)
            trade_df["date"] = pd.to_datetime(trade_df["date"])
            trade_df["month"] = trade_df["date"].dt.to_period("M")
            monthly_pnl = trade_df.groupby("month")["pnl"].sum()

            fig, ax = plt.subplots(figsize=(14, 5))
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly_pnl.values]
            ax.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors)
            ax.set_xticks(range(len(monthly_pnl)))
            ax.set_xticklabels([str(m) for m in monthly_pnl.index],
                               rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("P&L ($)")
            ax.set_title(f"Monthly P&L: {label}")
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            path = os.path.join(output_dir, f"monthly_pnl_{slug}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # ---- 4. Brier Score Comparison (Bar Chart) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, brier, period_label in [
        (axes[0], brier_is, "IS (2023-2024)"),
        (axes[1], brier_oos, "OOS (2025)"),
    ]:
        if brier is None:
            continue
        labels = []
        model_scores = []
        comp_scores = []

        for key, nice_label in [
            ("model_vs_enhanced_proxy", "Enhanced Proxy"),
            ("model_vs_base_proxy", "Base Proxy"),
            ("model_vs_naive_proxy", "Naive Proxy"),
            ("model_vs_kalshi_market", "Kalshi Market"),
        ]:
            bdata = brier.get(key, {})
            if bdata and "model_brier" in bdata:
                labels.append(nice_label)
                model_scores.append(bdata["model_brier"])
                comp_scores.append(bdata["market_brier"])

        if labels:
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, model_scores, width, label="NN Model", color="#4c72b0", alpha=0.8)
            ax.bar(x + width / 2, comp_scores, width, label="Comparison", color="#d62728", alpha=0.8)
            ax.set_ylabel("Brier Score (lower = better)")
            ax.set_title(f"Brier Score: {period_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "brier_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 5. OOS Calibration / Reliability Diagram ----
    if comparison_df_oos is not None:
        valid_oos = comparison_df_oos.dropna(subset=["model_prob", "outcome"])
        if len(valid_oos) > 50:
            try:
                cal_analyzer = CalibrationAnalyzer()
                cal_path = cal_analyzer.plot_reliability_diagram(
                    valid_oos["model_prob"], valid_oos["outcome"],
                    output_dir=output_dir,
                    title="OOS Model Calibration: Reliability Diagram",
                )
                saved.append(cal_path)
            except Exception as e:
                logger.warning("Reliability diagram failed: %s", e)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ===========================================================================
# Step 9: NN vs Ridge Comparison
# ===========================================================================

def compare_nn_vs_ridge(nn_comparison, ridge_comparison, period_label, output_dir):
    """Compare NN and Ridge model predictions head-to-head."""
    if nn_comparison is None or ridge_comparison is None:
        return None

    nn_df = nn_comparison.dropna(subset=["model_prob", "outcome"]).copy()
    ridge_df = ridge_comparison.dropna(subset=["model_prob", "outcome"]).copy()

    nn_brier = float(np.mean((nn_df["model_prob"] - nn_df["outcome"]) ** 2)) if len(nn_df) > 0 else np.nan
    ridge_brier = float(np.mean((ridge_df["model_prob"] - ridge_df["outcome"]) ** 2)) if len(ridge_df) > 0 else np.nan

    result = {
        "period": period_label,
        "nn_brier": nn_brier,
        "ridge_brier": ridge_brier,
        "nn_n": len(nn_df),
        "ridge_n": len(ridge_df),
        "brier_delta": nn_brier - ridge_brier if not (np.isnan(nn_brier) or np.isnan(ridge_brier)) else np.nan,
        "nn_better": nn_brier < ridge_brier if not (np.isnan(nn_brier) or np.isnan(ridge_brier)) else None,
    }

    # Seasonal breakdown
    for df, prefix in [(nn_df, "nn"), (ridge_df, "ridge")]:
        df["_date"] = pd.to_datetime(df["date"])
        df["_month"] = df["_date"].dt.month
        df["_season"] = df["_month"].apply(_get_season)
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            mask = df["_season"] == season
            if mask.sum() > 0:
                s_df = df[mask]
                result[f"{prefix}_{season.lower()}_brier"] = float(
                    np.mean((s_df["model_prob"] - s_df["outcome"]) ** 2)
                )

    return result


# ===========================================================================
# Step 10: Generate Report
# ===========================================================================

def generate_report(is_result, oos_result, best_config, brier_is, brier_oos,
                    strategies_df, nn_source, ridge_source,
                    nn_vs_ridge_data, proxy_diagnostics, output_dir):
    """Generate comprehensive text report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "=" * 78,
        "  KALSHI KXHIGHNY MAX-TRAINING BACKTEST REPORT",
        "  Enhanced Market Proxy + Max-Training Model Predictions",
        "=" * 78,
        "",
        "--- DATA SOURCES ---",
        f"  NN model source: {nn_source}",
        f"  Ridge model source: {ridge_source or 'Not available'}",
        f"  Market proxy: Enhanced (Ridge with lag1-3, rolling7d, clim mean/std)",
        f"  IS period: 2023-2024 (model trained on 1998-2020, val 2021-2022)",
        f"  OOS period: 2025 (model trained on 1998-2022, val 2023-2024)",
        f"  Market data: Real Kalshi KXHIGHNY settlements",
        "",
    ]

    # ---- Proxy Diagnostics ----
    lines.append("--- ENHANCED PROXY DIAGNOSTICS ---")
    for period_label, diag in proxy_diagnostics.items():
        lines.append(f"\n  {period_label}:")
        lines.append(f"    Train end: {diag.get('train_end_date', 'N/A')}")
        lines.append(f"    Ridge alpha: {diag.get('ridge_alpha', 'N/A')}")
        lines.append(f"    Features: {diag.get('feature_names', 'N/A')}")
        if diag.get('ridge_coefs'):
            coef_strs = [f"{n}={c:.4f}" for n, c in
                         zip(diag.get('feature_names', []), diag.get('ridge_coefs', []))]
            lines.append(f"    Coefs: {', '.join(coef_strs)}")
        lines.append(f"    Overall sigma: {diag.get('overall_sigma', 'N/A'):.2f}")
    lines.append("")

    # ---- Brier Score Comparison ----
    lines.append("--- BRIER SCORE COMPARISON ---")
    for period_label, brier in [("IS (2023-2024)", brier_is), ("OOS (2025)", brier_oos)]:
        if brier is None:
            continue
        lines.append(f"\n  {period_label}:")

        for comparison, bdata in brier.items():
            if comparison in ("seasonal", "monthly"):
                continue
            if not isinstance(bdata, dict):
                continue
            delta = bdata.get('brier_delta', 0)
            winner = "MODEL BETTER" if delta < 0 else "COMPARISON BETTER"
            lines.append(
                f"    {comparison}: model={bdata.get('model_brier', 'N/A'):.4f}, "
                f"comp={bdata.get('market_brier', 'N/A'):.4f}, "
                f"delta={delta:.4f} ({winner})"
            )

        seasonal = brier.get("seasonal", {})
        if seasonal:
            lines.append(f"\n    Seasonal ({period_label}):")
            for season in ["Winter", "Spring", "Summer", "Fall"]:
                sdata = seasonal.get(season, {})
                if sdata:
                    parts = [f"model={sdata['model_brier']:.4f}",
                             f"enhanced={sdata['enhanced_proxy_brier']:.4f}"]
                    if 'kalshi_brier' in sdata:
                        parts.append(f"kalshi={sdata['kalshi_brier']:.4f}")
                    parts.append(f"n={sdata['n']}")
                    lines.append(f"      {season:8s}: {', '.join(parts)}")
    lines.append("")

    # ---- NN vs Ridge ----
    if nn_vs_ridge_data:
        lines.append("--- NN vs RIDGE COMPARISON ---")
        for entry in nn_vs_ridge_data:
            if entry is None:
                continue
            lines.append(f"\n  {entry['period']}:")
            lines.append(f"    NN Brier:    {entry['nn_brier']:.4f} (n={entry['nn_n']})")
            lines.append(f"    Ridge Brier: {entry['ridge_brier']:.4f} (n={entry['ridge_n']})")
            delta = entry.get('brier_delta', 0)
            if entry.get('nn_better') is True:
                lines.append(f"    Winner: NN (delta={delta:.4f})")
            elif entry.get('nn_better') is False:
                lines.append(f"    Winner: Ridge (delta={delta:.4f})")
            else:
                lines.append(f"    Winner: Inconclusive")
        lines.append("")

    # ---- Strategy Grid Results ----
    lines.append("--- STRATEGY GRID SEARCH (IS 2023-2024) ---")
    if not strategies_df.empty:
        trading = strategies_df[strategies_df["n_trades"] > 0]
        profitable = strategies_df[strategies_df["total_pnl"] > 0]
        lines.append(f"  Total strategies: {len(strategies_df)}")
        lines.append(f"  With trades: {len(trading)}")
        lines.append(f"  Profitable: {len(profitable)}")
        if len(trading) > 0:
            pnls = trading["total_pnl"]
            lines.append(f"  Mean PnL: ${pnls.mean():.2f}")
            lines.append(f"  Median PnL: ${pnls.median():.2f}")
            lines.append(f"  Best PnL: ${pnls.max():.2f}")

            lines.append("\n  Top 5 by Sharpe:")
            finite = trading[trading["sharpe_ratio"].apply(lambda x: np.isfinite(x))]
            if not finite.empty:
                top5 = finite.nlargest(5, "sharpe_ratio")
                for _, row in top5.iterrows():
                    lines.append(
                        f"    {row['strategy_name']}: "
                        f"Sharpe={row['sharpe_ratio']:.2f}, PnL=${row['total_pnl']:.0f}, "
                        f"WR={row['win_rate']*100:.0f}%, N={row['n_trades']}"
                    )
    lines.append("")

    # ---- Selected Strategy ----
    lines.append("--- SELECTED BEST STRATEGY ---")
    if "error" in best_config:
        lines.append(f"  Error: {best_config['error']}")
    else:
        lines.append(f"  Name: {best_config.get('strategy_name', 'N/A')}")
        lines.append(f"  Selection reason: {best_config.get('selection_reason', 'N/A')}")
        lines.append(f"  EV threshold: {best_config.get('ev_threshold', 'N/A')}")
        lines.append(f"  Sizing: {best_config.get('sizing_method', 'N/A')}")
        lines.append(f"  Kelly fraction: {best_config.get('kelly_fraction', 'N/A')}")
        lines.append(f"  Max position: {best_config.get('max_position_frac', 'N/A')}")
    lines.append("")

    # ---- IS vs OOS Comparison Table ----
    lines.append("--- IS vs OOS COMPARISON ---")
    lines.append(f"  {'Metric':<20s} {'IS':>12s} {'OOS':>12s} {'Change':>12s}")
    lines.append(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    metrics = [
        ("Total PnL ($)", "total_pnl", "${:.2f}"),
        ("ROI (%)", "roi", "{:.1%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Win Rate (%)", "win_rate", "{:.1%}"),
        ("Max Drawdown ($)", "max_drawdown", "${:.2f}"),
        ("Trades", "n_trades", "{}"),
    ]

    is_m = is_result.to_summary_dict() if is_result else {}
    oos_m = oos_result.to_summary_dict() if oos_result else {}

    for label, key, fmt in metrics:
        is_val = is_m.get(key, 0)
        oos_val = oos_m.get(key, 0)
        try:
            is_str = fmt.format(is_val)
            oos_str = fmt.format(oos_val)
            if isinstance(is_val, (int, float)) and isinstance(oos_val, (int, float)):
                change = oos_val - is_val
                ch_str = fmt.format(change)
            else:
                ch_str = "N/A"
        except (ValueError, TypeError):
            is_str = str(is_val)
            oos_str = str(oos_val)
            ch_str = "N/A"
        lines.append(f"  {label:<20s} {is_str:>12s} {oos_str:>12s} {ch_str:>12s}")
    lines.append("")

    # ---- Seasonal PnL (OOS) ----
    if oos_result and oos_result.trades:
        lines.append("--- OOS SEASONAL P&L ---")
        seasonal = compute_seasonal_pnl(oos_result.trades)
        for season in SEASON_ORDER:
            sdata = seasonal.get(season, {})
            if sdata:
                lines.append(
                    f"  {season}: PnL=${sdata['total_pnl']:.2f}, "
                    f"trades={sdata['n_trades']}, WR={sdata['win_rate']*100:.1f}%"
                )
        lines.append("")

    # ---- Recommendation ----
    lines.append("--- RECOMMENDATION ---")
    if oos_result:
        if oos_result.sharpe_ratio >= 1.5 and oos_result.total_pnl > 0:
            lines.append("  VERDICT: VALIDATED")
            lines.append("  Strong OOS performance. Proceed to paper trading.")
        elif 0.5 <= oos_result.sharpe_ratio < 1.5 and oos_result.total_pnl > 0:
            lines.append("  VERDICT: CAUTIOUS")
            lines.append("  Edge exists but weaker than IS. Reduce sizing by 50%.")
        elif oos_result.total_pnl > 0:
            lines.append("  VERDICT: MARGINAL")
            lines.append("  Positive PnL but low Sharpe. Extended paper trading recommended.")
        else:
            lines.append("  VERDICT: NOT VALIDATED")
            lines.append("  OOS unprofitable. Diagnose before deploying.")
    else:
        lines.append("  No OOS results available.")

    lines.extend(["", "=" * 78])

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "expanded_backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info("Saved report to %s", report_path)
    return report_text


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    print("=" * 78)
    print("  KALSHI KXHIGHNY MAX-TRAINING BACKTEST")
    print("  Enhanced Market Proxy + Comprehensive Strategy Search")
    print("=" * 78)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== Check for prediction files =====
    if not check_prediction_files():
        print("  [WAITING] Prediction files not found. Required files:")
        print(f"    - {MAX_TRAIN_NN_IS}")
        print(f"    - {MAX_TRAIN_NN_OOS}")
        print(f"    OR fallback:")
        print(f"    - {EXPANDED_PRED_IS}")
        print(f"    - {EXPANDED_PRED_OOS}")
        print()
        print("  Checking every 30 seconds for up to 10 minutes...")

        max_wait = 600  # 10 minutes
        elapsed = 0
        while elapsed < max_wait:
            time.sleep(30)
            elapsed += 30
            if check_prediction_files():
                print(f"  [FOUND] Prediction files detected after {elapsed}s.")
                break
            print(f"  Still waiting... ({elapsed}s / {max_wait}s)")

        if not check_prediction_files():
            print("\n  [ERROR] Prediction files still not available after 10 minutes.")
            print("  Please run: python scripts/generate_max_training_predictions.py")
            print("  Exiting.")
            return

    # ===== Step 1: Load all data =====
    print("\nStep 1: Loading data...")

    nn_is, nn_source_is = load_nn_predictions("IS")
    nn_oos, nn_source_oos = load_nn_predictions("OOS")
    ridge_is, ridge_source_is = load_ridge_predictions("IS")
    ridge_oos, ridge_source_oos = load_ridge_predictions("OOS")

    nn_source = nn_source_is
    ridge_source = ridge_source_is

    print(f"  NN IS: {len(nn_is)} days ({nn_source_is})" if nn_is is not None else "  NN IS: NOT AVAILABLE")
    print(f"  NN OOS: {len(nn_oos)} days ({nn_source_oos})" if nn_oos is not None else "  NN OOS: NOT AVAILABLE")
    print(f"  Ridge IS: {len(ridge_is)} days ({ridge_source_is})" if ridge_is is not None else "  Ridge IS: NOT AVAILABLE")
    print(f"  Ridge OOS: {len(ridge_oos)} days ({ridge_source_oos})" if ridge_oos is not None else "  Ridge OOS: NOT AVAILABLE")

    if nn_is is None:
        print("\n  [ERROR] No model predictions available. Exiting.")
        return

    kalshi_is = load_kalshi_data("IS")
    kalshi_oos = load_kalshi_data("OOS")
    if kalshi_is is None:
        print("\n  [ERROR] Kalshi IS data not found. Exiting.")
        return

    print(f"  Kalshi IS: {len(kalshi_is)} rows, {kalshi_is['date'].nunique()} days")
    if kalshi_oos is not None:
        print(f"  Kalshi OOS: {len(kalshi_oos)} rows, {kalshi_oos['date'].nunique()} days")

    tmax_history = load_tmax_history()
    if tmax_history is None:
        print("\n  [ERROR] TMAX history not found. Exiting.")
        return
    print(f"  TMAX history: {len(tmax_history)} days")
    print()

    # ===== Step 2: Build market proxies =====
    print("Step 2: Building market proxies...")

    # Enhanced proxy for IS (trained before 2023)
    enhanced_proxy_is = EnhancedMarketProxy(tmax_history)
    enhanced_proxy_is.fit(train_end_date="2022-12-31")
    diag_is = enhanced_proxy_is.get_diagnostics()
    print(f"  Enhanced proxy (IS): alpha={diag_is['ridge_alpha']}, "
          f"sigma={diag_is['overall_sigma']:.2f}, features={len(diag_is['feature_names'])}")

    # Base proxy for comparison
    base_proxy_is = MarketProxy(tmax_history)
    base_proxy_is.fit(train_end_date="2022-12-31")
    base_diag_is = base_proxy_is.get_diagnostics()
    print(f"  Base proxy (IS): sigma={base_diag_is['overall_sigma']:.2f}")

    # Naive proxy
    naive_proxy = NaiveMarketProxy()
    print("  Naive proxy: initialized")

    # Enhanced proxy for OOS (trained before 2025)
    enhanced_proxy_oos = None
    if kalshi_oos is not None:
        enhanced_proxy_oos = EnhancedMarketProxy(tmax_history)
        enhanced_proxy_oos.fit(train_end_date="2024-12-31")
        diag_oos = enhanced_proxy_oos.get_diagnostics()
        print(f"  Enhanced proxy (OOS): alpha={diag_oos['ridge_alpha']}, "
              f"sigma={diag_oos['overall_sigma']:.2f}")

        base_proxy_oos = MarketProxy(tmax_history)
        base_proxy_oos.fit(train_end_date="2024-12-31")
    else:
        diag_oos = {}
        base_proxy_oos = None
    print()

    # ===== Step 3: Compute proxy probabilities =====
    print("Step 3: Computing proxy probabilities...")

    # IS: enhanced + base + naive
    kalshi_is_full = add_enhanced_proxy_probabilities(kalshi_is, enhanced_proxy_is, tmax_history)
    kalshi_is_full = add_base_proxy_probabilities(kalshi_is_full, base_proxy_is, tmax_history)
    # Add naive
    naive_probs = []
    for _, row in kalshi_is_full.iterrows():
        dt = pd.to_datetime(row["date"]).date()
        yt_date = dt - timedelta(days=1)
        hist_lkp = dict(zip(
            pd.to_datetime(tmax_history["date"]).dt.date,
            tmax_history["tmax_f"],
        ))
        yt = hist_lkp.get(yt_date)
        if yt is None:
            naive_probs.append(np.nan)
            continue
        try:
            p = naive_proxy.compute_bracket_prob(
                dt, yt,
                row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                row.get("direction", "between"),
            )
            naive_probs.append(p)
        except Exception:
            naive_probs.append(np.nan)
    kalshi_is_full["naive_proxy_prob"] = naive_probs

    print(f"  IS enhanced proxy: mean={kalshi_is_full['proxy_prob'].mean():.3f}")
    print(f"  IS base proxy: mean={kalshi_is_full['base_proxy_prob'].mean():.3f}")
    print(f"  IS naive proxy: mean={kalshi_is_full['naive_proxy_prob'].mean():.3f}")

    # OOS
    kalshi_oos_full = None
    if kalshi_oos is not None and enhanced_proxy_oos is not None:
        kalshi_oos_full = add_enhanced_proxy_probabilities(kalshi_oos, enhanced_proxy_oos, tmax_history)
        kalshi_oos_full = add_base_proxy_probabilities(kalshi_oos_full, base_proxy_oos, tmax_history)
        naive_probs_oos = []
        hist_lkp = dict(zip(
            pd.to_datetime(tmax_history["date"]).dt.date,
            tmax_history["tmax_f"],
        ))
        for _, row in kalshi_oos_full.iterrows():
            dt = pd.to_datetime(row["date"]).date()
            yt = hist_lkp.get(dt - timedelta(days=1))
            if yt is None:
                naive_probs_oos.append(np.nan)
                continue
            try:
                p = naive_proxy.compute_bracket_prob(
                    dt, yt,
                    row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                    row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                    row.get("direction", "between"),
                )
                naive_probs_oos.append(p)
            except Exception:
                naive_probs_oos.append(np.nan)
        kalshi_oos_full["naive_proxy_prob"] = naive_probs_oos

        print(f"  OOS enhanced proxy: mean={kalshi_oos_full['proxy_prob'].mean():.3f}")
        print(f"  OOS base proxy: mean={kalshi_oos_full['base_proxy_prob'].mean():.3f}")
    print()

    # ===== Step 4: Align model predictions =====
    print("Step 4: Aligning model predictions with market data...")

    # NN alignment
    nn_comparison_is = align_model_and_market(kalshi_is_full, nn_is)
    _merge_proxy_cols(nn_comparison_is, kalshi_is_full)
    print(f"  NN IS aligned: {len(nn_comparison_is)} rows, {nn_comparison_is['date'].nunique()} dates")

    nn_comparison_oos = None
    if kalshi_oos_full is not None and nn_oos is not None:
        nn_comparison_oos = align_model_and_market(kalshi_oos_full, nn_oos)
        _merge_proxy_cols(nn_comparison_oos, kalshi_oos_full)
        print(f"  NN OOS aligned: {len(nn_comparison_oos)} rows")

    # Ridge alignment
    ridge_comparison_is = None
    ridge_comparison_oos = None
    if ridge_is is not None:
        ridge_comparison_is = align_model_and_market(kalshi_is_full, ridge_is)
        _merge_proxy_cols(ridge_comparison_is, kalshi_is_full)
        print(f"  Ridge IS aligned: {len(ridge_comparison_is)} rows")
    if ridge_oos is not None and kalshi_oos_full is not None:
        ridge_comparison_oos = align_model_and_market(kalshi_oos_full, ridge_oos)
        _merge_proxy_cols(ridge_comparison_oos, kalshi_oos_full)
        print(f"  Ridge OOS aligned: {len(ridge_comparison_oos)} rows")
    print()

    # ===== Step 5: Strategy grid search (IS) =====
    print("Step 5: Running strategy grid search on IS data (NN model)...")
    all_results, strategies_df, backtest_data_is = run_strategy_grid_search(
        nn_comparison_is, proxy_col="proxy_prob",
    )

    trading = strategies_df[strategies_df["n_trades"] > 0]
    profitable = strategies_df[strategies_df["total_pnl"] > 0]
    print(f"  Strategies evaluated: {len(strategies_df)}")
    print(f"  With trades: {len(trading)}")
    print(f"  Profitable: {len(profitable)}")

    strategies_df.to_csv(os.path.join(OUTPUT_DIR, "all_strategies_is.csv"), index=False)
    print()

    # ===== Step 6: Select best strategy =====
    print("Step 6: Selecting best strategy...")
    best_config, best_row = select_best_strategy(strategies_df)

    if "error" in best_config:
        print(f"  Error: {best_config['error']}")
    else:
        print(f"  Selected: {best_config.get('strategy_name', 'N/A')}")
        print(f"  Reason: {best_config.get('selection_reason', 'N/A')}")
        print(f"  IS PnL: ${best_config.get('total_pnl', 0):.2f}")
        print(f"  IS Sharpe: {best_config.get('sharpe_ratio', 0):.2f}")
        print(f"  IS Win Rate: {best_config.get('win_rate', 0)*100:.1f}%")
        print(f"  IS Trades: {best_config.get('n_trades', 0)}")

    with open(os.path.join(OUTPUT_DIR, "best_strategy_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    # Find IS result for best strategy
    is_best_result = None
    for r in all_results:
        if r.strategy_name == best_config.get("strategy_name"):
            is_best_result = r
            break
    print()

    # ===== Step 7: OOS Validation =====
    print("Step 7: Running OOS validation...")
    oos_result = None
    if nn_comparison_oos is not None and "error" not in best_config:
        oos_result, backtest_data_oos = run_oos_backtest(
            nn_comparison_oos, best_config, proxy_col="proxy_prob",
        )
        print(f"  OOS PnL: ${oos_result.total_pnl:.2f}")
        print(f"  OOS Sharpe: {oos_result.sharpe_ratio:.2f}")
        print(f"  OOS Win Rate: {oos_result.win_rate*100:.1f}%")
        print(f"  OOS Trades: {oos_result.n_trades}")
    else:
        print("  Skipping OOS (no data or no strategy)")
    print()

    # ===== Step 8: Brier Scores =====
    print("Step 8: Computing Brier scores...")
    brier_is = compute_all_brier_scores(
        nn_comparison_is, "proxy_prob", "base_proxy_prob", "naive_proxy_prob",
    )
    brier_oos = None
    if nn_comparison_oos is not None:
        brier_oos = compute_all_brier_scores(
            nn_comparison_oos, "proxy_prob", "base_proxy_prob", "naive_proxy_prob",
        )

    for label, brier in [("IS", brier_is), ("OOS", brier_oos)]:
        if brier is None:
            continue
        print(f"\n  {label} Brier Scores:")
        for comp, bdata in brier.items():
            if comp in ("seasonal", "monthly") or not isinstance(bdata, dict):
                continue
            print(f"    {comp}: model={bdata.get('model_brier', 0):.4f}, "
                  f"comp={bdata.get('market_brier', 0):.4f}, "
                  f"delta={bdata.get('brier_delta', 0):.4f}")

    with open(os.path.join(OUTPUT_DIR, "brier_analysis.json"), "w") as f:
        json.dump({"IS": brier_is, "OOS": brier_oos}, f, indent=2, default=str)
    print()

    # ===== Step 8b: NN vs Ridge =====
    print("Step 8b: NN vs Ridge comparison...")
    nn_vs_ridge_data = []
    nn_ridge_is = compare_nn_vs_ridge(nn_comparison_is, ridge_comparison_is, "IS (2023-2024)", OUTPUT_DIR)
    nn_vs_ridge_data.append(nn_ridge_is)
    nn_ridge_oos = compare_nn_vs_ridge(nn_comparison_oos, ridge_comparison_oos, "OOS (2025)", OUTPUT_DIR)
    nn_vs_ridge_data.append(nn_ridge_oos)

    if nn_ridge_is:
        print(f"  IS: NN Brier={nn_ridge_is['nn_brier']:.4f}, Ridge Brier={nn_ridge_is['ridge_brier']:.4f}")
    if nn_ridge_oos:
        print(f"  OOS: NN Brier={nn_ridge_oos['nn_brier']:.4f}, Ridge Brier={nn_ridge_oos['ridge_brier']:.4f}")

    # Save NN vs Ridge comparison
    nn_ridge_rows = [r for r in nn_vs_ridge_data if r is not None]
    if nn_ridge_rows:
        pd.DataFrame(nn_ridge_rows).to_csv(
            os.path.join(OUTPUT_DIR, "nn_vs_ridge_comparison.csv"), index=False,
        )
    print()

    # ===== Step 9: Generate plots =====
    print("Step 9: Generating plots...")
    saved_plots = generate_plots(
        nn_comparison_is, nn_comparison_oos,
        is_best_result, oos_result,
        brier_is, brier_oos,
        OUTPUT_DIR,
        nn_vs_ridge=nn_vs_ridge_data,
    )
    print(f"  Generated {len(saved_plots)} plots")
    print()

    # ===== Step 10: Generate report =====
    print("Step 10: Generating comprehensive report...")
    proxy_diagnostics = {"IS (train<=2022)": diag_is}
    if diag_oos:
        proxy_diagnostics["OOS (train<=2024)"] = diag_oos

    report = generate_report(
        is_best_result, oos_result, best_config,
        brier_is, brier_oos,
        strategies_df, nn_source, ridge_source,
        nn_vs_ridge_data, proxy_diagnostics, OUTPUT_DIR,
    )
    print()
    print(report)

    # ===== Save IS/OOS comparison =====
    if is_best_result and oos_result:
        comparison_table = pd.DataFrame([
            {"period": "IS (2023-2024)", **is_best_result.to_summary_dict()},
            {"period": "OOS (2025)", **oos_result.to_summary_dict()},
        ])
        comparison_table.to_csv(
            os.path.join(OUTPUT_DIR, "is_vs_oos_comparison.csv"), index=False,
        )

    # ===== Save proxy diagnostics =====
    with open(os.path.join(OUTPUT_DIR, "proxy_diagnostics.json"), "w") as f:
        json.dump(proxy_diagnostics, f, indent=2, default=str)

    # ===== Save seasonal Brier breakdown =====
    seasonal_data = []
    for period_label, brier in [("IS", brier_is), ("OOS", brier_oos)]:
        if brier is None:
            continue
        for season, sdata in brier.get("seasonal", {}).items():
            row = {"period": period_label, "season": season}
            row.update(sdata)
            seasonal_data.append(row)
    if seasonal_data:
        pd.DataFrame(seasonal_data).to_csv(
            os.path.join(OUTPUT_DIR, "seasonal_brier_breakdown.csv"), index=False,
        )

    # ===== List output files =====
    print("\nOutput files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fpath} ({size_kb:.1f} KB)")

    print()
    print("=" * 78)
    print("  MAX-TRAINING BACKTEST COMPLETE!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 78)


def _merge_proxy_cols(comparison_df, kalshi_df):
    """Merge proxy columns into comparison_df if missing."""
    proxy_cols = ["proxy_prob", "base_proxy_prob", "naive_proxy_prob"]
    missing_cols = [c for c in proxy_cols if c not in comparison_df.columns]
    if not missing_cols:
        return

    available = [c for c in proxy_cols if c in kalshi_df.columns]
    if not available:
        return

    proxy_lookup = kalshi_df[["date", "ticker"] + available].copy()
    proxy_lookup["date"] = pd.to_datetime(proxy_lookup["date"]).dt.date
    comparison_df["date"] = comparison_df["date"].apply(
        lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
    )
    # Merge
    for col in missing_cols:
        if col in proxy_lookup.columns:
            lookup_sub = proxy_lookup[["date", "ticker", col]].drop_duplicates()
            comparison_df_tmp = comparison_df.merge(
                lookup_sub, on=["date", "ticker"], how="left", suffixes=("", "_new"),
            )
            if col + "_new" in comparison_df_tmp.columns:
                comparison_df[col] = comparison_df_tmp[col + "_new"]
            elif col in comparison_df_tmp.columns:
                comparison_df[col] = comparison_df_tmp[col]


if __name__ == "__main__":
    main()
