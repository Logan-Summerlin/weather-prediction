#!/usr/bin/env python3
"""
Expanded Kalshi KXHIGHNY Backtest with Improved Market Proxy.

Runs a comprehensive IS/OOS backtest using:
  1. The improved MarketProxy (regression-based, seasonally-varying sigma)
  2. Real Kalshi market data (2023-2024 IS, 2025 OOS)
  3. Model predictions from Analyst 1's expanded model (or fallback to existing)
  4. Ridge baseline predictions for comparison

Key improvements over the naive backtest:
  - Enhanced market proxy: multi-day regression + smooth climatology + monthly sigma
  - Comprehensive strategy grid search on IS period
  - Frozen strategy OOS validation
  - Brier score comparison (NN vs Ridge vs naive proxy vs enhanced proxy)
  - Seasonal analysis and reliability diagrams

Outputs:
  - results/kalshi_expanded_backtest/ (all results, plots, reports)

Usage:
    python scripts/run_expanded_backtest.py
"""

import json
import os
import re
import sys
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from src.market_proxy import MarketProxy, NaiveMarketProxy
from src.kalshi_client import build_historical_comparison, compute_brier_scores
from src.kalshi_backtester import BacktestAnalyzer, CalibrationAnalyzer, compute_seasonal_pnl
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

# Plot style
_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "kalshi_expanded_backtest")

# Prediction file paths (Analyst 1's expanded model)
EXPANDED_PRED_IS = os.path.join(DATA_DIR, "expanded_model_predictions_2023_2024.csv")
EXPANDED_PRED_OOS = os.path.join(DATA_DIR, "expanded_model_predictions_2025.csv")

# Fallback: existing real model predictions
FALLBACK_PRED_IS = os.path.join(DATA_DIR, "real_model_predictions_2023_2024.csv")
FALLBACK_PRED_OOS = os.path.join(DATA_DIR, "real_model_predictions_2025.csv")

# Real Kalshi market data
KALSHI_IS = os.path.join(DATA_DIR, "real_kalshi_2023_2024.csv")
KALSHI_OOS = os.path.join(DATA_DIR, "real_kalshi_2025.csv")

# Central Park TMAX full history
TMAX_HISTORY = os.path.join(DATA_DIR, "central_park_tmax_full_history.csv")

# Strategy grid parameters
STRATEGY_GRID = {
    "ev_thresholds": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    "sizing_methods": ["fixed", "proportional", "fractional_kelly", "capped_kelly"],
    "kelly_fractions": [0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
    "fee_rates": [0.07],
    "max_positions": [0.05, 0.10, 0.15, 0.20],
    "bankrolls": [10000],
}

# Selection criteria
SELECTION_CRITERIA = {
    "min_trades": 100,
    "min_win_rate": 0.30,
    "min_sharpe": 1.5,
    "max_drawdown": 2000,
}


# ===========================================================================
# Helper: Season mapping
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


# ===========================================================================
# Step 1: Load Data
# ===========================================================================

def load_predictions(period="IS"):
    """Load model predictions, falling back to existing files if expanded not available.

    Returns (predictions_df, source_label).
    """
    if period == "IS":
        expanded = EXPANDED_PRED_IS
        fallback = FALLBACK_PRED_IS
    else:
        expanded = EXPANDED_PRED_OOS
        fallback = FALLBACK_PRED_OOS

    if os.path.exists(expanded):
        df = pd.read_csv(expanded)
        logger.info("Loaded expanded predictions (%s): %d rows from %s",
                     period, len(df), expanded)
        return df, "expanded_model"
    elif os.path.exists(fallback):
        df = pd.read_csv(fallback)
        logger.info("Expanded predictions not found; using fallback (%s): %d rows from %s",
                     period, len(df), fallback)
        return df, "baseline_model"
    else:
        return None, None


def load_kalshi_data(period="IS"):
    """Load real Kalshi market data."""
    path = KALSHI_IS if period == "IS" else KALSHI_OOS
    if not os.path.exists(path):
        logger.error("Kalshi data not found: %s", path)
        return None

    df = pd.read_csv(path)
    logger.info("Loaded Kalshi %s data: %d rows, %d days",
                period, len(df), df["date"].nunique())
    return df


def load_tmax_history():
    """Load Central Park full TMAX history for the market proxy."""
    if not os.path.exists(TMAX_HISTORY):
        logger.error("TMAX history not found: %s", TMAX_HISTORY)
        return None
    df = pd.read_csv(TMAX_HISTORY)
    logger.info("Loaded TMAX history: %d rows", len(df))
    return df


# ===========================================================================
# Step 2: Build Market Proxy Probabilities
# ===========================================================================

def add_proxy_probabilities(kalshi_df, proxy, tmax_history_df):
    """Add market proxy probabilities to Kalshi data.

    For each market row, compute the proxy's probability using the
    previous day's actual TMAX (from the TMAX history).

    Parameters
    ----------
    kalshi_df : pd.DataFrame
        Kalshi market data with date, threshold_low, threshold_high, direction.
    proxy : MarketProxy or NaiveMarketProxy
        Fitted proxy.
    tmax_history_df : pd.DataFrame
        TMAX history for looking up previous days' temperatures.

    Returns
    -------
    pd.DataFrame
        Kalshi data with added 'proxy_prob' column.
    """
    # Build a date -> tmax lookup
    hist = tmax_history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    tmax_lookup = dict(zip(hist["date"], hist["tmax_f"]))

    result = kalshi_df.copy()
    result["date_obj"] = pd.to_datetime(result["date"]).dt.date

    proxy_probs = []
    for _, row in result.iterrows():
        target_date = row["date_obj"]
        yesterday = target_date - timedelta(days=1)
        day_before = target_date - timedelta(days=2)

        yesterday_tmax = tmax_lookup.get(yesterday)
        day_before_tmax = tmax_lookup.get(day_before)

        if yesterday_tmax is None:
            proxy_probs.append(np.nan)
            continue

        try:
            prob = proxy.compute_bracket_prob(
                target_date=target_date,
                yesterday_tmax=yesterday_tmax,
                threshold_low=row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                threshold_high=row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                direction=row.get("direction", "between"),
                day_before_tmax=day_before_tmax,
            )
            proxy_probs.append(prob)
        except Exception as e:
            logger.warning("Proxy error for %s: %s", target_date, e)
            proxy_probs.append(np.nan)

    result["proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])

    n_valid = sum(1 for p in proxy_probs if not np.isnan(p) if p is not None)
    logger.info("Added proxy probabilities: %d/%d valid", n_valid, len(result))
    return result


# ===========================================================================
# Step 3: Align Model Predictions with Market Data
# ===========================================================================

def align_model_and_market(market_df, predictions_df):
    """Align model predictions with market data.

    Computes model_prob for each bucket using the Gaussian (mu, sigma)
    model predictions and preserves Kalshi's original settlement outcomes.
    """
    # Save API outcomes
    api_outcomes = market_df[["date", "ticker", "actual_outcome"]].copy()
    market_clean = market_df.drop(columns=["actual_tmax"], errors="ignore")

    comparison_df = build_historical_comparison(
        model_predictions_df=predictions_df,
        historical_markets_df=market_clean,
    )

    if comparison_df.empty:
        raise ValueError("No overlapping dates between model and market data")

    # Use Kalshi's actual settlement outcome
    if "actual_outcome" in comparison_df.columns:
        comparison_df["outcome"] = comparison_df["actual_outcome"].astype(float)

    logger.info(
        "Aligned %d rows across %d dates. Mean model_prob=%.3f, market_prob=%.3f",
        len(comparison_df), comparison_df["date"].nunique(),
        comparison_df["model_prob"].mean(), comparison_df["market_prob"].mean(),
    )
    return comparison_df


# ===========================================================================
# Step 4: Run Strategy Grid Search (IS)
# ===========================================================================

def run_strategy_grid_search(comparison_df, proxy_col="proxy_prob"):
    """Run comprehensive strategy grid search on IS data.

    Uses the enhanced proxy probabilities as the 'market_price' for
    the backtesting engine (since the proxy represents what a rational
    market should price).

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Aligned comparison data with model_prob, proxy_prob, outcome.
    proxy_col : str
        Column to use as market price for the backtest.

    Returns
    -------
    tuple[list[BacktestResult], pd.DataFrame]
        (all_results, comparison_df_with_metrics)
    """
    backtest_data = comparison_df.copy()

    # Prepare columns for BacktestEngine:
    # - market_price: the proxy probability (what the model trades against)
    # - actual_outcome: binary outcome (1=YES, 0=NO)
    backtest_data["market_price"] = backtest_data[proxy_col]

    # Use the 'outcome' column (set from Kalshi's actual_outcome in alignment)
    if "outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["outcome"].astype(float)
    elif "actual_outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["actual_outcome"].astype(float)

    # Keep only needed columns to avoid confusion
    required = ["date", "model_prob", "market_price", "actual_outcome"]
    missing = [c for c in required if c not in backtest_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(backtest_data)
    # Select only the columns the engine needs, plus extras for reporting
    keep_cols = required + [c for c in backtest_data.columns
                            if c not in required and c in ["ticker", "bucket", "direction"]]
    backtest_data = backtest_data[keep_cols].dropna(subset=required).reset_index(drop=True)
    logger.info("Backtest data: %d rows (dropped %d NaN)", len(backtest_data), before - len(backtest_data))

    # Generate strategy grid
    strategies = generate_strategy_grid(**STRATEGY_GRID)
    logger.info("Generated %d strategy permutations", len(strategies))

    # Run all strategies
    all_results = []
    for strategy in strategies:
        engine = BacktestEngine(strategy)
        result = engine.run_backtest(backtest_data)
        all_results.append(result)

    all_results.sort(key=lambda r: r.total_pnl, reverse=True)

    # Build comparison DataFrame
    rows = [r.to_summary_dict() for r in all_results]
    strategies_df = pd.DataFrame(rows)

    logger.info(
        "Grid search complete: %d strategies, %d with trades, %d profitable",
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

    # Primary filter
    candidates = strategies_df[
        (strategies_df["n_trades"] >= crit["min_trades"])
        & (strategies_df["win_rate"] > crit["min_win_rate"])
        & (strategies_df["sharpe_ratio"] >= crit["min_sharpe"])
        & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        & (strategies_df["max_drawdown"] < crit["max_drawdown"])
    ].copy()

    reason = "primary_criteria"

    if candidates.empty:
        logger.warning("No strategies meet primary criteria. Relaxing Sharpe to 0.5...")
        candidates = strategies_df[
            (strategies_df["n_trades"] >= 50)
            & (strategies_df["win_rate"] > 0.25)
            & (strategies_df["sharpe_ratio"] >= 0.5)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            & (strategies_df["max_drawdown"] < 3000.0)
        ].copy()
        reason = "relaxed_criteria"

    if candidates.empty:
        # Fallback: any profitable strategy
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

    # Tie-breaking: if top candidates have similar Sharpe, pick higher PnL
    top_sharpe = candidates["sharpe_ratio"].max()
    if top_sharpe > 2.0:
        near_top = candidates[candidates["sharpe_ratio"] >= top_sharpe - 0.2]
        if len(near_top) > 1:
            best_row = near_top.loc[near_top["total_pnl"].idxmax()]
            return _config_from_row(best_row, "best_pnl_near_top_sharpe"), best_row

    best_row = candidates.loc[candidates["sharpe_ratio"].idxmax()]
    return _config_from_row(best_row, reason), best_row


def _config_from_row(row, reason=""):
    """Extract strategy configuration from a DataFrame row."""
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

    # Parse parameters from strategy name
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
    """Reconstruct a TradingStrategy from a config dict."""
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
# Step 6: Run OOS Validation
# ===========================================================================

def run_oos_backtest(comparison_df, strategy_config, proxy_col="proxy_prob"):
    """Run the frozen IS-selected strategy on OOS data."""
    strategy = reconstruct_strategy(strategy_config)

    backtest_data = comparison_df.copy()
    backtest_data["market_price"] = backtest_data[proxy_col]

    if "outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["outcome"].astype(float)
    elif "actual_outcome" in backtest_data.columns:
        backtest_data["actual_outcome"] = backtest_data["actual_outcome"].astype(float)

    required = ["date", "model_prob", "market_price", "actual_outcome"]
    keep_cols = required + [c for c in backtest_data.columns
                            if c not in required and c in ["ticker", "bucket", "direction"]]
    backtest_data = backtest_data[keep_cols].dropna(subset=required).reset_index(drop=True)

    engine = BacktestEngine(strategy)
    result = engine.run_backtest(backtest_data)

    logger.info(
        "OOS backtest: %d trades, PnL=$%.2f, Sharpe=%.2f, WinRate=%.1f%%",
        result.n_trades, result.total_pnl, result.sharpe_ratio, result.win_rate * 100,
    )
    return result, backtest_data


# ===========================================================================
# Step 7: Compute Brier Scores
# ===========================================================================

def compute_all_brier_scores(comparison_df, proxy_col="proxy_prob", naive_col="naive_proxy_prob"):
    """Compute comprehensive Brier score analysis.

    Compares: model vs enhanced proxy vs naive proxy vs Kalshi market.
    """
    df = comparison_df.dropna(subset=["model_prob", proxy_col, "outcome"]).copy()
    results = {}

    # Model vs enhanced proxy
    if len(df) > 0:
        results["model_vs_enhanced"] = compute_brier_scores(
            df["model_prob"], df[proxy_col], df["outcome"],
        )

    # Model vs naive proxy (if available)
    if naive_col in df.columns:
        df_naive = df.dropna(subset=[naive_col])
        if len(df_naive) > 0:
            results["model_vs_naive"] = compute_brier_scores(
                df_naive["model_prob"], df_naive[naive_col], df_naive["outcome"],
            )

    # Model vs Kalshi market prices
    if "market_prob" in df.columns:
        df_mkt = df.dropna(subset=["market_prob"])
        if len(df_mkt) > 0:
            results["model_vs_market"] = compute_brier_scores(
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
        seasonal[season] = {
            "model_brier": float(np.mean((s_df["model_prob"] - s_df["outcome"]) ** 2)),
            "enhanced_brier": float(np.mean((s_df[proxy_col] - s_df["outcome"]) ** 2)),
            "n": int(mask.sum()),
        }
        if naive_col in s_df.columns:
            s_naive = s_df.dropna(subset=[naive_col])
            if len(s_naive) > 0:
                seasonal[season]["naive_brier"] = float(
                    np.mean((s_naive[naive_col] - s_naive["outcome"]) ** 2)
                )
    results["seasonal"] = seasonal

    return results


# ===========================================================================
# Step 8: Generate Plots
# ===========================================================================

def generate_plots(comparison_df_is, comparison_df_oos, is_result, oos_result,
                   brier_is, brier_oos, output_dir):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # ---- 1. Model vs Market Probability Scatter (IS) ----
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
                ax.set_title(f"Model vs Proxy: {title}")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_vs_proxy_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. Cumulative PnL Curves (IS + OOS) ----
    fig, ax = plt.subplots(figsize=(14, 7))
    if is_result and is_result.trades:
        is_pnls = np.cumsum([t["pnl"] for t in is_result.trades])
        ax.plot(range(len(is_pnls)), is_pnls,
                label=f"IS: PnL=${is_result.total_pnl:.0f}, Sharpe={is_result.sharpe_ratio:.2f}",
                linewidth=1.5, color="#4c72b0")
    if oos_result and oos_result.trades:
        oos_pnls = np.cumsum([t["pnl"] for t in oos_result.trades])
        offset = len(is_pnls) if (is_result and is_result.trades) else 0
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
    for result, label, color in [
        (is_result, "IS (2023-2024)", "#4c72b0"),
        (oos_result, "OOS (2025)", "#d62728"),
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
            slug = "is" if "IS" in label else "oos"
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
        proxy_scores = []

        mvp = brier.get("model_vs_enhanced", {})
        if mvp:
            labels.append("Enhanced Proxy")
            model_scores.append(mvp.get("model_brier", 0))
            proxy_scores.append(mvp.get("market_brier", 0))

        mvn = brier.get("model_vs_naive", {})
        if mvn:
            labels.append("Naive Proxy")
            model_scores.append(mvn.get("model_brier", 0))
            proxy_scores.append(mvn.get("market_brier", 0))

        mvm = brier.get("model_vs_market", {})
        if mvm:
            labels.append("Kalshi Market")
            model_scores.append(mvm.get("model_brier", 0))
            proxy_scores.append(mvm.get("market_brier", 0))

        if labels:
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, model_scores, width, label="NN Model", color="#4c72b0", alpha=0.8)
            ax.bar(x + width / 2, proxy_scores, width, label="Comparison", color="#d62728", alpha=0.8)
            ax.set_ylabel("Brier Score (lower = better)")
            ax.set_title(f"Brier Score Comparison: {period_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "brier_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 5. Reliability Diagram (OOS) ----
    if comparison_df_oos is not None:
        valid_oos = comparison_df_oos.dropna(subset=["model_prob", "outcome"])
        if len(valid_oos) > 50:
            cal_analyzer = CalibrationAnalyzer()
            cal_path = cal_analyzer.plot_reliability_diagram(
                valid_oos["model_prob"], valid_oos["outcome"],
                output_dir=output_dir,
                title="OOS Model Calibration: Reliability Diagram",
            )
            saved.append(cal_path)

    # ---- 6. Seasonal Brier Heatmap ----
    seasonal_data = []
    for period_label, brier in [("IS", brier_is), ("OOS", brier_oos)]:
        if brier is None:
            continue
        for season, sdata in brier.get("seasonal", {}).items():
            row = {"period": period_label, "season": season}
            row["model_brier"] = sdata.get("model_brier", np.nan)
            row["enhanced_brier"] = sdata.get("enhanced_brier", np.nan)
            row["naive_brier"] = sdata.get("naive_brier", np.nan)
            row["n"] = sdata.get("n", 0)
            seasonal_data.append(row)

    if seasonal_data:
        seasonal_df = pd.DataFrame(seasonal_data)
        seasonal_df.to_csv(os.path.join(output_dir, "seasonal_brier_breakdown.csv"), index=False)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ===========================================================================
# Step 9: Generate Report
# ===========================================================================

def generate_report(is_result, oos_result, best_config, brier_is, brier_oos,
                    strategies_df, model_source, output_dir):
    """Generate comprehensive text report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "=" * 78,
        "  KALSHI KXHIGHNY EXPANDED BACKTEST REPORT",
        "  Enhanced Market Proxy + Comprehensive Strategy Search",
        "=" * 78,
        "",
        f"  Model source: {model_source}",
        f"  Market proxy: Enhanced (regression + smooth climatology + monthly sigma)",
        f"  IS period: 2023-2024",
        f"  OOS period: 2025",
        "",
    ]

    # ---- Model vs Proxy Comparison ----
    lines.append("--- BRIER SCORE COMPARISON ---")
    for period_label, brier in [("IS (2023-2024)", brier_is), ("OOS (2025)", brier_oos)]:
        if brier is None:
            continue
        lines.append(f"\n  {period_label}:")

        for comparison, bdata in brier.items():
            if comparison == "seasonal":
                continue
            if not isinstance(bdata, dict):
                continue
            lines.append(
                f"    {comparison}: model={bdata.get('model_brier', 'N/A'):.4f}, "
                f"comparison={bdata.get('market_brier', 'N/A'):.4f}, "
                f"delta={bdata.get('brier_delta', 'N/A'):.4f} "
                f"({'MODEL BETTER' if bdata.get('brier_delta', 0) < 0 else 'COMPARISON BETTER'})"
            )

        seasonal = brier.get("seasonal", {})
        if seasonal:
            lines.append(f"\n    Seasonal breakdown ({period_label}):")
            for season in ["Winter", "Spring", "Summer", "Fall"]:
                sdata = seasonal.get(season, {})
                if sdata:
                    lines.append(
                        f"      {season:8s}: model={sdata.get('model_brier', 0):.4f}, "
                        f"enhanced={sdata.get('enhanced_brier', 0):.4f}, "
                        f"naive={sdata.get('naive_brier', 'N/A')}, "
                        f"n={sdata.get('n', 0)}"
                    )

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

        # Top 5
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
        lines.append(f"  Sizing method: {best_config.get('sizing_method', 'N/A')}")
        lines.append(f"  Kelly fraction: {best_config.get('kelly_fraction', 'N/A')}")
        lines.append(f"  Max position: {best_config.get('max_position_frac', 'N/A')}")
    lines.append("")

    # ---- IS vs OOS Comparison ----
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
        for season in ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]:
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
            lines.append("  The strategy shows strong OOS performance.")
            lines.append("  Recommended: Proceed to paper trading with frozen parameters.")
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
    print("  KALSHI KXHIGHNY EXPANDED BACKTEST")
    print("  Enhanced Market Proxy + Comprehensive Strategy Search")
    print("=" * 78)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===== Step 1: Load all data =====
    print("Step 1: Loading data...")

    # Load predictions
    pred_is, model_source_is = load_predictions("IS")
    pred_oos, model_source_oos = load_predictions("OOS")

    if pred_is is None:
        print("\n  [ERROR] No model predictions available for IS period.")
        print("  Please run one of:")
        print("    - Analyst 1's expanded model training")
        print("    - python scripts/generate_real_predictions.py")
        print("\n  Exiting gracefully.")
        return

    model_source = model_source_is
    print(f"  IS predictions: {len(pred_is)} days ({model_source_is})")
    if pred_oos is not None:
        print(f"  OOS predictions: {len(pred_oos)} days ({model_source_oos})")
    else:
        print("  OOS predictions: NOT AVAILABLE (will skip OOS)")

    # Load Kalshi market data
    kalshi_is = load_kalshi_data("IS")
    kalshi_oos = load_kalshi_data("OOS")

    if kalshi_is is None:
        print("\n  [ERROR] Kalshi IS data not found. Exiting.")
        return

    print(f"  Kalshi IS: {len(kalshi_is)} rows, {kalshi_is['date'].nunique()} days")
    if kalshi_oos is not None:
        print(f"  Kalshi OOS: {len(kalshi_oos)} rows, {kalshi_oos['date'].nunique()} days")

    # Load TMAX history
    tmax_history = load_tmax_history()
    if tmax_history is None:
        print("\n  [ERROR] TMAX history not found. Exiting.")
        return
    print(f"  TMAX history: {len(tmax_history)} days")
    print()

    # ===== Step 2: Build market proxies =====
    print("Step 2: Building market proxies...")

    # Enhanced proxy (trained on data before 2023 for IS)
    enhanced_proxy = MarketProxy(tmax_history)
    enhanced_proxy.fit(train_end_date="2022-12-31")
    print("  Enhanced proxy fitted (train data up to 2022-12-31)")

    diag = enhanced_proxy.get_diagnostics()
    print(f"    Regression coefs: {[f'{c:.4f}' for c in diag['regression_coefs']]}")
    print(f"    Overall sigma: {diag['overall_sigma']:.2f}")

    # Naive proxy for comparison
    naive_proxy = NaiveMarketProxy()
    print("  Naive proxy initialized (40/60 blend)")
    print()

    # ===== Step 3: Add proxy probabilities =====
    print("Step 3: Computing proxy probabilities...")

    # Add enhanced proxy probabilities
    kalshi_is_enhanced = add_proxy_probabilities(kalshi_is, enhanced_proxy, tmax_history)
    # Save enhanced proxy column, add naive proxy, then rename
    kalshi_is_enhanced["enhanced_proxy_prob"] = kalshi_is_enhanced["proxy_prob"]
    kalshi_is_temp = add_proxy_probabilities(
        kalshi_is_enhanced.drop(columns=["proxy_prob"]), naive_proxy, tmax_history,
    )
    kalshi_is_enhanced["naive_proxy_prob"] = kalshi_is_temp["proxy_prob"]
    # Restore proxy_prob to the enhanced version
    kalshi_is_enhanced["proxy_prob"] = kalshi_is_enhanced["enhanced_proxy_prob"]
    kalshi_is_enhanced = kalshi_is_enhanced.drop(columns=["enhanced_proxy_prob"])

    print(f"  IS enhanced proxy: mean={kalshi_is_enhanced['proxy_prob'].mean():.3f}")
    print(f"  IS naive proxy: mean={kalshi_is_enhanced['naive_proxy_prob'].mean():.3f}")

    if kalshi_oos is not None:
        # For OOS, refit proxy with data up to end of 2024
        enhanced_proxy_oos = MarketProxy(tmax_history)
        enhanced_proxy_oos.fit(train_end_date="2024-12-31")
        print("  Enhanced proxy refitted for OOS (train data up to 2024-12-31)")

        # Add enhanced proxy for OOS
        kalshi_oos_enhanced = add_proxy_probabilities(kalshi_oos, enhanced_proxy_oos, tmax_history)
        kalshi_oos_enhanced["enhanced_proxy_prob"] = kalshi_oos_enhanced["proxy_prob"]
        kalshi_oos_temp = add_proxy_probabilities(
            kalshi_oos_enhanced.drop(columns=["proxy_prob"]), naive_proxy, tmax_history,
        )
        kalshi_oos_enhanced["naive_proxy_prob"] = kalshi_oos_temp["proxy_prob"]
        kalshi_oos_enhanced["proxy_prob"] = kalshi_oos_enhanced["enhanced_proxy_prob"]
        kalshi_oos_enhanced = kalshi_oos_enhanced.drop(columns=["enhanced_proxy_prob"])

        print(f"  OOS enhanced proxy: mean={kalshi_oos_enhanced['proxy_prob'].mean():.3f}")
        print(f"  OOS naive proxy: mean={kalshi_oos_enhanced['naive_proxy_prob'].mean():.3f}")
    else:
        kalshi_oos_enhanced = None
    print()

    # ===== Step 4: Align model predictions =====
    print("Step 4: Aligning model predictions with market data...")

    comparison_is = align_model_and_market(kalshi_is_enhanced, pred_is)
    # Merge proxy columns back (build_historical_comparison may have dropped them)
    # We need to ensure proxy_prob and naive_proxy_prob are in comparison_is
    if "proxy_prob" not in comparison_is.columns:
        proxy_lookup = kalshi_is_enhanced[["date", "ticker", "proxy_prob", "naive_proxy_prob"]].copy()
        proxy_lookup["date"] = pd.to_datetime(proxy_lookup["date"]).dt.date
        comparison_is["date"] = comparison_is["date"].apply(
            lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
        )
        comparison_is = comparison_is.merge(
            proxy_lookup, on=["date", "ticker"], how="left",
        )

    print(f"  IS aligned: {len(comparison_is)} rows, {comparison_is['date'].nunique()} dates")

    comparison_oos = None
    if kalshi_oos_enhanced is not None and pred_oos is not None:
        comparison_oos = align_model_and_market(kalshi_oos_enhanced, pred_oos)
        if "proxy_prob" not in comparison_oos.columns:
            proxy_lookup_oos = kalshi_oos_enhanced[["date", "ticker", "proxy_prob", "naive_proxy_prob"]].copy()
            proxy_lookup_oos["date"] = pd.to_datetime(proxy_lookup_oos["date"]).dt.date
            comparison_oos["date"] = comparison_oos["date"].apply(
                lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
            )
            comparison_oos = comparison_oos.merge(
                proxy_lookup_oos, on=["date", "ticker"], how="left",
            )
        print(f"  OOS aligned: {len(comparison_oos)} rows, {comparison_oos['date'].nunique()} dates")
    print()

    # ===== Step 5: Strategy grid search (IS) =====
    print("Step 5: Running strategy grid search on IS data...")
    print("  (Using enhanced proxy as market price...)")
    all_results, strategies_df, backtest_data_is = run_strategy_grid_search(
        comparison_is, proxy_col="proxy_prob",
    )

    trading = strategies_df[strategies_df["n_trades"] > 0]
    profitable = strategies_df[strategies_df["total_pnl"] > 0]
    print(f"  Strategies evaluated: {len(strategies_df)}")
    print(f"  With trades: {len(trading)}")
    print(f"  Profitable: {len(profitable)}")

    # Save all strategies
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

    # Save config
    with open(os.path.join(OUTPUT_DIR, "best_strategy_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    # Find the IS result for the best strategy
    is_best_result = None
    for r in all_results:
        if r.strategy_name == best_config.get("strategy_name"):
            is_best_result = r
            break
    print()

    # ===== Step 7: OOS Validation =====
    print("Step 7: Running OOS validation...")
    oos_result = None
    backtest_data_oos = None
    if comparison_oos is not None and "error" not in best_config:
        oos_result, backtest_data_oos = run_oos_backtest(
            comparison_oos, best_config, proxy_col="proxy_prob",
        )
        print(f"  OOS PnL: ${oos_result.total_pnl:.2f}")
        print(f"  OOS Sharpe: {oos_result.sharpe_ratio:.2f}")
        print(f"  OOS Win Rate: {oos_result.win_rate*100:.1f}%")
        print(f"  OOS Trades: {oos_result.n_trades}")
    else:
        print("  Skipping OOS (no data or no valid strategy)")
    print()

    # ===== Step 8: Brier Scores =====
    print("Step 8: Computing Brier scores...")
    brier_is = compute_all_brier_scores(comparison_is, "proxy_prob", "naive_proxy_prob")
    brier_oos = None
    if comparison_oos is not None:
        brier_oos = compute_all_brier_scores(comparison_oos, "proxy_prob", "naive_proxy_prob")

    # Print summary
    for label, brier in [("IS", brier_is), ("OOS", brier_oos)]:
        if brier is None:
            continue
        print(f"\n  {label} Brier Scores:")
        for comp, bdata in brier.items():
            if comp == "seasonal" or not isinstance(bdata, dict):
                continue
            print(f"    {comp}: model={bdata.get('model_brier', 0):.4f}, "
                  f"comp={bdata.get('market_brier', 0):.4f}, "
                  f"delta={bdata.get('brier_delta', 0):.4f}")

    # Save Brier analysis
    with open(os.path.join(OUTPUT_DIR, "brier_analysis.json"), "w") as f:
        json.dump({"IS": brier_is, "OOS": brier_oos}, f, indent=2, default=str)
    print()

    # ===== Step 9: Generate plots =====
    print("Step 9: Generating plots...")
    saved_plots = generate_plots(
        comparison_is, comparison_oos,
        is_best_result, oos_result,
        brier_is, brier_oos,
        OUTPUT_DIR,
    )
    print(f"  Generated {len(saved_plots)} plots")
    print()

    # ===== Step 10: Generate report =====
    print("Step 10: Generating comprehensive report...")
    report = generate_report(
        is_best_result, oos_result, best_config,
        brier_is, brier_oos,
        strategies_df, model_source, OUTPUT_DIR,
    )
    print()
    print(report)

    # ===== Save IS/OOS comparison table =====
    if is_best_result and oos_result:
        comparison_table = pd.DataFrame([
            {"period": "IS (2023-2024)", **is_best_result.to_summary_dict()},
            {"period": "OOS (2025)", **oos_result.to_summary_dict()},
        ])
        comparison_table.to_csv(
            os.path.join(OUTPUT_DIR, "is_vs_oos_comparison.csv"), index=False,
        )

    # ===== Save proxy diagnostics =====
    proxy_diag = {
        "enhanced_is": enhanced_proxy.get_diagnostics(),
    }
    if kalshi_oos_enhanced is not None:
        proxy_diag["enhanced_oos"] = enhanced_proxy_oos.get_diagnostics()
    with open(os.path.join(OUTPUT_DIR, "proxy_diagnostics.json"), "w") as f:
        json.dump(proxy_diag, f, indent=2, default=str)

    # ===== List output files =====
    print("\nOutput files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fpath} ({size_kb:.1f} KB)")

    print()
    print("=" * 78)
    print("  EXPANDED BACKTEST COMPLETE!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
