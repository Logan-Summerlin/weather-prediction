#!/usr/bin/env python3
"""
Step 4: Integrate MOS Proxy into Full Kalshi Backtest.

Runs the same comprehensive IS/OOS backtest as run_max_train_backtest.py,
but additionally includes the MOSMarketProxy as a comparison.

Compares:
  1. NN model vs MOS proxy
  2. NN model vs Enhanced proxy (existing)
  3. NN model vs Kalshi market
  4. Strategy profitability against MOS proxy vs enhanced proxy

Outputs:
    results/mos_backtest/  (all results, plots, reports)

Usage:
    python scripts/run_mos_backtest.py
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "mos_backtest")

# MOS files
MOS_COMBINED = os.path.join(DATA_DIR, "mos", "combined_mos_knyc.csv")
MOS_PROXY_MODULE = os.path.join(PROJECT_ROOT, "src", "mos_market_proxy.py")

# Prediction files
MAX_TRAIN_NN_IS = os.path.join(DATA_DIR, "max_train_nn_predictions_is.csv")
MAX_TRAIN_NN_OOS = os.path.join(DATA_DIR, "max_train_nn_predictions_oos.csv")
MAX_TRAIN_RIDGE_IS = os.path.join(DATA_DIR, "max_train_ridge_predictions_is.csv")
MAX_TRAIN_RIDGE_OOS = os.path.join(DATA_DIR, "max_train_ridge_predictions_oos.csv")

# Real Kalshi market data
KALSHI_IS = os.path.join(DATA_DIR, "real_kalshi_2023_2024.csv")
KALSHI_OOS = os.path.join(DATA_DIR, "real_kalshi_2025.csv")

# Central Park TMAX full history
TMAX_HISTORY = os.path.join(DATA_DIR, "central_park_tmax_full_history.csv")

MAX_WAIT_SECONDS = 900
POLL_INTERVAL = 30

# Strategy grid (matches existing backtest)
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


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Wait for dependencies
# ---------------------------------------------------------------------------
def wait_for_dependencies():
    """Wait for MOS data AND MOSMarketProxy to be available."""
    required_files = [MOS_COMBINED, MOS_PROXY_MODULE]
    labels = ["MOS combined data", "MOSMarketProxy module"]

    all_ready = all(os.path.exists(f) for f in required_files)
    if all_ready:
        logger.info("All MOS dependencies found immediately")
        return True

    print("  [WAITING] Not all MOS dependencies are available yet.")
    for f, label in zip(required_files, labels):
        status = "FOUND" if os.path.exists(f) else "MISSING"
        print(f"    {label}: {status} ({f})")
    print(f"  Checking every {POLL_INTERVAL}s for up to {MAX_WAIT_SECONDS}s...")

    elapsed = 0
    while elapsed < MAX_WAIT_SECONDS:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        all_ready = all(os.path.exists(f) for f in required_files)
        if all_ready:
            print(f"  [FOUND] All dependencies detected after {elapsed}s.")
            return True
        still_missing = [l for f, l in zip(required_files, labels) if not os.path.exists(f)]
        print(f"  Still waiting for: {', '.join(still_missing)} ({elapsed}s / {MAX_WAIT_SECONDS}s)")

    print(f"\n  [ERROR] Dependencies not found after {MAX_WAIT_SECONDS}s.")
    return False


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_nn_predictions(period="IS"):
    path = MAX_TRAIN_NN_IS if period == "IS" else MAX_TRAIN_NN_OOS
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("Loaded NN predictions (%s): %d rows", period, len(df))
        return df
    return None


def load_ridge_predictions(period="IS"):
    path = MAX_TRAIN_RIDGE_IS if period == "IS" else MAX_TRAIN_RIDGE_OOS
    if os.path.exists(path):
        df = pd.read_csv(path)
        logger.info("Loaded Ridge predictions (%s): %d rows", period, len(df))
        return df
    return None


def load_kalshi_data(period="IS"):
    path = KALSHI_IS if period == "IS" else KALSHI_OOS
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    logger.info("Loaded Kalshi %s: %d rows, %d days", period, len(df), df["date"].nunique())
    return df


def load_tmax_history():
    df = pd.read_csv(TMAX_HISTORY)
    logger.info("Loaded TMAX history: %d rows", len(df))
    return df


def load_mos_data():
    df = pd.read_csv(MOS_COMBINED)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Loaded MOS data: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Add proxy probabilities
# ---------------------------------------------------------------------------
def add_mos_proxy_probabilities(kalshi_df, mos_proxy, mos_data_df, tmax_history_df):
    """Add MOS proxy probabilities to Kalshi data.

    For each market row, looks up the MOS forecast for that date,
    then computes the proxy bracket probability.
    """
    from src.mos_market_proxy import MOSMarketProxy

    result = kalshi_df.copy()
    result["date_obj"] = pd.to_datetime(result["date"]).dt.date

    # Build MOS forecast lookup
    mos_lookup = {}
    for _, row in mos_data_df.iterrows():
        d = row["date"]
        mos_lookup[d] = row

    proxy_probs = []
    for _, row in result.iterrows():
        target_date = row["date_obj"]
        mos_row = mos_lookup.get(target_date)

        try:
            prob = mos_proxy.compute_bracket_prob(
                target_date=target_date,
                threshold_low=row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                threshold_high=row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                direction=row.get("direction", "between"),
            )
            proxy_probs.append(prob)
        except Exception as e:
            logger.warning("MOS proxy error for %s: %s", target_date, e)
            proxy_probs.append(np.nan)

    result["mos_proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])

    n_valid = sum(1 for p in proxy_probs if p is not None and not np.isnan(p))
    logger.info("Added MOS proxy: %d/%d valid", n_valid, len(result))
    return result


def add_enhanced_proxy_probabilities(kalshi_df, proxy, tmax_history_df):
    """Add enhanced proxy probabilities (same as in run_max_train_backtest.py)."""
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
            logger.warning("Enhanced proxy error for %s: %s", target_date, e)
            proxy_probs.append(np.nan)

    result["enhanced_proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])
    return result


def add_naive_proxy_probabilities(kalshi_df, naive_proxy, tmax_history_df):
    """Add naive proxy probabilities."""
    hist = tmax_history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    tmax_lookup = dict(zip(hist["date"], hist["tmax_f"]))

    result = kalshi_df.copy()
    result["date_obj"] = pd.to_datetime(result["date"]).dt.date

    proxy_probs = []
    for _, row in result.iterrows():
        target_date = row["date_obj"]
        yt = tmax_lookup.get(target_date - timedelta(days=1))
        if yt is None:
            proxy_probs.append(np.nan)
            continue
        try:
            p = naive_proxy.compute_bracket_prob(
                target_date, yt,
                row.get("threshold_low") if pd.notna(row.get("threshold_low")) else None,
                row.get("threshold_high") if pd.notna(row.get("threshold_high")) else None,
                row.get("direction", "between"),
            )
            proxy_probs.append(p)
        except Exception:
            proxy_probs.append(np.nan)

    result["naive_proxy_prob"] = proxy_probs
    result = result.drop(columns=["date_obj"])
    return result


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_model_and_market(market_df, predictions_df):
    """Align model predictions with market data."""
    from src.kalshi_client import build_historical_comparison

    market_clean = market_df.drop(columns=["actual_tmax"], errors="ignore")

    comparison_df = build_historical_comparison(
        model_predictions_df=predictions_df,
        historical_markets_df=market_clean,
    )

    if comparison_df.empty:
        raise ValueError("No overlapping dates between model and market data")

    if "actual_outcome" in comparison_df.columns:
        comparison_df["outcome"] = comparison_df["actual_outcome"].astype(float)

    logger.info("Aligned %d rows across %d dates",
                len(comparison_df), comparison_df["date"].nunique())
    return comparison_df


def merge_proxy_cols(comparison_df, kalshi_df, proxy_cols):
    """Merge proxy columns from kalshi_df into comparison_df."""
    available = [c for c in proxy_cols if c in kalshi_df.columns]
    if not available:
        return

    lookup = kalshi_df[["date", "ticker"] + available].copy()
    lookup["date"] = pd.to_datetime(lookup["date"]).dt.date
    comparison_df["date"] = comparison_df["date"].apply(
        lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
    )

    for col in available:
        if col not in comparison_df.columns:
            sub = lookup[["date", "ticker", col]].drop_duplicates()
            tmp = comparison_df.merge(sub, on=["date", "ticker"], how="left", suffixes=("", "_new"))
            if col + "_new" in tmp.columns:
                comparison_df[col] = tmp[col + "_new"]
            elif col in tmp.columns:
                comparison_df[col] = tmp[col]


# ---------------------------------------------------------------------------
# Strategy Grid Search
# ---------------------------------------------------------------------------
def run_strategy_grid_search(comparison_df, proxy_col="mos_proxy_prob"):
    """Run strategy grid search on IS data."""
    from src.trading import TradingStrategy, BacktestEngine, generate_strategy_grid

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

    logger.info("Grid search: %d strategies, %d with trades, %d profitable",
                len(all_results),
                sum(1 for r in all_results if r.n_trades > 0),
                sum(1 for r in all_results if r.total_pnl > 0))
    return all_results, strategies_df, backtest_data


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
    from src.trading import TradingStrategy
    return TradingStrategy(
        name=config.get("strategy_name", "OOS_Frozen"),
        ev_threshold=config.get("ev_threshold", 0.02),
        sizing_method=config.get("sizing_method", "fractional_kelly"),
        kelly_fraction=config.get("kelly_fraction", 0.25),
        max_position_frac=config.get("max_position_frac", 0.10),
        fee_rate=config.get("fee_rate", 0.07),
        bankroll=config.get("bankroll", 10000),
    )


# ---------------------------------------------------------------------------
# OOS Backtest
# ---------------------------------------------------------------------------
def run_oos_backtest(comparison_df, strategy_config, proxy_col="mos_proxy_prob"):
    from src.trading import BacktestEngine
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

    logger.info("OOS backtest: %d trades, PnL=$%.2f, Sharpe=%.2f, WR=%.1f%%",
                result.n_trades, result.total_pnl, result.sharpe_ratio, result.win_rate * 100)
    return result, backtest_data


# ---------------------------------------------------------------------------
# Brier Score Analysis
# ---------------------------------------------------------------------------
def compute_all_brier_scores(comparison_df, proxy_cols, labels):
    """Compute Brier scores for model vs each proxy."""
    from src.kalshi_client import compute_brier_scores

    df = comparison_df.dropna(subset=["model_prob", "outcome"]).copy()
    results = {}

    for proxy_col, label in zip(proxy_cols, labels):
        if proxy_col not in df.columns:
            continue
        df_valid = df.dropna(subset=[proxy_col])
        if len(df_valid) > 0:
            results[f"model_vs_{label}"] = compute_brier_scores(
                df_valid["model_prob"], df_valid[proxy_col], df_valid["outcome"],
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
            "n": int(mask.sum()),
        }
        for proxy_col, label in zip(proxy_cols, labels):
            if proxy_col in s_df.columns:
                s_valid = s_df.dropna(subset=[proxy_col])
                if len(s_valid) > 0:
                    entry[f"{label}_brier"] = float(
                        np.mean((s_valid[proxy_col] - s_valid["outcome"]) ** 2)
                    )
        if "market_prob" in s_df.columns:
            s_mkt = s_df.dropna(subset=["market_prob"])
            if len(s_mkt) > 0:
                entry["kalshi_brier"] = float(np.mean((s_mkt["market_prob"] - s_mkt["outcome"]) ** 2))
        seasonal[season] = entry
    results["seasonal"] = seasonal

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def generate_plots(comparison_is, comparison_oos, is_result_mos, oos_result_mos,
                   is_result_enhanced, oos_result_enhanced,
                   brier_is, brier_oos, output_dir):
    """Generate MOS backtest comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # ---- 1. Brier Score Comparison Bar Chart ----
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
            ("model_vs_mos_proxy", "MOS Proxy"),
            ("model_vs_enhanced_proxy", "Enhanced Proxy"),
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
    path = os.path.join(output_dir, "brier_comparison_with_mos.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. Model vs MOS Proxy Scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [
        (axes[0], comparison_is, "IS (2023-2024)"),
        (axes[1], comparison_oos, "OOS (2025)"),
    ]:
        if df is not None and "model_prob" in df.columns and "mos_proxy_prob" in df.columns:
            valid = df.dropna(subset=["model_prob", "mos_proxy_prob"])
            if len(valid) > 0:
                ax.scatter(valid["mos_proxy_prob"], valid["model_prob"],
                           alpha=0.3, s=10, edgecolors="none", c="#1f77b4")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                ax.set_xlabel("MOS Proxy Probability")
                ax.set_ylabel("Model Probability")
                ax.set_title(f"Model vs MOS Proxy: {title}")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_vs_mos_proxy_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 3. PnL Comparison: MOS proxy vs Enhanced proxy ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax, mos_result, enh_result, period in [
        (axes[0], is_result_mos, is_result_enhanced, "IS (2023-2024)"),
        (axes[1], oos_result_mos, oos_result_enhanced, "OOS (2025)"),
    ]:
        plotted = False
        if mos_result and mos_result.trades:
            mos_pnls = np.cumsum([t["pnl"] for t in mos_result.trades])
            ax.plot(range(len(mos_pnls)), mos_pnls,
                    label=f"vs MOS Proxy: PnL=${mos_result.total_pnl:.0f}, Sharpe={mos_result.sharpe_ratio:.2f}",
                    linewidth=1.5, color="#1f77b4")
            plotted = True
        if enh_result and enh_result.trades:
            enh_pnls = np.cumsum([t["pnl"] for t in enh_result.trades])
            ax.plot(range(len(enh_pnls)), enh_pnls,
                    label=f"vs Enhanced Proxy: PnL=${enh_result.total_pnl:.0f}, Sharpe={enh_result.sharpe_ratio:.2f}",
                    linewidth=1.5, color="#d62728")
            plotted = True
        if plotted:
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Trade Number")
            ax.set_ylabel("Cumulative P&L ($)")
            ax.set_title(f"P&L: {period}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "pnl_mos_vs_enhanced.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 4. Seasonal Brier Breakdown ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, brier, label in [(axes[0], brier_is, "IS"), (axes[1], brier_oos, "OOS")]:
        if brier is None:
            continue
        seasonal = brier.get("seasonal", {})
        if not seasonal:
            continue

        seasons = ["Winter", "Spring", "Summer", "Fall"]
        x = np.arange(len(seasons))
        width = 0.2

        model_b = [seasonal.get(s, {}).get("model_brier", np.nan) for s in seasons]
        mos_b = [seasonal.get(s, {}).get("mos_proxy_brier", np.nan) for s in seasons]
        enh_b = [seasonal.get(s, {}).get("enhanced_proxy_brier", np.nan) for s in seasons]
        kalshi_b = [seasonal.get(s, {}).get("kalshi_brier", np.nan) for s in seasons]

        ax.bar(x - 1.5 * width, model_b, width, label="NN Model", color="#4c72b0", alpha=0.8)
        ax.bar(x - 0.5 * width, mos_b, width, label="MOS Proxy", color="#2ca02c", alpha=0.8)
        ax.bar(x + 0.5 * width, enh_b, width, label="Enhanced Proxy", color="#d62728", alpha=0.8)
        ax.bar(x + 1.5 * width, kalshi_b, width, label="Kalshi", color="#9467bd", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(seasons, fontsize=8)
        ax.set_ylabel("Brier Score")
        ax.set_title(f"Seasonal Brier: {label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "seasonal_brier_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(mos_strategies_df, enh_strategies_df,
                    is_result_mos, oos_result_mos,
                    is_result_enh, oos_result_enh,
                    best_mos_config, best_enh_config,
                    brier_is, brier_oos, output_dir):
    """Generate comprehensive text report."""
    lines = [
        "=" * 78,
        "  MOS BACKTEST COMPARISON REPORT",
        "  MOS Proxy vs Enhanced Proxy vs Kalshi Market",
        "=" * 78,
        "",
    ]

    # ---- Brier Scores ----
    lines.append("--- BRIER SCORE COMPARISON ---")
    for period_label, brier in [("IS (2023-2024)", brier_is), ("OOS (2025)", brier_oos)]:
        if brier is None:
            continue
        lines.append(f"\n  {period_label}:")
        for comparison, bdata in brier.items():
            if comparison in ("seasonal",) or not isinstance(bdata, dict):
                continue
            if "model_brier" not in bdata:
                continue
            delta = bdata.get('brier_delta', 0)
            winner = "MODEL BETTER" if delta < 0 else "COMPARISON BETTER"
            lines.append(
                f"    {comparison}: model={bdata['model_brier']:.4f}, "
                f"comp={bdata['market_brier']:.4f}, delta={delta:.4f} ({winner})"
            )

        seasonal = brier.get("seasonal", {})
        if seasonal:
            lines.append(f"\n    Seasonal ({period_label}):")
            for season in ["Winter", "Spring", "Summer", "Fall"]:
                sdata = seasonal.get(season, {})
                if sdata:
                    parts = [f"model={sdata['model_brier']:.4f}"]
                    for k in sorted(sdata.keys()):
                        if k.endswith("_brier") and k != "model_brier":
                            parts.append(f"{k}={sdata[k]:.4f}")
                    parts.append(f"n={sdata['n']}")
                    lines.append(f"      {season:8s}: {', '.join(parts)}")
    lines.append("")

    # ---- Strategy Grid Search Results ----
    for label, sdf in [("MOS Proxy", mos_strategies_df), ("Enhanced Proxy", enh_strategies_df)]:
        lines.append(f"--- STRATEGY GRID: {label} ---")
        if sdf is not None and not sdf.empty:
            trading = sdf[sdf["n_trades"] > 0]
            profitable = sdf[sdf["total_pnl"] > 0]
            lines.append(f"  Total: {len(sdf)}, With trades: {len(trading)}, Profitable: {len(profitable)}")
            if len(trading) > 0:
                lines.append(f"  Mean PnL: ${trading['total_pnl'].mean():.2f}")
                lines.append(f"  Best PnL: ${trading['total_pnl'].max():.2f}")
        lines.append("")

    # ---- Selected Strategies ----
    for label, config in [("MOS Proxy", best_mos_config), ("Enhanced Proxy", best_enh_config)]:
        lines.append(f"--- SELECTED STRATEGY ({label}) ---")
        if config and "error" not in config:
            lines.append(f"  Name: {config.get('strategy_name', 'N/A')}")
            lines.append(f"  IS PnL: ${config.get('total_pnl', 0):.2f}")
            lines.append(f"  IS Sharpe: {config.get('sharpe_ratio', 0):.2f}")
            lines.append(f"  IS Win Rate: {config.get('win_rate', 0) * 100:.1f}%")
        elif config:
            lines.append(f"  Error: {config.get('error', 'N/A')}")
        lines.append("")

    # ---- IS vs OOS Comparison ----
    lines.append("--- IS vs OOS COMPARISON ---")
    lines.append(f"  {'Metric':<20s} {'MOS IS':>10s} {'MOS OOS':>10s} {'Enh IS':>10s} {'Enh OOS':>10s}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    metrics = [
        ("PnL ($)", "total_pnl", "${:.0f}"),
        ("Sharpe", "sharpe_ratio", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Trades", "n_trades", "{}"),
    ]
    for label, key, fmt in metrics:
        vals = []
        for result in [is_result_mos, oos_result_mos, is_result_enh, oos_result_enh]:
            if result:
                d = result.to_summary_dict()
                try:
                    vals.append(fmt.format(d.get(key, 0)))
                except (ValueError, TypeError):
                    vals.append("N/A")
            else:
                vals.append("N/A")
        lines.append(f"  {label:<20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s}")
    lines.append("")

    lines.extend(["", "=" * 78])

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "mos_backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  STEP 4: MOS PROXY BACKTEST")
    print("  Comparing NN Model vs MOS Proxy vs Enhanced Proxy vs Kalshi")
    print("=" * 78)
    print()

    # Wait for dependencies
    if not wait_for_dependencies():
        print("  FATAL: MOS dependencies not available. Exiting.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import MOS proxy
    from src.mos_market_proxy import MOSMarketProxy
    from src.enhanced_market_proxy import EnhancedMarketProxy
    from src.market_proxy import NaiveMarketProxy
    from src.kalshi_client import build_historical_comparison, compute_brier_scores
    from src.kalshi_backtester import CalibrationAnalyzer, compute_seasonal_pnl
    from src.trading import TradingStrategy, BacktestEngine, generate_strategy_grid

    # ===== Step 1: Load all data =====
    print("\nStep 1: Loading data...")
    nn_is = load_nn_predictions("IS")
    nn_oos = load_nn_predictions("OOS")
    kalshi_is = load_kalshi_data("IS")
    kalshi_oos = load_kalshi_data("OOS")
    tmax_history = load_tmax_history()
    mos_data = load_mos_data()

    if nn_is is None:
        print("  [ERROR] NN IS predictions not found. Exiting.")
        sys.exit(1)
    if kalshi_is is None:
        print("  [ERROR] Kalshi IS data not found. Exiting.")
        sys.exit(1)
    if tmax_history is None:
        print("  [ERROR] TMAX history not found. Exiting.")
        sys.exit(1)

    print(f"  NN IS: {len(nn_is)} days")
    print(f"  NN OOS: {len(nn_oos)} days" if nn_oos is not None else "  NN OOS: not available")
    print(f"  Kalshi IS: {len(kalshi_is)} rows")
    print(f"  Kalshi OOS: {len(kalshi_oos)} rows" if kalshi_oos is not None else "  Kalshi OOS: not available")
    print(f"  TMAX history: {len(tmax_history)} rows")
    print(f"  MOS data: {len(mos_data)} rows")
    print()

    # ===== Step 2: Build proxies =====
    print("Step 2: Building market proxies...")

    # MOS proxy
    mos_proxy_is = MOSMarketProxy(mos_data, tmax_history)
    mos_proxy_is.fit(train_end_date="2022-12-31")
    print(f"  MOS proxy (IS): fitted")

    # Enhanced proxy for IS
    enhanced_proxy_is = EnhancedMarketProxy(tmax_history)
    enhanced_proxy_is.fit(train_end_date="2022-12-31")
    print(f"  Enhanced proxy (IS): fitted")

    # Naive proxy
    naive_proxy = NaiveMarketProxy()

    # OOS proxies
    mos_proxy_oos = None
    enhanced_proxy_oos = None
    if kalshi_oos is not None:
        mos_proxy_oos = MOSMarketProxy(mos_data, tmax_history)
        mos_proxy_oos.fit(train_end_date="2024-12-31")
        enhanced_proxy_oos = EnhancedMarketProxy(tmax_history)
        enhanced_proxy_oos.fit(train_end_date="2024-12-31")
        print(f"  MOS proxy (OOS): fitted")
        print(f"  Enhanced proxy (OOS): fitted")
    print()

    # ===== Step 3: Compute proxy probabilities =====
    print("Step 3: Computing proxy probabilities...")

    # IS
    kalshi_is_full = add_mos_proxy_probabilities(kalshi_is, mos_proxy_is, mos_data, tmax_history)
    kalshi_is_full = add_enhanced_proxy_probabilities(kalshi_is_full, enhanced_proxy_is, tmax_history)
    kalshi_is_full = add_naive_proxy_probabilities(kalshi_is_full, naive_proxy, tmax_history)

    for col in ["mos_proxy_prob", "enhanced_proxy_prob", "naive_proxy_prob"]:
        if col in kalshi_is_full.columns:
            n_valid = kalshi_is_full[col].notna().sum()
            print(f"  IS {col}: {n_valid}/{len(kalshi_is_full)} valid, mean={kalshi_is_full[col].mean():.3f}")

    # OOS
    kalshi_oos_full = None
    if kalshi_oos is not None and mos_proxy_oos is not None:
        kalshi_oos_full = add_mos_proxy_probabilities(kalshi_oos, mos_proxy_oos, mos_data, tmax_history)
        kalshi_oos_full = add_enhanced_proxy_probabilities(kalshi_oos_full, enhanced_proxy_oos, tmax_history)
        kalshi_oos_full = add_naive_proxy_probabilities(kalshi_oos_full, naive_proxy, tmax_history)
        for col in ["mos_proxy_prob", "enhanced_proxy_prob"]:
            if col in kalshi_oos_full.columns:
                n_valid = kalshi_oos_full[col].notna().sum()
                print(f"  OOS {col}: {n_valid}/{len(kalshi_oos_full)} valid")
    print()

    # ===== Step 4: Align model predictions =====
    print("Step 4: Aligning model predictions...")
    nn_comparison_is = align_model_and_market(kalshi_is_full, nn_is)
    merge_proxy_cols(nn_comparison_is, kalshi_is_full,
                     ["mos_proxy_prob", "enhanced_proxy_prob", "naive_proxy_prob"])
    print(f"  NN IS aligned: {len(nn_comparison_is)} rows, {nn_comparison_is['date'].nunique()} dates")

    nn_comparison_oos = None
    if kalshi_oos_full is not None and nn_oos is not None:
        nn_comparison_oos = align_model_and_market(kalshi_oos_full, nn_oos)
        merge_proxy_cols(nn_comparison_oos, kalshi_oos_full,
                         ["mos_proxy_prob", "enhanced_proxy_prob", "naive_proxy_prob"])
        print(f"  NN OOS aligned: {len(nn_comparison_oos)} rows")
    print()

    # ===== Step 5: Strategy grid search - MOS proxy =====
    print("Step 5: Strategy grid search vs MOS proxy (IS)...")
    mos_results, mos_strategies_df, _ = run_strategy_grid_search(
        nn_comparison_is, proxy_col="mos_proxy_prob",
    )
    mos_trading = mos_strategies_df[mos_strategies_df["n_trades"] > 0]
    mos_profitable = mos_strategies_df[mos_strategies_df["total_pnl"] > 0]
    print(f"  MOS: {len(mos_strategies_df)} strategies, {len(mos_trading)} with trades, {len(mos_profitable)} profitable")

    # Also run grid search vs Enhanced proxy for comparison
    print("\n  Strategy grid search vs Enhanced proxy (IS)...")
    enh_results, enh_strategies_df, _ = run_strategy_grid_search(
        nn_comparison_is, proxy_col="enhanced_proxy_prob",
    )
    enh_trading = enh_strategies_df[enh_strategies_df["n_trades"] > 0]
    enh_profitable = enh_strategies_df[enh_strategies_df["total_pnl"] > 0]
    print(f"  Enhanced: {len(enh_strategies_df)} strategies, {len(enh_trading)} with trades, {len(enh_profitable)} profitable")

    # Save grid results
    mos_strategies_df.to_csv(os.path.join(OUTPUT_DIR, "mos_strategies_is.csv"), index=False)
    enh_strategies_df.to_csv(os.path.join(OUTPUT_DIR, "enhanced_strategies_is.csv"), index=False)
    print()

    # ===== Step 6: Select best strategies =====
    print("Step 6: Selecting best strategies...")
    best_mos_config, _ = select_best_strategy(mos_strategies_df)
    best_enh_config, _ = select_best_strategy(enh_strategies_df)

    for label, config in [("MOS proxy", best_mos_config), ("Enhanced proxy", best_enh_config)]:
        if "error" in config:
            print(f"  {label}: {config['error']}")
        else:
            print(f"  {label}: {config.get('strategy_name', 'N/A')}, "
                  f"PnL=${config.get('total_pnl', 0):.0f}, Sharpe={config.get('sharpe_ratio', 0):.2f}")

    with open(os.path.join(OUTPUT_DIR, "best_mos_strategy_config.json"), "w") as f:
        json.dump(best_mos_config, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "best_enh_strategy_config.json"), "w") as f:
        json.dump(best_enh_config, f, indent=2)

    # Find IS results
    is_mos_result = None
    for r in mos_results:
        if r.strategy_name == best_mos_config.get("strategy_name"):
            is_mos_result = r
            break
    is_enh_result = None
    for r in enh_results:
        if r.strategy_name == best_enh_config.get("strategy_name"):
            is_enh_result = r
            break
    print()

    # ===== Step 7: OOS validation =====
    print("Step 7: OOS validation...")
    oos_mos_result = None
    oos_enh_result = None

    if nn_comparison_oos is not None:
        if "error" not in best_mos_config:
            oos_mos_result, _ = run_oos_backtest(
                nn_comparison_oos, best_mos_config, proxy_col="mos_proxy_prob",
            )
            print(f"  MOS OOS: PnL=${oos_mos_result.total_pnl:.2f}, Sharpe={oos_mos_result.sharpe_ratio:.2f}")

        if "error" not in best_enh_config:
            oos_enh_result, _ = run_oos_backtest(
                nn_comparison_oos, best_enh_config, proxy_col="enhanced_proxy_prob",
            )
            print(f"  Enhanced OOS: PnL=${oos_enh_result.total_pnl:.2f}, Sharpe={oos_enh_result.sharpe_ratio:.2f}")
    else:
        print("  Skipping OOS (no data)")
    print()

    # ===== Step 8: Brier scores =====
    print("Step 8: Computing Brier scores...")
    proxy_cols = ["mos_proxy_prob", "enhanced_proxy_prob", "naive_proxy_prob"]
    proxy_labels = ["mos_proxy", "enhanced_proxy", "naive_proxy"]

    brier_is = compute_all_brier_scores(nn_comparison_is, proxy_cols, proxy_labels)
    brier_oos = None
    if nn_comparison_oos is not None:
        brier_oos = compute_all_brier_scores(nn_comparison_oos, proxy_cols, proxy_labels)

    for label, brier in [("IS", brier_is), ("OOS", brier_oos)]:
        if brier is None:
            continue
        print(f"\n  {label} Brier Scores:")
        for comp, bdata in brier.items():
            if comp in ("seasonal",) or not isinstance(bdata, dict):
                continue
            if "model_brier" not in bdata:
                continue
            print(f"    {comp}: model={bdata.get('model_brier', 0):.4f}, "
                  f"comp={bdata.get('market_brier', 0):.4f}, "
                  f"delta={bdata.get('brier_delta', 0):.4f}")

    with open(os.path.join(OUTPUT_DIR, "brier_analysis_with_mos.json"), "w") as f:
        json.dump({"IS": brier_is, "OOS": brier_oos}, f, indent=2, default=str)
    print()

    # ===== Step 9: Plots =====
    print("Step 9: Generating plots...")
    saved_plots = generate_plots(
        nn_comparison_is, nn_comparison_oos,
        is_mos_result, oos_mos_result,
        is_enh_result, oos_enh_result,
        brier_is, brier_oos,
        OUTPUT_DIR,
    )
    print(f"  Generated {len(saved_plots)} plots")
    print()

    # ===== Step 10: Report =====
    print("Step 10: Generating report...")
    report = generate_report(
        mos_strategies_df, enh_strategies_df,
        is_mos_result, oos_mos_result,
        is_enh_result, oos_enh_result,
        best_mos_config, best_enh_config,
        brier_is, brier_oos,
        OUTPUT_DIR,
    )
    print()
    print(report)

    # Save comparison table
    comparison_rows = []
    for label, is_r, oos_r in [
        ("MOS Proxy", is_mos_result, oos_mos_result),
        ("Enhanced Proxy", is_enh_result, oos_enh_result),
    ]:
        if is_r:
            comparison_rows.append({"proxy": label, "period": "IS", **is_r.to_summary_dict()})
        if oos_r:
            comparison_rows.append({"proxy": label, "period": "OOS", **oos_r.to_summary_dict()})
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(OUTPUT_DIR, "proxy_comparison_summary.csv"), index=False,
        )

    # List output files
    print("\nOutput files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fpath} ({size_kb:.1f} KB)")

    print()
    print("=" * 78)
    print("  MOS BACKTEST COMPLETE!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
