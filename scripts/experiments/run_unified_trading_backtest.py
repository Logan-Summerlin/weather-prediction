#!/usr/bin/env python3
"""
Unified Trading Backtest Using Pre-Computed Contract Probabilities.

Uses the unified_predictions.csv files (which contain pre-computed
contract-level probabilities from the best U-series models) to run
a more accurate EV-gated trading backtest.

Unlike the base backtest scripts that recompute bucket probabilities
from Gaussian (mu, sigma), this script uses the actual U-series model
outputs directly, giving a more accurate picture of trading P&L.

Tests multiple U-series model variants and reports the best strategy
for each city.

Usage:
    python scripts/run_unified_trading_backtest.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEE_RATE = 0.07
INITIAL_BANKROLL = 1000.0

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Model probability columns in unified_predictions.csv
MODEL_PROB_COLS = [
    "model_prob",           # Base Gaussian
    "u3_mlp_prob",          # U3 contract MLP
    "u4_platt_prob",        # U4 Platt on U3
    "u5_regime_prob",       # U5 Regime conditional
    "u6_ensemble_prob",     # U6 Calibrated ensemble
    "u7_extended_prob",     # U7 Extended MLP
    "u8_cv_prob",           # U8 CV ensemble
    "u9_kitchen_prob",      # U9 Kitchen sink
]

# Trading parameter grid
EV_THRESHOLDS = [0.02, 0.03, 0.04, 0.05]
KELLY_FRACTIONS = [0.10, 0.15, 0.20, 0.25]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_unified_predictions(city: str) -> pd.DataFrame:
    """Load unified predictions for a city."""
    if city == "chi":
        path = PROJECT_ROOT / "results" / "chicago" / "unified_predictions.csv"
    elif city == "phl":
        path = PROJECT_ROOT / "results" / "philadelphia" / "unified_predictions.csv"
    else:
        raise ValueError(f"Unknown city: {city}")

    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded %s unified predictions: %d rows", city.upper(), len(df))
    return df


# ===========================================================================
# Backtest Engine (Contract-Level)
# ===========================================================================

def run_contract_backtest(
    df: pd.DataFrame,
    model_prob_col: str,
    ev_threshold: float = 0.03,
    kelly_fraction: float = 0.15,
    max_contracts: int = 10,
    oos_only: bool = True,
) -> dict:
    """Run EV-gated backtest using pre-computed contract probabilities.

    Uses market_prob as the market price and model_prob_col as the
    model's probability estimate.  The actual_outcome column determines
    trade settlement.
    """
    # Filter to OOS period
    if oos_only and "period" in df.columns:
        work_df = df[df["period"] == "OOS"].copy()
    else:
        work_df = df.copy()

    if len(work_df) == 0:
        return {"total_pnl": 0, "n_trades": 0, "win_rate": 0, "sharpe_ratio": 0}

    # Check if model column exists
    if model_prob_col not in work_df.columns:
        return {"total_pnl": float("nan"), "n_trades": 0, "error": f"Column {model_prob_col} not found"}

    trades = []
    bankroll = INITIAL_BANKROLL
    bankroll_history = []

    dates = np.sort(work_df["date"].unique())

    for date in dates:
        day_df = work_df[work_df["date"] == date]
        day_pnl = 0.0

        for _, row in day_df.iterrows():
            model_prob = row[model_prob_col]
            market_prob = row["market_prob"]
            outcome = row["actual_outcome"]

            if pd.isna(outcome) or pd.isna(model_prob) or pd.isna(market_prob):
                continue

            model_prob = np.clip(model_prob, 0.001, 0.999)
            market_prob = np.clip(market_prob, 0.01, 0.99)

            # Simulate bid-ask spread
            spread = 0.02 + 0.08 * (1.0 - 2.0 * min(market_prob, 1.0 - market_prob))
            ask = np.clip(market_prob + spread / 2, 0.01, 0.99)
            bid = np.clip(market_prob - spread / 2, 0.01, 0.99)

            # EV computation
            ev_yes = model_prob * (1.0 - FEE_RATE) - ask
            ev_no = (1.0 - model_prob) * (1.0 - FEE_RATE) - (1.0 - bid)

            if ev_yes > ev_no:
                best_ev = ev_yes
                direction = "YES"
            else:
                best_ev = ev_no
                direction = "NO"

            if best_ev < ev_threshold:
                continue

            # Kelly sizing
            if direction == "YES":
                p = model_prob
                price = ask
            else:
                p = 1.0 - model_prob
                price = 1.0 - bid

            price = np.clip(price, 0.01, 0.99)
            net_payout = (1.0 - FEE_RATE) - price
            if net_payout <= 0:
                continue

            b = net_payout / price
            if b <= 0:
                continue

            q = 1.0 - p
            full_kelly = (p * b - q) / b
            if full_kelly <= 0:
                continue

            frac_kelly = full_kelly * kelly_fraction
            n_contracts = min(max_contracts, max(1, int(frac_kelly * bankroll / price)))

            cost = n_contracts * price
            if direction == "YES":
                won = int(outcome == 1)
            else:
                won = int(outcome == 0)

            if won:
                payout = n_contracts * (1.0 - FEE_RATE)
                pnl = payout - cost
            else:
                pnl = -cost

            day_pnl += pnl
            bankroll += pnl

            trades.append({
                "date": date,
                "direction": direction,
                "ev": float(best_ev),
                "won": won,
                "pnl": float(pnl),
                "bucket": row.get("bucket", ""),
            })

        bankroll_history.append({"date": date, "bankroll": bankroll, "daily_pnl": day_pnl})

    total_pnl = bankroll - INITIAL_BANKROLL
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / max(n_trades, 1)

    # Risk metrics
    if bankroll_history:
        daily_pnls = [bh["daily_pnl"] for bh in bankroll_history]
        daily_returns = [dp / INITIAL_BANKROLL for dp in daily_pnls]
        if len(daily_returns) > 1:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        bankrolls = [bh["bankroll"] for bh in bankroll_history]
        cummax = np.maximum.accumulate(bankrolls)
        drawdown = np.array(bankrolls) - cummax
        max_dd = float(np.min(drawdown))
        max_dd_pct = float(np.min(drawdown / np.maximum(cummax, 1e-10)) * 100)
    else:
        sharpe = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

    # Seasonal
    seasonal = {}
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["season"] = trades_df["date"].dt.month.map(SEASON_MAP)
        for season in SEASON_ORDER:
            s_df = trades_df[trades_df["season"] == season]
            seasonal[season] = {
                "n_trades": len(s_df),
                "pnl": float(s_df["pnl"].sum()) if len(s_df) > 0 else 0.0,
                "win_rate": float(s_df["won"].mean()) if len(s_df) > 0 else 0.0,
            }

    return {
        "model": model_prob_col,
        "total_pnl": float(total_pnl),
        "return_pct": float(total_pnl / INITIAL_BANKROLL * 100),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "n_trading_days": len(dates),
        "seasonal": seasonal,
        "ev_threshold": ev_threshold,
        "kelly_fraction": kelly_fraction,
        "max_contracts": max_contracts,
    }


# ===========================================================================
# Multi-Model Sweep
# ===========================================================================

def sweep_models_and_params(city: str) -> dict:
    """Sweep all U-series models and trading parameters for one city."""
    df = load_unified_predictions(city)
    available_cols = [c for c in MODEL_PROB_COLS if c in df.columns]
    logger.info("  Available model columns: %s", available_cols)

    all_results = []

    for model_col in available_cols:
        for ev_thresh in EV_THRESHOLDS:
            for kelly_frac in KELLY_FRACTIONS:
                result = run_contract_backtest(
                    df, model_col,
                    ev_threshold=ev_thresh,
                    kelly_fraction=kelly_frac,
                    max_contracts=10,
                    oos_only=True,
                )
                all_results.append(result)

    # Also test IS+OOS (full period) for the best model
    if available_cols:
        for model_col in available_cols:
            result = run_contract_backtest(
                df, model_col,
                ev_threshold=0.03,
                kelly_fraction=0.15,
                max_contracts=10,
                oos_only=False,
            )
            result["period"] = "IS+OOS"
            all_results.append(result)

    return {
        "city": city.upper(),
        "n_models": len(available_cols),
        "n_configs": len(all_results),
        "available_models": available_cols,
        "results": all_results,
    }


def find_best_per_city(sweep_data: dict) -> dict:
    """Find the best model and strategy for a city."""
    results = sweep_data["results"]
    # Filter to OOS only (exclude IS+OOS entries)
    oos_results = [r for r in results if r.get("period") != "IS+OOS" and not np.isnan(r.get("total_pnl", float("nan")))]

    if not oos_results:
        return {}

    profitable = [r for r in oos_results if r["total_pnl"] > 0]
    candidates = profitable if profitable else oos_results

    best_sharpe = max(candidates, key=lambda x: x["sharpe_ratio"])
    best_pnl = max(oos_results, key=lambda x: x["total_pnl"])

    # Conservative: profitable with drawdown < 8%
    conservative = [r for r in oos_results
                   if r["total_pnl"] > 0 and abs(r["max_drawdown_pct"]) < 8]
    best_conservative = max(conservative, key=lambda x: x["sharpe_ratio"]) if conservative else best_sharpe

    # Per-model best
    model_best = {}
    for model_col in sweep_data["available_models"]:
        model_results = [r for r in oos_results if r["model"] == model_col]
        if model_results:
            best = max(model_results, key=lambda x: x["sharpe_ratio"])
            model_best[model_col] = {
                "ev_threshold": best["ev_threshold"],
                "kelly_fraction": best["kelly_fraction"],
                "sharpe": best["sharpe_ratio"],
                "pnl": best["total_pnl"],
                "return_pct": best["return_pct"],
                "win_rate": best["win_rate"],
                "n_trades": best["n_trades"],
                "max_drawdown_pct": best["max_drawdown_pct"],
            }

    return {
        "city": sweep_data["city"],
        "n_profitable": len(profitable),
        "n_total": len(oos_results),
        "best_overall": {
            "model": best_sharpe["model"],
            "ev_threshold": best_sharpe["ev_threshold"],
            "kelly_fraction": best_sharpe["kelly_fraction"],
            "sharpe": best_sharpe["sharpe_ratio"],
            "pnl": best_sharpe["total_pnl"],
            "return_pct": best_sharpe["return_pct"],
            "win_rate": best_sharpe["win_rate"],
            "n_trades": best_sharpe["n_trades"],
            "max_drawdown_pct": best_sharpe["max_drawdown_pct"],
            "seasonal": best_sharpe.get("seasonal", {}),
        },
        "best_pnl": {
            "model": best_pnl["model"],
            "ev_threshold": best_pnl["ev_threshold"],
            "kelly_fraction": best_pnl["kelly_fraction"],
            "pnl": best_pnl["total_pnl"],
            "return_pct": best_pnl["return_pct"],
            "sharpe": best_pnl["sharpe_ratio"],
        },
        "best_conservative": {
            "model": best_conservative["model"],
            "ev_threshold": best_conservative["ev_threshold"],
            "kelly_fraction": best_conservative["kelly_fraction"],
            "sharpe": best_conservative["sharpe_ratio"],
            "pnl": best_conservative["total_pnl"],
            "return_pct": best_conservative["return_pct"],
            "max_drawdown_pct": best_conservative["max_drawdown_pct"],
        },
        "per_model_best": model_best,
    }


# ===========================================================================
# Visualization
# ===========================================================================

def plot_model_comparison(all_city_results: dict, output_dir: str) -> None:
    """Plot per-model best Sharpe and P&L for each city."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (city, info) in enumerate(all_city_results.items()):
        ax = axes[idx]
        model_best = info.get("per_model_best", {})
        if not model_best:
            continue

        models = list(model_best.keys())
        sharpes = [model_best[m]["sharpe"] for m in models]
        pnls = [model_best[m]["pnl"] for m in models]

        # Clean model names
        clean_names = [m.replace("_prob", "").replace("model", "base") for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width / 2, sharpes, width, label="Sharpe",
                       color=["#2ca02c" if s > 0 else "#d62728" for s in sharpes],
                       edgecolor="black", alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, pnls, width, label="P&L ($)",
                        color=["#1f77b4" if p > 0 else "#ff7f0e" for p in pnls],
                        edgecolor="black", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sharpe Ratio")
        ax2.set_ylabel("Total P&L ($)")
        ax.set_title(f"{city}: Per-Model Best Trading Results (OOS)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Unified Model Trading Comparison: CHI vs PHL", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(output_dir, "unified_model_trading_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", save_path)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the unified trading backtest across all cities and models."""
    output_dir = str(PROJECT_ROOT / "results" / "cross_city_comparison")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Unified Trading Backtest (Contract-Level Probabilities)")
    logger.info("  Models: %s", MODEL_PROB_COLS)
    logger.info("  EV thresholds: %s", EV_THRESHOLDS)
    logger.info("  Kelly fractions: %s", KELLY_FRACTIONS)
    logger.info("=" * 70)

    all_city_results = {}

    for city in ["chi", "phl"]:
        logger.info("Processing %s ...", city.upper())
        sweep = sweep_models_and_params(city)
        best = find_best_per_city(sweep)
        all_city_results[city.upper()] = best

        # Save detailed results
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "seasonal"}
            for r in sweep["results"]
        ])
        csv_path = os.path.join(output_dir, f"{city}_unified_backtest_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info("  Saved %s results CSV: %s (%d rows)", city.upper(), csv_path, len(results_df))

    # Save summary
    summary_path = os.path.join(output_dir, "unified_trading_best_strategies.json")
    with open(summary_path, "w") as f:
        json.dump(all_city_results, f, indent=2, default=str)
    logger.info("Saved unified best strategies to %s", summary_path)

    # Visualize
    plot_model_comparison(all_city_results, output_dir)

    # Print summary
    logger.info("=" * 70)
    logger.info("Unified Trading Backtest Summary")
    logger.info("=" * 70)

    for city, info in all_city_results.items():
        if not info:
            logger.info("  %s: No results", city)
            continue

        logger.info("")
        logger.info("  %s:", city)
        logger.info("    Profitable configs: %d / %d", info.get("n_profitable", 0), info.get("n_total", 0))

        best = info.get("best_overall", {})
        if best:
            logger.info("    BEST OVERALL: model=%s, EV=%.2f, Kelly=%.2f",
                        best.get("model", "?"), best.get("ev_threshold", 0), best.get("kelly_fraction", 0))
            logger.info("      Sharpe=%.2f, P&L=$%.1f (%.1f%%), Win=%.1f%%, DD=%.1f%%",
                        best.get("sharpe", 0), best.get("pnl", 0), best.get("return_pct", 0),
                        best.get("win_rate", 0) * 100, abs(best.get("max_drawdown_pct", 0)))

            seasonal = best.get("seasonal", {})
            if seasonal:
                logger.info("    Seasonal breakdown:")
                for season in SEASON_ORDER:
                    s = seasonal.get(season, {})
                    logger.info("      %s: %d trades, P&L=$%.1f, Win=%.1f%%",
                                season, s.get("n_trades", 0),
                                s.get("pnl", 0), s.get("win_rate", 0) * 100)

        # Per-model leaderboard
        model_best = info.get("per_model_best", {})
        if model_best:
            logger.info("    Per-model leaderboard:")
            sorted_models = sorted(model_best.items(), key=lambda x: x[1]["sharpe"], reverse=True)
            for model_name, m in sorted_models:
                logger.info("      %-20s Sharpe=%+.2f  P&L=$%+.1f  Win=%.0f%%  DD=%.1f%%",
                            model_name, m["sharpe"], m["pnl"],
                            m["win_rate"] * 100, abs(m["max_drawdown_pct"]))


if __name__ == "__main__":
    main()
