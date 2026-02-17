#!/usr/bin/env python3
"""
Multi-City Trading Strategy Sweep.

Systematically tests different trading strategy parameters across CHI and PHL
to find the optimal configuration for each city. Parameters swept:

  - EV threshold:    [0.01, 0.02, 0.03, 0.04, 0.05]
  - Kelly fraction:  [0.10, 0.15, 0.20, 0.25, 0.35]
  - Max contracts:   [5, 10, 15]

Uses the same backtest engine as the city-specific backtest scripts.
Reports the best strategy per city based on Sharpe ratio, P&L, and
risk-adjusted return.

Usage:
    python scripts/run_trading_strategy_sweep.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs

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
PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4
FEE_RATE = 0.07
INITIAL_BANKROLL = 1000.0

# Sweep grid
EV_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
KELLY_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.35]
MAX_CONTRACTS_LIST = [5, 10, 15]

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ===========================================================================
# Market Data Generation
# ===========================================================================

def generate_market_data(
    dates: np.ndarray,
    actual_tmax: np.ndarray,
    model_mu: np.ndarray,
    model_sigma: np.ndarray,
    bucket_edges: list,
    bucket_labels: list,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate simulated market data for backtesting."""
    rng = np.random.RandomState(seed)
    n_days = len(dates)
    all_rows = []

    for d in range(n_days):
        date = dates[d]
        tmax = actual_tmax[d]
        mu = model_mu[d]
        sig = max(model_sigma[d], 1e-6)

        market_mu = tmax + rng.normal(0, 2.5)
        market_sigma = sig * 1.15

        for b, ((lo, hi), label) in enumerate(zip(bucket_edges, bucket_labels)):
            market_sigma_safe = max(market_sigma, 1e-6)
            if lo <= -900:
                true_prob = norm.cdf(hi, loc=market_mu, scale=market_sigma_safe)
            elif hi >= 900:
                true_prob = 1.0 - norm.cdf(lo, loc=market_mu, scale=market_sigma_safe)
            else:
                true_prob = (norm.cdf(hi, loc=market_mu, scale=market_sigma_safe)
                            - norm.cdf(lo, loc=market_mu, scale=market_sigma_safe))

            noise = rng.normal(0, 0.06)
            market_prob = float(np.clip(true_prob + noise, 0.01, 0.99))

            if lo <= -900:
                model_prob = norm.cdf(hi, loc=mu, scale=sig)
            elif hi >= 900:
                model_prob = 1.0 - norm.cdf(lo, loc=mu, scale=sig)
            else:
                model_prob = (norm.cdf(hi, loc=mu, scale=sig) - norm.cdf(lo, loc=mu, scale=sig))
            model_prob = float(np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

            spread = 0.02 + (0.10 - 0.02) * (1.0 - 2.0 * min(market_prob, 1.0 - market_prob))
            bid_price = float(np.clip(market_prob - spread / 2, 0.01, 0.99))
            ask_price = float(np.clip(market_prob + spread / 2, 0.01, 0.99))

            if np.isnan(tmax):
                actual_outcome = np.nan
            elif lo <= -900:
                actual_outcome = 1 if tmax < hi else 0
            elif hi >= 900:
                actual_outcome = 1 if tmax >= lo else 0
            else:
                actual_outcome = 1 if lo <= tmax < hi else 0

            all_rows.append({
                "date": date,
                "bucket_label": label,
                "bucket_idx": b,
                "market_prob": market_prob,
                "model_prob": model_prob,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "actual_outcome": actual_outcome,
                "actual_tmax": tmax,
            })

    return pd.DataFrame(all_rows)


# ===========================================================================
# Backtest Engine
# ===========================================================================

def run_backtest(
    market_df: pd.DataFrame,
    ev_threshold: float,
    kelly_fraction: float,
    max_contracts: int,
) -> dict:
    """Run a single EV-gated backtest with given parameters."""
    trades = []
    bankroll = INITIAL_BANKROLL
    bankroll_history = []

    dates = np.sort(market_df["date"].unique())

    for date in dates:
        day_df = market_df[market_df["date"] == date]
        day_pnl = 0.0

        for _, row in day_df.iterrows():
            model_prob = row["model_prob"]
            ask = row["ask_price"]
            bid = row["bid_price"]
            outcome = row["actual_outcome"]

            if np.isnan(outcome):
                continue

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
            })

        bankroll_history.append({"date": date, "bankroll": bankroll, "daily_pnl": day_pnl})

    total_pnl = bankroll - INITIAL_BANKROLL
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / max(n_trades, 1)

    # Sharpe
    if bankroll_history:
        daily_pnls = [bh["daily_pnl"] for bh in bankroll_history]
        daily_returns = [dp / INITIAL_BANKROLL for dp in daily_pnls]
        if len(daily_returns) > 1:
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # Drawdown
        bankrolls = [bh["bankroll"] for bh in bankroll_history]
        cummax = np.maximum.accumulate(bankrolls)
        drawdown = np.array(bankrolls) - cummax
        max_dd = float(np.min(drawdown))
        max_dd_pct = float(np.min(drawdown / np.maximum(cummax, 1e-10)) * 100)
    else:
        sharpe = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

    # Seasonal P&L
    seasonal_pnl = {}
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["season"] = trades_df["date"].dt.month.map(SEASON_MAP)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            s_df = trades_df[trades_df["season"] == season]
            seasonal_pnl[season] = {
                "n_trades": len(s_df),
                "pnl": float(s_df["pnl"].sum()) if len(s_df) > 0 else 0.0,
                "win_rate": float(s_df["won"].mean()) if len(s_df) > 0 else 0.0,
            }

    return {
        "total_pnl": float(total_pnl),
        "return_pct": float(total_pnl / INITIAL_BANKROLL * 100),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd_pct),
        "seasonal": seasonal_pnl,
    }


# ===========================================================================
# Strategy Sweep
# ===========================================================================

def sweep_city(city_code: str) -> dict:
    """Run the full strategy sweep for one city."""
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    # Load predictions
    synthesis_path = os.path.join(cfg.results_dir, "synthesis", "synthesis_predictions.csv")
    if os.path.isfile(synthesis_path):
        preds_df = pd.read_csv(synthesis_path, parse_dates=["date"])
    else:
        base_path = os.path.join(cfg.results_dir, "base_predictions.csv")
        preds_df = pd.read_csv(base_path, parse_dates=["date"])

    # OOS portion
    n = len(preds_df)
    oos_start = int(n * 0.70)
    oos_df = preds_df.iloc[oos_start:].reset_index(drop=True)

    logger.info("  %s: %d OOS days, %d buckets",
                city_code.upper(), len(oos_df), len(cfg.bucket_edges))

    # Generate market data once (same seed for all strategies)
    market_df = generate_market_data(
        dates=pd.to_datetime(oos_df["date"]).values,
        actual_tmax=oos_df["actual_tmax"].values,
        model_mu=oos_df["mu"].values,
        model_sigma=oos_df["sigma"].values,
        bucket_edges=cfg.bucket_edges,
        bucket_labels=cfg.bucket_labels,
        seed=42,
    )

    # Sweep all parameter combinations
    results = []
    configs = list(itertools.product(EV_THRESHOLDS, KELLY_FRACTIONS, MAX_CONTRACTS_LIST))

    logger.info("  Running %d strategy configurations ...", len(configs))

    for ev_thresh, kelly_frac, max_contracts in configs:
        bt = run_backtest(market_df, ev_thresh, kelly_frac, max_contracts)
        bt["ev_threshold"] = ev_thresh
        bt["kelly_fraction"] = kelly_frac
        bt["max_contracts"] = max_contracts
        results.append(bt)

    return {
        "city": city_code.upper(),
        "ticker": cfg.kalshi_ticker,
        "n_oos_days": len(oos_df),
        "n_buckets": len(cfg.bucket_edges),
        "n_configs_tested": len(configs),
        "results": results,
    }


def find_best_strategies(sweep_results: list[dict]) -> dict:
    """Identify the best strategy for each city using multiple criteria."""
    best = {}

    for city_data in sweep_results:
        city = city_data["city"]
        results = city_data["results"]

        if not results:
            continue

        # Filter to profitable strategies
        profitable = [r for r in results if r["total_pnl"] > 0]

        # Best by Sharpe (among profitable, or overall if none profitable)
        candidates = profitable if profitable else results
        best_sharpe = max(candidates, key=lambda x: x["sharpe_ratio"])

        # Best by P&L
        best_pnl = max(results, key=lambda x: x["total_pnl"])

        # Best risk-adjusted: highest Sharpe with drawdown < 10%
        conservative = [r for r in results
                       if r["total_pnl"] > 0 and abs(r["max_drawdown_pct"]) < 10]
        if conservative:
            best_conservative = max(conservative, key=lambda x: x["sharpe_ratio"])
        else:
            best_conservative = best_sharpe

        best[city] = {
            "ticker": city_data["ticker"],
            "n_oos_days": city_data["n_oos_days"],
            "n_configs_tested": city_data["n_configs_tested"],
            "best_sharpe_strategy": {
                "ev_threshold": best_sharpe["ev_threshold"],
                "kelly_fraction": best_sharpe["kelly_fraction"],
                "max_contracts": best_sharpe["max_contracts"],
                "sharpe": best_sharpe["sharpe_ratio"],
                "pnl": best_sharpe["total_pnl"],
                "return_pct": best_sharpe["return_pct"],
                "win_rate": best_sharpe["win_rate"],
                "n_trades": best_sharpe["n_trades"],
                "max_drawdown_pct": best_sharpe["max_drawdown_pct"],
            },
            "best_pnl_strategy": {
                "ev_threshold": best_pnl["ev_threshold"],
                "kelly_fraction": best_pnl["kelly_fraction"],
                "max_contracts": best_pnl["max_contracts"],
                "sharpe": best_pnl["sharpe_ratio"],
                "pnl": best_pnl["total_pnl"],
                "return_pct": best_pnl["return_pct"],
                "win_rate": best_pnl["win_rate"],
                "n_trades": best_pnl["n_trades"],
                "max_drawdown_pct": best_pnl["max_drawdown_pct"],
            },
            "best_conservative_strategy": {
                "ev_threshold": best_conservative["ev_threshold"],
                "kelly_fraction": best_conservative["kelly_fraction"],
                "max_contracts": best_conservative["max_contracts"],
                "sharpe": best_conservative["sharpe_ratio"],
                "pnl": best_conservative["total_pnl"],
                "return_pct": best_conservative["return_pct"],
                "win_rate": best_conservative["win_rate"],
                "n_trades": best_conservative["n_trades"],
                "max_drawdown_pct": best_conservative["max_drawdown_pct"],
            },
            "n_profitable_configs": len(profitable),
            "n_total_configs": len(results),
        }

    return best


# ===========================================================================
# Visualization
# ===========================================================================

def plot_strategy_heatmaps(sweep_results: list[dict], output_dir: str) -> None:
    """Plot EV threshold x Kelly fraction heatmaps for each city."""
    for city_data in sweep_results:
        city = city_data["city"]
        results = city_data["results"]

        # Build heatmap for max_contracts=10 (default)
        df = pd.DataFrame(results)
        df_mc10 = df[df["max_contracts"] == 10]

        if len(df_mc10) == 0:
            continue

        # P&L heatmap
        pivot_pnl = df_mc10.pivot_table(
            values="total_pnl", index="kelly_fraction", columns="ev_threshold",
        )
        pivot_sharpe = df_mc10.pivot_table(
            values="sharpe_ratio", index="kelly_fraction", columns="ev_threshold",
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # P&L heatmap
        ax = axes[0]
        im = ax.imshow(pivot_pnl.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot_pnl.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot_pnl.columns])
        ax.set_yticks(range(len(pivot_pnl.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot_pnl.index])
        ax.set_xlabel("EV Threshold")
        ax.set_ylabel("Kelly Fraction")
        ax.set_title(f"{city}: Total P&L ($) (max_contracts=10)")
        for i in range(len(pivot_pnl.index)):
            for j in range(len(pivot_pnl.columns)):
                val = pivot_pnl.values[i, j]
                ax.text(j, i, f"${val:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if abs(val) > 50 else "black")
        plt.colorbar(im, ax=ax, label="P&L ($)")

        # Sharpe heatmap
        ax = axes[1]
        im = ax.imshow(pivot_sharpe.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot_sharpe.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot_sharpe.columns])
        ax.set_yticks(range(len(pivot_sharpe.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot_sharpe.index])
        ax.set_xlabel("EV Threshold")
        ax.set_ylabel("Kelly Fraction")
        ax.set_title(f"{city}: Sharpe Ratio (max_contracts=10)")
        for i in range(len(pivot_sharpe.index)):
            for j in range(len(pivot_sharpe.columns)):
                val = pivot_sharpe.values[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                       color="white" if abs(val) > 1.5 else "black")
        plt.colorbar(im, ax=ax, label="Sharpe")

        fig.suptitle(f"{city} ({city_data['ticker']}): Trading Strategy Sweep",
                    fontsize=14, fontweight="bold")
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"{city.lower()}_strategy_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s strategy heatmap to %s", city, save_path)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the full multi-city trading strategy sweep."""
    output_dir = str(PROJECT_ROOT / "results" / "cross_city_comparison")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Multi-City Trading Strategy Sweep")
    logger.info("  EV thresholds:   %s", EV_THRESHOLDS)
    logger.info("  Kelly fractions: %s", KELLY_FRACTIONS)
    logger.info("  Max contracts:   %s", MAX_CONTRACTS_LIST)
    logger.info("  Total configs:   %d", len(EV_THRESHOLDS) * len(KELLY_FRACTIONS) * len(MAX_CONTRACTS_LIST))
    logger.info("=" * 70)

    all_results = []

    for city in ["chi", "phl"]:
        logger.info("Sweeping %s ...", city.upper())
        result = sweep_city(city)
        all_results.append(result)

        # Save per-city sweep results
        city_sweep_path = os.path.join(output_dir, f"{city}_strategy_sweep.json")
        # Only save summary, not all individual results (too large)
        city_summary = {
            "city": result["city"],
            "ticker": result["ticker"],
            "n_oos_days": result["n_oos_days"],
            "n_buckets": result["n_buckets"],
            "n_configs_tested": result["n_configs_tested"],
        }
        with open(city_sweep_path, "w") as f:
            json.dump(city_summary, f, indent=2, default=str)

    # Find best strategies
    logger.info("Finding best strategies ...")
    best = find_best_strategies(all_results)

    best_path = os.path.join(output_dir, "best_trading_strategies.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2, default=str)
    logger.info("Saved best strategies to %s", best_path)

    # Save full sweep CSV for each city
    for city_data in all_results:
        city = city_data["city"].lower()
        df = pd.DataFrame(city_data["results"])
        # Drop seasonal dict column for CSV
        if "seasonal" in df.columns:
            df = df.drop(columns=["seasonal"])
        csv_path = os.path.join(output_dir, f"{city}_strategy_sweep_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s sweep CSV: %s (%d rows)", city.upper(), csv_path, len(df))

    # Visualize
    logger.info("Generating strategy heatmaps ...")
    plot_strategy_heatmaps(all_results, output_dir)

    # Print summary
    logger.info("=" * 70)
    logger.info("Trading Strategy Sweep Complete")
    logger.info("=" * 70)

    for city, info in best.items():
        logger.info("")
        logger.info("  %s (%s):", city, info["ticker"])
        logger.info("    Profitable configs: %d / %d",
                    info["n_profitable_configs"], info["n_total_configs"])

        bs = info["best_sharpe_strategy"]
        logger.info("    Best Sharpe strategy: EV=%.2f, Kelly=%.2f, MaxC=%d",
                    bs["ev_threshold"], bs["kelly_fraction"], bs["max_contracts"])
        logger.info("      Sharpe=%.2f, P&L=$%.1f (%.1f%%), Win=%.1f%%, DD=%.1f%%",
                    bs["sharpe"], bs["pnl"], bs["return_pct"],
                    bs["win_rate"] * 100, abs(bs["max_drawdown_pct"]))

        bp = info["best_pnl_strategy"]
        logger.info("    Best P&L strategy:   EV=%.2f, Kelly=%.2f, MaxC=%d",
                    bp["ev_threshold"], bp["kelly_fraction"], bp["max_contracts"])
        logger.info("      Sharpe=%.2f, P&L=$%.1f (%.1f%%), Win=%.1f%%, DD=%.1f%%",
                    bp["sharpe"], bp["pnl"], bp["return_pct"],
                    bp["win_rate"] * 100, abs(bp["max_drawdown_pct"]))

        bc = info["best_conservative_strategy"]
        logger.info("    Best conservative:   EV=%.2f, Kelly=%.2f, MaxC=%d",
                    bc["ev_threshold"], bc["kelly_fraction"], bc["max_contracts"])
        logger.info("      Sharpe=%.2f, P&L=$%.1f (%.1f%%), Win=%.1f%%, DD=%.1f%%",
                    bc["sharpe"], bc["pnl"], bc["return_pct"],
                    bc["win_rate"] * 100, abs(bc["max_drawdown_pct"]))


if __name__ == "__main__":
    main()
