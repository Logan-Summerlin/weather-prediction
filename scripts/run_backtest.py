#!/usr/bin/env python3
"""
Unified Multi-City Kalshi Market Backtest.

Simulates Kalshi market conditions and runs an EV-gated trading backtest
using calibrated model predictions for any supported city.  Reports P&L,
Brier score, and drawdown metrics.

The backtest uses:
  - EV threshold:    0.02 (only trade when expected value > 2 cents)
  - Fee rate:        0.07 (Kalshi standard 7%)
  - Kelly fraction:  0.25 (quarter Kelly for safety)
  - Position limits: max 10 contracts per bucket

Results are saved to results/<city>/backtest/.

Usage:
    python scripts/run_backtest.py --city chi
    python scripts/run_backtest.py --city phl
    python scripts/run_backtest.py --city atl
    python scripts/run_backtest.py --city aus
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# Use non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs
from src.trading import compute_drawdown_metrics

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

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Default backtest parameters
DEFAULT_EV_THRESHOLD = 0.02
DEFAULT_FEE_RATE = 0.07
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MAX_CONTRACTS = 10
DEFAULT_INITIAL_BANKROLL = 1000.0


# ===========================================================================
# Market Data Simulation
# ===========================================================================

def generate_market_data(
    dates: np.ndarray,
    actual_tmax: np.ndarray,
    model_mu: np.ndarray,
    model_sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
    bucket_labels: list[str],
    city_code: str,
    market_noise_std: float = 0.06,
    min_spread: float = 0.02,
    max_spread: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Kalshi market data for backtesting.

    Generates realistic market prices based on the actual temperature
    outcome with noise, bid-ask spreads, and volume.  The market is
    slightly noisier than the model, providing opportunities for the
    model to find edge.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates for each trading day.
    actual_tmax : np.ndarray
        Observed maximum temperatures (deg F), shape (n_days,).
    model_mu : np.ndarray
        Model predicted means, shape (n_days,).
    model_sigma : np.ndarray
        Model predicted standard deviations, shape (n_days,).
    bucket_edges : list of (low, high) tuples
        Kalshi contract bucket boundaries.
    bucket_labels : list[str]
        Human-readable bucket labels.
    city_code : str
        Short city code (e.g. "chi", "phl") for log messages.
    market_noise_std : float
        Noise added to true probabilities to simulate market inefficiency.
    min_spread : float
        Minimum bid-ask spread.
    max_spread : float
        Maximum bid-ask spread for illiquid buckets.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Market dataset with one row per (day, bucket), containing columns:
        date, bucket_label, bucket_idx, bucket_low, bucket_high,
        market_prob, model_prob, bid_price, ask_price, volume,
        actual_outcome, actual_tmax.
    """
    rng = np.random.RandomState(seed)
    n_days = len(dates)
    n_buckets = len(bucket_edges)
    all_rows = []

    for d in range(n_days):
        date = dates[d]
        tmax = actual_tmax[d]
        mu = model_mu[d]
        sig = max(model_sigma[d], 1e-6)

        # Market's view: slightly noisy Gaussian centered near actual
        market_mu = tmax + rng.normal(0, 2.5)
        market_sigma = sig * 1.15  # Market slightly wider than model

        for b, ((lo, hi), label) in enumerate(zip(bucket_edges, bucket_labels)):
            # True market probability (from market's Gaussian)
            market_sigma_safe = max(market_sigma, 1e-6)
            if lo <= -900:
                true_prob = norm.cdf(hi, loc=market_mu, scale=market_sigma_safe)
            elif hi >= 900:
                true_prob = 1.0 - norm.cdf(lo, loc=market_mu, scale=market_sigma_safe)
            else:
                true_prob = (
                    norm.cdf(hi, loc=market_mu, scale=market_sigma_safe)
                    - norm.cdf(lo, loc=market_mu, scale=market_sigma_safe)
                )

            # Market price = true prob + noise
            noise = rng.normal(0, market_noise_std)
            market_prob = float(np.clip(true_prob + noise, 0.01, 0.99))

            # Model probability
            if lo <= -900:
                model_prob = norm.cdf(hi, loc=mu, scale=sig)
            elif hi >= 900:
                model_prob = 1.0 - norm.cdf(lo, loc=mu, scale=sig)
            else:
                model_prob = (
                    norm.cdf(hi, loc=mu, scale=sig)
                    - norm.cdf(lo, loc=mu, scale=sig)
                )
            model_prob = float(np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))

            # Bid-ask spread: tighter near-the-money
            spread = min_spread + (max_spread - min_spread) * (
                1.0 - 2.0 * min(market_prob, 1.0 - market_prob)
            )
            bid_price = float(np.clip(market_prob - spread / 2, 0.01, 0.99))
            ask_price = float(np.clip(market_prob + spread / 2, 0.01, 0.99))

            # Volume: higher for near-the-money buckets
            vol_center = max(true_prob * 500, 5)
            volume = max(1, int(rng.poisson(lam=vol_center)))

            # Actual outcome: did TMAX fall in this bucket?
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
                "bucket_low": lo if lo > -900 else None,
                "bucket_high": hi if hi < 900 else None,
                "market_prob": market_prob,
                "model_prob": model_prob,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "volume": volume,
                "actual_outcome": actual_outcome,
                "actual_tmax": tmax,
            })

    df = pd.DataFrame(all_rows)
    logger.info(
        "Generated %s market data: %d rows across %d days, %d buckets",
        city_code.upper(), len(df), n_days, n_buckets,
    )
    return df


# ===========================================================================
# EV Computation
# ===========================================================================

def compute_ev_per_bucket(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> dict[str, np.ndarray]:
    """Compute expected value for YES and NO sides of each bucket.

    EV_yes = model_prob * (1 - fee) - market_price
    EV_no  = (1 - model_prob) * (1 - fee) - (1 - market_price)

    Parameters
    ----------
    model_probs : np.ndarray
        Model probabilities, shape (n,).
    market_probs : np.ndarray
        Market prices, shape (n,).
    fee_rate : float
        Fee rate on winnings (default 0.07).

    Returns
    -------
    dict
        Keys: ev_yes, ev_no, best_ev, best_direction (arrays).
    """
    ev_yes = model_probs * (1.0 - fee_rate) - market_probs
    ev_no = (1.0 - model_probs) * (1.0 - fee_rate) - (1.0 - market_probs)

    best_ev = np.maximum(ev_yes, ev_no)
    best_direction = np.where(ev_yes >= ev_no, "YES", "NO")

    return {
        "ev_yes": ev_yes,
        "ev_no": ev_no,
        "best_ev": best_ev,
        "best_direction": best_direction,
    }


# ===========================================================================
# Core Backtest Loop
# ===========================================================================

def run_ev_gated_backtest(
    market_df: pd.DataFrame,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    fee_rate: float = DEFAULT_FEE_RATE,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
) -> dict:
    """Run an EV-gated trading backtest on simulated market data.

    For each day and bucket:
      1. Compute EV for YES and NO sides.
      2. If best EV > threshold, compute Kelly fraction and size position.
      3. Execute trade and record P&L at settlement.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market dataset from generate_market_data().
    ev_threshold : float
        Minimum EV to trigger a trade (default 0.02).
    fee_rate : float
        Kalshi fee rate (default 0.07).
    kelly_fraction : float
        Fraction of Kelly to use (default 0.25).
    max_contracts : int
        Maximum contracts per bucket per day (default 10).
    initial_bankroll : float
        Starting bankroll in dollars (default 1000).

    Returns
    -------
    dict
        Keys: trades (list[dict]), daily_pnl (pd.Series),
        bankroll_series (pd.Series), total_pnl, win_rate, n_trades.
    """
    trades = []
    bankroll = initial_bankroll
    bankroll_history = []
    busted = False
    bust_date = None

    # Process day by day
    dates = market_df["date"].unique()
    dates_sorted = np.sort(dates)

    for date in dates_sorted:
        # Halt at bankruptcy — continuing to "trade" with no capital
        # silently inflates losses and corrupts drawdown statistics.
        if bankroll <= 0:
            busted = True
            bust_date = str(date)
            logger.warning("Bankroll exhausted on %s — halting backtest", date)
            break

        day_df = market_df[market_df["date"] == date].copy()
        day_pnl = 0.0

        for _, row in day_df.iterrows():
            model_prob = row["model_prob"]
            market_prob = row["market_prob"]
            ask = row["ask_price"]
            bid = row["bid_price"]
            outcome = row["actual_outcome"]
            bucket_label = row["bucket_label"]

            if np.isnan(outcome):
                continue

            # Compute EV for both sides
            ev_yes = model_prob * (1.0 - fee_rate) - ask
            ev_no = (1.0 - model_prob) * (1.0 - fee_rate) - (1.0 - bid)

            # Determine best direction
            if ev_yes > ev_no:
                best_ev = ev_yes
                direction = "YES"
                entry_price = ask
            else:
                best_ev = ev_no
                direction = "NO"
                entry_price = 1.0 - bid

            # Skip if EV below threshold
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
            net_payout = (1.0 - fee_rate) - price
            if net_payout <= 0:
                continue

            b = net_payout / price
            if b <= 0:
                continue

            q = 1.0 - p
            full_kelly = (p * b - q) / b
            if full_kelly <= 0:
                continue

            # Apply fractional Kelly
            frac_kelly = full_kelly * kelly_fraction

            # Size in contracts (1 contract = $1 face value).
            # Stake can never exceed the current bankroll.
            affordable = int(max(0.0, bankroll) / price)
            n_contracts = min(
                max_contracts,
                max(1, int(frac_kelly * bankroll / price)),
                affordable,
            )
            if n_contracts < 1:
                continue

            # Compute trade P&L
            cost = n_contracts * price
            if direction == "YES":
                won = int(outcome == 1)
            else:
                won = int(outcome == 0)

            if won:
                payout = n_contracts * (1.0 - fee_rate)
                pnl = payout - cost
            else:
                pnl = -cost

            day_pnl += pnl
            bankroll += pnl

            trades.append({
                "date": date,
                "bucket_label": bucket_label,
                "direction": direction,
                "model_prob": float(model_prob),
                "market_prob": float(market_prob),
                "entry_price": float(entry_price),
                "ev": float(best_ev),
                "kelly_frac": float(frac_kelly),
                "n_contracts": n_contracts,
                "cost": float(cost),
                "outcome": int(outcome),
                "won": won,
                "pnl": float(pnl),
                "bankroll_after": float(bankroll),
            })

        bankroll_history.append({
            "date": date,
            "daily_pnl": day_pnl,
            "bankroll": bankroll,
        })

    # Build output series
    if bankroll_history:
        bh_df = pd.DataFrame(bankroll_history)
        bh_df["date"] = pd.to_datetime(bh_df["date"])
        daily_pnl = bh_df.set_index("date")["daily_pnl"]
        bankroll_series = bh_df.set_index("date")["bankroll"]
    else:
        daily_pnl = pd.Series(dtype=float)
        bankroll_series = pd.Series(dtype=float)

    total_pnl = bankroll - initial_bankroll
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["won"])
    win_rate = wins / max(n_trades, 1)

    logger.info(
        "Backtest complete: %d trades, P&L=$%.2f, Win rate=%.1f%%",
        n_trades, total_pnl, win_rate * 100,
    )

    return {
        "trades": trades,
        "daily_pnl": daily_pnl,
        "bankroll_series": bankroll_series,
        "total_pnl": float(total_pnl),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "initial_bankroll": initial_bankroll,
        "final_bankroll": float(bankroll),
        "busted": busted,
        "bust_date": bust_date,
    }


# ===========================================================================
# Performance Metrics
# ===========================================================================

def compute_backtest_metrics(
    backtest_result: dict,
    market_df: pd.DataFrame,
    bucket_edges: list[tuple[float, float]],
) -> dict:
    """Compute comprehensive backtest performance metrics.

    Includes P&L statistics, risk metrics, Brier score comparisons,
    and seasonal breakdowns.

    Parameters
    ----------
    backtest_result : dict
        Output from run_ev_gated_backtest().
    market_df : pd.DataFrame
        Full market dataset.
    bucket_edges : list of (low, high) tuples
        Bucket boundaries for Brier score computation.

    Returns
    -------
    dict
        Comprehensive metrics summary suitable for JSON serialization.
    """
    trades = backtest_result["trades"]
    daily_pnl = backtest_result["daily_pnl"]
    bankroll_series = backtest_result["bankroll_series"]

    metrics: dict = {}

    # ---- P&L metrics ----
    metrics["total_pnl"] = backtest_result["total_pnl"]
    metrics["initial_bankroll"] = backtest_result["initial_bankroll"]
    metrics["final_bankroll"] = backtest_result["final_bankroll"]
    metrics["return_pct"] = (
        backtest_result["total_pnl"] / backtest_result["initial_bankroll"] * 100
    )
    metrics["n_trades"] = backtest_result["n_trades"]
    metrics["win_rate"] = backtest_result["win_rate"]
    metrics["n_trading_days"] = len(daily_pnl)

    # Average P&L per trade
    if trades:
        trade_pnls = [t["pnl"] for t in trades]
        metrics["avg_pnl_per_trade"] = float(np.mean(trade_pnls))
        metrics["median_pnl_per_trade"] = float(np.median(trade_pnls))
        metrics["avg_ev_traded"] = float(np.mean([t["ev"] for t in trades]))
    else:
        metrics["avg_pnl_per_trade"] = 0.0
        metrics["median_pnl_per_trade"] = 0.0
        metrics["avg_ev_traded"] = 0.0

    # ---- Drawdown metrics ----
    metrics.update(compute_drawdown_metrics(
        bankroll_series, backtest_result["initial_bankroll"],
    ))
    metrics["busted"] = backtest_result.get("busted", False)
    metrics["bust_date"] = backtest_result.get("bust_date")

    # ---- Sharpe ratio (annualized, assuming 252 trading days) ----
    if len(daily_pnl) > 1:
        daily_returns = daily_pnl / backtest_result["initial_bankroll"]
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        if std_ret > 0:
            metrics["sharpe_ratio"] = float(mean_ret / std_ret * np.sqrt(252))
        else:
            metrics["sharpe_ratio"] = 0.0
        metrics["daily_pnl_mean"] = float(daily_pnl.mean())
        metrics["daily_pnl_std"] = float(daily_pnl.std())
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["daily_pnl_mean"] = 0.0
        metrics["daily_pnl_std"] = 0.0

    # ---- Brier scores: model vs market ----
    dates_all = market_df["date"].unique()
    n_buckets = len(bucket_edges)

    # Build per-day probability matrices
    model_brier_sum = 0.0
    market_brier_sum = 0.0
    n_brier_points = 0

    for date in dates_all:
        day_df = market_df[market_df["date"] == date].sort_values("bucket_idx")
        if len(day_df) != n_buckets:
            continue

        model_probs = day_df["model_prob"].values
        market_probs = day_df["market_prob"].values
        outcomes = day_df["actual_outcome"].values

        if np.isnan(outcomes).any():
            continue

        model_brier_sum += float(np.sum((model_probs - outcomes) ** 2))
        market_brier_sum += float(np.sum((market_probs - outcomes) ** 2))
        n_brier_points += n_buckets

    if n_brier_points > 0:
        metrics["model_brier"] = model_brier_sum / n_brier_points
        metrics["market_brier"] = market_brier_sum / n_brier_points
        metrics["brier_edge"] = metrics["market_brier"] - metrics["model_brier"]
    else:
        metrics["model_brier"] = float("nan")
        metrics["market_brier"] = float("nan")
        metrics["brier_edge"] = float("nan")

    # ---- Seasonal breakdown ----
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["month"] = trades_df["date"].dt.month
        trades_df["season"] = trades_df["month"].map(SEASON_MAP)

        seasonal_metrics = {}
        for season in SEASON_ORDER:
            s_df = trades_df[trades_df["season"] == season]
            if len(s_df) == 0:
                seasonal_metrics[season] = {
                    "n_trades": 0, "pnl": 0.0, "win_rate": 0.0,
                }
                continue
            seasonal_metrics[season] = {
                "n_trades": len(s_df),
                "pnl": float(s_df["pnl"].sum()),
                "win_rate": float(s_df["won"].mean()),
                "avg_ev": float(s_df["ev"].mean()),
            }

        metrics["seasonal"] = seasonal_metrics
    else:
        metrics["seasonal"] = {}

    return metrics


# ===========================================================================
# Visualization
# ===========================================================================

def plot_pnl_curve(
    bankroll_series: pd.Series,
    initial_bankroll: float,
    output_path: str,
    title: str = "Backtest: Cumulative P&L",
) -> None:
    """Plot the bankroll / cumulative P&L curve.

    Parameters
    ----------
    bankroll_series : pd.Series
        Bankroll over time (date-indexed).
    initial_bankroll : float
        Starting bankroll for reference line.
    output_path : str
        File path to save the figure.
    title : str
        Plot title.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Bankroll
    ax = axes[0]
    ax.plot(bankroll_series.index, bankroll_series.values,
            linewidth=1.5, color="#1f77b4", label="Bankroll")
    ax.axhline(initial_bankroll, color="gray", linestyle="--",
               linewidth=1, label=f"Initial (${initial_bankroll:.0f})")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    ax = axes[1]
    cummax = bankroll_series.cummax()
    drawdown = bankroll_series - cummax
    ax.fill_between(drawdown.index, 0, drawdown.values,
                    color="red", alpha=0.3, label="Drawdown")
    ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
    ax.set_ylabel("Drawdown ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved P&L curve to %s", output_path)


def plot_brier_comparison(
    model_brier: float,
    market_brier: float,
    output_path: str,
    title: str = "Model vs Market Brier Score",
) -> None:
    """Plot a bar chart comparing model and market Brier scores.

    Parameters
    ----------
    model_brier : float
        Model's overall Brier score.
    market_brier : float
        Market's overall Brier score.
    output_path : str
        File path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Model", "Market"]
    scores = [model_brier, market_brier]
    colors = ["#2ca02c", "#ff7f0e"]

    bars = ax.bar(labels, scores, color=colors, edgecolor="black", width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate edge
    edge = market_brier - model_brier
    edge_color = "#2ca02c" if edge > 0 else "#d62728"
    ax.text(
        0.5, max(scores) * 0.85,
        f"Model edge: {edge:+.4f}",
        ha="center", fontsize=13, color=edge_color, fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Brier comparison to %s", output_path)


def plot_seasonal_pnl(
    seasonal_metrics: dict,
    output_path: str,
    title: str = "Backtest: Seasonal P&L Breakdown",
) -> None:
    """Plot seasonal P&L breakdown as a bar chart.

    Parameters
    ----------
    seasonal_metrics : dict
        Mapping of season name to metrics dict with 'pnl' key.
    output_path : str
        File path to save the figure.
    title : str
        Plot title.
    """
    seasons = SEASON_ORDER
    pnls = [seasonal_metrics.get(s, {}).get("pnl", 0.0) for s in seasons]
    n_trades = [seasonal_metrics.get(s, {}).get("n_trades", 0) for s in seasons]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#d62728" if p < 0 else "#2ca02c" for p in pnls]
    bars = ax.bar(seasons, pnls, color=colors, edgecolor="black", alpha=0.8)

    for bar, pnl, nt in zip(bars, pnls, n_trades):
        y_offset = 0.5 if pnl >= 0 else -0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"${pnl:.1f}\n({nt} trades)",
            ha="center", va="bottom" if pnl >= 0 else "top",
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("P&L ($)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved seasonal P&L chart to %s", output_path)


def plot_trade_ev_distribution(
    trades: list[dict],
    output_path: str,
    title: str = "Backtest: Trade EV Distribution",
) -> None:
    """Plot the distribution of traded EVs, colored by win/loss.

    Parameters
    ----------
    trades : list[dict]
        Trade records from backtest.
    output_path : str
        File path to save the figure.
    title : str
        Plot title.
    """
    if not trades:
        logger.warning("No trades to plot EV distribution for.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    evs = np.array([t["ev"] for t in trades])
    wins = np.array([t["won"] for t in trades], dtype=bool)

    ax.hist(evs[wins], bins=30, alpha=0.6, color="#2ca02c",
            label=f"Wins ({wins.sum()})", edgecolor="white")
    ax.hist(evs[~wins], bins=30, alpha=0.6, color="#d62728",
            label=f"Losses ({(~wins).sum()})", edgecolor="white")

    ax.axvline(evs.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean EV: {evs.mean():.3f}")

    ax.set_xlabel("Expected Value at Entry")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trade EV distribution to %s", output_path)


# ===========================================================================
# Prediction Loading
# ===========================================================================

def load_calibrated_predictions(
    results_dir: str,
    city_code: str,
    city_name: str,
) -> pd.DataFrame:
    """Load calibrated synthesis predictions for any city.

    Looks for the synthesis predictions CSV produced by the city's
    synthesis/calibration script.  Falls back to base predictions
    if synthesis not found.  For Austin, also checks for unified
    (U-series) predictions.

    Parameters
    ----------
    results_dir : str
        Path to results/<city>/ directory.
    city_code : str
        Short city code (e.g. "chi", "phl", "atl", "aus").
    city_name : str
        Full city name for error messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, mu, sigma, actual_tmax.
    """
    # Austin-specific: prefer unified predictions (U-series models) if available
    if city_code == "aus":
        unified_path = os.path.join(results_dir, "unified_predictions.csv")
        unified_summary_path = os.path.join(
            results_dir, "synthesis", "unified_benchmark_summary.json")

        if os.path.isfile(unified_path) and os.path.isfile(unified_summary_path):
            logger.info("Loading unified predictions from %s", unified_path)
            with open(unified_summary_path) as f:
                summary = json.load(f)
            best_model = summary.get("best_overall_model", "")
            df = pd.read_csv(unified_path, parse_dates=["date"])
            required = {"date", "mu", "sigma", "actual_tmax", "model_name"}
            if required.issubset(df.columns) and best_model:
                best_df = df[df["model_name"] == best_model].copy()
                if len(best_df) > 0:
                    logger.info("  Using best U-series model: %s (%d predictions)",
                               best_model, len(best_df))
                    return best_df

    synthesis_path = os.path.join(results_dir, "synthesis", "synthesis_predictions.csv")

    if os.path.isfile(synthesis_path):
        logger.info("Loading calibrated predictions from %s", synthesis_path)
        df = pd.read_csv(synthesis_path, parse_dates=["date"])
        required = {"date", "mu", "sigma", "actual_tmax"}
        if required.issubset(df.columns):
            return df

    # Fallback: check for base predictions
    base_path = os.path.join(results_dir, "base_predictions.csv")
    if os.path.isfile(base_path):
        logger.info("Loading base predictions from %s", base_path)
        df = pd.read_csv(base_path, parse_dates=["date"])
        required = {"date", "mu", "sigma", "actual_tmax"}
        if required.issubset(df.columns):
            return df

    # AUDIT FIX: Synthetic data fallback removed -- must have real predictions.
    raise RuntimeError(
        f"No calibrated predictions found in {results_dir}. "
        f"Run the {city_name} synthesis/calibration script first to generate real "
        f"predictions. Synthetic data fallback has been removed to prevent "
        f"silent corruption of backtest results."
    )


# ===========================================================================
# Real Kalshi Backtest (Austin-specific, available for future cities)
# ===========================================================================

def run_real_kalshi_backtest(
    preds_df: pd.DataFrame,
    cfg,
    city_code: str,
    backtest_dir: str,
    ev_threshold: float = DEFAULT_EV_THRESHOLD,
    fee_rate: float = DEFAULT_FEE_RATE,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
) -> Optional[dict]:
    """Run backtest using real Kalshi presettlement data.

    Loads real presettlement and settlement data, merges them, computes
    model probabilities for each contract row, and runs the same EV-gated
    backtest loop used for simulated data.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Model predictions with columns: date, mu, sigma, actual_tmax.
    cfg
        City config object (from get_city_config).
    city_code : str
        Short city code (e.g. "aus").
    backtest_dir : str
        Directory to save real Kalshi backtest outputs.
    ev_threshold : float
        Minimum EV to trigger a trade.
    fee_rate : float
        Kalshi fee rate.
    kelly_fraction : float
        Fraction of Kelly to use.
    max_contracts : int
        Maximum contracts per bucket per day.
    initial_bankroll : float
        Starting bankroll in dollars.

    Returns
    -------
    dict or None
        Backtest metrics if real Kalshi data exists, else None.
    """
    pre_path = Path(PROJECT_ROOT) / "data" / f"kalshi_presettlement_{city_code}.csv"
    settled_path = Path(PROJECT_ROOT) / "data" / f"real_kalshi_{city_code}_all.csv"

    if not pre_path.exists() or not settled_path.exists():
        logger.warning("Real Kalshi data not found, skipping real backtest")
        return None

    logger.info("Loading real Kalshi presettlement data from %s", pre_path)
    pre = pd.read_csv(pre_path)
    logger.info("Loading real Kalshi settlement data from %s", settled_path)
    settled = pd.read_csv(settled_path)

    # Normalize dates
    pre["date"] = pd.to_datetime(pre["date"]).dt.strftime("%Y-%m-%d")
    settled["date"] = pd.to_datetime(settled["date"]).dt.strftime("%Y-%m-%d")

    # Merge presettlement with settlement on date+ticker
    merged = settled.merge(
        pre[["date", "ticker", "presettlement_prob", "bid_cents", "ask_cents"]],
        on=["date", "ticker"],
        how="inner",
    )
    merged = merged.dropna(subset=["presettlement_prob", "actual_outcome"])
    logger.info("  Merged real Kalshi rows: %d", len(merged))

    if len(merged) == 0:
        logger.warning("No matched real Kalshi rows after merge")
        return None

    # Map model predictions to contract rows
    preds_by_date = dict(zip(
        preds_df["date"].astype(str),
        zip(preds_df["mu"].values, preds_df["sigma"].values),
    ))

    city_code_upper = city_code.upper()

    # Compute model prob for each contract row
    rows = []
    for _, row in merged.iterrows():
        date_str = row["date"]
        if date_str not in preds_by_date:
            continue
        mu, sigma = preds_by_date[date_str]
        sigma = max(sigma, 0.5)

        th_lo = float(row.get("threshold_low", float("nan")))
        th_hi = float(row.get("threshold_high", float("nan")))
        direction = str(row.get("direction", "between"))

        if direction in ("below", "less"):
            model_prob = norm.cdf(th_hi, mu, sigma)
        elif direction == "above":
            model_prob = 1.0 - norm.cdf(th_lo, mu, sigma)
        else:
            # "between" bucket
            if np.isnan(th_lo) or np.isnan(th_hi):
                continue
            model_prob = norm.cdf(th_hi, mu, sigma) - norm.cdf(th_lo, mu, sigma)

        model_prob = float(np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))
        market_prob = float(np.clip(row["presettlement_prob"], PROB_CLIP_MIN, PROB_CLIP_MAX))

        # Convert bid/ask cents to prices
        bid_price = row.get("bid_cents", market_prob * 100)
        if pd.isna(bid_price):
            bid_price = market_prob * 100
        bid_price = float(bid_price) / 100.0

        ask_price = row.get("ask_cents", market_prob * 100)
        if pd.isna(ask_price):
            ask_price = market_prob * 100
        ask_price = float(ask_price) / 100.0

        rows.append({
            "date": date_str,
            "ticker": row["ticker"],
            "bucket_label": row.get("bucket", row.get("bucket_label", "")),
            "model_prob": model_prob,
            "market_prob": market_prob,
            "bid_price": float(np.clip(bid_price, 0.01, 0.99)),
            "ask_price": float(np.clip(ask_price, 0.01, 0.99)),
            "actual_outcome": int(row["actual_outcome"]),
            "volume": row.get("volume", 10),
        })

    if not rows:
        logger.warning("No matched real Kalshi rows with model predictions")
        return None

    market_df = pd.DataFrame(rows)
    logger.info("  Real Kalshi market rows with model predictions: %d (%d unique dates)",
                len(market_df), market_df["date"].nunique())

    # Save real Kalshi market data
    real_market_path = os.path.join(backtest_dir, "real_kalshi_market_data.csv")
    market_df.to_csv(real_market_path, index=False)
    logger.info("  Saved real Kalshi market data to %s", real_market_path)

    # Run the same EV-gated backtest
    backtest_result = run_ev_gated_backtest(
        market_df=market_df,
        ev_threshold=ev_threshold,
        fee_rate=fee_rate,
        kelly_fraction=kelly_fraction,
        max_contracts=max_contracts,
        initial_bankroll=initial_bankroll,
    )

    # Compute Brier scores on real Kalshi contract rows
    model_brier_sum = 0.0
    market_brier_sum = 0.0
    n_brier = 0
    for _, r in market_df.iterrows():
        outcome = r["actual_outcome"]
        if np.isnan(outcome):
            continue
        model_brier_sum += (r["model_prob"] - outcome) ** 2
        market_brier_sum += (r["market_prob"] - outcome) ** 2
        n_brier += 1

    # Assemble metrics
    metrics = {
        "source": "real_kalshi_presettlement",
        "total_pnl": backtest_result["total_pnl"],
        "initial_bankroll": backtest_result["initial_bankroll"],
        "final_bankroll": backtest_result["final_bankroll"],
        "return_pct": backtest_result["total_pnl"] / backtest_result["initial_bankroll"] * 100,
        "n_trades": backtest_result["n_trades"],
        "win_rate": backtest_result["win_rate"],
        "n_trading_days": len(backtest_result["daily_pnl"]),
        "n_contract_rows": len(market_df),
        "n_unique_dates": int(market_df["date"].nunique()),
    }

    # Drawdown
    bankroll_series = backtest_result["bankroll_series"]
    metrics.update(compute_drawdown_metrics(
        bankroll_series, backtest_result["initial_bankroll"],
    ))
    metrics["busted"] = backtest_result.get("busted", False)
    metrics["bust_date"] = backtest_result.get("bust_date")

    # Sharpe
    daily_pnl = backtest_result["daily_pnl"]
    if len(daily_pnl) > 1:
        daily_returns = daily_pnl / backtest_result["initial_bankroll"]
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        metrics["sharpe_ratio"] = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    else:
        metrics["sharpe_ratio"] = 0.0

    # Brier scores
    if n_brier > 0:
        metrics["model_brier"] = model_brier_sum / n_brier
        metrics["market_brier"] = market_brier_sum / n_brier
        metrics["brier_edge"] = metrics["market_brier"] - metrics["model_brier"]
    else:
        metrics["model_brier"] = float("nan")
        metrics["market_brier"] = float("nan")
        metrics["brier_edge"] = float("nan")

    # Seasonal breakdown
    if backtest_result["trades"]:
        trades_df = pd.DataFrame(backtest_result["trades"])
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["month"] = trades_df["date"].dt.month
        trades_df["season"] = trades_df["month"].map(SEASON_MAP)
        seasonal_metrics = {}
        for season in SEASON_ORDER:
            s_df = trades_df[trades_df["season"] == season]
            if len(s_df) == 0:
                seasonal_metrics[season] = {"n_trades": 0, "pnl": 0.0, "win_rate": 0.0}
                continue
            seasonal_metrics[season] = {
                "n_trades": len(s_df),
                "pnl": float(s_df["pnl"].sum()),
                "win_rate": float(s_df["won"].mean()),
                "avg_ev": float(s_df["ev"].mean()),
            }
        metrics["seasonal"] = seasonal_metrics
    else:
        metrics["seasonal"] = {}

    # Save metrics (canonical filename — promotion evaluation reads this)
    metrics_path = os.path.join(backtest_dir, "real_kalshi_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("  Saved real Kalshi backtest metrics to %s", metrics_path)

    # Save trades
    if backtest_result["trades"]:
        trades_df_out = pd.DataFrame(backtest_result["trades"])
        trades_path = os.path.join(backtest_dir, "real_kalshi_trades.csv")
        trades_df_out.to_csv(trades_path, index=False)
        logger.info("  Saved %d real Kalshi trades to %s",
                     len(trades_df_out), trades_path)

    # Visualization: P&L curve for real Kalshi backtest
    if len(bankroll_series) > 0:
        plot_pnl_curve(
            bankroll_series,
            initial_bankroll,
            os.path.join(backtest_dir, "real_kalshi_pnl_curve.png"),
            title=f"{city_code_upper} Real Kalshi Backtest: Cumulative P&L",
        )

    return metrics


# ===========================================================================
# Main Orchestration
# ===========================================================================

def main() -> None:
    """Run the full market backtest for the selected city."""
    parser = argparse.ArgumentParser(
        description="Unified multi-city Kalshi market backtest.",
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=["nyc", "chi", "phl", "atl", "aus"],
        help="City code to run backtest for (chi, phl, atl, aus).",
    )
    args = parser.parse_args()

    city_code: str = args.city

    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    city_code_upper = city_code.upper()

    backtest_dir = os.path.join(cfg.results_dir, "backtest")
    os.makedirs(backtest_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("%s %s Market Backtest", cfg.city_name, cfg.kalshi_ticker)
    logger.info("  City:           %s", cfg.city_name)
    logger.info("  Ticker:         %s", cfg.kalshi_ticker)
    logger.info("  Station:        %s (%s)", cfg.target_station, cfg.target_station_name)
    logger.info("  Buckets:        %d", len(cfg.bucket_edges))
    logger.info("  EV threshold:   %.2f", DEFAULT_EV_THRESHOLD)
    logger.info("  Fee rate:       %.2f", DEFAULT_FEE_RATE)
    logger.info("  Kelly fraction: %.2f", DEFAULT_KELLY_FRACTION)
    logger.info("  Max contracts:  %d", DEFAULT_MAX_CONTRACTS)
    logger.info("  Output:         %s", backtest_dir)
    logger.info("=" * 70)

    # ---- Step 1: Load calibrated predictions ----
    logger.info("Step 1: Loading calibrated model predictions ...")
    preds_df = load_calibrated_predictions(
        cfg.results_dir, city_code, cfg.city_name,
    )
    logger.info("  Loaded %d predictions (%s to %s)",
                len(preds_df),
                preds_df["date"].min(),
                preds_df["date"].max())

    # Use only the OOS portion (last 30%) for backtesting
    n = len(preds_df)
    oos_start = int(n * 0.70)
    oos_df = preds_df.iloc[oos_start:].reset_index(drop=True)
    logger.info("  Using OOS portion: %d days", len(oos_df))

    # ---- Step 2: Generate simulated market data ----
    logger.info("Step 2: Generating simulated %s market data ...", cfg.kalshi_ticker)
    market_df = generate_market_data(
        dates=pd.to_datetime(oos_df["date"]).values,
        actual_tmax=oos_df["actual_tmax"].values,
        model_mu=oos_df["mu"].values,
        model_sigma=oos_df["sigma"].values,
        bucket_edges=cfg.bucket_edges,
        bucket_labels=cfg.bucket_labels,
        city_code=city_code,
        market_noise_std=0.06,
        seed=42,
    )

    # Save market data
    market_path = os.path.join(backtest_dir, "simulated_market_data.csv")
    market_df.to_csv(market_path, index=False)
    logger.info("  Saved market data to %s (%d rows)", market_path, len(market_df))

    # ---- Step 3: Run EV-gated backtest ----
    logger.info("Step 3: Running EV-gated backtest ...")
    backtest_result = run_ev_gated_backtest(
        market_df=market_df,
        ev_threshold=DEFAULT_EV_THRESHOLD,
        fee_rate=DEFAULT_FEE_RATE,
        kelly_fraction=DEFAULT_KELLY_FRACTION,
        max_contracts=DEFAULT_MAX_CONTRACTS,
        initial_bankroll=DEFAULT_INITIAL_BANKROLL,
    )

    # Save trades
    if backtest_result["trades"]:
        trades_df = pd.DataFrame(backtest_result["trades"])
        trades_path = os.path.join(backtest_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info("  Saved %d trades to %s",
                     len(trades_df), trades_path)

    # ---- Step 4: Compute metrics ----
    logger.info("Step 4: Computing backtest metrics ...")
    metrics = compute_backtest_metrics(
        backtest_result, market_df, cfg.bucket_edges,
    )

    # Save metrics JSON
    metrics_path = os.path.join(backtest_dir, "backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("  Saved metrics to %s", metrics_path)

    # ---- Step 5: Generate visualizations ----
    logger.info("Step 5: Generating backtest visualizations ...")

    # P&L curve
    if len(backtest_result["bankroll_series"]) > 0:
        plot_pnl_curve(
            backtest_result["bankroll_series"],
            DEFAULT_INITIAL_BANKROLL,
            os.path.join(backtest_dir, "pnl_curve.png"),
            title=f"{city_code_upper} {cfg.kalshi_ticker} Backtest: Cumulative P&L",
        )

    # Brier comparison
    if not np.isnan(metrics.get("model_brier", float("nan"))):
        plot_brier_comparison(
            metrics["model_brier"],
            metrics["market_brier"],
            os.path.join(backtest_dir, "brier_comparison.png"),
            title=f"{city_code_upper} Model vs Market: Brier Score",
        )

    # Seasonal P&L
    if metrics.get("seasonal"):
        plot_seasonal_pnl(
            metrics["seasonal"],
            os.path.join(backtest_dir, "seasonal_pnl.png"),
            title=f"{city_code_upper} {cfg.kalshi_ticker} Backtest: Seasonal P&L",
        )

    # Trade EV distribution
    if backtest_result["trades"]:
        plot_trade_ev_distribution(
            backtest_result["trades"],
            os.path.join(backtest_dir, "trade_ev_distribution.png"),
            title=f"{city_code_upper} Backtest: Trade EV Distribution",
        )

    # ---- Step 6: Real Kalshi backtest (canonical trading evaluation) ----
    # Runs for any city with presettlement data on disk; the simulated
    # backtest above is a smoke test only.
    logger.info("Step 6: Running real Kalshi presettlement backtest ...")
    real_kalshi_metrics = run_real_kalshi_backtest(
        preds_df=preds_df,
        cfg=cfg,
        city_code=city_code,
        backtest_dir=backtest_dir,
        ev_threshold=DEFAULT_EV_THRESHOLD,
        fee_rate=DEFAULT_FEE_RATE,
        kelly_fraction=DEFAULT_KELLY_FRACTION,
        max_contracts=DEFAULT_MAX_CONTRACTS,
        initial_bankroll=DEFAULT_INITIAL_BANKROLL,
    )

    # ---- Final summary ----
    logger.info("=" * 70)
    logger.info("%s %s Simulated Market Backtest Complete",
                city_code_upper, cfg.kalshi_ticker)
    logger.info("  Total trades:     %d", metrics["n_trades"])
    logger.info("  Win rate:         %.1f%%", metrics["win_rate"] * 100)
    logger.info("  Total P&L:        $%.2f", metrics["total_pnl"])
    logger.info("  Return:           %.1f%%", metrics["return_pct"])
    logger.info("  Max drawdown:     $%.2f (%.1f%%)",
                metrics["max_drawdown"], metrics["max_drawdown_pct"])
    logger.info("  Sharpe ratio:     %.2f", metrics["sharpe_ratio"])
    logger.info("  Model Brier:      %.4f", metrics.get("model_brier", float("nan")))
    logger.info("  Market Brier:     %.4f", metrics.get("market_brier", float("nan")))
    logger.info("  Brier edge:       %.4f", metrics.get("brier_edge", float("nan")))
    logger.info("  Output dir:       %s", backtest_dir)
    logger.info("=" * 70)

    # Seasonal summary
    if metrics.get("seasonal"):
        if city_code == "aus":
            logger.info("  Seasonal breakdown (simulated):")
        else:
            logger.info("  Seasonal breakdown:")
        for season in SEASON_ORDER:
            s = metrics["seasonal"].get(season, {})
            logger.info("    %s: %d trades, P&L=$%.2f, Win rate=%.1f%%",
                        season, s.get("n_trades", 0),
                        s.get("pnl", 0.0),
                        s.get("win_rate", 0.0) * 100)

    # ---- Austin-specific: Real Kalshi summary ----
    if city_code == "aus":
        if real_kalshi_metrics is not None:
            logger.info("")
            logger.info("=" * 70)
            logger.info("%s %s Real Kalshi Backtest Complete",
                        city_code_upper, cfg.kalshi_ticker)
            logger.info("  Total trades:     %d", real_kalshi_metrics["n_trades"])
            logger.info("  Win rate:         %.1f%%", real_kalshi_metrics["win_rate"] * 100)
            logger.info("  Total P&L:        $%.2f", real_kalshi_metrics["total_pnl"])
            logger.info("  Return:           %.1f%%", real_kalshi_metrics["return_pct"])
            logger.info("  Max drawdown:     $%.2f (%.1f%%)",
                        real_kalshi_metrics["max_drawdown"],
                        real_kalshi_metrics["max_drawdown_pct"])
            logger.info("  Sharpe ratio:     %.2f", real_kalshi_metrics["sharpe_ratio"])
            logger.info("  Model Brier:      %.4f", real_kalshi_metrics.get("model_brier", float("nan")))
            logger.info("  Market Brier:     %.4f", real_kalshi_metrics.get("market_brier", float("nan")))
            logger.info("  Brier edge:       %.4f", real_kalshi_metrics.get("brier_edge", float("nan")))
            logger.info("  Contract rows:    %d", real_kalshi_metrics["n_contract_rows"])
            logger.info("  Trading days:     %d", real_kalshi_metrics["n_trading_days"])
            logger.info("=" * 70)

            # Seasonal summary (real Kalshi)
            if real_kalshi_metrics.get("seasonal"):
                logger.info("  Seasonal breakdown (real Kalshi):")
                for season in SEASON_ORDER:
                    s = real_kalshi_metrics["seasonal"].get(season, {})
                    logger.info("    %s: %d trades, P&L=$%.2f, Win rate=%.1f%%",
                                season, s.get("n_trades", 0),
                                s.get("pnl", 0.0),
                                s.get("win_rate", 0.0) * 100)
        else:
            logger.warning("Real Kalshi backtest was skipped (data not available)")

        # ---- Save combined summary (Austin-specific) ----
        combined_summary = {
            "simulated_market": {
                "total_pnl": metrics["total_pnl"],
                "return_pct": metrics["return_pct"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "model_brier": metrics.get("model_brier"),
                "market_brier": metrics.get("market_brier"),
                "brier_edge": metrics.get("brier_edge"),
            },
            "real_kalshi": None,
        }
        if real_kalshi_metrics is not None:
            combined_summary["real_kalshi"] = {
                "total_pnl": real_kalshi_metrics["total_pnl"],
                "return_pct": real_kalshi_metrics["return_pct"],
                "sharpe_ratio": real_kalshi_metrics["sharpe_ratio"],
                "n_trades": real_kalshi_metrics["n_trades"],
                "win_rate": real_kalshi_metrics["win_rate"],
                "model_brier": real_kalshi_metrics.get("model_brier"),
                "market_brier": real_kalshi_metrics.get("market_brier"),
                "brier_edge": real_kalshi_metrics.get("brier_edge"),
                "n_contract_rows": real_kalshi_metrics["n_contract_rows"],
                "n_trading_days": real_kalshi_metrics["n_trading_days"],
            }
        combined_path = os.path.join(backtest_dir, "backtest_summary.json")
        with open(combined_path, "w") as f:
            json.dump(combined_summary, f, indent=2, default=str)
        logger.info("Saved combined backtest summary to %s", combined_path)


if __name__ == "__main__":
    main()
