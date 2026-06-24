#!/usr/bin/env python3
"""
Real Kalshi Pre-Settlement Backtest for Chicago and Philadelphia.

Runs an EV-gated trading backtest using REAL Kalshi pre-settlement
probabilities from the unified_predictions.csv files — not simulated
market data.  Tests multiple Unified-family model variants (U3, U5,
U7, U8, U9) against actual Kalshi market prices observed ~24 hours
before contract settlement.

The backtest uses:
  - EV threshold:    0.02 (only trade when expected value > 2 cents)
  - Fee rate:        0.07 (Kalshi standard 7%)
  - Kelly fraction:  0.25 (quarter Kelly for safety)
  - Position limits: max 10 contracts per bucket
  - Spread:          +/- 2 cents around presettlement mid price
  - Initial bankroll: $1000

Data sources:
  - results/chicago/unified_predictions.csv      (6520 rows, 1144 dates)
  - results/philadelphia/unified_predictions.csv  (2667 rows, 451 dates)

Outputs per city (results/<city>/backtest/):
  - real_kalshi_trades.csv           — all executed trades across all variants
  - real_kalshi_metrics.json         — comprehensive metrics for all variants
  - real_kalshi_pnl_curve.png        — P&L curve for best variant
  - real_kalshi_brier_comparison.png — model vs market Brier bar chart
  - real_kalshi_summary.csv          — summary table of all variants

Usage:
    python scripts/run_real_kalshi_backtest.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs  # noqa: E402
from src.trading import compute_drawdown_metrics, kalshi_fee_per_contract  # noqa: E402

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
PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Trading parameters
EV_THRESHOLD = 0.02
FEE_RATE = 0.07
KELLY_FRACTION = 0.25
MAX_CONTRACTS = 10
INITIAL_BANKROLL = 1000.0

# Spread assumption: +/- 2 cents around pre-settlement mid price
HALF_SPREAD = 0.02

# Filter out extreme market prices (uninformative / stale)
MARKET_PROB_FLOOR = 0.005
MARKET_PROB_CEILING = 0.995

# Model variants to evaluate (column name -> display label)
MODEL_VARIANTS: Dict[str, str] = {
    "u3_mlp_prob": "U3_mlp",
    "u5_regime_prob": "U5_regime",
    "u7_extended_prob": "U7_extended",
    "u8_cv_prob": "U8_cv",
    "u9_kitchen_prob": "U9_kitchen",
}

# City configurations to run
CITY_CODES = ["chi", "phl"]
CITY_RESULTS_MAP = {
    "chi": "chicago",
    "phl": "philadelphia",
}


# ===========================================================================
# Data Loading
# ===========================================================================

def load_unified_predictions(city_code: str) -> pd.DataFrame:
    """Load unified predictions CSV for a given city.

    Parameters
    ----------
    city_code : str
        City code ("chi" or "phl").

    Returns
    -------
    pd.DataFrame
        Unified predictions with all model variant probabilities and
        real Kalshi pre-settlement prices.

    Raises
    ------
    FileNotFoundError
        If the unified predictions file does not exist.
    """
    city_dir = CITY_RESULTS_MAP[city_code]
    path = PROJECT_ROOT / "results" / city_dir / "unified_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Unified predictions not found at {path}. "
            f"Run the unified benchmark first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    required_cols = {
        "date", "ticker", "bucket", "actual_outcome", "actual_tmax",
        "presettlement_prob", "season",
    }
    required_cols.update(MODEL_VARIANTS.keys())
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    # Honest evaluation: trade only on out-of-sample predictions. The
    # unified file also contains in-sample (period == "IS") rows, which
    # would inflate every P&L and Brier number if included.
    if "period" in df.columns:
        n_before = len(df)
        df = df[df["period"].astype(str).str.upper() == "OOS"].copy()
        logger.info(
            "OOS filter: %d -> %d rows (%d in-sample rows excluded)",
            n_before, len(df), n_before - len(df),
        )

    logger.info(
        "Loaded %d rows (%d unique dates) from %s",
        len(df), df["date"].nunique(), path,
    )
    return df


# ===========================================================================
# EV-Gated Backtest Engine
# ===========================================================================

def run_variant_backtest(
    df: pd.DataFrame,
    model_col: str,
    ev_threshold: float = EV_THRESHOLD,
    fee_rate: float = FEE_RATE,
    kelly_fraction: float = KELLY_FRACTION,
    max_contracts: int = MAX_CONTRACTS,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> Dict[str, Any]:
    """Run an EV-gated backtest for a single model variant using real
    Kalshi pre-settlement prices.

    For each row (date x bucket):
      1. Use presettlement_prob as the market mid price.
      2. Construct bid/ask with a +/- 2 cent spread.
      3. Use model_col probability as the model belief.
      4. Compute EV for YES and NO sides.
      5. If best EV > threshold, apply fractional Kelly sizing.
      6. Execute trade and compute P&L at settlement.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered unified predictions (NaN/extreme market rows removed).
    model_col : str
        Column name for the model variant probability.
    ev_threshold : float
        Minimum EV to trigger a trade.
    fee_rate : float
        Kalshi fee rate.
    kelly_fraction : float
        Fraction of full Kelly to use.
    max_contracts : int
        Maximum contracts per bucket per day.
    initial_bankroll : float
        Starting bankroll in dollars.

    Returns
    -------
    dict
        Keys: trades, daily_pnl, bankroll_series, total_pnl, win_rate,
        n_trades, initial_bankroll, final_bankroll.
    """
    trades: List[Dict[str, Any]] = []
    bankroll = initial_bankroll
    bankroll_history: List[Dict[str, Any]] = []
    busted = False
    bust_date = None

    dates_sorted = sorted(df["date"].unique())

    for date in dates_sorted:
        # Halt at bankruptcy — continuing to "trade" with no capital
        # silently inflates losses and corrupts drawdown statistics.
        if bankroll <= 0:
            busted = True
            bust_date = str(date)
            logger.warning("Bankroll exhausted on %s — halting backtest", date)
            break

        day_df = df[df["date"] == date]
        day_pnl = 0.0

        for _, row in day_df.iterrows():
            market_mid = row["presettlement_prob"]
            model_prob = row[model_col]
            outcome = row["actual_outcome"]
            bucket_label = row["bucket"]
            ticker = row["ticker"]

            # Skip invalid data
            if pd.isna(outcome) or pd.isna(market_mid) or pd.isna(model_prob):
                continue

            # Clip model probability
            model_prob = float(np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX))
            market_mid = float(np.clip(market_mid, 0.01, 0.99))

            # Construct bid/ask from mid with spread
            bid = max(0.01, market_mid - HALF_SPREAD)
            ask = min(0.99, market_mid + HALF_SPREAD)

            # Compute EV for both sides using Kalshi's real curved fee, charged
            # per contract on entry. A YES contract bought at `ask` costs
            # ask + fee(ask) and pays $1 if outcome=1; symmetric for NO.
            yes_price = ask
            no_price = 1.0 - bid
            ev_yes = model_prob * 1.0 - yes_price - kalshi_fee_per_contract(yes_price)
            ev_no = (1.0 - model_prob) * 1.0 - no_price - kalshi_fee_per_contract(no_price)

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

            price = float(np.clip(price, 0.01, 0.99))
            fee = kalshi_fee_per_contract(price)
            # Net winnings per contract = $1 payout - entry price - entry fee.
            net_payout = 1.0 - price - fee
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

            # Size in contracts (1 contract = $1 face value).
            # Stake (price + fee per contract) can never exceed the bankroll.
            unit_cost = price + fee
            affordable = int(max(0.0, bankroll) / unit_cost)
            n_contracts = min(
                max_contracts,
                max(1, int(frac_kelly * bankroll / price)),
                affordable,
            )
            if n_contracts < 1:
                continue

            # Compute trade P&L. The fee is paid on entry on every contract,
            # win or lose; a winning contract returns its $1 face value.
            cost = n_contracts * price + n_contracts * fee
            if direction == "YES":
                won = int(outcome == 1)
            else:
                won = int(outcome == 0)

            if won:
                pnl = n_contracts * 1.0 - cost
            else:
                pnl = -cost

            day_pnl += pnl
            bankroll += pnl

            trades.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "ticker": ticker,
                "bucket": bucket_label,
                "direction": direction,
                "model_prob": float(model_prob),
                "market_mid": float(market_mid),
                "bid": float(bid),
                "ask": float(ask),
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
# Metrics Computation
# ===========================================================================

def compute_variant_metrics(
    backtest_result: Dict[str, Any],
    df: pd.DataFrame,
    model_col: str,
) -> Dict[str, Any]:
    """Compute comprehensive metrics for a single model variant backtest.

    Parameters
    ----------
    backtest_result : dict
        Output from run_variant_backtest().
    df : pd.DataFrame
        Unified predictions data (filtered, used for Brier computation).
    model_col : str
        Column name for the model variant probability.

    Returns
    -------
    dict
        Comprehensive metrics dict suitable for JSON serialization.
    """
    trades = backtest_result["trades"]
    daily_pnl = backtest_result["daily_pnl"]
    bankroll_series = backtest_result["bankroll_series"]

    metrics: Dict[str, Any] = {}

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

    # ---- Sharpe ratio (annualized, 252 trading days) ----
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

    # ---- Brier scores: model vs Kalshi pre-settlement ----
    valid = df.dropna(subset=[model_col, "presettlement_prob", "actual_outcome"])
    if len(valid) > 0:
        model_probs = valid[model_col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).values
        market_probs = valid["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).values
        outcomes = valid["actual_outcome"].values.astype(float)

        metrics["model_brier"] = float(np.mean((model_probs - outcomes) ** 2))
        metrics["market_brier"] = float(np.mean((market_probs - outcomes) ** 2))
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

        seasonal_metrics: Dict[str, Any] = {}
        for season in SEASON_ORDER:
            s_df = trades_df[trades_df["season"] == season]
            if len(s_df) == 0:
                seasonal_metrics[season] = {
                    "n_trades": 0, "pnl": 0.0, "win_rate": 0.0, "avg_ev": 0.0,
                }
                continue
            seasonal_metrics[season] = {
                "n_trades": int(len(s_df)),
                "pnl": float(s_df["pnl"].sum()),
                "win_rate": float(s_df["won"].mean()),
                "avg_ev": float(s_df["ev"].mean()),
            }
        metrics["seasonal"] = seasonal_metrics
    else:
        metrics["seasonal"] = {
            s: {"n_trades": 0, "pnl": 0.0, "win_rate": 0.0, "avg_ev": 0.0}
            for s in SEASON_ORDER
        }

    return metrics


# ===========================================================================
# Visualization
# ===========================================================================

def plot_pnl_curve(
    bankroll_series: pd.Series,
    initial_bankroll: float,
    output_path: str,
    title: str,
) -> None:
    """Plot bankroll and drawdown over time for the best variant."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

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
    variant_briers: Dict[str, float],
    market_brier: float,
    output_path: str,
    title: str,
) -> None:
    """Plot a grouped bar chart comparing model variant Brier scores
    against the Kalshi pre-settlement baseline."""
    labels = list(variant_briers.keys()) + ["Kalshi Market"]
    scores = list(variant_briers.values()) + [market_brier]
    colors = ["#2ca02c"] * len(variant_briers) + ["#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, edgecolor="black", width=0.6)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate best model edge
    best_variant = min(variant_briers, key=variant_briers.get)
    best_edge = market_brier - variant_briers[best_variant]
    edge_color = "#2ca02c" if best_edge > 0 else "#d62728"
    ax.text(
        0.5, 0.92,
        f"Best edge ({best_variant}): {best_edge:+.4f}",
        ha="center", fontsize=12, color=edge_color, fontweight="bold",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Brier comparison to %s", output_path)


# ===========================================================================
# Summary Table Printing
# ===========================================================================

def print_city_summary(
    city_name: str,
    all_metrics: Dict[str, Dict[str, Any]],
    market_brier: float,
) -> None:
    """Print a formatted summary table for one city."""
    print()
    print("=" * 100)
    print(f"  {city_name.upper()} — Real Kalshi Pre-Settlement Backtest Summary")
    print("=" * 100)
    print(
        f"{'Variant':<16} {'Brier':>8} {'Mkt Brier':>10} {'Edge':>8} "
        f"{'P&L ($)':>10} {'Return%':>9} {'Sharpe':>8} "
        f"{'WinRate':>8} {'Trades':>8} {'MaxDD($)':>10} {'Beats Mkt':>10}"
    )
    print("-" * 100)

    for variant_label, m in all_metrics.items():
        model_b = m.get("model_brier", float("nan"))
        mkt_b = m.get("market_brier", float("nan"))
        edge = m.get("brier_edge", float("nan"))
        beats = "YES" if (not np.isnan(edge) and edge > 0) else "no"

        print(
            f"{variant_label:<16} "
            f"{model_b:>8.4f} "
            f"{mkt_b:>10.4f} "
            f"{edge:>+8.4f} "
            f"{m['total_pnl']:>10.2f} "
            f"{m['return_pct']:>8.1f}% "
            f"{m['sharpe_ratio']:>8.2f} "
            f"{m['win_rate'] * 100:>7.1f}% "
            f"{m['n_trades']:>8d} "
            f"{m['max_drawdown']:>10.2f} "
            f"{beats:>10}"
        )

    print("-" * 100)
    print(f"  Kalshi pre-settlement baseline Brier: {market_brier:.4f}")
    print("=" * 100)

    # Seasonal breakdown for each variant
    print()
    print(f"  {city_name.upper()} — Seasonal Breakdown")
    print("-" * 90)
    print(
        f"{'Variant':<16} {'Season':<8} {'Trades':>8} "
        f"{'P&L ($)':>10} {'WinRate':>9} {'Avg EV':>8}"
    )
    print("-" * 90)
    for variant_label, m in all_metrics.items():
        seasonal = m.get("seasonal", {})
        for season in SEASON_ORDER:
            s = seasonal.get(season, {})
            n_t = s.get("n_trades", 0)
            if n_t == 0:
                print(
                    f"{variant_label:<16} {season:<8} {'---':>8} "
                    f"{'---':>10} {'---':>9} {'---':>8}"
                )
            else:
                print(
                    f"{variant_label:<16} {season:<8} {n_t:>8d} "
                    f"{s['pnl']:>10.2f} {s['win_rate'] * 100:>8.1f}% "
                    f"{s.get('avg_ev', 0):>8.4f}"
                )
    print("-" * 90)
    print()


# ===========================================================================
# Per-City Orchestration
# ===========================================================================

def run_city_backtest(city_code: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Run the full real-Kalshi backtest for a single city across all
    model variants.

    Parameters
    ----------
    city_code : str
        City code ("chi" or "phl").

    Returns
    -------
    dict or None
        Mapping of variant label to metrics dict.  None if data cannot
        be loaded.
    """
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)
    city_dir = CITY_RESULTS_MAP[city_code]

    backtest_dir = str(PROJECT_ROOT / "results" / city_dir / "backtest")
    os.makedirs(backtest_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Real Kalshi Backtest — %s (%s)", cfg.city_name, cfg.kalshi_ticker)
    logger.info("  EV threshold:   %.2f", EV_THRESHOLD)
    logger.info("  Fee rate:       %.2f", FEE_RATE)
    logger.info("  Kelly fraction: %.2f", KELLY_FRACTION)
    logger.info("  Max contracts:  %d", MAX_CONTRACTS)
    logger.info("  Half spread:    %.2f", HALF_SPREAD)
    logger.info("  Output:         %s", backtest_dir)
    logger.info("=" * 70)

    # ---- Step 1: Load unified predictions ----
    try:
        df = load_unified_predictions(city_code)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Cannot run backtest for %s: %s", city_code, e)
        return None

    # ---- Step 2: Filter out rows with missing/extreme pre-settlement prices ----
    n_before = len(df)
    mask_valid = (
        df["presettlement_prob"].notna()
        & (df["presettlement_prob"] >= MARKET_PROB_FLOOR)
        & (df["presettlement_prob"] <= MARKET_PROB_CEILING)
        & df["actual_outcome"].notna()
    )
    df_filtered = df[mask_valid].copy()
    n_after = len(df_filtered)
    logger.info(
        "Filtered: %d -> %d rows (%d removed for NaN/extreme presettlement)",
        n_before, n_after, n_before - n_after,
    )

    if n_after == 0:
        logger.error("No valid rows remain after filtering for %s.", city_code)
        return None

    # ---- Step 3: Run backtest for each model variant ----
    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_trades: List[Dict[str, Any]] = []
    best_variant: Optional[str] = None
    best_pnl = -float("inf")
    best_bankroll_series: Optional[pd.Series] = None
    market_brier: Optional[float] = None

    for model_col, variant_label in MODEL_VARIANTS.items():
        # Check that the model column has valid data
        n_valid_model = df_filtered[model_col].notna().sum()
        if n_valid_model == 0:
            logger.warning(
                "Skipping %s for %s: no valid model probabilities.",
                variant_label, city_code,
            )
            continue

        logger.info("Running backtest for %s (%s) ...", variant_label, model_col)

        bt_result = run_variant_backtest(
            df=df_filtered,
            model_col=model_col,
            ev_threshold=EV_THRESHOLD,
            fee_rate=FEE_RATE,
            kelly_fraction=KELLY_FRACTION,
            max_contracts=MAX_CONTRACTS,
            initial_bankroll=INITIAL_BANKROLL,
        )

        metrics = compute_variant_metrics(bt_result, df_filtered, model_col)
        all_metrics[variant_label] = metrics

        # Tag trades with variant name and collect
        for t in bt_result["trades"]:
            t["variant"] = variant_label
        all_trades.extend(bt_result["trades"])

        # Track best variant by P&L
        if bt_result["total_pnl"] > best_pnl:
            best_pnl = bt_result["total_pnl"]
            best_variant = variant_label
            best_bankroll_series = bt_result["bankroll_series"]

        # Market Brier is the same across all variants
        if market_brier is None and not np.isnan(metrics.get("market_brier", float("nan"))):
            market_brier = metrics["market_brier"]

        logger.info(
            "  %s: P&L=$%.2f, %d trades, win=%.1f%%, Brier=%.4f, edge=%+.4f",
            variant_label,
            metrics["total_pnl"],
            metrics["n_trades"],
            metrics["win_rate"] * 100,
            metrics.get("model_brier", float("nan")),
            metrics.get("brier_edge", float("nan")),
        )

    if not all_metrics:
        logger.error("No variant produced valid results for %s.", city_code)
        return None

    if market_brier is None:
        market_brier = float("nan")

    # ---- Step 4: Save outputs ----

    # 4a. All trades CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_path = os.path.join(backtest_dir, "real_kalshi_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info("Saved %d trades to %s", len(trades_df), trades_path)

    # 4b. Metrics JSON (all variants)
    metrics_path = os.path.join(backtest_dir, "real_kalshi_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"market_brier": market_brier, "variants": all_metrics},
            f, indent=2, default=str,
        )
    logger.info("Saved metrics to %s", metrics_path)

    # 4c. Summary CSV
    summary_rows = []
    for variant_label, m in all_metrics.items():
        summary_rows.append({
            "variant": variant_label,
            "model_brier": m.get("model_brier", float("nan")),
            "market_brier": m.get("market_brier", float("nan")),
            "brier_edge": m.get("brier_edge", float("nan")),
            "total_pnl": m["total_pnl"],
            "return_pct": m["return_pct"],
            "sharpe_ratio": m["sharpe_ratio"],
            "win_rate": m["win_rate"],
            "n_trades": m["n_trades"],
            "max_drawdown": m["max_drawdown"],
            "max_drawdown_pct": m["max_drawdown_pct"],
            "avg_pnl_per_trade": m["avg_pnl_per_trade"],
            "avg_ev_traded": m["avg_ev_traded"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(backtest_dir, "real_kalshi_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    # 4d. P&L curve for best variant
    if best_bankroll_series is not None and len(best_bankroll_series) > 0:
        plot_pnl_curve(
            best_bankroll_series,
            INITIAL_BANKROLL,
            os.path.join(backtest_dir, "real_kalshi_pnl_curve.png"),
            title=(
                f"{cfg.city_name} Real Kalshi Backtest: "
                f"{best_variant} P&L Curve"
            ),
        )

    # 4e. Brier comparison chart
    variant_briers = {
        v: m["model_brier"]
        for v, m in all_metrics.items()
        if not np.isnan(m.get("model_brier", float("nan")))
    }
    if variant_briers and not np.isnan(market_brier):
        plot_brier_comparison(
            variant_briers,
            market_brier,
            os.path.join(backtest_dir, "real_kalshi_brier_comparison.png"),
            title=f"{cfg.city_name} — Model vs Kalshi Pre-Settlement Brier",
        )

    # ---- Step 5: Print summary ----
    print_city_summary(cfg.city_name, all_metrics, market_brier)

    return all_metrics


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the real Kalshi pre-settlement backtest for all cities."""
    logger.info("=" * 70)
    logger.info("Real Kalshi Pre-Settlement Backtest — Multi-City")
    logger.info("  Cities: %s", ", ".join(CITY_CODES))
    logger.info("  Model variants: %s", ", ".join(MODEL_VARIANTS.values()))
    logger.info("=" * 70)

    results: Dict[str, Optional[Dict]] = {}
    for city_code in CITY_CODES:
        results[city_code] = run_city_backtest(city_code)

    # ---- Grand summary ----
    print()
    print("=" * 100)
    print("  GRAND SUMMARY — All Cities, All Variants")
    print("=" * 100)
    print(
        f"{'City':<15} {'Variant':<16} {'Brier':>8} {'Edge':>8} "
        f"{'P&L ($)':>10} {'Return%':>9} {'Sharpe':>8} "
        f"{'WinRate':>8} {'Trades':>8} {'Beats Mkt':>10}"
    )
    print("-" * 100)

    for city_code in CITY_CODES:
        city_dir = CITY_RESULTS_MAP[city_code]
        city_label = city_dir.capitalize()
        city_metrics = results.get(city_code)
        if city_metrics is None:
            print(f"{city_label:<15} {'--- NO DATA ---'}")
            continue

        for variant_label, m in city_metrics.items():
            edge = m.get("brier_edge", float("nan"))
            beats = "YES" if (not np.isnan(edge) and edge > 0) else "no"
            print(
                f"{city_label:<15} {variant_label:<16} "
                f"{m.get('model_brier', float('nan')):>8.4f} "
                f"{edge:>+8.4f} "
                f"{m['total_pnl']:>10.2f} "
                f"{m['return_pct']:>8.1f}% "
                f"{m['sharpe_ratio']:>8.2f} "
                f"{m['win_rate'] * 100:>7.1f}% "
                f"{m['n_trades']:>8d} "
                f"{beats:>10}"
            )

    print("=" * 100)
    print()
    logger.info("Real Kalshi backtest complete for all cities.")


if __name__ == "__main__":
    main()
