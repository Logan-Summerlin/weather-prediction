#!/usr/bin/env python3
"""NYC U10 market-debias backtest on real Kalshi quotes (executable prices).

Unlike the U3-U9 real-Kalshi backtest (which trades every bucket against a
synthetic +/-2c spread around the pre-settlement mid), U10:

  * prices buckets with the walk-forward longshot-discount + book-coherence
    model (``src.market_debias``) — every probability strictly out-of-sample;
  * trades ONLY the structural edge: NO-side on longshot buckets with mid
    in (0.05, 0.25], the zone where the favorite-longshot bias has the same
    sign and comparable magnitude in 2023, 2024 and 2025;
  * executes at the REAL bid (buying NO at 1 - yes_bid) with Kalshi's
    curved per-contract fee — no synthetic-liquidity assumptions.

Evaluation window: 2025-01-01 (start of the standard OOS period used by all
unified variants) through the last date of the 2026 true-out-of-sample file
(``data/kalshi_nyc_2026_oos.csv``), which was fetched AFTER the strategy was
designed and acts as live-replay validation.

Outputs:
  * merges a ``U10_market_debias`` entry into
    results/backtest/real_kalshi_metrics.json  (variants schema)
  * results/backtest/u10_market_debias_trades.csv
  * results/backtest/u10_market_debias_pnl_curve.png
  * models/nyc/u10_market_debias.json  (curve fit on all data through 2025,
    the checkpoint a live 2026+ deployment would load)

Usage:
    python scripts/experiments/trading/run_nyc_market_debias_backtest.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_debias import walk_forward_debias, fit_market_debias  # noqa: E402
from src.trading import compute_drawdown_metrics, kalshi_fee_per_contract  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy parameters (match the U3-U9 backtest conventions where shared)
# ---------------------------------------------------------------------------
EV_THRESHOLD = 0.02        # same threshold as the U3-U9 real-Kalshi backtest
KELLY_FRACTION = 0.25      # same quarter-Kelly
MAX_CONTRACTS = 10         # same position cap
INITIAL_BANKROLL = 1000.0  # same bankroll

# Structural edge zone: longshot shorts only (see module docstring).
K_LO = 0.05
K_HI = 0.25

OOS_START = "2025-01-01"

SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

PROB_CLIP = (0.001, 0.999)


# ===========================================================================
# Data assembly
# ===========================================================================

def build_contract_frame() -> pd.DataFrame:
    """Assemble the 2023-2026 contract-level frame with real quotes.

    2023-2025 rows come from results/unified_predictions.csv joined with
    data/kalshi_presettlement_nyc.csv (bid/ask/volume).  2026 rows come from
    data/kalshi_nyc_2026_oos.csv (fetched from the Kalshi API with settlement
    results; pre-2026-May history is no longer served by the API).
    """
    up = pd.read_csv(PROJECT_ROOT / "results" / "unified_predictions.csv")
    pre = pd.read_csv(PROJECT_ROOT / "data" / "kalshi_presettlement_nyc.csv")
    quotes = pre[["date", "ticker", "bid_cents", "ask_cents",
                  "volume", "open_interest"]].drop_duplicates(["date", "ticker"])
    hist = up.merge(quotes, on=["date", "ticker"], how="left")
    hist = hist[["date", "ticker", "direction", "actual_outcome",
                 "presettlement_prob", "bid_cents", "ask_cents",
                 "volume", "open_interest"]].copy()

    frames = [hist]
    oos26_path = PROJECT_ROOT / "data" / "kalshi_nyc_2026_oos.csv"
    if oos26_path.exists():
        oos26 = pd.read_csv(oos26_path)
        oos26 = oos26[["date", "ticker", "direction", "actual_outcome",
                       "presettlement_prob", "bid_cents", "ask_cents",
                       "volume", "open_interest"]].copy()
        frames.append(oos26)
        logger.info("2026 true-OOS rows: %d (%s -> %s)", len(oos26),
                    oos26["date"].min(), oos26["date"].max())
    else:
        logger.warning("No 2026 OOS file found at %s", oos26_path)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["actual_outcome", "presettlement_prob"])
    df = df.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])
    df = df.reset_index(drop=True)
    logger.info("Contract frame: %d rows, %d dates (%s -> %s)",
                len(df), df["date"].nunique(), df["date"].min(), df["date"].max())
    return df


# ===========================================================================
# Backtest engine (real executable quotes)
# ===========================================================================

def run_longshot_short_backtest(df: pd.DataFrame) -> dict[str, Any]:
    """EV-gated longshot-short backtest at real bids with curved fees."""
    bankroll = INITIAL_BANKROLL
    trades: list[dict] = []
    bank_hist: list[dict] = []
    busted = False
    bust_date = None

    for d, day in df.groupby("date", sort=True):
        if bankroll <= 0:
            busted, bust_date = True, str(d)
            break
        day_pnl = 0.0
        for _, r in day.iterrows():
            k = float(np.clip(r["presettlement_prob"], *PROB_CLIP))
            if not (K_LO < k <= K_HI):
                continue
            bid = r["bid_cents"] / 100.0 if pd.notna(r["bid_cents"]) else np.nan
            if np.isnan(bid) or bid <= 0.0 or bid >= 1.0:
                continue
            p = float(r["u10_debias_prob"])
            no_price = 1.0 - bid
            fee = kalshi_fee_per_contract(no_price)
            ev = (1.0 - p) - no_price - fee
            if ev <= EV_THRESHOLD:
                continue
            net_payout = 1.0 - no_price - fee
            if net_payout <= 0:
                continue
            b = net_payout / no_price
            p_win = 1.0 - p
            full_kelly = (p_win * b - (1.0 - p_win)) / b
            if full_kelly <= 0:
                continue
            unit_cost = no_price + fee
            affordable = int(max(0.0, bankroll) / unit_cost)
            n = min(MAX_CONTRACTS,
                    max(1, int(full_kelly * KELLY_FRACTION * bankroll / no_price)),
                    affordable)
            if n < 1:
                continue
            cost = n * unit_cost
            won = int(r["actual_outcome"] == 0)
            pnl = n * 1.0 - cost if won else -cost
            day_pnl += pnl
            bankroll += pnl
            trades.append({
                "date": str(d), "ticker": r["ticker"], "direction_bucket": r["direction"],
                "side": "NO", "market_mid": k, "u10_prob": p,
                "entry_price": no_price, "ev": float(ev), "n_contracts": n,
                "cost": float(cost), "outcome": int(r["actual_outcome"]),
                "won": won, "pnl": float(pnl), "bankroll_after": float(bankroll),
            })
        bank_hist.append({"date": d, "daily_pnl": day_pnl, "bankroll": bankroll})

    bh = pd.DataFrame(bank_hist)
    bh["date"] = pd.to_datetime(bh["date"])
    return {
        "trades": trades,
        "daily_pnl": bh.set_index("date")["daily_pnl"] if len(bh) else pd.Series(dtype=float),
        "bankroll_series": bh.set_index("date")["bankroll"] if len(bh) else pd.Series(dtype=float),
        "total_pnl": float(bankroll - INITIAL_BANKROLL),
        "final_bankroll": float(bankroll),
        "busted": busted,
        "bust_date": bust_date,
    }


def compute_metrics(bt: dict[str, Any], eval_df: pd.DataFrame) -> dict[str, Any]:
    """Metrics in the same schema as the U3-U9 real-Kalshi backtest."""
    trades = bt["trades"]
    daily_pnl = bt["daily_pnl"]

    m: dict[str, Any] = {
        "total_pnl": bt["total_pnl"],
        "initial_bankroll": INITIAL_BANKROLL,
        "final_bankroll": bt["final_bankroll"],
        "return_pct": bt["total_pnl"] / INITIAL_BANKROLL * 100.0,
        "n_trades": len(trades),
        "win_rate": float(np.mean([t["won"] for t in trades])) if trades else 0.0,
        "n_trading_days": int(len(daily_pnl)),
    }
    if trades:
        pnls = [t["pnl"] for t in trades]
        m["avg_pnl_per_trade"] = float(np.mean(pnls))
        m["median_pnl_per_trade"] = float(np.median(pnls))
        m["avg_ev_traded"] = float(np.mean([t["ev"] for t in trades]))
    else:
        m["avg_pnl_per_trade"] = m["median_pnl_per_trade"] = m["avg_ev_traded"] = 0.0

    m.update(compute_drawdown_metrics(bt["bankroll_series"], INITIAL_BANKROLL))
    m["busted"] = bt["busted"]
    m["bust_date"] = bt["bust_date"]

    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        r = daily_pnl / INITIAL_BANKROLL
        m["sharpe_ratio"] = float(r.mean() / r.std() * np.sqrt(252))
    else:
        m["sharpe_ratio"] = 0.0
    m["daily_pnl_mean"] = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
    m["daily_pnl_std"] = float(daily_pnl.std()) if len(daily_pnl) > 1 else 0.0

    # Brier on the evaluation window (model = U10 debias, market = raw mid)
    y = eval_df["actual_outcome"].astype(float).values
    p = eval_df["u10_debias_prob"].clip(*PROB_CLIP).values
    k = eval_df["presettlement_prob"].clip(*PROB_CLIP).values
    m["model_brier"] = float(np.mean((p - y) ** 2))
    m["market_brier"] = float(np.mean((k - y) ** 2))
    m["brier_edge"] = m["market_brier"] - m["model_brier"]

    # Seasonal breakdown
    seasonal: dict[str, Any] = {}
    tdf = pd.DataFrame(trades)
    if len(tdf):
        tdf["season"] = pd.to_datetime(tdf["date"]).dt.month.map(SEASON_MAP)
    for s in SEASON_ORDER:
        sub = tdf[tdf["season"] == s] if len(tdf) else pd.DataFrame()
        seasonal[s] = {
            "n_trades": int(len(sub)),
            "pnl": float(sub["pnl"].sum()) if len(sub) else 0.0,
            "win_rate": float(sub["won"].mean()) if len(sub) else 0.0,
            "avg_ev": float(sub["ev"].mean()) if len(sub) else 0.0,
        }
    m["seasonal"] = seasonal

    # Per-year transparency (2026 rows are the post-design live replay)
    if len(tdf):
        tdf["year"] = pd.to_datetime(tdf["date"]).dt.year
        m["pnl_by_year"] = {
            str(yy): float(g["pnl"].sum()) for yy, g in tdf.groupby("year")
        }
    else:
        m["pnl_by_year"] = {}
    return m


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    df = build_contract_frame()

    # Walk-forward U10 probabilities (year-boundary refits, no leakage).
    df, models = walk_forward_debias(df)

    # Evaluation window: standard OOS start through the end of available data.
    eval_df = df[df["date"] >= OOS_START].copy()
    logger.info("Evaluation window: %s -> %s (%d rows, %d days)",
                eval_df["date"].min(), eval_df["date"].max(),
                len(eval_df), eval_df["date"].nunique())

    bt = run_longshot_short_backtest(eval_df)
    metrics = compute_metrics(bt, eval_df)
    metrics["strategy"] = {
        "name": "longshot_short",
        "description": (
            "NO-side only on buckets with pre-settlement mid in "
            f"({K_LO}, {K_HI}]; walk-forward longshot-discount + coherence "
            "pricing; executed at real bids with curved Kalshi fees"
        ),
        "ev_threshold": EV_THRESHOLD,
        "kelly_fraction": KELLY_FRACTION,
        "max_contracts": MAX_CONTRACTS,
        "execution": "real_bid_ask",
        "eval_window": [str(eval_df["date"].min()), str(eval_df["date"].max())],
    }

    logger.info("U10 P&L $%.2f over %d trades (win %.1f%%), maxDD %.1f%%, Sharpe %.2f",
                metrics["total_pnl"], metrics["n_trades"],
                metrics["win_rate"] * 100, metrics["max_drawdown_pct"],
                metrics["sharpe_ratio"])
    logger.info("P&L by year: %s", metrics["pnl_by_year"])
    logger.info("U10 Brier %.5f vs market %.5f (edge %+.5f)",
                metrics["model_brier"], metrics["market_brier"], metrics["brier_edge"])

    backtest_dir = PROJECT_ROOT / "results" / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    # ---- Trades CSV + P&L curve ----
    pd.DataFrame(bt["trades"]).to_csv(
        backtest_dir / "u10_market_debias_trades.csv", index=False)

    if len(bt["bankroll_series"]):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        s = bt["bankroll_series"]
        axes[0].plot(s.index, s.values, lw=1.5, color="#1f77b4")
        axes[0].axhline(INITIAL_BANKROLL, color="gray", ls="--", lw=1)
        axes[0].set_ylabel("Bankroll ($)")
        axes[0].set_title("NYC U10 market-debias longshot-short backtest "
                          "(real Kalshi quotes, 2025 -> 2026 true-OOS)")
        axes[0].grid(alpha=0.3)
        dd = s - s.cummax()
        axes[1].fill_between(dd.index, 0, dd.values, color="red", alpha=0.3)
        axes[1].set_ylabel("Drawdown ($)")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(backtest_dir / "u10_market_debias_pnl_curve.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Merge into real_kalshi_metrics.json (variants schema) ----
    metrics_path = backtest_dir / "real_kalshi_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {"market_brier": metrics["market_brier"], "variants": {}}
    all_metrics.setdefault("variants", {})["U10_market_debias"] = metrics
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Merged U10 metrics into %s", metrics_path)

    # ---- Live checkpoint: curve fit on ALL data through 2025 ----
    hist = df[df["date"] < "2026-01-01"]
    live_model = fit_market_debias(hist)
    ckpt = PROJECT_ROOT / "models" / "nyc" / "u10_market_debias.json"
    live_model.save(ckpt)
    logger.info("Saved live U10 checkpoint (fit %s -> %s, %d longshot rows) to %s",
                live_model.fit_date_min, live_model.fit_date_max,
                live_model.n_fit_rows, ckpt)


if __name__ == "__main__":
    main()
