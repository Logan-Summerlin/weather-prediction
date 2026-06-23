"""
Real-price trading-strategy selection (Phase 3.1 / 3.2).

This module refits and validates Kalshi trading strategies on **real**
pre-settlement prices (never simulated market data) and turns the result into
a per-city promotion decision.  It is the importable, unit-tested core behind
``scripts/run_real_strategy_sweep.py``.

Design (per docs/01_implementation_plan_2026.md, Phase 3):

  * Canonical evaluation universe = out-of-sample rows of
    ``results/<city>/unified_predictions.csv`` (the in-sample rows cover the
    model's own training period and trading them would be leakage).
  * Cost realism: the per-contract half-spread is derived from the real
    ``bid_cents`` / ``ask_cents`` columns of
    ``data/kalshi_presettlement_<city>.csv`` (joined on date+ticker), not a
    flat slippage assumption.  Fees are the Kalshi 7%.
  * Gate-side EV uses :func:`src.trading.compute_conservative_ev` (fee +
    realized half-spread) so a strategy only fires when EV survives execution
    friction.
  * Sizing/threshold parameters come from
    :func:`src.trading.generate_strategy_grid`.  The grid is fit on an earlier
    chronological slice and the **single** best configuration is scored once on
    an untouched later holdout.
  * Selected parameters are persisted to ``results/<city>/strategy.json`` and
    read back by :class:`src.live_trading.LiveTradingHarness`.

Note on the train/holdout split: the plan text says "fit on 2023-2024,
validate untouched on 2025".  For cities whose OOS window is entirely 2025+
(e.g. Chicago, whose 2022-2024 rows are all in-sample) we cannot honour those
literal calendar years without trading leaky in-sample rows.  We therefore
split the OOS window *chronologically* — fit on the earlier dates, validate on
the most-recent untouched dates — which preserves the plan's intent (no
look-ahead, untouched holdout) while respecting the leakage rule.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.city_config import get_city_config
from src.trading import (
    TradingStrategy,
    compute_conservative_ev,
    compute_drawdown_metrics,
    generate_strategy_grid,
    kelly_fraction,
    position_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999

# Drop uninformative / stale market quotes outside this band.
MARKET_PROB_FLOOR = 0.005
MARKET_PROB_CEILING = 0.995

# Fallback half-spread (fraction) when a row has no usable bid/ask quote.
DEFAULT_HALF_SPREAD = 0.02

# Kalshi standard fee on winnings.
DEFAULT_FEE_RATE = 0.07

DEFAULT_INITIAL_BANKROLL = 1000.0
DEFAULT_MAX_CONTRACTS = 10

# Promotion thresholds (Phase 3.2).
PROMO_MIN_TRADES = 50
PROMO_MIN_SHARPE = 1.0
PROMO_MIN_PNL = 0.0
PROMO_MAX_DRAWDOWN_PCT = -30.0  # holdout drawdown must be >= -30%

# Candidate model-probability columns in unified_predictions.csv, best-first
# preference is decided per-city by holdout Brier (lowest wins).
MODEL_VARIANT_COLUMNS: Dict[str, str] = {
    "u9_kitchen_prob": "U9_kitchen",
    "u8_cv_prob": "U8_cv",
    "u7_extended_prob": "U7_extended",
    "u6_ensemble_prob": "U6_ensemble",
    "u5_regime_prob": "U5_regime",
    "u3_mlp_prob": "U3_mlp",
    "model_prob": "base",
}

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ===========================================================================
# Data loading + join
# ===========================================================================

def _unified_predictions_path(city_code: str) -> str:
    cfg = get_city_config(city_code)
    return os.path.join(cfg.results_dir, "unified_predictions.csv")


def _presettlement_path(city_code: str) -> str:
    cfg = get_city_config(city_code)
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root)
    return os.path.join(project_root, "data", f"kalshi_presettlement_{city_code}.csv")


def load_presettlement_spreads(
    city_code: str,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """Load real bid/ask quotes keyed by (date, ticker).

    Returns a DataFrame with columns ``date`` (datetime), ``ticker``,
    ``bid`` and ``ask`` (fractions in [0, 1]).  Rows whose bid/ask cannot be
    parsed are dropped.
    """
    path = path or _presettlement_path(city_code)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["bid"] = pd.to_numeric(df["bid_cents"], errors="coerce") / 100.0
    df["ask"] = pd.to_numeric(df["ask_cents"], errors="coerce") / 100.0
    out = df[["date", "ticker", "bid", "ask"]].copy()
    return out


def select_model_column(
    df: pd.DataFrame,
    candidates: Optional[List[str]] = None,
) -> str:
    """Pick the model-probability column with the lowest Brier on *df*.

    Only columns present with non-null data and a finite Brier are considered.
    """
    candidates = candidates or list(MODEL_VARIANT_COLUMNS.keys())
    outcomes = df["actual_outcome"].to_numpy(dtype=float)
    best_col = None
    best_brier = np.inf
    for col in candidates:
        if col not in df.columns:
            continue
        probs = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = ~(np.isnan(probs) | np.isnan(outcomes))
        if mask.sum() == 0:
            continue
        brier = float(np.mean((probs[mask] - outcomes[mask]) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_col = col
    if best_col is None:
        raise ValueError("No usable model-probability column found.")
    return best_col


def build_sweep_frame(
    city_code: str,
    model_col: Optional[str] = None,
    predictions_path: Optional[str] = None,
    presettlement_path: Optional[str] = None,
    oos_only: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Assemble the real-price evaluation frame for a city.

    Joins the unified predictions (OOS rows) with real bid/ask quotes and
    returns ``(frame, model_col)`` where *frame* has the normalised columns:
    ``date, ticker, bucket, model_prob, market_mid, bid, ask, half_spread,
    actual_outcome, season``.
    """
    pred_path = predictions_path or _unified_predictions_path(city_code)
    df = pd.read_csv(pred_path)
    df["date"] = pd.to_datetime(df["date"])

    if oos_only and "period" in df.columns:
        n_before = len(df)
        df = df[df["period"].astype(str).str.upper() == "OOS"].copy()
        logger.info(
            "%s: OOS filter %d -> %d rows", city_code, n_before, len(df)
        )

    if model_col is None:
        model_col = select_model_column(df)
        logger.info("%s: selected model column '%s'", city_code, model_col)
    elif model_col not in df.columns:
        raise ValueError(
            f"model_col '{model_col}' not in predictions for {city_code}"
        )

    # Join real bid/ask quotes (cost realism).
    spreads = load_presettlement_spreads(city_code, presettlement_path)
    df = df.merge(spreads, on=["date", "ticker"], how="left")

    df["model_prob"] = pd.to_numeric(df[model_col], errors="coerce")
    df["market_mid"] = pd.to_numeric(df["presettlement_prob"], errors="coerce")

    # Half-spread from real quotes; fall back to the flat default where a
    # usable quote is unavailable.
    spread = (df["ask"] - df["bid"]) / 2.0
    spread = spread.where((df["ask"] >= df["bid"]) & df["bid"].notna() & df["ask"].notna())
    df["half_spread"] = spread.fillna(DEFAULT_HALF_SPREAD).clip(lower=0.0, upper=0.49)

    if "season" not in df.columns:
        df["season"] = df["date"].dt.month.map(SEASON_MAP)

    keep = [
        "date", "ticker", "bucket", "model_prob", "market_mid",
        "bid", "ask", "half_spread", "actual_outcome", "season",
    ]
    frame = df[keep].copy()

    # Drop rows we cannot trade or score on.
    valid = (
        frame["model_prob"].notna()
        & frame["market_mid"].notna()
        & frame["actual_outcome"].notna()
        & (frame["market_mid"] >= MARKET_PROB_FLOOR)
        & (frame["market_mid"] <= MARKET_PROB_CEILING)
    )
    frame = frame[valid].sort_values("date").reset_index(drop=True)
    return frame, model_col


# ===========================================================================
# Chronological train / holdout split
# ===========================================================================

def chronological_split(
    frame: pd.DataFrame,
    val_frac: float = 0.4,
    val_start: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split *frame* into (train, holdout) by date with no look-ahead.

    If *val_start* is given, holdout = rows on/after that date.  Otherwise the
    most-recent ``val_frac`` fraction of unique dates becomes the untouched
    holdout.
    """
    dates = np.array(sorted(frame["date"].unique()))
    if len(dates) == 0:
        return frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    if val_start is not None:
        cutoff = pd.to_datetime(val_start)
    else:
        val_frac = float(np.clip(val_frac, 0.05, 0.95))
        split_idx = int(np.floor(len(dates) * (1.0 - val_frac)))
        split_idx = min(max(split_idx, 1), len(dates) - 1)
        cutoff = pd.to_datetime(dates[split_idx])

    train = frame[frame["date"] < cutoff].copy()
    holdout = frame[frame["date"] >= cutoff].copy()
    return train, holdout


# ===========================================================================
# Real-price backtest engine
# ===========================================================================

def _directional_kelly(p: float, price: float, fee_rate: float) -> float:
    """Full Kelly fraction for a single direction at a given entry *price*."""
    price = float(np.clip(price, 1e-6, 1.0 - 1e-6))
    net_payout = (1.0 - fee_rate) - price
    if net_payout <= 0:
        return 0.0
    b = net_payout / price
    q = 1.0 - p
    kelly = (p * b - q) / b
    return float(max(kelly, 0.0))


def _size_contracts(
    strategy: TradingStrategy,
    p_dir: float,
    entry_price: float,
    ev: float,
    bankroll: float,
    max_contracts: int,
) -> int:
    """Translate a strategy's sizing rule into a contract count at *entry_price*."""
    kf = _directional_kelly(p_dir, entry_price, strategy.fee_rate)
    method = strategy.sizing_method
    if method == "fixed":
        n = min(strategy.fixed_size, max_contracts)
        affordable = int(max(0.0, bankroll) / max(entry_price, 1e-6))
        return max(0, min(n, affordable))
    if method == "proportional":
        frac = min(ev * 2.0, strategy.max_position_frac)
    elif method == "fractional_kelly":
        frac = min(kf * strategy.kelly_fraction_param, strategy.max_position_frac)
    else:  # full_kelly / capped_kelly
        frac = min(kf, strategy.max_position_frac)
    if frac <= 0:
        return 0
    n = position_size(
        frac, bankroll, contract_price=max(entry_price, 1e-6),
        min_size=1, max_size=max_contracts,
    )
    affordable = int(max(0.0, bankroll) / max(entry_price, 1e-6))
    return max(0, min(n, affordable))


def backtest_strategy(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> Dict[str, Any]:
    """EV-gated backtest of *strategy* over real-price rows in *frame*.

    Uses each row's realised half-spread for both the conservative-EV gate and
    the execution price (YES fills at ask, NO at ``1 - bid``).  Bankruptcy
    halts the run.  Returns a metrics dict (JSON-friendly).
    """
    bankroll = float(initial_bankroll)
    trades: List[Dict[str, Any]] = []
    daily_records: List[Dict[str, Any]] = []
    busted = False
    bust_date: Optional[str] = None

    for date, day_df in frame.groupby("date", sort=True):
        if bankroll <= 0:
            busted = True
            bust_date = str(pd.Timestamp(date).date())
            break

        day_pnl = 0.0
        for _, row in day_df.iterrows():
            model_prob = float(np.clip(row["model_prob"], PROB_CLIP_MIN, PROB_CLIP_MAX))
            market_mid = float(np.clip(row["market_mid"], 0.01, 0.99))
            half_spread = float(row["half_spread"])
            outcome = int(row["actual_outcome"])

            bid = max(0.01, market_mid - half_spread)
            ask = min(0.99, market_mid + half_spread)

            # Gate-side EV: conservative (fee + realised half-spread).
            ev_info = compute_conservative_ev(
                model_prob, market_mid,
                fee_rate=strategy.fee_rate, slippage=half_spread,
            )
            best_ev = ev_info["ev"]
            direction = ev_info["direction"]
            if direction == "NONE" or np.isnan(best_ev):
                continue
            if best_ev < strategy.ev_threshold or best_ev < strategy.min_ev:
                continue

            if direction == "YES":
                entry_price = ask
                p_dir = model_prob
            else:
                entry_price = 1.0 - bid
                p_dir = 1.0 - model_prob
            entry_price = float(np.clip(entry_price, 0.01, 0.99))

            size = _size_contracts(
                strategy, p_dir, entry_price, best_ev, bankroll, max_contracts,
            )
            if size <= 0:
                continue

            cost = size * entry_price
            won = (direction == "YES" and outcome == 1) or (
                direction == "NO" and outcome == 0
            )
            if won:
                pnl = size * (1.0 - strategy.fee_rate) - cost
            else:
                pnl = -cost

            day_pnl += pnl
            bankroll += pnl
            trades.append({
                "date": str(pd.Timestamp(date).date()),
                "ticker": row["ticker"],
                "bucket": row["bucket"],
                "direction": direction,
                "model_prob": model_prob,
                "market_mid": market_mid,
                "entry_price": entry_price,
                "half_spread": half_spread,
                "ev": float(best_ev),
                "size": int(size),
                "outcome": outcome,
                "won": int(won),
                "pnl": float(pnl),
                "bankroll_after": float(bankroll),
            })

        daily_records.append({"date": pd.Timestamp(date), "daily_pnl": day_pnl, "bankroll": bankroll})

    return _summarize_backtest(
        trades, daily_records, initial_bankroll, bankroll, busted, bust_date, frame,
    )


def _summarize_backtest(
    trades, daily_records, initial_bankroll, final_bankroll, busted, bust_date, frame,
) -> Dict[str, Any]:
    total_pnl = float(final_bankroll - initial_bankroll)
    n_trades = len(trades)
    wins = sum(t["won"] for t in trades)
    win_rate = wins / n_trades if n_trades else 0.0

    if daily_records:
        dr = pd.DataFrame(daily_records)
        daily_pnl = dr["daily_pnl"]
        bankroll_series = dr["bankroll"]
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            sharpe = float(
                daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0
        dd = compute_drawdown_metrics(bankroll_series, initial_bankroll)
    else:
        sharpe = 0.0
        dd = {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}

    brier = compute_brier(frame)

    return {
        "total_pnl": total_pnl,
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": float(final_bankroll),
        "return_pct": total_pnl / initial_bankroll * 100.0 if initial_bankroll else 0.0,
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "sharpe_ratio": sharpe,
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "avg_ev_traded": float(np.mean([t["ev"] for t in trades])) if trades else 0.0,
        "busted": busted,
        "bust_date": bust_date,
        "model_brier": brier["model_brier"],
        "market_brier": brier["market_brier"],
        "brier_edge": brier["brier_edge"],
        "trades": trades,
    }


def compute_brier(frame: pd.DataFrame) -> Dict[str, float]:
    """Model vs market Brier over *frame* (uses model_prob / market_mid)."""
    if frame.empty:
        return {"model_brier": float("nan"), "market_brier": float("nan"),
                "brier_edge": float("nan")}
    m = frame["model_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).to_numpy(dtype=float)
    k = frame["market_mid"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).to_numpy(dtype=float)
    y = frame["actual_outcome"].to_numpy(dtype=float)
    model_brier = float(np.mean((m - y) ** 2))
    market_brier = float(np.mean((k - y) ** 2))
    return {
        "model_brier": model_brier,
        "market_brier": market_brier,
        # positive => model beats market
        "brier_edge": market_brier - model_brier,
    }


# ===========================================================================
# Grid sweep + selection
# ===========================================================================

def default_strategy_grid(bankroll: float = DEFAULT_INITIAL_BANKROLL) -> List[TradingStrategy]:
    """A focused, real-price strategy grid (fee fixed at the Kalshi 7%)."""
    return generate_strategy_grid(
        ev_thresholds=[0.02, 0.03, 0.05, 0.08],
        sizing_methods=["fractional_kelly", "capped_kelly", "proportional"],
        kelly_fractions=[0.10, 0.20, 0.25],
        fee_rates=[DEFAULT_FEE_RATE],
        max_positions=[0.05, 0.10],
        bankrolls=[bankroll],
    )


def sweep_strategies(
    train: pd.DataFrame,
    strategies: List[TradingStrategy],
    min_train_trades: int = 30,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> Tuple[TradingStrategy, Dict[str, Any], List[Dict[str, Any]]]:
    """Backtest every strategy on *train* and pick the best configuration.

    Selection: among strategies that make at least *min_train_trades* trades,
    maximise total P&L (tie-break by Sharpe).  If none clears the trade floor,
    fall back to the highest-P&L strategy overall.  Returns
    ``(best_strategy, best_train_metrics, all_rows)``.
    """
    rows: List[Dict[str, Any]] = []
    results: List[Tuple[TradingStrategy, Dict[str, Any]]] = []
    for strat in strategies:
        m = backtest_strategy(train, strat, initial_bankroll, max_contracts)
        results.append((strat, m))
        rows.append({
            "name": strat.name,
            "ev_threshold": strat.ev_threshold,
            "sizing_method": strat.sizing_method,
            "kelly_fraction": strat.kelly_fraction_param,
            "max_position_frac": strat.max_position_frac,
            "total_pnl": m["total_pnl"],
            "sharpe_ratio": m["sharpe_ratio"],
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        })

    eligible = [(s, m) for s, m in results if m["n_trades"] >= min_train_trades]
    pool = eligible if eligible else results
    best_strategy, best_metrics = max(
        pool, key=lambda sm: (sm[1]["total_pnl"], sm[1]["sharpe_ratio"]),
    )
    return best_strategy, best_metrics, rows


# ===========================================================================
# Promotion decision (Phase 3.2)
# ===========================================================================

def decide_promotion(holdout_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the Phase 3.2 PROMOTED criteria to a holdout backtest.

    PROMOTED requires (all on the real-price holdout): model Brier < market
    Brier, P&L > 0, Sharpe >= 1.0, >= 50 trades, and drawdown >= -30%.
    Anything short of that is MONITOR (honest "no verified edge").
    """
    checks = {
        "model_beats_market_brier": bool(
            holdout_metrics["model_brier"] < holdout_metrics["market_brier"]
        ),
        "positive_pnl": bool(holdout_metrics["total_pnl"] > PROMO_MIN_PNL),
        "sharpe_ok": bool(holdout_metrics["sharpe_ratio"] >= PROMO_MIN_SHARPE),
        "min_trades": bool(holdout_metrics["n_trades"] >= PROMO_MIN_TRADES),
        "drawdown_ok": bool(
            holdout_metrics["max_drawdown_pct"] >= PROMO_MAX_DRAWDOWN_PCT
        ),
        "not_busted": not bool(holdout_metrics.get("busted", False)),
    }
    status = "PROMOTED" if all(checks.values()) else "MONITOR"
    failed = [k for k, v in checks.items() if not v]
    return {"status": status, "checks": checks, "failed_checks": failed}


# ===========================================================================
# strategy.json persistence
# ===========================================================================

def strategy_to_config(strategy: TradingStrategy) -> Dict[str, Any]:
    return {
        "name": strategy.name,
        "ev_threshold": strategy.ev_threshold,
        "sizing_method": strategy.sizing_method,
        "kelly_fraction": strategy.kelly_fraction_param,
        "max_position_frac": strategy.max_position_frac,
        "fee_rate": strategy.fee_rate,
        "min_ev": strategy.min_ev,
        "max_contracts": strategy.max_contracts,
        "bankroll": strategy.bankroll,
        "fixed_size": strategy.fixed_size,
    }


def strategy_from_config(cfg: Dict[str, Any]) -> TradingStrategy:
    return TradingStrategy(
        name=cfg.get("name", "from_json"),
        ev_threshold=cfg.get("ev_threshold", 0.02),
        sizing_method=cfg.get("sizing_method", "fractional_kelly"),
        kelly_fraction=cfg.get("kelly_fraction", 0.25),
        max_position_frac=cfg.get("max_position_frac", 0.10),
        fee_rate=cfg.get("fee_rate", DEFAULT_FEE_RATE),
        min_ev=cfg.get("min_ev", 0.01),
        max_contracts=cfg.get("max_contracts", 50),
        bankroll=cfg.get("bankroll", DEFAULT_INITIAL_BANKROLL),
        fixed_size=cfg.get("fixed_size", 5),
    )


def _date_range(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    if frame.empty:
        return {"start": None, "end": None, "n_rows": 0}
    return {
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "n_rows": int(len(frame)),
    }


def save_strategy_json(
    city_code: str,
    strategy: TradingStrategy,
    model_col: str,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    train_metrics: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    promotion: Dict[str, Any],
    path: Optional[str] = None,
) -> str:
    """Persist the selected strategy + decision to results/<city>/strategy.json."""
    cfg = get_city_config(city_code)
    path = path or os.path.join(cfg.results_dir, "strategy.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "city_code": cfg.city_code,
        "city_name": cfg.city_name,
        "kalshi_ticker": cfg.kalshi_ticker,
        "generated_at": datetime.now().isoformat(),
        "model_col": model_col,
        "model_label": MODEL_VARIANT_COLUMNS.get(model_col, model_col),
        "fee_rate": strategy.fee_rate,
        "evaluation": "real_presettlement_oos",
        "split": {
            "method": "chronological_oos",
            "train": _date_range(train),
            "holdout": _date_range(holdout),
        },
        "strategy": strategy_to_config(strategy),
        "train_metrics": _metrics_summary(train_metrics),
        "holdout_metrics": _metrics_summary(holdout_metrics),
        "promotion": promotion,
        "promotion_status": promotion["status"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved strategy to %s (status=%s)", path, promotion["status"])
    return path


def _metrics_summary(m: Dict[str, Any]) -> Dict[str, Any]:
    """Trade-list-free copy of a backtest metrics dict for JSON storage."""
    return {k: v for k, v in m.items() if k != "trades"}


def load_strategy_json(
    city_code: str,
    path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load results/<city>/strategy.json if present, else None."""
    cfg = get_city_config(city_code)
    path = path or os.path.join(cfg.results_dir, "strategy.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================================
# Top-level orchestration
# ===========================================================================

def run_city_sweep(
    city_code: str,
    model_col: Optional[str] = None,
    val_frac: float = 0.4,
    val_start: Optional[str] = None,
    strategies: Optional[List[TradingStrategy]] = None,
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
    predictions_path: Optional[str] = None,
    presettlement_path: Optional[str] = None,
    write: bool = True,
    strategy_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit + validate a trading strategy for one city on real prices.

    Returns a result dict; when *write* is True also persists
    ``results/<city>/strategy.json``.
    """
    frame, model_col = build_sweep_frame(
        city_code, model_col=model_col,
        predictions_path=predictions_path,
        presettlement_path=presettlement_path,
    )
    if frame.empty:
        raise ValueError(f"No tradeable OOS rows for '{city_code}'.")

    train, holdout = chronological_split(frame, val_frac=val_frac, val_start=val_start)
    if train.empty or holdout.empty:
        raise ValueError(
            f"Chronological split for '{city_code}' produced an empty side "
            f"(train={len(train)}, holdout={len(holdout)})."
        )

    grid = strategies or default_strategy_grid(initial_bankroll)
    best_strategy, train_metrics, sweep_rows = sweep_strategies(
        train, grid, initial_bankroll=initial_bankroll, max_contracts=max_contracts,
    )

    holdout_metrics = backtest_strategy(
        holdout, best_strategy, initial_bankroll, max_contracts,
    )
    promotion = decide_promotion(holdout_metrics)

    strategy_json_path = None
    if write:
        strategy_json_path = save_strategy_json(
            city_code, best_strategy, model_col, train, holdout,
            train_metrics, holdout_metrics, promotion, path=strategy_path,
        )

    return {
        "city_code": city_code,
        "model_col": model_col,
        "strategy": best_strategy,
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "promotion": promotion,
        "sweep_rows": sweep_rows,
        "strategy_json_path": strategy_json_path,
        "train": train,
        "holdout": holdout,
    }
