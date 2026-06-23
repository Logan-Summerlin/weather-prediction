"""
Paper-trading loop + historical replay (Phase 3.3).

Importable core behind ``scripts/run_paper_trading.py``.  Reads a daily signal
(:mod:`src.daily_inference`), prices the model against the **actual** Kalshi
contracts offered that day, runs the EV/Kelly gate through
:class:`src.live_trading.LiveTradingHarness`, settles against the realised
TMAX, and writes an audit log.  Strictly read-only with respect to Kalshi
(paper mode only; no authenticated endpoints).

Why contract-by-contract pricing: Kalshi re-strikes the daily-high bucket grid
per market (Jan offers even edges like 12-14F, a warm May day offers odd edges
like 69-71F), so a model bucketised onto ``CityConfig``'s single fixed grid
will not line up with the day's real contracts.  We therefore price the model
distribution (mu, sigma) directly against each market contract's own
(lo, hi, direction) through the verified settlement semantics
(:mod:`src.bucket_semantics`).  This is the riskiest seam and the one the
contract/grid-mismatch bug lives in.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.bucket_semantics import bucket_outcome, bucket_prob_gaussian
from src.city_config import get_city_config
from src.kalshi_client import parse_market_buckets
from src.live_trading import (
    DailyPrediction,
    LiveTradingHarness,
    load_city_strategy,
)

logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999


# ===========================================================================
# Market contract assembly
# ===========================================================================

def _safe_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def contracts_from_presettlement(
    city_code: str,
    market_date: str,
    presettlement_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Historical contracts for a date (replay; no network).

    Returns one dict per contract with ``ticker, label, lo, hi, direction,
    price`` and ``outcome`` (settled 1/0 if the file carries it).
    """
    if presettlement_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        presettlement_path = os.path.join(
            project_root, "data", f"kalshi_presettlement_{city_code}.csv"
        )
    df = pd.read_csv(presettlement_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    day = df[df["date"] == str(market_date)[:10]]

    contracts: List[Dict[str, Any]] = []
    for _, row in day.iterrows():
        price = _safe_float(row.get("presettlement_prob"))
        if math.isnan(price):
            continue
        contracts.append({
            "ticker": row.get("ticker", ""),
            "label": str(row.get("bucket", row.get("ticker", ""))),
            "lo": _safe_float(row.get("threshold_low")),
            "hi": _safe_float(row.get("threshold_high")),
            "direction": str(row.get("direction", "between")),
            "price": float(np.clip(price, 0.0, 1.0)),
        })
    return contracts


def contracts_from_live(city_code: str, client=None) -> List[Dict[str, Any]]:
    """Open-market contracts via the read-only Kalshi client (one call)."""
    from src.kalshi_client import KalshiClient

    cfg = get_city_config(city_code)
    client = client or KalshiClient()
    markets = client.get_markets(series_ticker=cfg.kalshi_ticker, status="open")
    parsed = parse_market_buckets(markets)

    contracts: List[Dict[str, Any]] = []
    for b in parsed:
        price = b.get("implied_prob")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        contracts.append({
            "ticker": b.get("ticker", ""),
            "label": b.get("ticker", ""),
            "lo": _safe_float(b.get("threshold_low")),
            "hi": _safe_float(b.get("threshold_high")),
            "direction": str(b.get("direction", "between")),
            "price": float(np.clip(price, 0.0, 1.0)),
        })
    return contracts


def model_prob_for_contract(mu: float, sigma: float, contract: Dict[str, Any]) -> float:
    """Model YES probability for one market contract under N(mu, sigma)."""
    p = bucket_prob_gaussian(
        mu, max(sigma, 0.5), contract["lo"], contract["hi"], contract["direction"],
    )
    return float(np.clip(float(p), PROB_CLIP_MIN, PROB_CLIP_MAX))


def build_market_prediction(
    city_code: str,
    market_date: str,
    mu: float,
    sigma: float,
    contracts: List[Dict[str, Any]],
    model_name: str = "",
) -> DailyPrediction:
    """Price the model onto the day's actual contracts as a DailyPrediction."""
    labels = [c["label"] for c in contracts]
    probs = np.array([model_prob_for_contract(mu, sigma, c) for c in contracts])
    return DailyPrediction(
        city_code=city_code, date=str(market_date)[:10],
        mu=float(mu), sigma=float(sigma),
        bucket_probs=probs, bucket_labels=labels, model_name=model_name,
    )


# ===========================================================================
# Settlement (contract-aware; not tied to the fixed cfg grid)
# ===========================================================================

def settle_paper_trades(
    harness: LiveTradingHarness,
    date: str,
    actual_tmax: float,
    contracts_by_label: Dict[str, Dict[str, Any]],
) -> List:
    """Settle a day's open trades using each contract's own (lo, hi, direction).

    Mirrors :meth:`LiveTradingHarness.settle_trades` but resolves the YES/NO
    outcome from the actual market contract terms via
    :func:`src.bucket_semantics.bucket_outcome`, so it is correct for the
    day's re-struck bucket grid.  Updates trade P&L and the kill switch.
    """
    fee = harness.strategy.fee_rate
    settled = []
    for trade in harness.trade_log:
        if trade.date != date or trade.settled:
            continue
        contract = contracts_by_label.get(trade.bucket_label)
        if contract is None:
            continue
        yes = int(bucket_outcome(
            actual_tmax, contract["lo"], contract["hi"], contract["direction"],
        ))
        if trade.direction == "YES":
            won = yes == 1
            pnl = (trade.size * (1.0 - fee) - trade.size * trade.market_price
                   if won else -trade.size * trade.market_price)
        else:
            won = yes == 0
            pnl = (trade.size * (1.0 - fee) - trade.size * (1.0 - trade.market_price)
                   if won else -trade.size * (1.0 - trade.market_price))
        trade.pnl = pnl
        trade.settled = True
        settled.append(trade)
        harness.kill_switch.check_daily_loss(pnl)

    if settled:
        logger.info(
            "%s settled %d trades for %s: PnL=$%.2f (TMAX=%.1f)",
            harness.city_code.upper(), len(settled), date,
            sum(t.pnl for t in settled), actual_tmax,
        )
    return settled


# ===========================================================================
# Paper-trading cycle
# ===========================================================================

def run_paper_cycle(
    harness: LiveTradingHarness,
    prediction: DailyPrediction,
    contracts: List[Dict[str, Any]],
    actual_tmax: Optional[float] = None,
    save_audit: bool = True,
) -> Dict[str, Any]:
    """Evaluate, (optionally) settle, and audit one day's paper trades."""
    market_prices = {c["label"]: c["price"] for c in contracts}
    contracts_by_label = {c["label"]: c for c in contracts}

    trades = harness.evaluate_trades(prediction, market_prices)
    settled = []
    if actual_tmax is not None and not math.isnan(actual_tmax):
        settled = settle_paper_trades(
            harness, prediction.date, float(actual_tmax), contracts_by_label,
        )
    audit_path = harness.save_audit_log(prediction.date) if save_audit else None

    return {
        "date": prediction.date,
        "city_code": harness.city_code,
        "n_signals": len(trades),
        "trades": [t.to_dict() for t in trades],
        "settled": [t.to_dict() for t in settled],
        "day_pnl": float(sum(t.pnl for t in settled)),
        "kill_switch_active": harness.kill_switch.is_active,
        "kill_switch_reason": harness.kill_switch.reason,
        "audit_path": audit_path,
    }


# ===========================================================================
# Historical replay (e.g. losing-week stress test)
# ===========================================================================

def replay_period(
    city_code: str,
    start_date: str,
    end_date: str,
    strategy=None,
    predictions_path: Optional[str] = None,
    presettlement_path: Optional[str] = None,
    save_audit: bool = False,
) -> Dict[str, Any]:
    """Replay paper trading over a historical date range using real prices.

    Drives the model's persisted (mu, sigma) against each day's real Kalshi
    contracts and settles on realised TMAX.  Used for the losing-week /
    kill-switch replay tests.  No network access.
    """
    cfg = get_city_config(city_code)
    if strategy is None:
        strategy = load_city_strategy(city_code)
    harness = LiveTradingHarness(city_code, mode="paper", strategy=strategy)

    if predictions_path is None:
        predictions_path = os.path.join(
            cfg.results_dir, "synthesis", "synthesis_predictions.csv"
        )
    preds = pd.read_csv(predictions_path)
    preds["date"] = pd.to_datetime(preds["date"])
    mask = (preds["date"] >= pd.to_datetime(start_date)) & (
        preds["date"] <= pd.to_datetime(end_date)
    )
    preds = preds[mask].sort_values("date")

    day_summaries: List[Dict[str, Any]] = []
    for _, row in preds.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        if harness.kill_switch.is_active:
            logger.warning("Kill switch active — stopping replay at %s", date_str)
            break

        contracts = contracts_from_presettlement(
            city_code, date_str, presettlement_path
        )
        if not contracts:
            continue

        prediction = build_market_prediction(
            cfg.city_code, date_str, float(row["mu"]), float(row["sigma"]),
            contracts, model_name="synthesis",
        )
        actual = row.get("actual_tmax")
        summary = run_paper_cycle(
            harness, prediction, contracts,
            actual_tmax=None if pd.isna(actual) else float(actual),
            save_audit=save_audit,
        )
        day_summaries.append(summary)

    settled = [t for t in harness.trade_log if t.settled]
    total_pnl = float(sum(t.pnl for t in settled))
    return {
        "city_code": city_code,
        "start_date": start_date,
        "end_date": end_date,
        "n_days": len(day_summaries),
        "n_trades": len(harness.trade_log),
        "n_settled": len(settled),
        "total_pnl": total_pnl,
        "kill_switch_active": harness.kill_switch.is_active,
        "kill_switch_reason": harness.kill_switch.reason,
        "summary": harness.get_summary(),
        "days": day_summaries,
    }
