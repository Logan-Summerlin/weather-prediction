"""
Multi-City Live Trading Harness for Kalshi Temperature Markets.

Extends the NYC trading infrastructure to support CHI (KXHIGHCHI) and
PHL (KXHIGHPHL) with:
  - City-aware Kalshi ticker routing
  - Per-city kill switches (independent of NYC)
  - Daily inference pipeline: data → features → predict → calibrate → bucketize → EV gate → trade
  - Paper-trading mode for CHI/PHL before going live
  - Audit logging per city

Usage:
    from src.live_trading import LiveTradingHarness
    harness = LiveTradingHarness("chi", mode="paper")
    harness.run_daily_cycle()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, CityConfig
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    TradeSignal,
    compute_ev_best,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999

# ---------------------------------------------------------------------------
# Kalshi ticker routing
# ---------------------------------------------------------------------------
KALSHI_TICKER_MAP = {
    "nyc": "KXHIGHNY",
    "chi": "KXHIGHCHI",
    "phl": "KXHIGHPHL",
}


def get_kalshi_ticker(city_code: str) -> str:
    """Return the Kalshi ticker prefix for a given city code.

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "nyc", "chi", "phl").

    Returns
    -------
    str
        Kalshi ticker prefix (e.g., "KXHIGHCHI").

    Raises
    ------
    ValueError
        If city_code is not recognized.
    """
    code = city_code.strip().lower()
    if code not in KALSHI_TICKER_MAP:
        raise ValueError(
            f"Unknown city code '{city_code}'. "
            f"Available: {list(KALSHI_TICKER_MAP.keys())}"
        )
    return KALSHI_TICKER_MAP[code]


# ---------------------------------------------------------------------------
# Per-city kill switch
# ---------------------------------------------------------------------------

@dataclass
class KillSwitch:
    """Per-city kill switch for halting trading on critical failures.

    Attributes
    ----------
    city_code : str
        City identifier.
    is_active : bool
        If True, trading is halted for this city.
    reason : str
        Reason for the kill switch activation.
    activated_at : str
        ISO timestamp of activation.
    max_daily_loss : float
        Maximum daily loss before auto-kill (in dollars).
    max_consecutive_losses : int
        Maximum consecutive losing trades before auto-kill.
    current_daily_loss : float
        Running daily loss tracker.
    consecutive_losses : int
        Running consecutive loss counter.
    """

    city_code: str
    is_active: bool = False
    reason: str = ""
    activated_at: str = ""
    max_daily_loss: float = 500.0
    max_consecutive_losses: int = 10
    current_daily_loss: float = 0.0
    consecutive_losses: int = 0

    def activate(self, reason: str) -> None:
        """Activate the kill switch."""
        self.is_active = True
        self.reason = reason
        self.activated_at = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "KILL SWITCH ACTIVATED for %s: %s", self.city_code, reason
        )

    def deactivate(self) -> None:
        """Deactivate the kill switch (manual reset required)."""
        self.is_active = False
        self.reason = ""
        self.activated_at = ""
        logger.info("Kill switch deactivated for %s", self.city_code)

    def check_daily_loss(self, pnl: float) -> bool:
        """Update daily loss and check if kill switch should trigger.

        Parameters
        ----------
        pnl : float
            P&L from the latest trade.

        Returns
        -------
        bool
            True if trading should continue, False if killed.
        """
        if pnl < 0:
            self.current_daily_loss += abs(pnl)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.current_daily_loss >= self.max_daily_loss:
            self.activate(
                f"Daily loss limit reached: ${self.current_daily_loss:.2f} "
                f">= ${self.max_daily_loss:.2f}"
            )
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.activate(
                f"Consecutive loss limit reached: {self.consecutive_losses} "
                f">= {self.max_consecutive_losses}"
            )
            return False

        return True

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self.current_daily_loss = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "city_code": self.city_code,
            "is_active": self.is_active,
            "reason": self.reason,
            "activated_at": self.activated_at,
            "max_daily_loss": self.max_daily_loss,
            "max_consecutive_losses": self.max_consecutive_losses,
            "current_daily_loss": self.current_daily_loss,
            "consecutive_losses": self.consecutive_losses,
        }


# ---------------------------------------------------------------------------
# Daily inference pipeline
# ---------------------------------------------------------------------------

@dataclass
class DailyPrediction:
    """Container for a single day's model prediction.

    Attributes
    ----------
    city_code : str
        City identifier.
    date : str
        Forecast date (YYYY-MM-DD).
    mu : float
        Predicted mean TMAX (°F).
    sigma : float
        Predicted std dev of TMAX (°F).
    bucket_probs : np.ndarray
        Probability for each Kalshi bucket.
    bucket_labels : list of str
        Labels for each bucket.
    model_name : str
        Name of the model that generated the prediction.
    """

    city_code: str
    date: str
    mu: float
    sigma: float
    bucket_probs: np.ndarray
    bucket_labels: List[str]
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (JSON-safe)."""
        return {
            "city_code": self.city_code,
            "date": self.date,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "bucket_probs": self.bucket_probs.tolist(),
            "bucket_labels": self.bucket_labels,
            "model_name": self.model_name,
        }


def gaussian_to_bucket_probs(
    mu: float,
    sigma: float,
    bucket_edges: List[tuple],
) -> np.ndarray:
    """Convert N(mu, sigma) to per-bucket probabilities.

    Parameters
    ----------
    mu : float
        Predicted mean.
    sigma : float
        Predicted standard deviation.
    bucket_edges : list of (float, float)
        Bucket boundary tuples with -999/999 sentinels.

    Returns
    -------
    np.ndarray
        Probability for each bucket, summing to 1.0.
    """
    sigma = max(sigma, 0.5)
    probs = []
    for lo, hi in bucket_edges:
        cdf_lo = 0.0 if lo <= -900 else float(norm.cdf(lo, mu, sigma))
        cdf_hi = 1.0 if hi >= 900 else float(norm.cdf(hi, mu, sigma))
        p = max(cdf_hi - cdf_lo, PROB_CLIP_MIN)
        probs.append(p)

    probs = np.array(probs)
    probs = probs / probs.sum()
    return np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Record of a single trade execution or paper trade.

    Attributes
    ----------
    city_code : str
        City identifier.
    date : str
        Trade date.
    ticker : str
        Kalshi contract ticker.
    bucket_label : str
        Human-readable bucket label.
    direction : str
        "YES" or "NO".
    size : int
        Number of contracts.
    model_prob : float
        Model probability for this bucket.
    market_price : float
        Market price at time of trade.
    ev : float
        Expected value of the trade.
    mode : str
        "live" or "paper".
    pnl : float
        Realized P&L (0 if not yet settled).
    settled : bool
        Whether the trade has been settled.
    """

    city_code: str
    date: str
    ticker: str
    bucket_label: str
    direction: str
    size: int
    model_prob: float
    market_price: float
    ev: float
    mode: str = "paper"
    pnl: float = 0.0
    settled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "city_code": self.city_code,
            "date": self.date,
            "ticker": self.ticker,
            "bucket_label": self.bucket_label,
            "direction": self.direction,
            "size": self.size,
            "model_prob": self.model_prob,
            "market_price": self.market_price,
            "ev": self.ev,
            "mode": self.mode,
            "pnl": self.pnl,
            "settled": self.settled,
        }


# ---------------------------------------------------------------------------
# Live Trading Harness
# ---------------------------------------------------------------------------

class LiveTradingHarness:
    """Multi-city live/paper trading harness.

    Orchestrates the daily inference→trading pipeline for a single city.
    Can run in "paper" mode (log trades without executing) or "live" mode
    (submit orders via Kalshi API).

    Parameters
    ----------
    city_code : str
        City identifier ("nyc", "chi", "phl").
    mode : str
        "paper" for paper trading, "live" for actual trading.
    strategy : TradingStrategy, optional
        Trading strategy. Defaults to conservative fractional Kelly.
    kill_switch : KillSwitch, optional
        Per-city kill switch. Created automatically if not provided.
    audit_dir : str, optional
        Directory for audit logs. Defaults to results/{city}/trading/.

    Examples
    --------
    >>> harness = LiveTradingHarness("chi", mode="paper")
    >>> harness.evaluate_trades(prediction, market_prices)
    """

    def __init__(
        self,
        city_code: str,
        mode: str = "paper",
        strategy: Optional[TradingStrategy] = None,
        kill_switch: Optional[KillSwitch] = None,
        audit_dir: Optional[str] = None,
    ):
        self.city_code = city_code.strip().lower()
        self.city_config = get_city_config(self.city_code)
        self.kalshi_ticker = get_kalshi_ticker(self.city_code)

        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got '{mode}'")
        self.mode = mode

        self.strategy = strategy or TradingStrategy(
            name=f"{self.city_code}_default",
            ev_threshold=0.03,
            sizing_method="fractional_kelly",
            kelly_fraction=0.20,
            max_position_frac=0.08,
            fee_rate=0.07,
            bankroll=10000.0,
        )

        self.kill_switch = kill_switch or KillSwitch(
            city_code=self.city_code,
            max_daily_loss=500.0,
            max_consecutive_losses=10,
        )

        if audit_dir is None:
            audit_dir = os.path.join(
                self.city_config.results_dir, "trading"
            )
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)

        self.trade_log: List[TradeRecord] = []

        logger.info(
            "LiveTradingHarness initialized: city=%s, ticker=%s, mode=%s",
            self.city_code, self.kalshi_ticker, self.mode,
        )

    def evaluate_trades(
        self,
        prediction: DailyPrediction,
        market_prices: Dict[str, float],
    ) -> List[TradeRecord]:
        """Evaluate trading opportunities for a day's prediction.

        Parameters
        ----------
        prediction : DailyPrediction
            Model prediction with bucket probabilities.
        market_prices : dict
            Mapping of bucket_label → market price (0–1).

        Returns
        -------
        list of TradeRecord
            List of trade signals (may be empty if no +EV opportunities
            or kill switch is active).
        """
        if self.kill_switch.is_active:
            logger.warning(
                "Kill switch active for %s: %s. No trades.",
                self.city_code, self.kill_switch.reason,
            )
            return []

        trades = []
        bucket_labels = prediction.bucket_labels
        bucket_probs = prediction.bucket_probs

        for i, label in enumerate(bucket_labels):
            model_prob = float(bucket_probs[i])
            market_price = market_prices.get(label, None)

            if market_price is None or np.isnan(market_price):
                continue

            signal = self.strategy.evaluate_trade(model_prob, market_price)

            if signal.direction == "NONE" or signal.size == 0:
                continue

            # Build ticker for this specific contract
            # Kalshi tickers follow pattern: KXHIGHCHI-YYYY-MM-DD-T{threshold}
            ticker = f"{self.kalshi_ticker}-{prediction.date}"

            trade = TradeRecord(
                city_code=self.city_code,
                date=prediction.date,
                ticker=ticker,
                bucket_label=label,
                direction=signal.direction,
                size=signal.size,
                model_prob=model_prob,
                market_price=market_price,
                ev=signal.ev,
                mode=self.mode,
            )
            trades.append(trade)

        if trades:
            logger.info(
                "%s %s: %d trade signals for %s",
                self.city_code.upper(), self.mode,
                len(trades), prediction.date,
            )

            if self.mode == "paper":
                for t in trades:
                    logger.info(
                        "  [PAPER] %s %s %d@%.3f (model=%.3f, EV=%.4f)",
                        t.direction, t.bucket_label, t.size,
                        t.market_price, t.model_prob, t.ev,
                    )

        self.trade_log.extend(trades)
        return trades

    def settle_trades(
        self,
        date: str,
        actual_tmax: float,
    ) -> List[TradeRecord]:
        """Settle open trades for a given date using actual TMAX.

        Parameters
        ----------
        date : str
            Settlement date (YYYY-MM-DD).
        actual_tmax : float
            Observed maximum temperature.

        Returns
        -------
        list of TradeRecord
            List of settled trades with updated P&L.
        """
        bucket_edges = self.city_config.bucket_edges
        settled = []

        for trade in self.trade_log:
            if trade.date != date or trade.settled:
                continue

            # Determine which bucket the actual temp falls in
            actual_in_bucket = False
            for i, (lo, hi) in enumerate(bucket_edges):
                label = self.city_config.bucket_labels[i]
                if label == trade.bucket_label:
                    if i == len(bucket_edges) - 1:
                        actual_in_bucket = lo <= actual_tmax <= hi
                    else:
                        actual_in_bucket = lo <= actual_tmax < hi
                    break

            # Compute P&L
            if trade.direction == "YES":
                if actual_in_bucket:
                    pnl = trade.size * (1.0 - self.strategy.fee_rate) - trade.size * trade.market_price
                else:
                    pnl = -trade.size * trade.market_price
            else:  # NO
                if not actual_in_bucket:
                    pnl = trade.size * (1.0 - self.strategy.fee_rate) - trade.size * (1.0 - trade.market_price)
                else:
                    pnl = -trade.size * (1.0 - trade.market_price)

            trade.pnl = pnl
            trade.settled = True
            settled.append(trade)

            # Check kill switch
            self.kill_switch.check_daily_loss(pnl)

        if settled:
            total_pnl = sum(t.pnl for t in settled)
            logger.info(
                "%s settled %d trades for %s: PnL=$%.2f (actual TMAX=%.1f°F)",
                self.city_code.upper(), len(settled), date, total_pnl, actual_tmax,
            )

        return settled

    def save_audit_log(self, date: str) -> str:
        """Save audit log for a trading day.

        Parameters
        ----------
        date : str
            Trading date (YYYY-MM-DD).

        Returns
        -------
        str
            Path to the saved audit log.
        """
        day_trades = [t for t in self.trade_log if t.date == date]

        audit = {
            "city_code": self.city_code,
            "kalshi_ticker": self.kalshi_ticker,
            "date": date,
            "mode": self.mode,
            "n_trades": len(day_trades),
            "total_pnl": sum(t.pnl for t in day_trades),
            "kill_switch": self.kill_switch.to_dict(),
            "strategy": {
                "name": self.strategy.name,
                "ev_threshold": self.strategy.ev_threshold,
                "sizing_method": self.strategy.sizing_method,
                "fee_rate": self.strategy.fee_rate,
            },
            "trades": [t.to_dict() for t in day_trades],
        }

        path = os.path.join(
            self.audit_dir,
            f"trading_audit_{self.city_code}_{date}.json",
        )
        with open(path, "w") as f:
            json.dump(audit, f, indent=2, default=str)
        logger.info("Saved audit log to %s", path)
        return path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all trades in the current session.

        Returns
        -------
        dict
            Summary statistics.
        """
        if not self.trade_log:
            return {
                "city_code": self.city_code,
                "mode": self.mode,
                "n_trades": 0,
                "total_pnl": 0.0,
            }

        settled_trades = [t for t in self.trade_log if t.settled]
        pnls = [t.pnl for t in settled_trades]

        return {
            "city_code": self.city_code,
            "mode": self.mode,
            "n_trades": len(self.trade_log),
            "n_settled": len(settled_trades),
            "total_pnl": sum(pnls) if pnls else 0.0,
            "win_rate": (
                sum(1 for p in pnls if p > 0) / len(pnls)
                if pnls else 0.0
            ),
            "avg_ev": float(np.mean([t.ev for t in self.trade_log])),
            "kill_switch_active": self.kill_switch.is_active,
        }


# ---------------------------------------------------------------------------
# Multi-city orchestrator
# ---------------------------------------------------------------------------

class MultiCityTradingOrchestrator:
    """Orchestrates trading across multiple cities.

    Manages independent LiveTradingHarness instances for each city,
    with independent kill switches and audit logs.

    Parameters
    ----------
    city_codes : list of str
        City identifiers to trade.
    mode : str
        Trading mode ("paper" or "live").
    strategies : dict, optional
        City-specific strategies. Keys are city codes, values are
        TradingStrategy instances.

    Examples
    --------
    >>> orch = MultiCityTradingOrchestrator(["chi", "phl"], mode="paper")
    >>> orch.get_status()
    """

    def __init__(
        self,
        city_codes: List[str],
        mode: str = "paper",
        strategies: Optional[Dict[str, TradingStrategy]] = None,
    ):
        self.city_codes = [c.strip().lower() for c in city_codes]
        self.mode = mode
        self.harnesses: Dict[str, LiveTradingHarness] = {}

        for code in self.city_codes:
            strategy = (strategies or {}).get(code, None)
            self.harnesses[code] = LiveTradingHarness(
                city_code=code,
                mode=mode,
                strategy=strategy,
            )

        logger.info(
            "MultiCityTradingOrchestrator: %d cities, mode=%s",
            len(self.city_codes), mode,
        )

    def get_harness(self, city_code: str) -> LiveTradingHarness:
        """Get the trading harness for a specific city.

        Parameters
        ----------
        city_code : str
            City identifier.

        Returns
        -------
        LiveTradingHarness
            The trading harness for the specified city.
        """
        code = city_code.strip().lower()
        if code not in self.harnesses:
            raise ValueError(
                f"No harness for city '{city_code}'. "
                f"Available: {list(self.harnesses.keys())}"
            )
        return self.harnesses[code]

    def get_status(self) -> Dict[str, Any]:
        """Get status of all city trading harnesses.

        Returns
        -------
        dict
            Summary of each city's trading status.
        """
        return {
            code: harness.get_summary()
            for code, harness in self.harnesses.items()
        }

    def activate_kill_switch(self, city_code: str, reason: str) -> None:
        """Activate kill switch for a specific city.

        Parameters
        ----------
        city_code : str
            City identifier.
        reason : str
            Reason for activation.
        """
        harness = self.get_harness(city_code)
        harness.kill_switch.activate(reason)

    def deactivate_kill_switch(self, city_code: str) -> None:
        """Deactivate kill switch for a specific city.

        Parameters
        ----------
        city_code : str
            City identifier.
        """
        harness = self.get_harness(city_code)
        harness.kill_switch.deactivate()

    def reset_daily(self) -> None:
        """Reset daily counters for all cities."""
        for harness in self.harnesses.values():
            harness.kill_switch.reset_daily()
