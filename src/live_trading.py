"""
Multi-City Live Trading Harness for Kalshi Temperature Markets.

Extends the NYC trading infrastructure to support CHI (KXHIGHCHI) and
PHL (KXHIGHPHIL) with:
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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.bucket_semantics import bucket_prob_from_edges
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
    "phl": "KXHIGHPHIL",
    "atl": "KXHIGHTATL",
    "aus": "KXHIGHAUS",
}


def load_city_strategy(
    city_code: str,
    path: Optional[str] = None,
) -> Optional[TradingStrategy]:
    """Build a TradingStrategy from ``results/<city>/strategy.json`` if present.

    This is the read-back of the Phase 3 real-price strategy refit
    (:mod:`src.strategy_selection`).  Returns ``None`` when no strategy file
    exists so callers can fall back to a default.

    Parameters
    ----------
    city_code : str
        City identifier.
    path : str, optional
        Override path to the strategy JSON.  Defaults to
        ``<results_dir>/strategy.json``.

    Returns
    -------
    TradingStrategy or None
    """
    code = city_code.strip().lower()
    if path is None:
        try:
            cfg = get_city_config(code)
        except ValueError:
            return None
        path = os.path.join(cfg.results_dir, "strategy.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read strategy file %s: %s", path, exc)
        return None

    cfg_block = data.get("strategy", {})
    return TradingStrategy(
        name=cfg_block.get("name", f"{code}_strategy"),
        ev_threshold=cfg_block.get("ev_threshold", 0.03),
        sizing_method=cfg_block.get("sizing_method", "fractional_kelly"),
        kelly_fraction=cfg_block.get("kelly_fraction", 0.20),
        max_position_frac=cfg_block.get("max_position_frac", 0.08),
        fee_rate=cfg_block.get("fee_rate", 0.07),
        min_ev=cfg_block.get("min_ev", 0.01),
        max_contracts=cfg_block.get("max_contracts", 50),
        bankroll=cfg_block.get("bankroll", 10000.0),
        fixed_size=cfg_block.get("fixed_size", 5),
    )


def get_kalshi_ticker(city_code: str) -> str:
    """Return the Kalshi ticker prefix for a given city code.

    Falls back to the registered ``CityConfig.kalshi_ticker`` for cities not in
    the local map, so every supported city can trade through the harness.

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
        If city_code is not recognized in the map or the city registry.
    """
    code = city_code.strip().lower()
    if code in KALSHI_TICKER_MAP:
        return KALSHI_TICKER_MAP[code]
    try:
        return get_city_config(code).kalshi_ticker
    except ValueError as exc:
        raise ValueError(
            f"Unknown city code '{city_code}'. "
            f"Available: {sorted(KALSHI_TICKER_MAP.keys())}"
        ) from exc


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
    # Settlement-rounding-aware probabilities (see src/bucket_semantics.py)
    probs = np.array([
        max(float(bucket_prob_from_edges(mu, sigma, lo, hi)), PROB_CLIP_MIN)
        for lo, hi in bucket_edges
    ])
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

        if strategy is None:
            strategy = load_city_strategy(self.city_code)
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


# ---------------------------------------------------------------------------
# Audit log validation (Phase D)
# ---------------------------------------------------------------------------

_AUDIT_REQUIRED_TOP_KEYS = {
    "city_code", "kalshi_ticker", "date", "mode", "n_trades",
    "total_pnl", "kill_switch", "strategy", "trades",
}

_TRADE_REQUIRED_KEYS = {
    "city_code", "date", "ticker", "bucket_label", "direction",
    "size", "model_prob", "market_price", "ev", "mode", "pnl", "settled",
}


def validate_audit_log(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single audit log dict for completeness and correctness.

    Checks that all required top-level and per-trade keys are present,
    and that trade field values fall within expected types and ranges.

    Parameters
    ----------
    audit : dict
        A single audit log dictionary (as produced by
        :meth:`LiveTradingHarness.save_audit_log`).

    Returns
    -------
    dict
        Validation result with keys:
        - "is_valid": bool
        - "missing_fields": list of str (missing top-level keys)
        - "invalid_trades": list of dicts with "trade_index" and "issues"
        - "warnings": list of str
    """
    missing_fields: List[str] = []
    invalid_trades: List[Dict[str, Any]] = []
    warnings_list: List[str] = []

    # Check top-level keys
    for key in _AUDIT_REQUIRED_TOP_KEYS:
        if key not in audit:
            missing_fields.append(key)

    # Validate trades
    trades = audit.get("trades", [])
    if not isinstance(trades, list):
        warnings_list.append(
            f"'trades' field is not a list (got {type(trades).__name__})"
        )
        trades = []

    for idx, trade in enumerate(trades):
        issues: List[str] = []

        # Check required trade keys
        for key in _TRADE_REQUIRED_KEYS:
            if key not in trade:
                issues.append(f"missing key '{key}'")

        # Validate data types and ranges for present keys
        direction = trade.get("direction")
        if direction is not None and direction not in ("YES", "NO"):
            issues.append(
                f"direction must be 'YES' or 'NO', got '{direction}'"
            )

        size = trade.get("size")
        if size is not None and (not isinstance(size, (int, float)) or size <= 0):
            issues.append(f"size must be > 0, got {size}")

        model_prob = trade.get("model_prob")
        if model_prob is not None:
            if not isinstance(model_prob, (int, float)):
                issues.append(
                    f"model_prob must be numeric, got {type(model_prob).__name__}"
                )
            elif not (0.0 <= model_prob <= 1.0):
                issues.append(f"model_prob must be in [0, 1], got {model_prob}")

        market_price = trade.get("market_price")
        if market_price is not None:
            if not isinstance(market_price, (int, float)):
                issues.append(
                    f"market_price must be numeric, got {type(market_price).__name__}"
                )
            elif not (0.0 <= market_price <= 1.0):
                issues.append(
                    f"market_price must be in [0, 1], got {market_price}"
                )

        mode = trade.get("mode")
        if mode is not None and mode not in ("paper", "live"):
            issues.append(f"mode must be 'paper' or 'live', got '{mode}'")

        if issues:
            invalid_trades.append({"trade_index": idx, "issues": issues})

    # Cross-check n_trades
    n_trades = audit.get("n_trades")
    if n_trades is not None and n_trades != len(audit.get("trades", [])):
        warnings_list.append(
            f"n_trades ({n_trades}) does not match len(trades) "
            f"({len(audit.get('trades', []))})"
        )

    is_valid = len(missing_fields) == 0 and len(invalid_trades) == 0

    return {
        "is_valid": is_valid,
        "missing_fields": missing_fields,
        "invalid_trades": invalid_trades,
        "warnings": warnings_list,
    }


def validate_audit_log_file(file_path: str) -> Dict[str, Any]:
    """Load and validate a JSON audit log file.

    Parameters
    ----------
    file_path : str
        Path to the JSON audit log file.

    Returns
    -------
    dict
        Validation result from :func:`validate_audit_log` plus:
        - "file_path": str, the input file path.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(file_path, "r") as f:
        audit = json.load(f)

    result = validate_audit_log(audit)
    result["file_path"] = file_path
    return result


def validate_run_audit_completeness(
    audit_dir: str,
    city_code: str,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Check all audit logs in a directory for a given date range.

    For each date in [start_date, end_date], checks whether an audit log
    file exists (pattern: ``trading_audit_{city}_{date}.json``) and
    validates each existing file.

    Parameters
    ----------
    audit_dir : str
        Directory containing audit log JSON files.
    city_code : str
        City identifier (e.g., "nyc", "chi").
    start_date : str
        Start date (YYYY-MM-DD), inclusive.
    end_date : str
        End date (YYYY-MM-DD), inclusive.

    Returns
    -------
    dict
        Completeness report with keys:
        - "total_expected": int, number of dates in range.
        - "total_found": int, number of audit files found.
        - "total_valid": int, number of valid audit files.
        - "missing_dates": list of str, dates with no audit file.
        - "invalid_logs": list of dicts with "date" and "issues".
        - "completeness_pct": float, percentage of dates with valid logs.
    """
    city_code = city_code.strip().lower()
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    dates: List[str] = []
    current = dt_start
    while current <= dt_end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    total_expected = len(dates)
    total_found = 0
    total_valid = 0
    missing_dates: List[str] = []
    invalid_logs: List[Dict[str, Any]] = []

    for date_str in dates:
        filename = f"trading_audit_{city_code}_{date_str}.json"
        filepath = os.path.join(audit_dir, filename)

        if not os.path.isfile(filepath):
            missing_dates.append(date_str)
            continue

        total_found += 1

        try:
            result = validate_audit_log_file(filepath)
        except (json.JSONDecodeError, OSError) as exc:
            invalid_logs.append({
                "date": date_str,
                "issues": [f"Failed to load/parse file: {exc}"],
            })
            continue

        if result["is_valid"]:
            total_valid += 1
        else:
            issues: List[str] = []
            if result["missing_fields"]:
                issues.append(
                    f"missing top-level fields: {result['missing_fields']}"
                )
            for inv in result["invalid_trades"]:
                issues.append(
                    f"trade #{inv['trade_index']}: {inv['issues']}"
                )
            invalid_logs.append({"date": date_str, "issues": issues})

    completeness_pct = (
        (total_valid / total_expected * 100.0) if total_expected > 0 else 0.0
    )

    return {
        "total_expected": total_expected,
        "total_found": total_found,
        "total_valid": total_valid,
        "missing_dates": missing_dates,
        "invalid_logs": invalid_logs,
        "completeness_pct": completeness_pct,
    }
