"""
Trading Strategy and Backtesting Framework for Kalshi KXHIGHNY Markets.

Provides tools for evaluating trading opportunities on Kalshi temperature
prediction markets using model-derived bucket probabilities:

  1. Expected Value (EV) computation for YES/NO binary contracts
  2. Kelly Criterion position sizing (full, fractional, capped)
  3. Configurable TradingStrategy with multiple sizing methods
  4. BacktestEngine for historical strategy simulation
  5. Comprehensive strategy grid search over parameter permutations
  6. Synthetic market data generator for testing without live API
  7. Report and visualization generation for backtest results

The framework is designed to work with the Gaussian (mu, sigma) predictions
from the synthesis model and the Kalshi bucket probability mapping from
the calibration module.
"""

import os
import sys
import json
import logging
import itertools
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

# Use non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.seasons import SEASON_MAP as SHARED_SEASON_MAP, SEASON_ORDER as SHARED_SEASON_ORDER
from src.utils import to_numpy as _shared_to_numpy

# Apply a clean plot style, with graceful fallback
_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meteorological season mapping (mirrors src/evaluate.py)
# ---------------------------------------------------------------------------
SEASON_MAP = SHARED_SEASON_MAP
SEASON_ORDER = SHARED_SEASON_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_SIZING_METHODS = [
    "fixed",
    "proportional",
    "full_kelly",
    "fractional_kelly",
    "capped_kelly",
]

# Conservative fee and slippage assumptions for EV threshold refitting
CONSERVATIVE_FEE_RATE = 0.07
SLIPPAGE_BPS = 0.02  # 2% slippage on execution price
CONSERVATIVE_FEE_TOTAL = CONSERVATIVE_FEE_RATE + SLIPPAGE_BPS  # 9% total cost

# Conservative EV thresholds (Phase D)
CONSERVATIVE_EV_THRESHOLD = 0.03  # Minimum EV after conservative costs
CONSERVATIVE_MIN_EV = 0.015  # Absolute minimum EV

# ---------------------------------------------------------------------------
# Kalshi trading fee model
#
# Kalshi's published general trading fee is a *curved* per-contract fee:
#
#     fee_per_trade = ceil( 0.07 * C * P * (1 - P) )   (rounded up to the cent)
#
# i.e. ~$0.0175/contract at P=0.50, falling toward zero at the price
# extremes. The fee is charged on the *traded* contracts at execution time
# (on entry, regardless of how the contract later settles).
#
# Earlier cost code in this repo approximated fees as a flat 7% of the $1
# payout (``payout = C * (1 - 0.07)``), i.e. ~$0.07/contract — 4-5x Kalshi's
# real fee at typical prices. That over-charge is large relative to the thin
# edge available against a sharp pre-settlement market and made genuinely
# +EV trades look unprofitable. ``kalshi_fee_per_contract`` is the accurate
# model; use it (with on-entry accounting) for realistic backtests.
# ---------------------------------------------------------------------------
KALSHI_FEE_COEFF = 0.07


def kalshi_fee_per_contract(price: float, coeff: float = KALSHI_FEE_COEFF) -> float:
    """Kalshi per-contract trading fee at a given execution price.

    Returns ``ceil(coeff * price * (1 - price) * 100) / 100`` — the documented
    Kalshi general-markets fee, rounded up to the next cent, per contract.

    Parameters
    ----------
    price : float
        Execution price of the contract being bought (0..1).
    coeff : float
        Fee coefficient (0.07 for Kalshi general markets).

    Returns
    -------
    float
        Per-contract fee in dollars (>= 0).
    """
    import math

    p = float(np.clip(price, 0.0, 1.0))
    return math.ceil(coeff * p * (1.0 - p) * 100.0) / 100.0


# ===========================================================================
# Helpers
# ===========================================================================

def _to_numpy(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array.

    Parameters
    ----------
    arr : array-like
        Input data (numpy array, pandas Series, or list).

    Returns
    -------
    np.ndarray
        1-D float64 array.
    """
    return _shared_to_numpy(arr)


# ===========================================================================
# 1. Expected Value Computation
# ===========================================================================

def compute_ev_yes(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
) -> float:
    """Compute expected value of buying a YES contract.

    EV_yes = model_prob * (1 - fee_rate) - market_price

    The payout on a winning YES contract is $1 minus fees. The cost is
    the market price. The expected value is the probability-weighted
    payout minus the cost.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).

    Returns
    -------
    float
        Expected value of buying YES. Positive means +EV.
    """
    if np.isnan(model_prob) or np.isnan(market_price):
        return float("nan")

    model_prob = float(np.clip(model_prob, 0.0, 1.0))
    market_price = float(np.clip(market_price, 0.0, 1.0))

    return model_prob * (1.0 - fee_rate) - market_price


def compute_ev_no(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
) -> float:
    """Compute expected value of buying a NO contract.

    EV_no = (1 - model_prob) * (1 - fee_rate) - (1 - market_price)

    The payout on a winning NO contract is $1 minus fees. The cost is
    (1 - market_price). The expected value is the probability-weighted
    payout minus the cost.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).

    Returns
    -------
    float
        Expected value of buying NO. Positive means +EV.
    """
    if np.isnan(model_prob) or np.isnan(market_price):
        return float("nan")

    model_prob = float(np.clip(model_prob, 0.0, 1.0))
    market_price = float(np.clip(market_price, 0.0, 1.0))

    return (1.0 - model_prob) * (1.0 - fee_rate) - (1.0 - market_price)


def compute_ev_best(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
) -> dict:
    """Compute the best trade direction (YES or NO) and its EV.

    Compares the EV of buying YES vs buying NO and returns whichever
    has the higher expected value.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).

    Returns
    -------
    dict
        Dictionary with keys:
        - "direction": "YES" or "NO"
        - "ev": float, expected value of the best direction
        - "ev_yes": float, EV of buying YES
        - "ev_no": float, EV of buying NO
    """
    ev_yes = compute_ev_yes(model_prob, market_price, fee_rate)
    ev_no = compute_ev_no(model_prob, market_price, fee_rate)

    if np.isnan(ev_yes) or np.isnan(ev_no):
        return {
            "direction": "NONE",
            "ev": float("nan"),
            "ev_yes": ev_yes,
            "ev_no": ev_no,
        }

    if ev_yes >= ev_no:
        return {
            "direction": "YES",
            "ev": ev_yes,
            "ev_yes": ev_yes,
            "ev_no": ev_no,
        }
    else:
        return {
            "direction": "NO",
            "ev": ev_no,
            "ev_yes": ev_yes,
            "ev_no": ev_no,
        }


# ===========================================================================
# 2. Kelly Criterion Sizing
# ===========================================================================

def kelly_fraction(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
) -> float:
    """Compute the full Kelly fraction for optimal bet sizing.

    For a YES bet:
      f* = (p * (1-fee) / price) - ((1-p) / (1-price))
      where p = model_prob, price = market_price

    For a NO bet:
      f* = ((1-p) * (1-fee) / (1-price)) - (p / price)

    Returns the Kelly fraction for whichever direction (YES or NO)
    yields positive Kelly. Returns 0 if neither direction is +EV.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).

    Returns
    -------
    float
        Kelly fraction (>= 0). Represents the optimal fraction of
        bankroll to wager.
    """
    if np.isnan(model_prob) or np.isnan(market_price):
        return 0.0

    model_prob = float(np.clip(model_prob, 0.0, 1.0))
    market_price = float(np.clip(market_price, 1e-10, 1.0 - 1e-10))

    p = model_prob
    q = 1.0 - p
    price = market_price

    # Kelly for YES: (p * payout / cost) - (q / loss)
    # Payout per dollar risked for YES: (1 - fee) / price - 1
    # Rewritten: f_yes = p * (1-fee) / price - q / (1-price)
    # But we want the fraction of bankroll, which is f / price for YES
    # Simplified Kelly for binary: f = (p * b - q) / b
    # where b = net odds = (1 - fee - price) / price for YES
    net_payout_yes = (1.0 - fee_rate) - price
    if price > 0:
        b_yes = net_payout_yes / price
    else:
        b_yes = 0.0

    if b_yes > 0:
        kelly_yes = (p * b_yes - q) / b_yes
    else:
        kelly_yes = -1.0  # Not viable

    # Kelly for NO
    no_price = 1.0 - price
    net_payout_no = (1.0 - fee_rate) - no_price
    if no_price > 0:
        b_no = net_payout_no / no_price
    else:
        b_no = 0.0

    if b_no > 0:
        kelly_no = (q * b_no - p) / b_no
    else:
        kelly_no = -1.0

    best_kelly = max(kelly_yes, kelly_no, 0.0)
    return float(best_kelly)


def fractional_kelly(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
    fraction: float = 0.25,
) -> float:
    """Compute a fractional Kelly bet size.

    Reduces the full Kelly fraction by the given multiplier to reduce
    variance and risk of ruin.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).
    fraction : float
        Multiplier on full Kelly (default 0.25 = quarter Kelly).

    Returns
    -------
    float
        Fractional Kelly fraction (>= 0).
    """
    return kelly_fraction(model_prob, market_price, fee_rate) * fraction


def capped_kelly(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
    max_fraction: float = 0.10,
) -> float:
    """Compute Kelly fraction with an upper cap.

    Uses full Kelly but enforces a maximum position size as a fraction
    of bankroll. Useful for limiting exposure on any single trade.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).
    max_fraction : float
        Maximum fraction of bankroll (default 0.10 = 10%).

    Returns
    -------
    float
        Capped Kelly fraction (>= 0, <= max_fraction).
    """
    return min(kelly_fraction(model_prob, market_price, fee_rate), max_fraction)


def position_size(
    kelly_frac: float,
    bankroll: float,
    contract_price: float = 1.0,
    min_size: int = 1,
    max_size: int = 100,
) -> int:
    """Convert a Kelly fraction to a discrete number of contracts.

    Parameters
    ----------
    kelly_frac : float
        Kelly fraction (0 to 1).
    bankroll : float
        Current bankroll in dollars.
    contract_price : float
        Price per contract in dollars (default 1.0).
    min_size : int
        Minimum number of contracts (default 1).
    max_size : int
        Maximum number of contracts (default 100).

    Returns
    -------
    int
        Number of contracts to trade. Returns 0 if kelly_frac <= 0.
    """
    if kelly_frac <= 0 or bankroll <= 0 or contract_price <= 0:
        return 0

    dollar_amount = kelly_frac * bankroll
    n_contracts = int(dollar_amount / contract_price)

    if n_contracts < min_size:
        return 0

    return min(n_contracts, max_size)


# ===========================================================================
# 3. Trade Signal
# ===========================================================================

@dataclass
class TradeSignal:
    """A single trade signal generated by a strategy.

    Attributes
    ----------
    direction : str
        "YES", "NO", or "NONE".
    size : int
        Number of contracts.
    ev : float
        Expected value of the trade.
    confidence : float
        Model's absolute deviation from 0.5 (0 to 0.5).
    model_prob : float
        Model's probability estimate.
    market_price : float
        Market price at time of signal.
    kelly_frac : float
        Kelly fraction used for sizing.
    """

    direction: str = "NONE"
    size: int = 0
    ev: float = 0.0
    confidence: float = 0.0
    model_prob: float = 0.5
    market_price: float = 0.5
    kelly_frac: float = 0.0


# ===========================================================================
# 4. TradingStrategy
# ===========================================================================

class TradingStrategy:
    """Configurable trading strategy for Kalshi market evaluation.

    Combines EV filtering, Kelly sizing, and position limits into a
    single reusable strategy object.

    Parameters
    ----------
    name : str
        Descriptive name for the strategy.
    ev_threshold : float
        Minimum EV required to trigger a trade (default 0.02).
    sizing_method : str
        One of "fixed", "proportional", "full_kelly",
        "fractional_kelly", "capped_kelly". Default "fractional_kelly".
    kelly_fraction : float
        Fraction multiplier for fractional Kelly (default 0.25).
    max_position_frac : float
        Maximum bankroll fraction per trade (default 0.10).
    fee_rate : float
        Fee rate on winnings (default 0.07).
    min_ev : float
        Absolute minimum EV below which no trade is made (default 0.01).
    max_contracts : int
        Maximum contracts per trade (default 50).
    bankroll : float
        Initial bankroll in dollars (default 10000).
    fixed_size : int
        Number of contracts for fixed sizing (default 5).

    Examples
    --------
    >>> strategy = TradingStrategy("Conservative", ev_threshold=0.05)
    >>> signal = strategy.evaluate_trade(model_prob=0.7, market_price=0.5)
    >>> signal.direction
    'YES'
    """

    def __init__(
        self,
        name: str = "Default",
        ev_threshold: float = 0.02,
        sizing_method: str = "fractional_kelly",
        kelly_fraction: float = 0.25,
        max_position_frac: float = 0.10,
        fee_rate: float = 0.07,
        min_ev: float = 0.01,
        max_contracts: int = 50,
        bankroll: float = 10000.0,
        fixed_size: int = 5,
    ):
        if sizing_method not in VALID_SIZING_METHODS:
            raise ValueError(
                f"sizing_method must be one of {VALID_SIZING_METHODS}, "
                f"got '{sizing_method}'"
            )

        self.name = name
        self.ev_threshold = ev_threshold
        self.sizing_method = sizing_method
        self.kelly_fraction_param = kelly_fraction
        self.max_position_frac = max_position_frac
        self.fee_rate = fee_rate
        self.min_ev = min_ev
        self.max_contracts = max_contracts
        self.bankroll = bankroll
        self.fixed_size = fixed_size

    def __repr__(self) -> str:
        return (
            f"TradingStrategy(name='{self.name}', ev_thresh={self.ev_threshold}, "
            f"sizing='{self.sizing_method}', kelly_f={self.kelly_fraction_param}, "
            f"fee={self.fee_rate}, bankroll={self.bankroll})"
        )

    def evaluate_trade(
        self,
        model_prob: float,
        market_price: float,
    ) -> TradeSignal:
        """Evaluate whether to trade and generate a signal.

        Parameters
        ----------
        model_prob : float
            Model's probability estimate (0 to 1).
        market_price : float
            Current market price (0 to 1).

        Returns
        -------
        TradeSignal
            Trade signal with direction, size, EV, and confidence.
        """
        best = compute_ev_best(model_prob, market_price, self.fee_rate)

        if np.isnan(best["ev"]):
            return TradeSignal(
                direction="NONE",
                model_prob=model_prob,
                market_price=market_price,
            )

        ev = best["ev"]
        direction = best["direction"]
        confidence = abs(model_prob - 0.5)

        # Check EV thresholds
        if ev < self.ev_threshold or ev < self.min_ev:
            return TradeSignal(
                direction="NONE",
                ev=ev,
                confidence=confidence,
                model_prob=model_prob,
                market_price=market_price,
            )

        # Compute Kelly fraction
        kf = kelly_fraction(model_prob, market_price, self.fee_rate)

        # Compute position size based on sizing method
        if self.sizing_method == "fixed":
            size = self.fixed_size
        elif self.sizing_method == "proportional":
            # Size proportional to EV
            prop_frac = min(ev * 2.0, self.max_position_frac)
            size = position_size(
                prop_frac, self.bankroll,
                min_size=1, max_size=self.max_contracts,
            )
        elif self.sizing_method == "full_kelly":
            frac = min(kf, self.max_position_frac)
            size = position_size(
                frac, self.bankroll,
                min_size=1, max_size=self.max_contracts,
            )
        elif self.sizing_method == "fractional_kelly":
            frac = min(kf * self.kelly_fraction_param, self.max_position_frac)
            size = position_size(
                frac, self.bankroll,
                min_size=1, max_size=self.max_contracts,
            )
        elif self.sizing_method == "capped_kelly":
            frac = min(kf, self.max_position_frac)
            size = position_size(
                frac, self.bankroll,
                min_size=1, max_size=self.max_contracts,
            )
        else:
            size = 0

        if size == 0:
            return TradeSignal(
                direction="NONE",
                ev=ev,
                confidence=confidence,
                model_prob=model_prob,
                market_price=market_price,
                kelly_frac=kf,
            )

        return TradeSignal(
            direction=direction,
            size=size,
            ev=ev,
            confidence=confidence,
            model_prob=model_prob,
            market_price=market_price,
            kelly_frac=kf,
        )

    def should_trade(
        self,
        model_prob: float,
        market_price: float,
    ) -> bool:
        """Check whether a trade meets strategy criteria.

        Parameters
        ----------
        model_prob : float
            Model's probability estimate (0 to 1).
        market_price : float
            Current market price (0 to 1).

        Returns
        -------
        bool
            True if trade meets EV threshold and size > 0.
        """
        signal = self.evaluate_trade(model_prob, market_price)
        return signal.direction != "NONE" and signal.size > 0


# ===========================================================================
# 5. BacktestResult
# ===========================================================================

@dataclass
class BacktestResult:
    """Container for backtest results.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy.
    trades : list[dict]
        List of executed trades with full details.
    daily_pnl : np.ndarray
        Daily P&L series.
    cumulative_pnl : np.ndarray
        Cumulative P&L series.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    max_drawdown : float
        Maximum peak-to-trough drawdown.
    win_rate : float
        Fraction of profitable trades.
    total_pnl : float
        Final cumulative P&L.
    roi : float
        Return on initial bankroll.
    avg_ev : float
        Average EV of trades taken.
    n_trades : int
        Number of trades executed.
    n_days : int
        Number of trading days in the backtest.
    """

    strategy_name: str = ""
    trades: list = field(default_factory=list)
    daily_pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    avg_ev: float = 0.0
    n_trades: int = 0
    n_days: int = 0

    def to_summary_dict(self) -> dict:
        """Convert to a flat dictionary for CSV/DataFrame export.

        Returns
        -------
        dict
            Summary metrics (excludes trade list and arrays).
        """
        return {
            "strategy_name": self.strategy_name,
            "n_trades": self.n_trades,
            "n_days": self.n_days,
            "total_pnl": self.total_pnl,
            "roi": self.roi,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_ev": self.avg_ev,
        }


# ===========================================================================
# 6. BacktestEngine
# ===========================================================================

class BacktestEngine:
    """Historical backtesting engine for trading strategies.

    Simulates trading on historical data where each row represents a
    day with model predictions, market prices, and actual outcomes.

    Parameters
    ----------
    strategy : TradingStrategy
        The strategy to backtest.

    Examples
    --------
    >>> strategy = TradingStrategy("Test", ev_threshold=0.02)
    >>> engine = BacktestEngine(strategy)
    >>> result = engine.run_backtest(historical_data)
    """

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def run_backtest(self, historical_data: pd.DataFrame) -> BacktestResult:
        """Run a backtest over historical data.

        Parameters
        ----------
        historical_data : pd.DataFrame
            Must contain columns:
            - date: trade date
            - model_prob: model's probability for the bucket
            - market_price: market price for the bucket
            - actual_outcome: 1 if event occurred, 0 otherwise

        Returns
        -------
        BacktestResult
            Comprehensive backtest results.
        """
        required_cols = ["date", "model_prob", "market_price", "actual_outcome"]
        missing = [c for c in required_cols if c not in historical_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = historical_data.sort_values("date").reset_index(drop=True)
        n_days = len(df)

        trades = []
        daily_pnl = np.zeros(n_days)
        bankroll = self.strategy.bankroll

        for i, row in df.iterrows():
            model_prob = row["model_prob"]
            market_price = row["market_price"]
            actual_outcome = row["actual_outcome"]

            signal = self.strategy.evaluate_trade(model_prob, market_price)

            if signal.direction == "NONE" or signal.size == 0:
                daily_pnl[i] = 0.0
                continue

            # Compute P&L for this trade
            if signal.direction == "YES":
                cost = signal.size * market_price
                if actual_outcome == 1:
                    revenue = signal.size * (1.0 - self.strategy.fee_rate)
                    pnl = revenue - cost
                else:
                    pnl = -cost
            else:  # NO
                cost = signal.size * (1.0 - market_price)
                if actual_outcome == 0:
                    revenue = signal.size * (1.0 - self.strategy.fee_rate)
                    pnl = revenue - cost
                else:
                    pnl = -cost

            daily_pnl[i] = pnl
            bankroll += pnl

            trade_record = {
                "date": row["date"],
                "direction": signal.direction,
                "size": signal.size,
                "model_prob": model_prob,
                "market_price": market_price,
                "actual_outcome": actual_outcome,
                "ev": signal.ev,
                "pnl": pnl,
                "bankroll_after": bankroll,
            }
            trades.append(trade_record)

        # Compute summary metrics
        cumulative_pnl = np.cumsum(daily_pnl)
        total_pnl = float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0
        roi = total_pnl / self.strategy.bankroll if self.strategy.bankroll > 0 else 0.0

        # Sharpe ratio (annualized, using days with trades)
        trade_pnls = [t["pnl"] for t in trades]
        if len(trade_pnls) > 1:
            mean_pnl = np.mean(trade_pnls)
            std_pnl = np.std(trade_pnls, ddof=1)
            if std_pnl > 0:
                sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252)
            else:
                sharpe_ratio = float("inf") if mean_pnl > 0 else 0.0
        elif len(trade_pnls) == 1:
            sharpe_ratio = float("inf") if trade_pnls[0] > 0 else (
                float("-inf") if trade_pnls[0] < 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        max_drawdown = _compute_max_drawdown(cumulative_pnl)

        # Win rate
        if trades:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            win_rate = wins / len(trades)
        else:
            win_rate = 0.0

        # Average EV
        if trades:
            avg_ev = float(np.mean([t["ev"] for t in trades]))
        else:
            avg_ev = 0.0

        result = BacktestResult(
            strategy_name=self.strategy.name,
            trades=trades,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_pnl=total_pnl,
            roi=roi,
            avg_ev=avg_ev,
            n_trades=len(trades),
            n_days=n_days,
        )

        logger.info(
            "Backtest '%s': %d trades, PnL=%.2f, ROI=%.2f%%, "
            "Sharpe=%.2f, MaxDD=%.2f, WinRate=%.1f%%",
            self.strategy.name, len(trades), total_pnl,
            roi * 100, sharpe_ratio, max_drawdown, win_rate * 100,
        )

        return result

    def run_multi_strategy_backtest(
        self,
        historical_data: pd.DataFrame,
        strategies: list,
    ) -> list:
        """Run multiple strategies on the same data and compare.

        Parameters
        ----------
        historical_data : pd.DataFrame
            Historical market data (see run_backtest for format).
        strategies : list[TradingStrategy]
            List of strategies to backtest.

        Returns
        -------
        list[BacktestResult]
            Results for each strategy, sorted by total P&L descending.
        """
        results = []
        for strategy in strategies:
            engine = BacktestEngine(strategy)
            result = engine.run_backtest(historical_data)
            results.append(result)

        results.sort(key=lambda r: r.total_pnl, reverse=True)
        return results


def compute_drawdown_metrics(
    bankroll_series,
    initial_bankroll: float,
) -> dict:
    """Compute max drawdown in dollars and as a percent of peak bankroll.

    The peak is floored at ``initial_bankroll`` so the percentage stays in
    [-100, 0] even for legacy bankroll series that dipped below zero
    (before stake-capping, backtests could "bet" with negative bankroll,
    producing nonsense values like -134%).

    Parameters
    ----------
    bankroll_series : array-like or pd.Series
        Bankroll value over time (chronological order).
    initial_bankroll : float
        Starting bankroll in dollars (must be > 0).

    Returns
    -------
    dict
        Keys: max_drawdown (dollars, <= 0), max_drawdown_pct (in [-100, 0]).
    """
    values = np.asarray(bankroll_series, dtype=float)
    if values.size == 0 or initial_bankroll <= 0:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}

    peak = np.maximum.accumulate(values)
    peak = np.clip(peak, initial_bankroll, None)
    drawdown = values - peak

    max_dd = float(drawdown.min())
    max_dd_pct = float(max(-100.0, (drawdown / peak).min() * 100.0))
    return {"max_drawdown": max_dd, "max_drawdown_pct": max_dd_pct}


def _compute_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown.

    Parameters
    ----------
    cumulative_pnl : np.ndarray
        Cumulative P&L series.

    Returns
    -------
    float
        Maximum drawdown (non-negative value). Returns 0 if the series
        is empty or monotonically increasing.
    """
    if len(cumulative_pnl) == 0:
        return 0.0

    peak = cumulative_pnl[0]
    max_dd = 0.0

    for val in cumulative_pnl:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    return float(max_dd)


# ===========================================================================
# 7. Strategy Grid Search
# ===========================================================================

def generate_strategy_grid(
    ev_thresholds: Optional[list] = None,
    sizing_methods: Optional[list] = None,
    kelly_fractions: Optional[list] = None,
    fee_rates: Optional[list] = None,
    max_positions: Optional[list] = None,
    bankrolls: Optional[list] = None,
) -> list:
    """Generate a comprehensive grid of strategy configurations.

    Parameters
    ----------
    ev_thresholds : list[float], optional
        EV threshold values. Defaults to [0.01, 0.02, 0.03, 0.05, 0.08, 0.10].
    sizing_methods : list[str], optional
        Sizing methods. Defaults to all valid methods.
    kelly_fractions : list[float], optional
        Kelly fractions. Defaults to [0.10, 0.15, 0.20, 0.25, 0.50].
    fee_rates : list[float], optional
        Fee rates. Defaults to [0.05, 0.07, 0.10].
    max_positions : list[float], optional
        Max position fractions. Defaults to [0.05, 0.10, 0.15, 0.20].
    bankrolls : list[float], optional
        Bankroll amounts. Defaults to [5000, 10000, 25000].

    Returns
    -------
    list[TradingStrategy]
        List of all strategy permutations.
    """
    if ev_thresholds is None:
        ev_thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    if sizing_methods is None:
        sizing_methods = list(VALID_SIZING_METHODS)
    if kelly_fractions is None:
        kelly_fractions = [0.10, 0.15, 0.20, 0.25, 0.50]
    if fee_rates is None:
        fee_rates = [0.05, 0.07, 0.10]
    if max_positions is None:
        max_positions = [0.05, 0.10, 0.15, 0.20]
    if bankrolls is None:
        bankrolls = [5000, 10000, 25000]

    strategies = []
    idx = 0

    for ev_t, sizing, kf, fee, max_pos, br in itertools.product(
        ev_thresholds, sizing_methods, kelly_fractions,
        fee_rates, max_positions, bankrolls,
    ):
        # Skip irrelevant kelly_fraction for non-kelly methods
        if sizing in ("fixed", "proportional") and kf != kelly_fractions[0]:
            continue

        name = (
            f"S{idx:04d}_ev{ev_t:.2f}_{sizing}_kf{kf:.2f}_"
            f"fee{fee:.2f}_mp{max_pos:.2f}_br{int(br)}"
        )
        strategy = TradingStrategy(
            name=name,
            ev_threshold=ev_t,
            sizing_method=sizing,
            kelly_fraction=kf,
            max_position_frac=max_pos,
            fee_rate=fee,
            max_contracts=50,
            bankroll=br,
        )
        strategies.append(strategy)
        idx += 1

    logger.info("Generated %d strategy permutations", len(strategies))
    return strategies


def run_comprehensive_backtest(
    historical_data: pd.DataFrame,
    output_dir: str = "results/kalshi_max_train_backtest",
    strategies: Optional[list] = None,
    max_strategies: int = 500,
) -> dict:
    """Run all strategy permutations and generate comprehensive results.

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical market data for backtesting.
    output_dir : str
        Directory to save results.
    strategies : list[TradingStrategy], optional
        Strategies to test. If None, generates the full grid.
    max_strategies : int
        Maximum number of strategies to evaluate (default 500).

    Returns
    -------
    dict
        Dictionary with keys:
        - "all_results": list of BacktestResult
        - "comparison_df": pd.DataFrame of all results
        - "best_by_sharpe": top results by Sharpe ratio
        - "best_by_roi": top results by ROI
        - "best_by_winrate": top results by win rate
    """
    os.makedirs(output_dir, exist_ok=True)

    if strategies is None:
        strategies = generate_strategy_grid()

    # Limit strategy count
    if len(strategies) > max_strategies:
        logger.info(
            "Limiting from %d to %d strategies", len(strategies), max_strategies
        )
        strategies = strategies[:max_strategies]

    logger.info("Running comprehensive backtest with %d strategies", len(strategies))

    all_results = []
    engine = BacktestEngine(strategies[0])  # Dummy, we'll use multi_strategy
    all_results = engine.run_multi_strategy_backtest(
        historical_data, strategies
    )

    # Build comparison DataFrame
    rows = [r.to_summary_dict() for r in all_results]
    comparison_df = pd.DataFrame(rows)

    # Save comparison CSV
    comparison_df.to_csv(
        os.path.join(output_dir, "strategy_comparison.csv"), index=False
    )

    # Best strategies by different metrics
    n_top = min(10, len(comparison_df))

    valid_sharpe = comparison_df[
        comparison_df["sharpe_ratio"].apply(lambda x: np.isfinite(x))
    ]
    best_by_sharpe = valid_sharpe.nlargest(n_top, "sharpe_ratio")
    best_by_sharpe.to_csv(
        os.path.join(output_dir, "best_strategies_sharpe.csv"), index=False
    )

    best_by_roi = comparison_df.nlargest(n_top, "roi")
    best_by_roi.to_csv(
        os.path.join(output_dir, "best_strategies_roi.csv"), index=False
    )

    best_by_winrate = comparison_df[
        comparison_df["n_trades"] > 0
    ].nlargest(n_top, "win_rate")
    best_by_winrate.to_csv(
        os.path.join(output_dir, "best_strategies_winrate.csv"), index=False
    )

    # Generate plots
    _plot_strategy_heatmap(comparison_df, output_dir)
    _plot_top_pnl_curves(all_results, output_dir, n_top=5)
    _plot_drawdown_analysis(all_results, output_dir, n_top=5)
    _plot_monthly_pnl(all_results, historical_data, output_dir, n_top=3)

    # Seasonal performance for top strategies
    _save_seasonal_performance(
        all_results, historical_data, output_dir, n_top=5
    )

    # Risk metrics
    _save_risk_metrics(all_results, output_dir, n_top=10)

    logger.info(
        "Comprehensive backtest complete: %d strategies evaluated, "
        "results saved to %s",
        len(all_results), output_dir,
    )

    return {
        "all_results": all_results,
        "comparison_df": comparison_df,
        "best_by_sharpe": best_by_sharpe,
        "best_by_roi": best_by_roi,
        "best_by_winrate": best_by_winrate,
    }


# ===========================================================================
# 8. Synthetic Market Data Generator
# ===========================================================================

def generate_synthetic_market_data(
    model_predictions_df: pd.DataFrame,
    n_buckets_per_day: int = 3,
    noise_std: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic Kalshi market data for backtesting.

    Creates synthetic market prices correlated with but not identical to
    model-implied probabilities, simulating real market inefficiency.

    Parameters
    ----------
    model_predictions_df : pd.DataFrame
        Must contain columns: date, model_mu, model_sigma, actual_tmax.
    n_buckets_per_day : int
        Number of bucket contracts per day (default 3).
    noise_std : float
        Standard deviation of noise added to model probabilities
        to simulate market inefficiency (default 0.08).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: trade date
        - bucket_label: bucket description (e.g. "65-69 F")
        - bucket_low: lower bound of bucket
        - bucket_high: upper bound of bucket
        - model_prob: model probability for the bucket
        - market_price: synthetic market price
        - actual_outcome: 1 if actual_tmax fell in bucket, 0 otherwise
        - bid_price: simulated bid (market_price - spread/2)
        - ask_price: simulated ask (market_price + spread/2)
        - volume: simulated volume proxy
    """
    required_cols = ["date", "model_mu", "model_sigma", "actual_tmax"]
    missing = [c for c in required_cols if c not in model_predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rng = np.random.RandomState(seed)
    df = model_predictions_df.copy()

    records = []
    for _, row in df.iterrows():
        mu = row["model_mu"]
        sigma = max(row["model_sigma"], 1e-6)
        actual = row["actual_tmax"]
        date = row["date"]

        # Generate bucket boundaries centered around mu
        center = int(round(mu / 5.0) * 5)
        bucket_starts = []
        for offset in range(-n_buckets_per_day // 2,
                            n_buckets_per_day // 2 + 1):
            bucket_starts.append(center + offset * 5)

        # Take exactly n_buckets_per_day
        bucket_starts = bucket_starts[:n_buckets_per_day]

        for b_start in bucket_starts:
            b_end = b_start + 4

            # Model probability: P(b_start - 0.5 <= X <= b_end + 0.5)
            p_upper = stats.norm.cdf((b_end + 0.5 - mu) / sigma)
            p_lower = stats.norm.cdf((b_start - 0.5 - mu) / sigma)
            model_prob = float(np.clip(p_upper - p_lower, 0.001, 0.999))

            # Market price = model prob + noise (market inefficiency)
            noise = rng.normal(0, noise_std)
            market_price = float(np.clip(model_prob + noise, 0.01, 0.99))

            # Actual outcome
            actual_outcome = 1 if (b_start - 0.5) <= actual <= (b_end + 0.5) else 0

            # Bid-ask spread (wider for less liquid buckets)
            spread = max(0.02, 0.05 * (1 - model_prob))
            bid_price = float(np.clip(market_price - spread / 2, 0.01, 0.99))
            ask_price = float(np.clip(market_price + spread / 2, 0.01, 0.99))

            # Volume proxy (higher near the money)
            volume = int(rng.poisson(lam=max(10, 100 * model_prob)))

            records.append({
                "date": date,
                "bucket_label": f"{b_start}-{b_end} F",
                "bucket_low": b_start,
                "bucket_high": b_end,
                "model_prob": model_prob,
                "market_price": market_price,
                "actual_outcome": actual_outcome,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "volume": volume,
            })

    result_df = pd.DataFrame(records)
    logger.info(
        "Generated synthetic market data: %d rows, %d days, %d buckets/day",
        len(result_df), len(df), n_buckets_per_day,
    )
    return result_df


# ===========================================================================
# 9. Visualization Helpers
# ===========================================================================

def _plot_strategy_heatmap(
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot a heatmap of EV threshold vs sizing method by Sharpe ratio.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Strategy comparison DataFrame.
    output_dir : str
        Directory to save the plot.
    """
    if comparison_df.empty:
        return

    # Parse strategy parameters from name
    try:
        df = comparison_df.copy()
        df["ev_threshold"] = df["strategy_name"].str.extract(r"ev(\d+\.\d+)").astype(float)
        df["sizing"] = df["strategy_name"].str.extract(r"_(\w+)_kf")

        if df["ev_threshold"].isna().all() or df["sizing"].isna().all():
            logger.warning("Could not parse strategy names for heatmap")
            return

        # Pivot for heatmap
        pivot = df.groupby(["ev_threshold", "sizing"])["sharpe_ratio"].mean()
        pivot = pivot.reset_index()
        pivot_table = pivot.pivot(
            index="ev_threshold", columns="sizing", values="sharpe_ratio"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            pivot_table.values, aspect="auto", cmap="RdYlGn",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot_table.index])
        ax.set_xlabel("Sizing Method")
        ax.set_ylabel("EV Threshold")
        ax.set_title("Strategy Heatmap: Mean Sharpe Ratio")
        fig.colorbar(im, ax=ax, label="Sharpe Ratio")

        fig.tight_layout()
        save_path = os.path.join(output_dir, "strategy_heatmap.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved strategy heatmap to %s", save_path)

    except Exception as e:
        logger.warning("Could not generate strategy heatmap: %s", e)


def _plot_top_pnl_curves(
    all_results: list,
    output_dir: str,
    n_top: int = 5,
) -> None:
    """Plot cumulative P&L curves for the top strategies.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All backtest results (assumed sorted by total_pnl descending).
    output_dir : str
        Directory to save the plot.
    n_top : int
        Number of top strategies to plot (default 5).
    """
    top_results = all_results[:n_top]
    if not top_results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in top_results:
        if len(result.cumulative_pnl) > 0:
            ax.plot(
                result.cumulative_pnl,
                label=f"{result.strategy_name} (PnL={result.total_pnl:.0f})",
                linewidth=1.2,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Top {n_top} Strategies: Cumulative P&L")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "pnl_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved P&L curves to %s", save_path)


def _plot_drawdown_analysis(
    all_results: list,
    output_dir: str,
    n_top: int = 5,
) -> None:
    """Plot drawdown analysis for top strategies.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All backtest results.
    output_dir : str
        Directory to save the plot.
    n_top : int
        Number of top strategies to plot.
    """
    top_results = all_results[:n_top]
    if not top_results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in top_results:
        cum_pnl = result.cumulative_pnl
        if len(cum_pnl) == 0:
            continue
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl
        ax.plot(
            drawdown,
            label=f"{result.strategy_name} (MaxDD={result.max_drawdown:.0f})",
            linewidth=1.0,
        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Drawdown ($)")
    ax.set_title(f"Top {n_top} Strategies: Drawdown Analysis")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "drawdown_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved drawdown analysis to %s", save_path)


def _plot_monthly_pnl(
    all_results: list,
    historical_data: pd.DataFrame,
    output_dir: str,
    n_top: int = 3,
) -> None:
    """Plot monthly P&L breakdown for top strategies.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All backtest results.
    historical_data : pd.DataFrame
        Historical data with date column.
    output_dir : str
        Directory to save the plot.
    n_top : int
        Number of top strategies to plot.
    """
    top_results = [r for r in all_results[:n_top] if r.trades]
    if not top_results:
        return

    fig, axes = plt.subplots(
        len(top_results), 1, figsize=(12, 4 * len(top_results)),
        squeeze=False,
    )

    for idx, result in enumerate(top_results):
        ax = axes[idx, 0]
        if not result.trades:
            continue

        trade_df = pd.DataFrame(result.trades)
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["month"] = trade_df["date"].dt.to_period("M")
        monthly = trade_df.groupby("month")["pnl"].sum()

        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
        ax.bar(range(len(monthly)), monthly.values, color=colors)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(
            [str(m) for m in monthly.index], rotation=45, ha="right",
            fontsize=8,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("P&L ($)")
        ax.set_title(f"{result.strategy_name}: Monthly P&L")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = os.path.join(output_dir, "monthly_pnl.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved monthly P&L to %s", save_path)


def _save_seasonal_performance(
    all_results: list,
    historical_data: pd.DataFrame,
    output_dir: str,
    n_top: int = 5,
) -> None:
    """Save seasonal performance breakdown for top strategies.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All backtest results.
    historical_data : pd.DataFrame
        Historical data with date column.
    output_dir : str
        Directory to save the CSV.
    n_top : int
        Number of top strategies to analyze.
    """
    top_results = [r for r in all_results[:n_top] if r.trades]
    if not top_results:
        return

    rows = []
    for result in top_results:
        trade_df = pd.DataFrame(result.trades)
        if trade_df.empty:
            continue

        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["month"] = trade_df["date"].dt.month
        trade_df["season"] = trade_df["month"].map(SEASON_MAP)

        for season in SEASON_ORDER:
            mask = trade_df["season"] == season
            if mask.sum() == 0:
                continue
            season_pnl = trade_df.loc[mask, "pnl"]
            rows.append({
                "strategy": result.strategy_name,
                "season": season,
                "n_trades": int(mask.sum()),
                "total_pnl": float(season_pnl.sum()),
                "mean_pnl": float(season_pnl.mean()),
                "win_rate": float((season_pnl > 0).mean()),
            })

    if rows:
        seasonal_df = pd.DataFrame(rows)
        save_path = os.path.join(output_dir, "seasonal_performance.csv")
        seasonal_df.to_csv(save_path, index=False)
        logger.info("Saved seasonal performance to %s", save_path)


def _save_risk_metrics(
    all_results: list,
    output_dir: str,
    n_top: int = 10,
) -> None:
    """Save risk metrics for top strategies.

    Parameters
    ----------
    all_results : list[BacktestResult]
        All backtest results.
    output_dir : str
        Directory to save the CSV.
    n_top : int
        Number of top strategies to analyze.
    """
    top_results = all_results[:n_top]
    if not top_results:
        return

    rows = []
    for result in top_results:
        trade_pnls = [t["pnl"] for t in result.trades]
        if not trade_pnls:
            continue

        pnl_arr = np.array(trade_pnls)

        # Value at Risk (5th percentile of P&L)
        var_5 = float(np.percentile(pnl_arr, 5)) if len(pnl_arr) > 0 else 0.0

        # Expected shortfall (mean of losses beyond VaR)
        losses_beyond_var = pnl_arr[pnl_arr <= var_5]
        es_5 = float(np.mean(losses_beyond_var)) if len(losses_beyond_var) > 0 else 0.0

        # Profit factor
        gross_profit = float(np.sum(pnl_arr[pnl_arr > 0]))
        gross_loss = float(np.abs(np.sum(pnl_arr[pnl_arr < 0])))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        rows.append({
            "strategy": result.strategy_name,
            "total_pnl": result.total_pnl,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "var_5pct": var_5,
            "expected_shortfall_5pct": es_5,
            "profit_factor": profit_factor,
            "max_win": float(np.max(pnl_arr)),
            "max_loss": float(np.min(pnl_arr)),
            "avg_win": float(np.mean(pnl_arr[pnl_arr > 0])) if (pnl_arr > 0).any() else 0.0,
            "avg_loss": float(np.mean(pnl_arr[pnl_arr < 0])) if (pnl_arr < 0).any() else 0.0,
        })

    if rows:
        risk_df = pd.DataFrame(rows)
        save_path = os.path.join(output_dir, "risk_metrics.csv")
        risk_df.to_csv(save_path, index=False)
        logger.info("Saved risk metrics to %s", save_path)


# ===========================================================================
# 10. Report Generation
# ===========================================================================

def generate_phase3_report(
    backtest_results: dict,
    output_dir: str = "results/kalshi_max_train_backtest",
) -> str:
    """Generate a comprehensive Phase 3 trading strategy report.

    Creates a text summary, machine-readable JSON metrics, and
    supporting data files.

    Parameters
    ----------
    backtest_results : dict
        Output from run_comprehensive_backtest().
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        The full report text.
    """
    os.makedirs(output_dir, exist_ok=True)

    comparison_df = backtest_results.get("comparison_df", pd.DataFrame())
    all_results = backtest_results.get("all_results", [])

    # Build text report
    lines = [
        "=" * 70,
        "NYC Temperature Prediction -- Phase 3: Trading Strategy Report",
        "=" * 70,
        "",
    ]

    # Overview
    lines.append(f"Total strategies evaluated: {len(all_results)}")
    trading_strategies = [r for r in all_results if r.n_trades > 0]
    lines.append(f"Strategies with trades: {len(trading_strategies)}")
    lines.append("")

    # Best strategies summary
    if not comparison_df.empty:
        lines.append("--- Top 5 by Total P&L ---")
        top_pnl = comparison_df.nlargest(5, "total_pnl")
        for _, row in top_pnl.iterrows():
            lines.append(
                f"  {row['strategy_name']}: PnL=${row['total_pnl']:.2f}, "
                f"ROI={row['roi']*100:.1f}%, Sharpe={row['sharpe_ratio']:.2f}, "
                f"WinRate={row['win_rate']*100:.0f}%, Trades={row['n_trades']}"
            )
        lines.append("")

        valid_sharpe = comparison_df[
            comparison_df["sharpe_ratio"].apply(lambda x: np.isfinite(x))
        ]
        if not valid_sharpe.empty:
            lines.append("--- Top 5 by Sharpe Ratio ---")
            top_sharpe = valid_sharpe.nlargest(5, "sharpe_ratio")
            for _, row in top_sharpe.iterrows():
                lines.append(
                    f"  {row['strategy_name']}: Sharpe={row['sharpe_ratio']:.2f}, "
                    f"PnL=${row['total_pnl']:.2f}, "
                    f"WinRate={row['win_rate']*100:.0f}%"
                )
            lines.append("")

        lines.append("--- Top 5 by Win Rate ---")
        top_wr = comparison_df[
            comparison_df["n_trades"] > 0
        ].nlargest(5, "win_rate")
        for _, row in top_wr.iterrows():
            lines.append(
                f"  {row['strategy_name']}: WinRate={row['win_rate']*100:.0f}%, "
                f"PnL=${row['total_pnl']:.2f}, Trades={row['n_trades']}"
            )
        lines.append("")

    # Aggregate statistics
    if trading_strategies:
        pnls = [r.total_pnl for r in trading_strategies]
        lines.append("--- Aggregate Statistics ---")
        lines.append(f"  Mean P&L: ${np.mean(pnls):.2f}")
        lines.append(f"  Median P&L: ${np.median(pnls):.2f}")
        lines.append(f"  Std P&L: ${np.std(pnls):.2f}")
        lines.append(f"  % Profitable strategies: "
                      f"{100 * sum(1 for p in pnls if p > 0) / len(pnls):.1f}%")
        lines.append("")

    lines.extend(["=" * 70, ""])

    report_text = "\n".join(lines)

    # Save text report
    report_path = os.path.join(output_dir, "phase3_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info("Saved Phase 3 report to %s", report_path)

    # Save JSON metrics
    metrics_dict = {
        "n_strategies": len(all_results),
        "n_with_trades": len(trading_strategies),
    }
    if not comparison_df.empty:
        top_row = comparison_df.nlargest(1, "total_pnl").iloc[0]
        metrics_dict["best_strategy"] = top_row["strategy_name"]
        metrics_dict["best_pnl"] = float(top_row["total_pnl"])
        metrics_dict["best_roi"] = float(top_row["roi"])
        metrics_dict["best_sharpe"] = float(top_row["sharpe_ratio"])

    metrics_path = os.path.join(output_dir, "phase3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Saved Phase 3 metrics to %s", metrics_path)

    return report_text


# ===========================================================================
# Phase D: Conservative EV, Kelly/Drawdown Validation, Strategy Factory
# ===========================================================================

def compute_conservative_ev(
    model_prob: float,
    market_price: float,
    fee_rate: float = 0.07,
    slippage: float = 0.02,
) -> dict:
    """Compute EV using conservative fee + slippage assumptions.

    Wraps :func:`compute_ev_best` but uses total cost (fee_rate + slippage)
    instead of fee_rate alone, giving a more conservative EV estimate that
    accounts for execution friction.

    Parameters
    ----------
    model_prob : float
        Model's estimated probability of the event occurring (0 to 1).
    market_price : float
        Current market price for the YES contract (0 to 1).
    fee_rate : float
        Fee rate on winnings (default 0.07 = 7%).
    slippage : float
        Estimated slippage as a fraction (default 0.02 = 2%).

    Returns
    -------
    dict
        Same format as :func:`compute_ev_best`:
        - "direction": "YES", "NO", or "NONE"
        - "ev": float, expected value of the best direction
        - "ev_yes": float, EV of buying YES
        - "ev_no": float, EV of buying NO
        - "total_cost": float, combined fee + slippage rate used
    """
    total_cost = fee_rate + slippage
    result = compute_ev_best(model_prob, market_price, fee_rate=total_cost)
    result["total_cost"] = total_cost
    return result


def validate_kelly_drawdown_policy(
    strategy: "TradingStrategy",
    max_acceptable_drawdown: float = 0.20,
    max_daily_var: float = 0.05,
) -> dict:
    """Validate that a TradingStrategy's Kelly/exposure settings are consistent
    with drawdown constraints.

    Parameters
    ----------
    strategy : TradingStrategy
        The trading strategy to validate.
    max_acceptable_drawdown : float
        Maximum acceptable single-trade loss as a fraction of bankroll
        (default 0.20 = 20%).
    max_daily_var : float
        Maximum acceptable daily value-at-risk as a fraction of bankroll
        (default 0.05 = 5%).

    Returns
    -------
    dict
        Validation result with keys:
        - "is_valid": bool, True if all checks pass.
        - "checks": list of dicts with "name", "passed", "detail".
        - "warnings": list of str, human-readable warnings.
        - "recommended_adjustments": dict, empty if valid.
    """
    checks = []
    warnings = []
    recommended = {}

    # Check 1: max_position_frac <= max_acceptable_drawdown
    check1_passed = strategy.max_position_frac <= max_acceptable_drawdown
    checks.append({
        "name": "max_position_vs_drawdown",
        "passed": check1_passed,
        "detail": (
            f"max_position_frac={strategy.max_position_frac:.4f} "
            f"{'<=' if check1_passed else '>'} "
            f"max_acceptable_drawdown={max_acceptable_drawdown:.4f}"
        ),
    })
    if not check1_passed:
        msg = (
            f"max_position_frac ({strategy.max_position_frac:.4f}) exceeds "
            f"max_acceptable_drawdown ({max_acceptable_drawdown:.4f}). "
            f"A single trade could lose more than the drawdown limit."
        )
        warnings.append(msg)
        logger.warning("Kelly/drawdown policy violation: %s", msg)
        recommended["max_position_frac"] = max_acceptable_drawdown

    # Check 2: kelly_fraction * max_position_frac implies bounded worst-case loss
    effective_max_kelly_bet = strategy.kelly_fraction_param * strategy.max_position_frac
    worst_case_loss_frac = effective_max_kelly_bet  # worst case: lose entire bet
    check2_passed = worst_case_loss_frac <= max_acceptable_drawdown
    checks.append({
        "name": "kelly_worst_case_loss",
        "passed": check2_passed,
        "detail": (
            f"kelly_fraction * max_position_frac = "
            f"{strategy.kelly_fraction_param:.4f} * {strategy.max_position_frac:.4f} "
            f"= {effective_max_kelly_bet:.4f} "
            f"{'<=' if check2_passed else '>'} "
            f"max_acceptable_drawdown={max_acceptable_drawdown:.4f}"
        ),
    })
    if not check2_passed:
        msg = (
            f"Effective max Kelly bet fraction ({effective_max_kelly_bet:.4f}) "
            f"exceeds max_acceptable_drawdown ({max_acceptable_drawdown:.4f}). "
            f"Worst-case loss exceeds drawdown limit."
        )
        warnings.append(msg)
        logger.warning("Kelly/drawdown policy violation: %s", msg)
        safe_kelly = max_acceptable_drawdown / max(strategy.max_position_frac, 1e-10)
        recommended["kelly_fraction"] = round(safe_kelly, 4)

    # Check 3: for fractional_kelly, effective max bet shouldn't exceed daily VaR
    if strategy.sizing_method in ("fractional_kelly", "capped_kelly"):
        effective_max_bet_dollars = (
            strategy.kelly_fraction_param
            * strategy.max_position_frac
            * strategy.bankroll
        )
        max_var_dollars = max_daily_var * strategy.bankroll
        check3_passed = effective_max_bet_dollars <= max_var_dollars
        checks.append({
            "name": "fractional_kelly_daily_var",
            "passed": check3_passed,
            "detail": (
                f"effective_max_bet=${effective_max_bet_dollars:.2f} "
                f"{'<=' if check3_passed else '>'} "
                f"max_daily_var=${max_var_dollars:.2f} "
                f"({max_daily_var:.1%} of bankroll=${strategy.bankroll:.2f})"
            ),
        })
        if not check3_passed:
            msg = (
                f"Effective max bet (${effective_max_bet_dollars:.2f}) exceeds "
                f"daily VaR limit (${max_var_dollars:.2f}). "
                f"Single trade exposure too high relative to daily risk budget."
            )
            warnings.append(msg)
            logger.warning("Kelly/drawdown policy violation: %s", msg)
            safe_kelly = max_var_dollars / max(
                strategy.max_position_frac * strategy.bankroll, 1e-10
            )
            recommended.setdefault("kelly_fraction", round(safe_kelly, 4))

    is_valid = all(c["passed"] for c in checks)

    return {
        "is_valid": is_valid,
        "checks": checks,
        "warnings": warnings,
        "recommended_adjustments": recommended,
    }


def create_conservative_strategy(
    name: str = "PhaseD_Conservative",
    bankroll: float = 10000.0,
) -> "TradingStrategy":
    """Create a TradingStrategy with conservative Phase D defaults.

    Uses conservative fee + slippage assumptions, tighter Kelly sizing,
    and lower position limits than the baseline strategy.

    Parameters
    ----------
    name : str
        Descriptive name for the strategy (default "PhaseD_Conservative").
    bankroll : float
        Initial bankroll in dollars (default 10000).

    Returns
    -------
    TradingStrategy
        A conservatively configured trading strategy.
    """
    return TradingStrategy(
        name=name,
        fee_rate=CONSERVATIVE_FEE_TOTAL,
        ev_threshold=CONSERVATIVE_EV_THRESHOLD,
        min_ev=CONSERVATIVE_MIN_EV,
        sizing_method="fractional_kelly",
        kelly_fraction=0.20,
        max_position_frac=0.08,
        max_contracts=25,
        bankroll=bankroll,
    )
