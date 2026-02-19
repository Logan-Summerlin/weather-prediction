"""
Kalshi KXHIGHNY Market Client for NYC Temperature Prediction.

Provides tools for integrating with Kalshi's prediction market data
for the KXHIGHNY ("Highest temperature in NYC today?") series:

  1. KalshiClient: HTTP client for the Kalshi public API with retry
     logic, rate limiting, and pagination support.
  2. Market data parsing: extract temperature thresholds, parse bucket
     structures, and resolve market outcomes.
  3. Model-vs-market comparison: compare Gaussian model probabilities
     to market-implied probabilities, compute Brier and log scores.
  4. Historical backtesting: align model predictions with resolved
     markets, compute accuracy metrics, and generate reports.
  5. Visualization: scatter plots, Brier score comparisons, EV
     distributions, and daily probability time series.

All API interactions are read-only (unauthenticated public endpoints).
The Kalshi API base URL is:
    https://api.elections.kalshi.com/trade-api/v2
"""

import os
import re
import sys
import json
import time
import logging
from typing import Optional, Union

import requests

from src.utils import to_numpy as _shared_to_numpy
import numpy as np
import pandas as pd
from scipy import stats

# Use non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
# Configuration Constants
# ---------------------------------------------------------------------------
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KXHIGHNY_SERIES = "KXHIGHNY"
DEFAULT_BUCKET_THRESHOLDS = list(range(0, 120, 5))  # 0, 5, 10, ..., 115
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
RATE_LIMIT_RPS = 10


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
# 1. KalshiClient — HTTP Client for Kalshi Public API
# ===========================================================================

class KalshiClient:
    """HTTP client for the Kalshi public market API.

    Provides methods for fetching series metadata, markets, orderbooks,
    and derived market-implied probabilities. Includes automatic retry
    with exponential backoff and simple rate limiting.

    Parameters
    ----------
    base_url : str
        Base URL for the Kalshi API. Defaults to the production
        trade-api/v2 endpoint.

    Examples
    --------
    >>> client = KalshiClient()
    >>> series = client.get_series("KXHIGHNY")
    >>> markets = client.get_markets("KXHIGHNY", status="open")
    """

    def __init__(self, base_url: str = KALSHI_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._last_request_time = 0.0
        self._min_interval = 1.0 / RATE_LIMIT_RPS

        logger.info("KalshiClient initialized with base_url=%s", self.base_url)

    def _rate_limit(self) -> None:
        """Enforce rate limiting by sleeping if necessary.

        Ensures at least ``1 / RATE_LIMIT_RPS`` seconds between
        consecutive requests.
        """
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _request(self, url: str, params: Optional[dict] = None) -> dict:
        """Make an HTTP GET request with retry logic and rate limiting.

        Parameters
        ----------
        url : str
            Full URL to request.
        params : dict, optional
            Query parameters for the request.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        ConnectionError
            If all retries are exhausted.
        ValueError
            If the response is not valid JSON or contains an error.
        """
        last_exception = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()

            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 404:
                    raise ValueError(
                        f"Resource not found (HTTP 404): {url}"
                    )
                if response.status_code >= 500:
                    raise ConnectionError(
                        f"Server error (HTTP {response.status_code}): {url}"
                    )

                response.raise_for_status()
                return response.json()

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    ConnectionError) as exc:
                last_exception = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Request failed after %d attempts: %s",
                        MAX_RETRIES, exc,
                    )

            except ValueError:
                raise

            except Exception as exc:
                last_exception = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Request failed after %d attempts: %s",
                        MAX_RETRIES, exc,
                    )

        raise ConnectionError(
            f"All {MAX_RETRIES} retries exhausted for {url}: {last_exception}"
        )

    def get_series(self, series_ticker: str = KXHIGHNY_SERIES) -> dict:
        """Fetch series metadata from the Kalshi API.

        Parameters
        ----------
        series_ticker : str
            The series ticker (e.g., "KXHIGHNY").

        Returns
        -------
        dict
            Series metadata including title, frequency, category, etc.
        """
        url = f"{self.base_url}/series/{series_ticker}"
        data = self._request(url)

        series = data.get("series", data)
        logger.info(
            "Fetched series '%s': %s",
            series_ticker,
            series.get("title", "unknown"),
        )
        return series

    def get_markets(
        self,
        series_ticker: str = KXHIGHNY_SERIES,
        status: str = "open",
        limit: int = 200,
    ) -> list[dict]:
        """Fetch markets for a series with pagination support.

        Parameters
        ----------
        series_ticker : str
            The series ticker to query.
        status : str
            Market status filter: "open", "closed", "settled", or
            "all". Default "open".
        limit : int
            Maximum number of markets per page (max 200).

        Returns
        -------
        list[dict]
            List of market dictionaries.
        """
        url = f"{self.base_url}/markets"
        all_markets = []
        cursor = None

        while True:
            params = {
                "series_ticker": series_ticker,
                "limit": min(limit, 200),
            }
            if status != "all":
                params["status"] = status
            if cursor is not None:
                params["cursor"] = cursor

            data = self._request(url, params=params)
            markets = data.get("markets", [])
            all_markets.extend(markets)

            # Check for next page
            cursor = data.get("cursor")
            if not cursor or len(markets) < limit:
                break

        logger.info(
            "Fetched %d markets for series '%s' (status=%s)",
            len(all_markets), series_ticker, status,
        )
        return all_markets

    def get_historical_markets(
        self,
        series_ticker: str = KXHIGHNY_SERIES,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch resolved/settled markets for historical backtesting.

        Parameters
        ----------
        series_ticker : str
            The series ticker.
        min_date : str, optional
            Minimum date filter (ISO format, e.g. "2024-01-01").
        max_date : str, optional
            Maximum date filter (ISO format, e.g. "2024-12-31").

        Returns
        -------
        list[dict]
            List of settled market dictionaries.
        """
        url = f"{self.base_url}/markets"
        all_markets = []
        cursor = None

        while True:
            params = {
                "series_ticker": series_ticker,
                "status": "settled",
                "limit": 200,
            }
            if min_date is not None:
                params["min_close_ts"] = min_date
            if max_date is not None:
                params["max_close_ts"] = max_date
            if cursor is not None:
                params["cursor"] = cursor

            data = self._request(url, params=params)
            markets = data.get("markets", [])
            all_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or len(markets) < 200:
                break

        logger.info(
            "Fetched %d historical markets for '%s' (%s to %s)",
            len(all_markets), series_ticker,
            min_date or "start", max_date or "end",
        )
        return all_markets

    def get_orderbook(self, market_ticker: str) -> dict:
        """Fetch the orderbook for a specific market.

        Parameters
        ----------
        market_ticker : str
            The market ticker (e.g., "KXHIGHNY-24DEC31-T85").

        Returns
        -------
        dict
            Orderbook data with "yes" and "no" arrays of
            [price, quantity] pairs.
        """
        url = f"{self.base_url}/markets/{market_ticker}/orderbook"
        data = self._request(url)

        orderbook = data.get("orderbook", data)
        n_yes = len(orderbook.get("yes", []))
        n_no = len(orderbook.get("no", []))
        logger.info(
            "Fetched orderbook for '%s': %d YES levels, %d NO levels",
            market_ticker, n_yes, n_no,
        )
        return orderbook

    def get_market_implied_probability(self, market_ticker: str) -> float:
        """Extract mid-price implied probability from the orderbook.

        Computes the midpoint between the best YES bid and (100 - best
        NO bid) as an estimate of the market-implied probability.

        Parameters
        ----------
        market_ticker : str
            The market ticker.

        Returns
        -------
        float
            Implied probability in [0, 1]. Returns NaN if the
            orderbook is empty or prices cannot be determined.
        """
        orderbook = self.get_orderbook(market_ticker)

        yes_bids = orderbook.get("yes", [])
        no_bids = orderbook.get("no", [])

        if not yes_bids and not no_bids:
            logger.warning(
                "Empty orderbook for '%s' — returning NaN",
                market_ticker,
            )
            return float("nan")

        # Best YES bid price (highest price someone will pay for YES)
        yes_price = yes_bids[0][0] if yes_bids else None
        # Best NO bid price (highest price someone will pay for NO)
        no_price = no_bids[0][0] if no_bids else None

        if yes_price is not None and no_price is not None:
            # Mid-price: average of YES bid and (100 - NO bid)
            implied_yes = yes_price / 100.0
            implied_from_no = (100 - no_price) / 100.0
            mid_prob = (implied_yes + implied_from_no) / 2.0
        elif yes_price is not None:
            mid_prob = yes_price / 100.0
        elif no_price is not None:
            mid_prob = (100 - no_price) / 100.0
        else:
            mid_prob = float("nan")

        logger.info(
            "Implied probability for '%s': %.4f",
            market_ticker, mid_prob,
        )
        return mid_prob


# ===========================================================================
# 2. Market Data Parsing Utilities
# ===========================================================================

def parse_market_threshold(market: dict) -> dict:
    """Extract temperature threshold/range from a market dictionary.

    Parses the market title or subtitle to determine the temperature
    threshold and direction (above/below/between).

    Parameters
    ----------
    market : dict
        A Kalshi market dictionary. Expected to have at least one of:
        "title", "subtitle", "yes_sub_title", or "ticker".

    Returns
    -------
    dict
        Dictionary with keys:
        - "ticker": str, the market ticker
        - "threshold": float or None, the temperature threshold
        - "threshold_low": float or None, lower bound for range buckets
        - "threshold_high": float or None, upper bound for range buckets
        - "direction": str, one of "above", "below", "between", or "unknown"

    Examples
    --------
    >>> market = {"ticker": "KXHIGHNY-T85", "title": "High temp >= 85°F"}
    >>> parse_market_threshold(market)
    {'ticker': 'KXHIGHNY-T85', 'threshold': 85.0, ...}
    """
    ticker = market.get("ticker", "")
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    yes_sub = market.get("yes_sub_title", "")

    # Combine text fields for parsing
    text = f"{title} {subtitle} {yes_sub}".strip()

    result = {
        "ticker": ticker,
        "threshold": None,
        "threshold_low": None,
        "threshold_high": None,
        "direction": "unknown",
    }

    # Pattern: ">= X" or "≥ X" or "above X" or "> X"
    match_above = re.search(
        r'(?:>=|≥|[Aa]bove|[Gg]reater\s+than(?:\s+or\s+equal\s+to)?)\s*'
        r'(\d+(?:\.\d+)?)',
        text,
    )
    if match_above:
        result["threshold"] = float(match_above.group(1))
        result["direction"] = "above"
        return result

    # Pattern: "<= X" or "≤ X" or "below X" or "< X" or "under X"
    match_below = re.search(
        r'(?:<=|≤|[Bb]elow|[Ll]ess\s+than(?:\s+or\s+equal\s+to)?|'
        r'[Uu]nder)\s*(\d+(?:\.\d+)?)',
        text,
    )
    if match_below:
        result["threshold"] = float(match_below.group(1))
        result["direction"] = "below"
        return result

    # Pattern: "X to Y" or "X-Y" or "between X and Y"
    match_range = re.search(
        r'(?:[Bb]etween\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*'
        r'(\d+(?:\.\d+)?)',
        text,
    )
    if match_range:
        lo = float(match_range.group(1))
        hi = float(match_range.group(2))
        result["threshold_low"] = min(lo, hi)
        result["threshold_high"] = max(lo, hi)
        result["direction"] = "between"
        return result

    # Fallback: try to extract threshold from ticker (e.g., "T85")
    match_ticker = re.search(r'T(\d+)', ticker)
    if match_ticker:
        result["threshold"] = float(match_ticker.group(1))
        result["direction"] = "above"
        return result

    logger.warning(
        "Could not parse threshold from market '%s' (title='%s')",
        ticker, title,
    )
    return result


def parse_market_buckets(markets: list[dict]) -> list[dict]:
    """Parse a list of markets into structured bucket data.

    Parameters
    ----------
    markets : list[dict]
        List of Kalshi market dictionaries.

    Returns
    -------
    list[dict]
        List of bucket dictionaries with keys:
        - "ticker": str
        - "threshold_low": float or None
        - "threshold_high": float or None
        - "market_yes_price": float or None
        - "market_no_price": float or None
        - "implied_prob": float
        - "direction": str
    """
    if not markets:
        logger.warning("Empty markets list — returning empty buckets")
        return []

    buckets = []

    for market in markets:
        parsed = parse_market_threshold(market)

        # Extract prices
        yes_price = market.get("yes_ask") or market.get("yes_bid")
        no_price = market.get("no_ask") or market.get("no_bid")
        last_price = market.get("last_price")

        # Compute implied probability from available price data
        if yes_price is not None and isinstance(yes_price, (int, float)):
            implied_prob = yes_price / 100.0
        elif last_price is not None and isinstance(last_price, (int, float)):
            implied_prob = last_price / 100.0
        elif no_price is not None and isinstance(no_price, (int, float)):
            implied_prob = (100 - no_price) / 100.0
        else:
            implied_prob = float("nan")

        # Determine threshold_low and threshold_high
        threshold_low = parsed.get("threshold_low")
        threshold_high = parsed.get("threshold_high")

        if parsed["direction"] == "above" and parsed["threshold"] is not None:
            threshold_low = parsed["threshold"]
            threshold_high = None  # open-ended above
        elif parsed["direction"] == "below" and parsed["threshold"] is not None:
            threshold_low = None  # open-ended below
            threshold_high = parsed["threshold"]

        bucket = {
            "ticker": parsed["ticker"],
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "market_yes_price": yes_price,
            "market_no_price": no_price,
            "implied_prob": float(np.clip(implied_prob, 0.0, 1.0))
                if not np.isnan(implied_prob) else float("nan"),
            "direction": parsed["direction"],
        }
        buckets.append(bucket)

    logger.info("Parsed %d market buckets", len(buckets))
    return buckets


def resolve_market_outcome(market: dict, actual_tmax: float) -> str:
    """Determine if a market contract would have paid out YES or NO.

    Parameters
    ----------
    market : dict
        A parsed market/bucket dictionary (from ``parse_market_threshold``
        or ``parse_market_buckets``). Must have "direction" and either
        "threshold", "threshold_low"/"threshold_high".
    actual_tmax : float
        The actual observed TMAX for the day.

    Returns
    -------
    str
        "YES" if the contract pays out YES, "NO" otherwise.
        Returns "UNKNOWN" if the market direction cannot be determined.
    """
    if np.isnan(actual_tmax):
        return "UNKNOWN"

    direction = market.get("direction", "unknown")

    if direction == "above":
        threshold = market.get("threshold") or market.get("threshold_low")
        if threshold is None:
            return "UNKNOWN"
        return "YES" if actual_tmax >= threshold else "NO"

    elif direction == "below":
        threshold = market.get("threshold") or market.get("threshold_high")
        if threshold is None:
            return "UNKNOWN"
        return "YES" if actual_tmax < threshold else "NO"

    elif direction == "between":
        lo = market.get("threshold_low")
        hi = market.get("threshold_high")
        if lo is None or hi is None:
            return "UNKNOWN"
        return "YES" if lo <= actual_tmax <= hi else "NO"

    return "UNKNOWN"


# ===========================================================================
# 3. Model-vs-Market Comparison
# ===========================================================================

def compare_model_to_market(
    model_mu: float,
    model_sigma: float,
    markets: list[dict],
) -> pd.DataFrame:
    """Compare model CDF probabilities to market-implied probabilities.

    For each market bucket, computes the model's probability using the
    Gaussian CDF and compares it to the market's implied probability.
    Also computes expected value (EV) for YES and NO positions.

    Parameters
    ----------
    model_mu : float
        Model-predicted mean temperature (deg F).
    model_sigma : float
        Model-predicted standard deviation (deg F).
    markets : list[dict]
        List of market dictionaries (from ``parse_market_buckets`` or
        raw Kalshi markets).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "bucket": str, human-readable bucket description
        - "model_prob": float, model-implied probability
        - "market_prob": float, market-implied probability
        - "prob_delta": float, model_prob - market_prob
        - "ev_yes": float, expected value of buying YES at market price
        - "ev_no": float, expected value of buying NO at market price
    """
    model_sigma = max(model_sigma, 1e-10)

    rows = []
    parsed_buckets = parse_market_buckets(markets) if markets else []

    for bucket in parsed_buckets:
        direction = bucket["direction"]
        market_prob = bucket["implied_prob"]

        # Compute model probability for this bucket
        if direction == "above" and bucket["threshold_low"] is not None:
            model_prob = 1.0 - stats.norm.cdf(
                bucket["threshold_low"], loc=model_mu, scale=model_sigma
            )
            label = f">= {bucket['threshold_low']:.0f}F"

        elif direction == "below" and bucket["threshold_high"] is not None:
            model_prob = stats.norm.cdf(
                bucket["threshold_high"], loc=model_mu, scale=model_sigma
            )
            label = f"< {bucket['threshold_high']:.0f}F"

        elif direction == "between":
            lo = bucket["threshold_low"] or 0
            hi = bucket["threshold_high"] or 120
            model_prob = (
                stats.norm.cdf(hi, loc=model_mu, scale=model_sigma)
                - stats.norm.cdf(lo, loc=model_mu, scale=model_sigma)
            )
            label = f"{lo:.0f}-{hi:.0f}F"

        else:
            model_prob = float("nan")
            label = bucket["ticker"]

        # Compute delta
        prob_delta = model_prob - market_prob if not (
            np.isnan(model_prob) or np.isnan(market_prob)
        ) else float("nan")

        # Compute EV (positive EV = favorable bet)
        # EV_YES = model_prob * (1 - market_price) - (1 - model_prob) * market_price
        # Simplified: EV_YES = model_prob - market_prob
        # EV_NO = (1 - model_prob) - (1 - market_prob) = market_prob - model_prob
        if not (np.isnan(model_prob) or np.isnan(market_prob)):
            ev_yes = model_prob - market_prob
            ev_no = market_prob - model_prob
        else:
            ev_yes = float("nan")
            ev_no = float("nan")

        rows.append({
            "bucket": label,
            "model_prob": float(model_prob),
            "market_prob": float(market_prob),
            "prob_delta": float(prob_delta),
            "ev_yes": float(ev_yes),
            "ev_no": float(ev_no),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "bucket", "model_prob", "market_prob",
            "prob_delta", "ev_yes", "ev_no",
        ])

    logger.info(
        "Compared model (mu=%.1f, sigma=%.1f) to %d market buckets",
        model_mu, model_sigma, len(df),
    )
    return df


def compute_brier_scores(
    model_probs: Union[np.ndarray, pd.Series, list],
    market_probs: Union[np.ndarray, pd.Series, list],
    outcomes: Union[np.ndarray, pd.Series, list],
) -> dict:
    """Compute Brier scores for model and market predictions separately.

    The Brier score measures the accuracy of probabilistic predictions:
        BS = mean((prob - outcome)^2)
    where outcome is 1 (YES) or 0 (NO). Lower is better (0 = perfect).

    Parameters
    ----------
    model_probs : array-like
        Model-predicted probabilities for YES outcomes.
    market_probs : array-like
        Market-implied probabilities for YES outcomes.
    outcomes : array-like
        Binary outcomes (1 = YES, 0 = NO).

    Returns
    -------
    dict
        Dictionary with keys:
        - "model_brier": float, model's Brier score
        - "market_brier": float, market's Brier score
        - "brier_delta": float, model - market (negative = model is better)
        - "n_samples": int
    """
    model_arr = _to_numpy(model_probs)
    market_arr = _to_numpy(market_probs)
    outcome_arr = _to_numpy(outcomes)

    # Validate lengths
    if not (len(model_arr) == len(market_arr) == len(outcome_arr)):
        raise ValueError(
            f"Length mismatch: model={len(model_arr)}, "
            f"market={len(market_arr)}, outcomes={len(outcome_arr)}"
        )

    # Remove entries with any NaN
    valid = ~(np.isnan(model_arr) | np.isnan(market_arr) | np.isnan(outcome_arr))
    model_arr = model_arr[valid]
    market_arr = market_arr[valid]
    outcome_arr = outcome_arr[valid]

    n = len(model_arr)
    if n == 0:
        logger.warning("No valid samples for Brier score computation")
        return {
            "model_brier": float("nan"),
            "market_brier": float("nan"),
            "brier_delta": float("nan"),
            "n_samples": 0,
        }

    model_brier = float(np.mean((model_arr - outcome_arr) ** 2))
    market_brier = float(np.mean((market_arr - outcome_arr) ** 2))

    result = {
        "model_brier": model_brier,
        "market_brier": market_brier,
        "brier_delta": model_brier - market_brier,
        "n_samples": n,
    }

    logger.info(
        "Brier scores (n=%d): model=%.4f, market=%.4f, delta=%.4f",
        n, model_brier, market_brier, result["brier_delta"],
    )
    return result


def compute_log_scores(
    model_probs: Union[np.ndarray, pd.Series, list],
    outcomes: Union[np.ndarray, pd.Series, list],
) -> dict:
    """Compute log score for calibration assessment.

    Log score measures the logarithmic accuracy of predictions:
        LS = -mean(outcome * log(prob) + (1-outcome) * log(1-prob))
    Lower is better (0 = perfect for deterministic outcomes).

    Parameters
    ----------
    model_probs : array-like
        Model-predicted probabilities for YES outcomes.
    outcomes : array-like
        Binary outcomes (1 = YES, 0 = NO).

    Returns
    -------
    dict
        Dictionary with keys:
        - "log_score": float, average negative log-likelihood
        - "n_samples": int
    """
    model_arr = _to_numpy(model_probs)
    outcome_arr = _to_numpy(outcomes)

    if len(model_arr) != len(outcome_arr):
        raise ValueError(
            f"Length mismatch: model_probs={len(model_arr)}, "
            f"outcomes={len(outcome_arr)}"
        )

    # Remove entries with any NaN
    valid = ~(np.isnan(model_arr) | np.isnan(outcome_arr))
    model_arr = model_arr[valid]
    outcome_arr = outcome_arr[valid]

    n = len(model_arr)
    if n == 0:
        logger.warning("No valid samples for log score computation")
        return {"log_score": float("nan"), "n_samples": 0}

    # Clamp probabilities to avoid log(0)
    eps = 1e-15
    model_arr = np.clip(model_arr, eps, 1.0 - eps)

    log_scores = -(
        outcome_arr * np.log(model_arr)
        + (1.0 - outcome_arr) * np.log(1.0 - model_arr)
    )

    result = {
        "log_score": float(np.mean(log_scores)),
        "n_samples": n,
    }

    logger.info("Log score (n=%d): %.4f", n, result["log_score"])
    return result


# ===========================================================================
# 4. Historical Data Utilities
# ===========================================================================

def build_historical_comparison(
    model_predictions_df: pd.DataFrame,
    historical_markets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align historical model predictions with historical market data by date.

    Parameters
    ----------
    model_predictions_df : pd.DataFrame
        DataFrame with model predictions. Must have columns:
        - "date": date or datetime
        - "model_mu": float, predicted mean
        - "model_sigma": float, predicted std
    historical_markets_df : pd.DataFrame
        DataFrame with historical market data. Must have columns:
        - "date": date or datetime
        - "bucket": str, bucket description
        - "market_prob": float, market-implied probability
        - "threshold_low": float or None
        - "threshold_high": float or None
        - "direction": str
        Optionally: "actual_tmax": float

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns from both inputs plus:
        - "model_prob": float, model probability for each bucket
        - "prob_delta": float, model_prob - market_prob
        - "outcome": int (1=YES, 0=NO), if actual_tmax is available
    """
    if model_predictions_df.empty or historical_markets_df.empty:
        logger.warning("Empty input DataFrame(s) — returning empty result")
        return pd.DataFrame()

    # Ensure date columns are datetime
    model_df = model_predictions_df.copy()
    market_df = historical_markets_df.copy()

    model_df["date"] = pd.to_datetime(model_df["date"]).dt.date
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date

    # Merge on date
    merged = market_df.merge(model_df, on="date", how="inner")

    if merged.empty:
        logger.warning("No overlapping dates between model and market data")
        return merged

    # Compute model probability for each row
    model_probs = []
    for _, row in merged.iterrows():
        mu = row["model_mu"]
        sigma = max(row["model_sigma"], 1e-10)
        direction = row.get("direction", "unknown")

        if direction == "above" and pd.notna(row.get("threshold_low")):
            prob = 1.0 - stats.norm.cdf(row["threshold_low"], loc=mu, scale=sigma)
        elif direction == "below" and pd.notna(row.get("threshold_high")):
            prob = stats.norm.cdf(row["threshold_high"], loc=mu, scale=sigma)
        elif direction == "between":
            lo = row.get("threshold_low", 0)
            hi = row.get("threshold_high", 120)
            if pd.isna(lo):
                lo = 0
            if pd.isna(hi):
                hi = 120
            prob = (
                stats.norm.cdf(hi, loc=mu, scale=sigma)
                - stats.norm.cdf(lo, loc=mu, scale=sigma)
            )
        else:
            prob = float("nan")

        model_probs.append(prob)

    merged["model_prob"] = model_probs
    merged["prob_delta"] = merged["model_prob"] - merged["market_prob"]

    # Resolve outcomes if actual_tmax is available
    if "actual_tmax" in merged.columns:
        outcomes = []
        for _, row in merged.iterrows():
            direction = row.get("direction", "unknown")
            actual = row["actual_tmax"]

            if np.isnan(actual):
                outcomes.append(float("nan"))
            elif direction == "above" and pd.notna(row.get("threshold_low")):
                outcomes.append(1.0 if actual >= row["threshold_low"] else 0.0)
            elif direction == "below" and pd.notna(row.get("threshold_high")):
                outcomes.append(1.0 if actual < row["threshold_high"] else 0.0)
            elif direction == "between":
                lo = row.get("threshold_low", 0)
                hi = row.get("threshold_high", 120)
                if pd.isna(lo):
                    lo = 0
                if pd.isna(hi):
                    hi = 120
                outcomes.append(1.0 if lo <= actual <= hi else 0.0)
            else:
                outcomes.append(float("nan"))

        merged["outcome"] = outcomes

    logger.info(
        "Built historical comparison: %d rows across %d unique dates",
        len(merged), merged["date"].nunique(),
    )
    return merged


def generate_market_report(
    comparison_df: pd.DataFrame,
    output_dir: str = "results/kalshi",
) -> dict:
    """Generate summary statistics and plots for model-vs-market comparison.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from ``build_historical_comparison`` or manually
        assembled DataFrame with columns: date, bucket, model_prob,
        market_prob, prob_delta. Optionally: outcome, model_mu,
        model_sigma.
    output_dir : str
        Directory to save reports and plots.

    Returns
    -------
    dict
        Summary statistics dictionary with keys:
        - "n_comparisons": int
        - "mean_prob_delta": float
        - "std_prob_delta": float
        - "mean_abs_prob_delta": float
        - "model_brier": float (if outcomes available)
        - "market_brier": float (if outcomes available)
        - "plots_saved": list[str]
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_saved = []

    if comparison_df.empty:
        logger.warning("Empty comparison DataFrame — generating minimal report")
        summary = {
            "n_comparisons": 0,
            "mean_prob_delta": float("nan"),
            "std_prob_delta": float("nan"),
            "mean_abs_prob_delta": float("nan"),
            "plots_saved": [],
        }
        return summary

    # Basic statistics
    n = len(comparison_df)
    prob_deltas = comparison_df["prob_delta"].dropna()

    summary = {
        "n_comparisons": n,
        "mean_prob_delta": float(prob_deltas.mean()) if len(prob_deltas) > 0
            else float("nan"),
        "std_prob_delta": float(prob_deltas.std()) if len(prob_deltas) > 0
            else float("nan"),
        "mean_abs_prob_delta": float(prob_deltas.abs().mean()) if len(prob_deltas) > 0
            else float("nan"),
    }

    # --- Plot 1: Model vs market probability scatter ---
    if "model_prob" in comparison_df.columns and "market_prob" in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        valid_mask = (
            comparison_df["model_prob"].notna()
            & comparison_df["market_prob"].notna()
        )
        if valid_mask.any():
            ax.scatter(
                comparison_df.loc[valid_mask, "market_prob"],
                comparison_df.loc[valid_mask, "model_prob"],
                alpha=0.5, s=20, edgecolors="none",
            )
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Agreement line")
            ax.set_xlabel("Market Probability")
            ax.set_ylabel("Model Probability")
            ax.set_title("Model vs Market Probability")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.legend()

        path = os.path.join(output_dir, "model_vs_market_scatter.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots_saved.append(path)
        logger.info("Saved scatter plot to %s", path)

    # --- Plot 2: EV delta distribution histogram ---
    if "prob_delta" in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        deltas = comparison_df["prob_delta"].dropna()
        if len(deltas) > 0:
            ax.hist(deltas, bins=30, edgecolor="white", alpha=0.75,
                    color="#2ca02c")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(deltas.mean(), color="red", linestyle="-.",
                       linewidth=1, label=f"Mean = {deltas.mean():.4f}")
            ax.set_xlabel("Probability Delta (Model - Market)")
            ax.set_ylabel("Count")
            ax.set_title("EV Delta Distribution")
            ax.legend()

        path = os.path.join(output_dir, "ev_delta_histogram.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots_saved.append(path)
        logger.info("Saved EV delta histogram to %s", path)

    # --- Plot 3: Daily probability comparison time series ---
    if "date" in comparison_df.columns and "model_prob" in comparison_df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        df_sorted = comparison_df.sort_values("date")
        ax.plot(
            pd.to_datetime(df_sorted["date"]),
            df_sorted["model_prob"],
            label="Model", linewidth=1.2, alpha=0.7,
        )
        if "market_prob" in df_sorted.columns:
            ax.plot(
                pd.to_datetime(df_sorted["date"]),
                df_sorted["market_prob"],
                label="Market", linewidth=1.2, alpha=0.7,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Probability")
        ax.set_title("Daily Probability Comparison")
        ax.legend()
        fig.autofmt_xdate()

        path = os.path.join(output_dir, "daily_probability_timeseries.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots_saved.append(path)
        logger.info("Saved time series plot to %s", path)

    # --- Plot 4 & metrics: Brier score comparison by month/season ---
    if "outcome" in comparison_df.columns:
        valid_outcome = comparison_df.dropna(
            subset=["model_prob", "market_prob", "outcome"]
        )

        if len(valid_outcome) > 0:
            model_brier = float(np.mean(
                (valid_outcome["model_prob"] - valid_outcome["outcome"]) ** 2
            ))
            market_brier = float(np.mean(
                (valid_outcome["market_prob"] - valid_outcome["outcome"]) ** 2
            ))
            summary["model_brier"] = model_brier
            summary["market_brier"] = market_brier

            # Monthly Brier scores
            valid_outcome = valid_outcome.copy()
            valid_outcome["month"] = pd.to_datetime(
                valid_outcome["date"]
            ).dt.month

            monthly_model = valid_outcome.groupby("month").apply(
                lambda g: np.mean((g["model_prob"] - g["outcome"]) ** 2)
            )
            monthly_market = valid_outcome.groupby("month").apply(
                lambda g: np.mean((g["market_prob"] - g["outcome"]) ** 2)
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(monthly_model))
            width = 0.35
            ax.bar(x - width / 2, monthly_model.values, width,
                   label="Model", color="#4c72b0", alpha=0.75)
            ax.bar(x + width / 2, monthly_market.values, width,
                   label="Market", color="#d62728", alpha=0.75)
            ax.set_xlabel("Month")
            ax.set_ylabel("Brier Score")
            ax.set_title("Brier Score Comparison by Month")
            ax.set_xticks(x)
            month_labels = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]
            ax.set_xticklabels(
                [month_labels[m - 1] for m in monthly_model.index]
            )
            ax.legend()

            path = os.path.join(output_dir, "brier_by_month.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots_saved.append(path)
            logger.info("Saved Brier by month to %s", path)

    # --- Table: Bucket-level accuracy summary ---
    if "bucket" in comparison_df.columns and "outcome" in comparison_df.columns:
        bucket_summary = comparison_df.groupby("bucket").agg(
            n=("outcome", "count"),
            mean_model_prob=("model_prob", "mean"),
            mean_market_prob=("market_prob", "mean"),
            mean_prob_delta=("prob_delta", "mean"),
            yes_rate=("outcome", "mean"),
        ).reset_index()

        csv_path = os.path.join(output_dir, "bucket_accuracy_summary.csv")
        bucket_summary.to_csv(csv_path, index=False)
        logger.info("Saved bucket summary to %s", csv_path)

    # Save summary metrics
    summary["plots_saved"] = plots_saved
    summary_path = os.path.join(output_dir, "market_report_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved market report summary to %s", summary_path)

    return summary
