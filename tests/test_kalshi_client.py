"""
Tests for the Kalshi market client module (src/kalshi_client.py).

Validates:
  - KalshiClient: initialization, get_series, get_markets, pagination,
    get_historical_markets, get_orderbook, implied probability
  - Retry logic: exponential backoff, max retries exhausted
  - Rate limiting: enforces minimum interval between requests
  - Error handling: HTTP 404, 500, timeout, connection errors
  - Market parsing: parse_market_threshold, parse_market_buckets,
    resolve_market_outcome with various formats and edge cases
  - Model-vs-market: compare_model_to_market, Brier scores, log scores
  - Historical utilities: build_historical_comparison, generate_market_report
  - Edge cases: empty inputs, NaN handling, malformed data, zero sigma

Target: at least 60 meaningful tests.

IMPORTANT: All API calls are mocked. No real HTTP requests are made.
"""

import os
import sys
import json
import math
import tempfile
import shutil
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kalshi_client import (
    KalshiClient,
    parse_market_threshold,
    parse_market_buckets,
    resolve_market_outcome,
    compare_model_to_market,
    compute_brier_scores,
    compute_log_scores,
    build_historical_comparison,
    generate_market_report,
    KALSHI_BASE_URL,
    KXHIGHNY_SERIES,
    DEFAULT_BUCKET_THRESHOLDS,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    RATE_LIMIT_RPS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="test_kalshi_client_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def client():
    """Create a KalshiClient instance for testing."""
    return KalshiClient()


@pytest.fixture
def sample_series_response():
    """Sample Kalshi series API response."""
    return {
        "series": {
            "ticker": "KXHIGHNY",
            "title": "Highest temperature in NYC today",
            "frequency": "daily",
            "category": "weather",
        }
    }


@pytest.fixture
def sample_markets_response():
    """Sample Kalshi markets API response."""
    return {
        "markets": [
            {
                "ticker": "KXHIGHNY-25FEB09-T45",
                "event_ticker": "KXHIGHNY-25FEB09",
                "title": "High temp >= 45°F in NYC",
                "yes_ask": 72,
                "no_ask": 32,
                "yes_bid": 68,
                "no_bid": 28,
                "last_price": 70,
                "status": "open",
            },
            {
                "ticker": "KXHIGHNY-25FEB09-T50",
                "event_ticker": "KXHIGHNY-25FEB09",
                "title": "High temp >= 50°F in NYC",
                "yes_ask": 45,
                "no_ask": 58,
                "yes_bid": 42,
                "no_bid": 55,
                "last_price": 43,
                "status": "open",
            },
        ],
        "cursor": None,
    }


@pytest.fixture
def sample_orderbook_response():
    """Sample Kalshi orderbook API response."""
    return {
        "orderbook": {
            "yes": [[65, 100], [60, 200], [55, 50]],
            "no": [[40, 150], [35, 100]],
        }
    }


@pytest.fixture
def sample_markets_for_parsing():
    """Sample markets with various title formats for parsing tests."""
    return [
        {
            "ticker": "KXHIGHNY-T85",
            "title": "High temp >= 85°F",
            "yes_ask": 20,
            "no_ask": 82,
        },
        {
            "ticker": "KXHIGHNY-T50",
            "title": "High temp >= 50°F",
            "yes_ask": 75,
            "no_ask": 28,
        },
        {
            "ticker": "KXHIGHNY-RANGE",
            "title": "High temp between 60 and 70°F",
            "yes_ask": 35,
            "no_ask": 68,
        },
    ]


@pytest.fixture
def sample_comparison_df():
    """Sample comparison DataFrame for report generation."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "bucket": [f">= {t}F" for t in np.random.choice([70, 75, 80, 85], n)],
        "model_prob": np.random.uniform(0.1, 0.9, n),
        "market_prob": np.random.uniform(0.1, 0.9, n),
        "prob_delta": np.random.uniform(-0.3, 0.3, n),
        "outcome": np.random.choice([0.0, 1.0], n),
        "direction": ["above"] * n,
        "threshold_low": np.random.choice([70, 75, 80, 85], n).astype(float),
        "threshold_high": [None] * n,
        "model_mu": np.random.uniform(70, 85, n),
        "model_sigma": np.random.uniform(3, 6, n),
    })


# ===========================================================================
# A. KalshiClient — Initialization
# ===========================================================================

class TestKalshiClientInit:
    """Tests for KalshiClient initialization."""

    def test_default_base_url(self):
        """Client uses default base URL when none specified."""
        client = KalshiClient()
        assert client.base_url == KALSHI_BASE_URL

    def test_custom_base_url(self):
        """Client accepts custom base URL."""
        custom_url = "https://custom.api.com/v2"
        client = KalshiClient(base_url=custom_url)
        assert client.base_url == custom_url

    def test_trailing_slash_stripped(self):
        """Trailing slash is stripped from base URL."""
        client = KalshiClient(base_url="https://api.example.com/v2/")
        assert client.base_url == "https://api.example.com/v2"

    def test_rate_limit_interval(self):
        """Rate limit minimum interval is correctly computed."""
        client = KalshiClient()
        expected = 1.0 / RATE_LIMIT_RPS
        assert abs(client._min_interval - expected) < 1e-10


# ===========================================================================
# A. KalshiClient — get_series
# ===========================================================================

class TestGetSeries:
    """Tests for KalshiClient.get_series."""

    @patch("src.kalshi_client.requests.get")
    def test_get_series_success(self, mock_get, client, sample_series_response):
        """get_series returns series metadata on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_series_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.get_series("KXHIGHNY")

        assert result["ticker"] == "KXHIGHNY"
        assert result["title"] == "Highest temperature in NYC today"
        assert result["frequency"] == "daily"

    @patch("src.kalshi_client.requests.get")
    def test_get_series_default_ticker(self, mock_get, client, sample_series_response):
        """get_series uses default KXHIGHNY ticker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_series_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client.get_series()

        called_url = mock_get.call_args[0][0]
        assert "KXHIGHNY" in called_url


# ===========================================================================
# A. KalshiClient — get_markets
# ===========================================================================

class TestGetMarkets:
    """Tests for KalshiClient.get_markets."""

    @patch("src.kalshi_client.requests.get")
    def test_get_markets_success(self, mock_get, client, sample_markets_response):
        """get_markets returns list of market dicts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_markets_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        markets = client.get_markets("KXHIGHNY", status="open")

        assert len(markets) == 2
        assert markets[0]["ticker"] == "KXHIGHNY-25FEB09-T45"
        assert markets[1]["ticker"] == "KXHIGHNY-25FEB09-T50"

    @patch("src.kalshi_client.requests.get")
    def test_get_markets_pagination(self, mock_get, client):
        """get_markets handles multi-page responses."""
        page1 = {
            "markets": [{"ticker": f"M{i}"} for i in range(200)],
            "cursor": "page2_cursor",
        }
        page2 = {
            "markets": [{"ticker": f"M{i}"} for i in range(200, 350)],
            "cursor": None,
        }

        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = page1
        mock_response1.raise_for_status.return_value = None

        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = page2
        mock_response2.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response1, mock_response2]

        markets = client.get_markets("KXHIGHNY", status="all")
        assert len(markets) == 350

    @patch("src.kalshi_client.requests.get")
    def test_get_markets_empty(self, mock_get, client):
        """get_markets returns empty list when no markets found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markets": [], "cursor": None}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        markets = client.get_markets("KXHIGHNY")
        assert markets == []

    @patch("src.kalshi_client.requests.get")
    def test_get_markets_respects_limit(self, mock_get, client):
        """get_markets passes correct limit param."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markets": [], "cursor": None}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client.get_markets("KXHIGHNY", limit=50)

        _, kwargs = mock_get.call_args
        assert kwargs.get("params", {}).get("limit") == 50


# ===========================================================================
# A. KalshiClient — get_historical_markets
# ===========================================================================

class TestGetHistoricalMarkets:
    """Tests for KalshiClient.get_historical_markets."""

    @patch("src.kalshi_client.requests.get")
    def test_historical_markets_with_dates(self, mock_get, client):
        """get_historical_markets passes date filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "markets": [{"ticker": "KXHIGHNY-HIST-1", "status": "settled"}],
            "cursor": None,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        markets = client.get_historical_markets(
            "KXHIGHNY", min_date="2024-01-01", max_date="2024-12-31"
        )

        assert len(markets) == 1
        _, kwargs = mock_get.call_args
        params = kwargs.get("params", {})
        assert params["min_close_ts"] == "2024-01-01"
        assert params["max_close_ts"] == "2024-12-31"
        assert params["status"] == "settled"

    @patch("src.kalshi_client.requests.get")
    def test_historical_markets_no_dates(self, mock_get, client):
        """get_historical_markets works without date filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markets": [], "cursor": None}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        markets = client.get_historical_markets("KXHIGHNY")
        assert markets == []


# ===========================================================================
# A. KalshiClient — get_orderbook
# ===========================================================================

class TestGetOrderbook:
    """Tests for KalshiClient.get_orderbook."""

    @patch("src.kalshi_client.requests.get")
    def test_get_orderbook_success(self, mock_get, client, sample_orderbook_response):
        """get_orderbook returns orderbook data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_orderbook_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        orderbook = client.get_orderbook("KXHIGHNY-25FEB09-T45")

        assert "yes" in orderbook
        assert "no" in orderbook
        assert len(orderbook["yes"]) == 3
        assert len(orderbook["no"]) == 2

    @patch("src.kalshi_client.requests.get")
    def test_get_orderbook_empty(self, mock_get, client):
        """get_orderbook handles empty orderbook."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [], "no": []}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        orderbook = client.get_orderbook("KXHIGHNY-EMPTY")
        assert orderbook["yes"] == []
        assert orderbook["no"] == []

    @patch("src.kalshi_client.requests.get")
    def test_get_orderbook_yes_only(self, mock_get, client):
        """get_orderbook handles orderbook with only YES bids."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [[70, 50]], "no": []}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        orderbook = client.get_orderbook("KXHIGHNY-YESONLY")
        assert len(orderbook["yes"]) == 1
        assert len(orderbook["no"]) == 0


# ===========================================================================
# A. KalshiClient — get_market_implied_probability
# ===========================================================================

class TestImpliedProbability:
    """Tests for KalshiClient.get_market_implied_probability."""

    @patch("src.kalshi_client.requests.get")
    def test_implied_prob_both_sides(self, mock_get, client):
        """Mid-price from both YES and NO bids."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [[70, 100]], "no": [[40, 100]]}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        prob = client.get_market_implied_probability("KXHIGHNY-T50")

        # yes_price=70 -> 0.70; no_price=40 -> (100-40)/100 = 0.60
        # mid = (0.70 + 0.60) / 2 = 0.65
        assert abs(prob - 0.65) < 1e-10

    @patch("src.kalshi_client.requests.get")
    def test_implied_prob_yes_only(self, mock_get, client):
        """Implied prob from YES bid only when NO is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [[80, 100]], "no": []}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        prob = client.get_market_implied_probability("KXHIGHNY-T50")
        assert abs(prob - 0.80) < 1e-10

    @patch("src.kalshi_client.requests.get")
    def test_implied_prob_no_only(self, mock_get, client):
        """Implied prob from NO bid only when YES is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [], "no": [[25, 100]]}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        prob = client.get_market_implied_probability("KXHIGHNY-T50")
        # (100 - 25) / 100 = 0.75
        assert abs(prob - 0.75) < 1e-10

    @patch("src.kalshi_client.requests.get")
    def test_implied_prob_empty_book(self, mock_get, client):
        """Returns NaN for completely empty orderbook."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "orderbook": {"yes": [], "no": []}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        prob = client.get_market_implied_probability("KXHIGHNY-EMPTY")
        assert np.isnan(prob)


# ===========================================================================
# A. KalshiClient — Retry logic
# ===========================================================================

class TestRetryLogic:
    """Tests for request retry logic."""

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep, mock_get, client):
        """Client retries on ConnectionError then succeeds."""
        import requests as req_lib

        fail_response = MagicMock(side_effect=req_lib.exceptions.ConnectionError("fail"))
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"series": {"ticker": "KXHIGHNY"}}
        success_response.raise_for_status.return_value = None

        mock_get.side_effect = [fail_response, success_response]

        result = client.get_series("KXHIGHNY")
        assert result["ticker"] == "KXHIGHNY"
        assert mock_get.call_count == 2

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep, mock_get, client):
        """Client raises ConnectionError after MAX_RETRIES failures."""
        import requests as req_lib

        mock_get.side_effect = req_lib.exceptions.ConnectionError("persistent failure")

        with pytest.raises(ConnectionError, match="retries exhausted"):
            client.get_series("KXHIGHNY")

        assert mock_get.call_count == MAX_RETRIES

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_retry_on_server_error(self, mock_sleep, mock_get, client):
        """Client retries on HTTP 500 errors."""
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status.side_effect = Exception("500 error")

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"series": {"ticker": "KXHIGHNY"}}
        success_response.raise_for_status.return_value = None

        mock_get.side_effect = [fail_response, success_response]

        result = client.get_series("KXHIGHNY")
        assert result["ticker"] == "KXHIGHNY"

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_retry_on_timeout(self, mock_sleep, mock_get, client):
        """Client retries on Timeout errors."""
        import requests as req_lib

        fail = MagicMock(side_effect=req_lib.exceptions.Timeout("timeout"))
        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"data": "ok"}
        success.raise_for_status.return_value = None

        mock_get.side_effect = [fail, success]

        result = client._request("https://example.com/test")
        assert result == {"data": "ok"}


# ===========================================================================
# A. KalshiClient — Error handling
# ===========================================================================

class TestErrorHandling:
    """Tests for HTTP error handling."""

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_404_raises_value_error(self, mock_sleep, mock_get, client):
        """HTTP 404 raises ValueError immediately (no retry)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="404"):
            client.get_series("NONEXISTENT")

    @patch("src.kalshi_client.requests.get")
    @patch("src.kalshi_client.time.sleep")
    def test_500_retries_then_fails(self, mock_sleep, mock_get, client):
        """HTTP 500 retries and eventually raises ConnectionError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(ConnectionError, match="retries exhausted"):
            client._request("https://example.com/fail")


# ===========================================================================
# A. KalshiClient — Rate limiting
# ===========================================================================

class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @patch("src.kalshi_client.time.sleep")
    @patch("src.kalshi_client.time.time")
    def test_rate_limit_sleeps_when_too_fast(self, mock_time, mock_sleep):
        """Rate limiter sleeps when requests are too close together."""
        client = KalshiClient()
        # Simulate: last request at time 100.0, now at 100.01 (too fast)
        client._last_request_time = 100.0
        mock_time.return_value = 100.01

        client._rate_limit()

        # Should sleep for approximately (min_interval - 0.01)
        assert mock_sleep.called
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg > 0

    @patch("src.kalshi_client.time.sleep")
    @patch("src.kalshi_client.time.time")
    def test_rate_limit_no_sleep_when_sufficient_gap(self, mock_time, mock_sleep):
        """Rate limiter does not sleep when enough time has passed."""
        client = KalshiClient()
        client._last_request_time = 100.0
        mock_time.return_value = 101.0  # 1 second later

        client._rate_limit()

        mock_sleep.assert_not_called()


# ===========================================================================
# B. Market Parsing — parse_market_threshold
# ===========================================================================

class TestParseMarketThreshold:
    """Tests for parse_market_threshold."""

    def test_above_format_gte(self):
        """Parse '>= X' format."""
        market = {"ticker": "KXHIGHNY-T85", "title": "High temp >= 85°F"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 85.0
        assert result["direction"] == "above"

    def test_above_format_word(self):
        """Parse 'above X' format."""
        market = {"ticker": "TEST", "title": "Temperature above 90 degrees"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 90.0
        assert result["direction"] == "above"

    def test_below_format(self):
        """Parse 'below X' format."""
        market = {"ticker": "TEST", "title": "Temperature below 32°F"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 32.0
        assert result["direction"] == "below"

    def test_below_format_under(self):
        """Parse 'under X' format."""
        market = {"ticker": "TEST", "title": "Temperature under 50°F"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 50.0
        assert result["direction"] == "below"

    def test_range_format_between(self):
        """Parse 'between X and Y' format."""
        market = {"ticker": "TEST", "title": "Temperature between 60 and 70°F"}
        result = parse_market_threshold(market)
        assert result["threshold_low"] == 60.0
        assert result["threshold_high"] == 70.0
        assert result["direction"] == "between"

    def test_range_format_to(self):
        """Parse 'X to Y' format."""
        market = {"ticker": "TEST", "title": "Temperature 55 to 65"}
        result = parse_market_threshold(market)
        assert result["threshold_low"] == 55.0
        assert result["threshold_high"] == 65.0
        assert result["direction"] == "between"

    def test_range_format_dash(self):
        """Parse 'X-Y' range format."""
        market = {"ticker": "TEST", "title": "Temperature 70-80°F"}
        result = parse_market_threshold(market)
        assert result["threshold_low"] == 70.0
        assert result["threshold_high"] == 80.0
        assert result["direction"] == "between"

    def test_ticker_fallback(self):
        """Falls back to ticker parsing when title is unclear."""
        market = {"ticker": "KXHIGHNY-T95", "title": "Some unclear title"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 95.0
        assert result["direction"] == "above"

    def test_unknown_direction(self):
        """Returns 'unknown' when no pattern matches."""
        market = {"ticker": "UNKNOWN", "title": "No temperature info here"}
        result = parse_market_threshold(market)
        assert result["direction"] == "unknown"
        assert result["threshold"] is None

    def test_empty_market(self):
        """Handles empty market dict gracefully."""
        result = parse_market_threshold({})
        assert result["direction"] == "unknown"
        assert result["ticker"] == ""

    def test_subtitle_parsing(self):
        """Parses threshold from subtitle field."""
        market = {
            "ticker": "TEST",
            "title": "",
            "subtitle": "Will the high temperature be >= 80°F?",
        }
        result = parse_market_threshold(market)
        assert result["threshold"] == 80.0
        assert result["direction"] == "above"

    def test_yes_sub_title_parsing(self):
        """Parses threshold from yes_sub_title field."""
        market = {
            "ticker": "TEST",
            "title": "",
            "yes_sub_title": "High temp above 75",
        }
        result = parse_market_threshold(market)
        assert result["threshold"] == 75.0
        assert result["direction"] == "above"

    def test_unicode_gte_symbol(self):
        """Parse unicode >= symbol."""
        market = {"ticker": "TEST", "title": "Temp \u2265 88\u00b0F"}
        result = parse_market_threshold(market)
        assert result["threshold"] == 88.0
        assert result["direction"] == "above"


# ===========================================================================
# B. Market Parsing — parse_market_buckets
# ===========================================================================

class TestParseMarketBuckets:
    """Tests for parse_market_buckets."""

    def test_parse_multiple_markets(self, sample_markets_for_parsing):
        """Parses a list of markets into structured buckets."""
        buckets = parse_market_buckets(sample_markets_for_parsing)
        assert len(buckets) == 3

        # First bucket: >= 85°F
        assert buckets[0]["ticker"] == "KXHIGHNY-T85"
        assert buckets[0]["threshold_low"] == 85.0
        assert buckets[0]["direction"] == "above"

        # Third bucket: between 60 and 70
        assert buckets[2]["direction"] == "between"
        assert buckets[2]["threshold_low"] == 60.0
        assert buckets[2]["threshold_high"] == 70.0

    def test_implied_prob_from_yes_ask(self):
        """Implied probability from yes_ask price."""
        markets = [{"ticker": "T1", "title": ">= 80°F", "yes_ask": 60}]
        buckets = parse_market_buckets(markets)
        assert abs(buckets[0]["implied_prob"] - 0.60) < 1e-10

    def test_implied_prob_from_last_price(self):
        """Implied probability from last_price when yes_ask missing."""
        markets = [{"ticker": "T1", "title": ">= 80°F", "last_price": 55}]
        buckets = parse_market_buckets(markets)
        assert abs(buckets[0]["implied_prob"] - 0.55) < 1e-10

    def test_implied_prob_from_no_price(self):
        """Implied probability from no_ask when yes prices missing."""
        markets = [{"ticker": "T1", "title": ">= 80°F", "no_ask": 40}]
        buckets = parse_market_buckets(markets)
        # (100 - 40) / 100 = 0.60
        assert abs(buckets[0]["implied_prob"] - 0.60) < 1e-10

    def test_implied_prob_nan_when_no_prices(self):
        """Returns NaN implied prob when no price data."""
        markets = [{"ticker": "T1", "title": ">= 80°F"}]
        buckets = parse_market_buckets(markets)
        assert np.isnan(buckets[0]["implied_prob"])

    def test_empty_markets_list(self):
        """Returns empty list for empty markets."""
        assert parse_market_buckets([]) == []

    def test_prob_clipped_to_0_1(self):
        """Implied probabilities are clipped to [0, 1]."""
        markets = [{"ticker": "T1", "title": ">= 50°F", "yes_ask": 150}]
        buckets = parse_market_buckets(markets)
        assert buckets[0]["implied_prob"] <= 1.0


# ===========================================================================
# B. Market Parsing — resolve_market_outcome
# ===========================================================================

class TestResolveMarketOutcome:
    """Tests for resolve_market_outcome."""

    def test_above_yes(self):
        """Above threshold returns YES."""
        market = {"direction": "above", "threshold": 85.0}
        assert resolve_market_outcome(market, 90.0) == "YES"

    def test_above_no(self):
        """Below threshold returns NO."""
        market = {"direction": "above", "threshold": 85.0}
        assert resolve_market_outcome(market, 80.0) == "NO"

    def test_above_exact_threshold(self):
        """Exact threshold for above returns YES."""
        market = {"direction": "above", "threshold": 85.0}
        assert resolve_market_outcome(market, 85.0) == "YES"

    def test_below_yes(self):
        """Below threshold returns YES."""
        market = {"direction": "below", "threshold": 32.0}
        assert resolve_market_outcome(market, 28.0) == "YES"

    def test_below_no(self):
        """At or above threshold for below returns NO."""
        market = {"direction": "below", "threshold": 32.0}
        assert resolve_market_outcome(market, 32.0) == "NO"

    def test_between_yes(self):
        """Within range returns YES."""
        market = {
            "direction": "between",
            "threshold_low": 60.0,
            "threshold_high": 70.0,
        }
        assert resolve_market_outcome(market, 65.0) == "YES"

    def test_between_no_below(self):
        """Below range returns NO."""
        market = {
            "direction": "between",
            "threshold_low": 60.0,
            "threshold_high": 70.0,
        }
        assert resolve_market_outcome(market, 55.0) == "NO"

    def test_between_no_above(self):
        """Above range returns NO."""
        market = {
            "direction": "between",
            "threshold_low": 60.0,
            "threshold_high": 70.0,
        }
        assert resolve_market_outcome(market, 75.0) == "NO"

    def test_between_boundary_low(self):
        """At lower boundary returns YES."""
        market = {
            "direction": "between",
            "threshold_low": 60.0,
            "threshold_high": 70.0,
        }
        assert resolve_market_outcome(market, 60.0) == "YES"

    def test_between_boundary_high(self):
        """At upper boundary returns YES."""
        market = {
            "direction": "between",
            "threshold_low": 60.0,
            "threshold_high": 70.0,
        }
        assert resolve_market_outcome(market, 70.0) == "YES"

    def test_nan_actual_returns_unknown(self):
        """NaN actual TMAX returns UNKNOWN."""
        market = {"direction": "above", "threshold": 85.0}
        assert resolve_market_outcome(market, float("nan")) == "UNKNOWN"

    def test_unknown_direction(self):
        """Unknown direction returns UNKNOWN."""
        market = {"direction": "unknown"}
        assert resolve_market_outcome(market, 70.0) == "UNKNOWN"

    def test_missing_threshold(self):
        """Missing threshold returns UNKNOWN."""
        market = {"direction": "above"}
        assert resolve_market_outcome(market, 70.0) == "UNKNOWN"

    def test_above_with_threshold_low_key(self):
        """Above direction uses threshold_low fallback."""
        market = {"direction": "above", "threshold_low": 90.0}
        assert resolve_market_outcome(market, 92.0) == "YES"


# ===========================================================================
# C. Model-vs-Market — compare_model_to_market
# ===========================================================================

class TestCompareModelToMarket:
    """Tests for compare_model_to_market."""

    def test_basic_comparison(self, sample_markets_for_parsing):
        """compare_model_to_market returns DataFrame with expected columns."""
        df = compare_model_to_market(75.0, 5.0, sample_markets_for_parsing)

        expected_cols = {
            "bucket", "model_prob", "market_prob",
            "prob_delta", "ev_yes", "ev_no",
        }
        assert set(df.columns) == expected_cols
        assert len(df) == 3

    def test_above_threshold_probability(self):
        """Model probability for above threshold is correct."""
        markets = [{"ticker": "T1", "title": ">= 80°F", "yes_ask": 50}]
        df = compare_model_to_market(80.0, 5.0, markets)

        # P(X >= 80) for N(80, 5) should be 0.5
        assert abs(df.iloc[0]["model_prob"] - 0.5) < 0.01

    def test_between_range_probability(self):
        """Model probability for between range is correct."""
        markets = [{
            "ticker": "T1",
            "title": "between 70 and 90",
            "yes_ask": 50,
        }]
        from scipy import stats
        expected = (
            stats.norm.cdf(90, loc=80, scale=5)
            - stats.norm.cdf(70, loc=80, scale=5)
        )
        df = compare_model_to_market(80.0, 5.0, markets)
        assert abs(df.iloc[0]["model_prob"] - expected) < 0.01

    def test_ev_computation(self):
        """EV_YES = model_prob - market_prob."""
        markets = [{"ticker": "T1", "title": ">= 50°F", "yes_ask": 40}]
        df = compare_model_to_market(80.0, 5.0, markets)

        model_prob = df.iloc[0]["model_prob"]
        market_prob = df.iloc[0]["market_prob"]

        assert abs(df.iloc[0]["ev_yes"] - (model_prob - market_prob)) < 1e-10
        assert abs(df.iloc[0]["ev_no"] - (market_prob - model_prob)) < 1e-10

    def test_empty_markets(self):
        """Returns empty DataFrame for empty markets list."""
        df = compare_model_to_market(75.0, 5.0, [])
        assert len(df) == 0
        assert "bucket" in df.columns

    def test_very_small_sigma(self):
        """Handles near-zero sigma gracefully."""
        markets = [{"ticker": "T1", "title": ">= 80°F", "yes_ask": 50}]
        df = compare_model_to_market(85.0, 0.0001, markets)
        # With mu=85, sigma~0, P(>=80) should be ~1.0
        assert df.iloc[0]["model_prob"] > 0.99


# ===========================================================================
# C. Model-vs-Market — compute_brier_scores
# ===========================================================================

class TestBrierScores:
    """Tests for compute_brier_scores."""

    def test_perfect_model(self):
        """Perfect predictions have Brier score = 0."""
        model_probs = [1.0, 0.0, 1.0, 0.0]
        market_probs = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1.0, 0.0, 1.0, 0.0]

        result = compute_brier_scores(model_probs, market_probs, outcomes)
        assert abs(result["model_brier"]) < 1e-10
        assert result["market_brier"] == 0.25  # (0.5)^2 = 0.25

    def test_worst_case_model(self):
        """Completely wrong predictions have Brier score = 1."""
        model_probs = [0.0, 1.0]
        market_probs = [0.5, 0.5]
        outcomes = [1.0, 0.0]

        result = compute_brier_scores(model_probs, market_probs, outcomes)
        assert abs(result["model_brier"] - 1.0) < 1e-10

    def test_brier_delta(self):
        """Brier delta is model - market."""
        model_probs = [0.8, 0.2]
        market_probs = [0.6, 0.4]
        outcomes = [1.0, 0.0]

        result = compute_brier_scores(model_probs, market_probs, outcomes)
        expected_model = np.mean([(0.8 - 1) ** 2, (0.2 - 0) ** 2])
        expected_market = np.mean([(0.6 - 1) ** 2, (0.4 - 0) ** 2])
        assert abs(result["brier_delta"] - (expected_model - expected_market)) < 1e-10

    def test_nan_handling(self):
        """NaN values are excluded from computation."""
        model_probs = [0.8, float("nan"), 0.2]
        market_probs = [0.6, 0.5, 0.4]
        outcomes = [1.0, 0.0, 0.0]

        result = compute_brier_scores(model_probs, market_probs, outcomes)
        assert result["n_samples"] == 2

    def test_length_mismatch_raises(self):
        """Raises ValueError on length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_brier_scores([0.5, 0.5], [0.5], [1.0, 0.0])

    def test_empty_after_nan_removal(self):
        """Returns NaN scores when all entries are NaN."""
        result = compute_brier_scores(
            [float("nan")], [float("nan")], [float("nan")]
        )
        assert result["n_samples"] == 0
        assert np.isnan(result["model_brier"])


# ===========================================================================
# C. Model-vs-Market — compute_log_scores
# ===========================================================================

class TestLogScores:
    """Tests for compute_log_scores."""

    def test_perfect_prediction(self):
        """Perfect deterministic predictions have low log score."""
        model_probs = [0.999, 0.001]
        outcomes = [1.0, 0.0]

        result = compute_log_scores(model_probs, outcomes)
        assert result["log_score"] < 0.01
        assert result["n_samples"] == 2

    def test_worst_case(self):
        """Completely wrong predictions have high log score."""
        model_probs = [0.001, 0.999]
        outcomes = [1.0, 0.0]

        result = compute_log_scores(model_probs, outcomes)
        assert result["log_score"] > 5.0

    def test_uniform_prediction(self):
        """50/50 predictions have log score of ln(2) ~ 0.693."""
        model_probs = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1.0, 0.0, 1.0, 0.0]

        result = compute_log_scores(model_probs, outcomes)
        assert abs(result["log_score"] - np.log(2)) < 0.01

    def test_nan_handling(self):
        """NaN values are excluded from computation."""
        model_probs = [0.8, float("nan")]
        outcomes = [1.0, 0.0]

        result = compute_log_scores(model_probs, outcomes)
        assert result["n_samples"] == 1

    def test_length_mismatch_raises(self):
        """Raises ValueError on length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_log_scores([0.5, 0.5], [1.0])

    def test_prob_clamped_to_avoid_log_zero(self):
        """Probabilities near 0 or 1 are clamped to avoid log(0)."""
        model_probs = [0.0, 1.0]
        outcomes = [1.0, 0.0]

        # Should not raise — probs are clamped internally
        result = compute_log_scores(model_probs, outcomes)
        assert not np.isnan(result["log_score"])
        assert np.isfinite(result["log_score"])


# ===========================================================================
# D. Historical Utilities — build_historical_comparison
# ===========================================================================

class TestBuildHistoricalComparison:
    """Tests for build_historical_comparison."""

    def test_basic_merge(self):
        """Merges model predictions with market data by date."""
        model_df = pd.DataFrame({
            "date": ["2024-07-01", "2024-07-02"],
            "model_mu": [85.0, 82.0],
            "model_sigma": [4.0, 5.0],
        })
        market_df = pd.DataFrame({
            "date": ["2024-07-01", "2024-07-02"],
            "bucket": [">= 85F", ">= 85F"],
            "market_prob": [0.55, 0.40],
            "threshold_low": [85.0, 85.0],
            "threshold_high": [None, None],
            "direction": ["above", "above"],
        })

        result = build_historical_comparison(model_df, market_df)
        assert len(result) == 2
        assert "model_prob" in result.columns
        assert "prob_delta" in result.columns

    def test_with_actual_tmax(self):
        """Includes outcome column when actual_tmax is available."""
        model_df = pd.DataFrame({
            "date": ["2024-07-01"],
            "model_mu": [85.0],
            "model_sigma": [4.0],
        })
        market_df = pd.DataFrame({
            "date": ["2024-07-01"],
            "bucket": [">= 80F"],
            "market_prob": [0.75],
            "threshold_low": [80.0],
            "threshold_high": [None],
            "direction": ["above"],
            "actual_tmax": [88.0],
        })

        result = build_historical_comparison(model_df, market_df)
        assert "outcome" in result.columns
        assert result.iloc[0]["outcome"] == 1.0  # 88 >= 80

    def test_no_overlapping_dates(self):
        """Returns empty DataFrame when dates don't overlap."""
        model_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "model_mu": [40.0],
            "model_sigma": [5.0],
        })
        market_df = pd.DataFrame({
            "date": ["2024-07-01"],
            "bucket": [">= 80F"],
            "market_prob": [0.50],
            "threshold_low": [80.0],
            "threshold_high": [None],
            "direction": ["above"],
        })

        result = build_historical_comparison(model_df, market_df)
        assert len(result) == 0

    def test_empty_inputs(self):
        """Returns empty DataFrame for empty inputs."""
        result = build_historical_comparison(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_between_direction(self):
        """Handles 'between' direction correctly."""
        model_df = pd.DataFrame({
            "date": ["2024-07-01"],
            "model_mu": [75.0],
            "model_sigma": [5.0],
        })
        market_df = pd.DataFrame({
            "date": ["2024-07-01"],
            "bucket": ["70-80F"],
            "market_prob": [0.50],
            "threshold_low": [70.0],
            "threshold_high": [80.0],
            "direction": ["between"],
            "actual_tmax": [75.0],
        })

        result = build_historical_comparison(model_df, market_df)
        assert result.iloc[0]["outcome"] == 1.0  # 75 is between 70-80
        assert result.iloc[0]["model_prob"] > 0


# ===========================================================================
# D. Historical Utilities — generate_market_report
# ===========================================================================

class TestGenerateMarketReport:
    """Tests for generate_market_report."""

    def test_generates_plots(self, sample_comparison_df, tmp_dir):
        """Report generates expected plot files."""
        result = generate_market_report(sample_comparison_df, output_dir=tmp_dir)

        assert result["n_comparisons"] == 50
        assert len(result["plots_saved"]) > 0
        for path in result["plots_saved"]:
            assert os.path.exists(path)

    def test_generates_summary_json(self, sample_comparison_df, tmp_dir):
        """Report saves summary JSON."""
        generate_market_report(sample_comparison_df, output_dir=tmp_dir)

        summary_path = os.path.join(tmp_dir, "market_report_summary.json")
        assert os.path.exists(summary_path)

        with open(summary_path) as f:
            summary = json.load(f)
        assert "n_comparisons" in summary

    def test_generates_brier_scores(self, sample_comparison_df, tmp_dir):
        """Report computes Brier scores when outcomes available."""
        result = generate_market_report(sample_comparison_df, output_dir=tmp_dir)
        assert "model_brier" in result
        assert "market_brier" in result

    def test_empty_dataframe(self, tmp_dir):
        """Handles empty DataFrame gracefully."""
        result = generate_market_report(pd.DataFrame(), output_dir=tmp_dir)
        assert result["n_comparisons"] == 0
        assert np.isnan(result["mean_prob_delta"])

    def test_generates_scatter_plot(self, sample_comparison_df, tmp_dir):
        """Generates model vs market scatter plot."""
        generate_market_report(sample_comparison_df, output_dir=tmp_dir)
        scatter_path = os.path.join(tmp_dir, "model_vs_market_scatter.png")
        assert os.path.exists(scatter_path)

    def test_generates_histogram(self, sample_comparison_df, tmp_dir):
        """Generates EV delta histogram."""
        generate_market_report(sample_comparison_df, output_dir=tmp_dir)
        hist_path = os.path.join(tmp_dir, "ev_delta_histogram.png")
        assert os.path.exists(hist_path)

    def test_generates_timeseries(self, sample_comparison_df, tmp_dir):
        """Generates daily probability time series plot."""
        generate_market_report(sample_comparison_df, output_dir=tmp_dir)
        ts_path = os.path.join(tmp_dir, "daily_probability_timeseries.png")
        assert os.path.exists(ts_path)


# ===========================================================================
# E. Constants and Module-Level Config
# ===========================================================================

class TestModuleConstants:
    """Tests for module-level configuration constants."""

    def test_base_url_is_string(self):
        """KALSHI_BASE_URL is a proper URL string."""
        assert isinstance(KALSHI_BASE_URL, str)
        assert KALSHI_BASE_URL.startswith("https://")

    def test_series_ticker(self):
        """KXHIGHNY_SERIES is the expected ticker."""
        assert KXHIGHNY_SERIES == "KXHIGHNY"

    def test_bucket_thresholds(self):
        """DEFAULT_BUCKET_THRESHOLDS is a proper range."""
        assert DEFAULT_BUCKET_THRESHOLDS == list(range(0, 120, 5))
        assert len(DEFAULT_BUCKET_THRESHOLDS) == 24

    def test_max_retries(self):
        """MAX_RETRIES is a positive integer."""
        assert MAX_RETRIES >= 1
        assert isinstance(MAX_RETRIES, int)

    def test_rate_limit(self):
        """RATE_LIMIT_RPS is a positive number."""
        assert RATE_LIMIT_RPS > 0


# ===========================================================================
# F. End-to-End Integration Flow (mocked)
# ===========================================================================

class TestEndToEndFlow:
    """Integration-style tests with mock data (no real API calls)."""

    def test_full_flow_mock_markets_to_report(self, tmp_dir):
        """End-to-end: mock markets -> parse -> compare -> report."""
        # Step 1: Create mock markets
        mock_markets = [
            {
                "ticker": "KXHIGHNY-25JUL01-T85",
                "title": "High temp >= 85°F",
                "yes_ask": 40,
                "no_ask": 63,
            },
            {
                "ticker": "KXHIGHNY-25JUL01-T90",
                "title": "High temp >= 90°F",
                "yes_ask": 15,
                "no_ask": 88,
            },
        ]

        # Step 2: Parse markets
        buckets = parse_market_buckets(mock_markets)
        assert len(buckets) == 2

        # Step 3: Compare model to market
        df = compare_model_to_market(82.0, 5.0, mock_markets)
        assert len(df) == 2
        assert not df["model_prob"].isna().any()

        # Step 4: Build historical comparison
        # Use distinct dates so the merge is 1:1
        model_df = pd.DataFrame({
            "date": ["2024-07-01", "2024-07-02"],
            "model_mu": [82.0, 82.0],
            "model_sigma": [5.0, 5.0],
        })
        market_df = pd.DataFrame({
            "date": ["2024-07-01", "2024-07-02"],
            "bucket": [">= 85F", ">= 90F"],
            "market_prob": [0.40, 0.15],
            "threshold_low": [85.0, 90.0],
            "threshold_high": [None, None],
            "direction": ["above", "above"],
            "actual_tmax": [87.0, 87.0],
        })

        comparison = build_historical_comparison(model_df, market_df)
        assert len(comparison) == 2
        assert "model_prob" in comparison.columns
        assert "outcome" in comparison.columns

        # 87 >= 85 -> YES, 87 < 90 -> NO
        outcomes = comparison.sort_values("bucket")["outcome"].tolist()
        # The bucket names might not perfectly sort, but check that we have YES and NO
        assert 1.0 in outcomes
        assert 0.0 in outcomes

        # Step 5: Generate report
        report = generate_market_report(comparison, output_dir=tmp_dir)
        assert report["n_comparisons"] == 2
        assert len(report["plots_saved"]) > 0

    @patch("src.kalshi_client.requests.get")
    def test_client_to_comparison_flow(self, mock_get):
        """Flow from client.get_markets to compare_model_to_market."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "markets": [
                {
                    "ticker": "KXHIGHNY-T75",
                    "title": "High temp >= 75°F",
                    "yes_ask": 60,
                    "no_ask": 43,
                    "status": "open",
                },
            ],
            "cursor": None,
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = KalshiClient()
        markets = client.get_markets("KXHIGHNY")

        df = compare_model_to_market(78.0, 4.0, markets)
        assert len(df) == 1
        assert df.iloc[0]["model_prob"] > 0.5  # mu=78 > threshold=75
