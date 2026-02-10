#!/usr/bin/env python3
"""Fetch Kalshi pre-settlement market prices for KXHIGHNY weather markets.

For each date in the target range, this script:
  1. Constructs the event ticker (KXHIGHNY-YYMMMDD or HIGHNY-YYMMMDD)
  2. Lists all markets for that event via the Kalshi public API
  3. For each market, fetches 1-hour candlestick data near market close
  4. Extracts the last available candle as the "pre-settlement" snapshot
  5. Saves results to data/kalshi_presettlement.csv

Usage:
    # Test on 3 recent dates:
    python scripts/fetch_kalshi_presettlement.py --test

    # Full fetch (2023-01-01 to 2025-12-31):
    python scripts/fetch_kalshi_presettlement.py

    # Custom date range:
    python scripts/fetch_kalshi_presettlement.py --start 2024-06-01 --end 2024-06-30

    # Resume from checkpoint:
    python scripts/fetch_kalshi_presettlement.py --resume
"""

import argparse
import csv
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV = DATA_DIR / "kalshi_presettlement.csv"
CHECKPOINT_CSV = DATA_DIR / "kalshi_presettlement_checkpoint.csv"

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXHIGHNY"

# Rate limiting: 1 request/second as specified
MIN_REQUEST_INTERVAL = 1.0

# Retry settings
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 2.0

# Checkpoint every N dates
CHECKPOINT_INTERVAL = 30

# Month abbreviations for event ticker
MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}

# Ticker prefix changed from HIGHNY to KXHIGHNY at some point
# 2023-2024 data uses HIGHNY-, 2025 uses KXHIGHNY-
# We'll try KXHIGHNY first, then fall back to HIGHNY
TICKER_PREFIXES = ["KXHIGHNY", "HIGHNY"]

# CSV output columns
CSV_COLUMNS = [
    "date", "ticker", "bucket", "threshold_low", "threshold_high",
    "direction", "strike_type", "presettlement_prob", "bid_cents",
    "ask_cents", "volume", "open_interest", "snapshot_time_utc",
]


# ===========================================================================
# HTTP Client with rate limiting and retry
# ===========================================================================

class RateLimitedClient:
    """Simple HTTP client with rate limiting and exponential backoff."""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        self._last_request_time = 0.0
        self._min_interval = min_interval
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "NYC-Temp-Prediction/1.0",
        })

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, params: dict = None) -> dict:
        """GET request with retry and rate limiting. Returns parsed JSON."""
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                resp = self._session.get(url, params=params, timeout=30)

                if resp.status_code == 404:
                    return None  # Not found is expected for some dates

                if resp.status_code == 429:
                    # Rate limited — back off more aggressively
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Rate limited (429). Waiting %.1fs... (attempt %d/%d)",
                        wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    raise ConnectionError(
                        f"Server error {resp.status_code} for {url}"
                    )

                resp.raise_for_status()
                return resp.json()

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_exc = exc
                # Check for network blocks
                err_str = str(exc).lower()
                if "refused" in err_str or "ssl" in err_str:
                    print(f"NETWORK ACCESS BLOCKED: {url}")
                    sys.exit(1)

                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("All retries exhausted for %s: %s", url, exc)

            except ConnectionError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Server error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)

            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Unexpected error (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)

        raise ConnectionError(
            f"All {MAX_RETRIES} retries exhausted for {url}: {last_exc}"
        )


# ===========================================================================
# Event/Market ticker construction
# ===========================================================================

def make_event_ticker(d: date, prefix: str) -> str:
    """Construct event ticker like KXHIGHNY-25FEB07 from a date."""
    yy = d.strftime("%y")
    mmm = MONTH_ABBR[d.month]
    dd = f"{d.day:02d}"
    return f"{prefix}-{yy}{mmm}{dd}"


def parse_bucket_from_ticker(ticker: str) -> dict:
    """Parse bucket info from a market ticker string.

    Examples:
        KXHIGHNY-25FEB07-T47   -> below 47F
        KXHIGHNY-25FEB07-B47.5 -> between (47, 49) — need title for exact
        KXHIGHNY-25FEB07-B49.5 -> between (49, 51)
    """
    result = {
        "threshold_low": "",
        "threshold_high": "",
        "direction": "unknown",
        "strike_type": "unknown",
        "bucket": "",
    }

    # The last segment after the date portion gives us the strike
    parts = ticker.split("-")
    if len(parts) < 3:
        return result

    strike = parts[-1]  # e.g., T47, B47.5

    if strike.startswith("T"):
        # Above/below threshold — need market title to distinguish
        try:
            val = float(strike[1:])
            result["threshold_low"] = val
            result["direction"] = "above"
            result["strike_type"] = "greater"
            result["bucket"] = f"Above {val:.0f}F"
        except ValueError:
            pass

    elif strike.startswith("B"):
        try:
            mid = float(strike[1:])
            # "Between" buckets: B47.5 means 47-49, B49.5 means 49-51, etc.
            # The midpoint pattern is X.5 where range is (X, X+2)
            lo = int(mid - 0.5)
            hi = int(mid + 1.5)
            result["threshold_low"] = lo
            result["threshold_high"] = hi
            result["direction"] = "between"
            result["strike_type"] = "between"
            result["bucket"] = f"{lo}-{hi}F"
        except ValueError:
            pass

    return result


def parse_bucket_from_market(market: dict) -> dict:
    """Parse bucket info from market dict, using title/subtitle and ticker."""
    ticker = market.get("ticker", "")
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    yes_sub = market.get("yes_sub_title", "")
    text = f"{title} {subtitle} {yes_sub}".strip()

    result = {
        "threshold_low": "",
        "threshold_high": "",
        "direction": "unknown",
        "strike_type": "unknown",
        "bucket": ticker,
    }

    import re

    # Pattern: "X to Y" or range
    match_range = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)', text
    )
    # Pattern: ">=" or "above"
    match_above = re.search(
        r'(?:>=|≥|[Aa]bove|[Gg]reater\s+than)\s*(\d+(?:\.\d+)?)', text
    )
    # Pattern: "<=" or "below" or "under"
    match_below = re.search(
        r'(?:<=|≤|[Bb]elow|[Ll]ess\s+than|[Uu]nder)\s*(\d+(?:\.\d+)?)', text
    )

    if match_range:
        lo = float(match_range.group(1))
        hi = float(match_range.group(2))
        result["threshold_low"] = min(lo, hi)
        result["threshold_high"] = max(lo, hi)
        result["direction"] = "between"
        result["strike_type"] = "between"
        result["bucket"] = f"{min(lo, hi):.0f}-{max(lo, hi):.0f}F"
    elif match_below:
        val = float(match_below.group(1))
        result["threshold_high"] = val
        result["direction"] = "below"
        result["strike_type"] = "less"
        result["bucket"] = f"Below {val:.0f}F"
    elif match_above:
        val = float(match_above.group(1))
        result["threshold_low"] = val
        result["direction"] = "above"
        result["strike_type"] = "greater"
        result["bucket"] = f"Above {val:.0f}F"
    else:
        # Fall back to ticker parsing
        return parse_bucket_from_ticker(ticker)

    return result


# ===========================================================================
# Core fetching logic
# ===========================================================================

def list_markets_for_event(client: RateLimitedClient, event_ticker: str) -> list:
    """List all markets for a given event ticker."""
    url = f"{KALSHI_BASE_URL}/markets"
    params = {
        "series_ticker": SERIES_TICKER,
        "event_ticker": event_ticker,
        "limit": 50,
    }
    data = client.get(url, params=params)
    if data is None:
        return []
    return data.get("markets", [])


def fetch_candlesticks(
    client: RateLimitedClient,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 60,
) -> list:
    """Fetch candlestick data for a market."""
    url = (
        f"{KALSHI_BASE_URL}/series/{SERIES_TICKER}"
        f"/markets/{ticker}/candlesticks"
    )
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    data = client.get(url, params=params)
    if data is None:
        return []
    return data.get("candlesticks", [])


def get_presettlement_snapshot(
    client: RateLimitedClient,
    ticker: str,
    target_date: date,
) -> dict:
    """Get the last candlestick before market close for a market.

    Market close is ~05:00 UTC on target_date. We look for candles
    from 00:00 UTC to 05:30 UTC on target_date.

    Returns dict with bid/ask/price/volume/oi or None if no data.
    """
    # Market close: ~05:00 UTC on target day
    # Look from 22:00 UTC day before to 05:30 UTC target day
    day_before = target_date - timedelta(days=1)
    start_dt = datetime(day_before.year, day_before.month, day_before.day,
                        22, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(target_date.year, target_date.month, target_date.day,
                      5, 30, 0, tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    candles = fetch_candlesticks(client, ticker, start_ts, end_ts)

    if not candles:
        return None

    # Sort by timestamp descending to get the last candle
    candles_sorted = sorted(candles, key=lambda c: c.get("end_period_ts", 0),
                            reverse=True)

    # Take the last candle (most recent before close)
    last = candles_sorted[0]

    # Extract prices — handle nested dict structure
    bid_close = None
    ask_close = None
    price_close = None

    yes_bid = last.get("yes_bid")
    if isinstance(yes_bid, dict):
        bid_close = yes_bid.get("close")
    elif isinstance(yes_bid, (int, float)):
        bid_close = yes_bid

    yes_ask = last.get("yes_ask")
    if isinstance(yes_ask, dict):
        ask_close = yes_ask.get("close")
    elif isinstance(yes_ask, (int, float)):
        ask_close = yes_ask

    price_data = last.get("price")
    if isinstance(price_data, dict):
        price_close = price_data.get("close")
    elif isinstance(price_data, (int, float)):
        price_close = price_data

    volume = last.get("volume", 0)
    open_interest = last.get("open_interest", 0)
    end_ts_val = last.get("end_period_ts", 0)

    # Compute midpoint probability
    if bid_close is not None and ask_close is not None:
        mid_cents = (bid_close + ask_close) / 2.0
    elif price_close is not None:
        mid_cents = price_close
    elif bid_close is not None:
        mid_cents = bid_close
    elif ask_close is not None:
        mid_cents = ask_close
    else:
        mid_cents = None

    presettlement_prob = mid_cents / 100.0 if mid_cents is not None else None

    # Convert timestamp to readable UTC time
    if end_ts_val:
        snapshot_time = datetime.fromtimestamp(
            end_ts_val, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        snapshot_time = ""

    return {
        "presettlement_prob": presettlement_prob,
        "bid_cents": bid_close,
        "ask_cents": ask_close,
        "volume": volume,
        "open_interest": open_interest,
        "snapshot_time_utc": snapshot_time,
    }


def process_single_date(
    client: RateLimitedClient,
    target_date: date,
) -> list:
    """Fetch pre-settlement data for all markets on a given date.

    Tries both KXHIGHNY and HIGHNY prefixes.

    Returns list of row dicts for the CSV.
    """
    rows = []
    markets = []

    # Try each prefix until we find markets
    for prefix in TICKER_PREFIXES:
        event_ticker = make_event_ticker(target_date, prefix)
        markets = list_markets_for_event(client, event_ticker)
        if markets:
            logger.info(
                "Found %d markets for %s (prefix=%s)",
                len(markets), target_date, prefix,
            )
            break
        logger.debug("No markets found with prefix %s for %s", prefix, target_date)

    if not markets:
        logger.warning("No markets found for %s with any prefix", target_date)
        return rows

    for market in markets:
        ticker = market.get("ticker", "")
        bucket_info = parse_bucket_from_market(market)

        snapshot = get_presettlement_snapshot(client, ticker, target_date)

        row = {
            "date": target_date.isoformat(),
            "ticker": ticker,
            "bucket": bucket_info["bucket"],
            "threshold_low": bucket_info["threshold_low"],
            "threshold_high": bucket_info["threshold_high"],
            "direction": bucket_info["direction"],
            "strike_type": bucket_info["strike_type"],
            "presettlement_prob": snapshot["presettlement_prob"] if snapshot else "",
            "bid_cents": snapshot["bid_cents"] if snapshot else "",
            "ask_cents": snapshot["ask_cents"] if snapshot else "",
            "volume": snapshot["volume"] if snapshot else "",
            "open_interest": snapshot["open_interest"] if snapshot else "",
            "snapshot_time_utc": snapshot["snapshot_time_utc"] if snapshot else "",
        }
        rows.append(row)

    return rows


# ===========================================================================
# Checkpointing
# ===========================================================================

def load_checkpoint() -> set:
    """Load set of already-processed dates from checkpoint file."""
    processed = set()
    if CHECKPOINT_CSV.exists():
        with open(CHECKPOINT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["date"])
    return processed


def save_checkpoint(all_rows: list):
    """Save all accumulated rows to checkpoint file."""
    if not all_rows:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Checkpoint saved: %d rows to %s", len(all_rows), CHECKPOINT_CSV)


def load_checkpoint_rows() -> list:
    """Load all rows from checkpoint file."""
    rows = []
    if CHECKPOINT_CSV.exists():
        with open(CHECKPOINT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def save_final_output(all_rows: list):
    """Save final output CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Final output saved: %d rows to %s", len(all_rows), OUTPUT_CSV)


# ===========================================================================
# Date range generation
# ===========================================================================

def generate_date_range(start: date, end: date) -> list:
    """Generate list of dates from start to end (inclusive)."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi pre-settlement market prices for KXHIGHNY"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: fetch only 3 recent dates (2025-02-07 to 2025-02-09)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD). Default: 2023-01-01",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: 2025-12-31",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint file",
    )
    args = parser.parse_args()

    # Determine date range
    if args.test:
        start_date = date(2025, 2, 7)
        end_date = date(2025, 2, 9)
        logger.info("TEST MODE: fetching %s to %s", start_date, end_date)
    else:
        start_date = (
            datetime.strptime(args.start, "%Y-%m-%d").date()
            if args.start else date(2023, 1, 1)
        )
        end_date = (
            datetime.strptime(args.end, "%Y-%m-%d").date()
            if args.end else date(2025, 12, 31)
        )

    all_dates = generate_date_range(start_date, end_date)
    total_dates = len(all_dates)
    logger.info("Date range: %s to %s (%d dates)", start_date, end_date, total_dates)

    # Load checkpoint if resuming
    all_rows = []
    processed_dates = set()
    if args.resume:
        all_rows = load_checkpoint_rows()
        processed_dates = load_checkpoint()
        logger.info(
            "Resuming: %d dates already processed (%d rows)",
            len(processed_dates), len(all_rows),
        )

    # Create client
    client = RateLimitedClient(min_interval=MIN_REQUEST_INTERVAL)

    # Process dates
    dates_done = 0
    dates_with_data = 0
    dates_no_data = 0
    total_markets = 0
    start_time = time.time()

    for i, target_date in enumerate(all_dates):
        date_str = target_date.isoformat()

        # Skip already-processed dates
        if date_str in processed_dates:
            dates_done += 1
            continue

        try:
            rows = process_single_date(client, target_date)

            if rows:
                all_rows.extend(rows)
                dates_with_data += 1
                total_markets += len(rows)
            else:
                dates_no_data += 1

            processed_dates.add(date_str)
            dates_done += 1

            # Progress logging
            elapsed = time.time() - start_time
            remaining_dates = total_dates - dates_done - len(
                [d for d in all_dates[:i] if d.isoformat() in processed_dates]
            )
            if dates_done > 0 and elapsed > 0:
                rate = elapsed / max(dates_done - len(load_checkpoint()), 1)
                eta_seconds = remaining_dates * rate
                eta_min = eta_seconds / 60.0
                logger.info(
                    "Progress: %d/%d dates | %d with data | %d markets | "
                    "ETA: %.1f min",
                    i + 1, total_dates, dates_with_data, total_markets,
                    eta_min,
                )

            # Checkpoint
            if dates_done % CHECKPOINT_INTERVAL == 0 and dates_done > 0:
                save_checkpoint(all_rows)

        except KeyboardInterrupt:
            logger.info("Interrupted! Saving checkpoint...")
            save_checkpoint(all_rows)
            sys.exit(0)

        except Exception as exc:
            logger.error("Error processing %s: %s", target_date, exc)
            # Save checkpoint on error and continue
            save_checkpoint(all_rows)
            continue

    # Save final output
    save_final_output(all_rows)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("  Dates processed: %d", dates_done)
    logger.info("  Dates with data: %d", dates_with_data)
    logger.info("  Dates with no data: %d", dates_no_data)
    logger.info("  Total market rows: %d", len(all_rows))
    logger.info("  Elapsed time: %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Output: %s", OUTPUT_CSV)
    logger.info("=" * 60)

    # Clean up checkpoint if we completed successfully
    if CHECKPOINT_CSV.exists() and dates_done == total_dates:
        CHECKPOINT_CSV.unlink()
        logger.info("Checkpoint file removed (run complete)")

    # Print time estimate for full run if in test mode
    if args.test and dates_with_data > 0:
        avg_time_per_date = elapsed / max(dates_with_data, 1)
        full_dates = 365 * 3  # ~3 years
        est_full_time = full_dates * avg_time_per_date
        logger.info("")
        logger.info("TIME ESTIMATE FOR FULL RUN:")
        logger.info("  Avg time per date: %.1f seconds", avg_time_per_date)
        logger.info("  Full run (~%d dates): %.1f min (%.1f hours)",
                     full_dates, est_full_time / 60, est_full_time / 3600)


if __name__ == "__main__":
    main()
