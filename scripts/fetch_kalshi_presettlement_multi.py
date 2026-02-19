#!/usr/bin/env python3
"""Fetch Kalshi pre-settlement market prices for CHI, PHL, and AUS weather markets.

Uses the existing settlement CSV files to know which dates/tickers to query,
then fetches 1-hour candlestick data near market close for each market to
get pre-settlement price snapshots.

Usage:
    # Fetch both cities:
    python scripts/fetch_kalshi_presettlement_multi.py

    # Fetch one city:
    python scripts/fetch_kalshi_presettlement_multi.py --city chi

    # Test mode (3 dates per city):
    python scripts/fetch_kalshi_presettlement_multi.py --test

    # Resume from checkpoint:
    python scripts/fetch_kalshi_presettlement_multi.py --resume
"""

import argparse
import csv
import os
import sys
import time
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pandas as pd
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

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Rate limiting: 0.5s between requests (tested to avoid 429)
MIN_REQUEST_INTERVAL = 0.5

# Retry settings
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 2.0

# Checkpoint every N dates
CHECKPOINT_INTERVAL = 20

# City configurations
CITY_CONFIGS = {
    "chi": {
        "settlement_csv": DATA_DIR / "real_kalshi_chi_all.csv",
        "output_csv": DATA_DIR / "kalshi_presettlement_chi.csv",
        "checkpoint_csv": DATA_DIR / "kalshi_presettlement_chi_checkpoint.csv",
        "series_tickers": ["KXHIGHCHI", "HIGHCHI"],
    },
    "phl": {
        "settlement_csv": DATA_DIR / "real_kalshi_phl_all.csv",
        "output_csv": DATA_DIR / "kalshi_presettlement_phl.csv",
        "checkpoint_csv": DATA_DIR / "kalshi_presettlement_phl_checkpoint.csv",
        "series_tickers": ["KXHIGHPHIL"],
    },
    "aus": {
        "settlement_csv": DATA_DIR / "real_kalshi_aus_all.csv",
        "output_csv": DATA_DIR / "kalshi_presettlement_aus.csv",
        "checkpoint_csv": DATA_DIR / "kalshi_presettlement_aus_checkpoint.csv",
        "series_tickers": ["KXHIGHAUS", "HIGHAUS"],
    },
    "atl": {
        "settlement_csv": DATA_DIR / "real_kalshi_atl_all.csv",
        "output_csv": DATA_DIR / "kalshi_presettlement_atl.csv",
        "checkpoint_csv": DATA_DIR / "kalshi_presettlement_atl_checkpoint.csv",
        "series_tickers": ["KXHIGHTATL", "HIGHTATL"],
    },
}

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
            "User-Agent": "Weather-Prediction/1.0",
        })

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, params: dict = None) -> dict:
        """GET request with retry and rate limiting."""
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                resp = self._session.get(url, params=params, timeout=30)

                if resp.status_code == 404:
                    return None

                if resp.status_code == 429:
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
                err_str = str(exc).lower()
                if "refused" in err_str or "ssl" in err_str:
                    logger.error("NETWORK ACCESS BLOCKED: %s", url)
                    sys.exit(1)

                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "Request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)

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
# Core fetching logic
# ===========================================================================

def get_presettlement_snapshot(
    client: RateLimitedClient,
    ticker: str,
    series_ticker: str,
    target_date: date,
) -> dict:
    """Get the last candlestick before market close for a market.

    Market close is ~05:00 UTC on target_date. We look for candles
    from 22:00 UTC day before to 05:30 UTC target day.
    """
    day_before = target_date - timedelta(days=1)
    start_dt = datetime(day_before.year, day_before.month, day_before.day,
                        22, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(target_date.year, target_date.month, target_date.day,
                      5, 30, 0, tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    url = (
        f"{KALSHI_BASE_URL}/series/{series_ticker}"
        f"/markets/{ticker}/candlesticks"
    )
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 60,
    }
    data = client.get(url, params=params)

    if data is None:
        return None

    candles = data.get("candlesticks", [])
    if not candles:
        return None

    # Sort by timestamp descending to get the last candle
    candles_sorted = sorted(candles, key=lambda c: c.get("end_period_ts", 0),
                            reverse=True)
    last = candles_sorted[0]

    # Extract prices
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


def load_settlement_data(city_code: str) -> pd.DataFrame:
    """Load settlement CSV and extract unique dates + tickers."""
    config = CITY_CONFIGS[city_code]
    df = pd.read_csv(config["settlement_csv"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def merge_presettlement_with_settlement(presettlement_rows: list,
                                         settlement_df: pd.DataFrame) -> pd.DataFrame:
    """Merge presettlement data with settlement ground truth.

    CRITICAL: Always use settlement data's bucket definitions (threshold_low,
    threshold_high, direction, etc.) since API title parsing may give wrong widths.
    """
    if not presettlement_rows:
        return pd.DataFrame(columns=CSV_COLUMNS)

    pre_df = pd.DataFrame(presettlement_rows)

    # Merge with settlement data to get correct bucket definitions
    merged = settlement_df.merge(
        pre_df[["date", "ticker", "presettlement_prob", "bid_cents",
                "ask_cents", "volume_pre", "open_interest_pre", "snapshot_time_utc"]],
        on=["date", "ticker"],
        how="inner",
    )

    # Build output using settlement bucket definitions
    result = pd.DataFrame({
        "date": merged["date"],
        "ticker": merged["ticker"],
        "bucket": merged["bucket"],
        "threshold_low": merged["threshold_low"],
        "threshold_high": merged["threshold_high"],
        "direction": merged["direction"],
        "strike_type": merged["strike_type"],
        "presettlement_prob": merged["presettlement_prob"],
        "bid_cents": merged["bid_cents"],
        "ask_cents": merged["ask_cents"],
        "volume": merged["volume_pre"],
        "open_interest": merged["open_interest_pre"],
        "snapshot_time_utc": merged["snapshot_time_utc"],
    })

    return result


# ===========================================================================
# Checkpointing
# ===========================================================================

def load_checkpoint(checkpoint_path: Path) -> tuple:
    """Load checkpoint: returns (processed_dates set, list of row dicts)."""
    processed = set()
    rows = []
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                processed.add(row["date"])
    return processed, rows


def save_checkpoint(checkpoint_path: Path, rows: list):
    """Save rows to checkpoint file."""
    if not rows:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Checkpoint saved: %d rows to %s", len(rows), checkpoint_path)


def save_final_output(output_path: Path, rows: list):
    """Save final output CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Final output saved: %d rows to %s", len(rows), output_path)


# ===========================================================================
# Per-city fetch logic
# ===========================================================================

def fetch_city(city_code: str, client: RateLimitedClient,
               resume: bool = False, test: bool = False):
    """Fetch all presettlement data for a city."""
    config = CITY_CONFIGS[city_code]
    settlement_df = load_settlement_data(city_code)

    # Get unique dates and tickers from settlement data
    dates_tickers = settlement_df[["date", "ticker", "event_ticker"]].drop_duplicates()
    unique_dates = sorted(dates_tickers["date"].unique())

    if test:
        unique_dates = unique_dates[-3:]
        logger.info("TEST MODE: using last 3 dates: %s", unique_dates)

    total_dates = len(unique_dates)
    logger.info("City %s: %d unique dates to process", city_code.upper(), total_dates)

    # Figure out which series ticker to use for the API
    # The series_ticker in the URL must match the actual prefix
    # We'll determine it from the event_ticker
    def get_series_ticker(event_ticker: str) -> str:
        """Extract series ticker from event ticker (e.g., HIGHCHI-22DEC22 -> HIGHCHI)."""
        return event_ticker.split("-")[0]

    # Load checkpoint
    all_rows = []
    processed_dates = set()
    if resume:
        processed_dates, all_rows = load_checkpoint(config["checkpoint_csv"])
        logger.info("Resuming: %d dates already processed (%d rows)",
                     len(processed_dates), len(all_rows))

    dates_done = 0
    dates_with_data = 0
    total_markets = 0
    start_time = time.time()

    for i, date_str in enumerate(unique_dates):
        if date_str in processed_dates:
            dates_done += 1
            continue

        try:
            # Get all tickers for this date
            date_rows = dates_tickers[dates_tickers["date"] == date_str]
            tickers = date_rows["ticker"].tolist()
            event_tickers = date_rows["event_ticker"].tolist()

            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            date_has_data = False

            for ticker, event_ticker in zip(tickers, event_tickers):
                series_ticker = get_series_ticker(event_ticker)

                snapshot = get_presettlement_snapshot(
                    client, ticker, series_ticker, target_date
                )

                if snapshot and snapshot["presettlement_prob"] is not None:
                    # Get bucket info from settlement data
                    settle_row = settlement_df[
                        (settlement_df["date"] == date_str) &
                        (settlement_df["ticker"] == ticker)
                    ].iloc[0]

                    row = {
                        "date": date_str,
                        "ticker": ticker,
                        "bucket": settle_row["bucket"],
                        "threshold_low": settle_row["threshold_low"],
                        "threshold_high": settle_row["threshold_high"],
                        "direction": settle_row["direction"],
                        "strike_type": settle_row["strike_type"],
                        "presettlement_prob": snapshot["presettlement_prob"],
                        "bid_cents": snapshot["bid_cents"],
                        "ask_cents": snapshot["ask_cents"],
                        "volume": snapshot["volume"],
                        "open_interest": snapshot["open_interest"],
                        "snapshot_time_utc": snapshot["snapshot_time_utc"],
                    }
                    all_rows.append(row)
                    total_markets += 1
                    date_has_data = True

            if date_has_data:
                dates_with_data += 1

            processed_dates.add(date_str)
            dates_done += 1

            # Progress logging every 10 dates
            if dates_done % 10 == 0:
                elapsed = time.time() - start_time
                remaining = total_dates - dates_done
                if dates_done > 0:
                    rate = elapsed / dates_done
                    eta_min = (remaining * rate) / 60.0
                    logger.info(
                        "[%s] Progress: %d/%d dates | %d with data | %d markets | "
                        "ETA: %.1f min",
                        city_code.upper(), dates_done, total_dates,
                        dates_with_data, total_markets, eta_min,
                    )

            # Checkpoint
            if dates_done % CHECKPOINT_INTERVAL == 0 and dates_done > 0:
                save_checkpoint(config["checkpoint_csv"], all_rows)

        except KeyboardInterrupt:
            logger.info("Interrupted! Saving checkpoint...")
            save_checkpoint(config["checkpoint_csv"], all_rows)
            sys.exit(0)

        except Exception as exc:
            logger.error("Error processing %s/%s: %s", city_code, date_str, exc)
            save_checkpoint(config["checkpoint_csv"], all_rows)
            continue

    # Save final output
    save_final_output(config["output_csv"], all_rows)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("[%s] FETCH COMPLETE", city_code.upper())
    logger.info("  Dates processed: %d", dates_done)
    logger.info("  Dates with data: %d", dates_with_data)
    logger.info("  Total market rows: %d", len(all_rows))
    logger.info("  Elapsed time: %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Output: %s", config["output_csv"])
    logger.info("=" * 60)

    # Clean up checkpoint
    if config["checkpoint_csv"].exists():
        config["checkpoint_csv"].unlink()
        logger.info("Checkpoint removed (run complete)")

    return len(all_rows), dates_with_data


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi pre-settlement prices for CHI and PHL"
    )
    parser.add_argument(
        "--city", type=str, choices=["chi", "phl", "aus", "atl"], default=None,
        help="Fetch only one city (default: both)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: last 3 dates per city",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint",
    )
    args = parser.parse_args()

    cities = [args.city] if args.city else ["phl", "chi", "aus"]
    client = RateLimitedClient(min_interval=MIN_REQUEST_INTERVAL)

    for city in cities:
        logger.info("=" * 60)
        logger.info("Starting fetch for %s", city.upper())
        logger.info("=" * 60)
        n_rows, n_dates = fetch_city(city, client,
                                      resume=args.resume, test=args.test)
        logger.info("Done with %s: %d rows, %d dates with data\n",
                     city.upper(), n_rows, n_dates)


if __name__ == "__main__":
    main()
