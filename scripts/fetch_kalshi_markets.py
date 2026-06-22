#!/usr/bin/env python3
"""
Fetch real Kalshi KXHIGHNY market data for 2023-2025.

Uses the Kalshi public API to download all settled KXHIGHNY contracts,
parse market structure (threshold, settlement, last price), and save
as CSV files for backtesting.

Market structure (per day):
  - "less" bucket: "Will high temp be < X?"  (1 per day)
  - "between" buckets: "Will high temp be Y-Z?" (4-5 per day, 2-degree width)
  - "greater" bucket: "Will high temp be > X?" (1 per day)

Outputs:
  - data/real_kalshi_2023_2024.csv
  - data/real_kalshi_2025.csv
"""

import os
import sys
import re
import time
import json
import logging
from datetime import datetime, date

import requests
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
RATE_LIMIT_RPS = 10


def fetch_all_settled_markets(series_ticker="KXHIGHNY", max_pages=500):
    """Fetch all settled KXHIGHNY markets using cursor-based pagination."""
    url = f"{KALSHI_BASE_URL}/markets"
    all_markets = []
    cursor = None
    page = 0
    min_interval = 1.0 / RATE_LIMIT_RPS

    while page < max_pages:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        time.sleep(min_interval)

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error("Request failed on page %d: %s", page, e)
            break

        markets = data.get("markets", [])
        all_markets.extend(markets)
        page += 1

        cursor = data.get("cursor")
        if not cursor or len(markets) < 200:
            break

        if page % 10 == 0:
            logger.info("  Fetched %d markets so far (page %d)...",
                        len(all_markets), page)

    logger.info("Fetched %d total settled markets across %d pages",
                len(all_markets), page)
    return all_markets


def parse_event_date(event_ticker):
    """Parse the date from an event ticker like 'KXHIGHNY-24DEC31' or 'HIGHNY-24DEC31'."""
    m = re.search(r"(?:KX)?HIGHNY-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None

    yy = int(m.group(1))
    month_str = m.group(2)
    dd = int(m.group(3))

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = month_map.get(month_str)
    if month is None:
        return None

    year = 2000 + yy
    try:
        return date(year, month, dd)
    except ValueError:
        return None


def parse_threshold_from_ticker(ticker):
    """Parse threshold from ticker string.

    Examples:
      - HIGHNY-23DEC31-T48    -> ('greater', 48.0)
      - HIGHNY-23DEC31-B47.5  -> ('between', 47.5)
      - HIGHNY-23DEC31-T41    -> could be 'less' depending on strike_type
    """
    m = re.search(r"-([TB])(\d+\.?\d*)", ticker)
    if m:
        letter = m.group(1)
        val = float(m.group(2))
        return letter, val
    return None, None


def parse_market_to_row(market):
    """Parse a single Kalshi market dict into a standardized row."""
    ticker = market.get("ticker", "")
    event_ticker = market.get("event_ticker", "")

    # Parse date from event ticker
    mkt_date = parse_event_date(event_ticker)
    if mkt_date is None:
        return None

    strike_type = market.get("strike_type", "")
    floor_strike = market.get("floor_strike")

    # Parse threshold from ticker
    letter, ticker_val = parse_threshold_from_ticker(ticker)

    # Determine direction and thresholds
    if strike_type == "greater":
        direction = "above"
        threshold = float(floor_strike) if floor_strike is not None else ticker_val
        if threshold is None:
            return None
        threshold_low = threshold
        threshold_high = np.nan
        bucket = f"Above {threshold:.0f}F"

    elif strike_type == "less":
        direction = "below"
        # For "less", floor_strike is often None. Use ticker value.
        if ticker_val is not None:
            threshold = ticker_val
        elif floor_strike is not None:
            threshold = float(floor_strike)
        else:
            return None
        threshold_low = np.nan
        threshold_high = threshold
        bucket = f"Below {threshold:.0f}F"

    elif strike_type == "between":
        direction = "between"
        # For "between", floor_strike is the lower bound of a 2-degree bucket
        # Title says "Will temp be Y-Z?" where Y=floor_strike, Z=floor_strike+1
        if floor_strike is not None:
            low = float(floor_strike)
        elif ticker_val is not None:
            # B47.5 means between 47-48, so floor is 47
            low = float(int(ticker_val))
        else:
            return None
        high = low + 2  # 2-degree buckets
        threshold = low
        threshold_low = low
        threshold_high = high
        bucket = f"{low:.0f}-{high:.0f}F"
    else:
        # Unknown strike type, skip
        return None

    # Settlement result
    result = market.get("result", "")
    if result == "yes":
        actual_outcome = 1
    elif result == "no":
        actual_outcome = 0
    else:
        actual_outcome = None

    # Last price (in cents)
    last_price = market.get("last_price")
    if last_price is not None:
        market_prob = float(last_price) / 100.0
    else:
        market_prob = np.nan

    # Actual temperature from expiration_value (only on winning bucket)
    exp_value = market.get("expiration_value")
    actual_tmax = None
    if exp_value is not None and exp_value != "":
        try:
            actual_tmax = float(exp_value)
        except (ValueError, TypeError):
            pass

    # Volume
    volume = market.get("volume", 0)

    # Bid/ask
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")

    return {
        "date": mkt_date,
        "ticker": ticker,
        "event_ticker": event_ticker,
        "bucket": bucket,
        "threshold": threshold,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "direction": direction,
        "strike_type": strike_type,
        "market_prob": market_prob,
        "actual_outcome": actual_outcome,
        "actual_tmax": actual_tmax,
        "result": result,
        "settlement_value": market.get("settlement_value"),
        "volume": volume,
        "bid_price": yes_bid / 100.0 if yes_bid is not None else np.nan,
        "ask_price": yes_ask / 100.0 if yes_ask is not None else np.nan,
    }


def fill_actual_tmax_from_ghcn(df, ghcn_path):
    """Fill missing actual_tmax from GHCN Central Park data.

    The Kalshi API only includes expiration_value on the winning bucket.
    Use real GHCN TMAX data to fill all rows.
    """
    ghcn = pd.read_csv(ghcn_path)
    ghcn["date"] = pd.to_datetime(ghcn["date"]).dt.date
    ghcn_map = dict(zip(ghcn["date"], ghcn["actual_tmax"]))

    filled = 0
    for idx, row in df.iterrows():
        d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
        if pd.isna(row["actual_tmax"]) or row["actual_tmax"] is None:
            ghcn_tmax = ghcn_map.get(d)
            if ghcn_tmax is not None:
                df.at[idx, "actual_tmax"] = ghcn_tmax
                filled += 1

    logger.info("Filled %d actual_tmax values from GHCN data", filled)
    return df


def recalculate_outcomes_from_actual(df):
    """Recalculate actual_outcome based on actual_tmax and bucket thresholds.

    This ensures outcomes are consistent with the actual temperature,
    even if the API result field is ambiguous for some rows.
    """
    recalc = 0
    for idx, row in df.iterrows():
        tmax = row["actual_tmax"]
        if pd.isna(tmax):
            continue

        direction = row["direction"]
        low = row["threshold_low"]
        high = row["threshold_high"]

        if direction == "above":
            expected = 1 if tmax > low else 0
        elif direction == "below":
            expected = 1 if tmax < high else 0
        elif direction == "between":
            expected = 1 if (low <= tmax < high) else 0
        else:
            continue

        if row["actual_outcome"] != expected:
            recalc += 1
        df.at[idx, "actual_outcome"] = expected

    logger.info("Recalculated %d outcome values based on actual TMAX", recalc)
    return df


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    raw_dir = os.path.join(data_dir, "raw", "kalshi_api")
    os.makedirs(raw_dir, exist_ok=True)

    # Path to GHCN Central Park data
    ghcn_path = os.path.join(data_dir, "nyc", "real_central_park_tmax_2023_2025.csv")

    logger.info("=" * 60)
    logger.info("Fetching all settled KXHIGHNY markets from Kalshi API...")
    logger.info("=" * 60)

    # Check if we already have cached API data
    raw_path = os.path.join(raw_dir, "all_settled_markets.json")
    if os.path.exists(raw_path):
        logger.info("Loading cached API response from %s", raw_path)
        with open(raw_path) as f:
            all_markets = json.load(f)
        logger.info("Loaded %d cached markets", len(all_markets))
    else:
        all_markets = fetch_all_settled_markets()
        with open(raw_path, "w") as f:
            json.dump(all_markets, f, indent=2, default=str)
        logger.info("Saved raw API response: %s (%d markets)", raw_path, len(all_markets))

    # Parse all markets
    rows = []
    skipped = 0
    for market in all_markets:
        row = parse_market_to_row(market)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1

    logger.info("Parsed %d markets, skipped %d", len(rows), skipped)

    if not rows:
        logger.error("No markets parsed!")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "threshold"]).reset_index(drop=True)

    logger.info("Total parsed markets: %d", len(df))
    logger.info("Date range: %s to %s", df["date"].min(), df["date"].max())
    logger.info("Unique dates: %d", df["date"].nunique())

    # Year distribution
    logger.info("Year distribution:")
    for year in sorted(df["date"].dt.year.unique()):
        mask = df["date"].dt.year == year
        n_contracts = mask.sum()
        n_days = df[mask]["date"].nunique()
        logger.info("  %d: %d contracts across %d days", year, n_contracts, n_days)

    # Fill actual_tmax from GHCN data
    if os.path.exists(ghcn_path):
        df = fill_actual_tmax_from_ghcn(df, ghcn_path)
    else:
        logger.warning("GHCN data not found at %s -- actual_tmax may be incomplete", ghcn_path)

    # For rows still missing actual_tmax, propagate from winning bucket within same date
    for d in df["date"].unique():
        mask = df["date"] == d
        day_df = df[mask]
        tmax_vals = day_df["actual_tmax"].dropna()
        if len(tmax_vals) > 0:
            df.loc[mask, "actual_tmax"] = tmax_vals.iloc[0]

    # Note: We do NOT recalculate outcomes from GHCN actual_tmax because
    # Kalshi settles based on the NWS Climatological Report, which can
    # differ slightly from GHCN. The API result field is the ground truth
    # for what would actually settle on Kalshi.
    # We keep actual_tmax from GHCN for model input purposes only.

    # Drop rows with missing critical data
    before = len(df)
    df = df.dropna(subset=["market_prob", "actual_outcome"]).reset_index(drop=True)
    logger.info("Dropped %d rows with missing market_prob or actual_outcome", before - len(df))

    # Filter out zero-volume markets (no meaningful price signal)
    zero_vol = (df["volume"] == 0).sum()
    logger.info("Markets with zero volume: %d (%.1f%%) -- keeping for completeness",
                zero_vol, 100 * zero_vol / len(df))

    # Split by year
    mask_2023_2024 = df["date"].dt.year.isin([2023, 2024])
    mask_2025 = df["date"].dt.year == 2025

    df_2023_2024 = df[mask_2023_2024].copy().reset_index(drop=True)
    df_2025 = df[mask_2025].copy().reset_index(drop=True)

    # Save CSVs
    path_2023_2024 = os.path.join(data_dir, "real_kalshi_2023_2024.csv")
    path_2025 = os.path.join(data_dir, "real_kalshi_2025.csv")

    if len(df_2023_2024) > 0:
        df_2023_2024.to_csv(path_2023_2024, index=False)
        logger.info("Saved 2023-2024: %s (%d rows, %d days)",
                     path_2023_2024, len(df_2023_2024), df_2023_2024["date"].nunique())
        # Quick stats
        logger.info("  actual_tmax filled: %d / %d",
                     df_2023_2024["actual_tmax"].notna().sum(), len(df_2023_2024))
        logger.info("  market_prob range: %.3f to %.3f",
                     df_2023_2024["market_prob"].min(), df_2023_2024["market_prob"].max())
        logger.info("  direction breakdown:")
        for d, c in df_2023_2024["direction"].value_counts().items():
            logger.info("    %s: %d", d, c)
    else:
        logger.warning("No 2023-2024 data found!")

    if len(df_2025) > 0:
        df_2025.to_csv(path_2025, index=False)
        logger.info("Saved 2025: %s (%d rows, %d days)",
                     path_2025, len(df_2025), df_2025["date"].nunique())
    else:
        logger.warning("No 2025 data found!")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Kalshi Market Data Summary")
    logger.info("  Total contracts: %d", len(df))
    logger.info("  2023-2024 contracts: %d (%d days)",
                len(df_2023_2024),
                df_2023_2024["date"].nunique() if len(df_2023_2024) else 0)
    logger.info("  2025 contracts: %d (%d days)",
                len(df_2025),
                df_2025["date"].nunique() if len(df_2025) else 0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
