#!/usr/bin/env python3
"""
Fetch real Kalshi settled market data for Chicago, Philadelphia, and Austin.

Uses the Kalshi public API to download all settled contracts for:
  - KXHIGHCHI (Chicago high temperature)
  - KXHIGHPHL (Philadelphia high temperature)

Market structure (per day):
  - "less" bucket: "Will high temp be < X?"  (1 per day)
  - "between" buckets: "Will high temp be Y-Z?" (4-5 per day, 2-degree width)
  - "greater" bucket: "Will high temp be > X?" (1 per day)

Outputs per city:
  - data/real_kalshi_{city}_all.csv  (all settled contracts)

Usage:
    python scripts/fetch_kalshi_multi_city.py
    python scripts/fetch_kalshi_multi_city.py --city chi
    python scripts/fetch_kalshi_multi_city.py --city phl
    python scripts/fetch_kalshi_multi_city.py --city aus
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
RATE_LIMIT_RPS = 10

CITY_CONFIG = {
    "chi": {
        "series_ticker": "KXHIGHCHI",
        "ticker_patterns": ["HIGHCHI", "KXHIGHCHI"],
        "target_station": "USW00094846",
        "data_subdir": "chicago",
        "ghcn_col": "TMAX",
        "label": "Chicago",
    },
    "phl": {
        "series_ticker": "KXHIGHPHIL",
        "ticker_patterns": ["HIGHPHIL", "KXHIGHPHIL"],
        "target_station": "USW00013739",
        "data_subdir": "philadelphia",
        "ghcn_col": "TMAX",
        "label": "Philadelphia",
    },
    "aus": {
        "series_ticker": "KXHIGHAUS",
        "ticker_patterns": ["HIGHAUS", "KXHIGHAUS"],
        "target_station": "USW00013904",
        "data_subdir": "austin",
        "ghcn_col": "TMAX",
        "label": "Austin",
    },
}


def fetch_all_settled_markets(series_ticker, max_pages=500):
    """Fetch all settled markets for a given series using cursor-based pagination."""
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

        retries = 0
        max_retries = 4
        while retries < max_retries:
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                if retries < max_retries:
                    logger.warning("Request failed (attempt %d/%d): %s. Retrying in %ds...",
                                   retries, max_retries, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("Request failed after %d retries: %s", max_retries, e)
                    return all_markets

        markets = data.get("markets", [])
        all_markets.extend(markets)
        page += 1

        cursor = data.get("cursor")
        if not cursor or len(markets) < 200:
            break

        if page % 10 == 0:
            logger.info("  Fetched %d markets so far (page %d)...",
                        len(all_markets), page)

    logger.info("Fetched %d total settled markets across %d pages for %s",
                len(all_markets), page, series_ticker)
    return all_markets


def parse_event_date(event_ticker, city_patterns):
    """Parse date from an event ticker for any city.

    Supports patterns like:
      - KXHIGHCHI-24DEC31
      - HIGHPHL-25JAN15
    """
    for pattern in city_patterns:
        regex = rf"(?:KX)?{re.escape(pattern)}-(\d{{2}})([A-Z]{{3}})(\d{{2}})"
        m = re.search(regex, event_ticker)
        if m:
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
    return None


def parse_market_to_row(market, city_patterns):
    """Parse a single Kalshi market dict into a standardized row."""
    ticker = market.get("ticker", "")
    event_ticker = market.get("event_ticker", "")

    mkt_date = parse_event_date(event_ticker, city_patterns)
    if mkt_date is None:
        return None

    strike_type = market.get("strike_type", "")
    floor_strike = market.get("floor_strike")

    # Parse threshold from ticker
    m = re.search(r"-([TB])(\d+\.?\d*)", ticker)
    if m:
        letter = m.group(1)
        ticker_val = float(m.group(2))
    else:
        letter, ticker_val = None, None

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
        if floor_strike is not None:
            low = float(floor_strike)
        elif ticker_val is not None:
            low = float(int(ticker_val))
        else:
            return None
        high = low + 2  # 2-degree buckets
        threshold = low
        threshold_low = low
        threshold_high = high
        bucket = f"{low:.0f}-{high:.0f}F"
    else:
        return None

    result = market.get("result", "")
    actual_outcome = 1 if result == "yes" else (0 if result == "no" else None)

    last_price = market.get("last_price")
    market_prob = float(last_price) / 100.0 if last_price is not None else np.nan

    exp_value = market.get("expiration_value")
    actual_tmax = None
    if exp_value is not None and exp_value != "":
        try:
            actual_tmax = float(exp_value)
        except (ValueError, TypeError):
            pass

    volume = market.get("volume", 0)
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


def fill_actual_tmax_from_ghcn(df, ghcn_csv_path, tmax_col="TMAX"):
    """Fill missing actual_tmax from GHCN station CSV.

    Expects CSV with 'DATE' or 'date' and tmax_col columns.
    """
    if not os.path.exists(ghcn_csv_path):
        logger.warning("GHCN file not found: %s", ghcn_csv_path)
        return df

    ghcn = pd.read_csv(ghcn_csv_path)

    # Normalize column names
    if "DATE" in ghcn.columns:
        ghcn = ghcn.rename(columns={"DATE": "date"})

    ghcn["date"] = pd.to_datetime(ghcn["date"]).dt.date

    if tmax_col not in ghcn.columns:
        logger.warning("Column %s not found in GHCN data. Available: %s",
                        tmax_col, ghcn.columns.tolist())
        return df

    ghcn_map = dict(zip(ghcn["date"], ghcn[tmax_col]))

    filled = 0
    for idx, row in df.iterrows():
        d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
        if pd.isna(row["actual_tmax"]) or row["actual_tmax"] is None:
            ghcn_tmax = ghcn_map.get(d)
            if ghcn_tmax is not None and not pd.isna(ghcn_tmax):
                df.at[idx, "actual_tmax"] = float(ghcn_tmax)
                filled += 1

    logger.info("Filled %d actual_tmax values from GHCN data", filled)
    return df


def fetch_city(city_code):
    """Download and save all settled Kalshi contracts for a city."""
    cfg = CITY_CONFIG[city_code]
    series_ticker = cfg["series_ticker"]
    city_patterns = cfg["ticker_patterns"]
    label = cfg["label"]

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    raw_dir = os.path.join(data_dir, "raw", "kalshi_api")
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Fetching %s settled markets (%s)...", label, series_ticker)
    logger.info("=" * 60)

    # Check cache
    raw_path = os.path.join(raw_dir, f"settled_{city_code}.json")
    if os.path.exists(raw_path):
        logger.info("Loading cached API response from %s", raw_path)
        with open(raw_path) as f:
            all_markets = json.load(f)
        logger.info("Loaded %d cached markets", len(all_markets))
    else:
        all_markets = fetch_all_settled_markets(series_ticker)
        if all_markets:
            with open(raw_path, "w") as f:
                json.dump(all_markets, f, indent=2, default=str)
            logger.info("Saved raw API response: %s (%d markets)", raw_path, len(all_markets))

    if not all_markets:
        logger.warning("No settled markets found for %s", series_ticker)
        return None

    # Parse
    rows = []
    skipped = 0
    for market in all_markets:
        row = parse_market_to_row(market, city_patterns)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1

    logger.info("Parsed %d markets, skipped %d", len(rows), skipped)

    if not rows:
        logger.warning("No markets parsed for %s!", series_ticker)
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "threshold"]).reset_index(drop=True)

    logger.info("%s: %d total contracts, %s to %s, %d unique dates",
                label, len(df), df["date"].min().date(), df["date"].max().date(),
                df["date"].nunique())

    # Year distribution
    for year in sorted(df["date"].dt.year.unique()):
        mask = df["date"].dt.year == year
        logger.info("  %d: %d contracts across %d days",
                     year, mask.sum(), df[mask]["date"].nunique())

    # Fill actual_tmax from GHCN station data
    ghcn_path = os.path.join(data_dir, cfg["data_subdir"], "raw", f"{cfg['target_station']}.csv")
    df = fill_actual_tmax_from_ghcn(df, ghcn_path, tmax_col=cfg["ghcn_col"])

    # Propagate actual_tmax from winning bucket to all buckets on same day
    for d in df["date"].unique():
        mask = df["date"] == d
        tmax_vals = df.loc[mask, "actual_tmax"].dropna()
        if len(tmax_vals) > 0:
            df.loc[mask, "actual_tmax"] = tmax_vals.iloc[0]

    # Drop rows with missing critical data
    before = len(df)
    df = df.dropna(subset=["market_prob", "actual_outcome"]).reset_index(drop=True)
    logger.info("Dropped %d rows with missing market_prob or actual_outcome", before - len(df))

    # Save
    out_path = os.path.join(data_dir, f"real_kalshi_{city_code}_all.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s (%d rows, %d days)", out_path, len(df), df["date"].nunique())

    # Also save in the format expected by contract_brier.py (same columns as NYC)
    # The existing real_kalshi_*.csv files are already in this format
    logger.info("actual_tmax coverage: %d / %d (%.1f%%)",
                df["actual_tmax"].notna().sum(), len(df),
                100 * df["actual_tmax"].notna().sum() / len(df))

    zero_vol = (df["volume"] == 0).sum()
    logger.info("Zero-volume contracts: %d / %d (%.1f%%)",
                zero_vol, len(df), 100 * zero_vol / len(df))

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Kalshi data for CHI/PHL")
    parser.add_argument("--city", type=str, default="both",
                        choices=["chi", "phl", "aus", "all"])
    args = parser.parse_args()

    cities = ["chi", "phl", "aus"] if args.city == "all" else [args.city]

    results = {}
    for city_code in cities:
        df = fetch_city(city_code)
        if df is not None:
            results[city_code] = df
        else:
            logger.warning("No data for %s", city_code)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for city_code, df in results.items():
        cfg = CITY_CONFIG[city_code]
        logger.info("  %s (%s): %d contracts, %d days, %s to %s",
                     cfg["label"], cfg["series_ticker"],
                     len(df), df["date"].nunique(),
                     df["date"].min().date(), df["date"].max().date())


if __name__ == "__main__":
    main()
