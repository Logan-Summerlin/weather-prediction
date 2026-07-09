#!/usr/bin/env python3
"""Fetch KXHIGHNY pre-settlement snapshots + settlement results for 2026.

Builds the true out-of-sample evaluation set for the NYC market-debias
variant (U10): for each 2026 date through --end, this captures

  1. The pre-settlement candle snapshot using the SAME window as the
     historical fetch (`fetch_kalshi_presettlement.py`): last 1-hour candle
     between 22:00 UTC on D-1 and 05:30 UTC on D — i.e. ~midnight ET at the
     start of the event day, well before the 7am ET model cutoff.
  2. The settled market result (yes/no) from the markets listing, which is
     the actual bucket outcome.

Output: data/kalshi_nyc_2026_oos.csv with the presettlement schema plus
`result` / `actual_outcome` columns.

Usage:
    python scripts/fetch_kalshi_nyc_2026_oos.py --start 2026-01-01 --end 2026-07-07
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "kalshi_nyc_2026_oos.csv"

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXHIGHNY"
MIN_INTERVAL = 0.4
MAX_RETRIES = 4

MONTH_ABBR = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
              7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}

CSV_COLUMNS = [
    "date", "ticker", "bucket", "threshold_low", "threshold_high",
    "direction", "strike_type", "presettlement_prob", "bid_cents",
    "ask_cents", "volume", "open_interest", "snapshot_time_utc",
    "result", "actual_outcome",
]

_last_request = [0.0]
_session = requests.Session()
_session.headers.update({"Accept": "application/json",
                         "User-Agent": "NYC-Temp-Prediction/1.0"})


def api_get(url: str, params: dict | None = None) -> dict | None:
    """Rate-limited GET with retry/backoff. Returns parsed JSON or None on 404."""
    for attempt in range(1, MAX_RETRIES + 1):
        wait = MIN_INTERVAL - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.time()
        try:
            resp = _session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(2.0 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            if attempt < MAX_RETRIES:
                logger.warning("Request failed (%d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(2.0 ** attempt)
            else:
                raise
    raise ConnectionError(f"Retries exhausted for {url}")


def parse_bucket(market: dict) -> dict:
    """Parse bucket semantics from market title/subtitle text (ticker fallback)."""
    ticker = market.get("ticker", "")
    text = " ".join(str(market.get(f, "")) for f in ("title", "subtitle", "yes_sub_title"))
    result = {"threshold_low": "", "threshold_high": "", "direction": "unknown",
              "strike_type": "unknown", "bucket": ticker}

    m_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)", text)
    m_above = re.search(r"(?:>=|≥|[Aa]bove|[Gg]reater\s+than|[Hh]igher\s+than)\s*(\d+(?:\.\d+)?)", text)
    m_below = re.search(r"(?:<=|≤|[Bb]elow|[Ll]ess\s+than|[Uu]nder|[Ll]ower\s+than)\s*(\d+(?:\.\d+)?)", text)

    if m_range:
        lo, hi = sorted((float(m_range.group(1)), float(m_range.group(2))))
        result.update(threshold_low=lo, threshold_high=hi, direction="between",
                      strike_type="between", bucket=f"{lo:.0f}-{hi:.0f}F")
    elif m_below:
        val = float(m_below.group(1))
        result.update(threshold_high=val, direction="below", strike_type="less",
                      bucket=f"Below {val:.0f}F")
    elif m_above:
        val = float(m_above.group(1))
        result.update(threshold_low=val, direction="above", strike_type="greater",
                      bucket=f"Above {val:.0f}F")
    else:
        strike = ticker.split("-")[-1]
        if strike.startswith("B"):
            try:
                mid = float(strike[1:])
                lo, hi = int(mid - 0.5), int(mid + 1.5)
                result.update(threshold_low=lo, threshold_high=hi, direction="between",
                              strike_type="between", bucket=f"{lo}-{hi}F")
            except ValueError:
                pass
    return result


def snapshot(ticker: str, d: date) -> dict | None:
    """Last 1h candle between 22:00 UTC D-1 and 05:30 UTC D (matches historical fetch)."""
    day_before = d - timedelta(days=1)
    start_ts = int(datetime(day_before.year, day_before.month, day_before.day,
                            22, 0, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(d.year, d.month, d.day, 5, 30, tzinfo=timezone.utc).timestamp())
    data = api_get(f"{BASE_URL}/series/{SERIES_TICKER}/markets/{ticker}/candlesticks",
                   {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60})
    candles = (data or {}).get("candlesticks", [])
    if not candles:
        return None

    def close_cents(candle, field):
        """Close price in cents; handles legacy `close` (cents int) and
        current `close_dollars` (dollar string) API schemas."""
        v = candle.get(field)
        if isinstance(v, dict):
            if v.get("close") is not None:
                return float(v["close"])
            if v.get("close_dollars") is not None:
                return float(v["close_dollars"]) * 100.0
            return None
        return float(v) if isinstance(v, (int, float)) else None

    def scalar(candle, base_field):
        v = candle.get(base_field)
        if isinstance(v, (int, float)):
            return v
        fp = candle.get(f"{base_field}_fp")
        try:
            return float(fp)
        except (TypeError, ValueError):
            return 0

    # Walk back from the latest candle to the most recent one with a
    # usable quote (late-night candles can be empty).
    for last in sorted(candles, key=lambda c: c.get("end_period_ts", 0), reverse=True):
        bid = close_cents(last, "yes_bid")
        ask = close_cents(last, "yes_ask")
        price = close_cents(last, "price")
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif price is not None:
            mid = price
        elif bid is not None or ask is not None:
            mid = bid if bid is not None else ask
        else:
            continue
        ts = last.get("end_period_ts", 0)
        return {
            "presettlement_prob": mid / 100.0,
            "bid_cents": bid if bid is not None else "",
            "ask_cents": ask if ask is not None else "",
            "volume": scalar(last, "volume"),
            "open_interest": scalar(last, "open_interest"),
            "snapshot_time_utc": datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC") if ts else "",
        }
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-07-07")
    args = ap.parse_args()
    d0 = datetime.strptime(args.start, "%Y-%m-%d").date()
    d1 = datetime.strptime(args.end, "%Y-%m-%d").date()

    rows: list[dict] = []
    d = d0
    n_days = 0
    while d <= d1:
        event = f"{SERIES_TICKER}-{d.strftime('%y')}{MONTH_ABBR[d.month]}{d.day:02d}"
        data = api_get(f"{BASE_URL}/markets",
                       {"series_ticker": SERIES_TICKER, "event_ticker": event, "limit": 50})
        markets = (data or {}).get("markets", [])
        settled = [mk for mk in markets if mk.get("result") in ("yes", "no")]
        if not settled:
            logger.warning("%s: no settled markets (%d listed)", d, len(markets))
            d += timedelta(days=1)
            continue
        for mk in settled:
            ticker = mk.get("ticker", "")
            info = parse_bucket(mk)
            snap = snapshot(ticker, d) or {}
            rows.append({
                "date": d.isoformat(), "ticker": ticker, **info,
                "presettlement_prob": snap.get("presettlement_prob", ""),
                "bid_cents": snap.get("bid_cents", ""),
                "ask_cents": snap.get("ask_cents", ""),
                "volume": snap.get("volume", ""),
                "open_interest": snap.get("open_interest", ""),
                "snapshot_time_utc": snap.get("snapshot_time_utc", ""),
                "result": mk.get("result", ""),
                "actual_outcome": 1 if mk.get("result") == "yes" else 0,
            })
        n_days += 1
        if n_days % 10 == 0:
            logger.info("Progress: %s (%d days, %d rows)", d, n_days, len(rows))
            OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                w.writeheader()
                w.writerows(rows)
        d += timedelta(days=1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    logger.info("Done: %d rows, %d days -> %s", len(rows), n_days, OUTPUT_CSV)


if __name__ == "__main__":
    main()
