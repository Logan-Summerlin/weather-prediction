"""
IEM ASOS Hourly Data Collection.

Downloads hourly ASOS/AWOS observations for configured stations from
the Iowa Environmental Mesonet (IEM) service. This source is available
operationally with low latency and supports the 6 AM ET cutoff.
"""

from __future__ import annotations

import csv
import os
import sys
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Iterable, Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

IEM_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DEFAULT_ASOS_FIELDS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "mslp",
    "alti",
    "vsby",
    "ceil",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# HTTP status codes worth retrying (IEM rate-limits aggressive scraping with
# 429; transient server errors are also retryable).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Module-level timestamp of the last network request, used to enforce a
# minimum inter-request spacing (politeness throttle) across calls.
_last_request_ts: float = 0.0


def _throttle(min_interval: float) -> None:
    """Sleep so consecutive network requests are at least *min_interval* apart."""
    global _last_request_ts
    if min_interval <= 0:
        return
    now = time.monotonic()
    wait = min_interval - (now - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()


def _retry_wait_seconds(exc: requests.RequestException, attempt: int,
                        backoff_base: float, max_wait: float) -> float:
    """Compute the backoff before the next retry, honoring ``Retry-After``."""
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After") if response.headers else None
        if retry_after:
            try:
                return min(float(retry_after), max_wait)
            except (TypeError, ValueError):
                pass
    # Exponential backoff with jitter: base^attempt (+/- 25%).
    base = backoff_base ** attempt
    jitter = base * 0.25 * (2 * random.random() - 1)
    return min(max(base + jitter, 0.0), max_wait)


def load_asos_station_map(mapping_csv: str) -> dict[str, str]:
    """Load ASOS station mapping from CSV."""
    mapping: dict[str, str] = {}
    with open(mapping_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("asos_available", "").strip().lower() != "yes":
                continue
            station_id = row["station_id"].strip()
            icao = row["icao"].strip()
            if station_id and icao:
                mapping[station_id] = icao
    return mapping


def build_asos_request_url(
    icao: str,
    start_date: str,
    end_date: str,
    data_fields: Optional[Iterable[str]] = None,
) -> str:
    """Build the IEM ASOS request URL."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    fields = list(data_fields or DEFAULT_ASOS_FIELDS)
    params = [
        ("station", icao),
        ("format", "csv"),
        ("tz", "Etc/UTC"),
        ("year1", start.year),
        ("month1", start.month),
        ("day1", start.day),
        ("year2", end.year),
        ("month2", end.month),
        ("day2", end.day),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "empty"),
    ]
    for field in fields:
        params.append(("data", field))

    query = "&".join(f"{key}={value}" for key, value in params)
    return f"{IEM_BASE_URL}?{query}"


def iter_date_chunks(start_date: str, end_date: str,
                     chunk_years: int = 1) -> list[tuple[str, str]]:
    """Split a date range into year-sized chunks for IEM requests."""
    if chunk_years < 1:
        raise ValueError("chunk_years must be >= 1")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    chunks: list[tuple[str, str]] = []
    current = start
    while current <= end:
        next_year = current.replace(year=current.year + chunk_years, month=1, day=1)
        chunk_end = min(end, next_year - timedelta(days=1))
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


def download_asos_station(
    icao: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    data_fields: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 4,
    backoff_base: float = 3.0,
    max_wait: float = 90.0,
    min_interval: float = 0.0,
) -> str:
    """Download hourly ASOS data for a single station.

    Retries on rate-limit (HTTP 429) and transient server errors with
    exponential backoff, honoring any ``Retry-After`` header.  ``min_interval``
    enforces a minimum spacing between successive network requests so bulk
    collection stays polite to the IEM service (defaults to 0 = no throttle,
    preserving fast unit tests).  Non-retryable errors propagate immediately.
    """
    os.makedirs(output_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(output_dir, f"{icao}.csv")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("Using cached ASOS file for %s", icao)
        return output_path

    url = build_asos_request_url(icao, start_date, end_date, data_fields)

    for attempt in range(max_retries + 1):
        _throttle(min_interval)
        logger.info("Requesting ASOS data (attempt %d): %s", attempt + 1, url)
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in _RETRYABLE_STATUS or attempt >= max_retries:
                raise
            wait = _retry_wait_seconds(exc, attempt + 1, backoff_base, max_wait)
            logger.warning(
                "ASOS request for %s got HTTP %s; retrying in %.1fs (attempt %d/%d)",
                icao, status, wait, attempt + 1, max_retries,
            )
            time.sleep(wait)
            continue

        with open(output_path, "wb") as handle:
            handle.write(response.content)

        logger.info("Saved ASOS data to %s (%.1f KB)",
                    output_path, os.path.getsize(output_path) / 1024)
        return output_path

    # Unreachable: the loop either returns or raises.
    raise RuntimeError(f"Exhausted retries downloading ASOS for {icao}")


def download_asos_station_range(
    icao: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    data_fields: Optional[Iterable[str]] = None,
    chunk_years: int = 1,
    min_interval: float = 0.0,
    max_retries: int = 4,
) -> list[str]:
    """Download ASOS data for a station in smaller date chunks."""
    paths: list[str] = []
    for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_years):
        chunk_dir = os.path.join(output_dir, icao)
        os.makedirs(chunk_dir, exist_ok=True)
        filename = f"{icao}_{chunk_start}_{chunk_end}.csv"
        output_path = os.path.join(chunk_dir, filename)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            paths.append(output_path)
            continue
        path = download_asos_station(
            icao=icao,
            output_dir=chunk_dir,
            start_date=chunk_start,
            end_date=chunk_end,
            data_fields=data_fields,
            output_path=output_path,
            min_interval=min_interval,
            max_retries=max_retries,
        )
        paths.append(path)
    return paths


def collect_asos_data(
    mapping_csv: str,
    output_dir: str = config.ASOS_RAW_DIR,
    start_date: str = config.ASOS_START_DATE,
    end_date: str = config.ASOS_END_DATE,
    data_fields: Optional[Iterable[str]] = None,
    chunk_years: int = 1,
    min_interval: float = 0.0,
    max_retries: int = 4,
) -> dict[str, str]:
    """Collect ASOS data for all mapped stations.

    ``min_interval`` spaces out network requests (seconds) to stay within the
    IEM service's rate limits during bulk collection; ``max_retries`` controls
    backoff retries on HTTP 429 / transient errors.
    """
    station_map = load_asos_station_map(mapping_csv)
    results: dict[str, str] = {}

    for station_id, icao in station_map.items():
        logger.info("Downloading ASOS for %s (%s)", station_id, icao)
        try:
            paths = download_asos_station_range(
                icao=icao,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                data_fields=data_fields,
                chunk_years=chunk_years,
                min_interval=min_interval,
                max_retries=max_retries,
            )
            if paths:
                results[station_id] = paths[-1]
        except requests.RequestException as exc:
            logger.error("Failed to download ASOS for %s (%s): %s",
                         station_id, icao, exc)

    logger.info("ASOS collection complete: %d stations", len(results))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download IEM ASOS hourly data.")
    parser.add_argument(
        "--mapping-csv",
        default=os.path.join(config.DATA_DIR, "asos_station_mapping.csv"),
        help="CSV with station_id -> ICAO mapping.",
    )
    parser.add_argument("--start-date", default=config.ASOS_START_DATE)
    parser.add_argument("--end-date", default=config.ASOS_END_DATE)
    parser.add_argument(
        "--output-dir",
        default=config.ASOS_RAW_DIR,
        help="Directory to store raw ASOS CSVs.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=DEFAULT_ASOS_FIELDS,
        help="IEM ASOS fields to request (default includes temp, dewpoint, wind).",
    )
    parser.add_argument(
        "--chunk-years",
        type=int,
        default=1,
        help="Download in year-sized chunks to avoid very large requests.",
    )
    args = parser.parse_args()

    collect_asos_data(
        mapping_csv=args.mapping_csv,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        data_fields=args.fields,
        chunk_years=args.chunk_years,
    )


if __name__ == "__main__":
    main()
