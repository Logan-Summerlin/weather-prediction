#!/usr/bin/env python3
"""Download GHCN .dly files for all stations in config_expanded.py.

Downloads in parallel batches from NOAA bulk server.
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config_expanded as config

BASE_URL = config.NOAA_BULK_BASE_URL
RAW_DIR = config.RAW_DATA_DIR
MAX_RETRIES = 3
TIMEOUT = 180
BATCH_SIZE = 10


def download_one(station_id):
    """Download a single .dly file. Returns (station_id, success, message)."""
    outpath = os.path.join(RAW_DIR, f"{station_id}.dly")
    if os.path.exists(outpath) and os.path.getsize(outpath) > 1000:
        return (station_id, True, f"Already cached ({os.path.getsize(outpath)} bytes)")

    url = f"{BASE_URL}{station_id}.dly"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size = os.path.getsize(outpath)
            return (station_id, True, f"Downloaded ({size:,} bytes)")
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
            else:
                return (station_id, False, f"FAILED after {MAX_RETRIES} attempts: {e}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    all_ids = list(config.ALL_STATIONS.keys())
    print(f"Downloading {len(all_ids)} station .dly files to {RAW_DIR}")
    print(f"Batch size: {BATCH_SIZE}, max retries: {MAX_RETRIES}")

    successes = 0
    failures = []

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {executor.submit(download_one, sid): sid for sid in all_ids}
        for future in as_completed(futures):
            sid, ok, msg = future.result()
            name = config.ALL_STATIONS.get(sid, "Unknown")
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {sid} ({name}): {msg}")
            if ok:
                successes += 1
            else:
                failures.append(sid)

    print(f"\nDone: {successes}/{len(all_ids)} succeeded")
    if failures:
        print(f"Failed stations: {failures}")
    return len(failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
