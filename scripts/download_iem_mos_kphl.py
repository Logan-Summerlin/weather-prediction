#!/usr/bin/env python3
"""Backward-compatible wrapper for MOS download (KPHL)."""

try:
    from scripts.download_iem_mos_data import run_for_station
except ModuleNotFoundError:
    from download_iem_mos_data import run_for_station  # type: ignore


if __name__ == "__main__":
    run_for_station("KPHL")
