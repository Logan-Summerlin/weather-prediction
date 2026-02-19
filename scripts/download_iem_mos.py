#!/usr/bin/env python3
"""Backward-compatible wrapper for MOS download (KNYC default)."""

try:
    from scripts.download_iem_mos_data import *  # noqa: F401,F403
except ModuleNotFoundError:
    from download_iem_mos_data import *  # type: ignore # noqa: F401,F403


if __name__ == "__main__":
    run_for_station("KNYC")
