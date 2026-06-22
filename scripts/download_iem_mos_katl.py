#!/usr/bin/env python3
"""Backward-compatible wrapper for MOS download (KATL)."""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from download_iem_mos_data import run_for_station
else:
    from scripts.download_iem_mos_data import run_for_station


if __name__ == "__main__":
    run_for_station("KATL")
