#!/usr/bin/env python3
"""Run the WGA-v2 benchmark family via the shared benchmark runner."""

from src.nyc_benchmark_registry import run_family


if __name__ == "__main__":
    run_family("wga_v2")
