#!/usr/bin/env python3
"""Run the E0-E22 best-model benchmark family via the shared benchmark runner."""

from src.nyc_benchmark_registry import run_family


if __name__ == "__main__":
    run_family("e0_e8_best_model")
