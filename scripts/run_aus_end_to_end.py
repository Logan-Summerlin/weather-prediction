#!/usr/bin/env python3
"""Run Austin end-to-end pipeline with contract-Brier guardrails.

Pipeline steps:
1) preprocessing
2) benchmark
3) NWS/Kalshi template benchmark
4) synthesis calibration
5) backtest
6) promotion evaluation

Guardrails:
- Goal contract Brier <= 0.11
- Reject suspiciously low contract Brier < 0.04
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "austin"

TARGET_BRIER = 0.11
MIN_SANITY_BRIER = 0.04


def run_step(cmd: list[str]) -> None:
    print(f"\n=== Running: {' '.join(cmd)} ===")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_brier() -> tuple[float, str]:
    synth_path = RESULTS_DIR / "synthesis" / "calibration_sweep_summary.json"
    if synth_path.exists():
        summary = json.loads(synth_path.read_text())
        brier = summary.get("best_brier")
        if brier is not None:
            return float(brier), f"{synth_path}"

    unified_path = RESULTS_DIR / "synthesis" / "unified_benchmark_summary.json"
    if unified_path.exists():
        summary = json.loads(unified_path.read_text())
        brier = summary.get("best_overall_brier", summary.get("best_brier"))
        if brier is not None:
            return float(brier), f"{unified_path}"

    benchmark_path = RESULTS_DIR / "benchmark_summary.json"
    if benchmark_path.exists():
        summary = json.loads(benchmark_path.read_text())
        brier = summary.get("best_brier")
        if brier is not None:
            return float(brier), f"{benchmark_path}"

    raise FileNotFoundError("No benchmark summary found (unified or base).")


def main() -> int:
    steps = [
        [sys.executable, "scripts/run_aus_preprocessing.py"],
        [sys.executable, "scripts/run_aus_benchmark.py"],
        [sys.executable, "scripts/run_city_nws_kalshi_template_benchmark.py", "--city", "aus"],
        [sys.executable, "scripts/run_aus_synthesis_calibration.py"],
        [sys.executable, "scripts/run_aus_backtest.py"],
        [sys.executable, "scripts/run_aus_promotion_evaluation.py"],
    ]

    for cmd in steps:
        run_step(cmd)

    brier, source = load_brier()
    print(f"\nAustin best contract Brier: {brier:.4f} (source: {source})")

    if brier < MIN_SANITY_BRIER:
        print(
            f"ERROR: Brier {brier:.4f} is below sanity floor {MIN_SANITY_BRIER:.2f}. "
            "This is likely a scoring/configuration bug."
        )
        return 2

    if brier > TARGET_BRIER:
        print(
            f"WARNING: Brier {brier:.4f} is above target {TARGET_BRIER:.2f}. "
            "Pipeline completed, but model target not met yet."
        )
        return 1

    print("SUCCESS: Austin pipeline completed and Brier target achieved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
