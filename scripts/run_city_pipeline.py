#!/usr/bin/env python3
"""Canonical city pipeline runner.

Runs one stage (or all stages) of the multi-city forecasting pipeline through a
single CLI. This script is intentionally thin and delegates to existing stage
entrypoints to preserve behavior while reducing wrapper duplication.

Examples
--------
python scripts/run_city_pipeline.py --city chi --stage benchmark
python scripts/run_city_pipeline.py --city phl --stage all
python scripts/run_city_pipeline.py --city atl --stage all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StageSpec:
    """Specification for one pipeline stage command."""

    name: str
    script_path: str
    description: str


STAGE_ORDER: tuple[str, ...] = (
    "data_collection",
    "preprocessing",
    "benchmark",
    "synthesis_calibration",
    "backtest",
    "promotion_evaluation",
)

STAGE_REGISTRY: dict[str, StageSpec] = {
    "data_collection": StageSpec(
        name="data_collection",
        script_path="scripts/run_data_collection.py",
        description="Collect and parse station observations.",
    ),
    "preprocessing": StageSpec(
        name="preprocessing",
        script_path="scripts/run_preprocessing.py",
        description="Build time-safe train/val/test features and targets.",
    ),
    "benchmark": StageSpec(
        name="benchmark",
        script_path="scripts/run_benchmark.py",
        description="Train/evaluate baseline and core probabilistic models.",
    ),
    "synthesis_calibration": StageSpec(
        name="synthesis_calibration",
        script_path="scripts/run_synthesis_calibration.py",
        description="Run synthesis models and post-hoc calibration.",
    ),
    "backtest": StageSpec(
        name="backtest",
        script_path="scripts/run_backtest.py",
        description="Run EV-aware backtest with conservative cost assumptions.",
    ),
    "promotion_evaluation": StageSpec(
        name="promotion_evaluation",
        script_path="scripts/run_promotion_evaluation.py",
        description="Evaluate promotion gates and readiness checks.",
    ),
}

SUPPORTED_CITIES: tuple[str, ...] = ("chi", "phl", "atl", "aus")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return list(STAGE_ORDER)
    return [stage]


def _run_stage(city: str, stage: str, dry_run: bool) -> int:
    spec = STAGE_REGISTRY[stage]
    script_abs = PROJECT_ROOT / spec.script_path
    cmd = [sys.executable, str(script_abs), "--city", city]

    logger.info("Stage=%s | City=%s", stage, city)
    logger.info("Description: %s", spec.description)
    logger.info("Command: %s", " ".join(cmd))

    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a canonical city pipeline stage or full pipeline.",
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=SUPPORTED_CITIES,
        help="City code to run.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[*STAGE_ORDER, "all"],
        help="Stage to run, or 'all' for ordered end-to-end execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="When --stage all, continue remaining stages after a failure.",
    )
    args = parser.parse_args()

    stages = _resolve_stages(args.stage)
    logger.info("Planned stages: %s", stages)

    failures: list[tuple[str, int]] = []
    for stage in stages:
        exit_code = _run_stage(args.city, stage, dry_run=args.dry_run)
        if exit_code != 0:
            failures.append((stage, exit_code))
            logger.error("Stage %s failed with exit code %d", stage, exit_code)
            if not args.continue_on_error:
                break

    if failures:
        failed_summary = ", ".join(f"{name}:{code}" for name, code in failures)
        logger.error("Pipeline finished with failures: %s", failed_summary)
        return 1

    logger.info("Pipeline completed successfully for city=%s stage=%s", args.city, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
