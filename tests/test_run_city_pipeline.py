"""Tests for scripts/run_city_pipeline.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_city_pipeline import STAGE_ORDER, _resolve_stages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_city_pipeline.py"


def test_resolve_all_stages_order():
    """`all` should expand to canonical stage order."""
    assert _resolve_stages("all") == list(STAGE_ORDER)


def test_dry_run_executes_successfully():
    """Dry-run should return success without running heavy pipeline steps."""
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--city", "chi", "--stage", "all", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Planned stages" in result.stderr


def test_single_stage_dry_run_succeeds():
    """Single-stage dry-run should also pass."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--city",
            "phl",
            "--stage",
            "benchmark",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0
