"""Guardrail tests for consolidated per-city script wrappers."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STAGES = {
    "data_collection",
    "preprocessing",
    "benchmark",
    "synthesis_calibration",
    "backtest",
    "promotion_evaluation",
}
CITIES = {"nyc", "chi", "phl", "atl", "aus"}
WRAPPER_PATTERN = re.compile(r"^run_(nyc|chi|phl|atl|aus)_(.+)\.py$")


def _wrapper_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(SCRIPTS_DIR.glob("run_*_*.py")):
        match = WRAPPER_PATTERN.match(path.name)
        if not match:
            continue
        stage = match.group(2)
        if stage in STAGES:
            paths.append(path)
    return paths


def test_expected_city_stage_wrappers_exist() -> None:
    """All city/stage compatibility wrappers should exist explicitly."""
    expected = {
        SCRIPTS_DIR / f"run_{city}_{stage}.py"
        for city in CITIES
        for stage in STAGES
    }
    discovered = set(_wrapper_paths())
    assert discovered == expected


def test_city_wrappers_are_thin_delegators() -> None:
    """Wrappers should stay tiny and only delegate to unified run_<stage>.py scripts."""
    for wrapper in _wrapper_paths():
        match = WRAPPER_PATTERN.match(wrapper.name)
        assert match is not None

        city, stage = match.group(1), match.group(2)
        expected_delegate = f"from scripts.run_{stage} import main"

        text = wrapper.read_text(encoding="utf-8")
        assert expected_delegate in text, f"{wrapper.name} must delegate to run_{stage}.py"
        assert f'"--city", "{city}"' in text, f"{wrapper.name} must hard-code --city {city}"

        stripped = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        significant_lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(significant_lines) <= 9, (
            f"{wrapper.name} grew beyond a thin shim ({len(significant_lines)} significant lines)."
        )
