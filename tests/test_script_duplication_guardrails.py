"""CI guardrails to prevent new script copy/paste duplication."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

ALLOWED_CITIES = {"nyc", "chi", "phl", "atl", "aus"}
ALLOWED_STAGES = {
    "data_collection",
    "preprocessing",
    "benchmark",
    "synthesis_calibration",
    "backtest",
    "promotion_evaluation",
    "end_to_end",
}
WRAPPER_RE = re.compile(r"^run_(nyc|chi|phl|atl|aus)_(.+)\.py$")


def _is_wrapper(path: Path) -> bool:
    match = WRAPPER_RE.match(path.name)
    return bool(match and match.group(2) in ALLOWED_STAGES)


def _normalized_script_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def test_no_unapproved_city_wrapper_names() -> None:
    for path in SCRIPTS_DIR.glob("run_*_*.py"):
        match = WRAPPER_RE.match(path.name)
        if not match:
            continue
        city, stage = match.group(1), match.group(2)
        assert city in ALLOWED_CITIES
        assert stage in ALLOWED_STAGES


def test_no_high_overlap_non_wrapper_scripts() -> None:
    candidate_scripts = [
        p for p in sorted(SCRIPTS_DIR.glob("run_*.py")) if not _is_wrapper(p)
    ]

    normalized = {p: _normalized_script_text(p) for p in candidate_scripts}

    for i, left in enumerate(candidate_scripts):
        for right in candidate_scripts[i + 1 :]:
            ratio = SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            assert ratio < 0.96, (
                f"High-overlap script copy detected: {left.name} vs {right.name} "
                f"(similarity={ratio:.3f}). Refactor shared logic instead of copying."
            )
