#!/usr/bin/env python3
"""Generate per-city backward-compatible wrapper scripts from one template."""

from __future__ import annotations

from pathlib import Path


CITIES = {
    "nyc": "New York City",
    "chi": "Chicago",
    "phl": "Philadelphia",
    "atl": "Atlanta",
    "aus": "Austin",
}
STAGES = {
    "data_collection": "Data Collection",
    "preprocessing": "Preprocessing",
    "benchmark": "Model Benchmark",
    "synthesis_calibration": "Synthesis Calibration",
    "backtest": "Backtest",
    "promotion_evaluation": "Promotion Evaluation",
}

TEMPLATE = '''#!/usr/bin/env python3
"""{city_name} {stage_title} — backward-compatible wrapper.

Delegates to the unified ``run_{stage}.py --city {city}``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = [sys.argv[0], "--city", "{city}"]
from scripts.run_{stage} import main
if __name__ == "__main__":
    main()
'''


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    for city, city_name in CITIES.items():
        for stage, stage_title in STAGES.items():
            target = scripts_dir / f"run_{city}_{stage}.py"
            target.write_text(
                TEMPLATE.format(city=city, city_name=city_name, stage=stage, stage_title=stage_title),
                encoding="utf-8",
            )
    print("Generated all city stage wrappers.")


if __name__ == "__main__":
    main()
