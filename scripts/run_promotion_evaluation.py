#!/usr/bin/env python3
"""
Promotion Readiness Evaluation — thin delegator.

All gate logic lives in src/promotion_report.py (the single canonical
promotion framework). This script is the pipeline entrypoint invoked by
scripts/run_city_pipeline.py and the per-city compatibility wrappers.

Outputs per city:
  - results/<city>/unified_promotion_report.json  (canonical report artifact)
  - results/<city>/promotion_report.json          (legacy path, same content)
  - results/<city>/promotion_gate_summary.png     (gate summary chart)
  - Console formatted report

Usage:
    python scripts/run_promotion_evaluation.py --city chi
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs  # noqa: E402
from src.promotion_report import (  # noqa: E402
    CITY_THRESHOLDS,
    evaluate_city,
    print_unified_report,
    save_unified_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def plot_gate_summary(
    gates: list,
    save_path: str,
    city_name: str,
    kalshi_ticker: str,
) -> None:
    """Create a horizontal bar chart summarizing gate pass/fail status."""
    names = [g.name for g in gates]
    statuses = [1 if g.passed else 0 for g in gates]
    colors = ["#2ca02c" if s else "#d62728" for s in statuses]
    labels = ["PASS" if s else "FAIL" for s in statuses]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, statuses, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.1, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title(f"{city_name} ({kalshi_ticker}) Promotion Gate Summary")

    for bar, label, gate in zip(bars, labels, gates):
        x_pos = bar.get_width() + 0.05
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{label}: {gate.details[:60]}",
                va="center", fontsize=8)

    ax.invert_yaxis()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved gate summary chart to %s", save_path)


def main() -> dict:
    """Run the promotion readiness evaluation for a city.

    Parses ``--city`` from sys.argv, delegates to the unified framework in
    src/promotion_report.py, and writes the report artifacts.

    Returns
    -------
    dict
        The unified promotion report dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Unified Model Promotion Readiness Evaluation"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=sorted(CITY_THRESHOLDS.keys()),
        help="City code to evaluate",
    )
    args = parser.parse_args()
    city_code = args.city.strip().lower()

    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    logger.info("=" * 70)
    logger.info(
        "%s (%s) Promotion Readiness Evaluation", cfg.city_name, cfg.kalshi_ticker
    )
    logger.info("=" * 70)

    gates, report = evaluate_city(city_code)

    for g in gates:
        logger.info(
            "  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details
        )

    # Canonical artifact
    unified_path = os.path.join(cfg.results_dir, "unified_promotion_report.json")
    save_unified_report(report, unified_path)

    # Legacy path kept for downstream consumers (same unified content)
    legacy_path = os.path.join(cfg.results_dir, "promotion_report.json")
    save_unified_report(report, legacy_path)

    chart_path = os.path.join(cfg.results_dir, "promotion_gate_summary.png")
    plot_gate_summary(gates, chart_path, cfg.city_name, cfg.kalshi_ticker)

    print_unified_report(report)

    logger.info("Promotion evaluation complete. Overall: %s", report["overall_status"])
    return report


if __name__ == "__main__":
    main()
