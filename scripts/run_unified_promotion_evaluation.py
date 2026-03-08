#!/usr/bin/env python3
"""
Unified Promotion Evaluation — Phase F.

Runs the standardized promotion evaluation across one or all cities,
producing a single canonical report artifact per city that includes:
  - Promotion gate pass/fail results (standardized schema)
  - Baseline comparisons (persistence, climatology, ridge)
  - Market-implied benchmark (Kalshi pre-settlement Brier)
  - Calibration summary
  - Trading summary with seasonal breakdown

This replaces both run_promotion_evaluation.py (v1) and
run_promotion_evaluation_v2.py (v2) with a unified schema.

Outputs per city:
  - results/{city}/unified_promotion_report.json  (canonical report artifact)
  - results/{city}/unified_promotion_summary.png  (gate summary chart)
  - Console formatted report

Usage:
    python scripts/run_unified_promotion_evaluation.py --city chi
    python scripts/run_unified_promotion_evaluation.py --city all
    python scripts/run_unified_promotion_evaluation.py  # defaults to all cities
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

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, list_cities  # noqa: E402
from src.promotion_report import (  # noqa: E402
    CITY_THRESHOLDS,
    evaluate_city,
    save_unified_report,
    print_unified_report,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Visualization
# ===========================================================================

def plot_gate_summary(
    report: dict,
    save_path: str,
) -> None:
    """Create a horizontal bar chart summarizing gate pass/fail status.

    Parameters
    ----------
    report : dict
        The unified promotion report.
    save_path : str
        File path to save the PNG figure.
    """
    gate_list = report["gates"]
    names = [g["name"] for g in gate_list]
    statuses = [1 if g["passed"] else 0 for g in gate_list]
    colors = ["#2ca02c" if s else "#d62728" for s in statuses]
    labels = ["PASS" if s else "FAIL" for s in statuses]

    fig, ax = plt.subplots(figsize=(14, max(4, len(names) * 0.55)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, statuses, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.1, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])

    city = report.get("city", "Unknown")
    ticker = report.get("kalshi_ticker", "")
    version = report.get("schema_version", "3.0")
    ax.set_title(f"{city} ({ticker}) Unified Promotion Gate Summary v{version}")

    for bar, label, gate in zip(bars, labels, gate_list):
        x_pos = bar.get_width() + 0.05
        detail_text = gate.get("details", "")[:75]
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{label}: {detail_text}",
            va="center",
            fontsize=7.5,
        )

    ax.invert_yaxis()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved gate summary chart to %s", save_path)


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict[str, dict]:
    """Run unified promotion evaluation.

    Returns
    -------
    dict
        Mapping of city_code to its unified promotion report.
    """
    parser = argparse.ArgumentParser(
        description="Unified Promotion Evaluation (Phase F)"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help=(
            "City code to evaluate (e.g., chi, phl, atl, aus, nyc) "
            "or 'all' for all cities with defined thresholds. Default: all"
        ),
    )
    args = parser.parse_args()
    city_arg = args.city.strip().lower()

    # Determine which cities to evaluate
    if city_arg == "all":
        cities = sorted(CITY_THRESHOLDS.keys())
    else:
        if city_arg not in CITY_THRESHOLDS:
            available = ", ".join(sorted(CITY_THRESHOLDS.keys()))
            parser.error(
                f"Unknown city '{city_arg}'. Available: {available}"
            )
        cities = [city_arg]

    logger.info("=" * 80)
    logger.info("Unified Promotion Evaluation — Phase F")
    logger.info("Cities: %s", ", ".join(cities))
    logger.info("=" * 80)

    all_reports: dict[str, dict] = {}

    for city_code in cities:
        cfg = get_city_config(city_code)
        logger.info("-" * 60)
        logger.info("Evaluating %s (%s)", cfg.city_name, cfg.kalshi_ticker)
        logger.info("-" * 60)

        try:
            gates, report = evaluate_city(city_code)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", city_code, e)
            continue

        all_reports[city_code] = report

        # Save report JSON
        report_path = os.path.join(cfg.results_dir, "unified_promotion_report.json")
        save_unified_report(report, report_path)

        # Save summary chart
        chart_path = os.path.join(cfg.results_dir, "unified_promotion_summary.png")
        plot_gate_summary(report, chart_path)

        # Console output
        print_unified_report(report)

    # Cross-city summary
    if len(all_reports) > 1:
        print()
        print("=" * 80)
        print("  CROSS-CITY UNIFIED PROMOTION SUMMARY")
        print("=" * 80)
        for code, rpt in sorted(all_reports.items()):
            status = rpt["overall_status"]
            edge = rpt.get("market_benchmark", {}).get("brier_edge")
            edge_str = f"edge={edge:+.4f}" if edge is not None else "edge=N/A"
            pnl = rpt.get("trading_summary", {}).get("total_pnl")
            pnl_str = f"P&L=${pnl:.0f}" if pnl is not None else "P&L=N/A"
            print(
                f"  {rpt['city']:20s} ({rpt['kalshi_ticker']})  "
                f"{rpt['pass_rate']:>12s}  {status:4s}  {edge_str}  {pnl_str}"
            )
        print("=" * 80)
        print()

    logger.info("Unified promotion evaluation complete.")
    return all_reports


if __name__ == "__main__":
    main()
