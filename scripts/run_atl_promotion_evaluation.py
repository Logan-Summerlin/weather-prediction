#!/usr/bin/env python3
"""
Atlanta Model Promotion Readiness Evaluation.

Evaluates whether the Atlanta (KXHIGHATL) model pipeline meets
the promotion gates required for production deployment:

1. Forecast Quality: OOS Brier score beats NWS baseline
2. Calibration: Reliability/ECE within tolerance bands
3. Trading: Positive paper-trading P&L after fees/slippage
4. Operations: Pipeline completes successfully with audit artifacts

Outputs a structured promotion report with PASS/FAIL for each gate.

Usage:
    python scripts/run_atl_promotion_evaluation.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Use non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Promotion thresholds (Atlanta-specific, similar to PHL for warmer SE city)
# ---------------------------------------------------------------------------
BRIER_THRESHOLD = 0.14          # Model Brier must be below this (like PHL)
NWS_BRIER_BASELINE = 0.12      # Atlanta NWS FFC baseline Brier
ECE_THRESHOLD = 0.05            # Expected Calibration Error must be below 5%
MIN_POSITIVE_PNL = 0.0          # Paper trading must show positive P&L
MAX_DRAWDOWN_THRESHOLD = -0.30  # Max drawdown must not exceed 30%
MIN_OOS_DAYS = 200              # Minimum OOS evaluation days
SEASONAL_BRIER_THRESHOLD = 0.20 # Atlanta has less extreme winters than CHI


# ===========================================================================
# PromotionGate
# ===========================================================================

class PromotionGate:
    """Represents a single promotion gate check."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.value = None
        self.threshold = None
        self.details = ""

    def evaluate(self, value, threshold, comparison="less"):
        """Evaluate whether the gate passes.

        Parameters
        ----------
        value : float
            The observed metric value.
        threshold : float
            The threshold to compare against.
        comparison : str
            Comparison mode: "less" (value < threshold),
            "greater" (value > threshold), or
            "abs_less" (abs(value) < threshold).

        Returns
        -------
        bool
            True if the gate passes.
        """
        self.value = value
        self.threshold = threshold
        if comparison == "less":
            self.passed = value < threshold
        elif comparison == "greater":
            self.passed = value > threshold
        elif comparison == "abs_less":
            self.passed = abs(value) < threshold
        return self.passed

    def to_dict(self) -> dict:
        """Serialize gate result to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
        }


# ===========================================================================
# Gate Category 1: Forecast Quality
# ===========================================================================

def check_forecast_quality(results_dir: str) -> list[PromotionGate]:
    """Check OOS Brier score gates.

    Verifies:
      - Overall OOS Brier is below the absolute threshold.
      - Model Brier beats the NWS baseline.
      - No season has a Brier score above the seasonal threshold.
      - Sufficient OOS days are available for evaluation.

    Parameters
    ----------
    results_dir : str
        Path to Atlanta results directory.

    Returns
    -------
    list[PromotionGate]
        List of evaluated forecast-quality gates.
    """
    gates: list[PromotionGate] = []

    # Gate 1: Overall OOS Brier below absolute threshold
    gate = PromotionGate("overall_brier", "OOS Brier score below threshold")
    benchmark_path = os.path.join(results_dir, "benchmark_summary.json")
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            results = json.load(f)
        best_brier = results.get("best_brier", 1.0)
        gate.evaluate(best_brier, BRIER_THRESHOLD, "less")
        gate.details = f"Best model Brier: {best_brier:.4f} (threshold: {BRIER_THRESHOLD})"
    else:
        gate.details = f"Benchmark results not found at {benchmark_path}"
        gate.passed = False
    gates.append(gate)

    # Gate 2: Beats NWS baseline
    gate2 = PromotionGate("beats_nws", "Model Brier beats NWS baseline")
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            results = json.load(f)
        best_brier = results.get("best_brier", 1.0)
        gate2.evaluate(best_brier, NWS_BRIER_BASELINE, "less")
        gate2.details = f"Model: {best_brier:.4f} vs NWS: {NWS_BRIER_BASELINE:.4f}"
    else:
        gate2.details = "Benchmark results not found"
        gate2.passed = False
    gates.append(gate2)

    # Gate 3: Seasonal stress test — no season exceeds threshold
    gate3 = PromotionGate("seasonal_brier", "No season exceeds Brier threshold")
    seasonal_path = os.path.join(results_dir, "seasonal_brier.json")
    if os.path.exists(seasonal_path):
        with open(seasonal_path) as f:
            seasonal = json.load(f)
        worst_season = max(seasonal.values())
        worst_name = max(seasonal, key=seasonal.get)
        gate3.evaluate(worst_season, SEASONAL_BRIER_THRESHOLD, "less")
        gate3.details = (
            f"Worst season: {worst_name}={worst_season:.4f} "
            f"(threshold: {SEASONAL_BRIER_THRESHOLD})"
        )
    else:
        gate3.details = f"Seasonal results not found at {seasonal_path}"
        gate3.passed = False
    gates.append(gate3)

    # Gate 4: Minimum OOS evaluation days
    gate4 = PromotionGate("min_oos_days", "Sufficient OOS evaluation days")
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            results = json.load(f)
        n_oos = results.get("n_oos_days", 0)
        gate4.evaluate(n_oos, MIN_OOS_DAYS, "greater")
        gate4.details = f"OOS days: {n_oos} (minimum: {MIN_OOS_DAYS})"
    else:
        gate4.details = "Benchmark results not found"
        gate4.passed = False
    gates.append(gate4)

    return gates


# ===========================================================================
# Gate Category 2: Calibration
# ===========================================================================

def check_calibration(results_dir: str) -> list[PromotionGate]:
    """Check calibration quality gates.

    Verifies:
      - Expected Calibration Error is within tolerance.
      - Reliability diagram has been generated.

    Parameters
    ----------
    results_dir : str
        Path to Atlanta results directory.

    Returns
    -------
    list[PromotionGate]
        List of evaluated calibration gates.
    """
    gates: list[PromotionGate] = []

    # Gate 1: ECE within tolerance
    gate = PromotionGate("ece", "Expected Calibration Error within tolerance")
    cal_path = os.path.join(results_dir, "synthesis", "calibration_summary.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        ece = cal.get("best_ece", 1.0)
        gate.evaluate(ece, ECE_THRESHOLD, "less")
        gate.details = f"ECE: {ece:.4f} (threshold: {ECE_THRESHOLD})"
    else:
        gate.details = f"Calibration results not found at {cal_path}"
        gate.passed = False
    gates.append(gate)

    # Gate 2: Reliability diagram artifact exists
    gate2 = PromotionGate("reliability_diagram", "Reliability diagram artifact exists")
    reliability_path = os.path.join(results_dir, "synthesis", "reliability_diagram.png")
    if os.path.exists(reliability_path):
        gate2.passed = True
        gate2.details = f"Found at {reliability_path}"
    else:
        gate2.passed = False
        gate2.details = f"Reliability diagram not found at {reliability_path}"
    gates.append(gate2)

    return gates


# ===========================================================================
# Gate Category 3: Trading Simulation
# ===========================================================================

def check_trading(results_dir: str) -> list[PromotionGate]:
    """Check trading simulation gates.

    Verifies:
      - Paper-trading P&L is positive after fees and slippage.
      - Maximum drawdown is within acceptable limits.

    Parameters
    ----------
    results_dir : str
        Path to Atlanta results directory.

    Returns
    -------
    list[PromotionGate]
        List of evaluated trading gates.
    """
    gates: list[PromotionGate] = []

    bt_path = os.path.join(results_dir, "backtest", "backtest_summary.json")

    # Gate 1: Positive P&L
    gate = PromotionGate("positive_pnl", "Paper trading shows positive P&L")
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt = json.load(f)
        total_pnl = bt.get("total_pnl", -999)
        gate.evaluate(total_pnl, MIN_POSITIVE_PNL, "greater")
        gate.details = f"Total P&L: ${total_pnl:.2f} (minimum: ${MIN_POSITIVE_PNL:.2f})"
    else:
        gate.details = f"Backtest results not found at {bt_path}"
        gate.passed = False
    gates.append(gate)

    # Gate 2: Max drawdown within limits
    gate2 = PromotionGate("max_drawdown", "Max drawdown within acceptable limits")
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt = json.load(f)
        max_dd = bt.get("max_drawdown", -1.0)
        gate2.evaluate(max_dd, MAX_DRAWDOWN_THRESHOLD, "greater")
        gate2.details = f"Max drawdown: {max_dd:.1%} (limit: {MAX_DRAWDOWN_THRESHOLD:.1%})"
    else:
        gate2.details = "Backtest results not found"
        gate2.passed = False
    gates.append(gate2)

    return gates


# ===========================================================================
# Gate Category 4: Operational Readiness
# ===========================================================================

def check_operations(results_dir: str, data_dir: str, models_dir: str) -> list[PromotionGate]:
    """Check operational readiness gates.

    Verifies:
      - All required processed data files exist.
      - Model checkpoints have been saved.
      - Scaler artifact exists for inference.

    Parameters
    ----------
    results_dir : str
        Path to Atlanta results directory.
    data_dir : str
        Path to Atlanta data directory.
    models_dir : str
        Path to Atlanta models directory.

    Returns
    -------
    list[PromotionGate]
        List of evaluated operational gates.
    """
    gates: list[PromotionGate] = []

    # Gate 1: Processed data files exist
    gate = PromotionGate("data_pipeline", "Processed data files exist")
    processed_dir = os.path.join(data_dir, "processed")
    required_files = [
        "features_train.csv",
        "features_val.csv",
        "features_test.csv",
        "target_train.csv",
        "target_val.csv",
        "target_test.csv",
        "scaler.pkl",
    ]
    found = sum(
        1 for f in required_files
        if os.path.exists(os.path.join(processed_dir, f))
    )
    gate.passed = found == len(required_files)
    gate.value = found
    gate.threshold = len(required_files)
    gate.details = f"{found}/{len(required_files)} required files present in {processed_dir}"
    gates.append(gate)

    # Gate 2: Model checkpoints exist
    gate2 = PromotionGate("model_checkpoints", "Model checkpoints saved")
    if os.path.isdir(models_dir):
        model_files = [
            f for f in os.listdir(models_dir)
            if f.endswith(".pt") or f.endswith(".pkl")
        ]
        gate2.passed = len(model_files) > 0
        gate2.value = len(model_files)
        gate2.details = f"{len(model_files)} model file(s) found in {models_dir}"
    else:
        gate2.passed = False
        gate2.details = f"Models directory not found: {models_dir}"
    gates.append(gate2)

    # Gate 3: Scaler artifact exists
    gate3 = PromotionGate("scaler_artifact", "Feature scaler artifact exists")
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        gate3.passed = True
        gate3.details = f"Scaler found at {scaler_path}"
    else:
        gate3.passed = False
        gate3.details = f"Scaler not found at {scaler_path}"
    gates.append(gate3)

    return gates


# ===========================================================================
# Report Generation
# ===========================================================================

def generate_report(all_gates: list[PromotionGate], output_path: str) -> dict:
    """Generate and save the promotion report as JSON.

    Parameters
    ----------
    all_gates : list[PromotionGate]
        All evaluated gates across all categories.
    output_path : str
        File path to write the JSON report.

    Returns
    -------
    dict
        The full report dictionary.
    """
    total = len(all_gates)
    passed = sum(1 for g in all_gates if g.passed)

    report = {
        "timestamp": datetime.now().isoformat(),
        "city": "atlanta",
        "kalshi_ticker": "KXHIGHATL",
        "target_station": "USW00013874",
        "target_station_name": "Hartsfield-Jackson Atlanta International Airport",
        "overall_status": "PASS" if passed == total else "FAIL",
        "gates_passed": passed,
        "gates_total": total,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)",
        "thresholds": {
            "brier_threshold": BRIER_THRESHOLD,
            "nws_brier_baseline": NWS_BRIER_BASELINE,
            "ece_threshold": ECE_THRESHOLD,
            "min_positive_pnl": MIN_POSITIVE_PNL,
            "max_drawdown_threshold": MAX_DRAWDOWN_THRESHOLD,
            "min_oos_days": MIN_OOS_DAYS,
            "seasonal_brier_threshold": SEASONAL_BRIER_THRESHOLD,
        },
        "gates": [g.to_dict() for g in all_gates],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def plot_gate_summary(all_gates: list[PromotionGate], save_path: str) -> None:
    """Create a horizontal bar chart summarizing gate pass/fail status.

    Parameters
    ----------
    all_gates : list[PromotionGate]
        All evaluated gates.
    save_path : str
        File path to save the figure.
    """
    names = [g.name for g in all_gates]
    statuses = [1 if g.passed else 0 for g in all_gates]
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
    ax.set_title("Atlanta (KXHIGHATL) Promotion Gate Summary")

    # Annotate each bar with status label
    for bar, label, gate in zip(bars, labels, all_gates):
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


def print_report(report: dict) -> None:
    """Print a formatted promotion report to the console.

    Parameters
    ----------
    report : dict
        The promotion report dictionary.
    """
    print("\n" + "=" * 70)
    print("ATLANTA (KXHIGHATL) PROMOTION READINESS EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Target Station: {report['target_station_name']} ({report['target_station']})")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Gates: {report['pass_rate']}")
    print("-" * 70)

    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        icon = "[+]" if gate["passed"] else "[-]"
        print(f"  {icon} {gate['name']}: {status}")
        print(f"      {gate['details']}")

    print("-" * 70)
    if report["overall_status"] == "PASS":
        print("RECOMMENDATION: Atlanta model is READY for promotion.")
    else:
        print("RECOMMENDATION: Atlanta model is NOT YET ready for promotion.")
        failed = [g for g in report["gates"] if not g["passed"]]
        print(f"  {len(failed)} gate(s) need attention:")
        for g in failed:
            print(f"    - {g['name']}: {g['details']}")
    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict:
    """Run the full Atlanta promotion readiness evaluation.

    Returns
    -------
    dict
        The promotion report dictionary.
    """
    logger.info("=" * 70)
    logger.info("Atlanta (KXHIGHATL) Promotion Readiness Evaluation")
    logger.info("=" * 70)

    # --- Load city config ---
    atl = get_city_config("atl")
    ensure_city_dirs(atl)

    results_dir = atl.results_dir
    data_dir = atl.data_dir
    models_dir = atl.models_dir

    logger.info("Results dir: %s", results_dir)
    logger.info("Data dir: %s", data_dir)
    logger.info("Models dir: %s", models_dir)

    all_gates: list[PromotionGate] = []

    # --- Category 1: Forecast Quality ---
    logger.info("-" * 50)
    logger.info("Category 1: Forecast Quality")
    logger.info("-" * 50)
    forecast_gates = check_forecast_quality(results_dir)
    all_gates.extend(forecast_gates)
    for g in forecast_gates:
        logger.info("  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details)

    # --- Category 2: Calibration ---
    logger.info("-" * 50)
    logger.info("Category 2: Calibration")
    logger.info("-" * 50)
    calibration_gates = check_calibration(results_dir)
    all_gates.extend(calibration_gates)
    for g in calibration_gates:
        logger.info("  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details)

    # --- Category 3: Trading Simulation ---
    logger.info("-" * 50)
    logger.info("Category 3: Trading Simulation")
    logger.info("-" * 50)
    trading_gates = check_trading(results_dir)
    all_gates.extend(trading_gates)
    for g in trading_gates:
        logger.info("  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details)

    # --- Category 4: Operational Readiness ---
    logger.info("-" * 50)
    logger.info("Category 4: Operational Readiness")
    logger.info("-" * 50)
    ops_gates = check_operations(results_dir, data_dir, models_dir)
    all_gates.extend(ops_gates)
    for g in ops_gates:
        logger.info("  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details)

    # --- Generate report ---
    report_path = os.path.join(results_dir, "promotion_report.json")
    report = generate_report(all_gates, report_path)

    # --- Generate summary chart ---
    chart_path = os.path.join(results_dir, "promotion_gate_summary.png")
    plot_gate_summary(all_gates, chart_path)

    # --- Print formatted report ---
    print_report(report)

    logger.info("Report saved to %s", report_path)
    logger.info("=" * 70)
    logger.info("Promotion evaluation complete. Overall: %s", report["overall_status"])
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    main()
