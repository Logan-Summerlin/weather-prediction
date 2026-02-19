#!/usr/bin/env python3
"""
Unified Model Promotion Readiness Evaluation.

Evaluates whether a city's model pipeline meets the promotion gates
required for production deployment:

1. Forecast Quality: OOS Brier score beats NWS baseline
2. Calibration: Reliability/ECE within tolerance bands
3. Trading: Positive paper-trading P&L after fees/slippage
4. Operations: Pipeline completes successfully with audit artifacts

Outputs a structured promotion report with PASS/FAIL for each gate.

Usage:
    python scripts/run_promotion_evaluation.py --city chi
    python scripts/run_promotion_evaluation.py --city phl
    python scripts/run_promotion_evaluation.py --city atl
    python scripts/run_promotion_evaluation.py --city aus
"""

from __future__ import annotations

import argparse
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

from src.city_config import get_city_config, ensure_city_dirs  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# City-specific config module mapping (for optional dynamic import)
# ---------------------------------------------------------------------------
CITY_CONFIG_MODULES = {
    "nyc": "config_expanded",
    "chi": "config_chicago",
    "phl": "config_philadelphia",
    "atl": "config_atlanta",
    "aus": "config_austin",
}

# ---------------------------------------------------------------------------
# City-specific promotion thresholds
# ---------------------------------------------------------------------------
CITY_THRESHOLDS = {
    "nyc": {
        "brier_threshold": 0.14,            # NYC moderate variance
        "nws_brier_baseline": 0.12,         # NYC NWS OKX baseline
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
    },
    "chi": {
        "brier_threshold": 0.16,            # Wider Chicago variance
        "nws_brier_baseline": 0.14,         # Chicago NWS LOT baseline
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.22,   # Chicago extremes
    },
    "phl": {
        "brier_threshold": 0.15,
        "nws_brier_baseline": 0.13,         # PHL NWS baseline
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
    },
    "atl": {
        "brier_threshold": 0.14,            # Like PHL for warmer SE city
        "nws_brier_baseline": 0.12,         # Atlanta NWS FFC baseline
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,   # Less extreme winters than CHI
    },
    "aus": {
        "brier_threshold": 0.16,            # Wider Austin variance
        "nws_brier_baseline": 0.14,         # Austin NWS baseline
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.22,   # Austin extremes
    },
}


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
# Helper: Load best Brier (unified > base fallback)
# ===========================================================================

def _load_best_brier(results_dir: str) -> tuple[float, str]:
    """Load the best model Brier score, preferring unified benchmark results.

    Checks unified_benchmark_summary.json first (U-series), then falls back
    to benchmark_summary.json (base benchmark).

    Parameters
    ----------
    results_dir : str
        Path to city results directory.

    Returns
    -------
    tuple[float, str]
        (best_brier, source_description) or (1.0, error_message) if not found.
    """
    # Prefer unified benchmark if available
    unified_path = os.path.join(results_dir, "synthesis", "unified_benchmark_summary.json")
    if os.path.exists(unified_path):
        with open(unified_path) as f:
            unified = json.load(f)
        best_brier = unified.get("best_brier", unified.get("best_contract_brier", 1.0))
        best_model = unified.get("best_model", "unknown")
        return best_brier, f"unified/{best_model} from {unified_path}"

    # Fallback to base benchmark
    benchmark_path = os.path.join(results_dir, "benchmark_summary.json")
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            results = json.load(f)
        best_brier = results.get("best_brier", 1.0)
        return best_brier, f"base benchmark from {benchmark_path}"

    return 1.0, "No benchmark results found"


# ===========================================================================
# Gate Category 1: Forecast Quality
# ===========================================================================

def check_forecast_quality(results_dir: str, thresholds: dict) -> list[PromotionGate]:
    """Check OOS Brier score gates.

    Verifies:
      - Overall OOS Brier is below the absolute threshold.
      - Model Brier beats the NWS baseline.
      - Model Brier beats Kalshi market (from real presettlement data).
      - No season has a Brier score above the seasonal threshold.
      - Sufficient OOS days are available for evaluation.

    Parameters
    ----------
    results_dir : str
        Path to city results directory.
    thresholds : dict
        City-specific promotion thresholds.

    Returns
    -------
    list[PromotionGate]
        List of evaluated forecast-quality gates.
    """
    gates: list[PromotionGate] = []
    brier_threshold = thresholds["brier_threshold"]
    nws_brier_baseline = thresholds["nws_brier_baseline"]
    seasonal_brier_threshold = thresholds["seasonal_brier_threshold"]
    min_oos_days = thresholds["min_oos_days"]

    # Load best Brier (unified > base fallback)
    best_brier, brier_source = _load_best_brier(results_dir)

    # Gate 1: Overall OOS Brier below absolute threshold
    gate = PromotionGate("overall_brier", "OOS Brier score below threshold")
    if best_brier < 1.0:
        gate.evaluate(best_brier, brier_threshold, "less")
        gate.details = (
            f"Best model Brier: {best_brier:.4f} "
            f"(threshold: {brier_threshold}, source: {brier_source})"
        )
    else:
        gate.details = f"Benchmark results not found ({brier_source})"
        gate.passed = False
    gates.append(gate)

    # Gate 2: Beats NWS baseline
    gate2 = PromotionGate("beats_nws", "Model Brier beats NWS baseline")
    if best_brier < 1.0:
        gate2.evaluate(best_brier, nws_brier_baseline, "less")
        gate2.details = (
            f"Model: {best_brier:.4f} vs NWS: {nws_brier_baseline:.4f} "
            f"(source: {brier_source})"
        )
    else:
        gate2.details = "Benchmark results not found"
        gate2.passed = False
    gates.append(gate2)

    # Gate 3: Beats Kalshi market Brier (from real presettlement data)
    gate3 = PromotionGate("beats_kalshi_market", "Model Brier beats Kalshi market")
    real_kalshi_bt_path = os.path.join(
        results_dir, "backtest", "real_kalshi_backtest_metrics.json"
    )
    if os.path.exists(real_kalshi_bt_path):
        with open(real_kalshi_bt_path) as f:
            rk = json.load(f)
        model_brier_rk = rk.get("model_brier", float("nan"))
        market_brier_rk = rk.get("market_brier", float("nan"))
        brier_edge_rk = rk.get("brier_edge", float("nan"))
        if not (np.isnan(model_brier_rk) or np.isnan(market_brier_rk)):
            # Model must have lower Brier than market (positive edge)
            gate3.evaluate(brier_edge_rk, 0.0, "greater")
            gate3.details = (
                f"Model: {model_brier_rk:.4f}, Market: {market_brier_rk:.4f}, "
                f"Edge: {brier_edge_rk:+.4f} (from real Kalshi presettlement)"
            )
        else:
            gate3.details = "Real Kalshi Brier scores are NaN"
            gate3.passed = False
    else:
        gate3.details = (
            f"Real Kalshi backtest not found at {real_kalshi_bt_path} "
            f"(skipped, non-blocking)"
        )
        # Non-blocking: if real Kalshi data doesn't exist yet, pass by default
        gate3.passed = True
    gates.append(gate3)

    # Gate 4: Seasonal stress test -- no season exceeds threshold
    gate4 = PromotionGate("seasonal_brier", "No season exceeds Brier threshold")
    seasonal_path = os.path.join(results_dir, "seasonal_brier.json")
    if os.path.exists(seasonal_path):
        with open(seasonal_path) as f:
            seasonal = json.load(f)
        worst_season = max(seasonal.values())
        worst_name = max(seasonal, key=seasonal.get)
        gate4.evaluate(worst_season, seasonal_brier_threshold, "less")
        gate4.details = (
            f"Worst season: {worst_name}={worst_season:.4f} "
            f"(threshold: {seasonal_brier_threshold})"
        )
    else:
        gate4.details = f"Seasonal results not found at {seasonal_path}"
        gate4.passed = False
    gates.append(gate4)

    # Gate 5: Minimum OOS evaluation days
    gate5 = PromotionGate("min_oos_days", "Sufficient OOS evaluation days")
    # Try unified first for OOS days, then base benchmark
    unified_path = os.path.join(
        results_dir, "synthesis", "unified_benchmark_summary.json"
    )
    benchmark_path = os.path.join(results_dir, "benchmark_summary.json")
    n_oos = 0
    if os.path.exists(unified_path):
        with open(unified_path) as f:
            unified = json.load(f)
        n_oos = unified.get("n_oos_days", 0)
    if n_oos == 0 and os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            results = json.load(f)
        n_oos = results.get("n_oos_days", 0)
    if n_oos > 0:
        gate5.evaluate(n_oos, min_oos_days, "greater")
        gate5.details = f"OOS days: {n_oos} (minimum: {min_oos_days})"
    else:
        gate5.details = "OOS day count not found in benchmark results"
        gate5.passed = False
    gates.append(gate5)

    return gates


# ===========================================================================
# Gate Category 2: Calibration
# ===========================================================================

def check_calibration(results_dir: str, thresholds: dict) -> list[PromotionGate]:
    """Check calibration quality gates.

    Verifies:
      - Expected Calibration Error is within tolerance.
      - Reliability diagram has been generated.

    Parameters
    ----------
    results_dir : str
        Path to city results directory.
    thresholds : dict
        City-specific promotion thresholds.

    Returns
    -------
    list[PromotionGate]
        List of evaluated calibration gates.
    """
    gates: list[PromotionGate] = []
    ece_threshold = thresholds["ece_threshold"]

    # Gate 1: ECE within tolerance
    gate = PromotionGate("ece", "Expected Calibration Error within tolerance")
    cal_path = os.path.join(results_dir, "synthesis", "calibration_summary.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        ece = cal.get("best_ece", 1.0)
        gate.evaluate(ece, ece_threshold, "less")
        gate.details = f"ECE: {ece:.4f} (threshold: {ece_threshold})"
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

def check_trading(results_dir: str, thresholds: dict) -> list[PromotionGate]:
    """Check trading simulation gates.

    Verifies:
      - Simulated paper-trading P&L is positive after fees and slippage.
      - Maximum drawdown is within acceptable limits.
      - Real Kalshi backtest P&L is positive (if data available).

    Parameters
    ----------
    results_dir : str
        Path to city results directory.
    thresholds : dict
        City-specific promotion thresholds.

    Returns
    -------
    list[PromotionGate]
        List of evaluated trading gates.
    """
    gates: list[PromotionGate] = []
    min_positive_pnl = thresholds["min_positive_pnl"]
    max_drawdown_threshold = thresholds["max_drawdown_threshold"]

    # Try combined backtest_summary.json first, fall back to backtest_metrics.json
    bt_path = os.path.join(results_dir, "backtest", "backtest_summary.json")
    bt_metrics_path = os.path.join(results_dir, "backtest", "backtest_metrics.json")

    # Gate 1: Positive P&L (simulated)
    gate = PromotionGate("positive_pnl", "Simulated paper trading shows positive P&L")
    bt_data = None
    if os.path.exists(bt_path):
        with open(bt_path) as f:
            bt_data = json.load(f)
        # Combined summary may have nested structure
        sim = bt_data.get("simulated_market", bt_data)
        total_pnl = sim.get("total_pnl", -999)
        gate.evaluate(total_pnl, min_positive_pnl, "greater")
        gate.details = f"Simulated P&L: ${total_pnl:.2f} (minimum: ${min_positive_pnl:.2f})"
    elif os.path.exists(bt_metrics_path):
        with open(bt_metrics_path) as f:
            bt_metrics = json.load(f)
        total_pnl = bt_metrics.get("total_pnl", -999)
        gate.evaluate(total_pnl, min_positive_pnl, "greater")
        gate.details = f"Simulated P&L: ${total_pnl:.2f} (minimum: ${min_positive_pnl:.2f})"
    else:
        gate.details = f"Backtest results not found at {bt_path} or {bt_metrics_path}"
        gate.passed = False
    gates.append(gate)

    # Gate 2: Max drawdown within limits
    gate2 = PromotionGate("max_drawdown", "Max drawdown within acceptable limits")
    if os.path.exists(bt_metrics_path):
        with open(bt_metrics_path) as f:
            bt_metrics = json.load(f)
        max_dd = bt_metrics.get("max_drawdown", -1.0)
        gate2.evaluate(max_dd, max_drawdown_threshold, "greater")
        gate2.details = f"Max drawdown: {max_dd:.1%} (limit: {max_drawdown_threshold:.1%})"
    elif os.path.exists(bt_path):
        with open(bt_path) as f:
            bt_data = json.load(f)
        sim = bt_data.get("simulated_market", bt_data)
        max_dd = sim.get("max_drawdown", sim.get("max_drawdown_pct", -1.0))
        gate2.evaluate(max_dd, max_drawdown_threshold, "greater")
        gate2.details = f"Max drawdown: {max_dd:.1%} (limit: {max_drawdown_threshold:.1%})"
    else:
        gate2.details = "Backtest results not found"
        gate2.passed = False
    gates.append(gate2)

    # Gate 3: Real Kalshi backtest positive P&L
    gate3 = PromotionGate("real_kalshi_pnl", "Real Kalshi backtest shows positive P&L")
    rk_path = os.path.join(results_dir, "backtest", "real_kalshi_backtest_metrics.json")
    if os.path.exists(rk_path):
        with open(rk_path) as f:
            rk = json.load(f)
        rk_pnl = rk.get("total_pnl", -999)
        rk_sharpe = rk.get("sharpe_ratio", 0.0)
        gate3.evaluate(rk_pnl, min_positive_pnl, "greater")
        gate3.details = (
            f"Real Kalshi P&L: ${rk_pnl:.2f}, Sharpe: {rk_sharpe:.2f} "
            f"(minimum P&L: ${min_positive_pnl:.2f})"
        )
    else:
        gate3.details = (
            f"Real Kalshi backtest not found at {rk_path} "
            f"(skipped, non-blocking)"
        )
        # Non-blocking: pass if data doesn't exist yet
        gate3.passed = True
    gates.append(gate3)

    return gates


# ===========================================================================
# Gate Category 4: Operational Readiness
# ===========================================================================

def check_operations(
    results_dir: str, data_dir: str, models_dir: str
) -> list[PromotionGate]:
    """Check operational readiness gates.

    Verifies:
      - All required processed data files exist.
      - Model checkpoints have been saved.
      - Scaler artifact exists for inference.

    Parameters
    ----------
    results_dir : str
        Path to city results directory.
    data_dir : str
        Path to city data directory.
    models_dir : str
        Path to city models directory.

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

def generate_report(
    all_gates: list[PromotionGate],
    output_path: str,
    cfg,
    thresholds: dict,
) -> dict:
    """Generate and save the promotion report as JSON.

    Parameters
    ----------
    all_gates : list[PromotionGate]
        All evaluated gates across all categories.
    output_path : str
        File path to write the JSON report.
    cfg : CityConfig
        The city configuration object.
    thresholds : dict
        City-specific promotion thresholds.

    Returns
    -------
    dict
        The full report dictionary.
    """
    total = len(all_gates)
    passed = sum(1 for g in all_gates if g.passed)

    report = {
        "timestamp": datetime.now().isoformat(),
        "city": cfg.city_code,
        "kalshi_ticker": cfg.kalshi_ticker,
        "target_station": cfg.target_station,
        "target_station_name": cfg.target_station_name,
        "overall_status": "PASS" if passed == total else "FAIL",
        "gates_passed": passed,
        "gates_total": total,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)",
        "thresholds": thresholds,
        "gates": [g.to_dict() for g in all_gates],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def plot_gate_summary(
    all_gates: list[PromotionGate],
    save_path: str,
    city_name: str,
    kalshi_ticker: str,
) -> None:
    """Create a horizontal bar chart summarizing gate pass/fail status.

    Parameters
    ----------
    all_gates : list[PromotionGate]
        All evaluated gates.
    save_path : str
        File path to save the figure.
    city_name : str
        Human-readable city name for chart title.
    kalshi_ticker : str
        Kalshi ticker for chart title.
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
    ax.set_title(f"{city_name} ({kalshi_ticker}) Promotion Gate Summary")

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
    city_upper = report.get("city", "").upper()
    ticker = report.get("kalshi_ticker", "")
    city_name_map = {
        "chi": "Chicago",
        "phl": "Philadelphia",
        "atl": "Atlanta",
        "aus": "Austin",
    }
    city_display = city_name_map.get(report.get("city", ""), city_upper)

    print("\n" + "=" * 70)
    print(f"{city_display.upper()} ({ticker}) PROMOTION READINESS EVALUATION")
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
        print(f"RECOMMENDATION: {city_display} model is READY for promotion.")
    else:
        print(f"RECOMMENDATION: {city_display} model is NOT YET ready for promotion.")
        failed = [g for g in report["gates"] if not g["passed"]]
        print(f"  {len(failed)} gate(s) need attention:")
        for g in failed:
            print(f"    - {g['name']}: {g['details']}")
    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict:
    """Run the full promotion readiness evaluation for a city.

    Parses ``--city`` from sys.argv, loads city config and thresholds,
    evaluates all promotion gates, and writes the report.

    Returns
    -------
    dict
        The promotion report dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Unified Model Promotion Readiness Evaluation"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=sorted(CITY_THRESHOLDS.keys()),
        help="City code to evaluate (chi, phl, atl, aus)",
    )
    args = parser.parse_args()
    city_code = args.city.strip().lower()

    # --- Load city config ---
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    thresholds = CITY_THRESHOLDS[city_code]

    results_dir = cfg.results_dir
    data_dir = cfg.data_dir
    models_dir = cfg.models_dir

    logger.info("=" * 70)
    logger.info(
        "%s (%s) Promotion Readiness Evaluation", cfg.city_name, cfg.kalshi_ticker
    )
    logger.info("=" * 70)
    logger.info("Results dir: %s", results_dir)
    logger.info("Data dir: %s", data_dir)
    logger.info("Models dir: %s", models_dir)

    all_gates: list[PromotionGate] = []

    # --- Category 1: Forecast Quality ---
    logger.info("-" * 50)
    logger.info("Category 1: Forecast Quality")
    logger.info("-" * 50)
    forecast_gates = check_forecast_quality(results_dir, thresholds)
    all_gates.extend(forecast_gates)
    for g in forecast_gates:
        logger.info(
            "  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details
        )

    # --- Category 2: Calibration ---
    logger.info("-" * 50)
    logger.info("Category 2: Calibration")
    logger.info("-" * 50)
    calibration_gates = check_calibration(results_dir, thresholds)
    all_gates.extend(calibration_gates)
    for g in calibration_gates:
        logger.info(
            "  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details
        )

    # --- Category 3: Trading Simulation ---
    logger.info("-" * 50)
    logger.info("Category 3: Trading Simulation")
    logger.info("-" * 50)
    trading_gates = check_trading(results_dir, thresholds)
    all_gates.extend(trading_gates)
    for g in trading_gates:
        logger.info(
            "  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details
        )

    # --- Category 4: Operational Readiness ---
    logger.info("-" * 50)
    logger.info("Category 4: Operational Readiness")
    logger.info("-" * 50)
    ops_gates = check_operations(results_dir, data_dir, models_dir)
    all_gates.extend(ops_gates)
    for g in ops_gates:
        logger.info(
            "  %s: %s — %s", g.name, "PASS" if g.passed else "FAIL", g.details
        )

    # --- Generate report ---
    report_path = os.path.join(results_dir, "promotion_report.json")
    report = generate_report(all_gates, report_path, cfg, thresholds)

    # --- Generate summary chart ---
    chart_path = os.path.join(results_dir, "promotion_gate_summary.png")
    plot_gate_summary(all_gates, chart_path, cfg.city_name, cfg.kalshi_ticker)

    # --- Print formatted report ---
    print_report(report)

    logger.info("Report saved to %s", report_path)
    logger.info("=" * 70)
    logger.info("Promotion evaluation complete. Overall: %s", report["overall_status"])
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    main()
