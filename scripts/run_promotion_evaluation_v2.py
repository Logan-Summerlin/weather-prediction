#!/usr/bin/env python3
"""
Multi-City Promotion Readiness Evaluation (v2).

Evaluates whether Chicago (KXHIGHCHI) and Philadelphia (KXHIGHPHL) model
pipelines meet the promotion gates required for production deployment.

Unlike the per-city v1 scripts, this version:
  - Reads from the actual files produced by the unified benchmark, honest
    benchmark, and backtest pipelines.
  - Computes ECE and seasonal Brier directly from unified_predictions.csv
    rather than relying on pre-computed summary files that may not exist.
  - Runs both cities in a single invocation for side-by-side comparison.

Promotion Gates (10):
  1. Overall Brier         — Best U-variant contract_brier below city threshold
  2. Beats Kalshi Market   — Best U-variant contract_brier < Kalshi pre-settlement
  3. No Seasonal Catastrophe — No season Brier > 0.22
  4. Sufficient OOS Data   — >= 100 unique OOS dates
  5. Calibration (ECE)     — ECE < 0.05 (computed from predictions)
  6. Positive Trading P&L  — Best backtest variant total_pnl > 0
  7. Acceptable Drawdown   — Max drawdown > -30%
  8. Data Pipeline Complete — All processed data files exist
  9. Model Predictions     — unified_predictions.csv exists with > 100 rows
 10. Test Infrastructure   — Pipeline test file exists

Outputs per city:
  - results/{city}/promotion_report_v2.json
  - results/{city}/promotion_summary_v2.png
  - Console pass/fail table

Usage:
    python scripts/run_promotion_evaluation_v2.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd

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
# Per-city promotion thresholds
# ---------------------------------------------------------------------------
CITY_THRESHOLDS = {
    "chi": {
        "brier_threshold": 0.16,
        "seasonal_brier_threshold": 0.22,
        "ece_threshold": 0.05,
        "min_oos_days": 100,
        "max_drawdown_threshold": -0.30,
        "min_predictions_rows": 100,
    },
    "phl": {
        "brier_threshold": 0.14,
        "seasonal_brier_threshold": 0.22,
        "ece_threshold": 0.05,
        "min_oos_days": 100,
        "max_drawdown_threshold": -0.30,
        "min_predictions_rows": 100,
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
        self.passed: bool = False
        self.value = None
        self.threshold = None
        self.details: str = ""

    def evaluate(self, value, threshold, comparison: str = "less") -> bool:
        """Evaluate whether the gate passes.

        Parameters
        ----------
        value : float
            The observed metric value.
        threshold : float
            The threshold to compare against.
        comparison : str
            ``"less"`` (value < threshold), ``"greater"`` (value > threshold),
            or ``"abs_less"`` (abs(value) < threshold).

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
# Helper: find best U-variant from unified benchmark results
# ===========================================================================

def _find_best_u_variant(benchmark: dict) -> tuple[str, float]:
    """Return (variant_name, contract_brier) for the best U-variant.

    Searches for keys starting with ``U`` that contain ``contract_brier``.
    Returns the variant with the lowest contract_brier.
    """
    best_name = None
    best_brier = float("inf")
    for name, metrics in benchmark.items():
        if not name.startswith("U"):
            continue
        cb = metrics.get("contract_brier")
        if cb is not None and cb < best_brier:
            best_brier = cb
            best_name = name
    return best_name, best_brier


# ===========================================================================
# Helper: compute ECE from predictions
# ===========================================================================

def _compute_ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Splits predicted probabilities into *n_bins* equal-width bins, then
    returns the weighted-average absolute difference between mean predicted
    probability and mean actual outcome per bin.

    Parameters
    ----------
    probs : array-like
        Predicted probabilities in [0, 1].
    outcomes : array-like
        Binary actual outcomes (0 or 1).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        The ECE value.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            # Last bin is inclusive on the right
            mask = (probs >= lo) & (probs <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        mean_pred = probs[mask].mean()
        mean_outcome = outcomes[mask].mean()
        ece += (n_in_bin / total) * abs(mean_pred - mean_outcome)
    return float(ece)


# ===========================================================================
# Helper: compute seasonal Brier from predictions
# ===========================================================================

def _compute_seasonal_brier(df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    """Compute per-season Brier score from a predictions DataFrame.

    Brier score = mean( (prob - outcome)^2 ) per season group.

    Parameters
    ----------
    df : DataFrame
        Must contain columns *prob_col*, ``actual_outcome``, and ``season``.
    prob_col : str
        Column name holding the predicted probability.

    Returns
    -------
    dict
        Mapping of season label (e.g. ``"DJF"``) to Brier score.
    """
    seasonal: dict[str, float] = {}
    for season, group in df.groupby("season"):
        probs = group[prob_col].values.astype(float)
        outcomes = group["actual_outcome"].values.astype(float)
        brier = float(np.mean((probs - outcomes) ** 2))
        seasonal[str(season)] = brier
    return seasonal


# ===========================================================================
# Helper: identify best U-variant probability column
# ===========================================================================

# Maps U-variant benchmark keys to their probability column name in the
# unified_predictions CSV.
_U_VARIANT_COL_MAP = {
    "U3_contract_mlp": "u3_mlp_prob",
    "U4_platt_on_u3": "u4_platt_prob",
    "U5_regime_conditional": "u5_regime_prob",
    "U6_calibrated_ensemble": "u6_ensemble_prob",
    "U7_extended_mlp": "u7_extended_prob",
    "U8_cv_ensemble": "u8_cv_prob",
    "U9_kitchen_sink": "u9_kitchen_prob",
}


def _best_u_prob_col(best_variant_name: str) -> str | None:
    """Return the predictions CSV column for the given U-variant name."""
    if best_variant_name in _U_VARIANT_COL_MAP:
        return _U_VARIANT_COL_MAP[best_variant_name]
    # Fallback: try substring matching
    for key, col in _U_VARIANT_COL_MAP.items():
        if best_variant_name.lower().replace("_", "") in key.lower().replace("_", ""):
            return col
    return None


# ===========================================================================
# Helper: identify best backtest variant
# ===========================================================================

def _find_best_backtest_variant(variants: dict) -> tuple[str, dict]:
    """Return (name, metrics_dict) for the backtest variant with highest P&L."""
    best_name = None
    best_pnl = -float("inf")
    best_metrics: dict = {}
    for name, metrics in variants.items():
        pnl = metrics.get("total_pnl", -float("inf"))
        if pnl > best_pnl:
            best_pnl = pnl
            best_name = name
            best_metrics = metrics
    return best_name, best_metrics


# ===========================================================================
# Gate evaluation functions
# ===========================================================================

def gate1_overall_brier(
    benchmark: dict | None,
    threshold: float,
) -> PromotionGate:
    """Gate 1: Best U-variant contract_brier below city threshold."""
    gate = PromotionGate("overall_brier", "Best model Brier below city threshold")
    if benchmark is None:
        gate.details = "unified_benchmark_results.json not found"
        return gate

    best_name, best_brier = _find_best_u_variant(benchmark)
    if best_name is None:
        gate.details = "No U-variant found in benchmark results"
        return gate

    gate.evaluate(best_brier, threshold, "less")
    gate.details = (
        f"{best_name} contract_brier={best_brier:.4f} "
        f"(threshold: {threshold})"
    )
    return gate


def gate2_beats_kalshi(benchmark: dict | None) -> PromotionGate:
    """Gate 2: Best U-variant contract_brier < Kalshi pre-settlement Brier."""
    gate = PromotionGate("beats_kalshi", "Best model Brier beats Kalshi market")
    if benchmark is None:
        gate.details = "unified_benchmark_results.json not found"
        return gate

    kalshi_entry = benchmark.get("Kalshi_PreSettlement", {})
    kalshi_brier = kalshi_entry.get("contract_brier")
    if kalshi_brier is None:
        gate.details = "Kalshi_PreSettlement entry missing from benchmark"
        return gate

    best_name, best_brier = _find_best_u_variant(benchmark)
    if best_name is None:
        gate.details = "No U-variant found in benchmark results"
        return gate

    gate.evaluate(best_brier, kalshi_brier, "less")
    gate.details = (
        f"{best_name}={best_brier:.4f} vs Kalshi={kalshi_brier:.4f} "
        f"(edge={kalshi_brier - best_brier:.4f})"
    )
    return gate


def gate3_seasonal_brier(
    predictions_df: pd.DataFrame | None,
    benchmark: dict | None,
    honest_benchmark: dict | None,
    threshold: float,
) -> PromotionGate:
    """Gate 3: No season's Brier > threshold.

    Tries to compute from unified_predictions.csv using the best U-variant
    probability column.  Falls back to honest_benchmark_results.json seasonal
    breakdowns if the predictions file is unavailable.
    """
    gate = PromotionGate("seasonal_brier", "No season exceeds Brier threshold")

    # Strategy A: compute directly from predictions
    if predictions_df is not None and benchmark is not None:
        best_name, _ = _find_best_u_variant(benchmark)
        prob_col = _best_u_prob_col(best_name) if best_name else None
        if prob_col and prob_col in predictions_df.columns:
            seasonal = _compute_seasonal_brier(predictions_df, prob_col)
            if seasonal:
                worst_name = max(seasonal, key=seasonal.get)
                worst_val = seasonal[worst_name]
                gate.evaluate(worst_val, threshold, "less")
                breakdown = ", ".join(
                    f"{s}={v:.4f}" for s, v in sorted(seasonal.items())
                )
                gate.details = (
                    f"Worst: {worst_name}={worst_val:.4f} "
                    f"(threshold: {threshold}) [{breakdown}]"
                )
                return gate

    # Strategy B: fall back to honest_benchmark_results.json
    if honest_benchmark is not None:
        # Find the best E-model (lowest test_brier) that has seasonal data
        best_e_name = None
        best_e_brier = float("inf")
        for name, metrics in honest_benchmark.items():
            tb = metrics.get("test_brier")
            if tb is not None and "seasonal" in metrics and tb < best_e_brier:
                best_e_brier = tb
                best_e_name = name
        if best_e_name is not None:
            seasonal = honest_benchmark[best_e_name]["seasonal"]
            worst_name = max(seasonal, key=seasonal.get)
            worst_val = seasonal[worst_name]
            gate.evaluate(worst_val, threshold, "less")
            breakdown = ", ".join(
                f"{s}={v:.4f}" for s, v in sorted(seasonal.items())
            )
            gate.details = (
                f"Worst: {worst_name}={worst_val:.4f} "
                f"(threshold: {threshold}) [from {best_e_name}: {breakdown}]"
            )
            return gate

    gate.details = "Could not compute seasonal Brier: no predictions or honest benchmark data"
    return gate


def gate4_sufficient_oos(
    predictions_df: pd.DataFrame | None,
    min_days: int,
) -> PromotionGate:
    """Gate 4: At least *min_days* unique OOS dates."""
    gate = PromotionGate("sufficient_oos", "Sufficient OOS evaluation days")
    if predictions_df is None:
        gate.details = "unified_predictions.csv not found"
        return gate

    oos_mask = predictions_df["period"].str.upper() == "OOS"
    n_oos_dates = predictions_df.loc[oos_mask, "date"].nunique()
    gate.evaluate(n_oos_dates, min_days, "greater")
    gate.details = f"OOS dates: {n_oos_dates} (minimum: {min_days})"
    return gate


def gate5_calibration_ece(
    predictions_df: pd.DataFrame | None,
    benchmark: dict | None,
    threshold: float,
) -> PromotionGate:
    """Gate 5: ECE < threshold, computed from predictions."""
    gate = PromotionGate("calibration_ece", "Expected Calibration Error within tolerance")
    if predictions_df is None or benchmark is None:
        gate.details = "unified_predictions.csv or benchmark not available"
        return gate

    best_name, _ = _find_best_u_variant(benchmark)
    prob_col = _best_u_prob_col(best_name) if best_name else None

    if prob_col is None or prob_col not in predictions_df.columns:
        gate.details = (
            f"Best U-variant '{best_name}' has no matching probability column "
            f"in predictions (tried '{prob_col}')"
        )
        return gate

    probs = predictions_df[prob_col].values.astype(float)
    outcomes = predictions_df["actual_outcome"].values.astype(float)

    # Remove NaN rows
    valid = ~(np.isnan(probs) | np.isnan(outcomes))
    probs = probs[valid]
    outcomes = outcomes[valid]

    ece = _compute_ece(probs, outcomes, n_bins=10)
    gate.evaluate(ece, threshold, "less")
    gate.details = f"ECE={ece:.4f} using {best_name} (threshold: {threshold}, n={len(probs)})"
    return gate


def gate6_positive_pnl(backtest: dict | None) -> PromotionGate:
    """Gate 6: Best backtest variant has positive total P&L."""
    gate = PromotionGate("positive_pnl", "Best variant trading P&L > 0")
    if backtest is None:
        gate.details = "backtest/real_kalshi_metrics.json not found"
        return gate

    variants = backtest.get("variants", {})
    if not variants:
        gate.details = "No variants found in backtest metrics"
        return gate

    best_name, best_metrics = _find_best_backtest_variant(variants)
    pnl = best_metrics.get("total_pnl", -999.0)
    gate.evaluate(pnl, 0.0, "greater")
    gate.details = (
        f"{best_name} total_pnl=${pnl:.2f} "
        f"(Sharpe={best_metrics.get('sharpe_ratio', 0):.2f}, "
        f"n_trades={best_metrics.get('n_trades', 0)})"
    )
    return gate


def gate7_acceptable_drawdown(
    backtest: dict | None,
    threshold: float,
) -> PromotionGate:
    """Gate 7: Best variant max drawdown > threshold (less negative)."""
    gate = PromotionGate("acceptable_drawdown", "Max drawdown within limits")
    if backtest is None:
        gate.details = "backtest/real_kalshi_metrics.json not found"
        return gate

    variants = backtest.get("variants", {})
    if not variants:
        gate.details = "No variants found in backtest metrics"
        return gate

    best_name, best_metrics = _find_best_backtest_variant(variants)
    # max_drawdown_pct is stored as a percentage (e.g. -7.35 means -7.35%)
    dd_pct = best_metrics.get("max_drawdown_pct", -100.0)
    # Convert from percentage to fraction for comparison
    dd_frac = dd_pct / 100.0
    gate.evaluate(dd_frac, threshold, "greater")
    gate.details = (
        f"{best_name} max_drawdown={dd_frac:.2%} "
        f"(limit: {threshold:.0%})"
    )
    return gate


def gate8_data_pipeline(data_dir: str) -> PromotionGate:
    """Gate 8: All required processed data files exist."""
    gate = PromotionGate("data_pipeline", "Data pipeline complete — processed files exist")
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
    found = []
    missing = []
    for f in required_files:
        if os.path.exists(os.path.join(processed_dir, f)):
            found.append(f)
        else:
            missing.append(f)

    gate.passed = len(missing) == 0
    gate.value = len(found)
    gate.threshold = len(required_files)
    if missing:
        gate.details = (
            f"{len(found)}/{len(required_files)} present. "
            f"Missing: {', '.join(missing)}"
        )
    else:
        gate.details = f"All {len(required_files)} required files present in {processed_dir}"
    return gate


def gate9_predictions_available(results_dir: str, min_rows: int) -> PromotionGate:
    """Gate 9: unified_predictions.csv exists and has reasonable data."""
    gate = PromotionGate("predictions_available", "Unified predictions file exists with data")
    pred_path = os.path.join(results_dir, "unified_predictions.csv")
    if not os.path.exists(pred_path):
        gate.details = f"File not found: {pred_path}"
        return gate

    try:
        df = pd.read_csv(pred_path)
        n_rows = len(df)
        gate.evaluate(n_rows, min_rows, "greater")
        gate.details = f"{n_rows} rows (minimum: {min_rows}) in {pred_path}"
    except Exception as e:
        gate.details = f"Error reading predictions file: {e}"
    return gate


def gate10_test_infrastructure(city_code: str) -> PromotionGate:
    """Gate 10: Pipeline test file exists."""
    gate = PromotionGate("test_infrastructure", "Pipeline test file exists")
    test_path = PROJECT_ROOT / "tests" / f"test_{city_code}_pipeline.py"
    if test_path.exists():
        gate.passed = True
        gate.details = f"Found: {test_path}"
    else:
        gate.passed = False
        gate.details = f"Not found: {test_path}"
    return gate


# ===========================================================================
# Orchestrator: evaluate all gates for a single city
# ===========================================================================

def evaluate_city(city_code: str) -> tuple[list[PromotionGate], dict]:
    """Run all 10 promotion gates for *city_code*.

    Parameters
    ----------
    city_code : str
        Short city code (``"chi"`` or ``"phl"``).

    Returns
    -------
    gates : list[PromotionGate]
        All 10 evaluated gates.
    report : dict
        The full JSON-serializable report.
    """
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)
    thresholds = CITY_THRESHOLDS[city_code]

    results_dir = cfg.results_dir
    data_dir = cfg.data_dir

    # ----- Load data files (gracefully handle missing) -----
    # 1. Unified benchmark
    benchmark_path = os.path.join(results_dir, "unified_benchmark_results.json")
    benchmark: dict | None = None
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            benchmark = json.load(f)

    # 2. Honest benchmark
    honest_path = os.path.join(results_dir, "honest_benchmark_results.json")
    honest_benchmark: dict | None = None
    if os.path.exists(honest_path):
        with open(honest_path) as f:
            honest_benchmark = json.load(f)

    # 3. Backtest metrics
    backtest_path = os.path.join(results_dir, "backtest", "real_kalshi_metrics.json")
    backtest: dict | None = None
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            backtest = json.load(f)

    # 4. Unified predictions
    pred_path = os.path.join(results_dir, "unified_predictions.csv")
    predictions_df: pd.DataFrame | None = None
    if os.path.exists(pred_path):
        predictions_df = pd.read_csv(pred_path)

    # ----- Evaluate gates -----
    gates: list[PromotionGate] = []

    # Gate 1: Overall Brier
    gates.append(gate1_overall_brier(benchmark, thresholds["brier_threshold"]))

    # Gate 2: Beats Kalshi
    gates.append(gate2_beats_kalshi(benchmark))

    # Gate 3: Seasonal Brier
    gates.append(gate3_seasonal_brier(
        predictions_df, benchmark, honest_benchmark,
        thresholds["seasonal_brier_threshold"],
    ))

    # Gate 4: Sufficient OOS
    gates.append(gate4_sufficient_oos(predictions_df, thresholds["min_oos_days"]))

    # Gate 5: Calibration ECE
    gates.append(gate5_calibration_ece(
        predictions_df, benchmark, thresholds["ece_threshold"],
    ))

    # Gate 6: Positive P&L
    gates.append(gate6_positive_pnl(backtest))

    # Gate 7: Acceptable Drawdown
    gates.append(gate7_acceptable_drawdown(backtest, thresholds["max_drawdown_threshold"]))

    # Gate 8: Data Pipeline
    gates.append(gate8_data_pipeline(data_dir))

    # Gate 9: Predictions Available
    gates.append(gate9_predictions_available(results_dir, thresholds["min_predictions_rows"]))

    # Gate 10: Test Infrastructure
    gates.append(gate10_test_infrastructure(city_code))

    # ----- Build report -----
    total = len(gates)
    passed = sum(1 for g in gates if g.passed)

    report = {
        "timestamp": datetime.now().isoformat(),
        "city": cfg.city_name,
        "city_code": city_code,
        "kalshi_ticker": cfg.kalshi_ticker,
        "target_station": cfg.target_station,
        "target_station_name": cfg.target_station_name,
        "overall_status": "PASS" if passed == total else "FAIL",
        "gates_passed": passed,
        "gates_total": total,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)",
        "thresholds": thresholds,
        "gates": [g.to_dict() for g in gates],
    }

    return gates, report


# ===========================================================================
# Report I/O
# ===========================================================================

def save_report(report: dict, output_path: str) -> None:
    """Persist the promotion report as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved report to %s", output_path)


def plot_gate_summary(
    all_gates: list[PromotionGate],
    city_name: str,
    ticker: str,
    save_path: str,
) -> None:
    """Create a horizontal bar chart showing PASS/FAIL for each gate.

    Parameters
    ----------
    all_gates : list[PromotionGate]
        All evaluated gates.
    city_name : str
        Human-readable city name for the chart title.
    ticker : str
        Kalshi ticker for the chart title.
    save_path : str
        File path to save the PNG figure.
    """
    names = [g.name for g in all_gates]
    statuses = [1 if g.passed else 0 for g in all_gates]
    colors = ["#2ca02c" if s else "#d62728" for s in statuses]
    labels = ["PASS" if s else "FAIL" for s in statuses]

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.55)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, statuses, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.1, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title(f"{city_name} ({ticker}) Promotion Gate Summary (v2)")

    for bar, label, gate in zip(bars, labels, all_gates):
        x_pos = bar.get_width() + 0.05
        detail_text = gate.details[:70] if gate.details else ""
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


def print_report(report: dict) -> None:
    """Print a formatted promotion report to the console."""
    city = report["city"]
    ticker = report["kalshi_ticker"]

    print()
    print("=" * 78)
    print(f"  {city.upper()} ({ticker}) — PROMOTION READINESS EVALUATION v2")
    print("=" * 78)
    print(f"  Timestamp      : {report['timestamp']}")
    print(f"  Target Station : {report['target_station_name']} ({report['target_station']})")
    print(f"  Overall Status : {report['overall_status']}")
    print(f"  Gates          : {report['pass_rate']}")
    print("-" * 78)

    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        icon = "[+]" if gate["passed"] else "[-]"
        print(f"  {icon} {gate['name']:25s} {status}")
        print(f"      {gate['details']}")

    print("-" * 78)
    if report["overall_status"] == "PASS":
        print(f"  RECOMMENDATION: {city} model is READY for promotion.")
    else:
        failed = [g for g in report["gates"] if not g["passed"]]
        print(f"  RECOMMENDATION: {city} model is NOT YET ready for promotion.")
        print(f"  {len(failed)} gate(s) need attention:")
        for g in failed:
            print(f"    - {g['name']}: {g['details']}")
    print("=" * 78)
    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> dict[str, dict]:
    """Run promotion evaluation for Chicago and Philadelphia.

    Returns
    -------
    dict
        Mapping of city_code to its promotion report dictionary.
    """
    logger.info("=" * 78)
    logger.info("Multi-City Promotion Readiness Evaluation v2")
    logger.info("=" * 78)

    all_reports: dict[str, dict] = {}

    for city_code in ["chi", "phl"]:
        cfg = get_city_config(city_code)
        logger.info("-" * 60)
        logger.info("Evaluating %s (%s)", cfg.city_name, cfg.kalshi_ticker)
        logger.info("-" * 60)

        gates, report = evaluate_city(city_code)
        all_reports[city_code] = report

        # --- Save JSON report ---
        report_path = os.path.join(cfg.results_dir, "promotion_report_v2.json")
        save_report(report, report_path)

        # --- Save summary chart ---
        chart_path = os.path.join(cfg.results_dir, "promotion_summary_v2.png")
        plot_gate_summary(gates, cfg.city_name, cfg.kalshi_ticker, chart_path)

        # --- Console output ---
        print_report(report)

    # --- Cross-city summary ---
    print()
    print("=" * 78)
    print("  CROSS-CITY PROMOTION SUMMARY")
    print("=" * 78)
    for code, rpt in all_reports.items():
        status = rpt["overall_status"]
        print(f"  {rpt['city']:20s} ({rpt['kalshi_ticker']})  {rpt['pass_rate']:>12s}  {status}")
    print("=" * 78)
    print()

    logger.info("Promotion evaluation v2 complete for all cities.")
    return all_reports


if __name__ == "__main__":
    main()
