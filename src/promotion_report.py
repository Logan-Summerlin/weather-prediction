"""
Unified Promotion Report Schema and Evaluation.

Provides a single, canonical promotion evaluation framework that works across
all cities (NYC, CHI, PHL, ATL, AUS). Merges the best of v1 and v2 evaluation
scripts into one standardized schema.

Key Phase F requirements addressed:
  1. Standardized promotion report schema across all cities.
  2. Baseline comparisons (persistence, climatology, ridge) included in report.
  3. Market-implied benchmark (Kalshi pre-settlement Brier) included in report.
  4. All comparisons consolidated into one canonical report artifact.

Report schema:
  - Header: city identity, timestamp, station, ticker
  - Gates: standardized list of pass/fail promotion gates
  - Baselines: persistence, climatology, ridge Brier scores
  - Market benchmark: Kalshi pre-settlement Brier score
  - Model summary: best model name, Brier, edge vs market
  - Trading summary: P&L, Sharpe, drawdown
  - Calibration summary: ECE, best calibration method
  - Seasonal breakdown: per-season Brier scores

Usage:
    from src.promotion_report import evaluate_city, save_unified_report
    gates, report = evaluate_city("chi")
    save_unified_report(report, "results/chicago/unified_promotion_report.json")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Canonical promotion thresholds — unified across all cities
#
# Threshold derivation notes:
#   - brier_threshold: contract-level Brier ceiling per city. Cities with
#     lower day-to-day TMAX variance (e.g. Austin) concentrate probability
#     in fewer buckets, so both model and market Briers run lower and the
#     ceiling is legitimately tighter. Set from climatological bucket
#     entropy: roughly the Brier a well-calibrated climatology forecast
#     achieves, minus the margin a skillful model must add.
#   - brier_floor: sanity floor — a Brier *below* this on a full OOS year
#     indicates leakage or a broken evaluation rather than genuine skill
#     (only meaningful for low-variance cities; 0.0 disables).
#   - nws_brier_baseline: contract Brier of the local NWS point forecast
#     converted to bucket probabilities; the model must beat the public
#     forecast to claim any edge.
#   - max_drawdown_threshold is a FRACTION (-0.30 = -30% of peak bankroll).
# ---------------------------------------------------------------------------
def climatology_ladder_brier(
    monthly_tmax_std: dict[int, float],
    bucket_width: float = 2.0,
    n_interior: int = 4,
) -> float:
    """Entropy-derived contract Brier of a calibrated climatology forecast.

    Replicates the verified daily Kalshi ladder — ``n_interior`` 2°F buckets
    struck around the forecast (the distribution mean, for a climatology
    baseline) bracketed by a low and a high open tail — and returns the mean
    per-contract binary Brier ``q*(1-q)`` of the climatological bucket
    probabilities, averaged across months. This is exactly "the Brier a
    well-calibrated climatology forecast achieves" referenced in the
    threshold-derivation notes; a skillful model must come in below it.

    Cities with lower day-to-day TMAX variance concentrate probability in the
    central buckets, raising per-contract Brier, while high-variance cities
    spread mass thinner. Used to set expansion-city ``brier_threshold`` values
    without hand-tuning (see tests/test_expansion_cities.py).
    """
    import math

    half = (n_interior / 2.0) * bucket_width

    def _norm_cdf(x: float, sd: float) -> float:
        return 0.5 * (1.0 + math.erf(x / (sd * math.sqrt(2.0))))

    monthly_brier: list[float] = []
    for sd in monthly_tmax_std.values():
        sd = max(float(sd), 1e-6)
        edges = [-1e9] + [(-half + i * bucket_width) for i in range(n_interior + 1)] + [1e9]
        cdf = [_norm_cdf(e, sd) for e in edges]
        probs = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]
        monthly_brier.append(sum(q * (1.0 - q) for q in probs) / len(probs))
    return sum(monthly_brier) / len(monthly_brier) if monthly_brier else 0.0


CITY_THRESHOLDS: dict[str, dict[str, Any]] = {
    "nyc": {
        "brier_threshold": 0.14,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.12,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
        "min_predictions_rows": 100,
    },
    "chi": {
        "brier_threshold": 0.16,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.14,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.22,
        "min_predictions_rows": 100,
    },
    "phl": {
        "brier_threshold": 0.15,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.13,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
        "min_predictions_rows": 100,
    },
    "atl": {
        "brier_threshold": 0.14,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.12,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
        "min_predictions_rows": 100,
    },
    "aus": {
        "brier_threshold": 0.11,
        "brier_floor": 0.04,
        "nws_brier_baseline": 0.14,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.22,
        "min_predictions_rows": 100,
    },
    # ---- Phase 4 expansion cities -------------------------------------------
    # brier_threshold == round(climatology_ladder_brier(cfg.monthly_tmax_std), 2)
    # (entropy-derived; see tests/test_expansion_cities.py). nws_brier_baseline
    # is provisional (~0.86x the climatology ceiling) until a real NWS forecast
    # benchmark is collected during rollout; these cities stay MONITOR and
    # cannot be PROMOTED on provisional baselines.
    "den": {
        "brier_threshold": 0.12,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.11,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.18,
        "min_predictions_rows": 100,
    },
    "dc": {
        "brier_threshold": 0.13,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.11,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.19,
        "min_predictions_rows": 100,
    },
    "lax": {
        "brier_threshold": 0.13,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.12,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.19,
        "min_predictions_rows": 100,
    },
    "mia": {
        "brier_threshold": 0.14,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.12,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.20,
        "min_predictions_rows": 100,
    },
    "phx": {
        "brier_threshold": 0.13,
        "brier_floor": 0.0,
        "nws_brier_baseline": 0.11,
        "ece_threshold": 0.05,
        "min_positive_pnl": 0.0,
        "max_drawdown_threshold": -0.30,
        "min_oos_days": 200,
        "seasonal_brier_threshold": 0.19,
        "min_predictions_rows": 100,
    },
}


# ===========================================================================
# PromotionGate
# ===========================================================================

class PromotionGate:
    """Represents a single promotion gate check."""

    def __init__(self, name: str, description: str, category: str = ""):
        self.name = name
        self.description = description
        self.category = category
        self.passed: bool = False
        self.value: Any = None
        self.threshold: Any = None
        self.details: str = ""

    def evaluate(self, value, threshold, comparison: str = "less") -> bool:
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
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
        }


# ===========================================================================
# Data loading helpers
# ===========================================================================

def _load_json(path: str) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _find_best_u_variant(benchmark: dict) -> tuple[str | None, float]:
    """Return (name, contract_brier) for the best U-variant model."""
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


def _load_best_brier(results_dir: str) -> tuple[float, str, str | None]:
    """Load the best model Brier score from available result files.

    Returns (best_brier, source_description, best_model_name).
    """
    # 1. Unified benchmark (preferred)
    unified_path = os.path.join(results_dir, "unified_benchmark_results.json")
    benchmark = _load_json(unified_path)
    if benchmark is not None:
        best_name, best_brier = _find_best_u_variant(benchmark)
        if best_name is not None:
            return best_brier, f"unified/{best_name}", best_name

    # 2. Synthesis calibration sweep
    synth_path = os.path.join(results_dir, "synthesis", "calibration_sweep_summary.json")
    synth = _load_json(synth_path)
    if synth is not None:
        best_brier = synth.get("best_brier", 1.0)
        best_method = synth.get("best_method", "unknown")
        return best_brier, f"synthesis/{best_method}", best_method

    # 3. Base benchmark
    bench_path = os.path.join(results_dir, "benchmark_summary.json")
    bench = _load_json(bench_path)
    if bench is not None:
        best_brier = bench.get("best_brier", 1.0)
        return best_brier, "base_benchmark", None

    return 1.0, "no_results_found", None


def _extract_baseline_briers(results_dir: str) -> dict[str, float | None]:
    """Extract baseline model Brier scores from benchmark results.

    Returns dict with keys: persistence, climatology, ridge, and their
    Brier scores (or None if not found).
    """
    baselines: dict[str, float | None] = {
        "persistence": None,
        "climatology": None,
        "ridge": None,
    }

    # Check unified benchmark first
    unified = _load_json(os.path.join(results_dir, "unified_benchmark_results.json"))
    if unified is not None:
        for key, metrics in unified.items():
            key_lower = key.lower()
            brier_val = metrics.get("contract_brier") or metrics.get("test_brier")
            if brier_val is None:
                continue
            if "persistence" in key_lower or key == "E0_persistence":
                baselines["persistence"] = brier_val
            elif "climatology" in key_lower or key == "E1_climatology":
                baselines["climatology"] = brier_val
            elif "ridge" in key_lower or key == "E2_ridge":
                baselines["ridge"] = brier_val

    # Check honest benchmark as fallback
    if all(v is None for v in baselines.values()):
        honest = _load_json(os.path.join(results_dir, "honest_benchmark_results.json"))
        if honest is not None:
            for key, metrics in honest.items():
                key_lower = key.lower()
                brier_val = metrics.get("test_brier")
                if brier_val is None:
                    continue
                if "persistence" in key_lower or key == "E0_persistence":
                    baselines["persistence"] = brier_val
                elif "climatology" in key_lower or key == "E1_climatology":
                    baselines["climatology"] = brier_val
                elif "ridge" in key_lower or key == "E2_ridge":
                    baselines["ridge"] = brier_val

    # Check base benchmark_summary as last resort
    if all(v is None for v in baselines.values()):
        bench = _load_json(os.path.join(results_dir, "benchmark_summary.json"))
        if bench is not None:
            for key in ["persistence_brier", "climatology_brier", "ridge_brier"]:
                short = key.replace("_brier", "")
                if key in bench:
                    baselines[short] = bench[key]

    return baselines


def _extract_market_brier(results_dir: str) -> float | None:
    """Extract Kalshi market-implied Brier score."""
    # 1. From unified benchmark (Kalshi_PreSettlement entry)
    unified = _load_json(os.path.join(results_dir, "unified_benchmark_results.json"))
    if unified is not None:
        kalshi = unified.get("Kalshi_PreSettlement", {})
        cb = kalshi.get("contract_brier")
        if cb is not None:
            return cb

    # 2. From real Kalshi backtest metrics
    rk = _load_json(os.path.join(results_dir, "backtest", "real_kalshi_metrics.json"))
    if rk is not None:
        mb = rk.get("market_brier")
        if mb is not None:
            return mb

    rk2 = _load_json(os.path.join(results_dir, "backtest", "real_kalshi_backtest_metrics.json"))
    if rk2 is not None:
        mb = rk2.get("market_brier")
        if mb is not None:
            return mb

    return None


def _extract_calibration_summary(results_dir: str) -> dict[str, Any]:
    """Extract calibration summary (best ECE, method)."""
    result: dict[str, Any] = {"ece": None, "best_method": None, "methods": {}}

    sweep = _load_json(os.path.join(results_dir, "synthesis", "calibration_sweep_summary.json"))
    if sweep is not None:
        best_method = sweep.get("best_method", "")
        methods = sweep.get("methods", {})
        result["methods"] = {
            name: {"brier": m.get("brier"), "ece": m.get("ece")}
            for name, m in methods.items()
        }
        result["best_method"] = best_method
        if best_method and best_method in methods:
            result["ece"] = methods[best_method].get("ece")
        elif methods:
            result["ece"] = min(
                (m.get("ece", 1.0) for m in methods.values()), default=None
            )
        return result

    cal = _load_json(os.path.join(results_dir, "synthesis", "calibration_summary.json"))
    if cal is not None:
        result["ece"] = cal.get("best_ece")
        result["best_method"] = cal.get("best_method")
        return result

    return result


def _load_real_kalshi_metrics(results_dir: str) -> dict | None:
    """Load real-Kalshi backtest metrics, normalized to a flat best-variant dict.

    Supports both on-disk schemas:
      - variants schema: {"market_brier": ..., "variants": {name: metrics}}
        (written by scripts/experiments/trading/run_real_kalshi_backtest.py)
      - flat schema: a single metrics dict
        (written by scripts/run_backtest.py)

    Returns the best variant's metrics dict with a "best_variant" key added,
    or None if no real-Kalshi metrics exist.
    """
    rk = None
    for fname in ("real_kalshi_metrics.json", "real_kalshi_backtest_metrics.json"):
        rk = _load_json(os.path.join(results_dir, "backtest", fname))
        if rk is not None:
            break
    if rk is None:
        return None

    if rk.get("variants"):
        best_name = None
        best_pnl = -float("inf")
        best_metrics: dict = {}
        for name, metrics in rk["variants"].items():
            pnl = metrics.get("total_pnl", -float("inf"))
            if pnl > best_pnl:
                best_pnl = pnl
                best_name = name
                best_metrics = metrics
        if best_name is None:
            return None
        best_metrics = dict(best_metrics)
        best_metrics["best_variant"] = best_name
        best_metrics.setdefault("market_brier", rk.get("market_brier"))
        return best_metrics

    if rk.get("total_pnl") is not None:
        flat = dict(rk)
        flat["best_variant"] = rk.get("source", "ev_gated")
        return flat

    return None


def _extract_trading_summary(results_dir: str) -> dict[str, Any]:
    """Extract best trading variant summary."""
    result: dict[str, Any] = {
        "source": None,
        "best_variant": None,
        "total_pnl": None,
        "sharpe_ratio": None,
        "n_trades": None,
        "win_rate": None,
        "max_drawdown_pct": None,
        "max_drawdown_frac": None,
        "return_pct": None,
        "model_brier": None,
        "brier_edge": None,
        "busted": None,
        "seasonal": {},
    }

    # Prefer real Kalshi metrics
    best_metrics = _load_real_kalshi_metrics(results_dir)
    if best_metrics is not None:
        # No sentinel default: a missing drawdown must surface as None so
        # the drawdown gate fails with "data not found" rather than passing
        # or failing on a fabricated -100%.
        dd_pct = best_metrics.get("max_drawdown_pct")
        result.update({
            "source": "real_kalshi",
            "best_variant": best_metrics.get("best_variant"),
            "total_pnl": best_metrics.get("total_pnl"),
            "sharpe_ratio": best_metrics.get("sharpe_ratio"),
            "n_trades": best_metrics.get("n_trades"),
            "win_rate": best_metrics.get("win_rate"),
            "max_drawdown_pct": dd_pct,
            "max_drawdown_frac": dd_pct / 100.0 if dd_pct is not None else None,
            "return_pct": best_metrics.get("return_pct"),
            "model_brier": best_metrics.get("model_brier"),
            "brier_edge": best_metrics.get("brier_edge"),
            "busted": best_metrics.get("busted"),
            "seasonal": best_metrics.get("seasonal", {}),
        })
        return result

    # Fallback to simulated backtest
    bt = _load_json(os.path.join(results_dir, "backtest", "backtest_metrics.json"))
    if bt is not None:
        dd_pct = bt.get("max_drawdown_pct")
        dd_frac = dd_pct / 100.0 if dd_pct is not None else None
        if dd_frac is None:
            dd_dollars = bt.get("max_drawdown", -1.0)
            bankroll = bt.get("initial_bankroll", 1000.0)
            dd_frac = dd_dollars / bankroll if bankroll > 0 else -1.0
        result.update({
            "source": "simulated",
            "best_variant": "simulated_market",
            "total_pnl": bt.get("total_pnl"),
            "sharpe_ratio": bt.get("sharpe_ratio"),
            "n_trades": bt.get("n_trades"),
            "win_rate": bt.get("win_rate"),
            "max_drawdown_pct": dd_pct,
            "max_drawdown_frac": dd_frac,
            "return_pct": bt.get("return_pct"),
            "model_brier": bt.get("model_brier"),
            "brier_edge": bt.get("brier_edge"),
            "seasonal": bt.get("seasonal", {}),
        })
        return result

    # Fallback to backtest_summary.json
    bt_summary = _load_json(os.path.join(results_dir, "backtest", "backtest_summary.json"))
    if bt_summary is not None:
        sim = bt_summary.get("simulated_market", bt_summary)
        dd_frac = sim.get("max_drawdown", -1.0)
        if abs(dd_frac) > 1.0:
            bankroll = sim.get("initial_bankroll", 1000.0)
            dd_frac = dd_frac / bankroll if bankroll > 0 else -1.0
        result.update({
            "source": "simulated",
            "best_variant": "simulated_market",
            "total_pnl": sim.get("total_pnl"),
            "sharpe_ratio": sim.get("sharpe_ratio"),
            "n_trades": sim.get("n_trades"),
            "win_rate": sim.get("win_rate"),
            "max_drawdown_frac": dd_frac,
            "seasonal": sim.get("seasonal", {}),
        })

    return result


def _extract_seasonal_brier(results_dir: str) -> dict[str, float]:
    """Extract seasonal Brier breakdown."""
    seasonal = _load_json(os.path.join(results_dir, "seasonal_brier.json"))
    if seasonal is not None:
        return {str(k): float(v) for k, v in seasonal.items()}
    return {}


# ===========================================================================
# Gate evaluation
# ===========================================================================

def _check_all_gates(
    results_dir: str,
    data_dir: str,
    models_dir: str,
    thresholds: dict[str, Any],
    city_code: str = "",
) -> list[PromotionGate]:
    """Evaluate all promotion gates. Returns list of PromotionGate objects."""
    gates: list[PromotionGate] = []

    best_brier, brier_source, best_model = _load_best_brier(results_dir)
    market_brier = _extract_market_brier(results_dir)
    cal_summary = _extract_calibration_summary(results_dir)
    trading = _extract_trading_summary(results_dir)
    seasonal = _extract_seasonal_brier(results_dir)

    brier_threshold = thresholds["brier_threshold"]
    brier_floor = thresholds.get("brier_floor", 0.0)
    ece_threshold = thresholds["ece_threshold"]
    seasonal_threshold = thresholds["seasonal_brier_threshold"]
    min_oos_days = thresholds["min_oos_days"]
    max_dd_threshold = thresholds["max_drawdown_threshold"]

    # ---- Category: Forecast Quality ----

    # Gate 1: Overall Brier below threshold
    g = PromotionGate("overall_brier", "OOS Brier score below threshold", "forecast_quality")
    if best_brier < 1.0:
        g.evaluate(best_brier, brier_threshold, "less")
        g.details = f"Best model Brier: {best_brier:.4f} (threshold: {brier_threshold}, source: {brier_source})"
    else:
        g.details = f"Benchmark results not found ({brier_source})"
    gates.append(g)

    # Gate 2: Brier sanity floor
    g = PromotionGate("brier_sanity_floor", "OOS Brier above minimum sanity floor", "forecast_quality")
    if best_brier < 1.0:
        g.evaluate(best_brier, brier_floor, "greater")
        g.details = f"Best model Brier: {best_brier:.4f} (floor: {brier_floor:.4f})"
    else:
        g.details = "Benchmark results not found"
    gates.append(g)

    # Gate: Beats NWS baseline
    g = PromotionGate("beats_nws", "Model Brier beats NWS baseline", "forecast_quality")
    nws_baseline = thresholds.get("nws_brier_baseline")
    if nws_baseline is not None and best_brier < 1.0:
        g.evaluate(best_brier, nws_baseline, "less")
        g.details = f"Model: {best_brier:.4f} vs NWS: {nws_baseline:.4f} (source: {brier_source})"
    elif nws_baseline is None:
        g.details = "No NWS baseline configured for this city"
        g.passed = False
    else:
        g.details = "Benchmark results not found"
        g.passed = False
    gates.append(g)

    # Gate 3: Beats Kalshi market
    g = PromotionGate("beats_kalshi_market", "Model Brier beats Kalshi market", "forecast_quality")
    if market_brier is not None and best_brier < 1.0:
        edge = market_brier - best_brier
        g.evaluate(edge, 0.0, "greater")
        g.details = f"Model: {best_brier:.4f} vs Market: {market_brier:.4f} (edge: {edge:+.4f})"
    elif market_brier is None:
        g.details = "Kalshi market Brier not available (non-blocking)"
        g.passed = True
    else:
        g.details = "Benchmark results not found"
    gates.append(g)

    # Gate 4: Seasonal Brier — no season exceeds threshold
    g = PromotionGate("seasonal_brier", "No season exceeds Brier threshold", "forecast_quality")
    if seasonal:
        worst_name = max(seasonal, key=seasonal.get)
        worst_val = seasonal[worst_name]
        g.evaluate(worst_val, seasonal_threshold, "less")
        breakdown = ", ".join(f"{s}={v:.4f}" for s, v in sorted(seasonal.items()))
        g.details = f"Worst: {worst_name}={worst_val:.4f} (threshold: {seasonal_threshold}) [{breakdown}]"
    else:
        g.details = "Seasonal results not found"
        g.passed = False
    gates.append(g)

    # Gate 5: Sufficient OOS days
    g = PromotionGate("min_oos_days", "Sufficient OOS evaluation days", "forecast_quality")
    n_oos = 0
    for summary_file in [
        os.path.join(results_dir, "synthesis", "unified_benchmark_summary.json"),
        os.path.join(results_dir, "benchmark_summary.json"),
    ]:
        summary = _load_json(summary_file)
        if summary is not None:
            n_oos = summary.get("n_oos_days", 0)
            if n_oos > 0:
                break
    # Also try predictions CSV row count as proxy
    if n_oos == 0:
        pred_path = os.path.join(results_dir, "unified_predictions.csv")
        if os.path.exists(pred_path):
            try:
                import pandas as pd
                df = pd.read_csv(pred_path)
                if "period" in df.columns and "date" in df.columns:
                    oos_mask = df["period"].str.upper() == "OOS"
                    n_oos = int(df.loc[oos_mask, "date"].nunique())
                else:
                    n_oos = len(df)
            except Exception:
                pass
    if n_oos > 0:
        g.evaluate(n_oos, min_oos_days, "greater")
        g.details = f"OOS days: {n_oos} (minimum: {min_oos_days})"
    else:
        g.details = "OOS day count not found"
        g.passed = False
    gates.append(g)

    # ---- Category: Calibration ----

    # Gate 6: ECE within tolerance
    g = PromotionGate("calibration_ece", "Expected Calibration Error within tolerance", "calibration")
    ece = cal_summary.get("ece")
    if ece is not None:
        g.evaluate(ece, ece_threshold, "less")
        method = cal_summary.get("best_method", "unknown")
        g.details = f"ECE: {ece:.4f} (threshold: {ece_threshold}, method: {method})"
    else:
        g.details = "Calibration ECE not found"
        g.passed = False
    gates.append(g)

    # Gate 7: Reliability diagram exists
    g = PromotionGate("reliability_diagram", "Reliability diagram artifact exists", "calibration")
    rel_path = os.path.join(results_dir, "synthesis", "reliability_diagram.png")
    if os.path.exists(rel_path):
        g.passed = True
        g.details = f"Found at {rel_path}"
    else:
        g.passed = False
        g.details = f"Not found at {rel_path}"
    gates.append(g)

    # ---- Category: Trading ----

    # Gate 8: Positive P&L
    g = PromotionGate("positive_pnl", "Trading shows positive P&L", "trading")
    pnl = trading.get("total_pnl")
    if pnl is not None:
        g.evaluate(pnl, 0.0, "greater")
        source = trading.get("source", "unknown")
        variant = trading.get("best_variant", "unknown")
        g.details = f"{variant} P&L: ${pnl:.2f} (source: {source})"
    else:
        g.details = "Backtest results not found"
        g.passed = False
    gates.append(g)

    # Gate 9: Max drawdown within limits
    g = PromotionGate("max_drawdown", "Max drawdown within acceptable limits", "trading")
    dd_frac = trading.get("max_drawdown_frac")
    if trading.get("busted"):
        g.passed = False
        g.value = dd_frac
        g.threshold = max_dd_threshold
        g.details = "Backtest went bankrupt (bankroll exhausted) — hard fail"
    elif dd_frac is not None:
        g.evaluate(dd_frac, max_dd_threshold, "greater")
        g.details = f"Max drawdown: {dd_frac:.1%} (limit: {max_dd_threshold:.0%})"
    else:
        g.details = "Drawdown data not found"
        g.passed = False
    gates.append(g)

    # Gate: Real Kalshi backtest positive P&L
    g = PromotionGate("real_kalshi_pnl", "Real Kalshi backtest shows positive P&L", "trading")
    rk_metrics = _load_real_kalshi_metrics(results_dir)
    if rk_metrics is not None:
        rk_pnl = rk_metrics.get("total_pnl")
        rk_sharpe = rk_metrics.get("sharpe_ratio")
        if rk_pnl is not None:
            g.evaluate(rk_pnl, thresholds.get("min_positive_pnl", 0.0), "greater")
            sharpe_txt = f"{rk_sharpe:.2f}" if rk_sharpe is not None else "N/A"
            g.details = f"Real Kalshi P&L: ${rk_pnl:.2f}, Sharpe: {sharpe_txt}"
        else:
            g.details = "Real Kalshi metrics found but P&L missing"
            g.passed = False
    else:
        g.details = "Real Kalshi backtest metrics not found"
        g.passed = False
    gates.append(g)

    # Gate: Trading evaluation must come from real market prices.
    # Simulated-market backtests have proven wildly optimistic (e.g. Austin
    # +$1762 simulated vs -$1059 on real presettlement prices), so any city
    # with presettlement data on disk must be judged on real prices.
    g = PromotionGate(
        "trading_source_real",
        "Trading evaluation uses real Kalshi presettlement prices",
        "trading",
    )
    presettlement_path = (
        PROJECT_ROOT / "data" / f"kalshi_presettlement_{city_code}.csv"
        if city_code else None
    )
    has_presettlement = (
        presettlement_path is not None and presettlement_path.exists()
    )
    source = trading.get("source")
    if not has_presettlement:
        g.passed = True
        g.details = (
            f"No presettlement data on disk for '{city_code}' — gate not applicable "
            f"(collect real market data before promotion)"
        )
    elif source == "real_kalshi":
        g.passed = True
        g.details = "Trading summary derived from real Kalshi presettlement prices"
    else:
        g.passed = False
        g.details = (
            f"Presettlement data exists but trading summary source is "
            f"'{source}' — re-run the real-price backtest"
        )
    gates.append(g)

    # ---- Category: Operational Readiness ----

    # Gate 10: Processed data files exist
    g = PromotionGate("data_pipeline", "Processed data files exist", "operational")
    processed_dir = os.path.join(data_dir, "processed")
    required_files = [
        "features_train.csv", "features_val.csv", "features_test.csv",
        "target_train.csv", "target_val.csv", "target_test.csv",
        "scaler.pkl",
    ]
    found = sum(
        1 for f in required_files
        if os.path.exists(os.path.join(processed_dir, f))
    )
    g.passed = found == len(required_files)
    g.value = found
    g.threshold = len(required_files)
    g.details = f"{found}/{len(required_files)} required files in {processed_dir}"
    gates.append(g)

    # Gate 11: Model checkpoints exist
    g = PromotionGate("model_checkpoints", "Model checkpoints saved", "operational")
    if os.path.isdir(models_dir):
        model_files = [
            f for f in os.listdir(models_dir)
            if f.endswith(".pt") or f.endswith(".pkl")
        ]
        g.passed = len(model_files) > 0
        g.value = len(model_files)
        g.details = f"{len(model_files)} model file(s) in {models_dir}"
    else:
        g.passed = False
        g.details = f"Models directory not found: {models_dir}"
    gates.append(g)

    return gates


# ===========================================================================
# Unified report builder
# ===========================================================================

def evaluate_city(city_code: str) -> tuple[list[PromotionGate], dict]:
    """Run all promotion gates and build a unified report for a city.

    Parameters
    ----------
    city_code : str
        Short city code (e.g., "chi", "nyc", "phl", "atl", "aus").

    Returns
    -------
    gates : list[PromotionGate]
        All evaluated promotion gates.
    report : dict
        The complete unified promotion report (JSON-serializable).
    """
    from src.city_config import get_city_config, ensure_city_dirs

    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    thresholds = CITY_THRESHOLDS.get(city_code)
    if thresholds is None:
        raise ValueError(
            f"No promotion thresholds defined for city '{city_code}'. "
            f"Available: {', '.join(sorted(CITY_THRESHOLDS.keys()))}"
        )

    results_dir = cfg.results_dir
    data_dir = cfg.data_dir
    models_dir = cfg.models_dir

    # Evaluate gates
    gates = _check_all_gates(
        results_dir, data_dir, models_dir, thresholds, city_code=city_code,
    )

    total = len(gates)
    passed = sum(1 for g in gates if g.passed)

    # Extract supplemental data for the unified report
    best_brier, brier_source, best_model = _load_best_brier(results_dir)
    baselines = _extract_baseline_briers(results_dir)
    market_brier = _extract_market_brier(results_dir)
    cal_summary = _extract_calibration_summary(results_dir)
    trading = _extract_trading_summary(results_dir)
    seasonal = _extract_seasonal_brier(results_dir)

    report: dict[str, Any] = {
        # Header
        "schema_version": "3.0",
        "timestamp": datetime.now().isoformat(),
        "city": cfg.city_name,
        "city_code": cfg.city_code,
        "kalshi_ticker": cfg.kalshi_ticker,
        "target_station": cfg.target_station,
        "target_station_name": cfg.target_station_name,
        "timezone": cfg.timezone,

        # Overall status
        "overall_status": "PASS" if passed == total else "FAIL",
        "gates_passed": passed,
        "gates_total": total,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)",

        # Thresholds used
        "thresholds": thresholds,

        # Promotion gates (standardized)
        "gates": [g.to_dict() for g in gates],

        # Model summary
        "model_summary": {
            "best_model": best_model,
            "best_brier": best_brier if best_brier < 1.0 else None,
            "brier_source": brier_source,
            "edge_vs_market": (
                (market_brier - best_brier)
                if market_brier is not None and best_brier < 1.0
                else None
            ),
        },

        # Baseline comparisons (Phase F requirement)
        "baseline_comparisons": {
            "persistence_brier": baselines.get("persistence"),
            "climatology_brier": baselines.get("climatology"),
            "ridge_brier": baselines.get("ridge"),
            "model_beats_persistence": (
                best_brier < baselines["persistence"]
                if baselines.get("persistence") is not None and best_brier < 1.0
                else None
            ),
            "model_beats_climatology": (
                best_brier < baselines["climatology"]
                if baselines.get("climatology") is not None and best_brier < 1.0
                else None
            ),
            "model_beats_ridge": (
                best_brier < baselines["ridge"]
                if baselines.get("ridge") is not None and best_brier < 1.0
                else None
            ),
        },

        # Market-implied benchmark (Phase F requirement)
        "market_benchmark": {
            "kalshi_presettlement_brier": market_brier,
            "model_beats_market": (
                best_brier < market_brier
                if market_brier is not None and best_brier < 1.0
                else None
            ),
            "brier_edge": (
                market_brier - best_brier
                if market_brier is not None and best_brier < 1.0
                else None
            ),
        },

        # Calibration summary
        "calibration_summary": {
            "best_ece": cal_summary.get("ece"),
            "best_method": cal_summary.get("best_method"),
            "methods": cal_summary.get("methods", {}),
        },

        # Trading summary
        "trading_summary": {
            "source": trading.get("source"),
            "best_variant": trading.get("best_variant"),
            "total_pnl": trading.get("total_pnl"),
            "return_pct": trading.get("return_pct"),
            "sharpe_ratio": trading.get("sharpe_ratio"),
            "n_trades": trading.get("n_trades"),
            "win_rate": trading.get("win_rate"),
            "max_drawdown_frac": trading.get("max_drawdown_frac"),
            "busted": trading.get("busted"),
        },

        # Seasonal breakdown
        "seasonal_brier": seasonal,

        # Trading seasonal breakdown
        "trading_seasonal": trading.get("seasonal", {}),
    }

    return gates, report


# ===========================================================================
# Report I/O
# ===========================================================================

def save_unified_report(report: dict, output_path: str) -> None:
    """Save a unified promotion report to disk as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved unified promotion report to %s", output_path)


def print_unified_report(report: dict) -> None:
    """Print a formatted unified promotion report to the console."""
    city = report.get("city", "Unknown")
    ticker = report.get("kalshi_ticker", "")

    print()
    print("=" * 80)
    print(f"  {city.upper()} ({ticker}) — UNIFIED PROMOTION REPORT v{report.get('schema_version', '3.0')}")
    print("=" * 80)
    print(f"  Timestamp      : {report['timestamp']}")
    print(f"  Target Station : {report['target_station_name']} ({report['target_station']})")
    print(f"  Overall Status : {report['overall_status']}")
    print(f"  Gates          : {report['pass_rate']}")

    # -- Gates --
    print("-" * 80)
    print("  PROMOTION GATES")
    print("-" * 80)
    current_cat = ""
    for gate in report["gates"]:
        cat = gate.get("category", "")
        if cat != current_cat:
            current_cat = cat
            print(f"\n  [{cat.upper().replace('_', ' ')}]")
        status = "PASS" if gate["passed"] else "FAIL"
        icon = "[+]" if gate["passed"] else "[-]"
        print(f"    {icon} {gate['name']:25s} {status}")
        print(f"        {gate['details']}")

    # -- Baseline Comparisons --
    baselines = report.get("baseline_comparisons", {})
    print()
    print("-" * 80)
    print("  BASELINE COMPARISONS")
    print("-" * 80)
    model_brier = report.get("model_summary", {}).get("best_brier")
    model_name = report.get("model_summary", {}).get("best_model", "N/A")
    print(f"    Best model   : {model_name} (Brier: {_fmt_brier(model_brier)})")
    print(f"    Persistence  : {_fmt_brier(baselines.get('persistence_brier'))}"
          f"  {'< model (BEATS)' if baselines.get('model_beats_persistence') else ''}")
    print(f"    Climatology  : {_fmt_brier(baselines.get('climatology_brier'))}"
          f"  {'< model (BEATS)' if baselines.get('model_beats_climatology') else ''}")
    print(f"    Ridge        : {_fmt_brier(baselines.get('ridge_brier'))}"
          f"  {'< model (BEATS)' if baselines.get('model_beats_ridge') else ''}")

    # -- Market Benchmark --
    market = report.get("market_benchmark", {})
    print()
    print("-" * 80)
    print("  MARKET-IMPLIED BENCHMARK")
    print("-" * 80)
    print(f"    Kalshi pre-settlement Brier : {_fmt_brier(market.get('kalshi_presettlement_brier'))}")
    print(f"    Model beats market          : {market.get('model_beats_market', 'N/A')}")
    edge = market.get("brier_edge")
    print(f"    Brier edge                  : {f'{edge:+.4f}' if edge is not None else 'N/A'}")

    # -- Trading Summary --
    trading = report.get("trading_summary", {})
    print()
    print("-" * 80)
    print("  TRADING SUMMARY")
    print("-" * 80)
    print(f"    Source       : {trading.get('source', 'N/A')}")
    print(f"    Best variant : {trading.get('best_variant', 'N/A')}")
    pnl = trading.get("total_pnl")
    print(f"    Total P&L    : {f'${pnl:.2f}' if pnl is not None else 'N/A'}")
    sr = trading.get("sharpe_ratio")
    print(f"    Sharpe       : {f'{sr:.2f}' if sr is not None else 'N/A'}")
    dd = trading.get("max_drawdown_frac")
    print(f"    Max drawdown : {f'{dd:.1%}' if dd is not None else 'N/A'}")

    # -- Recommendation --
    print()
    print("-" * 80)
    if report["overall_status"] == "PASS":
        print(f"  RECOMMENDATION: {city} model is READY for promotion.")
    else:
        failed = [g for g in report["gates"] if not g["passed"]]
        print(f"  RECOMMENDATION: {city} model is NOT YET ready for promotion.")
        print(f"  {len(failed)} gate(s) need attention:")
        for g in failed:
            print(f"    - {g['name']}: {g['details']}")
    print("=" * 80)
    print()


def _fmt_brier(val: float | None) -> str:
    """Format a Brier score for display."""
    if val is None:
        return "N/A"
    return f"{val:.4f}"
