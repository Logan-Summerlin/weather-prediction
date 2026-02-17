#!/usr/bin/env python3
"""
Cross-City Benchmark Comparison: NYC vs Chicago vs Philadelphia.

Compiles contract-level Brier scores across all three cities and produces:
  1. A unified CSV comparison table
  2. A JSON summary with best models per city
  3. A bar-chart visualization
  4. Trading strategy comparison (if backtest results available)

All metrics use contract Brier (not day-bucket Brier).

Usage:
    python scripts/run_cross_city_benchmark_comparison.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "cross_city_comparison"

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_nyc_results() -> dict:
    """Load NYC unified benchmark results."""
    results = {}

    # Unified outperformance benchmark
    unified_csv = RESULTS_ROOT / "prediction_market_benchmark" / "unified_outperformance" / "benchmark_summary.csv"
    if unified_csv.exists():
        df = pd.read_csv(unified_csv)
        for _, row in df.iterrows():
            variant = row["variant"]
            results[variant] = {
                "overall_brier": row.get("overall_brier", float("nan")),
                "oos_brier": row.get("oos_brier", float("nan")),
                "oos_ece": row.get("oos_ece", float("nan")),
                "brier_winter": row.get("brier_winter", float("nan")),
                "brier_spring": row.get("brier_spring", float("nan")),
                "brier_summer": row.get("brier_summer", float("nan")),
                "brier_fall": row.get("brier_fall", float("nan")),
            }
        logger.info("Loaded NYC unified benchmark: %d variants", len(results))

    return results


def load_city_results(city: str) -> dict:
    """Load CHI or PHL unified benchmark results."""
    results = {}

    # Unified benchmark JSON
    if city == "chi":
        json_path = RESULTS_ROOT / "chicago" / "unified_benchmark_results.json"
        real_csv = RESULTS_ROOT / "chicago" / "chi_real_data_benchmark_summary.csv"
    else:
        json_path = RESULTS_ROOT / "philadelphia" / "unified_benchmark_results.json"
        real_csv = RESULTS_ROOT / "philadelphia" / "phl_real_data_benchmark_summary.csv"

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for model_name, metrics in data.items():
            # Only include models with explicit contract_brier (U-series + Kalshi)
            # E-series base models have test_brier which is per-sample, not contract-level
            if "contract_brier" in metrics:
                results[model_name] = {
                    "contract_brier": metrics["contract_brier"],
                    "is_brier": metrics.get("is_brier", float("nan")),
                    "oos_brier": metrics.get("oos_brier", float("nan")),
                }
            else:
                # Skip E-series base models (test_brier != contract_brier)
                pass
        logger.info("Loaded %s unified benchmark: %d models", city.upper(), len(results))

    # Real data benchmark with seasonal breakdown
    if real_csv.exists():
        df = pd.read_csv(real_csv)
        for _, row in df.iterrows():
            model_name = row["model"]
            if model_name in results:
                results[model_name]["brier_DJF"] = row.get("brier_DJF", float("nan"))
                results[model_name]["brier_MAM"] = row.get("brier_MAM", float("nan"))
                results[model_name]["brier_JJA"] = row.get("brier_JJA", float("nan"))
                results[model_name]["brier_SON"] = row.get("brier_SON", float("nan"))

    return results


def load_backtest_results(city: str) -> dict:
    """Load backtest metrics for a city."""
    if city == "nyc":
        # NYC trading simulation
        trading_csv = RESULTS_ROOT / "prediction_market_benchmark" / "unified_outperformance" / "trading_simulation_results.csv"
        if trading_csv.exists():
            df = pd.read_csv(trading_csv)
            return {"trading_data": df.to_dict(orient="records")}
        return {}
    elif city == "chi":
        metrics_path = RESULTS_ROOT / "chicago" / "backtest" / "backtest_metrics.json"
    else:
        metrics_path = RESULTS_ROOT / "philadelphia" / "backtest" / "backtest_metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_calibration_results(city: str) -> dict:
    """Load synthesis calibration sweep results."""
    if city == "chi":
        path = RESULTS_ROOT / "chicago" / "synthesis" / "calibration_sweep_summary.json"
    elif city == "phl":
        path = RESULTS_ROOT / "philadelphia" / "synthesis" / "calibration_sweep_summary.json"
    else:
        return {}

    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ===========================================================================
# Comparison Tables
# ===========================================================================

def build_comparison_table() -> pd.DataFrame:
    """Build a unified comparison table across all three cities."""
    rows = []

    # --- NYC ---
    nyc = load_nyc_results()
    for variant, metrics in nyc.items():
        rows.append({
            "city": "NYC",
            "ticker": "KXHIGHNY",
            "model": variant,
            "contract_brier": metrics.get("overall_brier", float("nan")),
            "oos_brier": metrics.get("oos_brier", float("nan")),
            "oos_ece": metrics.get("oos_ece", float("nan")),
            "brier_DJF": metrics.get("brier_winter", float("nan")),
            "brier_MAM": metrics.get("brier_spring", float("nan")),
            "brier_JJA": metrics.get("brier_summer", float("nan")),
            "brier_SON": metrics.get("brier_fall", float("nan")),
        })

    # --- CHI ---
    chi = load_city_results("chi")
    for model, metrics in chi.items():
        rows.append({
            "city": "CHI",
            "ticker": "KXHIGHCHI",
            "model": model,
            "contract_brier": metrics.get("contract_brier", float("nan")),
            "oos_brier": metrics.get("oos_brier", float("nan")),
            "oos_ece": float("nan"),
            "brier_DJF": metrics.get("brier_DJF", float("nan")),
            "brier_MAM": metrics.get("brier_MAM", float("nan")),
            "brier_JJA": metrics.get("brier_JJA", float("nan")),
            "brier_SON": metrics.get("brier_SON", float("nan")),
        })

    # --- PHL ---
    phl = load_city_results("phl")
    for model, metrics in phl.items():
        rows.append({
            "city": "PHL",
            "ticker": "KXHIGHPHL",
            "model": model,
            "contract_brier": metrics.get("contract_brier", float("nan")),
            "oos_brier": metrics.get("oos_brier", float("nan")),
            "oos_ece": float("nan"),
            "brier_DJF": metrics.get("brier_DJF", float("nan")),
            "brier_MAM": metrics.get("brier_MAM", float("nan")),
            "brier_JJA": metrics.get("brier_JJA", float("nan")),
            "brier_SON": metrics.get("brier_SON", float("nan")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["city", "contract_brier"]).reset_index(drop=True)
    return df


def build_best_models_summary() -> dict:
    """Identify best model per city and build summary."""
    summary = {}

    # NYC — exclude Kalshi_Settled (hindsight), NWS (baseline reference)
    nyc = load_nyc_results()
    if nyc:
        nyc_models = {k: v for k, v in nyc.items()
                      if k not in ("Kalshi_Settled", "Kalshi_PreSettlement", "NWS")}
        best_nyc = min(nyc_models.items(), key=lambda x: x[1].get("overall_brier", float("inf")))
        kalshi_nyc = nyc.get("Kalshi_PreSettlement", {})
        summary["NYC"] = {
            "ticker": "KXHIGHNY",
            "best_model": best_nyc[0],
            "best_contract_brier": best_nyc[1].get("overall_brier", float("nan")),
            "best_oos_brier": best_nyc[1].get("oos_brier", float("nan")),
            "kalshi_presettlement_brier": kalshi_nyc.get("overall_brier", float("nan")),
            "edge_vs_market": kalshi_nyc.get("overall_brier", 0) - best_nyc[1].get("overall_brier", 0),
        }

    # CHI
    chi = load_city_results("chi")
    if chi:
        # Filter to U-series (contract_brier key)
        chi_models = {k: v for k, v in chi.items()
                      if "contract_brier" in v and not np.isnan(v["contract_brier"])
                      and k != "Kalshi_PreSettlement"}
        if chi_models:
            best_chi = min(chi_models.items(), key=lambda x: x[1]["contract_brier"])
            kalshi_chi = chi.get("Kalshi_PreSettlement", {})
            summary["CHI"] = {
                "ticker": "KXHIGHCHI",
                "best_model": best_chi[0],
                "best_contract_brier": best_chi[1]["contract_brier"],
                "best_oos_brier": best_chi[1].get("oos_brier", float("nan")),
                "kalshi_presettlement_brier": kalshi_chi.get("contract_brier", float("nan")),
                "edge_vs_market": kalshi_chi.get("contract_brier", 0) - best_chi[1]["contract_brier"],
            }

    # PHL
    phl = load_city_results("phl")
    if phl:
        phl_models = {k: v for k, v in phl.items()
                      if "contract_brier" in v and not np.isnan(v["contract_brier"])
                      and k != "Kalshi_PreSettlement"}
        if phl_models:
            best_phl = min(phl_models.items(), key=lambda x: x[1]["contract_brier"])
            kalshi_phl = phl.get("Kalshi_PreSettlement", {})
            summary["PHL"] = {
                "ticker": "KXHIGHPHL",
                "best_model": best_phl[0],
                "best_contract_brier": best_phl[1]["contract_brier"],
                "best_oos_brier": best_phl[1].get("oos_brier", float("nan")),
                "kalshi_presettlement_brier": kalshi_phl.get("contract_brier", float("nan")),
                "edge_vs_market": kalshi_phl.get("contract_brier", 0) - best_phl[1]["contract_brier"],
            }

    return summary


def build_trading_comparison() -> dict:
    """Compare backtest trading results across cities."""
    comparison = {}

    for city in ["chi", "phl"]:
        bt = load_backtest_results(city)
        if bt:
            comparison[city.upper()] = {
                "total_pnl": bt.get("total_pnl", 0),
                "return_pct": bt.get("return_pct", 0),
                "n_trades": bt.get("n_trades", 0),
                "win_rate": bt.get("win_rate", 0),
                "sharpe_ratio": bt.get("sharpe_ratio", 0),
                "max_drawdown": bt.get("max_drawdown", 0),
                "max_drawdown_pct": bt.get("max_drawdown_pct", 0),
                "model_brier": bt.get("model_brier", float("nan")),
                "market_brier": bt.get("market_brier", float("nan")),
                "brier_edge": bt.get("brier_edge", float("nan")),
                "seasonal": bt.get("seasonal", {}),
            }

    return comparison


# ===========================================================================
# Visualization
# ===========================================================================

def plot_cross_city_brier(summary: dict, output_path: str) -> None:
    """Plot cross-city best model Brier comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    cities = list(summary.keys())
    best_briers = [summary[c]["best_contract_brier"] for c in cities]
    market_briers = [summary[c].get("kalshi_presettlement_brier", 0) for c in cities]
    model_names = [summary[c]["best_model"] for c in cities]

    x = np.arange(len(cities))
    width = 0.35

    bars1 = ax.bar(x - width / 2, best_briers, width, label="Best Model",
                   color=["#1f77b4", "#2ca02c", "#ff7f0e"], edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, market_briers, width, label="Kalshi PreSettlement",
                   color=["#aec7e8", "#98df8a", "#ffbb78"], edgecolor="black", alpha=0.85)

    # Labels
    for bar, brier, name in zip(bars1, best_briers, model_names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{brier:.4f}\n({name})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, brier in zip(bars2, market_briers):
        if not np.isnan(brier):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{brier:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({summary[c]['ticker']})" for c in cities], fontsize=11)
    ax.set_ylabel("Contract Brier Score (lower is better)", fontsize=12)
    ax.set_title("Cross-City Benchmark: Best Model vs Kalshi Market\n(Contract Brier Scores)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Edge annotations
    for i, c in enumerate(cities):
        edge = summary[c].get("edge_vs_market", 0)
        if edge > 0:
            ax.annotate(f"Edge: +{edge:.4f}", xy=(i, min(best_briers[i], market_briers[i]) * 0.95),
                       fontsize=9, color="#2ca02c", fontweight="bold", ha="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross-city Brier comparison to %s", output_path)


def plot_trading_comparison(trading: dict, output_path: str) -> None:
    """Plot cross-city trading results comparison."""
    if not trading:
        return

    cities = list(trading.keys())
    n_cities = len(cities)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Total P&L
    ax = axes[0, 0]
    pnls = [trading[c]["total_pnl"] for c in cities]
    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pnls]
    bars = ax.bar(cities, pnls, color=colors, edgecolor="black", alpha=0.8)
    for bar, pnl in zip(bars, pnls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"${pnl:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Total P&L ($)")
    ax.set_title("Total P&L")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Win Rate
    ax = axes[0, 1]
    win_rates = [trading[c]["win_rate"] * 100 for c in cities]
    ax.bar(cities, win_rates, color="#1f77b4", edgecolor="black", alpha=0.8)
    for i, wr in enumerate(win_rates):
        ax.text(i, wr + 0.5, f"{wr:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Sharpe Ratio
    ax = axes[1, 0]
    sharpes = [trading[c]["sharpe_ratio"] for c in cities]
    colors_s = ["#2ca02c" if s >= 0 else "#d62728" for s in sharpes]
    ax.bar(cities, sharpes, color=colors_s, edgecolor="black", alpha=0.8)
    for i, s in enumerate(sharpes):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.set_title("Sharpe Ratio")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Max Drawdown
    ax = axes[1, 1]
    drawdowns = [abs(trading[c]["max_drawdown"]) for c in cities]
    ax.bar(cities, drawdowns, color="#d62728", edgecolor="black", alpha=0.7)
    for i, (dd, city_name) in enumerate(zip(drawdowns, cities)):
        pct = abs(trading[city_name]["max_drawdown_pct"])
        ax.text(i, dd + 0.5, f"${dd:.1f}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Max Drawdown ($)")
    ax.set_title("Max Drawdown (absolute)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Cross-City Trading Strategy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trading comparison to %s", output_path)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the full cross-city benchmark comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Cross-City Benchmark Comparison: NYC vs CHI vs PHL")
    logger.info("  Metric: Contract Brier Score")
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("=" * 70)

    # ---- Step 1: Build comparison table ----
    logger.info("Step 1: Building unified comparison table ...")
    comparison_df = build_comparison_table()
    csv_path = OUTPUT_DIR / "cross_city_benchmark_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info("  Saved comparison CSV: %s (%d rows)", csv_path, len(comparison_df))

    # ---- Step 2: Best models summary ----
    logger.info("Step 2: Building best models summary ...")
    best_summary = build_best_models_summary()
    summary_path = OUTPUT_DIR / "best_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(best_summary, f, indent=2, default=str)
    logger.info("  Saved summary JSON: %s", summary_path)

    # Print summary
    for city, info in best_summary.items():
        logger.info("  %s (%s): Best=%s, Brier=%.4f, Market=%.4f, Edge=%+.4f",
                    city, info["ticker"], info["best_model"],
                    info["best_contract_brier"],
                    info.get("kalshi_presettlement_brier", float("nan")),
                    info.get("edge_vs_market", 0))

    # ---- Step 3: Cross-city visualization ----
    logger.info("Step 3: Generating cross-city Brier comparison chart ...")
    plot_cross_city_brier(
        best_summary,
        str(OUTPUT_DIR / "cross_city_brier_comparison.png"),
    )

    # ---- Step 4: Trading comparison ----
    logger.info("Step 4: Building trading comparison ...")
    trading = build_trading_comparison()
    if trading:
        trading_path = OUTPUT_DIR / "trading_comparison.json"
        with open(trading_path, "w") as f:
            json.dump(trading, f, indent=2, default=str)
        logger.info("  Saved trading comparison: %s", trading_path)

        plot_trading_comparison(
            trading,
            str(OUTPUT_DIR / "trading_comparison.png"),
        )

        for city, info in trading.items():
            logger.info("  %s Trading: P&L=$%.1f (%.1f%%), Win=%.1f%%, Sharpe=%.2f, DD=$%.1f",
                        city, info["total_pnl"], info["return_pct"],
                        info["win_rate"] * 100, info["sharpe_ratio"],
                        abs(info["max_drawdown"]))
    else:
        logger.info("  No backtest results available yet.")

    # ---- Step 5: Calibration comparison ----
    logger.info("Step 5: Calibration comparison ...")
    cal_summary = {}
    for city in ["chi", "phl"]:
        cal = load_calibration_results(city)
        if cal:
            cal_summary[city.upper()] = cal
            logger.info("  %s best calibration: %s (Brier=%.4f, ECE=%.4f)",
                        city.upper(), cal["best_method"],
                        cal["best_brier"],
                        cal["methods"][cal["best_method"]]["ece"])

    if cal_summary:
        cal_path = OUTPUT_DIR / "calibration_comparison.json"
        with open(cal_path, "w") as f:
            json.dump(cal_summary, f, indent=2, default=str)

    # ---- Final summary ----
    logger.info("=" * 70)
    logger.info("Cross-City Comparison Complete")
    logger.info("  Output directory: %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
