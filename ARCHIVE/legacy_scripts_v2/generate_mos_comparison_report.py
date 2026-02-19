#!/usr/bin/env python3
"""
Step 5: Generate Honest MOS Comparison Report.

Loads results from both the existing enhanced-proxy backtest and the new
MOS-proxy backtest, then generates a comprehensive head-to-head comparison
answering:
  - Does our NN model have genuine alpha vs. MOS?
  - How do the proxies compare on Brier scores?
  - Is the strategy profitable against MOS (a harder benchmark)?
  - What is the honest verdict?

Outputs:
    results/mos_comparison_report/  (plots + report)
    reports/mos_integration_report.md  (PM-ready summary)

Usage:
    python scripts/generate_mos_comparison_report.py
"""

import json
import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXISTING_BACKTEST_DIR = os.path.join(PROJECT_ROOT, "results", "kalshi_max_train_backtest")
MOS_BACKTEST_DIR = os.path.join(PROJECT_ROOT, "results", "mos_backtest")
MOS_VALIDATION_DIR = os.path.join(PROJECT_ROOT, "results", "mos_validation")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "mos_comparison_report")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
def load_json(path, label=""):
    """Load a JSON file, return empty dict if missing."""
    if not os.path.exists(path):
        logger.warning("%s not found: %s", label, path)
        return {}
    with open(path) as f:
        return json.load(f)


def load_results():
    """Load all results from prior scripts."""
    results = {}

    # Existing backtest results
    results["existing_brier"] = load_json(
        os.path.join(EXISTING_BACKTEST_DIR, "brier_analysis.json"),
        "Existing Brier",
    )
    results["existing_best_strategy"] = load_json(
        os.path.join(EXISTING_BACKTEST_DIR, "best_strategy_config.json"),
        "Existing best strategy",
    )
    existing_is_oos = os.path.join(EXISTING_BACKTEST_DIR, "is_vs_oos_comparison.csv")
    if os.path.exists(existing_is_oos):
        results["existing_is_oos"] = pd.read_csv(existing_is_oos)

    # MOS backtest results
    results["mos_brier"] = load_json(
        os.path.join(MOS_BACKTEST_DIR, "brier_analysis_with_mos.json"),
        "MOS Brier",
    )
    results["mos_best_strategy"] = load_json(
        os.path.join(MOS_BACKTEST_DIR, "best_mos_strategy_config.json"),
        "MOS best strategy",
    )
    results["enh_best_strategy"] = load_json(
        os.path.join(MOS_BACKTEST_DIR, "best_enh_strategy_config.json"),
        "Enhanced best strategy (MOS run)",
    )
    proxy_comp = os.path.join(MOS_BACKTEST_DIR, "proxy_comparison_summary.csv")
    if os.path.exists(proxy_comp):
        results["proxy_comparison"] = pd.read_csv(proxy_comp)

    # MOS validation results
    results["mos_validation"] = load_json(
        os.path.join(MOS_VALIDATION_DIR, "mos_validation_metrics.json"),
        "MOS validation",
    )

    # Strategy CSVs
    mos_strat = os.path.join(MOS_BACKTEST_DIR, "mos_strategies_is.csv")
    if os.path.exists(mos_strat):
        results["mos_strategies"] = pd.read_csv(mos_strat)
    enh_strat = os.path.join(MOS_BACKTEST_DIR, "enhanced_strategies_is.csv")
    if os.path.exists(enh_strat):
        results["enh_strategies"] = pd.read_csv(enh_strat)

    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def extract_brier_summary(brier_data, period="IS"):
    """Extract a clean Brier summary dict from the raw JSON."""
    period_data = brier_data.get(period, {})
    if not period_data:
        return {}

    summary = {}
    for key, val in period_data.items():
        if key in ("seasonal", "monthly"):
            continue
        if isinstance(val, dict) and "model_brier" in val:
            summary[key] = {
                "model_brier": val["model_brier"],
                "comp_brier": val["market_brier"],
                "delta": val["brier_delta"],
                "n": val.get("n_samples", 0),
            }
    return summary


def count_profitable(strategies_df):
    """Count strategies with trades and positive PnL."""
    if strategies_df is None or strategies_df.empty:
        return {"total": 0, "with_trades": 0, "profitable": 0, "pct_profitable": 0}
    trading = strategies_df[strategies_df["n_trades"] > 0]
    profitable = trading[trading["total_pnl"] > 0]
    return {
        "total": len(strategies_df),
        "with_trades": len(trading),
        "profitable": len(profitable),
        "pct_profitable": len(profitable) / len(trading) * 100 if len(trading) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Determine verdict
# ---------------------------------------------------------------------------
def determine_verdict(results):
    """Determine the honest verdict: alpha, no alpha, or model inferior."""
    verdict = {
        "scenario": "UNKNOWN",
        "explanation": "",
        "recommendation": "",
    }

    # Get MOS validation MAE
    mos_val = results.get("mos_validation", {})
    mos_overall = mos_val.get("overall", {})
    mos_ensemble_mae = None
    for key in ["mos_ensemble_tmax_f", "gfs_mos_tmax_f", "nam_mos_tmax_f"]:
        if key in mos_overall:
            mos_ensemble_mae = mos_overall[key].get("mae")
            break

    # Get MOS vs model Brier comparison
    mos_brier = results.get("mos_brier", {})
    mos_is_brier = mos_brier.get("IS", {})
    mos_oos_brier = mos_brier.get("OOS", {})

    # Get MOS proxy Brier
    mos_proxy_is = mos_is_brier.get("model_vs_mos_proxy", {})
    mos_proxy_oos = mos_oos_brier.get("model_vs_mos_proxy", {}) if mos_oos_brier else {}

    # Get OOS MOS strategy result
    proxy_comp = results.get("proxy_comparison")
    mos_oos_pnl = None
    enh_oos_pnl = None
    if proxy_comp is not None and not proxy_comp.empty:
        mos_oos_rows = proxy_comp[(proxy_comp["proxy"] == "MOS Proxy") & (proxy_comp["period"] == "OOS")]
        if len(mos_oos_rows) > 0:
            mos_oos_pnl = float(mos_oos_rows.iloc[0].get("total_pnl", 0))
        enh_oos_rows = proxy_comp[(proxy_comp["proxy"] == "Enhanced Proxy") & (proxy_comp["period"] == "OOS")]
        if len(enh_oos_rows) > 0:
            enh_oos_pnl = float(enh_oos_rows.iloc[0].get("total_pnl", 0))

    # Decision logic
    model_beats_mos_brier = False
    if mos_proxy_is and mos_proxy_is.get("brier_delta", 0) < 0:
        model_beats_mos_brier = True

    model_beats_mos_oos = False
    if mos_proxy_oos and mos_proxy_oos.get("brier_delta", 0) < 0:
        model_beats_mos_oos = True

    mos_oos_profitable = mos_oos_pnl is not None and mos_oos_pnl > 0

    if model_beats_mos_brier and model_beats_mos_oos and mos_oos_profitable:
        verdict["scenario"] = "A: GENUINE ALPHA"
        verdict["explanation"] = (
            f"Model beats MOS proxy on Brier score in both IS and OOS, and "
            f"OOS strategy against MOS proxy is profitable (PnL=${mos_oos_pnl:.0f})."
        )
        verdict["recommendation"] = "Proceed to paper trading with MOS as market proxy."
    elif model_beats_mos_brier and not model_beats_mos_oos:
        verdict["scenario"] = "B: OVERFITTING / NO RELIABLE ALPHA"
        verdict["explanation"] = (
            f"Model beats MOS in IS but NOT in OOS. Likely overfitting to IS period."
        )
        verdict["recommendation"] = (
            "Do NOT proceed to live trading. Investigate adding MOS as input feature instead."
        )
    elif not model_beats_mos_brier:
        verdict["scenario"] = "C: MOS SUPERIOR"
        verdict["explanation"] = (
            f"MOS proxy has lower Brier score than our model. Our enhanced proxy "
            f"was too weak a benchmark, inflating apparent edge."
        )
        if mos_ensemble_mae is not None:
            verdict["explanation"] += f" MOS MAE={mos_ensemble_mae:.2f}F vs NN ~4.3F."
        verdict["recommendation"] = (
            "Rethink strategy. Consider using MOS forecasts as input features. "
            "The model's apparent edge against the enhanced proxy was a mirage."
        )
    else:
        verdict["scenario"] = "B: MARGINAL / INCONCLUSIVE"
        verdict["explanation"] = (
            "Results are mixed. Model may have slight edge but it's not robust."
        )
        verdict["recommendation"] = (
            "Extended OOS testing recommended before paper trading."
        )

    return verdict


# ---------------------------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------------------------
def generate_comparison_plots(results, output_dir):
    """Generate comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # ---- 1. Brier Score Comparison Bar Chart (All Proxies + Kalshi) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mos_brier = results.get("mos_brier", {})

    for ax, period, label in [(axes[0], "IS", "IS (2023-2024)"), (axes[1], "OOS", "OOS (2025)")]:
        period_data = mos_brier.get(period, {})
        if not period_data:
            ax.text(0.5, 0.5, f"No {period} data", ha="center", va="center", transform=ax.transAxes)
            continue

        comparisons = []
        model_briers = []
        comp_briers = []
        colors = []

        color_map = {
            "mos_proxy": "#2ca02c",
            "enhanced_proxy": "#d62728",
            "naive_proxy": "#ff7f0e",
            "kalshi_market": "#9467bd",
        }
        nice_names = {
            "mos_proxy": "MOS Proxy",
            "enhanced_proxy": "Enhanced Proxy",
            "naive_proxy": "Naive Proxy",
            "kalshi_market": "Kalshi Market",
        }

        for key in ["model_vs_mos_proxy", "model_vs_enhanced_proxy", "model_vs_naive_proxy", "model_vs_kalshi_market"]:
            bdata = period_data.get(key, {})
            if bdata and "model_brier" in bdata:
                short_key = key.replace("model_vs_", "")
                comparisons.append(nice_names.get(short_key, short_key))
                model_briers.append(bdata["model_brier"])
                comp_briers.append(bdata["market_brier"])
                colors.append(color_map.get(short_key, "#333333"))

        if comparisons:
            x = np.arange(len(comparisons))
            width = 0.35
            ax.bar(x - width / 2, model_briers, width, label="NN Model", color="#4c72b0", alpha=0.8)
            bars = ax.bar(x + width / 2, comp_briers, width, label="Comparison", alpha=0.8)
            for bar, col in zip(bars, colors):
                bar.set_color(col)

            ax.set_ylabel("Brier Score (lower = better)")
            ax.set_title(f"Brier Scores: {label}")
            ax.set_xticks(x)
            ax.set_xticklabels(comparisons, rotation=15, ha="right", fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "all_brier_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. MOS vs NN Forecast Error by Season ----
    mos_val = results.get("mos_validation", {})
    seasonal = mos_val.get("seasonal", {})
    if seasonal:
        fig, ax = plt.subplots(figsize=(10, 5))
        seasons = SEASON_ORDER
        x = np.arange(len(seasons))
        width = 0.2

        # Find available MOS columns
        sample_season = next(iter(seasonal.values()), {})
        mos_keys = [k for k in sample_season.keys() if k.endswith("_mae")]

        for i, mos_key in enumerate(mos_keys[:3]):
            mae_vals = [seasonal.get(s, {}).get(mos_key, np.nan) for s in seasons]
            nice = mos_key.replace("_tmax_f_mae", "").replace("_", " ").upper()
            ax.bar(x + i * width, mae_vals, width, label=nice, alpha=0.8)

        ax.axhline(4.3, color="#d62728", linestyle="--", linewidth=1.5, label="NN ~4.3F")
        ax.set_xticks(x + width)
        ax.set_xticklabels(seasons)
        ax.set_ylabel("MAE (F)")
        ax.set_title("Seasonal Point Forecast MAE: MOS vs NN")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = os.path.join(output_dir, "seasonal_mae_mos_vs_nn.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 3. Strategy Profitability Comparison ----
    mos_strat = results.get("mos_strategies")
    enh_strat = results.get("enh_strategies")
    if mos_strat is not None and enh_strat is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, sdf, label in [(axes[0], mos_strat, "vs MOS Proxy"), (axes[1], enh_strat, "vs Enhanced Proxy")]:
            trading = sdf[sdf["n_trades"] > 0]
            if len(trading) > 0:
                ax.hist(trading["total_pnl"], bins=50, alpha=0.7, color="#4c72b0", edgecolor="black", linewidth=0.3)
                ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
                profitable_pct = (trading["total_pnl"] > 0).mean() * 100
                ax.set_title(f"IS Strategy PnL Distribution ({label})\n{profitable_pct:.0f}% profitable")
                ax.set_xlabel("Total PnL ($)")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, "strategy_pnl_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 4. Proxy Brier scores side-by-side (only Brier of the proxies themselves) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    proxy_keys = ["model_vs_mos_proxy", "model_vs_enhanced_proxy", "model_vs_naive_proxy", "model_vs_kalshi_market"]
    nice_names_map = {
        "model_vs_mos_proxy": "MOS Proxy",
        "model_vs_enhanced_proxy": "Enhanced Proxy",
        "model_vs_naive_proxy": "Naive Proxy",
        "model_vs_kalshi_market": "Kalshi Market",
    }

    # Also add model Brier as first entry
    model_is = mos_brier.get("IS", {}).get("model_vs_mos_proxy", {}).get("model_brier")
    model_oos = mos_brier.get("OOS", {}).get("model_vs_mos_proxy", {}).get("model_brier")

    all_names = ["NN Model"]
    all_is = [model_is if model_is else np.nan]
    all_oos = [model_oos if model_oos else np.nan]

    is_data = mos_brier.get("IS", {})
    oos_data = mos_brier.get("OOS", {})

    for key in proxy_keys:
        bdata_is = is_data.get(key, {})
        bdata_oos = oos_data.get(key, {})
        if bdata_is and "market_brier" in bdata_is:
            all_names.append(nice_names_map.get(key, key))
            all_is.append(bdata_is["market_brier"])
            all_oos.append(bdata_oos.get("market_brier", np.nan) if bdata_oos else np.nan)

    if len(all_names) > 1:
        x = np.arange(len(all_names))
        width = 0.35
        ax.bar(x - width / 2, all_is, width, label="IS", color="#4c72b0", alpha=0.8)
        if any(not np.isnan(v) for v in all_oos):
            ax.bar(x + width / 2, all_oos, width, label="OOS", color="#d62728", alpha=0.8)
        ax.set_ylabel("Brier Score (lower = better)")
        ax.set_title("Brier Scores: All Forecasters")
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=15, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "all_forecaster_brier.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    logger.info("Generated %d comparison plots in %s", len(saved), output_dir)
    return saved


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
def generate_markdown_report(results, verdict, output_dir, report_dir):
    """Generate the PM-ready markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    mos_val = results.get("mos_validation", {})
    mos_brier = results.get("mos_brier", {})
    existing_brier = results.get("existing_brier", {})

    lines = []
    lines.append("# MOS Integration Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Verdict: {verdict['scenario']}**")
    lines.append("")
    lines.append(verdict["explanation"])
    lines.append("")
    lines.append(f"**Recommendation:** {verdict['recommendation']}")
    lines.append("")

    # ---- MOS Forecast Accuracy ----
    lines.append("## 1. MOS Forecast Accuracy vs Our Model")
    lines.append("")
    mos_overall = mos_val.get("overall", {})
    if mos_overall:
        lines.append("| Source | MAE (F) | RMSE (F) | R2 | Bias (F) | n |")
        lines.append("|--------|---------|----------|-----|----------|---|")
        for col, m in mos_overall.items():
            nice = col.replace("_tmax_f", "").replace("_", " ").upper()
            lines.append(f"| {nice} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} | {m['bias']:.2f} | {m['n']} |")
        lines.append(f"| NN Model (benchmark) | ~4.3 | ~5.7 | ~0.87 | - | - |")
        lines.append(f"| Ridge Model (benchmark) | ~4.3 | ~5.7 | ~0.88 | - | - |")
    else:
        lines.append("*MOS validation data not available.*")
    lines.append("")

    # Seasonal
    seasonal = mos_val.get("seasonal", {})
    if seasonal:
        lines.append("### Seasonal MAE Breakdown")
        lines.append("")
        sample = next(iter(seasonal.values()), {})
        mae_keys = [k for k in sample if k.endswith("_mae")]
        header_parts = ["Season", "n"] + [k.replace("_tmax_f_mae", "").upper() for k in mae_keys]
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")
        for s in SEASON_ORDER:
            sdata = seasonal.get(s, {})
            if sdata:
                parts = [s, str(sdata.get("n", 0))]
                for k in mae_keys:
                    parts.append(f"{sdata.get(k, 0):.2f}")
                lines.append("| " + " | ".join(parts) + " |")
    lines.append("")

    # ---- Brier Score Comparison ----
    lines.append("## 2. Brier Score Comparison (THE Critical Test)")
    lines.append("")

    for period in ["IS", "OOS"]:
        period_data = mos_brier.get(period, {})
        if not period_data:
            continue
        period_label = "IS (2023-2024)" if period == "IS" else "OOS (2025)"
        lines.append(f"### {period_label}")
        lines.append("")
        lines.append("| Comparison | Model Brier | Comp Brier | Delta | Winner |")
        lines.append("|-----------|-------------|------------|-------|--------|")
        for key, val in period_data.items():
            if key in ("seasonal", "monthly") or not isinstance(val, dict) or "model_brier" not in val:
                continue
            delta = val.get("brier_delta", 0)
            winner = "Model" if delta < 0 else "Comparison"
            nice = key.replace("model_vs_", "").replace("_", " ").title()
            lines.append(
                f"| vs {nice} | {val['model_brier']:.4f} | {val['market_brier']:.4f} | "
                f"{delta:+.4f} | {winner} |"
            )
        lines.append("")

        # Seasonal Brier
        seas = period_data.get("seasonal", {})
        if seas:
            lines.append(f"**Seasonal Brier ({period_label}):**")
            lines.append("")
            # Build header
            sample = next(iter(seas.values()), {})
            brier_keys = [k for k in sample if k.endswith("_brier")]
            header = ["Season", "n"] + brier_keys
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join(["---"] * len(header)) + " |")
            for s in SEASON_ORDER:
                sdata = seas.get(s, {})
                if sdata:
                    parts = [s, str(sdata.get("n", 0))]
                    for k in brier_keys:
                        parts.append(f"{sdata.get(k, 0):.4f}")
                    lines.append("| " + " | ".join(parts) + " |")
            lines.append("")

    # ---- Strategy Profitability ----
    lines.append("## 3. Strategy Profitability Against MOS Proxy")
    lines.append("")

    mos_counts = count_profitable(results.get("mos_strategies"))
    enh_counts = count_profitable(results.get("enh_strategies"))

    lines.append("| Metric | vs MOS Proxy | vs Enhanced Proxy |")
    lines.append("|--------|-------------|-------------------|")
    lines.append(f"| Total strategies | {mos_counts['total']} | {enh_counts['total']} |")
    lines.append(f"| With trades | {mos_counts['with_trades']} | {enh_counts['with_trades']} |")
    lines.append(f"| Profitable | {mos_counts['profitable']} | {enh_counts['profitable']} |")
    lines.append(f"| % Profitable | {mos_counts['pct_profitable']:.1f}% | {enh_counts['pct_profitable']:.1f}% |")
    lines.append("")

    # Best strategy comparison
    mos_best = results.get("mos_best_strategy", {})
    enh_best = results.get("enh_best_strategy", {})
    if mos_best and "error" not in mos_best:
        lines.append("**Best MOS Strategy (IS):**")
        lines.append(f"- PnL: ${mos_best.get('total_pnl', 0):.0f}")
        lines.append(f"- Sharpe: {mos_best.get('sharpe_ratio', 0):.2f}")
        lines.append(f"- Win Rate: {mos_best.get('win_rate', 0) * 100:.1f}%")
        lines.append(f"- Trades: {mos_best.get('n_trades', 0)}")
    lines.append("")

    # IS vs OOS
    proxy_comp = results.get("proxy_comparison")
    if proxy_comp is not None and not proxy_comp.empty:
        lines.append("### IS vs OOS Performance")
        lines.append("")
        lines.append("| Proxy | Period | PnL | Sharpe | Win Rate | Trades |")
        lines.append("|-------|--------|-----|--------|----------|--------|")
        for _, row in proxy_comp.iterrows():
            lines.append(
                f"| {row.get('proxy', 'N/A')} | {row.get('period', 'N/A')} | "
                f"${row.get('total_pnl', 0):.0f} | {row.get('sharpe_ratio', 0):.2f} | "
                f"{row.get('win_rate', 0) * 100:.1f}% | {row.get('n_trades', 0)} |"
            )
    lines.append("")

    # ---- Verdict ----
    lines.append("## 4. Verdict and Recommendation")
    lines.append("")
    lines.append(f"### Scenario: {verdict['scenario']}")
    lines.append("")
    lines.append(verdict["explanation"])
    lines.append("")
    lines.append(f"**Recommendation:** {verdict['recommendation']}")
    lines.append("")

    # Next steps
    lines.append("### Next Steps")
    lines.append("")
    if "ALPHA" in verdict["scenario"]:
        lines.append("1. Paper trade with MOS proxy for 30+ days")
        lines.append("2. Monitor Brier score degradation in real-time")
        lines.append("3. If still profitable, consider small live allocation")
    elif "SUPERIOR" in verdict["scenario"]:
        lines.append("1. Add MOS forecasts as input features to the NN")
        lines.append("2. Retrain with MOS as additional signal")
        lines.append("3. Re-evaluate edge after MOS integration")
    else:
        lines.append("1. Extended OOS testing (accumulate more 2025 data)")
        lines.append("2. Consider hybrid approach: MOS + NN ensemble")
        lines.append("3. Investigate seasonal pockets of edge")
    lines.append("")

    report_text = "\n".join(lines)

    # Save to both locations
    for path in [
        os.path.join(output_dir, "mos_comparison_report.md"),
        os.path.join(report_dir, "mos_integration_report.md"),
    ]:
        with open(path, "w") as f:
            f.write(report_text)
        logger.info("Saved report to %s", path)

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  STEP 5: GENERATE HONEST MOS COMPARISON REPORT")
    print("=" * 78)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if required results exist
    required_dirs = [MOS_BACKTEST_DIR, MOS_VALIDATION_DIR]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    if missing:
        print("  WARNING: Some result directories are missing:")
        for d in missing:
            print(f"    - {d}")
        print("  Will proceed with whatever data is available.\n")

    # Load results
    print("Loading results from prior scripts...")
    results = load_results()

    # Summarize what we found
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key}: {len(val)} items")
        elif isinstance(val, pd.DataFrame):
            print(f"  {key}: {len(val)} rows")
        else:
            print(f"  {key}: loaded")
    print()

    # Determine verdict
    print("Analyzing results and determining verdict...")
    verdict = determine_verdict(results)
    print(f"\n  VERDICT: {verdict['scenario']}")
    print(f"  {verdict['explanation']}")
    print(f"  RECOMMENDATION: {verdict['recommendation']}")
    print()

    # Generate plots
    print("Generating comparison plots...")
    saved_plots = generate_comparison_plots(results, OUTPUT_DIR)
    print(f"  Generated {len(saved_plots)} plots")
    print()

    # Generate report
    print("Generating comprehensive report...")
    report_text = generate_markdown_report(results, verdict, OUTPUT_DIR, REPORT_DIR)

    # Print report
    print()
    print("=" * 78)
    print("  REPORT PREVIEW")
    print("=" * 78)
    print()
    # Print just the first ~80 lines
    report_lines = report_text.split("\n")
    for line in report_lines[:80]:
        print(line)
    if len(report_lines) > 80:
        print(f"\n  ... ({len(report_lines) - 80} more lines, see full report)")
    print()

    # List output files
    print("Output files:")
    for dirpath in [OUTPUT_DIR, REPORT_DIR]:
        for fname in sorted(os.listdir(dirpath)):
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath) and "mos" in fname.lower():
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {fpath} ({size_kb:.1f} KB)")
    print()

    print("=" * 78)
    print("  MOS COMPARISON REPORT COMPLETE!")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Report:  {os.path.join(REPORT_DIR, 'mos_integration_report.md')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
