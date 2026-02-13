"""
Run Kalshi Real-Data Backtesting -- Part 1: In-Sample Strategy Discovery (2023-2024).

Uses REAL data from:
  1. Real Kalshi KXHIGHNY market data (fetched from Kalshi API)
  2. Real NYC Central Park temperature observations (from GHCN)
  3. Real model predictions (from trained neural network)

Steps:
  1. Load real Kalshi market data (2023-2024)
  2. Load real model predictions (Gaussian mu, sigma)
  3. Align model predictions with market data
  4. Run comprehensive strategy grid search (exact plan parameters)
  5. Full analysis: Brier scores, seasonal edge, strategy selection
  6. Generate all plots and reports
  7. Save best strategy config for OOS validation

Outputs:
  - results/kalshi_real_2023_2024/ (all strategies, plots, reports)

Usage:
    python run_kalshi_real_backtest.py
"""

import json
import os
import sys
import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.kalshi_client import (
    build_historical_comparison,
    compute_brier_scores,
    generate_market_report,
)
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    generate_strategy_grid,
    run_comprehensive_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Apply a clean plot style, with graceful fallback
_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)


# ============================================================================
# Season helpers
# ============================================================================

def _get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


# ============================================================================
# Step 1-2: Load Real Data
# ============================================================================

def load_real_data(data_dir="data"):
    """Load real Kalshi market data and model predictions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (market_df, predictions_df)
    """
    market_path = os.path.join(data_dir, "real_kalshi_2023_2024.csv")
    pred_path = os.path.join(data_dir, "real_model_predictions_2023_2024.csv")

    if not os.path.exists(market_path):
        raise FileNotFoundError(
            f"Real Kalshi market data not found: {market_path}\n"
            "Run: python scripts/fetch_kalshi_markets.py"
        )
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Real model predictions not found: {pred_path}\n"
            "Run: python scripts/generate_max_training_predictions.py"
        )

    market_df = pd.read_csv(market_path)
    predictions_df = pd.read_csv(pred_path)

    logger.info("Loaded real Kalshi market data: %d rows, %d days",
                len(market_df), market_df["date"].nunique())
    logger.info("Loaded real model predictions: %d days", len(predictions_df))

    return market_df, predictions_df


# ============================================================================
# Step 3-4: Align Model Predictions with Market Data
# ============================================================================

def align_model_and_market(market_df, predictions_df):
    """Align model predictions with market data using build_historical_comparison.

    For the backtesting engine, we need to use Kalshi's original settlement
    outcomes (API result field), not recomputed outcomes from GHCN TMAX.
    Kalshi settles based on NWS data, which can differ slightly from GHCN.
    """
    # build_historical_comparison expects market data without actual_tmax
    # (it will get actual_tmax from predictions_df)
    # But we also need to preserve the API settlement outcomes.

    # Save original API outcomes
    api_outcomes = market_df[["date", "ticker", "actual_outcome"]].copy()

    # Remove actual_tmax from market data (predictions_df carries it)
    market_clean = market_df.drop(columns=["actual_tmax"], errors="ignore")

    # Build comparison (this computes model_prob for each bucket)
    comparison_df = build_historical_comparison(
        model_predictions_df=predictions_df,
        historical_markets_df=market_clean,
    )

    if comparison_df.empty:
        raise ValueError("No overlapping dates between model and market data")

    # Use Kalshi's original settlement result instead of recomputed outcome.
    # The 'outcome' column from build_historical_comparison is based on
    # GHCN TMAX, but actual settlements use NWS data.
    # Merge back the API's actual_outcome.
    api_outcomes["date"] = pd.to_datetime(api_outcomes["date"]).dt.date
    comparison_df["date"] = comparison_df["date"].apply(
        lambda x: x if isinstance(x, date) else pd.Timestamp(x).date()
    )

    # The comparison_df may already have an actual_outcome column from the
    # market data merge. Replace 'outcome' with the API's actual_outcome.
    if "actual_outcome" in comparison_df.columns:
        comparison_df["outcome"] = comparison_df["actual_outcome"].astype(float)
    elif "outcome" in comparison_df.columns:
        # outcome was computed from GHCN; for maximum accuracy, use it
        pass

    logger.info(
        "Aligned %d rows across %d unique dates",
        len(comparison_df), comparison_df["date"].nunique(),
    )
    logger.info(
        "Mean model prob: %.3f, Mean market prob: %.3f, Mean delta: %.4f",
        comparison_df["model_prob"].mean(),
        comparison_df["market_prob"].mean(),
        comparison_df["prob_delta"].mean(),
    )

    return comparison_df


# ============================================================================
# Step 5: Comprehensive Strategy Grid Search
# ============================================================================

def run_strategy_grid_search(comparison_df, output_dir):
    """Run comprehensive strategy grid search per the backtesting plan."""
    os.makedirs(output_dir, exist_ok=True)

    backtest_data = comparison_df.copy()

    # Prepare columns for BacktestEngine
    if "actual_outcome" in backtest_data.columns and "outcome" in backtest_data.columns:
        backtest_data = backtest_data.drop(columns=["actual_outcome"])

    backtest_data = backtest_data.rename(columns={
        "market_prob": "market_price",
        "outcome": "actual_outcome",
    })

    required = ["date", "model_prob", "market_price", "actual_outcome"]
    missing = [c for c in required if c not in backtest_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with NaN in critical columns
    before = len(backtest_data)
    backtest_data = backtest_data.dropna(subset=required).reset_index(drop=True)
    logger.info("Dropped %d rows with NaN (kept %d)", before - len(backtest_data), len(backtest_data))

    # Generate strategy grid (exact plan parameters)
    strategies = generate_strategy_grid(
        ev_thresholds=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        sizing_methods=["fixed", "proportional", "fractional_kelly", "capped_kelly"],
        kelly_fractions=[0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
        fee_rates=[0.07],
        max_positions=[0.05, 0.10, 0.15, 0.20],
        bankrolls=[10000],
    )

    logger.info("Generated %d strategy permutations", len(strategies))

    # Run comprehensive backtest
    results = run_comprehensive_backtest(
        backtest_data,
        output_dir=output_dir,
        strategies=strategies,
        max_strategies=1000,
    )

    # Save all strategies CSV
    comparison_csv = results["comparison_df"]
    comparison_csv.to_csv(
        os.path.join(output_dir, "all_strategies.csv"), index=False,
    )

    logger.info(
        "Strategy grid search complete: %d strategies evaluated",
        len(results["all_results"]),
    )

    return results


# ============================================================================
# Step 6: Analyze Results
# ============================================================================

def analyze_results(comparison_df, backtest_results, output_dir):
    """Comprehensive analysis of in-sample 2023-2024 results."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = backtest_results["all_results"]
    strategies_df = backtest_results["comparison_df"]

    analysis = {}

    # ---- A) Brier Score Comparison ----
    valid = comparison_df.dropna(subset=["model_prob", "market_prob", "outcome"])
    if len(valid) > 0:
        brier = compute_brier_scores(
            model_probs=valid["model_prob"],
            market_probs=valid["market_prob"],
            outcomes=valid["outcome"],
        )
        analysis["brier"] = brier
        logger.info(
            "Brier scores: model=%.4f, market=%.4f, delta=%.4f (negative=model better)",
            brier["model_brier"], brier["market_brier"], brier["brier_delta"],
        )

        # Monthly Brier breakdown
        valid_copy = valid.copy()
        valid_copy["month"] = pd.to_datetime(valid_copy["date"]).dt.month
        monthly_brier = []
        for month in range(1, 13):
            mask = valid_copy["month"] == month
            if mask.sum() > 0:
                m_df = valid_copy[mask]
                m_brier = compute_brier_scores(
                    m_df["model_prob"], m_df["market_prob"], m_df["outcome"],
                )
                monthly_brier.append({
                    "month": month,
                    "model_brier": m_brier["model_brier"],
                    "market_brier": m_brier["market_brier"],
                    "brier_delta": m_brier["brier_delta"],
                    "n_samples": m_brier["n_samples"],
                })
        monthly_brier_df = pd.DataFrame(monthly_brier)
        monthly_brier_df.to_csv(
            os.path.join(output_dir, "monthly_brier.csv"), index=False,
        )
        analysis["monthly_brier"] = monthly_brier_df

        # Save Brier comparison JSON
        with open(os.path.join(output_dir, "brier_comparison.json"), "w") as f:
            json.dump(brier, f, indent=2)

    # ---- B) Best Strategies ----
    if not strategies_df.empty:
        trading = strategies_df[strategies_df["n_trades"] > 0].copy()

        if not trading.empty:
            finite_sharpe = trading[trading["sharpe_ratio"].apply(lambda x: np.isfinite(x))]
            if not finite_sharpe.empty:
                best_sharpe = finite_sharpe.nlargest(10, "sharpe_ratio")
                best_sharpe.to_csv(
                    os.path.join(output_dir, "top10_by_sharpe.csv"), index=False,
                )
                analysis["best_by_sharpe"] = best_sharpe

            best_pnl = trading.nlargest(10, "total_pnl")
            best_pnl.to_csv(
                os.path.join(output_dir, "top10_by_pnl.csv"), index=False,
            )
            analysis["best_by_pnl"] = best_pnl

    # ---- C) Seasonal Edge Analysis ----
    if "outcome" in comparison_df.columns:
        valid_copy = comparison_df.dropna(subset=["model_prob", "market_prob", "outcome"]).copy()
        valid_copy["date_dt"] = pd.to_datetime(valid_copy["date"])
        valid_copy["month"] = valid_copy["date_dt"].dt.month
        valid_copy["season"] = valid_copy["month"].apply(_get_season)

        seasonal_edge = []
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            mask = valid_copy["season"] == season
            if mask.sum() > 0:
                s_df = valid_copy[mask]
                s_brier = compute_brier_scores(
                    s_df["model_prob"], s_df["market_prob"], s_df["outcome"],
                )
                mean_delta = float(s_df["prob_delta"].mean())
                mean_abs_delta = float(s_df["prob_delta"].abs().mean())
                seasonal_edge.append({
                    "season": season,
                    "n_samples": s_brier["n_samples"],
                    "model_brier": s_brier["model_brier"],
                    "market_brier": s_brier["market_brier"],
                    "brier_delta": s_brier["brier_delta"],
                    "mean_prob_delta": mean_delta,
                    "mean_abs_prob_delta": mean_abs_delta,
                })

        seasonal_df = pd.DataFrame(seasonal_edge)
        seasonal_df.to_csv(
            os.path.join(output_dir, "seasonal_analysis.csv"), index=False,
        )
        analysis["seasonal_edge"] = seasonal_df
        logger.info("Seasonal edge analysis:\n%s", seasonal_df.to_string(index=False))

    # ---- D) Strategy Selection ----
    best_strategy_config = _select_best_strategy(strategies_df, all_results)
    analysis["best_strategy_config"] = best_strategy_config

    with open(os.path.join(output_dir, "best_strategy_config.json"), "w") as f:
        json.dump(best_strategy_config, f, indent=2)

    logger.info("Best strategy config: %s", json.dumps(best_strategy_config, indent=2))

    return analysis


def _select_best_strategy(strategies_df, all_results):
    """Select the best strategy using the plan's criteria."""
    import re

    if strategies_df.empty:
        return {"error": "No strategies to evaluate"}

    # Apply plan criteria
    candidates = strategies_df[
        (strategies_df["n_trades"] >= 100)
        & (strategies_df["win_rate"] > 0.30)
        & (strategies_df["sharpe_ratio"] >= 1.5)
        & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        & (strategies_df["max_drawdown"] < 2000.0)
    ].copy()

    if candidates.empty:
        logger.warning("No strategies meet all criteria. Relaxing...")
        candidates = strategies_df[
            (strategies_df["n_trades"] >= 50)
            & (strategies_df["win_rate"] > 0.25)
            & (strategies_df["sharpe_ratio"] >= 0.5)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            & (strategies_df["max_drawdown"] < 3000.0)
        ].copy()

    if candidates.empty:
        valid = strategies_df[
            (strategies_df["n_trades"] > 0)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        ]
        if valid.empty:
            return {"error": "No strategies with trades"}
        best = valid.loc[valid["sharpe_ratio"].idxmax()]
        return _config_from_strategy_row(best, reason="fallback_best_sharpe")

    # Tie-breaking
    top_sharpe = candidates["sharpe_ratio"].max()
    if top_sharpe > 2.0:
        near_top = candidates[candidates["sharpe_ratio"] >= top_sharpe - 0.2]
        if not near_top.empty:
            best = near_top.loc[near_top["total_pnl"].idxmax()]
            return _config_from_strategy_row(best, reason="best_pnl_near_top_sharpe")

    best = candidates.loc[candidates["sharpe_ratio"].idxmax()]
    return _config_from_strategy_row(best, reason="highest_sharpe")


def _config_from_strategy_row(row, reason=""):
    """Extract strategy configuration from a DataFrame row."""
    import re
    name = row["strategy_name"]

    config = {
        "strategy_name": name,
        "total_pnl": float(row["total_pnl"]),
        "roi": float(row["roi"]),
        "sharpe_ratio": float(row["sharpe_ratio"]),
        "max_drawdown": float(row["max_drawdown"]),
        "win_rate": float(row["win_rate"]),
        "n_trades": int(row["n_trades"]),
        "selection_reason": reason,
    }

    ev_match = re.search(r"ev(\d+\.\d+)", name)
    if ev_match:
        config["ev_threshold"] = float(ev_match.group(1))

    sizing_match = re.search(r"_(\w+)_kf", name)
    if sizing_match:
        config["sizing_method"] = sizing_match.group(1)

    kf_match = re.search(r"kf(\d+\.\d+)", name)
    if kf_match:
        config["kelly_fraction"] = float(kf_match.group(1))

    fee_match = re.search(r"fee(\d+\.\d+)", name)
    if fee_match:
        config["fee_rate"] = float(fee_match.group(1))

    mp_match = re.search(r"mp(\d+\.\d+)", name)
    if mp_match:
        config["max_position_frac"] = float(mp_match.group(1))

    br_match = re.search(r"br(\d+)", name)
    if br_match:
        config["bankroll"] = int(br_match.group(1))

    return config


# ============================================================================
# Visualization
# ============================================================================

def generate_all_plots(comparison_df, analysis, backtest_results, output_dir):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # ---- 1. Model vs Market Scatter ----
    fig, ax = plt.subplots(figsize=(8, 8))
    valid = comparison_df.dropna(subset=["model_prob", "market_prob"])
    if len(valid) > 0:
        ax.scatter(
            valid["market_prob"], valid["model_prob"],
            alpha=0.3, s=10, edgecolors="none", c="#1f77b4",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Agreement line")
        ax.set_xlabel("Market Probability", fontsize=12)
        ax.set_ylabel("Model Probability", fontsize=12)
        ax.set_title("Model vs Market Probability (Real Data, 2023-2024)", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "model_vs_market_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. Brier by Month ----
    if "monthly_brier" in analysis:
        monthly = analysis["monthly_brier"]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(monthly))
        width = 0.35
        ax.bar(x - width / 2, monthly["model_brier"], width,
               label="Model", color="#4c72b0", alpha=0.8)
        ax.bar(x + width / 2, monthly["market_brier"], width,
               label="Market", color="#d62728", alpha=0.8)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Brier Score (lower = better)", fontsize=12)
        ax.set_title("Brier Score: Model vs Market by Month (Real Data)", fontsize=14)
        ax.set_xticks(x)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax.set_xticklabels(
            [month_labels[m - 1] for m in monthly["month"]]
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        path = os.path.join(output_dir, "brier_by_month.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 3. Seasonal Edge ----
    if "seasonal_edge" in analysis:
        se_df = analysis["seasonal_edge"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#2ca02c" if d < 0 else "#d62728" for d in se_df["brier_delta"]]
        axes[0].bar(se_df["season"], se_df["brier_delta"], color=colors, alpha=0.8)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set_ylabel("Brier Delta (negative = model better)")
        axes[0].set_title("Brier Score Edge by Season")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[1].bar(se_df["season"], se_df["mean_abs_prob_delta"],
                     color="#ff7f0e", alpha=0.8)
        axes[1].set_ylabel("Mean |Prob Delta|")
        axes[1].set_title("Mean Probability Disagreement by Season")
        axes[1].grid(True, alpha=0.3, axis="y")
        path = os.path.join(output_dir, "seasonal_edge.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 4. Best Strategy PnL Curves ----
    all_results = backtest_results.get("all_results", [])
    trading_results = [r for r in all_results if r.n_trades > 0]
    if trading_results:
        top5 = sorted(trading_results, key=lambda r: r.total_pnl, reverse=True)[:5]
        fig, ax = plt.subplots(figsize=(14, 7))
        for result in top5:
            if len(result.cumulative_pnl) > 0:
                ax.plot(
                    result.cumulative_pnl,
                    label=f"PnL=${result.total_pnl:.0f}, Sharpe={result.sharpe_ratio:.2f}",
                    linewidth=1.2,
                )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trade Index", fontsize=12)
        ax.set_ylabel("Cumulative P&L ($)", fontsize=12)
        ax.set_title("Top 5 Strategies: Cumulative P&L (Real Data, 2023-2024)", fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "top5_pnl_curves.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        # ---- 5. Drawdown Analysis ----
        fig, ax = plt.subplots(figsize=(14, 6))
        for result in top5:
            cum_pnl = result.cumulative_pnl
            if len(cum_pnl) == 0:
                continue
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = running_max - cum_pnl
            ax.plot(drawdown, label=f"MaxDD=${result.max_drawdown:.0f}", linewidth=1.0)
        ax.set_xlabel("Trade Index", fontsize=12)
        ax.set_ylabel("Drawdown ($)", fontsize=12)
        ax.set_title("Top 5 Strategies: Drawdown (Real Data)", fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "drawdown_analysis.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 6. Strategy Heatmap ----
    strategies_df = backtest_results.get("comparison_df", pd.DataFrame())
    if not strategies_df.empty:
        try:
            import re
            df = strategies_df.copy()
            df["ev_threshold"] = df["strategy_name"].str.extract(r"ev(\d+\.\d+)").astype(float)
            df["sizing"] = df["strategy_name"].str.extract(r"_(\w+)_kf")
            if not df["ev_threshold"].isna().all() and not df["sizing"].isna().all():
                pivot = df.groupby(["ev_threshold", "sizing"])["sharpe_ratio"].mean()
                pivot = pivot.reset_index()
                pivot = pivot[pivot["sharpe_ratio"].apply(lambda x: np.isfinite(x))]
                if not pivot.empty:
                    pivot_table = pivot.pivot(
                        index="ev_threshold", columns="sizing", values="sharpe_ratio"
                    )
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(
                        pivot_table.values, aspect="auto", cmap="RdYlGn",
                        interpolation="nearest",
                    )
                    ax.set_xticks(range(len(pivot_table.columns)))
                    ax.set_xticklabels(pivot_table.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(pivot_table.index)))
                    ax.set_yticklabels([f"{v:.3f}" for v in pivot_table.index])
                    ax.set_xlabel("Sizing Method", fontsize=12)
                    ax.set_ylabel("EV Threshold", fontsize=12)
                    ax.set_title("Strategy Heatmap: Mean Sharpe Ratio (Real Data)", fontsize=14)
                    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
                    path = os.path.join(output_dir, "strategy_heatmap.png")
                    fig.tight_layout()
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    saved.append(path)
        except Exception as e:
            logger.warning("Could not generate strategy heatmap: %s", e)

    # ---- 7. Monthly P&L for best strategy ----
    if trading_results:
        best_result = max(trading_results, key=lambda r: r.total_pnl)
        if best_result.trades:
            trade_df = pd.DataFrame(best_result.trades)
            trade_df["date"] = pd.to_datetime(trade_df["date"])
            trade_df["month"] = trade_df["date"].dt.to_period("M")
            monthly_pnl = trade_df.groupby("month")["pnl"].sum()
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly_pnl.values]
            ax.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors)
            ax.set_xticks(range(len(monthly_pnl)))
            ax.set_xticklabels(
                [str(m) for m in monthly_pnl.index],
                rotation=45, ha="right", fontsize=8,
            )
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("P&L ($)", fontsize=12)
            ax.set_title(f"Best Strategy Monthly P&L (Real Data)", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")
            path = os.path.join(output_dir, "best_strategy_monthly_pnl.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    # ---- 8. Actual Temperature Time Series ----
    if "actual_tmax" in comparison_df.columns:
        daily = comparison_df.drop_duplicates("date").sort_values("date")
        fig, ax = plt.subplots(figsize=(14, 5))
        dates = pd.to_datetime(daily["date"])
        ax.plot(dates, daily["actual_tmax"], linewidth=0.8, color="#1f77b4", alpha=0.8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature (F)")
        ax.set_title("NYC Central Park TMAX (Real GHCN Data, 2023-2024)")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "actual_tmax_timeseries.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ============================================================================
# Report Generation
# ============================================================================

def generate_backtest_report(comparison_df, analysis, backtest_results, output_dir):
    """Generate a comprehensive text report."""
    os.makedirs(output_dir, exist_ok=True)

    strategies_df = backtest_results.get("comparison_df", pd.DataFrame())
    all_results = backtest_results.get("all_results", [])

    lines = [
        "=" * 78,
        "  KALSHI KXHIGHNY REAL DATA BACKTEST REPORT -- In-Sample 2023-2024",
        "=" * 78,
        "",
        "  DATA SOURCE: Real Kalshi API + Real GHCN Central Park Observations",
        "  MODEL: Neural network trained on 2018-2021, validated on 2022",
        "",
    ]

    # ---- Data Summary ----
    lines.append("--- DATA SUMMARY ---")
    lines.append(f"  Total market rows: {len(comparison_df):,}")
    lines.append(f"  Unique trading days: {comparison_df['date'].nunique():,}")

    if "actual_tmax" in comparison_df.columns:
        tmax_vals = comparison_df.drop_duplicates("date")["actual_tmax"].dropna()
        lines.append(f"  TMAX range: {tmax_vals.min():.1f}F to {tmax_vals.max():.1f}F")
        lines.append(f"  TMAX mean: {tmax_vals.mean():.1f}F")

    # Direction breakdown
    if "direction" in comparison_df.columns:
        for d, c in comparison_df["direction"].value_counts().items():
            lines.append(f"  {d} contracts: {c}")
    lines.append("")

    # ---- Brier Score Comparison ----
    if "brier" in analysis:
        brier = analysis["brier"]
        lines.append("--- BRIER SCORE COMPARISON (Model vs Market) ---")
        lines.append(f"  Model Brier score:  {brier['model_brier']:.4f}")
        lines.append(f"  Market Brier score: {brier['market_brier']:.4f}")
        lines.append(f"  Delta:              {brier['brier_delta']:.4f} "
                      f"({'MODEL BETTER' if brier['brier_delta'] < 0 else 'MARKET BETTER'})")
        lines.append(f"  N samples:          {brier['n_samples']:,}")
        if brier['market_brier'] > 0:
            pct = (brier['market_brier'] - brier['model_brier']) / brier['market_brier'] * 100
            lines.append(f"  Model edge:         {pct:.1f}% better Brier score")
        lines.append("")

    # ---- Seasonal Edge ----
    if "seasonal_edge" in analysis:
        lines.append("--- SEASONAL EDGE ANALYSIS ---")
        se_df = analysis["seasonal_edge"]
        for _, row in se_df.iterrows():
            edge_str = "MODEL+" if row["brier_delta"] < 0 else "MARKET+"
            lines.append(
                f"  {row['season']:8s}: Brier delta={row['brier_delta']:+.4f} ({edge_str}), "
                f"|prob_delta|={row['mean_abs_prob_delta']:.4f}, n={row['n_samples']}"
            )
        lines.append("")

    # ---- Strategy Grid Results ----
    lines.append("--- STRATEGY GRID RESULTS ---")
    lines.append(f"  Total strategies evaluated: {len(all_results):,}")
    trading_results = [r for r in all_results if r.n_trades > 0]
    lines.append(f"  Strategies with trades: {len(trading_results):,}")

    if not strategies_df.empty:
        profitable = strategies_df[strategies_df["total_pnl"] > 0]
        lines.append(f"  Profitable strategies: {len(profitable):,}")
        if len(trading_results) > 0:
            pnls = [r.total_pnl for r in trading_results]
            lines.append(f"  Mean P&L: ${np.mean(pnls):.2f}")
            lines.append(f"  Median P&L: ${np.median(pnls):.2f}")
            lines.append(f"  Std P&L: ${np.std(pnls):.2f}")
            lines.append(f"  % Profitable: {100 * sum(1 for p in pnls if p > 0) / len(pnls):.1f}%")
    lines.append("")

    # ---- Top 5 by P&L ----
    if "best_by_pnl" in analysis:
        lines.append("--- TOP 5 STRATEGIES BY P&L ---")
        top = analysis["best_by_pnl"].head(5)
        for _, row in top.iterrows():
            lines.append(
                f"  {row['strategy_name']}: "
                f"PnL=${row['total_pnl']:.2f}, "
                f"ROI={row['roi']*100:.1f}%, "
                f"Sharpe={row['sharpe_ratio']:.2f}, "
                f"WR={row['win_rate']*100:.0f}%, "
                f"DD=${row['max_drawdown']:.0f}, "
                f"N={row['n_trades']}"
            )
        lines.append("")

    # ---- Top 5 by Sharpe ----
    if "best_by_sharpe" in analysis:
        lines.append("--- TOP 5 STRATEGIES BY SHARPE RATIO ---")
        top = analysis["best_by_sharpe"].head(5)
        for _, row in top.iterrows():
            lines.append(
                f"  {row['strategy_name']}: "
                f"Sharpe={row['sharpe_ratio']:.2f}, "
                f"PnL=${row['total_pnl']:.2f}, "
                f"WR={row['win_rate']*100:.0f}%, "
                f"DD=${row['max_drawdown']:.0f}, "
                f"N={row['n_trades']}"
            )
        lines.append("")

    # ---- Selected Best Strategy ----
    if "best_strategy_config" in analysis:
        cfg = analysis["best_strategy_config"]
        lines.append("--- SELECTED BEST STRATEGY ---")
        lines.append(f"  Selection reason: {cfg.get('selection_reason', 'N/A')}")
        lines.append(f"  Name: {cfg.get('strategy_name', 'N/A')}")
        lines.append(f"  EV threshold: {cfg.get('ev_threshold', 'N/A')}")
        lines.append(f"  Sizing method: {cfg.get('sizing_method', 'N/A')}")
        lines.append(f"  Kelly fraction: {cfg.get('kelly_fraction', 'N/A')}")
        lines.append(f"  Fee rate: {cfg.get('fee_rate', 'N/A')}")
        lines.append(f"  Max position: {cfg.get('max_position_frac', 'N/A')}")
        lines.append(f"  Bankroll: ${cfg.get('bankroll', 'N/A')}")
        lines.append(f"  ---")
        lines.append(f"  Total P&L: ${cfg.get('total_pnl', 0):.2f}")
        lines.append(f"  ROI: {cfg.get('roi', 0)*100:.1f}%")
        lines.append(f"  Sharpe ratio: {cfg.get('sharpe_ratio', 0):.2f}")
        lines.append(f"  Max drawdown: ${cfg.get('max_drawdown', 0):.2f}")
        lines.append(f"  Win rate: {cfg.get('win_rate', 0)*100:.1f}%")
        lines.append(f"  N trades: {cfg.get('n_trades', 0)}")
        lines.append("")

    lines.extend(["=" * 78, ""])

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info("Saved backtest report to %s", report_path)
    return report_text


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 78)
    print("  KALSHI KXHIGHNY REAL-DATA BACKTEST -- Part 1: In-Sample (2023-2024)")
    print("=" * 78)
    print()
    print("  Using REAL data:")
    print("    - Kalshi API market data (4,377 contracts, 730 days)")
    print("    - GHCN Central Park TMAX observations")
    print("    - Neural network predictions (trained 2018-2021, val 2022)")
    print()

    output_dir = os.path.join("results", "kalshi_real_2023_2024")
    data_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # ===== Step 1-2: Load real data =====
    print("Step 1-2: Loading real Kalshi market data and model predictions...")
    market_df, predictions_df = load_real_data(data_dir)

    print(f"  Market data: {len(market_df):,} rows, {market_df['date'].nunique()} days")
    print(f"  Predictions: {len(predictions_df)} days")
    print(f"  Market prob range: {market_df['market_prob'].min():.3f} to "
          f"{market_df['market_prob'].max():.3f}")
    print(f"  Model mu range: {predictions_df['model_mu'].min():.1f}F to "
          f"{predictions_df['model_mu'].max():.1f}F")
    print()

    # ===== Step 3-4: Align model and market data =====
    print("Step 3-4: Aligning model predictions with market data...")
    comparison_df = align_model_and_market(market_df, predictions_df)

    print(f"  Aligned rows: {len(comparison_df):,}")
    print(f"  Unique dates: {comparison_df['date'].nunique()}")
    print(f"  Mean model prob: {comparison_df['model_prob'].mean():.3f}")
    print(f"  Mean market prob: {comparison_df['market_prob'].mean():.3f}")
    print(f"  Mean prob delta: {comparison_df['prob_delta'].mean():.4f}")
    print()

    # ===== Step 5: Strategy grid search =====
    print("Step 5: Running comprehensive strategy grid search...")
    print("  (This may take a few minutes...)")
    backtest_results = run_strategy_grid_search(comparison_df, output_dir)

    strategies_df = backtest_results["comparison_df"]
    all_results = backtest_results["all_results"]
    trading_results = [r for r in all_results if r.n_trades > 0]

    print(f"  Strategies evaluated: {len(all_results):,}")
    print(f"  Strategies with trades: {len(trading_results):,}")
    if not strategies_df.empty:
        profitable = strategies_df[strategies_df["total_pnl"] > 0]
        print(f"  Profitable strategies: {len(profitable):,}")
    print()

    # ===== Step 6: Analyze results =====
    print("Step 6: Analyzing in-sample results...")
    analysis = analyze_results(comparison_df, backtest_results, output_dir)
    print()

    # ===== Generate visualizations =====
    print("Generating visualizations...")
    saved_plots = generate_all_plots(comparison_df, analysis, backtest_results, output_dir)
    print(f"  Saved {len(saved_plots)} plots")
    print()

    # ===== Generate market report =====
    print("Generating market report...")
    generate_market_report(comparison_df, output_dir=output_dir)
    print()

    # ===== Generate text report =====
    print("Generating comprehensive report...")
    report = generate_backtest_report(comparison_df, analysis, backtest_results, output_dir)
    print()
    print(report)

    # ===== List output files =====
    print("Output files:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fpath} ({size_kb:.1f} KB)")

    print()
    print("=" * 78)
    print("  REAL DATA BACKTEST COMPLETE!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
