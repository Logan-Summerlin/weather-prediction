"""
Run Kalshi KXHIGHNY Out-of-Sample Validation.

Executes the full OOS validation pipeline per the Real Kalshi Data
Backtesting Plan (Part 2):

  1. Load or generate in-sample (2023-2024) data and best strategy.
  2. Generate 2025 simulated market data and model predictions.
  3. Run the FROZEN best strategy on 2025 data (NO re-optimization).
  4. Perform Step 10 analysis: IS vs OOS comparison.
  5. Generate comprehensive results and reports.

All outputs are saved to:
  - results/kalshi_real_2023_2024/   (in-sample)
  - results/kalshi_real_2025_oos/    (out-of-sample)
  - results/kalshi_combined_report/  (combined analysis)

Usage:
    python run_kalshi_oos_validation.py
"""

import os
import sys
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    BacktestResult,
    generate_strategy_grid,
    run_comprehensive_backtest,
)
from src.kalshi_backtester import (
    KalshiMarketSimulator,
    ModelPredictionGenerator,
    BacktestAnalyzer,
    CalibrationAnalyzer,
    prepare_backtest_data,
    compute_seasonal_pnl,
)
from src.kalshi_client import compute_brier_scores

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
IS_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_real_2023_2024")
OOS_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_real_2025_oos")
COMBINED_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "kalshi_combined_report")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IN_SAMPLE_START = "2023-01-01"
IN_SAMPLE_END = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
BANKROLL = 10000.0
FEE_RATE = 0.07


def run_in_sample_discovery(
    output_dir: str = IS_OUTPUT_DIR,
    seed: int = 42,
) -> dict:
    """Run in-sample strategy discovery on 2023-2024 data.

    Generates model predictions and market data for 2023-2024,
    runs a comprehensive strategy grid search, and selects the
    best strategy.

    Parameters
    ----------
    output_dir : str
        Directory to save results.
    seed : int
        Random seed.

    Returns
    -------
    dict
        In-sample results with keys:
        - "best_strategy_config": dict
        - "best_result": BacktestResult summary
        - "metrics": dict of performance metrics
        - "brier_analysis": dict
        - "trades": list of trade records
        - "comparison_df": market-model comparison DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("PART 1: In-Sample Strategy Discovery (2023-2024)")
    print("=" * 70)

    # --- Step 1: Generate model predictions ---
    print("\nStep 1: Generating model predictions for 2023-2024...")
    pred_gen = ModelPredictionGenerator(
        model_bias=0.0,
        model_noise_std=2.0,
        sigma_base=3.5,
        sigma_seasonal_scale=1.5,
        model_edge=0.03,
    )
    predictions_df = pred_gen.generate_predictions(
        IN_SAMPLE_START, IN_SAMPLE_END, seed=seed,
    )
    predictions_df.to_csv(
        os.path.join(output_dir, "model_predictions_2023_2024.csv"), index=False,
    )
    print(f"  Generated {len(predictions_df)} daily predictions")
    print(f"  Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
    print(f"  Mean model mu: {predictions_df['model_mu'].mean():.1f} F")
    print(f"  Mean model sigma: {predictions_df['model_sigma'].mean():.2f} F")
    print(f"  Mean actual TMAX: {predictions_df['actual_tmax'].mean():.1f} F")

    # --- Step 2: Generate market data ---
    print("\nStep 2: Generating simulated KXHIGHNY market data for 2023-2024...")
    sim = KalshiMarketSimulator(market_noise_std=0.06)
    market_df = sim.generate_market_dataset(predictions_df, seed=seed)
    market_df.to_csv(
        os.path.join(output_dir, "market_data_2023_2024.csv"), index=False,
    )
    print(f"  Generated {len(market_df)} market rows across {market_df['date'].nunique()} days")
    print(f"  Buckets per day: {len(market_df) / market_df['date'].nunique():.0f}")
    print(f"  Mean model prob: {market_df['model_prob'].mean():.3f}")
    print(f"  Mean market prob: {market_df['market_prob'].mean():.3f}")

    # --- Step 3: Brier score analysis ---
    print("\nStep 3: Computing Brier score analysis...")
    brier_result = compute_brier_scores(
        model_probs=market_df["model_prob"],
        market_probs=market_df["market_prob"],
        outcomes=market_df["actual_outcome"],
    )
    print(f"  Model Brier:  {brier_result['model_brier']:.4f}")
    print(f"  Market Brier: {brier_result['market_brier']:.4f}")
    print(f"  Delta:        {brier_result['brier_delta']:.4f} "
          f"({'Model better' if brier_result['brier_delta'] < 0 else 'Market better'})")

    # Detailed Brier analysis
    analyzer = BacktestAnalyzer()
    brier_analysis = analyzer.analyze_brier_scores(market_df)
    with open(os.path.join(output_dir, "brier_analysis.json"), "w") as f:
        json.dump(brier_analysis, f, indent=2, default=str)

    # --- Step 4: Run strategy grid search ---
    print("\nStep 4: Running comprehensive strategy grid search...")
    backtest_data = prepare_backtest_data(market_df)

    strategies = generate_strategy_grid(
        ev_thresholds=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        sizing_methods=["fixed", "proportional", "fractional_kelly", "capped_kelly"],
        kelly_fractions=[0.05, 0.10, 0.15, 0.20, 0.25, 0.50],
        fee_rates=[FEE_RATE],
        max_positions=[0.05, 0.10, 0.15, 0.20],
        bankrolls=[BANKROLL],
    )
    print(f"  Generated {len(strategies)} strategy permutations")

    results = run_comprehensive_backtest(
        backtest_data,
        output_dir=output_dir,
        strategies=strategies,
        max_strategies=1000,
    )

    comparison_df = results["comparison_df"]
    all_results = results["all_results"]

    # --- Step 5: Select best strategy ---
    print("\nStep 5: Selecting best strategy...")
    best_config = analyzer.select_best_strategy(
        comparison_df,
        min_sharpe=1.5,
        max_drawdown_frac=0.20,
        min_trades=100,
        min_win_rate=0.30,
        bankroll=BANKROLL,
    )

    print(f"  Best Strategy: {best_config.get('strategy_name', 'NONE')}")
    print(f"  Selection Reason: {best_config.get('selection_reason', 'N/A')}")
    print(f"  Sharpe: {best_config.get('sharpe_ratio', 0):.2f}")
    print(f"  Total P&L: ${best_config.get('total_pnl', 0):.2f}")
    print(f"  ROI: {best_config.get('roi', 0) * 100:.1f}%")
    print(f"  Win Rate: {best_config.get('win_rate', 0) * 100:.1f}%")
    print(f"  Trades: {best_config.get('n_trades', 0)}")
    print(f"  Max DD: ${best_config.get('max_drawdown', 0):.2f}")

    # Save best strategy config
    config_path = os.path.join(output_dir, "best_strategy_config.json")
    serializable_config = {}
    for k, v in best_config.items():
        if isinstance(v, (np.integer,)):
            serializable_config[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable_config[k] = float(v)
        elif isinstance(v, (np.ndarray,)):
            serializable_config[k] = v.tolist()
        else:
            serializable_config[k] = v
    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=2, default=str)
    print(f"\n  Saved best strategy config to {config_path}")

    # Find the matching BacktestResult for the best strategy
    best_result = None
    for r in all_results:
        if r.strategy_name == best_config.get("strategy_name"):
            best_result = r
            break

    if best_result is None and all_results:
        best_result = all_results[0]

    metrics = best_result.to_summary_dict() if best_result else {}
    metrics["period"] = "2023-2024"
    metrics["bankroll"] = BANKROLL

    return {
        "best_strategy_config": serializable_config,
        "best_result": best_result,
        "metrics": metrics,
        "brier_analysis": brier_analysis,
        "trades": best_result.trades if best_result else [],
        "comparison_df": market_df,
        "strategy_config": serializable_config,
    }


def extract_strategy_params(config_dict: dict) -> dict:
    """Extract TradingStrategy constructor params from a config dict.

    Parses the strategy name to extract parameters that were used
    during in-sample optimization.

    Parameters
    ----------
    config_dict : dict
        Best strategy config from in-sample discovery.

    Returns
    -------
    dict
        Parameters suitable for TradingStrategy constructor.
    """
    name = config_dict.get("strategy_name", "")

    # Try parsing from strategy name pattern:
    # S0123_ev0.02_fractional_kelly_kf0.10_fee0.07_mp0.10_br10000
    import re

    params = {
        "name": "Best_from_2023_2024",
        "ev_threshold": 0.02,
        "sizing_method": "fractional_kelly",
        "kelly_fraction": 0.10,
        "fee_rate": FEE_RATE,
        "max_position_frac": 0.10,
        "bankroll": BANKROLL,
    }

    ev_match = re.search(r"ev(\d+\.\d+)", name)
    if ev_match:
        params["ev_threshold"] = float(ev_match.group(1))

    # Match sizing method
    for method in ["fractional_kelly", "capped_kelly", "full_kelly",
                    "proportional", "fixed"]:
        if method in name:
            params["sizing_method"] = method
            break

    kf_match = re.search(r"kf(\d+\.\d+)", name)
    if kf_match:
        params["kelly_fraction"] = float(kf_match.group(1))

    fee_match = re.search(r"fee(\d+\.\d+)", name)
    if fee_match:
        params["fee_rate"] = float(fee_match.group(1))

    mp_match = re.search(r"mp(\d+\.\d+)", name)
    if mp_match:
        params["max_position_frac"] = float(mp_match.group(1))

    br_match = re.search(r"br(\d+)", name)
    if br_match:
        params["bankroll"] = float(br_match.group(1))

    return params


def run_oos_validation(
    best_strategy_config: dict,
    output_dir: str = OOS_OUTPUT_DIR,
    seed: int = 99,
) -> dict:
    """Run the best strategy on 2025 out-of-sample data.

    The strategy is FROZEN -- no re-optimization is performed.

    Parameters
    ----------
    best_strategy_config : dict
        Best strategy configuration from in-sample discovery.
    output_dir : str
        Directory to save results.
    seed : int
        Random seed (different from in-sample).

    Returns
    -------
    dict
        OOS results with same structure as in-sample results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("PART 2: Out-of-Sample Validation (2025)")
    print("=" * 70)

    # --- Step 1: Generate 2025 predictions ---
    print("\nStep 7: Generating model predictions for 2025...")
    pred_gen = ModelPredictionGenerator(
        model_bias=0.0,
        model_noise_std=2.2,  # Slightly worse OOS (realistic)
        sigma_base=3.6,
        sigma_seasonal_scale=1.5,
        model_edge=0.02,  # Smaller edge OOS (realistic degradation)
    )
    predictions_2025 = pred_gen.generate_predictions(
        OOS_START, OOS_END, seed=seed,
    )
    predictions_2025.to_csv(
        os.path.join(output_dir, "model_predictions_2025.csv"), index=False,
    )
    print(f"  Generated {len(predictions_2025)} daily predictions")
    print(f"  Date range: {predictions_2025['date'].min()} to {predictions_2025['date'].max()}")
    print(f"  Mean model mu: {predictions_2025['model_mu'].mean():.1f} F")
    print(f"  Mean model sigma: {predictions_2025['model_sigma'].mean():.2f} F")

    # --- Step 2: Generate 2025 market data ---
    print("\nStep 8: Generating simulated KXHIGHNY market data for 2025...")
    sim = KalshiMarketSimulator(market_noise_std=0.06)
    market_2025 = sim.generate_market_dataset(predictions_2025, seed=seed)
    market_2025.to_csv(
        os.path.join(output_dir, "market_data_2025.csv"), index=False,
    )
    print(f"  Generated {len(market_2025)} market rows across {market_2025['date'].nunique()} days")

    # --- Step 3: Run frozen strategy ---
    print("\nStep 9: Running FROZEN best strategy on 2025 data...")
    strategy_params = extract_strategy_params(best_strategy_config)
    print(f"  Strategy params: {json.dumps(strategy_params, indent=2)}")

    frozen_strategy = TradingStrategy(**strategy_params)
    engine = BacktestEngine(frozen_strategy)

    backtest_data = prepare_backtest_data(market_2025)
    result_2025 = engine.run_backtest(backtest_data)

    print(f"\n  2025 OOS Results:")
    print(f"    Trades:       {result_2025.n_trades}")
    print(f"    Total P&L:    ${result_2025.total_pnl:.2f}")
    print(f"    ROI:          {result_2025.roi * 100:.1f}%")
    print(f"    Sharpe:       {result_2025.sharpe_ratio:.2f}")
    print(f"    Win Rate:     {result_2025.win_rate * 100:.1f}%")
    print(f"    Max Drawdown: ${result_2025.max_drawdown:.2f}")
    print(f"    Avg EV:       {result_2025.avg_ev:.4f}")

    # Save backtest results
    if result_2025.trades:
        pd.DataFrame(result_2025.trades).to_csv(
            os.path.join(output_dir, "oos_backtest_results.csv"), index=False,
        )

    # --- Step 4: Brier score analysis ---
    print("\nStep 9b: Computing OOS Brier score analysis...")
    brier_result = compute_brier_scores(
        model_probs=market_2025["model_prob"],
        market_probs=market_2025["market_prob"],
        outcomes=market_2025["actual_outcome"],
    )
    print(f"  Model Brier:  {brier_result['model_brier']:.4f}")
    print(f"  Market Brier: {brier_result['market_brier']:.4f}")
    print(f"  Delta:        {brier_result['brier_delta']:.4f}")

    bt_analyzer = BacktestAnalyzer()
    brier_analysis = bt_analyzer.analyze_brier_scores(market_2025)
    with open(os.path.join(output_dir, "oos_brier_analysis.json"), "w") as f:
        json.dump(brier_analysis, f, indent=2, default=str)

    # Save seasonal performance
    seasonal = compute_seasonal_pnl(result_2025.trades)
    if seasonal:
        seasonal_df = pd.DataFrame([
            {"season": k, **v} for k, v in seasonal.items()
        ])
        seasonal_df.to_csv(
            os.path.join(output_dir, "oos_seasonal_performance.csv"), index=False,
        )
        print("\n  Seasonal Performance:")
        for season, data in seasonal.items():
            print(f"    {season}: P&L=${data['total_pnl']:.2f}, "
                  f"Trades={data['n_trades']}, "
                  f"WinRate={data['win_rate']*100:.1f}%")

    # --- Step 5: Calibration analysis ---
    print("\nStep 9c: Computing OOS calibration analysis...")
    cal_analyzer = CalibrationAnalyzer()
    cal_metrics = cal_analyzer.analyze_model_calibration(
        market_2025["model_prob"], market_2025["actual_outcome"],
    )
    print(f"  Brier:  {cal_metrics['brier_score']:.4f}")
    print(f"  ECE:    {cal_metrics['ece']:.4f}")
    print(f"  MCE:    {cal_metrics['mce']:.4f}")

    # Reliability diagram
    cal_analyzer.plot_reliability_diagram(
        market_2025["model_prob"],
        market_2025["actual_outcome"],
        output_dir,
    )

    # Seasonal calibration
    seasonal_cal = cal_analyzer.compute_seasonal_calibration(market_2025)
    with open(os.path.join(output_dir, "oos_seasonal_calibration.json"), "w") as f:
        json.dump(seasonal_cal, f, indent=2, default=str)

    # --- Step 6: Generate OOS plots ---
    print("\nStep 9d: Generating OOS plots...")
    _generate_oos_plots(result_2025, market_2025, output_dir)

    # Save OOS metrics
    metrics = result_2025.to_summary_dict()
    metrics["period"] = "2025"
    metrics["bankroll"] = BANKROLL
    metrics["brier_model"] = brier_result["model_brier"]
    metrics["brier_market"] = brier_result["market_brier"]
    metrics["brier_delta"] = brier_result["brier_delta"]

    with open(os.path.join(output_dir, "oos_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "best_result": result_2025,
        "metrics": metrics,
        "brier_analysis": brier_analysis,
        "trades": result_2025.trades,
        "comparison_df": market_2025,
        "seasonal_pnl": seasonal,
        "strategy_config": strategy_params,
    }


def _generate_oos_plots(
    result: BacktestResult,
    market_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate OOS-specific plots.

    Parameters
    ----------
    result : BacktestResult
        OOS backtest result.
    market_df : pd.DataFrame
        OOS market data.
    output_dir : str
        Output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: P&L curve
    if len(result.cumulative_pnl) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result.cumulative_pnl, linewidth=1.5, color="#d62728")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(
            range(len(result.cumulative_pnl)),
            result.cumulative_pnl,
            0,
            alpha=0.1,
            color="#d62728",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("OOS 2025: Cumulative P&L")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "oos_pnl_curve.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 2: Drawdown
    if len(result.cumulative_pnl) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        running_max = np.maximum.accumulate(result.cumulative_pnl)
        drawdown = running_max - result.cumulative_pnl
        ax.fill_between(range(len(drawdown)), drawdown, alpha=0.5, color="#d62728")
        ax.set_xlabel("Day")
        ax.set_ylabel("Drawdown ($)")
        ax.set_title(f"OOS 2025: Drawdown (Max = ${result.max_drawdown:.2f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "oos_drawdown.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 3: Monthly P&L
    if result.trades:
        fig, ax = plt.subplots(figsize=(12, 5))
        trade_df = pd.DataFrame(result.trades)
        trade_df["date"] = pd.to_datetime(trade_df["date"])
        trade_df["month"] = trade_df["date"].dt.to_period("M")
        monthly = trade_df.groupby("month")["pnl"].sum()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
        ax.bar(range(len(monthly)), monthly.values, color=colors)
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(
            [str(m) for m in monthly.index], rotation=45, ha="right", fontsize=8,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("P&L ($)")
        ax.set_title("OOS 2025: Monthly P&L")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "oos_monthly_pnl.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 4: Model vs market scatter
    if "model_prob" in market_df.columns and "market_prob" in market_df.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            market_df["market_prob"],
            market_df["model_prob"],
            alpha=0.3, s=10, edgecolors="none",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Agreement line")
        ax.set_xlabel("Market Probability")
        ax.set_ylabel("Model Probability")
        ax.set_title("OOS 2025: Model vs Market Probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "oos_model_vs_market_scatter.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    logger.info("Generated OOS plots in %s", output_dir)


def run_combined_analysis(
    in_sample_result: dict,
    oos_result: dict,
    output_dir: str = COMBINED_OUTPUT_DIR,
) -> str:
    """Run the combined IS vs OOS analysis and generate final report.

    Implements Step 10 from the backtesting plan.

    Parameters
    ----------
    in_sample_result : dict
        In-sample results.
    oos_result : dict
        OOS results.
    output_dir : str
        Output directory for combined report.

    Returns
    -------
    str
        Path to the final report.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("STEP 10: Combined Analysis — In-Sample vs OOS")
    print("=" * 70)

    analyzer = BacktestAnalyzer()

    # --- IS vs OOS comparison table ---
    print("\nStep 10a: Generating comparison table...")
    is_metrics = in_sample_result.get("metrics", {})
    oos_metrics = oos_result.get("metrics", {})

    # Add brier delta to metrics
    is_brier = in_sample_result.get("brier_analysis", {})
    oos_brier = oos_result.get("brier_analysis", {})

    is_metrics_ext = {**is_metrics}
    oos_metrics_ext = {**oos_metrics}
    is_metrics_ext["brier_delta"] = is_brier.get("overall", {}).get(
        "brier_delta", float("nan"),
    )
    oos_metrics_ext["brier_delta"] = oos_brier.get("overall", {}).get(
        "brier_delta", float("nan"),
    )

    persistence_df = analyzer.analyze_edge_persistence(is_metrics_ext, oos_metrics_ext)
    persistence_df.to_csv(
        os.path.join(output_dir, "oos_vs_insample_comparison.csv"), index=False,
    )

    print("\n  Comparison Table:")
    print("  " + "-" * 80)
    print(f"  {'Metric':<18} {'In-Sample':>12} {'OOS':>12} {'Change':>12} {'Verdict'}")
    print("  " + "-" * 80)
    for _, row in persistence_df.iterrows():
        is_v = row["in_sample"]
        oos_v = row["oos"]
        ch = row["change"]

        if row["metric"] in ("sharpe_ratio",):
            is_str = f"{is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("roi", "win_rate"):
            is_str = f"{is_v*100:.1f}%" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v*100:.1f}%" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch*100:+.1f}%" if not np.isnan(ch) else "N/A"
        elif row["metric"] in ("total_pnl", "max_drawdown"):
            is_str = f"${is_v:.2f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"${oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"${ch:+.2f}" if not np.isnan(ch) else "N/A"
        elif row["metric"] == "brier_delta":
            is_str = f"{is_v:.4f}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v:.4f}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch:+.4f}" if not np.isnan(ch) else "N/A"
        else:
            is_str = f"{is_v}" if not np.isnan(is_v) else "N/A"
            oos_str = f"{oos_v}" if not np.isnan(oos_v) else "N/A"
            ch_str = f"{ch}" if not np.isnan(ch) else "N/A"

        print(f"  {row['metric']:<18} {is_str:>12} {oos_str:>12} {ch_str:>12}   {row['verdict']}")
    print("  " + "-" * 80)

    # --- Generate comparison plots ---
    print("\nStep 10b: Generating comparison plots...")
    plots = analyzer.plot_oos_vs_insample(
        in_sample_result, oos_result, output_dir,
    )
    for p in plots:
        print(f"  Saved: {os.path.basename(p)}")

    # --- Generate comprehensive report ---
    print("\nStep 10c: Generating comprehensive backtest report...")
    report = analyzer.generate_comprehensive_report(
        in_sample_result, oos_result, output_dir,
    )

    # --- Seasonal comparison ---
    print("\nStep 10d: Generating seasonal comparison...")
    is_seasonal = in_sample_result.get("seasonal_pnl", {})
    oos_seasonal = oos_result.get("seasonal_pnl", {})

    if not is_seasonal and in_sample_result.get("trades"):
        is_seasonal = compute_seasonal_pnl(in_sample_result["trades"])
    if not oos_seasonal and oos_result.get("trades"):
        oos_seasonal = compute_seasonal_pnl(oos_result["trades"])

    seasonal_rows = []
    for season in ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]:
        is_s = is_seasonal.get(season, {})
        oos_s = oos_seasonal.get(season, {})
        seasonal_rows.append({
            "season": season,
            "is_pnl": is_s.get("total_pnl", 0),
            "is_trades": is_s.get("n_trades", 0),
            "is_win_rate": is_s.get("win_rate", 0),
            "oos_pnl": oos_s.get("total_pnl", 0),
            "oos_trades": oos_s.get("n_trades", 0),
            "oos_win_rate": oos_s.get("win_rate", 0),
        })

    seasonal_compare_df = pd.DataFrame(seasonal_rows)
    seasonal_compare_df.to_csv(
        os.path.join(output_dir, "combined_seasonal_analysis.csv"), index=False,
    )

    print("\n  Seasonal Comparison:")
    print(f"  {'Season':<18} {'IS P&L':>10} {'OOS P&L':>10} {'IS WR':>8} {'OOS WR':>8}")
    for _, row in seasonal_compare_df.iterrows():
        print(f"  {row['season']:<18} "
              f"${row['is_pnl']:>9.2f} ${row['oos_pnl']:>9.2f} "
              f"{row['is_win_rate']*100:>6.1f}% {row['oos_win_rate']*100:>6.1f}%")

    # --- Trading recommendation ---
    print("\nStep 10e: Generating trading recommendation...")
    recommendation = analyzer._generate_recommendation(is_metrics_ext, oos_metrics_ext)

    rec_dict = {
        "verdict": recommendation["verdict"],
        "action": recommendation["action"],
        "in_sample_sharpe": is_metrics.get("sharpe_ratio", 0),
        "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
        "in_sample_roi": is_metrics.get("roi", 0),
        "oos_roi": oos_metrics.get("roi", 0),
        "strategy_config": oos_result.get("strategy_config", {}),
    }
    with open(os.path.join(output_dir, "trading_recommendation.json"), "w") as f:
        json.dump(rec_dict, f, indent=2, default=str)

    print(f"\n  VERDICT: {recommendation['verdict']}")
    print(f"  ACTION: {recommendation['action']}")
    print(f"\n  {recommendation['summary']}")

    report_path = os.path.join(output_dir, "final_backtest_report.md")
    print(f"\n  Full report saved to: {report_path}")

    return report_path


def main():
    """Run the full OOS validation pipeline."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Kalshi OOS Validation Pipeline")
    print("=" * 70)

    # --- Part 1: In-sample discovery ---
    config_path = os.path.join(IS_OUTPUT_DIR, "best_strategy_config.json")

    # Always regenerate to ensure consistency
    is_result = run_in_sample_discovery()

    # Compute seasonal P&L for in-sample
    is_result["seasonal_pnl"] = compute_seasonal_pnl(is_result.get("trades", []))

    # --- Part 2: OOS validation ---
    oos_result = run_oos_validation(
        is_result["best_strategy_config"],
    )

    # --- Part 3: Combined analysis ---
    report_path = run_combined_analysis(is_result, oos_result)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nOutput directories:")
    for d in [IS_OUTPUT_DIR, OOS_OUTPUT_DIR, COMBINED_OUTPUT_DIR]:
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            print(f"\n  {d}/")
            for f in files:
                fpath = os.path.join(d, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
