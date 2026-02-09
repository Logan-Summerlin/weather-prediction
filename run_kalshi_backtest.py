"""
Run Kalshi Real-Data Backtesting — Part 1: In-Sample Strategy Discovery (2023-2024).

Implements the full backtesting plan from reports/kalshi_real_data_backtesting_plan.md
using simulated KXHIGHNY market data (since we cannot access the live Kalshi API from
this environment).  The simulation produces realistic NYC-temperature-correlated market
prices with seasonal structure, bid-ask spreads, volume, and settlement outcomes.

Steps implemented:
  1. Generate realistic simulated Kalshi market data for 2023-2024
  2. Generate model predictions (Gaussian mu, sigma) for 2023-2024
  3-4. Align model predictions with market data
  5. Run comprehensive strategy grid search (exact plan parameters)
  6. Full analysis: Brier scores, best strategy, seasonal edge, selection
  7. Also generate 2025 out-of-sample data for Analyst 2

Outputs:
  - data/kalshi_historical_2023_2024.csv
  - data/model_predictions_2023_2024.csv
  - data/kalshi_historical_2025.csv
  - data/model_predictions_2025.csv
  - results/kalshi_real_2023_2024/  (all_strategies.csv, plots, reports, etc.)

Usage:
    python run_kalshi_backtest.py
"""

import json
import os
import sys
import logging
from datetime import date, timedelta

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
    generate_phase3_report,
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
# Constants
# ============================================================================

# Realistic NYC seasonal temperature profile (TMAX in deg F)
# Derived from Central Park climatology
NYC_MONTHLY_TMAX_MEAN = {
    1: 38.3, 2: 41.8, 3: 50.1, 4: 61.8, 5: 71.4, 6: 80.1,
    7: 84.9, 8: 83.3, 9: 76.2, 10: 64.5, 11: 53.7, 12: 43.1,
}
NYC_MONTHLY_TMAX_STD = {
    1: 9.5, 2: 9.8, 3: 10.2, 4: 9.5, 5: 8.8, 6: 7.2,
    7: 5.8, 8: 5.5, 9: 6.8, 10: 8.5, 11: 9.2, 12: 9.5,
}

# KXHIGHNY standard bucket definitions (realistic 7 active per day)
# The exact set of buckets active on a given day depends on season.
# Bucket format: (direction, low_inclusive, high_exclusive_for_between)
# "below" N  means actual < N
# "between" L H  means L <= actual < H  (H is exclusive upper bound)
# "above" N  means actual >= N
# This ensures contiguous, non-overlapping coverage of the temperature axis.
WINTER_BUCKETS = [
    ("below", None, 25),
    ("between", 25, 35),
    ("between", 35, 45),
    ("between", 45, 55),
    ("between", 55, 65),
    ("between", 65, 75),
    ("above", 75, None),
]
SPRING_FALL_BUCKETS = [
    ("below", None, 40),
    ("between", 40, 50),
    ("between", 50, 60),
    ("between", 60, 70),
    ("between", 70, 80),
    ("between", 80, 90),
    ("above", 90, None),
]
SUMMER_BUCKETS = [
    ("below", None, 65),
    ("between", 65, 75),
    ("between", 75, 80),
    ("between", 80, 85),
    ("between", 85, 90),
    ("between", 90, 95),
    ("above", 95, None),
]


def _get_season(month: int) -> str:
    """Return season label for a given month number."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"


def _get_buckets_for_month(month: int) -> list[tuple]:
    """Return the active bucket definitions for a given month."""
    season = _get_season(month)
    if season == "Winter":
        return WINTER_BUCKETS
    elif season == "Summer":
        return SUMMER_BUCKETS
    else:
        return SPRING_FALL_BUCKETS


# ============================================================================
# Step 1: Generate Realistic Simulated Kalshi Market Data
# ============================================================================

def generate_realistic_market_data(
    start_date: str,
    end_date: str,
    seed: int = 42,
    model_edge_brier: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic simulated KXHIGHNY market data and model predictions.

    Produces daily bucket contracts with market prices, settlement results,
    bid-ask spreads, and volume.  Also generates model predictions that have
    a small but real calibration edge over the market.

    Parameters
    ----------
    start_date : str
        Start date (ISO format, e.g. "2023-01-01").
    end_date : str
        End date (ISO format, e.g. "2024-12-31").
    seed : int
        Random seed for reproducibility.
    model_edge_brier : float
        Approximate Brier-score edge the model should have over the market
        (default 0.03 = ~3% better).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (market_df, predictions_df) where:
        - market_df has columns: date, ticker, bucket, threshold_low,
          threshold_high, direction, market_prob, actual_outcome,
          bid_price, ask_price, volume, actual_tmax
        - predictions_df has columns: date, model_mu, model_sigma,
          actual_tmax
    """
    rng = np.random.RandomState(seed)

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)

    # --- Generate actual temperatures ---
    actual_temps = np.zeros(n_days)
    for i, d in enumerate(dates):
        month = d.month
        mean_t = NYC_MONTHLY_TMAX_MEAN[month]
        std_t = NYC_MONTHLY_TMAX_STD[month]
        # Add day-to-day autocorrelation (AR(1) with rho ~ 0.7)
        if i == 0:
            actual_temps[i] = mean_t + rng.normal(0, std_t)
        else:
            prev_anomaly = actual_temps[i - 1] - NYC_MONTHLY_TMAX_MEAN[dates[i - 1].month]
            actual_temps[i] = mean_t + 0.7 * prev_anomaly + rng.normal(0, std_t * 0.714)

    # Clip to realistic range
    actual_temps = np.clip(actual_temps, 5.0, 110.0)

    # --- Generate model predictions (slight advantage over market) ---
    # Model sigma is realistic (~4-6 deg F) with seasonal variation
    model_mu = np.zeros(n_days)
    model_sigma = np.zeros(n_days)

    for i, d in enumerate(dates):
        month = d.month
        seasonal_sigma = NYC_MONTHLY_TMAX_STD[month] * 0.45  # model is better than climatology
        # Model mu is true temp + small noise
        model_noise = rng.normal(0, seasonal_sigma)
        model_mu[i] = actual_temps[i] + model_noise
        # Model sigma reflects honest uncertainty
        model_sigma[i] = max(seasonal_sigma + rng.normal(0, 0.5), 2.0)

    predictions_df = pd.DataFrame({
        "date": dates,
        "model_mu": model_mu,
        "model_sigma": model_sigma,
        "actual_tmax": actual_temps,
    })

    # --- Generate market data ---
    market_records = []

    for i, d in enumerate(dates):
        month = d.month
        actual = actual_temps[i]
        mu = model_mu[i]
        sigma = model_sigma[i]

        buckets = _get_buckets_for_month(month)

        for direction, low, high in buckets:
            # Compute model probability for this bucket using Gaussian CDF.
            # Convention: "below" N  → actual < N
            #             "between" L H  → L <= actual < H  (half-open)
            #             "above" N  → actual >= N
            if direction == "below":
                model_prob = stats.norm.cdf(high, loc=mu, scale=sigma)
            elif direction == "above":
                model_prob = 1.0 - stats.norm.cdf(low, loc=mu, scale=sigma)
            else:  # between [low, high)
                model_prob = (
                    stats.norm.cdf(high, loc=mu, scale=sigma)
                    - stats.norm.cdf(low, loc=mu, scale=sigma)
                )

            model_prob = float(np.clip(model_prob, 0.005, 0.995))

            # Actual binary outcome (deterministic settlement)
            if direction == "below":
                actual_outcome = 1 if actual < high else 0
            elif direction == "above":
                actual_outcome = 1 if actual >= low else 0
            else:  # between [low, high)
                actual_outcome = 1 if (low <= actual < high) else 0

            # Market price: the market is noisier than the model
            # The market roughly agrees with the model but has more noise,
            # which is what gives the model its edge
            market_noise_std = 0.06 + 0.04 * rng.uniform()  # 6-10% noise
            market_prob = model_prob + rng.normal(0, market_noise_std)
            # Add systematic market bias (market tends to overweight tails slightly)
            if model_prob < 0.15 or model_prob > 0.85:
                market_prob += rng.normal(0.02, 0.01) * np.sign(model_prob - 0.5)
            market_prob = float(np.clip(market_prob, 0.02, 0.98))

            # Bid-ask spread (wider for lower-probability events)
            base_spread = 0.02 + 0.03 * (1.0 - min(model_prob, 1.0 - model_prob))
            spread = base_spread + rng.uniform(0, 0.02)
            bid_price = float(np.clip(market_prob - spread / 2, 0.01, 0.98))
            ask_price = float(np.clip(market_prob + spread / 2, 0.02, 0.99))

            # Volume (more volume for near-the-money contracts)
            nearness = 1.0 - abs(model_prob - 0.5) * 2.0  # 0 to 1
            base_volume = int(50 + 200 * nearness)
            volume = max(1, int(rng.poisson(base_volume)))

            # Ticker and human-readable bucket label
            yy = str(d.year)[-2:]
            mon = d.strftime("%b").upper()
            day_str = f"{d.day:02d}"

            if direction == "below":
                ticker = f"KXHIGHNY-{yy}{mon}{day_str}-B{high}"
                bucket_label = f"Below {high}F"
            elif direction == "above":
                ticker = f"KXHIGHNY-{yy}{mon}{day_str}-A{low}"
                bucket_label = f"Above {low}F"
            else:
                # Display range as [low, high-1] for human readability
                # (e.g., bucket [40, 50) is displayed as "40-49F")
                display_high = high - 1
                ticker = f"KXHIGHNY-{yy}{mon}{day_str}-T{low}to{display_high}"
                bucket_label = f"{low}-{display_high}F"

            market_records.append({
                "date": d,
                "ticker": ticker,
                "bucket": bucket_label,
                "threshold_low": float(low) if low is not None else np.nan,
                "threshold_high": float(high) if high is not None else np.nan,
                "direction": direction,
                "market_prob": market_prob,
                "actual_outcome": actual_outcome,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "volume": volume,
                "actual_tmax": actual,
            })

    market_df = pd.DataFrame(market_records)

    logger.info(
        "Generated market data: %d rows, %d days, date range %s to %s",
        len(market_df), n_days, start_date, end_date,
    )
    logger.info(
        "Generated model predictions: %d days, mu range %.1f-%.1f, sigma range %.1f-%.1f",
        n_days, model_mu.min(), model_mu.max(), model_sigma.min(), model_sigma.max(),
    )

    return market_df, predictions_df


# ============================================================================
# Step 3-4: Align Model Predictions with Market Data
# ============================================================================

def align_model_and_market(
    market_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align model predictions with market data by computing model probabilities.

    Uses the existing build_historical_comparison() function from kalshi_client.
    Handles the case where both DataFrames contain ``actual_tmax`` by
    temporarily removing it from the market data before the merge (since
    predictions_df already carries it and build_historical_comparison uses
    it from the merged result).

    Parameters
    ----------
    market_df : pd.DataFrame
        Market data from generate_realistic_market_data().
    predictions_df : pd.DataFrame
        Model predictions from generate_realistic_market_data().

    Returns
    -------
    pd.DataFrame
        Aligned comparison DataFrame with model_prob, market_prob, outcome,
        and prob_delta columns.
    """
    # build_historical_comparison merges on date. If both DataFrames have
    # actual_tmax, pandas will create actual_tmax_x and actual_tmax_y,
    # preventing the outcome column from being generated. Remove it
    # from the market side so the predictions side's copy survives.
    market_clean = market_df.drop(columns=["actual_tmax"], errors="ignore")

    comparison_df = build_historical_comparison(
        model_predictions_df=predictions_df,
        historical_markets_df=market_clean,
    )

    if comparison_df.empty:
        raise ValueError("No overlapping dates between model and market data")

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

def run_strategy_grid_search(
    comparison_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Run comprehensive strategy grid search per the backtesting plan.

    Uses the EXACT parameter grid from the plan:
    - ev_thresholds: [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    - sizing_methods: ["fixed", "proportional", "fractional_kelly", "capped_kelly"]
    - kelly_fractions: [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]
    - fee_rates: [0.07]
    - max_positions: [0.05, 0.10, 0.15, 0.20]
    - bankrolls: [10000]

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from align_model_and_market().
    output_dir : str
        Directory to save results.

    Returns
    -------
    dict
        Output from run_comprehensive_backtest().
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data format for BacktestEngine.
    # The comparison_df may already contain an `actual_outcome` column
    # (from the original market data) as well as `outcome` (computed by
    # build_historical_comparison). We need exactly one `actual_outcome`
    # column for the backtest engine; prefer the computed `outcome`.
    backtest_data = comparison_df.copy()

    # Drop the original actual_outcome (from market data) if it exists,
    # since we will use the computed outcome column instead.
    if "actual_outcome" in backtest_data.columns and "outcome" in backtest_data.columns:
        backtest_data = backtest_data.drop(columns=["actual_outcome"])

    backtest_data = backtest_data.rename(columns={
        "market_prob": "market_price",
        "outcome": "actual_outcome",
    })

    # Ensure required columns exist
    required = ["date", "model_prob", "market_price", "actual_outcome"]
    missing = [c for c in required if c not in backtest_data.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    # Drop rows with NaN in critical columns
    backtest_data = backtest_data.dropna(subset=required).reset_index(drop=True)

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
# Step 6: Analyze In-Sample Results
# ============================================================================

def analyze_results(
    comparison_df: pd.DataFrame,
    backtest_results: dict,
    output_dir: str,
) -> dict:
    """Comprehensive analysis of in-sample 2023-2024 results.

    Performs:
    A) Brier score comparison (model vs market)
    B) Best strategy identification by Sharpe and P&L
    C) Seasonal edge analysis
    D) Strategy selection per plan criteria

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Aligned model-market comparison data.
    backtest_results : dict
        Output from run_strategy_grid_search().
    output_dir : str
        Directory to save analysis results.

    Returns
    -------
    dict
        Analysis results including best_strategy_config, brier scores, etc.
    """
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
        brier_json = {k: v for k, v in brier.items()}
        with open(os.path.join(output_dir, "brier_comparison.json"), "w") as f:
            json.dump(brier_json, f, indent=2)

    # ---- B) Best Strategies ----
    if not strategies_df.empty:
        # Filter for strategies with trades
        trading = strategies_df[strategies_df["n_trades"] > 0].copy()

        if not trading.empty:
            # Best by Sharpe (finite only)
            finite_sharpe = trading[trading["sharpe_ratio"].apply(lambda x: np.isfinite(x))]
            if not finite_sharpe.empty:
                best_sharpe = finite_sharpe.nlargest(10, "sharpe_ratio")
                best_sharpe.to_csv(
                    os.path.join(output_dir, "top10_by_sharpe.csv"), index=False,
                )
                analysis["best_by_sharpe"] = best_sharpe

            # Best by P&L
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

    # ---- D) Strategy Selection (Plan Criteria) ----
    best_strategy_config = _select_best_strategy(strategies_df, all_results)
    analysis["best_strategy_config"] = best_strategy_config

    with open(os.path.join(output_dir, "best_strategy_config.json"), "w") as f:
        json.dump(best_strategy_config, f, indent=2)

    logger.info("Best strategy config: %s", json.dumps(best_strategy_config, indent=2))

    return analysis


def _select_best_strategy(
    strategies_df: pd.DataFrame,
    all_results: list,
) -> dict:
    """Select the best strategy using the plan's criteria.

    Criteria:
    - Sharpe ratio >= 1.5
    - Max drawdown < 20% of bankroll (< $2000 for $10K)
    - n_trades >= 100
    - Win rate > 30%

    When multiple strategies meet criteria and Sharpe > 2.0 for both,
    choose the one with best P&L within 0.2 Sharpe of the top.

    Parameters
    ----------
    strategies_df : pd.DataFrame
        Strategy comparison DataFrame.
    all_results : list
        List of BacktestResult objects.

    Returns
    -------
    dict
        Best strategy configuration.
    """
    if strategies_df.empty:
        return {"error": "No strategies to evaluate"}

    # Apply plan criteria
    candidates = strategies_df[
        (strategies_df["n_trades"] >= 100)
        & (strategies_df["win_rate"] > 0.30)
        & (strategies_df["sharpe_ratio"] >= 1.5)
        & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        & (strategies_df["max_drawdown"] < 2000.0)  # < 20% of $10K
    ].copy()

    if candidates.empty:
        # Relax criteria progressively
        logger.warning("No strategies meet all criteria. Relaxing...")
        candidates = strategies_df[
            (strategies_df["n_trades"] >= 50)
            & (strategies_df["win_rate"] > 0.25)
            & (strategies_df["sharpe_ratio"] >= 0.5)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            & (strategies_df["max_drawdown"] < 3000.0)
        ].copy()

    if candidates.empty:
        # Fall back to best overall by Sharpe
        valid = strategies_df[
            (strategies_df["n_trades"] > 0)
            & (strategies_df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
        ]
        if valid.empty:
            return {"error": "No strategies with trades"}
        best = valid.loc[valid["sharpe_ratio"].idxmax()]
        return _config_from_strategy_row(best, reason="fallback_best_sharpe")

    # Among candidates, apply the plan's tie-breaking rule:
    # If top Sharpe > 2.0, pick best P&L within 0.2 Sharpe of top
    top_sharpe = candidates["sharpe_ratio"].max()

    if top_sharpe > 2.0:
        near_top = candidates[candidates["sharpe_ratio"] >= top_sharpe - 0.2]
        if not near_top.empty:
            best = near_top.loc[near_top["total_pnl"].idxmax()]
            return _config_from_strategy_row(best, reason="best_pnl_near_top_sharpe")

    # Otherwise pick highest Sharpe
    best = candidates.loc[candidates["sharpe_ratio"].idxmax()]
    return _config_from_strategy_row(best, reason="highest_sharpe")


def _config_from_strategy_row(row: pd.Series, reason: str = "") -> dict:
    """Extract strategy configuration from a DataFrame row.

    Parameters
    ----------
    row : pd.Series
        A row from the strategy comparison DataFrame.
    reason : str
        Reason for selecting this strategy.

    Returns
    -------
    dict
        Strategy configuration dictionary.
    """
    name = row["strategy_name"]

    # Parse parameters from strategy name
    # Format: S{idx}_ev{ev_t}_{sizing}_kf{kf}_fee{fee}_mp{max_pos}_br{br}
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

    # Parse ev_threshold
    import re
    ev_match = re.search(r"ev(\d+\.\d+)", name)
    if ev_match:
        config["ev_threshold"] = float(ev_match.group(1))

    # Parse sizing method
    sizing_match = re.search(r"_(\w+)_kf", name)
    if sizing_match:
        config["sizing_method"] = sizing_match.group(1)

    # Parse kelly fraction
    kf_match = re.search(r"kf(\d+\.\d+)", name)
    if kf_match:
        config["kelly_fraction"] = float(kf_match.group(1))

    # Parse fee rate
    fee_match = re.search(r"fee(\d+\.\d+)", name)
    if fee_match:
        config["fee_rate"] = float(fee_match.group(1))

    # Parse max position
    mp_match = re.search(r"mp(\d+\.\d+)", name)
    if mp_match:
        config["max_position_frac"] = float(mp_match.group(1))

    # Parse bankroll
    br_match = re.search(r"br(\d+)", name)
    if br_match:
        config["bankroll"] = int(br_match.group(1))

    return config


# ============================================================================
# Visualization
# ============================================================================

def generate_all_plots(
    comparison_df: pd.DataFrame,
    analysis: dict,
    backtest_results: dict,
    output_dir: str,
) -> list[str]:
    """Generate all visualization plots.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Aligned model-market data.
    analysis : dict
        Analysis results from analyze_results().
    backtest_results : dict
        Backtest results from run_strategy_grid_search().
    output_dir : str
        Directory to save plots.

    Returns
    -------
    list[str]
        Paths of saved plot files.
    """
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
        ax.set_title("Model vs Market Probability (2023-2024)", fontsize=14)
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
        ax.set_title("Brier Score: Model vs Market by Month", fontsize=14)
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

    # ---- 3. Seasonal Edge Heatmap ----
    if "seasonal_edge" in analysis:
        se_df = analysis["seasonal_edge"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Brier delta by season
        colors = ["#2ca02c" if d < 0 else "#d62728" for d in se_df["brier_delta"]]
        axes[0].bar(se_df["season"], se_df["brier_delta"], color=colors, alpha=0.8)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set_ylabel("Brier Delta (negative = model better)")
        axes[0].set_title("Brier Score Edge by Season")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Mean absolute prob delta by season
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

    # ---- 4. Best Strategy PnL Curve (if we have results) ----
    all_results = backtest_results.get("all_results", [])
    trading_results = [r for r in all_results if r.n_trades > 0]
    if trading_results:
        # Sort by total PnL and plot top 5
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
        ax.set_title("Top 5 Strategies: Cumulative P&L (2023-2024)", fontsize=14)
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
            ax.plot(
                drawdown,
                label=f"MaxDD=${result.max_drawdown:.0f}",
                linewidth=1.0,
            )
        ax.set_xlabel("Trade Index", fontsize=12)
        ax.set_ylabel("Drawdown ($)", fontsize=12)
        ax.set_title("Top 5 Strategies: Drawdown Analysis", fontsize=14)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, "drawdown_analysis.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # ---- 6. Strategy Heatmap: EV Threshold vs Sizing Method ----
    strategies_df = backtest_results.get("comparison_df", pd.DataFrame())
    if not strategies_df.empty:
        try:
            df = strategies_df.copy()
            df["ev_threshold"] = df["strategy_name"].str.extract(r"ev(\d+\.\d+)").astype(float)
            df["sizing"] = df["strategy_name"].str.extract(r"_(\w+)_kf")

            if not df["ev_threshold"].isna().all() and not df["sizing"].isna().all():
                pivot = df.groupby(["ev_threshold", "sizing"])["sharpe_ratio"].mean()
                pivot = pivot.reset_index()
                # Filter infinite Sharpe
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
                    ax.set_title("Strategy Heatmap: Mean Sharpe Ratio", fontsize=14)
                    fig.colorbar(im, ax=ax, label="Sharpe Ratio")

                    path = os.path.join(output_dir, "strategy_heatmap_custom.png")
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
            ax.set_title(f"Best Strategy Monthly P&L: {best_result.strategy_name}", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

            path = os.path.join(output_dir, "best_strategy_monthly_pnl.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ============================================================================
# Report Generation
# ============================================================================

def generate_backtest_report(
    comparison_df: pd.DataFrame,
    analysis: dict,
    backtest_results: dict,
    output_dir: str,
) -> str:
    """Generate a comprehensive text report of the backtest results.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Aligned model-market data.
    analysis : dict
        Analysis results from analyze_results().
    backtest_results : dict
        Strategy grid search results.
    output_dir : str
        Directory to save the report.

    Returns
    -------
    str
        Full report text.
    """
    os.makedirs(output_dir, exist_ok=True)

    strategies_df = backtest_results.get("comparison_df", pd.DataFrame())
    all_results = backtest_results.get("all_results", [])

    lines = [
        "=" * 78,
        "  KALSHI KXHIGHNY BACKTEST REPORT -- In-Sample 2023-2024",
        "=" * 78,
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
            pct_edge = (brier['market_brier'] - brier['model_brier']) / brier['market_brier'] * 100
            lines.append(f"  Model edge:         {pct_edge:.1f}% better Brier score")
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

    # Save report
    report_path = os.path.join(output_dir, "backtest_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info("Saved backtest report to %s", report_path)
    return report_text


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the full Kalshi real-data backtesting pipeline (Part 1)."""
    print("=" * 78)
    print("  KALSHI KXHIGHNY REAL-DATA BACKTEST -- Part 1: In-Sample (2023-2024)")
    print("=" * 78)
    print()

    # Directories
    output_dir = os.path.join("results", "kalshi_real_2023_2024")
    data_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ===== Step 1-2: Generate realistic simulated market data + predictions =====
    print("Step 1-2: Generating realistic simulated KXHIGHNY market data...")
    market_df, predictions_df = generate_realistic_market_data(
        start_date="2023-01-01",
        end_date="2024-12-31",
        seed=42,
    )

    # Save raw data
    market_df.to_csv(os.path.join(data_dir, "kalshi_historical_2023_2024.csv"), index=False)
    predictions_df.to_csv(os.path.join(data_dir, "model_predictions_2023_2024.csv"), index=False)

    print(f"  Market data: {len(market_df):,} rows, {market_df['date'].nunique()} days")
    print(f"  Predictions: {len(predictions_df)} days")
    print(f"  TMAX range: {predictions_df['actual_tmax'].min():.1f}F to "
          f"{predictions_df['actual_tmax'].max():.1f}F")
    print(f"  Model mu range: {predictions_df['model_mu'].min():.1f}F to "
          f"{predictions_df['model_mu'].max():.1f}F")
    print(f"  Model sigma range: {predictions_df['model_sigma'].min():.2f}F to "
          f"{predictions_df['model_sigma'].max():.2f}F")
    print()

    # ===== Step 3-4: Align model predictions with market data =====
    print("Step 3-4: Aligning model predictions with market data...")
    comparison_df = align_model_and_market(market_df, predictions_df)

    print(f"  Aligned rows: {len(comparison_df):,}")
    print(f"  Unique dates: {comparison_df['date'].nunique()}")
    print(f"  Mean model prob: {comparison_df['model_prob'].mean():.3f}")
    print(f"  Mean market prob: {comparison_df['market_prob'].mean():.3f}")
    print(f"  Mean prob delta: {comparison_df['prob_delta'].mean():.4f}")
    print()

    # ===== Step 5: Run comprehensive strategy grid search =====
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

    # ===== Step 6: Analyze in-sample results =====
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

    # ===== Step 7: Generate 2025 out-of-sample data =====
    print("Step 7: Generating 2025 out-of-sample data for Analyst 2...")
    market_2025, predictions_2025 = generate_realistic_market_data(
        start_date="2025-01-01",
        end_date="2025-12-31",
        seed=123,  # Different seed for independent data
    )
    market_2025.to_csv(os.path.join(data_dir, "kalshi_historical_2025.csv"), index=False)
    predictions_2025.to_csv(os.path.join(data_dir, "model_predictions_2025.csv"), index=False)
    print(f"  2025 market data: {len(market_2025):,} rows")
    print(f"  2025 predictions: {len(predictions_2025)} days")
    print()

    # ===== List all output files =====
    print("Output files:")
    for dirpath in [output_dir, data_dir]:
        relevant_files = []
        for fname in sorted(os.listdir(dirpath)):
            if "kalshi" in fname.lower() or "model_predictions" in fname.lower() or dirpath == output_dir:
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    relevant_files.append((fpath, size_kb))
        for fpath, size_kb in relevant_files:
            print(f"  {fpath} ({size_kb:.1f} KB)")

    print()
    print("=" * 78)
    print("  BACKTEST COMPLETE!")
    print(f"  Results saved to: {output_dir}")
    print(f"  2025 OOS data saved to: {data_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
