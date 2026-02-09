"""
Run Phase 3: Kalshi Trading Strategy Backtesting.

Loads existing model predictions (or generates synthetic data), creates
synthetic Kalshi market data, runs a comprehensive multi-strategy
backtest, generates reports and visualizations, and saves everything
to results/phase3/.

Usage:
    python run_phase3.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import config
from src.trading import (
    TradingStrategy,
    BacktestEngine,
    generate_strategy_grid,
    run_comprehensive_backtest,
    generate_synthetic_market_data,
    generate_phase3_report,
)


def generate_model_predictions(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic model predictions for backtesting.

    Creates realistic temperature predictions with Gaussian (mu, sigma)
    outputs that mimic what the synthesis model would produce.

    Parameters
    ----------
    n_days : int
        Number of days to simulate (default 365 = 1 year).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, model_mu, model_sigma, actual_tmax.
    """
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    day_of_year = np.arange(n_days)

    # Seasonal temperature pattern (NYC-like)
    seasonal_mean = 55.0 + 25.0 * np.sin(
        2 * np.pi * (day_of_year - 80) / 365.0
    )

    # Model predictions with some noise
    model_mu = seasonal_mean + rng.normal(0, 2.0, size=n_days)
    model_sigma = 3.0 + 1.0 * rng.uniform(0, 1, size=n_days)

    # Actual temperatures (model predictions + noise)
    actual_tmax = model_mu + model_sigma * rng.randn(n_days)

    return pd.DataFrame({
        "date": dates,
        "model_mu": model_mu,
        "model_sigma": model_sigma,
        "actual_tmax": actual_tmax,
    })


def try_load_real_predictions() -> pd.DataFrame:
    """Attempt to load real model predictions from results/.

    Looks for saved predictions from Phase 4 experiments. If not found,
    returns an empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Model predictions or empty DataFrame.
    """
    # Check for Phase 4 results
    phase4_dir = os.path.join(config.RESULTS_DIR, "phase4")
    if os.path.exists(phase4_dir):
        # Look for prediction files
        for fname in os.listdir(phase4_dir):
            if "predictions" in fname.lower() and fname.endswith(".csv"):
                path = os.path.join(phase4_dir, fname)
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                    required = ["date", "model_mu", "model_sigma", "actual_tmax"]
                    if all(c in df.columns for c in required):
                        print(f"Loaded real predictions from {path}")
                        return df
                except Exception:
                    continue

    return pd.DataFrame()


def main():
    """Run the full Phase 3 trading strategy evaluation pipeline."""
    print("=" * 70)
    print("NYC Temperature Prediction -- Phase 3: Trading Strategy Evaluation")
    print("=" * 70)
    print()

    output_dir = os.path.join(config.RESULTS_DIR, "phase3")
    os.makedirs(output_dir, exist_ok=True)

    # ----- Step 1: Load or generate model predictions -----
    print("Step 1: Loading model predictions...")
    predictions_df = try_load_real_predictions()

    if predictions_df.empty:
        print("  No real predictions found. Generating synthetic predictions.")
        predictions_df = generate_model_predictions(n_days=365, seed=42)
    else:
        print(f"  Loaded {len(predictions_df)} real predictions.")

    print(f"  Predictions: {len(predictions_df)} days")
    print(f"  Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
    print(f"  Mean mu: {predictions_df['model_mu'].mean():.1f} F")
    print(f"  Mean sigma: {predictions_df['model_sigma'].mean():.2f} F")
    print()

    # Save predictions
    predictions_df.to_csv(
        os.path.join(output_dir, "model_predictions.csv"), index=False
    )

    # ----- Step 2: Generate synthetic market data -----
    print("Step 2: Generating synthetic Kalshi market data...")
    market_data = generate_synthetic_market_data(
        predictions_df,
        n_buckets_per_day=3,
        noise_std=0.08,
        seed=42,
    )
    print(f"  Generated {len(market_data)} market data rows")
    print(f"  Unique dates: {market_data['date'].nunique()}")
    print(f"  Buckets per day: ~{len(market_data) / market_data['date'].nunique():.1f}")
    print(f"  Mean model prob: {market_data['model_prob'].mean():.3f}")
    print(f"  Mean market price: {market_data['market_price'].mean():.3f}")
    print()

    # Save market data
    market_data.to_csv(
        os.path.join(output_dir, "synthetic_market_data.csv"), index=False
    )

    # ----- Step 3: Generate strategy grid -----
    print("Step 3: Generating strategy permutations...")
    strategies = generate_strategy_grid(
        ev_thresholds=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10],
        sizing_methods=["fixed", "proportional", "fractional_kelly", "capped_kelly"],
        kelly_fractions=[0.10, 0.20, 0.25],
        fee_rates=[0.05, 0.07, 0.10],
        max_positions=[0.05, 0.10, 0.15],
        bankrolls=[5000, 10000, 25000],
    )
    print(f"  Generated {len(strategies)} strategy permutations")
    print()

    # ----- Step 4: Run comprehensive backtest -----
    print("Step 4: Running comprehensive backtest...")
    print("  (This may take a few minutes for large strategy grids)")
    print()

    results = run_comprehensive_backtest(
        market_data,
        output_dir=output_dir,
        strategies=strategies,
        max_strategies=500,
    )

    comparison_df = results["comparison_df"]
    all_results = results["all_results"]

    # Print summary
    trading_results = [r for r in all_results if r.n_trades > 0]
    print(f"\n  Total strategies: {len(all_results)}")
    print(f"  Strategies with trades: {len(trading_results)}")

    if not comparison_df.empty:
        profitable = comparison_df[comparison_df["total_pnl"] > 0]
        print(f"  Profitable strategies: {len(profitable)}")

        if not profitable.empty:
            best = comparison_df.loc[comparison_df["total_pnl"].idxmax()]
            print(f"\n  Best strategy by P&L:")
            print(f"    Name: {best['strategy_name']}")
            print(f"    Total P&L: ${best['total_pnl']:.2f}")
            print(f"    ROI: {best['roi'] * 100:.1f}%")
            print(f"    Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"    Win Rate: {best['win_rate'] * 100:.0f}%")
            print(f"    Trades: {best['n_trades']}")

    print()

    # ----- Step 5: Generate report -----
    print("Step 5: Generating Phase 3 report...")
    report = generate_phase3_report(results, output_dir=output_dir)

    # Print report
    print()
    print(report)

    # List output files
    print("\nOutput files:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname} ({size_kb:.1f} KB)")

    print()
    print("=" * 70)
    print("Phase 3 complete! Results saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
