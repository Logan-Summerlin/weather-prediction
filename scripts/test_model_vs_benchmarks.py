#!/usr/bin/env python3
"""
test_model_vs_benchmarks.py

Comprehensive comparison of our NN prediction model against:
  1. Kalshi pre-settlement market prices (real prediction market consensus before settlement)
  2. NWS/MOS forecast probability distribution (operational weather forecast)

Computes Brier scores, log scores, calibration metrics, and trading simulations.

Outputs to results/prediction_market_benchmark/
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# Constants
# ==============================================================================
FEE_RATE = 0.07
PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999
TRADING_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]
OUTPUT_DIR = Path("results/prediction_market_benchmark")
N_CAL_BINS = 10

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}


# ==============================================================================
# Data Loading & Merging
# ==============================================================================
def load_all_data():
    """Load all data sources and return as dict."""
    print("Loading data...")
    pre = pd.read_csv("data/kalshi_presettlement.csv")
    s23 = pd.read_csv("data/real_kalshi_2023_2024.csv")
    s25 = pd.read_csv("data/real_kalshi_2025.csv")
    settled = pd.concat([s23, s25], ignore_index=True)

    model_is = pd.read_csv("data/max_train_nn_predictions_is.csv")
    model_oos = pd.read_csv("data/max_train_nn_predictions_oos.csv")
    model = pd.concat([model_is, model_oos], ignore_index=True)

    nws = pd.read_csv("results/prediction_market_benchmark/nws_probability_forecasts.csv")
    cp_tmax = pd.read_csv("data/central_park_tmax_full_history.csv")

    print(f"  Pre-settlement: {len(pre)} rows, {pre['date'].nunique()} dates")
    print(f"  Settled:        {len(settled)} rows, {settled['date'].nunique()} dates")
    print(f"  Model preds:    {len(model)} rows")
    print(f"  NWS forecasts:  {len(nws)} rows")
    return pre, settled, model, nws, cp_tmax


def build_merged_dataset(pre, settled, model, nws):
    """
    Merge pre-settlement prices with settled contract definitions (direction,
    thresholds, actual_outcome) and model/NWS forecasts.

    The settled data provides ground-truth contract definitions (2F between
    buckets) and verified actual outcomes. Pre-settlement data provides the
    market prices before settlement.
    """
    print("\nMerging datasets...")

    # Step 1: Join pre-settlement to settled on date+ticker
    # Settled gives correct direction, thresholds, actual_tmax, actual_outcome
    merged = pre.merge(
        settled[["date", "ticker", "direction", "threshold_low", "threshold_high",
                 "actual_outcome", "actual_tmax", "market_prob"]],
        on=["date", "ticker"],
        suffixes=("_pre", ""),
        how="inner"
    )

    # Use settled data's direction and thresholds as ground truth
    # Rename pre-settlement prob for clarity
    merged = merged.rename(columns={
        "presettlement_prob": "presettlement_prob",
        "market_prob": "settled_market_prob"
    })

    # Drop rows with NaN presettlement price (no market data available)
    n_before = len(merged)
    merged = merged.dropna(subset=["presettlement_prob"])
    n_dropped = n_before - len(merged)
    print(f"  Matched pre-settlement to settled: {n_before} rows")
    print(f"  Dropped {n_dropped} rows with missing presettlement_prob")

    # Step 2: Add model predictions
    merged = merged.merge(
        model[["date", "model_mu", "model_sigma"]],
        on="date",
        how="inner"
    )

    # Step 3: Add NWS forecasts
    merged = merged.merge(
        nws[["date", "nws_mu", "nws_sigma"]],
        on="date",
        how="inner"
    )

    # Step 4: Add period and season
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["period"] = np.where(merged["date_dt"].dt.year <= 2024, "IS", "OOS")
    merged["season"] = merged["date_dt"].dt.month.map(SEASON_MAP)
    merged["month"] = merged["date_dt"].dt.month

    print(f"  Final merged dataset: {len(merged)} rows, {merged['date'].nunique()} dates")
    print(f"  Period split: IS={len(merged[merged['period']=='IS'])}, OOS={len(merged[merged['period']=='OOS'])}")
    return merged


# ==============================================================================
# Probability Computation
# ==============================================================================
def compute_bucket_probs(df, mu_col, sigma_col):
    """Compute P(TMAX in bucket) from N(mu, sigma), vectorized."""
    probs = pd.Series(np.nan, index=df.index, dtype=float)

    below = df["direction"] == "below"
    above = df["direction"] == "above"
    between = df["direction"] == "between"

    # Below: P(X < threshold_high)
    if below.any():
        probs.loc[below] = norm.cdf(
            df.loc[below, "threshold_high"].values,
            df.loc[below, mu_col].values,
            df.loc[below, sigma_col].values
        )

    # Above: P(X >= threshold_low)
    if above.any():
        probs.loc[above] = 1.0 - norm.cdf(
            df.loc[above, "threshold_low"].values,
            df.loc[above, mu_col].values,
            df.loc[above, sigma_col].values
        )

    # Between: P(threshold_low <= X < threshold_high)
    if between.any():
        probs.loc[between] = (
            norm.cdf(
                df.loc[between, "threshold_high"].values,
                df.loc[between, mu_col].values,
                df.loc[between, sigma_col].values
            ) -
            norm.cdf(
                df.loc[between, "threshold_low"].values,
                df.loc[between, mu_col].values,
                df.loc[between, sigma_col].values
            )
        )

    return probs


def add_all_probabilities(df):
    """Compute model_prob, nws_prob and clip all probabilities."""
    print("\nComputing probabilities...")

    df["model_prob"] = compute_bucket_probs(df, "model_mu", "model_sigma")
    df["nws_prob"] = compute_bucket_probs(df, "nws_mu", "nws_sigma")

    # Clip all probabilities for numerical stability
    for col in ["model_prob", "nws_prob", "presettlement_prob", "settled_market_prob"]:
        df[col] = df[col].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)

    print(f"  Model prob range: [{df['model_prob'].min():.4f}, {df['model_prob'].max():.4f}]")
    print(f"  NWS prob range:   [{df['nws_prob'].min():.4f}, {df['nws_prob'].max():.4f}]")
    print(f"  Pre-settlement range: [{df['presettlement_prob'].min():.4f}, {df['presettlement_prob'].max():.4f}]")
    return df


# ==============================================================================
# Scoring Metrics
# ==============================================================================
def brier_score(probs, outcomes):
    """Brier score: mean squared error of probability vs binary outcome."""
    return np.mean((probs - outcomes) ** 2)


def log_score(probs, outcomes):
    """Log score (negative log-likelihood): lower = better."""
    p = np.clip(probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
    return -np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))


def compute_scores_for_slice(df, label):
    """Compute Brier and log scores for all three sources on a data slice."""
    outcomes = df["actual_outcome"].values.astype(float)
    n = len(df)
    results = []
    for source, col in [("Model", "model_prob"),
                         ("Kalshi_PreSettlement", "presettlement_prob"),
                         ("NWS", "nws_prob"),
                         ("Kalshi_Settled", "settled_market_prob")]:
        bs = brier_score(df[col].values, outcomes)
        ls = log_score(df[col].values, outcomes)
        results.append({
            "slice": label,
            "source": source,
            "brier_score": bs,
            "log_score": ls,
            "n_buckets": n
        })
    return results


def compute_all_scores(df):
    """Compute Brier/log scores across all slicing dimensions."""
    print("\nComputing scoring metrics...")
    all_results = []

    # Overall
    all_results.extend(compute_scores_for_slice(df, "Overall"))

    # By period
    for period in ["IS", "OOS"]:
        mask = df["period"] == period
        if mask.sum() > 0:
            all_results.extend(compute_scores_for_slice(df[mask], f"Period: {period}"))

    # By season
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        mask = df["season"] == season
        if mask.sum() > 0:
            all_results.extend(compute_scores_for_slice(df[mask], f"Season: {season}"))

    # By period x season
    for period in ["IS", "OOS"]:
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            mask = (df["period"] == period) & (df["season"] == season)
            if mask.sum() > 0:
                all_results.extend(
                    compute_scores_for_slice(df[mask], f"{period}_{season}")
                )

    # By direction type
    for direction in ["below", "between", "above"]:
        mask = df["direction"] == direction
        if mask.sum() > 0:
            all_results.extend(
                compute_scores_for_slice(df[mask], f"Direction: {direction}")
            )

    scores_df = pd.DataFrame(all_results)
    return scores_df


# ==============================================================================
# Calibration Analysis
# ==============================================================================
def calibration_data(probs, outcomes, n_bins=N_CAL_BINS):
    """Compute calibration bin data for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = np.zeros(n_bins)
    bin_mean_pred = np.zeros(n_bins)
    bin_mean_outcome = np.zeros(n_bins)

    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            bin_mean_pred[i] = probs[mask].mean()
            bin_mean_outcome[i] = outcomes[mask].mean()
        else:
            bin_mean_pred[i] = bin_centers[i]
            bin_mean_outcome[i] = np.nan

    return bin_centers, bin_mean_pred, bin_mean_outcome, bin_counts


def expected_calibration_error(probs, outcomes, n_bins=N_CAL_BINS):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            avg_pred = probs[mask].mean()
            avg_outcome = outcomes[mask].mean()
            ece += (count / n) * abs(avg_pred - avg_outcome)
    return ece


def compute_calibration(df):
    """Compute calibration data for all sources."""
    print("Computing calibration metrics...")
    outcomes = df["actual_outcome"].values.astype(float)
    cal_rows = []

    for source, col in [("Model", "model_prob"),
                         ("Kalshi_PreSettlement", "presettlement_prob"),
                         ("NWS", "nws_prob"),
                         ("Kalshi_Settled", "settled_market_prob")]:
        probs = df[col].values
        centers, mean_pred, mean_outcome, counts = calibration_data(probs, outcomes)
        ece = expected_calibration_error(probs, outcomes)

        for i in range(len(centers)):
            cal_rows.append({
                "source": source,
                "bin_center": centers[i],
                "mean_predicted": mean_pred[i],
                "mean_observed": mean_outcome[i],
                "count": int(counts[i]),
                "ece": ece
            })

    return pd.DataFrame(cal_rows)


# ==============================================================================
# Trading Simulation
# ==============================================================================
def run_trading_sim(df, signal_col, market_col, threshold, label):
    """
    Simulate trading: buy YES when signal > market + threshold,
    buy NO when signal < market - threshold.

    Cost = market price for YES, (1 - market price) for NO.
    Payout = $1 if correct, $0 if wrong. 7% fee on winnings.
    """
    signal = df[signal_col].values
    market = df[market_col].values
    outcome = df["actual_outcome"].values.astype(float)

    edge = signal - market

    # Buy YES: signal thinks more likely than market
    buy_yes = edge > threshold
    # Buy NO: signal thinks less likely than market
    buy_no = edge < -threshold

    total_trades = buy_yes.sum() + buy_no.sum()
    if total_trades == 0:
        return {
            "signal": label,
            "market": "Kalshi_PreSettlement",
            "threshold": threshold,
            "n_trades": 0,
            "n_yes_trades": 0,
            "n_no_trades": 0,
            "total_cost": 0,
            "gross_payout": 0,
            "fees": 0,
            "net_pnl": 0,
            "roi_pct": 0,
            "win_rate": 0,
            "avg_edge": 0,
            "sharpe": 0
        }

    # YES trades
    yes_cost = market[buy_yes]
    yes_wins = outcome[buy_yes] == 1
    yes_payout = np.where(yes_wins, 1.0, 0.0)
    yes_fees = yes_payout * FEE_RATE
    yes_net = yes_payout - yes_fees - yes_cost

    # NO trades
    no_cost = 1.0 - market[buy_no]
    no_wins = outcome[buy_no] == 0
    no_payout = np.where(no_wins, 1.0, 0.0)
    no_fees = no_payout * FEE_RATE
    no_net = no_payout - no_fees - no_cost

    all_net = np.concatenate([yes_net, no_net])
    all_cost = np.concatenate([yes_cost, no_cost])
    all_wins = np.concatenate([yes_wins, no_wins])

    total_cost = all_cost.sum()
    gross_payout = np.concatenate([yes_payout, no_payout]).sum()
    fees = np.concatenate([yes_fees, no_fees]).sum()
    net_pnl = all_net.sum()
    roi = (net_pnl / total_cost * 100) if total_cost > 0 else 0
    win_rate = all_wins.mean() if len(all_wins) > 0 else 0
    avg_edge = np.abs(edge[buy_yes | buy_no]).mean()
    sharpe = (all_net.mean() / all_net.std()) if all_net.std() > 0 else 0

    return {
        "signal": label,
        "market": "Kalshi_PreSettlement",
        "threshold": threshold,
        "n_trades": int(total_trades),
        "n_yes_trades": int(buy_yes.sum()),
        "n_no_trades": int(buy_no.sum()),
        "total_cost": round(total_cost, 2),
        "gross_payout": round(gross_payout, 2),
        "fees": round(fees, 2),
        "net_pnl": round(net_pnl, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(win_rate, 4),
        "avg_edge": round(avg_edge, 4),
        "sharpe": round(sharpe, 4)
    }


def run_all_trading_sims(df):
    """Run trading simulations for Model and NWS vs pre-settlement market."""
    print("\nRunning trading simulations...")
    results = []

    for threshold in TRADING_THRESHOLDS:
        # Model vs pre-settlement market
        for period_label, period_mask in [("All", df.index == df.index),
                                           ("IS", df["period"] == "IS"),
                                           ("OOS", df["period"] == "OOS")]:
            sub = df[period_mask]
            if len(sub) == 0:
                continue

            # Model signal
            r = run_trading_sim(sub, "model_prob", "presettlement_prob",
                                threshold, f"Model_{period_label}")
            results.append(r)

            # NWS signal
            r = run_trading_sim(sub, "nws_prob", "presettlement_prob",
                                threshold, f"NWS_{period_label}")
            results.append(r)

    return pd.DataFrame(results)


# ==============================================================================
# Report Generation
# ==============================================================================
def generate_report(df, scores_df, cal_df, trading_df):
    """Generate comprehensive markdown report."""
    print("\nGenerating report...")

    lines = []
    lines.append("# Model vs Benchmarks: Comprehensive Comparison Report")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares our neural network temperature prediction model against two benchmarks:")
    lines.append("1. **Kalshi pre-settlement market prices** - Real prediction market consensus captured before settlement")
    lines.append("2. **NWS/MOS forecast** - National Weather Service operational forecast distribution")
    lines.append("")
    lines.append(f"- Total bucket-level observations: **{len(df):,}**")
    lines.append(f"- Unique dates: **{df['date'].nunique()}**")
    lines.append(f"- Date range: {df['date'].min()} to {df['date'].max()}")
    lines.append(f"- IS period (2023-2024): {len(df[df['period']=='IS']):,} observations")
    lines.append(f"- OOS period (2025): {len(df[df['period']=='OOS']):,} observations")
    lines.append("")

    # --- Brier Scores ---
    lines.append("## 1. Brier Score Comparison (lower = better)")
    lines.append("")

    # Overall table
    overall = scores_df[scores_df["slice"] == "Overall"].copy()
    overall = overall.sort_values("brier_score")
    lines.append("### Overall")
    lines.append("")
    lines.append("| Source | Brier Score | Log Score | N |")
    lines.append("|--------|-------------|-----------|---|")
    for _, row in overall.iterrows():
        lines.append(f"| {row['source']} | {row['brier_score']:.4f} | {row['log_score']:.4f} | {row['n_buckets']:,} |")
    lines.append("")

    # By period
    lines.append("### By Period")
    lines.append("")
    lines.append("| Period | Source | Brier Score | Log Score | N |")
    lines.append("|--------|--------|-------------|-----------|---|")
    for period in ["IS", "OOS"]:
        period_data = scores_df[scores_df["slice"] == f"Period: {period}"].sort_values("brier_score")
        for _, row in period_data.iterrows():
            lines.append(f"| {period} | {row['source']} | {row['brier_score']:.4f} | {row['log_score']:.4f} | {row['n_buckets']:,} |")
    lines.append("")

    # By season
    lines.append("### By Season")
    lines.append("")
    lines.append("| Season | Source | Brier Score | Log Score | N |")
    lines.append("|--------|--------|-------------|-----------|---|")
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        season_data = scores_df[scores_df["slice"] == f"Season: {season}"].sort_values("brier_score")
        for _, row in season_data.iterrows():
            lines.append(f"| {season} | {row['source']} | {row['brier_score']:.4f} | {row['log_score']:.4f} | {row['n_buckets']:,} |")
    lines.append("")

    # By direction
    lines.append("### By Bucket Direction")
    lines.append("")
    lines.append("| Direction | Source | Brier Score | Log Score | N |")
    lines.append("|-----------|--------|-------------|-----------|---|")
    for direction in ["below", "between", "above"]:
        dir_data = scores_df[scores_df["slice"] == f"Direction: {direction}"].sort_values("brier_score")
        for _, row in dir_data.iterrows():
            lines.append(f"| {direction} | {row['source']} | {row['brier_score']:.4f} | {row['log_score']:.4f} | {row['n_buckets']:,} |")
    lines.append("")

    # --- Calibration ---
    lines.append("## 2. Calibration Analysis")
    lines.append("")

    ece_data = cal_df.groupby("source")["ece"].first().sort_values()
    lines.append("### Expected Calibration Error (ECE)")
    lines.append("")
    lines.append("| Source | ECE |")
    lines.append("|--------|-----|")
    for source, ece in ece_data.items():
        lines.append(f"| {source} | {ece:.4f} |")
    lines.append("")

    # Reliability diagram data
    lines.append("### Reliability Diagram Data (10 bins)")
    lines.append("")
    for source in ["Model", "Kalshi_PreSettlement", "NWS", "Kalshi_Settled"]:
        source_cal = cal_df[cal_df["source"] == source].copy()
        lines.append(f"**{source}**")
        lines.append("")
        lines.append("| Bin | Mean Predicted | Mean Observed | Count |")
        lines.append("|-----|---------------|---------------|-------|")
        for _, row in source_cal.iterrows():
            obs = f"{row['mean_observed']:.3f}" if not np.isnan(row["mean_observed"]) else "N/A"
            lines.append(f"| {row['bin_center']:.2f} | {row['mean_predicted']:.3f} | {obs} | {int(row['count'])} |")
        lines.append("")

    # --- Trading Simulation ---
    lines.append("## 3. Trading Simulation: Model vs Pre-Settlement Market")
    lines.append("")
    lines.append(f"Fee rate: {FEE_RATE*100:.0f}% on winnings")
    lines.append("")

    # Model trading results
    model_trading = trading_df[trading_df["signal"].str.startswith("Model")].copy()
    lines.append("### Model as Signal")
    lines.append("")
    lines.append("| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |")
    lines.append("|--------|-----------|--------|----------|---------|------|--------|")
    for _, row in model_trading.iterrows():
        period = row["signal"].replace("Model_", "")
        lines.append(f"| {period} | {row['threshold']:.2f} | {row['n_trades']} | "
                      f"{row['win_rate']:.1%} | ${row['net_pnl']:.2f} | "
                      f"{row['roi_pct']:.1f}% | {row['sharpe']:.3f} |")
    lines.append("")

    # NWS trading results
    nws_trading = trading_df[trading_df["signal"].str.startswith("NWS")].copy()
    lines.append("### NWS as Signal")
    lines.append("")
    lines.append("| Period | Threshold | Trades | Win Rate | Net P&L | ROI% | Sharpe |")
    lines.append("|--------|-----------|--------|----------|---------|------|--------|")
    for _, row in nws_trading.iterrows():
        period = row["signal"].replace("NWS_", "")
        lines.append(f"| {period} | {row['threshold']:.2f} | {row['n_trades']} | "
                      f"{row['win_rate']:.1%} | ${row['net_pnl']:.2f} | "
                      f"{row['roi_pct']:.1f}% | {row['sharpe']:.3f} |")
    lines.append("")

    # --- Key Findings ---
    lines.append("## 4. Key Findings")
    lines.append("")

    # Find best Brier for each slice
    overall_scores = scores_df[scores_df["slice"] == "Overall"]
    best_brier_source = overall_scores.loc[overall_scores["brier_score"].idxmin(), "source"]
    best_brier_val = overall_scores["brier_score"].min()
    model_brier = overall_scores.loc[overall_scores["source"] == "Model", "brier_score"].values[0]
    nws_brier = overall_scores.loc[overall_scores["source"] == "NWS", "brier_score"].values[0]
    pre_brier = overall_scores.loc[overall_scores["source"] == "Kalshi_PreSettlement", "brier_score"].values[0]

    lines.append(f"- **Best overall Brier score**: {best_brier_source} ({best_brier_val:.4f})")
    lines.append(f"- Model Brier: {model_brier:.4f}")
    lines.append(f"- NWS Brier: {nws_brier:.4f}")
    lines.append(f"- Pre-settlement market Brier: {pre_brier:.4f}")
    lines.append("")

    # Model vs NWS comparison
    if model_brier < nws_brier:
        lines.append(f"- Model BEATS NWS by {(nws_brier - model_brier):.4f} Brier points")
    else:
        lines.append(f"- NWS BEATS Model by {(model_brier - nws_brier):.4f} Brier points")

    # Model vs pre-settlement
    if model_brier < pre_brier:
        lines.append(f"- Model BEATS pre-settlement market by {(pre_brier - model_brier):.4f} Brier points")
    else:
        lines.append(f"- Pre-settlement market BEATS Model by {(model_brier - pre_brier):.4f} Brier points")
    lines.append("")

    # Best trading strategy
    best_model_trade = model_trading.loc[model_trading["net_pnl"].idxmax()] if len(model_trading) > 0 else None
    best_nws_trade = nws_trading.loc[nws_trading["net_pnl"].idxmax()] if len(nws_trading) > 0 else None

    if best_model_trade is not None:
        lines.append(f"- Best Model trading: threshold={best_model_trade['threshold']:.2f}, "
                      f"P&L=${best_model_trade['net_pnl']:.2f}, "
                      f"ROI={best_model_trade['roi_pct']:.1f}%, "
                      f"{int(best_model_trade['n_trades'])} trades ({best_model_trade['signal']})")

    if best_nws_trade is not None:
        lines.append(f"- Best NWS trading: threshold={best_nws_trade['threshold']:.2f}, "
                      f"P&L=${best_nws_trade['net_pnl']:.2f}, "
                      f"ROI={best_nws_trade['roi_pct']:.1f}%, "
                      f"{int(best_nws_trade['n_trades'])} trades ({best_nws_trade['signal']})")
    lines.append("")

    # Pre-settlement vs settled comparison
    lines.append("## 5. Pre-Settlement vs Settled Market Comparison")
    lines.append("")
    lines.append("The pre-settlement prices are the market consensus BEFORE the event resolves.")
    lines.append("The settled prices reflect the final market state at settlement.")
    lines.append("")
    settled_brier = overall_scores.loc[overall_scores["source"] == "Kalshi_Settled", "brier_score"].values[0]
    lines.append(f"- Pre-settlement market Brier: {pre_brier:.4f}")
    lines.append(f"- Settled market Brier: {settled_brier:.4f}")
    diff = pre_brier - settled_brier
    lines.append(f"- Difference: {diff:.4f} (pre-settlement is {'worse' if diff > 0 else 'better'})")
    lines.append("")
    lines.append("Pre-settlement prices reflect genuine forecasting uncertainty, while settled prices")
    lines.append("often approach 0 or 1 as the outcome becomes known. This makes pre-settlement the")
    lines.append("more meaningful benchmark for comparing forecast quality.")
    lines.append("")

    return "\n".join(lines)


# ==============================================================================
# Print Summary
# ==============================================================================
def print_summary(df, scores_df, cal_df, trading_df):
    """Print concise summary to stdout."""
    print("\n" + "=" * 80)
    print("MODEL vs BENCHMARKS: RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nDataset: {len(df):,} bucket observations across {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Overall Brier Scores
    overall = scores_df[scores_df["slice"] == "Overall"].sort_values("brier_score")
    print("\n--- BRIER SCORES (lower = better) ---")
    print(f"{'Source':<25} {'Brier':>8} {'Log Score':>10}")
    print("-" * 45)
    for _, row in overall.iterrows():
        print(f"{row['source']:<25} {row['brier_score']:>8.4f} {row['log_score']:>10.4f}")

    # By period
    print("\n--- BY PERIOD ---")
    for period in ["IS", "OOS"]:
        period_data = scores_df[scores_df["slice"] == f"Period: {period}"].sort_values("brier_score")
        print(f"\n  {period}:")
        for _, row in period_data.iterrows():
            print(f"    {row['source']:<25} Brier={row['brier_score']:.4f}  LogScore={row['log_score']:.4f}")

    # Calibration ECE
    ece_data = cal_df.groupby("source")["ece"].first().sort_values()
    print("\n--- CALIBRATION (ECE, lower = better) ---")
    for source, ece in ece_data.items():
        print(f"  {source:<25} ECE={ece:.4f}")

    # Trading highlights
    print("\n--- TRADING SIMULATION HIGHLIGHTS ---")
    print(f"  (vs pre-settlement market, {FEE_RATE*100:.0f}% fee on winnings)")

    for signal_prefix, label in [("Model_All", "Model (All)"),
                                   ("Model_OOS", "Model (OOS)"),
                                   ("NWS_All", "NWS (All)"),
                                   ("NWS_OOS", "NWS (OOS)")]:
        sub = trading_df[trading_df["signal"] == signal_prefix]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["net_pnl"].idxmax()]
        print(f"\n  {label} (best threshold={best['threshold']:.2f}):")
        print(f"    Trades: {int(best['n_trades'])}, Win rate: {best['win_rate']:.1%}")
        print(f"    Net P&L: ${best['net_pnl']:.2f}, ROI: {best['roi_pct']:.1f}%, Sharpe: {best['sharpe']:.3f}")

    print("\n" + "=" * 80)


# ==============================================================================
# Main
# ==============================================================================
def main():
    # Load data
    pre, settled, model, nws, cp_tmax = load_all_data()

    # Build merged dataset
    df = build_merged_dataset(pre, settled, model, nws)

    # Compute probabilities
    df = add_all_probabilities(df)

    # Compute scoring metrics
    scores_df = compute_all_scores(df)

    # Compute calibration
    cal_df = compute_calibration(df)

    # Run trading simulations
    trading_df = run_all_trading_sims(df)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save outputs
    print("\nSaving outputs...")

    # Full benchmark comparison dataset
    output_cols = [
        "date", "ticker", "bucket", "direction", "threshold_low", "threshold_high",
        "actual_tmax", "actual_outcome",
        "model_mu", "model_sigma", "model_prob",
        "nws_mu", "nws_sigma", "nws_prob",
        "presettlement_prob", "settled_market_prob",
        "period", "season"
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(OUTPUT_DIR / "full_benchmark_comparison.csv", index=False)
    print(f"  Saved full_benchmark_comparison.csv ({len(df)} rows)")

    # Brier scores
    scores_df.to_csv(OUTPUT_DIR / "presettlement_brier_scores.csv", index=False)
    print(f"  Saved presettlement_brier_scores.csv ({len(scores_df)} rows)")

    # Calibration data
    cal_df.to_csv(OUTPUT_DIR / "presettlement_calibration.csv", index=False)
    print(f"  Saved presettlement_calibration.csv ({len(cal_df)} rows)")

    # Trading simulation results
    trading_df.to_csv(OUTPUT_DIR / "trading_simulation_results.csv", index=False)
    print(f"  Saved trading_simulation_results.csv ({len(trading_df)} rows)")

    # Generate and save report
    report = generate_report(df, scores_df, cal_df, trading_df)
    report_path = OUTPUT_DIR / "full_benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved full_benchmark_report.md")

    # Print summary
    print_summary(df, scores_df, cal_df, trading_df)


if __name__ == "__main__":
    main()
