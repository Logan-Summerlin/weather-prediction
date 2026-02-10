#!/usr/bin/env python3
"""
NWS/MOS Probability Distribution Benchmark.

Converts NWS MOS point forecasts into full probability distributions using
historically-calibrated monthly error statistics, then compares against our
neural network model and Kalshi settled market probabilities.

Training data (pre-2023) is used exclusively for error distribution fitting
to prevent data leakage.

Outputs (results/prediction_market_benchmark/):
    - nws_probability_forecasts.csv
    - benchmark_comparison.csv
    - brier_scores_summary.csv
    - benchmark_report.md
"""

import os
import sys
import warnings
from datetime import date
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MOS_PATH = os.path.join(PROJECT_ROOT, "data", "mos", "combined_mos_knyc.csv")
ACTUALS_PATH = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
MODEL_IS_PATH = os.path.join(PROJECT_ROOT, "data", "max_train_nn_predictions_is.csv")
MODEL_OOS_PATH = os.path.join(PROJECT_ROOT, "data", "max_train_nn_predictions_oos.csv")
KALSHI_IS_PATH = os.path.join(PROJECT_ROOT, "data", "real_kalshi_2023_2024.csv")
KALSHI_OOS_PATH = os.path.join(PROJECT_ROOT, "data", "real_kalshi_2025.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "prediction_market_benchmark")

TRAIN_CUTOFF = "2023-01-01"  # Exclusive upper bound for training data
PROB_CLIP_LO = 0.001
PROB_CLIP_HI = 0.999

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


# ===================================================================
# 1. Load data
# ===================================================================
def load_mos_forecasts() -> pd.DataFrame:
    """Load MOS forecasts; prefer mos_ensemble_tmax_f, fallback to gfs."""
    df = pd.read_csv(MOS_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date

    # Build best_forecast column: ensemble first, then gfs fallback
    df["best_forecast"] = df["mos_ensemble_tmax_f"]
    mask_missing = df["best_forecast"].isna()
    if "gfs_mos_tmax_f" in df.columns:
        df.loc[mask_missing, "best_forecast"] = df.loc[mask_missing, "gfs_mos_tmax_f"]

    df = df.dropna(subset=["best_forecast"])
    df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
    print(f"[Load] MOS forecasts: {len(df)} days "
          f"({df['date'].min()} to {df['date'].max()})")
    return df[["date", "best_forecast"]].reset_index(drop=True)


def load_actuals() -> pd.DataFrame:
    """Load Central Park TMAX actuals."""
    df = pd.read_csv(ACTUALS_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df.dropna(subset=["tmax_f"])
    df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
    print(f"[Load] Actuals: {len(df)} days "
          f"({df['date'].min()} to {df['date'].max()})")
    return df[["date", "tmax_f"]].reset_index(drop=True)


def load_model_predictions() -> pd.DataFrame:
    """Load NN model predictions (IS + OOS), tagged with period."""
    is_df = pd.read_csv(MODEL_IS_PATH, parse_dates=["date"])
    oos_df = pd.read_csv(MODEL_OOS_PATH, parse_dates=["date"])
    is_df["period"] = "IS"
    oos_df["period"] = "OOS"
    df = pd.concat([is_df, oos_df], ignore_index=True)
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[Load] Model predictions: {len(df)} days "
          f"(IS={len(is_df)}, OOS={len(oos_df)})")
    return df


def load_kalshi_data() -> pd.DataFrame:
    """Load Kalshi settled data (IS + OOS), tagged with period."""
    is_df = pd.read_csv(KALSHI_IS_PATH, parse_dates=["date"])
    oos_df = pd.read_csv(KALSHI_OOS_PATH, parse_dates=["date"])
    is_df["period"] = "IS"
    oos_df["period"] = "OOS"
    df = pd.concat([is_df, oos_df], ignore_index=True)
    df["date"] = df["date"].dt.date
    df = df.sort_values(["date", "threshold"]).reset_index(drop=True)
    print(f"[Load] Kalshi data: {len(df)} rows over "
          f"{df['date'].nunique()} unique dates")
    return df


# ===================================================================
# 2. Fit monthly error distribution (training data only)
# ===================================================================
def fit_monthly_error_distribution(
    mos_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute monthly bias and sigma from training data (before 2023-01-01).

    Returns DataFrame with columns: month, bias, sigma, mae, n_samples.
    """
    cutoff = date(2023, 1, 1)

    merged = pd.merge(mos_df, actuals_df, on="date", how="inner")
    train = merged[merged["date"] < cutoff].copy()

    if train.empty:
        raise ValueError("No training data found before 2023-01-01")

    train["error"] = train["tmax_f"] - train["best_forecast"]
    train["month"] = pd.to_datetime(train["date"].astype(str)).dt.month

    print(f"\n[Fit] Training error distribution on {len(train)} days "
          f"({train['date'].min()} to {train['date'].max()})")

    # Overall stats
    overall_bias = train["error"].mean()
    overall_sigma = train["error"].std(ddof=1)
    overall_mae = train["error"].abs().mean()
    print(f"[Fit] Overall: bias={overall_bias:.2f}, sigma={overall_sigma:.2f}, "
          f"MAE={overall_mae:.2f}")

    # Monthly stats
    rows = []
    for month in range(1, 13):
        m_data = train[train["month"] == month]
        if len(m_data) >= 10:
            bias = m_data["error"].mean()
            sigma = m_data["error"].std(ddof=1)
            mae = m_data["error"].abs().mean()
            n = len(m_data)
        else:
            # Fallback to overall (unlikely with 19 years of data)
            bias = overall_bias
            sigma = overall_sigma
            mae = overall_mae
            n = len(m_data)
        rows.append({
            "month": month,
            "bias": bias,
            "sigma": sigma,
            "mae": mae,
            "n_samples": n,
        })

    err_dist = pd.DataFrame(rows)
    print("\n[Fit] Monthly error distribution:")
    for _, r in err_dist.iterrows():
        print(f"  Month {int(r['month']):2d}: bias={r['bias']:+.2f}F, "
              f"sigma={r['sigma']:.2f}F, MAE={r['mae']:.2f}F, "
              f"n={int(r['n_samples'])}")

    return err_dist


# ===================================================================
# 3. Compute bucket probabilities
# ===================================================================
def compute_bucket_prob(
    mu: float,
    sigma: float,
    threshold_low,
    threshold_high,
    direction: str,
) -> float:
    """
    Compute P(TMAX in bucket) given N(mu, sigma).

    Handles 'below', 'above', and 'between' bucket types.
    Clips output to [PROB_CLIP_LO, PROB_CLIP_HI].
    """
    sigma = max(sigma, 1e-6)
    dist = stats.norm(loc=mu, scale=sigma)

    if direction == "below":
        hi = float(threshold_high) if pd.notna(threshold_high) else 999.0
        prob = dist.cdf(hi)
    elif direction == "above":
        lo = float(threshold_low) if pd.notna(threshold_low) else -999.0
        prob = 1.0 - dist.cdf(lo)
    elif direction == "between":
        lo = float(threshold_low) if pd.notna(threshold_low) else -999.0
        hi = float(threshold_high) if pd.notna(threshold_high) else 999.0
        prob = dist.cdf(hi) - dist.cdf(lo)
    else:
        prob = 0.5

    return float(np.clip(prob, PROB_CLIP_LO, PROB_CLIP_HI))


def build_nws_forecasts(
    mos_df: pd.DataFrame,
    err_dist: pd.DataFrame,
    kalshi_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each Kalshi bucket on each test date, compute:
      - NWS probability: N(mos_forecast + bias[month], sigma[month])
      - Model probability: N(model_mu, model_sigma)

    Returns a row-per-bucket DataFrame.
    """
    # Build lookup dictionaries
    mos_lookup = dict(zip(mos_df["date"], mos_df["best_forecast"]))
    model_lookup = {}
    for _, row in model_df.iterrows():
        model_lookup[row["date"]] = (row["model_mu"], row["model_sigma"])

    bias_lookup = dict(zip(err_dist["month"].astype(int), err_dist["bias"]))
    sigma_lookup = dict(zip(err_dist["month"].astype(int), err_dist["sigma"]))

    # Overall fallback
    overall_bias = err_dist["bias"].mean()
    overall_sigma = err_dist["sigma"].mean()

    records = []
    skipped_no_mos = set()
    skipped_no_model = set()

    for _, krow in kalshi_df.iterrows():
        d = krow["date"]
        month = pd.Timestamp(d).month
        period = krow["period"]
        season = SEASON_MAP.get(month, "Unknown")

        # --- NWS probability ---
        mos_forecast = mos_lookup.get(d)
        if mos_forecast is None:
            skipped_no_mos.add(d)
            nws_mu = None
            nws_sigma = None
            nws_prob = None
        else:
            bias = bias_lookup.get(month, overall_bias)
            sigma = sigma_lookup.get(month, overall_sigma)
            nws_mu = mos_forecast + bias
            nws_sigma = sigma
            nws_prob = compute_bucket_prob(
                nws_mu, nws_sigma,
                krow.get("threshold_low"), krow.get("threshold_high"),
                krow["direction"],
            )

        # --- Model probability ---
        model_vals = model_lookup.get(d)
        if model_vals is None:
            skipped_no_model.add(d)
            model_mu_val = None
            model_sigma_val = None
            model_prob = None
        else:
            model_mu_val, model_sigma_val = model_vals
            model_prob = compute_bucket_prob(
                model_mu_val, model_sigma_val,
                krow.get("threshold_low"), krow.get("threshold_high"),
                krow["direction"],
            )

        # --- Market probability (clip for log score) ---
        market_prob = krow.get("market_prob")
        if pd.notna(market_prob):
            market_prob = float(np.clip(market_prob, PROB_CLIP_LO, PROB_CLIP_HI))

        records.append({
            "date": d,
            "period": period,
            "season": season,
            "month": month,
            "ticker": krow.get("ticker", ""),
            "bucket": krow.get("bucket", ""),
            "direction": krow["direction"],
            "threshold_low": krow.get("threshold_low"),
            "threshold_high": krow.get("threshold_high"),
            "actual_tmax": krow.get("actual_tmax"),
            "actual_outcome": krow.get("actual_outcome"),
            "mos_forecast": mos_forecast,
            "nws_mu": nws_mu,
            "nws_sigma": nws_sigma,
            "nws_prob": nws_prob,
            "model_mu": model_mu_val,
            "model_sigma": model_sigma_val,
            "model_prob": model_prob,
            "market_prob": market_prob,
        })

    result = pd.DataFrame(records)

    if skipped_no_mos:
        print(f"\n[Warning] {len(skipped_no_mos)} dates had no MOS forecast")
    if skipped_no_model:
        print(f"[Warning] {len(skipped_no_model)} dates had no model prediction")

    # Report coverage
    total_dates = result["date"].nunique()
    has_nws = result.dropna(subset=["nws_prob"])["date"].nunique()
    has_model = result.dropna(subset=["model_prob"])["date"].nunique()
    has_market = result.dropna(subset=["market_prob"])["date"].nunique()
    print(f"\n[Build] {len(result)} bucket-rows across {total_dates} dates")
    print(f"  NWS coverage: {has_nws}/{total_dates} dates")
    print(f"  Model coverage: {has_model}/{total_dates} dates")
    print(f"  Market coverage: {has_market}/{total_dates} dates")

    return result


# ===================================================================
# 4. Scoring functions
# ===================================================================
def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean Brier score: mean((p - y)^2)."""
    mask = ~(np.isnan(prob) | np.isnan(outcome))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((prob[mask] - outcome[mask]) ** 2))


def log_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean log score: -mean(y*log(p) + (1-y)*log(1-p))."""
    mask = ~(np.isnan(prob) | np.isnan(outcome))
    if mask.sum() == 0:
        return np.nan
    p = np.clip(prob[mask], PROB_CLIP_LO, PROB_CLIP_HI)
    y = outcome[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_stats(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10):
    """Compute calibration curve (bin midpoints, observed fractions, counts)."""
    mask = ~(np.isnan(prob) | np.isnan(outcome))
    p = prob[mask]
    y = outcome[mask]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mid = []
    bin_frac = []
    bin_count = []
    for i in range(n_bins):
        in_bin = (p >= bins[i]) & (p < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            in_bin = in_bin | (p == bins[i + 1])
        count = in_bin.sum()
        bin_count.append(count)
        bin_mid.append((bins[i] + bins[i + 1]) / 2)
        bin_frac.append(y[in_bin].mean() if count > 0 else np.nan)
    return np.array(bin_mid), np.array(bin_frac), np.array(bin_count)


def compute_all_scores(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Compute Brier and log scores for NWS, model, and market across
    different slices (overall, by period, by season).
    """
    sources = [
        ("NWS", "nws_prob"),
        ("Model", "model_prob"),
        ("Market", "market_prob"),
    ]

    rows = []
    outcome = df["actual_outcome"].values.astype(float)

    # --- Overall ---
    for name, col in sources:
        probs = df[col].values.astype(float)
        bs = brier_score(probs, outcome)
        ls = log_score(probs, outcome)
        n_valid = int((~np.isnan(probs)).sum())
        rows.append({
            "slice": "Overall",
            "source": name,
            "brier_score": bs,
            "log_score": ls,
            "n_buckets": n_valid,
        })

    # --- By period ---
    for period in ["IS", "OOS"]:
        sub = df[df["period"] == period]
        out = sub["actual_outcome"].values.astype(float)
        for name, col in sources:
            probs = sub[col].values.astype(float)
            bs = brier_score(probs, out)
            ls = log_score(probs, out)
            n_valid = int((~np.isnan(probs)).sum())
            rows.append({
                "slice": f"Period: {period}",
                "source": name,
                "brier_score": bs,
                "log_score": ls,
                "n_buckets": n_valid,
            })

    # --- By season ---
    for season in ["Winter", "Spring", "Summer", "Fall"]:
        sub = df[df["season"] == season]
        if sub.empty:
            continue
        out = sub["actual_outcome"].values.astype(float)
        for name, col in sources:
            probs = sub[col].values.astype(float)
            bs = brier_score(probs, out)
            ls = log_score(probs, out)
            n_valid = int((~np.isnan(probs)).sum())
            rows.append({
                "slice": f"Season: {season}",
                "source": name,
                "brier_score": bs,
                "log_score": ls,
                "n_buckets": n_valid,
            })

    return pd.DataFrame(rows)


# ===================================================================
# 5. Point-forecast accuracy metrics (MAE, bias) for NWS vs Model
# ===================================================================
def compute_point_forecast_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare NWS and Model point forecast accuracy using the first row per date
    (since nws_mu, model_mu, actual_tmax are the same across buckets for a date).
    """
    day_df = df.dropna(subset=["actual_tmax"]).drop_duplicates(
        subset="date", keep="first"
    ).copy()

    rows = []
    for period_label, sub in [("Overall", day_df),
                               ("IS", day_df[day_df["period"] == "IS"]),
                               ("OOS", day_df[day_df["period"] == "OOS"])]:
        if sub.empty:
            continue

        # NWS
        nws_sub = sub.dropna(subset=["nws_mu"])
        if not nws_sub.empty:
            nws_err = nws_sub["actual_tmax"] - nws_sub["nws_mu"]
            rows.append({
                "slice": period_label,
                "source": "NWS (bias-corrected)",
                "mae": nws_err.abs().mean(),
                "bias": nws_err.mean(),
                "rmse": np.sqrt((nws_err ** 2).mean()),
                "n_days": len(nws_sub),
            })
            # Raw MOS (without bias correction)
            raw_err = nws_sub["actual_tmax"] - nws_sub["mos_forecast"]
            rows.append({
                "slice": period_label,
                "source": "NWS (raw MOS)",
                "mae": raw_err.abs().mean(),
                "bias": raw_err.mean(),
                "rmse": np.sqrt((raw_err ** 2).mean()),
                "n_days": len(nws_sub),
            })

        # Model
        model_sub = sub.dropna(subset=["model_mu"])
        if not model_sub.empty:
            model_err = model_sub["actual_tmax"] - model_sub["model_mu"]
            rows.append({
                "slice": period_label,
                "source": "NN Model",
                "mae": model_err.abs().mean(),
                "bias": model_err.mean(),
                "rmse": np.sqrt((model_err ** 2).mean()),
                "n_days": len(model_sub),
            })

    return pd.DataFrame(rows)


# ===================================================================
# 6. Generate report
# ===================================================================
def generate_report(
    err_dist: pd.DataFrame,
    scores_df: pd.DataFrame,
    point_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> str:
    """Generate benchmark_report.md content."""

    # Pivot scores for display
    overall_scores = scores_df[scores_df["slice"] == "Overall"].copy()

    # Point forecast summary
    overall_point = point_df[point_df["slice"] == "Overall"].copy()

    # Count dates
    n_dates = bench_df["date"].nunique()
    n_is = bench_df[bench_df["period"] == "IS"]["date"].nunique()
    n_oos = bench_df[bench_df["period"] == "OOS"]["date"].nunique()

    lines = []
    lines.append("# NWS/MOS Probability Benchmark Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Total dates evaluated:** {n_dates} "
                 f"(IS: {n_is}, OOS: {n_oos})")
    lines.append(f"- **Total bucket-rows:** {len(bench_df)}")
    lines.append(f"- **Training cutoff:** {TRAIN_CUTOFF} "
                 f"(error dist fit on pre-2023 data only)")
    lines.append(f"- **NWS approach:** MOS ensemble + monthly bias correction "
                 f"+ monthly sigma -> N(mu, sigma)")
    lines.append("")

    # Monthly error distribution
    lines.append("## Monthly Error Distribution (Training Data)")
    lines.append("")
    lines.append("| Month | Bias (F) | Sigma (F) | MAE (F) | N |")
    lines.append("|-------|----------|-----------|---------|---|")
    for _, r in err_dist.iterrows():
        lines.append(f"| {int(r['month']):2d} | {r['bias']:+.2f} | "
                     f"{r['sigma']:.2f} | {r['mae']:.2f} | "
                     f"{int(r['n_samples'])} |")
    lines.append("")

    # Point forecast accuracy
    lines.append("## Point Forecast Accuracy (MAE in F)")
    lines.append("")
    lines.append("| Slice | Source | MAE | Bias | RMSE | N Days |")
    lines.append("|-------|--------|-----|------|------|--------|")
    for _, r in point_df.iterrows():
        lines.append(f"| {r['slice']} | {r['source']} | "
                     f"{r['mae']:.2f} | {r['bias']:+.2f} | "
                     f"{r['rmse']:.2f} | {int(r['n_days'])} |")
    lines.append("")

    # Brier & log scores
    lines.append("## Probability Scores (Brier / Log Score)")
    lines.append("")
    lines.append("Lower is better for both metrics.")
    lines.append("")
    lines.append("| Slice | Source | Brier Score | Log Score | N Buckets |")
    lines.append("|-------|--------|-------------|-----------|-----------|")
    for _, r in scores_df.iterrows():
        bs_str = f"{r['brier_score']:.4f}" if pd.notna(r["brier_score"]) else "N/A"
        ls_str = f"{r['log_score']:.4f}" if pd.notna(r["log_score"]) else "N/A"
        lines.append(f"| {r['slice']} | {r['source']} | "
                     f"{bs_str} | {ls_str} | {int(r['n_buckets'])} |")
    lines.append("")

    # Interpretation
    lines.append("## Key Findings")
    lines.append("")

    # Extract overall brier scores
    nws_bs = overall_scores.loc[
        overall_scores["source"] == "NWS", "brier_score"
    ].values
    model_bs = overall_scores.loc[
        overall_scores["source"] == "Model", "brier_score"
    ].values
    market_bs = overall_scores.loc[
        overall_scores["source"] == "Market", "brier_score"
    ].values

    if len(nws_bs) > 0 and len(model_bs) > 0:
        nws_val = nws_bs[0]
        model_val = model_bs[0]
        if pd.notna(nws_val) and pd.notna(model_val):
            if model_val < nws_val:
                pct = (1 - model_val / nws_val) * 100
                lines.append(
                    f"- **Model outperforms NWS** on Brier score by "
                    f"{pct:.1f}% ({model_val:.4f} vs {nws_val:.4f})"
                )
            else:
                pct = (1 - nws_val / model_val) * 100
                lines.append(
                    f"- **NWS outperforms Model** on Brier score by "
                    f"{pct:.1f}% ({nws_val:.4f} vs {model_val:.4f})"
                )

    if len(market_bs) > 0 and len(model_bs) > 0:
        mkt_val = market_bs[0]
        model_val = model_bs[0]
        if pd.notna(mkt_val) and pd.notna(model_val):
            if model_val < mkt_val:
                pct = (1 - model_val / mkt_val) * 100
                lines.append(
                    f"- **Model outperforms Market** on Brier score by "
                    f"{pct:.1f}% ({model_val:.4f} vs {mkt_val:.4f})"
                )
            else:
                pct = (1 - mkt_val / model_val) * 100
                lines.append(
                    f"- **Market outperforms Model** on Brier score by "
                    f"{pct:.1f}% ({mkt_val:.4f} vs {model_val:.4f})"
                )

    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("1. **NWS Distribution:** For each test date, the NWS "
                 "probability is modeled as N(MOS_forecast + bias_monthly, "
                 "sigma_monthly), where bias and sigma are computed from "
                 "training data (2004-2022) only.")
    lines.append("2. **Model Distribution:** N(model_mu, model_sigma) from "
                 "the neural network's probabilistic output.")
    lines.append("3. **Market Distribution:** Settled Kalshi market "
                 "probabilities (clipped to [0.001, 0.999]).")
    lines.append("4. **Brier Score:** mean((predicted_prob - outcome)^2). "
                 "Lower is better.")
    lines.append("5. **Log Score:** -mean(y*log(p) + (1-y)*log(1-p)). "
                 "Lower is better.")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("NWS/MOS Probability Distribution Benchmark")
    print("=" * 70)

    # --- Load data ---
    mos_df = load_mos_forecasts()
    actuals_df = load_actuals()
    model_df = load_model_predictions()
    kalshi_df = load_kalshi_data()

    # --- Fit error distribution on training data ---
    err_dist = fit_monthly_error_distribution(mos_df, actuals_df)

    # --- Build NWS + Model probability forecasts for each Kalshi bucket ---
    bench_df = build_nws_forecasts(mos_df, err_dist, kalshi_df, model_df)

    # --- Compute scoring metrics ---
    print("\n" + "=" * 70)
    print("Computing scores...")
    print("=" * 70)

    # Only use rows where we have all three sources for fair comparison
    valid_all = bench_df.dropna(
        subset=["nws_prob", "model_prob", "market_prob"]
    )
    print(f"\n[Score] Rows with all 3 sources: {len(valid_all)} "
          f"({valid_all['date'].nunique()} dates)")

    # Compute scores on the full dataset (each source using its own valid rows)
    scores_df = compute_all_scores(bench_df, "all")
    # Also compute on the matched subset for strict comparison
    scores_matched = compute_all_scores(valid_all, "matched")

    # Point forecast accuracy
    point_df = compute_point_forecast_metrics(bench_df)

    # --- Calibration ---
    print("\n[Calibration] 10-bin calibration stats:")
    for name, col in [("NWS", "nws_prob"), ("Model", "model_prob"),
                      ("Market", "market_prob")]:
        sub = valid_all.dropna(subset=[col])
        if sub.empty:
            continue
        probs = sub[col].values.astype(float)
        outcome = sub["actual_outcome"].values.astype(float)
        mids, fracs, counts = calibration_stats(probs, outcome)
        print(f"\n  {name}:")
        for m, f, c in zip(mids, fracs, counts):
            f_str = f"{f:.3f}" if not np.isnan(f) else "  N/A"
            print(f"    bin {m:.2f}: observed={f_str}, count={int(c)}")

    # --- Save outputs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. NWS probability forecasts (date-level summary)
    nws_daily = bench_df.dropna(subset=["nws_mu"]).drop_duplicates(
        subset="date", keep="first"
    )[["date", "period", "season", "month", "mos_forecast", "nws_mu",
       "nws_sigma", "actual_tmax"]].copy()
    nws_daily.to_csv(
        os.path.join(OUTPUT_DIR, "nws_probability_forecasts.csv"),
        index=False,
    )
    print(f"\n[Save] nws_probability_forecasts.csv ({len(nws_daily)} rows)")

    # 2. Full benchmark comparison (bucket-level)
    bench_df.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_comparison.csv"),
        index=False,
    )
    print(f"[Save] benchmark_comparison.csv ({len(bench_df)} rows)")

    # 3. Brier scores summary
    # Combine full and matched scores
    scores_df["comparison"] = "all_available"
    scores_matched["comparison"] = "matched_subset"
    all_scores = pd.concat([scores_df, scores_matched], ignore_index=True)

    # Add point forecast metrics
    point_df_out = point_df.copy()
    point_df_out.to_csv(
        os.path.join(OUTPUT_DIR, "point_forecast_comparison.csv"),
        index=False,
    )
    all_scores.to_csv(
        os.path.join(OUTPUT_DIR, "brier_scores_summary.csv"),
        index=False,
    )
    print(f"[Save] brier_scores_summary.csv ({len(all_scores)} rows)")
    print(f"[Save] point_forecast_comparison.csv ({len(point_df_out)} rows)")

    # 4. Report
    report = generate_report(err_dist, scores_matched, point_df, bench_df)
    report_path = os.path.join(OUTPUT_DIR, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[Save] benchmark_report.md")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Overall Brier Scores (matched subset)")
    print("=" * 70)
    summary = scores_matched[scores_matched["slice"] == "Overall"]
    for _, r in summary.iterrows():
        bs_str = f"{r['brier_score']:.4f}" if pd.notna(r["brier_score"]) else "N/A"
        ls_str = f"{r['log_score']:.4f}" if pd.notna(r["log_score"]) else "N/A"
        print(f"  {r['source']:>10s}: Brier={bs_str}  Log={ls_str}  "
              f"N={int(r['n_buckets'])}")

    print("\n" + "=" * 70)
    print("SUMMARY: Point Forecast MAE (F)")
    print("=" * 70)
    for _, r in point_df[point_df["slice"] == "Overall"].iterrows():
        print(f"  {r['source']:>25s}: MAE={r['mae']:.2f}  "
              f"Bias={r['bias']:+.2f}  RMSE={r['rmse']:.2f}")

    # IS vs OOS comparison
    print("\n" + "=" * 70)
    print("SUMMARY: IS vs OOS Brier Scores (matched subset)")
    print("=" * 70)
    for period in ["IS", "OOS"]:
        label = f"Period: {period}"
        sub = scores_matched[scores_matched["slice"] == label]
        if sub.empty:
            continue
        print(f"\n  {period}:")
        for _, r in sub.iterrows():
            bs_str = (f"{r['brier_score']:.4f}"
                      if pd.notna(r["brier_score"]) else "N/A")
            print(f"    {r['source']:>10s}: Brier={bs_str}  "
                  f"N={int(r['n_buckets'])}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
