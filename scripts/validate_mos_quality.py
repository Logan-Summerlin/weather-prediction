#!/usr/bin/env python3
"""
Step 3: Validate MOS Forecast Quality.

Loads the combined MOS forecast data (created by Analyst A) and the actual
Central Park TMAX history, then computes comprehensive forecast accuracy
metrics comparing GFS MOS, NAM MOS, and MOS ensemble against our NN and
Ridge model predictions.

Outputs:
    results/mos_validation/  (metrics JSON, summary CSV, 4+ plots)

Usage:
    python scripts/validate_mos_quality.py
"""

import json
import os
import sys
import time
import logging

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
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "mos_validation")

MOS_COMBINED = os.path.join(DATA_DIR, "mos", "combined_mos_knyc.csv")
TMAX_HISTORY = os.path.join(DATA_DIR, "central_park_tmax_full_history.csv")

# NN / Ridge prediction files (for MAE comparison)
NN_IS = os.path.join(DATA_DIR, "max_train_nn_predictions_is.csv")
NN_OOS = os.path.join(DATA_DIR, "max_train_nn_predictions_oos.csv")
RIDGE_IS = os.path.join(DATA_DIR, "max_train_ridge_predictions_is.csv")
RIDGE_OOS = os.path.join(DATA_DIR, "max_train_ridge_predictions_oos.csv")

# Benchmark MAEs from Phase 4 (approximate)
NN_BENCHMARK_MAE = 4.3   # ~4.29 F
RIDGE_BENCHMARK_MAE = 4.3  # ~4.33 F

MAX_WAIT_SECONDS = 900  # 15 minutes
POLL_INTERVAL = 30       # 30 seconds


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------
def _get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Fall"

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


# ---------------------------------------------------------------------------
# Wait for MOS data
# ---------------------------------------------------------------------------
def wait_for_mos_data():
    """Wait for MOS combined data file. Poll every 30s up to 15 min."""
    if os.path.exists(MOS_COMBINED):
        logger.info("MOS data found immediately: %s", MOS_COMBINED)
        return True

    print(f"  [WAITING] MOS data not yet available: {MOS_COMBINED}")
    print(f"  Checking every {POLL_INTERVAL}s for up to {MAX_WAIT_SECONDS}s...")

    elapsed = 0
    while elapsed < MAX_WAIT_SECONDS:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        if os.path.exists(MOS_COMBINED):
            print(f"  [FOUND] MOS data detected after {elapsed}s.")
            return True
        print(f"  Still waiting... ({elapsed}s / {MAX_WAIT_SECONDS}s)")

    print(f"\n  [ERROR] MOS data not found after {MAX_WAIT_SECONDS}s.")
    return False


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_mos_data():
    """Load the combined MOS forecast data."""
    df = pd.read_csv(MOS_COMBINED)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Loaded MOS data: %d rows, columns=%s", len(df), list(df.columns))
    return df


def load_tmax_history():
    """Load Central Park TMAX actual observations."""
    df = pd.read_csv(TMAX_HISTORY)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Loaded TMAX history: %d rows", len(df))
    return df


def load_model_predictions():
    """Load NN and Ridge prediction files, compute their MAE vs actuals."""
    results = {}
    for label, is_path, oos_path in [
        ("nn", NN_IS, NN_OOS),
        ("ridge", RIDGE_IS, RIDGE_OOS),
    ]:
        frames = []
        for path in [is_path, oos_path]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date")
            results[label] = combined
            logger.info("Loaded %s predictions: %d rows", label, len(combined))
    return results


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_forecast_metrics(forecast, actual, label=""):
    """Compute MAE, RMSE, R-squared, and bias for a forecast vs actual."""
    valid = ~(np.isnan(forecast) | np.isnan(actual))
    f = forecast[valid]
    a = actual[valid]
    n = len(f)
    if n == 0:
        return {"label": label, "n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    errors = f - a
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = float(np.mean(errors))
    return {"label": label, "n": n, "mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def run_validation(mos_df, tmax_df, model_preds):
    """Run comprehensive MOS validation and return all results."""
    # Merge MOS with actuals
    merged = mos_df.merge(
        tmax_df[["date", "tmax_f"]].rename(columns={"tmax_f": "actual_tmax"}),
        on="date", how="inner",
    )
    logger.info("Merged MOS+actual: %d rows, date range: %s to %s",
                len(merged), merged["date"].min(), merged["date"].max())

    if len(merged) == 0:
        raise ValueError("No overlapping dates between MOS and TMAX history")

    merged["month"] = merged["date"].apply(lambda d: d.month)
    merged["year"] = merged["date"].apply(lambda d: d.year)
    merged["season"] = merged["month"].apply(_get_season)

    results = {"date_range": {"start": str(merged["date"].min()), "end": str(merged["date"].max())}}

    # ---- 1. Overall metrics for each MOS product ----
    mos_cols = [c for c in ["gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]
                if c in merged.columns]

    overall_metrics = {}
    for col in mos_cols:
        m = compute_forecast_metrics(
            merged[col].values, merged["actual_tmax"].values, label=col,
        )
        overall_metrics[col] = m
        print(f"  {col}: MAE={m['mae']:.2f}F, RMSE={m['rmse']:.2f}, R2={m['r2']:.3f}, Bias={m['bias']:.2f}F, n={m['n']}")
    results["overall"] = overall_metrics

    # ---- 2. Monthly MAE breakdown ----
    monthly = {}
    for month in range(1, 13):
        mask = merged["month"] == month
        mdf = merged[mask]
        if len(mdf) == 0:
            continue
        month_data = {"n": int(len(mdf))}
        for col in mos_cols:
            m = compute_forecast_metrics(mdf[col].values, mdf["actual_tmax"].values)
            month_data[f"{col}_mae"] = m["mae"]
        monthly[month] = month_data
    results["monthly"] = monthly

    # ---- 3. Seasonal MAE breakdown ----
    seasonal = {}
    for season in SEASON_ORDER:
        mask = merged["season"] == season
        sdf = merged[mask]
        if len(sdf) == 0:
            continue
        season_data = {"n": int(len(sdf))}
        for col in mos_cols:
            m = compute_forecast_metrics(sdf[col].values, sdf["actual_tmax"].values)
            season_data[f"{col}_mae"] = m["mae"]
            season_data[f"{col}_bias"] = m["bias"]
        seasonal[season] = season_data
    results["seasonal"] = seasonal

    # ---- 4. Yearly MAE trends ----
    yearly = {}
    for year in sorted(merged["year"].unique()):
        mask = merged["year"] == year
        ydf = merged[mask]
        year_data = {"n": int(len(ydf))}
        for col in mos_cols:
            m = compute_forecast_metrics(ydf[col].values, ydf["actual_tmax"].values)
            year_data[f"{col}_mae"] = m["mae"]
        yearly[int(year)] = year_data
    results["yearly"] = yearly

    # ---- 5. Compare with NN/Ridge model MAE ----
    comparison = {}
    for model_label, pred_df in model_preds.items():
        overlap = merged.merge(
            pred_df[["date", "model_mu"]].rename(columns={"model_mu": f"{model_label}_mu"}),
            on="date", how="inner",
        )
        if len(overlap) > 0:
            mu_col = f"{model_label}_mu"
            m = compute_forecast_metrics(overlap[mu_col].values, overlap["actual_tmax"].values,
                                          label=model_label)
            comparison[model_label] = m
            print(f"  {model_label} (overlapping dates): MAE={m['mae']:.2f}F, n={m['n']}")

            # Also compute MOS MAE on same dates
            for col in mos_cols:
                m_mos = compute_forecast_metrics(overlap[col].values, overlap["actual_tmax"].values)
                comparison[f"{col}_vs_{model_label}"] = m_mos
    results["model_comparison"] = comparison

    # Store merged data for plotting
    results["_merged"] = merged
    results["_mos_cols"] = mos_cols
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def generate_plots(results, output_dir):
    """Generate MOS validation plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    merged = results["_merged"]
    mos_cols = results["_mos_cols"]

    # Pick the best MOS col (ensemble if available)
    primary_mos = "mos_ensemble_tmax_f" if "mos_ensemble_tmax_f" in mos_cols else mos_cols[0]

    # ---- 1. Actual vs MOS Scatter ----
    fig, axes = plt.subplots(1, len(mos_cols), figsize=(6 * len(mos_cols), 5), squeeze=False)
    for i, col in enumerate(mos_cols):
        ax = axes[0, i]
        valid = merged.dropna(subset=[col, "actual_tmax"])
        ax.scatter(valid["actual_tmax"], valid[col], alpha=0.2, s=8, edgecolors="none")
        lo, hi = 10, 110
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual TMAX (F)")
        ax.set_ylabel("MOS Forecast (F)")
        nice = col.replace("_tmax_f", "").replace("_", " ").upper()
        mae = results["overall"][col]["mae"]
        ax.set_title(f"{nice}\nMAE={mae:.2f}F, n={len(valid)}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "mos_actual_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 2. Monthly MAE Bar Chart (MOS vs NN vs Ridge) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(months))
    width = 0.2

    # MOS ensemble monthly MAEs
    mos_mae_monthly = []
    for m in months:
        mdata = results["monthly"].get(m, {})
        mos_mae_monthly.append(mdata.get(f"{primary_mos}_mae", np.nan))
    ax.bar(x - width, mos_mae_monthly, width, label="MOS Ensemble", color="#1f77b4", alpha=0.8)

    # NN benchmark line
    ax.axhline(NN_BENCHMARK_MAE, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"NN Benchmark ({NN_BENCHMARK_MAE}F)")

    # Ridge benchmark line
    ax.axhline(RIDGE_BENCHMARK_MAE, color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"Ridge Benchmark ({RIDGE_BENCHMARK_MAE}F)")

    # If we have per-month NN/Ridge MAE from overlapping data, add bars
    for model_label, color, offset in [("nn", "#d62728", 0), ("ridge", "#ff7f0e", width)]:
        model_maes = []
        has_data = False
        comp = results.get("model_comparison", {})
        # Recompute monthly from merged
        if model_label in comp:
            pred_key = f"{model_label}_mu"
            merged_ext = merged.copy()
            # Load model preds again for monthly
            for is_path, oos_path in [(NN_IS, NN_OOS)] if model_label == "nn" else [(RIDGE_IS, RIDGE_OOS)]:
                frames = []
                for p in [is_path, oos_path]:
                    if os.path.exists(p):
                        frames.append(pd.read_csv(p))
                if frames:
                    pred_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date")
                    pred_all["date"] = pd.to_datetime(pred_all["date"]).dt.date
                    merged_ext = merged_ext.merge(
                        pred_all[["date", "model_mu"]].rename(columns={"model_mu": pred_key}),
                        on="date", how="left",
                    )
            if pred_key in merged_ext.columns:
                for m in months:
                    mask = merged_ext["month"] == m
                    mdf = merged_ext[mask].dropna(subset=[pred_key, "actual_tmax"])
                    if len(mdf) > 0:
                        model_maes.append(float(np.mean(np.abs(mdf[pred_key] - mdf["actual_tmax"]))))
                        has_data = True
                    else:
                        model_maes.append(np.nan)
        if has_data:
            ax.bar(x + offset, model_maes, width,
                   label=f"{model_label.upper()} (overlapping)", color=color, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("MAE (F)")
    ax.set_title("Monthly MAE: MOS vs NN vs Ridge")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, "monthly_mae_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 3. Error Distribution Histogram ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in mos_cols:
        valid = merged.dropna(subset=[col, "actual_tmax"])
        errors = valid[col] - valid["actual_tmax"]
        nice = col.replace("_tmax_f", "").replace("_", " ").upper()
        ax.hist(errors, bins=50, alpha=0.5, label=f"{nice} (bias={errors.mean():.2f}F)", density=True)
    ax.set_xlabel("Forecast Error (F)")
    ax.set_ylabel("Density")
    ax.set_title("MOS Forecast Error Distribution")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "error_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 4. Rolling MAE Time Series ----
    fig, ax = plt.subplots(figsize=(14, 5))
    merged_sorted = merged.sort_values("date").copy()
    for col in mos_cols:
        abs_err = (merged_sorted[col] - merged_sorted["actual_tmax"]).abs()
        rolling = abs_err.rolling(90, min_periods=30).mean()
        nice = col.replace("_tmax_f", "").replace("_", " ").upper()
        ax.plot(merged_sorted["date"], rolling, label=nice, linewidth=1.2)

    ax.axhline(NN_BENCHMARK_MAE, color="#d62728", linestyle="--", linewidth=1,
               label=f"NN Benchmark ({NN_BENCHMARK_MAE}F)")
    ax.set_xlabel("Date")
    ax.set_ylabel("90-Day Rolling MAE (F)")
    ax.set_title("MOS Forecast Quality Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "rolling_mae_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # ---- 5. Yearly MAE Trend ----
    yearly = results.get("yearly", {})
    if yearly:
        fig, ax = plt.subplots(figsize=(10, 5))
        years = sorted(yearly.keys())
        for col in mos_cols:
            mae_vals = [yearly[y].get(f"{col}_mae", np.nan) for y in years]
            nice = col.replace("_tmax_f", "").replace("_", " ").upper()
            ax.plot(years, mae_vals, "o-", label=nice, linewidth=1.5, markersize=5)
        ax.axhline(NN_BENCHMARK_MAE, color="#d62728", linestyle="--", linewidth=1,
                   label=f"NN Benchmark ({NN_BENCHMARK_MAE}F)")
        ax.set_xlabel("Year")
        ax.set_ylabel("MAE (F)")
        ax.set_title("Yearly MAE Trend: MOS vs NN Benchmark")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, "yearly_mae_trend.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    logger.info("Generated %d plots in %s", len(saved), output_dir)
    return saved


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a comprehensive text summary."""
    print()
    print("=" * 78)
    print("  MOS FORECAST QUALITY VALIDATION REPORT")
    print("=" * 78)
    print()

    print(f"  Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print()

    # Overall
    print("--- OVERALL METRICS ---")
    for col, m in results["overall"].items():
        nice = col.replace("_tmax_f", "").replace("_", " ").upper()
        print(f"  {nice}:")
        print(f"    MAE={m['mae']:.2f}F  RMSE={m['rmse']:.2f}F  R2={m['r2']:.3f}  Bias={m['bias']:.2f}F  n={m['n']}")
    print()

    # vs Benchmarks
    print("--- COMPARISON WITH OUR MODELS ---")
    primary_mos_key = None
    for k in ["mos_ensemble_tmax_f", "gfs_mos_tmax_f", "nam_mos_tmax_f"]:
        if k in results["overall"]:
            primary_mos_key = k
            break
    if primary_mos_key:
        mos_mae = results["overall"][primary_mos_key]["mae"]
        print(f"  MOS Ensemble MAE:     {mos_mae:.2f}F")
        print(f"  NN Benchmark MAE:     {NN_BENCHMARK_MAE:.2f}F")
        print(f"  Ridge Benchmark MAE:  {RIDGE_BENCHMARK_MAE:.2f}F")
        print()
        if mos_mae < NN_BENCHMARK_MAE:
            diff = NN_BENCHMARK_MAE - mos_mae
            pct = diff / NN_BENCHMARK_MAE * 100
            print(f"  RESULT: MOS BEATS our NN by {diff:.2f}F ({pct:.1f}%)")
            print(f"  MOS is a MUCH better point forecaster than our model.")
        elif mos_mae > NN_BENCHMARK_MAE:
            diff = mos_mae - NN_BENCHMARK_MAE
            pct = diff / NN_BENCHMARK_MAE * 100
            print(f"  RESULT: Our NN beats MOS by {diff:.2f}F ({pct:.1f}%)")
        else:
            print(f"  RESULT: MOS and NN are tied")
    print()

    # Seasonal
    print("--- SEASONAL BREAKDOWN ---")
    for season in SEASON_ORDER:
        sdata = results["seasonal"].get(season, {})
        if sdata:
            parts = [f"n={sdata['n']}"]
            for k, v in sdata.items():
                if k.endswith("_mae"):
                    nice = k.replace("_tmax_f_mae", "").replace("_", " ").upper()
                    parts.append(f"{nice} MAE={v:.2f}F")
            print(f"  {season:8s}: {', '.join(parts)}")
    print()

    # Yearly trend
    print("--- YEARLY TRENDS ---")
    yearly = results.get("yearly", {})
    for year in sorted(yearly.keys()):
        ydata = yearly[year]
        parts = [f"n={ydata['n']}"]
        for k, v in ydata.items():
            if k.endswith("_mae"):
                nice = k.replace("_tmax_f_mae", "").replace("_", " ").upper()
                parts.append(f"{nice}={v:.2f}F")
        print(f"  {year}: {', '.join(parts)}")
    print()

    # Model comparison on overlapping dates
    comp = results.get("model_comparison", {})
    if comp:
        print("--- MODEL COMPARISON (OVERLAPPING DATES) ---")
        for label, m in comp.items():
            if isinstance(m, dict) and "mae" in m:
                print(f"  {label}: MAE={m['mae']:.2f}F, n={m['n']}")
    print()

    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("  STEP 3: VALIDATE MOS FORECAST QUALITY")
    print("=" * 78)
    print()

    # Wait for MOS data
    if not wait_for_mos_data():
        print("  FATAL: MOS data not available. Analyst A may still be working.")
        print("  Exiting.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\nLoading data...")
    mos_df = load_mos_data()
    tmax_df = load_tmax_history()
    model_preds = load_model_predictions()

    print(f"  MOS: {len(mos_df)} rows")
    print(f"  TMAX history: {len(tmax_df)} rows")
    for label, df in model_preds.items():
        print(f"  {label} predictions: {len(df)} rows")
    print()

    # Run validation
    print("Computing metrics...")
    results = run_validation(mos_df, tmax_df, model_preds)

    # Save metrics JSON (exclude _merged DataFrame)
    save_results = {k: v for k, v in results.items() if not k.startswith("_")}
    # Convert keys to strings for JSON
    if "monthly" in save_results:
        save_results["monthly"] = {str(k): v for k, v in save_results["monthly"].items()}
    if "yearly" in save_results:
        save_results["yearly"] = {str(k): v for k, v in save_results["yearly"].items()}
    with open(os.path.join(OUTPUT_DIR, "mos_validation_metrics.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save summary CSV
    rows = []
    for col, m in results["overall"].items():
        rows.append({"source": col, **m})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)

    # Generate plots
    print("\nGenerating plots...")
    saved_plots = generate_plots(results, OUTPUT_DIR)
    print(f"  Generated {len(saved_plots)} plots")

    # Print comprehensive summary
    print_summary(results)

    # List output files
    print("Output files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fpath} ({size_kb:.1f} KB)")
    print()
    print("MOS validation complete.")


if __name__ == "__main__":
    main()
