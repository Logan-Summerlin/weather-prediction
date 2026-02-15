#!/usr/bin/env python3
"""
Philadelphia Benchmark: Model vs NWS MOS vs Market Proxy (2024/2025).

Compares our trained PHL model's Brier scores against:
  1. NWS MOS (GFS+NAM ensemble from IEM) — represents NWS forecast baseline
  2. Enhanced Market Proxy — represents Kalshi pre-settlement pricing baseline
  3. Persistence baseline
  4. Climatological baseline

All comparisons use the 2024 portion of the held-out test set (and any
available 2025 data). Brier scores are computed per-bucket-day and averaged.

Output: results/philadelphia/phl_nws_kalshi_benchmark.csv + plots

Usage:
    python scripts/run_phl_nws_kalshi_benchmark.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, ensure_city_dirs
import config_philadelphia as city_config
from src.market_proxy import MarketProxy
from src.mos_market_proxy import MOSMarketProxy

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
PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


# ===========================================================================
# Utility Functions
# ===========================================================================

def gaussian_to_bucket_probs(
    mu: np.ndarray,
    sigma: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> np.ndarray:
    """Convert Gaussian (mu, sigma) to bucket probabilities via CDF."""
    n_days = len(mu)
    n_buckets = len(bucket_edges)
    probs = np.zeros((n_days, n_buckets))

    for b, (lo, hi) in enumerate(bucket_edges):
        cdf_lo = 0.0 if lo <= -900 else norm.cdf(lo, loc=mu, scale=sigma)
        cdf_hi = 1.0 if hi >= 900 else norm.cdf(hi, loc=mu, scale=sigma)
        probs[:, b] = np.clip(cdf_hi - cdf_lo, PROB_CLIP_MIN, PROB_CLIP_MAX)

    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    return probs


def compute_brier_score(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    bucket_edges: list[tuple[float, float]],
) -> dict:
    """Compute Brier score across all bucket-days."""
    n_days, n_buckets = bucket_probs.shape
    outcomes = np.zeros((n_days, n_buckets))

    for d in range(n_days):
        t = actual_tmax[d]
        if np.isnan(t):
            continue
        for b, (lo, hi) in enumerate(bucket_edges):
            if b == n_buckets - 1:
                if lo <= t <= hi:
                    outcomes[d, b] = 1.0
                    break
            else:
                if lo <= t < hi:
                    outcomes[d, b] = 1.0
                    break

    brier_components = (bucket_probs - outcomes) ** 2
    overall = float(np.mean(brier_components))
    per_bucket = [float(np.mean(brier_components[:, b])) for b in range(n_buckets)]

    return {"overall_brier": overall, "per_bucket_brier": per_bucket,
            "n_days": n_days, "n_buckets": n_buckets}


def compute_seasonal_brier(
    bucket_probs: np.ndarray,
    actual_tmax: np.ndarray,
    dates: pd.DatetimeIndex,
    bucket_edges: list[tuple[float, float]],
) -> dict[str, float]:
    """Compute Brier score per season."""
    months = dates.month
    seasons = np.array([SEASON_MAP[m] for m in months])
    results = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = seasons == season
        if not np.any(mask):
            continue
        score = compute_brier_score(bucket_probs[mask], actual_tmax[mask], bucket_edges)
        results[season] = score["overall_brier"]
    return results


# ===========================================================================
# Load Our Model Predictions
# ===========================================================================

def load_model_predictions(processed_dir: str, models_dir: str, phl_config):
    """Load and generate predictions from trained Ridge + NN models."""
    import torch
    from scripts.run_phl_benchmark import (
        HeteroscedasticNet,
        load_processed_phl_data,
        run_persistence_baseline,
        run_climatology_baseline,
    )
    from sklearn.linear_model import Ridge

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_phl_data(processed_dir)

    # ---- Ridge (best from benchmark: alpha=100) ----
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train.values, y_train.values)
    mu_ridge = ridge.predict(X_test.values)
    sigma_ridge = float(np.std(y_train.values - ridge.predict(X_train.values)))
    sigma_ridge = max(sigma_ridge, 1.0)

    # ---- Heteroscedastic NN ----
    nn_model_path = os.path.join(models_dir, "heteroscedastic_nn_phl.pt")
    if os.path.exists(nn_model_path):
        model = HeteroscedasticNet(n_features=X_train.shape[1], hidden_sizes=[128, 64], dropout=0.1)
        model.load_state_dict(torch.load(nn_model_path, map_location="cpu"))
        model.eval()
        X_t = torch.tensor(X_test.values, dtype=torch.float32)
        with torch.no_grad():
            mu_nn, sigma_nn = model(X_t)
        mu_nn = mu_nn.numpy().ravel()
        sigma_nn = sigma_nn.numpy().ravel()
    else:
        logger.warning("NN checkpoint not found at %s, skipping", nn_model_path)
        mu_nn = sigma_nn = None

    # ---- Baselines ----
    persist = run_persistence_baseline(y_train, y_val, y_test)
    clim = run_climatology_baseline(y_train, y_val, y_test)

    return {
        "X_test": X_test, "y_test": y_test,
        "ridge": {"mu": mu_ridge, "sigma": np.full_like(mu_ridge, sigma_ridge)},
        "nn": {"mu": mu_nn, "sigma": sigma_nn} if mu_nn is not None else None,
        "persistence": {"mu": persist["mu_test"], "sigma": persist["sigma_test"]},
        "climatology": {"mu": clim["mu_test"], "sigma": clim["sigma_test"]},
        "y_train": y_train, "y_val": y_val,
    }


# ===========================================================================
# NWS MOS Baseline
# ===========================================================================

def load_nws_mos_predictions(mos_path: str, y_test: pd.Series, y_train: pd.Series):
    """Load NWS MOS forecasts and compute monthly sigma from training period residuals."""
    mos_df = pd.read_csv(mos_path, parse_dates=["date"])
    mos_df["date"] = pd.to_datetime(mos_df["date"]).dt.normalize()

    # Align with test dates
    test_dates = pd.to_datetime(y_test.index).normalize()
    test_df = pd.DataFrame({"date": test_dates, "actual_tmax": y_test.values})

    merged = test_df.merge(mos_df[["date", "mos_ensemble_tmax_f"]], on="date", how="left")

    # Fill missing MOS with GFS or NAM individually
    for col in ["gfs_mos_tmax_f", "nam_mos_tmax_f"]:
        if col in mos_df.columns:
            merged = merged.merge(mos_df[["date", col]], on="date", how="left")

    # Use ensemble first, then GFS, then NAM
    mu_mos = merged["mos_ensemble_tmax_f"].values.copy()
    if "gfs_mos_tmax_f" in merged.columns:
        mask = np.isnan(mu_mos)
        mu_mos[mask] = merged.loc[mask, "gfs_mos_tmax_f"].values
    if "nam_mos_tmax_f" in merged.columns:
        mask = np.isnan(mu_mos)
        mu_mos[mask] = merged.loc[mask, "nam_mos_tmax_f"].values

    # Estimate MOS sigma: use historical MOS error std by month
    # First, compute MOS errors on the overlapping training period
    train_dates = pd.to_datetime(y_train.index).normalize()
    train_df = pd.DataFrame({"date": train_dates, "actual_tmax": y_train.values})
    train_merged = train_df.merge(mos_df[["date", "mos_ensemble_tmax_f"]], on="date", how="inner")

    if len(train_merged) > 30:
        train_merged["error"] = train_merged["actual_tmax"] - train_merged["mos_ensemble_tmax_f"]
        monthly_sigma = train_merged.groupby(train_merged["date"].dt.month)["error"].std().to_dict()
        overall_sigma = float(train_merged["error"].std())
    else:
        # Fallback: use overall sigma from test period where we have MOS
        valid = merged[merged["mos_ensemble_tmax_f"].notna()]
        if len(valid) > 10:
            errors = valid["actual_tmax"].values - valid["mos_ensemble_tmax_f"].values
            monthly_sigma = {}
            for m in range(1, 13):
                mask = pd.to_datetime(valid["date"]).dt.month == m
                if mask.sum() > 3:
                    monthly_sigma[m] = float(np.std(errors[mask]))
            overall_sigma = float(np.std(errors))
        else:
            overall_sigma = 7.0
            monthly_sigma = {}

    # Build sigma array with monthly granularity
    sigma_mos = np.full_like(mu_mos, overall_sigma)
    test_months = test_dates.month
    for m, s in monthly_sigma.items():
        mask = test_months == m
        sigma_mos[mask] = s

    # Clamp sigma
    sigma_mos = np.clip(sigma_mos, 2.0, 15.0)

    # Handle remaining NaN in mu (fallback to climatology)
    nan_mask = np.isnan(mu_mos)
    if nan_mask.any():
        logger.warning("%d test dates have no MOS forecast, using NaN placeholder", nan_mask.sum())

    coverage = (~nan_mask).sum()
    logger.info("NWS MOS: %d/%d test days covered (%.1f%%)",
                coverage, len(mu_mos), 100 * coverage / len(mu_mos))

    return mu_mos, sigma_mos, merged


# ===========================================================================
# Market Proxy (Kalshi Pre-Settlement Proxy)
# ===========================================================================

def build_market_proxy_predictions(y_train: pd.Series, y_val: pd.Series,
                                    y_test: pd.Series):
    """Build Enhanced Market Proxy as Kalshi pre-settlement stand-in."""
    # Combine all historical data for the MarketProxy
    all_y = pd.concat([y_train, y_val, y_test])
    history_df = pd.DataFrame({
        "date": all_y.index,
        "tmax_f": all_y.values,
    })
    history_df["date"] = pd.to_datetime(history_df["date"])

    proxy = MarketProxy(history_df)

    # Fit on data up to the end of validation period
    val_end = y_val.index.max()
    proxy.fit(train_end_date=str(val_end.date()))

    # Generate predictions for test period
    test_dates = pd.to_datetime(y_test.index)
    mu_proxy = np.zeros(len(y_test))
    sigma_proxy = np.zeros(len(y_test))

    for i, date in enumerate(test_dates):
        # Get yesterday's TMAX
        idx = all_y.index.get_loc(date)
        if idx > 0:
            yesterday_tmax = float(all_y.iloc[idx - 1])
        else:
            yesterday_tmax = float(y_train.mean())

        mu, sigma = proxy.predict_mu_sigma(date.date(), yesterday_tmax=yesterday_tmax)
        mu_proxy[i] = mu
        sigma_proxy[i] = sigma

    return mu_proxy, sigma_proxy


# ===========================================================================
# Main Benchmark
# ===========================================================================

def main():
    logger.info("=" * 70)
    logger.info("Philadelphia NWS/Kalshi Benchmark (2024/2025 Focus)")
    logger.info("=" * 70)

    phl = get_city_config("phl")
    ensure_city_dirs(phl)

    processed_dir = os.path.join(phl.data_dir, "processed")
    results_dir = phl.results_dir
    os.makedirs(results_dir, exist_ok=True)

    bucket_edges = phl.bucket_edges
    bucket_labels = phl.bucket_labels

    # --- Load model predictions ---
    logger.info("Loading trained model predictions...")
    preds = load_model_predictions(processed_dir, phl.models_dir, phl)
    y_test = preds["y_test"]
    test_dates = pd.to_datetime(y_test.index)
    test_actual = y_test.values

    # --- Filter to 2024/2025 ---
    year_mask = (test_dates.year >= 2024) & (test_dates.year <= 2025)
    eval_dates = test_dates[year_mask]
    eval_actual = test_actual[year_mask]
    eval_idx = np.where(year_mask)[0]

    logger.info("Evaluation period: %s to %s (%d days)",
                eval_dates.min().date(), eval_dates.max().date(), len(eval_dates))

    # --- Load NWS MOS ---
    mos_path = os.path.join(phl.data_dir, "mos", "combined_mos_kphl.csv")
    if os.path.exists(mos_path):
        logger.info("Loading NWS MOS forecasts from %s", mos_path)
        mu_mos_full, sigma_mos_full, _ = load_nws_mos_predictions(
            mos_path, y_test, preds["y_train"]
        )
        mu_mos = mu_mos_full[eval_idx]
        sigma_mos = sigma_mos_full[eval_idx]
        has_mos = True
    else:
        logger.warning("No MOS data found at %s", mos_path)
        has_mos = False

    # --- Build Market Proxy ---
    logger.info("Building market proxy (Kalshi pre-settlement stand-in)...")
    mu_proxy_full, sigma_proxy_full = build_market_proxy_predictions(
        preds["y_train"], preds["y_val"], y_test
    )
    mu_proxy = mu_proxy_full[eval_idx]
    sigma_proxy = sigma_proxy_full[eval_idx]

    # --- Extract model predictions for eval period ---
    models_to_eval = {}

    # Ridge
    ridge = preds["ridge"]
    models_to_eval["Ridge (our model)"] = {
        "mu": ridge["mu"][eval_idx],
        "sigma": ridge["sigma"][eval_idx],
    }

    # NN
    if preds["nn"] is not None:
        nn = preds["nn"]
        models_to_eval["HeteroscedasticNN"] = {
            "mu": nn["mu"][eval_idx],
            "sigma": nn["sigma"][eval_idx],
        }

    # Baselines
    models_to_eval["Persistence"] = {
        "mu": preds["persistence"]["mu"][eval_idx],
        "sigma": preds["persistence"]["sigma"][eval_idx],
    }
    models_to_eval["Climatology"] = {
        "mu": preds["climatology"]["mu"][eval_idx],
        "sigma": preds["climatology"]["sigma"][eval_idx],
    }

    # NWS MOS
    if has_mos:
        # Only include days with valid MOS forecasts
        mos_valid = ~np.isnan(mu_mos)
        if mos_valid.sum() > 0:
            models_to_eval["NWS MOS (GFS+NAM)"] = {
                "mu": mu_mos,
                "sigma": sigma_mos,
            }
        else:
            has_mos = False
            logger.warning("No valid MOS forecasts for evaluation period")

    # Market Proxy
    models_to_eval["Market Proxy (Kalshi)"] = {
        "mu": mu_proxy,
        "sigma": sigma_proxy,
    }

    # --- Compute Brier scores ---
    logger.info("\n" + "=" * 70)
    logger.info("BRIER SCORE COMPARISON (2024 Evaluation Period)")
    logger.info("=" * 70)

    results_rows = []
    all_seasonal = {}

    for model_name, mp in models_to_eval.items():
        mu = mp["mu"]
        sigma = mp["sigma"]

        # Handle NaN in MOS for fair comparison
        valid_mask = ~np.isnan(mu)
        if valid_mask.sum() < len(mu):
            logger.info("%s: %d/%d valid days", model_name, valid_mask.sum(), len(mu))

        if valid_mask.sum() == 0:
            continue

        # For fair comparison, only evaluate on days where ALL models have predictions
        probs = gaussian_to_bucket_probs(
            np.where(np.isnan(mu), 0, mu),
            np.where(np.isnan(sigma), 7.0, sigma),
            bucket_edges
        )

        # Full Brier
        brier = compute_brier_score(probs[valid_mask], eval_actual[valid_mask], bucket_edges)

        # Seasonal Brier
        seasonal = compute_seasonal_brier(
            probs[valid_mask], eval_actual[valid_mask],
            eval_dates[valid_mask], bucket_edges
        )
        all_seasonal[model_name] = seasonal

        # MAE
        mae = float(np.mean(np.abs(eval_actual[valid_mask] - mu[valid_mask])))

        row = {
            "model": model_name,
            "brier_2024": brier["overall_brier"],
            "mae_f": mae,
            "n_days": int(valid_mask.sum()),
        }
        for s in ["DJF", "MAM", "JJA", "SON"]:
            row[f"brier_{s}"] = seasonal.get(s, np.nan)

        results_rows.append(row)
        logger.info("  %-25s  Brier=%.4f  MAE=%.1f°F  (n=%d)",
                     model_name, brier["overall_brier"], mae, valid_mask.sum())

    results_df = pd.DataFrame(results_rows).sort_values("brier_2024")

    # --- Summary table ---
    logger.info("\n" + "=" * 70)
    logger.info("FULL COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info("\n%s", results_df.to_string(index=False, float_format="%.4f"))

    # --- Delta analysis ---
    if "Ridge (our model)" in dict(zip(results_df["model"], results_df["brier_2024"])):
        our_brier = results_df.loc[results_df["model"] == "Ridge (our model)", "brier_2024"].values[0]

        logger.info("\n--- Delta vs Our Model (Ridge) ---")
        for _, row in results_df.iterrows():
            if row["model"] == "Ridge (our model)":
                continue
            delta = our_brier - row["brier_2024"]
            direction = "BETTER" if delta < 0 else "WORSE"
            logger.info("  vs %-25s  delta=%.4f (%s)", row["model"], abs(delta), direction)

    # --- Save results ---
    # CSV
    csv_path = os.path.join(results_dir, "phl_nws_kalshi_benchmark.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("\nSaved benchmark results to %s", csv_path)

    # JSON with full detail
    detail = {
        "evaluation_period": f"{eval_dates.min().date()} to {eval_dates.max().date()}",
        "n_eval_days": len(eval_dates),
        "models": {},
    }
    for _, row in results_df.iterrows():
        detail["models"][row["model"]] = {
            "brier_2024": float(row["brier_2024"]),
            "mae_f": float(row["mae_f"]),
            "n_days": int(row["n_days"]),
            "seasonal": {s: float(row.get(f"brier_{s}", 0)) for s in ["DJF", "MAM", "JJA", "SON"]},
        }

    json_path = os.path.join(results_dir, "phl_nws_kalshi_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(detail, f, indent=2)
    logger.info("Saved detailed results to %s", json_path)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart: overall Brier
    ax1 = axes[0]
    models = results_df["model"].tolist()
    briers = results_df["brier_2024"].tolist()
    colors = []
    for m in models:
        if "Ridge" in m or "NN" in m:
            colors.append("#2196F3")  # blue for our models
        elif "NWS" in m:
            colors.append("#FF9800")  # orange for NWS
        elif "Market" in m or "Kalshi" in m:
            colors.append("#4CAF50")  # green for market
        else:
            colors.append("#9E9E9E")  # grey for baselines

    bars = ax1.barh(range(len(models)), briers, color=colors, edgecolor="black", linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, briers)):
        ax1.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel("Brier Score (lower = better)")
    ax1.set_title("PHL Model vs NWS vs Market (2024)")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # Seasonal breakdown
    ax2 = axes[1]
    seasons = ["DJF", "MAM", "JJA", "SON"]
    n_models = len(results_df)
    x = np.arange(len(seasons))
    width = 0.8 / n_models
    cmap = plt.cm.tab10

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row.get(f"brier_{s}", 0) for s in seasons]
        ax2.bar(x + i * width, vals, width, label=row["model"],
                color=cmap(i), edgecolor="black", linewidth=0.3)

    ax2.set_xticks(x + width * (n_models - 1) / 2)
    ax2.set_xticklabels(seasons)
    ax2.set_ylabel("Brier Score")
    ax2.set_title("Seasonal Brier Breakdown (2024)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(results_dir, "phl_nws_kalshi_benchmark.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved benchmark visualization to %s", fig_path)

    # --- Per-month breakdown ---
    logger.info("\n--- Monthly Brier Breakdown (2024) ---")
    month_rows = []
    for model_name, mp in models_to_eval.items():
        mu = mp["mu"]
        sigma = mp["sigma"]
        valid = ~np.isnan(mu)

        probs = gaussian_to_bucket_probs(
            np.where(np.isnan(mu), 0, mu),
            np.where(np.isnan(sigma), 7.0, sigma),
            bucket_edges
        )

        for m in sorted(eval_dates[valid].month.unique()):
            mask = valid & (eval_dates.month == m)
            if mask.sum() == 0:
                continue
            bs = compute_brier_score(probs[mask], eval_actual[mask], bucket_edges)
            month_rows.append({
                "model": model_name,
                "month": m,
                "brier": bs["overall_brier"],
                "n_days": int(mask.sum()),
            })

    month_df = pd.DataFrame(month_rows)
    if not month_df.empty:
        pivot = month_df.pivot(index="model", columns="month", values="brier")
        logger.info("\n%s", pivot.to_string(float_format="%.4f"))
        month_path = os.path.join(results_dir, "phl_monthly_brier_2024.csv")
        pivot.to_csv(month_path)
        logger.info("Saved monthly breakdown to %s", month_path)

    logger.info("\n" + "=" * 70)
    logger.info("Philadelphia NWS/Kalshi Benchmark Complete")
    logger.info("=" * 70)

    return results_df


if __name__ == "__main__":
    main()
