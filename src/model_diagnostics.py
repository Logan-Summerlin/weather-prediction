"""Model diagnostics: residual bias, sigma calibration, PIT, and model-vs-market.

Phase 2 deliverable #1.  Pure-logic helpers (no I/O, no plotting) that the
``scripts/run_model_diagnostics.py`` driver composes into a per-city
diagnostics bundle written under ``results/<city>/diagnostics/``.

Every probabilistic helper operates on Gaussian daily forecasts
``(date, mu, sigma, actual_tmax)``.  Contract-level Brier always routes
through :mod:`src.bucket_semantics` (settlement-rounding-aware) so the
diagnostics use the same edge arithmetic as the live trading path.

The four diagnostic families mirror the plan:

* **residual bias** by season and temperature regime — exposes a model that
  is systematically warm/cold in a slice that aggregate MAE hides.
* **sigma calibration** — predicted sigma vs realized error; a ratio far from
  1.0 means the distribution width is mis-specified (over/under-confident).
* **PIT** — probability integral transform uniformity (reuses
  :mod:`src.calibration`); non-uniform PIT is the canonical calibration
  failure.
* **model-vs-market** — per-bucket model Brier vs market Brier on real Kalshi
  presettlement contract rows, plus a per-day disagreement table.  This is the
  no-go arbiter: model Brier >= market Brier means MONITOR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.bucket_semantics import bucket_outcome, bucket_prob_gaussian
from src.calibration import compute_crps, compute_pit_values, pit_uniformity_test
from src.seasons import SEASON_MAP_SHORT

# Priority order for auto-selecting a probabilistic model from a
# base_predictions.csv that may contain several model families.  A genuinely
# distributional (heteroscedastic) model is preferred over constant-sigma
# baselines so PIT/sigma diagnostics are meaningful.
MODEL_PRIORITY = (
    "HeteroscedasticNN",
    "ridge_base",
    "Ridge",
    "Climatology",
    "Persistence",
)

PROB_CLIP_MIN = 1e-4
PROB_CLIP_MAX = 1.0 - 1e-4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def select_model(available: list[str]) -> str:
    """Pick the representative probabilistic model from those present.

    Falls back to the first available name (sorted) when none match the
    priority list, so the helper never raises on an unexpected registry.
    """
    for name in MODEL_PRIORITY:
        if name in available:
            return name
    if not available:
        raise ValueError("No models available in predictions frame.")
    return sorted(available)[0]


def load_base_predictions(
    path: Path, model: Optional[str] = None
) -> tuple[pd.DataFrame, str]:
    """Load ``base_predictions.csv`` and return daily Gaussian forecasts.

    Returns a frame with columns ``date, mu, sigma, actual_tmax`` (one row
    per day, sorted, de-duplicated) and the resolved model name.
    """
    df = pd.read_csv(path)
    required = {"date", "mu", "sigma", "actual_tmax"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    if "model_name" in df.columns:
        available = sorted(df["model_name"].dropna().unique().tolist())
        resolved = model or select_model(available)
        if resolved not in available:
            raise ValueError(
                f"Model {resolved!r} not in {path} (available: {available})"
            )
        df = df[df["model_name"] == resolved].copy()
    else:
        resolved = model or "model"

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["mu", "sigma", "actual_tmax"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "mu", "sigma", "actual_tmax"]], resolved


# ---------------------------------------------------------------------------
# Residual bias
# ---------------------------------------------------------------------------
def _slice_stats(residual: np.ndarray) -> dict:
    if residual.size == 0:
        return {"n": 0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return {
        "n": int(residual.size),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
    }


def residual_diagnostics(df: pd.DataFrame, n_regimes: int = 3) -> dict:
    """Residual (actual - mu) bias overall, by season, and by temperature regime.

    The temperature regime bins on the *forecast* mu (terciles by default) so
    the diagnostic answers "is the model biased on cold/normal/hot days?"
    without leaking the outcome.
    """
    residual = (df["actual_tmax"] - df["mu"]).to_numpy(dtype=float)
    dates = pd.DatetimeIndex(df["date"])
    out: dict = {"overall": _slice_stats(residual)}

    by_season = {}
    seasons = np.array([SEASON_MAP_SHORT[m] for m in dates.month])
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        mask = seasons == s
        if mask.any():
            by_season[s] = _slice_stats(residual[mask])
    out["by_season"] = by_season

    by_regime = {}
    mu = df["mu"].to_numpy(dtype=float)
    if mu.size >= n_regimes:
        # Quantile edges on mu; guard against duplicate edges (degenerate data).
        qs = np.linspace(0, 1, n_regimes + 1)
        edges = np.quantile(mu, qs)
        edges[0], edges[-1] = -np.inf, np.inf
        labels = ["cold", "normal", "hot"] if n_regimes == 3 else [
            f"regime_{i}" for i in range(n_regimes)
        ]
        bins = np.digitize(mu, edges[1:-1])
        for i, label in enumerate(labels):
            mask = bins == i
            if mask.any():
                stats = _slice_stats(residual[mask])
                stats["mu_range"] = [
                    float(mu[mask].min()),
                    float(mu[mask].max()),
                ]
                by_regime[label] = stats
    out["by_regime"] = by_regime
    return out


# ---------------------------------------------------------------------------
# Sigma calibration
# ---------------------------------------------------------------------------
def sigma_calibration(df: pd.DataFrame) -> dict:
    """Compare predicted sigma to realized error.

    For a calibrated Gaussian the realized RMSE equals the mean predicted
    sigma, so ``calibration_ratio = realized_rmse / mean_sigma`` should be
    ~1.0.  >1 means under-dispersed (over-confident); <1 means over-dispersed.
    """
    mu = df["mu"].to_numpy(dtype=float)
    sigma = np.maximum(df["sigma"].to_numpy(dtype=float), 1e-9)
    actual = df["actual_tmax"].to_numpy(dtype=float)
    abs_err = np.abs(actual - mu)
    realized_rmse = float(np.sqrt(np.mean((actual - mu) ** 2)))
    mean_sigma = float(np.mean(sigma))

    # Expected |error| for a calibrated Gaussian is sigma*sqrt(2/pi).
    expected_abs = sigma * np.sqrt(2.0 / np.pi)
    out = {
        "mean_sigma": mean_sigma,
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
        "sigma_unique": int(np.unique(np.round(sigma, 6)).size),
        "realized_rmse": realized_rmse,
        "calibration_ratio": realized_rmse / mean_sigma if mean_sigma > 0 else float("nan"),
        "mean_abs_error": float(np.mean(abs_err)),
        "mean_expected_abs_error": float(np.mean(expected_abs)),
        "constant_sigma": bool(np.unique(np.round(sigma, 6)).size == 1),
    }
    # A grossly inflated constant sigma is the Austin convergence pathology.
    out["sigma_pathology"] = bool(out["constant_sigma"] and mean_sigma > 20.0)
    return out


# ---------------------------------------------------------------------------
# PIT
# ---------------------------------------------------------------------------
def pit_diagnostics(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """PIT uniformity overall and per season (reuses :mod:`src.calibration`)."""
    pit = compute_pit_values(df["mu"], df["sigma"], df["actual_tmax"])
    overall = pit_uniformity_test(pit)
    overall["mean"] = float(np.mean(pit)) if pit.size else float("nan")
    overall["std"] = float(np.std(pit)) if pit.size else float("nan")
    overall["n"] = int(pit.size)

    dates = pd.DatetimeIndex(df["date"])
    seasons = np.array([SEASON_MAP_SHORT[m] for m in dates.month])
    by_season = {}
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        mask = seasons == s
        if mask.sum() >= 10:
            season_test = pit_uniformity_test(pit[mask])
            season_test["mean"] = float(np.mean(pit[mask]))
            season_test["n"] = int(mask.sum())
            by_season[s] = season_test

    crps = compute_crps(df["mu"], df["sigma"], df["actual_tmax"])
    return {
        "overall": overall,
        "by_season": by_season,
        "mean_crps": float(crps["mean_crps"]),
    }, pit


# ---------------------------------------------------------------------------
# Model vs market
# ---------------------------------------------------------------------------
def load_presettlement(path: Path) -> pd.DataFrame:
    """Load a Kalshi presettlement CSV with normalized dates."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def model_vs_market(
    pred_df: pd.DataFrame,
    contracts: pd.DataFrame,
    oos_start: Optional[pd.Timestamp] = None,
) -> dict:
    """Per-bucket model Brier vs market Brier on real presettlement rows.

    Joins daily Gaussian forecasts to contract rows on ``date``, prices each
    contract with the settlement-rounding-aware Gaussian probability, and
    compares to ``presettlement_prob``.  Returns aggregate Brier for model and
    market plus a per-strike-type breakdown and the no-go verdict.
    """
    mu_by_date = pred_df.set_index("date")["mu"]
    sigma_by_date = pred_df.set_index("date")["sigma"]

    rows = contracts[contracts["date"].isin(pred_df["date"])].copy()
    if oos_start is not None:
        rows = rows[rows["date"] >= oos_start]
    rows = rows.dropna(subset=["presettlement_prob"])
    if rows.empty:
        return {"n_contracts": 0, "n_days": 0, "verdict": "INSUFFICIENT_DATA"}

    mu = rows["date"].map(mu_by_date).to_numpy(dtype=float)
    sigma = rows["date"].map(sigma_by_date).to_numpy(dtype=float)

    model_prob = bucket_prob_gaussian(
        mu, sigma,
        rows["threshold_low"].to_numpy(dtype=float),
        rows["threshold_high"].to_numpy(dtype=float),
        rows["direction"].to_numpy(dtype=object),
    )
    model_prob = np.clip(model_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
    market_prob = np.clip(
        rows["presettlement_prob"].to_numpy(dtype=float), PROB_CLIP_MIN, PROB_CLIP_MAX
    )

    if "actual_outcome" in rows.columns and rows["actual_outcome"].notna().all():
        outcome = rows["actual_outcome"].to_numpy(dtype=float)
    else:
        actual = rows["date"].map(pred_df.set_index("date")["actual_tmax"]).to_numpy(dtype=float)
        outcome = bucket_outcome(
            actual,
            rows["threshold_low"].to_numpy(dtype=float),
            rows["threshold_high"].to_numpy(dtype=float),
            rows["direction"].to_numpy(dtype=object),
        ).astype(float)

    model_brier = float(np.mean((model_prob - outcome) ** 2))
    market_brier = float(np.mean((market_prob - outcome) ** 2))

    by_type = {}
    if "strike_type" in rows.columns:
        for st, grp_idx in rows.groupby("strike_type").groups.items():
            idx = rows.index.get_indexer(grp_idx)
            by_type[str(st)] = {
                "n": int(len(idx)),
                "model_brier": float(np.mean((model_prob[idx] - outcome[idx]) ** 2)),
                "market_brier": float(np.mean((market_prob[idx] - outcome[idx]) ** 2)),
            }

    disagreement = float(np.mean(np.abs(model_prob - market_prob)))
    return {
        "n_contracts": int(len(rows)),
        "n_days": int(rows["date"].nunique()),
        "date_min": str(rows["date"].min().date()),
        "date_max": str(rows["date"].max().date()),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "brier_edge": market_brier - model_brier,  # positive => model beats market
        "mean_abs_disagreement": disagreement,
        "by_strike_type": by_type,
        "verdict": "BEATS_MARKET" if model_brier < market_brier else "NO_EDGE_MONITOR",
    }


def disagreement_table(
    pred_df: pd.DataFrame,
    contracts: pd.DataFrame,
    oos_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Per-contract model vs market probability table (for inspection/plots)."""
    mu_by_date = pred_df.set_index("date")["mu"]
    sigma_by_date = pred_df.set_index("date")["sigma"]
    actual_by_date = pred_df.set_index("date")["actual_tmax"]

    rows = contracts[contracts["date"].isin(pred_df["date"])].copy()
    if oos_start is not None:
        rows = rows[rows["date"] >= oos_start]
    rows = rows.dropna(subset=["presettlement_prob"])
    if rows.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "model_prob", "market_prob", "disagreement", "outcome"]
        )

    mu = rows["date"].map(mu_by_date).to_numpy(dtype=float)
    sigma = rows["date"].map(sigma_by_date).to_numpy(dtype=float)
    model_prob = np.clip(
        bucket_prob_gaussian(
            mu, sigma,
            rows["threshold_low"].to_numpy(dtype=float),
            rows["threshold_high"].to_numpy(dtype=float),
            rows["direction"].to_numpy(dtype=object),
        ),
        PROB_CLIP_MIN, PROB_CLIP_MAX,
    )
    actual = rows["date"].map(actual_by_date).to_numpy(dtype=float)
    outcome = bucket_outcome(
        actual,
        rows["threshold_low"].to_numpy(dtype=float),
        rows["threshold_high"].to_numpy(dtype=float),
        rows["direction"].to_numpy(dtype=object),
    )
    out = pd.DataFrame(
        {
            "date": rows["date"].values,
            "ticker": rows.get("ticker", pd.Series(["?"] * len(rows))).values,
            "model_prob": model_prob,
            "market_prob": rows["presettlement_prob"].to_numpy(dtype=float),
            "outcome": outcome,
        }
    )
    out["disagreement"] = out["model_prob"] - out["market_prob"]
    return out.sort_values("date").reset_index(drop=True)
