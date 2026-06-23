"""
Daily cutoff-safe inference for one city-day (Phase 3.3).

Importable core behind ``scripts/run_daily_inference.py``.  The pipeline is:

  1. Enforce the 7am-ET cutoff manifest against the freshest available record
     for every inference feature the model consumes.  A *critical* feature that
     is absent, leaking, or stale trips the kill switch
     (:func:`src.schema_validation.enforce_inference_freshness`) — the run
     aborts and a kill-switch event is written instead of a signal.
  2. Produce the promoted model's calibrated forecast distribution (mu, sigma)
     for the market day.
  3. Convert the distribution to contract-bucket probabilities through the
     verified settlement semantics (:func:`src.bucket_semantics`).
  4. Write ``results/<city>/live/signals_<date>.json`` (consumed by the paper
     trading loop and, later, the Streamlit dashboard).

Honesty note on the forecast source: live ASOS/MOS feeds and trained
checkpoints are not present in this repository snapshot, so the default
forecast source reads the promoted model's *persisted* (mu, sigma) for the
date from ``results/<city>/synthesis/synthesis_predictions.csv``.  Callers may
inject ``mu``/``sigma`` directly (operational / test use).  The freshness gate
is the real, enforced contract regardless of forecast source.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.city_config import get_city_config
from src.data_sla import (
    get_cutoff_spec,
    latest_usable_timestamp,
    list_cutoff_features,
)
from src.live_trading import DailyPrediction, gaussian_to_bucket_probs
from src.schema_validation import (
    KillSwitchError,
    enforce_inference_freshness,
    validate_inference_freshness,
)

logger = logging.getLogger(__name__)

# Cutoff features the operational temperature model depends on.  (Soundings /
# prior-day settlement are recommended-only and degrade gracefully.)
DEFAULT_REQUIRED_FEATURES = [
    "asos_prior_day_daily",
    "asos_overnight_obs",
    "mos_tmax_morning",
]


# ===========================================================================
# Freshness inputs
# ===========================================================================

def assumed_fresh_timestamps(market_date, features: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build an 'all sources fresh' timestamp map for a backfill/replay run.

    Each feature is dated at its latest cutoff-safe instant, i.e. the manifest
    asserts it would have been available by 7am ET.  Use this only for
    historical replay; live runs must pass real observed timestamps.
    """
    features = features or list_cutoff_features()
    return {f: latest_usable_timestamp(f, market_date) for f in features}


def load_available_timestamps(path: str) -> Dict[str, Any]:
    """Load a {feature: ISO-timestamp|null} JSON map for freshness checking."""
    with open(path, "r") as f:
        raw = json.load(f)
    out: Dict[str, Any] = {}
    for feature, value in raw.items():
        out[feature] = pd.to_datetime(value, utc=True).to_pydatetime() if value else None
    return out


# ===========================================================================
# Forecast source
# ===========================================================================

def _synthesis_predictions_path(city_code: str) -> str:
    cfg = get_city_config(city_code)
    return os.path.join(cfg.results_dir, "synthesis", "synthesis_predictions.csv")


def load_promoted_forecast(
    city_code: str,
    market_date: str,
    predictions_path: Optional[str] = None,
) -> Dict[str, float]:
    """Return the promoted model's calibrated (mu, sigma) for *market_date*.

    Reads the persisted synthesis predictions (the calibrated distribution
    head).  Raises ``KeyError`` if the date is not present.
    """
    path = predictions_path or _synthesis_predictions_path(city_code)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No persisted forecast for {city_code}: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    row = df[df["date"] == str(market_date)[:10]]
    if row.empty:
        raise KeyError(f"No forecast row for {city_code} on {market_date}")
    r = row.iloc[-1]
    return {"mu": float(r["mu"]), "sigma": float(r["sigma"]),
            "model_name": "synthesis"}


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class DailySignal:
    """Result of a daily inference run for one city-day."""

    city_code: str
    date: str
    status: str  # "OK" or "KILL_SWITCH"
    mu: Optional[float] = None
    sigma: Optional[float] = None
    model_name: str = ""
    bucket_labels: List[str] = field(default_factory=list)
    bucket_probs: List[float] = field(default_factory=list)
    bucket_edges: List[List[float]] = field(default_factory=list)
    freshness: Dict[str, Any] = field(default_factory=dict)
    kill_switch_reasons: List[str] = field(default_factory=list)
    promotion_status: str = "UNKNOWN"
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city_code": self.city_code,
            "date": self.date,
            "status": self.status,
            "mu": self.mu,
            "sigma": self.sigma,
            "model_name": self.model_name,
            "bucket_labels": self.bucket_labels,
            "bucket_probs": self.bucket_probs,
            "bucket_edges": self.bucket_edges,
            "freshness": self.freshness,
            "kill_switch_reasons": self.kill_switch_reasons,
            "promotion_status": self.promotion_status,
            "generated_at": self.generated_at,
        }


def signals_path(city_code: str, market_date: str, live_dir: Optional[str] = None) -> str:
    cfg = get_city_config(city_code)
    live_dir = live_dir or os.path.join(cfg.results_dir, "live")
    return os.path.join(live_dir, f"signals_{str(market_date)[:10]}.json")


def _promotion_status(city_code: str) -> str:
    from src.strategy_selection import load_strategy_json
    data = load_strategy_json(city_code)
    if not data:
        return "UNKNOWN"
    return data.get("promotion_status", data.get("promotion", {}).get("status", "UNKNOWN"))


# ===========================================================================
# Orchestration
# ===========================================================================

def run_daily_inference(
    city_code: str,
    market_date: str,
    available_timestamps: Optional[Dict[str, Any]] = None,
    require_features: Optional[List[str]] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    predictions_path: Optional[str] = None,
    write: bool = True,
    live_dir: Optional[str] = None,
) -> DailySignal:
    """Run cutoff-safe daily inference for one city-day.

    On a critical freshness violation the kill switch trips: a ``KILL_SWITCH``
    signal is written (no forecast) and returned.  Otherwise the promoted
    forecast is converted to bucket probabilities and an ``OK`` signal is
    written.
    """
    cfg = get_city_config(city_code)
    market_date = str(market_date)[:10]
    require_features = require_features or DEFAULT_REQUIRED_FEATURES
    if available_timestamps is None:
        available_timestamps = assumed_fresh_timestamps(market_date, require_features)

    generated_at = datetime.now(timezone.utc).isoformat()

    # ---- 1. Freshness / kill switch ----
    try:
        enforce_inference_freshness(
            city_code, market_date, available_timestamps,
            require_features=require_features,
        )
        fresh_result = validate_inference_freshness(
            city_code, market_date, available_timestamps,
            require_features=require_features,
        )
    except KillSwitchError as exc:
        signal = DailySignal(
            city_code=cfg.city_code, date=market_date, status="KILL_SWITCH",
            kill_switch_reasons=list(exc.violations),
            freshness={"valid": False, "kill_switch": True},
            promotion_status=_promotion_status(city_code),
            generated_at=generated_at,
        )
        if write:
            _write_signal(signal, city_code, market_date, live_dir)
        logger.error("KILL SWITCH %s/%s: %s", city_code, market_date, exc.violations)
        return signal

    # ---- 2. Forecast (mu, sigma) ----
    model_name = "injected"
    if mu is None or sigma is None:
        fc = load_promoted_forecast(city_code, market_date, predictions_path)
        mu = float(fc["mu"]) if mu is None else mu
        sigma = float(fc["sigma"]) if sigma is None else sigma
        model_name = fc["model_name"]

    # ---- 3. Bucketize through verified settlement semantics ----
    bucket_probs = gaussian_to_bucket_probs(mu, sigma, cfg.bucket_edges)

    signal = DailySignal(
        city_code=cfg.city_code, date=market_date, status="OK",
        mu=float(mu), sigma=float(sigma), model_name=model_name,
        bucket_labels=list(cfg.bucket_labels),
        bucket_probs=[float(p) for p in bucket_probs],
        bucket_edges=[[float(lo), float(hi)] for lo, hi in cfg.bucket_edges],
        freshness={
            "valid": True,
            "kill_switch": False,
            "warnings": fresh_result.warnings,
            "degraded": bool(fresh_result.warnings),
        },
        promotion_status=_promotion_status(city_code),
        generated_at=generated_at,
    )
    if write:
        _write_signal(signal, city_code, market_date, live_dir)
    logger.info(
        "Daily inference OK %s/%s: mu=%.2f sigma=%.2f (%s)",
        city_code, market_date, mu, sigma, signal.promotion_status,
    )
    return signal


def _write_signal(signal: DailySignal, city_code: str, market_date: str, live_dir):
    path = signals_path(city_code, market_date, live_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(signal.to_dict(), f, indent=2, default=str)
    logger.info("Wrote signal to %s", path)
    return path


def load_signal(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def signal_to_prediction(signal: Dict[str, Any]) -> DailyPrediction:
    """Rehydrate a persisted OK signal into a DailyPrediction for trading."""
    return DailyPrediction(
        city_code=signal["city_code"],
        date=signal["date"],
        mu=float(signal.get("mu") or 0.0),
        sigma=float(signal.get("sigma") or 1.0),
        bucket_probs=np.asarray(signal["bucket_probs"], dtype=float),
        bucket_labels=list(signal["bucket_labels"]),
        model_name=signal.get("model_name", ""),
    )
