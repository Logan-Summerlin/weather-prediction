"""Market-debias model (U10): longshot-discount + book-coherence pricing.

Motivation
----------
NYC's KXHIGHNY pre-settlement market (last candle before ~05:30 UTC on the
event day, i.e. ~midnight ET — strictly before the 7am ET model cutoff) is
sharper than every weather-model variant in this repo: the optimal
model/market blend weight is ~0 in every liquidity and season slice.  The
residual, *stable* inefficiency is microstructural, not meteorological:

1. **Favorite-longshot bias.**  Buckets priced below ~20c systematically
   over-price the longshot.  The realized-frequency/price ratio for
   k <= 0.10 is ~0.45-0.5 in 2023, 2024 AND 2025 — a structural bias of
   retail flow, not a regime artifact.
2. **Book overround.**  Complete 5-6 bucket books sum to ~1.06 at the mid;
   coherence requires the true probabilities to sum to ~1.

U10 prices a bucket as: isotonic longshot discount (fit on all data strictly
before the evaluation year, never *raising* a price) followed by per-day
renormalization of complete books.  It uses ONLY information available at
the 7am ET cutoff (the midnight market snapshot), so it is cutoff-safe for
live inference.

Chronology contract
-------------------
``fit_market_debias`` must only ever be given rows dated strictly before
the rows passed to ``apply_market_debias``.  The convenience driver
``walk_forward_debias`` enforces this with year-boundary refits:
2023 rows receive the raw market mid (no history to fit), 2024 rows use the
2023-fitted curve, 2025 rows the 2023-24 curve, 2026 rows the 2023-25
curve, etc.  Unlike the U4-U9 stackers (whose IS-period model inputs are
in-sample), every U10 probability is honestly out-of-sample.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROB_CLIP_MIN = 0.001
PROB_CLIP_MAX = 0.999

# Longshot zone: the discount curve is fit and applied only at/below this mid.
LS_MAX = 0.20

# Coherence renormalization guards: only renormalize complete books whose raw
# mid sum is plausibly a full partition, and cap the correction factor.
RENORM_MIN_BUCKETS = 5
RENORM_SUM_LO = 0.90
RENORM_SUM_HI = 1.40
RENORM_FACTOR_LO = 0.85
RENORM_FACTOR_HI = 1.25


@dataclass
class MarketDebiasModel:
    """Fitted longshot-discount curve (piecewise-constant isotonic fit)."""

    iso_x: list[float] = field(default_factory=list)
    iso_y: list[float] = field(default_factory=list)
    ls_max: float = LS_MAX
    n_fit_rows: int = 0
    fit_date_min: str = ""
    fit_date_max: str = ""

    def predict_longshot(self, k: np.ndarray) -> np.ndarray:
        """Discounted probability for mids in the longshot zone."""
        if not self.iso_x:
            return np.asarray(k, dtype=float)
        return np.interp(np.asarray(k, dtype=float), self.iso_x, self.iso_y)

    def to_dict(self) -> dict:
        return {
            "model": "u10_market_debias",
            "iso_x": self.iso_x,
            "iso_y": self.iso_y,
            "ls_max": self.ls_max,
            "n_fit_rows": self.n_fit_rows,
            "fit_date_min": self.fit_date_min,
            "fit_date_max": self.fit_date_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarketDebiasModel":
        return cls(
            iso_x=list(d.get("iso_x", [])),
            iso_y=list(d.get("iso_y", [])),
            ls_max=float(d.get("ls_max", LS_MAX)),
            n_fit_rows=int(d.get("n_fit_rows", 0)),
            fit_date_min=str(d.get("fit_date_min", "")),
            fit_date_max=str(d.get("fit_date_max", "")),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MarketDebiasModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit_market_debias(train: pd.DataFrame) -> MarketDebiasModel:
    """Fit the longshot-discount isotonic curve.

    Parameters
    ----------
    train : pd.DataFrame
        Historical contract rows with ``presettlement_prob``,
        ``actual_outcome`` and ``date``.  Must predate the rows the model
        will be applied to (caller's responsibility; see module docstring).
    """
    df = train.dropna(subset=["presettlement_prob", "actual_outcome"]).copy()
    k = df["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX)
    lo = df[k <= LS_MAX]
    model = MarketDebiasModel(
        n_fit_rows=int(len(lo)),
        fit_date_min=str(df["date"].min()),
        fit_date_max=str(df["date"].max()),
    )
    if len(lo) < 50:
        return model  # not enough data: identity (raw market)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(
        lo["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).values,
        lo["actual_outcome"].astype(float).values,
    )
    model.iso_x = [float(v) for v in iso.X_thresholds_]
    model.iso_y = [float(v) for v in iso.y_thresholds_]
    return model


def apply_market_debias(df: pd.DataFrame, model: MarketDebiasModel) -> pd.DataFrame:
    """Apply longshot discount + coherence renormalization.

    Returns a copy of ``df`` with two new columns:
      - ``u10_disc_prob``: longshot-discounted probability (never above the
        raw mid in the longshot zone; raw mid elsewhere).
      - ``u10_debias_prob``: after per-day renormalization of complete books.
    """
    out = df.copy()
    k = out["presettlement_prob"].clip(PROB_CLIP_MIN, PROB_CLIP_MAX).values.astype(float)

    p = k.copy()
    sel = k <= model.ls_max
    if model.iso_x:
        # Discount only: never raise a longshot above its market price.
        p[sel] = np.minimum(model.predict_longshot(k[sel]), k[sel])
    out["u10_disc_prob"] = np.clip(p, PROB_CLIP_MIN, PROB_CLIP_MAX)

    disc_sum = out.groupby("date")["u10_disc_prob"].transform("sum")
    n_buckets = out.groupby("date")["u10_disc_prob"].transform("size")
    raw_sum = out.groupby("date")["presettlement_prob"].transform("sum")
    complete = (
        (n_buckets >= RENORM_MIN_BUCKETS)
        & (raw_sum > RENORM_SUM_LO)
        & (raw_sum < RENORM_SUM_HI)
    )
    factor = np.clip(1.0 / disc_sum, RENORM_FACTOR_LO, RENORM_FACTOR_HI)
    out["u10_debias_prob"] = np.where(
        complete,
        np.clip(out["u10_disc_prob"] * factor, PROB_CLIP_MIN, PROB_CLIP_MAX),
        out["u10_disc_prob"],
    )
    return out


def walk_forward_debias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, MarketDebiasModel]]:
    """Chronological year-by-year walk-forward application of U10.

    For each calendar year present in ``df``, fits the discount curve on all
    rows of strictly earlier years and applies it to that year's rows.  The
    earliest year has no history and receives the raw market mid.

    Returns (dataframe with u10 columns, {year: fitted model}).
    """
    out = df.copy()
    out["_year"] = pd.to_datetime(out["date"]).dt.year
    pieces = []
    models: dict[int, MarketDebiasModel] = {}
    for year in sorted(out["_year"].unique()):
        cur = out[out["_year"] == year]
        hist = out[out["_year"] < year]
        model = fit_market_debias(hist) if len(hist) else MarketDebiasModel()
        models[year] = model
        pieces.append(apply_market_debias(cur, model))
    result = pd.concat(pieces).sort_index().drop(columns=["_year"])
    return result, models


# ---------------------------------------------------------------------------
# Trading fee helper (kept local to avoid importing heavy trading deps here)
# ---------------------------------------------------------------------------

def kalshi_fee(price: float) -> float:
    """Kalshi general-markets fee per contract: ceil(0.07 * P * (1-P))."""
    if not (0.0 < price < 1.0):
        return 0.0
    return math.ceil(7.0 * price * (1.0 - price)) / 100.0
