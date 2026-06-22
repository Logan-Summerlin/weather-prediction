"""Fall frontal-passage features (Phase 2 deliverable #7, Philadelphia SON).

Philadelphia's autumn (SON) errors are dominated by cold-frontal passages: a
front sweeps through, the wind veers to the NW, dewpoint drops sharply, and
pressure rises — the next day's high is set by a drier, cooler post-frontal
airmass that a lag-only model misses.  These helpers turn ASOS daily
aggregates into cutoff-safe predictors of that regime.

**Cutoff safety:** every feature here is meant to be built from data available
by the 7am ET inference cutoff — i.e. the *prior* day's daily aggregates and
overnight obs.  The builder lags all inputs by one day so day D's features use
only D-1 (and earlier) observations.  Callers must not feed same-day daytime
values into live inference.

The raw ASOS daily schema (see ``data/<city>/processed/asos_daily/*.csv``):
``tmean_f, dewpoint_mean_f, dewpoint_afternoon_f, wind_dir_mean_deg,
wind_dir_evening_deg, wind_speed_mean_mph, slp_tendency_24h_mb`` …
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 8-point compass sector centers (meteorological degrees, wind FROM direction).
COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# Post-frontal cold-air-advection sectors in the mid-Atlantic: NW through N.
POST_FRONTAL_SECTORS = {"NW", "N", "W"}


def dewpoint_depression(temp_f, dewpoint_f) -> np.ndarray:
    """Dewpoint depression (T - Td) in deg F; larger => drier air."""
    return np.asarray(temp_f, dtype=float) - np.asarray(dewpoint_f, dtype=float)


def wind_direction_sector(deg, n_sectors: int = 8):
    """Map wind direction (deg) to an 8-point compass sector label.

    NaN directions map to ``None``.  Degrees are taken modulo 360 so e.g. 350
    and -10 both land in N.
    """
    deg_arr = np.asarray(deg, dtype=float)
    scalar = deg_arr.ndim == 0
    deg_arr = np.atleast_1d(deg_arr)
    out = np.empty(deg_arr.shape, dtype=object)
    width = 360.0 / n_sectors
    for i, d in enumerate(deg_arr):
        if np.isnan(d):
            out[i] = None
            continue
        idx = int((d % 360.0 + width / 2) // width) % n_sectors
        out[i] = COMPASS_8[idx] if n_sectors == 8 else idx
    return out[0] if scalar else out


def is_post_frontal_wind(deg) -> np.ndarray:
    """Boolean: wind direction is in the post-frontal (NW/N/W) sector."""
    sectors = wind_direction_sector(deg)
    sectors = np.atleast_1d(sectors)
    return np.array([s in POST_FRONTAL_SECTORS for s in sectors])


def frontal_passage_index(
    dewpoint_depression_change,
    slp_tendency_24h,
    wind_deg,
    depression_threshold: float = 3.0,
    slp_threshold: float = 1.0,
) -> np.ndarray:
    """A 0-3 frontal-passage score combining the three classic signals.

    +1 if dewpoint depression jumped (air dried out),
    +1 if pressure is rising (slp_tendency_24h > threshold),
    +1 if the wind is post-frontal (NW/N/W).  A value of 2-3 indicates a likely
    cold-frontal passage in the last 24h.
    """
    dd = np.asarray(dewpoint_depression_change, dtype=float)
    slp = np.asarray(slp_tendency_24h, dtype=float)
    score = np.zeros(dd.shape, dtype=float)
    score += (dd > depression_threshold).astype(float)
    score += (slp > slp_threshold).astype(float)
    score += is_post_frontal_wind(wind_deg).astype(float)
    return score


def build_frontal_features(daily: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Assemble a cutoff-safe frontal-passage feature frame from ASOS daily rows.

    Parameters
    ----------
    daily : DataFrame
        ASOS daily aggregates indexed (or sortable) by date, with at least
        ``tmean_f`` and ``dewpoint_mean_f``; optionally ``wind_dir_evening_deg``
        / ``wind_dir_mean_deg`` and ``slp_tendency_24h_mb``.
    lag : int
        Number of days to lag every feature so day D uses only D-1 data
        (default 1).  Set 0 only for offline analysis, never live inference.

    Returns
    -------
    DataFrame indexed like *daily* with columns:
        ``dewpoint_depression``, ``dewpoint_depression_trend``,
        ``wind_post_frontal``, ``slp_tendency_24h``, ``frontal_passage_index``.
    """
    df = daily.copy()
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["date"]))
    df = df.sort_index()

    temp = df.get("tmean_f")
    dew = df.get("dewpoint_mean_f")
    if temp is None or dew is None:
        raise ValueError("daily frame must contain tmean_f and dewpoint_mean_f")

    depression = dewpoint_depression(temp.to_numpy(), dew.to_numpy())
    depression = pd.Series(depression, index=df.index)
    trend = depression.diff()  # vs prior day

    wind_col = "wind_dir_evening_deg" if "wind_dir_evening_deg" in df else "wind_dir_mean_deg"
    wind_deg = df[wind_col] if wind_col in df else pd.Series(np.nan, index=df.index)
    post_frontal = pd.Series(is_post_frontal_wind(wind_deg.to_numpy()).astype(float), index=df.index)

    slp = df.get("slp_tendency_24h_mb", pd.Series(np.nan, index=df.index))

    fpi = frontal_passage_index(trend.fillna(0.0).to_numpy(), slp.fillna(0.0).to_numpy(),
                                wind_deg.to_numpy())

    out = pd.DataFrame(
        {
            "dewpoint_depression": depression,
            "dewpoint_depression_trend": trend,
            "wind_post_frontal": post_frontal,
            "slp_tendency_24h": slp,
            "frontal_passage_index": pd.Series(fpi, index=df.index),
        },
        index=df.index,
    )
    if lag:
        out = out.shift(lag)  # cutoff safety: day D sees only D-1 and earlier
    return out
