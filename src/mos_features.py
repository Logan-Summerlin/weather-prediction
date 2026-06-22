"""Cutoff-safe MOS/NWP features and the MOS-residual forecaster.

Phase 2 deliverable #2 (the primary accuracy lever).  The station-lag models
cannot beat an NWP-informed market; this module brings the morning MOS guidance
into the pipeline two ways:

1. **Features** — cutoff-safe predictors derived from the MOS archive:
   * ``mos_ensemble_tmax`` — the GFS/NAM ensemble day-ahead TMAX (the morning
     run, published well before the 7am ET cutoff).
   * ``mos_climo_anomaly`` — MOS TMAX minus the day-of-year climatology (how
     anomalous the guidance is vs normal).
   * ``gfs_nam_disagreement`` — ``|GFS - NAM|`` MOS TMAX (model spread; a
     cheap proxy for forecast uncertainty / regime difficulty).

2. **Residual model** — :class:`src.advanced_model.MOSCorrectionNet` learns the
   *correction* ``TMAX - MOS_TMAX`` rather than raw TMAX, which collapses the
   target variance and lets the network focus on the station-informed
   adjustment to the NWP baseline.  This replaces the weak 6-feature recalibration
   synthesis stage.

**Cutoff safety:** the MOS forecast valid for day D is taken from the most
recent run issued the *evening before* (12Z/18Z prior day) — already encoded in
the archive's day-ahead extraction — so it is available by 7am ET on day D.
The manifest entry is ``mos_tmax_morning`` (see ``src/data_sla.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Target-station ICAO used for each city's MOS archive.
CITY_MOS_STATION = {
    "nyc": "KNYC",
    "chi": "KORD",
    "phl": "KPHL",
    "atl": "KATL",
    "aus": "KAUS",
}

#: Candidate locations for a city's combined MOS CSV (first existing wins).
CITY_MOS_PATHS = {
    "nyc": ["data/mos/combined_mos_knyc.csv", "data/airport_mos/combined_mos_knyc.csv"],
    "chi": ["data/chicago/mos/combined_mos_kord.csv", "data/mos/combined_mos_kord.csv"],
    "phl": ["data/philadelphia/mos/combined_mos_kphl.csv", "data/mos/combined_mos_kphl.csv"],
    "atl": ["data/atlanta/mos/combined_mos_katl.csv", "data/mos/combined_mos_katl.csv"],
    "aus": ["data/austin/mos/combined_mos_kaus.csv", "data/mos/combined_mos_kaus.csv"],
}


def find_mos_path(city_code: str) -> Optional[Path]:
    """Return the first existing combined-MOS CSV for a city, or None."""
    for rel in CITY_MOS_PATHS.get(city_code, []):
        p = PROJECT_ROOT / rel
        if p.exists():
            return p
    return None


def load_mos(city_code: str) -> pd.DataFrame:
    """Load a city's combined MOS archive (date-indexed, normalized)."""
    path = find_mos_path(city_code)
    if path is None:
        raise FileNotFoundError(
            f"No combined MOS CSV for {city_code}. Run "
            f"scripts/download_iem_mos_data.py --city {city_code} "
            f"(station {CITY_MOS_STATION.get(city_code, '?')})."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def doy_climatology(tmax: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    """Smoothed day-of-year TMAX climatology (indexed 1..366), train-set only."""
    df = pd.DataFrame({"tmax": np.asarray(tmax, dtype=float), "doy": dates.dayofyear})
    clim = df.groupby("doy")["tmax"].mean().reindex(np.arange(1, 367))
    clim = clim.interpolate().bfill().ffill()
    # light circular smoothing
    vals = clim.to_numpy()
    pad = np.concatenate([vals[-7:], vals, vals[:7]])
    smooth = np.convolve(pad, np.ones(15) / 15.0, mode="same")[7:-7]
    return pd.Series(smooth, index=clim.index)


def gfs_nam_disagreement(gfs: np.ndarray, nam: np.ndarray) -> np.ndarray:
    """``|GFS - NAM|`` MOS TMAX; NaN where either is missing -> 0 (no spread info)."""
    gfs = np.asarray(gfs, dtype=float)
    nam = np.asarray(nam, dtype=float)
    diff = np.abs(gfs - nam)
    return np.where(np.isnan(diff), 0.0, diff)


def build_mos_features(
    mos: pd.DataFrame, climo_by_doy: pd.Series
) -> pd.DataFrame:
    """Assemble cutoff-safe MOS features indexed by date.

    Columns: ``mos_ensemble_tmax``, ``mos_climo_anomaly``,
    ``gfs_nam_disagreement``.  Rows with no MOS ensemble value are dropped (no
    guidance available that day).
    """
    df = mos.copy()
    df = df.dropna(subset=["mos_ensemble_tmax_f"])
    dates = pd.DatetimeIndex(df["date"])
    climo = climo_by_doy.reindex(dates.dayofyear).to_numpy()
    out = pd.DataFrame(
        {
            "mos_ensemble_tmax": df["mos_ensemble_tmax_f"].to_numpy(dtype=float),
            "mos_climo_anomaly": df["mos_ensemble_tmax_f"].to_numpy(dtype=float) - climo,
            "gfs_nam_disagreement": gfs_nam_disagreement(
                df.get("gfs_mos_tmax_f", pd.Series(np.nan, index=df.index)).to_numpy(),
                df.get("nam_mos_tmax_f", pd.Series(np.nan, index=df.index)).to_numpy(),
            ),
        },
        index=dates,
    )
    return out


# ---------------------------------------------------------------------------
# MOS-residual forecaster (wraps MOSCorrectionNet)
# ---------------------------------------------------------------------------
def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for the MOS-residual model") from exc


def train_mos_residual(
    X_train, baseline_train, y_train,
    X_val, baseline_val, y_val,
    hidden_sizes=None, epochs=200, lr=1e-3, weight_decay=1e-5,
    batch_size=256, patience=30, seed=0,
):
    """Train a :class:`MOSCorrectionNet` to predict ``TMAX - MOS`` residuals.

    ``baseline_*`` is the MOS ensemble TMAX (the network's additive baseline).
    Returns the trained net (best-val weights restored).
    """
    _require_torch()
    import torch
    from src.advanced_model import MOSCorrectionNet, gaussian_crps_loss

    torch.manual_seed(seed)
    Xtr = torch.tensor(np.asarray(X_train, dtype=np.float32))
    Xva = torch.tensor(np.asarray(X_val, dtype=np.float32))
    btr = torch.tensor(np.asarray(baseline_train, dtype=np.float32)).view(-1, 1)
    bva = torch.tensor(np.asarray(baseline_val, dtype=np.float32)).view(-1, 1)
    ytr = torch.tensor(np.asarray(y_train, dtype=np.float32)).view(-1, 1)
    yva = torch.tensor(np.asarray(y_val, dtype=np.float32)).view(-1, 1)

    net = MOSCorrectionNet(n_features=Xtr.shape[1], hidden_sizes=hidden_sizes)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    def _loss(out, target):
        return gaussian_crps_loss(out["mu"], out["sigma"], target)

    n = Xtr.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            out = net(Xtr[idx], btr[idx])
            loss = _loss(out, ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val = _loss(net(Xva, bva), yva).item()
        sched.step(val)
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def predict_mos_residual(net, X, baseline) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, sigma)`` from a trained MOS-residual net."""
    _require_torch()
    import torch
    net.eval()
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    bt = torch.tensor(np.asarray(baseline, dtype=np.float32)).view(-1, 1)
    with torch.no_grad():
        out = net(Xt, bt)
    return out["mu"].view(-1).numpy(), np.maximum(out["sigma"].view(-1).numpy(), 1e-3)
