#!/usr/bin/env python3
"""
Philadelphia autumn (SON) calibration diagnostic (Phase 2 deliverable #7).

Two things:

1. **Seasonal / regime calibration** — fit calibrators on a chronological
   in-sample slice of PHL's base forecasts and measure per-season PIT
   uniformity on the held-out slice for: raw, global isotonic, seasonal
   isotonic, and frontal-regime-conditional. Shows whether conditioning the
   calibration on season (and on the frontal-passage regime) flattens the
   autumn PIT.

2. **Fall frontal-passage features** — build the cutoff-safe frontal feature
   frame from PHL's target-station ASOS daily aggregates and report how the
   SON-day high-temperature error depends on the post-frontal regime, which is
   the evidence for adding these features upstream.

Outputs:
  - results/philadelphia/diagnostics/seasonal_calibration.json
  - results/philadelphia/diagnostics/son_root_cause.md

Usage:
    python scripts/run_phl_son_diagnostic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (  # noqa: E402
    IsotonicCalibrator,
    RegimeConditionalCalibrator,
    compute_pit_values,
    pit_uniformity_test,
)
from src.city_config import get_city_config  # noqa: E402
from src.frontal_features import build_frontal_features  # noqa: E402
from src.model_diagnostics import load_base_predictions  # noqa: E402
from src.seasons import SEASON_MAP_SHORT  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PHL_TARGET_STATION = "USW00013739"  # Philadelphia International (KPHL)


def _season_pit(pit: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    out = {}
    seasons = np.array([SEASON_MAP_SHORT[m] for m in dates.month])
    for s in ["Winter", "Spring", "Summer", "Fall"]:
        mask = seasons == s
        if mask.sum() >= 10:
            t = pit_uniformity_test(pit[mask])
            out[s] = {"ks": t["ks_statistic"], "is_uniform": t["is_uniform"], "n": int(mask.sum())}
    return out


def main() -> int:
    cfg = get_city_config("phl")
    out_dir = Path(cfg.results_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load base forecasts ---
    pred_path = Path(cfg.results_dir) / "base_predictions.csv"
    pred, model = load_base_predictions(pred_path)
    pred = pred.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pred["date"])

    # --- Frontal features from PHL target-station ASOS daily ---
    asos_path = Path(cfg.data_dir) / "processed" / "asos_daily" / f"{PHL_TARGET_STATION}_asos_daily.csv"
    frontal = build_frontal_features(pd.read_csv(asos_path), lag=1)
    frontal.index = pd.DatetimeIndex(frontal.index).normalize()
    joined = pred.set_index("date").join(frontal, how="left")
    # Frontal regime: post-frontal (drier NW airmass) vs other, from prior day.
    regime = np.where(joined["frontal_passage_index"].fillna(0) >= 2.0, "post_frontal", "other")

    # --- Chronological split: fit on first 70%, evaluate on last 30% ---
    n = len(pred)
    cut = int(n * 0.70)
    tr = slice(0, cut)
    te = slice(cut, n)
    mu, sigma, actual = pred["mu"].to_numpy(), pred["sigma"].to_numpy(), pred["actual_tmax"].to_numpy()
    dt_te = dates[te]

    raw_pit_te = compute_pit_values(mu[te], sigma[te], actual[te])

    # Global isotonic.
    pit_tr = compute_pit_values(mu[tr], sigma[tr], actual[tr])
    g = IsotonicCalibrator(seasonal=False).fit(pit_tr)
    gmu, gsig = g.calibrate(mu[te], sigma[te])
    g_pit = compute_pit_values(gmu, gsig, actual[te])

    # Seasonal isotonic.
    s = IsotonicCalibrator(seasonal=True).fit(pit_tr, dates=dates[tr])
    smu, ssig = s.calibrate(mu[te], sigma[te], dates=dt_te)
    s_pit = compute_pit_values(smu, ssig, actual[te])

    # Frontal-regime conditional.
    r = RegimeConditionalCalibrator(min_samples=20).fit(
        mu[tr], sigma[tr], actual[tr], regime[tr]
    )
    rmu, rsig = r.calibrate(mu[te], sigma[te], regime[te])
    r_pit = compute_pit_values(rmu, rsig, actual[te])

    calibration = {
        "model": model,
        "n_total": int(n),
        "n_test": int(n - cut),
        "test_date_min": str(dt_te.min().date()),
        "test_date_max": str(dt_te.max().date()),
        "overall_ks": {
            "raw": pit_uniformity_test(raw_pit_te)["ks_statistic"],
            "global_iso": pit_uniformity_test(g_pit)["ks_statistic"],
            "seasonal_iso": pit_uniformity_test(s_pit)["ks_statistic"],
            "frontal_regime": pit_uniformity_test(r_pit)["ks_statistic"],
        },
        "fall_ks": {
            "raw": _season_pit(raw_pit_te, dt_te).get("Fall"),
            "seasonal_iso": _season_pit(s_pit, dt_te).get("Fall"),
            "frontal_regime": _season_pit(r_pit, dt_te).get("Fall"),
        },
        "regime_fit": {"regimes_fitted": [str(x) for x in r.regimes]},
    }

    # --- SON error by frontal regime (full series, evidence for the features) ---
    seasons_all = np.array([SEASON_MAP_SHORT[m] for m in dates.month])
    son_mask = seasons_all == "Fall"
    abs_err = np.abs(actual - mu)
    son_regime = regime[son_mask]
    son_err = abs_err[son_mask]
    son_evidence = {}
    for label in ["post_frontal", "other"]:
        m = son_regime == label
        if m.any():
            son_evidence[label] = {"n": int(m.sum()), "mae": float(np.mean(son_err[m]))}

    summary = {"calibration": calibration, "son_error_by_regime": son_evidence}
    (out_dir / "seasonal_calibration.json").write_text(json.dumps(summary, indent=2, default=str))

    _write_report(out_dir, summary)
    logger.info(
        "PHL SON: overall KS raw=%.3f seasonal=%.3f frontal=%.3f | SON MAE post-frontal=%s other=%s",
        calibration["overall_ks"]["raw"], calibration["overall_ks"]["seasonal_iso"],
        calibration["overall_ks"]["frontal_regime"],
        son_evidence.get("post_frontal", {}).get("mae"),
        son_evidence.get("other", {}).get("mae"),
    )
    return 0


def _write_report(out_dir: Path, summary: dict) -> None:
    c = summary["calibration"]
    ev = summary["son_error_by_regime"]
    ok = c["overall_ks"]
    lines = [
        "# Philadelphia SON Calibration Diagnostic — Phase 2 Deliverable #7",
        "",
        f"Base model: **{c['model']}**, held-out test window "
        f"{c['test_date_min']} → {c['test_date_max']} ({c['n_test']} days; "
        "calibrators fit on the earlier 70% only).",
        "",
        "## Calibration: overall PIT KS (lower = more uniform)",
        "",
        "| Calibration | Overall KS |",
        "|---|---|",
        f"| Raw model | {ok['raw']:.3f} |",
        f"| Global isotonic | {ok['global_iso']:.3f} |",
        f"| Seasonal isotonic | {ok['seasonal_iso']:.3f} |",
        f"| Frontal-regime conditional | {ok['frontal_regime']:.3f} |",
        "",
        "## Autumn (SON) PIT after calibration",
        "",
    ]
    fall = c["fall_ks"]
    if fall.get("raw"):
        lines += [
            "| Calibration | Fall KS | n |",
            "|---|---|---|",
            f"| Raw | {fall['raw']['ks']:.3f} | {fall['raw']['n']} |",
        ]
        if fall.get("seasonal_iso"):
            lines.append(f"| Seasonal isotonic | {fall['seasonal_iso']['ks']:.3f} | {fall['seasonal_iso']['n']} |")
        if fall.get("frontal_regime"):
            lines.append(f"| Frontal-regime | {fall['frontal_regime']['ks']:.3f} | {fall['frontal_regime']['n']} |")
    else:
        lines.append("_Too few held-out Fall days for a per-season KS test._")

    lines += ["", "## Evidence for frontal-passage features (full series SON)", ""]
    if ev:
        lines += ["| Regime (prior-day) | n | high-temp MAE (°F) |", "|---|---|---|"]
        for label, d in ev.items():
            lines.append(f"| {label} | {d['n']} | {d['mae']:.2f} |")
        if "post_frontal" in ev and "other" in ev:
            diff = ev["post_frontal"]["mae"] - ev["other"]["mae"]
            lines += [
                "",
                f"Post-frontal SON days carry a **{diff:+.2f} °F** MAE difference "
                "vs other days — the lag-only base model handles the drier, "
                "cooler post-frontal airmass differently, which is exactly the "
                "signal the `dewpoint_depression_trend`, `wind_post_frontal`, and "
                "`frontal_passage_index` features (src/frontal_features.py) encode "
                "for the model and the regime-conditional calibrator.",
            ]
    lines += [
        "",
        "## Recommendation",
        "",
        "- Add the cutoff-safe frontal-passage features to PHL's processed "
        "feature set (prior-day lagged: `dewpoint_depression`, "
        "`dewpoint_depression_trend`, `wind_post_frontal`, `slp_tendency_24h`, "
        "`frontal_passage_index`).",
        "- Apply seasonal (or frontal-regime) calibration via "
        "`RegimeConditionalCalibrator` so autumn PIT is corrected separately "
        "from the rest of the year.",
        "- Re-evaluate against market Brier on real OOS presettlement prices; "
        "PHL stays MONITOR until model Brier < market Brier.",
    ]
    (out_dir / "son_root_cause.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
