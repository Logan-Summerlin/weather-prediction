#!/usr/bin/env python3
"""
Per-city model diagnostics (Phase 2 deliverable #1).

Surfaces *where* a city's probabilistic model fails — the failure slices that
aggregate metrics hide — before any model-optimization work:

  - residual bias by season and temperature regime
  - predicted sigma vs realized error (over/under-confidence; constant-sigma
    pathologies such as Austin's blown-up HeteroscedasticNN)
  - PIT uniformity (overall + per season)
  - per-bucket model Brier vs market Brier on real Kalshi presettlement rows,
    plus a per-contract model-vs-market disagreement table

All artifacts land under ``results/<city>/diagnostics/``:

  - diagnostics_summary.json   (machine-readable bundle of every metric)
  - residual_bias.json
  - sigma_calibration.json
  - pit.json + pit_histogram.png
  - model_vs_market.json
  - disagreement.csv
  - diagnostics_report.md       (human-readable narrative)

Usage:
    python scripts/run_model_diagnostics.py --city aus
    python scripts/run_model_diagnostics.py --city chi --model ridge_base
    python scripts/run_model_diagnostics.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config  # noqa: E402
from src.model_diagnostics import (  # noqa: E402
    disagreement_table,
    load_base_predictions,
    load_presettlement,
    model_vs_market,
    pit_diagnostics,
    residual_diagnostics,
    sigma_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CITIES = ["chi", "phl", "atl", "aus", "nyc"]
# Cities with a base_predictions.csv produced by the unified benchmark stage.
# NYC runs a separate registry pipeline and is addressed explicitly via --city.
ALL_CITIES = ["chi", "phl", "atl", "aus"]
PRESETTLEMENT = {
    "chi": "kalshi_presettlement_chi.csv",
    "phl": "kalshi_presettlement_phl.csv",
    "atl": "kalshi_presettlement_atl.csv",
    "aus": "kalshi_presettlement_aus.csv",
    "nyc": "kalshi_presettlement.csv",
}


def _diagnostics_dir(city_code: str) -> Path:
    cfg = get_city_config(city_code)
    out = Path(cfg.results_dir) / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _plot_pit(pit: np.ndarray, save_path: Path, city_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if pit.size:
        ax.hist(pit, bins=10, range=(0, 1), edgecolor="black", alpha=0.75)
        ax.axhline(pit.size / 10.0, color="red", ls="--", label="uniform")
    ax.set_title(f"{city_name} — PIT histogram (calibrated => flat)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def _render_report(city_code: str, model: str, summary: dict) -> str:
    cfg = get_city_config(city_code)
    res = summary["residual"]["overall"]
    sig = summary["sigma"]
    pit = summary["pit"]["overall"]
    mvm = summary["model_vs_market"]

    lines = [
        f"# Model Diagnostics — {cfg.city_name} ({city_code.upper()})",
        "",
        f"- Model evaluated: **{model}**",
        f"- Forecast days: {res['n']}",
        f"- Overall residual bias (actual - mu): **{res['bias']:+.2f} F**, "
        f"MAE {res['mae']:.2f} F, RMSE {res['rmse']:.2f} F",
        f"- Sigma calibration ratio (realized_rmse / mean_sigma): "
        f"**{sig['calibration_ratio']:.2f}** "
        f"(mean sigma {sig['mean_sigma']:.2f} F; "
        f"{'CONSTANT' if sig['constant_sigma'] else 'heteroscedastic'})",
        f"- PIT uniform: **{pit.get('is_uniform')}** "
        f"(KS {pit.get('ks_statistic', float('nan')):.3f}, "
        f"p {pit.get('p_value', float('nan')):.3f})",
        f"- Mean CRPS: {summary['pit']['mean_crps']:.3f} F",
        "",
    ]

    if sig.get("sigma_pathology"):
        lines += [
            "> ⚠️ **Sigma pathology detected:** the model emits a single "
            f"constant sigma of {sig['mean_sigma']:.1f} F. This is the "
            "convergence failure where sigma absorbs the mu residual and the "
            "distribution is effectively uninformative. Retrain with the "
            "Phase 0 fix (mu head initialized at target mean; log_sigma "
            "clamped) before trusting any probability from this model.",
            "",
        ]

    lines += ["## Residual bias by season", "", "| Season | n | bias (F) | MAE | RMSE |", "|---|---|---|---|---|"]
    for s, st in summary["residual"]["by_season"].items():
        lines.append(f"| {s} | {st['n']} | {st['bias']:+.2f} | {st['mae']:.2f} | {st['rmse']:.2f} |")

    lines += ["", "## Residual bias by temperature regime (mu terciles)", "",
              "| Regime | n | bias (F) | MAE |", "|---|---|---|---|"]
    for r, st in summary["residual"]["by_regime"].items():
        lines.append(f"| {r} | {st['n']} | {st['bias']:+.2f} | {st['mae']:.2f} |")

    lines += ["", "## Model vs market (real Kalshi presettlement)", ""]
    if mvm.get("n_contracts", 0) == 0:
        lines.append("_No overlapping presettlement contract rows for evaluated days._")
    else:
        lines += [
            f"- Contracts: {mvm['n_contracts']} over {mvm['n_days']} days "
            f"({mvm['date_min']} -> {mvm['date_max']})",
            f"- **Model Brier {mvm['model_brier']:.4f}** vs "
            f"**market Brier {mvm['market_brier']:.4f}** "
            f"(edge {mvm['brier_edge']:+.4f})",
            f"- Mean abs model-vs-market disagreement: {mvm['mean_abs_disagreement']:.3f}",
            f"- Verdict: **{mvm['verdict']}**",
        ]
    lines.append("")
    return "\n".join(lines)


def run_city(
    city_code: str,
    model: Optional[str] = None,
    oos_start: Optional[str] = None,
) -> dict:
    """Compute and persist the full diagnostics bundle for one city."""
    cfg = get_city_config(city_code)
    out_dir = _diagnostics_dir(city_code)

    pred_path = Path(cfg.results_dir) / "base_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No base_predictions.csv for {city_code} at {pred_path}. "
            "Run the benchmark stage first."
        )
    pred_df, resolved_model = load_base_predictions(pred_path, model=model)
    logger.info("[%s] %d days, model=%s", city_code, len(pred_df), resolved_model)

    residual = residual_diagnostics(pred_df)
    sigma = sigma_calibration(pred_df)
    pit_summary, pit_values = pit_diagnostics(pred_df)

    oos_ts = pd.Timestamp(oos_start) if oos_start else None
    pre_path = PROJECT_ROOT / "data" / PRESETTLEMENT.get(city_code, "")
    if pre_path.name and pre_path.exists():
        contracts = load_presettlement(pre_path)
        mvm = model_vs_market(pred_df, contracts, oos_start=oos_ts)
        disagree = disagreement_table(pred_df, contracts, oos_start=oos_ts)
    else:
        logger.warning("[%s] no presettlement file (%s); skipping market comparison",
                       city_code, pre_path)
        mvm = {"n_contracts": 0, "n_days": 0, "verdict": "NO_MARKET_DATA"}
        disagree = pd.DataFrame()

    summary = {
        "city_code": city_code,
        "city_name": cfg.city_name,
        "model": resolved_model,
        "n_days": int(len(pred_df)),
        "date_min": str(pred_df["date"].min().date()),
        "date_max": str(pred_df["date"].max().date()),
        "residual": residual,
        "sigma": sigma,
        "pit": pit_summary,
        "model_vs_market": mvm,
    }

    _write_json(out_dir / "residual_bias.json", residual)
    _write_json(out_dir / "sigma_calibration.json", sigma)
    _write_json(out_dir / "pit.json", pit_summary)
    _write_json(out_dir / "model_vs_market.json", mvm)
    _write_json(out_dir / "diagnostics_summary.json", summary)
    if not disagree.empty:
        disagree.to_csv(out_dir / "disagreement.csv", index=False)
    _plot_pit(pit_values, out_dir / "pit_histogram.png", cfg.city_name)
    (out_dir / "diagnostics_report.md").write_text(
        _render_report(city_code, resolved_model, summary)
    )

    logger.info(
        "[%s] residual bias %+.2fF | sigma ratio %.2f | PIT uniform=%s | %s",
        city_code, residual["overall"]["bias"], sigma["calibration_ratio"],
        pit_summary["overall"].get("is_uniform"), mvm.get("verdict"),
    )
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=CITIES, help="City code")
    parser.add_argument("--all", action="store_true", help="Run all cities")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--oos-start", default=None,
                        help="Restrict market comparison to dates >= this (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    if not args.city and not args.all:
        parser.error("provide --city <code> or --all")

    targets = ALL_CITIES if args.all else [args.city]
    failures = []
    for c in targets:
        try:
            run_city(c, model=args.model, oos_start=args.oos_start)
        except Exception as exc:  # noqa: BLE001 — report and continue across cities
            logger.error("[%s] diagnostics failed: %s", c, exc)
            failures.append(c)
    if failures and len(failures) == len(targets):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
