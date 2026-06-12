#!/usr/bin/env python3
"""
Build the honest per-city baseline ledger.

Reads the regenerated per-city artifacts (benchmark results, calibration
sweeps, real-Kalshi backtest metrics) and writes a single machine-generated
file — results/baseline_ledger.json — recording, per city:

  - OOS contract Brier and its source artifact
  - NWS baseline Brier
  - Kalshi market-implied Brier (real presettlement prices)
  - Best calibration ECE
  - Real-price backtest P&L / Sharpe / max drawdown / number of trades
  - Promotion gate pass counts

This ledger is the baseline that all later model/trading work must beat.
It is regenerated (never hand-edited) after each pipeline run:

    python scripts/build_baseline_ledger.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.city_config import get_city_config, list_cities  # noqa: E402
from src.promotion_report import (  # noqa: E402
    CITY_THRESHOLDS,
    _extract_calibration_summary,
    _extract_market_brier,
    _load_best_brier,
    _load_json,
    _load_real_kalshi_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LEDGER_PATH = PROJECT_ROOT / "results" / "baseline_ledger.json"


def build_city_entry(city_code: str) -> dict:
    """Assemble the ledger entry for one city from its result artifacts."""
    cfg = get_city_config(city_code)
    results_dir = cfg.results_dir
    thresholds = CITY_THRESHOLDS.get(city_code, {})

    best_brier, brier_source, best_model = _load_best_brier(results_dir)
    market_brier = _extract_market_brier(results_dir)
    cal_summary = _extract_calibration_summary(results_dir)
    real_metrics = _load_real_kalshi_metrics(results_dir)

    promotion = _load_json(
        os.path.join(results_dir, "unified_promotion_report.json")
    )

    entry: dict = {
        "kalshi_ticker": cfg.kalshi_ticker,
        "target_station": cfg.target_station,
        "oos_brier": best_brier if best_brier < 1.0 else None,
        "oos_brier_source": brier_source,
        "best_model": best_model,
        "nws_brier_baseline": thresholds.get("nws_brier_baseline"),
        "market_brier": market_brier,
        "beats_nws": (
            best_brier < thresholds["nws_brier_baseline"]
            if best_brier < 1.0 and thresholds.get("nws_brier_baseline") is not None
            else None
        ),
        "beats_market": (
            best_brier < market_brier
            if best_brier < 1.0 and market_brier is not None
            else None
        ),
        "ece": cal_summary.get("ece"),
        "calibration_method": cal_summary.get("best_method"),
    }

    if real_metrics is not None:
        entry["real_kalshi_backtest"] = {
            "variant": real_metrics.get("best_variant"),
            "total_pnl": real_metrics.get("total_pnl"),
            "return_pct": real_metrics.get("return_pct"),
            "sharpe_ratio": real_metrics.get("sharpe_ratio"),
            "max_drawdown_pct": real_metrics.get("max_drawdown_pct"),
            "n_trades": real_metrics.get("n_trades"),
            "win_rate": real_metrics.get("win_rate"),
            "model_brier": real_metrics.get("model_brier"),
            "market_brier": real_metrics.get("market_brier"),
            "busted": real_metrics.get("busted"),
        }
    else:
        entry["real_kalshi_backtest"] = None

    if promotion is not None:
        entry["promotion"] = {
            "overall_status": promotion.get("overall_status"),
            "gates_passed": promotion.get("gates_passed"),
            "gates_total": promotion.get("gates_total"),
            "failed_gates": [
                g["name"] for g in promotion.get("gates", []) if not g["passed"]
            ],
        }
    else:
        entry["promotion"] = None

    return entry


def main() -> dict:
    ledger = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Machine-generated honest baseline. Real-Kalshi presettlement "
            "backtests are the canonical trading evaluation; simulated-market "
            "results are excluded by design."
        ),
        "cities": {},
    }

    for city_code in sorted(list_cities()):
        try:
            ledger["cities"][city_code] = build_city_entry(city_code)
            logger.info("Ledger entry built for %s", city_code)
        except Exception as exc:  # pragma: no cover - defensive per-city isolation
            logger.error("Failed to build ledger entry for %s: %s", city_code, exc)
            ledger["cities"][city_code] = {"error": str(exc)}

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2, default=str)
    logger.info("Wrote baseline ledger to %s", LEDGER_PATH)

    return ledger


if __name__ == "__main__":
    main()
