#!/usr/bin/env python3
"""Audit cross-city Brier comparability across NYC/PHL/CHI outputs.

This script enforces canonical binary contract-row Brier scoring for all cities.
If any city metadata uses a different evaluation unit, the audit fails.

Outputs:
  - results/audits/cross_city_brier_scale_audit.json
  - results/audits/cross_city_brier_scale_audit.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "audits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def binary_uniform_row_brier() -> float:
    """Mean Brier for binary event with p=0.5 and balanced outcomes."""
    return 0.25


def main() -> None:
    nyc_summary = pd.read_csv(
        ROOT / "results/prediction_market_benchmark/unified_outperformance/benchmark_summary.csv"
    )
    nyc_kalshi = pd.read_csv(ROOT / "data/real_kalshi_2023_2024.csv")

    nyc_row = nyc_summary.loc[nyc_summary["variant"] == "U7_regime_conditional"].iloc[0]
    nyc_rows_per_day = float(nyc_kalshi.groupby("date").size().mean())

    phl_meta = json.loads((ROOT / "results/philadelphia/phl_real_data_benchmark_metadata.json").read_text())
    chi_meta = json.loads((ROOT / "results/chicago/chi_real_data_benchmark_metadata.json").read_text())

    required_contract_keys = {"benchmark_unit", "kalshi_contract_rows", "kalshi_contract_dates", "best_test_brier", "best_model"}
    for city, meta in (("phl", phl_meta), ("chi", chi_meta)):
        missing = sorted(required_contract_keys - set(meta.keys()))
        if missing:
            raise RuntimeError(
                f"{city.upper()} benchmark metadata missing canonical contract-row fields: {missing}. "
                "Re-run scripts/run_real_data_benchmark.py after cross-city audit fixes."
            )

    records = [
        {
            "city": "nyc",
            "benchmark": "U7_regime_conditional",
            "evaluation_unit": "binary contract-row",
            "row_count_per_day": nyc_rows_per_day,
            "raw_reported_brier": float(nyc_row["overall_brier"]),
            "daily_aggregate_brier_proxy": float(nyc_row["overall_brier"] * nyc_rows_per_day),
            "uniform_baseline_row_brier": binary_uniform_row_brier(),
            "skill_vs_uniform": float(1.0 - (nyc_row["overall_brier"] / binary_uniform_row_brier())),
        },
        {
            "city": "phl",
            "benchmark": phl_meta["best_model"],
            "evaluation_unit": str(phl_meta.get("benchmark_unit", "unknown")),
            "row_count_per_day": float(phl_meta["kalshi_contract_rows"]) / float(phl_meta["kalshi_contract_dates"]),
            "raw_reported_brier": float(phl_meta["best_test_brier"]),
            "daily_aggregate_brier_proxy": float(
                phl_meta["best_test_brier"]
                * (float(phl_meta["kalshi_contract_rows"]) / float(phl_meta["kalshi_contract_dates"]))
            ),
            "uniform_baseline_row_brier": binary_uniform_row_brier(),
            "skill_vs_uniform": float(1.0 - (phl_meta["best_test_brier"] / binary_uniform_row_brier())),
        },
        {
            "city": "chi",
            "benchmark": chi_meta["best_model"],
            "evaluation_unit": str(chi_meta.get("benchmark_unit", "unknown")),
            "row_count_per_day": float(chi_meta["kalshi_contract_rows"]) / float(chi_meta["kalshi_contract_dates"]),
            "raw_reported_brier": float(chi_meta["best_test_brier"]),
            "daily_aggregate_brier_proxy": float(
                chi_meta["best_test_brier"]
                * (float(chi_meta["kalshi_contract_rows"]) / float(chi_meta["kalshi_contract_dates"]))
            ),
            "uniform_baseline_row_brier": binary_uniform_row_brier(),
            "skill_vs_uniform": float(1.0 - (chi_meta["best_test_brier"] / binary_uniform_row_brier())),
        },
    ]

    bad_units = [r for r in records if r["evaluation_unit"] != "binary contract-row"]
    if bad_units:
        unit_txt = ", ".join(f"{r['city']}={r['evaluation_unit']}" for r in bad_units)
        raise RuntimeError(
            "Cross-city ranking blocked: mixed/non-canonical Brier evaluation units detected. "
            f"Expected binary contract-row for all cities, got {unit_txt}."
        )

    df = pd.DataFrame(records)
    out_json = OUT_DIR / "cross_city_brier_scale_audit.json"
    out_md = OUT_DIR / "cross_city_brier_scale_audit.md"

    out_json.write_text(json.dumps(records, indent=2))

    lines = [
        "# Cross-City Brier Scale Audit",
        "",
        "Raw Brier numbers are not directly comparable when each row means something different.",
        "",
        "| city | benchmark | evaluation unit | rows/day | raw Brier | daily aggregate proxy | row-level uniform baseline | skill vs uniform |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['city']} | {r['benchmark']} | {r['evaluation_unit']} | "
            f"{r['row_count_per_day']:.3f} | {r['raw_reported_brier']:.6f} | "
            f"{r['daily_aggregate_brier_proxy']:.6f} | {r['uniform_baseline_row_brier']:.6f} | "
            f"{r['skill_vs_uniform']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- All cities are now scored on binary contract rows (baseline 0.25).",
            "- Ranking is blocked if any input reverts to bucket-day/multiclass row definitions.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")

    print(df.to_string(index=False))
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
