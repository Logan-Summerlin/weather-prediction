#!/usr/bin/env python3
"""Audit cross-city Brier comparability across NYC/PHL/CHI outputs.

This script demonstrates why raw Brier values from different evaluation units
(contract rows vs bucket-day matrices) are not directly comparable.

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


def multiclass_uniform_row_brier(n_classes: int) -> float:
    """Mean per-class Brier when predicting uniform 1/K distribution."""
    k = float(n_classes)
    return float((((1.0 - 1.0 / k) ** 2) + (k - 1.0) * (1.0 / k**2)) / k)


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
            "evaluation_unit": "multiclass bucket-day row",
            "row_count_per_day": int(phl_meta["n_buckets"]),
            "raw_reported_brier": float(phl_meta["best_test_brier"]),
            "daily_aggregate_brier_proxy": float(phl_meta["best_test_brier"] * phl_meta["n_buckets"]),
            "uniform_baseline_row_brier": multiclass_uniform_row_brier(int(phl_meta["n_buckets"])),
            "skill_vs_uniform": float(
                1.0
                - (
                    phl_meta["best_test_brier"]
                    / multiclass_uniform_row_brier(int(phl_meta["n_buckets"]))
                )
            ),
        },
        {
            "city": "chi",
            "benchmark": chi_meta["best_model"],
            "evaluation_unit": "multiclass bucket-day row",
            "row_count_per_day": int(chi_meta["n_buckets"]),
            "raw_reported_brier": float(chi_meta["best_test_brier"]),
            "daily_aggregate_brier_proxy": float(chi_meta["best_test_brier"] * chi_meta["n_buckets"]),
            "uniform_baseline_row_brier": multiclass_uniform_row_brier(int(chi_meta["n_buckets"])),
            "skill_vs_uniform": float(
                1.0
                - (
                    chi_meta["best_test_brier"]
                    / multiclass_uniform_row_brier(int(chi_meta["n_buckets"]))
                )
            ),
        },
    ]

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
            "- NYC row-level Brier is binary-contract scale (baseline 0.25).",
            "- PHL/CHI row-level Brier is 57/62-class scale (uniform baselines ~0.017/0.016).",
            "- Multiplying row-level Brier by rows/day produces a rough daily aggregate proxy showing NYC/PHL/CHI are all in the same ballpark (~0.68-0.84), not an order-of-magnitude apart.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")

    print(df.to_string(index=False))
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
