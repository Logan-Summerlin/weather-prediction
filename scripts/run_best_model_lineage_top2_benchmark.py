#!/usr/bin/env python3
"""Audit experiment lineage vs canonical best model and benchmark top-2 best-model-derived variants.

This script is intentionally explicit about which probabilistic experiments are truly
constructed on top of canonical best-model predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "prediction_market_benchmark" / "top2_best_model_base"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


bench_e012 = _load_module("e012", "scripts/run_e0_e1_e2_benchmark.py")


def run() -> None:
    # 1) Ensure E0/E1/E2 benchmarks are freshly generated from canonical best-model predictions.
    bench_e012.main()

    # 2) Confirm E0 parity with previous canonical best_model_run benchmark.
    best_scores = pd.read_csv(
        ROOT / "results" / "prediction_market_benchmark" / "best_model_run" / "presettlement_brier_scores.csv"
    )
    e0_scores = pd.read_csv(
        ROOT / "results" / "prediction_market_benchmark" / "e0_e1_e2" / "E0_baseline_ensemble" / "presettlement_brier_scores.csv"
    )
    parity = best_scores.merge(e0_scores, on=["slice", "source"], suffixes=("_best", "_e0"))
    parity["abs_diff"] = (parity["brier_score_best"] - parity["brier_score_e0"]).abs()
    max_abs_diff = float(parity["abs_diff"].max())

    # 3) Rank models that are actually built on top of the canonical best model.
    # E0/E1/E2 are best-model-derived; E3-E8 in probabilistic_ensemble_experiments_v2
    # are retrained experimental models from raw features.
    e012_summary = pd.read_csv(ROOT / "results" / "prediction_market_benchmark" / "e0_e1_e2" / "e0_e1_e2_benchmark_summary.csv")
    top2 = e012_summary.sort_values("overall_model_brier").head(2).copy()
    top2.to_csv(OUT_DIR / "top2_best_model_benchmark_summary.csv", index=False)

    # 4) Export dedicated benchmark artifacts for top-2 models.
    selected_models = top2["model"].tolist()
    for model_name in selected_models:
        src = ROOT / "results" / "prediction_market_benchmark" / "e0_e1_e2" / model_name
        dst = OUT_DIR / model_name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in [
            "full_benchmark_comparison.csv",
            "presettlement_brier_scores.csv",
            "presettlement_calibration.csv",
            "trading_simulation_results.csv",
        ]:
            (dst / fname).write_bytes((src / fname).read_bytes())

    lineage = {
        "best_model_derived": ["E0_baseline_ensemble", "E1_global_isotonic", "E2_seasonal_calibration"],
        "not_best_model_derived": [
            "E3_weighted_ensemble_E4_uncertainty",
            "E5_mdn2",
            "E6_quantile",
            "E7_regularization_sweep",
            "E8_feature_pruning_sweep",
        ],
        "e0_parity_max_abs_brier_diff": max_abs_diff,
        "top2_selected": selected_models,
    }
    with open(OUT_DIR / "lineage_audit.json", "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)

    report_lines = [
        "# Best-Model Lineage Audit + Top-2 Benchmark\n",
        "## Lineage confirmation\n",
        "- Canonical best-model-derived variants: **E0, E1, E2**.",
        "- In `probabilistic_ensemble_experiments_v2`, **E3-E8 are retrained from raw features and are not built on canonical `data/best_model_predictions_*` artifacts**.",
        f"- E0 parity check vs `best_model_run`: max absolute Brier-score diff = **{max_abs_diff:.6f}**.\n",
        "## Top-2 benchmark (NWS + Kalshi PreSettlement) among best-model-derived variants\n",
        top2.to_string(index=False),
    ]
    (OUT_DIR / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(f"  - {OUT_DIR / 'top2_best_model_benchmark_summary.csv'}")
    print(f"  - {OUT_DIR / 'lineage_audit.json'}")
    print(f"  - {OUT_DIR / 'README.md'}")


if __name__ == "__main__":
    run()
