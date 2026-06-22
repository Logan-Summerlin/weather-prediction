"""
Feature Distribution Parity Verification.

Compares training feature distributions against inference-time feature
distributions to verify that the ASOS migration has resolved the
training/inference mismatch. Uses KS tests and summary statistics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ParityResult:
    """Result of a single feature distribution comparison."""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    train_mean: float
    train_std: float
    reference_mean: float
    reference_std: float
    mean_diff: float
    parity_pass: bool  # True if KS p-value > threshold


def compare_feature_distributions(
    train_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    ks_threshold: float = 0.05,
) -> list[ParityResult]:
    """Compare distributions of matching columns between two DataFrames.

    For each column present in both DataFrames, runs a two-sample KS test
    and computes summary statistics.

    Args:
        train_features: Training feature matrix (e.g., ASOS-based features)
        reference_features: Reference feature matrix (e.g., from inference pipeline)
        ks_threshold: p-value threshold for the KS test. Features with
            p-value > threshold pass the parity check.

    Returns:
        List of ParityResult for each compared feature.
    """
    common_cols = sorted(
        set(train_features.columns) & set(reference_features.columns)
    )
    if not common_cols:
        logger.warning("No common columns found between train and reference features.")
        return []

    results: list[ParityResult] = []
    for col in common_cols:
        train_vals = train_features[col].dropna().values
        ref_vals = reference_features[col].dropna().values

        if len(train_vals) < 2 or len(ref_vals) < 2:
            logger.warning(
                "Skipping feature '%s': insufficient non-null values "
                "(train=%d, reference=%d).",
                col, len(train_vals), len(ref_vals),
            )
            continue

        ks_stat, ks_pval = stats.ks_2samp(train_vals, ref_vals)

        train_mean = float(np.mean(train_vals))
        train_std = float(np.std(train_vals, ddof=1))
        ref_mean = float(np.mean(ref_vals))
        ref_std = float(np.std(ref_vals, ddof=1))
        mean_diff = train_mean - ref_mean

        results.append(
            ParityResult(
                feature_name=col,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval),
                train_mean=train_mean,
                train_std=train_std,
                reference_mean=ref_mean,
                reference_std=ref_std,
                mean_diff=mean_diff,
                parity_pass=ks_pval > ks_threshold,
            )
        )

    n_pass = sum(r.parity_pass for r in results)
    n_total = len(results)
    logger.info(
        "Feature parity check: %d/%d features passed (KS threshold=%.3f).",
        n_pass, n_total, ks_threshold,
    )

    return results


def compare_asos_vs_ghcn_features(
    asos_features_path: str,
    ghcn_features_path: str,
    ks_threshold: float = 0.05,
) -> list[ParityResult]:
    """Compare ASOS-based and GHCN-based feature matrices from CSV files.

    Loads both CSVs and runs compare_feature_distributions.
    """
    logger.info(
        "Loading ASOS features from %s and GHCN features from %s.",
        asos_features_path, ghcn_features_path,
    )
    asos_df = pd.read_csv(asos_features_path)
    ghcn_df = pd.read_csv(ghcn_features_path)

    # Drop non-numeric columns (e.g., date strings) before comparison
    asos_numeric = asos_df.select_dtypes(include=[np.number])
    ghcn_numeric = ghcn_df.select_dtypes(include=[np.number])

    logger.info(
        "ASOS shape: %s (%d numeric cols), GHCN shape: %s (%d numeric cols).",
        asos_df.shape, len(asos_numeric.columns),
        ghcn_df.shape, len(ghcn_numeric.columns),
    )

    return compare_feature_distributions(
        train_features=asos_numeric,
        reference_features=ghcn_numeric,
        ks_threshold=ks_threshold,
    )


def _json_default(o):
    """JSON serializer fallback for numpy scalar/array types.

    ``compare_feature_distributions`` populates ParityResult fields with numpy
    scalars (e.g. KS statistic/p-value as ``np.float64``, ``parity_pass`` as
    ``np.bool_``); these are not natively JSON serializable.
    """
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def generate_parity_report(
    results: list[ParityResult],
    output_dir: str,
    city_code: str,
) -> str:
    """Generate a parity verification report.

    Writes:
    1. JSON summary to {output_dir}/{city_code}_feature_parity.json
    2. Markdown report to {output_dir}/{city_code}_feature_parity.md

    Returns path to the markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- JSON report ---
    json_path = os.path.join(output_dir, f"{city_code}_feature_parity.json")
    n_pass = sum(r.parity_pass for r in results)
    n_fail = len(results) - n_pass

    json_payload = {
        "city_code": city_code,
        "total_features": len(results),
        "passed": int(n_pass),
        "failed": int(n_fail),
        "all_pass": bool(n_fail == 0),
        "features": [asdict(r) for r in results],
    }
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2, default=_json_default)
    logger.info("JSON parity report written to %s.", json_path)

    # --- Markdown report ---
    md_path = os.path.join(output_dir, f"{city_code}_feature_parity.md")

    lines: list[str] = []
    lines.append(f"# Feature Parity Report: {city_code.upper()}\n")
    lines.append(f"**Total features compared:** {len(results)}  ")
    lines.append(f"**Passed:** {n_pass}  ")
    lines.append(f"**Failed:** {n_fail}  ")
    overall = "PASS" if n_fail == 0 else "FAIL"
    lines.append(f"**Overall:** {overall}\n")

    # Summary table
    lines.append("## Results\n")
    lines.append(
        "| Feature | KS Stat | KS p-value | Train Mean | Ref Mean | Mean Diff | Status |"
    )
    lines.append(
        "|---------|---------|------------|------------|----------|-----------|--------|"
    )
    for r in sorted(results, key=lambda x: x.ks_pvalue):
        status = "PASS" if r.parity_pass else "**FAIL**"
        lines.append(
            f"| {r.feature_name} "
            f"| {r.ks_statistic:.4f} "
            f"| {r.ks_pvalue:.4f} "
            f"| {r.train_mean:.2f} "
            f"| {r.reference_mean:.2f} "
            f"| {r.mean_diff:+.2f} "
            f"| {status} |"
        )

    # Failing features detail
    failures = [r for r in results if not r.parity_pass]
    if failures:
        lines.append("\n## Failed Features Detail\n")
        for r in failures:
            lines.append(f"### {r.feature_name}\n")
            lines.append(f"- KS statistic: {r.ks_statistic:.4f}")
            lines.append(f"- KS p-value: {r.ks_pvalue:.6f}")
            lines.append(
                f"- Train: mean={r.train_mean:.2f}, std={r.train_std:.2f}"
            )
            lines.append(
                f"- Reference: mean={r.reference_mean:.2f}, std={r.reference_std:.2f}"
            )
            lines.append(f"- Mean difference: {r.mean_diff:+.2f}\n")

    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Markdown parity report written to %s.", md_path)

    return md_path


def verify_tmax_parity(
    asos_tmax: pd.Series,
    ghcn_tmax: pd.Series,
    max_acceptable_bias: float = 2.0,
) -> dict:
    """Verify TMAX parity between ASOS and GHCN sources.

    Checks:
    1. Mean bias < max_acceptable_bias
    2. Correlation > 0.95
    3. KS test p-value > 0.01

    Returns dict with metrics and pass/fail status.
    """
    asos_clean = asos_tmax.dropna()
    ghcn_clean = ghcn_tmax.dropna()

    # Align on shared index for correlation computation
    shared_idx = asos_clean.index.intersection(ghcn_clean.index)
    if len(shared_idx) > 1:
        asos_aligned = asos_clean.loc[shared_idx]
        ghcn_aligned = ghcn_clean.loc[shared_idx]
        correlation = float(np.corrcoef(asos_aligned.values, ghcn_aligned.values)[0, 1])
        paired_bias = float(np.mean(asos_aligned.values - ghcn_aligned.values))
    else:
        correlation = float("nan")
        paired_bias = float("nan")
        logger.warning(
            "Fewer than 2 shared index entries for TMAX correlation; "
            "falling back to unpaired mean bias."
        )

    # Unpaired mean bias (used when indices don't align)
    unpaired_bias = float(np.mean(asos_clean.values) - np.mean(ghcn_clean.values))
    mean_bias = paired_bias if not np.isnan(paired_bias) else unpaired_bias

    # KS test on full (unpaired) samples
    ks_stat, ks_pval = stats.ks_2samp(asos_clean.values, ghcn_clean.values)

    # Evaluate checks
    bias_ok = bool(abs(mean_bias) < max_acceptable_bias)
    corr_ok = bool((not np.isnan(correlation)) and correlation > 0.95)
    ks_ok = bool(ks_pval > 0.01)

    result = {
        "asos_n": int(len(asos_clean)),
        "ghcn_n": int(len(ghcn_clean)),
        "shared_n": int(len(shared_idx)) if len(shared_idx) > 1 else 0,
        "mean_bias": round(float(mean_bias), 4),
        "correlation": round(float(correlation), 4) if not np.isnan(correlation) else None,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pval), 4),
        "checks": {
            "bias_below_threshold": bias_ok,
            "correlation_above_0.95": corr_ok,
            "ks_pvalue_above_0.01": ks_ok,
        },
        "overall_pass": bias_ok and corr_ok and ks_ok,
    }

    status = "PASSED" if result["overall_pass"] else "FAILED"
    logger.info(
        "TMAX parity check %s: bias=%.2f, corr=%s, KS p=%.4f.",
        status,
        mean_bias,
        f"{correlation:.4f}" if not np.isnan(correlation) else "N/A",
        ks_pval,
    )

    return result
