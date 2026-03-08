#!/usr/bin/env python3
"""
Phase E: ASOS Training Data Migration Script.

Orchestrates the full ASOS migration pipeline for expansion cities,
replacing GHCN-Daily with ASOS-derived features as the primary training
data source.  This resolves the training/inference mismatch where models
trained on GHCN but infer on ASOS-derived features.

Pipeline steps per city:
  1. Write city-specific ASOS station mapping CSV.
  2. Collect ASOS hourly data from IEM (or verify cached).
  3. Aggregate hourly -> daily features (TMAX, TMIN, dewpoint, etc.).
  4. Generate ASOS vs GHCN TMAX cross-validation report.
  5. Rebuild processed feature splits using ASOS TMAX as primary source.
  6. Verify feature distribution parity (KS test).
  7. Summarise migration results.

Usage:
    python scripts/run_asos_migration.py --city chi
    python scripts/run_asos_migration.py --city phl
    python scripts/run_asos_migration.py --city atl
    python scripts/run_asos_migration.py --city aus
    python scripts/run_asos_migration.py --city chi --skip-collection
    python scripts/run_asos_migration.py --city chi --steps 4,5,6
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, ensure_city_dirs
from src.asos_feature_builder import (
    write_city_asos_mapping_csv,
    build_asos_features,
    load_asos_daily_for_city,
)
from src.asos_preprocessing import (
    aggregate_asos_directory,
    generate_asos_ghcn_report,
    write_asos_ghcn_markdown,
)
from src.feature_parity import (
    compare_asos_vs_ghcn_features,
    generate_parity_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _city_asos_raw_dir(cfg) -> str:
    return os.path.join(cfg.data_dir, "raw", "asos")


def _city_asos_daily_dir(cfg) -> str:
    return os.path.join(cfg.data_dir, "processed", "asos_daily")


def _city_asos_mapping_csv(cfg) -> str:
    return os.path.join(cfg.data_dir, "asos_station_mapping.csv")


def _city_migration_report_dir(cfg) -> str:
    return os.path.join(cfg.results_dir, "asos_migration")


def _city_asos_processed_dir(cfg) -> str:
    return os.path.join(cfg.data_dir, "processed")


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step1_write_mapping(cfg, city_code: str) -> str:
    """Step 1: Write city-specific ASOS station mapping CSV."""
    logger.info("=" * 60)
    logger.info("Step 1: Writing ASOS station mapping for %s", city_code.upper())
    logger.info("=" * 60)

    mapping_csv = _city_asos_mapping_csv(cfg)
    write_city_asos_mapping_csv(city_code, mapping_csv)
    logger.info("Mapping CSV written to %s", mapping_csv)
    return mapping_csv


def step2_collect_asos(cfg, city_code: str, mapping_csv: str) -> None:
    """Step 2: Collect ASOS hourly data from IEM."""
    logger.info("=" * 60)
    logger.info("Step 2: Collecting ASOS hourly data for %s", city_code.upper())
    logger.info("=" * 60)

    from src.asos_collection import collect_asos_data

    asos_raw_dir = _city_asos_raw_dir(cfg)
    os.makedirs(asos_raw_dir, exist_ok=True)

    # Use city config date range, falling back to project defaults
    start_date = cfg.start_date or "1998-01-01"
    end_date = cfg.end_date or "2024-12-31"

    results = collect_asos_data(
        mapping_csv=mapping_csv,
        output_dir=asos_raw_dir,
        start_date=start_date,
        end_date=end_date,
        chunk_years=1,
    )
    logger.info(
        "ASOS collection complete: %d stations downloaded to %s",
        len(results), asos_raw_dir,
    )


def step3_aggregate_daily(cfg, city_code: str, mapping_csv: str) -> None:
    """Step 3: Aggregate ASOS hourly -> daily features."""
    logger.info("=" * 60)
    logger.info("Step 3: Aggregating ASOS hourly -> daily for %s", city_code.upper())
    logger.info("=" * 60)

    asos_raw_dir = _city_asos_raw_dir(cfg)
    asos_daily_dir = _city_asos_daily_dir(cfg)
    os.makedirs(asos_daily_dir, exist_ok=True)

    outputs = aggregate_asos_directory(
        mapping_csv=mapping_csv,
        input_dir=asos_raw_dir,
        output_dir=asos_daily_dir,
    )
    logger.info(
        "Daily aggregation complete: %d stations -> %s",
        len(outputs), asos_daily_dir,
    )


def step4_ghcn_crossval(cfg, city_code: str, mapping_csv: str) -> None:
    """Step 4: Generate ASOS vs GHCN TMAX cross-validation report."""
    logger.info("=" * 60)
    logger.info("Step 4: ASOS vs GHCN cross-validation for %s", city_code.upper())
    logger.info("=" * 60)

    asos_daily_dir = _city_asos_daily_dir(cfg)
    ghcn_raw_dir = os.path.join(cfg.data_dir, "raw")
    report_dir = _city_migration_report_dir(cfg)
    os.makedirs(report_dir, exist_ok=True)

    report = generate_asos_ghcn_report(
        mapping_csv=mapping_csv,
        asos_daily_dir=asos_daily_dir,
        ghcn_raw_dir=ghcn_raw_dir,
        output_dir=report_dir,
    )

    if not report.empty:
        md_path = os.path.join(report_dir, "asos_ghcn_tmax_comparison.md")
        write_asos_ghcn_markdown(report, md_path)
        logger.info("Cross-validation report saved to %s", report_dir)

        # Log summary statistics
        logger.info("Cross-validation summary:")
        logger.info("  Stations compared: %d", len(report))
        logger.info(
            "  Mean bias (F): %.2f (std=%.2f)",
            report["mean_bias_f"].mean(),
            report["mean_bias_f"].std(),
        )
        logger.info("  Mean MAE (F): %.2f", report["mae_f"].mean())
        logger.info("  Mean correlation: %.3f", report["corr"].mean())
    else:
        logger.warning("No ASOS/GHCN overlap found for cross-validation.")


def step5_build_features(cfg, city_code: str) -> dict:
    """Step 5: Build ASOS-based training features."""
    logger.info("=" * 60)
    logger.info("Step 5: Building ASOS-based features for %s", city_code.upper())
    logger.info("=" * 60)

    asos_daily_dir = _city_asos_daily_dir(cfg)
    output_dir = _city_asos_processed_dir(cfg)

    result = build_asos_features(
        city_code=city_code,
        asos_daily_dir=asos_daily_dir,
        output_dir=output_dir,
    )
    return result


def step6_verify_parity(cfg, city_code: str) -> None:
    """Step 6: Verify feature distribution parity (ASOS vs GHCN)."""
    logger.info("=" * 60)
    logger.info("Step 6: Feature parity verification for %s", city_code.upper())
    logger.info("=" * 60)

    processed_dir = _city_asos_processed_dir(cfg)
    report_dir = _city_migration_report_dir(cfg)
    os.makedirs(report_dir, exist_ok=True)

    asos_features_path = os.path.join(processed_dir, "features_train.csv")
    # Check for a backup of original GHCN features
    ghcn_features_path = os.path.join(processed_dir, "features_train_ghcn_backup.csv")

    if not os.path.exists(ghcn_features_path):
        logger.info(
            "No GHCN backup features found at %s. "
            "Skipping parity comparison (no baseline to compare against).",
            ghcn_features_path,
        )
        # Still generate a stub report indicating parity check was skipped
        stub_report = {
            "city_code": city_code,
            "status": "skipped",
            "reason": "No GHCN baseline features backup available for comparison.",
        }
        json_path = os.path.join(report_dir, f"{city_code}_feature_parity.json")
        with open(json_path, "w") as fh:
            json.dump(stub_report, fh, indent=2)
        logger.info("Stub parity report written to %s", json_path)
        return

    results = compare_asos_vs_ghcn_features(
        asos_features_path=asos_features_path,
        ghcn_features_path=ghcn_features_path,
    )

    if results:
        md_path = generate_parity_report(results, report_dir, city_code)
        n_pass = sum(r.parity_pass for r in results)
        logger.info(
            "Parity check complete: %d/%d features passed. Report: %s",
            n_pass, len(results), md_path,
        )
    else:
        logger.warning("No common features found for parity comparison.")


def step7_summary(cfg, city_code: str) -> None:
    """Step 7: Print migration summary."""
    logger.info("=" * 60)
    logger.info("ASOS Migration Summary for %s", city_code.upper())
    logger.info("=" * 60)

    processed_dir = _city_asos_processed_dir(cfg)
    report_dir = _city_migration_report_dir(cfg)

    # Check what artifacts were produced
    artifacts = {
        "features_train": os.path.exists(os.path.join(processed_dir, "features_train.csv")),
        "features_val": os.path.exists(os.path.join(processed_dir, "features_val.csv")),
        "features_test": os.path.exists(os.path.join(processed_dir, "features_test.csv")),
        "target_train": os.path.exists(os.path.join(processed_dir, "target_train.csv")),
        "target_val": os.path.exists(os.path.join(processed_dir, "target_val.csv")),
        "target_test": os.path.exists(os.path.join(processed_dir, "target_test.csv")),
        "scaler": os.path.exists(os.path.join(processed_dir, "scaler.pkl")),
        "col_means": os.path.exists(os.path.join(processed_dir, "col_means.pkl")),
        "ghcn_crossval": os.path.exists(
            os.path.join(report_dir, "asos_ghcn_tmax_comparison.csv")
        ),
        "parity_report": os.path.exists(
            os.path.join(report_dir, f"{city_code}_feature_parity.json")
        ),
    }

    for name, exists in artifacts.items():
        status = "OK" if exists else "MISSING"
        logger.info("  [%s] %s", status, name)

    all_ok = all(artifacts.values())
    if all_ok:
        logger.info("All migration artifacts present.")
    else:
        missing = [k for k, v in artifacts.items() if not v]
        logger.warning("Missing artifacts: %s", missing)

    # Save summary JSON
    os.makedirs(report_dir, exist_ok=True)
    summary = {
        "city_code": city_code,
        "artifacts": artifacts,
        "all_present": all_ok,
    }
    summary_path = os.path.join(report_dir, f"{city_code}_migration_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Migration summary saved to %s", summary_path)

    # Next steps
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Retrain models: python scripts/run_benchmark.py --city %s", city_code)
    logger.info(
        "  2. Re-calibrate: python scripts/run_synthesis_calibration.py --city %s",
        city_code,
    )
    logger.info("  3. Backtest: python scripts/run_backtest.py --city %s", city_code)
    logger.info(
        "  4. Promote: python scripts/run_promotion_evaluation.py --city %s",
        city_code,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase E: ASOS Training Data Migration."
    )
    parser.add_argument(
        "--city",
        required=True,
        choices=["chi", "phl", "atl", "aus"],
        help="City code to migrate.",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip ASOS data download (use cached data).",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step numbers to run (e.g., '4,5,6'). "
             "Default runs all steps.",
    )
    parser.add_argument(
        "--backup-ghcn",
        action="store_true",
        help="Backup existing GHCN features before overwriting with ASOS features.",
    )
    args = parser.parse_args()

    city_code = args.city
    cfg = get_city_config(city_code)
    ensure_city_dirs(cfg)

    # Parse step list
    if args.steps:
        steps_to_run = {int(s.strip()) for s in args.steps.split(",")}
    else:
        steps_to_run = {1, 2, 3, 4, 5, 6, 7}

    if args.skip_collection:
        steps_to_run.discard(2)

    logger.info("=" * 60)
    logger.info("Phase E: ASOS Migration for %s", cfg.city_name)
    logger.info("Steps to run: %s", sorted(steps_to_run))
    logger.info("=" * 60)

    # Backup existing GHCN features if requested
    if args.backup_ghcn and 5 in steps_to_run:
        processed_dir = _city_asos_processed_dir(cfg)
        for split in ["train", "val", "test"]:
            src_path = os.path.join(processed_dir, f"features_{split}.csv")
            dst_path = os.path.join(processed_dir, f"features_{split}_ghcn_backup.csv")
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                import shutil
                shutil.copy2(src_path, dst_path)
                logger.info("Backed up %s -> %s", src_path, dst_path)

    # Run steps
    mapping_csv = None
    if 1 in steps_to_run:
        mapping_csv = step1_write_mapping(cfg, city_code)

    if mapping_csv is None:
        mapping_csv = _city_asos_mapping_csv(cfg)

    if 2 in steps_to_run:
        step2_collect_asos(cfg, city_code, mapping_csv)

    if 3 in steps_to_run:
        step3_aggregate_daily(cfg, city_code, mapping_csv)

    if 4 in steps_to_run:
        step4_ghcn_crossval(cfg, city_code, mapping_csv)

    if 5 in steps_to_run:
        step5_build_features(cfg, city_code)

    if 6 in steps_to_run:
        step6_verify_parity(cfg, city_code)

    if 7 in steps_to_run:
        step7_summary(cfg, city_code)

    logger.info("Phase E migration pipeline complete for %s.", city_code.upper())


if __name__ == "__main__":
    main()
