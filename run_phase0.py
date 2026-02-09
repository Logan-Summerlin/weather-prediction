#!/usr/bin/env python3
"""
Phase 0 Runner: Multi-source data scale-up.

Runs the operational data downloads for ASOS, IGRA soundings, and NWP
forecasts, plus optional GHCN collection. Includes preprocessing steps
for ASOS daily aggregation, IGRA soundings, and NWP GRIB extraction.
Designed to be resumable and time-safe (no training-only data in the
operational paths).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.asos_collection import collect_asos_data
from src.soundings_collection import download_soundings
from src.asos_preprocessing import run_asos_daily_pipeline
from src.nwp_collection import download_gfs_range, download_gefs_reforecast_range

logger = logging.getLogger(__name__)


def _import_soundings_preprocessing():
    """Conditionally import soundings preprocessing (may not exist yet)."""
    try:
        from src.soundings_preprocessing import run_soundings_preprocessing
        return run_soundings_preprocessing
    except ImportError:
        logger.warning(
            "src.soundings_preprocessing not available — skipping IGRA preprocessing."
        )
        return None


def _import_nwp_preprocessing():
    """Conditionally import NWP preprocessing (may not exist yet)."""
    try:
        from src.nwp_preprocessing import run_nwp_preprocessing
        return run_nwp_preprocessing
    except ImportError:
        logger.warning(
            "src.nwp_preprocessing not available — skipping NWP preprocessing."
        )
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 data downloads.")
    parser.add_argument(
        "--skip-asos",
        action="store_true",
        help="Skip ASOS hourly downloads.",
    )
    parser.add_argument(
        "--skip-igra",
        action="store_true",
        help="Skip IGRA soundings downloads.",
    )
    parser.add_argument(
        "--skip-nwp",
        action="store_true",
        help="Skip NWP downloads (GFS/GEFS).",
    )
    parser.add_argument(
        "--skip-asos-aggregate",
        action="store_true",
        help="Skip ASOS daily aggregation and GHCN comparison.",
    )
    parser.add_argument(
        "--skip-asos-ghcn-report",
        action="store_true",
        help="Skip the ASOS vs GHCN comparison report.",
    )
    parser.add_argument(
        "--skip-igra-preprocess",
        action="store_true",
        help="Skip IGRA soundings preprocessing.",
    )
    parser.add_argument(
        "--skip-nwp-preprocess",
        action="store_true",
        help="Skip NWP GRIB preprocessing.",
    )
    parser.add_argument(
        "--asos-mapping-csv",
        default=os.path.join(config.DATA_DIR, "asos_station_mapping.csv"),
        help="CSV with station_id -> ICAO mapping for ASOS downloads.",
    )
    parser.add_argument(
        "--asos-chunk-years",
        type=int,
        default=1,
        help="Year-size chunks for ASOS downloads.",
    )
    parser.add_argument(
        "--nwp-start-date",
        default=config.NWP_START_DATE,
        help="Start date for NWP downloads (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--nwp-end-date",
        default=config.NWP_END_DATE,
        help="End date for NWP downloads (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--nwp-model",
        choices=["gfs", "gefs_reforecast"],
        default="gfs",
        help="NWP model to download.",
    )
    parser.add_argument(
        "--nwp-member",
        type=int,
        default=0,
        help="GEFS member index (if gefs_reforecast).",
    )
    parser.add_argument(
        "--nwp-variables",
        nargs="*",
        default=config.NWP_VARIABLES,
        help="Herbie variable list.",
    )
    parser.add_argument(
        "--nwp-fxx",
        type=int,
        default=24,
        help="Forecast hour.",
    )
    args = parser.parse_args()

    # ---- ASOS Collection ----
    if not args.skip_asos:
        collect_asos_data(
            mapping_csv=args.asos_mapping_csv,
            output_dir=config.ASOS_RAW_DIR,
            start_date=config.ASOS_START_DATE,
            end_date=config.ASOS_END_DATE,
            chunk_years=args.asos_chunk_years,
        )

    # ---- ASOS Aggregation ----
    if not args.skip_asos_aggregate:
        run_asos_daily_pipeline(
            mapping_csv=args.asos_mapping_csv,
            asos_raw_dir=config.ASOS_RAW_DIR,
            asos_daily_dir=config.ASOS_DAILY_DIR,
            ghcn_raw_dir=config.RAW_DATA_DIR,
            report_dir=config.REPORTS_DIR,
            write_report=not args.skip_asos_ghcn_report,
        )

    # ---- IGRA Soundings Download ----
    if not args.skip_igra:
        download_soundings(
            station_id=config.IGRA_STATION_ID,
            start_date=config.IGRA_START_DATE,
            end_date=config.IGRA_END_DATE,
            hours=(0, 12),
            output_dir=config.IGRA_RAW_DIR,
        )

    # ---- IGRA Soundings Preprocessing ----
    if not args.skip_igra_preprocess:
        run_soundings = _import_soundings_preprocessing()
        if run_soundings is not None:
            run_soundings(
                sounding_dir=config.IGRA_RAW_DIR,
                output_dir=config.IGRA_DAILY_DIR,
            )

    # ---- NWP Download ----
    if not args.skip_nwp:
        if args.nwp_model == "gfs":
            download_gfs_range(
                start_date=args.nwp_start_date,
                end_date=args.nwp_end_date,
                cycle_hour=0,
                fxx=args.nwp_fxx,
                variables=args.nwp_variables,
                output_dir=config.NWP_RAW_DIR,
                model=args.nwp_model,
            )
        else:
            download_gefs_reforecast_range(
                start_date=args.nwp_start_date,
                end_date=args.nwp_end_date,
                cycle_hour=0,
                fxx=args.nwp_fxx,
                member=args.nwp_member,
                variables=args.nwp_variables,
                output_dir=config.NWP_RAW_DIR,
                model=args.nwp_model,
            )

    # ---- NWP Preprocessing ----
    if not args.skip_nwp_preprocess:
        run_nwp = _import_nwp_preprocessing()
        if run_nwp is not None:
            run_nwp(
                nwp_dir=config.NWP_RAW_DIR,
                output_dir=config.NWP_DAILY_DIR,
            )


if __name__ == "__main__":
    main()
