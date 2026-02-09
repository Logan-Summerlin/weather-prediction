#!/usr/bin/env python3
"""
Phase 0 Runner: Multi-source data scale-up.

Runs the operational data downloads for ASOS, IGRA soundings, and NWP
forecasts, plus optional GHCN collection. Designed to be resumable and
time-safe (no training-only data in the operational paths).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.asos_collection import collect_asos_data
from src.soundings_collection import download_soundings
from src.nwp_collection import download_gfs_range, download_gefs_reforecast_range


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
        default=["tmax_2m"],
        help="Herbie variable list.",
    )
    parser.add_argument(
        "--nwp-fxx",
        type=int,
        default=24,
        help="Forecast hour.",
    )
    args = parser.parse_args()

    if not args.skip_asos:
        collect_asos_data(
            mapping_csv=os.path.join(config.DATA_DIR, "asos_station_mapping.csv"),
            output_dir=config.ASOS_RAW_DIR,
            start_date=config.ASOS_START_DATE,
            end_date=config.ASOS_END_DATE,
            chunk_years=args.asos_chunk_years,
        )

    if not args.skip_igra:
        download_soundings(
            station_id=config.IGRA_STATION_ID,
            start_date=config.IGRA_START_DATE,
            end_date=config.IGRA_END_DATE,
            hours=(0, 12),
            output_dir=config.IGRA_RAW_DIR,
        )

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


if __name__ == "__main__":
    main()
