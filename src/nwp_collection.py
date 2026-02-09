"""
NWP Data Collection (GFS/GEFS).

Uses the Herbie library to download and subset GFS/GEFS forecasts for the
NYC grid point. This module provides the scaffolding for Phase 0 data
scale-up; it does not run by default without Herbie installed.
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Iterable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _require_herbie():
    try:
        from herbie import Herbie  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "The 'herbie-data' package is required for NWP downloads. "
            "Install it via pip install herbie-data."
        ) from exc


def download_gfs_point(
    date: datetime,
    fxx: int = 24,
    variables: Optional[Iterable[str]] = None,
    output_dir: str = config.NWP_RAW_DIR,
    model: str = "gfs",
) -> str:
    """Download a single GFS forecast file for the NYC grid point."""
    _require_herbie()
    from herbie import Herbie

    os.makedirs(output_dir, exist_ok=True)
    variables = list(variables or ["tmax_2m"])
    variable_level = ",".join(variables)

    herbie = Herbie(date, model=model, fxx=fxx, product="pgrb2.0p25")
    logger.info("Downloading %s %s f%03d", model, date.strftime("%Y-%m-%d %H"),
                fxx)
    path = herbie.download(variable=variable_level, save_dir=output_dir)
    return path


def download_gfs_range(
    start_date: str,
    end_date: str,
    cycle_hour: int = 0,
    fxx: int = 24,
    variables: Optional[Iterable[str]] = None,
    output_dir: str = config.NWP_RAW_DIR,
    model: str = "gfs",
) -> list[str]:
    """Download GFS forecasts for a date range (daily cadence)."""
    paths: list[str] = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    while current <= end:
        run_date = current.replace(hour=cycle_hour)
        try:
            path = download_gfs_point(
                date=run_date,
                fxx=fxx,
                variables=variables,
                output_dir=output_dir,
                model=model,
            )
            paths.append(path)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to download %s %s: %s",
                           model, run_date.strftime("%Y-%m-%d %H"), exc)
        current += timedelta(days=1)
    return paths


def download_gefs_reforecast_point(
    date: datetime,
    fxx: int = 24,
    member: int = 0,
    variables: Optional[Iterable[str]] = None,
    output_dir: str = config.NWP_RAW_DIR,
    model: str = "gefs_reforecast",
) -> str:
    """Download a GEFSv12 reforecast file for the NYC grid point."""
    _require_herbie()
    from herbie import Herbie

    os.makedirs(output_dir, exist_ok=True)
    variables = list(variables or ["tmax_2m"])
    variable_level = ",".join(variables)

    herbie = Herbie(date, model=model, fxx=fxx, member=member)
    logger.info("Downloading %s %s f%03d member %02d",
                model, date.strftime("%Y-%m-%d %H"), fxx, member)
    path = herbie.download(variable=variable_level, save_dir=output_dir)
    return path


def download_gefs_reforecast_range(
    start_date: str,
    end_date: str,
    cycle_hour: int = 0,
    fxx: int = 24,
    member: int = 0,
    variables: Optional[Iterable[str]] = None,
    output_dir: str = config.NWP_RAW_DIR,
    model: str = "gefs_reforecast",
) -> list[str]:
    """Download GEFS reforecast files for a date range (daily cadence)."""
    paths: list[str] = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    while current <= end:
        run_date = current.replace(hour=cycle_hour)
        try:
            path = download_gefs_reforecast_point(
                date=run_date,
                fxx=fxx,
                member=member,
                variables=variables,
                output_dir=output_dir,
                model=model,
            )
            paths.append(path)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to download %s %s member %02d: %s",
                           model, run_date.strftime("%Y-%m-%d %H"), member, exc)
        current += timedelta(days=1)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download NWP data via Herbie.")
    parser.add_argument("--date", required=True, help="Run date YYYY-MM-DD.")
    parser.add_argument("--hour", type=int, default=0, help="Cycle hour (UTC).")
    parser.add_argument("--fxx", type=int, default=24, help="Forecast hour.")
    parser.add_argument(
        "--model",
        choices=["gfs", "gefs_reforecast"],
        default="gfs",
        help="NWP model to download.",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=0,
        help="GEFS member index (only for gefs_reforecast).",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=["tmax_2m"],
        help="Variable list to download (Herbie syntax).",
    )
    parser.add_argument(
        "--output-dir",
        default=config.NWP_RAW_DIR,
        help="Directory to store downloaded GRIB files.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date for range downloads (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date for range downloads (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    if args.start_date and args.end_date:
        if args.model == "gfs":
            download_gfs_range(
                start_date=args.start_date,
                end_date=args.end_date,
                cycle_hour=args.hour,
                fxx=args.fxx,
                variables=args.variables,
                output_dir=args.output_dir,
                model=args.model,
            )
        else:
            download_gefs_reforecast_range(
                start_date=args.start_date,
                end_date=args.end_date,
                cycle_hour=args.hour,
                fxx=args.fxx,
                member=args.member,
                variables=args.variables,
                output_dir=args.output_dir,
                model=args.model,
            )
    else:
        date = datetime.strptime(args.date, "%Y-%m-%d").replace(hour=args.hour)
        if args.model == "gfs":
            download_gfs_point(
                date=date,
                fxx=args.fxx,
                variables=args.variables,
                output_dir=args.output_dir,
                model=args.model,
            )
        else:
            download_gefs_reforecast_point(
                date=date,
                fxx=args.fxx,
                member=args.member,
                variables=args.variables,
                output_dir=args.output_dir,
                model=args.model,
            )


if __name__ == "__main__":
    main()
