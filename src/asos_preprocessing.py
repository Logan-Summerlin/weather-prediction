"""
ASOS Daily Preprocessing and GHCN Cross-Validation.

Aggregates IEM ASOS hourly data into daily features used for operational
training/inference and validates ASOS-derived TMAX against GHCN-Daily.

Key outputs:
  - Daily aggregates: TMAX/TMIN/TMEAN, dewpoint stats, wind, SLP, cloud fraction
  - Optional ASOS vs GHCN bias report for overlap periods
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import config

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_AFTERNOON_HOURS_UTC = (18, 19, 20, 21, 22, 23)
DEFAULT_EVENING_HOURS_UTC = (18, 19, 20, 21, 22, 23)
WIND_KT_TO_MPH = 1.15078


@dataclass(frozen=True)
class AsosDailyConfig:
    afternoon_hours_utc: Iterable[int] = DEFAULT_AFTERNOON_HOURS_UTC
    evening_hours_utc: Iterable[int] = DEFAULT_EVENING_HOURS_UTC
    low_cloud_ceiling_ft: float = 5000.0


def _parse_asos_csv(path: str) -> pd.DataFrame:
    """Load an IEM ASOS CSV and normalize columns."""
    df = pd.read_csv(
        path,
        comment="#",
        na_values=["M", "NA", ""],
        keep_default_na=True,
    )
    if "valid" not in df.columns:
        raise ValueError(f"ASOS CSV missing 'valid' column: {path}")

    df["valid_utc"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df = df.dropna(subset=["valid_utc"]).copy()
    df["date_utc"] = df["valid_utc"].dt.date
    df["hour_utc"] = df["valid_utc"].dt.hour

    numeric_cols = ["tmpf", "dwpf", "drct", "sknt", "mslp", "ceil"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df


def _vector_mean_direction(degrees: pd.Series, speeds: Optional[pd.Series] = None) -> float:
    """Compute speed-weighted mean wind direction (degrees)."""
    if degrees.empty:
        return np.nan
    mask = degrees.notna()
    if speeds is not None:
        mask = mask & speeds.notna()
    if not mask.any():
        return np.nan

    deg = degrees[mask].astype(float)
    spd = speeds[mask].astype(float) if speeds is not None else pd.Series(
        1.0, index=deg.index
    )
    radians = np.deg2rad(deg)
    u = np.sin(radians) * spd
    v = np.cos(radians) * spd
    mean_u = u.mean()
    mean_v = v.mean()
    if np.isclose(mean_u, 0.0) and np.isclose(mean_v, 0.0):
        return np.nan
    angle = (np.degrees(np.arctan2(mean_u, mean_v)) + 360) % 360
    return float(angle)


def aggregate_asos_daily(
    df: pd.DataFrame,
    config_daily: Optional[AsosDailyConfig] = None,
) -> pd.DataFrame:
    """Aggregate an ASOS hourly DataFrame to daily features (UTC dates)."""
    config_daily = config_daily or AsosDailyConfig()
    grouped = df.groupby("date_utc")
    daily = grouped["tmpf"].agg(tmax_f="max", tmin_f="min", tmean_f="mean")

    daily["dewpoint_mean_f"] = grouped["dwpf"].mean()
    afternoon_mask = df["hour_utc"].isin(config_daily.afternoon_hours_utc)
    daily["dewpoint_afternoon_f"] = df[afternoon_mask].groupby("date_utc")[
        "dwpf"
    ].mean()

    wind_speed_mph = df["sknt"] * WIND_KT_TO_MPH
    daily["wind_speed_mean_mph"] = wind_speed_mph.groupby(df["date_utc"]).mean()
    daily["wind_speed_max_mph"] = wind_speed_mph.groupby(df["date_utc"]).max()
    daily["wind_dir_mean_deg"] = grouped.apply(
        lambda g: _vector_mean_direction(g["drct"], g["sknt"])
    )

    evening_mask = df["hour_utc"].isin(config_daily.evening_hours_utc)
    daily["wind_dir_evening_deg"] = df[evening_mask].groupby("date_utc").apply(
        lambda g: _vector_mean_direction(g["drct"], g["sknt"])
    )

    daily["slp_00z_mb"] = df[df["hour_utc"] == 0].groupby("date_utc")["mslp"].mean()
    daily["slp_12z_mb"] = df[df["hour_utc"] == 12].groupby("date_utc")["mslp"].mean()

    daily = daily.sort_index()
    daily["slp_tendency_24h_mb"] = daily["slp_00z_mb"] - daily["slp_00z_mb"].shift(1)

    ceil_valid = df["ceil"].notna()
    ceil_low = df["ceil"] < config_daily.low_cloud_ceiling_ft
    daily["cloud_fraction_low"] = (
        ceil_low.groupby(df["date_utc"]).sum()
        / ceil_valid.groupby(df["date_utc"]).sum()
    )

    daily["obs_count"] = grouped.size()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily.reset_index()


def _collect_asos_files(icao: str, input_dir: str) -> list[str]:
    direct_path = os.path.join(input_dir, f"{icao}.csv")
    if os.path.exists(direct_path):
        return [direct_path]

    station_dir = os.path.join(input_dir, icao)
    if not os.path.isdir(station_dir):
        return []
    return sorted(glob.glob(os.path.join(station_dir, f"{icao}_*.csv")))


def aggregate_asos_directory(
    mapping_csv: str,
    input_dir: str,
    output_dir: str,
    config_daily: Optional[AsosDailyConfig] = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate ASOS hourly data for all mapped stations."""
    from src.asos_collection import load_asos_station_map

    config_daily = config_daily or AsosDailyConfig()
    station_map = load_asos_station_map(mapping_csv)
    os.makedirs(output_dir, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    for station_id, icao in station_map.items():
        files = _collect_asos_files(icao, input_dir)
        if not files:
            LOGGER.warning("No ASOS files found for %s (%s)", station_id, icao)
            continue

        frames = []
        for path in files:
            try:
                frames.append(_parse_asos_csv(path))
            except Exception as exc:  # pragma: no cover - I/O dependent
                LOGGER.warning("Failed to parse %s: %s", path, exc)

        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["valid_utc"]
        )
        daily = aggregate_asos_daily(combined, config_daily=config_daily)
        daily.insert(1, "station_id", station_id)
        daily.insert(2, "icao", icao)

        output_path = os.path.join(output_dir, f"{station_id}_asos_daily.csv")
        daily.to_csv(output_path, index=False)
        LOGGER.info("Saved ASOS daily aggregates to %s", output_path)
        outputs[station_id] = daily

    return outputs


def _load_ghcn_daily(station_id: str, raw_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(raw_dir, f"{station_id}.csv")
    if not os.path.exists(path):
        LOGGER.warning("Missing GHCN CSV for %s (%s)", station_id, path)
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def compare_asos_ghcn_tmax(
    asos_daily: pd.DataFrame,
    ghcn_daily: pd.DataFrame,
) -> pd.Series:
    """Compute ASOS vs GHCN TMAX comparison metrics."""
    merged = asos_daily.merge(
        ghcn_daily[["date", "TMAX"]],
        on="date",
        how="inner",
        suffixes=("_asos", "_ghcn"),
    )
    merged = merged.dropna(subset=["tmax_f", "TMAX"])
    if merged.empty:
        return pd.Series(
            {
                "overlap_days": 0,
                "mean_bias_f": np.nan,
                "mae_f": np.nan,
                "rmse_f": np.nan,
                "corr": np.nan,
            }
        )

    diff = merged["tmax_f"] - merged["TMAX"]
    return pd.Series(
        {
            "overlap_days": len(merged),
            "mean_bias_f": diff.mean(),
            "mae_f": diff.abs().mean(),
            "rmse_f": np.sqrt((diff ** 2).mean()),
            "corr": merged["tmax_f"].corr(merged["TMAX"]),
        }
    )


def generate_asos_ghcn_report(
    mapping_csv: str,
    asos_daily_dir: str,
    ghcn_raw_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Generate a station-level ASOS vs GHCN TMAX comparison report."""
    from src.asos_collection import load_asos_station_map

    station_map = load_asos_station_map(mapping_csv)
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for station_id, icao in station_map.items():
        asos_path = os.path.join(asos_daily_dir, f"{station_id}_asos_daily.csv")
        if not os.path.exists(asos_path):
            LOGGER.warning("Missing ASOS daily file for %s", station_id)
            continue

        asos_daily = pd.read_csv(asos_path, parse_dates=["date"])
        ghcn_daily = _load_ghcn_daily(station_id, ghcn_raw_dir)
        if ghcn_daily is None:
            continue

        metrics = compare_asos_ghcn_tmax(asos_daily, ghcn_daily)
        metrics["station_id"] = station_id
        metrics["icao"] = icao
        rows.append(metrics)

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    report = report[
        [
            "station_id",
            "icao",
            "overlap_days",
            "mean_bias_f",
            "mae_f",
            "rmse_f",
            "corr",
        ]
    ].sort_values("station_id")

    csv_path = os.path.join(output_dir, "asos_ghcn_tmax_comparison.csv")
    report.to_csv(csv_path, index=False)
    LOGGER.info("Saved ASOS/GHCN comparison CSV to %s", csv_path)
    return report


def write_asos_ghcn_markdown(report: pd.DataFrame, output_path: str) -> None:
    """Write a Markdown summary for ASOS/GHCN comparisons."""
    if report.empty:
        content = "# ASOS vs GHCN TMAX Comparison\n\nNo overlapping data found."
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return

    lines = [
        "# ASOS vs GHCN TMAX Comparison",
        "",
        "Summary of daily TMAX differences between ASOS-derived values and "
        "GHCN-Daily for overlapping dates.",
        "",
        "| Station | ICAO | Overlap Days | Mean Bias (F) | MAE (F) | RMSE (F) | Corr |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in report.iterrows():
        lines.append(
            f"| {row['station_id']} | {row['icao']} | {int(row['overlap_days'])} "
            f"| {row['mean_bias_f']:.2f} | {row['mae_f']:.2f} | "
            f"{row['rmse_f']:.2f} | {row['corr']:.3f} |"
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_asos_daily_pipeline(
    mapping_csv: str = os.path.join(config.DATA_DIR, "asos_station_mapping.csv"),
    asos_raw_dir: str = config.ASOS_RAW_DIR,
    asos_daily_dir: str = config.ASOS_DAILY_DIR,
    ghcn_raw_dir: str = config.RAW_DATA_DIR,
    report_dir: str = config.REPORTS_DIR,
    write_report: bool = True,
) -> None:
    """Run ASOS daily aggregation and GHCN comparison."""
    aggregate_asos_directory(
        mapping_csv=mapping_csv,
        input_dir=asos_raw_dir,
        output_dir=asos_daily_dir,
    )

    if write_report:
        report = generate_asos_ghcn_report(
            mapping_csv=mapping_csv,
            asos_daily_dir=asos_daily_dir,
            ghcn_raw_dir=ghcn_raw_dir,
            output_dir=report_dir,
        )
        markdown_path = os.path.join(report_dir, "asos_ghcn_tmax_comparison.md")
        write_asos_ghcn_markdown(report, markdown_path)
        LOGGER.info("Saved ASOS/GHCN Markdown report to %s", markdown_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate ASOS hourly data.")
    parser.add_argument(
        "--mapping-csv",
        default=os.path.join(config.DATA_DIR, "asos_station_mapping.csv"),
    )
    parser.add_argument("--asos-raw-dir", default=config.ASOS_RAW_DIR)
    parser.add_argument("--asos-daily-dir", default=config.ASOS_DAILY_DIR)
    parser.add_argument("--ghcn-raw-dir", default=config.RAW_DATA_DIR)
    parser.add_argument("--report-dir", default=config.REPORTS_DIR)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    aggregate_asos_directory(
        mapping_csv=args.mapping_csv,
        input_dir=args.asos_raw_dir,
        output_dir=args.asos_daily_dir,
    )
    if not args.skip_report:
        report = generate_asos_ghcn_report(
            mapping_csv=args.mapping_csv,
            asos_daily_dir=args.asos_daily_dir,
            ghcn_raw_dir=args.ghcn_raw_dir,
            output_dir=args.report_dir,
        )
        markdown_path = os.path.join(args.report_dir, "asos_ghcn_tmax_comparison.md")
        write_asos_ghcn_markdown(report, markdown_path)


if __name__ == "__main__":
    main()
