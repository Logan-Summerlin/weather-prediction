"""
Tests for ASOS daily preprocessing and ASOS vs GHCN comparison.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.asos_preprocessing import (
    _parse_asos_csv,
    aggregate_asos_daily,
    compare_asos_ghcn_tmax,
)


def _write_csv(tmp_path: Path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "asos.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_aggregate_asos_daily_basic(tmp_path: Path) -> None:
    rows = [
        {
            "station": "KNYC",
            "valid": "2022-01-01 00:00",
            "tmpf": 50,
            "dwpf": 40,
            "drct": 0,
            "sknt": 10,
            "mslp": 1010,
            "ceil": 4000,
        },
        {
            "station": "KNYC",
            "valid": "2022-01-01 12:00",
            "tmpf": 60,
            "dwpf": 45,
            "drct": 90,
            "sknt": 20,
            "mslp": 1015,
            "ceil": 6000,
        },
        {
            "station": "KNYC",
            "valid": "2022-01-01 18:00",
            "tmpf": 70,
            "dwpf": 50,
            "drct": 180,
            "sknt": 30,
            "mslp": 1012,
            "ceil": 3000,
        },
        {
            "station": "KNYC",
            "valid": "2022-01-02 00:00",
            "tmpf": 55,
            "dwpf": 42,
            "drct": 270,
            "sknt": 10,
            "mslp": 1005,
            "ceil": 7000,
        },
    ]
    path = _write_csv(tmp_path, rows)
    df = _parse_asos_csv(path)
    daily = aggregate_asos_daily(df)

    day1 = daily.loc[daily["date"] == pd.Timestamp("2022-01-01")]
    assert day1["tmax_f"].iloc[0] == 70
    assert day1["tmin_f"].iloc[0] == 50
    assert day1["dewpoint_afternoon_f"].iloc[0] == 50
    assert abs(day1["wind_dir_mean_deg"].iloc[0] - 135.0) < 0.1
    assert day1["wind_dir_evening_deg"].iloc[0] == 180
    assert abs(day1["cloud_fraction_low"].iloc[0] - (2 / 3)) < 0.01
    assert day1["slp_00z_mb"].iloc[0] == 1010
    assert day1["slp_12z_mb"].iloc[0] == 1015

    day2 = daily.loc[daily["date"] == pd.Timestamp("2022-01-02")]
    assert day2["slp_tendency_24h_mb"].iloc[0] == -5


def test_compare_asos_ghcn_tmax() -> None:
    asos_daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")],
            "tmax_f": [50.0, 60.0],
        }
    )
    ghcn_daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")],
            "TMAX": [49.0, 62.0],
        }
    )
    metrics = compare_asos_ghcn_tmax(asos_daily, ghcn_daily)
    assert metrics["overlap_days"] == 2
    assert metrics["mean_bias_f"] == -0.5
