"""
Tests for ASOS daily preprocessing and ASOS vs GHCN comparison.

Covers:
  - _parse_asos_csv() — missing valid column, all-NaN numerics, empty DF
  - _vector_mean_direction() — north, calm, mixed, empty
  - aggregate_asos_daily() — basic, single obs, missing columns, multi-day
  - compare_asos_ghcn_tmax() — no overlap, perfect match, large bias, empty
  - write_asos_ghcn_markdown() — empty report, normal report
  - AsosDailyConfig — custom parameters
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.asos_preprocessing import (
    _parse_asos_csv,
    _vector_mean_direction,
    aggregate_asos_daily,
    compare_asos_ghcn_tmax,
    write_asos_ghcn_markdown,
    AsosDailyConfig,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _write_csv(tmp_path: Path, rows: list[dict], filename: str = "asos.csv") -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def _make_hourly_rows(
    station: str = "KNYC",
    date: str = "2022-01-01",
    hours: list[int] | None = None,
    tmpf: float = 50.0,
    dwpf: float = 40.0,
    drct: float = 180.0,
    sknt: float = 10.0,
    mslp: float = 1010.0,
    ceil: float = 5000.0,
) -> list[dict]:
    """Build hourly observation rows for a single day."""
    if hours is None:
        hours = list(range(0, 24))
    return [
        {
            "station": station,
            "valid": f"{date} {h:02d}:00",
            "tmpf": tmpf + h * 0.5,
            "dwpf": dwpf,
            "drct": drct,
            "sknt": sknt,
            "mslp": mslp,
            "ceil": ceil,
        }
        for h in hours
    ]


# ===========================================================================
# _parse_asos_csv tests
# ===========================================================================

class TestParseAsosCsv:

    def test_normal_parse(self, tmp_path):
        """Parses a well-formed ASOS CSV successfully."""
        rows = _make_hourly_rows(hours=[0, 6, 12, 18])
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        assert len(df) == 4
        assert "date_utc" in df.columns
        assert "hour_utc" in df.columns

    def test_missing_valid_column(self, tmp_path):
        """Raises ValueError when 'valid' column is missing."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"station": ["KNYC"], "tmpf": [50]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="valid"):
            _parse_asos_csv(str(path))

    def test_all_nan_numeric_columns(self, tmp_path):
        """All-NaN numeric columns don't crash parsing."""
        rows = [
            {"station": "KNYC", "valid": "2022-01-01 00:00",
             "tmpf": "M", "dwpf": "M", "drct": "M", "sknt": "M",
             "mslp": "M", "ceil": "M"},
        ]
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        assert len(df) == 1
        assert np.isnan(df["tmpf"].iloc[0])

    def test_empty_after_dropna(self, tmp_path):
        """DataFrame is empty when all timestamps are invalid."""
        rows = [
            {"station": "KNYC", "valid": "not-a-date", "tmpf": 50},
        ]
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        assert df.empty

    def test_comment_lines_skipped(self, tmp_path):
        """Lines starting with # are treated as comments."""
        content = (
            "# This is a comment\n"
            "station,valid,tmpf,dwpf,drct,sknt,mslp,ceil\n"
            "KNYC,2022-01-01 00:00,50,40,180,10,1010,5000\n"
        )
        path = tmp_path / "comment.csv"
        path.write_text(content)
        df = _parse_asos_csv(str(path))
        assert len(df) == 1


# ===========================================================================
# _vector_mean_direction tests
# ===========================================================================

class TestVectorMeanDirection:

    def test_north_wind(self):
        """All-north winds yield direction 0 (or 360)."""
        deg = pd.Series([0, 0, 0])
        spd = pd.Series([10, 10, 10])
        result = _vector_mean_direction(deg, spd)
        assert abs(result) < 1 or abs(result - 360) < 1

    def test_south_wind(self):
        """All-south winds yield direction ~180."""
        deg = pd.Series([180, 180, 180])
        spd = pd.Series([10, 10, 10])
        result = _vector_mean_direction(deg, spd)
        assert abs(result - 180) < 1

    def test_calm_winds(self):
        """All-zero speeds yield NaN (no meaningful direction)."""
        deg = pd.Series([0, 90, 180, 270])
        spd = pd.Series([0, 0, 0, 0])
        result = _vector_mean_direction(deg, spd)
        assert np.isnan(result)

    def test_mixed_directions_speed_weighted(self):
        """Speed-weighted mean of opposing winds cancels out."""
        deg = pd.Series([0, 180])
        spd = pd.Series([10, 10])
        result = _vector_mean_direction(deg, spd)
        # Equal north/south cancel -> NaN
        assert np.isnan(result)

    def test_empty_series(self):
        """Empty input yields NaN."""
        result = _vector_mean_direction(pd.Series([], dtype=float))
        assert np.isnan(result)

    def test_all_nan_directions(self):
        """All-NaN directions yield NaN."""
        deg = pd.Series([np.nan, np.nan])
        spd = pd.Series([10, 10])
        result = _vector_mean_direction(deg, spd)
        assert np.isnan(result)

    def test_no_speed_weights(self):
        """Without speed weights, all directions get equal weight."""
        deg = pd.Series([0, 90])
        result = _vector_mean_direction(deg)
        # Should be ~45 degrees
        assert abs(result - 45) < 1


# ===========================================================================
# aggregate_asos_daily tests (existing + new)
# ===========================================================================

class TestAggregateAsosDaily:

    def test_basic(self, tmp_path) -> None:
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

    def test_single_observation_day(self, tmp_path):
        """A day with a single observation still produces a valid row."""
        # Use hour 18 (within evening hours) to avoid pandas 3.0 empty-group issue
        rows = _make_hourly_rows(hours=[18])
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        daily = aggregate_asos_daily(df)
        assert len(daily) == 1
        assert daily["obs_count"].iloc[0] == 1
        # tmax == tmin for single observation
        assert daily["tmax_f"].iloc[0] == daily["tmin_f"].iloc[0]

    def test_missing_optional_columns(self, tmp_path):
        """Missing optional columns (ceil) become NaN, not crash."""
        rows = [
            {"station": "KNYC", "valid": "2022-01-01 18:00", "tmpf": 50, "dwpf": 40},
        ]
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        daily = aggregate_asos_daily(df)
        assert len(daily) == 1
        # Ceiling-related columns should be NaN
        assert np.isnan(daily["cloud_fraction_low"].iloc[0])

    def test_multi_day_with_gap(self, tmp_path):
        """Days with gaps between them don't fill the gap."""
        rows = _make_hourly_rows(date="2022-01-01", hours=[0, 18])
        rows += _make_hourly_rows(date="2022-01-03", hours=[0, 18])
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        daily = aggregate_asos_daily(df)
        # Should have exactly 2 days (not 3)
        assert len(daily) == 2

    def test_custom_config(self, tmp_path):
        """AsosDailyConfig with custom parameters works."""
        custom_cfg = AsosDailyConfig(
            afternoon_hours_utc=(14, 15, 16),
            evening_hours_utc=(20, 21, 22),
            low_cloud_ceiling_ft=3000.0,
        )
        rows = _make_hourly_rows(hours=[0, 14, 20])
        path = _write_csv(tmp_path, rows)
        df = _parse_asos_csv(path)
        daily = aggregate_asos_daily(df, config_daily=custom_cfg)
        assert len(daily) == 1


# ===========================================================================
# compare_asos_ghcn_tmax tests
# ===========================================================================

class TestCompareAsosGhcnTmax:

    def test_basic_comparison(self) -> None:
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

    def test_no_overlap(self):
        """No overlapping dates returns NaN metrics."""
        asos = pd.DataFrame({
            "date": [pd.Timestamp("2022-01-01")],
            "tmax_f": [50.0],
        })
        ghcn = pd.DataFrame({
            "date": [pd.Timestamp("2022-06-01")],
            "TMAX": [80.0],
        })
        metrics = compare_asos_ghcn_tmax(asos, ghcn)
        assert metrics["overlap_days"] == 0
        assert np.isnan(metrics["mae_f"])

    def test_perfect_match(self):
        """Identical values yield zero bias and MAE."""
        dates = [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")]
        asos = pd.DataFrame({"date": dates, "tmax_f": [50.0, 60.0]})
        ghcn = pd.DataFrame({"date": dates, "TMAX": [50.0, 60.0]})
        metrics = compare_asos_ghcn_tmax(asos, ghcn)
        assert metrics["mean_bias_f"] == 0.0
        assert metrics["mae_f"] == 0.0
        assert metrics["rmse_f"] == 0.0
        assert metrics["corr"] == pytest.approx(1.0, abs=1e-10)

    def test_large_bias(self):
        """Large consistent bias is computed correctly."""
        dates = [pd.Timestamp(f"2022-01-{d:02d}") for d in range(1, 11)]
        asos = pd.DataFrame({"date": dates, "tmax_f": [60.0] * 10})
        ghcn = pd.DataFrame({"date": dates, "TMAX": [50.0] * 10})
        metrics = compare_asos_ghcn_tmax(asos, ghcn)
        assert metrics["mean_bias_f"] == 10.0
        assert metrics["mae_f"] == 10.0

    def test_empty_asos(self):
        """Empty ASOS DataFrame returns zero overlap."""
        asos = pd.DataFrame({"date": [], "tmax_f": []})
        ghcn = pd.DataFrame({
            "date": [pd.Timestamp("2022-01-01")],
            "TMAX": [50.0],
        })
        metrics = compare_asos_ghcn_tmax(asos, ghcn)
        assert metrics["overlap_days"] == 0

    def test_nan_values_excluded(self):
        """NaN values in tmax_f or TMAX are excluded from comparison."""
        dates = [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")]
        asos = pd.DataFrame({"date": dates, "tmax_f": [50.0, np.nan]})
        ghcn = pd.DataFrame({"date": dates, "TMAX": [49.0, 60.0]})
        metrics = compare_asos_ghcn_tmax(asos, ghcn)
        assert metrics["overlap_days"] == 1


# ===========================================================================
# write_asos_ghcn_markdown tests
# ===========================================================================

class TestWriteAsosGhcnMarkdown:

    def test_empty_report(self, tmp_path):
        """Empty DataFrame writes 'no data found' message."""
        output_path = str(tmp_path / "report.md")
        write_asos_ghcn_markdown(pd.DataFrame(), output_path)
        with open(output_path) as f:
            content = f.read()
        assert "No overlapping data found" in content

    def test_normal_report(self, tmp_path):
        """Normal report has header and data row."""
        report = pd.DataFrame([{
            "station_id": "USW00094728",
            "icao": "KNYC",
            "overlap_days": 365,
            "mean_bias_f": 0.5,
            "mae_f": 1.2,
            "rmse_f": 1.5,
            "corr": 0.98,
        }])
        output_path = str(tmp_path / "report.md")
        write_asos_ghcn_markdown(report, output_path)
        with open(output_path) as f:
            content = f.read()
        assert "USW00094728" in content
        assert "KNYC" in content
        assert "365" in content
        assert "0.50" in content


# ===========================================================================
# AsosDailyConfig tests
# ===========================================================================

class TestAsosDailyConfig:

    def test_default_values(self):
        """Default config has expected values."""
        cfg = AsosDailyConfig()
        assert 18 in cfg.afternoon_hours_utc
        assert cfg.low_cloud_ceiling_ft == 5000.0

    def test_custom_parameters(self):
        """Custom parameters are stored correctly."""
        cfg = AsosDailyConfig(
            afternoon_hours_utc=(14, 15),
            low_cloud_ceiling_ft=3000.0,
        )
        assert cfg.afternoon_hours_utc == (14, 15)
        assert cfg.low_cloud_ceiling_ft == 3000.0

    def test_frozen(self):
        """Config is immutable (frozen dataclass)."""
        cfg = AsosDailyConfig()
        with pytest.raises(AttributeError):
            cfg.low_cloud_ceiling_ft = 9999.0
