"""
Tests for the GHCN-Daily data collection module.

Validates:
  - .dly fixed-width line parsing
  - Temperature conversion (tenths of C -> F)
  - Quality flag filtering
  - Date range filtering
  - Handling of missing values (-9999)
  - Pivot from long to wide format
  - File download error handling
"""

import os
import sys
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_collection import (
    tenths_c_to_fahrenheit,
    tenths_mm_to_inches,
    mm_to_inches,
    tenths_ms_to_mph,
    convert_element_value,
    parse_dly_line,
    parse_dly_file,
    pivot_station_data,
    download_dly_file,
    MISSING_VALUE,
)


# ===========================================================================
# Helper: build a synthetic .dly line
# ===========================================================================

def build_dly_line(station_id: str, year: int, month: int, element: str,
                   daily_values: list[tuple[int, str, str, str]]) -> str:
    """Build a synthetic GHCN .dly record line.

    Parameters
    ----------
    station_id : str
        11-character station ID.
    year : int
    month : int
    element : str
        4-character element type (e.g., "TMAX").
    daily_values : list[tuple[int, str, str, str]]
        List of up to 31 (value, mflag, qflag, sflag) tuples.
        Use -9999 for missing values.

    Returns
    -------
    str
        A 269-character .dly line.
    """
    # Pad station_id to 11 chars
    header = f"{station_id:<11s}{year:04d}{month:02d}{element:4s}"

    daily_part = ""
    for i in range(31):
        if i < len(daily_values):
            val, mf, qf, sf = daily_values[i]
            daily_part += f"{val:5d}{mf}{qf}{sf}"
        else:
            # Pad remaining days as missing
            daily_part += f"{MISSING_VALUE:5d}   "

    return header + daily_part


# ===========================================================================
# Temperature Conversion Tests
# ===========================================================================

class TestTemperatureConversion:
    """Test tenths_c_to_fahrenheit conversion."""

    def test_freezing_point(self):
        """0 tenths of C = 0.0 C = 32.0 F."""
        assert tenths_c_to_fahrenheit(0) == 32.0

    def test_boiling_point(self):
        """1000 tenths of C = 100.0 C = 212.0 F."""
        assert tenths_c_to_fahrenheit(1000) == 212.0

    def test_minus_40_crossover(self):
        """-400 tenths of C = -40.0 C = -40.0 F (the crossover point)."""
        assert tenths_c_to_fahrenheit(-400) == -40.0

    def test_body_temperature(self):
        """370 tenths of C = 37.0 C = 98.6 F."""
        result = tenths_c_to_fahrenheit(370)
        assert abs(result - 98.6) < 0.01

    def test_typical_summer_day(self):
        """300 tenths of C = 30.0 C = 86.0 F."""
        assert tenths_c_to_fahrenheit(300) == 86.0

    def test_typical_winter_day(self):
        """-100 tenths of C = -10.0 C = 14.0 F."""
        assert tenths_c_to_fahrenheit(-100) == 14.0

    def test_positive_small(self):
        """215 tenths of C = 21.5 C = 70.7 F."""
        result = tenths_c_to_fahrenheit(215)
        assert abs(result - 70.7) < 0.01

    def test_exact_formula(self):
        """Verify the exact formula: (value / 10) * 9/5 + 32."""
        for value in [-500, -200, -100, 0, 100, 200, 300, 500, 1000]:
            expected = (value / 10) * 9 / 5 + 32
            assert tenths_c_to_fahrenheit(value) == expected


class TestNonTemperatureConversions:
    """Test conversion helpers for precipitation, snow, and wind."""

    def test_tenths_mm_to_inches(self):
        assert abs(tenths_mm_to_inches(254) - 1.0) < 0.0001

    def test_mm_to_inches(self):
        assert abs(mm_to_inches(25.4) - 1.0) < 0.0001

    def test_tenths_ms_to_mph(self):
        assert abs(tenths_ms_to_mph(10) - 2.236936) < 0.0001

    def test_convert_element_value(self):
        value, units = convert_element_value("PRCP", 254)
        assert abs(value - 1.0) < 0.0001
        assert units == "in"


# ===========================================================================
# .dly Line Parsing Tests
# ===========================================================================

class TestParseDlyLine:
    """Test parsing of individual .dly record lines."""

    def test_basic_tmax_line(self):
        """Parse a line with valid TMAX values."""
        # 3 days of data: 25.0C, 26.5C, 27.0C, rest missing
        values = [
            (250, " ", " ", "S"),
            (265, " ", " ", "S"),
            (270, " ", " ", "S"),
        ]
        line = build_dly_line("USW00094728", 2020, 7, "TMAX", values)

        obs = parse_dly_line(line)

        assert len(obs) == 3
        assert obs[0]["station_id"] == "USW00094728"
        assert obs[0]["date"] == date(2020, 7, 1)
        assert obs[0]["element"] == "TMAX"
        assert obs[0]["value_raw"] == 250
        assert abs(obs[0]["value"] - 77.0) < 0.01
        assert obs[0]["units"] == "degF"

        assert obs[1]["date"] == date(2020, 7, 2)
        assert obs[1]["value_raw"] == 265

        assert obs[2]["date"] == date(2020, 7, 3)
        assert obs[2]["value_raw"] == 270

    def test_missing_values_excluded(self):
        """Days with -9999 (missing) should be excluded."""
        values = [
            (200, " ", " ", "S"),
            (MISSING_VALUE, " ", " ", "S"),  # Missing
            (220, " ", " ", "S"),
        ]
        line = build_dly_line("USW00094728", 2020, 1, "TMAX", values)
        obs = parse_dly_line(line)

        assert len(obs) == 2
        assert obs[0]["date"] == date(2020, 1, 1)
        assert obs[1]["date"] == date(2020, 1, 3)

    def test_quality_flag_filtering(self):
        """Observations with non-blank quality flags should be excluded."""
        values = [
            (200, " ", " ", "S"),  # Passes quality check
            (210, " ", "D", "S"),  # Failed: duplicate flag
            (220, " ", "G", "S"),  # Failed: gap check
            (230, " ", " ", "S"),  # Passes
        ]
        line = build_dly_line("USW00094728", 2020, 3, "TMAX", values)
        obs = parse_dly_line(line)

        assert len(obs) == 2
        assert obs[0]["value_raw"] == 200
        assert obs[1]["value_raw"] == 230

    def test_non_temperature_element_parsed(self):
        """Non-temperature elements should be parsed and converted."""
        values = [(100, " ", " ", "S")]
        line = build_dly_line("USW00094728", 2020, 6, "PRCP", values)
        obs = parse_dly_line(line)
        assert len(obs) == 1
        assert obs[0]["element"] == "PRCP"
        assert abs(obs[0]["value"] - (100 / 10) / 25.4) < 0.0001
        assert obs[0]["units"] == "in"

    def test_tmin_parsed(self):
        """TMIN elements should be parsed correctly."""
        values = [(50, " ", " ", "S"), (60, " ", " ", "S")]
        line = build_dly_line("USW00094728", 2020, 1, "TMIN", values)
        obs = parse_dly_line(line)

        assert len(obs) == 2
        assert obs[0]["element"] == "TMIN"
        assert obs[0]["value_raw"] == 50

    def test_february_28_days(self):
        """February in a non-leap year should have 28 days max."""
        # Provide 31 values, but only 28 should be parsed for Feb 2019
        values = [(200 + i, " ", " ", "S") for i in range(31)]
        line = build_dly_line("USW00094728", 2019, 2, "TMAX", values)
        obs = parse_dly_line(line)

        assert len(obs) == 28
        assert obs[-1]["date"] == date(2019, 2, 28)

    def test_february_29_leap_year(self):
        """February in a leap year should have 29 days."""
        values = [(200 + i, " ", " ", "S") for i in range(31)]
        line = build_dly_line("USW00094728", 2020, 2, "TMAX", values)
        obs = parse_dly_line(line)

        assert len(obs) == 29
        assert obs[-1]["date"] == date(2020, 2, 29)

    def test_negative_temperatures(self):
        """Negative temperature values should parse correctly."""
        values = [(-150, " ", " ", "S"), (-200, " ", " ", "S")]
        line = build_dly_line("USW00094728", 2020, 1, "TMIN", values)
        obs = parse_dly_line(line)

        assert len(obs) == 2
        assert obs[0]["value_raw"] == -150
        # -150 tenths C = -15.0 C = 5.0 F
        assert abs(obs[0]["value"] - 5.0) < 0.01
        assert obs[0]["units"] == "degF"

    def test_empty_line(self):
        """Empty or too-short lines should return empty list."""
        assert parse_dly_line("") == []
        assert parse_dly_line("short") == []

    def test_all_missing(self):
        """A line where all 31 values are -9999 should return empty list."""
        values = [(MISSING_VALUE, " ", " ", " ") for _ in range(31)]
        line = build_dly_line("USW00094728", 2020, 6, "TMAX", values)
        obs = parse_dly_line(line)
        assert obs == []

    def test_real_world_line_format(self):
        """Test parsing a line that matches the actual GHCN format.

        Real GHCN .dly lines are exactly 269 characters wide.
        """
        # Construct a realistic line manually
        # Station: USW00094728, Year: 2020, Month: 01, Element: TMAX
        header = "USW00094728202001TMAX"
        assert len(header) == 21

        # Day 1: 50 tenths C (5.0 C = 41.0 F), blank flags
        # Day 2: -30 tenths C, blank flags
        # Day 3: missing (-9999)
        # Days 4-31: missing
        day1 = "   50   "  # value=50, mflag=' ', qflag=' ', sflag=' '
        day2 = "  -30   "
        day3 = "-9999   "
        remaining = "-9999   " * 28
        line = header + day1 + day2 + day3 + remaining
        assert len(line) == 269

        obs = parse_dly_line(line)
        assert len(obs) == 2
        assert obs[0]["date"] == date(2020, 1, 1)
        assert obs[0]["value_raw"] == 50
        assert abs(obs[0]["value"] - 41.0) < 0.01
        assert obs[0]["units"] == "degF"
        assert obs[1]["date"] == date(2020, 1, 2)
        assert obs[1]["value_raw"] == -30


# ===========================================================================
# File Parsing Tests
# ===========================================================================

class TestParseDlyFile:
    """Test full .dly file parsing with date range filtering."""

    def _create_temp_dly(self, lines: list[str]) -> str:
        """Write lines to a temporary .dly file and return the path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False)
        for line in lines:
            tmp.write(line + "\n")
        tmp.close()
        return tmp.name

    def test_date_range_filtering(self):
        """Only records within the date range should be included."""
        # Line for Jan 2019 (outside range)
        values_2019 = [(200, " ", " ", "S")]
        line_2019 = build_dly_line("USW00094728", 2019, 1, "TMAX", values_2019)

        # Line for Jan 2020 (inside range)
        values_2020 = [(250, " ", " ", "S")]
        line_2020 = build_dly_line("USW00094728", 2020, 1, "TMAX", values_2020)

        # Line for Jan 2023 (outside range)
        values_2023 = [(300, " ", " ", "S")]
        line_2023 = build_dly_line("USW00094728", 2023, 1, "TMAX", values_2023)

        filepath = self._create_temp_dly([line_2019, line_2020, line_2023])

        try:
            df = parse_dly_file(filepath, start_date="2020-01-01",
                                end_date="2022-12-31")
            assert len(df) == 1
            assert df.iloc[0]["date"] == date(2020, 1, 1)
            assert df.iloc[0]["value_raw"] == 250
        finally:
            os.unlink(filepath)

    def test_multiple_elements(self):
        """Both TMAX and TMIN should be parsed from the same file."""
        tmax_vals = [(250, " ", " ", "S")]
        tmin_vals = [(150, " ", " ", "S")]
        line_tmax = build_dly_line("USW00094728", 2020, 6, "TMAX", tmax_vals)
        line_tmin = build_dly_line("USW00094728", 2020, 6, "TMIN", tmin_vals)

        filepath = self._create_temp_dly([line_tmax, line_tmin])

        try:
            df = parse_dly_file(filepath, start_date="2020-01-01",
                                end_date="2022-12-31")
            assert len(df) == 2
            elements = set(df["element"])
            assert elements == {"TMAX", "TMIN"}
        finally:
            os.unlink(filepath)

    def test_non_temperature_elements_included(self):
        """PRCP, SNOW, etc. should appear in output."""
        tmax_line = build_dly_line("USW00094728", 2020, 6, "TMAX",
                                   [(250, " ", " ", "S")])
        prcp_line = build_dly_line("USW00094728", 2020, 6, "PRCP",
                                   [(100, " ", " ", "S")])

        filepath = self._create_temp_dly([tmax_line, prcp_line])

        try:
            df = parse_dly_file(filepath, start_date="2020-01-01",
                                end_date="2022-12-31")
            assert "PRCP" in set(df["element"])
        finally:
            os.unlink(filepath)

    def test_empty_file(self):
        """An empty .dly file should return an empty DataFrame."""
        filepath = self._create_temp_dly([])

        try:
            df = parse_dly_file(filepath)
            assert df.empty
            assert "station_id" in df.columns
        finally:
            os.unlink(filepath)

    def test_no_date_range(self):
        """Without date range filters, all data should be returned."""
        values = [(200, " ", " ", "S"), (210, " ", " ", "S")]
        line = build_dly_line("USW00094728", 2020, 7, "TMAX", values)
        filepath = self._create_temp_dly([line])

        try:
            df = parse_dly_file(filepath)
            assert len(df) == 2
        finally:
            os.unlink(filepath)


# ===========================================================================
# Pivot Tests
# ===========================================================================

class TestPivotStationData:
    """Test long-to-wide pivoting of parsed data."""

    def test_basic_pivot(self):
        """Pivot should create TMAX and TMIN columns from long format."""
        data = pd.DataFrame([
            {"station_id": "ST1", "date": date(2020, 1, 1), "element": "TMAX",
             "value_raw": 250, "value": 77.0, "units": "degF",
             "mflag": " ", "qflag": " ", "sflag": "S"},
            {"station_id": "ST1", "date": date(2020, 1, 1), "element": "TMIN",
             "value_raw": 150, "value": 59.0, "units": "degF",
             "mflag": " ", "qflag": " ", "sflag": "S"},
            {"station_id": "ST1", "date": date(2020, 1, 2), "element": "TMAX",
             "value_raw": 260, "value": 78.8, "units": "degF",
             "mflag": " ", "qflag": " ", "sflag": "S"},
        ])

        pivot = pivot_station_data(data)

        assert "TMAX" in pivot.columns
        assert "TMIN" in pivot.columns
        assert len(pivot) == 2  # 2 dates
        assert pivot.loc[date(2020, 1, 1), "TMAX"] == 77.0
        assert pivot.loc[date(2020, 1, 1), "TMIN"] == 59.0

    def test_pivot_empty(self):
        """Pivoting empty data should return empty DataFrame."""
        data = pd.DataFrame(columns=[
            "station_id", "date", "element", "value_raw",
            "value", "units", "mflag", "qflag", "sflag"
        ])
        pivot = pivot_station_data(data)
        assert pivot.empty


# ===========================================================================
# Download Tests (mocked)
# ===========================================================================

class TestDownload:
    """Test .dly file download with mocked network."""

    @patch("src.data_collection.requests.get")
    def test_download_success(self, mock_get):
        """Successful download should save .dly file."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"fake dly content"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            path = download_dly_file(
                "USW00094728", tmpdir,
                base_url="https://example.com/"
            )
            assert os.path.exists(path)
            assert path.endswith("USW00094728.dly")
            with open(path, "rb") as f:
                assert f.read() == b"fake dly content"

    @patch("src.data_collection.requests.get")
    def test_download_http_error(self, mock_get):
        """HTTP errors should raise requests.HTTPError."""
        import requests as req
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(req.HTTPError):
                download_dly_file(
                    "NONEXISTENT", tmpdir,
                    base_url="https://example.com/"
                )


# ===========================================================================
# Integration-style test with a full mini file
# ===========================================================================

class TestEndToEnd:
    """Integration test: write a mini .dly file, parse it, pivot it."""

    def test_full_flow(self):
        """Parse a multi-month .dly file and verify the output."""
        # January 2020: days 1-3 have TMAX
        jan_tmax = [(200, " ", " ", "S"), (210, " ", " ", "S"), (220, " ", " ", "S")]
        jan_tmin = [(50, " ", " ", "S"), (60, " ", " ", "S"), (70, " ", " ", "S")]

        # July 2020: days 1-2 have TMAX, day 3 missing
        jul_tmax = [(300, " ", " ", "S"), (310, " ", " ", "S"),
                     (MISSING_VALUE, " ", " ", "S")]

        lines = [
            build_dly_line("USW00094728", 2020, 1, "TMAX", jan_tmax),
            build_dly_line("USW00094728", 2020, 1, "TMIN", jan_tmin),
            build_dly_line("USW00094728", 2020, 7, "TMAX", jul_tmax),
        ]

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False)
        for line in lines:
            tmp.write(line + "\n")
        tmp.close()

        try:
            df = parse_dly_file(tmp.name, "2020-01-01", "2020-12-31")
            assert len(df) > 0

            pivot = pivot_station_data(df)

            # Jan 1-3 should have both TMAX and TMIN
            assert pivot.loc[date(2020, 1, 1), "TMAX"] == tenths_c_to_fahrenheit(200)
            assert pivot.loc[date(2020, 1, 1), "TMIN"] == tenths_c_to_fahrenheit(50)

            # Jul 1-2 should have TMAX, Jul 3 should be missing
            assert pivot.loc[date(2020, 7, 1), "TMAX"] == tenths_c_to_fahrenheit(300)
            assert pivot.loc[date(2020, 7, 2), "TMAX"] == tenths_c_to_fahrenheit(310)
            # Jul 3 was -9999, so it shouldn't exist
            if date(2020, 7, 3) in pivot.index:
                assert pd.isna(pivot.loc[date(2020, 7, 3), "TMAX"])
        finally:
            os.unlink(tmp.name)
