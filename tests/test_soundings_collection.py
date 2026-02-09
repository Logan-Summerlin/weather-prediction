"""
Tests for IGRA soundings collection module.

Covers:
  - _daterange() iteration and boundary cases
  - _require_siphon() import failure handling
  - download_soundings() with mocked Siphon calls, caching, error handling
  - Edge cases: empty ranges, file naming, output paths
"""

import os
import sys
from datetime import datetime
from types import ModuleType
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.soundings_collection import (
    _daterange,
    _require_siphon,
    download_soundings,
)


# ===========================================================================
# _daterange tests
# ===========================================================================

class TestDaterange:
    """Tests for the _daterange() date iterator."""

    def test_single_day(self):
        """A single-day range yields exactly one datetime."""
        dates = list(_daterange("2022-01-15", "2022-01-15"))
        assert len(dates) == 1
        assert dates[0] == datetime(2022, 1, 15)

    def test_multiple_days(self):
        """Multi-day range yields correct count and boundaries."""
        dates = list(_daterange("2022-01-01", "2022-01-05"))
        assert len(dates) == 5
        assert dates[0] == datetime(2022, 1, 1)
        assert dates[-1] == datetime(2022, 1, 5)

    def test_month_boundary(self):
        """Range crossing a month boundary is contiguous."""
        dates = list(_daterange("2022-01-30", "2022-02-02"))
        assert len(dates) == 4
        assert dates[1] == datetime(2022, 1, 31)
        assert dates[2] == datetime(2022, 2, 1)

    def test_year_boundary(self):
        """Range crossing a year boundary works correctly."""
        dates = list(_daterange("2021-12-30", "2022-01-02"))
        assert len(dates) == 4
        assert dates[1] == datetime(2021, 12, 31)
        assert dates[2] == datetime(2022, 1, 1)

    def test_empty_range(self):
        """Start date after end date yields no results."""
        dates = list(_daterange("2022-01-10", "2022-01-05"))
        assert len(dates) == 0

    def test_leap_year(self):
        """Range spanning Feb 28-29 in a leap year works."""
        dates = list(_daterange("2020-02-28", "2020-03-01"))
        assert len(dates) == 3
        assert dates[1] == datetime(2020, 2, 29)

    def test_long_range(self):
        """A 365-day range yields exactly 365 dates."""
        dates = list(_daterange("2022-01-01", "2022-12-31"))
        assert len(dates) == 365


# ===========================================================================
# _require_siphon tests
# ===========================================================================

class TestRequireSiphon:
    """Tests for the _require_siphon() dependency check."""

    def test_siphon_available(self):
        """No error when siphon is importable."""
        mock_igra2 = MagicMock()
        mock_siphon = MagicMock()
        mock_sws = MagicMock()
        with patch.dict("sys.modules", {
            "siphon": mock_siphon,
            "siphon.simplewebservice": mock_sws,
            "siphon.simplewebservice.igra2": mock_igra2,
        }):
            _require_siphon()

    def test_siphon_missing_raises_import_error(self):
        """ImportError raised with helpful message when siphon is missing."""
        # Save original module references if they exist
        saved = {}
        for mod_name in ["siphon", "siphon.simplewebservice", "siphon.simplewebservice.igra2"]:
            if mod_name in sys.modules:
                saved[mod_name] = sys.modules[mod_name]

        try:
            # Remove siphon from sys.modules and set to None to block import
            for mod_name in ["siphon", "siphon.simplewebservice", "siphon.simplewebservice.igra2"]:
                sys.modules[mod_name] = None
            with pytest.raises(ImportError, match="siphon"):
                _require_siphon()
        finally:
            # Restore original state
            for mod_name in ["siphon", "siphon.simplewebservice", "siphon.simplewebservice.igra2"]:
                if mod_name in saved:
                    sys.modules[mod_name] = saved[mod_name]
                else:
                    sys.modules.pop(mod_name, None)


# ===========================================================================
# Helper to set up siphon mocking
# ===========================================================================

def _mock_siphon_modules():
    """Create mock siphon module hierarchy for sys.modules patching."""
    sample_df = pd.DataFrame({
        "pressure": [1000.0, 850.0, 500.0],
        "temperature": [20.0, 10.0, -20.0],
        "height": [100, 1500, 5500],
        "dewpoint": [15.0, 5.0, -30.0],
        "direction": [180, 270, 300],
        "speed": [5, 15, 25],
    })
    sample_header = {"station": "USM00072501", "lat": 40.87}

    mock_igra_class = MagicMock()
    mock_igra_class.request_data.return_value = (sample_df, sample_header)

    mock_igra2 = MagicMock()
    mock_igra2.IGRAUpperAir = mock_igra_class

    mock_sws = MagicMock()
    mock_sws.igra2 = mock_igra2

    mock_siphon = MagicMock()
    mock_siphon.simplewebservice = mock_sws
    mock_siphon.simplewebservice.igra2 = mock_igra2

    modules = {
        "siphon": mock_siphon,
        "siphon.simplewebservice": mock_sws,
        "siphon.simplewebservice.igra2": mock_igra2,
    }
    return modules, mock_igra_class


# ===========================================================================
# download_soundings tests
# ===========================================================================

class TestDownloadSoundings:
    """Tests for the download_soundings() function."""

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        output_dir = str(tmp_path / "new_dir")
        modules, _ = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            # Use an empty date range so no actual downloads happen
            download_soundings(
                station_id="USM00072501",
                start_date="2022-01-02",
                end_date="2022-01-01",
                output_dir=output_dir,
            )
        assert os.path.isdir(output_dir)

    def test_empty_date_range(self, tmp_path):
        """Empty date range (start > end) returns empty list."""
        modules, _ = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id="USM00072501",
                start_date="2022-01-10",
                end_date="2022-01-05",
                output_dir=str(tmp_path),
            )
        assert result == []

    def test_cached_file_skipping(self, tmp_path):
        """Pre-existing non-empty files are skipped (cache hit)."""
        station_id = "USM00072501"
        cached_path = tmp_path / f"{station_id}_2022011500.csv"
        cached_path.write_text("pressure,temperature\n1000,20\n")

        modules, mock_igra_class = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id=station_id,
                start_date="2022-01-15",
                end_date="2022-01-15",
                hours=(0,),
                output_dir=str(tmp_path),
            )

        assert len(result) == 1
        assert str(cached_path) in result[0]
        mock_igra_class.request_data.assert_not_called()

    def test_download_success(self, tmp_path):
        """Successful download saves CSV and header JSON files."""
        modules, _ = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id="USM00072501",
                start_date="2022-01-15",
                end_date="2022-01-15",
                hours=(0,),
                output_dir=str(tmp_path),
            )

        assert len(result) == 1
        assert os.path.exists(result[0])
        header_path = result[0].replace(".csv", ".header.json")
        assert os.path.exists(header_path)

    def test_download_multiple_hours(self, tmp_path):
        """Both 00Z and 12Z soundings are downloaded for each day."""
        modules, _ = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id="USM00072501",
                start_date="2022-01-15",
                end_date="2022-01-15",
                hours=(0, 12),
                output_dir=str(tmp_path),
            )

        assert len(result) == 2
        filenames = [os.path.basename(f) for f in result]
        assert "USM00072501_2022011500.csv" in filenames
        assert "USM00072501_2022011512.csv" in filenames

    def test_download_error_handling(self, tmp_path):
        """Network errors are logged but don't crash the pipeline."""
        modules, mock_igra_class = _mock_siphon_modules()
        mock_igra_class.request_data.side_effect = Exception("Network timeout")
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id="USM00072501",
                start_date="2022-01-15",
                end_date="2022-01-15",
                hours=(0,),
                output_dir=str(tmp_path),
            )

        assert len(result) == 0

    def test_output_filename_format(self, tmp_path):
        """Output files follow the {station}_{YYYYMMDDhh}.csv pattern."""
        modules, _ = _mock_siphon_modules()
        with patch.dict("sys.modules", modules):
            result = download_soundings(
                station_id="USM00072501",
                start_date="2022-03-05",
                end_date="2022-03-05",
                hours=(12,),
                output_dir=str(tmp_path),
            )

        assert len(result) == 1
        assert os.path.basename(result[0]) == "USM00072501_2022030512.csv"
