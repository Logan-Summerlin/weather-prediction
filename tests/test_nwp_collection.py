"""
Comprehensive tests for NWP data collection (GFS/GEFS via Herbie).

Tests cover:
  - _require_herbie: import success and failure
  - _build_herbie_search: all variable mappings, unknown vars, edge cases
  - download_gfs_point: mock Herbie object, parameter passing, output dir
  - download_gfs_range: date iteration, per-day error handling
  - download_gefs_reforecast_point: mock Herbie with member parameter
  - download_gefs_reforecast_range: date range with partial failures
  - Edge cases: single-day range, invalid parameters
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.nwp_collection import (
    _require_herbie,
    _build_herbie_search,
    download_gfs_point,
    download_gfs_range,
    download_gefs_reforecast_point,
    download_gefs_reforecast_range,
)


def _make_herbie_mock():
    """Create a mock herbie module and Herbie class for sys.modules injection."""
    mock_herbie_module = MagicMock()
    mock_herbie_instance = MagicMock()
    mock_herbie_instance.download.return_value = "/fake/path.grib2"
    mock_herbie_module.Herbie.return_value = mock_herbie_instance
    return mock_herbie_module, mock_herbie_instance


# ============================================================================
# _require_herbie
# ============================================================================


class TestRequireHerbie:
    """Tests for Herbie availability check."""

    def test_require_herbie_import_success(self):
        """When herbie is importable, _require_herbie should not raise."""
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"herbie": mock_module}):
            _require_herbie()

    def test_require_herbie_import_failure(self):
        """When herbie is not installed, should raise ImportError with message."""
        with patch.dict("sys.modules", {"herbie": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="herbie-data"):
                    _require_herbie()


# ============================================================================
# _build_herbie_search
# ============================================================================


class TestBuildHerbieSearch:
    """Tests for Herbie search-string construction."""

    def test_all_known_variables(self):
        """All NWP_VARIABLES from config should map to Herbie search patterns."""
        variables = [
            "tmax_2m", "tmp_850", "ugrd_10m", "vgrd_10m",
            "tcdc_eatm", "mslp", "apcp",
        ]
        result = _build_herbie_search(variables)
        assert result is not None
        assert ":TMAX:2 m" in result
        assert ":TMP:850 mb" in result
        assert ":UGRD:10 m" in result
        assert ":VGRD:10 m" in result
        assert ":TCDC:entire atmosphere" in result
        assert ":MSLP:mean sea level" in result
        assert ":APCP:surface" in result

    def test_single_variable(self):
        """A single known variable should return a single pattern."""
        result = _build_herbie_search(["tmax_2m"])
        assert result == r":TMAX:2 m"

    def test_unknown_variable_passed_through(self):
        """Unknown variable names should be passed through verbatim."""
        result = _build_herbie_search(["custom_var"])
        assert result == "custom_var"

    def test_mix_known_and_unknown(self):
        """Mix of known and unknown variables."""
        result = _build_herbie_search(["tmax_2m", "custom_var"])
        assert result is not None
        assert ":TMAX:2 m" in result
        assert "custom_var" in result
        assert "|" in result

    def test_none_input(self):
        """None input should return None."""
        result = _build_herbie_search(None)
        assert result is None

    def test_empty_list(self):
        """Empty list should return None."""
        result = _build_herbie_search([])
        assert result is None

    def test_pipe_separator(self):
        """Multiple variables should be joined by | separator."""
        result = _build_herbie_search(["tmax_2m", "mslp"])
        assert result is not None
        parts = result.split("|")
        assert len(parts) == 2


# ============================================================================
# download_gfs_point
# ============================================================================


class TestDownloadGfsPoint:
    """Tests for single GFS forecast download."""

    def test_download_gfs_point_basic(self, tmp_path):
        """Basic download should instantiate Herbie with correct params."""
        mock_module, mock_instance = _make_herbie_mock()
        mock_instance.download.return_value = str(tmp_path / "gfs.grib2")

        with patch.dict("sys.modules", {"herbie": mock_module}):
            date = datetime(2022, 1, 1, 0)
            result = download_gfs_point(
                date=date, fxx=24, variables=["tmax_2m"],
                output_dir=str(tmp_path), model="gfs",
            )

        mock_module.Herbie.assert_called_once_with(
            date, model="gfs", fxx=24, product="pgrb2.0p25",
        )
        mock_instance.download.assert_called_once()
        assert result == str(tmp_path / "gfs.grib2")

    def test_download_gfs_point_creates_output_dir(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        output_dir = str(tmp_path / "new_dir")
        mock_module, mock_instance = _make_herbie_mock()

        with patch.dict("sys.modules", {"herbie": mock_module}):
            download_gfs_point(
                date=datetime(2022, 1, 1),
                output_dir=output_dir,
            )

        assert Path(output_dir).exists()

    def test_download_gfs_point_search_string(self, tmp_path):
        """The search parameter should be built from variables."""
        mock_module, mock_instance = _make_herbie_mock()

        with patch.dict("sys.modules", {"herbie": mock_module}):
            download_gfs_point(
                date=datetime(2022, 1, 1),
                variables=["tmax_2m", "mslp"],
                output_dir=str(tmp_path),
            )

        # Herbie.download should have been called with a search string
        mock_instance.download.assert_called_once()
        call_kwargs = mock_instance.download.call_args
        search_arg = call_kwargs[1].get("search") if call_kwargs[1] else None
        assert search_arg is not None
        assert ":TMAX:2 m" in search_arg
        assert ":MSLP:mean sea level" in search_arg


# ============================================================================
# download_gfs_range
# ============================================================================


class TestDownloadGfsRange:
    """Tests for GFS range download (daily cadence)."""

    @patch("src.nwp_collection.download_gfs_point")
    def test_range_single_day(self, mock_point):
        """Single-day range should call download_gfs_point once."""
        mock_point.return_value = "path.grib2"
        paths = download_gfs_range("2022-01-01", "2022-01-01")
        assert mock_point.call_count == 1
        assert len(paths) == 1

    @patch("src.nwp_collection.download_gfs_point")
    def test_range_multiple_days(self, mock_point):
        """Multi-day range should iterate over each day."""
        mock_point.return_value = "path.grib2"
        paths = download_gfs_range("2022-01-01", "2022-01-03")
        assert mock_point.call_count == 3
        assert len(paths) == 3

    @patch("src.nwp_collection.download_gfs_point")
    def test_range_with_failure(self, mock_point):
        """If one day fails, the rest should still proceed."""
        mock_point.side_effect = [
            "path1.grib2",
            Exception("Download failed"),
            "path3.grib2",
        ]
        paths = download_gfs_range("2022-01-01", "2022-01-03")
        assert len(paths) == 2
        assert mock_point.call_count == 3

    @patch("src.nwp_collection.download_gfs_point")
    def test_range_cycle_hour(self, mock_point):
        """Cycle hour should be passed to each download call."""
        mock_point.return_value = "path.grib2"
        download_gfs_range("2022-01-01", "2022-01-01", cycle_hour=12)
        call_args = mock_point.call_args
        assert call_args[1]["date"].hour == 12


# ============================================================================
# download_gefs_reforecast_point
# ============================================================================


class TestDownloadGefsReforecastPoint:
    """Tests for single GEFS reforecast download."""

    def test_gefs_point_with_member(self, tmp_path):
        """GEFS download should pass the member parameter to Herbie."""
        mock_module, mock_instance = _make_herbie_mock()

        with patch.dict("sys.modules", {"herbie": mock_module}):
            download_gefs_reforecast_point(
                date=datetime(2022, 1, 1),
                fxx=24,
                member=5,
                output_dir=str(tmp_path),
            )

        call_kwargs = mock_module.Herbie.call_args[1]
        assert call_kwargs["member"] == 5
        assert call_kwargs["model"] == "gefs_reforecast"
        assert call_kwargs["variable_level"] == "pgrb2"

    def test_gefs_point_default_member(self, tmp_path):
        """Default member should be 0 (control run)."""
        mock_module, mock_instance = _make_herbie_mock()

        with patch.dict("sys.modules", {"herbie": mock_module}):
            download_gefs_reforecast_point(
                date=datetime(2022, 1, 1),
                output_dir=str(tmp_path),
            )

        call_kwargs = mock_module.Herbie.call_args[1]
        assert call_kwargs["member"] == 0


# ============================================================================
# download_gefs_reforecast_range
# ============================================================================


class TestDownloadGefsReforecastRange:
    """Tests for GEFS reforecast range download."""

    @patch("src.nwp_collection.download_gefs_reforecast_point")
    def test_gefs_range_basic(self, mock_point):
        """Basic range download should iterate over dates."""
        mock_point.return_value = "gefs.grib2"
        paths = download_gefs_reforecast_range("2022-01-01", "2022-01-02")
        assert mock_point.call_count == 2
        assert len(paths) == 2

    @patch("src.nwp_collection.download_gefs_reforecast_point")
    def test_gefs_range_with_failures(self, mock_point):
        """Partial failures should not stop the range download."""
        mock_point.side_effect = [
            Exception("Server error"),
            "path2.grib2",
            "path3.grib2",
        ]
        paths = download_gefs_reforecast_range("2022-01-01", "2022-01-03")
        assert len(paths) == 2
        assert mock_point.call_count == 3

    @patch("src.nwp_collection.download_gefs_reforecast_point")
    def test_gefs_range_passes_member(self, mock_point):
        """Member index should propagate to each point download."""
        mock_point.return_value = "gefs.grib2"
        download_gefs_reforecast_range(
            "2022-01-01", "2022-01-01", member=3,
        )
        call_kwargs = mock_point.call_args[1]
        assert call_kwargs["member"] == 3

    @patch("src.nwp_collection.download_gefs_reforecast_point")
    def test_gefs_range_single_day(self, mock_point):
        """Single-day GEFS range should call exactly once."""
        mock_point.return_value = "gefs.grib2"
        paths = download_gefs_reforecast_range("2022-06-15", "2022-06-15")
        assert mock_point.call_count == 1
        assert len(paths) == 1
