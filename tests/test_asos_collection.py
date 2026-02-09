"""
Tests for ASOS collection helpers.

Covers:
  - load_asos_station_map() — valid CSV, missing columns, empty, filtering
  - build_asos_request_url() — URL structure, custom fields, date parsing
  - download_asos_station() — mock HTTP, caching, HTTP errors
  - download_asos_station_range() — chunked downloads, single chunk
  - collect_asos_data() — mock end-to-end, empty mapping
  - iter_date_chunks() — edge cases
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.asos_collection import (
    load_asos_station_map,
    build_asos_request_url,
    download_asos_station,
    download_asos_station_range,
    collect_asos_data,
    iter_date_chunks,
    IEM_BASE_URL,
    DEFAULT_ASOS_FIELDS,
)


# ===========================================================================
# iter_date_chunks tests (original 3 + new edge cases)
# ===========================================================================

def test_iter_date_chunks_single_year():
    chunks = iter_date_chunks("2020-01-01", "2020-12-31", chunk_years=1)
    assert chunks == [("2020-01-01", "2020-12-31")]


def test_iter_date_chunks_multiple_years():
    chunks = iter_date_chunks("2019-06-01", "2021-02-10", chunk_years=1)
    assert chunks == [
        ("2019-06-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-02-10"),
    ]


def test_iter_date_chunks_two_year_blocks():
    chunks = iter_date_chunks("2018-01-01", "2021-12-31", chunk_years=2)
    assert chunks == [
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
    ]


def test_iter_date_chunks_same_day():
    """Start and end on the same day yields one chunk."""
    chunks = iter_date_chunks("2022-06-15", "2022-06-15", chunk_years=1)
    assert len(chunks) == 1
    assert chunks[0] == ("2022-06-15", "2022-06-15")


def test_iter_date_chunks_invalid_chunk_years():
    """chunk_years < 1 raises ValueError."""
    with pytest.raises(ValueError, match="chunk_years must be >= 1"):
        iter_date_chunks("2020-01-01", "2020-12-31", chunk_years=0)


def test_iter_date_chunks_end_before_start():
    """When end < start, returns empty list."""
    chunks = iter_date_chunks("2022-06-01", "2022-01-01", chunk_years=1)
    assert chunks == []


# ===========================================================================
# load_asos_station_map tests
# ===========================================================================

class TestLoadAsosStationMap:

    def test_valid_csv(self, tmp_path):
        """Loads station mappings from a well-formed CSV."""
        csv_path = tmp_path / "mapping.csv"
        csv_path.write_text(
            "station_id,icao,asos_available\n"
            "USW00094728,KNYC,yes\n"
            "USW00014735,KALB,yes\n"
        )
        result = load_asos_station_map(str(csv_path))
        assert result == {
            "USW00094728": "KNYC",
            "USW00014735": "KALB",
        }

    def test_missing_columns(self, tmp_path):
        """Raises KeyError when required columns (station_id/icao) are missing
        but asos_available is 'yes'."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("name,code,asos_available\nAlbany,KALB,yes\n")
        with pytest.raises(KeyError):
            load_asos_station_map(str(csv_path))

    def test_empty_csv(self, tmp_path):
        """Header-only CSV returns empty dict."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("station_id,icao,asos_available\n")
        result = load_asos_station_map(str(csv_path))
        assert result == {}

    def test_filters_asos_unavailable(self, tmp_path):
        """Stations with asos_available != 'yes' are excluded."""
        csv_path = tmp_path / "mixed.csv"
        csv_path.write_text(
            "station_id,icao,asos_available\n"
            "USW00094728,KNYC,yes\n"
            "USW00014735,KALB,no\n"
            "USW00013739,KPHL,\n"
        )
        result = load_asos_station_map(str(csv_path))
        assert "USW00094728" in result
        assert "USW00014735" not in result
        assert "USW00013739" not in result


# ===========================================================================
# build_asos_request_url tests
# ===========================================================================

class TestBuildAsosRequestUrl:

    def test_basic_url_structure(self):
        """URL contains station, dates, and all default fields."""
        url = build_asos_request_url("KNYC", "2022-01-01", "2022-12-31")
        assert url.startswith(IEM_BASE_URL)
        assert "station=KNYC" in url
        assert "year1=2022" in url
        assert "month1=1" in url
        assert "day1=1" in url
        assert "year2=2022" in url
        assert "month2=12" in url
        assert "day2=31" in url
        for field in DEFAULT_ASOS_FIELDS:
            assert f"data={field}" in url

    def test_custom_fields(self):
        """Custom field list replaces defaults."""
        url = build_asos_request_url(
            "KALB", "2022-01-01", "2022-06-30",
            data_fields=["tmpf", "dwpf"]
        )
        assert "data=tmpf" in url
        assert "data=dwpf" in url
        # Default fields NOT in custom list should be absent
        assert "data=ceil" not in url

    def test_format_csv(self):
        """URL specifies CSV format."""
        url = build_asos_request_url("KNYC", "2022-01-01", "2022-01-31")
        assert "format=csv" in url

    def test_utc_timezone(self):
        """URL specifies UTC timezone."""
        url = build_asos_request_url("KNYC", "2022-01-01", "2022-01-31")
        assert "tz=Etc/UTC" in url


# ===========================================================================
# download_asos_station tests
# ===========================================================================

class TestDownloadAsosStation:

    @patch("src.asos_collection.requests.get")
    def test_download_success(self, mock_get, tmp_path):
        """Successful download writes CSV to output_path."""
        mock_response = MagicMock()
        mock_response.content = b"station,valid,tmpf\nKNYC,2022-01-01 00:00,50\n"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = download_asos_station(
            "KNYC", str(tmp_path), "2022-01-01", "2022-01-31"
        )
        assert os.path.exists(result)
        with open(result) as f:
            assert "KNYC" in f.read()

    @patch("src.asos_collection.requests.get")
    def test_cached_file_skipping(self, mock_get, tmp_path):
        """Pre-existing non-empty file is returned without making HTTP request."""
        cached = tmp_path / "KNYC.csv"
        cached.write_text("station,valid,tmpf\nKNYC,2022-01-01 00:00,50\n")

        result = download_asos_station(
            "KNYC", str(tmp_path), "2022-01-01", "2022-01-31"
        )
        assert result == str(cached)
        mock_get.assert_not_called()

    @patch("src.asos_collection.requests.get")
    def test_http_error(self, mock_get, tmp_path):
        """HTTP errors propagate as exceptions."""
        import requests
        mock_get.side_effect = requests.HTTPError("500 Server Error")

        with pytest.raises(requests.HTTPError):
            download_asos_station(
                "KNYC", str(tmp_path), "2022-01-01", "2022-01-31"
            )


# ===========================================================================
# download_asos_station_range tests
# ===========================================================================

class TestDownloadAsosStationRange:

    @patch("src.asos_collection.download_asos_station")
    def test_single_chunk(self, mock_dl, tmp_path):
        """A short date range results in a single download call."""
        mock_dl.return_value = str(tmp_path / "KNYC_2022-01-01_2022-06-30.csv")

        paths = download_asos_station_range(
            "KNYC", str(tmp_path), "2022-01-01", "2022-06-30",
            chunk_years=1,
        )
        assert len(paths) == 1
        assert mock_dl.call_count == 1

    @patch("src.asos_collection.download_asos_station")
    def test_chunked_downloads(self, mock_dl, tmp_path):
        """Multi-year range is split into year-sized chunks."""
        mock_dl.side_effect = lambda **kwargs: kwargs.get("output_path", "dummy.csv")

        paths = download_asos_station_range(
            "KNYC", str(tmp_path), "2020-01-01", "2022-12-31",
            chunk_years=1,
        )
        assert len(paths) == 3
        assert mock_dl.call_count == 3


# ===========================================================================
# collect_asos_data tests
# ===========================================================================

class TestCollectAsosData:

    @patch("src.asos_collection.download_asos_station_range")
    @patch("src.asos_collection.load_asos_station_map")
    def test_end_to_end(self, mock_map, mock_dl, tmp_path):
        """Collects data for all mapped stations."""
        mock_map.return_value = {"USW00094728": "KNYC", "USW00014735": "KALB"}
        mock_dl.return_value = [str(tmp_path / "output.csv")]

        results = collect_asos_data(
            mapping_csv="dummy.csv",
            output_dir=str(tmp_path),
            start_date="2022-01-01",
            end_date="2022-12-31",
        )
        assert len(results) == 2
        assert mock_dl.call_count == 2

    @patch("src.asos_collection.load_asos_station_map")
    def test_empty_mapping(self, mock_map, tmp_path):
        """Empty station map returns empty results."""
        mock_map.return_value = {}

        results = collect_asos_data(
            mapping_csv="dummy.csv",
            output_dir=str(tmp_path),
        )
        assert results == {}
