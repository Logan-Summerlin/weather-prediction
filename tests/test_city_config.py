"""
Tests for multi-city configuration framework.

Tests the CityConfig dataclass, city registry, bucket index computation,
and directory management for NYC, Philadelphia, and Chicago configurations.
"""

import os
import sys
import tempfile
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import (
    CityConfig,
    get_city_config,
    list_cities,
    get_bucket_index,
    ensure_city_dirs,
)


class TestListCities:
    """Tests for list_cities()."""

    def test_returns_sorted_list(self):
        cities = list_cities()
        assert cities == sorted(cities)

    def test_contains_all_three_cities(self):
        cities = list_cities()
        assert "nyc" in cities
        assert "phl" in cities
        assert "chi" in cities

    def test_returns_list_of_strings(self):
        cities = list_cities()
        assert all(isinstance(c, str) for c in cities)


class TestGetCityConfig:
    """Tests for get_city_config()."""

    def test_nyc_config(self):
        cfg = get_city_config("nyc")
        assert cfg.city_code == "nyc"
        assert cfg.kalshi_ticker == "KXHIGHNY"
        assert cfg.target_station == "USW00094728"
        assert abs(cfg.target_lat - 40.7789) < 0.001
        assert abs(cfg.target_lon - (-73.9692)) < 0.001

    def test_phl_config(self):
        cfg = get_city_config("phl")
        assert cfg.city_code == "phl"
        assert cfg.kalshi_ticker == "KXHIGHPHL"
        assert cfg.target_station == "USW00013739"
        assert abs(cfg.target_lat - 39.8733) < 0.001

    def test_chi_config(self):
        cfg = get_city_config("chi")
        assert cfg.city_code == "chi"
        assert cfg.kalshi_ticker == "KXHIGHCHI"
        assert cfg.target_station == "USW00094846"
        assert abs(cfg.target_lat - 41.9742) < 0.001

    def test_case_insensitive(self):
        cfg1 = get_city_config("NYC")
        cfg2 = get_city_config("nyc")
        assert cfg1.city_code == cfg2.city_code

    def test_whitespace_stripped(self):
        cfg = get_city_config(" phl ")
        assert cfg.city_code == "phl"

    def test_invalid_city_raises(self):
        with pytest.raises(ValueError, match="Unknown city code"):
            get_city_config("xyz")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            get_city_config("")


class TestCityConfigFields:
    """Tests for CityConfig dataclass fields."""

    @pytest.fixture(params=["nyc", "phl", "chi"])
    def city_cfg(self, request):
        return get_city_config(request.param)

    def test_has_required_fields(self, city_cfg):
        required = [
            "city_name", "city_code", "kalshi_ticker",
            "target_station", "target_station_name",
            "target_lat", "target_lon", "timezone",
            "igra_station_id", "igra_station_name",
            "nwp_lat", "nwp_lon",
            "bucket_edges", "bucket_labels",
            "monthly_tmax_mean", "monthly_tmax_std",
            "data_dir", "models_dir", "results_dir",
        ]
        for field in required:
            assert hasattr(city_cfg, field), f"Missing field: {field}"

    def test_bucket_edges_and_labels_same_length(self, city_cfg):
        assert len(city_cfg.bucket_edges) == len(city_cfg.bucket_labels)

    def test_bucket_edges_cover_full_range(self, city_cfg):
        edges = city_cfg.bucket_edges
        assert edges[0][0] == -999  # first bucket starts at -infinity sentinel
        assert edges[-1][1] == 999  # last bucket ends at +infinity sentinel

    def test_bucket_edges_contiguous(self, city_cfg):
        edges = city_cfg.bucket_edges
        for i in range(len(edges) - 1):
            assert edges[i][1] == edges[i+1][0], f"Gap between bucket {i} and {i+1}"

    def test_monthly_means_12_months(self, city_cfg):
        assert len(city_cfg.monthly_tmax_mean) == 12
        assert set(city_cfg.monthly_tmax_mean.keys()) == set(range(1, 13))

    def test_monthly_stds_positive(self, city_cfg):
        for month, std in city_cfg.monthly_tmax_std.items():
            assert std > 0, f"Non-positive std for month {month}"

    def test_timezone_is_valid_string(self, city_cfg):
        assert isinstance(city_cfg.timezone, str)
        assert "/" in city_cfg.timezone  # e.g., "America/New_York"

    def test_station_ids_are_strings(self, city_cfg):
        assert isinstance(city_cfg.target_station, str)
        assert city_cfg.target_station.startswith("US")


class TestBucketEdges:
    """Tests specific to bucket configurations per city."""

    def test_nyc_has_57_buckets(self):
        cfg = get_city_config("nyc")
        assert len(cfg.bucket_edges) == 57

    def test_phl_has_57_buckets(self):
        cfg = get_city_config("phl")
        assert len(cfg.bucket_edges) == 57

    def test_chi_has_62_buckets(self):
        """Chicago needs extra low buckets for colder winters (-10 floor)."""
        cfg = get_city_config("chi")
        assert len(cfg.bucket_edges) == 62

    def test_nyc_first_bucket(self):
        cfg = get_city_config("nyc")
        # NYC first bucket: "Below 0" with sentinel lower bound
        assert cfg.bucket_edges[0] == (-999, 0.0)

    def test_phl_first_bucket(self):
        cfg = get_city_config("phl")
        # PHL first bucket: "Below 0" with sentinel lower bound
        assert cfg.bucket_edges[0] == (-999, 0.0)

    def test_chi_lower_floor(self):
        cfg = get_city_config("chi")
        # CHI first bucket: "Below -10" with sentinel lower bound
        assert cfg.bucket_edges[0] == (-999, -10.0)

    def test_phl_matches_nyc_buckets(self):
        nyc = get_city_config("nyc")
        phl = get_city_config("phl")
        assert nyc.bucket_edges == phl.bucket_edges


class TestGetBucketIndex:
    """Tests for get_bucket_index()."""

    def test_nyc_below_0(self):
        edges = get_city_config("nyc").bucket_edges
        assert get_bucket_index(-5, edges) == 0
        assert get_bucket_index(-10, edges) == 0

    def test_nyc_middle_bucket(self):
        edges = get_city_config("nyc").bucket_edges
        # 55°F: floor((55-0)/2) + 1 = 28
        assert get_bucket_index(55, edges) == 28

    def test_nyc_above_110(self):
        edges = get_city_config("nyc").bucket_edges
        # 105°F falls in the (104,106) bucket: floor((105-0)/2) + 1 = 53
        assert get_bucket_index(105, edges) == 53

    def test_boundary_lower_inclusive(self):
        edges = get_city_config("nyc").bucket_edges
        # 0°F is the start of the (0,2) bucket at index 1
        assert get_bucket_index(0, edges) == 1
        # 2°F is the start of the (2,4) bucket at index 2
        assert get_bucket_index(2, edges) == 2

    def test_chi_extra_low_bucket(self):
        edges = get_city_config("chi").bucket_edges
        assert get_bucket_index(-15, edges) == 0   # Below -10
        # 15°F: floor((15-(-10))/2) + 1 = 13
        assert get_bucket_index(15, edges) == 13

    def test_all_temperatures_have_bucket(self):
        """Every temperature should map to exactly one bucket."""
        for city in list_cities():
            edges = get_city_config(city).bucket_edges
            for temp in range(-20, 120):
                idx = get_bucket_index(temp, edges)
                assert 0 <= idx < len(edges), f"Invalid bucket for {city} at {temp}F"


class TestEnsureCityDirs:
    """Tests for ensure_city_dirs()."""

    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CityConfig(
                city_name="Test", city_code="test", kalshi_ticker="KXTEST",
                target_station="USW00000000", target_station_name="Test",
                target_lat=0.0, target_lon=0.0, timezone="UTC",
                igra_station_id="USM00000000", igra_station_name="Test",
                nwp_lat=0.0, nwp_lon=0.0,
                data_dir=os.path.join(tmpdir, "data"),
                models_dir=os.path.join(tmpdir, "models"),
                results_dir=os.path.join(tmpdir, "results"),
            )
            ensure_city_dirs(cfg)
            assert os.path.isdir(cfg.data_dir)
            assert os.path.isdir(cfg.models_dir)
            assert os.path.isdir(cfg.results_dir)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CityConfig(
                city_name="Test", city_code="test", kalshi_ticker="KXTEST",
                target_station="USW00000000", target_station_name="Test",
                target_lat=0.0, target_lon=0.0, timezone="UTC",
                igra_station_id="USM00000000", igra_station_name="Test",
                nwp_lat=0.0, nwp_lon=0.0,
                data_dir=os.path.join(tmpdir, "data"),
                models_dir=os.path.join(tmpdir, "models"),
                results_dir=os.path.join(tmpdir, "results"),
            )
            ensure_city_dirs(cfg)
            ensure_city_dirs(cfg)  # Should not raise
            assert os.path.isdir(cfg.data_dir)


class TestCityDataDirs:
    """Tests for city-specific directory paths."""

    def test_nyc_uses_root_dirs(self):
        """NYC uses root-level data/models/results for backward compat."""
        cfg = get_city_config("nyc")
        assert not cfg.data_dir.endswith("nyc")

    def test_phl_uses_city_subdirs(self):
        cfg = get_city_config("phl")
        assert "philadelphia" in cfg.data_dir
        assert "philadelphia" in cfg.models_dir
        assert "philadelphia" in cfg.results_dir

    def test_chi_uses_city_subdirs(self):
        cfg = get_city_config("chi")
        assert "chicago" in cfg.data_dir
        assert "chicago" in cfg.models_dir
        assert "chicago" in cfg.results_dir
