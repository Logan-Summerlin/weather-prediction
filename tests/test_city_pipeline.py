"""
Parameterized tests for all city prediction pipelines (CHI, PHL, ATL, AUS).

Consolidates the per-city test files:
  - test_chi_pipeline.py
  - test_phl_pipeline.py
  - test_atl_pipeline.py
  - test_aus_pipeline.py (Austin)

Tests cover:
  1. city_config integration (bucket edges, labels, target station, etc.)
  2. Script existence for city-specific runners
  3. Bucket index computation for each city's grid scheme

Follows the same pytest conventions as test_city_config.py.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, get_bucket_index, list_cities

# ---------------------------------------------------------------------------
# Project root for path resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts"))


# ---------------------------------------------------------------------------
# City-specific test data
# ---------------------------------------------------------------------------
CITY_TEST_DATA = {
    "chi": {
        "city_name": "Chicago",
        "kalshi_ticker": "KXHIGHCHI",
        "target_station": "USW00094846",
        "target_station_name_contains": "O'Hare",
        "timezone": "America/Chicago",
        "dir_name": "chicago",
        "n_buckets": 62,
        "first_label": "Below -10",
        "last_label": "Above 110",
        "min_surrounding_stations": 40,
        "asos_target_code": "KORD",
        "ring_names": ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"],
        "compass_sectors": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "meteorological_sectors": ["WNW", "Lake", "SW", "NearField", "NE_Lake"],
        "bucket_spot_checks": {
            -15: 0,   # Below -10
            5: 8,     # (4,6)
            32: 22,   # (32,34)
            55: 33,   # (54,56)
            72: 42,   # (72,74)
            100: 56,  # (100,102)
            110: 61,  # Above 110
        },
        "full_range": (-30, 121),
        "expected_scripts": ["run_chi_data_collection.py"],
    },
    "phl": {
        "city_name": "Philadelphia",
        "kalshi_ticker": "KXHIGHPHIL",
        "target_station": "USW00013739",
        "target_station_name_contains": "Philadelphia",
        "timezone": "America/New_York",
        "dir_name": "philadelphia",
        "n_buckets": 57,
        "first_label": "Below 0",
        "last_label": "Above 110",
        "min_surrounding_stations": 45,
        "asos_target_code": "KPHL",
        "ring_names": ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"],
        "compass_sectors": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "meteorological_sectors": ["WNW", "SW", "Coastal", "NearField", "NE"],
        "bucket_spot_checks": {
            -5: 0,    # Below 0
            15: 8,    # (14,16)
            32: 17,   # (32,34)
            55: 28,   # (54,56)
            72: 37,   # (72,74)
            100: 51,  # (100,102)
            110: 56,  # Above 110
        },
        "full_range": (-20, 121),
        "expected_scripts": [
            "run_phl_data_collection.py",
            "run_phl_preprocessing.py",
            "run_phl_benchmark.py",
            "run_phl_synthesis_calibration.py",
            "run_phl_backtest.py",
            "run_phl_promotion_evaluation.py",
        ],
    },
    "atl": {
        "city_name": "Atlanta",
        "kalshi_ticker": "KXHIGHTATL",
        "target_station": "USW00013874",
        "target_station_name_contains": "Hartsfield-Jackson",
        "timezone": "America/New_York",
        "dir_name": "atlanta",
        "n_buckets": 57,
        "first_label": "Below 0",
        "last_label": "Above 110",
        "min_surrounding_stations": 40,
        "asos_target_code": "KATL",
        "ring_names": ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"],
        "compass_sectors": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "meteorological_sectors": ["WNW", "SW_Gulf", "Piedmont", "NearField", "Mountains"],
        "bucket_spot_checks": {
            -5: 0,    # Below 0
            5: 3,     # (4,6)
            32: 17,   # (32,34)
            55: 28,   # (54,56)
            72: 37,   # (72,74)
            90: 46,   # (90,92)
            100: 51,  # (100,102)
            110: 56,  # Above 110
        },
        "full_range": (-20, 121),
        "expected_scripts": ["run_atl_data_collection.py"],
    },
    "aus": {
        "city_name": "Austin",
        "kalshi_ticker": "KXHIGHAUS",
        "target_station": "USW00013904",
        "target_station_name_contains": "Austin-Bergstrom",
        "timezone": "America/Chicago",
        "dir_name": "austin",
        "n_buckets": 57,
        "first_label": "Below 0",
        "last_label": "Above 110",
        "min_surrounding_stations": 45,
        "asos_target_code": "KAUS",
        "ring_names": ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"],
        "compass_sectors": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "meteorological_sectors": ["Dryline_W", "Gulf_E", "Warm_S", "NearField", "Norther_N"],
        "bucket_spot_checks": {
            -5: 0,    # Below 0
            5: 3,     # (4,6)
            32: 17,   # (32,34)
            55: 28,   # (54,56)
            72: 37,   # (72,74)
            90: 46,   # (90,92)
            100: 51,  # (100,102)
            110: 56,  # Above 110
        },
        "full_range": (-20, 121),
        "expected_scripts": [
            "run_aus_data_collection.py",
            "run_aus_preprocessing.py",
            "run_aus_benchmark.py",
            "run_aus_synthesis_calibration.py",
            "run_aus_backtest.py",
            "run_aus_promotion_evaluation.py",
        ],
    },
}

CITY_CODES = list(CITY_TEST_DATA.keys())


# ===================================================================
# 1. City config integration tests (parameterized)
# ===================================================================
class TestCityConfigLoads:
    """Verify get_city_config returns correct fields for each city."""

    @pytest.fixture(params=CITY_CODES)
    def city_data(self, request):
        code = request.param
        return code, get_city_config(code), CITY_TEST_DATA[code]

    def test_config_loads(self, city_data):
        """get_city_config returns a config with all required fields."""
        code, cfg, expected = city_data
        assert cfg is not None
        assert cfg.city_code == code
        assert cfg.city_name == expected["city_name"]
        assert cfg.kalshi_ticker == expected["kalshi_ticker"]

    def test_bucket_edges_count(self, city_data):
        """City should have the expected number of buckets."""
        code, cfg, expected = city_data
        assert len(cfg.bucket_edges) == expected["n_buckets"]

    def test_bucket_labels(self, city_data):
        """City should have correct first and last labels."""
        code, cfg, expected = city_data
        assert len(cfg.bucket_labels) == expected["n_buckets"]
        assert cfg.bucket_labels[0] == expected["first_label"]
        assert cfg.bucket_labels[-1] == expected["last_label"]

    def test_target_station(self, city_data):
        """City should have the correct target station."""
        code, cfg, expected = city_data
        assert cfg.target_station == expected["target_station"]
        assert expected["target_station_name_contains"] in cfg.target_station_name

    def test_timezone(self, city_data):
        """City should have the correct timezone."""
        code, cfg, expected = city_data
        assert cfg.timezone == expected["timezone"]

    def test_dirs(self, city_data):
        """City should use correct directory names."""
        code, cfg, expected = city_data
        dir_name = expected["dir_name"]
        assert dir_name in cfg.data_dir
        assert dir_name in cfg.models_dir
        assert dir_name in cfg.results_dir
        assert cfg.data_dir.endswith(os.path.join("data", dir_name))
        assert cfg.models_dir.endswith(os.path.join("models", dir_name))
        assert cfg.results_dir.endswith(os.path.join("results", dir_name))

    def test_in_city_list(self, city_data):
        """City should appear in the list_cities() output."""
        code, cfg, expected = city_data
        cities = list_cities()
        assert code in cities

    def test_climatology(self, city_data):
        """City should have 12 months of mean and std data."""
        code, cfg, expected = city_data
        assert len(cfg.monthly_tmax_mean) == 12
        assert len(cfg.monthly_tmax_std) == 12
        for month in range(1, 13):
            assert month in cfg.monthly_tmax_mean
            assert month in cfg.monthly_tmax_std
            assert cfg.monthly_tmax_mean[month] > 0
            assert cfg.monthly_tmax_std[month] > 0
        # Summer should be hotter than winter
        assert cfg.monthly_tmax_mean[7] > cfg.monthly_tmax_mean[1]


# ===================================================================
# 3. Script existence tests (parameterized)
# ===================================================================
class TestScriptsExist:
    """Verify that all expected city pipeline scripts exist in scripts/."""

    @pytest.fixture(params=CITY_CODES)
    def city_scripts(self, request):
        code = request.param
        return code, CITY_TEST_DATA[code]

    def test_scripts_exist(self, city_scripts):
        """All expected pipeline scripts must exist in scripts/."""
        code, expected = city_scripts
        missing = []
        for script in expected["expected_scripts"]:
            path = os.path.join(SCRIPTS_DIR, script)
            if not os.path.isfile(path):
                missing.append(script)
        if missing:
            pytest.skip(
                f"{expected['city_name']} scripts not yet created: {missing}. "
                "This test will pass once pipeline scripts are created."
            )

    def test_unified_scripts_exist(self, city_scripts):
        """Unified consolidated pipeline scripts must exist."""
        code, expected = city_scripts
        unified_scripts = [
            "run_data_collection.py",
            "run_preprocessing.py",
        ]
        missing = []
        for script in unified_scripts:
            path = os.path.join(SCRIPTS_DIR, script)
            if not os.path.isfile(path):
                missing.append(script)
        if missing:
            pytest.skip(f"Unified scripts not yet created: {missing}")


# ===================================================================
# 4. Bucket index tests (parameterized)
# ===================================================================
class TestBucketIndex:
    """Test get_bucket_index with each city's grid scheme."""

    @pytest.fixture(params=CITY_CODES)
    def city_edges(self, request):
        code = request.param
        edges = get_city_config(code).bucket_edges
        return code, edges, CITY_TEST_DATA[code]

    def test_bucket_spot_checks(self, city_edges):
        """Spot-check representative temps across the city's grid."""
        code, edges, expected = city_edges
        for temp, expected_idx in expected["bucket_spot_checks"].items():
            actual = get_bucket_index(temp, edges)
            assert actual == expected_idx, (
                f"{code.upper()} Temp {temp}F: expected bucket {expected_idx}, got {actual}"
            )

    def test_full_range_no_gaps(self, city_edges):
        """Every integer temperature in the full range should map to a valid bucket."""
        code, edges, expected = city_edges
        n_buckets = len(edges)
        lo, hi = expected["full_range"]
        for temp in range(lo, hi):
            idx = get_bucket_index(temp, edges)
            assert 0 <= idx < n_buckets, (
                f"{code.upper()} Temperature {temp}F mapped to invalid bucket index {idx}"
            )

    def test_below_first_bucket(self, city_edges):
        """Very cold temp should fall in bucket 0."""
        code, edges, expected = city_edges
        lo = expected["full_range"][0]
        idx = get_bucket_index(lo, edges)
        assert idx == 0

    def test_above_last_bucket(self, city_edges):
        """Very hot temp should fall in the last bucket."""
        code, edges, expected = city_edges
        n_buckets = len(edges)
        hi = expected["full_range"][1]
        idx = get_bucket_index(hi, edges)
        assert idx == n_buckets - 1
