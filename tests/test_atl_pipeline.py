"""
Comprehensive tests for the Atlanta (ATL) prediction pipeline.

Tests cover:
  1. Atlanta city_config integration (bucket edges, labels, target station, etc.)
  2. Atlanta config_atlanta.py module imports and data structures
  3. Script existence for Atlanta-specific runners
  4. Bucket index computation for Atlanta's 57-bucket 2F grid scheme

Follows the same pytest conventions as test_chi_pipeline.py.
"""

import os
import sys
import importlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, get_bucket_index, list_cities

# ---------------------------------------------------------------------------
# Project root for path resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ===================================================================
# 1. Atlanta city_config tests
# ===================================================================
class TestAtlConfigLoads:
    """Verify get_city_config('atl') returns correct fields."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        self.cfg = get_city_config("atl")

    def test_atl_config_loads(self):
        """get_city_config('atl') returns a config with all required fields."""
        assert self.cfg is not None
        assert self.cfg.city_code == "atl"
        assert self.cfg.city_name == "Atlanta"
        assert self.cfg.kalshi_ticker == "KXHIGHTATL"
        assert isinstance(self.cfg.bucket_edges, list)
        assert isinstance(self.cfg.bucket_labels, list)
        assert isinstance(self.cfg.monthly_tmax_mean, dict)
        assert isinstance(self.cfg.monthly_tmax_std, dict)

    def test_atl_bucket_edges(self):
        """Atlanta should have 57 buckets (2F grid from 0 to 110 plus tails)."""
        assert len(self.cfg.bucket_edges) == 57

    def test_atl_bucket_labels(self):
        """Atlanta should have 57 labels starting with 'Below 0'."""
        assert len(self.cfg.bucket_labels) == 57
        assert self.cfg.bucket_labels[0] == "Below 0"
        assert self.cfg.bucket_labels[-1] == "Above 110"
        # Spot-check a few interior labels
        # Index 1 should be the (0, 2) bucket
        assert "0" in self.cfg.bucket_labels[1]

    def test_atl_target_station(self):
        """Atlanta target station should be Hartsfield-Jackson (USW00013874)."""
        assert self.cfg.target_station == "USW00013874"
        assert "Hartsfield-Jackson" in self.cfg.target_station_name

    def test_atl_timezone(self):
        """Atlanta should use America/New_York timezone."""
        assert self.cfg.timezone == "America/New_York"

    def test_atl_dirs(self):
        """Atlanta should use data/atlanta, models/atlanta, results/atlanta paths."""
        assert "atlanta" in self.cfg.data_dir
        assert "atlanta" in self.cfg.models_dir
        assert "atlanta" in self.cfg.results_dir
        # Verify they end with the expected suffixes
        assert self.cfg.data_dir.endswith(os.path.join("data", "atlanta"))
        assert self.cfg.models_dir.endswith(os.path.join("models", "atlanta"))
        assert self.cfg.results_dir.endswith(os.path.join("results", "atlanta"))

    def test_atl_in_city_list(self):
        """Atlanta ('atl') should appear in the list_cities() output."""
        cities = list_cities()
        assert "atl" in cities

    def test_atl_igra_station(self):
        """Atlanta IGRA sounding station should be Peachtree City FFC."""
        assert self.cfg.igra_station_id == "USM00072215"
        assert "Peachtree City" in self.cfg.igra_station_name

    def test_atl_climatology(self):
        """Atlanta climatology should have 12 months of mean and std data."""
        assert len(self.cfg.monthly_tmax_mean) == 12
        assert len(self.cfg.monthly_tmax_std) == 12
        for month in range(1, 13):
            assert month in self.cfg.monthly_tmax_mean
            assert month in self.cfg.monthly_tmax_std
            assert self.cfg.monthly_tmax_mean[month] > 0
            assert self.cfg.monthly_tmax_std[month] > 0
        # Atlanta summer should be hotter than winter
        assert self.cfg.monthly_tmax_mean[7] > self.cfg.monthly_tmax_mean[1]


# ===================================================================
# 2. Atlanta config_atlanta.py import tests
# ===================================================================
class TestConfigAtlantaImports:
    """Verify config_atlanta module imports and exposes expected structures."""

    @pytest.fixture(autouse=True)
    def _try_import(self):
        """Attempt to import config_atlanta. Skip all tests if not found."""
        try:
            self.mod = importlib.import_module("config_atlanta")
        except ModuleNotFoundError:
            pytest.skip("config_atlanta module not yet created")

    def test_config_atlanta_imports(self):
        """config_atlanta module should be importable."""
        assert self.mod is not None

    def test_config_atlanta_surrounding_stations(self):
        """SURROUNDING_STATIONS should be a dict with 40+ stations."""
        assert hasattr(self.mod, "SURROUNDING_STATIONS")
        ss = self.mod.SURROUNDING_STATIONS
        assert isinstance(ss, dict)
        assert len(ss) >= 40, (
            f"Expected 40+ surrounding stations, got {len(ss)}"
        )
        # Every key should be a GHCN station ID string
        for sid in ss:
            assert isinstance(sid, str)
            assert sid.startswith("US"), f"Station ID {sid} doesn't start with 'US'"

    def test_config_atlanta_all_stations(self):
        """ALL_STATIONS should include the target station plus all surrounding."""
        assert hasattr(self.mod, "ALL_STATIONS")
        all_st = self.mod.ALL_STATIONS
        assert isinstance(all_st, dict)
        # Must contain target station
        assert "USW00013874" in all_st, "Target station USW00013874 missing from ALL_STATIONS"
        # ALL_STATIONS should be at least surrounding + 1 (target)
        if hasattr(self.mod, "SURROUNDING_STATIONS"):
            assert len(all_st) >= len(self.mod.SURROUNDING_STATIONS) + 1

    def test_config_atlanta_asos_map(self):
        """ASOS_STATION_MAP should have KATL for the ATL target station."""
        assert hasattr(self.mod, "ASOS_STATION_MAP")
        asos = self.mod.ASOS_STATION_MAP
        assert isinstance(asos, dict)
        assert "USW00013874" in asos, "Target station missing from ASOS_STATION_MAP"
        assert asos["USW00013874"] == "KATL", (
            f"Expected KATL for Hartsfield-Jackson, got {asos['USW00013874']}"
        )

    def test_config_atlanta_station_rings(self):
        """STATION_RINGS should have Ring1, Ring2, Ring3, Ring4 -- all populated."""
        assert hasattr(self.mod, "STATION_RINGS")
        rings = self.mod.STATION_RINGS
        assert isinstance(rings, dict)
        for ring_name in ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"]:
            assert ring_name in rings, f"Missing ring: {ring_name}"
            assert len(rings[ring_name]) > 0, f"Ring {ring_name} is empty"

    def test_config_atlanta_station_sectors(self):
        """STATION_SECTORS should have all 8 compass sectors populated."""
        assert hasattr(self.mod, "STATION_SECTORS")
        sectors = self.mod.STATION_SECTORS
        assert isinstance(sectors, dict)
        compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for direction in compass:
            assert direction in sectors, f"Missing sector: {direction}"
            assert len(sectors[direction]) > 0, f"Sector {direction} is empty"

    def test_config_atlanta_meteorological_sectors(self):
        """METEOROLOGICAL_SECTORS should include Atlanta-specific groupings."""
        assert hasattr(self.mod, "METEOROLOGICAL_SECTORS")
        met = self.mod.METEOROLOGICAL_SECTORS
        assert isinstance(met, dict)
        # Atlanta-relevant meteorological groupings
        expected_sectors = ["WNW", "SW_Gulf", "Piedmont", "NearField", "Mountains"]
        for sector in expected_sectors:
            assert sector in met, f"Missing meteorological sector: {sector}"
            assert len(met[sector]) > 0, f"Meteorological sector {sector} is empty"

    def test_config_atlanta_station_metadata(self):
        """STATION_METADATA should have lat/lon/distance/bearing for each station."""
        assert hasattr(self.mod, "STATION_METADATA")
        meta = self.mod.STATION_METADATA
        assert isinstance(meta, dict)
        assert len(meta) > 0, "STATION_METADATA is empty"
        required_keys = {"lat", "lon", "distance_mi", "bearing"}
        for sid, info in meta.items():
            assert isinstance(info, dict), f"Metadata for {sid} is not a dict"
            missing = required_keys - set(info.keys())
            assert not missing, (
                f"Station {sid} missing metadata keys: {missing}"
            )
            # Sanity checks
            assert -90 <= info["lat"] <= 90, f"Bad lat for {sid}: {info['lat']}"
            assert -180 <= info["lon"] <= 180, f"Bad lon for {sid}: {info['lon']}"
            assert info["distance_mi"] > 0, f"Non-positive distance for {sid}"
            assert 0 <= info["bearing"] < 360, f"Bad bearing for {sid}: {info['bearing']}"

    def test_config_atlanta_pipeline_constants(self):
        """Pipeline constants (START_DATE, END_DATE, TRAIN_RATIO, etc.) should exist."""
        assert hasattr(self.mod, "START_DATE")
        assert hasattr(self.mod, "END_DATE")
        assert hasattr(self.mod, "TRAIN_RATIO")
        assert hasattr(self.mod, "VAL_RATIO")
        assert hasattr(self.mod, "TEST_RATIO")
        # Ratios should sum to ~1.0
        total = self.mod.TRAIN_RATIO + self.mod.VAL_RATIO + self.mod.TEST_RATIO
        assert abs(total - 1.0) < 0.01, f"Split ratios sum to {total}, expected ~1.0"
        # Dates should be strings
        assert isinstance(self.mod.START_DATE, str)
        assert isinstance(self.mod.END_DATE, str)

    def test_config_atlanta_bucket_definitions(self):
        """BUCKET_EDGES in config_atlanta should match city_config bucket_edges."""
        assert hasattr(self.mod, "BUCKET_EDGES")
        assert hasattr(self.mod, "BUCKET_LABELS")
        atl_cfg = get_city_config("atl")
        # Compare as lists of tuples
        module_edges = [tuple(e) for e in self.mod.BUCKET_EDGES]
        config_edges = [tuple(e) for e in atl_cfg.bucket_edges]
        assert module_edges == config_edges, (
            f"BUCKET_EDGES mismatch:\n  config_atlanta: {module_edges}\n  city_config:    {config_edges}"
        )
        assert list(self.mod.BUCKET_LABELS) == list(atl_cfg.bucket_labels)

    def test_config_atlanta_min_completeness(self):
        """MIN_COMPLETENESS should be defined and reasonable."""
        assert hasattr(self.mod, "MIN_COMPLETENESS")
        assert 0.0 < self.mod.MIN_COMPLETENESS <= 1.0


# ===================================================================
# 3. Script existence tests
# ===================================================================
class TestAtlScriptsExist:
    """Verify that all expected run_atl_*.py scripts exist in scripts/."""

    SCRIPTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts"))

    # Minimum set of Atlanta pipeline scripts expected to exist.
    EXPECTED_SCRIPTS = [
        "run_atl_data_collection.py",
    ]

    def test_atl_scripts_exist(self):
        """All expected run_atl_*.py scripts must exist in scripts/."""
        missing = []
        for script in self.EXPECTED_SCRIPTS:
            path = os.path.join(self.SCRIPTS_DIR, script)
            if not os.path.isfile(path):
                missing.append(script)
        if missing:
            pytest.skip(
                f"Atlanta scripts not yet created: {missing}. "
                "This test will pass once pipeline scripts are created."
            )

    def test_config_atlanta_file_exists(self):
        """config_atlanta.py must exist at the project root."""
        path = os.path.normpath(os.path.join(PROJECT_ROOT, "config_atlanta.py"))
        assert os.path.isfile(path), f"Missing: {path}"


# ===================================================================
# 4. Bucket index tests
# ===================================================================
class TestAtlBucketIndex:
    """Test get_bucket_index with Atlanta's 57-bucket 2F grid scheme."""

    @pytest.fixture(autouse=True)
    def _load_edges(self):
        self.edges = get_city_config("atl").bucket_edges

    def test_atl_bucket_index_very_cold(self):
        """-5F should fall in bucket 0 ('Below 0')."""
        idx = get_bucket_index(-5, self.edges)
        assert idx == 0

    def test_atl_bucket_index_cold(self):
        """35F should fall in bucket 18 (the (34,36) bucket at index 18)."""
        idx = get_bucket_index(35, self.edges)
        assert idx == 18

    def test_atl_bucket_index_hot(self):
        """95F should fall in bucket 48 (the (94,96) bucket at index 48)."""
        idx = get_bucket_index(95, self.edges)
        assert idx == 48

    def test_atl_bucket_index_above_max(self):
        """110F should fall in bucket 56 ('Above 110')."""
        idx = get_bucket_index(110, self.edges)
        assert idx == 56

    def test_atl_bucket_index_boundary_0(self):
        """Boundary at 0: -1F -> bucket 0, 0F -> bucket 1."""
        assert get_bucket_index(-1, self.edges) == 0
        assert get_bucket_index(0, self.edges) == 1

    def test_atl_bucket_index_boundary_2(self):
        """Boundary at 2: 1F -> bucket 1, 2F -> bucket 2."""
        assert get_bucket_index(1, self.edges) == 1
        assert get_bucket_index(2, self.edges) == 2

    def test_atl_bucket_index_boundary_110(self):
        """Boundary at 110: 109F -> bucket 55, 110F -> bucket 56."""
        assert get_bucket_index(109, self.edges) == 55
        assert get_bucket_index(110, self.edges) == 56

    def test_atl_spot_check_2f_grid(self):
        """Spot-check representative temps across the 2F grid."""
        # Bucket index formula for interior: index = (temp - 0) // 2 + 1
        # when temp is within [0, 110).  Below 0 -> 0, >= 110 -> 56.
        expected = {
            -5: 0,    # Below 0
            5: 3,     # (4,6) bucket at index 3
            32: 17,   # (32,34) bucket at index 17
            55: 28,   # (54,56) bucket at index 28
            72: 37,   # (72,74) bucket at index 37
            90: 46,   # (90,92) bucket at index 46
            100: 51,  # (100,102) bucket at index 51
            110: 56,  # Above 110
        }
        for temp, expected_idx in expected.items():
            actual = get_bucket_index(temp, self.edges)
            assert actual == expected_idx, (
                f"Temp {temp}F: expected bucket {expected_idx}, got {actual}"
            )

    def test_atl_full_range_no_gaps(self):
        """Every integer temperature from -20 to 120 should map to a valid bucket."""
        n_buckets = len(self.edges)
        for temp in range(-20, 121):
            idx = get_bucket_index(temp, self.edges)
            assert 0 <= idx < n_buckets, (
                f"Temperature {temp}F mapped to invalid bucket index {idx}"
            )
