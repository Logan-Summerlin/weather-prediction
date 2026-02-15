"""
Comprehensive tests for the Chicago (CHI) prediction pipeline.

Tests cover:
  1. Chicago city_config integration (bucket edges, labels, target station, etc.)
  2. Chicago config_chicago.py module imports and data structures
  3. Script existence for Chicago-specific runners
  4. Bucket index computation for Chicago's 11-bucket scheme

Follows the same pytest conventions as test_city_config.py.
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
# 1. Chicago city_config tests
# ===================================================================
class TestChiConfigLoads:
    """Verify get_city_config('chi') returns correct fields."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        self.cfg = get_city_config("chi")

    def test_chi_config_loads(self):
        """get_city_config('chi') returns a config with all required fields."""
        assert self.cfg is not None
        assert self.cfg.city_code == "chi"
        assert self.cfg.city_name == "Chicago"
        assert self.cfg.kalshi_ticker == "KXHIGHCHI"
        assert isinstance(self.cfg.bucket_edges, list)
        assert isinstance(self.cfg.bucket_labels, list)
        assert isinstance(self.cfg.monthly_tmax_mean, dict)
        assert isinstance(self.cfg.monthly_tmax_std, dict)

    def test_chi_bucket_edges(self):
        """Chicago should have 11 buckets (not 10 like NYC/PHL) due to colder winters."""
        assert len(self.cfg.bucket_edges) == 11

    def test_chi_bucket_labels(self):
        """Chicago should have 11 labels starting with 'Below 10'."""
        assert len(self.cfg.bucket_labels) == 11
        assert self.cfg.bucket_labels[0] == "Below 10"
        assert self.cfg.bucket_labels[-1] == "Above 100"
        # Verify the full label sequence
        expected_labels = [
            "Below 10", "10-19", "20-29", "30-39", "40-49",
            "50-59", "60-69", "70-79", "80-89", "90-99", "Above 100",
        ]
        assert self.cfg.bucket_labels == expected_labels

    def test_chi_target_station(self):
        """Chicago target station should be O'Hare (USW00094846)."""
        assert self.cfg.target_station == "USW00094846"
        assert "O'Hare" in self.cfg.target_station_name

    def test_chi_timezone(self):
        """Chicago should use America/Chicago timezone."""
        assert self.cfg.timezone == "America/Chicago"

    def test_chi_dirs(self):
        """Chicago should use data/chicago, models/chicago, results/chicago paths."""
        assert "chicago" in self.cfg.data_dir
        assert "chicago" in self.cfg.models_dir
        assert "chicago" in self.cfg.results_dir
        # Verify they end with the expected suffixes
        assert self.cfg.data_dir.endswith(os.path.join("data", "chicago"))
        assert self.cfg.models_dir.endswith(os.path.join("models", "chicago"))
        assert self.cfg.results_dir.endswith(os.path.join("results", "chicago"))


# ===================================================================
# 2. Chicago config_chicago.py import tests
# ===================================================================
class TestConfigChicagoImports:
    """Verify config_chicago module imports and exposes expected structures."""

    @pytest.fixture(autouse=True)
    def _try_import(self):
        """Attempt to import config_chicago. Skip all tests if not found."""
        try:
            self.mod = importlib.import_module("config_chicago")
        except ModuleNotFoundError:
            pytest.skip("config_chicago module not yet created")

    def test_config_chicago_imports(self):
        """config_chicago module should be importable."""
        assert self.mod is not None

    def test_config_chicago_surrounding_stations(self):
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

    def test_config_chicago_all_stations(self):
        """ALL_STATIONS should include the target station plus all surrounding."""
        assert hasattr(self.mod, "ALL_STATIONS")
        all_st = self.mod.ALL_STATIONS
        assert isinstance(all_st, dict)
        # Must contain target station
        assert "USW00094846" in all_st, "Target station USW00094846 missing from ALL_STATIONS"
        # ALL_STATIONS should be at least surrounding + 1 (target)
        if hasattr(self.mod, "SURROUNDING_STATIONS"):
            assert len(all_st) >= len(self.mod.SURROUNDING_STATIONS) + 1

    def test_config_chicago_asos_map(self):
        """ASOS_STATION_MAP should have KORD for the O'Hare target station."""
        assert hasattr(self.mod, "ASOS_STATION_MAP")
        asos = self.mod.ASOS_STATION_MAP
        assert isinstance(asos, dict)
        assert "USW00094846" in asos, "Target station missing from ASOS_STATION_MAP"
        assert asos["USW00094846"] == "KORD", (
            f"Expected KORD for O'Hare, got {asos['USW00094846']}"
        )

    def test_config_chicago_station_rings(self):
        """STATION_RINGS should have Ring1, Ring2, Ring3, Ring4 -- all populated."""
        assert hasattr(self.mod, "STATION_RINGS")
        rings = self.mod.STATION_RINGS
        assert isinstance(rings, dict)
        for ring_name in ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"]:
            assert ring_name in rings, f"Missing ring: {ring_name}"
            assert len(rings[ring_name]) > 0, f"Ring {ring_name} is empty"

    def test_config_chicago_station_sectors(self):
        """STATION_SECTORS should have all 8 compass sectors populated."""
        assert hasattr(self.mod, "STATION_SECTORS")
        sectors = self.mod.STATION_SECTORS
        assert isinstance(sectors, dict)
        compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for direction in compass:
            assert direction in sectors, f"Missing sector: {direction}"
            assert len(sectors[direction]) > 0, f"Sector {direction} is empty"

    def test_config_chicago_meteorological_sectors(self):
        """METEOROLOGICAL_SECTORS should include Chicago-specific groupings."""
        assert hasattr(self.mod, "METEOROLOGICAL_SECTORS")
        met = self.mod.METEOROLOGICAL_SECTORS
        assert isinstance(met, dict)
        # Chicago-relevant meteorological groupings
        expected_sectors = ["WNW", "Lake", "SW", "NearField", "NE_Lake"]
        for sector in expected_sectors:
            assert sector in met, f"Missing meteorological sector: {sector}"
            assert len(met[sector]) > 0, f"Meteorological sector {sector} is empty"

    def test_config_chicago_station_metadata(self):
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

    def test_config_chicago_pipeline_constants(self):
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

    def test_config_chicago_bucket_definitions(self):
        """BUCKET_EDGES in config_chicago should match city_config bucket_edges."""
        assert hasattr(self.mod, "BUCKET_EDGES")
        assert hasattr(self.mod, "BUCKET_LABELS")
        chi_cfg = get_city_config("chi")
        # Compare as lists of tuples
        module_edges = [tuple(e) for e in self.mod.BUCKET_EDGES]
        config_edges = [tuple(e) for e in chi_cfg.bucket_edges]
        assert module_edges == config_edges, (
            f"BUCKET_EDGES mismatch:\n  config_chicago: {module_edges}\n  city_config:    {config_edges}"
        )
        assert list(self.mod.BUCKET_LABELS) == list(chi_cfg.bucket_labels)


# ===================================================================
# 3. Script existence tests
# ===================================================================
class TestChiScriptsExist:
    """Verify that all expected run_chi_*.py scripts exist in scripts/."""

    SCRIPTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts"))

    # Minimum set of Chicago pipeline scripts expected to exist.
    EXPECTED_SCRIPTS = [
        "run_chi_data_collection.py",
    ]

    def test_chi_scripts_exist(self):
        """All expected run_chi_*.py scripts must exist in scripts/."""
        missing = []
        for script in self.EXPECTED_SCRIPTS:
            path = os.path.join(self.SCRIPTS_DIR, script)
            if not os.path.isfile(path):
                missing.append(script)
        assert not missing, (
            f"Missing Chicago scripts in {self.SCRIPTS_DIR}: {missing}"
        )

    def test_chi_data_collection_script_exists(self):
        """run_chi_data_collection.py must exist as the primary ingestion entry point."""
        path = os.path.join(self.SCRIPTS_DIR, "run_chi_data_collection.py")
        assert os.path.isfile(path), f"Missing: {path}"


# ===================================================================
# 4. Bucket index tests
# ===================================================================
class TestChiBucketIndex:
    """Test get_bucket_index with Chicago's 11-bucket scheme."""

    @pytest.fixture(autouse=True)
    def _load_edges(self):
        self.edges = get_city_config("chi").bucket_edges

    def test_chi_bucket_index_cold(self):
        """5F should fall in bucket 0 ('Below 10')."""
        idx = get_bucket_index(5, self.edges)
        assert idx == 0

    def test_chi_bucket_index_very_cold(self):
        """-15F should also fall in bucket 0 ('Below 10')."""
        idx = get_bucket_index(-15, self.edges)
        assert idx == 0

    def test_chi_bucket_index_normal(self):
        """55F should fall in bucket 5 ('50-59')."""
        idx = get_bucket_index(55, self.edges)
        assert idx == 5

    def test_chi_bucket_index_hot(self):
        """105F should fall in bucket 10 ('Above 100')."""
        idx = get_bucket_index(105, self.edges)
        assert idx == 10

    def test_chi_bucket_index_boundary_10(self):
        """Boundary at 10: 9F -> bucket 0, 10F -> bucket 1."""
        assert get_bucket_index(9, self.edges) == 0
        assert get_bucket_index(10, self.edges) == 1

    def test_chi_bucket_index_boundary_20(self):
        """Boundary at 20: 19F -> bucket 1, 20F -> bucket 2."""
        assert get_bucket_index(19, self.edges) == 1
        assert get_bucket_index(20, self.edges) == 2

    def test_chi_bucket_index_boundary_100(self):
        """Boundary at 100: 99F -> bucket 9, 100F -> bucket 10."""
        assert get_bucket_index(99, self.edges) == 9
        assert get_bucket_index(100, self.edges) == 10

    def test_chi_all_decades_covered(self):
        """Spot-check that each decade maps to the correct bucket."""
        expected = {
            5: 0,    # Below 10
            15: 1,   # 10-19
            25: 2,   # 20-29
            35: 3,   # 30-39
            45: 4,   # 40-49
            55: 5,   # 50-59
            65: 6,   # 60-69
            75: 7,   # 70-79
            85: 8,   # 80-89
            95: 9,   # 90-99
            105: 10, # Above 100
        }
        for temp, expected_idx in expected.items():
            actual = get_bucket_index(temp, self.edges)
            assert actual == expected_idx, (
                f"Temp {temp}F: expected bucket {expected_idx}, got {actual}"
            )

    def test_chi_full_range_no_gaps(self):
        """Every integer temperature from -30 to 120 should map to a valid bucket."""
        n_buckets = len(self.edges)
        for temp in range(-30, 121):
            idx = get_bucket_index(temp, self.edges)
            assert 0 <= idx < n_buckets, (
                f"Temperature {temp}F mapped to invalid bucket index {idx}"
            )
