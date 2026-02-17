"""
Comprehensive tests for the Philadelphia (PHL) prediction pipeline.

Tests cover:
  1. Philadelphia city_config integration (bucket edges, labels, target station, etc.)
  2. Philadelphia config_philadelphia.py module imports and data structures
  3. Script existence for Philadelphia-specific runners
  4. Bucket index computation for Philadelphia's 57-bucket 2degF grid scheme

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
# 1. Philadelphia city_config tests
# ===================================================================
class TestPhlConfigLoads:
    """Verify get_city_config('phl') returns correct fields."""

    @pytest.fixture(autouse=True)
    def _load_config(self):
        self.cfg = get_city_config("phl")

    def test_phl_config_loads(self):
        """get_city_config('phl') returns a config with all required fields."""
        assert self.cfg is not None
        assert self.cfg.city_code == "phl"
        assert self.cfg.city_name == "Philadelphia"
        assert self.cfg.kalshi_ticker == "KXHIGHPHL"
        assert isinstance(self.cfg.bucket_edges, list)
        assert isinstance(self.cfg.bucket_labels, list)
        assert isinstance(self.cfg.monthly_tmax_mean, dict)
        assert isinstance(self.cfg.monthly_tmax_std, dict)

    def test_phl_bucket_edges(self):
        """Philadelphia should have 57 buckets (2degF grid from 0 to 110 plus tails)."""
        assert len(self.cfg.bucket_edges) == 57

    def test_phl_bucket_labels(self):
        """Philadelphia should have 57 labels starting with 'Below 0'."""
        assert len(self.cfg.bucket_labels) == 57
        assert self.cfg.bucket_labels[0] == "Below 0"
        assert self.cfg.bucket_labels[-1] == "Above 110"
        # Spot-check a few interior labels
        # Index 1 should be the (0, 2) bucket
        assert self.cfg.bucket_labels[1] == "0-2" or "0" in self.cfg.bucket_labels[1]

    def test_phl_target_station(self):
        """Philadelphia target station should be PHL International (USW00013739)."""
        assert self.cfg.target_station == "USW00013739"
        assert "Philadelphia" in self.cfg.target_station_name

    def test_phl_timezone(self):
        """Philadelphia should use America/New_York timezone."""
        assert self.cfg.timezone == "America/New_York"

    def test_phl_dirs(self):
        """Philadelphia should use data/philadelphia, models/philadelphia, results/philadelphia paths."""
        assert "philadelphia" in self.cfg.data_dir
        assert "philadelphia" in self.cfg.models_dir
        assert "philadelphia" in self.cfg.results_dir
        # Verify they end with the expected suffixes
        assert self.cfg.data_dir.endswith(os.path.join("data", "philadelphia"))
        assert self.cfg.models_dir.endswith(os.path.join("models", "philadelphia"))
        assert self.cfg.results_dir.endswith(os.path.join("results", "philadelphia"))


# ===================================================================
# 2. Philadelphia config_philadelphia.py import tests
# ===================================================================
class TestConfigPhiladelphiaImports:
    """Verify config_philadelphia module imports and exposes expected structures."""

    @pytest.fixture(autouse=True)
    def _try_import(self):
        """Attempt to import config_philadelphia. Skip all tests if not found."""
        try:
            self.mod = importlib.import_module("config_philadelphia")
        except ModuleNotFoundError:
            pytest.skip("config_philadelphia module not yet created")

    def test_config_philadelphia_imports(self):
        """config_philadelphia module should be importable."""
        assert self.mod is not None

    def test_config_philadelphia_surrounding_stations(self):
        """SURROUNDING_STATIONS should be a dict with 45+ stations."""
        assert hasattr(self.mod, "SURROUNDING_STATIONS")
        ss = self.mod.SURROUNDING_STATIONS
        assert isinstance(ss, dict)
        assert len(ss) >= 45, (
            f"Expected 45+ surrounding stations, got {len(ss)}"
        )
        # Every key should be a GHCN station ID string
        for sid in ss:
            assert isinstance(sid, str)
            assert sid.startswith("US"), f"Station ID {sid} doesn't start with 'US'"

    def test_config_philadelphia_all_stations(self):
        """ALL_STATIONS should include the target station plus all surrounding."""
        assert hasattr(self.mod, "ALL_STATIONS")
        all_st = self.mod.ALL_STATIONS
        assert isinstance(all_st, dict)
        # Must contain target station
        assert "USW00013739" in all_st, "Target station USW00013739 missing from ALL_STATIONS"
        # ALL_STATIONS should be at least surrounding + 1 (target)
        if hasattr(self.mod, "SURROUNDING_STATIONS"):
            assert len(all_st) >= len(self.mod.SURROUNDING_STATIONS) + 1

    def test_config_philadelphia_asos_map(self):
        """ASOS_STATION_MAP should have KPHL for the PHL International target station."""
        assert hasattr(self.mod, "ASOS_STATION_MAP")
        asos = self.mod.ASOS_STATION_MAP
        assert isinstance(asos, dict)
        assert "USW00013739" in asos, "Target station missing from ASOS_STATION_MAP"
        assert asos["USW00013739"] == "KPHL", (
            f"Expected KPHL for PHL International, got {asos['USW00013739']}"
        )

    def test_config_philadelphia_station_rings(self):
        """STATION_RINGS should have Ring1, Ring2, Ring3, Ring4 -- all populated."""
        assert hasattr(self.mod, "STATION_RINGS")
        rings = self.mod.STATION_RINGS
        assert isinstance(rings, dict)
        for ring_name in ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"]:
            assert ring_name in rings, f"Missing ring: {ring_name}"
            assert len(rings[ring_name]) > 0, f"Ring {ring_name} is empty"

    def test_config_philadelphia_station_sectors(self):
        """STATION_SECTORS should have all 8 compass sectors populated."""
        assert hasattr(self.mod, "STATION_SECTORS")
        sectors = self.mod.STATION_SECTORS
        assert isinstance(sectors, dict)
        compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for direction in compass:
            assert direction in sectors, f"Missing sector: {direction}"
            assert len(sectors[direction]) > 0, f"Sector {direction} is empty"

    def test_config_philadelphia_meteorological_sectors(self):
        """METEOROLOGICAL_SECTORS should include Philadelphia-specific groupings."""
        assert hasattr(self.mod, "METEOROLOGICAL_SECTORS")
        met = self.mod.METEOROLOGICAL_SECTORS
        assert isinstance(met, dict)
        # Philadelphia-relevant meteorological groupings
        expected_sectors = ["WNW", "SW", "Coastal", "NearField", "NE"]
        for sector in expected_sectors:
            assert sector in met, f"Missing meteorological sector: {sector}"
            assert len(met[sector]) > 0, f"Meteorological sector {sector} is empty"

    def test_config_philadelphia_station_metadata(self):
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

    def test_config_philadelphia_pipeline_constants(self):
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

    def test_config_philadelphia_bucket_definitions(self):
        """BUCKET_EDGES in config_philadelphia should match city_config bucket_edges."""
        assert hasattr(self.mod, "BUCKET_EDGES")
        assert hasattr(self.mod, "BUCKET_LABELS")
        phl_cfg = get_city_config("phl")
        # Compare as lists of tuples
        module_edges = [tuple(e) for e in self.mod.BUCKET_EDGES]
        config_edges = [tuple(e) for e in phl_cfg.bucket_edges]
        assert module_edges == config_edges, (
            f"BUCKET_EDGES mismatch:\n  config_philadelphia: {module_edges}\n  city_config:         {config_edges}"
        )
        assert list(self.mod.BUCKET_LABELS) == list(phl_cfg.bucket_labels)


# ===================================================================
# 3. Script existence tests
# ===================================================================
class TestPhlScriptsExist:
    """Verify that all expected run_phl_*.py scripts exist in scripts/."""

    SCRIPTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts"))

    # Minimum set of Philadelphia pipeline scripts expected to exist.
    EXPECTED_SCRIPTS = [
        "run_phl_data_collection.py",
        "run_phl_preprocessing.py",
        "run_phl_benchmark.py",
        "run_phl_synthesis_calibration.py",
        "run_phl_backtest.py",
        "run_phl_promotion_evaluation.py",
    ]

    def test_phl_scripts_exist(self):
        """All expected run_phl_*.py scripts must exist in scripts/."""
        missing = []
        for script in self.EXPECTED_SCRIPTS:
            path = os.path.join(self.SCRIPTS_DIR, script)
            if not os.path.isfile(path):
                missing.append(script)
        assert not missing, (
            f"Missing Philadelphia scripts in {self.SCRIPTS_DIR}: {missing}"
        )

    def test_phl_data_collection_script_exists(self):
        """run_phl_data_collection.py must exist as the primary ingestion entry point."""
        path = os.path.join(self.SCRIPTS_DIR, "run_phl_data_collection.py")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_phl_preprocessing_script_exists(self):
        """run_phl_preprocessing.py must exist for feature engineering."""
        path = os.path.join(self.SCRIPTS_DIR, "run_phl_preprocessing.py")
        assert os.path.isfile(path), f"Missing: {path}"


# ===================================================================
# 4. Bucket index tests
# ===================================================================
class TestPhlBucketIndex:
    """Test get_bucket_index with Philadelphia's 57-bucket 2degF grid scheme.

    PHL uses a 0degF floor and 110degF ceiling with 2degF-wide interior buckets:
      Bucket 0:  Below 0    (i.e. temp < 0)
      Bucket 1:  0-2        (i.e. 0 <= temp < 2)
      Bucket 2:  2-4
      ...
      Bucket 55: 108-110
      Bucket 56: Above 110  (i.e. temp >= 110)

    Interior index formula: index = (temp - 0) // 2 + 1  for 0 <= temp < 110.
    """

    @pytest.fixture(autouse=True)
    def _load_edges(self):
        self.edges = get_city_config("phl").bucket_edges

    def test_phl_bucket_index_very_cold(self):
        """-5F should fall in bucket 0 ('Below 0')."""
        idx = get_bucket_index(-5, self.edges)
        assert idx == 0

    def test_phl_bucket_index_cold(self):
        """15F should fall in bucket 8 (the (14,16) bucket at index 8)."""
        idx = get_bucket_index(15, self.edges)
        assert idx == 8

    def test_phl_bucket_index_moderate(self):
        """55F should fall in bucket 28 (the (54,56) bucket at index 28)."""
        idx = get_bucket_index(55, self.edges)
        assert idx == 28

    def test_phl_bucket_index_hot(self):
        """100F should fall in bucket 51 (the (100,102) bucket at index 51)."""
        idx = get_bucket_index(100, self.edges)
        assert idx == 51

    def test_phl_bucket_index_above_max(self):
        """115F should fall in bucket 56 ('Above 110')."""
        idx = get_bucket_index(115, self.edges)
        assert idx == 56

    def test_phl_bucket_index_boundary_0(self):
        """Boundary at 0: -1F -> bucket 0, 0F -> bucket 1."""
        assert get_bucket_index(-1, self.edges) == 0
        assert get_bucket_index(0, self.edges) == 1

    def test_phl_bucket_index_boundary_2(self):
        """Boundary at 2: 1F -> bucket 1, 2F -> bucket 2."""
        assert get_bucket_index(1, self.edges) == 1
        assert get_bucket_index(2, self.edges) == 2

    def test_phl_bucket_index_boundary_110(self):
        """Boundary at 110: 109F -> bucket 55, 110F -> bucket 56."""
        assert get_bucket_index(109, self.edges) == 55
        assert get_bucket_index(110, self.edges) == 56

    def test_phl_spot_check_2f_grid(self):
        """Spot-check representative temps across the 2degF grid."""
        # Bucket index formula for interior: index = (temp - 0) // 2 + 1
        # when temp is within [0, 110).  Below 0 -> 0, >= 110 -> 56.
        expected = {
            -5: 0,    # Below 0
            15: 8,    # (14,16) bucket at index 8
            32: 17,   # (32,34) bucket at index 17
            55: 28,   # (54,56) bucket at index 28
            72: 37,   # (72,74) bucket at index 37
            100: 51,  # (100,102) bucket at index 51
            110: 56,  # Above 110
        }
        for temp, expected_idx in expected.items():
            actual = get_bucket_index(temp, self.edges)
            assert actual == expected_idx, (
                f"Temp {temp}F: expected bucket {expected_idx}, got {actual}"
            )

    def test_phl_full_range_no_gaps(self):
        """Every integer temperature from -20 to 120 should map to a valid bucket."""
        n_buckets = len(self.edges)
        for temp in range(-20, 121):
            idx = get_bucket_index(temp, self.edges)
            assert 0 <= idx < n_buckets, (
                f"Temperature {temp}F mapped to invalid bucket index {idx}"
            )
