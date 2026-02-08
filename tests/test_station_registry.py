"""
Tests for the station registry and expanded configuration.

Validates station discovery results, geographic metadata, ring/sector
classifications, subset functions, and overall data integrity.
"""

import math
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
import config_expanded
from src.station_registry import (
    get_all_station_ids,
    get_stations_by_count,
    get_stations_by_radius,
    get_stations_by_ring,
    get_stations_by_sector,
    get_stations_by_met_sector,
    get_station_metadata,
    get_station_subsets,
    get_original_station_ids,
    get_expanded_sector_assignments,
)
from src.station_discovery import (
    haversine_distance,
    calculate_bearing,
    classify_ring,
    classify_sector,
    CP_LAT,
    CP_LON,
)


# ===========================================================================
# Test: Station Count and Coverage
# ===========================================================================

class TestStationCount:
    """Tests for total station count and coverage."""

    def test_total_station_count(self):
        """Expanded config should have ~50 surrounding stations."""
        count = len(config_expanded.SURROUNDING_STATIONS)
        assert 40 <= count <= 60, f"Expected 40-60 stations, got {count}"

    def test_all_stations_includes_target(self):
        """ALL_STATIONS should include Central Park target station."""
        assert config.TARGET_STATION in config_expanded.ALL_STATIONS

    def test_all_stations_count(self):
        """ALL_STATIONS = target + surrounding."""
        expected = len(config_expanded.SURROUNDING_STATIONS) + 1
        assert len(config_expanded.ALL_STATIONS) == expected

    def test_target_not_in_surrounding(self):
        """Target station should not be in surrounding stations."""
        assert config.TARGET_STATION not in config_expanded.SURROUNDING_STATIONS


# ===========================================================================
# Test: Original 14 Stations Preserved
# ===========================================================================

class TestOriginalStations:
    """Tests that all original 14 stations are in the expanded set."""

    def test_original_stations_count(self):
        """Original config should have exactly 14 surrounding stations."""
        assert len(config_expanded.ORIGINAL_STATIONS) == 14

    def test_all_original_in_expanded(self):
        """Every original station must appear in the expanded set."""
        expanded_ids = set(config_expanded.SURROUNDING_STATIONS.keys())
        for sid in config_expanded.ORIGINAL_STATIONS:
            assert sid in expanded_ids, (
                f"Original station {sid} missing from expanded set"
            )

    def test_original_matches_base_config(self):
        """ORIGINAL_STATIONS should match the base config's list."""
        assert set(config_expanded.ORIGINAL_STATIONS.keys()) == set(
            config.SURROUNDING_STATIONS.keys()
        )

    def test_get_original_station_ids(self):
        """get_original_station_ids should return the original 14 IDs."""
        ids = get_original_station_ids()
        assert len(ids) == 14
        assert set(ids) == set(config.SURROUNDING_STATIONS.keys())


# ===========================================================================
# Test: Station Metadata
# ===========================================================================

class TestStationMetadata:
    """Tests for STATION_METADATA completeness and correctness."""

    def test_metadata_count_matches_surrounding(self):
        """Every surrounding station should have metadata."""
        assert len(config_expanded.STATION_METADATA) == len(
            config_expanded.SURROUNDING_STATIONS
        )

    def test_metadata_has_required_fields(self):
        """Each metadata entry should have all required fields."""
        required_fields = {
            "name", "state", "lat", "lon", "distance_mi",
            "bearing", "ring", "sector",
        }
        for sid, meta in config_expanded.STATION_METADATA.items():
            for field in required_fields:
                assert field in meta, (
                    f"Station {sid} missing field '{field}'"
                )

    def test_metadata_lat_lon_reasonable(self):
        """Lat/lon should be in reasonable range for NE US."""
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert 38.0 <= meta["lat"] <= 44.0, (
                f"Station {sid} lat {meta['lat']} out of range"
            )
            assert -77.5 <= meta["lon"] <= -70.5, (
                f"Station {sid} lon {meta['lon']} out of range"
            )

    def test_metadata_distance_positive(self):
        """All distances should be positive (no station at zero)."""
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert meta["distance_mi"] > 0, (
                f"Station {sid} has non-positive distance"
            )

    def test_metadata_distance_within_250(self):
        """All stations should be within 250 miles."""
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert meta["distance_mi"] <= 250, (
                f"Station {sid} distance {meta['distance_mi']} exceeds 250mi"
            )

    def test_metadata_bearing_range(self):
        """Bearings should be in [0, 360)."""
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert 0 <= meta["bearing"] < 360, (
                f"Station {sid} bearing {meta['bearing']} out of range"
            )

    def test_metadata_ring_valid(self):
        """Ring should be one of the four valid rings."""
        valid = {"Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"}
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert meta["ring"] in valid, (
                f"Station {sid} has invalid ring '{meta['ring']}'"
            )

    def test_metadata_sector_valid(self):
        """Sector should be one of the eight valid sectors."""
        valid = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
        for sid, meta in config_expanded.STATION_METADATA.items():
            assert meta["sector"] in valid, (
                f"Station {sid} has invalid sector '{meta['sector']}'"
            )

    def test_get_station_metadata(self):
        """get_station_metadata should return correct data for a known station."""
        meta = get_station_metadata("USW00014735")  # Albany
        assert meta["name"] == "ALBANY INTL AP"
        assert meta["state"] == "NY"
        assert 130 <= meta["distance_mi"] <= 145

    def test_get_station_metadata_unknown_raises(self):
        """get_station_metadata should raise KeyError for unknown station."""
        with pytest.raises(KeyError):
            get_station_metadata("FAKE_STATION_ID")


# ===========================================================================
# Test: Distance Calculations
# ===========================================================================

class TestDistanceCalculations:
    """Tests for haversine distance accuracy."""

    def test_albany_distance(self):
        """Albany should be roughly 130-145 miles from Central Park."""
        meta = config_expanded.STATION_METADATA.get("USW00014735")
        if meta:
            dist = haversine_distance(CP_LAT, CP_LON, meta["lat"], meta["lon"])
            assert 130 <= dist <= 145, (
                f"Albany distance {dist:.1f} not in expected range"
            )

    def test_newark_distance(self):
        """Newark should be roughly 10-15 miles from Central Park."""
        meta = config_expanded.STATION_METADATA.get("USW00014734")
        if meta:
            dist = haversine_distance(CP_LAT, CP_LON, meta["lat"], meta["lon"])
            assert 8 <= dist <= 18, (
                f"Newark distance {dist:.1f} not in expected range"
            )

    def test_haversine_zero_distance(self):
        """Same point should give zero distance."""
        dist = haversine_distance(40.7789, -73.9692, 40.7789, -73.9692)
        assert dist < 0.01

    def test_metadata_distance_matches_haversine(self):
        """Stored distances should match fresh haversine calculations."""
        for sid, meta in config_expanded.STATION_METADATA.items():
            computed = haversine_distance(
                CP_LAT, CP_LON, meta["lat"], meta["lon"]
            )
            assert abs(computed - meta["distance_mi"]) < 1.0, (
                f"Station {sid}: stored={meta['distance_mi']:.1f}, "
                f"computed={computed:.1f}"
            )


# ===========================================================================
# Test: Bearing Calculations
# ===========================================================================

class TestBearingCalculations:
    """Tests for compass bearing accuracy."""

    def test_albany_roughly_north(self):
        """Albany is roughly north of Central Park."""
        meta = config_expanded.STATION_METADATA.get("USW00014735")
        if meta:
            assert meta["sector"] == "N"
            # Bearing should be close to 0/360
            assert meta["bearing"] < 22.5 or meta["bearing"] >= 337.5

    def test_jfk_roughly_southeast(self):
        """JFK should be in the SE sector."""
        meta = config_expanded.STATION_METADATA.get("USW00094789")
        if meta:
            assert meta["sector"] == "SE"
            assert 112.5 <= meta["bearing"] < 157.5

    def test_bearing_range(self):
        """Bearing should always be in [0, 360)."""
        bearing = calculate_bearing(40.7789, -73.9692, 42.0, -74.0)
        assert 0 <= bearing < 360

    def test_due_north_bearing(self):
        """Point due north should have bearing close to 0."""
        bearing = calculate_bearing(40.0, -74.0, 42.0, -74.0)
        assert bearing < 5 or bearing > 355

    def test_due_east_bearing(self):
        """Point due east should have bearing close to 90."""
        bearing = calculate_bearing(40.0, -74.0, 40.0, -72.0)
        assert 85 <= bearing <= 95


# ===========================================================================
# Test: Ring Classification
# ===========================================================================

class TestRingClassification:
    """Tests for distance ring assignments."""

    def test_ring_count_totals(self):
        """Sum of all ring counts should equal total stations."""
        total = sum(
            len(sids)
            for sids in config_expanded.STATION_RINGS.values()
        )
        assert total == len(config_expanded.SURROUNDING_STATIONS)

    def test_ring_no_overlap(self):
        """Stations should appear in exactly one ring."""
        all_ids = []
        for sids in config_expanded.STATION_RINGS.values():
            all_ids.extend(sids)
        assert len(all_ids) == len(set(all_ids)), "Duplicate station in rings"

    def test_ring1_near_field_distance(self):
        """Ring1_Near stations should be within 50 miles."""
        for sid in config_expanded.STATION_RINGS["Ring1_Near"]:
            meta = config_expanded.STATION_METADATA[sid]
            assert meta["distance_mi"] <= 50, (
                f"{sid} in Ring1_Near but distance={meta['distance_mi']:.1f}"
            )

    def test_ring4_far_distance(self):
        """Ring4_Far stations should be between 150 and 250 miles."""
        for sid in config_expanded.STATION_RINGS["Ring4_Far"]:
            meta = config_expanded.STATION_METADATA[sid]
            assert 150 <= meta["distance_mi"] <= 250, (
                f"{sid} in Ring4_Far but distance={meta['distance_mi']:.1f}"
            )

    def test_classify_ring_boundaries(self):
        """Test ring boundary classification."""
        assert classify_ring(25) == "Ring1_Near"
        assert classify_ring(50) == "Ring1_Near"
        assert classify_ring(50.1) == "Ring2_Regional"
        assert classify_ring(100) == "Ring2_Regional"
        assert classify_ring(100.1) == "Ring3_Extended"
        assert classify_ring(150) == "Ring3_Extended"
        assert classify_ring(150.1) == "Ring4_Far"
        assert classify_ring(200) == "Ring4_Far"

    def test_get_stations_by_ring(self):
        """get_stations_by_ring should return correct stations."""
        ring1 = get_stations_by_ring("Ring1_Near")
        assert len(ring1) == len(config_expanded.STATION_RINGS["Ring1_Near"])
        assert set(ring1) == set(config_expanded.STATION_RINGS["Ring1_Near"])

    def test_get_stations_by_ring_invalid(self):
        """Invalid ring name should raise ValueError."""
        with pytest.raises(ValueError):
            get_stations_by_ring("Ring5_VeryFar")


# ===========================================================================
# Test: Sector Classification
# ===========================================================================

class TestSectorClassification:
    """Tests for compass sector assignments."""

    def test_sector_count_totals(self):
        """Sum of all sector counts should equal total stations."""
        total = sum(
            len(sids)
            for sids in config_expanded.STATION_SECTORS.values()
        )
        assert total == len(config_expanded.SURROUNDING_STATIONS)

    def test_sector_no_overlap(self):
        """Stations should appear in exactly one sector."""
        all_ids = []
        for sids in config_expanded.STATION_SECTORS.values():
            all_ids.extend(sids)
        assert len(all_ids) == len(set(all_ids)), "Duplicate station in sectors"

    def test_all_eight_sectors_present(self):
        """All 8 compass sectors should have at least 1 station."""
        for sector in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            assert len(config_expanded.STATION_SECTORS[sector]) >= 1, (
                f"Sector {sector} has no stations"
            )

    def test_classify_sector_boundaries(self):
        """Test sector boundary classification."""
        assert classify_sector(0) == "N"
        assert classify_sector(22.4) == "N"
        assert classify_sector(22.5) == "NE"
        assert classify_sector(45) == "NE"
        assert classify_sector(90) == "E"
        assert classify_sector(135) == "SE"
        assert classify_sector(180) == "S"
        assert classify_sector(225) == "SW"
        assert classify_sector(270) == "W"
        assert classify_sector(315) == "NW"
        assert classify_sector(337.5) == "N"
        assert classify_sector(359.9) == "N"

    def test_get_stations_by_sector(self):
        """get_stations_by_sector should return correct stations."""
        north = get_stations_by_sector("N")
        assert len(north) == len(config_expanded.STATION_SECTORS["N"])
        assert set(north) == set(config_expanded.STATION_SECTORS["N"])

    def test_get_stations_by_sector_invalid(self):
        """Invalid sector name should raise ValueError."""
        with pytest.raises(ValueError):
            get_stations_by_sector("NNE")

    def test_sector_assignments_geographic_sense(self):
        """Spot-check that sector assignments make geographic sense."""
        # Atlantic City should be roughly south
        meta = config_expanded.STATION_METADATA.get("USW00093730")
        if meta:
            assert meta["sector"] == "S", (
                f"Atlantic City sector={meta['sector']}, expected S"
            )

        # Allentown should be roughly west
        meta = config_expanded.STATION_METADATA.get("USW00014737")
        if meta:
            assert meta["sector"] == "W", (
                f"Allentown sector={meta['sector']}, expected W"
            )


# ===========================================================================
# Test: Meteorological Sectors
# ===========================================================================

class TestMetSectors:
    """Tests for meteorological sector groupings."""

    def test_met_sector_wnw_contents(self):
        """WNW met sector should contain W + NW compass sectors."""
        wnw = set(config_expanded.METEOROLOGICAL_SECTORS["WNW"])
        expected = set(
            config_expanded.STATION_SECTORS["W"]
            + config_expanded.STATION_SECTORS["NW"]
        )
        assert wnw == expected

    def test_met_sector_coastal_contents(self):
        """Coastal met sector should contain E + SE compass sectors."""
        coastal = set(config_expanded.METEOROLOGICAL_SECTORS["Coastal"])
        expected = set(
            config_expanded.STATION_SECTORS["E"]
            + config_expanded.STATION_SECTORS["SE"]
        )
        assert coastal == expected

    def test_met_sector_nearfield_is_ring1(self):
        """NearField met sector should be Ring 1 stations."""
        nearfield = set(config_expanded.METEOROLOGICAL_SECTORS["NearField"])
        ring1 = set(config_expanded.STATION_RINGS["Ring1_Near"])
        assert nearfield == ring1

    def test_get_stations_by_met_sector(self):
        """get_stations_by_met_sector should return correct stations."""
        wnw = get_stations_by_met_sector("WNW")
        assert len(wnw) > 0
        assert set(wnw) == set(
            config_expanded.METEOROLOGICAL_SECTORS["WNW"]
        )

    def test_get_expanded_sector_assignments(self):
        """get_expanded_sector_assignments should return 5 sectors."""
        assignments = get_expanded_sector_assignments()
        assert len(assignments) == 5
        assert "WNW" in assignments
        assert "Coastal" in assignments
        assert "NearField" in assignments


# ===========================================================================
# Test: get_stations_by_count
# ===========================================================================

class TestGetStationsByCount:
    """Tests for get_stations_by_count function."""

    def test_returns_correct_count(self):
        """Should return exactly n stations."""
        for n in [5, 10, 15, 20, 30, 50]:
            if n <= len(config_expanded.STATION_METADATA):
                result = get_stations_by_count(n)
                assert len(result) == n

    def test_sorted_by_distance(self):
        """Returned stations should be sorted by ascending distance."""
        result = get_stations_by_count(20)
        meta = config_expanded.STATION_METADATA
        distances = [meta[sid]["distance_mi"] for sid in result]
        assert distances == sorted(distances)

    def test_n_equals_1(self):
        """n=1 should return the closest station."""
        result = get_stations_by_count(1)
        assert len(result) == 1
        # Should be the closest station
        all_sorted = get_all_station_ids()
        assert result[0] == all_sorted[0]

    def test_n_too_large_raises(self):
        """n exceeding total count should raise ValueError."""
        total = len(config_expanded.STATION_METADATA)
        with pytest.raises(ValueError):
            get_stations_by_count(total + 1)

    def test_n_zero_raises(self):
        """n=0 should raise ValueError."""
        with pytest.raises(ValueError):
            get_stations_by_count(0)

    def test_n_negative_raises(self):
        """Negative n should raise ValueError."""
        with pytest.raises(ValueError):
            get_stations_by_count(-5)


# ===========================================================================
# Test: get_stations_by_radius
# ===========================================================================

class TestGetStationsByRadius:
    """Tests for get_stations_by_radius function."""

    def test_radius_50_matches_ring1(self):
        """50-mile radius should match Ring1_Near stations."""
        result = set(get_stations_by_radius(50))
        ring1 = set(config_expanded.STATION_RINGS["Ring1_Near"])
        assert result == ring1

    def test_radius_100_includes_ring1_and_ring2(self):
        """100-mile radius should include Ring 1 and Ring 2."""
        result = set(get_stations_by_radius(100))
        ring1 = set(config_expanded.STATION_RINGS["Ring1_Near"])
        ring2 = set(config_expanded.STATION_RINGS["Ring2_Regional"])
        expected = ring1 | ring2
        assert result == expected

    def test_radius_250_includes_all(self):
        """250-mile radius should include all stations."""
        result = get_stations_by_radius(250)
        assert len(result) == len(config_expanded.STATION_METADATA)

    def test_radius_0_returns_empty(self):
        """0-mile radius should return no stations."""
        result = get_stations_by_radius(0)
        assert len(result) == 0

    def test_radius_negative_raises(self):
        """Negative radius should raise ValueError."""
        with pytest.raises(ValueError):
            get_stations_by_radius(-10)

    def test_radius_sorted_by_distance(self):
        """Results should be sorted by distance."""
        result = get_stations_by_radius(150)
        meta = config_expanded.STATION_METADATA
        distances = [meta[sid]["distance_mi"] for sid in result]
        assert distances == sorted(distances)


# ===========================================================================
# Test: Station Subsets
# ===========================================================================

class TestStationSubsets:
    """Tests for predefined station subsets."""

    def test_subset_sizes(self):
        """Subsets should be available for sizes 15, 20, 30, 40, 50."""
        subsets = get_station_subsets()
        for size in [15, 20, 30, 40, 50]:
            assert size in subsets, f"Missing subset for size {size}"
            assert len(subsets[size]) == min(
                size, len(config_expanded.STATION_METADATA)
            )

    def test_subsets_are_nested(self):
        """Smaller subsets should be approximately nested in larger ones.

        Not strictly required, but smaller subsets should share most
        stations with larger subsets.
        """
        subsets = get_station_subsets()
        for small, large in [(15, 20), (20, 30), (30, 40), (40, 50)]:
            if small in subsets and large in subsets:
                overlap = set(subsets[small]) & set(subsets[large])
                # At least 60% of the smaller set should be in the larger
                min_overlap = int(len(subsets[small]) * 0.6)
                assert len(overlap) >= min_overlap, (
                    f"Subsets {small} and {large} share only {len(overlap)} "
                    f"stations (need >= {min_overlap})"
                )

    def test_subset_diversity(self):
        """Each subset should have representation from multiple sectors."""
        subsets = get_station_subsets()
        meta = config_expanded.STATION_METADATA

        for size, ids in subsets.items():
            sectors_represented = set()
            for sid in ids:
                sectors_represented.add(meta[sid]["sector"])
            # Each subset should cover at least 5 of 8 sectors
            assert len(sectors_represented) >= 5, (
                f"Subset {size} covers only {len(sectors_represented)} sectors: "
                f"{sectors_represented}"
            )

    def test_subset_15_includes_some_originals(self):
        """Size-15 subset should include some original stations.

        The original 14 include many distant stations (Albany 136mi,
        Boston 188mi, Syracuse 195mi) that a 15-station diversity subset
        may replace with closer alternatives. But nearby originals like
        Newark, JFK, and Bridgeport should still appear.
        """
        subsets = get_station_subsets()
        original_ids = set(get_original_station_ids())
        subset_15 = set(subsets[15])
        # At least 5 of the original 14 should be in the 15-station subset
        overlap = original_ids & subset_15
        assert len(overlap) >= 5, (
            f"Only {len(overlap)} of 14 originals in 15-station subset"
        )

    def test_subset_50_covers_most(self):
        """Size-50 subset should contain 50 stations from the full set."""
        subsets = get_station_subsets()
        all_ids = get_all_station_ids()
        total = len(all_ids)
        expected_size = min(50, total)
        assert len(subsets[50]) == expected_size
        # All subset stations must be valid station IDs
        assert set(subsets[50]).issubset(set(all_ids))


# ===========================================================================
# Test: Data Files Exist
# ===========================================================================

class TestDataFiles:
    """Tests that required data files exist for all expanded stations."""

    def test_expanded_csv_exists(self):
        """stations_expanded.csv should exist."""
        path = os.path.join(config.DATA_DIR, "stations_expanded.csv")
        assert os.path.exists(path), f"Missing {path}"

    def test_station_csv_files_exist(self):
        """Each station should have a downloaded CSV file in data/raw/."""
        raw_dir = config.RAW_DATA_DIR
        missing = []
        for sid in config_expanded.SURROUNDING_STATIONS:
            csv_path = os.path.join(raw_dir, f"{sid}.csv")
            if not os.path.exists(csv_path):
                missing.append(sid)
        assert len(missing) == 0, (
            f"Missing CSV files for {len(missing)} stations: {missing[:5]}"
        )

    def test_target_station_csv_exists(self):
        """Central Park target station should have data."""
        csv_path = os.path.join(
            config.RAW_DATA_DIR, f"{config.TARGET_STATION}.csv"
        )
        assert os.path.exists(csv_path)


# ===========================================================================
# Test: get_all_station_ids
# ===========================================================================

class TestGetAllStationIds:
    """Tests for get_all_station_ids function."""

    def test_returns_all(self):
        """Should return all surrounding station IDs."""
        ids = get_all_station_ids()
        assert len(ids) == len(config_expanded.SURROUNDING_STATIONS)

    def test_sorted_by_distance(self):
        """Should be sorted by ascending distance."""
        ids = get_all_station_ids()
        meta = config_expanded.STATION_METADATA
        distances = [meta[sid]["distance_mi"] for sid in ids]
        assert distances == sorted(distances)

    def test_no_duplicates(self):
        """Should have no duplicate IDs."""
        ids = get_all_station_ids()
        assert len(ids) == len(set(ids))
