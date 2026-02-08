"""
Station Registry Module for NYC Temperature Prediction.

Provides utility functions to query and subset the expanded station list
by count, radius, ring, and sector. Supports the sensitivity experiments
that vary the number and geographic distribution of input stations.

All station metadata is sourced from config_expanded.py, which in turn
was generated from the GHCN station inventory.

Usage
-----
    from src.station_registry import get_stations_by_count, get_station_subsets
    closest_20 = get_stations_by_count(20)
    subsets = get_station_subsets()
"""

import os
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config_expanded


# ===========================================================================
# Core Query Functions
# ===========================================================================

def get_all_station_ids() -> list[str]:
    """Return all surrounding station IDs sorted by distance from Central Park.

    Returns
    -------
    list[str]
        Station IDs sorted by ascending distance.
    """
    metadata = config_expanded.STATION_METADATA
    sorted_ids = sorted(
        metadata.keys(),
        key=lambda sid: metadata[sid]["distance_mi"],
    )
    return sorted_ids


def get_stations_by_count(n: int) -> list[str]:
    """Return the n closest stations to Central Park.

    Parameters
    ----------
    n : int
        Number of stations to return. Must be >= 1 and <= total stations.

    Returns
    -------
    list[str]
        The n closest station IDs, sorted by distance.

    Raises
    ------
    ValueError
        If n is out of valid range.
    """
    all_ids = get_all_station_ids()
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > len(all_ids):
        raise ValueError(
            f"n={n} exceeds total station count ({len(all_ids)})"
        )
    return all_ids[:n]


def get_stations_by_radius(miles: float) -> list[str]:
    """Return all stations within the given radius of Central Park.

    Parameters
    ----------
    miles : float
        Maximum distance in miles (inclusive).

    Returns
    -------
    list[str]
        Station IDs within the radius, sorted by distance.

    Raises
    ------
    ValueError
        If miles is negative.
    """
    if miles < 0:
        raise ValueError(f"miles must be >= 0, got {miles}")

    metadata = config_expanded.STATION_METADATA
    result = [
        sid for sid in metadata
        if metadata[sid]["distance_mi"] <= miles
    ]
    result.sort(key=lambda sid: metadata[sid]["distance_mi"])
    return result


def get_stations_by_ring(ring_name: str) -> list[str]:
    """Return all stations in a given distance ring.

    Parameters
    ----------
    ring_name : str
        Ring name: 'Ring1_Near', 'Ring2_Regional', 'Ring3_Extended',
        or 'Ring4_Far'.

    Returns
    -------
    list[str]
        Station IDs in the ring, sorted by distance.

    Raises
    ------
    ValueError
        If ring_name is not a valid ring.
    """
    valid_rings = list(config_expanded.STATION_RINGS.keys())
    if ring_name not in valid_rings:
        raise ValueError(
            f"Invalid ring '{ring_name}'. Valid rings: {valid_rings}"
        )
    ids = config_expanded.STATION_RINGS[ring_name]
    metadata = config_expanded.STATION_METADATA
    return sorted(ids, key=lambda sid: metadata[sid]["distance_mi"])


def get_stations_by_sector(sector_name: str) -> list[str]:
    """Return all stations in a given compass sector.

    Parameters
    ----------
    sector_name : str
        Sector name: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'.

    Returns
    -------
    list[str]
        Station IDs in the sector, sorted by distance.

    Raises
    ------
    ValueError
        If sector_name is not a valid sector.
    """
    valid_sectors = list(config_expanded.STATION_SECTORS.keys())
    if sector_name not in valid_sectors:
        raise ValueError(
            f"Invalid sector '{sector_name}'. Valid sectors: {valid_sectors}"
        )
    ids = config_expanded.STATION_SECTORS[sector_name]
    metadata = config_expanded.STATION_METADATA
    return sorted(ids, key=lambda sid: metadata[sid]["distance_mi"])


def get_stations_by_met_sector(met_sector_name: str) -> list[str]:
    """Return all stations in a meteorological sector grouping.

    Meteorological sectors group compass sectors by weather significance:
      - WNW: W + NW (upstream cold-air advection)
      - SW:  S + SW (warm advection)
      - Coastal: E + SE (Atlantic moderation)
      - NearField: Ring 1 stations (urban/local)
      - NE: N + NE (New England influence)

    Parameters
    ----------
    met_sector_name : str
        Meteorological sector name.

    Returns
    -------
    list[str]
        Station IDs in the meteorological sector, sorted by distance.

    Raises
    ------
    ValueError
        If met_sector_name is not valid.
    """
    valid = list(config_expanded.METEOROLOGICAL_SECTORS.keys())
    if met_sector_name not in valid:
        raise ValueError(
            f"Invalid met sector '{met_sector_name}'. Valid: {valid}"
        )
    ids = config_expanded.METEOROLOGICAL_SECTORS[met_sector_name]
    metadata = config_expanded.STATION_METADATA
    return sorted(ids, key=lambda sid: metadata[sid]["distance_mi"])


def get_station_metadata(station_id: str) -> dict:
    """Return metadata for a single station.

    Parameters
    ----------
    station_id : str
        GHCN station ID.

    Returns
    -------
    dict
        Metadata dict with keys: name, state, lat, lon, distance_mi,
        bearing, ring, sector.

    Raises
    ------
    KeyError
        If station_id is not in the expanded station list.
    """
    if station_id not in config_expanded.STATION_METADATA:
        raise KeyError(
            f"Station '{station_id}' not found in expanded config."
        )
    return config_expanded.STATION_METADATA[station_id].copy()


# ===========================================================================
# Predefined Subsets (for sensitivity experiments)
# ===========================================================================

def _build_diverse_subset(n: int) -> list[str]:
    """Build a subset of n stations with proportional sector representation.

    Strategy: allocate slots to each sector proportional to its total count,
    then fill each sector's slots with the closest stations in that sector.
    Any remaining slots go to the overall closest unfilled stations.

    Parameters
    ----------
    n : int
        Target subset size.

    Returns
    -------
    list[str]
        Station IDs in the subset, sorted by distance.
    """
    metadata = config_expanded.STATION_METADATA
    sectors = config_expanded.STATION_SECTORS
    total_stations = len(metadata)

    if n >= total_stations:
        return get_all_station_ids()

    selected = set()

    # Allocate proportional slots per sector, minimum 1 per sector
    sector_names = list(sectors.keys())
    sector_counts = {s: len(sectors[s]) for s in sector_names}
    total_in_sectors = sum(sector_counts.values())

    # First pass: give each sector at least 1 station
    sector_slots = {s: 1 for s in sector_names}
    remaining = n - len(sector_names)

    if remaining > 0:
        # Distribute remaining slots proportionally
        for s in sector_names:
            extra = int(remaining * sector_counts[s] / total_in_sectors)
            sector_slots[s] += extra

        # Handle rounding shortfall
        allocated = sum(sector_slots.values())
        shortfall = n - allocated
        # Give extra slots to largest sectors first
        for s in sorted(sector_names, key=lambda x: sector_counts[x], reverse=True):
            if shortfall <= 0:
                break
            sector_slots[s] += 1
            shortfall -= 1

    # Fill each sector's slots with closest stations
    for sector_name in sector_names:
        sector_ids = sorted(
            sectors[sector_name],
            key=lambda sid: metadata[sid]["distance_mi"],
        )
        for sid in sector_ids[:sector_slots[sector_name]]:
            selected.add(sid)

    # If we ended up with more than n (due to rounding), trim farthest
    if len(selected) > n:
        selected_list = sorted(
            selected, key=lambda sid: metadata[sid]["distance_mi"]
        )
        selected = set(selected_list[:n])

    # If under n, add closest remaining stations
    if len(selected) < n:
        remaining_ids = sorted(
            [sid for sid in metadata if sid not in selected],
            key=lambda sid: metadata[sid]["distance_mi"],
        )
        for sid in remaining_ids:
            if len(selected) >= n:
                break
            selected.add(sid)

    return sorted(selected, key=lambda sid: metadata[sid]["distance_mi"])


def get_station_subsets() -> dict[int, list[str]]:
    """Return predefined station subsets for sensitivity experiments.

    Each subset maintains directional diversity (proportional representation
    from each sector).

    Returns
    -------
    dict[int, list[str]]
        Mapping of subset size -> list of station IDs.
        Predefined sizes: 15, 20, 30, 40, 50.
    """
    total = len(config_expanded.STATION_METADATA)
    sizes = [15, 20, 30, 40, 50]

    subsets = {}
    for size in sizes:
        actual_size = min(size, total)
        subsets[size] = _build_diverse_subset(actual_size)

    return subsets


def get_original_station_ids() -> list[str]:
    """Return the original 14 surrounding station IDs.

    Returns
    -------
    list[str]
        The 14 station IDs from the original config.py.
    """
    return list(config_expanded.ORIGINAL_STATIONS.keys())


def get_expanded_sector_assignments() -> dict[str, list[str]]:
    """Return expanded meteorological sector assignments for feature engineering.

    Uses the meteorologically meaningful groupings:
      - WNW (upstream cold-air advection): W + NW compass sectors
      - SW (warm advection): S + SW compass sectors
      - Coastal (Atlantic moderation): E + SE compass sectors
      - NearField (urban/local): all Ring 1 stations
      - NE (New England influence): N + NE compass sectors

    Returns
    -------
    dict[str, list[str]]
        Mapping of sector name to list of station IDs.
    """
    return {
        sector: list(stations)
        for sector, stations in config_expanded.METEOROLOGICAL_SECTORS.items()
    }


# ===========================================================================
# Summary / Reporting
# ===========================================================================

def print_station_summary() -> None:
    """Print a formatted summary of the expanded station registry."""
    metadata = config_expanded.STATION_METADATA
    rings = config_expanded.STATION_RINGS
    sectors = config_expanded.STATION_SECTORS

    print(f"Expanded Station Registry: {len(metadata)} surrounding stations")
    print(f"{'='*70}")

    print("\nRing Distribution:")
    for ring_name, sids in rings.items():
        print(f"  {ring_name}: {len(sids)} stations")

    print("\nSector Distribution:")
    for sector_name, sids in sectors.items():
        print(f"  {sector_name}: {len(sids)} stations")

    print("\nMeteorological Sectors:")
    met_sectors = config_expanded.METEOROLOGICAL_SECTORS
    for met_name, sids in met_sectors.items():
        print(f"  {met_name}: {len(sids)} stations")

    print(f"\n{'Station':<15} {'Name':<32} {'Dist':>6} {'Ring':<16} {'Sector':>6}")
    print("-" * 80)
    for sid in get_all_station_ids():
        m = metadata[sid]
        print(
            f"{sid:<15} {m['name'][:31]:<32} {m['distance_mi']:>5.0f}mi "
            f"{m['ring']:<16} {m['sector']:>6}"
        )


if __name__ == "__main__":
    print_station_summary()

    print(f"\n{'='*70}")
    print("Predefined Subsets:")
    for size, ids in get_station_subsets().items():
        # Show sector distribution
        sector_counts = {}
        for sid in ids:
            s = config_expanded.STATION_METADATA[sid]["sector"]
            sector_counts[s] = sector_counts.get(s, 0) + 1
        sectors_str = ", ".join(f"{k}:{v}" for k, v in sorted(sector_counts.items()))
        print(f"  {size} stations: {len(ids)} actual ({sectors_str})")
