"""
GHCN Station Discovery Module.

Downloads the GHCN station inventory and identifies suitable surrounding
stations for temperature prediction projects. Stations are selected based
on distance from a target station, data completeness, and geographic
diversity (ring and sector classification).

Supports multi-city discovery: NYC (default), Chicago, Philadelphia, or
any city specified by target coordinates and station ID.

This module is used once per city to build the expanded station list; it
does not need to run during normal training/prediction workflows.
"""

import math
import os
import sys
import logging
from typing import Optional

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GHCN_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"

# Central Park reference point
CP_LAT = config.TARGET_LAT   # 40.7789
CP_LON = config.TARGET_LON   # -73.9692

# Earth radius in miles (mean)
EARTH_RADIUS_MI = 3958.8

# Maximum distance from Central Park (miles)
MAX_DISTANCE_MI = 250

# Minimum data completeness (fraction)
MIN_COMPLETENESS = 0.80

# Date range requirements
REQUIRED_START_YEAR = 1985
REQUIRED_END_YEAR = 2024


# ===========================================================================
# Haversine and Bearing Calculations
# ===========================================================================

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points using Haversine.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of point 2 in decimal degrees.

    Returns
    -------
    float
        Distance in miles.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_MI * c


def calculate_bearing(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """Calculate the initial bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of point 2 in decimal degrees.

    Returns
    -------
    float
        Bearing in degrees (0-360, where 0=North, 90=East, etc.).
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def classify_ring(distance_mi: float) -> str:
    """Classify a station into a distance ring.

    Parameters
    ----------
    distance_mi : float
        Distance from Central Park in miles.

    Returns
    -------
    str
        Ring name: 'Ring1_Near', 'Ring2_Regional', 'Ring3_Extended', or
        'Ring4_Far'.
    """
    if distance_mi <= 50:
        return "Ring1_Near"
    elif distance_mi <= 100:
        return "Ring2_Regional"
    elif distance_mi <= 150:
        return "Ring3_Extended"
    else:
        return "Ring4_Far"


def classify_sector(bearing: float) -> str:
    """Classify a station into a compass sector based on bearing.

    Parameters
    ----------
    bearing : float
        Bearing in degrees (0-360).

    Returns
    -------
    str
        Sector name: 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', or 'NW'.
    """
    # Normalize bearing to 0-360
    bearing = bearing % 360

    if bearing >= 337.5 or bearing < 22.5:
        return "N"
    elif bearing < 67.5:
        return "NE"
    elif bearing < 112.5:
        return "E"
    elif bearing < 157.5:
        return "SE"
    elif bearing < 202.5:
        return "S"
    elif bearing < 247.5:
        return "SW"
    elif bearing < 292.5:
        return "W"
    else:
        return "NW"


# ===========================================================================
# Inventory Download and Parsing
# ===========================================================================

def download_file(url: str, output_path: str, timeout: int = 120) -> str:
    """Download a file from a URL to the local filesystem.

    Parameters
    ----------
    url : str
        URL to download from.
    output_path : str
        Local file path to save to.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    logger.info("Downloading %s ...", url)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)

    file_size = os.path.getsize(output_path)
    logger.info("Saved %s (%.1f MB)", output_path, file_size / (1024 * 1024))
    return output_path


def parse_ghcn_stations(filepath: str) -> pd.DataFrame:
    """Parse the ghcnd-stations.txt fixed-width file.

    Format:
        ID          1-11   Character
        LATITUDE   13-20   Real
        LONGITUDE  22-30   Real
        ELEVATION  32-37   Real
        STATE      39-40   Character
        NAME       42-71   Character

    Parameters
    ----------
    filepath : str
        Path to the ghcnd-stations.txt file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station_id, latitude, longitude, elevation,
        state, name.
    """
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line) < 42:
                continue
            station_id = line[0:11].strip()
            try:
                latitude = float(line[12:20].strip())
                longitude = float(line[21:30].strip())
            except (ValueError, IndexError):
                continue

            try:
                elevation = float(line[31:37].strip())
            except (ValueError, IndexError):
                elevation = None

            state = line[38:40].strip() if len(line) > 40 else ""
            name = line[41:71].strip() if len(line) > 41 else ""

            records.append({
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "elevation": elevation,
                "state": state,
                "name": name,
            })

    df = pd.DataFrame(records)
    logger.info("Parsed %d stations from %s", len(df), filepath)
    return df


def parse_ghcn_inventory(filepath: str) -> pd.DataFrame:
    """Parse the ghcnd-inventory.txt file.

    Format (space-delimited):
        ID          1-11   Character
        LATITUDE   13-20   Real
        LONGITUDE  22-30   Real
        ELEMENT    32-35   Character
        FIRSTYEAR  37-40   Integer
        LASTYEAR   42-45   Integer

    Parameters
    ----------
    filepath : str
        Path to the ghcnd-inventory.txt file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station_id, latitude, longitude, element,
        first_year, last_year.
    """
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line) < 45:
                continue
            station_id = line[0:11].strip()
            element = line[31:35].strip()
            try:
                first_year = int(line[36:40].strip())
                last_year = int(line[41:45].strip())
            except (ValueError, IndexError):
                continue

            records.append({
                "station_id": station_id,
                "element": element,
                "first_year": first_year,
                "last_year": last_year,
            })

    df = pd.DataFrame(records)
    logger.info("Parsed %d inventory records from %s", len(df), filepath)
    return df


# ===========================================================================
# Station Selection
# ===========================================================================

def discover_candidate_stations(
    stations_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    max_distance_mi: float = MAX_DISTANCE_MI,
    required_start_year: int = REQUIRED_START_YEAR,
    required_end_year: int = REQUIRED_END_YEAR,
    us_only: bool = True,
    target_count: int = 50,
    target_lat: float = CP_LAT,
    target_lon: float = CP_LON,
    target_station: str = config.TARGET_STATION,
) -> pd.DataFrame:
    """Discover candidate surrounding stations from the GHCN inventory.

    Selection criteria:
    1. Within max_distance_mi of the target station
    2. Has TMAX data covering the required date range
    3. Station ID starts with 'US' (if us_only=True)
    4. Not the target station itself
    5. Prioritize USW (airport/ASOS) stations for quality

    Parameters
    ----------
    stations_df : pd.DataFrame
        Parsed ghcnd-stations.txt data.
    inventory_df : pd.DataFrame
        Parsed ghcnd-inventory.txt data.
    max_distance_mi : float
        Maximum distance from the target station in miles.
    required_start_year : int
        Station must have data starting at or before this year.
    required_end_year : int
        Station must have data ending at or after this year.
    us_only : bool
        If True, only consider US stations (ID starts with 'US').
    target_count : int
        Approximate target number of stations to select.
    target_lat : float
        Latitude of the target station. Defaults to Central Park.
    target_lon : float
        Longitude of the target station. Defaults to Central Park.
    target_station : str
        GHCN station ID of the target station. Defaults to
        config.TARGET_STATION (Central Park).

    Returns
    -------
    pd.DataFrame
        DataFrame of selected stations with metadata columns:
        station_id, name, latitude, longitude, distance_mi, bearing,
        ring, sector, state, elevation, has_tmin, priority.
    """
    # Filter inventory for TMAX coverage
    tmax_inv = inventory_df[
        (inventory_df["element"] == "TMAX") &
        (inventory_df["first_year"] <= required_start_year) &
        (inventory_df["last_year"] >= required_end_year)
    ]["station_id"].unique()

    tmin_inv = inventory_df[
        (inventory_df["element"] == "TMIN") &
        (inventory_df["first_year"] <= required_start_year) &
        (inventory_df["last_year"] >= required_end_year)
    ]["station_id"].unique()

    tmin_set = set(tmin_inv)

    logger.info("Stations with TMAX coverage %d-%d: %d",
                required_start_year, required_end_year, len(tmax_inv))

    # Filter stations
    candidates = stations_df[stations_df["station_id"].isin(tmax_inv)].copy()

    if us_only:
        candidates = candidates[candidates["station_id"].str.startswith("US")]

    # Exclude the target station
    candidates = candidates[candidates["station_id"] != target_station]

    logger.info("US stations with TMAX coverage: %d", len(candidates))

    # Compute distance and bearing from target station
    candidates["distance_mi"] = candidates.apply(
        lambda row: haversine_distance(target_lat, target_lon,
                                       row["latitude"], row["longitude"]),
        axis=1,
    )

    # Filter by distance
    candidates = candidates[candidates["distance_mi"] <= max_distance_mi]
    logger.info("Stations within %d miles: %d", max_distance_mi, len(candidates))

    # Compute bearing and classifications
    candidates["bearing"] = candidates.apply(
        lambda row: calculate_bearing(target_lat, target_lon,
                                      row["latitude"], row["longitude"]),
        axis=1,
    )
    candidates["ring"] = candidates["distance_mi"].apply(classify_ring)
    candidates["sector"] = candidates["bearing"].apply(classify_sector)
    candidates["has_tmin"] = candidates["station_id"].isin(tmin_set)

    # Priority scoring: prefer USW stations (airport/ASOS = best quality)
    candidates["is_usw"] = candidates["station_id"].str.startswith("USW")
    # Also prefer stations with TMIN data
    candidates["priority"] = (
        candidates["is_usw"].astype(int) * 2 +
        candidates["has_tmin"].astype(int)
    )

    # Sort by priority (descending) then distance (ascending)
    candidates = candidates.sort_values(
        ["priority", "distance_mi"],
        ascending=[False, True],
    ).reset_index(drop=True)

    logger.info("Total candidate stations: %d", len(candidates))
    logger.info("  USW stations: %d", candidates["is_usw"].sum())
    logger.info("  With TMIN: %d", candidates["has_tmin"].sum())

    # Determine which original station IDs to preserve (NYC only)
    if target_station == config.TARGET_STATION:
        original_ids = set(config.SURROUNDING_STATIONS.keys())
    else:
        original_ids = None

    # Select stations: take all USW stations first, then fill with others
    # while maintaining sector diversity
    selected = _select_diverse_stations(
        candidates, target_count, original_station_ids=original_ids,
    )

    logger.info("Selected %d stations", len(selected))
    for ring in ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"]:
        ring_count = (selected["ring"] == ring).sum()
        logger.info("  %s: %d stations", ring, ring_count)
    for sector in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        sector_count = (selected["sector"] == sector).sum()
        logger.info("  %s: %d stations", sector, sector_count)

    return selected


def _select_diverse_stations(
    candidates: pd.DataFrame,
    target_count: int,
    original_station_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Select stations maintaining directional diversity.

    Strategy:
    1. Always include the original surrounding stations (if provided)
    2. Ensure every sector has at least 3 stations (if available)
    3. Ensure every ring has at least 5 stations (if available)
    4. Fill remaining slots with closest USW stations, round-robin by sector
    5. Cap at target_count

    Parameters
    ----------
    candidates : pd.DataFrame
        All candidate stations with priority, sector, ring columns.
    target_count : int
        Approximate target number of stations.
    original_station_ids : set or None
        Station IDs that must always be included (e.g. the original 14 NYC
        stations). When *None*, no stations are forced into the selection.

    Returns
    -------
    pd.DataFrame
        Selected stations.
    """
    if original_station_ids is None:
        original_station_ids = set()

    selected_ids = set()

    # Step 1: Include all original stations (if any)
    for sid in original_station_ids:
        if sid in candidates["station_id"].values:
            selected_ids.add(sid)
    logger.info("Step 1: Added %d original stations", len(selected_ids))

    # Step 2: Ensure each sector has at least 3 stations (closest first)
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for sector in sectors:
        sector_df = candidates[
            (candidates["sector"] == sector) &
            (candidates["is_usw"])
        ].sort_values("distance_mi")
        needed = max(0, 3 - len(
            [s for s in selected_ids
             if s in candidates[candidates["sector"] == sector]["station_id"].values]
        ))
        for _, row in sector_df.iterrows():
            if needed <= 0:
                break
            if row["station_id"] not in selected_ids:
                selected_ids.add(row["station_id"])
                needed -= 1

    logger.info("Step 2: After sector seeding: %d stations", len(selected_ids))

    # Step 3: Ensure each ring has at least 5 stations
    rings = ["Ring1_Near", "Ring2_Regional", "Ring3_Extended", "Ring4_Far"]
    for ring in rings:
        ring_df = candidates[
            (candidates["ring"] == ring) &
            (candidates["is_usw"])
        ].sort_values("distance_mi")
        ring_selected = [
            s for s in selected_ids
            if s in candidates[candidates["ring"] == ring]["station_id"].values
        ]
        needed = max(0, 5 - len(ring_selected))
        for _, row in ring_df.iterrows():
            if needed <= 0:
                break
            if row["station_id"] not in selected_ids:
                selected_ids.add(row["station_id"])
                needed -= 1

    logger.info("Step 3: After ring seeding: %d stations", len(selected_ids))

    # Step 4: Fill remaining slots with USW stations, round-robin by sector
    # This ensures even geographic distribution
    if len(selected_ids) < target_count:
        remaining = target_count - len(selected_ids)
        usw_remaining = candidates[
            (candidates["is_usw"]) &
            (~candidates["station_id"].isin(selected_ids))
        ].sort_values("distance_mi")

        # Build a per-sector queue of candidates
        sector_queues = {}
        for sector in sectors:
            sector_candidates = usw_remaining[usw_remaining["sector"] == sector]
            sector_queues[sector] = list(sector_candidates["station_id"])

        # Round-robin through sectors
        added = 0
        while added < remaining:
            made_progress = False
            for sector in sectors:
                if added >= remaining:
                    break
                if sector_queues[sector]:
                    sid = sector_queues[sector].pop(0)
                    selected_ids.add(sid)
                    added += 1
                    made_progress = True
            if not made_progress:
                break

    logger.info("Step 4: After filling: %d stations", len(selected_ids))

    # Return selected rows
    selected = candidates[candidates["station_id"].isin(selected_ids)].copy()
    selected = selected.sort_values("distance_mi").reset_index(drop=True)
    return selected


# ===========================================================================
# Main Discovery Pipeline
# ===========================================================================

def run_station_discovery(
    data_dir: Optional[str] = None,
    target_count: int = 50,
    target_lat: float = CP_LAT,
    target_lon: float = CP_LON,
    target_station: str = config.TARGET_STATION,
) -> pd.DataFrame:
    """Run the full station discovery pipeline.

    Downloads GHCN inventory files (if not cached), discovers candidate
    stations, selects the best ~target_count stations, and returns metadata.

    Parameters
    ----------
    data_dir : str, optional
        Directory to store inventory files. Defaults to config.DATA_DIR.
    target_count : int
        Target number of surrounding stations.
    target_lat : float
        Latitude of the target station. Defaults to Central Park.
    target_lon : float
        Longitude of the target station. Defaults to Central Park.
    target_station : str
        GHCN station ID of the target station. Defaults to
        config.TARGET_STATION (Central Park).

    Returns
    -------
    pd.DataFrame
        Selected station metadata.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    os.makedirs(data_dir, exist_ok=True)

    # Download inventory files
    stations_path = os.path.join(data_dir, "ghcnd-stations.txt")
    inventory_path = os.path.join(data_dir, "ghcnd-inventory.txt")

    if not os.path.exists(stations_path):
        download_file(GHCN_STATIONS_URL, stations_path)
    else:
        logger.info("Using cached %s", stations_path)

    if not os.path.exists(inventory_path):
        download_file(GHCN_INVENTORY_URL, inventory_path)
    else:
        logger.info("Using cached %s", inventory_path)

    # Parse
    stations_df = parse_ghcn_stations(stations_path)
    inventory_df = parse_ghcn_inventory(inventory_path)

    # Discover and select
    selected = discover_candidate_stations(
        stations_df, inventory_df,
        target_count=target_count,
        target_lat=target_lat,
        target_lon=target_lon,
        target_station=target_station,
    )

    # Ensure original 14 stations are included (NYC only)
    if target_station == config.TARGET_STATION:
        original_ids = set(config.SURROUNDING_STATIONS.keys())
        missing_originals = original_ids - set(selected["station_id"])
        if missing_originals:
            logger.warning(
                "Adding %d original stations not in discovery: %s",
                len(missing_originals), missing_originals,
            )
            for sid in missing_originals:
                station_info = stations_df[stations_df["station_id"] == sid]
                if not station_info.empty:
                    row = station_info.iloc[0]
                    dist = haversine_distance(target_lat, target_lon,
                                              row["latitude"], row["longitude"])
                    bearing = calculate_bearing(target_lat, target_lon,
                                                row["latitude"], row["longitude"])
                    new_row = {
                        "station_id": sid,
                        "name": row["name"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "elevation": row.get("elevation"),
                        "state": row.get("state", ""),
                        "distance_mi": dist,
                        "bearing": bearing,
                        "ring": classify_ring(dist),
                        "sector": classify_sector(bearing),
                        "has_tmin": True,
                        "is_usw": sid.startswith("USW"),
                        "priority": 3 if sid.startswith("USW") else 1,
                    }
                    selected = pd.concat(
                        [selected, pd.DataFrame([new_row])],
                        ignore_index=True,
                    )

    selected = selected.sort_values("distance_mi").reset_index(drop=True)

    # Save expanded stations CSV
    csv_path = os.path.join(data_dir, "stations_expanded.csv")
    selected.to_csv(csv_path, index=False)
    logger.info("Saved expanded station list to %s (%d stations)",
                csv_path, len(selected))

    return selected


def run_city_station_discovery(
    city_code: str,
    target_lat: float,
    target_lon: float,
    target_station: str,
    data_dir: Optional[str] = None,
    target_count: int = 50,
) -> pd.DataFrame:
    """Run station discovery for any city.

    Parameters
    ----------
    city_code : str
        City identifier (e.g., "nyc", "phl", "chi").
    target_lat, target_lon : float
        Latitude and longitude of the target station.
    target_station : str
        GHCN station ID of the target station.
    data_dir : str, optional
        Directory to store inventory and output files.
    target_count : int
        Target number of surrounding stations.

    Returns
    -------
    pd.DataFrame
        Selected station metadata.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    os.makedirs(data_dir, exist_ok=True)

    # Download / cache GHCN inventory files
    stations_path = os.path.join(data_dir, "ghcnd-stations.txt")
    inventory_path = os.path.join(data_dir, "ghcnd-inventory.txt")

    if not os.path.exists(stations_path):
        download_file(GHCN_STATIONS_URL, stations_path)
    else:
        logger.info("Using cached %s", stations_path)

    if not os.path.exists(inventory_path):
        download_file(GHCN_INVENTORY_URL, inventory_path)
    else:
        logger.info("Using cached %s", inventory_path)

    # Parse
    stations_df = parse_ghcn_stations(stations_path)
    inventory_df = parse_ghcn_inventory(inventory_path)

    # Discover and select (no original-station enforcement for non-NYC)
    selected = discover_candidate_stations(
        stations_df, inventory_df,
        target_count=target_count,
        target_lat=target_lat,
        target_lon=target_lon,
        target_station=target_station,
    )

    selected = selected.sort_values("distance_mi").reset_index(drop=True)

    # Save city-specific expanded stations CSV
    csv_path = os.path.join(data_dir, f"stations_expanded_{city_code}.csv")
    selected.to_csv(csv_path, index=False)
    logger.info("Saved expanded station list to %s (%d stations)",
                csv_path, len(selected))

    return selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GHCN Station Discovery")
    parser.add_argument(
        "--city", default="nyc", choices=["nyc", "phl", "chi"],
        help="City to discover stations for",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Target station count",
    )
    args = parser.parse_args()

    if args.city == "nyc":
        result = run_station_discovery(target_count=args.count)
    else:
        # Import city config
        from src.city_config import get_city_config
        cc = get_city_config(args.city)
        result = run_city_station_discovery(
            city_code=args.city,
            target_lat=cc.target_lat,
            target_lon=cc.target_lon,
            target_station=cc.target_station,
            target_count=args.count,
        )

    print(f"\nDiscovered {len(result)} stations:")
    print(result[["station_id", "name", "state", "distance_mi", "ring", "sector"]]
          .to_string(index=False))
