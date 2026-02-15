"""
Multi-City Configuration Framework.

Provides a CityConfig dataclass and registry for managing city-specific
parameters across the prediction market pipeline. Each city has its own
target station, surrounding station network, Kalshi contract definitions,
sounding station, NWP grid point, and climate parameters.

Usage:
    from src.city_config import get_city_config, list_cities
    nyc = get_city_config("nyc")
    phl = get_city_config("phl")
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CityConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class CityConfig:
    """Configuration container for a single city's prediction pipeline.

    Holds all city-specific parameters needed across ingestion, feature
    engineering, modeling, calibration, and trading layers.

    Attributes:
        city_name: Human-readable city name (e.g., "New York City").
        city_code: Short lowercase code used as a key (e.g., "nyc").
        kalshi_ticker: Kalshi market ticker prefix (e.g., "KXHIGHNY").
        target_station: GHCN station ID for the primary observation site.
        target_station_name: Human-readable name of the target station.
        target_lat: Latitude of the target station (decimal degrees).
        target_lon: Longitude of the target station (decimal degrees).
        timezone: IANA timezone string (e.g., "America/New_York").
        igra_station_id: IGRA station ID for upper-air soundings.
        igra_station_name: Human-readable name of the IGRA station.
        nwp_lat: Latitude of the NWP grid point for forecast extraction.
        nwp_lon: Longitude of the NWP grid point for forecast extraction.
        bucket_edges: List of (low, high) tuples defining Kalshi contract
            temperature buckets.  Use -999 / 999 as open-ended sentinels.
        bucket_labels: Human-readable labels matching *bucket_edges*.
        monthly_tmax_mean: Climatological mean TMAX (deg F) by month (1-12).
        monthly_tmax_std: Climatological std-dev of TMAX (deg F) by month.
        data_dir: Filesystem path to the city's data directory.
        models_dir: Filesystem path to the city's saved-model directory.
        results_dir: Filesystem path to the city's results/output directory.
    """

    # Identity
    city_name: str
    city_code: str
    kalshi_ticker: str

    # Target station
    target_station: str
    target_station_name: str
    target_lat: float
    target_lon: float

    # Timezone
    timezone: str

    # Upper-air sounding station (IGRA)
    igra_station_id: str
    igra_station_name: str

    # NWP grid point
    nwp_lat: float
    nwp_lon: float

    # Kalshi bucket definitions
    bucket_edges: List[Tuple[float, float]] = field(default_factory=list)
    bucket_labels: List[str] = field(default_factory=list)

    # Climatological parameters (month -> value)
    monthly_tmax_mean: Dict[int, float] = field(default_factory=dict)
    monthly_tmax_std: Dict[int, float] = field(default_factory=dict)

    # Filesystem paths
    data_dir: str = ""
    models_dir: str = ""
    results_dir: str = ""


# ---------------------------------------------------------------------------
# Shared bucket definitions (2°F resolution matching Kalshi contracts)
# ---------------------------------------------------------------------------
def _make_2f_bucket_grid(
    floor: int, ceiling: int
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Generate 2°F-resolution bucket edges and labels.

    Creates a grid of 2°F-wide buckets from *floor* to *ceiling*, with
    open-ended tails using -999/999 sentinels.  This matches the actual
    Kalshi contract structure (e.g., KXHIGHNY "between" contracts are 2°F).

    Parameters
    ----------
    floor : int
        Lower bound of the first non-tail bucket (must be even).
    ceiling : int
        Upper bound of the last non-tail bucket (must be even).

    Returns
    -------
    edges : list of (float, float)
        Bucket boundary tuples, starting with (-999, floor) and ending
        with (ceiling, 999).
    labels : list of str
        Human-readable labels (e.g., "Below 0", "0-2", "2-4", ...,
        "Above 110").
    """
    assert floor % 2 == 0 and ceiling % 2 == 0, "floor and ceiling must be even"
    edges: List[Tuple[float, float]] = [(-999, float(floor))]
    labels: List[str] = [f"Below {floor}"]
    for lo in range(floor, ceiling, 2):
        hi = lo + 2
        edges.append((float(lo), float(hi)))
        labels.append(f"{lo}-{hi}")
    edges.append((float(ceiling), 999))
    labels.append(f"Above {ceiling}")
    return edges, labels


# NYC and PHL: 0°F floor, 110°F ceiling → 57 buckets
_NYC_PHL_BUCKET_EDGES, _NYC_PHL_BUCKET_LABELS = _make_2f_bucket_grid(0, 110)

# Chicago: -10°F floor, 110°F ceiling → 62 buckets (colder winters)
_CHI_BUCKET_EDGES, _CHI_BUCKET_LABELS = _make_2f_bucket_grid(-10, 110)


# ---------------------------------------------------------------------------
# City configuration instances
# ---------------------------------------------------------------------------
_NYC_CONFIG = CityConfig(
    city_name="New York City",
    city_code="nyc",
    kalshi_ticker="KXHIGHNY",
    target_station="USW00094728",
    target_station_name="Central Park",
    target_lat=40.7789,
    target_lon=-73.9692,
    timezone="America/New_York",
    igra_station_id="USM00072501",
    igra_station_name="Upton/Brookhaven OKX",
    nwp_lat=40.7789,
    nwp_lon=-73.9692,
    bucket_edges=_NYC_PHL_BUCKET_EDGES.copy(),
    bucket_labels=_NYC_PHL_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 39.0,
        2: 42.0,
        3: 50.0,
        4: 62.0,
        5: 72.0,
        6: 80.0,
        7: 85.0,
        8: 84.0,
        9: 76.0,
        10: 65.0,
        11: 54.0,
        12: 43.0,
    },
    monthly_tmax_std={
        1: 11.0,
        2: 10.5,
        3: 10.0,
        4: 9.5,
        5: 8.0,
        6: 6.5,
        7: 5.5,
        8: 5.5,
        9: 6.5,
        10: 8.0,
        11: 9.5,
        12: 10.5,
    },
    # NYC keeps root-level directories for backward compatibility
    data_dir=os.path.join(PROJECT_ROOT, "data"),
    models_dir=os.path.join(PROJECT_ROOT, "models"),
    results_dir=os.path.join(PROJECT_ROOT, "results"),
)

_PHL_CONFIG = CityConfig(
    city_name="Philadelphia",
    city_code="phl",
    kalshi_ticker="KXHIGHPHL",
    target_station="USW00013739",
    target_station_name="Philadelphia International Airport",
    target_lat=39.8733,
    target_lon=-75.2269,
    timezone="America/New_York",
    igra_station_id="USM00072403",
    igra_station_name="Sterling VA / IAD",
    nwp_lat=39.8733,
    nwp_lon=-75.2269,
    bucket_edges=_NYC_PHL_BUCKET_EDGES.copy(),
    bucket_labels=_NYC_PHL_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 40.0,
        2: 44.0,
        3: 53.0,
        4: 64.0,
        5: 74.0,
        6: 83.0,
        7: 87.0,
        8: 85.0,
        9: 78.0,
        10: 66.0,
        11: 55.0,
        12: 44.0,
    },
    monthly_tmax_std={
        1: 11.0,
        2: 10.5,
        3: 10.0,
        4: 9.0,
        5: 8.0,
        6: 6.0,
        7: 5.5,
        8: 5.5,
        9: 6.5,
        10: 8.0,
        11: 9.5,
        12: 10.5,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "philadelphia"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "philadelphia"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "philadelphia"),
)

_CHI_CONFIG = CityConfig(
    city_name="Chicago",
    city_code="chi",
    kalshi_ticker="KXHIGHCHI",
    target_station="USW00094846",
    target_station_name="O'Hare International",
    target_lat=41.9742,
    target_lon=-87.9073,
    timezone="America/Chicago",
    igra_station_id="USM00074455",
    igra_station_name="Davenport DVN",
    nwp_lat=41.9742,
    nwp_lon=-87.9073,
    bucket_edges=_CHI_BUCKET_EDGES.copy(),
    bucket_labels=_CHI_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 32.0,
        2: 36.0,
        3: 47.0,
        4: 59.0,
        5: 70.0,
        6: 80.0,
        7: 84.0,
        8: 82.0,
        9: 75.0,
        10: 62.0,
        11: 48.0,
        12: 35.0,
    },
    monthly_tmax_std={
        1: 12.0,
        2: 12.0,
        3: 12.0,
        4: 11.0,
        5: 9.0,
        6: 7.0,
        7: 6.0,
        8: 6.0,
        9: 8.0,
        10: 10.0,
        11: 11.0,
        12: 12.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "chicago"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "chicago"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "chicago"),
)


# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------
_CITY_REGISTRY: Dict[str, CityConfig] = {
    "nyc": _NYC_CONFIG,
    "phl": _PHL_CONFIG,
    "chi": _CHI_CONFIG,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_city_config(city_code: str) -> CityConfig:
    """Return the CityConfig for a given city code.

    Args:
        city_code: Lowercase city identifier (e.g., "nyc", "phl", "chi").

    Returns:
        The corresponding CityConfig instance.

    Raises:
        ValueError: If *city_code* is not found in the registry.
    """
    code = city_code.strip().lower()
    if code not in _CITY_REGISTRY:
        available = ", ".join(sorted(_CITY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown city code '{city_code}'. Available cities: {available}"
        )
    return _CITY_REGISTRY[code]


def list_cities() -> List[str]:
    """Return a sorted list of all registered city codes.

    Returns:
        List of city code strings (e.g., ["chi", "nyc", "phl"]).
    """
    return sorted(_CITY_REGISTRY.keys())


def get_bucket_index(tmax: float, bucket_edges: List[Tuple[float, float]]) -> int:
    """Determine which bucket a temperature value falls into.

    Iterates through *bucket_edges* and returns the index of the first
    bucket whose range contains *tmax*.  The comparison uses
    ``low <= tmax < high`` for all buckets except the last, which uses
    ``low <= tmax <= high`` to capture the upper sentinel.

    Args:
        tmax: Observed or forecast maximum temperature (deg F).
        bucket_edges: List of (low, high) tuples defining bucket boundaries.

    Returns:
        Zero-based index of the matching bucket.

    Raises:
        ValueError: If *tmax* does not fall into any bucket.
    """
    n_buckets = len(bucket_edges)
    for i, (low, high) in enumerate(bucket_edges):
        if i == n_buckets - 1:
            # Last bucket: inclusive on both ends to capture sentinel
            if low <= tmax <= high:
                return i
        else:
            if low <= tmax < high:
                return i
    raise ValueError(
        f"Temperature {tmax} does not fall into any bucket. "
        f"Edges: {bucket_edges}"
    )


def ensure_city_dirs(city_config: CityConfig) -> None:
    """Create the data, models, and results directories for a city if missing.

    Args:
        city_config: The CityConfig whose directory paths should be created.
    """
    for dir_path in (city_config.data_dir, city_config.models_dir, city_config.results_dir):
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
