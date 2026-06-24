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
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.city_config_runtime_data import CITY_RUNTIME_DATA

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

    # Contract definition alignment
    bucket_low_inclusive: bool = True
    bucket_high_inclusive_last: bool = True
    contract_daily_boundary_local: str = "00:00-23:59"
    settlement_rounding: str = "integer_fahrenheit"

    # Operational assumptions
    observation_anchor_stations: List[str] = field(default_factory=list)
    operational_cutoff_local: str = "06:00"
    operational_sources: List[str] = field(default_factory=list)

    # Extended metadata consolidated from legacy per-city config modules
    all_stations: Dict[str, str] = field(default_factory=dict)
    surrounding_stations: Dict[str, str] = field(default_factory=dict)
    asos_station_map: Dict[str, str] = field(default_factory=dict)
    station_metadata: Dict[str, Dict] = field(default_factory=dict)
    station_rings: Dict[str, List[str]] = field(default_factory=dict)
    station_sectors: Dict[str, List[str]] = field(default_factory=dict)
    meteorological_sectors: Dict[str, List[str]] = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    min_completeness: float = 0.8
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    input_variables: List[str] = field(default_factory=lambda: ["TMAX", "TMIN"])
    max_forward_fill_days: int = 7
    batch_size: int = 32


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

# Expansion cities (Phase 4). Floors/ceilings reflect each climate's realized
# extremes so the open tails stay rare; the 2°F interior matches the verified
# Kalshi contract ladder (see results/expansion/contract_verification.json).
_DEN_BUCKET_EDGES, _DEN_BUCKET_LABELS = _make_2f_bucket_grid(-10, 110)
_DC_BUCKET_EDGES, _DC_BUCKET_LABELS = _make_2f_bucket_grid(0, 110)
_LAX_BUCKET_EDGES, _LAX_BUCKET_LABELS = _make_2f_bucket_grid(30, 110)
_MIA_BUCKET_EDGES, _MIA_BUCKET_LABELS = _make_2f_bucket_grid(30, 110)
_PHX_BUCKET_EDGES, _PHX_BUCKET_LABELS = _make_2f_bucket_grid(20, 120)


def _runtime(city_code: str, field_name: str, default):
    return CITY_RUNTIME_DATA.get(city_code, {}).get(field_name, default)


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
    # NYC reorganization (Phase 1 / Phase G): NYC's raw and processed data now
    # live under data/nyc/, matching the per-city layout. Models and results
    # remain at the repo root where NYC's benchmark artifacts already live.
    data_dir=os.path.join(PROJECT_ROOT, "data", "nyc"),
    models_dir=os.path.join(PROJECT_ROOT, "models"),
    results_dir=os.path.join(PROJECT_ROOT, "results"),
)

_PHL_CONFIG = CityConfig(
    city_name="Philadelphia",
    city_code="phl",
    kalshi_ticker="KXHIGHPHIL",
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



_ATL_CONFIG = CityConfig(
    city_name="Atlanta",
    city_code="atl",
    kalshi_ticker="KXHIGHTATL",
    target_station="USW00013874",
    target_station_name="Hartsfield-Jackson Atlanta International Airport",
    target_lat=33.6301,
    target_lon=-84.4418,
    timezone="America/New_York",
    igra_station_id="USM00072215",
    igra_station_name="Peachtree City FFC",
    nwp_lat=33.6301,
    nwp_lon=-84.4418,
    bucket_edges=_NYC_PHL_BUCKET_EDGES.copy(),
    bucket_labels=_NYC_PHL_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 52.0,
        2: 57.0,
        3: 64.0,
        4: 73.0,
        5: 81.0,
        6: 88.0,
        7: 91.0,
        8: 90.0,
        9: 84.0,
        10: 74.0,
        11: 63.0,
        12: 54.0,
    },
    monthly_tmax_std={
        1: 10.0,
        2: 10.0,
        3: 9.5,
        4: 8.5,
        5: 7.5,
        6: 6.0,
        7: 5.0,
        8: 5.0,
        9: 6.5,
        10: 8.0,
        11: 9.0,
        12: 10.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "atlanta"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "atlanta"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "atlanta"),
)


_AUS_CONFIG = CityConfig(
    city_name="Austin",
    city_code="aus",
    kalshi_ticker="KXHIGHAUS",
    target_station="USW00013904",
    target_station_name="Austin-Bergstrom International Airport",
    target_lat=30.1944,
    target_lon=-97.6700,
    timezone="America/Chicago",
    igra_station_id="USM00072254",
    igra_station_name="Del Rio DRT",
    nwp_lat=30.1944,
    nwp_lon=-97.6700,
    bucket_edges=_NYC_PHL_BUCKET_EDGES.copy(),
    bucket_labels=_NYC_PHL_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 62.0,
        2: 67.0,
        3: 74.0,
        4: 80.0,
        5: 87.0,
        6: 93.0,
        7: 96.0,
        8: 97.0,
        9: 90.0,
        10: 82.0,
        11: 71.0,
        12: 63.0,
    },
    monthly_tmax_std={
        1: 9.0,
        2: 9.0,
        3: 9.5,
        4: 8.5,
        5: 7.5,
        6: 6.0,
        7: 4.5,
        8: 4.5,
        9: 6.0,
        10: 8.0,
        11: 8.5,
        12: 9.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "austin"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "austin"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "austin"),
)

# ---------------------------------------------------------------------------
# Expansion cities (Phase 4) — registered only after live-API contract
# verification (results/expansion/contract_verification.json). Each settles on
# the NWS Daily Climatological Report for the listed airport; tickers were
# discovered from the public API (naming is irregular: DC/PHX carry a "T"
# prefix). These ship ASOS-first and start in MONITOR — no city is PROMOTED
# until it has >= 1 full year of real-price backtest passing every gate.
# ---------------------------------------------------------------------------
_DEN_CONFIG = CityConfig(
    city_name="Denver",
    city_code="den",
    kalshi_ticker="KXHIGHDEN",
    target_station="USW00003017",
    target_station_name="Denver International Airport",
    target_lat=39.8466,
    target_lon=-104.6562,
    timezone="America/Denver",
    igra_station_id="USM00072469",
    igra_station_name="Denver",
    nwp_lat=39.8466,
    nwp_lon=-104.6562,
    bucket_edges=_DEN_BUCKET_EDGES.copy(),
    bucket_labels=_DEN_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 45.0, 2: 47.0, 3: 54.0, 4: 61.0, 5: 71.0, 6: 82.0,
        7: 90.0, 8: 88.0, 9: 80.0, 10: 67.0, 11: 53.0, 12: 44.0,
    },
    monthly_tmax_std={
        1: 11.0, 2: 11.0, 3: 11.0, 4: 11.0, 5: 9.5, 6: 8.0,
        7: 6.5, 8: 6.5, 9: 9.0, 10: 10.0, 11: 10.5, 12: 11.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "denver"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "denver"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "denver"),
)

_DC_CONFIG = CityConfig(
    city_name="Washington DC",
    city_code="dc",
    kalshi_ticker="KXHIGHTDC",
    target_station="USW00013743",
    target_station_name="Reagan National Airport (Washington DC)",
    target_lat=38.8472,
    target_lon=-77.0344,
    timezone="America/New_York",
    igra_station_id="USM00072403",
    igra_station_name="Sterling VA / IAD",
    nwp_lat=38.8472,
    nwp_lon=-77.0344,
    bucket_edges=_DC_BUCKET_EDGES.copy(),
    bucket_labels=_DC_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 44.0, 2: 47.0, 3: 56.0, 4: 67.0, 5: 76.0, 6: 85.0,
        7: 89.0, 8: 87.0, 9: 80.0, 10: 68.0, 11: 57.0, 12: 47.0,
    },
    monthly_tmax_std={
        1: 10.0, 2: 10.0, 3: 10.0, 4: 9.5, 5: 8.0, 6: 6.5,
        7: 5.5, 8: 5.5, 9: 7.0, 10: 8.5, 11: 9.0, 12: 10.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "washington_dc"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "washington_dc"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "washington_dc"),
)

_LAX_CONFIG = CityConfig(
    city_name="Los Angeles",
    city_code="lax",
    kalshi_ticker="KXHIGHLAX",
    target_station="USW00023174",
    target_station_name="Los Angeles International Airport",
    target_lat=33.9381,
    target_lon=-118.3889,
    timezone="America/Los_Angeles",
    igra_station_id="USM00072293",
    igra_station_name="San Diego NKX",
    nwp_lat=33.9381,
    nwp_lon=-118.3889,
    bucket_edges=_LAX_BUCKET_EDGES.copy(),
    bucket_labels=_LAX_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 66.0, 2: 66.0, 3: 67.0, 4: 68.0, 5: 70.0, 6: 72.0,
        7: 75.0, 8: 77.0, 9: 77.0, 10: 74.0, 11: 71.0, 12: 66.0,
    },
    monthly_tmax_std={
        1: 6.5, 2: 6.5, 3: 6.0, 4: 6.5, 5: 6.0, 6: 5.5,
        7: 5.0, 8: 5.0, 9: 6.0, 10: 7.0, 11: 7.0, 12: 6.5,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "los_angeles"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "los_angeles"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "los_angeles"),
)

_MIA_CONFIG = CityConfig(
    city_name="Miami",
    city_code="mia",
    kalshi_ticker="KXHIGHMIA",
    target_station="USW00012839",
    target_station_name="Miami International Airport",
    target_lat=25.7906,
    target_lon=-80.3164,
    timezone="America/New_York",
    igra_station_id="USM00072202",
    igra_station_name="Miami FL",
    nwp_lat=25.7906,
    nwp_lon=-80.3164,
    bucket_edges=_MIA_BUCKET_EDGES.copy(),
    bucket_labels=_MIA_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 76.0, 2: 78.0, 3: 80.0, 4: 83.0, 5: 87.0, 6: 89.0,
        7: 91.0, 8: 91.0, 9: 89.0, 10: 86.0, 11: 82.0, 12: 78.0,
    },
    monthly_tmax_std={
        1: 5.5, 2: 5.5, 3: 5.0, 4: 4.5, 5: 4.0, 6: 3.5,
        7: 3.0, 8: 3.0, 9: 3.5, 10: 4.5, 11: 5.0, 12: 5.5,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "miami"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "miami"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "miami"),
)

_PHX_CONFIG = CityConfig(
    city_name="Phoenix",
    city_code="phx",
    kalshi_ticker="KXHIGHTPHX",
    target_station="USW00023183",
    target_station_name="Phoenix Sky Harbor International Airport",
    target_lat=33.4278,
    target_lon=-112.0037,
    timezone="America/Phoenix",
    igra_station_id="USM00072274",
    igra_station_name="Tucson AZ",
    nwp_lat=33.4278,
    nwp_lon=-112.0037,
    bucket_edges=_PHX_BUCKET_EDGES.copy(),
    bucket_labels=_PHX_BUCKET_LABELS.copy(),
    monthly_tmax_mean={
        1: 67.0, 2: 71.0, 3: 77.0, 4: 85.0, 5: 95.0, 6: 104.0,
        7: 106.0, 8: 104.0, 9: 100.0, 10: 89.0, 11: 76.0, 12: 66.0,
    },
    monthly_tmax_std={
        1: 7.0, 2: 7.5, 3: 8.0, 4: 8.5, 5: 7.5, 6: 6.5,
        7: 5.5, 8: 5.5, 9: 7.0, 10: 8.0, 11: 7.5, 12: 7.0,
    },
    data_dir=os.path.join(PROJECT_ROOT, "data", "phoenix"),
    models_dir=os.path.join(PROJECT_ROOT, "models", "phoenix"),
    results_dir=os.path.join(PROJECT_ROOT, "results", "phoenix"),
)


# ---------------------------------------------------------------------------
# City registry
# ---------------------------------------------------------------------------
_CITY_REGISTRY: Dict[str, CityConfig] = {
    "nyc": _NYC_CONFIG,
    "phl": _PHL_CONFIG,
    "chi": _CHI_CONFIG,
    "atl": _ATL_CONFIG,
    "aus": _AUS_CONFIG,
    # Phase 4 expansion (MONITOR until >= 1yr real-price backtest)
    "den": _DEN_CONFIG,
    "dc": _DC_CONFIG,
    "lax": _LAX_CONFIG,
    "mia": _MIA_CONFIG,
    "phx": _PHX_CONFIG,
}




# ---------------------------------------------------------------------------
# Runtime metadata hydration from consolidated city runtime data
# ---------------------------------------------------------------------------

def _hydrate_runtime_metadata(cfg: CityConfig) -> None:
    runtime = CITY_RUNTIME_DATA.get(cfg.city_code, {})
    cfg.all_stations = dict(runtime.get("ALL_STATIONS", {}))
    cfg.surrounding_stations = dict(runtime.get("SURROUNDING_STATIONS", {}))
    cfg.asos_station_map = dict(runtime.get("ASOS_STATION_MAP", {}))
    cfg.station_metadata = dict(runtime.get("STATION_METADATA", {}))
    cfg.station_rings = dict(runtime.get("STATION_RINGS", {}))
    cfg.station_sectors = dict(runtime.get("STATION_SECTORS", {}))
    cfg.meteorological_sectors = dict(runtime.get("METEOROLOGICAL_SECTORS", {}))
    cfg.start_date = runtime.get("START_DATE", "")
    cfg.end_date = runtime.get("END_DATE", "")
    cfg.min_completeness = float(runtime.get("MIN_COMPLETENESS", 0.8))
    cfg.train_ratio = float(runtime.get("TRAIN_RATIO", 0.7))
    cfg.val_ratio = float(runtime.get("VAL_RATIO", 0.15))
    cfg.input_variables = list(runtime.get("INPUT_VARIABLES", ["TMAX", "TMIN"]))
    cfg.max_forward_fill_days = int(runtime.get("MAX_FORWARD_FILL_DAYS", 7))
    cfg.batch_size = int(runtime.get("BATCH_SIZE", 32))

    cfg.observation_anchor_stations = [cfg.target_station]
    cfg.operational_sources = ["ghcn_daily", "asos", "nwp", "igra", "kalshi"]


for _cfg in _CITY_REGISTRY.values():
    _hydrate_runtime_metadata(_cfg)


# The auto-generated runtime snapshot was captured in a container whose
# repo root was /workspace/weather-prediction; rewrite that stale prefix to
# the actual repo root so absolute paths work in any checkout location.
_STALE_ROOT_PREFIX = "/workspace/weather-prediction"
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _portable_path(value):
    if isinstance(value, str) and value.startswith(_STALE_ROOT_PREFIX):
        return _REPO_ROOT + value[len(_STALE_ROOT_PREFIX):]
    return value


def get_city_runtime_config(city_code: str) -> SimpleNamespace:
    """Compatibility runtime namespace used by legacy scripts/modules."""
    cfg = get_city_config(city_code)
    exports = {
        key: _portable_path(value)
        for key, value in CITY_RUNTIME_DATA.get(cfg.city_code, {}).items()
    }
    exports.update(
        {
            "TARGET_STATION": cfg.target_station,
            "TARGET_LAT": cfg.target_lat,
            "TARGET_LON": cfg.target_lon,
            "START_DATE": cfg.start_date,
            "END_DATE": cfg.end_date,
            "ALL_STATIONS": cfg.all_stations,
            "SURROUNDING_STATIONS": cfg.surrounding_stations,
            "ASOS_STATION_MAP": cfg.asos_station_map,
            "STATION_METADATA": cfg.station_metadata,
            "STATION_RINGS": cfg.station_rings,
            "STATION_SECTORS": cfg.station_sectors,
            "METEOROLOGICAL_SECTORS": cfg.meteorological_sectors,
            "MIN_COMPLETENESS": cfg.min_completeness,
            "TRAIN_RATIO": cfg.train_ratio,
            "VAL_RATIO": cfg.val_ratio,
            "INPUT_VARIABLES": cfg.input_variables,
            "MAX_FORWARD_FILL_DAYS": cfg.max_forward_fill_days,
            "BATCH_SIZE": cfg.batch_size,
        }
    )
    return SimpleNamespace(**exports)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_city_config(city_code: str) -> CityConfig:
    """Return the CityConfig for a given city code.

    Args:
        city_code: Lowercase city identifier (e.g., "nyc", "phl", "chi", "atl", "aus").

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
