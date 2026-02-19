"""
Multi-City Operational Data Integration for ASOS, NWP, and Soundings.

Provides city-aware wrappers around the existing ASOS, NWP, and soundings
collection modules to support Chicago (KORD network) and Philadelphia
(KPHL network) in addition to NYC.

ASOS stations:
  - CHI: KORD + 45 surrounding stations from config_chicago.ASOS_STATION_MAP
  - PHL: KPHL + 50 surrounding stations from config_philadelphia.ASOS_STATION_MAP

NWP grid points:
  - CHI: 41.97°N, 87.91°W (O'Hare)
  - PHL: 39.87°N, 75.23°W (PHL International)

IGRA soundings:
  - CHI: Davenport (DVN) → USM00074455
  - PHL: Sterling (IAD) → USM00072403

Usage:
    from src.operational_data import get_operational_config, verify_asos_coverage
    cfg = get_operational_config("chi")
    verify_asos_coverage("chi")
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, CityConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operational data configuration
# ---------------------------------------------------------------------------

@dataclass
class OperationalDataConfig:
    """City-specific operational data source configuration.

    Combines ASOS station mappings, NWP grid points, and IGRA sounding
    station info into a single configuration object.

    Attributes
    ----------
    city_code : str
        City identifier.
    city_config : CityConfig
        Full city configuration from the registry.
    asos_stations : dict
        Mapping of GHCN station ID → ICAO code for ASOS-equipped stations.
    primary_asos : str
        ICAO code for the primary (target) ASOS station.
    nwp_lat : float
        NWP grid point latitude.
    nwp_lon : float
        NWP grid point longitude.
    igra_station_id : str
        IGRA upper-air station identifier.
    igra_station_name : str
        Human-readable name of the IGRA station.
    asos_data_dir : str
        Directory for ASOS hourly data files.
    nwp_data_dir : str
        Directory for NWP forecast data files.
    soundings_data_dir : str
        Directory for IGRA sounding data files.
    """

    city_code: str
    city_config: CityConfig
    asos_stations: Dict[str, str] = field(default_factory=dict)
    primary_asos: str = ""
    nwp_lat: float = 0.0
    nwp_lon: float = 0.0
    igra_station_id: str = ""
    igra_station_name: str = ""
    asos_data_dir: str = ""
    nwp_data_dir: str = ""
    soundings_data_dir: str = ""


def _get_city_config_module(city_code: str):
    """Dynamically load the per-city config module.

    Parameters
    ----------
    city_code : str
        City identifier (chi, phl, atl, aus, nyc).

    Returns
    -------
    module or None
        The imported config module, or None on failure.
    """
    import importlib

    _CONFIG_MODULES = {
        "chi": "config_chicago",
        "phl": "config_philadelphia",
        "atl": "config_atlanta",
        "aus": "config_austin",
    }

    try:
        if city_code in _CONFIG_MODULES:
            return importlib.import_module(_CONFIG_MODULES[city_code])
        elif city_code == "nyc":
            try:
                return importlib.import_module("config_expanded")
            except ImportError:
                return importlib.import_module("config")
        else:
            logger.warning("Unknown city code: %s", city_code)
            return None
    except ImportError:
        logger.warning("Could not import config for %s", city_code)
        return None


def _load_asos_map(city_code: str) -> Dict[str, str]:
    """Load ASOS station map for a city from its config module.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    dict
        GHCN station ID → ICAO code mapping.
    """
    cfg = _get_city_config_module(city_code)
    if cfg is not None and hasattr(cfg, "ASOS_STATION_MAP"):
        return dict(cfg.ASOS_STATION_MAP)
    return {}


# Primary ASOS stations per city
_PRIMARY_ASOS = {
    "nyc": "KJFK",
    "chi": "KORD",
    "phl": "KPHL",
}


def get_operational_config(city_code: str) -> OperationalDataConfig:
    """Build the operational data configuration for a city.

    Parameters
    ----------
    city_code : str
        City identifier ("nyc", "chi", "phl").

    Returns
    -------
    OperationalDataConfig
        Complete operational data configuration.
    """
    code = city_code.strip().lower()
    city_cfg = get_city_config(code)
    asos_map = _load_asos_map(code)

    config = OperationalDataConfig(
        city_code=code,
        city_config=city_cfg,
        asos_stations=asos_map,
        primary_asos=_PRIMARY_ASOS.get(code, ""),
        nwp_lat=city_cfg.nwp_lat,
        nwp_lon=city_cfg.nwp_lon,
        igra_station_id=city_cfg.igra_station_id,
        igra_station_name=city_cfg.igra_station_name,
        asos_data_dir=os.path.join(city_cfg.data_dir, "asos"),
        nwp_data_dir=os.path.join(city_cfg.data_dir, "nwp"),
        soundings_data_dir=os.path.join(city_cfg.data_dir, "soundings"),
    )

    logger.info(
        "Operational config for %s: %d ASOS stations, primary=%s, "
        "NWP=(%.2f, %.2f), IGRA=%s",
        code, len(asos_map), config.primary_asos,
        config.nwp_lat, config.nwp_lon, config.igra_station_id,
    )
    return config


# ---------------------------------------------------------------------------
# ASOS verification
# ---------------------------------------------------------------------------

def verify_asos_coverage(city_code: str) -> Dict[str, any]:
    """Verify ASOS hourly data availability for a city's station network.

    Checks which stations have ASOS data files in the expected directory.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    dict
        Verification report with keys:
        - "total_stations": number of stations with ASOS mapping
        - "data_available": stations with existing data files
        - "data_missing": stations without data files
        - "coverage_pct": percentage of stations with data
    """
    config = get_operational_config(city_code)
    asos_dir = config.asos_data_dir

    available = []
    missing = []

    for ghcn_id, icao in config.asos_stations.items():
        # Check for common ASOS data file patterns
        found = False
        if os.path.exists(asos_dir):
            for fname in os.listdir(asos_dir):
                if icao.lower() in fname.lower():
                    found = True
                    break

        if found:
            available.append({"ghcn_id": ghcn_id, "icao": icao})
        else:
            missing.append({"ghcn_id": ghcn_id, "icao": icao})

    total = len(config.asos_stations)
    coverage = len(available) / total * 100 if total > 0 else 0.0

    report = {
        "city_code": city_code,
        "total_stations": total,
        "data_available": available,
        "data_missing": missing,
        "n_available": len(available),
        "n_missing": len(missing),
        "coverage_pct": coverage,
    }

    logger.info(
        "ASOS coverage for %s: %d/%d stations (%.1f%%)",
        city_code, len(available), total, coverage,
    )
    return report


# ---------------------------------------------------------------------------
# NWP grid extraction config
# ---------------------------------------------------------------------------

# Standard NWP variables for temperature prediction
NWP_VARIABLES = [
    "tmax_2m",      # 2m max temperature
    "tmp_850",      # Temperature at 850 mb
    "ugrd_10m",     # U-wind at 10m
    "vgrd_10m",     # V-wind at 10m
    "tcdc_eatm",    # Total cloud cover
    "mslp",         # Mean sea-level pressure
    "apcp",         # Accumulated precipitation
]


def get_nwp_config(city_code: str) -> Dict[str, any]:
    """Get NWP data extraction configuration for a city.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    dict
        NWP configuration including grid point, variables, and paths.
    """
    config = get_operational_config(city_code)

    return {
        "city_code": city_code,
        "grid_lat": config.nwp_lat,
        "grid_lon": config.nwp_lon,
        "variables": NWP_VARIABLES,
        "output_dir": config.nwp_data_dir,
        "models": ["gfs", "nam"],
        "gfs_resolution": "0p25",  # 0.25° GFS
        "nam_resolution": "12km",   # 12 km NAM
        "forecast_hours": [0, 6, 12, 18, 24, 30, 36, 42, 48],
    }


# ---------------------------------------------------------------------------
# IGRA sounding config
# ---------------------------------------------------------------------------

# Standard IGRA sounding levels for temperature prediction
SOUNDING_LEVELS_MB = [1000, 925, 850, 700, 500, 300, 250]


def get_sounding_config(city_code: str) -> Dict[str, any]:
    """Get IGRA sounding configuration for a city.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    dict
        Sounding configuration including station, levels, and paths.
    """
    config = get_operational_config(city_code)

    return {
        "city_code": city_code,
        "station_id": config.igra_station_id,
        "station_name": config.igra_station_name,
        "levels_mb": SOUNDING_LEVELS_MB,
        "hours": [0, 12],  # 00Z and 12Z soundings
        "output_dir": config.soundings_data_dir,
        "derived_features": [
            "t850",             # Temperature at 850 mb
            "t500",             # Temperature at 500 mb
            "wind_dir_850",     # Wind direction at 850 mb
            "wind_speed_850",   # Wind speed at 850 mb
            "lapse_rate_850_500",  # Lapse rate between 850 and 500 mb
            "stability_index",  # T500 - Tdew500
        ],
    }


# ---------------------------------------------------------------------------
# NWP-derived feature definitions
# ---------------------------------------------------------------------------

def get_nwp_feature_names(city_code: str) -> List[str]:
    """Get the list of NWP-derived feature names for a city.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    list of str
        Feature names that would be added to the preprocessing pipeline.
    """
    prefix = city_code.upper()
    return [
        f"nwp_tmax_2m",
        f"nwp_t850",
        f"nwp_wind_u10",
        f"nwp_wind_v10",
        f"nwp_wind_speed",
        f"nwp_wind_dir",
        f"nwp_cloud_cover",
        f"nwp_mslp",
        f"nwp_precip",
        f"nwp_tmax_bias",        # NWP TMAX - station climatology
        f"nwp_model_spread",     # GFS - NAM difference
        f"sounding_t850",
        f"sounding_t500",
        f"sounding_wind_dir_850",
        f"sounding_wind_speed_850",
        f"sounding_lapse_rate",
        f"sounding_stability",
    ]


# ---------------------------------------------------------------------------
# Data availability summary
# ---------------------------------------------------------------------------

def get_data_availability_summary(city_code: str) -> Dict[str, any]:
    """Get a summary of all operational data availability for a city.

    Parameters
    ----------
    city_code : str
        City identifier.

    Returns
    -------
    dict
        Summary of ASOS, NWP, and sounding data availability.
    """
    config = get_operational_config(city_code)

    # Check ASOS
    asos_available = os.path.exists(config.asos_data_dir)
    asos_files = (
        len(os.listdir(config.asos_data_dir))
        if asos_available else 0
    )

    # Check NWP
    nwp_available = os.path.exists(config.nwp_data_dir)
    nwp_files = (
        len(os.listdir(config.nwp_data_dir))
        if nwp_available else 0
    )

    # Check soundings
    soundings_available = os.path.exists(config.soundings_data_dir)
    soundings_files = (
        len(os.listdir(config.soundings_data_dir))
        if soundings_available else 0
    )

    return {
        "city_code": city_code,
        "asos": {
            "dir_exists": asos_available,
            "n_files": asos_files,
            "n_stations_mapped": len(config.asos_stations),
            "primary_station": config.primary_asos,
        },
        "nwp": {
            "dir_exists": nwp_available,
            "n_files": nwp_files,
            "grid_point": (config.nwp_lat, config.nwp_lon),
        },
        "soundings": {
            "dir_exists": soundings_available,
            "n_files": soundings_files,
            "station_id": config.igra_station_id,
            "station_name": config.igra_station_name,
        },
    }
