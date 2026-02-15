"""
Expanded Configuration for Philadelphia Temperature Prediction.

Station network centered on Philadelphia International Airport (USW00013739).
Organized by distance ring and compass sector from PHL.

Station selection redesigned 2026-02-15 for balanced directional coverage.
Previous config overweighted NE (19/48 stations in NYC corridor). New network
uses 50 USW-class stations with proportional sector coverage:
    N:5, NE:7, E:2, SE:2, S:10, SW:11, W:7, NW:6

NE curated to 7 well-spaced stations (cut redundant JFK/LGA/Teterboro/
Westchester cluster). S/SW/W/NW expanded to capture Chesapeake warm advection,
Appalachian cold-air advection, and Virginia coastal signals.

Ring Classification (distance from PHL):
    Ring1_Near:      0 - 50 miles
    Ring2_Regional:  50 - 100 miles
    Ring3_Extended:  100 - 150 miles
    Ring4_Far:       150 - 250 miles

Sector Classification (compass bearing from PHL):
    N:  337.5 - 22.5 deg
    NE: 22.5 - 67.5 deg
    E:  67.5 - 112.5 deg
    SE: 112.5 - 157.5 deg
    S:  157.5 - 202.5 deg
    SW: 202.5 - 247.5 deg
    W:  247.5 - 292.5 deg
    NW: 292.5 - 337.5 deg
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.city_config import get_city_config

# ==============================================================================
# City Configuration
# ==============================================================================
CITY_CONFIG = get_city_config("phl")

# Target station
TARGET_STATION = CITY_CONFIG.target_station
TARGET_LAT = CITY_CONFIG.target_lat
TARGET_LON = CITY_CONFIG.target_lon
TARGET_VARIABLE = "TMAX"

# ==============================================================================
# Surrounding Stations (50 stations)
# ==============================================================================
# Format: {station_id: "Name, State (distance sector)"}
# All stations are USW-class with >= 80% TMAX completeness 1985-2024.
# Selection balances directional coverage; NE capped at 7 to avoid NYC bias.
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 7 stations
    "USW00094732": "NE PHILA AP, PA (18mi NE)",
    "USW00013781": "WILMINGTON NEW CASTLE CO AP, DE (24mi SW)",
    "USW00014706": "MCGUIRE AFB, NJ (35mi E)",
    "USW00014792": "TRENTON-MERCER AP, NJ (35mi NE)",
    "USW00013735": "MILLVILLE MUNI AP, NJ (36mi S)",
    "USW00093730": "ATLANTIC CITY INTL AP, NJ (46mi SE)",
    "USW00014780": "LAKEHURST AP, NJ (50mi E)",
    # --- Ring 2: Regional (50-100 mi) --- 13 stations
    "USW00014712": "READING RGNL AP, PA (52mi NW)",
    "USW00013707": "DOVER AFB, DE (53mi S)",
    "USW00013724": "ATLANTIC CITY MARINA, NJ (55mi SE)",
    "USW00014737": "ALLENTOWN LEHIGH VLY INTL AP, PA (55mi N)",
    "USW00013701": "ABERDEEN PHILLIPS FLD, MD (57mi SW)",
    "USW00014734": "NEWARK LIBERTY INTL AP, NJ (79mi NE)",
    "USW00013764": "GEORGETOWN-DELAWARE COASTAL AP, DE (82mi S)",
    "USW00014711": "HARRISBURG INTL AP, PA (85mi W)",
    "USW00014751": "HARRISBURG CAPITAL CITY AP, PA (89mi W)",
    "USW00013752": "ANNAPOLIS US NAVAL ACADEMY, MD (91mi SW)",
    "USW00094728": "CENTRAL PARK, NY (91mi NE)",
    "USW00093721": "BALTIMORE-WASHINGTON INTL AP, MD (92mi SW)",
    "USW00093733": "FT MEADE TPTN AAF, MD (99mi SW)",
    # --- Ring 3: Extended (100-150 mi) --- 14 stations
    "USW00014777": "WILKES-BARRE/SCRANTON INTL AP, PA (104mi N)",
    "USW00093720": "SALISBURY-WICOMICO RGNL AP, MD (107mi S)",
    "USW00013705": "ANDREWS AFB, MD (114mi SW)",
    "USW00013743": "REAGAN NATIONAL AP, VA (120mi SW)",
    "USW00013730": "FREDERICK, MD (123mi W)",
    "USW00013721": "PATUXENT RIVER NAS, MD (126mi SW)",
    "USW00004781": "ISLIP-LI MACARTHUR AP, NY (129mi NE)",
    "USW00014778": "WILLIAMSPORT RGNL AP, PA (130mi NW)",
    "USW00093728": "DAVISON AAF, VA (132mi SW)",
    "USW00093739": "WALLOPS ISLAND, VA (134mi S)",
    "USW00093738": "DULLES INTL AP, VA (135mi SW)",
    "USW00094702": "SIKORSKY MEM AP, CT (142mi NE)",
    "USW00013773": "QUANTICO MCAS, VA (146mi SW)",
    "USW00013734": "MARTINSBURG, WV (150mi W)",
    # --- Ring 4: Far (150-250 mi) --- 16 stations
    "USW00004725": "BINGHAMTON, NY (166mi N)",
    "USW00014736": "ALTOONA BLAIR CO AP, PA (166mi W)",
    "USW00013731": "FRONT ROYAL, VA (171mi W)",
    "USW00014748": "ELMIRA CORNING RGNL AP, NY (180mi NW)",
    "USW00014740": "HARTFORD-BRADLEY INTL AP, CT (195mi NE)",
    "USW00013702": "LANGLEY AFB, VA (202mi S)",
    "USW00093735": "FT EUSTIS FELKER AAF, VA (203mi S)",
    "USW00013750": "NORFOLK NAS, VA (211mi S)",
    "USW00013737": "NORFOLK INTL AP, VA (212mi S)",
    "USW00014735": "ALBANY INTL AP, NY (212mi N)",
    "USW00004787": "DUBOIS RGNL AP, PA (213mi NW)",
    "USW00013769": "OCEANA NAS, VA (216mi S)",
    "USW00004751": "BRADFORD RGNL AP, PA (222mi NW)",
    "USW00094704": "DANSVILLE MUNI AP, NY (227mi NW)",
    "USW00014771": "SYRACUSE HANCOCK INTL AP, NY (228mi N)",
    "USW00013736": "MORGANTOWN MUNI AP, WV (250mi W)",
}

# All stations (target + expanded surrounding) for convenience
ALL_STATIONS = {
    TARGET_STATION: "Philadelphia International Airport (Target)",
    **SURROUNDING_STATIONS,
}

# ==============================================================================
# ASOS/AWOS Mapping (operational station IDs)
# ==============================================================================
# Mapping of GHCN station IDs to ICAO codes for stations with ASOS/AWOS data.
# Stations without operational ASOS/AWOS are excluded and listed separately.
ASOS_STATION_MAP = {
    "USW00013739": "KPHL",
    "USW00094732": "KPNE",
    "USW00013781": "KILG",
    "USW00014706": "KWRI",
    "USW00014792": "KTTN",
    "USW00013735": "KMIV",
    "USW00093730": "KACY",
    "USW00014780": "KNEL",
    "USW00014712": "KRDG",
    "USW00013707": "KDOV",
    "USW00014737": "KABE",
    "USW00013701": "KAPG",
    "USW00014734": "KEWR",
    "USW00013764": "KGED",
    "USW00014711": "KMDT",
    "USW00014751": "KCXY",
    "USW00094728": "KNYC",
    "USW00093721": "KBWI",
    "USW00093733": "KFME",
    "USW00014777": "KAVP",
    "USW00093720": "KSBY",
    "USW00013705": "KADW",
    "USW00013743": "KDCA",
    "USW00013730": "KFDK",
    "USW00013721": "KNHK",
    "USW00004781": "KISP",
    "USW00014778": "KIPT",
    "USW00093728": "KDAA",
    "USW00093739": "KWAL",
    "USW00093738": "KIAD",
    "USW00094702": "KBDR",
    "USW00013773": "KNYG",
    "USW00013734": "KMRB",
    "USW00004725": "KBGM",
    "USW00014736": "KAOO",
    "USW00014748": "KELM",
    "USW00014740": "KBDL",
    "USW00013702": "KLFI",
    "USW00093735": "KFAF",
    "USW00013750": "KNGU",
    "USW00013737": "KORF",
    "USW00014735": "KALB",
    "USW00004787": "KDUJ",
    "USW00013769": "KNTU",
    "USW00004751": "KBFD",
    "USW00094704": "KDSV",
    "USW00014771": "KSYR",
    "USW00013736": "KMGW",
}

NON_ASOS_STATIONS = {
    "USW00013724": "ATLANTIC CITY MARINA (non-airport coastal station)",
    "USW00013752": "ANNAPOLIS US NAVAL ACADEMY (non-standard ASOS)",
    "USW00013731": "FRONT ROYAL (non-airport station)",
}

# ==============================================================================
# Data Quality (slightly relaxed for expanded set)
# ==============================================================================
MIN_COMPLETENESS = 0.80

# ==============================================================================
# Station Rings (distance classification)
# ==============================================================================
STATION_RINGS = {
    "Ring1_Near": [
        "USW00094732",
        "USW00013781",
        "USW00014706",
        "USW00014792",
        "USW00013735",
        "USW00093730",
        "USW00014780",
    ],
    "Ring2_Regional": [
        "USW00014712",
        "USW00013707",
        "USW00013724",
        "USW00014737",
        "USW00013701",
        "USW00014734",
        "USW00013764",
        "USW00014711",
        "USW00014751",
        "USW00013752",
        "USW00094728",
        "USW00093721",
        "USW00093733",
    ],
    "Ring3_Extended": [
        "USW00014777",
        "USW00093720",
        "USW00013705",
        "USW00013743",
        "USW00013730",
        "USW00013721",
        "USW00004781",
        "USW00014778",
        "USW00093728",
        "USW00093739",
        "USW00093738",
        "USW00094702",
        "USW00013773",
        "USW00013734",
    ],
    "Ring4_Far": [
        "USW00004725",
        "USW00014736",
        "USW00013731",
        "USW00014748",
        "USW00014740",
        "USW00013702",
        "USW00093735",
        "USW00013750",
        "USW00013737",
        "USW00014735",
        "USW00004787",
        "USW00013769",
        "USW00004751",
        "USW00094704",
        "USW00014771",
        "USW00013736",
    ],
}

# ==============================================================================
# Station Sectors (compass direction classification)
# ==============================================================================
STATION_SECTORS = {
    "N": [
        "USW00014737",
        "USW00014777",
        "USW00004725",
        "USW00014735",
        "USW00014771",
    ],
    "NE": [
        "USW00094732",
        "USW00014792",
        "USW00014734",
        "USW00094728",
        "USW00004781",
        "USW00094702",
        "USW00014740",
    ],
    "E": [
        "USW00014706",
        "USW00014780",
    ],
    "SE": [
        "USW00093730",
        "USW00013724",
    ],
    "S": [
        "USW00013735",
        "USW00013707",
        "USW00013764",
        "USW00093720",
        "USW00093739",
        "USW00013702",
        "USW00093735",
        "USW00013750",
        "USW00013737",
        "USW00013769",
    ],
    "SW": [
        "USW00013781",
        "USW00013701",
        "USW00013752",
        "USW00093721",
        "USW00093733",
        "USW00013705",
        "USW00013743",
        "USW00013721",
        "USW00093728",
        "USW00093738",
        "USW00013773",
    ],
    "W": [
        "USW00014711",
        "USW00014751",
        "USW00013730",
        "USW00013734",
        "USW00014736",
        "USW00013731",
        "USW00013736",
    ],
    "NW": [
        "USW00014712",
        "USW00014778",
        "USW00014748",
        "USW00004787",
        "USW00004751",
        "USW00094704",
    ],
}

# ==============================================================================
# Meteorological Sector Assignments (for feature engineering)
# ==============================================================================
# These group stations by meteorological relevance rather than pure compass direction.
# Philadelphia's weather is driven by:
#   - Cold-air advection from W/NW (Appalachian outbreaks)
#   - Warm advection from S/SW (Chesapeake/Gulf moisture)
#   - Atlantic moderation from E/SE (coastal marine influence)
#   - NYC corridor signals from NE (correlated mid-Atlantic)
METEOROLOGICAL_SECTORS = {
    "WNW": (  # Upstream cold-air advection: W and NW sectors (Appalachian stations)
        STATION_SECTORS["W"] + STATION_SECTORS["NW"]
    ),
    "SW": (  # Warm advection from south: S and SW sectors (Chesapeake/mid-Atlantic/Virginia)
        STATION_SECTORS["S"] + STATION_SECTORS["SW"]
    ),
    "Coastal": (  # Atlantic moderation: E and SE sectors
        STATION_SECTORS["E"] + STATION_SECTORS["SE"]
    ),
    "NearField": (  # Urban/local: all Ring 1 stations
        STATION_RINGS["Ring1_Near"]
    ),
    "NE": (  # NYC corridor influence: N and NE sectors
        STATION_SECTORS["N"] + STATION_SECTORS["NE"]
    ),
}

# ==============================================================================
# Station Metadata
# ==============================================================================
# Full metadata for each surrounding station, sourced from the GHCN inventory.
# Distances and bearings computed via haversine from PHL (39.8733, -75.2269).
STATION_METADATA = {
    # --- Ring 1: Near-field (0-50 mi) --- 7 stations
    "USW00094732": {
        "name": "NE PHILA AP",
        "state": "PA",
        "lat": 40.0789,
        "lon": -75.0133,
        "distance_mi": 18.16,
        "bearing": 38.46,
        "ring": "Ring1_Near",
        "sector": "NE",
    },
    "USW00013781": {
        "name": "WILMINGTON NEW CASTLE CO AP",
        "state": "DE",
        "lat": 39.6786,
        "lon": -75.6064,
        "distance_mi": 24.42,
        "bearing": 235.81,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USW00014706": {
        "name": "MCGUIRE AFB",
        "state": "NJ",
        "lat": 40.0156,
        "lon": -74.5914,
        "distance_mi": 34.71,
        "bearing": 73.23,
        "ring": "Ring1_Near",
        "sector": "E",
    },
    "USW00014792": {
        "name": "TRENTON-MERCER AP",
        "state": "NJ",
        "lat": 40.2769,
        "lon": -74.8158,
        "distance_mi": 35.36,
        "bearing": 37.80,
        "ring": "Ring1_Near",
        "sector": "NE",
    },
    "USW00013735": {
        "name": "MILLVILLE MUNI AP",
        "state": "NJ",
        "lat": 39.3678,
        "lon": -75.0722,
        "distance_mi": 35.87,
        "bearing": 167.18,
        "ring": "Ring1_Near",
        "sector": "S",
    },
    "USW00093730": {
        "name": "ATLANTIC CITY INTL AP",
        "state": "NJ",
        "lat": 39.4519,
        "lon": -74.5669,
        "distance_mi": 45.61,
        "bearing": 129.46,
        "ring": "Ring1_Near",
        "sector": "SE",
    },
    "USW00014780": {
        "name": "LAKEHURST AP",
        "state": "NJ",
        "lat": 40.0333,
        "lon": -74.3500,
        "distance_mi": 49.52,
        "bearing": 76.77,
        "ring": "Ring1_Near",
        "sector": "E",
    },
    # --- Ring 2: Regional (50-100 mi) --- 13 stations
    "USW00014712": {
        "name": "READING RGNL AP",
        "state": "PA",
        "lat": 40.3733,
        "lon": -75.9592,
        "distance_mi": 51.87,
        "bearing": 312.00,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USW00013707": {
        "name": "DOVER AFB",
        "state": "DE",
        "lat": 39.1301,
        "lon": -75.4660,
        "distance_mi": 52.91,
        "bearing": 194.02,
        "ring": "Ring2_Regional",
        "sector": "S",
    },
    "USW00013724": {
        "name": "ATLANTIC CITY MARINA",
        "state": "NJ",
        "lat": 39.3778,
        "lon": -74.4236,
        "distance_mi": 54.77,
        "bearing": 128.43,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USW00014737": {
        "name": "ALLENTOWN LEHIGH VLY INTL AP",
        "state": "PA",
        "lat": 40.6497,
        "lon": -75.4478,
        "distance_mi": 54.89,
        "bearing": 347.82,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00013701": {
        "name": "ABERDEEN PHILLIPS FLD",
        "state": "MD",
        "lat": 39.4664,
        "lon": -76.1686,
        "distance_mi": 57.30,
        "bearing": 241.32,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00014734": {
        "name": "NEWARK LIBERTY INTL AP",
        "state": "NJ",
        "lat": 40.6828,
        "lon": -74.1692,
        "distance_mi": 78.97,
        "bearing": 44.57,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00013764": {
        "name": "GEORGETOWN-DELAWARE COASTAL AP",
        "state": "DE",
        "lat": 38.6889,
        "lon": -75.3611,
        "distance_mi": 82.14,
        "bearing": 185.10,
        "ring": "Ring2_Regional",
        "sector": "S",
    },
    "USW00014711": {
        "name": "HARRISBURG INTL AP",
        "state": "PA",
        "lat": 40.1953,
        "lon": -76.7633,
        "distance_mi": 84.82,
        "bearing": 285.84,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00014751": {
        "name": "HARRISBURG CAPITAL CITY AP",
        "state": "PA",
        "lat": 40.2175,
        "lon": -76.8514,
        "distance_mi": 89.39,
        "bearing": 285.97,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00013752": {
        "name": "ANNAPOLIS US NAVAL ACADEMY",
        "state": "MD",
        "lat": 38.9861,
        "lon": -76.4833,
        "distance_mi": 90.78,
        "bearing": 228.26,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00094728": {
        "name": "CENTRAL PARK",
        "state": "NY",
        "lat": 40.7789,
        "lon": -73.9692,
        "distance_mi": 91.13,
        "bearing": 46.23,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00093721": {
        "name": "BALTIMORE-WASHINGTON INTL AP",
        "state": "MD",
        "lat": 39.1703,
        "lon": -76.6800,
        "distance_mi": 91.42,
        "bearing": 238.37,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00093733": {
        "name": "FT MEADE TPTN AAF",
        "state": "MD",
        "lat": 39.0850,
        "lon": -76.7694,
        "distance_mi": 98.62,
        "bearing": 236.89,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    # --- Ring 3: Extended (100-150 mi) --- 14 stations
    "USW00014777": {
        "name": "WILKES-BARRE/SCRANTON INTL AP",
        "state": "PA",
        "lat": 41.3336,
        "lon": -75.7228,
        "distance_mi": 104.20,
        "bearing": 345.70,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00093720": {
        "name": "SALISBURY-WICOMICO RGNL AP",
        "state": "MD",
        "lat": 38.3392,
        "lon": -75.5106,
        "distance_mi": 107.08,
        "bearing": 188.26,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00013705": {
        "name": "ANDREWS AFB",
        "state": "MD",
        "lat": 38.8108,
        "lon": -76.8669,
        "distance_mi": 113.98,
        "bearing": 230.74,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00013743": {
        "name": "REAGAN NATIONAL AP",
        "state": "VA",
        "lat": 38.8481,
        "lon": -77.0344,
        "distance_mi": 119.75,
        "bearing": 234.31,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00013730": {
        "name": "FREDERICK",
        "state": "MD",
        "lat": 39.4178,
        "lon": -77.3744,
        "distance_mi": 123.29,
        "bearing": 255.90,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00013721": {
        "name": "PATUXENT RIVER NAS",
        "state": "MD",
        "lat": 38.2861,
        "lon": -76.4117,
        "distance_mi": 126.03,
        "bearing": 210.83,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00004781": {
        "name": "ISLIP-LI MACARTHUR AP",
        "state": "NY",
        "lat": 40.7939,
        "lon": -73.1019,
        "distance_mi": 128.73,
        "bearing": 59.70,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00014778": {
        "name": "WILLIAMSPORT RGNL AP",
        "state": "PA",
        "lat": 41.2431,
        "lon": -76.9219,
        "distance_mi": 129.90,
        "bearing": 317.32,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    "USW00093728": {
        "name": "DAVISON AAF",
        "state": "VA",
        "lat": 38.7153,
        "lon": -77.1803,
        "distance_mi": 131.63,
        "bearing": 233.17,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00093739": {
        "name": "WALLOPS ISLAND",
        "state": "VA",
        "lat": 37.9408,
        "lon": -75.4664,
        "distance_mi": 134.14,
        "bearing": 185.58,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00093738": {
        "name": "DULLES INTL AP",
        "state": "VA",
        "lat": 38.9350,
        "lon": -77.4472,
        "distance_mi": 135.10,
        "bearing": 242.03,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00094702": {
        "name": "SIKORSKY MEM AP",
        "state": "CT",
        "lat": 41.1642,
        "lon": -73.1267,
        "distance_mi": 141.85,
        "bearing": 50.36,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00013773": {
        "name": "QUANTICO MCAS",
        "state": "VA",
        "lat": 38.5017,
        "lon": -77.3053,
        "distance_mi": 146.08,
        "bearing": 230.27,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00013734": {
        "name": "MARTINSBURG",
        "state": "WV",
        "lat": 39.4022,
        "lon": -77.9844,
        "distance_mi": 149.82,
        "bearing": 258.39,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    # --- Ring 4: Far (150-250 mi) --- 16 stations
    "USW00004725": {
        "name": "BINGHAMTON",
        "state": "NY",
        "lat": 42.1997,
        "lon": -75.9847,
        "distance_mi": 165.52,
        "bearing": 346.44,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00014736": {
        "name": "ALTOONA BLAIR CO AP",
        "state": "PA",
        "lat": 40.2964,
        "lon": -78.3200,
        "distance_mi": 165.96,
        "bearing": 281.21,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00013731": {
        "name": "FRONT ROYAL",
        "state": "VA",
        "lat": 38.8878,
        "lon": -78.2106,
        "distance_mi": 171.43,
        "bearing": 250.30,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00014748": {
        "name": "ELMIRA CORNING RGNL AP",
        "state": "NY",
        "lat": 42.1594,
        "lon": -76.8914,
        "distance_mi": 180.28,
        "bearing": 331.61,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00014740": {
        "name": "HARTFORD-BRADLEY INTL AP",
        "state": "CT",
        "lat": 41.9375,
        "lon": -72.6819,
        "distance_mi": 194.93,
        "bearing": 42.15,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00013702": {
        "name": "LANGLEY AFB",
        "state": "VA",
        "lat": 37.0831,
        "lon": -76.3606,
        "distance_mi": 202.09,
        "bearing": 197.82,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00093735": {
        "name": "FT EUSTIS FELKER AAF",
        "state": "VA",
        "lat": 37.1325,
        "lon": -76.6100,
        "distance_mi": 203.26,
        "bearing": 201.82,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00013750": {
        "name": "NORFOLK NAS",
        "state": "VA",
        "lat": 36.9372,
        "lon": -76.2892,
        "distance_mi": 210.78,
        "bearing": 196.16,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00013737": {
        "name": "NORFOLK INTL AP",
        "state": "VA",
        "lat": 36.8964,
        "lon": -76.2006,
        "distance_mi": 211.72,
        "bearing": 194.59,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00014735": {
        "name": "ALBANY INTL AP",
        "state": "NY",
        "lat": 42.7472,
        "lon": -73.7992,
        "distance_mi": 211.93,
        "bearing": 19.99,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00004787": {
        "name": "DUBOIS RGNL AP",
        "state": "PA",
        "lat": 41.1778,
        "lon": -78.8986,
        "distance_mi": 212.63,
        "bearing": 296.26,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00013769": {
        "name": "OCEANA NAS",
        "state": "VA",
        "lat": 36.8206,
        "lon": -76.0339,
        "distance_mi": 215.72,
        "bearing": 191.90,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00004751": {
        "name": "BRADFORD RGNL AP",
        "state": "PA",
        "lat": 41.8031,
        "lon": -78.6400,
        "distance_mi": 222.30,
        "bearing": 307.83,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00094704": {
        "name": "DANSVILLE MUNI AP",
        "state": "NY",
        "lat": 42.5708,
        "lon": -77.7133,
        "distance_mi": 226.70,
        "bearing": 326.10,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00014771": {
        "name": "SYRACUSE HANCOCK INTL AP",
        "state": "NY",
        "lat": 43.1111,
        "lon": -76.1039,
        "distance_mi": 228.27,
        "bearing": 348.82,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00013736": {
        "name": "MORGANTOWN MUNI AP",
        "state": "WV",
        "lat": 39.6428,
        "lon": -79.9167,
        "distance_mi": 249.69,
        "bearing": 268.02,
        "ring": "Ring4_Far",
        "sector": "W",
    },
}


# ==============================================================================
# Pipeline Constants (used by preprocessing and benchmark scripts)
# ==============================================================================
# These mirror the NYC config.py structure for pipeline compatibility.

# Date Range
START_DATE = "1985-01-01"
END_DATE = "2024-12-31"

# Input Features
INPUT_VARIABLES = ["TMAX", "TMIN"]

# Data Quality
MAX_FORWARD_FILL_DAYS = 3

# Train / Validation / Test Split Ratios (chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZES = [64, 32]
DROPOUT = 0.0

# Kalshi Contract Bucket Definitions (from CityConfig)
BUCKET_EDGES = list(CITY_CONFIG.bucket_edges)
BUCKET_LABELS = list(CITY_CONFIG.bucket_labels)
