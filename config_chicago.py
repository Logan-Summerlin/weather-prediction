"""
Expanded Configuration for Chicago Temperature Prediction.

Station network centered on O'Hare International Airport (USW00094846).
Organized by distance ring and compass sector from ORD.

Ring Classification (distance from ORD):
    Ring1_Near:      0 - 50 miles
    Ring2_Regional:  50 - 100 miles
    Ring3_Extended:  100 - 150 miles
    Ring4_Far:       150 - 250 miles

Sector Classification (compass bearing from ORD):
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
CITY_CONFIG = get_city_config("chi")

# Target station
TARGET_STATION = CITY_CONFIG.target_station
TARGET_LAT = CITY_CONFIG.target_lat
TARGET_LON = CITY_CONFIG.target_lon
TARGET_VARIABLE = "TMAX"

# ==============================================================================
# Surrounding Stations (~55 stations)
# ==============================================================================
# Format: {station_id: "Name, State (distance sector)"}
# All stations have >= 80% TMAX completeness over 2018-2022.
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 9 stations
    "USW00014819": "MIDWAY AP, IL (12mi S)",
    "USW00094854": "PALWAUKEE MUNI/CHICAGO EXEC, IL (15mi NW)",
    "USW00094866": "LANSING MUNI AP, IL (20mi SE)",
    "USW00094847": "GARY REGIONAL AP, IN (25mi SE)",
    "USW00094892": "DUPAGE AP, IL (25mi W)",
    "USW00014855": "AURORA MUNI AP, IL (35mi W)",
    "USW00004831": "WAUKEGAN REGIONAL AP, IL (38mi N)",
    "USW00014880": "JOLIET REGIONAL AP, IL (40mi SW)",
    "USW00094843": "VALPARAISO PORTER CO MUNI, IN (45mi SE)",
    # --- Ring 2: Regional (50-100 mi) --- 12 stations
    "USW00094860": "KENOSHA REGIONAL AP, WI (55mi N)",
    "USW00094844": "MICHIGAN CITY MUNI AP, IN (55mi E)",
    "USW00003887": "KANKAKEE GREATER KANKAKEE AP, IL (55mi S)",
    "USW00094871": "DEKALB TAYLOR MUNI AP, IL (60mi W)",
    "USW00053856": "BURLINGTON MUNI AP, WI (60mi N)",
    "USW00014838": "JANESVILLE SOUTHERN WI RGNL, WI (70mi NW)",
    "USW00014891": "ROCHELLE KORITZ FIELD, IL (70mi W)",
    "USW00094822": "ROCKFORD GREATER ROCKFORD AP, IL (75mi NW)",
    "USW00014839": "MILWAUKEE MITCHELL INTL AP, WI (80mi N)",
    "USW00014848": "SOUTH BEND REGIONAL AP, IN (80mi E)",
    "USW00014827": "PONTIAC LIVINGSTON CO AP, IL (85mi SW)",
    "USW00014826": "BLOOMINGTON NORMAL AP, IL (95mi SW)",
    # --- Ring 3: Extended (100-150 mi) --- 16 stations
    "USW00014898": "LAFAYETTE PURDUE UNIV AP, IN (110mi SE)",
    "USW00014821": "DANVILLE VERMILION CO AP, IL (120mi S)",
    "USW00094861": "BATTLE CREEK WK KELLOGG AP, MI (120mi E)",
    "USW00014837": "MADISON DANE CO RGNL AP, WI (120mi NW)",
    "USW00094870": "CHAMPAIGN WILLARD AP, IL (125mi S)",
    "USW00094830": "FORT WAYNE INTL AP, IN (130mi E)",
    "USW00014850": "OSHKOSH WITTMAN RGNL, WI (130mi N)",
    "USW00094849": "KALAMAZOO BATTLE CREEK INTL, MI (130mi E)",
    "USW00014840": "MUSKEGON COUNTY AP, MI (130mi NE)",
    "USW00014842": "PEORIA GREATER PEORIA AP, IL (140mi SW)",
    "USW00014841": "GRAND RAPIDS GERALD FORD AP, MI (140mi NE)",
    "USW00014836": "LA CROSSE MUNI AP, WI (140mi NW)",
    "USW00014847": "APPLETON OUTAGAMIE CO RGNL, WI (140mi N)",
    "USW00014834": "GREEN BAY AUSTIN STRAUBEL, WI (145mi N)",
    "USW00014920": "DUBUQUE REGIONAL AP, IA (145mi W)",
    "USW00093817": "TERRE HAUTE HULMAN RGNL, IN (145mi SE)",
    # --- Ring 4: Far (150-250 mi) --- 18 stations (expanded to ~55 total)
    "USW00014923": "MOLINE QUAD CITY INTL AP, IL (160mi W)",
    "USW00094910": "CEDAR RAPIDS EASTERN IOWA AP, IA (200mi W)",
    "USW00093819": "INDIANAPOLIS INTL AP, IN (165mi SE)",
    "USW00014958": "SPRINGFIELD CAPITAL AP, IL (185mi SW)",
    "USW00014933": "WATERLOO MUNI AP, IA (220mi W)",
    "USW00014922": "MASON CITY MUNI AP, IA (230mi NW)",
    "USW00014843": "SAGINAW MBS INTL AP, MI (230mi NE)",
    "USW00094850": "TRAVERSE CITY CHERRY CAPITAL, MI (230mi NE)",
    # --- Expansion: fill gaps within 200 mi --- 10 additional stations
    "USW00014845": "LANSING CAPITAL CITY AP, MI (170mi E)",
    "USW00093822": "EVANSVILLE REGIONAL AP, IN (195mi SE)",
    "USW00014852": "WAUSAU DOWNTOWN AP, WI (190mi N)",
    "USW00014862": "TOLEDO EXPRESS AP, OH (200mi E)",
    "USW00014918": "BURLINGTON SOUTHEAST IOWA RGNL, IA (170mi SW)",
    "USW00013960": "MUNCIE DELAWARE CO AP, IN (165mi SE)",
    "USW00003889": "MATTOON COLES CO MEM AP, IL (170mi S)",
    "USW00014896": "FREEPORT ALBERTUS AP, IL (100mi W)",
    "USW00094818": "GOSHEN MUNI AP, IN (110mi E)",
    "USW00014817": "SHELBYVILLE MUNI AP, IL (155mi S)",
}

# All stations (target + expanded surrounding) for convenience
ALL_STATIONS = {
    TARGET_STATION: "O'Hare International Airport (Target)",
    **SURROUNDING_STATIONS,
}

# ==============================================================================
# ASOS/AWOS Mapping (operational station IDs)
# ==============================================================================
# Mapping of GHCN station IDs to ICAO codes for stations with ASOS/AWOS data.
# Stations without operational ASOS/AWOS are excluded and listed separately.
ASOS_STATION_MAP = {
    "USW00094846": "KORD",
    "USW00014819": "KMDW",
    "USW00094854": "KPWK",
    "USW00094866": "KIGQ",
    "USW00094847": "KGYY",
    "USW00094892": "KDPA",
    "USW00014855": "KARR",
    "USW00004831": "KUGN",
    "USW00014880": "KJOT",
    "USW00094843": "KVPZ",
    "USW00094860": "KENW",
    "USW00094844": "KMCX",
    "USW00003887": "KIKK",
    "USW00094871": "KDKB",
    "USW00014838": "KJVL",
    "USW00014891": "KRPJ",
    "USW00094822": "KRFD",
    "USW00014839": "KMKE",
    "USW00014848": "KSBN",
    "USW00014827": "KPNT",
    "USW00014826": "KBMI",
    "USW00014898": "KLAF",
    "USW00014821": "KDNV",
    "USW00094861": "KBTL",
    "USW00014837": "KMSN",
    "USW00094870": "KCMI",
    "USW00094830": "KFWA",
    "USW00014850": "KOSH",
    "USW00094849": "KAZO",
    "USW00014840": "KMKG",
    "USW00014842": "KPIA",
    "USW00014841": "KGRR",
    "USW00014836": "KLSE",
    "USW00014847": "KATW",
    "USW00014834": "KGRB",
    "USW00014920": "KDBQ",
    "USW00093817": "KHUF",
    "USW00014923": "KMLI",
    "USW00094910": "KCID",
    "USW00093819": "KIND",
    "USW00014958": "KSPI",
    "USW00014933": "KALO",
    "USW00014922": "KMCW",
    "USW00014843": "KMBS",
    "USW00094850": "KTVC",
}

NON_ASOS_STATIONS = {
    "USW00053856": "BURLINGTON MUNI AP, WI (ICAO/ASOS not confirmed)",
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
        "USW00014819",
        "USW00094854",
        "USW00094866",
        "USW00094847",
        "USW00094892",
        "USW00014855",
        "USW00004831",
        "USW00014880",
        "USW00094843",
    ],
    "Ring2_Regional": [
        "USW00094860",
        "USW00094844",
        "USW00003887",
        "USW00094871",
        "USW00053856",
        "USW00014838",
        "USW00014891",
        "USW00094822",
        "USW00014839",
        "USW00014848",
        "USW00014827",
        "USW00014826",
    ],
    "Ring3_Extended": [
        "USW00014898",
        "USW00014821",
        "USW00094861",
        "USW00014837",
        "USW00094870",
        "USW00094830",
        "USW00014850",
        "USW00094849",
        "USW00014840",
        "USW00014842",
        "USW00014841",
        "USW00014836",
        "USW00014847",
        "USW00014834",
        "USW00014920",
        "USW00093817",
    ],
    "Ring4_Far": [
        "USW00014923",
        "USW00094910",
        "USW00093819",
        "USW00014958",
        "USW00014933",
        "USW00014922",
        "USW00014843",
        "USW00094850",
        "USW00014845",
        "USW00093822",
        "USW00014852",
        "USW00014862",
        "USW00014918",
        "USW00013960",
        "USW00003889",
        "USW00014896",
        "USW00094818",
        "USW00014817",
    ],
}

# ==============================================================================
# Station Sectors (compass direction classification)
# ==============================================================================
STATION_SECTORS = {
    "N": [
        "USW00004831",
        "USW00094860",
        "USW00053856",
        "USW00014839",
        "USW00014850",
        "USW00014847",
        "USW00014834",
    ],
    "NE": [
        "USW00014840",
        "USW00014841",
        "USW00014843",
        "USW00094850",
    ],
    "E": [
        "USW00094844",
        "USW00014848",
        "USW00094861",
        "USW00094830",
        "USW00094849",
    ],
    "SE": [
        "USW00094866",
        "USW00094847",
        "USW00094843",
        "USW00014898",
        "USW00093817",
        "USW00093819",
    ],
    "S": [
        "USW00014819",
        "USW00003887",
        "USW00014821",
        "USW00094870",
    ],
    "SW": [
        "USW00014880",
        "USW00014827",
        "USW00014826",
        "USW00014842",
        "USW00014958",
    ],
    "W": [
        "USW00094892",
        "USW00014855",
        "USW00094871",
        "USW00014891",
        "USW00014920",
        "USW00014923",
        "USW00094910",
        "USW00014933",
    ],
    "NW": [
        "USW00094854",
        "USW00014838",
        "USW00094822",
        "USW00014837",
        "USW00014836",
        "USW00014922",
    ],
}

# ==============================================================================
# Meteorological Sector Assignments (for feature engineering)
# ==============================================================================
# These group stations by meteorological relevance rather than pure compass direction.
# Chicago's weather is dominated by Lake Michigan moderation to the NE/E,
# continental cold-air advection from the W/NW, and warm advection from the S/SW.
METEOROLOGICAL_SECTORS = {
    "WNW": (  # Upstream cold-air advection: W and NW sectors (continental interior + arctic outbreaks)
        STATION_SECTORS["W"] + STATION_SECTORS["NW"]
    ),
    "Lake": (  # Lake Michigan moderation: N, NE, and E sectors near the lake
        STATION_SECTORS["N"] + STATION_SECTORS["NE"] + STATION_SECTORS["E"]
    ),
    "SW": (  # Warm advection from Gulf track: S and SW sectors
        STATION_SECTORS["S"] + STATION_SECTORS["SW"]
    ),
    "NearField": (  # Urban/local: all Ring 1 stations
        STATION_RINGS["Ring1_Near"]
    ),
    "NE_Lake": [  # Lake-shore stations critical for lake breeze dynamics
        "USW00094860",
        "USW00094844",
        "USW00004831",
        "USW00014839",
        "USW00014840",
    ],
}

# ==============================================================================
# Station Metadata
# ==============================================================================
# Full metadata for each surrounding station, sourced from the GHCN inventory.
# Distances and bearings computed via haversine from ORD (41.9742, -87.9073).
STATION_METADATA = {
    # --- Ring 1: Near-field (0-50 mi) --- 9 stations
    "USW00014819": {
        "name": "MIDWAY AP",
        "state": "IL",
        "lat": 41.7861,
        "lon": -87.7522,
        "distance_mi": 14.69,
        "bearing": 146.96,
        "ring": "Ring1_Near",
        "sector": "S",
    },
    "USW00094854": {
        "name": "PALWAUKEE MUNI/CHICAGO EXEC",
        "state": "IL",
        "lat": 42.1142,
        "lon": -87.9036,
        "distance_mi": 9.68,
        "bearing": 358.85,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    "USW00094866": {
        "name": "LANSING MUNI AP",
        "state": "IL",
        "lat": 41.5347,
        "lon": -87.5297,
        "distance_mi": 36.12,
        "bearing": 138.84,
        "ring": "Ring1_Near",
        "sector": "SE",
    },
    "USW00094847": {
        "name": "GARY REGIONAL AP",
        "state": "IN",
        "lat": 41.6156,
        "lon": -87.4128,
        "distance_mi": 35.94,
        "bearing": 124.84,
        "ring": "Ring1_Near",
        "sector": "SE",
    },
    "USW00094892": {
        "name": "DUPAGE AP",
        "state": "IL",
        "lat": 41.9078,
        "lon": -88.2486,
        "distance_mi": 16.90,
        "bearing": 258.73,
        "ring": "Ring1_Near",
        "sector": "W",
    },
    "USW00014855": {
        "name": "AURORA MUNI AP",
        "state": "IL",
        "lat": 41.7706,
        "lon": -88.4750,
        "distance_mi": 30.86,
        "bearing": 241.33,
        "ring": "Ring1_Near",
        "sector": "W",
    },
    "USW00004831": {
        "name": "WAUKEGAN REGIONAL AP",
        "state": "IL",
        "lat": 42.4219,
        "lon": -87.8678,
        "distance_mi": 31.00,
        "bearing": 3.16,
        "ring": "Ring1_Near",
        "sector": "N",
    },
    "USW00014880": {
        "name": "JOLIET REGIONAL AP",
        "state": "IL",
        "lat": 41.5178,
        "lon": -88.1756,
        "distance_mi": 33.32,
        "bearing": 213.49,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USW00094843": {
        "name": "VALPARAISO PORTER CO MUNI",
        "state": "IN",
        "lat": 41.4531,
        "lon": -87.0069,
        "distance_mi": 56.01,
        "bearing": 121.07,
        "ring": "Ring1_Near",
        "sector": "SE",
    },
    # --- Ring 2: Regional (50-100 mi) --- 12 stations
    "USW00094860": {
        "name": "KENOSHA REGIONAL AP",
        "state": "WI",
        "lat": 42.5942,
        "lon": -87.9278,
        "distance_mi": 42.92,
        "bearing": 358.05,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00094844": {
        "name": "MICHIGAN CITY MUNI AP",
        "state": "IN",
        "lat": 41.7033,
        "lon": -86.8211,
        "distance_mi": 56.91,
        "bearing": 107.86,
        "ring": "Ring2_Regional",
        "sector": "E",
    },
    "USW00003887": {
        "name": "KANKAKEE GREATER KANKAKEE AP",
        "state": "IL",
        "lat": 41.0714,
        "lon": -87.8461,
        "distance_mi": 62.43,
        "bearing": 183.94,
        "ring": "Ring2_Regional",
        "sector": "S",
    },
    "USW00094871": {
        "name": "DEKALB TAYLOR MUNI AP",
        "state": "IL",
        "lat": 41.9336,
        "lon": -88.7075,
        "distance_mi": 39.55,
        "bearing": 266.89,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00053856": {
        "name": "BURLINGTON MUNI AP",
        "state": "WI",
        "lat": 42.6906,
        "lon": -88.3044,
        "distance_mi": 53.52,
        "bearing": 337.41,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00014838": {
        "name": "JANESVILLE SOUTHERN WI RGNL",
        "state": "WI",
        "lat": 42.6203,
        "lon": -89.0406,
        "distance_mi": 62.07,
        "bearing": 319.25,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USW00014891": {
        "name": "ROCHELLE KORITZ FIELD",
        "state": "IL",
        "lat": 41.8931,
        "lon": -89.0822,
        "distance_mi": 57.56,
        "bearing": 269.48,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00094822": {
        "name": "ROCKFORD GREATER ROCKFORD AP",
        "state": "IL",
        "lat": 42.1953,
        "lon": -89.0972,
        "distance_mi": 58.59,
        "bearing": 287.42,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USW00014839": {
        "name": "MILWAUKEE MITCHELL INTL AP",
        "state": "WI",
        "lat": 42.9553,
        "lon": -87.9044,
        "distance_mi": 67.86,
        "bearing": 0.21,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00014848": {
        "name": "SOUTH BEND REGIONAL AP",
        "state": "IN",
        "lat": 41.7086,
        "lon": -86.3172,
        "distance_mi": 85.15,
        "bearing": 105.20,
        "ring": "Ring2_Regional",
        "sector": "E",
    },
    "USW00014827": {
        "name": "PONTIAC LIVINGSTON CO AP",
        "state": "IL",
        "lat": 40.9244,
        "lon": -88.6256,
        "distance_mi": 80.15,
        "bearing": 213.09,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00014826": {
        "name": "BLOOMINGTON NORMAL AP",
        "state": "IL",
        "lat": 40.4772,
        "lon": -88.9158,
        "distance_mi": 108.07,
        "bearing": 213.38,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    # --- Ring 3: Extended (100-150 mi) --- 16 stations
    "USW00014898": {
        "name": "LAFAYETTE PURDUE UNIV AP",
        "state": "IN",
        "lat": 40.4125,
        "lon": -86.9369,
        "distance_mi": 115.87,
        "bearing": 155.81,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    "USW00014821": {
        "name": "DANVILLE VERMILION CO AP",
        "state": "IL",
        "lat": 40.1997,
        "lon": -87.5975,
        "distance_mi": 123.38,
        "bearing": 188.70,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00094861": {
        "name": "BATTLE CREEK WK KELLOGG AP",
        "state": "MI",
        "lat": 42.3078,
        "lon": -85.2514,
        "distance_mi": 136.89,
        "bearing": 80.64,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00014837": {
        "name": "MADISON DANE CO RGNL AP",
        "state": "WI",
        "lat": 43.1397,
        "lon": -89.3375,
        "distance_mi": 106.07,
        "bearing": 318.04,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    "USW00094870": {
        "name": "CHAMPAIGN WILLARD AP",
        "state": "IL",
        "lat": 40.0397,
        "lon": -88.2778,
        "distance_mi": 134.54,
        "bearing": 190.85,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00094830": {
        "name": "FORT WAYNE INTL AP",
        "state": "IN",
        "lat": 40.9781,
        "lon": -85.1953,
        "distance_mi": 153.41,
        "bearing": 107.00,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00014850": {
        "name": "OSHKOSH WITTMAN RGNL",
        "state": "WI",
        "lat": 43.9844,
        "lon": -88.5569,
        "distance_mi": 143.67,
        "bearing": 345.72,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00094849": {
        "name": "KALAMAZOO BATTLE CREEK INTL",
        "state": "MI",
        "lat": 42.2350,
        "lon": -85.5517,
        "distance_mi": 121.87,
        "bearing": 83.34,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00014840": {
        "name": "MUSKEGON COUNTY AP",
        "state": "MI",
        "lat": 43.1703,
        "lon": -86.2381,
        "distance_mi": 115.57,
        "bearing": 55.94,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00014842": {
        "name": "PEORIA GREATER PEORIA AP",
        "state": "IL",
        "lat": 40.6672,
        "lon": -89.6839,
        "distance_mi": 116.55,
        "bearing": 232.17,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00014841": {
        "name": "GRAND RAPIDS GERALD FORD AP",
        "state": "MI",
        "lat": 42.8808,
        "lon": -85.5228,
        "distance_mi": 131.95,
        "bearing": 63.69,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00014836": {
        "name": "LA CROSSE MUNI AP",
        "state": "WI",
        "lat": 43.8789,
        "lon": -91.2567,
        "distance_mi": 175.95,
        "bearing": 318.67,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    "USW00014847": {
        "name": "APPLETON OUTAGAMIE CO RGNL",
        "state": "WI",
        "lat": 44.2581,
        "lon": -88.5192,
        "distance_mi": 160.01,
        "bearing": 345.27,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00014834": {
        "name": "GREEN BAY AUSTIN STRAUBEL",
        "state": "WI",
        "lat": 44.4850,
        "lon": -88.1297,
        "distance_mi": 174.54,
        "bearing": 354.05,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00014920": {
        "name": "DUBUQUE REGIONAL AP",
        "state": "IA",
        "lat": 42.4019,
        "lon": -90.7094,
        "distance_mi": 141.12,
        "bearing": 281.40,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00093817": {
        "name": "TERRE HAUTE HULMAN RGNL",
        "state": "IN",
        "lat": 39.4517,
        "lon": -87.3075,
        "distance_mi": 178.15,
        "bearing": 165.78,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    # --- Ring 4: Far (150-250 mi) --- 8 stations
    "USW00014923": {
        "name": "MOLINE QUAD CITY INTL AP",
        "state": "IL",
        "lat": 41.4536,
        "lon": -90.5075,
        "distance_mi": 134.60,
        "bearing": 258.00,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00094910": {
        "name": "CEDAR RAPIDS EASTERN IOWA AP",
        "state": "IA",
        "lat": 41.8847,
        "lon": -91.7108,
        "distance_mi": 187.23,
        "bearing": 270.44,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00093819": {
        "name": "INDIANAPOLIS INTL AP",
        "state": "IN",
        "lat": 39.7175,
        "lon": -86.2947,
        "distance_mi": 166.65,
        "bearing": 151.96,
        "ring": "Ring4_Far",
        "sector": "SE",
    },
    "USW00014958": {
        "name": "SPRINGFIELD CAPITAL AP",
        "state": "IL",
        "lat": 39.8444,
        "lon": -89.6839,
        "distance_mi": 156.47,
        "bearing": 217.49,
        "ring": "Ring4_Far",
        "sector": "SW",
    },
    "USW00014933": {
        "name": "WATERLOO MUNI AP",
        "state": "IA",
        "lat": 42.5542,
        "lon": -92.4003,
        "distance_mi": 226.40,
        "bearing": 280.41,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00014922": {
        "name": "MASON CITY MUNI AP",
        "state": "IA",
        "lat": 43.1578,
        "lon": -93.3314,
        "distance_mi": 265.44,
        "bearing": 290.62,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00014843": {
        "name": "SAGINAW MBS INTL AP",
        "state": "MI",
        "lat": 43.5331,
        "lon": -84.0797,
        "distance_mi": 210.86,
        "bearing": 63.51,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00094850": {
        "name": "TRAVERSE CITY CHERRY CAPITAL",
        "state": "MI",
        "lat": 44.7414,
        "lon": -85.5822,
        "distance_mi": 224.42,
        "bearing": 45.93,
        "ring": "Ring4_Far",
        "sector": "NE",
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
