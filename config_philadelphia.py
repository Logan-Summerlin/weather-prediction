"""
Expanded Configuration for Philadelphia Temperature Prediction.

Station network centered on Philadelphia International Airport (USW00013739).
Organized by distance ring and compass sector from PHL.

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
# Surrounding Stations (~48 stations)
# ==============================================================================
# Format: {station_id: "Name, State (distance sector)"}
# All stations have >= 80% TMAX completeness over 2018-2022.
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 8 stations
    "USW00094732": "NE PHILA AP, PA (18mi NE)",
    "USW00093780": "S JERSEY RGNL AP, NJ (21mi E)",
    "USW00013781": "WILMINGTON AP, DE (24mi SW)",
    "USW00054782": "HERITAGE FLD AP, PA (31mi NW)",
    "USW00054786": "DOYLESTOWN AP, PA (32mi N)",
    "USW00014780": "MCGUIRE AFB, NJ (35mi E)",
    "USW00014792": "TRENTON-MERCER AP, NJ (35mi NE)",
    "USW00093730": "ATLANTIC CITY INTL AP, NJ (46mi SE)",
    # --- Ring 2: Regional (50-100 mi) --- 17 stations
    "USW00014712": "READING RGNL AP, PA (52mi NW)",
    "USW00013707": "DOVER AFB, DE (53mi S)",
    "USW00013724": "ATLANTIC CITY MARINA, NJ (55mi SE)",
    "USW00014737": "ALLENTOWN LEHIGH VLY INTL AP, PA (55mi N)",
    "USW00054737": "LANCASTER AP, PA (59mi W)",
    "USW00054785": "SOMERSET AP, NJ (60mi NE)",
    "USW00014734": "NEWARK LIBERTY INTL AP, NJ (79mi NE)",
    "USW00054779": "AEROFLEX-ANDOVER AP, NJ (83mi N)",
    "USW00054743": "CALDWELL ESSEX CO AP, NJ (85mi NE)",
    "USW00054789": "MT POCONO POCONO MOUNTAINS MUN, PA (88mi N)",
    "USW00014768": "HARRISBURG INTL AP, PA (89mi W)",
    "USW00094728": "CENTRAL PARK, NY (91mi NE)",
    "USW00093721": "BALTIMORE-WASHINGTON INTL AP, MD (91mi SW)",
    "USW00094741": "TETERBORO AP, NJ (92mi NE)",
    "USW00094789": "JFK INTL AP, NY (94mi NE)",
    "USW00014732": "LAGUARDIA AP, NY (95mi NE)",
    "USW00054793": "SUSSEX AP, NJ (97mi N)",
    # --- Ring 3: Extended (100-150 mi) --- 18 stations
    "USW00014777": "WILKES-BARRE/SCRANTON INTL AP, PA (104mi N)",
    "USW00093738": "SALISBURY-WICOMICO AP, MD (107mi S)",
    "USW00093786": "OCEAN CITY MUNI AP, MD (108mi S)",
    "USW00014770": "SELINSGROVE PENN VALLEY AP, PA (108mi NW)",
    "USW00054787": "FARMINGDALE REPUBLIC AP, NY (112mi NE)",
    "USW00094745": "WESTCHESTER CO AP, NY (115mi NE)",
    "USW00093720": "REAGAN NATIONAL AP, DC (120mi SW)",
    "USW00004789": "MONTGOMERY ORANGE CO AP, NY (124mi NE)",
    "USW00013711": "PATUXENT RIVER NAS, MD (127mi SW)",
    "USW00054746": "MONTICELLO SULLIVAN, NY (128mi N)",
    "USW00004781": "ISLIP-LI MACARTHUR AP, NY (129mi NE)",
    "USW00014751": "WILLIAMSPORT RGNL AP, PA (130mi NW)",
    "USW00093739": "WALLOPS ISLAND, VA (134mi S)",
    "USW00013740": "DULLES INTL AP, VA (135mi SW)",
    "USW00054734": "DANBURY MUNI AP, CT (138mi NE)",
    "USW00014757": "POUGHKEEPSIE AP, NY (140mi NE)",
    "USW00054790": "SHIRLEY BROOKHAVEN AP, NY (140mi NE)",
    "USW00094702": "IGOR I SIKORSKY MEM AP, CT (142mi NE)",
    # --- Ring 4: Far (150-250 mi) --- 5 stations
    "USW00014758": "NEW HAVEN TWEED AP, CT (156mi NE)",
    "USW00004725": "BINGHAMTON, NY (166mi N)",
    "USW00014740": "HARTFORD-BRADLEY INTL AP, CT (195mi NE)",
    "USW00014735": "ALBANY INTL AP, NY (212mi N)",
    "USW00014771": "SYRACUSE HANCOCK INTL AP, NY (228mi N)",
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
    "USW00093780": "KVAY",
    "USW00013781": "KILG",
    "USW00054786": "KDYL",
    "USW00014780": "KWRI",
    "USW00014792": "KTTN",
    "USW00093730": "KACY",
    "USW00014712": "KRDG",
    "USW00013707": "KDOV",
    "USW00014737": "KABE",
    "USW00054737": "KLNS",
    "USW00054785": "KSMQ",
    "USW00014734": "KEWR",
    "USW00054779": "K12N",
    "USW00054743": "KCDW",
    "USW00054789": "KMPO",
    "USW00014768": "KMDT",
    "USW00094728": "KNYC",
    "USW00093721": "KBWI",
    "USW00094741": "KTEB",
    "USW00094789": "KJFK",
    "USW00014732": "KLGA",
    "USW00054793": "KFWN",
    "USW00014777": "KAVP",
    "USW00093738": "KSBY",
    "USW00093786": "KOXB",
    "USW00014770": "KSEG",
    "USW00054787": "KFRG",
    "USW00094745": "KHPN",
    "USW00093720": "KDCA",
    "USW00004789": "KMGJ",
    "USW00013711": "KNHK",
    "USW00054746": "KMSV",
    "USW00004781": "KISP",
    "USW00014751": "KIPT",
    "USW00093739": "KWAL",
    "USW00013740": "KIAD",
    "USW00054734": "KDXR",
    "USW00014757": "KPOU",
    "USW00054790": "KHWV",
    "USW00094702": "KBDR",
    "USW00014758": "KHVN",
    "USW00004725": "KBGM",
    "USW00014740": "KBDL",
    "USW00014735": "KALB",
    "USW00014771": "KSYR",
}

NON_ASOS_STATIONS = {
    "USW00054782": "HERITAGE FLD AP (ICAO/ASOS not confirmed)",
    "USW00013724": "ATLANTIC CITY MARINA (non-airport station)",
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
        "USW00093780",
        "USW00013781",
        "USW00054782",
        "USW00054786",
        "USW00014780",
        "USW00014792",
        "USW00093730",
    ],
    "Ring2_Regional": [
        "USW00014712",
        "USW00013707",
        "USW00013724",
        "USW00014737",
        "USW00054737",
        "USW00054785",
        "USW00014734",
        "USW00054779",
        "USW00054743",
        "USW00054789",
        "USW00014768",
        "USW00094728",
        "USW00093721",
        "USW00094741",
        "USW00094789",
        "USW00014732",
        "USW00054793",
    ],
    "Ring3_Extended": [
        "USW00014777",
        "USW00093738",
        "USW00093786",
        "USW00014770",
        "USW00054787",
        "USW00094745",
        "USW00093720",
        "USW00004789",
        "USW00013711",
        "USW00054746",
        "USW00004781",
        "USW00014751",
        "USW00093739",
        "USW00013740",
        "USW00054734",
        "USW00014757",
        "USW00054790",
        "USW00094702",
    ],
    "Ring4_Far": [
        "USW00014758",
        "USW00004725",
        "USW00014740",
        "USW00014735",
        "USW00014771",
    ],
}

# ==============================================================================
# Station Sectors (compass direction classification)
# ==============================================================================
STATION_SECTORS = {
    "N": [
        "USW00054786",
        "USW00014737",
        "USW00054779",
        "USW00054789",
        "USW00054793",
        "USW00014777",
        "USW00054746",
        "USW00004725",
        "USW00014735",
        "USW00014771",
    ],
    "NE": [
        "USW00094732",
        "USW00014792",
        "USW00054785",
        "USW00014734",
        "USW00054743",
        "USW00094728",
        "USW00094741",
        "USW00094789",
        "USW00014732",
        "USW00054787",
        "USW00094745",
        "USW00004789",
        "USW00004781",
        "USW00054734",
        "USW00014757",
        "USW00054790",
        "USW00094702",
        "USW00014758",
        "USW00014740",
    ],
    "E": [
        "USW00093780",
        "USW00014780",
    ],
    "SE": [
        "USW00093730",
        "USW00013724",
    ],
    "S": [
        "USW00013707",
        "USW00093738",
        "USW00093786",
        "USW00093739",
    ],
    "SW": [
        "USW00013781",
        "USW00093721",
        "USW00093720",
        "USW00013711",
        "USW00013740",
    ],
    "W": [
        "USW00054737",
        "USW00014768",
    ],
    "NW": [
        "USW00054782",
        "USW00014712",
        "USW00014770",
        "USW00014751",
    ],
}

# ==============================================================================
# Meteorological Sector Assignments (for feature engineering)
# ==============================================================================
# These group stations by meteorological relevance rather than pure compass direction.
METEOROLOGICAL_SECTORS = {
    "WNW": (  # Upstream cold-air advection: W and NW sectors (Appalachian stations)
        STATION_SECTORS["W"] + STATION_SECTORS["NW"]
    ),
    "SW": (  # Warm advection from south: S and SW sectors (Chesapeake/mid-Atlantic)
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
    # --- Ring 1: Near-field (0-50 mi) --- 8 stations
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
    "USW00093780": {
        "name": "S JERSEY RGNL AP",
        "state": "NJ",
        "lat": 39.9408,
        "lon": -74.8408,
        "distance_mi": 20.99,
        "bearing": 77.04,
        "ring": "Ring1_Near",
        "sector": "E",
    },
    "USW00013781": {
        "name": "WILMINGTON AP",
        "state": "DE",
        "lat": 39.6728,
        "lon": -75.6014,
        "distance_mi": 24.24,
        "bearing": 235.26,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USW00054782": {
        "name": "HERITAGE FLD AP",
        "state": "PA",
        "lat": 40.2381,
        "lon": -75.5547,
        "distance_mi": 30.59,
        "bearing": 325.59,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    "USW00054786": {
        "name": "DOYLESTOWN AP",
        "state": "PA",
        "lat": 40.3303,
        "lon": -75.1228,
        "distance_mi": 32.05,
        "bearing": 9.85,
        "ring": "Ring1_Near",
        "sector": "N",
    },
    "USW00014780": {
        "name": "MCGUIRE AFB",
        "state": "NJ",
        "lat": 40.0156,
        "lon": -74.5914,
        "distance_mi": 35.07,
        "bearing": 73.51,
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
    # --- Ring 2: Regional (50-100 mi) --- 17 stations
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
    "USW00054737": {
        "name": "LANCASTER AP",
        "state": "PA",
        "lat": 40.1206,
        "lon": -76.2944,
        "distance_mi": 59.03,
        "bearing": 287.17,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00054785": {
        "name": "SOMERSET AP",
        "state": "NJ",
        "lat": 40.6242,
        "lon": -74.6689,
        "distance_mi": 59.65,
        "bearing": 29.38,
        "ring": "Ring2_Regional",
        "sector": "NE",
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
    "USW00054779": {
        "name": "AEROFLEX-ANDOVER AP",
        "state": "NJ",
        "lat": 41.0092,
        "lon": -74.7364,
        "distance_mi": 82.61,
        "bearing": 18.03,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00054743": {
        "name": "CALDWELL ESSEX CO AP",
        "state": "NJ",
        "lat": 40.8764,
        "lon": -74.2828,
        "distance_mi": 85.28,
        "bearing": 35.34,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00054789": {
        "name": "MT POCONO POCONO MOUNTAINS MUN",
        "state": "PA",
        "lat": 41.1369,
        "lon": -75.3772,
        "distance_mi": 87.66,
        "bearing": 354.88,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00014768": {
        "name": "HARRISBURG INTL AP",
        "state": "PA",
        "lat": 40.2169,
        "lon": -76.8514,
        "distance_mi": 89.14,
        "bearing": 285.97,
        "ring": "Ring2_Regional",
        "sector": "W",
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
    "USW00094741": {
        "name": "TETERBORO AP",
        "state": "NJ",
        "lat": 40.8589,
        "lon": -74.0561,
        "distance_mi": 91.85,
        "bearing": 41.77,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00094789": {
        "name": "JFK INTL AP",
        "state": "NY",
        "lat": 40.6392,
        "lon": -73.7639,
        "distance_mi": 93.55,
        "bearing": 55.08,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00014732": {
        "name": "LAGUARDIA AP",
        "state": "NY",
        "lat": 40.7794,
        "lon": -73.8803,
        "distance_mi": 94.61,
        "bearing": 48.13,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00054793": {
        "name": "SUSSEX AP",
        "state": "NJ",
        "lat": 41.1992,
        "lon": -74.6258,
        "distance_mi": 96.90,
        "bearing": 18.82,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    # --- Ring 3: Extended (100-150 mi) --- 18 stations
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
    "USW00093738": {
        "name": "SALISBURY-WICOMICO AP",
        "state": "MD",
        "lat": 38.3392,
        "lon": -75.5106,
        "distance_mi": 107.08,
        "bearing": 188.26,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00093786": {
        "name": "OCEAN CITY MUNI AP",
        "state": "MD",
        "lat": 38.3092,
        "lon": -75.1233,
        "distance_mi": 108.21,
        "bearing": 177.02,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00014770": {
        "name": "SELINSGROVE PENN VALLEY AP",
        "state": "PA",
        "lat": 40.8192,
        "lon": -76.8658,
        "distance_mi": 108.25,
        "bearing": 307.66,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    "USW00054787": {
        "name": "FARMINGDALE REPUBLIC AP",
        "state": "NY",
        "lat": 40.7344,
        "lon": -73.4164,
        "distance_mi": 112.43,
        "bearing": 57.47,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00094745": {
        "name": "WESTCHESTER CO AP",
        "state": "NY",
        "lat": 41.0622,
        "lon": -73.7044,
        "distance_mi": 114.68,
        "bearing": 43.76,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00093720": {
        "name": "REAGAN NATIONAL AP",
        "state": "DC",
        "lat": 38.8481,
        "lon": -77.0344,
        "distance_mi": 119.75,
        "bearing": 234.31,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00004789": {
        "name": "MONTGOMERY ORANGE CO AP",
        "state": "NY",
        "lat": 41.5092,
        "lon": -74.2644,
        "distance_mi": 123.77,
        "bearing": 23.73,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00013711": {
        "name": "PATUXENT RIVER NAS",
        "state": "MD",
        "lat": 38.2878,
        "lon": -76.4089,
        "distance_mi": 126.57,
        "bearing": 210.43,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00054746": {
        "name": "MONTICELLO SULLIVAN",
        "state": "NY",
        "lat": 41.7014,
        "lon": -74.7950,
        "distance_mi": 128.32,
        "bearing": 10.00,
        "ring": "Ring3_Extended",
        "sector": "N",
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
    "USW00014751": {
        "name": "WILLIAMSPORT RGNL AP",
        "state": "PA",
        "lat": 41.2431,
        "lon": -76.9219,
        "distance_mi": 129.90,
        "bearing": 317.32,
        "ring": "Ring3_Extended",
        "sector": "NW",
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
    "USW00013740": {
        "name": "DULLES INTL AP",
        "state": "VA",
        "lat": 38.9350,
        "lon": -77.4472,
        "distance_mi": 135.10,
        "bearing": 242.03,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00054734": {
        "name": "DANBURY MUNI AP",
        "state": "CT",
        "lat": 41.3722,
        "lon": -73.4833,
        "distance_mi": 138.15,
        "bearing": 40.88,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00014757": {
        "name": "POUGHKEEPSIE AP",
        "state": "NY",
        "lat": 41.6258,
        "lon": -73.8817,
        "distance_mi": 140.07,
        "bearing": 29.74,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00054790": {
        "name": "SHIRLEY BROOKHAVEN AP",
        "state": "NY",
        "lat": 40.8211,
        "lon": -72.8675,
        "distance_mi": 140.44,
        "bearing": 61.45,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00094702": {
        "name": "IGOR I SIKORSKY MEM AP",
        "state": "CT",
        "lat": 41.1642,
        "lon": -73.1267,
        "distance_mi": 141.85,
        "bearing": 50.36,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    # --- Ring 4: Far (150-250 mi) --- 5 stations
    "USW00014758": {
        "name": "NEW HAVEN TWEED AP",
        "state": "CT",
        "lat": 41.2589,
        "lon": -72.8892,
        "distance_mi": 155.62,
        "bearing": 51.28,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
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
