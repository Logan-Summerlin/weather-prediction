"""
Expanded Configuration for Austin Temperature Prediction.

Station network centered on Austin-Bergstrom International Airport (USW00013904).
Organized by distance ring and compass sector from AUS.

Ring Classification (distance from AUS):
    Ring1_Near:      0 - 50 miles
    Ring2_Regional:  50 - 100 miles
    Ring3_Extended:  100 - 150 miles
    Ring4_Far:       150 - 200 miles

Sector Classification (compass bearing from AUS):
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
CITY_CONFIG = get_city_config("aus")

TARGET_STATION = CITY_CONFIG.target_station
TARGET_LAT = CITY_CONFIG.target_lat
TARGET_LON = CITY_CONFIG.target_lon
TARGET_VARIABLE = "TMAX"

# ==============================================================================
# Surrounding Stations (~50 stations within 200 miles)
# ==============================================================================
# Format: {station_id: "Name, State (distance sector)"}
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 7 stations
    "USW00013958": "AUSTIN-CAMP MABRY, TX (10mi NW)",
    "USC00412585": "DRIPPING SPRINGS 6 E, TX (19mi W)",
    "USC00417983": "SAN MARCOS, TX (27mi SW)",
    "USC00413507": "GEORGETOWN LAKE, TX (33mi N)",
    "USC00415429": "LULING, TX (36mi S)",
    "USC00411429": "CANYON DAM, TX (39mi SW)",
    "USC00414605": "JOHNSON CITY 2N, TX (45mi W)",
    # --- Ring 2: Regional (50-100 mi) --- 16 stations
    "USC00411250": "BURNET, TX (51mi NW)",
    "USW00012911": "RANDOLPH AFB, TX (58mi SW)",
    "USC00414792": "KILLEEN, TX (60mi N)",
    "USW00003902": "ROBERT GRAY AAF, TX (61mi N)",
    "USC00416368": "NIXON, TX (63mi S)",
    "USW00003933": "FT HOOD, TX (65mi N)",
    "USW00012921": "SAN ANTONIO INTL AP, TX (66mi SW)",
    "USC00413873": "HALLETTSVILLE 2 N, TX (66mi SE)",
    "USC00418446": "SOMERVILLE DAM, TX (68mi E)",
    "USC00410902": "BOERNE, TX (69mi SW)",
    "USC00415272": "LLANO, TX (70mi NW)",
    "USC00419952": "YOAKUM, TX (71mi SE)",
    "USC00413329": "FREDERICKSBURG, TX (74mi W)",
    "USC00411911": "COLUMBUS, TX (74mi SE)",
    "USW00012909": "SAN ANTONIO KELLY AFB, TX (78mi SW)",
    "USW00003904": "COLLEGE STN EASTERWOOD FLD, TX (83mi E)",
    # --- Ring 3: Extended (100-150 mi) --- 9 stations
    "USW00013959": "WACO RGNL AP, TX (102mi N)",
    "USW00012912": "VICTORIA RGNL AP, TX (102mi SE)",
    "USW00013928": "WACO J CONNALLY AFB, TX (106mi N)",
    "USW00012962": "HONDO MUNI AP, TX (107mi SW)",
    "USW00013973": "JUNCTION KIMBLE CO AP, TX (127mi W)",
    "USW00012935": "PALACIOS MUNI AP, TX (133mi SE)",
    "USW00012960": "HOUSTON INTERCONTINENTAL AP, TX (139mi E)",
    "USW00003969": "STEPHENVILLE CLARK FLD, TX (143mi N)",
    "USW00012918": "HOUSTON WILLIAM P HOBBY AP, TX (148mi E)",
    # --- Ring 4: Far (150-200 mi) --- 18 stations
    "USW00012947": "COTULLA LA SALLE CO AP, TX (152mi SW)",
    "USW00012906": "HOUSTON ELLINGTON AFB, TX (155mi E)",
    "USW00093914": "PALESTINE 2 NE, TX (164mi NE)",
    "USW00012924": "CORPUS CHRISTI, TX (167mi S)",
    "USW00012932": "ALICE INTL AP, TX (171mi S)",
    "USW00012926": "CORPUS CHRISTI NAS, TX (175mi S)",
    "USW00013911": "FT WORTH NAS, TX (178mi N)",
    "USW00012923": "GALVESTON SCHOLES FLD, TX (180mi E)",
    "USW00093985": "MINERAL WELLS AP, TX (180mi N)",
    "USW00093904": "FT WORTH MEACHAM FLD NAAF, TX (182mi N)",
    "USW00013961": "FT WORTH MEACHAM FLD, TX (183mi N)",
    "USW00012928": "KINGSVILLE NAAS, TX (186mi S)",
    "USW00023034": "SAN ANGELO, TX (186mi NW)",
    "USW00093987": "LUFKIN ANGELINA CO AP, TX (188mi NE)",
    "USW00013960": "DALLAS FAA AP, TX (189mi N)",
    "USW00003927": "DAL-FTW WSCMO AP, TX (191mi N)",
    "USW00013962": "ABILENE RGNL AP, TX (194mi NW)",
    "USW00022001": "DEL RIO LAUGHLIN AFB, TX (195mi W)",
}

# All stations (target + surrounding) for convenience
ALL_STATIONS = {
    TARGET_STATION: "Austin-Bergstrom International Airport (Target)",
    **SURROUNDING_STATIONS,
}

# ==============================================================================
# ASOS/AWOS Mapping (operational station IDs)
# ==============================================================================
ASOS_STATION_MAP = {
    "USW00013904": "KAUS",   # Austin-Bergstrom
    "USW00013958": "KATT",   # Camp Mabry
    "USW00012911": "KRND",   # Randolph AFB
    "USW00003902": "KGRK",   # Robert Gray AAF / Ft Hood
    "USW00003933": "KHLR",   # Ft Hood (Hood AAF)
    "USW00012921": "KSAT",   # San Antonio Intl
    "USW00012909": "KSKF",   # San Antonio Kelly AFB
    "USW00003904": "KCLL",   # College Station Easterwood
    "USW00013959": "KACT",   # Waco Regional
    "USW00012912": "KVCT",   # Victoria Regional
    "USW00013928": "KCNW",   # Waco Connally
    "USW00012962": "KHDO",   # Hondo Muni
    "USW00013973": "KJCT",   # Junction Kimble Co
    "USW00012935": "KPSX",   # Palacios Muni
    "USW00012960": "KIAH",   # Houston Intercontinental
    "USW00003969": "KSEP",   # Stephenville Clark
    "USW00012918": "KHOU",   # Houston Hobby
    "USW00012947": "KCOT",   # Cotulla
    "USW00012906": "KEFD",   # Houston Ellington
    "USW00012924": "KCRP",   # Corpus Christi
    "USW00012932": "KALI",   # Alice Intl
    "USW00012926": "KNGP",   # Corpus Christi NAS
    "USW00013911": "KNFW",   # Ft Worth NAS
    "USW00012923": "KGLS",   # Galveston Scholes
    "USW00093985": "KMWL",   # Mineral Wells
    "USW00093904": "KFTW",   # Ft Worth Meacham
    "USW00013961": "KFTW",   # Ft Worth Meacham (same ICAO)
    "USW00012928": "KNQI",   # Kingsville NAAS
    "USW00023034": "KSJT",   # San Angelo
    "USW00093987": "KLFK",   # Lufkin Angelina Co
    "USW00013960": "KDAL",   # Dallas FAA / Love Field
    "USW00003927": "KDFW",   # DFW WSCMO
    "USW00013962": "KABI",   # Abilene Regional
    "USW00022001": "KDLF",   # Del Rio Laughlin AFB
    "USW00093914": "KPSN",   # Palestine
}

NON_ASOS_STATIONS = {
    "USC00412585": "DRIPPING SPRINGS 6 E (cooperative)",
    "USC00417983": "SAN MARCOS (cooperative)",
    "USC00413507": "GEORGETOWN LAKE (cooperative)",
    "USC00415429": "LULING (cooperative)",
    "USC00411429": "CANYON DAM (cooperative)",
    "USC00414605": "JOHNSON CITY 2N (cooperative)",
    "USC00411250": "BURNET (cooperative)",
    "USC00414792": "KILLEEN (cooperative)",
    "USC00416368": "NIXON (cooperative)",
    "USC00413873": "HALLETTSVILLE 2 N (cooperative)",
    "USC00418446": "SOMERVILLE DAM (cooperative)",
    "USC00410902": "BOERNE (cooperative)",
    "USC00415272": "LLANO (cooperative)",
    "USC00419952": "YOAKUM (cooperative)",
    "USC00413329": "FREDERICKSBURG (cooperative)",
    "USC00411911": "COLUMBUS (cooperative)",
}

# ==============================================================================
# Data Quality
# ==============================================================================
MIN_COMPLETENESS = 0.80

# ==============================================================================
# Station Rings (distance classification)
# ==============================================================================
STATION_RINGS = {
    "Ring1_Near": [
        "USW00013958",
        "USC00412585",
        "USC00417983",
        "USC00413507",
        "USC00415429",
        "USC00411429",
        "USC00414605",
    ],
    "Ring2_Regional": [
        "USC00411250",
        "USW00012911",
        "USC00414792",
        "USW00003902",
        "USC00416368",
        "USW00003933",
        "USW00012921",
        "USC00413873",
        "USC00418446",
        "USC00410902",
        "USC00415272",
        "USC00419952",
        "USC00413329",
        "USC00411911",
        "USW00012909",
        "USW00003904",
    ],
    "Ring3_Extended": [
        "USW00013959",
        "USW00012912",
        "USW00013928",
        "USW00012962",
        "USW00013973",
        "USW00012935",
        "USW00012960",
        "USW00003969",
        "USW00012918",
    ],
    "Ring4_Far": [
        "USW00012947",
        "USW00012906",
        "USW00093914",
        "USW00012924",
        "USW00012932",
        "USW00012926",
        "USW00013911",
        "USW00012923",
        "USW00093985",
        "USW00093904",
        "USW00013961",
        "USW00012928",
        "USW00023034",
        "USW00093987",
        "USW00013960",
        "USW00003927",
        "USW00013962",
        "USW00022001",
    ],
}

# ==============================================================================
# Station Sectors (compass direction classification)
# ==============================================================================
STATION_SECTORS = {
    "N": [
        "USC00413507",
        "USC00414792",
        "USW00003902",
        "USW00003933",
        "USW00013959",
        "USW00013928",
        "USW00003969",
        "USW00013911",
        "USW00093985",
        "USW00093904",
        "USW00013961",
        "USW00013960",
        "USW00003927",
    ],
    "NE": [
        "USW00093914",
        "USW00093987",
    ],
    "E": [
        "USC00418446",
        "USW00003904",
        "USW00012960",
        "USW00012918",
        "USW00012906",
        "USW00012923",
    ],
    "SE": [
        "USC00413873",
        "USC00419952",
        "USC00411911",
        "USW00012912",
        "USW00012935",
    ],
    "S": [
        "USC00415429",
        "USC00416368",
        "USW00012924",
        "USW00012932",
        "USW00012926",
        "USW00012928",
    ],
    "SW": [
        "USC00417983",
        "USC00411429",
        "USW00012911",
        "USW00012921",
        "USC00410902",
        "USW00012909",
        "USW00012962",
        "USW00012947",
    ],
    "W": [
        "USC00412585",
        "USC00414605",
        "USC00413329",
        "USW00013973",
        "USW00022001",
    ],
    "NW": [
        "USW00013958",
        "USC00411250",
        "USC00415272",
        "USW00023034",
        "USW00013962",
    ],
}

# ==============================================================================
# Meteorological Sector Assignments (for feature engineering)
# ==============================================================================
# Austin's weather is dominated by Gulf moisture from the SE/E,
# dryline influence from the W/NW (Edwards Plateau),
# warm advection from the S/SW (Rio Grande / Mexico),
# and northerly cold fronts (Blue Northers) from the N.
METEOROLOGICAL_SECTORS = {
    "Dryline_W": (  # Upstream dry-air advection: W and NW sectors (Edwards Plateau / Hill Country)
        STATION_SECTORS["W"] + STATION_SECTORS["NW"]
    ),
    "Gulf_E": (  # Gulf of Mexico moisture: E and SE sectors
        STATION_SECTORS["E"] + STATION_SECTORS["SE"]
    ),
    "Warm_S": (  # Warm advection from south Texas / Mexico: S and SW sectors
        STATION_SECTORS["S"] + STATION_SECTORS["SW"]
    ),
    "NearField": (  # Local / urban: all Ring 1 stations
        STATION_RINGS["Ring1_Near"]
    ),
    "Norther_N": (  # Cold-front advection: N and NE sectors
        STATION_SECTORS["N"] + STATION_SECTORS["NE"]
    ),
}

# ==============================================================================
# Station Metadata
# ==============================================================================
STATION_METADATA = {
    "USW00013958": {
        "name": "AUSTIN-CAMP MABRY",
        "state": "TX",
        "lat": 30.3208,
        "lon": -97.7603,
        "distance_mi": 10.26,
        "bearing": 328.35,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    "USC00412585": {
        "name": "DRIPPING SPRINGS 6 E",
        "state": "TX",
        "lat": 30.2194,
        "lon": -97.9878,
        "distance_mi": 19.05,
        "bearing": 275.28,
        "ring": "Ring1_Near",
        "sector": "W",
    },
    "USC00417983": {
        "name": "SAN MARCOS",
        "state": "TX",
        "lat": 29.8833,
        "lon": -97.9494,
        "distance_mi": 27.23,
        "bearing": 217.93,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USC00413507": {
        "name": "GEORGETOWN LAKE",
        "state": "TX",
        "lat": 30.6764,
        "lon": -97.7208,
        "distance_mi": 33.44,
        "bearing": 354.82,
        "ring": "Ring1_Near",
        "sector": "N",
    },
    "USC00415429": {
        "name": "LULING",
        "state": "TX",
        "lat": 29.6756,
        "lon": -97.6578,
        "distance_mi": 35.85,
        "bearing": 178.83,
        "ring": "Ring1_Near",
        "sector": "S",
    },
    "USC00411429": {
        "name": "CANYON DAM",
        "state": "TX",
        "lat": 29.8608,
        "lon": -98.1958,
        "distance_mi": 39.00,
        "bearing": 233.90,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USC00414605": {
        "name": "JOHNSON CITY 2N",
        "state": "TX",
        "lat": 30.3,
        "lon": -98.4094,
        "distance_mi": 44.73,
        "bearing": 279.57,
        "ring": "Ring1_Near",
        "sector": "W",
    },
    "USC00411250": {
        "name": "BURNET",
        "state": "TX",
        "lat": 30.7586,
        "lon": -98.2339,
        "distance_mi": 51.45,
        "bearing": 319.40,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USW00012911": {
        "name": "RANDOLPH AFB",
        "state": "TX",
        "lat": 29.5325,
        "lon": -98.2622,
        "distance_mi": 57.88,
        "bearing": 217.96,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USC00414792": {
        "name": "KILLEEN",
        "state": "TX",
        "lat": 31.0658,
        "lon": -97.6919,
        "distance_mi": 60.22,
        "bearing": 358.77,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00003902": {
        "name": "ROBERT GRAY AAF",
        "state": "TX",
        "lat": 31.0667,
        "lon": -97.8333,
        "distance_mi": 61.05,
        "bearing": 350.89,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USC00416368": {
        "name": "NIXON",
        "state": "TX",
        "lat": 29.2828,
        "lon": -97.7675,
        "distance_mi": 63.26,
        "bearing": 185.33,
        "ring": "Ring2_Regional",
        "sector": "S",
    },
    "USW00003933": {
        "name": "FT HOOD",
        "state": "TX",
        "lat": 31.1333,
        "lon": -97.7167,
        "distance_mi": 64.93,
        "bearing": 357.56,
        "ring": "Ring2_Regional",
        "sector": "N",
    },
    "USW00012921": {
        "name": "SAN ANTONIO INTL AP",
        "state": "TX",
        "lat": 29.5442,
        "lon": -98.4839,
        "distance_mi": 66.30,
        "bearing": 227.55,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USC00413873": {
        "name": "HALLETTSVILLE 2 N",
        "state": "TX",
        "lat": 29.4706,
        "lon": -96.9397,
        "distance_mi": 66.46,
        "bearing": 138.62,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USC00418446": {
        "name": "SOMERVILLE DAM",
        "state": "TX",
        "lat": 30.3367,
        "lon": -96.5403,
        "distance_mi": 68.13,
        "bearing": 81.42,
        "ring": "Ring2_Regional",
        "sector": "E",
    },
    "USC00410902": {
        "name": "BOERNE",
        "state": "TX",
        "lat": 29.7986,
        "lon": -98.7353,
        "distance_mi": 69.36,
        "bearing": 247.05,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USC00415272": {
        "name": "LLANO",
        "state": "TX",
        "lat": 30.7425,
        "lon": -98.6542,
        "distance_mi": 69.78,
        "bearing": 303.12,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USC00419952": {
        "name": "YOAKUM",
        "state": "TX",
        "lat": 29.2739,
        "lon": -97.1556,
        "distance_mi": 70.69,
        "bearing": 153.99,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USC00413329": {
        "name": "FREDERICKSBURG",
        "state": "TX",
        "lat": 30.2392,
        "lon": -98.9089,
        "distance_mi": 74.03,
        "bearing": 272.71,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USC00411911": {
        "name": "COLUMBUS",
        "state": "TX",
        "lat": 29.6989,
        "lon": -96.5731,
        "distance_mi": 74.06,
        "bearing": 117.26,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USW00012909": {
        "name": "SAN ANTONIO KELLY AFB",
        "state": "TX",
        "lat": 29.3833,
        "lon": -98.5833,
        "distance_mi": 78.36,
        "bearing": 224.57,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00003904": {
        "name": "COLLEGE STN EASTERWOOD FLD",
        "state": "TX",
        "lat": 30.5911,
        "lon": -96.3631,
        "distance_mi": 82.57,
        "bearing": 70.28,
        "ring": "Ring2_Regional",
        "sector": "E",
    },
    "USW00013959": {
        "name": "WACO RGNL AP",
        "state": "TX",
        "lat": 31.6181,
        "lon": -97.2283,
        "distance_mi": 101.79,
        "bearing": 14.79,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00012912": {
        "name": "VICTORIA RGNL AP",
        "state": "TX",
        "lat": 28.8625,
        "lon": -96.93,
        "distance_mi": 102.21,
        "bearing": 154.02,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    "USW00013928": {
        "name": "WACO J CONNALLY AFB",
        "state": "TX",
        "lat": 31.6333,
        "lon": -97.0667,
        "distance_mi": 105.66,
        "bearing": 19.63,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00012962": {
        "name": "HONDO MUNI AP",
        "state": "TX",
        "lat": 29.36,
        "lon": -99.1742,
        "distance_mi": 107.06,
        "bearing": 237.79,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00013973": {
        "name": "JUNCTION KIMBLE CO AP",
        "state": "TX",
        "lat": 30.5106,
        "lon": -99.7664,
        "distance_mi": 126.89,
        "bearing": 280.44,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00012935": {
        "name": "PALACIOS MUNI AP",
        "state": "TX",
        "lat": 28.7247,
        "lon": -96.2536,
        "distance_mi": 132.56,
        "bearing": 139.65,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    "USW00012960": {
        "name": "HOUSTON INTERCONTINENTAL AP",
        "state": "TX",
        "lat": 29.9844,
        "lon": -95.3608,
        "distance_mi": 138.81,
        "bearing": 95.42,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00003969": {
        "name": "STEPHENVILLE CLARK FLD",
        "state": "TX",
        "lat": 32.2153,
        "lon": -98.1775,
        "distance_mi": 142.82,
        "bearing": 348.01,
        "ring": "Ring3_Extended",
        "sector": "N",
    },
    "USW00012918": {
        "name": "HOUSTON WILLIAM P HOBBY AP",
        "state": "TX",
        "lat": 29.6458,
        "lon": -95.2822,
        "distance_mi": 147.93,
        "bearing": 104.25,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00012947": {
        "name": "COTULLA LA SALLE CO AP",
        "state": "TX",
        "lat": 28.4586,
        "lon": -99.2228,
        "distance_mi": 152.09,
        "bearing": 218.33,
        "ring": "Ring4_Far",
        "sector": "SW",
    },
    "USW00012906": {
        "name": "HOUSTON ELLINGTON AFB",
        "state": "TX",
        "lat": 29.6167,
        "lon": -95.1667,
        "distance_mi": 155.15,
        "bearing": 104.28,
        "ring": "Ring4_Far",
        "sector": "E",
    },
    "USW00093914": {
        "name": "PALESTINE 2 NE",
        "state": "TX",
        "lat": 31.7831,
        "lon": -95.6039,
        "distance_mi": 164.39,
        "bearing": 47.58,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00012924": {
        "name": "CORPUS CHRISTI",
        "state": "TX",
        "lat": 27.7839,
        "lon": -97.5114,
        "distance_mi": 166.83,
        "bearing": 176.67,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00012932": {
        "name": "ALICE INTL AP",
        "state": "TX",
        "lat": 27.7414,
        "lon": -98.025,
        "distance_mi": 170.84,
        "bearing": 187.30,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00012926": {
        "name": "CORPUS CHRISTI NAS",
        "state": "TX",
        "lat": 27.6878,
        "lon": -97.2917,
        "distance_mi": 174.69,
        "bearing": 172.38,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00013911": {
        "name": "FT WORTH NAS",
        "state": "TX",
        "lat": 32.7667,
        "lon": -97.45,
        "distance_mi": 178.20,
        "bearing": 4.11,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00012923": {
        "name": "GALVESTON SCHOLES FLD",
        "state": "TX",
        "lat": 29.2703,
        "lon": -94.8642,
        "distance_mi": 180.04,
        "bearing": 110.07,
        "ring": "Ring4_Far",
        "sector": "E",
    },
    "USW00093985": {
        "name": "MINERAL WELLS AP",
        "state": "TX",
        "lat": 32.7817,
        "lon": -98.0603,
        "distance_mi": 180.24,
        "bearing": 352.77,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00093904": {
        "name": "FT WORTH MEACHAM FLD NAAF",
        "state": "TX",
        "lat": 32.8167,
        "lon": -97.35,
        "distance_mi": 182.16,
        "bearing": 5.86,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00013961": {
        "name": "FT WORTH MEACHAM FLD",
        "state": "TX",
        "lat": 32.8247,
        "lon": -97.3642,
        "distance_mi": 182.63,
        "bearing": 5.58,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00012928": {
        "name": "KINGSVILLE NAAS",
        "state": "TX",
        "lat": 27.5089,
        "lon": -97.8042,
        "distance_mi": 185.73,
        "bearing": 182.54,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00023034": {
        "name": "SAN ANGELO",
        "state": "TX",
        "lat": 31.3714,
        "lon": -100.4925,
        "distance_mi": 186.23,
        "bearing": 296.61,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00093987": {
        "name": "LUFKIN ANGELINA CO AP",
        "state": "TX",
        "lat": 31.2358,
        "lon": -94.7547,
        "distance_mi": 187.52,
        "bearing": 66.70,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00013960": {
        "name": "DALLAS FAA AP",
        "state": "TX",
        "lat": 32.8383,
        "lon": -96.8358,
        "distance_mi": 189.17,
        "bearing": 14.84,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00003927": {
        "name": "DAL-FTW WSCMO AP",
        "state": "TX",
        "lat": 32.8975,
        "lon": -97.0219,
        "distance_mi": 190.63,
        "bearing": 11.38,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00013962": {
        "name": "ABILENE RGNL AP",
        "state": "TX",
        "lat": 32.4106,
        "lon": -99.6819,
        "distance_mi": 193.78,
        "bearing": 322.72,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00022001": {
        "name": "DEL RIO LAUGHLIN AFB",
        "state": "TX",
        "lat": 29.3667,
        "lon": -100.7833,
        "distance_mi": 195.26,
        "bearing": 253.75,
        "ring": "Ring4_Far",
        "sector": "W",
    },
}

# ==============================================================================
# Pipeline Constants (used by preprocessing and benchmark scripts)
# ==============================================================================
START_DATE = "1985-01-01"
END_DATE = "2024-12-31"

INPUT_VARIABLES = ["TMAX", "TMIN"]
MAX_FORWARD_FILL_DAYS = 3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZES = [64, 32]
DROPOUT = 0.0

BUCKET_EDGES = list(CITY_CONFIG.bucket_edges)
BUCKET_LABELS = list(CITY_CONFIG.bucket_labels)
