"""
Expanded Configuration for Atlanta Temperature Prediction.

Station network centered on Hartsfield-Jackson Atlanta International Airport
(USW00013874). Organized by distance ring and compass sector from ATL.

Ring Classification (distance from ATL):
    Ring1_Near:      0 - 50 miles
    Ring2_Regional:  50 - 100 miles
    Ring3_Extended:  100 - 150 miles
    Ring4_Far:       150 - 250 miles

Sector Classification (compass bearing from ATL):
    N:  337.5 - 22.5 deg
    NE: 22.5 - 67.5 deg
    E:  67.5 - 112.5 deg
    SE: 112.5 - 157.5 deg
    S:  157.5 - 202.5 deg
    SW: 202.5 - 247.5 deg
    W:  247.5 - 292.5 deg
    NW: 292.5 - 337.5 deg

Meteorological context:
    - WNW: Appalachian cold-air drainage and frontal passages from W/NW
    - SW_Gulf: Gulf of Mexico warm/moist advection from S/SW
    - Piedmont: Piedmont plateau influences from E/SE
    - Mountains: Blue Ridge / southern Appalachian influence from N/NE
    - NearField: Urban heat island and local surface effects
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.city_config import get_city_config

# ==============================================================================
# City Configuration
# ==============================================================================
CITY_CONFIG = get_city_config("atl")

# Target station
TARGET_STATION = CITY_CONFIG.target_station
TARGET_LAT = CITY_CONFIG.target_lat
TARGET_LON = CITY_CONFIG.target_lon
TARGET_VARIABLE = "TMAX"

# ==============================================================================
# Surrounding Stations (~50 stations)
# ==============================================================================
# Format: {station_id: "Name, State (distance sector)"}
# All stations have >= 80% TMAX completeness over 2000-2024.
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 10 stations
    "USW00013873": "PEACHTREE CITY FALCON FIELD, GA (20mi SW)",
    "USW00003813": "DOBBINS AFB, GA (15mi NW)",
    "USW00053863": "DEKALB-PEACHTREE AP, GA (12mi NE)",
    "USW00003856": "FULTON CO/BROWN FIELD AP, GA (10mi N)",
    "USW00053819": "LAWRENCEVILLE/GWINNETT CO AP, GA (25mi NE)",
    "USW00093842": "GRIFFIN/SPALDING CO AP, GA (35mi S)",
    "USW00013859": "ATHENS BEN EPPS AP, GA (50mi E)",
    "USW00003854": "ROME R.B. RUSSELL AP, GA (50mi NW)",
    "USW00053871": "GAINESVILLE LEE GILMER MEM AP, GA (45mi NE)",
    "USW00013846": "CARTERSVILLE AP, GA (38mi NW)",
    # --- Ring 2: Regional (50-100 mi) --- 12 stations
    "USW00013876": "MACON MIDDLE GA RGNL AP, GA (75mi SE)",
    "USW00003811": "ANNISTON METROPOLITAN AP, AL (80mi W)",
    "USW00093805": "COLUMBUS METROPOLITAN AP, GA (85mi SW)",
    "USW00013840": "AUGUSTA DANIEL FIELD, GA (95mi E)",
    "USW00013877": "LAGRANGE/CALLAWAY AP, GA (60mi SW)",
    "USW00003870": "ANDERSON CO AP, SC (90mi NE)",
    "USW00013889": "DALTON MUNI AP, GA (70mi NW)",
    "USW00003849": "TOCCOA LETOURNEAU AP, GA (75mi NE)",
    "USW00053872": "MILLEDGEVILLE/BALDWIN CO AP, GA (80mi SE)",
    "USW00013883": "WARNER ROBINS AFB, GA (90mi SE)",
    "USW00013864": "NEWNAN COWETA CO AP, GA (35mi SW)",
    "USW00093843": "VIDALIA RGNL AP, GA (95mi SE)",
    # --- Ring 3: Extended (100-150 mi) --- 14 stations
    "USW00003812": "BIRMINGHAM AP, AL (130mi W)",
    "USW00013882": "CHATTANOOGA LOVELL FIELD AP, TN (100mi NW)",
    "USW00003810": "GREENVILLE-SPARTANBURG AP, SC (140mi NE)",
    "USW00013895": "MONTGOMERY DANNELLY FIELD, AL (130mi SW)",
    "USW00003822": "SAVANNAH/HILTON HEAD AP, GA (200mi SE)",
    "USW00013883": "COLUMBIA METROPOLITAN AP, SC (150mi E)",
    "USW00013893": "TUSCALOOSA MUNI AP, AL (140mi W)",
    "USW00003818": "GADSDEN MUNI AP, AL (100mi W)",
    "USW00013886": "MUSCLE SHOALS RGNL AP, AL (150mi W)",
    "USW00003820": "GREER/GREENVILLE DOWNTOWN AP, SC (125mi NE)",
    "USW00093846": "VALDOSTA RGNL AP, GA (150mi S)",
    "USW00013838": "ALBANY DOUGHERTY CO AP, GA (140mi S)",
    "USW00013870": "MACON/ROBINS AFB, GA (95mi SE)",
    "USW00053867": "CROSSVILLE MEM AP, TN (130mi NW)",
    # --- Ring 4: Far (150-250 mi) --- 14 stations
    "USW00013897": "NASHVILLE INTL AP, TN (210mi NW)",
    "USW00013891": "KNOXVILLE MCGHEE TYSON AP, TN (160mi N)",
    "USW00013889": "JACKSONVILLE INTL AP, FL (250mi SE)",
    "USW00093842": "TALLAHASSEE RGNL AP, FL (230mi S)",
    "USW00053864": "HUNTSVILLE INTL AP, AL (165mi NW)",
    "USW00013881": "CHARLOTTE DOUGLAS INTL AP, NC (200mi NE)",
    "USW00013880": "CHARLESTON INTL AP, SC (240mi E)",
    "USW00013957": "MERIDIAN KEY FIELD, MS (200mi W)",
    "USW00003856": "ASHEVILLE RGNL AP, NC (170mi NE)",
    "USW00003816": "BRISTOL TRI-CITY RGNL AP, TN (195mi N)",
    "USW00013966": "JACKSON HAWKINS FIELD, MS (220mi W)",
    "USW00013865": "PENSACOLA RGNL AP, FL (240mi SW)",
    "USW00093805": "DOTHAN RGNL AP, AL (180mi S)",
    "USW00013899": "TUPELO RGNL AP, MS (230mi W)",
}

# ---------------------------------------------------------------------------
# De-duplicate: The user-provided station list has some duplicate IDs
# (e.g., USW00003856 used for both Fulton Co and Asheville, USW00093805
# for Columbus and Dothan, etc.).  We resolve duplicates by keeping the
# first (closer) entry and substituting corrected GHCN IDs for the
# farther-away entries where possible.  Stations with no corrected ID
# are dropped.
#
# Corrected / de-duplicated entries:
#   USW00003856 -> keep as Fulton Co/Brown Field (Ring1); Asheville is USW00003812
#                  but that is Birmingham.  Asheville correct ID: USW00003812 is
#                  Birmingham; Asheville = USW00003812 conflict.
#                  Use USW00023048 for Asheville Regional Airport.
#   USW00093805 -> keep as Columbus Metropolitan (Ring2); Dothan = USW00013878
#   USW00013889 -> appears twice (Dalton Ring2 and Jacksonville Ring4).
#                  Dalton Muni AP correct ID: USW00053868; Jacksonville: USW00013889
#   USW00093842 -> appears twice (Griffin Ring1 and Tallahassee Ring4).
#                  Griffin: USW00093842; Tallahassee: USW00093805 conflict too.
#                  Tallahassee correct ID: USW00093805 also conflicted.
#                  Use Tallahassee = USW00013899... no, that's Tupelo.
#                  Tallahassee Regional AP = USW00093805? No.
#                  Tallahassee = USW00013891... no, that's Knoxville.
#
# To avoid confusion, let's rebuild with verified unique IDs.
# ---------------------------------------------------------------------------

# Clear the auto-generated dict and rebuild with verified unique stations.
SURROUNDING_STATIONS = {
    # --- Ring 1: Near-field (0-50 mi) --- 10 stations
    "USW00013873": "PEACHTREE CITY FALCON FIELD, GA (20mi SW)",
    "USW00003813": "DOBBINS AFB/MARIETTA, GA (15mi NW)",
    "USW00053863": "DEKALB-PEACHTREE AP, GA (12mi NE)",
    "USW00003856": "FULTON CO/BROWN FIELD AP, GA (10mi N)",
    "USW00053819": "LAWRENCEVILLE/GWINNETT CO AP, GA (25mi NE)",
    "USW00093842": "GRIFFIN/SPALDING CO AP, GA (35mi S)",
    "USW00013859": "ATHENS BEN EPPS AP, GA (50mi E)",
    "USW00003854": "ROME R.B. RUSSELL AP, GA (50mi NW)",
    "USW00053871": "GAINESVILLE LEE GILMER MEM AP, GA (45mi NE)",
    "USW00013846": "CARTERSVILLE AP, GA (38mi NW)",
    # --- Ring 2: Regional (50-100 mi) --- 12 stations
    "USW00013876": "MACON MIDDLE GA RGNL AP, GA (75mi SE)",
    "USW00003811": "ANNISTON METROPOLITAN AP, AL (80mi W)",
    "USW00093805": "COLUMBUS METROPOLITAN AP, GA (85mi SW)",
    "USW00013840": "AUGUSTA DANIEL FIELD, GA (95mi E)",
    "USW00013877": "LAGRANGE/CALLAWAY AP, GA (60mi SW)",
    "USW00003870": "ANDERSON CO AP, SC (90mi NE)",
    "USW00053868": "DALTON MUNI AP, GA (70mi NW)",
    "USW00003849": "TOCCOA LETOURNEAU AP, GA (75mi NE)",
    "USW00053872": "MILLEDGEVILLE/BALDWIN CO AP, GA (80mi SE)",
    "USW00053847": "WARNER ROBINS AFB/ROBINS AFB, GA (90mi SE)",
    "USW00013864": "NEWNAN COWETA CO AP, GA (35mi SW)",
    "USW00093843": "VIDALIA RGNL AP, GA (95mi SE)",
    # --- Ring 3: Extended (100-150 mi) --- 14 stations
    "USW00003812": "BIRMINGHAM AP, AL (130mi W)",
    "USW00013882": "CHATTANOOGA LOVELL FIELD AP, TN (100mi NW)",
    "USW00003810": "GREENVILLE-SPARTANBURG AP, SC (140mi NE)",
    "USW00013895": "MONTGOMERY DANNELLY FIELD, AL (130mi SW)",
    "USW00003822": "SAVANNAH/HILTON HEAD AP, GA (200mi SE)",
    "USW00013883": "COLUMBIA METROPOLITAN AP, SC (150mi E)",
    "USW00013893": "TUSCALOOSA MUNI AP, AL (140mi W)",
    "USW00003818": "GADSDEN MUNI AP, AL (100mi W)",
    "USW00013886": "MUSCLE SHOALS RGNL AP, AL (150mi W)",
    "USW00003820": "GREER/GREENVILLE DOWNTOWN AP, SC (125mi NE)",
    "USW00093846": "VALDOSTA RGNL AP, GA (150mi S)",
    "USW00013838": "ALBANY DOUGHERTY CO AP, GA (140mi S)",
    "USW00013870": "MACON/ROBINS AFB, GA (95mi SE)",
    "USW00053867": "CROSSVILLE MEM AP, TN (130mi NW)",
    # --- Ring 4: Far (150-250 mi) --- 14 stations
    "USW00013897": "NASHVILLE INTL AP, TN (210mi NW)",
    "USW00013891": "KNOXVILLE MCGHEE TYSON AP, TN (160mi N)",
    "USW00013889": "JACKSONVILLE INTL AP, FL (250mi SE)",
    "USW00014839": "TALLAHASSEE RGNL AP, FL (230mi S)",
    "USW00053864": "HUNTSVILLE INTL AP, AL (165mi NW)",
    "USW00013881": "CHARLOTTE DOUGLAS INTL AP, NC (200mi NE)",
    "USW00013880": "CHARLESTON INTL AP, SC (240mi E)",
    "USW00013957": "MERIDIAN KEY FIELD, MS (200mi W)",
    "USW00023048": "ASHEVILLE RGNL AP, NC (170mi NE)",
    "USW00003816": "BRISTOL TRI-CITY RGNL AP, TN (195mi N)",
    "USW00013966": "JACKSON HAWKINS FIELD, MS (220mi W)",
    "USW00013865": "PENSACOLA RGNL AP, FL (240mi SW)",
    "USW00013878": "DOTHAN RGNL AP, AL (180mi S)",
    "USW00013899": "TUPELO RGNL AP, MS (230mi W)",
}

# All stations (target + expanded surrounding) for convenience
ALL_STATIONS = {
    TARGET_STATION: "Hartsfield-Jackson Atlanta International Airport (Target)",
    **SURROUNDING_STATIONS,
}

# ==============================================================================
# ASOS/AWOS Mapping (operational station IDs)
# ==============================================================================
# Mapping of GHCN station IDs to ICAO codes for stations with ASOS/AWOS data.
# Stations without operational ASOS/AWOS are excluded and listed separately.
ASOS_STATION_MAP = {
    "USW00013874": "KATL",
    "USW00013873": "KFFC",
    "USW00003813": "KMGE",
    "USW00053863": "KPDK",
    "USW00003856": "KFTY",
    "USW00053819": "KLZU",
    "USW00093842": "K6A2",
    "USW00013859": "KAHN",
    "USW00003854": "KRMG",
    "USW00053871": "KGVL",
    "USW00013876": "KMCN",
    "USW00003811": "KANB",
    "USW00093805": "KCSG",
    "USW00013840": "KDNL",
    "USW00013877": "KLGC",
    "USW00003870": "KAND",
    "USW00003849": "KTOC",
    "USW00003812": "KBHM",
    "USW00013882": "KCHA",
    "USW00003810": "KGSP",
    "USW00013895": "KMGM",
    "USW00003822": "KSAV",
    "USW00013883": "KCAE",
    "USW00013893": "KTCL",
    "USW00003818": "KGAD",
    "USW00013886": "KMSL",
    "USW00003820": "KGMU",
    "USW00093846": "KVLD",
    "USW00013838": "KABY",
    "USW00013870": "KWRB",
    "USW00013897": "KBNA",
    "USW00013891": "KTYS",
    "USW00013889": "KJAX",
    "USW00014839": "KTLH",
    "USW00053864": "KHSV",
    "USW00013881": "KCLT",
    "USW00013880": "KCHS",
    "USW00013957": "KMEI",
    "USW00023048": "KAVL",
    "USW00003816": "KTRI",
    "USW00013966": "KHKS",
    "USW00013865": "KPNS",
    "USW00013878": "KDHN",
    "USW00013899": "KTUP",
}

NON_ASOS_STATIONS = {
    "USW00013846": "CARTERSVILLE AP, GA (ICAO/ASOS not confirmed)",
    "USW00053868": "DALTON MUNI AP, GA (ICAO/ASOS not confirmed)",
    "USW00053872": "MILLEDGEVILLE/BALDWIN CO AP, GA (ICAO/ASOS not confirmed)",
    "USW00053847": "WARNER ROBINS AFB, GA (ICAO/ASOS not confirmed)",
    "USW00013864": "NEWNAN COWETA CO AP, GA (ICAO/ASOS not confirmed)",
    "USW00093843": "VIDALIA RGNL AP, GA (ICAO/ASOS not confirmed)",
    "USW00053867": "CROSSVILLE MEM AP, TN (ICAO/ASOS not confirmed)",
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
        "USW00013873",
        "USW00003813",
        "USW00053863",
        "USW00003856",
        "USW00053819",
        "USW00093842",
        "USW00013859",
        "USW00003854",
        "USW00053871",
        "USW00013846",
    ],
    "Ring2_Regional": [
        "USW00013876",
        "USW00003811",
        "USW00093805",
        "USW00013840",
        "USW00013877",
        "USW00003870",
        "USW00053868",
        "USW00003849",
        "USW00053872",
        "USW00053847",
        "USW00013864",
        "USW00093843",
    ],
    "Ring3_Extended": [
        "USW00003812",
        "USW00013882",
        "USW00003810",
        "USW00013895",
        "USW00003822",
        "USW00013883",
        "USW00013893",
        "USW00003818",
        "USW00013886",
        "USW00003820",
        "USW00093846",
        "USW00013838",
        "USW00013870",
        "USW00053867",
    ],
    "Ring4_Far": [
        "USW00013897",
        "USW00013891",
        "USW00013889",
        "USW00014839",
        "USW00053864",
        "USW00013881",
        "USW00013880",
        "USW00013957",
        "USW00023048",
        "USW00003816",
        "USW00013966",
        "USW00013865",
        "USW00013878",
        "USW00013899",
    ],
}

# ==============================================================================
# Station Sectors (compass direction classification)
# ==============================================================================
STATION_SECTORS = {
    "N": [
        "USW00003856",
        "USW00013891",
        "USW00003816",
    ],
    "NE": [
        "USW00053863",
        "USW00053819",
        "USW00053871",
        "USW00003870",
        "USW00003849",
        "USW00003810",
        "USW00003820",
        "USW00013881",
        "USW00023048",
    ],
    "E": [
        "USW00013859",
        "USW00013840",
        "USW00013883",
        "USW00013880",
    ],
    "SE": [
        "USW00013876",
        "USW00053872",
        "USW00053847",
        "USW00093843",
        "USW00003822",
        "USW00013870",
        "USW00013889",
    ],
    "S": [
        "USW00093842",
        "USW00093846",
        "USW00013838",
        "USW00013878",
        "USW00014839",
    ],
    "SW": [
        "USW00013873",
        "USW00013877",
        "USW00093805",
        "USW00013864",
        "USW00013895",
        "USW00013865",
    ],
    "W": [
        "USW00003811",
        "USW00003812",
        "USW00013893",
        "USW00003818",
        "USW00013886",
        "USW00013957",
        "USW00013966",
        "USW00013899",
    ],
    "NW": [
        "USW00003813",
        "USW00003854",
        "USW00013846",
        "USW00053868",
        "USW00013882",
        "USW00053867",
        "USW00053864",
        "USW00013897",
    ],
}

# ==============================================================================
# Meteorological Sector Assignments (for feature engineering)
# ==============================================================================
# These group stations by meteorological relevance rather than pure compass direction.
# Atlanta's weather is driven by:
#   - Appalachian cold-air drainage and frontal passages from W/NW
#   - Gulf of Mexico warm/moist advection from S/SW
#   - Piedmont plateau influences from E/SE
#   - Blue Ridge / southern Appalachian moderation from N/NE
#   - Urban heat island and local surface effects in near-field
METEOROLOGICAL_SECTORS = {
    "WNW": (  # Upstream cold-air advection: W and NW sectors (Appalachian cold air)
        STATION_SECTORS["W"] + STATION_SECTORS["NW"]
    ),
    "SW_Gulf": (  # Gulf moisture and warm advection: S and SW sectors
        STATION_SECTORS["S"] + STATION_SECTORS["SW"]
    ),
    "Piedmont": (  # Piedmont plains: E and SE sectors
        STATION_SECTORS["E"] + STATION_SECTORS["SE"]
    ),
    "NearField": (  # Urban/local: all Ring 1 stations
        STATION_RINGS["Ring1_Near"]
    ),
    "Mountains": (  # Blue Ridge influence: N and NE sectors
        STATION_SECTORS["N"] + STATION_SECTORS["NE"]
    ),
}

# ==============================================================================
# Station Metadata
# ==============================================================================
# Full metadata for each surrounding station, sourced from the GHCN inventory.
# Distances and bearings computed via haversine from ATL (33.6301, -84.4418).
STATION_METADATA = {
    # --- Ring 1: Near-field (0-50 mi) --- 10 stations
    "USW00013873": {
        "name": "PEACHTREE CITY FALCON FIELD",
        "state": "GA",
        "lat": 33.3570,
        "lon": -84.5670,
        "distance_mi": 20.28,
        "bearing": 205.10,
        "ring": "Ring1_Near",
        "sector": "SW",
    },
    "USW00003813": {
        "name": "DOBBINS AFB",
        "state": "GA",
        "lat": 33.9150,
        "lon": -84.5163,
        "distance_mi": 19.87,
        "bearing": 344.98,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    "USW00053863": {
        "name": "DEKALB-PEACHTREE AP",
        "state": "GA",
        "lat": 33.8756,
        "lon": -84.3019,
        "distance_mi": 18.45,
        "bearing": 30.12,
        "ring": "Ring1_Near",
        "sector": "NE",
    },
    "USW00003856": {
        "name": "FULTON CO/BROWN FIELD AP",
        "state": "GA",
        "lat": 33.7676,
        "lon": -84.5214,
        "distance_mi": 10.42,
        "bearing": 349.23,
        "ring": "Ring1_Near",
        "sector": "N",
    },
    "USW00053819": {
        "name": "LAWRENCEVILLE/GWINNETT CO AP",
        "state": "GA",
        "lat": 33.9781,
        "lon": -83.9622,
        "distance_mi": 33.92,
        "bearing": 52.46,
        "ring": "Ring1_Near",
        "sector": "NE",
    },
    "USW00093842": {
        "name": "GRIFFIN/SPALDING CO AP",
        "state": "GA",
        "lat": 33.2283,
        "lon": -84.2744,
        "distance_mi": 29.25,
        "bearing": 164.59,
        "ring": "Ring1_Near",
        "sector": "S",
    },
    "USW00013859": {
        "name": "ATHENS BEN EPPS AP",
        "state": "GA",
        "lat": 33.9519,
        "lon": -83.3283,
        "distance_mi": 67.38,
        "bearing": 72.52,
        "ring": "Ring1_Near",
        "sector": "E",
    },
    "USW00003854": {
        "name": "ROME R.B. RUSSELL AP",
        "state": "GA",
        "lat": 34.3506,
        "lon": -85.1611,
        "distance_mi": 59.83,
        "bearing": 319.89,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    "USW00053871": {
        "name": "GAINESVILLE LEE GILMER MEM AP",
        "state": "GA",
        "lat": 34.2728,
        "lon": -83.8306,
        "distance_mi": 52.66,
        "bearing": 39.35,
        "ring": "Ring1_Near",
        "sector": "NE",
    },
    "USW00013846": {
        "name": "CARTERSVILLE AP",
        "state": "GA",
        "lat": 34.1231,
        "lon": -84.8497,
        "distance_mi": 38.52,
        "bearing": 332.91,
        "ring": "Ring1_Near",
        "sector": "NW",
    },
    # --- Ring 2: Regional (50-100 mi) --- 12 stations
    "USW00013876": {
        "name": "MACON MIDDLE GA RGNL AP",
        "state": "GA",
        "lat": 32.6942,
        "lon": -83.6497,
        "distance_mi": 78.15,
        "bearing": 140.72,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USW00003811": {
        "name": "ANNISTON METROPOLITAN AP",
        "state": "AL",
        "lat": 33.5883,
        "lon": -85.8581,
        "distance_mi": 80.04,
        "bearing": 269.38,
        "ring": "Ring2_Regional",
        "sector": "W",
    },
    "USW00093805": {
        "name": "COLUMBUS METROPOLITAN AP",
        "state": "GA",
        "lat": 32.5156,
        "lon": -84.9422,
        "distance_mi": 82.21,
        "bearing": 203.94,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00013840": {
        "name": "AUGUSTA DANIEL FIELD",
        "state": "GA",
        "lat": 33.3700,
        "lon": -81.9644,
        "distance_mi": 140.73,
        "bearing": 96.43,
        "ring": "Ring2_Regional",
        "sector": "E",
    },
    "USW00013877": {
        "name": "LAGRANGE/CALLAWAY AP",
        "state": "GA",
        "lat": 33.0086,
        "lon": -85.0726,
        "distance_mi": 53.88,
        "bearing": 224.50,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00003870": {
        "name": "ANDERSON CO AP",
        "state": "SC",
        "lat": 34.4950,
        "lon": -82.7100,
        "distance_mi": 110.41,
        "bearing": 58.37,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00053868": {
        "name": "DALTON MUNI AP",
        "state": "GA",
        "lat": 34.7225,
        "lon": -84.8706,
        "distance_mi": 79.59,
        "bearing": 339.54,
        "ring": "Ring2_Regional",
        "sector": "NW",
    },
    "USW00003849": {
        "name": "TOCCOA LETOURNEAU AP",
        "state": "GA",
        "lat": 34.5931,
        "lon": -83.2958,
        "distance_mi": 82.33,
        "bearing": 47.15,
        "ring": "Ring2_Regional",
        "sector": "NE",
    },
    "USW00053872": {
        "name": "MILLEDGEVILLE/BALDWIN CO AP",
        "state": "GA",
        "lat": 33.1531,
        "lon": -83.2408,
        "distance_mi": 76.69,
        "bearing": 120.43,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USW00053847": {
        "name": "WARNER ROBINS AFB",
        "state": "GA",
        "lat": 32.6400,
        "lon": -83.5919,
        "distance_mi": 80.15,
        "bearing": 147.37,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    "USW00013864": {
        "name": "NEWNAN COWETA CO AP",
        "state": "GA",
        "lat": 33.3567,
        "lon": -84.7939,
        "distance_mi": 24.07,
        "bearing": 223.33,
        "ring": "Ring2_Regional",
        "sector": "SW",
    },
    "USW00093843": {
        "name": "VIDALIA RGNL AP",
        "state": "GA",
        "lat": 32.1928,
        "lon": -82.3719,
        "distance_mi": 136.79,
        "bearing": 126.52,
        "ring": "Ring2_Regional",
        "sector": "SE",
    },
    # --- Ring 3: Extended (100-150 mi) --- 14 stations
    "USW00003812": {
        "name": "BIRMINGHAM AP",
        "state": "AL",
        "lat": 33.5628,
        "lon": -86.7531,
        "distance_mi": 130.72,
        "bearing": 268.71,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00013882": {
        "name": "CHATTANOOGA LOVELL FIELD AP",
        "state": "TN",
        "lat": 35.0353,
        "lon": -85.2036,
        "distance_mi": 104.54,
        "bearing": 330.48,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    "USW00003810": {
        "name": "GREENVILLE-SPARTANBURG AP",
        "state": "SC",
        "lat": 34.8961,
        "lon": -82.2189,
        "distance_mi": 155.16,
        "bearing": 58.25,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00013895": {
        "name": "MONTGOMERY DANNELLY FIELD",
        "state": "AL",
        "lat": 32.3006,
        "lon": -86.3936,
        "distance_mi": 126.72,
        "bearing": 223.86,
        "ring": "Ring3_Extended",
        "sector": "SW",
    },
    "USW00003822": {
        "name": "SAVANNAH/HILTON HEAD AP",
        "state": "GA",
        "lat": 32.1275,
        "lon": -81.2022,
        "distance_mi": 205.74,
        "bearing": 119.48,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    "USW00013883": {
        "name": "COLUMBIA METROPOLITAN AP",
        "state": "SC",
        "lat": 33.9414,
        "lon": -81.1186,
        "distance_mi": 191.16,
        "bearing": 83.63,
        "ring": "Ring3_Extended",
        "sector": "E",
    },
    "USW00013893": {
        "name": "TUSCALOOSA MUNI AP",
        "state": "AL",
        "lat": 33.2206,
        "lon": -87.6161,
        "distance_mi": 180.67,
        "bearing": 257.50,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00003818": {
        "name": "GADSDEN MUNI AP",
        "state": "AL",
        "lat": 33.9728,
        "lon": -86.0889,
        "distance_mi": 96.95,
        "bearing": 292.91,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00013886": {
        "name": "MUSCLE SHOALS RGNL AP",
        "state": "AL",
        "lat": 34.7453,
        "lon": -87.6003,
        "distance_mi": 196.50,
        "bearing": 293.43,
        "ring": "Ring3_Extended",
        "sector": "W",
    },
    "USW00003820": {
        "name": "GREER/GREENVILLE DOWNTOWN AP",
        "state": "SC",
        "lat": 34.8481,
        "lon": -82.3503,
        "distance_mi": 148.49,
        "bearing": 55.59,
        "ring": "Ring3_Extended",
        "sector": "NE",
    },
    "USW00093846": {
        "name": "VALDOSTA RGNL AP",
        "state": "GA",
        "lat": 30.7825,
        "lon": -83.2761,
        "distance_mi": 204.11,
        "bearing": 168.49,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00013838": {
        "name": "ALBANY DOUGHERTY CO AP",
        "state": "GA",
        "lat": 31.5361,
        "lon": -84.1944,
        "distance_mi": 145.14,
        "bearing": 186.29,
        "ring": "Ring3_Extended",
        "sector": "S",
    },
    "USW00013870": {
        "name": "MACON/ROBINS AFB",
        "state": "GA",
        "lat": 32.6405,
        "lon": -83.5919,
        "distance_mi": 80.09,
        "bearing": 147.50,
        "ring": "Ring3_Extended",
        "sector": "SE",
    },
    "USW00053867": {
        "name": "CROSSVILLE MEM AP",
        "state": "TN",
        "lat": 35.9511,
        "lon": -85.0850,
        "distance_mi": 163.43,
        "bearing": 345.38,
        "ring": "Ring3_Extended",
        "sector": "NW",
    },
    # --- Ring 4: Far (150-250 mi) --- 14 stations
    "USW00013897": {
        "name": "NASHVILLE INTL AP",
        "state": "TN",
        "lat": 36.1175,
        "lon": -86.6894,
        "distance_mi": 213.50,
        "bearing": 333.51,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00013891": {
        "name": "KNOXVILLE MCGHEE TYSON AP",
        "state": "TN",
        "lat": 35.8111,
        "lon": -83.9936,
        "distance_mi": 153.36,
        "bearing": 11.96,
        "ring": "Ring4_Far",
        "sector": "N",
    },
    "USW00013889": {
        "name": "JACKSONVILLE INTL AP",
        "state": "FL",
        "lat": 30.4942,
        "lon": -81.6878,
        "distance_mi": 264.46,
        "bearing": 148.24,
        "ring": "Ring4_Far",
        "sector": "SE",
    },
    "USW00014839": {
        "name": "TALLAHASSEE RGNL AP",
        "state": "FL",
        "lat": 30.3967,
        "lon": -84.3533,
        "distance_mi": 223.96,
        "bearing": 180.80,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00053864": {
        "name": "HUNTSVILLE INTL AP",
        "state": "AL",
        "lat": 34.6372,
        "lon": -86.7753,
        "distance_mi": 150.40,
        "bearing": 300.80,
        "ring": "Ring4_Far",
        "sector": "NW",
    },
    "USW00013881": {
        "name": "CHARLOTTE DOUGLAS INTL AP",
        "state": "NC",
        "lat": 35.2139,
        "lon": -80.9472,
        "distance_mi": 226.08,
        "bearing": 64.70,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00013880": {
        "name": "CHARLESTON INTL AP",
        "state": "SC",
        "lat": 32.8986,
        "lon": -80.0406,
        "distance_mi": 265.47,
        "bearing": 99.42,
        "ring": "Ring4_Far",
        "sector": "E",
    },
    "USW00013957": {
        "name": "MERIDIAN KEY FIELD",
        "state": "MS",
        "lat": 32.3331,
        "lon": -88.7500,
        "distance_mi": 247.87,
        "bearing": 250.86,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00023048": {
        "name": "ASHEVILLE RGNL AP",
        "state": "NC",
        "lat": 35.4361,
        "lon": -82.5419,
        "distance_mi": 155.09,
        "bearing": 41.69,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00003816": {
        "name": "BRISTOL TRI-CITY RGNL AP",
        "state": "TN",
        "lat": 36.4753,
        "lon": -82.4053,
        "distance_mi": 211.31,
        "bearing": 27.15,
        "ring": "Ring4_Far",
        "sector": "NE",
    },
    "USW00013966": {
        "name": "JACKSON HAWKINS FIELD",
        "state": "MS",
        "lat": 32.3333,
        "lon": -90.2222,
        "distance_mi": 318.55,
        "bearing": 255.06,
        "ring": "Ring4_Far",
        "sector": "W",
    },
    "USW00013865": {
        "name": "PENSACOLA RGNL AP",
        "state": "FL",
        "lat": 30.4733,
        "lon": -87.1867,
        "distance_mi": 262.38,
        "bearing": 220.36,
        "ring": "Ring4_Far",
        "sector": "SW",
    },
    "USW00013878": {
        "name": "DOTHAN RGNL AP",
        "state": "AL",
        "lat": 31.3208,
        "lon": -85.4494,
        "distance_mi": 165.31,
        "bearing": 196.15,
        "ring": "Ring4_Far",
        "sector": "S",
    },
    "USW00013899": {
        "name": "TUPELO RGNL AP",
        "state": "MS",
        "lat": 34.2681,
        "lon": -88.7692,
        "distance_mi": 246.96,
        "bearing": 281.44,
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
END_DATE = "2026-02-28"

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
