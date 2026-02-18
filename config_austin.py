"""
Expanded Configuration for Austin Temperature Prediction.

Station network centered on Austin-Bergstrom International Airport (USW00013904).
Organized by distance ring and compass sector from AUS.
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
# Surrounding Stations
# ==============================================================================
SURROUNDING_STATIONS = {
    # Ring 1: Near (0-60 mi)
    "USW00003904": "AUSTIN CAMP MABRY, TX (5mi NW)",
    "USW00013958": "SAN ANTONIO INTL AP, TX (67mi SW)",
    "USW00013959": "SAN ANTONIO STINSON MUNI AP, TX (70mi SW)",
    "USW00013962": "NEW BRAUNFELS RGNL AP, TX (45mi S)",
    "USW00013963": "SAN MARCOS RGNL AP, TX (28mi S)",
    "USW00013956": "GEORGETOWN MUNI AP, TX (30mi N)",
    # Ring 2: Regional (60-120 mi)
    "USW00013985": "WACO RGNL AP, TX (95mi N)",
    "USW00012960": "COLLEGE STATION EASTERWOOD FLD, TX (90mi E)",
    "USW00012927": "DEL RIO INTL AP, TX (135mi WSW)",
    "USW00013966": "KILLEEN RGNL AP, TX (60mi N)",
    "USW00013961": "BURNET MUNI AP, TX (45mi NW)",
    "USW00013965": "LA GRANGE - FAYETTE RGNL, TX (65mi ESE)",
    "USW00013964": "GIDDINGS-LEE CO AP, TX (50mi E)",
    # Ring 3: Extended (120-220 mi)
    "USW00012917": "HOUSTON HOBBY AP, TX (145mi E)",
    "USW00012918": "HOUSTON INTERCONTINENTAL AP, TX (155mi E)",
    "USW00012914": "CORPUS CHRISTI INTL AP, TX (170mi S)",
    "USW00012919": "LAREDO INTL AP, TX (215mi SSW)",
    "USW00013957": "DEL RIO LAUGHLIN AFB, TX (145mi WSW)",
    "USW00003927": "DEL RIO 4E, TX (132mi WSW)",
}

# All stations (target + surrounding) for convenience
ALL_STATIONS = {TARGET_STATION: "Austin-Bergstrom International Airport (Target)", **SURROUNDING_STATIONS}
ALL_STATIONS[TARGET_STATION] = "Austin-Bergstrom International Airport (Target)"

ASOS_STATION_MAP = {
    "USW00013904": "KAUS",
    "USW00003904": "KATT",
    "USW00013958": "KSAT",
    "USW00013959": "KSSF",
    "USW00013962": "KBAZ",
    "USW00013963": "KHYI",
    "USW00013956": "KGTU",
    "USW00013985": "KACT",
    "USW00012960": "KCLL",
    "USW00012927": "KDRT",
    "USW00013966": "KGRK",
    "USW00013961": "KBMQ",
    "USW00013965": "K3T5",
    "USW00013964": "KGYB",
    "USW00012917": "KHOU",
    "USW00012918": "KIAH",
    "USW00012914": "KCRP",
    "USW00012919": "KLRD",
    "USW00013957": "KDLF",
}

NON_ASOS_STATIONS = {
    "USW00003927": "Del Rio cooperative site",
}

MIN_COMPLETENESS = 0.80

STATION_RINGS = {
    "Ring1_Near": ["USW00003904", "USW00013963", "USW00013956", "USW00013961"],
    "Ring2_Regional": [
        "USW00013962",
        "USW00013958",
        "USW00013959",
        "USW00013966",
        "USW00013964",
        "USW00013965",
        "USW00013985",
        "USW00012960",
    ],
    "Ring3_Extended": ["USW00012927", "USW00012917", "USW00012918", "USW00012914", "USW00012919", "USW00013957", "USW00003927"],
    "Ring4_Far": [],
}

STATION_SECTORS = {
    "N": ["USW00013956", "USW00013985", "USW00013966"],
    "NE": ["USW00012960"],
    "E": ["USW00013964", "USW00013965", "USW00012917", "USW00012918"],
    "SE": ["USW00012914"],
    "S": ["USW00013962", "USW00013963"],
    "SW": ["USW00013958", "USW00013959", "USW00012919"],
    "W": ["USW00012927", "USW00003927"],
    "NW": ["USW00003904", "USW00013961", "USW00013957"],
}

METEOROLOGICAL_SECTORS = {
    "Dryline_W": STATION_SECTORS["W"] + STATION_SECTORS["NW"],
    "Gulf_E": STATION_SECTORS["E"] + STATION_SECTORS["SE"],
    "Warm_S": STATION_SECTORS["S"] + STATION_SECTORS["SW"],
    "NearField": STATION_RINGS["Ring1_Near"],
}

# Optional metadata map for downstream diagnostics
STATION_METADATA = {
    station_id: {
        "name": name,
        "ring": next((ring for ring, ids in STATION_RINGS.items() if station_id in ids), "Unknown"),
        "sector": next((sector for sector, ids in STATION_SECTORS.items() if station_id in ids), "Unknown"),
    }
    for station_id, name in SURROUNDING_STATIONS.items()
}

# ==============================================================================
# Pipeline Constants
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
