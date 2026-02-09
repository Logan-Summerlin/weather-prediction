"""
Configuration for NYC Temperature Prediction Project.

All configurable parameters are centralized here. No hardcoded values
should appear in source modules — import from this file instead.
"""

import os

# ==============================================================================
# Project Paths
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
STATIONS_FILE = os.path.join(DATA_DIR, "stations.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ASOS_RAW_DIR = os.path.join(RAW_DATA_DIR, "asos")
ASOS_DAILY_DIR = os.path.join(PROCESSED_DATA_DIR, "asos_daily")
IGRA_RAW_DIR = os.path.join(RAW_DATA_DIR, "igra")
NWP_RAW_DIR = os.path.join(RAW_DATA_DIR, "nwp")

# ==============================================================================
# NOAA Data Source
# ==============================================================================
# We use bulk .dly file downloads (no API token needed).
NOAA_BULK_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/"

# ==============================================================================
# Target Station — Central Park, NYC
# ==============================================================================
TARGET_STATION = "USW00094728"  # NY City Central Park
TARGET_LAT = 40.7789
TARGET_LON = -73.9692
TARGET_VARIABLE = "TMAX"

# ==============================================================================
# Date Range — Phase 6 (Scale Up): 40 years
# ==============================================================================
START_DATE = "1985-01-01"
END_DATE = "2024-12-31"

# ==============================================================================
# Operational Data Window (ASOS/IGRA/NWP)
# ==============================================================================
ASOS_START_DATE = "1998-01-01"
ASOS_END_DATE = END_DATE
IGRA_START_DATE = "2000-01-01"
IGRA_END_DATE = END_DATE
IGRA_STATION_ID = "USM00072501"  # Upton/Brookhaven (OKX)
IGRA_LEVELS_MB = [850.0, 500.0]
NWP_START_DATE = "2000-01-01"
NWP_END_DATE = END_DATE
NWP_VARIABLES = [
    "tmax_2m",
    "tmp_850",
    "ugrd_10m",
    "vgrd_10m",
    "tcdc_eatm",
    "mslp",
    "apcp",
]

# ==============================================================================
# Surrounding Input Stations
# ==============================================================================
# Format: {station_id: "description"}
# Selected for geographic coverage (~50-200 mi from Central Park), data
# completeness (>= 90%), and directional diversity.
SURROUNDING_STATIONS = {
    "USW00014735": "Albany, NY (Albany Airport)",
    "USW00014740": "Hartford, CT (Bradley International Airport)",
    "USW00094702": "Bridgeport, CT (Sikorsky Memorial Airport)",
    "USW00014732": "Islip, NY (Long Island MacArthur Airport)",
    "USW00093730": "Atlantic City, NJ (Atlantic City International Airport)",
    "USW00014792": "Trenton, NJ (Trenton-Mercer Airport)",
    "USW00013739": "Philadelphia, PA (Philadelphia International Airport)",
    "USW00014737": "Allentown, PA (Lehigh Valley International Airport)",
    "USW00014777": "Scranton, PA (Wilkes-Barre/Scranton International Airport)",
    "USW00014734": "Newark, NJ (Newark Liberty International Airport)",
    "USW00094789": "JFK Airport, NY (John F. Kennedy International Airport)",
    "USW00014739": "LaGuardia Airport, NY",
    "USW00014771": "White Plains, NY (Westchester County Airport)",
    "USW00014757": "Poughkeepsie, NY (Dutchess County Airport)",
}

# All stations (target + surrounding) for convenience
ALL_STATIONS = {TARGET_STATION: "Central Park, NYC (Target)", **SURROUNDING_STATIONS}

# ==============================================================================
# Input Features
# ==============================================================================
INPUT_VARIABLES = ["TMAX", "TMIN"]

# ==============================================================================
# Data Quality
# ==============================================================================
MIN_COMPLETENESS = 0.90  # Minimum fraction of non-missing days required
MAX_FORWARD_FILL_DAYS = 3  # Maximum gap length for forward-fill imputation

# ==============================================================================
# Train / Validation / Test Split Ratios (chronological)
# ==============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZES = [64, 32]
DROPOUT = 0.0

# ==============================================================================
# Quantile Regression (Confidence Intervals)
# ==============================================================================
QUANTILES = [0.025, 0.50, 0.975]  # For 95% prediction intervals
