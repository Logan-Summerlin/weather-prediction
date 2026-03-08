"""
Operational Data SLA (Service Level Agreement) Manifest.

Defines expected schemas, freshness, and quality requirements for every data
source in the weather prediction pipeline.  Each data source is described by
a ``DataSourceSLA`` instance that encodes:

- Column-level specifications (name, type, required/optional, value bounds).
- Minimum completeness (fraction of non-null values).
- Minimum row count.
- Maximum staleness for operational sources.
- Criticality level (``"critical"`` or ``"recommended"``).

The SLA registry acts as a single source of truth that downstream validation,
monitoring, and kill-switch logic can reference without embedding magic
numbers in multiple places.

Usage:
    from src.data_sla import (
        get_sla,
        list_sla_sources,
        get_sla_manifest_version,
        SLA_MANIFEST_VERSION,
    )

    sla = get_sla("ghcn_daily_raw")
    for col in sla.columns:
        print(col.name, col.required, col.min_value, col.max_value)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Manifest version (semver).  Bump when SLA definitions change materially.
# ---------------------------------------------------------------------------
SLA_MANIFEST_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Core data-contract dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column in a data source.

    Attributes:
        name:      Column name as it appears in the CSV / DataFrame.
        required:  If True the column must be present; if False it is
                   optional (absence is acceptable, not a validation error).
        dtype:     Expected logical type: ``"float"``, ``"str"``,
                   ``"datetime"``, or ``"int"``.
        min_value: Lower bound for numeric columns (inclusive).  None means
                   no lower bound enforced.
        max_value: Upper bound for numeric columns (inclusive).  None means
                   no upper bound enforced.
    """

    name: str
    required: bool = True
    dtype: str = "float"  # "float", "str", "datetime", "int"
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True)
class DataSourceSLA:
    """SLA definition for a single data source.

    Captures all quality expectations for a particular data product consumed
    or produced by the pipeline: which columns exist, their value domains,
    completeness requirements, minimum row counts, and freshness bounds.

    Attributes:
        name:                Unique identifier for this data source (the
                             registry lookup key).
        version:             SLA version string (semver) for this source's
                             definition.
        description:         Human-readable summary of the data source and
                             its role in the pipeline.
        columns:             Ordered list of ``ColumnSpec`` instances
                             describing the expected schema.
        criticality:         ``"critical"`` or ``"recommended"``.  Critical
                             sources must pass all checks before the pipeline
                             may proceed; recommended sources may be absent
                             without halting inference.
        min_completeness:    Minimum fraction (0.0 -- 1.0) of non-null values
                             across required columns.
        min_rows:            Minimum number of rows expected in the file.
        max_staleness_hours: For operational sources, the maximum acceptable
                             age (in hours) of the most recent record.  None
                             means staleness is not enforced (e.g., for
                             historical-only or derived artifacts).
        path_pattern:        Filesystem path pattern with ``{city}``,
                             ``{station_id}``, ``{icao}``, or ``{split}``
                             placeholders.  Informational only; not enforced
                             by the SLA itself.
        no_nan_columns:      Column names where NaN values are never
                             acceptable (stricter than ``min_completeness``).
        no_inf_columns:      Column names where +/-inf values are never
                             acceptable.
    """

    name: str
    version: str
    description: str
    columns: List[ColumnSpec]
    criticality: str  # "critical" or "recommended"
    min_completeness: float  # 0.0 .. 1.0
    min_rows: int
    max_staleness_hours: Optional[float] = None
    path_pattern: Optional[str] = None
    no_nan_columns: List[str] = field(default_factory=list)
    no_inf_columns: List[str] = field(default_factory=list)


# ===================================================================
# SLA definitions for each pipeline data source
# ===================================================================

# -------------------------------------------------------------------
# 1. GHCN-Daily raw station CSVs
# -------------------------------------------------------------------
_GHCN_DAILY_RAW = DataSourceSLA(
    name="ghcn_daily_raw",
    version="1.0.0",
    description=(
        "GHCN-Daily raw station CSV files containing observed daily weather "
        "variables (TMAX, TMIN, and optional precipitation/snow/wind).  "
        "Used as the primary data source for the training pipeline.  "
        "GHCN may also serve as secondary validation against ASOS-derived "
        "TMAX for cross-source bias checks."
    ),
    columns=[
        ColumnSpec(name="date", required=True, dtype="datetime"),
        ColumnSpec(name="TMAX", required=True, dtype="float",
                   min_value=-40.0, max_value=130.0),
        ColumnSpec(name="TMIN", required=True, dtype="float",
                   min_value=-60.0, max_value=120.0),
        ColumnSpec(name="PRCP", required=False, dtype="float",
                   min_value=0.0, max_value=30.0),
        ColumnSpec(name="SNOW", required=False, dtype="float",
                   min_value=0.0, max_value=100.0),
        ColumnSpec(name="SNWD", required=False, dtype="float",
                   min_value=0.0, max_value=200.0),
        ColumnSpec(name="AWND", required=False, dtype="float",
                   min_value=0.0, max_value=200.0),
    ],
    criticality="critical",
    min_completeness=0.80,
    min_rows=365,
    max_staleness_hours=None,  # historical archive; staleness N/A
    path_pattern="data/{city}/raw/{station_id}.csv",
)


# -------------------------------------------------------------------
# 2. ASOS hourly CSVs (IEM download format)
# -------------------------------------------------------------------
_ASOS_HOURLY = DataSourceSLA(
    name="asos_hourly",
    version="1.0.0",
    description=(
        "IEM ASOS hourly observations.  The authoritative operational data "
        "source for live inference.  Matches the operational pipeline and "
        "should be the primary training source per project conventions "
        "(train on ASOS, not GHCN-Daily)."
    ),
    columns=[
        ColumnSpec(name="valid", required=True, dtype="datetime"),
        ColumnSpec(name="tmpf", required=True, dtype="float",
                   min_value=-60.0, max_value=130.0),
        ColumnSpec(name="dwpf", required=True, dtype="float",
                   min_value=-80.0, max_value=100.0),
        ColumnSpec(name="relh", required=True, dtype="float",
                   min_value=0.0, max_value=100.0),
        ColumnSpec(name="drct", required=True, dtype="float",
                   min_value=0.0, max_value=360.0),
        ColumnSpec(name="sknt", required=True, dtype="float",
                   min_value=0.0, max_value=200.0),
        ColumnSpec(name="mslp", required=False, dtype="float",
                   min_value=870.0, max_value=1084.0),
        ColumnSpec(name="alti", required=False, dtype="float",
                   min_value=25.0, max_value=32.0),
        ColumnSpec(name="vsby", required=False, dtype="float",
                   min_value=0.0, max_value=20.0),
        ColumnSpec(name="ceil", required=False, dtype="float",
                   min_value=0.0, max_value=99999.0),
    ],
    criticality="critical",
    min_completeness=0.70,
    min_rows=24,  # at least one day of hourly observations
    max_staleness_hours=6.0,  # operational: must be recent
    path_pattern="data/{city}/asos/{icao}_hourly.csv",
)


# -------------------------------------------------------------------
# 3. ASOS daily aggregates (produced by asos_preprocessing.py)
# -------------------------------------------------------------------
_ASOS_DAILY = DataSourceSLA(
    name="asos_daily",
    version="1.0.0",
    description=(
        "Daily aggregates derived from ASOS hourly data by "
        "asos_preprocessing.py.  Contains daily TMAX, TMIN, TMEAN, "
        "dewpoint, wind, and observation count.  A day is considered "
        "valid only if obs_count >= 4."
    ),
    columns=[
        ColumnSpec(name="date", required=True, dtype="datetime"),
        ColumnSpec(name="station_id", required=True, dtype="str"),
        ColumnSpec(name="tmax_f", required=True, dtype="float",
                   min_value=-40.0, max_value=130.0),
        ColumnSpec(name="tmin_f", required=True, dtype="float",
                   min_value=-60.0, max_value=120.0),
        ColumnSpec(name="tmean_f", required=True, dtype="float",
                   min_value=-60.0, max_value=130.0),
        ColumnSpec(name="dewpoint_mean_f", required=True, dtype="float",
                   min_value=-80.0, max_value=100.0),
        ColumnSpec(name="wind_speed_mean_mph", required=True, dtype="float",
                   min_value=0.0, max_value=200.0),
        ColumnSpec(name="wind_dir_mean_deg", required=True, dtype="float",
                   min_value=0.0, max_value=360.0),
        ColumnSpec(name="obs_count", required=True, dtype="int",
                   min_value=4.0, max_value=48.0),
    ],
    criticality="critical",
    min_completeness=0.80,
    min_rows=30,  # at least roughly one month of daily data
    max_staleness_hours=12.0,
    path_pattern="data/{city}/asos/{icao}_daily.csv",
)


# -------------------------------------------------------------------
# 4. Processed feature files (output of data_preprocessing.py)
# -------------------------------------------------------------------
_PROCESSED_FEATURES = DataSourceSLA(
    name="processed_features",
    version="1.0.0",
    description=(
        "Preprocessed feature matrices for train/val/test splits.  Must "
        "include cyclical date encodings (sin_day, cos_day).  Additional "
        "columns (lagged station TMAX/TMIN, sector gradients, etc.) are "
        "city- and model-dependent and validated dynamically at load time.  "
        "No infinite values are permitted in any column."
    ),
    columns=[
        ColumnSpec(name="sin_day", required=True, dtype="float",
                   min_value=-1.0, max_value=1.0),
        ColumnSpec(name="cos_day", required=True, dtype="float",
                   min_value=-1.0, max_value=1.0),
        # Remaining feature columns vary by city and model configuration;
        # they are checked generically (no NaN/inf, reasonable ranges)
        # rather than enumerated here.
    ],
    criticality="critical",
    min_completeness=0.95,
    min_rows=100,
    max_staleness_hours=None,  # derived artifact; freshness N/A
    path_pattern="data/{city}/processed/features_{split}.csv",
    no_inf_columns=["sin_day", "cos_day"],
)


# -------------------------------------------------------------------
# 5. Processed target files (output of data_preprocessing.py)
# -------------------------------------------------------------------
_PROCESSED_TARGETS = DataSourceSLA(
    name="processed_targets",
    version="1.0.0",
    description=(
        "Target vectors (daily TMAX) for train/val/test splits.  Must "
        "contain exactly one column matching the city's TMAX naming "
        "convention (e.g., NYC_TMAX, {STATION}_TMAX).  No NaN values "
        "are permitted in the target column -- every row must have a "
        "valid observation for supervised training and evaluation."
    ),
    columns=[
        # The target column name is city-dependent (e.g., "NYC_TMAX",
        # "CHI_TMAX", "{station_id}_TMAX").  Downstream validators
        # should match against a *_TMAX or TMAX pattern.
        ColumnSpec(name="TMAX", required=True, dtype="float",
                   min_value=-40.0, max_value=130.0),
    ],
    criticality="critical",
    min_completeness=1.0,  # no missing targets allowed
    min_rows=100,
    max_staleness_hours=None,
    path_pattern="data/{city}/processed/target_{split}.csv",
    no_nan_columns=["TMAX"],
)


# -------------------------------------------------------------------
# 6. NWP (Numerical Weather Prediction) data
# -------------------------------------------------------------------
_NWP_DATA = DataSourceSLA(
    name="nwp_data",
    version="1.0.0",
    description=(
        "Numerical weather prediction (GFS/GEFS) grid-point extracts "
        "preprocessed by nwp_preprocessing.py.  Provides model-based "
        "forecast guidance including 2-m TMAX, 850-mb temperature, wind, "
        "and cloud cover.  Currently recommended but not blocking -- the "
        "pipeline can operate without NWP if this source is unavailable."
    ),
    columns=[
        ColumnSpec(name="forecast_date", required=True, dtype="datetime"),
        ColumnSpec(name="tmax_2m_f", required=True, dtype="float",
                   min_value=-60.0, max_value=140.0),
        ColumnSpec(name="tmp_850_f", required=False, dtype="float",
                   min_value=-80.0, max_value=120.0),
        ColumnSpec(name="wind_speed_10m_kt", required=False, dtype="float",
                   min_value=0.0, max_value=200.0),
        ColumnSpec(name="tcdc_pct", required=False, dtype="float",
                   min_value=0.0, max_value=100.0),
        ColumnSpec(name="mslp_hpa", required=False, dtype="float",
                   min_value=870.0, max_value=1084.0),
    ],
    criticality="recommended",
    min_completeness=0.70,
    min_rows=1,
    max_staleness_hours=12.0,
    path_pattern="data/{city}/nwp/gfs_daily.csv",
)


# -------------------------------------------------------------------
# 7. Sounding (upper-air IGRA) data
# -------------------------------------------------------------------
_SOUNDING_DATA = DataSourceSLA(
    name="sounding_data",
    version="1.0.0",
    description=(
        "IGRA upper-air sounding observations preprocessed by "
        "soundings_preprocessing.py.  Provides pressure-level "
        "temperatures, stability indices, and lapse rates.  "
        "Recommended supplemental source; not blocking for the core "
        "pipeline.  Soundings are issued at 00Z and 12Z."
    ),
    columns=[
        ColumnSpec(name="date", required=True, dtype="datetime"),
        ColumnSpec(name="station_id", required=True, dtype="str"),
        ColumnSpec(name="tmp_850_c", required=True, dtype="float",
                   min_value=-40.0, max_value=40.0),
        ColumnSpec(name="tmp_500_c", required=True, dtype="float",
                   min_value=-60.0, max_value=10.0),
        ColumnSpec(name="tmp_surface_c", required=False, dtype="float",
                   min_value=-50.0, max_value=55.0),
        ColumnSpec(name="wind_dir_850_deg", required=False, dtype="float",
                   min_value=0.0, max_value=360.0),
        ColumnSpec(name="wind_spd_850_kt", required=False, dtype="float",
                   min_value=0.0, max_value=200.0),
        ColumnSpec(name="lapse_rate_850_500", required=False, dtype="float",
                   min_value=-10.0, max_value=15.0),
    ],
    criticality="recommended",
    min_completeness=0.60,
    min_rows=1,
    max_staleness_hours=18.0,  # soundings are 00Z/12Z; allow some lag
    path_pattern="data/{city}/soundings/daily_features.csv",
)


# ===================================================================
# Internal registry (keyed by DataSourceSLA.name)
# ===================================================================

_SLA_REGISTRY: Dict[str, DataSourceSLA] = {
    sla.name: sla
    for sla in [
        _GHCN_DAILY_RAW,
        _ASOS_HOURLY,
        _ASOS_DAILY,
        _PROCESSED_FEATURES,
        _PROCESSED_TARGETS,
        _NWP_DATA,
        _SOUNDING_DATA,
    ]
}


# ===================================================================
# Public API
# ===================================================================

def get_sla(source_name: str) -> DataSourceSLA:
    """Look up an SLA definition by data source name.

    Parameters
    ----------
    source_name : str
        Registered data source name (e.g., ``"ghcn_daily_raw"``,
        ``"asos_hourly"``, ``"processed_features"``).

    Returns
    -------
    DataSourceSLA
        The SLA definition for the requested source.

    Raises
    ------
    KeyError
        If *source_name* is not found in the registry.
    """
    key = source_name.strip().lower()
    if key not in _SLA_REGISTRY:
        available = ", ".join(sorted(_SLA_REGISTRY.keys()))
        raise KeyError(
            f"Unknown data source '{source_name}'. "
            f"Registered sources: {available}"
        )
    return _SLA_REGISTRY[key]


def list_sla_sources() -> List[str]:
    """Return a sorted list of all registered data source names.

    Returns
    -------
    list of str
        Source names that can be passed to :func:`get_sla`.
    """
    return sorted(_SLA_REGISTRY.keys())


def get_sla_manifest_version() -> str:
    """Return the global SLA manifest version string.

    Returns
    -------
    str
        Semantic version of the manifest (e.g., ``"1.0.0"``).
    """
    return SLA_MANIFEST_VERSION


# -------------------------------------------------------------------
# Convenience helpers
# -------------------------------------------------------------------

def get_required_columns(source_name: str) -> List[str]:
    """Return the names of all required columns for a data source.

    Convenience function that filters the SLA column list to only those
    marked ``required=True``.

    Parameters
    ----------
    source_name : str
        Registered data source name.

    Returns
    -------
    list of str
        Column names that must be present in the data.
    """
    sla = get_sla(source_name)
    return [col.name for col in sla.columns if col.required]


def get_column_spec(source_name: str, column_name: str) -> Optional[ColumnSpec]:
    """Look up the specification for a single column within a data source.

    Parameters
    ----------
    source_name : str
        Registered data source name.
    column_name : str
        Column name to look up.

    Returns
    -------
    ColumnSpec or None
        The column specification if found, otherwise ``None``.
    """
    sla = get_sla(source_name)
    for col in sla.columns:
        if col.name == column_name:
            return col
    return None


def is_critical(source_name: str) -> bool:
    """Check whether a data source is marked as critical.

    Critical sources must pass all validation checks before the pipeline
    is allowed to proceed.  Non-critical (recommended) sources produce
    warnings but do not block execution.

    Parameters
    ----------
    source_name : str
        Registered data source name.

    Returns
    -------
    bool
        True if the source's criticality is ``"critical"``.
    """
    return get_sla(source_name).criticality == "critical"


# Backward-compatible alias for callers that used the old function name.
list_slas = list_sla_sources
