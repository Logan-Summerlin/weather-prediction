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


# ===================================================================
# 7am-ET inference cutoff manifest
# ===================================================================
# Phase 1 deliverable: a per-feature availability manifest documenting, for
# every operational inference feature source, whether the data it depends on
# is verifiably published by the hard 7:00 AM Eastern Time cutoff on the day
# of the market.  This is the contract that ``schema_validation`` and
# ``operational_features`` enforce at inference time; a feature that cannot be
# proven fresh by 7am ET is a kill-switch event (no trading on stale inputs).
#
# Two timing facts drive every entry below:
#   * 7am ET == 11:00 UTC during EDT (Mar-Nov) and 12:00 UTC during EST
#     (Nov-Mar).  We use the *conservative* (earliest) wall-clock equivalent,
#     11:00 UTC, when deciding whether a fixed-UTC product run is usable, so
#     the manifest never claims a product is available that is not yet out in
#     summer.
#   * Anything observed/issued *after* the cutoff is unusable for that day and
#     must fall back to the most recent pre-cutoff vintage.
# ---------------------------------------------------------------------------

#: Hard morning inference cutoff, expressed in US Eastern wall-clock time.
CUTOFF_HOUR_ET = 7

#: IANA timezone the cutoff is anchored to.  All cities, regardless of their
#: local timezone, share this single Eastern-Time cutoff.
CUTOFF_TIMEZONE = "America/New_York"

#: Conservative UTC hour that 7am ET maps to.  During EDT 7am ET = 11:00 UTC;
#: during EST 7am ET = 12:00 UTC.  We take the earlier (EDT) value so a
#: fixed-UTC model run is only declared "usable" if it is out by 11:00 UTC,
#: which holds year-round.
CUTOFF_HOUR_UTC_CONSERVATIVE = 11

#: Manifest version (bump when cutoff entries change materially).
CUTOFF_MANIFEST_VERSION = "1.0.0"


@dataclass(frozen=True)
class CutoffFeatureSpec:
    """7am-ET availability contract for a single inference feature source.

    Each operational feature consumed at inference time is backed by exactly
    one upstream product (ASOS hourly, a MOS run, a sounding cycle, the
    prior-day settlement feed, ...).  This dataclass records, per the project
    ground rules, the four things needed to decide whether that feature is
    usable at the 7am ET cutoff and what to do when it is not.

    Attributes:
        feature: Stable identifier used by validators and feature builders
                 (e.g. ``"asos_prior_day_daily"``, ``"mos_tmax_morning"``).
        source: Human-readable upstream product / provider.
        description: What the feature contributes to the model.
        publication_schedule: When the upstream product is published, in
                 prose (e.g. ``"hourly, ~5-20 min after the valid hour"``).
        latency_hours: Typical worst-case lag between a record's nominal
                 valid/issue time and its public availability.
        latest_usable_lag_hours: How far *before* the 7am ET cutoff the most
                 recent usable record's valid time sits, in hours.  ``0`` means
                 a record valid right at the cutoff is usable; ``25`` means the
                 freshest usable record is from the prior calendar day.  This
                 is the number that turns the cutoff into a concrete
                 ``latest_usable_timestamp`` (see
                 :func:`latest_usable_timestamp`).
        run_cycles_utc: For cycled NWP/MOS/sounding products, the fixed UTC
                 issue hours that are provably published by the conservative
                 cutoff (e.g. ``(0, 6)`` for the 00Z and 06Z MOS runs).  Empty
                 for continuously-published products such as hourly ASOS.
        fallback_behavior: What the operational path must do when the freshest
                 vintage is missing or post-cutoff.
        criticality: ``"critical"`` (stale -> kill switch / halt) or
                 ``"recommended"`` (stale -> degrade + warn, do not halt).
        max_staleness_hours: Maximum acceptable age, at the cutoff, of the
                 freshest record before the source is considered stale.  This
                 is what the freshness validator compares against.
        sla_source: Optional cross-link to a :class:`DataSourceSLA` ``name``
                 describing the same product's schema.
    """

    feature: str
    source: str
    description: str
    publication_schedule: str
    latency_hours: float
    latest_usable_lag_hours: float
    fallback_behavior: str
    criticality: str  # "critical" or "recommended"
    max_staleness_hours: float
    run_cycles_utc: tuple = ()
    sla_source: Optional[str] = None


# -------------------------------------------------------------------
# Cutoff manifest entries
# -------------------------------------------------------------------

_CUTOFF_ASOS_PRIOR_DAY = CutoffFeatureSpec(
    feature="asos_prior_day_daily",
    source="IEM ASOS (hourly METAR archive)",
    description=(
        "Daily ASOS aggregates (TMAX/TMIN/dewpoint/wind) for the prior "
        "calendar day (D-1), the autoregressive and station-network backbone "
        "of every city model."
    ),
    publication_schedule="hourly, ~5-20 min after each valid hour",
    latency_hours=1.0,
    # The full prior day (through 23:59 local) is complete and published well
    # before the next morning's 7am ET cutoff.  ~31h back from the cutoff
    # guarantees we never reach into the current day's incomplete record.
    latest_usable_lag_hours=7.0,
    run_cycles_utc=(),
    fallback_behavior=(
        "Use the most recent fully-observed prior day; if the freshest ASOS "
        "observation is older than max_staleness_hours, halt (kill switch)."
    ),
    criticality="critical",
    max_staleness_hours=12.0,
    sla_source="asos_daily",
)

_CUTOFF_ASOS_OVERNIGHT = CutoffFeatureSpec(
    feature="asos_overnight_obs",
    source="IEM ASOS (hourly METAR archive)",
    description=(
        "Latest available morning ASOS observation (temperature, dewpoint, "
        "wind, pressure tendency) up to the cutoff, used for same-day "
        "persistence / morning-state features."
    ),
    publication_schedule="hourly, ~5-20 min after each valid hour",
    latency_hours=1.0,
    # A 06:00 ET observation publishes ~06:20 ET, comfortably before 7am ET;
    # we require the freshest ob to be no more than 1h before the cutoff.
    latest_usable_lag_hours=1.0,
    run_cycles_utc=(),
    fallback_behavior=(
        "Step back one hour at a time to the last published observation; "
        "halt if no observation within max_staleness_hours of the cutoff."
    ),
    criticality="critical",
    max_staleness_hours=3.0,
    sla_source="asos_hourly",
)

_CUTOFF_MOS_MORNING = CutoffFeatureSpec(
    feature="mos_tmax_morning",
    source="NWS MOS (GFS MAV / NAM MET) via IEM",
    description=(
        "Today's MOS maximum-temperature guidance and MOS-climatology "
        "anomaly -- the primary NWP-informed signal.  Uses the freshest MOS "
        "cycle provably published by 7am ET."
    ),
    publication_schedule=(
        "00Z run issued ~02-03Z; 06Z run issued ~08-09Z.  06Z guidance "
        "(~08-09Z = 03-04 ET) is out before the conservative 11:00 UTC "
        "cutoff; the 12Z run (issued ~14-15Z) is NOT."
    ),
    latency_hours=3.0,
    latest_usable_lag_hours=0.0,
    run_cycles_utc=(0, 6),
    fallback_behavior=(
        "Prefer the 06Z run; fall back to the 00Z run if 06Z is missing.  "
        "If neither cycle is available, drop MOS features and flag a "
        "kill-switch event (model is NWP-blind for the day)."
    ),
    criticality="critical",
    max_staleness_hours=12.0,
    sla_source="nwp_data",
)

_CUTOFF_SOUNDING = CutoffFeatureSpec(
    feature="sounding_00z",
    source="IGRA / RAOB upper-air soundings",
    description=(
        "Upper-air stability and 850mb thermal/wind features.  CRITICAL "
        "TIMING NOTE: the 12Z sounding (valid 12:00 UTC) is issued AFTER the "
        "7am ET cutoff and must never feed live inference -- only the 00Z "
        "sounding of the market day is cutoff-safe."
    ),
    publication_schedule=(
        "00Z and 12Z launches; 00Z data available ~02-03Z.  12Z is "
        "post-cutoff."
    ),
    latency_hours=3.0,
    # 00Z of the market day is valid 7-8pm ET the prior evening: ~11h back.
    latest_usable_lag_hours=11.0,
    run_cycles_utc=(0,),
    fallback_behavior=(
        "Use the 00Z sounding of the market day; if absent, fall back to the "
        "prior day's 12Z sounding.  Soundings are recommended, not blocking: "
        "degrade gracefully and warn rather than halting."
    ),
    criticality="recommended",
    max_staleness_hours=30.0,
    sla_source="sounding_data",
)

_CUTOFF_PRIOR_SETTLEMENT = CutoffFeatureSpec(
    feature="prior_day_settlement",
    source="Kalshi settled-contract feed",
    description=(
        "Prior-day settled high-temperature outcome, used for realized-error "
        "tracking, calibration-drift monitoring, and persistence baselines."
    ),
    publication_schedule="settles the morning after the contract day",
    latency_hours=8.0,
    # The D-1 settlement is published overnight, available by the D 7am cutoff.
    latest_usable_lag_hours=7.0,
    run_cycles_utc=(),
    fallback_behavior=(
        "Use the latest settled day; if the prior day has not settled by the "
        "cutoff, skip settlement-dependent monitoring features and warn "
        "(non-blocking) but flag for the calibration-drift kill switch."
    ),
    criticality="recommended",
    max_staleness_hours=36.0,
    sla_source=None,
)


# -------------------------------------------------------------------
# Cutoff registry (keyed by CutoffFeatureSpec.feature)
# -------------------------------------------------------------------

_CUTOFF_REGISTRY: Dict[str, CutoffFeatureSpec] = {
    spec.feature: spec
    for spec in [
        _CUTOFF_ASOS_PRIOR_DAY,
        _CUTOFF_ASOS_OVERNIGHT,
        _CUTOFF_MOS_MORNING,
        _CUTOFF_SOUNDING,
        _CUTOFF_PRIOR_SETTLEMENT,
    ]
}


# ===================================================================
# Cutoff manifest public API
# ===================================================================

def get_cutoff_spec(feature: str) -> CutoffFeatureSpec:
    """Look up the 7am-ET cutoff spec for an inference feature.

    Parameters
    ----------
    feature : str
        Registered feature identifier (e.g. ``"mos_tmax_morning"``).

    Returns
    -------
    CutoffFeatureSpec
        The cutoff availability contract for the feature.

    Raises
    ------
    KeyError
        If *feature* is not registered in the cutoff manifest.
    """
    key = feature.strip().lower()
    if key not in _CUTOFF_REGISTRY:
        available = ", ".join(sorted(_CUTOFF_REGISTRY.keys()))
        raise KeyError(
            f"Unknown cutoff feature '{feature}'. "
            f"Registered features: {available}"
        )
    return _CUTOFF_REGISTRY[key]


def list_cutoff_features() -> List[str]:
    """Return a sorted list of all registered cutoff feature identifiers."""
    return sorted(_CUTOFF_REGISTRY.keys())


def get_cutoff_manifest_version() -> str:
    """Return the cutoff manifest version string (semver)."""
    return CUTOFF_MANIFEST_VERSION


def get_critical_cutoff_features() -> List[str]:
    """Return the identifiers of cutoff features whose staleness halts trading."""
    return sorted(
        f for f, spec in _CUTOFF_REGISTRY.items()
        if spec.criticality == "critical"
    )


def cutoff_instant_utc(market_date):
    """Return the UTC :class:`datetime` of the 7am-ET cutoff for a market day.

    The cutoff is resolved in the canonical Eastern timezone and converted to
    UTC, so it correctly reflects EST (12:00 UTC) vs EDT (11:00 UTC).  When the
    optional ``zoneinfo`` timezone database is unavailable, falls back to the
    conservative fixed offset (:data:`CUTOFF_HOUR_UTC_CONSERVATIVE`).

    Parameters
    ----------
    market_date : datetime.date or datetime.datetime or str
        The market day (the day the contract settles on).  Strings are parsed
        as ISO ``YYYY-MM-DD``.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC datetime of the 7am ET cutoff on *market_date*.
    """
    from datetime import datetime, date, timezone, timedelta

    if isinstance(market_date, str):
        market_date = date.fromisoformat(market_date[:10])
    elif isinstance(market_date, datetime):
        market_date = market_date.date()

    naive_local = datetime(
        market_date.year, market_date.month, market_date.day, CUTOFF_HOUR_ET
    )
    try:
        from zoneinfo import ZoneInfo

        local = naive_local.replace(tzinfo=ZoneInfo(CUTOFF_TIMEZONE))
        return local.astimezone(timezone.utc)
    except Exception:  # pragma: no cover - zoneinfo/tzdata missing
        # Conservative fallback: treat the cutoff as the EDT-equivalent UTC
        # hour (earliest), which never over-claims product availability.
        return datetime(
            market_date.year, market_date.month, market_date.day,
            CUTOFF_HOUR_UTC_CONSERVATIVE, tzinfo=timezone.utc,
        )


def latest_usable_timestamp(feature: str, market_date):
    """Compute the latest record timestamp usable for a feature at the cutoff.

    A record whose valid/issue time is *after* this instant has not been
    published by the 7am ET cutoff (or belongs to the current, incomplete
    day) and must not be used for *market_date* inference.

    Parameters
    ----------
    feature : str
        Registered cutoff feature identifier.
    market_date : datetime.date or datetime.datetime or str
        The market day.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC datetime; the newest record valid time that is
        cutoff-safe for *market_date*.
    """
    from datetime import timedelta

    spec = get_cutoff_spec(feature)
    cutoff = cutoff_instant_utc(market_date)
    return cutoff - timedelta(hours=spec.latest_usable_lag_hours)


def build_cutoff_manifest_table() -> List[Dict[str, object]]:
    """Return the cutoff manifest as a list of plain dicts (for export/report).

    Each row captures the documented availability contract for one inference
    feature, suitable for serialising to JSON/CSV or rendering in a report.
    """
    rows: List[Dict[str, object]] = []
    for feature in list_cutoff_features():
        spec = _CUTOFF_REGISTRY[feature]
        rows.append({
            "feature": spec.feature,
            "source": spec.source,
            "description": spec.description,
            "publication_schedule": spec.publication_schedule,
            "latency_hours": spec.latency_hours,
            "latest_usable_lag_hours": spec.latest_usable_lag_hours,
            "run_cycles_utc": list(spec.run_cycles_utc),
            "fallback_behavior": spec.fallback_behavior,
            "criticality": spec.criticality,
            "max_staleness_hours": spec.max_staleness_hours,
            "sla_source": spec.sla_source,
        })
    return rows
