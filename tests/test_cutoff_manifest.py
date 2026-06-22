"""
Tests for the 7am-ET inference cutoff manifest and freshness validators.

Covers:
  - Cutoff manifest registry in src/data_sla.py (entries, criticality, version).
  - cutoff_instant_utc EST/EDT handling and latest_usable_timestamp math.
  - The physically-critical sounding constraint (only 00Z is cutoff-safe).
  - validate_cutoff_freshness leakage + staleness detection.
  - validate_inference_freshness criticality-based escalation and the
    enforce_inference_freshness kill switch.
  - operational_features cutoff-safety wiring.
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_sla import (
    CUTOFF_HOUR_ET,
    CUTOFF_MANIFEST_VERSION,
    CutoffFeatureSpec,
    build_cutoff_manifest_table,
    cutoff_instant_utc,
    get_critical_cutoff_features,
    get_cutoff_spec,
    latest_usable_timestamp,
    list_cutoff_features,
)
from src.schema_validation import (
    KillSwitchError,
    enforce_inference_freshness,
    validate_cutoff_freshness,
    validate_inference_freshness,
)


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

def test_manifest_version_is_semver():
    assert re.match(r"^\d+\.\d+\.\d+$", CUTOFF_MANIFEST_VERSION)


def test_registry_has_expected_features():
    feats = set(list_cutoff_features())
    assert {
        "asos_prior_day_daily",
        "asos_overnight_obs",
        "mos_tmax_morning",
        "sounding_00z",
        "prior_day_settlement",
    } <= feats


def test_every_feature_has_complete_spec():
    for feature in list_cutoff_features():
        spec = get_cutoff_spec(feature)
        assert isinstance(spec, CutoffFeatureSpec)
        assert spec.source and spec.description
        assert spec.publication_schedule
        assert spec.fallback_behavior
        assert spec.criticality in ("critical", "recommended")
        assert spec.max_staleness_hours > 0


def test_cutoff_hour_is_7am():
    assert CUTOFF_HOUR_ET == 7


def test_critical_features_subset_of_all():
    assert set(get_critical_cutoff_features()) <= set(list_cutoff_features())
    # ASOS + MOS backbone must be critical.
    assert "asos_prior_day_daily" in get_critical_cutoff_features()
    assert "mos_tmax_morning" in get_critical_cutoff_features()


def test_unknown_feature_raises():
    with pytest.raises(KeyError):
        get_cutoff_spec("does_not_exist")


def test_manifest_table_round_trips():
    rows = build_cutoff_manifest_table()
    assert len(rows) == len(list_cutoff_features())
    for row in rows:
        assert {"feature", "source", "fallback_behavior", "criticality"} <= set(row)


# ---------------------------------------------------------------------------
# Cutoff instant: EST vs EDT
# ---------------------------------------------------------------------------

def test_cutoff_instant_edt_is_11utc():
    # July -> EDT (UTC-4): 7am ET == 11:00 UTC.
    assert cutoff_instant_utc("2026-07-01") == datetime(2026, 7, 1, 11, tzinfo=timezone.utc)


def test_cutoff_instant_est_is_12utc():
    # January -> EST (UTC-5): 7am ET == 12:00 UTC.
    assert cutoff_instant_utc("2026-01-01") == datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# latest_usable_timestamp
# ---------------------------------------------------------------------------

def test_sounding_only_00z_is_usable():
    # The 12Z sounding (12:00 UTC) is after the cutoff and must be excluded;
    # the latest usable sounding is exactly 00Z of the market day.
    usable = latest_usable_timestamp("sounding_00z", "2026-07-01")
    assert usable == datetime(2026, 7, 1, 0, tzinfo=timezone.utc)
    assert usable.hour == 0  # never reaches into 12Z


def test_mos_usable_up_to_cutoff():
    usable = latest_usable_timestamp("mos_tmax_morning", "2026-07-01")
    assert usable == cutoff_instant_utc("2026-07-01")


def test_prior_day_features_do_not_reach_current_day():
    usable = latest_usable_timestamp("asos_prior_day_daily", "2026-07-01")
    # Must be strictly before midnight UTC start of the market day chunk we
    # would call "today"; well before the cutoff.
    assert usable < cutoff_instant_utc("2026-07-01")


# ---------------------------------------------------------------------------
# validate_cutoff_freshness
# ---------------------------------------------------------------------------

def test_fresh_record_passes():
    md = "2026-07-01"
    # 06:00 EDT ASOS ob == 10:00 UTC: exactly the 1h-pre-cutoff horizon and
    # within the 3h staleness window [08:00, 10:00] UTC.
    res = validate_cutoff_freshness(
        "asos_overnight_obs", datetime(2026, 7, 1, 10, tzinfo=timezone.utc), md
    )
    assert res.valid, res.errors


def test_post_cutoff_record_is_leakage():
    md = "2026-07-01"
    # A 12Z sounding would be post-cutoff for the sounding feature.
    res = validate_cutoff_freshness(
        "sounding_00z", datetime(2026, 7, 1, 12, tzinfo=timezone.utc), md
    )
    assert not res.valid
    assert any("leakage" in e for e in res.errors)


def test_overnight_record_after_cutoff_is_leakage():
    md = "2026-07-01"
    # An ob valid at the cutoff itself violates the 1h pre-cutoff horizon.
    res = validate_cutoff_freshness(
        "asos_overnight_obs", datetime(2026, 7, 1, 11, tzinfo=timezone.utc), md
    )
    assert not res.valid


def test_stale_record_flagged():
    md = "2026-07-01"
    spec = get_cutoff_spec("asos_overnight_obs")
    # Older than max_staleness_hours before the cutoff.
    too_old = cutoff_instant_utc(md) - timedelta(hours=spec.max_staleness_hours + 1)
    res = validate_cutoff_freshness("asos_overnight_obs", too_old, md)
    assert not res.valid
    assert any("stale" in e for e in res.errors)


def test_missing_record_flagged():
    res = validate_cutoff_freshness("mos_tmax_morning", None, "2026-07-01")
    assert not res.valid
    assert any("no available record" in e for e in res.errors)


def test_naive_datetime_assumed_utc():
    md = "2026-07-01"
    res = validate_cutoff_freshness(
        "asos_overnight_obs", datetime(2026, 7, 1, 10), md
    )
    assert res.valid, res.errors


# ---------------------------------------------------------------------------
# Aggregate validation + kill switch
# ---------------------------------------------------------------------------

def _all_fresh(md="2026-07-01"):
    cutoff = cutoff_instant_utc(md)
    return {
        "asos_prior_day_daily": cutoff - timedelta(hours=8),
        "asos_overnight_obs": cutoff - timedelta(hours=1),
        "mos_tmax_morning": cutoff - timedelta(hours=2),
        "sounding_00z": datetime(2026, 7, 1, 0, tzinfo=timezone.utc),
        "prior_day_settlement": cutoff - timedelta(hours=8),
    }


def test_all_fresh_passes_aggregate():
    res = validate_inference_freshness("chi", "2026-07-01", _all_fresh())
    assert res.valid, res.errors
    assert res.stats["kill_switch"] is False


def test_critical_failure_trips_kill_switch():
    ts = _all_fresh()
    ts["mos_tmax_morning"] = None  # critical, missing
    res = validate_inference_freshness("chi", "2026-07-01", ts)
    assert not res.valid
    assert res.stats["kill_switch"] is True
    with pytest.raises(KillSwitchError):
        enforce_inference_freshness("chi", "2026-07-01", ts)


def test_recommended_failure_degrades_not_halts():
    ts = _all_fresh()
    ts["sounding_00z"] = None  # recommended, missing
    res = validate_inference_freshness("chi", "2026-07-01", ts)
    # Recommended-only failure must NOT trip the kill switch.
    assert res.valid
    assert res.warnings
    # enforce should not raise.
    enforce_inference_freshness("chi", "2026-07-01", ts)


def test_require_features_subset():
    ts = {"asos_prior_day_daily": cutoff_instant_utc("2026-07-01") - timedelta(hours=8)}
    res = validate_inference_freshness(
        "chi", "2026-07-01", ts, require_features=["asos_prior_day_daily"]
    )
    assert res.valid, res.errors
    assert res.stats["n_features_checked"] == 1


# ---------------------------------------------------------------------------
# operational_features wiring
# ---------------------------------------------------------------------------

def test_operational_features_required_cutoff_mapping():
    import pandas as pd

    from src.operational_features import (
        assert_operational_features_cutoff_safe,
        required_cutoff_features,
    )

    df = pd.DataFrame(
        {
            "USW00094846_TMAX_lag1": [1.0],
            "sounding_t850_f_lag1": [2.0],
            "sin_day": [0.1],
        }
    )
    req = required_cutoff_features(df)
    assert "asos_prior_day_daily" in req
    assert "sounding_00z" in req

    md = "2026-07-01"
    cutoff = cutoff_instant_utc(md)
    ts = {
        "asos_prior_day_daily": cutoff - timedelta(hours=8),
        "sounding_00z": datetime(2026, 7, 1, 0, tzinfo=timezone.utc),
    }
    # Should not raise (only required features enforced).
    assert_operational_features_cutoff_safe("chi", md, ts, feature_df=df)
