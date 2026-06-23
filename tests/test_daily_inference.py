"""Tests for Phase 3.3 daily cutoff-safe inference + kill switch."""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest

from src import daily_inference as di
from src.data_sla import cutoff_instant_utc, get_cutoff_spec


MARKET_DATE = "2025-06-10"


def test_assumed_fresh_passes_and_produces_buckets(tmp_path):
    signal = di.run_daily_inference(
        "chi", MARKET_DATE, mu=72.0, sigma=6.0,
        live_dir=str(tmp_path), write=True,
    )
    assert signal.status == "OK"
    assert signal.mu == 72.0
    probs = np.array(signal.bucket_probs)
    assert len(probs) == len(signal.bucket_labels)
    # gaussian_to_bucket_probs normalizes then clips to [0.001, 0.999]; the
    # post-clip sum is approximately (not exactly) 1 across many buckets.
    assert probs.sum() == pytest.approx(1.0, abs=0.05)
    assert (probs > 0).all()
    # file written
    path = di.signals_path("chi", MARKET_DATE, str(tmp_path))
    assert json.loads(open(path).read())["status"] == "OK"


def test_kill_switch_on_stale_critical_feature(tmp_path):
    spec = get_cutoff_spec("mos_tmax_morning")
    cutoff = cutoff_instant_utc(MARKET_DATE)
    timestamps = di.assumed_fresh_timestamps(MARKET_DATE, di.DEFAULT_REQUIRED_FEATURES)
    # Make MOS stale beyond tolerance.
    timestamps["mos_tmax_morning"] = cutoff - timedelta(
        hours=spec.max_staleness_hours + 12.0
    )
    signal = di.run_daily_inference(
        "chi", MARKET_DATE, available_timestamps=timestamps,
        mu=72.0, sigma=6.0, live_dir=str(tmp_path), write=True,
    )
    assert signal.status == "KILL_SWITCH"
    assert signal.kill_switch_reasons
    assert signal.bucket_probs == []  # no forecast emitted under kill switch
    # kill switch is still auditable on disk
    path = di.signals_path("chi", MARKET_DATE, str(tmp_path))
    assert json.loads(open(path).read())["status"] == "KILL_SWITCH"


def test_leakage_future_timestamp_trips_kill_switch(tmp_path):
    cutoff = cutoff_instant_utc(MARKET_DATE)
    timestamps = di.assumed_fresh_timestamps(MARKET_DATE, di.DEFAULT_REQUIRED_FEATURES)
    # A record valid AFTER the cutoff is post-cutoff leakage.
    timestamps["asos_overnight_obs"] = cutoff + timedelta(hours=3)
    signal = di.run_daily_inference(
        "chi", MARKET_DATE, available_timestamps=timestamps,
        mu=70.0, sigma=5.0, live_dir=str(tmp_path), write=True,
    )
    assert signal.status == "KILL_SWITCH"


def test_signal_to_prediction_roundtrip(tmp_path):
    signal = di.run_daily_inference(
        "phl", MARKET_DATE, mu=80.0, sigma=5.0,
        live_dir=str(tmp_path), write=True,
    )
    path = di.signals_path("phl", MARKET_DATE, str(tmp_path))
    loaded = di.load_signal(path)
    pred = di.signal_to_prediction(loaded)
    assert pred.city_code == "phl"
    assert pred.mu == 80.0
    assert len(pred.bucket_probs) == len(pred.bucket_labels)


def test_required_features_subset_only_checks_those(tmp_path):
    # Provide only the two ASOS features; restrict the requirement to them so
    # the absent MOS entry does not trip the switch.
    timestamps = di.assumed_fresh_timestamps(
        MARKET_DATE, ["asos_prior_day_daily", "asos_overnight_obs"]
    )
    signal = di.run_daily_inference(
        "chi", MARKET_DATE, available_timestamps=timestamps,
        require_features=["asos_prior_day_daily", "asos_overnight_obs"],
        mu=68.0, sigma=6.0, live_dir=str(tmp_path), write=True,
    )
    assert signal.status == "OK"
