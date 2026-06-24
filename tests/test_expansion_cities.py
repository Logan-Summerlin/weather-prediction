"""Tests for the Phase 4 expansion city registration.

Confirms den/dc/lax/mia/phx are registered consistently across city_config,
the runtime data, SUPPORTED_CITIES, CITY_THRESHOLDS (entropy-derived), and the
Kalshi fetch-script list, and that the verification artifact's recommended set
matches the registered cities.
"""

import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.city_config import get_city_config, get_city_runtime_config, list_cities  # noqa: E402
from src.promotion_report import CITY_THRESHOLDS, climatology_ladder_brier  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPANSION = ["den", "dc", "lax", "mia", "phx"]

# Verified series tickers (results/expansion/contract_verification.json).
EXPECTED_TICKERS = {
    "den": "KXHIGHDEN",
    "dc": "KXHIGHTDC",
    "lax": "KXHIGHLAX",
    "mia": "KXHIGHMIA",
    "phx": "KXHIGHTPHX",
}
EXPECTED_ASOS = {
    "den": "KDEN", "dc": "KDCA", "lax": "KLAX", "mia": "KMIA", "phx": "KPHX",
}


class TestRegistration:
    def test_all_expansion_cities_registered(self):
        cities = list_cities()
        for code in EXPANSION:
            assert code in cities

    @pytest.mark.parametrize("code", EXPANSION)
    def test_config_ticker_and_station(self, code):
        cfg = get_city_config(code)
        assert cfg.kalshi_ticker == EXPECTED_TICKERS[code]
        assert cfg.target_station.startswith("USW")
        assert cfg.asos_station_map.get(cfg.target_station) == EXPECTED_ASOS[code]
        assert cfg.timezone  # non-empty IANA tz

    @pytest.mark.parametrize("code", EXPANSION)
    def test_bucket_grid_well_formed(self, code):
        cfg = get_city_config(code)
        assert len(cfg.bucket_edges) == len(cfg.bucket_labels) > 0
        # Interior buckets are 2°F wide (matches the verified Kalshi ladder).
        lo, hi = cfg.bucket_edges[1]
        assert hi - lo == 2
        # Edges are contiguous and monotonic.
        for (a_lo, a_hi), (b_lo, b_hi) in zip(cfg.bucket_edges, cfg.bucket_edges[1:]):
            assert a_hi == b_lo

    @pytest.mark.parametrize("code", EXPANSION)
    def test_climatology_complete(self, code):
        cfg = get_city_config(code)
        assert set(cfg.monthly_tmax_mean) == set(range(1, 13))
        assert set(cfg.monthly_tmax_std) == set(range(1, 13))
        assert all(v > 0 for v in cfg.monthly_tmax_std.values())

    @pytest.mark.parametrize("code", EXPANSION)
    def test_runtime_config_resolves(self, code):
        rc = get_city_runtime_config(code)
        assert rc.TARGET_STATION == get_city_config(code).target_station
        assert rc.START_DATE and rc.END_DATE


class TestThresholds:
    @pytest.mark.parametrize("code", EXPANSION)
    def test_threshold_entry_exists(self, code):
        assert code in CITY_THRESHOLDS
        t = CITY_THRESHOLDS[code]
        assert t["max_drawdown_threshold"] == -0.30
        assert t["min_oos_days"] == 200

    @pytest.mark.parametrize("code", EXPANSION)
    def test_brier_threshold_is_entropy_derived(self, code):
        cfg = get_city_config(code)
        expected = round(climatology_ladder_brier(cfg.monthly_tmax_std), 2)
        assert CITY_THRESHOLDS[code]["brier_threshold"] == expected

    @pytest.mark.parametrize("code", EXPANSION)
    def test_nws_baseline_below_climatology_ceiling(self, code):
        t = CITY_THRESHOLDS[code]
        # A real forecast must beat climatology, so the NWS baseline sits at or
        # below the climatology-derived brier_threshold.
        assert t["nws_brier_baseline"] <= t["brier_threshold"]


class TestPipelineWiring:
    @staticmethod
    def _load(name):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(REPO_ROOT, "scripts", f"{name}.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so dataclass annotation resolution can find the
        # module in sys.modules.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_supported_cities_includes_expansion(self):
        mod = self._load("run_city_pipeline")
        for code in EXPANSION:
            assert code in mod.SUPPORTED_CITIES

    def test_fetch_script_lists_expansion(self):
        mod = self._load("fetch_kalshi_multi_city")
        for code in EXPANSION:
            assert code in mod.CITY_CONFIG
            assert mod.CITY_CONFIG[code]["series_ticker"] == EXPECTED_TICKERS[code]


class TestVerificationArtifact:
    def test_artifact_recommendation_matches_registry(self):
        path = os.path.join(REPO_ROOT, "results", "expansion",
                            "contract_verification.json")
        if not os.path.exists(path):
            pytest.skip("verification artifact not generated in this checkout")
        with open(path) as fh:
            payload = json.load(fh)
        assert set(payload["recommended"]) == set(EXPANSION)
        # Every recommended city must be VERIFIED + tradeable in the artifact.
        by_code = {r["code"]: r for r in payload["results"]}
        for code in payload["recommended"]:
            assert by_code[code]["status"] == "VERIFIED"
            assert by_code[code]["tradeable"] is True
