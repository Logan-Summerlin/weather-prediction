"""
Tests for the unified promotion report module (Phase F).

Covers:
  1. PromotionGate evaluation logic
  2. Data extraction helpers (baselines, market, calibration, trading)
  3. Full evaluate_city integration against existing city result artifacts
  4. Report schema validation (all required fields present)
  5. Report I/O (save and load roundtrip)
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.promotion_report import (
    CITY_THRESHOLDS,
    PromotionGate,
    _extract_baseline_briers,
    _extract_calibration_summary,
    _extract_market_brier,
    _extract_trading_summary,
    _find_best_u_variant,
    _load_best_brier,
    _load_json,
    evaluate_city,
    save_unified_report,
)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ===========================================================================
# PromotionGate unit tests
# ===========================================================================

class TestPromotionGate:
    def test_evaluate_less(self):
        g = PromotionGate("test", "test gate")
        assert g.evaluate(0.1, 0.2, "less") is True
        assert g.passed is True
        assert g.value == 0.1
        assert g.threshold == 0.2

    def test_evaluate_less_fail(self):
        g = PromotionGate("test", "test gate")
        assert g.evaluate(0.3, 0.2, "less") is False
        assert g.passed is False

    def test_evaluate_greater(self):
        g = PromotionGate("test", "test gate")
        assert g.evaluate(100, 50, "greater") is True
        assert g.passed is True

    def test_evaluate_greater_fail(self):
        g = PromotionGate("test", "test gate")
        assert g.evaluate(30, 50, "greater") is False

    def test_evaluate_abs_less(self):
        g = PromotionGate("test", "test gate")
        assert g.evaluate(-0.05, 0.1, "abs_less") is True
        assert g.evaluate(-0.15, 0.1, "abs_less") is False

    def test_to_dict(self):
        g = PromotionGate("test_gate", "A test", category="forecast_quality")
        g.evaluate(0.12, 0.15, "less")
        g.details = "some details"
        d = g.to_dict()
        assert d["name"] == "test_gate"
        assert d["description"] == "A test"
        assert d["category"] == "forecast_quality"
        assert d["passed"] is True
        assert d["value"] == 0.12
        assert d["threshold"] == 0.15
        assert d["details"] == "some details"

    def test_default_state(self):
        g = PromotionGate("x", "y")
        assert g.passed is False
        assert g.value is None
        assert g.threshold is None


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestLoadJson:
    def test_load_existing(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "val"}))
        result = _load_json(str(p))
        assert result == {"key": "val"}

    def test_load_missing(self, tmp_path):
        result = _load_json(str(tmp_path / "nonexistent.json"))
        assert result is None


class TestFindBestUVariant:
    def test_finds_best(self):
        benchmark = {
            "E0_persistence": {"test_brier": 0.015},
            "U3_contract_mlp": {"contract_brier": 0.114},
            "U7_extended_mlp": {"contract_brier": 0.109},
            "U9_kitchen_sink": {"contract_brier": 0.110},
            "Kalshi_PreSettlement": {"contract_brier": 0.125},
        }
        name, brier = _find_best_u_variant(benchmark)
        assert name == "U7_extended_mlp"
        assert brier == 0.109

    def test_no_u_variants(self):
        benchmark = {
            "E0_persistence": {"test_brier": 0.015},
            "Kalshi_PreSettlement": {"contract_brier": 0.125},
        }
        name, brier = _find_best_u_variant(benchmark)
        assert name is None
        assert brier == float("inf")

    def test_empty(self):
        name, brier = _find_best_u_variant({})
        assert name is None


class TestExtractBaselineBriers:
    def test_from_unified_benchmark(self, tmp_path):
        unified = {
            "E0_persistence": {"test_brier": 0.015},
            "E1_climatology": {"test_brier": 0.016},
            "E2_ridge": {"test_brier": 0.014, "alpha": 100.0},
            "U7_extended_mlp": {"contract_brier": 0.109},
        }
        (tmp_path / "unified_benchmark_results.json").write_text(json.dumps(unified))
        baselines = _extract_baseline_briers(str(tmp_path))
        assert baselines["persistence"] == 0.015
        assert baselines["climatology"] == 0.016
        assert baselines["ridge"] == 0.014

    def test_from_honest_benchmark(self, tmp_path):
        honest = {
            "E0_persistence": {"test_brier": 0.018},
            "E1_climatology": {"test_brier": 0.019},
            "E2_ridge": {"test_brier": 0.017},
        }
        (tmp_path / "honest_benchmark_results.json").write_text(json.dumps(honest))
        baselines = _extract_baseline_briers(str(tmp_path))
        assert baselines["persistence"] == 0.018

    def test_empty_dir(self, tmp_path):
        baselines = _extract_baseline_briers(str(tmp_path))
        assert baselines["persistence"] is None
        assert baselines["climatology"] is None
        assert baselines["ridge"] is None


class TestExtractMarketBrier:
    def test_from_unified_benchmark(self, tmp_path):
        unified = {
            "Kalshi_PreSettlement": {"contract_brier": 0.125},
            "U7_extended_mlp": {"contract_brier": 0.109},
        }
        (tmp_path / "unified_benchmark_results.json").write_text(json.dumps(unified))
        result = _extract_market_brier(str(tmp_path))
        assert result == 0.125

    def test_from_real_kalshi_backtest(self, tmp_path):
        bt_dir = tmp_path / "backtest"
        bt_dir.mkdir()
        rk = {"market_brier": 0.130, "variants": {}}
        (bt_dir / "real_kalshi_metrics.json").write_text(json.dumps(rk))
        result = _extract_market_brier(str(tmp_path))
        assert result == 0.130

    def test_not_found(self, tmp_path):
        result = _extract_market_brier(str(tmp_path))
        assert result is None


# ===========================================================================
# Report schema validation
# ===========================================================================

REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "timestamp",
    "city",
    "city_code",
    "kalshi_ticker",
    "target_station",
    "target_station_name",
    "overall_status",
    "gates_passed",
    "gates_total",
    "pass_rate",
    "thresholds",
    "gates",
    "model_summary",
    "baseline_comparisons",
    "market_benchmark",
    "calibration_summary",
    "trading_summary",
    "seasonal_brier",
]

REQUIRED_GATE_KEYS = [
    "name",
    "description",
    "category",
    "passed",
    "value",
    "threshold",
    "details",
]

REQUIRED_BASELINE_KEYS = [
    "persistence_brier",
    "climatology_brier",
    "ridge_brier",
    "model_beats_persistence",
    "model_beats_climatology",
    "model_beats_ridge",
]

REQUIRED_MARKET_KEYS = [
    "kalshi_presettlement_brier",
    "model_beats_market",
    "brier_edge",
]


# ===========================================================================
# Integration tests against real city data
# ===========================================================================

# Only run integration tests if city result dirs exist
def _city_has_results(city_code: str) -> bool:
    from src.city_config import get_city_config
    try:
        cfg = get_city_config(city_code)
        return os.path.isdir(cfg.results_dir)
    except Exception:
        return False


AVAILABLE_CITIES = [c for c in CITY_THRESHOLDS if _city_has_results(c)]


@pytest.mark.parametrize("city_code", AVAILABLE_CITIES)
class TestEvaluateCityIntegration:
    """Integration tests running evaluate_city against real result artifacts."""

    def test_evaluate_returns_gates_and_report(self, city_code):
        gates, report = evaluate_city(city_code)
        assert isinstance(gates, list)
        assert len(gates) > 0
        assert isinstance(report, dict)

    def test_report_has_required_keys(self, city_code):
        _, report = evaluate_city(city_code)
        for key in REQUIRED_TOP_LEVEL_KEYS:
            assert key in report, f"Missing top-level key: {key}"

    def test_gates_have_required_keys(self, city_code):
        _, report = evaluate_city(city_code)
        for gate in report["gates"]:
            for key in REQUIRED_GATE_KEYS:
                assert key in gate, f"Gate '{gate.get('name')}' missing key: {key}"

    def test_baseline_comparisons_have_required_keys(self, city_code):
        _, report = evaluate_city(city_code)
        baselines = report["baseline_comparisons"]
        for key in REQUIRED_BASELINE_KEYS:
            assert key in baselines, f"Missing baseline key: {key}"

    def test_market_benchmark_has_required_keys(self, city_code):
        _, report = evaluate_city(city_code)
        market = report["market_benchmark"]
        for key in REQUIRED_MARKET_KEYS:
            assert key in market, f"Missing market key: {key}"

    def test_schema_version(self, city_code):
        _, report = evaluate_city(city_code)
        assert report["schema_version"] == "3.0"

    def test_city_code_matches(self, city_code):
        _, report = evaluate_city(city_code)
        assert report["city_code"] == city_code

    def test_gates_count_consistency(self, city_code):
        gates, report = evaluate_city(city_code)
        assert len(gates) == report["gates_total"]
        passed_count = sum(1 for g in gates if g.passed)
        assert passed_count == report["gates_passed"]

    def test_overall_status_consistency(self, city_code):
        gates, report = evaluate_city(city_code)
        all_pass = all(g.passed for g in gates)
        if all_pass:
            assert report["overall_status"] == "PASS"
        else:
            assert report["overall_status"] == "FAIL"

    def test_report_is_json_serializable(self, city_code):
        _, report = evaluate_city(city_code)
        # Should not raise
        json_str = json.dumps(report, default=str)
        assert len(json_str) > 0


# ===========================================================================
# Save/load roundtrip test
# ===========================================================================

class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        report = {
            "schema_version": "3.0",
            "timestamp": "2026-03-08T00:00:00",
            "city": "Test City",
            "city_code": "tst",
            "overall_status": "PASS",
            "gates": [{"name": "g1", "passed": True}],
            "baseline_comparisons": {"persistence_brier": 0.015},
            "market_benchmark": {"kalshi_presettlement_brier": 0.125},
        }
        path = str(tmp_path / "test_report.json")
        save_unified_report(report, path)

        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["schema_version"] == "3.0"
        assert loaded["overall_status"] == "PASS"
        assert loaded["baseline_comparisons"]["persistence_brier"] == 0.015
        assert loaded["market_benchmark"]["kalshi_presettlement_brier"] == 0.125


# ===========================================================================
# Thresholds coverage test
# ===========================================================================

class TestThresholdsCoverage:
    """Verify all registered cities have promotion thresholds."""

    def test_all_cities_have_thresholds(self):
        from src.city_config import list_cities
        registered = list_cities()
        for city in registered:
            assert city in CITY_THRESHOLDS, (
                f"City '{city}' is registered but has no promotion thresholds"
            )

    def test_thresholds_have_required_keys(self):
        required = [
            "brier_threshold",
            "ece_threshold",
            "max_drawdown_threshold",
            "min_oos_days",
            "seasonal_brier_threshold",
        ]
        for city, thresholds in CITY_THRESHOLDS.items():
            for key in required:
                assert key in thresholds, (
                    f"City '{city}' missing threshold key: {key}"
                )
