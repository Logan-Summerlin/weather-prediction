"""Tests for the real-time EV dashboard opportunity service.

Covers config validation, signal loading (present/missing/stale), market
normalization, executable-price EV with curved fees + slippage, reason
codes, promotion-aware sizing, stale-market gating, and snapshot
persistence.  No network calls anywhere (mocked pollers / fixtures only).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from src.dashboard.opportunity_service import (
    ELIGIBLE_BLOCKED,
    ELIGIBLE_MONITOR,
    ELIGIBLE_NO_EDGE,
    ELIGIBLE_STALE,
    ELIGIBLE_TRADEABLE,
    REASON_CITY_NOT_PROMOTED,
    REASON_EV_BELOW_THRESHOLD,
    REASON_MARKET_STALE,
    REASON_SIGNAL_MISSING,
    REASON_SIGNAL_STALE,
    REASON_SPREAD_TOO_WIDE,
    DashboardConfig,
    MarketPoller,
    OpportunityService,
    compute_dashboard_ev,
    load_config,
    market_date_from_ticker,
    normalize_markets,
)
from src.trading import kalshi_fee_per_contract


NOW = datetime(2026, 7, 9, 15, 0, 0, tzinfo=timezone.utc)  # 11:00 ET
TODAY_ET = "2026-07-09"


def _now():
    return NOW


def make_market(ticker="KXHIGHNY-26JUL09-B86.87", title="86 to 87",
                yes_bid=20, yes_ask=22, no_bid=78, no_ask=80, **kw):
    m = {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "title": title,
        "status": "open",
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": 21,
        "volume": 100,
        "open_interest": 50,
    }
    m.update(kw)
    return m


def make_signal(city="nyc", date=TODAY_ET, mu=88.4, sigma=2.7,
                status="OK", promotion="MONITOR"):
    return {
        "city_code": city, "date": date, "status": status,
        "mu": mu, "sigma": sigma, "model_name": "synthesis",
        "bucket_labels": [], "bucket_probs": [], "bucket_edges": [],
        "freshness": {"valid": status == "OK", "kill_switch": status != "OK"},
        "kill_switch_reasons": [] if status == "OK" else ["stale asos"],
        "promotion_status": promotion,
        "generated_at": f"{date}T10:45:00+00:00",
    }


class FakePoller:
    """Deterministic in-memory market source (never touches the network)."""

    def __init__(self, markets_by_city, fresh=True, error=None):
        self.markets_by_city = markets_by_city
        self.fresh = fresh
        self.error = error

    def poll(self, city_code):
        return {
            "markets": self.markets_by_city.get(city_code, []),
            "polled_at": NOW.isoformat() if self.fresh or self.error is None else None,
            "fresh": self.fresh,
            "source": "live" if self.fresh else "none",
            "error": self.error,
        }


def make_service(tmp_path, markets_by_city, signals=None, fresh=True,
                 error=None, **cfg_kw):
    """Service wired to tmp dirs, fake poller, and tmp signal files."""
    cfg_kw.setdefault("mode", "monitor")
    cfg_kw.setdefault("export_dir", str(tmp_path / "export"))
    cfg_kw.setdefault("fixture_dir", str(tmp_path / "fixtures"))
    config = DashboardConfig(**cfg_kw)
    service = OpportunityService(
        config, poller=FakePoller(markets_by_city, fresh=fresh, error=error),
        now_fn=_now,
    )
    signal_dir = tmp_path / "signals"
    signal_dir.mkdir(exist_ok=True)
    signals = signals or {}
    paths = {}
    for city, sig in signals.items():
        p = signal_dir / f"{city}_signals_{sig['date']}.json"
        p.write_text(json.dumps(sig))
        paths[city] = str(p)
    service.find_latest_signal_path = lambda c: paths.get(c)
    return service


# ===========================================================================
# Config
# ===========================================================================

class TestConfig:
    def test_defaults_are_safe(self):
        cfg = DashboardConfig()
        assert cfg.mode == "monitor"
        assert cfg.cities == ["nyc", "chi", "phl"]
        assert cfg.show_monitor_sizing is False
        assert cfg.allow_stale_signals is False

    def test_live_mode_rejected(self):
        with pytest.raises(ValueError, match="live"):
            DashboardConfig(mode="live")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            DashboardConfig(mode="yolo")

    def test_unknown_city_rejected(self):
        with pytest.raises(ValueError):
            DashboardConfig(cities=["nyc", "atlantis"])

    def test_offline_mode_allows_stale_signals(self):
        cfg = DashboardConfig(mode="offline")
        assert cfg.allow_stale_signals is True

    def test_yaml_load_and_env_override(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "dash.yaml"
        yaml_path.write_text("mode: monitor\ncities: [nyc, chi]\nmin_ev: 0.05\n")
        monkeypatch.setenv("EV_DASHBOARD_MODE", "offline")
        cfg = load_config(path=str(yaml_path))
        assert cfg.mode == "offline"
        assert cfg.cities == ["nyc", "chi"]
        assert cfg.min_ev == 0.05

    def test_repo_default_yaml_is_valid(self):
        cfg = load_config(path=os.path.join("config", "dashboard.yaml"))
        assert cfg.cities == ["nyc", "chi", "phl"]

    def test_config_hash_stable(self):
        assert DashboardConfig().config_hash() == DashboardConfig().config_hash()


# ===========================================================================
# Market normalization
# ===========================================================================

class TestNormalization:
    def test_ticker_date_parsing(self):
        assert market_date_from_ticker("KXHIGHNY-26JUL09-B86.87") == "2026-07-09"
        assert market_date_from_ticker("garbage") is None

    def test_between_market(self):
        rows = normalize_markets([make_market()])
        r = rows[0]
        assert r["direction"] == "between"
        assert (r["lo"], r["hi"]) == (86.0, 87.0)
        assert r["yes_ask"] == pytest.approx(0.22)
        assert r["no_ask"] == pytest.approx(0.80)
        assert r["spread"] == pytest.approx(0.02)
        assert r["market_date"] == "2026-07-09"

    def test_above_and_below_markets(self):
        rows = normalize_markets([
            make_market(ticker="KXHIGHNY-26JUL09-T90", title="Above 90"),
            make_market(ticker="KXHIGHNY-26JUL09-B84", title="Below 84"),
        ])
        assert rows[0]["direction"] == "above" and rows[0]["lo"] == 90.0
        assert rows[0]["hi"] is None
        assert rows[1]["direction"] == "below" and rows[1]["hi"] == 84.0

    def test_missing_no_ask_falls_back_to_yes_bid(self):
        m = make_market(no_ask=None)
        r = normalize_markets([m])[0]
        assert r["no_ask"] == pytest.approx(1.0 - 0.20)

    def test_missing_quotes_yield_none(self):
        m = make_market(yes_bid=None, yes_ask=None, no_bid=None, no_ask=None)
        r = normalize_markets([m])[0]
        assert r["yes_ask"] is None and r["no_ask"] is None


# ===========================================================================
# EV engine
# ===========================================================================

class TestComputeDashboardEV:
    def test_yes_ev_uses_ask_not_mid(self):
        # model 0.30, yes bid/ask 20/28 -> mid 0.24 would flatter EV
        ev = compute_dashboard_ev(0.30, yes_ask=0.28, no_ask=0.80, slippage=0.0)
        fee = kalshi_fee_per_contract(0.28)
        assert ev["ev_yes"] == pytest.approx(0.30 - 0.28 - fee)
        assert ev["best_direction"] == "YES"
        assert ev["executable_price"] == pytest.approx(0.28)

    def test_no_side_ev(self):
        ev = compute_dashboard_ev(0.10, yes_ask=0.30, no_ask=0.75, slippage=0.0)
        fee = kalshi_fee_per_contract(0.75)
        assert ev["ev_no"] == pytest.approx(0.90 - 0.75 - fee)
        assert ev["best_direction"] == "NO"

    def test_slippage_applied(self):
        ev0 = compute_dashboard_ev(0.5, yes_ask=0.40, no_ask=0.62, slippage=0.0)
        ev2 = compute_dashboard_ev(0.5, yes_ask=0.40, no_ask=0.62, slippage=0.02)
        assert ev2["ev_yes"] < ev0["ev_yes"]

    def test_positive_at_mid_negative_after_costs(self):
        # model 0.25 vs mid 0.24 (bid 0.20 / ask 0.28): mid says +EV,
        # executable ask + fee says clearly negative.
        ev = compute_dashboard_ev(0.25, yes_ask=0.28, no_ask=0.81, slippage=0.01)
        assert ev["ev_yes"] < 0
        assert ev["ev_no"] < 0
        assert ev["ev_best"] < 0

    def test_missing_quotes(self):
        ev = compute_dashboard_ev(0.5, yes_ask=None, no_ask=None)
        assert ev["best_direction"] == "NONE"
        assert ev["ev_best"] is None


# ===========================================================================
# Signal / city state
# ===========================================================================

class TestSignals:
    def test_missing_signal_blocks_city(self, tmp_path):
        service = make_service(tmp_path, {"nyc": [make_market()]}, signals={},
                               cities=["nyc"])
        snap = service.build_snapshot()
        state = snap.cities["nyc"]
        assert state["status"] == "blocked"
        assert REASON_SIGNAL_MISSING in state["reason_codes"]
        assert state["refresh_command"].endswith("--city nyc")
        assert all(r["eligibility"] == ELIGIBLE_BLOCKED
                   for r in snap.opportunities)

    def test_stale_signal_blocks_city(self, tmp_path):
        service = make_service(
            tmp_path, {"nyc": [make_market()]},
            signals={"nyc": make_signal(date="2026-07-07")}, cities=["nyc"],
        )
        snap = service.build_snapshot()
        assert snap.cities["nyc"]["status"] == "blocked"
        assert REASON_SIGNAL_STALE in snap.cities["nyc"]["reason_codes"]

    def test_stale_signal_allowed_when_configured(self, tmp_path):
        service = make_service(
            tmp_path, {"nyc": [make_market()]},
            signals={"nyc": make_signal(date="2026-07-07")},
            cities=["nyc"], allow_stale_signals=True,
        )
        snap = service.build_snapshot()
        assert snap.cities["nyc"]["status"] == "ok"

    def test_kill_switch_signal_blocks_city(self, tmp_path):
        sig = make_signal(status="KILL_SWITCH", mu=None, sigma=None)
        service = make_service(tmp_path, {"nyc": [make_market()]},
                               signals={"nyc": sig}, cities=["nyc"])
        snap = service.build_snapshot()
        state = snap.cities["nyc"]
        assert state["status"] == "blocked"
        assert state["kill_switch_active"] is True

    def test_fresh_signal_ok(self, tmp_path):
        service = make_service(tmp_path, {"nyc": [make_market()]},
                               signals={"nyc": make_signal()}, cities=["nyc"])
        snap = service.build_snapshot()
        assert snap.cities["nyc"]["status"] == "ok"
        assert snap.cities["nyc"]["mu"] == pytest.approx(88.4)


# ===========================================================================
# Opportunity rows: eligibility, sizing, reason codes
# ===========================================================================

class TestOpportunities:
    def test_monitor_city_positive_ev_size_zero(self, tmp_path):
        # Cheap "Below 84" priced way under model probability -> +EV
        markets = [make_market(ticker="KXHIGHNY-26JUL09-B84", title="Below 84",
                               yes_bid=2, yes_ask=3, no_bid=97, no_ask=98)]
        sig = make_signal(mu=84.0, sigma=3.0, promotion="MONITOR")
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": sig}, cities=["nyc"])
        snap = service.build_snapshot()
        row = snap.opportunities[0]
        assert row["ev_best"] > 0
        assert row["eligibility"] == ELIGIBLE_MONITOR
        assert REASON_CITY_NOT_PROMOTED in row["reason_codes"]
        assert row["suggested_size"] == 0

    def test_promoted_city_positive_ev_gets_size(self, tmp_path):
        markets = [make_market(ticker="KXHIGHNY-26JUL09-B84", title="Below 84",
                               yes_bid=2, yes_ask=3, no_bid=97, no_ask=98)]
        sig = make_signal(mu=84.0, sigma=3.0, promotion="PROMOTED")
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": sig}, cities=["nyc"])
        snap = service.build_snapshot()
        row = snap.opportunities[0]
        assert row["eligibility"] == ELIGIBLE_TRADEABLE
        assert row["suggested_size"] > 0

    def test_no_edge_row(self, tmp_path):
        # Market priced at model probability (~13c) -> costs make it -EV
        sig = make_signal(mu=86.5, sigma=3.0)
        markets = [make_market(yes_bid=12, yes_ask=14, no_bid=86, no_ask=88)]
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": sig}, cities=["nyc"])
        snap = service.build_snapshot()
        row = snap.opportunities[0]
        assert row["eligibility"] == ELIGIBLE_NO_EDGE
        assert REASON_EV_BELOW_THRESHOLD in row["reason_codes"]
        assert row["suggested_size"] == 0

    def test_wide_spread_rejected(self, tmp_path):
        markets = [make_market(ticker="KXHIGHNY-26JUL09-B84", title="Below 84",
                               yes_bid=1, yes_ask=30, no_bid=70, no_ask=99)]
        sig = make_signal(mu=80.0, sigma=2.0, promotion="PROMOTED")
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": sig}, cities=["nyc"],
                               max_spread_cents=10.0)
        snap = service.build_snapshot()
        row = snap.opportunities[0]
        assert REASON_SPREAD_TOO_WIDE in row["reason_codes"]
        assert row["suggested_size"] == 0
        assert row["eligibility"] != ELIGIBLE_TRADEABLE

    def test_unparseable_contract_hidden_from_ranking(self, tmp_path):
        markets = [make_market(ticker="KXHIGHNY-26JUL09-X", title="mystery event")]
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": make_signal()}, cities=["nyc"])
        snap = service.build_snapshot()
        row = snap.opportunities[0]
        assert row["eligibility"] == ELIGIBLE_BLOCKED
        assert row["reason_codes"] == ["contract_parse_failed"]
        assert row["ev_best"] is None

    def test_failed_poll_never_fresh_opportunity(self, tmp_path):
        markets = [make_market(ticker="KXHIGHNY-26JUL09-B84", title="Below 84",
                               yes_bid=2, yes_ask=3, no_bid=97, no_ask=98)]
        sig = make_signal(mu=84.0, sigma=3.0, promotion="PROMOTED")
        service = make_service(tmp_path, {"nyc": markets},
                               signals={"nyc": sig}, cities=["nyc"],
                               fresh=False, error="connection reset")
        snap = service.build_snapshot()
        assert snap.cities["nyc"]["status"] == "stale"
        row = snap.opportunities[0]
        assert row["eligibility"] == ELIGIBLE_STALE
        assert REASON_MARKET_STALE in row["reason_codes"]
        assert row["suggested_size"] == 0
        assert snap.summary["n_tradeable_paper"] == 0
        assert any("connection reset" in w for w in snap.warnings)


# ===========================================================================
# Snapshot persistence
# ===========================================================================

class TestPersistence:
    def test_refresh_writes_snapshot_and_latest(self, tmp_path):
        service = make_service(tmp_path, {"nyc": [make_market()]},
                               signals={"nyc": make_signal()}, cities=["nyc"])
        snap = service.refresh()
        export = service.config.export_dir
        files = [f for f in os.listdir(export)
                 if f.startswith("opportunities_")]
        assert len(files) == 1
        with open(os.path.join(export, "latest_opportunities.json")) as f:
            latest = json.load(f)
        assert latest["generated_at"] == snap.generated_at
        assert latest["mode"] == "monitor"
        assert latest["config_hash"] == service.config.config_hash()
        assert latest["summary"]["fee_model"]

    def test_snapshot_retention(self, tmp_path):
        service = make_service(tmp_path, {"nyc": [make_market()]},
                               signals={"nyc": make_signal()}, cities=["nyc"],
                               snapshot_retention=2)
        import time
        for _ in range(3):
            service.refresh()
            time.sleep(1.1)  # timestamped filenames have 1s resolution
        files = [f for f in os.listdir(service.config.export_dir)
                 if f.startswith("opportunities_")]
        assert len(files) == 2

    def test_dashboard_data_reads_snapshot(self, tmp_path):
        from src.dashboard.dashboard_data import DashboardData

        service = make_service(tmp_path, {"nyc": [make_market()]},
                               signals={"nyc": make_signal()}, cities=["nyc"])
        service.refresh()
        counts = DashboardData(city_codes=["nyc"]).get_opportunity_counts(
            export_dir=service.config.export_dir
        )
        assert counts["available"] is True
        assert counts["by_city"]["nyc"]["n_contracts"] == 1


# ===========================================================================
# Offline fixture mode (end-to-end, no network)
# ===========================================================================

class TestOfflineMode:
    def test_bundled_fixtures_end_to_end(self, tmp_path):
        config = DashboardConfig(mode="offline",
                                 export_dir=str(tmp_path / "export"))
        service = OpportunityService(config, now_fn=_now)
        snap = service.build_snapshot()
        assert snap.mode == "offline"
        assert snap.summary["n_contracts"] > 0
        # Deterministic under fixed clock: rebuild matches (minus timestamps)
        snap2 = service.build_snapshot()
        assert snap.opportunities == snap2.opportunities
        assert snap.summary == snap2.summary
        # Every positive-EV row carries explicit cost assumptions + reasons
        for r in snap.opportunities:
            if r["ev_best"] is not None and r["ev_best"] > 0:
                assert r["fee_per_contract"] is not None
                assert r["fee_model"]
                assert r["eligibility"] in (
                    ELIGIBLE_TRADEABLE, ELIGIBLE_MONITOR, ELIGIBLE_NO_EDGE,
                )
        # MONITOR cities (chi/phl fixtures) never sized
        for r in snap.opportunities:
            if r["city_code"] in ("chi", "phl"):
                assert r["suggested_size"] == 0

    def test_offline_mode_makes_no_network_calls(self, tmp_path, monkeypatch):
        import src.kalshi_client as kc

        def boom(*a, **kw):
            raise AssertionError("network call in offline mode")

        monkeypatch.setattr(kc.KalshiClient, "_request", boom, raising=True)
        config = DashboardConfig(mode="offline",
                                 export_dir=str(tmp_path / "export"))
        OpportunityService(config, now_fn=_now).build_snapshot()


# ===========================================================================
# MarketPoller last-good cache
# ===========================================================================

class TestMarketPoller:
    def test_failed_poll_uses_last_good_cache(self, tmp_path):
        config = DashboardConfig(export_dir=str(tmp_path / "export"),
                                 cities=["nyc"])

        class GoodClient:
            def get_markets(self, **kw):
                return [make_market()]

        class BadClient:
            def get_markets(self, **kw):
                raise ConnectionError("kalshi down")

        good = MarketPoller(config, client=GoodClient())
        first = good.poll("nyc")
        assert first["fresh"] is True and first["source"] == "live"

        bad = MarketPoller(config, client=BadClient())
        second = bad.poll("nyc")
        assert second["fresh"] is False
        assert second["source"] == "last_good_cache"
        assert len(second["markets"]) == 1
        assert "kalshi down" in second["error"]

    def test_failed_poll_no_cache(self, tmp_path):
        config = DashboardConfig(export_dir=str(tmp_path / "export"))

        class BadClient:
            def get_markets(self, **kw):
                raise ConnectionError("down")

        result = MarketPoller(config, client=BadClient()).poll("nyc")
        assert result["fresh"] is False and result["source"] == "none"
        assert result["markets"] == []
