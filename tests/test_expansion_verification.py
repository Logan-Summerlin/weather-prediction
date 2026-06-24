"""Tests for src/expansion_verification.py (Phase 4.1).

All network-free: a FakeKalshiClient returns canned series/markets so the
discovery, parsing, liquidity, scoring, and selection logic are exercised
without touching the live API. Mirrors the real API's irregular ticker
naming and current ``*_dollars`` / ``*_fp`` market schema.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.expansion_verification import (  # noqa: E402
    CANDIDATE_CITIES,
    CandidateCity,
    discover_series_ticker,
    extract_settlement_station,
    run_verification,
    sample_liquidity,
    select_recommended,
    spread_adjusted_liquidity,
    summarize_bucket_structure,
    verify_city,
)


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------
def _market(event, sub, vol, oi, bid=None, ask=None, rules=None):
    m = {
        "ticker": f"{event}-X",
        "event_ticker": event,
        "title": f"Will the high temp be >X on {event}?",
        "yes_sub_title": sub,
        "volume_fp": vol,
        "open_interest_fp": oi,
    }
    if bid is not None:
        m["yes_bid_dollars"] = bid
    if ask is not None:
        m["yes_ask_dollars"] = ask
    if rules is not None:
        m["rules_primary"] = rules
    return m


def _day(ticker, day, rules, vol=100, oi=80, bid="0.40", ask="0.42"):
    subs = ["91° or above", "82° or below", "89° to 90°",
            "87° to 88°", "85° to 86°", "83° to 84°"]
    ev = f"{ticker}-{day}"
    return [
        _market(ev, subs[i], vol, oi,
                bid=bid if i == 0 else None,
                ask=ask if i == 0 else None,
                rules=rules)
        for i in range(len(subs))
    ]


class FakeKalshiClient:
    """Implements get_series / get_markets from a static fixture."""

    def __init__(self, series, markets_open, markets_settled):
        self._series = series                # ticker -> title
        self._open = markets_open            # ticker -> [market]
        self._settled = markets_settled      # ticker -> [market]

    def get_series(self, series_ticker):
        if series_ticker not in self._series:
            raise ValueError(f"404 {series_ticker}")
        return {"title": self._series[series_ticker]}

    def get_markets(self, series_ticker, status="open", limit=200):
        if status == "open":
            return list(self._open.get(series_ticker, []))
        return list(self._settled.get(series_ticker, []))


DEN_RULES = ("If the highest temperature recorded in Denver, CO for June 24, "
             "2026 as reported by the National Weather Service's "
             "Climatological Report (Daily), is greater than 90°, ...")
MIA_RULES = ("If the highest temperature recorded at Miami International "
             "Airport for June 24, 2026 as reported by the National Weather "
             "Service's Climatological Report (Daily), is greater than 95°, ...")


def build_client():
    series = {
        "KXHIGHDEN": "Highest temperature in Denver",
        "KXHIGHMIA": "Highest temperature in Miami",
        "KXHIGHHOU": "Highest temperature in Houston",  # exists but empty
    }
    open_m = {
        "KXHIGHDEN": _day("KXHIGHDEN", "26JUN24", DEN_RULES),
        "KXHIGHMIA": _day("KXHIGHMIA", "26JUN24", MIA_RULES, vol=500, oi=400,
                          bid="0.49", ask="0.50"),
        "KXHIGHHOU": [],
    }
    settled = {
        "KXHIGHDEN": _day("KXHIGHDEN", "26JUN23", DEN_RULES),
        "KXHIGHMIA": _day("KXHIGHMIA", "26JUN23", MIA_RULES, vol=500, oi=400),
        "KXHIGHHOU": [],
    }
    return FakeKalshiClient(series, open_m, settled)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
class TestSettlementStationParsing:
    def test_recorded_in_pattern(self):
        assert extract_settlement_station(DEN_RULES) == "Denver, CO"

    def test_recorded_at_pattern(self):
        assert extract_settlement_station(MIA_RULES) == "Miami International Airport"

    def test_washington_dc_at_pattern(self):
        rules = ("If the maximum temperature recorded at Washington DC for Jun "
                 "24, 2026, is greater than 86° fahrenheit according to ...")
        assert extract_settlement_station(rules) == "Washington DC"

    def test_none_on_unparseable(self):
        assert extract_settlement_station(None) is None
        assert extract_settlement_station("no station phrase here") is None


class TestBucketStructure:
    def test_counts_largest_event(self):
        markets = _day("KXHIGHDEN", "26JUN24", DEN_RULES)
        s = summarize_bucket_structure(markets)
        assert s["n_buckets"] == 6
        assert s["modal_width_f"] == 2.0
        assert "89° to 90°" in s["bucket_labels"]

    def test_empty(self):
        assert summarize_bucket_structure([])["n_buckets"] == 0


class TestLiquidity:
    def test_aggregates_volume_and_spread(self):
        markets = _day("KXHIGHDEN", "26JUN24", DEN_RULES, vol=100, oi=80,
                       bid="0.40", ask="0.42")
        liq = sample_liquidity(markets)
        assert liq["total_volume"] == pytest.approx(600)   # 6 buckets * 100
        assert liq["total_open_interest"] == pytest.approx(480)
        assert liq["mean_spread_cents"] == pytest.approx(2.0)  # 0.42-0.40
        assert liq["n_quoted_markets"] == 1

    def test_spread_adjusted_penalizes_wide(self):
        tight = {"total_volume": 1000, "total_open_interest": 0,
                 "mean_spread_cents": 1.0}
        wide = {"total_volume": 1000, "total_open_interest": 0,
                "mean_spread_cents": 9.0}
        assert spread_adjusted_liquidity(tight) > spread_adjusted_liquidity(wide)

    def test_zero_depth_scores_zero(self):
        assert spread_adjusted_liquidity(
            {"total_volume": 0, "total_open_interest": 0, "mean_spread_cents": 1}
        ) == 0.0

    def test_unquoted_uses_wide_proxy(self):
        liq = {"total_volume": 1000, "total_open_interest": 0,
               "mean_spread_cents": None}
        # proxy spread of 10c => 1000 / 11
        assert spread_adjusted_liquidity(liq) == pytest.approx(1000 / 11.0)


class TestDiscovery:
    def test_resolves_first_live_pattern(self):
        client = build_client()
        cand = CandidateCity("den", "Denver", ["KXHIGHNOPE", "KXHIGHDEN"])
        res = discover_series_ticker(client, cand)
        assert res["ticker"] == "KXHIGHDEN"
        assert "Denver" in res["title"]

    def test_returns_none_when_unknown(self):
        client = build_client()
        cand = CandidateCity("zzz", "Nowhere", ["KXHIGHZZZ"])
        assert discover_series_ticker(client, cand) is None


class TestVerifyCity:
    def test_verified_tradeable(self):
        client = build_client()
        res = verify_city(client, CandidateCity("den", "Denver", ["KXHIGHDEN"], mandated=True))
        assert res["verified"] and res["tradeable"]
        assert res["status"] == "VERIFIED"
        assert res["series_ticker"] == "KXHIGHDEN"
        assert res["settlement_station"] == "Denver, CO"
        assert res["bucket_structure"]["n_buckets"] == 6
        assert res["spread_adjusted_liquidity"] > 0

    def test_series_without_markets_is_blocked(self):
        client = build_client()
        res = verify_city(client, CandidateCity("hou", "Houston", ["KXHIGHHOU"]))
        assert res["verified"] is True
        assert res["tradeable"] is False
        assert res["status"] == "BLOCKED"

    def test_missing_series_blocked(self):
        client = build_client()
        res = verify_city(client, CandidateCity("zzz", "Nowhere", ["KXHIGHZZZ"]))
        assert res["verified"] is False
        assert res["status"] == "BLOCKED"


class TestSelection:
    def test_mandated_always_in_discretionary_ranked(self):
        results = [
            {"code": "den", "verified": True, "tradeable": True, "mandated": True,
             "spread_adjusted_liquidity": 1.0},
            {"code": "mia", "verified": True, "tradeable": True, "mandated": False,
             "spread_adjusted_liquidity": 100.0},
            {"code": "lax", "verified": True, "tradeable": True, "mandated": False,
             "spread_adjusted_liquidity": 50.0},
            {"code": "hou", "verified": True, "tradeable": False, "mandated": False,
             "spread_adjusted_liquidity": 0.0},
        ]
        chosen = select_recommended(results, top_n=1)
        assert chosen == ["den", "mia"]      # mandated + top-1 by liquidity
        assert "hou" not in chosen           # untradeable excluded

    def test_blocked_never_recommended(self):
        results = [
            {"code": "den", "verified": False, "tradeable": False, "mandated": True,
             "spread_adjusted_liquidity": 0.0},
        ]
        assert select_recommended(results) == []


class TestRunVerification:
    def test_end_to_end_with_fake_client(self):
        client = build_client()
        candidates = [
            CandidateCity("den", "Denver", ["KXHIGHDEN"], mandated=True),
            CandidateCity("mia", "Miami", ["KXHIGHMIA"]),
            CandidateCity("hou", "Houston", ["KXHIGHHOU"]),
        ]
        payload = run_verification(client, candidates=candidates, top_n=3)
        assert payload["recommended"] == ["den", "mia"]
        assert payload["blocked"] == ["hou"]
        assert {r["code"] for r in payload["results"]} == {"den", "mia", "hou"}


def test_candidate_registry_has_mandated_cities():
    assert CANDIDATE_CITIES["den"].mandated
    assert CANDIDATE_CITIES["dc"].mandated
    # Houston is a discretionary candidate, not mandated.
    assert not CANDIDATE_CITIES["hou"].mandated
