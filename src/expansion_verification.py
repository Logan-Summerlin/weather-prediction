"""
Kalshi contract verification for portfolio expansion (Phase 4).

Before any new city is hard-coded into the pipeline, its market must be
verified against the live Kalshi public API. Series-ticker naming is
*irregular* (Atlanta is ``KXHIGHTATL``, Phoenix is ``KXHIGHTPHX``,
Washington DC is ``KXHIGHTDC``, while Denver is ``KXHIGHDEN`` and
Philadelphia is ``KXHIGHPHIL``), so the real ticker has to be discovered
by probing candidate patterns rather than assumed.

This module is the network-free, importable core. It accepts any object
exposing ``get_series(ticker)`` and ``get_markets(series_ticker, status)``
(the read-only :class:`src.kalshi_client.KalshiClient` satisfies this), so
the selection logic is fully unit-testable with a fake client.

For each candidate city it records:
  * the resolved series ticker (and the human title Kalshi reports),
  * the settlement station parsed from the contract ``rules_primary`` text,
  * the bucket structure (count + width) via the contract sub-titles,
  * a liquidity sample (volume, open interest, mean bid/ask spread) drawn
    from the most recent event-days.

Candidates are then ranked by *spread-adjusted liquidity*; the top cities
(excluding any that fail verification or have no tradeable market) are the
recommended expansion set. No authenticated or order-placing endpoints are
ever touched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
# The mandated expansion targets (Denver, Washington DC) plus the discretionary
# pool that the plan asks us to rank by verified liquidity. ``ticker_candidates``
# lists patterns to probe in order; the first that resolves to a live series
# wins. Order encodes the irregular-naming guesses observed in the wild.
@dataclass(frozen=True)
class CandidateCity:
    code: str
    label: str
    ticker_candidates: Sequence[str]
    mandated: bool = False


CANDIDATE_CITIES: Dict[str, CandidateCity] = {
    "den": CandidateCity(
        "den", "Denver",
        ["KXHIGHDEN", "KXHIGHTDEN", "KXHIGHDENVER"], mandated=True,
    ),
    "dc": CandidateCity(
        "dc", "Washington DC",
        ["KXHIGHTDC", "KXHIGHDC", "KXHIGHWASH", "KXHIGHDCA"], mandated=True,
    ),
    "mia": CandidateCity(
        "mia", "Miami",
        ["KXHIGHMIA", "KXHIGHTMIA", "KXHIGHMIAMI"],
    ),
    "lax": CandidateCity(
        "lax", "Los Angeles",
        ["KXHIGHLAX", "KXHIGHTLAX", "KXHIGHLA", "KXHIGHLOS"],
    ),
    "hou": CandidateCity(
        "hou", "Houston",
        ["KXHIGHHOU", "KXHIGHTHOU", "KXHIGHHOUSTON", "KXHIGHIAH"],
    ),
    "phx": CandidateCity(
        "phx", "Phoenix",
        ["KXHIGHTPHX", "KXHIGHPHX", "KXHIGHPHOENIX"],
    ),
}

# Number of discretionary cities to recommend (mandated cities are always
# included regardless of rank, per the plan).
DEFAULT_TOP_N = 3


# ---------------------------------------------------------------------------
# Client protocol (KalshiClient satisfies this)
# ---------------------------------------------------------------------------
class SupportsKalshiReads(Protocol):
    def get_series(self, series_ticker: str) -> dict: ...

    def get_markets(
        self, series_ticker: str, status: str = ..., limit: int = ...
    ) -> List[dict]: ...


# ---------------------------------------------------------------------------
# Numeric helpers — the live API returns money as strings ("0.03"), counts in
# *_fp fields, and frequently leaves the legacy keys null. Coerce defensively.
# ---------------------------------------------------------------------------
def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_present(market: dict, keys: Sequence[str]) -> float:
    """Return the first numerically-present field among *keys* (else 0.0)."""
    for key in keys:
        if key in market and market[key] not in (None, ""):
            val = _as_float(market[key])
            if val:
                return val
    return 0.0


# ---------------------------------------------------------------------------
# Series-ticker discovery
# ---------------------------------------------------------------------------
def discover_series_ticker(
    client: SupportsKalshiReads, candidate: CandidateCity
) -> Optional[Dict[str, str]]:
    """Probe candidate patterns; return the first that resolves to a series.

    Returns ``{"ticker": ..., "title": ...}`` or ``None`` if no candidate
    pattern resolves. ``get_series`` is expected to raise (ValueError /
    ConnectionError) on a missing ticker; both are treated as "not this one".
    """
    for ticker in candidate.ticker_candidates:
        try:
            series = client.get_series(ticker)
        except Exception as exc:  # noqa: BLE001 - any failure means "not found"
            logger.debug("Series probe %s failed: %s", ticker, exc)
            continue
        if not series:
            continue
        title = series.get("title") or series.get("name") or ""
        logger.info("Resolved %s -> %s (%s)", candidate.code, ticker, title)
        return {"ticker": ticker, "title": title}
    logger.warning(
        "No live series found for %s (tried %s)",
        candidate.code, ", ".join(candidate.ticker_candidates),
    )
    return None


# ---------------------------------------------------------------------------
# Settlement station extraction from contract rules
# ---------------------------------------------------------------------------
_RULES_STATION_PATTERNS = [
    # "highest temperature recorded in Denver, CO for ..."
    re.compile(r"temperature\s+recorded\s+in\s+(.+?)\s+for\b", re.IGNORECASE),
    # "maximum temperature recorded at Washington DC for ..."
    re.compile(r"temperature\s+recorded\s+at\s+(.+?)\s+for\b", re.IGNORECASE),
    # "... at Miami International Airport for ..."
    re.compile(r"\bat\s+(.+?)\s+for\s+\w+\s+\d", re.IGNORECASE),
]


def extract_settlement_station(rules_primary: Optional[str]) -> Optional[str]:
    """Parse the settlement location from a contract's ``rules_primary``.

    Kalshi weather rules read like "If the highest temperature recorded in
    Denver, CO for June 24, 2026 ... NWS Climatological Report (Daily) ...".
    Returns the location phrase (e.g. "Denver, CO", "Miami International
    Airport") or ``None`` if it cannot be parsed.
    """
    if not rules_primary:
        return None
    for pat in _RULES_STATION_PATTERNS:
        m = pat.search(rules_primary)
        if m:
            station = m.group(1).strip().rstrip(",.")
            # Guard against runaway matches that swallowed half the sentence.
            if 0 < len(station) <= 60:
                return station
    return None


# ---------------------------------------------------------------------------
# Bucket-structure summary
# ---------------------------------------------------------------------------
def summarize_bucket_structure(markets: Sequence[dict]) -> Dict[str, Any]:
    """Summarize the per-day contract bucket grid for one event.

    Groups markets by ``event_ticker`` and inspects the largest event so the
    summary reflects a full day's bucket ladder rather than a partial page.
    """
    if not markets:
        return {"n_buckets": 0, "bucket_labels": [], "modal_width_f": None}

    by_event: Dict[str, List[dict]] = {}
    for mk in markets:
        by_event.setdefault(mk.get("event_ticker", ""), []).append(mk)
    # Representative event = the one with the most buckets.
    event_markets = max(by_event.values(), key=len)

    labels = [
        (mk.get("yes_sub_title") or mk.get("subtitle") or "").strip()
        for mk in event_markets
    ]
    labels = [lbl for lbl in labels if lbl]

    # Infer modal bucket width from "A° to B°" interior buckets.
    widths: List[float] = []
    for lbl in labels:
        m = re.search(r"(\d+)\D+to\D+(\d+)", lbl)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            widths.append(float(hi - lo + 1))
    modal_width = None
    if widths:
        modal_width = max(set(widths), key=widths.count)

    return {
        "n_buckets": len(event_markets),
        "bucket_labels": labels,
        "modal_width_f": modal_width,
        "n_events_sampled": len(by_event),
    }


# ---------------------------------------------------------------------------
# Liquidity sampling
# ---------------------------------------------------------------------------
def sample_liquidity(markets: Sequence[dict]) -> Dict[str, Any]:
    """Aggregate volume, open interest, and bid/ask spread over *markets*.

    Tolerant of the live schema (``volume_fp``, ``open_interest_fp``,
    ``*_dollars`` price strings) and the legacy schema (``volume``,
    ``open_interest``, integer-cent ``yes_bid``/``yes_ask``).
    """
    total_volume = 0.0
    total_oi = 0.0
    spreads_cents: List[float] = []

    for mk in markets:
        total_volume += _first_present(mk, ["volume_fp", "volume", "volume_24h_fp"])
        total_oi += _first_present(mk, ["open_interest_fp", "open_interest"])

        # Spread in cents. Dollar fields are fractional ("0.03" -> 3c).
        ask_d = _as_float(mk.get("yes_ask_dollars"))
        bid_d = _as_float(mk.get("yes_bid_dollars"))
        if ask_d > 0 and bid_d > 0:
            spreads_cents.append((ask_d - bid_d) * 100.0)
            continue
        ask_c = _as_float(mk.get("yes_ask"))
        bid_c = _as_float(mk.get("yes_bid"))
        if ask_c > 0 and bid_c > 0:
            spreads_cents.append(ask_c - bid_c)

    mean_spread = (
        sum(spreads_cents) / len(spreads_cents) if spreads_cents else None
    )
    return {
        "total_volume": total_volume,
        "total_open_interest": total_oi,
        "mean_spread_cents": mean_spread,
        "n_quoted_markets": len(spreads_cents),
        "n_markets_sampled": len(markets),
    }


def spread_adjusted_liquidity(liquidity: Dict[str, Any]) -> float:
    """Single rank-able score: depth penalized by quoted spread.

    score = (volume + open_interest) / (1 + mean_spread_cents)

    Higher volume/OI lifts the score; a wider spread (worse executable
    liquidity) divides it down. A market with no quoted two-sided spread is
    treated as maximally wide via the unquoted-penalty so it cannot
    out-rank a genuinely quoted market on raw volume alone.
    """
    depth = liquidity.get("total_volume", 0.0) + liquidity.get(
        "total_open_interest", 0.0
    )
    if depth <= 0:
        return 0.0
    spread = liquidity.get("mean_spread_cents")
    if spread is None:
        # No two-sided quotes observed: apply a conservative wide-spread proxy.
        spread = 10.0
    return depth / (1.0 + max(spread, 0.0))


# ---------------------------------------------------------------------------
# Per-city verification
# ---------------------------------------------------------------------------
def verify_city(
    client: SupportsKalshiReads,
    candidate: CandidateCity,
    liquidity_status: str = "settled",
    liquidity_event_days: int = 7,
) -> Dict[str, Any]:
    """Verify a single candidate end-to-end.

    Liquidity is sampled from the most recent ``liquidity_event_days`` event
    days (the live API exposes lifetime per-contract volume/OI on settled
    markets, so recent settled events are the cleanest 7-day proxy), with
    open markets folded in for bucket-structure and current-spread reads.
    """
    result: Dict[str, Any] = {
        "code": candidate.code,
        "label": candidate.label,
        "mandated": candidate.mandated,
        "verified": False,
        "tradeable": False,
        "status": "BLOCKED",
        "notes": [],
    }

    resolved = discover_series_ticker(client, candidate)
    if resolved is None:
        result["notes"].append("no live Kalshi series found")
        return result
    result["series_ticker"] = resolved["ticker"]
    result["series_title"] = resolved["title"]

    open_markets = _safe_markets(client, resolved["ticker"], "open")
    recent = _recent_event_markets(
        _safe_markets(client, resolved["ticker"], liquidity_status),
        liquidity_event_days,
    )
    structure_source = open_markets or recent
    sample = (recent + open_markets) or open_markets

    if not structure_source and not sample:
        result["notes"].append("series exists but exposes no markets")
        result["verified"] = True
        return result

    rules = _first_rules_text(structure_source or sample)
    result["settlement_station"] = extract_settlement_station(rules)
    result["bucket_structure"] = summarize_bucket_structure(structure_source)
    result["liquidity"] = sample_liquidity(sample)
    result["spread_adjusted_liquidity"] = spread_adjusted_liquidity(
        result["liquidity"]
    )

    result["verified"] = True
    result["tradeable"] = (
        result["liquidity"]["total_volume"] > 0
        or result["liquidity"]["total_open_interest"] > 0
    )
    if result["settlement_station"] is None:
        result["notes"].append("could not parse settlement station from rules")
    if not result["tradeable"]:
        result["notes"].append("no tradeable volume/open-interest observed")
        result["status"] = "BLOCKED"
    else:
        result["status"] = "VERIFIED"
    return result


def _safe_markets(
    client: SupportsKalshiReads, ticker: str, status: str
) -> List[dict]:
    try:
        return list(client.get_markets(ticker, status=status))
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_markets(%s, %s) failed: %s", ticker, status, exc)
        return []


def _recent_event_markets(
    markets: Sequence[dict], event_days: int
) -> List[dict]:
    """Keep markets belonging to the most recent *event_days* event tickers."""
    if not markets:
        return []
    # Event tickers embed the date (e.g. KXHIGHDEN-26JUN24); sort descending
    # so the newest event days come first, then keep that many.
    events = sorted(
        {mk.get("event_ticker", "") for mk in markets if mk.get("event_ticker")},
        reverse=True,
    )
    keep = set(events[:event_days])
    if not keep:
        return list(markets)
    return [mk for mk in markets if mk.get("event_ticker") in keep]


def _first_rules_text(markets: Sequence[dict]) -> Optional[str]:
    for mk in markets:
        rules = mk.get("rules_primary")
        if rules:
            return rules
    return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_recommended(
    results: Sequence[Dict[str, Any]], top_n: int = DEFAULT_TOP_N
) -> List[str]:
    """Return the recommended expansion set: all mandated verified-tradeable
    cities plus the ``top_n`` discretionary cities by spread-adjusted
    liquidity. Cities that failed verification or have no tradeable market
    are never recommended.
    """
    eligible = [r for r in results if r.get("verified") and r.get("tradeable")]
    mandated = [r["code"] for r in eligible if r.get("mandated")]
    discretionary = sorted(
        (r for r in eligible if not r.get("mandated")),
        key=lambda r: r.get("spread_adjusted_liquidity", 0.0),
        reverse=True,
    )
    chosen = mandated + [r["code"] for r in discretionary[:top_n]]
    # Preserve a stable, de-duplicated order.
    seen: set = set()
    ordered: List[str] = []
    for code in chosen:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def run_verification(
    client: SupportsKalshiReads,
    candidates: Optional[Sequence[CandidateCity]] = None,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """Verify every candidate and assemble the artifact payload."""
    if candidates is None:
        candidates = list(CANDIDATE_CITIES.values())

    results = [verify_city(client, c) for c in candidates]
    recommended = select_recommended(results, top_n=top_n)

    return {
        "candidates": [c.code for c in candidates],
        "top_n_discretionary": top_n,
        "results": results,
        "recommended": recommended,
        "blocked": [
            r["code"] for r in results if not (r.get("verified") and r.get("tradeable"))
        ],
    }
