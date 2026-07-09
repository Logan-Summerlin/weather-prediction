"""
Real-time positive-EV opportunity service (docs/02_realtime_ev_dashboard_plan.md).

Pure-Python service (no UI dependencies) that builds the dashboard's
opportunity table for NYC / CHI / PHL:

  1. Load the latest cutoff-safe daily signal per city
     (``results/<city>/live/signals_YYYY-MM-DD.json``).
  2. Poll open Kalshi markets (read-only, public API) or load offline
     fixtures, and normalize contracts into a single executable-quote schema.
  3. Price the model distribution against each live contract's own parsed
     (lo, hi, direction) through the verified settlement semantics.
  4. Compute conservative executable EV (ask prices + curved Kalshi fee +
     slippage buffer; never mid prices) with machine-readable reason codes.
  5. Persist append-only snapshots under ``results/dashboard/``.

Guardrails (section 8 of the plan):
  - No authenticated order placement anywhere in this module.
  - MONITOR/BLOCKED cities never get a nonzero suggested size by default.
  - Stale signals or stale market snapshots make all rows non-actionable.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from src.city_config import get_city_config, PROJECT_ROOT
from src.paper_trading import model_prob_for_contract
from src.trading import (
    TradingStrategy,
    fractional_kelly,
    kalshi_fee_per_contract,
    position_size,
)
from src.live_trading import load_city_strategy
from src.strategy_selection import load_strategy_json

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

VALID_MODES = ("offline", "monitor", "paper")  # "live" is explicitly out of scope

# Eligibility badges (section 5.2 of the plan)
ELIGIBLE_TRADEABLE = "TRADEABLE-PAPER"
ELIGIBLE_MONITOR = "MONITOR-ONLY"
ELIGIBLE_BLOCKED = "BLOCKED"
ELIGIBLE_NO_EDGE = "NO EDGE"
ELIGIBLE_STALE = "STALE MARKET"

# Reason codes (section 6, Phase D of the plan)
REASON_EV_BELOW_THRESHOLD = "ev_below_threshold"
REASON_SPREAD_TOO_WIDE = "spread_too_wide"
REASON_LIQUIDITY_MISSING = "liquidity_missing"
REASON_CITY_NOT_PROMOTED = "city_not_promoted"
REASON_CALIBRATION_MISSING = "calibration_missing"
REASON_DATA_SLA_FAILED = "data_sla_failed"
REASON_KILL_SWITCH_ACTIVE = "kill_switch_active"
REASON_SIGNAL_MISSING = "signal_missing"
REASON_SIGNAL_STALE = "signal_stale"
REASON_MARKET_STALE = "market_stale"
REASON_PARSE_FAILED = "contract_parse_failed"

FEE_MODEL_LABEL = "kalshi_curved_ceil(0.07*P*(1-P))_on_entry"


# ===========================================================================
# Configuration
# ===========================================================================

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "dashboard.yaml")
DEFAULT_FIXTURE_DIR = os.path.join(PROJECT_ROOT, "data", "fixtures", "dashboard")
DEFAULT_EXPORT_DIR = os.path.join(PROJECT_ROOT, "results", "dashboard")


@dataclass
class DashboardConfig:
    """Local dashboard settings (config/dashboard.yaml)."""

    cities: List[str] = field(default_factory=lambda: ["nyc", "chi", "phl"])
    mode: str = "monitor"
    refresh_interval_seconds: int = 60
    max_market_age_seconds: float = 300.0
    min_ev: float = 0.01
    slippage_cents: float = 1.0
    max_spread_cents: float = 10.0
    orderbook_top_n: int = 0  # 0 = never fetch order books automatically
    export_dir: str = DEFAULT_EXPORT_DIR
    fixture_dir: str = DEFAULT_FIXTURE_DIR
    snapshot_retention: int = 200
    raw_cache_retention: int = 100
    allow_stale_signals: bool = False  # offline/demo only
    show_monitor_sizing: bool = False  # dev flag; never default-on
    bankroll: float = 1000.0

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unsafe or unknown dashboard mode '{self.mode}'. "
                f"Allowed: {VALID_MODES} ('live' is out of scope by design)."
            )
        self.cities = [str(c).strip().lower() for c in self.cities]
        for city in self.cities:
            get_city_config(city)  # raises ValueError on unknown city
        if self.mode == "offline":
            # Fixtures are historical by definition.
            self.allow_stale_signals = True

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def slippage_dollars(self) -> float:
        return float(self.slippage_cents) / 100.0

    @property
    def max_spread_dollars(self) -> float:
        return float(self.max_spread_cents) / 100.0


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DashboardConfig:
    """Load DashboardConfig from YAML with environment-variable overrides.

    Environment overrides (no secrets, local paths/modes only):
      EV_DASHBOARD_MODE, EV_DASHBOARD_CITIES (comma-separated),
      EV_DASHBOARD_EXPORT_DIR, EV_DASHBOARD_FIXTURE_DIR.
    """
    data: Dict[str, Any] = {}
    path = path or os.environ.get("EV_DASHBOARD_CONFIG") or DEFAULT_CONFIG_PATH
    if path and os.path.isfile(path):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

    env_mode = os.environ.get("EV_DASHBOARD_MODE")
    if env_mode:
        data["mode"] = env_mode
    env_cities = os.environ.get("EV_DASHBOARD_CITIES")
    if env_cities:
        data["cities"] = [c for c in env_cities.split(",") if c.strip()]
    env_export = os.environ.get("EV_DASHBOARD_EXPORT_DIR")
    if env_export:
        data["export_dir"] = env_export
    env_fixture = os.environ.get("EV_DASHBOARD_FIXTURE_DIR")
    if env_fixture:
        data["fixture_dir"] = env_fixture

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    known = set(DashboardConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown dashboard config keys: %s", sorted(unknown))
    return DashboardConfig(**{k: v for k, v in data.items() if k in known})


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class OpportunityRow:
    """One live contract priced against the model (plan section 7.1)."""

    city_code: str
    city_name: str = ""
    market_date: str = ""
    ticker: str = ""
    event_ticker: str = ""
    label: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    direction: str = "unknown"
    model_mu: Optional[float] = None
    model_sigma: Optional[float] = None
    model_yes_prob: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    mid_prob: Optional[float] = None  # display only; never used for EV
    executable_price: Optional[float] = None
    spread: Optional[float] = None
    fee_per_contract: Optional[float] = None
    slippage_per_contract: float = 0.0
    fee_model: str = FEE_MODEL_LABEL
    ev_yes: Optional[float] = None
    ev_no: Optional[float] = None
    ev_best: Optional[float] = None
    best_direction: str = "NONE"
    suggested_size: int = 0
    promotion_status: str = "UNKNOWN"
    eligibility: str = ELIGIBLE_NO_EDGE
    reason_codes: List[str] = field(default_factory=list)
    market_age_seconds: Optional[float] = None
    signal_age_seconds: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CityDashboardState:
    """Per-city health/signal/market state for the dashboard header cards."""

    city_code: str
    city_name: str = ""
    status: str = "ok"  # ok | stale | blocked
    promotion_status: str = "UNKNOWN"
    signal_path: Optional[str] = None
    signal_date: Optional[str] = None
    signal_generated_at: Optional[str] = None
    signal_status: Optional[str] = None  # OK | KILL_SWITCH | missing | stale
    model_name: str = ""
    mu: Optional[float] = None
    sigma: Optional[float] = None
    market_poll_ok: bool = False
    market_poll_age_seconds: Optional[float] = None
    market_source: str = ""  # live | fixture | last_good_cache | none
    n_markets: int = 0
    n_opportunities: int = 0
    n_positive_ev: int = 0
    kill_switch_active: bool = False
    reason_codes: List[str] = field(default_factory=list)
    refresh_command: str = ""
    strategy_name: str = ""
    ev_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DashboardSnapshot:
    """Full dashboard state written on each refresh (plan section 7.2)."""

    generated_at: str
    mode: str
    config_hash: str
    cities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    opportunities: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Market normalization
# ===========================================================================

_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def market_date_from_ticker(ticker: str) -> Optional[str]:
    """Parse the event date out of a Kalshi ticker (e.g. ``-25JUL09``)."""
    m = _TICKER_DATE_RE.search(ticker or "")
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    return f"20{yy}-{month:02d}-{int(dd):02d}"


def _cents_to_dollars(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v / 100.0


def normalize_markets(raw_markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw Kalshi market dicts into a single executable-quote schema.

    Keeps every field needed to compute executable YES and NO EV: bid/ask on
    both sides (dollars), spread, threshold semantics, event date, and
    volume/open-interest when present.  Mid-price implied probability is
    retained for display only.
    """
    from src.kalshi_client import parse_market_threshold

    rows: List[Dict[str, Any]] = []
    for market in raw_markets or []:
        parsed = parse_market_threshold(market)
        lo = parsed.get("threshold_low")
        hi = parsed.get("threshold_high")
        if parsed["direction"] == "above" and parsed.get("threshold") is not None:
            lo, hi = parsed["threshold"], None
        elif parsed["direction"] == "below" and parsed.get("threshold") is not None:
            lo, hi = None, parsed["threshold"]

        yes_bid = _cents_to_dollars(market.get("yes_bid"))
        yes_ask = _cents_to_dollars(market.get("yes_ask"))
        no_bid = _cents_to_dollars(market.get("no_bid"))
        no_ask = _cents_to_dollars(market.get("no_ask"))
        last = _cents_to_dollars(market.get("last_price"))

        # Consistency fallbacks: on Kalshi, no_ask == 1 - yes_bid at top of book.
        if no_ask is None and yes_bid is not None:
            no_ask = round(1.0 - yes_bid, 4)
        if yes_ask is None and no_bid is not None:
            yes_ask = round(1.0 - no_bid, 4)

        spread = None
        if yes_bid is not None and yes_ask is not None:
            spread = round(yes_ask - yes_bid, 4)

        mid = None
        if yes_bid is not None and yes_ask is not None:
            mid = (yes_bid + yes_ask) / 2.0
        elif last is not None:
            mid = last

        ticker = market.get("ticker", "")
        rows.append({
            "ticker": ticker,
            "event_ticker": market.get("event_ticker", ""),
            "title": market.get("title", "") or market.get("yes_sub_title", ""),
            "market_date": market_date_from_ticker(ticker) or "",
            "lo": lo,
            "hi": hi,
            "direction": parsed["direction"],
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "last_price": last,
            "mid_prob": mid,
            "spread": spread,
            "volume": market.get("volume"),
            "open_interest": market.get("open_interest"),
        })
    return rows


# ===========================================================================
# Market polling with last-good cache
# ===========================================================================

class MarketPoller:
    """Read-only Kalshi poller with raw-response caching and last-good reuse.

    Never uses authenticated endpoints.  On a failed poll it returns the
    last-good cached snapshot (marked stale) so a failure can never produce a
    false fresh opportunity.
    """

    def __init__(self, config: DashboardConfig, client=None):
        self.config = config
        self._client = client
        self.raw_dir = os.path.join(config.export_dir, "raw_kalshi")

    @property
    def client(self):
        if self._client is None:
            from src.kalshi_client import KalshiClient

            self._client = KalshiClient()
        return self._client

    def _cache_path(self, city_code: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return os.path.join(self.raw_dir, f"{ts}_{city_code}.json")

    def _latest_cached(self, city_code: str) -> Optional[Dict[str, Any]]:
        pattern = os.path.join(self.raw_dir, f"*_{city_code}.json")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        try:
            with open(files[-1], "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, city_code: str, payload: Dict[str, Any]) -> None:
        os.makedirs(self.raw_dir, exist_ok=True)
        with open(self._cache_path(city_code), "w") as f:
            json.dump(payload, f, default=str)
        # Retention: keep the newest N raw files per city.
        pattern = os.path.join(self.raw_dir, f"*_{city_code}.json")
        files = sorted(glob.glob(pattern))
        for old in files[: max(0, len(files) - self.config.raw_cache_retention)]:
            try:
                os.remove(old)
            except OSError:
                pass

    def poll(self, city_code: str) -> Dict[str, Any]:
        """Poll open markets for a city.

        Returns ``{"markets": [...], "polled_at": iso, "fresh": bool,
        "source": "live"|"last_good_cache"|"none", "error": str|None}``.
        """
        cfg = get_city_config(city_code)
        try:
            markets = self.client.get_markets(
                series_ticker=cfg.kalshi_ticker, status="open"
            )
            payload = {
                "markets": markets,
                "polled_at": datetime.now(timezone.utc).isoformat(),
                "fresh": True,
                "source": "live",
                "error": None,
            }
            self._write_cache(city_code, payload)
            return payload
        except Exception as exc:  # network/HTTP/parse failures
            logger.warning("Kalshi poll failed for %s: %s", city_code, exc)
            cached = self._latest_cached(city_code)
            if cached is not None:
                cached = dict(cached)
                cached["fresh"] = False
                cached["source"] = "last_good_cache"
                cached["error"] = str(exc)
                return cached
            return {
                "markets": [],
                "polled_at": None,
                "fresh": False,
                "source": "none",
                "error": str(exc),
            }


# ===========================================================================
# Canonical dashboard EV (executable prices + curved fee + slippage)
# ===========================================================================

def compute_dashboard_ev(
    model_prob: float,
    yes_ask: Optional[float],
    no_ask: Optional[float],
    slippage: float = 0.01,
) -> Dict[str, Any]:
    """Canonical cost-aware EV for one contract (plan Phase D).

    EV is computed at *executable* prices only — the ask on the side being
    bought, plus a slippage buffer, plus the curved Kalshi per-contract fee
    ``ceil(0.07*P*(1-P))`` charged on entry.  Mid prices are never used.

    Returns dict with ``ev_yes``, ``ev_no``, ``ev_best``, ``best_direction``,
    ``executable_price``, ``fee_per_contract``.
    """
    ev_yes = float("nan")
    ev_no = float("nan")
    fee_yes = fee_no = None
    exec_yes = exec_no = None

    if model_prob is not None and not math.isnan(model_prob):
        if yes_ask is not None and not math.isnan(yes_ask):
            exec_yes = min(yes_ask + slippage, 1.0)
            fee_yes = kalshi_fee_per_contract(exec_yes)
            ev_yes = model_prob * 1.0 - exec_yes - fee_yes
        if no_ask is not None and not math.isnan(no_ask):
            exec_no = min(no_ask + slippage, 1.0)
            fee_no = kalshi_fee_per_contract(exec_no)
            ev_no = (1.0 - model_prob) * 1.0 - exec_no - fee_no

    candidates = []
    if not math.isnan(ev_yes):
        candidates.append(("YES", ev_yes, exec_yes, fee_yes))
    if not math.isnan(ev_no):
        candidates.append(("NO", ev_no, exec_no, fee_no))

    if not candidates:
        return {
            "ev_yes": None, "ev_no": None, "ev_best": None,
            "best_direction": "NONE", "executable_price": None,
            "fee_per_contract": None,
        }

    best = max(candidates, key=lambda c: c[1])
    return {
        "ev_yes": None if math.isnan(ev_yes) else round(ev_yes, 6),
        "ev_no": None if math.isnan(ev_no) else round(ev_no, 6),
        "ev_best": round(best[1], 6),
        "best_direction": best[0],
        "executable_price": round(best[2], 6),
        "fee_per_contract": round(best[3], 6),
    }


# ===========================================================================
# Opportunity service
# ===========================================================================

class OpportunityService:
    """Builds the real-time opportunity snapshot for all configured cities."""

    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        poller: Optional[MarketPoller] = None,
        now_fn=None,
    ):
        self.config = config or DashboardConfig()
        self.poller = poller or (
            MarketPoller(self.config) if self.config.mode != "offline" else None
        )
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    # ---------------------------------------------------------------- signals

    def _live_dir(self, city_code: str) -> str:
        return os.path.join(get_city_config(city_code).results_dir, "live")

    def find_latest_signal_path(self, city_code: str) -> Optional[str]:
        """Latest ``signals_YYYY-MM-DD.json``: city live dir, then fixtures."""
        candidates = sorted(
            glob.glob(os.path.join(self._live_dir(city_code), "signals_*.json"))
        )
        if candidates:
            return candidates[-1]
        fixture = os.path.join(self.config.fixture_dir, f"signals_{city_code}.json")
        if self.config.mode == "offline" and os.path.isfile(fixture):
            return fixture
        return None

    def load_latest_signal(self, city_code: str) -> Optional[Dict[str, Any]]:
        path = self.find_latest_signal_path(city_code)
        if path is None:
            return None
        try:
            with open(path, "r") as f:
                signal = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Unreadable signal %s: %s", path, exc)
            return None
        signal["_path"] = path
        return signal

    def _today_et(self) -> str:
        return self._now().astimezone(ET).strftime("%Y-%m-%d")

    # ---------------------------------------------------------------- markets

    def _load_fixture_markets(self, city_code: str) -> Dict[str, Any]:
        path = os.path.join(self.config.fixture_dir, f"markets_{city_code}.json")
        if not os.path.isfile(path):
            return {"markets": [], "polled_at": None, "fresh": False,
                    "source": "none", "error": f"fixture missing: {path}"}
        with open(path, "r") as f:
            data = json.load(f)
        markets = data.get("markets", data) if isinstance(data, dict) else data
        return {
            "markets": markets,
            "polled_at": self._now().isoformat(),
            "fresh": True,
            "source": "fixture",
            "error": None,
        }

    def get_city_markets(self, city_code: str) -> Dict[str, Any]:
        if self.config.mode == "offline":
            return self._load_fixture_markets(city_code)
        return self.poller.poll(city_code)

    # ------------------------------------------------------------- city state

    def _build_city_state(
        self, city_code: str, signal: Optional[Dict[str, Any]],
        poll: Dict[str, Any],
    ) -> CityDashboardState:
        cfg = get_city_config(city_code)
        state = CityDashboardState(
            city_code=city_code,
            city_name=cfg.city_name,
            refresh_command=f"python scripts/run_daily_inference.py --city {city_code}",
        )

        strategy_data = load_strategy_json(city_code)
        if strategy_data:
            state.promotion_status = strategy_data.get(
                "promotion_status",
                strategy_data.get("promotion", {}).get("status", "UNKNOWN"),
            )
            strat_block = strategy_data.get("strategy", {})
            state.strategy_name = strat_block.get("name", "")
            state.ev_threshold = strat_block.get("ev_threshold")

        # Signal health
        if signal is None:
            state.status = "blocked"
            state.signal_status = "missing"
            state.reason_codes.append(REASON_SIGNAL_MISSING)
        else:
            state.signal_path = signal.get("_path")
            state.signal_date = signal.get("date")
            state.signal_generated_at = signal.get("generated_at")
            state.signal_status = signal.get("status")
            state.model_name = signal.get("model_name", "")
            state.mu = signal.get("mu")
            state.sigma = signal.get("sigma")
            if signal.get("promotion_status") not in (None, "", "UNKNOWN"):
                state.promotion_status = signal["promotion_status"]

            if signal.get("status") == "KILL_SWITCH":
                state.status = "blocked"
                state.kill_switch_active = True
                state.reason_codes.append(REASON_KILL_SWITCH_ACTIVE)
                state.reason_codes.append(REASON_DATA_SLA_FAILED)
            elif signal.get("mu") is None or signal.get("sigma") is None:
                state.status = "blocked"
                state.reason_codes.append(REASON_SIGNAL_MISSING)
            elif (signal.get("date") != self._today_et()
                  and not self.config.allow_stale_signals):
                state.status = "blocked"
                state.signal_status = "stale"
                state.reason_codes.append(REASON_SIGNAL_STALE)

        # Market health
        state.market_poll_ok = bool(poll.get("fresh"))
        state.market_source = poll.get("source", "")
        state.n_markets = len(poll.get("markets", []))
        if poll.get("polled_at"):
            try:
                polled = datetime.fromisoformat(str(poll["polled_at"]))
                if polled.tzinfo is None:
                    polled = polled.replace(tzinfo=timezone.utc)
                state.market_poll_age_seconds = max(
                    0.0, (self._now() - polled).total_seconds()
                )
            except ValueError:
                state.market_poll_age_seconds = None
        if not state.market_poll_ok or (
            state.market_poll_age_seconds is not None
            and state.market_poll_age_seconds > self.config.max_market_age_seconds
        ):
            if REASON_MARKET_STALE not in state.reason_codes:
                state.reason_codes.append(REASON_MARKET_STALE)
            if state.status == "ok":
                state.status = "stale"
        return state

    # ------------------------------------------------------------ opportunity

    def _build_row(
        self,
        state: CityDashboardState,
        signal: Optional[Dict[str, Any]],
        market: Dict[str, Any],
        strategy: TradingStrategy,
    ) -> OpportunityRow:
        cfg = get_city_config(state.city_code)
        row = OpportunityRow(
            city_code=state.city_code,
            city_name=cfg.city_name,
            market_date=market.get("market_date", ""),
            ticker=market.get("ticker", ""),
            event_ticker=market.get("event_ticker", ""),
            label=market.get("title") or market.get("ticker", ""),
            lo=market.get("lo"),
            hi=market.get("hi"),
            direction=market.get("direction", "unknown"),
            yes_bid=market.get("yes_bid"),
            yes_ask=market.get("yes_ask"),
            no_bid=market.get("no_bid"),
            no_ask=market.get("no_ask"),
            mid_prob=market.get("mid_prob"),
            spread=market.get("spread"),
            volume=market.get("volume"),
            open_interest=market.get("open_interest"),
            promotion_status=state.promotion_status,
            slippage_per_contract=self.config.slippage_dollars,
            market_age_seconds=state.market_poll_age_seconds,
        )
        reasons: List[str] = []

        # Contract parser confidence guard (plan section 8.6)
        if row.direction == "unknown" or (row.lo is None and row.hi is None):
            row.eligibility = ELIGIBLE_BLOCKED
            row.reason_codes = [REASON_PARSE_FAILED]
            return row

        # Model probability against the contract's own terms
        if signal is not None and signal.get("mu") is not None:
            row.model_mu = float(signal["mu"])
            row.model_sigma = float(signal.get("sigma") or 1.0)
            contract = {
                "lo": float("nan") if row.lo is None else float(row.lo),
                "hi": float("nan") if row.hi is None else float(row.hi),
                "direction": row.direction,
            }
            row.model_yes_prob = model_prob_for_contract(
                row.model_mu, row.model_sigma, contract
            )
            if signal.get("generated_at"):
                try:
                    gen = datetime.fromisoformat(str(signal["generated_at"]))
                    if gen.tzinfo is None:
                        gen = gen.replace(tzinfo=timezone.utc)
                    row.signal_age_seconds = max(
                        0.0, (self._now() - gen).total_seconds()
                    )
                except ValueError:
                    pass

        # EV at executable prices
        if row.model_yes_prob is not None:
            ev = compute_dashboard_ev(
                row.model_yes_prob, row.yes_ask, row.no_ask,
                slippage=self.config.slippage_dollars,
            )
            row.ev_yes = ev["ev_yes"]
            row.ev_no = ev["ev_no"]
            row.ev_best = ev["ev_best"]
            row.best_direction = ev["best_direction"]
            row.executable_price = ev["executable_price"]
            row.fee_per_contract = ev["fee_per_contract"]

        # ---- eligibility ladder ----
        city_blocked = state.status == "blocked"
        market_stale = REASON_MARKET_STALE in state.reason_codes

        if city_blocked:
            row.eligibility = ELIGIBLE_BLOCKED
            row.reason_codes = list(state.reason_codes)
            return row
        if market_stale:
            row.eligibility = ELIGIBLE_STALE
            reasons.append(REASON_MARKET_STALE)

        if row.best_direction == "NONE" or row.ev_best is None:
            reasons.append(REASON_LIQUIDITY_MISSING)
            if row.eligibility != ELIGIBLE_STALE:
                row.eligibility = ELIGIBLE_NO_EDGE
            row.reason_codes = reasons
            return row

        min_ev = max(self.config.min_ev, strategy.ev_threshold)
        has_edge = row.ev_best >= min_ev
        if not has_edge:
            reasons.append(REASON_EV_BELOW_THRESHOLD)
        if row.spread is not None and row.spread > self.config.max_spread_dollars:
            reasons.append(REASON_SPREAD_TOO_WIDE)
            has_edge = False

        if not has_edge:
            if row.eligibility != ELIGIBLE_STALE:
                row.eligibility = ELIGIBLE_NO_EDGE
            row.reason_codes = reasons
            return row

        if market_stale:
            # Positive EV on a stale snapshot is never actionable.
            row.reason_codes = reasons
            return row

        promoted = str(state.promotion_status).upper() in ("PROMOTED", "PASS", "READY")
        if not promoted:
            row.eligibility = ELIGIBLE_MONITOR
            reasons.append(REASON_CITY_NOT_PROMOTED)
            row.reason_codes = reasons
            if self.config.show_monitor_sizing:
                row.suggested_size = self._size_position(row, strategy)
            return row

        row.eligibility = ELIGIBLE_TRADEABLE
        row.reason_codes = reasons
        row.suggested_size = self._size_position(row, strategy)
        return row

    def _size_position(self, row: OpportunityRow, strategy: TradingStrategy) -> int:
        """Capped fractional-Kelly size at the executable price."""
        if row.executable_price is None or row.model_yes_prob is None:
            return 0
        if row.best_direction == "YES":
            p, price = row.model_yes_prob, row.executable_price
        else:
            p, price = 1.0 - row.model_yes_prob, row.executable_price
        kelly = fractional_kelly(p, price, fee_rate=0.0,
                                 fraction=strategy.kelly_fraction_param)
        kelly = min(kelly, strategy.max_position_frac)
        return position_size(
            kelly, self.config.bankroll, contract_price=price,
            max_size=strategy.max_contracts,
        )

    # ---------------------------------------------------------------- snapshot

    def build_snapshot(self) -> DashboardSnapshot:
        generated_at = self._now().isoformat()
        cities: Dict[str, Dict[str, Any]] = {}
        all_rows: List[OpportunityRow] = []
        warnings: List[str] = []

        for city_code in self.config.cities:
            signal = self.load_latest_signal(city_code)
            poll = self.get_city_markets(city_code)
            state = self._build_city_state(city_code, signal, poll)
            if poll.get("error"):
                warnings.append(f"{city_code}: market poll: {poll['error']}")

            strategy = load_city_strategy(city_code) or TradingStrategy(
                name=f"{city_code}_default", ev_threshold=0.03,
                kelly_fraction=0.20, max_position_frac=0.08,
                bankroll=self.config.bankroll,
            )

            markets = normalize_markets(poll.get("markets", []))
            rows = [
                self._build_row(state, signal, m, strategy) for m in markets
            ]
            state.n_opportunities = len(rows)
            state.n_positive_ev = sum(
                1 for r in rows
                if r.ev_best is not None and r.ev_best >= self.config.min_ev
            )
            cities[city_code] = state.to_dict()
            all_rows.extend(rows)

        all_rows.sort(
            key=lambda r: (r.ev_best if r.ev_best is not None else float("-inf")),
            reverse=True,
        )
        evs = [r.ev_best for r in all_rows if r.ev_best is not None]
        summary = {
            "n_cities": len(self.config.cities),
            "n_healthy_cities": sum(
                1 for c in cities.values() if c["status"] == "ok"
            ),
            "n_blocked_cities": sum(
                1 for c in cities.values() if c["status"] == "blocked"
            ),
            "n_stale_cities": sum(
                1 for c in cities.values() if c["status"] == "stale"
            ),
            "n_contracts": len(all_rows),
            "n_positive_ev": sum(1 for e in evs if e >= self.config.min_ev),
            "n_tradeable_paper": sum(
                1 for r in all_rows if r.eligibility == ELIGIBLE_TRADEABLE
            ),
            "n_monitor_only": sum(
                1 for r in all_rows if r.eligibility == ELIGIBLE_MONITOR
            ),
            "max_ev_after_costs": max(evs) if evs else None,
            "total_suggested_paper_exposure": round(sum(
                r.suggested_size * (r.executable_price or 0.0) for r in all_rows
            ), 2),
            "fee_model": FEE_MODEL_LABEL,
            "slippage_cents": self.config.slippage_cents,
        }

        return DashboardSnapshot(
            generated_at=generated_at,
            mode=self.config.mode,
            config_hash=self.config.config_hash(),
            cities=cities,
            opportunities=[r.to_dict() for r in all_rows],
            summary=summary,
            warnings=warnings,
        )

    # ------------------------------------------------------------- persistence

    def save_snapshot(self, snapshot: DashboardSnapshot) -> str:
        """Write timestamped snapshot + ``latest_opportunities.json`` copy."""
        os.makedirs(self.config.export_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.export_dir, f"opportunities_{ts}.json")
        payload = snapshot.to_dict()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        latest = os.path.join(self.config.export_dir, "latest_opportunities.json")
        with open(latest, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        # Retention cleanup
        files = sorted(glob.glob(
            os.path.join(self.config.export_dir, "opportunities_*.json")
        ))
        for old in files[: max(0, len(files) - self.config.snapshot_retention)]:
            try:
                os.remove(old)
            except OSError:
                pass
        logger.info("Wrote opportunity snapshot to %s", path)
        return path

    def refresh(self) -> DashboardSnapshot:
        """Build and persist one snapshot."""
        snapshot = self.build_snapshot()
        self.save_snapshot(snapshot)
        return snapshot


# ===========================================================================
# Paper-trade workflow (paper mode only; never live)
# ===========================================================================

def run_paper_evaluation(
    snapshot: DashboardSnapshot,
    config: DashboardConfig,
) -> Dict[str, Any]:
    """Route TRADEABLE-PAPER rows through the existing paper harness.

    Uses :class:`src.live_trading.LiveTradingHarness` in ``paper`` mode with
    the existing audit conventions under ``results/<city>/trading/``.
    Reproducible from the saved snapshot (rows carry all pricing inputs).
    """
    if config.mode != "paper":
        raise ValueError(
            f"Paper evaluation requires mode='paper' (got '{config.mode}')."
        )
    from src.live_trading import LiveTradingHarness
    from src.paper_trading import build_market_prediction, run_paper_cycle

    results: Dict[str, Any] = {"snapshot_generated_at": snapshot.generated_at,
                               "cities": {}}
    rows = [r for r in snapshot.opportunities
            if r["eligibility"] == ELIGIBLE_TRADEABLE and r["suggested_size"] > 0]
    by_city: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_city.setdefault(r["city_code"], []).append(r)

    for city_code, city_rows in by_city.items():
        contracts = [{
            "ticker": r["ticker"],
            "label": r["label"] or r["ticker"],
            "lo": float("nan") if r["lo"] is None else float(r["lo"]),
            "hi": float("nan") if r["hi"] is None else float(r["hi"]),
            "direction": r["direction"],
            "price": float(r["executable_price"]),
        } for r in city_rows]
        first = city_rows[0]
        prediction = build_market_prediction(
            city_code, first["market_date"] or snapshot.generated_at[:10],
            float(first["model_mu"]), float(first["model_sigma"]),
            contracts, model_name="dashboard",
        )
        harness = LiveTradingHarness(city_code, mode="paper")
        results["cities"][city_code] = run_paper_cycle(
            harness, prediction, contracts, actual_tmax=None, save_audit=True,
        )
    results["n_rows_submitted"] = len(rows)
    return results
