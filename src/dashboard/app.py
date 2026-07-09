"""
Streamlit UI for the local real-time positive-EV dashboard.

Launch through the wrapper (recommended):
    python scripts/run_ev_dashboard.py --mode monitor

or directly:
    streamlit run src/dashboard/app.py

Mode/config are passed via environment variables (EV_DASHBOARD_MODE,
EV_DASHBOARD_CONFIG) so the Streamlit process needs no CLI plumbing.
All state comes from :mod:`src.dashboard.opportunity_service`; this module
is presentation only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Allow `streamlit run src/dashboard/app.py` from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from src.dashboard.opportunity_service import (  # noqa: E402
    ELIGIBLE_MONITOR,
    ELIGIBLE_TRADEABLE,
    OpportunityService,
    load_config,
    run_paper_evaluation,
)

_BADGE_COLORS = {
    "TRADEABLE-PAPER": "green",
    "MONITOR-ONLY": "orange",
    "BLOCKED": "red",
    "NO EDGE": "gray",
    "STALE MARKET": "violet",
}


@st.cache_resource
def get_service() -> OpportunityService:
    config = load_config()
    return OpportunityService(config)


def _fmt(x, pct=False, cents=False):
    if x is None:
        return "—"
    if pct:
        return f"{100 * x:.1f}%"
    if cents:
        return f"{100 * x:.1f}c"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def render_header(snapshot, config):
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    st.title("Real-Time Positive-EV Dashboard — NYC / CHI / PHL")
    cols = st.columns(5)
    cols[0].metric("UTC", now_utc.strftime("%H:%M:%S"))
    cols[1].metric("ET", now_et.strftime("%H:%M:%S"))
    cols[2].metric("Mode", config.mode.upper())
    cols[3].metric("Refresh", f"{config.refresh_interval_seconds}s")
    cols[4].metric("Snapshot", snapshot.generated_at[11:19])
    st.caption(
        ":no_entry: **Live order placement is disabled by design.** "
        f"Fee model: `{snapshot.summary.get('fee_model')}` · "
        f"slippage buffer {config.slippage_cents:.1f}c · "
        "EV uses executable ask prices, never mids."
    )


def render_summary(snapshot):
    s = snapshot.summary
    cols = st.columns(5)
    cols[0].metric("Healthy cities", f"{s['n_healthy_cities']}/{s['n_cities']}")
    cols[1].metric("Stale / blocked", f"{s['n_stale_cities']} / {s['n_blocked_cities']}")
    cols[2].metric("+EV contracts", s["n_positive_ev"])
    cols[3].metric("Paper exposure", f"${s['total_suggested_paper_exposure']:.2f}")
    max_ev = s.get("max_ev_after_costs")
    cols[4].metric("Max EV after costs", _fmt(max_ev, cents=True))


def render_city_cards(snapshot):
    cols = st.columns(max(len(snapshot.cities), 1))
    for col, (code, c) in zip(cols, snapshot.cities.items()):
        icon = {"ok": ":large_green_circle:", "stale": ":large_yellow_circle:",
                "blocked": ":red_circle:"}.get(c["status"], ":white_circle:")
        with col:
            st.subheader(f"{icon} {c['city_name']} ({code.upper()})")
            st.write(f"**Promotion:** {c['promotion_status']}")
            st.write(f"**Signal:** {c['signal_status'] or 'missing'} "
                     f"({c['signal_date'] or 'n/a'}, {c['model_name'] or '—'})")
            if c["mu"] is not None:
                st.write(f"**Forecast:** μ={c['mu']:.1f}°F σ={c['sigma']:.1f}")
            st.write(f"**Markets:** {c['n_markets']} "
                     f"({c['market_source'] or 'none'}"
                     + (f", {c['market_poll_age_seconds']:.0f}s old" if
                        c["market_poll_age_seconds"] is not None else "") + ")")
            if c["kill_switch_active"]:
                st.error("KILL SWITCH ACTIVE")
            if c["reason_codes"]:
                st.caption("Reasons: " + ", ".join(c["reason_codes"]))
            if c["status"] == "blocked" and c.get("refresh_command"):
                st.code(c["refresh_command"], language="bash")


def render_opportunities(snapshot, config):
    import pandas as pd

    rows = snapshot.opportunities
    st.subheader("Opportunity table")
    if not rows:
        st.info("No parsed contracts in the current snapshot.")
        return

    fcols = st.columns(4)
    city_filter = fcols[0].multiselect(
        "Cities", sorted({r["city_code"] for r in rows}),
        default=sorted({r["city_code"] for r in rows}))
    min_ev = fcols[1].number_input("Min EV ($)", value=float(config.min_ev),
                                   step=0.01, format="%.2f")
    include_monitor = fcols[2].checkbox("Include MONITOR cities", value=True)
    only_actionable = fcols[3].checkbox("Only positive-EV rows", value=False)

    def keep(r):
        if r["city_code"] not in city_filter:
            return False
        if not include_monitor and r["eligibility"] == ELIGIBLE_MONITOR:
            return False
        if only_actionable and (r["ev_best"] is None or r["ev_best"] < min_ev):
            return False
        return True

    kept = [r for r in rows if keep(r)]
    parser_warnings = [r for r in rows if "contract_parse_failed" in r["reason_codes"]]

    df = pd.DataFrame([{
        "City": r["city_code"].upper(),
        "Ticker": r["ticker"],
        "Bucket": r["label"],
        "Model P(YES)": r["model_yes_prob"],
        "YES bid/ask": f"{_fmt(r['yes_bid'], cents=True)} / {_fmt(r['yes_ask'], cents=True)}",
        "NO ask": _fmt(r["no_ask"], cents=True),
        "Spread": _fmt(r["spread"], cents=True),
        "Fee": _fmt(r["fee_per_contract"], cents=True),
        "EV YES": r["ev_yes"],
        "EV NO": r["ev_no"],
        "EV best": r["ev_best"],
        "Dir": r["best_direction"],
        "Size": r["suggested_size"],
        "Status": r["eligibility"],
        "Reasons": ", ".join(r["reason_codes"]),
    } for r in kept if "contract_parse_failed" not in r["reason_codes"]])
    st.dataframe(df, width="stretch", hide_index=True)

    if parser_warnings:
        with st.expander(f"Parser warnings ({len(parser_warnings)} contracts hidden from ranking)"):
            for r in parser_warnings:
                st.write(f"`{r['ticker']}` — {r['label']}")

    # Detail drawer
    tickers = [r["ticker"] for r in kept]
    if tickers:
        pick = st.selectbox("Inspect contract", ["(none)"] + tickers)
        if pick != "(none)":
            row = next(r for r in kept if r["ticker"] == pick)
            st.json(row)


def render_diagnostics(snapshot):
    tabs = st.tabs(["Health", "Warnings", "Raw snapshot", "Known limitations"])
    with tabs[0]:
        try:
            from src.dashboard.dashboard_data import DashboardData
            dash = DashboardData(city_codes=list(snapshot.cities.keys()))
            st.json(dash.get_multi_city_status())
        except Exception as exc:
            st.warning(f"Health aggregation unavailable: {exc}")
    with tabs[1]:
        if snapshot.warnings:
            for w in snapshot.warnings:
                st.warning(w)
        else:
            st.success("No cross-city warnings.")
    with tabs[2]:
        st.json(snapshot.to_dict())
    with tabs[3]:
        st.markdown(
            "- **No live orders.** Paper mode only; authenticated execution is "
            "out of scope.\n"
            "- **MONITOR cities size 0.** Positive mathematical EV on CHI/PHL "
            "is *not* a verified market-beating edge.\n"
            "- **Contract parsing is text-based.** Unparseable contracts are "
            "excluded from EV ranking (see parser warnings).\n"
            "- **Signals are cutoff-safe but may be stale**; a stale signal "
            "blocks the city for today.\n"
            "- **EV assumes top-of-book fills** at ask + slippage buffer; "
            "depth beyond top of book is not modeled."
        )


def main():
    st.set_page_config(page_title="EV Dashboard", layout="wide")
    service = get_service()
    config = service.config

    with st.sidebar:
        st.header("Controls")
        st.write(f"Mode: **{config.mode}**")
        st.write(f"Cities: {', '.join(c.upper() for c in config.cities)}")
        auto = st.checkbox("Auto-refresh", value=False)
        if st.button("Refresh now", type="primary"):
            st.session_state.pop("snapshot", None)
        st.divider()
        st.caption("Snapshots persist to "
                   f"`{os.path.relpath(config.export_dir, _REPO_ROOT)}/`")

    if "snapshot" not in st.session_state:
        with st.spinner("Building opportunity snapshot..."):
            st.session_state["snapshot"] = service.refresh()
    snapshot = st.session_state["snapshot"]

    render_header(snapshot, config)
    render_summary(snapshot)
    st.divider()
    render_city_cards(snapshot)
    st.divider()
    render_opportunities(snapshot, config)

    # Export + paper actions
    ecols = st.columns(3)
    ecols[0].download_button(
        "Export snapshot (JSON)",
        data=json.dumps(snapshot.to_dict(), indent=2, default=str),
        file_name=f"opportunities_{snapshot.generated_at[:19]}.json",
        mime="application/json",
    )
    n_eligible = snapshot.summary.get("n_tradeable_paper", 0)
    if config.mode == "paper":
        if ecols[1].button(f"Paper-trade eligible rows ({n_eligible})",
                           disabled=n_eligible == 0):
            result = run_paper_evaluation(snapshot, config)
            st.success(f"Paper evaluation submitted {result['n_rows_submitted']} rows.")
            st.json(result)
    else:
        ecols[1].caption("Paper actions require `--mode paper`.")

    st.divider()
    render_diagnostics(snapshot)

    if auto:
        time.sleep(max(5, config.refresh_interval_seconds))
        st.session_state.pop("snapshot", None)
        st.rerun()


if __name__ == "__main__":
    main()
