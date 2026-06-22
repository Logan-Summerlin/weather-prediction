#!/usr/bin/env python3
"""
Emit the 7am-ET inference cutoff manifest as browsable JSON + Markdown.

The authoritative manifest lives in code (``src/data_sla.py`` cutoff
registry); this script serialises it to ``results/cutoff_manifest.json`` and
``docs/cutoff_manifest.md`` so the per-feature availability contract is
reviewable without reading Python.  Regenerate after changing any
``CutoffFeatureSpec``.

Usage:
    python scripts/build_cutoff_manifest.py
    python scripts/build_cutoff_manifest.py --market-date 2026-07-01
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_sla import (
    CUTOFF_HOUR_ET,
    CUTOFF_TIMEZONE,
    CUTOFF_MANIFEST_VERSION,
    build_cutoff_manifest_table,
    cutoff_instant_utc,
    latest_usable_timestamp,
    list_cutoff_features,
)

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def build_payload(market_date: str) -> dict:
    rows = build_cutoff_manifest_table()
    cutoff = cutoff_instant_utc(market_date)
    for row in rows:
        row["latest_usable_utc_example"] = latest_usable_timestamp(
            row["feature"], market_date
        ).isoformat()
    return {
        "manifest_version": CUTOFF_MANIFEST_VERSION,
        "cutoff_hour_et": CUTOFF_HOUR_ET,
        "cutoff_timezone": CUTOFF_TIMEZONE,
        "example_market_date": market_date,
        "example_cutoff_utc": cutoff.isoformat(),
        "features": rows,
    }


def write_markdown(payload: dict, path: str) -> None:
    lines = [
        "# 7am-ET Inference Cutoff Manifest",
        "",
        f"> Version {payload['manifest_version']} — generated from "
        "`src/data_sla.py` (do not edit by hand; run "
        "`python scripts/build_cutoff_manifest.py`).",
        "",
        f"Hard cutoff: **{payload['cutoff_hour_et']}:00 "
        f"{payload['cutoff_timezone']}** for every city. Example market day "
        f"`{payload['example_market_date']}` resolves the cutoff to "
        f"`{payload['example_cutoff_utc']}` (UTC). A feature whose freshest "
        "record is after its *latest usable* time would leak post-cutoff "
        "information; a critical feature that is missing or stale at the "
        "cutoff is a kill-switch event.",
        "",
        "| Feature | Source | Criticality | Publication / latency | "
        "Latest usable (example) | Fallback |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["features"]:
        lines.append(
            "| `{feature}` | {source} | {crit} | {pub} (~{lat}h) | "
            "`{usable}` | {fb} |".format(
                feature=row["feature"],
                source=row["source"],
                crit=row["criticality"],
                pub=row["publication_schedule"].replace("|", "/"),
                lat=row["latency_hours"],
                usable=row["latest_usable_utc_example"],
                fb=row["fallback_behavior"].replace("|", "/"),
            )
        )
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the 7am-ET cutoff manifest.")
    parser.add_argument(
        "--market-date",
        default="2026-07-01",
        help="Example market day used to resolve concrete cutoff timestamps.",
    )
    args = parser.parse_args()

    payload = build_payload(args.market_date)

    json_path = os.path.join(PROJECT_ROOT, "results", "cutoff_manifest.json")
    md_path = os.path.join(PROJECT_ROOT, "docs", "cutoff_manifest.md")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    write_markdown(payload, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Manifest covers {len(list_cutoff_features())} inference features.")


if __name__ == "__main__":
    main()
