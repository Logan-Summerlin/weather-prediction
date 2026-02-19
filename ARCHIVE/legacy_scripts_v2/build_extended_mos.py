#!/usr/bin/env python3
"""
Build extended combined MOS CSV with multi-station average airport proxy backfill for 2000-2003.

Approach:
1. Compute monthly bias offsets for each airport station (KJFK, KLGA, KEWR) relative to KNYC
   using the overlap period 2004-2019 (training portion only).
2. Apply offsets to harmonize airport forecasts to KNYC-equivalent values.
3. Average the three harmonized airport values to create a robust proxy.
4. For 2000-06-01 through 2003-12-31: use the multi-station average proxy.
5. For 2004-01-01 onward: use existing KNYC data unchanged.
6. Validate against actual Central Park TMAX observations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
MOS_DIR = BASE / "data" / "mos"
DATA_DIR = BASE / "data"

KNYC_PATH = MOS_DIR / "combined_mos_knyc.csv"
CP_ACTUAL_PATH = DATA_DIR / "central_park_tmax_full_history.csv"

AIRPORT_GFS = {
    "KJFK": MOS_DIR / "kjfk_gfs_mos_tmax.csv",
    "KLGA": MOS_DIR / "klga_gfs_mos_tmax.csv",
    "KEWR": MOS_DIR / "kewr_gfs_mos_tmax.csv",
}
AIRPORT_NAM = {
    "KJFK": MOS_DIR / "kjfk_nam_mos_tmax.csv",
    "KLGA": MOS_DIR / "klga_nam_mos_tmax.csv",
    "KEWR": MOS_DIR / "kewr_nam_mos_tmax.csv",
}

OUTPUT_PATH = MOS_DIR / "combined_mos_extended.csv"
ERA_PATH = MOS_DIR / "mos_era_indicator.csv"

# Overlap period for computing monthly offsets (train only, no test/calibration contamination)
OFFSET_START = "2004-01-01"
OFFSET_END = "2019-12-31"

# Backfill range
BACKFILL_START = "2000-06-01"
BACKFILL_END = "2003-12-31"


def load_airport_data(paths: dict, model_type: str) -> dict:
    """Load airport MOS CSVs into a dict of DataFrames keyed by station."""
    result = {}
    for station, path in paths.items():
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[["date", "tmax_f"]].rename(columns={"tmax_f": f"{station}_{model_type}"})
        # Drop duplicates (keep first occurrence per date)
        df = df.drop_duplicates(subset="date", keep="first")
        result[station] = df
    return result


def compute_monthly_offsets(knyc_col: pd.Series, airport_col: pd.Series,
                            dates: pd.Series) -> dict:
    """
    Compute monthly offset = mean(KNYC - airport) over the offset period.
    Returns dict: {month_int: offset_value}.
    """
    mask = dates.between(pd.Timestamp(OFFSET_START), pd.Timestamp(OFFSET_END))
    both_valid = mask & knyc_col.notna() & airport_col.notna()

    knyc_sub = knyc_col[both_valid]
    airport_sub = airport_col[both_valid]
    months_sub = dates[both_valid].dt.month

    offsets = {}
    for m in range(1, 13):
        m_mask = months_sub == m
        if m_mask.sum() > 0:
            offsets[m] = (knyc_sub[m_mask] - airport_sub[m_mask]).mean()
        else:
            offsets[m] = 0.0
    return offsets


def apply_monthly_offsets(airport_col: pd.Series, dates: pd.Series,
                          offsets: dict) -> pd.Series:
    """Apply monthly offsets to harmonize airport forecasts to KNYC-equivalent."""
    harmonized = airport_col.copy()
    for m, offset in offsets.items():
        m_mask = dates.dt.month == m
        harmonized[m_mask] = airport_col[m_mask] + offset
    return harmonized


def main():
    print("=" * 70)
    print("BUILD EXTENDED MOS: Multi-station average airport proxy backfill")
    print("=" * 70)

    # ── Load KNYC data ────────────────────────────────────────────────────
    knyc = pd.read_csv(KNYC_PATH, parse_dates=["date"])
    print(f"\nKNYC data: {len(knyc)} rows, {knyc['date'].min().date()} to {knyc['date'].max().date()}")

    # ── Load airport GFS data ─────────────────────────────────────────────
    gfs_data = load_airport_data(AIRPORT_GFS, "gfs")
    nam_data = load_airport_data(AIRPORT_NAM, "nam")

    # ── Merge all airport GFS into one DataFrame ──────────────────────────
    gfs_merged = gfs_data["KJFK"]
    for station in ["KLGA", "KEWR"]:
        gfs_merged = gfs_merged.merge(gfs_data[station], on="date", how="outer")

    nam_merged = nam_data["KJFK"]
    for station in ["KLGA", "KEWR"]:
        nam_merged = nam_merged.merge(nam_data[station], on="date", how="outer")

    print(f"\nAirport GFS merged: {len(gfs_merged)} rows, "
          f"{gfs_merged['date'].min().date()} to {gfs_merged['date'].max().date()}")
    print(f"Airport NAM merged: {len(nam_merged)} rows, "
          f"{nam_merged['date'].min().date()} to {nam_merged['date'].max().date()}")

    # ── Merge KNYC + airport data for offset computation ──────────────────
    # GFS offsets
    gfs_for_offsets = knyc[["date", "gfs_mos_tmax_f"]].merge(gfs_merged, on="date", how="inner")
    print(f"\nGFS offset computation: {len(gfs_for_offsets)} overlap rows")

    # NAM offsets
    nam_for_offsets = knyc[["date", "nam_mos_tmax_f"]].merge(nam_merged, on="date", how="inner")
    print(f"NAM offset computation: {len(nam_for_offsets)} overlap rows")

    # ── Compute monthly offsets ───────────────────────────────────────────
    print(f"\nComputing monthly offsets using {OFFSET_START} to {OFFSET_END}...")

    gfs_offsets = {}
    for station in ["KJFK", "KLGA", "KEWR"]:
        col = f"{station}_gfs"
        offsets = compute_monthly_offsets(
            gfs_for_offsets["gfs_mos_tmax_f"],
            gfs_for_offsets[col],
            gfs_for_offsets["date"]
        )
        gfs_offsets[station] = offsets

    nam_offsets = {}
    for station in ["KJFK", "KLGA", "KEWR"]:
        col = f"{station}_nam"
        offsets = compute_monthly_offsets(
            nam_for_offsets["nam_mos_tmax_f"],
            nam_for_offsets[col],
            nam_for_offsets["date"]
        )
        nam_offsets[station] = offsets

    # Print offset tables
    print("\n── GFS Monthly Offsets (KNYC - Airport, added to airport to harmonize) ──")
    print(f"{'Month':>6}  {'KJFK':>8}  {'KLGA':>8}  {'KEWR':>8}")
    for m in range(1, 13):
        print(f"{m:>6}  {gfs_offsets['KJFK'][m]:>8.2f}  {gfs_offsets['KLGA'][m]:>8.2f}  "
              f"{gfs_offsets['KEWR'][m]:>8.2f}")

    print("\n── NAM Monthly Offsets (KNYC - Airport, added to airport to harmonize) ──")
    print(f"{'Month':>6}  {'KJFK':>8}  {'KLGA':>8}  {'KEWR':>8}")
    for m in range(1, 13):
        print(f"{m:>6}  {nam_offsets['KJFK'][m]:>8.2f}  {nam_offsets['KLGA'][m]:>8.2f}  "
              f"{nam_offsets['KEWR'][m]:>8.2f}")

    # ── Build proxy for backfill period ───────────────────────────────────
    print(f"\nBuilding proxy for backfill period {BACKFILL_START} to {BACKFILL_END}...")

    # Create date range for backfill
    backfill_dates = pd.date_range(BACKFILL_START, BACKFILL_END, freq="D")
    backfill = pd.DataFrame({"date": backfill_dates})

    # Merge airport GFS data for backfill period
    backfill = backfill.merge(gfs_merged, on="date", how="left")
    backfill = backfill.merge(nam_merged, on="date", how="left")

    # Apply harmonization offsets to each airport station
    for station in ["KJFK", "KLGA", "KEWR"]:
        gfs_col = f"{station}_gfs"
        nam_col = f"{station}_nam"

        backfill[f"{gfs_col}_harm"] = apply_monthly_offsets(
            backfill[gfs_col], backfill["date"], gfs_offsets[station]
        )
        backfill[f"{nam_col}_harm"] = apply_monthly_offsets(
            backfill[nam_col], backfill["date"], nam_offsets[station]
        )

    # Compute multi-station average proxy for GFS
    gfs_harm_cols = [f"{s}_gfs_harm" for s in ["KJFK", "KLGA", "KEWR"]]
    backfill["proxy_gfs"] = backfill[gfs_harm_cols].mean(axis=1)

    # Compute multi-station average proxy for NAM
    nam_harm_cols = [f"{s}_nam_harm" for s in ["KJFK", "KLGA", "KEWR"]]
    backfill["proxy_nam"] = backfill[nam_harm_cols].mean(axis=1)

    # Compute ensemble: average of GFS and NAM where both exist, else just GFS
    backfill["proxy_ensemble"] = np.where(
        backfill["proxy_nam"].notna(),
        (backfill["proxy_gfs"] + backfill["proxy_nam"]) / 2,
        backfill["proxy_gfs"]
    )

    # Round to 1 decimal place
    backfill["proxy_gfs"] = backfill["proxy_gfs"].round(1)
    backfill["proxy_nam"] = backfill["proxy_nam"].round(1)
    backfill["proxy_ensemble"] = backfill["proxy_ensemble"].round(1)

    # Report backfill coverage
    gfs_coverage = backfill["proxy_gfs"].notna().sum()
    nam_coverage = backfill["proxy_nam"].notna().sum()
    ens_coverage = backfill["proxy_ensemble"].notna().sum()
    total_days = len(backfill)
    print(f"\nBackfill coverage ({total_days} total days):")
    print(f"  GFS proxy: {gfs_coverage} days ({100*gfs_coverage/total_days:.1f}%)")
    print(f"  NAM proxy: {nam_coverage} days ({100*nam_coverage/total_days:.1f}%)")
    print(f"  Ensemble:  {ens_coverage} days ({100*ens_coverage/total_days:.1f}%)")

    # ── Build extended CSV ────────────────────────────────────────────────
    print("\nBuilding extended combined MOS CSV...")

    # Format backfill rows to match KNYC column format
    backfill_out = pd.DataFrame({
        "date": backfill["date"],
        "gfs_mos_tmax_f": backfill["proxy_gfs"],
        "gfs_runtime": pd.NaT,  # No single runtime for averaged proxy
        "nam_mos_tmax_f": backfill["proxy_nam"],
        "nam_runtime": pd.NaT,
        "mos_ensemble_tmax_f": backfill["proxy_ensemble"],
        "mos_source": "airport_proxy",
    })

    # Drop rows where we have no proxy at all (missing GFS from all airports)
    backfill_out = backfill_out.dropna(subset=["gfs_mos_tmax_f"])

    # KNYC original data (2004+)
    knyc_out = knyc.copy()
    knyc_out["mos_source"] = "knyc_native"

    # Concatenate
    extended = pd.concat([backfill_out, knyc_out], ignore_index=True)
    extended = extended.sort_values("date").reset_index(drop=True)

    # Save
    extended.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved extended MOS CSV: {OUTPUT_PATH}")
    print(f"  Shape: {extended.shape}")
    print(f"  Date range: {extended['date'].min()} to {extended['date'].max()}")

    # ── Build era indicator CSV ───────────────────────────────────────────
    era_df = pd.DataFrame({
        "date": extended["date"],
        "mos_era": (extended["mos_source"] == "knyc_native").astype(int),
    })
    era_df.to_csv(ERA_PATH, index=False)
    print(f"Saved era indicator CSV: {ERA_PATH}")

    # ── Validation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # 1. Date range
    print(f"\n1. Date range: {extended['date'].min()} to {extended['date'].max()}")

    # 2. Check for gaps
    dates_sorted = pd.to_datetime(extended["date"]).sort_values().reset_index(drop=True)
    diffs = dates_sorted.diff().dt.days
    gaps = diffs[diffs > 1]
    if len(gaps) == 0:
        print("2. No date gaps found -- continuous coverage.")
    else:
        print(f"2. Found {len(gaps)} gaps in date sequence:")
        for idx in gaps.index[:10]:
            gap_start = dates_sorted[idx - 1]
            gap_end = dates_sorted[idx]
            print(f"   {gap_start.date()} -> {gap_end.date()} ({int(diffs[idx])-1} missing days)")
        if len(gaps) > 10:
            print(f"   ... and {len(gaps) - 10} more gaps")

    # 3. Verify 2004+ data is identical to original KNYC
    knyc_orig = pd.read_csv(KNYC_PATH)
    ext_2004 = extended[extended["mos_source"] == "knyc_native"].copy()
    ext_2004 = ext_2004.drop(columns=["mos_source"]).reset_index(drop=True)
    # Compare key columns
    cols_to_check = ["date", "gfs_mos_tmax_f", "nam_mos_tmax_f", "mos_ensemble_tmax_f"]
    knyc_check = knyc_orig[cols_to_check].copy()
    ext_check = ext_2004[cols_to_check].copy()
    # Convert dates to string for comparison
    knyc_check["date"] = pd.to_datetime(knyc_check["date"]).dt.strftime("%Y-%m-%d")
    ext_check["date"] = pd.to_datetime(ext_check["date"]).dt.strftime("%Y-%m-%d")

    match = True
    for col in cols_to_check:
        if col == "date":
            if not (knyc_check[col].values == ext_check[col].values).all():
                print(f"3. MISMATCH in column '{col}'!")
                match = False
        else:
            kv = knyc_check[col].values.astype(float)
            ev = ext_check[col].values.astype(float)
            # Handle NaN comparison
            both_nan = np.isnan(kv) & np.isnan(ev)
            both_equal = np.isclose(kv, ev, equal_nan=True)
            if not (both_nan | both_equal).all():
                mismatches = (~(both_nan | both_equal)).sum()
                print(f"3. MISMATCH in column '{col}': {mismatches} differences!")
                match = False

    if match:
        print(f"3. 2004+ data matches original KNYC CSV exactly ({len(knyc_orig)} rows verified).")
    else:
        print("3. WARNING: 2004+ data does NOT match original KNYC CSV!")

    # 4. Compute MAE vs actual Central Park TMAX
    cp_actual = pd.read_csv(CP_ACTUAL_PATH, parse_dates=["date"])

    # Merge extended with actuals
    eval_df = extended[["date", "mos_ensemble_tmax_f", "mos_source"]].copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    eval_df = eval_df.merge(cp_actual, on="date", how="left")

    # Proxy period (2000-2003)
    proxy_mask = eval_df["mos_source"] == "airport_proxy"
    proxy_valid = proxy_mask & eval_df["tmax_f"].notna() & eval_df["mos_ensemble_tmax_f"].notna()
    proxy_errors = (eval_df.loc[proxy_valid, "mos_ensemble_tmax_f"]
                    - eval_df.loc[proxy_valid, "tmax_f"]).abs()
    proxy_mae = proxy_errors.mean()
    proxy_bias = (eval_df.loc[proxy_valid, "mos_ensemble_tmax_f"]
                  - eval_df.loc[proxy_valid, "tmax_f"]).mean()
    proxy_n = proxy_valid.sum()

    # KNYC native period (2004-2019, train period for fair comparison)
    native_mask = (eval_df["mos_source"] == "knyc_native") & (eval_df["date"] < "2020-01-01")
    native_valid = native_mask & eval_df["tmax_f"].notna() & eval_df["mos_ensemble_tmax_f"].notna()
    native_errors = (eval_df.loc[native_valid, "mos_ensemble_tmax_f"]
                     - eval_df.loc[native_valid, "tmax_f"]).abs()
    native_mae = native_errors.mean()
    native_bias = (eval_df.loc[native_valid, "mos_ensemble_tmax_f"]
                   - eval_df.loc[native_valid, "tmax_f"]).mean()
    native_n = native_valid.sum()

    # Also check by year within proxy period
    print(f"\n4. MAE vs Actual Central Park TMAX:")
    print(f"\n   Airport Proxy (2000-2003):")
    print(f"     N days: {proxy_n}")
    print(f"     MAE:    {proxy_mae:.2f}°F")
    print(f"     Bias:   {proxy_bias:+.2f}°F")

    print(f"\n   KNYC Native (2004-2019):")
    print(f"     N days: {native_n}")
    print(f"     MAE:    {native_mae:.2f}°F")
    print(f"     Bias:   {native_bias:+.2f}°F")

    print(f"\n   MAE degradation (proxy vs native): {proxy_mae - native_mae:+.2f}°F")

    # Yearly breakdown for proxy period
    print(f"\n   Yearly breakdown (proxy period):")
    proxy_df = eval_df[proxy_valid].copy()
    proxy_df["year"] = proxy_df["date"].dt.year
    for year, grp in proxy_df.groupby("year"):
        yr_mae = (grp["mos_ensemble_tmax_f"] - grp["tmax_f"]).abs().mean()
        yr_bias = (grp["mos_ensemble_tmax_f"] - grp["tmax_f"]).mean()
        yr_n = len(grp)
        # Separate GFS-only vs ensemble
        gfs_only = grp[grp["date"] < "2002-04-03"]  # Before NAM starts
        if len(gfs_only) > 0:
            gfs_mae = (gfs_only["mos_ensemble_tmax_f"] - gfs_only["tmax_f"]).abs().mean()
            print(f"     {year}: MAE={yr_mae:.2f}°F, Bias={yr_bias:+.2f}°F, N={yr_n} "
                  f"(GFS-only days: {len(gfs_only)}, GFS-only MAE: {gfs_mae:.2f}°F)")
        else:
            print(f"     {year}: MAE={yr_mae:.2f}°F, Bias={yr_bias:+.2f}°F, N={yr_n}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    proxy_rows = (extended["mos_source"] == "airport_proxy").sum()
    native_rows = (extended["mos_source"] == "knyc_native").sum()
    print(f"  Extended MOS file: {OUTPUT_PATH}")
    print(f"  Total rows:        {len(extended)}")
    print(f"  Airport proxy:     {proxy_rows} rows ({BACKFILL_START} to {BACKFILL_END})")
    print(f"  KNYC native:       {native_rows} rows (2004-01-01 to {extended['date'].max()})")
    print(f"  Date range:        {extended['date'].min()} to {extended['date'].max()}")
    print(f"  Proxy MAE:         {proxy_mae:.2f}°F")
    print(f"  Native MAE:        {native_mae:.2f}°F (2004-2019 reference)")
    print(f"  Era indicator:     {ERA_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
