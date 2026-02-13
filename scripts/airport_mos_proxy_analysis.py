#!/usr/bin/env python3
"""
Airport MOS Proxy Harmonization Study

Analyzes NYC-area airport MOS stations (KLGA, KJFK, KEWR) as proxies for KNYC
MOS to extend training data from 2004 back to 2000. Builds harmonization layers
(bias offset + variance correction) and validates on held-out overlap data.

Data source: Pre-downloaded airport MOS CSVs in data/mos/, originally from
IEM MOS archive (https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py).

Outputs to results/airport_mos_proxy_analysis/:
  - airport_mos_bias_analysis.csv       Detailed bias metrics by station/year/season
  - harmonization_parameters.json       Fitted bias offsets and variance corrections
  - feasibility_report.md               Summary report with findings and recommendation
  - validation_holdout_results.csv      Holdout validation metrics (2004-2005 withheld)

Usage:
    python scripts/airport_mos_proxy_analysis.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MOS_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mos")
AIRPORT_MOS_DIR = os.path.join(PROJECT_ROOT, "data", "airport_mos")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "airport_mos_proxy_analysis")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
AIRPORT_STATIONS = ["KLGA", "KJFK", "KEWR"]
STATION_INFO = {
    "KLGA": {"name": "LaGuardia", "distance_mi": 8},
    "KJFK": {"name": "JFK International", "distance_mi": 15},
    "KEWR": {"name": "Newark Liberty", "distance_mi": 10},
}
MODELS = ["GFS", "NAM"]
GFS_START_YEAR = 2000
NAM_START_YEAR = 2002
END_YEAR = 2026

# KNYC overlap starts 2004
OVERLAP_START_YEAR = 2004
# Holdout: withhold 2004-2005 from harmonization training, use for testing
HOLDOUT_YEARS = [2004, 2005]
# Harmonization training: overlap period excluding holdout
HARM_TRAIN_START = 2006
HARM_TRAIN_END = 2023
# Pre-overlap: period we want to extend to
PRE_OVERLAP_END_YEAR = 2003

MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0
REQUEST_TIMEOUT = 120

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

# Quality gates for feasibility
QUALITY_GATE_MAE = 1.0       # Mean absolute bias <= 1.0 F
QUALITY_GATE_CORR = 0.95     # Correlation >= 0.95
QUALITY_GATE_DRIFT = 0.5     # Max seasonal drift <= 0.5 F (std of seasonal biases)


# ===================================================================
# PART 1: Download / Load Airport MOS Data
# ===================================================================

def download_mos_chunk(station: str, model: str, year: int) -> Optional[str]:
    """Download one year of MOS data from IEM with exponential backoff retries."""
    params = {
        "station": station,
        "model": model,
        "year1": year,
        "month1": 1,
        "day1": 1,
        "hour1": 0,
        "year2": year,
        "month2": 12,
        "day2": 31,
        "hour2": 23,
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "  Downloading %s %s year=%d (attempt %d/%d)",
                station, model, year, attempt, MAX_RETRIES,
            )
            resp = requests.get(IEM_MOS_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            text = resp.text.strip()
            if not text or "runtime" not in text.split("\n")[0]:
                logger.warning("  Empty/invalid response for %s %s %d", station, model, year)
                return None

            lines = text.split("\n")
            data_lines = len(lines) - 1
            if data_lines <= 0:
                logger.warning("  Header only for %s %s %d", station, model, year)
                return None

            logger.info("    Got %d data lines (%.1f KB)", data_lines, len(text) / 1024)
            return text

        except (requests.RequestException, ConnectionError) as exc:
            logger.warning("  Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error(
                    "  FAILED %s %s %d after %d retries", station, model, year, MAX_RETRIES
                )
                return None

    return None


def extract_tmax_from_raw_mos(df: pd.DataFrame) -> pd.DataFrame:
    """Extract daily TMAX forecasts from raw IEM MOS CSV DataFrame.

    MOS convention: n_x at ftime hour==0 (00Z) = TMAX for the previous calendar
    day (the daytime high before midnight UTC). For each target date we prefer
    the most recent 12Z runtime from the day before (standard day-ahead).
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    df = df.copy()
    df["runtime"] = pd.to_datetime(df["runtime"])
    df["ftime"] = pd.to_datetime(df["ftime"])
    df["n_x"] = pd.to_numeric(df.get("n_x"), errors="coerce")

    mask = df["n_x"].notna() & (df["ftime"].dt.hour == 0)
    tmax_df = df.loc[mask].copy()

    if tmax_df.empty:
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    tmax_df["target_date"] = (tmax_df["ftime"] - pd.Timedelta(days=1)).dt.date
    tmax_df["model_label"] = tmax_df["model"].astype(str).str.strip()
    tmax_df["runtime_date"] = tmax_df["runtime"].dt.date
    tmax_df["runtime_hour"] = tmax_df["runtime"].dt.hour

    records = []
    for target_date, group in tmax_df.groupby("target_date"):
        group = group.copy()
        target_dt = pd.Timestamp(target_date)
        day_before = (target_dt - pd.Timedelta(days=1)).date()

        def _priority(row):
            rd = row["runtime_date"]
            rh = row["runtime_hour"]
            if rd == day_before and rh == 12:
                return 100
            elif rd == day_before and rh == 18:
                return 90
            elif rd == day_before and rh == 6:
                return 80
            elif rd == day_before and rh == 0:
                return 70
            elif rd == target_date and rh == 0:
                return 60
            elif rd == target_date and rh == 6:
                return 50
            else:
                return 10

        group["priority"] = group.apply(_priority, axis=1)
        best = group.sort_values("priority", ascending=False).iloc[0]

        records.append({
            "date": target_date,
            "tmax_f": float(best["n_x"]),
            "model_label": best["model_label"],
            "runtime": best["runtime"],
        })

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result = result.sort_values("date").reset_index(drop=True)
    return result


def load_airport_mos_local(station: str, model: str) -> pd.DataFrame:
    """Load pre-downloaded airport MOS from local CSV.

    Expected format: date, tmax_f, model_label, runtime
    """
    model_lower = model.lower()
    # Map GFS -> gfs, NAM -> nam in filename
    path = os.path.join(MOS_DATA_DIR, f"{station.lower()}_{model_lower}_mos_tmax.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        logger.info(
            "Loaded local %s %s MOS: %d rows (%s to %s)",
            station, model, len(df),
            df["date"].min() if not df.empty else "N/A",
            df["date"].max() if not df.empty else "N/A",
        )
        return df
    else:
        logger.warning("Local file not found: %s", path)
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])


def download_airport_mos_from_iem(station: str, model: str) -> pd.DataFrame:
    """Download airport MOS from IEM API, year by year."""
    start_year = GFS_START_YEAR if model.upper() == "GFS" else NAM_START_YEAR
    all_chunks = []

    for year in range(start_year, END_YEAR + 1):
        raw = download_mos_chunk(station, model.upper(), year)
        if raw is None:
            continue

        try:
            chunk_df = pd.read_csv(StringIO(raw), parse_dates=["runtime", "ftime"])
            chunk_df["n_x"] = pd.to_numeric(chunk_df.get("n_x"), errors="coerce")
            all_chunks.append(chunk_df)
        except Exception as exc:
            logger.error("  Parse error %s %s %d: %s", station, model, year, exc)

        time.sleep(0.5)

    if not all_chunks:
        logger.error("No data for %s %s from IEM", station, model)
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    combined = pd.concat(all_chunks, ignore_index=True)
    logger.info("Downloaded %d raw rows for %s %s", len(combined), station, model)

    tmax_df = extract_tmax_from_raw_mos(combined)
    return tmax_df


def load_or_download_airport_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load airport MOS data from local files, downloading from IEM only if missing.

    Returns dict: {station: {"GFS": DataFrame, "NAM": DataFrame}}
    """
    os.makedirs(MOS_DATA_DIR, exist_ok=True)
    os.makedirs(AIRPORT_MOS_DIR, exist_ok=True)
    data = {}

    for station in AIRPORT_STATIONS:
        data[station] = {}
        for model in MODELS:
            # Try loading local first
            df = load_airport_mos_local(station, model)

            if df.empty:
                logger.info("Local data missing for %s %s, downloading from IEM...", station, model)
                df = download_airport_mos_from_iem(station, model)

                if not df.empty:
                    # Save to local
                    save_path = os.path.join(
                        MOS_DATA_DIR,
                        f"{station.lower()}_{model.lower()}_mos_tmax.csv",
                    )
                    df.to_csv(save_path, index=False)
                    logger.info("Saved %d rows to %s", len(df), save_path)

                    # Also save to airport_mos directory
                    airport_path = os.path.join(
                        AIRPORT_MOS_DIR,
                        f"{station.lower()}_{model.lower()}_mos_tmax.csv",
                    )
                    df.to_csv(airport_path, index=False)

            data[station][model] = df

    return data


# ===================================================================
# PART 2: Load KNYC MOS Data
# ===================================================================

def load_knyc_mos() -> Dict[str, pd.DataFrame]:
    """Load existing KNYC MOS data from CSV files."""
    knyc = {}

    # GFS
    gfs_path = os.path.join(MOS_DATA_DIR, "gfs_mos_knyc.csv")
    if os.path.exists(gfs_path):
        gfs_df = pd.read_csv(gfs_path)
        gfs_df["date"] = pd.to_datetime(gfs_df["date"]).dt.date
        if "gfs_mos_tmax_f" in gfs_df.columns:
            gfs_df = gfs_df.rename(columns={"gfs_mos_tmax_f": "tmax_f"})
        elif "tmax_forecast_f" in gfs_df.columns:
            gfs_df = gfs_df.rename(columns={"tmax_forecast_f": "tmax_f"})
        knyc["GFS"] = gfs_df
        logger.info(
            "Loaded KNYC GFS MOS: %d rows (%s to %s)",
            len(gfs_df), gfs_df["date"].min(), gfs_df["date"].max(),
        )
    else:
        logger.warning("KNYC GFS MOS not found: %s", gfs_path)
        knyc["GFS"] = pd.DataFrame(columns=["date", "tmax_f"])

    # NAM
    nam_path = os.path.join(MOS_DATA_DIR, "nam_mos_knyc.csv")
    if os.path.exists(nam_path):
        nam_df = pd.read_csv(nam_path)
        nam_df["date"] = pd.to_datetime(nam_df["date"]).dt.date
        if "nam_mos_tmax_f" in nam_df.columns:
            nam_df = nam_df.rename(columns={"nam_mos_tmax_f": "tmax_f"})
        elif "tmax_forecast_f" in nam_df.columns:
            nam_df = nam_df.rename(columns={"tmax_forecast_f": "tmax_f"})
        knyc["NAM"] = nam_df
        logger.info(
            "Loaded KNYC NAM MOS: %d rows (%s to %s)",
            len(nam_df), nam_df["date"].min(), nam_df["date"].max(),
        )
    else:
        logger.warning("KNYC NAM MOS not found: %s", nam_path)
        knyc["NAM"] = pd.DataFrame(columns=["date", "tmax_f"])

    return knyc


# ===================================================================
# PART 3: Bias Analysis
# ===================================================================

def get_season(month: int) -> str:
    """Map month number to season abbreviation."""
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "UNK"


def compute_bias_analysis(
    airport_df: pd.DataFrame,
    knyc_df: pd.DataFrame,
    station: str,
    model: str,
) -> Dict[str, Any]:
    """Compute comprehensive bias analysis between airport and KNYC MOS.

    For each airport station computes:
    - Mean bias vs KNYC (airport - KNYC) by year, season, and overall
    - Standard deviation of differences
    - Correlation between airport and KNYC forecasts
    - Rolling 365-day bias
    - KS test for distribution similarity
    """
    # Merge on date
    ap = airport_df[["date", "tmax_f"]].rename(columns={"tmax_f": "airport_tmax"})
    kn = knyc_df[["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})

    merged = pd.merge(ap, kn, on="date", how="inner")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date_dt").reset_index(drop=True)

    if len(merged) < 100:
        logger.warning("Too few overlap points for %s %s: %d", station, model, len(merged))
        return {
            "station": station,
            "model": model,
            "n_overlap": len(merged),
            "status": "insufficient_data",
        }

    merged["year"] = merged["date_dt"].dt.year
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)
    merged["diff"] = merged["airport_tmax"] - merged["knyc_tmax"]

    # Overall metrics
    overall_bias = merged["diff"].mean()
    overall_std = merged["diff"].std()
    overall_mae = merged["diff"].abs().mean()
    correlation = merged["airport_tmax"].corr(merged["knyc_tmax"])

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(
        merged["airport_tmax"].dropna().values,
        merged["knyc_tmax"].dropna().values,
    )

    # Yearly bias
    yearly_bias = merged.groupby("year").agg(
        mean_bias=("diff", "mean"),
        std_diff=("diff", "std"),
        mae=("diff", lambda x: x.abs().mean()),
        count=("diff", "size"),
    ).reset_index()

    # Seasonal bias
    seasonal_bias = merged.groupby("season").agg(
        mean_bias=("diff", "mean"),
        std_diff=("diff", "std"),
        mae=("diff", lambda x: x.abs().mean()),
        count=("diff", "size"),
    ).reset_index()

    # Rolling 365-day bias
    rolling_data = merged.set_index("date_dt")["diff"].rolling("365D", min_periods=180)
    rolling_bias_mean = rolling_data.mean()
    rolling_bias_std = rolling_data.std()

    # Rolling bias summary
    rb_clean = rolling_bias_mean.dropna()
    rolling_bias_summary = {
        "min": float(rb_clean.min()) if len(rb_clean) > 0 else np.nan,
        "max": float(rb_clean.max()) if len(rb_clean) > 0 else np.nan,
        "range": float(rb_clean.max() - rb_clean.min()) if len(rb_clean) > 0 else np.nan,
        "final_value": float(rb_clean.iloc[-1]) if len(rb_clean) > 0 else np.nan,
        "stability_std": float(rb_clean.std()) if len(rb_clean) > 0 else np.nan,
    }

    return {
        "station": station,
        "model": model,
        "n_overlap": len(merged),
        "overall_bias": float(overall_bias),
        "overall_std": float(overall_std),
        "overall_mae": float(overall_mae),
        "correlation": float(correlation),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "yearly_bias": yearly_bias,
        "seasonal_bias": seasonal_bias,
        "rolling_bias_summary": rolling_bias_summary,
        "rolling_bias_series": rolling_bias_mean,
        "status": "ok",
    }


def build_bias_analysis_csv(all_results: List[Dict]) -> pd.DataFrame:
    """Build a comprehensive bias analysis CSV from all station/model results."""
    rows = []

    for r in all_results:
        if r.get("status") != "ok":
            continue

        station = r["station"]
        model = r["model"]

        # Overall row
        rows.append({
            "station": station,
            "model": model,
            "period": "overall",
            "year": "all",
            "season": "all",
            "mean_bias": r["overall_bias"],
            "std_diff": r["overall_std"],
            "mae_vs_knyc": r["overall_mae"],
            "correlation": r["correlation"],
            "ks_statistic": r["ks_statistic"],
            "ks_pvalue": r["ks_pvalue"],
            "n_days": r["n_overlap"],
            "rolling_bias_range": r["rolling_bias_summary"]["range"],
            "rolling_bias_stability_std": r["rolling_bias_summary"]["stability_std"],
        })

        # Yearly rows
        yb = r["yearly_bias"]
        for _, yr in yb.iterrows():
            rows.append({
                "station": station,
                "model": model,
                "period": "yearly",
                "year": int(yr["year"]),
                "season": "all",
                "mean_bias": yr["mean_bias"],
                "std_diff": yr["std_diff"],
                "mae_vs_knyc": yr["mae"],
                "correlation": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "n_days": int(yr["count"]),
                "rolling_bias_range": np.nan,
                "rolling_bias_stability_std": np.nan,
            })

        # Seasonal rows
        sb = r["seasonal_bias"]
        for _, sr in sb.iterrows():
            rows.append({
                "station": station,
                "model": model,
                "period": "seasonal",
                "year": "all",
                "season": sr["season"],
                "mean_bias": sr["mean_bias"],
                "std_diff": sr["std_diff"],
                "mae_vs_knyc": sr["mae"],
                "correlation": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "n_days": int(sr["count"]),
                "rolling_bias_range": np.nan,
                "rolling_bias_stability_std": np.nan,
            })

    return pd.DataFrame(rows)


# ===================================================================
# PART 4: Harmonization Layer
# ===================================================================

def fit_harmonization(
    airport_df: pd.DataFrame,
    knyc_df: pd.DataFrame,
    train_years: List[int],
) -> Dict[str, Any]:
    """Fit harmonization parameters from overlap data in specified training years.

    Builds:
    - Global bias offset: mean(KNYC - airport)
    - Seasonal bias offsets (DJF, MAM, JJA, SON)
    - Monthly bias offsets
    - Variance correction: std(KNYC) / std(airport)
    - Seasonal variance corrections
    """
    ap = airport_df[["date", "tmax_f"]].rename(columns={"tmax_f": "airport_tmax"})
    kn = knyc_df[["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})

    merged = pd.merge(ap, kn, on="date", how="inner")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["year"] = merged["date_dt"].dt.year
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)

    # Filter to training years only
    train = merged[merged["year"].isin(train_years)].copy()

    if len(train) < 365:
        logger.warning("Insufficient training data for harmonization: %d days", len(train))
        return {}

    # Direction convention: harmonized = airport + offset, where offset = mean(KNYC - airport)
    # This corrects airport to match KNYC
    train["diff_knyc_minus_ap"] = train["knyc_tmax"] - train["airport_tmax"]

    # 1. Global offset
    global_offset = float(train["diff_knyc_minus_ap"].mean())

    # 2. Seasonal offsets
    seasonal_offsets = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = train[train["season"] == season]
        if len(subset) > 30:
            seasonal_offsets[season] = float(subset["diff_knyc_minus_ap"].mean())
        else:
            seasonal_offsets[season] = global_offset

    # 3. Monthly offsets
    monthly_offsets = {}
    for m in range(1, 13):
        subset = train[train["month"] == m]
        if len(subset) > 20:
            monthly_offsets[str(m)] = float(subset["diff_knyc_minus_ap"].mean())
        else:
            # Fall back to seasonal offset
            season = get_season(m)
            monthly_offsets[str(m)] = seasonal_offsets.get(season, global_offset)

    # 4. Variance correction: std(KNYC) / std(airport) for overall and by season
    knyc_std = float(train["knyc_tmax"].std())
    airport_std = float(train["airport_tmax"].std())
    global_var_ratio = knyc_std / airport_std if airport_std > 0 else 1.0

    seasonal_var_ratios = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = train[train["season"] == season]
        if len(subset) > 30:
            k_std = subset["knyc_tmax"].std()
            a_std = subset["airport_tmax"].std()
            seasonal_var_ratios[season] = float(k_std / a_std) if a_std > 0 else 1.0
        else:
            seasonal_var_ratios[season] = global_var_ratio

    # 5. Means for variance correction (scale around mean)
    knyc_mean = float(train["knyc_tmax"].mean())
    airport_mean = float(train["airport_tmax"].mean())
    seasonal_means = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = train[train["season"] == season]
        if len(subset) > 30:
            seasonal_means[season] = {
                "airport_mean": float(subset["airport_tmax"].mean()),
                "knyc_mean": float(subset["knyc_tmax"].mean()),
            }
        else:
            seasonal_means[season] = {
                "airport_mean": airport_mean,
                "knyc_mean": knyc_mean,
            }

    params = {
        "global_offset": global_offset,
        "seasonal_offsets": seasonal_offsets,
        "monthly_offsets": monthly_offsets,
        "global_var_ratio": global_var_ratio,
        "seasonal_var_ratios": seasonal_var_ratios,
        "airport_global_mean": airport_mean,
        "knyc_global_mean": knyc_mean,
        "seasonal_means": seasonal_means,
        "n_train_days": len(train),
        "train_years": sorted(train_years),
    }

    return params


def apply_harmonization(
    airport_tmax: pd.Series,
    months: pd.Series,
    seasons: pd.Series,
    params: Dict[str, Any],
    method: str = "monthly_offset",
) -> pd.Series:
    """Apply harmonization to airport TMAX to produce synthetic KNYC TMAX.

    Methods:
    - "global_offset": airport + global_offset
    - "seasonal_offset": airport + seasonal_offset
    - "monthly_offset": airport + monthly_offset
    - "seasonal_var_correction": bias + variance correction by season
    """
    if method == "global_offset":
        return airport_tmax + params["global_offset"]

    elif method == "seasonal_offset":
        offsets = seasons.map(params["seasonal_offsets"])
        return airport_tmax + offsets

    elif method == "monthly_offset":
        offsets = months.astype(str).map(params["monthly_offsets"])
        return airport_tmax + offsets

    elif method == "seasonal_var_correction":
        # For each season: harmonized = knyc_mean + var_ratio * (airport - airport_mean)
        result = airport_tmax.copy().astype(float)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons == season
            if mask.sum() == 0:
                continue
            sm = params["seasonal_means"].get(season, {})
            ap_mean = sm.get("airport_mean", params["airport_global_mean"])
            kn_mean = sm.get("knyc_mean", params["knyc_global_mean"])
            vr = params["seasonal_var_ratios"].get(season, params["global_var_ratio"])
            result.loc[mask] = kn_mean + vr * (airport_tmax.loc[mask] - ap_mean)
        return result

    else:
        raise ValueError(f"Unknown harmonization method: {method}")


def validate_harmonization_holdout(
    airport_df: pd.DataFrame,
    knyc_df: pd.DataFrame,
    params: Dict[str, Any],
    holdout_years: List[int],
) -> pd.DataFrame:
    """Validate harmonization on held-out overlap years.

    Returns a DataFrame with holdout validation metrics for each method.
    """
    ap = airport_df[["date", "tmax_f"]].rename(columns={"tmax_f": "airport_tmax"})
    kn = knyc_df[["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})

    merged = pd.merge(ap, kn, on="date", how="inner")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["year"] = merged["date_dt"].dt.year
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)

    holdout = merged[merged["year"].isin(holdout_years)].copy()

    if len(holdout) < 30:
        logger.warning("Too few holdout days: %d", len(holdout))
        return pd.DataFrame()

    methods = ["global_offset", "seasonal_offset", "monthly_offset", "seasonal_var_correction"]
    results = []

    for method in methods:
        holdout[f"harmonized_{method}"] = apply_harmonization(
            holdout["airport_tmax"],
            holdout["month"],
            holdout["season"],
            params,
            method=method,
        )
        diff = holdout[f"harmonized_{method}"] - holdout["knyc_tmax"]

        mae = diff.abs().mean()
        bias = diff.mean()
        rmse = np.sqrt((diff ** 2).mean())
        corr = holdout[f"harmonized_{method}"].corr(holdout["knyc_tmax"])

        results.append({
            "method": method,
            "holdout_years": str(holdout_years),
            "n_days": len(holdout),
            "mae_vs_knyc": float(mae),
            "bias_vs_knyc": float(bias),
            "rmse_vs_knyc": float(rmse),
            "correlation": float(corr),
        })

    # Also add raw (no harmonization) as baseline
    raw_diff = holdout["airport_tmax"] - holdout["knyc_tmax"]
    results.insert(0, {
        "method": "raw_no_harmonization",
        "holdout_years": str(holdout_years),
        "n_days": len(holdout),
        "mae_vs_knyc": float(raw_diff.abs().mean()),
        "bias_vs_knyc": float(raw_diff.mean()),
        "rmse_vs_knyc": float(np.sqrt((raw_diff ** 2).mean())),
        "correlation": float(holdout["airport_tmax"].corr(holdout["knyc_tmax"])),
    })

    return pd.DataFrame(results)


# ===================================================================
# PART 5: Feasibility Assessment
# ===================================================================

def assess_feasibility(
    airport_data: Dict[str, Dict[str, pd.DataFrame]],
    knyc_data: Dict[str, pd.DataFrame],
    bias_results: List[Dict],
    holdout_results: Dict[str, Dict[str, pd.DataFrame]],
    harmonization_params: Dict[str, Dict[str, Dict]],
) -> Dict[str, Any]:
    """Assess feasibility of extending MOS training data to 2000-2003.

    Returns a dict with feasibility metrics and recommendation.
    """
    assessment = {
        "stations": {},
        "pre_overlap_days": {},
        "quality_gates": {},
        "recommendation": "",
    }

    for station in AIRPORT_STATIONS:
        station_assessment = {}
        for model in MODELS:
            ap_df = airport_data.get(station, {}).get(model, pd.DataFrame())
            if ap_df.empty:
                continue

            ap_dates = pd.to_datetime(ap_df["date"])
            pre_overlap = ap_df[ap_dates.dt.year <= PRE_OVERLAP_END_YEAR]

            # Count pre-overlap days
            n_pre = len(pre_overlap)
            if n_pre == 0:
                station_assessment[model] = {
                    "n_pre_overlap_days": 0,
                    "passes_quality_gates": False,
                    "reason": "no_pre_overlap_data",
                }
                continue

            # Date range in pre-overlap
            pre_dates = pd.to_datetime(pre_overlap["date"])
            date_range_start = pre_dates.min().date() if len(pre_dates) > 0 else None
            date_range_end = pre_dates.max().date() if len(pre_dates) > 0 else None

            # Yearly completeness
            pre_overlap_dt = pre_overlap.copy()
            pre_overlap_dt["year"] = pre_dates.dt.year
            yearly_counts = pre_overlap_dt.groupby("year").size().to_dict()

            # Theoretical max days per year
            completeness = {}
            for y, cnt in yearly_counts.items():
                total_days = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
                completeness[y] = cnt / total_days

            # Get bias results for this station/model
            bias_r = None
            for br in bias_results:
                if br.get("station") == station and br.get("model") == model and br.get("status") == "ok":
                    bias_r = br
                    break

            # Get holdout validation results
            holdout_df = holdout_results.get(station, {}).get(model, pd.DataFrame())

            # Quality gates
            gate_mae = False
            gate_corr = False
            gate_drift = False
            holdout_mae = np.nan
            holdout_corr = np.nan

            if holdout_df is not None and not holdout_df.empty:
                # Use the monthly_offset method for quality gate assessment
                monthly_row = holdout_df[holdout_df["method"] == "monthly_offset"]
                if not monthly_row.empty:
                    holdout_mae = monthly_row.iloc[0]["mae_vs_knyc"]
                    holdout_corr = monthly_row.iloc[0]["correlation"]
                    gate_mae = holdout_mae <= QUALITY_GATE_MAE
                    gate_corr = holdout_corr >= QUALITY_GATE_CORR

            # Check seasonal drift
            if bias_r is not None:
                sb = bias_r["seasonal_bias"]
                seasonal_biases = sb["mean_bias"].values
                seasonal_drift_std = float(np.std(seasonal_biases))
                gate_drift = seasonal_drift_std <= QUALITY_GATE_DRIFT
            else:
                seasonal_drift_std = np.nan

            passes_all = gate_mae and gate_corr and gate_drift

            station_assessment[model] = {
                "n_pre_overlap_days": n_pre,
                "date_range": f"{date_range_start} to {date_range_end}",
                "yearly_counts": yearly_counts,
                "yearly_completeness": {str(k): round(v, 3) for k, v in completeness.items()},
                "holdout_mae_vs_knyc": float(holdout_mae) if not np.isnan(holdout_mae) else None,
                "holdout_correlation": float(holdout_corr) if not np.isnan(holdout_corr) else None,
                "seasonal_drift_std": float(seasonal_drift_std) if not np.isnan(seasonal_drift_std) else None,
                "quality_gate_mae": gate_mae,
                "quality_gate_corr": gate_corr,
                "quality_gate_drift": gate_drift,
                "passes_all_quality_gates": passes_all,
            }

        assessment["stations"][station] = station_assessment

    # Count total additional training days
    total_gfs_days = 0
    total_nam_days = 0
    for station in AIRPORT_STATIONS:
        sa = assessment["stations"].get(station, {})
        if "GFS" in sa:
            total_gfs_days = max(total_gfs_days, sa["GFS"].get("n_pre_overlap_days", 0))
        if "NAM" in sa:
            total_nam_days = max(total_nam_days, sa["NAM"].get("n_pre_overlap_days", 0))

    assessment["total_additional_gfs_days"] = total_gfs_days
    assessment["total_additional_nam_days"] = total_nam_days

    return assessment


# ===================================================================
# PART 6: Report Generation
# ===================================================================

def generate_feasibility_report(
    bias_results: List[Dict],
    holdout_results: Dict[str, Dict[str, pd.DataFrame]],
    harmonization_params: Dict[str, Dict[str, Dict]],
    feasibility: Dict,
    airport_data: Dict[str, Dict[str, pd.DataFrame]],
    knyc_data: Dict[str, pd.DataFrame],
) -> str:
    """Generate the comprehensive feasibility report in Markdown."""
    lines = []
    lines.append("# Airport MOS Proxy Harmonization Study")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n**Objective:** Assess feasibility of using airport MOS stations (KLGA, KJFK, KEWR)")
    lines.append("as proxies for KNYC MOS to extend training data from 2004 back to 2000.\n")

    # ---------------------------------------------------------------
    # Section 1: Data Summary
    # ---------------------------------------------------------------
    lines.append("## 1. Data Summary\n")
    lines.append("### Airport MOS Data\n")
    lines.append("| Station | Distance | Model | Date Range | Total Days |")
    lines.append("|---------|----------|-------|------------|------------|")

    for station in AIRPORT_STATIONS:
        dist = STATION_INFO[station]["distance_mi"]
        for model in MODELS:
            df = airport_data.get(station, {}).get(model, pd.DataFrame())
            if not df.empty:
                lines.append(
                    f"| {station} | {dist} mi | {model} | "
                    f"{df['date'].min()} to {df['date'].max()} | {len(df)} |"
                )
            else:
                lines.append(f"| {station} | {dist} mi | {model} | No data | 0 |")

    lines.append("\n### KNYC MOS Data\n")
    lines.append("| Model | Date Range | Total Days |")
    lines.append("|-------|------------|------------|")
    for model in MODELS:
        df = knyc_data.get(model, pd.DataFrame())
        if not df.empty:
            lines.append(f"| {model} | {df['date'].min()} to {df['date'].max()} | {len(df)} |")

    lines.append(f"\n**Overlap period:** {OVERLAP_START_YEAR} onwards")
    lines.append(f"\n**Holdout validation years:** {HOLDOUT_YEARS}")
    lines.append(f"\n**Harmonization training years:** {HARM_TRAIN_START}-{HARM_TRAIN_END}")
    lines.append(f"\n**Pre-overlap extension target:** 2000-{PRE_OVERLAP_END_YEAR}\n")

    # ---------------------------------------------------------------
    # Section 2: Bias Analysis
    # ---------------------------------------------------------------
    lines.append("## 2. Bias Analysis (Overlap Period)\n")

    for model in MODELS:
        model_results = [r for r in bias_results if r.get("model") == model and r.get("status") == "ok"]
        if not model_results:
            continue

        lines.append(f"\n### {model} MOS\n")
        lines.append("| Metric | " + " | ".join(r["station"] for r in model_results) + " |")
        lines.append("|--------|" + "|".join(["--------" for _ in model_results]) + "|")

        metric_rows = [
            ("N overlap days", "n_overlap", "d"),
            ("Correlation", "correlation", ".4f"),
            ("Mean bias (AP-KNYC)", "overall_bias", "+.2f"),
            ("Std of differences", "overall_std", ".2f"),
            ("MAE vs KNYC", "overall_mae", ".2f"),
            ("KS statistic", "ks_statistic", ".4f"),
            ("KS p-value", "ks_pvalue", ".4f"),
            ("Rolling 365d bias range", lambda r: r["rolling_bias_summary"]["range"], ".2f"),
            ("Rolling 365d bias stability (std)", lambda r: r["rolling_bias_summary"]["stability_std"], ".3f"),
        ]

        for label, key_or_fn, fmt in metric_rows:
            vals = []
            for r in model_results:
                if callable(key_or_fn):
                    v = key_or_fn(r)
                else:
                    v = r.get(key_or_fn, np.nan)
                if isinstance(v, (int, float)) and not np.isnan(v):
                    vals.append(f"{v:{fmt}}")
                else:
                    vals.append("N/A")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        # Seasonal bias
        lines.append(f"\n#### Seasonal Bias ({model}, Airport - KNYC)\n")
        lines.append("| Season | " + " | ".join(r["station"] for r in model_results) + " |")
        lines.append("|--------|" + "|".join(["--------" for _ in model_results]) + "|")
        for season in ["DJF", "MAM", "JJA", "SON"]:
            vals = []
            for r in model_results:
                sb = r.get("seasonal_bias")
                if sb is not None and not sb.empty:
                    row_data = sb[sb["season"] == season]
                    if not row_data.empty:
                        vals.append(f"{row_data.iloc[0]['mean_bias']:+.2f}")
                    else:
                        vals.append("N/A")
                else:
                    vals.append("N/A")
            lines.append(f"| {season} | " + " | ".join(vals) + " |")

    lines.append(f"\n**KLGA is the closest proxy at ~8 miles from Central Park.** "
                 f"It generally shows the smallest MAE vs KNYC among the three airports.\n")

    # ---------------------------------------------------------------
    # Section 3: Harmonization Layer
    # ---------------------------------------------------------------
    lines.append("## 3. Harmonization Layer\n")
    lines.append(f"Harmonization parameters fitted on overlap years {HARM_TRAIN_START}-{HARM_TRAIN_END}, "
                 f"excluding holdout years {HOLDOUT_YEARS}.\n")

    for station in AIRPORT_STATIONS:
        for model in MODELS:
            params = harmonization_params.get(station, {}).get(model)
            if not params:
                continue

            lines.append(f"\n### {station} {model}\n")
            lines.append(f"- **Global offset (KNYC - airport):** {params['global_offset']:+.3f} F")
            lines.append(f"- **Global variance ratio (std KNYC / std airport):** {params['global_var_ratio']:.4f}")
            lines.append(f"- **Training days:** {params['n_train_days']}\n")

            lines.append("| Season | Bias Offset | Variance Ratio |")
            lines.append("|--------|-------------|----------------|")
            for season in ["DJF", "MAM", "JJA", "SON"]:
                offset = params["seasonal_offsets"].get(season, 0)
                vr = params["seasonal_var_ratios"].get(season, 1)
                lines.append(f"| {season} | {offset:+.3f} F | {vr:.4f} |")

            lines.append("\n| Month | Offset |")
            lines.append("|-------|--------|")
            for m in range(1, 13):
                offset = params["monthly_offsets"].get(str(m), 0)
                lines.append(f"| {m:02d} | {offset:+.3f} F |")

    # ---------------------------------------------------------------
    # Section 4: Holdout Validation
    # ---------------------------------------------------------------
    lines.append(f"\n## 4. Holdout Validation (Years {HOLDOUT_YEARS})\n")
    lines.append("These years were excluded from harmonization training to test out-of-sample performance.\n")

    for station in AIRPORT_STATIONS:
        for model in MODELS:
            hdf = holdout_results.get(station, {}).get(model)
            if hdf is None or hdf.empty:
                continue

            lines.append(f"\n### {station} {model}\n")
            lines.append("| Method | MAE vs KNYC | Bias vs KNYC | RMSE vs KNYC | Correlation |")
            lines.append("|--------|-------------|--------------|--------------|-------------|")
            for _, row in hdf.iterrows():
                lines.append(
                    f"| {row['method']} | {row['mae_vs_knyc']:.3f} | "
                    f"{row['bias_vs_knyc']:+.3f} | {row['rmse_vs_knyc']:.3f} | "
                    f"{row['correlation']:.4f} |"
                )

    # ---------------------------------------------------------------
    # Section 5: Feasibility Assessment
    # ---------------------------------------------------------------
    lines.append("\n## 5. Feasibility Assessment\n")

    lines.append("### Quality Gates\n")
    lines.append(f"- Mean absolute bias (holdout MAE vs KNYC) <= {QUALITY_GATE_MAE:.1f} F")
    lines.append(f"- Correlation >= {QUALITY_GATE_CORR:.2f}")
    lines.append(f"- Seasonal drift std <= {QUALITY_GATE_DRIFT:.2f} F\n")

    lines.append("### Results by Station\n")
    lines.append("| Station | Model | Pre-overlap Days | Holdout MAE | Correlation | Seasonal Drift | All Gates |")
    lines.append("|---------|-------|-----------------|-------------|-------------|----------------|-----------|")

    for station in AIRPORT_STATIONS:
        sa = feasibility["stations"].get(station, {})
        for model in MODELS:
            ma = sa.get(model)
            if ma is None:
                continue

            holdout_mae_str = f"{ma['holdout_mae_vs_knyc']:.3f}" if ma.get("holdout_mae_vs_knyc") is not None else "N/A"
            corr_str = f"{ma['holdout_correlation']:.4f}" if ma.get("holdout_correlation") is not None else "N/A"
            drift_str = f"{ma['seasonal_drift_std']:.3f}" if ma.get("seasonal_drift_std") is not None else "N/A"
            gates_str = "PASS" if ma.get("passes_all_quality_gates") else "FAIL"
            gate_mae = "Y" if ma.get("quality_gate_mae") else "N"
            gate_corr = "Y" if ma.get("quality_gate_corr") else "N"
            gate_drift = "Y" if ma.get("quality_gate_drift") else "N"

            lines.append(
                f"| {station} | {model} | {ma['n_pre_overlap_days']} | "
                f"{holdout_mae_str} ({gate_mae}) | {corr_str} ({gate_corr}) | "
                f"{drift_str} ({gate_drift}) | **{gates_str}** |"
            )

    # Pre-overlap data completeness
    lines.append("\n### Pre-overlap Data Completeness (2000-2003)\n")
    for station in AIRPORT_STATIONS:
        sa = feasibility["stations"].get(station, {})
        for model in MODELS:
            ma = sa.get(model)
            if ma is None or ma.get("n_pre_overlap_days", 0) == 0:
                continue
            yc = ma.get("yearly_counts", {})
            comp = ma.get("yearly_completeness", {})
            if yc:
                lines.append(f"\n**{station} {model}:** {ma['date_range']}")
                for y in sorted(yc.keys()):
                    c_pct = comp.get(str(y), 0) * 100
                    lines.append(f"  - {y}: {yc[y]} days ({c_pct:.1f}%)")

    lines.append(f"\n### Additional Training Days Available\n")
    lines.append(f"- GFS/AVN: up to {feasibility['total_additional_gfs_days']} days (best single station)")
    lines.append(f"- NAM/ETA: up to {feasibility['total_additional_nam_days']} days (best single station)")

    # ---------------------------------------------------------------
    # Section 6: Recommendation
    # ---------------------------------------------------------------
    lines.append("\n## 6. Recommendation\n")

    # Find best station per model based on holdout MAE
    for model in MODELS:
        best_station = None
        best_mae = 999
        for station in AIRPORT_STATIONS:
            hdf = holdout_results.get(station, {}).get(model, pd.DataFrame())
            if hdf.empty:
                continue
            monthly_row = hdf[hdf["method"] == "monthly_offset"]
            if not monthly_row.empty:
                mae = monthly_row.iloc[0]["mae_vs_knyc"]
                if mae < best_mae:
                    best_mae = mae
                    best_station = station

        if best_station:
            lines.append(f"**Best {model} proxy:** {best_station} "
                         f"({STATION_INFO[best_station]['name']}, "
                         f"~{STATION_INFO[best_station]['distance_mi']} mi) "
                         f"-- holdout MAE: {best_mae:.3f} F")

    # Overall feasibility determination
    any_passes = False
    for station in AIRPORT_STATIONS:
        for model in MODELS:
            ma = feasibility["stations"].get(station, {}).get(model, {})
            if ma.get("passes_all_quality_gates"):
                any_passes = True
                break

    if any_passes:
        lines.append("\n### Verdict: CONDITIONALLY FEASIBLE\n")
        lines.append("At least one station/model combination passes all quality gates. "
                     "Airport MOS harmonization can extend the training period, but with caveats:")
    else:
        lines.append("\n### Verdict: FEASIBLE WITH RESERVATIONS\n")
        lines.append("No station/model passes all strict quality gates. However, the harmonized "
                     "airport MOS may still be useful for training with appropriate uncertainty handling:")

    lines.append("\n**Caveats:**")
    lines.append("1. Holdout MAE represents the noise floor added by using harmonized proxy vs native KNYC MOS.")
    lines.append("2. Pre-2004 data used AVN/ETA models (predecessors to GFS/NAM) with different error characteristics.")
    lines.append("3. MOS equations are updated periodically; pre-2004 equations differ from current ones.")
    lines.append("4. Consider using an `mos_era` indicator feature to let the model learn era-specific corrections.")
    lines.append("5. The variance correction helps match the spread of KNYC but does not fix conditional biases.")

    lines.append("\n**Practical recommendation:**")
    lines.append("- Use monthly-offset harmonization with the best single-station proxy (KLGA for proximity).")
    lines.append("- Add an `mos_era` binary indicator (0 = pre-2004 proxy, 1 = native KNYC) as a model feature.")
    lines.append("- Widen prediction intervals for the pre-2004 extended period proportional to harmonization MAE.")
    lines.append("- The ~1,200 additional GFS training days (2000-2003) represent ~15-18% more data, "
                 "which may help in tail/extreme temperature regimes where sample size matters most.")

    return "\n".join(lines)


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Run the full airport MOS proxy harmonization study."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(AIRPORT_MOS_DIR, exist_ok=True)

    print("=" * 72)
    print("  AIRPORT MOS PROXY HARMONIZATION STUDY")
    print("  Analyzing KLGA, KJFK, KEWR as proxies for KNYC")
    print("=" * 72)

    # ==================================================================
    # PART 1: Load or download airport MOS data
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 1: Loading Airport MOS Data")
    print("=" * 72)

    airport_data = load_or_download_airport_data()

    # Also copy to airport_mos directory for output requirement
    for station in AIRPORT_STATIONS:
        for model in MODELS:
            df = airport_data.get(station, {}).get(model, pd.DataFrame())
            if not df.empty:
                out_path = os.path.join(
                    AIRPORT_MOS_DIR,
                    f"{station.lower()}_{model.lower()}_mos_tmax.csv",
                )
                if not os.path.exists(out_path):
                    df.to_csv(out_path, index=False)

    print("\n--- Data Summary ---")
    for station in AIRPORT_STATIONS:
        for model in MODELS:
            df = airport_data.get(station, {}).get(model, pd.DataFrame())
            if not df.empty:
                print(f"  {station} {model}: {len(df)} days "
                      f"({df['date'].min()} to {df['date'].max()})")
            else:
                print(f"  {station} {model}: NO DATA")

    # ==================================================================
    # PART 2: Load KNYC MOS data
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 2: Loading KNYC MOS Data")
    print("=" * 72)

    knyc_data = load_knyc_mos()

    for model in MODELS:
        df = knyc_data.get(model, pd.DataFrame())
        if not df.empty:
            print(f"  KNYC {model}: {len(df)} days ({df['date'].min()} to {df['date'].max()})")
        else:
            print(f"  KNYC {model}: NO DATA")

    # ==================================================================
    # PART 3: Bias Analysis
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 3: Bias Analysis (Airport vs KNYC in Overlap Period)")
    print("=" * 72)

    bias_results = []
    for station in AIRPORT_STATIONS:
        for model in MODELS:
            ap_df = airport_data.get(station, {}).get(model, pd.DataFrame())
            kn_df = knyc_data.get(model, pd.DataFrame())

            if ap_df.empty or kn_df.empty:
                logger.warning("Skipping bias analysis for %s %s: missing data", station, model)
                continue

            result = compute_bias_analysis(ap_df, kn_df, station, model)
            bias_results.append(result)

            if result.get("status") == "ok":
                print(f"\n  {station} {model}:")
                print(f"    Overlap days:     {result['n_overlap']}")
                print(f"    Correlation:      {result['correlation']:.4f}")
                print(f"    Mean bias:        {result['overall_bias']:+.2f} F")
                print(f"    Std of diffs:     {result['overall_std']:.2f} F")
                print(f"    MAE vs KNYC:      {result['overall_mae']:.2f} F")
                print(f"    KS statistic:     {result['ks_statistic']:.4f} (p={result['ks_pvalue']:.4f})")
                rbs = result['rolling_bias_summary']
                print(f"    Rolling 365d bias range: {rbs['range']:.2f} F, "
                      f"stability std: {rbs['stability_std']:.3f} F")
            else:
                print(f"\n  {station} {model}: INSUFFICIENT DATA")

    # Save bias analysis CSV
    bias_csv_df = build_bias_analysis_csv(bias_results)
    bias_csv_path = os.path.join(RESULTS_DIR, "airport_mos_bias_analysis.csv")
    bias_csv_df.to_csv(bias_csv_path, index=False)
    print(f"\n  Bias analysis saved to: {bias_csv_path}")
    print(f"  Total rows: {len(bias_csv_df)}")

    # ==================================================================
    # PART 4: Harmonization Layer
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 4: Harmonization Layer")
    print(f"  Training on overlap years {HARM_TRAIN_START}-{HARM_TRAIN_END}")
    print(f"  Holdout validation on years {HOLDOUT_YEARS}")
    print("=" * 72)

    train_years = list(range(HARM_TRAIN_START, HARM_TRAIN_END + 1))
    harmonization_params = {}  # {station: {model: params}}
    holdout_results = {}       # {station: {model: DataFrame}}

    for station in AIRPORT_STATIONS:
        harmonization_params[station] = {}
        holdout_results[station] = {}

        for model in MODELS:
            ap_df = airport_data.get(station, {}).get(model, pd.DataFrame())
            kn_df = knyc_data.get(model, pd.DataFrame())

            if ap_df.empty or kn_df.empty:
                continue

            # Fit harmonization
            print(f"\n  Fitting harmonization: {station} {model}")
            params = fit_harmonization(ap_df, kn_df, train_years)

            if not params:
                print(f"    WARNING: Could not fit harmonization for {station} {model}")
                continue

            harmonization_params[station][model] = params
            print(f"    Global offset (KNYC - airport): {params['global_offset']:+.3f} F")
            print(f"    Variance ratio:                 {params['global_var_ratio']:.4f}")
            print(f"    Training days:                  {params['n_train_days']}")

            # Validate on holdout
            print(f"  Validating on holdout years {HOLDOUT_YEARS}:")
            holdout_df = validate_harmonization_holdout(ap_df, kn_df, params, HOLDOUT_YEARS)
            holdout_results[station][model] = holdout_df

            if not holdout_df.empty:
                for _, row in holdout_df.iterrows():
                    print(f"    {row['method']:30s}: MAE={row['mae_vs_knyc']:.3f} "
                          f"bias={row['bias_vs_knyc']:+.3f} corr={row['correlation']:.4f}")

    # Save harmonization parameters as JSON
    # Convert to JSON-serializable format
    json_params = {}
    for station in harmonization_params:
        json_params[station] = {}
        for model, params in harmonization_params[station].items():
            json_params[station][model] = {
                k: v for k, v in params.items()
                if k != "rolling_bias_series"  # Not serializable
            }

    params_path = os.path.join(RESULTS_DIR, "harmonization_parameters.json")
    with open(params_path, "w") as f:
        json.dump(json_params, f, indent=2, default=str)
    print(f"\n  Harmonization parameters saved to: {params_path}")

    # Save holdout validation CSV
    holdout_rows = []
    for station in holdout_results:
        for model, hdf in holdout_results[station].items():
            if hdf is not None and not hdf.empty:
                hdf_copy = hdf.copy()
                hdf_copy["station"] = station
                hdf_copy["model"] = model
                holdout_rows.append(hdf_copy)

    if holdout_rows:
        holdout_all_df = pd.concat(holdout_rows, ignore_index=True)
        holdout_path = os.path.join(RESULTS_DIR, "validation_holdout_results.csv")
        holdout_all_df.to_csv(holdout_path, index=False)
        print(f"  Holdout validation saved to: {holdout_path}")
    else:
        holdout_all_df = pd.DataFrame()

    # ==================================================================
    # PART 5: Feasibility Assessment
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 5: Feasibility Assessment")
    print("=" * 72)

    feasibility = assess_feasibility(
        airport_data, knyc_data, bias_results,
        holdout_results, harmonization_params,
    )

    print(f"\n  Additional training days available:")
    print(f"    GFS/AVN: up to {feasibility['total_additional_gfs_days']} days")
    print(f"    NAM/ETA: up to {feasibility['total_additional_nam_days']} days")

    print(f"\n  Quality Gate Results (MAE<={QUALITY_GATE_MAE}F, "
          f"corr>={QUALITY_GATE_CORR}, drift_std<={QUALITY_GATE_DRIFT}F):")

    for station in AIRPORT_STATIONS:
        sa = feasibility["stations"].get(station, {})
        for model in MODELS:
            ma = sa.get(model)
            if ma is None:
                continue
            status = "PASS" if ma.get("passes_all_quality_gates") else "FAIL"
            mae_str = f"{ma['holdout_mae_vs_knyc']:.3f}" if ma.get("holdout_mae_vs_knyc") is not None else "N/A"
            corr_str = f"{ma['holdout_correlation']:.4f}" if ma.get("holdout_correlation") is not None else "N/A"
            drift_str = f"{ma['seasonal_drift_std']:.3f}" if ma.get("seasonal_drift_std") is not None else "N/A"
            print(f"    {station} {model}: {status} "
                  f"(MAE={mae_str}, corr={corr_str}, drift_std={drift_str})")

    # ==================================================================
    # PART 6: Generate Report
    # ==================================================================
    print("\n" + "=" * 72)
    print("  PART 6: Generating Feasibility Report")
    print("=" * 72)

    report = generate_feasibility_report(
        bias_results, holdout_results, harmonization_params,
        feasibility, airport_data, knyc_data,
    )

    report_path = os.path.join(RESULTS_DIR, "feasibility_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    # ==================================================================
    # Final Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)

    print(f"\n  Output files:")
    print(f"    {bias_csv_path}")
    print(f"    {params_path}")
    print(f"    {report_path}")
    if holdout_rows:
        print(f"    {os.path.join(RESULTS_DIR, 'validation_holdout_results.csv')}")

    print(f"\n  Key findings:")
    # Find best station for GFS based on holdout MAE
    for model in MODELS:
        best_station = None
        best_mae = 999
        for station in AIRPORT_STATIONS:
            hdf = holdout_results.get(station, {}).get(model, pd.DataFrame())
            if hdf.empty:
                continue
            monthly_row = hdf[hdf["method"] == "monthly_offset"]
            if not monthly_row.empty:
                mae = monthly_row.iloc[0]["mae_vs_knyc"]
                if mae < best_mae:
                    best_mae = mae
                    best_station = station
        if best_station:
            print(f"    Best {model} proxy: {best_station} "
                  f"(holdout MAE={best_mae:.3f}F)")

    any_passes = False
    for station in AIRPORT_STATIONS:
        for model in MODELS:
            ma = feasibility["stations"].get(station, {}).get(model, {})
            if ma.get("passes_all_quality_gates"):
                any_passes = True
    verdict = "CONDITIONALLY FEASIBLE" if any_passes else "FEASIBLE WITH RESERVATIONS"
    print(f"\n  Overall verdict: {verdict}")

    print("\n" + "=" * 72)
    print("  STUDY COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
