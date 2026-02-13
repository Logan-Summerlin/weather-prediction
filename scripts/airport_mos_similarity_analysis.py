#!/usr/bin/env python3
"""
Airport MOS Similarity Analysis for KNYC Harmonization

Downloads MOS data from NYC-area airport stations (KJFK, KLGA, KEWR),
analyzes their similarity to KNYC MOS forecasts, and evaluates
harmonization strategies for extending training data back to 2000.

Output:
  - results/airport_mos_analysis/similarity_report.md
  - results/airport_mos_analysis/station_comparison.csv
  - data/mos/{station}_{model}_mos_tmax.csv

Usage:
    python scripts/airport_mos_similarity_analysis.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "airport_mos_analysis")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IEM_MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
AIRPORT_STATIONS = ["KJFK", "KLGA", "KEWR"]
STATION_NAMES = {
    "KJFK": "JFK International (~15 mi)",
    "KLGA": "LaGuardia (~8 mi)",
    "KEWR": "Newark Liberty (~10 mi)",
}

# GFS/AVN data starts earlier for airports than KNYC
GFS_START_YEAR = 2000
NAM_START_YEAR = 2002
END_YEAR = 2026

# KNYC overlap starts ~late 2003 for GFS, ~early 2004 for NAM
OVERLAP_START = 2004
HARMONIZATION_TRAIN_END = 2019
HARMONIZATION_TEST_START = 2020
HARMONIZATION_TEST_END = 2023

MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0
REQUEST_TIMEOUT = 120

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


# ===================================================================
# PART 1: Download Airport MOS Data
# ===================================================================

def download_mos_chunk(station: str, model: str, year: int) -> Optional[str]:
    """Download one year of MOS data from IEM with retries."""
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
            data_lines = len(lines) - 1  # exclude header
            if data_lines <= 0:
                logger.warning("  Header only (no data) for %s %s %d", station, model, year)
                return None

            logger.info(
                "    Got %d data lines (%.1f KB)",
                data_lines, len(text) / 1024,
            )
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


def extract_tmax_from_mos(df: pd.DataFrame) -> pd.DataFrame:
    """Extract daily TMAX forecasts from raw MOS DataFrame.

    Logic:
    - n_x at ftime hour==0 (00Z) = TMAX for the day ending at that time
    - The target date is ftime.date() - 1 day (the daytime max before midnight UTC)
    - For duplicate target dates, keep the latest runtime
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    # Ensure datetime types
    df = df.copy()
    df["runtime"] = pd.to_datetime(df["runtime"])
    df["ftime"] = pd.to_datetime(df["ftime"])

    # Filter to valid n_x at 00Z ftime (TMAX)
    mask = df["n_x"].notna() & (df["ftime"].dt.hour == 0)
    tmax_df = df.loc[mask].copy()

    if tmax_df.empty:
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    # Target date: the calendar day of the high (before midnight UTC)
    tmax_df["target_date"] = (tmax_df["ftime"] - pd.Timedelta(days=1)).dt.date

    # Keep model label (AVN/GFS/ETA/NAM) from data
    tmax_df["model_label"] = tmax_df["model"].astype(str).str.strip()

    # For each target date, prefer the most recent 12Z runtime from day before target
    tmax_df["runtime_date"] = tmax_df["runtime"].dt.date
    tmax_df["runtime_hour"] = tmax_df["runtime"].dt.hour

    records = []
    for target_date, group in tmax_df.groupby("target_date"):
        group = group.copy()
        target_dt = pd.Timestamp(target_date)
        day_before = (target_dt - pd.Timedelta(days=1)).date()

        # Priority: 12Z day-before > 18Z day-before > 06Z day-before > 00Z day-before > 00Z same day
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


def download_airport_mos(station: str, model_query: str, start_year: int) -> pd.DataFrame:
    """Download full MOS history for an airport station.

    Parameters
    ----------
    station : str
        Airport station ID (KJFK, KLGA, KEWR).
    model_query : str
        IEM model name to query ("GFS" or "NAM").
    start_year : int
        First year to download.

    Returns
    -------
    pd.DataFrame
        Columns: date, tmax_f, model_label, runtime
    """
    all_chunks = []

    for year in range(start_year, END_YEAR + 1):
        raw = download_mos_chunk(station, model_query, year)
        if raw is None:
            continue

        try:
            chunk_df = pd.read_csv(StringIO(raw), parse_dates=["runtime", "ftime"])
            # Coerce n_x to numeric (handles empty strings)
            chunk_df["n_x"] = pd.to_numeric(chunk_df["n_x"], errors="coerce")
            all_chunks.append(chunk_df)
        except Exception as exc:
            logger.error("  Parse error %s %s %d: %s", station, model_query, year, exc)

        time.sleep(0.5)  # Be polite to IEM

    if not all_chunks:
        logger.error("No data for %s %s", station, model_query)
        return pd.DataFrame(columns=["date", "tmax_f", "model_label", "runtime"])

    combined = pd.concat(all_chunks, ignore_index=True)
    logger.info(
        "Downloaded %d total rows for %s %s", len(combined), station, model_query
    )

    tmax_df = extract_tmax_from_mos(combined)
    logger.info(
        "Extracted %d TMAX forecasts for %s %s (%s to %s)",
        len(tmax_df), station, model_query,
        tmax_df["date"].min() if not tmax_df.empty else "N/A",
        tmax_df["date"].max() if not tmax_df.empty else "N/A",
    )
    return tmax_df


def download_all_airport_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Download GFS and NAM MOS for all airport stations.

    Returns
    -------
    dict
        {station: {"gfs": DataFrame, "nam": DataFrame}}
    """
    os.makedirs(MOS_DATA_DIR, exist_ok=True)
    data = {}

    for station in AIRPORT_STATIONS:
        logger.info("=" * 60)
        logger.info("Downloading MOS data for %s (%s)", station, STATION_NAMES[station])
        logger.info("=" * 60)

        # GFS (includes AVN for pre-2004)
        logger.info("--- GFS/AVN MOS ---")
        gfs_df = download_airport_mos(station, "GFS", GFS_START_YEAR)
        gfs_path = os.path.join(MOS_DATA_DIR, f"{station.lower()}_gfs_mos_tmax.csv")
        if not gfs_df.empty:
            gfs_df.to_csv(gfs_path, index=False)
            logger.info("Saved %d rows to %s", len(gfs_df), gfs_path)

        # NAM (includes ETA for pre-~2009)
        logger.info("--- NAM/ETA MOS ---")
        nam_df = download_airport_mos(station, "NAM", NAM_START_YEAR)
        nam_path = os.path.join(MOS_DATA_DIR, f"{station.lower()}_nam_mos_tmax.csv")
        if not nam_df.empty:
            nam_df.to_csv(nam_path, index=False)
            logger.info("Saved %d rows to %s", len(nam_df), nam_path)

        data[station] = {"gfs": gfs_df, "nam": nam_df}

    return data


# ===================================================================
# PART 2: Load and Align KNYC MOS
# ===================================================================

def load_knyc_mos() -> Dict[str, pd.DataFrame]:
    """Load existing KNYC MOS data from CSV files."""
    knyc = {}

    # GFS
    gfs_path = os.path.join(MOS_DATA_DIR, "gfs_mos_knyc.csv")
    if os.path.exists(gfs_path):
        gfs_df = pd.read_csv(gfs_path)
        gfs_df["date"] = pd.to_datetime(gfs_df["date"]).dt.date
        # Rename to consistent column
        if "gfs_mos_tmax_f" in gfs_df.columns:
            gfs_df = gfs_df.rename(columns={"gfs_mos_tmax_f": "tmax_f"})
        elif "tmax_forecast_f" in gfs_df.columns:
            gfs_df = gfs_df.rename(columns={"tmax_forecast_f": "tmax_f"})
        knyc["gfs"] = gfs_df
        logger.info("Loaded KNYC GFS MOS: %d rows (%s to %s)",
                     len(gfs_df), gfs_df["date"].min(), gfs_df["date"].max())
    else:
        logger.warning("KNYC GFS MOS file not found: %s", gfs_path)
        knyc["gfs"] = pd.DataFrame(columns=["date", "tmax_f"])

    # NAM
    nam_path = os.path.join(MOS_DATA_DIR, "nam_mos_knyc.csv")
    if os.path.exists(nam_path):
        nam_df = pd.read_csv(nam_path)
        nam_df["date"] = pd.to_datetime(nam_df["date"]).dt.date
        if "nam_mos_tmax_f" in nam_df.columns:
            nam_df = nam_df.rename(columns={"nam_mos_tmax_f": "tmax_f"})
        elif "tmax_forecast_f" in nam_df.columns:
            nam_df = nam_df.rename(columns={"tmax_forecast_f": "tmax_f"})
        knyc["nam"] = nam_df
        logger.info("Loaded KNYC NAM MOS: %d rows (%s to %s)",
                     len(nam_df), nam_df["date"].min(), nam_df["date"].max())
    else:
        logger.warning("KNYC NAM MOS file not found: %s", nam_path)
        knyc["nam"] = pd.DataFrame(columns=["date", "tmax_f"])

    return knyc


def load_central_park_actual() -> pd.DataFrame:
    """Load Central Park actual TMAX observations."""
    path = os.path.join(PROJECT_ROOT, "data", "central_park_tmax_full_history.csv")
    if not os.path.exists(path):
        logger.error("Central Park TMAX file not found: %s", path)
        return pd.DataFrame(columns=["date", "tmax_f"])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info("Loaded Central Park actual TMAX: %d rows (%s to %s)",
                 len(df), df["date"].min(), df["date"].max())
    return df


# ===================================================================
# PART 3: Similarity Analysis
# ===================================================================

def get_season(month: int) -> str:
    """Map month to season abbreviation."""
    for season, months in SEASONS.items():
        if month in months:
            return season
    return "UNK"


def compute_similarity_metrics(
    airport_df: pd.DataFrame,
    knyc_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    station: str,
    model: str,
) -> Dict:
    """Compute similarity metrics between airport and KNYC MOS for overlap period.

    Returns a dictionary of metric results.
    """
    # Merge airport and KNYC on date
    ap = airport_df[["date", "tmax_f"]].rename(columns={"tmax_f": "airport_tmax"})
    kn = knyc_df[["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})

    merged = pd.merge(ap, kn, on="date", how="inner")

    # Also merge actual CP data
    act = actual_df[["date", "tmax_f"]].rename(columns={"tmax_f": "actual_tmax"})
    merged = pd.merge(merged, act, on="date", how="inner")

    # Filter to overlap period only (2004-2026)
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged = merged[
        (merged["date_dt"].dt.year >= OVERLAP_START) &
        (merged["date_dt"].dt.year <= END_YEAR)
    ].copy()

    if len(merged) < 100:
        logger.warning("Too few overlap points for %s %s: %d", station, model, len(merged))
        return {"station": station, "model": model, "n_overlap": len(merged)}

    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)
    merged["diff"] = merged["airport_tmax"] - merged["knyc_tmax"]
    merged["airport_error"] = merged["airport_tmax"] - merged["actual_tmax"]
    merged["knyc_error"] = merged["knyc_tmax"] - merged["actual_tmax"]

    # 1. Correlation
    corr = merged["airport_tmax"].corr(merged["knyc_tmax"])

    # 2. Mean bias (airport - KNYC) overall and by season
    mean_bias = merged["diff"].mean()
    seasonal_bias = merged.groupby("season")["diff"].mean().to_dict()

    # 3. MAE between airport and KNYC
    mae_ap_knyc = merged["diff"].abs().mean()

    # 4. RMSE between airport and KNYC
    rmse_ap_knyc = np.sqrt((merged["diff"] ** 2).mean())

    # 5. Forecast skill: MAE vs actual
    airport_mae_vs_actual = merged["airport_error"].abs().mean()
    knyc_mae_vs_actual = merged["knyc_error"].abs().mean()

    # 6. Bias stability: rolling 365-day bias
    merged_sorted = merged.sort_values("date_dt")
    rolling_bias = merged_sorted.set_index("date_dt")["diff"].rolling("365D").mean()
    bias_stability_std = rolling_bias.dropna().std()
    bias_stability_range = rolling_bias.dropna().max() - rolling_bias.dropna().min()

    # 7. KS test: error distributions
    ks_stat, ks_pval = stats.ks_2samp(
        merged["airport_error"].dropna(),
        merged["knyc_error"].dropna(),
    )

    # 8. Monthly bias
    monthly_bias = merged.groupby("month")["diff"].mean().to_dict()
    monthly_bias_std = merged.groupby("month")["diff"].std().to_dict()

    # Seasonal MAE between airport and KNYC
    seasonal_mae = merged.groupby("season")["diff"].apply(lambda x: x.abs().mean()).to_dict()

    results = {
        "station": station,
        "model": model,
        "n_overlap": len(merged),
        "correlation": corr,
        "mean_bias": mean_bias,
        "mae_ap_vs_knyc": mae_ap_knyc,
        "rmse_ap_vs_knyc": rmse_ap_knyc,
        "airport_mae_vs_actual": airport_mae_vs_actual,
        "knyc_mae_vs_actual": knyc_mae_vs_actual,
        "skill_difference": airport_mae_vs_actual - knyc_mae_vs_actual,
        "bias_stability_std": bias_stability_std,
        "bias_stability_range": bias_stability_range,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pval,
        "seasonal_bias": seasonal_bias,
        "seasonal_mae": seasonal_mae,
        "monthly_bias": monthly_bias,
        "monthly_bias_std": monthly_bias_std,
        "rolling_bias_series": rolling_bias,  # For reporting
    }

    return results


# ===================================================================
# PART 4: Harmonization Strategy Evaluation
# ===================================================================

def evaluate_harmonization(
    airport_df: pd.DataFrame,
    knyc_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    station: str,
    model: str,
) -> Dict:
    """Evaluate harmonization strategies for the best proxy station.

    Train on 2004-2019, test on 2020-2023.
    """
    # Merge all three
    ap = airport_df[["date", "tmax_f"]].rename(columns={"tmax_f": "airport_tmax"})
    kn = knyc_df[["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})
    act = actual_df[["date", "tmax_f"]].rename(columns={"tmax_f": "actual_tmax"})

    merged = pd.merge(ap, kn, on="date", how="inner")
    merged = pd.merge(merged, act, on="date", how="inner")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["year"] = merged["date_dt"].dt.year
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)
    merged["diff"] = merged["airport_tmax"] - merged["knyc_tmax"]

    # Split
    train = merged[merged["year"] <= HARMONIZATION_TRAIN_END].copy()
    test = merged[
        (merged["year"] >= HARMONIZATION_TEST_START) &
        (merged["year"] <= HARMONIZATION_TEST_END)
    ].copy()

    if len(train) < 365 or len(test) < 100:
        logger.warning("Insufficient data for harmonization eval: train=%d, test=%d",
                       len(train), len(test))
        return {}

    results = {
        "station": station,
        "model": model,
        "n_train": len(train),
        "n_test": len(test),
    }

    # Method 0: Raw (no harmonization) — airport vs KNYC
    raw_mae = test["diff"].abs().mean()
    results["raw_mae_vs_knyc"] = raw_mae

    # Raw airport vs actual
    raw_airport_mae_actual = (test["airport_tmax"] - test["actual_tmax"]).abs().mean()
    results["raw_airport_mae_vs_actual"] = raw_airport_mae_actual

    # KNYC vs actual (baseline reference)
    knyc_mae_actual = (test["knyc_tmax"] - test["actual_tmax"]).abs().mean()
    results["knyc_mae_vs_actual"] = knyc_mae_actual

    # Method 1: Constant bias offset
    const_bias = train["diff"].mean()
    test["harmonized_const"] = test["airport_tmax"] - const_bias
    const_mae_knyc = (test["harmonized_const"] - test["knyc_tmax"]).abs().mean()
    const_mae_actual = (test["harmonized_const"] - test["actual_tmax"]).abs().mean()
    results["const_offset"] = const_bias
    results["const_mae_vs_knyc"] = const_mae_knyc
    results["const_mae_vs_actual"] = const_mae_actual

    # Method 2: Seasonal bias offset
    seasonal_offsets = train.groupby("season")["diff"].mean().to_dict()
    test["seasonal_offset"] = test["season"].map(seasonal_offsets)
    test["harmonized_seasonal"] = test["airport_tmax"] - test["seasonal_offset"]
    seasonal_mae_knyc = (test["harmonized_seasonal"] - test["knyc_tmax"]).abs().mean()
    seasonal_mae_actual = (test["harmonized_seasonal"] - test["actual_tmax"]).abs().mean()
    results["seasonal_offsets"] = seasonal_offsets
    results["seasonal_mae_vs_knyc"] = seasonal_mae_knyc
    results["seasonal_mae_vs_actual"] = seasonal_mae_actual

    # Method 3: Monthly bias offset
    monthly_offsets = train.groupby("month")["diff"].mean().to_dict()
    test["monthly_offset"] = test["month"].map(monthly_offsets)
    test["harmonized_monthly"] = test["airport_tmax"] - test["monthly_offset"]
    monthly_mae_knyc = (test["harmonized_monthly"] - test["knyc_tmax"]).abs().mean()
    monthly_mae_actual = (test["harmonized_monthly"] - test["actual_tmax"]).abs().mean()
    results["monthly_offsets"] = monthly_offsets
    results["monthly_mae_vs_knyc"] = monthly_mae_knyc
    results["monthly_mae_vs_actual"] = monthly_mae_actual

    return results


def evaluate_multistation_average(
    airport_data: Dict[str, Dict[str, pd.DataFrame]],
    knyc_data: Dict[str, pd.DataFrame],
    actual_df: pd.DataFrame,
    model: str,
) -> Dict:
    """Evaluate using an average of multiple airport stations."""
    # Collect airport data for this model
    dfs_list = []
    for station in AIRPORT_STATIONS:
        if station in airport_data and model in airport_data[station]:
            df = airport_data[station][model][["date", "tmax_f"]].copy()
            df = df.rename(columns={"tmax_f": f"tmax_{station}"})
            dfs_list.append(df)

    if len(dfs_list) < 2:
        return {}

    # Merge all airports
    merged = dfs_list[0]
    for df in dfs_list[1:]:
        merged = pd.merge(merged, df, on="date", how="inner")

    # Compute average
    tmax_cols = [c for c in merged.columns if c.startswith("tmax_K")]
    merged["airport_avg"] = merged[tmax_cols].mean(axis=1)

    # Merge KNYC and actual
    kn = knyc_data[model][["date", "tmax_f"]].rename(columns={"tmax_f": "knyc_tmax"})
    act = actual_df[["date", "tmax_f"]].rename(columns={"tmax_f": "actual_tmax"})
    merged = pd.merge(merged, kn, on="date", how="inner")
    merged = pd.merge(merged, act, on="date", how="inner")
    merged["date_dt"] = pd.to_datetime(merged["date"])
    merged["year"] = merged["date_dt"].dt.year
    merged["month"] = merged["date_dt"].dt.month
    merged["season"] = merged["month"].apply(get_season)

    # Split
    train = merged[merged["year"] <= HARMONIZATION_TRAIN_END].copy()
    test = merged[
        (merged["year"] >= HARMONIZATION_TEST_START) &
        (merged["year"] <= HARMONIZATION_TEST_END)
    ].copy()

    if len(test) < 100:
        return {}

    train["diff"] = train["airport_avg"] - train["knyc_tmax"]
    test["diff"] = test["airport_avg"] - test["knyc_tmax"]

    # Raw average vs KNYC
    raw_mae = test["diff"].abs().mean()

    # Constant harmonization
    const_bias = train["diff"].mean()
    test["harmonized"] = test["airport_avg"] - const_bias
    const_mae_knyc = (test["harmonized"] - test["knyc_tmax"]).abs().mean()
    const_mae_actual = (test["harmonized"] - test["actual_tmax"]).abs().mean()

    # Monthly harmonization
    monthly_offsets = train.groupby("month")["diff"].mean().to_dict()
    test["monthly_offset"] = test["month"].map(monthly_offsets)
    test["harmonized_monthly"] = test["airport_avg"] - test["monthly_offset"]
    monthly_mae_knyc = (test["harmonized_monthly"] - test["knyc_tmax"]).abs().mean()
    monthly_mae_actual = (test["harmonized_monthly"] - test["actual_tmax"]).abs().mean()

    return {
        "method": "multi_station_average",
        "model": model,
        "stations": AIRPORT_STATIONS,
        "n_train": len(train),
        "n_test": len(test),
        "raw_mae_vs_knyc": raw_mae,
        "const_offset": const_bias,
        "const_mae_vs_knyc": const_mae_knyc,
        "const_mae_vs_actual": const_mae_actual,
        "monthly_mae_vs_knyc": monthly_mae_knyc,
        "monthly_mae_vs_actual": monthly_mae_actual,
    }


# ===================================================================
# PART 5: Pre-Overlap Data Quality Check
# ===================================================================

def check_pre_overlap_quality(
    airport_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    station: str,
    model: str,
) -> Dict:
    """Check data quality for the pre-KNYC-overlap period."""
    ap = airport_df[["date", "tmax_f", "model_label"]].copy()
    ap["date_dt"] = pd.to_datetime(ap["date"])
    ap["year"] = ap["date_dt"].dt.year

    # Pre-overlap: before KNYC started
    pre = ap[ap["year"] < OVERLAP_START].copy()
    overlap = ap[ap["year"] >= OVERLAP_START].copy()

    # Merge with actuals
    act = actual_df[["date", "tmax_f"]].rename(columns={"tmax_f": "actual_tmax"})
    pre_merged = pd.merge(pre, act, on="date", how="inner")
    overlap_merged = pd.merge(overlap, act, on="date", how="inner")

    results = {
        "station": station,
        "model": model,
    }

    # Availability by year
    yearly_counts = pre.groupby("year").size().to_dict()
    results["yearly_availability"] = yearly_counts
    results["total_pre_overlap_days"] = len(pre)

    # Model labels by year
    yearly_labels = pre.groupby("year")["model_label"].apply(
        lambda x: x.value_counts().to_dict()
    ).to_dict()
    results["yearly_model_labels"] = yearly_labels

    # MAE vs actual for pre-overlap
    if len(pre_merged) > 0:
        pre_merged["error"] = pre_merged["tmax_f"] - pre_merged["actual_tmax"]
        results["pre_mae_vs_actual"] = pre_merged["error"].abs().mean()
        results["pre_bias_vs_actual"] = pre_merged["error"].mean()
        results["pre_rmse_vs_actual"] = np.sqrt((pre_merged["error"] ** 2).mean())
        results["pre_n_matched"] = len(pre_merged)
    else:
        results["pre_mae_vs_actual"] = np.nan
        results["pre_bias_vs_actual"] = np.nan

    # MAE vs actual for overlap (for comparison)
    if len(overlap_merged) > 0:
        overlap_merged["error"] = overlap_merged["tmax_f"] - overlap_merged["actual_tmax"]
        results["overlap_mae_vs_actual"] = overlap_merged["error"].abs().mean()
        results["overlap_bias_vs_actual"] = overlap_merged["error"].mean()
        results["overlap_n_matched"] = len(overlap_merged)
    else:
        results["overlap_mae_vs_actual"] = np.nan
        results["overlap_bias_vs_actual"] = np.nan

    return results


# ===================================================================
# PART 6: Report Generation
# ===================================================================

def generate_report(
    similarity_results: List[Dict],
    harmonization_results: List[Dict],
    multistation_results: List[Dict],
    pre_overlap_results: List[Dict],
    airport_data: Dict[str, Dict[str, pd.DataFrame]],
    knyc_data: Dict[str, pd.DataFrame],
) -> str:
    """Generate the full markdown report."""
    lines = []
    lines.append("# Airport MOS Similarity Analysis Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n**Purpose:** Evaluate NYC-area airport MOS stations as proxies for KNYC "
                 f"to extend MOS training data back to 2000 (GFS/AVN) and 2002 (NAM/ETA).\n")

    # ---------------------------------------------------------------
    # Data Summary
    # ---------------------------------------------------------------
    lines.append("## 1. Data Summary\n")
    lines.append("### Airport MOS Data Availability\n")
    lines.append("| Station | Model | Date Range | Total Days | Model Labels |")
    lines.append("|---------|-------|------------|------------|--------------|")

    for station in AIRPORT_STATIONS:
        for model_key in ["gfs", "nam"]:
            if station in airport_data and model_key in airport_data[station]:
                df = airport_data[station][model_key]
                if not df.empty:
                    labels = df["model_label"].value_counts().to_dict()
                    label_str = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
                    lines.append(
                        f"| {station} | {model_key.upper()} | "
                        f"{df['date'].min()} to {df['date'].max()} | "
                        f"{len(df)} | {label_str} |"
                    )
                else:
                    lines.append(f"| {station} | {model_key.upper()} | No data | 0 | - |")

    lines.append("\n### KNYC MOS Data Availability\n")
    lines.append("| Model | Date Range | Total Days |")
    lines.append("|-------|------------|------------|")
    for model_key in ["gfs", "nam"]:
        if model_key in knyc_data and not knyc_data[model_key].empty:
            df = knyc_data[model_key]
            lines.append(f"| {model_key.upper()} | {df['date'].min()} to {df['date'].max()} | {len(df)} |")

    # ---------------------------------------------------------------
    # Similarity Metrics
    # ---------------------------------------------------------------
    lines.append("\n## 2. Similarity Analysis (Overlap Period 2004-2026)\n")

    for model_key in ["gfs", "nam"]:
        model_results = [r for r in similarity_results if r.get("model") == model_key]
        if not model_results:
            continue

        lines.append(f"\n### {model_key.upper()} MOS\n")
        lines.append("| Metric | " + " | ".join(r["station"] for r in model_results) + " |")
        lines.append("|--------|" + "|".join(["-----" for _ in model_results]) + "|")

        metrics = [
            ("N overlap days", "n_overlap", "d"),
            ("Correlation", "correlation", ".4f"),
            ("Mean bias (AP-KNYC)", "mean_bias", ".2f"),
            ("MAE vs KNYC", "mae_ap_vs_knyc", ".2f"),
            ("RMSE vs KNYC", "rmse_ap_vs_knyc", ".2f"),
            ("Airport MAE vs actual", "airport_mae_vs_actual", ".2f"),
            ("KNYC MAE vs actual", "knyc_mae_vs_actual", ".2f"),
            ("Skill diff (AP-KNYC)", "skill_difference", ".2f"),
            ("Bias stability (365d std)", "bias_stability_std", ".3f"),
            ("Bias stability (range)", "bias_stability_range", ".2f"),
            ("KS statistic", "ks_statistic", ".4f"),
            ("KS p-value", "ks_pvalue", ".4f"),
        ]

        for label, key, fmt in metrics:
            vals = []
            for r in model_results:
                v = r.get(key, "N/A")
                if isinstance(v, (int, float)) and not np.isnan(v):
                    vals.append(f"{v:{fmt}}")
                else:
                    vals.append("N/A")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        # Seasonal bias table
        lines.append(f"\n#### Seasonal Bias (Airport - KNYC, {model_key.upper()})\n")
        lines.append("| Season | " + " | ".join(r["station"] for r in model_results) + " |")
        lines.append("|--------|" + "|".join(["-----" for _ in model_results]) + "|")
        for season in ["DJF", "MAM", "JJA", "SON"]:
            vals = []
            for r in model_results:
                sb = r.get("seasonal_bias", {})
                v = sb.get(season, np.nan)
                vals.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
            lines.append(f"| {season} | " + " | ".join(vals) + " |")

        # Monthly bias table
        lines.append(f"\n#### Monthly Bias (Airport - KNYC, {model_key.upper()})\n")
        lines.append("| Month | " + " | ".join(r["station"] for r in model_results) + " |")
        lines.append("|-------|" + "|".join(["-----" for _ in model_results]) + "|")
        for m in range(1, 13):
            vals = []
            for r in model_results:
                mb = r.get("monthly_bias", {})
                v = mb.get(m, np.nan)
                vals.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
            lines.append(f"| {m:02d} | " + " | ".join(vals) + " |")

    # ---------------------------------------------------------------
    # Harmonization Evaluation
    # ---------------------------------------------------------------
    lines.append("\n## 3. Harmonization Strategy Evaluation\n")
    lines.append(f"Train period: 2004-{HARMONIZATION_TRAIN_END}, "
                 f"Test period: {HARMONIZATION_TEST_START}-{HARMONIZATION_TEST_END}\n")

    for model_key in ["gfs", "nam"]:
        model_harm = [r for r in harmonization_results if r.get("model") == model_key]
        if not model_harm:
            continue

        lines.append(f"\n### {model_key.upper()} MOS Harmonization\n")

        for r in model_harm:
            station = r["station"]
            lines.append(f"\n#### {station} ({STATION_NAMES.get(station, '')})\n")
            lines.append(f"- Train: {r.get('n_train', 'N/A')} days, Test: {r.get('n_test', 'N/A')} days\n")
            lines.append("| Method | MAE vs KNYC | MAE vs Actual | Notes |")
            lines.append("|--------|-------------|---------------|-------|")

            lines.append(f"| Raw (no harmonization) | {r.get('raw_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('raw_airport_mae_vs_actual', 'N/A'):.3f} | Baseline |")
            lines.append(f"| Constant offset ({r.get('const_offset', 0):.2f}F) | "
                         f"{r.get('const_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('const_mae_vs_actual', 'N/A'):.3f} | Single bias correction |")
            lines.append(f"| Seasonal offset | {r.get('seasonal_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('seasonal_mae_vs_actual', 'N/A'):.3f} | 4 seasonal offsets |")
            lines.append(f"| Monthly offset | {r.get('monthly_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('monthly_mae_vs_actual', 'N/A'):.3f} | 12 monthly offsets |")
            lines.append(f"| KNYC direct | - | "
                         f"{r.get('knyc_mae_vs_actual', 'N/A'):.3f} | Reference |")

            # Print seasonal offsets if available
            so = r.get("seasonal_offsets", {})
            if so:
                lines.append(f"\nSeasonal offsets: " + ", ".join(
                    f"{k}: {v:.2f}F" for k, v in sorted(so.items())
                ))

        # Multi-station results
        ms = [r for r in multistation_results if r.get("model") == model_key]
        if ms:
            r = ms[0]
            lines.append(f"\n#### Multi-Station Average ({', '.join(r.get('stations', []))})\n")
            lines.append("| Method | MAE vs KNYC | MAE vs Actual |")
            lines.append("|--------|-------------|---------------|")
            lines.append(f"| Raw average | {r.get('raw_mae_vs_knyc', 'N/A'):.3f} | - |")
            lines.append(f"| Constant offset ({r.get('const_offset', 0):.2f}F) | "
                         f"{r.get('const_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('const_mae_vs_actual', 'N/A'):.3f} |")
            lines.append(f"| Monthly offset | {r.get('monthly_mae_vs_knyc', 'N/A'):.3f} | "
                         f"{r.get('monthly_mae_vs_actual', 'N/A'):.3f} |")

    # ---------------------------------------------------------------
    # Pre-Overlap Quality
    # ---------------------------------------------------------------
    lines.append("\n## 4. Pre-Overlap Data Quality (2000-2003)\n")

    for r in pre_overlap_results:
        station = r["station"]
        model = r["model"]
        lines.append(f"\n### {station} {model.upper()}\n")

        # Yearly availability
        ya = r.get("yearly_availability", {})
        yl = r.get("yearly_model_labels", {})
        if ya:
            lines.append("| Year | Days Available | Model Label |")
            lines.append("|------|---------------|-------------|")
            for year in sorted(ya.keys()):
                labels = yl.get(year, {})
                label_str = ", ".join(f"{k}: {v}" for k, v in labels.items())
                lines.append(f"| {year} | {ya[year]} | {label_str} |")
            lines.append(f"\n**Total pre-overlap days:** {r.get('total_pre_overlap_days', 0)}")

        # Quality comparison
        lines.append("\n| Period | MAE vs Actual | Bias vs Actual | N Days |")
        lines.append("|--------|---------------|----------------|--------|")
        lines.append(
            f"| Pre-overlap ({min(ya.keys()) if ya else '?'}-2003) | "
            f"{r.get('pre_mae_vs_actual', 'N/A'):.2f} | "
            f"{r.get('pre_bias_vs_actual', 'N/A'):.2f} | "
            f"{r.get('pre_n_matched', 0)} |"
        ) if ya else None
        lines.append(
            f"| Overlap (2004+) | "
            f"{r.get('overlap_mae_vs_actual', 'N/A'):.2f} | "
            f"{r.get('overlap_bias_vs_actual', 'N/A'):.2f} | "
            f"{r.get('overlap_n_matched', 0)} |"
        )

    # ---------------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------------
    lines.append("\n## 5. Recommendation\n")

    # Find best single station by lowest MAE vs KNYC for GFS
    gfs_harm = [r for r in harmonization_results if r.get("model") == "gfs" and r.get("monthly_mae_vs_knyc")]
    if gfs_harm:
        best = min(gfs_harm, key=lambda x: x.get("monthly_mae_vs_knyc", 999))
        best_station = best["station"]
        lines.append(f"### Best Single Proxy: **{best_station}** ({STATION_NAMES.get(best_station, '')})\n")

        # Summary
        gfs_sim = [r for r in similarity_results if r.get("station") == best_station and r.get("model") == "gfs"]
        if gfs_sim:
            r = gfs_sim[0]
            lines.append(f"- **Correlation with KNYC:** {r.get('correlation', 'N/A'):.4f}")
            lines.append(f"- **Mean bias:** {r.get('mean_bias', 'N/A'):.2f}F")
            lines.append(f"- **MAE vs KNYC:** {r.get('mae_ap_vs_knyc', 'N/A'):.2f}F")

        lines.append(f"- **Best harmonization method:** Monthly offset "
                     f"(MAE vs KNYC on test: {best.get('monthly_mae_vs_knyc', 'N/A'):.3f}F)")

    # Multi-station recommendation
    gfs_ms = [r for r in multistation_results if r.get("model") == "gfs"]
    if gfs_ms:
        r = gfs_ms[0]
        lines.append(f"\n### Multi-Station Average")
        lines.append(f"- Monthly offset MAE vs KNYC: {r.get('monthly_mae_vs_knyc', 'N/A'):.3f}F")
        lines.append(f"- Monthly offset MAE vs actual: {r.get('monthly_mae_vs_actual', 'N/A'):.3f}F")

    # Extended date range recommendation
    lines.append("\n### Recommended Extended Date Ranges\n")
    lines.append("| Model | Current Start | Extended Start | Added Years |")
    lines.append("|-------|---------------|----------------|-------------|")
    lines.append("| GFS/AVN | 2004-01 | 2000-06* | ~3.5 years |")
    lines.append("| NAM/ETA | 2004-02 | 2002-06* | ~1.5 years |")
    lines.append("\n*Exact start depends on data availability at each airport station.")

    # Caveats
    lines.append("\n### Data Quality Concerns & Caveats\n")
    lines.append("1. **Model transitions:** AVN→GFS transition (2003/2004) and ETA→NAM transition "
                 "(~2005-2009) may introduce subtle systematic shifts.")
    lines.append("2. **MOS equation updates:** MOS equations are re-derived periodically, which can "
                 "cause discontinuities.")
    lines.append("3. **Airport microclimate:** Airport stations have different local climates than "
                 "Central Park (urban heat island, proximity to water, runway effects).")
    lines.append("4. **Pre-overlap verification:** The harmonization offsets are trained on "
                 "2004-2019 but applied to 2000-2003 data that used different NWP models.")
    lines.append("5. **Recommendation:** Use monthly harmonization with the best proxy station. "
                 "Consider adding a small noise term or widening prediction intervals for the "
                 "pre-2004 extended period to account for increased uncertainty.")

    return "\n".join(lines)


def save_comparison_csv(similarity_results: List[Dict], harmonization_results: List[Dict]):
    """Save station comparison metrics as CSV."""
    rows = []
    for r in similarity_results:
        row = {
            "station": r.get("station"),
            "model": r.get("model"),
            "n_overlap": r.get("n_overlap"),
            "correlation": r.get("correlation"),
            "mean_bias": r.get("mean_bias"),
            "mae_vs_knyc": r.get("mae_ap_vs_knyc"),
            "rmse_vs_knyc": r.get("rmse_ap_vs_knyc"),
            "airport_mae_vs_actual": r.get("airport_mae_vs_actual"),
            "knyc_mae_vs_actual": r.get("knyc_mae_vs_actual"),
            "skill_difference": r.get("skill_difference"),
            "bias_stability_std": r.get("bias_stability_std"),
            "ks_statistic": r.get("ks_statistic"),
            "ks_pvalue": r.get("ks_pvalue"),
        }
        # Add seasonal biases
        for season in ["DJF", "MAM", "JJA", "SON"]:
            row[f"bias_{season}"] = r.get("seasonal_bias", {}).get(season)
        rows.append(row)

    # Add harmonization info
    for r in harmonization_results:
        # Find matching row
        for row in rows:
            if row["station"] == r.get("station") and row["model"] == r.get("model"):
                row["harm_raw_mae"] = r.get("raw_mae_vs_knyc")
                row["harm_const_mae"] = r.get("const_mae_vs_knyc")
                row["harm_seasonal_mae"] = r.get("seasonal_mae_vs_knyc")
                row["harm_monthly_mae"] = r.get("monthly_mae_vs_knyc")
                row["const_offset"] = r.get("const_offset")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "station_comparison.csv")
    df.to_csv(path, index=False)
    logger.info("Saved comparison CSV to %s", path)
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Run the full airport MOS similarity analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MOS_DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("  AIRPORT MOS SIMILARITY ANALYSIS")
    print("  Analyzing KJFK, KLGA, KEWR as proxies for KNYC")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Part 1: Download airport MOS data
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 1: Downloading Airport MOS Data")
    print("=" * 70)

    airport_data = download_all_airport_data()

    # Print download summary
    print("\n--- Download Summary ---")
    for station in AIRPORT_STATIONS:
        for model_key in ["gfs", "nam"]:
            df = airport_data.get(station, {}).get(model_key, pd.DataFrame())
            if not df.empty:
                print(f"  {station} {model_key.upper()}: {len(df)} days "
                      f"({df['date'].min()} to {df['date'].max()})")
            else:
                print(f"  {station} {model_key.upper()}: NO DATA")

    # ------------------------------------------------------------------
    # Part 2: Load KNYC and Central Park actual
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 2: Loading KNYC MOS and Central Park Actual Data")
    print("=" * 70)

    knyc_data = load_knyc_mos()
    actual_df = load_central_park_actual()

    # ------------------------------------------------------------------
    # Part 3: Similarity Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 3: Similarity Analysis (Overlap Period)")
    print("=" * 70)

    similarity_results = []
    for station in AIRPORT_STATIONS:
        for model_key in ["gfs", "nam"]:
            ap_df = airport_data.get(station, {}).get(model_key, pd.DataFrame())
            kn_df = knyc_data.get(model_key, pd.DataFrame())

            if ap_df.empty or kn_df.empty:
                logger.warning("Skipping %s %s: missing data", station, model_key)
                continue

            logger.info("Computing similarity: %s %s", station, model_key.upper())
            metrics = compute_similarity_metrics(ap_df, kn_df, actual_df, station, model_key)
            similarity_results.append(metrics)

            # Print key metrics
            print(f"\n  {station} {model_key.upper()}:")
            print(f"    Overlap days: {metrics.get('n_overlap', 0)}")
            print(f"    Correlation:  {metrics.get('correlation', 'N/A'):.4f}"
                  if isinstance(metrics.get('correlation'), float) else "    Correlation: N/A")
            print(f"    Mean bias:    {metrics.get('mean_bias', 'N/A'):.2f}F"
                  if isinstance(metrics.get('mean_bias'), float) else "    Mean bias: N/A")
            print(f"    MAE vs KNYC:  {metrics.get('mae_ap_vs_knyc', 'N/A'):.2f}F"
                  if isinstance(metrics.get('mae_ap_vs_knyc'), float) else "    MAE vs KNYC: N/A")

    # ------------------------------------------------------------------
    # Part 4: Harmonization Strategy Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 4: Harmonization Strategy Evaluation")
    print("=" * 70)

    harmonization_results = []
    for station in AIRPORT_STATIONS:
        for model_key in ["gfs", "nam"]:
            ap_df = airport_data.get(station, {}).get(model_key, pd.DataFrame())
            kn_df = knyc_data.get(model_key, pd.DataFrame())

            if ap_df.empty or kn_df.empty:
                continue

            logger.info("Evaluating harmonization: %s %s", station, model_key.upper())
            harm = evaluate_harmonization(ap_df, kn_df, actual_df, station, model_key)
            if harm:
                harmonization_results.append(harm)

                print(f"\n  {station} {model_key.upper()} harmonization (test period {HARMONIZATION_TEST_START}-{HARMONIZATION_TEST_END}):")
                print(f"    Raw MAE vs KNYC:      {harm.get('raw_mae_vs_knyc', 'N/A'):.3f}F")
                print(f"    Constant offset MAE:  {harm.get('const_mae_vs_knyc', 'N/A'):.3f}F (offset={harm.get('const_offset', 0):.2f}F)")
                print(f"    Seasonal offset MAE:  {harm.get('seasonal_mae_vs_knyc', 'N/A'):.3f}F")
                print(f"    Monthly offset MAE:   {harm.get('monthly_mae_vs_knyc', 'N/A'):.3f}F")
                print(f"    KNYC MAE vs actual:   {harm.get('knyc_mae_vs_actual', 'N/A'):.3f}F")

    # Multi-station average
    multistation_results = []
    for model_key in ["gfs", "nam"]:
        if model_key in knyc_data and not knyc_data[model_key].empty:
            ms = evaluate_multistation_average(airport_data, knyc_data, actual_df, model_key)
            if ms:
                multistation_results.append(ms)
                print(f"\n  Multi-station average {model_key.upper()}:")
                print(f"    Raw MAE vs KNYC:      {ms.get('raw_mae_vs_knyc', 'N/A'):.3f}F")
                print(f"    Const offset MAE:     {ms.get('const_mae_vs_knyc', 'N/A'):.3f}F")
                print(f"    Monthly offset MAE:   {ms.get('monthly_mae_vs_knyc', 'N/A'):.3f}F")

    # ------------------------------------------------------------------
    # Part 5: Pre-Overlap Data Quality Check
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 5: Pre-Overlap Data Quality Check (2000-2003)")
    print("=" * 70)

    pre_overlap_results = []
    for station in AIRPORT_STATIONS:
        for model_key in ["gfs", "nam"]:
            ap_df = airport_data.get(station, {}).get(model_key, pd.DataFrame())
            if ap_df.empty:
                continue

            # Only if there's pre-overlap data
            ap_dates = pd.to_datetime(ap_df["date"])
            if ap_dates.min().year >= OVERLAP_START:
                continue

            logger.info("Checking pre-overlap quality: %s %s", station, model_key.upper())
            pq = check_pre_overlap_quality(ap_df, actual_df, station, model_key)
            pre_overlap_results.append(pq)

            print(f"\n  {station} {model_key.upper()} pre-overlap:")
            ya = pq.get("yearly_availability", {})
            for year in sorted(ya.keys()):
                yl = pq.get("yearly_model_labels", {}).get(year, {})
                label_str = ", ".join(f"{k}: {v}" for k, v in yl.items())
                print(f"    {year}: {ya[year]} days ({label_str})")
            if not np.isnan(pq.get("pre_mae_vs_actual", np.nan)):
                print(f"    Pre-overlap MAE vs actual:  {pq['pre_mae_vs_actual']:.2f}F "
                      f"(bias: {pq['pre_bias_vs_actual']:.2f}F)")
                print(f"    Overlap MAE vs actual:      {pq.get('overlap_mae_vs_actual', 'N/A'):.2f}F "
                      f"(bias: {pq.get('overlap_bias_vs_actual', 'N/A'):.2f}F)")

    # ------------------------------------------------------------------
    # Part 6: Generate Report and Save Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 6: Generating Report")
    print("=" * 70)

    report = generate_report(
        similarity_results,
        harmonization_results,
        multistation_results,
        pre_overlap_results,
        airport_data,
        knyc_data,
    )

    report_path = os.path.join(RESULTS_DIR, "similarity_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    # Save comparison CSV
    comp_df = save_comparison_csv(similarity_results, harmonization_results)
    print(f"\n  Comparison CSV saved to: {os.path.join(RESULTS_DIR, 'station_comparison.csv')}")
    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")

    # Print final summary
    print("\n  FINAL SUMMARY:")
    if comp_df is not None and not comp_df.empty:
        print("\n  Station Comparison (GFS):")
        gfs_comp = comp_df[comp_df["model"] == "gfs"]
        if not gfs_comp.empty:
            print(gfs_comp[["station", "correlation", "mean_bias", "mae_vs_knyc",
                            "harm_monthly_mae"]].to_string(index=False))

        print("\n  Station Comparison (NAM):")
        nam_comp = comp_df[comp_df["model"] == "nam"]
        if not nam_comp.empty:
            print(nam_comp[["station", "correlation", "mean_bias", "mae_vs_knyc",
                            "harm_monthly_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
