"""
MOS-based Market Probability Proxy for Kalshi KXHIGHNY Backtesting.

Uses actual NWS Model Output Statistics (MOS) forecasts from the IEM archive
as the market proxy. This represents what a well-informed Kalshi participant
would use: the NWS day-ahead max temperature forecast plus the historical
forecast error distribution.

The interface matches EnhancedMarketProxy for plug-in compatibility.

Usage:
    >>> mos_df = pd.read_csv("data/mos/combined_mos_knyc.csv")
    >>> actual_df = pd.read_csv("data/nyc/central_park_tmax_full_history.csv")
    >>> proxy = MOSMarketProxy(mos_df, actual_df)
    >>> proxy.fit(train_end_date="2022-12-31")
    >>> mu, sigma = proxy.predict_mu_sigma(date(2023, 7, 15))
"""

import logging
import os
import sys
from datetime import date, timedelta
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MOSMarketProxy:
    """Market proxy using actual NWS MOS forecasts.

    This represents what a well-informed Kalshi participant would use:
    the NWS day-ahead max temperature forecast + historical forecast
    error distribution.

    Interface matches EnhancedMarketProxy for plug-in compatibility.

    Parameters
    ----------
    mos_forecasts_df : pd.DataFrame
        MOS forecast data. Must have columns: date, mos_ensemble_tmax_f.
        Optionally: gfs_mos_tmax_f, nam_mos_tmax_f.
    actual_tmax_df : pd.DataFrame
        Observed TMAX data. Must have columns: date, tmax_f.
    """

    def __init__(
        self,
        mos_forecasts_df: pd.DataFrame,
        actual_tmax_df: pd.DataFrame,
    ):
        if mos_forecasts_df is None or mos_forecasts_df.empty:
            raise ValueError("mos_forecasts_df must be a non-empty DataFrame")
        if actual_tmax_df is None or actual_tmax_df.empty:
            raise ValueError("actual_tmax_df must be a non-empty DataFrame")

        # Validate MOS columns
        if "mos_ensemble_tmax_f" not in mos_forecasts_df.columns:
            # Try to build ensemble from individual models
            has_gfs = "gfs_mos_tmax_f" in mos_forecasts_df.columns
            has_nam = "nam_mos_tmax_f" in mos_forecasts_df.columns
            if has_gfs or has_nam:
                cols = []
                if has_gfs:
                    cols.append("gfs_mos_tmax_f")
                if has_nam:
                    cols.append("nam_mos_tmax_f")
                mos_forecasts_df = mos_forecasts_df.copy()
                mos_forecasts_df["mos_ensemble_tmax_f"] = (
                    mos_forecasts_df[cols].mean(axis=1)
                )
            else:
                raise ValueError(
                    "mos_forecasts_df must have 'mos_ensemble_tmax_f' column "
                    "or individual model columns (gfs_mos_tmax_f, nam_mos_tmax_f)"
                )

        required_actual = {"date", "tmax_f"}
        if not required_actual.issubset(actual_tmax_df.columns):
            raise ValueError(
                f"actual_tmax_df must have columns: {required_actual}. "
                f"Got: {set(actual_tmax_df.columns)}"
            )

        # Store MOS data
        mos = mos_forecasts_df.copy()
        mos["date"] = pd.to_datetime(mos["date"]).dt.date
        mos = mos.sort_values("date").drop_duplicates(subset="date", keep="first")
        self._mos = mos.reset_index(drop=True)

        # Store actual TMAX data
        act = actual_tmax_df.copy()
        act["date"] = pd.to_datetime(act["date"]).dt.date
        act = act.sort_values("date").drop_duplicates(subset="date", keep="first")
        act = act.dropna(subset=["tmax_f"])
        self._actuals = act.reset_index(drop=True)

        # Build lookup dictionaries for fast access
        self._mos_lookup: Dict[date, float] = {}
        self._gfs_lookup: Dict[date, float] = {}
        self._nam_lookup: Dict[date, float] = {}
        self._actual_lookup: Dict[date, float] = {}

        for _, row in self._mos.iterrows():
            d = row["date"]
            if pd.notna(row.get("mos_ensemble_tmax_f")):
                self._mos_lookup[d] = float(row["mos_ensemble_tmax_f"])
            if pd.notna(row.get("gfs_mos_tmax_f")):
                self._gfs_lookup[d] = float(row["gfs_mos_tmax_f"])
            if pd.notna(row.get("nam_mos_tmax_f")):
                self._nam_lookup[d] = float(row["nam_mos_tmax_f"])

        for _, row in self._actuals.iterrows():
            self._actual_lookup[row["date"]] = float(row["tmax_f"])

        # Fitted attributes
        self.monthly_sigma: Dict[int, float] = {}
        self.monthly_bias: Dict[int, float] = {}
        self.monthly_mae: Dict[int, float] = {}
        self.overall_sigma: Optional[float] = None
        self.overall_mae: Optional[float] = None
        self.overall_rmse: Optional[float] = None
        self.overall_bias: Optional[float] = None
        self.n_train_days: int = 0
        self._train_end_date: Optional[date] = None
        self._is_fitted: bool = False

    def fit(self, train_end_date: str) -> "MOSMarketProxy":
        """Compute historical forecast error statistics by month.

        For each month, computes std(actual - mos_forecast) using data
        up to train_end_date. This gives the uncertainty envelope around
        the MOS point forecast.

        Parameters
        ----------
        train_end_date : str
            Cutoff date "YYYY-MM-DD". Only data on or before is used.

        Returns
        -------
        MOSMarketProxy
            Self, for method chaining.
        """
        cutoff = pd.to_datetime(train_end_date).date()
        self._train_end_date = cutoff

        # Build paired (forecast, actual) DataFrame for training period
        records = []
        for d, mos_val in self._mos_lookup.items():
            if d <= cutoff and d in self._actual_lookup:
                records.append({
                    "date": d,
                    "mos_forecast": mos_val,
                    "actual": self._actual_lookup[d],
                })

        if not records:
            raise ValueError(
                "No overlapping MOS forecast + actual data found before "
                f"{cutoff}. Check date ranges."
            )

        train_df = pd.DataFrame(records)
        train_df["error"] = train_df["actual"] - train_df["mos_forecast"]
        train_df["month"] = train_df["date"].apply(lambda d: d.month)

        self.n_train_days = len(train_df)
        logger.info(
            "Fitting MOSMarketProxy on %d days (%s to %s)",
            self.n_train_days,
            train_df["date"].min(),
            train_df["date"].max(),
        )

        # Overall statistics
        errors = train_df["error"].values
        self.overall_sigma = float(np.std(errors, ddof=1))
        self.overall_mae = float(np.mean(np.abs(errors)))
        self.overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.overall_bias = float(np.mean(errors))

        logger.info(
            "Overall: MAE=%.2f, RMSE=%.2f, bias=%.2f, sigma=%.2f",
            self.overall_mae, self.overall_rmse,
            self.overall_bias, self.overall_sigma,
        )

        # Monthly statistics
        for month in range(1, 13):
            mask = train_df["month"] == month
            month_errors = train_df.loc[mask, "error"]

            if len(month_errors) >= 10:
                self.monthly_sigma[month] = float(month_errors.std(ddof=1))
                self.monthly_bias[month] = float(month_errors.mean())
                self.monthly_mae[month] = float(month_errors.abs().mean())
            else:
                # Fall back to overall statistics
                self.monthly_sigma[month] = self.overall_sigma
                self.monthly_bias[month] = self.overall_bias
                self.monthly_mae[month] = self.overall_mae

        logger.info(
            "Monthly sigma: %s",
            {m: f"{s:.2f}" for m, s in sorted(self.monthly_sigma.items())},
        )

        self._is_fitted = True
        return self

    def predict_mu_sigma(
        self,
        target_date: Union[date, str],
        **kwargs,
    ) -> Tuple[float, float]:
        """Return (mu, sigma) where mu = MOS ensemble forecast, sigma = monthly error std.

        Parameters
        ----------
        target_date : date or str
            Date to forecast.
        **kwargs
            Accepted for interface compatibility (yesterday_tmax, etc.)
            but not used since MOS provides its own forecast.

        Returns
        -------
        tuple[float, float]
            (mu, sigma) in degrees Fahrenheit.
        """
        if not self._is_fitted:
            raise RuntimeError("MOSMarketProxy not fitted. Call fit() first.")

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        # Get MOS forecast for this date
        mu = self._get_mos_forecast(target_date)
        if mu is None:
            # No MOS forecast available; fall back to yesterday's TMAX if given
            yesterday_tmax = kwargs.get("yesterday_tmax")
            if yesterday_tmax is not None:
                mu = float(yesterday_tmax)
            else:
                # Last resort: use the nearest available MOS forecast
                mu = self._get_nearest_mos_forecast(target_date)

        # Get monthly sigma
        month = target_date.month
        sigma = self.monthly_sigma.get(month, self.overall_sigma or 5.0)

        return float(mu), float(sigma)

    def compute_bracket_prob(
        self,
        target_date: Union[date, str],
        threshold_low: Optional[float],
        threshold_high: Optional[float],
        direction: str,
        **kwargs,
    ) -> float:
        """P(L <= TMAX < U) using N(mos_forecast, sigma_monthly).

        Parameters match EnhancedMarketProxy.compute_bracket_prob() for
        compatibility.

        Parameters
        ----------
        target_date : date or str
        threshold_low : float or None
        threshold_high : float or None
        direction : str
            "above", "below", or "between"
        **kwargs
            Accepted for interface compatibility (yesterday_tmax, etc.)

        Returns
        -------
        float
            Probability in [0.02, 0.98].
        """
        mu, sigma = self.predict_mu_sigma(target_date, **kwargs)
        sigma = max(sigma, 1e-6)

        if direction == "above" and threshold_low is not None:
            prob = 1.0 - stats.norm.cdf(threshold_low, loc=mu, scale=sigma)
        elif direction == "below" and threshold_high is not None:
            prob = stats.norm.cdf(threshold_high, loc=mu, scale=sigma)
        elif direction == "between":
            lo = threshold_low if threshold_low is not None else -999
            hi = threshold_high if threshold_high is not None else 999
            prob = (
                stats.norm.cdf(hi, loc=mu, scale=sigma)
                - stats.norm.cdf(lo, loc=mu, scale=sigma)
            )
        else:
            prob = 0.5

        return float(np.clip(prob, 0.02, 0.98))

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the fitted proxy."""
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "train_end_date": str(self._train_end_date),
            "n_train_days": self.n_train_days,
            "overall_mae": self.overall_mae,
            "overall_rmse": self.overall_rmse,
            "overall_bias": self.overall_bias,
            "overall_sigma": self.overall_sigma,
            "monthly_sigma": dict(self.monthly_sigma),
            "monthly_bias": dict(self.monthly_bias),
            "monthly_mae": dict(self.monthly_mae),
            "mos_date_range": (
                str(min(self._mos_lookup.keys())),
                str(max(self._mos_lookup.keys())),
            ) if self._mos_lookup else None,
            "actual_date_range": (
                str(min(self._actual_lookup.keys())),
                str(max(self._actual_lookup.keys())),
            ) if self._actual_lookup else None,
        }

    def generate_proxy_forecasts(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate daily (mu, sigma) proxy forecasts over a date range.

        Parameters
        ----------
        start_date, end_date : str
            Date range in "YYYY-MM-DD" format.

        Returns
        -------
        pd.DataFrame
            Columns: date, proxy_mu, proxy_sigma.
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        records = []
        current = start
        while current <= end:
            mu, sigma = self.predict_mu_sigma(current)
            if mu is not None:
                records.append({
                    "date": current,
                    "proxy_mu": mu,
                    "proxy_sigma": sigma,
                })
            current += timedelta(days=1)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_mos_forecast(self, target_date: date) -> Optional[float]:
        """Get the MOS ensemble forecast for a specific date.

        Falls back to individual models if ensemble is missing.

        Returns
        -------
        float or None
            The MOS TMAX forecast, or None if not available.
        """
        # Try ensemble first
        val = self._mos_lookup.get(target_date)
        if val is not None:
            return val

        # Fall back to GFS only
        val = self._gfs_lookup.get(target_date)
        if val is not None:
            return val

        # Fall back to NAM only
        val = self._nam_lookup.get(target_date)
        if val is not None:
            return val

        return None

    def _get_nearest_mos_forecast(self, target_date: date) -> float:
        """Find the nearest available MOS forecast to target_date.

        Used as last-resort fallback when no MOS data exists for the date.

        Returns
        -------
        float
            The nearest MOS forecast, or 60.0 as absolute fallback.
        """
        if not self._mos_lookup:
            return 60.0  # NYC annual average approximation

        min_dist = None
        best_val = 60.0
        for d, val in self._mos_lookup.items():
            dist = abs((d - target_date).days)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                best_val = val
            if dist == 0:
                break

        return best_val
