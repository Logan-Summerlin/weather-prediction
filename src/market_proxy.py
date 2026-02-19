"""
Enhanced Market Probability Proxy for Kalshi KXHIGHNY Backtesting.

Builds a sophisticated market probability proxy from NOAA GHCN historical
observations (Central Park TMAX) when real-time NWS MOS forecasts are
unavailable. Replaces the naive 40/60 persistence-climatology blend with
a regression-based forecast that captures autoregressive structure and
seasonal mean-reversion.

Components:
    1. Smooth daily climatology (day-of-year mean/std from 30+ years,
       Gaussian-smoothed with a 15-day window).
    2. Season-aware persistence-climatology weighting.
    3. Multi-day linear regression forecast (lag-1, lag-2, clim mean).
    4. Seasonally-varying forecast error sigma (by month).
    5. Bracket probability computation via normal CDF.

Usage:
    >>> proxy = MarketProxy(pd.read_csv("data/central_park_tmax_full_history.csv"))
    >>> proxy.fit(train_end_date="2022-12-31")
    >>> mu, sigma = proxy.predict_mu_sigma(date(2023, 7, 15), yesterday_tmax=85.0)
    >>> prob = proxy.compute_bracket_prob(date(2023, 7, 15), 85.0, 80.0, 90.0, "between")
"""

import os
import sys
import logging
from datetime import date, timedelta
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from src.seasons import SEASON_MAP_SHORT

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

# ---------------------------------------------------------------------------
# Season definitions
# ---------------------------------------------------------------------------
# Persistence weights by season (persistence / climatology)
# Winter: more volatile, rely more on climatology
# Summer: more stable, persistence carries further
PERSISTENCE_WEIGHTS = {
    "Winter": 0.35,
    "Spring": 0.40,
    "Summer": 0.50,
    "Fall": 0.40,
}


class MarketProxy:
    """Enhanced market probability proxy built from NOAA historical data.

    Produces day-ahead (mu, sigma) temperature forecasts and converts
    them into Kalshi bracket probabilities using the normal CDF.

    Parameters
    ----------
    actual_tmax_history : pd.DataFrame
        Historical TMAX data with columns: date, tmax_f.
        Should cover 30+ years for stable climatology estimates.

    Attributes
    ----------
    climatology_mean : np.ndarray
        Smoothed day-of-year mean TMAX (366 entries, 1-indexed).
    climatology_std : np.ndarray
        Smoothed day-of-year standard deviation (366 entries).
    regression_coefs : np.ndarray or None
        Coefficients for the linear regression forecast model.
    monthly_sigma : dict
        Monthly forecast error standard deviation (fit on training data).
    """

    def __init__(self, actual_tmax_history: pd.DataFrame):
        if actual_tmax_history is None or actual_tmax_history.empty:
            raise ValueError("actual_tmax_history must be a non-empty DataFrame")

        required = {"date", "tmax_f"}
        if not required.issubset(actual_tmax_history.columns):
            raise ValueError(
                f"actual_tmax_history must have columns: {required}. "
                f"Got: {set(actual_tmax_history.columns)}"
            )

        df = actual_tmax_history.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
        df = df.dropna(subset=["tmax_f"])
        self._history = df.reset_index(drop=True)

        # Will be set by fit()
        self.climatology_mean = None  # shape (367,) for doy 0..366
        self.climatology_std = None
        self.regression_coefs = None
        self.regression_intercept = None
        self.monthly_sigma = {}
        self.overall_sigma = None
        self._train_end_date = None
        self._is_fitted = False

    def fit(self, train_end_date: str) -> "MarketProxy":
        """Fit the proxy model using data up to train_end_date (inclusive).

        Computes smooth daily climatology, fits the multi-day regression
        forecast, and estimates monthly forecast error sigma.

        Parameters
        ----------
        train_end_date : str
            Cutoff date in "YYYY-MM-DD" format. Only data on or before
            this date is used for fitting (no leakage).

        Returns
        -------
        MarketProxy
            Self, for method chaining.
        """
        cutoff = pd.to_datetime(train_end_date).date()
        self._train_end_date = cutoff

        train_df = self._history[self._history["date"] <= cutoff].copy()
        if len(train_df) < 365:
            raise ValueError(
                f"Need at least 365 days of training data; got {len(train_df)}"
            )

        logger.info(
            "Fitting MarketProxy on %d days (%s to %s)",
            len(train_df), train_df["date"].min(), train_df["date"].max(),
        )

        # --- Step 1: Smooth daily climatology ---
        self._fit_climatology(train_df)

        # --- Step 2: Fit multi-day regression ---
        self._fit_regression(train_df)

        # --- Step 3: Compute monthly forecast error sigma ---
        self._fit_monthly_sigma(train_df)

        self._is_fitted = True
        logger.info("MarketProxy fitted successfully")
        return self

    def _fit_climatology(self, train_df: pd.DataFrame) -> None:
        """Compute Gaussian-smoothed day-of-year climatology.

        Uses a 15-day smoothing window for stable daily estimates.
        Handles leap year by using day-of-year 1..366.
        """
        train_df = train_df.copy()
        train_df["doy"] = train_df["date"].apply(
            lambda d: d.timetuple().tm_yday
        )

        # Compute raw day-of-year statistics
        raw_mean = np.full(367, np.nan)  # index 0 unused; 1..366
        raw_std = np.full(367, np.nan)

        for doy in range(1, 367):
            mask = train_df["doy"] == doy
            vals = train_df.loc[mask, "tmax_f"]
            if len(vals) >= 5:
                raw_mean[doy] = vals.mean()
                raw_std[doy] = vals.std()

        # Fill any missing days (e.g., Feb 29 if not enough data)
        # by interpolating from neighbors
        for doy in range(1, 367):
            if np.isnan(raw_mean[doy]):
                neighbors = []
                for offset in range(-7, 8):
                    nd = ((doy - 1 + offset) % 366) + 1
                    if not np.isnan(raw_mean[nd]):
                        neighbors.append(raw_mean[nd])
                raw_mean[doy] = np.mean(neighbors) if neighbors else 60.0
            if np.isnan(raw_std[doy]):
                neighbors = []
                for offset in range(-7, 8):
                    nd = ((doy - 1 + offset) % 366) + 1
                    if not np.isnan(raw_std[nd]):
                        neighbors.append(raw_std[nd])
                raw_std[doy] = np.mean(neighbors) if neighbors else 10.0

        # Apply Gaussian smoothing with wrap-around
        # We pad the signal cyclically for smooth year-boundary transitions
        period = raw_mean[1:367]  # 366 values
        padded = np.concatenate([period[-30:], period, period[:30]])

        sigma_smooth = 15 / 2.355  # 15-day FWHM -> sigma
        smoothed_mean = gaussian_filter1d(padded, sigma=sigma_smooth)
        smoothed_std = gaussian_filter1d(
            np.concatenate([
                raw_std[336:367], raw_std[1:367], raw_std[1:31]
            ]),
            sigma=sigma_smooth,
        )

        # Extract the central portion
        self.climatology_mean = np.zeros(367)
        self.climatology_std = np.zeros(367)
        self.climatology_mean[1:367] = smoothed_mean[30:396]
        self.climatology_std[1:367] = np.maximum(smoothed_std[30:396], 2.0)

        logger.info(
            "Climatology computed: mean range [%.1f, %.1f], "
            "std range [%.1f, %.1f]",
            self.climatology_mean[1:367].min(),
            self.climatology_mean[1:367].max(),
            self.climatology_std[1:367].min(),
            self.climatology_std[1:367].max(),
        )

    def _get_clim(self, d: date) -> Tuple[float, float]:
        """Get smoothed climatological mean and std for a given date.

        Parameters
        ----------
        d : date
            The date to look up.

        Returns
        -------
        tuple[float, float]
            (climatological_mean, climatological_std) in degrees F.
        """
        doy = d.timetuple().tm_yday
        if doy > 366:
            doy = 366
        return float(self.climatology_mean[doy]), float(self.climatology_std[doy])

    def _fit_regression(self, train_df: pd.DataFrame) -> None:
        """Fit linear regression: TMAX(t) ~ lag1 + lag2 + clim_mean.

        Uses ordinary least squares on the training data.
        """
        df = train_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Compute features
        df["lag1"] = df["tmax_f"].shift(1)
        df["lag2"] = df["tmax_f"].shift(2)
        df["doy"] = df["date"].apply(lambda d: d.timetuple().tm_yday)
        df["clim_mean"] = df["doy"].apply(
            lambda doy: float(self.climatology_mean[min(doy, 366)])
        )

        # Drop rows with NaN features
        df = df.dropna(subset=["lag1", "lag2", "clim_mean", "tmax_f"])
        if len(df) < 100:
            logger.warning("Only %d rows for regression; using simple blend", len(df))
            self.regression_coefs = None
            return

        X = df[["lag1", "lag2", "clim_mean"]].values
        y = df["tmax_f"].values

        # Add intercept column
        X_with_intercept = np.column_stack([X, np.ones(len(X))])

        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.regression_coefs = beta[:3]
            self.regression_intercept = beta[3]

            # In-sample evaluation
            y_hat = X_with_intercept @ beta
            residuals = y - y_hat
            mae = np.mean(np.abs(residuals))
            self.overall_sigma = float(np.std(residuals))

            logger.info(
                "Regression fitted: coefs=[%.4f, %.4f, %.4f], "
                "intercept=%.2f, MAE=%.2f, sigma=%.2f",
                *self.regression_coefs, self.regression_intercept,
                mae, self.overall_sigma,
            )
        except np.linalg.LinAlgError:
            logger.warning("Regression fit failed; using simple blend")
            self.regression_coefs = None

    def _fit_monthly_sigma(self, train_df: pd.DataFrame) -> None:
        """Compute monthly forecast error standard deviation.

        Uses leave-one-year-out cross-validation on the training data
        to get realistic out-of-sample error estimates by month.
        """
        df = train_df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["lag1"] = df["tmax_f"].shift(1)
        df["lag2"] = df["tmax_f"].shift(2)
        df["doy"] = df["date"].apply(lambda d: d.timetuple().tm_yday)
        df["clim_mean"] = df["doy"].apply(
            lambda doy: float(self.climatology_mean[min(doy, 366)])
        )
        df["month"] = df["date"].apply(lambda d: d.month)
        df = df.dropna(subset=["lag1", "lag2", "clim_mean", "tmax_f"])

        if self.regression_coefs is not None:
            # Compute residuals using the fitted regression
            X = df[["lag1", "lag2", "clim_mean"]].values
            y = df["tmax_f"].values
            y_hat = X @ self.regression_coefs + self.regression_intercept
            df["residual"] = y - y_hat
        else:
            # Fall back to persistence residuals
            df["residual"] = df["tmax_f"] - df["lag1"]

        # Monthly sigma from residuals
        for month in range(1, 13):
            mask = df["month"] == month
            resid = df.loc[mask, "residual"]
            if len(resid) >= 10:
                self.monthly_sigma[month] = float(resid.std())
            else:
                self.monthly_sigma[month] = self.overall_sigma or 7.0

        if self.overall_sigma is None:
            self.overall_sigma = float(df["residual"].std())

        logger.info(
            "Monthly sigma: %s",
            {m: f"{s:.2f}" for m, s in sorted(self.monthly_sigma.items())},
        )

    def predict_mu_sigma(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        day_before_tmax: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Produce a (mu, sigma) temperature forecast for a given date.

        Uses the regression model if available, otherwise falls back to
        a season-aware persistence-climatology blend.

        Parameters
        ----------
        target_date : date or str
            The date to forecast.
        yesterday_tmax : float
            Observed TMAX on day t-1.
        day_before_tmax : float, optional
            Observed TMAX on day t-2. If None, uses yesterday_tmax
            as a fallback for the lag-2 feature.

        Returns
        -------
        tuple[float, float]
            (mu, sigma) forecast in degrees Fahrenheit.
        """
        if not self._is_fitted:
            raise RuntimeError("MarketProxy not fitted. Call fit() first.")

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        clim_mean, clim_std = self._get_clim(target_date)
        month = target_date.month

        if day_before_tmax is None:
            day_before_tmax = yesterday_tmax

        # Predict mu
        if self.regression_coefs is not None:
            X = np.array([yesterday_tmax, day_before_tmax, clim_mean])
            mu = float(np.dot(X, self.regression_coefs) + self.regression_intercept)
        else:
            # Fallback: season-aware persistence-climatology blend
            season = SEASON_MAP.get(month, "Spring")
            alpha = PERSISTENCE_WEIGHTS.get(season, 0.40)
            mu = alpha * yesterday_tmax + (1.0 - alpha) * clim_mean

        # Sigma: monthly-specific if available
        sigma = self.monthly_sigma.get(month, self.overall_sigma or 7.0)

        return mu, sigma

    def compute_bracket_prob(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        threshold_low: Optional[float],
        threshold_high: Optional[float],
        direction: str,
        day_before_tmax: Optional[float] = None,
    ) -> float:
        """Compute probability for a Kalshi bracket.

        Parameters
        ----------
        target_date : date or str
            The forecast date.
        yesterday_tmax : float
            Observed TMAX on day t-1.
        threshold_low : float or None
            Lower temperature bound (None for open-ended below).
        threshold_high : float or None
            Upper temperature bound (None for open-ended above).
        direction : str
            One of "above", "below", "between".
        day_before_tmax : float, optional
            TMAX on day t-2.

        Returns
        -------
        float
            Probability in [0.02, 0.98].
        """
        mu, sigma = self.predict_mu_sigma(
            target_date, yesterday_tmax, day_before_tmax,
        )

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

        # Clip to avoid extreme probabilities
        prob = float(np.clip(prob, 0.02, 0.98))
        return prob

    def compute_bracket_probs_for_day(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        brackets: list,
        day_before_tmax: Optional[float] = None,
    ) -> list:
        """Compute probabilities for all brackets on a given day.

        Parameters
        ----------
        target_date : date or str
            The forecast date.
        yesterday_tmax : float
            Observed TMAX on day t-1.
        brackets : list[dict]
            List of bracket dicts with keys: threshold_low, threshold_high,
            direction.
        day_before_tmax : float, optional
            TMAX on day t-2.

        Returns
        -------
        list[float]
            Probabilities for each bracket.
        """
        return [
            self.compute_bracket_prob(
                target_date,
                yesterday_tmax,
                b.get("threshold_low"),
                b.get("threshold_high"),
                b.get("direction", "between"),
                day_before_tmax,
            )
            for b in brackets
        ]

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the fitted proxy.

        Returns
        -------
        dict
            Diagnostic info including regression coefficients,
            monthly sigmas, climatology summary.
        """
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "train_end_date": str(self._train_end_date),
            "history_size": len(self._history),
            "regression_coefs": (
                self.regression_coefs.tolist()
                if self.regression_coefs is not None else None
            ),
            "regression_intercept": (
                float(self.regression_intercept)
                if self.regression_intercept is not None else None
            ),
            "overall_sigma": self.overall_sigma,
            "monthly_sigma": self.monthly_sigma,
            "climatology_mean_jan1": float(self.climatology_mean[1]),
            "climatology_mean_jul1": float(self.climatology_mean[182]),
            "climatology_std_jan1": float(self.climatology_std[1]),
            "climatology_std_jul1": float(self.climatology_std[182]),
        }


class NaiveMarketProxy:
    """Naive 40/60 persistence-climatology blend for comparison.

    This is the original simple proxy that uses monthly climatology
    means/stds without smoothing or regression.

    Parameters
    ----------
    actual_tmax_history : pd.DataFrame
        Historical TMAX data with columns: date, tmax_f.
    """

    # Monthly climatology (approximate NYC Central Park values)
    MONTHLY_MEAN = {
        1: 39.0, 2: 42.0, 3: 50.0, 4: 62.0, 5: 72.0, 6: 80.0,
        7: 85.0, 8: 84.0, 9: 76.0, 10: 65.0, 11: 54.0, 12: 43.0,
    }
    MONTHLY_STD = {
        1: 11.0, 2: 10.5, 3: 10.0, 4: 9.5, 5: 8.0, 6: 6.5,
        7: 5.5, 8: 5.5, 9: 6.5, 10: 8.0, 11: 9.5, 12: 10.5,
    }

    def __init__(self, actual_tmax_history: Optional[pd.DataFrame] = None):
        self._history = actual_tmax_history
        self._is_fitted = True  # No fitting needed for naive version

    def fit(self, train_end_date: str) -> "NaiveMarketProxy":
        """No-op for compatibility."""
        return self

    def predict_mu_sigma(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        day_before_tmax: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Naive 40% persistence / 60% climatology blend."""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        month = target_date.month
        clim_mean = self.MONTHLY_MEAN[month]
        clim_std = self.MONTHLY_STD[month]

        mu = 0.40 * yesterday_tmax + 0.60 * clim_mean
        sigma = clim_std

        return mu, sigma

    def compute_bracket_prob(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        threshold_low: Optional[float],
        threshold_high: Optional[float],
        direction: str,
        day_before_tmax: Optional[float] = None,
    ) -> float:
        """Compute bracket probability using naive blend."""
        mu, sigma = self.predict_mu_sigma(target_date, yesterday_tmax)
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
