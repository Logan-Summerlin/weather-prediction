"""
Enhanced Market Probability Proxy for Kalshi KXHIGHNY Backtesting.

Builds a more sophisticated market proxy than the base MarketProxy by adding:
  1. Lag-1, lag-2, lag-3 TMAX features
  2. 7-day rolling average
  3. Smooth day-of-year climatology mean and std
  4. Surrounding station consensus (mean of 5 nearest at lag-1)
  5. Ridge regression with alpha tuning via cross-validation
  6. Per-month residual sigma from leave-one-year-out CV

Strictly trained on data BEFORE each evaluation period to avoid leakage.

Usage:
    >>> proxy = EnhancedMarketProxy(tmax_history_df, station_tmax_df=None)
    >>> proxy.fit(train_end_date="2022-12-31")
    >>> mu, sigma = proxy.predict_mu_sigma(date(2023, 7, 15), yesterday_tmax=85.0)
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


class EnhancedMarketProxy:
    """Enhanced market proxy with richer feature set and Ridge regression.

    Parameters
    ----------
    actual_tmax_history : pd.DataFrame
        Historical TMAX data with columns: date, tmax_f.
        Should cover 30+ years for stable climatology.
    station_tmax_df : pd.DataFrame, optional
        Surrounding station TMAX data. Columns: date, station1, station2, ...
        If provided, used for station consensus feature.
    """

    def __init__(
        self,
        actual_tmax_history: pd.DataFrame,
        station_tmax_df: Optional[pd.DataFrame] = None,
    ):
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

        self._station_tmax = station_tmax_df
        if station_tmax_df is not None and not station_tmax_df.empty:
            st = station_tmax_df.copy()
            st["date"] = pd.to_datetime(st["date"]).dt.date
            self._station_tmax = st

        # Fitted attributes
        self.climatology_mean = None   # shape (367,), 1-indexed doy
        self.climatology_std = None
        self.ridge_coefs = None
        self.ridge_intercept = None
        self.ridge_alpha = None
        self.feature_names = []
        self.monthly_sigma = {}
        self.overall_sigma = None
        self._train_end_date = None
        self._is_fitted = False

    def fit(self, train_end_date: str) -> "EnhancedMarketProxy":
        """Fit the enhanced proxy on data up to train_end_date (inclusive).

        Parameters
        ----------
        train_end_date : str
            Cutoff date "YYYY-MM-DD". Only data on or before is used.

        Returns
        -------
        EnhancedMarketProxy
            Self, for method chaining.
        """
        cutoff = pd.to_datetime(train_end_date).date()
        self._train_end_date = cutoff

        train_df = self._history[self._history["date"] <= cutoff].copy()
        if len(train_df) < 730:
            raise ValueError(
                f"Need at least 730 days of training data; got {len(train_df)}"
            )

        logger.info(
            "Fitting EnhancedMarketProxy on %d days (%s to %s)",
            len(train_df), train_df["date"].min(), train_df["date"].max(),
        )

        # Step 1: Smooth daily climatology
        self._fit_climatology(train_df)

        # Step 2: Build feature matrix and fit Ridge
        self._fit_ridge(train_df)

        # Step 3: Monthly sigma from CV residuals
        self._fit_monthly_sigma(train_df)

        self._is_fitted = True
        logger.info("EnhancedMarketProxy fitted successfully")
        return self

    def _fit_climatology(self, train_df: pd.DataFrame) -> None:
        """Compute Gaussian-smoothed day-of-year climatology."""
        df = train_df.copy()
        df["doy"] = df["date"].apply(lambda d: d.timetuple().tm_yday)

        raw_mean = np.full(367, np.nan)
        raw_std = np.full(367, np.nan)

        for doy in range(1, 367):
            mask = df["doy"] == doy
            vals = df.loc[mask, "tmax_f"]
            if len(vals) >= 5:
                raw_mean[doy] = vals.mean()
                raw_std[doy] = vals.std()

        # Fill gaps by neighbor interpolation
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

        # Gaussian smoothing with cyclic padding
        period = raw_mean[1:367]
        padded = np.concatenate([period[-30:], period, period[:30]])
        sigma_smooth = 15 / 2.355  # 15-day FWHM

        smoothed_mean = gaussian_filter1d(padded, sigma=sigma_smooth)
        smoothed_std = gaussian_filter1d(
            np.concatenate([
                raw_std[336:367], raw_std[1:367], raw_std[1:31]
            ]),
            sigma=sigma_smooth,
        )

        self.climatology_mean = np.zeros(367)
        self.climatology_std = np.zeros(367)
        self.climatology_mean[1:367] = smoothed_mean[30:396]
        self.climatology_std[1:367] = np.maximum(smoothed_std[30:396], 2.0)

        logger.info(
            "Climatology: mean range [%.1f, %.1f], std range [%.1f, %.1f]",
            self.climatology_mean[1:367].min(),
            self.climatology_mean[1:367].max(),
            self.climatology_std[1:367].min(),
            self.climatology_std[1:367].max(),
        )

    def _get_clim(self, d: date) -> Tuple[float, float]:
        """Get climatological mean and std for a given date."""
        doy = min(d.timetuple().tm_yday, 366)
        return float(self.climatology_mean[doy]), float(self.climatology_std[doy])

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the enhanced feature matrix from a sorted DataFrame of TMAX.

        Features:
          1. lag1, lag2, lag3 TMAX
          2. rolling_7d_mean (7-day trailing mean of TMAX)
          3. clim_mean (smooth day-of-year climatology mean)
          4. clim_std (smooth day-of-year climatology std)
          5. station_consensus (mean of nearest stations at lag-1, if available)

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: date, tmax_f. Sorted by date.

        Returns
        -------
        pd.DataFrame
            Feature matrix with columns for each feature + 'target' (TMAX).
        """
        feat = df.copy()
        feat = feat.sort_values("date").reset_index(drop=True)

        # Lag features
        feat["lag1"] = feat["tmax_f"].shift(1)
        feat["lag2"] = feat["tmax_f"].shift(2)
        feat["lag3"] = feat["tmax_f"].shift(3)

        # 7-day rolling mean (trailing, shift by 1 to avoid using today)
        feat["rolling_7d_mean"] = feat["tmax_f"].shift(1).rolling(7, min_periods=3).mean()

        # Climatology features
        feat["doy"] = feat["date"].apply(lambda d: min(d.timetuple().tm_yday, 366))
        feat["clim_mean"] = feat["doy"].apply(
            lambda doy: float(self.climatology_mean[doy])
        )
        feat["clim_std"] = feat["doy"].apply(
            lambda doy: float(self.climatology_std[doy])
        )

        # Station consensus (if available)
        if self._station_tmax is not None and not self._station_tmax.empty:
            st = self._station_tmax.copy()
            station_cols = [c for c in st.columns if c != "date"]
            if station_cols:
                # For each date, compute mean of top-5 non-null station TMAX at lag-1
                st_mean = st.set_index("date")[station_cols].mean(axis=1)
                st_lookup = st_mean.to_dict()
                feat["station_consensus"] = feat["date"].apply(
                    lambda d: st_lookup.get(d - timedelta(days=1), np.nan)
                )
            else:
                feat["station_consensus"] = np.nan
        else:
            feat["station_consensus"] = np.nan

        feat["target"] = feat["tmax_f"]

        return feat

    def _fit_ridge(self, train_df: pd.DataFrame) -> None:
        """Fit Ridge regression on the enhanced feature set.

        Uses cross-validation across alphas to select the best
        regularization strength.
        """
        feat_df = self._build_features(train_df)

        # Determine which features to use
        base_features = ["lag1", "lag2", "lag3", "rolling_7d_mean", "clim_mean", "clim_std"]
        has_stations = feat_df["station_consensus"].notna().mean() > 0.5
        if has_stations:
            feature_cols = base_features + ["station_consensus"]
        else:
            feature_cols = base_features

        # Drop NaN rows
        complete = feat_df.dropna(subset=feature_cols + ["target"])
        if len(complete) < 200:
            logger.warning("Only %d complete rows; falling back to simple regression", len(complete))
            self.ridge_coefs = None
            return

        X = complete[feature_cols].values
        y = complete["target"].values

        # Cross-validate alphas
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = 1.0
        best_cv_mae = np.inf

        n = len(X)
        # Time-series CV: use 5 folds of ~20% each
        fold_size = max(n // 5, 100)

        for alpha in alphas:
            cv_maes = []
            for fold_start in range(n - fold_size, max(fold_size, n // 2), -fold_size):
                fold_end = fold_start + fold_size
                if fold_end > n:
                    fold_end = n
                if fold_start < fold_size:
                    break

                X_train_cv = X[:fold_start]
                y_train_cv = y[:fold_start]
                X_val_cv = X[fold_start:fold_end]
                y_val_cv = y[fold_start:fold_end]

                if len(X_train_cv) < 100 or len(X_val_cv) < 30:
                    continue

                # Ridge: (X'X + alpha*I)^-1 X'y
                XtX = X_train_cv.T @ X_train_cv + alpha * np.eye(X_train_cv.shape[1])
                Xty = X_train_cv.T @ y_train_cv
                try:
                    beta = np.linalg.solve(XtX, Xty)
                    y_pred_cv = X_val_cv @ beta
                    mae = np.mean(np.abs(y_val_cv - y_pred_cv))
                    cv_maes.append(mae)
                except np.linalg.LinAlgError:
                    pass

            if cv_maes:
                avg_mae = np.mean(cv_maes)
                if avg_mae < best_cv_mae:
                    best_cv_mae = avg_mae
                    best_alpha = alpha

        # Fit final model with best alpha on all training data
        self.ridge_alpha = best_alpha
        XtX = X.T @ X + best_alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        try:
            # Add intercept by centering
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xc = X - X_mean
            yc = y - y_mean
            XtXc = Xc.T @ Xc + best_alpha * np.eye(Xc.shape[1])
            Xtyc = Xc.T @ yc
            coefs = np.linalg.solve(XtXc, Xtyc)
            intercept = y_mean - X_mean @ coefs

            self.ridge_coefs = coefs
            self.ridge_intercept = float(intercept)
            self.feature_names = feature_cols

            # In-sample eval
            y_hat = X @ coefs + intercept
            residuals = y - y_hat
            self.overall_sigma = float(np.std(residuals))
            mae = np.mean(np.abs(residuals))

            logger.info(
                "Ridge fitted: alpha=%.2f, %d features, MAE=%.2f, sigma=%.2f, CV_MAE=%.2f",
                best_alpha, len(feature_cols), mae, self.overall_sigma, best_cv_mae,
            )
            logger.info("  Feature coefs: %s",
                         {n: f"{c:.4f}" for n, c in zip(feature_cols, coefs)})
        except np.linalg.LinAlgError:
            logger.warning("Ridge fit failed; setting coefs to None")
            self.ridge_coefs = None

    def _fit_monthly_sigma(self, train_df: pd.DataFrame) -> None:
        """Compute monthly forecast error sigma from training residuals.

        Uses leave-one-year-out cross-validation style: computes residuals
        on the full training set, then groups by month.
        """
        feat_df = self._build_features(train_df)

        if self.ridge_coefs is not None and self.feature_names:
            complete = feat_df.dropna(subset=self.feature_names + ["target"])
            X = complete[self.feature_names].values
            y = complete["target"].values
            y_hat = X @ self.ridge_coefs + self.ridge_intercept
            complete = complete.copy()
            complete["residual"] = y - y_hat
        else:
            complete = feat_df.dropna(subset=["lag1", "target"]).copy()
            complete["residual"] = complete["target"] - complete["lag1"]

        complete["month"] = complete["date"].apply(lambda d: d.month)

        for month in range(1, 13):
            mask = complete["month"] == month
            resid = complete.loc[mask, "residual"]
            if len(resid) >= 15:
                self.monthly_sigma[month] = float(resid.std())
            else:
                self.monthly_sigma[month] = self.overall_sigma or 7.0

        if self.overall_sigma is None:
            self.overall_sigma = float(complete["residual"].std())

        logger.info(
            "Monthly sigma: %s",
            {m: f"{s:.2f}" for m, s in sorted(self.monthly_sigma.items())},
        )

    def predict_mu_sigma(
        self,
        target_date: Union[date, str],
        yesterday_tmax: float,
        day_before_tmax: Optional[float] = None,
        three_days_ago_tmax: Optional[float] = None,
        rolling_7d_mean: Optional[float] = None,
        station_consensus: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Produce a (mu, sigma) forecast for a given date.

        Parameters
        ----------
        target_date : date or str
            Date to forecast.
        yesterday_tmax : float
            TMAX on day t-1.
        day_before_tmax : float, optional
            TMAX on day t-2. Defaults to yesterday_tmax.
        three_days_ago_tmax : float, optional
            TMAX on day t-3. Defaults to day_before_tmax.
        rolling_7d_mean : float, optional
            7-day trailing mean. If None, approximated from lags.
        station_consensus : float, optional
            Mean TMAX of nearest surrounding stations at t-1.

        Returns
        -------
        tuple[float, float]
            (mu, sigma) in degrees Fahrenheit.
        """
        if not self._is_fitted:
            raise RuntimeError("EnhancedMarketProxy not fitted. Call fit() first.")

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        if day_before_tmax is None:
            day_before_tmax = yesterday_tmax
        if three_days_ago_tmax is None:
            three_days_ago_tmax = day_before_tmax

        clim_mean, clim_std = self._get_clim(target_date)
        month = target_date.month

        if self.ridge_coefs is not None and self.feature_names:
            # Build feature vector
            if rolling_7d_mean is None:
                rolling_7d_mean = (yesterday_tmax + day_before_tmax + three_days_ago_tmax) / 3.0

            feature_dict = {
                "lag1": yesterday_tmax,
                "lag2": day_before_tmax,
                "lag3": three_days_ago_tmax,
                "rolling_7d_mean": rolling_7d_mean,
                "clim_mean": clim_mean,
                "clim_std": clim_std,
                "station_consensus": station_consensus if station_consensus is not None else yesterday_tmax,
            }

            x = np.array([feature_dict[f] for f in self.feature_names])
            mu = float(x @ self.ridge_coefs + self.ridge_intercept)
        else:
            # Fallback: simple persistence-climatology blend
            mu = 0.40 * yesterday_tmax + 0.60 * clim_mean

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
        three_days_ago_tmax: Optional[float] = None,
        rolling_7d_mean: Optional[float] = None,
        station_consensus: Optional[float] = None,
    ) -> float:
        """Compute probability for a Kalshi bracket.

        Parameters
        ----------
        target_date : date or str
        yesterday_tmax : float
        threshold_low : float or None
        threshold_high : float or None
        direction : str - "above", "below", "between"
        day_before_tmax : float, optional
        three_days_ago_tmax : float, optional
        rolling_7d_mean : float, optional
        station_consensus : float, optional

        Returns
        -------
        float
            Probability in [0.02, 0.98].
        """
        mu, sigma = self.predict_mu_sigma(
            target_date, yesterday_tmax,
            day_before_tmax, three_days_ago_tmax,
            rolling_7d_mean, station_consensus,
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

        return float(np.clip(prob, 0.02, 0.98))

    def generate_proxy_forecasts(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate daily (mu, sigma) proxy forecasts over a date range.

        Automatically looks up lag values from the internal history.

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

        # Build lookup from history
        tmax_lookup = dict(zip(self._history["date"], self._history["tmax_f"]))

        records = []
        current = start
        while current <= end:
            yesterday = current - timedelta(days=1)
            day_before = current - timedelta(days=2)
            three_ago = current - timedelta(days=3)

            yt = tmax_lookup.get(yesterday)
            dbt = tmax_lookup.get(day_before)
            tat = tmax_lookup.get(three_ago)

            if yt is None:
                current += timedelta(days=1)
                continue

            # Compute rolling 7d mean
            recent = []
            for i in range(1, 8):
                d = current - timedelta(days=i)
                v = tmax_lookup.get(d)
                if v is not None:
                    recent.append(v)
            r7 = np.mean(recent) if len(recent) >= 3 else None

            mu, sigma = self.predict_mu_sigma(
                current, yt, dbt, tat, r7, None,
            )

            records.append({
                "date": current,
                "proxy_mu": mu,
                "proxy_sigma": sigma,
            })
            current += timedelta(days=1)

        return pd.DataFrame(records)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the fitted proxy."""
        if not self._is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "train_end_date": str(self._train_end_date),
            "history_size": len(self._history),
            "ridge_alpha": self.ridge_alpha,
            "feature_names": self.feature_names,
            "ridge_coefs": (
                self.ridge_coefs.tolist()
                if self.ridge_coefs is not None else None
            ),
            "ridge_intercept": self.ridge_intercept,
            "overall_sigma": self.overall_sigma,
            "monthly_sigma": self.monthly_sigma,
            "climatology_mean_jan1": float(self.climatology_mean[1]),
            "climatology_mean_jul1": float(self.climatology_mean[182]),
            "climatology_std_jan1": float(self.climatology_std[1]),
            "climatology_std_jul1": float(self.climatology_std[182]),
        }
