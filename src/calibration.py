"""
Calibration Pipeline for Probabilistic Temperature Forecasts.

Provides tools for assessing and improving the calibration of
heteroscedastic Gaussian (mu, sigma) temperature predictions:

  1. PIT (Probability Integral Transform) histogram analysis
  2. Reliability diagram (expected vs observed coverage)
  3. Isotonic regression calibration (global or per-season)
  4. Interval coverage assessment at multiple nominal levels
  5. CRPS (Continuous Ranked Probability Score) computation
  6. Sharpness assessment (prediction interval widths)
  7. Kalshi KXHIGHNY bucket probability mapping
  8. Comprehensive calibration report generation

A well-calibrated probabilistic forecast has PIT values that are
uniformly distributed, and interval coverages that match their
nominal levels.
"""

import os
import sys
import json
import logging
import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

# Use non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.seasons import SEASON_MAP as SHARED_SEASON_MAP, SEASON_ORDER as SHARED_SEASON_ORDER
from src.utils import to_numpy as _shared_to_numpy

# Apply a clean plot style, with graceful fallback
_PREFERRED_STYLE = "seaborn-v0_8-whitegrid"
if _PREFERRED_STYLE in plt.style.available:
    plt.style.use(_PREFERRED_STYLE)

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
# Meteorological season mapping (mirrors src/evaluate.py)
# ---------------------------------------------------------------------------
SEASON_MAP = SHARED_SEASON_MAP
SEASON_ORDER = SHARED_SEASON_ORDER

# Constants
_SQRT_PI = np.sqrt(np.pi)
_INV_SQRT_PI = 1.0 / _SQRT_PI
_INV_SQRT_2 = 1.0 / np.sqrt(2.0)


# ===========================================================================
# Helpers
# ===========================================================================

def _to_numpy(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array.

    Parameters
    ----------
    arr : array-like
        Input data (numpy array, pandas Series, or list).

    Returns
    -------
    np.ndarray
        1-D float64 array.
    """
    return _shared_to_numpy(arr)


def _validate_probabilistic_inputs(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align probabilistic prediction arrays.

    Drops entries where any of mu, sigma, or observations is NaN.
    Raises ValueError when lengths differ.

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations (must be positive).
    observations : array-like
        Actual observed values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Cleaned (mu, sigma, observations) arrays.

    Raises
    ------
    ValueError
        If input arrays have different lengths or if all entries are
        invalid after cleaning.
    """
    mu_arr = _to_numpy(mu)
    sigma_arr = _to_numpy(sigma)
    obs_arr = _to_numpy(observations)

    if not (mu_arr.shape[0] == sigma_arr.shape[0] == obs_arr.shape[0]):
        raise ValueError(
            f"Length mismatch: mu={mu_arr.shape[0]}, sigma={sigma_arr.shape[0]}, "
            f"observations={obs_arr.shape[0]}"
        )

    # Drop entries where any value is NaN
    valid = ~(np.isnan(mu_arr) | np.isnan(sigma_arr) | np.isnan(obs_arr))
    mu_arr = mu_arr[valid]
    sigma_arr = sigma_arr[valid]
    obs_arr = obs_arr[valid]

    return mu_arr, sigma_arr, obs_arr


def _get_season_for_month(month: int) -> str:
    """Return the season name for a given month number (1-12).

    Parameters
    ----------
    month : int
        Calendar month (1 = January, ..., 12 = December).

    Returns
    -------
    str
        Season name (e.g. "Winter (DJF)").
    """
    return SEASON_MAP[month]


# ===========================================================================
# 1. PIT (Probability Integral Transform) Histogram
# ===========================================================================

def compute_pit_values(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
) -> np.ndarray:
    """Compute PIT (Probability Integral Transform) values.

    PIT_i = Phi((y_i - mu_i) / sigma_i), where Phi is the standard
    normal CDF. For a well-calibrated Gaussian model, PIT values are
    uniformly distributed on [0, 1].

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations (positive).
    observations : array-like
        Actual observed values.

    Returns
    -------
    np.ndarray
        1-D array of PIT values in [0, 1].

    Raises
    ------
    ValueError
        If input arrays have different lengths.
    """
    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    if len(mu_arr) == 0:
        logger.warning("Empty arrays after NaN removal — returning empty PIT")
        return np.array([], dtype=np.float64)

    # Clamp sigma to avoid division by zero
    sigma_arr = np.maximum(sigma_arr, 1e-10)

    z = (obs_arr - mu_arr) / sigma_arr
    pit_values = stats.norm.cdf(z)

    logger.info("Computed %d PIT values (mean=%.3f, std=%.3f)",
                len(pit_values), pit_values.mean(), pit_values.std())
    return pit_values


def pit_uniformity_test(pit_values: np.ndarray) -> dict:
    """Test PIT values for uniformity using the Kolmogorov-Smirnov test.

    Parameters
    ----------
    pit_values : np.ndarray
        1-D array of PIT values.

    Returns
    -------
    dict
        Dictionary with keys "ks_statistic", "p_value", and
        "is_uniform" (True if p > 0.05).
    """
    pit_values = _to_numpy(pit_values)

    if len(pit_values) < 2:
        logger.warning("Too few PIT values for KS test")
        return {
            "ks_statistic": float("nan"),
            "p_value": float("nan"),
            "is_uniform": False,
        }

    ks_stat, p_value = stats.kstest(pit_values, "uniform")

    result = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "is_uniform": bool(p_value > 0.05),
    }

    logger.info("PIT KS test: statistic=%.4f, p-value=%.4f, uniform=%s",
                ks_stat, p_value, result["is_uniform"])
    return result


def plot_pit_histogram(
    pit_values: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "PIT Histogram",
    show: bool = False,
) -> plt.Figure:
    """Plot a PIT histogram with a uniform reference line.

    A well-calibrated model produces bars of roughly equal height.
    U-shaped histograms indicate underdispersion (sigma too small);
    dome-shaped histograms indicate overdispersion.

    Parameters
    ----------
    pit_values : np.ndarray
        1-D array of PIT values in [0, 1].
    n_bins : int
        Number of histogram bins (default 10).
    save_path : str, optional
        File path to save the figure. If None, figure is not saved.
    title : str
        Plot title.
    show : bool
        If True, call plt.show() after saving.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    pit_values = _to_numpy(pit_values)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram of PIT values
    ax.hist(pit_values, bins=n_bins, range=(0, 1), density=True,
            edgecolor="white", alpha=0.75, color="#4c72b0",
            label="PIT histogram")

    # Uniform reference line
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
               label="Uniform reference")

    # KS test annotation
    if len(pit_values) >= 2:
        ks_result = pit_uniformity_test(pit_values)
        ax.text(0.02, 0.98,
                f"KS stat = {ks_result['ks_statistic']:.4f}\n"
                f"p-value = {ks_result['p_value']:.4f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.5))

    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved PIT histogram to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)

    return fig


# ===========================================================================
# 2. Reliability Diagram
# ===========================================================================

def compute_reliability(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
    nominal_levels: Optional[list[float]] = None,
) -> dict:
    """Compute expected vs observed coverage for a reliability diagram.

    For each nominal level p, compute the fraction of observations
    that fall within the central p% prediction interval of the
    Gaussian(mu, sigma) distribution.

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations.
    observations : array-like
        Actual observed values.
    nominal_levels : list[float], optional
        Nominal coverage levels (e.g. [0.1, 0.2, ..., 0.9]).
        If None, defaults to [0.1, 0.2, ..., 0.9].

    Returns
    -------
    dict
        Dictionary with keys:
        - "nominal_levels": list of nominal coverage fractions
        - "observed_coverages": list of observed coverage fractions
        - "n_samples": int, number of samples used
    """
    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    if nominal_levels is None:
        nominal_levels = [0.1 * i for i in range(1, 10)]

    n = len(mu_arr)
    if n == 0:
        logger.warning("Empty arrays — returning empty reliability data")
        return {
            "nominal_levels": nominal_levels,
            "observed_coverages": [float("nan")] * len(nominal_levels),
            "n_samples": 0,
        }

    sigma_arr = np.maximum(sigma_arr, 1e-10)

    observed_coverages = []
    for level in nominal_levels:
        # For central p% interval: alpha = (1-p)/2
        alpha = (1.0 - level) / 2.0
        z_alpha = stats.norm.ppf(1.0 - alpha)  # e.g. 1.96 for 95%

        lower = mu_arr - z_alpha * sigma_arr
        upper = mu_arr + z_alpha * sigma_arr

        covered = np.mean((obs_arr >= lower) & (obs_arr <= upper))
        observed_coverages.append(float(covered))

    result = {
        "nominal_levels": nominal_levels,
        "observed_coverages": observed_coverages,
        "n_samples": n,
    }

    logger.info("Computed reliability at %d nominal levels for %d samples",
                len(nominal_levels), n)
    return result


def plot_reliability_diagram(
    reliability_data: dict,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
    show: bool = False,
) -> plt.Figure:
    """Plot a reliability diagram (expected vs observed coverage).

    A perfectly calibrated model lies on the diagonal.

    Parameters
    ----------
    reliability_data : dict
        Output from compute_reliability().
    save_path : str, optional
        File path to save the figure.
    title : str
        Plot title.
    show : bool
        If True, call plt.show() after saving.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    nominal = reliability_data["nominal_levels"]
    observed = reliability_data["observed_coverages"]
    n = reliability_data["n_samples"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Confidence bands (approximate binomial 95% CI)
    if n > 0:
        nominal_arr = np.array(nominal)
        se = np.sqrt(nominal_arr * (1 - nominal_arr) / n)
        ax.fill_between(nominal_arr,
                        nominal_arr - 1.96 * se,
                        nominal_arr + 1.96 * se,
                        alpha=0.15, color="gray",
                        label="95% confidence band")

    # Observed coverage
    ax.plot(nominal, observed, "o-", color="#d62728", linewidth=2,
            markersize=6, label="Model")

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved reliability diagram to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)

    return fig


# ===========================================================================
# 2b. Seasonal Reliability Breakdown
# ===========================================================================

def compute_seasonal_reliability(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
    dates: Union[pd.DatetimeIndex, np.ndarray, list],
    nominal_levels: Optional[list[float]] = None,
    min_samples: int = 10,
) -> dict[str, dict]:
    """Compute reliability diagrams broken down by meteorological season.

    Splits the data by season (DJF, MAM, JJA, SON) and calls
    ``compute_reliability()`` on each subset.  Seasons with fewer than
    *min_samples* observations are silently skipped.

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations (positive).
    observations : array-like
        Actual observed values.
    dates : DatetimeIndex or array-like
        Dates corresponding to each prediction.  Used to assign
        observations to meteorological seasons.
    nominal_levels : list[float], optional
        Nominal coverage levels passed through to
        ``compute_reliability()``.  If None, the default levels are used.
    min_samples : int
        Minimum number of samples required to compute reliability for
        a given season (default 10).

    Returns
    -------
    dict[str, dict]
        Dictionary keyed by season name (e.g. "Winter (DJF)"), each
        containing the output of ``compute_reliability()`` for that
        season's subset.  Seasons with fewer than *min_samples*
        observations are omitted.
    """
    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    # Align dates length with cleaned arrays
    if len(dates) != len(mu_arr):
        logger.warning(
            "Date array length (%d) does not match cleaned input length (%d); "
            "cannot compute seasonal reliability",
            len(dates), len(mu_arr),
        )
        return {}

    months = dates.month
    result: dict[str, dict] = {}

    for season_name in SEASON_ORDER:
        season_months = [m for m, s in SEASON_MAP.items() if s == season_name]
        mask = np.isin(months, season_months)
        n_season = int(mask.sum())

        if n_season < min_samples:
            logger.info(
                "Skipping season '%s' for reliability: only %d samples (min=%d)",
                season_name, n_season, min_samples,
            )
            continue

        rel = compute_reliability(
            mu_arr[mask], sigma_arr[mask], obs_arr[mask],
            nominal_levels=nominal_levels,
        )
        result[season_name] = rel

    logger.info(
        "Computed seasonal reliability for %d/%d seasons",
        len(result), len(SEASON_ORDER),
    )
    return result


# ===========================================================================
# 3. Isotonic Regression Calibration
# ===========================================================================

class IsotonicCalibrator:
    """Isotonic regression calibrator for Gaussian probabilistic forecasts.

    Fits isotonic regression to PIT values (CDF outputs) to correct
    systematic calibration errors. Can operate globally or per-season.

    The isotonic regression maps raw CDF values to calibrated CDF values,
    enforcing monotonicity. After calibration, mu and sigma are adjusted
    so that the calibrated CDF matches the isotonic mapping.

    Parameters
    ----------
    seasonal : bool
        If True, fit separate isotonic regressions for each
        meteorological season (DJF, MAM, JJA, SON). Default False.

    Examples
    --------
    >>> calibrator = IsotonicCalibrator()
    >>> calibrator.fit(pit_values_train, pit_values_train)
    >>> mu_cal, sigma_cal = calibrator.calibrate(mu_test, sigma_test)
    """

    def __init__(self, seasonal: bool = False):
        self.seasonal = seasonal
        self._global_model: Optional[IsotonicRegression] = None
        self._seasonal_models: dict[str, IsotonicRegression] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fit."""
        return self._fitted

    def fit(
        self,
        pit_values: np.ndarray,
        target_cdf: Optional[np.ndarray] = None,
        dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
    ) -> "IsotonicCalibrator":
        """Fit the isotonic regression calibrator.

        Parameters
        ----------
        pit_values : np.ndarray
            Raw PIT values from the model on a calibration set.
        target_cdf : np.ndarray, optional
            Target CDF values. If None, uses empirical CDF of the
            PIT values (i.e., maps PIT to the uniform distribution).
        dates : DatetimeIndex or array-like, optional
            Required when seasonal=True. Dates corresponding to each
            PIT value for seasonal grouping.

        Returns
        -------
        IsotonicCalibrator
            Self, for method chaining.

        Raises
        ------
        ValueError
            If seasonal=True but dates is not provided, or if
            lengths mismatch.
        """
        pit_values = _to_numpy(pit_values)

        if len(pit_values) == 0:
            raise ValueError("Cannot fit calibrator on empty PIT values")

        if target_cdf is None:
            # Map to uniform: sort PIT and assign uniform quantiles
            sorted_idx = np.argsort(pit_values)
            target_cdf = np.zeros_like(pit_values)
            target_cdf[sorted_idx] = (np.arange(len(pit_values)) + 0.5) / len(pit_values)
        else:
            target_cdf = _to_numpy(target_cdf)
            if len(target_cdf) != len(pit_values):
                raise ValueError(
                    f"Length mismatch: pit_values={len(pit_values)}, "
                    f"target_cdf={len(target_cdf)}"
                )

        if self.seasonal:
            if dates is None:
                raise ValueError("dates must be provided when seasonal=True")
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)
            if len(dates) != len(pit_values):
                raise ValueError(
                    f"Length mismatch: pit_values={len(pit_values)}, "
                    f"dates={len(dates)}"
                )

            months = dates.month
            for season_name in SEASON_ORDER:
                season_months = [m for m, s in SEASON_MAP.items()
                                 if s == season_name]
                mask = np.isin(months, season_months)

                if mask.sum() < 2:
                    logger.warning(
                        "Season %s has fewer than 2 samples, using global fit",
                        season_name
                    )
                    continue

                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                iso.fit(pit_values[mask], target_cdf[mask])
                self._seasonal_models[season_name] = iso

            # Also fit a global model as fallback
            self._global_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._global_model.fit(pit_values, target_cdf)

        else:
            self._global_model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._global_model.fit(pit_values, target_cdf)

        self._fitted = True
        logger.info(
            "Fitted isotonic calibrator (seasonal=%s) on %d samples",
            self.seasonal, len(pit_values)
        )
        return self

    def calibrate_cdf(
        self,
        pit_values: np.ndarray,
        dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
    ) -> np.ndarray:
        """Transform PIT values through the fitted isotonic regression.

        Parameters
        ----------
        pit_values : np.ndarray
            Raw PIT values to calibrate.
        dates : DatetimeIndex or array-like, optional
            Required for seasonal calibration.

        Returns
        -------
        np.ndarray
            Calibrated PIT values.

        Raises
        ------
        RuntimeError
            If the calibrator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        pit_values = _to_numpy(pit_values)

        if len(pit_values) == 0:
            return np.array([], dtype=np.float64)

        if self.seasonal and dates is not None:
            if not isinstance(dates, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(dates)

            calibrated = np.zeros_like(pit_values)
            months = dates.month

            for season_name in SEASON_ORDER:
                season_months = [m for m, s in SEASON_MAP.items()
                                 if s == season_name]
                mask = np.isin(months, season_months)

                if mask.sum() == 0:
                    continue

                model = self._seasonal_models.get(
                    season_name, self._global_model
                )
                calibrated[mask] = model.transform(pit_values[mask])

            return calibrated
        else:
            return self._global_model.transform(pit_values)

    def calibrate(
        self,
        mu: Union[np.ndarray, pd.Series, list],
        sigma: Union[np.ndarray, pd.Series, list],
        dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply calibration to new predictions, returning adjusted mu and sigma.

        Adjusts predictions so that the calibrated CDF at the median
        and at +/- 1 sigma matches the isotonic-corrected values.

        The approach:
        1. Compute PIT at several z-score reference points.
        2. Apply isotonic calibration to those PITs.
        3. Invert to get new z-scores and solve for adjusted mu, sigma.

        Parameters
        ----------
        mu : array-like
            Predicted means.
        sigma : array-like
            Predicted standard deviations.
        dates : DatetimeIndex or array-like, optional
            Required for seasonal calibration.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (calibrated_mu, calibrated_sigma).

        Raises
        ------
        RuntimeError
            If the calibrator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        mu_arr = _to_numpy(mu)
        sigma_arr = _to_numpy(sigma)
        sigma_arr = np.maximum(sigma_arr, 1e-10)

        n = len(mu_arr)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Compute PIT at mu (z=0) and at mu+sigma (z=1)
        pit_at_median = stats.norm.cdf(0.0) * np.ones(n)  # 0.5 for all
        pit_at_plus1 = stats.norm.cdf(1.0) * np.ones(n)   # ~0.8413

        # Calibrate both PIT reference points
        cal_median = self.calibrate_cdf(pit_at_median, dates)
        cal_plus1 = self.calibrate_cdf(pit_at_plus1, dates)

        # Clamp to avoid extreme z-scores
        cal_median = np.clip(cal_median, 1e-6, 1 - 1e-6)
        cal_plus1 = np.clip(cal_plus1, 1e-6, 1 - 1e-6)

        # Invert to z-scores
        z_median = stats.norm.ppf(cal_median)
        z_plus1 = stats.norm.ppf(cal_plus1)

        # The original z=0 maps to new z_median, z=1 maps to z_plus1
        # Original physical value at z=0: mu
        # Original physical value at z=1: mu + sigma
        # New distribution: new_mu + new_sigma * z_median = mu
        #                   new_mu + new_sigma * z_plus1  = mu + sigma
        # Solving:
        dz = z_plus1 - z_median
        # Guard against degenerate case where isotonic maps both to same value
        dz = np.where(np.abs(dz) < 1e-8, 1.0, dz)

        calibrated_sigma = sigma_arr / dz
        calibrated_mu = mu_arr - calibrated_sigma * z_median

        # Ensure sigma remains positive
        calibrated_sigma = np.maximum(calibrated_sigma, 1e-6)

        logger.info(
            "Calibrated %d predictions: mean sigma %.2f -> %.2f",
            n, sigma_arr.mean(), calibrated_sigma.mean()
        )

        return calibrated_mu, calibrated_sigma

    def save(self, filepath: str) -> None:
        """Save the calibrator to disk.

        Parameters
        ----------
        filepath : str
            Path to save the calibrator (pickle format).
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "seasonal": self.seasonal,
                "global_model": self._global_model,
                "seasonal_models": self._seasonal_models,
                "fitted": self._fitted,
            }, f)
        logger.info("Saved calibrator to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsotonicCalibrator":
        """Load a calibrator from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved calibrator.

        Returns
        -------
        IsotonicCalibrator
            The loaded calibrator instance.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        calibrator = cls(seasonal=data["seasonal"])
        calibrator._global_model = data["global_model"]
        calibrator._seasonal_models = data["seasonal_models"]
        calibrator._fitted = data["fitted"]

        logger.info("Loaded calibrator from %s (seasonal=%s, fitted=%s)",
                    filepath, calibrator.seasonal, calibrator._fitted)
        return calibrator


class RegimeConditionalCalibrator:
    """Calibrate separately within each weather *regime*, with a global fallback.

    Generalizes :class:`IsotonicCalibrator(seasonal=True)` from the four
    meteorological seasons to arbitrary discrete regime labels — e.g. a
    fall frontal-passage indicator, a season x frontal cross, or a
    volatility bin.  A regime with too few calibration samples falls back to
    the pooled global calibrator, so fragmenting the data is safe.

    Implemented by composition: one non-seasonal :class:`IsotonicCalibrator`
    per regime plus one global calibrator, reusing the tested mu/sigma
    inversion math rather than reimplementing it.

    Parameters
    ----------
    min_samples : int
        Minimum calibration samples for a regime to get its own calibrator
        (default 30).  Below this, the regime uses the global fit.
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._regime_models: dict[object, IsotonicCalibrator] = {}
        self._global: Optional[IsotonicCalibrator] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def regimes(self) -> list:
        """Regime labels that received a dedicated calibrator."""
        return list(self._regime_models.keys())

    def fit(
        self,
        mu: Union[np.ndarray, pd.Series, list],
        sigma: Union[np.ndarray, pd.Series, list],
        observations: Union[np.ndarray, pd.Series, list],
        regimes: Union[np.ndarray, pd.Series, list],
    ) -> "RegimeConditionalCalibrator":
        """Fit per-regime and global calibrators on a calibration set.

        ``regimes`` is an array of hashable labels aligned with the
        predictions.  PIT values are computed internally from (mu, sigma, obs).
        """
        mu_arr, sigma_arr = _to_numpy(mu), np.maximum(_to_numpy(sigma), 1e-10)
        obs_arr = _to_numpy(observations)
        regime_arr = np.asarray(regimes, dtype=object)
        if not (len(mu_arr) == len(sigma_arr) == len(obs_arr) == len(regime_arr)):
            raise ValueError("mu, sigma, observations, regimes must be equal length")
        if len(mu_arr) == 0:
            raise ValueError("Cannot fit on empty calibration set")

        pit = compute_pit_values(mu_arr, sigma_arr, obs_arr)

        self._global = IsotonicCalibrator(seasonal=False)
        self._global.fit(pit)

        self._regime_models = {}
        for label in pd.unique(regime_arr):
            mask = regime_arr == label
            if mask.sum() >= self.min_samples:
                cal = IsotonicCalibrator(seasonal=False)
                cal.fit(pit[mask])
                self._regime_models[label] = cal
            else:
                logger.info(
                    "Regime %r has %d < %d samples; using global fallback",
                    label, int(mask.sum()), self.min_samples,
                )
        self._fitted = True
        return self

    def calibrate(
        self,
        mu: Union[np.ndarray, pd.Series, list],
        sigma: Union[np.ndarray, pd.Series, list],
        regimes: Union[np.ndarray, pd.Series, list],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the per-regime calibrator (global fallback) to new predictions."""
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")
        mu_arr, sigma_arr = _to_numpy(mu), np.maximum(_to_numpy(sigma), 1e-10)
        regime_arr = np.asarray(regimes, dtype=object)
        cal_mu = np.empty_like(mu_arr, dtype=float)
        cal_sigma = np.empty_like(sigma_arr, dtype=float)
        for label in pd.unique(regime_arr):
            mask = regime_arr == label
            model = self._regime_models.get(label, self._global)
            cal_mu[mask], cal_sigma[mask] = model.calibrate(mu_arr[mask], sigma_arr[mask])
        return cal_mu, cal_sigma


# ===========================================================================
# 4. Interval Coverage Assessment
# ===========================================================================

def compute_interval_coverage(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
    levels: Optional[list[float]] = None,
) -> dict:
    """Compute prediction interval coverage at specified nominal levels.

    For each level p, computes what fraction of observations fall
    within the central p% prediction interval of Gaussian(mu, sigma).

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations.
    observations : array-like
        Actual observed values.
    levels : list[float], optional
        Nominal coverage levels as fractions (e.g. [0.50, 0.80, 0.90, 0.95]).
        Defaults to [0.50, 0.80, 0.90, 0.95].

    Returns
    -------
    dict
        Dictionary with keys:
        - "levels": list of nominal levels
        - "coverages": list of observed coverage fractions
        - "n_samples": number of samples used
    """
    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95]

    n = len(mu_arr)
    if n == 0:
        logger.warning("Empty arrays — returning NaN coverages")
        return {
            "levels": levels,
            "coverages": [float("nan")] * len(levels),
            "n_samples": 0,
        }

    sigma_arr = np.maximum(sigma_arr, 1e-10)

    coverages = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        z = stats.norm.ppf(1.0 - alpha)

        lower = mu_arr - z * sigma_arr
        upper = mu_arr + z * sigma_arr

        covered = float(np.mean((obs_arr >= lower) & (obs_arr <= upper)))
        coverages.append(covered)

    result = {
        "levels": levels,
        "coverages": coverages,
        "n_samples": n,
    }

    for level, cov in zip(levels, coverages):
        logger.info("Coverage at %.0f%%: %.1f%% (nominal: %.1f%%)",
                    level * 100, cov * 100, level * 100)

    return result


# ===========================================================================
# 5. CRPS Computation
# ===========================================================================

def compute_crps(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
) -> dict:
    """Compute CRPS (Continuous Ranked Probability Score) for Gaussian predictions.

    Uses the closed-form Gaussian CRPS formula:
        CRPS = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.

    Lower CRPS is better (0 = perfect deterministic forecast).

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations.
    observations : array-like
        Actual observed values.
    dates : DatetimeIndex or array-like, optional
        If provided, also computes seasonal CRPS breakdown.

    Returns
    -------
    dict
        Dictionary with keys:
        - "mean_crps": float, average CRPS across all samples
        - "crps_values": np.ndarray, per-sample CRPS
        - "n_samples": int
        - "seasonal_crps": dict (only if dates provided), mapping
          season name -> {"mean_crps": float, "n": int}
    """
    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    n = len(mu_arr)
    if n == 0:
        logger.warning("Empty arrays — returning NaN CRPS")
        return {
            "mean_crps": float("nan"),
            "crps_values": np.array([], dtype=np.float64),
            "n_samples": 0,
        }

    sigma_arr = np.maximum(sigma_arr, 1e-10)

    z = (obs_arr - mu_arr) / sigma_arr
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)

    crps_values = sigma_arr * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - _INV_SQRT_PI)
    mean_crps = float(np.mean(crps_values))

    result = {
        "mean_crps": mean_crps,
        "crps_values": crps_values,
        "n_samples": n,
    }

    # Seasonal breakdown
    if dates is not None:
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        if len(dates) != n:
            logger.warning(
                "dates length (%d) does not match data length (%d) - "
                "skipping seasonal breakdown", len(dates), n
            )
        else:
            months = dates.month
            seasonal_crps = {}
            for season_name in SEASON_ORDER:
                season_months = [m for m, s in SEASON_MAP.items()
                                 if s == season_name]
                mask = np.isin(months, season_months)
                if mask.sum() == 0:
                    continue
                seasonal_crps[season_name] = {
                    "mean_crps": float(np.mean(crps_values[mask])),
                    "n": int(mask.sum()),
                }
            result["seasonal_crps"] = seasonal_crps

    logger.info("Mean CRPS: %.4f (n=%d)", mean_crps, n)
    return result


# ===========================================================================
# 6. Sharpness Assessment
# ===========================================================================

def compute_sharpness(
    sigma: Union[np.ndarray, pd.Series, list],
    levels: Optional[list[float]] = None,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
) -> dict:
    """Measure average prediction interval width at various confidence levels.

    Sharpness measures how narrow the prediction intervals are.
    Narrower intervals are better, conditional on proper coverage.

    Parameters
    ----------
    sigma : array-like
        Predicted standard deviations.
    levels : list[float], optional
        Confidence levels to compute interval widths for.
        Defaults to [0.50, 0.80, 0.90, 0.95].
    dates : DatetimeIndex or array-like, optional
        If provided, also computes seasonal sharpness breakdown.

    Returns
    -------
    dict
        Dictionary with keys:
        - "levels": list of nominal levels
        - "mean_widths": list of average interval widths at each level
        - "mean_sigma": float, average sigma
        - "n_samples": int
        - "seasonal_sharpness": dict (only if dates provided), mapping
          season name -> {"mean_sigma": float, "n": int}
    """
    sigma_arr = _to_numpy(sigma)

    # Remove NaNs
    valid = ~np.isnan(sigma_arr)
    sigma_arr = sigma_arr[valid]

    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95]

    n = len(sigma_arr)
    if n == 0:
        logger.warning("Empty sigma array — returning NaN sharpness")
        return {
            "levels": levels,
            "mean_widths": [float("nan")] * len(levels),
            "mean_sigma": float("nan"),
            "n_samples": 0,
        }

    sigma_arr = np.maximum(sigma_arr, 1e-10)

    mean_widths = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        z = stats.norm.ppf(1.0 - alpha)
        width = 2.0 * z * sigma_arr
        mean_widths.append(float(np.mean(width)))

    result = {
        "levels": levels,
        "mean_widths": mean_widths,
        "mean_sigma": float(np.mean(sigma_arr)),
        "n_samples": n,
    }

    # Seasonal breakdown
    if dates is not None:
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        # Apply same NaN mask to dates
        dates_filtered = dates[valid] if len(dates) == len(valid) else dates
        if len(dates_filtered) == n:
            months = dates_filtered.month
            seasonal_sharpness = {}
            for season_name in SEASON_ORDER:
                season_months = [m for m, s in SEASON_MAP.items()
                                 if s == season_name]
                mask = np.isin(months, season_months)
                if mask.sum() == 0:
                    continue
                seasonal_sharpness[season_name] = {
                    "mean_sigma": float(np.mean(sigma_arr[mask])),
                    "n": int(mask.sum()),
                }
            result["seasonal_sharpness"] = seasonal_sharpness

    logger.info("Sharpness — mean sigma: %.3f, 95%% PI width: %.2f",
                result["mean_sigma"], mean_widths[-1] if mean_widths else 0)
    return result


# ===========================================================================
# 7a. Bucket Probability Validation
# ===========================================================================

def validate_bucket_probabilities(
    buckets: dict[str, float],
    tolerance: float = 0.01,
) -> dict:
    """Validate probability-sum and monotonicity of bucket probabilities.

    Enforces three invariants on the output of
    ``cdf_to_kalshi_buckets()``:

    1. All probabilities are non-negative.
    2. Probabilities sum to approximately 1.0 (within *tolerance*).
    3. CDF monotonicity: the cumulative sum of probabilities (in
       bucket order) is non-decreasing.

    Parameters
    ----------
    buckets : dict[str, float]
        Mapping of bucket label to probability, as returned by
        ``cdf_to_kalshi_buckets()``.
    tolerance : float
        Acceptable deviation of the probability sum from 1.0
        (default 0.01).

    Returns
    -------
    dict
        Validation results with keys:

        - ``is_valid`` (bool): True if all checks pass.
        - ``prob_sum`` (float): Sum of all bucket probabilities.
        - ``max_negative`` (float): Most-negative probability found
          (0.0 if none are negative).
        - ``sum_deviation`` (float): Absolute difference from 1.0.
        - ``monotonicity_violations`` (int): Number of positions where
          the cumulative sum decreased.
        - ``warnings`` (list[str]): Human-readable descriptions of
          any violations detected.
    """
    warnings_list: list[str] = []
    probs = np.array(list(buckets.values()), dtype=np.float64)

    # --- Non-negativity ---
    neg_mask = probs < 0.0
    max_negative = float(probs[neg_mask].min()) if neg_mask.any() else 0.0
    if neg_mask.any():
        n_neg = int(neg_mask.sum())
        msg = (
            f"Found {n_neg} negative bucket probabilit{'y' if n_neg == 1 else 'ies'} "
            f"(most negative: {max_negative:.6f})"
        )
        warnings_list.append(msg)
        logger.warning("validate_bucket_probabilities: %s", msg)

    # --- Probability sum ---
    prob_sum = float(probs.sum())
    sum_deviation = abs(prob_sum - 1.0)
    if sum_deviation > tolerance:
        msg = (
            f"Bucket probabilities sum to {prob_sum:.6f} "
            f"(deviation {sum_deviation:.6f} exceeds tolerance {tolerance})"
        )
        warnings_list.append(msg)
        logger.warning("validate_bucket_probabilities: %s", msg)

    # --- CDF monotonicity ---
    cumsum = np.cumsum(probs)
    diffs = np.diff(cumsum)
    mono_violations = int(np.sum(diffs < -1e-12))
    if mono_violations > 0:
        msg = (
            f"CDF monotonicity violated at {mono_violations} "
            f"position{'s' if mono_violations > 1 else ''}"
        )
        warnings_list.append(msg)
        logger.warning("validate_bucket_probabilities: %s", msg)

    is_valid = len(warnings_list) == 0

    return {
        "is_valid": is_valid,
        "prob_sum": prob_sum,
        "max_negative": max_negative,
        "sum_deviation": sum_deviation,
        "monotonicity_violations": mono_violations,
        "warnings": warnings_list,
    }


# ===========================================================================
# 7b. Kalshi Bucket Probability Mapping
# ===========================================================================

def cdf_to_kalshi_buckets(
    mu: float,
    sigma: float,
    bucket_width: int = 5,
    temp_range: tuple[int, int] = (0, 120),
    calibrator: Optional[IsotonicCalibrator] = None,
) -> dict[str, float]:
    """Convert a Gaussian (mu, sigma) prediction to Kalshi KXHIGHNY bucket probabilities.

    Kalshi buckets are integer degree-F ranges (e.g., "65-69 F").
    Computes P(bucket) = CDF(upper+0.5) - CDF(lower-0.5) for each bucket,
    where the 0.5 offset accounts for rounding of integer temperatures.

    Parameters
    ----------
    mu : float
        Predicted mean temperature (deg F).
    sigma : float
        Predicted standard deviation (deg F).
    bucket_width : int
        Width of each bucket in degrees F (default 5).
    temp_range : tuple[int, int]
        (min, max) temperature range to cover.
    calibrator : IsotonicCalibrator, optional
        If provided, use calibrated CDF instead of raw Gaussian.

    Returns
    -------
    dict[str, float]
        Mapping of bucket label (e.g. "65-69 F") to probability.
        Includes edge buckets "Below {min} F" and "Above {max-1} F".
    """
    sigma = max(sigma, 1e-10)
    lo, hi = temp_range

    buckets = {}

    # Below range bucket
    cdf_lo = stats.norm.cdf((lo - 0.5 - mu) / sigma)
    if calibrator is not None and calibrator.is_fitted:
        cdf_lo = float(calibrator.calibrate_cdf(np.array([cdf_lo]))[0])
    buckets[f"Below {lo} F"] = float(cdf_lo)

    # Interior buckets
    for bucket_start in range(lo, hi, bucket_width):
        bucket_end = bucket_start + bucket_width - 1

        cdf_upper = stats.norm.cdf((bucket_end + 0.5 - mu) / sigma)
        cdf_lower = stats.norm.cdf((bucket_start - 0.5 - mu) / sigma)

        if calibrator is not None and calibrator.is_fitted:
            cdf_upper = float(
                calibrator.calibrate_cdf(np.array([cdf_upper]))[0]
            )
            cdf_lower = float(
                calibrator.calibrate_cdf(np.array([cdf_lower]))[0]
            )

        prob = cdf_upper - cdf_lower
        buckets[f"{bucket_start}-{bucket_end} F"] = float(max(prob, 0.0))

    # Above range bucket
    cdf_hi = stats.norm.cdf((hi - 0.5 - mu) / sigma)
    if calibrator is not None and calibrator.is_fitted:
        cdf_hi = float(calibrator.calibrate_cdf(np.array([cdf_hi]))[0])
    buckets[f"Above {hi - 1} F"] = float(1.0 - cdf_hi)

    logger.info(
        "Kalshi buckets: mu=%.1f, sigma=%.1f, %d buckets, "
        "top bucket: %s (%.3f)",
        mu, sigma, len(buckets),
        max(buckets, key=buckets.get), max(buckets.values())
    )

    # Validate bucket probabilities before returning
    validation = validate_bucket_probabilities(buckets)
    if not validation["is_valid"]:
        logger.warning(
            "Bucket probability validation failed for mu=%.1f, sigma=%.1f: %s",
            mu, sigma, "; ".join(validation["warnings"]),
        )

    return buckets


# ===========================================================================
# 8. Comprehensive Calibration Report
# ===========================================================================

def generate_calibration_report(
    mu: Union[np.ndarray, pd.Series, list],
    sigma: Union[np.ndarray, pd.Series, list],
    observations: Union[np.ndarray, pd.Series, list],
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray, list]] = None,
    output_dir: str = "results/calibration",
    model_name: str = "Model",
) -> dict:
    """Generate a comprehensive calibration report.

    Runs all calibration diagnostics, generates a multi-panel figure,
    and saves summary metrics.

    Parameters
    ----------
    mu : array-like
        Predicted means.
    sigma : array-like
        Predicted standard deviations.
    observations : array-like
        Actual observed values.
    dates : DatetimeIndex or array-like, optional
        Dates for seasonal breakdown.
    output_dir : str
        Directory to save reports and plots.
    model_name : str
        Label for the model.

    Returns
    -------
    dict
        Dictionary with all calibration metrics:
        - "pit_ks_test": KS test results
        - "reliability": reliability diagram data
        - "seasonal_reliability": seasonal reliability breakdown
          (present only when *dates* are provided)
        - "nll": negative log-likelihood results (overall and seasonal)
        - "coverage": interval coverage data
        - "crps": CRPS results
        - "sharpness": sharpness results
        - "model_name": str
    """
    os.makedirs(output_dir, exist_ok=True)

    mu_arr, sigma_arr, obs_arr = _validate_probabilistic_inputs(
        mu, sigma, observations
    )

    n = len(mu_arr)
    logger.info("Generating calibration report for '%s' (%d samples)",
                model_name, n)

    # 1. PIT analysis
    pit_values = compute_pit_values(mu_arr, sigma_arr, obs_arr)
    ks_result = pit_uniformity_test(pit_values)

    pit_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_pit_histogram.png")
    plot_pit_histogram(pit_values, save_path=pit_path,
                       title=f"{model_name}: PIT Histogram")

    # 2. Reliability diagram
    reliability = compute_reliability(mu_arr, sigma_arr, obs_arr)

    rel_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_reliability.png")
    plot_reliability_diagram(reliability, save_path=rel_path,
                             title=f"{model_name}: Reliability Diagram")

    # 2b. Seasonal reliability (when dates are available)
    seasonal_reliability: Optional[dict] = None
    dates_idx: Optional[pd.DatetimeIndex] = None
    if dates is not None:
        if not isinstance(dates, pd.DatetimeIndex):
            dates_idx = pd.DatetimeIndex(dates)
        else:
            dates_idx = dates
        # Only attempt if date length matches cleaned arrays
        if len(dates_idx) == n:
            seasonal_reliability = compute_seasonal_reliability(
                mu_arr, sigma_arr, obs_arr, dates_idx,
            )
        else:
            logger.warning(
                "Date array length (%d) does not match cleaned sample count (%d); "
                "skipping seasonal reliability",
                len(dates_idx), n,
            )

    # 2c. NLL (Negative Log-Likelihood) for Gaussian predictions
    sigma_safe = np.maximum(sigma_arr, 1e-10)
    nll_values = (
        0.5 * np.log(2.0 * np.pi * sigma_safe ** 2)
        + (obs_arr - mu_arr) ** 2 / (2.0 * sigma_safe ** 2)
    )
    nll_result: dict = {
        "mean_nll": float(np.mean(nll_values)),
        "median_nll": float(np.median(nll_values)),
        "n_samples": n,
    }

    # Seasonal NLL breakdown
    if dates_idx is not None and len(dates_idx) == n:
        months = dates_idx.month
        seasonal_nll: dict[str, dict] = {}
        for season_name in SEASON_ORDER:
            season_months = [m for m, s in SEASON_MAP.items()
                             if s == season_name]
            mask = np.isin(months, season_months)
            n_season = int(mask.sum())
            if n_season == 0:
                continue
            seasonal_nll[season_name] = {
                "mean_nll": float(np.mean(nll_values[mask])),
                "median_nll": float(np.median(nll_values[mask])),
                "n_samples": n_season,
            }
        nll_result["seasonal"] = seasonal_nll

    logger.info("NLL — overall mean: %.4f", nll_result["mean_nll"])

    # 3. Interval coverage
    coverage = compute_interval_coverage(mu_arr, sigma_arr, obs_arr)

    # 4. CRPS
    crps_result = compute_crps(mu_arr, sigma_arr, obs_arr, dates=dates)

    # 5. Sharpness
    sharpness = compute_sharpness(sigma_arr, dates=dates)

    # 6. Multi-panel figure
    _plot_calibration_summary(
        pit_values, reliability, coverage, sharpness,
        model_name=model_name,
        save_path=os.path.join(output_dir,
                               f"{model_name.lower().replace(' ', '_')}_calibration_summary.png"),
    )

    # 7. Save metrics to CSV
    metrics_summary = {
        "model_name": model_name,
        "n_samples": n,
        "ks_statistic": ks_result["ks_statistic"],
        "ks_p_value": ks_result["p_value"],
        "pit_is_uniform": ks_result["is_uniform"],
        "mean_crps": crps_result["mean_crps"],
        "mean_sigma": sharpness["mean_sigma"],
        "mean_nll": nll_result["mean_nll"],
        "median_nll": nll_result["median_nll"],
    }
    for level, cov in zip(coverage["levels"], coverage["coverages"]):
        metrics_summary[f"coverage_{int(level*100)}pct"] = cov

    # Add seasonal NLL columns to CSV
    if "seasonal" in nll_result:
        for season_name, snll in nll_result["seasonal"].items():
            safe_key = season_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            metrics_summary[f"nll_{safe_key}"] = snll["mean_nll"]

    csv_path = os.path.join(output_dir,
                            f"{model_name.lower().replace(' ', '_')}_calibration_metrics.csv")
    pd.DataFrame([metrics_summary]).to_csv(csv_path, index=False)
    logger.info("Saved calibration metrics to %s", csv_path)

    report = {
        "pit_ks_test": ks_result,
        "reliability": reliability,
        "coverage": coverage,
        "crps": crps_result,
        "sharpness": sharpness,
        "nll": nll_result,
        "model_name": model_name,
    }

    if seasonal_reliability is not None:
        report["seasonal_reliability"] = seasonal_reliability

    logger.info("Calibration report complete for '%s'", model_name)
    return report


def _plot_calibration_summary(
    pit_values: np.ndarray,
    reliability: dict,
    coverage: dict,
    sharpness: dict,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Generate a 2x2 multi-panel calibration summary figure.

    Panels:
      - Top-left: PIT histogram
      - Top-right: Reliability diagram
      - Bottom-left: Coverage plot (nominal vs observed)
      - Bottom-right: Sharpness (interval width by level)

    Parameters
    ----------
    pit_values : np.ndarray
        PIT values.
    reliability : dict
        Output from compute_reliability().
    coverage : dict
        Output from compute_interval_coverage().
    sharpness : dict
        Output from compute_sharpness().
    model_name : str
        Label for the model.
    save_path : str, optional
        Path to save the figure.
    show : bool
        If True, call plt.show() after saving.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{model_name}: Calibration Summary", fontsize=14, y=1.02)

    # Panel 1: PIT histogram
    ax = axes[0, 0]
    ax.hist(pit_values, bins=10, range=(0, 1), density=True,
            edgecolor="white", alpha=0.75, color="#4c72b0")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("PIT Histogram")
    ax.set_xlim(0, 1)

    # Panel 2: Reliability diagram
    ax = axes[0, 1]
    nominal = reliability["nominal_levels"]
    observed = reliability["observed_coverages"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.plot(nominal, observed, "o-", color="#d62728", linewidth=2, markersize=5)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    # Panel 3: Coverage plot
    ax = axes[1, 0]
    cov_levels = coverage["levels"]
    cov_values = coverage["coverages"]
    x_pos = np.arange(len(cov_levels))
    bar_width = 0.35
    bars1 = ax.bar(x_pos - bar_width / 2, cov_levels, bar_width,
                   label="Nominal", color="#4c72b0", alpha=0.7)
    bars2 = ax.bar(x_pos + bar_width / 2, cov_values, bar_width,
                   label="Observed", color="#d62728", alpha=0.7)
    ax.set_xlabel("Coverage Level")
    ax.set_ylabel("Coverage")
    ax.set_title("Interval Coverage")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(l*100)}%" for l in cov_levels])
    ax.legend()

    # Panel 4: Sharpness (interval widths)
    ax = axes[1, 1]
    sharp_levels = sharpness["levels"]
    sharp_widths = sharpness["mean_widths"]
    ax.bar(range(len(sharp_levels)), sharp_widths,
           color="#2ca02c", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Mean Interval Width (deg F)")
    ax.set_title("Sharpness")
    ax.set_xticks(range(len(sharp_levels)))
    ax.set_xticklabels([f"{int(l*100)}%" for l in sharp_levels])

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved calibration summary to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)

    return fig
