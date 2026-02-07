"""
Baseline Models for NYC Temperature Prediction.

Provides four baseline models for comparison with the neural network:
  1. PersistenceModel     -- predict tomorrow's TMAX = today's actual TMAX
  2. ClimatologyModel     -- predict TMAX from historical day-of-year average
  3. LinearRegressionModel -- OLS on the full scaled feature set
  4. RidgeRegressionModel  -- L2-regularized regression (configurable alpha)

All models share a common interface:
  - fit(X_train, y_train) -- train/calibrate the model
  - predict(X, y_prev=None) -- generate predictions
  - name property -- return human-readable model name

Also includes:
  - compute_metrics() -- compute MAE, RMSE, MBE, R-squared
  - run_all_baselines() -- fit all models and evaluate on val/test sets
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

# Add project root to path so config is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Persistence Baseline
# ===========================================================================

class PersistenceModel:
    """Persistence (naive) baseline: predict TMAX(t) = TMAX(t-1).

    The simplest weather forecasting baseline. Tomorrow's maximum temperature
    is predicted to be the same as today's observed maximum temperature.

    This model does not use the feature matrix -- only the target time series.
    The ``fit`` method stores the training targets, and ``predict`` shifts
    the actual target series by one day to produce lag-1 predictions.

    Notes
    -----
    The persistence baseline is an *oracle* in the sense that it uses
    yesterday's actual observation (not yesterday's prediction).  This is the
    standard definition used in weather-forecast verification.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Return the model name."""
        return "Persistence"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PersistenceModel":
        """Store the training target series for later use.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (not used; kept for interface consistency).
        y_train : pd.Series
            Training target values with DatetimeIndex.

        Returns
        -------
        PersistenceModel
            Self, for method chaining.

        Raises
        ------
        TypeError
            If ``y_train`` is not a pandas Series.
        """
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series with a DatetimeIndex")

        self._y_train = y_train.copy()
        self._is_fitted = True

        logger.info(
            "PersistenceModel fit: stored %d training targets (last value: %.1f)",
            len(y_train),
            y_train.iloc[-1],
        )
        return self

    def predict(
        self, X: pd.DataFrame, y_prev: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Generate persistence (lag-1) predictions.

        Each prediction equals the actual observed TMAX from the previous day.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex.  Used only for its index
            (dates and length), not its feature values.
        y_prev : pd.Series, optional
            Actual target values from which to derive lag-1 predictions.
            Should include all actual values up to and including the dates
            being predicted, so that ``shift(1)`` produces correct lag values.

            Recommended usage:
              - Training (in-sample): ``y_prev=y_train`` or ``None``
              - Validation: ``y_prev=pd.concat([y_train, y_val])``
              - Test: ``y_prev=pd.concat([y_train, y_val, y_test])``

            If ``None``, the training data stored during ``fit()`` is used.

        Returns
        -------
        np.ndarray
            Array of predictions, one per row in X.  The first prediction
            may be ``NaN`` if no prior actual value is available.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("PersistenceModel must be fitted before predicting")

        if y_prev is None:
            y_prev = self._y_train

        n = len(X)
        if n == 0:
            return np.array([])

        dates = X.index

        # Shift the actual values by 1 position to get lag-1 predictions
        shifted = y_prev.shift(1)

        # Align to the prediction dates
        predictions = shifted.reindex(dates)

        # Boundary handling: if the first prediction is NaN, try to find the
        # last actual value that precedes the first prediction date.
        # This covers the case where y_prev includes preceding-split actuals
        # (e.g., pd.concat([y_train, y_val])) but shift(1) at the first date
        # of a new split might still be NaN if that date isn't in y_prev.
        if pd.isna(predictions.iloc[0]):
            prior = y_prev[y_prev.index < dates[0]]
            if len(prior) > 0:
                predictions.iloc[0] = prior.iloc[-1]

        result = predictions.values.astype(float)

        n_valid = int(np.sum(~np.isnan(result)))
        logger.info(
            "PersistenceModel predict: %d predictions (%d valid)", len(result), n_valid
        )
        return result


# ===========================================================================
# Climatological Average Baseline
# ===========================================================================

class ClimatologyModel:
    """Climatological average baseline: predict TMAX(t) from day-of-year mean.

    Uses the historical average TMAX for each calendar day (day-of-year
    1--366), computed exclusively from the training set, as the prediction
    for any date with that day-of-year.

    For days-of-year not observed in training (e.g., Feb 29 when the
    training period contains no leap year), the overall training-set mean
    is used as a fallback.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Return the model name."""
        return "Climatology"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ClimatologyModel":
        """Compute day-of-year TMAX averages from training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (not used; kept for interface consistency).
        y_train : pd.Series
            Training target values with DatetimeIndex.

        Returns
        -------
        ClimatologyModel
            Self, for method chaining.

        Raises
        ------
        TypeError
            If ``y_train`` is not a pandas Series.
        """
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series with a DatetimeIndex")

        # Compute day-of-year averages
        doy = y_train.index.dayofyear
        doy_means = y_train.groupby(doy).mean()
        self._doy_means = doy_means.to_dict()

        # Overall mean as fallback for unseen days-of-year
        self._overall_mean = float(y_train.mean())
        self._is_fitted = True

        logger.info(
            "ClimatologyModel fit: computed averages for %d unique days-of-year "
            "(overall mean: %.1f)",
            len(self._doy_means),
            self._overall_mean,
        )
        return self

    def predict(
        self, X: pd.DataFrame, y_prev: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Generate climatological predictions based on day-of-year.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex.  Used only for its index
            (dates), not its feature values.
        y_prev : pd.Series, optional
            Not used by this model.  Accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Array of predictions, one per row in X.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("ClimatologyModel must be fitted before predicting")

        n = len(X)
        if n == 0:
            return np.array([])

        doy = X.index.dayofyear
        predictions = np.array(
            [self._doy_means.get(d, self._overall_mean) for d in doy]
        )

        n_fallback = sum(1 for d in doy if d not in self._doy_means)
        if n_fallback > 0:
            logger.warning(
                "ClimatologyModel: %d predictions used overall-mean fallback "
                "(unseen day-of-year)",
                n_fallback,
            )

        logger.info("ClimatologyModel predict: %d predictions", len(predictions))
        return predictions


# ===========================================================================
# Linear Regression Baseline
# ===========================================================================

class LinearRegressionModel:
    """Ordinary Least Squares linear regression baseline.

    Uses sklearn's ``LinearRegression`` on the full scaled feature set.
    Serves as a simple parametric baseline to judge whether a neural network
    captures non-linear relationships in the data.
    """

    def __init__(self):
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Return the model name."""
        return "Linear Regression"

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "LinearRegressionModel":
        """Fit the OLS model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training feature matrix (scaled).
        y_train : pd.Series or np.ndarray
            Training target values.

        Returns
        -------
        LinearRegressionModel
            Self, for method chaining.
        """
        self._model = LinearRegression()

        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        self._model.fit(X, y)
        self._n_features = X.shape[1]
        self._is_fitted = True

        logger.info(
            "LinearRegressionModel fit: %d features, %d samples",
            X.shape[1],
            X.shape[0],
        )
        return self

    def predict(
        self, X: pd.DataFrame, y_prev: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Generate predictions using the fitted linear model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (scaled).
        y_prev : pd.Series, optional
            Not used by this model.  Accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Array of predictions, one per row in X.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LinearRegressionModel must be fitted before predicting"
            )

        X_arr = X.values if hasattr(X, "values") else np.asarray(X)

        if X_arr.shape[0] == 0:
            return np.array([])

        predictions = self._model.predict(X_arr)

        logger.info(
            "LinearRegressionModel predict: %d predictions (mean=%.1f, std=%.1f)",
            len(predictions),
            np.nanmean(predictions),
            np.nanstd(predictions),
        )
        return predictions

    @property
    def coefficients(self) -> np.ndarray:
        """Return the fitted model coefficients.

        Returns
        -------
        np.ndarray
            Model coefficients (one per feature).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing coefficients")
        return self._model.coef_

    @property
    def intercept(self) -> float:
        """Return the fitted model intercept.

        Returns
        -------
        float
            Model intercept.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing intercept")
        return float(self._model.intercept_)


# ===========================================================================
# Ridge Regression Baseline
# ===========================================================================

class RidgeRegressionModel:
    """Ridge (L2-regularized) regression baseline.

    Uses sklearn's ``Ridge`` on the full scaled feature set.  The
    regularization strength (``alpha``) is configurable and controls
    the trade-off between fitting the training data and keeping
    coefficient magnitudes small.

    Parameters
    ----------
    alpha : float
        Regularization strength.  Larger values mean stronger
        regularization.  Must be positive.  Default is 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._alpha = alpha
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Return the model name (includes alpha value)."""
        return f"Ridge (alpha={self._alpha})"

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "RidgeRegressionModel":
        """Fit the Ridge model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training feature matrix (scaled).
        y_train : pd.Series or np.ndarray
            Training target values.

        Returns
        -------
        RidgeRegressionModel
            Self, for method chaining.
        """
        self._model = Ridge(alpha=self._alpha)

        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        self._model.fit(X, y)
        self._n_features = X.shape[1]
        self._is_fitted = True

        logger.info(
            "RidgeRegressionModel fit: alpha=%.4f, %d features, %d samples",
            self._alpha,
            X.shape[1],
            X.shape[0],
        )
        return self

    def predict(
        self, X: pd.DataFrame, y_prev: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Generate predictions using the fitted Ridge model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (scaled).
        y_prev : pd.Series, optional
            Not used by this model.  Accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Array of predictions, one per row in X.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RidgeRegressionModel must be fitted before predicting"
            )

        X_arr = X.values if hasattr(X, "values") else np.asarray(X)

        if X_arr.shape[0] == 0:
            return np.array([])

        predictions = self._model.predict(X_arr)

        logger.info(
            "RidgeRegressionModel predict: %d predictions (mean=%.1f, std=%.1f)",
            len(predictions),
            np.nanmean(predictions),
            np.nanstd(predictions),
        )
        return predictions

    @property
    def coefficients(self) -> np.ndarray:
        """Return the fitted model coefficients.

        Returns
        -------
        np.ndarray
            Model coefficients (one per feature).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing coefficients")
        return self._model.coef_

    @property
    def intercept(self) -> float:
        """Return the fitted model intercept.

        Returns
        -------
        float
            Model intercept.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before accessing intercept")
        return float(self._model.intercept_)


# ===========================================================================
# Evaluation Metrics
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Rows where either the true or predicted value is NaN are excluded
    from all computations.

    Parameters
    ----------
    y_true : np.ndarray
        Actual (observed) values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
          - ``mae``  : Mean Absolute Error
          - ``rmse`` : Root Mean Squared Error
          - ``mbe``  : Mean Bias Error (positive = over-prediction)
          - ``r_squared`` : Coefficient of determination (R^2)

        All values are ``NaN`` if no valid pairs remain after NaN removal.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Exclude NaN in either array
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t = y_t[valid]
    y_p = y_p[valid]

    if len(y_t) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mbe": np.nan,
            "r_squared": np.nan,
        }

    errors = y_p - y_t
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mbe = float(np.mean(errors))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "mbe": mbe, "r_squared": r_squared}


# ===========================================================================
# Convenience Runner
# ===========================================================================

def run_all_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Fit all baseline models and evaluate on validation and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (scaled), with DatetimeIndex.
    X_val : pd.DataFrame
        Validation feature matrix (scaled), with DatetimeIndex.
    X_test : pd.DataFrame
        Test feature matrix (scaled), with DatetimeIndex.
    y_train : pd.Series
        Training target values (unscaled, in degrees F).
    y_val : pd.Series
        Validation target values.
    y_test : pd.Series
        Test target values.

    Returns
    -------
    dict
        Dictionary keyed by model name.  Each value is a dict containing:
          - ``model`` : the fitted model object
          - ``val_predictions`` : np.ndarray of validation predictions
          - ``test_predictions`` : np.ndarray of test predictions
          - ``val_metrics`` : dict of validation metrics (MAE, RMSE, etc.)
          - ``test_metrics`` : dict of test metrics
    """
    models = [
        PersistenceModel(),
        ClimatologyModel(),
        LinearRegressionModel(),
        RidgeRegressionModel(alpha=1.0),
    ]

    # Pre-build concatenated actuals for the persistence model
    y_prev_val = pd.concat([y_train, y_val])
    y_prev_test = pd.concat([y_train, y_val, y_test])

    results: dict = {}

    for model in models:
        logger.info("-" * 50)
        logger.info("Fitting: %s", model.name)
        logger.info("-" * 50)

        model.fit(X_train, y_train)

        # Generate predictions -- persistence needs the concatenated actuals;
        # other models ignore y_prev.
        if isinstance(model, PersistenceModel):
            val_preds = model.predict(X_val, y_prev=y_prev_val)
            test_preds = model.predict(X_test, y_prev=y_prev_test)
        else:
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

        # Compute metrics
        val_metrics = compute_metrics(y_val.values, val_preds)
        test_metrics = compute_metrics(y_test.values, test_preds)

        results[model.name] = {
            "model": model,
            "val_predictions": val_preds,
            "test_predictions": test_preds,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        logger.info(
            "  Val  MAE: %.2f | RMSE: %.2f | R2: %.4f",
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["r_squared"],
        )
        logger.info(
            "  Test MAE: %.2f | RMSE: %.2f | R2: %.4f",
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r_squared"],
        )

    # Summary table
    logger.info("=" * 70)
    logger.info("Baseline Results Summary")
    logger.info("=" * 70)
    logger.info(
        "%-25s | %8s | %8s | %8s | %8s",
        "Model",
        "Val MAE",
        "Val RMSE",
        "Test MAE",
        "Test RMSE",
    )
    logger.info("-" * 70)
    for name, res in results.items():
        logger.info(
            "%-25s | %8.2f | %8.2f | %8.2f | %8.2f",
            name,
            res["val_metrics"]["mae"],
            res["val_metrics"]["rmse"],
            res["test_metrics"]["mae"],
            res["test_metrics"]["rmse"],
        )
    logger.info("=" * 70)

    return results
