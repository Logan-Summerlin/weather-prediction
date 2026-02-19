"""
Kalshi KXHIGHNY Out-of-Sample Backtesting Module.

Provides reusable components for simulating Kalshi markets, generating
model predictions, and analyzing backtest performance across in-sample
and out-of-sample periods:

  1. KalshiMarketSimulator: Generate realistic KXHIGHNY bucket contracts
     with market prices, bid-ask spreads, and volume.
  2. ModelPredictionGenerator: Generate Gaussian (mu, sigma) predictions
     with realistic NYC seasonal temperature patterns.
  3. BacktestAnalyzer: Analyze Brier scores, edge persistence, strategy
     selection, and generate comprehensive reports.
  4. CalibrationAnalyzer: Reliability diagrams, PIT histograms, and
     per-season calibration assessment.

All classes are designed to work with the existing TradingStrategy,
BacktestEngine, and BacktestResult infrastructure in src/trading.py.
"""

import os
import sys
import json
import logging
import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

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
# Meteorological season mapping (mirrors src/trading.py)
# ---------------------------------------------------------------------------
SEASON_MAP = SHARED_SEASON_MAP
SEASON_ORDER = SHARED_SEASON_ORDER

# ---------------------------------------------------------------------------
# Constants for realistic NYC temperature simulation
# ---------------------------------------------------------------------------
# Monthly mean TMAX (deg F) for NYC Central Park (approximate climatology)
NYC_MONTHLY_TMAX_MEAN = {
    1: 39.0, 2: 42.0, 3: 50.0, 4: 62.0, 5: 72.0, 6: 80.0,
    7: 85.0, 8: 84.0, 9: 76.0, 10: 65.0, 11: 54.0, 12: 43.0,
}
# Monthly std dev of daily TMAX (higher in winter, lower in summer)
NYC_MONTHLY_TMAX_STD = {
    1: 11.0, 2: 10.5, 3: 10.0, 4: 9.5, 5: 8.0, 6: 6.5,
    7: 5.5, 8: 5.5, 9: 6.5, 10: 8.0, 11: 9.5, 12: 10.5,
}

# Canonical KXHIGHNY bucket definitions (2°F resolution from city_config)
from src.city_config import get_city_config

_nyc_cfg = get_city_config("nyc")
KXHIGHNY_BUCKET_EDGES = _nyc_cfg.bucket_edges
KXHIGHNY_BUCKET_LABELS = _nyc_cfg.bucket_labels


def _to_numpy(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array."""
    return _shared_to_numpy(arr)


def _get_season(month: int) -> str:
    """Return the season name for a given month number (1-12)."""
    return SEASON_MAP[month]


# ===========================================================================
# 1. KalshiMarketSimulator
# ===========================================================================

class KalshiMarketSimulator:
    """Generate realistic simulated KXHIGHNY market data.

    Creates daily bucket contracts with market-implied probabilities that
    are informed by the actual temperature but include realistic pricing
    noise, bid-ask spreads, and volume patterns.

    Parameters
    ----------
    bucket_edges : list[tuple[float, float]], optional
        List of (low, high) temperature edges for each bucket.
        Defaults to standard KXHIGHNY 10-degree buckets.
    bucket_labels : list[str], optional
        Labels for each bucket. Must match len(bucket_edges).
    market_noise_std : float
        Standard deviation of noise added to true probabilities to
        simulate market inefficiency (default 0.06).
    min_spread : float
        Minimum bid-ask spread in probability units (default 0.02).
    max_spread : float
        Maximum bid-ask spread for illiquid buckets (default 0.10).

    Examples
    --------
    >>> sim = KalshiMarketSimulator()
    >>> buckets = sim.generate_daily_buckets(
    ...     date=datetime.date(2025, 7, 15),
    ...     actual_tmax=88.0, model_mu=86.0, model_sigma=4.5,
    ... )
    >>> len(buckets) == 10
    True
    """

    def __init__(
        self,
        bucket_edges: Optional[list] = None,
        bucket_labels: Optional[list] = None,
        market_noise_std: float = 0.06,
        min_spread: float = 0.02,
        max_spread: float = 0.10,
    ):
        self.bucket_edges = bucket_edges or list(KXHIGHNY_BUCKET_EDGES)
        self.bucket_labels = bucket_labels or list(KXHIGHNY_BUCKET_LABELS)
        if len(self.bucket_edges) != len(self.bucket_labels):
            raise ValueError(
                f"bucket_edges length ({len(self.bucket_edges)}) must match "
                f"bucket_labels length ({len(self.bucket_labels)})"
            )
        self.market_noise_std = market_noise_std
        self.min_spread = min_spread
        self.max_spread = max_spread

    def _compute_true_bucket_prob(
        self,
        lo: float,
        hi: float,
        actual_mu: float,
        actual_sigma: float,
    ) -> float:
        """Compute the true probability of TMAX falling in [lo, hi).

        Uses a Gaussian centered on the market's best estimate of TMAX,
        which is slightly noisier than the model's estimate.

        Parameters
        ----------
        lo : float
            Lower bound of the bucket.
        hi : float
            Upper bound of the bucket.
        actual_mu : float
            Market's center estimate for TMAX.
        actual_sigma : float
            Market's uncertainty for TMAX.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        actual_sigma = max(actual_sigma, 1e-10)
        if lo <= -900:
            return float(stats.norm.cdf(hi, loc=actual_mu, scale=actual_sigma))
        if hi >= 900:
            return float(1.0 - stats.norm.cdf(lo, loc=actual_mu, scale=actual_sigma))
        return float(
            stats.norm.cdf(hi, loc=actual_mu, scale=actual_sigma)
            - stats.norm.cdf(lo, loc=actual_mu, scale=actual_sigma)
        )

    def generate_daily_buckets(
        self,
        date: Union[datetime.date, str],
        actual_tmax: float,
        model_mu: float,
        model_sigma: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> list[dict]:
        """Generate bucket contracts for a single day.

        Parameters
        ----------
        date : date or str
            The trading date.
        actual_tmax : float
            The actual observed TMAX for the day (used for settlement).
        model_mu : float
            Model's predicted mean TMAX.
        model_sigma : float
            Model's predicted standard deviation.
        rng : RandomState, optional
            Random number generator for reproducibility.

        Returns
        -------
        list[dict]
            List of bucket dictionaries, each with keys:
            - date, bucket_label, bucket_low, bucket_high
            - true_prob, market_prob, model_prob
            - bid_price, ask_price, volume
            - actual_outcome (1 if TMAX in bucket, else 0)
            - direction, threshold_low, threshold_high
        """
        if rng is None:
            rng = np.random.RandomState()

        if isinstance(date, str):
            date = pd.to_datetime(date).date()

        # Market's view: slightly noisier Gaussian centered near actual
        # The market is informed but not perfect
        market_mu = actual_tmax + rng.normal(0, 2.5)
        market_sigma = model_sigma * 1.15  # Market is slightly wider

        buckets = []
        for (lo, hi), label in zip(self.bucket_edges, self.bucket_labels):
            # True probability (from market's Gaussian)
            true_prob = self._compute_true_bucket_prob(
                lo, hi, market_mu, market_sigma,
            )

            # Market price: true prob + noise
            noise = rng.normal(0, self.market_noise_std)
            market_prob = float(np.clip(true_prob + noise, 0.01, 0.99))

            # Model probability
            model_sigma_safe = max(model_sigma, 1e-10)
            if lo <= -900:
                model_prob = float(stats.norm.cdf(
                    hi, loc=model_mu, scale=model_sigma_safe,
                ))
            elif hi >= 900:
                model_prob = float(1.0 - stats.norm.cdf(
                    lo, loc=model_mu, scale=model_sigma_safe,
                ))
            else:
                model_prob = float(
                    stats.norm.cdf(hi, loc=model_mu, scale=model_sigma_safe)
                    - stats.norm.cdf(lo, loc=model_mu, scale=model_sigma_safe)
                )
            model_prob = float(np.clip(model_prob, 0.001, 0.999))

            # Bid-ask spread: tighter for high-prob (near-the-money) buckets
            prob_distance = abs(market_prob - 0.5)
            spread = self.min_spread + (self.max_spread - self.min_spread) * (
                1.0 - 2.0 * min(market_prob, 1.0 - market_prob)
            )
            bid_price = float(np.clip(market_prob - spread / 2, 0.01, 0.99))
            ask_price = float(np.clip(market_prob + spread / 2, 0.01, 0.99))

            # Volume: higher for buckets near the expected TMAX
            vol_center = max(true_prob * 500, 5)
            volume = max(1, int(rng.poisson(lam=vol_center)))

            # Outcome: did actual TMAX fall in this bucket?
            if lo <= -900:
                actual_outcome = 1 if actual_tmax < hi else 0
            elif hi >= 900:
                actual_outcome = 1 if actual_tmax >= lo else 0
            else:
                actual_outcome = 1 if lo <= actual_tmax < hi else 0

            # Direction for compatibility with build_historical_comparison
            if lo <= -900:
                direction = "below"
                threshold_low = None
                threshold_high = float(hi)
            elif hi >= 900:
                direction = "above"
                threshold_low = float(lo)
                threshold_high = None
            else:
                direction = "between"
                threshold_low = float(lo)
                threshold_high = float(hi)

            buckets.append({
                "date": date,
                "bucket_label": label,
                "bucket_low": lo if lo > -900 else None,
                "bucket_high": hi if hi < 900 else None,
                "true_prob": true_prob,
                "market_prob": market_prob,
                "model_prob": model_prob,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "volume": volume,
                "actual_outcome": actual_outcome,
                "direction": direction,
                "threshold_low": threshold_low,
                "threshold_high": threshold_high,
            })

        return buckets

    def generate_market_dataset(
        self,
        predictions_df: pd.DataFrame,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a full market dataset aligned with model predictions.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Must have columns: date, model_mu, model_sigma, actual_tmax.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Full market dataset with one row per bucket per day.
        """
        required = ["date", "model_mu", "model_sigma", "actual_tmax"]
        missing = [c for c in required if c not in predictions_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rng = np.random.RandomState(seed)
        all_rows = []

        for _, row in predictions_df.iterrows():
            date = pd.to_datetime(row["date"]).date()
            buckets = self.generate_daily_buckets(
                date=date,
                actual_tmax=row["actual_tmax"],
                model_mu=row["model_mu"],
                model_sigma=row["model_sigma"],
                rng=rng,
            )
            all_rows.extend(buckets)

        df = pd.DataFrame(all_rows)
        logger.info(
            "Generated market dataset: %d rows across %d days",
            len(df), predictions_df["date"].nunique(),
        )
        return df


# ===========================================================================
# 2. ModelPredictionGenerator
# ===========================================================================

class ModelPredictionGenerator:
    """Generate realistic NYC temperature model predictions.

    Produces Gaussian (mu, sigma) predictions with proper seasonal
    patterns, where the model has a small but real edge over the
    market (slightly tighter sigma than actual residuals).

    Parameters
    ----------
    model_bias : float
        Systematic bias in model mean (deg F). Default 0.0.
    model_noise_std : float
        Random noise in model mean (deg F). Default 2.0.
    sigma_base : float
        Base model sigma (deg F). Default 3.5.
    sigma_seasonal_scale : float
        How much sigma varies by season. Default 1.5.
    model_edge : float
        How much tighter the model sigma is vs. actual residuals,
        as a fraction (e.g. 0.05 = 5% tighter). Default 0.03.

    Examples
    --------
    >>> gen = ModelPredictionGenerator(model_edge=0.03)
    >>> df = gen.generate_predictions("2025-01-01", "2025-12-31", seed=42)
    >>> "model_mu" in df.columns and "actual_tmax" in df.columns
    True
    """

    def __init__(
        self,
        model_bias: float = 0.0,
        model_noise_std: float = 2.0,
        sigma_base: float = 3.5,
        sigma_seasonal_scale: float = 1.5,
        model_edge: float = 0.03,
    ):
        self.model_bias = model_bias
        self.model_noise_std = model_noise_std
        self.sigma_base = sigma_base
        self.sigma_seasonal_scale = sigma_seasonal_scale
        self.model_edge = model_edge

    def _get_climatology(self, date: datetime.date) -> tuple[float, float]:
        """Get climatological mean and std for a given date.

        Uses linear interpolation between monthly values for smooth
        seasonal transitions.

        Parameters
        ----------
        date : datetime.date
            The date to get climatology for.

        Returns
        -------
        tuple[float, float]
            (climatological_mean, climatological_std) in deg F.
        """
        month = date.month
        day = date.day

        # Interpolate between this month and next month
        next_month = month % 12 + 1
        frac = (day - 1) / 30.0  # rough fraction through month

        mean_val = (
            NYC_MONTHLY_TMAX_MEAN[month] * (1 - frac)
            + NYC_MONTHLY_TMAX_MEAN[next_month] * frac
        )
        std_val = (
            NYC_MONTHLY_TMAX_STD[month] * (1 - frac)
            + NYC_MONTHLY_TMAX_STD[next_month] * frac
        )

        return mean_val, std_val

    def generate_predictions(
        self,
        start_date: str,
        end_date: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate daily model predictions for a date range.

        Parameters
        ----------
        start_date : str
            Start date (ISO format, e.g. "2025-01-01").
        end_date : str
            End date (ISO format, e.g. "2025-12-31").
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - date: trading date
            - model_mu: model predicted mean (deg F)
            - model_sigma: model predicted std (deg F)
            - actual_tmax: actual observed TMAX (deg F)
            - climatology_mean: climatological mean for the date
            - climatology_std: climatological std for the date
        """
        rng = np.random.RandomState(seed)
        dates = pd.date_range(start_date, end_date, freq="D")

        records = []
        prev_actual = None

        for dt in dates:
            date = dt.date()
            clim_mean, clim_std = self._get_climatology(date)

            # Actual temperature: climatology + weather noise
            # Include autocorrelation with previous day
            if prev_actual is not None:
                # Autocorrelation: pull toward yesterday + mean reversion
                weather_anomaly = 0.4 * (prev_actual - clim_mean)
                actual_tmax = clim_mean + weather_anomaly + rng.normal(0, clim_std * 0.75)
            else:
                actual_tmax = clim_mean + rng.normal(0, clim_std)

            # Clip to physically reasonable range
            actual_tmax = float(np.clip(actual_tmax, -10.0, 115.0))
            prev_actual = actual_tmax

            # Model prediction: biased toward actual (model has skill)
            model_mu = actual_tmax + self.model_bias + rng.normal(
                0, self.model_noise_std,
            )

            # Model sigma: base + seasonal adjustment
            # Winter has higher sigma, summer lower
            seasonal_factor = clim_std / np.mean(list(NYC_MONTHLY_TMAX_STD.values()))
            model_sigma = (
                self.sigma_base
                + self.sigma_seasonal_scale * (seasonal_factor - 1.0)
                + rng.uniform(-0.3, 0.3)
            )
            model_sigma = float(np.clip(model_sigma, 1.5, 8.0))

            records.append({
                "date": date,
                "model_mu": float(model_mu),
                "model_sigma": float(model_sigma),
                "actual_tmax": actual_tmax,
                "climatology_mean": clim_mean,
                "climatology_std": clim_std,
            })

        df = pd.DataFrame(records)
        logger.info(
            "Generated %d daily predictions from %s to %s "
            "(mean mu=%.1f, mean sigma=%.2f, mean actual=%.1f)",
            len(df), start_date, end_date,
            df["model_mu"].mean(), df["model_sigma"].mean(),
            df["actual_tmax"].mean(),
        )
        return df


# ===========================================================================
# 3. BacktestAnalyzer
# ===========================================================================

class BacktestAnalyzer:
    """Analyze backtest results across in-sample and out-of-sample periods.

    Provides Brier score analysis, edge persistence comparison, strategy
    selection, and comprehensive report generation per the backtesting
    plan in reports/kalshi_real_data_backtesting_plan.md.

    Examples
    --------
    >>> analyzer = BacktestAnalyzer()
    >>> brier = analyzer.analyze_brier_scores(comparison_df)
    >>> report = analyzer.generate_comprehensive_report(
    ...     in_sample_result, oos_result, "results/combined",
    ... )
    """

    def analyze_brier_scores(
        self,
        comparison_df: pd.DataFrame,
    ) -> dict:
        """Perform detailed Brier score analysis by season, month, and bucket.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Must have columns: date, model_prob, market_prob,
            actual_outcome (or outcome).

        Returns
        -------
        dict
            Brier analysis with keys:
            - "overall": dict with model_brier, market_brier, brier_delta
            - "by_month": dict mapping month -> brier metrics
            - "by_season": dict mapping season -> brier metrics
            - "by_bucket": dict mapping bucket_label -> brier metrics
            - "n_samples": int
        """
        df = comparison_df.copy()

        # Normalize outcome column name
        if "outcome" in df.columns and "actual_outcome" not in df.columns:
            df["actual_outcome"] = df["outcome"]
        elif "actual_outcome" not in df.columns:
            raise ValueError(
                "comparison_df must have 'actual_outcome' or 'outcome' column"
            )

        # Drop NaN rows
        required = ["model_prob", "market_prob", "actual_outcome"]
        df = df.dropna(subset=required)
        n = len(df)

        if n == 0:
            return {
                "overall": {
                    "model_brier": float("nan"),
                    "market_brier": float("nan"),
                    "brier_delta": float("nan"),
                },
                "by_month": {},
                "by_season": {},
                "by_bucket": {},
                "n_samples": 0,
            }

        def _brier(probs, outcomes):
            p = _to_numpy(probs)
            o = _to_numpy(outcomes)
            return float(np.mean((p - o) ** 2))

        # Overall Brier
        model_brier = _brier(df["model_prob"], df["actual_outcome"])
        market_brier = _brier(df["market_prob"], df["actual_outcome"])

        result = {
            "overall": {
                "model_brier": model_brier,
                "market_brier": market_brier,
                "brier_delta": model_brier - market_brier,
                "n_samples": n,
            },
            "by_month": {},
            "by_season": {},
            "by_bucket": {},
            "n_samples": n,
        }

        # By month
        df["_date"] = pd.to_datetime(df["date"])
        df["_month"] = df["_date"].dt.month
        for month, group in df.groupby("_month"):
            if len(group) < 5:
                continue
            result["by_month"][int(month)] = {
                "model_brier": _brier(group["model_prob"], group["actual_outcome"]),
                "market_brier": _brier(group["market_prob"], group["actual_outcome"]),
                "n": len(group),
            }
            result["by_month"][int(month)]["brier_delta"] = (
                result["by_month"][int(month)]["model_brier"]
                - result["by_month"][int(month)]["market_brier"]
            )

        # By season
        df["_season"] = df["_month"].map(SEASON_MAP)
        for season, group in df.groupby("_season"):
            if len(group) < 10:
                continue
            result["by_season"][season] = {
                "model_brier": _brier(group["model_prob"], group["actual_outcome"]),
                "market_brier": _brier(group["market_prob"], group["actual_outcome"]),
                "n": len(group),
            }
            result["by_season"][season]["brier_delta"] = (
                result["by_season"][season]["model_brier"]
                - result["by_season"][season]["market_brier"]
            )

        # By bucket
        if "bucket_label" in df.columns:
            for bucket, group in df.groupby("bucket_label"):
                if len(group) < 5:
                    continue
                result["by_bucket"][bucket] = {
                    "model_brier": _brier(
                        group["model_prob"], group["actual_outcome"],
                    ),
                    "market_brier": _brier(
                        group["market_prob"], group["actual_outcome"],
                    ),
                    "n": len(group),
                }
                result["by_bucket"][bucket]["brier_delta"] = (
                    result["by_bucket"][bucket]["model_brier"]
                    - result["by_bucket"][bucket]["market_brier"]
                )

        logger.info(
            "Brier analysis: model=%.4f, market=%.4f, delta=%.4f (n=%d)",
            model_brier, market_brier, model_brier - market_brier, n,
        )
        return result

    def analyze_edge_persistence(
        self,
        in_sample_metrics: dict,
        oos_metrics: dict,
    ) -> pd.DataFrame:
        """Compare in-sample and OOS metrics to assess edge persistence.

        Implements Step 10 from the backtesting plan.

        Parameters
        ----------
        in_sample_metrics : dict
            In-sample metrics with keys: sharpe_ratio, roi, win_rate,
            max_drawdown, total_pnl, n_trades, brier_delta.
        oos_metrics : dict
            Out-of-sample metrics with same keys.

        Returns
        -------
        pd.DataFrame
            Comparison table with columns: metric, in_sample, oos,
            change, verdict.
        """
        metrics = [
            "sharpe_ratio", "roi", "win_rate", "max_drawdown",
            "total_pnl", "n_trades", "brier_delta",
        ]

        rows = []
        for metric in metrics:
            is_val = in_sample_metrics.get(metric, float("nan"))
            oos_val = oos_metrics.get(metric, float("nan"))

            if isinstance(is_val, (int, float)) and isinstance(oos_val, (int, float)):
                if not np.isnan(is_val) and not np.isnan(oos_val):
                    change = oos_val - is_val
                else:
                    change = float("nan")
            else:
                change = float("nan")

            # Generate verdict
            verdict = self._get_verdict(metric, is_val, oos_val, change)

            rows.append({
                "metric": metric,
                "in_sample": is_val,
                "oos": oos_val,
                "change": change,
                "verdict": verdict,
            })

        df = pd.DataFrame(rows)
        logger.info("Edge persistence analysis complete: %d metrics compared", len(df))
        return df

    def _get_verdict(
        self,
        metric: str,
        is_val: float,
        oos_val: float,
        change: float,
    ) -> str:
        """Generate a verdict string for a metric comparison.

        Parameters
        ----------
        metric : str
            Name of the metric.
        is_val : float
            In-sample value.
        oos_val : float
            Out-of-sample value.
        change : float
            Change from in-sample to OOS.

        Returns
        -------
        str
            Verdict description.
        """
        if np.isnan(change):
            return "Insufficient data"

        if metric == "sharpe_ratio":
            if oos_val >= 1.5:
                return "STRONG: OOS Sharpe >= 1.5, strategy validated"
            elif oos_val >= 0.5:
                return "MODERATE: OOS Sharpe 0.5-1.5, edge exists but weaker"
            else:
                return "WEAK: OOS Sharpe < 0.5, possible overfit"

        elif metric == "roi":
            if oos_val > 0:
                return "POSITIVE: OOS profitable"
            else:
                return "NEGATIVE: OOS unprofitable"

        elif metric == "win_rate":
            if abs(change) <= 0.05:
                return "STABLE: Win rate within +/-5%"
            elif change < -0.05:
                return "DEGRADED: Win rate dropped >5%"
            else:
                return "IMPROVED: Win rate increased >5%"

        elif metric == "max_drawdown":
            if is_val > 0 and oos_val <= 2 * is_val:
                return "STABLE: OOS DD < 2x in-sample"
            elif is_val > 0:
                return "ELEVATED: OOS DD > 2x in-sample"
            else:
                return "OK"

        elif metric == "brier_delta":
            if oos_val < 0:
                return "EDGE PERSISTS: Model still beats market"
            else:
                return "EDGE LOST: Market now beats model"

        elif metric == "total_pnl":
            if oos_val > 0:
                return "PROFITABLE"
            else:
                return "UNPROFITABLE"

        return "See details"

    def select_best_strategy(
        self,
        results_df: pd.DataFrame,
        min_sharpe: float = 1.5,
        max_drawdown_frac: float = 0.20,
        min_trades: int = 100,
        min_win_rate: float = 0.30,
        bankroll: float = 10000.0,
    ) -> dict:
        """Select the best strategy from a grid of backtest results.

        Implements the selection criteria from the backtesting plan
        (Step 6.D): Sharpe >= 1.5, max DD < 20%, n_trades >= 100,
        win rate > 30%. If Sharpe for both top candidates is > 2.0,
        prefer the one with higher P&L (within 0.2 Sharpe tolerance).

        Parameters
        ----------
        results_df : pd.DataFrame
            Strategy comparison DataFrame with columns: strategy_name,
            sharpe_ratio, total_pnl, roi, win_rate, max_drawdown, n_trades.
        min_sharpe : float
            Minimum Sharpe ratio threshold.
        max_drawdown_frac : float
            Maximum drawdown as fraction of bankroll.
        min_trades : int
            Minimum number of trades for statistical significance.
        min_win_rate : float
            Minimum win rate.
        bankroll : float
            Bankroll for computing max drawdown threshold.

        Returns
        -------
        dict
            Best strategy configuration with keys: strategy_name,
            sharpe_ratio, total_pnl, roi, win_rate, max_drawdown,
            n_trades, selection_reason.
        """
        if results_df.empty:
            return {"strategy_name": "NONE", "selection_reason": "No strategies"}

        df = results_df.copy()

        # Filter by criteria
        max_dd_abs = max_drawdown_frac * bankroll
        qualified = df[
            (df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            & (df["sharpe_ratio"] >= min_sharpe)
            & (df["max_drawdown"] <= max_dd_abs)
            & (df["n_trades"] >= min_trades)
            & (df["win_rate"] >= min_win_rate)
        ]

        if qualified.empty:
            # Relax criteria: at least profitable with trades
            relaxed = df[
                (df["total_pnl"] > 0)
                & (df["n_trades"] > 0)
                & (df["sharpe_ratio"].apply(lambda x: np.isfinite(x)))
            ]
            if relaxed.empty:
                # Just pick the least bad strategy
                best_row = df.loc[df["total_pnl"].idxmax()]
                return {
                    **best_row.to_dict(),
                    "selection_reason": (
                        "No qualified strategies; selected best P&L"
                    ),
                }
            # From relaxed, pick highest Sharpe
            best_row = relaxed.loc[relaxed["sharpe_ratio"].idxmax()]
            return {
                **best_row.to_dict(),
                "selection_reason": (
                    f"Relaxed criteria (Sharpe={best_row['sharpe_ratio']:.2f})"
                ),
            }

        # Among qualified, apply the Sharpe > 2.0 / P&L preference rule
        top_candidates = qualified.nlargest(5, "sharpe_ratio")

        if len(top_candidates) >= 2:
            top_two = top_candidates.head(2)
            if (top_two["sharpe_ratio"] > 2.0).all():
                sharpe_diff = abs(
                    top_two.iloc[0]["sharpe_ratio"]
                    - top_two.iloc[1]["sharpe_ratio"]
                )
                if sharpe_diff <= 0.2:
                    # Pick the one with higher P&L
                    best_idx = top_two["total_pnl"].idxmax()
                    best_row = top_two.loc[best_idx]
                    return {
                        **best_row.to_dict(),
                        "selection_reason": (
                            "Both Sharpe > 2.0 and within 0.2; "
                            "selected higher P&L"
                        ),
                    }

        # Default: highest Sharpe among qualified
        best_row = qualified.loc[qualified["sharpe_ratio"].idxmax()]
        return {
            **best_row.to_dict(),
            "selection_reason": "Highest Sharpe among qualified strategies",
        }

    def generate_comprehensive_report(
        self,
        in_sample_result: dict,
        oos_result: dict,
        output_dir: str,
    ) -> str:
        """Generate the final comprehensive backtest report in markdown.

        Implements the full Step 10 analysis from the backtesting plan
        including executive summary, calibration, stability analysis,
        risk assessment, and trading recommendation.

        Parameters
        ----------
        in_sample_result : dict
            In-sample backtest summary with keys: metrics (dict),
            brier_analysis (dict), strategy_config (dict).
        oos_result : dict
            OOS backtest summary with same structure.
        output_dir : str
            Directory to save the report.

        Returns
        -------
        str
            The full report text in markdown format.
        """
        os.makedirs(output_dir, exist_ok=True)

        is_metrics = in_sample_result.get("metrics", {})
        oos_metrics = oos_result.get("metrics", {})
        is_brier = in_sample_result.get("brier_analysis", {})
        oos_brier = oos_result.get("brier_analysis", {})
        strategy_config = in_sample_result.get("strategy_config", {})

        # Compute edge persistence
        is_m = {**is_metrics}
        oos_m = {**oos_metrics}
        is_m["brier_delta"] = is_brier.get("overall", {}).get("brier_delta", float("nan"))
        oos_m["brier_delta"] = oos_brier.get("overall", {}).get("brier_delta", float("nan"))
        persistence_df = self.analyze_edge_persistence(is_m, oos_m)

        # Determine overall recommendation
        recommendation = self._generate_recommendation(is_m, oos_m)

        lines = [
            "# Kalshi KXHIGHNY Comprehensive Backtest Report",
            "",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"**Strategy:** {strategy_config.get('name', 'Best Strategy')}",
            f"**In-Sample Period:** {is_metrics.get('period', '2023-2024')}",
            f"**Out-of-Sample Period:** {oos_metrics.get('period', '2025')}",
            "",
            f"**Overall Verdict:** {recommendation['verdict']}",
            "",
            recommendation["summary"],
            "",
            "---",
            "",
            "## 2. Model Calibration",
            "",
            "### Brier Score Comparison",
            "",
            "| Period | Model Brier | Market Brier | Delta | Interpretation |",
            "|--------|-------------|--------------|-------|----------------|",
        ]

        is_overall = is_brier.get("overall", {})
        oos_overall = oos_brier.get("overall", {})
        lines.append(
            f"| In-Sample | {is_overall.get('model_brier', float('nan')):.4f} "
            f"| {is_overall.get('market_brier', float('nan')):.4f} "
            f"| {is_overall.get('brier_delta', float('nan')):.4f} "
            f"| {'Model better' if is_overall.get('brier_delta', 0) < 0 else 'Market better'} |"
        )
        lines.append(
            f"| OOS | {oos_overall.get('model_brier', float('nan')):.4f} "
            f"| {oos_overall.get('market_brier', float('nan')):.4f} "
            f"| {oos_overall.get('brier_delta', float('nan')):.4f} "
            f"| {'Model better' if oos_overall.get('brier_delta', 0) < 0 else 'Market better'} |"
        )

        # Seasonal Brier
        lines.extend([
            "",
            "### Seasonal Brier Breakdown (OOS)",
            "",
            "| Season | Model Brier | Market Brier | Delta | N |",
            "|--------|-------------|--------------|-------|---|",
        ])
        for season in SEASON_ORDER:
            s_data = oos_brier.get("by_season", {}).get(season, {})
            if s_data:
                lines.append(
                    f"| {season} | {s_data.get('model_brier', float('nan')):.4f} "
                    f"| {s_data.get('market_brier', float('nan')):.4f} "
                    f"| {s_data.get('brier_delta', float('nan')):.4f} "
                    f"| {s_data.get('n', 0)} |"
                )

        # In-sample results
        lines.extend([
            "",
            "---",
            "",
            "## 3. In-Sample Results (2023-2024)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total P&L | ${is_metrics.get('total_pnl', 0):.2f} |",
            f"| ROI | {is_metrics.get('roi', 0) * 100:.1f}% |",
            f"| Sharpe Ratio | {is_metrics.get('sharpe_ratio', 0):.2f} |",
            f"| Win Rate | {is_metrics.get('win_rate', 0) * 100:.1f}% |",
            f"| Max Drawdown | ${is_metrics.get('max_drawdown', 0):.2f} |",
            f"| Trades | {is_metrics.get('n_trades', 0)} |",
            f"| Avg EV | {is_metrics.get('avg_ev', 0):.4f} |",
        ])

        # OOS results
        lines.extend([
            "",
            "---",
            "",
            "## 4. Out-of-Sample Results (2025)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total P&L | ${oos_metrics.get('total_pnl', 0):.2f} |",
            f"| ROI | {oos_metrics.get('roi', 0) * 100:.1f}% |",
            f"| Sharpe Ratio | {oos_metrics.get('sharpe_ratio', 0):.2f} |",
            f"| Win Rate | {oos_metrics.get('win_rate', 0) * 100:.1f}% |",
            f"| Max Drawdown | ${oos_metrics.get('max_drawdown', 0):.2f} |",
            f"| Trades | {oos_metrics.get('n_trades', 0)} |",
            f"| Avg EV | {oos_metrics.get('avg_ev', 0):.4f} |",
        ])

        # Stability analysis
        lines.extend([
            "",
            "---",
            "",
            "## 5. Stability Analysis",
            "",
            "| Metric | In-Sample | OOS | Change | Verdict |",
            "|--------|-----------|-----|--------|---------|",
        ])
        for _, row in persistence_df.iterrows():
            is_v = row["in_sample"]
            oos_v = row["oos"]
            ch = row["change"]
            # Format values appropriately
            if row["metric"] in ("sharpe_ratio",):
                is_str = f"{is_v:.2f}" if not np.isnan(is_v) else "N/A"
                oos_str = f"{oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
                ch_str = f"{ch:+.2f}" if not np.isnan(ch) else "N/A"
            elif row["metric"] in ("roi", "win_rate"):
                is_str = f"{is_v * 100:.1f}%" if not np.isnan(is_v) else "N/A"
                oos_str = f"{oos_v * 100:.1f}%" if not np.isnan(oos_v) else "N/A"
                ch_str = f"{ch * 100:+.1f}%" if not np.isnan(ch) else "N/A"
            elif row["metric"] in ("max_drawdown", "total_pnl"):
                is_str = f"${is_v:.2f}" if not np.isnan(is_v) else "N/A"
                oos_str = f"${oos_v:.2f}" if not np.isnan(oos_v) else "N/A"
                ch_str = f"${ch:+.2f}" if not np.isnan(ch) else "N/A"
            elif row["metric"] == "brier_delta":
                is_str = f"{is_v:.4f}" if not np.isnan(is_v) else "N/A"
                oos_str = f"{oos_v:.4f}" if not np.isnan(oos_v) else "N/A"
                ch_str = f"{ch:+.4f}" if not np.isnan(ch) else "N/A"
            else:
                is_str = f"{is_v}" if not np.isnan(is_v) else "N/A"
                oos_str = f"{oos_v}" if not np.isnan(oos_v) else "N/A"
                ch_str = f"{ch}" if not np.isnan(ch) else "N/A"

            lines.append(
                f"| {row['metric']} | {is_str} | {oos_str} "
                f"| {ch_str} | {row['verdict']} |"
            )

        # Risk assessment
        lines.extend([
            "",
            "---",
            "",
            "## 6. Risk Assessment",
            "",
            f"- **Max Drawdown (IS):** ${is_metrics.get('max_drawdown', 0):.2f} "
            f"({is_metrics.get('max_drawdown', 0) / max(is_metrics.get('bankroll', 10000), 1) * 100:.1f}% of bankroll)",
            f"- **Max Drawdown (OOS):** ${oos_metrics.get('max_drawdown', 0):.2f} "
            f"({oos_metrics.get('max_drawdown', 0) / max(oos_metrics.get('bankroll', 10000), 1) * 100:.1f}% of bankroll)",
        ])

        # VaR and ES from trades
        oos_trades = oos_result.get("trades", [])
        if oos_trades:
            pnls = np.array([t["pnl"] for t in oos_trades])
            var_5 = float(np.percentile(pnls, 5))
            losses = pnls[pnls <= var_5]
            es_5 = float(np.mean(losses)) if len(losses) > 0 else var_5
            lines.extend([
                f"- **Value at Risk (5%):** ${var_5:.2f}",
                f"- **Expected Shortfall (5%):** ${es_5:.2f}",
            ])

        # Seasonal edge
        lines.extend([
            "",
            "---",
            "",
            "## 7. Seasonal Edge Analysis",
            "",
        ])
        oos_seasonal = oos_result.get("seasonal_pnl", {})
        if oos_seasonal:
            lines.extend([
                "| Season | P&L | Trades | Win Rate |",
                "|--------|-----|--------|----------|",
            ])
            for season in SEASON_ORDER:
                s_data = oos_seasonal.get(season, {})
                if s_data:
                    lines.append(
                        f"| {season} | ${s_data.get('total_pnl', 0):.2f} "
                        f"| {s_data.get('n_trades', 0)} "
                        f"| {s_data.get('win_rate', 0) * 100:.1f}% |"
                    )
        else:
            lines.append("*Seasonal data not available.*")

        # Trading recommendation
        lines.extend([
            "",
            "---",
            "",
            "## 8. Trading Recommendation",
            "",
            f"**Recommendation:** {recommendation['action']}",
            "",
            recommendation["details"],
            "",
            "### Strategy Configuration",
            "",
            "```json",
            json.dumps(strategy_config, indent=2, default=str),
            "```",
            "",
        ])

        report_text = "\n".join(lines)

        # Save report
        report_path = os.path.join(output_dir, "final_backtest_report.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info("Saved comprehensive report to %s", report_path)

        return report_text

    def _generate_recommendation(
        self,
        is_metrics: dict,
        oos_metrics: dict,
    ) -> dict:
        """Generate the trading recommendation based on OOS performance.

        Implements the interpretation guidelines from the backtesting plan.

        Parameters
        ----------
        is_metrics : dict
            In-sample metrics.
        oos_metrics : dict
            Out-of-sample metrics.

        Returns
        -------
        dict
            Recommendation with keys: verdict, action, summary, details.
        """
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        oos_roi = oos_metrics.get("roi", 0)
        oos_brier = oos_metrics.get("brier_delta", 0)

        if oos_sharpe >= 1.5 and oos_roi > 0:
            return {
                "verdict": "VALIDATED",
                "action": "Proceed to live paper trading with frozen parameters",
                "summary": (
                    "The strategy demonstrates strong out-of-sample performance "
                    f"(Sharpe={oos_sharpe:.2f}, ROI={oos_roi*100:.1f}%). "
                    "The model's edge persists on unseen data."
                ),
                "details": (
                    "The OOS Sharpe ratio exceeds 1.5 with positive ROI. "
                    "Per the backtesting plan, this validates the strategy for "
                    "paper trading. Recommended next steps:\n"
                    "1. Run paper trading for 30-60 days\n"
                    "2. Monitor for regime changes in market efficiency\n"
                    "3. If paper trading confirms, deploy with half-Kelly sizing"
                ),
            }
        elif 0.5 <= oos_sharpe < 1.5 and oos_roi > 0:
            return {
                "verdict": "CAUTIOUS",
                "action": "Edge exists but weaker than in-sample; reduce sizing by 50%",
                "summary": (
                    "The strategy is profitable OOS but with reduced edge "
                    f"(Sharpe={oos_sharpe:.2f}, ROI={oos_roi*100:.1f}%). "
                    "Consider more conservative position sizing."
                ),
                "details": (
                    "The OOS Sharpe ratio is 0.5-1.5 with positive ROI. "
                    "The model has some edge but it is weaker than in-sample. "
                    "Recommendations:\n"
                    "1. Reduce Kelly fraction by 50%\n"
                    "2. Paper trade for 60-90 days before going live\n"
                    "3. Focus on seasons where edge is strongest"
                ),
            }
        elif oos_brier > 0:
            return {
                "verdict": "NO EDGE",
                "action": "Do NOT trade live; model calibration has degraded",
                "summary": (
                    "The model's Brier score is worse than the market OOS "
                    f"(delta={oos_brier:.4f}). The model has lost its "
                    "fundamental edge."
                ),
                "details": (
                    "The OOS Brier delta is positive, meaning the market "
                    "is now better calibrated than our model. No trading "
                    "strategy can compensate for worse predictions. "
                    "Diagnose:\n"
                    "1. Has the market become more efficient?\n"
                    "2. Has the model's training data gone stale?\n"
                    "3. Are there seasonal calibration issues?"
                ),
            }
        else:
            return {
                "verdict": "OVERFIT",
                "action": "Do NOT trade live; in-sample results likely overfit",
                "summary": (
                    "OOS performance is significantly degraded "
                    f"(Sharpe={oos_sharpe:.2f}, ROI={oos_roi*100:.1f}%). "
                    "The in-sample edge does not generalize."
                ),
                "details": (
                    "The OOS Sharpe ratio is below 0.5 or ROI is negative. "
                    "The in-sample results were likely overfit. "
                    "Investigate:\n"
                    "1. Were strategy parameters overfit to 2023-2024?\n"
                    "2. Did market regime change between periods?\n"
                    "3. Is the model overfitting to historical patterns?"
                ),
            }

    def plot_oos_vs_insample(
        self,
        in_sample_result: dict,
        oos_result: dict,
        output_dir: str,
    ) -> list[str]:
        """Generate comparison plots between in-sample and OOS performance.

        Parameters
        ----------
        in_sample_result : dict
            In-sample results with 'trades' and 'metrics' keys.
        oos_result : dict
            OOS results with 'trades' and 'metrics' keys.
        output_dir : str
            Directory to save plots.

        Returns
        -------
        list[str]
            Paths to saved plot files.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []

        is_trades = in_sample_result.get("trades", [])
        oos_trades = oos_result.get("trades", [])

        # Plot 1: Combined P&L curves
        fig, ax = plt.subplots(figsize=(12, 6))
        if is_trades:
            is_pnls = np.cumsum([t["pnl"] for t in is_trades])
            ax.plot(range(len(is_pnls)), is_pnls,
                    label="In-Sample (2023-2024)", linewidth=1.5,
                    color="#4c72b0")
        if oos_trades:
            oos_pnls = np.cumsum([t["pnl"] for t in oos_trades])
            offset = len(is_pnls) if is_trades else 0
            ax.plot(range(offset, offset + len(oos_pnls)), oos_pnls,
                    label="Out-of-Sample (2025)", linewidth=1.5,
                    color="#d62728")
            if is_trades:
                ax.axvline(offset, color="gray", linestyle="--",
                           linewidth=0.8, label="IS/OOS boundary")

        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("In-Sample vs Out-of-Sample: Cumulative P&L")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, "combined_pnl_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(path)

        # Plot 2: IS vs OOS metric comparison bar chart
        is_metrics = in_sample_result.get("metrics", {})
        oos_metrics = oos_result.get("metrics", {})

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sharpe
        ax = axes[0]
        vals = [is_metrics.get("sharpe_ratio", 0), oos_metrics.get("sharpe_ratio", 0)]
        colors = ["#4c72b0", "#d62728"]
        ax.bar(["In-Sample", "OOS"], vals, color=colors, alpha=0.75)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio Comparison")
        ax.axhline(0, color="black", linewidth=0.5)

        # ROI
        ax = axes[1]
        vals = [is_metrics.get("roi", 0) * 100, oos_metrics.get("roi", 0) * 100]
        ax.bar(["In-Sample", "OOS"], vals, color=colors, alpha=0.75)
        ax.set_ylabel("ROI (%)")
        ax.set_title("ROI Comparison")
        ax.axhline(0, color="black", linewidth=0.5)

        # Win Rate
        ax = axes[2]
        vals = [is_metrics.get("win_rate", 0) * 100, oos_metrics.get("win_rate", 0) * 100]
        ax.bar(["In-Sample", "OOS"], vals, color=colors, alpha=0.75)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate Comparison")

        fig.tight_layout()
        path = os.path.join(output_dir, "insample_vs_oos_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(path)

        # Plot 3: Monthly P&L comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        for ax, trades, title, color in [
            (ax1, is_trades, "In-Sample Monthly P&L", "#4c72b0"),
            (ax2, oos_trades, "OOS Monthly P&L", "#d62728"),
        ]:
            if trades:
                trade_df = pd.DataFrame(trades)
                trade_df["date"] = pd.to_datetime(trade_df["date"])
                trade_df["month"] = trade_df["date"].dt.to_period("M")
                monthly = trade_df.groupby("month")["pnl"].sum()
                bar_colors = [
                    "#2ca02c" if v >= 0 else "#d62728" for v in monthly.values
                ]
                ax.bar(range(len(monthly)), monthly.values, color=bar_colors)
                ax.set_xticks(range(len(monthly)))
                ax.set_xticklabels(
                    [str(m) for m in monthly.index],
                    rotation=45, ha="right", fontsize=8,
                )
                ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("P&L ($)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        path = os.path.join(output_dir, "combined_monthly_pnl.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(path)

        logger.info("Generated %d comparison plots in %s", len(saved_plots), output_dir)
        return saved_plots


# ===========================================================================
# 4. CalibrationAnalyzer
# ===========================================================================

class CalibrationAnalyzer:
    """Analyze model calibration using reliability diagrams, PIT histograms,
    and per-season breakdowns.

    Wraps the calibration module functions with additional analysis
    specific to Kalshi bucket probability calibration.

    Examples
    --------
    >>> analyzer = CalibrationAnalyzer()
    >>> metrics = analyzer.analyze_model_calibration(model_probs, outcomes)
    >>> analyzer.plot_reliability_diagram(model_probs, outcomes, "results/")
    """

    def analyze_model_calibration(
        self,
        model_probs: Union[np.ndarray, pd.Series, list],
        outcomes: Union[np.ndarray, pd.Series, list],
        n_bins: int = 10,
    ) -> dict:
        """Compute comprehensive calibration metrics for model probabilities.

        Parameters
        ----------
        model_probs : array-like
            Model-predicted probabilities for YES outcomes.
        outcomes : array-like
            Binary outcomes (1 = YES, 0 = NO).
        n_bins : int
            Number of bins for reliability assessment (default 10).

        Returns
        -------
        dict
            Calibration metrics with keys:
            - "brier_score": float
            - "log_score": float
            - "reliability": dict with bin-level calibration data
            - "ece": float (Expected Calibration Error)
            - "mce": float (Maximum Calibration Error)
            - "n_samples": int
        """
        model_arr = _to_numpy(model_probs)
        outcome_arr = _to_numpy(outcomes)

        # Remove NaN
        valid = ~(np.isnan(model_arr) | np.isnan(outcome_arr))
        model_arr = model_arr[valid]
        outcome_arr = outcome_arr[valid]
        n = len(model_arr)

        if n == 0:
            return {
                "brier_score": float("nan"),
                "log_score": float("nan"),
                "reliability": {},
                "ece": float("nan"),
                "mce": float("nan"),
                "n_samples": 0,
            }

        # Brier score
        brier = float(np.mean((model_arr - outcome_arr) ** 2))

        # Log score
        eps = 1e-15
        clamped = np.clip(model_arr, eps, 1 - eps)
        log_score = -float(np.mean(
            outcome_arr * np.log(clamped)
            + (1.0 - outcome_arr) * np.log(1.0 - clamped)
        ))

        # Reliability / calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        ece_sum = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = (model_arr >= bin_edges[i]) & (model_arr < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (model_arr >= bin_edges[i]) & (model_arr <= bin_edges[i + 1])

            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                bin_data.append({
                    "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                    "mean_predicted": float("nan"),
                    "mean_observed": float("nan"),
                    "n": 0,
                })
                continue

            mean_pred = float(np.mean(model_arr[mask]))
            mean_obs = float(np.mean(outcome_arr[mask]))
            cal_error = abs(mean_pred - mean_obs)

            ece_sum += cal_error * n_in_bin
            mce = max(mce, cal_error)

            bin_data.append({
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "mean_predicted": mean_pred,
                "mean_observed": mean_obs,
                "calibration_error": cal_error,
                "n": n_in_bin,
            })

        ece = ece_sum / n

        result = {
            "brier_score": brier,
            "log_score": log_score,
            "reliability": bin_data,
            "ece": float(ece),
            "mce": float(mce),
            "n_samples": n,
        }

        logger.info(
            "Calibration analysis: Brier=%.4f, ECE=%.4f, MCE=%.4f (n=%d)",
            brier, ece, mce, n,
        )
        return result

    def plot_reliability_diagram(
        self,
        model_probs: Union[np.ndarray, pd.Series, list],
        outcomes: Union[np.ndarray, pd.Series, list],
        output_dir: str,
        title: str = "Model Calibration: Reliability Diagram",
        n_bins: int = 10,
    ) -> str:
        """Plot reliability diagram for model probabilities.

        Parameters
        ----------
        model_probs : array-like
            Model-predicted probabilities.
        outcomes : array-like
            Binary outcomes.
        output_dir : str
            Directory to save the plot.
        title : str
            Plot title.
        n_bins : int
            Number of calibration bins.

        Returns
        -------
        str
            Path to saved plot.
        """
        os.makedirs(output_dir, exist_ok=True)

        cal_data = self.analyze_model_calibration(model_probs, outcomes, n_bins)
        bins = cal_data["reliability"]

        fig, ax = plt.subplots(figsize=(7, 7))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

        # Plot bin data
        pred_vals = []
        obs_vals = []
        for b in bins:
            if b["n"] > 0 and not np.isnan(b["mean_predicted"]):
                pred_vals.append(b["mean_predicted"])
                obs_vals.append(b["mean_observed"])

        if pred_vals:
            ax.plot(pred_vals, obs_vals, "o-", color="#d62728",
                    linewidth=2, markersize=8, label="Model")

        # Annotate ECE and MCE
        ax.text(
            0.02, 0.98,
            f"ECE = {cal_data['ece']:.4f}\nMCE = {cal_data['mce']:.4f}\n"
            f"Brier = {cal_data['brier_score']:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        path = os.path.join(output_dir, "oos_calibration_reliability.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved reliability diagram to %s", path)
        return path

    def compute_seasonal_calibration(
        self,
        comparison_df: pd.DataFrame,
        n_bins: int = 10,
    ) -> dict:
        """Compute per-season calibration metrics.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Must have columns: date, model_prob, actual_outcome (or outcome).
        n_bins : int
            Number of calibration bins per season.

        Returns
        -------
        dict
            Mapping season name -> calibration metrics dict.
        """
        df = comparison_df.copy()

        # Normalize outcome column
        if "outcome" in df.columns and "actual_outcome" not in df.columns:
            df["actual_outcome"] = df["outcome"]

        df["_date"] = pd.to_datetime(df["date"])
        df["_month"] = df["_date"].dt.month
        df["_season"] = df["_month"].map(SEASON_MAP)

        results = {}
        for season in SEASON_ORDER:
            mask = df["_season"] == season
            group = df[mask].dropna(subset=["model_prob", "actual_outcome"])

            if len(group) < 10:
                results[season] = {
                    "brier_score": float("nan"),
                    "ece": float("nan"),
                    "n_samples": len(group),
                }
                continue

            cal = self.analyze_model_calibration(
                group["model_prob"], group["actual_outcome"], n_bins,
            )
            results[season] = {
                "brier_score": cal["brier_score"],
                "log_score": cal["log_score"],
                "ece": cal["ece"],
                "mce": cal["mce"],
                "n_samples": cal["n_samples"],
            }

        logger.info("Computed seasonal calibration for %d seasons", len(results))
        return results


# ===========================================================================
# 5. Utility functions for backtest data preparation
# ===========================================================================

def prepare_backtest_data(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare market data for use with BacktestEngine.

    Renames columns to match BacktestEngine expectations and ensures
    proper data types.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market data from KalshiMarketSimulator.generate_market_dataset().

    Returns
    -------
    pd.DataFrame
        DataFrame ready for BacktestEngine.run_backtest().
    """
    df = market_df.copy()

    # Ensure required columns exist with expected names
    rename_map = {}
    if "market_prob" in df.columns and "market_price" not in df.columns:
        rename_map["market_prob"] = "market_price"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["date", "model_prob", "market_price", "actual_outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    return df


def compute_seasonal_pnl(trades: list[dict]) -> dict:
    """Compute P&L breakdown by season from a list of trades.

    Parameters
    ----------
    trades : list[dict]
        Trade records with 'date' and 'pnl' keys.

    Returns
    -------
    dict
        Mapping season name -> dict with total_pnl, n_trades, win_rate,
        mean_pnl.
    """
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)

    results = {}
    for season in SEASON_ORDER:
        mask = df["season"] == season
        if mask.sum() == 0:
            continue
        season_df = df[mask]
        results[season] = {
            "total_pnl": float(season_df["pnl"].sum()),
            "n_trades": int(mask.sum()),
            "win_rate": float((season_df["pnl"] > 0).mean()),
            "mean_pnl": float(season_df["pnl"].mean()),
        }

    return results
