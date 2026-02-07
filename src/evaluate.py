"""
Evaluation Framework for NYC Temperature Prediction.

Computes performance metrics, generates comparison tables, and produces
diagnostic visualizations for baseline and neural-network models.

Metrics computed:
  - MAE (primary), RMSE, R-squared, bias, max absolute error
  - Percentage of predictions within +/-1, 2, 3 degF thresholds
  - Seasonal breakdown (DJF, MAM, JJA, SON)

Visualizations:
  - Actual vs. predicted scatter
  - Time-series overlay
  - Residual histogram
  - Residuals by calendar month (box plot)
  - Multi-model bar-chart comparison
"""

import os
import sys
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

# Use non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
# Meteorological season mapping
# ---------------------------------------------------------------------------
SEASON_MAP = {
    12: "Winter (DJF)",
    1: "Winter (DJF)",
    2: "Winter (DJF)",
    3: "Spring (MAM)",
    4: "Spring (MAM)",
    5: "Spring (MAM)",
    6: "Summer (JJA)",
    7: "Summer (JJA)",
    8: "Summer (JJA)",
    9: "Fall (SON)",
    10: "Fall (SON)",
    11: "Fall (SON)",
}

SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]


# ===========================================================================
# Helpers
# ===========================================================================

def _to_numpy(arr: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array, stripping NaNs.

    Parameters
    ----------
    arr : array-like
        Input data (numpy array, pandas Series, or list).

    Returns
    -------
    np.ndarray
        1-D float64 array.
    """
    out = np.asarray(arr, dtype=np.float64).ravel()
    return out


def _validate_inputs(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align actual/predicted arrays.

    Drops paired entries where either value is NaN. Raises ValueError
    when lengths differ.

    Parameters
    ----------
    y_actual : array-like
        Actual (ground-truth) values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cleaned (actual, predicted) arrays with NaNs removed.

    Raises
    ------
    ValueError
        If input arrays have different lengths.
    """
    actual = _to_numpy(y_actual)
    pred = _to_numpy(y_pred)

    if actual.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_actual has {actual.shape[0]} elements, "
            f"y_pred has {pred.shape[0]} elements."
        )

    # Drop pairs where either value is NaN
    valid = ~(np.isnan(actual) | np.isnan(pred))
    actual = actual[valid]
    pred = pred[valid]

    return actual, pred


# ===========================================================================
# Core Metrics
# ===========================================================================

def compute_metrics(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    model_name: Optional[str] = None,
) -> dict:
    """Compute a full suite of regression metrics.

    Parameters
    ----------
    y_actual : array-like
        Actual observed values (degF).
    y_pred : array-like
        Model predictions (degF).
    model_name : str, optional
        Label for the model. Included in the returned dictionary if
        provided.

    Returns
    -------
    dict
        Dictionary with keys: model_name, n, mae, rmse, r2, bias,
        within_1f, within_2f, within_3f, max_abs_error.
        If the inputs are empty after NaN removal, all numeric values
        are ``float('nan')``.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)

    result: dict = {}
    if model_name is not None:
        result["model_name"] = model_name

    n = len(actual)
    result["n"] = n

    if n == 0:
        logger.warning("Empty arrays after NaN removal — returning NaN metrics")
        result.update({
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "bias": float("nan"),
            "within_1f": float("nan"),
            "within_2f": float("nan"),
            "within_3f": float("nan"),
            "max_abs_error": float("nan"),
        })
        return result

    errors = pred - actual
    abs_errors = np.abs(errors)

    # MAE
    mae = float(np.mean(abs_errors))

    # RMSE
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0.0:
        # All actuals are identical — R-squared is undefined
        r2 = float("nan")
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    # Bias (mean signed error): positive = over-prediction
    bias = float(np.mean(errors))

    # Threshold percentages
    within_1f = float(np.mean(abs_errors <= 1.0) * 100.0)
    within_2f = float(np.mean(abs_errors <= 2.0) * 100.0)
    within_3f = float(np.mean(abs_errors <= 3.0) * 100.0)

    # Max absolute error
    max_abs_error = float(np.max(abs_errors))

    result.update({
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "within_1f": within_1f,
        "within_2f": within_2f,
        "within_3f": within_3f,
        "max_abs_error": max_abs_error,
    })

    logger.info(
        "Metrics%s — MAE: %.2f, RMSE: %.2f, R2: %.4f, Bias: %.2f",
        f" ({model_name})" if model_name else "",
        mae, rmse, r2, bias,
    )
    return result


# ===========================================================================
# Seasonal Breakdown
# ===========================================================================

def compute_seasonal_metrics(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    dates: Union[pd.DatetimeIndex, np.ndarray, list],
) -> dict[str, dict]:
    """Compute MAE, RMSE, and bias for each meteorological season.

    Seasons follow standard meteorological convention:
      - Winter (DJF): December, January, February
      - Spring (MAM): March, April, May
      - Summer (JJA): June, July, August
      - Fall   (SON): September, October, November

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    dates : DatetimeIndex or array-like of datetime-like
        Dates corresponding to each prediction. Must be the same length
        as ``y_actual`` and ``y_pred``.

    Returns
    -------
    dict[str, dict]
        Mapping of season name -> {"mae", "rmse", "bias", "n"}.
        Seasons with no data points are omitted.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)

    # Convert dates to a pandas DatetimeIndex for reliable .month access
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    if len(dates) != len(actual):
        raise ValueError(
            f"dates length ({len(dates)}) does not match data length "
            f"({len(actual)}) after NaN removal. Pass unfiltered arrays; "
            "NaN-paired rows will be dropped automatically."
        )

    months = dates.month
    seasonal: dict[str, dict] = {}

    for season_name in SEASON_ORDER:
        season_months = [m for m, s in SEASON_MAP.items() if s == season_name]
        mask = np.isin(months, season_months)
        n = int(mask.sum())
        if n == 0:
            continue

        s_actual = actual[mask]
        s_pred = pred[mask]
        s_errors = s_pred - s_actual

        seasonal[season_name] = {
            "mae": float(np.mean(np.abs(s_errors))),
            "rmse": float(np.sqrt(np.mean(s_errors ** 2))),
            "bias": float(np.mean(s_errors)),
            "n": n,
        }

    return seasonal


# ===========================================================================
# Comparison Table
# ===========================================================================

def format_metrics_table(results_dict: dict[str, dict]) -> str:
    """Format a multi-model metrics comparison as a human-readable table.

    Parameters
    ----------
    results_dict : dict[str, dict]
        Mapping of model_name -> metrics dictionary (as returned by
        ``compute_metrics``).

    Returns
    -------
    str
        Formatted text table suitable for console output or file saving.
    """
    if not results_dict:
        return "(no models to compare)"

    # Column definitions: (header, key, format_spec, width)
    columns = [
        ("Model", None, "s", 25),
        ("N", "n", "d", 6),
        ("MAE", "mae", ".2f", 8),
        ("RMSE", "rmse", ".2f", 8),
        ("R2", "r2", ".4f", 8),
        ("Bias", "bias", ".2f", 8),
        ("+/-1F%", "within_1f", ".1f", 8),
        ("+/-2F%", "within_2f", ".1f", 8),
        ("+/-3F%", "within_3f", ".1f", 8),
        ("MaxErr", "max_abs_error", ".2f", 8),
    ]

    # Build header
    header_parts = []
    for hdr, _key, _fmt, width in columns:
        header_parts.append(f"{hdr:>{width}}")
    header = " | ".join(header_parts)
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for model_name, metrics in results_dict.items():
        row_parts = []
        for hdr, key, fmt, width in columns:
            if key is None:
                # Model name column
                val_str = f"{model_name:>{width}}"
            else:
                val = metrics.get(key, float("nan"))
                if isinstance(val, float) and np.isnan(val):
                    val_str = f"{'N/A':>{width}}"
                elif fmt == "d":
                    val_str = f"{int(val):>{width}}"
                else:
                    val_str = f"{val:>{width}{fmt}}"
            row_parts.append(val_str)
        lines.append(" | ".join(row_parts))

    lines.append(separator)
    return "\n".join(lines)


# ===========================================================================
# Visualizations
# ===========================================================================

def plot_actual_vs_predicted(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    model_name: str,
    save_path: str,
    show: bool = False,
) -> None:
    """Scatter plot of actual vs. predicted values.

    Includes a y = x reference line and annotates with R-squared and MAE.

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    model_name : str
        Name of the model (for the title).
    save_path : str
        File path to save the figure (e.g., "results/scatter.png").
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)
    metrics = compute_metrics(actual, pred)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(actual, pred, alpha=0.5, s=18, edgecolors="none", label="Predictions")

    # y = x reference line
    lo = min(actual.min(), pred.min()) - 2
    hi = max(actual.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual Temperature (\u00b0F)")
    ax.set_ylabel("Predicted Temperature (\u00b0F)")
    ax.set_title(
        f"{model_name}: Actual vs Predicted\n"
        f"MAE = {metrics['mae']:.2f}\u00b0F  |  R\u00b2 = {metrics['r2']:.4f}"
    )
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved scatter plot to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_time_series(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    dates: Union[pd.DatetimeIndex, np.ndarray, list],
    model_name: str,
    save_path: str,
    n_days: int = 60,
    show: bool = False,
) -> None:
    """Time-series overlay of actual vs. predicted values.

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    dates : DatetimeIndex or array-like
        Dates for the x-axis.
    model_name : str
        Name of the model.
    save_path : str
        File path to save the figure.
    n_days : int
        Number of days to display (from the start of the series).
        If 0 or negative, show the entire series.
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    # Trim dates to match data length (NaN removal may have shortened it)
    if len(dates) > len(actual):
        dates = dates[: len(actual)]

    # Subset to first n_days
    if n_days > 0:
        actual = actual[:n_days]
        pred = pred[:n_days]
        dates = dates[:n_days]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, actual, label="Actual", linewidth=1.5, color="#1f77b4")
    ax.plot(dates, pred, label=f"{model_name}", linewidth=1.2, color="#ff7f0e",
            linestyle="--")

    # Shade the error region
    ax.fill_between(dates, actual, pred, alpha=0.15, color="#ff7f0e")

    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (\u00b0F)")
    ax.set_title(f"{model_name}: Actual vs Predicted (first {len(actual)} days)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved time-series plot to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_residual_histogram(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    model_name: str,
    save_path: str,
    show: bool = False,
) -> None:
    """Histogram of prediction residuals (predicted - actual).

    Annotates the mean and standard deviation of the residuals and
    draws a vertical line at zero.

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    model_name : str
        Name of the model.
    save_path : str
        File path to save the figure.
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)
    residuals = pred - actual

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(residuals, bins=30, edgecolor="white", alpha=0.75, color="#2ca02c")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")

    res_mean = np.mean(residuals)
    res_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

    ax.axvline(res_mean, color="red", linestyle="-.", linewidth=1,
               label=f"Mean = {res_mean:.2f}")

    ax.set_xlabel("Residual (\u00b0F): Predicted \u2212 Actual")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{model_name}: Residual Distribution\n"
        f"Mean = {res_mean:.2f}\u00b0F  |  Std = {res_std:.2f}\u00b0F"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved residual histogram to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_residuals_by_month(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    dates: Union[pd.DatetimeIndex, np.ndarray, list],
    model_name: str,
    save_path: str,
    show: bool = False,
) -> None:
    """Box plot of residuals grouped by calendar month.

    Reveals seasonal bias patterns (e.g., consistent winter under-prediction).

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    dates : DatetimeIndex or array-like
        Dates corresponding to each prediction.
    model_name : str
        Name of the model.
    save_path : str
        File path to save the figure.
    show : bool
        If True, call ``plt.show()`` after saving.
    """
    actual, pred = _validate_inputs(y_actual, y_pred)
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    if len(dates) > len(actual):
        dates = dates[: len(actual)]

    residuals = pred - actual
    months = dates.month

    # Group residuals by month
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    data_by_month = []
    tick_labels = []
    for m in range(1, 13):
        mask = months == m
        if mask.any():
            data_by_month.append(residuals[mask])
            tick_labels.append(month_labels[m - 1])
        else:
            data_by_month.append([])
            tick_labels.append(month_labels[m - 1])

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(data_by_month, tick_labels=tick_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Residual (\u00b0F): Predicted \u2212 Actual")
    ax.set_title(f"{model_name}: Residuals by Month")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved residuals-by-month plot to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_baseline_comparison(
    results_dict: dict[str, dict],
    metric: str = "mae",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Bar chart comparing a selected metric across models.

    Parameters
    ----------
    results_dict : dict[str, dict]
        Mapping of model_name -> metrics dictionary.
    metric : str
        Key of the metric to compare (default "mae"). Must exist in
        each model's metrics dict.
    save_path : str, optional
        File path to save the figure. If None, the figure is not saved.
    show : bool
        If True, call ``plt.show()`` after saving.

    Raises
    ------
    ValueError
        If ``results_dict`` is empty.
    """
    if not results_dict:
        raise ValueError("results_dict is empty — nothing to plot")

    model_names = list(results_dict.keys())
    values = [results_dict[m].get(metric, float("nan")) for m in model_names]

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.5), 5))

    bars = ax.bar(model_names, values, color="#4c72b0", edgecolor="white")

    # Annotate each bar with its value
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(v for v in values if not np.isnan(v)),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    metric_label = metric.upper().replace("_", " ")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Baseline Model Comparison: {metric_label}")
    ax.set_xlabel("Model")

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved comparison bar chart to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


# ===========================================================================
# Report Generation
# ===========================================================================

def generate_baseline_report(
    results_dict: dict[str, dict],
    output_dir: str,
    dates_dict: Optional[dict[str, pd.DatetimeIndex]] = None,
    actuals_dict: Optional[dict[str, np.ndarray]] = None,
    preds_dict: Optional[dict[str, np.ndarray]] = None,
) -> str:
    """Generate a comprehensive baseline evaluation report.

    Computes and formats all metrics, saves the comparison table as
    a text file, generates all plots for each model, and returns the
    full report text.

    Parameters
    ----------
    results_dict : dict[str, dict]
        Mapping of model_name -> metrics dictionary.
    output_dir : str
        Directory to save the report text and plots.
    dates_dict : dict[str, DatetimeIndex], optional
        Mapping of model_name -> dates array. Required for time-series
        and monthly-residual plots.
    actuals_dict : dict[str, array-like], optional
        Mapping of model_name -> actual values. Required for plots.
    preds_dict : dict[str, array-like], optional
        Mapping of model_name -> predicted values. Required for plots.

    Returns
    -------
    str
        The full report text.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Comparison table
    table_text = format_metrics_table(results_dict)

    report_lines = [
        "=" * 70,
        "NYC Temperature Prediction — Baseline Evaluation Report",
        "=" * 70,
        "",
        table_text,
        "",
    ]

    # 2. Seasonal breakdown (if dates are available)
    if dates_dict and actuals_dict and preds_dict:
        report_lines.append("--- Seasonal Breakdown (MAE / RMSE / Bias) ---")
        report_lines.append("")

        for model_name in results_dict:
            if model_name in dates_dict and model_name in actuals_dict:
                seasonal = compute_seasonal_metrics(
                    actuals_dict[model_name],
                    preds_dict[model_name],
                    dates_dict[model_name],
                )
                report_lines.append(f"  {model_name}:")
                for season, sm in seasonal.items():
                    report_lines.append(
                        f"    {season}: MAE={sm['mae']:.2f}, "
                        f"RMSE={sm['rmse']:.2f}, Bias={sm['bias']:.2f} "
                        f"(n={sm['n']})"
                    )
                report_lines.append("")

    report_lines.extend(["=" * 70, ""])

    report_text = "\n".join(report_lines)

    # Save the text report
    report_path = os.path.join(output_dir, "baseline_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info("Saved evaluation report to %s", report_path)

    # 3. Comparison bar chart
    comparison_path = os.path.join(output_dir, "baseline_comparison_mae.png")
    plot_baseline_comparison(results_dict, metric="mae", save_path=comparison_path)

    # 4. Per-model diagnostic plots (if raw arrays are provided)
    if actuals_dict and preds_dict:
        for model_name in results_dict:
            if model_name not in actuals_dict or model_name not in preds_dict:
                continue

            safe_name = model_name.lower().replace(" ", "_")

            plot_actual_vs_predicted(
                actuals_dict[model_name],
                preds_dict[model_name],
                model_name,
                os.path.join(output_dir, f"{safe_name}_scatter.png"),
            )
            plot_residual_histogram(
                actuals_dict[model_name],
                preds_dict[model_name],
                model_name,
                os.path.join(output_dir, f"{safe_name}_residual_hist.png"),
            )

            if dates_dict and model_name in dates_dict:
                plot_time_series(
                    actuals_dict[model_name],
                    preds_dict[model_name],
                    dates_dict[model_name],
                    model_name,
                    os.path.join(output_dir, f"{safe_name}_timeseries.png"),
                )
                plot_residuals_by_month(
                    actuals_dict[model_name],
                    preds_dict[model_name],
                    dates_dict[model_name],
                    model_name,
                    os.path.join(output_dir, f"{safe_name}_residuals_month.png"),
                )

    return report_text


# ===========================================================================
# Convenience Function
# ===========================================================================

def evaluate_predictions(
    y_actual: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    dates: Union[pd.DatetimeIndex, np.ndarray, list, None] = None,
    model_name: str = "Model",
    output_dir: Optional[str] = None,
) -> dict:
    """One-call evaluation: compute metrics and optionally generate plots.

    Parameters
    ----------
    y_actual : array-like
        Actual observed values.
    y_pred : array-like
        Model predictions.
    dates : DatetimeIndex or array-like, optional
        Dates for time-aware plots. If None, date-dependent plots
        are skipped.
    model_name : str
        Label for the model.
    output_dir : str, optional
        If provided, all diagnostic plots are saved to this directory.

    Returns
    -------
    dict
        Metrics dictionary from ``compute_metrics``.
    """
    metrics = compute_metrics(y_actual, y_pred, model_name=model_name)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")

        plot_actual_vs_predicted(
            y_actual, y_pred, model_name,
            os.path.join(output_dir, f"{safe_name}_scatter.png"),
        )
        plot_residual_histogram(
            y_actual, y_pred, model_name,
            os.path.join(output_dir, f"{safe_name}_residual_hist.png"),
        )

        if dates is not None:
            plot_time_series(
                y_actual, y_pred, dates, model_name,
                os.path.join(output_dir, f"{safe_name}_timeseries.png"),
            )
            plot_residuals_by_month(
                y_actual, y_pred, dates, model_name,
                os.path.join(output_dir, f"{safe_name}_residuals_month.png"),
            )

            seasonal = compute_seasonal_metrics(y_actual, y_pred, dates)
            metrics["seasonal"] = seasonal

    logger.info("Evaluation complete for '%s'", model_name)
    return metrics
