"""
Dashboard Data Aggregation for Multi-City Temperature Prediction.

Provides classes and functions to aggregate model predictions, market data,
trading results, and operational health metrics across all cities
(NYC, CHI, PHL) for the operational dashboard.

Usage:
    from src.dashboard.dashboard_data import DashboardData
    dash = DashboardData()
    status = dash.get_multi_city_status()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from src.city_config import get_city_config, list_cities, CityConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check definitions
# ---------------------------------------------------------------------------

@dataclass
class HealthCheck:
    """Result of a single health check.

    Attributes
    ----------
    name : str
        Human-readable check name.
    status : str
        "ok", "warning", or "critical".
    message : str
        Details about the check result.
    timestamp : str
        ISO timestamp when the check was performed.
    """

    name: str
    status: str  # "ok", "warning", "critical"
    message: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# City status
# ---------------------------------------------------------------------------

@dataclass
class CityStatus:
    """Operational status summary for a single city.

    Attributes
    ----------
    city_code : str
        City identifier.
    city_name : str
        Human-readable city name.
    data_freshness : str
        Timestamp of latest processed data.
    model_status : str
        "trained", "stale", or "missing".
    latest_brier : float
        Most recent benchmark Brier score.
    best_model : str
        Name of the best-performing model.
    trading_mode : str
        "live", "paper", or "disabled".
    health_checks : list of HealthCheck
        Results of operational health checks.
    n_active_positions : int
        Number of currently open trading positions.
    daily_pnl : float
        Today's P&L.
    cumulative_pnl : float
        Cumulative P&L.
    """

    city_code: str
    city_name: str = ""
    data_freshness: str = ""
    model_status: str = "missing"
    latest_brier: float = float("nan")
    best_model: str = ""
    trading_mode: str = "disabled"
    health_checks: List[HealthCheck] = field(default_factory=list)
    n_active_positions: int = 0
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city_code": self.city_code,
            "city_name": self.city_name,
            "data_freshness": self.data_freshness,
            "model_status": self.model_status,
            "latest_brier": self.latest_brier,
            "best_model": self.best_model,
            "trading_mode": self.trading_mode,
            "health_checks": [h.to_dict() for h in self.health_checks],
            "n_active_positions": self.n_active_positions,
            "daily_pnl": self.daily_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "overall_health": self._overall_health(),
        }

    def _overall_health(self) -> str:
        """Compute overall health from individual checks."""
        if not self.health_checks:
            return "unknown"
        statuses = [h.status for h in self.health_checks]
        if "critical" in statuses:
            return "critical"
        if "warning" in statuses:
            return "warning"
        return "ok"


# ---------------------------------------------------------------------------
# Dashboard data aggregator
# ---------------------------------------------------------------------------

class DashboardData:
    """Aggregates operational data across all cities for dashboard display.

    Scans results directories, trading logs, and model checkpoints to
    build a comprehensive operational status view.

    Parameters
    ----------
    city_codes : list of str, optional
        Cities to include. Defaults to all registered cities.
    """

    def __init__(self, city_codes: Optional[List[str]] = None):
        self.city_codes = city_codes or list_cities()
        self.city_configs = {
            code: get_city_config(code) for code in self.city_codes
        }

    def get_city_status(self, city_code: str) -> CityStatus:
        """Get operational status for a single city.

        Parameters
        ----------
        city_code : str
            City identifier.

        Returns
        -------
        CityStatus
            Comprehensive status for the city.
        """
        cfg = self.city_configs[city_code]
        status = CityStatus(
            city_code=city_code,
            city_name=cfg.city_name,
        )

        # Check data freshness
        status.health_checks.append(self._check_data_freshness(cfg))

        # Check model availability
        model_check = self._check_model_availability(cfg)
        status.health_checks.append(model_check)
        status.model_status = (
            "trained" if model_check.status == "ok" else "missing"
        )

        # Load benchmark results
        benchmark = self._load_benchmark_results(cfg)
        if benchmark:
            status.latest_brier = benchmark.get("best_brier", float("nan"))
            status.best_model = benchmark.get("best_model", "")

        # Load trading status
        trading = self._load_trading_status(cfg)
        if trading:
            status.trading_mode = trading.get("mode", "disabled")
            status.daily_pnl = trading.get("daily_pnl", 0.0)
            status.cumulative_pnl = trading.get("cumulative_pnl", 0.0)
            status.n_active_positions = trading.get("n_active_positions", 0)

        # Check calibration
        status.health_checks.append(self._check_calibration(cfg))

        return status

    def get_multi_city_status(self) -> Dict[str, Any]:
        """Get operational status for all cities.

        Returns
        -------
        dict
            Multi-city dashboard data with keys:
            - "cities": dict of city_code → CityStatus
            - "cross_city": cross-city correlation and summary
            - "timestamp": ISO timestamp
        """
        cities = {}
        for code in self.city_codes:
            try:
                cities[code] = self.get_city_status(code).to_dict()
            except Exception as e:
                logger.error("Failed to get status for %s: %s", code, e)
                cities[code] = {"city_code": code, "error": str(e)}

        return {
            "cities": cities,
            "cross_city": self._compute_cross_city_summary(cities),
            "timestamp": datetime.utcnow().isoformat(),
            "n_cities": len(self.city_codes),
        }

    # -----------------------------------------------------------------------
    # Health check implementations
    # -----------------------------------------------------------------------

    def _check_data_freshness(self, cfg: CityConfig) -> HealthCheck:
        """Check if processed data is recent enough."""
        processed_dir = os.path.join(cfg.data_dir, "processed")

        if not os.path.exists(processed_dir):
            return HealthCheck(
                name="data_freshness",
                status="critical",
                message=f"Processed data directory missing: {processed_dir}",
            )

        # Check modification time of features_test.csv
        test_file = os.path.join(processed_dir, "features_test.csv")
        if not os.path.exists(test_file):
            return HealthCheck(
                name="data_freshness",
                status="warning",
                message="features_test.csv missing",
            )

        mtime = datetime.fromtimestamp(os.path.getmtime(test_file))
        age_days = (datetime.now() - mtime).days

        if age_days > 30:
            return HealthCheck(
                name="data_freshness",
                status="warning",
                message=f"Data is {age_days} days old (last modified: {mtime.isoformat()})",
            )

        return HealthCheck(
            name="data_freshness",
            status="ok",
            message=f"Data updated {age_days} days ago",
        )

    def _check_model_availability(self, cfg: CityConfig) -> HealthCheck:
        """Check if trained models exist for the city."""
        models_dir = cfg.models_dir

        if not os.path.exists(models_dir):
            return HealthCheck(
                name="model_availability",
                status="warning",
                message=f"Models directory missing: {models_dir}",
            )

        # Check for any .pt files
        pt_files = [
            f for f in os.listdir(models_dir) if f.endswith(".pt")
        ] if os.path.exists(models_dir) else []

        if not pt_files:
            return HealthCheck(
                name="model_availability",
                status="warning",
                message="No trained model checkpoints found",
            )

        return HealthCheck(
            name="model_availability",
            status="ok",
            message=f"{len(pt_files)} model checkpoints available",
        )

    def _check_calibration(self, cfg: CityConfig) -> HealthCheck:
        """Check if calibration artifacts exist."""
        cal_dir = os.path.join(cfg.models_dir, "calibrators")

        if not os.path.exists(cal_dir):
            return HealthCheck(
                name="calibration",
                status="warning",
                message="No calibrators directory found",
            )

        cal_files = [
            f for f in os.listdir(cal_dir) if f.endswith(".pkl")
        ]

        if not cal_files:
            return HealthCheck(
                name="calibration",
                status="warning",
                message="No calibration files found",
            )

        return HealthCheck(
            name="calibration",
            status="ok",
            message=f"{len(cal_files)} calibrators available",
        )

    # -----------------------------------------------------------------------
    # Data loading helpers
    # -----------------------------------------------------------------------

    def _load_benchmark_results(self, cfg: CityConfig) -> Optional[Dict]:
        """Load the latest benchmark results JSON for a city."""
        results_path = os.path.join(
            cfg.results_dir, "unified_benchmark_results.json"
        )
        if not os.path.exists(results_path):
            return None

        try:
            with open(results_path) as f:
                results = json.load(f)

            # Find best model
            best_model = None
            best_brier = float("inf")
            for name, data in results.items():
                brier = data.get(
                    "contract_brier",
                    data.get("test_brier", float("inf")),
                )
                if isinstance(brier, (int, float)) and brier < best_brier:
                    best_brier = brier
                    best_model = name

            return {
                "best_model": best_model,
                "best_brier": best_brier,
                "n_models": len(results),
                "results_path": results_path,
            }
        except Exception as e:
            logger.error("Failed to load benchmark for %s: %s",
                        cfg.city_code, e)
            return None

    def _load_trading_status(self, cfg: CityConfig) -> Optional[Dict]:
        """Load the latest trading status for a city."""
        trading_dir = os.path.join(cfg.results_dir, "trading")
        if not os.path.exists(trading_dir):
            return None

        # Find most recent audit log
        audit_files = sorted([
            f for f in os.listdir(trading_dir)
            if f.startswith("trading_audit_") and f.endswith(".json")
        ])

        if not audit_files:
            return None

        try:
            with open(os.path.join(trading_dir, audit_files[-1])) as f:
                audit = json.load(f)
            return {
                "mode": audit.get("mode", "unknown"),
                "daily_pnl": audit.get("total_pnl", 0.0),
                "n_active_positions": audit.get("n_trades", 0),
                "cumulative_pnl": 0.0,  # Would need to aggregate all audits
            }
        except Exception:
            return None

    def _compute_cross_city_summary(
        self, cities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute cross-city summary metrics."""
        briers = {}
        for code, data in cities.items():
            if isinstance(data, dict) and "latest_brier" in data:
                brier = data["latest_brier"]
                if isinstance(brier, (int, float)) and not np.isnan(brier):
                    briers[code] = brier

        return {
            "n_healthy": sum(
                1 for d in cities.values()
                if isinstance(d, dict)
                and d.get("overall_health") == "ok"
            ),
            "n_warning": sum(
                1 for d in cities.values()
                if isinstance(d, dict)
                and d.get("overall_health") == "warning"
            ),
            "n_critical": sum(
                1 for d in cities.values()
                if isinstance(d, dict)
                and d.get("overall_health") == "critical"
            ),
            "best_brier_by_city": briers,
            "avg_brier": (
                float(np.mean(list(briers.values())))
                if briers else float("nan")
            ),
        }

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_status_json(self, output_path: str) -> str:
        """Export multi-city status to JSON file.

        Parameters
        ----------
        output_path : str
            Path to write the JSON file.

        Returns
        -------
        str
            Path to the written file.
        """
        status = self.get_multi_city_status()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(status, f, indent=2, default=str)
        logger.info("Exported dashboard status to %s", output_path)
        return output_path
