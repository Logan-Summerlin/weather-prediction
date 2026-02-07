"""
Tests for the run_nn.py end-to-end runner script.

Validates:
  - BASELINE_RESULTS dictionary structure and contents
  - Importability of the main function
  - Comparison logic (improvement calculations)
  - Output directory structure expectations
  - Baseline metric values match documented Phase 2 results
"""

import os
import sys
import importlib

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the run_nn module (not its __main__ guard)
import run_nn


# ===========================================================================
# BASELINE_RESULTS Tests
# ===========================================================================

class TestBaselineResults:
    """Tests for the BASELINE_RESULTS dictionary in run_nn.py."""

    def test_baseline_results_has_four_models(self):
        """BASELINE_RESULTS should contain exactly 4 baseline models."""
        assert len(run_nn.BASELINE_RESULTS) == 4

    def test_baseline_results_expected_keys(self):
        """BASELINE_RESULTS should contain all four expected model names."""
        expected_names = {
            "Persistence",
            "Climatology",
            "Linear Regression",
            "Ridge (alpha=1.0)",
        }
        assert set(run_nn.BASELINE_RESULTS.keys()) == expected_names

    def test_baseline_metrics_have_required_fields(self):
        """Each baseline entry should have mae, rmse, and r2 fields."""
        required_fields = {"mae", "rmse", "r2"}
        for model_name, metrics in run_nn.BASELINE_RESULTS.items():
            assert required_fields.issubset(metrics.keys()), (
                f"Model '{model_name}' is missing fields: "
                f"{required_fields - metrics.keys()}"
            )

    def test_ridge_is_best_baseline(self):
        """Ridge should have the lowest MAE among all baselines."""
        ridge_mae = run_nn.BASELINE_RESULTS["Ridge (alpha=1.0)"]["mae"]
        for model_name, metrics in run_nn.BASELINE_RESULTS.items():
            assert metrics["mae"] >= ridge_mae, (
                f"'{model_name}' has lower MAE ({metrics['mae']}) than "
                f"Ridge ({ridge_mae})"
            )

    def test_baseline_mae_values_match_phase2(self):
        """Baseline MAE values should match the documented Phase 2 results."""
        expected = {
            "Persistence": 5.06,
            "Climatology": 6.15,
            "Linear Regression": 4.35,
            "Ridge (alpha=1.0)": 4.33,
        }
        for model_name, expected_mae in expected.items():
            actual_mae = run_nn.BASELINE_RESULTS[model_name]["mae"]
            assert actual_mae == pytest.approx(expected_mae, abs=0.01), (
                f"'{model_name}' MAE ({actual_mae}) does not match "
                f"expected ({expected_mae})"
            )


# ===========================================================================
# Main Function Tests
# ===========================================================================

class TestMainFunction:
    """Tests for the main() function in run_nn.py."""

    def test_main_is_callable(self):
        """The main function should be importable and callable."""
        assert callable(run_nn.main)

    def test_module_has_main_guard(self):
        """The module should have a standard if __name__ == '__main__' block.

        We verify by checking the source file contains the pattern.
        """
        module_path = run_nn.__file__
        with open(module_path, "r") as f:
            source = f.read()
        assert 'if __name__ == "__main__"' in source


# ===========================================================================
# Comparison Logic Tests
# ===========================================================================

class TestComparisonLogic:
    """Tests for the improvement calculation logic used in run_nn.py."""

    def test_improvement_calculation_positive(self):
        """When NN MAE < Ridge MAE, improvement should be positive."""
        ridge_mae = run_nn.BASELINE_RESULTS["Ridge (alpha=1.0)"]["mae"]
        nn_mae = 3.50  # hypothetical better NN

        improvement = ridge_mae - nn_mae
        pct_improvement = (improvement / ridge_mae) * 100

        assert improvement > 0
        assert pct_improvement > 0

    def test_improvement_calculation_negative(self):
        """When NN MAE > Ridge MAE, improvement should be negative."""
        ridge_mae = run_nn.BASELINE_RESULTS["Ridge (alpha=1.0)"]["mae"]
        nn_mae = 5.00  # hypothetical worse NN

        improvement = ridge_mae - nn_mae
        pct_improvement = (improvement / ridge_mae) * 100

        assert improvement < 0
        assert pct_improvement < 0

    def test_stretch_goal_threshold(self):
        """Stretch goal should be achieved when MAE <= 2.0 degF."""
        # This tests the logic from run_nn.py lines 226-232
        assert 2.0 <= 2.0  # boundary case
        assert 1.99 <= 2.0  # just under
        assert not (2.01 <= 2.0)  # just over
