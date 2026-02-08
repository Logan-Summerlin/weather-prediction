"""
Tests for the experiment framework (src/experiments.py).

Validates:
  - ExperimentConfig creation and defaults
  - ExperimentResult fields
  - select_features: all modes, column subsetting
  - run_experiment: tiny model, metrics returned
  - run_experiment_suite: batch execution, result DataFrame
  - generate_experiment_report: file creation, text content
  - Error handling: failed experiments, invalid configs
  - Feature subsetting correctness
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiments import (
    ExperimentConfig,
    ExperimentResult,
    select_features,
    run_experiment,
    run_experiment_suite,
    generate_experiment_report,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_feature_columns():
    """Generate feature column names matching the project structure."""
    stations = [
        "USW00014735", "USW00014740", "USW00094702",
        "USW00014732", "USW00093730",
    ]
    cols = []
    for st in stations:
        cols.append(f"{st}_TMAX_lag1")
        cols.append(f"{st}_TMIN_lag1")
    cols.extend(["sin_day", "cos_day"])
    return cols  # 12 columns: 5*2 + 2


@pytest.fixture
def feature_columns():
    """Return the 12 feature column names."""
    return _make_feature_columns()


@pytest.fixture
def synthetic_data():
    """Create small synthetic train/val/test data with correct columns.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test) as
    pandas DataFrames/Series.
    """
    np.random.seed(42)
    cols = _make_feature_columns()
    n_train, n_val, n_test = 100, 20, 20

    X_train = pd.DataFrame(
        np.random.randn(n_train, len(cols)), columns=cols,
    )
    y_train = pd.Series(
        60.0 + 10 * np.random.randn(n_train), name="NYC_TMAX",
    )
    X_val = pd.DataFrame(
        np.random.randn(n_val, len(cols)), columns=cols,
    )
    y_val = pd.Series(
        60.0 + 10 * np.random.randn(n_val), name="NYC_TMAX",
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, len(cols)), columns=cols,
    )
    y_test = pd.Series(
        60.0 + 10 * np.random.randn(n_test), name="NYC_TMAX",
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def tmp_dir():
    """Create and yield a temporary directory, then clean up."""
    d = tempfile.mkdtemp(prefix="test_exp_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# ExperimentConfig Tests
# ===========================================================================

class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        cfg = ExperimentConfig()
        assert cfg.name == "default"
        assert cfg.model_class == "enhanced_mlp"
        assert cfg.hidden_sizes == [128, 64]
        assert cfg.dropout == 0.1
        assert cfg.loss_type == "mse"
        assert cfg.learning_rate == 0.001
        assert cfg.batch_size == 64
        assert cfg.max_epochs == 50
        assert cfg.patience == 15
        assert cfg.features == "all"
        assert cfg.use_batch_norm is False

    def test_custom_values(self):
        """Custom values should override defaults."""
        cfg = ExperimentConfig(
            name="test_exp",
            model_class="lstm",
            hidden_sizes=[256],
            dropout=0.3,
            loss_type="huber",
            learning_rate=0.0001,
            max_epochs=10,
        )
        assert cfg.name == "test_exp"
        assert cfg.model_class == "lstm"
        assert cfg.hidden_sizes == [256]
        assert cfg.dropout == 0.3
        assert cfg.loss_type == "huber"
        assert cfg.learning_rate == 0.0001
        assert cfg.max_epochs == 10

    def test_model_kwargs_independent(self):
        """Each config should have independent model_kwargs."""
        cfg1 = ExperimentConfig(name="a")
        cfg2 = ExperimentConfig(name="b")
        cfg1.model_kwargs["x"] = 1
        assert "x" not in cfg2.model_kwargs

    def test_hidden_sizes_independent(self):
        """Each config should have independent hidden_sizes."""
        cfg1 = ExperimentConfig(name="a")
        cfg2 = ExperimentConfig(name="b")
        cfg1.hidden_sizes.append(999)
        assert 999 not in cfg2.hidden_sizes


# ===========================================================================
# ExperimentResult Tests
# ===========================================================================

class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_default_values(self):
        """Result should have default pending status."""
        r = ExperimentResult()
        assert r.status == "pending"
        assert r.config_name == ""
        assert np.isnan(r.mae)

    def test_custom_values(self):
        """Custom values should be stored."""
        r = ExperimentResult(
            config_name="test", mae=4.0, rmse=5.0, r2=0.85,
            status="success",
        )
        assert r.config_name == "test"
        assert r.mae == 4.0
        assert r.status == "success"


# ===========================================================================
# Feature Selection Tests
# ===========================================================================

class TestSelectFeatures:
    """Tests for select_features function."""

    def test_all_keeps_everything(self, synthetic_data):
        """'all' mode should return all columns."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "all")
        assert list(result.columns) == list(X_train.columns)

    def test_tmax_only(self, synthetic_data):
        """'tmax_only' should keep only TMAX columns + date."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "tmax_only")
        for col in result.columns:
            assert "TMAX" in col or col in ("sin_day", "cos_day")
        # Should have 5 TMAX + 2 date = 7 columns
        assert len(result.columns) == 7

    def test_tmin_only(self, synthetic_data):
        """'tmin_only' should keep only TMIN columns + date."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "tmin_only")
        for col in result.columns:
            assert "TMIN" in col or col in ("sin_day", "cos_day")
        assert len(result.columns) == 7

    def test_no_date(self, synthetic_data):
        """'no_date' should drop sin_day and cos_day."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "no_date")
        assert "sin_day" not in result.columns
        assert "cos_day" not in result.columns
        assert len(result.columns) == 10  # 5*2

    def test_invalid_mode_raises(self, synthetic_data):
        """Unknown mode should raise ValueError."""
        X_train = synthetic_data[0]
        with pytest.raises(ValueError, match="Unknown feature mode"):
            select_features(X_train, "invalid_mode")

    def test_case_insensitive(self, synthetic_data):
        """Mode should be case-insensitive."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "TMAX_ONLY")
        assert all("TMAX" in c or c in ("sin_day", "cos_day")
                    for c in result.columns)

    def test_preserves_values(self, synthetic_data):
        """Feature values should be preserved (not modified)."""
        X_train = synthetic_data[0]
        result = select_features(X_train, "tmax_only")
        tmax_col = [c for c in X_train.columns if "TMAX" in c][0]
        np.testing.assert_array_equal(
            result[tmax_col].values, X_train[tmax_col].values
        )


# ===========================================================================
# run_experiment Tests
# ===========================================================================

class TestRunExperiment:
    """Tests for run_experiment function."""

    def test_basic_experiment(self, synthetic_data, tmp_dir):
        """A basic experiment should complete successfully."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="basic_test",
            model_class="enhanced_mlp",
            hidden_sizes=[8],
            dropout=0.0,
            max_epochs=3,
            patience=3,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.status == "success"
        assert np.isfinite(result.mae)
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.r2)
        assert result.n_params > 0
        assert result.best_epoch > 0

    def test_lstm_experiment(self, synthetic_data, tmp_dir):
        """LSTM experiment should complete successfully."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="lstm_test",
            model_class="lstm",
            hidden_sizes=[16],
            dropout=0.0,
            max_epochs=3,
            patience=3,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.status == "success"
        assert np.isfinite(result.mae)

    def test_huber_loss(self, synthetic_data, tmp_dir):
        """Experiment with Huber loss should work."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="huber_test",
            model_class="enhanced_mlp",
            hidden_sizes=[8],
            loss_type="huber",
            max_epochs=3,
            patience=3,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.status == "success"

    def test_mae_loss(self, synthetic_data, tmp_dir):
        """Experiment with MAE loss should work."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="mae_test",
            model_class="enhanced_mlp",
            hidden_sizes=[8],
            loss_type="mae",
            max_epochs=3,
            patience=3,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.status == "success"

    def test_result_metrics_reasonable(self, synthetic_data, tmp_dir):
        """MAE, RMSE should be positive; R2 should be finite."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="metrics_test",
            model_class="enhanced_mlp",
            hidden_sizes=[16, 8],
            max_epochs=5,
            patience=5,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.mae > 0
        assert result.rmse > 0
        assert result.rmse >= result.mae  # RMSE >= MAE always

    def test_train_time_recorded(self, synthetic_data, tmp_dir):
        """Training time should be positive."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        cfg = ExperimentConfig(
            name="time_test",
            model_class="enhanced_mlp",
            hidden_sizes=[8],
            max_epochs=2,
            patience=2,
            batch_size=32,
        )
        result = run_experiment(
            cfg,
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            X_test.values, y_test.values,
            models_dir=tmp_dir,
        )
        assert result.train_time_s > 0


# ===========================================================================
# run_experiment_suite Tests
# ===========================================================================

class TestRunExperimentSuite:
    """Tests for run_experiment_suite function."""

    def test_suite_returns_dataframe(self, synthetic_data, tmp_dir):
        """Suite should return a pandas DataFrame."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        configs = [
            ExperimentConfig(name="exp_a", hidden_sizes=[8],
                             max_epochs=2, patience=2, batch_size=32),
            ExperimentConfig(name="exp_b", hidden_sizes=[16],
                             max_epochs=2, patience=2, batch_size=32),
        ]
        df = run_experiment_suite(
            configs,
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_dir=tmp_dir,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns
        assert "mae" in df.columns
        assert "status" in df.columns

    def test_suite_all_succeed(self, synthetic_data, tmp_dir):
        """All experiments should succeed with valid configs."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        configs = [
            ExperimentConfig(name="s1", hidden_sizes=[8],
                             max_epochs=2, patience=2, batch_size=32),
            ExperimentConfig(name="s2", hidden_sizes=[8],
                             loss_type="huber",
                             max_epochs=2, patience=2, batch_size=32),
        ]
        df = run_experiment_suite(
            configs,
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_dir=tmp_dir,
        )
        assert all(df["status"] == "success")

    def test_suite_with_feature_selection(self, synthetic_data, tmp_dir):
        """Suite should handle feature selection correctly."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        configs = [
            ExperimentConfig(name="all_feats", hidden_sizes=[8],
                             features="all",
                             max_epochs=2, patience=2, batch_size=32),
            ExperimentConfig(name="tmax_only", hidden_sizes=[8],
                             features="tmax_only",
                             max_epochs=2, patience=2, batch_size=32),
        ]
        df = run_experiment_suite(
            configs,
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_dir=tmp_dir,
        )
        assert len(df) == 2
        assert all(df["status"] == "success")

    def test_suite_expected_columns(self, synthetic_data, tmp_dir):
        """Result DataFrame should have all expected columns."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        configs = [
            ExperimentConfig(name="col_test", hidden_sizes=[8],
                             max_epochs=2, patience=2, batch_size=32),
        ]
        df = run_experiment_suite(
            configs,
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_dir=tmp_dir,
        )
        expected_cols = [
            "name", "model_class", "mae", "rmse", "r2", "bias",
            "best_epoch", "best_val_mae", "n_params", "train_time_s",
            "status",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


# ===========================================================================
# generate_experiment_report Tests
# ===========================================================================

class TestGenerateExperimentReport:
    """Tests for generate_experiment_report function."""

    def _make_results_df(self) -> pd.DataFrame:
        """Create a synthetic results DataFrame."""
        return pd.DataFrame([
            {
                "name": "Exp A", "model_class": "enhanced_mlp",
                "hidden_sizes": "[64, 32]", "dropout": 0.1,
                "loss_type": "mse", "features": "all",
                "mae": 4.1, "rmse": 5.5, "r2": 0.87,
                "bias": -0.1, "best_epoch": 20,
                "best_val_mae": 4.0, "n_params": 1000,
                "train_time_s": 5.0, "status": "success",
            },
            {
                "name": "Exp B", "model_class": "enhanced_mlp",
                "hidden_sizes": "[128, 64]", "dropout": 0.1,
                "loss_type": "huber", "features": "tmax_only",
                "mae": 4.5, "rmse": 5.8, "r2": 0.85,
                "bias": 0.2, "best_epoch": 15,
                "best_val_mae": 4.3, "n_params": 2000,
                "train_time_s": 8.0, "status": "success",
            },
            {
                "name": "Exp C", "model_class": "lstm",
                "hidden_sizes": "[64]", "dropout": 0.0,
                "loss_type": "mse", "features": "all",
                "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "bias": float("nan"),
                "best_epoch": 0, "best_val_mae": float("nan"),
                "n_params": 500, "train_time_s": 1.0,
                "status": "FAILED: NaN loss",
            },
        ])

    def test_report_creates_files(self, tmp_dir):
        """Report should create CSV and text files."""
        df = self._make_results_df()
        generate_experiment_report(df, output_dir=tmp_dir)

        assert os.path.isfile(os.path.join(tmp_dir, "experiment_results.csv"))
        assert os.path.isfile(os.path.join(tmp_dir, "experiment_report.txt"))

    def test_report_creates_plot(self, tmp_dir):
        """Report should create architecture comparison plot."""
        df = self._make_results_df()
        generate_experiment_report(df, output_dir=tmp_dir)
        assert os.path.isfile(
            os.path.join(tmp_dir, "architecture_comparison.png")
        )

    def test_report_text_content(self, tmp_dir):
        """Report text should contain experiment names and metrics."""
        df = self._make_results_df()
        report = generate_experiment_report(df, output_dir=tmp_dir)
        assert "Exp A" in report
        assert "Exp B" in report
        assert "4.1" in report or "4.100" in report
        assert "success" in report.lower() or "Successful" in report

    def test_report_mentions_failed(self, tmp_dir):
        """Report should mention failed experiments."""
        df = self._make_results_df()
        report = generate_experiment_report(df, output_dir=tmp_dir)
        assert "Failed" in report or "FAILED" in report

    def test_report_csv_roundtrip(self, tmp_dir):
        """CSV saved by report should be readable."""
        df = self._make_results_df()
        generate_experiment_report(df, output_dir=tmp_dir)

        csv_path = os.path.join(tmp_dir, "experiment_results.csv")
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3
        assert "mae" in loaded.columns

    def test_report_returns_string(self, tmp_dir):
        """Report function should return a non-empty string."""
        df = self._make_results_df()
        report = generate_experiment_report(df, output_dir=tmp_dir)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_empty_results(self, tmp_dir):
        """Report should handle all-failed results gracefully."""
        df = pd.DataFrame([{
            "name": "Bad", "model_class": "mlp",
            "hidden_sizes": "[8]", "dropout": 0.0,
            "loss_type": "mse", "features": "all",
            "mae": float("nan"), "rmse": float("nan"),
            "r2": float("nan"), "bias": float("nan"),
            "best_epoch": 0, "best_val_mae": float("nan"),
            "n_params": 100, "train_time_s": 0.5,
            "status": "FAILED: error",
        }])
        report = generate_experiment_report(df, output_dir=tmp_dir)
        assert "No successful experiments" in report

    def test_feature_ablation_plot(self, tmp_dir):
        """Feature ablation plot should be created when relevant."""
        df = self._make_results_df()
        generate_experiment_report(df, output_dir=tmp_dir)
        assert os.path.isfile(
            os.path.join(tmp_dir, "feature_ablation.png")
        )

    def test_reference_mae_in_report(self, tmp_dir):
        """Reference MAE should appear in the report."""
        df = self._make_results_df()
        report = generate_experiment_report(
            df, output_dir=tmp_dir, reference_mae=4.29,
        )
        assert "4.29" in report


# ===========================================================================
# Edge Case / Error Handling Tests
# ===========================================================================

class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_single_sample_data(self, tmp_dir):
        """Experiment should handle very small datasets."""
        np.random.seed(99)
        X = np.random.randn(5, 10)
        y = np.random.randn(5)

        cfg = ExperimentConfig(
            name="tiny",
            model_class="enhanced_mlp",
            hidden_sizes=[4],
            max_epochs=2,
            patience=2,
            batch_size=5,
        )
        result = run_experiment(
            cfg, X, y, X, y, X, y,
            models_dir=tmp_dir,
        )
        # Should complete (success or error, but no crash)
        assert result.status in ("success",) or result.status.startswith("FAILED")

    def test_experiment_config_name_in_result(self, tmp_dir):
        """Result config_name should match the config name."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)

        cfg = ExperimentConfig(
            name="name_match_test",
            model_class="enhanced_mlp",
            hidden_sizes=[4],
            max_epochs=2,
            patience=2,
            batch_size=10,
        )
        result = run_experiment(
            cfg, X, y, X, y, X, y,
            models_dir=tmp_dir,
        )
        assert result.config_name == "name_match_test"
