"""
Comprehensive tests for CHI/PHL remaining tasks implementation.

Tests cover:
  Task 1: WGA Architecture for CHI/PHL (wga_data_pipeline.py)
  Task 2: Extended E-Series Models (extended_models.py)
  Task 3: Model Checkpoint Persistence (model_checkpoint.py)
  Task 4: Live Trading Harness (live_trading.py)
  Task 5: Operational Dashboard Integration (dashboard/dashboard_data.py)
  Task 6: ASOS/NWP/Sounding Data Integration (operational_data.py)
"""

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ===================================================================
# Task 3: Model Checkpoint Persistence
# ===================================================================
class TestModelCheckpoint:
    """Tests for src/model_checkpoint.py."""

    def test_import(self):
        from src.model_checkpoint import (
            save_pytorch_model,
            load_pytorch_model,
            save_sklearn_model,
            load_sklearn_model,
            save_scaler,
            load_scaler,
            save_column_metadata,
            load_column_metadata,
            save_calibrator,
            load_calibrator,
            save_city_model_suite,
            list_saved_models,
        )

    def test_save_load_pytorch_model(self):
        """Save and load a PyTorch model."""
        from src.model_checkpoint import save_pytorch_model, load_pytorch_model

        class SimpleNet(nn.Module):
            def __init__(self, n_features=10, hidden_sizes=None, dropout=0.0):
                super().__init__()
                self.n_features = n_features
                self.hidden_sizes = hidden_sizes or [32, 16]
                self.dropout_rate = dropout
                self.fc = nn.Sequential(
                    nn.Linear(n_features, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2),
                )

            def forward(self, x):
                return self.fc(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleNet(n_features=10)
            # Set some non-default weights
            with torch.no_grad():
                model.fc[0].weight.fill_(0.42)

            path = save_pytorch_model(model, tmpdir, "test_model",
                                      metadata={"test_metric": 0.95})

            assert os.path.exists(path)
            assert os.path.exists(os.path.join(tmpdir, "test_model_meta.json"))

            loaded = load_pytorch_model(SimpleNet, tmpdir, "test_model",
                                        n_features=10)
            assert isinstance(loaded, SimpleNet)
            # Verify weights match
            assert torch.allclose(loaded.fc[0].weight,
                                  torch.full_like(loaded.fc[0].weight, 0.42))

    def test_save_load_scaler(self):
        """Save and load a scaler."""
        from src.model_checkpoint import save_scaler, load_scaler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a dict as a simple stand-in for a scaler
            scaler = {"mean": [1.0, 2.0], "std": [0.5, 1.0]}
            path = save_scaler(scaler, tmpdir)
            assert os.path.exists(path)

            loaded = load_scaler(tmpdir)
            assert loaded["mean"] == [1.0, 2.0]
            assert loaded["std"] == [0.5, 1.0]

    def test_save_load_column_metadata(self):
        """Save and load column metadata."""
        from src.model_checkpoint import save_column_metadata, load_column_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            cols = ["feat_a", "feat_b", "feat_c"]
            path = save_column_metadata(cols, tmpdir)
            assert os.path.exists(path)

            meta = load_column_metadata(tmpdir)
            assert meta["columns"] == cols
            assert meta["n_features"] == 3

    def test_save_load_calibrator(self):
        """Save and load calibrators."""
        from src.model_checkpoint import save_calibrator, load_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            cal = {"type": "isotonic", "params": [1, 2, 3]}
            path = save_calibrator(cal, tmpdir, "isotonic")
            assert os.path.exists(path)

            loaded = load_calibrator(tmpdir, "isotonic")
            assert loaded["type"] == "isotonic"

    def test_save_city_model_suite(self):
        """Save a complete model suite."""
        from src.model_checkpoint import save_city_model_suite, list_saved_models

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_city_model_suite(
                "chi",
                tmpdir,
                scaler={"mean": [0.0]},
                columns=["feat_a", "feat_b"],
                calibrators={"isotonic": {"type": "isotonic"}},
            )
            assert "scaler" in paths
            assert "col_metadata" in paths
            assert "calibrator_isotonic" in paths

            listing = list_saved_models(tmpdir)
            assert "sklearn" in listing
            assert "other" in listing

    def test_load_nonexistent_raises(self):
        """Loading nonexistent files should raise FileNotFoundError."""
        from src.model_checkpoint import load_scaler, load_calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_scaler(tmpdir, "nonexistent.pkl")
            with pytest.raises(FileNotFoundError):
                load_calibrator(tmpdir, "nonexistent")


# ===================================================================
# Task 4: Live Trading Harness
# ===================================================================
class TestLiveTradingHarness:
    """Tests for src/live_trading.py."""

    def test_import(self):
        from src.live_trading import (
            LiveTradingHarness,
            MultiCityTradingOrchestrator,
            KillSwitch,
            DailyPrediction,
            TradeRecord,
            get_kalshi_ticker,
            gaussian_to_bucket_probs,
        )

    def test_kalshi_ticker_routing(self):
        """Test Kalshi ticker routing for all cities."""
        from src.live_trading import get_kalshi_ticker

        assert get_kalshi_ticker("nyc") == "KXHIGHNY"
        assert get_kalshi_ticker("chi") == "KXHIGHCHI"
        assert get_kalshi_ticker("phl") == "KXHIGHPHL"

        with pytest.raises(ValueError):
            get_kalshi_ticker("unknown")

    def test_kill_switch_activation(self):
        """Test kill switch activation and deactivation."""
        from src.live_trading import KillSwitch

        ks = KillSwitch(city_code="chi", max_daily_loss=100.0)
        assert not ks.is_active

        ks.activate("Test reason")
        assert ks.is_active
        assert ks.reason == "Test reason"

        ks.deactivate()
        assert not ks.is_active

    def test_kill_switch_daily_loss_trigger(self):
        """Test automatic kill switch on daily loss threshold."""
        from src.live_trading import KillSwitch

        ks = KillSwitch(city_code="chi", max_daily_loss=50.0)
        ks.check_daily_loss(-20.0)
        assert not ks.is_active
        ks.check_daily_loss(-35.0)
        assert ks.is_active
        assert "Daily loss limit" in ks.reason

    def test_kill_switch_consecutive_losses(self):
        """Test kill switch on consecutive losses."""
        from src.live_trading import KillSwitch

        ks = KillSwitch(city_code="phl", max_consecutive_losses=3,
                        max_daily_loss=99999)
        ks.check_daily_loss(-1.0)
        ks.check_daily_loss(-1.0)
        assert not ks.is_active
        ks.check_daily_loss(-1.0)
        assert ks.is_active

    def test_kill_switch_reset_daily(self):
        """Test daily counter reset."""
        from src.live_trading import KillSwitch

        ks = KillSwitch(city_code="chi", max_daily_loss=100.0)
        ks.check_daily_loss(-30.0)
        assert ks.current_daily_loss == 30.0
        ks.reset_daily()
        assert ks.current_daily_loss == 0.0

    def test_gaussian_to_bucket_probs(self):
        """Test Gaussian to bucket probability conversion."""
        from src.live_trading import gaussian_to_bucket_probs

        edges = [(-999, 50), (50, 60), (60, 70), (70, 80), (80, 999)]
        probs = gaussian_to_bucket_probs(65.0, 5.0, edges)

        assert len(probs) == 5
        assert abs(sum(probs) - 1.0) < 0.01
        # Peak should be at the 60-70 bucket
        assert probs[2] == max(probs)

    def test_daily_prediction_serialization(self):
        """Test DailyPrediction serialization."""
        from src.live_trading import DailyPrediction

        pred = DailyPrediction(
            city_code="chi",
            date="2024-01-15",
            mu=65.0,
            sigma=5.0,
            bucket_probs=np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
            bucket_labels=["<50", "50-60", "60-70", "70-80", ">80"],
        )
        d = pred.to_dict()
        assert d["city_code"] == "chi"
        assert d["mu"] == 65.0
        assert len(d["bucket_probs"]) == 5

    def test_live_trading_harness_init(self):
        """Test LiveTradingHarness initialization for CHI and PHL."""
        from src.live_trading import LiveTradingHarness

        harness_chi = LiveTradingHarness("chi", mode="paper")
        assert harness_chi.city_code == "chi"
        assert harness_chi.kalshi_ticker == "KXHIGHCHI"
        assert harness_chi.mode == "paper"

        harness_phl = LiveTradingHarness("phl", mode="paper")
        assert harness_phl.city_code == "phl"
        assert harness_phl.kalshi_ticker == "KXHIGHPHL"

    def test_live_trading_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        from src.live_trading import LiveTradingHarness

        with pytest.raises(ValueError):
            LiveTradingHarness("chi", mode="invalid")

    def test_evaluate_trades_with_kill_switch(self):
        """Test that trades are blocked when kill switch is active."""
        from src.live_trading import LiveTradingHarness, DailyPrediction

        harness = LiveTradingHarness("chi", mode="paper")
        harness.kill_switch.activate("Test")

        pred = DailyPrediction(
            city_code="chi", date="2024-01-15", mu=65.0, sigma=5.0,
            bucket_probs=np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
            bucket_labels=["<50", "50-60", "60-70", "70-80", ">80"],
        )
        trades = harness.evaluate_trades(pred, {"60-70": 0.30})
        assert len(trades) == 0

    def test_multi_city_orchestrator(self):
        """Test MultiCityTradingOrchestrator."""
        from src.live_trading import MultiCityTradingOrchestrator

        orch = MultiCityTradingOrchestrator(["chi", "phl"], mode="paper")
        assert "chi" in orch.harnesses
        assert "phl" in orch.harnesses

        status = orch.get_status()
        assert "chi" in status
        assert "phl" in status

    def test_multi_city_kill_switch(self):
        """Test independent kill switches per city."""
        from src.live_trading import MultiCityTradingOrchestrator

        orch = MultiCityTradingOrchestrator(["chi", "phl"], mode="paper")
        orch.activate_kill_switch("chi", "Data issue")

        assert orch.harnesses["chi"].kill_switch.is_active
        assert not orch.harnesses["phl"].kill_switch.is_active

    def test_trade_record_serialization(self):
        """Test TradeRecord serialization."""
        from src.live_trading import TradeRecord

        trade = TradeRecord(
            city_code="chi", date="2024-01-15",
            ticker="KXHIGHCHI-2024-01-15", bucket_label="60-70",
            direction="YES", size=5, model_prob=0.40,
            market_price=0.30, ev=0.05, mode="paper",
        )
        d = trade.to_dict()
        assert d["city_code"] == "chi"
        assert d["direction"] == "YES"

    def test_audit_log_save(self):
        """Test saving audit logs."""
        from src.live_trading import LiveTradingHarness

        with tempfile.TemporaryDirectory() as tmpdir:
            harness = LiveTradingHarness("chi", mode="paper",
                                          audit_dir=tmpdir)
            path = harness.save_audit_log("2024-01-15")
            assert os.path.exists(path)

            with open(path) as f:
                audit = json.load(f)
            assert audit["city_code"] == "chi"
            assert audit["mode"] == "paper"

    def test_get_summary_empty(self):
        """Test get_summary with no trades."""
        from src.live_trading import LiveTradingHarness

        harness = LiveTradingHarness("chi", mode="paper")
        summary = harness.get_summary()
        assert summary["n_trades"] == 0
        assert summary["total_pnl"] == 0.0


# ===================================================================
# Task 6: ASOS/NWP/Sounding Data Integration
# ===================================================================
class TestOperationalData:
    """Tests for src/operational_data.py."""

    def test_import(self):
        from src.operational_data import (
            OperationalDataConfig,
            get_operational_config,
            verify_asos_coverage,
            get_nwp_config,
            get_sounding_config,
            get_nwp_feature_names,
            get_data_availability_summary,
        )

    def test_chi_operational_config(self):
        """Test operational config for Chicago."""
        from src.operational_data import get_operational_config

        cfg = get_operational_config("chi")
        assert cfg.city_code == "chi"
        assert cfg.primary_asos == "KORD"
        assert len(cfg.asos_stations) > 0
        assert cfg.igra_station_id != ""

    def test_phl_operational_config(self):
        """Test operational config for Philadelphia."""
        from src.operational_data import get_operational_config

        cfg = get_operational_config("phl")
        assert cfg.city_code == "phl"
        assert cfg.primary_asos == "KPHL"
        assert len(cfg.asos_stations) > 0

    def test_nwp_config(self):
        """Test NWP configuration."""
        from src.operational_data import get_nwp_config

        nwp_chi = get_nwp_config("chi")
        assert nwp_chi["city_code"] == "chi"
        assert "gfs" in nwp_chi["models"]
        assert "nam" in nwp_chi["models"]
        assert len(nwp_chi["variables"]) == 7

    def test_sounding_config(self):
        """Test IGRA sounding configuration."""
        from src.operational_data import get_sounding_config

        snd_chi = get_sounding_config("chi")
        assert snd_chi["city_code"] == "chi"
        assert snd_chi["station_id"] != ""
        assert 0 in snd_chi["hours"]
        assert 12 in snd_chi["hours"]

    def test_nwp_feature_names(self):
        """Test NWP feature name list."""
        from src.operational_data import get_nwp_feature_names

        feats = get_nwp_feature_names("chi")
        assert len(feats) > 10
        assert "nwp_tmax_2m" in feats
        assert "sounding_t850" in feats

    def test_data_availability_summary(self):
        """Test data availability summary structure."""
        from src.operational_data import get_data_availability_summary

        summary = get_data_availability_summary("chi")
        assert "asos" in summary
        assert "nwp" in summary
        assert "soundings" in summary
        assert summary["city_code"] == "chi"

    def test_asos_coverage_verification(self):
        """Test ASOS coverage verification."""
        from src.operational_data import verify_asos_coverage

        report = verify_asos_coverage("chi")
        assert report["city_code"] == "chi"
        assert report["total_stations"] > 0
        assert isinstance(report["coverage_pct"], float)


# ===================================================================
# Task 1: WGA Architecture for CHI/PHL
# ===================================================================
class TestWGADataPipeline:
    """Tests for src/wga_data_pipeline.py."""

    def test_import(self):
        from src.wga_data_pipeline import (
            WGADataBuilder,
            WGADataset,
            create_wga_dataloader,
        )

    def test_wga_builder_init_chi(self):
        """Test WGA data builder initialization for Chicago."""
        from src.wga_data_pipeline import WGADataBuilder

        builder = WGADataBuilder("chi")
        assert builder.city_code == "chi"
        assert builder.n_stations > 0
        assert builder.n_station_features == 2
        assert builder.n_metadata_features == 13
        assert builder.bearings_rad.shape[0] == builder.n_stations

    def test_wga_builder_init_phl(self):
        """Test WGA data builder initialization for Philadelphia."""
        from src.wga_data_pipeline import WGADataBuilder

        builder = WGADataBuilder("phl")
        assert builder.city_code == "phl"
        assert builder.n_stations > 0

    def test_wga_build_tensors_shape(self):
        """Test that build_tensors produces correctly shaped tensors."""
        from src.wga_data_pipeline import WGADataBuilder

        builder = WGADataBuilder("chi")
        n_days = 100
        n_stations = builder.n_stations

        # Create synthetic wide-format features
        cols = ["sin_day", "cos_day"]
        for sid in builder.station_order[:5]:  # Use first 5 stations
            cols.extend([f"{sid}_TMAX_lag1", f"{sid}_TMIN_lag1"])
        cols.append(f"{builder.city_config.target_station}_TMAX_lag1")

        X = pd.DataFrame(
            np.random.randn(n_days, len(cols)),
            columns=cols,
        )
        y = pd.Series(np.random.randn(n_days) * 10 + 65, name="TMAX")

        tensors = builder.build_tensors(X, y)

        assert tensors["station_features"].shape == (n_days, n_stations, 2)
        assert tensors["station_metadata"].shape == (n_days, n_stations, 13)
        assert tensors["station_bearings"].shape == (n_days, n_stations)
        assert tensors["wind_direction"].shape == (n_days,)
        assert tensors["station_mask"].shape == (n_days, n_stations)
        assert tensors["target"].shape == (n_days, 1)

    def test_wga_dataset(self):
        """Test WGADataset and DataLoader."""
        from src.wga_data_pipeline import WGADataBuilder, WGADataset, create_wga_dataloader

        builder = WGADataBuilder("chi")
        n_days = 50

        cols = ["sin_day", "cos_day"]
        for sid in builder.station_order[:3]:
            cols.extend([f"{sid}_TMAX_lag1", f"{sid}_TMIN_lag1"])

        X = pd.DataFrame(np.random.randn(n_days, len(cols)), columns=cols)
        y = pd.Series(np.random.randn(n_days) * 10 + 65)

        tensors = builder.build_tensors(X, y)
        dataset = WGADataset(tensors)
        assert len(dataset) == n_days

        loader = create_wga_dataloader(tensors, batch_size=16, shuffle=False)
        batch = next(iter(loader))
        assert batch["station_features"].shape[0] == 16

    def test_station_metadata_precomputed(self):
        """Test that station metadata is correctly precomputed."""
        from src.wga_data_pipeline import WGADataBuilder

        builder = WGADataBuilder("chi")

        # Check metadata array shape
        assert builder.metadata_array.shape == (builder.n_stations, 13)

        # Check bearing radians are valid
        assert np.all(builder.bearings_rad >= 0)
        assert np.all(builder.bearings_rad <= 2 * np.pi + 0.01)

        # Check distances are normalized
        assert np.all(builder.distances_norm >= 0)
        assert np.all(builder.distances_norm <= 1.01)


# ===================================================================
# Task 2: Extended E-Series Models
# ===================================================================
class TestExtendedModels:
    """Tests for src/extended_models.py."""

    def test_import(self):
        from src.extended_models import (
            HeteroscedasticNet,
            train_e6_regularized_nn,
            train_e7_wide_nn,
            train_e8_deep_regularized_nn,
            train_e9_ridge_stacker,
            train_e10_lasso_stacker,
            train_e11_elasticnet_stacker,
        )

    def test_heteroscedastic_net_forward(self):
        """Test HeteroscedasticNet forward pass."""
        from src.extended_models import HeteroscedasticNet

        net = HeteroscedasticNet(n_features=20, hidden_sizes=[64, 32])
        x = torch.randn(16, 20)
        mu, sigma = net(x)

        assert mu.shape == (16, 1)
        assert sigma.shape == (16, 1)
        assert (sigma > 0).all()  # sigma should be positive

    def test_heteroscedastic_net_with_batch_norm(self):
        """Test HeteroscedasticNet with batch normalization."""
        from src.extended_models import HeteroscedasticNet

        net = HeteroscedasticNet(
            n_features=15, hidden_sizes=[32, 16],
            use_batch_norm=True, dropout=0.2,
        )
        net.train()
        x = torch.randn(32, 15)
        mu, sigma = net(x)

        assert mu.shape == (32, 1)
        assert sigma.shape == (32, 1)

    def test_e9_ridge_stacker(self):
        """Test Ridge stacker on synthetic base model outputs."""
        from src.extended_models import train_e9_ridge_stacker

        n_val, n_test = 200, 50
        y_val = np.random.randn(n_val) * 10 + 65
        y_test = np.random.randn(n_test) * 10 + 65

        base_preds = {
            "model_a": (
                y_val + np.random.randn(n_val) * 3,
                np.full(n_val, 5.0),
                y_test + np.random.randn(n_test) * 3,
                np.full(n_test, 5.0),
            ),
            "model_b": (
                y_val + np.random.randn(n_val) * 4,
                np.full(n_val, 6.0),
                y_test + np.random.randn(n_test) * 4,
                np.full(n_test, 6.0),
            ),
        }

        result = train_e9_ridge_stacker(base_preds, y_val, y_test)
        assert "mu_val" in result
        assert "sigma_val" in result
        assert "mu_test" in result
        assert len(result["mu_val"]) == n_val
        assert len(result["mu_test"]) == n_test

    def test_e10_lasso_stacker(self):
        """Test Lasso stacker."""
        from src.extended_models import train_e10_lasso_stacker

        n_val, n_test = 200, 50
        y_val = np.random.randn(n_val) * 10 + 65
        y_test = np.random.randn(n_test) * 10 + 65

        base_preds = {
            "model_a": (
                y_val + np.random.randn(n_val) * 3,
                np.full(n_val, 5.0),
                y_test + np.random.randn(n_test) * 3,
                np.full(n_test, 5.0),
            ),
        }

        result = train_e10_lasso_stacker(base_preds, y_val, y_test)
        assert len(result["mu_test"]) == n_test

    def test_e11_elasticnet_stacker(self):
        """Test ElasticNet stacker."""
        from src.extended_models import train_e11_elasticnet_stacker

        n_val, n_test = 200, 50
        y_val = np.random.randn(n_val) * 10 + 65
        y_test = np.random.randn(n_test) * 10 + 65

        base_preds = {
            "model_a": (
                y_val + np.random.randn(n_val) * 3,
                np.full(n_val, 5.0),
                y_test + np.random.randn(n_test) * 3,
                np.full(n_test, 5.0),
            ),
        }

        result = train_e11_elasticnet_stacker(base_preds, y_val, y_test)
        assert len(result["mu_test"]) == n_test


# ===================================================================
# Task 5: Operational Dashboard
# ===================================================================
class TestDashboard:
    """Tests for src/dashboard/dashboard_data.py."""

    def test_import(self):
        from src.dashboard.dashboard_data import (
            DashboardData,
            HealthCheck,
            CityStatus,
        )

    def test_health_check_creation(self):
        """Test HealthCheck creation and serialization."""
        from src.dashboard.dashboard_data import HealthCheck

        check = HealthCheck(
            name="data_freshness", status="ok",
            message="Data is 2 days old",
        )
        assert check.status == "ok"
        d = check.to_dict()
        assert d["name"] == "data_freshness"
        assert "timestamp" in d

    def test_city_status_overall_health(self):
        """Test CityStatus overall health computation."""
        from src.dashboard.dashboard_data import CityStatus, HealthCheck

        status = CityStatus(
            city_code="chi",
            city_name="Chicago",
            health_checks=[
                HealthCheck("data", "ok", "fine"),
                HealthCheck("model", "warning", "stale"),
            ],
        )
        d = status.to_dict()
        assert d["overall_health"] == "warning"

    def test_city_status_critical(self):
        """Test CityStatus critical health."""
        from src.dashboard.dashboard_data import CityStatus, HealthCheck

        status = CityStatus(
            city_code="phl",
            health_checks=[
                HealthCheck("data", "ok", "fine"),
                HealthCheck("model", "critical", "missing"),
            ],
        )
        d = status.to_dict()
        assert d["overall_health"] == "critical"

    def test_dashboard_data_init(self):
        """Test DashboardData initialization."""
        from src.dashboard.dashboard_data import DashboardData

        dash = DashboardData(city_codes=["chi", "phl"])
        assert "chi" in dash.city_configs
        assert "phl" in dash.city_configs

    def test_dashboard_multi_city_status(self):
        """Test multi-city status retrieval."""
        from src.dashboard.dashboard_data import DashboardData

        dash = DashboardData(city_codes=["chi", "phl"])
        status = dash.get_multi_city_status()

        assert "cities" in status
        assert "cross_city" in status
        assert "timestamp" in status
        assert status["n_cities"] == 2

    def test_dashboard_export_json(self):
        """Test exporting dashboard status to JSON."""
        from src.dashboard.dashboard_data import DashboardData

        dash = DashboardData(city_codes=["chi"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "status.json")
            dash.export_status_json(output_path)
            assert os.path.exists(output_path)

            with open(output_path) as f:
                data = json.load(f)
            assert "cities" in data

    def test_dashboard_city_status(self):
        """Test individual city status retrieval."""
        from src.dashboard.dashboard_data import DashboardData

        dash = DashboardData(city_codes=["chi"])
        status = dash.get_city_status("chi")

        assert status.city_code == "chi"
        assert status.city_name == "Chicago"
        assert len(status.health_checks) > 0


# ===================================================================
# Cross-task integration tests
# ===================================================================
class TestCrossTaskIntegration:
    """Integration tests spanning multiple task implementations."""

    def test_checkpoint_with_live_trading(self):
        """Test that checkpoint-loaded models can feed into trading."""
        from src.model_checkpoint import save_column_metadata, load_column_metadata
        from src.live_trading import gaussian_to_bucket_probs, DailyPrediction

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save column metadata
            save_column_metadata(["feat_a", "feat_b"], tmpdir)
            meta = load_column_metadata(tmpdir)
            assert meta["n_features"] == 2

            # Generate predictions and bucket probs
            mu, sigma = 65.0, 5.0
            edges = [(-999, 50), (50, 60), (60, 70), (70, 80), (80, 999)]
            probs = gaussian_to_bucket_probs(mu, sigma, edges)

            pred = DailyPrediction(
                city_code="chi", date="2024-01-15",
                mu=mu, sigma=sigma,
                bucket_probs=probs,
                bucket_labels=["<50", "50-60", "60-70", "70-80", ">80"],
            )
            assert pred.bucket_probs.sum() > 0.99

    def test_wga_builder_with_both_cities(self):
        """Test WGA builder works for both CHI and PHL."""
        from src.wga_data_pipeline import WGADataBuilder

        chi_builder = WGADataBuilder("chi")
        phl_builder = WGADataBuilder("phl")

        # Both should have stations
        assert chi_builder.n_stations > 30
        assert phl_builder.n_stations > 30

        # Station orderings should be deterministic
        assert chi_builder.station_order == sorted(chi_builder.station_order)
        assert phl_builder.station_order == sorted(phl_builder.station_order)

    def test_operational_data_feeds_dashboard(self):
        """Test that operational data config feeds into dashboard."""
        from src.operational_data import get_operational_config
        from src.dashboard.dashboard_data import DashboardData

        # Get operational config
        cfg = get_operational_config("chi")
        assert cfg.primary_asos == "KORD"

        # Dashboard should work alongside
        dash = DashboardData(city_codes=["chi"])
        status = dash.get_city_status("chi")
        assert status.city_code == "chi"
