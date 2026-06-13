"""
Tests for Phase E ASOS Training Data Migration modules.

Covers:
  - src/asos_feature_builder.py:
      write_city_asos_mapping_csv, load_asos_daily_for_city,
      merge_asos_stations, compute_asos_completeness, build_asos_features
  - src/feature_parity.py:
      compare_feature_distributions, compare_asos_vs_ghcn_features,
      generate_parity_report, verify_tmax_parity, ParityResult
  - scripts/run_asos_migration.py:
      step functions, path helpers
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.asos_feature_builder import (
    write_city_asos_mapping_csv,
    load_asos_daily_for_city,
    merge_asos_stations,
    compute_asos_completeness,
    build_asos_features,
    _ASOS_TO_GHCN,
)
from src.feature_parity import (
    compare_feature_distributions,
    compare_asos_vs_ghcn_features,
    generate_parity_report,
    verify_tmax_parity,
    ParityResult,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_asos_daily_csv(
    tmp_path: Path,
    station_id: str,
    icao: str = "KORD",
    n_days: int = 365,
    start_date: str = "2020-01-01",
    tmax_base: float = 60.0,
    seed: int = 42,
) -> str:
    """Create a synthetic ASOS daily CSV for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    df = pd.DataFrame({
        "date": dates,
        "station_id": station_id,
        "icao": icao,
        "tmax_f": tmax_base + rng.randn(n_days) * 10,
        "tmin_f": (tmax_base - 15) + rng.randn(n_days) * 8,
        "tmean_f": (tmax_base - 7) + rng.randn(n_days) * 9,
        "dewpoint_mean_f": 45 + rng.randn(n_days) * 5,
        "dewpoint_afternoon_f": 48 + rng.randn(n_days) * 5,
        "wind_speed_mean_mph": 8 + rng.randn(n_days) * 3,
        "wind_speed_max_mph": 15 + rng.randn(n_days) * 5,
        "wind_dir_mean_deg": rng.uniform(0, 360, n_days),
        "wind_dir_evening_deg": rng.uniform(0, 360, n_days),
        "slp_00z_mb": 1013 + rng.randn(n_days) * 5,
        "slp_12z_mb": 1013 + rng.randn(n_days) * 5,
        "slp_tendency_24h_mb": rng.randn(n_days) * 2,
        "cloud_fraction_low": rng.uniform(0, 1, n_days),
        "obs_count": rng.randint(18, 24, n_days),
    })

    output_path = tmp_path / f"{station_id}_asos_daily.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)


def _make_city_asos_dir(tmp_path: Path, city_code: str = "chi") -> str:
    """Create a directory of ASOS daily CSVs for a city's stations."""
    from src.city_config import get_city_config

    cfg = get_city_config(city_code)
    asos_daily_dir = tmp_path / "asos_daily"
    asos_daily_dir.mkdir(parents=True, exist_ok=True)

    # Create CSVs for a subset of the city's ASOS stations (first 5)
    asos_map = cfg.asos_station_map
    for i, (station_id, icao) in enumerate(list(asos_map.items())[:5]):
        _make_asos_daily_csv(
            asos_daily_dir, station_id, icao=icao,
            n_days=365, seed=42 + i,
        )

    return str(asos_daily_dir)


# ===========================================================================
# write_city_asos_mapping_csv tests
# ===========================================================================

class TestWriteCityAsosMappingCsv:

    def test_writes_valid_csv(self, tmp_path):
        """Creates a mapping CSV with correct columns."""
        output = tmp_path / "mapping.csv"
        write_city_asos_mapping_csv("chi", str(output))

        assert output.exists()
        df = pd.read_csv(output)
        assert "station_id" in df.columns
        assert "icao" in df.columns
        assert "asos_available" in df.columns
        assert (df["asos_available"] == "yes").all()
        assert len(df) > 0

    def test_all_city_codes(self, tmp_path):
        """Works for all expansion cities."""
        for city in ["chi", "phl", "atl", "aus"]:
            output = tmp_path / f"{city}_mapping.csv"
            write_city_asos_mapping_csv(city, str(output))
            df = pd.read_csv(output)
            assert len(df) > 0, f"Empty mapping for {city}"

    def test_station_count_matches_config(self, tmp_path):
        """Number of stations matches city config's asos_station_map."""
        from src.city_config import get_city_config

        for city in ["chi", "phl", "atl", "aus"]:
            cfg = get_city_config(city)
            output = tmp_path / f"{city}_mapping.csv"
            write_city_asos_mapping_csv(city, str(output))
            df = pd.read_csv(output)
            assert len(df) == len(cfg.asos_station_map), (
                f"Mismatch for {city}: CSV has {len(df)} rows, "
                f"config has {len(cfg.asos_station_map)} ASOS stations"
            )


# ===========================================================================
# load_asos_daily_for_city tests
# ===========================================================================

class TestLoadAsosDailyForCity:

    def test_loads_existing_files(self, tmp_path):
        """Loads ASOS daily CSVs for stations that exist on disk."""
        asos_dir = _make_city_asos_dir(tmp_path, "chi")

        with patch("src.asos_feature_builder.get_city_config") as mock_cfg:
            cfg = MagicMock()
            cfg.asos_station_map = {"STATION_A": "KORD", "STATION_B": "KMDW"}
            cfg.start_date = ""
            cfg.end_date = ""
            mock_cfg.return_value = cfg

            # Create files for STATION_A only
            _make_asos_daily_csv(Path(asos_dir), "STATION_A", n_days=100)

            result = load_asos_daily_for_city("chi", asos_dir)
            assert "STATION_A" in result
            assert "STATION_B" not in result  # file doesn't exist
            assert len(result["STATION_A"]) == 100

    def test_empty_directory(self, tmp_path):
        """Returns empty dict when no ASOS files exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("src.asos_feature_builder.get_city_config") as mock_cfg:
            cfg = MagicMock()
            cfg.asos_station_map = {"STATION_A": "KORD"}
            cfg.start_date = ""
            cfg.end_date = ""
            mock_cfg.return_value = cfg

            result = load_asos_daily_for_city("chi", str(empty_dir))
            assert result == {}


# ===========================================================================
# merge_asos_stations tests
# ===========================================================================

class TestMergeAsosStations:

    def test_basic_merge(self):
        """Merges two station DataFrames into wide format."""
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        station_a = pd.DataFrame(
            {"tmax_f": np.arange(50, 60), "tmin_f": np.arange(30, 40)},
            index=dates,
        )
        station_b = pd.DataFrame(
            {"tmax_f": np.arange(55, 65), "tmin_f": np.arange(35, 45)},
            index=dates,
        )

        merged = merge_asos_stations({"ST_A": station_a, "ST_B": station_b})
        assert "ST_A_TMAX" in merged.columns
        assert "ST_A_TMIN" in merged.columns
        assert "ST_B_TMAX" in merged.columns
        assert "ST_B_TMIN" in merged.columns
        assert len(merged) == 10

    def test_custom_variables(self):
        """Only requested variables are included."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        station = pd.DataFrame(
            {"tmax_f": [60, 61, 62, 63, 64], "tmin_f": [40, 41, 42, 43, 44]},
            index=dates,
        )
        merged = merge_asos_stations({"ST": station}, variables=["TMAX"])
        assert "ST_TMAX" in merged.columns
        assert "ST_TMIN" not in merged.columns

    def test_empty_input_raises(self):
        """Empty station dict raises ValueError."""
        with pytest.raises(ValueError, match="No station data"):
            merge_asos_stations({})

    def test_misaligned_dates(self):
        """Stations with different date ranges merge with NaN fill."""
        dates_a = pd.date_range("2022-01-01", periods=10, freq="D")
        dates_b = pd.date_range("2022-01-05", periods=10, freq="D")
        station_a = pd.DataFrame(
            {"tmax_f": np.arange(50, 60)}, index=dates_a,
        )
        station_b = pd.DataFrame(
            {"tmax_f": np.arange(55, 65)}, index=dates_b,
        )
        merged = merge_asos_stations({"A": station_a, "B": station_b})
        # Total date range should span both
        assert len(merged) == 14  # Jan 1-14
        # A has data for first 10 days, NaN for last 4
        assert merged["A_TMAX"].isna().sum() == 4
        # B has data for last 10 days, NaN for first 4
        assert merged["B_TMAX"].isna().sum() == 4


# ===========================================================================
# compute_asos_completeness tests
# ===========================================================================

class TestComputeAsosCompleteness:

    def test_full_completeness(self):
        """All data present yields 100% completeness."""
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        merged = pd.DataFrame({
            "ST_TMAX": np.arange(100),
            "ST_TMIN": np.arange(100),
        }, index=dates)

        result = compute_asos_completeness(merged, ["ST"])
        tmax_row = result[result["variable"] == "TMAX"].iloc[0]
        assert tmax_row["completeness_pct"] == 1.0
        assert tmax_row["non_missing"] == 100

    def test_partial_completeness(self):
        """Missing values reduce completeness."""
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
        tmax = np.arange(100, dtype=float)
        tmax[50:] = np.nan
        merged = pd.DataFrame({"ST_TMAX": tmax}, index=dates)

        result = compute_asos_completeness(merged, ["ST"])
        tmax_row = result[result["variable"] == "TMAX"].iloc[0]
        assert tmax_row["completeness_pct"] == 0.5

    def test_missing_station(self):
        """Station not in merged DataFrame has 0% completeness."""
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        merged = pd.DataFrame({"OTHER_TMAX": np.arange(10)}, index=dates)

        result = compute_asos_completeness(merged, ["MISSING_ST"])
        assert result["completeness_pct"].iloc[0] == 0.0


# ===========================================================================
# compare_feature_distributions tests
# ===========================================================================

class TestCompareFeatureDistributions:

    def test_identical_distributions(self):
        """Identical data passes parity check."""
        rng = np.random.RandomState(42)
        data = rng.randn(1000, 3)
        cols = ["feat_a", "feat_b", "feat_c"]
        df = pd.DataFrame(data, columns=cols)

        results = compare_feature_distributions(df, df.copy())
        assert len(results) == 3
        assert all(r.parity_pass for r in results)
        assert all(r.ks_pvalue > 0.99 for r in results)

    def test_different_distributions(self):
        """Very different distributions fail parity check."""
        rng = np.random.RandomState(42)
        train = pd.DataFrame({"feat": rng.randn(500)})
        ref = pd.DataFrame({"feat": rng.randn(500) + 10})  # shifted mean

        results = compare_feature_distributions(train, ref)
        assert len(results) == 1
        assert not results[0].parity_pass
        assert results[0].ks_pvalue < 0.001

    def test_no_common_columns(self):
        """No common columns returns empty results."""
        train = pd.DataFrame({"a": [1, 2, 3]})
        ref = pd.DataFrame({"b": [1, 2, 3]})

        results = compare_feature_distributions(train, ref)
        assert results == []

    def test_custom_threshold(self):
        """Custom KS threshold is applied correctly."""
        rng = np.random.RandomState(42)
        train = pd.DataFrame({"feat": rng.randn(200)})
        ref = pd.DataFrame({"feat": rng.randn(200) + 0.3})  # slight shift

        # Very strict threshold -> more likely to fail
        strict = compare_feature_distributions(train, ref, ks_threshold=0.99)
        # Very lenient threshold -> more likely to pass
        lenient = compare_feature_distributions(train, ref, ks_threshold=0.001)

        # With shifted data and strict threshold, should fail
        if strict:
            assert not strict[0].parity_pass
        # With lenient threshold, p-value just needs to be > 0.001
        # (may still fail with large enough shift)

    def test_nan_handling(self):
        """NaN values are excluded from comparison."""
        train = pd.DataFrame({"feat": [1.0, 2.0, np.nan, 4.0, 5.0]})
        ref = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})

        results = compare_feature_distributions(train, ref)
        assert len(results) == 1


# ===========================================================================
# compare_asos_vs_ghcn_features tests
# ===========================================================================

class TestCompareAsosVsGhcnFeatures:

    def test_from_csv_files(self, tmp_path):
        """Loads and compares features from CSV files."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        cols = ["feat_a", "feat_b", "feat_c"]

        asos_path = tmp_path / "asos_features.csv"
        ghcn_path = tmp_path / "ghcn_features.csv"

        pd.DataFrame(data, columns=cols).to_csv(asos_path, index=False)
        pd.DataFrame(data + 0.01, columns=cols).to_csv(ghcn_path, index=False)

        results = compare_asos_vs_ghcn_features(str(asos_path), str(ghcn_path))
        assert len(results) == 3


# ===========================================================================
# generate_parity_report tests
# ===========================================================================

class TestGenerateParityReport:

    def test_generates_json_and_markdown(self, tmp_path):
        """Both JSON and Markdown reports are generated."""
        results = [
            ParityResult(
                feature_name="feat_a",
                ks_statistic=0.05,
                ks_pvalue=0.8,
                train_mean=50.0,
                train_std=10.0,
                reference_mean=50.1,
                reference_std=10.0,
                mean_diff=-0.1,
                parity_pass=True,
            ),
            ParityResult(
                feature_name="feat_b",
                ks_statistic=0.3,
                ks_pvalue=0.001,
                train_mean=60.0,
                train_std=12.0,
                reference_mean=55.0,
                reference_std=11.0,
                mean_diff=5.0,
                parity_pass=False,
            ),
        ]

        md_path = generate_parity_report(results, str(tmp_path), "chi")

        assert os.path.exists(md_path)
        assert md_path.endswith(".md")

        json_path = os.path.join(str(tmp_path), "chi_feature_parity.json")
        assert os.path.exists(json_path)

        with open(json_path) as fh:
            data = json.load(fh)
        assert data["total_features"] == 2
        assert data["passed"] == 1
        assert data["failed"] == 1
        assert data["all_pass"] is False

    def test_all_pass_report(self, tmp_path):
        """Report with all passing features shows PASS overall."""
        results = [
            ParityResult(
                feature_name="feat",
                ks_statistic=0.02,
                ks_pvalue=0.95,
                train_mean=50.0,
                train_std=10.0,
                reference_mean=50.0,
                reference_std=10.0,
                mean_diff=0.0,
                parity_pass=True,
            ),
        ]

        generate_parity_report(results, str(tmp_path), "atl")

        json_path = os.path.join(str(tmp_path), "atl_feature_parity.json")
        with open(json_path) as fh:
            data = json.load(fh)
        assert data["all_pass"] is True


# ===========================================================================
# verify_tmax_parity tests
# ===========================================================================

class TestVerifyTmaxParity:

    def test_identical_series(self):
        """Identical TMAX series passes all checks."""
        idx = pd.date_range("2022-01-01", periods=365, freq="D")
        rng = np.random.RandomState(42)
        tmax = pd.Series(60 + rng.randn(365) * 10, index=idx)

        result = verify_tmax_parity(tmax, tmax.copy())
        assert result["overall_pass"] is True
        assert result["checks"]["bias_below_threshold"] is True
        assert result["checks"]["correlation_above_0.95"] is True
        assert result["checks"]["ks_pvalue_above_0.01"] is True

    def test_large_bias_fails(self):
        """Large bias between ASOS and GHCN fails the check."""
        idx = pd.date_range("2022-01-01", periods=365, freq="D")
        rng = np.random.RandomState(42)
        asos = pd.Series(60 + rng.randn(365) * 10, index=idx)
        ghcn = asos - 5.0  # 5°F systematic bias

        result = verify_tmax_parity(asos, ghcn, max_acceptable_bias=2.0)
        assert result["checks"]["bias_below_threshold"] is False
        assert abs(result["mean_bias"]) > 2.0

    def test_low_correlation_fails(self):
        """Uncorrelated series fail the correlation check."""
        idx = pd.date_range("2022-01-01", periods=365, freq="D")
        rng = np.random.RandomState(42)
        asos = pd.Series(rng.randn(365) * 10, index=idx)
        ghcn = pd.Series(rng.randn(365) * 10, index=idx)

        result = verify_tmax_parity(asos, ghcn)
        assert result["checks"]["correlation_above_0.95"] is False

    def test_no_shared_index(self):
        """Non-overlapping indices still compute unpaired metrics."""
        asos = pd.Series([50, 60, 70], index=pd.date_range("2022-01-01", periods=3))
        ghcn = pd.Series([50, 60, 70], index=pd.date_range("2023-01-01", periods=3))

        result = verify_tmax_parity(asos, ghcn)
        # Should still compute something (unpaired bias = 0)
        assert result["asos_n"] == 3
        assert result["ghcn_n"] == 3


# ===========================================================================
# build_asos_features integration test (mocked data)
# ===========================================================================

class TestBuildAsosFeatures:

    def test_full_pipeline(self, tmp_path):
        """Integration test: full ASOS feature building pipeline."""
        # Create synthetic ASOS daily data for a minimal station network
        asos_dir = tmp_path / "asos_daily"
        asos_dir.mkdir()
        output_dir = tmp_path / "processed"

        # Create data for target + 2 surrounding stations
        target_sid = "TARGET_ST"
        surr_sids = ["SURR_A", "SURR_B"]
        all_sids = [target_sid] + surr_sids

        for i, sid in enumerate(all_sids):
            _make_asos_daily_csv(
                asos_dir, sid, n_days=1000,
                start_date="2018-01-01", seed=42 + i,
            )

        # Mock the city config
        mock_cfg = MagicMock()
        mock_cfg.city_code = "chi"
        mock_cfg.target_station = target_sid
        mock_cfg.asos_station_map = {sid: f"K{sid}" for sid in all_sids}
        mock_cfg.all_stations = {sid: f"Station {sid}" for sid in all_sids}
        mock_cfg.surrounding_stations = {sid: f"Station {sid}" for sid in surr_sids}
        mock_cfg.start_date = "2018-01-01"
        mock_cfg.end_date = "2020-09-30"
        mock_cfg.min_completeness = 0.5
        mock_cfg.max_forward_fill_days = 7
        mock_cfg.train_ratio = 0.7
        mock_cfg.val_ratio = 0.15
        mock_cfg.input_variables = ["TMAX", "TMIN"]

        with patch("src.asos_feature_builder.get_city_config", return_value=mock_cfg):
            result = build_asos_features(
                city_code="chi",
                asos_daily_dir=str(asos_dir),
                output_dir=str(output_dir),
            )

        # Check outputs
        assert "X_train" in result
        assert "y_train" in result
        assert "scaler" in result
        assert "col_means" in result

        assert len(result["X_train"]) > 0
        assert len(result["X_val"]) > 0
        assert len(result["X_test"]) > 0

        # Check files were saved
        assert (output_dir / "features_train.csv").exists()
        assert (output_dir / "features_val.csv").exists()
        assert (output_dir / "features_test.csv").exists()
        assert (output_dir / "target_train.csv").exists()
        assert (output_dir / "scaler.pkl").exists()
        assert (output_dir / "col_means.pkl").exists()

        # Verify no data leakage: train dates < val dates < test dates
        train_max = result["X_train"].index.max()
        val_min = result["X_val"].index.min()
        test_min = result["X_test"].index.min()
        assert train_max < val_min
        assert val_min <= test_min

    def test_no_data_raises(self, tmp_path):
        """Raises FileNotFoundError when no ASOS data exists."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_cfg = MagicMock()
        mock_cfg.city_code = "chi"
        mock_cfg.asos_station_map = {"ST": "KORD"}
        mock_cfg.start_date = ""
        mock_cfg.end_date = ""
        mock_cfg.min_completeness = 0.5

        with patch("src.asos_feature_builder.get_city_config", return_value=mock_cfg):
            with pytest.raises(FileNotFoundError, match="No ASOS daily data"):
                build_asos_features("chi", str(empty_dir), str(output_dir))


# ===========================================================================
# Migration script step tests
# ===========================================================================

class TestMigrationSteps:

    def test_step1_write_mapping(self, tmp_path):
        """Step 1 creates a valid mapping CSV."""
        import dataclasses

        from src.city_config import get_city_config

        # Copy the config: get_city_config returns the shared registry
        # singleton, and mutating it in place pollutes every later test.
        cfg = dataclasses.replace(
            get_city_config("chi"), data_dir=str(tmp_path / "data"),
        )
        os.makedirs(cfg.data_dir, exist_ok=True)

        from scripts.run_asos_migration import step1_write_mapping

        mapping_csv = step1_write_mapping(cfg, "chi")
        assert os.path.exists(mapping_csv)

        df = pd.read_csv(mapping_csv)
        assert len(df) > 0
        assert "station_id" in df.columns

    def test_path_helpers(self, tmp_path):
        """Path helpers generate consistent city-specific paths."""
        from scripts.run_asos_migration import (
            _city_asos_raw_dir,
            _city_asos_daily_dir,
            _city_asos_mapping_csv,
            _city_migration_report_dir,
            _city_asos_processed_dir,
        )

        cfg = MagicMock()
        cfg.data_dir = "/data/chicago"
        cfg.results_dir = "/results/chicago"

        assert _city_asos_raw_dir(cfg) == "/data/chicago/raw/asos"
        assert _city_asos_daily_dir(cfg) == "/data/chicago/processed/asos_daily"
        assert _city_asos_mapping_csv(cfg) == "/data/chicago/asos_station_mapping.csv"
        assert _city_migration_report_dir(cfg) == "/results/chicago/asos_migration"
        assert _city_asos_processed_dir(cfg) == "/data/chicago/processed"
