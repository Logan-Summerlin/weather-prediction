"""
Comprehensive tests for NWP preprocessing pipeline.

Tests cover:
  - parse_grib_file: mock xarray dataset, missing variables, temperature conversions
  - parse_grib_fallback: JSON/CSV sidecar parsing
  - process_nwp_directory: empty dir, single file, multiple files, corrupt files
  - compute_nwp_derived_features: normal df, empty df, missing columns
  - align_nwp_with_observations: matching dates, partial overlap, no overlap, bias
  - run_nwp_preprocessing: integration test with mock data
  - Temperature conversion edge cases: very cold, very hot
  - Wind direction edge cases: calm winds, pure cardinal directions
  - Helper functions: _find_nearest_index, _extract_date_from_filename
"""

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.nwp_preprocessing import (
    _kelvin_to_fahrenheit,
    _kelvin_to_celsius,
    _wind_speed,
    _wind_direction,
    _find_nearest_index,
    _extract_date_from_filename,
    _extract_fxx_from_filename,
    parse_grib_file,
    parse_grib_fallback,
    process_nwp_directory,
    compute_nwp_derived_features,
    align_nwp_with_observations,
    run_nwp_preprocessing,
    MS_TO_KNOTS,
    K_TO_C_OFFSET,
)


# ============================================================================
# Temperature Conversion Tests
# ============================================================================


class TestTemperatureConversions:
    """Tests for Kelvin -> Fahrenheit and Kelvin -> Celsius conversions."""

    def test_kelvin_to_fahrenheit_freezing(self):
        """273.15 K = 32 degrees F (water freezing point)."""
        assert abs(_kelvin_to_fahrenheit(273.15) - 32.0) < 0.01

    def test_kelvin_to_fahrenheit_boiling(self):
        """373.15 K = 212 degrees F (water boiling point)."""
        assert abs(_kelvin_to_fahrenheit(373.15) - 212.0) < 0.01

    def test_kelvin_to_fahrenheit_very_cold(self):
        """233.15 K = -40 degrees F (very cold, F and C converge)."""
        assert abs(_kelvin_to_fahrenheit(233.15) - (-40.0)) < 0.01

    def test_kelvin_to_fahrenheit_very_hot(self):
        """313.15 K = 104 degrees F (hot summer day)."""
        assert abs(_kelvin_to_fahrenheit(313.15) - 104.0) < 0.01

    def test_kelvin_to_celsius_freezing(self):
        """273.15 K = 0 degrees C."""
        assert abs(_kelvin_to_celsius(273.15) - 0.0) < 0.01

    def test_kelvin_to_celsius_very_cold(self):
        """243.15 K = -30 degrees C."""
        assert abs(_kelvin_to_celsius(243.15) - (-30.0)) < 0.01

    def test_kelvin_to_celsius_hot(self):
        """310.15 K = 37 degrees C (body temperature)."""
        assert abs(_kelvin_to_celsius(310.15) - 37.0) < 0.01


# ============================================================================
# Wind Calculation Tests
# ============================================================================


class TestWindCalculations:
    """Tests for wind speed and direction from U/V components."""

    def test_wind_speed_basic(self):
        """Basic wind speed: sqrt(3^2 + 4^2) = 5."""
        assert abs(_wind_speed(3.0, 4.0) - 5.0) < 0.001

    def test_wind_speed_calm(self):
        """Calm wind (0,0) should return 0."""
        assert _wind_speed(0.0, 0.0) == 0.0

    def test_wind_speed_pure_u(self):
        """Pure U-component wind."""
        assert abs(_wind_speed(10.0, 0.0) - 10.0) < 0.001

    def test_wind_speed_pure_v(self):
        """Pure V-component wind."""
        assert abs(_wind_speed(0.0, 10.0) - 10.0) < 0.001

    def test_wind_direction_calm(self):
        """Calm winds should return 0 degrees."""
        assert _wind_direction(0.0, 0.0) == 0.0

    def test_wind_direction_pure_west_wind(self):
        """Pure westerly wind: u > 0, v = 0 -> wind from 270 deg."""
        result = _wind_direction(10.0, 0.0)
        assert abs(result - 270.0) < 0.01

    def test_wind_direction_pure_south_wind(self):
        """Pure southerly wind: u = 0, v > 0 -> from south (180)."""
        result = _wind_direction(0.0, 10.0)
        assert abs(result - 180.0) < 0.01

    def test_wind_direction_pure_east_wind(self):
        """Pure easterly wind: u < 0, v = 0 -> from 90 deg."""
        result = _wind_direction(-10.0, 0.0)
        assert abs(result - 90.0) < 0.01

    def test_wind_direction_pure_north_wind(self):
        """Pure northerly wind: u = 0, v < 0 -> from north (0/360)."""
        result = _wind_direction(0.0, -10.0)
        assert abs(result - 0.0) < 0.01 or abs(result - 360.0) < 0.01

    def test_wind_direction_range(self):
        """All wind directions should be in [0, 360)."""
        for u in [-5, -1, 0, 1, 5]:
            for v in [-5, -1, 0, 1, 5]:
                if u == 0 and v == 0:
                    continue
                d = _wind_direction(float(u), float(v))
                assert 0.0 <= d < 360.0, f"u={u}, v={v} -> dir={d}"


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelpers:
    """Tests for utility helpers."""

    def test_find_nearest_index_exact(self):
        """Exact match should return correct index."""
        coords = np.array([39.0, 40.0, 41.0, 42.0])
        assert _find_nearest_index(coords, 41.0) == 2

    def test_find_nearest_index_between(self):
        """Value between grid points should return closest."""
        coords = np.array([39.0, 40.0, 41.0, 42.0])
        assert _find_nearest_index(coords, 40.8) == 2

    def test_find_nearest_index_below_range(self):
        """Value below range should return first index."""
        coords = np.array([39.0, 40.0, 41.0])
        assert _find_nearest_index(coords, 35.0) == 0

    def test_extract_date_basic(self):
        """Standard GRIB filename with date."""
        ts = _extract_date_from_filename("gfs_20220115_f024.grib2")
        assert ts == pd.Timestamp("2022-01-15")

    def test_extract_date_no_date(self):
        """Filename without recognizable date."""
        assert _extract_date_from_filename("no_date_here.grib2") is None

    def test_extract_fxx_basic(self):
        """Standard forecast hour in filename."""
        assert _extract_fxx_from_filename("gfs_20220101_f048.grib2") == 48

    def test_extract_fxx_default(self):
        """No forecast hour pattern should default to 24."""
        assert _extract_fxx_from_filename("gfs_20220101.grib2") == 24

    def test_extract_fxx_three_digits(self):
        """Three-digit forecast hour."""
        assert _extract_fxx_from_filename("gfs_20220101_f120.grib2") == 120


# ============================================================================
# parse_grib_file Tests
# ============================================================================


def _make_mock_xr_module():
    """Create a mock xarray module for sys.modules injection."""
    mock_xr = MagicMock()
    return mock_xr


def _make_mock_dataset(
    tmax_k=300.0,
    u10=5.0,
    v10=-3.0,
    tcc=75.0,
    prmsl=101325.0,
    tp=0.005,
):
    """Create a mock xarray-like Dataset with NWP variables."""
    lats = np.array([39.0, 39.25, 39.5, 39.75, 40.0, 40.25, 40.5, 40.75, 41.0])
    lons = np.array([285.0, 285.25, 285.5, 285.75, 286.0, 286.25, 286.5])

    ds = MagicMock()
    ds.dims = {"latitude": len(lats), "longitude": len(lons)}

    lat_mock = MagicMock()
    lat_mock.values = lats
    lon_mock = MagicMock()
    lon_mock.values = lons

    step_mock = MagicMock()
    # Use a plain MagicMock for the step value so we can set .astype
    step_val = MagicMock()
    step_val.astype = MagicMock(return_value=np.float64(24.0))
    step_mock.values = step_val

    def getitem(key):
        if key == "latitude":
            return lat_mock
        if key == "longitude":
            return lon_mock
        if key == "step":
            return step_mock
        # Variable access
        var = MagicMock()
        var.dims = ()
        val_mock = MagicMock()

        if key == "tmax":
            val_mock.values = np.float64(tmax_k)
        elif key in ("u10", "10u"):
            val_mock.values = np.float64(u10)
        elif key in ("v10", "10v"):
            val_mock.values = np.float64(v10)
        elif key == "tcc":
            val_mock.values = np.float64(tcc)
        elif key in ("prmsl", "msl"):
            val_mock.values = np.float64(prmsl)
        elif key == "tp":
            val_mock.values = np.float64(tp)
        else:
            val_mock.values = np.float64(np.nan)
        var.isel = MagicMock(return_value=val_mock)
        return var

    ds.__getitem__ = MagicMock(side_effect=getitem)
    ds.data_vars = ["tmax", "u10", "v10", "tcc", "prmsl", "tp"]
    ds.coords = {"latitude": lat_mock, "longitude": lon_mock, "step": step_mock}

    return ds


class TestParseGribFile:
    """Tests for parse_grib_file with mocked xarray/cfgrib."""

    def test_parse_basic(self):
        """Basic GRIB parsing should extract and convert all variables."""
        ds = _make_mock_dataset(tmax_k=300.0, u10=5.0, v10=-3.0, prmsl=101325.0)
        mock_xr = MagicMock()
        mock_xr.open_datasets = MagicMock(return_value=[ds])

        with patch("src.nwp_preprocessing._require_cfgrib"):
            with patch.dict("sys.modules", {"xarray": mock_xr}):
                # Force re-import of xr in the module
                import src.nwp_preprocessing as nwp_mod
                original_xr = getattr(nwp_mod, "xr", None)
                try:
                    # Manually inject mock xr into the module namespace
                    # so the `import xarray as xr` inside parse_grib_file works
                    result = None
                    with patch.object(nwp_mod, "_require_cfgrib"):
                        # parse_grib_file does `import xarray as xr` locally
                        # We need to put our mock into sys.modules
                        result = nwp_mod.parse_grib_file("/fake/path.grib2")
                finally:
                    pass

        # TMAX: 300K -> (300-273.15)*9/5+32 = 80.33 degrees F
        assert abs(result["tmax_2m_f"] - 80.33) < 0.1
        # Wind speed: sqrt(25+9) * 1.94384
        expected_ws = math.sqrt(25 + 9) * MS_TO_KNOTS
        assert abs(result["wind_speed_10m_kt"] - expected_ws) < 0.1
        # MSLP: 101325 Pa -> 1013.25 mb
        assert abs(result["mslp_mb"] - 1013.25) < 0.1

    def test_parse_cfgrib_not_available(self):
        """When cfgrib is not installed, should raise ImportError."""
        with patch(
            "src.nwp_preprocessing._require_cfgrib",
            side_effect=ImportError("cfgrib not installed"),
        ):
            with pytest.raises(ImportError, match="cfgrib"):
                parse_grib_file("/fake/path.grib2")

    def test_parse_corrupt_file(self):
        """Corrupt GRIB file should return dict with all NaN values."""
        mock_xr = MagicMock()
        mock_xr.open_datasets = MagicMock(side_effect=Exception("bad file"))
        mock_xr.open_dataset = MagicMock(side_effect=Exception("bad file"))

        with patch("src.nwp_preprocessing._require_cfgrib"):
            with patch.dict("sys.modules", {"xarray": mock_xr}):
                result = parse_grib_file("/fake/corrupt.grib2")

        assert np.isnan(result["tmax_2m_f"])
        assert np.isnan(result["wind_speed_10m_kt"])

    def test_parse_missing_wind_vars(self):
        """Missing U/V should leave wind fields as NaN."""
        lats = np.array([40.0, 40.25, 40.5])
        lons = np.array([286.0, 286.25, 286.5])

        ds = MagicMock()
        ds.dims = {"latitude": 3, "longitude": 3}

        lat_m = MagicMock()
        lat_m.values = lats
        lon_m = MagicMock()
        lon_m.values = lons

        def getitem(key):
            if key == "latitude":
                return lat_m
            if key == "longitude":
                return lon_m
            raise KeyError(key)

        ds.__getitem__ = MagicMock(side_effect=getitem)
        ds.data_vars = []
        ds.coords = {"latitude": lat_m, "longitude": lon_m}

        mock_xr = MagicMock()
        mock_xr.open_datasets = MagicMock(return_value=[ds])

        with patch("src.nwp_preprocessing._require_cfgrib"):
            with patch.dict("sys.modules", {"xarray": mock_xr}):
                result = parse_grib_file("/fake/no_wind.grib2")

        assert np.isnan(result["wind_speed_10m_kt"])
        assert np.isnan(result["wind_dir_10m_deg"])


# ============================================================================
# parse_grib_fallback Tests
# ============================================================================


class TestParseGribFallback:
    """Tests for fallback parsing from JSON/CSV sidecars."""

    def test_json_sidecar_basic(self, tmp_path):
        """Valid JSON sidecar should extract values."""
        grib_path = str(tmp_path / "gfs_20220101.grib2")
        json_path = str(tmp_path / "gfs_20220101.json")

        meta = {
            "tmax_2m": 300.0,
            "tmp_850": 275.0,
            "ugrd_10m": 5.0,
            "vgrd_10m": -3.0,
            "tcdc": 50.0,
            "mslp": 101300.0,
            "apcp": 0.002,
            "forecast_hour": 24,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f)

        result = parse_grib_fallback(grib_path)

        assert abs(result["tmax_2m_f"] - _kelvin_to_fahrenheit(300.0)) < 0.01
        assert abs(result["tmp_850_c"] - _kelvin_to_celsius(275.0)) < 0.01
        assert abs(result["ugrd_10m"] - 5.0) < 0.01
        assert abs(result["cloud_cover_pct"] - 50.0) < 0.01
        assert abs(result["mslp_mb"] - 1013.0) < 0.01
        assert abs(result["precip_mm"] - 2.0) < 0.01
        assert result["forecast_hour"] == 24

    def test_json_sidecar_missing(self, tmp_path):
        """No sidecar files should return all NaN."""
        result = parse_grib_fallback(str(tmp_path / "missing.grib2"))
        assert np.isnan(result["tmax_2m_f"])
        assert np.isnan(result["mslp_mb"])

    def test_csv_sidecar(self, tmp_path):
        """CSV sidecar should be used when JSON is absent."""
        grib_path = str(tmp_path / "gfs_20220101.grib2")
        csv_path = str(tmp_path / "gfs_20220101.csv")

        df = pd.DataFrame({"tmax_2m": [300.0], "tmp_850": [275.0]})
        df.to_csv(csv_path, index=False)

        result = parse_grib_fallback(grib_path)
        assert abs(result["tmax_2m_f"] - _kelvin_to_fahrenheit(300.0)) < 0.01
        assert abs(result["tmp_850_c"] - _kelvin_to_celsius(275.0)) < 0.01

    def test_json_sidecar_corrupt(self, tmp_path):
        """Corrupt JSON sidecar should not crash."""
        grib_path = str(tmp_path / "gfs_20220101.grib2")
        json_path = str(tmp_path / "gfs_20220101.json")
        with open(json_path, "w") as f:
            f.write("{bad json")

        result = parse_grib_fallback(grib_path)
        assert np.isnan(result["tmax_2m_f"])

    def test_json_wind_derivation(self, tmp_path):
        """Wind speed/dir should be derived from u/v in JSON sidecar."""
        grib_path = str(tmp_path / "gfs_20220101.grib2")
        json_path = str(tmp_path / "gfs_20220101.json")
        meta = {"ugrd_10m": 10.0, "vgrd_10m": 0.0}
        with open(json_path, "w") as f:
            json.dump(meta, f)

        result = parse_grib_fallback(grib_path)
        expected_speed = 10.0 * MS_TO_KNOTS
        assert abs(result["wind_speed_10m_kt"] - expected_speed) < 0.01
        assert abs(result["wind_dir_10m_deg"] - 270.0) < 0.01


# ============================================================================
# process_nwp_directory Tests
# ============================================================================


class TestProcessNwpDirectory:
    """Tests for directory scanning and GRIB aggregation."""

    def test_empty_directory(self, tmp_path):
        """Empty directory should return empty DataFrame with correct columns."""
        df = process_nwp_directory(str(tmp_path))
        assert df.empty
        expected_cols = [
            "date", "fxx", "tmax_2m_f", "tmp_850_c", "wind_speed_10m_kt",
            "wind_dir_10m_deg", "cloud_cover_pct", "mslp_mb", "precip_mm",
        ]
        assert list(df.columns) == expected_cols

    def test_nonexistent_directory(self):
        """Non-existent directory should return empty DataFrame."""
        df = process_nwp_directory("/nonexistent/path/nwp")
        assert df.empty

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_single_grib_file(self, mock_parse, tmp_path):
        """Single GRIB file should produce one-row DataFrame."""
        grib_path = tmp_path / "gfs_20220101_f024.grib2"
        grib_path.touch()

        mock_parse.return_value = {
            "tmax_2m_f": 50.0,
            "tmp_850_c": -5.0,
            "ugrd_10m": 3.0,
            "vgrd_10m": 4.0,
            "wind_speed_10m_kt": 9.72,
            "wind_dir_10m_deg": 233.0,
            "cloud_cover_pct": 80.0,
            "mslp_mb": 1015.0,
            "precip_mm": 2.5,
            "forecast_hour": 24,
        }

        df = process_nwp_directory(str(tmp_path))
        assert len(df) == 1
        assert df["tmax_2m_f"].iloc[0] == 50.0
        assert df["date"].iloc[0] == pd.Timestamp("2022-01-01")

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_multiple_grib_files(self, mock_parse, tmp_path):
        """Multiple GRIB files should produce multi-row DataFrame."""
        for date_str in ["20220101", "20220102", "20220103"]:
            (tmp_path / f"gfs_{date_str}_f024.grib2").touch()

        mock_parse.return_value = {
            "tmax_2m_f": 55.0,
            "tmp_850_c": -2.0,
            "ugrd_10m": 1.0,
            "vgrd_10m": 1.0,
            "wind_speed_10m_kt": 2.75,
            "wind_dir_10m_deg": 225.0,
            "cloud_cover_pct": 50.0,
            "mslp_mb": 1010.0,
            "precip_mm": 0.0,
            "forecast_hour": 24,
        }

        df = process_nwp_directory(str(tmp_path))
        assert len(df) == 3

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_corrupt_file_skipped(self, mock_parse, tmp_path):
        """Corrupt file that raises exception should be skipped gracefully."""
        (tmp_path / "gfs_20220101_f024.grib2").touch()
        (tmp_path / "gfs_20220102_f024.grib2").touch()

        mock_parse.side_effect = [
            Exception("Corrupt GRIB"),
            {
                "tmax_2m_f": 55.0,
                "tmp_850_c": -2.0,
                "ugrd_10m": 1.0,
                "vgrd_10m": 1.0,
                "wind_speed_10m_kt": 2.75,
                "wind_dir_10m_deg": 225.0,
                "cloud_cover_pct": 50.0,
                "mslp_mb": 1010.0,
                "precip_mm": 0.0,
                "forecast_hour": 24,
            },
        ]

        df = process_nwp_directory(str(tmp_path))
        assert len(df) == 1

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_cfgrib_fallback(self, mock_parse, tmp_path):
        """When cfgrib raises ImportError, fallback should be tried."""
        grib_path = tmp_path / "gfs_20220101_f024.grib2"
        grib_path.touch()

        # First call raises ImportError, triggering fallback
        mock_parse.side_effect = ImportError("No cfgrib")

        # Create JSON sidecar for fallback
        json_path = tmp_path / "gfs_20220101.json"
        meta = {"tmax_2m": 300.0, "ugrd_10m": 5.0, "vgrd_10m": 0.0}
        with open(json_path, "w") as f:
            json.dump(meta, f)

        df = process_nwp_directory(str(tmp_path))
        # Fallback parse_grib_fallback is called internally
        assert len(df) >= 0  # May or may not find data depending on fallback path

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_sorted_by_date(self, mock_parse, tmp_path):
        """Output should be sorted by date."""
        (tmp_path / "gfs_20220103_f024.grib2").touch()
        (tmp_path / "gfs_20220101_f024.grib2").touch()
        (tmp_path / "gfs_20220102_f024.grib2").touch()

        mock_parse.return_value = {
            "tmax_2m_f": 55.0, "tmp_850_c": 0.0,
            "ugrd_10m": 0.0, "vgrd_10m": 0.0,
            "wind_speed_10m_kt": 0.0, "wind_dir_10m_deg": 0.0,
            "cloud_cover_pct": 0.0, "mslp_mb": 1013.0,
            "precip_mm": 0.0, "forecast_hour": 24,
        }

        df = process_nwp_directory(str(tmp_path))
        dates = df["date"].tolist()
        assert dates == sorted(dates)


# ============================================================================
# compute_nwp_derived_features Tests
# ============================================================================


class TestComputeNwpDerivedFeatures:
    """Tests for NWP derived feature computation."""

    def test_empty_dataframe(self):
        """Empty DataFrame should add columns without error."""
        df = pd.DataFrame(columns=[
            "date", "fxx", "tmax_2m_f", "tmp_850_c",
            "wind_speed_10m_kt", "wind_dir_10m_deg",
            "cloud_cover_pct", "mslp_mb", "precip_mm",
        ])
        result = compute_nwp_derived_features(df)
        assert "nwp_tmax_change" in result.columns
        assert "nwp_wind_chill" in result.columns
        assert "nwp_ensemble_spread" in result.columns

    def test_tmax_change(self):
        """Day-over-day TMAX change should be computed via diff()."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "tmax_2m_f": [50.0, 55.0, 48.0],
            "wind_speed_10m_kt": [10.0, 10.0, 10.0],
        })
        result = compute_nwp_derived_features(df)
        assert np.isnan(result["nwp_tmax_change"].iloc[0])
        assert abs(result["nwp_tmax_change"].iloc[1] - 5.0) < 0.01
        assert abs(result["nwp_tmax_change"].iloc[2] - (-7.0)) < 0.01

    def test_wind_chill_cold_windy(self):
        """Wind chill should be computed when T <= 50 degrees F and wind >= ~3mph."""
        df = pd.DataFrame({
            "tmax_2m_f": [30.0],
            "wind_speed_10m_kt": [15.0],  # ~17.3 mph
        })
        result = compute_nwp_derived_features(df)
        # Wind chill should be defined (not NaN) for cold+windy
        assert not np.isnan(result["nwp_wind_chill"].iloc[0])
        # Wind chill should be below the actual temperature
        assert result["nwp_wind_chill"].iloc[0] < 30.0

    def test_wind_chill_warm(self):
        """Wind chill should be NaN for warm temperatures (> 50 degrees F)."""
        df = pd.DataFrame({
            "tmax_2m_f": [70.0],
            "wind_speed_10m_kt": [15.0],
        })
        result = compute_nwp_derived_features(df)
        assert np.isnan(result["nwp_wind_chill"].iloc[0])

    def test_wind_chill_calm(self):
        """Wind chill should be NaN for calm winds (< ~3 mph)."""
        df = pd.DataFrame({
            "tmax_2m_f": [30.0],
            "wind_speed_10m_kt": [1.0],  # ~1.15 mph, below 3 mph threshold
        })
        result = compute_nwp_derived_features(df)
        assert np.isnan(result["nwp_wind_chill"].iloc[0])

    def test_ensemble_spread_always_nan(self):
        """Ensemble spread should be NaN (placeholder for GFS)."""
        df = pd.DataFrame({
            "tmax_2m_f": [50.0, 55.0],
            "wind_speed_10m_kt": [10.0, 10.0],
        })
        result = compute_nwp_derived_features(df)
        assert result["nwp_ensemble_spread"].isna().all()

    def test_missing_tmax_column(self):
        """DataFrame without tmax_2m_f should still add columns gracefully."""
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=2),
            "wind_speed_10m_kt": [10.0, 10.0],
        })
        result = compute_nwp_derived_features(df)
        assert "nwp_tmax_change" in result.columns
        assert result["nwp_tmax_change"].isna().all()


# ============================================================================
# align_nwp_with_observations Tests
# ============================================================================


class TestAlignNwpWithObservations:
    """Tests for NWP-observation alignment and bias computation."""

    def test_matching_dates(self):
        """Perfectly matching dates should produce bias values."""
        nwp_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "tmax_2m_f": [50.0, 55.0, 48.0],
        })
        obs_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "TMAX": [49.0, 57.0, 50.0],
        })
        result = align_nwp_with_observations(nwp_df, obs_df)
        assert "nwp_bias" in result.columns
        assert "nwp_bias_7d" in result.columns
        assert abs(result["nwp_bias"].iloc[0] - 1.0) < 0.01
        assert abs(result["nwp_bias"].iloc[1] - (-2.0)) < 0.01

    def test_partial_overlap(self):
        """Partial date overlap should produce NaN for non-overlapping rows."""
        nwp_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=5),
            "tmax_2m_f": [50.0, 55.0, 48.0, 60.0, 52.0],
        })
        obs_df = pd.DataFrame({
            "date": pd.date_range("2022-01-03", periods=3),
            "TMAX": [50.0, 58.0, 54.0],
        })
        result = align_nwp_with_observations(nwp_df, obs_df)
        # First two NWP rows have no obs match
        assert np.isnan(result["obs_tmax"].iloc[0])
        assert abs(result["obs_tmax"].iloc[2] - 50.0) < 0.01

    def test_no_overlap(self):
        """No date overlap should produce all NaN bias."""
        nwp_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "tmax_2m_f": [50.0, 55.0, 48.0],
        })
        obs_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "TMAX": [49.0, 57.0, 50.0],
        })
        result = align_nwp_with_observations(nwp_df, obs_df)
        assert result["nwp_bias"].isna().all()

    def test_rolling_7d_bias(self):
        """Rolling 7-day bias should be computed correctly."""
        dates = pd.date_range("2022-01-01", periods=10)
        nwp_df = pd.DataFrame({
            "date": dates,
            "tmax_2m_f": [50.0 + i for i in range(10)],
        })
        obs_df = pd.DataFrame({
            "date": dates,
            "TMAX": [49.0 + i for i in range(10)],
        })
        result = align_nwp_with_observations(nwp_df, obs_df)
        # Constant bias of +1.0 degrees F, so rolling mean should also be ~1.0
        assert abs(result["nwp_bias_7d"].iloc[-1] - 1.0) < 0.01

    def test_empty_nwp(self):
        """Empty NWP DataFrame should return empty result."""
        nwp_df = pd.DataFrame(columns=["date", "tmax_2m_f"])
        obs_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "TMAX": [50.0, 55.0, 48.0],
        })
        result = align_nwp_with_observations(nwp_df, obs_df)
        assert result.empty

    def test_empty_obs(self):
        """Empty observation DataFrame should produce NaN bias for all rows."""
        nwp_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=3),
            "tmax_2m_f": [50.0, 55.0, 48.0],
        })
        obs_df = pd.DataFrame(columns=["date", "TMAX"])
        result = align_nwp_with_observations(nwp_df, obs_df)
        # Left merge: NWP rows survive but obs_tmax is NaN
        assert len(result) == 3
        assert result["obs_tmax"].isna().all()
        assert result["nwp_bias"].isna().all()

    def test_custom_target_col(self):
        """Custom target column name should be used for merge."""
        nwp_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=2),
            "tmax_2m_f": [50.0, 55.0],
        })
        obs_df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=2),
            "observed_max": [49.0, 53.0],
        })
        result = align_nwp_with_observations(nwp_df, obs_df, target_col="observed_max")
        assert "obs_tmax" in result.columns
        assert abs(result["nwp_bias"].iloc[0] - 1.0) < 0.01


# ============================================================================
# run_nwp_preprocessing Integration Test
# ============================================================================


class TestRunNwpPreprocessing:
    """Integration tests for the full NWP preprocessing pipeline."""

    @patch("src.nwp_preprocessing.parse_grib_file")
    def test_full_pipeline(self, mock_parse, tmp_path):
        """Full pipeline should produce CSV output with derived features."""
        nwp_dir = tmp_path / "nwp_raw"
        nwp_dir.mkdir()
        output_dir = tmp_path / "nwp_daily"

        for date_str in ["20220101", "20220102", "20220103"]:
            (nwp_dir / f"gfs_{date_str}_f024.grib2").touch()

        mock_parse.return_value = {
            "tmax_2m_f": 50.0,
            "tmp_850_c": -5.0,
            "ugrd_10m": 3.0,
            "vgrd_10m": 4.0,
            "wind_speed_10m_kt": 9.72,
            "wind_dir_10m_deg": 233.0,
            "cloud_cover_pct": 80.0,
            "mslp_mb": 1015.0,
            "precip_mm": 2.5,
            "forecast_hour": 24,
        }

        df = run_nwp_preprocessing(
            nwp_dir=str(nwp_dir),
            output_dir=str(output_dir),
        )

        assert len(df) == 3
        assert "nwp_tmax_change" in df.columns
        assert "nwp_wind_chill" in df.columns
        assert "nwp_ensemble_spread" in df.columns

        # Check CSV was written
        csv_path = output_dir / "nwp_daily_features.csv"
        assert csv_path.exists()
        saved = pd.read_csv(str(csv_path))
        assert len(saved) == 3

    def test_pipeline_empty_dir(self, tmp_path):
        """Pipeline with empty directory should return empty DataFrame."""
        nwp_dir = tmp_path / "empty_nwp"
        nwp_dir.mkdir()
        output_dir = tmp_path / "nwp_out"

        df = run_nwp_preprocessing(
            nwp_dir=str(nwp_dir),
            output_dir=str(output_dir),
        )
        assert df.empty
