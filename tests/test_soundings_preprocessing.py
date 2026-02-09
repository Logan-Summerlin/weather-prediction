"""
Tests for IGRA soundings preprocessing module.

Covers:
  - parse_sounding_csv() — normal, missing columns, empty file, missing file
  - extract_level_features() — exact match, interpolation, level not found
  - compute_stability_features() — normal, missing surface, missing 850mb
  - process_single_sounding() — integration of parsing + extraction
  - aggregate_soundings_daily() — multiple files, 00Z vs 12Z, missing files
  - celsius_to_fahrenheit() — conversion correctness
  - Edge cases: NaN handling, single-level sounding, corrupt data
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.soundings_preprocessing import (
    celsius_to_fahrenheit,
    parse_sounding_csv,
    extract_level_features,
    compute_stability_features,
    process_single_sounding,
    aggregate_soundings_daily,
    run_soundings_preprocessing,
    _interpolate_to_level,
    _get_surface_temperature,
    _parse_filename_metadata,
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

@pytest.fixture
def sample_sounding_df():
    """A realistic sounding DataFrame with standard pressure levels."""
    return pd.DataFrame({
        "pressure": [1013.0, 1000.0, 925.0, 850.0, 700.0, 500.0, 300.0],
        "height": [10, 110, 760, 1500, 3100, 5600, 9200],
        "temperature": [22.0, 20.0, 15.0, 10.0, 2.0, -20.0, -45.0],
        "dewpoint": [18.0, 16.0, 10.0, 5.0, -5.0, -35.0, -55.0],
        "direction": [180, 190, 220, 250, 270, 280, 290],
        "speed": [5, 8, 12, 18, 25, 35, 50],
        "u_wind": [-5, -7, -10, -15, -25, -34, -48],
        "v_wind": [0, -3, -6, -8, 0, 6, 12],
        "station": ["USM00072501"] * 7,
        "station_number": [72501] * 7,
        "latitude": [40.87] * 7,
        "longitude": [-72.87] * 7,
        "elevation": [20] * 7,
        "time": ["2022-01-15 00:00:00"] * 7,
    })


@pytest.fixture
def sample_sounding_csv(tmp_path, sample_sounding_df):
    """Write a sample sounding CSV and return its path."""
    path = tmp_path / "USM00072501_2022011500.csv"
    sample_sounding_df.to_csv(path, index=False)
    return str(path)


def _write_sounding_csv(
    path: str,
    pressures: list,
    temperatures: list,
    heights: list | None = None,
    dewpoints: list | None = None,
    directions: list | None = None,
    speeds: list | None = None,
) -> str:
    """Helper to write a minimal sounding CSV."""
    data = {"pressure": pressures, "temperature": temperatures}
    if heights is not None:
        data["height"] = heights
    if dewpoints is not None:
        data["dewpoint"] = dewpoints
    if directions is not None:
        data["direction"] = directions
    if speeds is not None:
        data["speed"] = speeds
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ===========================================================================
# celsius_to_fahrenheit tests
# ===========================================================================

class TestCelsiusToFahrenheit:

    def test_freezing_point(self):
        assert celsius_to_fahrenheit(0.0) == 32.0

    def test_boiling_point(self):
        assert celsius_to_fahrenheit(100.0) == 212.0

    def test_negative(self):
        assert abs(celsius_to_fahrenheit(-40.0) - (-40.0)) < 0.01

    def test_nan(self):
        assert np.isnan(celsius_to_fahrenheit(np.nan))


# ===========================================================================
# parse_sounding_csv tests
# ===========================================================================

class TestParseSoundingCsv:

    def test_normal_load(self, sample_sounding_csv):
        """Loads a well-formed sounding CSV correctly."""
        df = parse_sounding_csv(sample_sounding_csv)
        assert len(df) == 7
        assert "pressure" in df.columns
        assert "temperature" in df.columns
        assert df["pressure"].dtype == np.float64

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_sounding_csv("/nonexistent/path.csv")

    def test_empty_file(self, tmp_path):
        """Raises ValueError for an empty CSV."""
        path = tmp_path / "empty.csv"
        path.write_text("pressure,temperature\n")
        with pytest.raises(ValueError, match="empty"):
            parse_sounding_csv(str(path))

    def test_missing_required_columns(self, tmp_path):
        """Raises ValueError when required columns are absent."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"height": [1000], "speed": [10]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            parse_sounding_csv(str(path))

    def test_numeric_coercion(self, tmp_path):
        """Non-numeric values in numeric columns become NaN."""
        path = tmp_path / "coerce.csv"
        pd.DataFrame({
            "pressure": [1000.0, "bad"],
            "temperature": [20.0, "invalid"],
            "height": [100, "N/A"],
        }).to_csv(path, index=False)
        df = parse_sounding_csv(str(path))
        assert df["pressure"].iloc[0] == 1000.0
        assert np.isnan(df["pressure"].iloc[1])
        assert np.isnan(df["temperature"].iloc[1])

    def test_time_parsing(self, sample_sounding_csv):
        """Time column is parsed to datetime."""
        df = parse_sounding_csv(sample_sounding_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["time"])


# ===========================================================================
# _interpolate_to_level tests
# ===========================================================================

class TestInterpolateToLevel:

    def test_exact_match(self, sample_sounding_df):
        """Returns exact value when target pressure exists."""
        result = _interpolate_to_level(sample_sounding_df, 850.0, "temperature")
        assert abs(result - 10.0) < 0.01

    def test_interpolation_between_levels(self, sample_sounding_df):
        """Interpolation between two known pressure levels gives reasonable value."""
        # 775 mb is between 850 (10C) and 700 (2C)
        result = _interpolate_to_level(sample_sounding_df, 775.0, "temperature")
        assert 2.0 < result < 10.0

    def test_target_below_range(self, sample_sounding_df):
        """Returns NaN when target pressure is above the sounding range."""
        result = _interpolate_to_level(sample_sounding_df, 1100.0, "temperature")
        assert np.isnan(result)

    def test_target_above_range(self, sample_sounding_df):
        """Returns NaN when target pressure is below the sounding range."""
        result = _interpolate_to_level(sample_sounding_df, 200.0, "temperature")
        assert np.isnan(result)

    def test_all_nan_column(self):
        """Returns NaN when the column has no valid data."""
        df = pd.DataFrame({
            "pressure": [1000.0, 500.0],
            "temperature": [np.nan, np.nan],
        })
        result = _interpolate_to_level(df, 750.0, "temperature")
        assert np.isnan(result)

    def test_empty_dataframe(self):
        """Returns NaN for empty input."""
        df = pd.DataFrame({"pressure": [], "temperature": []})
        result = _interpolate_to_level(df, 850.0, "temperature")
        assert np.isnan(result)


# ===========================================================================
# extract_level_features tests
# ===========================================================================

class TestExtractLevelFeatures:

    def test_exact_level_match(self, sample_sounding_df):
        """Extracts features when exact pressure level exists."""
        features = extract_level_features(sample_sounding_df, 850.0)
        assert abs(features["t850_c"] - 10.0) < 0.01
        assert abs(features["t850_f"] - 50.0) < 0.01
        assert abs(features["dewpoint_850_c"] - 5.0) < 0.01
        assert features["wind_dir_850"] == 250
        assert features["wind_speed_850"] == 18
        assert features["height_850_m"] == 1500

    def test_500mb_level(self, sample_sounding_df):
        """Correct extraction at 500mb level."""
        features = extract_level_features(sample_sounding_df, 500.0)
        assert abs(features["t500_c"] - (-20.0)) < 0.01
        assert abs(features["t500_f"] - (-4.0)) < 0.01
        assert features["height_500_m"] == 5600

    def test_interpolation_needed(self, sample_sounding_df):
        """Features are interpolated when exact level is not available."""
        features = extract_level_features(sample_sounding_df, 775.0)
        # Temperature should be between 850mb (10C) and 700mb (2C)
        assert 2.0 < features["t775_c"] < 10.0
        assert not np.isnan(features["t775_f"])

    def test_level_not_found(self):
        """Returns NaN features when level is outside sounding range."""
        df = pd.DataFrame({
            "pressure": [1000.0, 900.0],
            "temperature": [20.0, 15.0],
            "height": [100, 700],
        })
        features = extract_level_features(df, 500.0)
        assert np.isnan(features["t500_c"])
        assert np.isnan(features["t500_f"])

    def test_missing_optional_columns(self):
        """Gracefully handles missing optional columns (dewpoint, wind)."""
        df = pd.DataFrame({
            "pressure": [850.0],
            "temperature": [10.0],
        })
        features = extract_level_features(df, 850.0)
        assert abs(features["t850_c"] - 10.0) < 0.01
        assert np.isnan(features["dewpoint_850_c"])
        assert np.isnan(features["wind_dir_850"])
        assert np.isnan(features["wind_speed_850"])
        assert np.isnan(features["height_850_m"])

    def test_nan_temperature_at_exact_level(self):
        """NaN propagated when temperature is missing at exact level."""
        df = pd.DataFrame({
            "pressure": [850.0],
            "temperature": [np.nan],
        })
        features = extract_level_features(df, 850.0)
        assert np.isnan(features["t850_c"])
        assert np.isnan(features["t850_f"])


# ===========================================================================
# _get_surface_temperature tests
# ===========================================================================

class TestGetSurfaceTemperature:

    def test_normal_surface(self, sample_sounding_df):
        """Returns temperature at highest pressure above 900mb."""
        result = _get_surface_temperature(sample_sounding_df)
        # 1013 mb has temp 22.0C
        assert abs(result - 22.0) < 0.01

    def test_no_surface_level(self):
        """Returns NaN when no observation is above 900mb."""
        df = pd.DataFrame({
            "pressure": [850.0, 500.0],
            "temperature": [10.0, -20.0],
        })
        result = _get_surface_temperature(df)
        assert np.isnan(result)

    def test_all_nan_temperatures(self):
        """Returns NaN when all temperatures are NaN."""
        df = pd.DataFrame({
            "pressure": [1013.0, 850.0],
            "temperature": [np.nan, np.nan],
        })
        result = _get_surface_temperature(df)
        assert np.isnan(result)


# ===========================================================================
# compute_stability_features tests
# ===========================================================================

class TestComputeStabilityFeatures:

    def test_normal_stability(self, sample_sounding_df):
        """Stability index and lapse rate computed correctly."""
        features = compute_stability_features(sample_sounding_df)
        # T_surface = 22C, T850 = 10C -> stability = 10 - 22 = -12
        assert abs(features["t_surface_c"] - 22.0) < 0.01
        assert abs(features["stability_index"] - (-12.0)) < 0.01
        # Lapse rate: (T850 - T500) / (h500 - h850) * 1000
        # = (10 - (-20)) / (5600 - 1500) * 1000 = 30/4.1 = ~7.32 C/km
        expected_lapse = (10.0 - (-20.0)) / ((5600 - 1500) / 1000.0)
        assert abs(features["lapse_rate_850_500"] - expected_lapse) < 0.1

    def test_missing_surface(self):
        """NaN stability index when no surface observation."""
        df = pd.DataFrame({
            "pressure": [850.0, 500.0],
            "temperature": [10.0, -20.0],
            "height": [1500, 5600],
        })
        features = compute_stability_features(df)
        assert np.isnan(features["t_surface_c"])
        assert np.isnan(features["stability_index"])

    def test_missing_850mb(self):
        """NaN stability and lapse rate when 850mb is missing."""
        df = pd.DataFrame({
            "pressure": [1013.0, 500.0],
            "temperature": [22.0, -20.0],
            "height": [10, 5600],
        })
        features = compute_stability_features(df)
        # 850 can be interpolated between 1013 and 500, so it should work
        assert not np.isnan(features["stability_index"])

    def test_missing_500mb_height(self):
        """NaN lapse rate when height is missing for 500mb."""
        df = pd.DataFrame({
            "pressure": [1013.0, 850.0, 500.0],
            "temperature": [22.0, 10.0, -20.0],
            # No height column
        })
        features = compute_stability_features(df)
        assert np.isnan(features["lapse_rate_850_500"])
        # Stability index should still work
        assert abs(features["stability_index"] - (-12.0)) < 0.01

    def test_surface_fahrenheit(self, sample_sounding_df):
        """Surface temperature is also provided in Fahrenheit."""
        features = compute_stability_features(sample_sounding_df)
        expected_f = 22.0 * 9 / 5 + 32
        assert abs(features["t_surface_f"] - expected_f) < 0.01


# ===========================================================================
# process_single_sounding tests
# ===========================================================================

class TestProcessSingleSounding:

    def test_integration(self, sample_sounding_csv):
        """Processes a sounding file and returns all expected features."""
        features = process_single_sounding(
            sample_sounding_csv, levels_mb=[850.0, 500.0]
        )
        # Level features
        assert "t850_c" in features
        assert "t850_f" in features
        assert "t500_c" in features
        assert "t500_f" in features
        assert "height_850_m" in features
        assert "height_500_m" in features
        # Stability features
        assert "t_surface_c" in features
        assert "stability_index" in features
        assert "lapse_rate_850_500" in features
        # Metadata
        assert "file_path" in features

    def test_single_level(self, sample_sounding_csv):
        """Works with a single requested level."""
        features = process_single_sounding(
            sample_sounding_csv, levels_mb=[850.0]
        )
        assert "t850_c" in features
        assert "t500_c" not in features

    def test_missing_file(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            process_single_sounding(
                str(tmp_path / "nonexistent.csv"), levels_mb=[850.0]
            )


# ===========================================================================
# _parse_filename_metadata tests
# ===========================================================================

class TestParseFilenameMetadata:

    def test_valid_00z(self):
        meta = _parse_filename_metadata("USM00072501_2022011500.csv")
        assert meta["station_id"] == "USM00072501"
        assert meta["hour"] == 0
        assert meta["date"] == pd.Timestamp("2022-01-15").date()

    def test_valid_12z(self):
        meta = _parse_filename_metadata("USM00072501_2022071512.csv")
        assert meta["hour"] == 12

    def test_invalid_format(self):
        meta = _parse_filename_metadata("random_file.csv")
        assert meta == {}

    def test_header_json(self):
        """Header JSON files don't match the pattern."""
        meta = _parse_filename_metadata("USM00072501_2022011500.header.json")
        assert meta == {}


# ===========================================================================
# aggregate_soundings_daily tests
# ===========================================================================

class TestAggregateSoundingsDaily:

    def _create_sounding_dir(self, tmp_path, station_id, dates_hours, make_bad=False):
        """Create a directory with sample sounding CSV files."""
        sounding_dir = tmp_path / "soundings"
        sounding_dir.mkdir(exist_ok=True)

        for date_str, hour in dates_hours:
            filename = f"{station_id}_{date_str}{hour:02d}.csv"
            filepath = sounding_dir / filename
            if make_bad:
                filepath.write_text("bad,data\n")
            else:
                _write_sounding_csv(
                    str(filepath),
                    pressures=[1013.0, 850.0, 500.0],
                    temperatures=[22.0, 10.0, -20.0],
                    heights=[10, 1500, 5600],
                    dewpoints=[18.0, 5.0, -35.0],
                    directions=[180, 250, 280],
                    speeds=[5, 18, 35],
                )
        return str(sounding_dir)

    def test_single_sounding(self, tmp_path):
        """Processes a single sounding file correctly."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501", [("20220115", 0)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00072501", [850.0, 500.0]
        )
        assert len(df) == 1
        assert df["hour"].iloc[0] == 0
        assert "t850_c" in df.columns
        assert "t500_c" in df.columns
        assert "stability_index" in df.columns

    def test_multiple_hours(self, tmp_path):
        """Both 00Z and 12Z soundings for the same day are separate rows."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501",
            [("20220115", 0), ("20220115", 12)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00072501", [850.0, 500.0]
        )
        assert len(df) == 2
        assert set(df["hour"].tolist()) == {0, 12}

    def test_multiple_days(self, tmp_path):
        """Multiple days produce correct number of records."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501",
            [("20220115", 0), ("20220115", 12),
             ("20220116", 0), ("20220116", 12)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00072501", [850.0, 500.0]
        )
        assert len(df) == 4
        assert df["date"].nunique() == 2

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty DataFrame."""
        sounding_dir = tmp_path / "empty_soundings"
        sounding_dir.mkdir()
        df = aggregate_soundings_daily(
            str(sounding_dir), "USM00072501", [850.0, 500.0]
        )
        assert df.empty

    def test_no_matching_station(self, tmp_path):
        """Directory with files for other stations returns empty DataFrame."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501", [("20220115", 0)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00099999", [850.0]
        )
        assert df.empty

    def test_corrupt_file_skipped(self, tmp_path):
        """Corrupt sounding files are skipped with a warning."""
        sounding_dir = tmp_path / "mixed"
        sounding_dir.mkdir()

        # Good file
        good_path = sounding_dir / "USM00072501_2022011500.csv"
        _write_sounding_csv(
            str(good_path),
            pressures=[1013.0, 850.0, 500.0],
            temperatures=[22.0, 10.0, -20.0],
            heights=[10, 1500, 5600],
        )
        # Bad file (missing required columns)
        bad_path = sounding_dir / "USM00072501_2022011512.csv"
        pd.DataFrame({"bogus": [1, 2, 3]}).to_csv(str(bad_path), index=False)

        df = aggregate_soundings_daily(
            str(sounding_dir), "USM00072501", [850.0, 500.0]
        )
        assert len(df) == 1

    def test_sorted_output(self, tmp_path):
        """Output is sorted by date and hour."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501",
            [("20220116", 12), ("20220115", 0), ("20220116", 0)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00072501", [850.0, 500.0]
        )
        dates = df["date"].tolist()
        hours = df["hour"].tolist()
        # Check sorting
        for i in range(1, len(df)):
            assert (dates[i], hours[i]) >= (dates[i - 1], hours[i - 1])

    def test_file_path_not_in_output(self, tmp_path):
        """The internal file_path column is dropped from output."""
        sounding_dir = self._create_sounding_dir(
            tmp_path, "USM00072501", [("20220115", 0)]
        )
        df = aggregate_soundings_daily(
            sounding_dir, "USM00072501", [850.0]
        )
        assert "file_path" not in df.columns


# ===========================================================================
# run_soundings_preprocessing tests
# ===========================================================================

class TestRunSoundingsPreprocessing:

    def test_saves_output_csv(self, tmp_path):
        """Pipeline saves output CSV to the specified directory."""
        sounding_dir = tmp_path / "raw"
        sounding_dir.mkdir()
        output_dir = tmp_path / "processed"

        filepath = sounding_dir / "USM00072501_2022011500.csv"
        _write_sounding_csv(
            str(filepath),
            pressures=[1013.0, 850.0, 500.0],
            temperatures=[22.0, 10.0, -20.0],
            heights=[10, 1500, 5600],
        )

        df = run_soundings_preprocessing(
            sounding_dir=str(sounding_dir),
            output_dir=str(output_dir),
            station_id="USM00072501",
            levels_mb=[850.0, 500.0],
        )

        assert len(df) == 1
        expected_csv = output_dir / "USM00072501_soundings_daily.csv"
        assert expected_csv.exists()

    def test_empty_input_returns_empty(self, tmp_path):
        """Pipeline handles empty input directory gracefully."""
        sounding_dir = tmp_path / "empty"
        sounding_dir.mkdir()
        output_dir = tmp_path / "output"

        df = run_soundings_preprocessing(
            sounding_dir=str(sounding_dir),
            output_dir=str(output_dir),
            station_id="USM00072501",
            levels_mb=[850.0],
        )

        assert df.empty
