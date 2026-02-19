# Codebase Simplification Report

> **Date:** 2026-02-19
> **Branch:** `claude/simplify-codebase-ACa7q`
> **Status:** Simplifications implemented and cleanup completed
> **Principles:** DRY, KISS, YAGNI

---

## Executive Summary

A deep analysis of the entire codebase was conducted using 5 parallel analyst subagents covering: scripts consolidation, config file duplication, src/ module duplication, tests/docs structure, and trading/eval modules. The analysis identified **~48,769 lines of dead/duplicate code** that can be removed or consolidated. Implementation began but was halted mid-way. This report documents all findings and proposed changes for future reference.

---

## 1. Analysis Findings

### 1.1 ARCHIVE/ — Dead Code (76 files, ~48,000 lines)

The `ARCHIVE/` directory contains 10 subdirectories of orphaned legacy files with **zero imports** from any active code:

| Subdirectory | Files | Description |
|---|---|---|
| `legacy_docs/` | 2 | Old research docs |
| `legacy_docs_v2/` | 9 | Superseded documentation |
| `legacy_experiments/` | 7 | Old experimental scripts |
| `legacy_reports/` | 8 | Old benchmark/audit reports |
| `legacy_root_runners/` | 7 | Old root-level runner scripts |
| `legacy_runners/` | 1 | Old backtest runner |
| `legacy_scripts/` | 6 | Old benchmark scripts |
| `legacy_scripts_v2/` | 22 | Large batch of superseded scripts |
| `legacy_tests/` | 6 | Old test files |
| `legacy_root_docs/` | 8 | Audit docs (KEEP — referenced in MEMORY.md) |

**Recommendation:** Delete all subdirectories except `legacy_root_docs/` (which contains audit reference documents cited in MEMORY.md).

### 1.2 Per-City Pipeline Scripts — 72% Duplication (24 scripts → 6)

Four cities (CHI, PHL, ATL, AUS) each have 6 nearly-identical pipeline scripts:

| Script Pattern | Per-City Lines | Duplication |
|---|---|---|
| `run_<city>_data_collection.py` | ~105 | 95% identical |
| `run_<city>_preprocessing.py` | ~500 | 85% identical |
| `run_<city>_benchmark.py` | ~1,184 | 80% identical |
| `run_<city>_synthesis_calibration.py` | ~1,426 | 80% identical |
| `run_<city>_backtest.py` | ~1,003 | 80% identical |
| `run_<city>_promotion_evaluation.py` | ~605 | 85% identical |

**Total:** 24 scripts × ~820 avg lines = ~19,680 lines → consolidatable to ~3,930 lines (6 unified scripts with `--city` argument)

**Proposed consolidated scripts:**
```
scripts/run_data_collection.py       --city {chi,phl,atl,aus}
scripts/run_preprocessing.py         --city {chi,phl,atl,aus}
scripts/run_benchmark.py             --city {chi,phl,atl,aus}
scripts/run_synthesis_calibration.py --city {chi,phl,atl,aus}
scripts/run_backtest.py              --city {chi,phl,atl,aus}
scripts/run_promotion_evaluation.py  --city {chi,phl,atl,aus}
```

**Pattern:** Each consolidated script uses `argparse` for `--city`, loads the appropriate config via `importlib.import_module(f"config_{city_name}")` and `get_city_config(city_code)`, then runs the shared pipeline logic.

### 1.3 src/ Module Duplication — ~1,640 Redundant Lines

| Duplicated Pattern | Files Containing It | Lines Each |
|---|---|---|
| `SEASON_MAP` dict | calibration.py, evaluate.py, trading.py, kalshi_backtester.py, market_proxy.py, contract_brier.py | ~15 |
| `SEASON_ORDER` list | Same 6 files | ~1 |
| `_to_numpy()` function | calibration.py, evaluate.py, trading.py, kalshi_backtester.py, kalshi_client.py | ~10-15 |
| `_get_season()` helper | kalshi_backtester.py, contract_brier.py | ~5 |
| Brier scoring logic | contract_brier.py, evaluate.py, kalshi_backtester.py | ~50 |

**Proposed shared modules:**

**`src/seasons.py`** (new):
```python
SEASON_MAP = {
    12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
    3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
    6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
    9: "Fall (SON)", 10: "Fall (SON)", 11: "Fall (SON)",
}
SEASON_ORDER = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]
SEASON_MAP_SHORT = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}
def get_season(month: int) -> str: ...
def get_season_months(season_name: str) -> list: ...
```

**`src/utils.py`** (new):
```python
def to_numpy(arr) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array."""
    return np.asarray(arr, dtype=np.float64).ravel()
```

**Files to update:**
- `src/calibration.py` — replace inline SEASON_MAP/SEASON_ORDER/_to_numpy with imports
- `src/evaluate.py` — same
- `src/trading.py` — same
- `src/kalshi_backtester.py` — same (also _get_season)
- `src/kalshi_client.py` — replace _to_numpy with import
- `src/market_proxy.py` — use SEASON_MAP_SHORT from shared module

### 1.4 Per-City Test Files — 80% Duplication (3 files → 1)

| Test File | Lines | Tests |
|---|---|---|
| `tests/test_chi_pipeline.py` | ~340 | 27 tests |
| `tests/test_phl_pipeline.py` | ~339 | 27 tests |
| `tests/test_atl_pipeline.py` | ~310 | ~25 tests |

These test files are structurally identical — they test the same city_config patterns, bucket index logic, script existence, and config module imports. Only the city-specific values differ.

**Proposed:** Single `tests/test_city_pipeline.py` using `@pytest.mark.parametrize` over `["chi", "phl", "atl"]` with city-specific expected values in a data dict.

### 1.5 Config File Duplication — Dual Config Systems

Two competing config systems exist:

1. **`src/city_config.py`** — Centralized `CityConfig` dataclass registry with `get_city_config("chi")` etc.
2. **Per-city modules** — `config_chicago.py`, `config_philadelphia.py`, `config_atlanta.py`, `config_austin.py`

The per-city modules contain station metadata (SURROUNDING_STATIONS, STATION_RINGS, STATION_SECTORS, etc.) that is NOT in `city_config.py`. They also duplicate bucket definitions that ARE in `city_config.py`.

**Active imports of per-city configs:**
- `config_chicago` imported by: all `run_chi_*.py` scripts, `src/operational_data.py`, `src/wga_data_pipeline.py`
- `config_philadelphia` imported by: all `run_phl_*.py` scripts, `src/operational_data.py`, `src/wga_data_pipeline.py`
- `config_atlanta` imported by: all `run_atl_*.py` scripts
- `config_austin` imported by: all `run_aus_*.py` scripts
- `config` (NYC) imported by: 16 src/ modules
- `config_expanded` imported by: `src/station_registry.py`

**Recommendation:** Cannot simply delete per-city config files — `src/operational_data.py` and `src/wga_data_pipeline.py` import them directly. Either:
1. Migrate station metadata into `city_config.py` (preferred long-term), or
2. Keep per-city configs but remove duplicated bucket definitions

### 1.6 MOS Download Scripts — 3 Nearly-Identical Scripts

| Script | Lines |
|---|---|
| `scripts/download_mos_data.py` | ~150 |
| `scripts/download_mos_data_chi.py` | ~150 |
| `scripts/download_mos_data_phl.py` | ~150 |

**Proposed:** Single `scripts/download_mos_data.py` with `--city` argument.

### 1.7 Experimental Scripts — Should Be Organized

Several scripts in `scripts/` are experimental or one-off analysis scripts that clutter the main scripts directory:
- `scripts/analyze_*.py`
- `scripts/debug_*.py`
- Various exploration scripts

**Proposed:** Move to `scripts/experiments/` subdirectory.

---

## 2. Changes Completed

The following simplifications are now implemented in the active codebase:

### 2.1 Shared Utility Modules Created
- Added `src/seasons.py` with shared season constants and helpers:
  - `SEASON_MAP`, `SEASON_ORDER`, `SEASON_MAP_SHORT`
  - `get_season()`, `get_season_months()`
- Added `src/utils.py` with shared `to_numpy()` helper.

### 2.2 src/ Modules Updated (6 of 6 complete)
- `src/calibration.py` — season mapping and numpy conversion now rely on shared modules.
- `src/evaluate.py` — season mapping and numpy conversion now rely on shared modules.
- `src/trading.py` — season mapping and numpy conversion now rely on shared modules.
- `src/kalshi_backtester.py` — season mapping and numpy conversion now rely on shared modules.
- `src/kalshi_client.py` — numpy conversion now relies on shared module.
- `src/market_proxy.py` — short season mapping now relies on shared module.

### 2.3 MOS Script Consolidation Completed
- Added unified script: `scripts/download_iem_mos_data.py` with station parameterization (`--station`).
- Converted existing city-specific MOS scripts into thin backward-compatible wrappers:
  - `scripts/download_iem_mos.py`
  - `scripts/download_iem_mos_kord.py`
  - `scripts/download_iem_mos_kphl.py`

---

## 3. Final Cleanup Completed

- **ARCHIVE cleanup:** Deleted all legacy archive subdirectories except `legacy_root_docs/`. Added `ARCHIVE/DELETED_LEGACY_FILES_SUMMARY.md` with a concise summary for each removed file.
- **Per-city pipeline consolidation:** Unified city-template pipeline scripts are active (`run_data_collection.py`, `run_preprocessing.py`, `run_benchmark.py`, `run_synthesis_calibration.py`, `run_backtest.py`, `run_promotion_evaluation.py`) with per-city wrappers retained for compatibility.
- **Test consolidation:** Removed `tests/test_chi_pipeline.py`, `tests/test_phl_pipeline.py`, and `tests/test_atl_pipeline.py` after consolidating coverage into `tests/test_city_pipeline.py`.
- **Documentation updates:** Updated active architecture docs to reflect consolidated tests and ARCHIVE state.

---

## 4. Impact Summary

| Category | Current | Proposed | Lines Saved |
|---|---|---|---|
| ARCHIVE/ dead code | ~48,000 lines | 0 (delete) | ~48,000 |
| Per-city scripts (24) | ~19,680 lines | ~3,930 lines (6 scripts) | ~15,750 |
| src/ duplication | ~1,640 lines | ~80 lines (2 shared modules) | ~1,560 |
| Per-city tests (3) | ~990 lines | ~250 lines (1 parameterized) | ~740 |
| MOS scripts (3) | ~450 lines | ~180 lines (1 script) | ~270 |
| **Total** | | | **~66,320 lines** |

---

## 5. Implementation Order (Recommended)

1. Delete ARCHIVE subdirectories (safe, no imports)
2. Create `src/seasons.py` and `src/utils.py`
3. Update all 6 src/ modules to use shared imports
4. Consolidate per-city scripts (data_collection first as simplest)
5. Consolidate test files
6. Consolidate MOS scripts
7. Clean up config duplication
8. Organize experimental scripts
9. Update documentation
10. Run full test suite to validate
11. Commit and push

---

## 6. Risk Notes

- **Config imports:** `src/operational_data.py` and `src/wga_data_pipeline.py` import per-city configs directly. These must be migrated before config files can be deleted.
- **NYC scripts:** NYC pipeline scripts (`run_e0_e8_best_model_benchmark.py`, `run_wga_v2_benchmark.py`, `run_unified_outperformance_benchmark.py`) are structurally different from the city-template scripts and should NOT be consolidated.
- **`config.py` and `config_expanded.py`:** These NYC-specific configs are imported by 17 src/ modules and cannot be easily merged into city_config.py without significant refactoring.
- **Test expectations:** The per-city test files contain city-specific expected values (bucket counts, station IDs, ring names, etc.) that must be preserved in the parameterized version.
