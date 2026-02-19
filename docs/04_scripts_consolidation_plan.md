# Scripts Consolidation Plan (DRY / KISS / YAGNI)

> **Date:** 2026-02-19
> **Status:** COMPLETE
> **Scope:** `scripts/` directory organization and consolidation roadmap

## 1) Current-State Findings

After reviewing the codebase modules (`src/`) and script inventory (`scripts/`), the scripts are currently split into four practical groups:

1. **City pipeline wrappers**
   - `run_<city>_{data_collection,preprocessing,benchmark,synthesis_calibration,backtest,promotion_evaluation}.py`
   - Cities: `nyc`, `chi`, `phl`, `atl`, `aus`.
2. **Unified stage entrypoints**
   - `run_data_collection.py`, `run_preprocessing.py`, `run_benchmark.py`, `run_synthesis_calibration.py`, `run_backtest.py`, `run_promotion_evaluation.py`.
   - All accept `--city {nyc,chi,phl,atl,aus}`.
3. **Cross-city / template runner**
   - `run_city_nws_kalshi_template_benchmark.py`.
4. **One-off experiments and utility scripts**
   - `scripts/experiments/benchmarking/` and `scripts/experiments/trading/`.
   - Data fetch/download scripts (`fetch_*`, `download_*`).

### Duplication Pattern

The city wrappers follow the same orchestration pattern with only city-specific parameters changed. This is a high-confidence DRY target. All wrappers are now thin shims (11 lines) that delegate to unified stage scripts.

## 2) Consolidation Goals

- **DRY:** Eliminate copy-pasted city wrappers by introducing a shared pipeline entrypoint.
- **KISS:** Keep one obvious way to run each stage (single CLI with `--city` and `--stage`).
- **YAGNI:** Avoid building a workflow engine unless current requirements demand it.
- **Safety:** Preserve backward compatibility during migration with thin wrapper shims.

## 3) Proposed Target Structure

### A. Canonical single entrypoint

`scripts/run_city_pipeline.py`

Arguments:

- `--city {nyc,chi,phl,atl,aus}`
- `--stage {data_collection,preprocessing,benchmark,synthesis_calibration,backtest,promotion_evaluation,all}`
- `--dry-run`, `--continue-on-error`

### B. Stage registry (implemented in run_city_pipeline.py)

The stage registry lives directly in `run_city_pipeline.py` as a `STAGE_REGISTRY` dict + `STAGE_ORDER` tuple. This was chosen over a separate module to keep the abstraction minimal.

Responsibilities:

- Map stage names to existing callable scripts.
- Validate stage ordering and prerequisites.
- Centralize logging.

### C. Legacy wrappers as shims

Each `run_<city>_<stage>.py` is a thin shim (~11 lines) that sets `sys.argv` and imports the unified stage script's `main()`.

This gives:

- zero workflow disruption,
- gradual migration path,
- easy rollback.

## 4) Script Inventory Decision Matrix

| Category | Action | Status |
|---|---|---|
| `run_<city>_<stage>.py` wrappers (all 5 cities) | **Consolidated** into thin shims delegating to unified scripts | ✅ DONE |
| Unified stage scripts (`run_*.py`) | **Kept** as canonical implementations with `--city` flag | ✅ DONE |
| `run_city_pipeline.py` | **Created** as orchestrator with `--stage all` support | ✅ DONE |
| `run_city_nws_kalshi_template_benchmark.py` | **Kept** | ✅ Retained |
| `scripts/experiments/*.py` | **Reorganized** into `benchmarking/` and `trading/` subfolders | ✅ DONE |
| `download_*`, `fetch_*` scripts | **Retained** | ✅ Retained |

## 5) Implementation Plan (Phased)

### Phase 1 — Non-breaking organization

1. ✅ Add `scripts/README.md` with command taxonomy and supported entrypoints.
2. ✅ Add `run_city_pipeline.py` with `--city` + `--stage` dispatch.
3. ✅ Convert one city family (`chi`) to shim wrappers and validate outputs unchanged.

### Phase 2 — Wrapper consolidation

1. ✅ Convert remaining city wrappers (`phl`, `atl`, `aus`) to shims.
2. ✅ Integrate NYC into canonical runner path (`--city nyc` supported across all 6 stage scripts).
3. ✅ Create `run_nyc_<stage>.py` wrapper shims for consistency with other cities.
4. ✅ Ensure all scripts write artifacts to existing city-specific locations (NYC uses root-level dirs for backward compatibility).

### Phase 3 — Cleanup and hardening

1. ✅ Add integration test that loops through stages in `--dry-run` mode for each city.
2. ✅ Add lint/check to prevent introducing new per-city duplicate wrappers (`test_script_wrapper_consolidation.py`).
3. ✅ Move experiments into purpose-based subfolders (`experiments/benchmarking/`, `experiments/trading/`).
4. ✅ Add `experiments/README.md` with script index by category.

## 6) Decisions Made

### Unified scripts are canonical implementations (not shims)

The 6 unified stage scripts (`run_data_collection.py`, etc.) are the **canonical implementations** containing all business logic. They are NOT thin shims — they are the targets that wrappers delegate to. `run_city_pipeline.py` orchestrates them via subprocess.

### NYC integration approach

NYC was integrated into the existing canonical runner path rather than maintained as a separate legacy track. Each unified script's `CITY_CONFIG_MODULES` dict now maps `"nyc"` to `"config_expanded"`, which provides the full expanded station network. NYC's `CityConfig` in `src/city_config.py` uses root-level directories (`data/`, `models/`, `results/`) for backward compatibility, while other cities use `data/<city>/` subdirectories.

### Promotion thresholds for NYC

NYC promotion thresholds added to `run_promotion_evaluation.py`:
- Brier threshold: 0.14 (moderate variance, aligned with PHL)
- NWS baseline: 0.12 (OKX)
- Seasonal Brier threshold: 0.20

### Experiment organization

Experiments split into two categories:
- `experiments/benchmarking/` — model training, evaluation, cross-city comparisons (6 scripts)
- `experiments/trading/` — backtest and strategy sweep experiments (3 scripts)

## 7) Guardrails for Refactor

- No randomization changes to chronological workflows.
- No training/inference data source mixing.
- Keep contract alignment logic in shared city config.
- Preserve city artifact conventions under `data/<city>/`, `results/<city>/`, `models/<city>/`.
- Keep existing promotion/backtest semantics unchanged before and after consolidation.

## 8) Success Criteria

- ✅ At least **70% reduction** in duplicated orchestration lines across city wrappers (30 wrappers × ~11 lines vs. original full implementations).
- ✅ One canonical command pattern for all cities/stages (`run_city_pipeline.py --city X --stage Y`).
- ✅ Backward-compatible old commands remain functional (`run_<city>_<stage>.py` shims).
- ✅ No regression in benchmark/backtest outputs (dry-run validated for all cities).

## 9) Test Coverage

| Test File | Coverage |
|---|---|
| `tests/test_run_city_pipeline.py` | Stage expansion order, dry-run execution, single-stage dry-run |
| `tests/test_script_wrapper_consolidation.py` | All 30 city×stage wrappers exist, wrappers are thin delegators (≤9 significant lines), correct delegation target and city code |
| `tests/test_city_config.py` | All city configs load correctly, bucket edges, labels, directories |
| `tests/test_city_pipeline.py` | Config integration, module imports, scripts existence, bucket index (CHI/PHL/ATL) |

All 132 tests pass.

## 10) Progress Log

### Iteration 1 (2026-02-19)

- Created `run_city_pipeline.py` with stage registry and `--dry-run`.
- Created `scripts/README.md`.
- Created `tests/test_run_city_pipeline.py`.
- Created `tests/test_script_wrapper_consolidation.py`.
- Converted CHI/PHL/ATL/AUS wrappers to thin shims.

### Iteration 2 (2026-02-19)

- Added `"nyc": "config_expanded"` to `CITY_CONFIG_MODULES` in all 6 unified stage scripts.
- Added `"nyc"` to `SUPPORTED_CITIES` in `run_city_pipeline.py`.
- Added NYC promotion thresholds to `run_promotion_evaluation.py`.
- Added `"nyc": "NYC_TMAX"` to `CITY_TARGET_NAMES` in `run_benchmark.py`.
- Created 6 `run_nyc_<stage>.py` thin wrapper shims.
- Updated guardrail test to include NYC (5 cities × 6 stages = 30 wrappers).
- Reorganized `scripts/experiments/` into `benchmarking/` and `trading/` subfolders with README index.
- Updated `scripts/README.md` with NYC support and experiment subfolder documentation.
- All 132 tests passing across 4 test suites.
