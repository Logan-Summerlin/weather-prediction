# Scripts Consolidation Plan (DRY / KISS / YAGNI)

> **Date:** 2026-02-19  
> **Scope:** `scripts/` directory organization and consolidation roadmap

## 1) Current-State Findings

After reviewing the codebase modules (`src/`) and script inventory (`scripts/`), the scripts are currently split into four practical groups:

1. **City pipeline wrappers**
   - `run_<city>_{data_collection,preprocessing,benchmark,synthesis_calibration,backtest,promotion_evaluation}.py`
   - Cities currently present: `chi`, `phl`, `atl`, `aus`.
2. **Legacy NYC wrappers**
   - `run_data_collection.py`, `run_preprocessing.py`, `run_benchmark.py`, `run_synthesis_calibration.py`, `run_backtest.py`, `run_promotion_evaluation.py`.
3. **Cross-city / template runner**
   - `run_city_nws_kalshi_template_benchmark.py`.
4. **One-off experiments and utility scripts**
   - `scripts/experiments/*.py` and data fetch/download scripts.

### Duplication Pattern

The city wrappers follow the same orchestration pattern with only city-specific parameters changed. This is a high-confidence DRY target.

## 2) Consolidation Goals

- **DRY:** Eliminate copy-pasted city wrappers by introducing a shared pipeline entrypoint.
- **KISS:** Keep one obvious way to run each stage (single CLI with `--city` and `--stage`).
- **YAGNI:** Avoid building a workflow engine unless current requirements demand it.
- **Safety:** Preserve backward compatibility during migration with thin wrapper shims.

## 3) Proposed Target Structure

### A. Canonical single entrypoint

Create:

- `scripts/run_city_pipeline.py`

With arguments:

- `--city {nyc,chi,phl,atl,aus,...}`
- `--stage {data_collection,preprocessing,benchmark,synthesis_calibration,backtest,promotion_evaluation,all}`
- optional `--config`, `--start-date`, `--end-date`, `--dry-run` (only if already needed)

### B. Reusable stage registry (minimal abstraction)

Create module:

- `src/pipeline/stages.py` (or `src/pipeline_runner.py` if keeping structure simple)

Responsibilities:

- Map stage names to existing callable functions.
- Validate stage ordering and prerequisites.
- Centralize logging and artifact paths.

### C. Keep legacy wrappers as shims initially

Each existing `run_<city>_<stage>.py` becomes a tiny shim that calls `run_city_pipeline.py` with fixed args.

This gives:

- zero workflow disruption,
- gradual migration path,
- easy rollback.

## 4) Script Inventory Decision Matrix

| Category | Action | Rationale |
|---|---|---|
| `run_<city>_<stage>.py` wrappers | **Consolidate** into single parameterized runner | Highest duplication, low risk |
| NYC generic wrappers (`run_*.py`) | **Alias to canonical runner** | Preserve old commands while converging |
| `run_city_nws_kalshi_template_benchmark.py` | **Keep** (or fold later) | Already parameterized; useful template |
| `scripts/experiments/*.py` | **Retain**, but group by purpose and add README index | Experimental scope differs; premature merge risk |
| `download_*`, `fetch_*` scripts | **Retain**, standardize naming/options | Operational utilities with distinct responsibilities |

## 5) Implementation Plan (Phased)

### Phase 1 — Non-breaking organization

1. ✅ Add `scripts/README.md` with command taxonomy and supported entrypoints.
2. ✅ Add `run_city_pipeline.py` with `--city` + `--stage` dispatch.
3. ✅ Convert one city family (`chi`) to shim wrappers and validate outputs unchanged.

### Phase 2 — Wrapper consolidation

1. ✅ Convert remaining city wrappers (`phl`, `atl`, `aus`) to shims.
2. ⏳ Convert NYC legacy `run_*.py` wrappers to shims.
3. ✅ Ensure all scripts write artifacts to existing city-specific locations.

### Phase 3 — Cleanup and hardening

1. ✅ Add integration test that loops through stages in `--dry-run` mode for each city.
2. ⏳ Add lint/check to prevent introducing new per-city duplicate wrappers.
3. ⏳ Optionally move long-tail experiments into subfolders (`experiments/leakage_audits`, `experiments/ensemble`, etc.).

## 8) Progress Update (Completed in this iteration)

### Completed

- Added canonical runner `scripts/run_city_pipeline.py` with:
  - stage registry,
  - ordered `--stage all` orchestration,
  - `--dry-run` support,
  - optional `--continue-on-error` behavior.
- Added `scripts/README.md` with script taxonomy, canonical command examples,
  and explicit separation between production entrypoints vs experiments/utilities.
- Added test coverage in `tests/test_run_city_pipeline.py` for:
  - stage expansion order,
  - end-to-end `--dry-run` invocation,
  - single-stage dry-run success.

### Outstanding

- Decide whether to keep existing stage scripts (`run_data_collection.py`, etc.) as
  canonical implementations or demote them to thin shims that dispatch through
  `run_city_pipeline.py`.
- Add CI guardrail/lint rule to flag newly added duplicate per-city wrappers that
  include non-trivial logic.
- Evaluate introducing `nyc` into the same canonical runner path or formally
  documenting NYC as a legacy compatibility track.
- Reorganize `scripts/experiments/` into purpose-based subfolders once current
  active experiments are tagged.

## 6) Guardrails for Refactor

- No randomization changes to chronological workflows.
- No training/inference data source mixing.
- Keep contract alignment logic in shared city config.
- Preserve city artifact conventions under `data/<city>/`, `results/<city>/`, `models/<city>/`.
- Keep existing promotion/backtest semantics unchanged before and after consolidation.

## 7) Success Criteria

- At least **70% reduction** in duplicated orchestration lines across city wrappers.
- One canonical command pattern for all cities/stages.
- Backward-compatible old commands remain functional during transition.
- No regression in benchmark/backtest outputs for a representative city replay.
