# Repository Rationalization Plan (KISS / DRY / YAGNI)

> **Date:** 2026-02-20  
> **Intent:** Aggressively reduce active code and maintenance overhead while preserving behavior, tests, documentation quality, and full local data/results workflows.

## 1) Current State (Measured)

### File volume snapshot

- Total tracked files in repo: **1242**
- Top-level concentration:
  - `data/`: **502 files**
  - `ARCHIVE/`: **350 files**
  - `results/`: **242 files**
  - `scripts/`: **65 files**
  - `src/`: **40 files**
  - `tests/`: **28 files**

### Python LOC snapshot

- `src/`: **24,610 lines** across 40 modules
- `scripts/`: **23,982 lines** across 63 scripts
- `tests/`: **17,764 lines** across 28 test files

### High-confidence duplication zones

1. **City wrapper scripts:** 30 wrapper files, 330 total lines (11 lines each), all same delegation pattern.
2. **Config split-brain:** `src/city_config.py` + per-city `config_*.py` + NYC `config.py/config_expanded.py` duplicating contract/bucket/city metadata ownership.
3. **Benchmark runner sprawl:** three NYC benchmark scripts (`run_e0_e8_*`, `run_wga_v2_*`, `run_unified_*`) are very large and partially overlapping in orchestration logic.
4. **Test repetition:** repeated city/test patterns across modules that are good candidates for parameterization and fixture reuse.

---

## 2) Rationalization Principles (Hard Rules)

- **KISS:** one canonical path per workflow stage.
- **DRY:** one source of truth for city metadata and pipeline stage orchestration.
- **YAGNI:** keep only experiment code that remains on the promotion path.
- **No functionality loss:** every removed entrypoint gets either a compatibility shim or a documented migration command.
- **No operational regression:** preserve contract alignment, cutoff safety, calibration-before-trading, and kill-switch requirements.
- **No data/output externalization:** retain in-repo `data/` and `results/` workflows as first-class project assets.

---

## 3) Target End State

1. **Single canonical execution interface** for all city/stage runs.
2. **Single city configuration system** consumed by all modules.
3. **Clear separation of active vs legacy code paths** while retaining all required data/results in-repo.
4. **Research code quarantined** behind explicit `experiments/legacy` boundaries.
5. **Minimal, role-based documentation set** with one obvious reference per concern.

---

## 4) Execution Plan (Aggressive, Ordered)

## Phase A — Stop the bleeding (1 PR) ✅ COMPLETED (2026-02-21)

### A1. Canonicalize command surface
- Keep canonical entrypoints:
  - `scripts/run_city_pipeline.py`
  - `scripts/run_<stage>.py` (6 unified stage scripts)
- Convert `run_<city>_<stage>.py` wrappers to either:
  - generated wrappers from one template, or
  - minimal static wrappers with enforced format.

**Success metric:** wrapper surface managed by one implementation pattern with no manual drift.

### A2. Enforce no-new-duplication gate
- Add CI test to fail if new per-city wrappers are introduced outside approved pattern.
- Add CI check for high-overlap script copies (AST/function signature similarity threshold).

**Success metric:** no net growth in duplicate entrypoint logic.

**Completion notes (2026-02-21):**
- Added `scripts/generate_city_stage_wrappers.py` as the single wrapper template generator and regenerated all `run_<city>_<stage>.py` wrappers from it.
- Added CI guardrails in `tests/test_script_wrapper_consolidation.py` and `tests/test_script_duplication_guardrails.py` to block wrapper drift and detect high-overlap copied orchestration scripts.

---

## Phase B — Collapse configuration duplication (1–2 PRs) ✅ COMPLETED (2026-02-21)

### B1. Define single `CityConfig` schema
- Expand `src/city_config.py` to include station/ring/sector metadata currently split across `config_*.py` and `config_expanded.py`.
- Add explicit fields for:
  - contract spec (bucket edges, inclusivity, settlement conventions)
  - observation anchor station(s)
  - operational data source availability and cutoff assumptions

### B2. Migrate consumers
- Update `src/operational_data.py`, `src/wga_data_pipeline.py`, and scripts to consume only `get_city_config()`.
- Remove direct runtime imports of `config_chicago`, `config_philadelphia`, `config_atlanta`, `config_austin`, and NYC legacy config modules.

### B3. Decommission legacy config usage
- Keep short-term compatibility stubs that re-export from `city_config.py`.
- Remove stubs after one release cycle and parity verification.

**Success metric:** one metadata source of truth; zero direct runtime dependency on legacy config modules.

**Completion notes (2026-02-21):**
- Consolidated runtime city metadata into `src/city_config.py` + `src/city_config_runtime_data.py` and expanded `CityConfig` with contract-spec, operational cutoff/source assumptions, and station-network metadata fields.
- Updated `src/operational_data.py`, `src/wga_data_pipeline.py`, and all six unified stage scripts to consume `get_city_config()` / `get_city_runtime_config()` only (no runtime imports of per-city legacy config modules).
- Replaced `config_chicago.py`, `config_philadelphia.py`, `config_atlanta.py`, `config_austin.py`, and `config_expanded.py` with compatibility stubs that re-export from `city_config`.

---

## Phase C — Decompose oversized benchmark scripts (2 PRs)

### C1. Extract reusable benchmark engine
- Create `src/benchmark_runner.py` with:
  - dataset loading
  - model registry
  - train/eval loop
  - artifact writing
  - reporting hooks
- City benchmark scripts become thin model-set declarations + config.

### C2. Unify overlapping NYC benchmark code
- Merge shared orchestration from:
  - `scripts/run_e0_e8_best_model_benchmark.py`
  - `scripts/run_wga_v2_benchmark.py`
  - `scripts/run_unified_outperformance_benchmark.py`
- Retain distinct model families through model registries, not script forks.

**Success metric:** >50% LOC reduction across NYC benchmark runners with parity on golden windows.

---

## Phase D — Rationalize tests (1 PR)

### D1. Consolidate repetitive tests
- Parameterize repeated city/test patterns.
- Move duplicated setup code into shared factories/fixtures.

### D2. Focus on behavior-significant invariants
- Prioritize tests for:
  - time-safe splitting
  - no leakage
  - calibration reliability
  - bucket sum-to-one and boundary correctness
  - EV after fees/slippage assumptions
- Remove redundant shape/type tests repeated across modules.

**Success metric:** lower test LOC and runtime with unchanged or improved defect detection.

---

## Phase E — Documentation minimization with full clarity (1 PR)

### E1. Canonical documentation set
- Keep a concise canonical set aligned to operations and model governance.
- Ensure NYC project plan remains explicitly indexed from `README.md`.

### E2. Resolve doc overlap
- Merge overlapping guidance docs.
- Mark superseded docs with deprecation headers (no silent drift).

**Success metric:** no contradictory instructions across active docs.

---

## 5) Guardrails (Must Stay True During Cuts)

- Contract bucket definitions remain bit-for-bit aligned with settlement logic.
- Calibration remains mandatory before bucketization/trading.
- Live pipeline uses only cutoff-safe operational inputs.
- Backtests continue to include fees + conservative slippage.
- Kill-switch and data-integrity halts remain active and test-covered.
- Existing in-repo data/results workflows remain supported.

---

## 6) Concrete Reduction Targets

- **Script duplication:** reduce duplicate entrypoint logic by at least **50%**.
- **Config duplication:** reduce to **one runtime source of truth**.
- **Benchmark orchestration LOC:** reduce NYC benchmark script LOC by **>50%** via shared engine.
- **Test LOC/runtime:** reduce repetitive test overhead by **25–35%** with parameterization.
- **Net Python LOC:** reduce by **30–45%** without dropping required capabilities.

---

## 7) Rollout / Risk Control

- Ship each phase independently with parity checks.
- Maintain migration aliases for one release cycle.
- Add golden-day replay tests before removing any legacy path.
- Freeze feature work during rationalization window to reduce merge churn.

---

## 8) First 10 Tasks (Immediate Backlog)

1. Add baseline inventory script (`scripts/audit_repo_footprint.py`) and commit metrics snapshot.
2. Add CI policy rejecting wrapper script drift from approved pattern.
3. Implement expanded `CityConfig` schema in `src/city_config.py`.
4. Migrate one city (CHI) end-to-end to new config source.
5. Add parity tests for CHI old-vs-new config behavior.
6. Migrate remaining cities to unified config consumption.
7. Extract shared benchmark engine from one NYC benchmark script.
8. Plug second and third NYC benchmark scripts into same engine.
9. Consolidate repeated tests using parameterized fixtures.
10. Resolve doc overlap and publish a final canonical doc index.
