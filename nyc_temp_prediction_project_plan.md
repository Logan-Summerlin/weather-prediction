# NYC Temperature Prediction Project Plan (Operational + Trading)

**Last updated:** 2026-02-13  
**Scope:** Contract-aligned probabilistic forecasting, calibration, bucketization, EV-aware trading simulation for KXHIGHNY.

## 1) Mission

Produce a calibrated daily probability distribution for NYC max temperature at contract resolution, convert to exact KXHIGHNY bucket probabilities, compare against market-implied probabilities, and only trade when edge remains positive after costs.

## 2) Current implemented state

### Contract and benchmark alignment
- Benchmarking is aligned to KXHIGHNY contract buckets and pre-settlement/settled market references.
- Model lineage currently analyzed through:
  - **E0–E22** (`run_e0_e8_best_model_benchmark.py`)
  - **E38–E42** (`run_wga_v2_benchmark.py`)
  - **U0–U9** (`run_unified_outperformance_benchmark.py`)

### Forecasting stack
- Base forecast streams: flat model + WGA model probability pipelines.
- Advanced variants include contract-level Brier-optimized MLP synthesis, Platt+isotonic recalibration, and regime-conditional features.
- Unified variants (U-family) combine flat, WGA, NWS, and market-state features with gating diagnostics.

### Calibration + diagnostics
- Calibration layers include isotonic and Platt+isotonic at multiple stages.
- Reliability/ECE and Brier decomposition outputs are generated in benchmark artifacts.
- Seasonal stress slices and OOS-focused diagnostics are included in unified reports.

### Trading evaluation
- EV-gated simulation is active with threshold sweeps and paper-trading promotion checks.
- Fees/spread-aware assumptions are integrated in benchmark simulation outputs.

## 3) High-priority gaps to close

1. Formalize a hard-cutoff data availability manifest for every live feature.
2. Add explicit automated kill-switch checks into the daily orchestration path.
3. Increase execution realism for queue position/fill uncertainty in backtests.
4. Finalize production promotion rubric across E/WGA/U families.

## 4) Layered operating plan

### Layer A — Ingestion
- Preserve strict separation of operational vs training-only sources.
- Attach provenance metadata and cutoff eligibility per source.

### Layer B — Feature engineering
- Keep deterministic, time-safe transforms and missingness-aware logic.
- Maintain feature parity between historical training and live inference.

### Layer C — Modeling
- Maintain three active families: E-core, WGA extensions (E38–E42), Unified U-series.
- Optimize for calibrated probabilities and OOS Brier/reliability.

### Layer D — Calibration + bucketization
- Enforce exact contract bucket conversion and probability mass checks.
- Maintain chronological calibration windows and monitor drift.

### Layer E — Trading + risk
- Use cost-adjusted EV gating and capped risk exposures.
- Halt on missing critical inputs, schema breaks, or calibration anomalies.

## 5) Promotion gates before live scaling

### Forecast gate
- OOS Brier must consistently beat NWS baseline and remain stable across seasonal slices.

### Calibration gate
- Reliability/ECE and interval checks within configured tolerance bands.

### Trading gate
- Positive conservative paper-trading profile with acceptable drawdown behavior.

### Operations gate
- Complete daily run by cutoff with full audit artifacts and no critical validation failures.

## 6) Repository hygiene updates completed

- Moved clearly legacy exploratory scripts to `ARCHIVE/legacy_experiments/`.
- Preserved legacy backtest runner in `ARCHIVE/legacy_runners/`.
- Updated documentation set under `docs/` for current E42/U9-era state.
- Updated `.claude/rules/MEMORY.md` to synchronize active model/benchmark status.
