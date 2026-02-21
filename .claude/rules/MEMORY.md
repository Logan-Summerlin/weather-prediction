# Project Memory

> **RULE:** Keep this file concise, current, and operationally useful.

## Current Project State (2026-02)
- Repository focus: multi-city probabilistic weather forecasting for Kalshi daily max-temperature contracts.
- Primary objective: calibrated predictive distributions converted into contract bucket probabilities, then EV-filtered trading decisions with risk controls.
- Core city support: NYC, Chicago, Philadelphia, Atlanta, Austin.
- Architecture is modularized across ingestion, feature engineering, modeling, calibration/bucketization, and trading simulation.
- Multi-city script flow has been unified (city passed via `--city`) with thin compatibility wrappers for legacy per-city commands.
- Promotion evaluations are implemented; city readiness differs by market edge and calibration robustness.

## Known Strategic Status by City
- **NYC:** mature benchmark stack (E/WGA/U families), strongest reference implementation.
- **Chicago:** strong outperformance signal and positive real-market-style backtest edge.
- **Philadelphia:** weaker edge; calibration and simulated-market robustness remain key concerns.
- **Atlanta:** pipeline complete and promotion gates passed.
- **Austin:** pipeline complete but promotion criteria not yet consistently met.

## Operational Non-Negotiables
1. Contract alignment first (station, day boundary/timezone, units, bucket semantics, settlement rules).
2. Strict cutoff-time safety for live inputs (no delayed/training-only leakage into inference).
3. Chronological splits only; never random shuffle time-series data.
4. Train/inference feature parity is mandatory; quantify and correct mismatch when unavoidable.
5. Trading requires calibrated probabilities and full-cost EV accounting (fees + slippage + execution uncertainty).
6. Kill switch required for missing critical inputs, schema drift, calibration drift, or execution anomalies.

## Quality and Evaluation Expectations
- Use proper probabilistic metrics (CRPS/NLL) and contract-level bucket metrics (including Brier).
- Benchmark against persistence, climatology, linear/ridge, and market-implied probabilities when available.
- Evaluate by regime/season; avoid aggregate-only reporting.
- Keep model complexity only when it materially improves out-of-sample calibration and contract performance.

## Repo Working Memory
- City/contract definitions and bucket logic: `src/city_config.py`.
- Unified city pipeline scripts live under `scripts/` and should be preferred over bespoke one-off scripts.
- Artifacts should remain organized under `data/<city>/`, `models/<city>/`, and `results/<city>/`.

## LLM Workflow Reminder
- For coding and research tasks, **always delegate implementation/research execution to the `analyst` subagent first**, then synthesize and review results before finalizing.
