# Canonical Documentation Index

> Last updated: 2026-06-12  
> Status: Active reference

This index defines the **only active documentation set** for repository operations and governance.

## Active docs (authoritative)

1. `README.md` — repository quickstart and navigation.
2. `Project_Plan` — delivery roadmap and city rollout execution plan.
3. `docs/01_implementation_plan_2026.md` — phased implementation plan
   (model optimization, 10-city expansion, EV dashboard) with Phase 0
   findings and the honest-baseline policy.
4. `results/baseline_ledger.json` — machine-generated honest per-city
   baseline metrics (never hand-edit; regenerate via
   `python scripts/build_baseline_ledger.py`).

## Removed superseded docs

The previously archived overlap docs were removed during rationalization cleanup to eliminate redundant maintenance surface:

- `docs/01_current_state_and_directory.md`
- `docs/02_model_families_and_methods.md`
- `docs/03_principles_and_city_portability.md`
- `docs/04_scripts_consolidation_plan.md`

When conflicts exist, follow this precedence:

1. `README.md`
2. `Project_Plan`
3. Archived docs (historical context only)
