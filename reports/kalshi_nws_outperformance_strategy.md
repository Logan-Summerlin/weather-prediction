# Strategy Memo: Pathways to Beat Kalshi PreSettlement and NWS

Date: 2026-02-12  
Author: GPT-5.2-Codex technical analysis

## 1) Current state from benchmark evidence

### What is already working
- The current model family consistently beats NWS in bucket-level Brier score (0.1335 vs 0.1418 overall in the best-model report).
- The best-model lineage variants E0–E8 all remain in a narrow Brier band around ~0.1333–0.1344 (except feature-pruning sweep at 0.1364), indicating a stable but plateaued performance region.
- The model is better calibrated than NWS and Kalshi pre-settlement by ECE in the cited run (Model ECE 0.0230 vs NWS 0.0324 and PreSettlement 0.0557).

### What is not yet working
- The model does not beat Kalshi PreSettlement on Brier in full benchmark periods (Model ~0.1335 vs PreSettlement ~0.1271).
- Under realistic crossing execution assumptions with fees/slippage, strategy P&L is negative OOS across thresholds in the latest benchmark report.
- Variant search in E0–E8 delivers marginal ranking improvements but no structural leap versus market prices.

### Implication
The project has a strong forecasting base and calibration discipline, but likely needs:
1. Better regime-conditioned distribution quality (especially tails and transition days), and
2. Better execution/selectivity (trade less, only on robust edge clusters), and
3. Better alignment between offline objective and tradable EV under microstructure costs.

## 2) Why plateau likely happened (cross-reading docs + code)

### A) Model-side plateau
- Archived catalog shows biggest historical gain came from MOS-residualization; pure architecture scaling (deeper MLP/RNN/Conv) was not decisive.
- Phase-1 summaries show tight MAE/RMSE differences across many architectures/seeds, consistent with diminishing returns from architecture-only changes.

### B) Trading-side mismatch
- Benchmark evolution suggests score improvement does not automatically convert to positive market P&L once execution assumptions are realistic.
- Existing trading module includes EV and Kelly-style tools, but there is still room for robust risk filters (uncertainty gating, market-liquidity-aware throttles, event-cluster controls).

### C) Operational meteorology failure modes (from Temperature_Forecasting_ML_LLM_Report)
Most likely remaining alpha is regime-specific:
- Sea-breeze timing and coastal transitions,
- Cloud/radiation-driven Tmax errors,
- Frontal/change-point days,
- Nocturnal stability/mixing errors.

## 3) Most promising model families going forward (ranked)

## Tier 1 (highest expected payoff)

### 1. Regime-conditioned synthesis model (station model + NWP/MOS + market context)
**Rationale:** This directly matches the operations workflow in the research report: baseline guidance + local correction + uncertainty.  
**How:** Use `src/synthesis_model.py` as the meta-learner layer combining:
- Station/local model distribution,
- MOS/NWP spread and bias features,
- Market state descriptors (depth, spread, staleness, imbalance).

**Key requirement:** output full distribution (quantiles or mixture), then calibrate on holdout.

### 2. Wind-gated attention + probabilistic head as primary local model
**Rationale:** `src/wind_gated_attention.py` already encodes physically meaningful upwind weighting and missing-mask support; this is exactly where coastal advection and frontal flows can be learned better than flat tabular MLPs.

**Upgrade path:**
- Promote Gaussian head to mixture density head (2–3 components) for bimodality / regime switches,
- Add regime tokens (marine/onshore proxy, cloud class, frontal tendency),
- Use CRPS/NLL hybrid objective with monotonic quantile post-checks.

### 3. Calibration as first-class model component
**Rationale:** Results show decent ECE but high-probability bins still appear overconfident in reliability tables. Tail calibration errors can destroy trading EV despite acceptable global Brier.

**Upgrade path:**
- Conditional isotonic (season + spread bin + regime class),
- Online calibration drift monitor + auto-halt if reliability degrades,
- Convert calibration diagnostics into trade eligibility flags.

## Tier 2 (medium expected payoff)

### 4. Distributional ensemble diversification
- Keep seed averaging, but diversify by objective and inductive bias:
  - Residual NN,
  - Tree booster residual model,
  - Wind-attention model,
  - Quantile model.
- Blend with performance- and uncertainty-aware weights (dynamic stacking).

### 5. Quantile and MDN extensions (already tested partially in E5/E6)
- E5/E6 are close to top variants; do not discard.
- Re-run with stronger regularization and explicit tail-loss weighting for extreme buckets where market edges are concentrated.

### 6. Conformal overlays for risk control
- Use conformalized quantiles to control miscoverage regime-by-regime.
- Trade only when conformal interval width indicates confident edge.

### 7. Capacity-scaled architecture sweeps (deeper/wider nets, but earned)
- Add a disciplined sweep for hidden depth/width to test whether the current plateau is partly under-capacity in high-regime-complexity windows.
- Candidate shapes for tabular backbones:
  - `[128, 64]` (current-like baseline),
  - `[256, 128, 64]`,
  - `[384, 192, 96, 48]` with stronger dropout + weight decay,
  - residual MLP blocks with skip connections and LayerNorm.
- Candidate shapes for attention/synthesis backbones:
  - increase station embedding dim (32 → 64),
  - increase attention dim (16 → 32),
  - increase synthesis hidden stack (e.g., 128/64/32 → 256/128/64).
- Guardrails:
  - only retain larger models if they improve OOS CRPS/Brier and do not degrade calibration,
  - require stability across seeds and seasonal slices,
  - enforce training-time regularization and early stopping.

### 8. Station-network expansion (more stations, smarter aggregation)
- Expand from current station set to a broader metro + upstream ring, then prune by data quality and lead-time value.
- Add sectorized station pools (N/NE/E/SE/S/SW/W/NW + coastal-inland split) and dynamic upwind/downwind composites.
- Add station reliability weights learned from:
  - long-run missingness,
  - regime-conditional bias,
  - latency/availability by cutoff.
- Evaluate value of extra stations with ablation ladders:
  - base stations,
  - +nearby ring,
  - +upstream ring,
  - +full set with mask-aware attention.

### 9. Training-data expansion and temporal coverage upgrades
- Extend training window with all available historical years that satisfy time-safe feature parity.
- Create regime-balanced sampling/weighting so rare but tradable events (frontal jumps, marine intrusions, extreme tails) are not underlearned.
- Add rolling retraining experiments:
  - long-history model,
  - recency-weighted model,
  - hybrid (long backbone + short-horizon bias adapter).
- Add data-source parity checks so new years/sources do not reintroduce hidden train/inference mismatch.

### 10. AVN/ETA MOS historical backfill (candidate extension to 2002)
- Test whether legacy AVN MOS and ETA MOS can be harmonized with current GFS/NAM MOS ensemble inputs to extend training history back to ~2002.
- Build a source-harmonization layer that:
  - maps AVN/ETA variable names and units to current schema,
  - applies model-era indicators (pre/post model upgrades),
  - learns source-specific bias offsets before ensemble blending.
- Run strict parity checks to prevent hidden train/inference mismatch:
  - only use fields available by the live cutoff,
  - preserve issuance-time semantics and valid-time alignment,
  - track coverage/missingness differences by era.
- Evaluate whether the longer history improves distributional skill (CRPS/Brier/calibration) and tail-event robustness versus possible nonstationarity costs.
- Explicitly test if this backfill enables extending the validation period by one additional year while keeping a clean final holdout for trading realism.
- Keep this pathway behind a feature flag until backfill quality and calibration stability pass acceptance thresholds.

## Tier 3 (selective experiments)

### 11. Sequence architectures only for targeted subproblem
- Prior RNN/Conv evidence was weak broadly.
- Restrict sequence models to nowcast-like short horizon correction modules (e.g., pre-cutoff latest-hour update signal), not whole-system replacement.

### 12. Multi-task learning
- Jointly predict Tmax + key regime indicators (cloud category, marine influence class, frontal flag) to improve representation robustness.

## 4) Concrete optimization plan to beat PreSettlement

## Phase A — Contract/truth strictness (must lock first)
Checklist:
- [ ] Re-validate Kalshi contract site, day boundary, inclusivity/rounding.
- [ ] Verify training target equals settlement definition exactly.
- [ ] Audit all live features for cutoff-time availability.
- [ ] Add an automated “time-safe feature audit” artifact per run.

Success metric: zero unresolved alignment mismatches.

## Phase B — Forecast quality leap (distribution-focused)
Experiments:
1. **WGA-MDN**: wind-gated attention + 2–3 component MDN head.  
2. **Synthesis-Stacker**: meta-learner blending local distro + MOS/NWP + market state.  
3. **Conditional calibration grid**: isotonic by season × spread tercile × regime.
4. **Capacity sweep**: hidden-layer/depth sweeps for residual MLP + synthesis model with strict regularization.
5. **Station expansion ladder**: compare incremental station-network enlargements.
6. **Data-history extension run**: long-history vs recency-weighted training.
7. **AVN/ETA MOS backfill study**: legacy-MOS harmonization to assess 2002-era extension + one-year longer validation window.

Selection metrics (chronological only):
- Primary: CRPS + bucket Brier (overall + OOS + regime slices)
- Secondary: MAE/RMSE
- Calibration gates: PIT and interval coverage within tolerance.

## Phase C — EV-aware execution redesign
Core changes:
- Add “edge quality score” = f(prob edge, calibration confidence, market spread/depth, regime confidence).
- Trade only if edge quality > threshold; enforce sparse trading.
- Replace global threshold with dynamic threshold by liquidity and uncertainty.
- Use capped fractional Kelly + daily and cluster exposure limits.

Backtest requirements:
- Crossing assumptions + fees + slippage + stale-quote penalties.
- Bootstrap confidence intervals for OOS P&L and Sharpe.
- Stress tests by season and volatility regime.

## Phase D — Paper-trade gate before scale
Promotion criteria:
- OOS Brier <= PreSettlement on recent rolling window OR statistically indistinguishable but with positive EV after costs.
- Positive paper-trade P&L with bounded drawdown over minimum sample window.
- No calibration drift alerts for N consecutive weeks.

## 5) High-impact feature expansions (time-safe)

These are aligned with the Temperature_Forecasting_ML_LLM_Report and should be prioritized if available by cutoff:
- Cloud/radiation proxies near dawn (critical for Tmax trajectory).
- Marine-flow/sea-breeze proxies (wind direction shift tendency + SST contrast proxy).
- Pressure tendency and frontal proximity indicators.
- Regime stability features (overnight inversion/mixing proxies).
- Better station-sector composites (upwind/downwind grouped aggregates).
- Ensemble-spread and disagreement features from multiple operational forecast sources.
- Station-level reliability and latency features to inform masking/attention.
- Explicit missingness indicators for every major feature family.

## 6) Risks and how to mitigate

### Risk 1: Training–inference mismatch
Mitigation:
- Mirror live feature computation in offline builds.
- Track per-feature population shift and missingness.

### Risk 2: Overfitting in small edge region
Mitigation:
- Use nested chronological validation and strict final holdout.
- Tune trade thresholds only on pre-designated calibration period.

### Risk 3: Calibration drift kills EV
Mitigation:
- Reliability monitor with hard kill switch in trading system.
- Regime-conditioned recalibration schedule.

### Risk 4: Market microstructure eats apparent edge
Mitigation:
- Conservative fill model and slippage buffer.
- Prefer fewer, higher-confidence trades.

## 7) Recommended immediate next sprint (2 weeks)

1. Implement WGA-MDN prototype and evaluate against current best-model baseline.
2. Add conditional isotonic calibration layer and diagnostics report.
3. Run targeted architecture-capacity sweep (hidden layers/width) with strict OOS and calibration gates.
4. Run station-expansion ablation ladder and keep only cutoff-safe stations.
5. Run AVN/ETA MOS backfill feasibility test (schema map + bias harmonization + parity audit).
6. Build EV-aware trade gating (quality score + dynamic threshold).
7. Run full historical replay with realistic execution assumptions.
8. Produce decision memo: go/no-go for paper trading.

## 8) Bottom line

You likely already have enough forecasting skill to beat NWS consistently.  
To beat Kalshi PreSettlement robustly, the next leap is unlikely to come from larger generic networks alone.
It should come from **regime-aware distribution modeling + conditional calibration + selective EV-aware execution** under realistic market microstructure constraints.

---

## 9) Implementation tracker (started 2026-02-12)

### Implemented in this sprint

1. **Conditional calibration grid prototype (partial Phase B.3)**
   - Added `E9_conditional_calibration_grid` into `scripts/run_e0_e8_best_model_benchmark.py`.
   - Method implemented: season × spread-tercile × regime-tercile calibration key with hierarchical fallback (cell → season → global).
   - Regime proxy used for first pass: absolute day-over-day change in model `mu` (stable/transition/volatile bins).
   - Training/calibration chronology preserved (fit on 2023 only; evaluated on 2023–2025 benchmark set).

2. **EV-aware execution gating prototype (partial Phase C)**
   - Added dynamic edge-quality gating experiment for the best-Brier model in `scripts/run_e0_e8_best_model_benchmark.py`.
   - Implemented quality score from: `|edge|`, quoted spread proxy, and model uncertainty (`sigma`-normalized).
   - Implemented dynamic threshold: `0.01 + 0.5*spread + 0.04*sigma_norm`, then quality cut filters.
   - Results artifact added: `results/prediction_market_benchmark/e0_e8_best_model_base/ev_edge_quality_gating_results.csv` (now includes 95% bootstrap CIs for net P&L and ROI).

3. **Contract/time-safe audit artifact (partial Phase A)**
   - Added contract and time-safety checks to `scripts/run_e0_e8_best_model_benchmark.py`.
   - New artifact: `results/prediction_market_benchmark/e0_e8_best_model_base/contract_and_timesafe_audit.json`.
   - Checks now include direction/threshold integrity and snapshot lag vs fixed 05:00 UTC cutoff.

4. **Liquidity/depth/staleness-aware gating + risk controls (Phase C first implementation)**
   - Extended gating quality with depth proxy (`volume`, `open_interest`) and staleness penalty (snapshot lag).
   - Extended dynamic threshold with depth and staleness terms.
   - Added cluster exposure cap: max 2 trades per (date, 2°F strike neighborhood).
   - Added capped fractional Kelly sizing in simulation (25% Kelly, max 0.30 stake, 0.25 minimum stake in this prototype).

5. **Paper-trading promotion gate automation (Phase D partial implementation)**
   - Added automated go/no-go gate evaluation to `scripts/run_e0_e8_best_model_benchmark.py`.
   - New artifact: `results/prediction_market_benchmark/e0_e8_best_model_base/paper_trading_gate_report.json`.
   - Current checks include: (a) OOS Brier vs PreSettlement, (b) OOS gated P&L with positive lower CI bound, (c) ECE threshold, (d) tail reliability threshold.

6. **Expanded contract parser/audit checks (Phase A advancement)**
   - Extended contract audit with ticker parser + date/strike consistency checks (supports `HIGHNY-...` and `KXHIGHNY-...`).
   - Added rounded-temperature settlement-rule consistency check against realized `actual_outcome` under direction-specific bucket logic.
   - In current run, parser/date/strike checks pass with zero mismatches; settlement-rule mismatch remains non-zero and now quantified for follow-up.

### Results from implementation run

Run command:
`python scripts/run_e0_e8_best_model_benchmark.py`

#### Forecast/Brier impact
- `E9_conditional_calibration_grid` did **not** improve over prior variants in this first pass.
- It matched `E2_seasonal_calibration` exactly in this run:
  - Overall model Brier: **0.1342465**
  - OOS model Brier: **0.1315870**
- Current best variant remains **E3_weighted_ensemble_E4_uncertainty**:
  - Overall model Brier: **0.1333057**
  - OOS model Brier: **0.1300457**

Interpretation:
- Conditional cells were likely too sparse with one calibration year and fell back heavily to seasonal/global calibrators.
- Next iteration should either (a) simplify cell granularity or (b) use a larger calibration window while preserving chronological purity.

#### Trading/EV impact (dynamic gating)
- Dynamic gating reduced trade count materially (e.g., 5,183 baseline-style threshold-0.02 model trades historically vs ~1.8k–2.6k gated in these cuts), but remained negative P&L in this prototype.
- Best all-period gated result in this run:
  - quality_cut=0.05, trades=1,813, net P&L = **-69.84**, ROI **-10.76%**.
- OOS gated outcomes remained negative across cuts (roughly **-47.5 to -56.6** net P&L).
- Added date-block bootstrap confidence intervals (`n=1000`) for gated strategies:
  - Best all-period cut (`q=0.05`) net P&L 95% CI: **[-97.19, -42.89]**; ROI 95% CI: **[-14.85%, -6.62%]**.
  - Best OOS cut (`q=0.05`) net P&L 95% CI: **[-67.30, -26.59]**; ROI 95% CI: **[-17.97%, -7.11%]**.

Interpretation:
- Sparse selective trading alone is insufficient with current edge quality definition.
- CI bands remain strictly negative in this run, increasing confidence that this prototype is not yet tradable.
- We still need stronger calibration confidence signals + tighter microstructure filters (staleness/depth/queue-position proxies) and likely better model edge quality on tradable tails.

#### Trading/EV impact (depth/staleness-aware gating + risk controls)
- After adding depth/staleness-aware thresholds, cluster caps, and capped Kelly sizing, net losses reduced in absolute dollars but remained negative.
- Best all-period result in this pass:
  - quality_cut=0.05, trades=1,321, avg_stake=0.25, net P&L = **-14.95**, ROI **-12.41%**.
- Best OOS result in this pass:
  - quality_cut=0.04, trades=1,053, avg_stake=0.25, net P&L = **-11.10**, ROI **-11.51%**.
- 95% bootstrap CIs remained mostly negative:
  - all-period q=0.04 P&L 95% CI **[-22.60, -9.01]**,
  - OOS q=0.04 P&L 95% CI **[-16.74, -5.95]**.

Interpretation:
- Phase C risk controls improved loss containment, but current edge quality still does not clear profitability after costs.
- Next step remains richer microstructure features and stronger calibration-confidence gating.

#### Contract/time-safe audit findings
- Direction/threshold checks passed: no invalid between-bucket ordering, no missing required threshold bounds by direction.
- Snapshot timing check passed in this dataset: 0 rows after 05:00 UTC cutoff.
- Open issue: summing `settled_market_prob` by date shows large non-unit mass counts, so this field should not be treated as normalized daily bucket mass without explicit market-set partitioning.
- New parser checks (current implementation):
  - `rows_with_unparseable_ticker`: **0**
  - `rows_with_ticker_date_mismatch`: **0**
  - `rows_with_ticker_strike_mismatch`: **0**
  - `rows_with_outcome_rule_mismatch`: **95** (requires explicit contract rounding/inclusivity rules resolution)

#### Paper-trading gate findings (new)
- Automated promotion gate is currently **NOT READY** (`promotion_ready=false`).
- Failing checks in this run:
  - OOS Brier still above PreSettlement (0.1300 vs 0.1271).
  - Best OOS gated strategy remains negative with strictly negative 95% CI lower bound.
  - Tail reliability gate fails (max abs reliability-bin gap ~0.513 > 0.20 threshold).
- Passing check:
  - ECE gate passes (0.02285 <= 0.03 threshold).

### Not yet implemented (from this memo)

#### Phase A
- [~] Re-validate contract site/day-boundary/inclusivity rounding with automated checks. *(advanced: threshold/direction/cutoff + ticker/date/strike parser checks now automated; remaining gap is explicit settlement rounding/inclusivity rule reconciliation, reflected by 95 outcome-rule mismatches)*
- [x] Add explicit time-safe feature-audit artifact per run.

#### Phase B
- [ ] WGA-MDN model training/evaluation integration in benchmark harness.
- [ ] Synthesis-Stacker with market-state inputs.
- [x] Conditional calibration grid prototype (first pass, no gain yet).
- [ ] Capacity sweep for residual + synthesis backbones under strict calibration gates.
- [ ] Station expansion ablation ladder.
- [ ] Data-history extension run.
- [ ] AVN/ETA MOS backfill feasibility implementation.

#### Phase C
- [x] EV-aware edge-quality + dynamic-threshold prototype (first pass, still negative P&L).
- [x] Liquidity/depth/staleness-aware dynamic thresholds using richer market microstructure.
- [x] Cluster exposure limits + explicit capped fractional Kelly in this benchmark script family.
- [x] Bootstrap confidence intervals for gated strategy variants (date-block bootstrap, n=1000).

#### Phase D
- [~] Paper-trading gate criteria automation and monitoring integration. *(implemented in benchmark harness with `paper_trading_gate_report.json`; still pending live-monitor wiring and rolling-window alerting)*
