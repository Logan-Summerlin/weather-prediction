# Weather Model Status + Benchmark Review (Current Repository Snapshot)

## Scope
This review consolidates current model status and benchmark outputs from city-level promotion reports, unified benchmark artifacts, and project planning docs.

## Executive status by city

| City | Current status | Best model (current artifact) | Benchmark score snapshot | Trading snapshot |
|---|---|---|---|---|
| NYC | Operational pipeline; promotion evaluation complete (pass) | U7/U8/U9 family appears strongest by period | U7 real-Kalshi benchmark Brier 0.1091 vs market 0.1253 (edge +0.0162); U9 later benchmark Brier 0.1060 vs market 0.1099 (edge +0.0039) | U7 real-Kalshi backtest +$2,406 (Sharpe 6.07); U9 +$340 (Sharpe 2.76) but narrower edge |
| Chicago | Promotion v2 PASS (10/10) | U7_extended_mlp (promotion gate winner), with U8_cv_ensemble best raw unified benchmark | Promotion gate: U7 contract Brier 0.1091, Kalshi 0.1253 (edge +0.0162); unified benchmark: U8 0.1087 | Real-Kalshi gate PnL +$2,405.95, max DD -3.08% |
| Philadelphia | Promotion v2 PASS (10/10) | U9_kitchen_sink | Promotion gate: U9 contract Brier 0.1060, Kalshi 0.1099 (edge +0.0039) | Real-Kalshi gate PnL +$339.60, max DD -10.86% |
| Atlanta | Promotion FAIL (11/13) | Ridge baseline currently best in benchmark artifacts | Benchmark summary best Brier 0.01527 (Ridge); passes most quality gates | Fails risk gate on extreme drawdown metric and missing model checkpoint artifacts |
| Austin | Promotion FAIL (7/13) | Conflicting artifacts: base benchmark says Ridge best; synthesis summary says U9 best | Base benchmark best Brier 0.1795; synthesis summary best overall Brier 0.01655 (U9) | Real-Kalshi gate negative (-$1,059.59), fails drawdown and checkpoint gates |

## Detailed benchmark status

### 1) NYC model families (E-series, WGA-v2, Unified)
- Project plan marks NYC as fully operational and cites completed promotion evaluation.
- Current documented top lineage by benchmark period:
  - U7_extended (real-Kalshi eval window): contract Brier 0.1091, material edge vs market.
  - U8_cv_ensemble (cross-city benchmark): contract Brier 0.1087.
  - U9_kitchen_sink (later real-Kalshi eval window): contract Brier 0.1060, but much narrower market edge.
- Interpretation: statistical fit improved over time, but tradable edge appears to compress as model sophistication increases and/or market efficiency rises.

### 2) Chicago unified model stack
Unified benchmark contract Brier (lower better):
- U0 0.1885 → U1 0.1440 → U2 0.1181 → U3 0.1143 → U5 0.1147 → U7 0.1096 → **U8 0.1087 (best)**; U9 slight regression to 0.1097.
- Promotion winner remains U7_extended_mlp because it couples strong Brier + strong real-Kalshi trading + favorable drawdown.
- Interpretation: Chicago model development is healthy, with clear gains from calibration and synthesis layers and a robust production candidate.

### 3) Philadelphia unified model stack
Unified benchmark contract Brier:
- U0 0.1959 → U1 0.1340 → U2 0.1058 (strong) → U7 0.1108 → U8 0.1091 → **U9 0.1060 (near-best, selected in promotion)**.
- Promotion indicates a small but positive edge to Kalshi (0.0039), acceptable but thin.
- Interpretation: model quality is production-viable, but trading edge margin is fragile.

### 4) Atlanta stack
- Base benchmark: Ridge best at 0.01527 with large OOS count (2,192 days).
- Promotion fails are operational/risk-oriented, not core forecast-skill failures:
  - extreme drawdown metric failure,
  - missing persisted model checkpoints under `models/atlanta`.
- Interpretation: forecasting core is decent, but production reliability and risk framework are not yet promotion-safe.

### 5) Austin stack
- Base benchmark says Ridge best at 0.1795.
- Unified synthesis summary says U9 best at 0.01655 (very different scale).
- Promotion report references model 0.2375 vs market 0.1446 and fails multiple trading/risk gates.
- Interpretation: Austin currently has benchmark-definition inconsistency and likely metric-unit mismatch across artifacts, making promotion decisions unreliable.

## Cross-cutting diagnosis (ML + prediction-market perspective)

1. **Benchmark scale inconsistency remains a major governance risk**
   - Cross-city Brier scale audit already notes row-level Brier is not directly comparable across evaluation unit definitions.
   - Austin appears to show this issue most severely (0.016-level vs 0.17–0.23-level depending on artifact).

2. **Calibration is helping materially, but market edge is city-dependent**
   - Chicago has healthy edge buffer; Philadelphia edge is thin; Austin currently negative in real-Kalshi backtest.
   - You should treat positive benchmark edge as necessary but insufficient without robust execution assumptions.

3. **Promotion framework quality improved in v2 for CHI/PHL, but not yet standardized for ATL/AUS**
   - CHI/PHL v2 reports are coherent and pass all gates.
   - ATL/AUS still show artifact-path and checkpoint issues that can force false negatives or unreliable go/no-go calls.

4. **Risk controls are underdeveloped outside CHI/PHL**
   - Massive reported drawdowns in ATL/AUS indicate position-sizing/exposure-control logic is not production-grade for those cities.

## Improvement plan (prioritized)

### Priority 0 — Measurement and benchmark integrity (must do before model iteration)
1. **Unify scoring units and contract-row definitions across all city pipelines.**
   - Enforce one canonical benchmark schema: `{evaluation_unit, rows_per_day, row_brier, day_brier}`.
2. **Version benchmark artifacts and require promotion to consume only the canonical version.**
   - Prevent mixing `benchmark_summary.json`, `*_benchmark_summary.csv`, and synthesis summaries with incompatible scales.
3. **Add an automated “metric consistency gate.”**
   - Fail run when score scales differ by >X between artifacts for same city/time window.

### Priority 1 — Trading robustness
1. **Adopt conservative execution model as default in every city backtest.**
   - Explicit spread + slippage + partial fill penalties.
2. **Replace current sizing for ATL/AUS with capped fractional Kelly + hard per-contract caps.**
   - Add adjacency exposure netting across neighboring buckets.
3. **Add regime-aware trade suppression.**
   - No-trade when calibration uncertainty or market dislocation exceeds threshold.

### Priority 2 — Calibration and distribution quality
1. **Use rolling, time-local calibration windows with drift checks.**
   - Especially important where edge is narrow (PHL) or unstable (AUS).
2. **Track reliability by season/regime and bucket difficulty.**
   - Promote only if worst-slice reliability remains within tolerance.
3. **Evaluate CRPS/NLL alongside contract Brier in promotion packets.**
   - Prevent over-optimizing contract buckets at cost of distribution integrity.

### Priority 3 — City-specific model actions
- **Chicago:** keep U7/U8 as champion-challenger pair; focus on edge durability under stricter cost assumptions.
- **Philadelphia:** prioritize edge expansion (feature robustness, regime-specific calibration), not just tiny Brier wins.
- **Atlanta:** operationalize artifact persistence + checkpointing; fix risk/sizing before any live deployment.
- **Austin:** stop trading experimentation until benchmark-unit mismatch and promotion artifact wiring are resolved.

## Deployment recommendation today
- **Ready / near-ready:** Chicago.
- **Paper-trade with tight limits:** Philadelphia.
- **Not deployment-ready:** Atlanta (risk/ops), Austin (metric integrity + negative real-Kalshi PnL).

