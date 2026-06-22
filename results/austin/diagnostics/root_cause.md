# Austin (AUS) Root-Cause Analysis — Phase 2 Deep-Dive

**Question:** why does Austin show the worst model-vs-market gap of the
portfolio (Phase 0 ledger: model Brier 0.236 vs market 0.145, busted backtest),
*after* the settlement station was verified correct?

**Verdict:** Austin's failure is **model quality**, not contract alignment.
Two compounding causes, in priority order:

1. **A broken probabilistic head** — the deployed `HeteroscedasticNN` emits a
   single constant sigma of **54.6 °F** (the Phase 0 variance-collapse
   pathology), so every probability it produces is near-uninformative.
2. **No NWP/MOS inputs** — the model sees only station lags. The market is
   NWP-informed; lags alone cannot close the gap even once the head is fixed.

The GHCN→ASOS offset and the settlement station are **not** the cause (see §3).

---

## 1. The deployed model is mis-specified (dominant cause)

From `results/austin/diagnostics/` (run `scripts/run_model_diagnostics.py
--city aus`):

| Metric | Value | Reading |
|---|---|---|
| Mean sigma | **54.6 °F** (constant) | Variance collapse — sigma absorbed the mu residual |
| Calibration ratio (realized_rmse / mean_sigma) | **0.11** | ~9× over-dispersed; distribution is uninformative |
| PIT uniform | **False** (KS 0.396, p≈0) | Gross calibration failure |
| Model vs market Brier (3,214 OOS contracts) | **0.2166 vs 0.1446** | NO-EDGE / MONITOR |

A constant 54.6 °F sigma on a target whose realized day-ahead RMSE is ~5.8 °F
means the model spreads probability across the entire bucket ladder regardless
of conditions. This is the exact convergence pathology fixed in Phase 0 (mu
head initialized at target mean; log_sigma clamped) — but the Austin
checkpoint predates / did not receive that fix.

**Retraining closes most of the gap.** The Phase 2 distribution-head
comparison (`scripts/run_distribution_head_comparison.py --city aus`, clamped
log_sigma) on the same processed features yields, on the held-out test window:

| Head | OOS CRPS (°F) | Contract Brier |
|---|---|---|
| Gaussian | 2.95 | 0.171 |
| **Quantile (best)** | **2.40** | **0.169** |
| Mixture (2-comp) | 2.95 | 0.169 |

i.e. a correctly-trained head drops contract Brier from **0.217 → ~0.169**.
That single fix removes roughly two-thirds of the distance to the market
(0.145). The constant-sigma model was the dominant problem.

## 2. The residual gap is the missing-NWP gap

Even fixed, ~0.169 still does not beat the market's 0.145. The remaining gap
is structural: the feature set is station lags only, with **no MOS/NWP**. The
market prices the morning GFS/NAM MOS guidance; a lag-only model is
systematically behind on frontal timing and airmass changes. This matches the
Phase 0 portfolio-wide diagnosis and is the Phase 2 NWP-integration lever
(deliverable #2): wire `MOSCorrectionNet` on the `TMAX − MOS_TMAX` residual
once KAUS MOS is collected.

Residual structure that NWP would most help (from the diagnostics report):

- **Summer bias −1.24 °F** (model runs warm in JJA), **Fall bias +1.10 °F**
  (runs cold in SON) — seasonal/regime errors a MOS anomaly feature targets.
- **Hot-regime bias −1.12 °F** (mu top tercile) — the high-temperature tail,
  where Austin's contracts concentrate, is exactly where lag-only models drift.

## 3. Ruled out: station alignment and the ASOS/GHCN offset

- **Settlement station** verified correct in Phase 0 (Bergstrom; the Camp
  Mabry hypothesis was disproven against settled rows).
- **ASOS vs GHCN at the target station KAUS** (`asos_ghcn_tmax_comparison.md`):
  mean bias **−0.29 °F**, MAE 1.07 °F, RMSE 1.97 °F, corr 0.992 over 9,739
  overlap days. The offset at the contract station is small and unbiased — it
  cannot explain a 0.07 Brier gap. (Some *network* stations show larger
  offsets, e.g. KSEP +0.37/MAE 3.76, KPSN +1.37/MAE 4.21, but these are
  upwind context features, not the target.)

## 4. Recommended sequence

1. **Retrain Austin** with the Phase 0-fixed head (clamped log_sigma, mu init
   at mean) and adopt the best distribution head (quantile per the comparison).
   Regenerate the ledger; expect contract Brier ≈ 0.17.
2. **Collect KAUS MOS** and wire `MOSCorrectionNet` (Phase 2 deliverable #2);
   re-evaluate model vs market Brier.
3. **Gate:** only if model Brier < market Brier on real OOS presettlement
   prices does Austin move toward PROMOTED. Until then Austin stays **MONITOR**
   — do not tune EV thresholds to manufacture P&L.
