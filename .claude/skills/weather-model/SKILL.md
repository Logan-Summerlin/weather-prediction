---
name: weather-model
description: Use this skill to build, audit, and operate NYC/PHL/CHI weather prediction market pipelines with contract-aligned probabilistic forecasts, mandatory Kalshi PreSettlement ingestion, Contract Brier scoring, EV gating, and risk-managed execution.
---
# Weather Model Skill
## When to use
- Use for NYC/PHL/CHI model work.
- Use for city-template expansion work.
- Use for forecasting + trading audits.
- Use for contract alignment validation.
- Use for calibration and bucketization checks.
- Use for Kalshi benchmark and EV logic.
- Use for backtest realism and promotion gates.
## Non-negotiable rules
- Never use simulated weather outcomes.
- Never use proxy/derived settlement truth.
- Never fabricate missing Kalshi rows.
- Never skip Kalshi PreSettlement fetching.
- Never report only bucket-day Brier.
- Always compute Contract Brier first.
- Always use each day’s listed contracts.
- Always include fees/slippage in EV.
- Always use time-safe, lagged features.
- Always keep chronological data splits.
- Always keep train/calibration/test separated.
- Always halt if critical inputs are missing.
## Repo map (read first)
- `src/city_config.py` city registry + bucket edges.
- `config.py` NYC legacy configuration.
- `config_chicago.py` Chicago station network.
- `config_philadelphia.py` Philadelphia network.
- `src/data_collection.py` GHCN ingestion helpers.
- `src/data_preprocessing.py` feature pipeline.
- `src/model.py` baseline/heteroscedastic NN.
- `src/advanced_model.py` advanced architectures.
- `src/synthesis_model.py` synthesis/meta layer.
- `src/calibration.py` calibration + CDF bucketization.
- `src/contract_brier.py` contract-row scoring.
- `src/kalshi_client.py` market API client.
- `src/kalshi_backtester.py` execution simulation.
- `scripts/fetch_kalshi_presettlement.py` pre-settle.
- `scripts/fetch_kalshi_presettlement_multi.py` multi-city.
- `scripts/run_chi_*` CHI end-to-end scripts.
- `scripts/run_phl_*` PHL end-to-end scripts.
- `scripts/run_city_nws_kalshi_template_benchmark.py` shared benchmark.
## Contract alignment checklist (must pass before modeling)
- Confirm ticker family and naming pattern.
- Confirm settlement station and source IDs.
- Confirm local timezone and day boundary.
- Confirm rounding and inclusivity conventions.
- Confirm threshold semantics per direction.
- Confirm open-tail behavior for extreme bins.
- Confirm settlement source precedence policy.
- Confirm revision/outage handling policy.
- Confirm listed contract rows by date.
- Confirm mapping to model bucket definitions.
- Save contract alignment artifact.
## NYC/PHL/CHI contract anchors
- NYC ticker prefix: `KXHIGHNY`.
- PHL ticker prefix: `KXHIGHPHL`.
- CHI ticker prefix: `KXHIGHCHI`.
- NYC station: `USW00094728`.
- PHL station: `USW00013739`.
- CHI station: `USW00094846`.
- NYC timezone: America/New_York.
- PHL timezone: America/New_York.
- CHI timezone: America/Chicago.
- Bucket edges come from city registry.
- Listed rows can vary by date.
## Required split windows
- Train: 2000-01-01 → 2021-12-31.
- Calibration: 2022-01-01 → 2023-12-31.
- Test: 2024-01-01 → 2025-12-31.
- No random shuffling ever.
- Fit preprocessing params on train only.
- Fit model params on train only.
- Fit post-hoc calibration on calibration only.
- Evaluate final quality on test only.
- Keep split boundaries explicit in reports.
## Operational cutoff and parity policy
- Define one hard morning cutoff.
- Use only sources available by cutoff.
- Training-only archives cannot leak live.
- Every live feature must have live source.
- Every feature needs availability timestamp.
- Audit train/live feature parity regularly.
- Quantify mismatch if unavoidable.
- Correct mismatch via calibration/offset models.
- Track residual mismatch in monitoring.
## Five-layer architecture
### 1) Ingestion
- Pull station archives per city config.
- Parse NOAA/GHCN daily fields.
- Keep one file per station.
- Persist raw immutable snapshots.
- Validate schema and units early.
- Emit completeness and latency reports.
- Fail fast on missing target station.
### 2) Features
- Merge station panels into daily matrix.
- Build lagged predictors (no leakage).
- Add persistence/trend signals.
- Add sector and ring composites.
- Add wind-conditioned advection composites.
- Add cyclical day-of-year features.
- Add moisture/pressure proxies when safe.
- Add missingness indicators where needed.
- Save reproducible feature dictionary.
### 3) Modeling
- Keep persistence and climatology baselines.
- Keep ridge baseline on same features.
- Train heteroscedastic base NN.
- Optionally train WGA/advanced models.
- Optionally train synthesis/meta model.
- Use CRPS or NLL as main objective.
- Track MAE/RMSE as diagnostics only.
- Output full predictive distribution.
### 4) Calibration and bucketization
- Use held-out calibration period.
- Fit isotonic CDF calibration by default.
- Compare Platt + isotonic variants.
- Check PIT/reliability/coverage diagnostics.
- Map calibrated CDF to contract buckets.
- Clip probabilities to stable bounds.
- Normalize each date to sum 1.
- Validate contract boundary semantics.
### 5) Trading and execution
- Pull order books and market constraints.
- Convert prices to implied probabilities.
- Compute EV net of all costs.
- Trade only if edge clears threshold.
- Use capped fractional Kelly sizing.
- Enforce contract/day/portfolio caps.
- Enforce correlated bucket caps.
- Halt on data or calibration anomalies.
- Log everything for audit.
## Mandatory Kalshi PreSettlement step
- Fetch PreSettlement snapshots every run.
- Fetch historical rows for backtests.
- Capture bid/ask/mid probabilities.
- Capture volume/open interest.
- Capture snapshot timestamps.
- Join PreSettlement to settled outcomes.
- Use joined rows for Contract Brier.
- Use joined rows for market baseline.
- Halt benchmark if join is empty.
- Never backfill missing rows with proxies.
## Command templates
```bash
# PreSettlement
python scripts/fetch_kalshi_presettlement.py
python scripts/fetch_kalshi_presettlement_multi.py --city chi
python scripts/fetch_kalshi_presettlement_multi.py --city phl
python scripts/fetch_kalshi_presettlement_multi.py --resume

# CHI pipeline
python scripts/run_chi_data_collection.py
python scripts/run_chi_preprocessing.py
python scripts/run_chi_benchmark.py
python scripts/run_city_nws_kalshi_template_benchmark.py --city chi
python scripts/run_chi_synthesis_calibration.py
python scripts/run_chi_backtest.py
python scripts/run_chi_promotion_evaluation.py

# PHL pipeline
python scripts/run_phl_data_collection.py
python scripts/run_phl_preprocessing.py
python scripts/run_phl_benchmark.py
python scripts/run_city_nws_kalshi_template_benchmark.py --city phl
python scripts/run_phl_synthesis_calibration.py
python scripts/run_phl_backtest.py
python scripts/run_phl_promotion_evaluation.py
```
## Probability output requirements
- Distribution output is required.
- Accept Gaussian `(mu, sigma)` output.
- Accept mixture output if calibrated.
- Accept quantile output with monotonicity.
- Preserve uncertainty through synthesis.
- Validate sigma floors/ceilings.
- Validate tails vs climatology.
- Validate season/regime spread behavior.
- Never trade point-forecast-only models.
## Bucketization contract logic
- Read edges from city config registry.
- Treat `-999/999` as open tails.
- Compute `P = F(hi) - F(lo)`.
- Handle `below/above/between` correctly.
- Clip to `[1e-4, 1-1e-4]`.
- Renormalize by date after clipping.
- Verify sums are exactly 1 within tolerance.
- Verify one-hot outcome mapping per day.
- Verify date-local contract boundaries.
## Contract Brier policy
- Primary model selection metric.
- Evaluate on contract-row dataset.
- Use actual settled `actual_outcome` only.
- Use actual listed contracts only.
- Compare against market baseline rows.
- Report overall and stratified scores.
- Stratify by season and volatility.
- Stratify by spread/liquidity bins.
- Use confidence intervals/bootstraps.
- Block promotion if not competitive.
## Why Contract Brier is mandatory
- It scores tradable decisions.
- It reflects actual uncertainty region.
- It avoids trivial-grid dilution artifacts.
- It aligns directly with EV targeting.
- It enables apples-to-apples market comparisons.
- It captures tail-risk mistakes better.
- It supports promotion gate realism.
## Prohibited shortcuts
- No synthetic weather labels.
- No proxy settlement proxies.
- No synthetic market probability series.
- No random train/test shuffling.
- No post-cutoff feature leakage.
- No uncalibrated probability trading.
- No fee-free or slippage-free EV.
- No deployment without risk kill switch.
## Template code: split construction
```python
from __future__ import annotations
import pandas as pd

TRAIN_START, TRAIN_END = "2000-01-01", "2021-12-31"
CAL_START, CAL_END = "2022-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", "2025-12-31"

def build_time_splits(X: pd.DataFrame, y: pd.Series):
    X = X.sort_index()
    y = y.sort_index()
    idx = X.index
    m_train = (idx >= TRAIN_START) & (idx <= TRAIN_END)
    m_cal = (idx >= CAL_START) & (idx <= CAL_END)
    m_test = (idx >= TEST_START) & (idx <= TEST_END)
    counts = {"train": int(m_train.sum()), "cal": int(m_cal.sum()), "test": int(m_test.sum())}
    if min(counts.values()) <= 0:
        raise ValueError(f"Missing split rows: {counts}")
    return X.loc[m_train], y.loc[m_train], X.loc[m_cal], y.loc[m_cal], X.loc[m_test], y.loc[m_test]
```
## Template code: PreSettlement + settlement join
```python
from __future__ import annotations
import pandas as pd


def load_presettlement(city: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/kalshi_presettlement_{city}.csv")
    req = ["date", "ticker", "direction", "threshold_low", "threshold_high", "presettlement_prob"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing PreSettlement columns: {miss}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_settled(city: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/real_kalshi_{city}_all.csv")
    req = ["date", "ticker", "direction", "threshold_low", "threshold_high", "actual_outcome"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing settled columns: {miss}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def build_contract_rows(city: str) -> pd.DataFrame:
    pre = load_presettlement(city)
    settled = load_settled(city)
    rows = pre.merge(settled, on=["date", "ticker", "direction", "threshold_low", "threshold_high"], how="inner")
    if rows.empty:
        raise ValueError("No joined contract rows. Stop run.")
    rows["market_prob"] = rows["presettlement_prob"].clip(1e-4, 1 - 1e-4)
    return rows
```
## Template code: contract probability mapping
```python
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def map_contract_probs(rows, mu_by_date, sigma_by_date):
    mu = rows["date"].map(mu_by_date).to_numpy(float)
    sigma = np.maximum(0.5, rows["date"].map(sigma_by_date).to_numpy(float))
    lo = rows["threshold_low"].to_numpy(float)
    hi = rows["threshold_high"].to_numpy(float)
    direction = rows["direction"].astype(str).to_numpy()
    p = np.full(len(rows), np.nan)
    m_below = np.isin(direction, ["below", "less"])
    m_above = direction == "above"
    m_between = direction == "between"
    p[m_below] = norm.cdf(hi[m_below], mu[m_below], sigma[m_below])
    p[m_above] = 1.0 - norm.cdf(lo[m_above], mu[m_above], sigma[m_above])
    p[m_between] = norm.cdf(hi[m_between], mu[m_between], sigma[m_between]) - norm.cdf(lo[m_between], mu[m_between], sigma[m_between])
    return np.clip(p, 1e-4, 1 - 1e-4)
```
## Template code: Contract Brier + EV
```python
from __future__ import annotations
import numpy as np


def contract_brier(pred_prob, outcome):
    p = np.asarray(pred_prob, float)
    y = np.asarray(outcome, float)
    return float(np.mean((p - y) ** 2))


def ev_yes(model_prob, ask_cents, fee_cents=1.0, slip_cents=1.0):
    return model_prob - (ask_cents / 100.0) - ((fee_cents + slip_cents) / 100.0)


def ev_no(model_prob, bid_cents, fee_cents=1.0, slip_cents=1.0):
    no_price = (100.0 - bid_cents) / 100.0
    return (1.0 - model_prob) - no_price - ((fee_cents + slip_cents) / 100.0)
```
## Template code: daily normalization
```python
from __future__ import annotations
import numpy as np


def normalize_daily(prob_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(prob_matrix, float).copy()
    p = np.clip(p, 1e-8, 1.0)
    s = p.sum(axis=1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("Non-positive daily probability sum")
    return p / s
```
## Baseline and complexity policy
- Keep persistence baseline always.
- Keep climatology baseline always.
- Keep ridge/linear baseline always.
- Keep raw MOS baseline if available.
- Add complexity only after OOS wins.
- Wins must persist after calibration.
- Wins must persist on Contract Brier.
- Wins must persist after EV costs.
## Calibration policy
- Fit calibration only on 2022/2023.
- Freeze base model before calibration.
- Compare isotonic vs Platt variants.
- Check reliability slope/intercept.
- Check PIT uniformity and tails.
- Check seasonal interval coverage.
- Persist selected calibrator and metadata.
- Revalidate on 2024/2025 test.
## Evaluation outputs required
- Contract Brier overall.
- Contract Brier by season.
- Contract Brier by regime.
- Contract Brier by liquidity tier.
- Contract Brier vs market baseline.
- CRPS/NLL for distribution quality.
- MAE/RMSE as secondary diagnostics.
- Reliability and PIT summaries.
- Coverage tables (50/80/90).
- Calibration drift rolling metrics.
## Trading rules
- Trade only positive net EV.
- Require minimum edge margin.
- Require minimum liquidity threshold.
- Cap size using fractional Kelly.
- Cap per-contract exposure.
- Cap per-day total exposure.
- Cap adjacent-bucket correlation exposure.
- Skip stale or anomalous books.
- Respect exchange constraints and fees.
- Log all orders and rationale.
## Kill-switch triggers
- Missing critical weather inputs.
- Missing PreSettlement contract rows.
- Missing settlement outcomes.
- Contract schema mismatch.
- Calibration artifact unavailable.
- Abnormal prediction spikes.
- Persistent calibration drift.
- Repeated execution anomalies.
- Risk limits breach.
- Logging/audit pipeline failure.
## Monitoring requirements
- Source latency and completeness.
- Feature null rates and drifts.
- Forecast distribution shifts.
- Calibration reliability drift.
- Contract Brier rolling windows.
- EV forecast vs realized PnL.
- Slippage forecast vs realized.
- Fill rates and reject rates.
- Exposure and drawdown dashboards.
- Incident + halt timeline.
## Backtest realism requirements
- Use decision-time available inputs only.
- Use actual listed contracts daily.
- Use actual PreSettlement snapshots.
- Use historical fee schedules.
- Use conservative slippage assumptions.
- Use depth-aware fill constraints.
- Use partial fill handling.
- Use latency and cancellation logic.
- Tune thresholds without test leakage.
- Report sensitivity and uncertainty.
## Promotion gate (all pass)
- Contract alignment verified.
- Time-safety audit clean.
- Calibration diagnostics acceptable.
- Contract Brier beats baselines.
- Contract Brier competitive vs market.
- Positive EV after full costs.
- Risk and drawdown limits respected.
- Kill switch tested.
- Reproducibility artifacts complete.
- Runbook updated for operations.
## Daily live workflow
- Fetch operational weather sources.
- Validate cutoff-safe completeness.
- Build lagged feature row.
- Run model distribution forecast.
- Apply frozen calibration map.
- Bucketize to contract probabilities.
- Pull market + PreSettlement context.
- Compute EV net of costs.
- Execute risk-filtered trades only.
- Persist artifacts and logs.
## Data dictionary requirements
- Field name, type, units.
- Live source and endpoint.
- Availability SLA timestamp.
- Transform lineage and code owner.
- Train/live usage classification.
- Null-handling rule.
- Drift monitor definition.
- Contract relevance tag.
- Version/deprecation metadata.
## Model card requirements
- Model family and version.
- Train/calibration/test windows.
- Objective + scoring rules.
- Hyperparameters and search space.
- Baselines and relative outcomes.
- Calibration strategy and diagnostics.
- Failure modes and mitigations.
- Deployment constraints and halts.
## Deliverables checklist
- Technical design doc.
- Data dictionary.
- Model card.
- Calibration report.
- Trading specification.
- Backtest report.
- Daily operations runbook.
- Incident response playbook.
- Promotion gate summary.
## Acceptance commands
```bash
python -m pytest tests/test_city_config.py
python scripts/run_chi_preprocessing.py
python scripts/run_phl_preprocessing.py
python scripts/run_city_nws_kalshi_template_benchmark.py --city chi
python scripts/run_city_nws_kalshi_template_benchmark.py --city phl
python scripts/run_chi_backtest.py
python scripts/run_phl_backtest.py
```
## Optional diagnostics
```bash
python scripts/run_cross_city_benchmark_comparison.py
python scripts/audit_cross_city_brier_scale.py
python scripts/run_unified_trading_backtest.py
python scripts/run_real_kalshi_backtest.py
```
## Final reminders
- Real data only, no simulations.
- Real settlement outcomes only.
- Real PreSettlement rows only.
- Contract Brier is primary metric.
- Use day-actual listed contracts.
- Respect split windows exactly.
- Never trade uncalibrated probabilities.
- Never ignore costs or slippage.
- Prefer robustness over complexity.
- Optimize long-run risk-adjusted return.
