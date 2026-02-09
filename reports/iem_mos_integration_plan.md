# IEM MOS Integration Plan: NWS Forecast-Based Market Proxy

**Date:** 2026-02-09
**Status:** Blocked on network access (`mesonet.agron.iastate.edu` returns 403)
**Priority:** HIGH -- this is the single most important remaining improvement

---

## 1. Why This Matters

The current backtest uses a **constructed market proxy** (Ridge regression on lagged TMAX + climatology) to simulate what Kalshi market prices would be. This proxy has a Brier score of ~0.183, while actual Kalshi market settlements have Brier ~0.002-0.025. The proxy is **7-8x less accurate** than real market pricing.

Real Kalshi participants price contracts using NWS forecasts (GFS MOS, NAM MOS, NBM). Until we compare our model against these same forecasts, we cannot honestly assess whether our model has genuine trading edge.

| Proxy | IS Brier | OOS Brier | Quality |
|-------|----------|-----------|---------|
| Our NN model | 0.177 | 0.180 | -- |
| Enhanced GHCN proxy | 0.183 | 0.188 | Weak (no forecast skill) |
| Naive 40/60 proxy | 0.188 | 0.192 | Very weak |
| **Kalshi actual market** | **0.025** | **0.002** | **Gold standard** |
| IEM MOS proxy (expected) | ~0.05-0.10 | ~0.05-0.10 | **Closest to market** |

The honest question: does our 47-station NN beat what free NWS forecasts already provide?

---

## 2. What IEM MOS Provides

The Iowa Environmental Mesonet (IEM) archives NWS Model Output Statistics (MOS) for thousands of stations. For KNYC (Central Park / NYC), it provides:

### Available MOS Models

| Model | Available Since | Update Frequency | Notes |
|-------|----------------|-----------------|-------|
| **GFS MOS** | June 2007 | 4x/day (00Z, 06Z, 12Z, 18Z) | Primary; most reliable archive |
| **NAM MOS** | June 2007 | 4x/day | Higher resolution, shorter range |
| **NBS/NBE** (National Blend) | 2020 | 4x/day | Probabilistic; best calibration |

### Key Field: Max Temperature Forecast (`N/X`)

The `N/X` field is the MOS-derived max temperature forecast in Fahrenheit. The **12Z run** is the most relevant for Kalshi (markets open ~10 AM ET = 15Z, so the 12Z forecast is the last available before market open).

### API Endpoints

**Single runtime query (JSON):**
```
https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2024-01-15%2012:00Z
```

**Bulk CSV download (preferred for historical archive):**
```
https://mesonet.agron.iastate.edu/mos/csv.php?station=KNYC&model=GFS&sts=2007-06-01&ets=2025-12-31
```

**Web interface for manual download:**
```
https://mesonet.agron.iastate.edu/mos/fe.phtml
```

---

## 3. Network Access Required

**Domain:** `mesonet.agron.iastate.edu`
**Protocol:** HTTPS (port 443)
**Authentication:** None (free, public API)

### Current Status

The proxy allowlist includes `mesonet.agron.iastate.edu` in user configuration, but the proxy JWT token embedded in the container has not been refreshed. The token contains a hardcoded `allowed_hosts` list that does not yet include this domain.

**To resolve:** The container session needs to be restarted or the proxy JWT token needs to be regenerated to pick up the updated allowlist. This is a platform-level change, not a code change.

### Verification Command

Once access is granted, verify with:
```bash
curl -s "https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2024-01-15%2012:00Z" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2)[:500])"
```

Expected: JSON object with MOS forecast fields including `n_x` (max temp).

---

## 4. Implementation Plan

### Step 1: Download MOS Archive (~5 min)

Create `scripts/download_iem_mos.py`:

```python
# Download GFS MOS and NAM MOS for KNYC
# Date range: 2007-06-01 to 2025-12-31
# Use bulk CSV endpoint for efficiency
# Save to: data/mos/gfs_mos_knyc.csv, data/mos/nam_mos_knyc.csv

# Bulk URL:
# https://mesonet.agron.iastate.edu/mos/csv.php?station=KNYC&model=GFS&sts=2007-06-01&ets=2025-12-31

# Parse the N/X (max temp) field from the 12Z run for each day
# Output columns: date, gfs_mos_tmax_f, nam_mos_tmax_f, runtime
```

### Step 2: Build MOS-Based Market Proxy (~30 min)

Create `src/mos_market_proxy.py`:

```python
class MOSMarketProxy:
    """Market proxy using actual NWS MOS forecasts.

    This represents what a well-informed Kalshi participant would use:
    the NWS day-ahead max temperature forecast + historical forecast
    error distribution.
    """

    def __init__(self, mos_forecasts_df: pd.DataFrame):
        """
        Parameters
        ----------
        mos_forecasts_df : pd.DataFrame
            Columns: date, gfs_mos_tmax_f, nam_mos_tmax_f
            The day-ahead MOS forecast issued before market open.
        """
        pass

    def fit(self, train_end_date: str):
        """Compute historical forecast error sigma by month.

        For each month, compute std(actual - mos_forecast) using
        data up to train_end_date. This gives the uncertainty
        envelope around the MOS point forecast.
        """
        pass

    def predict_mu_sigma(self, target_date, **kwargs):
        """Return (mu, sigma) where mu = MOS forecast, sigma = monthly error std."""
        pass

    def compute_bracket_prob(self, target_date, threshold_low, threshold_high, direction, **kwargs):
        """P(L <= TMAX < U) using N(mos_forecast, sigma_monthly)."""
        pass
```

**Key design decisions:**
- Use the **average of GFS MOS and NAM MOS** as the point forecast (ensemble of two models)
- Compute forecast error sigma by month from historical MOS errors
- This sigma will be ~3-5°F (much tighter than our current proxy's ~6-8°F), because MOS forecasts are much more skillful than persistence+climatology
- For dates where NAM MOS is missing, fall back to GFS MOS alone

### Step 3: Validate MOS Forecast Quality (~15 min)

Before using as a proxy, verify MOS accuracy:

```python
# Compare MOS forecast MAE vs our model MAE vs climatology
# Expected: MOS MAE ~3-4°F (better than our NN's 4.4°F)
# This would mean the MOS proxy is HARDER to beat than our current proxy

# If MOS MAE < NN MAE, the honest conclusion is:
#   Our model does NOT add value beyond free NWS forecasts
# If MOS MAE > NN MAE:
#   Our model has genuine edge over NWS forecasts
```

### Step 4: Re-Run Full Backtest (~10 min)

Modify `scripts/run_max_train_backtest.py` to add MOS proxy as a fourth comparison:

```python
# Current proxies:
# 1. enhanced_proxy (Ridge + 6 features)
# 2. base_proxy (Ridge + 3 features)
# 3. naive_proxy (40/60 persistence/climatology)

# Add:
# 4. mos_proxy (actual NWS MOS forecasts)
```

The existing backtest pipeline (`add_enhanced_proxy_probabilities()` function) already supports plugging in different proxy objects -- the MOS proxy just needs to implement the same `compute_bracket_prob()` interface.

### Step 5: Generate Honest Comparison Report

The final report should answer:

1. **MOS forecast MAE** vs our NN MAE vs Ridge MAE
2. **MOS-based Brier** vs our model Brier (the critical test)
3. **Strategy profitability** when trading against MOS proxy instead of climatological proxy
4. **Seasonal edge analysis** -- do we beat MOS in specific seasons?
5. **Verdict**: Is there genuine alpha, or was it all proxy weakness?

---

## 5. Expected Outcomes

### Scenario A: Model Beats MOS (Optimistic)
- NN Brier < MOS Brier across most seasons
- Some profitable strategies survive against MOS proxy
- Implies genuine edge from 47-station spatial information
- **Action:** Proceed to paper trading

### Scenario B: Model Matches MOS (Neutral)
- NN Brier ≈ MOS Brier (within ±0.005)
- Few or no profitable strategies against MOS proxy
- Implies our model recapitulates what NWS already provides
- **Action:** Investigate adding NWS forecasts as input features

### Scenario C: MOS Beats Model (Likely)
- MOS Brier < NN Brier
- No profitable strategies against MOS proxy
- NWS post-processed forecasts are better calibrated than our NN
- **Action:** Use MOS as a feature input to our model (hybrid approach)

**Historical context:** NWS MOS day-ahead max temperature forecasts for NYC typically achieve MAE ~2.5-3.5°F. Our NN achieves 4.4-4.5°F MAE. This suggests Scenario C is most likely, but the Brier score comparison (which accounts for probability calibration, not just point accuracy) may tell a different story.

---

## 6. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/download_iem_mos.py` | **CREATE** | Download GFS/NAM MOS archive for KNYC |
| `src/mos_market_proxy.py` | **CREATE** | MOS-based market proxy class |
| `data/mos/gfs_mos_knyc.csv` | **CREATE** | GFS MOS forecast archive |
| `data/mos/nam_mos_knyc.csv` | **CREATE** | NAM MOS forecast archive |
| `scripts/run_max_train_backtest.py` | **MODIFY** | Add MOS proxy to comparison |
| `results/kalshi_max_train_backtest/` | **UPDATE** | Re-generate with MOS results |

### Existing Infrastructure to Reuse

| Module | What to Reuse |
|--------|--------------|
| `src/asos_collection.py` | IEM API client pattern (same base URL, similar auth) |
| `src/enhanced_market_proxy.py` | Proxy interface (`predict_mu_sigma`, `compute_bracket_prob`) |
| `scripts/run_max_train_backtest.py` | Backtest pipeline, Brier computation, plot generation |
| `src/kalshi_client.py` | `compute_brier_scores()`, `build_historical_comparison()` |

---

## 7. Estimated Timeline

Once `mesonet.agron.iastate.edu` is accessible:

| Step | Duration | Dependency |
|------|----------|-----------|
| Download MOS archive | 5 min | Network access |
| Build MOS proxy module | 30 min | Download complete |
| Validate MOS forecast quality | 15 min | Proxy built |
| Re-run backtest with MOS | 10 min | Validation complete |
| Generate final comparison report | 10 min | Backtest complete |
| **Total** | **~70 min** | -- |

---

## 8. How to Verify Network Access

Run these commands in sequence:

```bash
# Test 1: Basic connectivity
curl -sI "https://mesonet.agron.iastate.edu/" | head -3

# Test 2: MOS API endpoint
curl -s "https://mesonet.agron.iastate.edu/api/1/mos.json?station=KNYC&model=GFS&runtime=2024-07-01%2012:00Z" | python3 -m json.tool | head -20

# Test 3: Bulk CSV download
curl -s "https://mesonet.agron.iastate.edu/mos/csv.php?station=KNYC&model=GFS&sts=2024-01-01&ets=2024-01-31" | head -5
```

If all three return HTTP 200 with data, network access is confirmed and implementation can begin immediately.
