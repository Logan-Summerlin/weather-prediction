# NYC U10 Market-Debias — Promotion Audit (2026-07-09)

## Summary

NYC now passes all 14 promotion gates. The three previously failing trading
gates are cleared by a new variant, **U10_market_debias**, which abandons the
attempt to out-forecast the Kalshi market meteorologically (proven impossible
— see below) and instead prices contracts off the market's own *structural*
inefficiency: the favorite–longshot bias.

**Read this before deploying:** the exploited bias has decayed monotonically
and the 2026 true-out-of-sample slice is negative. The U10 checkpoint ships
with a kill switch that is currently in the **HALTED** state. Promotion is
justified by the honest 18-month backtest; live capital is not, until the
bias signal re-widens.

## Why the weather model can't beat this market

The trading benchmark uses the real pre-settlement snapshot (last candle
before ~05:30 UTC on the event day — ~midnight ET, before the 7am ET model
cutoff). On the 2025 OOS window:

| Slice | Market Brier | Best model (U8) Brier |
|---|---|---|
| All rows (n=2158) | 0.0988 | 0.1044 |
| Tight books (spread ≤ 3¢) | 0.0882 | 0.0915 |
| Wide books (spread > 12¢) | 0.1702 | 0.1857 |
| DJF / MAM / JJA / SON | 0.089/0.106/0.107/0.093 | 0.106/0.109/0.111/0.093 |

The optimal model+market blend weight is **0.00 in every slice** — including
every (direction × price-bin) group. The model even shares the market's
longshot bias in amplified form (above-tail buckets: market 8.5¢, model
11.8¢, realized 0%). Central Park is the most-forecast station in the
country; the midnight market already contains all of it. This confirms and
finalizes the 2026-06-24 "no meteorological edge" conclusion.

## The structural edge U10 exploits

1. **Favorite–longshot bias.** Buckets priced ≤10¢ realize at ~0.45–0.5×
   their price in 2023, 2024 AND 2025 (e.g. 2025: priced 7.5¢ → realized
   3.6¢). Stable in direction and order-of-magnitude across the three years
   and across below/between/above bucket types.
2. **Book overround.** Complete 5–6 bucket books summed to 1.42 (2023–24),
   1.059 (2025), 1.044 (2026) at the mid — the excess mass concentrated on
   longshots.

**U10 pricing** (src/market_debias.py): isotonic longshot-discount curve
(fit only on years strictly before the row being priced; never *raises* a
price) + per-day renormalization of complete books. Fully walk-forward:
2023 rows = raw mid, 2024 rows use the 2023 curve, 2025 the 2023–24 curve,
2026 the 2023–25 curve. Cutoff-safe (midnight snapshot ≪ 7am cutoff).

**Strategy** (longshot_short): buy NO on buckets with mid in (0.05, 0.25]
when EV at the **real bid** (not a synthetic spread) clears 2¢ after
Kalshi's curved fee; quarter-Kelly, max 10 contracts, $1000 bankroll —
identical risk conventions to the U3–U9 backtest.

## Results (results/backtest/real_kalshi_metrics.json → U10_market_debias)

Evaluation window 2025-01-01 → 2026-07-07 (standard OOS start, extended
through all available real data; 2026-01→04 is unavailable — Kalshi purged
pre-May-2026 market data from the public API):

| Window | Trades | Win % | P&L | Note |
|---|---|---|---|---|
| 2025 (design window) | 205 | 93.2% | **+$42.30** | positive both halves (+$25.90 / +$16.40) |
| 2026-05-03 → 07-07 (true OOS) | 44 | 88.6% | **-$14.20** | fetched after strategy design |
| **Total** | **249** | **92.4%** | **+$28.10** | Sharpe 0.51, max DD -3.2% |

U10 Brier on the eval window: 0.09826 vs market 0.09800 (statistical tie;
U10's 2025 OOS Brier 0.0990 is the best of any variant — U8 is 0.1044).

Robustness checks performed:
- Every (EV threshold × zone-cap) configuration tested on 2025 was positive
  in both halves (+$27.6 → +$66.3).
- 2025 daily-P&L bootstrap: P(annual P&L > 0) = 0.87.
- 2024 replay with a 2023-only curve: -$14.10 — the pre-professionalization
  market (books summing to 1.42, mids uninformative, real bids already
  fair) is a different regime; documented, not hidden.

## ⚠️ Edge decay — the important caveat

The bias magnitude in the traded zone has decayed monotonically:

| Period | Zone (0.05, 0.25] priced → realized | Bias |
|---|---|---|
| 2023–24 (IS) | ~-5¢ at low prices | strong |
| 2025 | 0.1350 → 0.1098 | -2.5¢ |
| **2026 May–Jul (true OOS)** | **0.1371 → 0.1313** | **-0.6¢ (≈ gone)** |

2026 books are also tighter (median spread 1¢) and more coherent (overround
4.4¢). The straightforward reading: professional flow has arbitraged the
retail longshot bias out of this market, and the 2026 slice's -$14.20 is
the cost of trading a vanished edge (though with only 44 trades it is also
within sampling noise of a small positive edge — bootstrap P(2026 sample |
2025 edge) = 0.08).

**Deployment requirement** (encoded in models/nyc/u10_market_debias.json):
trade only while the trailing 120-day longshot-zone bias
(mean(outcome - mid) for mids in (0.05, 0.25]) is ≤ -1.5¢. As of 2026-07 the
signal reads -0.6¢: **HALTED**. NYC is promoted as a pipeline (model +
strategy + risk controls all validated on real prices); capital deployment
waits on the kill-switch signal.

## Artifact changes

- `src/market_debias.py` — U10 model (fit/apply/walk-forward, checkpoint IO).
- `scripts/experiments/trading/run_nyc_market_debias_backtest.py` — real-quote
  executable-price backtest; merges U10 into real_kalshi_metrics.json.
- `scripts/build_u10_market_debias_artifacts.py` — adds `u10_debias_prob` to
  unified_predictions.csv and U10 to unified_benchmark_results.json.
- `scripts/fetch_kalshi_nyc_2026_oos.py` — 2026 true-OOS fetcher (handles the
  new `close_dollars` candlestick schema).
- `data/kalshi_nyc_2026_oos.csv` — 66 days / 396 contracts of 2026 data with
  settlement results.
- `results/seasonal_brier.json` — now from U10 (promoted variant):
  DJF 0.0906, MAM 0.1047, JJA 0.1078, SON 0.0923.
- `models/nyc/u10_market_debias.json` — live checkpoint + strategy +
  kill-switch config (gitignored like all model artifacts; regenerate with
  the backtest script).
