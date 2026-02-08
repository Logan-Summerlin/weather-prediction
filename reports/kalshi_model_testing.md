# Testing the NYC Daily Max Temperature Model with Kalshi Market Data

This document describes how to validate our NYC daily maximum temperature model against Kalshi’s public market data for the **KXHIGHNY** series (“Highest temperature in NYC today?”). The goal is to align our model outputs with the market’s contract structure and compare model-implied probabilities to market-implied probabilities, as required by the prediction-market project plan.

## 1) Preconditions and contract alignment

Before pulling market data, confirm that our model’s target definition matches the market contract:

1. **Station + definition**: The KXHIGHNY series tracks the highest temperature recorded in **Central Park, New York** on a given day. This aligns with the project plan’s contract-alignment requirement and the typical station target (USW00094728).
2. **Timezone**: Verify the market’s day boundary (local NYC time) to align with our daily aggregation.
3. **Rounding**: Confirm if market outcomes use integer Fahrenheit and any rounding rules.

This step is mandatory to avoid misaligned evaluation (per the “Market Definition and Alignment” phase of the project plan).

## 2) Kalshi API basics (public market data)

Kalshi provides **unauthenticated** market data endpoints on:

```
https://api.elections.kalshi.com/trade-api/v2
```

Even though the subdomain says “elections,” the API serves *all* Kalshi markets, including weather. You can access series, markets, and orderbook data without API keys. For example, the documentation’s quick start uses the **KXHIGHNY** series as the canonical NYC high-temperature market.

## 3) Fetch the KXHIGHNY series (contract metadata)

Use the series endpoint to confirm metadata and ensure we are targeting the correct market:

```python
import requests

url = "https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHNY"
series = requests.get(url).json()["series"]

print(series["title"])
print(series["frequency"])
print(series["category"])
```

This endpoint is critical for confirming the series ticker and any updates to naming or frequency.

## 4) Pull active markets for the series

Kalshi represents each day (or time window) as a market tied to an event. To fetch the active markets:

```python
import requests

markets_url = (
    "https://api.elections.kalshi.com/trade-api/v2/markets"
    "?series_ticker=KXHIGHNY&status=open"
)
markets = requests.get(markets_url).json()["markets"]

for market in markets:
    print(market["ticker"], market["event_ticker"], market["yes_ask"], market["no_ask"])
```

If no markets are open (e.g., outside trading hours or historical review), remove `status=open` or use `status=all` to retrieve historical markets.

## 5) Fetch orderbook data (market-implied probabilities)

For any specific market ticker, request the orderbook to get the most recent prices:

```python
import requests

market_ticker = markets[0]["ticker"]
orderbook_url = (
    f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_ticker}/orderbook"
)
orderbook = requests.get(orderbook_url).json()["orderbook"]

top_yes = orderbook["yes"][0]  # [price, quantity]
top_no = orderbook["no"][0]

print("Top YES bid:", top_yes)
print("Top NO bid:", top_no)
```

Kalshi orderbooks return bids for YES and NO, which are complementary. For probability inference, use the best available YES and NO prices (accounting for bid-ask spreads if you want a midprice).

## 6) Map markets to model outputs

The KXHIGHNY series uses discrete outcome buckets (e.g., “Highest temperature >= 90°F”). To evaluate our model:

1. **Generate the model’s predictive distribution** for the day’s TMAX.
2. **Compute** `P(TMAX ≥ threshold)` for each bucket.
3. **Convert** market prices into implied probabilities (e.g., price/100 for YES).
4. **Compare** model vs market implied probabilities using:
   - Brier score for threshold events.
   - Log score or CRPS for continuous distribution evaluation.
   - EV-style divergence checks for trading decisions.

## 7) Suggested evaluation workflow

**Daily batch evaluation:**
1. Pull the open KXHIGHNY markets.
2. For each market, parse its threshold range or condition (from `market["title"]` or other fields).
3. Compute model probability for that condition.
4. Compare model probability to market-implied probability.
5. Log any deltas that exceed the project plan’s decision threshold.

**Historical backtest:**
1. Pull historical markets via `status=all` and resolve them by date.
2. For each historical market date, use the model’s *historical* forecast (no leakage).
3. Compute accuracy metrics (Brier/log score) and simulated EV.

## 8) Operational checkpoints

- **Alignment check**: Reconfirm Central Park as the station and the time window daily.
- **Drift check**: Compare market-implied probabilities to model calibration drift metrics (PIT/reliability).
- **Tail focus**: Flag extreme heat thresholds (e.g., ≥ 95°F) for targeted evaluation.

This procedure provides a repeatable framework to test our model against the KXHIGHNY market data while staying aligned with the prediction-market success criteria.
