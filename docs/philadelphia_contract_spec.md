# KXHIGHPHL Contract Specification — Philadelphia Daily High Temperature

**Created:** 2026-02-15
**Status:** Draft — verify against live Kalshi contract details before trading

---

## Contract Overview

| Field | Value |
|-------|-------|
| **Ticker** | KXHIGHPHL |
| **Full Name** | Philadelphia Daily High Temperature |
| **Contract Type** | Daily event contract — temperature bucket |
| **Market** | Kalshi (kalshi.com) |
| **Underlying** | Official daily maximum temperature (Fahrenheit) at the settlement station |

---

## Settlement Station

| Field | Value |
|-------|-------|
| **Station Name** | Philadelphia International Airport |
| **GHCN-D ID** | USW00013739 |
| **ICAO / NWS ID** | KPHL |
| **Latitude** | 39.8721 N |
| **Longitude** | 75.2411 W |
| **Elevation** | 10 ft (3 m) |
| **WFO** | PHI (Mount Holly, NJ) |
| **Timezone** | Eastern Time (ET) — UTC-5 standard / UTC-4 daylight |

---

## Measurement Standard

- **Quantity:** Daily maximum temperature (TMAX).
- **Unit:** Degrees Fahrenheit (integer rounding per NWS convention).
- **Source:** National Weather Service (NWS) official daily climate observation, published via the NWS Daily Climate Report (CLI product) for KPHL.
- **Observation period:** Midnight-to-midnight local Eastern Time (ET).

---

## Bucket Definitions

KXHIGHPHL uses the same 10 degree F-wide bucket structure as KXHIGHNY. Philadelphia's climate is very similar to NYC — both are mid-Atlantic coastal cities at comparable latitudes with nearly identical annual temperature distributions — so the same bucket boundaries provide appropriate coverage.

### Standard Bucket Structure (10 Buckets)

| Bucket Index | Label | Lower Bound (inclusive) | Upper Bound (exclusive) | Edge Tuple |
|:---:|--------|:-----------------------:|:----------------------:|:----------:|
| 0 | Below 20 | -infinity | 20 | (-999, 20) |
| 1 | 20-29 | 20 | 30 | (20, 30) |
| 2 | 30-39 | 30 | 40 | (30, 40) |
| 3 | 40-49 | 40 | 50 | (40, 50) |
| 4 | 50-59 | 50 | 60 | (50, 60) |
| 5 | 60-69 | 60 | 70 | (60, 70) |
| 6 | 70-79 | 70 | 80 | (70, 80) |
| 7 | 80-89 | 80 | 90 | (80, 90) |
| 8 | 90-99 | 90 | 100 | (90, 100) |
| 9 | Above 100 | 100 | +infinity | (100, 999) |

### Python Representation

```python
KXHIGHPHL_BUCKET_EDGES = [
    (-999, 20), (20, 30), (30, 40), (40, 50), (50, 60),
    (60, 70), (70, 80), (80, 90), (90, 100), (100, 999),
]
KXHIGHPHL_BUCKET_LABELS = [
    "Below 20", "20-29", "30-39", "40-49", "50-59",
    "60-69", "70-79", "80-89", "90-99", "Above 100",
]
```

### Boundary Convention

- **Lower bound:** Inclusive. A TMAX of exactly 50 degrees F settles in the "50-59" bucket.
- **Upper bound:** Exclusive. A TMAX of exactly 60 degrees F settles in the "60-69" bucket, not "50-59."
- **Tail buckets:** "Below 20" captures any TMAX strictly below 20 degrees F. "Above 100" captures any TMAX of 100 degrees F or higher.
- **Partition property:** Buckets are mutually exclusive and exhaustive — exactly one bucket settles YES for any observed TMAX.

---

## Day Boundary

- **Calendar day definition:** Midnight-to-midnight Eastern Time (ET).
- **Daylight saving transitions:** The day boundary follows local Eastern Time, so the observation window shifts between UTC-5 (EST) and UTC-4 (EDT) with seasonal clock changes.
- **Contract date:** Refers to the local calendar date of the observation period.
- **Note:** Philadelphia shares the Eastern timezone with NYC, so the day boundary convention is identical to KXHIGHNY.

---

## Settlement Timing and Data Source

- **Primary source:** NWS official Daily Climate Report (CLI product) for KPHL, issued by WFO Mount Holly (PHI).
- **Typical publication:** The official TMAX for a given calendar day is typically published in the CLI product by the following morning (approximately 6-8 AM ET the next day).
- **Settlement timing:** Contracts settle after the NWS publishes the official observation, usually the day after the contract date.
- **Dispute resolution:** In case of delayed or revised observations, Kalshi's settlement rules govern which value is used. Refer to Kalshi's rulebook for edge cases (station outages, corrected observations).

---

## Seasonal Bucket Range Adjustments

Kalshi may adjust the set of actively listed buckets on a seasonal basis:

- **Winter (DJF):** Tail-end summer buckets (90-99, Above 100) may not be listed due to near-zero probability. The "Below 20" bucket sees moderate activity during cold outbreaks.
- **Summer (JJA):** The lowest buckets (Below 20, 20-29) may not be listed. The "90-99" bucket becomes active; "Above 100" may or may not be listed depending on forecast conditions.
- **Transition seasons (MAM, SON):** Most or all buckets may be listed depending on forecast uncertainty.

The model pipeline should handle the case where a subset of buckets is listed by redistributing unlisted bucket probability mass or by assigning it to the nearest listed tail bucket.

---

## Philadelphia Climate Context

Philadelphia's climate is classified as humid subtropical (Koppen Cfa), very similar to NYC but typically 1-3 degrees F warmer on average due to its slightly lower latitude and greater distance from the open Atlantic:

- **Record high:** 106 degrees F (August 1918)
- **Record low daily max:** Approximately -1 degrees F (February 1934)
- **Winter:** Cold outbreaks below 20 degrees F occur several times per season, but sub-10 degree F highs are rare (unlike Chicago).
- **Summer:** Heat waves above 90 degrees F are common in July and August; 100+ degree F days occur occasionally.
- **Urban heat island:** PHL airport is somewhat insulated from the strongest urban heat island effects of Center City, but suburban development around the airport can contribute a modest warm bias relative to rural stations.

The similarity to NYC's climate profile supports using the same bucket structure as KXHIGHNY (10 buckets, floor at 20 degrees F).

---

## Comparison with KXHIGHNY (NYC) and KXHIGHCHI (Chicago)

| Attribute | KXHIGHNY (NYC) | KXHIGHPHL (Philadelphia) | KXHIGHCHI (Chicago) |
|-----------|----------------|--------------------------|---------------------|
| Settlement station | Central Park (USW00094728) | PHL Airport (USW00013739) | O'Hare (USW00094846) |
| NWS ID | KNYC | KPHL | KORD |
| Timezone | Eastern (ET) | Eastern (ET) | Central (CT) |
| Number of buckets | 10 | 10 | 11 |
| Lowest bucket floor | 20 degrees F | 20 degrees F | 10 degrees F |
| Highest bucket ceiling | 100 degrees F | 100 degrees F | 100 degrees F |
| Bucket width | 10 degrees F | 10 degrees F | 10 degrees F |

---

## Disclaimer

> **This document is a working specification for internal model development purposes.** Exact contract terms, bucket boundaries, settlement rules, and dispute resolution procedures are defined by Kalshi and may change between contract series or seasons. Always verify the current contract specification on [kalshi.com](https://kalshi.com) before placing live trades. This document does not constitute financial advice.
