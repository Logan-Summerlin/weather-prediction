# KXHIGHCHI Contract Specification — Chicago Daily High Temperature

**Created:** 2026-02-15
**Status:** Draft — verify against live Kalshi contract details before trading

---

## Contract Overview

| Field | Value |
|-------|-------|
| **Ticker** | KXHIGHCHI |
| **Full Name** | Chicago Daily High Temperature |
| **Contract Type** | Daily event contract — temperature bucket |
| **Market** | Kalshi (kalshi.com) |
| **Underlying** | Official daily maximum temperature (Fahrenheit) at the settlement station |

---

## Settlement Station

| Field | Value |
|-------|-------|
| **Station Name** | O'Hare International Airport |
| **GHCN-D ID** | USW00094846 |
| **ICAO / NWS ID** | KORD |
| **Latitude** | 41.9742 N |
| **Longitude** | 87.9073 W |
| **Elevation** | 662 ft (202 m) |
| **WFO** | LOT (Chicago) |
| **Timezone** | Central Time (CT) — UTC-6 standard / UTC-5 daylight |

---

## Measurement Standard

- **Quantity:** Daily maximum temperature (TMAX).
- **Unit:** Degrees Fahrenheit (integer rounding per NWS convention).
- **Source:** National Weather Service (NWS) official daily climate observation, published via the NWS Daily Climate Report (CLI product) for KORD.
- **Observation period:** Midnight-to-midnight local Central Time (CT).

---

## Bucket Definitions

KXHIGHCHI uses 10 degree F-wide buckets spanning the full range of plausible Chicago daily highs, with open-ended tail buckets on both ends. Chicago's wider annual temperature range (colder winters than NYC) warrants an additional low-end bucket extending down to 10 degrees F.

### Standard Bucket Structure (11 Buckets)

| Bucket Index | Label | Lower Bound (inclusive) | Upper Bound (exclusive) | Edge Tuple |
|:---:|--------|:-----------------------:|:----------------------:|:----------:|
| 0 | Below 10 | -infinity | 10 | (-999, 10) |
| 1 | 10-19 | 10 | 20 | (10, 20) |
| 2 | 20-29 | 20 | 30 | (20, 30) |
| 3 | 30-39 | 30 | 40 | (30, 40) |
| 4 | 40-49 | 40 | 50 | (40, 50) |
| 5 | 50-59 | 50 | 60 | (50, 60) |
| 6 | 60-69 | 60 | 70 | (60, 70) |
| 7 | 70-79 | 70 | 80 | (70, 80) |
| 8 | 80-89 | 80 | 90 | (80, 90) |
| 9 | 90-99 | 90 | 100 | (90, 100) |
| 10 | Above 100 | 100 | +infinity | (100, 999) |

### Python Representation

```python
KXHIGHCHI_BUCKET_EDGES = [
    (-999, 10), (10, 20), (20, 30), (30, 40), (40, 50),
    (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 999),
]
KXHIGHCHI_BUCKET_LABELS = [
    "Below 10", "10-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80-89", "90-99", "Above 100",
]
```

### Boundary Convention

- **Lower bound:** Inclusive. A TMAX of exactly 50 degrees F settles in the "50-59" bucket.
- **Upper bound:** Exclusive. A TMAX of exactly 60 degrees F settles in the "60-69" bucket, not "50-59."
- **Tail buckets:** "Below 10" captures any TMAX strictly below 10 degrees F. "Above 100" captures any TMAX of 100 degrees F or higher.
- **Partition property:** Buckets are mutually exclusive and exhaustive — exactly one bucket settles YES for any observed TMAX.

---

## Day Boundary

- **Calendar day definition:** Midnight-to-midnight Central Time (CT).
- **Daylight saving transitions:** The day boundary follows local Central Time, so the observation window shifts between UTC-6 (CST) and UTC-5 (CDT) with seasonal clock changes.
- **Contract date:** Refers to the local calendar date of the observation period.

---

## Settlement Timing and Data Source

- **Primary source:** NWS official Daily Climate Report (CLI product) for KORD, issued by WFO Chicago (LOT).
- **Typical publication:** The official TMAX for a given calendar day is typically published in the CLI product by the following morning (approximately 6-8 AM CT the next day).
- **Settlement timing:** Contracts settle after the NWS publishes the official observation, usually the day after the contract date.
- **Dispute resolution:** In case of delayed or revised observations, Kalshi's settlement rules govern which value is used. Refer to Kalshi's rulebook for edge cases (station outages, corrected observations).

---

## Seasonal Bucket Range Adjustments

Kalshi may adjust the set of actively listed buckets on a seasonal basis:

- **Winter (DJF):** Tail-end summer buckets (90-99, Above 100) may not be listed due to near-zero probability. The "Below 10" bucket sees meaningful activity; sub-zero days are possible.
- **Summer (JJA):** The lowest buckets (Below 10, 10-19, 20-29) may not be listed. The "90-99" and "Above 100" buckets become active.
- **Transition seasons (MAM, SON):** Most or all buckets may be listed depending on forecast uncertainty.

The model pipeline should handle the case where a subset of buckets is listed by redistributing unlisted bucket probability mass or by assigning it to the nearest listed tail bucket.

---

## Chicago Climate Context

Chicago's continental climate, moderated by Lake Michigan, produces a wider annual temperature range than coastal cities:

- **Record high:** 109 degrees F (July 1934)
- **Record low daily max:** Approximately -11 degrees F (January 1985)
- **Winter:** Extended cold spells below 20 degrees F are common; arctic outbreaks can produce single-digit or sub-zero highs.
- **Summer:** Heat waves above 90 degrees F occur regularly; 100+ degree F days are rare but historically observed.
- **Lake effect:** Onshore (easterly) winds off Lake Michigan can suppress daytime highs by 5-15 degrees F in spring and early summer.

This climate profile motivates the additional "Below 10" bucket compared to the KXHIGHNY specification.

---

## Comparison with KXHIGHNY (NYC)

| Attribute | KXHIGHNY (NYC) | KXHIGHCHI (Chicago) |
|-----------|----------------|---------------------|
| Settlement station | Central Park (USW00094728) | O'Hare (USW00094846) |
| NWS ID | KNYC | KORD |
| Timezone | Eastern (ET) | Central (CT) |
| Number of buckets | 10 | 11 |
| Lowest bucket floor | 20 degrees F | 10 degrees F |
| Highest bucket ceiling | 100 degrees F | 100 degrees F |
| Bucket width | 10 degrees F | 10 degrees F |

---

## Disclaimer

> **This document is a working specification for internal model development purposes.** Exact contract terms, bucket boundaries, settlement rules, and dispute resolution procedures are defined by Kalshi and may change between contract series or seasons. Always verify the current contract specification on [kalshi.com](https://kalshi.com) before placing live trades. This document does not constitute financial advice.
