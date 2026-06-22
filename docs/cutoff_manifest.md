# 7am-ET Inference Cutoff Manifest

> Version 1.0.0 — generated from `src/data_sla.py` (do not edit by hand; run `python scripts/build_cutoff_manifest.py`).

Hard cutoff: **7:00 America/New_York** for every city. Example market day `2026-07-01` resolves the cutoff to `2026-07-01T11:00:00+00:00` (UTC). A feature whose freshest record is after its *latest usable* time would leak post-cutoff information; a critical feature that is missing or stale at the cutoff is a kill-switch event.

| Feature | Source | Criticality | Publication / latency | Latest usable (example) | Fallback |
|---|---|---|---|---|---|
| `asos_overnight_obs` | IEM ASOS (hourly METAR archive) | critical | hourly, ~5-20 min after each valid hour (~1.0h) | `2026-07-01T10:00:00+00:00` | Step back one hour at a time to the last published observation; halt if no observation within max_staleness_hours of the cutoff. |
| `asos_prior_day_daily` | IEM ASOS (hourly METAR archive) | critical | hourly, ~5-20 min after each valid hour (~1.0h) | `2026-07-01T04:00:00+00:00` | Use the most recent fully-observed prior day; if the freshest ASOS observation is older than max_staleness_hours, halt (kill switch). |
| `mos_tmax_morning` | NWS MOS (GFS MAV / NAM MET) via IEM | critical | 00Z run issued ~02-03Z; 06Z run issued ~08-09Z.  06Z guidance (~08-09Z = 03-04 ET) is out before the conservative 11:00 UTC cutoff; the 12Z run (issued ~14-15Z) is NOT. (~3.0h) | `2026-07-01T11:00:00+00:00` | Prefer the 06Z run; fall back to the 00Z run if 06Z is missing.  If neither cycle is available, drop MOS features and flag a kill-switch event (model is NWP-blind for the day). |
| `prior_day_settlement` | Kalshi settled-contract feed | recommended | settles the morning after the contract day (~8.0h) | `2026-07-01T04:00:00+00:00` | Use the latest settled day; if the prior day has not settled by the cutoff, skip settlement-dependent monitoring features and warn (non-blocking) but flag for the calibration-drift kill switch. |
| `sounding_00z` | IGRA / RAOB upper-air soundings | recommended | 00Z and 12Z launches; 00Z data available ~02-03Z.  12Z is post-cutoff. (~3.0h) | `2026-07-01T00:00:00+00:00` | Use the 00Z sounding of the market day; if absent, fall back to the prior day's 12Z sounding.  Soundings are recommended, not blocking: degrade gracefully and warn rather than halting. |
