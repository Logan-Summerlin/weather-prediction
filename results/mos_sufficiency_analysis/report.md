# MOS Data Sufficiency Analysis & AVN/ETA Backfill Feasibility Study

Run date: 2026-02-13 07:21:21

```
========================================================================
PART 1: MOS DATA COVERAGE ANALYSIS
========================================================================

1.1 Year-by-Year Coverage
------------------------------------------------------------------------
  Year  Days    GFS   GFS%    NAM   NAM%   Both  Both%  Either Either%
------------------------------------------------------------------------
  2004   366    366 100.0%    311  85.0%    311  85.0%     366  100.0%
  2005   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2006   365    365 100.0%    363  99.5%    363  99.5%     365  100.0%
  2007   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2008   366    366 100.0%    366 100.0%    366 100.0%     366  100.0%
  2009   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2010   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2011   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2012   366    366 100.0%    366 100.0%    366 100.0%     366  100.0%
  2013   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2014   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2015   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2016   366    366 100.0%    366 100.0%    366 100.0%     366  100.0%
  2017   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2018   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2019   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2020   366    366 100.0%    366 100.0%    366 100.0%     366  100.0%
  2021   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2022   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2023   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2024   366    366 100.0%    366 100.0%    366 100.0%     366  100.0%
  2025   365    365 100.0%    365 100.0%    365 100.0%     365  100.0%
  2026    42     42 100.0%     42 100.0%     42 100.0%      42  100.0%

1.2 Gap Identification
------------------------------------------------------------------------
Days missing GFS only: 0
Days missing NAM only: 57
Days missing BOTH:     0

Largest NAM gaps (top 5):
  2004-01-01 to 2004-02-24 (55 days)
  2006-10-10 to 2006-10-11 (2 days)

1.3 Seasonal Coverage (% days with BOTH models)
------------------------------------------------------------------------
  Year      DJF      MAM      JJA      SON
------------------------------------------------------------------------
  2004    39.6%   100.0%   100.0%   100.0%
  2005   100.0%   100.0%   100.0%   100.0%
  2006   100.0%   100.0%   100.0%    97.8%
  2007   100.0%   100.0%   100.0%   100.0%
  2008   100.0%   100.0%   100.0%   100.0%
  2009   100.0%   100.0%   100.0%   100.0%
  2010   100.0%   100.0%   100.0%   100.0%
  2011   100.0%   100.0%   100.0%   100.0%
  2012   100.0%   100.0%   100.0%   100.0%
  2013   100.0%   100.0%   100.0%   100.0%
  2014   100.0%   100.0%   100.0%   100.0%
  2015   100.0%   100.0%   100.0%   100.0%
  2016   100.0%   100.0%   100.0%   100.0%
  2017   100.0%   100.0%   100.0%   100.0%
  2018   100.0%   100.0%   100.0%   100.0%
  2019   100.0%   100.0%   100.0%   100.0%
  2020   100.0%   100.0%   100.0%   100.0%
  2021   100.0%   100.0%   100.0%   100.0%
  2022   100.0%   100.0%   100.0%   100.0%
  2023   100.0%   100.0%   100.0%   100.0%
  2024   100.0%   100.0%   100.0%   100.0%
  2025   100.0%   100.0%   100.0%   100.0%
  2026   100.0%      N/A      N/A      N/A

1.4 Effective 'Both Models Available' Range
------------------------------------------------------------------------
First date with both GFS+NAM: 2004-02-25
Last date with both GFS+NAM:  2026-02-11
Total days with both:         8,021
First year with >=90% both-coverage: 2005

========================================================================
PART 2: MOS QUALITY ASSESSMENT
========================================================================

2.1 MAE by Year (F)
------------------------------------------------------------------------
  Year   GFS MAE   NAM MAE   Ens MAE  GFS Bias  NAM Bias   Spread      N
------------------------------------------------------------------------
  2004      2.72      2.85      2.61     -0.70     -1.00     2.11    366
  2005      2.80      3.06      2.66     -1.04     -1.24     2.44    365
  2006      2.65      2.84      2.51     -0.04     -0.60     2.40    365
  2007      2.89      2.72      2.54     +0.97     -0.07     2.52    365
  2008      2.67      2.63      2.30     +0.70     -0.51     2.57    366
  2009      2.80      2.81      2.48     +0.86     -0.30     2.70    365
  2010      2.61      2.64      2.41     -0.50     -1.14     2.24    365
  2011      2.79      3.17      2.75     -0.55     -1.38     2.16    365
  2012      2.77      2.73      2.57     +1.24     +0.76     2.19    366
  2013      2.61      2.65      2.37     +0.07     -0.43     2.20    365
  2014      2.46      2.78      2.45     -0.59     -1.25     2.08    365
  2015      3.07      3.00      2.83     -1.96     -2.07     2.27    365
  2016      2.93      2.52      2.57     -1.57     -0.71     2.02    366
  2017      2.95      2.88      2.71     +0.05     -0.61     2.18    365
  2018      2.67      2.98      2.55     +0.21     -0.92     2.35    365
  2019      2.75      2.77      2.48     +0.10     -0.90     2.34    365
  2020      2.61      2.90      2.58     -0.48     -1.21     2.07    366
  2021      2.55      2.61      2.37     -0.58     -0.89     1.88    365
  2022      2.67      2.91      2.57     -0.96     -1.32     2.23    365
  2023      2.55      2.67      2.40     -1.04     -1.48     2.06    365
  2024      2.26      2.88      2.31     -0.70     -1.74     2.42    366
  2025      2.34      2.65      2.26     -0.54     -1.05     2.13    365
  2026      2.77      2.93      2.71     +0.28     +0.84     2.28     36

2.2 MAE by Season (F) — All Years Combined
------------------------------------------------------------------------
  Season   GFS MAE   NAM MAE   Ens MAE  GFS Bias  NAM Bias   Spread
------------------------------------------------------------------------
     DJF      2.88      3.05      2.75     -0.18     -0.89     2.32
     MAM      3.34      3.25      3.04     -0.83     -1.59     2.54
     JJA      2.28      2.40      2.10     +0.04     -0.75     2.09
     SON      2.24      2.51      2.15     -0.30     -0.38     2.06

2.3 GFS-NAM Spread (Disagreement) Statistics
------------------------------------------------------------------------
Mean spread:   2.25 F
Median spread: 2.00 F
Std spread:    1.93 F
90th pctile:   5.00 F
95th pctile:   6.00 F
Max spread:    17.00 F

Spread trend by year:
  2004: mean=2.11, median=2.00, p90=5.00
  2005: mean=2.44, median=2.00, p90=5.00
  2006: mean=2.40, median=2.00, p90=5.00
  2007: mean=2.52, median=2.00, p90=5.00
  2008: mean=2.57, median=2.00, p90=5.00
  2009: mean=2.70, median=2.00, p90=5.60
  2010: mean=2.24, median=2.00, p90=5.00
  2011: mean=2.16, median=2.00, p90=5.00
  2012: mean=2.19, median=2.00, p90=4.00
  2013: mean=2.20, median=2.00, p90=5.00
  2014: mean=2.08, median=2.00, p90=4.00
  2015: mean=2.27, median=2.00, p90=5.00
  2016: mean=2.02, median=2.00, p90=4.00
  2017: mean=2.18, median=2.00, p90=5.00
  2018: mean=2.35, median=2.00, p90=5.00
  2019: mean=2.34, median=2.00, p90=5.00
  2020: mean=2.07, median=2.00, p90=4.00
  2021: mean=1.88, median=1.00, p90=4.00
  2022: mean=2.23, median=2.00, p90=4.60
  2023: mean=2.06, median=2.00, p90=4.00
  2024: mean=2.42, median=2.00, p90=5.00
  2025: mean=2.13, median=2.00, p90=5.00
  2026: mean=2.28, median=2.00, p90=5.00

2.4 Systematic Bias Shifts Over Time
------------------------------------------------------------------------
GFS bias trend: slope=-0.0275 F/year, R2=0.055, p=0.2820
  -> No statistically significant trend (p>=0.05)
NAM bias trend: slope=-0.0117 F/year, R2=0.013, p=0.6015
  -> No statistically significant trend (p>=0.05)

Structural break test (pre-2014 vs post-2014):
  GFS bias pre-2014: +0.10 F (n=3653)
  GFS bias post-2014: -0.66 F (n=4419)
  Welch t-test: t=9.56, p=0.0000
  NAM bias pre-2014: -0.59 F (n=3596)
  NAM bias post-2014: -1.16 F (n=4419)
  Welch t-test: t=7.05, p=0.0000

2.5 Rolling 365-Day MAE Trend (sampled annually)
------------------------------------------------------------------------
  Mid-2005: GFS=2.59, NAM=2.82, Ensemble=2.45
  Mid-2006: GFS=2.82, NAM=3.09, Ensemble=2.71
  Mid-2007: GFS=2.59, NAM=2.64, Ensemble=2.43
  Mid-2008: GFS=2.80, NAM=2.66, Ensemble=2.38
  Mid-2009: GFS=2.76, NAM=2.92, Ensemble=2.46
  Mid-2010: GFS=2.69, NAM=2.64, Ensemble=2.36
  Mid-2011: GFS=2.81, NAM=3.02, Ensemble=2.75
  Mid-2012: GFS=2.79, NAM=2.74, Ensemble=2.56
  Mid-2013: GFS=2.55, NAM=2.64, Ensemble=2.39
  Mid-2014: GFS=2.53, NAM=2.93, Ensemble=2.51
  Mid-2015: GFS=2.85, NAM=2.97, Ensemble=2.72
  Mid-2016: GFS=3.16, NAM=2.69, Ensemble=2.73
  Mid-2017: GFS=2.94, NAM=2.59, Ensemble=2.60
  Mid-2018: GFS=2.69, NAM=3.14, Ensemble=2.69
  Mid-2019: GFS=2.59, NAM=2.62, Ensemble=2.29
  Mid-2020: GFS=2.67, NAM=2.92, Ensemble=2.58
  Mid-2021: GFS=2.71, NAM=2.82, Ensemble=2.58
  Mid-2022: GFS=2.59, NAM=2.62, Ensemble=2.39
  Mid-2023: GFS=2.63, NAM=2.91, Ensemble=2.56
  Mid-2024: GFS=2.28, NAM=2.73, Ensemble=2.21
  Mid-2025: GFS=2.34, NAM=2.78, Ensemble=2.32

========================================================================
PART 3: CALIBRATION DATA SUFFICIENCY
========================================================================

3.1 Current Calibration Set Size
------------------------------------------------------------------------
IS predictions file: 731 rows (2023-2024)
  2023 dates: 365
  2024 dates: 366
OOS predictions file: 365 rows (2025)

Estimated calibration contract rows (2023 only):
  365 days x ~5.5 contracts/day = ~2007 rows
Estimated calibration contract rows (2023+2024):
  731 days x ~5.5 contracts/day = ~4020 rows

3.2 What Would Adding 2022 Look Like?
------------------------------------------------------------------------

  Year 2022:
    Total days:  365
    GFS coverage: 365/365 (100.0%)
    NAM coverage: 365/365 (100.0%)
    GFS MAE: 2.67 F
    NAM MAE: 2.91 F
    Ens MAE: 2.57 F

  Year 2023:
    Total days:  365
    GFS coverage: 365/365 (100.0%)
    NAM coverage: 365/365 (100.0%)
    GFS MAE: 2.55 F
    NAM MAE: 2.67 F
    Ens MAE: 2.40 F

  Year 2024:
    Total days:  366
    GFS coverage: 366/366 (100.0%)
    NAM coverage: 366/366 (100.0%)
    GFS MAE: 2.26 F
    NAM MAE: 2.88 F
    Ens MAE: 2.31 F

3.3 Statistical Power: Samples Per Reliability Bin
------------------------------------------------------------------------
For isotonic/Platt calibration, recommended 200-500+ samples per bin.

Simulating Kalshi-style contracts with strikes every 5F:
  Strikes: [np.int64(20), np.int64(25), np.int64(30), np.int64(35), np.int64(40), np.int64(45), np.int64(50), np.int64(55), np.int64(60), np.int64(65), np.int64(70), np.int64(75), np.int64(80), np.int64(85), np.int64(90), np.int64(95), np.int64(100)]

  2023 only (~365 days, ~6,205 total contract-rows):
    Prob bin    0-10%:   3212 samples  [OK]
    Prob bin   10-20%:     95 samples  [LOW]
    Prob bin   20-30%:     59 samples  [LOW]
    Prob bin   30-40%:     34 samples  [INSUFFICIENT]
    Prob bin   40-50%:     43 samples  [INSUFFICIENT]
    Prob bin   50-60%:     40 samples  [INSUFFICIENT]
    Prob bin   60-70%:     64 samples  [LOW]
    Prob bin   70-80%:     71 samples  [LOW]
    Prob bin   80-90%:     88 samples  [LOW]
    Prob bin  90-100%:   2499 samples  [OK]

  2023+2024 (~731 days, ~12,427 total contract-rows):
    Prob bin    0-10%:   6447 samples  [OK]
    Prob bin   10-20%:    173 samples  [LOW]
    Prob bin   20-30%:    108 samples  [LOW]
    Prob bin   30-40%:     82 samples  [LOW]
    Prob bin   40-50%:     94 samples  [LOW]
    Prob bin   50-60%:     92 samples  [LOW]
    Prob bin   60-70%:    132 samples  [LOW]
    Prob bin   70-80%:    137 samples  [LOW]
    Prob bin   80-90%:    149 samples  [LOW]
    Prob bin  90-100%:   5013 samples  [OK]

  2022+2023 (hypothetical) (~730 days):
    Would roughly double 2023-only counts (assuming similar distribution)

3.4 Bootstrap Confidence Intervals on Brier Score
------------------------------------------------------------------------

  2023 only (n=1,825 contract-rows across 5 strikes):
    Brier score: 0.0274
    95% Bootstrap CI: [0.0222, 0.0330]
    CI width: 0.0108

  2023+2024 (n=3,655 contract-rows across 5 strikes):
    Brier score: 0.0278
    95% Bootstrap CI: [0.0242, 0.0316]
    CI width: 0.0074

  Interpretation:
    Doubling calibration data (2023->2023+2024) narrows bootstrap CI by ~sqrt(2).
    Adding 2022 (if model predictions available) would further narrow by ~sqrt(3/2).
    For stable isotonic calibration, aim for 500+ samples in each probability bin.

========================================================================
PART 4: AVN/ETA MOS BACKFILL FEASIBILITY
========================================================================

4.1 Background
------------------------------------------------------------------------
AVN (Aviation Model) was the predecessor to GFS, retired ~2005.
ETA was the predecessor to NAM, retired ~2006.
If IEM has archived AVN/ETA MOS data, it could extend our MOS
history back to ~2000-2002, adding 2-4 years of training data.

4.2 IEM Download Attempts
------------------------------------------------------------------------

Attempting AVN MOS download from IEM...
  AVN 2000-2002: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows
  AVN 2002-2004: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows
  AVN 2004-2006: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows

Attempting ETA MOS download from IEM...
  ETA 2000-2002: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows
  ETA 2002-2004: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows
  ETA 2004-2006: GOT DATA (1 lines, 0.3 KB)
    Columns: ['runtime', 'ftime', 'model', 'n_x', 'tmp', 'dpt', 'cld', 'wdr', 'wsp', 'p06', 'p12', 'q06', 'q12', 't06_1', 't06_2', 't12_1', 't12_2', 'snw', 'cig', 'vis', 'obv', 'poz', 'pos', 'typ', 'sky', 'gst', 't03', 'pzr', 'psn', 'ppl', 'pra', 's06', 'slv', 'i06', 'lcb', 'swh', 'station', 'dur', 'mht', 'twd', 'tws', 'hid', 'sol', 'q24', 'p24', 't24', 'ccg', 'ppo', 'pco', 'cvs', 'lp1', 'lc1', 'cp1', 'cc1', 's12', 'i12', 's24', 'pzp', 'prs', 'txn', 'xnd', 'tsd', 'dsd', 'ssd', 'gsd', 'ifc', 'ifv', 'mvc', 'mvv', 'liv', 'wsd', 'p01', 'pc1', 't06', 't12']
    Rows: 0
    N/X (TMAX) valid: 0 rows

4.3 AVN/ETA Data Quality Assessment
------------------------------------------------------------------------

AVN MOS: No data retrieved from IEM.
  This is expected if IEM did not archive AVN MOS for KNYC.

ETA MOS: No data retrieved from IEM.
  This is expected if IEM did not archive ETA MOS for KNYC.

4.4 AVN/ETA Backfill Feasibility Summary
------------------------------------------------------------------------
NOT FEASIBLE: No usable AVN/ETA MOS data found in IEM archive for KNYC.

Possible reasons:
  - IEM may not have archived AVN/ETA MOS for KNYC station
  - Legacy MOS format may differ from current IEM API expectations
  - AVN/ETA MOS may only be available for airport stations (KJFK, KLGA)

Alternative approaches to extend calibration data:
  1. Use 2022 + 2023 for calibration (requires model retrain on 2004-2021)
  2. Use cross-validated calibration on 2023-2024 IS period
  3. Use temporal block bootstrap from 2023 to generate synthetic calibration data
  4. Consider GFS MOS-only ensemble (available from 2004-01-01)

========================================================================
PART 5: VALIDATION SET SIZE RECOMMENDATION
========================================================================

5.1 Current Setup
------------------------------------------------------------------------
  Train:       2004-2022 (19 years, ~6,935 days)
  Calibrate:   2023 only (1 year, ~365 days)
  Test (IS):   2023-2024 (2 years, ~731 days)
  OOS:         2025 (1 year, ~365 days)

  Calibration contracts: ~365 x 5.5 = ~2,008 rows
  Concern: Is 2,008 contract-rows sufficient for stable isotonic calibration?

5.2 Option A: Extend Calibration to 2022+2023
------------------------------------------------------------------------
  Train:       2004-2021 (18 years, ~6,570 days)
  Calibrate:   2022-2023 (2 years, ~730 days)
  Test (IS):   2022-2024 (3 years, ~1,096 days)
  OOS:         2025 (1 year, ~365 days)

  Calibration contracts: ~730 x 5.5 = ~4,015 rows (+100% vs current)
  Training data loss:    ~365 days (~5.3% reduction)

  MOS quality comparison:
    2022 ensemble MAE: 2.57 F
    2023 ensemble MAE: 2.40 F
    Difference: 0.18 F (similar)

  Pros:
    + Doubles calibration data
    + No need for legacy MOS harmonization
    + Model predictions for 2022 can be generated by re-running inference
  Cons:
    - Requires model retrain on 2004-2021 (excluding 2022)
    - Slight training data reduction
    - 2022 model predictions not yet generated

5.3 Option B: AVN/ETA Backfill (if available)
------------------------------------------------------------------------
  Train:       2002-2021 (20 years, ~7,300 days)
  Calibrate:   2022-2023 (2 years, ~730 days)
  Test (IS):   2022-2024 (3 years, ~1,096 days)
  OOS:         2025 (1 year, ~365 days)

  Pros:
    + More training data (if AVN/ETA quality is acceptable)
    + Doubles calibration data
  Cons:
    - AVN/ETA may not be available (see Part 4)
    - MOS harmonization introduces noise (bias correction imperfect)
    - Early-2000s MOS likely less accurate (model physics improvements since then)
    - Additional engineering complexity for uncertain gain

5.4 Option C: Cross-Validated Calibration on IS Period
------------------------------------------------------------------------
  Train:       2004-2022 (19 years, unchanged)
  Calibrate:   5-fold temporal CV on 2023-2024 (use all IS data)
  Test (IS):   2023-2024 (2 years, ~731 days)
  OOS:         2025 (1 year, ~365 days)

  Effective calibration: ~585 rows per fold (80% of 731 days)
  Contract rows per fold: ~3,218

  Pros:
    + No retraining needed
    + Uses all available IS data for calibration
    + Standard ML approach for limited calibration data
  Cons:
    - Temporal CV has autocorrelation leakage risk
    - Slightly more complex implementation
    - Final calibrator still trained on limited data

5.5 Quantitative Tradeoff Analysis
------------------------------------------------------------------------
Standard error reduction from doubling calibration data:
  SE(2023) / SE(2022+2023) = sqrt(2008/4015) = 0.707
  -> 29.3% reduction in calibration uncertainty

Training data impact:
  2004-2022: ~6,935 days
  2004-2021: ~6,570 days (5.3% reduction)
  Expected MAE impact of losing 1 year of training: minimal
    (diminishing returns: going from 19->18 years is <3% data loss)

MOS quality stability (is 2022 representative of 2023?):
  GFS error distribution 2022 vs 2023 (KS test): stat=0.063, p=0.4639
    -> Distributions are statistically similar (p>0.05)
  NAM error distribution 2022 vs 2023 (KS test): stat=0.066, p=0.4096
    -> Distributions are statistically similar (p>0.05)

5.6 FINAL RECOMMENDATION
========================================================================

RECOMMENDED: Option A — Extend calibration to 2022+2023

Rationale:
  1. Doubles calibration data from ~2,008 to ~4,015 contract-rows
  2. Reduces calibration uncertainty by ~29%
  3. Training data loss is minimal (5.3%, from 19 to 18 years)
  4. No complex harmonization needed (unlike AVN/ETA backfill)
  5. 2022 MOS quality is comparable to 2023 (no structural break)

Implementation steps:
  1. Retrain best model on 2004-2021 data
  2. Generate predictions for 2022-2024
  3. Calibrate on 2022-2023 contract data
  4. Evaluate on 2024 (test) and 2025 (OOS)

Fallback: If retraining is too costly, use Option C (cross-validated
calibration on existing 2023-2024 IS data) as a lighter-weight alternative.
```